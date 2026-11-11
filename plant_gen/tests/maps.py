"""Tests for growth maps (`growth/maps.py` + the call sites in `development.py`).

Locks the contract a later refactor could silently break:

1. **No-op**: no maps, a disabled map, a zero-amount map and a flat-1.0 map on
   every target (including the positional channels, which switch on the
   per-step resolve and the light field) all build the *exact* same plant --
   an exact byte fingerprint of the finished build, not the rounded golden
   digest.
2. **Each target moves the thing it names, in the stated direction**, measured
   against a number: a 1.5x Pitch map lands at 1.5x the launch angle, a 1.5x
   Internode Length map (flowering nodes) with a 1.5x Length map on every level
   (vegetative ones) is exactly 1.5x total length, a Steps map scales branch
   node counts on both the determinate and indeterminate paths, gravitropism
   scales how far laterals wander off their setpoint. The level targets are
   covered in full by `tests/branch_maps.py`.
3. **Determinism**: a map reading `random` and `height` builds identically twice.
4. **Presets**: a profile with maps round-trips through JSON losslessly, and
   builds the same plant after the trip.

All numeric -- the project standard.

    python -m plant_gen.tests.maps
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from ..growth import maps as gm
from ..growth import serialize, species
from ..growth.curves import Curve
from ..growth.development import _tilt_from_vertical, build, grow
from ..growth.environment import Environment
from ..growth.maps import GROWTH_CHANNELS, MAP_TARGETS, ParamMap, map_factor
from ..growth.masks import MaskEntry
from ..growth.morphology import Attachment, MorphologyProfile, generic_perennial

SEED = 0


# --- instruments ----------------------------------------------------------------

def _fingerprint(plant) -> str:
    """Exact (unrounded) hash of everything a map could move: topology, every
    node's placement / direction / radius, every organ's placement, facing and
    scale. Byte-identical builds hash equal; any float drift does not."""
    h = hashlib.sha256()
    for axis in sorted(plant.axes.values(), key=lambda a: a.id):
        h.update(repr((axis.id, axis.order, axis.parent_axis_id, axis.parent_phytomer_index,
                       axis.flowering_from, len(axis.phytomers))).encode())
        h.update(np.asarray(axis.base_position, dtype=np.float64).tobytes())
        for ph in axis.phytomers:
            h.update(np.asarray(ph.tip_position, dtype=np.float64).tobytes())
            h.update(np.asarray(ph.direction, dtype=np.float64).tobytes())
            h.update(np.float64(ph.radius).tobytes())
            h.update(np.float64(ph.final_length).tobytes())
    for organ in sorted(plant.all_organs(), key=lambda o: o.id):
        h.update(repr((organ.id, organ.kind)).encode())
        for v in (organ.position, organ.direction, organ.normal):
            h.update(np.asarray(v, dtype=np.float64).tobytes())
        h.update(np.float64(organ.scale).tobytes())
    return h.hexdigest()


def _with(profile: MorphologyProfile, *maps: ParamMap) -> MorphologyProfile:
    return replace(profile, maps=list(maps))


def _on_level(profile: MorphologyProfile, i: int, *maps: ParamMap) -> MorphologyProfile:
    levels = list(profile.levels)
    levels[i] = replace(levels[i], maps=list(maps))
    return replace(profile, levels=levels)


def _laterals(plant, order: int | None = None):
    """Vegetative laterals (not inflorescence branches), optionally one order."""
    return [a for a in plant.axes.values()
            if a.parent_axis_id >= 0 and a.structural_depth < 1
            and (order is None or a.order == order)]


def _launch_angle(plant, axis) -> float:
    """Degrees between a lateral's launch direction and its parent node's --
    the branch angle as the panel states it (0 = pressed to the stem)."""
    parent = plant.axes[axis.parent_axis_id].phytomers[axis.parent_phytomer_index]
    c = float(np.dot(parent.direction, axis.base_direction))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _upper_lower(plant) -> tuple[int, int]:
    """Laterals whose base sits in the upper / lower half of the plant's height."""
    lo, hi = plant.bounds()
    mid = 0.5 * (float(lo[2]) + float(hi[2]))
    lats = _laterals(plant)
    upper = sum(1 for a in lats if a.base_position[2] >= mid)
    return upper, len(lats) - upper


def _setpoint_deviation(plant) -> float:
    """Mean |tilt - setpoint| over lateral nodes after their first, in degrees --
    how far wander carried them off the angle gravitropism holds them to."""
    dev = [abs(_tilt_from_vertical(ph.direction) - a.setpoint_tilt_deg)
           for a in _laterals(plant) for ph in a.phytomers[1:]]
    return float(np.mean(dev))


# --- 1. no-op ---------------------------------------------------------------------

def test_map_factor_is_one_without_reading_channels():
    def explode(_name):
        raise AssertionError("channels read with no map on the target")
    p = generic_perennial()
    for target in MAP_TARGETS:
        assert map_factor(p, target, explode) == 1.0
    # A map on another target, or a disabled one, must not read channels either.
    p = _with(p, ParamMap("gravitropism", "height", Curve.constant(2.0)),
              ParamMap("internode_length", "height", Curve.constant(2.0), enabled=False))
    assert map_factor(p, "internode_length", explode) == 1.0


def test_map_factor_blend_multiply_and_identity():
    ramp = Curve.linear(0.0, 2.0)  # y = 2x
    p = _with(generic_perennial(), ParamMap("gravitropism", "height", ramp, amount=0.5))
    # 1 + 0.5 * (2*0.75 - 1) = 1.25
    assert abs(map_factor(p, "gravitropism", {"height": 0.75}) - 1.25) < 1e-12
    # Two maps on one target multiply: 1.25 * 3.0.
    p = _with(generic_perennial(), ParamMap("gravitropism", "height", ramp, amount=0.5),
              ParamMap("gravitropism", "factor", Curve.constant(3.0)))
    got = map_factor(p, "gravitropism", {"height": 0.75, "factor": 0.1})
    assert abs(got - 3.75) < 1e-12, got
    # An unreadable channel is identity, never zero.
    assert map_factor(p, "gravitropism", {"factor": 0.1}) == 3.0
    # Clamped at 0: amount 3 on a curve at 0 would give -2.
    p = _with(generic_perennial(), ParamMap("gravitropism", "age", Curve.constant(0.0),
                                            amount=3.0))
    assert map_factor(p, "gravitropism", {"age": 0.5}) == 0.0


def test_neutral_maps_build_identically():
    base = generic_perennial()
    want = _fingerprint(build(base, seed=SEED))
    wild = Curve.linear(0.0, 3.0)
    variants = {
        "disabled": [ParamMap(t, "height", wild, enabled=False) for t in MAP_TARGETS],
        "zero amount": [ParamMap(t, "random", wild, amount=0.0) for t in MAP_TARGETS],
        # Flat 1.0 on every target and every channel -- including height/shade,
        # which turn on the per-step resolve; the extra resolve must not move a bit.
        "flat": [ParamMap(t, c, Curve.constant(1.0))
                 for t in MAP_TARGETS for c in GROWTH_CHANNELS],
    }
    for label, maps in variants.items():
        got = _fingerprint(build(_with(base, *maps), seed=SEED))
        assert got == want, f"{label} maps changed the build"


def test_neutral_shade_map_with_environment_builds_identically():
    # A shade map rebuilds the light field every step even on a plant that does
    # not respond to light; flat, it must still be a no-op.
    base = generic_perennial()
    want = _fingerprint(build(base, seed=SEED, environment=Environment()))
    got = _fingerprint(build(_on_level(base, 1, ParamMap(gm.PITCH, "shade",
                                                         Curve.constant(1.0))),
                             seed=SEED, environment=Environment()))
    assert got == want


# --- 2. each target responds ------------------------------------------------------


def test_pitch_map_scales_launch_angle():
    base = generic_perennial()
    p0 = grow(base, seed=SEED)
    p1 = grow(_on_level(base, 1, ParamMap(gm.PITCH, "factor", Curve.constant(1.5))), seed=SEED)
    a0 = float(np.mean([_launch_angle(p0, a) for a in _laterals(p0)]))
    a1 = float(np.mean([_launch_angle(p1, a) for a in _laterals(p1)]))
    print(f"         mean launch angle {a0:.2f} -> {a1:.2f} deg (1.5x = {1.5 * a0:.2f})")
    assert abs(a1 - 1.5 * a0) < 1.5, (a0, a1)


def test_internode_length_scales_total_length_exactly():
    # The plant's Internode Length map sizes the flowering nodes, each level's
    # Length map its vegetative ones: 1.5x on all of them is 1.5x the lot.
    base = generic_perennial()
    p0 = grow(base, seed=SEED)
    mapped = _with(base, ParamMap("internode_length", "factor", Curve.constant(1.5)))
    for i in range(len(base.levels)):
        mapped = _on_level(mapped, i, ParamMap(gm.LENGTH, "factor", Curve.constant(1.5)))
    p1 = grow(mapped, seed=SEED)
    assert p0.phytomer_count == p1.phytomer_count, "internode map must not move topology"
    l0 = sum(ph.final_length for ph in p0.all_phytomers())
    l1 = sum(ph.final_length for ph in p1.all_phytomers())
    print(f"         total internode length {l0:.4f} -> {l1:.4f} m (ratio {l1 / l0:.9f})")
    assert abs(l1 / l0 - 1.5) < 1e-9


def test_internode_length_map_scales_only_flowering_nodes():
    # Foxglove: a stem of vegetative nodes ending in its raceme.
    base = species.profile("FOXGLOVE")
    t0 = grow(base, seed=SEED).axes[0]
    t1 = grow(_with(base, ParamMap("internode_length", "factor", Curve.constant(1.5))),
              seed=SEED).axes[0]
    veg = t0.flowering_from
    assert veg > 0
    same = all(a.final_length == b.final_length
               for a, b in zip(t0.phytomers[:veg], t1.phytomers[:veg], strict=True))
    scaled = max(abs(b.final_length / a.final_length - 1.5)
                 for a, b in zip(t0.phytomers[veg:], t1.phytomers[veg:], strict=True))
    print(f"         {veg} vegetative stem nodes untouched: {same}; flowering nodes x1.5 "
          f"(worst {scaled:.1e})")
    assert same and scaled < 1e-9


def test_growth_map_cuts_lateral_nodes_terminal():
    base = generic_perennial()
    assert Attachment.TERMINAL in base.inflorescence.attachments
    p0 = grow(base, seed=SEED)
    p1 = grow(_on_level(base, 1, ParamMap(gm.GROWTH, "factor", Curve.constant(0.5))), seed=SEED)
    n0 = np.mean([len(a.phytomers) for a in _laterals(p0, 1)])
    n1 = np.mean([len(a.phytomers) for a in _laterals(p1, 1)])
    print(f"         determinate first-order lateral nodes {n0:.2f} -> {n1:.2f} at 0.5x")
    assert n1 < 0.75 * n0, (n0, n1)
    # The trunk is another level: its node count is untouched.
    assert len(p1.axes[0].phytomers) == len(p0.axes[0].phytomers)


def test_growth_map_cuts_lateral_nodes_axillary():
    # Flowers along the stems, not at the tips: a lateral is all vegetative
    # nodes, so halving Growth % halves it.
    base = generic_perennial()
    base = replace(base, inflorescence=replace(
        base.inflorescence, attachments=frozenset({Attachment.AXILLARY})))
    n0 = np.mean([len(a.phytomers) for a in _laterals(grow(base, seed=SEED), 1)])
    half = _on_level(base, 1, ParamMap(gm.GROWTH, "factor", Curve.constant(0.5)))
    n1 = np.mean([len(a.phytomers) for a in _laterals(grow(half, seed=SEED), 1)])
    print(f"         axillary first-order lateral nodes {n0:.2f} -> {n1:.2f} at 0.5x")
    assert 0.35 * n0 < n1 < 0.65 * n0, (n0, n1)


def _swaying_perennial():
    """The generic perennial with a little circumnutation: a lean for
    gravitropism to correct. (Growth Irregularity, whose wander these tests
    leaned on, is gone -- and a dead
    straight stem gives gravitropism nothing to do.)"""
    return replace(generic_perennial(), circumnutation_deg=4.0)


def test_gravitropism_scales_setpoint_hold():
    base = _swaying_perennial()
    p0, pz, p2 = (grow(_with(base, *m), seed=SEED) for m in (
        (), (ParamMap("gravitropism", "factor", Curve.constant(0.0)),),
        (ParamMap("gravitropism", "factor", Curve.constant(2.5)),)))
    d0, dz, d2 = (_setpoint_deviation(p) for p in (p0, pz, p2))
    tips = [float(np.mean([a.tip_position()[2] for a in _laterals(p)])) for p in (p0, pz, p2)]
    print(f"         lateral |tilt - setpoint| {d0:.3f} deg -> x0 {dz:.3f}, x2.5 {d2:.3f}; "
          f"mean lateral tip z {tips[0]:.4f} / {tips[1]:.4f} / {tips[2]:.4f} m")
    assert dz > 1.5 * d0 and d2 < 0.5 * d0, (d0, dz, d2)
    assert tips[1] != tips[0] and tips[2] != tips[0]


def test_on_branch_gates_a_per_axis_target():
    # gravitropism applies to every axis; `on_branch` with a 0 -> 1 curve turns
    # it off on the trunk only. Per-axis factors are fixed at birth on the axis.
    base = _swaying_perennial()
    p0 = grow(base, seed=SEED)
    p1 = grow(_with(base, ParamMap("gravitropism", "on_branch", Curve.linear(0.0, 1.0))),
              seed=SEED)
    assert p1.axes[0].map_gravitropism == 0.0
    assert all(a.map_gravitropism == 1.0 for a in p1.axes.values() if a.order >= 1)
    trunk0 = [ph.direction.tolist() for ph in p0.axes[0].phytomers]
    trunk1 = [ph.direction.tolist() for ph in p1.axes[0].phytomers]
    assert trunk0 != trunk1, "trunk gravitropism 0 should change its path"


# --- 3. determinism ---------------------------------------------------------------

def test_mapped_build_is_deterministic():
    p = _with(generic_perennial(),
              ParamMap("internode_length", "age", Curve.linear(1.2, 0.8)),
              ParamMap("gravitropism", "node_index", Curve.linear(1.5, 0.5)))
    p = _on_level(p, 1, ParamMap(gm.PITCH, "height", Curve.linear(0.7, 1.3)),
                  ParamMap(gm.GROWTH, "random", Curve.linear(0.6, 1.4)),
                  ParamMap(gm.LENGTH, "age", Curve.linear(1.2, 0.8)))
    a = _fingerprint(build(p, seed=SEED))
    b = _fingerprint(build(p, seed=SEED))
    assert a == b
    assert a != _fingerprint(build(generic_perennial(), seed=SEED)), "maps had no effect"


# --- 4. presets ---------------------------------------------------------------------

def test_maps_round_trip_through_json():
    p = _with(generic_perennial(),
              ParamMap("internode_length", "height", Curve.from_points([(0.0, 0.2), (0.6, 1.0),
                                                                      (1.0, 1.8)]), amount=0.75),
              ParamMap("gravitropism", "on_branch", Curve.constant(0.5), enabled=False))
    levels = list(p.levels)
    levels[1] = replace(levels[1], maps=[ParamMap(gm.YAW, "random", Curve.linear(0.0, 2.0))],
                        masks=[MaskEntry("factor", 0.1, 0.8, invert=True, op="multiply")])
    p = replace(p, levels=levels)
    text = serialize.to_json("mapped", p)
    back = serialize.from_json(text).profile
    assert serialize.profile_to_dict(back) == serialize.profile_to_dict(p)
    assert isinstance(back.maps[0], ParamMap) and back.maps[0].amount == 0.75
    assert back.maps[1].enabled is False
    assert isinstance(back.levels[1].maps[0], ParamMap)
    assert isinstance(back.levels[1].masks[0], MaskEntry) and back.levels[1].masks[0].invert
    assert _fingerprint(build(back, seed=SEED)) == _fingerprint(build(p, seed=SEED))


def test_preset_without_maps_key_loads_empty():
    # A preset written before maps existed has no "maps" key at all.
    d = serialize.profile_to_dict(generic_perennial())
    d.pop("maps")
    assert serialize.profile_from_dict(d).maps == []


# --- plain runner (no pytest needed) -----------------------------------------------

def _run() -> int:
    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 -- test runner reports, not raises
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
