"""Tests for the inflorescence maps (`Inflorescence.maps`, growth/maps.py
`FLOWER_TARGETS`): a flower component's per-flower / per-branch maps.

1. **No-op**: a disabled, a zero-amount and a flat-1.0 map on every target and
   every channel build the exact same plant (pedicels included) on a raceme, a
   panicle, an umbel and a cyme.
2. **Each target scales exactly what it names**: a constant 1.5 map is 1.5x
   that value on every flower / branch / flowering node, nothing else moves.
3. **Per flower**: a `random` map gives every flower its own factor, the same
   on a rebuild.
4. **Presets**: inflorescence maps round-trip through JSON.

    python -m plant_gen.tests.flower_maps
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from ..growth import maps as gm
from ..growth import serialize, species
from ..growth.curves import Curve
from ..growth.development import build, grow
from ..growth.maps import FLOWER_TARGETS, GROWTH_CHANNELS, ParamMap
from ..growth.model import OrganKind
from .inflorescence_golden import _profile_for

SEED = 0


def _fingerprint(plant) -> str:
    """Exact hash of topology, nodes and every organ's placement, facing,
    scale and stalk."""
    h = hashlib.sha256()
    for axis in sorted(plant.axes.values(), key=lambda a: a.id):
        h.update(repr((axis.id, axis.order, axis.parent_axis_id, axis.flowering_from,
                       len(axis.phytomers))).encode())
        for ph in axis.phytomers:
            h.update(np.asarray(ph.tip_position, dtype=np.float64).tobytes())
            h.update(np.asarray(ph.direction, dtype=np.float64).tobytes())
            h.update(np.float64(ph.radius).tobytes())
    for organ in sorted(plant.all_organs(), key=lambda o: o.id):
        h.update(repr((organ.id, organ.kind, organ.pedicel_length, organ.pedicel_bend,
                       organ.pedicel_width, organ.elevation_deg, organ.pitch_deg)).encode())
        for v in (organ.position, organ.direction, organ.normal):
            h.update(np.asarray(v, dtype=np.float64).tobytes())
        h.update(np.float64(organ.scale).tobytes())
    return h.hexdigest()


def _with(profile, *maps: ParamMap):
    return replace(profile, inflorescence=replace(profile.inflorescence, maps=list(maps)))


def _profiles():
    return {"raceme": species.profile("FOXGLOVE"), "panicle": _profile_for("panicle"),
            "umbel": _profile_for("compound_umbel"), "cyme": _profile_for("cyme")}


def _flowers(plant):
    return {o.id: o for o in plant.all_organs() if o.kind == OrganKind.FLOWER}


def _branches(plant):
    return {a.id: a for a in plant.axes.values() if a.structural_depth >= 1}


def _const(target: str, value: float = 1.5, channel: str = "factor") -> ParamMap:
    return ParamMap(target, channel, Curve.constant(value))


# --- 1. no-op ---------------------------------------------------------------------

def test_neutral_maps_build_identically():
    wild = Curve.linear(0.0, 3.0)
    variants = {
        "disabled": [ParamMap(t, "height", wild, enabled=False) for t in FLOWER_TARGETS],
        "zero amount": [ParamMap(t, "random", wild, amount=0.0) for t in FLOWER_TARGETS],
        "flat": [ParamMap(t, c, Curve.constant(1.0))
                 for t in FLOWER_TARGETS for c in GROWTH_CHANNELS],
    }
    for name, base in _profiles().items():
        want = _fingerprint(build(base, seed=SEED))
        for label, maps in variants.items():
            got = _fingerprint(build(_with(base, *maps), seed=SEED))
            assert got == want, f"{name}: {label} maps changed the build"


# --- 2. each target ---------------------------------------------------------------

_ORGAN_FIELDS = {
    gm.PEDICEL_LENGTH: "pedicel_length", gm.PEDICEL_ANGLE: "elevation_deg",
    gm.PEDICEL_BEND: "pedicel_bend", gm.PEDICEL_WIDTH: "pedicel_width",
    gm.FLOWER_PITCH: "pitch_deg",
}


def test_flower_targets_scale_each_flower_exactly():
    base = species.profile("FOXGLOVE")
    f0 = _flowers(grow(base, seed=SEED))
    assert f0
    for target, fld in _ORGAN_FIELDS.items():
        f1 = _flowers(grow(_with(base, _const(target)), seed=SEED))
        assert f1.keys() == f0.keys(), target
        worst = max(abs(getattr(f1[k], fld) - 1.5 * getattr(f0[k], fld)) for k in f0)
        others = [g for g in _ORGAN_FIELDS.values() if g != fld]
        same = all(getattr(f1[k], g) == getattr(f0[k], g) for k in f0 for g in others)
        print(f"         {target}: {len(f0)} flowers x1.5 (worst {worst:.1e}), others same {same}")
        assert worst < 1e-12 and same, target


def test_umbel_maps_scale_stalks_not_aim():
    base = _profile_for("compound_umbel")
    f0 = _flowers(grow(base, seed=SEED))
    for target, fld in _ORGAN_FIELDS.items():
        f1 = _flowers(grow(_with(base, _const(target)), seed=SEED))
        ratio = 1.5 if target in (gm.PEDICEL_LENGTH, gm.PEDICEL_BEND, gm.PEDICEL_WIDTH) else 1.0
        worst = max(abs(getattr(f1[k], fld) - ratio * getattr(f0[k], fld)) for k in f0)
        assert worst < 1e-12, (target, worst)


def test_flower_spacing_scales_flowering_nodes_only():
    base = species.profile("FOXGLOVE")
    t0 = grow(base, seed=SEED).axes[0]
    t1 = grow(_with(base, _const(gm.FLOWER_SPACING)), seed=SEED).axes[0]
    veg = t0.flowering_from
    assert veg > 0 and t1.flowering_from == veg
    same = all(a.final_length == b.final_length
               for a, b in zip(t0.phytomers[:veg], t1.phytomers[:veg], strict=True))
    worst = max(abs(b.final_length / a.final_length - 1.5)
                for a, b in zip(t0.phytomers[veg:], t1.phytomers[veg:], strict=True))
    print(f"         {veg} vegetative nodes untouched: {same}; "
          f"{len(t0.phytomers) - veg} flowering nodes x1.5 (worst {worst:.1e})")
    assert same and worst < 1e-12


def test_branch_targets_scale_each_branch():
    base = _profile_for("panicle")
    b0 = _branches(grow(base, seed=SEED))
    assert b0
    fields = {gm.INFL_BRANCH_LENGTH: "inflorescence_length",
              gm.INFL_BRANCH_BEND: "inflorescence_bend",
              gm.INFL_BRANCH_DROOP: "inflorescence_droop"}
    for target, fld in fields.items():
        b1 = _branches(grow(_with(base, _const(target)), seed=SEED))
        worst = max(abs(getattr(b1[k], fld) - 1.5 * getattr(b0[k], fld)) for k in b0)
        print(f"         {target}: {len(b0)} branches x1.5 (worst {worst:.1e})")
        assert b1.keys() == b0.keys() and worst < 1e-12, target
    # Width keeps its cap: a branch is never thicker than its stem.
    b1 = _branches(grow(_with(base, _const(gm.INFL_BRANCH_WIDTH, 10.0)), seed=SEED))
    assert all(a.inflorescence_width == 1.0 for a in b1.values())
    # Size: more nodes on each branch.
    n0 = sum(len(a.phytomers) for a in b0.values())
    n1 = sum(len(a.phytomers) for a in _branches(
        grow(_with(base, _const(gm.INFL_BRANCH_NODES, 2.0)), seed=SEED)).values())
    print(f"         {gm.INFL_BRANCH_NODES} x2: branch nodes {n0} -> {n1}")
    assert n1 > 1.3 * n0


def test_branch_angle_scales_each_branch_bud():
    base = _profile_for("panicle")

    def angles(plant):
        # the rachis nodes' buds (a vegetative node's below is a level's)
        return {b.id: b.elevation_deg for a in plant.axes.values() if a.flowering_from >= 0
                for ph in a.phytomers[a.flowering_from:] for b in ph.buds}
    a0 = angles(grow(base, seed=SEED))
    a1 = angles(grow(_with(base, _const(gm.INFL_BRANCH_ANGLE)), seed=SEED))
    assert a0 and a0.keys() == a1.keys()
    worst = max(abs(a1[k] - 1.5 * a0[k]) for k in a0)
    print(f"         {len(a0)} rachis buds x1.5 (worst {worst:.1e})")
    assert worst < 1e-12


def test_branch_angle_reaches_cyme_relays():
    base = _profile_for("cyme")
    f0 = _fingerprint(grow(base, seed=SEED))
    f1 = _fingerprint(grow(_with(base, _const(gm.INFL_BRANCH_ANGLE)), seed=SEED))
    assert f0 != f1, "a relay's angle map did nothing"


# --- 3. per flower ----------------------------------------------------------------

def test_random_map_varies_per_flower_and_is_deterministic():
    base = species.profile("FOXGLOVE")
    p = _with(base, ParamMap(gm.PEDICEL_LENGTH, "random", Curve.linear(0.5, 1.5)))
    f0 = _flowers(grow(base, seed=SEED))
    f1 = _flowers(grow(p, seed=SEED))
    ratios = [f1[k].pedicel_length / f0[k].pedicel_length for k in f0
              if f0[k].pedicel_length > 0]
    spread = max(ratios) - min(ratios)
    print(f"         {len(ratios)} flowers, ratio {min(ratios):.3f}..{max(ratios):.3f}")
    assert spread > 0.5 and all(0.5 <= r <= 1.5 for r in ratios)
    assert _fingerprint(build(p, seed=SEED)) == _fingerprint(build(p, seed=SEED))


def test_extra_inflorescence_maps_are_its_own():
    # A further inflorescence's maps are read only by its own flowers: one that
    # grows nowhere builds the same with a map as without.
    base = species.profile("FOXGLOVE")
    other = replace(base.inflorescence, maps=[_const(gm.PEDICEL_LENGTH, 3.0)])
    other.placement = replace(other.placement, levels=(99,))
    p = replace(base, extra_inflorescences=[other])
    assert _fingerprint(build(p, seed=SEED)) == _fingerprint(
        build(replace(base, extra_inflorescences=[replace(other, maps=[])]), seed=SEED))


# --- 4. presets ---------------------------------------------------------------------

def test_inflorescence_maps_round_trip_through_json():
    p = _with(species.profile("FOXGLOVE"),
              ParamMap(gm.PEDICEL_ANGLE, "height", Curve.from_points([(0.0, 0.5), (1.0, 1.5)]),
                       amount=0.75, seed=3),
              ParamMap(gm.FLOWER_SPACING, "age", Curve.constant(1.2), enabled=False))
    back = serialize.from_json(serialize.to_json("mapped", p)).profile
    assert serialize.profile_to_dict(back) == serialize.profile_to_dict(p)
    assert isinstance(back.inflorescence.maps[0], ParamMap)
    assert back.inflorescence.maps[0].seed == 3 and back.inflorescence.maps[1].enabled is False
    assert _fingerprint(build(back, seed=SEED)) == _fingerprint(build(p, seed=SEED))


def test_preset_without_inflorescence_maps_loads_empty():
    d = serialize.profile_to_dict(species.profile("FOXGLOVE"))
    d["inflorescence"].pop("maps")
    assert serialize.profile_from_dict(d).inflorescence.maps == []


# --- plain runner (no pytest needed) -----------------------------------------------

def _run() -> int:
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
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
