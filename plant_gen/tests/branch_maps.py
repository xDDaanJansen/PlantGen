"""Tests for the branching levels' masks and maps.

1. **Every level map target is the parameter it names, scaled.** A constant 0.5
   map on a target builds *exactly* (byte for byte) the plant with that level
   parameter halved -- 0.5 is a power of two, so the product is exact whichever
   order the multiplies run in. Steps, whose halving would also shorten the
   plant's span, is checked by node count instead.
2. **Masks thin branches by their fold**: a factor band keeps only the branches
   in it, Invert the rest, Modulo every Nth node, a random band about its
   fraction; a masked bud stays dormant for good; the stems level's mask thins
   the clump; a full band is exactly no mask.
3. **Custom attributes and classes while growing**: a class reads exactly as
   its own stack, a Distance attribute gates by closeness, Noise is seeded and
   deterministic, a name the plant lacks reads identity, and a reference cycle
   resolves.
4. **The v6 -> v7 preset migration** moves the old plant maps onto the levels.

    python -m plant_gen.tests.branch_maps
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np

from ..growth import maps as gm
from ..growth import serialize
from ..growth import signals as sg
from ..growth.curves import Curve
from ..growth.development import build, grow
from ..growth.maps import ParamMap
from ..growth.masks import MaskEntry
from ..growth.model import BudBlock, BudState, resolve_geometry
from ..growth.morphology import generic_perennial
from .maps import _fingerprint

SEED = 0
HALF = Curve.constant(0.5)


def _level(profile, i: int, **changes):
    levels = list(profile.levels)
    levels[i] = replace(levels[i], **changes)
    return replace(profile, levels=levels)


def _branches(plant, order: int = 1):
    return [a for a in plant.axes.values()
            if a.order == order and a.parent_axis_id >= 0 and a.structural_depth < 1
            and not a.is_fork]


def _branch_buds(plant):
    """Every bud on the stems that could become a level-1 branch."""
    return [b for a in plant.axes.values() if a.order == 0 and a.structural_depth < 1
            for ph in a.phytomers for b in ph.buds]


# --- 1. each target is its parameter, scaled ----------------------------------------

_EQUIVALENT = (
    # (level, field, value, target, other level fields)
    (1, "start_pitch_deg", 48.0, gm.PITCH, {}),
    (1, "start_yaw_deg", 20.0, gm.YAW, {}),
    (1, "start_roll_deg", 40.0, gm.ROLL, {}),
    (1, "start_length_scale", 0.6, gm.START_LENGTH, {}),
    (0, "start_length_scale", 0.02925, gm.START_LENGTH, {}),
    (1, "start_width_scale", 0.8, gm.START_WIDTH, {}),
    (1, "spiral_pitch_deg", 10.0, gm.SPIRAL_PITCH, {}),
    (1, "spiral_roll_deg", 30.0, gm.SPIRAL_ROLL, {"spiral_pitch_deg": 10.0}),
    (0, "fork_probability", 0.4, gm.FORK_PROBABILITY, {"growth_pattern": "dichotomous"}),
    (0, "fork_spread_deg", 30.0, gm.FORK_SPREAD,
     {"growth_pattern": "dichotomous", "fork_probability": 0.5}),
)


def test_each_target_is_its_parameter_scaled():
    for i, field, value, target, extra in _EQUIVALENT:
        base = _level(generic_perennial(), i, **extra, **{field: value})
        mapped = _level(base, i, maps=[ParamMap(target, "factor", HALF)])
        scaled = _level(base, i, **{field: value * 0.5})
        got, want = _fingerprint(build(mapped, seed=SEED)), _fingerprint(build(scaled, seed=SEED))
        assert got == want, f"{target} map != {field} x 0.5"
        assert got != _fingerprint(build(base, seed=SEED)), f"{target} map did nothing"
        print(f"         {target:17s} x0.5 == {field} {value} -> {value * 0.5}")


def test_length_and_width_maps_scale_every_node_of_their_level():
    # Length: this internode; Width: this node's radius -- on the level's own
    # (vegetative) nodes only. Level 1 has no children here, so x0.5 on every
    # node is exactly Start Length / Start Width x 0.5.
    base = generic_perennial()
    for target, field in ((gm.LENGTH, "start_length_scale"), (gm.WIDTH, "start_width_scale")):
        mapped = _level(base, 1, maps=[ParamMap(target, "factor", HALF)])
        scaled = _level(base, 1, **{field: getattr(base.levels[1], field) * 0.5})
        assert _fingerprint(build(mapped, seed=SEED)) == _fingerprint(build(scaled, seed=SEED)), \
            target
    # A gradient: Length by factor 0.5 -> 1.5 scales each vegetative internode
    # of the stems by exactly 0.5 + its factor.
    ramp = _level(base, 0, maps=[ParamMap(gm.LENGTH, "factor", Curve.linear(0.5, 1.5))])
    t0, t1 = grow(base, seed=SEED).axes[0], grow(ramp, seed=SEED).axes[0]
    veg = t0.flowering_from if t0.flowering_from >= 0 else len(t0.phytomers)
    worst = max(abs(b.final_length / a.final_length - (0.5 + b.factor))
                for a, b in zip(t0.phytomers[:veg], t1.phytomers[:veg], strict=True))
    flowering_same = all(a.final_length == b.final_length for a, b in
                         zip(t0.phytomers[veg:], t1.phytomers[veg:], strict=True))
    print(f"         stems Length by factor: worst error vs 0.5 + factor {worst:.2e}, "
          f"flowering nodes untouched: {flowering_same}")
    assert worst < 1e-9 and flowering_same


def test_setpoint_map_aims_every_branch_by_the_world():
    # Gravitropic Setpoint: with full-strength gravitropism and no wander, every
    # node after a branch's first heads exactly its setpoint x the map from
    # vertical -- every branch, whichever way it faces (a frame-relative bend,
    # Spiral Pitch, turns each member of a whorl a different way).
    base = replace(_level(generic_perennial(), 1, whorl=4), gravitropism=1.0,
                   circumnutation_deg=0.0)
    mapped = _level(base, 1, maps=[ParamMap(gm.SETPOINT, "factor", HALF)])
    plant = grow(mapped, seed=SEED)
    worst, checked = 0.0, 0
    for a in _branches(plant):
        veg = a.flowering_from if a.flowering_from >= 0 else len(a.phytomers)
        for ph in a.phytomers[1:veg]:
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(ph.direction[2])))))
            worst = max(worst, abs(tilt - 0.5 * a.setpoint_tilt_deg))
            checked += 1
    print(f"         {checked} branch nodes: worst tilt error vs setpoint x 0.5 {worst:.2e} deg")
    assert checked > 0 and worst < 1e-6
    assert _fingerprint(build(mapped, seed=SEED)) != _fingerprint(build(base, seed=SEED))


def test_growth_map_cuts_branch_node_counts():
    base = generic_perennial()
    p = grow(_level(base, 1, maps=[ParamMap(gm.GROWTH, "factor", HALF)]), seed=SEED)
    want = math.ceil(base.levels[1].steps * 0.5)
    counts = {a.flowering_from if a.flowering_from >= 0 else len(a.phytomers)
              for a in _branches(p)}
    print(f"         level-1 vegetative nodes {base.levels[1].steps} -> {sorted(counts)} "
          f"(want {want})")
    assert counts == {want}, counts
    # The stems are another level and keep theirs.
    assert len(p.axes[0].phytomers) == len(grow(base, seed=SEED).axes[0].phytomers)


def test_growth_cuts_branches_without_renormalising():
    # Growth % builds the lower part of the full branch: every node it builds
    # sits where the uncut branch's does (same factor, same length), the last
    # one grown only the part reached -- and a terminal branch still flowers
    # at its cut tip.
    base = generic_perennial()
    steps = base.levels[1].steps
    g = 0.55
    reach = g * steps
    built = math.ceil(reach)
    full = {a.id: a for a in _branches(grow(base, seed=SEED))}
    cut = _branches(grow(_level(base, 1, growth=g), seed=SEED))
    assert cut and all(a.id in full for a in cut)
    worst_factor = worst_length = worst_last = 0.0
    for a in cut:
        b = full[a.id]
        veg = a.flowering_from if a.flowering_from >= 0 else len(a.phytomers)
        assert veg == built, (veg, built)
        if a.flowering_from >= 0:
            assert len(a.phytomers) > a.flowering_from     # flowers at the cut tip
        for i in range(built):
            worst_factor = max(worst_factor, abs(a.phytomers[i].factor - b.phytomers[i].factor))
        for i in range(built - 1):
            worst_length = max(worst_length, abs(a.phytomers[i].final_length
                                                 - b.phytomers[i].final_length))
        frac = reach - (built - 1)
        last = a.phytomers[built - 1].final_length / b.phytomers[built - 1].final_length
        worst_last = max(worst_last, abs(last - frac))
    print(f"         Growth {g:.0%} of {steps}: {built} nodes, "
          f"the last at {reach - built + 1:.2f}; worst factor diff {worst_factor:.1e}, "
          f"length diff {worst_length:.1e}, "
          f"last-node ratio error {worst_last:.1e} over {len(cut)} branches")
    assert worst_factor < 1e-12 and worst_length < 1e-12 and worst_last < 1e-9


def test_emergence_delay_is_timing_only():
    # Emergence Delay only holds parts back: growth runs exactly as without it,
    # so the full-grown plant is the same plant, byte for byte -- every id,
    # node, organ and its variation draw.
    base = generic_perennial()
    delayed = _level(base, 1, emergence_delay=4)
    assert delayed.total_steps == base.total_steps and delayed.emergence_span == 4
    assert _fingerprint(build(base, seed=SEED)) == _fingerprint(build(delayed, seed=SEED))
    for day_path in (False, True):
        kw = {"day": 400.0} if day_path else {}
        assert (_fingerprint(build(base, seed=SEED, **kw))
                == _fingerprint(build(delayed, seed=SEED, **kw)))


def test_emergence_delay_reveals_the_grown_plant():
    # Part-way, the plant shown is a subset of the full one, and a delayed
    # branch at display step t is exactly that branch at step t - delay.
    base = generic_perennial()
    d = 4
    delayed = _level(base, 1, emergence_delay=d)
    total, span = base.total_steps, delayed.emergence_span
    full = grow(base, seed=SEED)
    worst = 0
    for show in (6, 12, 18, total, total + 2):
        p = grow(delayed, seed=SEED, age=show / (total + span))
        undelayed_at = {k: grow(base, seed=SEED, age=min(k, total) / total)
                        for k in {show, show - d} if k >= 0}
        for a in p.axes.values():
            f = full.axes[a.id]                              # a subset of the full plant
            lengths = [ph.final_length for ph in a.phytomers]
            assert lengths == [ph.final_length for ph in f.phytomers[:len(lengths)]]
            want = undelayed_at[show - (d if a.emergence_shift else 0)]
            n_want = len(want.axes[a.id].phytomers) if a.id in want.axes else 0
            worst = max(worst, abs(len(a.phytomers) - n_want))
    shown = [sum(len(a.phytomers) for a in _branches(grow(delayed, seed=SEED,
                                                           age=k / (total + span))))
             for k in (6, 12, 18, total, total + span)]
    plain = [sum(len(a.phytomers) for a in _branches(grow(base, seed=SEED,
                                                         age=min(k, total) / total)))
             for k in (6, 12, 18, total, total + span)]
    print(f"         delay {d}: shown over {total} + {span} steps; branch nodes at display "
          f"steps 6/12/18/{total}/{total + span}: {shown} (undelayed {plain}); "
          f"worst node-count mismatch vs undelayed at t - delay: {worst}")
    assert worst == 0 and shown[-1] == plain[-1] and shown[1] < plain[1]


def test_emergence_delay_map_staggers_stems():
    base = generic_perennial()
    stems = base.levels[0]
    ramp = ParamMap(gm.EMERGENCE_DELAY, "node_index", Curve.linear(0.0, 1.0))
    p = _level(base, 0, whorl=4, emergence_delay=6, maps=[*stems.maps, ramp])
    assert p.emergence_span == 6
    plant = grow(p, seed=SEED)
    shifts = sorted(plant.axes[i].emergence_shift for i in plant.basal_axis_ids)
    print(f"         stems' emergence shifts under a 0 -> 6 ramp: {shifts}")
    assert shifts == [0, 2, 4, 6], shifts
    young = grow(p, seed=SEED, age=8 / (p.total_steps + 6))
    nodes = sorted(len(young.axes[i].phytomers) for i in young.basal_axis_ids)
    print(f"         at display step 8 their nodes: {nodes}")
    assert len(young.basal_axis_ids) == 4 and nodes[0] < nodes[-1]


def test_growth_at_full_is_byte_identical():
    base = generic_perennial()
    want = _fingerprint(build(base, seed=SEED))
    assert _fingerprint(build(_level(base, 1, growth=1.0), seed=SEED)) == want
    assert _fingerprint(build(_level(base, 1, maps=[ParamMap(gm.GROWTH, "factor",
                                                             Curve.constant(1.0))]),
                              seed=SEED)) == want


def test_neutral_level_maps_and_masks_build_identically():
    base = generic_perennial()
    want = _fingerprint(build(base, seed=SEED))
    flat = [ParamMap(t, c, Curve.constant(1.0)) for t in gm.LEVEL_TARGETS
            for c in ("factor", "random", "height")]
    full = [MaskEntry("factor", 0.0, 1.0), MaskEntry("height", 0.0, 1.0)]
    for label, changes in (("flat maps", {"maps": flat}), ("full-band masks", {"masks": full}),
                           ("zero amount", {"maps": [ParamMap(t, "random", HALF, amount=0.0)
                                                     for t in gm.LEVEL_TARGETS]})):
        p = base
        for i in range(len(base.levels)):
            p = _level(p, i, **changes)
        assert _fingerprint(build(p, seed=SEED)) == want, label


# --- 2. masks -----------------------------------------------------------------------

def test_factor_band_keeps_only_its_branches():
    base = _level(generic_perennial(), 1, whorl=3)
    p = grow(_level(base, 1, masks=[MaskEntry("factor", 0.0, 0.5)]), seed=SEED)
    kept = _branches(p)
    along = [p.axes[a.parent_axis_id].phytomers[a.parent_phytomer_index].factor for a in kept]
    masked = [b for b in _branch_buds(p) if b.masked]
    print(f"         {len(kept)} branches kept, all at factor <= 0.5 "
          f"(max {max(along):.3f}); {len(masked)} buds masked")
    assert kept and max(along) <= 0.5
    assert masked and all(b.state == BudState.DORMANT and b.blocked_by == BudBlock.MASK
                          for b in masked)
    # Invert keeps the other side.
    q = grow(_level(base, 1, masks=[MaskEntry("factor", 0.0, 0.5, invert=True)]), seed=SEED)
    along = [q.axes[a.parent_axis_id].phytomers[a.parent_phytomer_index].factor
             for a in _branches(q)]
    assert along and min(along) > 0.5


def test_height_band_reads_the_whole_plant():
    # Height is read against the plant's whole (mature-placed) extent, not the
    # plant so far: a branch whose stem node sits in the upper half is kept,
    # every other one masked -- exactly, since the stems do not move when their
    # branches go.
    base = _level(generic_perennial(), 1, whorl=2)
    ref = grow(base, seed=SEED)
    resolve_geometry(ref)
    lo, hi = ref.bounds()
    stem = ref.axes[0]

    def h(i):
        return (float(stem.phytomers[i].tip_position[2]) - lo[2]) / (hi[2] - lo[2])

    want = sorted({i for i in range(len(stem.phytomers)) if h(i) >= 0.5}
                  & {a.parent_phytomer_index for a in _branches(ref)})
    p = grow(_level(base, 1, masks=[MaskEntry("height", 0.5, 1.0)]), seed=SEED)
    got = sorted({a.parent_phytomer_index for a in _branches(p)})
    print(f"         upper-half stem nodes {want}; branches kept on {got}")
    assert got and got == want
    buds = [b for b in _branch_buds(p) if b.mask_windows]
    assert buds and all(b.mask_windows[0] == (0.0 if b.masked else 1.0) for b in buds)


def test_modulo_keeps_every_other_node():
    p = grow(_level(generic_perennial(), 1, masks=[MaskEntry("node_index", modulo=2)]),
             seed=SEED)
    nodes = sorted({a.parent_phytomer_index for a in _branches(p)})
    print(f"         branches on stem nodes {nodes}")
    assert nodes and all(n % 2 == 0 for n in nodes)


def test_random_band_thins_by_its_fraction():
    base = _level(generic_perennial(), 1, whorl=4)
    p = grow(_level(base, 1, masks=[MaskEntry("random", 0.0, 0.3)]), seed=SEED)
    buds = _branch_buds(p)
    kept = len(_branches(p))
    total = len([b for b in buds if b.masked]) + kept
    frac = kept / total
    print(f"         random band 0..0.3 kept {kept}/{total} = {frac:.2f}")
    assert 0.12 < frac < 0.5, frac
    assert _fingerprint(build(_level(base, 1, masks=[MaskEntry("random", 0.0, 0.3)]),
                              seed=SEED)) == \
        _fingerprint(build(_level(base, 1, masks=[MaskEntry("random", 0.0, 0.3)]), seed=SEED))


def test_stems_mask_thins_the_clump():
    base = _level(generic_perennial(), 0, whorl=8)
    n0 = len(grow(base, seed=SEED).basal_axis_ids)
    p = grow(_level(base, 0, masks=[MaskEntry("random", 0.0, 0.5)]), seed=SEED)
    print(f"         stems {n0} -> {len(p.basal_axis_ids)} under a 0..0.5 random mask")
    assert n0 == 8 and 0 < len(p.basal_axis_ids) < 8


# --- 3. custom attributes and classes ------------------------------------------------

def test_class_reads_as_its_own_stack():
    base = generic_perennial()
    direct = _level(base, 1, masks=[MaskEntry("factor", 0.0, 0.5)])
    via = _level(base, 1, masks=[MaskEntry("class:low")])
    signals = sg.Signals(classes=[sg.ClassDef("low", [MaskEntry("factor", 0.0, 0.5)])])
    assert _fingerprint(build(via, seed=SEED, signals=signals)) == \
        _fingerprint(build(direct, seed=SEED))


def test_missing_attribute_reads_identity():
    base = generic_perennial()
    p = _level(base, 1, masks=[MaskEntry("class:nothing"), MaskEntry("attr:nothing", 0.5, 1.0)],
               maps=[ParamMap(gm.PITCH, "attr:nothing", HALF)])
    assert _fingerprint(build(p, seed=SEED)) == _fingerprint(build(base, seed=SEED))


def test_distance_attribute_gates_by_closeness():
    base = _level(generic_perennial(), 1, masks=[MaskEntry("attr:near", 0.5, 1.0)])
    far = sg.Signals(attributes=[sg.CustomAttribute("near", "DISTANCE", point=(0.0, 0.0, 50.0),
                                                    radius=0.05)])
    assert not _branches(grow(base, seed=SEED, signals=far))
    # Centred on the stems with a radius past the plant: every branch is close.
    near = sg.Signals(attributes=[sg.CustomAttribute("near", "DISTANCE", point=(0.0, 0.0, 0.0),
                                                     radius=50.0)])
    assert len(_branches(grow(base, seed=SEED, signals=near))) == \
        len(_branches(grow(generic_perennial(), seed=SEED)))


def test_noise_attribute_is_seeded_and_deterministic():
    pts = np.random.default_rng(1).random((500, 3)) * 4.0
    n = sg.noise3(pts)
    assert np.all(np.abs(n) <= 1.1) and n.std() > 0.1
    assert np.allclose(sg.noise3(np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 7.0]])), 0.0)
    assert not np.allclose(sg.noise01(pts, 3.0, 0), sg.noise01(pts, 3.0, 1))
    base = _level(generic_perennial(), 1, whorl=3,
                  masks=[MaskEntry("attr:patch", 0.5, 1.0)])
    signals = sg.Signals(attributes=[sg.CustomAttribute("patch", "NOISE", noise_scale=6.0)])
    a = build(base, seed=SEED, signals=signals)
    assert _fingerprint(a) == _fingerprint(build(base, seed=SEED, signals=signals))
    kept, full = len(_branches(a)), len(_branches(grow(_level(base, 1, masks=[]), seed=SEED)))
    print(f"         noise mask kept {kept}/{full} branches")
    assert 0 < kept < full


def test_restrict_keeps_only_what_growth_reads():
    signals = sg.Signals(
        attributes=[sg.CustomAttribute("n", "NOISE"),
                    sg.CustomAttribute("mix", "COMBINE", channel_a="attr:n", channel_b="height"),
                    sg.CustomAttribute("unused", "NOISE")],
        classes=[sg.ClassDef("c", [MaskEntry("attr:mix")]), sg.ClassDef("other")])
    got = sg.restrict(signals, ["class:c", "factor"])
    assert [a.name for a in got.attributes] == ["n", "mix"]
    assert [c.name for c in got.classes] == ["c"]
    assert sg.restrict(signals, ["factor", "attr:missing"]) is None


def test_reference_cycle_resolves():
    signals = sg.Signals(attributes=[
        sg.CustomAttribute("a", "COMBINE", channel_a="attr:b", channel_b="factor"),
        sg.CustomAttribute("b", "COMBINE", channel_a="attr:a", channel_b="factor")])
    ev = sg.Evaluator(signals, np.arange(3), {"factor": np.array([0.2, 0.5, 1.0])},
                      np.zeros((3, 3)))
    values = ev.all()
    assert set(values) == {"attr:a", "attr:b"}
    assert all(np.all(np.isfinite(v)) for v in values.values())


# --- 4. migration -----------------------------------------------------------------------

def test_v6_plant_maps_move_onto_levels():
    profile = generic_perennial()
    d = {"schema_version": 6, "kind": "species", "name": "old",
         "profile": serialize.profile_to_dict(profile)}
    curve = {"samples": [1.0, 2.0]}
    d["profile"]["maps"] = [
        {"target": t, "channel": "factor", "curve": copy.deepcopy(curve), "amount": 1.0,
         "enabled": True}
        for t in ("branch_angle", "branch_length", "fork_chance", "internode_length",
                  "gravitropism")]
    back = serialize.from_json(serialize.json.dumps(d)).profile
    assert [m.target for m in back.maps] == ["internode_length", "gravitropism"]
    stems, branches = back.levels
    assert [m.target for m in stems.maps] == [gm.FORK_PROBABILITY, gm.LENGTH]
    assert [m.target for m in branches.maps] == [gm.PITCH, gm.GROWTH, gm.FORK_PROBABILITY,
                                                  gm.LENGTH]


def test_v9_steps_map_becomes_growth():
    profile = _level(generic_perennial(), 1, maps=[ParamMap(gm.PITCH, "factor", HALF)])
    d = {"schema_version": 9, "kind": "species", "name": "old",
         "profile": serialize.profile_to_dict(profile)}
    d["profile"]["levels"][1]["maps"].append(dict(d["profile"]["levels"][1]["maps"][0],
                                                  target="steps"))
    back = serialize.from_json(serialize.json.dumps(d)).profile
    assert [m.target for m in back.levels[1].maps] == [gm.PITCH, gm.GROWTH]


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
