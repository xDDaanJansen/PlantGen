"""Tests for the per-level growth patterns, headless.

Each pattern against its closed form, on a plant with the wander,
gravitropism and droop off, so the angles are exact:

1. **Monopodial** is the default: stating it changes nothing; its tip node
   is bare.
2. **Sympodial**: a chain of units, each Unit Steps nodes (the last one what
   is left), one relay each, the units' nodes summing to the level's Steps;
   each relay leaves its unit's last node Spread / 2 off its heading.
3. **A unit's tip** flowers exactly when the inflorescence is Terminal, else
   it dies back.
4. **Dichasial / Pleiochasial**: 2 / Relays relays per unit, so generation g
   has n ** g units.
5. **Aligned Relay**: the relay continues exactly in line; the spent apex's
   first flowering node leans Spread / 2 aside.
6. **Unequal Fork**: the leader deviates Spread / 2 x (1 - Balance) and keeps
   the budget; each weak fork deviates Spread / 2 x (1 + Balance) with budget
   and base width x (1 - Balance / 2).
7. **Node-borne flowers**: where the attachment says, and only there -- a
   solitary one is the bud itself (no stalk), sitting at the node, down the
   internode or opposite the leaf; a stalk opens its inflorescence; every
   unit of a chain bears them.

    python -m plant_gen.tests.growth_patterns
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..growth import development as dev
from ..growth.model import OrganKind
from ..growth.morphology import (
    Attachment,
    BranchLevel,
    GrowthPattern,
    Inflorescence,
    InflorescenceKind,
    MorphologyProfile,
    OrientationMode,
    Placement,
)


def _profile(*levels: BranchLevel, **kw) -> MorphologyProfile:
    base = dict(gravitropism=0.0, circumnutation_deg=0.0,
                droop=0.0)
    base.update(kw)
    return MorphologyProfile(levels=list(levels), **base)


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _units(plant, level=0):
    return sorted((a for a in plant.axes.values() if a.level == level),
                  key=lambda a: a.id)


def _vegetative(axis) -> int:
    return axis.flowering_from if axis.flowering_from >= 0 else len(axis.phytomers)


def _generation(plant, axis) -> int:
    g = 0
    while axis.is_fork:
        axis = plant.axes[axis.parent_axis_id]
        g += 1
    return g


def _node_borne(kind: str, *modes: str) -> Inflorescence:
    """Flowers along every stem (the whole window -- the panel's), not at tips."""
    return Inflorescence(kind=kind, flower_count=5, attachments=frozenset(modes),
                         internode_position=0.5,
                         placement=Placement(start=0.0, end=1.0, on_trunk=True))


STEM = BranchLevel(start_pitch_deg=0.0, steps=10, start_length_scale=0.045,
                   start_width_scale=0.005, fork_spread_deg=40.0)
SYMPODIAL = replace(STEM, growth_pattern=GrowthPattern.SYMPODIAL, unit_steps=3)
NO_FLOWERS = Inflorescence(kind=InflorescenceKind.NONE)


def test_monopodial_is_the_default():
    a = dev.grow(_profile(STEM), seed=3)
    b = dev.grow(_profile(replace(STEM, growth_pattern=GrowthPattern.MONOPODIAL)), seed=3)
    assert [len(x.phytomers) for x in a.axes.values()] == \
        [len(x.phytomers) for x in b.axes.values()]
    assert len(_units(a)) == 1


def test_tip_is_bare():
    # An axis's last vegetative node bears no side branch, whatever carries
    # the tip on -- the leader itself, a relay, or forks -- while every other
    # node does.
    branch = BranchLevel(start_pitch_deg=50.0, steps=3)
    for pattern in (GrowthPattern.MONOPODIAL, GrowthPattern.DICHOTOMOUS,
                    GrowthPattern.SYMPODIAL):
        plant = dev.grow(_profile(replace(STEM, growth_pattern=pattern, unit_steps=3),
                                  branch), seed=2)
        for axis in _units(plant):
            tip = _vegetative(axis) - 1
            at = {a.parent_phytomer_index for a in plant.axes.values()
                  if a.parent_axis_id == axis.id and not a.is_fork}
            assert at == set(range(tip)), (pattern, axis.id, sorted(at))


def test_forking_node_grows_no_side_branch():
    # Where a tip forks, the forks carry it on -- following the parent's
    # heading, Spread / 2 off it -- and that node grows no side branch at the
    # branch level's pitch beside them.
    branch = BranchLevel(start_pitch_deg=50.0, steps=3)
    level = replace(STEM, growth_pattern=GrowthPattern.DICHOTOMOUS, fork_probability=1.0,
                    steps=20)
    plant = dev.grow(_profile(level, branch), seed=9)
    stem = plant.axes[0]
    fork_node = len(stem.phytomers) - 1
    forks = [a for a in plant.axes.values() if a.parent_axis_id == stem.id and a.is_fork]
    side = [a for a in plant.axes.values() if a.parent_axis_id == stem.id and not a.is_fork]
    assert forks and all(abs(_angle(f.base_direction, stem.phytomers[-1].direction) - 20.0)
                         < 1e-6 for f in forks)
    assert side and all(a.parent_phytomer_index < fork_node for a in side), \
        sorted(a.parent_phytomer_index for a in side)


def test_sympodial_chain():
    plant = dev.grow(_profile(SYMPODIAL), seed=4)
    units = _units(plant)
    # 10 nodes in units of 3: 3 + 3 + 3 + 1, one relay each
    assert [_vegetative(u) for u in units] == [3, 3, 3, 1], [_vegetative(u) for u in units]
    assert sum(_vegetative(u) for u in units) == SYMPODIAL.steps
    for parent, relay in zip(units, units[1:], strict=False):
        assert relay.parent_axis_id == parent.id and relay.is_fork
        tip = parent.phytomers[-1]
        assert abs(_angle(relay.base_direction, tip.direction) - 20.0) < 1e-6
        assert np.allclose(relay.base_position, tip.tip_position)
        # a relay leads: it keeps the chain's setpoint, not its launch angle
        assert relay.setpoint_tilt_deg == units[0].setpoint_tilt_deg


def test_sympodial_rights_itself():
    # With gravitropism the relays kink off and bend back upright: the chain's
    # top stays over its base (a zig-zag) instead of stair-stepping away.
    level = replace(SYMPODIAL, steps=12, fork_spread_deg=60.0, fork_whorl_deg=180.0)
    plant = dev.grow(_profile(level, gravitropism=0.6), seed=4)
    units = _units(plant)
    top = units[-1].phytomers[-1].tip_position
    base = units[0].base_position
    lean = math.degrees(math.atan2(math.hypot(*(top - base)[:2]), (top - base)[2]))
    tilts = [_angle(u.phytomers[-1].direction, (0, 0, 1)) for u in units]
    assert lean < 12.0, (lean, tilts)


def test_unit_tip_flowers_when_terminal():
    # Terminal (the default): every unit ends in its flower, straight after
    # its Unit Steps. Flowers only at the nodes, or none: every unit ends bare.
    flowering = dev.grow(_profile(SYMPODIAL), seed=4)
    units = _units(flowering)
    assert [_vegetative(u) for u in units] == [3, 3, 3, 1]
    assert all(u.flowering_from >= 0 or u.terminal_organ is not None for u in units)
    for inf in (_node_borne(InflorescenceKind.SOLITARY, Attachment.AXILLARY), NO_FLOWERS):
        bare = dev.grow(_profile(SYMPODIAL, inflorescence=inf), seed=4)
        assert [_vegetative(u) for u in _units(bare)] == [3, 3, 3, 1]
        assert all(u.flowering_from < 0 and u.terminal_organ is None for u in _units(bare))


def test_dichasial_and_pleiochasial():
    for pattern, n in ((GrowthPattern.DICHASIAL, 2), (GrowthPattern.PLEIOCHASIAL, 4)):
        level = replace(SYMPODIAL, growth_pattern=pattern, relay_count=4, steps=9)
        plant = dev.grow(_profile(level), seed=5)
        by_gen: dict = {}
        for u in _units(plant):
            by_gen.setdefault(_generation(plant, u), []).append(u)
        assert {g: len(v) for g, v in by_gen.items()} == {0: 1, 1: n, 2: n * n}, \
            (pattern, {g: len(v) for g, v in by_gen.items()})
        for u in by_gen[1]:
            tip = plant.axes[u.parent_axis_id].phytomers[-1]
            assert abs(_angle(u.base_direction, tip.direction) - 20.0) < 1e-6


def test_aligned_relay():
    level = replace(SYMPODIAL, growth_pattern=GrowthPattern.PSEUDO_RELAY)
    # a raceme: the spent apex's flowering region leans its first node aside
    plant = dev.grow(_profile(level, inflorescence=Inflorescence(
        kind=InflorescenceKind.RACEME, flower_count=5)), seed=6)
    units = _units(plant)
    assert len(units) == 4
    for parent, relay in zip(units, units[1:], strict=False):
        tip = parent.phytomers[parent.flowering_from - 1]
        assert _angle(relay.base_direction, tip.direction) < 1e-6
        lean = _angle(parent.phytomers[parent.flowering_from].direction, tip.direction)
        assert abs(lean - 20.0) < 1e-6, lean
    # solitary (the default): the one flower capping the spent apex leans aside
    plant = dev.grow(_profile(level), seed=6)
    units = _units(plant)
    for parent, relay in zip(units, units[1:], strict=False):
        tip = parent.phytomers[-1]
        organ = parent.terminal_organ
        assert organ is not None
        faces = organ.direction    # as `resolve` placed it
        assert abs(_angle(faces, tip.direction) - 20.0) < 1e-6
        assert _angle(faces, relay.base_direction) > 19.0   # away from the relay


def test_unequal_fork():
    level = replace(STEM, growth_pattern=GrowthPattern.PSEUDO_FORK, fork_probability=1.0,
                    fork_balance=0.6, steps=20, width_scale_per_step=0.9)
    p = _profile(level)
    plant = dev.grow(p, seed=9)
    dev._level_radii(plant, p, 9)
    stem = plant.axes[0]
    kids = sorted((a for a in plant.axes.values() if a.parent_axis_id == stem.id and a.is_fork),
                  key=lambda a: a.id)
    assert len(kids) == 2
    leader, weak = kids
    tip = stem.phytomers[-1]
    remaining = 20 - len(stem.phytomers)
    assert abs(_angle(leader.base_direction, tip.direction) - 20.0 * 0.4) < 1e-6
    assert abs(_angle(weak.base_direction, tip.direction) - 20.0 * 1.6) < 1e-6
    assert leader.node_budget == remaining
    assert weak.node_budget == max(2, round(remaining * 0.7))
    assert abs(weak.phytomers[0].radius / leader.phytomers[0].radius - 0.7) < 1e-12


def test_dichotomous_unchanged_by_balance():
    # Balance belongs to the Unequal Fork only: an equal fork ignores it.
    level = replace(STEM, growth_pattern=GrowthPattern.DICHOTOMOUS, fork_probability=1.0,
                    steps=20)
    a = dev.grow(_profile(level), seed=9)
    b = dev.grow(_profile(replace(level, fork_balance=0.9)), seed=9)
    assert [(x.id, len(x.phytomers), x.node_budget) for x in a.axes.values()] == \
        [(x.id, len(x.phytomers), x.node_budget) for x in b.axes.values()]


def test_flowers_only_where_attached():
    # Nothing but the attachments places a flower: Terminal alone -- tips only;
    # Axillary alone -- nodes only, never a tip (a tip no attachment takes stops).
    branch = BranchLevel(start_pitch_deg=50.0, steps=4)
    tips = dev.grow(_profile(STEM, branch), seed=3)
    assert all(ph.organs == [] or all(o.kind != OrganKind.FLOWER for o in ph.organs)
               for a in tips.axes.values() for ph in a.phytomers)
    assert all(a.terminal_organ is not None for a in tips.axes.values())
    nodes = dev.grow(_profile(STEM, branch, inflorescence=_node_borne(
        InflorescenceKind.SOLITARY, Attachment.AXILLARY)), seed=3)
    assert all(a.terminal_organ is None and a.flowering_from < 0 for a in nodes.axes.values())
    assert sum(o.kind == OrganKind.FLOWER for a in nodes.axes.values()
               for ph in a.phytomers for o in ph.organs) > 0


def test_node_borne_solitary_sits_where_attached():
    # A solitary flower borne at the nodes is the bud itself -- no stalk axis --
    # at the node (Axillary), halfway down the internode (Internodal), or half
    # a turn round from the axil (Leaf-opposed).
    where = {}
    for mode in (Attachment.AXILLARY, Attachment.INTERNODAL, Attachment.LEAF_OPPOSED):
        plant = dev.grow(_profile(STEM, inflorescence=_node_borne(
            InflorescenceKind.SOLITARY, mode)), seed=1)
        assert not [a for a in plant.axes.values() if a.structural_depth >= 1
                    and not a.is_fork], mode
        stem = plant.axes[0]
        flowers = [(ph, o) for ph in stem.phytomers for o in ph.organs
                   if o.kind == OrganKind.FLOWER]
        assert flowers, mode
        ph, o = flowers[0]
        seat = o.position - o.direction * 0.0   # no stalk: it sits at its anchor
        along = (float(np.linalg.norm(seat - ph.base_position))
                 / max(float(np.linalg.norm(ph.tip_position - ph.base_position)), 1e-12))
        where[mode] = (round(along, 6), o.divergence_deg)
    assert where[Attachment.AXILLARY][0] == 1.0
    assert where[Attachment.INTERNODAL][0] == 0.5
    assert where[Attachment.LEAF_OPPOSED][0] == 1.0
    turn = (where[Attachment.LEAF_OPPOSED][1] - where[Attachment.AXILLARY][1]) % 360.0
    assert abs(turn - 180.0) < 1e-9, turn


def test_axillary_stalks_open():
    # A stalk released at a node is an inflorescence: it opens its raceme after
    # its peduncle, and its bare peduncle nodes bear no flowers of their own.
    plant = dev.grow(_profile(STEM, BranchLevel(start_pitch_deg=50.0, steps=4),
                              inflorescence=_node_borne(InflorescenceKind.RACEME,
                                                        Attachment.AXILLARY)), seed=3)
    stalks = [a for a in plant.axes.values() if a.structural_depth >= 1 and not a.is_fork]
    assert stalks and all(a.flowering_from >= 0 for a in stalks)
    assert all(o.kind != OrganKind.FLOWER for a in stalks
               for ph in a.phytomers[:a.flowering_from] for o in ph.organs)


def test_axillary_along_every_unit():
    # Node-borne flowers on a sympodial chain: every unit bears them (a relay
    # is a stem like any other), on each node but its top one -- where the
    # relay leaves -- and no unit ends in a terminal one (the plant flowers
    # only at its nodes, so a spent unit's tip stops bare).
    level = replace(SYMPODIAL, steps=12)
    plant = dev.grow(_profile(level, inflorescence=_node_borne(
        InflorescenceKind.SOLITARY, Attachment.AXILLARY)), seed=4)
    for u in _units(plant):
        per = [sum(o.kind == OrganKind.FLOWER for o in ph.organs) for ph in u.phytomers]
        assert per == [1] * (len(per) - 1) + [0], (u.id, per)
        assert u.terminal_organ is None and u.flowering_from < 0


def _flowered(axis) -> bool:
    return (axis.flowering_from >= 0 or axis.terminal_organ is not None
            or any(o.kind == OrganKind.FLOWER for ph in axis.phytomers for o in ph.organs))


def test_flowers_only_on_placed_levels():
    # The flower component placed on the stems only: the branches grow as on
    # a flowerless plant -- no tip inflorescence, no flower spacing -- their
    # nodes exactly a flowerless plant's. Placed nowhere (no flower
    # component): nothing flowers.
    branch = BranchLevel(start_pitch_deg=50.0, steps=4)
    raceme = Inflorescence(kind=InflorescenceKind.RACEME, flower_count=6)
    stems_only = replace(raceme, placement=replace(raceme.placement, levels=(0,)))
    plant = dev.grow(_profile(STEM, branch, inflorescence=stems_only), seed=3)
    flowerless = dev.grow(_profile(STEM, branch, inflorescence=NO_FLOWERS), seed=3)
    assert _flowered(plant.axes[0])
    branches = [a for a in plant.axes.values() if a.level == 1]
    assert branches and not any(_flowered(a) for a in branches)
    # side branches only, by where they leave: the flowerless stem's tip grows
    # on as one more level-1 axis
    bare = {a.parent_phytomer_index: [round(ph.final_length, 9) for ph in a.phytomers]
            for a in flowerless.axes.values() if a.level == 1 and not a.is_continuation}
    assert {a.parent_phytomer_index: [round(ph.final_length, 9) for ph in a.phytomers]
            for a in branches} == bare
    nowhere = replace(raceme, placement=replace(raceme.placement, levels=()))
    plant = dev.grow(_profile(STEM, branch, inflorescence=nowhere), seed=3)
    assert not any(_flowered(a) for a in plant.axes.values())


def _flowers(plant):
    """(axis, organ) for every flower, terminal ones included."""
    out = [(a, o) for a in plant.axes.values() for ph in a.phytomers for o in ph.organs
           if o.kind == OrganKind.FLOWER]
    return out + [(a, a.terminal_organ) for a in plant.axes.values()
                  if a.terminal_organ is not None]


def _on(inf: Inflorescence, *levels: int) -> Inflorescence:
    return replace(inf, placement=replace(inf.placement, levels=tuple(levels)))


RACEME = Inflorescence(kind=InflorescenceKind.RACEME, flower_count=6)


def test_two_inflorescences_combine():
    # A Terminal raceme and Axillary solitaries from a second inflorescence on
    # the same stems: different spots, so both grow -- the tip a raceme of the
    # first, every node below a flower of the second at its own size.
    extra = replace(_on(_node_borne(InflorescenceKind.SOLITARY, Attachment.AXILLARY), 0),
                    flower_size=2.0)
    plant = dev.grow(_profile(STEM, inflorescence=_on(RACEME, 0),
                              extra_inflorescences=[extra]), seed=3)
    stem = plant.axes[0]
    assert stem.flowering_from > 0 and stem.inflorescence == 0
    own = [o for a, o in _flowers(plant) if a.id == 0 and o.inflorescence == 1]
    assert len(own) == stem.flowering_from - 1          # every leafy node but the tip's
    assert all(o.final_scale == 2.0 for o in own)
    assert all(o.inflorescence == 0 for a, o in _flowers(plant)
               if a.id == 0 and o not in own)


def test_contested_tip_goes_to_the_first():
    # Both want the tips: on the stems (where both are placed) the first has
    # them; on the branches (only the second) the second does.
    branch = BranchLevel(start_pitch_deg=50.0, steps=4)
    solitary = Inflorescence(kind=InflorescenceKind.SOLITARY)
    plant = dev.grow(_profile(STEM, branch, inflorescence=_on(RACEME, 0),
                              extra_inflorescences=[_on(solitary, 0, 1)]), seed=3)
    assert plant.axes[0].flowering_from > 0
    assert {o.inflorescence for a, o in _flowers(plant) if a.level == 0} == {0}
    branches = [a for a in plant.axes.values() if a.level == 1]
    assert branches and all(a.terminal_organ is not None and a.terminal_organ.inflorescence == 1
                            and a.flowering_from < 0 for a in branches)


def test_extra_inflorescence_on_its_levels_only():
    # Placed on the branches only, an axillary inflorescence flowers their
    # nodes and leaves the stems' alone.
    branch = BranchLevel(start_pitch_deg=50.0, steps=4)
    extra = _on(_node_borne(InflorescenceKind.SOLITARY, Attachment.AXILLARY), 1)
    plant = dev.grow(_profile(STEM, branch, inflorescence=_on(RACEME, 0),
                              extra_inflorescences=[extra]), seed=3)
    by_level = {}
    for a, o in _flowers(plant):
        by_level.setdefault(o.inflorescence, set()).add(a.level)
    assert by_level == {0: {0}, 1: {1}}, by_level


def _tilt(d) -> float:
    d = np.asarray(d, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(d[2] / np.linalg.norm(d))))))


def test_parent_aligned_follows_the_branch_it_ends():
    # A raceme ending a branch, Parent-aligned at full strength, carries on
    # along that branch -- not the stem the branch grew from.
    branch = BranchLevel(start_pitch_deg=50.0, steps=4)
    raceme = replace(RACEME, orientation_mode=OrientationMode.PARENT_ALIGNED,
                     orientation_strength=1.0)
    plant = dev.grow(_profile(STEM, branch, inflorescence=raceme), seed=3)
    branches = [a for a in plant.axes.values() if a.level == 1 and a.flowering_from > 0]
    assert branches
    for a in branches:
        leafy = _tilt(a.phytomers[a.flowering_from - 1].direction)
        assert leafy > 30.0
        assert all(abs(_tilt(ph.direction) - leafy) < 3.0
                   for ph in a.phytomers[a.flowering_from:]), a.id


def test_orientation_turns_smoothly():
    # The turn onto an orientation takes as many nodes as a smooth arc needs
    # (a facet's worth per node), and a long inflorescence still arrives by
    # its midpoint and holds; a short one turns faster to arrive in time.
    for flowers, most in ((40, 7.5), (6, 60.0)):
        raceme = replace(RACEME, flower_count=flowers,
                         orientation_mode=OrientationMode.HORIZONTAL, orientation_strength=1.0)
        # Leaning a little: a dead-vertical stem has no plane to tilt in.
        stem = dev.grow(_profile(replace(STEM, start_pitch_deg=5.0), inflorescence=raceme,
                                 gravitropism=1.0), seed=3).axes[0]
        f = stem.flowering_from
        tilts = [_tilt(ph.direction) for ph in stem.phytomers]
        turns = [abs(tilts[i] - tilts[i - 1]) for i in range(f, len(tilts))]
        assert max(turns) <= most, (flowers, max(turns))
        mid = f + (len(tilts) - f) // 2 + 1
        assert all(abs(t - 90.0) < 3.0 for t in tilts[mid:]), (flowers, tilts[f:])


def _climb(plant) -> float:
    """The most any aimed inflorescence's flowering region turns back up after
    reaching its lowest tilt (0 for one that hangs)."""
    worst = 0.0
    for a in plant.axes.values():
        if a.orientation_target_deg < 0.0 or a.flowering_from < 0:
            continue
        lowest = 0.0
        for ph in a.phytomers[a.flowering_from:]:
            t = _tilt(ph.direction)
            lowest = max(lowest, t)
            worst = max(worst, lowest - t)
    return worst


def test_hanging_raceme_stays_hanging():
    # A pendulous raceme on a branch that sags under it: the sag swings it, and
    # it hangs back down rather than being carried past straight down and
    # curling up into a loop (it climbed back to 84 degrees from 180).
    branch = BranchLevel(start_pitch_deg=50.0, steps=5, start_length_scale=1.0)
    raceme = replace(RACEME, flower_count=14, orientation_mode=OrientationMode.PENDULOUS,
                     orientation_strength=1.0)
    # Checked without the ground: this little plant's raceme hangs 6 cm past
    # its base, where the ground would (rightly) lay it down
    # (tests/ground_support.py) -- that is not the swing this guards.
    from ..growth import physics
    orig = physics._ground_lift
    physics._ground_lift = lambda *a, **k: None
    try:
        plant = dev.build(_profile(STEM, branch, inflorescence=_on(raceme, 1),
                                   gravitropism=0.35), seed=3, age=1.0)
    finally:
        physics._ground_lift = orig
    assert _climb(plant) < 3.0, _climb(plant)


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
