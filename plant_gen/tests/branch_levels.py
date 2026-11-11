"""Tests for the branching levels, headless.

Every level control against its closed form, on a plant with the wander,
gravitropism, internode variation and droop off, so the numbers are exact:

1. **Stems**: Whorl stems, in the level's slots at the basal spread, each
   splayed Start Pitch from vertical.
2. **Branches per node** = the next level's Whorl, at its slot angles.
3. **Launch**: a branch leaves its parent at Start Pitch; with Start Yaw its
   angle to the parent is acos(cos pitch x cos yaw).
4. **Start Roll** twists the branch's frame about its heading.
5. **Steps**: a vegetative axis builds exactly Steps nodes, then flowers.
6. **Length**: internode i = base x Length / Step ** i; a stem's base is its
   level's Start Length (metres), a branch's the parent's nominal internode
   where it attaches x Start Length.
7. **Width**: node i = base x Width / Step ** i; a stem's base is its level's
   Start Width (metres), a branch's the parent's radius where it attaches x
   Start Width.
8. **Spiral**: with no roll the heading turns Pitch / Step per node, in one
   plane (k nodes: k x pitch); with roll it leaves that plane.
9. **Fork**: Probability 0 never forks; a fork splits into Branches children,
   each Spread / 2 off the parent tip, sharing its remaining nodes.

    python -m plant_gen.tests.branch_levels
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..growth import development as dev
from ..growth.morphology import (
    BranchLevel,
    Inflorescence,
    InflorescenceKind,
    MorphologyProfile,
)


def _profile(*levels: BranchLevel, **kw) -> MorphologyProfile:
    """A plant with nothing but the levels shaping it."""
    base = dict(gravitropism=0.0, circumnutation_deg=0.0,
                droop=0.0)
    base.update(kw)
    return MorphologyProfile(levels=list(levels), **base)


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _children(plant, axis):
    return [a for a in plant.axes.values()
            if a.parent_axis_id == axis.id and not a.is_fork and a.structural_depth < 1]


STEM = BranchLevel(start_pitch_deg=0.0, steps=12, start_length_scale=0.045,
                   start_width_scale=0.005)
BRANCH = BranchLevel(start_pitch_deg=50.0, steps=6)


def test_stems_whorl_spread_and_splay():
    p = _profile(replace(STEM, whorl=5, start_pitch_deg=20.0), basal_spread=0.03)
    plant = dev.grow(p, seed=3)
    stems = [plant.axes[i] for i in plant.basal_axis_ids]
    assert len(stems) == 5, len(stems)
    # a phyllotactic disc (Vogel): stem n n golden angles round, at
    # spread x sqrt((n + 1/2) / count) -- the crown filled, not its rim
    golden = 180.0 * (3.0 - math.sqrt(5.0))
    for n, axis in enumerate(stems):
        r = np.linalg.norm(axis.base_position)
        assert abs(r - 0.03 * math.sqrt((n + 0.5) / 5)) < 1e-9, (n, r)
        az = math.degrees(math.atan2(axis.base_position[1], axis.base_position[0])) % 360.0
        assert abs(((az - golden * n) + 180.0) % 360.0 - 180.0) < 1e-6, (n, az)
        assert abs(_angle(axis.base_direction, (0, 0, 1)) - 20.0) < 1e-6
    # a set Whorl Angle steps them instead of the golden angle
    p = _profile(replace(STEM, whorl=4, whorl_angle_deg=90.0), basal_spread=0.03)
    plant = dev.grow(p, seed=3)
    for n, i in enumerate(plant.basal_axis_ids):
        pos = plant.axes[i].base_position
        az = math.degrees(math.atan2(pos[1], pos[0])) % 360.0
        assert abs(((az - 90.0 * n) + 180.0) % 360.0 - 180.0) < 1e-6, (n, az)


def test_branches_per_node_is_the_next_whorl():
    for whorl in (1, 2, 3, 4):
        p = _profile(STEM, replace(BRANCH, whorl=whorl, spiral_angle_deg=0.0))
        plant = dev.grow(p, seed=1)
        stem = plant.axes[plant.basal_axis_ids[0]]
        per_node: dict[int, int] = {}
        for c in _children(plant, stem):
            per_node[c.parent_phytomer_index] = per_node.get(c.parent_phytomer_index, 0) + 1
        assert set(per_node.values()) == {whorl}, (whorl, per_node)
        # every node but the tip: a monopodial stem's last node is its own
        assert sorted(per_node) == list(range(STEM.steps - 1)), per_node


def test_launch_pitch_and_yaw():
    for pitch, yaw in ((50.0, 0.0), (30.0, 0.0), (50.0, 25.0), (70.0, -40.0)):
        p = _profile(STEM, replace(BRANCH, start_pitch_deg=pitch, start_yaw_deg=yaw))
        plant = dev.grow(p, seed=2)
        stem = plant.axes[plant.basal_axis_ids[0]]
        want = math.degrees(math.acos(math.cos(math.radians(pitch))
                                      * math.cos(math.radians(yaw))))
        for c in _children(plant, stem):
            got = _angle(c.base_direction, stem.phytomers[c.parent_phytomer_index].direction)
            assert abs(got - want) < 1e-6, (pitch, yaw, got, want)


def test_yaw_turns_a_square_branch_round_its_parent():
    # Yaw swings a branch sideways round its parent by the full angle at any
    # pitch: one leaving square to an upright stem moves Yaw degrees round it.
    # (It turned about the outward radial, a spin about the branch's own length
    # there: the swing was cos(pitch) x yaw.)
    base = dev.grow(_profile(STEM, replace(BRANCH, start_pitch_deg=90.0)), seed=2)
    yawed = dev.grow(_profile(STEM, replace(BRANCH, start_pitch_deg=90.0, start_yaw_deg=12.0)),
                     seed=2)
    a = sorted((c for c in base.axes.values() if c.order == 1), key=lambda c: c.id)
    b = sorted((c for c in yawed.axes.values() if c.order == 1), key=lambda c: c.id)
    assert len(a) == len(b) > 0
    for x, y in zip(a, b, strict=True):
        assert abs(_angle(x.base_direction, y.base_direction) - 12.0) < 1e-6
        assert abs(float(y.base_direction[2])) < 1e-9          # still square to the stem


def test_start_roll_twists_the_frame():
    plain = dev.grow(_profile(STEM, BRANCH), seed=4)
    rolled = dev.grow(_profile(STEM, replace(BRANCH, start_roll_deg=35.0)), seed=4)
    a = [c for c in plain.axes.values() if c.order == 1]
    b = [c for c in rolled.axes.values() if c.order == 1]
    assert len(a) == len(b) > 0
    for x, y in zip(a, b, strict=True):
        assert np.allclose(x.base_direction, y.base_direction)
        assert abs(_angle(x.base_reference, y.base_reference) - 35.0) < 1e-6


def test_upper_side_up_lays_a_distichous_level_flat():
    # Upper Side Up: each branch's frame starts at its top (world up across its
    # heading), so with Start Roll 90 a Distichous level on it puts its
    # branchlets out to the sides of every branch of a whorl: a branchlet
    # leaving at pitch p rises exactly cos(p) x its node's rise (the side is
    # level). Off, the whorl's members come out turned every which way.
    whorl = replace(BRANCH, whorl=5, spiral_angle_deg=36.0, start_pitch_deg=80.0, steps=8,
                    start_roll_deg=90.0)
    twig = BranchLevel(parent=1, start_pitch_deg=60.0, steps=3, spiral_angle_deg=180.0)

    def worst(upper: bool) -> tuple[float, int]:
        plant = dev.grow(_profile(STEM, replace(whorl, upper_side_up=upper), twig), seed=5)
        err, n = 0.0, 0
        for c in plant.axes.values():
            if c.order != 2 or c.is_fork:
                continue
            node = plant.axes[c.parent_axis_id].phytomers[c.parent_phytomer_index].direction
            err = max(err, abs(float(c.base_direction[2])
                               - math.cos(math.radians(60.0)) * float(node[2])))
            n += 1
        return err, n

    on, n = worst(True)
    off, _ = worst(False)
    print(f"         {n} branchlets: worst rise error {on:.1e} (Upper Side Up off: {off:.2f})")
    assert n > 0 and on < 1e-9 and off > 0.1
    # and the branches' own frames start at their top
    plant = dev.grow(_profile(STEM, replace(whorl, upper_side_up=True, start_roll_deg=0.0)),
                     seed=5)
    for c in plant.axes.values():
        if c.order == 1:
            d = c.base_direction
            up = np.array([0.0, 0.0, 1.0]) - d * d[2]
            assert np.allclose(c.base_reference, up / np.linalg.norm(up), atol=1e-12)


def test_steps_then_flowers():
    p = _profile(replace(STEM, steps=9), replace(BRANCH, steps=4))
    plant = dev.grow(p, seed=5)
    for axis in plant.axes.values():
        if axis.structural_depth >= 1:
            continue
        want = 9 if axis.order == 0 else 4
        veg = axis.flowering_from if axis.flowering_from >= 0 else len(axis.phytomers)
        assert veg == want, (axis.order, veg, want)


def test_length_scales():
    stem = replace(STEM, start_length_scale=0.06, length_scale_per_step=0.9)
    branch = replace(BRANCH, start_length_scale=0.5, length_scale_per_step=0.8)
    p = _profile(stem, branch, internode_length=0.05)   # flowers' base only
    plant = dev.grow(p, seed=6)
    s = plant.axes[plant.basal_axis_ids[0]]
    for i in range(stem.steps):
        assert abs(s.phytomers[i].final_length - 0.06 * 0.9 ** i) < 1e-12, i
    for c in _children(plant, s):
        j = c.parent_phytomer_index
        for k in range(branch.steps):
            want = 0.06 * 0.9 ** j * 0.5 * 0.8 ** k
            assert abs(c.phytomers[k].final_length - want) < 1e-12, (j, k)


def test_width_scales():
    stem = replace(STEM, start_width_scale=0.006, width_scale_per_step=0.9)
    branch = replace(BRANCH, start_width_scale=0.6, width_scale_per_step=0.85)
    p = _profile(stem, branch)
    plant = dev.grow(p, seed=7)
    dev._level_radii(plant, p, 7)
    s = plant.axes[plant.basal_axis_ids[0]]
    for i, ph in enumerate(s.phytomers[:stem.steps]):
        assert abs(ph.radius - 0.006 * 0.9 ** i) < 1e-12, i
    for c in _children(plant, s):
        at = s.phytomers[c.parent_phytomer_index].radius
        for k, ph in enumerate(c.phytomers[:branch.steps]):
            assert abs(ph.radius - at * 0.6 * 0.85 ** k) < 1e-12, k


def test_rachis_tapers_to_its_top_stalk():
    # A raceme's rachis narrows from where its flowering region starts to its
    # top flower's stalk width, so it ends as thin as the stalk beside it;
    # every stalk is read against the region's start, so the top one meets the
    # tip exactly. Below the region, Width / Step's narrowing spread by length
    # (first and last node per node, evenly per metre between), and the rachis
    # sets off at that rate: no corner in the taper where the flowers start.
    stem = replace(STEM, start_width_scale=0.006, width_scale_per_step=0.9)
    inf = Inflorescence(kind=InflorescenceKind.RACEME, flower_count=8, pedicel_width=0.4)
    p = _profile(stem, inflorescence=inf)
    plant = dev.grow(p, seed=7)
    dev._level_radii(plant, p, 7)
    s = plant.axes[plant.basal_axis_ids[0]]
    f = s.flowering_from
    assert 2 < f < len(s.phytomers) - 1
    r = [ph.radius for ph in s.phytomers]
    length = [ph.final_length for ph in s.phytomers]
    assert abs(r[0] - 0.006) < 1e-12 and abs(r[f - 1] - 0.006 * 0.9 ** (f - 1)) < 1e-12
    rates = [math.log(r[i] / r[i - 1]) / length[i] for i in range(1, f)]
    assert max(rates) - min(rates) < 1e-9 * abs(min(rates)), rates      # even by length
    start = r[f - 1]
    end = start * 0.4 * inf.pedicel_width_profile(1.0)
    assert all(r[i] < r[i - 1] for i in range(f, len(r))), "the rachis narrows all the way"
    assert abs(r[-1] - end) < 1e-12
    # the first rachis step continues the stem's rate (the cubic's start slope)
    first = math.log(r[f] / start) / length[f]
    assert abs(first - rates[-1]) < 0.25 * abs(rates[-1]) or first < rates[-1], (first, rates[-1])
    for i in range(f, len(r)):
        assert s.stalk_radius(i) == start


def test_taper_spread_by_length():
    # Internodes shrinking fast toward the tip: per node the width would drop
    # most where the nodes bunch; by length it narrows evenly per metre.
    stem = replace(STEM, start_width_scale=0.006, width_scale_per_step=0.9,
                   length_scale_per_step=0.5)
    p = _profile(stem)
    plant = dev.grow(p, seed=7)
    dev._level_radii(plant, p, 7)
    s = plant.axes[plant.basal_axis_ids[0]]
    r = [ph.radius for ph in s.phytomers]
    length = [ph.final_length for ph in s.phytomers]
    n = len(r)
    assert abs(r[0] - 0.006) < 1e-12 and abs(r[-1] - 0.006 * 0.9 ** (n - 1)) < 1e-12
    rates = [math.log(r[i] / r[i - 1]) / length[i] for i in range(1, n) if length[i] > 0]
    assert max(rates) - min(rates) < 1e-9 * abs(min(rates)), rates


def _slid(plant, s):
    """Per node of stem `s`: its branches' signed slide in internodes (+ up the
    internode above, - down the one below), read off where they really leave."""
    out = {}
    for c in _children(plant, s):
        i = c.parent_phytomer_index
        ph = s.phytomers[i]
        nxt = s.phytomers[i + 1] if i + 1 < len(s.phytomers) else ph
        d = c.base_position - ph.tip_position
        below = float(np.linalg.norm(ph.tip_position - ph.base_position))
        above = float(np.linalg.norm(nxt.tip_position - nxt.base_position))
        along = float(np.dot(d, nxt.tip_position - nxt.base_position)) / above ** 2 \
            if np.dot(d, ph.tip_position - ph.base_position) >= 0 else \
            -float(np.linalg.norm(d)) / below
        out.setdefault(i, []).append(round(along, 9))
    return out


def test_slide_moves_pairs_apart():
    # Slide 1 on an opposite pair: the first a quarter internode up the one
    # above its node, the second a quarter down the one below; 0 leaves both at
    # the node; Slide Variation slides some nodes up and some down.
    for slide, want in ((0.0, [0.0, 0.0]), (1.0, [0.25, -0.25]), (-0.4, [0.1, -0.1])):
        plant = dev.grow(_profile(STEM, replace(BRANCH, whorl=2, slide=slide)), seed=7)
        s = plant.axes[plant.basal_axis_ids[0]]
        got = _slid(plant, s)
        assert got
        for node, slid in got.items():
            assert sorted(slid, reverse=True) == sorted(want, reverse=True), (slide, node, slid)
    plant = dev.grow(_profile(STEM, replace(BRANCH, whorl=2, slide_variation=1.0)), seed=7)
    s = plant.axes[plant.basal_axis_ids[0]]
    firsts = [max(v) for v in _slid(plant, s).values()]
    assert all(abs(x) <= 0.25 + 1e-9 for x in firsts)
    assert len({round(x, 6) for x in firsts}) > 1           # differs node to node


def test_spiral_pitch_and_roll():
    for pitch in (5.0, 12.0):
        p = _profile(replace(STEM, spiral_pitch_deg=pitch))
        stem = dev.grow(p, seed=8).axes[0]
        d = [ph.direction for ph in stem.phytomers[:STEM.steps]]
        for k in range(1, 8):
            assert abs(_angle(d[0], d[k]) - k * pitch) < 1e-6, (pitch, k)
        # no roll: every heading stays in one plane
        n = np.cross(d[0], d[1])
        assert max(abs(float(np.dot(n / np.linalg.norm(n), x))) for x in d) < 1e-9
    rolled = dev.grow(_profile(replace(STEM, spiral_pitch_deg=10.0, spiral_roll_deg=40.0)),
                      seed=8).axes[0]
    d = [ph.direction for ph in rolled.phytomers[:STEM.steps]]
    n = np.cross(d[0], d[1])
    assert max(abs(float(np.dot(n / np.linalg.norm(n), x))) for x in d) > 0.05
    for k in range(1, STEM.steps):   # every step still turns exactly Pitch / Step
        assert abs(_angle(d[k - 1], d[k]) - 10.0) < 1e-6, k


def test_fork():
    off = dev.grow(_profile(replace(STEM, growth_pattern="dichotomous", fork_probability=0.0,
                                    steps=20)), seed=9)
    assert not any(a.is_fork for a in off.axes.values())
    forky = replace(STEM, growth_pattern="dichotomous", fork_probability=1.0, fork_branches=3,
                    fork_spread_deg=40.0, steps=20)
    plant = dev.grow(_profile(forky), seed=9)
    stem = plant.axes[0]
    kids = [a for a in plant.axes.values() if a.parent_axis_id == stem.id and a.is_fork]
    assert len(kids) == 3, len(kids)
    tip = stem.phytomers[-1]
    remaining = 20 - len(stem.phytomers)
    for k in kids:
        assert abs(_angle(k.base_direction, tip.direction) - 20.0) < 1e-6
        assert k.node_budget == max(2, remaining)
    # statistical: over many single-node rolls the fork rate is the probability
    hits = trials = 0
    for seed in range(200):
        pl = dev.grow(_profile(replace(STEM, growth_pattern="dichotomous",
                                       fork_probability=0.15, steps=6)), seed=seed)
        s = pl.axes[0]
        trials += 1
        hits += any(a.is_fork and a.parent_axis_id == s.id for a in pl.axes.values())
    # nodes 2 and 3 are rolled (never the base two, never the last two)
    want = 1.0 - (1.0 - 0.15) ** 2
    assert abs(hits / trials - want) < 0.08, (hits / trials, want)


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
