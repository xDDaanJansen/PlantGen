"""Tests for tip continuation, headless.

A stem or branch that has built its Steps, its tip not flowering, grows on in
line as the first level growing on its level -- which continues in turn -- so
every line of the plant ends in its lowest level. Checked against closed
forms on a plant with gravitropism, wander and droop off:

1. The continuation is one axis of the next level, leaving the tip node in
   line with the tip's heading, at exactly the tip's radius.
2. The line runs down to the lowest level and stops there.
3. A tip that flowers is spent: it does not continue.
4. A Growth % cut ends at its cut tip, as before.
5. A sympodial chain continues from its last unit only.
6. Same seed, same plant.

    python -m plant_gen.tests.tip_continuation
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..growth import development as dev
from ..growth.morphology import (
    BranchLevel,
    GrowthPattern,
    Inflorescence,
    InflorescenceKind,
    MorphologyProfile,
)

SEED = 3
NO_FLOWERS = Inflorescence(kind=InflorescenceKind.NONE)
STEM = BranchLevel(start_pitch_deg=0.0, steps=8, start_length_scale=0.045,
                   start_width_scale=0.005, width_scale_per_step=0.9)
BRANCH = BranchLevel(parent=0, start_pitch_deg=50.0, steps=5, width_scale_per_step=0.85)
TWIG = BranchLevel(parent=1, start_pitch_deg=50.0, steps=3, width_scale_per_step=0.8)


def _profile(*levels: BranchLevel, **kw) -> MorphologyProfile:
    base = dict(gravitropism=0.0, circumnutation_deg=0.0, droop=0.0, inflorescence=NO_FLOWERS)
    base.update(kw)
    return MorphologyProfile(levels=list(levels), **base)


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _continuation(plant, axis):
    found = [a for a in plant.axes.values()
             if a.parent_axis_id == axis.id and a.is_continuation]
    assert len(found) <= 1, found
    return found[0] if found else None


def _stem(plant):
    return plant.axes[plant.basal_axis_ids[0]]


def test_a_finished_tip_grows_on_as_the_next_level():
    plant = dev.build(_profile(STEM, BRANCH), seed=SEED)
    stem = _stem(plant)
    assert len(stem.phytomers) == STEM.steps
    cont = _continuation(plant, stem)
    assert cont is not None
    assert cont.level == 1 and cont.order == stem.order + 1
    assert cont.parent_phytomer_index == len(stem.phytomers) - 1
    tip = stem.phytomers[-1]
    assert np.allclose(cont.phytomers[0].base_position, tip.tip_position, atol=1e-12)
    # in line as grown; the bending pass then bends the joint as one beam
    grown = dev.grow(_profile(STEM, BRANCH), seed=SEED)
    g_cont = _continuation(grown, _stem(grown))
    assert _angle(g_cont.base_direction, _stem(grown).phytomers[-1].direction) < 1e-6
    assert _angle(cont.phytomers[0].direction, tip.direction) < 0.5
    # exactly the tip's radius, then its own level's Width / Step
    assert math.isclose(cont.phytomers[0].radius, tip.radius, rel_tol=1e-12)
    assert cont.phytomers[-1].radius < tip.radius
    # it holds its level's angle: the level's Start Pitch from straight up
    assert math.isclose(cont.setpoint_tilt_deg, BRANCH.start_pitch_deg, abs_tol=1e-9)
    # ... not stacked: a twig continuing a branch holds the twig level's angle
    deep = dev.build(_profile(STEM, BRANCH, TWIG), seed=SEED)
    twig = _continuation(deep, _continuation(deep, _stem(deep)))
    assert math.isclose(twig.setpoint_tilt_deg, TWIG.start_pitch_deg, abs_tol=1e-9)


def test_the_line_runs_down_to_the_lowest_level():
    plant = dev.build(_profile(STEM, BRANCH, TWIG), seed=SEED)
    line, axis = [], _stem(plant)
    while axis is not None:
        line.append(axis.level)
        axis = _continuation(plant, axis)
    assert line == [0, 1, 2], line
    # no axis of the lowest level continues; every other finished one does
    for a in plant.axes.values():
        if a.level == 2:
            assert _continuation(plant, a) is None


def test_a_flowering_tip_does_not_continue():
    raceme = Inflorescence(kind=InflorescenceKind.RACEME, flower_count=6)
    on_stems = replace(raceme, placement=replace(raceme.placement, levels=(0,)))
    plant = dev.build(_profile(STEM, BRANCH, inflorescence=on_stems), seed=SEED)
    stem = _stem(plant)
    assert stem.flowering_from >= 0 or stem.terminal_organ is not None
    assert _continuation(plant, stem) is None


def test_a_growth_cut_ends_at_its_cut_tip():
    plant = dev.build(_profile(replace(STEM, growth=0.5), BRANCH), seed=SEED)
    assert _continuation(plant, _stem(plant)) is None


def test_a_cut_that_rounds_up_still_ends():
    # 0.95 of 8 nodes is 7.6: the last node is built partly, so the axis has
    # all 8 nodes -- and is still a cut, which ends at its tip
    plant = dev.build(_profile(replace(STEM, growth=0.95), BRANCH), seed=SEED)
    stem = _stem(plant)
    assert len(stem.phytomers) == STEM.steps
    assert _continuation(plant, stem) is None


def test_a_fork_off_a_cut_chain_ends():
    forked = replace(STEM, steps=12, growth=0.8, fork_probability=1.0,
                     growth_pattern=GrowthPattern.PSEUDO_FORK)
    plant = dev.build(_profile(forked, BRANCH), seed=SEED)
    forks = [a for a in plant.axes.values() if a.level == 0 and a.is_fork]
    assert forks
    assert not any(_continuation(plant, a) for a in plant.axes.values() if a.level == 0)


def test_a_weak_fork_keeps_its_chains_cut():
    # an unequal fork's weaker branch gets a rounded share of the budget; it
    # must keep the chain's cut with it -- a budget rounded down under its
    # reach (10.5 -> 10 under 10.19) read as whole and grew on
    runner = replace(STEM, steps=22, growth=0.98, fork_probability=0.14, fork_balance=0.6,
                     growth_pattern=GrowthPattern.PSEUDO_FORK)
    for seed in (13, 19, 20, 26, 27):           # each has a fork that rounds down
        plant = dev.build(_profile(runner, BRANCH), seed=seed)
        forks = [a for a in plant.axes.values() if a.level == 0 and a.is_fork]
        for a in forks:
            assert a.node_reach < a.node_budget, (seed, a.id, a.node_reach, a.node_budget)
        assert not any(_continuation(plant, a) for a in plant.axes.values() if a.level == 0), seed


def test_a_sympodial_chain_continues_from_its_last_unit():
    sym = replace(STEM, steps=9, growth_pattern=GrowthPattern.SYMPODIAL, unit_steps=3,
                  fork_spread_deg=30.0)
    plant = dev.build(_profile(sym, BRANCH), seed=SEED)
    units = [a for a in plant.axes.values() if a.level == 0]
    assert len(units) == 3, len(units)
    conts = [_continuation(plant, u) for u in units]
    assert sum(c is not None for c in conts) == 1
    last = max(units, key=lambda a: a.structural_depth)
    assert _continuation(plant, last) is not None


def test_deterministic():
    p = _profile(STEM, BRANCH, TWIG)
    a, b = dev.build(p, seed=SEED), dev.build(p, seed=SEED)
    key = [(x.id, x.level, x.is_continuation, len(x.phytomers),
            [round(float(ph.radius), 12) for ph in x.phytomers]) for x in a.axes.values()]
    assert key == [(x.id, x.level, x.is_continuation, len(x.phytomers),
                    [round(float(ph.radius), 12) for ph in x.phytomers])
                   for x in b.axes.values()]


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
