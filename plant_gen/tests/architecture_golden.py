"""Golden-output regression for the branching levels.

`tests/structural.py` says a plant is *sound*, `tests/branch_levels.py` that
each level control meets its closed form, and `tests/inflorescence_golden.py`
that an inflorescence *kind* is unchanged. None guards the whole *vegetative*
skeleton a level stack grows: a change to how stems, branches or forks are laid
out can pass every one of them while quietly rebuilding it.

So this captures a handful of stacks -- a lone stem, a clump, two and three
levels, a whorled level, a spiral, a forking level (determinate and
indeterminate) -- and requires the exact skeleton to reproduce, down to per-axis
order / structural_depth / is_fork and the full node path. Which case moves
tells which mechanism changed: a fork tweak moves only the fork cases, a
phyllotaxis one only the branched ones.

    python -m plant_gen.tests.architecture_golden --capture   # rewrite
    python -m plant_gen.tests.architecture_golden             # verify

Capture is a separate flag on purpose: a verify that rewrote its own
expectations would pass forever. Every case is kept small (a few short levels)
so the whole tree fits node-for-node -- the finest granularity a topology change
could hide in.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ..growth.development import build
from ..growth.morphology import Attachment, BranchLevel, LevelPhyllotaxis, generic_perennial

GOLDEN_PATH = Path(__file__).with_name("goldens") / "architecture.json"

SEED = 3
AGE = 1.0
_PLACES = 6


def _whorled(level: BranchLevel) -> BranchLevel:
    spiral, whorl, angle = LevelPhyllotaxis.PRESETS[LevelPhyllotaxis.WHORLED]
    level.phyllotaxis = LevelPhyllotaxis.WHORLED
    level.spiral_angle_deg, level.whorl, level.whorl_angle_deg = spiral, whorl, angle
    return level


def _profile_for(key: str):
    """One level stack, kept small so its whole tree can be captured."""
    p = generic_perennial()
    # The stems' Start Length / Width are metres (the values the old Internode
    # Length x 1 and Base Radius x 1 gave, so the captured golden holds).
    stem = BranchLevel(start_pitch_deg=0.0, steps=10, start_length_scale=0.045,
                       start_width_scale=0.009, width_scale_per_step=0.93,
                       length_variation=0.18)
    branch = BranchLevel(start_pitch_deg=50.0, steps=5, start_length_scale=0.7,
                         length_scale_per_step=0.9, start_width_scale=0.6,
                         width_scale_per_step=0.9, length_variation=0.18)
    twig = BranchLevel(start_pitch_deg=40.0, steps=3, start_length_scale=0.6,
                       start_width_scale=0.6, length_variation=0.18)
    if key == "stem":
        p.levels = [stem]
    elif key == "clump":
        stem.whorl, stem.start_pitch_deg = 5, 15.0
        p.levels = [stem]
    elif key == "two_levels":
        p.levels = [stem, branch]
    elif key == "three_levels":
        p.levels = [stem, branch, twig]
    elif key == "whorled":
        p.levels = [stem, _whorled(branch)]
    elif key == "spiral":
        stem.spiral_pitch_deg, stem.spiral_roll_deg = 4.0, 30.0
        p.levels = [stem, branch]
    elif key == "fork_determinate":
        stem.growth_pattern, stem.fork_probability, stem.steps = "dichotomous", 0.35, 14
        p.levels = [stem]
    elif key == "fork_indeterminate":
        # Formerly Axis Type Indeterminate: flowers along the stems, not at the
        # tips -- what such a plant migrates to (serialize schema 9).
        stem.growth_pattern, stem.fork_probability, stem.steps = "dichotomous", 0.35, 14
        stem.fork_branches, stem.fork_spread_deg = 3, 50.0
        p.levels = [stem]
        p.inflorescence = replace(p.inflorescence,
                                  attachments=frozenset({Attachment.AXILLARY}))
    else:
        raise KeyError(key)
    return p


CASES = ("stem", "clump", "two_levels", "three_levels", "whorled", "spiral",
         "fork_determinate", "fork_indeterminate")


def all_keys() -> tuple:
    """The one source both capture and verify read, so they cannot disagree
    about the set."""
    return CASES


def _round(vec) -> list:
    return [round(float(c), _PLACES) for c in vec]


def digest(plant) -> dict:
    """The whole skeleton, at a granularity a topology change cannot hide in.

    Per axis: its order, structural_depth and is_fork -- the three fields a
    level's forking writes, so a regression in how a fork inherits them shows up here --
    plus its full node path. And the overall counts and bounds."""
    axes = []
    for axis in sorted(plant.axes.values(), key=lambda a: a.id):
        axes.append({
            "id": axis.id,
            "order": axis.order,
            "structural_depth": axis.structural_depth,
            "is_fork": axis.is_fork,
            "parent": axis.parent_axis_id,
            "nodes": len(axis.phytomers),
            "path": [_round(ph.tip_position) for ph in axis.phytomers],
        })
    lo, hi = plant.bounds() if plant.phytomer_count else ([0] * 3, [0] * 3)
    return {
        "nodes": plant.phytomer_count,
        "axes": len(plant.axes),
        "orders": sorted({a.order for a in plant.axes.values()}),
        "fork_axes": sum(1 for a in plant.axes.values() if a.is_fork),
        "bounds_lo": _round(lo),
        "bounds_hi": _round(hi),
        "axis_list": axes,
    }


def build_all() -> dict:
    return {key: digest(build(_profile_for(key), seed=SEED, age=AGE)) for key in all_keys()}


def _first_difference(want, got, path: str = "") -> str | None:
    """The first place two digests diverge, depth-first, so a count mismatch is
    reported before the positions that follow from it."""
    if isinstance(want, dict) and isinstance(got, dict):
        for key in want:
            if key not in got:
                return f"{path}.{key}: missing"
            found = _first_difference(want[key], got[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return f"{path}: length {len(want)} -> {len(got)}"
        for i, (a, b) in enumerate(zip(want, got, strict=True)):
            found = _first_difference(a, b, f"{path}[{i}]")
            if found:
                return found
        return None
    if want != got:
        return f"{path}: {want!r} -> {got!r}"
    return None


def capture() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(build_all(), indent=1, sort_keys=True), encoding="utf-8")
    print(f"captured {len(all_keys())} architecture cases -> {GOLDEN_PATH}")


def verify() -> int:
    if not GOLDEN_PATH.exists():
        print(f"no golden file at {GOLDEN_PATH}; run with --capture first")
        return 1
    want = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    got = build_all()

    print("\narchitecture golden comparison")
    print("-" * 64)
    failed = 0
    for key in all_keys():
        if key not in want:
            print(f"  [new ] {key} - not in the golden file")
            continue
        diff = _first_difference(want[key], got[key])
        if diff is None:
            d = got[key]
            print(f"  [PASS] {key:<26} {d['nodes']:>4} nodes  {d['axes']:>3} axes  "
                  f"{d['fork_axes']:>3} forks  orders {d['orders']}")
        else:
            failed += 1
            print(f"  [FAIL] {key:<26} first difference at {diff}")
    print(f"\n  {len(all_keys()) - failed} reproduced, {failed} changed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true",
                        help="rewrite the golden file from current behaviour")
    args = parser.parse_args()
    if args.capture:
        capture()
        return 0
    return verify()


if __name__ == "__main__":
    sys.exit(main())
