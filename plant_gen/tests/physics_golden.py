"""Golden-output regression for the unified gravity solver.

`tests/structural.py` says a plant is *sound* -- nothing kinks, gravity points
down, topology survives -- and `tests/inflorescence_golden.py` says an
inflorescence *kind* is unchanged. Neither says the *bend* is unchanged: a beam
tweak that quietly droops every stem 5 degrees further passes all of both.

So this captures five deliberately isolated load cases and requires the exact
bent geometry to reproduce. Isolated is the whole point: each exercises one
mechanism, so *which case moved* tells you *which mechanism changed*. Change the
self-weight term and only `bare_stem` (and everything downstream) moves; change
the catkin sag and only `catkin` moves; change pedicel droop and `raceme`/
`pendant` move while `bare_stem` does not.

    python -m plant_gen.tests.physics_golden --capture   # rewrite
    python -m plant_gen.tests.physics_golden             # verify

Capture is a separate flag on purpose: a verify that rewrote its own
expectations would pass forever. Every case runs at `max_branch_order = 0` (a
single stem plus its own inflorescence) so the plant is small enough to capture
its whole bent path node-for-node -- the finest granularity a physics change
could hide in -- and so the case varies only its one load, nothing structural.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..growth.development import build
from ..growth.model import OrganKind
from ..growth.morphology import (
    Attachment,
    InflorescenceKind,
    OrientationMode,
    Placement,
    generic_perennial,
)
from ..growth.species import corylus_avellana

GOLDEN_PATH = Path(__file__).with_name("goldens") / "physics.json"

SEED = 0
AGE = 1.0
_PLACES = 6


def _profile_for(key: str):
    """One load case, built to isolate a single bending mechanism.

    Four are a generic perennial varying only their load; the fifth is the real
    hazel, because "the catkin hangs from its own mass with no catkin-specific
    dial" is the acid test of unification and a synthetic catkin would not
    exercise it. All four generic cases keep only the stems level."""
    if key == "catkin":
        # The hazel itself: catkins arise on its branches, so this one keeps its
        # branching. Small already (27 nodes). Retuned in Stage 6 like the rest.
        return corylus_avellana()

    profile = generic_perennial()
    profile.levels = profile.levels[:1]
    infl = profile.inflorescence
    if key == "bare_stem":
        # A leafy stem bending under wood and leaf mass only -- no flower load.
        infl.kind = InflorescenceKind.SOLITARY
        profile.flower_weight = 0.0
    elif key == "heavy_terminal":
        # One big heavy flower on the apex: the terminal-nod / lump-load case.
        # Driven by size, since mass is scale**3 and a default-size flower sits
        # on the apex with too small a moment arm to bend a stiff stem.
        infl.kind = InflorescenceKind.SOLITARY
        profile.flower_size = 10.0
    elif key == "raceme":
        # Flowers distributed up an axis, each on its own pedicel: the
        # distributed-load and pedicel-sag case.
        infl.kind = InflorescenceKind.RACEME
    elif key == "pendant":
        # An axillary raceme grown to hang: orientation setpoint plus gravity,
        # the case the whole "borne at a node, and it hangs" work is about.
        infl.kind = InflorescenceKind.RACEME
        infl.attachments = frozenset({Attachment.AXILLARY})
        infl.orientation_mode = OrientationMode.PENDULOUS
        infl.orientation_strength = 1.0
        infl.placement = Placement(on_trunk=True, on_branches=True, start=0.4, end=1.0)
    else:
        raise KeyError(key)
    return profile


CASES = ("bare_stem", "heavy_terminal", "raceme", "pendant", "catkin")


def all_keys() -> tuple:
    """The one source both capture and verify read, so they cannot disagree
    about the set -- the failure `inflorescence_golden.all_keys` was written to
    stop."""
    return CASES


def _round(vec) -> list:
    return [round(float(c), _PLACES) for c in vec]


def digest(plant) -> dict:
    """The whole bent shape, at a granularity a physics change cannot hide in.

    Every axis's full node path (the bent stem), every organ's position and
    facing (pedicel droop moves these), and every pedicel and catkin polyline
    (where `model._sag` actually acts). Small because each case is a single
    stem. Positions rounded -- the point is a real bend change, not float drift.
    """
    axis_paths = []
    for axis in sorted(plant.axes.values(), key=lambda a: a.id):
        axis_paths.append({
            "id": axis.id,
            "order": axis.order,
            "nodes": len(axis.phytomers),
            "path": [_round(ph.tip_position) for ph in axis.phytomers],
        })

    organs = []
    for organ in sorted(plant.all_organs(), key=lambda o: o.id):
        entry = {
            "id": organ.id,
            "kind": organ.kind,
            "pos": _round(organ.position),
            "dir": _round(organ.direction),
        }
        if organ.pedicel_points:
            entry["pedicel"] = [_round(p) for p in organ.pedicel_points]
        bodies = getattr(organ, "catkin_bodies", None)
        if bodies:
            entry["catkin"] = [[_round(p) for p in body] for body in bodies]
        organs.append(entry)

    phytomers = plant.all_phytomers()
    mean_z = (sum(float(ph.tip_position[2]) for ph in phytomers) / len(phytomers)
              if phytomers else 0.0)
    lo, hi = plant.bounds() if plant.phytomer_count else ([0] * 3, [0] * 3)
    return {
        "nodes": plant.phytomer_count,
        "axes": len(plant.axes),
        "flowers": len(plant.organs_of_kind(OrganKind.FLOWER)),
        "catkins": len(plant.organs_of_kind(OrganKind.CATKIN)),
        "bounds_lo": _round(lo),
        "bounds_hi": _round(hi),
        "mean_node_z": round(mean_z, _PLACES),
        "axis_paths": axis_paths,
        "organs": organs,
    }


def build_all() -> dict:
    return {key: digest(build(_profile_for(key), seed=SEED, age=AGE)) for key in all_keys()}


def _first_difference(want, got, path: str = "") -> str | None:
    """The first place two digests diverge, depth-first, so a count mismatch is
    reported before the hundred positions that follow from it. Mirrors
    `inflorescence_golden._first_difference`."""
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
    print(f"captured {len(all_keys())} load cases -> {GOLDEN_PATH}")


def verify() -> int:
    if not GOLDEN_PATH.exists():
        print(f"no golden file at {GOLDEN_PATH}; run with --capture first")
        return 1
    want = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    got = build_all()

    print("\nphysics golden comparison")
    print("-" * 64)
    failed = 0
    for key in all_keys():
        if key not in want:
            print(f"  [new ] {key} - not in the golden file")
            continue
        diff = _first_difference(want[key], got[key])
        if diff is None:
            d = got[key]
            print(f"  [PASS] {key:<16} {d['nodes']:>4} nodes  {d['flowers']:>3} flowers  "
                  f"{d['catkins']:>2} catkins  meanZ {d['mean_node_z']:+.3f}")
        else:
            failed += 1
            print(f"  [FAIL] {key:<16} first difference at {diff}")
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
