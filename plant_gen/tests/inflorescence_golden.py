"""Golden-output regression for the nine inflorescence kinds.

Phase 3 replaces the topology half of `InflorescenceKind` with a real
mechanism (vegetativeness), and that is a refactor of *shipped, working*
behaviour rather than a new feature. `tests/structural.py` is not enough of a
guard for it: those ten assertions check that a plant is sound, not that it is
the *same* plant. A raceme quietly becoming a spike passes all ten.

So: capture a digest of every kind before the change, and require them to
reproduce after. The digest is deliberately readable rather than a bare hash --
a hash mismatch tells you something moved and nothing about what, and the whole
point is to be able to diff a failure.

    python -m plant_gen.tests.inflorescence_golden --capture   # rewrite
    python -m plant_gen.tests.inflorescence_golden             # verify

Capture is deliberately a separate flag. A verify that silently rewrote its own
expectations would pass forever.
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

GOLDEN_PATH = Path(__file__).with_name("goldens") / "inflorescence.json"

KINDS = (
    InflorescenceKind.SOLITARY,
    InflorescenceKind.RACEME,
    InflorescenceKind.SPIKE,
    InflorescenceKind.CORYMB,
    InflorescenceKind.UMBEL,
    InflorescenceKind.FLAT_UMBEL,
    InflorescenceKind.CYME,
    # The four named cymose forms. They are `CYME`'s own relay mechanism with
    # `laterals_per_node` and `relay_rotation_deg` stamped, so what these guard
    # is that the stamps keep producing four *different* plants -- the
    # Alternate/Spiral defect (two enum entries, one plant) is exactly what a
    # kind built from shared controls is prone to, and a digest is the only
    # thing that would catch it.
    InflorescenceKind.SCORPIOID,
    InflorescenceKind.HELICOID,
    InflorescenceKind.DICHASIUM,
    InflorescenceKind.PLEIOCHASIUM,
)

SEED = 0
AGE = 1.0
_PLACES = 6


# What the panel's own `_KIND_DEFAULTS` stamps when each cymose form is picked.
# Mirrored here rather than imported because `blender.properties` needs bpy, and
# without them all four kinds would be one plant: the whole point of a named
# form is the numbers it sets.
_CYME_STAMPS = {
    # laterals_per_node, relay_rotation_deg, branching_depth, branch_angle_deg
    InflorescenceKind.CYME: (2, 90.0, 3, 45.0),
    InflorescenceKind.SCORPIOID: (1, 180.0, 8, 45.0),
    InflorescenceKind.HELICOID: (1, 0.0, 8, 60.0),
    InflorescenceKind.DICHASIUM: (2, 90.0, 3, 45.0),
    InflorescenceKind.PLEIOCHASIUM: (3, 60.0, 2, 45.0),
}


EXTRA = {
    # key -> (kind, {field: value}), for cases a bare kind cannot express.
    #
    # `compound_corymb_cut` exists because the flat-solve ordering bug it
    # guards is only visible on a **truncated** axis, and every branch in the
    # entry above finishes. Asking for more flowers than the span can hold is
    # what reproduces it: the upper branches build short, `_renormalise_
    # truncated_regions` re-reads their gradients, and until it was moved ahead
    # of `solve_flat_head` that re-read silently overwrote the solved pedicels
    # -- 66-76 mm off flat against 4-14 mm on the branches that finished.
    "compound_corymb_cut": (InflorescenceKind.CORYMB,
                            {"branching_depth": 1, "flower_count": 20}),

    # The three layers, guarded one at a time and then together. A kind used to
    # decide all three at once, so these are the entries that would catch one
    # of them leaking back into another -- an orientation entry whose *counts*
    # move, say, means it is doing something structural, which it must not.
    "axillary_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.AXILLARY}),
         "placement": Placement(on_trunk=True, on_branches=True, start=0.4, end=1.0)}),
    "pendulous_raceme": (
        InflorescenceKind.RACEME,
        {"orientation_mode": OrientationMode.PENDULOUS, "orientation_strength": 1.0}),
    "axillary_pendulous_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.AXILLARY}),
         "orientation_mode": OrientationMode.PENDULOUS, "orientation_strength": 1.0,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.4, end=1.0)}),
    # `DYNAMIC` is the one orientation that is not a dispatch to something that
    # already existed, so it earns an entry of its own: its setpoint *moves*
    # while the axis fills with flowers, which no post-growth bend can do.
    "load_bent_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.AXILLARY}),
         "orientation_mode": OrientationMode.PENDULOUS, "orientation_strength": 1.0,
         "orientation_load_driven": True,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.4, end=1.0)}),
    "internodal_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.INTERNODAL}), "internode_position": 0.4,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.4, end=1.0)}),
    "cauline_raceme": (
        InflorescenceKind.RACEME,
        # Was `Attachment.CAULINE`, which fired on the trunk AND the branches --
        # the conflation the CAULIFLOROUS/RAMIFLOROUS split exists to undo. This
        # entry now guards the trunk half only, so its counts are lower than the
        # pre-split capture by exactly the branch-borne ones it used to include.
        {"attachments": frozenset({Attachment.CAULIFLOROUS}), "cauline_min_age": 6,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.0, end=0.5)}),
    "ramiflorous_raceme": (
        InflorescenceKind.RACEME,
        # The other half, so both sides of the split are guarded rather than
        # only the one that already had a fixture.
        {"attachments": frozenset({Attachment.RAMIFLOROUS}), "cauline_min_age": 6,
         "_max_branch_order": 1,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.0, end=0.5)}),
    "leaf_opposed_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.LEAF_OPPOSED}),
         "placement": Placement(on_trunk=True, on_branches=True, start=0.3, end=1.0)}),
    "basal_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.BASAL})}),
    "adventitious_raceme": (
        InflorescenceKind.RACEME,
        {"attachments": frozenset({Attachment.ADVENTITIOUS}), "cauline_min_age": 6,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.0, end=0.6)}),
    "terminal_and_axillary_raceme": (
        InflorescenceKind.RACEME,
        # The point of the set: the apex flowers AND the nodes below bear their
        # own, which was mutually exclusive while this was a single value.
        {"attachments": frozenset({Attachment.TERMINAL, Attachment.AXILLARY}),
         "placement": Placement(on_trunk=True, on_branches=True, start=0.3, end=1.0)}),
    "axillary_cluster_raceme": (
        InflorescenceKind.RACEME,
        # The count half of what "axillary cluster" names -- guards that
        # `attachment_per_node` genuinely multiplies a position.
        {"attachments": frozenset({Attachment.AXILLARY}), "attachment_per_node": 3,
         "placement": Placement(on_trunk=True, on_branches=True, start=0.3, end=1.0)}),
}


COMPOUND = {
    # key -> (kind, branching_depth)
    #
    # `branching_depth` is one control across every kind that has a compound
    # form, so what these guard is that turning it on produces the *compound*
    # plant and, just as importantly, that leaving it at 0 keeps every simple
    # kind byte-identical -- which is the half a structural check cannot see.
    #
    # These run on the stems level alone, unlike the simple kinds. A
    # compound inflorescence *is* branching, so growing one on a plant that
    # also branches 27 ways multiplies the fixture by 27 for no extra
    # information -- measured, a compound umbel on the branched profile is 527
    # axes and 1960 flowers, and a triple umbel does not finish in two minutes.
    # Isolating it is also what the fixture says it does: vary only the thing
    # the key names.
    "panicle": (InflorescenceKind.RACEME, 1),
    "compound_umbel": (InflorescenceKind.UMBEL, 1),
    "triple_umbel": (InflorescenceKind.UMBEL, 2),
    "compound_flat_umbel": (InflorescenceKind.FLAT_UMBEL, 1),
    "compound_corymb": (InflorescenceKind.CORYMB, 1),
    "compound_spike": (InflorescenceKind.SPIKE, 1),
}


def _profile_for(key: str):
    """One base profile, varying only what the key names, so any difference in
    the digest is attributable to that and nothing else."""
    profile = generic_perennial()
    if key in EXTRA:
        kind, fields = EXTRA[key]
        profile.inflorescence.kind = kind
        # Branching off by default, so an entry isolates the one thing it names.
        # `_max_branch_order` (branch levels kept below the stems) is the
        # deliberate exception: RAMIFLOROUS breaks
        # from BRANCH wood by definition, so with no branches its fixture builds
        # a plant with zero flowers and guards nothing at all -- the exact
        # "fixture silently guarding less than it looks" failure `all_keys`
        # exists to prevent.
        profile.levels = profile.levels[:1 + fields.get("_max_branch_order", 0)]
        for name, value in fields.items():
            if name.startswith("_"):
                continue
            setattr(profile.inflorescence, name, value)
        return profile
    kind, depth = COMPOUND.get(key, (key, None))
    profile.inflorescence.kind = kind
    if depth is not None:
        profile.levels = profile.levels[:1]
    stamp = _CYME_STAMPS.get(kind)
    if stamp is not None:
        (profile.inflorescence.laterals_per_node,
         profile.inflorescence.relay_rotation_deg,
         profile.inflorescence.branching_depth,
         profile.inflorescence.branch_angle_deg) = stamp
    if depth is not None:
        profile.inflorescence.branching_depth = depth
    return profile


def digest(plant) -> dict:
    """A canonical, readable structural summary.

    Covers exactly what the veg rewrite could plausibly move: which axes exist
    and at what order, where each one started flowering, whether its apex
    became a flower, and where every organ ended up. Positions are rounded --
    the point is to catch a structural change, not float drift.
    """
    axes = []
    for axis in sorted(plant.axes.values(), key=lambda a: a.id):
        axes.append({
            "id": axis.id,
            "order": axis.order,
            "parent": axis.parent_axis_id,
            "phytomers": len(axis.phytomers),
            "flowering_from": axis.flowering_from,
            "apex_state": axis.apex.state if axis.apex is not None else None,
            "terminal_organ": axis.terminal_organ.kind if axis.terminal_organ else None,
        })

    organs = []
    for organ in sorted(plant.all_organs(), key=lambda o: o.id):
        organs.append({
            "id": organ.id,
            "kind": organ.kind,
            "birth_step": organ.birth_step,
            "pos": [round(float(c), _PLACES) for c in organ.position],
            # Facing, not just placement. Added after this fixture passed
            # unchanged through a rewrite of what Flower Pitch means -- every
            # flower on all nine kinds was re-aimed and the digest could not
            # see it, because a stalk places an organ and pitch only turns it.
            # The same blind spot has cost a full debug cycle twice elsewhere
            # in this project; a fixture that shares it is worth less than it
            # looks.
            "dir": [round(float(c), _PLACES) for c in organ.direction],
            "normal": [round(float(c), _PLACES) for c in organ.normal],
            "pedicel_length": round(float(organ.pedicel_length), _PLACES),
            "scale": round(float(organ.scale), _PLACES),
        })

    lo, hi = plant.bounds() if plant.phytomer_count else ([0] * 3, [0] * 3)
    return {
        "phytomers": plant.phytomer_count,
        "axis_count": len(plant.axes),
        "leaves": len(plant.organs_of_kind(OrganKind.LEAF)),
        "flowers": len(plant.organs_of_kind(OrganKind.FLOWER)),
        "bounds_lo": [round(float(c), _PLACES) for c in lo],
        "bounds_hi": [round(float(c), _PLACES) for c in hi],
        "axes": axes,
        "organs": organs,
    }


def all_keys() -> tuple:
    """Every fixture entry: the bare kinds, then the compound variants.

    One function so `capture` and `verify` cannot disagree about the set. They
    did: `verify` looped over `KINDS` while `build_all` had grown the compound
    entries, so six of them were written into the golden file every capture and
    checked by nothing -- a fixture silently guarding less than it looked."""
    return tuple(KINDS) + tuple(COMPOUND) + tuple(EXTRA)


def build_all() -> dict:
    return {key: digest(build(_profile_for(key), seed=SEED, age=AGE))
            for key in all_keys()}


def _first_difference(want: dict, got: dict, path: str = "") -> str | None:
    """The first place two digests diverge, as a readable path. Depth-first so
    a count mismatch is reported before the hundred organ positions that follow
    from it."""
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
    GOLDEN_PATH.write_text(json.dumps(build_all(), indent=1, sort_keys=True))
    print(f"captured {len(all_keys())} entries -> {GOLDEN_PATH}")


def verify() -> int:
    if not GOLDEN_PATH.exists():
        print(f"no golden file at {GOLDEN_PATH}; run with --capture first")
        return 1
    want = json.loads(GOLDEN_PATH.read_text())
    got = build_all()

    print("\ninflorescence golden comparison")
    print("-" * 64)
    failed = 0
    for kind in all_keys():
        if kind not in want:
            print(f"  [new ] {kind} - not in the golden file")
            continue
        diff = _first_difference(want[kind], got[kind])
        if diff is None:
            d = got[kind]
            print(f"  [PASS] {kind:<20} {d['phytomers']:>4} nodes  "
                  f"{d['axis_count']:>3} axes  {d['flowers']:>3} flowers")
        else:
            failed += 1
            print(f"  [FAIL] {kind:<20} first difference at {diff}")
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
