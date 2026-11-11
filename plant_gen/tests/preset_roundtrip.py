"""Round-trip guard for the species preset format, and the generator for the
bundled species presets.

    python -m plant_gen.tests.preset_roundtrip            # verify
    python -m plant_gen.tests.preset_roundtrip --generate # (re)write presets/

Verify is the guard. For every built-in species:

  (a) `profile_to_dict(p)` must equal `profile_to_dict(round_trip(p))` -- the
      full profile survives JSON -> profile -> JSON losslessly. This is airtight
      for correctness: a dropped field shows up the moment its value differs
      from the dataclass default.
  (b) `digest(build(p))` must equal `digest(build(round_trip(p)))` -- and since
      growth is a pure function of the profile, an identical serialised profile
      is an identical plant. This is the same byte-identical contract the 40
      goldens give, one layer up.

Generate is a separate flag on purpose, exactly like the golden capture: a
verify that silently rewrote the bundled presets would pass forever.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..growth import serialize
from ..growth.curves import Curve
from ..growth.development import build
from ..growth.masks import MaskEntry, MaskStack
from ..growth.morphology import (
    Attachment,
    Inflorescence,
    InflorescenceKind,
    MorphologyProfile,
    Topology,
    Vegetativeness,
)
from ..growth.species import SPECIES
from .inflorescence_golden import digest

PRESETS_DIR = Path(__file__).parent.parent / "presets"

SEED = 0
AGE = 1.0

# Display name + one-line description per species key, transcribed once from the
# Species dropdown (blender/properties.py). These become the JSON's own home for
# name/description as the enum is retired (rollout phase 3).
META = {
    "GENERIC": ("Generic perennial",
                "A plain single-stem determinate perennial - the starting "
                "point, not a real species"),
    "MEADOW_FOXTAIL": ("Meadow foxtail",
                       "Alopecurus pratensis - a tufted grass, many unbranched "
                       "culms each ending in a dense spike"),
    "RUDBECKIA": ("Rudbeckia",
                  "Rudbeckia fulgida, orange coneflower - a clump branching "
                  "only near the top, each branch ending in one daisy head"),
    "ANEMONE": ("Japanese anemone",
                "Anemone hupehensis - tall wiry stems forking into an airy "
                "spray with one large flower per branch"),
    "FOXGLOVE": ("Foxglove",
                 "Digitalis purpurea - a flat basal rosette that bolts into "
                 "one tall unbranched spire of tubular flowers"),
    "HAZEL": ("Hazel",
              "Corylus avellana - a leafy twig whose stems bear no flowers at "
              "all, hung instead with catkins in bunches near the tips of its "
              "side shoots. Needs a catkin mesh assigned under Flowers > "
              "Catkins > Shape before the catkins appear"),
    "LABURNUM": ("Laburnum",
                 "Laburnum anagyroides, golden chain - long racemes hanging "
                 "from buds along last year's shoots, while the shoots "
                 "themselves bear no flowers. The worked example of an "
                 "inflorescence borne along a stem rather than at its tip"),
    "LYCOPODIUM": ("Clubmoss",
                   "Lycopodium - a fern-ally whose growing tip forks into two "
                   "equal halves over and over, the one worked example of a "
                   "forking level. Foliage-only, it never flowers; needs "
                   "only a leaf mesh for its microphylls"),
    "ARAUCARIA": ("Araucaria",
                  "Araucaria, monkey-puzzle - a conifer with a whorl of "
                  "exactly four branches at every tier up a bare trunk, the "
                  "one worked example of a whorled branching level. "
                  "Foliage-only, it never flowers; the long internode is what "
                  "spaces the whorls into tiers"),
}


def _round_trip(profile):
    return serialize.profile_from_dict(serialize.profile_to_dict(profile))


def _synthetic_maximal() -> MorphologyProfile:
    """A profile that sets the optional/hidden fields every shipped species
    leaves at their default -- so the decode paths the 9 species never exercise
    (an `X | None` optional carrying a real dataclass, a multi-element
    `attachments` frozenset, a non-empty `topology_override`) are still proven.
    These are precisely the fields "baked full profile" exists to capture, and a
    user-authored preset is the first thing that will set them."""
    return MorphologyProfile(
        veg=Vegetativeness(apex=0.3, lateral_offset=0.7, decline=0.2, threshold=0.1),
        lateral_inflorescence=Inflorescence(
            kind=InflorescenceKind.RACEME, flower_count=5,
            attachments=frozenset({Attachment.AXILLARY})),
        inflorescence=Inflorescence(
            attachments=frozenset({Attachment.TERMINAL, Attachment.AXILLARY}),
            topology_override=Topology.RELAY),
        masks={
            "leaf": MaskStack(
                entries=[
                    MaskEntry(channel="height", lo=0.2, hi=0.8),
                    MaskEntry(channel="age", lo=0.0, hi=1.0, invert=True, op="subtract",
                              curve=Curve.linear(1.0, 0.0)),
                ],
                keep=0.6, seed=3),
            "flower": MaskStack(entries=[MaskEntry(channel="factor")]),
        },
    )


def _check_synthetic() -> int:
    """Round-trip the maximal profile and assert the optional-dataclass and
    frozenset paths reconstruct to the right types and values, not just that the
    dict compares equal."""
    p = _synthetic_maximal()
    rt = _round_trip(p)
    checks = {
        "lossless": serialize.profile_to_dict(p) == serialize.profile_to_dict(rt),
        "veg type": isinstance(rt.veg, Vegetativeness),
        "veg values": rt.veg == p.veg,
        "lateral type": isinstance(rt.lateral_inflorescence, Inflorescence),
        "lateral kind": rt.lateral_inflorescence.kind == InflorescenceKind.RACEME,
        "lateral attachments": rt.lateral_inflorescence.attachments == frozenset(
            {Attachment.AXILLARY}),
        "attachments set": rt.inflorescence.attachments == frozenset(
            {Attachment.TERMINAL, Attachment.AXILLARY}),
        "topology_override": rt.inflorescence.topology_override == Topology.RELAY,
        "masks keys": set(rt.masks) == {"leaf", "flower"},
        "mask stack type": isinstance(rt.masks["leaf"], MaskStack),
        "mask entry type": isinstance(rt.masks["leaf"].entries[0], MaskEntry),
        "mask entry count": len(rt.masks["leaf"].entries) == 2,
        "mask window": (rt.masks["leaf"].entries[0].lo == 0.2
                        and rt.masks["leaf"].entries[0].hi == 0.8),
        "mask keep/seed": rt.masks["leaf"].keep == 0.6 and rt.masks["leaf"].seed == 3,
        "mask curve type": isinstance(rt.masks["leaf"].entries[1].curve, Curve),
        "mask invert": rt.masks["leaf"].entries[1].invert is True,
    }
    bad = [name for name, ok in checks.items() if not ok]
    print(f"  [{'PASS' if not bad else 'FAIL'}] {'SYNTHETIC_MAX':16} "
          f"optional dataclasses + frozenset + override + masks"
          + ("" if not bad else "  <-- " + ", ".join(bad)))
    return len(bad) > 0


def _check_widget_curve() -> bool:
    """A curve drawn in a widget (smooth handles, no authoring line) survives
    a preset: its exact samples grow the plant, and its own points + handle
    types come back to re-seed the widget. Before, it was saved as samples only
    and the widget came back flat on apply. True on failure."""
    samples = [0.0, 0.01, 0.2, 0.7, 0.95, 1.0]           # an Auto curve's table
    curve = Curve(samples, [(0.0, 0.0), (0.4, 0.3), (1.0, 1.0)],
                  ("AUTO", "AUTO_CLAMPED", "VECTOR"))
    back = serialize._decode_curve(serialize._encode_curve(curve))
    ok = (back.samples == curve.samples and back.points == curve.points
          and back.handles == curve.handles)
    # a hand-authored (linear) curve keeps its old, points-only format
    line = Curve.from_points([(0.0, 1.0), (1.0, 0.5)])
    ok = ok and set(serialize._encode_curve(line)) == {"points"}
    print(f"  [{'PASS' if ok else 'FAIL'}] {'WIDGET_CURVE':16} exact samples + points + handles")
    return not ok


RETIRED = {
    "MEADOW_FOXTAIL": "alopecurus_pratensis",
    "FOXGLOVE": "digitalis_purpurea",
    "ANEMONE": "anemone_hupehensis",
    "RUDBECKIA": "rudbeckia_fulgida",
    "ARAUCARIA": "araucaria_araucana",
    "HAZEL": "corylus_avellana",
    "LABURNUM": "laburnum_anagyroides",
    "LYCOPODIUM": "lycopodium_clavatum",
    "GENERIC": "rudbeckia_fulgida",
}
"""Built-in species whose bundled preset was replaced by a hand-authored one
(the file named here) -- not written by `--generate` (user decisions
2026-10-21, ANEMONE 2026-10-25). The species themselves stay: tests build them from code, and the
old Species dropdown stores its pick by position."""


def generate() -> int:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    for key, make in SPECIES.items():
        if key in RETIRED:
            print(f"  skipped {key.lower():22} retired -> {RETIRED[key]}.json")
            continue
        name, description = META.get(key, (key.title(), ""))
        path = PRESETS_DIR / f"{key.lower()}.json"
        serialize.save_json(path, name, make(), description, preset_id=key)
        print(f"  wrote {path.name:22} {name}")
    print(f"\n  {len(SPECIES)} presets written to {PRESETS_DIR}")
    return 0


def verify() -> int:
    failures = 0
    for key, make in SPECIES.items():
        profile = make()
        rt = _round_trip(profile)

        lossless = serialize.profile_to_dict(profile) == serialize.profile_to_dict(rt)
        same_plant = (digest(build(profile, seed=SEED, age=AGE))
                      == digest(build(rt, seed=SEED, age=AGE)))

        ok = lossless and same_plant
        failures += not ok
        flags = "" if ok else (
            "  <-- " + ", ".join(
                w for w, good in (("not lossless", lossless), ("plant differs", same_plant))
                if not good))
        d = digest(build(rt, seed=SEED, age=AGE))
        print(f"  [{'PASS' if ok else 'FAIL'}] {key:16} "
              f"{d['phytomers']:4} nodes {d['axis_count']:4} axes "
              f"{d['flowers']:4} flowers{flags}")

    failures += _check_synthetic()
    failures += _check_widget_curve()

    total = len(SPECIES) + 2
    print(f"\n  {total - failures} of {total} round-trip clean")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true",
                        help="(re)write the bundled presets/ JSON files")
    args = parser.parse_args()
    sys.exit(generate() if args.generate else verify())
