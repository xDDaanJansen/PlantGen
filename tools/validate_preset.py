"""Validate a PlantGen species preset headlessly -- no Blender needed.

Usage:  python tools/validate_preset.py <preset.json>

`plant_gen.growth` and `.tests` are bpy-free, so a preset can be checked
without opening Blender. This does three things and exits non-zero on any
failure, so it works as a CI-style gate and as the plant-authoring skill's
verification step:

  1. round-trip -- the loaded profile must re-serialize to the same data;
  2. an age sweep (0.0 / 0.5 / 1.0) printing phytomers / axes / height / leaves
     / flowers, so the plant's size and structure are visible at a glance;
  3. the 14 structural checks against the mature plant.

Structural soundness is not botanical correctness -- a preset can pass every
check and still not look like its target. Grow it in Blender for that. But a
failure here is a real defect (an axis that never terminates, a kink, a
non-monotonic taper, a dormant bud with no recorded reason, ...).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip())  # the usage line
        return 2
    preset = Path(sys.argv[1])
    if not preset.is_file():
        print(f"no such preset file: {preset}")
        return 2

    from plant_gen.growth import serialize
    from plant_gen.growth.development import build
    from plant_gen.growth.model import OrganKind
    from plant_gen.tests import structural

    try:
        loaded = serialize.load_json(preset)
    except serialize.SpeciesFileError as e:
        print(f"not a valid species preset: {e}")
        return 1
    profile = loaded.profile

    # 1. round-trip
    again = serialize.from_json(
        serialize.to_json(loaded.name, profile, loaded.description)).profile
    if serialize.profile_to_dict(again) != serialize.profile_to_dict(profile):
        print("FAIL: preset does not round-trip through serialize")
        return 1

    # 2. age sweep
    print(f"preset: {loaded.name or preset.stem}")
    print(f"  {'age':>4} {'phytomers':>10} {'axes':>5} {'height':>8} "
          f"{'leaves':>7} {'flowers':>8}")
    for age in (0.0, 0.5, 1.0):
        p = build(profile, seed=0, age=age)
        if p.phytomer_count:
            lo, hi = p.bounds()
            h = hi[2] - lo[2]
        else:
            h = 0.0
        print(f"  {age:>4.1f} {p.phytomer_count:>10} {len(p.axes):>5} "
              f"{h:>7.3f}m {len(p.organs_of_kind(OrganKind.LEAF)):>7} "
              f"{len(p.organs_of_kind(OrganKind.FLOWER)):>8}")

    # 3. structural checks
    exp = profile.expectations()

    def grow_fn(age: float):
        return build(profile, seed=0, age=age)

    results = structural.run_checks(grow_fn(1.0), exp, grow_fn=grow_fn, profile=profile)
    ok = structural.report(results, f"structural checks - {loaded.name or preset.stem}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
