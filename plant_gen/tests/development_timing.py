"""Development timing: how each internode and leaf
grows in after it forms -- the Internode Elongation modes.

    python -m plant_gen.tests.development_timing
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..growth.curves import Curve, default_elongation, elongation_mode
from ..growth.development import build, grow
from ..growth.morphology import generic_perennial

SEED = 0


def _with_mode(profile, mode):
    curve, steps = elongation_mode(mode)
    return replace(profile, elongation=curve, elongation_steps=steps)


def _stretching(plant) -> int:
    """The most internodes on one stem still short of their full length."""
    most = 0
    for a in plant.axes.values():
        if a.parent_axis_id < 0:
            most = max(most, sum(
                1 for ph in a.phytomers
                if np.linalg.norm(ph.tip_position - ph.base_position) < 0.999 * ph.final_length))
    return most


def test_overlapping_is_the_old_default():
    base = generic_perennial()
    curve, steps = elongation_mode("OVERLAPPING")
    assert steps == base.elongation_steps
    assert curve.points == default_elongation().points
    assert list(curve.samples) == list(base.elongation.samples)


def test_sequential_stretches_one_internode_at_a_time():
    base = generic_perennial()
    counts = {m: [_stretching(grow(_with_mode(base, m), seed=SEED, age=a)) for a in (0.35, 0.55)]
              for m in ("SEQUENTIAL", "OVERLAPPING", "DELAYED")}
    print(f"         most internodes stretching on a stem at age 0.35 / 0.55: {counts}")
    assert counts["SEQUENTIAL"] == [1, 1]
    assert max(counts["OVERLAPPING"]) > 1 and max(counts["DELAYED"]) > 1


def test_modes_reach_the_same_grown_plant():
    # Only the timing differs: fully grown, every internode is at its length.
    base = generic_perennial()
    for mode in ("SEQUENTIAL", "DELAYED"):
        p = grow(_with_mode(base, mode), seed=SEED)
        assert _stretching(p) == 0, mode


def test_a_long_head_does_not_stretch_the_season():
    # A flowering region is laid down whole (`_preform_region`), so its flower
    # count is not clock time: past the steps a region needs to open and
    # mature (2 + the elongation steps), more flowers must not lengthen the
    # season -- an Alopecurus spike of 200 spikelets once made it 214 steps and
    # its culm stood full grown 6 days in. The culm must still be growing at
    # a quarter of the season, and the fully grown head carries every flower.
    from ..growth import species
    from ..growth.model import OrganKind
    fox = species.meadow_foxtail()
    long = replace(fox, inflorescence=replace(fox.inflorescence, flower_count=200))
    short = replace(fox, inflorescence=replace(fox.inflorescence, flower_count=20))
    assert long.total_steps == short.total_steps, (long.total_steps, short.total_steps)
    early = grow(long, seed=SEED, age=0.25)
    culm = early.axes[early.basal_axis_ids[0]]
    assert len(culm.phytomers) < long.levels[0].steps, len(culm.phytomers)
    grown = grow(long, seed=SEED)
    flowers = sum(1 for a in grown.axes.values() for ph in a.phytomers for o in ph.organs
                  if o.kind == OrganKind.FLOWER)
    assert flowers == 200 * len(grown.basal_axis_ids), flowers


def test_stems_thicken_as_they_grow():
    # Stem Thickening: a young stem is a slimmer version of its final taper,
    # thickening as it lengthens to exactly its stated width once grown; a flat
    # curve is the old full-width-from-the-start.
    base = generic_perennial()
    flat = replace(base, stem_thickening=Curve.constant(1.0))
    radii = [build(base, seed=SEED, day=float(d)).axes[0].phytomers[0].radius
             for d in (110, 130, 150, 175, 260)]
    stated = build(flat, seed=SEED, day=130.0).axes[0].phytomers[0].radius
    print(f"         stem base radius over Days 110..260: "
          f"{[round(r * 1000, 2) for r in radii]} mm (stated {stated * 1000:.2f})")
    assert all(a < b for a, b in zip(radii, radii[1:], strict=False))
    assert radii[0] < 0.5 * stated and radii[-1] == stated


def _run() -> int:
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
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
