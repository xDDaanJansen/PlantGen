"""Tests for a flower's Size Random Range, headless, against its closed form.

Every flower takes its size times (1 + Random Range x a signed draw seeded on
its own id) -- the flower capping an axis's tip (a cyme's, a solitary head's)
as well as the flowers along a raceme. A tip flower used to be made at the
bare size, so a plant of single heads (Rudbeckia) drew every head the same.

    python -m plant_gen.tests.flower_size
"""

from __future__ import annotations

from dataclasses import replace

from ..growth import development as dev
from ..growth import species
from ..growth.model import OrganKind
from ..growth.morphology import InflorescenceKind

SEED = 3


def _cyme(variation: float):
    base = species.profile("ANEMONE")
    return replace(base, inflorescence=replace(
        base.inflorescence, kind=InflorescenceKind.DICHASIUM, laterals_per_node=2,
        branching_depth=2, flower_size_variation=variation))


def _tip_flowers(profile):
    plant = dev.build(profile, seed=SEED)
    return [a.terminal_organ for a in plant.axes.values()
            if a.terminal_organ is not None and a.terminal_organ.kind == OrganKind.FLOWER]


def test_a_tip_flower_takes_its_random_range():
    profile = _cyme(0.5)
    flowers = _tip_flowers(profile)
    assert len(flowers) > 2
    size = profile.flower_size_of(profile.inflorescence)
    for organ in flowers:
        want = size * (1.0 + 0.5 * dev._signed(SEED, dev._T_FLOWER_SIZE, organ.id))
        assert abs(organ.final_scale - want) < 1e-12, (organ.id, organ.final_scale, want)
    assert len({o.final_scale for o in flowers}) == len(flowers)


def test_no_random_range_keeps_every_tip_flower_its_size():
    profile = _cyme(0.0)
    size = profile.flower_size_of(profile.inflorescence)
    assert {o.final_scale for o in _tip_flowers(profile)} == {size}


# --- plain runner (no pytest needed) -----------------------------------------

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
