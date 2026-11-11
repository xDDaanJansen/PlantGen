"""Tests for a cyme's bloom order, headless, against its closed form.

A cyme opens from its terminal flower outward: each flower's place on the
bloom timeline (`Organ.bloom_start`, 0 opens first, 1 last) is its generation
over the cyme's Branching Depth -- the terminal flower 0, each relay one step
later -- for Middle first and Top first, and the reverse for Outside first
and Bottom first. The order is display only: the grown plant is the same
whichever it is.

    python -m plant_gen.tests.bloom_order
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..growth import development as dev
from ..growth import species
from ..growth.model import OrganKind
from ..growth.morphology import BloomOrder, InflorescenceKind

DEPTH = 3


def _cyme(order: str):
    base = species.profile("ANEMONE")
    return replace(base, inflorescence=replace(
        base.inflorescence, kind=InflorescenceKind.DICHASIUM, laterals_per_node=2,
        branching_depth=DEPTH, bloom_order=order))


def _cyme_flowers(plant) -> dict:
    """axis id -> (generation, terminal flower) for every flower-capped axis."""
    return {aid: (a.structural_depth, a.terminal_organ) for aid, a in plant.axes.items()
            if a.terminal_organ is not None and a.terminal_organ.kind == OrganKind.FLOWER}


def test_middle_first_opens_the_terminal_flower_first():
    for order in (BloomOrder.CENTRIFUGAL, BloomOrder.BASIPETAL):
        flowers = _cyme_flowers(dev.build(_cyme(order), seed=3))
        assert flowers
        for aid, (depth, organ) in flowers.items():
            assert organ.bloom_start == depth / DEPTH, (order, aid, depth, organ.bloom_start)
        assert {d for d, _o in flowers.values()} == set(range(DEPTH + 1))


def test_outside_first_reverses_it():
    for order in (BloomOrder.CENTRIPETAL, BloomOrder.ACROPETAL):
        flowers = _cyme_flowers(dev.build(_cyme(order), seed=3))
        for aid, (depth, organ) in flowers.items():
            assert organ.bloom_start == 1.0 - depth / DEPTH, (order, aid, organ.bloom_start)


def test_a_relay_opens_after_the_flower_it_relays():
    plant = dev.build(_cyme(BloomOrder.CENTRIFUGAL), seed=3)
    checked = 0
    for a in plant.axes.values():
        parent = plant.axes.get(a.parent_axis_id)
        # A relay (depth 1 on), not a side branch beginning a cyme of its own.
        if a.structural_depth < 1 or a.terminal_organ is None or parent is None:
            continue
        if parent.terminal_organ is None:
            continue
        assert a.terminal_organ.bloom_start > parent.terminal_organ.bloom_start
        checked += 1
    assert checked > 0


def test_the_order_is_display_only():
    a = dev.build(_cyme(BloomOrder.CENTRIFUGAL), seed=3)
    b = dev.build(_cyme(BloomOrder.CENTRIPETAL), seed=3)
    assert a.axes.keys() == b.axes.keys()
    for aid in a.axes:
        pa, pb = a.axes[aid].phytomers, b.axes[aid].phytomers
        assert len(pa) == len(pb)
        assert all(np.array_equal(x.tip_position, y.tip_position)
                   for x, y in zip(pa, pb, strict=True))


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
