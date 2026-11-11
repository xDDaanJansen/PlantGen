"""Tests for the inflorescence's own spiral (`Inflorescence.spiral_deg`,
`development._flower_arrangement`), against closed forms.

On a spike whose leaves are two-ranked (the meadow-foxtail profile, 180 deg):
Spiral 0 follows the leaves' arrangement exactly; Spiral 137.5 turns each
flowering node's flower 137.5 deg from the last, whatever the leaves do; a
ring asked for with Flowers per Whorl (2 or more) still wins.

    python -m plant_gen.tests.flower_spiral
"""

from __future__ import annotations

from dataclasses import replace

from ..growth import species
from ..growth.development import grow
from ..growth.model import OrganKind

SEED = 0


def _profile(**inf):
    p = species.meadow_foxtail()
    return replace(p, inflorescence=replace(p.inflorescence, **inf))


def _flower_nodes(plant):
    """Per stem, each flowering node's flower divergences, node by node."""
    out = []
    for axis in plant.axes.values():
        nodes = [[o.divergence_deg for o in ph.organs if o.kind == OrganKind.FLOWER]
                 for ph in axis.phytomers]
        nodes = [n for n in nodes if n]
        if len(nodes) > 3:
            out.append(nodes)
    return out


def _step(a, b):
    return (b - a) % 360.0


def test_leaves_are_two_ranked():
    la = species.meadow_foxtail().leaf_arrangement
    assert la.spiral_angle_deg == 180.0 and la.whorl == 1, (la.spiral_angle_deg, la.whorl)


def test_zero_follows_the_leaves():
    stems = _flower_nodes(grow(_profile(spiral_deg=0.0), seed=SEED))
    assert stems, "no flowering stems"
    for nodes in stems:
        steps = {round(_step(a[0], b[0]), 6) for a, b in zip(nodes[:-1], nodes[1:], strict=True)}
        assert steps == {180.0}, steps


def test_spiral_turns_each_node():
    stems = _flower_nodes(grow(_profile(spiral_deg=137.5), seed=SEED))
    assert stems, "no flowering stems"
    for nodes in stems:
        assert all(len(n) == 1 for n in nodes), "one flower a node"
        for a, b in zip(nodes[:-1], nodes[1:], strict=True):
            assert abs(_step(a[0], b[0]) - 137.5) < 1e-6, (a, b)


def test_whorl_wins():
    stems = _flower_nodes(grow(_profile(spiral_deg=137.5, whorl_count=3), seed=SEED))
    assert stems, "no flowering stems"
    for nodes in stems:
        assert all(len(n) == 3 for n in nodes), [len(n) for n in nodes][:5]


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
