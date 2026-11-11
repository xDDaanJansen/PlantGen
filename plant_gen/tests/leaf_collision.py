"""Tests for the leaf collision resolve (`geometry/leaf_collision.py`), each
against a closed form.

    python -m plant_gen.tests.leaf_collision
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry.leaf_collision import (
    Leaf,
    Shape,
    Stems,
    _frame,
    resolve,
    segments_hit_triangles,
    simplify,
)

# A 0.02 x 0.1 blade: two triangles, base at the origin, length +Y, face +Z.
W, L = 0.02, 0.1
BLADE = Shape([(-W / 2, 0, 0), (W / 2, 0, 0), (W / 2, L, 0), (-W / 2, L, 0)],
              [(0, 1, 2), (0, 2, 3)])
X, Y, Z = np.eye(3)


def _leaf(**kw):
    base = dict(shape=0, base=np.zeros(3), direction=X, normal=Z, scale=1.0,
                pivot=np.zeros(3), movable=True, max_turn=math.radians(30), prune=True)
    base.update(kw)
    return Leaf(**base)


def _vertical_stem(x, y=0.0, r=0.005):
    return Stems(np.array([[x, y, -1.0]]), np.array([[x, y, 1.0]]), np.array([r]))


NO_STEMS = Stems(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))


def test_segment_triangle_hit_point():
    hit, pts = segments_hit_triangles([(0.2, 0.3, -1)], [(0.2, 0.3, 1)],
                                      [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]])
    assert hit[0, 0] and np.allclose(pts[0, 0], (0.2, 0.3, 0.0))
    hit, _ = segments_hit_triangles([(0.8, 0.8, -1)], [(0.8, 0.8, 1)],
                                    [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]])
    assert not hit[0, 0]                      # outside the triangle (u + v > 1)
    hit, _ = segments_hit_triangles([(0.2, 0.3, 0.5)], [(0.2, 0.3, 1)],
                                    [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]])
    assert not hit[0, 0]                      # segment ends before the plane


def test_clear_leaf_is_untouched():
    out = resolve([_leaf()], [BLADE], _vertical_stem(0.5))
    assert not out.clashed[0] and not out.pruned[0] and out.angle[0] == 0.0
    assert np.array_equal(out.turn[0], np.eye(3))


def test_stem_through_blade_turns_sideways_by_the_first_clear_step():
    # The stem crosses the blade's plane 0.05 out along it; it leaves the blade
    # once 0.05 sin(t) > W/2, t > asin(0.2) = 11.54 deg -> the first 5 deg step
    # past that is 15 deg, sideways (about the face normal, the first tried).
    out = resolve([_leaf()], [BLADE], _vertical_stem(0.05))
    assert out.clashed[0] and not out.pruned[0] and not out.unresolved[0]
    assert math.isclose(math.degrees(out.angle[0]), 15.0, abs_tol=1e-9)
    d = out.turn[0] @ X
    assert np.allclose(d, (math.cos(math.radians(15)), math.sin(math.radians(15)), 0))


def test_attachment_zone_is_not_a_collision():
    # The leaf grows out of a stem through its pivot: inside the exempt zone.
    out = resolve([_leaf(exempt=0.0075)], [BLADE], _vertical_stem(0.0))
    assert not out.clashed[0]
    out = resolve([_leaf(exempt=0.0)], [BLADE], _vertical_stem(0.0))
    assert out.clashed[0]


def test_unclearable_leaf_is_pruned_or_kept():
    # No turn allowed: prune it, or (not prunable) keep it, flagged.
    out = resolve([_leaf(max_turn=0.0)], [BLADE], _vertical_stem(0.05))
    assert out.pruned[0]
    out = resolve([_leaf(max_turn=0.0, prune=False)], [BLADE], _vertical_stem(0.05))
    assert not out.pruned[0] and out.unresolved[0] and out.angle[0] == 0.0


def test_younger_leaf_yields_to_the_older():
    # A lies along +X; B (younger, order 1) rises through A's blade from
    # below. A keeps its pose; B turns.
    tilt = math.radians(30)
    d_b = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    n_b = np.cross(X, d_b)
    old = _leaf(order=0.0)
    young = _leaf(order=1.0, base=np.array([0.05, -0.03, -0.017]),
                  pivot=np.array([0.05, -0.03, -0.017]), direction=d_b, normal=n_b)
    out = resolve([young, old], [BLADE], NO_STEMS)
    assert out.clashed[0] and not out.clashed[1]
    assert out.angle[1] == 0.0 and out.angle[0] > 0.0 and not out.pruned[0]
    # The turned pose really is clear: re-run with it baked in.
    turned = _leaf(order=1.0, base=young.base, pivot=young.pivot,
                   direction=out.turn[0] @ d_b, normal=out.turn[0] @ n_b)
    again = resolve([turned, old], [BLADE], NO_STEMS)
    assert not again.clashed[0]


def test_fixed_obstacle_never_moves():
    tilt = math.radians(30)
    d_b = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    n_b = np.cross(X, d_b)
    fixed = _leaf(movable=False, order=5.0)
    young = _leaf(order=0.0, base=np.array([0.05, -0.03, -0.017]),
                  pivot=np.array([0.05, -0.03, -0.017]), direction=d_b, normal=n_b)
    out = resolve([fixed, young], [BLADE], NO_STEMS)
    # even though the fixed leaf is "younger", the movable one yields
    assert out.angle[0] == 0.0 and not out.clashed[0]
    assert out.clashed[1] and (out.angle[1] > 0.0 or out.pruned[1])


def _wall(bottom, extra=None):
    """Vertical stems side by side at x = 0.05 (y -0.12..0.12, 4 mm apart, 3 mm
    radius) from `bottom` up 1 m: a blade along +X can only pass under them."""
    ys = np.linspace(-0.12, 0.12, 61)
    a = np.array([(0.05, y, bottom) for y in ys])
    b = a + np.array([0.0, 0.0, 1.0])
    r = np.full(len(ys), 0.003)
    if extra is not None:
        a, b, r = np.vstack([a, extra[0]]), np.vstack([b, extra[1]]), np.append(r, extra[2])
    return Stems(a, b, r)


def test_no_turn_below_the_ground():
    # The leaf sits 2 mm above the wall's foot; tipping it down 5 deg passes
    # under the wall (the tip then 0.1 sin 5 - 0.002 = 6.7 mm lower). With
    # the wall's foot the stems' lowest point, that is below the ground: no
    # clear pose within 30 deg, pruned. With a stem reaching lower elsewhere,
    # the same 5 deg turn down is allowed.
    leaf = _leaf(base=np.array([0.0, 0.0, 0.502]), pivot=np.array([0.0, 0.0, 0.502]))
    out = resolve([leaf], [BLADE], _wall(0.5))
    assert out.clashed[0] and out.pruned[0]
    kept = resolve([_leaf(base=leaf.base, pivot=leaf.pivot, prune=False)], [BLADE], _wall(0.5))
    low = 0.502 + min(0.0, float((kept.turn[0] @ (L * X))[2]))
    assert kept.unresolved[0] and low >= 0.5 - 1e-9
    far = (np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]), 0.003)
    out = resolve([leaf], [BLADE], _wall(0.5, far))
    assert not out.pruned[0] and math.isclose(math.degrees(out.angle[0]), 5.0, abs_tol=1e-9)
    assert (out.turn[0] @ X)[2] < 0.0                 # tipped down


def test_upright_obstacle_takes_z_along_its_direction():
    # A flower (upright asset: +Z along `direction`, +Y along `normal`) posed
    # with direction +Z, normal +X lays the blade asset's length (+Y) along +X
    # -- the same place as a flat leaf along +X, so a leaf rising through it
    # yields exactly as it does to that leaf.
    f = _frame(Z, X, upright=True)
    assert np.allclose(f @ Z, Z) and np.allclose(f @ Y, X) and np.isclose(np.linalg.det(f), 1.0)
    tilt = math.radians(30)
    d_b = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    n_b = np.cross(X, d_b)
    flower = _leaf(movable=False, direction=Z, normal=X, upright=True)
    young = _leaf(order=0.0, base=np.array([0.05, -0.03, -0.017]),
                  pivot=np.array([0.05, -0.03, -0.017]), direction=d_b, normal=n_b)
    out = resolve([flower, young], [BLADE], NO_STEMS)
    as_leaf = resolve([_leaf(movable=False), young], [BLADE], NO_STEMS)
    assert out.clashed[1] and out.angle[0] == 0.0
    assert out.angle[1] == as_leaf.angle[1] and out.pruned[1] == as_leaf.pruned[1]


def test_stalk_through_a_stem_counts():
    # A leaf with a stalk passing through a stem: the stalk's points inside it.
    stalk = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.06, 0.0, 0.0]])
    leaf = _leaf(base=np.array([0.06, 0.0, 0.0]), stalk=stalk, stalk_radius=0.001,
                 max_turn=0.0, prune=False)
    out = resolve([leaf], [BLADE], _vertical_stem(0.03))
    assert out.unresolved[0]


def test_a_stalk_is_a_tube():
    # An older leaf's stalk runs under the young blade, 2 mm below its plane
    # and across it, never crossing it: a 3 mm thick stalk touches the blade,
    # a 1 mm one does not.
    stalk = np.array([[0.05, -0.05 + 0.0125 * k, -0.002] for k in range(9)])
    for radius, touches in ((0.003, True), (0.001, False)):
        old = _leaf(order=0.0, base=np.array([0.05, 0.05, -0.002]),
                    pivot=np.array([0.05, -0.05, -0.002]), direction=Y, normal=Z,
                    stalk=stalk, stalk_radius=radius, movable=False)
        young = _leaf(order=1.0, max_turn=0.0, prune=False)
        out = resolve([old, young], [BLADE], NO_STEMS)
        assert out.clashed[1] == touches, (radius, out.clashed)


def test_deterministic():
    leaves = [_leaf(order=float(k), base=np.array([0.0, 0.01 * k, 0.0]),
                    pivot=np.array([0.0, 0.01 * k, 0.0]),
                    direction=np.array([1.0, 0.1 * k, 0.05 * k]))
              for k in range(6)]
    a = resolve(leaves, [BLADE], _vertical_stem(0.05))
    b = resolve(leaves, [BLADE], _vertical_stem(0.05))
    assert np.array_equal(a.angle, b.angle) and np.array_equal(a.pruned, b.pruned)
    assert all(np.array_equal(p, q) for p, q in zip(a.turn, b.turn, strict=True))


def test_simplify_keeps_the_outline():
    # A 20 x 50 grid blade: the proxy keeps its bounding box and is much smaller.
    xs, ys = np.linspace(-0.01, 0.01, 20), np.linspace(0.0, 0.1, 50)
    verts = np.array([(x, y, 0.002 * math.sin(30 * y)) for y in ys for x in xs])
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a = j * len(xs) + i
            tris += [(a, a + 1, a + len(xs) + 1), (a, a + len(xs) + 1, a + len(xs))]
    shape = simplify(verts, tris)
    assert len(shape.tris) < len(tris) / 5 and len(shape.tris) > 20
    lo, hi = shape.verts.min(axis=0), shape.verts.max(axis=0)
    cell = np.array([0.02 / 6, 0.1 / 14, 0.004 / 3])
    assert np.all(lo <= verts.min(axis=0) + cell) and np.all(hi >= verts.max(axis=0) - cell)


# --- plain runner (no pytest needed) -----------------------------------------------

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
