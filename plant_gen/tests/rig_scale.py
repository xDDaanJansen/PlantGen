"""Tests that Build Rig's fast lookups give what the slow ones gave, headless.

A plant's rig work must not be organs x vertices: Laburnum has 36,000 organs on
19 million vertices, and one mask per organ never finished in 15 minutes. Each
fast lookup here is held to the plain version, exactly.

1. group_indices matches one nonzero(ids == id) per id, order included.
2. With `where`, it matches the masked version, and an empty pick is empty.

    python -m plant_gen.tests.rig_scale
"""

from __future__ import annotations

import numpy as np

from ..geometry import rig_layout as rl


def test_group_indices_matches_masks():
    ids = np.random.default_rng(3).integers(-1, 40, 5000)
    groups = rl.group_indices(ids)
    assert list(groups) == [int(v) for v in np.unique(ids)]
    for key, idx in groups.items():
        assert np.array_equal(idx, np.nonzero(ids == key)[0])


def test_group_indices_where():
    rng = np.random.default_rng(4)
    ids, where = rng.integers(0, 25, 4000), rng.random(4000) < 0.3
    groups = rl.group_indices(ids, where)
    assert list(groups) == [int(v) for v in np.unique(ids[where])]
    for key, idx in groups.items():
        assert np.array_equal(idx, np.nonzero(where & (ids == key))[0])
    assert rl.group_indices(ids, np.zeros(len(ids), dtype=bool)) == {}


def _plain_nearest(point, chains):
    """The original walk: every internode of every axis, first nearest wins."""
    best = (None, 0, None, np.inf)
    for aid, chain in chains.items():
        pts = chain.points
        for k in range(len(pts) - 1):
            d = pts[k + 1] - pts[k]
            l2 = float(d @ d)
            t = 0.0 if l2 < 1e-24 else min(max(float((point - pts[k]) @ d) / l2, 0.0), 1.0)
            foot = pts[k] + t * d
            dist = float(np.linalg.norm(point - foot))
            if dist < best[3]:
                best = (aid, k, foot, dist)
    return best[:3]


class _Chain:
    def __init__(self, points):
        self.points = points


def test_nearest_on_axes_matches_the_walk():
    rng = np.random.default_rng(5)
    chains = {}
    for aid in range(12):
        steps = rng.normal(0.0, 0.05, (9, 3))
        steps[3] = 0.0                                  # a zero-length internode
        chains[aid * 3] = _Chain(np.cumsum(np.vstack((rng.random((1, 3)), steps)), axis=0))
    segments = rl.axis_segments(chains)
    queries = [rng.random(3) for _ in range(300)]
    queries += [c.points[k] for c in chains.values() for k in range(len(c.points))]  # on a node
    for q in queries:
        aid, k, foot = rl.nearest_on_axes(q, segments)
        plain = _plain_nearest(q, chains)
        assert (aid, k) == plain[:2]
        assert np.array_equal(foot, plain[2])
    assert rl.nearest_on_axes(np.zeros(3), rl.axis_segments({})) == (None, 0, None)


def _plain_assign(co, axes, stalks):
    """The original: every vertex against every segment."""
    aa, ab, akey, arad = rl._segment_arrays(axes)
    sa, sb, skey, _ = rl._segment_arrays([(oid, pts, None) for oid, pts in stalks])
    d = rl._segments_distance(co, aa, ab)
    j = np.argmin(d, axis=1)
    dmin = d[np.arange(len(co)), j]
    ds = rl._segments_distance(co, sa, sb)
    k = np.argmin(ds, axis=1)
    dk = ds[np.arange(len(co)), k]
    return akey[j], np.where(dk < dmin - arad[j], skey[k], -1), skey[k], dk


def test_assign_stem_matches_every_vertex_against_every_segment():
    rng = np.random.default_rng(6)
    axes, stalks = [], []
    for aid in range(30):
        pts = np.cumsum(np.vstack((rng.random((1, 3)), rng.normal(0.0, 0.06, (7, 3)))), axis=0)
        axes.append((aid, pts, rng.random(7) * 0.02))
        for s in range(2):
            foot = pts[rng.integers(0, 8)]
            stalks.append((1000 + 2 * aid + s, np.vstack((foot, foot + rng.normal(0.0, 0.05, 3)))))
    # Vertices the way a mesh has them: clustered on the lines, some far off.
    around = np.vstack([p[rng.integers(0, 8, 200)] for _a, p, _r in axes])
    far_off = rng.random((400, 3)) * 2.0 - 0.5
    co = np.vstack((around + rng.normal(0.0, 0.015, around.shape), far_off))
    got = rl.assign_stem(co, axes, stalks)
    want = _plain_assign(co, axes, stalks)
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(g, w)


def test_assign_stem_on_a_flat_set_of_points():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    co = np.column_stack((np.linspace(0.0, 1.0, 50), np.zeros(50), np.zeros(50)))
    axis_of, owner, near, near_d = rl.assign_stem(co, [(7, pts, [0.01])], [])
    assert (axis_of == 7).all() and (owner == -1).all() and (near == -1).all()
    assert np.isinf(near_d).all()


def test_near_points_matches_every_vertex_against_every_target():
    rng = np.random.default_rng(7)
    lines = np.cumsum(rng.normal(0.0, 0.02, (600, 3)), axis=0)
    co = np.vstack((lines + rng.normal(0.0, 0.01, lines.shape), rng.random((300, 3)) * 3.0))
    targets = co[rng.choice(len(co), 80, replace=False)] + rng.normal(0.0, 0.001, (80, 3))
    reach = rng.random(len(co)) * 0.06
    plain = np.linalg.norm(co[:, None, :] - targets[None, :, :], axis=-1).min(axis=1) <= reach
    assert np.array_equal(rl.near_points(co, targets, reach), plain)


def _plain_vote(labels, edges, iterations=4):
    """The original: every vertex's ring counted, every pass."""
    labels = np.asarray(labels).copy()
    ring = [[] for _ in labels]
    for x, y in edges:
        ring[x].append(y)
        ring[y].append(x)
    for _ in range(iterations):
        fresh = labels.copy()
        changed = 0
        for v, near in enumerate(ring):
            if not near:
                continue
            values, counts = np.unique(labels[near], return_counts=True)
            k = int(np.argmax(counts))
            if values[k] != labels[v] and counts[k] * 2 > len(near):
                fresh[v] = values[k]
                changed += 1
        labels = fresh
        if not changed:
            break
    return labels


def test_vote_labels_matches_counting_every_ring():
    rng = np.random.default_rng(8)
    for _ in range(20):
        n = 300
        edges = np.unique(np.sort(rng.integers(0, n, (900, 2)), axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        labels = rng.integers(-3, 3, n)
        assert np.array_equal(rl.vote_labels(labels, edges), _plain_vote(labels, edges))


def _organ(oid, kind):
    return rl.Organ(oid, kind, np.zeros(3), np.array([0.0, 0.0, 0.01]))


def test_flowers_sharing_an_axis_ride_it():
    organs = [_organ(1, "flower"), _organ(2, "flower"), _organ(3, "flower"),  # a raceme
              _organ(4, "flower"),                                            # a lone head
              _organ(5, "leaf"), _organ(6, "leaf"),                           # leaves keep bones
              _organ(7, "flower")]                                            # no axis found
    axis_of = {1: 10, 2: 10, 3: 10, 4: 11, 5: 10, 6: 10}
    assert rl.flowers_on_shared_axes(organs, axis_of) == {1, 2, 3}


def test_leaves_ride_only_past_the_bone_limit():
    leaves = [_organ(i, "leaf") for i in range(9)]
    axis_of = {i: 0 for i in range(8)}                    # a lone leaf has no axis mates
    axis_of[8] = 1
    assert rl.organs_riding(leaves, axis_of, 100) == set()
    over = rl.MAX_BONES - 8
    assert rl.organs_riding(leaves, axis_of, over) == set(range(8))


def _plain_components(n, edges):
    """The original union-find, named by each piece's lowest vertex."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(v) for v in range(n)])


def test_components_matches_union_find():
    rng = np.random.default_rng(9)
    for pairs in (0, 40, 200, 900):
        edges = rng.integers(0, 400, (pairs, 2)).reshape(-1, 2)
        assert np.array_equal(rl.components(400, edges), _plain_components(400, edges))
    chain = np.column_stack((np.arange(999, 0, -1), np.arange(998, -1, -1)))  # a long line
    assert (rl.components(1000, chain) == 0).all()


def test_nearest_many_matches_one_at_a_time():
    rng = np.random.default_rng(10)
    chains = {i: _Chain(np.cumsum(rng.normal(0.0, 0.1, (10, 3)), axis=0)) for i in range(20)}
    segments = rl.axis_segments(chains)
    points = rng.normal(0.0, 0.4, (1500, 3))
    for point, got in zip(points, rl.nearest_many_on_axes(points, segments), strict=True):
        want = rl.nearest_on_axes(point, segments)
        assert got[:2] == want[:2] and np.array_equal(got[2], want[2])


def _plain_seams(co, edges, weights, reach, threshold=1.0, iterations=30, max_influences=4):
    """The original dict walk."""
    def l1(a, b):
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in a.keys() | b.keys())

    seam = {int(v) for a, b in edges if l1(weights[a], weights[b]) > threshold for v in (a, b)}
    if not seam:
        return
    targets = co[np.array(sorted(seam))]
    zone = np.linalg.norm(co[:, None] - targets[None], axis=-1).min(axis=1) <= reach
    near = {}
    for a, b in edges:
        a, b = int(a), int(b)
        if zone[a] or zone[b]:
            near.setdefault(a, []).append(b)
            near.setdefault(b, []).append(a)
    moving = [v for v in np.nonzero(zone)[0].tolist() if v in near]
    for _ in range(iterations):
        fresh = {}
        for v in moving:
            acc = dict(weights[v])
            for u in near[v]:
                for k, w in weights[u].items():
                    acc[k] = acc.get(k, 0.0) + w
            scale = 1.0 / (len(near[v]) + 1)
            fresh[v] = {k: w * scale for k, w in acc.items()}
        for v, w in fresh.items():
            weights[v] = w
    for v in moving:
        top = sorted(weights[v].items(), key=lambda kv: -kv[1])[:max_influences]
        total = sum(w for _, w in top) or 1.0
        weights[v] = {k: w / total for k, w in top if w > 1e-4}


def test_smooth_seams_matches_the_dict_walk():
    rng = np.random.default_rng(11)
    side = 40
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    co = np.column_stack((xs.ravel() * 0.01, ys.ravel() * 0.01, np.zeros(side * side)))
    idx = np.arange(side * side).reshape(side, side)
    edges = np.vstack((np.column_stack((idx[:, :-1].ravel(), idx[:, 1:].ravel())),
                       np.column_stack((idx[:-1].ravel(), idx[1:].ravel()))))
    weights = [{"a": 1.0} if x < 20 else ({"b": 0.6, "c": 0.4} if y < 25 else {"d": 1.0})
               for y in range(side) for x in range(side)]
    reach = np.full(len(co), 0.035)
    want = [dict(d) for d in weights]
    _plain_seams(co, edges, want, reach)
    got = [dict(d) for d in weights]
    moved = rl.smooth_seams(co, edges, got, reach)
    assert moved > 0
    for g, w in zip(got, want, strict=True):
        assert g.keys() == w.keys()
        assert all(abs(g[k] - w[k]) < 1e-12 for k in g)
    calm = [{"a": 1.0} for _ in range(len(co))]
    assert rl.smooth_seams(co, edges, calm, reach) == 0
    _ = rng


def _run():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
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
