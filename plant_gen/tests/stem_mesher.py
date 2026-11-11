"""Tests for the Built weld (`geometry/stem_mesher.py`), headless.

Locks what the mesher promises, numerically:

1. **Closed and clean**: every mesh -- a lone trunk, a Y at 45 and 80 degrees,
   an opposite pair, a five-branch spiral, a thick whorl, thick forks at 60 and
   35 degrees -- is a closed 2-manifold (every edge on exactly two faces), with
   no degenerate face and one UV per face corner.
2. **Quads**: every face is a quad but the tubes' end caps and a few
   triangles in the welds (where the clean-up collapsed a crowded edge) --
   the transitions ease a hole's border down to the branch's ring in 3-to-1
   quad units.
3. **On the weld surface, unfolded**: every transition vertex lies on the
   junction's rolling-ball surface, and no transition face faces into it.
4. **The branch keeps its own sides**: a welded branch has the side count its
   thickness asks for, however many vertices its hole's border has.
5. **Parent UVs untouched** below a junction.
6. **The fillet leaves the parent tangentially**.
7. **UVs through the weld**: the parent's island and the branch's meet at
   the crease, each face textured by its own tube -- square texels, at one
   texel size over the whole plant (a twig's bark is the trunk's scale).
8. **Weld Range**: the ball is Weld Range x the branch radius x the sine of
   its entry angle; a larger range lifts more of the parent.
9. **Not welded**: a stalk lying along its parent stays a capped tube.
10. **Crowded branches weld on their node**: branches whose welds overlap
    (a branch on its parent's first node, siblings leaving together, a thick
    opposite pair) are welded together where they leave, none moved.
11. **Cross-sections keep their shape** at any side count (corners, ribs).
12. **The thicker side shoot leads**: where a branch is much thicker than the
    stem after its node, the thick path is meshed as the one tube.
13. **Smoothness sets the budget**; **determinism**.
14. **The worker process** a live edit meshes in (`stem_worker`) builds
    exactly the mesh this process does.

    python -m plant_gen.tests.stem_mesher
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..geometry import stem_mesher as sm


def _skeleton(branches):
    """A trunk up z 0..1 (width 0.02 -> 0.01) plus branches (node z, azimuth
    deg, angle from the trunk deg, length, width), in the write-out's layout."""
    verts, edges = [], []
    attrs = {k: [] for k in ("width", "tube", "parent_tube", "flare", "order", "axis_id")}

    def emit(p, w, tube, parent, order):
        verts.append(tuple(float(c) for c in p))
        for k, v in (("width", w), ("tube", tube), ("parent_tube", parent), ("flare", 0.0),
                     ("order", order), ("axis_id", tube)):
            attrs[k].append(v)
        return len(verts) - 1

    prev = None
    for z in np.linspace(0, 1, 11):
        v = emit((0, 0, z), 0.02 - 0.01 * z, 0, -1, 0)
        if prev is not None:
            edges.append((prev, v))
        prev = v
    starts = {0: (np.zeros(3), np.array([0.0, 0.0, 1.0]))}   # tube -> (base, direction)
    for b, spec in enumerate(branches, start=1):
        z, az, ang, length, w = spec[:5]
        on = spec[5] if len(spec) > 5 else 0        # the tube it leaves (0 = trunk)
        a, e = np.radians(az), np.radians(ang)
        d = np.array([np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), np.cos(e)])
        o, od = starts[on]
        base = o + od * z                           # z: distance along that tube
        starts[b] = (base, d)
        prev = emit(base, w, b, on, 1 if on == 0 else 2)
        for k in range(1, 6):
            v = emit(base + d * length * k / 5, w * (1 - 0.1 * k), b, 0, 1)
            edges.append((prev, v))
            prev = v
    return verts, edges, attrs


_FIELDS = {}


def _build(branches, section=None, **kw):
    """Build, keeping each junction's weld field for the surface checks."""
    v, e, a = _skeleton(branches)
    orig = sm._junction

    def keep(par, child, *args):
        j = orig(par, child, *args)
        if isinstance(j, dict):
            _FIELDS[child.tid] = j["field"]
        return j

    sm._junction = keep
    try:
        return sm.build_stems(v, e, a, section=section or sm.SectionSpec(sides=32), **kw)
    finally:
        sm._junction = orig


def _open_edges(out) -> int:
    use = Counter()
    for f in out.face_lists():
        for i in range(len(f)):
            x, y = f[i], f[(i + 1) % len(f)]
            use[(min(x, y), max(x, y))] += 1
    return sum(1 for c in use.values() if c != 2)


CASES = {
    "trunk": [],
    "y45": [(0.5, 0, 45, 0.4, 0.008)],
    "y80_thick": [(0.5, 30, 80, 0.4, 0.013)],
    "opposite": [(0.5, 0, 50, 0.3, 0.007), (0.5, 180, 50, 0.3, 0.007)],
    "spiral5": [(0.2 + 0.12 * i, 137.5 * i, 55, 0.25, 0.006) for i in range(5)],
    "whorl4_thick": [(0.5, 90 * i, 60, 0.3, 0.012) for i in range(4)],
    # a branch nearly as thick as its parent (the pipe model's usual case),
    # steep and shallow
    "fork60": [(0.5, 0, 60, 0.4, 0.0142)],
    "fork35": [(0.5, 0, 35, 0.4, 0.0142)],
    # clusters: a thick branch splitting right after it leaves the trunk, and
    # two thick branches leaving together side by side
    "chain": [(0.5, 0, 35, 0.4, 0.013), (0.02, 60, 40, 0.3, 0.011, 1)],
    "pair": [(0.5, 0, 45, 0.35, 0.012), (0.52, 70, 45, 0.35, 0.012)],
}


def test_closed_manifold_clean_uvs():
    for name, branches in CASES.items():
        out = _build(branches)
        faces = out.face_lists()
        assert _open_edges(out) == 0, name
        assert not any(len(set(f)) < len(f) for f in faces), name
        assert len(out.corner_uv) == sum(len(f) for f in faces), name


def test_quads_but_caps_and_a_few_triangles():
    """Every face is a quad but the tubes' end caps and the few triangles the
    clean-up leaves where it collapsed a crowded edge: under 8 % of each weld."""
    for name, branches in CASES.items():
        out = _build(branches)
        caps = (len(branches) + 1) + (len(branches) + 1 - out.junctions)   # tips + bases
        assert int(np.sum(out.face_sizes > 4)) == caps, name
        assert not np.any(out.face_sizes < 3), name
        for c, j in out.debug["junction"].items():
            if "faces" in j:
                tris = sum(1 for vids, _ in j["faces"] if len(vids) == 3)
                assert tris <= 0.08 * len(j["faces"]), (name, c, tris, len(j["faces"]))


def test_on_the_surface_unfolded():
    """Every transition vertex on its weld surface (|f| under 5 % of the
    ball), and every transition face facing out of it (its normal within 72
    degrees of the surface's, averaged at its corners). The thick 35-degree
    fork may keep a few pinched faces at the very tip of its crotch."""
    for name in ("y45", "y80_thick", "opposite", "spiral5", "fork60", "fork35"):
        out = _build(CASES[name], fillet=0.5)
        folded = 0
        for c, j in out.debug["junction"].items():
            if "faces" not in j:
                continue
            fld = _FIELDS[c]
            ids = np.unique(np.concatenate(j["rings"][1:-1]))
            f, _ = fld(out.positions[ids])
            assert float(np.abs(f).max()) < 0.05 * fld.k, (name, c)
            for vids, _uv in j["faces"]:
                p = out.positions[list(vids)]
                n = np.zeros(3)
                for i in range(len(p)):
                    n += np.cross(p[i], p[(i + 1) % len(p)])
                _, g = fld(p)
                g = (g / np.linalg.norm(g, axis=1, keepdims=True)).mean(axis=0)
                if float(n @ g) < 0.3 * np.linalg.norm(n) * np.linalg.norm(g):
                    folded += 1
        assert folded <= (16 if name == "fork35" else 1), (name, folded)


def test_branch_keeps_its_own_sides():
    """A thick fork's hole border is far longer than its ring: the branch
    still has the sides its thickness asks for, and the border more."""
    out = _build(CASES["fork35"])
    j = out.debug["junction"][1]
    assert len(j["rings"][0]) > len(j["rings"][-1])
    assert out.debug["sides"][1] == len(j["rings"][-1])
    assert out.debug["sides"][1] % 2 == 0


def test_parent_uvs_untouched():
    plain = _build([])
    welded = _build(CASES["y45"])
    # the trunk's first two rows are far below the junction, identical in both
    n = 2 * plain.debug["sides"][0] * 4
    assert welded.debug["sides"][0] == plain.debug["sides"][0]
    np.testing.assert_allclose(plain.corner_uv[:n], welded.corner_uv[:n], atol=1e-12)


def test_islands_meet_at_the_crease():
    """Each weld face carries the parent's island or the branch's, by which
    side of the crease (parent distance == branch distance) it lies on -- so
    the border ring's UVs in the parent-side faces are the parent's own quads'
    UVs, and the branch ring's in the branch-side faces the branch's: no seam
    anywhere but along the crease. Both islands are used."""
    for name in ("y45", "y80_thick", "fork35"):
        out = _build(CASES[name], fillet=0.5)
        tube_uv = {}
        n_quads = int(np.sum(out.face_sizes == 4))
        for s in np.concatenate(([0], np.cumsum(out.face_sizes)[:-1]))[:n_quads]:
            for k in range(4):
                v = int(out.face_loops[s + k])
                tube_uv.setdefault(v, []).append(out.corner_uv[s + k])
        for c, j in out.debug["junction"].items():
            if "faces" not in j:
                continue
            ends = set(j["rings"][0].tolist()) | set(j["rings"][-1].tolist())
            sides = Counter()
            for vids, uv in j["faces"]:
                a, _, _, b, _, _ = _FIELDS[c].gated(out.positions[list(vids)].mean(axis=0)[None])
                sides[bool(a[0] > b[0])] += 1
                for v, (u_, v_) in zip(vids, uv, strict=True):
                    if v not in ends or v not in tube_uv:
                        continue
                    own = np.array(tube_uv[v])
                    du = (own[:, 0] - u_) / 2.0      # tiles = 2: whole turns apart are the same
                    assert np.min(np.abs(du - np.round(du)) + np.abs(own[:, 1] - v_)) < 1e-6, \
                        (name, c, v)
            assert sides[True] and sides[False], (name, c, sides)


def test_fillet_leaves_parent_tangentially():
    """The first fillet step (border -> first transition ring) runs close to
    the parent's tangent plane: under 35 deg off it on average, where a plain
    cylinder pushed into the parent would leave at the branch angle (80 here)."""
    for fillet in (0.5, 1.0, 1.5):
        out = _build(CASES["y80_thick"], fillet=fillet)
        rings = out.debug["junction"][1]["rings"]
        border, first = out.positions[rings[0]], out.positions[rings[1]]
        near = border[np.argmin(np.linalg.norm(first[:, None] - border[None], axis=2), axis=1)]
        radial = near.copy()
        radial[:, 2] = 0.0          # the trunk runs along z through the origin
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        step = first - near
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        off = np.degrees(np.arcsin(np.clip(np.abs(np.sum(step * radial, axis=1)), 0, 1)))
        assert float(off.mean()) < 35.0, (fillet, float(off.mean()))


def _transition_distortion(out):
    """Per transition triangle, the anisotropy of its 3D -> UV map: the ratio
    of its singular values (1 = square texels, whatever their size -- the
    join grades from the parent's texel size to the child's)."""
    pos = out.positions
    vals = []
    for j in out.debug["junction"].values():
        if "faces" not in j:
            continue
        for vids, uv in j["faces"]:
            P, U = pos[list(vids)], np.asarray(uv)
            for t0, t1, t2 in ((0, 1, 2), (0, 2, 3))[:len(vids) - 2]:
                e1, e2 = P[t1] - P[t0], P[t2] - P[t0]
                n = np.cross(e1, e2)
                if np.linalg.norm(n) < 1e-15:
                    continue
                x = e1 / np.linalg.norm(e1)
                y = np.cross(n / np.linalg.norm(n), x)
                frame = np.array([[e1 @ x, e2 @ x], [e1 @ y, e2 @ y]])
                jac = np.stack([U[t1] - U[t0], U[t2] - U[t0]], axis=1) @ np.linalg.inv(frame)
                sv = np.linalg.svd(jac, compute_uv=False)
                vals.append(float(sv[0] / max(sv[1], 1e-12)))
    return float(np.median(vals)), float(np.percentile(vals, 95))


def test_uv_distortion_bounded():
    """Texels through the weld stay square: each face is textured by the
    parent's or the branch's own tube projection, so anisotropy is median
    under 1.15 and 95th percentile under 2.5 (measured 1.01-1.05 /
    1.2-2.1; the branch's island run down over the whole weld measured
    1.5 / up to 7)."""
    for name in ("y45", "y80_thick", "spiral5", "opposite", "fork35"):
        med, p95 = _transition_distortion(_build(CASES[name], fillet=0.5))
        assert med < 1.15 and p95 < 2.5, (name, round(float(med), 2), round(float(p95), 2))


def test_weld_range_sets_the_ball():
    """The ball is Weld Range x the branch radius x sin(entry angle) (above
    the mesh's own resolution), and a larger one lifts more of the parent."""
    js = {f: _build(CASES["y45"], fillet=f).debug["junction"][1] for f in (2.0, 3.0)}
    for f, j in js.items():
        assert abs(j["k"] - f * j["r_c"] * math.sin(math.radians(45.0))) < 1e-6 * j["r_c"]
    assert js[3.0]["lo"] < js[2.0]["lo"] and js[3.0]["hi"] > js[2.0]["hi"]


def test_one_texel_size():
    """Every face -- trunk, thin branch, weld -- carries the same texture
    scale: sqrt(UV area / surface area) is 1 / tile size (Tiles Around round
    the thickest stem's base) on every tube face, and close to it in welds."""
    out = _build(CASES["spiral5"], tiles=2, fillet=0.5)
    widest = max(_skeleton(CASES["spiral5"])[2]["width"])     # (nothing flared)
    want = 2 / (2 * math.pi * widest)
    weld = {tuple(v) for j in out.debug["junction"].values() if "faces" in j
            for v, _ in j["faces"]}
    uv, k = out.corner_uv, 0
    tube_d, weld_d = [], []
    for f in out.face_lists():
        n = len(f)
        P, T = out.positions[f], uv[k:k + n]
        k += n
        if n < 4:
            continue                                   # caps and collapsed corners
        area = 0.5 * np.linalg.norm(sum(np.cross(P[i], P[(i + 1) % n]) for i in range(n)))
        uva = 0.5 * abs(sum(T[i, 0] * T[(i + 1) % n, 1] - T[(i + 1) % n, 0] * T[i, 1]
                            for i in range(n)))
        if area > 1e-12:
            (weld_d if tuple(f) in weld else tube_d).append(math.sqrt(uva / area) / want)
    tube_d, weld_d = np.array(tube_d), np.array(weld_d)
    assert np.all(np.abs(tube_d - 1) < 0.08), (tube_d.min(), tube_d.max())
    assert abs(np.median(weld_d) - 1) < 0.15, np.median(weld_d)


def test_not_welded():
    """A branch leaving at 8 degrees lies along its parent: welding it would
    fuse the two for many radii, so it stays a capped tube."""
    out = _build([(0.5, 0, 8, 0.4, 0.006)])
    assert out.junctions == 0 and out.fallbacks == 1


def test_crowded_branches_weld_on_their_node():
    """Branches whose welds would overlap -- one on its parent's first node
    (inside that parent's own weld), two siblings leaving together, and an
    opposite pair nearly as thick as their stem, whose openings would need
    more than the stem's whole circumference -- are welded together where
    they leave, none moved off its node; the mesh stays closed."""
    w_mid = 0.015                                   # the trunk's width at z 0.5
    for branches in (CASES["chain"],
                     [(0.5, 0, 50, 0.3, 0.010), (0.52, 20, 50, 0.3, 0.008)],
                     [(0.5, 0, 50, 0.3, 0.93 * w_mid), (0.5, 180, 50, 0.3, 0.93 * w_mid)]):
        out = _build(branches, fillet=0.5)
        assert out.fallbacks == 0 and out.junctions == len(branches), out.debug["why"]
        assert out.debug["clusters"], "not welded together"
        assert "slides" not in out.debug
        assert _open_edges(out) == 0


def test_thicker_side_shoot_leads():
    """A stem 9 mm up to a node and 4 mm after it, with an 8 mm branch there:
    the branch carries on as the stem's tube and the thin continuation is the
    one welded into it (a thick branch welded into a thin stem folds)."""
    v, e, a = _skeleton([(0.3, 0, 30, 0.4, 0.008)])
    tube, z = np.asarray(a["tube"]), np.asarray(v)[:, 2]
    a["width"] = [0.004 if (t == 0 and zz > 0.35) else w
                  for t, zz, w in zip(tube, z, a["width"], strict=True)]
    out = sm.build_stems(v, e, a, section=sm.SectionSpec(sides=32), fillet=0.5)
    assert out.debug["leads"] == [1], out.debug["leads"]
    assert out.junctions == 1 and out.fallbacks == 0, out.debug["why"]
    assert _open_edges(out) == 0


def test_smoothness_sets_the_budget():
    """One edge length sets sides and ring spacing: Smoothness 1 is coarse,
    8 fine, and a thin branch never has more sides than the trunk."""
    counts = {s: len(_build(CASES["spiral5"], smoothness=s).positions) for s in (1, 4, 8)}
    assert counts[1] < counts[4] < counts[8], counts
    out = _build(CASES["spiral5"])
    sides = out.debug["sides"]
    assert all(sides[t] <= sides[0] for t in sides), sides


def test_cross_sections_keep_their_shape():
    """Every tube -- welded (14+ sides) or not -- traces its section's true
    outline: a triangle's, square's and hexagon's corners are vertices and every
    vertex lies on the polygon; five ribs are five ribs (a multiple of five
    sides, at least four a rib); a flattened section is squashed. The welded
    mesh stays closed."""
    shapes = {
        "triangle": sm.SectionSpec(sides=32, polygon=3),
        "square": sm.SectionSpec(sides=32, polygon=4),
        "hexagon": sm.SectionSpec(sides=32, polygon=6),
        "ribbed": sm.SectionSpec(sides=64, lobe_count=5, lobe_depth=0.2),
        "flattened": sm.SectionSpec(sides=32, flatten=0.5),
    }
    for name, spec in shapes.items():
        out = _build(CASES["spiral5"], section=spec, fillet=0.5)
        assert _open_edges(out) == 0, name
        sides = out.debug["sides"]
        for t, n in sides.items():
            assert n % 2 == 0, (name, t, n)
            if spec.polygon:
                assert n % spec.polygon == 0, (name, t, n)
            if spec.lobe_count:
                assert n % spec.lobe_count == 0 and n >= 4 * spec.lobe_count, (name, t, n)
        # the unit outline each side count samples, against the shape's own
        for n in set(sides.values()):
            ring = sm._section(spec, n)
            ang = np.arctan2(ring[:, 1] / (1.0 - spec.flatten), ring[:, 0])
            rad = np.hypot(ring[:, 0], ring[:, 1] / (1.0 - spec.flatten))
            if spec.polygon:
                w = 2 * math.pi / spec.polygon
                want = math.cos(w / 2) / np.cos(np.mod(ang, w) - w / 2)
                corners = np.isclose(rad, 1.0).sum()
                assert corners == spec.polygon, (name, n, corners)
            else:
                want = 1.0 + np.cos(spec.lobe_count * ang) * spec.lobe_depth
            assert np.allclose(rad, want, atol=1e-9), (name, n)
        if spec.lobe_count:
            n = min(sides.values())
            rad = np.hypot(*sm._section(spec, n).T)
            peaks = np.sum((rad > np.roll(rad, 1)) & (rad > np.roll(rad, -1)))
            assert peaks == spec.lobe_count, (name, n, peaks)


def test_section_per_level_and_weld_off():
    """A section per branching level: the trunk (order 0) square, the branches
    (order 1) five-ribbed -- each tube takes its own level's and keeps its
    shape. With welding off every branch is a capped tube and the mesh is still
    closed (every edge on two faces)."""
    square = sm.SectionSpec(sides=32, polygon=4)
    ribbed = sm.SectionSpec(sides=64, lobe_count=5, lobe_depth=0.2)
    out = _build(CASES["spiral5"], section={0: square, 1: ribbed}, fillet=0.5)
    assert _open_edges(out) == 0
    sides = out.debug["sides"]
    assert sides[0] % 4 == 0, sides
    assert all(sides[t] % 5 == 0 and sides[t] >= 20 for t in sides if t != 0), sides
    off = _build(CASES["spiral5"], weld=False)
    assert off.junctions == 0 and off.fallbacks == 5, (off.junctions, off.fallbacks)
    assert set(off.debug["why"].values()) == {"weld off"}
    assert _open_edges(off) == 0


def test_hidden_tube_keeps_its_branches():
    """A level's Show off: the hidden trunk (tube 0) is not built, its five
    branches are -- each a closed, capped tube, unwelded -- and no vertex is
    left on the trunk's line."""
    full = _build(CASES["spiral5"])
    out = _build(CASES["spiral5"], hidden={0})
    assert out.junctions == 0 and out.fallbacks == 5, (out.junctions, out.fallbacks)
    assert set(out.debug["why"].values()) == {"on a hidden tube"}
    assert _open_edges(out) == 0
    assert 0 < len(out.positions) < len(full.positions)
    tube = out.point_attrs["tube"]
    assert 0 not in set(tube.tolist()) and len(set(tube.tolist())) == 5


def test_deterministic():
    a = _build(CASES["spiral5"], twist=0.2)
    b = _build(CASES["spiral5"], twist=0.2)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.face_loops, b.face_loops)
    assert np.array_equal(a.corner_uv, b.corner_uv)


def test_worker_process_builds_the_same_mesh():
    """`stem_worker`, run as its own process the way a live edit runs it, makes
    exactly the mesh `build_stems` makes here -- per-level sections, settings
    and a module switch changed in this process all carried across -- and
    refuses a job keyed on other code, writing nothing."""
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    from ..geometry import stem_worker as sw

    v, e, a = _skeleton(CASES["spiral5"])
    section = {0: sm.SectionSpec(sides=32, polygon=4),
               1: sm.SectionSpec(sides=64, lobe_count=5, lobe_depth=0.2)}
    kw = dict(twist=0.2, fillet=1.2, smoothness=3.0)
    held = sm._EDGES_PER_SIZE
    default = sm.build_stems(v, e, a, section=section, **kw)
    sm._EDGES_PER_SIZE = held * 1.25         # the worker must see this, not the file's
    try:
        here = sm.build_stems(v, e, a, section=section, **kw)
        switches = {n: x for n, x in vars(sm).items()
                    if n.startswith("_") and n[1:2].isupper()
                    and isinstance(x, (bool, int, float, str, tuple))}
        with tempfile.TemporaryDirectory() as tmp:
            job, out = os.path.join(tmp, "job.pkl"), os.path.join(tmp, "out.npz")
            with open(job, "wb") as fh:
                pickle.dump(sw.pack(v, e, a, section, kw, switches), fh)
            run = [sys.executable, sw.__file__, job]
            ok = subprocess.run(run + [out, sw.mesher_digest(sm.__file__), str(os.getpid())],
                                capture_output=True, text=True, timeout=300)
            assert ok.returncode == 0, ok.stderr[-800:]
            there = sw.load(sm, out)
            stale = subprocess.run(run + [out + ".x", "0" * 32, str(os.getpid())],
                                   capture_output=True, timeout=300)
            assert stale.returncode == sw.EXIT_STALE_CODE, stale.returncode
            assert not os.path.exists(out + ".x")
    finally:
        sm._EDGES_PER_SIZE = held
    assert len(here.positions) != len(default.positions)    # the switch did change it
    for k in ("positions", "face_sizes", "face_loops", "corner_uv"):
        assert np.array_equal(getattr(here, k), getattr(there, k)), k
    assert here.point_attrs.keys() == there.point_attrs.keys()
    for k, x in here.point_attrs.items():
        y = there.point_attrs[k]
        assert np.array_equal(x, y) and np.asarray(x).dtype == y.dtype, k
    assert (here.junctions, here.fallbacks) == (there.junctions, there.fallbacks)


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
