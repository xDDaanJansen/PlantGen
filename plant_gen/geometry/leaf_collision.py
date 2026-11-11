"""Leaf collision: turn leaves out of the stems and out of each other, prune the
rest -- bpy-free, deterministic.

Natsura's Deflection effector ("awareness of the plant's own shape") and
SpeedTree's collision vocabulary (Prune = delete it) are the references.
The resolve is the smallest
change that works, leaf by leaf:

  1. every leaf is posed exactly as the node graph will draw it -- its asset's
     mesh (decimated to a proxy, `simplify`) at its base, length along
     `direction`, face along `normal`, times its uniform `scale`;
  2. leaves settle oldest first (the younger yields: it grew into the space
     the older one left); a leaf that touches a stem, a settled leaf or a
     settled leaf's stalk is turned about where it attaches to its stem --
     sideways, up / down, or rolled about its midrib, together with its
     stalk, in growing steps up to its Max Turn -- and takes the first pose
     that is clear;
  3. a leaf no turn clears is pruned (when its component allows it), else it
     keeps the pose that touches least.

Fixed obstacles (`Leaf.movable` False) settle first and never turn: leaves of
components without Collision, and the flowers (`Leaf.upright`: an asset that
runs along +Z, as the node graph instances flowers). A turn never takes a leaf
below the stems' lowest point (the ground), nor lower than it already was when
it starts there -- such a pose counts as not clear.

Touching inside a leaf's own attachment zone (`Leaf.exempt` round its pivot:
the stem it grows from, the base it shares with its rosette or whorl) is not a
collision -- that is where a leaf is meant to meet the stem.

Intersection is exact on the proxies: a segment (a stem's centre line, a
stalk, a proxy's edge) crossing a triangle (Moller-Trumbore), plus a leaf's
vertices within a stem's radius (a thick stem a leaf's edge passes inside).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12
_STEP_DEG = 5.0
"""Turn steps: Max Turn is tried in steps of about this many degrees."""
_DIRECTIONS = 8
"""Turn directions per step round the leaf's base (sideways, up, down and the
diagonals), before the two rolls."""
_BELOW = 1 << 30
"""What a turn taking a leaf below the ground counts as: never clear."""


@dataclass
class Shape:
    """A leaf asset's collision proxy, in the asset's own frame (length +Y,
    face +Z, origin at the leaf's base), unit scale."""
    verts: np.ndarray                      # (V, 3)
    tris: np.ndarray                       # (T, 3) vertex indices
    edges: np.ndarray = field(default=None)  # (E, 2), filled by `__post_init__`

    def __post_init__(self):
        self.verts = np.asarray(self.verts, dtype=np.float64).reshape(-1, 3)
        self.tris = np.asarray(self.tris, dtype=np.int64).reshape(-1, 3)
        if self.edges is None:
            e = np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]],
                                self.tris[:, [2, 0]]])
            e = np.sort(e, axis=1)
            self.edges = np.unique(e, axis=0) if len(e) else np.zeros((0, 2), np.int64)


@dataclass
class Leaf:
    shape: int                 # index into the shapes list
    base: np.ndarray           # (3,) the asset's origin (the blade's base)
    direction: np.ndarray      # (3,) length axis
    normal: np.ndarray         # (3,) face axis
    scale: float
    pivot: np.ndarray          # (3,) where it turns: its foot on the stem
    stalk: np.ndarray | None = None   # (P, 3) petiole points, pivot first
    stalk_radius: float = 0.0
    exempt: float = 0.0        # attachment zone radius round the pivot
    order: float = 0.0         # settles in ascending order (older first)
    movable: bool = False      # False: a fixed obstacle, never turned
    max_turn: float = 0.0      # radians
    prune: bool = False        # pruned when no turn clears it
    upright: bool = False      # the asset runs along +Z, faces +Y (a flower), not +Y / +Z


@dataclass
class Stems:
    """The stems as capsules: segment ends and radius."""
    a: np.ndarray              # (S, 3)
    b: np.ndarray              # (S, 3)
    radius: np.ndarray         # (S,)


@dataclass
class Outcome:
    turn: list                 # per leaf: (3, 3) rotation about its pivot (identity kept)
    angle: np.ndarray          # (N,) the turn, radians
    pruned: np.ndarray         # (N,) bool
    clashed: np.ndarray        # (N,) bool: collided in its first pose
    unresolved: np.ndarray     # (N,) bool: still colliding (kept: not prunable)


# --- the proxy ----------------------------------------------------------------
def simplify(verts, tris, cells: tuple = (4, 10, 2)) -> Shape:
    """A coarse proxy of a leaf mesh by vertex clustering: the bounding box cut
    into `cells` (across, along, through), each cell's vertices merged at their
    mean, degenerate and repeated triangles dropped. A blade keeps its outline
    and curl at about the cell size; `cells` (4, 10, 2) gives ~50 tris (the
    resolve's cost goes with the square of it: ~100 tris was 3.5x slower on a
    crowded rosette for the same leaves pruned to within a few)."""
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(t) == 0:
        return Shape(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    n = np.asarray(cells, dtype=np.int64)
    cell = np.minimum(((v - lo) / span * n).astype(np.int64), n - 1)
    key = (cell[:, 0] * n[1] + cell[:, 1]) * n[2] + cell[:, 2]
    uniq, inverse = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inverse, v)
    counts = np.bincount(inverse, minlength=len(uniq)).astype(np.float64)
    new_v = sums / counts[:, None]
    nt = inverse[t]
    ok = (nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 2] != nt[:, 0])
    nt = nt[ok]
    if len(nt):
        nt = np.unique(np.sort(nt, axis=1), axis=0)
    return Shape(new_v, nt)


# --- primitives ---------------------------------------------------------------
def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS or angle == 0.0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def _frame(direction, normal, upright: bool = False) -> np.ndarray:
    """Columns: the asset's X (width), Y (length), Z (face) in world space --
    what the node graph's Axes to Rotation (Y primary, Z secondary) builds; an
    `upright` asset (a flower: Z primary, Y secondary) takes Z along `direction`
    and Y along `normal`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), _EPS)
    n = np.asarray(normal, dtype=np.float64)
    n = n - d * float(n @ d)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        n = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n = n - d * float(n @ d)
        nn = float(np.linalg.norm(n))
    n = n / nn
    if upright:
        return np.column_stack([np.cross(n, d), n, d])
    return np.column_stack([np.cross(d, n), d, n])


def segments_hit_triangles(p0, p1, tri) -> tuple[np.ndarray, np.ndarray]:
    """Moller-Trumbore for every segment against every triangle.

    `p0`, `p1` (M, 3) segment ends; `tri` (K, 3, 3). Returns `hit` (M, K) bool
    and the hit points (M, K, 3) (meaningful where hit)."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(tri, dtype=np.float64).reshape(-1, 3, 3)
    m, k = len(p0), len(tri)
    if m == 0 or k == 0:
        return np.zeros((m, k), dtype=bool), np.zeros((m, k, 3))
    d = (p1 - p0)[:, None, :]                          # (M, 1, 3)
    e1 = (tri[:, 1] - tri[:, 0])[None]                 # (1, K, 3)
    e2 = (tri[:, 2] - tri[:, 0])[None]
    h = np.cross(d, e2)                                # (M, K, 3)
    a = np.einsum("mkj,mkj->mk", np.broadcast_to(e1, h.shape), h)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p0[:, None, :] - tri[None, :, 0]               # (M, K, 3)
    u = f * np.einsum("mkj,mkj->mk", s, h)
    q = np.cross(s, np.broadcast_to(e1, s.shape))
    v = f * np.einsum("mkj,mkj->mk", np.broadcast_to(d, q.shape), q)
    t = f * np.einsum("mkj,mkj->mk", np.broadcast_to(e2, q.shape), q)
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)
    points = p0[:, None, :] + t[..., None] * np.broadcast_to(d, s.shape)
    return hit, points


def _box_pairs(lo_a, hi_a, lo_b, hi_b) -> tuple[np.ndarray, np.ndarray]:
    """Every index pair (i, j) whose closed boxes overlap on all three axes, in
    row-major order -- exactly `np.nonzero` of the dense overlap matrix. Only
    the first test runs over the whole matrix (the axis `b` spreads widest
    along, which rules out the most); the other two axes are compared on the
    pairs that pass it -- the same comparisons, a fraction of the work."""
    if not len(lo_a) or not len(lo_b):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ax = int(np.argmax(hi_b.max(axis=0) - lo_b.min(axis=0)))
    near = hi_a[:, ax, None] >= lo_b[None, :, ax]
    near &= lo_a[:, ax, None] <= hi_b[None, :, ax]
    ii, jj = np.nonzero(near)
    if not len(ii):
        return ii, jj
    ok = np.ones(len(ii), dtype=bool)
    for k in range(3):
        if k != ax:
            ok &= hi_a[ii, k] >= lo_b[jj, k]
            ok &= lo_a[ii, k] <= hi_b[jj, k]
    return ii[ok], jj[ok]


def _crossings(p0, p1, tri):
    """Moller-Trumbore on the segment / triangle PAIRS whose boxes overlap only:
    the hit pairs' segment and triangle indices and the hit points."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)))
    if not len(p0) or not len(tri):
        return empty
    si, ti = _box_pairs(np.minimum(p0, p1), np.maximum(p0, p1),
                        tri.min(axis=1), tri.max(axis=1))
    if not len(si):
        return empty
    a, d = p0[si], p1[si] - p0[si]
    t0 = tri[ti, 0]
    e1, e2 = tri[ti, 1] - t0, tri[ti, 2] - t0
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    sv = a - t0
    u = f * np.einsum("ij,ij->i", sv, h)
    q = np.cross(sv, e1)
    v = f * np.einsum("ij,ij->i", d, q)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)
    return si[hit], ti[hit], a[hit] + t[hit, None] * d[hit]


def _beyond(points, centre, r) -> np.ndarray:
    """Which points lie outside a zone of radius `r` (all of them for r <= 0)."""
    r = np.asarray(r, dtype=np.float64)
    return (r <= 0.0) | (np.linalg.norm(points - centre, axis=1) > r)


def points_to_segments(p, a, b) -> np.ndarray:
    """Distance of every point (N, 3) to every segment a-b (S, 3): (N, S)."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    ab = b - a
    ll = np.maximum(np.einsum("sj,sj->s", ab, ab), _EPS)
    ap = p[:, None, :] - a[None]
    t = np.clip(np.einsum("nsj,sj->ns", ap, ab) / ll, 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - closest, axis=2)


# --- a posed leaf -------------------------------------------------------------
class _Posed:
    """One leaf's proxy in world space for a given turn about its pivot: its
    triangles, its segments (blade edges, then the stalk's) and their box."""

    __slots__ = ("verts", "tris", "seg_a", "seg_b", "stalk", "centre", "radius", "lo", "hi")

    def __init__(self, local_verts, shape: Shape, pivot, turn, stalk):
        world = pivot + local_verts @ turn.T
        self.verts = world
        self.tris = world[shape.tris] if len(shape.tris) else np.zeros((0, 3, 3))
        a = world[shape.edges[:, 0]] if len(shape.edges) else np.zeros((0, 3))
        b = world[shape.edges[:, 1]] if len(shape.edges) else np.zeros((0, 3))
        pts = world
        self.stalk = None
        if stalk is not None and len(stalk) > 1:
            s = pivot + stalk @ turn.T
            a, b = np.vstack([a, s[:-1]]), np.vstack([b, s[1:]])
            pts = np.vstack([world, s])
            self.stalk = s
        self.seg_a, self.seg_b = a, b
        if len(pts):
            self.centre = pts.mean(axis=0)
            self.radius = float(np.linalg.norm(pts - self.centre, axis=1).max())
            self.lo, self.hi = pts.min(axis=0), pts.max(axis=0)
        else:
            self.centre = np.asarray(pivot, np.float64)
            self.radius = 0.0
            self.lo = self.hi = self.centre


class _Pool:
    """The settled leaves near the one being placed, flattened into one set of
    triangles and one of segments (each with its owner) and their boxes -- so a
    candidate pose is tested against all of them in one vectorised call."""

    def __init__(self, items):
        owners = range(len(items))
        tris = [p.tris for _leaf, p in items]
        self.tris = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        self.tri_owner = np.concatenate(
            [np.full(len(t), o) for o, t in zip(owners, tris, strict=True)]
            or [np.zeros(0, np.int64)]).astype(np.int64)
        seg_a = [p.seg_a for _leaf, p in items]
        self.seg_a = np.concatenate(seg_a) if seg_a else np.zeros((0, 3))
        self.seg_b = (np.concatenate([p.seg_b for _leaf, p in items]) if items
                      else np.zeros((0, 3)))
        self.seg_owner = np.concatenate(
            [np.full(len(a), o) for o, a in zip(owners, seg_a, strict=True)]
            or [np.zeros(0, np.int64)]).astype(np.int64)
        self.tri_lo, self.tri_hi = (self.tris.min(axis=1), self.tris.max(axis=1))
        self.seg_lo = np.minimum(self.seg_a, self.seg_b)
        self.seg_hi = np.maximum(self.seg_a, self.seg_b)
        self.pivot = (np.array([np.asarray(leaf.pivot, np.float64) for leaf, _p in items])
                      if items else np.zeros((0, 3)))
        self.exempt = np.array([float(leaf.exempt) for leaf, _p in items])
        # The stalks as tubes (their radius): segments, and points along them
        # (ends and midpoints); the blades' vertices, to test against stalks.
        sa, sb, sr, so, pp, pr, po = [], [], [], [], [], [], []
        for o, (leaf, posed) in enumerate(items):
            st = posed.stalk
            if st is None or leaf.stalk_radius <= 0.0:
                continue
            sa.append(st[:-1])
            sb.append(st[1:])
            sr.append(np.full(len(st) - 1, float(leaf.stalk_radius)))
            so.append(np.full(len(st) - 1, o))
            dense = _stalk_points(st)
            pp.append(dense)
            pr.append(np.full(len(dense), float(leaf.stalk_radius)))
            po.append(np.full(len(dense), o))
        self.stalk_a, self.stalk_b = _cat(sa, (0, 3)), _cat(sb, (0, 3))
        self.stalk_r = _cat(sr, (0,))
        self.stalk_owner = _cat(so, (0,)).astype(np.int64)
        self.stalk_pts, self.stalk_pt_r = _cat(pp, (0, 3)), _cat(pr, (0,))
        self.stalk_pt_owner = _cat(po, (0,)).astype(np.int64)
        self.verts = _cat([posed.verts for _leaf, posed in items], (0, 3))
        self.vert_owner = _cat([np.full(len(posed.verts), o)
                                for o, (_leaf, posed) in enumerate(items)], (0,)).astype(np.int64)


def _cat(parts, shape) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(shape)


def _stalk_points(stalk) -> np.ndarray:
    """A stalk's points and its segments' midpoints."""
    return np.vstack([stalk, (stalk[:-1] + stalk[1:]) * 0.5])


def _near_faces(points, radius, tris):
    """Points within their `radius` of a triangle's face (the projection inside
    it): the pairs' point and triangle indices."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if not len(points) or not len(tris):
        return empty
    pi, ti = _box_pairs(points - radius[:, None], points + radius[:, None],
                        tris.min(axis=1), tris.max(axis=1))
    if not len(pi):
        return empty
    t0 = tris[ti, 0]
    e1, e2 = tris[ti, 1] - t0, tris[ti, 2] - t0
    nrm = np.cross(e1, e2)
    length = np.linalg.norm(nrm, axis=1)
    ok = length > 1e-14
    nrm = nrm / np.where(ok, length, 1.0)[:, None]
    w = points[pi] - t0
    height = np.einsum("ij,ij->i", w, nrm)
    ok &= np.abs(height) < radius[pi]
    flat = w - height[:, None] * nrm
    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    d20 = np.einsum("ij,ij->i", flat, e1)
    d21 = np.einsum("ij,ij->i", flat, e2)
    den = d00 * d11 - d01 * d01
    ok &= np.abs(den) > 1e-20
    den = np.where(ok, den, 1.0)
    v = (d11 * d20 - d01 * d21) / den
    u = (d00 * d21 - d01 * d20) / den
    ok &= (v >= 0.0) & (u >= 0.0) & (u + v <= 1.0)
    return pi[ok], ti[ok]


def _overlapping(lo, hi, box_lo, box_hi) -> np.ndarray:
    return np.all(hi >= box_lo, axis=1) & np.all(lo <= box_hi, axis=1)


def _batch_hits(leaf: Leaf, local, stalk, shape: Shape, pivot, rots, stems: Stems,
                near: np.ndarray, pool: _Pool, cap: int | None = None,
                skip: np.ndarray | None = None) -> np.ndarray:
    """How many collision pieces each of a batch of poses (`rots`, (C, 3, 3)
    turns about the pivot) has -- all C tested in one vectorised pass, the
    count per pose: stem centre lines through the blade, blade vertices and
    stalk points inside a stem, blade edges / stalk through the settled
    leaves' blades and theirs through this blade, each outside the attachment
    zones.

    `cap`: a pose with `cap` pieces or more can no longer be chosen (the resolve
    only takes a pose with fewer than its best so far; a leaf that is pruned
    when nothing clears it, only one with none -- cap 1). The checks run one
    kind at a time and a pose stops being checked once its count reaches the
    cap: its count is then some part of its total, but still >= cap, and every
    pose below the cap is counted in full -- so the resolve chooses exactly as
    with every count in full. `skip`: poses already known to be over it."""
    c = len(rots)
    counts = np.zeros(c, dtype=np.int64)
    alive = np.arange(c) if skip is None else np.nonzero(~skip)[0]
    if not len(alive):
        return counts
    batch = _PoseBatch(local, stalk, shape, pivot, rots[alive])
    for check in _CHECKS:
        counts[alive] += check(batch, leaf, pivot, stems, near, pool)
        if cap is not None:
            under = counts[alive] < cap
            if not under.all():
                alive = alive[under]
                if not len(alive):
                    break
                batch = batch.subset(under)
    return counts


class _PoseBatch:
    """A batch of C poses of one leaf: its blade's vertices, triangles and edges
    (with the stalk's segments after them), its stalk, and their box."""

    def __init__(self, local, stalk, shape, pivot, rots):
        self.local, self.stalk_local, self.shape, self.pivot = local, stalk, shape, pivot
        self.rots = rots
        self.c = len(rots)
        verts = pivot + np.einsum("vj,cij->cvi", local, rots)              # (C, V, 3)
        self.nv, self.nt = local.shape[0], len(shape.tris)
        self.tris = verts[:, shape.tris].reshape(-1, 3, 3)                  # (C*T, 3, 3)
        ea, eb = verts[:, shape.edges[:, 0]], verts[:, shape.edges[:, 1]]  # (C, E, 3)
        self.s = self.sp = None
        if stalk is not None and len(stalk) > 1:
            s = pivot + np.einsum("pj,cij->cpi", stalk, rots)              # (C, P, 3)
            ea = np.concatenate([ea, s[:, :-1]], axis=1)
            eb = np.concatenate([eb, s[:, 1:]], axis=1)
            self.s, self.sp = s, s.reshape(-1, 3)
        self.ne = ea.shape[1]
        self.ea, self.eb = ea.reshape(-1, 3), eb.reshape(-1, 3)
        self.flat_v = verts.reshape(-1, 3)
        every = self.flat_v if self.sp is None else np.vstack([self.flat_v, self.sp])
        self.lo, self.hi = every.min(axis=0), every.max(axis=0)

    def subset(self, keep):
        return _PoseBatch(self.local, self.stalk_local, self.shape, self.pivot, self.rots[keep])


def _check_stems(b, leaf, pivot, stems, near, pool):
    """Stem centre lines through the blade; blade vertices and stalk points
    inside a stem."""
    counts = np.zeros(b.c, dtype=np.int64)
    if not (len(near) and b.nt):
        return counts
    zone = float(leaf.exempt)
    a, e, r = stems.a[near], stems.b[near], stems.radius[near]
    grow = float(r.max())
    keep = _overlapping(np.minimum(a, e) - grow, np.maximum(a, e) + grow, b.lo, b.hi)
    if not keep.any():
        return counts
    a, e, r = a[keep], e[keep], r[keep]
    _si, ti, p = _crossings(a, e, b.tris)
    if len(ti):
        ok = _beyond(p, pivot, zone)
        counts += np.bincount(ti[ok] // b.nt, minlength=b.c)
    inside = (points_to_segments(b.flat_v, a, e) < r[None]).any(axis=1)
    inside &= _beyond(b.flat_v, pivot, zone)
    counts += inside.reshape(b.c, b.nv).sum(axis=1)
    if b.sp is not None:
        inside = (points_to_segments(b.sp, a, e) < (r[None] + leaf.stalk_radius)).any(axis=1)
        inside &= _beyond(b.sp, pivot, zone)
        counts += inside.reshape(b.c, -1).sum(axis=1)
    return counts


def _check_edges_vs_their_blades(b, leaf, pivot, stems, near, pool):
    """This blade's edges (and stalk) through the settled leaves' blades."""
    counts = np.zeros(b.c, dtype=np.int64)
    if not (len(pool.tri_owner) and b.ne):
        return counts
    sel = np.nonzero(_overlapping(pool.tri_lo, pool.tri_hi, b.lo, b.hi))[0]
    if len(sel):
        si, ti, p = _crossings(b.ea, b.eb, pool.tris[sel])
        if len(si):
            owner = pool.tri_owner[sel][ti]
            ok = (_beyond(p, pivot, float(leaf.exempt))
                  & _beyond(p, pool.pivot[owner], pool.exempt[owner]))
            counts += np.bincount(si[ok] // b.ne, minlength=b.c)
    return counts


def _check_their_edges_vs_blade(b, leaf, pivot, stems, near, pool):
    """The settled leaves' edges (and stalks) through this blade."""
    counts = np.zeros(b.c, dtype=np.int64)
    if not (len(pool.seg_owner) and b.nt):
        return counts
    sel = np.nonzero(_overlapping(pool.seg_lo, pool.seg_hi, b.lo, b.hi))[0]
    if len(sel):
        si, ti, p = _crossings(pool.seg_a[sel], pool.seg_b[sel], b.tris)
        if len(si):
            owner = pool.seg_owner[sel][si]
            ok = (_beyond(p, pivot, float(leaf.exempt))
                  & _beyond(p, pool.pivot[owner], pool.exempt[owner]))
            counts += np.bincount(ti[ok] // b.nt, minlength=b.c)
    return counts


def _check_stalk_tubes(b, leaf, pivot, stems, near, pool):
    """The stalks are tubes, not lines: a blade touching one's surface
    collides, and this stalk's tube against the settled blades."""
    counts = np.zeros(b.c, dtype=np.int64)
    zone = float(leaf.exempt)
    if len(pool.stalk_owner):
        counts += _blade_vs_their_stalks(b.flat_v, b.nv, b.c, pivot, zone, pool, b.lo, b.hi)
        if b.nt:
            counts += _their_stalks_vs_blade(b.tris, b.nt, b.c, pivot, zone, pool, b.lo, b.hi)
    if b.sp is not None and leaf.stalk_radius > 0.0 and len(pool.tri_owner):
        counts += _stalk_vs_their_blades(b.s, float(leaf.stalk_radius), b.c, pivot, zone, pool,
                                         b.lo, b.hi)
    return counts


_CHECKS = (_check_stems, _check_edges_vs_their_blades, _check_their_edges_vs_blade,
           _check_stalk_tubes)
"""What `_batch_hits` counts, in the order the full count always added it."""


def _blade_vs_their_stalks(flat_v, nv, c, pivot, zone, pool, lo, hi):
    """Per pose: this blade's vertices inside the settled leaves' stalks."""
    grow = float(pool.stalk_r.max())
    sel = np.nonzero(_overlapping(np.minimum(pool.stalk_a, pool.stalk_b) - grow,
                                  np.maximum(pool.stalk_a, pool.stalk_b) + grow, lo, hi))[0]
    if not len(sel):
        return 0
    d = points_to_segments(flat_v, pool.stalk_a[sel], pool.stalk_b[sel])
    vi, si = np.nonzero(d < pool.stalk_r[sel][None])
    if not len(vi):
        return 0
    owner = pool.stalk_owner[sel][si]
    ok = (_beyond(flat_v[vi], pivot, zone)
          & _beyond(flat_v[vi], pool.pivot[owner], pool.exempt[owner]))
    hit = np.zeros(len(flat_v), dtype=bool)
    hit[vi[ok]] = True
    return hit.reshape(c, nv).sum(axis=1)


def _their_stalks_vs_blade(tris, nt, c, pivot, zone, pool, lo, hi):
    """Per pose: points along the settled leaves' stalks within their radius of
    this blade's faces."""
    pts, rad, own = pool.stalk_pts, pool.stalk_pt_r, pool.stalk_pt_owner
    keep = _overlapping(pts - rad[:, None], pts + rad[:, None], lo, hi)
    if not keep.any():
        return 0
    pi, ti = _near_faces(pts[keep], rad[keep], tris)
    if not len(pi):
        return 0
    q, owner = pts[keep][pi], own[keep][pi]
    ok = _beyond(q, pivot, zone) & _beyond(q, pool.pivot[owner], pool.exempt[owner])
    return np.bincount(ti[ok] // nt, minlength=c)


def _stalk_vs_their_blades(s, r_own, c, pivot, zone, pool, lo, hi):
    """Per pose: this stalk (C, P, 3) as a tube against the settled blades --
    its points near their faces, their vertices inside it."""
    counts = np.zeros(c, dtype=np.int64)
    dense = np.concatenate([s, (s[:, :-1] + s[:, 1:]) * 0.5], axis=1)       # (C, Q, 3)
    nq = dense.shape[1]
    flat_q = dense.reshape(-1, 3)
    sel = np.nonzero(_overlapping(pool.tri_lo - r_own, pool.tri_hi + r_own, lo, hi))[0]
    if len(sel):
        pi, ti = _near_faces(flat_q, np.full(len(flat_q), r_own), pool.tris[sel])
        if len(pi):
            q, owner = flat_q[pi], pool.tri_owner[sel][ti]
            ok = _beyond(q, pivot, zone) & _beyond(q, pool.pivot[owner], pool.exempt[owner])
            counts += np.bincount(pi[ok] // nq, minlength=c)
    keep = _overlapping(pool.verts, pool.verts, lo - r_own, hi + r_own)
    if keep.any():
        pv, po = pool.verts[keep], pool.vert_owner[keep]
        ns = s.shape[1] - 1
        d = points_to_segments(pv, s[:, :-1].reshape(-1, 3), s[:, 1:].reshape(-1, 3))
        vi, si = np.nonzero(d < r_own)
        if len(vi):
            ok = (_beyond(pv[vi], pivot, zone)
                  & _beyond(pv[vi], pool.pivot[po[vi]], pool.exempt[po[vi]]))
            counts += np.bincount(si[ok] // ns, minlength=c)
    return counts


def _candidates(leaf: Leaf, frame: np.ndarray) -> list:
    """The turns tried, smallest first, as one batch per step: round the base
    (the axis in the leaf's width / face plane: sideways about its face normal,
    up / down about its width), then rolled either way about its length."""
    out = []
    if leaf.max_turn <= 0.0:
        return out
    steps = max(1, int(math.ceil(math.degrees(leaf.max_turn) / _STEP_DEG)))
    width, length, face = frame[:, 0], frame[:, 1], frame[:, 2]
    for k in range(1, steps + 1):
        angle = leaf.max_turn * k / steps
        batch = []
        for j in range(_DIRECTIONS):
            phi = 2.0 * math.pi * j / _DIRECTIONS
            # j = 0 sideways (about the face normal), j = 2 up (about the width
            # axis: the tip toward the face), then round; ties go in this order.
            axis = math.cos(phi) * face + math.sin(phi) * width
            batch.append(_rotation(axis, angle))
        batch.append(_rotation(length, angle))
        batch.append(_rotation(length, -angle))
        out.append((angle, np.array(batch)))
    return out


def resolve(leaves: list, shapes: list, stems: Stems) -> Outcome:
    """Settle every leaf (see the module notes). Leaves that are not `movable`
    are fixed obstacles settled first; the rest in ascending `order` (then list
    order). Deterministic: the same input gives the same output."""
    n = len(leaves)
    eye = np.eye(3)
    turn = [eye] * n
    angle = np.zeros(n)
    pruned = np.zeros(n, dtype=bool)
    clashed = np.zeros(n, dtype=bool)
    unresolved = np.zeros(n, dtype=bool)
    if n == 0:
        return Outcome(turn, angle, pruned, clashed, unresolved)

    local, frames, stalks = [], [], []
    for leaf in leaves:
        shape = shapes[leaf.shape]
        frame = _frame(leaf.direction, leaf.normal, leaf.upright)
        frames.append(frame)
        world = np.asarray(leaf.base, np.float64) + float(leaf.scale) * shape.verts @ frame.T
        local.append(world - np.asarray(leaf.pivot, np.float64))
        stalks.append(None if leaf.stalk is None
                      else np.asarray(leaf.stalk, np.float64) - leaf.pivot)

    order = sorted(range(n), key=lambda i: (leaves[i].movable, leaves[i].order, i))
    # The ground: the stems' lowest point.
    floor = (float(min(stems.a[:, 2].min(), stems.b[:, 2].min())) if len(stems.a)
             else -math.inf)
    settled: list = []                    # (leaf, posed)
    settled_centre: list = []             # each one's bounding sphere
    settled_radius: list = []

    def settle(leaf, posed):
        settled.append((leaf, posed))
        settled_centre.append(posed.centre)
        settled_radius.append(posed.radius)
    seg_mid = (stems.a + stems.b) * 0.5 if len(stems.a) else np.zeros((0, 3))
    seg_half = (np.linalg.norm(stems.b - stems.a, axis=1) * 0.5
                if len(stems.a) else np.zeros(0))
    for i in order:
        leaf = leaves[i]
        shape = shapes[leaf.shape]
        pivot = np.asarray(leaf.pivot, np.float64)
        if not leaf.movable:
            settle(leaf, _Posed(local[i], shape, pivot, eye, stalks[i]))
            continue
        pts = local[i] if stalks[i] is None else np.vstack([local[i], stalks[i]])
        reach = float(np.linalg.norm(pts, axis=1).max()) if len(pts) else 0.0
        # Everything a turn of this leaf could reach: a sphere of its reach
        # round the pivot.
        if len(seg_mid):
            gap = np.linalg.norm(seg_mid - pivot, axis=1) - seg_half - stems.radius
            near = np.nonzero(gap < reach)[0]
        else:
            near = np.zeros(0, dtype=np.int64)
        # The settled leaves whose spheres could meet this one's reach (a hair
        # wider than the exact test: a leaf too far to touch only adds pairs
        # that fail, so a superset changes no count).
        if settled:
            dist = np.linalg.norm(np.asarray(settled_centre) - pivot, axis=1)
            close = np.nonzero(dist < np.asarray(settled_radius) + reach + 1e-9)[0]
            pool = _Pool([settled[j] for j in close])
        else:
            pool = _Pool([])

        # A turn may not take it below the ground -- nor, starting lower, any
        # lower than it is.
        ground = min(floor, float(pivot[2] + pts[:, 2].min())) if len(pts) else -math.inf

        # Only a pose with fewer pieces than the best so far is ever taken (a
        # leaf its component prunes: only one with none), so a pose's checks
        # stop once it reaches that (`_batch_hits` cap) -- the same choices.
        def hits(rots, cap=None, i=i, leaf=leaf, shape=shape, pivot=pivot, near=near,
                 pool=pool, pts=pts, ground=ground):
            below = None
            if len(pts) and math.isfinite(ground):
                low = pivot[2] + np.einsum("vj,cj->cv", pts, rots[:, 2, :]).min(axis=1)
                below = low < ground - 1e-9
            counts = _batch_hits(leaf, local[i], stalks[i], shape, pivot, rots, stems, near,
                                 pool, cap, below if cap is not None else None)
            if below is not None:
                counts = counts + np.where(below, _BELOW, 0)
            return counts

        best_hits = int(hits(eye[None], 1 if leaf.prune else None)[0])
        best_angle, best_turn = 0.0, eye
        if best_hits:
            clashed[i] = True
            for a, rots in _candidates(leaf, frames[i]):
                counts = hits(rots, 1 if leaf.prune else best_hits)
                k = int(np.argmin(counts))          # the first with the fewest
                if counts[k] < best_hits:
                    best_hits, best_angle, best_turn = int(counts[k]), a, rots[k]
                if best_hits == 0:
                    break
            if best_hits and leaf.prune:
                pruned[i] = True
                continue
        unresolved[i] = best_hits > 0
        turn[i], angle[i] = best_turn, best_angle
        settle(leaf, _Posed(local[i], shape, pivot, best_turn, stalks[i]))
    return Outcome(turn, angle, pruned, clashed, unresolved)
