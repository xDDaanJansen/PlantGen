"""The stem mesher: the skeleton's chains swept into tubes and joined at every
junction **by construction** -- one mesh, no boolean, no remesh.

Weld Method = Built. The
Manifold-boolean weld cuts its seam through the parent's existing quads and can
only blur the vertices that happen to be there; a volume remesh gives the
smoothest joins but re-tessellates the whole plant (the polygon budget) and
fuses anything close (catkins). Here the volume's *shape* is built locally:

1. **Tubes, adaptively.** Each tube (one `tube` id per skeleton chain) is swept
   along a smooth centre line (Bezier, auto handles) with a transported frame +
   Twist and the cross-section. Its side count follows its thickness (a share
   of the widest stem). Rings are placed only where the tube needs them -- where
   it bends, where its taper stops being linear, as often as the bark lumps
   and the twist need -- so straight runs become long quads (floored at the
   plant height / 300; Smoothness scales every tolerance).
2. **The weld surface** is the parent and branch tubes joined by a rolling-ball
   fillet of radius Weld Range x the branch radius: the round union of the two
   tubes' distance fields, each counting only its *exposed* surface (a branch's
   part buried inside its parent must not bulge the parent's far side). This is
   what a volume remesh with a fillet converges to, evaluated per junction.
3. **The hole** in the parent is exactly the region that surface lifts off the
   parent (sampled on the parent surface), snapped outward to whole parent
   columns and rows inserted near-square to the parent's faces. The branch's
   first ring sits where the surface has stopped touching the parent.
4. **The transition** from the hole's border to the branch's first ring:
   meridians leave the parent tangentially (Hermite), are projected onto the
   weld surface, and rings are laid at equal arc length along them. The ring
   vertex count eases from the border's to the branch's own, so a thin branch
   in a thick fork keeps its few sides (quads, a triangle where a count drops).
5. UVs: the parent's island is untouched (the hole only removes quads); the
   branch's island runs down through the transition to the border.

A child whose footprint cannot be placed (off the parent's ends, overlapping a
sibling's columns, wrapping its parent, too short to leave the fillet) falls
back to a tube capped inside its parent -- counted in `MeshOut.fallbacks`.

Studied, not ported: MTree's manifold mesher (GPL-3.0) for the idea of a hole
joined to the child ring; the fillet is the standard round-union / rolling-ball
construction. Everything here is written in our own terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_DENSE = 8
"""Samples per skeleton segment when evaluating the smoothed centre line."""
_TRANSITION_MIN, _TRANSITION_MAX = 3, 32
"""Fillet rings per junction: by its length and how far it turns, in this range."""
_MIN_SIN = 0.3
"""The shallowest entry angle a junction is sized for (~17 deg); flatter is clamped."""
_SPLIT_RATIO, _SPLIT_SIN = 0.85, math.sin(math.radians(30.0))
"""A child at least this thick against its parent, leaving shallower than this,
is a split of the stem (`_junction`): not welded as a branch."""
_MIN_WELD_SIN = math.sin(math.radians(12.0))
"""Below this entry angle a branch is not welded: it stays a capped tube (it
lies along its parent; the fillet, shrinking with the angle, is gone anyway)."""

# One resolution for the whole plant: the edge length, the plant's size over
# _EDGES_PER_SIZE x Smoothness. It sets every tube's side count (its
# circumference over the edge) and how far apart rings may be.
_EDGES_PER_SIZE = 50.0
"""Edges across the plant's size per unit of Smoothness (200 at the default 4)."""
_CATKIN_SIDES_PER_SMOOTHNESS = 2.5
"""A catkin tube's least sides per unit of Smoothness (10 at the default 4)."""
_DOME_WIDTH = 0.97
"""A catkin point narrower than this share of its widest is on a rounded end
(and gets a ring of its own)."""
_ORGAN_TURN_DEG = 2.0
"""A catkin point whose two segments turn more than this many degrees is on a
bend (and gets a ring of its own)."""
_SPAN_BARK, _SPAN_PLAIN = 3.0, 8.0
"""Longest ring spacing on a straight run, in edges: with bark (its lumps need
vertices to move) and without. Straight runs are one long quad otherwise."""
# Where a tube needs rings, at Smoothness 4 and scaled by 4 / Smoothness.
_BEND_DEG = 9.0
"""Most the centre line may turn between two rings."""
_TAPER_TOL = 0.03
"""Most the radius may leave a straight taper between two rings, in radii."""
_FILLET_DEG = 14.0
"""Most a fillet meridian may turn between two transition rings."""
_RING_PCT = 90.0
"""Which meridian sets a weld's ring count (percentile of their costs): a low one --
the crotch's sharply turning meridians would put a dozen rings through its
narrow channel, which every ring has to pass."""

_BALLS = ((1.5, 1.0), (1.1, 0.75), (0.75, 0.5))
"""The smallest rolling ball, as (edges, branch radii -- whichever is less):
a fillet much tighter than the mesh is sampled as a crease, and at a shallow
fork's apex its rings fold (1.5 edges took the worst ring-to-ring bend there
from 83 to 50 deg). Each junction takes the first that fits beside its
neighbours."""
_BALL_SIN_EXP = 1.0

_SDF_SEGS = 8
"""Segments of each local centre line a junction's distance field is built on."""


@dataclass
class SectionSpec:
    sides: int = 32
    """The most sides a tube takes (the thickest get it; see `_tube_sides`)."""
    lobe_count: int = 0
    lobe_depth: float = 0.0
    flatten: float = 0.0
    polygon: int = 0
    """Corners of a polygonal section (3 a triangle, 4 a square...), 0 for a
    round or lobed one. The shape is traced exactly whatever a tube's side
    count: its corners always land on vertices."""


@dataclass
class MeshOut:
    positions: np.ndarray
    face_sizes: np.ndarray = None   # (F,) corners per face
    face_loops: np.ndarray = None   # (sum sizes,) vertex index per corner
    corner_uv: np.ndarray = None    # (sum sizes, 2)
    point_attrs: dict = field(default_factory=dict)    # name -> (V,) array
    fallbacks: int = 0
    debug: dict = field(default_factory=dict)
    junctions: int = 0

    def face_lists(self) -> list[list[int]]:
        """The faces as vertex lists (for tests; the bridge writes the arrays)."""
        if self.face_sizes is None:
            return []
        ends = np.cumsum(self.face_sizes)
        return [self.face_loops[e - s:e].tolist() for s, e in zip(self.face_sizes, ends,
                                                                    strict=True)]


# ---------------------------------------------------------------------------
# the centre line
# ---------------------------------------------------------------------------

def _chains(n_verts: int, edges, tube: np.ndarray) -> dict[int, list[int]]:
    """tube id -> its vertex indices in order from its base (the lowest index,
    the first one the skeleton write-out emitted) to its tip."""
    adj = [[] for _ in range(n_verts)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    members: dict[int, list[int]] = {}
    for i in range(n_verts):
        members.setdefault(int(tube[i]), []).append(i)
    out = {}
    for t, verts in members.items():
        ends = [v for v in verts if len(adj[v]) <= 1]
        start = min(ends) if ends else min(verts)
        order, prev, cur = [start], -1, start
        while True:
            nxt = [k for k in adj[cur] if k != prev and int(tube[k]) == t]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            order.append(cur)
        out[t] = order
    return out


def _smooth(points: np.ndarray):
    """Dense samples of the auto-handle Bezier through `points`, with each
    sample's (segment, t) so per-point attributes can be interpolated."""
    n = len(points)
    if n < 2:
        return points.copy(), np.zeros(n, dtype=np.int64), np.zeros(n)
    seg = np.diff(points, axis=0)
    seg_len = _norm(seg, axis=1)
    unit = seg / np.maximum(seg_len, 1e-12)[:, None]
    unit[seg_len <= 1e-12] = 0.0
    tang = np.zeros_like(points)
    tang[:-1] += unit
    tang[1:] += unit
    tn = _norm(tang, axis=1)
    tang = tang / np.maximum(tn, 1e-12)[:, None]
    tang[tn <= 1e-12] = 0.0
    p0, p3 = points[:-1], points[1:]
    p1 = p0 + tang[:-1] * (seg_len / 3.0)[:, None]
    p2 = p3 - tang[1:] * (seg_len / 3.0)[:, None]
    t = np.linspace(0.0, 1.0, _DENSE, endpoint=False)
    u = 1.0 - t
    b0, b1, b2, b3 = u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3
    dense = (b0[None, :, None] * p0[:, None, :] + b1[None, :, None] * p1[:, None, :]
             + b2[None, :, None] * p2[:, None, :] + b3[None, :, None] * p3[:, None, :])
    dense = np.concatenate([dense.reshape(-1, 3), points[-1:]])
    seg_idx = np.concatenate([np.repeat(np.arange(n - 1), _DENSE), [n - 2]])
    seg_t = np.concatenate([np.tile(t, n - 1), [1.0]])
    return dense, seg_idx, seg_t



def _norm(x, axis=-1, keepdims=False):
    """np.linalg.norm along an axis, without its Python wrapper (the same
    sqrt(add.reduce(x * x)) for real arrays; ~200k calls a plant)."""
    return np.sqrt(np.add.reduce(x * x, axis=axis, keepdims=keepdims))

class _Line:
    """One tube's smoothed centre line: position, tangent and every point
    attribute as functions of arc length."""

    def __init__(self, points: np.ndarray, attrs: dict[str, np.ndarray]):
        self.dense, seg, t = _smooth(points)
        steps = _norm(np.diff(self.dense, axis=0), axis=1)
        self.s = np.concatenate(([0.0], np.cumsum(steps)))
        self.length = float(self.s[-1])
        # attributes at the dense samples, linear within each skeleton segment
        self.attrs = {}
        for name, vals in attrs.items():
            vals = np.asarray(vals, dtype=float)
            if len(vals) < 2:
                self.attrs[name] = np.full(len(self.dense), vals[0] if len(vals) else 0.0)
            else:
                nxt = vals[np.minimum(seg + 1, len(vals) - 1)]
                self.attrs[name] = vals[seg] * (1.0 - t) + nxt * t

    def at(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        # tangent from a small central difference along the dense curve; the
        # point and both difference samples in one interpolation per axis
        h = max(self.length * 1e-3, 1e-6)
        flat = s.reshape(-1)
        n = flat.size
        x = np.concatenate([flat, np.clip(flat - h, 0, self.length),
                            np.clip(flat + h, 0, self.length)])
        allp = np.stack([np.interp(x, self.s, self.dense[:, k]) for k in range(3)], axis=-1)
        pos = allp[:n].reshape(s.shape + (3,))
        tan = (allp[2 * n:] - allp[n:2 * n]).reshape(s.shape + (3,))
        tan /= np.maximum(_norm(tan, axis=-1, keepdims=True), 1e-12)
        return pos, tan

    def attr(self, name, s):
        return np.interp(np.clip(s, 0.0, self.length), self.s, self.attrs[name])


def _frames(tangents: np.ndarray, first_normal: np.ndarray) -> np.ndarray:
    """Parallel-transported normals along `tangents` (Rodrigues per step, in
    plain floats: thousands of tiny numpy calls were most of the mesher's time)."""
    tl = tangents.tolist()
    fn = first_normal.tolist()
    t0 = tl[0]
    d = fn[0] * t0[0] + fn[1] * t0[1] + fn[2] * t0[2]
    n = [fn[0] - t0[0] * d, fn[1] - t0[1] * d, fn[2] - t0[2] * d]
    ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
    n = [n[0] / ln, n[1] / ln, n[2] / ln]
    out = [n]
    for i in range(1, len(tl)):
        a, b = tl[i - 1], tl[i]
        ax = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        sn = math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2])
        if sn > 1e-9:
            ax = [ax[0] / sn, ax[1] / sn, ax[2] / sn]
            cs = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            ang = math.atan2(sn, cs)
            c, s = math.cos(ang), math.sin(ang)
            kd = ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]
            kx = [ax[1] * n[2] - ax[2] * n[1], ax[2] * n[0] - ax[0] * n[2],
                  ax[0] * n[1] - ax[1] * n[0]]
            n = [n[q] * c + kx[q] * s + ax[q] * kd * (1 - c) for q in range(3)]
        d = n[0] * b[0] + n[1] * b[1] + n[2] * b[2]
        n = [n[0] - b[0] * d, n[1] - b[1] * d, n[2] - b[2] * d]
        ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        n = [n[0] / ln, n[1] / ln, n[2] / ln]
        out.append(n)
    return np.array(out)


def _seam_normal(tangent: np.ndarray) -> np.ndarray:
    """Where a tube's u seam starts: its underside (world down projected off the
    tangent), or world X on a vertical stem."""
    down = np.array([0.0, 0.0, -1.0])
    n = down - tangent * float(np.dot(down, tangent))
    if float(np.linalg.norm(n)) < 0.2:
        n = np.array([1.0, 0.0, 0.0]) - tangent * float(tangent[0])
    return n


def _perpendicular(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Unit vectors perpendicular to each row of `t`, toward `ref` where it can."""
    e = ref[None, :] - t * (t @ ref)[:, None]
    bad = _norm(e, axis=1) < 1e-6
    if bad.any():
        alt = np.where(np.abs(t[bad, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        e[bad] = alt - t[bad] * np.sum(t[bad] * alt, axis=1, keepdims=True)
    return e / _norm(e, axis=1, keepdims=True)


def _section(spec: SectionSpec, sides: int) -> np.ndarray:
    """(sides, 2) unit cross-section: a polygon's own outline (corners at
    radius 1, on vertices), else the GN profile's maths -- radius 1 + cos(lobes
    x angle) x depth -- squashed by Flatten. `_snap_sides` makes `sides` a
    multiple of the corners / lobes, so the shape is sampled, not aliased: a
    square traced by a 14-gon was a 14-gon, five lobes on 8 sides vanished."""
    ang = 2.0 * math.pi * np.arange(sides) / sides
    if spec.polygon >= 3:
        wedge = 2.0 * math.pi / spec.polygon
        rad = math.cos(wedge / 2.0) / np.cos(np.mod(ang, wedge) - wedge / 2.0)
    else:
        rad = 1.0 + np.cos(spec.lobe_count * ang) * spec.lobe_depth
    return np.stack([np.cos(ang) * rad, np.sin(ang) * rad * (1.0 - spec.flatten)], axis=-1)


def _snap_sides(spec: SectionSpec, sides: int) -> int:
    """`sides` raised to what the section needs: even (a hole's border is), a
    multiple of a polygon's corners or of the lobes (so each lands on a
    vertex), and at least four vertices a lobe (a lobe needs a trough and a
    peak with a vertex between each)."""
    unit = 2
    if spec.polygon >= 3:
        unit = math.lcm(2, spec.polygon)
    elif spec.lobe_count > 0 and spec.lobe_depth != 0.0:
        unit = math.lcm(2, spec.lobe_count)
        sides = max(sides, 4 * spec.lobe_count)
    return int(unit * math.ceil(sides / unit))


def _tube_sides(spec: SectionSpec, width: float, edge: float) -> int:
    """Sides by thickness: the circumference over the plant's edge length --
    the same edge round every stem, so a pedicel is not built like a trunk --
    from 6 up to the section's most, snapped to what its shape needs."""
    top = max(6, spec.sides - spec.sides % 2)
    return _snap_sides(spec, int(min(top, max(6, 2 * round(math.pi * width
                                                            / max(edge, 1e-12))))))


# ---------------------------------------------------------------------------
# the weld surface: two tubes' distance fields and their round union
# ---------------------------------------------------------------------------

class _TubeField:
    """Signed distance to one tube near a junction: a local stretch of its
    centre line as straight segments with a linear radius. Both ends run on
    (a window cut from a longer stem), or the start is a flat cap (a branch
    starting inside its parent)."""

    def __init__(self, pts: np.ndarray, radii: np.ndarray, cap_start: bool):
        self.a, self.b = pts[:-1], pts[1:]
        self.ra, self.rb = radii[:-1], radii[1:]
        self.ab = self.b - self.a
        self.len2 = np.maximum(np.sum(self.ab * self.ab, axis=1), 1e-18)
        self.cap_start = cap_start
        self._dr = self.rb - self.ra
        # for picking each point's nearest segment by matrix products (below)
        self._o = pts[0].copy()
        al = self.a - self._o
        self._aT, self._abT = al.T.copy(), self.ab.T.copy()
        self._a2 = np.add.reduce(al * al, axis=1)
        self._a_ab = np.add.reduce(al * self.ab, axis=1)

    def __call__(self, P: np.ndarray):
        # Which segment is nearest, from two (n, 3) x (3, m) products -- |P - a|
        # and the projection expanded about a local origin -- instead of the
        # (n, m, 3) differences (most of the mesher's time); then the distance
        # and gradient computed directly, as ever, on that one segment.
        Q = P - self._o
        t = (Q @ self._abT - self._a_ab) / self.len2
        tc = np.minimum(np.maximum(t, 0.0), 1.0)
        tc[:, -1] = np.maximum(t[:, -1], 0.0)                          # runs on past the end
        if not self.cap_start:
            tc[:, 0] = np.minimum(t[:, 0], 1.0)                        # ...and before the start
        rel2 = np.add.reduce(Q * Q, axis=1)[:, None] - 2.0 * (Q @ self._aT) + self._a2
        dist = np.sqrt(np.maximum(rel2 - self.len2 * tc * (2.0 * t - tc), 0.0))
        tr = np.minimum(np.maximum(tc, 0.0), 1.0)
        d = dist - (self.ra + self._dr * tr)
        behind = t[:, 0] < 0.0 if self.cap_start else None
        if self.cap_start and behind.any():
            length = math.sqrt(float(self.len2[0]))
            h = -t[behind, 0] * length
            q = dist[behind, 0] - self.ra[0]
            d[behind, 0] = np.where(q > 0.0, np.sqrt(np.maximum(q, 0.0) ** 2 + h * h), h)
        k = d.argmin(axis=1)
        rows = np.arange(len(P))
        # exactly as the full evaluation computed them, on the chosen segment
        tk = tc[rows, k]
        diff = (P - self.a[k]) - tk[:, None] * self.ab[k]
        distk = np.sqrt(np.add.reduce(diff * diff, axis=1))
        trk = np.minimum(np.maximum(tk, 0.0), 1.0)
        dmin = distk - (self.ra[k] + self._dr[k] * trk)
        g = diff / np.maximum(distk, 1e-15)[:, None]
        if self.cap_start:
            cap = (k == 0) & behind
            if cap.any():
                length = math.sqrt(float(self.len2[0]))
                axis = self.ab[0] / length
                h = -t[cap, 0] * length
                qd = distk[cap] - self.ra[0]
                dmin[cap] = np.where(qd > 0.0, np.sqrt(np.maximum(qd, 0.0) ** 2 + h * h), h)
                q = np.maximum(qd, 0.0)
                radial = diff[cap] / np.maximum(distk[cap], 1e-15)[:, None]
                v = radial * q[:, None] - axis[None, :] * h[:, None]
                g[cap] = v / np.maximum(_norm(v, axis=1), 1e-15)[:, None]
        return dmin, g


class _TubePair:
    """Two `_TubeField`s evaluated together -- a weld's parent and branch,
    asked for at the same points every time: one pass over both tubes'
    segments, each tube's nearest picked among its own. The same values as
    calling each; half the calls (the field is evaluated ~20k times a plant,
    and each call is mostly fixed cost)."""

    def __init__(self, f1: _TubeField, f2: _TubeField):
        self.m1 = len(f1.a)
        self.m = self.m1 + len(f2.a)
        self._o = f1._o
        self.a = np.concatenate([f1.a, f2.a])
        self.ab = np.concatenate([f1.ab, f2.ab])
        al = self.a - self._o
        self._aT, self._abT = al.T.copy(), self.ab.T.copy()
        self._a2 = np.add.reduce(al * al, axis=1)
        self._a_ab = np.add.reduce(al * self.ab, axis=1)
        self.len2 = np.concatenate([f1.len2, f2.len2])
        self.ra = np.concatenate([f1.ra, f2.ra])
        self._dr = np.concatenate([f1._dr, f2._dr])
        self.parts = [(0, self.m1, f1.cap_start), (self.m1, self.m, f2.cap_start)]

    def __call__(self, P: np.ndarray):
        """(d1, g1, d2, g2): each tube's signed distance and gradient."""
        Q = P - self._o
        t = (Q @ self._abT - self._a_ab) / self.len2
        tc = np.minimum(np.maximum(t, 0.0), 1.0)
        for lo, hi, cap_start in self.parts:
            tc[:, hi - 1] = np.maximum(t[:, hi - 1], 0.0)              # runs on past the end
            if not cap_start:
                tc[:, lo] = np.minimum(t[:, lo], 1.0)                  # ...and before the start
        rel2 = np.add.reduce(Q * Q, axis=1)[:, None] - 2.0 * (Q @ self._aT) + self._a2
        dist = np.sqrt(np.maximum(rel2 - self.len2 * tc * (2.0 * t - tc), 0.0))
        tr = np.minimum(np.maximum(tc, 0.0), 1.0)
        d = dist - (self.ra + self._dr * tr)
        rows = np.arange(len(P))
        out = []
        for lo, hi, cap_start in self.parts:
            behind = t[:, lo] < 0.0 if cap_start else None
            if cap_start and behind.any():
                length = math.sqrt(float(self.len2[lo]))
                h = -t[behind, lo] * length
                q = dist[behind, lo] - self.ra[lo]
                d[behind, lo] = np.where(q > 0.0, np.sqrt(np.maximum(q, 0.0) ** 2 + h * h), h)
            k = d[:, lo:hi].argmin(axis=1) + lo
            tk = tc[rows, k]
            diff = (P - self.a[k]) - tk[:, None] * self.ab[k]
            distk = np.sqrt(np.add.reduce(diff * diff, axis=1))
            trk = np.minimum(np.maximum(tk, 0.0), 1.0)
            dmin = distk - (self.ra[k] + self._dr[k] * trk)
            g = diff / np.maximum(distk, 1e-15)[:, None]
            if cap_start:
                cap = (k == lo) & behind
                if cap.any():
                    length = math.sqrt(float(self.len2[lo]))
                    axis = self.ab[lo] / length
                    h = -t[cap, lo] * length
                    qd = distk[cap] - self.ra[lo]
                    dmin[cap] = np.where(qd > 0.0, np.sqrt(np.maximum(qd, 0.0) ** 2 + h * h), h)
                    qq = np.maximum(qd, 0.0)
                    radial = diff[cap] / np.maximum(distk[cap], 1e-15)[:, None]
                    v = radial * qq[:, None] - axis[None, :] * h[:, None]
                    g[cap] = v / np.maximum(_norm(v, axis=1), 1e-15)[:, None]
            out += [dmin, g]
        return tuple(out)


class _FieldOps:
    """Moving points onto a weld surface f = 0 (`__call__` -> f, grad f)."""

    def project(self, P, iters: int = 4, max_step: float = None):
        """Newton steps along the gradient: only for points already close to
        the surface (the gradient jumps between the tubes across a crease)."""
        max_step = max_step if max_step is not None else max(self.k, 1e-9)
        for _ in range(iters):
            f, g = self(P)
            gg = np.maximum(np.add.reduce(g * g, axis=1), 0.25)
            step = np.minimum(np.maximum(f / gg, -max_step), max_step)
            P = P - step[:, None] * g
        return P

    def cast(self, P, N, reach: float, iters: int = 7):
        """Each point moved along its own direction N (unit) to where it
        crosses the surface, within +-reach: bracketed, then regula falsi
        (Illinois). Points whose line never crosses stay put. Moving along a
        smooth surface's own normals keeps neighbours in order -- following
        the gradient folded the fillet where it jumps between the tubes."""
        f0, _ = self(P)
        dirn = np.where(f0 > 0.0, -1.0, 1.0)          # inside -> out, outside -> in
        lo = np.zeros(len(P))
        hi = np.full(len(P), reach)
        f_lo = f0 * dirn
        f_hi = self(P + (dirn * reach)[:, None] * N)[0] * dirn
        ok = (f_lo <= 0.0) & (f_hi >= 0.0)
        side = np.zeros(len(P))
        for _ in range(iters):
            t = np.where(ok, lo - f_lo * (hi - lo) / np.where(f_hi - f_lo > 1e-18, f_hi - f_lo,
                                                               1e-18), 0.0)
            t = np.clip(t, lo, hi)
            f_t = self(P + (dirn * t)[:, None] * N)[0] * dirn
            below = f_t < 0.0
            # Illinois: halve the stale end's value when the same side repeats
            f_hi = np.where(below & (side < 0), 0.5 * f_hi, f_hi)
            f_lo = np.where(~below & (side > 0), 0.5 * f_lo, f_lo)
            lo, f_lo = np.where(below, t, lo), np.where(below, f_t, f_lo)
            hi, f_hi = np.where(below, hi, t), np.where(below, f_hi, f_t)
            side = np.where(below, -1.0, 1.0)
        t = lo - f_lo * (hi - lo) / np.where(f_hi - f_lo > 1e-18, f_hi - f_lo, 1e-18)
        t = np.where(ok, np.clip(t, lo, hi), 0.0)
        return P + (dirn * t)[:, None] * N, ok


def _gate(P, root, axis, toward):
    """1 up to _GATE_NEAR deg off a branch's azimuth round its parent's axis,
    easing to 0 at _GATE_FAR."""
    rel = P - root
    rel = rel - np.outer(rel @ axis, axis)
    cos_az = (rel @ toward) / np.maximum(np.sqrt(np.add.reduce(rel * rel, axis=1)), 1e-15)
    far = math.cos(math.radians(_GATE_FAR))
    w = np.minimum(np.maximum((cos_az - far) / (math.cos(math.radians(_GATE_NEAR)) - far),
                              0.0), 1.0)
    return w * w * (3.0 - 2.0 * w)


class _WeldField(_FieldOps):
    """The weld surface f = 0 of one junction: the round union (rolling ball of
    radius k) of the parent and branch tubes. On the parent's far half (the
    side facing away from the branch) the branch's distance is faded out: a
    branch nearly as thick as its parent starts with its surface just under
    the parent's far side, and a plain round union swells it there (a
    volume's morphological fillet does not). Every real crease lies on the
    near half. Hiding by *burial* instead -- the nearest surface point lying
    inside the other tube -- also hid shallow creases, and jumped: the zero
    set got cliffs."""

    def __init__(self, parent: _TubeField, child: _TubeField, k: float,
                 root: np.ndarray, axis: np.ndarray, toward: np.ndarray):
        self.parent, self.child, self.k = parent, child, k
        self.root, self.axis = root, axis       # a point on the parent's axis, its direction
        self.toward = toward                    # the branch's azimuth, off the parent's axis
        self._pair = _TubePair(parent, child)

    def gated(self, P):
        """(a, grad a, a used, b, grad b, b used): the distances to parent and
        branch, raw and as the union reads them."""
        a, ga, b, gb = self._pair(P)
        w = _gate(P, self.root, self.axis, self.toward)
        return a, ga, a, b, gb, b + (1.0 - w) * 4.0 * self.k

    def owner(self, P):
        """Which tube each point's surface belongs to: 0 the parent, 1 the
        branch (whichever it is nearer, as the union reads them)."""
        a, _, _, b, _, _ = self.gated(P)
        return (a > b).astype(np.int64)

    def __call__(self, P):
        a_raw, ga, a, b_raw, gb, b = self.gated(P)
        k = self.k
        ua, ub = np.maximum(k - a, 0.0), np.maximum(k - b, 0.0)
        ln = np.sqrt(ua * ua + ub * ub)
        f = np.maximum(k, np.minimum(a, b)) - ln
        both = (ua > 0.0) & (ub > 0.0)
        g = np.where((a <= b)[:, None], ga, gb)
        if both.any():
            gm = (ua[both, None] * ga[both] + ub[both, None] * gb[both]) / ln[both, None]
            g[both] = gm
        # The fillet only ever adds material: inside either tube is inside,
        # whatever the exposure gate made of it (hiding both buried sheets at
        # once turned the solid inside out near a thick branch's start).
        inside_a = a_raw < f
        inside_b = b_raw < np.minimum(f, a_raw)
        f = np.where(inside_a, a_raw, f)
        g = np.where(inside_a[:, None], ga, g)
        f = np.where(inside_b, b_raw, f)
        g = np.where(inside_b[:, None], gb, g)
        return f, g

class _MultiField(_FieldOps):
    """The weld surface of a cluster: the round union (rolling ball of radius
    k) of several tubes -- a stem and every branch leaving it together, and
    their branches. A branch's distance is faded out on its parent's far side
    (`gates[i]` = (point on the parent's axis, the axis, the branch's azimuth),
    None for the root stem), as in the single weld."""

    def __init__(self, tubes: list, gates: list, k: float, exact: bool = False):
        self.tubes, self.gates, self.k = tubes, gates, k
        self.exact = exact                      # a true rolling ball (`_ball_blend`)

    def distances(self, P):
        """(raw distances (m, n), gradients (m, n, 3), distances as the union
        reads them (m, n)), per tube."""
        D, G, U = [], [], []
        for tf, gate in zip(self.tubes, self.gates, strict=True):
            d, g = tf(P)
            D.append(d)
            G.append(g)
            U.append(d if gate is None else d + (1.0 - _gate(P, *gate)) * 4.0 * self.k)
        return np.stack(D), np.stack(G), np.stack(U)

    def __call__(self, P):
        D, G, U = self.distances(P)
        k = self.k
        cols = np.arange(len(P))
        if self.exact and len(U) >= 2:
            f, g = _ball_blend(U, G, k)
        else:
            u = np.maximum(k - U, 0.0)
            ln = np.sqrt(np.sum(u * u, axis=0))
            near = np.argmin(U, axis=0)
            f = np.maximum(k, U[near, cols]) - ln
            g = G[near, cols]
            blend = ln > 0.0
            if blend.any():
                g[blend] = np.einsum("nm,nmk->mk", u[:, blend], G[:, blend]) / ln[blend, None]
        # the fillet only ever adds material: inside any tube is inside
        inner = np.argmin(D, axis=0)
        d_in = D[inner, cols]
        inside = d_in < f
        f = np.where(inside, d_in, f)
        g = np.where(inside[:, None], G[inner, cols], g)
        return f, g

    def owner(self, P):
        """Which tube each point's surface belongs to: the nearest, as the
        union reads them."""
        return np.argmin(self.distances(P)[2], axis=0)


def _ball_blend(U, G, k: float):
    """The two nearest surfaces (distances U, gradients G, one row per tube)
    joined by a ball of radius k rolled along their crease: (f, grad f).
    The plain round union (k - |(k-a, k-b)|) is that ball only where the two
    meet square; in the sharp crotch above a rising branch it rounded off half
    as far, and in the wide one below it swelled. Here the distance to the
    ball's centre is measured with the angle between the two surfaces (c, the
    cosine of the angle between their gradients), the ball touching each where
    its centre's foot lies (its radius sized by the angle
    so it reaches k along each)."""
    order = np.argsort(U, axis=0)
    cols = np.arange(U.shape[1])
    i1, i2 = order[0], order[1]
    a, b = U[i1, cols], U[i2, cols]
    g1, g2 = G[i1, cols], G[i2, cols]
    f, g = a.copy(), g1.copy()
    # (a slit between two nearly facing surfaces: the ball's centre would sit
    # arbitrarily deep -- held at 0.9)
    c = np.clip(np.sum(g1 * g2, axis=1), -0.9, 0.9)
    # The ball sized by the crotch's angle so it touches each surface k
    # from the crease (radius k tan(angle / 2)): a fixed radius sat deep in
    # a sharp crotch and curved a stem radius up the trunk; this way the
    # trunk stays straight to within k of the branch and a sharp crotch
    # gets a tight bend, a wide one a gentle, short one.
    kk = k * np.sqrt((1.0 + c) / (1.0 - c))
    da, db = a - kk, b - kk
    ca, cb = da - c * db, db - c * da
    sel = (ca < 0.0) & (cb < 0.0)          # the ball's sector: inside both tangent lines
    if sel.any():
        s2 = 1.0 - c[sel] ** 2
        dist = np.sqrt(np.maximum((da[sel] ** 2 + db[sel] ** 2 - 2.0 * c[sel] * da[sel] * db[sel])
                                  / s2, 1e-30))
        f[sel] = kk[sel] - dist
        g[sel] = -(ca[sel, None] * g1[sel] + cb[sel, None] * g2[sel]) / (s2 * dist)[:, None]
    return f, g


def _local_field(line: _Line, s_lo: float, s_hi: float, cap_start: bool) -> _TubeField:
    s = np.linspace(max(0.0, s_lo), min(line.length, s_hi), _SDF_SEGS + 1)
    pts, _ = line.at(s)
    return _TubeField(pts, line.attr("width", s), cap_start)


# ---------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------

@dataclass
class _Tube:
    tid: int
    parent: int
    line: _Line
    sides: int
    mean_width: float
    ring_s: np.ndarray = None
    base_index: int = 0        # first vertex of ring 0
    frame_n: np.ndarray = None  # per ring
    frame_b: np.ndarray = None
    tan: np.ndarray = None
    centre: np.ndarray = None
    hole: set = field(default_factory=set)   # (row, col) quads removed
    welded: bool = False       # joined to its parent (no base cap)
    plain_rings: bool = False  # too thin to show bark: straight runs ring by bends
    light: bool = False        # a light stalk: keeps `_LIGHT_STALK_SIDES`
    v_curve: tuple = None      # (s, v) samples: V, arc length in texture tiles
    tex: float = 1.0           # the texture's world size (one tile), the whole plant's
    raw: tuple = None          # (points, attrs) from the skeleton, before smoothing
    section: SectionSpec = None  # the shape swept along it (its branching level's)
    domes: tuple = ()          # arc lengths of extra rings (a catkin's rounded ends)

    def v_at(self, s):
        return np.interp(np.clip(s, 0.0, self.line.length), *self.v_curve)

    def u_span(self, s):
        """Tiles once round the tube at s: its circumference over the tile."""
        return 2.0 * math.pi * self.line.attr("width", s) / self.tex


class _Builder:
    """Vertices in numpy blocks (a tube's rings, a transition ring) and faces
    as blocks of quads plus loose polygons (caps, transition faces), each with
    its corner UVs."""

    def __init__(self, names):
        self.names = list(names)
        self._pos, self._attr = [], {n: [] for n in self.names}
        self.count = 0
        self.frozen_pos = None
        self.frozen_attr = None
        self._quads, self._quad_uv = [], []
        self._polys, self._poly_uv = [], []

    def add_block(self, pos, attrs) -> int:
        start = self.count
        self._pos.append(np.asarray(pos, dtype=float).reshape(-1, 3))
        for n in self.names:
            self._attr[n].append(np.asarray(attrs[n], dtype=float))
        self.count += len(self._pos[-1])
        return start

    def freeze(self) -> None:
        """Snapshot every vertex so far (the tubes) for the transitions to read."""
        self.frozen_pos = np.concatenate(self._pos) if self._pos else np.zeros((0, 3))
        self.frozen_attr = {n: (np.concatenate(v) if v else np.zeros(0))
                            for n, v in self._attr.items()}

    def quads(self, quads: np.ndarray, uvs: np.ndarray) -> None:
        self._quads.append(np.asarray(quads, dtype=np.int64).reshape(-1, 4))
        self._quad_uv.append(np.asarray(uvs, dtype=float).reshape(-1, 4, 2))

    def poly(self, verts, uvs) -> None:
        self._polys.append(list(verts))
        self._poly_uv.extend(uvs)

    def result(self):
        pos = np.concatenate(self._pos) if self._pos else np.zeros((0, 3))
        attrs = {n: (np.concatenate(v) if v else np.zeros(0)) for n, v in self._attr.items()}
        quads = np.concatenate(self._quads) if self._quads else np.zeros((0, 4), np.int64)
        quad_uv = np.concatenate(self._quad_uv) if self._quad_uv else np.zeros((0, 4, 2))
        sizes = np.concatenate([np.full(len(quads), 4, np.int64),
                                np.array([len(p) for p in self._polys], np.int64)])
        loops = np.concatenate([quads.ravel(),
                                np.array([v for p in self._polys for v in p], np.int64)])
        uv = np.concatenate([quad_uv.reshape(-1, 2),
                             np.array(self._poly_uv, float).reshape(-1, 2)])
        return pos, attrs, sizes, loops, uv


def _adaptive_rings(line: _Line, start: float, *, q: float, edge: float, bark: float,
                    twist_step: float) -> np.ndarray:
    """Ring positions (arc length) from `start` to the tip: a ring only where
    the tube needs one. Greedy along the dense centre line: the next ring goes
    at the last sample before the line has turned more than the bend
    tolerance, the radius has left a straight taper, or the span has grown past
    what the bark / the twist allow. Never closer than half an edge (finer is
    below the plant's own resolution)."""
    floor = 0.5 * edge
    s = line.s
    n = len(s)
    if line.length <= start + floor:
        return np.array([start, line.length]) if line.length > start else np.array([start])
    tan = np.gradient(line.dense, axis=0)
    tan /= np.maximum(_norm(tan, axis=1, keepdims=True), 1e-12)
    w = line.attrs["width"]
    cos_bend = math.cos(math.radians(_BEND_DEG * q))
    span_max = (_SPAN_BARK if bark > 1e-3 else _SPAN_PLAIN) * edge
    rings = [start]
    i0 = int(np.searchsorted(s, start, side="right"))
    last_s, last_i = start, max(i0 - 1, 0)
    last_t = tan[last_i]
    last_w = float(np.interp(start, s, w))
    i = i0
    while i < n:
        span = s[i] - last_s
        ok = span <= max(floor, min(span_max, twist_step))
        ok = ok and float(tan[i] @ last_t) >= cos_bend
        if ok and i - last_i > 1:
            seg = slice(last_i + 1, i)
            frac = (s[seg] - last_s) / max(span, 1e-12)
            lin = last_w + (float(w[i]) - last_w) * frac
            ok = float(np.max(np.abs(w[seg] - lin))) <= _TAPER_TOL * q * max(last_w, float(w[i]))
        if ok or span < floor:
            i += 1
            continue
        j = i - 1 if s[i - 1] - last_s >= floor else i
        rings.append(float(s[j]))
        last_s, last_i, last_t, last_w = float(s[j]), j, tan[j], float(w[j])
        i = j + 1
    if line.length - rings[-1] < 0.5 * floor and len(rings) > 1:
        rings[-1] = line.length
    else:
        rings.append(line.length)
    return np.array(rings)


def build_stems(verts, edges, attrs: dict, *, section, smoothness: float = 4.0,
                tiles: int = 2, twist: float = 0.0, fillet: float = 1.0,
                bark: float = 0.0, weld: bool = True, hidden=frozenset(),
                uv_size: float = 0.0, light_stalks: bool = False) -> MeshOut:
    """The whole stem mesh from the skeleton write-out (`bridge._stem_arrays`):
    `verts` (V,3), `edges`, `attrs` name -> V values (needs width, tube,
    parent_tube, flare). `fillet` is the Weld Range (the fillet radius, in
    branch radii); `bark` only decides how finely straight runs are ringed.
    Returns positions, faces, per-corner UVs and every input attribute plus
    `pg_u` / `pg_v` per output vertex.

    `light_stalks` builds stalks (pedicels: no phytomer node) with
    `_LIGHT_STALK_SIDES` sides, for a plant with tens of thousands of flowers,
    and tubes too thin to show bark drop its ring spacing on straight runs.
    Bends and welds stay as dense. Stalks still go through the junction pass: a
    tube taken out before it changes which junctions cluster, and one huge
    cluster weld cost minutes (Laburnum: 0 clusters, 197 s).

    `section` is one `SectionSpec` for every tube, or a dict of them by
    branching level -- a tube takes its `level` attribute's (else its axis
    order's; the deepest level's past the stack). `weld` False builds every
    junction as a capped tube starting inside its parent (Weld Junctions off).
    `hidden` tube ids are laid out but not built (a branching level's Show
    off); a tube on a hidden one, or itself
    hidden, is never welded -- its base is capped where it starts."""
    verts = np.asarray(verts, dtype=float)
    n_in = len(verts)
    if n_in == 0:
        return MeshOut(positions=np.zeros((0, 3)), face_sizes=np.zeros(0, np.int64),
                       face_loops=np.zeros(0, np.int64), corner_uv=np.zeros((0, 2)))
    tube_id = np.asarray(attrs["tube"], dtype=np.int64)
    parent_id = np.asarray(attrs["parent_tube"], dtype=np.int64)
    width = np.asarray(attrs["width"], dtype=float)
    flare = np.asarray(attrs.get("flare", np.zeros(n_in)), dtype=float)
    names = [n for n in attrs if len(attrs[n]) == n_in]
    int_names = {n for n in names if np.asarray(attrs[n]).dtype.kind in "iub"}

    unflared = width[flare < 1e-4]
    widest = float(unflared.max()) if len(unflared) else float(width.max())
    q = 4.0 / max(float(smoothness), 1.0)
    size = max(float(np.max(np.ptp(verts, axis=0))) if n_in > 1 else 0.0, 20.0 * widest, 1e-4)
    edge = size / (_EDGES_PER_SIZE * max(float(smoothness), 0.25))

    # One texel size over the whole plant: Tiles Around is how many times the
    # bark repeats round the thickest stem's base, and every tube is textured
    # at that world scale -- U round it its circumference in tiles (a thin twig
    # takes a fraction of a tile), V its arc length. (U 0..Tiles on every tube
    # shrank the bark with the branch: a twig's grain was many times finer
    # than the trunk's.) `uv_size` > 0 sets that scale outright -- the world
    # size of one tile -- so it no longer moves when the thickest stem does.
    tex = float(uv_size) if uv_size > 0.0 else 2.0 * math.pi * widest / max(tiles, 1)

    def set_line(tb: _Tube, pts, a) -> None:
        tb.raw = (pts, a)
        tb.line = _Line(pts, a)
        tb.tex = tex
        tb.v_curve = (tb.line.s, tb.line.s / tex)

    def section_of(order: int) -> SectionSpec:
        if isinstance(section, SectionSpec):
            return section
        return section[min(max(int(order), 0), max(section))]

    chains = _chains(n_in, edges, tube_id)
    parent_of = {t: int(parent_id[order[0]]) for t, order in chains.items()}
    base_width = _dominant_forks(chains, parent_of, verts, width)
    # Where welds compete, which goes first: a stem's continuation (a fork or
    # a sympodial relay -- it keeps its parent's branch order), then a side
    # branch, then a stalk (a pedicel or a catkin's: no phytomer node, so the
    # skeleton writes it with node_index -1); thickness only between equals.
    node_index = np.asarray(attrs.get("node_index", np.zeros(n_in)), dtype=np.int64)
    order_of = np.asarray(attrs.get("order", np.zeros(n_in)), dtype=np.int64)

    def _node_order(order):
        """A tube's own branch order, read at its phytomer nodes (a stalk
        joined on at its tip carries the next order); None for a stalk."""
        order = np.asarray(order)
        nodes = order[node_index[order] >= 0]
        return int(np.min(order_of[nodes])) if len(nodes) else None

    def _rank(t):
        own = _node_order(chains[t])
        if own is None:
            return 0                                  # a stalk
        par = chains.get(parent_of[t])
        if par is not None and own == _node_order(par):
            return 2                                  # the stem carrying on
        return 1                                      # a side branch
    rank = {t: _rank(t) for t in chains}
    tubes: dict[int, _Tube] = {}
    attr_arrays = {n: np.asarray(attrs[n], dtype=float) for n in names}   # once, not per tube
    for t, order in chains.items():
        pts = verts[order]
        a = {n: attr_arrays[n][order] for n in names}
        if t in base_width:
            a["width"][0] = base_width[t]
        w_t = a["width"]
        mean_w = float(np.mean(w_t))
        # Sides by the width a branch has where it leaves its parent, not its
        # mean: a fork as thick as its parent at its base but tapering got a
        # third of the parent's sides, and its weld had to squeeze 26 border
        # vertices down to 8 -- a fan of slivers in the crotch.
        base_w = float(np.max(w_t[: max(2, len(order) // 4)]))
        ref_w = max(mean_w, 0.75 * base_w) if parent_of[t] >= 0 else mean_w
        # The level of the tube's own axis, read at its tip (a chain the
        # dominant-fork swap joined starts on its parent's vertices).
        # its level's section (the bridge writes each vertex's `level`; bare
        # skeletons -- the tests -- key by `order`, the chain's level)
        key = "level" if "level" in a else "order"
        spec = section_of(a[key][-1] if key in a else 0)
        sides = _tube_sides(spec, ref_w, edge)
        # (most of its vertices: a stem a bunch hangs from shares one with it)
        organ = "catkin" in a and float(np.mean(a["catkin"])) > 0.5
        if organ:
            # A catkin is an organ, seen as close as a leaf: its sides by the
            # Smoothness alone, not by its share of the plant's size (on a 5 m
            # hazel the edge is ~2.7 cm, a catkin's whole girth -- 6 sides).
            sides = max(sides, _snap_sides(spec, int(round(_CATKIN_SIDES_PER_SMOOTHNESS
                                                           * float(smoothness)))))
        light = light_stalks and rank[t] == 0 and not organ
        if light:
            sides = _LIGHT_STALK_SIDES
        tubes[t] = _Tube(t, parent_of[t], None, sides, mean_w)
        tubes[t].light = light
        tubes[t].section = spec
        # Bark lumps need rings only where a tube is thick enough to show them:
        # a thin spur rings by its bends, and its straight runs stay long.
        tubes[t].plain_rings = light_stalks and 2.0 * math.pi * mean_w < _SPAN_BARK * edge
        set_line(tubes[t], pts, a)
        if organ:
            # ... and a ring at each of its own points where it narrows -- its
            # rounded ends, whose points are spaced to make a dome -- or bends
            # (where it turns down to hang): the plant's ring spacing, ~1.3 cm
            # on a 5 m hazel, put none on a 4 mm dome (a point) and one on the
            # turn (an elbow). Its straight runs keep the plant's spacing.
            seg = np.diff(pts, axis=0)
            seg = seg / np.maximum(_norm(seg, axis=1, keepdims=True), 1e-12)
            turn = np.zeros(len(pts))
            turn[1:-1] = np.degrees(np.arccos(np.clip(np.sum(seg[:-1] * seg[1:], axis=1),
                                                      -1.0, 1.0)))
            narrow = np.nonzero((w_t < _DOME_WIDTH * float(np.max(w_t)))
                                | (turn > _ORGAN_TURN_DEG))[0]
            s_pts = tubes[t].line.s[np.minimum(np.arange(len(pts)) * _DENSE,
                                               len(tubes[t].line.s) - 1)]
            tubes[t].domes = tuple(float(s_pts[i]) for i in narrow)
            # Its V by the length over its surface, not along its centre line:
            # on a rounded end the surface turns in toward the pole, so a short
            # step along the line is a long one over the dome, and V by the
            # line stretched the texture there. (A stem's slow taper would move
            # its V by parts in a million: left as it was.)
            line = tubes[t].line
            w_d = line.attrs["width"]
            step = np.hypot(np.diff(line.s), np.diff(w_d))
            meridian = np.concatenate(([0.0], np.cumsum(step)))
            tubes[t].v_curve = (line.s, meridian / tex)

    def side_len(tb: _Tube, s: float) -> float:
        return 2.0 * math.pi * float(tb.line.attr("width", s)) / tb.sides

    def mirrored(a: int, b: int, ja: dict, jb: dict) -> bool:
        """Whether two branches of one stem take a single weld each, in their
        own halves of it: roughly opposite (more than 120 deg apart), and the
        stem's side between them -- which each hole then leaves to the stem --
        barely lifted by either's fillet at the last ball: by k - (r_p - r_c),
        at most _PAIR_LIFT of the stem's radius (0.05 for a branch 0.7 as wide;
        nearly as wide as the stem, 0.4 -- the fillets meet, and the pair keeps
        the hub)."""
        if tubes[a].parent != tubes[b].parent:
            return False
        par = tubes[tubes[a].parent]
        _, t = par.line.at(np.array([0.5 * (ja["s0"] + jb["s0"])]))
        flat = []
        for j in (ja, jb):
            f = j["dir"] - t[0] * float(j["dir"] @ t[0])
            n = float(np.linalg.norm(f))
            if n < 1e-9:
                return False
            flat.append(f / n)
        if float(flat[0] @ flat[1]) > math.cos(math.radians(120.0)):
            return False
        b_e, b_r = _BALLS[-1]
        for j in (ja, jb):
            k_last = max(j["k_base"], min(b_e * edge, b_r * j["r_c"])) if j["k_base"] > 0 else 0.0
            if k_last - (j["r_p"] - j["r_c"]) > _PAIR_LIFT * j["r_p"]:
                return False
        return True

    # Each junction first tries the larger rolling ball -- smooth through the
    # crotch, where a ball smaller than the mesh folds the rings -- and only
    # where that weld would not fit (a sibling's, its parent's own weld, a
    # neighbour's hole) retries with the small one: a densely forked plant
    # lost a quarter of its welds to the larger ball everywhere.
    cache: dict = {}
    level: dict[int, int] = {}          # child -> index into _BALLS
    cluster_list: list[dict] = []       # branches welded as one (found on the first pass)
    hub_sides: dict[int, int] = {}      # tube -> side count a hub pair holds it to
    cluster_start: dict[int, float] = {}

    def weld_sides(tb: _Tube) -> int:
        """A welded tube's side floor: `_WELD_MIN_SIDES`, up to its section's
        most, snapped to its shape."""
        if tb.light:
            return tb.sides
        top = max(6, tb.section.sides - tb.section.sides % 2)
        return _snap_sides(tb.section, max(tb.sides, min(top, _WELD_MIN_SIDES)))
    n_attempts = len(_BALLS) + 8
    for _attempt in range(n_attempts):
        for tb in tubes.values():
            tb.hole = set()
        # --- junctions: the weld surface, the hole it needs, where the branch starts
        junction = {}   # child tid -> dict
        by_parent: dict[int, list[int]] = {}
        fallback: set[int] = set()
        why: dict[int, str] = {}        # child -> why it was not welded (debug)
        blocked_by: dict[int, int] = {}  # child -> the tube whose weld took its place (debug)
        for t, tb in tubes.items():
            par = tubes.get(tb.parent)
            if par is None or tb.line.length <= 0:
                continue
            if t in hidden or tb.parent in hidden:
                fallback.add(t)
                why[t] = "hidden" if t in hidden else "on a hidden tube"
                continue
            if not weld:
                fallback.add(t)
                why[t] = "weld off"
                continue
            ball = _BALLS[level.get(t, 0)]
            if (t, ball) not in cache:
                cache[t, ball] = _junction(par, tb, fillet, edge, ball)
            j = cache[t, ball]
            if isinstance(j, dict):
                j = dict(j)                  # later stages write rows/cells into it
            if isinstance(j, str):
                fallback.add(t)
                why[t] = j
                continue
            junction[t] = j
            by_parent.setdefault(tb.parent, []).append(t)
        # A welded branch and the stem it joins need enough sides for a smooth
        # join: an n-sided tube bends 360/n deg from face to face, so 6 sides
        # showed 60 deg creases running up every weld. Only welded tubes pay
        # for it; the rest keep their thickness-based count.
        for t in junction:
            for w in (t, tubes[t].parent):
                tubes[w].sides = weld_sides(tubes[w])
        for t, n_s in hub_sides.items():
            tubes[t].sides = n_s

        # --- clusters: junctions whose welds would run into each other are
        # welded as one (a branch splitting right after it leaves its stem, two
        # thick branches whose holes overlap) -- found once, at the large ball
        if _attempt == 0:
            structure = {c: j for c, j in junction.items() if rank[c] >= 1}
            for root_t, members in _find_clusters(tubes, structure):
                if any(m in cluster_start for m in members) or root_t in cluster_start:
                    continue
                start_root = junction[root_t]["start"] if root_t in junction else 0.0
                members = list(members)
                cl = None
                for _grow in range(4):
                    cl = _cluster_setup(tubes[root_t], members, tubes, junction, start_root,
                                        edge)
                    if isinstance(cl, str):
                        break
                    near = [c for c in by_parent.get(root_t, []) if c not in members
                            and junction[c]["lo"] < cl["s_hi"] + edge
                            and junction[c]["hi"] > cl["s_lo"] - edge]
                    near += [c for c in junction if c not in members
                             and tubes[c].parent in cl["starts"]
                             and junction[c]["lo"] < cl["starts"][tubes[c].parent] + edge]
                    if not near:
                        break
                    members += near
                    cl = None
                if not isinstance(cl, dict):
                    continue
                if any(o["root"] == root_t and o["s_lo"] < cl["s_hi"] + edge
                       and cl["s_lo"] < o["s_hi"] + edge for o in cluster_list):
                    continue
                if _hub_pair(cl, tubes):
                    # A single weld each, in its half of the stem, if the side
                    # between them is barely lifted (`mirrored`) or their welds
                    # at the last ball keep clear of each other anyway (a pair
                    # at two heights); else the hub.
                    last = []
                    for m in cl["members"]:
                        key = (m, _BALLS[-1])
                        if key not in cache:
                            cache[key] = _junction(tubes[tubes[m].parent], tubes[m], fillet,
                                                   edge, _BALLS[-1])
                        last.append(cache[key])
                    clear = all(isinstance(x, dict) for x in last)
                    if clear:
                        (a0, a1, ra), (b0, b1, rb) = (_weld_zone(tubes[m], x) for m, x
                                                      in zip(cl["members"], last, strict=True))
                        clear = _segment_distance(a0, a1, b0, b1) >= ra + rb
                    if clear or mirrored(*cl["members"], *(junction[m] for m in cl["members"])):
                        continue
                cluster_list.append(cl)
                cluster_start.update(cl["starts"])
            # A hub pair's stem and branches take one count: the stem's, to the
            # nearest multiple of 4 (each ring's four sides meet the cube's).
            # Its stretch of stem reaches past the crotches by _HUB_PAD stem
            # radii each way (no further than halfway to the next weld): cut
            # just past the higher branch's crotch, the rows from the cube up to
            # the stem's ring bunched there into slivers that folded.
            for cl in cluster_list:
                if _hub_pair(cl, tubes):
                    root_t = cl["root"]
                    pad = _HUB_PAD * cl["r_root"]
                    others = [(junction[c]["lo"], junction[c]["hi"])
                              for c in by_parent.get(root_t, []) if c not in cl["members"]]
                    others += [(o["s_lo"], o["s_hi"]) for o in cluster_list
                               if o is not cl and o["root"] == root_t]
                    below = [hi for lo, hi in others if hi <= cl["s_lo"]]
                    above = [lo for lo, hi in others if lo >= cl["s_hi"]]
                    floor_ = max(below, default=None)
                    floor_ = (0.5 * (floor_ + cl["s_lo"]) if floor_ is not None
                              else (junction[root_t]["start"] if root_t in junction else 0.0)
                              + edge)
                    ceil_ = min(above, default=None)
                    ceil_ = (0.5 * (ceil_ + cl["s_hi"]) if ceil_ is not None
                             else tubes[root_t].line.length - edge)
                    cl["s_lo"] = max(cl["s_lo"] - pad, min(floor_, cl["s_lo"]))
                    cl["s_hi"] = min(cl["s_hi"] + pad, max(ceil_, cl["s_hi"]))
            for cl in cluster_list:
                if _hub_pair(cl, tubes):
                    n_s = 4 * max(2, int(round(tubes[cl["root"]].sides / 4.0)))
                    for t in [cl["root"], *cl["members"]]:
                        hub_sides[t] = n_s
                        tubes[t].sides = n_s
        for cl in cluster_list:
            for m in cl["members"]:
                junction.pop(m, None)
                fallback.discard(m)
                why.pop(m, None)
                kids_of = by_parent.get(tubes[m].parent)
                if kids_of and m in kids_of:
                    kids_of.remove(m)

        # --- siblings whose welds would run into each other ----------------------
        # Each weld is built from its own branch and parent only: where two
        # branches leave together (a fork, a whorl), their fillets would pass
        # through each other. Only one is welded there -- by rank (`rank`), then the
        # thicker; the other stays a capped tube. (By thickness
        # alone, a raceme's first pedicel -- as thick as its stem at Pedicel
        # Width ~1 -- took the weld from the sympodial relay beside it, so the
        # stem's own continuation was the capped tube.)
        def outranks(a, b, junction=junction) -> bool:
            return (rank[a], junction[a]["r_c"]) > (rank[b], junction[b]["r_c"])

        for kids in by_parent.values():
            zone = {c: _weld_zone(tubes[c], junction[c]) for c in kids}
            for c in sorted(kids, key=lambda c: (-rank[c], -junction[c]["r_c"])):
                if c in fallback:
                    continue
                for o in kids:
                    if o == c or o in fallback or outranks(o, c):
                        continue
                    (a0, a1, ra), (b0, b1, rb) = zone[c], zone[o]
                    # (the capsules of two level, opposite branches rising at
                    # 45 deg cross inside their stem, where no weld is)
                    if _segment_distance(a0, a1, b0, b1) < ra + rb \
                            and not mirrored(c, o, junction[c], junction[o]):
                        fallback.add(o)
                        why[o] = "sibling"
                        blocked_by[o] = c
        for pt in by_parent:
            by_parent[pt] = [c for c in by_parent[pt] if c not in fallback]

        # --- clusters of footprints; rows near-square to the parent's faces -----
        clusters: dict[int, list] = {}
        for pt, kids in by_parent.items():
            par = tubes[pt]
            ivs = []
            # the parent's own tube starts past its weld to *its* parent: a hole
            # below that would be cut into the weld surface, not the tube
            par_start = junction[pt]["start"] if pt in junction and pt not in fallback \
                else cluster_start.get(pt, 0.0)
            for c in kids:
                j = junction[c]
                h = side_len(par, j["s0"])
                off_parent = (j["lo"] < par_start + 0.5 * h
                              or j["hi"] > par.line.length - 0.5 * h)
                too_short = tubes[c].line.length < j["start"] + edge
                in_band = any(cl["root"] == pt and j["lo"] < cl["s_hi"] + edge
                              and j["hi"] > cl["s_lo"] - edge for cl in cluster_list)
                if in_band:
                    fallback.add(c)
                    why[c] = "in a cluster's weld"
                    continue
                if off_parent or too_short or _columns(j, par) >= par.sides - 1:
                    fallback.add(c)
                    why[c] = ("in parent's weld" if j["lo"] < par_start + 0.5 * h else
                              "off parent" if off_parent else
                              "too short" if too_short else "wraps parent")
                    continue
                ivs.append([j["lo"], j["hi"], [c]])
            ivs.sort()
            merged = []
            for lo, hi, cs in ivs:
                if merged and lo < merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                    merged[-1][2] += cs
                else:
                    merged.append([lo, hi, cs])
            for lo, hi, cs in merged:
                # Rows at 3:2 to the parent's faces, so every stretch of the hole's
                # border sends meridians up the weld. The border may then carry more
                # vertices than the branch's ring: the transition eases the count
                # down, and the branch keeps its own sides all along its length.
                # (Matching the border to the ring 1:1 gave a thick shallow fork a
                # single 9 cm row -- the fillet folded between its two corners --
                # or 60-odd sides on the whole branch.)
                h = _ROW_H * side_len(par, 0.5 * (lo + hi))
                rows = max(1, int(math.ceil((hi - lo) / max(h, 1e-9) - 1e-6)))
                h = (hi - lo) / rows
                for c in cs:
                    jc = junction[c]
                    a = min(int(math.floor((jc["lo"] - lo) / h + 1e-6)), rows - 1)
                    b = min(max(a + 1, int(math.ceil((jc["hi"] - lo) / h - 1e-6))), rows)
                    jc["rows"] = (lo, hi, rows)
                    jc["span"] = (a, b)
            clusters[pt] = [m for m in merged if all(c not in fallback for c in m[2])]

        bands: dict[int, list] = {}
        grades: dict[int, list] = {}
        for cl in cluster_list:
            bands.setdefault(cl["root"], []).append((cl["s_lo"], cl["s_hi"]))
            if _hub_pair(cl, tubes):
                grades.setdefault(cl["root"], []).extend([(cl["s_lo"], -1.0), (cl["s_hi"], 1.0)])
                for m in cl["members"]:
                    grades.setdefault(m, []).append((cl["starts"][m], 1.0))
        for t, tb in tubes.items():
            start = junction[t]["start"] if (t in junction and t not in fallback) \
                else cluster_start.get(t, 0.0)
            jt = junction.get(t)
            if jt is not None and t not in fallback and "span" in jt:
                # The weld keeps its hole's count all the way to the branch, and
                # a band of the plain branch past the fillet's end steps it down
                # to the branch's sides (`_bridge`): its first ring that much
                # further out, the band's rows _ROW_H to the branch's faces.
                nb = 2 * ((jt["span"][1] - jt["span"][0]) + _columns(jt, tubes[tb.parent]))
                ratio = max(nb, tb.sides) / max(min(nb, tb.sides), 1)
                steps = int(math.ceil(math.log(ratio) / math.log(1.6) - 1e-9)) if ratio > 1 else 0
                if ratio > _LOOP_MATCH_MAX:
                    # A branch far thinner than its hole (a thick pair's half-
                    # stem hole into a branch a third its count) cannot take
                    # the border's count to the fillet's end: its rings crowd
                    # the hole's corners and fold. It eases down as before.
                    steps = 0
                stops = [tb.line.length - 2.0 * edge]
                stops += [junction[c]["lo"] - 2.0 * edge for c in by_parent.get(t, [])
                          if c in junction and c not in fallback]
                stops += [cl["s_lo"] - 2.0 * edge for cl in cluster_list if cl["root"] == t]
                band = min(steps * _ROW_H * side_len(tb, start), min(stops) - start)
                jt["band"] = max(0.0, band)
                start += jt["band"]
            grade_t = list(grades.get(t, ()))
            # Rings graded up from each run of holes, its own rows' height
            # first: the stem's next ring could be a node away, and a quad
            # that long over a hole smeared the crotch's bend up the trunk.
            for lo_, hi_, cs_ in clusters.get(t, []):
                grade_t.append((hi_, 1.0, (hi_ - lo_) / junction[cs_[0]]["rows"][2]))
            _place(tb, start, clusters.get(t, []), junction, q, edge, bark, twist,
                   bands=bands.get(t, ()), grade=grade_t)
        # a cluster's stretch of its root: the one row between its two rings, gone
        for cl in cluster_list:
            root = tubes[cl["root"]]
            r_lo = int(np.argmin(np.abs(root.ring_s - cl["s_lo"])))
            cl["ring_lo"] = r_lo
            root.hole |= {(r_lo, col) for col in range(root.sides)}

        # --- holes (a sibling's columns taken -> that child falls back) --------
        # Two holes sharing rows of their stem, both already at the last
        # ball, each keep to half its columns, a strip of the stem's own
        # columns left between them. (A branch 0.7 of its stem's width
        # lifts the stem's sides too -- they lie within the ball of it --
        # so each hole wrapped 200 deg and the second never fitted; the
        # stem's side lifts by only (k - (r_p - r_c)), 0.05 r_p here.)
        spans: dict[int, list] = {}
        for c, j in junction.items():
            if c in fallback or "rows" not in j:
                continue
            par = tubes[tubes[c].parent]
            base = int(np.argmin(np.abs(par.ring_s - j["rows"][0])))
            spans.setdefault(tubes[c].parent, []).append(
                (c, base + j["span"][0], base + j["span"][1]))
        for pt, lst in spans.items():
            for c, r0, r1 in lst:
                if level.get(c, 0) == len(_BALLS) - 1 and any(
                        o != c and r0 < o1 and o0 < r1 for o, o0, o1 in lst):
                    junction[c]["col_cap"] = max(1, tubes[pt].sides // 2 - 1)
        hole_of: dict[int, set] = {}
        for c, j in junction.items():
            if c in fallback or "rows" not in j:
                continue
            par = tubes[tubes[c].parent]
            lo, hi, rows = j["rows"]
            base = int(np.argmin(np.abs(par.ring_s - lo)))
            r0, r1 = base + j["span"][0], base + j["span"][1]
            col0, cols = _column_range(j, par, r0, r1)
            cells = {(r, (col0 + k) % par.sides) for r in range(r0, r1) for k in range(cols)}
            if r1 >= len(par.ring_s) or cells & par.hole or cols >= par.sides - 1:
                fallback.add(c)
                why[c] = "hole collision"
                blocked_by[c] = next((o for o, cut in hole_of.items()
                                      if tubes[o].parent == tubes[c].parent and cells & cut), -1)
                _place(tubes[c], 0.0, [], junction, q, edge, bark, twist)  # capped, inside
                continue
            par.hole |= cells
            hole_of[c] = cells
            j["cells"] = (r0, r1, col0, cols)


        # a collision is between neighbours: every branch on that parent retries
        crowded = {tubes[c].parent for c, w in why.items() if w in ("sibling", "hole collision")}
        retry = {c for c in junction if tubes[c].parent in crowded}
        retry |= {c for c, w in why.items() if w == "in parent's weld"}
        retry |= {tubes[c].parent for c, w in why.items() if w == "in parent's weld"
                  and tubes[c].parent in junction}
        retry = {c for c in retry if level.get(c, 0) < len(_BALLS) - 1}
        # No branch is moved off its node to make room: where welds do not fit
        # they are retried with a smaller ball, and what still does not fit
        # stays a capped tube. (A slide up the parent broke phyllotaxis -- every
        # opposite pair of branches as thick as their stem came out 12-17 mm
        # apart -- and slid branches ran into the next node's, which then lost
        # their welds.)
        if not retry:
            break
        for c in retry:
            level[c] = level.get(c, 0) + 1

    # --- ring geometry (one numpy block per tube) ---------------------------
    out = _Builder(names + ["pg_u", "pg_v"])
    for t in sorted(tubes):
        tb = tubes[t]
        sec = _section(tb.section, tb.sides)
        nr, ns = len(tb.ring_s), tb.sides
        w = tb.line.attr("width", tb.ring_s)
        rings = tb.centre[:, None, :] + w[:, None, None] * (
            sec[None, :, 0, None] * tb.frame_n[:, None, :]
            + sec[None, :, 1, None] * tb.frame_b[:, None, :])
        block = {n: np.repeat(tb.line.attr(n, tb.ring_s), ns) for n in names}
        block["pg_v"] = np.repeat(tb.v_at(tb.ring_s), ns)
        block["pg_u"] = np.tile(np.arange(ns) / ns, nr)
        tb.base_index = out.add_block(rings.reshape(-1, 3), block)
    out.freeze()

    def vid(tb, r, j):
        return tb.base_index + r * tb.sides + (j % tb.sides)

    # --- tube faces ---------------------------------------------------------
    for t in sorted(tubes):
        tb = tubes[t]
        if t in hidden:
            continue  # its vertices, unused, are dropped below
        nr, ns = len(tb.ring_s), tb.sides
        if nr >= 2:
            keep = np.ones((nr - 1, ns), dtype=bool)
            for a, b in tb.hole:
                keep[a, b] = False
            r, k = np.nonzero(keep)
            k1 = (k + 1) % ns
            base = tb.base_index
            quads = np.stack([base + r * ns + k, base + r * ns + k1,
                              base + (r + 1) * ns + k1, base + (r + 1) * ns + k], axis=1)
            pill = _pill_uv(tb) if tb.domes else None
            if pill is not None:
                uvs = np.stack([pill(r, r, k), pill(r, r, k + 1),
                                pill(r, r + 1, k + 1), pill(r, r + 1, k)], axis=1)
            else:
                v_ring, span = tb.v_at(tb.ring_s), tb.u_span(tb.ring_s)
                v0, v1 = v_ring[r], v_ring[r + 1]
                a0, a1 = k / ns, (k + 1) / ns
                uvs = np.stack([np.stack([a0 * span[r], v0], 1), np.stack([a1 * span[r], v0], 1),
                                np.stack([a1 * span[r + 1], v1], 1),
                                np.stack([a0 * span[r + 1], v1], 1)], axis=1)
            out.quads(quads, uvs)
        else:
            pill = None
        # the tip is always capped; the base is capped on a basal axis and on a
        # child that could not be welded
        cols = np.arange(ns)
        _cap(out, tb, nr - 1, vid, reverse=False,
             uvs=None if pill is None else pill(np.full(ns, nr - 2), np.full(ns, nr - 1), cols))
        if tb.parent < 0 or t in fallback or (t not in junction and t not in cluster_start):
            _cap(out, tb, 0, vid, reverse=True,
                 uvs=None if pill is None else pill(np.zeros(ns, int), np.zeros(ns, int), cols))

    # --- transitions: the parent's hole border into the child's first ring --
    for c, j in junction.items():
        if c in fallback or "cells" not in j:
            continue
        _transition(out, tubes[tubes[c].parent], tubes[c], j, vid, names, tiles, q, edge)
        tubes[c].welded = True
    for cl in cluster_list:
        if not _hub_weld(out, cl, tubes, vid, names, edge):
            _cluster_weld(out, cl, tubes, vid, names, tiles, q, edge)

    welded = sum(1 for c in junction if c not in fallback and "cells" in junction[c])
    welded += sum(len(cl["members"]) for cl in cluster_list)
    pos, attrs_raw, sizes, loops, uv = out.result()
    # Only the vertices faces use: the inside of a hole (a rectangle of the
    # parent's rows and columns) and a cluster's cut stretch leave some behind.
    used = np.zeros(len(pos), dtype=bool)
    used[loops] = True
    new_id = np.cumsum(used) - 1
    pos, loops = pos[used], new_id[loops]
    attrs_raw = {n: a[used] for n, a in attrs_raw.items()}
    for d in list(junction.values()) + cluster_list:
        if "rings" in d:
            d["rings"] = [new_id[np.asarray(r)] for r in d["rings"]]
        if "faces" in d:
            d["faces"] = [([int(new_id[v]) for v in f], uv_) for f, uv_ in d["faces"]]
    attrs_out = {}
    for n, arr in attrs_raw.items():
        attrs_out[n] = np.rint(arr).astype(np.int64) if n in int_names else arr
    for j in list(junction.values()) + cluster_list:
        j.pop("field", None)
    return MeshOut(positions=pos, face_sizes=sizes, face_loops=loops, corner_uv=uv,
                   point_attrs=attrs_out, fallbacks=len(fallback - set(hidden)),
                   junctions=welded,
                   debug={"junction": junction, "fallback": sorted(fallback), "why": why,
                          "blocked_by": blocked_by,
                          "clusters": cluster_list,
                          "leads": sorted(base_width),
                          "holes": {t: sorted(tb.hole) for t, tb in tubes.items() if tb.hole},
                          "sides": {t: tb.sides for t, tb in tubes.items()},
                          "rings": {t: len(tb.ring_s) for t, tb in tubes.items()}})


def _place(tb: _Tube, start: float, clusters, junction, q, edge, bark, twist,
           bands=(), grade=()) -> None:
    """A tube's rings from `start` (adaptive; the rows of its children's holes
    inserted; across a cluster's band only its two end rings; graded away from
    each (s, direction) in `grade`) and their frames: transported, then
    twisted. A tube's `domes` (a catkin's rounded ends) are rings as well."""
    if tb.plain_rings:
        bark = 0.0
    # The mesher's attempt loop places every tube again on every pass; most
    # come back with the same inputs (start, sides, holes, bands, grades) and
    # the same rings -- skipped, and the greedy ring scan kept per start.
    key = (start, tb.sides, tuple((lo, hi, junction[cs[0]]["rows"][2]) for lo, hi, cs in clusters),
           tuple(bands), tuple(tuple(g) for g in grade), q, edge, bark, twist)
    if getattr(tb, "_placed", None) == key:
        return
    twist_step = (10.0 * tb.mean_width / (tb.sides * abs(twist))) if abs(twist) > 1e-6 \
        else float("inf")
    scans = tb.__dict__.setdefault("_scans", {})
    scan_key = (start, twist_step, q, edge, bark)
    if scan_key not in scans:
        scans[scan_key] = list(_adaptive_rings(tb.line, start, q=q, edge=edge, bark=bark,
                                               twist_step=twist_step))
    base = list(scans[scan_key])
    for lo, hi, cs in clusters:
        rows = junction[cs[0]]["rows"][2]
        h = (hi - lo) / rows
        keep = [s for s in base if s < lo - 0.5 * h or s > hi + 0.5 * h]
        base = sorted(keep + list(np.linspace(lo, hi, rows + 1)))
    for lo, hi in bands:
        keep = [s for s in base if s < lo - 0.3 * edge or s > hi + 0.3 * edge]
        base = sorted(keep + [lo, hi])
    # Graded away from a weld: rows an edge tall at the weld's ring, each next
    # one at most twice the last, until the tube's own rings are as close.
    # (Its own rings can be 5-25 mm apart on a straight stretch; a quad that
    # long beside the weld's 2 mm rows smeared the weld's bend down its whole
    # length -- a hard line where weld met tube, and a long thin strip.)
    zones = [(lo, hi) for lo, hi, _cs in clusters] + list(bands)
    for anchor, sign, *first in grade:
        cur, step, new = anchor, (first[0] if first else edge), []
        while True:
            nxt = cur + sign * step
            if not (start + 0.5 * edge < nxt < tb.line.length - 0.5 * edge):
                break
            if any(lo - 0.5 * edge <= nxt <= hi + 0.5 * edge for lo, hi in zones):
                break
            ahead = [x for x in base if (x - cur) * sign > 1e-9]
            if ahead and min(abs(x - cur) for x in ahead) <= 1.5 * step:
                break
            new.append(nxt)
            cur, step = nxt, 2.0 * step
        base = sorted(base + new)
    if tb.domes:
        merged = sorted(set(base) | {s for s in tb.domes if start < s < tb.line.length})
        base = [s for k, s in enumerate(merged) if k == 0 or s - merged[k - 1] > 1e-7]
    tb.ring_s = np.array(base)
    centre, tan = tb.line.at(tb.ring_s)
    normals = _frames(tan, _seam_normal(tan[0]))
    twist_ang = 2.0 * math.pi * twist * tb.ring_s / (20.0 * max(tb.mean_width, 1e-9))
    cs_, sn_ = np.cos(twist_ang)[:, None], np.sin(twist_ang)[:, None]
    bin_ = np.cross(tan, normals)
    tb.frame_n = normals * cs_ + bin_ * sn_
    tb.frame_b = bin_ * cs_ - normals * sn_
    tb.tan, tb.centre = tan, centre
    tb._placed = key


def _junction(par: _Tube, child: _Tube, fillet: float, edge: float, ball: tuple):
    """Size one junction from its weld surface: the stretch of the parent
    (lo..hi along it, and +-beta round it from the branch's azimuth) the
    surface lifts off, and the arc length along the branch where it has
    stopped touching the parent (`start`). None if there is nothing to weld."""
    p0, d = child.line.at(0.0)
    k_near = int(np.argmin(_norm(par.line.dense - p0, axis=1)))
    s0 = float(par.line.s[k_near])
    _, tp = par.line.at(s0)
    sin_raw = float(np.linalg.norm(np.cross(d, tp)))
    # Shallow by the skeleton's own segments: the smoothed stem rounds a node
    # it bends at (a fork whose side shoot took the lead) and halves the angle
    # a 25 deg branch leaves it at.
    cp, pp = child.raw[0], par.raw[0]
    kp = min(int(np.argmin(_norm(pp - cp[0], axis=1))), len(pp) - 2)
    dc, dp = cp[1] - cp[0], pp[kp + 1] - pp[kp]
    sin_seg = float(np.linalg.norm(np.cross(dc, dp))
                    / max(np.linalg.norm(dc) * np.linalg.norm(dp), 1e-18))
    if max(sin_raw, sin_seg) < _MIN_WELD_SIN:
        # A stalk lying along its parent stays close to it for many radii: a
        # weld would fuse it along its length (what the volume does to catkins)
        return "shallow"
    sin_t = max(_MIN_SIN, sin_raw)
    r_p = float(par.line.attr("width", s0))
    r_c = float(child.line.attr("width", min(r_p / sin_t, child.line.length)))
    if r_p <= 0 or r_c <= 0:
        return "no width"
    if r_c >= _SPLIT_RATIO * r_p and sin_raw < _SPLIT_SIN:
        # Nearly as thick as its parent and leaving shallow -- a fork, a
        # sympodial relay: the weld's footprint runs a long way down the parent
        # and the transition folds back on itself into a flap (measured: 50
        # faces past 150 deg on one relay). Such a junction is a split of the
        # stem into two, not a branch on it; until that weld exists it stays a
        # capped tube rather than a folded one.
        return "needs a split weld"
    # The rolling ball: Weld Range x the branch radius, shrinking with the entry
    # angle -- in a shallow crotch a full-size ball fills far up between branch
    # and parent, and the two merge (reported at low angles).
    k = k_base = max(0.0, fillet) * r_c * sin_raw ** _BALL_SIN_EXP
    if k > 0.0:
        # never finer than the mesh that has to build it (a ball smaller than
        # an edge pinched the faces into the crotch) -- up to half the branch's
        # radius, so a stalk thinner than an edge keeps a small weld
        k = max(k, min(ball[0] * edge, ball[1] * r_c))
    tol = 0.02 * r_c
    reach = (r_p + r_c + k) / sin_t + k            # how far along either could matter
    s_lo, s_hi = s0 - 1.5 * reach, s0 + 1.5 * reach
    root, _ = par.line.at(s0)
    toward = _perpendicular(tp[None, :], d)[0]
    field_ = _WeldField(_local_field(par.line, s_lo - r_p, s_hi + r_p, cap_start=False),
                        _local_field(child.line, 0.0, 2.0 * reach + 4.0 * r_c, cap_start=True), k,
                        root, tp, toward)

    # The parent surface the weld lifts: sampled round and along it, coarse
    # over the whole window, then fine over what the coarse pass found.
    def lifted_box(s_a, s_b, half, n_s, n_phi):
        ss = np.linspace(max(s_a, 0.0), min(s_b, par.line.length), n_s)
        c, t = par.line.at(ss)
        e1 = _perpendicular(t, d)
        e2 = np.cross(t, e1)
        phi = np.linspace(-half, half, n_phi)
        r = par.line.attr("width", ss)
        P = (c[:, None, :] + r[:, None, None] * (np.cos(phi)[None, :, None] * e1[:, None, :]
                                                 + np.sin(phi)[None, :, None] * e2[:, None, :]))
        b_eff = field_.gated(P.reshape(-1, 3))[5]
        lifted = (b_eff < k + tol).reshape(n_s, n_phi)
        if not lifted.any():
            return None
        rows_in = np.nonzero(lifted.any(axis=1))[0]
        ds, dphi = ss[1] - ss[0], phi[1] - phi[0]
        return (float(ss[rows_in[0]]) - ds, float(ss[rows_in[-1]]) + ds,
                float(np.max(np.abs(phi[lifted.any(axis=0)]))) + dphi)

    if s_hi <= 0.0 or s_lo >= par.line.length:
        return "off parent"
    box = lifted_box(s_lo, s_hi, math.pi, 32, 33)
    if box is None:
        return "no contact"
    box = lifted_box(box[0], box[1], min(box[2], math.pi), 24, 25) or box
    lo, hi, beta = box

    # where the branch has left the weld: every point of its ring clear of the
    # parent by the fillet radius (and outside it)
    n_c, n_a = 48, 12
    sc = np.linspace(0.0, min(child.line.length, 2.0 * reach + 2.0 * r_c), n_c)
    cc, tc = child.line.at(sc)
    f1 = _perpendicular(tc, tp)
    f2 = np.cross(tc, f1)
    ang = 2.0 * math.pi * np.arange(n_a) / n_a
    wc = child.line.attr("width", sc)
    Q = (cc[:, None, :] + wc[:, None, None] * (np.cos(ang)[None, :, None] * f1[:, None, :]
                                               + np.sin(ang)[None, :, None] * f2[:, None, :]))
    a_raw, _ = field_.parent(Q.reshape(-1, 3))
    touching = (a_raw < k + tol).reshape(n_c, n_a).any(axis=1)
    idx = np.nonzero(touching)[0]
    last = int(idx[-1]) if len(idx) else -1
    if last >= n_c - 1:
        start = float("inf")          # never leaves within the window: too short
    else:
        start = float(sc[last + 1]) + 0.5 * (sc[1] - sc[0])
    return dict(s0=s0, lo=lo, hi=hi, beta=min(beta, math.pi), r_p=r_p, r_c=r_c, k=k,
                k_base=k_base, start=start, dir=d, sin=sin_t, field=field_)


def _dominant_forks(chains: dict, parent_of: dict, verts, width) -> dict:
    """Where a branch is thicker than the stem it leaves carries on (a fork
    whose side shoot took the lead: the stem 9 mm up to the node, 4 mm after
    it, the branch 8 mm), the thick path is made the one tube and the thin
    continuation the branch -- welding a thick branch into a thin stem folds
    the whole crotch. Edits `chains` / `parent_of` in place; returns the base
    width of each continuation made a branch (its own, not the node's)."""
    base_width: dict[int, float] = {}

    def node_of(p, c):
        return int(np.argmin(_norm(verts[chains[p]] - verts[chains[c][0]], axis=1)))

    order = {t: i for i, t in enumerate(parent_of)}     # kids stay in this order
    children: dict[int, list] = {}
    for t, q in parent_of.items():
        children.setdefault(q, []).append(t)

    def kids_of(p):
        return sorted(children.get(p, ()), key=order.__getitem__)

    def adopt(g, new):
        children[parent_of[g]].remove(g)
        parent_of[g] = new
        children.setdefault(new, []).append(g)

    todo = [t for t, p in parent_of.items() if p < 0]
    while todo:
        p = todo.pop()
        swapped = True
        while swapped:
            swapped = False
            kids = [c for c in kids_of(p) if len(chains[c]) > 2]
            best = None
            for c in kids:
                k = node_of(p, c)
                after_p = chains[p][k + 1:k + 3]
                if len(chains[p]) - k < 4:        # the stem's own tip: not a fork
                    continue
                w_c = float(np.min(width[chains[c][1:3]]))
                w_p = float(np.max(width[after_p]))
                if w_c > _LEAD_RATIO * w_p and (best is None or (k, -w_c) < (best[1], -best[2])):
                    best = (c, k, w_c)
            if best is None:
                break
            c, k, _ = best
            op, oc = chains[p], chains[c]
            moving = [g for g in kids_of(p) if g != c and node_of(p, g) > k]
            staying = kids_of(c)
            chains[p] = op[:k + 1] + oc[1:]
            chains[c] = [oc[0]] + op[k + 1:]
            base_width[c] = float(width[op[k + 1]])
            for g in moving:
                adopt(g, c)
            for g in staying:
                adopt(g, p)
            swapped = True
        todo += kids_of(p)
    return base_width


def _weld_zone(child: _Tube, j) -> tuple:
    """The stretch of a branch its weld occupies, as a capsule: from where
    its axis leaves the parent to its first ring, as thick as the branch plus
    the fillet."""
    root, heading = child.line.at(0.0)
    exit_ = root + heading * min(j["r_p"] / j["sin"], j["start"])
    end, _ = child.line.at(min(j["start"], child.line.length))
    return exit_, end, j["r_c"] + j["k"]


def _segment_distance(p0, p1, q0, q1) -> float:
    """Closest distance between segments p0-p1 and q0-q1."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-18 and e < 1e-18:
        return float(np.linalg.norm(r))
    if a < 1e-18:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e < 1e-18:
            s, t = min(1.0, max(0.0, -c / a)), 0.0
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(1.0, max(0.0, -c / a)), 0.0
            elif t > 1.0:
                s, t = min(1.0, max(0.0, (b - c) / a)), 1.0
    return float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))


def _columns(j, par: _Tube) -> int:
    """Parent columns a child's footprint spans (at least 1)."""
    n = max(1, int(math.ceil(2.0 * j["beta"] / (2.0 * math.pi / par.sides) - 1e-6)))
    return min(n, j.get("col_cap", n))


def _column_range(j, par: _Tube, r0: int, r1: int):
    """(first column, count) of the footprint: centred on the child's azimuth in
    the parent's ring frame at the middle hole row."""
    rm = min((r0 + r1) // 2, len(par.ring_s) - 1)
    d = j["dir"]
    t = par.tan[rm]
    flat = d - t * float(np.dot(d, t))
    ang = math.atan2(float(np.dot(flat, par.frame_b[rm])), float(np.dot(flat, par.frame_n[rm])))
    cols = _columns(j, par)
    centre_col = ang / (2.0 * math.pi) * par.sides
    col0 = int(math.floor(centre_col - cols / 2.0 + 0.5))
    return col0 % par.sides, cols


def _cap(out: _Builder, tb: _Tube, r: int, vid, reverse: bool, uvs=None) -> None:
    ring = [vid(tb, r, k) for k in range(tb.sides)]
    if uvs is None:
        rad = float(tb.line.attr("width", tb.ring_s[r])) / tb.tex     # at the plant's texel size
        uvs = [(0.5 + rad * math.cos(2 * math.pi * k / tb.sides),
                0.5 + rad * math.sin(2 * math.pi * k / tb.sides)) for k in range(tb.sides)]
    else:
        uvs = [tuple(map(float, uv)) for uv in uvs]
    if reverse:
        ring, uvs = ring[::-1], uvs[::-1]
    out.poly(ring, uvs)


_PILL_SEAM = 0.85
"""Where a catkin's dome island meets its cylinder: the ring that has narrowed
to this share of the catkin's full width (~58 degrees from the pole on a
hemisphere) -- the split that keeps both mappings within ~1.2x stretch."""
_PILL_CROWN = 0.35
"""A catkin's crown / tip: where it has narrowed below this share of its full
width (the domes run down to ~1% of it)."""
_PILL_GAP = 0.02
"""The gap between a catkin's three UV islands, in tiles."""


def _pill_uv(tb: _Tube):
    """A catkin's unwrap as a pill's: its cylinder a straight rectangle (U its
    full circumference, V its length over the surface) and each rounded end a
    disc round its pole (radius the length over the surface from the pole,
    angle the angle round the catkin), seamed where the end has narrowed to
    `_PILL_SEAM` of the full width. Each pole is the narrowest ring at its end
    -- the last catkin of a bunch hangs from its stalk's tip, so its tube runs
    on through the stalk, which keeps a plain tube's UVs past the crown.
    Returns `uv(row, ring, column)` -- the UV of ring `ring`'s vertex `column`
    for a face in row `row` (a seam ring takes its row's island) -- or None
    when it has no cylinder to unwrap."""
    w = np.asarray(tb.line.attr("width", tb.ring_s), dtype=float)
    nr, ns = len(w), tb.sides
    if nr < 4:
        return None
    # The catkin itself: back from its tip across its full-width middle to
    # where it narrows to its crown (a stalk it hangs from runs on before the
    # crown, and may be thicker than the catkin -- a stalk takes its width from
    # the axis it leaves). A tube with no narrow tip (a bare stalk) is plain.
    pole_b = nr - 1
    full, pole_t = 0.0, -1
    for i in range(pole_b - 1, -1, -1):
        full = max(full, float(w[i]))
        if w[i] < _PILL_CROWN * full:
            pole_t = i
            while pole_t > 0 and w[pole_t - 1] < w[pole_t]:
                pole_t -= 1
            break
    if pole_t < 0 or w[pole_b] >= _PILL_CROWN * full:
        return None
    body = w[pole_t:pole_b + 1]
    wide = pole_t + np.nonzero(body >= _PILL_SEAM * float(np.max(body)))[0]
    if len(wide) < 2:
        return None
    top, bot = int(wide[0]), int(wide[-1])
    if bot - top < 1:
        return None
    v = np.asarray(tb.v_at(tb.ring_s), dtype=float)          # length over the surface, tiles
    span = 2.0 * math.pi * float(np.max(body)) / tb.tex
    rho_top = w[pole_t] / tb.tex + np.abs(v - v[pole_t])      # from each pole (its cap's radius on)
    rho_bot = w[pole_b] / tb.tex + np.abs(v[pole_b] - v)
    r_top, r_bot = float(rho_top[top]), float(rho_bot[bot])
    c_top = (span + _PILL_GAP + r_top, r_top)
    c_bot = (span + 2.0 * _PILL_GAP + 2.0 * r_top + r_bot, r_bot)
    plain_span = 2.0 * math.pi * w / tb.tex                    # a plain tube's U, ring by ring

    def uv(row, ring, col):
        row, ring, col = (np.asarray(x) for x in np.broadcast_arrays(row, ring, col))
        ang = 2.0 * math.pi * col / ns
        out = np.empty(row.shape + (2,))
        plain = (row < pole_t) | (row >= pole_b)
        dome_t = (row + 1 <= top) & ~plain
        dome_b = (row >= bot) & ~plain
        mid = ~(dome_t | dome_b | plain)
        out[plain, 0] = col[plain] / ns * plain_span[ring[plain]]
        out[plain, 1] = v[ring[plain]]
        out[mid, 0] = col[mid] / ns * span
        out[mid, 1] = v[ring[mid]] - v[top]
        rt = rho_top[ring[dome_t]]
        out[dome_t, 0] = c_top[0] + rt * np.cos(ang[dome_t])
        out[dome_t, 1] = c_top[1] + rt * np.sin(ang[dome_t])
        rb = rho_bot[ring[dome_b]]
        out[dome_b, 0] = c_bot[0] + rb * np.cos(ang[dome_b])
        out[dome_b, 1] = c_bot[1] + rb * np.sin(ang[dome_b])
        return out
    return uv


def _border_loop(par: _Tube, cells, vid):
    """The hole's border, as (vertex, ring, column) going round the rectangle."""
    r0, r1, col0, cols = cells
    loop = [(r0, col0 + k) for k in range(cols + 1)]
    loop += [(r, col0 + cols) for r in range(r0 + 1, r1)]
    loop += [(r1, col0 + cols - k) for k in range(cols + 1)]
    loop += [(r, col0) for r in range(r1 - 1, r0, -1)]
    return [(vid(par, r, k), r, k % par.sides) for r, k in loop]


_KEEP_CHILD = ("tube", "parent_tube", "axis_id", "order", "structural_depth")
"""Identity attributes a transition vertex takes from its child, not blended."""


_WALK_STEPS = 12
"""Steps a meridian takes across the weld surface (at 1.6x the straight distance)."""


def _walk(field_: _WeldField, B: np.ndarray, C: np.ndarray, normal0: np.ndarray,
          via: np.ndarray, axis: np.ndarray, n_q: int) -> np.ndarray:
    """Paths on the weld surface from each B to its C: step toward C within
    the tangent plane, pull back onto the surface, stop on arrival. Each path
    sets off along the parent's axis toward the contact's middle (`via`) --
    all in parallel, so neighbours never converge and cross -- and keeps that
    heading while C lies straight out across the surface: a walker on the
    parent above an acute crotch sees its C horizontally across the gap and
    must go down into the fillet first. Returns (n, n_q, 3), resampled evenly
    by arc length."""
    n = len(B)
    h = np.maximum(1.6 * _norm(C - B, axis=1) / _WALK_STEPS, 1e-12)
    h_max = float(h.max())

    def unit(v):
        return v / np.maximum(_norm(v, axis=1, keepdims=True), 1e-15)

    along = (via[None, :] - B) @ axis
    to = C - B
    tang0 = to - normal0 * np.sum(to * normal0, axis=1, keepdims=True)
    prev = unit(tang0 / np.maximum(_norm(to, axis=1, keepdims=True), 1e-15)
                + 0.6 * np.sign(along)[:, None] * axis[None, :])
    P = B.copy()
    paths = [B.copy()]
    done = np.zeros(n, dtype=bool)
    for _ in range(3 * _WALK_STEPS):
        if done.all():
            break
        _, g = field_(P)
        nrm = unit(g)
        to = C - P
        arrive = ~done & (_norm(to, axis=1) <= 1.5 * h)
        tang = to - nrm * np.sum(to * nrm, axis=1, keepdims=True)
        # the pull toward C is its tangent share of the straight way there:
        # weak while C lies straight out across the surface, so the heading
        # carries on until the surface turns toward C
        pull = tang / np.maximum(_norm(to, axis=1, keepdims=True), 1e-15)
        heading = prev - nrm * np.sum(prev * nrm, axis=1, keepdims=True)
        dirn = unit(pull + 0.6 * unit(heading))
        step = field_.project(P + h[:, None] * dirn, iters=2, max_step=2.0 * h_max)
        step[arrive] = C[arrive]
        step[done] = P[done]
        prev = np.where(done[:, None], prev, dirn)
        done |= arrive
        P = step
        paths.append(P.copy())
    P = np.where(done[:, None], P, C)
    paths.append(P)
    paths = np.stack(paths, axis=1)                                    # (n, S, 3)
    seg = _norm(np.diff(paths, axis=1), axis=2)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, axis=1)], axis=1)
    target = np.linspace(0.0, 1.0, n_q)[None, :] * cum[:, -1:]
    i = np.clip((cum[:, None, :] <= target[:, :, None]).sum(axis=2) - 1, 0, paths.shape[1] - 2)
    rows = np.arange(n)[:, None]
    c0, c1 = cum[rows, i], cum[rows, i + 1]
    w = ((target - c0) / np.maximum(c1 - c0, 1e-15))[..., None]
    out = paths[rows, i] * (1.0 - w) + paths[rows, i + 1] * w
    out[:, 0], out[:, -1] = B, C
    return out


def _periodic(u_src: np.ndarray, vals: np.ndarray, u_new: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-vertex values round a closed loop, by u in
    [0, 1) (u_src ascending from 0)."""
    u_ext = np.concatenate([u_src, [1.0]])
    v_ext = np.concatenate([vals, vals[:1]], axis=0)
    un = np.mod(u_new, 1.0)
    i = np.clip(np.searchsorted(u_ext, un, side="right") - 1, 0, len(u_src) - 1)
    t = (un - u_ext[i]) / np.maximum(u_ext[i + 1] - u_ext[i], 1e-12)
    if v_ext.ndim > 1:
        t = t[:, None]
    return v_ext[i] * (1.0 - t) + v_ext[i + 1] * t


def _transition(out: _Builder, par: _Tube, child: _Tube, j, vid, names, tiles, q,
                floor) -> None:
    """One branch's weld: its parent's hole border into its first ring."""
    field_: _WeldField = j["field"]
    border = _border_loop(par, j["cells"], vid)
    b_ids = np.array([v for v, _, _ in border])
    b_pts = out.frozen_pos[b_ids]
    b_rows = np.array([r for _, r, _ in border])
    c0 = child.centre[0]
    n0, bn0 = child.frame_n[0], child.frame_b[0]
    # Wind the border counter-clockwise about the child's heading, like its ring.
    # The loop is already a cycle: only its direction is chosen here. Sorting it
    # by angle instead scrambled the corners of a steep child's footprint (they
    # do not project in angular order) into diagonal, non-manifold faces.
    rel = b_pts - c0
    ang = np.arctan2(rel @ bn0, rel @ n0)
    turn = float(np.sum(np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))))
    if turn < 0.0:
        b_ids, b_pts, b_rows, ang = b_ids[::-1], b_pts[::-1], b_rows[::-1], ang[::-1]
    # u runs 0..1 round both loops: on the border from the vertex facing the
    # child's column 0, on the child's ring from its column 0
    k0 = int(np.argmin(np.abs(np.angle(np.exp(1j * ang)))))
    b_ids, b_pts, b_rows = np.roll(b_ids, -k0), np.roll(b_pts, -k0, axis=0), np.roll(b_rows, -k0)
    radial = b_pts - par.centre[b_rows]
    radial /= np.maximum(_norm(radial, axis=1, keepdims=True), 1e-12)
    c_ids = np.array([vid(child, 0, k) for k in range(child.sides)])
    w0 = max(float(child.line.attr("width", child.ring_s[0])), 1e-9)
    root, heading = child.line.at(0.0)            # the middle of the contact:
    p_axis = par.tan[int(np.median(b_rows))]      # where the branch's axis
    sin_t = max(_MIN_SIN, float(np.linalg.norm(np.cross(heading, p_axis))))  # leaves
    via = root + heading * min(j["r_p"] / sin_t, float(child.ring_s[0]))     # the parent
    owners = [(par, j["lo"] - 2.0 * j["r_p"], j["hi"] + 2.0 * j["r_p"]),
              (child, 0.0, float(child.ring_s[0]) + 2.0 * w0)]
    _bridge(out, field_, b_ids, radial, c_ids, via, p_axis, w0, owners, names, tiles, q,
            floor, j, align=False)


def _bridge(out: _Builder, field_, b_ids, radial, c_ids, via, p_axis, w0, owners, names,
            tiles, q, floor, j, align: bool = True, relax_iters: int = None) -> None:
    """Mesh the weld surface between two closed loops of existing vertices --
    `b_ids` (the outer: a hole's border, or a stem's ring below a cluster;
    `radial` its outward normals) and `c_ids` (the inner: a branch's ring, or
    a cluster's rings joined by seams) -- both wound the same way. Rings on
    meridians walked across the surface, eased from one loop's vertex count to
    the other's in quad units, relaxed on the surface, crowded edges collapsed.
    Each face takes the UVs of the tube it lies on (`owners[field_.owner()]`:
    (tube, s range)). Writes the vertices and faces into `out`; the rings and
    faces into `j`."""
    b_pts = out.frozen_pos[b_ids]
    c_pts = out.frozen_pos[c_ids]
    nb, nc = len(b_ids), len(c_ids)

    def arc_u(pts):
        e = _norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        return np.concatenate(([0.0], np.cumsum(e[:-1]))) / max(float(e.sum()), 1e-12), e

    u_b, edge = arc_u(b_pts)
    if align:
        # the inner loop's start: where the meridians are shortest overall
        u_c0, _ = arc_u(c_pts)
        best, best_cost = 0, float("inf")
        for off in range(nc):
            u_rot = (u_c0 - u_c0[off]) % 1.0
            order = np.argsort(u_rot)
            cost = float(np.sum(_norm(
                _periodic(u_rot[order], c_pts[order], u_b) - b_pts, axis=1) ** 2))
            if cost < best_cost:
                best, best_cost = off, cost
        c_ids = np.roll(c_ids, -best)
        c_pts = out.frozen_pos[c_ids]
    u_c, c_edge = arc_u(c_pts)

    # One meridian per vertex of the longer loop, from the outer loop to the
    # inner at the same u, *walked on the weld surface*: small steps toward it
    # within the surface's tangent plane, each pulled back onto it -- evenly
    # spaced and in order. (Casting a Hermite sheet onto the surface, along its
    # normals or the distance gradient, folded the crotch, where the sheet
    # curves tighter than the distance it has to move.)
    u_m = u_b if nb >= nc else u_c
    n_m = len(u_m)
    B = b_pts if nb >= nc else _periodic(u_b, b_pts, u_m)
    R = radial if nb >= nc else _periodic(u_b, radial, u_m)
    R = R / np.maximum(_norm(R, axis=1, keepdims=True), 1e-12)
    C = _periodic(u_c, c_pts, u_m) if nb >= nc else c_pts
    n_q = 32
    M = _walk(field_, B, C, R, via, p_axis, n_q)
    M = _relax_grid(field_, M, max(field_.k, 0.25 * w0))
    seg = _norm(np.diff(M, axis=1), axis=2)                   # (n_m, n_q-1)
    cum = np.concatenate([np.zeros((n_m, 1)), np.cumsum(seg, axis=1)], axis=1)
    L = cum[:, -1]

    # Rings go where the meridians turn (the fillet's arc), not evenly by
    # length: each stretch of a meridian costs its length over a 3:1 face
    # plus its turn over the turn allowed per ring, and rings are laid at
    # equal cost. Equal arc length left a tight fillet arc between two rings.
    tdir = np.diff(M, axis=1)
    tdir /= np.maximum(_norm(tdir, axis=2, keepdims=True), 1e-12)
    bend = np.arccos(np.clip(np.sum(tdir[:, 1:] * tdir[:, :-1], axis=2), -1.0, 1.0))
    step = 0.5 * (float(edge.mean()) + float(c_edge.mean()))
    bend_seg = np.zeros_like(seg)
    bend_seg[:, 1:] += 0.5 * bend
    bend_seg[:, :-1] += 0.5 * bend
    # The turn may draw rings together, but never closer than a fraction of an
    # edge: a small fillet in a shallow fork's crotch turns ~180 deg in 2 mm,
    # and the turn alone crushed a dozen rings into that channel.
    turn_cost = np.minimum(bend_seg / math.radians(_FILLET_DEG * q), seg / (_RING_FLOOR * floor))
    cost = seg / max(3.0 * step, 1e-12) + turn_cost
    ccum = np.concatenate([np.zeros((n_m, 1)), np.cumsum(cost, axis=1)], axis=1)
    frac = ccum / np.maximum(ccum[:, -1:], 1e-12)
    if "members" in j:
        # A cluster's loops by height along the stem: each ring at the same
        # fraction of every meridian's climb, so over the stem's ring above they
        # are level and toward a branch they tilt only as far as its ring does.
        # (By cost, a meridian curling onto a branch put its rings low in the
        # fillet while one running straight up spaced them evenly -- the loops
        # were pulled down into the crotch.) Where a meridian barely climbs, its
        # cost fraction stays.
        climb = np.maximum.accumulate((M - M[:, :1]) @ p_axis, axis=1)
        span = climb[:, -1:]
        by_height = climb / np.maximum(span, 1e-12)
        flat = span[:, 0] < 0.25 * float(np.median(span))
        frac = np.where(flat[:, None], frac, by_height)
    band = float(j.get("band", 0.0)) if "members" not in j else 0.0
    matched = band > 0.0
    if matched:
        # The fillet ends `band` before the branch's first ring (the branch
        # starts that much further out): every ring up to there keeps the
        # border's count -- plain quads, columns straight through the bend --
        # and the step down to the branch's sides is made in the band, on the
        # plain branch. Rings by cost over the fillet (as below), by length in
        # the band.
        a_end = np.maximum(L - band, 0.5 * L)
        c_end = np.array([np.interp(a_end[m], cum[m], ccum[m]) for m in range(n_m)])
        t_end = 0.8
        in_band = cum > a_end[:, None]
        frac = np.where(in_band,
                        t_end + (1.0 - t_end) * (cum - a_end[:, None])
                        / np.maximum(L - a_end, 1e-12)[:, None],
                        t_end * ccum / np.maximum(c_end, 1e-12)[:, None])
        n_w = min(int(math.ceil(float(np.percentile(c_end, _RING_PCT)))) - 1,
                  int(float(np.percentile(a_end, 90)) / (_RING_FLOOR * floor)))
        n_w = int(min(_TRANSITION_MAX, max(_TRANSITION_MIN, n_w)))
        ratio = max(nb, nc) / max(min(nb, nc), 1)
        n_band = int(math.ceil(math.log(ratio) / math.log(1.6) - 1e-9)) - 1 if nb != nc else 0
        n_band = max(n_band, 0)
        ts = np.concatenate([t_end * np.arange(1, n_w + 1) / (n_w + 1), [t_end],
                             t_end + (1.0 - t_end) * np.arange(1, n_band + 1) / (n_band + 1)])
        n_rings = len(ts)
        n_red = n_band + 1
    else:
        # never closer than the plant's resolution floor (a pedicel's tiny join)
        n_rings = min(int(math.ceil(float(np.percentile(ccum[:, -1], _RING_PCT)))) - 1,
                      int(float(np.percentile(L, 90)) / (_RING_FLOOR * floor)))
        n_rings = int(min(_TRANSITION_MAX, max(_TRANSITION_MIN, n_rings)))
        # enough rings that the vertex count changes by at most 1.6x per ring
        ratio = max(nb, nc) / max(min(nb, nc), 1)
        n_red = int(math.ceil(math.log(ratio) / math.log(1.6) - 1e-9)) if nb != nc else 0
        n_rings = max(n_rings, n_red + 1, 2)
        ts = np.arange(1, n_rings + 1) / (n_rings + 1)
    j["bend"] = float(np.percentile(bend.sum(axis=1), 90))

    # every meridian's point at each cost fraction, and its arc from the border
    rows = np.arange(n_m)[:, None]
    idx = np.clip((frac[:, None, :] < ts[None, :, None]).sum(axis=2) - 1, 0, n_q - 2)
    f0, f1 = frac[rows, idx], frac[rows, idx + 1]
    w = (ts[None, :] - f0) / np.maximum(f1 - f0, 1e-12)
    along = M[rows, idx] * (1.0 - w[..., None]) + M[rows, idx + 1] * w[..., None]  # (n_m, R, 3)
    arc = cum[rows, idx] * (1.0 - w) + cum[rows, idx + 1] * w
    counts = [nb]
    for i in range(1, n_rings + 1):
        if matched:
            k = i - (n_rings - n_red + 1)          # 0 at the fillet's end, then the band
            counts.append(nb if k <= 0 else 2 * round(0.5 * nb * (nc / nb) ** (k / n_red)))
            continue
        f_ = i / (n_rings + 1)
        c = 2 * round(0.5 * nb * (nc / nb) ** f_)
        c = int(min(max(c, min(nb, nc)), max(nb, nc)))
        # enough vertices for the ring's own length: mid-weld rings run long
        # through the crotch, and a count eased only by index left 5-7 mm
        # edges there (a face that long bends past 30 deg on its own)
        ring = along[:, i - 1]
        length = float(_norm(np.roll(ring, -1, axis=0) - ring, axis=1).sum())
        tgt = (1.0 - f_) * float(edge.mean()) + f_ * float(c_edge.mean())
        c = max(c, 2 * int(math.ceil(0.5 * length / max(tgt, 1e-12))))
        counts.append(c)
    counts.append(nc)
    for i in range(len(counts) - 2, 0, -1):      # keep every step within 1.6x
        big, small = max(counts[i], counts[i + 1]), min(counts[i], counts[i + 1])
        if big - small > 2 * (small // 3):
            counts[i] = counts[i + 1] + (1 if counts[i] > counts[i + 1] else -1) * 2 * (small // 3)
    counts = counts[1:-1]
    # every step even, so the 3-to-1 units fit
    counts = [c + ((c - nb) % 2) for c in counts]

    # each ring's vertices at their u between the meridians, on the surface
    us = [np.arange(n_i) / n_i for n_i in counts]
    pts = np.concatenate([_periodic(u_m, along[:, i], u_i) for i, u_i in enumerate(us)])
    pts = field_.project(pts, iters=2, max_step=0.25 * max(field_.k, 0.25 * w0))
    left = np.concatenate([_periodic(u_m, L - arc[:, i], u_i) for i, u_i in enumerate(us)])
    u_all = np.concatenate(us)
    t_all = 1.0 - left / np.maximum(_periodic(u_m, L, u_all), 1e-12)
    b_attr = np.stack([out.frozen_attr[n][b_ids] for n in names], axis=1)
    c_attr = np.stack([out.frozen_attr[n][c_ids] for n in names], axis=1)
    keep = np.array([n in _KEEP_CHILD for n in names])
    a_ = _periodic(u_b, b_attr, u_all)
    b_ = _periodic(u_c, c_attr, u_all)
    attr_all = np.where(keep[None, :], b_, a_ + (b_ - a_) * t_all[:, None])
    v_all = t_all.copy()               # placement only: the UVs are projected below

    first = out.count                  # where this weld's vertices will start
    ring_ids = [np.asarray(b_ids)]
    at = 0
    for n_i in counts:
        ring_ids.append(first + at + np.arange(n_i))
        at += n_i
    ring_ids.append(np.asarray(c_ids))
    j["rings"] = [np.asarray(r, dtype=np.int64) for r in ring_ids]  # tests / debugging

    # every vertex the strips touch: position, u (unwrapped per ring), v, attrs
    pos_of, u_of, v_of, attr_of = {}, {}, {}, {}
    for vid_, p, u, a in zip(b_ids.tolist(), b_pts, u_b, b_attr, strict=True):
        pos_of[vid_], u_of[vid_], v_of[vid_], attr_of[vid_] = p, u, 0.0, a
    for vid_, p, u, a in zip(c_ids.tolist(), c_pts, u_c, c_attr, strict=True):
        pos_of[vid_], u_of[vid_], v_of[vid_], attr_of[vid_] = p, u, 1.0, a
    for k, vid_ in enumerate(range(first, first + len(pts))):
        pos_of[vid_], u_of[vid_], v_of[vid_], attr_of[vid_] = pts[k], u_all[k], v_all[k], \
            attr_all[k]
    extra_pos, extra_attr = [], []
    next_id = [first + len(pts)]

    def midpoint(a, b0, b1, t):
        """A unit's interior vertex: halfway from ring vertex `a` to the point
        t of the way along the other ring's edge b0-b1."""
        ub1 = u_of[b1] + (1.0 if u_of[b1] < u_of[b0] - 0.5 else 0.0)
        ua = u_of[a]
        ub = u_of[b0] + (ub1 - u_of[b0]) * t
        if ua < ub - 0.5:
            ua += 1.0
        elif ua > ub + 0.5:
            ub += 1.0
        extra_pos.append(0.5 * pos_of[a] + 0.5 * (pos_of[b0] * (1 - t) + pos_of[b1] * t))
        extra_attr.append(0.5 * attr_of[a] + 0.5 * (attr_of[b0] * (1 - t) + attr_of[b1] * t))
        vid_ = next_id[0]
        next_id[0] += 1
        pos_of[vid_], u_of[vid_] = extra_pos[-1], (0.5 * ua + 0.5 * ub) % 1.0
        v_of[vid_] = 0.5 * v_of[a] + 0.5 * (v_of[b0] * (1 - t) + v_of[b1] * t)
        attr_of[vid_] = extra_attr[-1]
        return vid_

    # the 3-to-1 units: evenly round each ring, the pattern turned a
    # golden-ratio step each ring so their poles never stack in a column
    ring_u = [u_b] + us + [u_c]
    placed_u: list[float] = []
    faces = []
    for i in range(len(ring_ids) - 1):
        ua, ub_ = ring_u[i], ring_u[i + 1]
        u_long = ua if len(ua) >= len(ub_) else ub_
        n_l = len(u_long)
        r_i = abs(len(ua) - len(ub_)) // 2
        grid = (np.arange(max(r_i, 1)) + (i * 0.618034) % 1.0) / max(r_i, 1)
        cost = np.abs(((u_long[:, None] - grid[None, :]) + 0.5) % 1.0 - 0.5).min(axis=1)
        cost = cost * max(r_i, 1)
        if placed_u:
            du = np.abs(((u_long[:, None] - np.array(placed_u)[None, :]) + 0.5) % 1.0 - 0.5)
            cost = cost + np.clip(1.0 - du.min(axis=1) * n_l / 3.0, 0.0, 1.0)
        strip, starts = _quad_strip(ring_ids[i].tolist(), ring_ids[i + 1].tolist(), midpoint,
                                    cost=cost)
        faces.extend(list(f) for f in strip)
        placed_u = [float(u_long[(k + 1) % n_l]) for k in starts]
    if extra_pos:
        pts = np.concatenate([pts, np.array(extra_pos)])
        attr_all = np.concatenate([attr_all, np.array(extra_attr)])
    pts = _relax_on_surface(field_, pts, first, faces, out.frozen_pos,
                            max(field_.k, 0.25 * w0), iters=relax_iters)
    # Clean-up, as by hand: where the crotch crowds vertices together, faces
    # get one edge a tenth of the others and fold sharply across it (130 deg
    # measured). Those short edges are collapsed -- a quad becomes a
    # triangle there -- and the weld relaxed once more.
    faces, pts, attr_all, remap = _collapse_short(faces, pts, attr_all, first,
                                                 out.frozen_pos, _COLLAPSE_FRAC)
    if remap is not None:
        j["rings"] = [np.array([remap.get(int(v), int(v)) for v in r], dtype=np.int64)
                      for r in j["rings"]]
        pts = _relax_on_surface(field_, pts, first, faces, out.frozen_pos,
                                max(field_.k, 0.25 * w0), iters=8)

    # UVs: each face takes the island of the tube it lies on -- the one its
    # surface is nearest, as the union reads them -- so the islands meet at
    # the crease, where the wood's grain turns: a face on the parent's side
    # carries the parent's own island on (its u round the parent's axis, v
    # along it, so the border meets the parent's quads exactly), one on a
    # branch's side that branch's. (The branch's island run down over the whole
    # weld warped it and spread the branch's texture onto the trunk. A ring
    # laid on the crease instead sheared a thick fork, whose crease is no
    # single loop round it.)
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    row = {int(v): k for k, v in enumerate(used)}
    where = np.array([pts[v - first] if v >= first else out.frozen_pos[v] for v in used])
    tube_uv = [_tube_uv(tb, where, s_lo, s_hi) for tb, s_lo, s_hi in owners]
    centres = np.array([where[[row[v] for v in vids]].mean(axis=0) for vids in faces])
    face_owner = field_.owner(centres)
    uv_faces = []
    for vids, o in zip(faces, face_owner.tolist(), strict=True):
        uvs = tube_uv[o][[row[v] for v in vids]]
        u = uvs[:, 0] + np.round(uvs[0, 0] - uvs[:, 0])     # one side of u = 0 | 1
        uv_faces.append((vids, [(float(a_) * sp, float(b_)) for a_, b_, sp in zip(
            u, uvs[:, 1], uvs[:, 2], strict=True)]))
    # per vertex: the section angle (flare lobes) and V of the tube it is on
    mine = np.array([row[v] for v in range(first, first + len(pts))])
    v_owner = field_.owner(pts)
    pg = np.stack([tube_uv[o][m] for o, m in zip(v_owner.tolist(), mine.tolist(),
                                                 strict=True)]) if len(pts) else np.zeros((0, 2))
    block = {n: attr_all[:, i] for i, n in enumerate(names)}
    block["pg_u"] = pg[:, 0] % 1.0
    block["pg_v"] = pg[:, 1]
    out.add_block(pts, block)
    for vids, uv in uv_faces:
        out.poly(vids, uv)
    j["faces"] = uv_faces


# ---------------------------------------------------------------------------
# clusters: several branches welded as one
# ---------------------------------------------------------------------------

def _azimuth(par: _Tube, s: float, d: np.ndarray) -> float:
    """A direction's angle round a tube's axis at arc length s (in a frame
    transported from the tube's own start, so comparable along it)."""
    _, t = par.line.at(np.array([s]))
    e1 = _perpendicular(t, np.array([0.0, 0.0, 1.0]) if abs(float(t[0, 2])) < 0.9
                        else np.array([1.0, 0.0, 0.0]))[0]
    e2 = np.cross(t[0], e1)
    return math.atan2(float(d @ e2), float(d @ e1))


def _find_clusters(tubes: dict, junction: dict) -> list:
    """Junctions whose welds would run into each other, grouped: (root stem,
    [branches]). Two junctions interact when their footprints on the same
    stem overlap (along it and round it), when their weld zones come within a
    branch radius plus a fillet of each other, or when one branch leaves the
    other inside the other's own weld (a branch splitting right after it
    leaves its stem). A group whose branches hang from more than one stem
    outside it is left to the single welds."""
    kids = list(junction)
    parent_of = {c: tubes[c].parent for c in kids}
    link = {c: c for c in kids}

    def find(c):
        while link[c] != c:
            link[c] = link[link[c]]
            c = link[c]
        return c

    def join(a, b):
        link[find(a)] = find(b)

    zone = {c: _weld_zone(tubes[c], junction[c]) for c in kids}
    for c in kids:
        p = parent_of[c]
        if p in junction:                    # leaves its parent inside that one's weld
            side = 2.0 * math.pi * float(tubes[p].line.attr("width", junction[c]["s0"])) \
                / tubes[p].sides
            if junction[c]["lo"] < junction[p]["start"] + side:
                join(c, p)
    for i, a in enumerate(kids):
        for b in kids[i + 1:]:
            ja, jb = junction[a], junction[b]
            if parent_of[a] == parent_of[b]:
                par = tubes[parent_of[a]]
                side = 2.0 * math.pi * float(par.line.attr("width", ja["s0"])) / par.sides
                along = ja["lo"] < jb["hi"] + side and jb["lo"] < ja["hi"] + side
                da = _azimuth(par, ja["s0"], ja["dir"])
                db = _azimuth(par, jb["s0"], jb["dir"])
                gap = abs(math.atan2(math.sin(da - db), math.cos(da - db)))
                round_ = gap < ja["beta"] + jb["beta"] + 2.0 * math.pi / par.sides
                if along and round_:
                    join(a, b)
                    continue
            (a0, a1, ra), (b0, b1, rb) = zone[a], zone[b]
            if _segment_distance(a0, a1, b0, b1) < ra + rb:
                join(a, b)
    groups: dict[int, list] = {}
    for c in kids:
        groups.setdefault(find(c), []).append(c)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        roots = {parent_of[c] for c in members} - set(members)
        if len(roots) == 1:
            out.append((roots.pop(), sorted(members)))
    return out


def _cluster_setup(root: _Tube, members: list, tubes: dict, junction: dict, start_of_root: float,
                   edge: float):
    """The weld surface of a cluster and where it lies: the stretch of the root
    stem it replaces (s_lo..s_hi, all the way round), and where each branch
    leaves it (its first clear ring). A string saying why if it cannot."""
    js = {m: junction[m] for m in members}
    on_root = [j for m, j in js.items() if tubes[m].parent == root.tid]
    hub = _opposite_pair(root, members, tubes,
                         float(np.mean([j["s0"] for j in on_root])))
    if hub:
        # A hub's pair is crowded by definition -- their welds share the stem
        # -- so it takes the ball a crowded single weld steps down to (the
        # last of _BALLS). (At the first, the junctions' own, its fillets were
        # twice a lone weld's and a pair read as a swollen node.)
        b_e, b_r = _BALLS[-1]
        k = float(np.median([max(j["k_base"], min(b_e * edge, b_r * j["r_c"]))
                             if j["k_base"] > 0.0 else 0.0 for j in js.values()]))
    else:
        k = float(np.median([j["k"] for j in js.values()]))
    r_root = float(root.line.attr("width", float(np.mean(
        [j["s0"] for m, j in js.items() if tubes[m].parent == root.tid]))))
    w_lo = min(j["lo"] for j in on_root) - 2.0 * (r_root + k)
    w_hi = max(j["hi"] for j in on_root) + 2.0 * (r_root + k)
    for m, j in js.items():                       # branches of branches reach further up
        if tubes[m].parent != root.tid:
            w_hi += j["hi"] - j["lo"]
    span = w_hi - w_lo
    fields, gates, idx = [_local_field(root.line, w_lo - r_root, w_hi + r_root,
                                       cap_start=False)], [None], {root.tid: 0}
    for m in members:
        tb = tubes[m]
        r_m = float(tb.line.attr("width", 0.0))
        fields.append(_local_field(tb.line, 0.0, min(tb.line.length, 2.0 * span + 4.0 * r_m),
                                   cap_start=True))
        par = tubes[tb.parent]
        j = js[m]
        p_pt, p_tan = par.line.at(np.array([j["s0"]]))
        gates.append((p_pt[0], p_tan[0], _perpendicular(p_tan, j["dir"])[0]))
        idx[m] = len(fields) - 1
    # A true rolling ball for a pair the hub welds (the band weld folded more
    # on it -- its rows are laid for the round union's shallower fillets).
    field_ = _MultiField(fields, gates, k, exact=bool(hub))
    tol = 0.02 * k

    # the root's stretch the weld lifts off, all the way round
    ss = np.linspace(max(w_lo, 0.0), min(w_hi, root.line.length), 64)
    c, t = root.line.at(ss)
    e1 = _perpendicular(t, np.array([0.0, 0.0, 1.0]))
    e2 = np.cross(t, e1)
    phi = np.linspace(-math.pi, math.pi, 48, endpoint=False)
    r = root.line.attr("width", ss)
    P = (c[:, None, :] + r[:, None, None] * (np.cos(phi)[None, :, None] * e1[:, None, :]
                                             + np.sin(phi)[None, :, None] * e2[:, None, :]))
    D0, _, U = field_.distances(P.reshape(-1, 3))
    if field_.exact:
        # where the weld surface leaves the stem's own (the ball reaches
        # further than k round a sharp crotch)
        lifted = (field_(P.reshape(-1, 3))[0] < D0[0] - tol)
    else:
        lifted = U[1:].min(axis=0) < k + tol
    lifted = lifted.reshape(len(ss), len(phi)).any(axis=1)
    if not lifted.any():
        return "no contact"
    rows_in = np.nonzero(lifted)[0]
    ds = ss[1] - ss[0]
    s_lo, s_hi = float(ss[rows_in[0]]) - ds, float(ss[rows_in[-1]]) + ds
    if s_lo < start_of_root + edge or s_hi > root.line.length - edge:
        return "off root"

    # each branch's first ring clear of every other tube in the cluster
    starts = {}
    for m in members:
        tb = tubes[m]
        sc = np.linspace(0.0, min(tb.line.length, 2.0 * span + 2.0 * float(
            tb.line.attr("width", 0.0))), 48)
        cc, tc = tb.line.at(sc)
        f1 = _perpendicular(tc, t[len(t) // 2])
        f2 = np.cross(tc, f1)
        ang = 2.0 * math.pi * np.arange(12) / 12
        wc = tb.line.attr("width", sc)
        Q = (cc[:, None, :] + wc[:, None, None] * (np.cos(ang)[None, :, None] * f1[:, None, :]
                                                   + np.sin(ang)[None, :, None] * f2[:, None, :]))
        D, _, _ = field_.distances(Q.reshape(-1, 3))
        if field_.exact:
            touching = field_(Q.reshape(-1, 3))[0] < D[idx[m]] - tol
        else:
            touching = np.delete(D, idx[m], axis=0).min(axis=0) < k + tol
        touching = touching.reshape(len(sc), 12).any(axis=1)
        hit = np.nonzero(touching)[0]
        last = int(hit[-1]) if len(hit) else -1
        if last >= len(sc) - 1:
            return "branch never clears"
        starts[m] = float(sc[last + 1]) + 0.5 * (sc[1] - sc[0])
        if starts[m] + edge > tb.line.length:
            return "branch too short"
    return dict(root=root.tid, members=list(members), k=k, field=field_, s_lo=s_lo,
                s_hi=s_hi, starts=starts, r_root=r_root)


def _euler_loop(rings: list, seams: dict) -> list:
    """One closed loop over several rings joined by seams (a tree): round ring
    0 from its first vertex; at a seam's end, out along the seam, round the
    other ring and back -- every ring in its own order, so the loop winds the
    way they all do. `seams[(a, b)]` = (vertex index on ring a, index on ring
    b, interior vertex ids from a to b)."""
    adj: dict[int, list] = {}
    for (a, b), (ka, kb, path) in seams.items():
        adj.setdefault(a, []).append((ka, b, kb, list(path)))
        adj.setdefault(b, []).append((kb, a, ka, list(path)[::-1]))

    def tour(r, entry, came_from):
        ring = rings[r]
        n = len(ring)
        seq = []
        for step in range(n + 1):
            k = (entry + step) % n
            seq.append(ring[k])
            if step == n:
                break
            for kk, other, k_other, path in adj.get(r, []):
                if kk == k and other != came_from:
                    seq += path
                    seq += tour(other, k_other, r)
                    seq += path[::-1]
                    seq.append(ring[k])
        return seq

    loop = tour(0, 0, -1)
    return loop[:-1]




def _hub_pair(cl: dict, tubes: dict) -> bool:
    """Whether a cluster is one the hub welds: two branches, both leaving the
    root stem, on roughly opposite sides of it (more than 120 deg apart)."""
    return _opposite_pair(tubes[cl["root"]], cl["members"], tubes,
                          0.5 * (cl["s_lo"] + cl["s_hi"]))


def _opposite_pair(root: _Tube, members, tubes: dict, s_mid: float) -> bool:
    """Two branches, both on `root`, more than 120 deg apart round it at s_mid."""
    if len(members) != 2 or any(tubes[m].parent != root.tid for m in members):
        return False
    c, t = root.line.at(np.array([s_mid]))
    dirs = []
    for m in members:
        d = tubes[m].line.at(np.array([0.0]))[1][0]
        d = d - t[0] * float(d @ t[0])
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return False
        dirs.append(d / n)
    return float(dirs[0] @ dirs[1]) < math.cos(math.radians(120.0))


def _hub_weld(out: _Builder, cl: dict, tubes: dict, vid, names, edge: float) -> bool:
    """Mesh a pair of branches and their stem as a hub: a small cube at the
    junction, its bottom and top faces open to the stem below and above, its two
    sides open to the branches, front and back a plain grid of quads. Every ring
    has N = 4m vertices (`hub_sides`): each splits into four sides of m, and
    neighbouring ports share the cube's edges, so the four tubes meet edge to
    edge and nothing converts 1 to 3. Each port then runs to its tube's real
    ring in plain rows of N quads. All of it laid onto the weld surface, cast
    out from the junction's skeleton. False (nothing written) if the pair's
    rings do not all hold N vertices or the cube will not fit the cut."""
    if not _hub_pair(cl, tubes):
        return False
    root = tubes[cl["root"]]
    m1, m2 = cl["members"]
    field_ = cl["field"]
    r_lo = cl["ring_lo"]
    N = root.sides
    if N % 4 or tubes[m1].sides != N or tubes[m2].sides != N:
        return False
    mm = N // 4
    rings = {"bottom": [vid(root, r_lo, k) for k in range(N)],
             "top": [vid(root, r_lo + 1, k) for k in range(N)],
             "a": [vid(tubes[m1], 0, k) for k in range(N)],
             "b": [vid(tubes[m2], 0, k) for k in range(N)]}

    # The cube's frame: up the stem, x toward branch a, y in front. Each side
    # port sits where its branch meets the stem's surface and is as tall as
    # that footprint: a branch leaving at angle theta to the stem crosses its
    # surface r_c cot(theta) above its node, in an oval r / sin(theta) tall.
    # (At the node's own height and the branch's radius, a rising branch's
    # upper crotch lay far above the cube's corner, and the stem's rows and the
    # branch's were both cast onto it and folded over each other.) The cube is
    # centred between the two footprints and sheared from one to the other --
    # a slid pair's are at two heights.
    s_a, s_b = float(root.ring_s[r_lo]), float(root.ring_s[r_lo + 1])
    s_m = 0.5 * (cl["s_lo"] + cl["s_hi"])
    cc, up = root.line.at(np.array([s_m]))
    up = up[0]
    r_c = float(root.line.attr("width", s_m))
    foot = []
    for m in (m1, m2):
        p0, d0 = tubes[m].line.at(np.array([0.0]))
        cos_t = float(np.clip(d0[0] @ up, -0.87, 0.87))
        sin_t = math.sqrt(1.0 - cos_t * cos_t)
        r_m = float(tubes[m].line.attr("width", tubes[m].ring_s[0]))
        foot.append((float((p0[0] - cc[0]) @ up) + r_c * cos_t / sin_t, r_m / sin_t, r_m))
    (z_a, h_a, r_a), (z_b, h_b, r_b) = foot
    s_c = s_m + 0.5 * (z_a + z_b)
    s_c = min(max(s_c, s_a + 0.25 * (s_b - s_a)), s_b - 0.25 * (s_b - s_a))
    cc, up = root.line.at(np.array([s_c]))
    C, up = cc[0], up[0]
    z_a, z_b = z_a + s_m - s_c, z_b + s_m - s_c
    toward = out.frozen_pos[np.array(rings["a"])].mean(axis=0) - C
    ax = toward - up * float(toward @ up)
    ax /= max(float(np.linalg.norm(ax)), 1e-12)
    ay = np.cross(up, ax)
    r_c = float(root.line.attr("width", s_c))
    hx, hy = r_c, 0.5 * (r_a + r_b)
    # room: the cube's lowest and highest corner a clear 40% of the way from
    # the cut's ends, the rows to the stem's rings get the rest
    lo_room = 0.6 * (s_c - s_a)
    hi_room = 0.6 * (s_b - s_c)
    z_a = float(np.clip(z_a, -lo_room, hi_room))
    z_b = float(np.clip(z_b, -lo_room, hi_room))
    fit = min(1.0, (lo_room + min(z_a, z_b)) / max(max(h_a, h_b), 1e-12),
              (hi_room - max(z_a, z_b)) / max(max(h_a, h_b), 1e-12))
    if fit * min(h_a, h_b) <= 0.2 * r_c:
        return False
    hz_a, hz_b = fit * h_a, fit * h_b
    cl["hub_frame"] = dict(r_c=r_c, r_a=r_a, r_b=r_b, hz_a=hz_a, hz_b=hz_b, z_a=z_a, z_b=z_b,
                           s_a=s_a, s_b=s_b, s_c=s_c, fit=fit, C=C, up=up, ax=ax, ay=ay)

    # the cube's surface lattice: (i, j, k) in 0..mm with one coordinate at an end
    lattice = {}
    pts = []
    for i in range(mm + 1):
        for j in range(mm + 1):
            for k in range(mm + 1):
                if min(i, j, k) > 0 and max(i, j, k) < mm:
                    continue
                lattice[(i, j, k)] = len(pts)
                w = i / mm
                shear = z_b + (z_a - z_b) * w
                hz = hz_b + (hz_a - hz_b) * w
                pts.append(C + (2.0 * w - 1.0) * hx * ax + (2.0 * j / mm - 1.0) * hy * ay
                           + ((2.0 * k / mm - 1.0) * hz + shear) * up)
    pts = np.array(pts)

    # the junction's skeleton, every new point cast out from its nearest point
    spine = []
    for tb, lo, hi in [(root, s_a, s_b)] + [(tubes[x], 0.0, float(tubes[x].ring_s[0]))
                                            for x in (m1, m2)]:
        sp, _ = tb.line.at(np.linspace(max(lo, 0.0), min(hi, tb.line.length), 9))
        spine += [(sp[i], sp[i + 1]) for i in range(len(sp) - 1)]
    seg_a = np.array([a for a, _b in spine])
    seg_ab = np.array([b - a for a, b in spine])
    seg_l2 = np.maximum(np.sum(seg_ab * seg_ab, axis=1), 1e-18)
    reach = 2.0 * r_c

    def onto_surface(P):
        t = np.clip(np.einsum("nmk,mk->nm", P[:, None, :] - seg_a[None], seg_ab) / seg_l2[None],
                    0.0, 1.0)
        near = seg_a[None] + t[..., None] * seg_ab[None]
        k = np.argmin(_norm(P[:, None, :] - near, axis=2), axis=1)
        ray = P - near[np.arange(len(P)), k]
        d = _norm(ray, axis=1)
        ray /= np.maximum(d, 1e-15)[:, None]
        # a point outside casts in no further than the skeleton: past it, a
        # thin branch's far side is outside again, the search found no
        # crossing and the point was left floating off the surface
        out_ = field_(P)[0] > 0.0
        hit, ok = field_.cast(P, ray, np.where(out_, 0.95 * d, reach))
        return np.where(ok[:, None], hit, P)

    pts = onto_surface(pts)

    def square(fixed_axis, value):
        """The 4m lattice points round one cube face, in order."""
        out_ = []
        for u in range(mm):
            out_.append((u, 0))
        for v in range(mm):
            out_.append((mm, v))
        for u in range(mm, 0, -1):
            out_.append((u, mm))
        for v in range(mm, 0, -1):
            out_.append((0, v))
        keys = []
        for u, v in out_:
            c3 = [0, 0, 0]
            free = [a for a in range(3) if a != fixed_axis]
            c3[fixed_axis] = value
            c3[free[0]], c3[free[1]] = u, v
            keys.append(lattice[tuple(c3)])
        return keys

    ports = {"bottom": square(2, 0), "top": square(2, mm), "a": square(0, mm), "b": square(0, 0)}
    tube_dir = {"bottom": up, "top": up,
                "a": tubes[m1].line.at(np.array([0.0]))[1][0],
                "b": tubes[m2].line.at(np.array([0.0]))[1][0]}

    def sense(P, d):
        c = P.mean(axis=0)
        return float(np.sum(np.cross(P - c, np.roll(P, -1, axis=0) - c) @ d))

    # new vertices: the lattice, then each port's rows out to its ring
    first = out.count
    new_pos = list(pts)
    faces = []
    parts = []                      # which piece each face is (debug)
    for name in ("bottom", "top", "a", "b"):
        ring = rings[name]
        Q = out.frozen_pos[np.array(ring)]
        port = list(ports[name])
        P = pts[np.array(port)]
        if sense(P, tube_dir[name]) * sense(Q, tube_dir[name]) < 0.0:
            port = port[::-1]
            P = P[::-1]
        offset = min(range(N), key=lambda o: float(np.sum(_norm(
            np.roll(P, -o, axis=0) - Q, axis=1))))
        port = port[offset:] + port[:offset]
        P = pts[np.array(port)]
        gap = float(np.mean(_norm(P - Q, axis=1)))
        rows = max(1, int(round(gap / max(edge, 1e-9))))
        if name in ("a", "b"):
            rows = max(rows, _HUB_SIDE_ROWS)    # the branch turns out across them
        # rows along the tube's own direction: the bottom runs ring -> port,
        # the top and the branches port -> ring
        start, end = (Q, P) if name == "bottom" else (P, Q)
        start_ids = ring if name == "bottom" else [first + v for v in port]
        end_ids = [first + v for v in port] if name == "bottom" else ring
        chain = [start_ids]
        for r in range(1, rows):
            t = r / rows
            mid = onto_surface(start * (1.0 - t) + end * t)
            base = first + len(new_pos)
            new_pos.extend(mid)
            chain.append(list(range(base, base + N)))
        chain.append(end_ids)
        for r_i, (ra, rb) in enumerate(zip(chain[:-1], chain[1:], strict=True)):
            for k in range(N):
                k1 = (k + 1) % N
                faces.append([ra[k], ra[k1], rb[k1], rb[k]])
                parts.append((name, r_i if name != "bottom" else rows - 1 - r_i))
    # front and back: the cube's +-y faces, wound outward
    for j, flip in ((0, False), (mm, True)):
        for i in range(mm):
            for k in range(mm):
                f = [lattice[(i, j, k)], lattice[(i + 1, j, k)], lattice[(i + 1, j, k + 1)],
                     lattice[(i, j, k + 1)]]
                f = [first + v for v in f]
                faces.append(f[::-1] if flip else f)
                parts.append(("back" if flip else "front", 0))

    new_pos = np.array(new_pos)
    # Relaxed on the surface, the rings held: each new point toward its
    # neighbours' mean and cast back out. The cube's edges land a little off
    # the real crotch lines and the rows leaving them start crossed; on a
    # layout this regular the relaxation evens them out.
    nbrs: dict = {}
    for f in faces:
        for i in range(4):
            a, b = f[i], f[(i + 1) % 4]
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
    movable = np.array(sorted(v - first for v in nbrs if v >= first))
    if len(movable):
        adj = [np.array(sorted(nbrs[first + v])) for v in movable]
        for _ in range(_HUB_RELAX):
            def at(ids):
                return np.where((ids >= first)[:, None], new_pos[np.clip(ids - first, 0, None)],
                                out.frozen_pos[np.minimum(ids, len(out.frozen_pos) - 1)])
            mean = np.array([at(a).mean(axis=0) for a in adj])
            cur = new_pos[movable]
            new_pos[movable] = onto_surface(cur + 0.5 * (mean - cur))
    if len(movable) and _HUB_FILL:
        # Smoothed in space, a few passes: each point toward its neighbours'
        # mean, the part of the move into the surface dropped (a convex side
        # does not shrink). Where a crotch's fillet is narrower than a quad, a
        # vertex on it sits up to half an edge below its neighbours -- a dent
        # in the shading -- and comes up into line; crossed rows untangle.
        # (Run to convergence, 30 passes, it filled every crotch like a soap
        # film -- a web up to a stem radius thick where the rings were far
        # apart; held on the surface, the dents stayed.)
        def smooth(P, cap):
            P = P.copy()
            for _ in range(_HUB_FILL):
                def at(ids):
                    return np.where((ids >= first)[:, None], P[np.clip(ids - first, 0, None)],
                                    out.frozen_pos[np.minimum(ids, len(out.frozen_pos) - 1)])
                mean = np.array([at(a).mean(axis=0) for a in adj])
                cur = P[movable]
                _f, g = field_(cur)
                nrm = g / np.maximum(_norm(g, axis=1, keepdims=True), 1e-15)
                move = 0.5 * (mean - cur)
                into = np.minimum(np.sum(move * nrm, axis=1, keepdims=True), 0.0)
                nxt = cur + move - nrm * into
                if cap >= 0.0:
                    # back down to the cap, along the gradient
                    f2, g2 = field_(nxt)
                    over = np.maximum(f2 - cap * r_c, 0.0)
                    nxt = nxt - (over / np.maximum(np.sum(g2 * g2, axis=1), 0.25))[:, None] * g2
                P[movable] = nxt
            return P

        def worst_bend(P):
            """The sharpest bend between two of the hub's faces (deg)."""
            q = np.where((np.array(faces) >= first)[..., None],
                         P[np.clip(np.array(faces) - first, 0, None)],
                         out.frozen_pos[np.minimum(np.array(faces), len(out.frozen_pos) - 1)])
            n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
            n /= np.maximum(_norm(n, axis=1, keepdims=True), 1e-18)
            owner = {}
            for fi, f in enumerate(faces):
                for i in range(4):
                    owner[(f[i], f[(i + 1) % 4])] = fi
            worst = 0.0
            for (a_, b_), fi in owner.items():
                gi = owner.get((b_, a_))
                if gi is not None and gi > fi:
                    worst = max(worst, float(np.degrees(np.arccos(np.clip(n[fi] @ n[gi], -1, 1)))))
            return worst

        # Held near the surface first (the trunk straight, the crotch tight);
        # a hub that folds that way -- a strongly slid pair on a thin stem,
        # its sheared cube's corner a sliver -- is smoothed freely instead.
        base = new_pos
        new_pos = smooth(base, _HUB_FILL_CAP)
        if worst_bend(new_pos) > _HUB_FOLD_DEG:
            new_pos = smooth(base, -1.0)
            cl["hub_soft"] = True
    # attributes: each new vertex its nearest ring vertex's
    ring_ids = np.array([v for r in rings.values() for v in r])
    ring_pos = out.frozen_pos[ring_ids]
    near = np.argmin(_norm(new_pos[:, None, :] - ring_pos[None], axis=2), axis=1)
    src = ring_ids[near]
    # UVs: each face on the tube it lies on, as the single weld does
    owners = [(root, cl["s_lo"] - 2.0 * cl["r_root"], cl["s_hi"] + 2.0 * cl["r_root"])]
    owners += [(tubes[x], 0.0, float(tubes[x].ring_s[0]) + 2.0 * float(
        tubes[x].line.attr("width", tubes[x].ring_s[0]))) for x in (m1, m2)]

    def position(v):
        return new_pos[v - first] if v >= first else out.frozen_pos[v]

    used = sorted({v for f in faces for v in f})
    row = {v: i for i, v in enumerate(used)}
    where = np.array([position(v) for v in used])
    tube_uv = [_tube_uv(tb, where, lo, hi) for tb, lo, hi in owners]
    centres = np.array([where[[row[v] for v in f]].mean(axis=0) for f in faces])
    face_owner = field_.owner(centres)
    v_owner = field_.owner(new_pos)
    block = {nm: out.frozen_attr[nm][src] for nm in names}
    pg = np.stack([tube_uv[o][row[first + i]] if (first + i) in row else (0.0, 0.0, 1.0)
                   for i, o in enumerate(v_owner.tolist())])
    block["pg_u"] = pg[:, 0] % 1.0
    block["pg_v"] = pg[:, 1]
    assert out.add_block(new_pos, block) == first
    uv_faces = []
    for f, o in zip(faces, face_owner.tolist(), strict=True):
        uvs = tube_uv[o][[row[v] for v in f]]
        u = uvs[:, 0] + np.round(uvs[0, 0] - uvs[:, 0])
        uv = [(float(a_) * sp, float(b_)) for a_, b_, sp in zip(u, uvs[:, 1], uvs[:, 2],
                                                               strict=True)]
        out.poly(f, uv)
        uv_faces.append((f, uv))
    cl["faces"] = uv_faces
    cl["hub_parts"] = parts
    cl["hub"] = True
    return True

def _level_stacks(out: _Builder, cl: dict, root: _Tube, vid, names, edge: float):
    """The level loops of a cluster's weld: the stem's own rings carried up from
    its ring below the cut and down from its ring above it, each a full ring at
    one height along the stem -- the stem's columns, laid out on the weld surface
    -- for as long as it clears every branch, with quads between. Returns (top
    ring of the lower stack, its centre, bottom ring of the upper stack, its
    centre, the stem's tangent there): the loops the band then joins through the
    branches. None if no ring clears below and above (the band runs the joint)."""
    field_ = cl["field"]
    r_lo = cl["ring_lo"]
    s_a, s_b = float(root.ring_s[r_lo]), float(root.ring_s[r_lo + 1])
    if s_b - s_a <= 2.0 * edge:
        return None
    sec = _section(root.section, root.sides)
    n_a, n_b = root.frame_n[r_lo], root.frame_n[r_lo + 1]

    def ring_at(s):
        t = (s - s_a) / max(s_b - s_a, 1e-12)
        c, tan = root.line.at(np.array([s]))
        c, tan = c[0], tan[0]
        nrm = n_a * (1.0 - t) + n_b * t
        nrm -= tan * float(nrm @ tan)
        nrm /= max(float(np.linalg.norm(nrm)), 1e-12)
        bi = np.cross(tan, nrm)
        w = float(root.line.attr("width", s))
        P = c + w * (sec[:, 0, None] * nrm + sec[:, 1, None] * bi)
        rad = P - c
        rad /= np.maximum(_norm(rad, axis=1, keepdims=True), 1e-12)
        hit, ok = field_.cast(P, rad, 2.0 * w)
        return np.where(ok[:, None], hit, P), c, tan, t

    # where a level ring stops clearing the branches: some point of it lies on
    # a branch's own surface (the union's owner is not the stem)
    ss = np.linspace(s_a, s_b, 40)
    on_branch = np.array([bool((field_.owner(ring_at(x)[0]) != 0).any()) for x in ss])
    if not on_branch.any():
        return None
    gap = 0.3 * edge
    s_low = float(ss[np.argmax(on_branch)]) - gap        # last clear height from below
    s_high = float(ss[len(ss) - 1 - np.argmax(on_branch[::-1])]) + gap

    def heights(room):
        """Evenly over the clear room, about an edge apart; one ring from half
        an edge of room (a compact pair leaves only a few millimetres)."""
        if room < 0.5 * edge:
            return []
        n = max(1, int(room / max(edge, 1e-9) + 0.5))
        return [room * i / n for i in range(1, n + 1)]

    lows = [s_a + h for h in heights(s_low - s_a)]
    highs = [s_b - h for h in heights(s_b - s_high)][::-1]
    if not lows or not highs:
        return None

    ns = root.sides
    lo_ring = [vid(root, r_lo, k) for k in range(ns)]
    hi_ring = [vid(root, r_lo + 1, k) for k in range(ns)]
    a_lo = np.stack([out.frozen_attr[nm][lo_ring] for nm in names], axis=1)
    a_hi = np.stack([out.frozen_attr[nm][hi_ring] for nm in names], axis=1)
    stack_ids = {}
    centres = {}
    for s_ in lows + highs:
        P, c, tan, t = ring_at(s_)
        block = {nm: a_lo[:, i] * (1.0 - t) + a_hi[:, i] * t for i, nm in enumerate(names)}
        block["pg_u"] = np.arange(ns) / ns
        block["pg_v"] = np.full(ns, float(root.v_at(s_)))
        start = out.add_block(P, block)
        stack_ids[s_] = list(range(start, start + ns))
        centres[s_] = (c, tan)

    def strip(ring_a, s_ring_a, ring_b, s_ring_b):
        """Quads between two rings of the stem's columns, wound as its own."""
        v0, v1 = float(root.v_at(s_ring_a)), float(root.v_at(s_ring_b))
        sp0, sp1 = float(root.u_span(s_ring_a)), float(root.u_span(s_ring_b))
        for k in range(ns):
            k1 = (k + 1) % ns
            a0, a1 = k / ns, (k + 1) / ns
            out.poly([ring_a[k], ring_a[k1], ring_b[k1], ring_b[k]],
                     [(a0 * sp0, v0), (a1 * sp0, v0), (a1 * sp1, v1), (a0 * sp1, v1)])

    chain_lo = [(s_a, lo_ring)] + [(x, stack_ids[x]) for x in lows]
    chain_hi = [(x, stack_ids[x]) for x in highs] + [(s_b, hi_ring)]
    for (sa_, ra_), (sb_, rb_) in zip(chain_lo[:-1], chain_lo[1:], strict=True):
        strip(ra_, sa_, rb_, sb_)
    for (sa_, ra_), (sb_, rb_) in zip(chain_hi[:-1], chain_hi[1:], strict=True):
        strip(ra_, sa_, rb_, sb_)
    out.freeze()
    top_lo, bot_hi = lows[-1], highs[0]
    return (stack_ids[top_lo], centres[top_lo][0], stack_ids[bot_hi], centres[bot_hi][0],
            centres[bot_hi][1])

def _cluster_weld(out: _Builder, cl: dict, tubes: dict, vid, names, tiles, q, edge) -> None:
    """Mesh one cluster's weld: the root stem's ring below it into one loop made
    of the root's ring above it and every branch's first ring, joined by seams
    walked across the crotches between them."""
    root = tubes[cl["root"]]
    field_ = cl["field"]
    r_lo = cl["ring_lo"]
    lower = [vid(root, r_lo, k) for k in range(root.sides)]
    upper = [vid(root, r_lo + 1, k) for k in range(root.sides)]
    c_lo, c_hi, t_hi = root.centre[r_lo], root.centre[r_lo + 1], root.tan[r_lo + 1]
    stacks = _level_stacks(out, cl, root, vid, names, edge)
    if stacks is not None:
        lower, c_lo, upper, c_hi, t_hi = stacks
    cl["level"] = stacks is not None
    b_ids = np.array(lower)
    radial = out.frozen_pos[b_ids] - c_lo
    radial /= np.maximum(_norm(radial, axis=1, keepdims=True), 1e-12)
    rings = [list(upper)]
    for m in cl["members"]:
        rings.append([vid(tubes[m], 0, k) for k in range(tubes[m].sides)])
    pos = [out.frozen_pos[np.array(r)] for r in rings]

    # the seams: a tree over the rings. A branch leaving the root stem is joined
    # to the root's upper ring (ring 0) from the vertex facing that branch round
    # the stem, so a pair splits the ring into matching front and back halves;
    # any other ring (a branch's branch) to its nearest by the shortest pair of
    # vertices. Walked across the surface between them.
    n = len(rings)
    best = {}
    for a in range(n):
        for b in range(a + 1, n):
            d = _norm(pos[a][:, None, :] - pos[b][None, :, :], axis=2)
            ka, kb = np.unravel_index(int(np.argmin(d)), d.shape)
            best[(a, b)] = (float(d[ka, kb]), int(ka), int(kb))
    c1, t1 = c_hi, t_hi
    out0 = pos[0] - c1
    out0 -= np.outer(out0 @ t1, t1)
    out0 /= np.maximum(_norm(out0, axis=1, keepdims=True), 1e-15)
    on_root = {i + 1 for i, m in enumerate(cl["members"]) if tubes[m].parent == root.tid}
    for b in on_root:
        toward = pos[b].mean(axis=0) - c1
        toward -= t1 * float(toward @ t1)
        ka = int(np.argmax(out0 @ (toward / max(float(np.linalg.norm(toward)), 1e-15))))
        kb = int(np.argmin(_norm(pos[b] - pos[0][ka], axis=1)))
        best[(0, b)] = (-1.0, ka, kb)            # ahead of every nearest pair
    in_tree, plan = {0}, []
    used_at = {r: set() for r in range(n)}
    while len(in_tree) < n:
        cand = [(v[0], a, b) for (a, b), v in best.items() if (a in in_tree) != (b in in_tree)]
        _, a, b = min(cand)
        _, ka, kb = best[(a, b)]
        if a not in in_tree:
            a, b, ka, kb = b, a, kb, ka
        # two seams may not leave a ring from the same vertex
        while ka in used_at[a]:
            ka = (ka + 1) % len(rings[a])
        used_at[a].add(ka)
        used_at[b].add(kb)
        pa, pb = pos[a][ka], pos[b][kb]
        tgt = 0.5 * (float(np.mean(_norm(np.roll(pos[a], -1, axis=0) - pos[a], axis=1)))
                     + float(np.mean(np.linalg.norm(np.roll(pos[b], -1, axis=0) - pos[b],
                                                    axis=1))))
        length = float(np.linalg.norm(pb - pa))
        plan.append([a, b, ka, kb, max(2, int(round(length / max(tgt, 1e-9))) + 1)])
        in_tree.add(b)
    # The seams off the root's ring share one vertex count -- the larger -- so the
    # loop carries as many vertices across the front of the joint as the back
    # (a 2-vertex seam on one side and a 9-vertex one on the other left the
    # rows from below crowding one side where they met at the top).
    off_root = [row for row in plan if row[0] == 0 and row[1] in on_root]
    if len(off_root) > 1:
        shared = max(row[4] for row in off_root)
        for row in off_root:
            row[4] = shared
    seams = {}
    for a, b, ka, kb, n_q in plan:
        pa, pb = pos[a][ka], pos[b][kb]
        length = float(np.linalg.norm(pb - pa))
        path_ids = []
        if n_q > 2:
            _, g = field_(pa[None, :])
            g = g / np.maximum(_norm(g, axis=1, keepdims=True), 1e-12)
            axis = (pb - pa) / max(length, 1e-12)
            path = _walk(field_, pa[None, :], pb[None, :], g, 0.5 * (pa + pb), axis, n_q)[0]
            inner = field_.project(path[1:-1], iters=3)
            ia = out.frozen_attr
            va, vb = rings[a][ka], rings[b][kb]
            ts = np.linspace(0.0, 1.0, n_q)[1:-1]
            block = {nm: ia[nm][va] * (1.0 - ts) + ia[nm][vb] * ts for nm in names}
            block["pg_u"] = np.zeros(len(inner))
            block["pg_v"] = np.zeros(len(inner))
            start = out.add_block(inner, block)
            path_ids = list(range(start, start + len(inner)))
        seams[(a, b)] = (ka, kb, path_ids)
    out.freeze()
    loop = np.array(_euler_loop(rings, seams), dtype=np.int64)
    via = c_hi
    w0 = float(np.mean([tubes[m].line.attr("width", tubes[m].ring_s[0]) for m in cl["members"]]))
    owners = [(root, cl["s_lo"] - 2.0 * cl["r_root"], cl["s_hi"] + 2.0 * cl["r_root"])]
    owners += [(tubes[m], 0.0, float(tubes[m].ring_s[0]) + 2.0 * float(
        tubes[m].line.attr("width", tubes[m].ring_s[0]))) for m in cl["members"]]
    cl["seams"] = {k: v[2] for k, v in seams.items()}
    # Meridians walked toward the far side of a branch that leaves another
    # branch pick either way round it and cross: the relaxation untangles them,
    # given the passes (66 folded faces at 16, 9 at 200 on a split branch).
    _bridge(out, field_, b_ids, radial, loop, via, root.tan[r_lo], w0, owners, names, tiles,
            q, edge, cl, align=True, relax_iters=_CLUSTER_RELAX)


def _tube_uv(tb: _Tube, P: np.ndarray, s_lo: float, s_hi: float) -> np.ndarray:
    """(u, v, span) of every point in P on the tube's own island: u its angle
    round the tube (0..1, in the ring frames the tube's columns were laid in),
    v the tube's V at its nearest point along the centre line within
    [s_lo, s_hi], span the tiles once round the tube there (U = u x span)."""
    line = tb.line
    keep = np.nonzero((line.s >= s_lo - 1e-9) & (line.s <= s_hi + 1e-9))[0]
    if len(keep) < 2:
        keep = np.arange(len(line.s))
    a, b = line.dense[keep[:-1]], line.dense[keep[1:]]
    sa, sb = line.s[keep[:-1]], line.s[keep[1:]]
    ab = b - a
    l2 = np.maximum(np.sum(ab * ab, axis=1), 1e-18)
    out = np.zeros((len(P), 3))
    for start in range(0, len(P), 4096):
        Q = P[start:start + 4096]
        t = np.clip(np.einsum("nmk,mk->nm", Q[:, None, :] - a[None], ab) / l2[None], 0.0, 1.0)
        d = _norm(Q[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2)
        m = np.argmin(d, axis=1)
        s = sa[m] + (sb[m] - sa[m]) * t[np.arange(len(Q)), m]
        centre, tan = line.at(s)
        fn = np.stack([np.interp(s, tb.ring_s, tb.frame_n[:, c]) for c in range(3)], axis=1)
        fn -= tan * np.sum(fn * tan, axis=1, keepdims=True)
        fn /= np.maximum(_norm(fn, axis=1, keepdims=True), 1e-12)
        fb = np.cross(tan, fn)
        rel = Q - centre
        out[start:start + 4096, 0] = (np.arctan2(np.sum(rel * fb, axis=1),
                                                 np.sum(rel * fn, axis=1)) / (2.0 * math.pi)) % 1.0
        out[start:start + 4096, 1] = tb.v_at(s)
        out[start:start + 4096, 2] = tb.u_span(s)
    return out


def _unit_starts(cost: np.ndarray, r: int) -> set:
    """Where the r 3-to-1 units of a strip go on its longer ring (n vertices):
    the r cheapest starts, no two within 4 vertices of each other (a unit
    spans 3 edges). Falls back to even spacing if the cheap ones will not
    fit apart."""
    n = len(cost)
    if r <= 0:
        return set()
    even = {int(round(k * n / r + (n / r - 3.0) / 2.0)) % n for k in range(r)}
    chosen: list[int] = []
    for i in np.argsort(cost, kind="stable"):
        i = int(i)
        if i > n - 3:
            continue
        if all(min(abs(i - c), n - abs(i - c)) >= 4 for c in chosen):
            chosen.append(i)
            if len(chosen) == r:
                return set(chosen)
    return even


def _quad_strip(ring_a: list, ring_b: list, midpoint, cost=None) -> list:
    """Quads between two closed rings (both from u = 0, same winding) whose
    counts differ by an even number: where they line up, one quad per edge;
    each 3-to-1 unit takes three edges of the longer ring down to one of the
    shorter through two new vertices and four quads -- placed where `cost`
    (per vertex of the longer ring, as a unit start) is lowest, evenly round
    without one. Faces keep the convention (a_k, a_k+1, b_k+1, b_k)."""
    flip = len(ring_a) < len(ring_b)
    a, b = (ring_b, ring_a) if flip else (ring_a, ring_b)
    na, nb = len(a), len(b)
    r = (na - nb) // 2
    starts = _unit_starts(np.zeros(na) if cost is None else np.asarray(cost), r)
    faces = []
    i = jj = 0
    while i < na:
        if i in starts:
            a0, a1, a2, a3 = a[i % na], a[(i + 1) % na], a[(i + 2) % na], a[(i + 3) % na]
            b0, b1 = b[jj % nb], b[(jj + 1) % nb]
            m0 = midpoint(a1, b0, b1, 1.0 / 3.0)
            m1 = midpoint(a2, b0, b1, 2.0 / 3.0)
            faces += [(a0, a1, m0, b0), (a1, a2, m1, m0), (a2, a3, b1, m1), (m0, m1, b1, b0)]
            i += 3
        else:
            faces.append((a[i % na], a[(i + 1) % na], b[(jj + 1) % nb], b[jj % nb]))
            i += 1
        jj += 1
    if flip:
        faces = [f[::-1] for f in faces]
    return faces, sorted(starts)

_RELAX_ITERS = 16
_COLLAPSE_FRAC = 0.3
_WELD_MIN_SIDES = 14
_LIGHT_STALK_SIDES = 4
"""A light stalk's sides. Even: a hole's border is. It keeps them through the weld
pass, whose floor had a 1 mm pedicel at 14."""
_RING_FLOOR = 0.25
_ROW_H = 1.5
_GATE_NEAR, _GATE_FAR = 90.0, 170.0
_GRID_ITERS = 48
_GRID_ACROSS = 0.5
_LOOP_MATCH_MAX = 2.0
"""Most a hole's border may outnumber its branch's ring for the weld to keep the
border's count (`_bridge`)."""
_PAIR_LIFT = 0.15
"""How far the stem's side between an opposite pair may still be lifted by their
fillets for each to take a single weld in its half (`mirrored`), in stem radii."""
_HUB_RELAX = 100
"""Relaxation passes over a hub's new points (`_hub_weld`)."""
_HUB_PAD = 1.0
"""How far a hub's stretch of stem reaches past the crotches, in stem radii."""
_HUB_FILL = 5
"""Smoothing passes over a hub's new points after the relaxation (0 = off): each
toward its neighbours in space, never into the surface -- a dented vertex comes
up into line, crossed rows untangle. Few on purpose: more spread the lift into
a web across the crotch (`_hub_weld`)."""
_HUB_FILL_CAP = 0.05
"""How far that smoothing may lift a point off the weld surface, in stem radii
(unbounded, it rounded the crotches and bowed the trunk; a hub that folds
anyway takes that free smoothing, `_HUB_FOLD_DEG`)."""
_HUB_SIDE_ROWS = 2
"""Fewest rows from a hub's side port to its branch (one took the whole turn out
of the stem in a single pinched quad)."""
_HUB_FOLD_DEG = 150.0
"""A hub whose held smoothing leaves two faces bent past this is smoothed
freely instead (`_hub_weld`)."""
# A side shoot this much thicker than the stem after its node has taken the
# lead: the thick path is meshed as the one tube (near-equal forks are left).
_LEAD_RATIO = 1.3
_CLUSTER_RELAX = 400


def _relax_grid(field_: _WeldField, M: np.ndarray, scale: float) -> np.ndarray:
    """Smooth the meridian field as the regular grid it is -- every point
    toward the mean of its four neighbours (the meridians either side at the
    same arc fraction, the points before and after on its own), along the
    surface, then back onto it; the border and the child's ring fixed. Walked
    meridians cross where they all funnel into a crotch; a regular grid
    relaxed with its ends held untangles them, and rings laid along smooth
    meridians follow smooth arcs."""
    M = M.copy()
    n_m, n_q = M.shape[:2]
    nrm = None
    prev_m = (np.arange(n_m) - 1) % n_m           # np.roll(M, 1, axis=0) by index
    next_m = (np.arange(n_m) + 1) % n_m
    for it in range(_GRID_ITERS):
        inner = M[:, 1:-1].reshape(-1, 3)
        if it % 4 == 0:
            f, g = field_(inner)
            gg = np.maximum(np.sum(g * g, axis=1), 0.25)
            inner = inner - np.clip(f / gg, -scale, scale)[:, None] * g
            M[:, 1:-1] = inner.reshape(n_m, n_q - 2, 3)
            nrm = g / np.sqrt(gg)[:, None]
        wu = 0.5 * _GRID_ACROSS
        mean = wu * (M[prev_m, 1:-1] + M[next_m, 1:-1]) \
            + (0.5 - wu) * (M[:, :-2] + M[:, 2:])
        move = (mean - M[:, 1:-1]).reshape(-1, 3)
        move -= nrm * np.sum(move * nrm, axis=1, keepdims=True)
        M[:, 1:-1] += 0.6 * move.reshape(n_m, n_q - 2, 3)
    inner = field_.project(M[:, 1:-1].reshape(-1, 3), iters=3, max_step=scale)
    M[:, 1:-1] = inner.reshape(n_m, n_q - 2, 3)
    return M


def _edges_of(f):
    for a, b in zip(f, f[1:] + f[:1], strict=True):
        yield (a, b) if a < b else (b, a)


def _collapse_short(faces, pts, attrs, first: int, fixed_pos, frac: float):
    """Collapse every weld edge shorter than `frac` x the weld's median edge,
    shortest first: its free end is merged into the other (both free: into
    their midpoint). A collapse is kept only if the faces stay a closed
    2-manifold patch (every edge used as often as before -- inner edges twice,
    the border's and the branch ring's once) with no face repeating a vertex.
    Returns the new faces, the surviving free vertices (renumbered from
    `first`), their attributes and the old -> new id map (None if nothing
    collapsed).

    Each trial changes only the faces holding the merged vertex, so only they
    and their edges are re-made and re-counted -- the same tests on the only
    edges whose counts can move (every other edge already passed them). A
    patch that does not start clean (a face repeating a vertex, an edge used
    more than twice) takes the full re-count every trial instead
    (`_collapse_short_full`), which decides those cases as it always did."""
    faces = [list(f) for f in faces]
    base = {}
    for f in faces:
        if len(f) < 3 or len(set(f)) < len(f):
            return _collapse_short_full(faces, pts, attrs, first, fixed_pos, frac)
        for e in _edges_of(f):
            base[e] = base.get(e, 0) + 1
    if any(n > 2 for n in base.values()):
        return _collapse_short_full(faces, pts, attrs, first, fixed_pos, frac)
    pos = {first + k: p for k, p in enumerate(pts)}

    def where(v):
        return pos[v] if v >= first else fixed_pos[v]

    lens = {e: float(np.linalg.norm(where(e[0]) - where(e[1]))) for e in base}
    limit = frac * float(np.median(list(lens.values())))
    faces_of: dict[int, set] = {}
    for i, f in enumerate(faces):
        for v in f:
            faces_of.setdefault(v, set()).add(i)
    merged: dict[int, int] = {}
    for e in sorted((e for e, ln in lens.items() if ln < limit), key=lambda e: lens[e]):
        a, b = (merged.get(e[0], e[0]), merged.get(e[1], e[1]))
        while a in merged:
            a = merged[a]
        while b in merged:
            b = merged[b]
        if a == b or (a < first and b < first):
            continue
        keep, drop = (a, b) if (a < first or (b >= first and a < b)) else (b, a)
        touched = sorted(faces_of.get(drop, ()))
        new_faces = {}
        bad = False
        for i in touched:
            g = [keep if v == drop else v for v in faces[i]]
            g = [v for j, v in enumerate(g) if v != g[j - 1]]    # repeated corner
            if len(g) >= 3 and len(set(g)) < len(g):
                bad = True
                break
            new_faces[i] = g
        if bad:
            continue
        delta: dict = {}
        for i in touched:
            for ed in _edges_of(faces[i]):
                delta[ed] = delta.get(ed, 0) - 1
            g = new_faces[i]
            if len(g) >= 3:
                for ed in _edges_of(g):
                    delta[ed] = delta.get(ed, 0) + 1
        moved = {ed: base.get(ed, 0) + d for ed, d in delta.items() if d}
        if any(n > 2 for n in moved.values()):
            continue
        if not all(n == 1 for ed, n in moved.items()
                   if base.get(ed, 0) == 1 and drop not in ed):
            continue
        if not all(n == 2 or (n == 1 and base.get(ed, 0) == 1)
                   for ed, n in moved.items() if n > 0):
            continue
        if keep >= first:
            pos[keep] = 0.5 * (pos[keep] + pos[drop])
        for ed, n in moved.items():
            if n > 0:
                base[ed] = n
            else:
                base.pop(ed, None)
        for i in touched:
            g = new_faces[i]
            if len(g) >= 3:
                faces[i] = g
                faces_of.setdefault(keep, set()).add(i)
            else:
                for v in faces[i]:
                    if v != drop:
                        faces_of.get(v, set()).discard(i)
                faces[i] = None
        faces_of.pop(drop, None)
        merged[drop] = keep
    if not merged:
        return faces, pts, attrs, None
    faces = [f for f in faces if f is not None]
    alive = sorted({v for f in faces for v in f if v >= first})
    new_id = {v: first + k for k, v in enumerate(alive)}
    remap = {}
    for v in range(first, first + len(pts)):
        t = v
        while t in merged:
            t = merged[t]
        remap[v] = new_id[t] if t >= first else t
    faces = [[new_id.get(v, v) for v in f] for f in faces]
    new_pts = np.array([pos[v] for v in alive])
    new_attrs = attrs[[v - first for v in alive]]
    return faces, new_pts, new_attrs, remap


def _collapse_short_full(faces, pts, attrs, first: int, fixed_pos, frac: float):
    """`_collapse_short` re-counting the whole patch every trial (any start)."""
    faces = [list(f) for f in faces]
    pos = {first + k: p for k, p in enumerate(pts)}

    def where(v):
        return pos[v] if v >= first else fixed_pos[v]

    def edge_counts(fs):
        c = {}
        for f in fs:
            for a, b in zip(f, f[1:] + f[:1], strict=True):
                e = (min(a, b), max(a, b))
                c[e] = c.get(e, 0) + 1
        return c

    base = edge_counts(faces)
    lens = {e: float(np.linalg.norm(where(e[0]) - where(e[1]))) for e in base}
    limit = frac * float(np.median(list(lens.values())))
    merged: dict[int, int] = {}
    for e in sorted((e for e, ln in lens.items() if ln < limit), key=lambda e: lens[e]):
        a, b = (merged.get(e[0], e[0]), merged.get(e[1], e[1]))
        while a in merged:
            a = merged[a]
        while b in merged:
            b = merged[b]
        if a == b or (a < first and b < first):
            continue
        keep, drop = (a, b) if (a < first or (b >= first and a < b)) else (b, a)
        trial = []
        for f in faces:
            g = [keep if v == drop else v for v in f]
            g = [v for i, v in enumerate(g) if v != g[i - 1]]    # repeated corner
            if len(g) >= 3:
                trial.append(g)
        if any(len(set(g)) < len(g) for g in trial):
            continue
        counts = edge_counts(trial)
        if any(n > 2 for n in counts.values()):
            continue
        boundary_ok = all(counts.get(e, 0) == 1 for e, n in base.items()
                          if n == 1 and drop not in e)
        inner_ok = all(n == 2 or (n == 1 and base.get(e, 0) == 1) for e, n in counts.items())
        if not (boundary_ok and inner_ok):
            continue
        if keep >= first:
            pos[keep] = 0.5 * (pos[keep] + pos[drop])
        faces = trial
        base = counts
        merged[drop] = keep
    if not merged:
        return faces, pts, attrs, None
    alive = sorted({v for f in faces for v in f if v >= first})
    new_id = {v: first + k for k, v in enumerate(alive)}
    remap = {}
    for v in range(first, first + len(pts)):
        t = v
        while t in merged:
            t = merged[t]
        remap[v] = new_id[t] if t >= first else t
    faces = [[new_id.get(v, v) for v in f] for f in faces]
    new_pts = np.array([pos[v] for v in alive])
    new_attrs = attrs[[v - first for v in alive]]
    return faces, new_pts, new_attrs, remap


def _relax_on_surface(field_: _WeldField, pts: np.ndarray, first: int, faces, fixed_pos,
                      scale: float, iters: int = None) -> np.ndarray:
    """Even out the transition's free vertices on the weld surface: each moves
    toward the mean of its neighbours, then back onto the surface; the border
    and the branch's ring stay put. With the boundary fixed this cannot fold
    (a convex-combination map), and it undoes the crowding where the
    meridians all funnel through the crotch's saddle."""
    n = len(pts)
    edges = set()
    for vids in faces:
        for a, b in zip(vids, vids[1:] + vids[:1], strict=True):
            edges.add((min(a, b), max(a, b)))
    e = np.array(sorted(edges), dtype=np.int64)
    # local indices: the free vertices first, then the fixed ones they touch
    fixed = np.unique(e[e < first])
    local = np.where(e >= first, e - first, n + np.searchsorted(fixed, e))
    fixed_xyz = fixed_pos[fixed]
    src = np.concatenate([local[:, 0], local[:, 1]])     # each edge both ways
    dst = np.concatenate([local[:, 1], local[:, 0]])
    keep = src < n
    src, dst = src[keep], dst[keep]
    deg = np.maximum(np.bincount(src, minlength=n)[:n], 1).astype(float)
    nrm = None
    for it in range(_RELAX_ITERS if iters is None else iters):
        if it % 4 == 0:
            # onto the surface, and its normals for the next few steps
            f, g = field_(pts)
            gg = np.maximum(np.sum(g * g, axis=1), 0.25)
            pts = pts - np.clip(f / gg, -scale, scale)[:, None] * g
            nrm = g / np.sqrt(gg)[:, None]
        allpos = np.concatenate([pts, fixed_xyz])
        mean = np.stack([np.bincount(src, weights=allpos[dst, c], minlength=n)[:n]
                         for c in range(3)], axis=1) / deg[:, None]
        # Only along the surface: the plain average cut straight across the
        # crotch's curve (by up to a centimetre) and never got back onto it.
        move = mean - pts
        move -= nrm * np.sum(move * nrm, axis=1, keepdims=True)
        pts = pts + 0.7 * move
    return field_.project(pts, iters=3, max_step=scale)
