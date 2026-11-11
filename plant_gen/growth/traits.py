"""Trait preferences -- which mesh of a component's collection each point takes.

Pure, bpy-free. Without traits a component picks among its collection's meshes
uniformly at random (`instancing._assign_type`). A Traits sheet (the Traits
panel) gives each mesh a **Weight** (how common it is overall) and, per column
-- a point channel (Age, Height, Shade, a custom attribute, a class...) -- a
**preference** `w` in -1..1:

    chance of mesh j at a point  ~  Weight_j * prod over columns of
                                    (1 + w_jc * (2 * c - 1))

where `c` is the point's 0-1 value on that column's channel. So +1 makes a
mesh's chance proportional to `c` (favoured where the channel is high), -1 to
`1 - c` (where it is low), and 0 leaves it untouched; values between blend.
The pick stays random -- the same seeded draw the uniform pick used, now walked
through these weights -- so neighbouring points still mix.

- A column whose channel is not on the points reads as 0.5: no effect.
- A point where every mesh scores 0 falls back to the Weights alone, and if
  those are all 0 too, to the plain uniform pick -- a sheet never silently
  stops a component drawing.
- `is_neutral`: every Weight equal and every preference 0 is exactly the
  uniform pick; the caller then keeps the old code path, byte-identical.
"""

from __future__ import annotations

import numpy as np


def is_neutral(weights, prefs) -> bool:
    """True when the sheet cannot change the uniform pick: no meshes, all
    Weights equal (and positive) and every preference 0."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return True
    p = np.asarray(prefs, dtype=np.float64)
    return bool(np.all(w == w[0]) and w[0] > 0.0 and (p.size == 0 or not np.any(p)))


def chances(weights, prefs, values) -> np.ndarray:
    """Per point, each mesh's unnormalised chance.

    `weights` (k,): each mesh's Weight (>= 0).
    `prefs` (k, m): each mesh's preference per column, -1..1.
    `values` (n, m): each point's 0-1 value per column (NaN = channel absent).
    Returns (n, k)."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    k = w.shape[0]
    p = np.clip(np.asarray(prefs, dtype=np.float64).reshape(k, -1), -1.0, 1.0)
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    v = v.reshape(n, p.shape[1])
    v = np.where(np.isnan(v), 0.5, np.clip(v, 0.0, 1.0))
    out = np.broadcast_to(w, (n, k)).copy()
    for col in range(p.shape[1]):
        out *= np.clip(1.0 + p[None, :, col] * (2.0 * v[:, col:col + 1] - 1.0), 0.0, None)
    return out


def pick(weights, prefs, values, draw) -> np.ndarray:
    """Each point's mesh index (0..k-1), walking its seeded uniform `draw`
    (n,), in [0, 1), through the normalised chances."""
    ch = chances(weights, prefs, values)
    n, k = ch.shape
    total = ch.sum(axis=1)
    # A point every mesh scored 0: the Weights alone, then uniform.
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    fallback = w if w.sum() > 0.0 else np.ones(k)
    dead = total <= 0.0
    if np.any(dead):
        ch[dead] = fallback
        total = ch.sum(axis=1)
    cum = np.cumsum(ch, axis=1)
    target = np.asarray(draw, dtype=np.float64) * total
    idx = (cum > target[:, None]).argmax(axis=1)
    return np.clip(idx, 0, k - 1).astype(np.int64)
