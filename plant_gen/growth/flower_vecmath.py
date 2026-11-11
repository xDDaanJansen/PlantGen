"""Shared mathutils-free vector/rotation helpers for the pure skeleton modules."""

from __future__ import annotations

import math

Vec = tuple[float, float, float]


def length(v: Vec) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v: Vec) -> Vec:
    n = length(v)
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else (0.0, 0.0, 1.0)


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v: Vec, s: float) -> Vec:
    return (v[0] * s, v[1] * s, v[2] * s)


def rotate_axis(v: Vec, axis: Vec, angle: float) -> Vec:
    """Rodrigues rotation of v around a unit axis by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    cr = cross(axis, v)
    d = dot(axis, v) * (1.0 - c)
    return (v[0] * c + cr[0] * s + axis[0] * d,
            v[1] * c + cr[1] * s + axis[1] * d,
            v[2] * c + cr[2] * s + axis[2] * d)


def ring_azimuths(n: int, start: float = 0.0, sweep: float = 2.0 * math.pi) -> list[float]:
    """`n` evenly-spaced azimuth angles around a ring — the placement math shared by
    stem-level whorled phyllotaxis (`skeleton/phyllotaxis.py`) and flower-internal
    petal whorls (`skeleton/flower.py`)."""
    return [start + i * (sweep / n) for i in range(n)]


def rotate_toward(d: Vec, target: Vec, angle: float, fallback_axis: Vec) -> Vec:
    """Rotate `d` at most `angle` radians around the axis that turns it toward
    `target`, clamped so it never rotates past `target`. `fallback_axis` is used
    when `d` is exactly parallel/antiparallel to `target` (axis undefined).

    Shared by `skeleton/stem.py` (bends the tip toward world down/up -- Gravity)
    and `skeleton/flower.py` (bends a petal toward world down/up -- Petal Gravity):
    both are the same "rotate toward a fixed world target, clamp at arrival" model,
    just applied to different skeletons."""
    remaining = math.acos(max(-1.0, min(1.0, dot(d, target))))
    if remaining < 1e-9:
        return d
    axis = cross(d, target)
    if length(axis) < 1e-9:
        axis = fallback_axis
    else:
        axis = normalize(axis)
    return normalize(rotate_axis(d, axis, min(angle, remaining)))
