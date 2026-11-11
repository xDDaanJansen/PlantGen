"""A plant's custom attributes and classes -- pure, bpy-free.

The derived channels any mask or map can read by name: a **custom attribute**
(`attr:<name>`) is a new per-point dimension -- closeness to a point, seeded 3D
noise, or two channels combined -- and a **class** (`class:<name>`) is a named
mask stack, 1 where it keeps a point.

One evaluator for both sides of the plant: the bridge reads them over the
finished plant's points (render-time masks and maps), and growth reads them per
bud / node / axis while it grows. So a
Noise attribute has one pattern, not one per side, and the bridge only
translates its panel lists into these dataclasses.

Lazily resolved and memoised: only the keys something asks for are computed.
A reference cycle (A reads B reads A) reads as identity (1.0) where it closes,
rather than recursing.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from . import masks

_U64 = 0xFFFFFFFFFFFFFFFF


@dataclass
class CustomAttribute:
    """One custom attribute. `kind` is DISTANCE / NOISE / COMBINE.

    `point` is the Distance target in the plant's own space (the bridge
    converts the object's position); None reads 0. `channel_a` / `channel_b`
    are channel keys -- plain ids, or `attr:` / `class:` references."""

    name: str
    kind: str = "DISTANCE"
    point: tuple[float, float, float] | None = None
    radius: float = 0.3
    noise_scale: float = 10.0
    seed: int = 0
    op: str = "MULTIPLY"
    channel_a: str = "height"
    channel_b: str = "shade"


@dataclass
class ClassDef:
    """A named classification: a mask stack, 1 where it keeps a point."""

    name: str
    entries: list[masks.MaskEntry] = field(default_factory=list)


@dataclass
class Signals:
    """Every custom attribute and class of one plant, plus the plant seed the
    class stacks and Combine's random inputs draw from."""

    attributes: list[CustomAttribute] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    seed: int = 0

    def is_empty(self) -> bool:
        return not self.attributes and not self.classes

    def defines(self, key: str) -> bool:
        """Whether `key` names one of these attributes / classes."""
        if key.startswith(masks.CUSTOM_PREFIX):
            name = key[len(masks.CUSTOM_PREFIX):]
            return any(a.name == name for a in self.attributes)
        if key.startswith(masks.CLASS_PREFIX):
            name = key[len(masks.CLASS_PREFIX):]
            return any(c.name == name for c in self.classes)
        return False


def is_derived(key: str) -> bool:
    """A custom-attribute or class channel key."""
    return key.startswith((masks.CUSTOM_PREFIX, masks.CLASS_PREFIX))


# --- noise -----------------------------------------------------------------
#
# Gradient ("improved") noise over a fixed permutation, vectorised. Its own
# implementation so growth -- which has no bpy -- and the render side read the
# same field.

_PERM = np.random.default_rng(0x5EED).permutation(256).astype(np.int64)
_P = np.concatenate([_PERM, _PERM])


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _grad(h, x, y, z):
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


def noise3(points) -> np.ndarray:
    """Gradient noise at (N, 3) points, roughly -1..1, 0 on every lattice point."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cell = np.floor(p)
    f = p - cell
    i = cell.astype(np.int64) & 255
    x, y, z = f[:, 0], f[:, 1], f[:, 2]
    u, v, w = _fade(x), _fade(y), _fade(z)
    X, Y, Z = i[:, 0], i[:, 1], i[:, 2]
    A = _P[X] + Y
    AA, AB = _P[A] + Z, _P[A + 1] + Z
    B = _P[X + 1] + Y
    BA, BB = _P[B] + Z, _P[B + 1] + Z

    def lerp(t, a, b):
        return a + t * (b - a)

    return lerp(w,
                lerp(v, lerp(u, _grad(_P[AA], x, y, z), _grad(_P[BA], x - 1, y, z)),
                     lerp(u, _grad(_P[AB], x, y - 1, z), _grad(_P[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, _grad(_P[AA + 1], x, y, z - 1),
                             _grad(_P[BA + 1], x - 1, y, z - 1)),
                     lerp(u, _grad(_P[AB + 1], x, y - 1, z - 1),
                          _grad(_P[BB + 1], x - 1, y - 1, z - 1))))


def noise01(points, scale: float, seed: int) -> np.ndarray:
    """A Noise attribute's value: the field at `points * scale`, offset by the
    seed, mapped to 0..1."""
    off = np.array([seed * 17.31, seed * 5.77, seed * 11.13])
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3) * float(scale) + off
    return np.clip((noise3(p) + 1.0) * 0.5, 0.0, 1.0)


# --- combine ---------------------------------------------------------------

def combine(op: str, a, b):
    if op == "MULTIPLY":
        return a * b
    if op == "ADD":
        return np.minimum(a + b, 1.0)
    if op == "SUBTRACT":
        return np.maximum(a - b, 0.0)
    if op == "DIFFERENCE":
        return np.abs(a - b)
    if op == "MIN":
        return np.minimum(a, b)
    if op == "MAX":
        return np.maximum(a, b)
    return (a + b) * 0.5          # AVERAGE


# --- the evaluator ---------------------------------------------------------

class Evaluator:
    """Derived channels over one set of points, resolved on demand.

    `base` gives the plain channels: a mapping key -> array, or a callable
    `key -> array | None` (None: unavailable, read as 0 for an attribute's
    input and as identity by a class's mask). `positions` is an (N, 3) array or
    a callable returning one -- only Distance / Noise call it. `ids` seed the
    per-point random draws."""

    def __init__(self, signals: Signals, ids,
                 base: Mapping | Callable[[str], np.ndarray | None],
                 positions):
        self.signals = signals
        self.ids = np.asarray(ids)
        self.n = int(self.ids.shape[0])
        self._base = base
        self._positions = positions
        self._pos = None
        self._attrs = {a.name: a for a in signals.attributes}
        self._classes = {c.name: c for c in signals.classes}
        self._out: dict[str, np.ndarray] = {}
        self._visiting: set[str] = set()

    def _base_values(self, key: str):
        if callable(self._base):
            v = self._base(key)
        else:
            v = self._base.get(key)
        return None if v is None else np.asarray(v, dtype=np.float64)

    def _points(self) -> np.ndarray:
        if self._pos is None:
            p = self._positions() if callable(self._positions) else self._positions
            self._pos = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        return self._pos

    def _input(self, key: str, salt: int) -> np.ndarray:
        """One input of a Combine: a derived channel, `random`, or a plain one
        (0 where unavailable)."""
        if is_derived(key):
            return self.get(key)
        if key == masks.RANDOM_CHANNEL:
            return masks._hash01(self.ids, salt & _U64)
        v = self._base_values(key)
        return np.zeros(self.n) if v is None else v

    def _attribute(self, a: CustomAttribute) -> np.ndarray:
        if a.kind == "DISTANCE":
            if a.point is None:
                return np.zeros(self.n)
            d = np.linalg.norm(self._points() - np.asarray(a.point, dtype=np.float64), axis=1)
            return np.clip(1.0 - d / max(float(a.radius), 1e-6), 0.0, 1.0)
        if a.kind == "NOISE":
            return noise01(self._points(), a.noise_scale, a.seed)
        salt = self.signals.seed * 2654435761 + zlib.crc32(a.name.encode("utf-8"))
        va = self._input(a.channel_a, salt)
        vb = self._input(a.channel_b, salt * 3 + 1)
        return np.clip(combine(a.op, va, vb), 0.0, 1.0)

    def _class(self, c: ClassDef) -> np.ndarray:
        stack = masks.MaskStack(entries=list(c.entries), keep=1.0, seed=int(self.signals.seed))
        if stack.is_empty():
            return np.ones(self.n)
        needed: dict[str, np.ndarray] = {}
        for e in c.entries:
            if e.channel == masks.RANDOM_CHANNEL:
                continue            # synthesised by the evaluator itself
            if is_derived(e.channel):
                needed[e.channel] = self.get(e.channel)
            else:
                v = self._base_values(e.channel)
                if v is not None:
                    needed[e.channel] = v
                raw = masks.ORDINAL_CHANNELS.get(e.channel)
                if raw and e.modulo >= 2:
                    rv = self._base_values(raw)
                    if rv is not None:
                        needed[raw] = rv
        return (masks.evaluate(stack, needed, self.ids)[0] > 0.5).astype(np.float64)

    def get(self, key: str) -> np.ndarray:
        """The values of `attr:<name>` / `class:<name>`; identity (1.0) for a
        name the plant does not have, or where a reference cycle closes."""
        if key in self._out:
            return self._out[key]
        if key in self._visiting:
            return np.ones(self.n)
        self._visiting.add(key)
        if key.startswith(masks.CUSTOM_PREFIX):
            a = self._attrs.get(key[len(masks.CUSTOM_PREFIX):])
            v = self._attribute(a) if a is not None else np.ones(self.n)
        else:
            c = self._classes.get(key[len(masks.CLASS_PREFIX):])
            v = self._class(c) if c is not None else np.ones(self.n)
        self._visiting.discard(key)
        self._out[key] = v
        return v

    def all(self) -> dict[str, np.ndarray]:
        """Every attribute and class, keyed `attr:` / `class:`."""
        for name in self._attrs:
            self.get(masks.CUSTOM_PREFIX + name)
        for name in self._classes:
            self.get(masks.CLASS_PREFIX + name)
        return dict(self._out)


def restrict(signals: Signals | None, keys) -> Signals | None:
    """Only the attributes and classes `keys` reach (directly, or through a
    Combine's inputs or a class's entries) -- None when that is nothing. What
    growth is handed and hashed by, so editing a class only a component's
    masks read does not re-grow the plant."""
    if signals is None:
        return None
    attrs = {a.name: a for a in signals.attributes}
    classes = {c.name: c for c in signals.classes}
    need_attrs: set[str] = set()
    need_classes: set[str] = set()
    todo = [k for k in keys if is_derived(k)]
    while todo:
        key = todo.pop()
        if key.startswith(masks.CUSTOM_PREFIX):
            name = key[len(masks.CUSTOM_PREFIX):]
            if name in need_attrs or name not in attrs:
                continue
            need_attrs.add(name)
            a = attrs[name]
            if a.kind == "COMBINE":
                todo += [k for k in (a.channel_a, a.channel_b) if is_derived(k)]
        else:
            name = key[len(masks.CLASS_PREFIX):]
            if name in need_classes or name not in classes:
                continue
            need_classes.add(name)
            todo += [e.channel for e in classes[name].entries if is_derived(e.channel)]
    if not need_attrs and not need_classes:
        return None
    return Signals(attributes=[a for a in signals.attributes if a.name in need_attrs],
                   classes=[c for c in signals.classes if c.name in need_classes],
                   seed=signals.seed)


def reads_position(signals: Signals | None, key: str, wanted=("height", "shade"),
                   _seen=None) -> bool:
    """Whether reading `key` needs world positions: one of the `wanted` plain
    channels (height, shade), or a Distance / Noise attribute -- directly or
    through a Combine or a class. With `wanted=("shade",)` it answers whether
    the key needs the light field."""
    if key in wanted:
        return True
    if signals is None or not is_derived(key):
        return False
    seen = set() if _seen is None else _seen
    if key in seen:
        return False
    seen.add(key)
    positional = "height" in wanted       # Distance / Noise need positions, not light
    if key.startswith(masks.CUSTOM_PREFIX):
        name = key[len(masks.CUSTOM_PREFIX):]
        for a in signals.attributes:
            if a.name == name:
                if a.kind in ("DISTANCE", "NOISE"):
                    return positional
                if a.kind != "COMBINE":
                    return False
                return (reads_position(signals, a.channel_a, wanted, seen)
                        or reads_position(signals, a.channel_b, wanted, seen))
        return False
    name = key[len(masks.CLASS_PREFIX):]
    for c in signals.classes:
        if c.name == name:
            return any(reads_position(signals, e.channel, wanted, seen)
                       for e in c.entries if not e.mute)
    return False
