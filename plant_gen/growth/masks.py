"""Point mask / map system -- pure, bpy-free. 

Per component, a stack of "filters" decides which of that component's organ points
render. Each filter reads one normalised 0-1 point attribute (a "channel"), keeps
the BAND `lo` <= value <= `hi` (1 inside, 0 outside), optionally inverts it (keep
the outside), and folds into a running per-point value by a combine op (the same
replace/add/subtract/multiply/divide/max/min set Natsura's Map node exposes). The
folded value is a per-point KEEP value in
[0, 1]: a point survives when a seeded, per-point, order-independent uniform draw
falls under it -- so a band is a deterministic keep/drop, and `prune`
(`MaskStack.keep`) is a final keep-fraction multiply that thins the survivors at
random. Min and max both bound the kept range: set min 0.4, max 0.5 and only
0.4..0.5 survives (the old one-sided ramp needed two entries for a band).

Decisions this implements:
- Approach A + D, bpy-free half. The stack lives on `MorphologyProfile` and
  serialises into presets (P3); the panel edits it (P4); the bridge evaluates it
  AFTER `build()` and bakes one 0-1 attribute per component that GN gates on (P5).
- **Render-only**: evaluated after `build()`/physics, so a mask never changes the
  load the pipe model / gravity already sized the plant for.
- **No-op by default**: an empty stack yields `mask == 1` everywhere -> all
  survive -> byte-identical. The bridge skips baking/gating an empty stack.
- **Deterministic**: the survival draw is a splitmix64 hash of `(organ_id, seed)`,
  independent of array order -- the repo's seed-by-id discipline.
- **Continuous by construction**: `evaluate` returns the continuous `mask`
  alongside `survive`, so a later milestone can drive instance scale/density from
  the same value (deferred; v1 spends it only on survival).

Note vs. the plan: prune is a `MaskStack.keep` factor rather than a "random
pseudo-channel", because the probabilistic survival draw already turns any
sub-1 mask into seeded thinning -- so a keep-fraction is all prune needs, and
the RANDOM channel is kept only as a general noise source for maps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .curves import Curve

RANDOM_CHANNEL = "random"
# Channels that are a 0/1 flag rather than a continuous 0-1 range. A band is the
# wrong control for these -- min/max just reads as "keep both / keep one". So they
# ignore lo/hi and always select the "1" side (invert flips to "0"), and adding one
# filters immediately instead of being a no-op at [0, 1].
BINARY_CHANNELS = frozenset({"on_branch", "apex", "inflorescence"})

# Index channels that carry a RAW integer ordinal a `modulo` can act on, mapped to
# the key their raw values arrive under in `channels`. (The band still reads the
# normalised channel; modulo reads the raw ordinal.)
ORDINAL_CHANNELS = {"node_index": "node_ordinal", "branch_index": "branch_ordinal"}


def effective_window(channel: str, lo: float, hi: float) -> tuple[float, float]:
    """The window actually applied: a binary channel is pinned to select its "1"
    side (invert then keeps "0"); everything else uses the entry's own lo/hi."""
    if channel in BINARY_CHANNELS or channel.startswith(CLASS_PREFIX):
        return 0.5, 1.0
    return lo, hi


CLASS_PREFIX = "class:"
"""A named class's membership arrives as a channel keyed `class:<name>` (the
bridge evaluates the class's own stack into it). It is a 0/1 membership, so it
reads as binary -- keep the class, or invert to keep everything outside it."""

CUSTOM_PREFIX = "attr:"
"""A plant's custom attribute arrives as a channel keyed `attr:<name>`, a plain
0-1 value windowed like any continuous channel."""
# How a filter folds onto the running keep value. The same set Natsura's Map node
# exposes (clean-room -- names/behaviour only, no ported code), clamped to [0, 1]:
# replace/add/subtract/multiply/divide/max/min. For the hard 0/1 keep-bands these
# reduce to the obvious set ops (multiply == min == AND, add == max == OR, subtract
# removes), but the full set is kept so they read the same as Natsura.
COMBINE_OPS = ("replace", "add", "subtract", "multiply", "divide", "max", "min")

_UINT64 = 0xFFFFFFFFFFFFFFFF
_TWO53 = float(1 << 53)


@dataclass(frozen=True)
class MaskChannel:
    """A mappable 0-1 per-point scalar.

    `source`:
      'point'  -- read from the baked point-attribute dict the bridge supplies.
      'random' -- a seeded per-point uniform the evaluator synthesises (noise).

    The concrete set is finalised by the P0 attribute audit; this seed list is
    the *wanted* channels and is trimmed/extended once the audit reports which the
    bridge already emits and how each is normalised.
    """

    key: str
    label: str
    source: str = "point"


MASK_CHANNELS: dict[str, MaskChannel] = {
    c.key: c
    for c in (
        MaskChannel("height", "Height"),
        MaskChannel("factor", "Factor"),
        MaskChannel("age", "Age"),
        MaskChannel("split_depth", "Split Depth"),
        MaskChannel("shade", "Shade"),
        MaskChannel("bloom", "Bloom"),
        MaskChannel(RANDOM_CHANNEL, "Random", source="random"),
    )
}


@dataclass
class MaskEntry:
    """One filter in a stack. Reads `channel`, keeps the BAND `lo` <= value <= `hi`
    (1 inside, 0 outside), applies `curve` then `invert` (keep the outside), and
    folds onto the accumulator by `op` (the first entry is the base, its op
    ignored). A full band `[0, 1]` keeps everything -- a fresh filter is a no-op
    until you narrow it."""

    channel: str
    lo: float = 0.0
    hi: float = 1.0
    invert: bool = False
    op: str = "multiply"
    curve: Curve | None = None
    mute: bool = False
    modulo: int = 0
    """For an index channel (node/branch index): keep only points whose RAW ordinal
    is a multiple of `modulo` -- 2 = every other, 3 = every third. 0 is off. Combines
    with the band (AND) before `invert`, so 'every other node, upper half' is one
    entry. Needs the channel's raw ordinal in `channels` (see `ORDINAL_CHANNELS`)."""
    """Temporarily skip this map -- evaluated as if it were not in the stack, so
    its salt index (and therefore every other map's seeded draw) is unchanged."""


@dataclass
class MaskStack:
    """A component's ordered maps plus a prune fraction. Empty and `keep == 1`
    is a true no-op."""

    entries: list[MaskEntry] = field(default_factory=list)
    keep: float = 1.0  # prune: final keep-fraction (seeded thin)
    seed: int = 0

    def is_empty(self) -> bool:
        return not self.entries and self.keep >= 1.0


# -- helpers ----------------------------------------------------------------

def _hash01(ids: np.ndarray, salt: int) -> np.ndarray:
    """splitmix64 over `id ^ salt` -> uniform [0, 1), deterministic per id and
    independent of array order (so shuffling the points cannot change who
    survives). Matches the repo's `default_rng([...])`-per-element discipline,
    vectorised."""
    x = ids.astype(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x ^ np.uint64(salt & _UINT64)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / _TWO53


def _random_salt(seed: int, index: int) -> int:
    return (seed * 7919 + (index + 1) * 104729) & _UINT64


def _remap(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The keep-BAND: 1.0 where `lo` <= value <= `hi`, 0.0 outside. Order-
    independent in `lo`/`hi`, so min and max both bound the kept range -- a single
    filter selects a slice (set min 0.4, max 0.5 -> only 0.4..0.5 survives). The
    full band `[0, 1]` keeps everything."""
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return ((values >= a) & (values <= b)).astype(np.float64)


def _apply_curve(t: np.ndarray, curve: Curve) -> np.ndarray:
    """Vectorised `Curve.__call__`: linear interpolation over the curve's own
    uniformly-spaced sample table, clamped -- identical to the scalar path."""
    samples = np.asarray(curve.samples, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(samples))
    return np.interp(np.clip(t, 0.0, 1.0), xs, samples)


def _fold(acc: np.ndarray, w: np.ndarray, op: str) -> np.ndarray:
    if op == "replace":
        return w                                       # override with this filter
    if op == "add":
        return np.clip(acc + w, 0.0, 1.0)              # union (OR for 0/1)
    if op == "subtract":
        return np.clip(acc - w, 0.0, 1.0)              # drop this filter's points
    if op == "multiply":
        return acc * w                                 # intersect (AND for 0/1)
    if op == "divide":
        return np.clip(acc / np.where(w > 1e-6, w, 1.0), 0.0, 1.0)  # w==0 leaves acc
    if op == "max":
        return np.maximum(acc, w)
    if op == "min":
        return np.minimum(acc, w)
    raise ValueError(f"unknown combine op: {op!r}")


# -- the evaluator ----------------------------------------------------------

def evaluate(
    stack: MaskStack | None,
    channels: Mapping[str, Sequence[float]],
    organ_id: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `stack` over `organ_id` points, reading channel arrays from
    `channels`. Returns `(mask, survive)`:

    - `mask`   -- continuous keep-probability per point, in [0, 1].
    - `survive` -- boolean per point (`_hash01(organ_id, seed) < mask`).

    An empty/None stack (and `keep >= 1`) returns all-ones / all-True -- the
    no-op. A channel an entry names but `channels` does not supply is treated as
    identity (removes nothing), never as zero.
    """
    ids = np.asarray(organ_id)
    n = int(ids.shape[0]) if ids.ndim else 0
    if stack is None or (not stack.entries and stack.keep >= 1.0):
        return np.ones(n), np.ones(n, dtype=bool)

    acc: np.ndarray | None = None
    for i, entry in enumerate(stack.entries):
        if entry.mute:
            continue  # skipped, but `i` stays put so no other map's salt shifts
        w = entry_window(stack, i, channels, ids)
        acc = w if acc is None else _fold(acc, w, entry.op)

    if acc is None:
        acc = np.ones(n)
    mask = np.clip(acc * float(stack.keep), 0.0, 1.0)
    survive = _hash01(ids, stack.seed & _UINT64) < mask
    return mask, survive


def entry_window(stack: MaskStack, i: int, channels: Mapping[str, Sequence[float]],
                 ids) -> np.ndarray:
    """Entry `i`'s own keep window over the points, before it folds onto the
    stack: its channel banded by min / max (and Modulo), through its curve, with
    Invert. What a mask preview shows for one entry, and what `evaluate` folds."""
    ids = np.asarray(ids)
    n = int(ids.shape[0]) if ids.ndim else 0
    entry = stack.entries[i]
    channel = MASK_CHANNELS.get(entry.channel)
    if channel is not None and channel.source == "random":
        values = _hash01(ids, _random_salt(stack.seed, i))
    elif entry.channel in channels:
        values = np.asarray(channels[entry.channel], dtype=np.float64)
    else:
        values = np.ones(n)  # unknown/absent channel -> identity
    lo, hi = effective_window(entry.channel, entry.lo, entry.hi)
    w = _remap(values, lo, hi)
    if entry.modulo >= 2:
        raw_key = ORDINAL_CHANNELS.get(entry.channel)
        raw = channels.get(raw_key) if raw_key else None
        if raw is not None:
            keep = (np.asarray(raw).astype(np.int64) % entry.modulo == 0)
            w = w * keep.astype(np.float64)
    if entry.curve is not None:
        w = _apply_curve(w, entry.curve)
    if entry.invert:
        w = 1.0 - w
    return w
