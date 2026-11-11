"""Growth maps -- per-point signal -> curve -> multiplier on a growth decision.

Pure, bpy-free. The render-side masks (`growth/masks.py`) decide which organ
points *show* after the plant is built; a growth map instead changes a growth
**decision** while the plant is being built: how likely this bud is to break,
how wide this branch launches, how long this internode is, how many nodes this
lateral builds, how hard this axis corrects toward its gravitropic setpoint.

This is the Natsura "map every parameter" idea brought in on this project's
terms (clean-room -- the idea, not any code). What keeps it compatible with the
physics-first model is the same contract the environment layer already obeys
(`development._release_buds`, where light scales each bud's branch density):

- **A map only multiplies a quantity a decision already computes.** It never
  creates, places or terminates anything by itself. Branch density is still the
  species' statement of how bushy it is; a map only says *where along the plant*
  that bushiness is spent, the way shade already does.
- **No maps (or no enabled map on a target) is byte-identical.** `map_factor`
  returns exactly 1.0 without reading a single channel, and every call site in
  `development.py` is guarded so the no-map path does not even build the
  channel reader. The goldens reproduce unchanged.
- **Several maps on one target multiply.** Each map's factor is
  `1 + amount * (curve(x) - 1)`, so `amount` blends from "no effect" (0) to "the
  curve as drawn" (1), and a flat curve at 1.0 is a no-op at any amount.

Channels share their ids with the render-side mask channels so one UI list can
serve both. They are *growth-time* readings, so a few are normalised against
what growth knows at the moment the decision is made rather than against the
finished plant -- see `GROWTH_CHANNELS` for each, and the caveats there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .curves import Curve

if TYPE_CHECKING:  # pragma: no cover -- typing only; morphology imports this module
    from .morphology import MorphologyProfile


# --- targets ---------------------------------------------------------------

INTERNODE_LENGTH = "internode_length"
GRAVITROPISM = "gravitropism"

MAP_TARGETS: tuple[str, ...] = (
    INTERNODE_LENGTH,
    GRAVITROPISM,
)
"""The plant-level targets (`profile.maps`), applied in `development.py`:

- `internode_length` -- per *flowering* node, in `_add_phytomer`: the finished
  internode -- the inflorescence's spacing. A vegetative internode is its
  level's `length` target.
- `gravitropism`     -- per axis, fixed at its birth (`Axis.map_gravitropism`):
  the strength of the correction toward its gravitropic setpoint, on top of the
  species' value or the inflorescence's own override. Strength saturates at 1
  (`_apply_gravitropism` clamps), so a map can only raise it that far.
"""

# The plant-level targets that moved onto the branching levels (presets before
# schema 7, objects before branch-levels schema 5); read only by the migrations.
BRANCH_ANGLE = "branch_angle"
BRANCH_LENGTH = "branch_length"
FORK_CHANCE = "fork_chance"

PITCH = "pitch"
YAW = "yaw"
ROLL = "roll"
GROWTH = "growth"
START_LENGTH = "start_length"
LENGTH = "length"
START_WIDTH = "start_width"
WIDTH = "width"
SPIRAL_PITCH = "spiral_pitch"
SPIRAL_ROLL = "spiral_roll"
FORK_PROBABILITY = "fork_probability"
FORK_SPREAD = "fork_spread"
EMERGENCE_DELAY = "emergence_delay"
SETPOINT = "setpoint"

LEVEL_TARGETS: tuple[str, ...] = (
    PITCH, YAW, ROLL, GROWTH, START_LENGTH, LENGTH, START_WIDTH, WIDTH,
    SPIRAL_PITCH, SPIRAL_ROLL, FORK_PROBABILITY, FORK_SPREAD, EMERGENCE_DELAY,
    SETPOINT,
)
"""A branching level's targets (`BranchLevel.maps`), each a multiplier on that
level's parameter:

- per bud (a branch's birth, read at its node on the parent): `pitch` (Start
  Pitch, the angle from the parent, before `_elevation_from_axis`), `yaw`;
- per branch, fixed at its birth (read at its attachment): `roll`, `growth`
  (Growth %, `Axis.growth`), `start_length`, `start_width` (`Axis.map_width`),
  `emergence_delay` (on the stems level: per stem);
- per node of this level's axes: `length` (this internode), `width` (this
  node's radius), `spiral_pitch`, `spiral_roll`, `fork_probability`,
  `setpoint` (the gravitropic set-point angle from vertical this node's growth
  is corrected toward -- unlike Spiral Pitch, in the world's frame: below 1 a
  node aims more upright, whichever way its branch faces);
- per fork: `fork_spread`.

On the stems level the per-bud / per-branch ones read each stem at its base.
"""

FLOWER_SPACING = "flower_spacing"
PEDICEL_LENGTH = "pedicel_length"
PEDICEL_ANGLE = "pedicel_angle"
PEDICEL_BEND = "pedicel_bend"
PEDICEL_WIDTH = "pedicel_width"
FLOWER_PITCH = "flower_pitch"
INFL_BRANCH_ANGLE = "infl_branch_angle"
INFL_BRANCH_SPACING = "infl_branch_spacing"
INFL_BRANCH_NODES = "infl_branch_nodes"
INFL_BRANCH_LENGTH = "infl_branch_length"
INFL_BRANCH_WIDTH = "infl_branch_width"
INFL_BRANCH_BEND = "infl_branch_bend"
INFL_BRANCH_DROOP = "infl_branch_droop"

FLOWER_TARGETS: tuple[str, ...] = (
    FLOWER_SPACING, PEDICEL_LENGTH, PEDICEL_ANGLE, PEDICEL_BEND, PEDICEL_WIDTH,
    FLOWER_PITCH, INFL_BRANCH_ANGLE, INFL_BRANCH_SPACING, INFL_BRANCH_NODES,
    INFL_BRANCH_LENGTH, INFL_BRANCH_WIDTH, INFL_BRANCH_BEND, INFL_BRANCH_DROOP,
)
"""An inflorescence's targets (`Inflorescence.maps`, its flower component's),
each a multiplier on that parameter where growth reads it:

- per flowering node: `flower_spacing` (the internode, on top of the Along
  Flowers curve), `infl_branch_spacing` (a rachis node's);
- per flower, read at its node with the flower's own id and birth:
  `pedicel_length`, `pedicel_angle`, `pedicel_bend`, `pedicel_width`,
  `flower_pitch` -- on the whole value, its +/- included;
- per inflorescence branch: `infl_branch_angle` (per bud, at its node; a
  cyme's relay at its parent's tip), and, fixed at the branch's birth (read
  at its attachment), `infl_branch_nodes` (its size), `infl_branch_length`,
  `infl_branch_width`, `infl_branch_bend`, `infl_branch_droop`.
"""

LEGACY_TO_LEVEL: dict[str, tuple[str, int, bool]] = {
    BRANCH_ANGLE: (PITCH, 1, False),
    BRANCH_LENGTH: (GROWTH, 1, False),
    FORK_CHANCE: (FORK_PROBABILITY, 0, False),
    INTERNODE_LENGTH: (LENGTH, 0, True),
}
"""How an old plant-level map moves onto the levels: (level target, first
level it lands on, whether it also stays on the plant). Branch Angle and
Branch Length only ever acted on branches (level 1 on); Forking on every
level; Internode Length on every node -- now each level's Length for the
vegetative ones, and still the plant's for the flowering ones."""

RANDOM_SALT: dict[str, int] = {
    BRANCH_ANGLE: 0, INTERNODE_LENGTH: 1, BRANCH_LENGTH: 2, GRAVITROPISM: 3,
    FORK_CHANCE: 4,
    # a level target that replaced a plant one draws that one's `random`, so a
    # migrated map is exact
    PITCH: 0, LENGTH: 1, GROWTH: 2, FORK_PROBABILITY: 4,
    YAW: 5, ROLL: 6, START_LENGTH: 7, START_WIDTH: 8, WIDTH: 9,
    SPIRAL_PITCH: 10, SPIRAL_ROLL: 11, FORK_SPREAD: 12, EMERGENCE_DELAY: 13,
    FLOWER_SPACING: 14, PEDICEL_LENGTH: 15, PEDICEL_ANGLE: 16, PEDICEL_BEND: 17,
    PEDICEL_WIDTH: 18, FLOWER_PITCH: 19, INFL_BRANCH_ANGLE: 20, INFL_BRANCH_SPACING: 21,
    INFL_BRANCH_NODES: 22, INFL_BRANCH_LENGTH: 23, INFL_BRANCH_WIDTH: 24,
    INFL_BRANCH_BEND: 25, INFL_BRANCH_DROOP: 26, SETPOINT: 27,
}
"""Each target's word in the `random` channel's seed. Fixed numbers, never a
list position: moving a target must not re-draw every other target's noise."""


# --- channels --------------------------------------------------------------

GROWTH_CHANNELS: tuple[str, ...] = (
    "factor",
    "node_index",
    "branch_depth",
    "on_branch",
    "split_depth",
    "age",
    "random",
    "height",
    "shade",
)
"""The 0-1 signals a map can read at growth time. Ids match the render-side
mask channels (`masks.MASK_CHANNELS` / the panel's channel list). Each is read
for the *element the decision is about* -- a bud, a node, or a newly born axis
(for an axis: its own order/depth, read at its attachment on the parent):

- `factor`       -- position along the element's own axis (`Phytomer.factor`,
  index over the axis's *expected* node count). An axis reads its attachment's
  factor on the parent; a basal stem reads 0.
- `node_index`   -- index / max(expected - 1, 1), clamped. At growth time the
  expected count is the only count there is, so this equals `factor` here;
  render-side it is normalised by the *final* node count and can differ.
- `branch_depth` -- `axis.order / max(levels - 1, 1)`. Render-side normalises
  by the deepest order the finished plant reached instead, so the two agree
  only when the plant reaches its deepest level.
- `on_branch`    -- 1 when `axis.order >= 1`, else 0.
- `split_depth`  -- `max(structural_depth, 0)` over the deepest the profile can
  go: the larger of the trunk / lateral inflorescence `branching_depth` and,
  on a forking level, half its Steps (floor 1). Render-side
  normalises by the finished plant's deepest split.
- `age`          -- the element's own birth: `birth_step / total_steps`. A bud
  re-evaluated every step in `_release_buds` reads the step it was *made*, not
  the current one, so its value never drifts.
- `random`       -- a seeded uniform per element, independent per target (a
  random density map and a random angle map do not correlate).
- `height`       -- `(z - plant min z) / plant height span` against the whole
  plant at this age with every node at its mature place (measured by a grow
  without the height readers, `development._height_extent`), so a node's
  height is its place on the finished, pre-bend plant. Needs resolved
  positions: when an enabled map reads `height` or `shade`, `grow()` resolves
  every step (what a responding plant already does); a node built this step
  reads its mature place, extrapolated (`development._settled_position`).
- `shade`        -- `1 - environment.light_at(position)` when the plant is grown
  with an `Environment`, else 0. The light field is rebuilt each step for it.

Plus the plant's own custom attributes and classes (`attr:<name>` /
`class:<name>`, `growth/signals.py`) when growth is handed its `Signals`: read
at the element's position with its own channels; a Distance or Noise one needs
positions, like `height`. A name the plant does not have reads identity.
"""

POSITIONAL_CHANNELS = frozenset({"height", "shade"})
"""Channels that need world positions -- reading one is what makes `grow()`
resolve (and, for shade, rebuild the light field) every step."""


@dataclass
class ParamMap:
    """One signal -> curve -> multiplier on one growth target.

    `curve` maps the channel value (x in [0, 1]) to a raw multiplier; `amount`
    blends it in: `factor = 1 + amount * (curve(x) - 1)`. The default curve is
    flat at 1.0, so a freshly added map is a no-op until its curve is drawn --
    the same "a fresh filter removes nothing" rule the render masks follow.
    """

    target: str
    channel: str
    curve: Curve = field(default_factory=lambda: Curve.constant(1.0))
    amount: float = 1.0
    enabled: bool = True
    seed: int = 0
    """The map's own seed, mixed into its `random` draw (0: the draw every map
    on this target shares, as always)."""


def enabled(stack, target: str) -> list[ParamMap]:
    """The enabled maps on `target` in a list of maps (a profile's or a
    level's), in stack order."""
    if not stack:
        return []
    return [m for m in stack if m.enabled and m.target == target]


def stack_bound(stack, target: str, samples: int = 257) -> float:
    """The largest factor `stack_factor` can give on `target` over any channel
    values -- each map's curve sampled across 0..1 (1.0 without a map). An
    upper bound: what the plant's clock is sized from where a mapped value can
    make growth take longer (`BranchLevel.emergence_delay`)."""
    bound = 1.0
    for m in enabled(stack, target):
        peak = max(abs(1.0 + float(m.amount) * (float(m.curve(i / (samples - 1))) - 1.0))
                   for i in range(samples))
        bound *= peak
    return bound


def maps_for(profile: MorphologyProfile, target: str) -> list[ParamMap]:
    """The enabled plant-level maps on `target`, in stack order. Empty for any
    profile with no maps -- the common case, and a single attribute read."""
    return enabled(getattr(profile, "maps", None), target)


def has_map(profile: MorphologyProfile, target: str) -> bool:
    """Whether any enabled plant-level map scales `target`. The guard every call
    site uses so the no-map path does no work at all."""
    stack = getattr(profile, "maps", None)
    if not stack:
        return False
    return any(m.enabled and m.target == target for m in stack)


def level_has_map(level, target: str) -> bool:
    """Whether any enabled map on this branching level scales `target`."""
    stack = getattr(level, "maps", None) if level is not None else None
    if not stack:
        return False
    return any(m.enabled and m.target == target for m in stack)


def inflorescences(profile: MorphologyProfile) -> list:
    """Every inflorescence a profile grows (primary, lateral, the extra ones)."""
    out = [profile.inflorescence]
    if getattr(profile, "lateral_inflorescence", None) is not None:
        out.append(profile.lateral_inflorescence)
    out.extend(getattr(profile, "extra_inflorescences", None) or ())
    return out


def channel_keys(profile: MorphologyProfile):
    """Every channel an enabled map (plant, level or inflorescence) or a level
    mask reads."""
    for m in getattr(profile, "maps", None) or ():
        if m.enabled and m.target in MAP_TARGETS:
            yield m.channel
    for inf in inflorescences(profile):
        for m in getattr(inf, "maps", None) or ():
            if m.enabled and m.target in FLOWER_TARGETS:
                yield m.channel
    for level in profile.levels:
        for m in getattr(level, "maps", None) or ():
            if m.enabled and m.target in LEVEL_TARGETS:
                yield m.channel
        for e in getattr(level, "masks", None) or ():
            if not e.mute:
                yield e.channel


def reads_channels(profile: MorphologyProfile, channels) -> bool:
    """Whether any enabled map or level mask reads one of `channels`."""
    return any(key in channels for key in channel_keys(profile))


def _read(channels, name: str) -> float | None:
    if callable(channels):
        return channels(name)
    return channels.get(name)


def map_factor(profile: MorphologyProfile, target: str,
               channels: Mapping[str, float] | Callable[[str], float | None]) -> float:
    """The product of every enabled map on `target`, clamped at 0.

    `channels` is a plain dict, or a callable `name -> value` (None = channel
    unavailable). The callable form is what lets development compute only the
    channels a map actually reads -- `height` needs a position lookup and
    `random` a seeded draw, neither worth paying for a map that reads `factor`.

    Returns exactly 1.0 -- without reading `channels` at all -- when nothing
    targets `target`. A map whose channel is unknown or unavailable contributes
    1.0 (identity), never 0: a signal that cannot be read must not silently
    switch a decision off. Channel values are clamped to [0, 1] before the curve
    (the curve itself clamps too; this keeps the contract explicit).
    """
    return stack_factor(getattr(profile, "maps", None), target, channels)


def stack_factor(stack, target: str,
                 channels: Mapping[str, float] | Callable[[str], float | None]) -> float:
    """`map_factor` over any list of maps -- a branching level's."""
    active = enabled(stack, target)
    if not active:
        return 1.0
    factor = 1.0
    for m in active:
        seed = int(getattr(m, "seed", 0))
        if seed and m.channel == "random" and hasattr(channels, "seeded_random"):
            x = channels.seeded_random(seed)
        else:
            x = _read(channels, m.channel)
        if x is None:
            continue
        x = min(1.0, max(0.0, float(x)))
        factor *= 1.0 + float(m.amount) * (float(m.curve(x)) - 1.0)
    return max(0.0, factor)
