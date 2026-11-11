"""Parameter mapping -- the registry of what every parameter's maps are, and
the per-plant map. Pure, bpy-free.

Every numeric parameter has one map icon. Behind it, a parameter's maps are
evaluated where the parameter is used:

- a **Random Range on a parameter that had a +/-** is that +/- value itself
  (`VARIANCE`): today's per-element draw, so every plant stays exactly as it
  was;
- a parameter with a **per-element evaluator** keeps its maps there: a level
  map (`LEVEL_HOOKS`, growth/maps.py), a plant growth map (`PLANT_HOOKS`), a
  component's instance map (`INSTANCE_HOOKS`, render);
- **any other** parameter maps once per plant (`apply`): a seeded draw per
  plant through the map's curve, at the panel -> model boundary.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .curves import Curve

OWNER_PLANT = "PLANT"
OWNER_LEVEL = "LEVEL"
OWNER_COMPONENT = "COMPONENT"

# --- per-element evaluators ----------------------------------------------------

LEVEL_HOOKS: dict[str, str] = {
    # level property -> level map target (growth/maps.py LEVEL_TARGETS)
    "start_pitch": "pitch",
    "start_yaw": "yaw",
    "start_roll": "roll",
    "growth": "growth",
    "start_length_scale": "start_length",
    "length_scale_per_step": "length",
    "start_width_scale": "start_width",
    "width_scale_per_step": "width",
    "spiral_pitch": "spiral_pitch",
    "spiral_roll": "spiral_roll",
    "fork_probability": "fork_probability",
    "fork_spread": "fork_spread",
    "emergence_delay": "emergence_delay",
    "gravitropic_setpoint": "setpoint",
}
PLANT_HOOKS: dict[str, str] = {
    # plant property -> plant growth map target (growth/maps.py MAP_TARGETS)
    "internode_length": "internode_length",
    "gravitropism": "gravitropism",
}
FLOWER_HOOKS: dict[str, str] = {
    # flower component property -> inflorescence map target (growth/maps.py
    # FLOWER_TARGETS), read as its inflorescence grows
    "inflorescence_internode_scale": "flower_spacing",
    "pedicel_length": "pedicel_length",
    "pedicel_angle": "pedicel_angle",
    "pedicel_bend": "pedicel_bend",
    "pedicel_width": "pedicel_width",
    "flower_pitch": "flower_pitch",
    "inflorescence_branch_angle": "infl_branch_angle",
    "inflorescence_branch_spacing": "infl_branch_spacing",
    "inflorescence_branch_nodes": "infl_branch_nodes",
    "inflorescence_branch_length": "infl_branch_length",
    "inflorescence_branch_width": "infl_branch_width",
    "inflorescence_branch_bend": "infl_branch_bend",
    "inflorescence_branch_droop": "infl_branch_droop",
}
INSTANCE_HOOKS: dict[str, str] = {
    # component property (or row of its own) -> instance map target
    "size": "SIZE",
    "instance_pitch": "PITCH",
    "instance_yaw": "YAW",
    "instance_roll": "ROLL",
    # a placed point's Fold / Parent Roll (`instancing._reorient_placed`)
    "gen_fold": "FOLD",
    "gen_roll_parent": "ROLL_PARENT",
    # a leaf / bract's petiole (`petioles._add_petioles`) and young shape
    # (`instancing._emerging_shapes`)
    "petiole_length": "PETIOLE_LENGTH",
    "petiole_width": "PETIOLE_WIDTH",
    "petiole_bend": "PETIOLE_BEND",
    "petiole_droop": "PETIOLE_DROOP",
    "emerging_curl": "EMERGING_CURL",
    "emerging_midrib_fold": "EMERGING_FOLD",
    # a flower's place on the bloom timeline (`instancing._apply_instance_maps`)
    "bloom_timing": "BLOOM",
}
ROTATION_TARGETS = frozenset({"PITCH", "YAW", "ROLL", "FOLD", "ROLL_PARENT", "PETIOLE_BEND",
                              "PETIOLE_DROOP"})
"""Instance-map targets that add an angle (+/- Angle round the curve's middle)
rather than multiply."""
OFFSET_TARGETS = frozenset({"BLOOM"})
"""Instance-map targets that add (curve - 1) x Amount -- a shift, read off the
same curve a multiplier is (flat at 1 = none, Random Range +/- Range) -- rather
than multiply: a flower's place on the bloom timeline, which is 0 for the
first to open and so cannot be scaled."""
ORIENTATION_TARGETS = frozenset({"SIZE", "PITCH", "YAW", "ROLL", "BLOOM"})
"""The instance-map targets `instancing._apply_instance_maps` applies; the rest are
read where their parameter is."""

# --- the +/- values, by the parameter they vary -----------------------------------

VARIANCE_LEVEL: dict[str, str] = {
    "start_pitch": "start_pitch_variation",
    "start_yaw": "start_yaw_variation",
    "start_roll": "start_roll_variation",
    "length_scale_per_step": "length_variation",
    "width_scale_per_step": "width_variation",
    "slide": "slide_variation",
}
VARIANCE_PLANT: dict[str, str] = {
    "basal_spread": "basal_spread_variation",
    "stem_stiffness": "stiffness_variation",
}
VARIANCE_FLOWER: dict[str, str] = {
    # a flower component's (its inflorescence's) -- also the plant's legacy copies
    "inflorescence_branch_count": "inflorescence_branch_count_variation",
    "inflorescence_branch_angle": "inflorescence_branch_angle_variation",
    "inflorescence_branch_nodes": "inflorescence_branch_nodes_variation",
    "inflorescence_branch_spacing": "inflorescence_branch_spacing_variation",
    "inflorescence_branch_length": "inflorescence_branch_length_variation",
    "inflorescence_branch_width": "inflorescence_branch_width_variation",
    "inflorescence_branch_bend": "inflorescence_branch_bend_variation",
    "inflorescence_branch_droop": "inflorescence_branch_droop_variation",
    "flower_count": "flower_count_variation",
    "inflorescence_internode_scale": "flower_spacing_variation",
    "head_dome": "head_dome_variation",
    "flower_spread": "flower_spread_variation",
    "flower_pitch": "flower_pitch_variation",
    "pedicel_width": "pedicel_width_variation",
    "pedicel_bend": "pedicel_bend_variation",
    "pedicel_length": "pedicel_variation",
    "pedicel_angle": "pedicel_angle_variation",
    "catkin_length": "catkin_length_variation",
    "catkin_count": "catkin_count_variation",
    "catkin_stalk": "catkin_stalk_variation",
    "catkin_hang": "catkin_droop_variation",
    "size": "flower_size_variation",
}
VARIANCE_ORGAN: dict[str, str] = {
    # a leaf / bract / bud / fruit component's
    "size": "gen_size_variance",
    "gen_pitch": "gen_pitch_variance",
    "gen_fold": "gen_fold_variance",
    "gen_roll_self": "gen_roll_self_variance",
    "gen_roll_parent": "gen_roll_parent_variance",
}
VARIANCE_LEAF: dict[str, str] = {
    **VARIANCE_ORGAN,
    "petiole_length": "petiole_length_variation",
    "petiole_bend": "petiole_bend_variation",
    "petiole_droop": "petiole_droop_variation",
    "petiole_angle": "petiole_angle_variation",
}

# --- the profile curves, by the parameter they multiply -----------------------------
# A multiplier curve read along a fixed axis (along the flowering region, along
# a pedicel, an instance's age...) is a map with a fixed driver: its "Curve"
# row in the Mapping list IS that curve, stored where it always was (exact by
# construction, like a +/- value's Random Range). At its neutral (flat at 1;
# `CURVE_NEUTRAL` for one that is not a multiplier) = not mapped.
# (curve key, (short name, driver caption)). Not here, kept as curves on their
# own controls: the shapes with no "does nothing" setting -- Head Dome,
# Flowering Onset, Age Senescence.

_FLOWERING = ("Along Flowers", "along the flowering region (first flower -> last)")
_ALONG_BRANCH = ("Along Branch", "along the inflorescence branch (stem -> tip)")
CURVE_PLANT: dict[str, tuple[str, tuple[str, str]]] = {
    "stem_stiffness": ("stiffness", ("Along Axis", "along each axis (base -> tip)")),
}
CURVE_FLOWER: dict[str, tuple[str, tuple[str, str]]] = {
    # a flower component's own curves
    "size": ("flower_size", _FLOWERING),
    "inflorescence_internode_scale": ("flower_spacing", _FLOWERING),
    "flower_pitch": ("flower_pitch", _FLOWERING),
    "pedicel_length": ("pedicel_length", _FLOWERING),
    "pedicel_angle": ("pedicel_angle", _FLOWERING),
    "pedicel_bend": ("pedicel_bend", _FLOWERING),
    "pedicel_width": ("pedicel_width", ("Along Pedicel", "along the pedicel (stem -> flower)")),
    "inflorescence_branch_angle": ("infl_branch_angle", _FLOWERING),
    "inflorescence_branch_nodes": ("infl_branch_size", _FLOWERING),
    "inflorescence_branch_spacing": ("infl_branch_spacing", _FLOWERING),
    "inflorescence_branch_length": ("infl_branch_length", _FLOWERING),
    "inflorescence_branch_width": ("infl_branch_width", _ALONG_BRANCH),
    "inflorescence_branch_bend": ("infl_branch_bend", _FLOWERING),
    "inflorescence_branch_droop": ("infl_branch_droop", _FLOWERING),
    # not a multiplier: the share of the way from Young Size to full, by the
    # inflorescence's own development -- neutral is its straight default
    "inflorescence_young_size": ("young_growth", (
        "By Development", "by the inflorescence's development (just laid down -> mature)")),
}
CURVE_NEUTRAL: dict[str, tuple[tuple[float, float], ...]] = {
    # a curve's "no map" shape where that is not flat at 1 (its profile default)
    "young_growth": ((0.0, 0.0), (1.0, 1.0)),
}


def curve_neutral_y(key: str, x: float) -> float:
    """The neutral curve's value at `x` (straight lines only)."""
    points = CURVE_NEUTRAL.get(key)
    if points is None:
        return 1.0
    (x0, y0), (x1, y1) = points[0], points[-1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
CURVE_ORGAN: dict[str, tuple[str, tuple[str, str]]] = {
    # a leaf / bract / bud / fruit component's
    "size": ("size", ("Instance Age", "by instance age (new tip -> old)")),
}
CURVE_LEAF: dict[str, tuple[str, tuple[str, str]]] = {
    **CURVE_ORGAN,
    "petiole_width": ("petiole_width", ("Along Petiole", "along the petiole (stem -> leaf)")),
}
FLOWERING_CURVES = frozenset(k for k, axis in CURVE_FLOWER.values() if axis == _FLOWERING)
"""Read at a flower's position along the flowering region: undefined where a
kind has no such position (a terminal cluster -- `kind_gradients_apply`)."""


def curve_table(owner: str, component_type: str = "") -> dict[str, tuple[str, tuple[str, str]]]:
    """The profile curves an owner can show, by the parameter they multiply."""
    if owner == OWNER_PLANT:
        return CURVE_PLANT
    if owner == OWNER_LEVEL:
        return {}
    if component_type == "flower":
        return CURVE_FLOWER
    if component_type in ("leaf", "bract"):
        return CURVE_LEAF
    return CURVE_ORGAN


# Rows with no value of their own -- a map icon and a name only.
PSEUDO_PARAMS: dict[str, str] = {
    "instance_pitch": "Pitch (each instance)",
    "instance_yaw": "Yaw (each instance)",
    "instance_roll": "Roll (each instance)",
    "petiole_angle": "Petiole Angle",
    "catkin_hang": "Hang",
    "bloom_timing": "Bloom Timing (each flower)",
    "gravitropic_setpoint": "Gravitropic Setpoint",
}

VARIANCE_PROPS = frozenset(
    list(VARIANCE_LEVEL.values()) + list(VARIANCE_PLANT.values())
    + list(VARIANCE_FLOWER.values()) + list(VARIANCE_LEAF.values())
    + ["leaf_size_variation", "leaf_yaw_variation", "leaf_roll_variation"])
"""Every +/- property: never drawn as a parameter of its own (it is a Random
Range map now), never mappable."""


def variance_table(owner: str, component_type: str = "") -> dict[str, str]:
    """The +/- values an owner can show, by the parameter they vary."""
    if owner == OWNER_LEVEL:
        return VARIANCE_LEVEL
    if owner == OWNER_PLANT:
        return VARIANCE_PLANT
    if component_type == "flower":
        return VARIANCE_FLOWER
    if component_type in ("leaf", "bract"):
        return VARIANCE_LEAF
    return VARIANCE_ORGAN


def hook_target(owner: str, param: str) -> tuple[str, str] | None:
    """(store, target) of a parameter's per-element evaluator -- 'LMAP' a level
    map, 'GMAP' a plant growth map, 'FMAP' an inflorescence map, 'IMAP' an
    instance map -- or None (its maps are per plant)."""
    if owner == OWNER_LEVEL and param in LEVEL_HOOKS:
        return "LMAP", LEVEL_HOOKS[param]
    if owner == OWNER_PLANT and param in PLANT_HOOKS:
        return "GMAP", PLANT_HOOKS[param]
    if owner == OWNER_COMPONENT and param in FLOWER_HOOKS:
        return "FMAP", FLOWER_HOOKS[param]
    if owner == OWNER_COMPONENT and param in INSTANCE_HOOKS:
        return "IMAP", INSTANCE_HOOKS[param]
    return None


# --- "evaluated per" -------------------------------------------------------------

_LEVEL_PER = {
    "pitch": "branch", "yaw": "branch", "roll": "branch", "growth": "branch",
    "start_length": "branch", "start_width": "branch", "length": "node", "width": "node",
    "spiral_pitch": "node", "spiral_roll": "node", "fork_probability": "node",
    "fork_spread": "fork", "emergence_delay": "branch", "setpoint": "node",
}


def evaluated_per(owner: str, param: str, *, stems_level: bool = False,
                  component_type: str = "") -> str:
    """What one value of a parameter's maps belongs to: "plant", "stem",
    "branch", "node", "fork", "flower", "leaf", "instance"..."""
    hook = hook_target(owner, param)
    if owner == OWNER_LEVEL:
        if hook is not None:
            per = _LEVEL_PER.get(hook[1], "branch")
        elif param in VARIANCE_LEVEL:
            per = "node" if param.endswith("_per_step") else "branch"
        else:
            return "plant"
        return "stem" if stems_level and per == "branch" else per
    if owner == OWNER_PLANT:
        if param == "internode_length":
            return "flowering node"
        if param in ("gravitropism", "stem_stiffness"):
            return "axis"
        if param == "basal_spread":
            return "stem"
        return "plant"
    if param.startswith(("petiole_", "emerging_")) and param in INSTANCE_HOOKS:
        return "leaf"
    if param == "bloom_timing":
        return "flower"
    if param in INSTANCE_HOOKS:
        return "instance"
    if param == "inflorescence_internode_scale":
        return "flowering node"
    if param == "inflorescence_branch_spacing":
        return "rachis node"
    table = variance_table(owner, component_type)
    if param in table:
        if param.startswith("inflorescence_branch"):
            return "inflorescence branch"
        if param in ("flower_count", "head_dome", "flower_spread"):
            return "inflorescence"
        if param.startswith("catkin"):
            return "catkin"
        if param.startswith("petiole"):
            return "leaf"
        if param.startswith("gen_"):
            return "instance"
        return "flower" if component_type == "flower" else "instance"
    return "plant"


# --- per-plant maps --------------------------------------------------------------

_T_PLANT_MAP = 0x5EED_0001


def plant_draw(plant_seed: int, owner_seed: int, param: str, map_seed: int) -> float:
    """The per-plant Random driver: one seeded uniform per (plant seed, owner
    seed, parameter, map seed). The parameter by crc32 -- `hash(str)` changes per
    process."""
    return float(np.random.default_rng(
        [int(plant_seed) & 0xFFFFFFFF, int(owner_seed) & 0xFFFFFFFF,
         zlib.crc32(param.encode("utf-8")), int(map_seed) & 0xFFFFFFFF,
         _T_PLANT_MAP]).random())


@dataclass
class PlantMap:
    """One map evaluated once per plant. `curve` maps the draw (x) to a
    multiplier (or, `additive`, to 0..1 round a middle of 0.5); a Random Range
    (`preset` 'RANGE') needs no curve."""
    param: str
    preset: str = "CURVE"
    curve: Curve | None = None
    amount: float = 1.0
    range: float = 0.2
    angle: float = 10.0
    seed: int = 0
    additive: bool = False
    enabled: bool = True


def term(m: PlantMap, u: float) -> tuple[float, float]:
    """(multiplier, offset) one map contributes for draw `u`."""
    if m.preset == "RANGE":
        spread = 2.0 * u - 1.0
        if m.additive:
            return 1.0, m.amount * m.angle * spread
        return 1.0 + m.amount * m.range * spread, 0.0
    y = float(m.curve(u)) if m.curve is not None else (0.5 if m.additive else 1.0)
    if m.additive:
        return 1.0, m.amount * m.angle * (2.0 * y - 1.0)
    return 1.0 + m.amount * (y - 1.0), 0.0


def apply(value, maps, plant_seed: int, owner_seed: int, *, integer: bool = False):
    """A parameter's value under its per-plant maps, in stack order: each
    multiplies (or, an angle, adds). No enabled map returns `value` itself,
    untouched -- the no-map path is byte-identical."""
    active = [m for m in maps if m.enabled]
    if not active:
        return value
    out = float(value)
    for m in active:
        mul, add = term(m, plant_draw(plant_seed, owner_seed, m.param, m.seed))
        out = out * mul + add
    if integer:
        return int(round(out))
    return out
