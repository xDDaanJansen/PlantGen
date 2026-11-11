"""bpy-land bridge: a grown `Plant` becomes two Blender meshes plus the GN
modifier. Replaces `skeleton.py`, which spoke the old NodeArrays model.

Feeds `mesh.py`'s existing Geometry Nodes group unchanged in shape: a stem
mesh of vertices and edges carrying a `width` point attribute, and a sibling
points-only object carrying `kind` / `organ_id` / `direction` / `scale`. That group
was verified numerically against the previous model and its hard-won parts
(width straight into `Curve to Mesh`'s Scale input, never `Set Curve Radius`;
`Mesh to Curve` doing tree-to-splines natively) are all still load-bearing.

One thing this bridge does that the old one could not: axes are explicit
objects here, so a child axis's first vertex *is* its parent phytomer's tip
vertex rather than a coincident duplicate. The skeleton is one connected mesh
by construction instead of by numerical luck.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from collections import Counter
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Matrix

from ..growth import maps as growth_maps_lib
from ..growth import masks, serialize
from ..growth import signals as growth_signals
from ..growth import species as species_lib
from ..growth.curves import Curve
from ..growth.development import build
from ..growth.environment import Environment as GrowthEnvironment
from ..growth.model import BudState, LeafForm, LifecycleState, OrganKind
from ..growth.morphology import (
    Lifespan,
    MorphologyProfile,
    PhenologyCalendar,
    PhenoWindow,
)
from ..growth.physics import stem_centers_of_mass
from . import curves, instancing, mapping, overlay, petioles, structure
from . import environment as env_bridge
from . import mesh as gn_mesh

_PEDICEL_STUB = 0.15
"""How far along a straight stalk its own base vertex sits, as a fraction of
the stalk. Matches roughly where a curved stalk's first polyline point lands,
so a stalk reads the same whether or not it is bent."""


def is_plant(obj) -> bool:
    return bool(obj) and obj.type == 'MESH' and obj.get("plantgen") is not None


def resolve_plant(context):
    """The plant a panel or operator should act on for the current selection: the
    active object if it is a plant, else the plant it belongs to -- its rig, skin,
    organs or a wind-debug object -- found by walking the parent chain, then by
    name. This lets you hide the static plant to watch the animation and still edit
    it by selecting the rig/skin (or the plant itself in the outliner)."""
    obj = getattr(context, "active_object", None)
    if obj is None:
        return None
    if is_plant(obj):
        return obj
    parent, depth = obj.parent, 0
    while parent is not None and depth < 5:      # rig->plant, skin->rig->plant, organs->plant
        if is_plant(parent):
            return parent
        parent, depth = parent.parent, depth + 1
    name = obj.name                              # debug objects are not parented -> by name
    for suffix in ("_WindDebugVolume", "_WindDebugPlane", "_Organs", "_skin", "_Rig"):
        i = name.find(suffix)
        if i > 0:
            cand = bpy.data.objects.get(name[:i])
            if cand is not None and is_plant(cand):
                return cand
    return None


def _stiffness_from_factor(factor: float) -> float:
    """Map the panel's 0-1 Stem Stiffness onto Young's modulus in pascals, which
    spans orders of magnitude.

    The panel shows a 0-1 factor rather than the raw modulus -- "150000000.00"
    is exactly the kind of number the UI contract keeps out of the panel, since
    it means nothing read off a plant. Since the beam was completed to real SI
    (Stage 1), this is a genuine E: the range runs `10^6.7 .. 10^9.7` Pa, i.e.
    ~5 MPa (a weeping, herbaceous stem) to ~5 GPa (hardwood), and 0.5 lands on
    ~1.5e8 -- the provisional default a generic perennial is tuned around."""
    return float(10.0 ** (6.7 + 3.0 * min(1.0, max(0.0, factor))))


def _factor_from_stiffness(value: float) -> float:
    """The inverse, for seeding the panel from a species."""
    from math import log10

    return min(1.0, max(0.0, (log10(max(float(value), 1.0)) - 6.7) / 3.0))


# -- The panel <-> profile mapping, written once ---------------------------
# Both directions read this table: `profile_from_object` applies the panel over
# the embedded base, and `apply_profile` seeds the panel from a base (species or
# preset). Keeping it in one place is not tidiness -- two hand-written copies of
# a 40-field mapping drift, and a field that converts one way but not the other
# fails silently as "applying that preset didn't change anything".
#
# Each entry is (panel property, dotted path on the profile, to-profile,
# to-panel). Both converters default to identity.
_UPPER = (lambda v: v.lower(), lambda v: v.upper())
_LOWER = _UPPER
# Phase 5 lifespan: a `str | None` enum, None being the single-season default.
_LIFESPAN = (
    lambda v: {'ANNUAL': Lifespan.ANNUAL,
               'PERENNIAL': Lifespan.HERBACEOUS_PERENNIAL}.get(v),
    lambda v: {Lifespan.ANNUAL: 'ANNUAL',
               Lifespan.HERBACEOUS_PERENNIAL: 'PERENNIAL'}.get(v, 'NONE'),
)
_STIFFNESS = (_stiffness_from_factor, _factor_from_stiffness)
# A multi-select enum arrives as a set of identifiers and the profile wants a
# frozenset of model values, so the pair is `_LOWER` mapped over a set.
_ATTACHMENTS = (
    lambda v: frozenset(x.lower() for x in v),
    lambda v: {x.upper() for x in v},
)

_FIELDS = (
    # Stem count, splay, forking and branch orders are the branching stack's
    # (`branch_levels`, `_LEVEL_FIELDS`), not flat fields.
    ("basal_spread", "basal_spread"),
    ("basal_spread_variation", "basal_spread_variation"),
    # WS3 senescence. The Age Senescence curve maps itself via the curve
    # system (senescence_onset); this is the retain-vs-shed toggle. The Dead
    # Colour is render-only and set as a modifier input, not on the profile.
    ("senescence_retain_dead_wood", "senescence_retain_dead_wood"),
    ("senescence_shade_threshold", "senescence_shade_threshold"),

    ("inflorescence_kind", "inflorescence.kind", *_UPPER),
    ("inflorescence_orientation", "inflorescence.orientation_mode", *_LOWER),
    ("inflorescence_orientation_deg", "inflorescence.orientation_deg"),
    ("inflorescence_orientation_strength", "inflorescence.orientation_strength"),
    ("inflorescence_orientation_load_driven", "inflorescence.orientation_load_driven"),
    ("inflorescence_branching_depth", "inflorescence.branching_depth"),
    ("inflorescence_branch_angle", "inflorescence.branch_angle_deg"),
    ("inflorescence_branch_nodes", "inflorescence.branch_nodes"),
    ("inflorescence_branch_count", "inflorescence.branch_count"),
    ("inflorescence_branch_count_variation", "inflorescence.branch_count_variation"),
    ("inflorescence_branch_angle_variation", "inflorescence.branch_angle_variation"),
    ("inflorescence_branch_nodes_variation", "inflorescence.branch_nodes_variation"),
    ("inflorescence_branch_spacing", "inflorescence.branch_spacing"),
    ("inflorescence_branch_spacing_variation", "inflorescence.branch_spacing_variation"),
    ("inflorescence_branch_length", "inflorescence.branch_length"),
    ("inflorescence_branch_length_variation", "inflorescence.branch_length_variation"),
    ("inflorescence_branch_width", "inflorescence.branch_width"),
    ("inflorescence_branch_width_variation", "inflorescence.branch_width_variation"),
    ("inflorescence_branch_bend", "inflorescence.branch_bend"),
    ("inflorescence_branch_bend_variation", "inflorescence.branch_bend_variation"),
    ("inflorescence_branch_droop", "inflorescence.branch_droop"),
    ("inflorescence_branch_droop_variation", "inflorescence.branch_droop_variation"),
    ("flower_count", "inflorescence.flower_count"),
    ("inflorescence_whorl_count", "inflorescence.whorl_count"),
    ("inflorescence_spiral", "inflorescence.spiral_deg"),
    # inflorescence.bract_size is a Bract component's Size -- see
    # `_primary_bract_component`, applied in `profile_from_object`.
    ("inflorescence_laterals", "inflorescence.laterals_per_node"),
    ("inflorescence_relay_rotation", "inflorescence.relay_rotation_deg"),
    ("pedicel_length", "inflorescence.pedicel_length"),
    ("pedicel_angle", "inflorescence.pedicel_angle_deg"),
    ("cluster_spread", "inflorescence.cluster_spread_deg"),
    ("head_radius", "inflorescence.head_radius"),
    ("head_dome", "inflorescence.head_dome"),
    ("head_dome_variation", "inflorescence.head_dome_variation"),
    ("pedicel_bend", "inflorescence.pedicel_bend"),
    ("pedicel_width", "inflorescence.pedicel_width"),
    ("pedicel_width_variation", "inflorescence.pedicel_width_variation"),
    ("inflorescence_internode_scale", "inflorescence.internode_scale"),
    ("inflorescence_young_size", "inflorescence.young_size"),
    ("bloom_order", "inflorescence.bloom_order", *_UPPER),
    ("flower_spread", "inflorescence.flower_spread_deg"),
    ("flower_spread_variation", "inflorescence.flower_spread_variation"),
    ("flower_pitch", "inflorescence.flower_pitch_deg"),
    ("flower_pitch_variation", "inflorescence.flower_pitch_variation"),
    ("flower_count_variation", "inflorescence.flower_count_variation"),
    ("flower_spacing_variation", "inflorescence.flower_spacing_variation"),
    ("flower_scale", "flower_size"),
    ("flower_size_variation", "inflorescence.flower_size_variation"),
    ("pedicel_variation", "inflorescence.pedicel_variation"),
    ("pedicel_angle_variation", "inflorescence.pedicel_angle_variation"),
    ("pedicel_bend_variation", "inflorescence.pedicel_bend_variation"),

    ("catkin_length", "catkin_length"),
    ("catkin_count", "catkin_count"),
    ("catkin_count_variation", "catkin_count_variation"),
    ("catkin_noise", "catkin_noise"),
    ("catkin_length_variation", "catkin_length_variation"),
    ("catkin_width", "catkin_width"),
    ("catkin_stalk", "catkin_stalk"),
    ("catkin_stalk_variation", "catkin_stalk_variation"),
    ("catkin_droop_variation", "catkin_droop_variation"),
    ("inflorescence_attachments", "inflorescence.attachments", *_ATTACHMENTS),
    ("inflorescence_per_node", "inflorescence.attachment_per_node"),
    ("inflorescence_internode_position", "inflorescence.internode_position"),
    ("inflorescence_cauline_age", "inflorescence.cauline_min_age"),
    ("inflorescence_on_trunk", "inflorescence.placement.on_trunk"),
    ("inflorescence_on_branches", "inflorescence.placement.on_branches"),
    ("inflorescence_start", "inflorescence.placement.start"),
    ("inflorescence_end", "inflorescence.placement.end"),

    ("internode_length", "internode_length"),
    ("leaf_angle", "leaf_angle_deg"),
    ("leaf_size", "leaf_size"),
    ("leaf_size_variation", "leaf_size_variation"),
    ("leaf_yaw", "leaf_yaw_deg"),
    ("leaf_yaw_variation", "leaf_yaw_variation"),
    ("leaf_roll", "leaf_roll_deg"),
    ("leaf_roll_variation", "leaf_roll_variation"),
    ("gravitropism", "gravitropism"),
    ("circumnutation", "circumnutation_deg"),
    ("circumnutation_rate", "circumnutation_rate_deg"),
    ("elongation_steps", "elongation_steps"),
    ("leaf_expansion_steps", "leaf_expansion_steps"),
    ("autotropism", "autotropism"),

    ("light_seeking", "light_seeking"),
    ("flower_heliotropism", "inflorescence.heliotropism"),
    ("leaf_heliotropism", "leaf_heliotropism"),
    ("obstacle_clearance", "obstacle_clearance"),

    ("emergence_day", "emergence_day"),
    ("season_length_days", "season_length_days"),
    ("lifespan", "lifespan", *_LIFESPAN),

    ("droop", "droop"),
    ("stem_stiffness", "stem_stiffness", *_STIFFNESS),
    ("stiffness_variation", "stiffness_variation"),
    ("rigid_base", "rigid_base"),
    ("leaf_weight", "leaf_weight"),
    ("flower_weight", "flower_weight"),
    ("fruit_size", "fruit_size"),
    ("fruit_weight", "fruit_weight"),
    ("fruit_set", "fruit_set"),
    ("flowering_droop", "flowering_droop"),
    ("peduncle_reach", "peduncle_reach"),
    ("peduncle_resolution", "peduncle_resolution"),
)


def _dotted_set(profile, path: str, value) -> None:
    target = profile
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _dotted_get(profile, path: str):
    target = profile
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _calendar_from_object(s, base_calendar):
    """Build a `PhenologyCalendar` from the plant's authored window properties
    (the calendar widget), or ``None`` when it carries no window at all -- which
    is exactly the ``calendar = None`` an evergreen has, so a Season-only plant
    hits the same developmental gate fallbacks and stays byte-identical.

    ``leaf_out`` / ``leaf_fall`` are not authored yet (deferred to the woody
    deciduous work), so they are carried through from the base calendar untouched
    rather than dropped.
    """
    def win(enabled, start, days):
        return PhenoWindow(start=float(start), duration=float(days)) if enabled else None
    bloom = win(s.bloom_enabled, s.bloom_start, s.bloom_days)
    senescence = win(s.autumn_enabled, s.autumn_start, s.autumn_days)
    fruit = win(s.fruit_enabled, s.fruit_start, s.fruit_days)
    leaf_out = base_calendar.leaf_out if base_calendar is not None else None
    leaf_fall = base_calendar.leaf_fall if base_calendar is not None else None
    if not any((bloom, senescence, fruit, leaf_out, leaf_fall)):
        return None
    return PhenologyCalendar(bloom=bloom, senescence=senescence, fruit=fruit,
                             leaf_out=leaf_out, leaf_fall=leaf_fall)


def _calendar_to_object(s, calendar) -> None:
    """Seed the authored window properties from a profile's calendar -- the
    reflection half, run on preset/species apply. Disabling a phase keeps its
    stored start/duration so re-enabling restores them; only an active window
    writes the numbers. "Autumn" is the ``senescence`` window."""
    def put(prefix, window):
        setattr(s, prefix + "_enabled", window is not None)
        if window is not None:
            setattr(s, prefix + "_start", float(window.start))
            setattr(s, prefix + "_days", float(window.duration))
    put("bloom", calendar.bloom if calendar is not None else None)
    put("autumn", calendar.senescence if calendar is not None else None)
    put("fruit", calendar.fruit if calendar is not None else None)


def _base_profile(obj) -> MorphologyProfile:
    """The base a plant resolves from, before the panel and curves overlay it.

    The embedded preset stash when the plant has one -- so the .blend is
    self-contained and the base never changes unless the user re-applies a preset
    -- falling back to the named species for a plant that predates the preset
    system (or whose stash is somehow unreadable), which reproduces the old
    `species`-dropdown behaviour exactly. Read through `from_json` so an old
    stash migrates forward like any preset file.
    """
    stash = obj.plantgen.preset_embedded
    if stash:
        try:
            return serialize.from_json(stash).profile
        except (ValueError, KeyError, TypeError):
            pass  # corrupt/legacy stash -> fall back rather than break the build
    return species_lib.profile(obj.plantgen.species)


# Flower authoring is being ported from the plant onto the flower component
#. These are the `_FIELDS` whose values move:
# the whole `inflorescence.*` spec, plus the flower-size scalar and the catkin
# body (which live at the profile's top level). When a plant has a flower
# component, its own copies of these override the plant's.
# flower_scale is NOT here: the component's own generic `size` is the flower
# size now (one control, "Size"), fed into profile.flower_size below.
# flower_weight IS here -- a top-level profile field moved onto the component.
_FLOWER_EXTRA_PROPS = frozenset({
    "flower_weight",
    "catkin_length", "catkin_count", "catkin_count_variation",
    "catkin_noise", "catkin_length_variation", "catkin_width", "catkin_stalk",
    "catkin_stalk_variation", "catkin_droop_variation",
})
_FLOWER_FIELDS = tuple(
    e for e in _FIELDS
    if e[1].startswith("inflorescence.") or e[0] in _FLOWER_EXTRA_PROPS)


def primary_leaf_component(s):
    """The leaf component that owns where grown leaves sit -- the first leaf
    component placed On Organs (muted ones included, so Mute stays view-only),
    the way the primary flower component owns the inflorescence. Its
    Phyllotaxis is the plant's leaf arrangement."""
    for c in s.components:
        if c.type == 'leaf' and 'ORGAN' in set(c.placement):
            return c
    return None


def _leaf_arrangement_from(c):
    from ..growth.morphology import LeafArrangement
    return LeafArrangement(spiral_angle_deg=math.degrees(float(mapping.cval(c, "gen_divergence"))),
                           whorl=int(mapping.cval(c, "gen_whorl_count")),
                           whorl_angle_deg=math.degrees(float(mapping.cval(c, "gen_whorl_angle"))))


def _leaf_arrangement_to(c, arrangement) -> None:
    """Write a profile's leaf arrangement onto a leaf component's Phyllotaxis --
    raw, so the preset/Custom callbacks do not fire -- and name the pattern
    when it is one of the dropdown's presets."""
    from .properties import _PHYLLO_PRESETS
    vals = (math.radians(arrangement.spiral_angle_deg), int(arrangement.whorl),
            math.radians(arrangement.whorl_angle_deg))
    c["gen_divergence"], c["gen_whorl_count"], c["gen_whorl_angle"] = vals
    name = next((k for k, v in _PHYLLO_PRESETS.items()
                 if abs(v[0] - vals[0]) < 1e-6 and v[1] == vals[1]
                 and abs(v[2] - vals[2]) < 1e-6), "CUSTOM")
    c["gen_phyllotaxis"] = c.bl_rna.properties["gen_phyllotaxis"].enum_items[name].value


_LEVELS_SCHEMA = 9
"""The branching stack's per-object schema (`s["branch_levels_schema"]`):
2 -- the stems level's Start Length / Start Width in metres, the leaf
arrangement on the primary leaf component; 3 -- the internode and radius
variations and the cross-section on the levels; 4 -- Show Stems as each
level's Show box; 5 -- the Growth Maps on branch angle / length / forking as
level maps; 6 -- the levels as a tree (each its parent) and the components
placed on levels (the Plant Structure panel); 7 -- each level's Fork toggle as
its Growth Pattern; 8 -- Axis Type gone, an Indeterminate plant's flowers
Axillary. A plant regrown under an older one is converted once, step
by step."""

# The plant's old Cross Section dropdown, in item order: an EnumProperty stores
# the index, which is all an object regrown before the move still holds.
_OLD_SECTIONS = ("circular", "triangular", "square", "hexagonal", "polygonal",
                 "flattened", "winged", "ribbed", "star", "custom")


def _migrate_plant(s, base) -> None:
    """Bring a plant's own settings to `_LEVELS_SCHEMA` once, from the values
    its removed plant-level properties still hold:

    * to 2: the stems level's Start Length / Width x the old Internode Length /
      Base Radius (so the stems keep their size), and the primary leaf
      component seeded with the base profile's leaf arrangement;
    * to 3: the old Internode Variation onto every level's Length variation,
      Radius Variation onto the stems level's Width variation, and the old
      Cross Section (with its sides / lobes / flatten) onto every level;
    * to 4: Show Stems off onto the stems level's Show (which hides them all);
    * to 5: Growth Maps on Branch Angle / Branch Length / Forking onto the levels
      as Pitch / Steps / Fork Probability maps, and Internode Length copied onto
      every level as Length (it stays, now on flowering nodes) --
      `growth.maps.LEGACY_TO_LEVEL`;
    * to 6: each level's parent the level before it (the chain it was), and
      each component's All Levels / level links as placements;
    * to 7: a level's Fork on as its Growth Pattern Dichotomous;
    * to 8: Axis Type removed -- an Indeterminate plant's inflorescences
      Axillary instead of Terminal (on the plant and its flower components);
    * to 9: several inflorescences -- every flower component after the first
      keeps being a variant (Own Inflorescence off)."""
    schema = s.get("branch_levels_schema", 1)
    if schema >= _LEVELS_SCHEMA:
        return
    try:
        if schema < 2:
            if len(s.branch_levels):
                stems = s.branch_levels[0]
                stems.start_length_scale *= float(s.get("internode_length", 0.045))
                stems.start_width_scale *= float(s.get("base_radius", 0.005))
            leaf = primary_leaf_component(s)
            if leaf is not None:
                _leaf_arrangement_to(leaf, base.leaf_arrangement)
        if schema < 3:
            index = int(s.get("cross_section", 0))
            shape = _OLD_SECTIONS[index] if 0 <= index < len(_OLD_SECTIONS) else "circular"
            for i, lv in enumerate(s.branch_levels):
                lv.length_variation = float(s.get("internode_variation", 0.18))
                lv.width_variation = float(s.get("radius_variation", 0.0)) if i == 0 else 0.0
                lv.cross_section = "circular" if shape == "custom" else shape
                lv.profile_sides = int(s.get("profile_sides", 8))
                lv.lobe_count = int(s.get("lobe_count", 5))
                lv.lobe_depth = float(s.get("lobe_depth", 0.35))
                lv.flatten = float(s.get("flatten", 0.6))
        if schema < 4 and not s.get("enable_stem", True) and len(s.branch_levels):
            s.branch_levels[0].show = False
        if schema < 5 and len(s.branch_levels):
            _migrate_growth_maps_to_levels(s)
        if schema < 6:
            _migrate_to_tree(s)
        if schema < 7:
            for lv in s.branch_levels:
                if lv.fork:
                    lv.growth_pattern = 'dichotomous'
        if schema < 8 and s.axis_type == 'INDETERMINATE':
            # Indeterminate's tips never flowered and its laterals did: its
            # inflorescences become Axillary instead of Terminal -- on the
            # plant (a new flower component's seed) and on every flower
            # component, which owns the inflorescence.
            for holder in [s, *(c for c in s.components if c.type == 'flower')]:
                attached = set(holder.inflorescence_attachments)
                attached.discard('TERMINAL')
                attached.add('AXILLARY')
                holder.inflorescence_attachments = attached
            s.axis_type = 'DETERMINATE'
        if schema < 9:
            # One inflorescence per plant until now: every flower component
            # after the first was a variant on its flowers -- it stays one.
            for c in [c for c in s.components if c.type == 'flower'][1:]:
                c.own_inflorescence = False
        s["branch_levels_schema"] = _LEVELS_SCHEMA
    except (AttributeError, RuntimeError):
        pass  # not writable while a panel draws -- done on the next regrow


def _migrate_to_tree(s) -> None:
    structure.ensure_level_uids(s)
    for i, lv in enumerate(s.branch_levels):
        lv.parent_uid = s.branch_levels[i - 1].uid if i > 0 else 0
    structure.ensure_component_uids(s)
    s.placements.clear()
    for c in s.components:
        if c.all_levels:
            structure.place(s, 0, c.uid)
        else:
            for link in c.level_links:
                structure.place(s, link.uid, c.uid)
    structure.rebuild_tree(s)


# The Growth Maps target enum's stored numbers of the targets that moved.
_LEGACY_GROWTH_TARGETS = {1: growth_maps_lib.BRANCH_ANGLE, 2: growth_maps_lib.INTERNODE_LENGTH,
                          3: growth_maps_lib.BRANCH_LENGTH, 5: growth_maps_lib.FORK_CHANCE}


def _migrate_growth_maps_to_levels(s) -> None:
    obj = s.id_data
    next_uid = 1 + max((m.uid for lv in s.branch_levels for m in lv.maps), default=0)
    drop = []
    for i, gm in enumerate(s.growth_maps):
        # the raw stored number (a removed enum item reads as nothing); unset is
        # the old default, Branch Angle
        legacy = _LEGACY_GROWTH_TARGETS.get(int(gm.get("target", 1)))
        rule = growth_maps_lib.LEGACY_TO_LEVEL.get(legacy)
        if rule is None:
            continue
        target, first, stays = rule
        src = curves.ensure_map_curve(
            obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name("gmap", gm.uid))
        for lv in list(s.branch_levels)[first:]:
            m = lv.maps.add()
            m.uid = next_uid
            next_uid += 1
            m.target = target
            m.channel, m.attr_name, m.class_name = gm.channel, gm.attr_name, gm.class_name
            m.amount, m.enabled = gm.amount, gm.enabled
            curves.copy_curve(src, curves.ensure_map_curve(
                obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name(LEVEL_MAP_KIND, m.uid)))
        if stays:
            gm.target = growth_maps_lib.INTERNODE_LENGTH
        else:
            drop.append(i)
    for i in reversed(drop):
        curves.remove_named_component_curve(
            obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name("gmap", s.growth_maps[i].uid))
        s.growth_maps.remove(i)
    s.growth_maps_active = 0


def _primary_flower_component(s):
    """The flower component that owns the plant's primary inflorescence -- the
    first flower-type component (the plant's flower settings seed it)."""
    for c in s.components:
        if c.type == 'flower':
            return c
    return None


def inflorescence_owners(s) -> list:
    """The flower components that grow an inflorescence of their own, in
    priority order: the first flower component always, then every later one
    with Own Inflorescence on. Position i is inflorescence i
    (`MorphologyProfile.inflorescence_at`); the rest are variants that draw on
    the flowers grown on their levels."""
    out = []
    for c in s.components:
        if c.type == 'flower' and (not out or c.own_inflorescence):
            out.append(c)
    return out


def component_inflorescence(s, c):
    """The inflorescence index flower component `c` grows, or None for a
    variant."""
    for i, owner in enumerate(inflorescence_owners(s)):
        if owner.uid == c.uid:
            return i
    return None


def inflorescence_levels(s, i: int) -> set:
    """The level indices inflorescence `i` grows on: its owner's placement --
    the primary's with every variant's too (a variant draws its flowers)."""
    owners = inflorescence_owners(s)
    if not (0 <= i < len(owners)):
        return set()
    out = set(structure.component_level_indices(s, owners[i]))
    if i == 0:
        for c in s.components:
            if c.type == 'flower' and component_inflorescence(s, c) is None:
                out.update(structure.component_level_indices(s, c))
    return out


def contested_spots(s, c) -> list:
    """`(attachment label, level name, winner label)` for every spot flower
    component `c`'s own inflorescence asks for on a level where one higher in
    the list already wants it -- the spots it loses."""
    mine = component_inflorescence(s, c)
    if not mine:
        return []
    owners = inflorescence_owners(s)
    names = [lv.name for lv in s.branch_levels]
    labels = {e.identifier: e.name
              for e in c.bl_rna.properties["inflorescence_attachments"].enum_items}
    out = []
    for level in sorted(inflorescence_levels(s, mine)):
        for mode in sorted(c.inflorescence_attachments):
            for i in range(mine):
                if (level in inflorescence_levels(s, i)
                        and mode in owners[i].inflorescence_attachments):
                    out.append((labels.get(mode, mode), names[level] if level < len(names)
                                else "?", owners[i].label or owners[i].name))
                    break
    return out


def _hidden_inflorescences(s) -> set:
    """The inflorescences no live component draws: the owner muted and no
    unmuted variant to take its flowers. Their stalks and inflorescence axes
    go with them, as a muted flower component's always did."""
    owners = inflorescence_owners(s)
    live_variant = any(c.type == 'flower' and not c.mute
                       and component_inflorescence(s, c) is None for c in s.components)
    return {i for i, c in enumerate(owners) if c.mute and not live_variant}


def _flowers_live(s) -> bool:
    """Whether the plant's flowers render: a flower component that is not
    muted, or -- with no flower component at all -- the plant's own flag. A
    muted component turns off everything that is its: flowers, pedicels,
    catkins and inflorescence branches. Growth is untouched (the primary
    component still sources the inflorescence), so muting is a pure view
    toggle and un-muting brings back the identical plant."""
    comps = [c for c in s.components if c.type == 'flower']
    if not comps:
        return bool(s.enable_flowers)
    return any(not c.mute for c in comps)


def _inflorescence_axes(plant, which=None) -> set:
    """Every axis that is inflorescence structure rather than stem: a branch,
    ray, relay or node-borne stalk the inflorescence released
    (`structural_depth >= 1` and not a vegetative fork), with its whole subtree.
    What a muted flower component hides along with its flowers. `which`: only
    those of these inflorescences (`Axis.inflorescence`)."""
    hidden: set = set()
    for axis in plant.axes_in_topological_order():
        if axis.parent_axis_id in hidden or (
                axis.structural_depth >= 1 and not axis.is_fork
                and (which is None or max(axis.inflorescence, 0) in which)):
            hidden.add(axis.id)
    return hidden


def _hidden_organ_ids(plant, hidden_axes: set) -> set:
    """The organs (and terminal organs) carried on `hidden_axes`."""
    out: set = set()
    for aid in hidden_axes:
        axis = plant.axes[aid]
        for phytomer in axis.phytomers:
            out.update(o.id for o in phytomer.organs)
        if axis.terminal_organ is not None:
            out.add(axis.terminal_organ.id)
    return out


def _primary_bract_component(s):
    """The bract component that owns grown bracts -- the first bract-type
    component placed On Organs (muted included, so muting never changes the
    grown plant). Its Size is `inflorescence.bract_size`: the bract as a
    fraction of the leaf its flowering node would have carried."""
    for c in s.components:
        if c.type == 'bract' and 'ORGAN' in c.placement:
            return c
    return None


def _buds_live(s) -> bool:
    """Whether the plant's grown buds and apices are emitted as points: a Bud
    component placed On Organs that is not muted. There is no plant switch any
    more -- a Bud component *is* the switch (the bract rule), muting it is a pure
    view toggle, and which buds show (tip vs side) is its masks' call."""
    return any(c.type == 'bud' and not c.mute and 'ORGAN' in c.placement
               for c in s.components)


_BUDS_MIGRATED_KEY = "plantgen_buds_migrated"


def _seed_bud_components(obj, s) -> None:
    """Once per plant: carry an old file's Buds panel onto its Bud components.

    Before the port a bud rendered at `bud_size` (metres) x the component's Size,
    and only if `enable_buds` / `enable_apices` let its points through. Those
    props are gone; their stored values survive as raw ID properties on the
    group, read here with `.get` and only if they were ever set. An On-Organs-only
    component takes `bud_size` into its Size (so it renders at the size it did)
    whenever the old panel was used at all, and one of the two switches on its
    own becomes an Apex mask (tip buds only, or side buds only). A component
    that also generates points is left as it is: those points never read
    `bud_size`, so one Size cannot keep both.

    The object is stamped migrated either way, so components added later start
    clean at Size 1."""
    if obj.get(_BUDS_MIGRATED_KEY):
        return
    obj[_BUDS_MIGRATED_KEY] = True
    buds_on = bool(s.get("enable_buds", False))
    apices_on = bool(s.get("enable_apices", False))
    stored_size = s.get("bud_size")
    # Seed when the old panel was ever used: a switch on, or a Bud Size set --
    # a component tuned against a set Bud Size (Size 4 x 0.06 m) would otherwise
    # jump to Size x its raw asset the moment it is unmuted.
    if not (buds_on or apices_on or stored_size is not None):
        return
    bud_size = float(stored_size) if stored_size is not None else 0.004
    for c in s.components:
        if c.type != 'bud' or set(c.placement) != {'ORGAN'}:
            continue
        c.size = float(c.size) * bud_size
        if buds_on != apices_on:
            f = c.filters.add()
            f.channel = "apex"
            f.invert = buds_on          # buds only -> keep the non-apex side


def _primary_fruit_component(s):
    """The fruit component that owns the plant's fruit -- the first one placed On
    Organs, muted included (so muting never changes the grown plant). Its Size
    is the fruit size and its Fruit Set / Weight are the plant's. Only once it
    has been seeded, so a profile read outside a regrow (saving a preset) never
    picks up a fresh component's defaults."""
    for c in s.components:
        if c.type == 'fruit' and 'ORGAN' in c.placement:
            return c if c.fruit_authored else None
    return None


def _fruit_live(s) -> bool:
    """Whether grown fruit render: an unmuted fruit component, or -- with no
    fruit component at all -- the plant's own flag (like flowers). Growth is
    untouched either way: fruit still set and still load the branch."""
    comps = [c for c in s.components if c.type == 'fruit']
    if not comps:
        return bool(s.enable_fruit)
    return any(not c.mute for c in comps)


def _seed_fruit_components(obj, s) -> None:
    """Copy the plant's fruit config onto any not-yet-seeded fruit component, once
    (the `fruit_authored` guard): Fruit Set and Weight, and -- for one placed On
    Organs -- the plant's Fruit Size as its Size, the flower precedent (one size
    control, Size)."""
    for c in s.components:
        if c.type != 'fruit' or c.fruit_authored:
            continue
        c.fruit_set = s.fruit_set
        c.fruit_weight = s.fruit_weight
        if 'ORGAN' in c.placement:
            c.size = float(s.fruit_size)
        c.fruit_authored = True


def _seed_flower_components(obj, s) -> None:
    """Copy the plant's flower config (scalars and curves) onto any not-yet-
    authored flower component, so the component inherits the plant's inflorescence
    (the additive 'copy over' step) instead of resetting it to component defaults.
    Runs once per component (the `flower_authored` guard). `inflorescence_kind` is
    set first so its kind-default stamp cannot land on top of the copied values."""
    for c in s.components:
        if c.type != 'flower' or c.flower_authored:
            continue
        if hasattr(c, "inflorescence_kind"):
            c.inflorescence_kind = s.inflorescence_kind
        for prop, _path, *_conv in _FLOWER_FIELDS:
            if hasattr(c, prop) and hasattr(s, prop):
                setattr(c, prop, getattr(s, prop))
        # The component's Size is the flower size now, seeded from the plant's
        # old Flower Size so the flower keeps its scale.
        c.size = float(s.flower_scale)
        curves.seed_component_flower_curves(obj, c)
        c.flower_authored = True
    for c in s.components:
        if c.type == 'flower' and not c.heliotropism_authored:
            c.flower_heliotropism = float(s.flower_heliotropism)
            c.heliotropism_authored = True
        if c.type == 'flower' and not c.sun_authored:
            c.sun_object = s.sun_object
            c.sun_authored = True


def plant_sun_used(s) -> bool:
    """Whether the plant's own Sun Object is read: for Leaf Heliotropism, or
    for the flowers of a plant with no flower component."""
    return s.leaf_heliotropism > 0.0 or (
        not inflorescence_owners(s) and s.flower_heliotropism > 0.0)


def sun_objects(s) -> list:
    """Every object this plant's heliotropism tracks, in a fixed order: the
    plant's (if read), then each inflorescence owner's that turns its flowers."""
    out = [s.sun_object] if plant_sun_used(s) and s.sun_object is not None else []
    out.extend(c.sun_object for c in inflorescence_owners(s)
               if c.sun_object is not None and c.flower_heliotropism > 0.0)
    return out


def _seed_level_stiffness(s) -> None:
    """Give a level the plant's Stiffness once (its `stiffness_authored`
    guard), so moving the control onto the levels leaves every existing plant
    bending exactly as it did."""
    for lv in s.branch_levels:
        if not lv.stiffness_authored:
            lv.stiffness = float(s.stem_stiffness)
            lv.stiffness_authored = True


def _leaf_loads(s) -> list:
    """Each leaf component's placed leaves as `LeafLoad`s -- one per generated
    placement it uses -- so the bend weighs them at the nodes they will sit on
    (`physics._leaf_load_mass`). Mute is view-only, so a muted component still
    weighs; a component placed nowhere has no leaves and none."""
    from ..growth.morphology import LeafLoad

    loads = []
    for c in s.components:
        if c.type != 'leaf' or c.collection is None:
            continue
        on_plant = any(q.comp_uid == c.uid and q.level_uid == 0 for q in s.placements)
        levels = () if on_plant else tuple(sorted(structure.component_level_indices(s, c)))
        if not on_plant and not levels:
            continue
        mass = (float(mapping.cval(c, "leaf_weight")) * float(c.size) ** 2
                * (1.0 - float(c.prune)))
        if mass <= 0.0:
            continue
        whorl = max(1, int(mapping.cval(c, "gen_whorl_count")))
        counts = {'NODE': ("node", whorl), 'TIP': ("tip", whorl),
                  'INTERSECTION': ("intersection", whorl),
                  'INTERNODE': ("internode",
                                whorl * max(1, int(mapping.cval(c, "internode_count")))),
                  'BASAL': ("basal", whorl * max(1, int(mapping.cval(c, "basal_count")))),
                  'EXTRA': ("random", float(mapping.cval(c, "extra_count")))}
        for mode in c.placement:
            if mode in counts:
                name, count = counts[mode]
                loads.append(LeafLoad(mass=mass, mode=name, count=float(count),
                                      levels=levels))
    return loads


def _seed_leaf_components(s) -> None:
    """Give a leaf component the plant's Leaf Weight once (its
    `leaf_weight_authored` guard), so moving the control onto the component
    leaves every existing plant's bend exactly as it was."""
    for c in s.components:
        if c.type == 'leaf' and not c.leaf_weight_authored:
            c.leaf_weight = float(s.leaf_weight)
            c.leaf_weight_authored = True


def profile_from_object(obj) -> MorphologyProfile:
    """The base profile, with the panel's own values applied over it.

    Starting from a rich base rather than a bare `MorphologyProfile()` is what
    lets a preset be richer than the parameter surface: `plastochron`,
    `pipe_per_leaf`, organ masses, `veg`, `topology_override` and the rest are real
    traits with no slider, and they survive because they are simply never
    overwritten here. Anything the panel *does* expose is the user's, and wins --
    including the curve widgets, which `apply_profile` seeded from this same base.
    """
    profile = _base_profile(obj)
    profile.name = obj.name
    s = obj.plantgen
    _migrate_plant(s, profile)
    # Per-plant maps: every mapped value is
    # replaced here, where the panel becomes the model; none, nothing changes.
    plant_maps = mapping.plant_index(obj) if len(s.param_maps) else {}
    for prop, path, *converters in _FIELDS:
        to_profile = converters[0] if converters else None
        value = mapping.mapped(obj, plant_maps, mapping.PLANT, 0, prop, getattr(s, prop))
        _dotted_set(profile, path, to_profile(value) if to_profile else value)
    # Port (additive): a flower component's own copies of the inflorescence
    # fields override the plant's, which remain the legacy fallback. Only fields
    # actually present on the component are overridden, so this grows field by
    # field as they are copied across (properties._PORTED_FLOWER_PROPS).
    flower_comp = _primary_flower_component(s)
    if flower_comp is not None:
        for prop, path, *converters in _FLOWER_FIELDS:
            if not hasattr(flower_comp, prop):
                continue
            to_profile = converters[0] if converters else None
            value = mapping.mapped(obj, plant_maps, mapping.COMPONENT, flower_comp.uid, prop,
                                   getattr(flower_comp, prop))
            _dotted_set(profile, path, to_profile(value) if to_profile else value)
        # The flower's size is the component's own Size (one control), fed into
        # the growth base size; the flower_size gradient curve + variance still
        # modulate it. No bake `*= size` for flowers (that would double it).
        _dotted_set(profile, "flower_size", float(flower_comp.size))
        # Placement is by masking now, not an along-stem window: an inflorescence
        # arises on every eligible node and the component's masks thin the flower
        # points. The on-trunk/on-branches/start/end controls are gone.
        _dotted_set(profile, "inflorescence.placement.on_trunk", True)
        _dotted_set(profile, "inflorescence.placement.on_branches", True)
        _dotted_set(profile, "inflorescence.placement.start", 0.0)
        _dotted_set(profile, "inflorescence.placement.end", 1.0)
    # Bracts exist because a Bract component asks for them, at its Size -- the
    # one bract-size control. No bract component, no bracts grown.
    bract_comp = _primary_bract_component(s)
    _dotted_set(profile, "inflorescence.bract_size",
                float(bract_comp.size) if bract_comp is not None else 0.0)
    # Fruit follow the primary fruit component: its Size is the fruit size (the
    # plant's Fruit Size merged into it -- fruit mass goes with its cube), its
    # Fruit Set and Weight are the plant's. With none, the plant's own values.
    fruit_comp = _primary_fruit_component(s)
    if fruit_comp is not None:
        profile.fruit_size = float(fruit_comp.size)
        profile.fruit_set = float(mapping.cval(fruit_comp, "fruit_set"))
        profile.fruit_weight = float(mapping.cval(fruit_comp, "fruit_weight"))
    # The phenology calendar is authored per-object once a preset is applied or
    # the calendar widget is used (`calendar_authored`); until then the base's
    # own calendar stands, so a plant that predates this is byte-identical. It is
    # a nested PhenologyCalendar, not a flat field, so it is rebuilt here rather
    # than through `_FIELDS`.
    if s.calendar_authored:
        profile.calendar = _calendar_from_object(s, profile.calendar)
    # Creates whatever this plant is missing, seeded from the base so a curve
    # added to the registry arrives at the base's own shape rather than flat --
    # and leaves every curve the owner already authored alone.
    curves.ensure_curves(obj, profile)
    curves.apply_to_profile(obj, profile)
    # Flower curves follow the flower fields onto the component: override the
    # inflorescence gradient curves from the primary flower component's own ones.
    if flower_comp is not None:
        curves.apply_component_flower_curves(obj, flower_comp, profile)
        profile.inflorescence.maps = _flower_maps(obj, flower_comp)
    profile.maps = _growth_maps(obj, s)
    if len(s.branch_levels) == 0:
        try:
            # A plant from before the branching stack: seed it from its base,
            # so the panel shows the levels the plant is actually built with.
            _levels_to_object(s, profile.levels)
        except (AttributeError, RuntimeError):
            pass      # not writable while a panel draws -- the base's stand
    if len(s.branch_levels):
        profile.levels = _levels_from_object(s)
        # Flowers grow only on the levels a flower component is placed on,
        # muted or not (Mute is view-only); with none, nowhere -- a level none
        # is on takes no flower spacing or orientation either. Each component
        # with its own inflorescence grows it on its levels; a variant's levels
        # are the primary's too, so it has flowers to draw wherever it is.
        owners = inflorescence_owners(s)
        primary_levels = set()
        for c in s.components:
            if c.type == 'flower' and (not owners or c.uid == owners[0].uid
                                       or component_inflorescence(s, c) is None):
                primary_levels.update(structure.component_level_indices(s, c))
        _dotted_set(profile, "inflorescence.placement.levels", tuple(sorted(primary_levels)))
        profile.extra_inflorescences = [
            _component_inflorescence(obj, s, c, profile) for c in owners[1:]]
    leaf = primary_leaf_component(s)
    if leaf is not None:
        profile.leaf_arrangement = _leaf_arrangement_from(leaf)
        # The grown leaves' weight is the component's that owns them.
        if leaf.leaf_weight_authored:
            profile.leaf_weight = float(mapping.cval(leaf, "leaf_weight"))
    elif instancing.has_components(s, 'leaf'):
        # Leaf components, none On Organs: the grown leaves draw nothing, so they
        # weigh nothing -- every leaf that weighs is one you can see.
        profile.leaf_weight = 0.0
    # The leaves components place themselves weigh too, at their nodes.
    profile.leaf_loads = _leaf_loads(s)
    return profile


def _component_inflorescence(obj, s, comp, profile):
    """A further flower component's own inflorescence: the primary's as a base,
    overridden by every inflorescence field and curve the component carries,
    at its Size, on its levels."""
    import copy
    import types
    holder = types.SimpleNamespace(inflorescence=copy.deepcopy(profile.inflorescence))
    plant_maps = mapping.plant_index(obj) if len(s.param_maps) else {}
    for prop, path, *converters in _FLOWER_FIELDS:
        if not path.startswith("inflorescence.") or not hasattr(comp, prop):
            continue
        to_profile = converters[0] if converters else None
        value = mapping.mapped(obj, plant_maps, mapping.COMPONENT, comp.uid, prop,
                               getattr(comp, prop))
        _dotted_set(holder, path, to_profile(value) if to_profile else value)
    curves.apply_component_flower_curves(obj, comp, holder)
    inf = holder.inflorescence
    inf.maps = _flower_maps(obj, comp)
    inf.placement.on_trunk = inf.placement.on_branches = True
    inf.placement.start, inf.placement.end = 0.0, 1.0
    inf.placement.levels = tuple(structure.component_level_indices(s, comp))
    inf.flower_size = float(comp.size)
    return inf


# The branching stack's panel <-> profile mapping: (level property, BranchLevel
# field). Names differ only by the `_deg` the growth side carries.
_LEVEL_FIELDS = (
    ("phyllotaxis", "phyllotaxis"),
    ("spiral_angle", "spiral_angle_deg"),
    ("whorl", "whorl"),
    ("whorl_angle", "whorl_angle_deg"),
    ("slide", "slide"),
    ("slide_variation", "slide_variation"),
    ("start_pitch", "start_pitch_deg"),
    ("start_yaw", "start_yaw_deg"),
    ("start_roll", "start_roll_deg"),
    ("start_pitch_variation", "start_pitch_variation_deg"),
    ("start_yaw_variation", "start_yaw_variation_deg"),
    ("start_roll_variation", "start_roll_variation_deg"),
    ("upper_side_up", "upper_side_up"),
    ("steps", "steps"),
    ("growth", "growth"),
    ("emergence_delay", "emergence_delay"),
    ("start_length_scale", "start_length_scale"),
    ("length_scale_per_step", "length_scale_per_step"),
    ("length_variation", "length_variation"),
    ("start_width_scale", "start_width_scale"),
    ("width_scale_per_step", "width_scale_per_step"),
    ("width_variation", "width_variation"),
    ("seed", "seed"),
    # cross_section before lobe_count / lobe_depth: its update stamps the shape's
    # defaults, which the level's own values then overwrite
    ("cross_section", "cross_section"),
    ("profile_sides", "profile_sides"),
    ("lobe_count", "lobe_count"),
    ("lobe_depth", "lobe_depth"),
    ("flatten", "flatten"),
    ("spiral_pitch", "spiral_pitch_deg"),
    ("spiral_roll", "spiral_roll_deg"),
    ("growth_pattern", "growth_pattern"),
    ("unit_steps", "unit_steps"),
    ("relay_count", "relay_count"),
    ("fork_balance", "fork_balance"),
    ("fork_branches", "fork_branches"),
    ("fork_probability", "fork_probability"),
    ("fork_spread", "fork_spread_deg"),
    ("fork_whorl", "fork_whorl_deg"),
)


_LEVEL_SCALE = {"growth": 0.01}
"""Level properties the panel shows in other units than the profile holds:
Growth % is 1-100 on the panel, a 0-1 fraction in `BranchLevel.growth`."""


def _level_name(i: int) -> str:
    return "Stems" if i == 0 else f"Branches {i}"


def _resolved_parents(levels) -> list:
    """Each level's parent index (-1: a stems level), as growth reads them."""
    from types import SimpleNamespace

    from ..growth.morphology import MorphologyProfile
    view = SimpleNamespace(levels=levels)
    return [MorphologyProfile.level_parent(view, i) for i in range(len(levels))]


def _default_level_names(parents) -> list:
    """Stems (Stems 2, 3.. for more stems levels), Branches <depth> below."""
    names, roots = [], 0
    for p in parents:
        if p < 0:
            roots += 1
            names.append("Stems" if roots == 1 else f"Stems {roots}")
        else:
            depth, q = 1, p
            while parents[q] >= 0:
                q = parents[q]
                depth += 1
            names.append(f"Branches {depth}")
    return names


LEVEL_MAP_KIND = "lmap"
"""A level map's curve-node prefix, under `curves.GROWTH_MAP_OWNER`; its uid is
unique over every level's maps, so a level can move without renaming curves."""


def _mask_entries(filters) -> list:
    """A mask list (a component's, a class's, a level's) as `MaskEntry`s."""
    return [masks.MaskEntry(channel=channel_key(e), lo=float(e.lo), hi=float(e.hi),
                            invert=bool(e.invert), op=e.op, mute=bool(e.mute),
                            modulo=int(e.math_value) if e.math_op == 'MODULO' else 0)
            for e in filters]


def _set_channel_ref(ref, key: str) -> None:
    """Point a mask entry / map at a channel key (`attr:` / `class:` included)."""
    if key.startswith(masks.CUSTOM_PREFIX):
        ref.channel, ref.attr_name = "custom", key[len(masks.CUSTOM_PREFIX):]
    elif key.startswith(masks.CLASS_PREFIX):
        ref.channel, ref.class_name = "class", key[len(masks.CLASS_PREFIX):]
    else:
        ref.channel = key


def _curve_points(curve) -> tuple:
    """A curve's authoring points, or -- for one sampled from a widget, which
    has none -- 17 of its samples, so it is not written back flat."""
    if curve.points:
        return tuple(curve.points)
    samples = list(curve.samples)
    if len(samples) < 2:
        return ((0.0, 1.0), (1.0, 1.0))
    xs = np.linspace(0.0, 1.0, 17)
    ys = np.interp(xs, np.linspace(0.0, 1.0, len(samples)), samples)
    return tuple((float(x), float(y)) for x, y in zip(xs, ys, strict=True))


def _levels_from_object(s) -> list:
    """The panel's branching stack as `BranchLevel`s, in order, with each
    level's masks and maps. A map carries its curve node's uid (`_uid`, not a
    field -- never serialised), so `_levels_to_object` keeps its drawn curve."""
    from ..growth.morphology import BranchLevel
    obj = s.id_data
    out = []
    plant_maps = mapping.plant_index(obj) if len(s.param_maps) else {}
    for lv in s.branch_levels:
        kw = {}
        for prop, fld in _LEVEL_FIELDS:
            v = mapping.mapped(obj, plant_maps, mapping.LEVEL, lv.uid, prop, getattr(lv, prop))
            kw[fld] = v if isinstance(v, (bool, str)) else (
                int(v) if isinstance(v, int) else float(v))
            if prop in _LEVEL_SCALE:
                kw[fld] *= _LEVEL_SCALE[prop]
        # Stiffness: the level's own (a 0-1 factor -> pascals), once seeded from
        # the plant's (`_seed_level_stiffness`); 0 = the plant's.
        kw["stiffness"] = (_stiffness_from_factor(lv.stiffness) if lv.stiffness_authored
                           else 0.0)
        kw["masks"] = _mask_entries(lv.filters)
        maps = []
        for m in lv.maps:
            node = curves.get_component_curve(
                obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name(LEVEL_MAP_KIND, m.uid))
            pm = growth_maps_lib.ParamMap(
                target=m.target, channel=channel_key(m),
                curve=curves.to_curve(node) if node is not None else Curve.constant(1.0),
                amount=float(m.amount), enabled=bool(m.enabled), seed=int(m.seed))
            pm._uid = m.uid
            maps.append(pm)
        kw["maps"] = maps
        kw["name"] = lv.name
        index = {x.uid: k for k, x in enumerate(s.branch_levels)}
        kw["parent"] = index.get(lv.parent_uid, -1) if lv.parent_uid else -1
        level = BranchLevel(**kw)
        if lv.uid > 0:
            level._uid = lv.uid       # not a field: the level's identity on this object
        out.append(level)
    return out


def _level_maps_to_object(obj, lv, maps, old_nodes: dict, used: set, next_uid: list) -> None:
    """Write a level's maps: a map that came from this object keeps its uid and
    curve node (the first time it appears -- a copied level's maps get new uids
    and copies of the node); any other is written from its curve."""
    for pm in maps:
        m = lv.maps.add()
        m.target = pm.target
        _set_channel_ref(m, pm.channel)
        m.amount = float(pm.amount)
        m.enabled = bool(pm.enabled)
        m.seed = int(getattr(pm, "seed", 0))
        src = getattr(pm, "_uid", None)
        if src in old_nodes and src not in used:
            m.uid = src
            used.add(src)
            continue
        m.uid = next_uid[0]
        next_uid[0] += 1
        node = curves.ensure_map_curve(
            obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name(LEVEL_MAP_KIND, m.uid))
        if src in old_nodes:
            curves.copy_curve(old_nodes[src], node)
        else:
            curves.set_points(node, _curve_points(pm.curve), 'VECTOR')


def _level_masks_to_object(lv, entries) -> None:
    for e in entries:
        f = lv.filters.add()
        _set_channel_ref(f, e.channel)
        f.lo, f.hi, f.invert, f.op, f.mute = e.lo, e.hi, e.invert, e.op, e.mute
        if e.modulo >= 2:
            f.math_op, f.math_value = 'MODULO', e.modulo


_LEVEL_KEPT = ("material", "color", "color_authored", "rig_bone_angle", "fork")
"""A branching level's settings that live only on the panel (no `BranchLevel`
field): kept by uid across `_levels_to_object`."""


def _levels_to_object(s, levels) -> None:
    """Rebuild the panel's branching stack from a profile's `levels`. The
    phyllotaxis dropdown is written last and the numbers after it, so a named
    pattern's preset write cannot overwrite hand-set values (a CUSTOM level
    keeps its own numbers, a named one gets them back identical). Show is
    the view's, not the species': each place keeps its own.

    A level's Blender-side settings -- its Material, swatch colour, Bone Angle
    and Fork toggle (`_LEVEL_KEPT`), none of them in a `BranchLevel` -- are
    kept by uid, so a tree edit (adding, removing, moving a level) leaves every
    other level as it was: they used to reset to their defaults, which cleared
    every level's material on Add Stems Level."""
    obj = s.id_data
    shown = [lv.show for lv in s.branch_levels]
    old_uids = [lv.uid for lv in s.branch_levels]
    expanded = {lv.uid: lv.expanded for lv in s.branch_levels}
    kept = {lv.uid: {p: tuple(getattr(lv, p)) if p == "color" else getattr(lv, p)
                     for p in _LEVEL_KEPT} for lv in s.branch_levels}
    old_nodes = {}
    for old in s.branch_levels:
        for m in old.maps:
            node = curves.get_component_curve(
                obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name(LEVEL_MAP_KIND, m.uid))
            if node is not None:
                old_nodes[m.uid] = node
    used: set = set()
    next_uid = [max(old_nodes, default=0) + 1]
    # A level keeps its uid (components link to it by uid): one read from this
    # object carries it; one from a preset takes the uid of the level it
    # replaces at the same place; a new one (an added copy) gets a fresh one.
    taken: set = set()
    next_level = max(old_uids, default=0) + 1
    parents = _resolved_parents(levels)
    defaults = _default_level_names(parents)
    s.branch_levels.clear()
    for i, level in enumerate(levels):
        lv = s.branch_levels.add()
        lv.name = getattr(level, "name", "") or defaults[i]
        lv.show = shown[i] if i < len(shown) else True
        uid = getattr(level, "_uid", None) or (old_uids[i] if i < len(old_uids) else 0)
        if uid <= 0 or uid in taken:
            uid, next_level = next_level, next_level + 1
        taken.add(uid)
        lv.uid = uid
        lv.expanded = expanded.get(uid, True)
        for prop, value in kept.get(uid, {}).items():
            setattr(lv, prop, value)
        lv.phyllotaxis = level.phyllotaxis
        for prop, fld in _LEVEL_FIELDS:
            if prop in _LEVEL_SCALE:
                setattr(lv, prop, getattr(level, fld) / _LEVEL_SCALE[prop])
            elif prop != "phyllotaxis":
                setattr(lv, prop, getattr(level, fld))
        if getattr(level, "stiffness", 0.0) > 0.0:
            lv.stiffness = _factor_from_stiffness(level.stiffness)
            lv.stiffness_authored = True
        else:
            lv.stiffness_authored = False            # seeded from the plant's at regrow
        # the value writes above mark a named pattern Custom -- restore it
        lv["phyllotaxis"] = lv.bl_rna.properties["phyllotaxis"].enum_items[
            level.phyllotaxis].value
        _level_masks_to_object(lv, getattr(level, "masks", None) or [])
        _level_maps_to_object(obj, lv, getattr(level, "maps", None) or [], old_nodes, used,
                              next_uid)
    for uid, node in old_nodes.items():
        if uid not in used:
            obj.plantgen.curve_group.nodes.remove(node)
    # the tree, by uid (0: a stems level)
    for lv, p in zip(s.branch_levels, parents, strict=True):
        lv.parent_uid = s.branch_levels[p].uid if p >= 0 else 0
    # A placement on a level that is gone goes with it.
    for k in reversed(range(len(s.placements))):
        if s.placements[k].level_uid and s.placements[k].level_uid not in taken:
            s.placements.remove(k)
    s.branch_levels_active = min(s.branch_levels_active, max(len(s.branch_levels) - 1, 0))
    s["branch_levels_schema"] = _LEVELS_SCHEMA
    structure.rebuild_tree(s)


# Channels a growth map or level mask can read while the plant grows
# (`growth.maps`), custom attributes and classes included
# (`growth.signals`); the rest of the mask list (bloom, apex, branch index)
# only exists on the finished plant, and a growth map on one is a no-op.
GROWTH_MAP_CHANNELS = frozenset(growth_maps_lib.GROWTH_CHANNELS)


def growth_readable(key: str) -> bool:
    """Whether growth can read this channel key."""
    return key in GROWTH_MAP_CHANNELS or growth_signals.is_derived(key)


def _growth_maps(obj, s) -> list:
    """The plant's Growth Maps panel list as `growth.maps.ParamMap`s: each map's
    own curve node (`comp:0:gmap<uid>`), Driver, Amount, on/off. A map whose curve
    node is missing (before the first regrow creates it) is left out."""
    out = []
    for m in s.growth_maps:
        node = curves.get_component_curve(
            obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name("gmap", m.uid))
        if node is None:
            continue
        out.append(growth_maps_lib.ParamMap(
            target=m.target, channel=channel_key(m), curve=curves.to_curve(node),
            amount=float(m.amount), enabled=bool(m.enabled), seed=int(m.seed)))
    return out


def _flower_maps(obj, comp) -> list:
    """A flower component's inflorescence maps as `growth.maps.ParamMap`s (its
    curves `fmap<uid>` under the component; one not yet created is left out)."""
    out = []
    for m in comp.growth_maps:
        node = curves.get_component_curve(obj, comp.uid, curves.map_curve_name("fmap", m.uid))
        if node is None:
            continue
        out.append(growth_maps_lib.ParamMap(
            target=m.target, channel=channel_key(m), curve=curves.to_curve(node),
            amount=float(m.amount), enabled=bool(m.enabled), seed=int(m.seed)))
    return out



def apply_profile(obj, profile) -> None:
    """Seed every panel control and curve widget from `profile`, so the panel
    shows what the base actually is rather than silently disagreeing with it.

    The reflection half of applying a base (a species or a preset); the caller
    sets the embedded stash with `stash_profile` and regrows. Only the
    panel-mapped subset is reflected -- the hidden traits stay in the base and
    reach growth through `_base_profile`, which is the whole point of a
    self-contained stash.
    """
    s = obj.plantgen
    for prop, path, *converters in _FIELDS:
        to_panel = converters[1] if len(converters) > 1 else None
        value = _dotted_get(profile, path)
        setattr(s, prop, to_panel(value) if to_panel else value)
    curves.seed_from_profile(obj, profile)
    # Reflect the base's calendar into the authored window properties and mark
    # them authoritative, so from here the plant's phenology is driven by these
    # per-object values (which mirror the applied calendar exactly -- identical
    # plant) and the calendar widget can edit them.
    _calendar_to_object(s, profile.calendar)
    s.calendar_authored = True
    # The flower component owns the inflorescence; applying a base sets the plant
    # flower props above, so un-author flower components to re-seed them from the
    # freshly applied values at the next regrow (else a stale component would
    # mask the preset). Removed with the plant flower props in the port's phase 4.
    for c in s.components:
        if c.type == 'flower':
            c.flower_authored = False
        elif c.type == 'fruit':
            c.fruit_authored = False
    _growth_maps_to_object(obj, getattr(profile, "maps", None) or [])
    _levels_to_object(s, profile.levels)
    leaf = primary_leaf_component(s)
    if leaf is not None:
        _leaf_arrangement_to(leaf, profile.leaf_arrangement)
        leaf.leaf_weight = float(profile.leaf_weight)
        leaf.leaf_weight_authored = True


def _growth_maps_to_object(obj, maps) -> None:
    """The reflection half of growth maps: rebuild the Growth Maps list (and each
    map's curve node) from a profile's `maps`, so a preset carrying maps shows
    them -- and keeps them, since `profile_from_object` reads the list back."""
    s = obj.plantgen
    for m in s.growth_maps:
        curves.remove_named_component_curve(
            obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name("gmap", m.uid))
    s.growth_maps.clear()
    for i, pm in enumerate(maps, start=1):
        m = s.growth_maps.add()
        m.uid = i
        m.target = pm.target
        if pm.channel.startswith(masks.CUSTOM_PREFIX):
            m.channel, m.attr_name = "custom", pm.channel[len(masks.CUSTOM_PREFIX):]
        elif pm.channel.startswith(masks.CLASS_PREFIX):
            m.channel, m.class_name = "class", pm.channel[len(masks.CLASS_PREFIX):]
        else:
            m.channel = pm.channel
        m.amount = float(pm.amount)
        m.enabled = bool(pm.enabled)
        node = curves.ensure_map_curve(
            obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name("gmap", m.uid))
        curves.set_points(node, _curve_points(pm.curve), 'VECTOR')
    s.growth_maps_active = 0


def stash_profile(obj, name, profile, description="", preset_id="") -> None:
    """Embed `profile` on the object as its self-contained base, and remember
    which preset it came from. Stored as the full file envelope (not a bare
    profile dict) so an old stash migrates forward on load exactly like a preset
    file does."""
    s = obj.plantgen
    s.preset_name = name or obj.name
    s.preset_embedded = serialize.to_json(
        name or obj.name, profile, description, preset_id=preset_id)
    s.preset_id = preset_id


def ensure_migrated(obj) -> bool:
    """Give a pre-preset-system plant a self-contained base from its legacy
    `species` value, without changing how it looks. Returns whether it migrated.

    The `species` dropdown is retired; a plant made before that had its base by a
    live name lookup on `species`. This stamps that same species profile as the
    plant's embedded base and names it, leaving **every panel value and curve
    untouched** -- so the resolved profile is byte-identical (`_base_profile`
    already fell back to the species for a stash-less plant), the plant looks the
    same, and it is merely self-contained now. Re-stamps the growth signature,
    which the embedded base is part of, so an undo does not read it as stale.
    """
    s = obj.plantgen
    if s.preset_embedded:
        return False
    key = getattr(s, "species", "") or "GENERIC"
    profile = species_lib.profile(key)
    try:
        label = s.bl_rna.properties["species"].enum_items[key].name
    except (KeyError, AttributeError):
        label = key
    stash_profile(obj, label, profile, preset_id=key)
    stamp(obj)
    return True


def ensure_day_migrated(obj) -> bool:
    """One-time: derive a pre-calendar plant's `day` from its legacy `age`.

    Day now drives growth; a plant made before that
    carried its state on the Age dial. ``day = emergence_day + age *
    season_length_days`` is the inverse of `progress_for_day` under the default
    identity growth curve, so an Age-1.0 plant lands on day 270 -- the same mature
    plant, geometry unchanged. Marked done so a later Day scrub is never
    overwritten by a re-migration (Age goes stale the moment Day takes over).
    """
    s = obj.plantgen
    if s.calendar_migrated:
        return False
    s.day = s.emergence_day + s.age * s.season_length_days
    s.calendar_migrated = True
    return True


def _write_point_attribute(mesh, name: str, dtype: str, values: list) -> None:
    attr = mesh.attributes.new(name=name, type=dtype, domain='POINT')
    if len(values) == 0:  # (a list or a numpy array -- no truthiness test)
        return
    if dtype == 'FLOAT_VECTOR':
        attr.data.foreach_set("vector", [c for v in values for c in v])
    elif dtype == 'FLOAT_COLOR':
        attr.data.foreach_set("color", [c for v in values for c in v])
    else:
        attr.data.foreach_set("value", values)


def _section_spec(stem_mesher, level):
    """A branching level's cross-section as the mesher's `SectionSpec`. A
    polygon's side count is its corner count, not a resolution: the mesher
    traces it at whatever sides a tube's thickness asks for (up to the round
    section's 32), corners on vertices."""
    pin = level.section_inputs()
    lobed = int(pin["lobe_count"]) > 0
    polygon = int(pin["sides"]) if (not lobed and int(pin["sides"]) < 32) else 0
    return stem_mesher.SectionSpec(
        sides=32 if polygon else int(pin["sides"]), lobe_count=int(pin["lobe_count"]),
        lobe_depth=float(pin["lobe_depth"]), flatten=float(pin["flatten"]),
        polygon=polygon)


WELD_NOW = "now"
"""`regrow(weld=...)`: a weld not cached is built before returning, waiting --
what anything reading the finished plant needs (bakes, timelapse frames, a
script that measures or exports)."""
WELD_BACKGROUND = "background"
"""`regrow(weld=...)`: a worker process builds it while the preview shows, and
it lands by itself when done (the Weld button)."""
WELD_LATER = "later"
"""`regrow(weld=...)`: it is not built -- the unwelded preview shows until the
Weld button (every interactive edit: live update, the panel's buttons,
presets, undo; the default)."""


def _write_built_stems(obj, mesh, verts, edges, stem_attrs, stem_spec, profile,
                       hidden_tubes=frozenset(), weld: str = WELD_NOW) -> bool:
    """Weld Method = Built: the skeleton swept into one quad mesh with every
    junction built as a hole + fillet rings (`geometry.stem_mesher`), written in
    place of the skeleton with the same point attributes, `pg_u` / `pg_v` and a
    real UVMap. The count of junctions that fell back to a capped tube is kept
    on the object for the panel. Returns whether the mesh was written.

    The welding is nearly all of it (18.8 s of 19.6 s on a 460-tube plant), so
    it is a step of its own: a weld already cached is always written, and one
    that is not is built as `weld` says (`WELD_NOW` / `WELD_BACKGROUND` /
    `WELD_LATER`) -- the last two writing the bare skeleton meanwhile, which
    the node graph sweeps into plain unwelded tubes. With Weld Junctions off
    there is no weld to wait for: the capped tubes are built here whatever
    `weld` says (0.8 s at 460 tubes). A worker already building this very
    input is kept, and waited on rather than started over when it is needed
    now; one building anything else is killed."""
    from ..geometry import stem_mesher

    s = obj.plantgen
    section = {i: _section_spec(stem_mesher, level)
               for i, level in enumerate(profile.levels)} or stem_mesher.SectionSpec()
    names = [name for name, _ in stem_spec]
    attrs = {name: stem_attrs[name] for name in names}
    kw = dict(smoothness=float(s.stem_smoothness), uv_size=float(s.uv_size),
              twist=float(s.stem_twist), fillet=float(s.stem_fillet), bark=float(s.bark_noise),
              weld=bool(s.stem_weld), hidden=frozenset(hidden_tubes),
              light_stalks=bool(s.light_stalks))
    key = _stem_mesh_key(stem_mesher, verts, edges, attrs, section, kw)
    uid = obj.session_uid
    out = _cached_stem_mesh(stem_mesher, uid, key)
    job = _STEM_JOBS.get(uid)
    if job is not None and (out is not None or job.key != key):
        job.cancel()                        # building a weld the plant has moved past
        del _STEM_JOBS[uid]
        job = None
    if out is None and kw["weld"] and weld != WELD_NOW:
        if job is None and weld == WELD_BACKGROUND:
            job = _StemJob.start(stem_mesher, key, stem_spec, verts, edges, attrs, section, kw)
            if job is not None:
                _STEM_JOBS[uid] = job
                if not bpy.app.timers.is_registered(_poll_stem_jobs):
                    bpy.app.timers.register(_poll_stem_jobs, first_interval=_JOB_POLL_SECONDS)
        if job is not None or weld == WELD_LATER:
            _write_skeleton_preview(obj, mesh, verts, edges, stem_attrs, stem_spec,
                                    hidden_tubes, key)
            return True
    if out is None and job is not None:     # this very weld is already building
        out = job.wait(stem_mesher)
        _STEM_JOBS.pop(uid, None)
    if out is None:
        out = stem_mesher.build_stems(verts, edges, attrs, section=section, **kw)
        out.debug = {}                      # the junction fields: tests' only, and large
        _disk_save(key, out)
    _remember_stem_mesh(uid, key, out)
    return _write_mesh_out(obj, mesh, out, key, stem_spec)


def _write_mesh_out(obj, mesh, out, key: bytes, stem_spec) -> bool:
    """Write a built stem mesh into the plant's mesh -- unless it is already
    there (the same mesher input, the mesh not touched since): then it is left,
    and the node graph is not re-run for a stem that did not change. Returns
    whether it wrote."""
    obj.pop(_STEM_PREVIEW_KEY, None)
    obj["plantgen_weld"] = [int(out.junctions), int(out.fallbacks)]
    written = str(obj.get(_STEM_WRITTEN_KEY, "")).split(":")
    if (written[0] == key.hex() and len(written) == 2
            and len(mesh.vertices) == len(out.positions)
            and len(mesh.polygons) == len(out.face_sizes)
            and written[1] == _positions_digest(mesh_positions(mesh))):
        return False
    mesh.clear_geometry()
    # Written straight from the arrays: `from_pydata` round-trips Python lists
    # and runs a shade_flat pass (a third of the write for a 150k-vertex stem).
    # Polygons added this way carry no `sharp_face`, i.e. they are smooth.
    mesh.vertices.add(len(out.positions))
    mesh.vertices.foreach_set("co", out.positions.astype(np.float32).ravel())
    mesh.loops.add(len(out.face_loops))
    mesh.loops.foreach_set("vertex_index", out.face_loops.astype(np.int32))
    mesh.polygons.add(len(out.face_sizes))
    starts = np.concatenate(([0], np.cumsum(out.face_sizes)[:-1])).astype(np.int32)
    mesh.polygons.foreach_set("loop_start", starts)
    mesh.update(calc_edges=True)
    for name, dtype in tuple(stem_spec) + (("pg_u", 'FLOAT'), ("pg_v", 'FLOAT')):
        _write_point_attribute(mesh, name, dtype, out.point_attrs[name])
    uv = mesh.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", out.corner_uv.astype(np.float32).ravel())
    obj[_STEM_WRITTEN_KEY] = key.hex() + ":" + _positions_digest(out.positions.astype(np.float32))
    return True


def _write_skeleton_preview(obj, mesh, verts, edges, stem_attrs, stem_spec, hidden_tubes,
                            key: bytes) -> None:
    """While a worker builds the welded mesh: the bare skeleton -- every vertex
    and attribute the mesher was handed, less a hidden level's tubes -- which
    the node graph sweeps into plain, unwelded tubes (`mesh.ensure_gn_modifier`
    reads `_STEM_PREVIEW_KEY`). ~3 ms to write and ~50 ms to sweep, against
    seconds for the weld."""
    gone = {i for i, t in enumerate(stem_attrs["tube"]) if t in hidden_tubes}
    if gone:
        verts, edges, stem_attrs = _drop_stem_verts(verts, edges, dict(stem_attrs), gone)
    mesh.clear_geometry()
    mesh.from_pydata(verts, edges, [])
    _write_attributes(mesh, stem_attrs, stem_spec)
    obj.pop(_STEM_WRITTEN_KEY, None)
    obj[_STEM_PREVIEW_KEY] = key.hex()


_STEM_WRITTEN_KEY = "plantgen_stem_written"
"""On the plant: "mesher input key:digest of the positions written" for the
stem mesh last written, so an unchanged one is not rewritten (`_write_mesh_out`)."""

_STEM_PREVIEW_KEY = "plantgen_stem_preview"
"""On the plant while it shows the unwelded preview: the mesher input key of
the weld it lacks. A worker's weld lands only on a plant still lacking that
very key, so an edit or an undo since (which changes or restores this with
everything else) turns it away. The node graph sweeps the skeleton while it
is set (`mesh.ensure_gn_modifier`)."""

_STEM_ARRAYS_CACHE: dict = {}
"""obj session_uid -> (grown plant, arguments, `_stem_arrays` result): the
skeleton arrays again while the grow cache hands back the same plant object."""


def mesh_positions(mesh) -> np.ndarray:
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _positions_digest(positions: np.ndarray) -> str:
    import hashlib

    return hashlib.blake2b(np.ascontiguousarray(positions, dtype=np.float32).tobytes(),
                           digest_size=16).hexdigest()


_STEM_MESH_CACHE: dict = {}
"""obj session_uid -> [(input key, stem_mesher output), ...], most recent first,
as deep as the grow cache. **A regrow whose skeleton is unchanged reuses the stem
mesh** -- the mesher is nearly all of a regrow on a flowering plant (24.7 s of
25 s measured: ~380 pedicel welds) and an edit that touches no stem (leaves,
flower assets, colours, masks, maps) hands it identical input. Keyed on the
input itself (positions, edges, every attribute, sections, settings and the
mesher's own switches), so undo and A/B flicks hit too."""


def _cached_stem_mesh(stem_mesher, uid, key: bytes):
    """The stem mesh for this input from memory (made the most recent) or from
    disk, or None."""
    entries = _STEM_MESH_CACHE.setdefault(uid, [])
    for i, (k, out) in enumerate(entries):
        if k == key:
            entries.insert(0, entries.pop(i))
            return out
    out = _disk_load(stem_mesher, key)
    if out is not None:
        _remember_stem_mesh(uid, key, out)
    return out


def _remember_stem_mesh(uid, key: bytes, out) -> None:
    entries = _STEM_MESH_CACHE.setdefault(uid, [])
    entries[:] = [(k, o) for k, o in entries if k != key]
    entries.insert(0, (key, out))
    del entries[_STEM_CACHE_DEPTH:]


def _stem_mesh_key(stem_mesher, verts, edges, attrs, section, kw) -> bytes:
    """A digest of everything the stem mesher reads."""
    import hashlib

    h = hashlib.blake2b(digest_size=20)
    h.update(np.ascontiguousarray(np.asarray(verts, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(edges, dtype=np.int64)).tobytes())
    for name in sorted(attrs):
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(attrs[name])).tobytes())
    h.update(repr(section if not isinstance(section, dict)
                  else sorted(section.items())).encode())
    h.update(repr(sorted((k, v if not isinstance(v, frozenset) else sorted(v))
                         for k, v in kw.items())).encode())
    # the mesher's module switches (a reload or a script may change them) and
    # its source, so a cached mesh never outlives the code that made it
    h.update(repr(sorted(_mesher_switches(stem_mesher).items())).encode())
    h.update(_source_digest(stem_mesher))
    return h.digest()


def _mesher_switches(stem_mesher) -> dict:
    """The mesher's module switches (its `_UPPER` names of plain values): part
    of its input, since a reload or a script may change them -- and a worker
    process, which loads the file afresh, is handed them."""
    return {n: v for n, v in vars(stem_mesher).items()
            if n.startswith("_") and n[1:2].isupper()
            and isinstance(v, (bool, int, float, str, tuple))}


def _source_digest(module) -> bytes:
    import hashlib

    got = _SOURCE_DIGESTS.get(id(module))
    if got is None:
        try:
            with open(module.__file__, "rb") as fh:
                got = hashlib.blake2b(fh.read(), digest_size=16).digest()
        except OSError:
            got = b"?"
        _SOURCE_DIGESTS[id(module)] = got
    return got


_SOURCE_DIGESTS: dict = {}

_STEM_CACHE_DEPTH = 5
"""Stem meshes kept in memory per plant: the current one plus the last four."""

_DISK_CACHE_BYTES = 1 << 30
"""Most the on-disk stem mesh cache keeps (oldest dropped first)."""


def _cache_dir(*parts):
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "plantgen_cache", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _disk_cache_dir():
    return _cache_dir("stem_mesh")


def _disk_load(stem_mesher, key: bytes):
    """A stem mesh from the on-disk cache (survives a reload or a reopen, and
    is where a worker leaves its mesh), or None. Any failure is a miss."""
    import os

    from ..geometry import stem_worker

    try:
        path = os.path.join(_disk_cache_dir(), key.hex() + ".npz")
        if not os.path.exists(path):
            return None
        out = stem_worker.load(stem_mesher, path)
        os.utime(path)                      # recently used: kept by the trim
        return out
    except Exception:  # noqa: BLE001 -- a cache: never worse than a miss
        return None


def _disk_save(key: bytes, out) -> None:
    """Store a stem mesh on disk and trim the cache to `_DISK_CACHE_BYTES`."""
    import os

    from ..geometry import stem_worker

    try:
        stem_worker.save(os.path.join(_disk_cache_dir(), key.hex() + ".npz"), out)
    except Exception:  # noqa: BLE001 -- a cache: failing to store costs nothing
        return
    _disk_trim()


def _disk_trim() -> None:
    """Drop the least recently used stem meshes past `_DISK_CACHE_BYTES`
    (never a worker's file still being written)."""
    import os

    try:
        folder = _disk_cache_dir()
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.endswith(".npz") and ".tmp." not in f]
        files.sort(key=os.path.getmtime, reverse=True)
        total = 0
        for f in files:
            total += os.path.getsize(f)
            if total > _DISK_CACHE_BYTES:
                os.remove(f)
    except Exception:  # noqa: BLE001 -- a cache: failing to trim costs nothing
        pass


# -- Welds off Blender's thread ------------------------------------------------

class _StemJob:
    """One plant's weld building in a worker process (`geometry.stem_worker`)
    for the Weld button: started by `_write_built_stems`, landed by
    `_poll_stem_jobs`, and killed the moment an edit or an undo moves the
    plant past its input. The worker writes its mesh straight into the on-disk
    cache, where any later regrow of the same input finds it."""

    def __init__(self, key: bytes, stem_spec, proc, out_path: str, files: tuple):
        self.key, self.stem_spec, self.proc = key, tuple(stem_spec), proc
        self.out_path, self.files = out_path, files

    @classmethod
    def start(cls, stem_mesher, key: bytes, stem_spec, verts, edges, attrs, section, kw):
        """Start the worker; None when it cannot be (the caller then builds in
        Blender, as before there were workers)."""
        import os
        import pickle
        import subprocess
        import sys

        from ..geometry import stem_worker

        try:
            folder = _cache_dir("jobs")
            name = f"{key.hex()}.{os.getpid()}.{next(_JOB_SERIAL)}"
            job_path = os.path.join(folder, name + ".pkl")
            log_path = os.path.join(folder, name + ".log")
            with open(job_path, "wb") as fh:
                pickle.dump(stem_worker.pack(verts, edges, attrs, section, kw,
                                             _mesher_switches(stem_mesher)),
                            fh, protocol=pickle.HIGHEST_PROTOCOL)
            out_path = os.path.join(_disk_cache_dir(), key.hex() + ".npz")
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(
                    [sys.executable, stem_worker.__file__, job_path, out_path,
                     _source_digest(stem_mesher).hex(), str(os.getpid())],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:  # noqa: BLE001 -- no worker: the mesh is built in Blender
            import traceback
            print("PlantGen: could not start a stem worker; building in Blender")
            traceback.print_exc()
            return None
        return cls(key, stem_spec, proc, out_path, (job_path, log_path))

    def finished(self) -> bool:
        return self.proc.poll() is not None

    def wait(self, stem_mesher):
        """Wait for the worker; its mesh, or None if it failed."""
        try:
            self.proc.wait()
        except Exception:  # noqa: BLE001
            self.cancel()
            return None
        return self.collect(stem_mesher)

    def collect(self, stem_mesher):
        """The finished worker's mesh, or None when it failed (its log printed)."""
        from ..geometry import stem_worker

        out = None
        if self.proc.returncode == 0:
            try:
                out = stem_worker.load(stem_mesher, self.out_path)
            except Exception:  # noqa: BLE001 -- reported below
                out = None
        if out is None:
            print(f"PlantGen: stem worker failed (exit {self.proc.returncode}):\n"
                  + self._log_tail())
        self._cleanup()
        return out

    def cancel(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 -- it is going either way
                pass
        self._cleanup()

    def _log_tail(self) -> str:
        try:
            with open(self.files[1], "rb") as fh:
                return fh.read()[-2000:].decode("utf-8", "replace")
        except OSError:
            return ""

    def _cleanup(self) -> None:
        import os

        leftovers = (*self.files, f"{self.out_path[:-4]}.{self.proc.pid}.tmp.npz")
        for path in leftovers:
            try:
                os.remove(path)
            except OSError:
                pass


_STEM_JOBS: dict = {}
"""obj session_uid -> its running `_StemJob` (one a plant at most)."""

_JOB_SERIAL = itertools.count()

_JOB_POLL_SECONDS = 0.1


def _poll_stem_jobs():
    """Timer: land each finished worker's mesh on its plant. Never lets an
    exception out -- a timer that raises is silently unregistered."""
    try:
        _land_stem_jobs()
    except Exception:  # noqa: BLE001 -- see docstring
        import traceback
        traceback.print_exc()
    return _JOB_POLL_SECONDS if _STEM_JOBS else None


def _land_stem_jobs() -> None:
    """Write each finished worker's mesh over its plant's preview -- when the
    plant still waits for exactly that input (`_STEM_PREVIEW_KEY`); a mesh no
    plant wants any more is only cached."""
    from ..geometry import stem_mesher

    for uid, job in list(_STEM_JOBS.items()):
        if _STEM_JOBS.get(uid) is not job or not job.finished():
            continue                        # (still building, or replaced meanwhile)
        del _STEM_JOBS[uid]
        out = job.collect(stem_mesher)
        obj = _object_by_uid(uid)
        if obj is None:
            continue
        if out is not None:
            _remember_stem_mesh(uid, job.key, out)
        if obj.get(_STEM_PREVIEW_KEY) != job.key.hex():
            continue
        if out is None:
            regrow(obj, weld=WELD_NOW)      # the worker failed: build it here instead
            continue
        if _write_mesh_out(obj, obj.data, out, job.key, job.stem_spec):
            obj.data.update()
        gn_mesh.push_stem_mode(obj)
        _disk_trim()
        from . import live

        live.log_event("stems-landed", obj)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()


def _object_by_uid(uid):
    return next((o for o in bpy.data.objects if o.session_uid == uid), None)


def weld_state(obj) -> str:
    """Where the plant's stem weld stands, for the panel and the Weld button:
    'OFF' (Weld Junctions off -- nothing to weld), 'WELDING' (a worker is
    building it), 'UNWELDED' (the preview: junctions not joined), 'WELDED', or
    'NONE' (no stems at all)."""
    if not obj.plantgen.stem_weld:
        return 'OFF'
    want = obj.get(_STEM_PREVIEW_KEY)
    if want:
        job = _STEM_JOBS.get(obj.session_uid)
        return 'WELDING' if job is not None and job.key.hex() == want else 'UNWELDED'
    mesh = obj.data
    if len(mesh.polygons):          # only the built stem mesh has faces
        return 'WELDED'
    return 'UNWELDED' if len(mesh.vertices) else 'NONE'


def cancel_stem_jobs() -> None:
    """Kill every worker (the add-on unregistering, another file loading)."""
    for job in _STEM_JOBS.values():
        job.cancel()
    _STEM_JOBS.clear()
    if bpy.app.timers.is_registered(_poll_stem_jobs):
        bpy.app.timers.unregister(_poll_stem_jobs)
    _LIVE.clear()                           # (the compute worker itself stays up)
    if bpy.app.timers.is_registered(_poll_live):
        bpy.app.timers.unregister(_poll_live)


# -- Growth and leaf collision off Blender's thread (live edits) ---------------
#
# A live edit's regrow ran on Blender's main thread, freezing it: on a plant
# with leaf collision mostly the collision resolve, on a big one the growth.
# `regrow_live` hands both to one long-lived worker process
# (`compute_worker`) and lands each result in the cache the regrow reads -- a
# grown plant in `_GROW_CACHE` under its grow key, a collision outcome in
# `_SETTLE_CACHE` under its input digest -- then regrows, which finds both and
# only writes the plant. Same code on the same input: the plant is the one the
# regrow would have made itself. Explicit regrows (buttons, presets, undo,
# bakes, timelapse frames) stay as they were.

_LIVE: dict = {}
"""obj session_uid -> its live pipeline: {"stage": "grow" / "collide", "job",
"key", "sig" (`growth_signature` when it started), "restart"}."""

_WORKER: dict = {"proc": None, "digest": None, "log": None}

_WORKER_SERIAL = itertools.count()

_LIVE_POLL_SECONDS = 0.05

_PROBING: set = set()
"""Plants whose `_organ_points` is a probe: a collision cache miss raises
`_CollisionNeeded` with the solver's input instead of resolving."""


class _CollisionNeeded(Exception):
    def __init__(self, key, leaves, shapes, stems):
        super().__init__("leaf collision not cached")
        self.key, self.leaves, self.shapes, self.stems = key, leaves, shapes, stems


def _worker_digest() -> str:
    """The digest of the code this Blender session runs (computed once)."""
    if _WORKER["digest"] is None:
        import os

        from .. import compute_worker
        _WORKER["digest"] = compute_worker.code_digest(
            os.path.dirname(os.path.abspath(compute_worker.__file__)))
    return _WORKER["digest"]


def _compute_worker():
    """The running worker process, started on first use; None when it cannot
    be (the caller then regrows in Blender, as before)."""
    proc = _WORKER["proc"]
    if proc is not None and proc.poll() is None:
        return proc
    import os
    import subprocess
    import sys

    from .. import compute_worker
    try:
        folder = _cache_dir("compute")
        log = open(os.path.join(folder, f"worker.{os.getpid()}.log"), "wb")  # noqa: SIM115
        proc = subprocess.Popen(
            [sys.executable, compute_worker.__file__, folder, _worker_digest(),
             str(os.getpid())],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001 -- no worker: live edits regrow in Blender
        import traceback
        print("PlantGen: could not start the compute worker; regrowing in Blender")
        traceback.print_exc()
        return None
    _WORKER["proc"], _WORKER["log"] = proc, log
    return proc


def stop_compute_worker() -> None:
    """Stop the worker (the add-on unregistering)."""
    proc, log = _WORKER["proc"], _WORKER["log"]
    _WORKER["proc"] = _WORKER["log"] = None
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 -- it is going either way
            pass
    if log is not None:
        log.close()


def _submit(kind: str, payload):
    """Queue a job; its id, or None when there is no worker."""
    import os
    import pickle

    proc = _compute_worker()
    if proc is None:
        return None
    job_id = f"{os.getpid()}_{next(_WORKER_SERIAL)}"
    path = os.path.join(_cache_dir("compute"), job_id + ".job")
    try:
        with open(path + ".tmp", "wb") as fh:
            pickle.dump((kind, payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)
        proc.stdin.write((job_id + "\n").encode("ascii"))
        proc.stdin.flush()
    except Exception:  # noqa: BLE001 -- a broken worker: regrow in Blender
        import traceback
        traceback.print_exc()
        stop_compute_worker()
        return None
    if not bpy.app.timers.is_registered(_poll_live):
        bpy.app.timers.register(_poll_live, first_interval=_LIVE_POLL_SECONDS)
    return job_id


def _collect(job_id: str):
    """('pending', None), ('done', result) or ('failed', traceback text)."""
    import os
    import pickle

    base = os.path.join(_cache_dir("compute"), job_id)
    for ext, state in ((".out", "done"), (".err", "failed")):
        path = base + ext
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if state == "done":
                return state, pickle.loads(data)
            return state, data.decode("utf-8", "replace")
    proc = _WORKER["proc"]
    if proc is None or proc.poll() is not None:
        return "failed", "the compute worker stopped"
    return "pending", None


def _wants_collision(obj) -> bool:
    return any(c.collide and not c.mute and c.type in ("leaf", "bract")
               for c in obj.plantgen.components)


def regrow_state(obj) -> str | None:
    """Where a live update of this plant is: 'QUEUED' (an edit waiting out live
    update's debounce), 'GROWING' (its growth in the compute worker), 'SETTLING'
    (its leaf collision there), or None -- nothing pending. For the indicators
    (the panel's status line, `regrow_report`); a synchronous regrow holds
    Blender while it runs, so it never shows here."""
    try:
        st = _LIVE.get(obj.session_uid)
        if st is not None:
            return 'SETTLING' if st["stage"] == "collide" else 'GROWING'
        from . import live
        if obj.name in live._dirty and obj.plantgen.live_update:
            return 'QUEUED'
    except (ReferenceError, AttributeError):
        pass
    return None


def regrow_live(obj) -> None:
    """Live update's regrow: the same plant `regrow` makes, its growth and leaf
    collision computed off Blender's thread (see above). Falls back to
    `regrow` whenever a worker cannot do it (none, or under Grow Over Timeline,
    whose leaf reference and frames stay in Blender)."""
    uid = obj.session_uid
    if uid in _LIVE:
        _LIVE[uid]["restart"] = True        # one in flight: start over when it lands
        return
    if not _live_start(obj):
        regrow(obj)


def _live_start(obj) -> bool:
    """Start a plant's pipeline; False when the worker cannot take it."""
    s = obj.plantgen
    if s.animate_day:
        return False
    # What `regrow` runs first: settings the profile is read from.
    _prepare_component_curves(obj, s)
    _seed_flower_components(obj, s)
    _seed_leaf_components(s)
    _seed_level_stiffness(s)
    _seed_bud_components(obj, s)
    _seed_fruit_components(obj, s)
    day = float(s.day)
    profile = profile_from_object(obj)
    key = _grow_key(obj, profile, day)
    if any(k == key for k, *_ in _GROW_CACHE.get(obj.name, ())):
        return _live_collide(obj)
    environment = env_bridge.build_environment(obj, profile)
    job = _submit("grow", (profile, int(s.seed), day, environment,
                           growth_signals_for(obj, profile)))
    if job is None:
        return False
    _LIVE[obj.session_uid] = {"stage": "grow", "job": job, "key": key, "profile": profile,
                              "environment": environment, "sig": growth_signature(obj),
                              "restart": False, "t0": time.perf_counter()}
    return True


def _live_collide(obj) -> bool:
    """With the plant grown: hand its leaf collision to the worker if the
    resolve is not cached, else finish now."""
    if not _wants_collision(obj):
        _live_finish(obj)
        return True
    _PROBING.add(obj.name)
    try:
        _organ_points(obj, obj.plantgen, float(obj.plantgen.day))
    except _CollisionNeeded as need:
        job = _submit("collide", (need.leaves, need.shapes, need.stems))
        if job is None:
            return False
        _LIVE[obj.session_uid] = {"stage": "collide", "job": job, "key": need.key,
                                  "sig": growth_signature(obj), "restart": False,
                                  "t0": time.perf_counter()}
        return True
    finally:
        _PROBING.discard(obj.name)
    _live_finish(obj)
    return True


def _live_finish(obj) -> None:
    """Both caches hold this plant's results: the regrow only writes it."""
    regrow(obj)
    from . import live

    live.log_event("live-landed", obj)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _poll_live():
    """Timer: land finished worker results. Never lets an exception out -- a
    timer that raises is silently unregistered."""
    try:
        _land_live()
    except Exception:  # noqa: BLE001 -- see docstring
        import traceback
        traceback.print_exc()
    return _LIVE_POLL_SECONDS if _LIVE else None


def _land_live() -> None:
    for uid, st in list(_LIVE.items()):
        state, value = _collect(st["job"])
        if state == "pending":
            continue
        _LIVE.pop(uid, None)
        if "t0" in st:                     # Debug Tools > Timings: the worker's share
            _worker_time(uid, "Growth (background)" if st["stage"] == "grow"
                         else "Leaf collision (background)", time.perf_counter() - st["t0"])
        obj = _object_by_uid(uid)
        if obj is None or not is_plant(obj):
            continue
        if state == "failed":
            print(f"PlantGen: compute worker job failed, regrowing in Blender:\n{value}")
            regrow(obj)
            continue
        moved = st["restart"] or growth_signature(obj) != st["sig"]
        if st["stage"] == "grow":
            if not moved:
                plant = value
                apply_pruning(obj, plant)
                entries = _GROW_CACHE.setdefault(obj.name, [])
                entries.insert(0, (st["key"], plant, st["profile"], st["environment"]))
                del entries[_GROW_CACHE_DEPTH:]
        else:
            petioles._SETTLE_CACHE[obj.name] = (st["key"], value)
        if moved:
            if not _live_start(obj):        # its inputs moved on meanwhile: again
                regrow(obj)
        elif st["stage"] == "grow":
            if not _live_collide(obj):
                regrow(obj)
        else:
            _live_finish(obj)


def _write_attributes(mesh, values: dict, spec) -> None:
    """Write a name -> values dict as point attributes, typed by `spec`."""
    for name, dtype in spec:
        _write_point_attribute(mesh, name, dtype, values[name])


_DERIVED_ATTRS = (
    ("height_fraction", 'FLOAT'),
    ("instance_random", 'FLOAT'),
)
"""Written in `regrow` rather than by either array builder, because both need
the whole plant measured first."""


def _add_height_fraction(stem_attrs: dict, verts: list,
                         organ_attrs: dict, organ_positions: list) -> None:
    """0 at the plant's lowest point, 1 at its highest, over stem and organs
    together. A plant with no height at all reads 0 everywhere rather than
    dividing by zero."""
    zs = [v[2] for v in verts] + [p[2] for p in organ_positions]
    low = min(zs) if zs else 0.0
    span = (max(zs) - low) if zs else 0.0
    if span <= 0.0:
        stem_attrs["height_fraction"] = [0.0] * len(verts)
        organ_attrs["height_fraction"] = [0.0] * len(organ_positions)
        return
    stem_attrs["height_fraction"] = [(v[2] - low) / span for v in verts]
    organ_attrs["height_fraction"] = [(p[2] - low) / span for p in organ_positions]


_STEM_ATTRS = (
    ("width", 'FLOAT'),
    ("order", 'INT'),
    ("birth_step", 'FLOAT'),
    # Debug-legibility attributes. `axis_id` and `node_index` are what make the
    # failure modes the rebuild was written against readable in the viewport at
    # all: a five-node trunk under a forest of laterals looks identical to a
    # healthy plant until you can see which vertices belong to which axis.
    ("axis_id", 'INT'),
    ("node_index", 'INT'),
    # Depth inside the inflorescence, -1 on vegetative stem. The counterpart to
    # `order`, and the whole point of separating them: on a tree the two run to
    # completely different depths, and no other view can show which is which.
    ("structural_depth", 'INT'),
    # The beam solver's four per-node readouts, one gravity debug view each.
    # Display-only: they turn what `physics.apply_gravity` computed at every node
    # -- the realized curvature (deg/m), the load carried beyond it (kg), the
    # local stiffness (Pa) and the bending moment (N*m) -- into something the
    # viewport can shade, so a heavy head folding a thin stem is visible as the
    # load and moment that did it, not only as the resulting shape.
    ("curvature", 'FLOAT'),
    ("load", 'FLOAT'),
    ("stiffness", 'FLOAT'),
    ("moment", 'FLOAT'),
    # WS3 lifecycle: 0 alive, 1 dead. A shed axis emits no vertex at all; a dead
    # (retained) one keeps its geometry and carries this so the render can tint
    # the wood toward the Dead Colour.
    ("dead", 'FLOAT'),
    # Trunk flare weight, 1 at the ground falling to 0 up the flare (0 above it
    # and on every branch): the GN Surface stage shapes the buttress lobes by it.
    ("flare", 'FLOAT'),
    # Which swept tube (one per spline / chain) a vertex belongs to, and the
    # tube it branches from (-1 for a basal axis) -- what the weld reads to
    # know which surface a junction joins. Not axis_id: a pedicel keeps its
    # parent axis's id but is its own tube.
    ("tube", 'INT'),
    ("parent_tube", 'INT'),
)


def _stem_arrays(plant, profile, show_levels=(), show_catkins: bool = True,
                 total_steps: int = 1, show_pedicels: bool = True,
                 hidden_axes=frozenset(), trunk_flare: float = 0.0):
    """Vertices, edges and per-vertex attributes for the stem skeleton.

    A vertex sits at every phytomer's distal node, plus one at each basal
    axis's ground point. Each vertex carries its **own** phytomer's radius, so
    the swept tube passes through the pipe-model radius at every node.

    It used to carry the radius of the internode *above* it instead, which
    renders every node one step thin. That is invisible on a long, gradual axis
    (adjacent radii are nearly equal) but it collapses a short branch: a 3-node
    lateral computed as [1.13, 0.80, 0.30] drew as [0.80, 0.30, 0.30] -- losing
    its own base and reading as a uniform stub -- because a branch has no ground
    vertex of its own to carry that base radius, unlike a basal axis.

    Attributes come back as a name -> values dict rather than a positional
    tuple: the set only grows, and every caller wants to write them in a loop.

    `show_levels` holds each branching level's Show box: an axis on a hidden
    level is still written (its branches start from its nodes), but its tube
    comes back in `hidden_tubes`, which the mesher lays out and does not build.
    Only that level goes: the levels after it, pedicels, catkins and the
    inflorescence (which has no level of its own) stay, unwelded where they
    leave a hidden tube.
    """
    verts, edges = [], []
    attrs = {name: [] for name, _ in _STEM_ATTRS}
    # Extra 0-1 channels the Filter Preview reads so its colours run over the stem
    # too, not just the organs -- so a filter that excludes the main stem shows it
    # red. Not written to the mesh (not in _STEM_ATTRS), consumed straight from
    # this dict by `_preview_values`. Same normalisation as the organ channels
    # (`_organ_mask_context`), so a stem node and the leaf on it read the same value.
    for _extra in (*_STRUCT_CHANNELS, "age"):
        attrs[_extra] = []
    _infl_axes = _inflorescence_axes(plant)
    _max_depth = max((max(a.structural_depth, 0) for a in plant.axes.values()),
                     default=0)
    _depth_norm = float(_max_depth) if _max_depth > 0 else 1.0
    _max_order = max((a.order for a in plant.axes.values()), default=0)
    _order_norm = float(_max_order) if _max_order > 0 else 1.0
    _axis_rank = {aid: i for i, aid in enumerate(sorted(plant.axes))}
    _rank_norm = float(len(_axis_rank) - 1) if len(_axis_rank) > 1 else 1.0
    _axis_node_norm = {aid: (float(len(ax.phytomers) - 1) if len(ax.phytomers) > 1 else 1.0)
                       for aid, ax in plant.axes.items()}
    # organ id -> its stalk's vertices (a catkin's: stalk + bodies), for flowers
    # and catkins: lets `regrow` take one away when the flower component's masks
    # drop it (`_drop_stem_verts`).
    pedicel_verts: dict[int, list[int]] = {}
    hidden_levels = {i for i, shown in enumerate(show_levels) if not shown}
    hidden_tubes: set[int] = set()
    tip_vert: dict[tuple[int, int], int] = {}
    pedicels: list[tuple[int, tuple, float, int]] = []
    # Edges per vertex. GN's Mesh to Curve breaks a spline at every vertex that
    # is not degree 2, so a branch hung straight off its parent's node vertex
    # chopped the parent into one capped cylinder per internode run -- the gaps
    # where two pieces met at an angle. Every child chain now starts from a
    # duplicate of the node (`attach`); only a free tip is continued, so each
    # axis sweeps as one unbroken tube.
    degree: list[int] = []

    def link(a: int, b: int) -> None:
        edges.append((a, b))
        degree[a] += 1
        degree[b] += 1

    # Which child continues a parent's free tip: the thickest (a fork's leader),
    # so a pseudomonopodial trunk runs on as one tube; the others start fresh.
    tip_heir: dict[tuple[int, int], int] = {}
    for axis in plant.axes.values():
        if axis.parent_axis_id < 0 or not axis.phytomers:
            continue
        parent = plant.axes.get(axis.parent_axis_id)
        if parent is None or not parent.phytomers \
                or axis.parent_phytomer_index != parent.phytomers[-1].index:
            continue
        key = (axis.parent_axis_id, axis.parent_phytomer_index)
        rival = plant.axes.get(tip_heir.get(key, -1))
        if rival is None or (axis.phytomers[0].radius, -axis.id) > \
                (rival.phytomers[0].radius, -rival.id):
            tip_heir[key] = axis.id

    # The chain being written: its tube id and the tube it branches from.
    chain = {"tube": -1, "parent": -1, "next": 0}

    def new_tube(parent: int, hidden: bool = False) -> None:
        chain["tube"], chain["parent"] = chain["next"], parent
        chain["next"] += 1
        if hidden:
            hidden_tubes.add(chain["tube"])

    def emit(position, width, order, birth_step, axis_id, node_index,
             curvature=0.0, load=0.0, stiffness=0.0, moment=0.0,
             structural_depth=-1, dead=0.0, factor=0.0, flare=0.0) -> int:
        index = len(verts)
        degree.append(0)
        attrs["flare"].append(float(flare))
        attrs["tube"].append(chain["tube"])
        attrs["parent_tube"].append(chain["parent"])
        verts.append(tuple(float(c) for c in position))
        attrs["width"].append(float(width))
        attrs["order"].append(int(order))
        attrs["birth_step"].append(float(birth_step))
        attrs["axis_id"].append(int(axis_id))
        attrs["node_index"].append(int(node_index))
        attrs["curvature"].append(float(curvature))
        attrs["load"].append(float(load))
        attrs["stiffness"].append(float(stiffness))
        attrs["moment"].append(float(moment))
        attrs["structural_depth"].append(int(structural_depth))
        attrs["dead"].append(float(dead))
        attrs["factor"].append(float(factor))
        attrs["age"].append(min(1.0, max(0.0, float(birth_step) / max(total_steps, 1))))
        attrs["split_depth"].append(max(int(structural_depth), 0) / _depth_norm)
        attrs["branch_depth"].append(max(int(order), 0) / _order_norm)
        attrs["branch_index"].append(_axis_rank.get(int(axis_id), 0) / _rank_norm)
        attrs["on_branch"].append(1.0 if order >= 1 else 0.0)
        attrs["node_frac"].append(max(int(node_index), 0) / _axis_node_norm.get(int(axis_id), 1.0))
        attrs["node_ordinal"].append(float(max(int(node_index), 0)))
        attrs["branch_ordinal"].append(float(_axis_rank.get(int(axis_id), 0)))
        _ax = plant.axes.get(int(axis_id))
        attrs["on_inflorescence"].append(
            0.0 if _ax is None else _on_inflorescence(_ax, int(node_index), _infl_axes)
            if node_index >= 0 else (1.0 if _ax.id in _infl_axes else 0.0))
        return index

    def attach(node_vert: int, width: float, order: int, axis_id: int,
               structural_depth: int, dead: float, continue_tip: bool = True,
               weld: bool = True, hidden: bool = False) -> int:
        """The vertex a child chain starts from: the node itself when it is a
        free tip the child may continue (one spline runs on), else a duplicate
        of it carrying the child's own width and identity (a new spline that
        starts inside the parent). `weld=False` records no parent tube: the
        chain is swept as a tube of its own, never welded (a catkin). A chain
        continues its parent's only when both are shown or both `hidden`, so a
        tube is hidden whole (a terminal flower's stalk off a hidden stem is a
        tube of its own)."""
        if continue_tip and degree[node_vert] <= 1 \
                and (attrs["tube"][node_vert] in hidden_tubes) == hidden:
            chain["tube"] = attrs["tube"][node_vert]
            chain["parent"] = attrs["parent_tube"][node_vert] if weld else -1
            return node_vert
        new_tube(attrs["tube"][node_vert] if weld else -1, hidden)
        return emit(verts[node_vert], width, order, attrs["birth_step"][node_vert],
                    axis_id, -1, structural_depth=structural_depth, dead=dead,
                    factor=attrs["factor"][node_vert])

    def flare_weight(height: float, base_radius: float) -> float:
        """1 at the ground, easing (quadratically) to 0 at the flare's top --
        three base radii up, raised with the flare so a strong one also runs
        higher, the way a buttressed trunk does."""
        top = 3.0 * base_radius * (1.0 + trunk_flare)
        if trunk_flare <= 0.0 or top <= 0.0 or height >= top:
            return 0.0
        return (1.0 - height / top) ** 2

    def flare_base(ground: int, axis, dead: float) -> int:
        """A basal axis widened toward the ground -- the trunk flare. The ground
        vertex is flared in place and extra control points are laid up the first
        internode so the flare curves rather than ramping node to node. Returns
        the last vertex laid."""
        if trunk_flare <= 0.0:
            return ground
        base_radius = float(axis.phytomers[0].radius)
        attrs["width"][ground] = base_radius * (1.0 + trunk_flare)
        attrs["flare"][ground] = 1.0
        p0 = np.asarray(verts[ground], dtype=float)
        seg = np.asarray(axis.phytomers[0].tip_position, dtype=float) - p0
        length = float(np.linalg.norm(seg))
        if length <= 1e-9:
            return ground
        top = 3.0 * base_radius * (1.0 + trunk_flare)
        previous = ground
        for frac in (0.12, 0.3, 0.55, 0.8):
            height = top * frac
            if height >= length * 0.85:
                break
            weight = flare_weight(height, base_radius)
            index = emit(p0 + seg / length * height, base_radius * (1.0 + trunk_flare * weight),
                         axis.order, 0.0, axis.id, -1,
                         structural_depth=axis.structural_depth, dead=dead, flare=weight)
            link(previous, index)
            previous = index
        return previous

    def axis_dir(axis, phytomer_index: int):
        """The parent axis's heading at one of its nodes, for the collar."""
        for k, ph in enumerate(axis.phytomers):
            if ph.index == phytomer_index:
                a = axis.phytomers[k - 1].tip_position if k else axis.base_position
                d = np.asarray(ph.tip_position, dtype=float) - np.asarray(a, dtype=float)
                n = float(np.linalg.norm(d))
                return d / n if n > 1e-9 else None
        return None

    for axis in plant.axes_in_topological_order():
        if not axis.phytomers:
            continue
        if axis.lifecycle == LifecycleState.SHED:
            continue  # a shed axis (and, via the cascade, its subtree) is gone
        if axis.id in hidden_axes:
            continue  # inflorescence structure of a muted flower component
        hide = bool(hidden_levels) and (axis.structural_depth < 1 or axis.is_fork) \
            and structure._axis_level(axis, len(show_levels)) in hidden_levels
        axis_dead = 1.0 if axis.lifecycle == LifecycleState.DEAD else 0.0
        if axis.parent_axis_id >= 0:
            key = (axis.parent_axis_id, axis.parent_phytomer_index)
            node = tip_vert.get(key)
            if node is None:
                continue  # parent phytomer was never built; nothing to hang from
            own = float(axis.phytomers[0].radius)
            previous = attach(node, own, axis.order, axis.id, axis.structural_depth,
                              axis_dead, continue_tip=tip_heir.get(key) == axis.id,
                              hidden=hide)
            if previous != node:
                # The start sits on the parent's centre line, inside it. A
                # branch as thick as its parent started with a cap as wide as
                # the parent, whose rim grazed the parent's surface -- a sliver
                # bump in the weld. Kept inside: 0.85 x the parent there; it
                # widens to its own radius where it leaves.
                attrs["width"][previous] = min(attrs["width"][previous],
                                               0.85 * float(attrs["width"][node]))
                if axis.parent_along_internode != 1.0:
                    # A branch that leaves off its node (a level's Slide, an
                    # internodal stalk) starts where it leaves, on the parent's
                    # centre line, not at the node.
                    verts[previous] = tuple(float(c) for c in axis.base_position)
        else:
            new_tube(-1, hide)
            previous = emit(axis.base_position, axis.phytomers[0].radius,
                            axis.order, 0.0, axis.id, -1,
                            structural_depth=axis.structural_depth, dead=axis_dead)
            previous = flare_base(previous, axis, axis_dead)

        # How far up a basal axis each node sits, for the flare (None on branches).
        climbed = (float(np.linalg.norm(np.asarray(verts[previous], dtype=float)
                                        - np.asarray(axis.base_position, dtype=float)))
                   if axis.parent_axis_id < 0 else None)
        for k, phytomer in enumerate(axis.phytomers):
            radius, flare = phytomer.radius, 0.0
            if climbed is not None:
                climbed += float(np.linalg.norm(
                    np.asarray(phytomer.tip_position, dtype=float)
                    - np.asarray(verts[previous], dtype=float)))
                flare = flare_weight(climbed, axis.phytomers[0].radius)
                radius = radius * (1.0 + trunk_flare * flare)
            index = emit(phytomer.tip_position, radius, axis.order,
                         phytomer.birth_step, axis.id, phytomer.index,
                         curvature=phytomer.debug_curvature,
                         load=phytomer.debug_load,
                         stiffness=phytomer.debug_stiffness,
                         moment=phytomer.debug_moment,
                         structural_depth=axis.structural_depth, dead=axis_dead,
                         factor=phytomer.factor, flare=flare)
            link(previous, index)
            tip_vert[(axis.id, phytomer.index)] = index
            previous = index

            # A pedicel is a real stalk, not just an offset -- collect them
            # now and append after the axis chain so the vertex it hangs from
            # already exists.
            for organ in phytomer.organs:
                if organ.pedicel_length > 0.0:
                    pedicels.append((index, organ, float(axis.stalk_radius(k)), axis))
        if axis.terminal_organ is not None and axis.terminal_organ.pedicel_length > 0.0:
            pedicels.append((previous, axis.terminal_organ,
                             float(axis.stalk_radius(len(axis.phytomers) - 1)), axis))

    for node_vert, organ, radius, axis in pedicels:
        if organ.kind == OrganKind.CATKIN and not show_catkins:
            continue
        if organ.kind == OrganKind.FLOWER and not show_pedicels:
            continue  # a flower's stalk goes with its (muted) flower component
        # A pedicel/catkin is its axis's own stalk, so it dies with the axis
        # (shed axes never reach here -- their pedicels were not collected).
        ped_dead = 1.0 if axis.lifecycle == LifecycleState.DEAD else 0.0
        # A young flower's stalk is thinner as well as shorter: the same
        # development fraction (`scale / final_scale`, its node's elongation --
        # see `model._organ_expansion`) that shortened it in growth. 1.0 once grown.
        grown = (organ.scale / organ.final_scale
                 if organ.kind == OrganKind.FLOWER and organ.final_scale > 0.0 else 1.0)
        base_width = max(1e-5, radius * organ.pedicel_width * grown)

        if organ.kind == OrganKind.CATKIN:
            # One stalk, then the bunch forks off its end -- a real fork in the
            # geometry, which is what makes it read as several catkins on one
            # pedicel rather than one fused tapering shape. See
            # `model.catkin_bunch`.
            # Every stalk vertex is remembered, because each catkin forks from
            # its *own* attachment rather than from the tip. Chaining them all
            # from the last vertex put the bodies in the right places and ran a
            # visible strand back to the tip from each -- see
            # `model.catkin_bunch`.
            # A catkin is an organ, not a branch: its stalk and bodies are never
            # welded (a weld fused and mangled whole bunches of them).
            start = attach(node_vert, base_width, axis.order + 1, axis.id,
                           axis.structural_depth, ped_dead, weld=False)
            stalk_verts = [node_vert]
            body_verts: list[int] = [] if start == node_vert else [start]
            previous_vert = start
            for point in organ.pedicel_points[1:]:
                index = emit(point, base_width, axis.order + 1, 0.0, axis.id, -1,
                             structural_depth=axis.structural_depth, dead=ped_dead,
                             factor=attrs["factor"][node_vert])
                link(previous_vert, index)
                previous_vert = index
                stalk_verts.append(index)
            for body, at, body_radii in zip(
                    organ.catkin_bodies, organ.catkin_attach,
                    organ.catkin_radii, strict=True):
                hang = stalk_verts[min(at, len(stalk_verts) - 1)]
                previous_vert = attach(hang, body_radii[0], axis.order + 1, axis.id,
                                       axis.structural_depth, ped_dead, weld=False)
                if previous_vert != hang:
                    body_verts.append(previous_vert)
                # Radii come from the growth side, computed against the same
                # positions that placed these vertices. Deriving them here from
                # a vertex index instead flattened the rounded end into a disc,
                # because the vertices are deliberately unevenly spaced -- see
                # `model.catkin_bunch`.
                for point, width in zip(body[1:], body_radii[1:], strict=True):
                    index = emit(point, width,
                                 axis.order + 1, 0.0, axis.id, -1,
                                 structural_depth=axis.structural_depth, dead=ped_dead,
                                 factor=attrs["factor"][node_vert])
                    link(previous_vert, index)
                    previous_vert = index
                    body_verts.append(index)
            # Stalk + bodies, all hanging off the node: a catkin the flower
            # component's masks reject goes whole (`_catkin_mask_drops`).
            pedicel_verts[int(organ.id)] = stalk_verts[1:] + body_verts
            continue

        # A curved pedicel resolves to a polyline; a straight one is just its
        # endpoint. Either way the chain starts at the node vertex that
        # already exists, so the stalk stays attached to the stem.
        if organ.pedicel_points:
            run = list(organ.pedicel_points[1:])
        else:
            # A straight stalk gets a short base stub of its own rather than
            # running as a single segment from the node. Without it the only
            # vertex carrying the stalk's width is the far end, so the whole
            # stalk interpolates out of the *stem's* radius -- which is the
            # "stalks are as thick as the stem at the base" this started from.
            start_pt = np.asarray(verts[node_vert], dtype=float)
            reach = np.asarray(organ.position, dtype=float) - start_pt
            run = [start_pt + reach * _PEDICEL_STUB, start_pt + reach]
        # A stalk tapers along its own length, shaped by the inflorescence's
        # own Pedicel Width curve -- see Inflorescence.pedicel_width_profile for
        # why that curve is read here rather than across the inflorescence, and
        # for what the taper looked like before it existed.
        taper = profile.inflorescence_at(organ.inflorescence, axis.order).pedicel_width_profile
        stem_width = attrs["width"][node_vert]
        # A lateral stalk starts from its own copy of the node (so the stem runs
        # on unbroken); a tip-joined one continues the axis's free tip.
        previous_vert = attach(node_vert, max(1e-5, base_width * float(taper(0.0))),
                               axis.order + 1, axis.id, axis.structural_depth, ped_dead)
        stalk = pedicel_verts.setdefault(int(organ.id), []) \
            if organ.kind == OrganKind.FLOWER else []
        if previous_vert != node_vert:
            stalk.append(previous_vert)
        for i, point in enumerate(run):
            along = i / max(len(run) - 1, 1)
            width = max(1e-5, base_width * float(taper(along)))
            if organ.tip_joined:
                # A stalk that continues its axis starts at the stem's own
                # thickness and narrows into the pedicel's, quickest near the
                # base -- the stem running on into its flower, not a thin
                # stalk stuck on a blunt stem end. See `model._tip_join`.
                ease = 1.0 - (1.0 - along) ** 2
                width = stem_width + (width - stem_width) * ease
            # It keeps its parent axis's id -- it is that axis's own stalk, not
            # a new axis -- with node_index -1 marking it as not a phytomer node.
            index = emit(point, width,
                         axis.order + 1, 0.0, axis.id, -1,
                         structural_depth=axis.structural_depth, dead=ped_dead,
                         factor=attrs["factor"][node_vert])
            link(previous_vert, index)
            previous_vert = index
            stalk.append(index)

    return verts, edges, attrs, pedicel_verts, frozenset(hidden_tubes)


def _catkin_mask_drops(obj, plant, pedicel_verts: dict, height_of, total_steps: int) -> set:
    """The ids of the catkins the flower components' masks reject.

    A catkin is not an instanced point -- it is a curve swept into the skeleton
    (`model.OrganKind.CATKIN`) -- so it never reached the component claim that
    masks flowers, and a flower component's masks did nothing to it (reported:
    a Height/On Branch mask thinned the flowers but not the catkins). Here each
    catkin is evaluated as the point it hangs from: the same structural channels
    (`_organ_mask_context`), its birth age and its height on the plant-wide
    0-1 scale (`height_of`), so a mask means the same on a catkin as on a
    flower. A catkin is kept if any unmuted flower component's masks keep it --
    a point no component wants renders nothing, the flower rule -- and then
    loses that component's Prune fraction. No flower component: nothing drops."""
    s = obj.plantgen
    comps = [c for c in s.components if c.type == 'flower' and not c.mute]
    catkins = [o for o in plant.all_organs()
               if o.kind == OrganKind.CATKIN and int(o.id) in pedicel_verts]
    if not comps or not catkins:
        return set()
    ctx = _organ_mask_context(plant)
    ids = np.array([int(o.id) for o in catkins], dtype=np.int64)
    attrs = {name: [] for name in (*_STRUCT_CHANNELS, "age", "height_fraction")}
    for o in catkins:
        struct = ctx.get(int(o.id), (0.0,) * len(_STRUCT_CHANNELS))
        for name, val in zip(_STRUCT_CHANNELS, struct, strict=True):
            attrs[name].append(float(val))
        attrs["age"].append(min(1.0, max(0.0, o.birth_step / max(total_steps, 1))))
        attrs["height_fraction"].append(height_of(o.position))
    chan = instancing._channels_from(attrs)
    chan.update(derived_channels(obj, [o.position for o in catkins], chan, ids))
    levels = structure.point_levels(plant, len(s.branch_levels), attrs["branch_ordinal"])
    seed = int(s.seed)
    claimed = np.zeros(len(ids), dtype=bool)   # some component's masks keep it
    survive = np.zeros(len(ids), dtype=bool)   # ...and its Prune spared it
    for ci, c in enumerate(comps):
        stack = instancing._filters_stack(c, seed)
        keeps = (np.ones(len(ids), dtype=bool) if stack.is_empty()
                 else masks.evaluate(stack, chan, ids)[0] > 0.5)
        keeps &= structure.component_level_ok(s, c, levels)
        keeps &= structure.component_inflorescence_ok(s, c, [o.inflorescence for o in catkins])
        mine = keeps & ~claimed                # first component in list order owns it
        claimed |= keeps
        if c.prune > 0.0:
            psalt = (instancing._cseed(seed, c) * 40499 + (ci + 1) * 2971215073
                     + 7 * 1597334677) & instancing._U64
            mine &= masks._hash01(ids, psalt) >= float(c.prune)
        survive |= mine
    return {int(i) for i in ids[~survive]}
"""Per plant (by object name): {component uid: (turned, pruned, still touching)}
from the last regrow's leaf collision resolve -- the component's Collision group
shows it. Render-side, read-only for the panel."""
"""Per plant (by name): the last leaf collision input digest and its outcome --
a regrow that changes nothing a leaf's pose depends on skips the resolve."""
"""Leaf asset collision proxies (`leaf_collision.simplify`), keyed on the
object, its mesh and a digest of its vertices -- re-read only when it changes."""
"""A flower proxy's clustering cells (across, across, along its +Z)."""
"""Per plant (by name): (key, {(component uid, vkey): None (pruned) or the turn
in the leaf's own frame (3x3, None for no turn)}, the reference Day)."""
"""Days sampled across Start Day..End Day for the reference (growth only)."""


def _drop_stem_verts(verts, edges, attrs, drop: set):
    """The stem arrays without the vertices in `drop` (and any edge touching
    one), every per-vertex attribute list filtered alongside. Used to take a
    masked-out flower's stalk with it -- a stalk is a chain hanging off its
    node, so nothing else ever depends on its vertices."""
    if not drop:
        return verts, edges, attrs
    n = len(verts)
    keep = [i not in drop for i in range(n)]
    remap, nxt = {}, 0
    for i in range(n):
        if keep[i]:
            remap[i] = nxt
            nxt += 1
    verts = [v for v, k in zip(verts, keep, strict=True) if k]
    edges = [(remap[a], remap[b]) for a, b in edges if a in remap and b in remap]
    for name, values in list(attrs.items()):
        if isinstance(values, list) and len(values) == n:
            attrs[name] = [x for x, k in zip(values, keep, strict=True) if k]
    return verts, edges, attrs


_ORGAN_ATTRS = (
    ("kind", 'INT'),
    # NOT "id". Confirmed live in this 5.2 build (2026-08-15): a plain "id"
    # named attribute survives a direct point-domain read intact, but any read
    # in a *realized* field context returns Blender's own per-instance identity
    # hashes instead -- probed values came back as 323095439, 626048946,
    # -842402296 where the source held 0, 1, 2. Node placement does not help:
    # a Named Attribute node sitting before Instance on Points still reads the
    # poisoned values when its field is consumed after realization, because a
    # field is evaluated where it is used. So the fix has to be the name.
    # Vertex-color Green needs exactly this per-organ value on realized leaf
    # vertices, which is the read that would have been silently wrong.
    ("organ_id", 'INT'),
    ("direction", 'FLOAT_VECTOR'),
    ("normal", 'FLOAT_VECTOR'),
    ("scale", 'FLOAT'),
    # Where this flower falls on the bloom timeline. Read in GN against a
    # Progress socket so the open band can travel without a regrow.
    ("bloom_start", 'FLOAT'),
    # When this flower actually appeared, on that same timeline. A separate
    # quantity from `bloom_start`, which is bloom *order* and knows nothing
    # about growth -- see `_bloom_openness` for why both are needed.
    ("bloom_birth", 'FLOAT'),
    # Heteroblasty: 0 = juvenile blade, 1 = adult. Chooses which asset
    # collection this leaf draws from, so the two forms can be genuinely
    # different meshes rather than one mesh at two sizes.
    ("form", 'INT'),
    # WS3 lifecycle: 0 alive, up to 1 dead -- a browning ramp for foliage, 1.0
    # for anything on a dead branch. A shed organ is not emitted at all.
    ("dead", 'FLOAT'),
    # Calendar phase 4: autumn discoloration stage, 0 (green) to 1 (fully
    # turned), from the day's position in the senescence window. Written on
    # leaves and bracts (0 on everything else) and passed through to the
    # realized leaf geometry, for a leaf shader to mix its own autumn colour on
    # -- the addon supplies the stage, the shader does the colour. Distinct from
    # `dead`, which browns whole senesced limbs; this turns all foliage together.
    ("autumn", 'FLOAT'),
)


def _wanted(organ, show_leaves: bool, show_flowers: bool, show_bracts: bool,
            show_fruit: bool = True) -> bool:
    """Whether this organ is switched on.

    A bract is a leaf everywhere else in the pipeline -- same kind, same
    instancing branch, same deformers -- and is told apart here only by its
    form, which is exactly how the Diagnostics panel counts it separately too.
    """
    if organ.kind == OrganKind.FLOWER:
        return show_flowers
    if organ.kind == OrganKind.FRUIT:
        return show_fruit
    if organ.kind == OrganKind.LEAF:
        return show_bracts if organ.form == LeafForm.BRACT else show_leaves
    return True
"""`kind` codes beyond leaf (0) and flower (1). Buds and apices share one asset
collection and one instancing branch -- a spent tip wanting its own asset is
deferred, exactly as WS1 5.2 says -- but keep separate codes so each can be
switched on its own and so a debug view can tell them apart. Fruit (4) is its
own branch: a rigid instanced asset like a bud, with no blade deform."""


def _bud_points(plant, hidden_axes=frozenset(), total_steps: int = 1):
    """Every visible meristem as (position, direction, normal, id, kind, struct,
    age) -- `struct` the `_STRUCT_CHANNELS` tuple of the node it sits on and
    `age` its birth on the 0-1 growth timeline, so a Bud component's masks read
    a bud exactly as they read the leaf beside it (they used to read all zeros,
    so Branch Depth / Node / Age masks did nothing on buds).

    Walked off the axes rather than `plant.all_organs()`, because a bud is not
    an organ: it is a meristem that may never become one. `Bud.position` and
    `.direction` are resolved by `resolve_geometry` alongside the organs, so
    nothing has to be recomputed here.

    **Only meristems that are still buds.** An axillary bud that already grew
    into a branch, or aborted, is not drawn -- it used to be, so a bud mesh sat
    at the foot of every branch. An apex is drawn unless its tip became a
    flower (a flowering region or a terminal flower); a shed axis draws none.
    Which of the rest render is the Bud component's masks (the Apex channel
    splits tip buds from side buds).

    The normal is the stem tangent at the node the bud sits on, which gives the
    instance a stable second axis -- a bud has no blade to orient, so any
    consistent perpendicular does, and the tangent is the one that reads as
    "up the stem".
    """
    norms = _axis_norms(plant)
    infl_axes = _inflorescence_axes(plant)
    steps = max(total_steps, 1)
    for axis in plant.axes_in_topological_order():
        if axis.id in hidden_axes or axis.lifecycle == LifecycleState.SHED:
            continue
        if not axis.phytomers:
            continue
        common, node_norm, b_ord = norms[axis.id]
        for pi, phytomer in enumerate(axis.phytomers):
            if phytomer.is_split:
                continue
            struct = (float(phytomer.factor), *common, pi / node_norm, float(pi), b_ord,
                      _on_inflorescence(axis, pi, infl_axes))
            for bud in phytomer.buds:
                if bud.state != BudState.DORMANT:
                    continue
                yield (bud.position, bud.direction, phytomer.direction, bud.id,
                       instancing._BUD_KIND, struct, min(1.0, bud.birth_step / steps))
        if (axis.apex is not None and axis.terminal_organ is None
                and axis.flowering_from < 0):
            tip = axis.phytomers[-1]
            last = len(axis.phytomers) - 1
            struct = (float(tip.factor), *common, 1.0, float(last), b_ord,
                      _on_inflorescence(axis, last + 1, infl_axes))
            yield (axis.apex.position, axis.apex.direction, tip.direction,
                   axis.apex.id, instancing._APEX_KIND, struct, min(1.0, tip.birth_step / steps))


def _sessile_seats(plant) -> dict:
    """`organ.id -> (stem direction, stem radius)` for every stalkless flower
    or fruit on a node -- what `_on_surface` needs to seat it on the stem."""
    seats = {}
    for axis in plant.axes.values():
        for phytomer in axis.phytomers:
            for organ in phytomer.organs:
                if organ.kind in (OrganKind.FLOWER, OrganKind.FRUIT) \
                        and not organ.pedicel_points:
                    seats[organ.id] = (phytomer.direction, float(phytomer.radius))
    return seats


def _on_surface(organ, seat) -> tuple:
    """Where an organ point is drawn: where growth put it, except a stalkless
    flower / fruit on a node, whose base growth leaves on the stem's centre
    line -- inside the stem, so it clipped. It is moved out to the surface: by
    the stem's radius there, straight out from the stem toward the way the
    flower faces. A stalked one needs nothing: its stalk carries it out."""
    position = np.asarray(organ.position, dtype=float)
    if seat is not None:
        tangent, radius = seat
        tangent = np.asarray(tangent, dtype=float)
        facing = np.asarray(organ.direction, dtype=float)
        out = facing - tangent * float(np.dot(facing, tangent))
        norm = float(np.linalg.norm(out))
        if norm > 1e-6 and radius > 0.0:
            position = position + out * (radius / norm)
    return tuple(float(c) for c in position)


def _organ_arrays(plant, bracts: bool = True,
                  show_leaves: bool = True, show_flowers: bool = True,
                  show_bracts: bool = True, show_buds: bool = False,
                  show_fruit: bool = True, onset: float = 0.0, total_steps: int = 1,
                  autumn: float = 0.0, hidden_axes=frozenset(), hidden_organs=frozenset(),
                  birth_window=None):
    """Organ points and their attributes.

    `bracts` is False when the plant has no bract collection, and forces a
    bract's form back to 0 (the foliage collection). The fallback belongs here
    rather than in the node graph because this is the only place that knows both
    the plant's decisions and what the scene actually
    provides: an unresolved collection makes Collection Info yield nothing, so
    leaving the forms alone would silently *delete* every organ of that form
    rather than drawing it from the collection that does exist.

    `onset`/`total_steps` put each flower's birth on the same 0..1 timeline
    Bloom Progress runs on, so the node group can refuse to open a flower that
    does not exist yet. `onset` is `_flowering_age`, which is why it has to be
    computed before this is called rather than with the other panel stats.
    A species with a bloom calendar runs Progress on its bloom window instead
    (`effective_bloom_progress`), so `birth_window` (birth step -> position in
    that window) states the birth on the window too: measured on the old
    maturity clock, Digitalis's flowers arrived up to 79% open, with no bud.
    """
    positions = []
    attrs = {name: [] for name, _ in _ORGAN_ATTRS}
    # Extra 0-1 channels the component filters read (not written to the mesh --
    # consumed straight from this dict by the filter evaluator and the preview).
    # The structural ones need the axis -> phytomer link that `all_organs()` drops,
    # so gather them once by walking the tree; `age` is the organ's own birth time.
    for _extra in (*_STRUCT_CHANNELS, "age", "apex", "unfolded", "vkey"):
        attrs[_extra] = []
    organ_ctx = _organ_mask_context(plant)
    seats = _sessile_seats(plant)
    span = max(1.0 - onset, 1e-4)
    for organ in plant.all_organs():
        if organ.scale <= 0.0:
            continue
        if organ.id in hidden_organs:
            continue  # on a muted flower component's inflorescence axes
        if not _wanted(organ, show_leaves, show_flowers, show_bracts, show_fruit):
            continue
        if organ.kind == OrganKind.CATKIN:
            # No instance: a catkin's geometry is a curve in the skeleton mesh,
            # written by `_stem_arrays`. See `model.OrganKind.CATKIN`.
            continue
        if organ.lifecycle == LifecycleState.SHED:
            continue  # a shed organ (a leaf, or anything on a shed branch) is gone
        positions.append(_on_surface(organ, seats.get(organ.id)))
        if organ.kind == OrganKind.FLOWER:
            kind_code = 1
        elif organ.kind == OrganKind.FRUIT:
            kind_code = instancing._FRUIT_KIND
        else:
            kind_code = 0  # leaf or bract
        attrs["kind"].append(kind_code)
        attrs["organ_id"].append(int(organ.id))
        attrs["vkey"].append(int(organ.id))
        attrs["direction"].append(tuple(float(c) for c in organ.direction))
        attrs["normal"].append(tuple(float(c) for c in organ.normal))
        attrs["scale"].append(float(organ.scale))
        attrs["bloom_start"].append(float(organ.bloom_start))
        birth_age = organ.birth_step / max(total_steps, 1)
        birth = (birth_window(organ.birth_step) if birth_window is not None
                 else (birth_age - onset) / span)
        attrs["bloom_birth"].append(min(1.0, max(0.0, birth)))
        struct = organ_ctx.get(int(organ.id), (0.0,) * len(_STRUCT_CHANNELS))
        for name, val in zip(_STRUCT_CHANNELS, struct, strict=True):
            attrs[name].append(float(val))
        attrs["age"].append(min(1.0, max(0.0, birth_age)))
        attrs["apex"].append(0.0)
        # How far a leaf has unfolded: its expansion so far (Leaf Unfolding).
        attrs["unfolded"].append(
            min(1.0, organ.scale / organ.final_scale)
            if organ.kind == OrganKind.LEAF and organ.final_scale > 0.0 else 1.0)
        form = int(organ.form)
        if form == LeafForm.BRACT and not bracts:
            form = LeafForm.JUVENILE
        attrs["form"].append(form)
        attrs["dead"].append(1.0 if organ.lifecycle == LifecycleState.DEAD else 0.0)
        # Foliage turns together as the plant enters autumn; only leaves and
        # bracts (kind LEAF) carry the stage, a flower or bud stays 0.
        attrs["autumn"].append(autumn if organ.kind == OrganKind.LEAF else 0.0)

    if show_buds:
        for position, direction, normal, bud_id, kind, struct, age in _bud_points(
                plant, hidden_axes, total_steps):
            positions.append(tuple(float(c) for c in position))
            attrs["kind"].append(kind)
            attrs["organ_id"].append(int(bud_id))
            attrs["vkey"].append(int(bud_id))
            attrs["direction"].append(tuple(float(c) for c in direction))
            attrs["normal"].append(tuple(float(c) for c in normal))
            # Unit scale: a Bud component's Size is the bud's size (a multiplier
            # on its asset, like every other type), applied in `_assign_type`.
            attrs["scale"].append(1.0)
            # A bud does not bloom; every remaining attribute takes its neutral
            # value so the arrays stay square.
            attrs["bloom_start"].append(0.0)
            attrs["bloom_birth"].append(0.0)
            attrs["form"].append(0)
            attrs["dead"].append(0.0)
            attrs["autumn"].append(0.0)
            for name, val in zip(_STRUCT_CHANNELS, struct, strict=True):
                attrs[name].append(float(val))
            attrs["age"].append(float(age))
            attrs["apex"].append(1.0 if kind == instancing._APEX_KIND else 0.0)
            attrs["unfolded"].append(1.0)
    return positions, attrs


# The per-organ structural channels the filters read, in the order the extras
# dict and `_organ_mask_context` tuples pack them. `node_frac` (not `node_index`)
# because the stem mesh already has an INT `node_index` attribute -- the channel
# is the normalised 0-1 version, kept under its own key to avoid the collision.
_STRUCT_CHANNELS = ("factor", "split_depth", "branch_depth", "branch_index",
                    "on_branch", "node_frac", "node_ordinal", "branch_ordinal",
                    "on_inflorescence")


def _on_inflorescence(axis, pi: int, infl_axes) -> float:
    """1 on an inflorescence, 0 on the leafy stem: a node at or past where its
    axis began flowering (`Axis.flowering_from`), or any node of an axis the
    inflorescence released (its branches, rays, stalks: `_inflorescence_axes`).
    `pi` past the last node is the axis's tip."""
    if axis.id in infl_axes:
        return 1.0
    return 1.0 if 0 <= axis.flowering_from <= pi else 0.0


def _axis_norms(plant) -> dict:
    """Per axis id: `((split, branch_depth, branch_index, on_branch), node_norm,
    branch_ordinal)` -- the axis-level half of every point's structural mask
    channels, normalised once for the plant so organs and bud points read the
    same 0-1 ranges."""
    max_depth = max((max(a.structural_depth, 0) for a in plant.axes.values()),
                    default=0)
    depth_norm = float(max_depth) if max_depth > 0 else 1.0
    max_order = max((a.order for a in plant.axes.values()), default=0)
    order_norm = float(max_order) if max_order > 0 else 1.0
    rank = {aid: i for i, aid in enumerate(sorted(plant.axes))}
    rank_norm = float(len(rank) - 1) if len(rank) > 1 else 1.0
    out = {}
    for aid, axis in plant.axes.items():
        common = (max(axis.structural_depth, 0) / depth_norm,
                  max(axis.order, 0) / order_norm,
                  rank[aid] / rank_norm,
                  1.0 if axis.order >= 1 else 0.0)
        node_norm = float(len(axis.phytomers) - 1) if len(axis.phytomers) > 1 else 1.0
        out[aid] = (common, node_norm, float(rank[aid]))
    return out


def _organ_mask_context(plant) -> dict:
    """`organ.id -> (factor, split_depth, branch_depth, branch_index, on_branch,
    node_index)` for the component filters, gathered by walking axes -> phytomers
    -> organs (the link `all_organs()` flattens away).

    - `factor` -- the phytomer's own 0-1 position along its axis; a terminal organ
      takes its axis's tip.
    - `split_depth` -- `axis.structural_depth` (-1 on vegetative stem, rising with
      each fork / inflorescence split) normalised to 0-1 against the deepest split.
    - `branch_depth` -- `axis.order` (0 on the main stem, rising with each branch
      level) normalised against the deepest branch.
    - `branch_index` -- a distinct 0-1 value per axis (rank among all axis ids), so
      a filter can pick out particular branches.
    - `on_branch` -- 0 on the main stem (order 0), 1 on any branch.
    - `node_index` -- the phytomer's 0-1 position by node count along its axis (0 at
      the base to 1 at the tip); the discrete-node sibling of `factor`.
    All normalised so a filter reads the same 0-1 range on any plant."""
    ctx: dict[int, tuple] = {}
    norms = _axis_norms(plant)
    infl_axes = _inflorescence_axes(plant)
    for aid, axis in plant.axes.items():
        # node_ordinal / branch_ordinal are the RAW integer indices (not normalised),
        # so a mask modulo (every Nth node / branch) has a whole number to work on.
        common, node_norm, b_ord = norms[aid]
        for pi, phytomer in enumerate(axis.phytomers):
            node_index = pi / node_norm
            infl = _on_inflorescence(axis, pi, infl_axes)
            for organ in phytomer.organs:
                ctx[int(organ.id)] = (float(phytomer.factor), *common, node_index,
                                      float(pi), b_ord, infl)
        if axis.terminal_organ is not None:
            tip = float(axis.phytomers[-1].factor) if axis.phytomers else 1.0
            last = float(len(axis.phytomers) - 1) if axis.phytomers else 0.0
            # a terminal flower (or the fruit it set) is a one-flower inflorescence
            infl = max(_on_inflorescence(axis, len(axis.phytomers), infl_axes),
                       1.0 if axis.terminal_organ.kind in (OrganKind.FLOWER,
                                                           OrganKind.FRUIT) else 0.0)
            ctx[int(axis.terminal_organ.id)] = (tip, *common, 1.0, last, b_ord, infl)
    return ctx


def _flower_spacing(plant) -> float:
    """Mean distance between neighbouring flowers along a flowering region.

    Reported because **Flower Size and Flower Spacing are independent, and the
    usable range of the first depends on the second.** A flower asset wider
    than the gap to its neighbour interpenetrates, and with a golden-angle
    arrangement the overlapping flowers merge into the spiral rows, which reads
    as them converging on a few points rather than as being too big. That was
    reported as the plant "going wonky" at Flower Size 1, and it took measuring
    the asset against the spacing to see it was arithmetic: 19.6 mm flowers
    15.5 mm apart.

    Coupling the two would be the wrong fix -- how big a flower is and how far
    apart they sit are separate facts about a plant -- so the panel says the
    numbers instead.
    """
    import numpy as np

    gaps = []
    for axis in plant.axes.values():
        if axis.flowering_from < 0:
            continue
        pos = [o.position for ph in axis.phytomers for o in ph.organs
               if o.kind == OrganKind.FLOWER]
        gaps += [float(np.linalg.norm(pos[i + 1] - pos[i]))
                 for i in range(len(pos) - 1)]
    return float(sum(gaps) / len(gaps)) if gaps else 0.0


def _tidy_organs_object(obj, organs_obj) -> None:
    """Keep the organs object attached to its plant and out of the way.

    It is **not** optional and cannot be deleted: it is the Object input the
    node group instances every leaf and flower on. Measured on the Anemone,
    unplugging it takes the evaluated mesh from 12,198 vertices to 6,528 --
    it carries 46% of the plant. But it is pure plumbing, so it has no business
    sitting loose in the Outliner next to real objects.

    Parented with an identity parent-inverse so its world matrix *is* the
    plant's: the points are written in the plant's own local coordinates, so
    the two coincide exactly and moving the plant carries them along.

    **Cosmetic only, and deliberately so.** The Object Info node reads it in
    `'ORIGINAL'` space, which ignores the object's transform entirely, so none
    of this can change the geometry -- which is why resetting a transform
    someone may have nudged is safe rather than destructive.

    **Hidden with `hide_set`, never `hide_viewport`.** The eye hides an object
    from the view layer while leaving it in the depsgraph; the monitor icon
    (`hide_viewport`) can drop it out of evaluation entirely, and an Object
    Info node reading an unevaluated object returns nothing at all -- the exact
    failure that made a custom stem profile come out as a bare one-vertex
    spline earlier today.
    """
    # Each set only when it differs: a write tags the object, even to its own
    # value, and the depsgraph then re-evaluates whatever reads it.
    if organs_obj.parent is not obj:
        organs_obj.parent = obj
        organs_obj.matrix_parent_inverse = Matrix.Identity(4)
    if organs_obj.matrix_basis != Matrix.Identity(4):
        organs_obj.matrix_basis = Matrix.Identity(4)
    if not organs_obj.hide_render:
        organs_obj.hide_render = True
    # `hide_set` needs the object to be in the active view layer, which it is
    # not while a plant is being built from a script or a background render.
    try:
        if not organs_obj.hide_get():
            organs_obj.hide_set(True)
    except RuntimeError:
        pass


def _ensure_organs_object(obj) -> bpy.types.Object:
    """Find or create the points-only object holding leaf/flower instancing
    points. Looked up by a remembered name rather than a naming convention, so
    renaming the plant object later does not orphan the link, and linked into
    the plant's own collection rather than `bpy.context.collection` -- the
    active Outliner collection is not necessarily where the object lives."""
    key = "plantgen_organs_name"
    existing = obj.get(key)
    organs_obj = bpy.data.objects.get(existing) if existing else None
    if organs_obj is None:
        organs_mesh = bpy.data.meshes.new(f"{obj.name}_Organs")
        organs_obj = bpy.data.objects.new(f"{obj.name}_Organs", organs_mesh)
        organs_obj["plantgen_organs"] = True
        collection = obj.users_collection[0] if obj.users_collection else bpy.context.collection
        collection.objects.link(organs_obj)
        obj[key] = organs_obj.name
    # Re-asserted on every regrow, not just at creation, so plants made before
    # this existed are tidied the first time they are rebuilt.
    _tidy_organs_object(obj, organs_obj)
    return organs_obj


_COMS_KEY = "plantgen_coms"
"""On the plant: each stem's centre of mass at the last regrow, flat x, y, z."""
_COM_NAMES_KEY = "plantgen_com_names"
"""On the plant: its centre-of-mass marker empties' names, newline-separated."""


def _sync_com_empty(obj) -> None:
    """Create, move or remove the centre-of-mass markers -- **one per stem**,
    each at the centre of mass that stem carries (its branches, leaves and
    flowers), which is what that stem leans toward -- following
    `obj.plantgen.show_center_of_mass` and the centres stored at the last
    regrow. (One marker at the whole plant's centre averaged a clump's stems
    into a point none of them leans toward.)

    Empties -- small spheres in space -- rather than a viewer colour, because
    the whole point is to *see where the mass sits* and watch each stem lean
    toward it. Parented to the plant with an identity inverse (the `_Organs`
    object's own pattern), so a stored centre is read as a plain local offset
    and the markers ride along when the plant is moved.
    """
    names = [n for n in str(obj.get(_COM_NAMES_KEY, "")).split("\n") if n]
    legacy = obj.get("plantgen_com_name")            # the one-marker era's
    if legacy and legacy not in names:
        names.insert(0, legacy)
    empties = [e for e in (bpy.data.objects.get(n) for n in names) if e is not None]
    flat = list(obj.get(_COMS_KEY, []))
    coms = [flat[i:i + 3] for i in range(0, len(flat) - 2, 3)]
    if not obj.plantgen.show_center_of_mass:
        coms = []
    for surplus in empties[len(coms):]:
        bpy.data.objects.remove(surplus, do_unlink=True)
    empties = empties[:len(coms)]
    collection = obj.users_collection[0] if obj.users_collection else bpy.context.collection
    size = max(float(obj.plantgen.last_height), 0.05) * 0.06
    while len(empties) < len(coms):
        empty = bpy.data.objects.new(f"{obj.name}_COM", None)
        empty["plantgen_com_marker"] = True
        empty.empty_display_type = 'SPHERE'
        empty.show_in_front = True
        collection.objects.link(empty)
        empties.append(empty)
    for empty, com in zip(empties, coms, strict=True):
        empty.parent = obj
        empty.matrix_parent_inverse = Matrix.Identity(4)
        empty.location = (float(com[0]), float(com[1]), float(com[2]))
        # Sized to the plant so it reads at any scale without swamping the geometry.
        empty.empty_display_size = size
    obj.pop("plantgen_com_name", None)
    if empties:
        obj[_COM_NAMES_KEY] = "\n".join(e.name for e in empties)
    else:
        obj.pop(_COM_NAMES_KEY, None)


_GROWN_KEY = "plantgen_grown"
"""Custom property holding the signature of the settings the current mesh was
grown from. On the object on purpose: undo restores it from the same snapshot
as the mesh, so the two can be compared against the settings actually present.
"""

_TOPO_KEY = "plantgen_topology"
"""Custom property holding a hash of the last-grown plant's **axis structure**.

Written at every regrow so the Prune operator, which has only the inspector
snapshot in hand, can stamp its axis ids against the topology they were picked
from. A pruned axis id names the same limb only while this is unchanged; the
bridge clears the prune set when it is not. Deliberately a *topology* hash and
not `growth_signature` -- the growth signature also moves on render-only tweaks
(Dead Colour, bloom, tip radius) that reassign no axis, and clearing pruning on
those would break the plan's own 'cosmetic tweaks don't disturb it'."""


def topology_signature(plant) -> str:
    """A hash of the plant's axis tree: every axis's id, parent, order and node
    count. This is exactly what a pruned axis id depends on -- if the tree is
    the same, id X is the same limb -- and nothing else, so a plant whose only
    change is cosmetic hashes identically."""
    rows = tuple(sorted(
        (a.id, a.parent_axis_id, a.order, len(a.phytomers))
        for a in plant.axes.values()))
    return hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=16).hexdigest()

_grown: set[str] = set()
"""Plants this session has regrown. Only these can carry a mesh that no undo
step ever recorded -- one loaded from a file was saved beside its own mesh."""


def growth_signature(obj, include_day: bool = True) -> str:
    """A hash of every input `regrow` reads (`include_day` False: all but the
    Day -- the Timelapse Cache's key, whose frames ARE the Days).

    Built from `_FIELDS` rather than a hand-kept list, so a field that reaches
    the profile cannot be forgotten here -- a missing one would not raise, it
    would quietly stop a stale mesh being noticed.
    """
    s = obj.plantgen
    parts = [repr(getattr(s, name, None)) for name, *_ in _FIELDS]
    parts.append(repr((s.seed, round(s.day, 6) if include_day else None)))
    parts.append(repr(curves.signature(obj)))
    parts.append(_scene_signature(obj))
    # The embedded base carries every hidden trait `_FIELDS` cannot (plastochron,
    # veg, masses...), so a preset swap that leaves the panel-mapped fields
    # unchanged still changes the plant -- hashed here (not stored raw; it is
    # ~13 KB) or an undo restoring a different base would leave a stale mesh.
    parts.append(hashlib.blake2b(
        s.preset_embedded.encode("utf-8"), digest_size=16).hexdigest())
    # Manual pruning is applied at bridge time, not through the profile, so it
    # is not in `_FIELDS` -- but it changes the mesh (a pruned limb is shed or
    # dead), so it belongs in the staleness signature or an undo that restores
    # a different prune set would leave the mesh out of step. This is separate
    # from the topology signature the prune set is *validated* against, so there
    # is no circularity: toggling prune-retain does not clear the set.
    parts.append(repr((sorted(p.axis for p in s.inspect_pruned),
                       s.prune_retain_dead_wood)))
    # Calendar windows are authored per-object but handled outside `_FIELDS` (a
    # nested PhenologyCalendar), yet they change the day-path plant -- hashed here
    # or an undo restoring a different calendar would leave a stale mesh.
    parts.append(repr((
        s.calendar_authored,
        s.bloom_enabled, round(s.bloom_start, 4), round(s.bloom_days, 4),
        s.autumn_enabled, round(s.autumn_start, 4), round(s.autumn_days, 4),
        s.fruit_enabled, round(s.fruit_start, 4), round(s.fruit_days, 4))))
    # Components: since the leaf / flower / bract / bud / fruit ports they carry
    # growth inputs (the primary flower component's inflorescence, fruit Set /
    # Weight / Size, bract Size) as well as every render choice (masks, mute,
    # Size, placement) -- all of it outside `_FIELDS`, so none of it was hashed
    # and an undo restoring a different component setting left a stale mesh.
    # Walked generically off RNA, so a prop shared onto the component later is
    # covered without touching this. (Their curves are in `curves.signature`.)
    parts.append(_components_signature(s))
    # Plant-level classification + growth maps (and a Distance attribute's
    # target transform, which moves its values with no setting changing).
    parts.append(repr([_rna_values(x) for x in (*s.custom_attrs, *s.classes,
                                                   *s.growth_maps, *s.branch_levels,
                                                   *s.placements)]))
    parts.append(repr([_matrix_key(a.target) for a in s.custom_attrs
                       if a.kind == "DISTANCE" and a.target is not None]))
    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _rna_values(pg) -> list:
    """Every stored RNA property value of a property group, pointers by name and
    collections recursed, in a stable order -- for the staleness hash."""
    out = []
    for prop in pg.bl_rna.properties:
        name = prop.identifier
        if name in ("rna_type", "expanded"):     # view state, not the plant
            continue
        value = getattr(pg, name, None)
        if prop.type == 'COLLECTION':
            out.append((name, [_rna_values(item) for item in value]))
        elif prop.type == 'POINTER':
            out.append((name, getattr(value, "name", None)))
        elif prop.type in ('FLOAT', 'INT', 'BOOLEAN') and getattr(prop, "is_array", False):
            out.append((name, tuple(value)))
        elif prop.type == 'ENUM' and prop.is_enum_flag:
            out.append((name, tuple(sorted(value))))
        else:
            out.append((name, value))
    return out


def _components_signature(s) -> str:
    blob = repr([_rna_values(c) for c in s.components]).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _matrix_key(other) -> tuple:
    return tuple(round(v, 6) for row in other.matrix_world for v in row)


def _scene_signature(obj) -> str:
    """What growth reads from the scene rather than from the panel.

    Stage 3 voxelises obstacles in **plant-local** space and steers effectors
    the same way, so the plant's own matrix and every one of theirs decides the
    result -- with no setting changing at all. Measured: nudging an obstacle
    slab 50 mm took the plant from 596.141 mm to 601.157 mm, and the panel-only
    signature was byte-identical, so an undo restoring a different transform
    left a stale mesh nothing could detect.

    Gated exactly as `environment.build_environment` gates itself, so a plant
    that responds to nothing hashes nothing and costs nothing.
    """
    s = obj.plantgen
    has_obstacles = s.obstacle_collection is not None and s.obstacle_clearance > 0.0
    has_effectors = s.effector_collection is not None
    # Heliotropism reads the tracked object's position in plant-local space, so
    # its matrix and the plant's own both decide the result with no panel value
    # changing -- exactly the case obstacles and effectors are hashed for.
    suns = sun_objects(s)
    has_sun = bool(suns)
    ground = s.ground_object if s.ground_object is not None \
        and s.ground_object.type == 'MESH' else None
    if not (has_obstacles or has_effectors or has_sun or ground is not None):
        # Growth is computed in plant-local space, so a plant that responds to
        # nothing grows the same wherever it stands -- and hashing its matrix
        # here would call an ordinary move "stale" and regrow it for nothing.
        return "-"

    parts = [repr(_matrix_key(obj))]

    if has_obstacles:
        parts.append("obs:" + repr(tuple(sorted(
            (other.name, _matrix_key(other), len(other.data.vertices))
            for other in s.obstacle_collection.all_objects
            if other.type == 'MESH' and other is not obj))))
    else:
        parts.append("obs:-")

    if has_effectors:
        rows = []
        for other in s.effector_collection.all_objects:
            settings = getattr(other, "plantgen_effector", None)
            values = () if settings is None else tuple(
                repr(getattr(settings, prop.identifier, None))
                for prop in settings.bl_rna.properties if not prop.is_readonly)
            rows.append((other.name, _matrix_key(other), values))
        parts.append("eff:" + repr(tuple(sorted(rows))))
    else:
        parts.append("eff:-")

    if has_sun:
        parts.append("sun:" + repr(tuple((o.name, _matrix_key(o)) for o in suns)))
    else:
        parts.append("sun:-")

    if ground is not None:
        parts.append("ground:" + repr((ground.name, _matrix_key(ground),
                                       len(ground.data.vertices))))

    return "|".join(parts)


def stamp(obj) -> None:
    """Record the settings the current mesh was grown from, and start watching
    this plant for the mesh and the settings drifting apart."""
    obj[_GROWN_KEY] = growth_signature(obj)
    _grown.add(obj.name)


def is_stamped(obj) -> bool:
    return obj.get(_GROWN_KEY) is not None


def is_stale(obj) -> bool:
    """Whether this plant's mesh was grown from settings it no longer has.

    False when there is no stamp at all -- a plant that predates the stamp is
    not known to be wrong, and regrowing every such plant on the first undo
    would be a surprise rather than a repair.
    """
    stamp = obj.get(_GROWN_KEY)
    return stamp is not None and stamp != growth_signature(obj)


def recently_grown() -> tuple[str, ...]:
    return tuple(_grown)


def apply_pruning(obj, plant) -> None:
    """Mark manually pruned limbs on a freshly built plant, and keep the prune
    set honest about the topology it was picked against.

    Runs after `build` (so after the age/shade fronts) and applies on top:
    manual > (age u environment), so a pruned axis dies whatever the fronts
    decided, shed by default and dead wood on the retain toggle. `cascade` takes
    the whole subtree, the same helper the fronts use.

    The topology is stamped every call so the Prune operator, which only has the
    inspector snapshot, can capture it. If the stored `prune_signature` no longer
    matches -- the plant was reseeded, re-aged, rebranched, anything that
    reassigns axis ids -- the ids name different limbs now, so the set is cleared
    and a warning flag is raised for the panel. A cosmetic change leaves the
    topology hash identical, so it survives, which is the whole reason this is a
    topology hash and not the growth signature.
    """
    from ..growth.lifecycle import cascade, children_map
    from ..growth.model import LifecycleState

    s = obj.plantgen
    current = topology_signature(plant)
    obj[_TOPO_KEY] = current
    if not len(s.inspect_pruned):
        return
    if s.prune_signature and s.prune_signature != current:
        s.inspect_pruned.clear()
        s.prune_signature = ""
        s.prune_stale = True
        return

    branch_state = (LifecycleState.DEAD if s.prune_retain_dead_wood
                    else LifecycleState.SHED)
    children = children_map(plant)
    for entry in s.inspect_pruned:
        axis = plant.axes.get(entry.axis)
        if axis is not None:
            cascade(plant, axis, branch_state, LifecycleState.SHED, children)


def _grow_and_prune(obj, day: float | None = None):
    """The canonical grow the renderer draws from: the profile from the object's
    panel + embedded preset, the Stage-3 environment (None whenever the plant
    responds to nothing -- the default, and free -- see
    `environment.build_environment`), the day-driven build (`build` maps `day`
    to developmental progress via the season window + growth curve; `age` stays
    the derived internal, deliberately not re-synced), and WS3 manual pruning
    stamped on top. Returns (plant, profile, environment).

    Single-sourced so anything needing *the same plant* -- `regrow` to mesh it,
    `rig.build_rig` to skeletonise it -- reads one grow, not a copy
    that can drift from the render. `day` grows it to another Day than the
    panel's (the timelapse's leaf reference, `ensure_leaf_reference`).
    """
    s = obj.plantgen
    day = float(s.day) if day is None else float(day)
    profile = profile_from_object(obj)
    key = _grow_key(obj, profile, day)
    entries = _GROW_CACHE.setdefault(obj.name, [])
    for i, (k, plant, cached_profile, environment) in enumerate(entries):
        if k == key:
            entries.insert(0, entries.pop(i))          # most recent first
            return plant, cached_profile, environment
    environment = env_bridge.build_environment(obj, profile)
    plant = build(profile, seed=s.seed, day=day, environment=environment,
                  signals=growth_signals_for(obj, profile))
    apply_pruning(obj, plant)
    entries.insert(0, (key, plant, profile, environment))
    del entries[_GROW_CACHE_DEPTH:]
    return plant, profile, environment


_GROW_CACHE_DEPTH = 5
"""Grows kept per plant: the current one plus the last four, so undoing a few
growth edits, or A/B-flicking a value back, is a hit instead of a full regrow."""

_GROW_CACHE: dict = {}
"""obj name -> [(grow key, plant, profile, environment), ...], most recent first.

**Render-side edits reuse the grown plant instead of re-growing it.** Growth +
physics is nearly all of a regrow -- measured 7.3 s of 7.4 s on a 175-catkin
plant -- while everything a component's masks, Size, Density, Prune, mute and
placement decide runs after it in ~0.1 s. Before this, every mask tweak paid
the full grow, which is what kept classification from feeling live.

Safe because the key is everything growth reads (`_grow_key`) and nothing
after `_grow_and_prune` mutates the plant (the bridge only reads it; the shade
pass re-lights the environment against it, which is idempotent). A plain
module dict, so it does not survive a reload or reopen -- the first regrow
after either simply grows. Undo is fine: the key is recomputed from the
restored settings, so a restored different state misses and regrows."""


def _grow_key(obj, profile, day: float) -> str:
    """Everything `build` + `apply_pruning` read, hashed: the resolved profile
    itself (so any path into it -- panel, component, curve, preset -- is covered
    without a list to keep in step), the seed and Day, the scene inputs the
    environment is built from (`_scene_signature`), and the prune set."""
    s = obj.plantgen
    data = serialize.profile_to_dict(profile)
    for level in data.get("levels", []):
        level.pop("name", None)          # a rename is not a growth change
    blob = json.dumps(data, sort_keys=True, default=repr)
    parts = (blob, repr((s.seed, round(float(day), 6))), _scene_signature(obj),
             repr(growth_signals_for(obj, profile)),
             repr((sorted(p.axis for p in s.inspect_pruned), s.prune_retain_dead_wood,
                   s.prune_signature)))
    return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def regrow_is_cheap(obj) -> bool:
    """Whether a regrow now would come from the caches -- the grow already done
    for these settings -- so it costs a fraction of a second, not seconds: a
    live regrow never welds (`WELD_LATER`), and the rest after growth is quick.
    Live update then regrows almost at once instead of waiting for a drag to
    end."""
    entries = _GROW_CACHE.get(obj.name)
    if not entries:
        return False
    try:
        key = _grow_key(obj, profile_from_object(obj), float(obj.plantgen.day))
    except Exception:  # noqa: BLE001 -- a guess for scheduling, never fatal
        return False
    return any(k == key for k, *_ in entries)


def grown_plant(obj):
    """The built, pruned `Plant` the renderer draws -- for the rig/bake to read,
    so the armature matches the geometry exactly. Same plant `regrow` meshes."""
    return _grow_and_prune(obj)[0]


def plant_at_day(obj, day):
    """The `Plant` built at an arbitrary Day, WITHOUT WS3 manual pruning or its
    obj-stamping side effect -- for the rig bake to read each frame's node frames
    without thrashing the object per frame. The rig's rest pose and skin are
    built from `grown_plant` (pruned) at the mature day; a per-frame plant that
    still carries a limb the mature plant pruned simply has no matching rig bone,
    so it is ignored."""
    s = obj.plantgen
    profile = profile_from_object(obj)
    environment = env_bridge.build_environment(obj, profile)
    return build(profile, seed=s.seed, day=day, environment=environment,
                 signals=growth_signals_for(obj, profile))


def _wants_shade(s) -> bool:
    """Whether this regrow needs the shade channel computed. Shade is the one
    channel not already sitting on the points: it means sampling the self-shading
    light field, which is real work, so it is done only when something consumes
    it -- a shade filter on any component, or the Shade debug view."""
    if s.debug_view_mode == 'shade':
        return True
    return any(e.channel == 'shade' for c in s.components
               for e in (*c.filters, *(m for m in c.maps if m.enabled)))


def channel_key(ref) -> str:
    """The key a mask entry / map / custom input reads from the channel dict:
    its plain channel id, or `attr:<name>` / `class:<name>` for a reference to
    one of the plant's custom attributes / classes."""
    if ref.channel == "custom":
        return masks.CUSTOM_PREFIX + ref.attr_name
    if ref.channel == "class":
        return masks.CLASS_PREFIX + ref.class_name
    return ref.channel


def _ref_key(channel: str, attr_name: str, class_name: str) -> str:
    if channel == "custom":
        return masks.CUSTOM_PREFIX + attr_name
    if channel == "class":
        return masks.CLASS_PREFIX + class_name
    return channel


def signals_from_object(obj):
    """The plant's custom attributes and classes as bpy-free `growth.signals`
    (a Distance target converted into the plant's own space), or None when it
    has neither."""
    s = obj.plantgen
    if not (len(s.custom_attrs) or len(s.classes)):
        return None
    to_local = obj.matrix_world.inverted()
    attributes = []
    for a in s.custom_attrs:
        point = None
        if a.kind == "DISTANCE" and a.target is not None:
            point = tuple(float(v) for v in to_local @ a.target.matrix_world.translation)
        attributes.append(growth_signals.CustomAttribute(
            name=a.name, kind=a.kind, point=point, radius=float(a.radius),
            noise_scale=float(a.noise_scale), seed=int(a.seed), op=a.op,
            channel_a=_ref_key(a.channel_a, a.attr_a, a.class_a),
            channel_b=_ref_key(a.channel_b, a.attr_b, a.class_b)))
    classes = [growth_signals.ClassDef(c.name, _mask_entries(c.filters)) for c in s.classes]
    return growth_signals.Signals(attributes=attributes, classes=classes, seed=int(s.seed))


def growth_signals_for(obj, profile):
    """The signals growth needs: the attributes and classes a map or level
    mask reads while growing (and what those read in turn), else None -- so
    editing one that growth never reads does not re-grow the plant."""
    keys = [k for k in growth_maps_lib.channel_keys(profile) if growth_signals.is_derived(k)]
    if not keys:
        return None
    return growth_signals.restrict(signals_from_object(obj), keys)


def derived_channels(obj, positions, base: dict, ids) -> dict:
    """Every custom attribute (`attr:<name>`) and class (`class:<name>`) of the
    plant, evaluated over these points -- the Natsura-parity "chain
    classification for as many dimensions as you want" (clean-room).

    A custom attribute is a new dimension: closeness to an object (in plant-local
    space), seeded noise, or two channels combined -- any channel, including
    other attributes and classes. A class is a named mask stack evaluated with
    the same bpy-free evaluator a component's masks use, so it can read custom
    attributes and other classes too. The evaluation is `growth.signals` --
    the same one growth reads them with, so both see one Noise pattern."""
    signals = signals_from_object(obj)
    if signals is None:
        return {}
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    return growth_signals.Evaluator(signals, np.asarray(ids), base, pos).all()
"""Where a frame's `struct` holds its phytomer's own index (`_placement_frames`)."""
"""Axis lean (sine of its angle from vertical) over which `_azimuth_zero` turns
from the node's own reference to the branch's horizontal side."""
"""Per plant (by object name), what the Classify view plots after the last
regrow: the organ points' channel dict (incl. derived `attr:` / `class:` keys),
`kind` / `form` / `organ_id`, the final `render_drop`, and `owner` -- the index
into `s.components` of the component that claimed each point (-1 unclaimed).
Render-side only and read-only for the view; nothing in growth reads it."""


def _prepare_component_curves(obj, s) -> None:
    """Give every component a stable uid (backfilling old files and any collision)
    and a flat size-curve node for each generated component, so its popover has a
    node to draw and its curve is there for the bake to sample. Runs at regrow,
    never in draw (both write datablocks)."""
    seen = set()
    nxt = max((c.uid for c in s.components), default=0) + 1
    for c in s.components:
        if c.uid <= 0 or c.uid in seen:
            c.uid = nxt
            nxt += 1
        seen.add(c.uid)
    structure.ensure_level_uids(s)
    if not len(s.tree_rows):
        structure.rebuild_tree(s)
    generated = set(instancing._GENERATED_MODES)
    for c in s.components:
        if generated & set(c.placement):
            curves.ensure_component_curve(obj, c.uid)
        if c.type in ("leaf", "bract"):
            # the petiole's taper: flat (1.0 all along) until drawn
            curves.ensure_component_curve(obj, c.uid, "petiole_width")
        if c.type == 'flower':
            curves.ensure_component_flower_curves(obj, c.uid)
        _ensure_map_curves(obj, c.maps, c.uid, "imap")
        _ensure_map_curves(obj, c.growth_maps, c.uid, "fmap")
    _ensure_map_curves(obj, s.growth_maps, curves.GROWTH_MAP_OWNER, "gmap")
    _ensure_map_curves(obj, s.param_maps, curves.GROWTH_MAP_OWNER, "pmap")
    # One uid space over every level's maps (a level can move).
    _ensure_map_curves(obj, [m for lv in s.branch_levels for m in lv.maps],
                       curves.GROWTH_MAP_OWNER, LEVEL_MAP_KIND)
    curves.repaint_component_icons(obj)


def _ensure_map_curves(obj, maps, owner_uid: int, kind: str) -> None:
    """Give every map a unique uid and a curve node (created flat = no-op if it
    went missing), so draw code and the bake always find one."""
    seen = set()
    nxt = max((m.uid for m in maps), default=0) + 1
    for m in maps:
        if m.uid <= 0 or m.uid in seen:
            m.uid = nxt
            nxt += 1
        seen.add(m.uid)
        curves.ensure_map_curve(obj, owner_uid, curves.map_curve_name(kind, m.uid))


def _organ_points(obj, s, day: float):
    """Everything `regrow` builds before it writes: the plant grown to `day`,
    its stem skeleton arrays and its organ points with every attribute, the
    components' claims and drops, and the petioles with the leaf collision
    settled. Writes nothing to the object (only the module's per-plant
    records: `LAST_POINTS`, `LAST_COLLISIONS` and the caches), so
    `ensure_leaf_reference` can run it at another Day."""
    with _stage("Growth"):
        plant, profile, environment = _grow_and_prune(obj, day)

    _fc = _primary_flower_component(s)
    # A flower component is itself the "grow flowers" switch, now that the Flowers
    # panel's enable toggle is gone, and muting it turns off everything that is
    # its: flowers, pedicels, catkins and the inflorescence's own branches.
    # Without one, the plant's own flag still gates.
    show_flowers = _flowers_live(s)
    hidden_axes = (frozenset() if _fc is None
                   else frozenset(_inflorescence_axes(plant)) if not show_flowers
                   else frozenset(_inflorescence_axes(plant, _hidden_inflorescences(s))))
    # Catkins belong to the flower component: its mute (and masks, see
    # `_catkin_mask_drops`) switch them; the plant flag only gates a plant with
    # no flower component.
    show_catkins = show_flowers and (_fc is not None or s.enable_catkins)
    stem_args = (tuple(lv.show for lv in s.branch_levels), show_catkins, profile.total_steps,
                 show_flowers, hidden_axes, float(s.trunk_flare))
    cached = _STEM_ARRAYS_CACHE.get(obj.session_uid)
    if cached is not None and cached[0] is plant and cached[1] == stem_args:
        arrays = cached[2]
    else:
        arrays = _stem_arrays(
            plant, profile, show_levels=list(stem_args[0]), show_catkins=show_catkins,
            total_steps=profile.total_steps, show_pedicels=show_flowers,
            hidden_axes=hidden_axes, trunk_flare=float(s.trunk_flare))
        _STEM_ARRAYS_CACHE[obj.session_uid] = (plant, stem_args, arrays)
    # copies: the attributes are added to below (height, levels, previews)
    verts, edges = list(arrays[0]), list(arrays[1])
    stem_attrs = {name: list(values) for name, values in arrays[2].items()}
    pedicel_verts, hidden_tubes = dict(arrays[3]), arrays[4]
    # Measured here rather than with the other stats below, because the organ
    # arrays need it: `bloom_birth` is stated relative to this onset.
    onset = _flowering_age(plant, profile)
    # Bracts render when a Bract component is live; with none (or all muted)
    # there are none. `bracts` keeps their form distinct whenever a component
    # will claim them, whatever the legacy bract collection holds.
    show_bracts = any(c.type == 'bract' and not c.mute for c in s.components)
    # On a bloom calendar, Progress is the Day's place in the bloom window, so a
    # flower's birth is its birth DAY's place in that window (the step's own Day,
    # as growth reads it): one clock, and a flower opens from a bud.
    birth_window = None
    if profile.bloom_window_progress(day) is not None:
        clock = max(profile.total_steps + profile.emergence_span, 1)

        def birth_window(step):
            return profile.bloom_window_progress(profile.day_for_progress(step / clock))
    positions, organ_attrs = _organ_arrays(
        plant, bracts=show_bracts or s.bract_asset_collection is not None,
        show_leaves=s.enable_leaves, show_flowers=show_flowers,
        show_bracts=show_bracts, show_buds=_buds_live(s),
        show_fruit=_fruit_live(s), onset=onset, total_steps=profile.total_steps,
        autumn=profile.autumn_progress(day),
        hidden_axes=hidden_axes,
        hidden_organs=frozenset(_hidden_organ_ids(plant, hidden_axes)),
        birth_window=birth_window)

    # Phase 3 candidate-point pool: append along-stems generated points (leaflets,
    # thorns, moss) before the derived attributes below, so height/shade/vcolour
    # cover them too. Their instance_index bake is owned by the generating
    # component, applied in `_assign_components`.
    gen_specs = instancing._generate_placement(obj, s, plant, positions, organ_attrs, hidden_axes,
                                    profile=profile)

    # Vertex-color Alpha, and the whole-plant gradient a wind shader wants.
    # Normalised across stem *and* organs together so one number means the same
    # height on a leaf as on the stem beside it. Deliberately not per-spline
    # curve factor: with branching and multi-stem clumps, position along an
    # axis and height above the plant's base stop being the same number, and it
    # is height that is being asked for.
    _add_height_fraction(stem_attrs, verts, organ_attrs, positions)

    # Vertex-color Blue: one value for the whole object, identical on every
    # vertex, for per-instance variation once the plant is scattered. Same
    # seeded-by-string convention the parametric add-on uses for its own, so a shader
    # written against one addon behaves the same on the other.
    instance_random = random.Random(f"{s.seed}:vcolor_instance").random()
    stem_attrs["instance_random"] = [instance_random] * len(verts)
    organ_attrs["instance_random"] = [instance_random] * len(positions)

    # Shade (0 open sky, 1 fully self-shaded), lazily: only when a shade mask or
    # the Shade debug view needs it. The light field is rebuilt against the
    # finished, deformed plant here -- not reused from the last growth step --
    # so the sample is correct whatever physics and scale did to positions. A
    # plant that responds to nothing has no environment (the default); shade is
    # about its own canopy, so build a bare one just to self-shade.
    stem_spec = _STEM_ATTRS + _DERIVED_ATTRS
    if _wants_shade(s):
        shade_env = environment if environment is not None else GrowthEnvironment()
        shade_env.update_light(plant)
        stem_attrs["shade"] = (1.0 - shade_env.light_at_many(verts)).tolist()
        organ_attrs["shade"] = (1.0 - shade_env.light_at_many(positions)).tolist()
        stem_spec = stem_spec + (("shade", 'FLOAT'),)

    # Components (Phase 2) select/drop each type's points by mask + Density weight
    # and bake their master instance_index. No components of a type -> its legacy
    # single-collection path, byte-identical (no render_drop / instance_index).
    # Run before the stem mesh is written (and after shade, which masks may read)
    # so a flower the masks drop can take its stalk with it: the stalk lives in
    # the stem mesh, and without this a masked-out flower left a bare pedicel.
    # The plant's custom attributes and classes, over every organ point (grown
    # and generated), for the masks and maps below to read.
    organ_attrs.update(derived_channels(obj, positions, instancing._channels_from(organ_attrs),
                                        organ_attrs["organ_id"]))
    # Which branching level each point is on: a component takes points only on
    # the levels it is linked to (the Plant Structure tree).
    organ_attrs["level"] = structure.point_levels(plant, len(s.branch_levels),
                                        organ_attrs["branch_ordinal"]).tolist()
    # Which inflorescence each flower is: a component growing its own takes
    # only its own's flowers.
    infl_of = {int(o.id): int(o.inflorescence) for o in plant.all_organs()
               if o.kind == OrganKind.FLOWER}
    organ_attrs["inflorescence"] = [
        infl_of.get(int(oid), 0) if int(k) == 1 else 0
        for oid, k in zip(organ_attrs["organ_id"], organ_attrs["kind"], strict=True)]
    instancing._assign_components(obj, s, organ_attrs, gen_specs)
    dropped = set()
    if "render_drop" in organ_attrs and pedicel_verts:
        dropped |= {int(oid) for oid, k, d in zip(
            organ_attrs["organ_id"], organ_attrs["kind"], organ_attrs["render_drop"],
            strict=True) if k == 1 and d > 0.5}
    if show_catkins and pedicel_verts:
        # Catkins are curves in this mesh, not organ points, so the flower
        # component's masks are evaluated for them here -- on the same 0-1
        # height scale `_add_height_fraction` gave everything above.
        zs = [v[2] for v in verts] + [p[2] for p in positions]
        low = min(zs) if zs else 0.0
        span = (max(zs) - low) if zs else 0.0

        def height_of(p, low=low, span=span):
            return (float(p[2]) - low) / span if span > 0.0 else 0.0
        dropped |= _catkin_mask_drops(obj, plant, pedicel_verts, height_of,
                                      profile.total_steps)
    catkin_ids = ({int(o.id) for o in plant.all_organs() if o.kind == OrganKind.CATKIN}
                  if pedicel_verts else set())
    if catkin_ids & set(pedicel_verts):
        # The catkins' vertices (stalk and bodies): the mesher gives their tubes
        # an organ's resolution, and GN dresses them in a Catkin Material. Only
        # on a plant with catkins -- every other plant's stem mesh (and its
        # cache key) stays as it was.
        flag = [0] * len(verts)
        for oid in catkin_ids:
            for v in pedicel_verts.get(oid, ()):
                flag[v] = 1
        stem_attrs["catkin"] = flag
        stem_spec = stem_spec + (("catkin", 'INT'),)
    if dropped:
        gone = {v for oid in dropped for v in pedicel_verts.get(oid, ())}
        verts, edges, stem_attrs = _drop_stem_verts(verts, edges, stem_attrs, gone)
    # Petioles: a leaf component's stalks, for the points it drew (claims and
    # drops are settled just above), into the stem mesh beside the pedicels.
    petioles._add_petioles(obj, s, plant, positions, organ_attrs, verts, edges, stem_attrs,
                  hidden_tubes)
    return SimpleNamespace(plant=plant, profile=profile, environment=environment,
                           verts=verts, edges=edges, stem_attrs=stem_attrs,
                           stem_spec=stem_spec, hidden_tubes=hidden_tubes,
                           positions=positions, organ_attrs=organ_attrs, onset=onset)


# -- Regrow timings (Debug Tools > Timings) ------------------------------------
# Where a plant's last regrow spent its time, stage by stage, for the Timings
# sub-panel: nothing reads these back, and a plant never timed has no entry.
# Live update's background work (growth, leaf collision in the compute worker)
# is measured from submit to landing and folded into the regrow that lands it.
_TIMINGS: dict = {}          # session_uid -> {"stages": [(name, s)], "total": s, "worker": bool}
_T_WORKER: dict = {}         # session_uid -> {stage: s}, waiting for its regrow
_T_RUN: list = []            # the running regrow's stage dict (a stack: regrow may nest)


class _stage:
    """Time a regrow stage: `with _stage("Leaf collision"): ...`. Adds to the
    running regrow's record; outside a regrow it times nothing."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _T_RUN:
            d = _T_RUN[-1]
            d[self.name] = d.get(self.name, 0.0) + time.perf_counter() - self.t
        return False


def regrow_timings(obj) -> dict | None:
    """The last regrow's timings for this plant (`_TIMINGS`), or None."""
    try:
        return _TIMINGS.get(obj.session_uid)
    except ReferenceError:
        return None


def _worker_time(uid, name: str, seconds: float) -> None:
    w = _T_WORKER.setdefault(uid, {})
    w[name] = w.get(name, 0.0) + seconds


def regrow(obj, weld: str = WELD_LATER) -> None:
    """Grow and write the plant (`_regrow`), timing its stages for Debug Tools
    > Timings (`regrow_timings`)."""
    t0 = time.perf_counter()
    _T_RUN.append({})
    try:
        _regrow(obj, weld)
    finally:
        stages = _T_RUN.pop()
        try:
            uid = obj.session_uid
        except ReferenceError:
            uid = None
        if uid is not None:
            total = time.perf_counter() - t0
            nested = stages.pop("_organ_points", 0.0)
            inner = stages.get("Growth", 0.0) + stages.get("Leaf collision", 0.0)
            stages["Organ placement"] = max(0.0, nested - inner)
            stages["Other (settings, stats, modifier)"] = max(
                0.0, total - nested - stages.get("Stem mesh", 0.0)
                - stages.get("Organ points write", 0.0))
            worker = _T_WORKER.pop(uid, {})
            for name, sec in worker.items():
                stages[name] = stages.get(name, 0.0) + sec
            _TIMINGS[uid] = {"stages": sorted(stages.items(), key=lambda kv: -kv[1]),
                             "total": total + sum(worker.values()), "worker": bool(worker)}


def _regrow(obj, weld: str = WELD_LATER) -> None:
    """Grow the plant from its settings and write it. `weld` says what happens
    to a stem weld that is not cached yet (`_write_built_stems`): `WELD_LATER`,
    the default -- every interactive edit -- leaves it to the Weld button and
    shows the unwelded preview; `WELD_BACKGROUND` (the Weld button) has a
    worker build it; `WELD_NOW` builds it before returning -- for anything
    that reads the finished plant (bakes, timelapse frames, measuring or
    exporting from a script)."""
    s = obj.plantgen
    # A regrow shows the live plant: never a recorded timelapse frame (the feed
    # below pushes the Timelapse Cache sockets off).
    obj["plantgen_tl_showing"] = False
    _prepare_component_curves(obj, s)
    _seed_flower_components(obj, s)
    _seed_leaf_components(s)
    _seed_level_stiffness(s)
    _seed_bud_components(obj, s)
    _seed_fruit_components(obj, s)
    from .properties import sync_variation_starts
    sync_variation_starts(s.variations)          # Variations' Day axis starts at this Day
    # Under Grow Over Timeline, leaf collision comes from one reference Day.
    petioles.ensure_leaf_reference(obj)
    with _stage("_organ_points"):
        pts = _organ_points(obj, s, float(s.day))
    plant, profile, environment, onset = pts.plant, pts.profile, pts.environment, pts.onset
    verts, edges, stem_attrs, stem_spec = pts.verts, pts.edges, pts.stem_attrs, pts.stem_spec
    hidden_tubes, positions, organ_attrs = pts.hidden_tubes, pts.positions, pts.organ_attrs

    # Mask Preview debug view: colour the WHOLE plant -- stem and organs -- by the
    # active map's window, a red->green gradient that slides with Min/Max. The stem
    # carries the same channel value (height, shade) so the gradient runs unbroken
    # base to tip, not just over the organs.
    want_preview = s.debug_view_mode == 'mask_preview'
    if want_preview:
        stem_attrs["mask_preview"] = instancing._preview_values(s, stem_attrs, len(verts), plant)
        stem_spec = stem_spec + (("mask_preview", 'FLOAT'),)

    if verts:
        # Which level's cross-section each vertex's tube takes (a tree of
        # levels: a level is no longer its depth).
        stem_attrs["level"] = structure._section_levels(plant, profile, stem_attrs)
        stem_spec = stem_spec + (("level", 'INT'),)

    mesh = obj.data
    if verts:
        # The stems are swept -- and joined, unless Weld Junctions is off -- here,
        # in Python (geometry.stem_mesher); the GN Stem Tube passes this mesh
        # through.
        with _stage("Stem mesh"):
            if _write_built_stems(obj, mesh, verts, edges, stem_attrs, stem_spec, profile,
                                  hidden_tubes, weld=weld):
                mesh.update()
    else:
        mesh.clear_geometry()
        obj.pop(_STEM_WRITTEN_KEY, None)
        obj.pop(_STEM_PREVIEW_KEY, None)
        mesh.from_pydata(verts, edges, [])
        _write_attributes(mesh, stem_attrs, stem_spec)
        mesh.update()

    organs_obj = _ensure_organs_object(obj)
    organs_mesh = organs_obj.data
    organs_mesh.clear_geometry()
    organs_mesh.from_pydata(positions, [], [])
    # Instancing masks (render-only): decide which organ points survive, per
    # component, and bake one `render_drop` gate a single GN Delete Geometry
    # reads. Added to the spec only when a mask is active, so an unmasked plant
    # writes exactly what it always did (byte-identical no-op).
    organ_spec = _ORGAN_ATTRS + _DERIVED_ATTRS
    if "shade" in organ_attrs:
        organ_spec = organ_spec + (("shade", 'FLOAT'),)
    if want_preview:
        organ_attrs["mask_preview"] = instancing._preview_values(s, organ_attrs, len(positions))
        organ_spec = organ_spec + (("mask_preview", 'FLOAT'),)
    if s.debug_view_mode == 'component':
        # Written only while that view is on, so a plant without it is
        # byte-identical.
        organ_attrs["component_color"] = instancing._component_colours(obj, s, len(positions))
        organ_spec = organ_spec + (("component_color", 'FLOAT_COLOR'),)
    if "render_drop" in organ_attrs:
        organ_spec = organ_spec + (("render_drop", 'FLOAT'),)
    if gn_mesh.LEAF_CURL_ATTR in organ_attrs:
        organ_spec = organ_spec + ((gn_mesh.LEAF_CURL_ATTR, 'FLOAT'),
                                   (gn_mesh.LEAF_FOLD_ATTR, 'FLOAT'))
    if "instance_index" in organ_attrs:
        organ_spec = organ_spec + (("instance_index", 'INT'),)
    if "flower_block_offset" in organ_attrs:
        organ_spec = organ_spec + (("flower_block_offset", 'INT'),
                                   ("flower_block_max", 'INT'))
    with _stage("Organ points write"):
        _write_attributes(organs_mesh, organ_attrs, organ_spec)
        organs_mesh.update()

    # Written *before* the modifier is fed, not with the other stats below:
    # Bloom's From Age mode derives Progress from this, and the socket values
    # are pushed inside `ensure_gn_modifier`. Setting it afterwards would leave
    # every regrow one species behind.
    s.last_flowering_age = onset
    # The calendar bloom window's position at this Day (phase 3), or -1 when the
    # species has no bloom calendar. From Age reads it in preference to maturity,
    # so a calendar species' bloom opens across its own window. A day-of-year
    # concept, so it is the Day that drives it -- carried on the object because
    # `effective_bloom_progress` sees only the property group, not the profile.
    win = profile.bloom_window_progress(s.day)
    s.last_bloom_window = -1.0 if win is None else win
    # The Mapping list and the level swatches follow whatever the regrow saw
    # (a preset, an undo, a +/- edited elsewhere).
    mapping.ensure_level_colours(s)
    mapping.rebuild_rows(s)

    gn_mesh.ensure_gn_modifier(obj, organs_obj)

    # The inspector overlay's snapshot. Taken here rather than on demand
    # because `plant` is a local and is about to go out of scope -- there is
    # nothing left to look up afterwards.
    overlay.capture(obj, plant)

    s.last_phytomers = plant.phytomer_count
    s.last_axes = len(plant.axes)
    # Bracts are leaves to everything downstream -- same organ kind, same
    # instancing branch, same deformers -- but reporting them as foliage would
    # make "Leaves" jump the moment Bract Size leaves zero, for a change that
    # added no foliage at all. Counted apart here and only here.
    all_leaves = plant.organs_of_kind(OrganKind.LEAF)
    s.last_bracts = sum(1 for leaf in all_leaves if leaf.form == LeafForm.BRACT)
    # Catkins, not bunches: one organ carries a whole stalk-full, and the
    # number anyone wants to see is how many catkins are on the plant.
    s.last_catkins = sum(max(1, len(o.catkin_bodies))
                         for o in plant.organs_of_kind(OrganKind.CATKIN))
    s.last_leaves = len(all_leaves) - s.last_bracts
    s.last_flowers = len(plant.organs_of_kind(OrganKind.FLOWER))
    # Calendar phase 4: flowers that have set fruit (data only -- not drawn until
    # the M2 fruit render). Reported so the fruit lifecycle is visible as a
    # count even though nothing is instanced for it.
    s.last_fruit = len(plant.organs_of_kind(OrganKind.FRUIT))
    s.last_flower_spacing = _flower_spacing(plant)
    s.last_solid_voxels = env_bridge.solid_voxels(environment)
    s.last_height, s.last_width = _visible_extent(plant, positions)
    _write_diagnostics(s, plant)

    # Each stem's own centre of mass -- what that stem leans toward -- stored flat
    # (x, y, z per stem) so the markers can be toggled without a rebuild, and
    # refreshed here so they track the plant last grown.
    coms = stem_centers_of_mass(plant, profile)
    obj.pop("plantgen_com", None)               # (the one whole-plant point, retired)
    if coms:
        obj[_COMS_KEY] = [float(c) for com in coms for c in com]
    else:
        obj.pop(_COMS_KEY, None)
    _sync_com_empty(obj)

    # Stamped last, once every input has been read and the mesh written, so the
    # stamp describes the plant that is actually there.
    stamp(obj)
    # Whatever asked for this grow, the plant is now up to date -- so drop any
    # pending debounce rather than letting it fire a second, identical grow.
    # Imported here because `live` imports this module at load time.
    from . import live

    live.clear_dirty(obj.name)
    live.log_event("regrow", obj)


def _visible_extent(plant, organ_positions) -> tuple[float, float]:
    """How big the plant reads, for the panel.

    `Plant.bounds()` covers the *skeleton* only -- phytomer tips and basal
    points -- which is right for the structural checks, since they ask about
    axis geometry. It is wrong as a reported size: a single unbranched stem
    with a big basal rosette measured 1.6 m tall and 1 mm wide, because
    nothing but the stem itself was counted. Organs are included here without
    touching `bounds()`, so the checks are unaffected.

    Still attachment points rather than realized blade geometry -- a leaf's
    mesh extends past its own attachment and only Geometry Nodes knows how far.
    Under-reporting slightly beats reporting a millimetre.
    """
    if not plant.phytomer_count:
        return 0.0, 0.0
    lo, hi = plant.bounds()
    lo, hi = list(lo), list(hi)
    for p in organ_positions:
        for axis in range(3):
            lo[axis] = min(lo[axis], p[axis])
            hi[axis] = max(hi[axis], p[axis])
    return float(hi[2] - lo[2]), float(max(hi[0] - lo[0], hi[1] - lo[1]))


def _flowering_age(plant, profile) -> float:
    """The Age at which this plant actually made its first flower.

    Measured off the grown plant, not read off `flowering_onset` -- and that
    correction is the whole reason From Age works at all. `flowering_onset` is
    a *permission* gate, not a schedule: its own docstring says it is
    deliberately "fully open long before any axis completes its node count",
    and the default reaches 1.0 by Age 0.25 and leaves zero at 0.035. Deriving
    a bloom start from it stretched the bloom across essentially the plant's
    whole life, so Progress tracked Age almost exactly and every plant at the
    default Age of 1.0 sat at Progress 1.0 -- every flower fully open, always,
    which is exactly what it looked like.

    What really decides when a plant flowers is structural: an axis flowers
    once it has laid down its level's Steps. There is no curve to
    invert for that, but the grown plant already knows the answer, so ask it.
    `birth_step / total_steps` is the age coordinate every curve in
    `development` is sampled on, so this is in the same units as Age by
    construction.

    A plant too young to have flowered yet returns 1.0, which gives Progress 0
    -- all buds, which is correct rather than a fallback.
    """
    flowers = plant.organs_of_kind(OrganKind.FLOWER)
    if not flowers:
        return 1.0
    first = min(organ.birth_step for organ in flowers)
    return min(1.0, max(0.0, first / max(profile.total_steps, 1)))


def _write_diagnostics(s, plant) -> None:
    """Counts the viewport cannot show. A bud that never released produces no
    geometry, so no amount of attribute colouring will reveal it -- it only
    ever appears as a number."""
    counts = Counter()
    for axis in plant.axes.values():
        if axis.apex is not None:
            counts[axis.apex.state] += 1
        for phytomer in axis.phytomers:
            for bud in phytomer.buds:
                counts[bud.state] += 1

    s.last_buds_active = counts[BudState.ACTIVE]
    s.last_buds_dormant = counts[BudState.DORMANT]
    s.last_buds_released = counts[BudState.RELEASED]
    s.last_buds_aborted = counts[BudState.ABORTED]

    axes_by_order = Counter(axis.order for axis in plant.axes.values())
    nodes_by_order = Counter()
    for axis in plant.axes.values():
        nodes_by_order[axis.order] += len(axis.phytomers)
    s.last_axes_by_order = "  ".join(
        f"{order}: {axes_by_order[order]}" for order in sorted(axes_by_order))
    s.last_phytomers_by_order = "  ".join(
        f"{order}: {nodes_by_order[order]}" for order in sorted(nodes_by_order))
    s.last_flowering_axes = sum(
        1 for axis in plant.axes.values() if axis.flowering_from >= 0)

    # WS3: what senescence and pruning removed. A SHED item produces no
    # geometry, so it is a number here or it is invisible -- the exact reason
    # the bud counts above exist. Dead branches keep their wood (retained);
    # organs are only ever ALIVE or SHED, never DEAD (a dead branch is bare).
    s.last_dead_axes = sum(1 for a in plant.axes.values()
                           if a.lifecycle == LifecycleState.DEAD)
    s.last_shed_axes = sum(1 for a in plant.axes.values()
                           if a.lifecycle == LifecycleState.SHED)
    s.last_shed_organs = sum(1 for o in plant.all_organs()
                             if o.lifecycle == LifecycleState.SHED)

    # A regrow invalidates whatever the checks last said about this plant.
    s.checks.clear()
    s.checks_ran = False


def ensure_asset_folder(obj, pointer_attr: str, suffix: str) -> None:
    """Create and wire a named asset subcollection for a feature this plant has
    turned on but has no folder for yet.

    On demand -- called from `regrow` when the feature is enabled -- so a plant
    that never uses the feature never spawns an empty folder for it. Runs in
    regrow's own safe context, where `_ensure_organs_object` already creates
    datablocks; Blender warns against creating datablocks in property update
    callbacks, so this is deliberately not one.

    Idempotent: does nothing once the pointer is set (whether by this or by the
    user), so it fires at most once per feature per plant.
    """
    s = obj.plantgen
    if getattr(s, pointer_attr) is not None:
        return
    home = obj.users_collection[0] if obj.users_collection else None
    if home is None:
        # No home collection to nest under -- nothing sensible to do, and the
        # panel's "No collection" hint still points the way.
        return
    sub = bpy.data.collections.new(f"{obj.name}_{suffix}")
    home.children.link(sub)
    setattr(s, pointer_attr, sub)


DEFAULT_PRESET = "rudbeckia_fulgida"
"""What Add Plant grows (user decision 2026-10-31, replacing the Generic
perennial): a full 1-to-1 preset -- settings, curves, components and assets --
so a new plant is a finished, real one."""


def create_plant(context, name: str = "Plant",
                 preset: str | None = DEFAULT_PRESET) -> bpy.types.Object:
    """Create a plant and its named collection structure.

    Everything for one plant lives under a `<Name>_Plant` collection: the plant
    object `<Name>` and its `<Name>_Organs` instancing helper (added by regrow).
    By default it is the `preset` applied (`DEFAULT_PRESET`), whose components
    draw from the preset's own appended assets. `preset=None` -- or a preset
    that cannot be found -- grows the bare plant instead (the code's generic
    profile; the doc tooling stages on it): `<Name>_Leafassets` /
    `<Name>_Flowerassets` folders the user drops meshes into, wired to the
    plant's own pointers, so it renders a bare stem until they are filled.

    Bud folders are created on demand (see `ensure_asset_folder`), so a plant
    that never uses them spawns no empty folder. Bracts deliberately get **no**
    on-demand folder: with a size but no bract collection they render from the
    Leaf collection, and auto-creating an empty folder would replace that
    graceful fallback with nothing -- so a distinct bract mesh stays an explicit
    manual choice.

    Name collisions fall back to Blender's own `.001` de-duplication.
    """
    name = (name or "Plant").strip() or "Plant"

    plant_col = bpy.data.collections.new(f"{name}_Plant")
    # At the scene root, not the active collection: a plant is a top-level thing
    # and its <Name>_Plant group landing wherever a collection happens to be
    # active in the Outliner is surprising. (User's call, 2026-08-30.)
    context.scene.collection.children.link(plant_col)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    obj["plantgen"] = True
    plant_col.objects.link(obj)
    obj.location = context.scene.cursor.location

    from . import presets
    entry = presets.find_preset(preset) if preset else None

    s = obj.plantgen
    if entry is None:
        # The bare plant: wire the two near-universal asset folders up front.
        # `_ensure_organs_object` (in regrow) links `<Name>_Organs` into
        # `obj.users_collection[0]`, which is this same `<Name>_Plant`, so the
        # whole plant stays in one collection.
        for suffix, attr in (("Leafassets", "leaf_asset_collection"),
                             ("Flowerassets", "flower_asset_collection")):
            sub = bpy.data.collections.new(f"{name}_{suffix}")
            plant_col.children.link(sub)
            setattr(s, attr, sub)

    for other in context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj

    if entry is not None:
        # The full preset, exactly as one picked from the Library.
        presets.apply_preset_file(obj, entry["path"])
    else:
        # The code's generic profile, stamped as the self-contained base.
        generic = species_lib.profile("GENERIC")
        stash_profile(obj, "Generic perennial", generic, preset_id="GENERIC")
        apply_profile(obj, generic)

    # Born on the day system already (Day defaults to a mature 270), so the
    # legacy-Age day migration never touches it on a later load.
    s.calendar_migrated = True

    if entry is None:
        _add_default_flower_component(s)
    regrow(obj)
    return obj


DEFAULT_FLOWER_NAME = "Replace Me"


def _add_default_flower_component(s) -> None:
    """A new plant's flower component (user decision 2026-10-16): growth only
    flowers where a flower component is placed, so without one a mesh dropped
    in <Name>_Flowerassets drew nothing. On the Plant row (every level's tips),
    drawing from that folder, named to be replaced. Made after the preset is
    applied, like the + button's, so it takes the preset's flower settings at
    the regrow."""
    from . import mapping, structure
    c = s.components.add()
    c.type = 'flower'
    c.color = mapping.next_component_colour(s)
    c.uid = max((x.uid for x in s.components), default=0) + 1
    c.collection = s.flower_asset_collection
    c.name_override = DEFAULT_FLOWER_NAME
    structure.place(s, 0, c.uid)                 # 0: the Plant row, every level
    s.components_active = len(s.components) - 1
    structure.rebuild_tree(s)
