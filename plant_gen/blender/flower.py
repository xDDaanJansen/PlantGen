"""Controller: standalone flower objects — the FlowerGen "flower workshop".

A flower built here is independent of any stem: `create_flower` makes a new
Collection containing the flower's objects, so the result can be saved as a
reusable asset and dropped straight into an existing stem's Flowers.asset_collection
(the existing collection-instancing path in `plant.py` needs no changes to consume
it). State lives on the main object itself
(`obj.plantgen_flower` + its own curve node group), mirroring this package's stem
controller (`plant.py`) — a separate file rather than folding into `plant.py`
because a standalone flower is a fully independent object lifecycle (no stem, no
attachment points, no age/branch_id/generation concerns).

Four components, each independently parameterized (see `skeleton/flower.py`):
Petals and Sepals (the calyx) are both "ribbon" appendages -- a poly spline whose
native per-point `radius` is repurposed as width, meshed via a flat arc profile
then bowed per-vertex post-mesh (its own per-component Cup, see `_apply_cup`);
the Disc and Receptacle are both "revolved
body" profiles -- a poly spline whose native per-point `radius` is repurposed as
radius-vs-height, meshed via a circle profile. Petals live on the main object's
own curve data; Sepals/Disc/Receptacle are each a SEPARATE child object, since one
`Curve to Mesh` call can only apply one profile to everything it meshes, and each
of the four needs its own -- see `_build_flower_group`. All four read their native
per-point radius into `Curve to Mesh`'s own `Scale` input -- the identical
"hard-won" recipe `plant.py`'s `_build_mesh_group` already uses for the stem
(bypassing `Set Curve Radius`, proven inert for legacy curves in this Blender
version). Petal/Sepal twist rides on the spline's native per-point `tilt`
(Blender's built-in mechanism for rotating a swept profile's frame along a curve,
respected automatically by `Curve to Mesh` -- no extra GN wiring needed) rather
than a hand-rolled transported frame.

Built flat in local XY (petals/sepals start at the rim, disc growing along +Z,
receptacle growing along -Z below the rim) to match the existing leaf/flower
attachment rotation convention in `plant._write_attachment_points` (local +Y =
forward/direction, +Z = face normal) -- so a flower built here drops into an
existing stem's Flowers slot with correct orientation, with no fixup, whenever
that integration is wired. `build_flower_mesh`'s own final pass then shifts
every point up so local (0,0,0) -- what actually lands on a stem's attachment
point once instanced -- sits at the receptacle's own lowest point (the rim
itself, unshifted, when there's no receptacle) rather than at the rim, so the
flower's true base is what touches down, not somewhere mid-flower.
"""

import dataclasses
import math
import random
from contextlib import contextmanager

import bpy
from mathutils import Matrix

from ..growth.flower import (
    _RIBBON_PROFILE_RESOLUTION,
    DOME_CAP,
    FlowerMeshParams,
    FlowerSkeleton,
    PetalSpine,
    build_flower_mesh,
)
from ..growth.flower_params import Ramp
from . import gn_layout

_FLOWER_TAG = "plantgen_flower"
_DISC_TAG = "plantgen_flower_disc"
_SEPAL_TAG = "plantgen_flower_sepal"
_RECEPTACLE_TAG = "plantgen_flower_receptacle"
_PETAL_FUSION_TAG = "plantgen_flower_petal_fusion"
_SEPAL_FUSION_TAG = "plantgen_flower_sepal_fusion"
_PETAL_TOP_SHAPE_TAG = "plantgen_flower_petal_top_shape"
_PETAL_BOTTOM_SHAPE_TAG = "plantgen_flower_petal_bottom_shape"
_SEPAL_TOP_SHAPE_TAG = "plantgen_flower_sepal_top_shape"
_SEPAL_BOTTOM_SHAPE_TAG = "plantgen_flower_sepal_bottom_shape"
_FLOWER_GROUP_NAME = "PlantGen_Flower"
_SUB_PREFIX = "PlantGen Flower \u00b7 "   # its stages' sub-groups
_DISC_POINTS_SOCKET = "Disc Points"
_PETAL_FUSION_POINTS_SOCKET = "Petal Fusion Points"
_SEPAL_FUSION_POINTS_SOCKET = "Sepal Fusion Points"
_PETAL_CUP_SOCKET = "Petal Cup"
_PETAL_WIDTH_REF_SOCKET = "Petal Width Ref"
_DISC_CONCAVE_SOCKET = "Disc Concave"
_SEPAL_POINTS_SOCKET = "Sepal Points"
_SEPAL_CUP_SOCKET = "Sepal Cup"
_SEPAL_WIDTH_REF_SOCKET = "Sepal Width Ref"
_RECEPTACLE_POINTS_SOCKET = "Receptacle Points"
_PETAL_TOP_SHAPE_POINTS_SOCKET = "Petal Top Shape Points"
_PETAL_BOTTOM_SHAPE_POINTS_SOCKET = "Petal Bottom Shape Points"
_SEPAL_TOP_SHAPE_POINTS_SOCKET = "Sepal Top Shape Points"
_SEPAL_BOTTOM_SHAPE_POINTS_SOCKET = "Sepal Bottom Shape Points"
_PETAL_RESOLUTION_SOCKET = "Petal Resolution"
_SEPAL_RESOLUTION_SOCKET = "Sepal Resolution"
# Tube weld (Fuse To Tube): whether the whorl's lobes are welded into its
# tube, how many lobes its first layer has, and how many sides the tube is
# revolved with -- `count * (_RIBBON_PROFILE_RESOLUTION - 1)` while welding
# (one tube-rim vertex per lobe base vertex, neighbours sharing their edge
# vertex), else the plain 16 every revolved body uses.
_PETAL_WELD_SOCKET = "Petal Tube Weld"
_SEPAL_WELD_SOCKET = "Sepal Tube Weld"
_PETAL_COUNT_SOCKET = "Petal Count"
_SEPAL_COUNT_SOCKET = "Sepal Count"
_PETAL_TUBE_SIDES_SOCKET = "Petal Tube Sides"
_SEPAL_TUBE_SIDES_SOCKET = "Sepal Tube Sides"
# Receptacle weld: with both the receptacle and the corolla tube matching the
# disc radius, their rims share one circle -- the receptacle is revolved with
# the tube's side count and welded onto the tube's base ring (`_base_weld`).
_RECEPTACLE_WELD_SOCKET = "Receptacle Tube Weld"
_RECEPTACLE_SIDES_SOCKET = "Receptacle Sides"
_TUBE_SIDES = 16
# Resolution: points ACROSS a petal / sepal (its swept profile; along it is
# Resolution) and the sides of the disc (the tubes' and receptacle's ride on
# their Sides sockets above).
_PETAL_WIDTH_RES_SOCKET = "Petal Width Resolution"
_SEPAL_WIDTH_RES_SOCKET = "Sepal Width Resolution"
_DISC_SIDES_SOCKET = "Disc Sides"
_PETAL_FLIP_SOCKET = "Petals Flip Normals"
_SEPAL_FLIP_SOCKET = "Sepals Flip Normals"
_DISC_FLIP_SOCKET = "Disc Flip Normals"
_RECEPTACLE_FLIP_SOCKET = "Receptacle Flip Normals"
_PETAL_TUBE_FLIP_SOCKET = "Corolla Tube Flip Normals"
_SEPAL_TUBE_FLIP_SOCKET = "Calyx Tube Flip Normals"
# UVs: a welded whorl is unwrapped as one piece -- the tube below V = split,
# its lobes above it (split = the tube's share of the tube + lobe length).
# The flower's material (flower_material.py) and the UV checker override.
_FLOWER_MAT_SOCKET = "Flower Material"
_CHECKER_SOCKET = "UV Checker"
_CHECKER_MAT_SOCKET = "Checker Material"
# UVs: the flower flattened (`_uv_layout`) -- each part's per-ring record on
# a carrier curve (point x/y/z + radius) and the centre of the medallion it is
# laid out in, all at one scale (equal texel density everywhere).
_UV_PARTS = ("disc", "receptacle", "petal_tube", "sepal_tube", "petals", "sepals")
_UV_CARRIER_SOCKETS = {p: f"UV {p.replace('_', ' ').title()} Carrier" for p in _UV_PARTS}
_UV_CENTER_SOCKETS = {p: f"UV {p.replace('_', ' ').title()} Centre" for p in _UV_PARTS}
_UV_SCALE_SOCKET = "UV Scale"
_UV_TAGS = {p: f"plantgen_flower_uv_{p}" for p in _UV_PARTS}
_SLOT_TIE = 1e-3
"""Added to the tube weld's turn (in slots) before it is rounded, so a lobe
sitting exactly half a slot between two rim vertices always takes the same one
(`_tube_weld`)."""

_WELD_DISTANCE = 1e-5
"""Merge distance for the welded rim: the snapped lobe and tube vertices sit
~1e-7 apart (the same angle, two float paths); neighbouring rim vertices are
~0.2 mm apart on a small flower."""

# The Bloom slider's UI floor: live-tuning found the bud pose at raw bloom 0.0
# too knife-edged, and 0.07 reads as the better "closed" extreme -- rather than
# just moving the default, the whole 0..1 UI range is remapped onto raw bloom
# [_BLOOM_UI_FLOOR, 1.0] (same affine-remap idea as `FLOWER_CURVE_SPECS`'
# `y_remap`, just for this one scalar instead of a curve), so typing 0 in the
# widget IS the good closed pose and 1 is still fully open. `skeleton/flower.py`
# never sees this -- its own `bloom` field stays the plain raw 0..1 value.
_BLOOM_UI_FLOOR = 0.07


def _bloom_ui_to_raw(ui_bloom: float) -> float:
    return _BLOOM_UI_FLOOR + ui_bloom * (1.0 - _BLOOM_UI_FLOOR)


def _bloom_raw_to_ui(raw_bloom: float) -> float:
    return (raw_bloom - _BLOOM_UI_FLOOR) / (1.0 - _BLOOM_UI_FLOOR)

# Every flower-domain curve widget: (key, panel label, X-axis label, Y-axis
# label, default control points, shape-preset library, Y-remap) -- mirrors
# `plant.CURVE_SPECS`'s row shape; `library` picks which shape-preset menu
# `ui_flower.py` shows for this curve ("petal"/"disc" botanical shape libraries
# in `operators.py`, or "generic" for the stem's original abstract shapes --
# see `operators.FLOWER_SHAPE_LIBRARIES`). Most curves ramp over the petal's own
# position along its length (`"0 (base) -> 1 (tip)"`); the `_age_profile`
# curves instead ramp over this petal's Age (`"0 (young) -> 1 (mature)"`, see
# `skeleton/flower.py`'s `petal_layers` docstring) -- genuinely different X
# axes, so each spec carries its own caption rather than a hardcoded one. Sepals
# reuse the exact same curve shapes/library choices as their Petal counterparts
# (they're the same kind of appendage, see skeleton/flower.py's module docstring).
#
# `y_remap`, when not None, is an `(eval_at_0, eval_at_1)` pair: the WIDGET
# itself always edits/displays in a plain 0..1 Y range (so its grid/zoom stays
# at Blender's normal scale, editable), and `flower_params_from_object` affinely
# remaps that 0..1 edit value to `[eval_at_0, eval_at_1]` before it's used to
# build geometry. Only the two Pitch Age Profile curves need this today --
# their multiplier has to reach -1 (to flip Pitch's sign for mature/bottom
# layers), and editing a curve whose Y axis spans -1..1 in the widget itself
# made the built-in curve grid/zoom too coarse to place points precisely (the
# widget's navigation is scaled to its own clip range) -- see git history for
# the abandoned "widen the node's clip_min_y/max_y" attempt at fixing that,
# which fixed the corruption but not the awkward editing.
_LENGTH_X = "0 (base) → 1 (tip)"
_AGE_X = "0 (young) → 1 (mature)"
_TIP_ZONE_X = "0 (tip zone start) → 1 (tip)"
_WIDTH_X = "0 (one edge) → 1 (other edge)"
_WHORL_X = "0 (first petal) → 1 (opposite it)"
_CAP_Y = "Cap shape (scaled by the Top/Bottom Shape amount)"
FLOWER_CURVE_SPECS = [
    ("maturity_scale_profile", "Maturity Scale Profile", _AGE_X, "Scale multiplier",
     [(0.0, 0.8), (1.0, 1.0)], "generic", None),
    ("petal_width_profile", "Petal Width Profile", _LENGTH_X, "Width multiplier",
     [(0.0, 0.3), (0.35, 1.0), (1.0, 0.0)], "petal", None),
    ("petal_tube_profile", "Petal Tube Profile", "0 (rim) → 1 (where lobes start)",
     "Radius multiplier", [(0.0, 1.0), (1.0, 1.0)], "corolla", None),
    ("petal_scale_age_profile", "Petal Scale Age Profile", _AGE_X, "Scale multiplier",
     [(0.0, 0.8), (1.0, 1.0)], "generic", None),
    ("petal_curl_updown_profile", "Petal Curl Up/Down Profile", _LENGTH_X,
     "Curl Up/Down multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_curl_updown_age_profile", "Petal Curl Up/Down Age Profile", _AGE_X,
     "Curl Up/Down multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_curl_leftright_profile", "Petal Curl Left/Right Profile", _LENGTH_X,
     "Curl Left/Right multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_curl_outin_profile", "Petal Curl Out/In Profile", _LENGTH_X,
     "Curl Out/In multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_gravity_profile", "Petal Gravity Profile", _LENGTH_X, "Gravity multiplier",
     [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_twist_profile", "Petal Twist Profile", _LENGTH_X, "Twist multiplier",
     [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_top_shape_profile", "Petal Top Shape Profile", _WIDTH_X,
     _CAP_Y, DOME_CAP, "cap", None),
    ("petal_bottom_shape_profile", "Petal Bottom Shape Profile", _WIDTH_X,
     _CAP_Y, DOME_CAP, "cap", None),
    ("petal_whorl_length_profile", "Petal Length Around the Whorl", _WHORL_X,
     "Length multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_whorl_width_profile", "Petal Width Around the Whorl", _WHORL_X,
     "Width multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_whorl_pitch_profile", "Petal Pitch Around the Whorl", _WHORL_X,
     "Pitch multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("petal_rotation_pitch_age_profile", "Petal Pitch Age Profile", _AGE_X,
     "0 (multiplier −1) → 1 (multiplier +1)", [(0.0, 1.0), (1.0, 1.0)], "generic",
     (-1.0, 1.0)),
    ("petal_bud_curl_profile", "Petal Bud Curl Profile", _LENGTH_X,
     "Bud Curl multiplier", [(0.0, 1.0), (0.5, 1.1), (1.0, 1.5)], "generic", None),
    ("petal_bud_tip_height_profile", "Petal Bud Tip Height Profile", _TIP_ZONE_X,
     "Height multiplier", [(0.0, 0.0), (0.3, 0.06), (0.6, 0.3), (1.0, 1.0)], "generic", None),
    ("petal_bud_tip_radius_profile", "Petal Bud Tip Radius Profile", _TIP_ZONE_X,
     "Radius multiplier", [(0.0, 1.0), (1.0, 1.0)], "disc", None),
    ("sepal_width_profile", "Sepal Width Profile", _LENGTH_X, "Width multiplier",
     [(0.0, 0.3), (0.35, 1.0), (1.0, 0.0)], "petal", None),
    ("sepal_tube_profile", "Sepal Tube Profile", "0 (rim) → 1 (where lobes start)",
     "Radius multiplier", [(0.0, 1.0), (1.0, 1.0)], "corolla", None),
    ("sepal_scale_age_profile", "Sepal Scale Age Profile", _AGE_X, "Scale multiplier",
     [(0.0, 0.8), (1.0, 1.0)], "generic", None),
    ("sepal_curl_updown_profile", "Sepal Curl Up/Down Profile", _LENGTH_X,
     "Curl Up/Down multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("sepal_curl_updown_age_profile", "Sepal Curl Up/Down Age Profile", _AGE_X,
     "Curl Up/Down multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("sepal_curl_leftright_profile", "Sepal Curl Left/Right Profile", _LENGTH_X,
     "Curl Left/Right multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("sepal_curl_outin_profile", "Sepal Curl Out/In Profile", _LENGTH_X,
     "Curl Out/In multiplier", [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("sepal_gravity_profile", "Sepal Gravity Profile", _LENGTH_X, "Gravity multiplier",
     [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("sepal_twist_profile", "Sepal Twist Profile", _LENGTH_X, "Twist multiplier",
     [(0.0, 1.0), (1.0, 1.0)], "generic", None),
    ("sepal_top_shape_profile", "Sepal Top Shape Profile", _WIDTH_X,
     _CAP_Y, DOME_CAP, "cap", None),
    ("sepal_bottom_shape_profile", "Sepal Bottom Shape Profile", _WIDTH_X,
     _CAP_Y, DOME_CAP, "cap", None),
    ("sepal_rotation_pitch_age_profile", "Sepal Pitch Age Profile", _AGE_X,
     "0 (multiplier −1) → 1 (multiplier +1)", [(0.0, 1.0), (1.0, 1.0)], "generic",
     (-1.0, 1.0)),
    ("sepal_bud_curl_profile", "Sepal Bud Curl Profile", _LENGTH_X,
     "Bud Curl multiplier", [(0.0, 1.0), (0.5, 1.1), (1.0, 1.5)], "generic", None),
    ("sepal_bud_tip_height_profile", "Sepal Bud Tip Height Profile", _TIP_ZONE_X,
     "Height multiplier", [(0.0, 0.0), (0.3, 0.06), (0.6, 0.3), (1.0, 1.0)], "generic", None),
    ("sepal_bud_tip_radius_profile", "Sepal Bud Tip Radius Profile", _TIP_ZONE_X,
     "Radius multiplier", [(0.0, 1.0), (1.0, 1.0)], "disc", None),
    ("disc_profile", "Disc Profile", "0 (rim) → 1 (top)", "Radius multiplier",
     [(0.0, 1.0), (0.3, 0.97), (0.6, 0.85), (0.85, 0.55), (1.0, 0.0)], "disc", None),
    ("receptacle_profile", "Receptacle Profile", "0 (rim) → 1 (bottom)",
     "Radius multiplier", [(0.0, 1.0), (1.0, 0.0)], "disc", None),
]
_FLOWER_CURVE_KEYS = [spec[0] for spec in FLOWER_CURVE_SPECS]
_FLOWER_CURVE_REMAPS = {spec[0]: spec[6] for spec in FLOWER_CURVE_SPECS}

# Curves whose Y axis legitimately goes ABOVE 1.0, and how far.
#
# Hard-won, found live: a `ShaderNodeFloatCurve`'s mapping defaults to
# `use_clip = True` with `clip_max_y = 1.0`, and `mapping.update()` silently
# CLAMPS offending points down to that ceiling -- it does not reject the write,
# raise, or even clamp consistently (setting the six Bell points, only the
# second-to-last came back as 1.0, the rest kept their authored values). Every
# radius-MULTIPLIER curve here is affected, because a flaring corolla tube is
# 2-3x its base radius by definition. This silently flattened part of every
# funnel/bell/urn/salverform profile the moment it was written, which is a good
# part of why those shapes never matched their botanical namesakes.
#
# Same underlying clip mechanism as the `y_remap` column above, opposite fix:
# a remap is right when the widget's own 0..1 grid is the natural way to EDIT
# the value (Top/Bottom Shape's signed push), and widening the clip is right
# when the real range is what the user should see and drag in (a radius
# multiplier, where "2x" is a meaningful reading and 0..1 is not).
_FLOWER_CURVE_Y_MAX = {
    "petal_tube_profile": 4.0,
    "sepal_tube_profile": 4.0,
    "disc_profile": 1.5,            # FLOWER_DISC_SHAPES' "convex" bulges to 1.15
    "petal_bud_tip_radius_profile": 1.5,   # same library
    "sepal_bud_tip_radius_profile": 1.5,
}


def is_flower(obj) -> bool:
    return bool(obj) and obj.type == 'CURVE' and obj.get(_FLOWER_TAG) is not None


# --- per-object curve widgets --------------------------------------------------
# Reuses `plant.py`'s curve-widget plumbing directly (`_set_curve_points`/
# `_ramp_from_node`/`raw_ramp` are generic over any ShaderNodeFloatCurve node, not
# stem-specific) rather than duplicating it.

def ensure_flower_curves(obj) -> dict:
    """Return {key: FloatCurve node} for this flower, creating its group if needed."""
    s = obj.plantgen_flower
    ng = s.curve_group
    if ng is None:
        ng = bpy.data.node_groups.new("PlantGen_Flower_Curves", 'ShaderNodeTree')
        s.curve_group = ng
    nodes = {}
    for key, _label, _x_label, _y_label, defaults, _library, _y_remap in FLOWER_CURVE_SPECS:
        node = next((n for n in ng.nodes if n.get("pg_key") == key), None)
        created = node is None
        if created:
            node = ng.nodes.new('ShaderNodeFloatCurve')
            node["pg_key"] = key
            node.label = key
        # Widen the clip ceiling BEFORE writing any points -- `mapping.update()`
        # clamps to it, so a default authored above 1.0 would otherwise be
        # flattened on the way in (see `_FLOWER_CURVE_Y_MAX`). Re-applied on
        # every ensure, not just at creation, so flowers made before this was
        # fixed stop having their tube profiles quietly squashed too.
        y_max = _FLOWER_CURVE_Y_MAX.get(key)
        if y_max is not None and node.mapping.clip_max_y != y_max:
            node.mapping.clip_max_y = y_max
            node.mapping.update()
        if created:
            _set_curve_points(node, defaults)
        nodes[key] = node
    return nodes


def get_flower_curves(obj) -> dict | None:
    """Fetch existing curve nodes without creating anything (safe for draw())."""
    if not obj:
        return None
    ng = obj.plantgen_flower.curve_group
    if ng is None:
        return None
    nodes = {n.get("pg_key"): n for n in ng.nodes if n.get("pg_key")}
    return nodes if set(_FLOWER_CURVE_KEYS) <= set(nodes) else None


def set_flower_curve(obj, key: str, ramp: Ramp, handle_type: str = 'AUTO') -> None:
    _set_curve_points(ensure_flower_curves(obj)[key], ramp.points, handle_type)


# --- settings -> skeleton -------------------------------------------------------

def flower_params_from_object(obj) -> FlowerMeshParams:
    s = obj.plantgen_flower
    curves = ensure_flower_curves(obj)
    ramp_kwargs = {}
    for key in _FLOWER_CURVE_KEYS:
        ramp = _ramp_from_node(curves[key])
        y_remap = _FLOWER_CURVE_REMAPS[key]
        if y_remap is not None:
            lo, hi = y_remap
            ramp = Ramp(points=[(x, lo + y * (hi - lo)) for x, y in ramp.points])
        ramp_kwargs[key] = ramp
    return FlowerMeshParams(
        bloom=_bloom_ui_to_raw(s.bloom),
        maturity=s.maturity,
        petal_enabled=s.petal_enabled,
        petal_bud_rotation_pitch=s.petal_bud_rotation_pitch,
        petal_bud_curl=s.petal_bud_curl, petal_bud_sink=s.petal_bud_sink,
        petal_bud_clamp_overlap=s.petal_bud_clamp_overlap,
        petal_bud_tip_start=s.petal_bud_tip_start,
        petal_bud_tip_lift=s.petal_bud_tip_lift,
        petal_bud_tip_falloff=s.petal_bud_tip_falloff,
        petal_bloom_start=s.petal_bloom_start, petal_bloom_end=s.petal_bloom_end,
        petal_count=s.petal_count, petal_count_variance=s.petal_count_variance,
        petal_length=s.petal_length, petal_length_variance=s.petal_length_variance,
        petal_width=s.petal_width, petal_width_variance=s.petal_width_variance,
        petal_cup=s.petal_cup, petal_cup_variance=s.petal_cup_variance,
        petal_top_shape=s.petal_top_shape,
        petal_top_shape_variance=s.petal_top_shape_variance,
        petal_bottom_shape=s.petal_bottom_shape,
        petal_bottom_shape_variance=s.petal_bottom_shape_variance,
        petal_bud_cup=s.petal_bud_cup,
        petal_scale=s.petal_scale, petal_scale_variance=s.petal_scale_variance,
        resolution=s.resolution,
        petal_width_resolution=s.petal_width_resolution,
        sepal_width_resolution=s.sepal_width_resolution,
        petal_tube_sides=s.petal_tube_sides,
        sepal_tube_sides=s.sepal_tube_sides,
        disc_sides=s.disc_sides,
        receptacle_sides=s.receptacle_sides,
        petal_flip_normals=s.petal_flip_normals,
        sepal_flip_normals=s.sepal_flip_normals,
        disc_flip_normals=s.disc_flip_normals,
        receptacle_flip_normals=s.receptacle_flip_normals,
        petal_tube_flip_normals=s.petal_tube_flip_normals,
        sepal_tube_flip_normals=s.sepal_tube_flip_normals,
        petal_sink=s.petal_sink, petal_sink_variance=s.petal_sink_variance,
        petal_vertical_offset=s.petal_vertical_offset,
        petal_vertical_offset_variance=s.petal_vertical_offset_variance,
        petal_spread_horizontal=s.petal_spread_horizontal,
        petal_spread_horizontal_variance=s.petal_spread_horizontal_variance,
        petal_tube_enabled=s.petal_tube_enabled,
        petal_tube_length=s.petal_tube_length,
        petal_tube_length_variance=s.petal_tube_length_variance,
        petal_tube_match_disc=s.petal_tube_match_disc,
        petal_tube_radius=s.petal_tube_radius,
        petal_tube_radius_variance=s.petal_tube_radius_variance,
        petal_tube_resolution=s.petal_tube_resolution,
        petal_tube_noise=s.petal_tube_noise,
        petal_tube_match_angle=s.petal_tube_match_angle,
        petal_tube_fuse=s.petal_tube_fuse,
        petal_bud_tube_length=s.petal_bud_tube_length,
        petal_bud_tube_flare=s.petal_bud_tube_flare,
        petal_clamp_overlap=s.petal_clamp_overlap,
        petal_layers=s.petal_layers, petal_layer_spacing=s.petal_layer_spacing,
        petal_layer_spacing_variance=s.petal_layer_spacing_variance,
        petal_layer_divergence=s.petal_layer_divergence,
        petal_curl_updown=s.petal_curl_updown,
        petal_curl_updown_variance=s.petal_curl_updown_variance,
        petal_curl_leftright=s.petal_curl_leftright,
        petal_curl_leftright_variance=s.petal_curl_leftright_variance,
        petal_curl_outin=s.petal_curl_outin, petal_curl_outin_variance=s.petal_curl_outin_variance,
        petal_curl_roll=s.petal_curl_roll, petal_curl_roll_variance=s.petal_curl_roll_variance,
        petal_gravity=s.petal_gravity, petal_gravity_variance=s.petal_gravity_variance,
        petal_twist=s.petal_twist, petal_twist_variance=s.petal_twist_variance,
        petal_rotation_yaw=s.petal_rotation_yaw,
        petal_rotation_yaw_variance=s.petal_rotation_yaw_variance,
        petal_rotation_pitch=s.petal_rotation_pitch,
        petal_rotation_pitch_variance=s.petal_rotation_pitch_variance,
        petal_rotation_roll=s.petal_rotation_roll,
        petal_rotation_roll_variance=s.petal_rotation_roll_variance,
        sepal_enabled=s.sepal_enabled,
        sepal_count=s.sepal_count, sepal_count_variance=s.sepal_count_variance,
        sepal_length=s.sepal_length, sepal_length_variance=s.sepal_length_variance,
        sepal_width=s.sepal_width, sepal_width_variance=s.sepal_width_variance,
        sepal_cup=s.sepal_cup, sepal_cup_variance=s.sepal_cup_variance,
        sepal_top_shape=s.sepal_top_shape,
        sepal_top_shape_variance=s.sepal_top_shape_variance,
        sepal_bottom_shape=s.sepal_bottom_shape,
        sepal_bottom_shape_variance=s.sepal_bottom_shape_variance,
        sepal_bud_cup=s.sepal_bud_cup,
        sepal_bud_tip_start=s.sepal_bud_tip_start,
        sepal_bud_tip_falloff=s.sepal_bud_tip_falloff,
        sepal_scale=s.sepal_scale, sepal_scale_variance=s.sepal_scale_variance,
        sepal_bud_rotation_pitch=s.sepal_bud_rotation_pitch,
        sepal_bud_curl=s.sepal_bud_curl, sepal_bud_sink=s.sepal_bud_sink,
        sepal_bud_clamp_overlap=s.sepal_bud_clamp_overlap,
        sepal_bud_tip_lift=s.sepal_bud_tip_lift,
        sepal_bloom_start=s.sepal_bloom_start, sepal_bloom_end=s.sepal_bloom_end,
        sepal_resolution=s.sepal_resolution,
        sepal_sink=s.sepal_sink, sepal_sink_variance=s.sepal_sink_variance,
        sepal_vertical_offset=s.sepal_vertical_offset,
        sepal_vertical_offset_variance=s.sepal_vertical_offset_variance,
        sepal_azimuth_offset=s.sepal_azimuth_offset, sepal_z_offset=s.sepal_z_offset,
        sepal_spread_horizontal=s.sepal_spread_horizontal,
        sepal_spread_horizontal_variance=s.sepal_spread_horizontal_variance,
        sepal_tube_enabled=s.sepal_tube_enabled,
        sepal_tube_length=s.sepal_tube_length,
        sepal_tube_length_variance=s.sepal_tube_length_variance,
        sepal_tube_match_disc=s.sepal_tube_match_disc,
        sepal_tube_radius=s.sepal_tube_radius,
        sepal_tube_radius_variance=s.sepal_tube_radius_variance,
        sepal_tube_resolution=s.sepal_tube_resolution,
        sepal_tube_noise=s.sepal_tube_noise,
        sepal_tube_match_angle=s.sepal_tube_match_angle,
        sepal_tube_fuse=s.sepal_tube_fuse,
        sepal_bud_tube_length=s.sepal_bud_tube_length,
        sepal_bud_tube_flare=s.sepal_bud_tube_flare,
        sepal_clamp_overlap=s.sepal_clamp_overlap,
        sepal_layers=s.sepal_layers, sepal_layer_spacing=s.sepal_layer_spacing,
        sepal_layer_spacing_variance=s.sepal_layer_spacing_variance,
        sepal_layer_divergence=s.sepal_layer_divergence,
        sepal_curl_updown=s.sepal_curl_updown,
        sepal_curl_updown_variance=s.sepal_curl_updown_variance,
        sepal_curl_leftright=s.sepal_curl_leftright,
        sepal_curl_leftright_variance=s.sepal_curl_leftright_variance,
        sepal_curl_outin=s.sepal_curl_outin, sepal_curl_outin_variance=s.sepal_curl_outin_variance,
        sepal_curl_roll=s.sepal_curl_roll, sepal_curl_roll_variance=s.sepal_curl_roll_variance,
        sepal_gravity=s.sepal_gravity, sepal_gravity_variance=s.sepal_gravity_variance,
        sepal_twist=s.sepal_twist, sepal_twist_variance=s.sepal_twist_variance,
        sepal_rotation_yaw=s.sepal_rotation_yaw,
        sepal_rotation_yaw_variance=s.sepal_rotation_yaw_variance,
        sepal_rotation_pitch=s.sepal_rotation_pitch,
        sepal_rotation_pitch_variance=s.sepal_rotation_pitch_variance,
        sepal_rotation_roll=s.sepal_rotation_roll,
        sepal_rotation_roll_variance=s.sepal_rotation_roll_variance,
        disc_enabled=s.disc_enabled,
        disc_radius=s.disc_radius, disc_radius_variance=s.disc_radius_variance,
        disc_height=s.disc_height, disc_height_variance=s.disc_height_variance,
        disc_resolution=s.disc_resolution,
        receptacle_enabled=s.receptacle_enabled,
        receptacle_match_disc=s.receptacle_match_disc,
        receptacle_radius=s.receptacle_radius,
        receptacle_radius_variance=s.receptacle_radius_variance,
        receptacle_height=s.receptacle_height,
        receptacle_height_variance=s.receptacle_height_variance,
        receptacle_resolution=s.receptacle_resolution,
        seed=s.seed,
        **ramp_kwargs,
    )


def flower_params_for_preset(obj) -> FlowerMeshParams:
    """Like `flower_params_from_object`, but with every curve (see
    `FlowerMeshParams.RAMP_FIELDS`) captured as its actual authored control points
    instead of a dense evaluation bake -- see `plant.raw_ramp`. Keeps a saved .json
    preset small/git-diffable and the curves editable after a reload, same
    reasoning as `blender/presets.py`'s `_stem_params_for_preset`/
    `phyllotaxis_params_for_preset`."""
    params = flower_params_from_object(obj)
    curves = get_flower_curves(obj)
    if curves:
        ramp_kwargs = {
            field: raw_ramp(curves[field]) for field in params.RAMP_FIELDS
        }
        params = dataclasses.replace(params, **ramp_kwargs)
    return params


# --- attributed geometry ---------------------------------------------------------

def _write_ribbon_curve(curve_data, petals: list[PetalSpine]) -> None:
    """Author a whorl of ribbon appendages (Petals or Sepals -- both use this
    directly) as N poly splines -- native per-point `radius` repurposed as width,
    native per-point `tilt` as accumulated twist (see module docstring).
    `_build_flower_group`'s flat ribbon profile spans local x in [-0.5, 0.5]
    (total width 1.0), so a native radius of R here produces a cross-section
    exactly R units wide, mirroring how the stem's profile circle has Radius =
    1.0 so its own native radius equals true stem radius directly."""
    curve_data.splines.clear()
    for petal in petals:
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(petal.points) - 1)
        for i, (co, w, tilt) in enumerate(
            zip(petal.points, petal.widths, petal.tilts, strict=True)
        ):
            spline.points[i].co = (co[0], co[1], co[2], 1.0)
            spline.points[i].radius = w
            spline.points[i].tilt = tilt


def _write_disc_curve(curve_data, disc) -> None:
    """Author a revolved body (Disc or Receptacle -- both use this directly) as a
    single poly spline -- native per-point `radius` repurposed as the body's own
    radius-vs-height profile (see module docstring), meshed with a CIRCLE profile
    (not the ribbons' flat line) so it revolves into a solid body instead of a
    ribbon. An empty `disc` (see `skeleton/flower.py`'s `receptacle_enabled`
    handling) simply clears the spline, producing empty GN geometry."""
    curve_data.splines.clear()
    if not disc.points:
        return
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(disc.points) - 1)
    for i, (co, r) in enumerate(zip(disc.points, disc.radii, strict=True)):
        spline.points[i].co = (co[0], co[1], co[2], 1.0)
        spline.points[i].radius = r


_SHAPE_CARRIER_OFFSET = 1000.0


def _write_shape_carrier_curve(curve_data, values: list[float]) -> None:
    """Author a Top/Bottom Shape carrier (see `skeleton.flower._build_shape_carrier`'s
    docstring) as a single poly spline whose native per-point `radius` holds one
    push value per ribbon cross-section profile point -- read back in
    `_build_flower_group`'s `_ribbon_mesh` via Sample Index, keyed by each output
    mesh vertex's own profile-point index (`Index % _RIBBON_PROFILE_RESOLUTION`).
    Point `co` positions are never read by anything -- only `radius` matters --
    so they're arbitrary, just spread out for readability in the outliner/viewport.

    The values are DIMENSIONLESS (multiples of the target ring's own width), not
    meters -- `_push_boundary_ring` multiplies by the ring's live measured chord.

    Values are shifted up by `_SHAPE_CARRIER_OFFSET` before writing and shifted
    back down by the same amount in `_build_flower_group`'s GN read -- confirmed
    live that `SplinePoint.radius` silently CLAMPS negative values to 0 (unlike
    every other repurposed field in this codebase, all of which only ever store
    non-negative magnitudes), and Top/Bottom Shape genuinely needs negative
    pushes (to notch the cap inward, not just dome it outward)."""
    curve_data.splines.clear()
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(values) - 1)
    for i, v in enumerate(values):
        spline.points[i].co = (float(i), 0.0, 0.0, 1.0)
        spline.points[i].radius = v + _SHAPE_CARRIER_OFFSET


def _ensure_child_curve_object(obj, tag: str, suffix: str) -> bpy.types.Object:
    """Find or create a child `Curve` object tagged `tag` on this flower --
    mirrors `plant.ensure_attachment_object`'s find-or-create shape, but for a
    Curve datablock instead of a vertices-only Mesh. Shared by Disc/Sepals/
    Receptacle, each of which needs its own child object (see module docstring:
    one `Curve to Mesh` call = one profile, and each component needs its own)."""
    child = next((c for c in obj.children if c.get(tag)), None)
    if child is None:
        curve_data = bpy.data.curves.new(f"{obj.name}_{suffix}", 'CURVE')
        child = bpy.data.objects.new(f"{obj.name}_{suffix}", curve_data)
        child[tag] = True
        child.parent = obj
        child.matrix_parent_inverse = Matrix.Identity(4)
        collection = obj.users_collection[0] if obj.users_collection else bpy.context.collection
        collection.objects.link(child)
    return child


def _ensure_disc_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _DISC_TAG, "Disc")


def _ensure_sepal_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _SEPAL_TAG, "Sepals")


def _ensure_receptacle_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _RECEPTACLE_TAG, "Receptacle")


def _ensure_petal_fusion_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _PETAL_FUSION_TAG, "PetalFusion")


def _ensure_sepal_fusion_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _SEPAL_FUSION_TAG, "SepalFusion")


def _ensure_petal_top_shape_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _PETAL_TOP_SHAPE_TAG, "PetalTopShape")


def _ensure_petal_bottom_shape_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _PETAL_BOTTOM_SHAPE_TAG, "PetalBottomShape")


def _ensure_sepal_top_shape_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _SEPAL_TOP_SHAPE_TAG, "SepalTopShape")


def _ensure_sepal_bottom_shape_object(obj) -> bpy.types.Object:
    return _ensure_child_curve_object(obj, _SEPAL_BOTTOM_SHAPE_TAG, "SepalBottomShape")


# --- the GN modifier (built once, reused by every flower) -----------------------

# Bump whenever _build_flower_group's node structure changes -- same staleness-
# detection convention as plant.py's _MESH_GROUP_VERSION.
_FLOWER_GROUP_VERSION = 43  # v43: tube weld's turn nudged off its half-slot tie
_PG_VERSION_KEY = "pg_version"


def _build_flower_group() -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.new(_FLOWER_GROUP_NAME, 'GeometryNodeTree')
    # The version is stamped at the END (a build that fails part-way must not
    # be reused as complete -- it was, once, after a builder error).
    iface = ng.interface
    iface.new_socket(name="Geometry", socket_type="NodeSocketGeometry", in_out='INPUT')
    iface.new_socket(name=_DISC_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(name=_PETAL_CUP_SOCKET, socket_type="NodeSocketFloat", in_out='INPUT')
    iface.new_socket(name=_PETAL_WIDTH_REF_SOCKET, socket_type="NodeSocketFloat", in_out='INPUT')
    iface.new_socket(name=_DISC_CONCAVE_SOCKET, socket_type="NodeSocketBool", in_out='INPUT')
    iface.new_socket(name=_SEPAL_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(name=_SEPAL_CUP_SOCKET, socket_type="NodeSocketFloat", in_out='INPUT')
    iface.new_socket(name=_SEPAL_WIDTH_REF_SOCKET, socket_type="NodeSocketFloat", in_out='INPUT')
    iface.new_socket(name=_RECEPTACLE_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(name=_PETAL_FUSION_POINTS_SOCKET, socket_type="NodeSocketObject",
                     in_out='INPUT')
    iface.new_socket(name=_SEPAL_FUSION_POINTS_SOCKET, socket_type="NodeSocketObject",
                     in_out='INPUT')
    iface.new_socket(
        name=_PETAL_TOP_SHAPE_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(
        name=_PETAL_BOTTOM_SHAPE_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(
        name=_SEPAL_TOP_SHAPE_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(
        name=_SEPAL_BOTTOM_SHAPE_POINTS_SOCKET, socket_type="NodeSocketObject", in_out='INPUT')
    iface.new_socket(name=_PETAL_RESOLUTION_SOCKET, socket_type="NodeSocketInt", in_out='INPUT')
    iface.new_socket(name=_SEPAL_RESOLUTION_SOCKET, socket_type="NodeSocketInt", in_out='INPUT')
    for name, kind in ((_PETAL_WELD_SOCKET, "NodeSocketBool"),
                       (_SEPAL_WELD_SOCKET, "NodeSocketBool"),
                       (_PETAL_COUNT_SOCKET, "NodeSocketInt"),
                       (_SEPAL_COUNT_SOCKET, "NodeSocketInt"),
                       (_PETAL_TUBE_SIDES_SOCKET, "NodeSocketInt"),
                       (_SEPAL_TUBE_SIDES_SOCKET, "NodeSocketInt"),
                       (_RECEPTACLE_WELD_SOCKET, "NodeSocketBool"),
                       (_RECEPTACLE_SIDES_SOCKET, "NodeSocketInt"),
                       (_PETAL_WIDTH_RES_SOCKET, "NodeSocketInt"),
                       (_SEPAL_WIDTH_RES_SOCKET, "NodeSocketInt"),
                       (_DISC_SIDES_SOCKET, "NodeSocketInt"),
                       (_PETAL_FLIP_SOCKET, "NodeSocketBool"),
                       (_SEPAL_FLIP_SOCKET, "NodeSocketBool"),
                       (_DISC_FLIP_SOCKET, "NodeSocketBool"),
                       (_RECEPTACLE_FLIP_SOCKET, "NodeSocketBool"),
                       (_PETAL_TUBE_FLIP_SOCKET, "NodeSocketBool"),
                       (_SEPAL_TUBE_FLIP_SOCKET, "NodeSocketBool"),
                       (_FLOWER_MAT_SOCKET, "NodeSocketMaterial"),
                       (_CHECKER_SOCKET, "NodeSocketBool"),
                       (_CHECKER_MAT_SOCKET, "NodeSocketMaterial"),
                       (_UV_SCALE_SOCKET, "NodeSocketFloat"),
                       *((_UV_CARRIER_SOCKETS[p], "NodeSocketObject") for p in _UV_PARTS),
                       *((_UV_CENTER_SOCKETS[p], "NodeSocketVector") for p in _UV_PARTS)):
        iface.new_socket(name=name, socket_type=kind, in_out='INPUT')
    iface.new_socket(name="Geometry", socket_type="NodeSocketGeometry", in_out='OUTPUT')

    nodes, links = ng.nodes, ng.links
    col_in, col_mesh, col_join, col_out = -900, 0, 600, 900

    gin = nodes.new("NodeGroupInput")
    gin.location = (col_in, 0)
    gout = nodes.new("NodeGroupOutput")
    gout.location = (col_out, 0)

    def _ring_chord(mesh_socket, idx_socket, cross_idx_socket, p_socket, *, x: int, y: int):
        """This vertex's own RING measured edge to edge, live off the already-
        meshed geometry: returns `(chord_vector, chord_length)` sockets.

        `idx - cross_idx` is always that ring's own cross-position-0 vertex
        regardless of which spline or ring the vertex belongs to, and
        `+ (P-1)` is the opposite edge -- so two Sample Index reads give the
        ring's real width including whatever Curl/Twist/Cup has already done
        to it. Shared by `_apply_cup` (which needs the vector, as its own
        "width direction") and `_push_boundary_ring` (which needs the length,
        to scale a dimensionless cap push into real distance); they used to
        each build this chain, and they must not drift apart."""
        edge0_index = nodes.new("ShaderNodeMath")
        edge0_index.operation = 'SUBTRACT'
        links.new(idx_socket, edge0_index.inputs[0])
        links.new(cross_idx_socket, edge0_index.inputs[1])
        edge0_index.location = (x, y)

        edge0_index_int = nodes.new("FunctionNodeFloatToInt")
        edge0_index_int.rounding_mode = 'ROUND'
        links.new(edge0_index.outputs["Value"], edge0_index_int.inputs["Float"])
        edge0_index_int.location = (x + 200, y)

        edge1_index = nodes.new("ShaderNodeMath")
        edge1_index.operation = 'ADD'
        links.new(_math('SUBTRACT', p_socket, 1.0, x, y - 150), edge1_index.inputs[1])
        links.new(edge0_index.outputs["Value"], edge1_index.inputs[0])
        edge1_index.location = (x + 200, y - 150)

        edge1_index_int = nodes.new("FunctionNodeFloatToInt")
        edge1_index_int.rounding_mode = 'ROUND'
        links.new(edge1_index.outputs["Value"], edge1_index_int.inputs["Float"])
        edge1_index_int.location = (x + 400, y - 150)

        position = nodes.new("GeometryNodeInputPosition")
        position.location = (x, y - 300)

        sample_edge0 = nodes.new("GeometryNodeSampleIndex")
        sample_edge0.data_type = 'FLOAT_VECTOR'
        sample_edge0.domain = 'POINT'
        links.new(mesh_socket, sample_edge0.inputs["Geometry"])
        links.new(position.outputs["Position"], sample_edge0.inputs["Value"])
        links.new(edge0_index_int.outputs["Integer"], sample_edge0.inputs["Index"])
        sample_edge0.location = (x + 600, y)

        sample_edge1 = nodes.new("GeometryNodeSampleIndex")
        sample_edge1.data_type = 'FLOAT_VECTOR'
        sample_edge1.domain = 'POINT'
        links.new(mesh_socket, sample_edge1.inputs["Geometry"])
        links.new(position.outputs["Position"], sample_edge1.inputs["Value"])
        links.new(edge1_index_int.outputs["Integer"], sample_edge1.inputs["Index"])
        sample_edge1.location = (x + 600, y - 150)

        chord_vec = nodes.new("ShaderNodeVectorMath")
        chord_vec.operation = 'SUBTRACT'
        links.new(sample_edge1.outputs["Value"], chord_vec.inputs[0])
        links.new(sample_edge0.outputs["Value"], chord_vec.inputs[1])
        chord_vec.location = (x + 800, y)

        chord_len = nodes.new("ShaderNodeVectorMath")
        chord_len.operation = 'LENGTH'
        links.new(chord_vec.outputs["Vector"], chord_len.inputs[0])
        chord_len.location = (x + 1000, y)

        return chord_vec.outputs["Vector"], chord_len.outputs["Value"]

    def _push_boundary_ring(
        mesh_socket, resolution_socket, shape_points_socket, p_socket, *, is_tip: bool,
        y: int,
    ):
        """Push ONE boundary ring (the base ring of every spline if `is_tip` is
        False, the tip ring if True) outward along its own local "growth"
        direction, by a per-cross-section-point amount authored on a
        dedicated shape-carrier object (`_write_shape_carrier_curve`) -- the
        actual mechanism behind Top/Bottom Shape (see
        `skeleton/flower.py`'s `petal_top_shape_profile` docstring for why
        this can't be a per-point-varying Profile Curve, which Curve to Mesh
        has no way to support).

        The carrier's values are DIMENSIONLESS (multiples of the ring's own
        width); this multiplies them by that ring's live measured chord (see
        `_ring_chord`) to get the real push. That indirection is what makes a
        cap curve read identically at any petal size, track per-petal Width/
        Scale variance and per-layer Age scaling for free, and -- crucially --
        vanish smoothly at a tapered tip, where the earlier absolute-distance
        version tore a collapsed ring apart instead of shaping it.

        Deliberately uses ONLY `Index`/`Position`/`Sample Index` -- confirmed
        live that a custom attribute captured on the pre-sweep curve does NOT
        survive Curve to Mesh onto the output mesh, so nothing here relies on
        that. Ring identification: with a fixed profile resolution P
        (`_RIBBON_PROFILE_RESOLUTION`) and a RUNTIME main-curve-points-per-
        spline count N (`resolution_socket`, since Resolution is a user-facing
        slider), each spline occupies N*P consecutive vertices; `Index %
        (N*P)` gives the position within its own spline regardless of which
        spline (petal) a vertex belongs to (confirmed live with 5 petals --
        no cross-contamination), and `// P` of that gives the ring number
        (0=base, N-1=tip). The "growth direction" at a boundary vertex is
        `normalize(this vertex's Position - the corresponding vertex one ring
        further in, same cross-section index)` -- read via Sample Index on
        THIS SAME mesh (not the pre-sweep curve), so it's whatever Blender
        actually computed for that vertex, including tilt/curl bending,
        rather than a hand-reconstructed frame (confirmed live that a
        Python-side reconstruction using the same fixed bend-axis convention
        `_build_petal_spine` uses for CURL does NOT match Blender's actual
        swept-profile frame once Twist is nonzero)."""
        idx = nodes.new("GeometryNodeInputIndex")
        idx.location = (col_mesh + 300, y)

        n_times_p = nodes.new("ShaderNodeMath")
        n_times_p.operation = 'MULTIPLY'
        links.new(p_socket, n_times_p.inputs[1])
        links.new(resolution_socket, n_times_p.inputs[0])
        n_times_p.location = (col_mesh + 300, y - 120)

        local_idx = nodes.new("ShaderNodeMath")
        local_idx.operation = 'MODULO'
        links.new(idx.outputs["Index"], local_idx.inputs[0])
        links.new(n_times_p.outputs["Value"], local_idx.inputs[1])
        local_idx.location = (col_mesh + 500, y)

        main_idx = nodes.new("ShaderNodeMath")
        main_idx.operation = 'DIVIDE'
        links.new(p_socket, main_idx.inputs[1])
        links.new(local_idx.outputs["Value"], main_idx.inputs[0])
        main_idx.location = (col_mesh + 700, y)

        main_idx_floor = nodes.new("ShaderNodeMath")
        main_idx_floor.operation = 'FLOOR'
        links.new(main_idx.outputs["Value"], main_idx_floor.inputs[0])
        main_idx_floor.location = (col_mesh + 900, y)

        is_boundary = nodes.new("ShaderNodeMath")
        is_boundary.operation = 'COMPARE'
        is_boundary.inputs[2].default_value = 0.1
        is_boundary.location = (col_mesh + 1100, y)
        links.new(main_idx_floor.outputs["Value"], is_boundary.inputs[0])
        if is_tip:
            target = nodes.new("ShaderNodeMath")
            target.operation = 'SUBTRACT'
            target.inputs[1].default_value = 1.0
            links.new(resolution_socket, target.inputs[0])
            target.location = (col_mesh + 900, y - 200)
            links.new(target.outputs["Value"], is_boundary.inputs[1])
        else:
            is_boundary.inputs[1].default_value = 0.0

        prev_idx = nodes.new("ShaderNodeMath")
        prev_idx.operation = 'SUBTRACT' if is_tip else 'ADD'
        links.new(p_socket, prev_idx.inputs[1])
        links.new(idx.outputs["Index"], prev_idx.inputs[0])
        prev_idx.location = (col_mesh + 500, y - 350)

        prev_idx_int = nodes.new("FunctionNodeFloatToInt")
        prev_idx_int.rounding_mode = 'ROUND'
        links.new(prev_idx.outputs["Value"], prev_idx_int.inputs["Float"])
        prev_idx_int.location = (col_mesh + 700, y - 350)

        position = nodes.new("GeometryNodeInputPosition")
        position.location = (col_mesh + 300, y - 550)

        sample_pos = nodes.new("GeometryNodeSampleIndex")
        sample_pos.data_type = 'FLOAT_VECTOR'
        sample_pos.domain = 'POINT'
        links.new(mesh_socket, sample_pos.inputs["Geometry"])
        links.new(position.outputs["Position"], sample_pos.inputs["Value"])
        links.new(prev_idx_int.outputs["Integer"], sample_pos.inputs["Index"])
        sample_pos.location = (col_mesh + 900, y - 550)

        sub_vec = nodes.new("ShaderNodeVectorMath")
        sub_vec.operation = 'SUBTRACT'
        links.new(position.outputs["Position"], sub_vec.inputs[0])
        links.new(sample_pos.outputs["Value"], sub_vec.inputs[1])
        sub_vec.location = (col_mesh + 1100, y - 550)

        norm_vec = nodes.new("ShaderNodeVectorMath")
        norm_vec.operation = 'NORMALIZE'
        links.new(sub_vec.outputs["Vector"], norm_vec.inputs[0])
        norm_vec.location = (col_mesh + 1300, y - 550)

        profile_idx = nodes.new("ShaderNodeMath")
        profile_idx.operation = 'MODULO'
        links.new(p_socket, profile_idx.inputs[1])
        links.new(idx.outputs["Index"], profile_idx.inputs[0])
        profile_idx.location = (col_mesh + 500, y - 750)

        profile_idx_int = nodes.new("FunctionNodeFloatToInt")
        profile_idx_int.rounding_mode = 'ROUND'
        links.new(profile_idx.outputs["Value"], profile_idx_int.inputs["Float"])
        profile_idx_int.location = (col_mesh + 700, y - 750)

        shape_obj_info = nodes.new("GeometryNodeObjectInfo")
        links.new(shape_points_socket, shape_obj_info.inputs["Object"])
        shape_obj_info.location = (col_mesh + 500, y - 950)

        shape_radius_attr = nodes.new("GeometryNodeInputNamedAttribute")
        shape_radius_attr.data_type = 'FLOAT'
        shape_radius_attr.inputs["Name"].default_value = "radius"
        shape_radius_attr.location = (col_mesh + 500, y - 1100)

        sample_push = nodes.new("GeometryNodeSampleIndex")
        sample_push.data_type = 'FLOAT'
        sample_push.domain = 'POINT'
        links.new(shape_obj_info.outputs["Geometry"], sample_push.inputs["Geometry"])
        links.new(shape_radius_attr.outputs["Attribute"], sample_push.inputs["Value"])
        links.new(profile_idx_int.outputs["Integer"], sample_push.inputs["Index"])
        sample_push.location = (col_mesh + 900, y - 950)

        # Undo `_write_shape_carrier_curve`'s offset (SplinePoint.radius
        # silently clamps negative values to 0 -- see that function's docstring).
        push_factor = nodes.new("ShaderNodeMath")
        push_factor.operation = 'SUBTRACT'
        push_factor.inputs[1].default_value = _SHAPE_CARRIER_OFFSET
        links.new(sample_push.outputs["Value"], push_factor.inputs[0])
        push_factor.location = (col_mesh + 1100, y - 950)

        # ... then scale that dimensionless factor by THIS ring's own live
        # width, which is what makes the cap size-independent -- see this
        # function's docstring and `_ring_chord`.
        _chord_vec, chord_len = _ring_chord(
            mesh_socket, idx.outputs["Index"], profile_idx.outputs["Value"], p_socket,
            x=col_mesh + 500, y=y - 1350)

        push_amount = nodes.new("ShaderNodeMath")
        push_amount.operation = 'MULTIPLY'
        links.new(push_factor.outputs["Value"], push_amount.inputs[0])
        links.new(chord_len, push_amount.inputs[1])
        push_amount.location = (col_mesh + 1700, y - 950)

        offset_vec = nodes.new("ShaderNodeVectorMath")
        offset_vec.operation = 'SCALE'
        links.new(norm_vec.outputs["Vector"], offset_vec.inputs[0])
        links.new(push_amount.outputs["Value"], offset_vec.inputs["Scale"])
        offset_vec.location = (col_mesh + 1900, y - 750)

        mask_xyz = nodes.new("ShaderNodeCombineXYZ")
        links.new(is_boundary.outputs["Value"], mask_xyz.inputs["X"])
        links.new(is_boundary.outputs["Value"], mask_xyz.inputs["Y"])
        links.new(is_boundary.outputs["Value"], mask_xyz.inputs["Z"])
        mask_xyz.location = (col_mesh + 1700, y - 1150)

        offset_masked = nodes.new("ShaderNodeVectorMath")
        offset_masked.operation = 'MULTIPLY'
        links.new(offset_vec.outputs["Vector"], offset_masked.inputs[0])
        links.new(mask_xyz.outputs["Vector"], offset_masked.inputs[1])
        offset_masked.location = (col_mesh + 2100, y - 750)

        set_pos = nodes.new("GeometryNodeSetPosition")
        links.new(mesh_socket, set_pos.inputs["Geometry"])
        links.new(offset_masked.outputs["Vector"], set_pos.inputs["Offset"])
        set_pos.location = (col_mesh + 2300, y)

        return set_pos.outputs["Geometry"]

    def _apply_cup(mesh_socket, resolution_socket, cup_socket, width_ref_socket, y: int,
                   p_socket=None):
        """Bows the WHOLE ribbon into a shallow dome/bowl (SpeedTree/leaf-style
        "Cup") -- a post-mesh per-vertex push, NOT baked into the swept
        Profile Curve the way an earlier "Curvature" field was (Curve to
        Mesh's own per-point Scale ties a Profile-Curve-driven bend's real-
        world deflection to local width TWICE over -- both the profile's own
        shape and its Scale carry the same width factor -- which is why that
        separate field was folded into this one instead of kept alongside
        it: once Cup's own single width factor below was added, there was no
        longer a meaningful difference between the two). Still scaled by the
        ring's own ACTUAL width (measured live off the mesh, see
        `width_factor` below) so it tapers to nothing exactly where the
        ribbon itself does (the tip, and a narrow base) -- an earlier
        width-INDEPENDENT version read as a hard spike right at the tip,
        where forcing a fixed push onto a near-zero-width ring folds the
        geometry rather than curling it (confirmed against a hand-drawn
        reference: a real cupped petal/leaf tapers smoothly all the way to
        its point, it doesn't hold a constant bend up to a sudden crease).

        Reuses `_push_boundary_ring`'s Index-ring-math primitives (fixed
        profile resolution P, runtime per-spline point count N, `Index %
        (N*P)` to locate a vertex within its own spline, `// P` for its ring
        number, `% P` for its cross-section position) but applies to EVERY
        ring, not just the tip/base boundary -- so it needs its own push
        DIRECTION per ring (not a single fixed axis), since a ring's own
        swept orientation rotates continuously along Curl/Twist. Built from
        only Index/Position/Sample Index, the same "read the ALREADY-MESHED
        geometry, don't hand-reconstruct the frame" discipline
        `_push_boundary_ring` already established (see its own docstring for
        why a hand-reconstructed frame doesn't match Blender's real swept-
        profile frame once Twist is nonzero):

        - `chord_dir` = normalize(this ring's own edge-to-edge vector,
          cross-position P-1 minus cross-position 0) -- the ring's own
          "width" direction.
        - `growth_dir` = normalize(a neighboring ring's Position, same
          cross-position, one ring further along MINUS one ring back) -- the
          ring's own "length" direction, one-sided (self-sampled) at the two
          end rings exactly like `_push_boundary_ring`'s own tip/base cases,
          resolved dynamically per-vertex here since every ring is in play
          (not just one fixed tip/base flag known at build time).
        - push direction = normalize(cross(chord_dir, growth_dir)) -- the
          ring's own swept-profile normal, well-defined (non-degenerate)
          even though the profile itself is always flat (unlike reading the
          arc's own bend direction, which would degenerate to a zero vector
          on a flat profile).
        - push amount = `cup_socket` (meters, at the ring's own edges) times
          a fixed symmetric parabola `(2 * unit_x)^2` (0.0 at the center,
          1.0 at both edges, `unit_x` = this vertex's cross-position
          remapped -0.5..0.5, matching the arc profile's own Start/End) --
          the MARGIN curls the most, the midline stays flattest, matching a
          real petal/leaf rim rather than a hidden internal ridge (an
          earlier `1 - (2*unit_x)^2` version did the opposite -- centered
          the push and pinned the edges flat -- which read as a visible
          spike right at the tip, where the ring's whole width collapses
          toward 0 and that ridge becomes the only thing not sitting at the
          boundary). No per-position hand-curve like Top/Bottom Shape's
          carrier objects, since Cup is meant to be a single-dial "round the
          whole thing off" control, mirroring the leaf blade Cup's own
          single-float design (`blender/plant.py`'s `leaf_cup`) exactly."""
        idx = nodes.new("GeometryNodeInputIndex")
        idx.location = (col_mesh + 300, y)

        n_times_p = nodes.new("ShaderNodeMath")
        n_times_p.operation = 'MULTIPLY'
        links.new(p_socket, n_times_p.inputs[1])
        links.new(resolution_socket, n_times_p.inputs[0])
        n_times_p.location = (col_mesh + 300, y - 120)

        local_idx = nodes.new("ShaderNodeMath")
        local_idx.operation = 'MODULO'
        links.new(idx.outputs["Index"], local_idx.inputs[0])
        links.new(n_times_p.outputs["Value"], local_idx.inputs[1])
        local_idx.location = (col_mesh + 500, y)

        cross_idx = nodes.new("ShaderNodeMath")
        cross_idx.operation = 'MODULO'
        links.new(p_socket, cross_idx.inputs[1])
        links.new(idx.outputs["Index"], cross_idx.inputs[0])
        cross_idx.location = (col_mesh + 500, y - 200)

        # unit_x: cross-position 0..P-1 -> -0.5..0.5, matching the arc
        # profile's own Start (-0.5,0,0) / End (0.5,0,0).
        unit_x = nodes.new("ShaderNodeMath")
        unit_x.operation = 'DIVIDE'
        links.new(_math('SUBTRACT', p_socket, 1.0, col_mesh + 500, y - 200),
                  unit_x.inputs[1])
        links.new(cross_idx.outputs["Value"], unit_x.inputs[0])
        unit_x.location = (col_mesh + 700, y - 200)

        unit_x_centered = nodes.new("ShaderNodeMath")
        unit_x_centered.operation = 'SUBTRACT'
        unit_x_centered.inputs[1].default_value = 0.5
        links.new(unit_x.outputs["Value"], unit_x_centered.inputs[0])
        unit_x_centered.location = (col_mesh + 900, y - 200)

        # bulge = (2 * unit_x_centered)^2 -- 0.0 at the ring's own center,
        # 1.0 at both cross-section edges: the margin should curl the MOST
        # the further it sits from the petal's own midline, not the least
        # (an earlier `1 - (2x)^2` version pushed the CENTER hardest and
        # pinned the edges at zero -- a hidden internal ridge rather than
        # the margin itself curling, which read fine at wide rings but
        # produced a visible spike right at the tip, where width collapses
        # toward 0 and that ridge is suddenly the only thing NOT sitting at
        # the flat boundary).
        two_x = nodes.new("ShaderNodeMath")
        two_x.operation = 'MULTIPLY'
        two_x.inputs[1].default_value = 2.0
        links.new(unit_x_centered.outputs["Value"], two_x.inputs[0])
        two_x.location = (col_mesh + 1100, y - 200)

        bulge = nodes.new("ShaderNodeMath")
        bulge.operation = 'POWER'
        bulge.inputs[1].default_value = 2.0
        links.new(two_x.outputs["Value"], bulge.inputs[0])
        bulge.location = (col_mesh + 1300, y - 200)

        push_bulge = nodes.new("ShaderNodeMath")
        push_bulge.operation = 'MULTIPLY'
        links.new(cup_socket, push_bulge.inputs[0])
        links.new(bulge.outputs["Value"], push_bulge.inputs[1])
        push_bulge.location = (col_mesh + 1700, y - 200)

        # Ring-boundary detection, for one-sided growth-direction sampling at
        # the tip/base rings (mirrors `_push_boundary_ring`'s own tip/base
        # cases, but resolved dynamically per-vertex here since every ring
        # needs a valid growth direction, not just the two boundary ones).
        ring = nodes.new("ShaderNodeMath")
        ring.operation = 'DIVIDE'
        links.new(p_socket, ring.inputs[1])
        links.new(local_idx.outputs["Value"], ring.inputs[0])
        ring.location = (col_mesh + 700, y - 450)

        ring_floor = nodes.new("ShaderNodeMath")
        ring_floor.operation = 'FLOOR'
        links.new(ring.outputs["Value"], ring_floor.inputs[0])
        ring_floor.location = (col_mesh + 900, y - 450)

        is_base_ring = nodes.new("ShaderNodeMath")
        is_base_ring.operation = 'COMPARE'
        is_base_ring.inputs[1].default_value = 0.0
        is_base_ring.inputs[2].default_value = 0.1
        links.new(ring_floor.outputs["Value"], is_base_ring.inputs[0])
        is_base_ring.location = (col_mesh + 1100, y - 450)

        last_ring = nodes.new("ShaderNodeMath")
        last_ring.operation = 'SUBTRACT'
        last_ring.inputs[1].default_value = 1.0
        links.new(resolution_socket, last_ring.inputs[0])
        last_ring.location = (col_mesh + 900, y - 600)

        is_tip_ring = nodes.new("ShaderNodeMath")
        is_tip_ring.operation = 'COMPARE'
        is_tip_ring.inputs[2].default_value = 0.1
        links.new(ring_floor.outputs["Value"], is_tip_ring.inputs[0])
        links.new(last_ring.outputs["Value"], is_tip_ring.inputs[1])
        is_tip_ring.location = (col_mesh + 1100, y - 600)

        # next_index = idx + P, unless already at the tip ring (then +0, a
        # self-sample -- same one-sided fallback `_push_boundary_ring` uses).
        one_minus_tip = nodes.new("ShaderNodeMath")
        one_minus_tip.operation = 'SUBTRACT'
        one_minus_tip.inputs[0].default_value = 1.0
        links.new(is_tip_ring.outputs["Value"], one_minus_tip.inputs[1])
        one_minus_tip.location = (col_mesh + 1300, y - 600)

        next_delta = nodes.new("ShaderNodeMath")
        next_delta.operation = 'MULTIPLY'
        links.new(p_socket, next_delta.inputs[1])
        links.new(one_minus_tip.outputs["Value"], next_delta.inputs[0])
        next_delta.location = (col_mesh + 1500, y - 600)

        next_index = nodes.new("ShaderNodeMath")
        next_index.operation = 'ADD'
        links.new(idx.outputs["Index"], next_index.inputs[0])
        links.new(next_delta.outputs["Value"], next_index.inputs[1])
        next_index.location = (col_mesh + 1700, y - 600)

        next_index_int = nodes.new("FunctionNodeFloatToInt")
        next_index_int.rounding_mode = 'ROUND'
        links.new(next_index.outputs["Value"], next_index_int.inputs["Float"])
        next_index_int.location = (col_mesh + 1900, y - 600)

        # prev_index = idx - P, unless already at the base ring (then +0).
        one_minus_base = nodes.new("ShaderNodeMath")
        one_minus_base.operation = 'SUBTRACT'
        one_minus_base.inputs[0].default_value = 1.0
        links.new(is_base_ring.outputs["Value"], one_minus_base.inputs[1])
        one_minus_base.location = (col_mesh + 1300, y - 750)

        prev_delta = nodes.new("ShaderNodeMath")
        prev_delta.operation = 'MULTIPLY'
        links.new(_math('MULTIPLY', p_socket, -1.0, col_mesh + 1300, y - 850),
                  prev_delta.inputs[1])
        links.new(one_minus_base.outputs["Value"], prev_delta.inputs[0])
        prev_delta.location = (col_mesh + 1500, y - 750)

        prev_index = nodes.new("ShaderNodeMath")
        prev_index.operation = 'ADD'
        links.new(idx.outputs["Index"], prev_index.inputs[0])
        links.new(prev_delta.outputs["Value"], prev_index.inputs[1])
        prev_index.location = (col_mesh + 1700, y - 750)

        prev_index_int = nodes.new("FunctionNodeFloatToInt")
        prev_index_int.rounding_mode = 'ROUND'
        links.new(prev_index.outputs["Value"], prev_index_int.inputs["Float"])
        prev_index_int.location = (col_mesh + 1900, y - 750)

        position = nodes.new("GeometryNodeInputPosition")
        position.location = (col_mesh + 300, y - 950)

        sample_next = nodes.new("GeometryNodeSampleIndex")
        sample_next.data_type = 'FLOAT_VECTOR'
        sample_next.domain = 'POINT'
        links.new(mesh_socket, sample_next.inputs["Geometry"])
        links.new(position.outputs["Position"], sample_next.inputs["Value"])
        links.new(next_index_int.outputs["Integer"], sample_next.inputs["Index"])
        sample_next.location = (col_mesh + 2100, y - 600)

        sample_prev = nodes.new("GeometryNodeSampleIndex")
        sample_prev.data_type = 'FLOAT_VECTOR'
        sample_prev.domain = 'POINT'
        links.new(mesh_socket, sample_prev.inputs["Geometry"])
        links.new(position.outputs["Position"], sample_prev.inputs["Value"])
        links.new(prev_index_int.outputs["Integer"], sample_prev.inputs["Index"])
        sample_prev.location = (col_mesh + 2100, y - 750)

        growth_vec = nodes.new("ShaderNodeVectorMath")
        growth_vec.operation = 'SUBTRACT'
        links.new(sample_next.outputs["Value"], growth_vec.inputs[0])
        links.new(sample_prev.outputs["Value"], growth_vec.inputs[1])
        growth_vec.location = (col_mesh + 2300, y - 700)

        growth_dir = nodes.new("ShaderNodeVectorMath")
        growth_dir.operation = 'NORMALIZE'
        links.new(growth_vec.outputs["Vector"], growth_dir.inputs[0])
        growth_dir.location = (col_mesh + 2500, y - 700)

        # This ring's own edge-to-edge chord -- its "width" direction, and its
        # actual measured width. See `_ring_chord`, shared with
        # `_push_boundary_ring`.
        chord_vec_socket, local_width_socket = _ring_chord(
            mesh_socket, idx.outputs["Index"], cross_idx.outputs["Value"], p_socket,
            x=col_mesh + 700, y=y - 950)

        chord_dir = nodes.new("ShaderNodeVectorMath")
        chord_dir.operation = 'NORMALIZE'
        links.new(chord_vec_socket, chord_dir.inputs[0])
        chord_dir.location = (col_mesh + 1900, y - 1000)

        # width_factor: this ring's own ACTUAL width (the real distance
        # between its two edges, measured live off the mesh) divided by the
        # whorl's nominal peak width, clamped to 1.0 so `cup_socket` still
        # reads as "the push at full width" even where the Width Profile's own
        # peak exceeds 1.0x. Ties Cup's real-world magnitude to the SAME ring
        # the bulge parabola is already keyed to, so it fades out smoothly and
        # exactly where the ribbon's own width does -- see this function's own
        # docstring for why a width-independent version produced a spike at
        # the tip instead.
        width_ref_safe = nodes.new("ShaderNodeMath")
        width_ref_safe.operation = 'MAXIMUM'
        width_ref_safe.inputs[1].default_value = 1e-6
        links.new(width_ref_socket, width_ref_safe.inputs[0])
        width_ref_safe.location = (col_mesh + 1700, y - 1400)

        width_ratio = nodes.new("ShaderNodeMath")
        width_ratio.operation = 'DIVIDE'
        links.new(local_width_socket, width_ratio.inputs[0])
        links.new(width_ref_safe.outputs["Value"], width_ratio.inputs[1])
        width_ratio.location = (col_mesh + 1900, y - 1300)

        width_factor = nodes.new("ShaderNodeMath")
        width_factor.operation = 'MINIMUM'
        width_factor.inputs[1].default_value = 1.0
        links.new(width_ratio.outputs["Value"], width_factor.inputs[0])
        width_factor.location = (col_mesh + 2100, y - 1300)

        push_amount = nodes.new("ShaderNodeMath")
        push_amount.operation = 'MULTIPLY'
        links.new(push_bulge.outputs["Value"], push_amount.inputs[0])
        links.new(width_factor.outputs["Value"], push_amount.inputs[1])
        push_amount.location = (col_mesh + 2300, y - 1300)

        normal_raw = nodes.new("ShaderNodeVectorMath")
        normal_raw.operation = 'CROSS_PRODUCT'
        links.new(chord_dir.outputs["Vector"], normal_raw.inputs[0])
        links.new(growth_dir.outputs["Vector"], normal_raw.inputs[1])
        normal_raw.location = (col_mesh + 2700, y - 850)

        normal_dir = nodes.new("ShaderNodeVectorMath")
        normal_dir.operation = 'NORMALIZE'
        links.new(normal_raw.outputs["Vector"], normal_dir.inputs[0])
        normal_dir.location = (col_mesh + 2900, y - 850)

        offset_vec = nodes.new("ShaderNodeVectorMath")
        offset_vec.operation = 'SCALE'
        links.new(normal_dir.outputs["Vector"], offset_vec.inputs[0])
        links.new(push_amount.outputs["Value"], offset_vec.inputs["Scale"])
        offset_vec.location = (col_mesh + 3100, y - 400)

        set_pos = nodes.new("GeometryNodeSetPosition")
        links.new(mesh_socket, set_pos.inputs["Geometry"])
        links.new(offset_vec.outputs["Vector"], set_pos.inputs["Offset"])
        set_pos.location = (col_mesh + 3300, y)

        return set_pos.outputs["Geometry"]

    def _ribbon_mesh(
        source_geometry, cup_socket, width_ref_socket, y: int, *,
        resolution_socket, top_shape_points_socket, bottom_shape_points_socket,
        p_socket,
    ):
        """A ribbon appendage (Petals or Sepals): the curve's native per-point
        `radius` drives width via Curve to Mesh's own Scale input directly --
        NOT Set Curve Radius, which plant.py's _build_mesh_group already proved
        inert for legacy-curve-sourced geometry in this Blender version. The
        profile is a flat line (Start/Middle/End collinear -- a genuine curved
        profile used to live here for "Curvature", since folded into Cup, see
        `_apply_cup`'s own docstring for why); Resolution is set EXPLICITLY
        (matching `_RIBBON_PROFILE_RESOLUTION`) rather than relying on the
        node's own implicit default, since `_push_boundary_ring` above depends
        on Python and this node graph agreeing on that count exactly. After
        the sweep, applies Cup (see `_apply_cup`, every ring), then pushes the
        tip ring then the base ring (see `_push_boundary_ring`) -- three
        sequential Set Position passes, each composing on top of the previous
        one's output."""
        radius_attr = nodes.new("GeometryNodeInputNamedAttribute")
        radius_attr.data_type = 'FLOAT'
        radius_attr.inputs["Name"].default_value = "radius"
        radius_attr.location = (col_in, y - 150)

        profile = nodes.new("GeometryNodeCurveArc")
        profile.mode = 'POINTS'
        profile.inputs["Resolution"].default_value = _RIBBON_PROFILE_RESOLUTION
        links.new(p_socket, profile.inputs["Resolution"])
        profile.inputs["Start"].default_value = (-0.5, 0.0, 0.0)
        profile.inputs["Middle"].default_value = (0.0, 0.0, 0.0)
        profile.inputs["End"].default_value = (0.5, 0.0, 0.0)
        profile.location = (col_mesh, y)

        to_mesh = nodes.new("GeometryNodeCurveToMesh")
        to_mesh.inputs["Fill Caps"].default_value = True
        to_mesh.location = (col_mesh, y - 300)
        links.new(source_geometry, to_mesh.inputs["Curve"])
        links.new(profile.outputs["Curve"], to_mesh.inputs["Profile Curve"])
        links.new(radius_attr.outputs["Attribute"], to_mesh.inputs["Scale"])

        ribbon = _apply_cup(
            to_mesh.outputs["Mesh"], resolution_socket, cup_socket, width_ref_socket,
            y - 5200, p_socket=p_socket)
        ribbon = _push_boundary_ring(
            ribbon, resolution_socket, top_shape_points_socket, p_socket,
            is_tip=True, y=y - 1600)
        ribbon = _push_boundary_ring(
            ribbon, resolution_socket, bottom_shape_points_socket, p_socket,
            is_tip=False, y=y - 3400)
        return ribbon

    def _revolved_mesh(points_socket_name: str, y: int, *, always_flip: bool = False,
                        concave_socket=None, sides_socket=None):
        """A revolved body (Disc or Receptacle): a SEPARATE child object (see
        _ensure_child_curve_object) so it can use its own CIRCLE profile instead
        of the ribbons' flat line. Revolving a radius-vs-height profile (rather
        than a fixed Mesh primitive) is what makes hemisphere/flat/cone/convex/
        concave all reachable from the same mechanism, just different authored
        control points -- see skeleton/flower.py's `_build_disc`.

        Fill Caps=False deliberately -- filling would seal the rim end (t=0)
        with a flat "plate", which is invisible for a body that closes to a
        point at its other end but becomes a visible plug if it doesn't. Sweeping
        DOWNWARD (Disc with a negative Height, or the Receptacle always) leaves
        the resulting faces pointing into the hollow instead of out toward the
        viewer -- `always_flip=True` (Receptacle, which is always downward by
        construction) unconditionally flips; otherwise `concave_socket` (Disc)
        switches between flipped/unflipped based on the actual signed height,
        computed by Python since the curve object's own geometry doesn't
        otherwise expose that sign directly.

        A third mode -- neither flag given -- skips FlipFaces/Switch entirely
        and returns the meshed geometry as-is: a body that ALWAYS sweeps
        upward from the shared rim (a corolla/calyx Fusion tube, see
        skeleton/flower.py's `petal_fusion`) already has correct winding with
        no sign to switch on, unlike Disc (which can point either way)."""
        obj_info = nodes.new("GeometryNodeObjectInfo")
        obj_info.location = (col_in, y)
        links.new(gin.outputs[points_socket_name], obj_info.inputs["Object"])

        radius_attr = nodes.new("GeometryNodeInputNamedAttribute")
        radius_attr.data_type = 'FLOAT'
        radius_attr.inputs["Name"].default_value = "radius"
        radius_attr.location = (col_in, y - 300)

        circle_profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
        circle_profile.inputs["Resolution"].default_value = _TUBE_SIDES
        if sides_socket is not None:
            links.new(sides_socket, circle_profile.inputs["Resolution"])
        circle_profile.inputs["Radius"].default_value = 1.0
        circle_profile.location = (col_mesh, y)

        to_mesh = nodes.new("GeometryNodeCurveToMesh")
        to_mesh.inputs["Fill Caps"].default_value = False
        to_mesh.location = (col_mesh, y - 300)
        links.new(obj_info.outputs["Geometry"], to_mesh.inputs["Curve"])
        links.new(circle_profile.outputs["Curve"], to_mesh.inputs["Profile Curve"])
        links.new(radius_attr.outputs["Attribute"], to_mesh.inputs["Scale"])
        revolved = to_mesh.outputs["Mesh"]

        if not always_flip and concave_socket is None:
            return revolved

        flipped = nodes.new("GeometryNodeFlipFaces")
        flipped.location = (col_mesh, y - 600)
        links.new(revolved, flipped.inputs["Mesh"])

        if always_flip:
            return flipped.outputs["Mesh"]

        switch = nodes.new("GeometryNodeSwitch")
        switch.input_type = 'GEOMETRY'
        links.new(concave_socket, switch.inputs["Switch"])
        links.new(revolved, switch.inputs["False"])
        links.new(flipped.outputs["Mesh"], switch.inputs["True"])
        switch.location = (col_mesh, y - 900)
        return switch.outputs["Output"]

    def _math(op, a, b=None, x=0, y=0):
        node = nodes.new("ShaderNodeMath")
        node.operation = op
        node.location = (x, y)
        for slot, value in ((0, a), (1, b)):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                node.inputs[slot].default_value = float(value)
            else:
                links.new(value, node.inputs[slot])
        return node.outputs["Value"]

    def _angle_of(vector_socket, x, y):
        sep = nodes.new("ShaderNodeSeparateXYZ")
        sep.location = (x, y)
        links.new(vector_socket, sep.inputs["Vector"])
        return _math('ARCTAN2', sep.outputs["Y"], sep.outputs["X"], x + 180, y), sep

    def _wrap(angle, x, y):
        """An angle folded into (-pi, pi]."""
        return _math('ARCTAN2', _math('SINE', angle, None, x, y),
                     _math('COSINE', angle, None, x, y - 120), x + 180, y)

    def _named_bool(name, x, y):
        node = nodes.new("GeometryNodeInputNamedAttribute")
        node.data_type = 'BOOLEAN'
        node.inputs["Name"].default_value = name
        node.location = (x, y)
        return node.outputs["Attribute"]

    def _store_bool(geometry, name, value, x, y):
        node = nodes.new("GeometryNodeStoreNamedAttribute")
        node.data_type = 'BOOLEAN'
        node.domain = 'POINT'
        node.inputs["Name"].default_value = name
        node.location = (x, y)
        links.new(geometry, node.inputs["Geometry"])
        links.new(value, node.inputs["Value"])
        return node.outputs["Geometry"]

    def _tagged_start(geometry, name, x, y):
        """The lowest vertex index carrying the boolean tag `name`."""
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (x, y - 120)
        stat = nodes.new("GeometryNodeAttributeStatistic")
        stat.data_type = 'FLOAT'
        stat.domain = 'POINT'
        stat.location = (x + 180, y)
        links.new(geometry, stat.inputs["Geometry"])
        links.new(_named_bool(name, x, y - 240), stat.inputs["Selection"])
        links.new(index.outputs["Index"], stat.inputs["Attribute"])
        return stat.outputs["Min"]

    def _tag_rings(tube, sides_socket, x, y):
        """Tag a freshly revolved tube's base ring (`pg_base`, its first
        `sides` vertices) and rim (`pg_rim`, its last) -- the welds find them
        by tag, whatever geometry they end up joined into."""
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (x, y - 120)
        size = nodes.new("GeometryNodeAttributeDomainSize")
        size.component = 'MESH'
        size.location = (x, y - 240)
        links.new(tube, size.inputs["Geometry"])
        on_base = _math('LESS_THAN', index.outputs["Index"],
                        _math('SUBTRACT', sides_socket, 0.5, x + 180, y - 120), x + 360, y - 120)
        on_rim = _math('GREATER_THAN', index.outputs["Index"],
                       _math('SUBTRACT', _math('SUBTRACT', size.outputs["Point Count"],
                                               sides_socket, x + 180, y - 240),
                             0.5, x + 360, y - 240), x + 540, y - 240)
        tagged = _store_bool(_store_bool(tube, "pg_base", on_base, x + 720, y), "pg_rim",
                             on_rim, x + 900, y)
        face_tag = nodes.new("GeometryNodeStoreNamedAttribute")
        face_tag.data_type = 'BOOLEAN'
        face_tag.domain = 'FACE'
        face_tag.inputs["Name"].default_value = "pg_tube_face"
        face_tag.inputs["Value"].default_value = True
        face_tag.location = (x + 1080, y)
        links.new(tagged, face_tag.inputs["Geometry"])
        return face_tag.outputs["Geometry"]

    def _base_weld(tube, receptacle, weld_socket, sides_socket, y: int):
        """Weld the receptacle onto the corolla tube's base: ONE surface from
        the receptacle's underside up the tube. With both matching the disc
        radius their rims share one circle; the receptacle is revolved with
        the tube's side count (`sides_socket`), turned about its axis (under
        half a slot) so its rim vertices land on the tube's base-ring angles,
        snapped onto them exactly (radius and height too), and merged there.
        Returns (tube with the receptacle welded in, receptacle): the second
        empty when `weld_socket` is on, both untouched when off."""
        _mark("Base ring: the tube's first ring -- angle, radius, height")
        x = col_mesh + 400
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (x, y)
        position = nodes.new("GeometryNodeInputPosition")
        position.location = (x, y - 120)

        def sample(geometry, index_socket, yy):
            node = nodes.new("GeometryNodeSampleIndex")
            node.data_type = 'FLOAT_VECTOR'
            node.domain = 'POINT'
            node.location = (x + 360, yy)
            links.new(geometry, node.inputs["Geometry"])
            links.new(position.outputs["Position"], node.inputs["Value"])
            links.new(index_socket, node.inputs["Index"])
            return node.outputs["Value"]

        step = _math('DIVIDE', 2.0 * math.pi, sides_socket, x, y - 240)
        # The tube's base ring: its first tagged vertex's angle, radius, height.
        base0 = sample(tube, _tagged_start(tube, "pg_base", x, y - 400), y - 400)
        phi_b, base_sep = _angle_of(base0, x + 540, y - 400)
        flat = nodes.new("ShaderNodeCombineXYZ")
        flat.location = (x + 540, y - 520)
        links.new(base_sep.outputs["X"], flat.inputs["X"])
        links.new(base_sep.outputs["Y"], flat.inputs["Y"])
        radius = nodes.new("ShaderNodeVectorMath")
        radius.operation = 'LENGTH'
        radius.location = (x + 720, y - 520)
        links.new(flat.outputs["Vector"], radius.inputs[0])
        # Turn the receptacle so its rim vertex 0 sits on a base-ring angle.
        _mark("Turn the receptacle: its rim onto the base ring's angles")
        zero = nodes.new("FunctionNodeInputInt")
        zero.integer = 0
        zero.location = (x, y - 700)
        rec0 = sample(receptacle, zero.outputs["Integer"], y - 700)
        alpha0, _sep = _angle_of(rec0, x + 540, y - 700)
        off = _wrap(_math('SUBTRACT', phi_b, alpha0, x + 900, y - 700), x + 1080, y - 700)
        delta = _math('SUBTRACT', off, _math('MULTIPLY', step, _math(
            'ROUND', _math('DIVIDE', off, step, x + 1440, y - 780), None, x + 1620, y - 780),
            x + 1800, y - 780), x + 1980, y - 700)
        turned = nodes.new("ShaderNodeVectorRotate")
        turned.rotation_type = 'Z_AXIS'
        turned.location = (x + 2160, y - 700)
        links.new(position.outputs["Position"], turned.inputs["Vector"])
        links.new(delta, turned.inputs["Angle"])
        rec_turned = nodes.new("GeometryNodeSetPosition")
        rec_turned.location = (x + 2340, y)
        links.new(receptacle, rec_turned.inputs["Geometry"])
        links.new(turned.outputs["Vector"], rec_turned.inputs["Position"])
        # Snap its rim ring (ring 0: the revolve starts at the rim) exactly
        # onto the base ring's vertices.
        _mark("Snap: the receptacle's rim exactly onto the base ring")
        rec_rim = _math('LESS_THAN', index.outputs["Index"],
                        _math('SUBTRACT', sides_socket, 0.5, x + 2160, y - 900),
                        x + 2340, y - 900)
        alpha, _sep = _angle_of(position.outputs["Position"], x + 2160, y - 1000)
        slot = _math('ROUND', _math('DIVIDE', _wrap(_math('SUBTRACT', alpha, phi_b,
                                                          x + 2520, y - 1000),
                                                    x + 2700, y - 1000),
                                    step, x + 3060, y - 1000), None, x + 3240, y - 1000)
        theta = _math('ADD', phi_b, _math('MULTIPLY', slot, step, x + 3420, y - 1000),
                      x + 3600, y - 1000)
        target = nodes.new("ShaderNodeCombineXYZ")
        target.location = (x + 3960, y - 1000)
        links.new(_math('MULTIPLY', radius.outputs["Value"],
                        _math('COSINE', theta, None, x + 3780, y - 960), x + 3780, y - 960),
                  target.inputs["X"])
        links.new(_math('MULTIPLY', radius.outputs["Value"],
                        _math('SINE', theta, None, x + 3780, y - 1080), x + 3780, y - 1080),
                  target.inputs["Y"])
        links.new(base_sep.outputs["Z"], target.inputs["Z"])
        # The receptacle is revolved downward and flipped to face out on its
        # own (`always_flip`); joined to the upward tube that flip runs against
        # the tube's winding across the seam, so it is undone here.
        unflip = nodes.new("GeometryNodeFlipFaces")
        unflip.location = (x + 2520, y + 200)
        links.new(rec_turned.outputs["Geometry"], unflip.inputs["Mesh"])
        snap = nodes.new("GeometryNodeSetPosition")
        snap.location = (x + 4140, y)
        links.new(unflip.outputs["Mesh"], snap.inputs["Geometry"])
        links.new(rec_rim, snap.inputs["Selection"])
        links.new(target.outputs["Vector"], snap.inputs["Position"])
        # Merge the two rings (tagged), nothing else.
        _mark("Merge the two rings; Receptacle Tube Weld on / off")
        rec_tagged = _store_bool(snap.outputs["Geometry"], "pg_base_weld", rec_rim,
                                 x + 4320, y)
        tube_tagged = _store_bool(tube, "pg_base_weld", _named_bool("pg_base", x + 4140, y - 300),
                                  x + 4320, y - 300)
        both = nodes.new("GeometryNodeJoinGeometry")
        both.location = (x + 4500, y)
        links.new(tube_tagged, both.inputs["Geometry"])
        links.new(rec_tagged, both.inputs["Geometry"])
        merge = nodes.new("GeometryNodeMergeByDistance")
        merge.location = (x + 4680, y)
        links.new(both.outputs["Geometry"], merge.inputs["Geometry"])
        links.new(_named_bool("pg_base_weld", x + 4500, y - 200), merge.inputs["Selection"])
        merge.inputs["Distance"].default_value = _WELD_DISTANCE
        untag = nodes.new("GeometryNodeRemoveAttribute")
        untag.location = (x + 4860, y)
        links.new(merge.outputs["Geometry"], untag.inputs["Geometry"])
        untag.inputs["Name"].default_value = "pg_base_weld"
        tube_out = nodes.new("GeometryNodeSwitch")
        tube_out.input_type = 'GEOMETRY'
        tube_out.location = (x + 5040, y)
        links.new(weld_socket, tube_out.inputs["Switch"])
        links.new(tube, tube_out.inputs["False"])
        links.new(untag.outputs["Geometry"], tube_out.inputs["True"])
        rec_out = nodes.new("GeometryNodeSwitch")
        rec_out.input_type = 'GEOMETRY'
        rec_out.location = (x + 5040, y - 200)
        links.new(weld_socket, rec_out.inputs["Switch"])
        links.new(receptacle, rec_out.inputs["False"])        # True: welded in
        return tube_out.outputs["Output"], rec_out.outputs["Output"]

    def _on_domain(field, domain, x, y):
        node = nodes.new("GeometryNodeFieldOnDomain")
        node.data_type = 'FLOAT'
        node.domain = domain
        node.location = (x, y)
        links.new(field, node.inputs["Value"])
        return node.outputs["Value"]

    def _store_uv(geometry, u, v, x, y, selection=None):
        """Write the `UVMap` (face corners) from per-corner U and V fields."""
        uv = nodes.new("ShaderNodeCombineXYZ")
        uv.location = (x, y - 200)
        links.new(u, uv.inputs["X"])
        links.new(v, uv.inputs["Y"])
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = 'FLOAT2'
        store.domain = 'CORNER'
        store.inputs["Name"].default_value = "UVMap"
        store.location = (x + 180, y)
        links.new(geometry, store.inputs["Geometry"])
        links.new(uv.outputs["Vector"], store.inputs["Value"])
        if selection is not None:
            links.new(selection, store.inputs["Selection"])
        return store.outputs["Geometry"]

    def _named(name, data_type, x, y):
        node = nodes.new("GeometryNodeInputNamedAttribute")
        node.data_type = data_type
        node.inputs["Name"].default_value = name
        node.location = (x, y)
        return node.outputs["Attribute"]

    def _store_corner(geometry, name, data_type, value, x, y, selection=None):
        node = nodes.new("GeometryNodeStoreNamedAttribute")
        node.data_type = data_type
        node.domain = 'CORNER'
        node.inputs["Name"].default_value = name
        node.location = (x, y)
        links.new(geometry, node.inputs["Geometry"])
        links.new(value, node.inputs["Value"])
        if selection is not None:
            links.new(selection, node.inputs["Selection"])
        return node.outputs["Geometry"]

    def _uv_place(centre_socket, lx, ly, angle, x, y):
        """A medallion point: `centre + scale * R(angle) * (lx, ly)` -- (U, V)."""
        c = nodes.new("ShaderNodeSeparateXYZ")
        c.location = (x, y - 200)
        links.new(centre_socket, c.inputs["Vector"])
        ca = _math('COSINE', angle, None, x, y)
        sa = _math('SINE', angle, None, x, y - 60)
        scale = gin.outputs[_UV_SCALE_SOCKET]
        rx = _math('SUBTRACT', _math('MULTIPLY', lx, ca, x + 180, y),
                   _math('MULTIPLY', ly, sa, x + 180, y - 60), x + 360, y)
        ry = _math('ADD', _math('MULTIPLY', lx, sa, x + 180, y - 120),
                   _math('MULTIPLY', ly, ca, x + 180, y - 180), x + 360, y - 120)
        u = _math('ADD', c.outputs["X"], _math('MULTIPLY', rx, scale, x + 540, y), x + 720, y)
        v = _math('ADD', c.outputs["Y"], _math('MULTIPLY', ry, scale, x + 540, y - 120),
                  x + 720, y - 120)
        return u, v

    def _uv_fan(centre_socket, rho, arc, x, y):
        """A point of an unrolled tube (a fan, `_fan_piece`): `rho` from the
        fan's apex, `arc` round its ring from the fan's middle line -- the
        fan opening upward, `centre + scale * R(pi/2) * rho (cos, sin)(arc / rho)`."""
        psi = _math('DIVIDE', arc, _math('MAXIMUM', rho, 1e-9, x, y - 60), x + 180, y)
        lx = _math('MULTIPLY', rho, _math('COSINE', psi, None, x + 360, y), x + 540, y)
        ly = _math('MULTIPLY', rho, _math('SINE', psi, None, x + 360, y - 60), x + 540, y - 60)
        return _uv_place(centre_socket, lx, ly, 0.5 * math.pi, x + 720, y)

    def _uv_record(geometry, part, record_field, across_field, x, y, flip_after=None,
                   along_push=None):
        """Each corner's UV from its vertex's record (see `_uv_layout`): the
        carrier point for its ring, by mode --
          1 a circle: x its equal-area radius, round the part's centre --
            turning the way that keeps its UVs wound like every other
            piece's: from each face's own winding (`flip_after`: faces
            flipped later, the welded receptacle);
          3 a tube ring unrolled into a fan (`_uv_fan`): x its distance from
            the apex, y its open radius; cut at the -X side (a corner takes
            its FACE's side of the cut);
          0 a laid-flat strip, upright: x the row's height, y its width, z
            the strip's middle (U) -- an end row moved along by Top / Bottom
            Shape as the mesh's is (`along_push`, in row widths);
        written straight onto the FACE CORNERS, which a weld's merge never
        averages (vertex attributes it would). The ring is kept on the
        corners too, for the weld to finish a fused whorl (`pg_uv_ring`)."""
        _mark("UV record: this vertex's carrier point and mode")
        info = nodes.new("GeometryNodeObjectInfo")
        info.transform_space = 'ORIGINAL'
        info.location = (x, y)
        links.new(gin.outputs[_UV_CARRIER_SOCKETS[part]], info.inputs["Object"])
        position = nodes.new("GeometryNodeInputPosition")
        position.location = (x, y - 160)
        point = nodes.new("GeometryNodeSampleIndex")
        point.data_type = 'FLOAT_VECTOR'
        point.domain = 'POINT'
        point.location = (x + 180, y)
        links.new(info.outputs["Geometry"], point.inputs["Geometry"])
        links.new(position.outputs["Position"], point.inputs["Value"])
        links.new(record_field, point.inputs["Index"])
        mode = nodes.new("GeometryNodeSampleIndex")
        mode.data_type = 'FLOAT'
        mode.domain = 'POINT'
        mode.location = (x + 180, y - 200)
        links.new(info.outputs["Geometry"], mode.inputs["Geometry"])
        links.new(_named("radius", 'FLOAT', x, y - 300), mode.inputs["Value"])
        links.new(record_field, mode.inputs["Index"])
        rec = nodes.new("ShaderNodeSeparateXYZ")
        rec.location = (x + 360, y)
        links.new(point.outputs["Value"], rec.inputs["Vector"])
        # index-derived values are the VERTEX's: evaluate them on points
        rx = _on_domain(rec.outputs["X"], 'POINT', x + 540, y)
        rz = _on_domain(rec.outputs["Z"], 'POINT', x + 540, y - 60)
        m = _on_domain(mode.outputs["Value"], 'POINT', x + 540, y - 120)
        ry = (_on_domain(_math('MULTIPLY', across_field, rec.outputs["Y"], x + 360, y - 180),
                         'POINT', x + 540, y - 180)
              if across_field is not None else None)
        pos = nodes.new("ShaderNodeSeparateXYZ")
        pos.location = (x + 540, y - 300)
        links.new(position.outputs["Position"], pos.inputs["Vector"])
        own = _math('ARCTAN2', pos.outputs["Y"], pos.outputs["X"], x + 720, y - 300)
        centre = gin.outputs[_UV_CENTER_SOCKETS[part]]
        # the circle's turn: its map (out from the pole, round the axis) is
        # wound like the face exactly when (out x round) . normal > 0 --
        # then mirrored, so every piece reads the same way
        _mark("Circle mode: mirrored so its UVs wind like the face")
        def face_vector(vector, yy, domain='FACE'):
            node = nodes.new("GeometryNodeFieldOnDomain")
            node.data_type = 'FLOAT_VECTOR'
            node.domain = domain
            node.location = (x + 360, yy)
            links.new(vector, node.inputs["Value"])
            return node.outputs["Value"]
        weighted = nodes.new("ShaderNodeVectorMath")
        weighted.operation = 'SCALE'
        weighted.location = (x + 180, y - 900)
        links.new(position.outputs["Position"], weighted.inputs[0])
        links.new(rx, weighted.inputs["Scale"])
        face_pos = face_vector(position.outputs["Position"], y - 1000)
        mean_pos = nodes.new("ShaderNodeVectorMath")
        mean_pos.operation = 'SCALE'
        mean_pos.location = (x + 540, y - 1000)
        links.new(face_pos, mean_pos.inputs[0])
        links.new(_on_domain(rx, 'FACE', x + 360, y - 1100), mean_pos.inputs["Scale"])
        out = nodes.new("ShaderNodeVectorMath")              # the way rx grows
        out.operation = 'SUBTRACT'
        out.location = (x + 720, y - 900)
        # (the product per VERTEX, then averaged over the face)
        links.new(face_vector(face_vector(weighted.outputs["Vector"], y - 850, 'POINT'), y - 900),
                  out.inputs[0])
        links.new(mean_pos.outputs["Vector"], out.inputs[1])
        fp = nodes.new("ShaderNodeSeparateXYZ")
        fp.location = (x + 540, y - 1200)
        links.new(face_pos, fp.inputs["Vector"])
        round_dir = nodes.new("ShaderNodeCombineXYZ")
        round_dir.location = (x + 720, y - 1200)
        links.new(_math('MULTIPLY', fp.outputs["Y"], -1.0, x + 720, y - 1150),
                  round_dir.inputs["X"])
        links.new(fp.outputs["X"], round_dir.inputs["Y"])
        cross = nodes.new("ShaderNodeVectorMath")
        cross.operation = 'CROSS_PRODUCT'
        cross.location = (x + 900, y - 1000)
        links.new(out.outputs["Vector"], cross.inputs[0])
        links.new(round_dir.outputs["Vector"], cross.inputs[1])
        normal = nodes.new("GeometryNodeInputNormal")
        normal.location = (x + 720, y - 1300)
        agree = nodes.new("ShaderNodeVectorMath")
        agree.operation = 'DOT_PRODUCT'
        agree.location = (x + 1080, y - 1000)
        links.new(cross.outputs["Vector"], agree.inputs[0])
        links.new(face_vector(normal.outputs["Normal"], y - 1300), agree.inputs[1])
        mirror = _math('GREATER_THAN', agree.outputs["Value"], 0.0, x + 1260, y - 1000)
        if flip_after is not None:
            mirror = _math('ABSOLUTE', _math('SUBTRACT', mirror, flip_after, x + 1260, y - 1100),
                           None, x + 1440, y - 1100)
        turn = _math('MULTIPLY', own, _math('SUBTRACT', 1.0, _math(
            'MULTIPLY', mirror, 2.0, x + 1440, y - 1000), x + 1620, y - 1000), x + 1800, y - 1000)
        _mark("UVs placed per mode: circle, upright strip, unrolled fan")
        radial_u, radial_v = _uv_place(centre, rx, 0.0, turn, x + 900, y)
        strip_along = rx if along_push is None else _math('ADD', rx, _on_domain(_math(
            'MULTIPLY', along_push, rec.outputs["Y"], x + 360, y - 520), 'POINT',
            x + 540, y - 520), x + 720, y - 520)
        strip_u, strip_v = _uv_place(centre, strip_along, _math(
            'SUBTRACT', ry if ry is not None else 0.0, rz, x + 720, y - 240), 0.5 * math.pi,
            x + 900, y - 300)
        face_angle = _math('ARCTAN2', _on_domain(pos.outputs["Y"], 'FACE', x + 540, y - 600),
                           _on_domain(pos.outputs["X"], 'FACE', x + 540, y - 660),
                           x + 720, y - 600)
        phi = _math('ADD', face_angle, _wrap(_math('SUBTRACT', own, face_angle, x + 720, y - 700),
                                             x + 900, y - 700), x + 1080, y - 600)
        fan_u, fan_v = _uv_fan(centre, rx, _math('MULTIPLY', _on_domain(
            rec.outputs["Y"], 'POINT', x + 900, y - 800), phi, x + 1080, y - 800),
            x + 1260, y - 600)
        _mark("The vertex's mode picked; UVs written to face corners")
        is_radial = _math('MULTIPLY', _math('GREATER_THAN', m, 0.5, x + 720, y - 400),
                          _math('LESS_THAN', m, 1.5, x + 720, y - 460), x + 900, y - 420)
        is_fan = _math('MULTIPLY', _math('GREATER_THAN', m, 2.5, x + 1980, y - 400),
                       _math('LESS_THAN', m, 3.5, x + 1980, y - 460), x + 2160, y - 420)
        u = _math('ADD', _math('ADD', strip_u, _math('MULTIPLY', is_radial, _math(
            'SUBTRACT', radial_u, strip_u, x + 1620, y), x + 1800, y), x + 1980, y),
            _math('MULTIPLY', is_fan, _math('SUBTRACT', fan_u, strip_u, x + 2160, y - 200),
                  x + 2340, y - 200), x + 2520, y - 100)
        v = _math('ADD', _math('ADD', strip_v, _math('MULTIPLY', is_radial, _math(
            'SUBTRACT', radial_v, strip_v, x + 1620, y - 120), x + 1800, y - 120),
            x + 1980, y - 120),
            _math('MULTIPLY', is_fan, _math('SUBTRACT', fan_v, strip_v, x + 2160, y - 320),
                  x + 2340, y - 320), x + 2520, y - 220)
        g = _store_uv(geometry, u, v, x + 2700, y)
        return _store_corner(g, "pg_uv_ring", 'FLOAT', _on_domain(record_field, 'POINT',
                                                                  x + 2340, y + 150),
                             x + 2880, y + 150)

    def _uv_ribbon(geometry, part, resolution_socket, p_socket, y):
        """A petal / sepal whorl's records: spline (petal) j, ring i -> record
        2P + j * N + i (the carrier's order, after the open flower's Top and
        Bottom Shape values, P each), and the vertex's place across its ring
        (-0.5..0.5) for a laid-flat strip's width."""
        _mark("Strip record: petal, ring, place across the ring")
        x = col_mesh + 500
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (x, y)
        per = _math('MULTIPLY', resolution_socket, p_socket, x, y - 120)
        petal = _math('FLOOR', _math('DIVIDE', index.outputs["Index"], per, x + 180, y),
                      None, x + 360, y)
        local = _math('FLOORED_MODULO', index.outputs["Index"], per, x + 180, y - 120)
        ring = _math('FLOOR', _math('DIVIDE', local, p_socket, x + 360, y - 120),
                     None, x + 540, y - 120)
        k = _math('SUBTRACT', local, _math('MULTIPLY', ring, p_socket, x + 540, y - 240),
                  x + 720, y - 120)
        record = _math('ADD', _math('ADD', _math('MULTIPLY', petal, resolution_socket,
                                                 x + 540, y), ring, x + 720, y),
                       _math('MULTIPLY', p_socket, 2.0, x + 540, y + 60), x + 900, y + 60)
        across = _math('SUBTRACT', _math('DIVIDE', k, _math('SUBTRACT', p_socket, 1.0,
                                                            x + 720, y - 240),
                                         x + 900, y - 120), 0.5, x + 1080, y - 120)
        # Top / Bottom Shape move the tip / base ring along the ribbon by a
        # value per cross-section point times the ring's width
        # (`_push_boundary_ring`): the strip's end rows follow -- by the OPEN
        # flower's values (they change with Bloom), kept on the UV carrier
        _mark("End rows moved like the mesh's Top / Bottom Shape")
        info = nodes.new("GeometryNodeObjectInfo")
        info.transform_space = 'ORIGINAL'
        info.location = (x, y - 400)
        links.new(gin.outputs[_UV_CARRIER_SOCKETS[part]], info.inputs["Object"])
        position = nodes.new("GeometryNodeInputPosition")
        position.location = (x, y - 550)

        def shape(offset, yy):
            value = nodes.new("GeometryNodeSampleIndex")
            value.data_type = 'FLOAT_VECTOR'
            value.domain = 'POINT'
            value.location = (x + 180, yy)
            links.new(info.outputs["Geometry"], value.inputs["Geometry"])
            links.new(position.outputs["Position"], value.inputs["Value"])
            links.new(_math('ADD', k, offset, x, yy - 100), value.inputs["Index"])
            sep = nodes.new("ShaderNodeSeparateXYZ")
            sep.location = (x + 360, yy)
            links.new(value.outputs["Value"], sep.inputs["Vector"])
            return sep.outputs["X"]
        is_tip = _math('COMPARE', ring, _math('SUBTRACT', resolution_socket, 1.0,
                                               x + 540, y - 520), x + 720, y - 500)
        is_tip.node.inputs[2].default_value = 0.1
        is_base = _math('LESS_THAN', ring, 0.5, x + 720, y - 600)
        push = _math('SUBTRACT', _math('MULTIPLY', is_tip, shape(0.0, y - 400),
                                       x + 900, y - 450),
                     _math('MULTIPLY', is_base, shape(p_socket, y - 700), x + 900, y - 650),
                     x + 1080, y - 500)
        return _uv_record(geometry, part, record, across, x + 1260, y, along_push=push)

    def _uv_revolved(geometry, part, sides_socket, y, flip_after=None):
        """A revolved body's records: one per ring (its profile point)."""
        x = col_mesh + 500
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (x, y)
        ring = _math('FLOOR', _math('DIVIDE', index.outputs["Index"], sides_socket, x + 180, y),
                     None, x + 360, y)
        return _uv_record(geometry, part, ring, None, x + 540, y, flip_after)

    def _tube_weld(ribbon, tube, weld_socket, count_socket, resolution_socket, sides_socket,
                   y: int, p_socket, uv_centre_socket, uv_grid_socket):
        """Weld a whorl's first-layer lobes into its tube: ONE surface.

        A Merge by Distance alone cannot do it (see the note above the final
        Join): the tube's rim and a lobe's base ring are spaced differently
        round the same circle. So the tube is revolved with exactly
        `count * (P - 1)` sides (`sides_socket`, P the lobe's cross-section
        points), and each lobe's base ring is SNAPPED onto those rim vertices:
        lobe j takes P consecutive rim slots, centred on its own azimuth (the
        middle of its base) and running the way its base runs along the rim --
        so neighbouring lobes share their edge vertex and every rim vertex has
        a lobe vertex on it. The slots are measured off the tube's own first
        rim vertex (radius, height and angle), whatever angle the circle
        profile starts at -- the tube turned a fraction of a slot so lobe
        centres fall between rim vertices -- and a lobe running against the
        rim's vertex order is flipped, so the winding is one across the seam.
        Then only those vertices are merged. Returns (lobes, tube):
        welded into the first with the second empty when `weld_socket` is on,
        both untouched when off."""
        _mark("Each vertex: its lobe, ring and place across (k)")
        x = col_mesh + 400
        P = p_socket                        # points across a lobe
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (x, y)
        position = nodes.new("GeometryNodeInputPosition")
        position.location = (x, y - 120)

        # This vertex's lobe (j), ring and cross-section position (k).
        per_lobe = _math('MULTIPLY', resolution_socket, P, x, y - 240)
        lobe = _math('FLOOR', _math('DIVIDE', index.outputs["Index"], per_lobe, x, y - 360),
                     None, x + 180, y - 360)
        local = _math('SUBTRACT', index.outputs["Index"],
                      _math('MULTIPLY', lobe, per_lobe, x + 180, y - 480), x + 360, y - 480)
        ring = _math('FLOOR', _math('DIVIDE', local, P, x + 360, y - 600), None, x + 540, y - 600)
        k = _math('SUBTRACT', local, _math('MULTIPLY', ring, P, x + 540, y - 720), x + 720, y - 720)
        is_base = _math('LESS_THAN', ring, 0.5, x + 720, y - 840)
        first_layer = _math('LESS_THAN', lobe, _math('SUBTRACT', count_socket, 0.5,
                                                     x + 720, y - 960), x + 900, y - 960)
        base_mask = nodes.new("FunctionNodeBooleanMath")
        base_mask.operation = 'AND'
        base_mask.location = (x + 1080, y - 900)
        links.new(is_base, base_mask.inputs[0])
        links.new(first_layer, base_mask.inputs[1])

        def sample(geometry, index_socket, yy, value=None):
            node = nodes.new("GeometryNodeSampleIndex")
            node.data_type = 'FLOAT_VECTOR'
            node.domain = 'POINT'
            node.location = (x + 1260, yy)
            links.new(geometry, node.inputs["Geometry"])
            links.new(value or position.outputs["Position"], node.inputs["Value"])
            links.new(index_socket, node.inputs["Index"])
            return node.outputs["Value"]

        def lobe_frame(lobe_field, yy):
            """A lobe's azimuth (the angle of the middle of its base) and which
            way round the rim its base runs from vertex 0 to P-1 (+1 ccw, -1
            cw: the sign of that chord along the rim's tangent there, ccw when
            it has none). Not its ends' own angles: a lobe leaving the rim
            straight up has a cross-section frame Blender picks for a vertical
            curve, so its base can run radially -- both ends at one angle."""
            start = _math('MULTIPLY', lobe_field, per_lobe, x + 1080, yy)
            end0 = sample(ribbon, start, yy)
            end1 = sample(ribbon, _math('ADD', start, _math('SUBTRACT', P, 1.0, x + 900, yy - 180),
                                    x + 1080, yy - 180), yy - 180)
            middle = nodes.new("ShaderNodeVectorMath")
            middle.operation = 'ADD'
            middle.location = (x + 1440, yy - 90)
            links.new(end0, middle.inputs[0])
            links.new(end1, middle.inputs[1])
            azimuth, _sep = _angle_of(middle.outputs["Vector"], x + 1620, yy - 90)
            chord = nodes.new("ShaderNodeVectorMath")
            chord.operation = 'SUBTRACT'
            chord.location = (x + 1440, yy - 220)
            links.new(end1, chord.inputs[0])
            links.new(end0, chord.inputs[1])
            tangent = nodes.new("ShaderNodeCombineXYZ")
            tangent.location = (x + 1980, yy - 220)
            links.new(_math('MULTIPLY', _math('SINE', azimuth, None, x + 1800, yy - 220), -1.0,
                            x + 1800, yy - 260), tangent.inputs["X"])
            links.new(_math('COSINE', azimuth, None, x + 1800, yy - 340), tangent.inputs["Y"])
            along = nodes.new("ShaderNodeVectorMath")
            along.operation = 'DOT_PRODUCT'
            along.location = (x + 2160, yy - 220)
            links.new(chord.outputs["Vector"], along.inputs[0])
            links.new(tangent.outputs["Vector"], along.inputs[1])
            runs_cw = _math('LESS_THAN', along.outputs["Value"], 0.0, x + 2340, yy - 220)
            return azimuth, _math('SUBTRACT', 1.0,
                                  _math('MULTIPLY', runs_cw, 2.0, x + 2520, yy - 220),
                                  x + 2700, yy - 220)

        _mark("Each lobe: its azimuth and the way its base runs")
        azimuth, direction = lobe_frame(lobe, y - 1080)

        _mark("The tube's rim: start, radius, the way it runs")
        # The tube's rim: the ring tagged `pg_rim` when it was revolved (its
        # last ring -- found by tag, not index, since a welded receptacle may
        # sit in the same geometry), from its lowest index.
        rim_start = _tagged_start(tube, "pg_rim", x, y - 1500)
        rim0 = sample(tube, rim_start, y - 1500)
        phi_raw, rim_sep = _angle_of(rim0, x + 1440, y - 1500)
        radius = nodes.new("ShaderNodeVectorMath")
        radius.operation = 'LENGTH'
        radius.location = (x + 1620, y - 1620)
        flat = nodes.new("ShaderNodeCombineXYZ")
        flat.location = (x + 1440, y - 1620)
        links.new(rim_sep.outputs["X"], flat.inputs["X"])
        links.new(rim_sep.outputs["Y"], flat.inputs["Y"])
        links.new(flat.outputs["Vector"], radius.inputs[0])
        # Which way the tube's own rim vertices run (+1 ccw, -1 cw).
        rim1 = sample(tube, _math('ADD', rim_start, 1.0, x + 1080, y - 1740), y - 1740)
        phi1, _sep = _angle_of(rim1, x + 1440, y - 1740)
        rim_ccw = _math('GREATER_THAN', _wrap(_math('SUBTRACT', phi1, phi_raw,
                                                    x + 1800, y - 1740),
                                              x + 1980, y - 1740), 0.0, x + 2340, y - 1740)
        rim_dir = _math('SUBTRACT', _math('MULTIPLY', rim_ccw, 2.0, x + 2520, y - 1740), 1.0,
                        x + 2700, y - 1740)

        _mark("Turn the tube: lobe slots on whole rim vertices")
        # Turn the tube about its axis (by under half a slot -- invisible on a
        # circle) so a lobe's first slot is a whole number, never an exact .5
        # (which rounds either way on float noise and slid a lobe one slot
        # over its neighbour): a lobe spans P - 1 slots, so its centre falls
        # BETWEEN rim vertices when P - 1 is odd, ON one when it is even.
        step = _math('DIVIDE', 2.0 * math.pi, sides_socket, x + 1620, y - 1380)
        zero = nodes.new("FunctionNodeInputInt")
        zero.integer = 0
        zero.location = (x + 900, y - 1900)
        az0, _d0 = lobe_frame(zero.outputs["Integer"], y - 1900)
        spans = _math('SUBTRACT', P, 1.0, x + 2520, y - 2000)
        odd = _math('SUBTRACT', spans, _math('MULTIPLY', _math(
            'FLOOR', _math('MULTIPLY', spans, 0.5, x + 2700, y - 2000), None, x + 2700, y - 2050),
            2.0, x + 2700, y - 2100), x + 2880, y - 2000)
        off = _wrap(_math('SUBTRACT', _math('ADD', az0, _math(
            'MULTIPLY', step, _math('MULTIPLY', odd, 0.5, x + 2880, y - 2050),
            x + 2880, y - 1900), x + 3060, y - 1900), phi_raw, x + 3240, y - 1900),
                    x + 3420, y - 1900)
        # The turn is a whole slot either way of a half-slot tie whenever lobe 0
        # sits on the tube's start angle (the usual case: both at 0) -- ROUND
        # then took either side on float noise, so as Bloom moved a lobe's
        # base by a hair the whole ring welded one slot over (same vertices,
        # other faces: no two Bloom stages of one flower shared a topology).
        # Nudged off the tie, one side always wins.
        delta = _math('SUBTRACT', off, _math('MULTIPLY', step, _math(
            'ROUND', _math('ADD', _math('DIVIDE', off, step, x + 3780, y - 1980), _SLOT_TIE,
                           x + 3870, y - 1980), None, x + 3960, y - 1980),
            x + 4140, y - 1980), x + 4320, y - 1900)
        phi0 = _math('ADD', phi_raw, delta, x + 4500, y - 1900)
        turned = nodes.new("ShaderNodeVectorRotate")
        turned.rotation_type = 'Z_AXIS'
        turned.location = (x + 4500, y - 2100)
        links.new(position.outputs["Position"], turned.inputs["Vector"])
        links.new(delta, turned.inputs["Angle"])
        tube_turned = nodes.new("GeometryNodeSetPosition")
        tube_turned.location = (x + 4680, y - 2100)
        links.new(tube, tube_turned.inputs["Geometry"])
        links.new(turned.outputs["Vector"], tube_turned.inputs["Position"])

        _mark("Each base vertex's rim slot: where it snaps to")
        # The lobe's P slots, centred on its azimuth; this vertex's slot.
        centre = _math('DIVIDE', _wrap(_math('SUBTRACT', azimuth, phi0, x + 1800, y - 1460),
                                       x + 1980, y - 1460), step, x + 2340, y - 1460)
        first_slot = _math('ROUND', _math('SUBTRACT', centre,
                                          _math('MULTIPLY', direction, _math(
                                              'MULTIPLY', _math('SUBTRACT', P, 1.0,
                                                                x + 2160, y - 1660),
                                              0.5, x + 2340, y - 1660),
                                                x + 2340, y - 1560), x + 2520, y - 1460),
                           None, x + 2700, y - 1460)
        slot = _math('ADD', first_slot, _math('MULTIPLY', direction, k, x + 2880, y - 1300),
                     x + 3060, y - 1380)
        theta = _math('ADD', phi0, _math('MULTIPLY', slot, step, x + 3240, y - 1380),
                      x + 3420, y - 1380)
        target = nodes.new("ShaderNodeCombineXYZ")
        target.location = (x + 3780, y - 1380)
        links.new(_math('MULTIPLY', radius.outputs["Value"],
                        _math('COSINE', theta, None, x + 3600, y - 1300), x + 3600, y - 1300),
                  target.inputs["X"])
        links.new(_math('MULTIPLY', radius.outputs["Value"],
                        _math('SINE', theta, None, x + 3600, y - 1460), x + 3600, y - 1460),
                  target.inputs["Y"])
        links.new(rim_sep.outputs["Z"], target.inputs["Z"])

        _mark("Wrap: each lobe row arced round the flower's axis")
        # Wrap: a fused lobe is part of the corolla, so each of its rows
        # follows the corolla's round surface instead of a straight strip --
        # the part of a vertex's offset from its row's middle that runs round
        # the flower's axis becomes an arc at the row's own radius (same
        # length), the rest (Cup's bow, the row's lean) kept. An upright lobe
        # becomes a shell continuing the bell, a flat limb stays flat (an arc
        # in its plane), a lobe folded in for the bud curves into a dome. The
        # turn is held to one lobe's share of the circle (pi / count), so a
        # bud's converging lobes close up instead of overlapping.
        row_start = _math('ADD', _math('MULTIPLY', lobe, per_lobe, x + 1080, y - 2800),
                          _math('MULTIPLY', ring, P, x + 1080, y - 2900), x + 1260, y - 2800)
        mid_a = sample(ribbon, _math('ADD', row_start, _math('FLOOR', _math(
            'MULTIPLY', _math('SUBTRACT', P, 1.0, x + 900, y - 2950), 0.5, x + 1080, y - 2950),
            None, x + 1080, y - 3000), x + 1260, y - 2950), y - 2950)
        mid_b = sample(ribbon, _math('ADD', row_start, _math('FLOOR', _math(
            'MULTIPLY', P, 0.5, x + 1080, y - 3100), None, x + 1080, y - 3150),
            x + 1260, y - 3100), y - 3100)
        middle_sum = nodes.new("ShaderNodeVectorMath")
        middle_sum.operation = 'ADD'
        middle_sum.location = (x + 1440, y - 3000)
        links.new(mid_a, middle_sum.inputs[0])
        links.new(mid_b, middle_sum.inputs[1])
        row_mid = nodes.new("ShaderNodeVectorMath")
        row_mid.operation = 'SCALE'
        row_mid.location = (x + 1620, y - 3000)
        links.new(middle_sum.outputs["Vector"], row_mid.inputs[0])
        row_mid.inputs["Scale"].default_value = 0.5
        mid_angle, mid_sep = _angle_of(row_mid.outputs["Vector"], x + 1800, y - 3000)
        mid_flat = nodes.new("ShaderNodeCombineXYZ")
        mid_flat.location = (x + 1980, y - 3100)
        links.new(mid_sep.outputs["X"], mid_flat.inputs["X"])
        links.new(mid_sep.outputs["Y"], mid_flat.inputs["Y"])
        mid_r = nodes.new("ShaderNodeVectorMath")
        mid_r.operation = 'LENGTH'
        mid_r.location = (x + 2160, y - 3100)
        links.new(mid_flat.outputs["Vector"], mid_r.inputs[0])
        around = nodes.new("ShaderNodeCombineXYZ")
        around.location = (x + 2340, y - 3000)
        links.new(_math('MULTIPLY', _math('SINE', mid_angle, None, x + 2160, y - 2950), -1.0,
                        x + 2340, y - 2950), around.inputs["X"])
        links.new(_math('COSINE', mid_angle, None, x + 2160, y - 3050), around.inputs["Y"])
        offset = nodes.new("ShaderNodeVectorMath")
        offset.operation = 'SUBTRACT'
        offset.location = (x + 2340, y - 3200)
        links.new(position.outputs["Position"], offset.inputs[0])
        links.new(row_mid.outputs["Vector"], offset.inputs[1])
        along_round = nodes.new("ShaderNodeVectorMath")
        along_round.operation = 'DOT_PRODUCT'
        along_round.location = (x + 2520, y - 3100)
        links.new(offset.outputs["Vector"], along_round.inputs[0])
        links.new(around.outputs["Vector"], along_round.inputs[1])
        # A row wider than its share of the circle (pi / count each side) is
        # wrapped COMPRESSED evenly to fit, not clamped -- a clamp piled its
        # edge vertices onto one angle, a fold (a crease, and overlapping UVs).
        share = _math('DIVIDE', math.pi, count_socket, x + 2520, y - 3300)
        row_end0 = sample(ribbon, row_start, y - 3500)
        row_end1 = sample(ribbon, _math('ADD', row_start, _math('SUBTRACT', P, 1.0,
                                                                x + 900, y - 3600),
                                        x + 1080, y - 3600), y - 3600)
        row_chord = nodes.new("ShaderNodeVectorMath")
        row_chord.operation = 'SUBTRACT'
        row_chord.location = (x + 1440, y - 3550)
        links.new(row_end1, row_chord.inputs[0])
        links.new(row_end0, row_chord.inputs[1])
        row_round = nodes.new("ShaderNodeVectorMath")
        row_round.operation = 'DOT_PRODUCT'
        row_round.location = (x + 1620, y - 3550)
        links.new(row_chord.outputs["Vector"], row_round.inputs[0])
        links.new(around.outputs["Vector"], row_round.inputs[1])
        safe_r = _math('MAXIMUM', mid_r.outputs["Value"], 1e-6, x + 2520, y - 3400)
        half_turn = _math('DIVIDE', _math('MULTIPLY', _math('ABSOLUTE', row_round.outputs[
            "Value"], None, x + 1800, y - 3550), 0.5, x + 1980, y - 3550), safe_r,
            x + 2160, y - 3550)
        fit = _math('MINIMUM', 1.0, _math('DIVIDE', share, _math('MAXIMUM', half_turn, 1e-9,
                                                                 x + 2340, y - 3550),
                                          x + 2520, y - 3550), x + 2700, y - 3550)
        turn = _math('MULTIPLY', _math('DIVIDE', along_round.outputs["Value"], safe_r,
                                       x + 2700, y - 3200), fit, x + 3060, y - 3200)
        removed = nodes.new("ShaderNodeVectorMath")
        removed.operation = 'SCALE'
        removed.location = (x + 2700, y - 3000)
        links.new(around.outputs["Vector"], removed.inputs[0])
        links.new(along_round.outputs["Value"], removed.inputs["Scale"])
        unrolled = nodes.new("ShaderNodeVectorMath")
        unrolled.operation = 'SUBTRACT'
        unrolled.location = (x + 2880, y - 3000)
        links.new(position.outputs["Position"], unrolled.inputs[0])
        links.new(removed.outputs["Vector"], unrolled.inputs[1])
        rolled = nodes.new("ShaderNodeVectorRotate")
        rolled.rotation_type = 'Z_AXIS'
        rolled.location = (x + 3240, y - 3000)
        links.new(unrolled.outputs["Vector"], rolled.inputs["Vector"])
        links.new(turn, rolled.inputs["Angle"])
        above_base = _math('GREATER_THAN', ring, 0.5, x + 3060, y - 3400)
        wrap_sel = nodes.new("FunctionNodeBooleanMath")
        wrap_sel.operation = 'AND'
        wrap_sel.location = (x + 3240, y - 3400)
        links.new(above_base, wrap_sel.inputs[0])
        links.new(first_layer, wrap_sel.inputs[1])
        wrapped = nodes.new("GeometryNodeSetPosition")
        wrapped.location = (x + 3420, y + 200)
        links.new(ribbon, wrapped.inputs["Geometry"])
        links.new(wrap_sel.outputs["Boolean"], wrapped.inputs["Selection"])
        links.new(rolled.outputs["Vector"], wrapped.inputs["Position"])

        _mark("Snap the base rings; flip lobes running against the rim")
        snap = nodes.new("GeometryNodeSetPosition")
        snap.location = (x + 3960, y)
        links.new(wrapped.outputs["Geometry"], snap.inputs["Geometry"])
        links.new(base_mask.outputs["Boolean"], snap.inputs["Selection"])
        links.new(target.outputs["Vector"], snap.inputs["Position"])
        # One consistent winding across the seam: a lobe continuing the tube
        # wall agrees with it exactly when its base runs round the rim the way
        # the tube's rim vertices do -- so a lobe running the other way is
        # flipped (per lobe: a face's lobe from its index; quads only, the
        # ribbon's profile being open).
        face_lobe = _math('FLOOR', _math('DIVIDE', index.outputs["Index"], _math(
            'MULTIPLY', _math('SUBTRACT', resolution_socket, 1.0, x + 3600, y - 2300),
            _math('SUBTRACT', P, 1.0, x + 3600, y - 2400), x + 3780, y - 2300),
            x + 3960, y - 2300), None, x + 4140, y - 2300)
        _az_f, face_dir = lobe_frame(face_lobe, y - 2400)
        against = _math('GREATER_THAN', _math('ABSOLUTE', _math(
            'SUBTRACT', face_dir, rim_dir, x + 3780, y - 2500), None, x + 3960, y - 2500),
            1.0, x + 4140, y - 2500)
        in_layer = _math('LESS_THAN', face_lobe, _math('SUBTRACT', count_socket, 0.5,
                                                       x + 3960, y - 2600), x + 4140, y - 2600)
        flip_sel = nodes.new("FunctionNodeBooleanMath")
        flip_sel.operation = 'AND'
        flip_sel.location = (x + 4320, y - 2500)
        links.new(against, flip_sel.inputs[0])
        links.new(in_layer, flip_sel.inputs[1])
        flipped = nodes.new("GeometryNodeFlipFaces")
        flipped.location = (x + 4140, y + 200)
        links.new(snap.outputs["Geometry"], flipped.inputs["Mesh"])
        links.new(flip_sel.outputs["Boolean"], flipped.inputs["Selection"])

        # Tag the snapped base rings and the tube's rim; merge only those.
        tag_lobes = nodes.new("GeometryNodeStoreNamedAttribute")
        tag_lobes.data_type = 'BOOLEAN'
        tag_lobes.domain = 'POINT'
        tag_lobes.inputs["Name"].default_value = "pg_tube_weld"
        tag_lobes.location = (x + 4140, y)
        _mark("UVs: tube + lobes as one rectangle, looked up per corner")
        # Finish the fused whorl's UVs, BEFORE the merge (face corners survive
        # it, vertex attributes would be averaged): the tube and its lobes are
        # ONE piece, laid out in Python (`_fused_piece`: cut once, between two
        # lobes -- where lobe 0's slots start -- a rectangle with the lobes
        # standing on it). Each corner looks its vertex up in the piece's grid on
        # the tube's carrier: a lobe's by (its sector from the cut, row,
        # column in slot order), the tube's by (ring, slot from the cut -- a
        # corner on the cut taking its face's side).
        spans = _math('SUBTRACT', P, 1.0, x + 3600, y + 1400)
        centre0 = _math('DIVIDE', _wrap(_math('SUBTRACT', az0, phi0, x + 3600, y + 1300),
                                        x + 3780, y + 1300), step, x + 3960, y + 1300)
        first0 = _math('ROUND', _math('SUBTRACT', centre0, _math(
            'MULTIPLY', _d0, _math('MULTIPLY', spans, 0.5, x + 3960, y + 1200),
            x + 4140, y + 1200), x + 4140, y + 1300), None, x + 4320, y + 1300)
        min0 = _math('SUBTRACT', first0, _math('MULTIPLY', _math(
            'LESS_THAN', _d0, 0.0, x + 4320, y + 1200), spans, x + 4500, y + 1200),
            x + 4680, y + 1300)
        cut_angle = _math('ADD', phi0, _math('MULTIPLY', min0, step, x + 4680, y + 1200),
                          x + 4860, y + 1300)

        def from_cut(angle, xx, yy):
            """An angle round the flower measured from the cut (0 at one side
            of it, 2 pi at the other)."""
            return _math('FLOORED_MODULO', _math('SUBTRACT', angle, cut_angle, xx, yy),
                         2.0 * math.pi, xx + 180, yy)

        grid = nodes.new("GeometryNodeObjectInfo")
        grid.transform_space = 'ORIGINAL'
        grid.location = (x + 5400, y + 1500)
        links.new(uv_grid_socket, grid.inputs["Object"])

        def grid_uv(index_field, xx, yy):
            node = nodes.new("GeometryNodeSampleIndex")
            node.data_type = 'FLOAT_VECTOR'
            node.domain = 'POINT'
            node.location = (xx, yy)
            links.new(grid.outputs["Geometry"], node.inputs["Geometry"])
            links.new(position_in_grid.outputs["Position"], node.inputs["Value"])
            links.new(index_field, node.inputs["Index"])
            sep = nodes.new("ShaderNodeSeparateXYZ")
            sep.location = (xx + 180, yy)
            links.new(node.outputs["Value"], sep.inputs["Vector"])
            return _uv_place(uv_centre_socket, sep.outputs["X"], sep.outputs["Y"], 0.0,
                             xx + 360, yy)

        position_in_grid = nodes.new("GeometryNodeInputPosition")
        position_in_grid.location = (x + 5400, y + 1300)
        # lobes: sector from the slot centre, column in slot order
        slot_centre = _math('ADD', phi0, _math('MULTIPLY', _math(
            'ADD', first_slot, _math('MULTIPLY', direction, _math('MULTIPLY', spans, 0.5,
                                                                  x + 3600, y + 900),
                                     x + 3780, y + 900), x + 3960, y + 900), step,
            x + 4140, y + 900), x + 4320, y + 900)
        sector = _math('MINIMUM', _math('FLOOR', _math('DIVIDE', from_cut(
            slot_centre, x + 4500, y + 1000), _math('DIVIDE', 2.0 * math.pi, count_socket,
                                                   x + 4680, y + 1100), x + 4860, y + 1000),
            None, x + 5040, y + 1000), _math('SUBTRACT', count_socket, 1.0, x + 5040, y + 1100),
            x + 5220, y + 1000)
        column = _math('ADD', k, _math('MULTIPLY', _math('LESS_THAN', direction, 0.0,
                                                         x + 4500, y + 800),
                                       _math('SUBTRACT', spans, _math('MULTIPLY', k, 2.0,
                                                                      x + 4500, y + 700),
                                             x + 4680, y + 700), x + 4860, y + 800),
                       x + 5040, y + 800)
        per_lobe_grid = _math('MULTIPLY', resolution_socket, P, x + 5040, y + 600)
        lobe_index = _on_domain(_math('ADD', _math('MULTIPLY', sector, per_lobe_grid,
                                                   x + 5220, y + 600),
                                      _math('ADD', _math('MULTIPLY', ring, P, x + 5220, y + 700),
                                            column, x + 5400, y + 700), x + 5580, y + 600),
                                'POINT', x + 5760, y + 600)
        in_layer = _math('GREATER_THAN', _on_domain(first_layer, 'POINT', x + 4500, y + 500),
                         0.5, x + 4680, y + 500)
        lobe_u, lobe_v = grid_uv(lobe_index, x + 5940, y + 900)
        lobes_uv = _store_uv(flipped.outputs["Mesh"], lobe_u, lobe_v, x + 6660, y + 900,
                             selection=in_layer)
        # tube: ring (kept on its corners at creation), slot from the cut
        tpos = nodes.new("ShaderNodeSeparateXYZ")
        tpos.location = (x + 3600, y + 400)
        links.new(position.outputs["Position"], tpos.inputs["Vector"])
        face_angle = _math('ARCTAN2', _on_domain(tpos.outputs["Y"], 'FACE', x + 3780, y + 400),
                           _on_domain(tpos.outputs["X"], 'FACE', x + 3780, y + 340),
                           x + 3960, y + 400)
        vert_angle = _math('ARCTAN2', tpos.outputs["Y"], tpos.outputs["X"], x + 3780, y + 250)
        phi = _math('ADD', from_cut(face_angle, x + 4140, y + 400), _wrap(_math(
            'SUBTRACT', vert_angle, face_angle, x + 4140, y + 250), x + 4320, y + 250),
            x + 4680, y + 300)
        slot = _math('ROUND', _math('DIVIDE', phi, step, x + 4860, y + 300), None,
                     x + 5040, y + 300)
        tube_index = _math('ADD', _math('MULTIPLY', count_socket, per_lobe_grid,
                                        x + 5220, y + 400),
                           _math('ADD', _math('MULTIPLY', _named("pg_uv_ring", 'FLOAT',
                                                                 x + 5040, y + 150),
                                              _math('ADD', sides_socket, 1.0, x + 5040, y + 200),
                                              x + 5220, y + 200), slot, x + 5400, y + 250),
                           x + 5580, y + 300)
        tube_u, tube_v = grid_uv(tube_index, x + 5940, y + 400)
        tube_uv = _store_uv(tube_turned.outputs["Geometry"], tube_u, tube_v, x + 6480, y + 400,
                            selection=_named_bool("pg_tube_face", x + 6300, y + 300))
        links.new(lobes_uv, tag_lobes.inputs["Geometry"])
        links.new(base_mask.outputs["Boolean"], tag_lobes.inputs["Value"])
        _mark("Tag both sides of the seam and merge them")
        on_rim = _named_bool("pg_rim", x + 4140, y - 1700)
        tag_tube = nodes.new("GeometryNodeStoreNamedAttribute")
        tag_tube.data_type = 'BOOLEAN'
        tag_tube.domain = 'POINT'
        tag_tube.inputs["Name"].default_value = "pg_tube_weld"
        tag_tube.location = (x + 4140, y - 1500)
        links.new(tube_uv, tag_tube.inputs["Geometry"])
        links.new(on_rim, tag_tube.inputs["Value"])
        both = nodes.new("GeometryNodeJoinGeometry")
        both.location = (x + 4320, y)
        links.new(tag_lobes.outputs["Geometry"], both.inputs["Geometry"])
        links.new(tag_tube.outputs["Geometry"], both.inputs["Geometry"])
        tagged = nodes.new("GeometryNodeInputNamedAttribute")
        tagged.data_type = 'BOOLEAN'
        tagged.inputs["Name"].default_value = "pg_tube_weld"
        tagged.location = (x + 4320, y - 200)
        merge = nodes.new("GeometryNodeMergeByDistance")
        merge.location = (x + 4500, y)
        links.new(both.outputs["Geometry"], merge.inputs["Geometry"])
        links.new(tagged.outputs["Attribute"], merge.inputs["Selection"])
        merge.inputs["Distance"].default_value = _WELD_DISTANCE
        untag = nodes.new("GeometryNodeRemoveAttribute")
        untag.location = (x + 4680, y)
        links.new(merge.outputs["Geometry"], untag.inputs["Geometry"])
        untag.inputs["Name"].default_value = "pg_tube_weld"

        _mark("Tube Weld on / off")
        lobes_out = nodes.new("GeometryNodeSwitch")
        lobes_out.input_type = 'GEOMETRY'
        lobes_out.location = (x + 4860, y)
        links.new(weld_socket, lobes_out.inputs["Switch"])
        links.new(ribbon, lobes_out.inputs["False"])
        links.new(untag.outputs["Geometry"], lobes_out.inputs["True"])
        tube_out = nodes.new("GeometryNodeSwitch")
        tube_out.input_type = 'GEOMETRY'
        tube_out.location = (x + 4860, y - 200)
        links.new(weld_socket, tube_out.inputs["Switch"])
        links.new(tube, tube_out.inputs["False"])          # True: nothing (welded in)
        return lobes_out.outputs["Output"], tube_out.outputs["Output"]

    # ---- Stages ------------------------------------------------------------
    # The tree is assembled from named sub-groups (`PlantGen Flower · <stage>`):
    # one per part (each built, then given its UVs), one per weld, and the
    # finish. Each is built by the SAME helpers the flat tree used to call --
    # while a stage builds, `nodes` / `links` / `gin` point at its sub-group,
    # whose inputs carry the main group's own socket names -- so it computes
    # node for node what the flat tree did (every flower fingerprinted 0-diff),
    # only encapsulated. Inside a stage each helper's nodes sit in a frame
    # saying what that step does; the layout is `plant_gen`'s
    # `gn_layout.tidy`, borrowed cosmetically (absent: left unarranged).
    inputs = [(item.name, item.socket_type) for item in ng.interface.items_tree
              if item.item_type == 'SOCKET' and item.in_out == 'INPUT']
    stage_frames: list = []
    claimed: set = set()
    frame_stack: list = []     # [label, nodes before] per framed call in progress

    def _close(label, before) -> None:
        new = [n for n in nodes
               if n.as_pointer() not in before and n.as_pointer() not in claimed]
        if new:
            stage_frames.append((label, new))
            claimed.update(n.as_pointer() for n in new)

    def _framed(label, fn):
        """`fn`, its new nodes -- those no framed helper inside it took --
        recorded as a frame of the stage being built (as several, where it
        calls `_mark`). `label` may be a function of its (args, kwargs)."""
        def run(*args, **kwargs):
            text = label(args, kwargs) if callable(label) else label
            frame_stack.append([text, {n.as_pointer() for n in nodes}])
            try:
                return fn(*args, **kwargs)
            finally:
                text, before = frame_stack.pop()
                _close(text, before)
        return run

    def _mark(label) -> None:
        """Inside a framed helper: what it built since its last mark is one
        frame, what it builds from here the next, labelled `label`. (The
        helpers above call it at build time, by when it is defined.)"""
        if frame_stack:
            top = frame_stack[-1]
            _close(top[0], top[1])
            top[0], top[1] = label, {n.as_pointer() for n in nodes}

    _ring_chord = _framed(
        lambda a, kw: "Ring width, edge to edge" + (
            f" (for {frame_stack[-1][0].split(':')[0]})" if frame_stack else ""), _ring_chord)
    _push_boundary_ring = _framed(
        lambda a, kw: "Top Shape: the tip ring pushed out by its cap curve"
        if kw.get("is_tip") else "Bottom Shape: the base ring pushed out by its cap curve",
        _push_boundary_ring)
    _apply_cup = _framed("Cup: bow every ring across its width", _apply_cup)
    _ribbon_mesh = _framed("Ribbon: the spine swept into a strip", _ribbon_mesh)
    _revolved_mesh = _framed("Revolve: the profile swept round the axis", _revolved_mesh)
    _tag_rings = _framed("Tag the base ring and the rim (the welds find them)", _tag_rings)
    _base_weld = _framed("Weld: receptacle rim onto the tube's base ring", _base_weld)
    _uv_record = _framed("UVs: each vertex placed from the layout record", _uv_record)
    _uv_ribbon = _framed("UVs: strips, true to the open flower's shape", _uv_ribbon)
    _uv_revolved = _framed("UVs: the revolved body", _uv_revolved)
    _tube_weld = _framed("Weld: lobes wrapped round, snapped to the rim, merged",
                         _tube_weld)

    def _layout(tree, frames) -> None:
        try:
            gn_layout.tidy(tree, frames)
        except Exception as exc:  # noqa: BLE001 -- a layout is cosmetic
            print(f"PlantGen: flower node layout skipped ({exc})")

    def _stage(name, note, geo_in, geo_out, build, feeds=None):
        """Build `build` -- given its geometry inputs, returning its geometry
        outputs -- as the sub-group `PlantGen Flower · <name>`, exposing only
        the main inputs it reads; `note` frames whatever no helper framed.
        Returns its group node in the main tree, the geometry inputs fed from
        `feeds`, every other input from the main Group Input by name."""
        nonlocal nodes, links, gin
        full = _SUB_PREFIX + name
        old = bpy.data.node_groups.get(full)
        if old is not None:
            bpy.data.node_groups.remove(old, do_unlink=True)
        sub = bpy.data.node_groups.new(full, 'GeometryNodeTree')
        for sock_name in geo_in:
            sub.interface.new_socket(sock_name, in_out='INPUT',
                                     socket_type='NodeSocketGeometry')
        for sock_name, kind in inputs:
            sub.interface.new_socket(sock_name, in_out='INPUT', socket_type=kind)
        for sock_name in geo_out:
            sub.interface.new_socket(sock_name, in_out='OUTPUT',
                                     socket_type='NodeSocketGeometry')
        sub_in = sub.nodes.new("NodeGroupInput")
        sub_out = sub.nodes.new("NodeGroupOutput")
        saved = (nodes, links, gin)
        nodes, links, gin = sub.nodes, sub.links, sub_in
        stage_frames.clear()
        claimed.clear()
        try:
            outs = build(*(sub_in.outputs[n] for n in geo_in))
            if not isinstance(outs, tuple):
                outs = (outs,)
            for sock, sock_name in zip(outs, geo_out, strict=True):
                links.new(sock, sub_out.inputs[sock_name])
            rest = [n for n in sub.nodes if n.as_pointer() not in claimed
                    and n.bl_idname not in ("NodeGroupInput", "NodeGroupOutput")]
            frames = ([(note, rest)] if rest else []) + list(stage_frames)
        finally:
            nodes, links, gin = saved
        used = {s.identifier for s in sub_in.outputs if s.is_linked}
        for item in [i for i in sub.interface.items_tree
                     if i.item_type == 'SOCKET' and i.in_out == 'INPUT'
                     and i.identifier not in used]:
            sub.interface.remove(item)
        _layout(sub, frames)
        node = nodes.new("GeometryNodeGroup")
        node.node_tree = sub
        for sock in node.inputs:
            source = (feeds or {}).get(sock.name)
            links.new(source if source is not None else gin.outputs[sock.name], sock)
        return node

    # -- the parts: each built from its carrier, then its flattening record
    # (`_uv_record`) read off its freshly built vertex order -- before any
    # weld merges or moves it.
    def _petals():
        return _uv_ribbon(_ribbon_mesh(
            gin.outputs["Geometry"],
            gin.outputs[_PETAL_CUP_SOCKET], gin.outputs[_PETAL_WIDTH_REF_SOCKET], 150,
            resolution_socket=gin.outputs[_PETAL_RESOLUTION_SOCKET],
            top_shape_points_socket=gin.outputs[_PETAL_TOP_SHAPE_POINTS_SOCKET],
            bottom_shape_points_socket=gin.outputs[_PETAL_BOTTOM_SHAPE_POINTS_SOCKET],
            p_socket=gin.outputs[_PETAL_WIDTH_RES_SOCKET]),
            "petals", gin.outputs[_PETAL_RESOLUTION_SOCKET],
            gin.outputs[_PETAL_WIDTH_RES_SOCKET], 300)

    def _sepals():
        sepal_obj_info = nodes.new("GeometryNodeObjectInfo")
        sepal_obj_info.location = (col_in, -250)
        links.new(gin.outputs[_SEPAL_POINTS_SOCKET], sepal_obj_info.inputs["Object"])
        return _uv_ribbon(_ribbon_mesh(
            sepal_obj_info.outputs["Geometry"],
            gin.outputs[_SEPAL_CUP_SOCKET], gin.outputs[_SEPAL_WIDTH_REF_SOCKET], -500,
            resolution_socket=gin.outputs[_SEPAL_RESOLUTION_SOCKET],
            top_shape_points_socket=gin.outputs[_SEPAL_TOP_SHAPE_POINTS_SOCKET],
            bottom_shape_points_socket=gin.outputs[_SEPAL_BOTTOM_SHAPE_POINTS_SOCKET],
            p_socket=gin.outputs[_SEPAL_WIDTH_RES_SOCKET]),
            "sepals", gin.outputs[_SEPAL_RESOLUTION_SOCKET],
            gin.outputs[_SEPAL_WIDTH_RES_SOCKET], -350)

    def _disc():
        return _uv_revolved(_revolved_mesh(
            _DISC_POINTS_SOCKET, -900, concave_socket=gin.outputs[_DISC_CONCAVE_SOCKET],
            sides_socket=gin.outputs[_DISC_SIDES_SOCKET]), "disc",
            gin.outputs[_DISC_SIDES_SOCKET], -900)

    def _receptacle():
        return _uv_revolved(_revolved_mesh(
            _RECEPTACLE_POINTS_SOCKET, -1700, always_flip=True,
            sides_socket=gin.outputs[_RECEPTACLE_SIDES_SOCKET]), "receptacle",
            gin.outputs[_RECEPTACLE_SIDES_SOCKET], -1700,
            flip_after=gin.outputs[_RECEPTACLE_WELD_SOCKET])    # `_base_weld` unflips it

    def _fusion_tube(points_socket, sides_socket, part, y):
        def build():
            return _tag_rings(_uv_revolved(
                _revolved_mesh(points_socket, y, sides_socket=gin.outputs[sides_socket]),
                part, gin.outputs[sides_socket], y),
                gin.outputs[sides_socket], col_mesh + 200, y)
        return build

    petals = _stage("Petals", "Petals: the spine object's curves",
                    (), ("Petals",), _petals)
    sepals = _stage("Sepals", "Sepals: their own spine object", (), ("Sepals",), _sepals)
    disc = _stage("Disc", "Disc: its own profile object", (), ("Disc",), _disc)
    receptacle = _stage("Receptacle", "Receptacle: its own profile object",
                        (), ("Receptacle",), _receptacle)
    corolla = _stage("Corolla Tube", "Corolla tube: the petals' fusion profile", (),
                     ("Tube",), _fusion_tube(_PETAL_FUSION_POINTS_SOCKET,
                                             _PETAL_TUBE_SIDES_SOCKET, "petal_tube", -2500))
    calyx = _stage("Calyx Tube", "Calyx tube: the sepals' fusion profile", (),
                   ("Tube",), _fusion_tube(_SEPAL_FUSION_POINTS_SOCKET,
                                           _SEPAL_TUBE_SIDES_SOCKET, "sepal_tube", -3300))

    # -- the welds. The receptacle onto the corolla tube's base first, so the
    # lobe weld after it finds the tube's rim (by its tag) whether or not it is
    # welded. A lobe weld is `_tube_weld` alone (Fuse To Tube): a plain Merge
    # by Distance across a whorl's lobes and its tube was built and abandoned
    # -- the tube's rim and a lobe's base ring sit on one circle at DIFFERENT
    # angles, so no threshold makes them coincide without skewing faces;
    # `_tube_weld` makes them the SAME vertices (matching side count, lobe
    # bases snapped onto the rim, then merged).
    def _receptacle_weld(tube, rec):
        return _base_weld(tube, rec, gin.outputs[_RECEPTACLE_WELD_SOCKET],
                          gin.outputs[_PETAL_TUBE_SIDES_SOCKET], 10000)

    def _lobe_weld(weld, count, resolution, sides, width_res, part, y):
        def build(lobes, tube):
            return _tube_weld(
                lobes, tube, gin.outputs[weld], gin.outputs[count], gin.outputs[resolution],
                gin.outputs[sides], y, gin.outputs[width_res],
                gin.outputs[_UV_CENTER_SOCKETS[part]], gin.outputs[_UV_CARRIER_SOCKETS[part]])
        return build

    base_weld = _stage(
        "Receptacle Weld", "Receptacle Tube Weld: one surface from below up the tube",
        ("Tube", "Receptacle"), ("Tube", "Receptacle"), _receptacle_weld,
        feeds={"Tube": corolla.outputs["Tube"], "Receptacle": receptacle.outputs["Receptacle"]})
    corolla_weld = _stage(
        "Corolla Weld", "Petal Tube Weld (Fuse To Tube): the petals into the tube",
        ("Lobes", "Tube"), ("Lobes", "Tube"),
        _lobe_weld(_PETAL_WELD_SOCKET, _PETAL_COUNT_SOCKET, _PETAL_RESOLUTION_SOCKET,
                   _PETAL_TUBE_SIDES_SOCKET, _PETAL_WIDTH_RES_SOCKET, "petal_tube", 4000),
        feeds={"Lobes": petals.outputs["Petals"], "Tube": base_weld.outputs["Tube"]})
    calyx_weld = _stage(
        "Calyx Weld", "Sepal Tube Weld (Fuse To Tube): the sepals into the tube",
        ("Lobes", "Tube"), ("Lobes", "Tube"),
        _lobe_weld(_SEPAL_WELD_SOCKET, _SEPAL_COUNT_SOCKET, _SEPAL_RESOLUTION_SOCKET,
                   _SEPAL_TUBE_SIDES_SOCKET, _SEPAL_WIDTH_RES_SOCKET, "sepal_tube", 7000),
        feeds={"Lobes": sepals.outputs["Sepals"], "Tube": calyx.outputs["Tube"]})

    # -- the finish: Flip Normals per component -- AFTER the welds, so a part
    # welded into another goes with it (a fused tube is inside the petals' /
    # sepals' geometry by now, a welded receptacle inside the corolla tube's)
    # and the surface keeps one winding; face corners (the UVs) go with their
    # faces. Then one mesh, the welds' and the unwrap's helper attributes
    # removed, the material (or the UV checker).
    parts = ("Petals", "Sepals", "Disc", "Receptacle", "Corolla Tube", "Calyx Tube")

    def _finish(*meshes):
        join = nodes.new("GeometryNodeJoinGeometry")
        join.location = (col_join, 0)
        for k, (mesh_output, flip_socket) in enumerate(zip(meshes, (
                _PETAL_FLIP_SOCKET, _SEPAL_FLIP_SOCKET, _DISC_FLIP_SOCKET,
                _RECEPTACLE_FLIP_SOCKET, _PETAL_TUBE_FLIP_SOCKET, _SEPAL_TUBE_FLIP_SOCKET),
                strict=True)):
            flip = nodes.new("GeometryNodeFlipFaces")
            flip.location = (col_join - 200, -150 * k)
            links.new(mesh_output, flip.inputs["Mesh"])
            links.new(gin.outputs[flip_socket], flip.inputs["Selection"])
            links.new(flip.outputs["Mesh"], join.inputs["Geometry"])
        result = join.outputs["Geometry"]
        for k, tag in enumerate(("pg_rim", "pg_base", "pg_tube_face",
                                 "pg_uv_ring")):   # the welds' and the unwrap's helpers
            untag = nodes.new("GeometryNodeRemoveAttribute")
            untag.location = (col_join + 150 * (k + 1), 0)
            links.new(result, untag.inputs["Geometry"])
            untag.inputs["Name"].default_value = tag
            result = untag.outputs["Geometry"]
        dressed = nodes.new("GeometryNodeSetMaterial")
        dressed.location = (col_join + 450, 0)
        links.new(result, dressed.inputs["Geometry"])
        links.new(gin.outputs[_FLOWER_MAT_SOCKET], dressed.inputs["Material"])
        checker = nodes.new("GeometryNodeSetMaterial")
        checker.location = (col_join + 600, 0)
        links.new(dressed.outputs["Geometry"], checker.inputs["Geometry"])
        links.new(gin.outputs[_CHECKER_SOCKET], checker.inputs["Selection"])
        links.new(gin.outputs[_CHECKER_MAT_SOCKET], checker.inputs["Material"])
        return checker.outputs["Geometry"]

    finish = _stage(
        "Finish", "Flip Normals per part, join, clean up, material", parts, ("Geometry",),
        _finish, feeds=dict(zip(parts, (
            corolla_weld.outputs["Lobes"], calyx_weld.outputs["Lobes"], disc.outputs["Disc"],
            base_weld.outputs["Receptacle"], corolla_weld.outputs["Tube"],
            calyx_weld.outputs["Tube"]), strict=True)))
    links.new(finish.outputs["Geometry"], gout.inputs["Geometry"])

    stages = (petals, sepals, disc, receptacle, corolla, calyx)
    _layout(ng, [
        ("Parts: each built from its carrier object, then given its UVs", list(stages)),
        ("Welds: receptacle onto the corolla tube, lobes into their tubes",
         [base_weld, corolla_weld, calyx_weld]),
        ("Finish: flip normals, join, material", [finish]),
    ])

    ng[_PG_VERSION_KEY] = _FLOWER_GROUP_VERSION     # stamped LAST
    return ng


def ensure_flower_gn_modifier(
    obj, disc_points_obj, *, petal_cup: float,
    petal_width_ref: float, disc_concave: bool,
    sepal_points_obj, sepal_cup: float, sepal_width_ref: float,
    receptacle_points_obj,
    petal_fusion_points_obj, sepal_fusion_points_obj,
    petal_top_shape_points_obj, petal_bottom_shape_points_obj,
    sepal_top_shape_points_obj, sepal_bottom_shape_points_obj,
    petal_resolution: int, sepal_resolution: int,
    petal_weld: tuple = (False, 0), sepal_weld: tuple = (False, 0),
    receptacle_weld: bool = False,
    widths: tuple = (_RIBBON_PROFILE_RESOLUTION, _RIBBON_PROFILE_RESOLUTION),
    sides: tuple = (_TUBE_SIDES, _TUBE_SIDES, _TUBE_SIDES, _TUBE_SIDES),
    material=None,
    checker: bool = False,
    uv: dict | None = None,
    flips: tuple = (False,) * 6,
) -> None:
    """`petal_weld` / `sepal_weld`: (weld the whorl's lobes into its tube, its
    first layer's lobe count) -- see `_tube_weld`. `receptacle_weld`: weld the
    receptacle onto the corolla tube's base -- see `_base_weld`. `flips`:
    Flip Normals for (petals, sepals, disc, receptacle, corolla tube, calyx
    tube)."""
    ng = bpy.data.node_groups.get(_FLOWER_GROUP_NAME)
    replaced = False
    if ng is not None and ng.get(_PG_VERSION_KEY) != _FLOWER_GROUP_VERSION:
        bpy.data.node_groups.remove(ng, do_unlink=True)
        ng = None
        replaced = True
    if ng is None:
        ng = _build_flower_group()

    mod = obj.modifiers.get(_FLOWER_GROUP_NAME)
    if mod is None:
        mod = obj.modifiers.new(_FLOWER_GROUP_NAME, 'NODES')
    if mod.node_group is not ng:
        mod.node_group = ng

    values = {
        _DISC_POINTS_SOCKET: disc_points_obj,
        _PETAL_CUP_SOCKET: petal_cup,
        _PETAL_WIDTH_REF_SOCKET: petal_width_ref,
        _DISC_CONCAVE_SOCKET: disc_concave,
        _SEPAL_POINTS_SOCKET: sepal_points_obj,
        _SEPAL_CUP_SOCKET: sepal_cup,
        _SEPAL_WIDTH_REF_SOCKET: sepal_width_ref,
        _RECEPTACLE_POINTS_SOCKET: receptacle_points_obj,
        _PETAL_FUSION_POINTS_SOCKET: petal_fusion_points_obj,
        _SEPAL_FUSION_POINTS_SOCKET: sepal_fusion_points_obj,
        _PETAL_TOP_SHAPE_POINTS_SOCKET: petal_top_shape_points_obj,
        _PETAL_BOTTOM_SHAPE_POINTS_SOCKET: petal_bottom_shape_points_obj,
        _SEPAL_TOP_SHAPE_POINTS_SOCKET: sepal_top_shape_points_obj,
        _SEPAL_BOTTOM_SHAPE_POINTS_SOCKET: sepal_bottom_shape_points_obj,
        _PETAL_RESOLUTION_SOCKET: petal_resolution,
        _SEPAL_RESOLUTION_SOCKET: sepal_resolution,
    }
    # `widths`: points across a (petal, sepal); `sides`: around the (corolla
    # tube, calyx tube, disc, receptacle) -- a welded tube takes the lobes'
    # count instead, a welded receptacle the corolla tube's.
    for (weld, count), weld_s, count_s, sides_s, p, own_sides in (
            (petal_weld, _PETAL_WELD_SOCKET, _PETAL_COUNT_SOCKET, _PETAL_TUBE_SIDES_SOCKET,
             widths[0], sides[0]),
            (sepal_weld, _SEPAL_WELD_SOCKET, _SEPAL_COUNT_SOCKET, _SEPAL_TUBE_SIDES_SOCKET,
             widths[1], sides[1])):
        values[weld_s] = bool(weld)
        values[count_s] = int(count)
        values[sides_s] = int(count) * (int(p) - 1) if weld else int(own_sides)
    values[_PETAL_WIDTH_RES_SOCKET] = int(widths[0])
    values[_SEPAL_WIDTH_RES_SOCKET] = int(widths[1])
    values[_DISC_SIDES_SOCKET] = int(sides[2])
    for flip_socket, flip in zip((_PETAL_FLIP_SOCKET, _SEPAL_FLIP_SOCKET, _DISC_FLIP_SOCKET,
                                  _RECEPTACLE_FLIP_SOCKET, _PETAL_TUBE_FLIP_SOCKET,
                                  _SEPAL_TUBE_FLIP_SOCKET), flips, strict=True):
        values[flip_socket] = bool(flip)
    values[_FLOWER_MAT_SOCKET] = material
    values[_CHECKER_SOCKET] = bool(checker)
    if checker:
        from .flower_material import checker_material
        values[_CHECKER_MAT_SOCKET] = checker_material()
    else:
        values[_CHECKER_MAT_SOCKET] = None
    uv = uv or {}
    values[_UV_SCALE_SOCKET] = float(uv.get("scale", 1.0))
    for part in _UV_PARTS:
        values[_UV_CARRIER_SOCKETS[part]] = uv.get("carriers", {}).get(part)
        values[_UV_CENTER_SOCKETS[part]] = tuple(uv.get("centres", {}).get(part, (0.0, 0.0))) + (
            0.0,)
    values[_RECEPTACLE_WELD_SOCKET] = bool(receptacle_weld)
    values[_RECEPTACLE_SIDES_SOCKET] = (values[_PETAL_TUBE_SIDES_SOCKET] if receptacle_weld
                                        else int(sides[3]))
    for socket_name, value in values.items():
        ident = _socket_identifier(ng, socket_name)
        _set_modifier_input(mod, ident, value)
    if replaced:
        # Removing the old group (do_unlink) emptied EVERY flower's modifier,
        # not just this one's -- a flower sitting in a stem's asset collection,
        # or a baked bloom stage, would draw nothing until someone edited it.
        # Rebuild them all onto the new group from their own settings.
        for other in list(bpy.data.objects):
            other_mod = other.modifiers.get(_FLOWER_GROUP_NAME) if is_flower(other) else None
            if other is not obj and other_mod is not None and other_mod.node_group is None:
                rebuild_flower(other)


# --- public operations ------------------------------------------------------------

def owner_of_flower(pg) -> bpy.types.Object | None:
    """Find the object whose `.plantgen_flower` is this PropertyGroup (for update
    callbacks) -- mirrors `plant.owner_of`."""
    ptr = pg.as_pointer()
    for obj in bpy.data.objects:
        if is_flower(obj) and obj.plantgen_flower.as_pointer() == ptr:
            return obj
    return None


_suppressed: set[int] = set()  # obj.as_pointer() values currently batching edits


@contextmanager
def suppress_rebuild(obj):
    """Batch several `obj.plantgen_flower` property writes into a single rebuild --
    mirrors `plant.suppress_rebuild` (same crash-avoidance reasoning: applying a
    whole preset back-to-back would otherwise fire one rebuild per field)."""
    ptr = obj.as_pointer()
    _suppressed.add(ptr)
    try:
        yield
    finally:
        _suppressed.discard(ptr)


def rebuild_flower(obj) -> bool:
    if not is_flower(obj) or obj.as_pointer() in _suppressed:
        return False
    params = flower_params_from_object(obj)
    skel = build_flower_mesh(params)
    _write_ribbon_curve(obj.data, skel.petals)

    sepal_obj = _ensure_sepal_object(obj)
    _write_ribbon_curve(sepal_obj.data, skel.sepals)

    disc_obj = _ensure_disc_object(obj)
    _write_disc_curve(disc_obj.data, skel.disc)
    # The disc's own last point sitting BELOW its first (rim) point means the
    # revolve pointed downward (concave/sunken) instead of up (see
    # skeleton/flower.py's _build_disc), so its faces need flipping. Compared
    # against the disc's own first point rather than literal 0.0 -- the whole
    # skeleton may have been shifted up by build_flower_mesh's attachment-origin
    # pass (see its docstring) when a receptacle is present, which moves the rim
    # off Z=0 but preserves the first/last relationship this check needs. An
    # empty (disabled) disc has no points at all -- guard against that directly
    # rather than indexing points[-1]/[0].
    disc_concave = bool(skel.disc.points) and skel.disc.points[-1][2] < skel.disc.points[0][2]

    receptacle_obj = _ensure_receptacle_object(obj)
    _write_disc_curve(receptacle_obj.data, skel.receptacle)

    petal_fusion_obj = _ensure_petal_fusion_object(obj)
    _write_disc_curve(petal_fusion_obj.data, skel.petal_fusion)

    sepal_fusion_obj = _ensure_sepal_fusion_object(obj)
    _write_disc_curve(sepal_fusion_obj.data, skel.sepal_fusion)

    petal_top_shape_obj = _ensure_petal_top_shape_object(obj)
    _write_shape_carrier_curve(petal_top_shape_obj.data, skel.petal_top_shape)
    petal_bottom_shape_obj = _ensure_petal_bottom_shape_object(obj)
    _write_shape_carrier_curve(petal_bottom_shape_obj.data, skel.petal_bottom_shape)
    sepal_top_shape_obj = _ensure_sepal_top_shape_object(obj)
    _write_shape_carrier_curve(sepal_top_shape_obj.data, skel.sepal_top_shape)
    sepal_bottom_shape_obj = _ensure_sepal_bottom_shape_object(obj)
    _write_shape_carrier_curve(sepal_bottom_shape_obj.data, skel.sepal_bottom_shape)

    s = obj.plantgen_flower
    petal_weld = _weld_spec(skel.petals, skel.petal_fusion, params.petal_tube_fuse,
                            params.petal_layers)
    sepal_weld = _weld_spec(skel.sepals, skel.sepal_fusion, params.sepal_tube_fuse,
                            params.sepal_layers)
    petal_welded, sepal_welded = petal_weld[0], sepal_weld[0]
    # Both matching the disc radius: their rims are one circle -- weld.
    receptacle_welded = bool(skel.receptacle.points and skel.petal_fusion.points
                             and params.receptacle_match_disc
                             and params.petal_tube_match_disc)
    ensure_flower_gn_modifier(
        obj, disc_obj, petal_cup=skel.petal_cup,
        petal_width_ref=s.petal_width,
        disc_concave=disc_concave,
        sepal_points_obj=sepal_obj,
        sepal_cup=skel.sepal_cup, sepal_width_ref=s.sepal_width,
        receptacle_points_obj=receptacle_obj,
        petal_fusion_points_obj=petal_fusion_obj, sepal_fusion_points_obj=sepal_fusion_obj,
        petal_top_shape_points_obj=petal_top_shape_obj,
        petal_bottom_shape_points_obj=petal_bottom_shape_obj,
        sepal_top_shape_points_obj=sepal_top_shape_obj,
        sepal_bottom_shape_points_obj=sepal_bottom_shape_obj,
        petal_resolution=s.resolution, sepal_resolution=s.sepal_resolution,
        petal_weld=petal_weld, sepal_weld=sepal_weld,
        receptacle_weld=receptacle_welded,
        widths=(max(2, params.petal_width_resolution), max(2, params.sepal_width_resolution)),
        sides=(params.petal_tube_sides, params.sepal_tube_sides, params.disc_sides,
               params.receptacle_sides),
        material=s.material, checker=s.uv_checker,
        flips=(params.petal_flip_normals, params.sepal_flip_normals, params.disc_flip_normals,
               params.receptacle_flip_normals, params.petal_tube_flip_normals,
               params.sepal_tube_flip_normals),
        uv=_write_uv_carriers(obj, _uv_layout(params, petal_welded, sepal_welded)))
    return True


_UV_GAP = 0.02             # of the square: between pieces (half of it round the outside)
_UV_FAN_FILL = 0.95        # an unrolled tube ring may span this much of a full turn
_UV_FAN_REACH = 20.0       # a fan's farthest apex (x the tube's size): a near-straight band


def _equal_area(points, radii, order, r0: float) -> list[float]:
    """A surface of revolution flattened into rings keeping AREA: walking the
    profile in `order` from radius `r0`, each ring's radius is
    sqrt(r0^2 + 2 * the area per radian swept so far) -- equal texel density
    everywhere (a disc / receptacle drawn as a circle)."""
    out = [0.0] * len(points)
    if not order:
        return out
    area = 0.5 * r0 * r0
    prev = order[0]
    out[prev] = r0
    for idx in order[1:]:
        ds = math.hypot(radii[idx] - radii[prev], points[idx][2] - points[prev][2])
        area += 0.5 * (radii[idx] + radii[prev]) * ds
        out[idx] = math.sqrt(2.0 * area)
        prev = idx
    return out


def _revolved_records(spine) -> tuple[list, float]:
    """A pole body's records (disc, receptacle): equal-area rings from its
    pole (the end with no radius) -- a circle."""
    n = len(spine.points)
    if not n:
        return [], 0.0
    order = list(range(n)) if spine.radii[0] <= spine.radii[-1] else list(range(n))[::-1]
    rings = _equal_area(spine.points, spine.radii, order, 0.0)
    return [(r, 0.0, 0.0, 1.0) for r in rings], max(rings)


def _profile_lengths(spine) -> list[float]:
    """Each ring's distance up a revolved body's profile from its first."""
    along = [0.0]
    for i in range(1, len(spine.points)):
        along.append(along[-1] + math.hypot(spine.radii[i] - spine.radii[i - 1],
                                            spine.points[i][2] - spine.points[i - 1][2]))
    return along


def _cone_apex(along, radii) -> float:
    """How far below a tube's base (along its profile) the apex of the cone
    that best fits it sits: radius ~ k * (s + apex), least squares weighted
    by each ring's share of the length. A cone unrolls flat exactly (a fan),
    so the nearer the tube is to its cone the less it stretches. A tube that
    does not widen (a cylinder, or narrowing) gets a far apex -- its fan is a
    near-straight band; never so near that a ring would pass a full turn."""
    n = len(along)
    reach = _UV_FAN_REACH * max(along[-1] + max(radii), 1e-6)
    apex = reach
    if n >= 2 and along[-1] > 1e-9:
        w = [0.5 * (along[min(i + 1, n - 1)] - along[max(i - 1, 0)]) for i in range(n)]
        sw = sum(w)
        ms = sum(wi * s for wi, s in zip(w, along, strict=True)) / sw
        mr = sum(wi * r for wi, r in zip(w, radii, strict=True)) / sw
        var = sum(wi * (s - ms) ** 2 for wi, s in zip(w, along, strict=True))
        cov = sum(wi * (s - ms) * (r - mr) for wi, s, r in zip(w, along, radii, strict=True))
        k = cov / var if var > 1e-18 else 0.0
        if k > 1e-6:
            apex = min(reach, (mr - k * ms) / k)
    # ring i (arc 2 pi r) at distance apex + s spans pi r / (apex + s) each side
    need = max(r / _UV_FAN_FILL - s for s, r in zip(along, radii, strict=True))
    return max(apex, need, 1e-6)


def _fan_piece(tube) -> tuple[list, tuple]:
    """An unfused tube as ONE piece: cut once and unrolled as a fan round the
    apex of its best-fit cone (`_cone_apex`): ring i an arc at distance
    apex + s_i (s its true length up the profile) and of its true length
    2 pi r_i -- true lengths both ways, only a tube straying from its cone
    shears. (A fused tube is a rectangle with its lobes, `_fused_piece`.)
    Records: one per ring (mode 3: x = distance from the apex, y = open
    radius), unrolled by the GN (`_uv_fan`). Returns those and the piece's box
    in the fan's own frame (u = -rho sin psi, v = rho cos psi: opening up)."""
    along = _profile_lengths(tube)
    radii = list(tube.radii)
    apex = _cone_apex(along, radii)
    recs = [(apex + s, r, 0.0, 3.0) for s, r in zip(along, radii, strict=True)]
    pts = []
    for rho, r, _z, _m in recs:
        half = math.pi * r / rho
        pts += [(rho, half * (t / 16.0 - 1.0)) for t in range(33)]
    us = [-rho * math.sin(p) for rho, p in pts]
    vs = [rho * math.cos(p) for rho, p in pts]
    return recs, (min(us), min(vs), max(us), max(vs))


_UV_THROAT_FROM = 0.3      # the throat: the tube from this fraction of its length up


def _throat_radius(along, radii) -> float:
    """The tube's mean radius over its throat (area-weighted, from
    `_UV_THROAT_FROM` of its length up) -- where a fused whorl's rectangle is
    true (foxglove spots); the narrower base below is squeezed round."""
    lo = _UV_THROAT_FROM * along[-1]
    area = moment = 0.0
    for i in range(len(radii) - 1):
        if along[i + 1] <= lo:
            continue
        ds = along[i + 1] - max(along[i], lo)
        r = 0.5 * (radii[i] + radii[i + 1])
        area += r * ds
        moment += r * r * ds
    return moment / area if area > 1e-18 else max(radii)


def _fused_piece(tube, lobes, count: int, width_points: int) -> tuple[list, tuple]:
    """A tube and its fused first-layer lobes as ONE straight piece (user
    decision 2026-10-07): the tube cut once -- between the two lobes where
    lobe 0's slots start, as `_tube_weld` cuts it -- and laid out as a
    RECTANGLE, round the tube across, up its profile at true length. Its
    width is the throat's true circumference (`_throat_radius`), so the
    throat and mouth are true and the narrower base is squeezed round (the
    trade the user took for straight rows). Each lobe goes straight up from
    its share of the top edge, rows at their true distance and width (no
    wider than that share: a flaring row is squeezed, never overlapping).

    Records, one per vertex of the welded mesh (mode 5: x, y = its place in
    the piece): the lobes first -- sector from the cut q, row, column in
    slot order -- then the tube -- ring, slot from the cut (the cut's slot
    twice, once per side). Returns those and the piece's box. Across runs
    against the slot order, which keeps every piece's UVs wound alike."""
    first = lobes[:count]
    rows = len(first[0].points)
    P = max(2, int(width_points))
    along = _profile_lengths(tube)
    width = 2.0 * math.pi * _throat_radius(along, list(tube.radii))
    share = width / count
    full = count * (P - 1)
    # which lobe sits in which sector (lobe 0 in sector 0: the cut is at its start)
    az = [math.atan2(lb.points[0][1], lb.points[0][0]) for lb in first]
    in_sector = list(range(count))
    for i, a in enumerate(az):
        in_sector[round(((a - az[0]) % (2.0 * math.pi)) / (2.0 * math.pi / count)) % count] = i
    records = []
    for q in range(count):
        lobe = first[in_sector[q]]
        middle = width - (q + 0.5) * share
        s = along[-1]
        for row in range(rows):
            if row:
                s += math.dist(lobe.points[row], lobe.points[row - 1])
            w = share if row == 0 else min(lobe.widths[row], share)
            for k in range(P):
                records.append((middle - (k / (P - 1) - 0.5) * w, s))
    for s in along:
        for j in range(full + 1):
            records.append((width * (1.0 - j / full), s))
    top = max(v for _u, v in records)
    return ([(u, v, 0.0, 5.0) for u, v in records], (0.0, 0.0, width, top))


def _strip_rows(lobe) -> list:
    """A free petal / sepal laid flat at its true shape: (distance up it,
    width) per row."""
    rows, s = [], 0.0
    for j, (pt, w) in enumerate(zip(lobe.points, lobe.widths, strict=True)):
        if j:
            s += math.dist(pt, lobe.points[j - 1])
        rows.append((s, w))
    return rows


def _uv_layout(params, petal_welded: bool, sepal_welded: bool) -> dict:
    """The whole flower's UVs: one texture, every part at ONE scale (equal
    texel density), packed tight into the square. Built from the OPEN flower
    (Bloom 1), so the unwrap is the same at every bloom stage and a texture
    does not slide as the flower opens.

    Pieces: a fused tube and its lobes ONE piece, cut once, a rectangle
    true at the throat with the lobes on top (`_fused_piece`); an unfused tube one fan
    (`_fan_piece`); every free petal / sepal an upright strip at its true
    shape (a whorl's side by side, in rows); the disc and receptacle
    equal-area circles. Packed into the smallest square (`_pack_groups`)."""
    skel = build_flower_mesh(dataclasses.replace(params, bloom=1.0))
    records = {p: [] for p in _UV_PARTS}
    centres = {}
    groups = []                      # blocks of (width, height, place(x, y))
    whorl_rows = {}                  # whorl -> one record list per lobe
    whorl_shapes = {}                # whorl -> its open Top then Bottom Shape (P each)

    def add_circle(part, spine):
        recs, r = _revolved_records(spine)
        if r <= 1e-9:
            return
        records[part] = recs

        def place(x, y):
            centres[part] = (x + r, y + r)
        groups.append([(2.0 * r, 2.0 * r, place)])

    def add_whorl(tube_part, tube, whorl, lobes, layers, welded, width_points, top, bottom):
        rows = whorl_rows.setdefault(whorl, [[] for _lobe in lobes])
        P = max(2, int(width_points))
        whorl_shapes[whorl] = [(v, 0.0, 0.0, 0.0) for v in
                               (list(top) + [0.0] * P)[:P] + (list(bottom) + [0.0] * P)[:P]]
        count = max(1, len(lobes) // max(1, layers)) if lobes else 1
        fused = welded and bool(tube.points) and bool(lobes)
        first = lobes[:count] if fused else []
        if tube.points:
            if fused:
                # one flattened piece; its lobes' own row records unused
                records[tube_part], (u0, v0, u1, v1) = _fused_piece(tube, first, count,
                                                                    width_points)
                for i, lobe in enumerate(first):
                    rows[i] = [(0.0, 0.0, 0.0, 0.0)] * len(lobe.points)
            else:
                records[tube_part], (u0, v0, u1, v1) = _fan_piece(tube)

            def place_fan(x, y):
                centres[tube_part] = (x - u0, y - v0)
            groups.append([(u1 - u0, v1 - v0, place_fan)])
        strips = []                  # the free ones: side by side, in rows
        for i in range(len(first), len(lobes)):
            strip = _strip_rows(lobes[i])
            w_max = max(w for _s, w in strip)
            # Top / Bottom Shape push the end rows out (in row widths)
            below = max([0.0] + list(bottom)) * strip[0][1]
            above = max([0.0] + list(top)) * strip[-1][1]

            def place_strip(x, y, i=i, strip=strip, w_max=w_max, below=below):
                rows[i] = [(y + below + s, w, x + 0.5 * w_max, 0.0) for s, w in strip]
            strips.append((w_max, below + strip[-1][0] + above, place_strip))
        if strips:
            groups.append(strips)

    add_circle("receptacle", skel.receptacle)
    add_circle("disc", skel.disc)
    add_whorl("petal_tube", skel.petal_fusion, "petals", skel.petals, params.petal_layers,
              petal_welded, params.petal_width_resolution, skel.petal_top_shape,
              skel.petal_bottom_shape)
    add_whorl("sepal_tube", skel.sepal_fusion, "sepals", skel.sepals, params.sepal_layers,
              sepal_welded, params.sepal_width_resolution, skel.sepal_top_shape,
              skel.sepal_bottom_shape)
    groups = [g for g in ([pc for pc in g if pc[0] > 1e-9 and pc[1] > 1e-9] for g in groups) if g]
    if not groups:
        return {"records": records, "centres": {}, "scale": 1.0}
    side = _pack_groups(groups)
    for whorl, rows in whorl_rows.items():
        records[whorl] = whorl_shapes[whorl] + [rec for lobe_rows in rows for rec in lobe_rows]
    scale = 1.0 / side
    return {"records": records,
            "centres": {p: (cx * scale, cy * scale) for p, (cx, cy) in centres.items()},
            "scale": scale}


def _block(pieces, rows: int, gap: float) -> tuple[float, float, list]:
    """Pieces side by side in `rows` rows (bottoms aligned): the block's width,
    height and each piece's corner in it."""
    per = math.ceil(len(pieces) / rows)
    offsets, width, y = [], 0.0, 0.0
    for start in range(0, len(pieces), per):
        line = pieces[start:start + per]
        x = 0.0
        for w, _h, _place in line:
            offsets.append((x, y))
            x += w + gap
        width = max(width, x - gap)
        y += max(h for _w, h, _place in line) + gap
    return width, y - gap, offsets


def _smallest_square(groups, rows) -> tuple[float, list]:
    """The smallest square side (bisected) the groups' blocks pack in, and
    each block's corner."""
    def fits(side):
        gap = _UV_GAP * side
        sizes = [_block(g, r, gap)[:2] for g, r in zip(groups, rows, strict=True)]
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
        return _skyline(sizes, order, side, gap)
    lo = math.sqrt(sum(w * h for g in groups for w, h, _place in g))
    hi, spots = lo, fits(lo)
    for _attempt in range(200):
        if spots is not None:
            break
        hi *= 1.25
        spots = fits(hi)
    else:
        raise RuntimeError("flower UV pieces did not pack")
    for _step in range(24):
        mid = 0.5 * (lo + hi)
        found = fits(mid)
        if found is None:
            lo = mid
        else:
            hi, spots = mid, found
    return hi, spots


def _pack_groups(groups) -> float:
    """Pack the pieces into the smallest square, `_UV_GAP` of it between
    pieces and half that round the outside. Each group -- a whorl's free
    petals / sepals, else one piece -- is a block of its pieces side by side,
    in as many rows (up to 5) as packs smallest; blocks tallest first, each at
    the lowest then leftmost free spot of a skyline. Places every piece and
    returns the square's side."""
    import itertools

    best = None
    for rows in itertools.product(*(range(1, min(len(g), 5) + 1) for g in groups)):
        side, spots = _smallest_square(groups, rows)
        if best is None or side < best[0] * (1.0 - 1e-9):
            best = (side, rows, spots)
    side, rows, spots = best
    gap = _UV_GAP * side
    for g, r, (x, y) in zip(groups, rows, spots, strict=True):
        for (_w, _h, place), (ox, oy) in zip(g, _block(g, r, gap)[2], strict=True):
            place(x + ox, y + oy)
    return side


def _skyline(sizes, order, side: float, gap: float) -> list | None:
    """Bottom-left skyline packing into a `side` square, or None if it fails."""
    sky = [(0.0, 0.0, side)]                       # (x, height, width) segments
    spots = [None] * len(sizes)
    eps = 1e-12 * side
    for i in order:
        w, h = sizes[i][0] + gap, sizes[i][1] + gap
        best = None
        for a in range(len(sky)):
            x = sky[a][0]
            if x + w > side + eps:
                break
            y, b = 0.0, a
            while b < len(sky) and sky[b][0] < x + w - eps:
                y = max(y, sky[b][1])
                b += 1
            if y + h <= side + eps and (best is None or (y, x) < best):
                best = (y, x)
        if best is None:
            return None
        y, x = best
        spots[i] = (x + 0.5 * gap, y + 0.5 * gap)
        out = []
        for sx, sy, sw in sky:
            end = sx + sw
            if end <= x + eps or sx >= x + w - eps:
                out.append((sx, sy, sw))
                continue
            if sx < x:
                out.append((sx, sy, x - sx))
            if end > x + w:
                out.append((x + w, sy, end - x - w))
        out.append((x, y + h, w))
        out.sort()
        sky = []
        for seg in out:
            if sky and abs(sky[-1][1] - seg[1]) <= eps:
                sky[-1] = (sky[-1][0], sky[-1][1], sky[-1][2] + seg[2])
            else:
                sky.append(seg)
    return spots


def _write_uv_carriers(obj, layout: dict) -> dict:
    """Author each part's UV records on its own hidden child carrier curve
    (point x/y/z, radius = mode) -- read back per ring by `_uv_record`."""
    carriers = {}
    for part in _UV_PARTS:
        child = _ensure_child_curve_object(obj, _UV_TAGS[part], f"UV_{part}")
        curve_data = child.data
        curve_data.splines.clear()
        recs = layout["records"].get(part) or []
        if recs:
            spline = curve_data.splines.new('POLY')
            spline.points.add(len(recs) - 1)
            for i, (x, y, z, mode) in enumerate(recs):
                spline.points[i].co = (x, y, z, 1.0)
                spline.points[i].radius = mode
        try:
            if not child.hide_get():
                child.hide_set(True)        # holds numbers, not geometry to see
        except RuntimeError:
            pass                            # not in this view layer (a baked stage)
        carriers[part] = child
    return {"carriers": carriers, "centres": layout["centres"], "scale": layout["scale"]}


def _weld_spec(lobes, tube, fuse: bool, layers: int) -> tuple:
    """(weld, first-layer lobe count) for `_tube_weld`: Fuse To Tube on, a tube
    actually built, and lobes to weld -- every layer has the same count."""
    if not fuse or not tube.points or not lobes:
        return (False, 0)
    return (True, len(lobes) // max(1, int(layers)))


def create_flower(context) -> bpy.types.Object:
    """New standalone flower: its own Collection (so the result is immediately
    assignable to an existing stem's Flowers.asset_collection) containing the
    flower's petal Curve object (+ its child Sepals/Disc/Receptacle objects)."""
    collection = bpy.data.collections.new("PlantGen_Flower")
    context.scene.collection.children.link(collection)

    curve = bpy.data.curves.new("PlantGen_Flower", 'CURVE')
    obj = bpy.data.objects.new("PlantGen_Flower", curve)
    obj[_FLOWER_TAG] = True
    collection.objects.link(obj)
    obj.location = context.scene.cursor.location

    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj

    ensure_flower_curves(obj)
    rebuild_flower(obj)
    return obj


def _bloom_shape_distance(a: FlowerSkeleton, b: FlowerSkeleton) -> float:
    """Total petal/sepal vertex displacement between two builds of the SAME
    recipe at different `bloom` -- valid because bloom never changes point
    COUNT (only position/width/tilt, see `_local_bloom`'s callers), so points
    line up 1:1 by index with no matching needed. Disc/Receptacle are
    excluded -- they don't move with `bloom`, only `maturity` (held fixed
    across bake stages), so including them would just dilute the signal with
    zeros."""
    total = 0.0
    for spines_a, spines_b in ((a.petals, b.petals), (a.sepals, b.sepals)):
        for spine_a, spine_b in zip(spines_a, spines_b, strict=True):
            for pa, pb in zip(spine_a.points, spine_b.points, strict=True):
                total += math.dist(pa, pb)
    return total


def _bloom_arc_positions(params: FlowerMeshParams, stages: int, samples: int = 160) -> list[float]:
    """`stages` UI-bloom values (0..1, endpoints always included) spaced evenly
    by ACTUAL mesh change rather than evenly in raw bloom units.

    Several shape fields ride on easing curves that flatten out well before
    `bloom` reaches 1 (`tip_fade`'s power curve is the biggest offender -- see
    its docstring), so equal steps in raw bloom do NOT produce equal steps in
    the mesh: baking 10 stages the old evenly-spaced way left the last several
    looking nearly identical, because most of the mesh's own motion was
    already spent earlier in the range. This reparameterizes by arc length:
    sample the recipe densely, measure how much the petals/sepals actually
    moved between consecutive samples (`_bloom_shape_distance`), and place
    `stages` stops at equal increments of that cumulative motion instead of
    equal increments of bloom -- more stops land where the shape is still
    changing fast, fewer where it's already saturated."""
    fine_ui = [i / (samples - 1) for i in range(samples)]
    fine_skeletons = [
        build_flower_mesh(dataclasses.replace(params, bloom=_bloom_ui_to_raw(ui)))
        for ui in fine_ui
    ]

    cumulative = [0.0]
    for i in range(1, samples):
        cumulative.append(
            cumulative[-1] + _bloom_shape_distance(fine_skeletons[i - 1], fine_skeletons[i]))
    total_arc = cumulative[-1]
    if total_arc <= 1e-9:
        # bloom has no visual effect on this recipe at all -- fall back to even spacing.
        return [i / (stages - 1) for i in range(stages)]

    positions = []
    j = 0
    for i in range(stages):
        target = i / (stages - 1) * total_arc
        while j < samples - 2 and cumulative[j + 1] < target:
            j += 1
        span = cumulative[j + 1] - cumulative[j]
        frac = (target - cumulative[j]) / span if span > 1e-12 else 0.0
        positions.append(fine_ui[j] + (fine_ui[j + 1] - fine_ui[j]) * frac)
    positions[0], positions[-1] = 0.0, 1.0
    return positions


def bake_bloom_stages(
    source_obj, context, stages: int = 3, variance: float = 0.0,
) -> list[bpy.types.Object]:
    """Bake `stages` fixed-Bloom copies of `source_obj` (e.g. bud/half-open/
    full-bloom for the default 3), spaced evenly by actual mesh change rather
    than raw bloom value -- see `_bloom_arc_positions` -- and add each to the
    shared pool (see add_to_pool) -- lets a stem's Flowers show buds
    toward the tip fading to open blooms toward the base instead of every
    instance sharing one bloom value (see plant.py's
    _instance_branch(pick_by_position=True), which relies on these being
    named/ordered ascending by bloom so Collection Info's alphabetical Separate
    Children ordering lines up with bloom order -- true regardless of jitter
    below, since each stop's name is derived from its OWN actual final bloom
    value, not its nominal even slot).

    `variance` (absolute, on the same 0..1 Bloom scale, seeded by
    `source_obj`'s own Seed the same way every other varied field in this
    add-on is -- see model.parameter.varied) jitters each stage off its perfectly
    even slot and clamps back into 0..1, so re-baking with the same Seed always
    reproduces the same stages, and re-rolling Seed re-rolls them like any other
    varied field.

    Each stop is a FULL, independent, still-live flower (its own object, its own
    plantgen_flower properties/curve widgets, its own Sepals/Disc/Receptacle
    children) with every setting copied from `source_obj` except Bloom fixed at
    that stop's value -- not a mesh bake (see bake_to_mesh for that).
    `source_obj` itself is left untouched, so its own Bloom slider keeps
    working as a live preview.

    A stop's Sepals/Disc/Receptacle children end up belonging to NO Collection
    at all (verified live: an Object Info node reads an object's geometry via a
    direct ID reference, same as the modifier inputs these are wired through --
    it doesn't need the referenced object to be in any Collection, or even in
    the scene, to evaluate correctly). They stay reachable/safe from Blender's
    orphan-data purge via the real references that already exist regardless
    (the parent relationship, and the GN modifier's own object-socket inputs
    holding them) -- no dedicated "Parts" collection needed just to give them
    somewhere to sit."""
    if not is_flower(source_obj) or stages < 2:
        return []
    params = flower_params_for_preset(source_obj)
    spacing = max(params.petal_length, params.disc_radius, 0.05) * 2.5
    pool = _get_or_create_collection(FLOWER_POOL_NAME)
    arc_positions = _bloom_arc_positions(params, stages)

    stops = []
    for i in range(stages):
        even = arc_positions[i]
        if variance:
            rng = random.Random(f"{params.seed}:{i}:bloom_stage")
            ui_bloom = min(1.0, max(0.0, even + rng.uniform(-variance, variance)))
        else:
            ui_bloom = even
        name = f"{source_obj.name}_Bloom{round(ui_bloom * 100):03d}"

        curve = bpy.data.curves.new(name, 'CURVE')
        stop_obj = bpy.data.objects.new(name, curve)
        stop_obj[_FLOWER_TAG] = True
        # Linked into the pool right away -- this is where it belongs long-term
        # anyway (see add_to_pool) -- so _ensure_child_curve_object (called via
        # apply_flower_params_to_object -> rebuild_flower below), which always
        # puts a new child in `stop_obj.users_collection[0]`, has somewhere to
        # put the children too. They're stripped back out of the pool below,
        # right after being created, since only the main object belongs there.
        pool.objects.link(stop_obj)
        stop_obj.location = source_obj.location
        stop_obj.location.x += spacing * (i + 1)

        ensure_flower_curves(stop_obj)
        stage_params = dataclasses.replace(params, bloom=_bloom_ui_to_raw(ui_bloom))
        # Materials are pointers, not shape params -- carry them over too.
        src = source_obj.plantgen_flower
        stop_obj.plantgen_flower.material = src.material
        apply_flower_params_to_object(stop_obj, stage_params)

        for child in stop_obj.children:
            if child.name in pool.objects:
                pool.objects.unlink(child)
        stops.append(stop_obj)
    return stops


# The one shared pool every flower's "Add To Pool" button feeds, and every
# stem's Flowers falls back to when its own asset_collection is left unset
# (see plant.rebuild()) -- a fixed, well-known name rather than a per-flower/
# per-stem picker: letting users name/match their own collection turned out to
# be pure friction (two independent pointers that had to be kept in sync by
# hand, for a case Milestone 1 doesn't need -- one stem at a time, no
# multi-species scenes yet). A stem can still override this with its own
# curated Collection by setting Flowers.asset_collection explicitly; that
# takes precedence whenever it's set.
FLOWER_POOL_NAME = "PlantGen_FlowerAssets"

# Where bake_to_mesh puts its fully-realized, no-longer-editable output --
# also a sibling of FLOWER_POOL_NAME for the same Separate-Children reason, not
# a literal "FlowerAssets > Baked" nested collection.
FLOWER_BAKED_NAME = f"{FLOWER_POOL_NAME}_Baked"


def default_pool() -> bpy.types.Collection | None:
    """The shared pool `add_to_pool` manages, if it's been created yet (None
    before the first `Add To Pool` click anywhere in this file) -- what
    `plant.rebuild()` falls back to when a stem's own Flowers.asset_collection
    is unset."""
    return bpy.data.collections.get(FLOWER_POOL_NAME)


def _get_or_create_collection(name: str) -> bpy.types.Collection:
    """The shared collection `name`, created on first use -- and always
    reachable from the current scene: it used to be linked only into the scene
    it was created in, so a bake from another scene (the next species' own)
    filled a collection that scene did not show, and the baked flowers could
    not be found (2026-10-21). A collection may sit in several scenes."""
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    root = bpy.context.scene.collection
    if collection != root and collection not in root.children_recursive:
        root.children.link(collection)
    return collection


def add_to_pool(obj) -> bool:
    """Link this flower's MAIN object (only) into the shared FLOWER_POOL_NAME
    Collection, creating it on first use -- the fix for the Collection-Info
    "Separate Children" hazard: the flower's own private Collection (from
    `create_flower`) also holds its Disc/Sepals/Receptacle child objects, so
    pointing a stem's asset_collection at that Collection directly would offer
    those three as extra pickable "flowers" too. Linking just `obj` here
    (objects can belong to multiple Collections at once) keeps the private
    Collection intact for editing while giving the stem's picker exactly one
    instanceable object for this flower. No-ops (returns False) if already
    linked; True if it just linked."""
    collection = _get_or_create_collection(FLOWER_POOL_NAME)
    if obj.name in collection.objects:
        return False
    collection.objects.link(obj)
    return True


def is_in_pool(obj) -> bool:
    """Whether this flower's main object is currently in the shared
    FLOWER_POOL_NAME Collection -- drives the Add To Pool button's toggled/
    depressed look."""
    pool = default_pool()
    return pool is not None and obj.name in pool.objects


def remove_from_pool(obj) -> bool:
    """The reverse of add_to_pool -- unlink this flower's main object from the
    shared pool. No-ops (returns False) if it wasn't in there (or the pool
    doesn't exist yet); True if it just unlinked."""
    pool = default_pool()
    if pool is None or obj.name not in pool.objects:
        return False
    pool.objects.unlink(obj)
    return True


def bake_to_mesh(source_obj, context) -> bpy.types.Object:
    """Freeze `source_obj` (any live flower -- the master recipe or one of its
    baked bloom stages) into a single static Mesh object: no modifiers, no
    plantgen_flower properties, no Sepals/Disc/Receptacle children -- just the
    fully-realized geometry exactly as currently shown, in the shared
    FLOWER_BAKED_NAME collection. For when an asset's shape is finished and you
    want the lightest-weight, no-longer-parametric version of it; `source_obj`
    itself is left untouched and still fully live/editable."""
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(eval_obj)
    name = f"{source_obj.name}_Baked"
    baked_obj = bpy.data.objects.new(name, mesh)
    baked_obj.matrix_world = source_obj.matrix_world.copy()
    _get_or_create_collection(FLOWER_BAKED_NAME).objects.link(baked_obj)
    return baked_obj


def apply_flower_params_to_object(obj, p: FlowerMeshParams) -> None:
    """Write a loaded `FlowerMeshParams` (see `model/io.py`'s Flower preset
    envelope) onto `obj.plantgen_flower` + its curve widgets, then rebuild once."""
    s = obj.plantgen_flower
    with suppress_rebuild(obj):
        s.bloom = _bloom_raw_to_ui(p.bloom)
        s.maturity = p.maturity
        s.petal_enabled = p.petal_enabled
        s.petal_bud_rotation_pitch = p.petal_bud_rotation_pitch
        s.petal_bud_curl, s.petal_bud_sink = p.petal_bud_curl, p.petal_bud_sink
        s.petal_bud_clamp_overlap = p.petal_bud_clamp_overlap
        s.petal_bud_tip_start = p.petal_bud_tip_start
        s.petal_bud_tip_lift = p.petal_bud_tip_lift
        s.petal_bud_tip_falloff = p.petal_bud_tip_falloff
        s.petal_bloom_start, s.petal_bloom_end = p.petal_bloom_start, p.petal_bloom_end
        s.petal_count, s.petal_count_variance = p.petal_count, p.petal_count_variance
        s.petal_length, s.petal_length_variance = p.petal_length, p.petal_length_variance
        s.petal_width, s.petal_width_variance = p.petal_width, p.petal_width_variance
        s.petal_cup, s.petal_cup_variance = p.petal_cup, p.petal_cup_variance
        s.petal_top_shape, s.petal_top_shape_variance = (
            p.petal_top_shape, p.petal_top_shape_variance
        )
        s.petal_bottom_shape, s.petal_bottom_shape_variance = (
            p.petal_bottom_shape, p.petal_bottom_shape_variance
        )
        s.petal_bud_cup = p.petal_bud_cup
        s.petal_scale, s.petal_scale_variance = p.petal_scale, p.petal_scale_variance
        s.resolution, s.seed = p.resolution, p.seed
        s.petal_width_resolution = p.petal_width_resolution
        s.sepal_width_resolution = p.sepal_width_resolution
        s.petal_tube_sides = p.petal_tube_sides
        s.sepal_tube_sides = p.sepal_tube_sides
        s.disc_sides = p.disc_sides
        s.receptacle_sides = p.receptacle_sides
        s.petal_flip_normals = p.petal_flip_normals
        s.sepal_flip_normals = p.sepal_flip_normals
        s.disc_flip_normals = p.disc_flip_normals
        s.receptacle_flip_normals = p.receptacle_flip_normals
        s.petal_tube_flip_normals = p.petal_tube_flip_normals
        s.sepal_tube_flip_normals = p.sepal_tube_flip_normals
        s.petal_sink, s.petal_sink_variance = p.petal_sink, p.petal_sink_variance
        s.petal_vertical_offset, s.petal_vertical_offset_variance = (
            p.petal_vertical_offset, p.petal_vertical_offset_variance
        )
        s.petal_spread_horizontal, s.petal_spread_horizontal_variance = (
            p.petal_spread_horizontal, p.petal_spread_horizontal_variance
        )
        s.petal_clamp_overlap = p.petal_clamp_overlap
        s.petal_tube_enabled = p.petal_tube_enabled
        s.petal_tube_length, s.petal_tube_length_variance = (
            p.petal_tube_length, p.petal_tube_length_variance
        )
        s.petal_tube_match_disc = p.petal_tube_match_disc
        s.petal_tube_radius, s.petal_tube_radius_variance = (
            p.petal_tube_radius, p.petal_tube_radius_variance
        )
        s.petal_tube_resolution = p.petal_tube_resolution
        s.petal_tube_noise = p.petal_tube_noise
        s.petal_tube_match_angle = p.petal_tube_match_angle
        s.petal_tube_fuse = p.petal_tube_fuse
        s.petal_bud_tube_length = p.petal_bud_tube_length
        s.petal_bud_tube_flare = p.petal_bud_tube_flare
        s.petal_layers = p.petal_layers
        s.petal_layer_spacing, s.petal_layer_spacing_variance = (
            p.petal_layer_spacing, p.petal_layer_spacing_variance
        )
        s.petal_layer_divergence = p.petal_layer_divergence
        s.petal_curl_updown, s.petal_curl_updown_variance = (
            p.petal_curl_updown, p.petal_curl_updown_variance
        )
        s.petal_curl_leftright, s.petal_curl_leftright_variance = (
            p.petal_curl_leftright, p.petal_curl_leftright_variance
        )
        s.petal_curl_outin, s.petal_curl_outin_variance = (
            p.petal_curl_outin, p.petal_curl_outin_variance
        )
        s.petal_curl_roll, s.petal_curl_roll_variance = (
            p.petal_curl_roll, p.petal_curl_roll_variance
        )
        s.petal_gravity, s.petal_gravity_variance = p.petal_gravity, p.petal_gravity_variance
        s.petal_twist, s.petal_twist_variance = p.petal_twist, p.petal_twist_variance
        s.petal_rotation_yaw, s.petal_rotation_yaw_variance = (
            p.petal_rotation_yaw, p.petal_rotation_yaw_variance
        )
        s.petal_rotation_pitch, s.petal_rotation_pitch_variance = (
            p.petal_rotation_pitch, p.petal_rotation_pitch_variance
        )
        s.petal_rotation_roll, s.petal_rotation_roll_variance = (
            p.petal_rotation_roll, p.petal_rotation_roll_variance
        )

        s.sepal_enabled = p.sepal_enabled
        s.sepal_count, s.sepal_count_variance = p.sepal_count, p.sepal_count_variance
        s.sepal_length, s.sepal_length_variance = p.sepal_length, p.sepal_length_variance
        s.sepal_width, s.sepal_width_variance = p.sepal_width, p.sepal_width_variance
        s.sepal_cup, s.sepal_cup_variance = p.sepal_cup, p.sepal_cup_variance
        s.sepal_top_shape, s.sepal_top_shape_variance = (
            p.sepal_top_shape, p.sepal_top_shape_variance
        )
        s.sepal_bottom_shape, s.sepal_bottom_shape_variance = (
            p.sepal_bottom_shape, p.sepal_bottom_shape_variance
        )
        s.sepal_bud_cup = p.sepal_bud_cup
        s.sepal_scale, s.sepal_scale_variance = p.sepal_scale, p.sepal_scale_variance
        s.sepal_bud_rotation_pitch = p.sepal_bud_rotation_pitch
        s.sepal_bud_curl, s.sepal_bud_sink = p.sepal_bud_curl, p.sepal_bud_sink
        s.sepal_bud_clamp_overlap = p.sepal_bud_clamp_overlap
        s.sepal_bud_tip_start = p.sepal_bud_tip_start
        s.sepal_bud_tip_lift = p.sepal_bud_tip_lift
        s.sepal_bud_tip_falloff = p.sepal_bud_tip_falloff
        s.sepal_bloom_start, s.sepal_bloom_end = p.sepal_bloom_start, p.sepal_bloom_end
        s.sepal_resolution = p.sepal_resolution
        s.sepal_sink, s.sepal_sink_variance = p.sepal_sink, p.sepal_sink_variance
        s.sepal_vertical_offset, s.sepal_vertical_offset_variance = (
            p.sepal_vertical_offset, p.sepal_vertical_offset_variance
        )
        s.sepal_azimuth_offset, s.sepal_z_offset = p.sepal_azimuth_offset, p.sepal_z_offset
        s.sepal_spread_horizontal, s.sepal_spread_horizontal_variance = (
            p.sepal_spread_horizontal, p.sepal_spread_horizontal_variance
        )
        s.sepal_clamp_overlap = p.sepal_clamp_overlap
        s.sepal_tube_enabled = p.sepal_tube_enabled
        s.sepal_tube_length, s.sepal_tube_length_variance = (
            p.sepal_tube_length, p.sepal_tube_length_variance
        )
        s.sepal_tube_match_disc = p.sepal_tube_match_disc
        s.sepal_tube_radius, s.sepal_tube_radius_variance = (
            p.sepal_tube_radius, p.sepal_tube_radius_variance
        )
        s.sepal_tube_resolution = p.sepal_tube_resolution
        s.sepal_tube_noise = p.sepal_tube_noise
        s.sepal_tube_match_angle = p.sepal_tube_match_angle
        s.sepal_tube_fuse = p.sepal_tube_fuse
        s.sepal_bud_tube_length = p.sepal_bud_tube_length
        s.sepal_bud_tube_flare = p.sepal_bud_tube_flare
        s.sepal_layers = p.sepal_layers
        s.sepal_layer_spacing, s.sepal_layer_spacing_variance = (
            p.sepal_layer_spacing, p.sepal_layer_spacing_variance
        )
        s.sepal_layer_divergence = p.sepal_layer_divergence
        s.sepal_curl_updown, s.sepal_curl_updown_variance = (
            p.sepal_curl_updown, p.sepal_curl_updown_variance
        )
        s.sepal_curl_leftright, s.sepal_curl_leftright_variance = (
            p.sepal_curl_leftright, p.sepal_curl_leftright_variance
        )
        s.sepal_curl_outin, s.sepal_curl_outin_variance = (
            p.sepal_curl_outin, p.sepal_curl_outin_variance
        )
        s.sepal_curl_roll, s.sepal_curl_roll_variance = (
            p.sepal_curl_roll, p.sepal_curl_roll_variance
        )
        s.sepal_gravity, s.sepal_gravity_variance = p.sepal_gravity, p.sepal_gravity_variance
        s.sepal_twist, s.sepal_twist_variance = p.sepal_twist, p.sepal_twist_variance
        s.sepal_rotation_yaw, s.sepal_rotation_yaw_variance = (
            p.sepal_rotation_yaw, p.sepal_rotation_yaw_variance
        )
        s.sepal_rotation_pitch, s.sepal_rotation_pitch_variance = (
            p.sepal_rotation_pitch, p.sepal_rotation_pitch_variance
        )
        s.sepal_rotation_roll, s.sepal_rotation_roll_variance = (
            p.sepal_rotation_roll, p.sepal_rotation_roll_variance
        )

        s.disc_enabled = p.disc_enabled
        s.disc_radius, s.disc_radius_variance = p.disc_radius, p.disc_radius_variance
        s.disc_height, s.disc_height_variance = p.disc_height, p.disc_height_variance
        s.disc_resolution = p.disc_resolution

        s.receptacle_enabled = p.receptacle_enabled
        s.receptacle_match_disc = p.receptacle_match_disc
        s.receptacle_radius, s.receptacle_radius_variance = (
            p.receptacle_radius, p.receptacle_radius_variance
        )
        s.receptacle_height, s.receptacle_height_variance = (
            p.receptacle_height, p.receptacle_height_variance
        )
        s.receptacle_resolution = p.receptacle_resolution

        for field in p.RAMP_FIELDS:
            ramp = getattr(p, field)
            y_remap = _FLOWER_CURVE_REMAPS[field]
            if y_remap is not None:
                # Inverse of flower_params_from_object's own remap -- widget
                # space is always 0..1, this ramp is in real (post-remap)
                # units, so writing it straight into the widget without
                # inverting first silently mis-set every affected field
                # (confirmed: Top/Bottom Shape's real no-op is 0.0, but that
                # was landing in the widget as raw Y=0.0, which reads back as
                # -0.01 -- a real, visible push, not a no-op).
                lo, hi = y_remap
                ramp = Ramp(points=[(x, (y - lo) / (hi - lo)) for x, y in ramp.points])
            set_flower_curve(obj, field, ramp)
    rebuild_flower(obj)


# --- curve-widget and modifier helpers (from the parametric add-on's `blender/plant.py`) ---

CURVE_SAMPLES = 24  # resolution when baking a widget curve -> Ramp


def _set_curve_points(node, points, handle_type: str = 'AUTO') -> None:
    c = node.mapping.curves[0]
    pts = list(points)
    while len(c.points) > 2:
        c.points.remove(c.points[-1])
    c.points[0].location = pts[0]
    c.points[-1].location = pts[-1]
    for p in pts[1:-1]:
        c.points.new(*p)
    if handle_type != 'AUTO':
        # CurveMapPoint only exposes location/handle_type/select — no numeric handle
        # offset control — so 'AUTO_CLAMPED' (no overshoot past the control points)
        # is as close as script-authored curves get to hand-tuned bezier handles.
        for p in c.points:
            p.handle_type = handle_type
    node.mapping.update()


def _ramp_from_node(node, samples: int = CURVE_SAMPLES) -> Ramp:
    cm = node.mapping
    cm.update()
    c = cm.curves[0]
    pts = [(i / (samples - 1), float(cm.evaluate(c, i / (samples - 1))))
           for i in range(samples)]
    return Ramp(points=pts)


def raw_ramp(node) -> Ramp:
    """The widget's actual authored control points (few), not a baked sample.

    Used for presets, so a saved `.json` preset stays small/git-diffable and the
    curve stays editable after a reload — `_ramp_from_node`'s dense bake is for
    feeding `build_stem`/`build_phyllotaxis`, which don't care about point count.
    """
    return Ramp(points=[(p.location[0], p.location[1]) for p in node.mapping.curves[0].points])


def _socket_identifier(ng, name: str) -> str:
    return next(
        item.identifier for item in ng.interface.items_tree
        if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == name
    )


def _set_modifier_input(mod, ident: str, value) -> None:
    """Blender 5.2 replaced the old `mod[identifier]` IDProperty dict access
    for GN modifier inputs with `mod.properties.inputs.<identifier>.value` --
    `mod[ident]` now raises "this type doesn't support IDProperties"
    unconditionally, for every input on every modifier. Found live: a fresh
    NodesModifier's every single input socket (including ones the old code
    never even touched, like the built-in Geometry input) failed the same
    way, confirming it's a structural API change, not a per-socket or
    per-instance quirk.
    """
    input_prop = getattr(mod.properties.inputs, ident)
    if input_prop.value != value:
        input_prop.value = value
