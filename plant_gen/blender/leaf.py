"""Controller: cut a low-poly leaf mesh from a texture atlas alpha channel.

The bpy bridge for the pure engine `growth/leaf.py`. It reads a Blender Image's
alpha, hands it to `growth.leaf.cut_leaf` / `cut_atlas` / `generate_leaf`, then
turns the returned boundary loop into a real Mesh object:

    * TRIANGULATION happens here (a bmesh concern the engine deliberately leaves
      out -- Blender ships no pure-Python Delaunay). `bmesh.ops.triangle_fill`
      fills the ordered outline loop with triangles WITHOUT adding vertices, so
      the whole mesh's vertex count stays equal to the boundary count and the
      engine's MAX_LEAF_VERTS cap still bounds the finished mesh. It also handles
      the concave/lobed outlines the boundary tracer produces.
    * UVs come straight from `CutLeaf.uvs` (atlas-relative), one per boundary
      vertex, so every leaf cut from the same atlas shares one material and
      batches.
    * an optional alpha-clipped Principled material wires the atlas image to
      Base Color + Alpha so the cut renders as the textured leaf immediately.

A cut makes a NEW object each time, tagged `plantgen_leaf` and dropped in the
`PlantGen_Leaves` collection -- a reusable-asset shape, so a cut leaf can be
assigned straight into a plant's leaf asset collection. Settings live scene-side
(`context.scene.plantgen_leaf`), not on the object, since they precede the
object a cut creates.
"""

import json
import os
import zlib
from contextlib import contextmanager

import bmesh
import bpy
import numpy as np

from ..growth import leaf as leaf_engine
from . import leaf_material

_LEAF_TAG = "plantgen_leaf"
# A generated (procedural) leaf carries this second tag as well as _LEAF_TAG. It
# distinguishes leaves the generate params can rebuild live from cutout leaves
# (which are traced from an atlas and have no generate params).
_LEAF_GEN_TAG = "plantgen_leaf_generated"
# A cutout rebuilt as a deformable triangle mesh carries this tag plus its traced
# outline (leaf-local XY, flattened) and its affine XY -> atlas-UV map, so it can
# be re-meshed live from `obj.plantgen_leaf_gen` like a generated leaf.
_LEAF_CUT_TAG = "plantgen_leaf_cutout"
_CUT_OUTLINE = "plantgen_cut_outline"
_CUT_UV = "plantgen_cut_uv"
_CUT_PIXEL = "plantgen_cut_pixel"      # atlas pixel size in metres (1 / ppu)
_LEAVES_COLLECTION = "PlantGen_Leaves"
_MASK_ATTR = "LeafMask"

# Which UI prop feeds which engine kwarg, for the numeric parameters that carry a
# per-parameter variance (`<prop>_var`). Order matches leaf_engine.VARIABLE_PARAMS.
VAR_PROP_TO_KEY = (
    ("gen_length", "length"), ("gen_width", "width"), ("gen_res_v", "stations"),
    ("gen_res_h", "res_h"), ("gen_margin_teeth", "margin_teeth"),
    ("gen_margin_depth", "margin_depth"), ("gen_curl", "curl"), ("gen_cup", "cup"),
    ("gen_twist", "twist"), ("gen_gravity", "gravity"), ("gen_wave", "wave"),
    ("gen_wave_count", "wave_count"), ("gen_noise", "noise"),
    ("gen_noise_scale", "noise_scale"),
    ("gen_rachis_length", "rachis_length"), ("gen_spread", "spread"),
    ("gen_leaflet_taper", "leaflet_taper"),
    ("gen_crumple", "crumple"), ("gen_crumple_scale", "crumple_scale"),
)
_VAR_PROPS = tuple(f"{prop}_var" for prop, _ in VAR_PROP_TO_KEY)
# Integer engine kwargs (rounded after variance) and their minimums.
_INT_KEYS = {"stations": 2, "res_h": 1, "margin_teeth": 1, "wave_count": 1}

# The generate parameter fields, shared verbatim by the scene template
# (`scene.plantgen_leaf`) and the per-object live params
# (`obj.plantgen_leaf_gen`). One list so a copy/build never drifts. The
# outline itself is not a field: it is the holder's stored PROFILE (below).
GEN_FIELDS = (
    "gen_shape", "gen_length", "gen_width", "gen_res_v",
    "gen_res_h", "gen_seed", "gen_margin", "gen_margin_teeth", "gen_margin_depth",
    "gen_curl", "gen_cup", "gen_twist", "gen_gravity", "gen_wave", "gen_wave_count",
    "gen_noise", "gen_noise_scale", "gen_crumple", "gen_crumple_scale",
    # compound leaves (Simple = the plain blade)
    "gen_compound", "gen_leaflets", "gen_terminal", "gen_sub_pairs",
    "gen_rachis_length", "gen_rachis_width", "gen_petiolule", "gen_spread",
    "gen_leaflet_taper", "gen_pinna_length", "gen_rachis_curl", "gen_petiole",
) + _VAR_PROPS
# Keys older presets may carry that no longer map to a field (the retired
# "Custom Profile" toggle -- the profile curve is now always the outline).
LEGACY_FIELDS = frozenset({"gen_use_profile"})

# Every holder stores its outline as a half-width profile (base -> tip samples in
# [0, 1]) in an ID property -- per leaf and in the .blend, so each leaf keeps its
# own shape and the one shared curve widget is only the EDITOR for the leaf being
# edited. A FloatVectorProperty cannot hold 33 values (max 32), hence the ID prop.
_PROFILE_KEY = "profile"
PROFILE_SAMPLES = 33          # the preset format's sample count
CUSTOM_SHAPE = "custom"       # gen_shape marker: the curve was edited by hand


def is_leaf(obj) -> bool:
    return bool(obj) and obj.type == 'MESH' and obj.get(_LEAF_TAG) is not None


def is_generated_leaf(obj) -> bool:
    return bool(obj) and obj.type == 'MESH' and obj.get(_LEAF_GEN_TAG) is not None


def is_cutout_leaf(obj) -> bool:
    """A cutout rebuilt as a deformable triangle mesh (live-editable)."""
    return bool(obj) and obj.type == 'MESH' and obj.get(_LEAF_CUT_TAG) is not None


def is_editable_leaf(obj) -> bool:
    """A leaf whose mesh rebuilds live from `obj.plantgen_leaf_gen`."""
    return is_generated_leaf(obj) or is_cutout_leaf(obj)


def active_holder(context):
    """The params holder being edited: the active generated leaf's live params,
    else the scene template (the panel draws the same one)."""
    view_layer = getattr(context, "view_layer", None)
    obj = view_layer.objects.active if view_layer is not None else None
    if is_generated_leaf(obj):
        return obj.plantgen_leaf_gen
    scene = getattr(context, "scene", None)
    return scene.plantgen_leaf if scene is not None else None


def lookup_shape(key: str):
    """A default (LEAF_SHAPES) or custom-folder shape by key, or None."""
    return leaf_engine.LEAF_SHAPES.get(key) or custom_shapes().get(key)


def holder_profile(p) -> tuple:
    """The holder's outline profile. A holder without one (never edited since
    this model) falls back to its shape's profile, then the default shape."""
    stored = p.get(_PROFILE_KEY)
    if stored is not None and len(stored) >= 2:
        return tuple(float(v) for v in stored)
    shape = lookup_shape(p.gen_shape) or leaf_engine.LEAF_SHAPES[
        leaf_engine.DEFAULT_LEAF_SHAPE]
    return shape.profile


def set_holder_profile(p, samples) -> None:
    p[_PROFILE_KEY] = [float(max(0.0, min(1.0, v))) for v in samples]


def copy_params(src, dst) -> None:
    """Copy every generate field AND the profile from one holder to another,
    without the per-field live rebuild or the shape callback firing."""
    with live_update_paused():
        for f in GEN_FIELDS:
            setattr(dst, f, getattr(src, f))
    set_holder_profile(dst, holder_profile(src))


def apply_shape(p, key: str) -> bool:
    """Make `p` take the shape `key`: its profile becomes the holder's outline and
    Width becomes Length * the shape's authored proportion. If `p` is the holder
    the curve widget is showing, the widget is reloaded too. False if unknown.
    Custom starts a new outline: one straight line from the full-width base to
    the tip (`start_custom`)."""
    if key == CUSTOM_SHAPE:
        start_custom(p)
        return True
    shape = lookup_shape(key)
    if shape is None:
        return False
    set_holder_profile(p, shape.profile)
    with live_update_paused():
        p.gen_width = max(1e-4, p.gen_length * shape.aspect)
    if _EDITOR["key"] == _holder_key(p):
        editor_load(p)
    return True


# A new Custom outline: tip (1, 0), the full-width base (0, 1) -- a draggable
# point -- and the base's midrib anchor (0, 0). Straight segments.
_CUSTOM_START = ((1.0, 0.0, 'VECTOR', 'VECTOR'), (0.0, 1.0, 'VECTOR', 'VECTOR'),
                 (0.0, 0.0, 'VECTOR', 'VECTOR'))


def start_custom(p) -> None:
    """Give `p` a fresh Custom outline: a single diagonal line, three points on
    the curve widget, ready to be shaped by adding and dragging points."""
    n = PROFILE_SAMPLES
    set_holder_profile(p, [1.0 - i / (n - 1) for i in range(n)])
    if _EDITOR["key"] == _holder_key(p):
        prof = leaf_profile(create=True)
        _write_path(prof, _CUSTOM_START)
        prof.initialize(_SEG_COUNT)
        _EDITOR["sig"] = _curve_signature(prof)
        _redraw_panels()


def _gen_kwargs(p) -> dict:
    """`generate_compound_leaf` kwargs from a generate-params holder (scene
    template or object), with each parameter's per-parameter variance
    (`<prop>_var`) applied via the global Seed -- so a seed change re-rolls a new
    variant of the same base. The outline is always the holder's stored profile
    (the leaflet's, for a compound leaf)."""
    base = {key: getattr(p, prop) for prop, key in VAR_PROP_TO_KEY}
    variances = {key: getattr(p, f"{prop}_var") for prop, key in VAR_PROP_TO_KEY}
    vals = leaf_engine.randomize(base, variances, seed=p.gen_seed)
    for key, lo in _INT_KEYS.items():
        vals[key] = max(lo, int(round(vals[key])))
    vals["length"] = max(1e-4, vals["length"])
    vals["width"] = max(1e-4, vals["width"])
    # `shape` only labels the venation; a custom/edited key maps to the default.
    shape = p.gen_shape if p.gen_shape in leaf_engine.LEAF_SHAPES else (
        leaf_engine.DEFAULT_LEAF_SHAPE)
    vals["rachis_length"] = max(0.0, vals["rachis_length"])
    return dict(shape=shape, margin=p.gen_margin, seed=p.gen_seed,
                width_samples=np.asarray(holder_profile(p)),
                compound=p.gen_compound, leaflets=p.gen_leaflets,
                terminal=p.gen_terminal, sub_pairs=p.gen_sub_pairs,
                rachis_width=p.gen_rachis_width, petiolule=p.gen_petiolule,
                pinna_length=p.gen_pinna_length, rachis_curl=p.gen_rachis_curl,
                petiole=p.gen_petiole, **vals)


# --- Profile curve (the outline editor) --------------------------------------
# The panel's curve widget shows the half-width silhouette (base left -> tip
# right) of the holder being edited; editing it rewrites that holder's stored
# profile (see the editor sync below). It is backed by a `CurveProfile` (NOT the RGB-curve
# `CurveMapping`): only CurveProfile's Free/Aligned handle types expose the
# draggable Bezier handles a leaf outline wants (CurveMapping offers only
# Auto/Vector/Auto-Clamped -- no draggable tangents). A CurveProfile cannot be
# created standalone (there is no bpy.data.curve_profiles), so it is hosted on a
# Bevel modifier's `custom_profile` on a hidden carrier object: not linked to any
# collection (invisible in the viewport) and fake-user'd so it persists in the
# .blend. The carrier's mesh is empty and never evaluated -- it exists only to own
# the CurveProfile datablock and be the `data` argument for `template_curveprofile`.
_PROFILE_OBJ = "PlantGen_LeafProfile"
_PROFILE_MOD = "profile"
# A gentle symmetric single hump -- the default silhouette a fresh curve shows.
_DEFAULT_PROFILE = (0.0, 0.55, 0.9, 1.0, 0.9, 0.55, 0.0)


def profile_carrier(create: bool = True):
    """The Bevel modifier that OWNS the Custom Profile `CurveProfile`, hosted on a
    hidden, unlinked, fake-user carrier object. This modifier is the `data`
    argument for the panel's `template_curveprofile`. Returns None if absent and
    `create` is False."""
    obj = bpy.data.objects.get(_PROFILE_OBJ)
    if obj is None:
        if not create:
            return None
        me = bpy.data.meshes.get(_PROFILE_OBJ) or bpy.data.meshes.new(_PROFILE_OBJ)
        obj = bpy.data.objects.new(_PROFILE_OBJ, me)
        obj.use_fake_user = True             # persist without a scene link
    mod = obj.modifiers.get(_PROFILE_MOD)
    if mod is None:
        if not create:
            return None
        mod = obj.modifiers.new(_PROFILE_MOD, 'BEVEL')
        mod.custom_profile.use_clip = True     # keep half-widths inside [0, 1]
        _seed_profile(mod.custom_profile, _DEFAULT_PROFILE)
    return mod


def leaf_profile(create: bool = True):
    """The Custom Profile `CurveProfile` datablock (or None)."""
    mod = profile_carrier(create=create)
    return mod.custom_profile if mod is not None else None


def _clamp01(v) -> float:
    return float(max(0.0, min(1.0, v)))


# A sample at or below this is on the midrib: its end anchor stands for it.
_ON_MIDRIB = 1e-6


def _anchored_path(samples) -> list:
    """The widget path for `samples` (base -> tip half-widths over even x), tip ->
    base: the samples between two midrib ANCHORS, (1, 0) and (0, 0).

    Blender never lets the artist drag a CurveProfile's first or last point, so
    those two cannot be the leaf's tip and base -- the base width would be stuck
    wherever it was seeded (or at (0, 1) after the widget's Preset / Reset). The
    anchors sit on the midrib, where the outline closes anyway; the tip and base
    samples are ordinary, draggable points beside them. An end sample already on
    the midrib is left out: its anchor is that point, and a duplicate on top of it
    would catch the click."""
    n = max(2, len(samples))
    xs = [i / (n - 1) for i in range(n)]
    inner = [(xs[i], _clamp01(samples[i])) for i in range(n)]
    if inner[0][1] <= _ON_MIDRIB:
        inner = inner[1:]
    if inner and inner[-1][1] <= _ON_MIDRIB:
        inner = inner[:-1]
    return [(1.0, 0.0)] + inner[::-1] + [(0.0, 0.0)]


def profile_is_anchored(prof) -> bool:
    """True when the widget's locked end points are the midrib anchors."""
    pts = prof.points
    if len(pts) < 2:
        return False
    a, b = pts[0].location, pts[len(pts) - 1].location
    return (abs(a[0] - 1.0) < 1e-6 and abs(a[1]) < 1e-6
            and abs(b[0]) < 1e-6 and abs(b[1]) < 1e-6)


def _write_path(prof, path) -> None:
    """Make `prof`'s points exactly `path` [(x, y, handle_1, handle_2)], in order
    (see `_seed_profile` for why points are grown first, then overwritten)."""
    pts = prof.points
    n = len(path)
    while len(pts) > 2:
        pts.remove(pts[-1])
    # Grow to n points; add at distinct x so update()'s de-duplication keeps them.
    while len(pts) < n:
        k = len(pts)
        pts.add(k / (n - 1), 0.5)
    # Overwrite in order -> collection order == path order == decreasing x.
    for i, (x, y, h1, h2) in enumerate(path):
        pts[i].location = (x, y)
        pts[i].handle_type_1 = h1
        pts[i].handle_type_2 = h2
    prof.update()


def reanchor_profile(prof) -> bool:
    """Give a curve whose locked end points are off the midrib (the widget's
    Preset menu or Reset: Line runs (1, 0) -> (0, 1), its base stuck at full
    width) its midrib anchors, keeping every point it has -- so the old ends
    become ordinary, draggable points and the shape is unchanged. False if it
    was anchored already."""
    if profile_is_anchored(prof):
        return False
    path = [(p.location[0], p.location[1], p.handle_type_1, p.handle_type_2)
            for p in prof.points]
    if not path:
        return False
    tip, base = path[0], path[-1]
    if not (abs(tip[0] - 1.0) < 1e-6 and abs(tip[1]) < 1e-6):
        path.insert(0, (1.0, 0.0, 'VECTOR', 'VECTOR'))
    if not (abs(base[0]) < 1e-6 and abs(base[1]) < 1e-6):
        path.append((0.0, 0.0, 'VECTOR', 'VECTOR'))
    _write_path(prof, path)
    return True


def _seed_profile(prof, samples, handle: str = 'AUTO') -> None:
    """Replace `prof`'s control points with `samples` (y over evenly spaced x in
    [0, 1]) between the midrib anchors (`_anchored_path`). A CurveProfile is a
    PATH drawn in point-collection order, and its `add()` inserts by nearest path
    segment -- which scrambles the order into a self-crossing zigzag when seeded
    programmatically. So `add()` is used only to reach the right point COUNT; the
    final step overwrites every point's location in order.

    The path must run RIGHT to LEFT (tip x=1 -> base x=0): the widget is a bevel
    profile editor whose native path goes (1,0) -> (0,1), and it draws the shaded
    fill by closing the path round the LEFT/bottom corner. A left-to-right path
    makes that fill polygon cross itself (a stray diagonal + triangles over the
    curve). Sampling sorts by x, so direction does not matter to the leaf.

    `handle` is every point's handle type: AUTO gives a smooth curve from a few
    points (but overshoots at steep steps); VECTOR joins the points with straight
    segments -- exactly the linear interpolation the engine renders with, so the
    full sample set with VECTOR reproduces a profile exactly. The artist switches a
    point to Free/Aligned in the widget to drag its handles. The anchors are
    always VECTOR: the outline drops straight to the midrib at both ends."""
    target = _anchored_path(samples)
    last = len(target) - 1
    _write_path(prof, [(x, y) + (('VECTOR',) * 2 if i in (0, last) else (handle,) * 2)
                       for i, (x, y) in enumerate(target)])


def set_profile_curve(samples, handle: str = 'AUTO') -> None:
    """Replace the curve widget's points with `samples` (see `_seed_profile`)."""
    _seed_profile(leaf_profile(create=True), samples, handle)


def sample_profile_curve(count: int = 33) -> tuple:
    """Sample the Custom Profile curve into `count` half-width values in [0, 1]
    over uniform x (base -> tip). CurveProfile is a PATH (its `evaluate`/`segments`
    are arc-length parameterised and its points are stored unsorted), so read the
    dense evaluated polyline, sort it by x, and linearly resample onto a uniform x
    grid -- robust to the point order and to small Bezier x-overshoot."""
    prof = leaf_profile(create=True)
    prof.update()
    prof.initialize(max(4 * count, _SEG_COUNT))
    # The evaluated segments do not land on the control points, so a sharp corner
    # (a VECTOR point -- a notch) would be cut by the interpolation. Control points
    # always lie on the curve: include them, making corners (and both ends) exact.
    seg = np.array([(p.location[0], p.location[1]) for p in prof.segments]
                   + [(p.location[0], p.location[1]) for p in prof.points],
                   dtype=np.float64)
    # Where the path runs straight up and down (the drop from the tip / base to its
    # midrib anchor) several samples share one x: the outline there is the highest
    # of them, and np.interp needs each x once.
    ux, inv = np.unique(np.round(seg[:, 0], 9), return_inverse=True)
    uy = np.full(len(ux), -np.inf)
    np.maximum.at(uy, inv, seg[:, 1])
    xs = np.linspace(0.0, 1.0, count)
    ys = np.interp(xs, ux, uy)
    return tuple(_clamp01(y) for y in ys)


# Evaluated segment count the curve is sampled/fingerprinted at.
_SEG_COUNT = 4 * PROFILE_SAMPLES


# --- Editor sync: one curve widget, many leaves ------------------------------
# The widget shows the ACTIVE holder's profile. A timer (registered by ui_leaf)
# polls because a CurveProfile edit fires no property callback: when the active
# holder changes it loads that holder's profile onto the curve; when the curve's
# fingerprint changes under the same holder the artist edited it, so the curve is
# sampled back into the holder, the shape switches to Custom and the leaf is
# rebuilt. The holder's stored profile is the truth; the curve is a view of it.
# Module state is fine here: it is only a cache of "what the widget shows" --
# undo/redo and file loads reset it (handlers in ui_leaf), so the curve is then
# reloaded from the (restored) holder rather than read back into it.
_EDITOR = {"key": None, "sig": None, "migrated": False}


def _holder_key(p):
    # `session_uid`, not the name: a deleted leaf's name can be reused (or a leaf
    # renamed), and a name key would then leave the old leaf's curve showing.
    idd = p.id_data
    return (type(idd).__name__, idd.session_uid)


def _curve_signature(prof) -> tuple:
    """Fingerprint of the curve's evaluated shape: the sampled segments (catches a
    Free-handle drag, which moves no point) plus the points and handle types."""
    if len(prof.segments) < 2:
        prof.initialize(_SEG_COUNT)
    sig = [round(c, 6) for s in prof.segments for c in s.location]
    for pt in prof.points:
        sig += [round(pt.location[0], 6), round(pt.location[1], 6),
                pt.handle_type_1, pt.handle_type_2]
    return tuple(sig)


def editor_reset(new_file: bool = False) -> None:
    """Forget what the widget shows (after undo/redo/file load): the next tick
    reloads it from the holder. A new file also re-runs the legacy migration."""
    _EDITOR["key"] = None
    _EDITOR["sig"] = None
    if new_file:
        _EDITOR["migrated"] = False


def editor_load(p) -> None:
    """Show holder `p`'s profile on the curve widget -- exactly (VECTOR points
    through its samples), so reading it back gives the same profile."""
    set_profile_curve(holder_profile(p), handle='VECTOR')
    prof = leaf_profile(create=True)
    prof.initialize(_SEG_COUNT)
    _EDITOR["key"] = _holder_key(p)
    _EDITOR["sig"] = _curve_signature(prof)
    _redraw_panels()


def _redraw_panels() -> None:
    wm = bpy.context.window_manager
    for win in getattr(wm, "windows", ()):
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def editor_tick(context) -> None:
    """One editor-sync step (see the section comment). Only runs while the leaf
    tool is in Generate mode, so a file that never uses it gets no carrier."""
    scene = getattr(context, "scene", None)
    s = getattr(scene, "plantgen_leaf", None) if scene is not None else None
    if s is None or s.mode != "generate":
        return
    if not _EDITOR["migrated"]:
        migrate_legacy_leaves()
        _EDITOR["migrated"] = True
    p = active_holder(context)
    if p is None:
        return
    if _EDITOR["key"] != _holder_key(p) or leaf_profile(create=False) is None:
        editor_load(p)
        return
    prof = leaf_profile(create=False)
    sig = _curve_signature(prof)
    if sig == _EDITOR["sig"]:
        return
    # The artist edited the curve: it becomes this holder's outline.
    set_holder_profile(p, sample_profile_curve(PROFILE_SAMPLES))
    # The widget's Preset / Reset puts the locked end points off the midrib (a
    # base stuck at full width): add the anchors, same points, same shape.
    if reanchor_profile(prof):
        prof.initialize(_SEG_COUNT)
    _EDITOR["sig"] = _curve_signature(prof)
    if p.gen_shape != CUSTOM_SHAPE:
        with live_update_paused():
            p.gen_shape = CUSTOM_SHAPE
    obj = p.id_data
    if is_generated_leaf(obj):
        try:
            rebuild_leaf(obj)
        except ValueError:
            pass


# --- Legacy leaves (built before profiles were stored per leaf) ---------------
def _profile_from_mesh(obj):
    """Recover a flat legacy leaf's outline from its own mesh: every grid row's
    half-width over half the Width, one sample per row. The rows are evenly
    spaced, so rebuilding at the same Resolution V lands every row exactly on a
    sample -- the same mesh back (resampling to fewer samples would round the
    corners). Exact for a leaf with no deform, no margin teeth and no size
    variance (then the rows ARE the profile the engine laid down); None
    otherwise."""
    g = obj.plantgen_leaf_gen
    toothed = (g.gen_margin != "entire" and g.gen_margin_depth > 0.0
               and not getattr(g, "gen_use_profile", False))
    if toothed or g.gen_length_var or g.gen_width_var or g.gen_res_v_var:
        return None
    me = obj.data
    if len(me.vertices) < 3 or g.gen_width <= 0.0:
        return None
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if np.any(np.abs(co[:, 2]) > 1e-9):
        return None                                # deformed: rows are not planar
    ys = np.round(co[:, 1], 7)
    rows = np.unique(ys)
    length = float(rows.max())
    if rows.size < 2 or length <= 0.0:
        return None
    half = np.array([np.abs(co[ys == y, 0]).max() for y in rows])
    frac = half / (0.5 * float(g.gen_width))
    if not np.allclose(np.diff(rows), length / (rows.size - 1), atol=1e-6):
        # Not the engine's uniform grid (a mesh left stale by an older state):
        # the outline envelope it shows, resampled -- the closest profile to it.
        frac = np.interp(np.linspace(0.0, 1.0, PROFILE_SAMPLES), rows / length, frac)
    return tuple(_clamp01(v) for v in frac)


def migrate_legacy_leaves() -> None:
    """Give every generated leaf that has no stored profile yet the outline its
    mesh already shows, so the first edit after this change does not reshape it.
    Leaves whose mesh cannot be read back exactly keep their shape's profile."""
    for obj in bpy.data.objects:
        if not is_generated_leaf(obj):
            continue
        g = obj.plantgen_leaf_gen
        if g.get(_PROFILE_KEY) is not None:
            continue
        prof = _profile_from_mesh(obj)
        set_holder_profile(g, prof if prof is not None else holder_profile(g))


# --- Custom shapes (the preferences' Leaf Shapes Folder) ---------------------
# Every *.json leaf preset in the folder becomes a shape keyed "custom:<file
# name>", built by the engine's `shape_from_preset` (the same rule the default
# shapes were baked with). Rescanned whenever the folder setting or the folder's
# modification time changes -- adding/removing a file shows up at once;
# Save Preset forces a rescan so overwriting a file does too.
_ADDON = __package__.split(".")[0] if __package__ else "plant_gen"
_CUSTOM = {"dir": None, "mtime": None, "shapes": {}, "numbers": {}}


def custom_shape_dir() -> str:
    addon = bpy.context.preferences.addons.get(_ADDON)
    path = getattr(addon.preferences, "leaf_shapes_path", "") if addon else ""
    return bpy.path.abspath(path) if path else ""


def custom_shapes(refresh: bool = False) -> dict:
    """The custom shapes {key: LeafShape}, cached per folder + folder mtime."""
    folder = custom_shape_dir()
    try:
        mtime = os.stat(folder).st_mtime if folder and os.path.isdir(folder) else None
    except OSError:
        mtime = None
    if refresh or folder != _CUSTOM["dir"] or mtime != _CUSTOM["mtime"]:
        _CUSTOM["dir"], _CUSTOM["mtime"] = folder, mtime
        _CUSTOM["shapes"], _CUSTOM["numbers"] = _scan_custom_shapes(folder)
    return _CUSTOM["shapes"]


def custom_shape_number(key: str) -> int:
    """The custom shape's stable enum number (from its file name)."""
    custom_shapes()
    return _CUSTOM["numbers"][key]


def _scan_custom_shapes(folder: str):
    shapes, numbers, used = {}, {}, set()
    if not folder or not os.path.isdir(folder):
        return shapes, numbers
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".json":
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("type") != "plantgen_leaf_preset":
                continue
            shape = leaf_engine.shape_from_preset(data, stem)
        except (OSError, ValueError, AttributeError) as exc:
            print(f"plant_gen: skipped leaf shape {name}: {exc}")
            continue
        key = f"custom:{stem}"
        # Stable enum number from the file name (EnumProperty stores the number),
        # clear of the default shapes' pinned 0..999; probe on a collision.
        number = 1000 + zlib.crc32(stem.lower().encode("utf-8")) % 100_000_000
        while number in used:
            number += 1
        used.add(number)
        shapes[key] = shape
        numbers[key] = number
    return shapes, numbers


def _image_alpha(image: bpy.types.Image) -> np.ndarray:
    """(H, W) float64 alpha in [0, 1], row 0 = TOP.

    Blender stores pixels as a flat RGBA array in BOTTOM-up row order; we pull
    the alpha channel (or luminance if the image has no alpha) via `foreach_get`
    -- far faster than indexing `image.pixels` -- and flip vertically into the
    engine's top-origin contract."""
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError(f"image '{image.name}' has zero size (not loaded?)")
    channels = image.channels
    buf = np.empty(w * h * channels, dtype=np.float32)
    image.pixels.foreach_get(buf)
    px = buf.reshape(h, w, channels)
    alpha = px[:, :, 3] if channels >= 4 else px[:, :, 0]
    alpha = alpha[::-1]  # bottom-up -> top-origin
    return np.ascontiguousarray(alpha, dtype=np.float64)


def _get_or_create_collection(context, name: str) -> bpy.types.Collection:
    """The shared collection `name`, created on first use and always linked
    into the current scene: linked only where it was created, a leaf generated
    from another scene landed outside its view layer and selecting it raised."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    root = context.scene.collection
    if coll not in root.children_recursive:
        root.children.link(coll)
    return coll


def _triangulate(verts: np.ndarray, uvs: np.ndarray, name: str) -> bpy.types.Mesh:
    """Ordered boundary loop -> a new Mesh, triangulated with `triangle_fill`
    (no new verts) and carrying an atlas UV layer. `verts` is (N, 3), `uvs`
    (N, 2), index-aligned."""
    bm = bmesh.new()
    bmverts = [bm.verts.new((float(x), float(y), float(z))) for x, y, z in verts]
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    n = len(bmverts)
    edges = [bm.edges.new((bmverts[i], bmverts[(i + 1) % n])) for i in range(n)]
    # normal=+Z pins face orientation to the leaf contract (face normal +Z),
    # rather than letting the fill guess a best-fit normal for the planar loop.
    result = bmesh.ops.triangle_fill(
        bm, use_beauty=True, use_dissolve=False, edges=edges, normal=(0.0, 0.0, 1.0))
    if not any(isinstance(g, bmesh.types.BMFace) for g in result.get("geom", ())):
        bm.free()
        raise ValueError("triangulation produced no faces (degenerate outline)")

    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        for loop in face.loops:
            u, v = uvs[loop.vert.index]
            loop[uv_layer].uv = (float(u), float(v))

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    _shade_smooth(mesh)
    mesh.update()
    return mesh


def _write_generated(mesh: bpy.types.Mesh, leaf) -> None:
    """(Re)fill an existing Mesh from a generate-path `CutLeaf` (faces + UVs +
    the edge mask as the red vertex colour). Clears the mesh first, so it works
    for both the initial build (empty mesh) and a live rebuild in place (keeping
    the object, its material slots and its modifiers). Faces may be quads or tris."""
    mesh.clear_geometry()
    mesh.from_pydata(
        [(float(x), float(y), float(z)) for x, y, z in leaf.verts],
        [], [list(f) for f in leaf.faces])
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            u, v = leaf.uvs[mesh.loops[li].vertex_index]
            uv_layer.data[li].uv = (float(u), float(v))

    if leaf.mask is not None:
        # Red = edge/wave mask (0 rigid midrib -> 1 wavy margin). POINT-domain
        # FLOAT_COLOR is index-aligned with verts, so foreach_set is exact.
        ca = mesh.color_attributes.new(name=_MASK_ATTR, type='FLOAT_COLOR', domain='POINT')
        cols = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
        cols[:, 0] = np.asarray(leaf.mask, dtype=np.float32)
        cols[:, 3] = 1.0
        ca.data.foreach_set("color", cols.ravel())
    _shade_smooth(mesh)
    mesh.update()


def _shade_smooth(mesh: bpy.types.Mesh) -> None:
    """Flag every polygon smooth-shaded (data-level, no operator/context needed)."""
    if mesh.polygons:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))


def _build_object(context, leaf, name: str) -> tuple[bpy.types.Object, bpy.types.Mesh]:
    """Build a `CutLeaf` into a tagged, collected object. A leaf carrying faces
    (generate path's quad grid) is built directly; a boundary-loop leaf (cutout
    path) is triangulated. Does NOT touch selection or location -- the caller
    places it (one leaf at the cursor, or a batch laid out in a row)."""
    if leaf.faces:
        mesh = bpy.data.meshes.new(name)
        _write_generated(mesh, leaf)
    else:
        mesh = _triangulate(leaf.verts, leaf.uvs, name)
    obj = bpy.data.objects.new(name, mesh)
    obj[_LEAF_TAG] = True
    _get_or_create_collection(context, _LEAVES_COLLECTION).objects.link(obj)
    return obj, mesh


def rebuild_leaf(obj) -> None:
    """Regenerate an editable leaf's mesh IN PLACE from its live per-object params
    (`obj.plantgen_leaf_gen`): a generated blade from its profile, a mesh
    cutout from its stored outline. The object, its transform, material slots and
    modifiers survive -- only the mesh geometry is rewritten. Drives the live
    update: a param edit calls this. Raises ValueError on degenerate params."""
    if is_cutout_leaf(obj):
        gen = _cutout_mesh(obj)
    else:
        gen = leaf_engine.generate_compound_leaf(**_gen_kwargs(obj.plantgen_leaf_gen))
    _write_generated(obj.data, gen)


# The params a cutout mesh uses (its size and outline come from the trace): its
# density (stored in Resolution V) and the deformers, each with its
# per-parameter variance via Seed.
_CUT_KEYS = ("stations", "curl", "cup", "twist", "gravity", "wave",
             "wave_count", "noise", "noise_scale", "crumple", "crumple_scale")


def _cut_kwargs(p) -> dict:
    pairs = [(prop, key) for prop, key in VAR_PROP_TO_KEY if key in _CUT_KEYS]
    base = {key: getattr(p, prop) for prop, key in pairs}
    variances = {key: getattr(p, f"{prop}_var") for prop, key in pairs}
    vals = leaf_engine.randomize(base, variances, seed=p.gen_seed)
    for key, lo in _INT_KEYS.items():
        if key in vals:
            vals[key] = max(lo, int(round(vals[key])))
    vals["density"] = vals.pop("stations")
    return dict(seed=p.gen_seed, **vals)


def _cutout_mesh(obj):
    """The triangle mesh for a cutout leaf from its stored outline + atlas UV map."""
    outline = np.asarray(obj[_CUT_OUTLINE], dtype=np.float64).reshape(-1, 2)
    uv_map = np.asarray(obj[_CUT_UV], dtype=np.float64).reshape(3, 2)
    return leaf_engine.cutout_grid(outline, uv_map, pixel=float(obj.get(_CUT_PIXEL, 0.0)),
                                   **_cut_kwargs(obj.plantgen_leaf_gen))


def _cut_material(image):
    """The leaf material for an atlas: one that already shows this image as its
    Base Color is reused (re-cutting does not pile up copies); otherwise a new
    leaf material with the atlas plugged in (its alpha cuts the leaves out)."""
    for mat in bpy.data.materials:
        if (leaf_material.is_leaf_material(mat)
                and mat.plantgen_leaf.base_color_image == image):
            return mat
    mat = leaf_material.new_material(f"{image.name}_leaf")
    mat.plantgen_leaf.base_color_image = image
    return mat


def _select_only(context, objects, active=None):
    for o in context.selected_objects:
        o.select_set(False)
    for o in objects:
        o.select_set(True)
    context.view_layer.objects.active = active or (objects[0] if objects else None)


def cut_atlas(context) -> tuple[list, tuple[str, ...], int]:
    """Cut EVERY separate part of the selected atlas into its own object.

    All parts share one material and are laid out in a row at the cursor so
    they don't overlap. Returns (objects, warnings, total_vertex_count). Raises
    ValueError on a missing image or an atlas with no parts above Min Region."""
    s = context.scene.plantgen_leaf
    if s.image is None:
        raise ValueError("no atlas image selected")

    alpha = _image_alpha(s.image)
    w, h = s.image.size
    # Mesh mode: keep the RAW trace -- the engine's one-sided smoothing then
    # simplifies it without ever stepping outward over alpha (the boundary budget's
    # two-sided reduction can).
    boundary = leaf_engine.MAX_LEAF_VERTS if s.cut_quads else s.boundary_verts
    leaves = leaf_engine.cut_atlas(
        alpha, atlas_size=(w, h), threshold=s.threshold,
        boundary_verts=boundary, pixels_per_unit=s.pixels_per_unit,
        base=s.base, trace=s.trace, min_area_frac=s.atlas_min_area)

    material = _cut_material(s.image) if s.create_material else None
    widths = [float(leaf.verts[:, 0].max() - leaf.verts[:, 0].min()) for leaf in leaves]
    gap = max(0.01, 0.15 * (max(widths) if widths else 0.0))

    cursor = context.scene.cursor.location
    x = float(cursor.x)
    objs, warnings, total = [], [], 0
    for i, leaf in enumerate(leaves):
        name = f"{s.image.name}_leaf_{i:02d}"
        if s.cut_quads:
            obj, built = _build_cutout_mesh(context, leaf, name, s)
        else:
            obj, _mesh = _build_object(context, leaf, name)
            built = leaf
        if material is not None:
            leaf_material.assign(obj, material)
        obj.location = (x, float(cursor.y), float(cursor.z))
        x += widths[i] + gap
        objs.append(obj)
        warnings.extend(leaf.warnings)
        warnings.extend(built.warnings)
        total += built.boundary_count

    _select_only(context, objs)
    return objs, tuple(dict.fromkeys(warnings)), total


def _build_cutout_mesh(context, leaf, name, s):
    """Build one traced cutout as a deformable triangle-mesh leaf: store its
    outline and atlas UV map on the object, seed its live params (density from
    the cutout settings, deformers flat), and mesh it."""
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    obj[_LEAF_TAG] = True
    obj[_LEAF_CUT_TAG] = True
    obj[_CUT_OUTLINE] = [float(c) for c in np.asarray(leaf.verts)[:, :2].ravel()]
    obj[_CUT_UV] = [float(c) for c in leaf_engine.uv_affine(leaf.verts, leaf.uvs).ravel()]
    obj[_CUT_PIXEL] = 1.0 / max(float(s.pixels_per_unit), 1e-9)
    g = obj.plantgen_leaf_gen
    with live_update_paused():
        g.gen_res_v = s.cut_density
    _get_or_create_collection(context, _LEAVES_COLLECTION).objects.link(obj)
    built = _cutout_mesh(obj)
    _write_generated(obj.data, built)
    return obj, built


def copy_cutout(context, src, seed: int) -> bpy.types.Object:
    """A new mesh cutout from `src`: the same traced outline, atlas UV map and
    live params, with Seed `seed` -- so its variances, Noise and Crumple make
    it differ. Shares `src`'s material; not placed or selected (the caller
    lays copies out). Raises ValueError on a degenerate outline."""
    obj = bpy.data.objects.new(src.name, bpy.data.meshes.new(src.data.name))
    for key in (_LEAF_TAG, _LEAF_CUT_TAG, _CUT_OUTLINE, _CUT_UV, _CUT_PIXEL):
        if key in src:
            value = src[key]
            obj[key] = value.to_list() if hasattr(value, "to_list") else value
    copy_params(src.plantgen_leaf_gen, obj.plantgen_leaf_gen)
    with live_update_paused():
        obj.plantgen_leaf_gen.gen_seed = seed
    _get_or_create_collection(context, _LEAVES_COLLECTION).objects.link(obj)
    _write_generated(obj.data, _cutout_mesh(obj))
    material = leaf_material.data_material(src)
    if material is not None:
        leaf_material.assign(obj, material)
    return obj


def cutout_extent(obj) -> tuple[float, float]:
    """A cutout's flat outline size (x, y) in leaf units."""
    outline = np.asarray(obj[_CUT_OUTLINE], dtype=np.float64).reshape(-1, 2)
    return float(np.ptp(outline[:, 0])), float(np.ptp(outline[:, 1]))


def generate_leaf(context, params=None, material=None
                  ) -> tuple[bpy.types.Object, tuple[str, ...], int]:
    """Build a procedural blade from a generate-params holder (`params`, default
    the scene template `scene.plantgen_leaf`). The new object is tagged as a
    generated leaf and its params are seeded onto `obj.plantgen_leaf_gen`, so
    every field can then be edited live (a param change rebuilds the mesh in
    place). It wears `material` (a leaf material to share -- New Leaf (copy)
    passes the source leaf's), else the shared main leaf material
    ("PlantGen Leaf Material").

    Returns (object, warnings, vertex_count). Raises ValueError on a degenerate
    outline."""
    p = params if params is not None else context.scene.plantgen_leaf
    gen = leaf_engine.generate_compound_leaf(**_gen_kwargs(p))
    shape = lookup_shape(p.gen_shape)
    label = shape.label if shape is not None else "Custom"
    if p.gen_compound != "simple":
        label = f"{label}_{p.gen_compound}"
    obj, _mesh = _build_object(context, gen, f"PlantGen_{label}_leaf")
    obj[_LEAF_GEN_TAG] = True
    # Seed the per-object live params + profile from the source (no per-field
    # rebuild -- the mesh is already built).
    copy_params(p, obj.plantgen_leaf_gen)
    leaf_material.assign(obj, material if material is not None
                         else leaf_material.main_material())
    obj.location = context.scene.cursor.location
    _select_only(context, [obj])
    return obj, tuple(gen.warnings), gen.boundary_count


# Live-update guard: set while seeding params so the property update callbacks do
# not each trigger a redundant rebuild during the initial copy.
_LIVE_SUSPENDED = [False]


def live_update_suspended() -> bool:
    return _LIVE_SUSPENDED[0]


@contextmanager
def live_update_paused():
    """Suspend the per-field live rebuild for a block of param writes (restoring
    the previous state after, so it nests)."""
    prev = _LIVE_SUSPENDED[0]
    _LIVE_SUSPENDED[0] = True
    try:
        yield
    finally:
        _LIVE_SUSPENDED[0] = prev


def apply_preset(holder, data) -> list[str]:
    """Apply a saved preset dict (as written by the Save Preset operator) to a
    generate-params holder (scene template or a generated leaf's live params):
    every known `gen_*` field, and its `profile_samples` as the holder's outline.
    Live update is suspended during the copy so no per-field rebuild fires -- the
    caller rebuilds once after. If the holder is the one the curve widget shows,
    the widget is reloaded. Returns human-readable warnings for fields it had to
    skip (an unknown key, or a value the property rejects -- e.g. a shape that
    no longer exists)."""
    warnings: list[str] = []
    params = data.get("params", {}) if isinstance(data, dict) else {}
    with live_update_paused():
        for key, value in params.items():
            if key in LEGACY_FIELDS:
                continue
            if key not in GEN_FIELDS:
                warnings.append(f"ignored unknown field '{key}'")
                continue
            try:
                setattr(holder, key, value)
            except (TypeError, ValueError):
                warnings.append(f"could not set '{key}' to {value!r}")
    samples = data.get("profile_samples") if isinstance(data, dict) else None
    if samples:
        set_holder_profile(holder, [float(s) for s in samples])
        # The dropdown must not name a shape the loaded curve is not.
        shape = lookup_shape(holder.gen_shape)
        if shape is None or len(shape.profile) != len(samples) or not np.allclose(
                shape.profile, holder_profile(holder), atol=1e-6):
            with live_update_paused():
                holder.gen_shape = CUSTOM_SHAPE
    else:                                  # no drawn curve saved: use the shape's
        shape = lookup_shape(holder.gen_shape)
        if shape is not None:
            set_holder_profile(holder, shape.profile)
    if _EDITOR["key"] == _holder_key(holder):
        editor_load(holder)
    return warnings
