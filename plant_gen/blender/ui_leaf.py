"""Leaf tool: scene-level settings + operators + N-panel, self-contained.

The leaf workshop for `plant_gen` -- cut leaves from a texture atlas, or
generate a procedural blade. Kept as one isolated module (its own PropertyGroups,
Scene/Object pointers, operators and panel, registered here) rather than folded
into the big `properties.py`/`operators.py`/`ui.py`, so the feature adds almost
nothing to those files' surface (only the Leaf Shapes Folder preference -- an
add-on can register a single AddonPreferences class, which lives in operators.py).
The heavy lifting is in `blender/leaf.py` (bridge) and `growth/leaf.py` (pure
engine); this file is UI + registration only.

Two homes for the generate params, sharing one field set (`_gen_props`) plus a
stored outline profile each:
  * `scene.plantgen_leaf` -- the TEMPLATE for the next new leaf (+ cutout
    settings). Only the Shape callback (load the shape) runs; no rebuilds.
  * `obj.plantgen_leaf_gen` -- the per-object LIVE params on a generated leaf.
    Each field has an update callback that rebuilds the mesh in place, so a leaf
    stays fully editable after creation.
The panel's profile curve edits whichever of the two is active (editor timer).
"""

import json
import os

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..growth.leaf import (
    DEFAULT_ATLAS_MIN_AREA_FRAC,
    DEFAULT_BOUNDARY_VERTS,
    DEFAULT_CRUMPLE_SCALE,
    DEFAULT_CUT_DENSITY,
    DEFAULT_LEAF_LENGTH,
    DEFAULT_LEAF_MARGIN,
    DEFAULT_LEAF_SEED,
    DEFAULT_LEAF_SHAPE,
    DEFAULT_LEAF_STATIONS,
    DEFAULT_LEAF_WIDTH,
    DEFAULT_LEAFLET_TAPER,
    DEFAULT_LEAFLETS,
    DEFAULT_MARGIN_DEPTH,
    DEFAULT_MARGIN_TEETH,
    DEFAULT_NOISE_SCALE,
    DEFAULT_PETIOLE,
    DEFAULT_PETIOLULE,
    DEFAULT_PINNA_LENGTH,
    DEFAULT_PIXELS_PER_UNIT,
    DEFAULT_RACHIS_LENGTH,
    DEFAULT_RACHIS_WIDTH,
    DEFAULT_RES_H,
    DEFAULT_SPREAD,
    DEFAULT_SUB_PAIRS,
    DEFAULT_TRIFOLIATE_RACHIS,
    DEFAULT_WAVE_COUNT,
    LEAF_MARGINS,
    LEAF_SHAPE_NOTES,
    LEAF_SHAPES,
    MARGIN_DEPTH_MAX,
    MAX_LEAF_VERTS,
    default_rachis_length,
)

# The per-parameter variance "+/-" glyph (`icons.py`, shared with FlowerGen).
from . import icons as _pg_icons
from . import leaf as leaf_ctl
from . import leaf_material as leaf_mat
from . import plant_icons


def _live_update(self, context):
    """Property update callback on a per-object param: rebuild that leaf's mesh
    in place (a generated blade or a mesh cutout). No-op while seeding params
    (guarded) or on anything else."""
    if leaf_ctl.live_update_suspended():
        return
    obj = self.id_data
    if leaf_ctl.is_editable_leaf(obj):
        try:
            leaf_ctl.rebuild_leaf(obj)
        except ValueError:
            pass


# EnumProperty stores the item NUMBER, so each shape's number is pinned: a leaf
# saved with the previous 29-shape list keeps its shape wherever the name still
# exists (rhombic -> rhomboid). Numbers of removed shapes are never reused. A NEW
# shape gets the next free number here -- never renumber an existing one.
_SHAPE_ENUM_IDS = {
    "acicular": 0, "cordate": 2, "cuneate": 3, "deltoid": 4, "elliptic": 5,
    "falcate": 6, "flabellate": 7, "hastate": 8, "lanceolate": 9, "linear": 10,
    "obcordate": 13, "obovate": 16, "orbicular": 17, "ovate": 18,
    "reniform": 23, "rhomboid": 24, "spatulate": 27, "subulate": 28,
    "acuminate": 29, "aristate": 30, "digitate": 31, "obtuse": 32,
    "pinnatisect": 33, "spear_shaped": 34, "truncate": 35,
}


_CUSTOM_NUMBER = 999                     # the "Custom" (hand-edited curve) marker
_HEADING_NUMBERS = (900_000_001, 900_000_002)
# Dynamic enum items must stay referenced from Python while Blender uses them.
_SHAPE_ITEMS_CACHE: list = []


def _shape_items(self, context):
    """Shape dropdown: a Default Shapes section (the built-ins, alphabetical, with
    pinned numbers), a Custom Shapes section (the preferences' folder, numbered
    from the file name) and the Custom marker the dropdown shows once the curve
    has been edited by hand. Empty-identifier items are section headings."""
    items = [("", "Default Shapes", "", _HEADING_NUMBERS[0])]
    spare = max(_SHAPE_ENUM_IDS.values()) + 1
    for i, (key, shape) in enumerate(sorted(LEAF_SHAPES.items(),
                                            key=lambda kv: kv[1].label)):
        items.append((key, shape.label,
                      LEAF_SHAPE_NOTES.get(key, f"{shape.label} blade outline"),
                      _SHAPE_ENUM_IDS.get(key, spare + i)))
    customs = leaf_ctl.custom_shapes()
    if customs:
        items.append(("", "Custom Shapes", "", _HEADING_NUMBERS[1]))
        for key, shape in sorted(customs.items(), key=lambda kv: kv[1].label.lower()):
            items.append((key, shape.label, f"{shape.label} (Leaf Shapes Folder)",
                          leaf_ctl.custom_shape_number(key)))
    items.append(None)
    items.append((leaf_ctl.CUSTOM_SHAPE, "Custom",
                  "A hand-drawn outline. Pick it to start a new one from a "
                  "single diagonal line", _CUSTOM_NUMBER))
    _SHAPE_ITEMS_CACHE[:] = items
    return _SHAPE_ITEMS_CACHE


def _shape_update(live):
    """Update callback for `gen_shape`: load the shape into the holder -- its
    profile becomes the outline (and the curve widget) and Width becomes Length *
    its authored proportion -- then run the per-field live rebuild (if any) once.
    Skipped while params are copied/loaded in bulk (live update suspended), so a
    preset's own profile and width win. Picking "Custom" starts a new outline:
    a single diagonal line (an edited curve becomes Custom without this firing)."""
    def update(self, context):
        if leaf_ctl.live_update_suspended():
            return
        if leaf_ctl.apply_shape(self, self.gen_shape) and live is not None:
            live(self, context)
    return update


_RACHIS_DEFAULTS = (DEFAULT_RACHIS_LENGTH, DEFAULT_TRIFOLIATE_RACHIS)


def _compound_update(live):
    """Update callback for `gen_compound`: a Rachis Length still at a type's
    default moves to the new type's (Trifoliate's short terminal stalk, the
    rest's shared length) -- one Rachis Length, starting where the type wants
    it; a value the user set is kept. Then the live rebuild, once. Skipped
    while params are copied / loaded in bulk, so a preset's own value wins."""
    def update(self, context):
        if leaf_ctl.live_update_suspended():
            return
        want = default_rachis_length(self.gen_compound)
        now = float(self.gen_rachis_length)
        # (a float property holds 32 bits: 0.15 reads back 0.15000000596...)
        if abs(now - want) > 1e-6 and any(abs(now - d) < 1e-6 for d in _RACHIS_DEFAULTS):
            self.gen_rachis_length = want          # its own update rebuilds
        elif live is not None:
            live(self, context)
    return update


def _gen_props(update):
    """The generate parameter fields as a fresh annotation dict. `update` is the
    per-field update callback (the live-rebuild callback for the per-object copy,
    or None for the scene template). Built fresh each call so the two owning
    classes never share a property-deferred instance."""
    props = {
        "gen_shape": EnumProperty(
            name="Leaf Shape", default=_SHAPE_ENUM_IDS[DEFAULT_LEAF_SHAPE],
            items=_shape_items,
            description="Blade outline. Picking a shape loads it into the profile "
                        "curve and sets Width from its authored proportion",
            update=_shape_update(update)),
        # Retired "Custom Profile" toggle: no longer drawn or copied, kept only so
        # leaves saved with it load and migrate (see leaf.migrate_legacy_leaves).
        "gen_use_profile": BoolProperty(
            name="Custom Profile (legacy)", default=False, options={"HIDDEN"}),
        "gen_length": FloatProperty(
            name="Length", default=DEFAULT_LEAF_LENGTH, min=1e-4, soft_max=1.0,
            subtype="DISTANCE", unit="LENGTH",
            description="Blade length, base to tip (metric)", update=update),
        "gen_width": FloatProperty(
            name="Width", default=DEFAULT_LEAF_WIDTH, min=1e-4, soft_max=1.0,
            subtype="DISTANCE", unit="LENGTH",
            description="Full blade width across the midrib where the profile "
                        "curve reaches the top of its box (metric)",
            update=update),
        "gen_res_v": IntProperty(
            name="Resolution V", default=DEFAULT_LEAF_STATIONS, min=2,
            max=MAX_LEAF_VERTS // 2, soft_max=96,
            description="Vertical resolution -- grid rows along the length. Higher "
                        "gives crisper teeth (teeth sit on the row boundary)",
            update=update),
        "gen_res_h": IntProperty(
            name="Resolution H", default=DEFAULT_RES_H, min=1,
            max=MAX_LEAF_VERTS // 4, soft_max=24,
            description="Horizontal resolution -- half-columns across the width, "
                        "each side. Independent of the vertical resolution",
            update=update),
        "gen_seed": IntProperty(
            name="Seed", default=DEFAULT_LEAF_SEED, min=0,
            description="Global random seed for this leaf (drives the noise deform)",
            update=update),
        # --- margin (edge profile) ---
        "gen_margin": EnumProperty(
            name="Margin", default=DEFAULT_LEAF_MARGIN,
            items=[(k, k.capitalize(),
                    {"entire": "Smooth, no teeth",
                     "dentate": "Sharp triangle teeth, sharp valleys",
                     "crenate": "Rounded-tip teeth with sharp valleys",
                     "denticulate": "Dentate but finer (more, smaller teeth)",
                     "serrate": "Sharp forward-raked teeth",
                     "serrulate": "Serrate but finer (more, smaller teeth)",
                     "doubleserrate": "Serrate teeth carrying smaller serrations",
                     "sinuate": "Smooth rounded wave",
                     "undulate": "Large rounded wave (fewer)",
                     "lobate": "Largest rounded wave, deep valleys",
                     "spiny": "Sharp, narrow spine points"}[k])
                   for k in LEAF_MARGINS],
            description="Edge profile (in-plane silhouette). Teeth = frequency, "
                        "Depth = size", update=update),
        "gen_margin_teeth": IntProperty(
            name="Teeth", default=DEFAULT_MARGIN_TEETH, min=1, soft_max=48,
            description="Teeth (or waves) along each edge", update=update),
        "gen_margin_depth": FloatProperty(
            name="Depth", default=DEFAULT_MARGIN_DEPTH, min=0.0, max=MARGIN_DEPTH_MAX,
            subtype="FACTOR",
            description="Tooth depth as a fraction of the local half-width "
                        f"(clamped at {MARGIN_DEPTH_MAX} to avoid clipping)",
            update=update),
        # --- deformers (all 0 = flat blade) ---
        "gen_curl": FloatProperty(
            name="Curl", default=0.0, min=-1.0, max=1.0, subtype="FACTOR",
            description="Bend the blade along its length into an arc: + curls the "
                        "tip up, - down (constant curvature)", update=update),
        "gen_cup": FloatProperty(
            name="Cup", default=0.0, min=-1.0, max=1.0, subtype="FACTOR",
            description="Curve the cross-section: + lifts the two edges "
                        "(a gutter/taco), - drops them", update=update),
        "gen_twist": FloatProperty(
            name="Twist", default=0.0, min=-1.0, max=1.0, subtype="FACTOR",
            description="Progressively roll the cross-section about the midrib "
                        "from base to tip", update=update),
        "gen_gravity": FloatProperty(
            name="Gravity", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
            description="Downward droop under the blade's own weight; the tip "
                        "sags most (cantilever profile)", update=update),
        "gen_wave": FloatProperty(
            name="Wave", default=0.0, min=-1.0, max=1.0, subtype="FACTOR",
            description="Ripple the edges up and down along the length; the "
                        "midrib stays flat (edge-masked, red vertex colour)",
            update=update),
        "gen_wave_count": IntProperty(
            name="Wave Count", default=DEFAULT_WAVE_COUNT, min=1, soft_max=8,
            description="How many ripples run along the length (with Wave)",
            update=update),
        "gen_noise": FloatProperty(
            name="Noise", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
            description="Perlin-like organic wobble; small values add natural "
                        "imperfection (uses the Seed)", update=update),
        "gen_noise_scale": FloatProperty(
            name="Noise Scale", default=DEFAULT_NOISE_SCALE, min=1.0, soft_max=12.0,
            description="Noise frequency -- cells of wobble across the blade",
            update=update),
        "gen_crumple": FloatProperty(
            name="Crumple", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
            description="Scrunch the whole blade like a dried leaf: the midrib bends "
                        "up, down and sideways at random along its length, and the "
                        "blade follows as one piece (Noise only roughens its surface; "
                        "uses the Seed)", update=update),
        "gen_crumple_scale": FloatProperty(
            name="Crumple Scale", default=DEFAULT_CRUMPLE_SCALE, min=1.0, soft_max=16.0,
            description="How many bends run along the blade (with Crumple)",
            update=update),
        # --- compound leaf (Simple = one blade; the blade above is the leaflet) ---
        "gen_compound": EnumProperty(
            name="Compound", default="simple", items=_COMPOUND_ITEMS,
            description="Leaf type: one blade, or leaflets (each the blade above) "
                        "on a rachis. Picking Trifoliate shortens a default Rachis "
                        "Length to its short terminal stalk", update=_compound_update(update)),
        "gen_leaflets": IntProperty(
            name="Leaflets", default=DEFAULT_LEAFLETS, min=1, max=30, soft_max=12,
            description="Palmate: leaflets in the fan. Pinnate: leaflet pairs. "
                        "Bipinnate: pinna pairs", update=update),
        "gen_terminal": BoolProperty(
            name="Terminal Leaflet", default=True,
            description="End the rachis in a leaflet (odd-pinnate); off ends it "
                        "between the last pair (even-pinnate). Bipinnate: a terminal "
                        "pinna and terminal leaflets on each pinna", update=update),
        "gen_sub_pairs": IntProperty(
            name="Leaflet Pairs", default=DEFAULT_SUB_PAIRS, min=1, max=30,
            soft_max=12, description="Bipinnate: leaflet pairs on each pinna",
            update=update),
        "gen_rachis_length": FloatProperty(
            name="Rachis Length", default=DEFAULT_RACHIS_LENGTH, min=0.0,
            soft_max=0.6, subtype="DISTANCE", unit="LENGTH",
            description="Length of the central stalk the leaflets sit on "
                        "(trifoliate: up to the terminal leaflet)", update=update),
        "gen_rachis_width": FloatProperty(
            name="Rachis Width", default=DEFAULT_RACHIS_WIDTH, min=0.0,
            soft_max=0.02, subtype="DISTANCE", unit="LENGTH",
            description="Width of the rachis strip at its base (it tapers to half "
                        "at the tip)", update=update),
        "gen_petiolule": FloatProperty(
            name="Stalks", default=DEFAULT_PETIOLULE, min=0.0, soft_max=0.05,
            subtype="DISTANCE", unit="LENGTH",
            description="Length of each leaflet's own stalk (petiolule), scaled "
                        "with the leaflet", update=update),
        "gen_spread": FloatProperty(
            name="Spread", default=DEFAULT_SPREAD, min=0.0, max=120.0,
            description="Angle (degrees) of the side leaflets from the axis they "
                        "leave -- palmate: of the outermost leaflets", update=update),
        "gen_leaflet_taper": FloatProperty(
            name="Taper", default=DEFAULT_LEAFLET_TAPER, min=0.05, max=1.0,
            subtype="FACTOR",
            description="Size of the base pair (palmate: outermost leaflets; "
                        "bipinnate: lowest pinnae) relative to the tip / centre "
                        "one", update=update),
        "gen_pinna_length": FloatProperty(
            name="Pinna Length", default=DEFAULT_PINNA_LENGTH, min=0.05, max=2.0,
            subtype="FACTOR",
            description="Bipinnate: each pinna's length as a fraction of the "
                        "rachis length", update=update),
        "gen_petiole": FloatProperty(
            name="Petiole", default=DEFAULT_PETIOLE, min=0.0, soft_max=0.3,
            subtype="DISTANCE", unit="LENGTH",
            description="Length of the leaf's main stalk, from its base up to "
                        "where the leaflets start. On a plant the leaf component's "
                        "own Petiole is drawn as well -- shorten one of them",
            update=update),
        "gen_rachis_curl": FloatProperty(
            name="Rachis Curl", default=0.0, min=-1.0, max=1.0, subtype="FACTOR",
            description="Bend the whole leaf along the rachis into an arc: + up, "
                        "- down (the leaflets ride along)", update=update),
    }
    # Per-parameter variance (like the parametric add-on's Parameter.variance): a fraction the
    # global Seed jitters the value by. All default 0 (no randomisation).
    for prop, label in _VAR_LABELS.items():
        props[f"{prop}_var"] = FloatProperty(
            name=f"{label} Variance", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
            description=f"Randomise {label} by up to this fraction, driven by Seed",
            update=update)
    return props


# The numeric generate params that carry a variance slider, with their short label.
_VAR_LABELS = {
    "gen_length": "Length", "gen_width": "Width", "gen_res_v": "Resolution V",
    "gen_res_h": "Resolution H", "gen_margin_teeth": "Teeth",
    "gen_margin_depth": "Depth", "gen_curl": "Curl", "gen_cup": "Cup",
    "gen_twist": "Twist", "gen_gravity": "Gravity", "gen_wave": "Wave",
    "gen_wave_count": "Wave Count", "gen_noise": "Noise",
    "gen_noise_scale": "Noise Scale", "gen_rachis_length": "Rachis Length",
    "gen_spread": "Spread", "gen_leaflet_taper": "Taper",
    "gen_crumple": "Crumple", "gen_crumple_scale": "Crumple Scale",
}

# Compound leaf types, numbers pinned (EnumProperty stores the number).
_COMPOUND_ITEMS = [
    ("simple", "Simple", "One blade", 0),
    ("trifoliate", "Trifoliate", "Three leaflets: a terminal one on the rachis tip "
     "and a pair at its base (Rachis Length 0: all from one point, like clover)", 1),
    ("palmate", "Palmate", "Leaflets fanned from one point (horse chestnut, lupin)", 2),
    ("pinnate", "Pinnate", "Leaflet pairs along the rachis, with or without a "
     "terminal leaflet (ash, rose)", 3),
    ("bipinnate", "Bipinnate", "Pinnate of pinnates: each side branch (pinna) "
     "carries its own leaflet pairs (mimosa, acacia)", 4),
]


class PLANTGEN_PG_leaf(PropertyGroup):
    """Leaf-tool settings on `scene.plantgen_leaf`. CUTOUT settings + the
    GENERATE template for the next new leaf (generate fields injected below)."""
    mode: EnumProperty(
        name="Mode", default="cutout",
        items=[
            ("cutout", "Cutout", "Trace leaves from a texture atlas's alpha channel"),
            ("generate", "Generate", "Build a procedural blade from a shape + size"),
        ],
        description="How to produce the leaf mesh")

    # --- CUTOUT mode ---
    image: PointerProperty(
        name="Atlas", type=bpy.types.Image,
        description="Leaf texture atlas to cut from; its alpha channel is the silhouette")
    threshold: FloatProperty(
        name="Alpha Threshold", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
        description="Alpha at or above this counts as opaque leaf")
    boundary_verts: IntProperty(
        name="Boundary Verts", default=DEFAULT_BOUNDARY_VERTS, min=3,
        max=MAX_LEAF_VERTS, soft_max=128,
        description="Target vertex count of each traced outline. Hard-capped at "
                    f"{MAX_LEAF_VERTS}")
    pixels_per_unit: FloatProperty(
        name="Pixels / Metre", default=DEFAULT_PIXELS_PER_UNIT, min=1.0,
        soft_max=8000.0,
        description="Atlas pixels per Blender unit -- sets the leaf's real-world size")
    atlas_min_area: FloatProperty(
        name="Min Region", default=DEFAULT_ATLAS_MIN_AREA_FRAC, min=0.0,
        soft_max=0.02, precision=4, subtype="FACTOR",
        description="Opaque parts smaller than this fraction of the whole atlas "
                    "are skipped as specks/dust when cutting every part")
    base: EnumProperty(
        name="Base End", default="auto",
        items=[
            ("auto", "Auto (bottom)", "Petiole at the image bottom (tip-up atlas)"),
            ("bottom", "Bottom", "Petiole at the image bottom (tip up)"),
            ("top", "Top", "Petiole at the image top (tip down)"),
        ],
        description="Which vertical end of the image is the petiole/attachment")
    trace: EnumProperty(
        name="Trace", default="auto",
        items=[
            ("auto", "Auto", "Fast sub-pixel rail, falling back to boundary-follow "
             "when the leaf is lobed"),
            ("rail", "Rail", "Sub-pixel scanline rail; assumes a single-lobe outline"),
            ("boundary", "Boundary", "Pixel-resolution boundary follow; any shape, "
             "preserves lobes"),
        ],
        description="Silhouette tracing method")
    create_material: BoolProperty(
        name="Create Material", default=True,
        description="Give the cut leaves a leaf material with the atlas as its Base "
                    "Color (its alpha cuts the leaves out). Reuses the atlas's leaf "
                    "material if one exists")
    cut_quads: BoolProperty(
        name="Deformable Mesh", default=True,
        description="Rebuild each cut leaf as an even triangle mesh inside its "
                    "traced silhouette, so it can be curled, cupped, twisted... like "
                    "a generated leaf. Off: the traced outline as a flat fill")
    cut_density: IntProperty(
        name="Density", default=DEFAULT_CUT_DENSITY, min=2,
        max=200, soft_max=60,
        description="Triangle edges across each cut leaf's larger dimension. The "
                    "silhouette is kept regardless; this sets the inside")


# Inject the generate template fields (no live update -- the template only seeds
# a NEW leaf; edits after creation happen on the per-object copy).
PLANTGEN_PG_leaf.__annotations__.update(_gen_props(None))


class PLANTGEN_PG_leaf_gen(PropertyGroup):
    """Per-object LIVE generate params on `obj.plantgen_leaf_gen`. Every field
    rebuilds the leaf mesh in place on change (fields injected below)."""


PLANTGEN_PG_leaf_gen.__annotations__ = _gen_props(_live_update)


def _uv_checker_changed(self, context):
    """Swap the leaf to the plant's shared UV checker (an object-level override,
    so its own material is kept) and, like the plant's toggle, switch Solid 3D
    views to Texture colouring while on (back to Material when off)."""
    obj = self.id_data
    if not leaf_ctl.is_editable_leaf(obj):
        return
    leaf_mat.set_uv_checker(obj, self.uv_checker)
    screen = getattr(context, "screen", None) if context is not None else None
    for area in (screen.areas if screen is not None else ()):
        if area.type != 'VIEW_3D':
            continue
        shading = area.spaces.active.shading
        if shading.type != 'SOLID':
            continue
        if self.uv_checker:
            shading.color_type = 'TEXTURE'
        elif shading.color_type == 'TEXTURE':
            shading.color_type = 'MATERIAL'


# Not a generate field (not copied by New Leaf, not in presets): a view toggle.
PLANTGEN_PG_leaf_gen.__annotations__["uv_checker"] = BoolProperty(
    name="UV Checker", default=False, update=_uv_checker_changed,
    description="Show the leaf with a checker texture instead of its material, to "
                "judge the UVs. Switches the viewport to texture colouring")


def _var_holder(context):
    """The params holder the buttons edit: the active mesh cutout's or generated
    leaf's live params if one is selected, else the scene template -- the same
    routing the panel uses."""
    obj = context.active_object
    if leaf_ctl.is_cutout_leaf(obj):
        return obj.plantgen_leaf_gen
    return leaf_ctl.active_holder(context)


class PLANTGEN_OT_leaf_edit_variance(Operator):
    """Variance lives behind a small +/- button on the row it varies (the parametric add-on's
    pattern) rather than an inline slider -- the button opens this popup and shows
    depressed when the parameter carries variance. The popup draws the holder's
    REAL variance property (not an operator copy), so it edits live and Alt
    applies it to every selected leaf, like any other field."""

    bl_idname = "plantgen.leaf_edit_variance"
    bl_label = "Edit Variance"
    bl_description = ("Set how much this parameter randomly deviates on each Seed "
                      "re-roll (Alt while editing: every selected leaf)")
    bl_options = {"INTERNAL"}

    prop_name: StringProperty(options={"HIDDEN"})   # e.g. "gen_curl_var"
    label: StringProperty(options={"HIDDEN"})

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=250)

    def draw(self, context):
        holder = _var_holder(context)
        if hasattr(holder, self.prop_name):
            self.layout.prop(holder, self.prop_name, text=self.label or self.prop_name,
                             slider=True)

    def execute(self, context):
        return {"FINISHED"}


class PLANTGEN_OT_leaf_save_preset(Operator, ExportHelper):
    """Save the current generate settings -- shape, margin, deform, resolution,
    seed, every per-parameter variance, and the profile curve -- to a simple JSON
    preset. Saved into the preferences' Leaf Shapes Folder (the default location
    when set), it appears under Custom Shapes in the Shape list."""

    bl_idname = "plantgen.leaf_save_preset"
    bl_label = "Save Leaf Preset"
    bl_description = ("Write all current generate settings (including the profile "
                      "curve) to a JSON preset -- saved into the Leaf Shapes Folder, "
                      "it becomes a Custom Shape")
    bl_options = {"REGISTER"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        # Default name: the edited leaf, else the shape; default folder: the
        # Leaf Shapes Folder so the save shows up as a custom shape.
        p = _var_holder(context)
        obj = context.active_object
        shape = leaf_ctl.lookup_shape(p.gen_shape)
        if leaf_ctl.is_generated_leaf(obj):
            base = obj.name
        else:
            base = shape.label if shape is not None else "Leaf Shape"
        folder = leaf_ctl.custom_shape_dir()
        name = bpy.path.clean_name(base) + ".json"
        self.filepath = os.path.join(folder, name) if folder else name
        return super().invoke(context, event)

    def execute(self, context):
        p = _var_holder(context)
        params = {f: _jsonable(getattr(p, f)) for f in leaf_ctl.GEN_FIELDS}
        data = {
            "type": "plantgen_leaf_preset",
            "version": 1,
            "params": params,
            "profile_samples": list(leaf_ctl.holder_profile(p)),
        }
        try:
            with open(self.filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            self.report({'ERROR'}, f"Could not write preset: {exc}")
            return {'CANCELLED'}
        leaf_ctl.custom_shapes(refresh=True)      # an overwrite keeps the dir mtime
        self.report({'INFO'}, f"Saved leaf preset to {self.filepath}")
        return {'FINISHED'}


class PLANTGEN_OT_leaf_load_preset(Operator, ImportHelper):
    """Load generate settings from a JSON preset written by Save Preset -- shape,
    margin, deform, resolution, seed, variances, and the Custom Profile curve --
    onto the active generated leaf (rebuilding it live) or, with none active, the
    scene template for the next new leaf."""

    bl_idname = "plantgen.leaf_load_preset"
    bl_label = "Load Leaf Preset"
    bl_description = ("Load generate settings (including the profile curve) from a "
                      "JSON preset file")
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self.report({'ERROR'}, f"Could not read preset: {exc}")
            return {'CANCELLED'}
        if not isinstance(data, dict) or data.get("type") != "plantgen_leaf_preset":
            self.report({'ERROR'}, "Not a PlantGen leaf preset (.json)")
            return {'CANCELLED'}
        warnings = leaf_ctl.apply_preset(_var_holder(context), data)
        obj = context.active_object
        if leaf_ctl.is_generated_leaf(obj):
            try:
                leaf_ctl.rebuild_leaf(obj)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
        for w in list(warnings)[:3]:
            self.report({'WARNING'}, w)
        self.report({'INFO'}, "Loaded leaf preset")
        return {'FINISHED'}


def _jsonable(value):
    """Coerce a Blender property value to a JSON-native scalar."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


class PLANTGEN_OT_leaf_cut(Operator):
    bl_idname = "plantgen.leaf_cut"
    bl_label = "Cut Atlas"
    bl_description = (
        "Cut every separate opaque part of the selected atlas into its own "
        "low-poly leaf mesh")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            objs, warnings, nverts = leaf_ctl.cut_atlas(context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        for warning in list(warnings)[:3]:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, f"Cut {len(objs)} leaf part(s), {nverts} verts total")
        return {'FINISHED'}


class PLANTGEN_OT_leaf_material_new(Operator):
    bl_idname = "plantgen.leaf_material_new"
    bl_label = "New Leaf Material"
    bl_description = ("Give the active leaf a new leaf material of its own (the "
                      "master leaf shader, untextured)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return leaf_ctl.is_editable_leaf(context.active_object)

    def execute(self, context):
        leaf_mat.assign(context.active_object, leaf_mat.new_material())
        return {"FINISHED"}


def _varies_with_seed(obj) -> bool:
    """Whether a new Seed changes this leaf: Noise, Crumple or a variance on a
    parameter it uses (a mesh cutout uses only its density and deformers)."""
    g = obj.plantgen_leaf_gen
    if g.gen_noise > 0.0 or g.gen_crumple > 0.0:
        return True
    pairs = leaf_ctl.VAR_PROP_TO_KEY
    if leaf_ctl.is_cutout_leaf(obj):
        pairs = [(prop, key) for prop, key in pairs if key in leaf_ctl._CUT_KEYS]
    return any(getattr(g, f"{prop}_var") > 0.0 for prop, _key in pairs)


class PLANTGEN_OT_leaf_variations(Operator):
    """An array of the active leaf: each copy takes the next Seed, so the leaf's
    per-parameter variances (its +/- buttons), Noise and Crumple make every one
    different. Copies are ordinary live-editable leaves -- generated blades, or
    mesh cutouts of the same outline -- sharing the material."""

    bl_idname = "plantgen.leaf_variations"
    bl_label = "Make Variations"
    bl_description = ("Make a grid of copies of the active leaf, each with the next "
                      "Seed -- its variances, Noise and Crumple make each one "
                      "different")
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(name="Count", default=8, min=1, soft_max=50, max=200,
                       description="How many variations to make")
    columns: IntProperty(name="Columns", default=4, min=1, soft_max=20,
                         description="Leaves per row of the grid")
    gap: FloatProperty(name="Gap", default=0.25, min=0.0, soft_max=2.0,
                       description="Space between leaves, as a fraction of the "
                                   "leaf's size")

    @classmethod
    def poll(cls, context):
        return leaf_ctl.is_editable_leaf(context.active_object)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "count")
        layout.prop(self, "columns")
        layout.prop(self, "gap")
        if not _varies_with_seed(context.active_object):
            col = layout.column(align=True)
            col.label(text="Nothing varies with Seed yet -- the copies", icon="ERROR")
            col.label(text="will match. Give some parameters variance", icon="BLANK1")
            col.label(text="(their +/- button) or add Noise / Crumple.",
                      icon="BLANK1")

    def execute(self, context):
        src = context.active_object
        g = src.plantgen_leaf_gen
        cutout = leaf_ctl.is_cutout_leaf(src)
        if cutout:
            # A cutout's size is its trace's; only the deformers vary.
            width, length = leaf_ctl.cutout_extent(src)
            step_x = width * (1.0 + self.gap)
            step_y = length * (1.0 + self.gap)
        else:
            material = leaf_mat.data_material(src)
            template = context.scene.plantgen_leaf
            leaf_ctl.copy_params(g, template)
            # Grid cells sized for the largest leaf the variances allow.
            step_x = g.gen_width * (1.0 + g.gen_width_var) * (1.0 + self.gap)
            step_y = g.gen_length * (1.0 + g.gen_length_var) * (1.0 + self.gap)
        made = []
        for i in range(self.count):
            try:
                if cutout:
                    obj = leaf_ctl.copy_cutout(context, src, g.gen_seed + i + 1)
                else:
                    with leaf_ctl.live_update_paused():
                        template.gen_seed = g.gen_seed + i + 1
                    obj, _w, _n = leaf_ctl.generate_leaf(context, material=material)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                break
            col, row = i % self.columns, i // self.columns
            obj.location = (src.location.x + (col + 1) * step_x,
                            src.location.y - row * step_y, src.location.z)
            obj.rotation_euler = src.rotation_euler
            obj.scale = src.scale
            made.append(obj)
        for o in context.selected_objects:
            o.select_set(False)
        for o in made + [src]:
            o.select_set(True)
        context.view_layer.objects.active = src
        if not _varies_with_seed(src):
            self.report({'WARNING'}, f"Made {len(made)} copies -- identical: nothing "
                                     "varies with Seed (add variance, Noise or "
                                     "Crumple)")
        else:
            self.report({'INFO'}, f"Made {len(made)} variations "
                                  f"(Seeds {g.gen_seed + 1}-{g.gen_seed + len(made)})")
        return {'FINISHED'}


class PLANTGEN_OT_leaf_generate(Operator):
    bl_idname = "plantgen.leaf_generate"
    bl_label = "Generate Leaf"
    bl_description = ("Build a new procedural leaf blade. If a generated leaf is "
                      "active, the new one copies its settings")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # New Leaf while editing one: seed the template from the active leaf so
        # the new blade starts identical, then diverge via live edits.
        active = context.active_object
        material = None
        if leaf_ctl.is_generated_leaf(active):
            leaf_ctl.copy_params(active.plantgen_leaf_gen,
                                 context.scene.plantgen_leaf)
            material = leaf_mat.data_material(active)   # the copy shares the look
        try:
            obj, warnings, nverts = leaf_ctl.generate_leaf(context, material=material)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        for warning in list(warnings)[:3]:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, f"Generated '{obj.name}' ({nverts} verts)")
        return {'FINISHED'}


class PLANTGEN_PT_leaf(Panel):
    bl_label = "Leaf Generator"
    bl_idname = "PLANTGEN_PT_leaf"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PlantGen"

    def draw(self, context):
        s = context.scene.plantgen_leaf
        layout = self.layout
        layout.prop(s, "mode", expand=True)
        obj = context.active_object
        if leaf_ctl.is_cutout_leaf(obj):
            self._draw_cutout_edit(layout, obj)
        if s.mode == "cutout":
            self._draw_cutout(layout, s)
        else:
            self._draw_generate(layout, s, context)

    def _draw_cutout_edit(self, layout, obj):
        """A selected mesh cutout: its density, seed and deformers (live) and
        its material -- the generated leaf's controls minus shape and size."""
        box = layout.box()
        box.label(text=f"Editing cutout: {obj.name}")
        p = obj.plantgen_leaf_gen
        col = box.column(align=True)
        self._pv(col, p, "gen_res_v", text="Density")
        col.prop(p, "gen_seed")
        box.operator("plantgen.leaf_variations", text="Make Variations",
                     icon="MOD_ARRAY")
        self._draw_deform(box, p)
        self._draw_material(box, obj)

    def _draw_cutout(self, layout, s):
        layout.template_ID(s, "image", new="image.new", open="image.open")
        if s.image is None:
            layout.label(text="Pick a leaf atlas image to cut", icon="INFO")
            return
        col = layout.column(align=True)
        col.prop(s, "threshold", slider=True)
        col.prop(s, "boundary_verts")
        col.prop(s, "pixels_per_unit")
        col.prop(s, "atlas_min_area")
        col = layout.column(align=True)
        col.prop(s, "base")
        col.prop(s, "trace")
        layout.prop(s, "create_material")
        layout.prop(s, "cut_quads")
        if s.cut_quads:
            layout.prop(s, "cut_density")
        layout.operator("plantgen.leaf_cut", **plant_icons.kw("PG:leaf"))

    def _draw_generate(self, layout, s, context):
        # The actions first, right under the mode switch: a new leaf (or a
        # copy / variations of the one being edited) and the presets -- then
        # the settings they use.
        obj = context.active_object
        editing = leaf_ctl.is_generated_leaf(obj)
        actions = layout.column(align=True)
        if editing:
            row = actions.row(align=True)
            row.operator("plantgen.leaf_generate", text="New Leaf (copy)",
                         icon="ADD")
            row.operator("plantgen.leaf_variations", text="Make Variations",
                         icon="MOD_ARRAY")
        else:
            actions.operator("plantgen.leaf_generate", text="Generate Leaf",
                             **plant_icons.kw("PG:leaf"))
        row = actions.row(align=True)
        row.operator("plantgen.leaf_save_preset", text="Save Preset",
                     icon="FILE_TICK")
        row.operator("plantgen.leaf_load_preset", text="Load Preset",
                     icon="FILE_FOLDER")
        if editing:
            box = layout.box()
            box.label(text=f"Editing: {obj.name}")
            box.label(text="Changes update the leaf live")
            self._draw_gen_fields(box, obj.plantgen_leaf_gen)
            self._draw_material(layout, obj)
        else:
            self._draw_gen_fields(layout, s)

    @staticmethod
    def _draw_material(layout, obj):
        """The leaf's material, in a fold: picker (shareable between leaves) +
        UV Checker toggle, then the shared leaf-shader editor (its preset and
        Textures & Settings, `ui_materials.draw_editor`)."""
        head, body = layout.panel("PLANTGEN_leaf_material")
        head.label(text="Material")
        if body is None:
            return
        g = obj.plantgen_leaf_gen
        row = body.row(align=True)
        if g.uv_checker:
            row.label(text="UV Checker on (material kept)")
        else:
            row.template_ID(obj, "active_material", new="plantgen.leaf_material_new")
        row.prop(g, "uv_checker", text="", icon="TEXTURE", toggle=True)
        mat = leaf_mat.data_material(obj)
        if mat is None:
            return
        from . import ui_materials
        ui_materials.draw_editor(body, mat)

    @staticmethod
    def _pv(layout, p, name, text=None):
        """Draw a param plus its variance +/- button (opens a popup; depressed
        when the parameter carries variance) -- the parametric add-on's row shape."""
        row = layout.row(align=True)
        if text is None:
            row.prop(p, name)
        else:
            row.prop(p, name, text=text)
        var_prop = f"{name}_var"
        if hasattr(p, var_prop):
            icon_id = _pg_icons.variance_icon_id() if _pg_icons is not None else 0
            kw = {"icon_value": icon_id} if icon_id else {"icon": "MOD_NOISE"}
            op = row.operator("plantgen.leaf_edit_variance", text="",
                              depress=bool(getattr(p, var_prop)), **kw)
            op.prop_name = var_prop
            op.label = f"{p.bl_rna.properties[name].name} Variance"

    def _draw_gen_fields(self, layout, p):
        pv = self._pv
        # its own label column: the dropdown's built-in one is a quarter of the
        # row and cut "Leaf Shape" to "Leaf Sh..."
        split = layout.split(factor=0.4)
        split.label(text="Leaf Shape")
        split.prop(p, "gen_shape", text="")
        # The profile curve IS the outline (base left -> tip right). The editor
        # timer keeps it showing this holder and applies edits live; draw() must
        # not create the carrier datablock, so it may be absent for a moment.
        # Collapsible (a layout panel keeps its open state per region).
        header, body = layout.panel("PLANTGEN_leaf_profile", default_closed=False)
        header.label(text="Profile Curve")
        if body is not None:
            mod = leaf_ctl.profile_carrier(create=False)
            if mod is not None:
                body.template_curveprofile(mod, "custom_profile")
            else:
                body.label(text="Preparing profile curve...", icon="TIME")
        compound = p.gen_compound != "simple"
        col = layout.column(align=True)
        pv(col, p, "gen_length", text="Leaflet Length" if compound else None)
        pv(col, p, "gen_width", text="Leaflet Width" if compound else None)
        col = layout.column(align=True)
        pv(col, p, "gen_res_v")
        pv(col, p, "gen_res_h")
        col.prop(p, "gen_seed")

        head, body = layout.panel("PLANTGEN_leaf_margin")
        head.label(text="Margin")
        if body is not None:
            body.prop(p, "gen_margin", text="")
            if p.gen_margin != "entire":
                col = body.column(align=True)
                pv(col, p, "gen_margin_teeth")
                pv(col, p, "gen_margin_depth")

        self._draw_deform(layout, p)
        self._draw_compound(layout, p)

    def _draw_compound(self, layout, p):
        """Compound leaf: the type, then only the controls that type reads."""
        pv = self._pv
        head, box = layout.panel("PLANTGEN_leaf_compound")
        head.label(text="Compound")
        if box is None:
            return
        box.prop(p, "gen_compound", text="")
        kind = p.gen_compound
        if kind == "simple":
            return
        col = box.column(align=True)
        if kind == "palmate":
            col.prop(p, "gen_leaflets")
        elif kind in ("pinnate", "bipinnate"):
            col.prop(p, "gen_leaflets", text="Pairs" if kind == "pinnate" else "Pinna Pairs")
            if kind == "bipinnate":
                col.prop(p, "gen_sub_pairs")
            col.prop(p, "gen_terminal")
        col = box.column(align=True)
        col.prop(p, "gen_petiole")
        if kind != "palmate":
            pv(col, p, "gen_rachis_length")
        if kind == "bipinnate":
            col.prop(p, "gen_pinna_length")
        pv(col, p, "gen_spread")
        pv(col, p, "gen_leaflet_taper")
        col = box.column(align=True)
        col.prop(p, "gen_petiolule")
        col.prop(p, "gen_rachis_width")
        col.prop(p, "gen_rachis_curl")

    def _draw_deform(self, layout, p):
        pv = self._pv
        head, box = layout.panel("PLANTGEN_leaf_deform")
        head.label(text="Deform")
        if box is None:
            return
        col = box.column(align=True)
        pv(col, p, "gen_curl")
        pv(col, p, "gen_cup")
        pv(col, p, "gen_twist")
        pv(col, p, "gen_gravity")
        pv(col, p, "gen_wave")
        if p.gen_wave != 0.0:
            pv(col, p, "gen_wave_count")
        pv(col, p, "gen_noise")
        if p.gen_noise != 0.0:
            pv(col, p, "gen_noise_scale")
        pv(col, p, "gen_crumple")
        if p.gen_crumple != 0.0:
            pv(col, p, "gen_crumple_scale")


_classes = (
    PLANTGEN_PG_leaf,
    PLANTGEN_PG_leaf_gen,
    PLANTGEN_OT_leaf_edit_variance,
    PLANTGEN_OT_leaf_save_preset,
    PLANTGEN_OT_leaf_load_preset,
    PLANTGEN_OT_leaf_material_new,
    PLANTGEN_OT_leaf_variations,
    PLANTGEN_OT_leaf_cut,
    PLANTGEN_OT_leaf_generate,
    PLANTGEN_PT_leaf,
)


# --- Profile-curve editor: poll timer + resets --------------------------------
# A CurveProfile edit fires no callback, so a short timer runs the editor sync
# (`leaf.editor_tick`). Undo/redo and file loads only RESET the sync state (the
# timer reloads the curve from the restored holder on its next tick) -- no
# datablock writes inside those handlers.
_TIMER_INTERVAL = 0.2
_ALIVE = [False]
_LAST_ERROR = [""]


def _editor_timer():
    if not _ALIVE[0]:
        return None                       # a timer left by a reloaded module stops
    try:
        leaf_ctl.editor_tick(bpy.context)
    except Exception:
        # A timer that raises is silently unregistered (landmine) -- never let it,
        # and print each distinct failure once rather than every tick.
        import traceback
        err = traceback.format_exc()
        if err != _LAST_ERROR[0]:
            _LAST_ERROR[0] = err
            print("plant_gen: leaf profile editor error:\n" + err)
    return _TIMER_INTERVAL


@persistent
def _plantgen_leaf_on_load(*_args):
    leaf_ctl.editor_reset(new_file=True)


@persistent
def _plantgen_leaf_on_undo(*_args):
    leaf_ctl.editor_reset()


_HANDLERS = (
    ("load_post", _plantgen_leaf_on_load),
    ("undo_post", _plantgen_leaf_on_undo),
    ("redo_post", _plantgen_leaf_on_undo),
)


def _remove_handlers():
    # By NAME, not identity: a reload replaces the function objects (landmine).
    for attr, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, attr)
        for h in list(lst):
            if getattr(h, "__name__", "") == fn.__name__:
                lst.remove(h)


def register():
    leaf_mat.register()
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.plantgen_leaf = PointerProperty(type=PLANTGEN_PG_leaf)
    bpy.types.Object.plantgen_leaf_gen = PointerProperty(type=PLANTGEN_PG_leaf_gen)
    _remove_handlers()
    for attr, fn in _HANDLERS:
        getattr(bpy.app.handlers, attr).append(fn)
    leaf_ctl.editor_reset(new_file=True)
    _ALIVE[0] = True
    if not bpy.app.timers.is_registered(_editor_timer):
        bpy.app.timers.register(_editor_timer, first_interval=_TIMER_INTERVAL,
                                persistent=True)


def unregister():
    _ALIVE[0] = False
    if bpy.app.timers.is_registered(_editor_timer):
        bpy.app.timers.unregister(_editor_timer)
    _remove_handlers()
    del bpy.types.Object.plantgen_leaf_gen
    del bpy.types.Scene.plantgen_leaf
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    leaf_mat.unregister()
