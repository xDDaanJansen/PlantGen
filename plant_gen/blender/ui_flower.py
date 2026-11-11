"""N-panel UI: the standalone FlowerGen "flower workshop" — edits the ACTIVE
flower's own parameters + curve widgets. Sibling of `ui.py`'s stem panel, not
nested under it: a flower is an independent object, not part of a stem.

Popover Panel + Shape Menu pairs are generated per `flower.FLOWER_CURVE_SPECS`
entry by `_make_flower_curve_classes` (same idea as `ui.py`'s
`_make_curve_classes`, now that there are several flower curves instead of 1 --
see that function's own docstring for why hand-duplicating stops being worth
it). Each curve's Shape Presets menu draws from whichever
`flower_ops.FLOWER_SHAPE_LIBRARIES` entry its spec's `library` column names
(botanical petal outlines for Petal Width Profile, disc-body shapes for Disc
Profile, the stem's original generic curves for the shape-only Curl/Gravity/
Twist/Age profiles). Each spec also carries its OWN X-axis caption, since the
`_age_profile` curves ramp over this petal's Age (0=young/top -> 1=mature/
bottom), not its position along the petal's length like the rest.

Petals/Sepals/Disc/Receptacle/Preset are each their own collapsible sub-panel
(native Blender `bl_parent_id` nesting, via `_FlowerSubpanel`); Layers, Shape
and Bud nest one level further, UNDER Petals (and, mirrored, under Sepals), and
Curl/Rotation nest one level further still, UNDER Shape (`bl_parent_id`
overridden directly on each subclass, same idiom `ui.py`'s own stem Curl panel
uses to nest under the stem's Shape panel) -- so their fields don't need a
redundant "Petal "/"Sepal " prefix, since the panel hierarchy already says which
component they belong to. Sibling sub-panels of the same parent render in
REGISTRATION order (see `_classes` below), which is how Curl ends up above
Rotation with no separate ordering code needed, and how Sepals/Receptacle end up
positioned after Petals/Disc respectively. Sepals and Receptacle both use the
native Blender modifier-panel checkbox-in-header idiom (`draw_header`), same as
the stem's own Leaves/Flowers panels (`ui.py`) -- both default off so adding
them doesn't change any existing flower's current look.

Not every "Bud X" value works the same way underneath (see `skeleton/flower.py`):
Bud Curl, Bud Pitch, Bud Sink, Bud Cup and Bud Clamp Overlap are all true
PAIRS -- an authored bud-value AND an authored open-value, lerped by this
whorl's own `local_bloom` (Bud Cup and Bud Clamp Overlap used to be fading
MULTIPLIERS onto the single open value instead, but that made an independent
bud pose harder to predict than just authoring it directly, so both were
converted to true pairs), while the whole Tip Shape system
(Tip Zone Start, Tip Pointiness, Tip Shape, Tip Falloff, Bloom Start/End) is a
bud-ONLY overlay with no open-pose counterpart at all. All of it still lives
together in one place per whorl -- `PLANTGEN_PT_flower_petal_bud`/
`..._sepal_bud`, labelled "Bud Shape" -- since it's all "what this whorl looks
like fully closed," labelled to match its open-pose counterpart by name where
one exists (Curl Up/Down, Rotation Pitch, Radial Offset, Cup, Clamp Overlap)
so the pairing is obvious without a link/toggle mechanism.
`PLANTGEN_PT_flower_bloom` itself is just the flower-wide master slider --
nothing else lives there.
"""

import bpy
from bpy.types import Menu, Panel

from ..growth.flower_params import Ramp
from . import flower, flower_ops, icons, plant_icons


def _draw_custom_marker(layout, active: bool) -> None:
    """First row of every "does this match a named preset" dropdown (Shape
    Presets, Position Presets, Corolla Type, Petal Shape) -- a non-interactive
    label, not a button, since Custom isn't a real settable state the way
    Inflorescence Type's own "None" is (there's no ground-truth field behind
    it to reset; it's just whatever combination of fields currently is). Makes
    the "nothing below matches" state visible IN the list itself instead of
    only inferred from the collapsed dropdown button's own fallback text --
    checked exactly when `active` (the caller already computed that no preset
    matches)."""
    layout.label(text="Custom", icon="CHECKMARK" if active else "BLANK1")


def _fields_close(a, b) -> bool:
    """Value equality for preset "is this one currently active?" matching.

    Handles `Ramp` by comparing authored CONTROL POINTS, not a dense
    evaluation: a preset writes its points straight into the curve widget, so a
    widget still holding that preset's shape holds those exact points, while
    any hand-edit changes or adds one. Comparing evaluated samples instead
    would call two visibly different curves equal whenever they happened to
    agree at the sample positions."""
    if isinstance(a, Ramp) or isinstance(b, Ramp):
        if not (isinstance(a, Ramp) and isinstance(b, Ramp)):
            return False
        if len(a.points) != len(b.points):
            return False
        return all(
            abs(ax - bx) < 1e-4 and abs(ay - by) < 1e-4
            for (ax, ay), (bx, by) in zip(a.points, b.points, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < 1e-4
    return a == b


def _live_preset_value(obj, s, curves, name: str):
    """The flower's CURRENT value for one field a preset dict names -- routing
    curve names to the curve widget's authored points and everything else to
    the PropertyGroup, exactly mirroring how `_apply_flower_preset_fields`
    WRITES them. Returns None when the curve group doesn't exist yet.

    Curve values come back in the preset's own units, undoing any `y_remap` on
    the way out, since the preset dict holds real (post-remap) units while the
    widget always stores a plain 0..1 edit range."""
    if name not in flower._FLOWER_CURVE_REMAPS:
        return getattr(s, name, None)
    if not curves:
        return None
    ramp = flower.raw_ramp(curves[name])
    y_remap = flower._FLOWER_CURVE_REMAPS[name]
    if y_remap is not None:
        lo, hi = y_remap
        ramp = Ramp(points=[(x, lo + y * (hi - lo)) for x, y in ramp.points])
    return ramp


def _matching_preset_label(obj, presets_fields) -> str | None:
    """Which preset (if any) describes the flower's live state, for showing its
    real name on the dropdown instead of the generic "Presets" -- the same idea
    as `ui._matching_flower_position_label`, extended to presets whose dicts
    also carry curve values (see `_fields_close`, which knows how to compare
    a `Ramp`). `presets_fields` is an iterable of `(label, fields_dict)`.

    Most fields win when several match, on the same reasoning as the stem-side
    version: the preset that pins more of the current state is the one that
    describes it. Any hand-edit to a matched field drops the label straight
    back to "Presets", which is the honest answer once the shape is no longer
    the named one."""
    s = obj.plantgen_flower
    curves = flower.get_flower_curves(obj)
    best: tuple[int, str] | None = None
    for label, fields in presets_fields:
        if all(
            _fields_close(_live_preset_value(obj, s, curves, name), value)
            for name, value in fields.items()
        ):
            if best is None or len(fields) > best[0]:
                best = (len(fields), label)
    return best[1] if best else None


def _matching_corolla_label(obj) -> str | None:
    return _matching_preset_label(obj, (
        (label, fields_fn(obj))
        for _key, label, _description, fields_fn in flower_ops.FLOWER_COROLLA_PRESETS))


def _matching_petal_shape_label(obj) -> str | None:
    return _matching_preset_label(obj, (
        (label, flower_ops._petal_shape_fields(key))
        for key, label in flower_ops.FLOWER_SHAPE_LIBRARIES_LIST["petal"]))


def _draw_flower_shape_menu_items(layout, curve_key: str, library: str) -> None:
    for shape, label in flower_ops.FLOWER_SHAPE_LIBRARIES_LIST[library]:
        op = layout.operator("plantgen.flower_curve_shape", text=label)
        op.curve_key = curve_key
        op.library = library
        op.shape = shape


def _make_flower_curve_classes(
    key: str, label: str, x_label: str, y_label: str, library: str,
) -> tuple[type, type]:
    """Build the (Panel, Menu) pair for one `flower.FLOWER_CURVE_SPECS` entry --
    same shape as `ui.py`'s `_make_curve_classes`, pointed at
    `flower.get_flower_curves`/`obj.plantgen_flower` instead."""
    panel_idname = f"PLANTGEN_PT_flower_{key}_popover"
    menu_idname = f"PLANTGEN_MT_flower_{key}_shapes"

    def panel_draw(self, context):
        curves = flower.get_flower_curves(context.active_object)
        if not curves:
            return
        row = self.layout.row()
        row.label(text=f"X: {x_label}")
        row.label(text=f"Y: {y_label}")
        self.layout.template_curve_mapping(curves[key], "mapping")
        self.layout.menu(menu_idname, text="Shape Presets")

    def menu_draw(self, context):
        _draw_flower_shape_menu_items(self.layout, key, library)

    # A popover only (opened by its curve button): the header region, as
    # Blender's own popovers -- a tabless sidebar panel lands in a "Misc" tab.
    panel_cls = type(panel_idname, (Panel,), {
        "bl_label": label, "bl_idname": panel_idname,
        "bl_space_type": 'VIEW_3D', "bl_region_type": 'HEADER', "bl_ui_units_x": 14,
        "draw": panel_draw,
    })
    menu_cls = type(menu_idname, (Menu,), {
        "bl_label": "Shape Presets", "bl_idname": menu_idname,
        "draw": menu_draw,
    })
    return panel_cls, menu_cls


_curve_panel_classes: list[type] = []
_curve_menu_classes: list[type] = []
_popover_idname_by_key: dict[str, str] = {}

for _key, _label, _x_label, _y_label, _defaults, _library, _y_remap in flower.FLOWER_CURVE_SPECS:
    _panel_cls, _menu_cls = _make_flower_curve_classes(_key, _label, _x_label, _y_label, _library)
    _curve_panel_classes.append(_panel_cls)
    _curve_menu_classes.append(_menu_cls)
    _popover_idname_by_key[_key] = _panel_cls.bl_idname


def _slider(
    layout, s, prop_name: str, text: str | None = None,
    curve_key: str | None = None, age_curve_key: str | None = None,
    no_variance: bool = False, value_active: bool = True,
) -> None:
    """Same compact slider + variance-popup + curve-popover pattern as
    `ui._slider_with_variance`, scoped to `obj.plantgen_flower`
    (`root="plantgen_flower"` on the shared `plantgen.edit_variance` operator).
    `curve_key` opens the along-length profile popover (if any); `age_curve_key`
    opens a SECOND popover for the trait's Age profile (if any) -- a petal can
    have both at once (e.g. Curl Up/Down). `no_variance=True` skips the variance
    button entirely for the handful of fields deliberately authored as plain
    scalars with no per-petal `_variance` counterpart (e.g. Bud Curl).

    `value_active=False` greys out just the VALUE field, leaving the row's
    curve popover live -- for a radius whose number is currently driven by
    something else (Match Disc Radius) while its Profile curve still very much
    applies. Greying or hiding the whole row would strand that curve with no
    way to open it."""
    row = layout.row(align=True)
    value = row.row(align=True)
    value.active = value_active
    value.prop(s, prop_name, text=text)
    if curve_key:
        row.popover(
            panel=_popover_idname_by_key[curve_key], text="",
            icon_value=icons.curve_icon_id(f"flower_{curve_key}"))
    if age_curve_key:
        row.popover(
            panel=_popover_idname_by_key[age_curve_key], text="",
            icon_value=icons.curve_icon_id(f"flower_{age_curve_key}"))
    if not no_variance:
        variance = getattr(s, f"{prop_name}_variance")
        op = row.operator(
            "plantgen.flower_edit_variance", text="", icon_value=icons.variance_icon_id(),
            depress=bool(variance))
        op.prop_name = prop_name
        op.label = (text or s.bl_rna.properties[prop_name].name) + " Variance"
        op.root = "plantgen_flower"


class PLANTGEN_PT_flower(Panel):
    bl_label = "Flower Generator"
    bl_idname = "PLANTGEN_PT_flower"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PlantGen"

    def draw(self, context):
        layout = self.layout
        layout.operator("plantgen.add_flower", **plant_icons.kw("PG:flower"))

        obj = context.active_object
        if not flower.is_flower(obj):
            layout.label(text="Select a flower to edit it", icon="INFO")
            return

        layout.label(text=f"Editing: {obj.name}")
        # The flower's preset, at the top (was its own sub-panel until 2026-10-14).
        row = layout.row(align=True)
        row.operator("plantgen.save_flower_preset", icon="EXPORT")
        row.operator("plantgen.load_flower_preset", icon="IMPORT")
        row = layout.row(align=True)
        row.operator(
            "plantgen.flower_add_to_pool", icon="LINKED", depress=flower.is_in_pool(obj))
        row.operator("plantgen.flower_bake_to_mesh", icon="MESH_DATA")
        s = obj.plantgen_flower

        col = layout.column(align=True)
        col.prop(s, "seed")


class _FlowerSubpanel(Panel):
    """Shared base for every collapsible section below the base flower fields --
    mirrors `ui._StemSubpanel`. Not registered itself, concrete subclasses only."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "PLANTGEN_PT_flower"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return flower.is_flower(context.active_object)


class PLANTGEN_PT_flower_bloom(_FlowerSubpanel):
    """Flower-wide Bloom master (bud <-> open) -- see skeleton/flower.py's
    `bloom` docstring. Registered first (before Petals) since it drives the
    whole flower, not one component -- just the slider now, nothing else.

    Everything this whorl looks like fully closed -- Curl Up/Down, Rotation
    Pitch and Radial Offset's bud-side values, the Curvature Multiplier, and
    the bud-only Tip Shape system -- lives in that whorl's own "Bud Shape"
    subpanel instead (see PLANTGEN_PT_flower_petal_bud), not scattered across
    a flower-wide panel plus each open-pose subpanel: "everything about this
    whorl lives under that whorl's own panel" as the one organizing rule, no
    exceptions. Bake Bloom Stages lives here too -- it's the other way (besides
    dragging this slider) to get variation across Bloom, see
    flower.bake_bloom_stages. Maturity -- the flower's global size over
    developmental time, a SEPARATE axis from Bloom (see growth/flower.py's
    `maturity` docstring) -- sits right under it (its own sub-panel until
    2026-10-14)."""
    bl_idname = "PLANTGEN_PT_flower_bloom"
    bl_label = "Bloom"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        layout = self.layout
        layout.prop(s, "bloom", slider=True)
        row = layout.row(align=True)
        row.prop(s, "maturity", slider=True)
        if flower.get_flower_curves(obj):
            row.popover(
                panel=_popover_idname_by_key["maturity_scale_profile"], text="",
                icon_value=icons.curve_icon_id("flower_maturity_scale_profile"))
        layout.separator()
        layout.prop(s, "bloom_stage_count", text="Stages")
        layout.prop(s, "bloom_stage_variance", text="Stages Variance", slider=True)
        layout.operator("plantgen.flower_bake_bloom_stages", icon="RENDER_RESULT")


class PLANTGEN_PT_flower_petals(_FlowerSubpanel):
    bl_idname = "PLANTGEN_PT_flower_petals"
    bl_label = "Petals"

    def draw_header(self, context):
        self.layout.prop(context.active_object.plantgen_flower, "petal_enabled", text="")

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        if not s.petal_enabled:
            self.layout.label(text="Disabled", icon="INFO")
            return
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        row = layout.row(align=True)
        row.label(text="Corolla Type")
        row.menu("PLANTGEN_MT_flower_corolla_presets",
                 text=_matching_corolla_label(obj) or "Custom")
        row = layout.row(align=True)
        row.label(text="Petal Shape")
        row.menu("PLANTGEN_MT_flower_petal_shape_presets",
                 text=_matching_petal_shape_label(obj) or "Custom")
        # Mesh first -- how finely each petal is built (with many small
        # flowers the first thing to set), and its winding.
        head, body = layout.panel("PLANTGEN_flower_petal_mesh", default_closed=True)
        head.label(text="Mesh")
        if body is not None:
            col = body.column(align=True)
            col.prop(s, "resolution")
            col.prop(s, "petal_width_resolution")
            body.prop(s, "petal_flip_normals")
        # Outline -- Width (the Sides, along the petal's LENGTH) and Top/Bottom
        # Shape (the cap edges, across its WIDTH) are the three curves that
        # together make the outline, so they sit on consecutive rows.
        head, body = layout.panel("PLANTGEN_flower_petal_outline")
        head.label(text="Outline")
        if body is not None:
            _slider(body, s, "petal_length")
            _slider(body, s, "petal_width", curve_key="petal_width_profile" if curves else None)
            _slider(body, s, "petal_top_shape",
                    curve_key="petal_top_shape_profile" if curves else None)
            _slider(body, s, "petal_bottom_shape",
                    curve_key="petal_bottom_shape_profile" if curves else None)
            _slider(body, s, "petal_cup")
            _slider(body, s, "petal_scale",
                    age_curve_key="petal_scale_age_profile" if curves else None)
        # Placement -- how many, and where they sit around the flower.
        head, body = layout.panel("PLANTGEN_flower_petal_placement")
        head.label(text="Placement")
        if body is not None:
            _slider(body, s, "petal_count")
            _slider(body, s, "petal_sink")
            _slider(body, s, "petal_vertical_offset")
            _slider(body, s, "petal_spread_horizontal")
            body.prop(s, "petal_clamp_overlap")
            # Around the Whorl: Length / Width / Pitch by a petal's place on
            # the ring, mirrored about the first petal -- a bilateral corolla.
            whead, wbody = body.panel("PLANTGEN_flower_petal_whorl", default_closed=True)
            whead.label(text="Around the Whorl")
            if wbody is not None and curves:
                col = wbody.column(align=True)
                for key, label in (("petal_whorl_length_profile", "Length"),
                                   ("petal_whorl_width_profile", "Width"),
                                   ("petal_whorl_pitch_profile", "Pitch")):
                    col.popover(panel=_popover_idname_by_key[key], text=label,
                                icon_value=icons.curve_icon_id(f"flower_{key}"))


class PLANTGEN_PT_flower_petal_layers(_FlowerSubpanel):
    """Nested under Petals (bl_parent_id override below), not the top flower
    panel -- see module docstring."""
    bl_idname = "PLANTGEN_PT_flower_petal_layers"
    bl_parent_id = "PLANTGEN_PT_flower_petals"
    bl_label = "Layers"

    def draw(self, context):
        s = context.active_object.plantgen_flower
        layout = self.layout
        layout.prop(s, "petal_layers")
        if s.petal_layers > 1:
            _slider(layout, s, "petal_layer_spacing")
            layout.prop(s, "petal_layer_divergence")


class PLANTGEN_PT_flower_petal_shape(_FlowerSubpanel):
    """Deform (was "Shape"; the outline is the Petals panel's Outline): how
    the petal is posed -- Gravity, Twist, then Curl and Rotation below.
    Nested under Petals (bl_parent_id override below)."""
    bl_idname = "PLANTGEN_PT_flower_petal_shape"
    bl_parent_id = "PLANTGEN_PT_flower_petals"
    bl_label = "Deform"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "petal_gravity", curve_key="petal_gravity_profile" if curves else None)
        _slider(layout, s, "petal_twist", curve_key="petal_twist_profile" if curves else None)


class PLANTGEN_PT_flower_petal_shape_curl(_FlowerSubpanel):
    """Nested under Shape (bl_parent_id override below) -- a real collapsible
    sub-panel, same idiom `ui.py`'s `PLANTGEN_PT_stem_shape_curl` uses to nest
    under the stem's own Shape panel. Registered BEFORE the Rotation sub-panel
    below (see `_classes`), so it renders above it."""
    bl_idname = "PLANTGEN_PT_flower_petal_shape_curl"
    bl_parent_id = "PLANTGEN_PT_flower_petal_shape"
    bl_label = "Curl"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "petal_curl_updown", text="Up/Down",
                curve_key="petal_curl_updown_profile" if curves else None,
                age_curve_key="petal_curl_updown_age_profile" if curves else None)
        _slider(layout, s, "petal_curl_leftright", text="Left/Right",
                curve_key="petal_curl_leftright_profile" if curves else None)
        _slider(layout, s, "petal_curl_outin", text="Out/In",
                curve_key="petal_curl_outin_profile" if curves else None)
        _slider(layout, s, "petal_curl_roll", text="Roll")


class PLANTGEN_PT_flower_petal_shape_rotation(_FlowerSubpanel):
    """Nested under Shape (bl_parent_id override below) -- a real collapsible
    sub-panel, same idiom as Curl above. Registered AFTER Curl (see
    `_classes`), so it renders below it."""
    bl_idname = "PLANTGEN_PT_flower_petal_shape_rotation"
    bl_parent_id = "PLANTGEN_PT_flower_petal_shape"
    bl_label = "Rotation"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "petal_rotation_yaw", text="Yaw")
        _slider(layout, s, "petal_rotation_pitch", text="Pitch",
                age_curve_key="petal_rotation_pitch_age_profile" if curves else None)
        _slider(layout, s, "petal_rotation_roll", text="Roll")


class PLANTGEN_PT_flower_petal_bud(_FlowerSubpanel):
    """Nested under Petals (bl_parent_id override below), sibling of Layers/
    Shape -- this whorl's whole closed-bud pose in one place, see the module
    docstring for the pair-vs-overlay breakdown. Curl Up/Down, Rotation Pitch,
    Radial Offset, Cup and Clamp Overlap's bud values are all labelled to
    match their open-pose counterpart by name (they're two ends of the same
    dial Bloom lerps between, see `_build_petal_group` in skeleton/flower.py)
    -- no link toggle needed, proximity + matching names already makes the
    pairing obvious. Only the Tip Shape system below the separator has no
    open-pose counterpart to pair with at all. Registered AFTER Shape
    (see `_classes`), so it renders below it."""
    bl_idname = "PLANTGEN_PT_flower_petal_bud"
    bl_parent_id = "PLANTGEN_PT_flower_petals"
    bl_label = "Bud Shape"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        layout.label(text="This whorl's pose at Bloom = 0 (closed).")
        layout.label(text="Blends to the open pose by Bloom = 1.")
        _slider(layout, s, "petal_bud_curl", text="Curl Up/Down", no_variance=True,
                curve_key="petal_bud_curl_profile" if curves else None)
        _slider(layout, s, "petal_bud_rotation_pitch", text="Rotation Pitch", no_variance=True)
        _slider(layout, s, "petal_bud_sink", text="Radial Offset", no_variance=True)
        _slider(layout, s, "petal_bud_cup", text="Cup", no_variance=True)
        _slider(layout, s, "petal_bud_clamp_overlap", text="Clamp Overlap", no_variance=True)
        layout.separator()
        row = layout.row(align=True)
        row.prop(s, "petal_bloom_start", text="Start")
        row.prop(s, "petal_bloom_end", text="End")
        layout.prop(s, "petal_bud_tip_start")
        _slider(layout, s, "petal_bud_tip_lift", text="Tip Pointiness", no_variance=True,
                curve_key="petal_bud_tip_height_profile" if curves else None)
        if curves:
            row = layout.row(align=True)
            row.label(text="Tip Shape")
            row.popover(
                panel=_popover_idname_by_key["petal_bud_tip_radius_profile"], text="",
                icon_value=icons.curve_icon_id("flower_petal_bud_tip_radius_profile"))
        layout.prop(s, "petal_bud_tip_falloff")


class PLANTGEN_PT_flower_petal_tube(_FlowerSubpanel):
    """The shared corolla tube the petal lobes emerge from (Sympetalous/
    Gamopetalous) -- a sub-panel of Petals since 2026-10-14 (the corolla IS
    the petal whorl), beside its Layers / Shape / Bud Shape. Its own Length /
    Radius / Resolution / Profile stay independent controls, never derived from
    the petal's Length / Width (the user asked to decouple those -- see
    growth/flower.py's petal_tube_enabled docstring). Same enable-checkbox-in-
    header idiom as Sepals / Receptacle."""
    bl_idname = "PLANTGEN_PT_flower_petal_tube"
    bl_parent_id = "PLANTGEN_PT_flower_petals"
    bl_label = "Corolla Tube"

    def draw_header(self, context):
        self.layout.prop(context.active_object.plantgen_flower, "petal_tube_enabled", text="")

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        if not s.petal_tube_enabled:
            self.layout.label(text="Disabled", icon="INFO")
            return
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "petal_tube_length")
        layout.prop(s, "petal_tube_match_disc")
        _slider(layout, s, "petal_tube_radius",
                curve_key="petal_tube_profile" if curves else None,
                value_active=not s.petal_tube_match_disc)
        layout.prop(s, "petal_tube_noise")
        col = layout.column(align=True)
        col.prop(s, "petal_tube_resolution")
        _sides_row(col, s, "petal_tube_sides",
                   s.petal_count * (s.petal_width_resolution - 1) if s.petal_tube_fuse
                   else None, "from the petals")
        col = layout.column(align=True)
        col.prop(s, "petal_bud_tube_length")
        col.prop(s, "petal_bud_tube_flare")
        layout.separator()
        layout.prop(s, "petal_tube_match_angle")
        layout.prop(s, "petal_tube_fuse")
        _flip_row(layout, s, "petal_tube_flip_normals",
                  "Petals" if s.petal_tube_fuse and s.petal_enabled else None)


class PLANTGEN_PT_flower_sepals(_FlowerSubpanel):
    """The calyx whorl -- a sibling of Petals (not nested under it), since it's
    an independent second whorl, not a sub-aspect of Petals. Uses the native
    Blender modifier-panel checkbox-in-header idiom (`draw_header`), same as the
    stem's own Leaves/Flowers panels (`ui.py`'s `PLANTGEN_PT_stem_leaves`) --
    shows "Disabled" and nothing else while `sepal_enabled` is off."""
    bl_idname = "PLANTGEN_PT_flower_sepals"
    bl_label = "Sepals"

    def draw_header(self, context):
        self.layout.prop(context.active_object.plantgen_flower, "sepal_enabled", text="")

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        if not s.sepal_enabled:
            self.layout.label(text="Disabled", icon="INFO")
            return
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        # Mesh / Outline / Placement -- as Petals'.
        head, body = layout.panel("PLANTGEN_flower_sepal_mesh", default_closed=True)
        head.label(text="Mesh")
        if body is not None:
            col = body.column(align=True)
            col.prop(s, "sepal_resolution")
            col.prop(s, "sepal_width_resolution")
            body.prop(s, "sepal_flip_normals")
        head, body = layout.panel("PLANTGEN_flower_sepal_outline")
        head.label(text="Outline")
        if body is not None:
            _slider(body, s, "sepal_length")
            _slider(body, s, "sepal_width", curve_key="sepal_width_profile" if curves else None)
            _slider(body, s, "sepal_top_shape",
                    curve_key="sepal_top_shape_profile" if curves else None)
            _slider(body, s, "sepal_bottom_shape",
                    curve_key="sepal_bottom_shape_profile" if curves else None)
            _slider(body, s, "sepal_cup")
            _slider(body, s, "sepal_scale",
                    age_curve_key="sepal_scale_age_profile" if curves else None)
        head, body = layout.panel("PLANTGEN_flower_sepal_placement")
        head.label(text="Placement")
        if body is not None:
            _slider(body, s, "sepal_count")
            _slider(body, s, "sepal_sink")
            _slider(body, s, "sepal_vertical_offset")
            body.prop(s, "sepal_azimuth_offset")
            body.prop(s, "sepal_z_offset")
            _slider(body, s, "sepal_spread_horizontal")
            body.prop(s, "sepal_clamp_overlap")


class PLANTGEN_PT_flower_sepal_layers(_FlowerSubpanel):
    """Nested under Sepals (bl_parent_id override below) -- mirrors
    PLANTGEN_PT_flower_petal_layers exactly."""
    bl_idname = "PLANTGEN_PT_flower_sepal_layers"
    bl_parent_id = "PLANTGEN_PT_flower_sepals"
    bl_label = "Layers"

    def draw(self, context):
        s = context.active_object.plantgen_flower
        layout = self.layout
        layout.prop(s, "sepal_layers")
        if s.sepal_layers > 1:
            _slider(layout, s, "sepal_layer_spacing")
            layout.prop(s, "sepal_layer_divergence")


class PLANTGEN_PT_flower_sepal_shape(_FlowerSubpanel):
    """Deform (was "Shape"), nested under Sepals -- mirrors
    PLANTGEN_PT_flower_petal_shape exactly."""
    bl_idname = "PLANTGEN_PT_flower_sepal_shape"
    bl_parent_id = "PLANTGEN_PT_flower_sepals"
    bl_label = "Deform"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "sepal_gravity", curve_key="sepal_gravity_profile" if curves else None)
        _slider(layout, s, "sepal_twist", curve_key="sepal_twist_profile" if curves else None)


class PLANTGEN_PT_flower_sepal_shape_curl(_FlowerSubpanel):
    """Nested under Sepal Shape -- mirrors PLANTGEN_PT_flower_petal_shape_curl
    exactly, including registering BEFORE Rotation below so Curl renders above it."""
    bl_idname = "PLANTGEN_PT_flower_sepal_shape_curl"
    bl_parent_id = "PLANTGEN_PT_flower_sepal_shape"
    bl_label = "Curl"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "sepal_curl_updown", text="Up/Down",
                curve_key="sepal_curl_updown_profile" if curves else None,
                age_curve_key="sepal_curl_updown_age_profile" if curves else None)
        _slider(layout, s, "sepal_curl_leftright", text="Left/Right",
                curve_key="sepal_curl_leftright_profile" if curves else None)
        _slider(layout, s, "sepal_curl_outin", text="Out/In",
                curve_key="sepal_curl_outin_profile" if curves else None)
        _slider(layout, s, "sepal_curl_roll", text="Roll")


class PLANTGEN_PT_flower_sepal_shape_rotation(_FlowerSubpanel):
    """Nested under Sepal Shape -- mirrors PLANTGEN_PT_flower_petal_shape_rotation
    exactly, registered AFTER Curl above so Rotation renders below it."""
    bl_idname = "PLANTGEN_PT_flower_sepal_shape_rotation"
    bl_parent_id = "PLANTGEN_PT_flower_sepal_shape"
    bl_label = "Rotation"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "sepal_rotation_yaw", text="Yaw")
        _slider(layout, s, "sepal_rotation_pitch", text="Pitch",
                age_curve_key="sepal_rotation_pitch_age_profile" if curves else None)
        _slider(layout, s, "sepal_rotation_roll", text="Roll")


class PLANTGEN_PT_flower_sepal_bud(_FlowerSubpanel):
    """Nested under Sepals (bl_parent_id override below) -- mirrors
    PLANTGEN_PT_flower_petal_bud exactly, for the calyx whorl."""
    bl_idname = "PLANTGEN_PT_flower_sepal_bud"
    bl_parent_id = "PLANTGEN_PT_flower_sepals"
    bl_label = "Bud Shape"

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        layout.label(text="This whorl's pose at Bloom = 0 (closed).")
        layout.label(text="Blends to the open pose by Bloom = 1.")
        _slider(layout, s, "sepal_bud_curl", text="Curl Up/Down", no_variance=True,
                curve_key="sepal_bud_curl_profile" if curves else None)
        _slider(layout, s, "sepal_bud_rotation_pitch", text="Rotation Pitch", no_variance=True)
        _slider(layout, s, "sepal_bud_sink", text="Radial Offset", no_variance=True)
        _slider(layout, s, "sepal_bud_cup", text="Cup", no_variance=True)
        _slider(layout, s, "sepal_bud_clamp_overlap", text="Clamp Overlap", no_variance=True)
        layout.separator()
        row = layout.row(align=True)
        row.prop(s, "sepal_bloom_start", text="Start")
        row.prop(s, "sepal_bloom_end", text="End")
        layout.prop(s, "sepal_bud_tip_start")
        _slider(layout, s, "sepal_bud_tip_lift", text="Tip Pointiness", no_variance=True,
                curve_key="sepal_bud_tip_height_profile" if curves else None)
        if curves:
            row = layout.row(align=True)
            row.label(text="Tip Shape")
            row.popover(
                panel=_popover_idname_by_key["sepal_bud_tip_radius_profile"], text="",
                icon_value=icons.curve_icon_id("flower_sepal_bud_tip_radius_profile"))
        layout.prop(s, "sepal_bud_tip_falloff")


class PLANTGEN_PT_flower_sepal_tube(_FlowerSubpanel):
    """The shared calyx tube the sepal lobes emerge from (Gamosepalous) --
    mirrors PLANTGEN_PT_flower_petal_tube exactly, for the calyx whorl: a
    sub-panel of Sepals."""
    bl_idname = "PLANTGEN_PT_flower_sepal_tube"
    bl_parent_id = "PLANTGEN_PT_flower_sepals"
    bl_label = "Calyx Tube"

    def draw_header(self, context):
        self.layout.prop(context.active_object.plantgen_flower, "sepal_tube_enabled", text="")

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        if not s.sepal_tube_enabled:
            self.layout.label(text="Disabled", icon="INFO")
            return
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "sepal_tube_length")
        layout.prop(s, "sepal_tube_match_disc")
        _slider(layout, s, "sepal_tube_radius",
                curve_key="sepal_tube_profile" if curves else None,
                value_active=not s.sepal_tube_match_disc)
        layout.prop(s, "sepal_tube_noise")
        col = layout.column(align=True)
        col.prop(s, "sepal_tube_resolution")
        _sides_row(col, s, "sepal_tube_sides",
                   s.sepal_count * (s.sepal_width_resolution - 1) if s.sepal_tube_fuse
                   else None, "from the sepals")
        col = layout.column(align=True)
        col.prop(s, "sepal_bud_tube_length")
        col.prop(s, "sepal_bud_tube_flare")
        layout.separator()
        layout.prop(s, "sepal_tube_match_angle")
        layout.prop(s, "sepal_tube_fuse")
        _flip_row(layout, s, "sepal_tube_flip_normals",
                  "Sepals" if s.sepal_tube_fuse and s.sepal_enabled else None)


class PLANTGEN_PT_flower_disc(_FlowerSubpanel):
    bl_idname = "PLANTGEN_PT_flower_disc"
    bl_label = "Disc"

    def draw_header(self, context):
        self.layout.prop(context.active_object.plantgen_flower, "disc_enabled", text="")

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        if not s.disc_enabled:
            self.layout.label(text="Disabled", icon="INFO")
            return
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        _slider(layout, s, "disc_radius", curve_key="disc_profile" if curves else None)
        _slider(layout, s, "disc_height")
        col = layout.column(align=True)
        col.prop(s, "disc_resolution", text="Resolution")
        col.prop(s, "disc_sides")
        layout.prop(s, "disc_flip_normals")


def _flip_row(layout, s, prop: str, follows) -> None:
    """A Flip Normals toggle; a part welded into another (`follows`, that
    one's name) is flipped with it, so its own toggle is drawn inactive."""
    if follows is None:
        layout.prop(s, prop)
        return
    row = layout.row(align=True)
    row.active = False
    row.prop(s, prop, text=f"Flip Normals (welded: follows {follows})")


def _sides_row(layout, s, prop: str, derived, why: str) -> None:
    """A Sides field; when a weld sets the count instead (`derived`), it is
    drawn inactive with the count actually used."""
    if derived is None:
        layout.prop(s, prop)
        return
    row = layout.row(align=True)
    row.active = False
    row.prop(s, prop, text=f"Sides ({derived}, {why})")


class PLANTGEN_PT_flower_receptacle(_FlowerSubpanel):
    """The receptacle -- a sibling near Disc, bottom-most (see module docstring):
    the base the whole flower sits on, tapering down toward where a stem would
    attach. Same enable-checkbox-in-header idiom as Sepals above."""
    bl_idname = "PLANTGEN_PT_flower_receptacle"
    bl_label = "Receptacle"

    def draw_header(self, context):
        self.layout.prop(context.active_object.plantgen_flower, "receptacle_enabled", text="")

    def draw(self, context):
        obj = context.active_object
        s = obj.plantgen_flower
        if not s.receptacle_enabled:
            self.layout.label(text="Disabled", icon="INFO")
            return
        curves = flower.get_flower_curves(obj)
        layout = self.layout
        layout.prop(s, "receptacle_match_disc")
        _slider(layout, s, "receptacle_radius",
                curve_key="receptacle_profile" if curves else None,
                value_active=not s.receptacle_match_disc)
        _slider(layout, s, "receptacle_height")
        col = layout.column(align=True)
        col.prop(s, "receptacle_resolution", text="Resolution")
        welded = (s.receptacle_match_disc and s.petal_tube_enabled
                  and s.petal_tube_match_disc)
        tube_sides = (s.petal_count * (s.petal_width_resolution - 1) if s.petal_tube_fuse
                      else s.petal_tube_sides)
        _sides_row(col, s, "receptacle_sides", tube_sides if welded else None,
                   "welded to the tube")
        tube_fused = s.petal_tube_fuse and s.petal_enabled
        _flip_row(layout, s, "receptacle_flip_normals",
                  ("Petals" if tube_fused else "Corolla Tube") if welded else None)


class PLANTGEN_MT_flower_corolla_presets(Menu):
    """'Presets' dropdown in the Petals box's Corolla Type row -- one click sets
    a whole named botanical corolla form (Cruciform/Tubular/etc, see
    `flower_ops.FLOWER_COROLLA_PRESETS`) at once, instead of hand-tuning Count/
    Fusion/Width or Fusion Profile separately. Mirrors `ui.py`'s
    `PLANTGEN_MT_flower_position_presets` flat-list style, just for a flower's
    own SHAPE (`obj.plantgen_flower`) instead of its placement on a stem --
    lives here rather than in `ui.py` for that reason. Label-matching
    (`_matching_corolla_label`) mirrors the position presets' own dropdown
    button, same "most-specific match wins" rule."""
    bl_idname = "PLANTGEN_MT_flower_corolla_presets"
    bl_label = "Presets"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        _draw_custom_marker(layout, _matching_corolla_label(obj) is None)
        layout.separator()
        for key, label, _description, _fields_fn in flower_ops.FLOWER_COROLLA_PRESETS:
            op = layout.operator("plantgen.flower_corolla_preset", text=label)
            op.preset = key


class PLANTGEN_MT_flower_petal_shape_presets(Menu):
    """'Presets' dropdown in the Petals box's Petal Shape row -- one click sets
    a named botanical outline (Ovate/Obovate/Spatulate/etc, see
    `flower_ops.FLOWER_PETAL_SHAPES`), meaning BOTH the Width Profile that
    shapes the sides and the Top Shape cap that finishes the tip.

    A top-level row rather than a submenu inside the Width curve's own popover
    (where the width-only version of this list still lives): now that these
    write more than that one curve, hiding them behind it would read as the
    Width popover silently clobbering unrelated settings."""
    bl_idname = "PLANTGEN_MT_flower_petal_shape_presets"
    bl_label = "Presets"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        _draw_custom_marker(layout, _matching_petal_shape_label(obj) is None)
        layout.separator()
        for key, label in flower_ops.FLOWER_SHAPE_LIBRARIES_LIST["petal"]:
            op = layout.operator("plantgen.flower_petal_shape_preset", text=label)
            op.preset = key


class PLANTGEN_PT_flower_material(_FlowerSubpanel):
    """Each part's material (the leaf shader, reused) and the UV checker --
    see `blender/flower_material.py`."""
    bl_idname = "PLANTGEN_PT_flower_material"
    bl_label = "Material"

    def draw(self, context):
        from . import flower_material
        flower_material.draw_materials(self.layout, context.active_object)


_classes = (
    *_curve_menu_classes, *_curve_panel_classes,
    PLANTGEN_PT_flower,
    PLANTGEN_PT_flower_bloom,
    PLANTGEN_PT_flower_petals,
    PLANTGEN_PT_flower_petal_layers,
    PLANTGEN_PT_flower_petal_shape,
    PLANTGEN_PT_flower_petal_shape_curl,
    PLANTGEN_PT_flower_petal_shape_rotation,
    PLANTGEN_PT_flower_petal_bud,
    PLANTGEN_PT_flower_petal_tube,
    PLANTGEN_PT_flower_sepals,
    PLANTGEN_PT_flower_sepal_layers,
    PLANTGEN_PT_flower_sepal_shape,
    PLANTGEN_PT_flower_sepal_shape_curl,
    PLANTGEN_PT_flower_sepal_shape_rotation,
    PLANTGEN_PT_flower_sepal_bud,
    PLANTGEN_PT_flower_sepal_tube,
    PLANTGEN_PT_flower_disc,
    PLANTGEN_PT_flower_receptacle,
    PLANTGEN_MT_flower_corolla_presets,
    PLANTGEN_MT_flower_petal_shape_presets,
    PLANTGEN_PT_flower_material,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
