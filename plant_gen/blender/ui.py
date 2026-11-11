"""The panels, organised **component-primary**: the top-level panels are the
parts of a plant (Plant / Stem / Components / Environment /
Tooling -- leaves, bracts, flowers, buds and fruit are all Components now, each
drawing its own settings), and the botanical *stages* (morphology / development /
environment / physical form) are sub-headers inside them rather than top-level
tabs. This matches the nearest analog -- SpeedTree is component-primary too --
and it gives almost every parameter one clear home, because the params that
looked "shared" (phyllotaxis, internode, radius, branching, gravity) are really
**Stem/axis** properties that leaves, buds and flowers inherit their node
positions from.

The placement rule (the tiebreaker for every future param): a parameter lives
with the **component it shapes** -- unless it is (a) a whole-plant response
sourced from outside the plant (**Environment**). (Physical Form is retired:
stiffness is each level's, weights each component's, and the stiffness preset,
curve and variance sit in Stem > Growth.)

The three "apical" params are deliberately three different things in three
homes: apical **control** (order modulation of rate/internode/thickness) ->
Stem/Shape; apical **dominance** (the age curve) -> Plant/Development; bud
**suppression** distance -> Stem/Bud Activation.

Each component is its own top-level panel sharing the parametric add-on's "PlantGen" tab
rather than a subpanel of one 'Plant Generator' header: a `bl_parent_id` is a hard
registration dependency, so this approach's panels would fail to register at all
without the other's. Borrowing a widget (see `curves.pg_icons`) is cheap and
degrades to nothing; borrowing a parent panel is neither.
"""

from __future__ import annotations

import bpy
from bpy.types import Menu, Panel, UIList

from ..growth import masks, morphology
from . import (
    bridge,
    classify_overlay,
    curves,
    effectors,
    instancing,
    mapping,
    operators,
    overlay,
    petioles,
    plant_icons,
    presets,
    properties,
    rig,
    showcase,
    stem_material,
    structure,
)
from . import mesh as gn_mesh

# The two sides a binary channel selects, for a clear label in place of min/max.
_BINARY_SIDES = {"on_branch": ("Branches", "Main stem"),
                 "apex": ("Tip buds", "Side buds"),
                 "inflorescence": ("Inflorescence", "Leafy stem")}


def _plant(context):
    """The plant these panels edit, or None. Resolves any selected component (rig,
    skin, organs, wind-debug) back to its plant, so you can hide the static plant
    to watch the animation and still edit it by selecting the rig/skin."""
    return bridge.resolve_plant(context)


def _curve_button(layout, key: str) -> None:
    """Same click-to-expand popover the parametric add-on puts beside a driven slider, with
    the same live thumbnail of the curve's own shape as its icon."""
    panel_id = f"PLANTGEN_PT_{key}_popover"
    icon_id = curves.icon_id(key)
    if icon_id:
        layout.popover(panel=panel_id, text="", icon_value=icon_id)
    else:
        layout.popover(panel=panel_id, text="", icon="FCURVE")


def _variance_button(layout, s, variance_prop: str, label: str, **_kw) -> None:
    """Retired: a +/- value is a Random Range map now, shown in the Mapping
    panel and reached by the parameter's map icon.
    Kept as a no-op so every old call site simply draws nothing."""

def _component_curve_button(layout, uid: int, name: str = "size") -> None:
    """The curve popover beside a component's control -- the same click-to-expand
    widget as the plant curves, with the same live thumbnail of the curve's own
    shape as its icon. Points at the active component's `name` curve. Size uses
    the standalone popover; every flower curve has a generated one
    (`PLANTGEN_PT_c_<name>_curve`)."""
    panel = ("PLANTGEN_PT_component_size_curve" if name == "size"
             else f"PLANTGEN_PT_c_{name}_curve")
    icon_id = curves.component_icon_id(uid, name)
    if icon_id:
        layout.popover(panel=panel, text="", icon_value=icon_id)
    else:
        layout.popover(panel=panel, text="", icon="FCURVE")


def _bloom_window_hint(layout, s) -> None:
    """Where the Day sits in the calendar bloom window (From Age): before it
    every flower is still a bud, past it every flower is open -- why a plant
    can show a single bloom stage. Nothing without a bloom calendar."""
    if s.last_bloom_window < 0.0 or not s.bloom_enabled:
        return
    lo = float(s.emergence_day + s.bloom_start)
    hi = lo + float(s.bloom_days)
    span = f"days {lo:.0f}-{hi:.0f}"
    if s.day < lo:
        text, icon = f"Day {s.day:.0f} is before the bloom window ({span}): all buds", 'INFO'
    elif s.day >= hi:
        text, icon = f"Day {s.day:.0f} is past the bloom window ({span}): all open", 'INFO'
    else:
        through = 100.0 * (s.day - lo) / max(hi - lo, 1e-9)
        text, icon = f"Day {s.day:.0f}: {through:.0f}% into the bloom window ({span})", 'PG:bloom'
    layout.label(text=text, **plant_icons.kw(icon))


def _debug_button(layout, s, mode: str) -> None:
    """The small eye that jumps straight to the debug view explaining the
    parameter it sits beside. The parametric add-on's pattern: a debug view is far more
    useful attached to the control it describes than buried in a list of
    modes elsewhere in the panel. A Debug Tool: absent unless that
    preference is on."""
    if not operators.debug_tools():
        return
    op = layout.operator("plantgen.toggle_debug_view", text="",
                         icon="HIDE_OFF" if s.debug_view_mode == mode else "HIDE_ON",
                         depress=(s.debug_view_mode == mode))
    op.mode = mode


def _field(layout, s, prop: str, *, text: str | None = None,
           curve: str | None = None, variance: str | None = None,
           debug: str | None = None, slider: bool = False,
           component: bool = False, uid: int = 0) -> None:
    """One control, plus small buttons for whatever else belongs to it: its
    curve along the axis, its variance, and the debug view that shows what it
    does. All are click-to-expand rather than inline, so a parameter with all
    of them costs one row instead of four.

    `component=True` sources the row (and its curve/variance) from a flower
    component `s` with uid `uid` rather than the plant, so the same field code
    draws the inflorescence on either panel (the flower port). Debug views are
    plant-level and dropped in the component context.
    """
    row = layout.row(align=True)
    if text is None:
        row.prop(s, prop, slider=slider)
    else:
        row.prop(s, prop, text=text, slider=slider)
    if curve:
        if component:
            _component_curve_button(row, uid, curve)
        else:
            _curve_button(row, curve)
    if variance:
        _variance_button(row, s, variance,
                         (text or s.bl_rna.properties[prop].name) + " Variance",
                         component=component)
    if debug and not component:
        _debug_button(row, s, debug)


def _gate_box(layout, s, prop: str, text: str, icon: str = 'NONE'):
    """A component box whose header carries its own enable-toggle.

    Greyed rather than hidden when off (`active = False`), so the settings stay
    readable and keep their values -- the same call every conditional box in
    this panel already makes.
    """
    box = layout.box()
    head = box.row(align=True)
    head.prop(s, prop, text="")
    head.label(text=text, icon=icon)
    box.active = getattr(s, prop)
    return box


# The inflorescence-kind capability facts live in `morphology` -- the single
# source the model properties, this panel, and the generated spec all read, so
# the panel can never claim a control is inert when the model still honours it.
# The panel enum is the uppercase of the model's own kind string (see
# `bridge._UPPER`), so `_kind` lowercases it to read those sets directly.

def _kind(s) -> str:
    return s.inflorescence_kind.lower()


def _relayed(s) -> bool:
    return _kind(s) in morphology.RELAY_KINDS


def _capped(s) -> bool:
    return _kind(s) in morphology.CAPPED_KINDS


def _sessile(s) -> bool:
    """No Pedicel box and no Flower Pitch: the flower's facing is not the user's
    to set, because it is capped (one terminal flower, no pedicel) or a sessile
    spike. The inverse of `morphology.kind_has_pedicels`; see
    `morphology.CAPPED_KINDS` / `SESSILE_KINDS` for the measured reasoning."""
    return not morphology.kind_has_pedicels(_kind(s))


def _terminal_cluster(s) -> bool:
    return _kind(s) in morphology.TERMINAL_CLUSTER_KINDS


def _flat_topped(s) -> bool:
    return _kind(s) in morphology.FLAT_TOPPED_KINDS


def _gradient(s, key: str) -> str | None:
    """A flowering gradient's curve key, or None where the curve cannot be read.

    A terminal cluster has no along-inflorescence position -- its flowers all sit
    at one point -- so its gradient curves are undefined, not merely ineffective;
    withholding the button is the panel catching up with what the model already
    does (`morphology.kind_gradients_apply`). Pedicel Width is the exception the
    caller relies on: it runs along a single stalk, so it is passed as a plain
    key rather than through here."""
    return key if morphology.kind_gradients_apply(_kind(s)) else None


def _make_curve_panel(key: str, label: str, x_label: str, y_label: str):
    """One popover Panel and one shape Menu per curve, generated rather than
    hand-written -- eight curves would otherwise be sixteen near-identical
    classes, and the count only goes up. Structure, idname pattern, width and
    menu contents all follow the parametric add-on's `ui._make_curve_classes`.

    The X caption is per-curve, not fixed: a shape curve is read along an axis
    and a development curve over the plant's life, and captioning the second
    like the first would invite exactly the wrong edit."""
    panel_id = f"PLANTGEN_PT_{key}_popover"
    menu_id = f"PLANTGEN_MT_{key}_shapes"

    def draw_panel(self, context):
        layout = self.layout
        plant = _plant(context)
        node = curves.get_curves(plant).get(key) if plant else None
        if node is None:
            # This plant predates the curve. Say so -- an empty popover reads
            # as a broken widget, which is exactly how the all-or-nothing
            # version of `get_curves` was reported.
            layout.label(text="Not on this plant yet.", icon='INFO')
            layout.label(text="Press Regrow to add it.")
            return
        row = layout.row()
        row.label(text=f"X: {x_label}")
        row.label(text=f"Y: {y_label}")
        layout.template_curve_mapping(node, "mapping")
        layout.menu(menu_id, text=curves.matching_shape(node) or "Custom")

    def draw_menu(self, context):
        layout = self.layout
        plant = _plant(context)
        node = curves.get_curves(plant).get(key) if plant else None
        if node is not None:
            # A non-interactive label, not a button: "Custom" is not a settable
            # state, just the absence of a match. Shown in the list so the
            # "nothing below matches" case is visible rather than inferred.
            matched = curves.matching_shape(node)
            layout.label(text="Custom", icon="CHECKMARK" if matched is None else "BLANK1")
            layout.separator()
        for ident, shape_label, _points, _smooth in curves.SHAPES:
            op = layout.operator("plantgen.curve_shape", text=shape_label)
            op.curve_key = key
            op.shape = ident

    # A popover only (opened by its curve button): the header region, as
    # Blender's own popovers -- a tabless sidebar panel lands in a "Misc" tab.
    panel = type(panel_id, (Panel,), {
        "bl_label": label, "bl_idname": panel_id,
        "bl_space_type": 'VIEW_3D', "bl_region_type": 'HEADER', "bl_ui_units_x": 14,
        "draw": draw_panel,
    })
    menu = type(menu_id, (Menu,), {
        "bl_label": "Shape Presets", "bl_idname": menu_id, "draw": draw_menu,
    })
    return panel, menu


_curve_classes: list[type] = []
for _spec in curves.CURVE_SPECS:
    _curve_classes.extend(_make_curve_panel(_spec.key, _spec.label,
                                            _spec.x_label, _spec.y_label))


class _ComponentPanel(Panel):
    """A top-level component panel: it shows in the PlantGen tab only when a
    plant is the active object. The taxonomy is component-primary, so each part
    of the plant is its own top-level panel rather than a sub-panel nested under
    one 'Plant Generator' header -- collapse Flowers without collapsing Stem."""

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PlantGen"

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None


# ============================================================================
# PLANT -- the global identity and the age timeline
# ============================================================================


class PLANTGEN_MT_preset_library(Menu):
    """The species preset Library -- bundled presets first, then the user's own,
    then Load-from-file for a preset that lives outside either folder. Each entry
    applies onto the active plant. Scanning the folders happens here, on open,
    not in the panel draw -- so the panel stays free of disk I/O."""

    bl_idname = "PLANTGEN_MT_preset_library"
    bl_label = "Species Presets"

    def draw(self, context):
        layout = self.layout
        entries = presets.list_presets()
        bundled = [e for e in entries if e["source"] == "bundled"]
        user = [e for e in entries if e["source"] == "user"]

        if bundled:
            layout.label(text="Bundled species")
            for entry in bundled:
                op = layout.operator("plantgen.apply_preset", text=entry["name"])
                op.filepath = entry["path"]
        if user:
            layout.separator()
            layout.label(text="My presets")
            for entry in user:
                op = layout.operator("plantgen.apply_preset", text=entry["name"])
                op.filepath = entry["path"]
        layout.separator()
        layout.operator("plantgen.load_preset", text="Load from File...",
                        icon="FILEBROWSER")


_REGROW_STATUS = {          # bridge.regrow_state -> the status line under Regrow
    'QUEUED': "Regrowing when the edit settles...",
    'GROWING': "Growing in the background...",
    'SETTLING': "Settling leaves in the background...",
}

_WELD_STATUS = {
    'UNWELDED': ("Stems not welded", 'ERROR'),
    'WELDING': ("Welding in the background...", 'SORTTIME'),
    'WELDED': ("Stems welded", 'CHECKMARK'),
}


def _draw_weld_status(layout, state: str) -> None:
    """Under the Weld button (Stem > Surface): whether the stems are welded -- every edit shows
    them unwelded until Weld is pressed (`bridge.weld_state`) -- and that an
    edit after welding needs a new weld. Nothing with Weld Junctions off."""
    if state not in _WELD_STATUS:
        return
    text, icon = _WELD_STATUS[state]
    col = layout.column(align=True)
    col.label(text=text, icon=icon)
    col.label(text="Changing parameters after welding", icon="INFO")
    col.label(text="will require rebuilding the welds.", icon="BLANK1")


class PLANTGEN_PT_plant(Panel):
    """The root panel: it alone shows without a plant (so Add Plant is always
    reachable) -- Add Plant, Setup, docs, Regrow + Live, the preset picker, the
    Master Seed, and the plant's time: Day, the Phenology Calendar, Lifespan,
    the Growth Timelapse. Variations below. (The rig, wind and export are the
    Rig & Export panel, near the end of the tab.)"""

    bl_label = "Plant Generator"
    bl_idname = "PLANTGEN_PT_plant"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PlantGen"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        obj = _plant(context)
        # A small header row: Add Plant and the docs button.
        row = layout.row(align=True)
        row.operator("plantgen.add_plant", **plant_icons.kw(_PLANT_ICON))
        row.operator("plantgen.open_docs", text="", icon="QUESTION")

        if obj is None:
            layout.label(text="Select a plant to edit it", icon="INFO")
            return

        s = obj.plantgen
        layout.label(text=f"Editing: {obj.name}")
        row = layout.row(align=True)
        busy = bridge.regrow_state(obj)
        row.operator("plantgen.regrow", icon="FILE_REFRESH",
                     text="Regrowing..." if busy else "Regrow")
        row.prop(s, "live_update", text="", icon="UV_SYNC_SELECT", toggle=True)
        if busy:
            layout.label(text=_REGROW_STATUS[busy], icon='SORTTIME')
        # Which individual of the species this is; every level's and
        # component's own Seed is mixed with it.
        layout.prop(s, "seed", text="Master Seed")

        # Preset library -- the plant's base. Replaces the retired species
        # dropdown: pick a bundled species or your own saved preset, save the
        # current plant, or reload its preset's library version. The menu button
        # shows the current preset name, read straight off the object (no disk
        # read).
        row = layout.row(align=True)
        row.menu("PLANTGEN_MT_preset_library",
                 text=s.preset_name or "Choose a preset...", icon="PRESET")
        row.operator("plantgen.save_preset", text="", icon="FILE_TICK")
        if s.preset_id:
            row.operator("plantgen.reload_preset", text="", icon="FILE_REFRESH")

        # The plant's time -- the real-days calendar. Day drives growth; the
        # plant emerges on Emergence Day and matures over Season Length (both on
        # the calendar widget, the only surface for the within-year windows).
        # Day is greyed while the timelapse drives it from the timeline frame.
        # Lifespan is an across-years trait, so it stays a field. (No Flowering
        # Onset curve since 2026-10-10, user decision: the calendar's bloom
        # window says when the plant flowers; the curve stays in the model as
        # the gate only where no bloom window is set.)
        time = _MapLayout(layout.column(align=True), s)
        day_row = time.row(align=True)
        day_row.active = not s.animate_day
        _field(day_row, s, "day", debug="birth_step")
        time.operator("plantgen.toggle_calendar", text="Open Phenology Calendar",
                      **plant_icons.kw("PG:calendar"))
        time.prop(s, "lifespan")

        # Growth Timelapse: the real plant regrown every frame (exact, slow:
        # for renders).
        head, body = layout.panel("PLANTGEN_plant_timelapse", default_closed=True)
        head.label(text="Growth Timelapse")
        if body is not None:
            scene = context.scene
            _draw_timelapse(body, s, scene.frame_start, scene.frame_end)

        # Variations: rows / grids of copies of this plant, and baking them.
        head, body = layout.panel("PLANTGEN_plant_variations", default_closed=True)
        head.label(text="Variations")
        if body is not None:
            _draw_variations(body, obj)


def _draw_timelapse(box, s, f0: int, f1: int) -> None:
    """Growth Timelapse: the real plant regrown on every frame over a Day span."""
    box.prop(s, "animate_day")
    span = box.column(align=True)
    span.active = s.animate_day
    span.prop(s, "timelapse_start_day")
    span.prop(s, "timelapse_end_day")
    info = box.column(align=True)
    info.active = s.animate_day
    info.label(text=f"Frames {f0}-{f1} play Day {int(s.timelapse_start_day)}"
               f" -> {int(s.timelapse_end_day)}")
    if s.animate_day:
        # Timelapse Cache: each frame regrown once is recorded and played back
        # from then on; Bake fills the range in one go.
        from . import timelapse_cache
        done, total = timelapse_cache.stats(s.id_data, bpy.context.scene)
        cache = box.column(align=True)
        cache.label(text=f"Playback cache: {done} / {total} frames",
                    icon="CHECKMARK" if total and done >= total else "PREVIEW_RANGE")
        row = cache.row(align=True)
        row.operator("plantgen.cache_timelapse", icon="RENDER_ANIMATION")
        row.operator("plantgen.clear_timelapse", text="", icon="TRASH")
        wrapped_label(box, "Each frame is grown once (slow), then plays from the cache; "
                           "any edit clears it. Renders always regrow exactly")
    else:
        wrapped_label(info, "Regrows the real plant every frame, recorded for smooth "
                            "playback after the first pass")


def wrapped_label(layout, text: str) -> None:
    """A help sentence wrapped to the sidebar's width. Blender cuts a label too
    long for its row out of the middle ("Approximates Unre...ly."), so a line
    split by hand reads only at the width it was split for."""
    import textwrap
    ctx = bpy.context
    scale = ctx.preferences.system.ui_scale or 1.0      # 0 in a background Blender
    width = ctx.region.width if ctx.region is not None else 280 * scale
    # ~6 px a character at UI scale 1, less the panel's and a box's margins
    chars = max(18, int((width - 52 * scale) / (6.0 * scale)))
    col = layout.column(align=True)
    for line in textwrap.wrap(text, chars):
        col.label(text=line)


def _warning_label(layout, text: str) -> None:
    """A warning label in Blender's alert colour, text and ERROR icon alike
    (Python cannot colour a label otherwise)."""
    row = layout.row()
    row.alert = True
    row.label(text=text, icon="ERROR")


def _draw_rig(rig_box, s, obj, has_rig: bool, stale: bool) -> None:
    """Rig: the deform skeleton + skin, built from the plant as it stands --
    its build settings first, then Build Rig, the view, and Export last."""
    angles = rig_box.column(align=True)
    angles.label(text="Bone Angle (per level)")
    for lv in s.branch_levels:
        angles.prop(lv, "rig_bone_angle", text=lv.name)
    rig_box.operator("plantgen.build_rig", icon="ARMATURE_DATA")
    if has_rig:
        view = rig_box.row(align=True)
        view.prop(s, "rig_view", toggle=True, icon="HIDE_OFF")
        bones = view.row(align=True)
        bones.enabled = s.rig_view
        bones.prop(s, "rig_show_bones", toggle=True, icon="BONE_DATA")
        shape = rig_box.row(align=True)
        shape.enabled = s.rig_view and s.rig_show_bones
        shape.prop(s, "rig_bone_display", text="Display")
        fast = rig_box.row(align=True)
        fast.enabled = s.rig_view
        fast.prop(s, "rig_fast_preview", icon="EXPERIMENTAL")
        if s.rig_fast_preview and s.wind_secondary_enable:
            rig_box.label(text="Vertex Wind on: playing the full mesh (slower)", icon="INFO")
        found = rig.find_rig(obj)
        counts = found[0].get("pgs_counts") if found else None
        if counts:
            rig_box.label(text=f"{counts['stem']} stem + {counts['organ']} organ bones "
                          "+ root", icon="INFO")
        if stale:
            warn = rig_box.box()
            warn.alert = True                       # red, text and icon
            warn.label(text="The plant changed since the rig was built", icon="ERROR")
            warn.label(text="Rig View and Export show the old plant --")
            warn.label(text="Build Rig again to update them")
        rig_box.operator("plantgen.export_rig", icon="EXPORT")
        rig_box.operator("plantgen.export_nanite_assembly", icon="EXPORT")


def _draw_wind(wind_box, s, obj, has_rig: bool) -> None:
    """Wind -- two PREVIEWS on the rig, neither exported: Bone Wind
    approximates Unreal's Dynamic Wind (which simulates the real sway from
    the wind groups Export Rig writes), Vertex Wind the flutter an engine's
    foliage material makes from the vertex colours."""
    if not has_rig:
        wind_box.label(text="Build Rig first -- wind moves the rig", icon="INFO")
    elif not s.rig_view:
        wind_box.label(text="Turn on Rig View to see it", icon="INFO")
    body = wind_box.column()
    body.enabled = has_rig
    bone = body.column(align=True)
    bone.prop(s, "wind_enable", icon="BONE_DATA")
    bone_settings = bone.column(align=True)
    bone_settings.active = s.wind_enable
    bone_settings.prop(s, "wind_strength", text="Strength")
    bone_settings.prop(s, "wind_stiffness", text="Stiffness")
    wrapped_label(body, "Approximates Unreal's wind -- preview only. Unreal simulates "
                        "the real sway from the wind groups Export Rig writes")
    body.separator()
    vert = body.column(align=True)
    vert.prop(s, "wind_secondary_enable", icon="VERTEXSEL")
    vert_settings = vert.column(align=True)
    vert_settings.active = s.wind_secondary_enable
    vert_settings.prop(s, "wind_secondary_strength", text="Strength")
    wrapped_label(body, "Flutters leaf tips from the vertex colours -- preview only, "
                        "an engine's foliage material does this itself")
    body.separator()
    gusts = body.column(align=True)
    gusts.active = s.wind_enable or s.wind_secondary_enable
    gusts.label(text="Gusts (both previews)")
    gusts.prop(s, "wind_direction_deg", slider=True, text="Direction")
    gusts.prop(s, "wind_gust_scale")
    gusts.prop(s, "wind_gust_speed")
    gusts.prop(s, "wind_turbulence")
    gusts.prop(s, "wind_seed")
    if operators.debug_tools():
        dbg = body.column(align=True)
        dbg.prop(s, "wind_debug", text="Show Gusts")
        if s.wind_debug != 'OFF':
            dbg.prop(s, "wind_debug_res")
    groups = rig.wind_summary(obj)
    if groups:
        # What Export Rig writes for Unreal's Dynamic Wind: Unreal simulates
        # the sway itself from these; Bone Wind above only previews it.
        ue = body.column(align=True)
        ue.label(text="Unreal wind groups (written by Export Rig)")
        for name, st, n in groups:
            if st.get("bUseDualInfluence"):
                value = f"{st['MinInfluence']:.2f}-{st['MaxInfluence']:.2f}"
            else:
                value = f"{st['Influence']:.2f}"
            ue.label(text=f"{name}: {n} bones, influence {value}")


# ============================================================================
# STEM -- the axis, and everything that is a property of it rather than of the
# organs hanging off it
# ============================================================================


def _draw_pruning(layout, obj, s) -> None:
    """Manual pruning (Phase 3): a cursor action -- X in the inspector overlay
    removes the limb under the cursor. Its own button starts that overlay here,
    beside the count, the retain default, a Clear and the warning shown when a
    topology change wiped the set (the Inspector itself is a Debug Tool,
    hidden by default)."""
    n = len(s.inspect_pruned)
    if s.prune_stale:
        layout.label(text="Pruning cleared: topology changed", icon='ERROR')
    live_now = overlay.is_active(obj)
    layout.operator("plantgen.toggle_inspect", text="Stop Pruning" if live_now else "Prune Limbs",
                    depress=live_now)
    if live_now:
        layout.label(text="Point at a limb and press X")
    layout.label(text=f"{n} limb{'' if n == 1 else 's'} pruned" if n else "No limbs pruned")
    layout.prop(s, "prune_retain_dead_wood", text="Keep Dead Wood")
    layout.operator("plantgen.clear_pruning", icon='LOOP_BACK')


def _section(layout, idname: str, text: str, closed: bool = False):
    """A collapsible section inside the Stem panel (`layout.panel`); its body,
    or None while folded."""
    head, body = layout.panel(f"PLANTGEN_stem_{idname}", default_closed=closed)
    head.label(text=text)
    return body


class PLANTGEN_PT_stem(_ComponentPanel):
    """Physiology: how the whole plant functions and responds, one fold each --
    Tropisms, Timing, Stiffness, Pruning, Surface; then its responses to the
    world (Light, Obstacles, Effectors, Senescence); then what is made of it
    (Rig, Wind). (Was "Plant-wide"; Environment, Senescence and Rig & Export
    were their own panels until 2026-10-14, user decision.) How many stems and
    branches there are, where they sit, how long and how thick, is the
    Components panel's level stack; UVs and materials are Texturing's.
    """

    bl_label = "Physiology"                   # (was "Stem", "Plant-wide"; the id kept)
    bl_idname = "PLANTGEN_PT_stem"
    bl_order = 13

    def draw(self, context):
        plant = _plant(context)
        s = plant.plantgen
        layout = _MapLayout(self.layout, s)

        # Tropisms -- where an axis turns as it grows. (Whether tips flower is
        # the flower component's Terminal attachment.)
        body = _section(layout, "tropisms", "Tropisms")
        if body:
            col = body.column(align=True)
            col.prop(s, "gravitropism", slider=True)
            col.prop(s, "autotropism", slider=True)
            col.prop(s, "circumnutation", slider=True)
            if s.circumnutation > 0.0:
                col.prop(s, "circumnutation_rate")

        # Timing -- how each internode and leaf grows in after it forms
        #: the mode leads, its curve and Duration
        # follow.
        body = _section(layout, "timing", "Timing")
        if body:
            col = body.column(align=True)
            col.label(text="Internode Elongation")
            row = col.row(align=True)
            row.prop(s, "elongation_mode", text="")
            _curve_button(row, "elongation")
            col.prop(s, "elongation_steps", text="Duration")
            col = body.column(align=True)
            row = col.row(align=True)
            row.prop(s, "leaf_expansion_steps", text="Leaf Expansion")
            _curve_button(row, "leaf_expansion")
            row = col.row(align=True)
            row.label(text="Stem Thickening")
            _curve_button(row, "stem_thickening")

        # Stiffness -- how the grown stems bend under their load. Its value is
        # each level's own (Branching > Width); the preset, the base->tip curve
        # and the per-stem variance stay here, plant-wide. (Moved from the
        # retired Physical Form panel.)
        body = _section(layout, "stiffness", "Stiffness")
        if body:
            col = body.column(align=True)
            col.prop(s, "structural_preset", text="Preset")
            row = col.row(align=True)
            row.label(text="Value: on each level")
            _map_button(row, s, s, "stem_stiffness")      # its curve + range
            _variance_button(row, s, "stiffness_variation", "Stiffness Variance")
            _debug_button(row, s, "stiffness")
            # The load it carries: leaf weight is each leaf component's; the
            # plant's own only matters while it has none (its grown leaves drawn
            # from the plant's collection).
            if not instancing.has_components(s, "leaf"):
                _field(col, s, "leaf_weight", debug="load")

        # Pruning -- limbs cut by hand (Phase 3), its own button beside them.
        body = _section(layout, "pruning", "Pruning", closed=True)
        if body:
            _draw_pruning(body, plant, s)

        # Surface -- how finely the stems are built and whether their junctions
        # are welded. The section, flare and natural detail are the Branching
        # levels' (the stems level carries the flare and detail).
        body = _section(layout, "surface", "Surface")
        if body:
            col = body.column(align=True)
            col.prop(s, "stem_smoothness")
            col.prop(s, "stem_weld", text="Weld Junctions")
            weld = col.column(align=True)
            weld.active = s.stem_weld
            weld.prop(s, "stem_fillet")
            weld.prop(s, "light_stalks")
            state = bridge.weld_state(plant)
            if s.stem_weld:
                body.operator("plantgen.weld", icon="AUTOMERGE_ON",
                              text="Welding..." if state == 'WELDING' else "Weld")
                _draw_weld_status(body, state)
            if state == 'WELDED':                         # (not the unwelded preview)
                built, capped = (list(plant.get("plantgen_weld", [])) + [0, 0])[:2]
                if built or capped:
                    # Junctions that could not be built (a branch too short to clear
                    # its fillet, overlapping a sibling, off its parent's end) stay
                    # capped tubes inside the parent.
                    weld.label(text=f"{built} built, {capped} capped", icon="INFO")

        # Light -- the tips' turn toward light, and flowers / leaves facing the sun.
        body = _section(layout, "light", "Light", closed=True)
        if body:
            col = body.column(align=True)
            col.prop(s, "light_seeking", slider=True)
            col.prop(s, "leaf_heliotropism", slider=True)
            # Flower Heliotropism and its sun are each flower component's; this
            # Sun Object is the leaves'.
            if bridge.plant_sun_used(s):
                col.prop(s, "sun_object", text="Sun Object")
                if s.sun_object is None:
                    body.label(text="Pick a Sun Object for the leaves to face",
                               icon="INFO")

        # Obstacles -- scene geometry the growing tips keep clear of.
        body = _section(layout, "obstacles", "Obstacles", closed=True)
        if body:
            body.prop(s, "obstacle_collection", text="Collection")
            col = body.column(align=True)
            col.active = s.obstacle_collection is not None
            col.prop(s, "obstacle_clearance")
            if s.obstacle_collection is not None and s.obstacle_clearance <= 0.0:
                body.label(text="Clearance is 0 - obstacles ignored", icon="INFO")

        # Ground -- a terrain the bent plant rests on (flat ground without one).
        body = _section(layout, "ground", "Ground", closed=True)
        if body:
            body.prop(s, "ground_object", text="Object")

        # Effectors -- empties that steer growth.
        body = _section(layout, "effectors", "Effectors", closed=True)
        if body:
            body.prop(s, "effector_collection", text="Collection")
            body.operator("plantgen.add_effector", icon="ADD")

        # Senescence (WS3) -- what dies and how it looks: a post-growth pass, off
        # by default (a flat Age Senescence curve: nothing dies). The Age
        # death-front is its curve alone; shade-death is judged against the
        # plant's own light field, so it needs no obstacle.
        body = _section(layout, "senescence", "Senescence", closed=True)
        if body:
            row = body.row(align=True)
            row.label(text="Age Senescence")
            _curve_button(row, "senescence")
            body.prop(s, "senescence_shade_threshold")
            body.prop(s, "senescence_retain_dead_wood")
            body.prop(s, "senescence_dead_color")

        # Rig and Wind -- the deform rig built from the plant as it stands, its
        # FBX / USD export (with Unreal's wind file), and the wind previews on it.
        has_rig = rig.find_rig(plant) is not None
        stale = has_rig and rig.rig_stale(plant)
        head, body = layout.panel("PLANTGEN_dev_rig", default_closed=True)
        if stale:
            _warning_label(head, "Rig  (out of date)")
        else:
            head.label(text="Rig")
        if body is not None:
            _draw_rig(body, s, plant, has_rig, stale)
        head, body = layout.panel("PLANTGEN_dev_wind", default_closed=True)
        head.label(text="Wind")
        if body is not None:
            _draw_wind(body, s, plant, has_rig)



# ============================================================================
# BRANCHING -- the per-level stack (branching-overhaul-plan phase 5)
# ============================================================================


class PLANTGEN_UL_branch_levels(UIList):
    """One row per branching level: its name, how many nodes it grows, and a
    Show box. The names are the stack's own ("Stems", "Branches 1", ...) and
    follow a level's place, so they are labels, not renamable fields."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            body = row.row(align=True)
            body.active = item.show
            body.label(text=item.name, **plant_icons.kw(_level_icon(index == 0)))
            info = body.row()
            info.alignment = 'RIGHT'
            info.label(text=f"{item.steps} steps" + (
                "" if item.growth_pattern == 'monopodial' else f"  {item.growth_pattern}"))
            row.prop(item, "show", text="", emboss=False,
                     icon='CHECKBOX_HLT' if item.show else 'CHECKBOX_DEHLT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", **plant_icons.kw(_level_icon(False)))


def _level_box(layout, text: str):
    """A labelled group box for one part of the active level; its column."""
    box = layout.box()
    box.label(text=text)
    return box.column(align=True)


_TREE_PREVIEWS = None
_TREE_ICONS: dict = {}
"""The Plant Structure's guide lines: three small images (a line, a branch, the
last branch) loaded as custom icons, one fixed-width icon cell per depth, so
they line up from row to row -- a Python list has no tree drawing of its own."""

_TREE_GUIDES = {'|': "tree_vert", 'T': "tree_branch", 'L': "tree_last"}
_TREE_SWATCH = 0.6      # the tree's colour swatch width, UI units (square in a row)


def _load_tree_icons() -> None:
    global _TREE_PREVIEWS
    import os

    import bpy.utils.previews
    folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")
    _TREE_PREVIEWS = bpy.utils.previews.new()
    for name in _TREE_GUIDES.values():
        path = os.path.join(folder, name + ".png")
        if os.path.exists(path):
            _TREE_ICONS[name] = _TREE_PREVIEWS.load(name, path, 'IMAGE').icon_id


def _free_tree_icons() -> None:
    global _TREE_PREVIEWS
    if _TREE_PREVIEWS is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_TREE_PREVIEWS)
        _TREE_PREVIEWS = None
    _TREE_ICONS.clear()


class PLANTGEN_UL_structure(UIList):
    """The Plant Structure tree: the Plant, the levels and the components placed
    on them, each row with its guide lines. Double-click (or F2) a name to
    rename it; the box on the right shows or mutes it."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        s = data
        if self.layout_type not in {'DEFAULT', 'COMPACT'}:
            layout.label(text="", icon='OUTLINER')
            return
        row = layout.row(align=True)
        for ch in item.guides:
            icon_id = _TREE_ICONS.get(_TREE_GUIDES.get(ch, ""), 0)
            if icon_id:
                row.label(text="", icon_value=icon_id)
            else:
                row.label(text="", icon='BLANK1')
        right = layout.row(align=True)
        right.alignment = 'RIGHT'
        if item.kind == 'UNPLACED':
            row.label(text="Not Placed", icon='ASSET_MANAGER')
            return
        if item.kind in ('COMPONENT', 'LIBRARY'):
            if item.kind == 'LIBRARY':
                c = s.components[item.ref] if 0 <= item.ref < len(s.components) else None
            elif 0 <= item.ref < len(s.placements):
                _i, c = structure.component_by_uid(s, s.placements[item.ref].comp_uid)
            else:
                c = None
            if c is None:
                return
            sw = row.row()
            sw.ui_units_x = _TREE_SWATCH
            sw.prop(c, "color", text="")
            name = row.row()
            name.active = not c.mute
            name.prop(c, "label", text="", emboss=False,
                      **plant_icons.kw(_COMPONENT_ICONS.get(c.type, "DOT")))
            n = structure.component_users(s, c.uid)
            if n > 1:
                right.label(text=str(n), icon='LINKED')
            right.prop(c, "mute", text="", emboss=False,
                       icon='CHECKBOX_DEHLT' if c.mute else 'CHECKBOX_HLT')
            return
        if item.kind == 'PLANT':
            row.prop(s, "structure_plant_expanded", text="", emboss=False,
                     icon='DISCLOSURE_TRI_DOWN' if s.structure_plant_expanded
                     else 'DISCLOSURE_TRI_RIGHT')
            sw = row.row()
            sw.ui_units_x = _TREE_SWATCH
            sw.prop(s, "plant_color", text="")
            row.label(text="Plant", **plant_icons.kw(_PLANT_ICON))
            return
        if not 0 <= item.ref < len(s.branch_levels):
            return
        lv = s.branch_levels[item.ref]
        row.prop(lv, "expanded", text="", emboss=False,
                 icon='DISCLOSURE_TRI_DOWN' if lv.expanded else 'DISCLOSURE_TRI_RIGHT')
        sw = row.row()
        sw.ui_units_x = _TREE_SWATCH
        sw.prop(lv, "color", text="")
        name = row.row()
        name.active = lv.show
        name.prop(lv, "name", text="", emboss=False,
                  **plant_icons.kw(_level_icon(lv.parent_uid == 0)))
        right.prop(lv, "show", text="", emboss=False,
                   icon='CHECKBOX_HLT' if lv.show else 'CHECKBOX_DEHLT')


class PLANTGEN_MT_component_new_placed(Menu):
    """A new component, placed on the Components tree's selection."""

    bl_idname = "PLANTGEN_MT_component_new_placed"
    bl_label = "New Component"

    def draw(self, context):
        for identifier, name, _desc in properties.COMPONENT_TYPE_ITEMS:
            op = self.layout.operator("plantgen.component_add", text=name,
                                      **plant_icons.kw(_COMPONENT_ICONS.get(identifier, "DOT")))
            op.type, op.place = identifier, True


class PLANTGEN_MT_structure_add(Menu):
    """The tree's +: a new component or a library one on the selection, or a
    branching level under it."""

    bl_idname = "PLANTGEN_MT_structure_add"
    bl_label = "Add"

    def draw(self, context):
        s = _plant(context).plantgen
        layout = self.layout
        target = structure.structure_target(s)
        where = "Plant" if target == 0 else next(
            (lv.name for lv in s.branch_levels if lv.uid == target), "?")
        layout.label(text=f"On {where}")
        layout.menu("PLANTGEN_MT_component_new_placed", text="New Component", icon='ADD')
        layout.separator()
        for c in s.components:
            layout.operator("plantgen.tree_place", text=c.label,
                            **plant_icons.kw(_COMPONENT_ICONS.get(c.type, "DOT"))).comp_uid = c.uid
        layout.separator()
        layout.operator("plantgen.tree_add_level", text="Stems",
                        **plant_icons.kw(_level_icon(True))).stems = True
        row = layout.row()
        row.enabled = target != 0      # branches grow on a level, never the Plant
        row.operator("plantgen.tree_add_level", text="Branches",
                     **plant_icons.kw(_level_icon(False))).stems = False


class PLANTGEN_MT_structure_move_to(Menu):
    """Every level the selection can move onto, in tree order, indented."""

    bl_idname = "PLANTGEN_MT_structure_move_to"
    bl_label = "Move To"

    def draw(self, context):
        s = _plant(context).plantgen
        layout = self.layout
        level = s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(
            s.branch_levels)
        current = (s.branch_levels[s.branch_levels_active].parent_uid if level else
                   s.placements[s.structure_place].level_uid
                   if s.structure_focus == 'COMPONENT'
                   and 0 <= s.structure_place < len(s.placements) else None)

        def entry(uid, text, icon, depth):
            row = layout.row()
            # levels are locked by kind: stems on the Plant, branches on a level
            row.enabled = uid != current and (
                not level or structure.level_can_move(s, s.branch_levels_active, uid))
            row.operator("plantgen.tree_move_to", text="    " * depth + text,
                         **plant_icons.kw(icon)).uid = uid

        entry(0, "Plant", _PLANT_ICON, 0)

        def walk(parent, depth):
            for i in structure.child_levels_of(s, parent):
                lv = s.branch_levels[i]
                entry(lv.uid, lv.name, _level_icon(parent == 0), depth)
                walk(lv.uid, depth + 1)

        walk(0, 1)


class PLANTGEN_MT_structure_specials(Menu):
    bl_idname = "PLANTGEN_MT_structure_specials"
    bl_label = "Components"

    def draw(self, context):
        layout = self.layout
        layout.menu("PLANTGEN_MT_structure_move_to", icon='OUTLINER')
        layout.separator()
        layout.operator("plantgen.component_copy", icon='DUPLICATE')
        layout.operator("plantgen.component_remove", icon='TRASH')
        layout.separator()
        layout.operator("plantgen.tree_expand", text="Expand All",
                        icon='DISCLOSURE_TRI_DOWN').expand = True
        layout.operator("plantgen.tree_expand", text="Collapse All",
                        icon='DISCLOSURE_TRI_RIGHT').expand = False


class PLANTGEN_PT_structure(_ComponentPanel):
    """The plant's structure as a tree: the Plant, its stems levels, the branch
    levels growing on them, and the components placed on each (a component on
    the Plant row goes on every level), then the components placed nowhere yet.
    The selected level's or component's settings follow, in a foldable box.
    One component on several levels shares every setting (Natsura's
    decorations, clean-room); where several components of a type are eligible
    for a point, it goes to one at random in proportion to their Density, and a
    point no component wants renders nothing."""

    bl_label = "Components"
    bl_idname = "PLANTGEN_PT_structure"    # (the old Plant Structure's: keeps its fold)
    bl_order = 11

    def draw(self, context):
        s = _plant(context).plantgen
        layout = self.layout
        if not len(s.branch_levels) or not len(s.tree_rows):
            # Seeded by the next regrow (bridge), never from here: draw() cannot
            # write RNA.
            layout.label(text="Regrow to set up the plant structure", icon="INFO")
            return
        row = layout.row()
        row.template_list("PLANTGEN_UL_structure", "", s, "tree_rows", s, "tree_active",
                          rows=8)
        side = row.column(align=True)
        side.menu("PLANTGEN_MT_structure_add", text="", icon='ADD')
        side.operator("plantgen.tree_remove", text="", icon='REMOVE')
        side.separator()
        side.operator("plantgen.component_copy", text="", icon='DUPLICATE')
        side.menu("PLANTGEN_MT_structure_specials", text="", icon='DOWNARROW_HLT')
        side.separator()
        # Component Colours: every organ painted in its component's swatch.
        _debug_button(side, s, 'component')
        side.separator()
        side.operator("plantgen.tree_move", text="", icon='TRIA_UP').direction = "UP"
        side.operator("plantgen.tree_move", text="", icon='TRIA_DOWN').direction = "DOWN"
        side.separator()
        can_out = can_in = True
        if s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(s.branch_levels):
            # locked by kind: a stems level never indents, a branch level never
            # outdents onto the Plant
            i = s.branch_levels_active
            lv = s.branch_levels[i]
            up = structure.level_index_of(s, lv.parent_uid)
            can_out = up >= 0 and structure.level_can_move(s, i, s.branch_levels[up].parent_uid)
            can_in = lv.parent_uid != 0 and structure.child_levels_of(s, lv.parent_uid)[0] != i
        sub = side.column(align=True)
        sub.enabled = can_out
        sub.operator("plantgen.tree_indent", text="", icon='BACK').direction = "OUT"
        sub = side.column(align=True)
        sub.enabled = can_in
        sub.operator("plantgen.tree_indent", text="", icon='FORWARD').direction = "IN"

        if s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(s.branch_levels):
            lv = s.branch_levels[s.branch_levels_active]
            head, body = layout.panel_prop(s, "structure_level_open")
            _settings_header(head, lv.name)
            if body is not None:
                _draw_level_details(_MapLayout(body, s), context, s, s.branch_levels_active)
        elif s.structure_focus in ('COMPONENT', 'LIBRARY') and 0 <= s.components_active < len(
                s.components):
            c = s.components[s.components_active]
            head, body = layout.panel_prop(s, "structure_component_open")
            _settings_header(head, c.label)
            if body is not None:
                _draw_component_details(_MapLayout(body, s), context, s)


def _settings_header(head, name: str) -> None:
    """A settings box's header in the Components panel: what is selected."""
    head.label(text=f"{name} Settings")


_RELAYING_PATTERNS = ('sympodial', 'dichasial', 'pleiochasial', 'pseudomonopodial_relay')
_FORKING_PATTERNS = ('dichotomous', 'pseudomonopodial_fork')


def _draw_level_details(layout, context, s, i: int) -> None:
    """A branching level's settings, grouped Phyllotaxis / Orientation / Growth /
    Length / Width / Cross Section / Spiral / Fork, then its Masks and Maps. On
    the stems level the same fields read as the base's spread -- Whorl is how
    many stems, Pitch how far they splay -- and a Surface box carries the flare
    and the stems' natural detail."""
    lv = s.branch_levels[i]
    stems = lv.parent_uid == 0

    if stems:
        # A stems level has no parent to spiral round: its stems all emerge at
        # once in one ring (node 0), so Phyllotaxis and Spiral would do nothing
        # but overwrite the count. Only the ring itself is drawn.
        col = _level_box(layout, "Arrangement")
        col.prop(lv, "whorl", text="Stems")
        sub = col.column(align=True)
        sub.active = lv.whorl > 1
        sub.prop(lv, "whorl_angle", text="Spacing")
        # How far apart the stems stand at the ground (a plant-level trait,
        # shown with the stem count it spaces).
        row = sub.row(align=True)
        row.prop(s, "basal_spread", text="Spread")
        _variance_button(row, s, "basal_spread_variation", "Spread Variation")
    else:
        col = _level_box(layout, "Phyllotaxis")
        col.prop(lv, "phyllotaxis", text="")
        col.prop(lv, "spiral_angle")
        col.prop(lv, "whorl")
        col.prop(lv, "whorl_angle")
        row = col.row(align=True)
        row.prop(lv, "slide", slider=True)
        _variance_button(row, lv, "slide_variation", "Slide Variation", level=True)

    col = _level_box(layout, "Orientation")
    for prop, text in (("start_pitch", "Splay" if stems else "Pitch"),
                       ("start_yaw", "Yaw"), ("start_roll", "Roll")):
        row = col.row(align=True)
        row.prop(lv, prop, text=text)
        _variance_button(row, lv, prop + "_variation", text + " Variation", level=True)
    if not stems:
        col.prop(lv, "upper_side_up")

    # Growth -- the pattern leads, Steps (on a relaying level, the whole chain's
    # length), Growth % and Emergence Delay follow, then only the settings the
    # pattern reads.
    col = _level_box(layout, "Growth")
    col.prop(lv, "growth_pattern", text="")
    col.prop(lv, "steps")
    col.prop(lv, "growth", slider=True)
    col.prop(lv, "emergence_delay")
    col.prop(lv, "seed")
    pattern = lv.growth_pattern
    if pattern in _RELAYING_PATTERNS:
        col.separator()
        # (A unit's tip flowers when the flower component is Terminal.)
        col.prop(lv, "unit_steps")
        if pattern == 'pleiochasial':
            col.prop(lv, "relay_count")
        col.prop(lv, "fork_spread", text="Lean" if pattern == 'pseudomonopodial_relay'
                 else "Spread")
        col.prop(lv, "fork_whorl", text="Whorl")
    elif pattern in _FORKING_PATTERNS:
        col.separator()
        col.prop(lv, "fork_probability", slider=True)
        col.prop(lv, "fork_branches", text="Branches")
        if pattern == 'pseudomonopodial_fork':
            col.prop(lv, "fork_balance", slider=True)
        col.prop(lv, "fork_spread", text="Spread")
        col.prop(lv, "fork_whorl", text="Whorl")

    col = _level_box(layout, "Length")
    # The stems have no parent: their Start is metres, not a ratio.
    col.prop(lv, "start_length_scale", text="Start (m)" if stems else "Start")
    row = col.row(align=True)
    row.prop(lv, "length_scale_per_step", text="Per Step")
    _variance_button(row, lv, "length_variation", "Length Variation", level=True)

    col = _level_box(layout, "Width")
    col.prop(lv, "start_width_scale", text="Radius (m)" if stems else "Start")
    row = col.row(align=True)
    row.prop(lv, "width_scale_per_step", text="Per Step")
    _variance_button(row, lv, "width_variation", "Width Variation", level=True)
    col.prop(lv, "stiffness", slider=True)

    # The shape swept along this level's axes -- each level its own.
    col = _level_box(layout, "Cross Section")
    col.prop(lv, "cross_section", text="")
    shape = lv.cross_section
    if shape == 'polygonal':
        col.prop(lv, "profile_sides")
    elif shape == 'flattened':
        col.prop(lv, "flatten", slider=True)
    elif shape in ('winged', 'ribbed', 'star'):
        label = {'winged': "Wings", 'ribbed': "Ribs", 'star': "Points"}[shape]
        col.prop(lv, "lobe_count", text=label)
        col.prop(lv, "lobe_depth", text="Depth", slider=True)

    col = _level_box(layout, "Spiral")
    col.prop(lv, "spiral_pitch", text="Pitch / Step")
    col.prop(lv, "spiral_roll", text="Roll / Step")
    # The angle from vertical each node grows toward (world-frame, unlike the
    # spiral): a map only -- the angle itself is each branch's launch.
    _pseudo_map_row(col, s, lv, "gravitropic_setpoint")


    if stems:
        # The stems' surface: the flare at their base, and the natural
        # detail of the swept stems (render-side, over the whole plant).
        col = _level_box(layout, "Surface")
        row = col.row(align=True)
        row.prop(s, "trunk_flare", slider=True)
        row.prop(s, "flare_lobes")
        col.prop(s, "stem_gnarl", slider=True)
        col.prop(s, "bark_noise", slider=True)
        col.prop(s, "stem_twist", slider=True)

    # Masks -- which of this level's branches grow (a rejected one never
    # does, nor anything on it); on the stems level, which stems emerge.
    box = layout.box()
    box.label(text="Masks")
    frow = box.row()
    frow.template_list("PLANTGEN_UL_level_mask", "level_masks", lv, "filters", lv,
                       "filters_active", rows=2)
    fcol = frow.column(align=True)
    fcol.menu("PLANTGEN_MT_level_mask_add", text="", icon="ADD")
    fcol.operator("plantgen.mask_remove", text="", icon="REMOVE").owner = "LEVEL"
    fcol.separator()
    up = fcol.operator("plantgen.mask_move", text="", icon="TRIA_UP")
    up.direction, up.owner = "UP", "LEVEL"
    down = fcol.operator("plantgen.mask_move", text="", icon="TRIA_DOWN")
    down.direction, down.owner = "DOWN", "LEVEL"
    if 0 <= lv.filters_active < len(lv.filters):
        entry = lv.filters[lv.filters_active]
        _draw_mask_details(box, entry, lv.filters_active, s, random_band=True)
        _growth_channel_warning(box, entry)
    else:
        box.label(text="Keep only some " + ("stems" if stems else "branches")
                  + " -- by position, height, noise...", icon="INFO")


class PLANTGEN_UL_level_mask(UIList):
    """A branching level's masks: attribute, invert, a preview eye (where this
    mask kept / removed the level's branches), mute. `data` is the active level."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            name = _MASK_CHANNEL_NAMES.get(item.channel, item.channel)
            if item.channel == "custom":
                name = item.attr_name or name
            elif item.channel == "class":
                name = "Class: " + (item.class_name or "?")
            label = row.row()
            label.active = not item.mute
            label.label(text=name)
            row.prop(item, "invert", text="", icon='ARROW_LEFTRIGHT')
            s = data.id_data.plantgen
            on = (s.debug_view_mode == 'mask_preview'
                  and s.mask_preview_level == s.branch_levels_active
                  and index == data.filters_active)
            row.operator("plantgen.level_mask_debug", text="", emboss=False,
                         icon='HIDE_OFF' if on else 'HIDE_ON', depress=on).index = index
            row.prop(item, "mute", text="", emboss=False,
                     icon='CHECKBOX_DEHLT' if item.mute else 'CHECKBOX_HLT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="MOD_MASK")


def _growth_channel_warning(layout, ref) -> None:
    """Flag a growth-time mask / map on a channel that exists only on the
    finished plant (bloom, apex, branch index): it has no effect while growing."""
    if not bridge.growth_readable(bridge.channel_key(ref)):
        layout.label(text="This attribute only exists on the grown plant -- "
                     "no effect while growing", icon="ERROR")


class PLANTGEN_MT_level_mask_add(Menu):
    """A new level mask's attribute, picked up front -- only those growth can
    read (a branch is decided while the plant grows)."""

    bl_idname = "PLANTGEN_MT_level_mask_add"
    bl_label = "Add Mask"

    def draw(self, context):
        layout = self.layout
        for identifier, name, _desc in properties.MASK_CHANNEL_ITEMS:
            if identifier in ("custom", "class") or bridge.growth_readable(identifier):
                op = layout.operator("plantgen.mask_add", text=name)
                op.channel, op.owner = identifier, "LEVEL"


# ============================================================================
# FLOWERS
# ============================================================================


_OLD_WOOD_ATTACHMENTS = {'CAULIFLOROUS', 'RAMIFLOROUS', 'ADVENTITIOUS'}
"""The attachments gated by Wood Age -- all three break from wood laid
down seasons earlier, so all three need that control."""


def _draw_catkins(layout, s, *, component: bool = False, uid: int = 0) -> None:
    """The Catkins box.

    Drawn only for the Catkin type, which is also the only type that grows
    them -- `development._add_catkins` tests the same thing. The settings
    outlive the kind on purpose (a stamp is never cleared here, so tuning
    survives a trip through another type and comes back), which makes it all
    the more important that nothing reads them unless Catkin is chosen.

    A catkin is the one thing here that is neither an inflorescence on an axis
    nor an instanced asset -- it is a curve hung at a node -- so it gets its own
    box rather than being folded into either. How many share a node is the
    Only the *body* is here. Where catkins are borne and which way they hang
    are the shared Borne and Orientation controls above -- a catkin is an
    axillary, pendulous inflorescence, and saying so through the same controls
    every other kind uses is the whole point of the three-layer split.

    A bunch is one stalk carrying several catkins, which is what
    `catkin_count` says -- deliberately not the leaf arrangement, since hazel's
    foliage is spiral and its catkins still come in threes.
    """
    # No on/off toggle: the flower component owns its catkins, so muting it (or
    # its masks) is what switches them off -- a second switch here was the same
    # control twice.
    box = layout.box()
    box.label(text="Catkins")
    _field(box, s, "catkin_length", text="Length", variance="catkin_length_variation",
           component=component, uid=uid)
    col = box.column()
    # Everything below describes a catkin, so with no length there is nothing
    # for them to describe. Greyed rather than hidden: the settings are kept,
    # and the rows say what giving it a length would let you shape.
    col.active = s.catkin_length > 0.0
    _field(col, s, "catkin_count", text="Per Stalk",
           variance="catkin_count_variation", component=component, uid=uid)
    col.prop(s, "catkin_width", text="Width")
    _field(col, s, "catkin_stalk", text="Stalk",
           variance="catkin_stalk_variation", component=component, uid=uid)
    if mapping.owner_of(s) is not None:
        _pseudo_map_row(col, s.id_data.plantgen, s, "catkin_hang")
    col.prop(s, "catkin_noise", text="Noise", slider=True)
    if hasattr(s, "catkin_material"):
        col.prop(s, "catkin_material")


def _draw_flowers(layout, s, plant_s, *, component: bool = False, uid: int = 0,
                  primary: bool = True) -> None:
    """The whole inflorescence UI, drawn from `s` -- the plant, or (component=True,
    with its `uid`) a flower component, which sources the same fields and curves.
    `plant_s` is always the plant: Bloom *display* (mode/progress/spread) still
    drives the GN layer from the plant, so it is drawn from there even on the
    component. The flower-port shared draw; the plant Flowers panel goes in
    phase 4 and this then serves the component alone. `primary` False: a
    further inflorescence, which takes the plant's flower Weight and catkin body
    from the first flower component."""
    def field(lo, prop, **kw):
        _field(lo, s, prop, component=component, uid=uid, **kw)

    if s.inflorescence_kind == 'NONE':
        box = layout.box()
        box.label(text="Inflorescence")
        box.prop(s, "inflorescence_kind", text="Type")
        box.label(text="Stems finish and stop without flowering.")
        return

    catkin = s.inflorescence_kind == 'CATKIN'
    box = layout.box()
    box.label(text="Inflorescence")
    box.prop(s, "inflorescence_kind", text="Type")
    col = box.column(align=True)
    col.label(text="Attachment Location")
    col.prop(s, "inflorescence_attachments", expand=True)
    box.prop(s, "inflorescence_per_node", text="Per Node")
    borne = set(s.inflorescence_attachments)
    if borne - {'TERMINAL'}:
        if 'INTERNODAL' in borne:
            box.prop(s, "inflorescence_internode_position", text="Along Internode",
                     slider=True)
        if borne & _OLD_WOOD_ATTACHMENTS:
            box.prop(s, "inflorescence_cauline_age", text="Wood Age")
        # Which nodes along the stems carry the inflorescence is a mask now, not
        # an on-trunk/on-branches/start/end window (removed): it arises on every
        # eligible node and the flower component's masks thin it.
    box.prop(s, "inflorescence_orientation", text="Orientation")
    if s.inflorescence_orientation == 'CUSTOM':
        box.prop(s, "inflorescence_orientation_deg", text="Angle")
    box.prop(s, "inflorescence_orientation_strength", text="Strength", slider=True)
    if s.inflorescence_orientation_strength > 0.0:
        box.prop(s, "inflorescence_orientation_load_driven",
                 text="Reach It As Flowers Open")
    if catkin:
        if primary:
            _draw_catkins(layout, s, component=component, uid=uid)
        else:
            layout.label(text="Catkin body: set on the first flower component", icon="INFO")
        return

    box = layout.box()
    box.label(text="Flowers")
    # The flower mesh is the component's own Collection (drawn above in the
    # component panel); the legacy plant asset field is only for the plant panel.
    if not component:
        box.prop(s, "flower_asset_collection", text="Collection")
    if _relayed(s):
        bb = layout.box()
        bb.label(text="Relays")
        field(bb, "inflorescence_branching_depth", text="Depth", debug="structural_depth")
        bb.prop(s, "inflorescence_laterals", text="Per Node")
        bb.prop(s, "inflorescence_relay_rotation", text="Rotation")
        field(bb, "inflorescence_branch_angle", text="Zigzag Angle",
              variance="inflorescence_branch_angle_variation")
        field(bb, "inflorescence_branch_length", text="Length",
              variance="inflorescence_branch_length_variation")
        bb.prop(s, "inflorescence_branch_nodes", text="Nodes")
        field(bb, "inflorescence_branch_width", text="Width",
              variance="inflorescence_branch_width_variation", slider=True)
        field(bb, "inflorescence_branch_bend", text="Bend",
              variance="inflorescence_branch_bend_variation", slider=True)
        field(bb, "inflorescence_branch_droop", text="Droop",
              variance="inflorescence_branch_droop_variation", slider=True)
    elif s.inflorescence_kind != 'SOLITARY':
        bb = layout.box()
        bb.label(text="Branching")
        field(bb, "inflorescence_branching_depth", text="Depth", debug="structural_depth")
        if s.inflorescence_branching_depth > 0:
            cluster = _terminal_cluster(s)
            field(bb, "inflorescence_branch_count",
                  text="Rays" if cluster else "Count",
                  variance="inflorescence_branch_count_variation")
            if not cluster:
                field(bb, "inflorescence_branch_angle", text="Angle",
                      variance="inflorescence_branch_angle_variation")
                field(bb, "inflorescence_branch_spacing", text="Spacing",
                      variance="inflorescence_branch_spacing_variation", slider=True)
            field(bb, "inflorescence_branch_nodes", text="Nodes",
                  variance="inflorescence_branch_nodes_variation")
            field(bb, "inflorescence_branch_length", text="Length",
                  variance="inflorescence_branch_length_variation")
            field(bb, "inflorescence_branch_width", text="Width",
                  variance="inflorescence_branch_width_variation", slider=True)
            field(bb, "inflorescence_branch_bend", text="Bend",
                  variance="inflorescence_branch_bend_variation", slider=True)
            field(bb, "inflorescence_branch_droop", text="Droop",
                  variance="inflorescence_branch_droop_variation", slider=True)
        field(box, "flower_count", text="Count", variance="flower_count_variation")
        if not _terminal_cluster(s):
            box.prop(s, "inflorescence_whorl_count", text="Per Whorl")
            spiral = box.row(align=True)
            spiral.active = s.inflorescence_whorl_count < 2
            spiral.prop(s, "inflorescence_spiral", text="Spiral")
        if _terminal_cluster(s):
            if s.inflorescence_kind != 'FLAT_UMBEL' or s.inflorescence_branching_depth > 0:
                box.prop(s, "cluster_spread")
        else:
            field(box, "inflorescence_internode_scale", text="Spacing",
                  variance="flower_spacing_variation")
            # The plant's spacing base the Flower Spacing multiplies (and the
            # rachis's Branch Spacing, an inflorescence branch's Length).
            box.prop(plant_s, "internode_length", text="Spacing Base")
        if _flat_topped(s):
            box.prop(s, "head_radius")
            field(box, "head_dome", curve="head_dome",
                  variance="head_dome_variation", slider=True)

    # Flower size is the component's own Size (drawn above, with its flower_size
    # curve + variance); no separate Flower Size row. Weight is the mass that
    # nods a blooming tip (it scales with the cube of Size) -- it drives the
    # pedicel droop, so it lives with the flowers now, not in Physical Form.
    if primary:
        field(box, "flower_weight", text="Weight")
    else:
        box.label(text="Weight: set on the first flower component", icon="INFO")
    # How young a just-laid-down inflorescence looks (spacing, flower, stalk),
    # with its growth-to-full curve; a mature one is always full size.
    field(box, "inflorescence_young_size", text="Young Size", slider=True)
    if not _capped(s) and not _terminal_cluster(s):
        field(box, "flower_spread", variance="flower_spread_variation")
    if _sessile(s) and not _capped(s):
        field(box, "flower_pitch", variance="flower_pitch_variation")
    if component:
        # The flowers turn their faces to the plant's Sun Object (Physiology >
        # Light), the stalks left where they are.
        field(box, "flower_heliotropism", text="Heliotropism", slider=True)
        if s.flower_heliotropism > 0.0:
            box.prop(s, "sun_object", text="Sun Object")
            if s.sun_object is None:
                box.label(text="Pick a Sun Object for the flowers to face", icon="INFO")

    if not _sessile(s):
        box = layout.box()
        box.label(text="Pedicel")
        field(box, "pedicel_width", text="Width", variance="pedicel_width_variation", slider=True)
        field(box, "pedicel_bend", text="Bend", variance="pedicel_bend_variation", slider=True)
        if _flat_topped(s):
            pass
        elif _terminal_cluster(s):
            field(box, "pedicel_length", text="Length", variance="pedicel_variation")
        else:
            field(box, "pedicel_length", text="Length", variance="pedicel_variation")
            field(box, "pedicel_angle", text="Angle", variance="pedicel_angle_variation")

    box = layout.box()
    box.label(text="Bloom")
    if s.inflorescence_kind != 'SOLITARY':
        # Bloom order re-sources from the component; the display below (mode /
        # progress / spread) still drives the GN layer from the plant, so it is
        # drawn from plant_s. Per-component bloom display is a later GN change.
        # A solitary flower has no order: it is its inflorescence's only one.
        field(box, "bloom_order", text="Order", debug="bloom_order")
    if component:
        # Each flower earlier or later than its order (a Random Range
        # mixes buds and open flowers): an instance map, no value of its own.
        _pseudo_map_row(box, plant_s, s, "bloom_timing", text="Timing (each flower)")
    box.prop(plant_s, "bloom_continuous")
    if plant_s.bloom_continuous:
        # Continuous pairs stages vertex by vertex: a stage that does not
        # share the first one's topology blends into the wrong vertices.
        odd = gn_mesh.bloom_stage_mismatch(
            getattr(s, "collection", None) if component
            else plant_s.flower_asset_collection)
        if odd:
            warn = box.column(align=True)
            warn.alert = True
            warn.label(text=f"Not the same mesh as the first stage: {', '.join(odd)}",
                       icon="ERROR")
            warn.label(text="Blending scrambles it -- bake every stage from one flower")
    box.prop(plant_s, "bloom_mode")
    col = box.column(align=True)
    col.active = plant_s.bloom_mode != 'STATIC'
    if plant_s.bloom_mode == 'AGE':
        row = col.row(align=True)
        row.enabled = False
        if plant_s.last_bloom_window >= 0.0:
            note = "from the calendar bloom window"
        else:
            note = f"flowers from Age {plant_s.last_flowering_age:.2f}"
        row.label(text=f"Progress: {gn_mesh.effective_bloom_progress(plant_s):.2f}"
                       f"   ({note})")
        _debug_button(row, plant_s, "bloom")
        _bloom_window_hint(col, plant_s)
    else:
        _field(col, plant_s, "bloom_progress", debug="bloom", slider=True)
    col.prop(plant_s, "bloom_spread", text="Spread", slider=True)
    if plant_s.bloom_mode == 'MANUAL':
        col.label(text="Keyframe Progress to animate")


# ============================================================================
# COMPONENTS -- multiple instances per type, each a collection filtered by its own
# inline attribute-window list
# ============================================================================


_MASK_CHANNEL_NAMES = {i: n for i, n, _ in properties.MASK_CHANNEL_ITEMS}

# PlantGen's own icons ("PG:<name>", `plant_icons`): pass any of these through
# `plant_icons.kw`, never straight to `icon=`.
_COMPONENT_ICONS = {
    "leaf": "PG:leaf", "bract": "PG:bract", "flower": "PG:flower", "bud": "PG:bud",
    "fruit": "PG:fruit",
}
_PLANT_ICON = "PG:plant"


def _level_icon(stems: bool) -> str:
    """A branching level's icon: the stems level, or a level of branches."""
    return "PG:stems" if stems else "PG:branches"


def _swatch(row, item):
    """A square colour chip: a fixed-width sub-row holding just the colour prop, so
    it renders as tall as it is wide instead of stretching across the row.
    `ui_units_x` sets an absolute width (~half a UI unit) that matches the list's
    row height -- tuned live to read square, where `scale_x` came out rectangular."""
    sw = row.row(align=True)
    sw.ui_units_x = 0.5
    sw.prop(item, "color", text="")


# ============================================================================
# PARAMETER MAPPING
# ============================================================================


def _map_button(layout, s, pg, param: str) -> None:
    """A parameter's map icon in its owner's colour: filled when mapped,
    outlined when not; the maps on hover, a click opens them (or adds one)."""
    found = mapping.owner_of(pg)
    if found is None:
        return
    owner, uid = found
    style = "on" if mapping.is_mapped(s, owner, uid, param) else "off"
    op = layout.operator("plantgen.map_param", text="", emboss=False,
                         icon_value=mapping.icon(s, owner, uid, style))
    op.owner, op.owner_uid, op.param = owner, uid, param


def _pseudo_map_row(layout, s, pg, param: str, text: str | None = None) -> None:
    """A row for a mapped quantity with no value of its own (an instance's
    rotation, a petiole's start angle): its name and its map icon."""
    row = layout.row(align=True)
    row.label(text=mapping.param_label(pg, param) if text is None else text)
    _map_button(row, s, pg, param)


class _MapLayout:
    """A panel's layout that draws every mappable numeric field of a level, a
    component or the plant with that parameter's map icon after it -- so the
    panels need no per-field change. Everything else is passed straight to the
    real layout; the sub-layouts it makes are wrapped too."""

    __slots__ = ("_layout", "_plant")

    def __init__(self, layout, s):
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_plant", s)

    def __getattr__(self, name):
        return getattr(self._layout, name)

    def __setattr__(self, name, value):
        setattr(self._layout, name, value)

    def _wrap(self, sub):
        return _MapLayout(sub, self._plant) if sub is not None else None

    def row(self, *args, **kw):
        return self._wrap(self._layout.row(*args, **kw))

    def column(self, *args, **kw):
        return self._wrap(self._layout.column(*args, **kw))

    def box(self, *args, **kw):
        return self._wrap(self._layout.box(*args, **kw))

    def split(self, *args, **kw):
        return self._wrap(self._layout.split(*args, **kw))

    def grid_flow(self, *args, **kw):
        return self._wrap(self._layout.grid_flow(*args, **kw))

    def column_flow(self, *args, **kw):
        return self._wrap(self._layout.column_flow(*args, **kw))

    def panel(self, *args, **kw):
        head, body = self._layout.panel(*args, **kw)
        return self._wrap(head), self._wrap(body)

    def panel_prop(self, *args, **kw):
        head, body = self._layout.panel_prop(*args, **kw)
        return self._wrap(head), self._wrap(body)

    def prop(self, data, prop, **kw):
        if not mapping.is_mappable(data, prop):
            return self._layout.prop(data, prop, **kw)
        row = self._layout.row(align=True)
        row.prop(data, prop, **kw)
        _map_button(row, self._plant, data, prop)
        return None


def _maps_hint(layout) -> None:
    """How a map is made, said once at the top of Mapping & Classification
    (it ended every settings box until 2026-10-14; each map icon's own tooltip
    says the rest)."""
    row = layout.row()
    row.label(text="Map any parameter with the icon beside it")


def _owner_swatch(row, s, owner: str, uid: int) -> None:
    """An owner's colour swatch -- the very property its Components tree row
    draws, so editing either one edits both."""
    pg = mapping.owner_pg(s, owner, uid)
    if pg is None:
        return
    sw = row.row()
    sw.ui_units_x = _TREE_SWATCH
    sw.prop(pg, "plant_color" if owner == mapping.PLANT else "color", text="")


def _owner_icon(s, owner: str, uid: int) -> str:
    """The owner's icon in the Components tree."""
    pg = mapping.owner_pg(s, owner, uid)
    if owner == mapping.PLANT or pg is None:
        return _PLANT_ICON
    if owner == mapping.LEVEL:
        return _level_icon(pg.parent_uid == 0)
    return _COMPONENT_ICONS.get(pg.type, "DOT")


class PLANTGEN_UL_map_rows(UIList):
    """Every map on the plant, grouped under its owner as in the Components
    tree: a foldable header with the owner's swatch (the same colour as its
    tree row), icon and name; its maps below on the tree's guide lines, by
    parameter name, their kind greyed and a box that switches each off."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        s = data
        if self.layout_type not in {'DEFAULT', 'COMPACT'}:
            layout.label(text="", icon='FCURVE')
            return
        row = layout.row(align=True)
        if item.kind == "OWNER":
            key = f"{item.owner}:{item.owner_uid}"
            fold = next((f for f in s.map_folds if f.key == key), None)
            if fold is not None:
                row.prop(fold, "expanded", text="", emboss=False,
                         icon='DISCLOSURE_TRI_DOWN' if fold.expanded else 'DISCLOSURE_TRI_RIGHT')
            _owner_swatch(row, s, item.owner, item.owner_uid)
            row.label(text=mapping.owner_title(s, item.owner, item.owner_uid),
                      **plant_icons.kw(_owner_icon(s, item.owner, item.owner_uid)))
            return
        pg = mapping.owner_pg(s, item.owner, item.owner_uid)
        m = mapping.stored(s, item.kind, item.owner, item.owner_uid, item.index)
        for ch in item.guides:
            icon_id = _TREE_ICONS.get(_TREE_GUIDES.get(ch, ""), 0)
            if icon_id:
                row.label(text="", icon_value=icon_id)
            else:
                row.label(text="", icon='BLANK1')
        on = item.enabled if m is None else m.enabled
        body = row.row(align=True)
        body.active = on
        body.label(text=mapping.param_label(pg, item.param))
        right = layout.row(align=True)
        right.alignment = 'RIGHT'
        kind = right.row(align=True)
        kind.active = False
        kind.label(text=mapping.entry_name(s, item.kind, item.owner, item.owner_uid,
                                           item.param, item.index))
        right.prop(item if m is None else m, "enabled", text="", emboss=False,
                   icon='CHECKBOX_HLT' if on else 'CHECKBOX_DEHLT')


def _draw_map_details(layout, plant, s, row) -> None:
    """The selected map's settings, in a foldable box headed by its owner and
    parameter (like the Components panel's settings box)."""
    owner, uid, param = row.owner, row.owner_uid, row.param
    pg = mapping.owner_pg(s, owner, uid)
    if pg is None:
        return
    head, body = layout.panel_prop(s, "mapping_details_open")
    head.label(text=f"{mapping.owner_title(s, owner, uid)} \u203a "
                    f"{mapping.param_label(pg, param)}")
    if body is None:
        return
    if row.kind == "CURVE":
        _draw_curve_map(body, s, row, pg)
        return
    sub = body.row()
    sub.enabled = False
    sub.label(text="Evaluated once per plant" if row.kind == "PMAP"
              else "Evaluated " + mapping.per_label(s, owner, uid, param))
    if row.kind == "FMAP" and all(c.uid != uid for c in bridge.inflorescence_owners(s)):
        sub = body.row()
        sub.enabled = False
        sub.label(text="No effect: this component draws another one's flowers "
                  "(Own Inflorescence off)", icon="INFO")
    box = body.box()
    if row.kind == "VAR":
        # The mode, as on every other map: another one turns it into that map.
        box.menu("PLANTGEN_MT_map_mode", text="Random Range")
        data, prop = mapping.var_field(s, owner, uid, param)
        if prop is not None:
            col = box.column()
            col.active = row.enabled
            col.prop(data, prop, text="Range \u00b1")
        note = box.column(align=True)
        note.enabled = False
        note.label(text="A random value per element within \u00b1 Range;")
        note.label(text="the " + ("plant's" if owner == mapping.PLANT else "owner's")
                   + " Seed reshuffles it.")
        return
    m = mapping.stored(s, row.kind, owner, uid, row.index)
    if m is None:
        return
    box.menu("PLANTGEN_MT_map_mode", text=mapping.preset_label(owner, param, m.preset))
    col = box.column()
    col.active = m.enabled
    col.use_property_split = True
    col.use_property_decorate = False
    rotation = row.kind == "IMAP" and m.target in mapping.pm.ROTATION_TARGETS
    additive = rotation or (row.kind == "PMAP" and mapping.is_additive(pg, param))
    if row.kind == "PMAP":
        # Say why there is no Driver: one value for the whole plant.
        info = box.column(align=True)
        info.enabled = False
        info.label(text="One value for the whole plant: it can't vary", icon="INFO")
        info.label(text="along an attribute -- the curve reads a random draw")
        curve = mapping.curve_entry(s, owner, uid, param)
        if curve is not None:
            info.label(text=f"To vary it {curve[2]}, use its {curve[1]} curve")
    else:
        _draw_channel_ref(col, m, s, text="Driver")
    if m.preset == 'RANGE':
        col.prop(m, "angle" if additive else "range", text="Range \u00b1")
    elif additive:
        col.prop(m, "angle")
    col.prop(m, "amount", slider=True)
    if m.channel == "random":
        col.prop(m, "seed")
    if row.kind in {"LMAP", "GMAP", "FMAP"}:
        _growth_channel_warning(col, m)
    if not (row.kind == "PMAP" and m.preset == 'RANGE'):
        node = mapping.curve_node(plant, row.kind, owner, uid, m)
        if node is None:
            box.label(text="Regrow to create this map's curve", icon="INFO")
        else:
            cap = box.row()
            cap.label(text=f"X: {'random' if row.kind == 'PMAP' else m.channel} (0 -> 1)")
            cap.label(text="Y: -Angle / 0.5 none / +Angle" if additive else "Y: multiplier")
            box.template_curve_mapping(node, "mapping")


_CURVE_Y = {spec.key: spec.y_label for spec in curves.CURVE_SPECS}
_CURVE_Y.update({"size": "size multiplier", "petiole_width": "width multiplier"})


def _draw_curve_map(layout, s, row, pg) -> None:
    """A profile curve's settings: its fixed driver, the curve itself (stored
    where it always was) and its shape presets."""
    entry = mapping.curve_entry(s, row.owner, row.owner_uid, row.param)
    node = mapping.param_curve_node(s, row.owner, row.owner_uid, row.param)
    if entry is None or node is None:
        layout.label(text="Regrow to create this curve", icon="INFO")
        return
    key, _short, caption = entry
    sub = layout.row()
    sub.enabled = False
    sub.label(text="Driver: " + caption)
    box = layout.box()
    col = box.column()
    col.active = row.enabled
    cap = col.row()
    cap.enabled = False
    neutral = ("straight = even growth" if key in mapping.pm.CURVE_NEUTRAL
               else "1 = the value as set")
    cap.label(text="Y: " + _CURVE_Y.get(key, "multiplier") + f" ({neutral})")
    col.template_curve_mapping(node, "mapping")
    col.menu("PLANTGEN_MT_map_curve_shapes", text=curves.matching_shape(node) or "Custom")


class PLANTGEN_MT_map_curve_shapes(Menu):
    """Shape presets for the profile curve selected in the Mapping list -- on
    its own owner, whichever component is active."""

    bl_idname = "PLANTGEN_MT_map_curve_shapes"
    bl_label = "Shape Presets"

    def draw(self, context):
        layout = self.layout
        plant = _plant(context)
        s = plant.plantgen if plant is not None else None
        row = mapping.active_row(s) if s is not None else None
        if row is None or row.kind != "CURVE":
            return
        entry = mapping.curve_entry(s, row.owner, row.owner_uid, row.param)
        node = mapping.param_curve_node(s, row.owner, row.owner_uid, row.param)
        if entry is None or node is None:
            return
        matched = curves.matching_shape(node)
        layout.label(text="Custom", icon="CHECKMARK" if matched is None else "BLANK1")
        layout.separator()
        for ident, shape_label, _points, _smooth in curves.SHAPES:
            op = layout.operator("plantgen.curve_shape", text=shape_label)
            op.curve_key, op.shape = entry[0], ident
            if row.owner == mapping.COMPONENT:
                op.component, op.comp_uid = True, row.owner_uid


class PLANTGEN_PT_mapping(_ComponentPanel):
    """Mapping & Classification: every map on the plant in one list, grouped
    under the part it belongs to (its Components tree colour) -- a parameter's
    map icon opens it here, the selected map's settings follow in a foldable
    box -- then the classification blocks every mask and map can read, each a
    fold: Custom Attributes and Classes (were their own panel until
    2026-10-11)."""

    bl_label = "Mapping & Classification"
    bl_idname = "PLANTGEN_PT_mapping"
    bl_order = 15

    def draw(self, context):
        plant = _plant(context)
        s = plant.plantgen
        layout = self.layout
        _maps_hint(layout)
        self._draw_maps(layout, plant, s)
        head, body = layout.panel("PLANTGEN_map_custom_attrs", default_closed=True)
        head.label(text=f"Custom Attributes ({len(s.custom_attrs)})")
        if body is not None:
            _draw_custom_attrs(body, s)
        head, body = layout.panel("PLANTGEN_map_classes", default_closed=True)
        head.label(text=f"Classes ({len(s.classes)})")
        if body is not None:
            _draw_classes(body, s)

    @staticmethod
    def _draw_maps(layout, plant, s) -> None:
        row = layout.row()
        row.template_list("PLANTGEN_UL_map_rows", "", s, "map_rows", s, "map_rows_active",
                          rows=8)
        side = row.column(align=True)
        side.operator("plantgen.map_search_add", text="", icon='ADD')
        side.operator("plantgen.map_row_remove", text="", icon='REMOVE')
        if not len(s.map_rows):
            layout.label(text="No maps yet -- or + to search for a parameter", icon="INFO")
            return
        r = mapping.active_row(s)
        if r is None or r.kind == "OWNER":
            layout.label(text="Pick a map to edit it", icon="INFO")
            return
        _draw_map_details(layout, plant, s, r)


class PLANTGEN_MT_mask_add(Menu):
    """Pick the attribute for a new filter up front -- easier than adding a blank
    one and hunting for it in a dropdown. Tuned in the details below."""

    bl_idname = "PLANTGEN_MT_mask_add"
    bl_label = "Add Mask"

    def draw(self, context):
        layout = self.layout
        for identifier, name, _desc in properties.MASK_CHANNEL_ITEMS:
            op = layout.operator("plantgen.mask_add", text=name)
            op.channel = identifier


class PLANTGEN_UL_components(UIList):
    """One row per component: its type as an icon, the collection it instances, the
    Density mix-weight, and a mute. Where several components of a type are eligible
    for a point, it goes to one in proportion to these weights."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            _swatch(row, item)
            body = row.row(align=True)
            body.active = not item.mute
            name = item.collection.name if item.collection else "(no collection)"
            body.label(text=name, **plant_icons.kw(_COMPONENT_ICONS.get(item.type, "DOT")))
            weight = row.row(align=True)
            weight.active = not item.mute
            weight.ui_units_x = 2.2          # a small weight box on the row
            weight.prop(item, "density", text="")
            row.prop(item, "mute", text="", emboss=False,
                     icon='CHECKBOX_DEHLT' if item.mute else 'CHECKBOX_HLT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", **plant_icons.kw(_COMPONENT_ICONS.get(item.type, "DOT")))


class PLANTGEN_UL_mask(UIList):
    """One row per filter on a component: a colour swatch, the attribute it reads,
    an invert marker, a per-attribute preview eye, and a mute. The window is tuned
    in the details below -- the row stays a legible summary. `data` is the component
    PG (only the active component's filters are drawn), reaching the growth PG
    through the object it lives on."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            _swatch(row, item)
            name = _MASK_CHANNEL_NAMES.get(item.channel, item.channel)
            label = row.row()
            label.active = not item.mute      # a muted filter's name reads greyed
            label.label(text=name)
            # Invert toggles right here in the row -- Blender's own "invert vertex
            # group" icon, highlighted when on.
            row.prop(item, "invert", text="", icon='ARROW_LEFTRIGHT')
            s = data.id_data.plantgen
            on = (s.debug_view_mode == 'mask_preview' and s.mask_preview_level < 0
                  and index == data.filters_active)
            row.operator("plantgen.mask_debug", text="", emboss=False,
                         icon='HIDE_OFF' if on else 'HIDE_ON', depress=on).index = index
            row.prop(item, "mute", text="", emboss=False,
                     icon='CHECKBOX_DEHLT' if item.mute else 'CHECKBOX_HLT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="MOD_MASK")


def _draw_channel_ref(layout, ref, s, prop="channel", attr_prop="attr_name",
                      class_prop="class_name", text=None) -> None:
    """A channel dropdown, plus -- for Custom Attribute / Class -- the name picker
    searching the plant's own lists."""
    layout.prop(ref, prop, text=text) if text is not None else layout.prop(ref, prop)
    value = getattr(ref, prop)
    if value == "custom":
        layout.prop_search(ref, attr_prop, s, "custom_attrs", text="")
    elif value == "class":
        layout.prop_search(ref, class_prop, s, "classes", text="")


def _draw_mask_details(layout, entry, index: int, s, random_band: bool = False) -> None:
    """The active mask entry's settings -- shared by a component's masks, a
    class's stack and a level's masks. A Random entry's band is drawn only on a
    level (`random_band`): its width is the share of buds that grow, the
    level's only random thinning -- a component thins with Prune."""
    det = layout.column(align=True)
    det.active = not entry.mute   # a muted filter's controls read greyed
    _draw_channel_ref(det, entry, s)
    if entry.channel in masks.BINARY_CHANNELS or entry.channel == "class":
        # A 0/1 flag: no min/max -- it keeps one side, invert (in the row) flips
        # to the other. Say which, so it reads plainly.
        on, off = (("In class", "Outside it") if entry.channel == "class"
                   else _BINARY_SIDES.get(entry.channel, ("On", "Off")))
        keep, other = (off, on) if entry.invert else (on, off)
        det.label(text=f"Keeps: {keep}  (invert → {other})", icon="INFO")
    elif entry.channel == "random" and random_band:
        # One slider: the band's width is the share of the buds kept.
        det.prop(entry, "probability", slider=True)
        det.label(text=f"Keeps ~{entry.probability:.0%} of the buds, at random "
                       "(the level's Seed)", icon="INFO")
    elif entry.channel != "random":
        sub = det.row(align=True)
        sub.prop(entry, "lo")
        sub.prop(entry, "hi")
    # Index channels can also pick a subset by a math op (modulo = every Nth).
    if entry.channel in masks.ORDINAL_CHANNELS:
        det.prop(entry, "math_op")
        if entry.math_op == "MODULO":
            det.prop(entry, "math_value", text="modulo")
    if index > 0:
        det.prop(entry, "op")


class PLANTGEN_UL_named(UIList):
    """One row per custom attribute / class: its name, renamable in place."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False,
                        icon="FILTER" if hasattr(item, "filters") else "NODE_SOCKET_FLOAT")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="FILTER")


class PLANTGEN_UL_class_mask(UIList):
    """A class's mask stack: attribute name, invert, mute (no preview eye -- the
    Filter Preview follows the active component)."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            name = _MASK_CHANNEL_NAMES.get(item.channel, item.channel)
            if item.channel == "custom":
                name = item.attr_name or name
            elif item.channel == "class":
                name = "Class: " + (item.class_name or "?")
            label = row.row()
            label.active = not item.mute
            label.label(text=name)
            row.prop(item, "invert", text="", icon='ARROW_LEFTRIGHT')
            row.prop(item, "mute", text="", emboss=False,
                     icon='CHECKBOX_DEHLT' if item.mute else 'CHECKBOX_HLT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="MOD_MASK")


class PLANTGEN_UL_maps(UIList):
    """One row per map: on/off, what it drives, and what drives it. The curve and
    amount are in the details below the list."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            label = row.row()
            label.active = item.enabled
            target = item.bl_rna.properties["target"].enum_items[item.target].name
            driver = _MASK_CHANNEL_NAMES.get(item.channel, item.channel)
            if item.channel == "custom":
                driver = item.attr_name or driver
            elif item.channel == "class":
                driver = "Class: " + (item.class_name or "?")
            label.label(text=f"{target}  <-  {driver}")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="FCURVE")


def _draw_maps(layout, plant, pg, prop: str, owner: str, curve_uid: int, kind: str,
               title: str, hint: str) -> None:
    """A map list and the active map's settings + curve, for either a
    component's instance maps or the plant's growth maps."""
    box = layout.box()
    box.label(text=title)
    maps = getattr(pg, prop)
    row = box.row()
    row.template_list("PLANTGEN_UL_maps", prop, pg, prop, pg, prop + "_active", rows=2)
    col = row.column(align=True)
    col.operator("plantgen.map_add", text="", icon="ADD").owner = owner
    col.operator("plantgen.map_remove", text="", icon="REMOVE").owner = owner
    if not len(maps):
        box.label(text=hint, icon="INFO")
        return
    active = getattr(pg, prop + "_active")
    if not (0 <= active < len(maps)):
        return
    m = maps[active]
    det = box.column(align=True)
    det.active = m.enabled
    det.prop(m, "target")
    _draw_channel_ref(det, m, plant.plantgen, text="Driver")
    det.prop(m, "amount", slider=True)
    if hasattr(m, "angle") and m.target in mapping.pm.ROTATION_TARGETS:
        det.prop(m, "angle")
    node = curves.get_component_curve(plant, curve_uid, curves.map_curve_name(kind, m.uid))
    if node is None:
        box.label(text="Regrow to create this map's curve", icon="INFO")
        return
    cap = box.row()
    cap.label(text="X: driver (0 -> 1)")
    rotation = hasattr(m, "angle") and m.target in mapping.pm.ROTATION_TARGETS
    cap.label(text="Y: -Angle / 0.5 none / +Angle" if rotation else "Y: multiplier")
    box.template_curve_mapping(node, "mapping")


def _active_component(context):
    """The active component of the active plant, or None."""
    plant = _plant(context)
    s = plant.plantgen if plant else None
    if s and 0 <= s.components_active < len(s.components):
        return plant, s.components[s.components_active]
    return plant, None


class PLANTGEN_PT_component_size_curve(Panel):
    """The active component's own Size curve, read at each generated instance's
    own age. One popover for whichever component is active, rather than one class
    per component -- the node it draws is resolved live from the active uid."""

    bl_label = "Size over Age"
    bl_idname = "PLANTGEN_PT_component_size_curve"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'               # a popover only, never a sidebar panel
    bl_ui_units_x = 14

    def draw(self, context):
        layout = self.layout
        plant, comp = _active_component(context)
        node = curves.get_component_curve(plant, comp.uid) if comp else None
        if node is None:
            layout.label(text="Not on this plant yet.", icon='INFO')
            layout.label(text="Press Regrow to add it.")
            return
        row = layout.row()
        row.label(text="X: instance age (0 new tip → 1 old)")
        row.label(text="Y: size ×")
        layout.template_curve_mapping(node, "mapping")
        layout.menu("PLANTGEN_MT_component_size_shapes",
                    text=curves.matching_shape(node) or "Custom")


class PLANTGEN_MT_component_size_shapes(Menu):
    """Shape presets for the active component's Size curve -- the plant curves'
    menu, retargeted at the component via `curve_shape.component`."""

    bl_idname = "PLANTGEN_MT_component_size_shapes"
    bl_label = "Shape Presets"

    def draw(self, context):
        layout = self.layout
        _plant_obj, comp = _active_component(context)
        node = curves.get_component_curve(_plant_obj, comp.uid) if comp else None
        if node is not None:
            matched = curves.matching_shape(node)
            layout.label(text="Custom", icon="CHECKMARK" if matched is None else "BLANK1")
            layout.separator()
        for ident, shape_label, _points, _smooth in curves.SHAPES:
            op = layout.operator("plantgen.curve_shape", text=shape_label)
            op.shape = ident
            op.component = True


def _make_component_curve_panel(name: str, label: str, x_label: str, y_label: str):
    """One popover + shape menu for a flower component's `name` curve, generated
    like the plant curve popovers but resolved live from the active component's
    uid rather than a fixed key."""
    panel_id = f"PLANTGEN_PT_c_{name}_curve"
    menu_id = f"PLANTGEN_MT_c_{name}_shapes"

    def draw_panel(self, context):
        layout = self.layout
        plant, comp = _active_component(context)
        node = curves.get_component_curve(plant, comp.uid, name) if comp else None
        if node is None:
            layout.label(text="Not on this plant yet.", icon='INFO')
            layout.label(text="Press Regrow to add it.")
            return
        row = layout.row()
        row.label(text=f"X: {x_label}")
        row.label(text=f"Y: {y_label}")
        layout.template_curve_mapping(node, "mapping")
        layout.menu(menu_id, text=curves.matching_shape(node) or "Custom")

    def draw_menu(self, context):
        layout = self.layout
        plant, comp = _active_component(context)
        node = curves.get_component_curve(plant, comp.uid, name) if comp else None
        if node is not None:
            matched = curves.matching_shape(node)
            layout.label(text="Custom", icon="CHECKMARK" if matched is None else "BLANK1")
            layout.separator()
        for ident, shape_label, _points, _smooth in curves.SHAPES:
            op = layout.operator("plantgen.curve_shape", text=shape_label)
            op.curve_key = name
            op.shape = ident
            op.component = True

    # A popover only (opened by its curve button): the header region, as
    # Blender's own popovers -- a tabless sidebar panel lands in a "Misc" tab.
    panel = type(panel_id, (Panel,), {
        "bl_label": label, "bl_idname": panel_id,
        "bl_space_type": 'VIEW_3D', "bl_region_type": 'HEADER', "bl_ui_units_x": 14,
        "draw": draw_panel,
    })
    menu = type(menu_id, (Menu,), {
        "bl_label": "Shape Presets", "bl_idname": menu_id, "draw": draw_menu,
    })
    return panel, menu


# A popover per inflorescence curve, for the flower component (Phase 3 of the
# flower port). Generated from the same specs the plant curves use.
_component_curve_classes: list[type] = []
for _spec in curves.CURVE_SPECS:
    if _spec.attr.startswith("inflorescence."):
        _component_curve_classes.extend(_make_component_curve_panel(
            _spec.key, _spec.label, _spec.x_label, _spec.y_label))
# The petiole's width along it (its taper), on leaf / bract components.
_component_curve_classes.extend(_make_component_curve_panel(
    "petiole_width", "Petiole Width", "along the petiole (0 stem, 1 leaf)", "width x"))


def _draw_component_details(layout, context, s) -> None:
    """The active component's settings: where it is placed, then what it
    instances and how. Each component instances a collection onto the candidate
    points its own masks keep, on the levels it is placed on; where several components of
    a type are eligible for a point, it goes to one at random in proportion to
    their Density weights. A point no component is eligible for renders nothing,
    and Prune deletes a random fraction of a component's instances."""
    if not (0 <= s.components_active < len(s.components)):
        return
    c = s.components[s.components_active]
    box = layout.box()
    box.active = not c.mute
    # Where it is placed (the Plant Structure places it; one component on
    # several levels shares every setting below).
    where = structure.component_level_names(s, c.uid)
    box.label(text="Placed on: " + (", ".join(where) if where else
                                    "nowhere yet -- Move To a level (the menu by the tree)"),
              icon='NONE' if where else "INFO")
    box.prop(c, "type")
    box.prop(c, "collection")
    box.prop(c, "seed")
    # Placement is organ-only for flowers (a scattered bloom stage is meaningless).
    if c.type != "flower":
        box.label(text="Placement")
        box.grid_flow(columns=3, even_columns=True, align=True).prop(
            c, "placement", expand=True)
        place = set(c.placement)
        structured = {"NODE", "INTERNODE", "TIP", "INTERSECTION", "BASAL"} & place
        if "INTERNODE" in place:
            box.prop(c, "internode_count")
        if "BASAL" in place:
            box.prop(c, "basal_count")
        if "EXTRA" in place:
            box.prop(c, "extra_count", slider=True)
        sz = box.row(align=True)
        sz.prop(c, "size", slider=True)
        # Size Variation reaches grown (On Organs) points as well as placed ones.
        _variance_button(sz, c, "gen_size_variance", "Size Variation", component=True)
        owns_leaves = c.type == "leaf" and bridge.primary_leaf_component(s) == c
        if structured or owns_leaves:
            # Phyllotaxis: a preset seeds the spiral/whorl; how the whorl at each
            # anchor is arranged and how the spiral advances between anchors.
            # The primary leaf component's is also where the grown leaves sit.
            arr = box.column(align=True)
            arr.prop(c, "gen_phyllotaxis")
            arr.prop(c, "gen_divergence")
            row = arr.row(align=True)
            row.prop(c, "gen_whorl_count")
            sub = row.row(align=True)
            sub.active = c.gen_whorl_count > 1
            sub.prop(c, "gen_whorl_angle")
        if structured or "EXTRA" in place:
            orient = box.column(align=True)
            pr = orient.row(align=True)
            pr.prop(c, "gen_pitch")
            _variance_button(pr, c, "gen_pitch_variance", "Pitch Variation",
                             component=True)
            pr.prop(c, "gen_fold")
            _variance_button(pr, c, "gen_fold_variance", "Fold Variation", component=True)
            row = orient.row(align=True)
            row.prop(c, "gen_roll_self", text="Roll Self")
            _variance_button(row, c, "gen_roll_self_variance", "Roll Self Variation",
                             component=True)
            row.prop(c, "gen_roll_parent", text="Parent Roll")
            _variance_button(row, c, "gen_roll_parent_variance",
                             "Parent Roll Variation", component=True)
        if c.type == "leaf":
            # Grown or placed, its leaves load the stems (placed ones at the
            # nodes they sit on, `physics._leaf_load_mass`).
            box.prop(c, "leaf_weight", text="Weight")
        if c.type in ("leaf", "bract"):
            # (grown and placed alike: applied after the pose and the maps)
            box.prop(c, "sky_roll", slider=True)
            # Leaf Unfolding: how a young leaf is packed, relaxing to flat as it
            # reaches full size.
            emerge = box.column(align=True)
            emerge.label(text="Emerging (unfolds as it grows)")
            emerge.prop(c, "emerging_curl", text="Curl")
            emerge.prop(c, "emerging_midrib_fold", text="Midrib Fold")
            # Petiole -- the leaf's stalk, shaped like a flower's pedicel, folded
            # into its own group (its length in the header while folded).
            head, body = box.panel("PLANTGEN_component_petiole", default_closed=True)
            head.label(text="Petiole" if c.petiole_length <= 0.0
                       else f"Petiole  ({c.petiole_length * 1000:.0f} mm)")
            if body is not None:
                pet = body.column(align=True)
                row = pet.row(align=True)
                row.prop(c, "petiole_length", text="Length")
                _variance_button(row, c, "petiole_length_variation",
                                 "Petiole Length Variation", component=True)
                shape = pet.column(align=True)
                shape.active = c.petiole_length > 0.0
                row = shape.row(align=True)
                row.prop(c, "petiole_width", text="Width")
                row = shape.row(align=True)
                row.prop(c, "petiole_bend", text="Bend")
                _variance_button(row, c, "petiole_bend_variation", "Petiole Bend Variation",
                                 component=True)
                row = shape.row(align=True)
                row.prop(c, "petiole_droop", text="Droop")
                _variance_button(row, c, "petiole_droop_variation", "Petiole Droop Variation",
                                 component=True)
                _pseudo_map_row(shape, s, c, "petiole_angle")
            # Collision -- turn out of the stems and older leaves, prune the rest.
            head, body = box.panel("PLANTGEN_component_collision", default_closed=True)
            head.prop(c, "collide", text="")
            head.label(text="Collision")
            if body is not None:
                col = body.column(align=True)
                col.active = c.collide
                col.prop(c, "collide_max_turn", text="Max Turn")
                col.prop(c, "collide_prune")
                if c.collide:
                    # Measured: 26-27 of a 32 s regrow on a 600-blade grass tussock.
                    wrapped_label(body, "Slows every regrow considerably on a plant with "
                                        "many leaves: each one is tested against the others")
                stats = petioles.LAST_COLLISIONS.get(s.id_data.name, {}).get(c.uid)
                if stats and c.collide:
                    col.label(text=f"{stats[0]} turned, {stats[1]} pruned"
                              + (f", {stats[2]} still touching" if stats[2] else ""),
                              icon="INFO")
    elif c.type == "flower":
        # For a flower, Size IS the flower size: its flower_size gradient
        # curve and variance ride on the Size row (there is no Flower Size).
        sz = box.row(align=True)
        sz.prop(c, "size", slider=True)
        _variance_button(sz, c, "flower_size_variation", "Flower Size Variation",
                         component=True)
    else:
        box.prop(c, "size", slider=True)
    box.prop(c, "density")
    box.prop(c, "prune", slider=True)
    if c.type == "bud" and "ORGAN" in c.placement:
        # The old Buds panel's switches are masks now: On Organs shows every
        # dormant bud and shoot tip, and the Apex channel splits them.
        box.label(text="On Organs: the plant's dormant buds and shoot tips",
                  icon="INFO")
        box.label(text="Tip vs side buds: add an Apex mask", icon="BLANK1")
    if c.type == "fruit":
        # The primary fruit component (first On Organs) owns the plant's
        # fruit: its Size is the fruit size, and Set / Weight live here.
        primary_f = next((i for i, x in enumerate(s.components)
                          if x.type == "fruit" and "ORGAN" in x.placement), -1)
        if s.components_active == primary_f:
            box.separator()
            box.label(text="Fruit")
            box.prop(c, "fruit_set", text="Fruit Set", slider=True)
            box.prop(c, "fruit_weight", text="Weight")
            # Fruit lives on the calendar, not on a here-and-now slider: it
            # appears only inside the fruit window. Say so, rather than leave
            # the rows reading as inert when the Day sits outside it.
            box.label(text="Set / ripen / drop: the Phenology Calendar")
        elif "ORGAN" in c.placement:
            box.label(text="Fruit Set / Weight are set on the first fruit component",
                      icon="INFO")
    if c.type == "flower":
        box.label(text="Collection is a bud-to-open bloom sequence", icon="INFO")
        # A flower component that grows its own inflorescence draws the whole
        # of it here (the first always does); a variant draws on the flowers
        # the inflorescences on its levels grow, so it shows only the render
        # fields above. (Compared by index -- RNA hands back a fresh wrapper
        # each access, so `is` never holds for the same element.)
        primary_i = next((i for i, x in enumerate(s.components)
                          if x.type == "flower"), -1)
        box.separator()
        if s.components_active != primary_i:
            box.prop(c, "own_inflorescence")
        if bridge.component_inflorescence(s, c) is not None:
            box.label(text="Inflorescence")
            for mode, level, winner in bridge.contested_spots(s, c):
                box.label(text=f"{mode} on {level}: taken by {winner} (higher in the list)",
                          icon="ERROR")
            _draw_flowers(box, c, s, component=True, uid=c.uid,
                          primary=s.components_active == primary_i)
        else:
            box.label(text="Variant: draws on the flowers grown on its levels",
                      icon="INFO")

    # -- The component's own mask list: which points it claims.
    box.separator()
    box.label(text="Masks")
    frow = box.row()
    frow.template_list(
        "PLANTGEN_UL_mask", "", c, "filters", c, "filters_active", rows=3)
    fcol = frow.column(align=True)
    fcol.menu("PLANTGEN_MT_mask_add", text="", icon="ADD")
    fcol.operator("plantgen.mask_remove", text="", icon="REMOVE")
    fcol.separator()
    fcol.operator("plantgen.mask_move", text="", icon="TRIA_UP").direction = "UP"
    fcol.operator("plantgen.mask_move", text="", icon="TRIA_DOWN").direction = "DOWN"

    fa = c.filters_active
    if 0 <= fa < len(c.filters):
        _draw_mask_details(box, c.filters[fa], fa, s)

    # The Classify view: this component's points on two attributes, its mask
    # window on them as a draggable rectangle (`classify_overlay`).
    ccol = box.column(align=True)
    ccol.operator("plantgen.classify_view", icon="SELECT_SET", text="Classify View",
                  depress=classify_overlay.is_active(_plant(context)))
    for which, label in (("x", "X"), ("y", "Y")):
        _draw_channel_ref(ccol.row(align=True), s, s, prop=f"classify_{which}",
                          attr_prop=f"classify_{which}_attr",
                          class_prop=f"classify_{which}_class", text=label)

    # -- Traits: which mesh of the collection each point it takes gets (the
    # sheet, `ui_traits`). Not for a flower: its meshes are bloom stages.
    if c.type != "flower":
        from . import ui_traits
        head, body = box.panel("PLANTGEN_component_traits", default_closed=True)
        on = len(c.trait_columns) or len(c.trait_rows)
        head.label(text="Traits" + (f"  ({len(c.trait_columns)} properties)" if on else ""))
        if body is not None:
            # The plain layout: the sheet's cells are not parameters to map.
            ui_traits.draw_sheet(body._layout if isinstance(body, _MapLayout) else body, s, c)

    # -- Instance maps: masks decide WHICH points this component takes; maps
    # decide HOW each is drawn there (Size / Pitch / Yaw / Roll per point).
    for param in ("instance_pitch", "instance_yaw", "instance_roll"):
        _pseudo_map_row(box, s, c, param)


# ============================================================================
# TEXTURING -- materials, UVs and the exported vertex colours, in one place
# ============================================================================


class PLANTGEN_PT_texturing(_ComponentPanel):
    """Everything about how the plant is textured: each level's stem material,
    the stems' UVs, and the vertex colours it exports. Leaves, flowers and fruit
    wear their own assets' materials (the leaf tool, FlowerGen)."""

    bl_label = "Texturing"
    bl_idname = "PLANTGEN_PT_texturing"
    bl_order = 16

    def draw(self, context):
        plant = _plant(context)
        s = plant.plantgen
        layout = self.layout

        # Materials: every material the plant wears, one list -- each level's
        # stem material (none = the parent level's), then each asset
        # component's (`ui_materials`); the selected row's editor below.
        head, body = layout.panel("PLANTGEN_texturing_materials")
        head.label(text="Materials")
        if body is not None:
            from . import ui_materials
            ui_materials.draw(body, plant)

        # UVs: one absolute size (metres per tile) on every stem tube.
        head, body = layout.panel("PLANTGEN_texturing_uv")
        head.label(text="UV")
        if body is not None:
            row = body.row(align=True)
            row.prop(s, "uv_size", text="Size")
            row.prop(s, "uv_checker", text="", icon="TEXTURE", toggle=True)

        # Vertex colours: four real channels on the exported Color attribute;
        # only the tint that shows them here is viewport-only.
        head, body = layout.panel("PLANTGEN_texturing_vcol", default_closed=True)
        head.label(text="Vertex Colors")
        if body is not None:
            body.label(text="Real data on the Color attribute")
            col = body.column(align=True)
            active = s.debug_view_mode
            for mode in ('color', 'vcol_r', 'vcol_g', 'vcol_b', 'vcol_a'):
                label = next(m[1] for m in properties.DEBUG_VIEW_MODES if m[0] == mode)
                col.operator("plantgen.toggle_debug_view", text=label,
                             depress=(active == mode)).mode = mode



# ============================================================================
# VARIATIONS -- a row / grid of copies of the plant, and baking them to meshes
# ============================================================================


def _variation_axis(layout, v, axis: str, title: str, s=None) -> None:
    """One stepped parameter: picked by search, then its start and step."""
    from . import variations
    box = layout.box()
    ident = getattr(v, f"{axis}_param")
    row = box.row(align=True)
    row.label(text=title)
    row.operator("plantgen.variations_pick", text=variations.param_name(ident),
                 icon="VIEWZOOM").axis = axis
    if ident:
        col = box.column(align=True)
        col.prop(v, f"{axis}_start")
        col.prop(v, f"{axis}_step")
    if ident == "bloom_progress" and s is not None and s.bloom_mode != 'MANUAL':
        # Only Manual reads the slider: From Age works Progress out from the
        # Day, Static shows each flower's own order.
        box.label(text="Bloom Progress steps only with Bloom set to Manual", icon="INFO")


def _draw_variations(layout, obj) -> None:
    """A row or a grid of copies of this plant, each with its own seed and one or
    two plant-wide parameters stepped across them (Day +36 per plant...); then
    Bake, each to a single mesh. The plant stays the master: Build replaces the
    set from its current settings. On a variation, the source's settings show."""
    from . import variations
    source = variations.source_of(obj)
    if source is not obj:
        layout.label(text=f"A variation of {source.name}: its settings below")
    v = source.plantgen.variations
    layout.row().prop(v, "layout", expand=True)
    col = layout.column(align=True)
    col.prop(v, "count_x")
    if v.layout == 'GRID':
        col.prop(v, "count_y")
    col.prop(v, "seed_step")

    _variation_axis(layout, v, "x", "Along Rows" if v.layout == 'GRID' else "Along the Row",
                    source.plantgen)
    if v.layout == 'GRID':
        _variation_axis(layout, v, "y", "Across Rows", source.plantgen)

    box = layout.box()
    box.prop(v, "auto_spacing")
    sub = box.column(align=True)
    sub.active = not v.auto_spacing
    sub.prop(v, "spacing_x")
    if v.layout == 'GRID':
        sub.prop(v, "spacing_y")
    row = box.row()
    row.prop(v, "labels")
    row.prop(v, "weld")

    n = v.count_x * (v.count_y if v.layout == 'GRID' else 1)
    built = variations.generated(source)
    layout.operator("plantgen.variations_build", icon="MESH_GRID",
                    text=f"{'Rebuild' if built else 'Build'} {n} Plant{'s' if n != 1 else ''}")
    row = layout.row(align=True)
    row.operator("plantgen.variations_bake", icon="MESH_DATA", text="Bake to Meshes")
    row.operator("plantgen.variations_clear", icon="TRASH", text="Clear")
    layout.operator("plantgen.variations_export", icon="EXPORT",
                    text="Export Nanite Assembly")
    if built:
        layout.label(text=f"{len(built)} built in {variations.collection_name(source)}")


def _draw_custom_attrs(layout, s) -> None:
    """Custom attributes: new per-point dimensions (closeness to an object,
    noise, two channels combined) any mask or map reads as Custom Attribute >
    <name> (Natsura-parity "chainable classification", clean-room)."""
    box = layout.column()
    row = box.row()
    row.template_list("PLANTGEN_UL_named", "custom_attrs", s, "custom_attrs",
                      s, "custom_attrs_active", rows=2)
    col = row.column(align=True)
    col.operator("plantgen.plant_list_add", text="", icon="ADD").list_name = \
        "custom_attrs"
    col.operator("plantgen.plant_list_remove", text="", icon="REMOVE").list_name = \
        "custom_attrs"
    if 0 <= s.custom_attrs_active < len(s.custom_attrs):
        a = s.custom_attrs[s.custom_attrs_active]
        det = box.column(align=True)
        det.prop(a, "kind")
        if a.kind == "DISTANCE":
            det.prop(a, "target")
            det.prop(a, "radius")
        elif a.kind == "NOISE":
            det.prop(a, "noise_scale")
            det.prop(a, "seed")
        else:
            det.prop(a, "op")
            _draw_channel_ref(det, a, s, prop="channel_a", attr_prop="attr_a",
                              class_prop="class_a")
            _draw_channel_ref(det, a, s, prop="channel_b", attr_prop="attr_b",
                              class_prop="class_b")
        box.label(text=f"Read it as Custom Attribute > {a.name}", icon="INFO")
    else:
        box.label(text="A new per-point dimension to mask or map on", icon="INFO")


def _draw_classes(layout, s) -> None:
    """Classes: named mask stacks any component's mask or map reads as Class >
    <name>; a class can read custom attributes and other classes -- build a
    classification once, chain it, reuse it everywhere."""
    box = layout.column()
    row = box.row()
    row.template_list("PLANTGEN_UL_named", "classes", s, "classes",
                      s, "classes_active", rows=2)
    col = row.column(align=True)
    col.operator("plantgen.plant_list_add", text="", icon="ADD").list_name = "classes"
    col.operator("plantgen.plant_list_remove", text="", icon="REMOVE").list_name = \
        "classes"
    if not (0 <= s.classes_active < len(s.classes)):
        box.label(text="A named mask stack, reusable by any component", icon="INFO")
        return
    k = s.classes[s.classes_active]
    box.label(text=f"{k.name}: keeps the points its masks keep")
    frow = box.row()
    frow.template_list("PLANTGEN_UL_class_mask", "", k, "filters", k,
                       "filters_active", rows=3)
    fcol = frow.column(align=True)
    fcol.operator("plantgen.mask_add", text="", icon="ADD").owner = "CLASS"
    fcol.operator("plantgen.mask_remove", text="", icon="REMOVE").owner = "CLASS"
    fcol.separator()
    up = fcol.operator("plantgen.mask_move", text="", icon="TRIA_UP")
    up.direction, up.owner = "UP", "CLASS"
    down = fcol.operator("plantgen.mask_move", text="", icon="TRIA_DOWN")
    down.direction, down.owner = "DOWN", "CLASS"
    if 0 <= k.filters_active < len(k.filters):
        _draw_mask_details(box, k.filters[k.filters_active], k.filters_active, s)
    box.label(text=f"Read it as Class > {k.name}", icon="INFO")


# ============================================================================
# TOOLING -- diagnostics and views, grouped at the bottom
# ============================================================================


class PLANTGEN_PT_tooling(Panel):
    """Debug Tools: the tools that inspect a plant rather than shape it, grouped
    so they collapse together out of the way, and shown only with the Debug
    Tools preference on. No plant requirement of its own, so Parameter Sweep
    (which makes plants rather than editing one) still shows with nothing
    selected; the plant-scoped tools poll individually."""

    bl_label = "Debug Tools"
    bl_idname = "PLANTGEN_PT_tooling"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PlantGen"
    bl_order = 70
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return operators.debug_tools(context)

    def draw(self, context):
        pass


class _ToolSubpanel(Panel):
    """A tool that acts on the active plant, nested under Tooling."""

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "PLANTGEN_PT_tooling"

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None


class PLANTGEN_PT_timings(_ToolSubpanel):
    """Where the active plant's last regrow spent its time, slowest first
    (`bridge.regrow_timings`), so a slow edit shows its own cause -- leaf
    collision on a dense plant, growth after a seed change. Live update's
    background growth / collision count from hand-off to landing."""

    bl_label = "Timings"
    bl_idname = "PLANTGEN_PT_timings"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        t = bridge.regrow_timings(_plant(context))
        if not t:
            layout.label(text="Regrow the plant to measure it", icon="INFO")
            return
        total = max(t["total"], 1e-9)
        layout.label(text=f"Last regrow: {total:.2f} s")
        col = layout.column(align=True)
        for name, sec in t["stages"]:
            if sec < 0.005:
                continue
            row = col.row(align=True)
            split = row.split(factor=0.6, align=True)
            split.label(text=name)
            split.label(text=f"{sec:.2f} s   {100.0 * sec / total:.0f}%")
        wrapped_label(layout, "Geometry Nodes then draws the plant: its time is the "
                              "viewport's Evaluation statistic, not counted here")


class PLANTGEN_PT_stats(_ToolSubpanel):
    """What the last grow actually produced, and whether it is structurally
    sound. Deliberately more than a node count: the counts that matter most are
    the ones no viewport overlay can show, because a bud that never released
    produces no geometry to colour."""

    bl_label = "Diagnostics"
    bl_idname = "PLANTGEN_PT_stats"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = _plant(context).plantgen
        layout = self.layout

        box = layout.box()
        box.label(text="Undo Trace")
        col = box.column(align=True)
        col.label(text="Clear, reproduce the problem, then Save.")
        row = col.row(align=True)
        row.operator("plantgen.clear_undo_log", icon="TRASH")
        row.operator("plantgen.save_undo_log", icon="TEXT")

        box = layout.box()
        box.label(text="Last Grow")
        col = box.column(align=True)
        col.label(text=f"Nodes: {s.last_phytomers}    Stems & branches: {s.last_axes}")
        col.label(text=f"Leaves: {s.last_leaves}    Flowers: {s.last_flowers}")
        if s.last_fruit:
            # Calendar phase 4: flowers that have set fruit (rendered by a Fruit
            # component, inside the calendar fruit window).
            col.label(text=f"Fruit set: {s.last_fruit}")
        if any(c.type == "bract" for c in s.components):
            # Its own row rather than folded into Leaves: bracts can silently
            # come out at zero -- only a Bract component placed On Organs grows
            # them, and only flowering region nodes carry them.
            col.label(text=f"Bracts: {s.last_bracts}")
            if s.last_bracts == 0:
                col.label(text="No bracts - needs a Bract component On Organs and a "
                               "flowering region", icon="INFO")
        # The flower asset and size: the primary flower component's collection and
        # Size, else the legacy plant ones. (Read `flower_size` here before the
        # fix -- a prop that never existed, so this row raised mid-draw.)
        fc = next((c for c in s.components if c.type == "flower"), None)
        f_col = fc.collection if fc is not None else s.flower_asset_collection
        f_size = float(fc.size) if fc is not None else float(s.flower_scale)
        if s.last_flowers and s.last_flower_spacing > 0.0 and f_col:
            # Flower Size and Flower Spacing are independent controls, so how
            # big a flower can get before it runs into its neighbour depends on
            # a number that lives somewhere else. Overlapping flowers merge
            # into the golden-angle spiral rows, which reads as them converging
            # on a few points rather than as being too large -- so the panel
            # says the two numbers rather than leaving it to be discovered.
            widths = [max(o.dimensions) for o in f_col.objects]
            if widths:
                wide = max(widths) * f_size
                gap = s.last_flower_spacing
                col.label(text=f"Flower {wide * 1000:.1f} mm, spaced {gap * 1000:.1f} mm")
                if wide > gap:
                    col.label(text=f"Flowers overlap {wide / gap:.1f}x - "
                                   "raise Flower Spacing or lower Flower Size",
                              icon="INFO")
        if s.inflorescence_kind == 'CATKIN' and s.catkin_length > 0.0:
            # Its own row, and the same reasoning as the bract row above: a
            # catkin count can silently be zero. Catkin Placement defaults to
            # branches only and the upper half, so a species asking for catkins
            # on an unbranched plant, or with the window closed, gets none --
            # with no error anywhere, since nothing is wrong.
            col.label(text=f"Catkins: {s.last_catkins}")
            if s.last_catkins == 0:
                col.label(text="No catkins - check Catkin Placement", icon="INFO")
        col.label(text=f"Height: {s.last_height:.3f} m    Width: {s.last_width:.3f} m")
        if s.obstacle_collection is not None:
            col.label(text=f"Obstacle voxels: {s.last_solid_voxels}")

        box = layout.box()
        box.label(text="Buds")
        col = box.column(align=True)
        col.label(text=f"Active: {s.last_buds_active}    Released: {s.last_buds_released}")
        col.label(text=f"Dormant: {s.last_buds_dormant}    Aborted: {s.last_buds_aborted}")

        # Senescence, shown only when a plant actually loses something -- a
        # SHED item draws no geometry, so these counts are the only sign of it.
        if s.last_dead_axes or s.last_shed_axes or s.last_shed_organs:
            box = layout.box()
            box.label(text="Senescence")
            col = box.column(align=True)
            col.label(text=f"Dead branches: {s.last_dead_axes}    "
                      f"Shed: {s.last_shed_axes}")
            col.label(text=f"Shed leaves/flowers: {s.last_shed_organs}")

        box = layout.box()
        box.label(text="By Branch Order")
        col = box.column(align=True)
        col.label(text=f"Stems: {s.last_axes_by_order or '-'}")
        col.label(text=f"Nodes: {s.last_phytomers_by_order or '-'}")
        col.label(text=f"Flowering axes: {s.last_flowering_axes} of {s.last_axes}")

        box = layout.box()
        row = box.row()
        row.label(text="Structural Checks")
        row.operator("plantgen.run_checks", text="Run", icon="PLAY")

        if not s.checks_ran:
            box.label(text="Not run since the last regrow", icon="DOT")
            return

        failed = sum(1 for c in s.checks if c.status == 'fail')
        col = box.column(align=True)
        for c in s.checks:
            icon = {'pass': "CHECKMARK", 'fail': "CANCEL"}.get(c.status, "DOT")
            row = col.row()
            row.alert = c.status == 'fail'
            row.label(text=f"{c.number}. {c.name}", icon=icon)
            # A failure's detail is the whole value of running these -- "max
            # order 17 <= 2" says what to go and look at. A pass needs no
            # explanation and would just make the panel unreadable.
            if c.status == 'fail':
                sub = col.row()
                sub.alert = True
                sub.label(text=f"      {c.detail}")
        if failed:
            box.label(text=f"{failed} failing", icon="ERROR")


class PLANTGEN_PT_inspect(_ToolSubpanel):
    """Point at a component, read what it is.

    The Geometry Nodes colour views answer "how does this quantity vary along
    the plant"; this answers "what is *this* thing" -- and it is the only place
    the developmental state can be read at all, because a dormant bud produces
    no geometry for any overlay to tint.

    Two controls, and both are categorical by design: what to draw and what the
    colours mean. Anything else about the overlay is a property of the plant,
    which is what the card in the viewport is for.
    """

    bl_label = "Inspector"
    bl_idname = "PLANTGEN_PT_inspect"

    def draw(self, context):
        layout = self.layout
        obj = _plant(context)
        s = obj.plantgen
        live_now = overlay.is_active(obj)

        layout.operator("plantgen.toggle_inspect",
                        text="Stop Inspecting" if live_now else "Inspect",
                        icon="ZOOM_SELECTED", depress=live_now)
        if not live_now:
            return

        col = layout.column(align=True)
        col.label(text="Show")
        col.row(align=True).prop(s, "inspect_show", expand=True)
        col.separator()
        col.prop(s, "inspect_color_by", text="Colour")

        box = layout.box()
        box.label(text="Point at a component. P pins its card.")
        pins = overlay.pinned(obj)
        if pins:
            # Mirrored here as well as in the viewport, because a pinned card
            # can be scrolled off screen while its component stays interesting.
            snap = overlay.snapshot_for(obj)
            for i in pins:
                if snap is None or i >= len(snap):
                    continue
                title, rows = overlay.describe(snap, i)
                sub = box.box()
                head = sub.row(align=True)
                head.label(text=title)
                # Flush right and unembossed, so it reads as a close cross on
                # the card rather than as another button in the panel -- the
                # same shape Blender's own list rows use for removing an entry.
                shut = head.row(align=True)
                shut.alignment = 'RIGHT'
                shut.operator("plantgen.unpin", text="", icon="X",
                              emboss=False).index = i
                grid = sub.column(align=True)
                for label, value in rows:
                    split = grid.split(factor=0.5)
                    split.label(text=label)
                    split.label(text=value)


class PLANTGEN_PT_debug(_ToolSubpanel):
    """Viewport-only attribute overlays. Grouped the way they are read rather
    than the way they are stored: the four export channels first, then the
    structural views that answer "why does the plant look like that"."""

    bl_label = "Debug View"
    bl_idname = "PLANTGEN_PT_debug"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = _plant(context).plantgen
        layout = self.layout

        active = s.debug_view_mode
        row = layout.row()
        row.operator("plantgen.toggle_debug_view", text="Off", icon="RESTRICT_VIEW_ON",
                     depress=(active == 'off')).mode = 'off'

        # A real object marker rather than a viewport tint, so it sits apart from
        # the colour views below. Shows where the plant's mass balances -- the
        # point every stem leans toward.
        markers = layout.box()
        markers.label(text="Markers")
        markers.prop(s, "show_center_of_mass", text="Center of Mass", icon="EMPTY_AXIS")

        box = layout.box()
        box.label(text="Components")
        col = box.column(align=True)
        col.operator("plantgen.toggle_debug_view", text="Component Colours",
                     depress=(active == 'component')).mode = 'component'

        box = layout.box()
        box.label(text="Structure")
        col = box.column(align=True)
        for mode in ('order', 'axis_id', 'node_index', 'birth_step', 'width', 'kind'):
            label = next(m[1] for m in properties.DEBUG_VIEW_MODES if m[0] == mode)
            col.operator("plantgen.toggle_debug_view", text=label,
                         depress=(active == mode)).mode = mode

        box = layout.box()
        box.label(text="Gravity")
        # The one beam that bends every organ, shown as its four ingredients: how
        # sharply it turns, and the load, stiffness and moment that decided it.
        col = box.column(align=True)
        for mode in ('curvature', 'load', 'stiffness', 'moment'):
            label = next(m[1] for m in properties.DEBUG_VIEW_MODES if m[0] == mode)
            col.operator("plantgen.toggle_debug_view", text=label,
                         depress=(active == mode)).mode = mode

        box = layout.box()
        box.label(text="Bloom")
        # The bloom views, whose controls live in the Flowers box rather than
        # beside a single field -- so they get a group here rather than an eye
        # button. Bloom Order vs Bloom Birth is the whole of how From Age holds a
        # flower shut until it exists; Organ Size is each organ's own expansion.
        col = box.column(align=True)
        for mode in ('bloom_order', 'bloom', 'bloom_birth', 'organ_scale'):
            label = next(m[1] for m in properties.DEBUG_VIEW_MODES if m[0] == mode)
            col.operator("plantgen.toggle_debug_view", text=label,
                         depress=(active == mode)).mode = mode

        box = layout.box()
        box.label(text="Phenology")
        # The calendar/lifecycle views. Their controls live on the Phenology
        # Calendar or the Senescence box, not beside a panel field, so like the
        # bloom views they get a group here. Ripeness is the fruit counterpart to
        # Autumn's foliage stage; Senescence is alive-vs-dead.
        col = box.column(align=True)
        for mode in ('dead', 'autumn', 'ripeness'):
            label = next(m[1] for m in properties.DEBUG_VIEW_MODES if m[0] == mode)
            col.operator("plantgen.toggle_debug_view", text=label,
                         depress=(active == mode)).mode = mode

        if active != 'off':
            layout.label(text="Debug tint is viewport-only", icon="INFO")


class PLANTGEN_PT_showcase(Panel):
    """Comparison grids, for judging a whole axis of the parameter surface at
    once. A tool that makes plants rather than editing one, so it needs no
    active plant -- which is why it (alone among the Tooling sub-panels) does
    not poll on one."""

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "PLANTGEN_PT_tooling"
    bl_label = "Parameter Sweep"
    bl_idname = "PLANTGEN_PT_showcase"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 30

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.prop(scene, "plantgen_showcase", text="")
        layout.prop(scene, "plantgen_showcase_steps")
        layout.operator("plantgen.showcase", icon="MESH_GRID")

        template = showcase.find_template(context)
        if template is None:
            layout.label(text="No plant to sweep from", icon="ERROR")
            return
        layout.label(text=f"Varying: {template.name}")
        # The count is worth showing before the click rather than after: an
        # enum ignores the Variations field and uses every state it has, so a
        # 9-entry dropdown makes nine plants however the field is set.
        prop = next((p for p in showcase.sweepable_properties()
                     if p.identifier == scene.plantgen_showcase), None)
        if prop is not None:
            values = showcase.sweep_values(
                prop, getattr(template.plantgen, prop.identifier),
                scene.plantgen_showcase_steps)
            layout.label(text=f"{len(values)} plants: {values[0][0]} to {values[-1][0]}")


class PLANTGEN_PT_docs(Panel):
    """Regenerate the LLM-facing parameter spec from the model. A build tool,
    not a plant operation, so -- like Parameter Sweep -- it needs no active
    plant and does not poll on one. Reads the model itself, so it cannot drift."""

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "PLANTGEN_PT_tooling"
    bl_label = "Documentation"
    bl_idname = "PLANTGEN_PT_docs"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 40

    def draw(self, context):
        layout = self.layout
        layout.operator("plantgen.generate_docs", icon="FILE_TEXT")
        layout.label(text="Rebuilds the spec and the HTML docs.", icon="INFO")
        layout.operator("plantgen.open_docs", icon="HELP")


# ============================================================================
# EFFECTOR -- a different object, so its own poll-gated top-level panel
# ============================================================================


class PLANTGEN_PT_effector(Panel):
    """The selected effector's own settings.

    A separate top-level panel rather than a row in the plant's Environment
    box, because an effector is a different object: selecting it deselects the
    plant, so there is no plant panel on screen to put it in. It sits in the
    same tab, so the two read as one tool.
    """

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PlantGen"
    bl_label = "Plant Effector"
    bl_idname = "PLANTGEN_PT_effector"
    bl_order = 5

    @classmethod
    def poll(cls, context):
        return effectors.is_effector(context.active_object)

    def draw(self, context):
        obj = context.active_object
        e = obj.plantgen_effector
        layout = self.layout

        layout.prop(e, "kind", text="")
        col = layout.column(align=True)
        row = col.row()
        row.prop(e, "radius")
        # Target reaches everywhere, so a radius on screen would be a lie about
        # what the drawn gizmo means.
        row.active = effectors.uses_radius(e.kind)
        col.prop(e, "strength", slider=True)

        if effectors.uses_axis(e.kind):
            # These three read the Empty's rotation as well as its position.
            # The gizmo already shows the axis, but nothing about a rotation
            # widget says it is wired to anything, so say it.
            layout.label(text="Rotate this empty to aim it")

        owners = [o.name for o in bpy.data.objects
                  if bridge.is_plant(o) and o.plantgen.effector_collection is not None
                  and obj.name in o.plantgen.effector_collection.all_objects]
        if owners:
            layout.label(text="Steers: " + ", ".join(owners))
        else:
            # The one failure mode this feature has: an effector nothing reads.
            # It is invisible without saying so, since the Empty looks the same
            # either way.
            layout.label(text="No plant reads this effector", icon="ERROR")


# Parents before children: a `bl_parent_id` is a registration dependency, so
# Tooling must precede its tools.
_classes = (
    PLANTGEN_MT_preset_library,
    PLANTGEN_MT_mask_add,
    PLANTGEN_MT_level_mask_add,
    PLANTGEN_UL_level_mask,
    PLANTGEN_UL_components,
    PLANTGEN_UL_mask,
    PLANTGEN_UL_maps,
    PLANTGEN_UL_named,
    PLANTGEN_UL_class_mask,
    PLANTGEN_UL_branch_levels,
    PLANTGEN_MT_component_size_shapes,
    PLANTGEN_PT_component_size_curve,
    PLANTGEN_PT_plant,
    PLANTGEN_PT_stem,
    PLANTGEN_UL_structure, PLANTGEN_MT_component_new_placed,
    PLANTGEN_MT_structure_add, PLANTGEN_MT_structure_move_to,
    PLANTGEN_MT_structure_specials,
    PLANTGEN_PT_structure,
    PLANTGEN_UL_map_rows, PLANTGEN_MT_map_curve_shapes, PLANTGEN_PT_mapping,
    PLANTGEN_PT_texturing,
    PLANTGEN_PT_tooling,
    PLANTGEN_PT_timings,
    PLANTGEN_PT_stats,
    PLANTGEN_PT_inspect,
    PLANTGEN_PT_debug,
    PLANTGEN_PT_showcase,
    PLANTGEN_PT_docs,
    PLANTGEN_PT_effector,
)


def register():
    _load_tree_icons()
    for cls in (*_curve_classes, *_component_curve_classes, *_classes):
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed((*_curve_classes, *_component_curve_classes, *_classes)):
        bpy.utils.unregister_class(cls)
    _free_tree_icons()
    from . import map_icons
    map_icons.unregister()
    plant_icons.unregister()
