"""Texturing's Materials list: every material the plant wears, in one flat
list like Blender's own material slots -- a row per branching level (its slot:
the material it wears, or greyed the one it inherits), then every material
the leaf / bract / flower / bud / fruit
components' meshes wear, each once. The selected row's editor follows below:
a level's stem material box, or an asset material's preset and Textures &
Settings.

The asset meshes are instanced, so those are their own materials (the leaf
shader: the leaf tool's and FlowerGen's alike). A material is often shared --
every mesh, component and plant wearing it changes with an edit.

The rows are `s.texture_rows` (a UIList shows one collection, and this one
mixes levels and materials): UI state, kept out of presets, rebuilt from the
plant (`rebuild_rows`). Draw cannot write, so when the list no longer matches
the plant -- a level or component added, a mesh given another material -- the
draw asks a one-shot timer to rebuild it."""

from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Menu, Operator, UIList

from . import leaf_material, stem_material

_TYPES = ("leaf", "bract", "flower", "bud", "fruit")
_PENDING: set = set()
"""Plants (by name) a rebuild is already queued for."""


def asset_materials(s) -> list:
    """[(component, [(material, n meshes wearing it)])] for every leaf / bract /
    flower / bud / fruit component with a collection, materials in first-seen
    order."""
    out = []
    for c in s.components:
        if c.type not in _TYPES or c.collection is None:
            continue
        seen: dict = {}
        for o in c.collection.all_objects:
            for slot in getattr(o, "material_slots", ()):
                mat = slot.material
                if mat is not None:
                    seen[mat] = seen.get(mat, 0) + 1
        out.append((c, list(seen.items())))
    return out


def _wanted(s) -> list:
    """The rows the list should hold: (kind, uid, material, count) -- each
    level's slot, then every asset material once, in first-seen order."""
    rows = [("LEVEL", lv.uid, None, 0) for lv in s.branch_levels]
    seen = set()
    for _c, mats in asset_materials(s):
        for mat, _n in mats:
            if mat.name not in seen:
                seen.add(mat.name)
                rows.append(("MATERIAL", 0, mat, 0))
    return rows


def _row_key(r) -> tuple:
    return (r.kind, r.uid, r.material, r.count)


def in_sync(s) -> bool:
    return [_row_key(r) for r in s.texture_rows] == _wanted(s)


def rebuild_rows(s) -> None:
    """Rewrite the rows from the plant, keeping the selection on the same row
    (or the first level). Never from draw."""
    keep = None
    if 0 <= s.texture_rows_active < len(s.texture_rows):
        r = s.texture_rows[s.texture_rows_active]
        keep = (r.kind, r.uid, r.material)
    s.texture_rows.clear()
    active = 0
    for i, (kind, uid, mat, n) in enumerate(_wanted(s)):
        row = s.texture_rows.add()
        row.kind, row.uid, row.material, row.count = kind, uid, mat, n
        if keep == (kind, uid, mat):
            active = i
    s.texture_rows_active = active


def _request_rebuild(obj) -> None:
    """Ask a timer to rebuild the plant's rows (draw cannot write)."""
    name = obj.name
    if name in _PENDING:
        return
    _PENDING.add(name)

    def run():
        _PENDING.discard(name)
        try:          # a timer that raises is silently unregistered
            o = bpy.data.objects.get(name)
            if o is not None and not in_sync(o.plantgen):
                rebuild_rows(o.plantgen)
                for win in bpy.context.window_manager.windows:
                    for area in win.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
        return None

    bpy.app.timers.register(run, first_interval=0.0)


class PLANTGEN_UL_texture_rows(UIList):
    """The Materials list, drawn like Blender's material slots: each row a
    material's preview and name. A level's row is its slot (an inherited
    material greyed); then the asset materials."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        s = data
        if item.kind == "LEVEL":
            levels = [lv.uid for lv in s.branch_levels]
            if item.uid not in levels:
                return
            i = levels.index(item.uid)
            lv = s.branch_levels[i]
            mat = lv.material
            inherited = mat is None
            if inherited:
                worn = stem_material.level_materials(s)
                mat = worn[i] if i < len(worn) else None
        else:
            mat, inherited = item.material, False
        if self.layout_type not in {'DEFAULT', 'COMPACT'}:
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=layout.icon(mat) if mat else 0)
            return
        row = layout.row(align=True)
        name = row.row(align=True)
        name.active = not inherited
        if mat is not None:
            name.label(text=mat.name, icon_value=layout.icon(mat))
        else:
            name.label(text="none", icon='MATERIAL')


class PLANTGEN_OT_asset_material_preset(Operator):
    """Set this material's shader settings to a leaf shader preset (every mesh
    wearing it changes)"""

    bl_idname = "plantgen.asset_material_preset"
    bl_label = "Leaf Shader Preset"
    bl_options = {"REGISTER", "UNDO"}

    material: StringProperty(options={"HIDDEN"})
    preset: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        for key, _label, desc, _v in leaf_material.PRESETS:
            if key == properties.preset:
                return desc + " (every mesh wearing the material changes)"
        return cls.__doc__

    def execute(self, context):
        mat = bpy.data.materials.get(self.material)
        if not leaf_material.is_leaf_material(mat) or not leaf_material.apply_preset(
                mat, self.preset):
            self.report({'ERROR'}, "Not a leaf-shader material")
            return {'CANCELLED'}
        return {"FINISHED"}


class PLANTGEN_MT_asset_material_presets(Menu):
    """The leaf shader presets, for the material the menu was opened on
    (`context.material`, set by `context_pointer_set`)."""

    bl_idname = "PLANTGEN_MT_asset_material_presets"
    bl_label = "Leaf Shader Presets"

    def draw(self, context):
        mat = getattr(context, "material", None)
        if mat is None:
            return
        for key, label, _desc, _v in leaf_material.PRESETS:
            op = self.layout.operator("plantgen.asset_material_preset", text=label)
            op.material, op.preset = mat.name, key


def draw(layout, obj) -> None:
    """The Materials list, then the selected row's editor."""
    s = obj.plantgen
    if not len(s.branch_levels):
        layout.label(text="Regrow to set up the levels", icon="INFO")
        return
    if not in_sync(s):
        _request_rebuild(obj)
    layout.template_list("PLANTGEN_UL_texture_rows", "", s, "texture_rows",
                         s, "texture_rows_active", rows=5)
    i = s.texture_rows_active
    if not 0 <= i < len(s.texture_rows):
        return
    r = s.texture_rows[i]
    if r.kind == "LEVEL":
        lv = next((x for x in s.branch_levels if x.uid == r.uid), None)
        if lv is None:
            return
        parent = next((x for x in s.branch_levels if x.uid == lv.parent_uid), None)
        stem_material.draw_level_material(layout, lv, parent.name if parent else "")
    elif r.kind == "MATERIAL" and r.material is not None:
        box = layout.box()
        box.label(text=r.material.name)
        draw_editor(box, r.material)


def draw_editor(layout, mat) -> None:
    """A leaf-shader material's editor -- its Preset, then Textures & Settings
    in a closed fold. The one editor for every place that edits one: Texturing,
    FlowerGen's Material and the leaf tool's (each draws its own picker above).
    The preset menu reads the material from the layout (`context_pointer_set`),
    so it acts on this material wherever it is drawn."""
    if not leaf_material.is_leaf_material(mat):
        layout.label(text="Not a PlantGen leaf material", icon="INFO")
        return
    node = leaf_material.group_node(mat)
    if node is None:
        layout.label(text="Leaf shader node missing", icon="ERROR")
        return
    row = layout.row(align=True)
    row.label(text="Preset")
    match = leaf_material.matching_preset(mat)
    row.context_pointer_set("material", mat)
    row.menu("PLANTGEN_MT_asset_material_presets",
             text=match[1] if match else "Custom", icon="PRESET")
    head, body = layout.panel("plantgen_asset_material_settings", default_closed=True)
    head.label(text="Textures & Settings")
    if body is None:
        return
    # The texture slots are the leaf tool's (`Material.plantgen_leaf`, registered
    # with it): absent while that optional tool is not loaded.
    pg = getattr(mat, "plantgen_leaf", None)
    col = body.column(align=True)
    if pg is None:
        col.label(text="Texture slots need the leaf tool", icon="INFO")
    for prop, *_rest in leaf_material.TEXTURE_SLOTS if pg is not None else ():
        split = col.split(factor=0.32, align=True)
        split.label(text=pg.bl_rna.properties[prop].name)
        split.template_ID(pg, prop, open="image.open")
    if pg is not None and pg.normal_image is not None:
        nmap = leaf_material.normal_map_node(mat)
        if nmap is not None:
            col.prop(nmap.inputs["Strength"], "default_value", text="Normal Strength")
    col = body.column(align=True)
    thin = node.inputs.get("Thin Wall")
    thin_on = bool(thin.default_value) if thin is not None else True
    for name in leaf_material.SETTINGS:
        sock = node.inputs.get(name)
        if sock is None or sock.is_linked or (name in leaf_material.SOLID_ONLY and thin_on):
            continue
        label = "Subsurface" if name == "Translucency" and not thin_on else name
        col.prop(sock, "default_value", text=label)
        if name == "Thin Wall" and thin_on:
            # EEVEE (and Material Preview) passes a fraction of the light Cycles
            # does through a thin wall: backlit, ~0.026 against 0.148 (2026-10-18).
            col.label(text="Light passes through fully in Cycles only", icon="INFO")


_classes = (PLANTGEN_UL_texture_rows, PLANTGEN_OT_asset_material_preset,
            PLANTGEN_MT_asset_material_presets)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
