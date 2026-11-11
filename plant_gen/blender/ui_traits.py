"""The Traits sheet: a spreadsheet per component, in its settings box (the
Components panel) -- a row per mesh of its collection, a column per point
property -- deciding which mesh each point takes (`growth/traits.py`; the
pick in `instancing._pick_assets`).

Each row has a Weight (how common the mesh is) and, per column, a preference
-1..1: +1 the mesh is taken where the property is high, -1 where it is low, 0
indifferent. Stored on the component, so the same mesh can lean differently in
another plant or component. A sheet at its defaults (Weights equal,
preferences 0) is the plain uniform pick, byte-identical.

Leaf, bract, bud and fruit components only: a flower component's collection is
its bloom sequence (bud to open), one stage per mesh, not variants to choose
between."""

from __future__ import annotations

import bpy
from bpy.props import IntProperty
from bpy.types import Operator

from . import bridge, properties, ui
from .operators import _redraw, _undo_step

_CHANNEL_NAMES = {i: n for i, n, _d in properties.MASK_CHANNEL_ITEMS}


def _component(context):
    """The plant and its selected component (the Components tree's selection)."""
    obj = bridge.resolve_plant(context)
    if obj is None:
        return None, None
    s = obj.plantgen
    if 0 <= s.components_active < len(s.components):
        return obj, s.components[s.components_active]
    return obj, None


def _meshes(c) -> list:
    return list(c.collection.objects) if c.collection is not None else []


def _in_sync(c) -> bool:
    return [r.asset for r in c.trait_rows] == _meshes(c)


def _sync_rows(c) -> None:
    """Rows = the collection's meshes, in its order: a new mesh gets Weight 1 and
    preferences 0, a row whose mesh left the collection goes, the rest keep
    their values."""
    ncol = len(c.trait_columns)
    kept = {}
    for r in c.trait_rows:
        if r.asset is not None:
            kept[r.asset.name] = (r.weight, [cell.value for cell in r.cells])
    c.trait_rows.clear()
    for o in _meshes(c):
        row = c.trait_rows.add()
        row.asset = o
        weight, cells = kept.get(o.name, (1.0, []))
        row.weight = weight
        for ci in range(ncol):
            row.cells.add().value = cells[ci] if ci < len(cells) else 0.0


def _column_label(col) -> str:
    if col.channel == "custom":
        return col.attr_name or "Attribute"
    if col.channel == "class":
        return col.class_name or "Class"
    return _CHANNEL_NAMES.get(col.channel, col.channel)


class _TraitOp(Operator):
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj, c = _component(context)
        return c is not None and c.type != "flower" and c.collection is not None

    def _apply(self, c) -> None:
        raise NotImplementedError

    message = ""

    def execute(self, context):
        obj, c = _component(context)
        with _undo_step(self.message):
            self._apply(c)
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_trait_rows_update(_TraitOp):
    """Match the sheet's rows to the component's collection"""

    bl_idname = "plantgen.trait_rows_update"
    bl_label = "Update Rows"
    bl_description = ("One row per mesh of the component's collection: new meshes get "
                      "Weight 1, rows of meshes no longer in it go")
    message = "Update trait rows"

    def _apply(self, c):
        _sync_rows(c)


class PLANTGEN_OT_trait_column_add(_TraitOp):
    """Add a property column to the Traits sheet"""

    bl_idname = "plantgen.trait_column_add"
    bl_label = "Add Property"
    bl_description = "Add a property column: each mesh's preference for where it is high or low"
    message = "Add trait column"

    def _apply(self, c):
        used = {col.channel for col in c.trait_columns}
        col = c.trait_columns.add()
        col.channel = next((ch for ch in ("age", "height", "shade", "branch_depth")
                            if ch not in used), "age")
        _sync_rows(c)


class PLANTGEN_OT_trait_column_remove(_TraitOp):
    """Remove a property column from the Traits sheet"""

    bl_idname = "plantgen.trait_column_remove"
    bl_label = "Remove Property"
    bl_description = "Remove this property column and its preferences"
    message = "Remove trait column"

    index: IntProperty(default=-1, options={'HIDDEN'})

    def _apply(self, c):
        if not 0 <= self.index < len(c.trait_columns):
            return
        c.trait_columns.remove(self.index)
        for r in c.trait_rows:
            if self.index < len(r.cells):
                r.cells.remove(self.index)


class PLANTGEN_OT_trait_sheet_clear(_TraitOp):
    """Clear the Traits sheet: back to the plain uniform pick"""

    bl_idname = "plantgen.trait_sheet_clear"
    bl_label = "Clear Sheet"
    bl_description = "Remove every column and row: the component picks its meshes uniformly again"
    message = "Clear traits sheet"

    def _apply(self, c):
        c.trait_columns.clear()
        c.trait_rows.clear()


def draw_sheet(layout, s, c) -> None:
    """The component's Traits sheet, drawn in its settings box (Components
    panel) -- the sheet edits the selected component, so it sits with the rest
    of it. Not for a flower component (its meshes are bloom stages)."""
    meshes = _meshes(c)
    if c.collection is None or not meshes:
        layout.label(text="Give the component a collection of meshes first", icon="INFO")
        return
    if not len(c.trait_rows):
        col = layout.column(align=True)
        col.label(text=f"{len(meshes)} meshes, picked at random.")
        col.label(text="Add a property to lean each mesh toward where")
        col.label(text="it is high or low (Age, Height, Shade...).")
        row = layout.row(align=True)
        row.operator("plantgen.trait_column_add", icon="ADD")
        row.operator("plantgen.trait_rows_update", text="Weights Only", icon="LINENUMBERS_ON")
        return
    if not _in_sync(c):
        warn = layout.row(align=True)
        warn.alert = True
        warn.label(text="The collection changed", icon="ERROR")
        warn.operator("plantgen.trait_rows_update", icon="FILE_REFRESH")

    # The columns: which property each one reads.
    cols = layout.column(align=True)
    for i, col in enumerate(c.trait_columns):
        row = cols.row(align=True)
        ui._draw_channel_ref(row, col, s, text="")
        row.operator("plantgen.trait_column_remove", text="", icon="X").index = i
    cols.operator("plantgen.trait_column_add", icon="ADD")

    # The sheet.
    sheet = layout.box().column(align=True)
    head = sheet.split(factor=0.36, align=True)
    head.label(text="Mesh")
    cells = head.row(align=True)
    cells.label(text="Weight")
    for col in c.trait_columns:
        cells.label(text=_column_label(col))
    ncol = len(c.trait_columns)
    for r in c.trait_rows:
        line = sheet.split(factor=0.36, align=True)
        name = r.asset.name if r.asset is not None else "(missing)"
        line.label(text=name, icon="OBJECT_DATA" if r.asset is not None else "ERROR")
        vals = line.row(align=True)
        vals.prop(r, "weight", text="")
        for cell in list(r.cells)[:ncol]:
            vals.prop(cell, "value", text="", slider=True)
    foot = layout.row()
    foot.label(text="+1 favoured where high, -1 where low")
    foot.operator("plantgen.trait_sheet_clear", text="", icon="TRASH")


_classes = (PLANTGEN_OT_trait_rows_update, PLANTGEN_OT_trait_column_add,
            PLANTGEN_OT_trait_column_remove, PLANTGEN_OT_trait_sheet_clear)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
