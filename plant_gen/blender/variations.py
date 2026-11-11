"""Variations: a row or a grid of copies of a plant, each with its own seed,
one or two plant-wide parameters stepped across them (Day +36 per plant...),
and Bake: each one to a single plain mesh.

The source plant stays the master. Its copies live in their own collection and
are generated, never edited: Build throws the old set away and copies the
source again, so a tweak to the source reaches every variation by rebuilding.
A copy is a full object copy -- every level, component, placement, mask and map
comes along with the plant's own settings -- with its own mesh and organs
object; the curve group is shared (the curves are the source's, held constant
across the set). Settings live on the source (`s.variations`).
"""

from __future__ import annotations

import math

import bpy

from . import bridge

MARKER = "plantgen_variation"
"""On every generated object: the name of the plant it was copied from. A
marked plant is never a source, and its own panel says it is generated."""

_LABEL_HEIGHT = 0.05

_SKIP = {"seed", "day_display"}

PLANT_SIZE = "plant_size"
"""A composite axis, not a setting: at factor f the clump is f times as wide
(Basal Spread), every crown level (one growing from the plant itself --
culms, tillers) f times as long with f^2 as many stems (the same stems per
ground area: a wider tussock, not a denser one), and its leaves f times as
large -- a tuft's footprint is mostly its blades (a species' blade length
spans a range: Alopecurus 10-30 cm), so with leaves held the first build read
as one size throughout. Flowers keep their size (user decision 2026-10-21)."""


def _apply_size(s, f: float) -> None:
    s.basal_spread = s.basal_spread * f
    for c in s.components:
        if c.type in ("leaf", "bract"):
            c.size = c.size * f
    for lv in s.branch_levels:
        if lv.parent_uid != 0:
            continue
        lv.whorl = max(1, int(round(lv.whorl * f * f)))
        lv.start_length_scale = lv.start_length_scale * f


def parameter_items():
    """(identifier, name) of every plant-wide number a variation axis can step:
    the plant's own numeric settings (the Seed is stepped on its own), and
    Plant Size (`PLANT_SIZE`)."""
    rna = bpy.types.Object.bl_rna.properties["plantgen"].fixed_type
    out = [(PLANT_SIZE, "Plant Size")]
    for p in rna.properties:
        if p.is_readonly or p.type not in {'INT', 'FLOAT'} or p.identifier in _SKIP:
            continue
        if getattr(p, "array_length", 0) > 1 or getattr(p, "is_hidden", False):
            continue
        if p.identifier.startswith(("last_", "classify_", "cal_col_", "mask_preview",
                                    "structure_", "tree_", "map_rows", "debug_")):
            continue
        if p.identifier.endswith(("_active", "_authored", "_schema")):
            continue
        out.append((p.identifier, p.name or p.identifier))
    return out


def source_of(obj):
    """The plant a variation was copied from (itself if it is a source)."""
    if obj is None:
        return None
    name = obj.get(MARKER)
    if isinstance(name, str) and name in bpy.data.objects:
        return bpy.data.objects[name]
    return obj


def param_name(ident: str) -> str:
    if not ident:
        return "None"
    if ident == PLANT_SIZE:
        return "Plant Size"
    rna = bpy.types.Object.bl_rna.properties["plantgen"].fixed_type
    prop = rna.properties.get(ident)
    return prop.name if prop is not None else ident


def axis_value(source_s, ident: str, start: float, step: float, index: int):
    """The value axis `ident` takes at `index`: start + index * step, clamped to
    the property's own limits, an integer rounded."""
    if ident == PLANT_SIZE:
        return max(0.05, float(start) + float(step) * index)
    rna = bpy.types.Object.bl_rna.properties["plantgen"].fixed_type
    prop = rna.properties[ident]
    value = float(start) + float(step) * index
    value = min(max(value, float(prop.hard_min)), float(prop.hard_max))
    if prop.type == 'INT':
        return int(round(value))
    return value


def collection_name(source) -> str:
    return f"PGS_Variations_{source.name}"


def generated(source) -> list:
    """The variations currently built from `source`."""
    col = bpy.data.collections.get(collection_name(source))
    if col is None:
        return []
    return [o for o in col.objects if o.get(MARKER) == source.name and bridge.is_plant(o)]


def clear(source) -> int:
    """Delete `source`'s variations: the plants, their organs objects, labels
    and meshes, then the collection. The count of plants removed."""
    col = bpy.data.collections.get(collection_name(source))
    if col is None:
        return 0
    n = 0
    for obj in list(col.all_objects):
        data = obj.data
        if bridge.is_plant(obj):
            n += 1
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)
    bpy.data.collections.remove(col)
    return n


def _collection(source):
    col = bpy.data.collections.new(collection_name(source))
    parent = source.users_collection[0] if source.users_collection else \
        bpy.context.scene.collection
    parent.children.link(col)
    return col


def _copy(source, col, name: str):
    """A full copy of the source plant in `col`: its own mesh, no organs object
    or GN modifier yet (the first regrow makes its own), live update off (the
    shared curve group would otherwise regrow every copy on each source curve
    edit)."""
    obj = source.copy()
    obj.data = source.data.copy()
    for key in list(obj.keys()):
        if key.startswith(("plantgen_organs", "plantgen_stem_preview",
                           "plantgen_weld")):
            del obj[key]
    obj.modifiers.clear()
    obj.parent = None
    obj[MARKER] = source.name
    obj.name = name
    col.objects.link(obj)
    obj.plantgen.live_update = False
    return obj


def _label(text: str, col, size: float, location) -> None:
    curve = bpy.data.curves.new("PGS_VariationLabel", type='FONT')
    curve.body = text
    curve.size = size
    curve.align_x = 'CENTER'
    obj = bpy.data.objects.new("PGS_VariationLabel", curve)
    obj[MARKER] = "label"
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)   # reads from the front
    obj.location = location
    col.objects.link(obj)


def _caption(value) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.3g}"


def _extent(obj, depsgraph) -> tuple[float, float]:
    """The plant's footprint width and height as rendered: its evaluated bound
    box AND every GN instance's own (flowers, fruit) -- the evaluated box alone
    leaves the instances out and read half the real width on the user's plant."""
    from mathutils import Vector

    points = [obj.matrix_world @ Vector(c) for c in obj.evaluated_get(depsgraph).bound_box]
    for inst in depsgraph.object_instances:
        if inst.is_instance and inst.parent is not None and inst.parent.original == obj:
            m = inst.matrix_world
            points += [m @ Vector(c) for c in inst.object.bound_box]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys)), max(zs) - min(zs)


def build(source, progress=None) -> tuple[int, str]:
    """Rebuild `source`'s variations from its settings. (plant count, note)."""
    v = source.plantgen.variations
    from .properties import sync_variation_starts
    sync_variation_starts(v)
    nx = max(1, int(v.count_x))
    ny = max(1, int(v.count_y)) if v.layout == 'GRID' else 1
    axes = [(v.x_param, v.x_start, v.x_step), (v.y_param, v.y_start, v.y_step)]
    clear(source)
    col = _collection(source)
    base_seed, seed_step = int(source.plantgen.seed), max(1, int(v.seed_step))
    weld = bridge.WELD_NOW if v.weld else bridge.WELD_LATER
    plants = []
    total = nx * ny
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            obj = _copy(source, col, f"{source.name}_var_{k + 1:03d}")
            s = obj.plantgen
            # every plant its own seed, Seed Step apart (none the source's own)
            s.seed = base_seed + (k + 1) * seed_step
            caption = []
            for axis, (ident, start, step) in enumerate(axes):
                if not ident or (axis == 1 and v.layout != 'GRID'):
                    continue
                value = axis_value(source.plantgen, ident, start, step,
                                   i if axis == 0 else j)
                if ident == PLANT_SIZE:
                    _apply_size(s, value)
                    caption.append(f"Size {value:.2f}x")
                    continue
                setattr(s, ident, value)
                caption.append(f"{param_name(ident)} {_caption(value)}")
            caption.append(f"seed {s.seed}")
            bridge.regrow(obj, weld=weld)
            plants.append((obj, i, j, "  ".join(caption)))
            if progress is not None:
                progress(k + 1, total)
    _lay_out(source, v, plants, col)
    return len(plants), ""


def baked_name(source) -> str:
    return f"PGS_Variations_{source.name}_Baked"


def _layer_collection(layer, name: str):
    if layer.collection.name == name:
        return layer
    for child in layer.children:
        found = _layer_collection(child, name)
        if found is not None:
            return found
    return None


def bake(source, progress=None) -> int:
    """Bake every variation to one plain mesh object: welded (when its plant
    welds), every leaf, flower and fruit realized -- no modifier, no organs
    object -- in its own collection, at its plant's place. The live variations
    are hidden, not deleted (Rebuild / Clear still work on them). Count baked."""
    from .rig import _realize_full

    plants = generated(source)
    old = bpy.data.collections.get(baked_name(source))
    if old is not None:
        for obj in list(old.all_objects):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if isinstance(data, bpy.types.Mesh) and data.users == 0:
                bpy.data.meshes.remove(data)
        bpy.data.collections.remove(old)
    if not plants:
        return 0
    col = bpy.data.collections.new(baked_name(source))
    parent = source.users_collection[0] if source.users_collection else \
        bpy.context.scene.collection
    parent.children.link(col)
    for k, obj in enumerate(plants):
        if obj.plantgen.stem_weld and obj.get("plantgen_stem_preview"):
            bridge.regrow(obj, weld=bridge.WELD_NOW)      # never bake the preview tubes
        mesh = _realize_full(obj)
        mesh.name = f"{obj.name}_baked"
        out = bpy.data.objects.new(f"{obj.name}_baked", mesh)
        out.matrix_world = obj.matrix_world.copy()
        out[MARKER] = f"{source.name}:baked"
        col.objects.link(out)
        if progress is not None:
            progress(k + 1, len(plants))
    live = _layer_collection(bpy.context.view_layer.layer_collection, collection_name(source))
    if live is not None:
        live.hide_viewport = True
    return len(plants)


def _lay_out(source, v, plants, col) -> None:
    """To the right of the source, row by row away from the camera: the given
    spacing, or (Auto) the widest plant's footprint with a quarter to spare."""
    if not plants:
        return
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sizes = [_extent(obj, depsgraph) for obj, *_ in plants]
    widest = max(max(w for w, _ in sizes), 1e-3)
    tallest = max(max(h for _, h in sizes), 1e-3)
    sx = widest * 1.25 if v.auto_spacing else float(v.spacing_x)
    sy = widest * 1.25 if v.auto_spacing else float(v.spacing_y)
    ox, oy, oz = source.matrix_world.translation
    size = tallest * _LABEL_HEIGHT
    for obj, i, j, caption in plants:
        x, y = ox + (i + 1) * sx, oy + j * sy
        obj.location = (x, y, oz)
        if v.labels:
            _label(caption, col, size, (x, y, oz - size * 1.6))
