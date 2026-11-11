"""The whole plant, for a 1-to-1 preset: every setting on `obj.plantgen`
(levels, components, placements, masks, every kind of map, classes, custom
attributes, mutes, variations...), every curve on the plant, and the assets
those settings point at, bundled in a .blend beside the preset.

`dump` reads the settings generically off the RNA (so a new setting is saved
without touching this file) minus view state and per-scene links; `restore`
writes them back as stored values -- no update callbacks fire, so nothing a
setting's update would stamp (kind defaults, a map preset's curve) can
overwrite what the preset says -- then the derived state (tree, map rows,
icons) is rebuilt and the caller regrows. Curves go into a fresh curve group of
the plant's own, so a plant that shared one (a variation) never edits another.

Assets (component collections, level materials) are written with
`bpy.data.libraries.write`, which carries everything they use (meshes, their
materials, node groups); images are packed into that file for the write and
unpacked back to their own files after, so the preset folder is complete and
the user's file is left as it was. Loading appends only what the current file
lacks, by name (an asset of that name already here is reused as is).
"""

from __future__ import annotations

import os
from pathlib import Path

import bpy

FORMAT = 1
"""The plant section's own format (beside the profile's schema_version)."""

ASSET_HOLDER = "PlantGen Preset Assets"
"""The collection appended preset assets are linked under (hidden), so the
objects they hold are in the scene for the plant to instance."""

_EXCLUDE_TOP = frozenset({
    # what the profile / the preset envelope carries
    "species", "preset_id", "preset_name", "preset_embedded",
    # view / runtime state
    "name", "checks", "checks_ran", "debug_view_mode", "inspect_show", "inspect_color_by",
    "inspect_active", "inspect_pinned", "show_center_of_mass", "live_update",
    "wind_debug", "wind_debug_res", "wind_skin_debug",
    "rig_view", "rig_show_bones", "rig_bone_display", "rig_fast_preview",
    "components_active", "growth_maps_active", "map_rows", "map_rows_active", "map_folds",
    "mapping_details_open", "branch_levels_active", "structure_focus",
    "structure_parent_uid", "structure_place", "tree_rows", "tree_active",
    "structure_plant_expanded", "structure_level_open", "structure_component_open",
    "mask_preview_level", "custom_attrs_active", "classes_active",
    "texture_rows", "texture_rows_active",
    # this scene's, not the plant's
    "sun_object", "obstacle_collection", "effector_collection", "ground_object",
    # saved as curves, below
    "curve_group",
})
_EXCLUDE_NESTED = frozenset({"sun_object"})
"""A component's own scene links -- saved no more than the plant's."""
_EXCLUDE_PREFIX = ("last_", "classify_")
_EXCLUDE_ANY = frozenset({"rna_type", "label", "probability"})
"""label, a mask entry's probability: computed (get / set) from what is saved."""

_ID_DATA = {"Collection": "collections", "Material": "materials", "Object": "objects",
            "Image": "images", "NodeTree": "node_groups"}
_BUNDLED = ("Collection", "Material")
"""The datablocks a preset carries in its .blend (objects are scene things)."""


# --- settings -----------------------------------------------------------------

def _dump(pg, top: bool = False) -> dict:
    out = {}
    for prop in pg.bl_rna.properties:
        key = prop.identifier
        if key in _EXCLUDE_ANY or (top and (key in _EXCLUDE_TOP
                                            or key.startswith(_EXCLUDE_PREFIX))) \
                or (not top and key in _EXCLUDE_NESTED):
            continue
        value = getattr(pg, key)
        if prop.type == 'COLLECTION':
            out[key] = [_dump(item) for item in value]
        elif prop.type == 'POINTER':
            kind = prop.fixed_type.identifier
            if kind in _ID_DATA:
                out[key] = None if value is None else {"id": kind, "name": value.name}
            elif value is not None:
                out[key] = _dump(value)
        elif prop.is_readonly:
            continue
        elif prop.type == 'ENUM':
            out[key] = sorted(value) if prop.is_enum_flag else value
        elif getattr(prop, "is_array", False):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _resolve(ref):
    if not isinstance(ref, dict):
        return None
    store = getattr(bpy.data, _ID_DATA.get(ref.get("id", ""), ""), None)
    return store.get(ref.get("name", "")) if store is not None else None


_ENUM_RENAMES = {
    # a level map's Steps target became Growth % (its stored number kept)
    ("PLANTGEN_PG_level_map", "target"): {"steps": "growth"},
}
"""Enum items renamed since a preset was saved: (property group, property)
-> {old identifier: new}."""


def _assign(pg, prop, value) -> None:
    """Store `value` without firing the property's update."""
    key = prop.identifier
    try:
        if prop.type == 'ENUM':
            if isinstance(value, str):
                value = _ENUM_RENAMES.get((type(pg).__name__, key), {}).get(value, value)
            items = {e.identifier: e.value for e in prop.enum_items}
            if not items:                       # a dynamic enum: through RNA
                setattr(pg, key, set(value) if prop.is_enum_flag else value)
                return
            if prop.is_enum_flag:
                pg[key] = sum(items[v] for v in value if v in items)
            elif value in items:
                pg[key] = items[value]
            return
        pg[key] = value
    except (TypeError, ValueError, KeyError, AttributeError):
        try:
            setattr(pg, key, value)
        except (TypeError, ValueError, AttributeError):
            pass


def _restore(pg, data: dict, missing: set) -> None:
    for prop in pg.bl_rna.properties:
        key = prop.identifier
        if key not in data:
            continue
        value = data[key]
        if prop.type == 'COLLECTION':
            items = getattr(pg, key)
            items.clear()
            for entry in value:
                _restore(items.add(), entry, missing)
        elif prop.type == 'POINTER':
            if prop.fixed_type.identifier in _ID_DATA:
                target = _resolve(value)
                if value is not None and target is None:
                    missing.add(f"{value.get('id')} '{value.get('name')}'")
                try:
                    setattr(pg, key, target)
                except (TypeError, AttributeError):
                    pass
            elif isinstance(value, dict):
                _restore(getattr(pg, key), value, missing)
        elif not prop.is_readonly:
            _assign(pg, prop, value)


# --- curves -------------------------------------------------------------------

def _dump_curves(s) -> dict:
    from .curves import _KEY
    group = s.curve_group
    out = {}
    for node in (group.nodes if group is not None else ()):
        key = node.get(_KEY)
        if not key:
            continue
        m = node.mapping
        out[str(key)] = {
            "clip": [m.use_clip, m.clip_min_x, m.clip_min_y, m.clip_max_x, m.clip_max_y],
            "extend": m.extend,
            "points": [[p.location[0], p.location[1], p.handle_type]
                       for p in m.curves[0].points]}
    return out


def _restore_curves(obj, data: dict) -> None:
    """Every curve into a fresh group of this plant's own (the old one goes
    when nothing else uses it)."""
    from .curves import _GROUP_NAME, _KEY
    s = obj.plantgen
    old = s.curve_group
    group = bpy.data.node_groups.new(_GROUP_NAME, 'ShaderNodeTree')
    group.use_fake_user = True
    for key, entry in data.items():
        node = group.nodes.new('ShaderNodeFloatCurve')
        node[_KEY] = key
        node.label = key
        m = node.mapping
        use_clip, x0, y0, x1, y1 = entry["clip"]
        m.use_clip = use_clip
        m.clip_min_x, m.clip_min_y, m.clip_max_x, m.clip_max_y = x0, y0, x1, y1
        m.extend = entry.get("extend", m.extend)
        curve = m.curves[0]
        points = entry["points"]
        while len(curve.points) > max(len(points), 2):
            curve.points.remove(curve.points[-1])
        while len(curve.points) < len(points):
            curve.points.new(0.5, 0.5)
        for point, (x, y, handle) in zip(curve.points, points, strict=False):
            point.location = (x, y)
            point.handle_type = handle
        m.update()
        m.reset_view()
    s.curve_group = group
    if old is not None and not any(o.plantgen.curve_group == old
                                   for o in bpy.data.objects):
        bpy.data.node_groups.remove(old)


# --- assets -------------------------------------------------------------------

def _refs(data, kinds=_BUNDLED):
    """Every (kind, name) datablock reference in a dumped section."""
    if isinstance(data, dict):
        if data.get("id") in kinds and "name" in data:
            yield data["id"], data["name"]
            return
        for value in data.values():
            yield from _refs(value, kinds)
    elif isinstance(data, list):
        for value in data:
            yield from _refs(value, kinds)


def _closure(roots) -> set:
    """`roots` and every datablock they use, at any depth."""
    uses: dict = {}
    for datablock, users in bpy.data.user_map().items():
        for user in users:
            uses.setdefault(user, set()).add(datablock)
    seen, todo = set(), list(roots)
    while todo:
        datablock = todo.pop()
        if datablock in seen:
            continue
        seen.add(datablock)
        todo.extend(uses.get(datablock, ()))
    return seen


def write_assets(path: Path, plant: dict) -> dict | None:
    """Write the datablocks `plant` points at (and all they use) to `path`;
    the preset's `assets` entry, or None when it points at none."""
    names = {"Collection": set(), "Material": set()}
    for kind, name in _refs(plant):
        names[kind].add(name)
    roots = {bpy.data.collections[n] for n in names["Collection"] if n in bpy.data.collections}
    roots |= {bpy.data.materials[n] for n in names["Material"] if n in bpy.data.materials}
    if not roots:
        return None
    packed = []
    try:
        for image in (d for d in _closure(roots) if isinstance(d, bpy.types.Image)):
            if image.packed_file is None and image.source == 'FILE' and \
                    os.path.isfile(bpy.path.abspath(image.filepath)):
                image.pack()
                packed.append(image)
        bpy.data.libraries.write(str(path), roots, path_remap='RELATIVE_ALL',
                                 fake_user=True, compress=True)
    finally:
        for image in packed:
            try:
                image.unpack(method='USE_ORIGINAL')
            except RuntimeError:
                pass
    return {"file": path.name, "collections": sorted(names["Collection"]),
            "materials": sorted(names["Material"])}


def _holder():
    holder = bpy.data.collections.get(ASSET_HOLDER)
    if holder is None:
        holder = bpy.data.collections.new(ASSET_HOLDER)
    scene = bpy.context.scene
    if holder.name not in scene.collection.children:
        scene.collection.children.link(holder)

    def find(layer):
        if layer.collection == holder:
            return layer
        for child in layer.children:
            found = find(child)
            if found is not None:
                return found
        return None
    layer = find(bpy.context.view_layer.layer_collection)
    if layer is not None:
        layer.hide_viewport = True
    return holder


def _empty(col) -> bool:
    return not len(col.all_objects) and not len(col.children)


def load_assets(folder: Path, assets: dict | None) -> list:
    """Append whatever `assets` names that this file lacks from the preset's
    .blend; the names neither here nor in it (empty when complete).

    A same-named collection that is EMPTY counts as lacking: a new plant named
    like the preset's source makes its own empty `<Name>_Leafassets` /
    `_Flowerassets` folders, which used to shadow the preset's -- its
    components then drew nothing. The preset's collection is appended (as
    `<name>.001`) and its contents moved into the empty one, so every
    reference by name lands on a filled folder; the shell is removed."""
    if not assets:
        return []
    have = bpy.data.collections
    want_c = [n for n in assets.get("collections", ()) if n not in have or _empty(have[n])]
    want_m = [n for n in assets.get("materials", ()) if n not in bpy.data.materials]
    blend = folder / assets.get("file", "")
    if (want_c or want_m) and blend.is_file():
        with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
            dst.collections = [n for n in want_c if n in src.collections]
            dst.materials = [n for n in want_m if n in src.materials]
            asked = list(dst.collections)
        if want_c:
            holder = _holder()
            for name, col in zip(asked, dst.collections, strict=True):
                if col is None:
                    continue
                target = have.get(name)
                if target is not None and target != col:      # fill the empty folder
                    for o in list(col.objects):
                        target.objects.link(o)
                    for child in list(col.children):
                        target.children.link(child)
                    have.remove(col)
                elif col.name not in holder.children:
                    holder.children.link(col)
    return [n for n in want_c if n not in have or _empty(have[n])] + \
        [n for n in want_m if n not in bpy.data.materials]


# --- the whole plant ----------------------------------------------------------

def dump(obj) -> dict:
    """The plant section of a preset: settings and curves."""
    s = obj.plantgen
    return {"format": FORMAT, "settings": _dump(s, top=True), "curves": _dump_curves(s)}


def restore(obj, plant: dict, folder: Path, assets: dict | None) -> list:
    """Make `obj` the saved plant: assets appended, every setting and curve
    restored, the derived state rebuilt. What could not be found (empty when
    the copy is complete). The caller regrows."""
    from . import map_icons, mapping, structure
    missing = set(load_assets(Path(folder), assets))
    s = obj.plantgen
    _restore(s, plant.get("settings", {}), missing)
    _restore_curves(obj, plant.get("curves", {}))
    s["structure_focus"] = 1                  # a level selected (view state)
    s["branch_levels_active"] = 0
    s["components_active"] = 0
    structure.rebuild_tree(s)
    mapping.rebuild_rows(s)
    map_icons.refresh(s)
    return sorted(missing)
