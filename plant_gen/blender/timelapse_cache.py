"""Timelapse Cache -- Grow Over Timeline played back smoothly.

Grow Over Timeline regrows the real plant on every frame (exact, but ~1.5 s a
frame: growth and the post-growth passes are Python). Every frame regrown is
recorded here -- the plant's finished output: its mesh (stems, anything
realized) and each instance's asset + transform -- and from then on that frame
is shown from the record instead (`mesh._build_timelapse_cache`, a socket push),
no regrow at all. Measured on a Digitalis in a 476-object scene: ~0.65 fps live,
~23 fps from the record (most of what is left is the scene's own redraw).

* **Written in place.** The record goes into two hidden helper objects that keep
  their identity -- swapping an object's mesh datablock made Blender rebuild the
  scene's relations (~45 ms a frame here), writing into the same mesh does not.
  A frame whose topology matches the one shown only rewrites positions + UVs.
* **Exact, then invalid.** A frame is the plant as it was grown at that frame;
  any edit that changes growth (`bridge.growth_signature`, Day aside), the Day
  range or the frame range drops the whole record. Renders never use it (they
  regrow and weld, `live.drive_timelapse_day`).
* **In memory**, per plant (by `session_uid`); gone on reload.
* Instances are re-made from the asset OBJECTS they came from, by name: an
  asset's per-asset Red gradient (a GN store on the instanced geometry) is not
  in the record -- it is a preview of the shape, not of the vertex colours.
"""

from __future__ import annotations

import zlib

import bpy
import numpy as np

from . import mesh as gn_mesh

_CACHES: dict[int, _Cache] = {}


class _Frame:
    __slots__ = ("co", "loops", "starts", "mat", "uv", "sharp", "topo", "materials",
                 "refs", "xf")


class _Cache:
    def __init__(self, key: tuple) -> None:
        self.key = key
        self.frames: dict[int, _Frame] = {}
        self.ref_names: list[str] = []          # every asset any frame instances
        self.ref_id: dict[str, int] = {}


def _key(obj, scene) -> tuple:
    """What a recorded frame depends on. The Bloom display (mode, spread, smooth
    blending) never regrows, so it is not in the growth signature, yet the record
    holds the flowers as they were drawn: switching From Age and Manual left
    frames of both in one record, and playback flickered between them. Progress
    stays out: it is keyed per frame in Manual, and would drop the record each one."""
    from . import bridge
    s = obj.plantgen
    return (bridge.growth_signature(obj, include_day=False),
            round(float(s.timelapse_start_day), 6), round(float(s.timelapse_end_day), 6),
            int(scene.frame_start), int(scene.frame_end),
            s.bloom_mode, round(float(s.bloom_spread), 6), bool(s.bloom_continuous))


def _cache(obj, scene, create: bool = True) -> _Cache | None:
    """The plant's record if it still matches the plant (else dropped)."""
    key = _key(obj, scene)
    found = _CACHES.get(obj.session_uid)
    if found is not None and found.key != key:
        found = None
        _CACHES.pop(obj.session_uid, None)
    if found is None and create:
        found = _CACHES[obj.session_uid] = _Cache(key)
    return found


def stats(obj, scene) -> tuple[int, int]:
    """(frames recorded and still valid, frames in the scene's range)."""
    total = max(0, int(scene.frame_end) - int(scene.frame_start) + 1)
    found = _CACHES.get(obj.session_uid)
    if found is None or found.key != _key(obj, scene):
        return 0, total
    inside = sum(1 for f in found.frames if scene.frame_start <= f <= scene.frame_end)
    return inside, total


def clear(obj) -> None:
    """Drop the plant's record and show the live plant."""
    _CACHES.pop(obj.session_uid, None)
    hide(obj)


def clear_all() -> None:
    _CACHES.clear()


# --- recording -------------------------------------------------------------------

def record(obj, scene, frame: int) -> None:
    """Record the plant's evaluated output as frame `frame` (call right after
    it was regrown for that frame)."""
    cache = _cache(obj, scene)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    geo = obj.evaluated_get(depsgraph).evaluated_geometry()
    fr = _Frame()
    m = geo.mesh
    if m is not None and len(m.vertices):
        nv, nl, npoly = len(m.vertices), len(m.loops), len(m.polygons)
        fr.co = np.empty(nv * 3, np.float32)
        m.vertices.foreach_get("co", fr.co)
        fr.loops = np.empty(nl, np.int32)
        m.loops.foreach_get("vertex_index", fr.loops)
        fr.starts = np.empty(npoly, np.int32)
        m.polygons.foreach_get("loop_start", fr.starts)
        fr.mat = np.empty(npoly, np.int32)
        m.polygons.foreach_get("material_index", fr.mat)
        uv_layer = m.uv_layers.get("UVMap")
        fr.uv = None
        if uv_layer is not None:
            fr.uv = np.empty(nl * 2, np.float32)
            uv_layer.data.foreach_get("uv", fr.uv)
        sharp = m.attributes.get("sharp_face")
        fr.sharp = None
        if sharp is not None:
            fr.sharp = np.empty(npoly, bool)
            sharp.data.foreach_get("value", fr.sharp)
        fr.topo = (nv, zlib.crc32(fr.loops.tobytes()), zlib.crc32(fr.starts.tobytes()))
        # An empty slot draws with the plant OBJECT's material for that slot
        # (the stems: GN sets none, the object's slot carries the stem
        # material); the helper has no slots of its own, so record that one.
        slots = [sl.material for sl in obj.material_slots]
        fr.materials = tuple(
            mt.name if mt is not None
            else (slots[i].name if i < len(slots) and slots[i] is not None else "")
            for i, mt in enumerate(m.materials))
    else:
        fr.co = None
        fr.topo = None
        fr.materials = ()
    fr.refs, fr.xf = np.zeros(0, np.int32), np.zeros(0, np.float32)
    points = geo.instances_pointcloud()
    if points is not None and len(points.points):
        n = len(points.points)
        index = np.empty(n, np.int32)
        points.attributes[".reference_index"].data.foreach_get("value", index)
        xf = np.empty(n * 16, np.float32)
        points.attributes["instance_transform"].data.foreach_get("value", xf)
        names = [getattr(r, "name", "") for r in geo.instance_references()]
        ids = np.full(len(names), -1, np.int32)
        for i, name in enumerate(names):
            if not name or bpy.data.objects.get(name) is None:
                continue                         # not an object: not re-makeable
            if name not in cache.ref_id:
                cache.ref_id[name] = len(cache.ref_names)
                cache.ref_names.append(name)
            ids[i] = cache.ref_id[name]
        refs = ids[index]
        keep = refs >= 0
        fr.refs = refs[keep]
        fr.xf = xf.reshape(n, 16)[keep].reshape(-1)
    cache.frames[int(frame)] = fr


# --- showing ---------------------------------------------------------------------------

def _helpers(obj):
    """(mesh object, instance-points object, assets collection), made on first
    use: hidden, unselectable, never rendered, in the plant's collection."""
    made = []
    home = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection

    def helper(key: str, name: str):
        ob = bpy.data.objects.get(obj.get(key, ""))
        if ob is None:
            ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
            home.objects.link(ob)
            ob.hide_render = True
            ob.hide_select = True
            made.append(ob)
            obj[key] = ob.name
        return ob

    mesh_ob = helper("plantgen_tl_mesh", f"{obj.name}_tlmesh")
    points_ob = helper("plantgen_tl_points", f"{obj.name}_tlinstances")
    coll = bpy.data.collections.get(obj.get("plantgen_tl_assets", ""))
    if coll is None:
        coll = bpy.data.collections.new(f"{obj.name}_tlassets")   # never in the scene
        obj["plantgen_tl_assets"] = coll.name
    if made:
        # In the depsgraph before it is hidden, or Object Info reads nothing.
        bpy.context.view_layer.update()
        for ob in made:
            ob.hide_set(True)
    return mesh_ob, points_ob, coll


def _natural(name: str):
    """Collection Info's order for separated children: case-insensitive
    natural sort by name (verified: references keep link order, instances
    are sorted)."""
    import re
    return [(0, int(tok)) if tok.isdigit() else (1, tok.lower())
            for tok in re.split(r"(\d+)", name) if tok]


def _write_mesh(me, fr: _Frame) -> None:
    if fr.co is None:
        me.clear_geometry()
        me["pgs_topo"] = ""
        return
    # The material list first: clearing it resets every face's material index
    # to 0, so indices written before it were lost whenever the list changed.
    if tuple(mt.name if mt is not None else "" for mt in me.materials) != fr.materials:
        me.materials.clear()
        for name in fr.materials:
            me.materials.append(bpy.data.materials.get(name) if name else None)
    topo = repr(fr.topo)
    if me.get("pgs_topo") == topo:
        me.vertices.foreach_set("co", fr.co)
        if fr.uv is not None and me.uv_layers.get("UVMap") is not None:
            me.uv_layers["UVMap"].data.foreach_set("uv", fr.uv)
        # Per face too: the same topology can come with another material list
        # (a frame with material-less geometry joined gains an empty slot 0),
        # and indices kept from the last frame then point at the wrong slot --
        # the stems lost their material on those frames.
        me.polygons.foreach_set("material_index", fr.mat)
        sharp = me.attributes.get("sharp_face")
        if fr.sharp is not None:
            sharp = sharp or me.attributes.new("sharp_face", 'BOOLEAN', 'FACE')
            sharp.data.foreach_set("value", fr.sharp)
        elif sharp is not None:
            me.attributes.remove(sharp)
        me.update()
    else:
        me.clear_geometry()
        nv, nl, npoly = len(fr.co) // 3, len(fr.loops), len(fr.starts)
        me.vertices.add(nv)
        me.vertices.foreach_set("co", fr.co)
        me.loops.add(nl)
        me.loops.foreach_set("vertex_index", fr.loops)
        me.polygons.add(npoly)
        me.polygons.foreach_set("loop_start", fr.starts)
        me.polygons.foreach_set("material_index", fr.mat)
        if fr.uv is not None:
            layer = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
            layer.data.foreach_set("uv", fr.uv)
        if fr.sharp is not None:
            attr = me.attributes.get("sharp_face") or me.attributes.new(
                "sharp_face", 'BOOLEAN', 'FACE')
            attr.data.foreach_set("value", fr.sharp)
        me.update(calc_edges=True)
        me["pgs_topo"] = topo


def _write_points(me, cache: _Cache, coll, fr: _Frame) -> None:
    # The assets collection holds every asset recorded so far; an instance picks
    # its asset's rank in the collection's name order.
    have = {o.name for o in coll.objects}
    for name in cache.ref_names:
        if name not in have and bpy.data.objects.get(name) is not None:
            coll.objects.link(bpy.data.objects[name])
    order = sorted((o.name for o in coll.objects), key=_natural)
    rank = {name: i for i, name in enumerate(order)}
    pick = np.array([rank.get(name, 0) for name in cache.ref_names], np.int32)
    n = len(fr.refs)
    if len(me.vertices) != n:
        me.clear_geometry()
        me.vertices.add(n)
    if n:
        xf = fr.xf.reshape(n, 4, 4)
        # Blender stores a 4x4 column-major: the translation is the 4th column.
        me.vertices.foreach_set("co", np.ascontiguousarray(xf[:, 3, :3]).reshape(-1))
        ref = me.attributes.get(gn_mesh.TL_REF_ATTR) or me.attributes.new(
            gn_mesh.TL_REF_ATTR, 'INT', 'POINT')
        ref.data.foreach_set("value", pick[fr.refs])
        mat = me.attributes.get(gn_mesh.TL_XF_ATTR) or me.attributes.new(
            gn_mesh.TL_XF_ATTR, 'FLOAT4X4', 'POINT')
        mat.data.foreach_set("value", fr.xf)
    me.update()


def show(obj, scene, frame: int) -> bool:
    """Show recorded frame `frame` in place of the live plant; False (and the
    live plant) if there is no valid record of it."""
    cache = _cache(obj, scene, create=False)
    fr = cache.frames.get(int(frame)) if cache is not None else None
    if fr is None:
        hide(obj)
        return False
    mesh_ob, points_ob, coll = _helpers(obj)
    _write_mesh(mesh_ob.data, fr)
    _write_points(points_ob.data, cache, coll, fr)
    if not obj.get("plantgen_tl_showing", False):
        obj["plantgen_tl_showing"] = True
        gn_mesh.push_timelapse(obj)
    return True


def hide(obj) -> None:
    """Show the live plant again (no-op if it already is)."""
    if obj.get("plantgen_tl_showing", False):
        obj["plantgen_tl_showing"] = False
        gn_mesh.push_timelapse(obj)
