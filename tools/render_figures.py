"""Render the manual's figures into plant_gen/docs/images/<name>.jpg.

Runs in a background, factory-fresh Blender -- no file, nothing saved:

    blender -b --factory-startup --python tools/render_figures.py -- [figure names]

Each figure grows a fresh Add Plant plant (the Generic perennial the manual's
numbers are measured on) once per variant, realizes each result into a
temporary scene side by side with a label, and renders that scene with
Workbench through an orthographic front camera. (On a copy of a working file
it works the same; never point it at the working file itself: render probes
go in a separate temporary scene.)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "plant_gen" / "docs" / "images"
sys.path.insert(0, str(REPO))

# name -> (source, [(caption, settings)]). A source is a fresh Add Plant plant
# with plain stand-in leaf and flower meshes: "NEW" keeps its flower component
# on the Plant row (a flower at every tip), "NEW_HEAD" has it on the Stems
# level (one head at the stem's tip, drawn turned so its stem bends across the
# picture). Any other name is an object in the file.
# A setting key is "plant.<prop>", "level<i>.<prop>" or "flower.<prop>" (the
# first flower component); values are set, then the plant regrows.
FIGURES = {
    "r-growth-days": ("NEW", [
        ("Day 120", {"plant.day": 120.0}),
        ("Day 200", {"plant.day": 200.0}),
        ("Day 270", {"plant.day": 270.0}),
    ]),
    "r-nod-stiffness": ("NEW_HEAD", [
        ("Stiffness 0.5", {"level0.stiffness": 0.5, "flower.flower_weight": 0.2}),
        ("Stiffness 0.2", {"level0.stiffness": 0.2, "flower.flower_weight": 0.2}),
        ("Stiffness 0.1", {"level0.stiffness": 0.1, "flower.flower_weight": 0.2}),
    ]),
    "r-branch-growth": ("NEW", [
        ("Growth 100%", {"level1.growth": 100.0}),
        ("Growth 50%", {"level1.growth": 50.0}),
    ]),
    "r-seeds": ("NEW", [
        (f"Seed {s}", {"plant.seed": s}) for s in range(5)
    ]),
}

_GAP = 0.25          # space between variants, as a fraction of the widest one
_WIDTH_PX = 1400
_BACKGROUND = (0.94, 0.93, 0.90)


def _target(obj, key):
    s = obj.plantgen
    where, prop = key.split(".", 1)
    if where == "plant":
        return s, prop
    if where.startswith("level"):
        return s.branch_levels[int(where[5:])], prop
    if where == "flower":
        comp = next((c for c in s.components if c.type == 'flower'), None)
        return (comp if comp is not None else s), prop   # no component: the plant's own
    raise KeyError(key)


def _material(name, rgb):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def _stand_in_leaf():
    """A plain blade: flat in XY, face up, base at the origin, tip along +Y."""
    me = bpy.data.meshes.new("_fig_leaf")
    w, length = 0.008, 0.07
    me.from_pydata([(0, 0, 0), (w, length * 0.35, 0), (0, length, 0), (-w, length * 0.35, 0)], [],
                   [(0, 1, 2, 3)])
    me.materials.append(_material("Leaf", (0.22, 0.45, 0.18)))
    return bpy.data.objects.new("_fig_leaf", me)


def _stand_in_flower():
    """A plain cup facing +Z, base at the origin."""
    import bmesh
    me = bpy.data.meshes.new("_fig_flower")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=10, radius1=0.004,
                          radius2=0.016, depth=0.014)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 0.007))
    bm.to_mesh(me)
    bm.free()
    me.materials.append(_material("Flower", (0.80, 0.42, 0.62)))
    return bpy.data.objects.new("_fig_flower", me)


_FRESH: dict = {}


def _source(name):
    """The figure's plant: a named object, or a fresh plant made once per
    source name and shared by every figure that names it."""
    if name not in ("NEW", "NEW_HEAD"):
        return bpy.data.objects[name]
    if name not in _FRESH:
        from plant_gen.blender import bridge
        obj = bridge.create_plant(bpy.context, name=f"Fig{name.title().replace('_', '')}",
                                  preset=None)
        s = obj.plantgen
        s.live_update = False
        s.leaf_asset_collection.objects.link(_stand_in_leaf())
        s.flower_asset_collection.objects.link(_stand_in_flower())
        # Add Plant's own flower component ("Replace Me") is on the Plant row,
        # a flower at every tip; NEW_HEAD moves it to the Stems level, the
        # stem's tip alone.
        if name == "NEW_HEAD":
            from plant_gen.blender import structure
            comp = next(c for c in s.components if c.type == 'flower')
            for q in s.placements:
                if q.comp_uid == comp.uid:
                    q.level_uid = s.branch_levels[0].uid
            structure.rebuild_tree(s)
        bridge.regrow(obj)
        _FRESH[name] = obj
    return _FRESH[name]


def _stem_bend_azimuth(obj) -> float:
    """Which way the first stem's tip has moved off its base, about Z."""
    from plant_gen.blender import bridge
    from plant_gen.growth.development import build
    s = obj.plantgen
    p = build(bridge.profile_from_object(obj), seed=int(s.seed), day=float(s.day))
    stem = p.axes[min(p.axes)]
    dx, dy = (stem.phytomers[-1].tip_position - stem.phytomers[0].tip_position)[:2]
    return math.atan2(dy, dx)


def _realize(obj):
    from plant_gen.blender import bridge, rig
    bridge.regrow(obj)
    bpy.context.view_layer.update()
    return rig._realize_full(obj)


def _label(text, scene, x, z, size):
    curve = bpy.data.curves.new(f"label_{text}", type='FONT')
    curve.body = text
    curve.size = size
    curve.align_x = 'CENTER'
    ob = bpy.data.objects.new(curve.name, curve)
    ob.location = (x, 0.0, z)
    ob.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.collection.objects.link(ob)
    return ob


def _bounds(mesh):
    import numpy as np
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return co.min(axis=0), co.max(axis=0)


def render(name: str) -> Path:
    source, variants = FIGURES[name]
    obj = _source(source)
    obj.plantgen.live_update = False
    window = bpy.context.window
    home = window.scene if window is not None else None
    scene = bpy.data.scenes.new(f"_figure_{name}")
    saved = []
    try:
        meshes = []
        for caption, settings in variants:
            for key, value in settings.items():
                target, prop = _target(obj, key)
                saved.append((target, prop, getattr(target, prop)))
                setattr(target, prop, value)
            mesh = _realize(obj)
            if source == "NEW_HEAD":             # turn the bend across the picture
                from mathutils import Matrix
                mesh.transform(Matrix.Rotation(-_stem_bend_azimuth(obj), 4, 'Z'))
            meshes.append((caption, mesh))
            for target, prop, value in reversed(saved):
                setattr(target, prop, value)
            saved.clear()
        boxes = [_bounds(m) for _c, m in meshes]
        widest = max(hi[0] - lo[0] for lo, hi in boxes)
        tallest = max(hi[2] for _lo, hi in boxes)
        x = 0.0
        centres = []
        for (caption, mesh), (lo, hi) in zip(meshes, boxes, strict=True):
            ob = bpy.data.objects.new(f"fig_{caption}", mesh)
            cx = x + widest / 2
            ob.location = (cx - (lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.0)
            scene.collection.objects.link(ob)
            centres.append(cx)
            x += widest * (1 + _GAP)
        size = tallest * 0.06
        for caption, cx in zip((c for c, _m in meshes), centres, strict=True):
            _label(caption, scene, cx, -size * 1.8, size)
        span_x = x - widest * _GAP
        bottom, top = -size * 2.4, tallest * 1.04
        cam_data = bpy.data.cameras.new("_figure_cam")
        cam_data.type = 'ORTHO'
        aspect = span_x / (top - bottom)
        cam_data.ortho_scale = max(span_x, top - bottom) * 1.06
        cam = bpy.data.objects.new("_figure_cam", cam_data)
        cam.location = (span_x / 2, -10.0, (top + bottom) / 2)
        cam.rotation_euler = (math.radians(90), 0.0, 0.0)
        scene.collection.objects.link(cam)
        scene.camera = cam
        r = scene.render
        r.engine = 'BLENDER_WORKBENCH'
        if aspect >= 1.0:
            r.resolution_x, r.resolution_y = _WIDTH_PX, max(200, int(_WIDTH_PX / aspect))
        else:                        # a tall figure: cap its height instead
            r.resolution_x, r.resolution_y = max(200, int(_WIDTH_PX * 0.7 * aspect)), \
                int(_WIDTH_PX * 0.7)
        r.resolution_percentage = 100
        r.image_settings.file_format = 'JPEG'     # ~90 KB a figure; a PNG was ~700 KB
        r.image_settings.color_mode = 'RGB'
        r.image_settings.quality = 88
        scene.view_settings.view_transform = 'Standard'
        sh = scene.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'MATERIAL'
        sh.background_type = 'VIEWPORT'
        sh.background_color = _BACKGROUND
        sh.show_shadows = False
        sh.show_cavity = True
        scene.display.render_aa = '8'
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{name}.jpg"
        r.filepath = str(out)
        if window is not None:
            window.scene = scene
        with bpy.context.temp_override(scene=scene):
            bpy.ops.render.render(write_still=True, scene=scene.name)
        print(f"[figures] wrote {out}")
        return out
    finally:
        for target, prop, value in reversed(saved):
            setattr(target, prop, value)
        if window is not None and home is not None:
            window.scene = home          # the plant regrows in its own scene next
        bpy.data.scenes.remove(scene)


# --- UV figures ------------------------------------------------------------------
# The stems alone (a new plant with nothing in its leaf / flower folders),
# welded, under the UV Checker: the whole plant, a close-up of a branch
# junction from two sides, and that junction's UV layout over the tiled checker.

UV_FIGURES = ("r-uv-stems", "r-uv-weld", "r-uv-layout")
_UV: dict = {}
_UV_REACH = 5.0          # the close-up keeps the stem within this many radii of the joint
_STEMS_REACH = 14.0      # the stem view: this many radii around the lower stem
_LAYOUT_REACH = 4.0      # the UV layout: the faces within this many radii


def _uv_plant() -> dict:
    """Grow, weld and realize the stems once for every UV figure; find a
    branch junction on the first stem."""
    if _UV:
        return _UV
    import numpy as np

    from plant_gen.blender import bridge, rig
    from plant_gen.growth.development import build
    obj = bridge.create_plant(bpy.context, name="FigStems", preset=None)
    s = obj.plantgen
    s.live_update = False
    s.uv_checker = True
    bridge.regrow(obj, weld=bridge.WELD_NOW)
    bpy.context.view_layer.update()
    mesh = rig._realize_full(obj)
    p = build(bridge.profile_from_object(obj), seed=int(s.seed), day=float(s.day))
    stem = p.axes[min(p.axes)]
    branches = [a for a in p.axes.values() if a.parent_axis_id == stem.id and a.phytomers]
    # a typical joint: up off the flared foot (its cap would crowd the
    # layout), and of those the one farthest from any other branch's base
    # (a neighbour would intrude on the close-up)
    top = max(ph.tip_position[2] for ph in stem.phytomers)

    def foot(a):
        ph = a.phytomers[0]
        return np.asarray(ph.tip_position) - np.asarray(ph.direction) * ph.length

    feet = {a.id: foot(a) for a in branches}
    middle = [a for a in branches if 0.15 * top < feet[a.id][2] < 0.6 * top] or branches
    chosen = max(middle, key=lambda a: min(
        (np.linalg.norm(feet[a.id] - feet[b.id]) for b in branches if b is not a),
        default=1e9))
    first = chosen.phytomers[0]
    base = feet[chosen.id]
    at = min(stem.phytomers, key=lambda ph: np.linalg.norm(np.asarray(ph.tip_position) - base))
    _UV.update(obj=obj, mesh=mesh, base=base, branch=np.asarray(first.direction),
               stem=np.asarray(at.direction), radius=float(at.radius))
    return _UV


def _near_joint(mesh, centre, reach):
    """A copy of `mesh` with only the faces whose every vertex lies within
    `reach` of `centre`, moved so `centre` is the origin."""
    import bmesh
    from mathutils import Vector
    bm = bmesh.new()
    bm.from_mesh(mesh)
    c = Vector(centre)
    far = [f for f in bm.faces if any((v.co - c).length > reach for v in f.verts)]
    bmesh.ops.delete(bm, geom=far, context='FACES')
    # only the piece joined to the joint: a stray bit of another branch the
    # sphere clipped would float beside it
    start = min(bm.faces, key=lambda f: (f.calc_center_median() - c).length)
    keep, todo = {start}, [start]
    while todo:
        for e in todo.pop().edges:
            for f in e.link_faces:
                if f not in keep:
                    keep.add(f)
                    todo.append(f)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f not in keep], context='FACES')
    bmesh.ops.translate(bm, verts=bm.verts, vec=-c)
    out = bpy.data.meshes.new("_joint")
    bm.to_mesh(out)
    bm.free()
    for mat in mesh.materials:
        out.materials.append(mat)
    return out


def _facing(towards_viewer, up):
    """The rotation that turns `towards_viewer` to face the figure camera (-Y)
    with `up` kept as near +Z as it goes."""
    import numpy as np
    from mathutils import Matrix
    f = np.asarray(towards_viewer, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float) - f * np.dot(up, f)
    u /= np.linalg.norm(u)
    r = np.cross(-f, u)
    return Matrix((tuple(r), tuple(-f), tuple(u))).to_4x4()


def _shoot(name, scene, centre_x, centre_z, span_x, span_z, color_type='TEXTURE'):
    """Render `scene` through an orthographic front camera framing the given
    box, Workbench, to docs/images/<name>.jpg."""
    cam_data = bpy.data.cameras.new("_figure_cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max(span_x, span_z) * 1.06
    cam = bpy.data.objects.new("_figure_cam", cam_data)
    cam.location = (centre_x, -10.0, centre_z)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    aspect = span_x / span_z
    r = scene.render
    r.engine = 'BLENDER_WORKBENCH'
    if aspect >= 1.0:
        r.resolution_x, r.resolution_y = _WIDTH_PX, max(200, int(_WIDTH_PX / aspect))
    else:
        r.resolution_x, r.resolution_y = max(200, int(_WIDTH_PX * 0.7 * aspect)), \
            int(_WIDTH_PX * 0.7)
    r.image_settings.file_format = 'JPEG'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.quality = 90
    scene.view_settings.view_transform = 'Standard'
    sh = scene.display.shading
    sh.light = 'STUDIO'
    sh.color_type = color_type
    sh.background_type = 'VIEWPORT'
    sh.background_color = _BACKGROUND
    sh.show_shadows = False
    sh.show_cavity = False
    scene.display.render_aa = '8'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.jpg"
    r.filepath = str(out)
    with bpy.context.temp_override(scene=scene):
        bpy.ops.render.render(write_still=True, scene=scene.name)
    print(f"[figures] wrote {out}")
    return out


def render_uv(name: str) -> Path:
    uv = _uv_plant()
    if name == "r-uv-layout":
        return _uv_layout(name, uv)
    window = bpy.context.window
    home = window.scene if window is not None else None
    scene = bpy.data.scenes.new(f"_figure_{name}")
    try:
        if window is not None:
            window.scene = scene
        import numpy as np
        if name == "r-uv-stems":
            # the lower stem around the first joints, close enough to read
            # the cells: a stretch _STEMS_REACH radii either side
            reach = _STEMS_REACH * uv["radius"]
            centre = uv["base"] + uv["stem"] * reach * 0.6
            mesh = _near_joint(uv["mesh"], centre, reach)
            mesh.transform(_facing(np.cross(uv["stem"], uv["branch"]), uv["stem"]))
            lo, hi = _bounds(mesh)
            ob = bpy.data.objects.new("fig_stems", mesh)
            scene.collection.objects.link(ob)
            return _shoot(name, scene, (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2,
                          (hi[0] - lo[0]) * 1.04, (hi[2] - lo[2]) * 1.04)
        # r-uv-weld: the joint side-on (stem and branch in the picture's
        # plane) and three-quarter (from the branch's front and side)
        reach = _UV_REACH * uv["radius"]
        side = np.cross(uv["stem"], uv["branch"])
        out_of = uv["branch"] - uv["stem"] * np.dot(uv["branch"], uv["stem"])
        views = [("Side", side), ("Three-quarter", side / np.linalg.norm(side)
                                  + out_of / np.linalg.norm(out_of))]
        gap = reach * 0.4
        size = reach * 0.16
        for k, (caption, toward) in enumerate(views):
            mesh = _near_joint(uv["mesh"], uv["base"], reach)
            mesh.transform(_facing(toward, uv["stem"]))
            ob = bpy.data.objects.new(f"fig_{caption}", mesh)
            ob.location = (k * (2 * reach + gap), 0.0, 0.0)
            scene.collection.objects.link(ob)
            _label(caption, scene, k * (2 * reach + gap), -reach - size * 1.6, size)
        span_x = 2 * (2 * reach) + gap
        return _shoot(name, scene, (2 * reach + gap) / 2, -size, span_x,
                      2 * reach + size * 3.0)
    finally:
        if window is not None and home is not None:
            window.scene = home
        bpy.data.scenes.remove(scene)


_MARGIN_PX = 20
_GAP_PX = 40
_PANEL_BG = 0.08


def _islands(mesh, co):
    """Each face's UV island: faces joined where they share a vertex at the
    same UV."""
    import numpy as np
    loop_vert = np.empty(len(mesh.loops), dtype=int)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    parent = list(range(len(mesh.polygons)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict = {}
    for poly in mesh.polygons:
        for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
            key = (int(loop_vert[li]), round(float(co[li, 0]), 5), round(float(co[li, 1]), 5))
            parent[root(poly.index)] = root(seen.setdefault(key, poly.index))
    ids: dict = {}
    return [ids.setdefault(root(p.index), len(ids)) for p in mesh.polygons]


def _panel(mesh, co, faces, colour_of, grid, ppt):
    """One panel: `faces` in UV space over the tiled, dimmed checker, each
    face's edges in its island's colour. Rows run bottom-up, as UV v does."""
    import numpy as np
    loops = [li for f in faces for li in range(mesh.polygons[f].loop_start,
                                                mesh.polygons[f].loop_start
                                                + mesh.polygons[f].loop_total)]
    lo, hi = co[loops].min(axis=0), co[loops].max(axis=0)
    w = int((hi[0] - lo[0]) * ppt) + 2 * _MARGIN_PX
    h = int((hi[1] - lo[1]) * ppt) + 2 * _MARGIN_PX
    gh, gw = grid.shape[:2]
    u = lo[0] + (np.arange(w) - _MARGIN_PX) / ppt
    v = lo[1] + (np.arange(h) - _MARGIN_PX) / ppt
    gx = ((u % 1.0) * gw).astype(int).clip(0, gw - 1)
    gy = ((v % 1.0) * gh).astype(int).clip(0, gh - 1)
    canvas = grid[gy[:, None], gx[None, :]].copy()
    canvas[:, :, :3] *= 0.55
    canvas[:, :, 3] = 1.0
    starts, ends, colour = [], [], []
    for f in faces:
        poly = mesh.polygons[f]
        idx = list(range(poly.loop_start, poly.loop_start + poly.loop_total))
        starts.extend(idx)
        ends.extend(idx[1:] + idx[:1])
        colour.extend([colour_of[f]] * len(idx))
    a = (co[starts] - lo) * ppt + _MARGIN_PX
    b = (co[ends] - lo) * ppt + _MARGIN_PX
    steps = np.ceil(np.linalg.norm(b - a, axis=1)).astype(int) + 1
    t = np.concatenate([np.linspace(0.0, 1.0, n) for n in steps])
    rep = np.repeat(np.arange(len(a)), steps)
    pts = a[rep] + (b[rep] - a[rep]) * t[:, None]
    x = pts[:, 0].round().astype(int).clip(0, w - 1)
    y = pts[:, 1].round().astype(int).clip(0, h - 1)
    canvas[y, x, :3] = np.asarray(colour)[rep]
    return canvas


def _uv_layout(name: str, uv) -> Path:
    """The joint's faces in UV space, as the UV editor shows them, over the
    tiled checker (one tile = one UV Size of stem). The stem's island and the
    branch's sit far apart in v (the branch starts at v = 0, the stem's hole
    is at the joint's height), so each piece is its own panel, side by side,
    at its true UVs: the stem's island with the hole, then the branch's with
    its fillet. Drawn with numpy and saved through Blender's image API
    (Blender's Python has no PIL)."""
    import numpy as np
    mesh = _near_joint(uv["mesh"], uv["base"], _LAYOUT_REACH * uv["radius"])
    layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers[0]
    co = np.empty(len(layer.data) * 2)
    layer.data.foreach_get("uv", co)
    co = co.reshape(-1, 2)
    island = _islands(mesh, co)
    palette = [(1.0, 1.0, 1.0), (1.0, 0.72, 0.25), (0.45, 0.85, 1.0),
               (0.6, 1.0, 0.5), (1.0, 0.5, 0.75)]
    colour_of = [palette[i % len(palette)] for i in island]
    # pieces: islands whose UV boxes lie within a tile of each other
    boxes = {}
    for f, i in enumerate(island):
        poly = mesh.polygons[f]
        pts = co[poly.loop_start:poly.loop_start + poly.loop_total]
        lo, hi = boxes.get(i, (pts.min(axis=0), pts.max(axis=0)))
        boxes[i] = (np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0)))
    piece = {i: i for i in boxes}
    for i in boxes:
        for j in boxes:
            (alo, ahi), (blo, bhi) = boxes[i], boxes[j]
            if i < j and np.all(alo <= bhi + 1.0) and np.all(blo <= ahi + 1.0):
                old, new = piece[j], piece[i]
                piece = {k: (new if p == old else p) for k, p in piece.items()}
    groups: dict = {}
    for f, i in enumerate(island):
        groups.setdefault(piece[i], []).append(f)
    panels_faces = sorted(groups.values(), key=len, reverse=True)   # the stem first
    img = bpy.data.images.get("PlantGen UV Checker")
    grid = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)
    spans = []
    for faces in panels_faces:
        loops = [li for f in faces for li in range(mesh.polygons[f].loop_start,
                                                    mesh.polygons[f].loop_start
                                                    + mesh.polygons[f].loop_total)]
        spans.append(co[loops].max(axis=0) - co[loops].min(axis=0))
    tall = max(s[1] for s in spans)
    ppt = int(min(1100 / tall, 1300 / sum(s[0] for s in spans), 360))   # pixels per tile
    panels = [_panel(mesh, co, faces, colour_of, grid, ppt) for faces in panels_faces]
    height = max(p.shape[0] for p in panels)
    parts = []
    for k, p in enumerate(panels):
        if k:
            gap = np.full((height, _GAP_PX, 4), _PANEL_BG, dtype=np.float32)
            gap[:, :, 3] = 1.0
            parts.append(gap)
        pad = np.full((height - p.shape[0], p.shape[1], 4), _PANEL_BG, dtype=np.float32)
        pad[:, :, 3] = 1.0
        parts.append(np.concatenate([p, pad]))          # rows run up: pad above
    canvas = np.concatenate(parts, axis=1)
    h, w = canvas.shape[:2]
    out_img = bpy.data.images.new(name, w, h, alpha=False)
    out_img.pixels[:] = np.ascontiguousarray(canvas).ravel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    out_img.filepath_raw = str(out)
    out_img.file_format = 'PNG'
    out_img.save()
    print(f"[figures] wrote {out} ({w} x {h}, {len(panels)} pieces, "
          f"{len(boxes)} islands)")
    return out


def main(argv):
    names = argv or list(FIGURES) + list(UV_FIGURES)
    from plant_gen import register
    try:
        register()
    except ValueError:
        pass                         # already registered from the user's add-ons
    failed = []
    for name in names:
        try:
            render_uv(name) if name in UV_FIGURES else render(name)
        except Exception:            # noqa: BLE001 -- report it, render the rest
            import traceback
            traceback.print_exc()
            failed.append(name)
    if failed:
        print("[figures] FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
