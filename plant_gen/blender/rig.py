"""The rig -- an engine-first deform skeleton + smooth skin for a grown plant,
its Rig View in the viewport and its export (FBX / USD + Unreal wind data).

`build_rig` (layout and weights bpy-free in `geometry/rig_layout.py`) builds
the skeleton at the plant's current Day; `skin` is the realized welded mesh;
`apply_rig_view` shows the plant as it; `export_rig` writes the bind pose.
All bpy-side and **read-only** over the model -- it never touches the growth
layer, so it cannot move a golden. (The growth bake that keyframed growth onto
an older per-phytomer rig was deleted 2026-10-12; growth on the rig is to be
designed afresh.)

Positions are the plant's own object-local coordinates (the frame
`bridge.regrow` writes the skeleton mesh in), so the armature is authored at
identity and aligns with the plant when parented to it.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil

import bpy
import numpy as np
from mathutils import Vector

from ..growth.model import LifecycleState

SKIN_ATTRIBUTES = ("Color", "vcol_r", "organ_id", "instance_random")
"""What the rig's skin keeps beyond its geometry, UVs and materials: the
Quixel `Color` (exported), and Red / organ id / per-plant random for Vertex
Wind."""


class RigError(RuntimeError):
    """A rig that cannot be built, said in the user's terms."""


_KIND_NAMES = {0: "leaf", 1: "flower", 2: "bud", 3: "apex", 4: "fruit"}
_BRACT_FORM = 2
DEFAULT_BONE_ANGLE = 15.0           # degrees, for an axis on a level the panel lacks


def _organ_kind(kind: int, form: int) -> str:
    if kind == 0 and form == _BRACT_FORM:
        return "bract"
    return _KIND_NAMES.get(kind, "organ")


def _skin_arrays(mesh):
    """The realized skin's positions and the attributes the rig routes on."""
    n = len(mesh.vertices)
    co = np.empty(n * 3)
    mesh.vertices.foreach_get("co", co)
    out = {"co": co.reshape(n, 3)}
    for name, dtype, default in (("kind", np.int64, 0), ("width", np.float64, 0.0),
                                 ("organ_id", np.int64, -1), ("axis_id", np.int64, -1),
                                 ("tube", np.int64, -1), ("form", np.int64, 0)):
        a = mesh.attributes.get(name)
        if a is None or a.domain != 'POINT':
            out[name] = np.full(n, default, dtype=dtype)
            continue
        buf = np.empty(n, dtype=np.float64 if dtype is np.float64 else np.int64)
        a.data.foreach_get("value", buf)
        out[name] = buf.astype(dtype)
    return out


def _drawn_organ_bases(obj) -> dict:
    """organ_id -> where GN instances it (the organs object's point, in the
    plant's own frame)."""
    organs_obj = bpy.data.objects.get(obj.get("plantgen_organs_name", ""))
    if organs_obj is None or organs_obj.type != 'MESH':
        return {}
    me = organs_obj.data
    n = len(me.vertices)
    a = me.attributes.get("organ_id")
    if not n or a is None:
        return {}
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    to_plant = np.array(obj.matrix_world.inverted() @ organs_obj.matrix_world)
    co = co @ to_plant[:3, :3].T + to_plant[:3, 3]
    ids = np.empty(n, dtype=np.int64)
    a.data.foreach_get("value", ids)
    return {int(i): co[k] for k, i in enumerate(ids)}


def _organ_nodes(plant) -> dict:
    """organ_id -> (axis id, phytomer index) of the node that bears it."""
    out = {}
    for axis in plant.axes.values():
        if axis.lifecycle == LifecycleState.SHED:
            continue
        for j, phy in enumerate(axis.phytomers):
            for organ in phy.organs:
                out[int(organ.id)] = (axis.id, j)
        if axis.terminal_organ is not None and axis.phytomers:
            out[int(axis.terminal_organ.id)] = (axis.id, len(axis.phytomers) - 1)
    return out


def _bone_angle_of(obj):
    levels = obj.plantgen.branch_levels

    def angle(level: int) -> float:
        if 0 <= level < len(levels):
            return math.degrees(levels[level].rig_bone_angle)
        return DEFAULT_BONE_ANGLE
    return angle


def build_rig(obj, name=None, preview=True):
    """Build `obj`'s rig at its current Day: an armature (`<plant>_Rig`) and the
    skinned mesh (`<plant>_skin`) as hidden helpers under the plant, which
    then shows it (Rig View). The plant must be grown (and welded) first.
    `preview` False skips Fast Preview's helpers, for a rig only exported.
    Returns the counts."""
    from ..geometry import rig_layout as rl
    from . import bridge

    clear_rig(obj)                      # no skin -> Rig View off: realize the live plant
    from . import mesh as gn_mesh
    gn_mesh.push_rig_view(obj)
    plant = bridge.grown_plant(obj)
    skin_mesh = _realize_full(obj)
    base = name or obj.name
    skin_mesh.name = base + "_skin"
    arrays = _skin_arrays(skin_mesh)
    co, kind, width = arrays["co"], arrays["kind"], arrays["width"]
    is_organ = (kind != 0) | (width <= 1e-9)
    is_stem = ~is_organ
    if not is_stem.any():
        bpy.data.meshes.remove(skin_mesh)
        raise RigError("Nothing to rig: the plant has no stems at this Day "
                       f"({obj.plantgen.day:.0f})")

    flex = rl.sideways_turn(plant, bridge.profile_from_object(obj))
    stem, chains = rl.stem_bones(plant, _bone_angle_of(obj),
                                 skip=lambda a: a.lifecycle == LifecycleState.SHED,
                                 flex=flex)

    # Drawn organs: base where GN instances it, tip its farthest drawn vertex.
    bases = _drawn_organ_bases(obj)
    organ_ids = arrays["organ_id"]
    organs = []
    organ_verts = rl.group_indices(organ_ids, is_organ)
    for oid, m in organ_verts.items():
        pts = co[m]
        base_pt = bases.get(oid)
        if base_pt is None:
            base_pt = pts[int(np.argmin(pts[:, 2]))]
        tip = pts[int(np.argmax(np.linalg.norm(pts - base_pt, axis=1)))]
        k = int(np.bincount(kind[m]).argmax())
        f = int(np.bincount(np.maximum(arrays["form"][m], 0)).argmax())
        organs.append(rl.Organ(int(oid), _organ_kind(k, f), np.asarray(base_pt), tip))

    # Stalks: an organ drawn off its node (a petiole / pedicel between) has a
    # stalk from the node's centre out to it -- the model's pedicel path when
    # it has one. Its bone pivots at the node.
    nodes = _organ_nodes(plant)
    model_organs = {int(o.id): o for a in plant.axes.values() for p in a.phytomers
                    for o in p.organs}
    model_organs.update({int(a.terminal_organ.id): a.terminal_organ
                         for a in plant.axes.values() if a.terminal_organ is not None})
    parents, roots, stalks, stalk_info, organ_axis = {}, {}, [], {}, {}
    segments = rl.axis_segments(chains)
    placed = [o for o in organs
              if nodes.get(o.oid) is None or nodes[o.oid][0] not in chains]
    nearest_placed = dict(zip((o.oid for o in placed),
                              rl.nearest_many_on_axes([o.base for o in placed], segments),
                              strict=True))
    for o in organs:
        node = nodes.get(o.oid)
        if node is not None and node[0] in chains:
            chain = chains[node[0]]
            root = chain.points[min(node[1], len(chain.points) - 2) + 1]
        else:
            # A placed organ (a component's leaf / flower on a level, not one
            # growth made): no node in the model -- it hangs off the nearest
            # point of any stem's centre line.
            aid, k, root = nearest_placed[o.oid]
            if aid is None:
                continue
            node = (aid, k)
            chain = chains[aid]
        parents[o.oid] = chain.covering(node[1])
        organ_axis[o.oid] = node[0]
        axis = plant.axes[node[0]]
        radius = float(axis.phytomers[min(node[1], len(axis.phytomers) - 1)].radius)
        if np.linalg.norm(o.base - root) <= 1.5 * radius:
            continue                                  # sits on the stem: no stalk
        model = model_organs.get(o.oid)
        path = ([np.asarray(q, dtype=np.float64) for q in (model.pedicel_points or [])]
                if model is not None else [])
        line = np.array([root, *path[1:], o.base]) if path else np.array([root, o.base])
        stalks.append((o.oid, line))
        roots[o.oid] = root
        stalk_info[o.oid] = (node[0], root, radius, float(np.linalg.norm(o.base - root)))
    axes_geo = [(aid, c.points, [float(ph.radius) for ph in plant.axes[aid].phytomers])
                for aid, c in chains.items()]
    stem_segments = [(b.name, b.head, b.tail) for b in stem if b.name != rl.ROOT]

    def nearest_stem(point):
        best, best_d = rl.ROOT, np.inf
        for bname, h, tl in stem_segments:
            d, _ = rl._segment_project(point[None, :], h, tl)
            if d[0] < best_d:
                best, best_d = bname, float(d[0])
        return best

    riders = rl.organs_riding(organs, organ_axis, len(stem))
    bones = stem + rl.organ_bones([o for o in organs if o.oid not in riders],
                                  roots, parents, nearest_stem)
    if len(bones) > rl.MAX_BONES:
        bpy.data.meshes.remove(skin_mesh)
        raise RigError(f"{len(bones)} bones: Unreal imports at most {rl.MAX_BONES}. "
                       "Raise Bone Angle, or grow fewer branches")
    bone_len = {b.name: float(np.linalg.norm(b.tail - b.head)) for b in bones}
    organ_bone = {o.oid: parents[o.oid] if o.oid in riders else f"{o.kind}_{o.oid}"
                  for o in organs}

    # Weights. Organs ride their own bone rigidly; a stem vertex gets its
    # axis's two-bone blend, a stalk's foot hands over from that to the
    # organ's bone, and any remaining seam (stems welded together, a stalk in
    # a fillet) is relaxed -- at most four bones a vertex.
    groups: dict[str, tuple[list, list]] = {}

    def add(bname, idx, w):
        g = groups.setdefault(bname, ([], []))
        g[0].extend(int(i) for i in idx)
        g[1].extend(float(x) for x in w)

    def stem_weights(aid, idx, depth=0):
        """The axis's weights at `idx`, one dict a vertex: its two bones
        blended across each joint, and -- where it leaves its parent -- into
        the PARENT's own weights at that vertex (not one fixed parent bone:
        the crotch fillet above a rising branch lies along the parent's next
        bones, and a single bone there tore it)."""
        axis = plant.axes.get(aid)
        parent_chain = chains.get(axis.parent_axis_id) if axis is not None else None
        parent_bone = (parent_chain.covering(axis.parent_phytomer_index)
                       if parent_chain is not None else None)
        a, b, wb = rl.axis_weights(co[idx], chains[aid], bone_len, None)
        out = []
        for ba, bb, w in zip(a, b, wb, strict=True):
            d = {ba: 1.0 - w}
            if bb != ba and w > 0.0:
                d[bb] = d.get(bb, 0.0) + w
            out.append(d)
        if parent_chain is None or parent_bone is None or depth >= 8:
            return out
        # Off its parent: the branch's own weights by how far a vertex has
        # pulled away from the parent's surface, the parent's (as they are at
        # that vertex) the rest.
        parent_axis = plant.axes[axis.parent_axis_id]
        radius = float(parent_axis.phytomers[min(max(axis.parent_phytomer_index, 0),
                                                 len(parent_axis.phytomers) - 1)].radius)
        first = bone_len[chains[aid].names[0]]
        own = rl.surface_handover(co[idx], parent_chain.points, radius,
                                  max(radius, rl.BLEND * first))
        ks = np.nonzero(own < 1.0)[0]
        if len(ks):
            under = stem_weights(parent_chain.axis_id, np.asarray(idx)[ks], depth + 1)
            for k, pd in zip(ks, under, strict=True):
                h = float(own[k])
                d = {name: w * h for name, w in out[k].items()}
                for name, pw in pd.items():
                    d[name] = d.get(name, 0.0) + (1.0 - h) * pw
                out[k] = d
        return out

    for oid, idx in organ_verts.items():
        g = groups.setdefault(organ_bone[oid], ([], []))
        g[0].extend(idx.tolist())
        g[1].extend([1.0] * len(idx))
    stem_idx = np.nonzero(is_stem)[0]
    stem_co = co[stem_idx]
    axis_of, owner, near, near_d = rl.assign_stem(stem_co, axes_geo, stalks)
    me = skin_mesh
    edges = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    local = np.full(len(co), -1, dtype=np.int64)
    local[stem_idx] = np.arange(len(stem_idx))
    stem_edges = local[edges]
    stem_edges = stem_edges[(stem_edges >= 0).all(axis=1)]
    # Mend strays by the mesh's own connectivity: one label per vertex (an
    # axis id, or -2 - organ id for a stalk), voted over its neighbours.
    labels = rl.vote_labels(np.where(owner >= 0, -2 - owner, axis_of), stem_edges)
    labels = rl.settle_pieces(labels, rl.components(len(stem_idx), stem_edges))
    owner = np.where(labels <= -2, -2 - labels, -1)
    axis_of = np.where(labels >= 0, labels, axis_of)
    # A stalk's vertices belong to the axis it grows from, whichever axis
    # happens to be nearest.
    owned = rl.group_indices(owner, owner >= 0)
    for oid, (aid, *_rest) in stalk_info.items():
        if oid in owned:
            axis_of[owned[oid]] = aid
    weights: list[dict] = [{} for _ in range(len(stem_idx))]
    for aid in np.unique(axis_of[axis_of >= 0]):
        sel = np.nonzero(axis_of == aid)[0]
        for k, d in zip(sel, stem_weights(int(aid), stem_idx[sel]), strict=True):
            weights[k] = d
    for k in np.nonzero(axis_of < 0)[0]:              # nothing to place it on
        weights[k] = {nearest_stem(stem_co[k]): 1.0}
    # A stalk's foot: the same hand-over for every vertex near it (the stalk's
    # own and the fillet's), by how far out along the stalk it is past the
    # stem's surface -- a smoothstep over the stem's radius or a quarter of
    # the stalk -- so the weld never tears.
    lines = {}
    for o2, pts in stalks:
        lines.setdefault(o2, pts)           # the first, as a search would find
    nearest = rl.group_indices(near)
    for oid, (_aid, root, radius, length) in stalk_info.items():
        own = owned.get(oid)
        if own is None or oid not in organ_bone:
            continue
        line = lines[oid]
        u = line[1] - line[0]
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        girth = float(np.median(near_d[own]))
        sel = nearest.get(oid, np.empty(0, dtype=np.int64))
        sel = sel[near_d[sel] < 4.0 * girth + 0.5 * radius]
        along = (stem_co[sel] - root) @ u
        hand = rl._smooth((along - radius) / max(radius, 0.25 * length, 1e-6))
        bone = organ_bone[oid]
        for k, h in zip(sel, hand, strict=True):
            d = {name: w * (1.0 - h) for name, w in weights[k].items()} if h < 1.0 else {}
            if h > 0.0:
                d[bone] = d.get(bone, 0.0) + float(h)
            weights[k] = d
    rl.smooth_seams(stem_co, stem_edges, weights, np.maximum(3.0 * width[stem_idx], 1e-3))
    for k, d in enumerate(weights):
        v = int(stem_idx[k])
        for bname, w in d.items():
            add(bname, [v], [w])

    # The skin keeps only what the rig, wind and export read -- the plant's
    # working attributes (load, tube, axis ids...) made every frame's deform
    # and the plant's copy of it (Rig View) drag 41 attributes along.
    keep = {"position", "material_index", "sharp_face", "sharp_edge", *SKIN_ATTRIBUTES}
    keep |= {uv.name for uv in skin_mesh.uv_layers}
    for a in list(skin_mesh.attributes):
        if a.name in keep or a.name.startswith(".") or a.is_internal or a.is_required:
            continue
        skin_mesh.attributes.remove(skin_mesh.attributes[a.name])

    arm = _armature_from_bones(base + "_Rig", bones, obj)
    skin = bpy.data.objects.new(skin_mesh.name, skin_mesh)
    coll = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    coll.objects.link(skin)
    skin.hide_render = True
    colours = skin_mesh.color_attributes
    if "Color" in colours:              # the Quixel channels are what an exporter writes
        colours.active_color = colours["Color"]
        colours.render_color_index = colours.find("Color")
    _write_weights(skin, groups)
    mod = skin.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    skin.parent = arm
    arm.parent = obj
    counts = {"stem": len(stem) - 1, "organ": len(bones) - len(stem), "root": 1,
              "verts": len(co), "stalks": len(stalks), "riders": len(riders)}
    arm["pgs_counts"] = counts
    # Which stem bone each bone-less flower rides, for the assembly's instances.
    arm["pgs_riders"] = json.dumps({str(oid): parents[oid] for oid in sorted(riders)})
    # Unreal Dynamic Wind's simulation groups, written beside an export.
    arm["pgs_wind"] = json.dumps(rl.wind_groups(bones))
    if preview:
        _build_fast_preview(obj, base, arm, skin, is_organ, bones, organ_bone, organs)
    # What the plant was built from: the same fingerprint of every input a
    # regrow reads that undo uses (`bridge.growth_signature`) -- a later edit
    # by any route (live update off, a preset, undo, a Day change) shows.
    arm["pgs_source"] = bridge.growth_signature(obj)
    # ...and without the Day: under Grow Over Timeline the timeline owns the
    # Day, so playing it is no edit (`rig_stale`).
    arm["pgs_source_settings"] = bridge.growth_signature(obj, include_day=False)
    return counts


def _write_weights(skin, groups: dict) -> None:
    """Every vertex group's weights, through bmesh's deform layer: `VertexGroup.add`
    walks the group list for each vertex it writes, so 17,800 groups x 5 million
    weights never finished. `groups`: name -> (vertex indices, weights); a vertex
    listed twice in one group keeps its largest weight."""
    import bmesh

    for name in groups:
        skin.vertex_groups.new(name=name)
    bm = bmesh.new()
    bm.from_mesh(skin.data)
    layer = bm.verts.layers.deform.verify()
    bm.verts.ensure_lookup_table()
    verts = bm.verts
    for number, (idx, w) in enumerate(groups.values()):
        weights = np.asarray(w)
        if weights.min() < weights.max():
            order = np.argsort(weights, kind="stable")
            idx, w = np.asarray(idx)[order].tolist(), weights[order].tolist()
        for i, x in zip(idx, w, strict=True):
            verts[i][layer][number] = x
    bm.to_mesh(skin.data)
    bm.free()


def _preview_names(base):
    return base + "_rigstems", base + "_rigtrackers"


def _build_fast_preview(obj, base, arm, skin, is_organ, bones, organ_bone, organs):
    """Fast Preview's two helpers: the skin without its organs (the stems and
    stalks, same weights) and the organ trackers -- per organ id three points
    (attach, along the organ's bone, across it) bound to its bone, their rest
    positions in `rest`, so GN can move each organ's live instance with its
    bone (`mesh._build_rig_fast_preview`). Ids with no organ sit at the origin
    on the root."""
    import bmesh

    from ..geometry import rig_layout as rl

    coll = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    stems_name, trackers_name = _preview_names(base)

    stems_mesh = skin.data.copy()
    stems_mesh.name = stems_name
    bm = bmesh.new()
    bm.from_mesh(stems_mesh)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(is_organ)[0]], context='VERTS')
    bm.to_mesh(stems_mesh)
    bm.free()
    stems = bpy.data.objects.new(stems_name, stems_mesh)
    coll.objects.link(stems)
    for g in skin.vertex_groups:                 # names: the weights came with the mesh
        if stems.vertex_groups.get(g.name) is None:
            stems.vertex_groups.new(name=g.name)

    by_name = {b.name: b for b in bones}
    top = max((o.oid for o in organs), default=-1)
    points = np.zeros(((top + 1) * 3, 3))
    owner = [rl.ROOT] * len(points)
    for o in organs:
        b = by_name.get(organ_bone.get(o.oid))
        if b is None:
            continue
        head = np.asarray(b.head, dtype=np.float64)
        along = np.asarray(b.tail, dtype=np.float64) - head
        length = max(float(np.linalg.norm(along)), 1e-3)
        along = along / max(float(np.linalg.norm(along)), 1e-12)
        side = np.cross(along, (0.0, 0.0, 1.0))
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(along, (1.0, 0.0, 0.0))
        side = side / np.linalg.norm(side)
        k = o.oid * 3
        points[k:k + 3] = (head, head + along * length, head + side * length)
        owner[k:k + 3] = [b.name] * 3
    tmesh = bpy.data.meshes.new(trackers_name)
    tmesh.vertices.add(len(points))
    tmesh.vertices.foreach_set("co", points.ravel())
    rest = tmesh.attributes.new("rest", 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set("vector", points.ravel())
    tmesh.update()
    trackers = bpy.data.objects.new(trackers_name, tmesh)
    coll.objects.link(trackers)
    groups = {}
    for i, bname in enumerate(owner):
        groups.setdefault(bname, []).append(i)
    for bname, idx in groups.items():
        trackers.vertex_groups.new(name=bname).add(idx, 1.0, 'REPLACE')

    for o in (stems, trackers):
        mod = o.modifiers.new("Armature", 'ARMATURE')
        mod.object = arm
        o.parent = arm
        o.hide_render = True


def fast_preview_parts(obj):
    """(stems skin, trackers) when Rig View should use Fast Preview -- it is
    on, Vertex Wind is off (it moves single vertices, which instances lack),
    and the rig has the parts; else None (the full skin)."""
    s = obj.plantgen
    if not s.rig_fast_preview or s.wind_secondary_enable or s.bloom_continuous:
        return None
    found = find_rig(obj)
    if found is None:
        return None
    base = found[0].name[:-len("_Rig")] if found[0].name.endswith("_Rig") else obj.name
    stems, trackers = (bpy.data.objects.get(n) for n in _preview_names(base))
    if stems is None or trackers is None:
        return None
    return stems, trackers


def _armature_from_bones(name, bones, obj):
    """The armature for `bones` (rig_layout.Bone), in the plant's own frame,
    with a bone collection per level and organ kind."""
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    coll = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    coll.objects.link(arm)
    arm_data.display_type = 'OCTAHEDRAL'
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    prev_active = bpy.context.view_layer.objects.active
    hidden = arm.hide_get()
    arm.hide_set(False)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm_data.edit_bones
    made = {}
    collections = {}
    for b in bones:
        eb = ebs.new(b.name)
        eb.head = Vector(b.head.tolist())
        eb.tail = Vector(b.tail.tolist())
        if b.roll_ref is not None and np.linalg.norm(b.roll_ref) > 1e-9:
            eb.align_roll(Vector(b.roll_ref.tolist()))
        bc = collections.get(b.group)
        if bc is None:
            bc = collections[b.group] = arm_data.collections.new(b.group)
        bc.assign(eb)
        if b.width > 0.0:
            eb["pgs_width"] = b.width
        if b.flex > 0.0:
            eb["pgs_flex"] = b.flex                 # Bone Wind: how far it gives
        if b.organ:
            eb["pgs_organ"] = True
        made[b.name] = (eb, b)
    for eb, b in made.values():
        if b.parent is not None and b.parent in made:
            eb.parent = made[b.parent][0]
            eb.use_connect = b.connect
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.hide_set(hidden)
    bpy.context.view_layer.objects.active = prev_active
    return arm


# -- Phase 2: skinning ---------------------------------------------------------

def _realize_full(obj):
    """A persistent mesh of the plant's FULL evaluated geometry, INCLUDING the GN
    instances (flowers / fruit / buds), via a throwaway Realize Instances
    modifier appended after the plant's own GN modifier.

    Measured necessity: `to_mesh()` and `convert` capture the realized tube and
    leaves but *not* GN instances (they came back flowerless), and a GN instance
    reports its `.object` as the plant itself, so the instances cannot be walked
    with identity either. A downstream Realize Instances node bakes them into the
    mesh, carrying each flower's `kind`/`organ_id` through -- which is exactly
    what the skin routes on. Caller owns the returned mesh.
    """
    ng = bpy.data.node_groups.new("PGS_rig_realize", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    rz = ng.nodes.new("GeometryNodeRealizeInstances")
    ng.links.new(gin.outputs[0], rz.inputs[0])
    from . import mesh as gn_mesh
    # Realized, so each leaf's own base-to-tip Red can join the per-organ
    # Green / Blue / Alpha in one Color (the shared Compose Vertex Colours).
    compose = gn_mesh.compose_vertex_colors_group()
    if compose is not None:
        node = ng.nodes.new("GeometryNodeGroup")
        node.node_tree = compose
        ng.links.new(rz.outputs[0], node.inputs["Geometry"])
        node.inputs[gn_mesh._DEAD_COLOR_SOCKET].default_value = \
            tuple(obj.plantgen.senescence_dead_color)
        ng.links.new(node.outputs["Geometry"], gout.inputs[0])
    else:
        ng.links.new(rz.outputs[0], gout.inputs[0])
    mod = obj.modifiers.new("PGS_rig_realize", 'NODES')
    mod.node_group = ng
    debug_on = obj.plantgen.debug_view_mode != 'off'
    if debug_on:
        gn_mesh.push_debug(obj, off=True)   # never bake / export a debug colour
    from . import timelapse_cache
    timelapse_cache.hide(obj)               # the live plant, never a recorded frame
    extras = _modifiers_after_plant(obj, mod)
    for extra in extras:
        extra.show_viewport = False
    try:
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(
            obj.evaluated_get(deps), preserve_all_data_layers=True, depsgraph=deps)
    finally:
        for extra in extras:
            extra.show_viewport = True
        obj.modifiers.remove(mod)
        bpy.data.node_groups.remove(ng)
        if debug_on:
            gn_mesh.push_debug(obj)
    return me


def _modifiers_after_plant(obj, realize) -> list:
    """The visible modifiers stacked after the plant's own, less `realize`: what
    a user added for the viewport (a ruler) is not the plant, and realized with
    it the geometry would land in the skin, the rig and every export."""
    from . import mesh as gn_mesh
    mods = list(obj.modifiers)
    plant = obj.modifiers.get(gn_mesh._MESH_GROUP_NAME)
    if plant is None:
        return []
    return [m for m in mods[mods.index(plant) + 1:] if m != realize and m.show_viewport]


_PREVIEW_MODS = ("PGS Leaf Wind", "PGS Skin VCol Debug")
"""Preview-only skin modifiers an export mutes."""


RIG_EXPORT_FORMATS = {".fbx": "FBX", ".usdc": "USD"}


def _op_kwargs(op, wanted: dict) -> dict:
    """`wanted`, less any option this Blender's exporter does not have."""
    props = {p.identifier for p in op.get_rna_type().properties}
    return {k: v for k, v in wanted.items() if k in props}


EXPORT_ATTRIBUTES = ("Color",)
"""What an exported skin carries beyond its geometry, UVs, materials and
weights: the Quixel channels. The plant's working attributes (load, tube,
organ ids...) stay home."""


def _export_mesh(skin):
    """A copy of the skin's mesh with only what an engine reads: geometry,
    UVs, materials, weights (vertex groups live on the mesh) and
    `EXPORT_ATTRIBUTES`. The caller removes it."""
    me = skin.data.copy()
    me.name = skin.data.name + "_export"
    keep = {"position", "material_index", "sharp_face", "sharp_edge", *EXPORT_ATTRIBUTES}
    keep |= {uv.name for uv in me.uv_layers}
    for a in list(me.attributes):
        if a.name in keep or a.name.startswith(".") or a.is_internal or a.is_required:
            continue
        me.attributes.remove(me.attributes[a.name])
    if "Color" in me.color_attributes:
        me.color_attributes.active_color = me.color_attributes["Color"]
        me.color_attributes.render_color_index = me.color_attributes.find("Color")
    return me


def _write_texture(image, folder) -> str | None:
    """Write `image` into `folder` (a packed one from its bytes): its file name,
    or None if it has no pixels on hand."""
    source = bpy.path.abspath(image.filepath) if image.filepath else ""
    name = os.path.basename(source) or bpy.path.clean_name(image.name) + ".png"
    target = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    if image.packed_file is not None:
        with open(target, "wb") as fh:
            fh.write(image.packed_file.data)
    elif os.path.isfile(source):
        if os.path.normcase(source) != os.path.normcase(target):
            shutil.copyfile(source, target)
    else:
        return None
    return name


def _surface_of(mat):
    """A PlantGen material's surface as (image, RGB, roughness): its Base Color
    texture, else the shader group's colour."""
    nt = mat.node_tree if mat.use_nodes else None
    colour, rough, image = tuple(mat.diffuse_color)[:3], 0.5, None
    if nt is None:
        return image, colour, rough
    node = nt.nodes.get("PGS Base Color")
    if node is not None and node.type == 'TEX_IMAGE' and node.image is not None \
            and any(s.is_linked for s in node.outputs):
        image = node.image
    group = next((n for n in nt.nodes if n.type == 'GROUP'), None)
    if group is not None:
        base = group.inputs.get("Base Color")
        if base is not None and not base.is_linked:
            colour = tuple(base.default_value)[:3]
        sock = group.inputs.get("Roughness")
        if sock is not None and not sock.is_linked:
            rough = float(sock.default_value)
    return image, colour, rough


def author_usd_materials(filepath, materials) -> int:
    """Give a written USD file real materials. Blender's exporter cannot see
    through PlantGen's shader groups and leaves its Material prims empty (an
    importer then falls back to one flat colour, and a Nanite Assembly's parts
    all merge into one slot): each becomes a UsdPreviewSurface -- its Base Color
    texture, copied to `textures/` beside the file, or its colour. Never an
    opacity: UE 5.7's importer makes any textured opacity translucent (it has
    no masked material), which Nanite cannot draw -- the renderer asserts and
    the editor goes down. Organs are cut-out meshes; the outline is geometry.
    Returns how many were written."""
    from pxr import Sdf, Tf, Usd, UsdShade

    stage = Usd.Stage.Open(filepath)
    prims = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdShade.Material)}
    folder = os.path.join(os.path.dirname(os.path.abspath(filepath)), "textures")
    float_, color3 = Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Color3f
    done = 0
    for mat in materials:
        prim = prims.get(Tf.MakeValidIdentifier(mat.name)) if mat is not None else None
        if prim is None:
            continue
        for child in list(prim.GetChildren()):
            stage.RemovePrim(child.GetPath())
        image, colour, rough = _surface_of(mat)
        path = prim.GetPath()
        pbr = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", float_).Set(rough)
        pbr.CreateInput("metallic", float_).Set(0.0)
        pbr.CreateInput("diffuseColor", color3).Set(tuple(float(c) for c in colour))
        name = _write_texture(image, folder) if image is not None else None
        if name is not None:
            reader = UsdShade.Shader.Define(stage, path.AppendChild("UV"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(stage, path.AppendChild("BaseColor"))
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{name}")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result")
            rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr.GetInput("diffuseColor").ConnectToSource(rgb)
        output = UsdShade.Material(prim).CreateSurfaceOutput()
        output.ConnectToSource(pbr.ConnectableAPI(), "surface")
        done += 1
    stage.GetRootLayer().Save()
    return done


def author_usd_joint_axes(filepath) -> int:
    """Turn every skeleton's joints so +X runs along the bone (Blender's bones
    run along +Y): Unreal's Dynamic Wind takes a bone's X as its forward, bending
    across it and twisting about it, so on Y-bones the twist was a sideways
    swing -- worst on a flower at the end of a long stalk. Bind and rest change
    together, so nothing moves. One-joint skeletons (assembly parts) are left
    alone. Returns the joints turned."""
    from pxr import Gf, Usd, UsdSkel, Vt

    stage = Usd.Stage.Open(filepath)
    turn = Gf.Matrix4d((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))   # new X = old Y
    done = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdSkel.Skeleton):
            continue
        skel = UsdSkel.Skeleton(prim)
        joints = [str(j) for j in skel.GetJointsAttr().Get() or []]
        if len(joints) < 2:
            continue
        index = {j: i for i, j in enumerate(joints)}
        bind = [turn * m for m in skel.GetBindTransformsAttr().Get()]
        rest = []
        for i, joint in enumerate(joints):
            parent = index.get(joint.rsplit("/", 1)[0]) if "/" in joint else None
            rest.append(bind[i] if parent is None else bind[i] * bind[parent].GetInverse())
        skel.GetBindTransformsAttr().Set(Vt.Matrix4dArray(bind))
        skel.GetRestTransformsAttr().Set(Vt.Matrix4dArray(rest))
        done += len(joints)
    stage.GetRootLayer().Save()
    return done


USD_EXPORT_OPTIONS = dict(selected_objects_only=True, export_armatures=True,
                          export_animation=False, export_meshes=True,
                          export_materials=True, export_uvmaps=True)
"""The USD exporter's options for a rig: the selection only, the bind pose."""


@contextlib.contextmanager
def export_scene(arm, skin, export_mesh, extra=()):
    """The rig ready to export, restored after: the armature in the plant's own
    frame (root at the origin) and its rest pose, the skin wearing
    `export_mesh` (only what an engine reads) with its preview modifiers
    muted, and those two plus `extra` objects visible, renderable (USD
    exports in render mode) and the only ones selected."""
    layer = bpy.context.view_layer
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    prev_sel = list(bpy.context.selected_objects)
    prev_active = layer.objects.active
    # Render visibility too: the helpers are hidden from renders (the plant
    # renders the skin in Rig View), and USD exports in render mode.
    prev_hidden = {o: (o.hide_viewport, o.hide_get(), o.hide_select, o.hide_render)
                   for o in (arm, skin)}
    prev_parent, prev_pose = arm.parent, arm.data.pose_position
    prev_mesh = skin.data
    muted = []
    for nm in _PREVIEW_MODS:
        m = skin.modifiers.get(nm)
        if m is not None:
            muted.append((m, m.show_viewport, m.show_render))
            m.show_viewport = m.show_render = False
    try:
        for o in prev_hidden:
            o.hide_viewport = False
            o.hide_set(False)
            o.hide_select = False
            o.hide_render = False
        arm.parent = None                       # the plant's frame: root at the origin
        arm.data.pose_position = 'REST'         # the bind pose, whatever is posed
        skin.data = export_mesh
        for o in prev_sel:
            o.select_set(False)
        for o in (arm, skin, *extra):
            o.select_set(True)
        layer.objects.active = arm
        layer.update()
        yield
    finally:
        skin.data = prev_mesh
        arm.data.pose_position = prev_pose
        arm.parent = prev_parent
        for m, sv, sr in muted:
            m.show_viewport, m.show_render = sv, sr
        for o in bpy.context.selected_objects:
            o.select_set(False)
        for o, (viewport, hidden, unselectable, unrendered) in prev_hidden.items():
            o.hide_viewport = viewport
            o.hide_set(hidden)
            o.hide_select = unselectable
            o.hide_render = unrendered
        for o in prev_sel:
            try:
                o.select_set(True)
            except (ReferenceError, RuntimeError):
                pass
        if prev_active is not None:
            layer.objects.active = prev_active


def export_rig(obj, filepath):
    """Export `obj`'s rig as a static skeletal mesh -- the armature and the
    skinned mesh, in the bind (rest) pose -- to FBX or USD by the file's
    extension. Engine-first: the plant's own frame (its root at the origin,
    whatever the plant's placement in the scene), the deform bones only, no
    leaf bones, the skin's weights, UVs, materials and the `Color` channels
    written raw (LINEAR: they are data, not colours) -- and nothing else of
    the plant's working attributes (`_export_mesh`). Selection, visibility,
    parenting, the pose and the skin's own mesh are restored afterwards.
    Returns a summary."""
    found = find_rig(obj)
    if found is None:
        return {"error": "no rig"}
    arm, skin = found
    fmt = RIG_EXPORT_FORMATS.get(os.path.splitext(filepath)[1].lower())
    if fmt is None:
        return {"error": f"unknown format: {filepath}"}
    export_mesh = _export_mesh(skin)
    renamed = []
    try:
        with export_scene(arm, skin, export_mesh):
            if fmt == "FBX":
                # Unreal's FBX import makes the armature OBJECT an extra root
                # bone unless it is named "Armature" -- which would put a joint
                # in the skeleton the wind file does not name.
                other = bpy.data.objects.get("Armature")
                if other is not None and other is not arm:
                    renamed.append((other, other.name))
                    other.name = "Armature__pgs_export"
                renamed.append((arm, arm.name))
                arm.name = "Armature"
                op = bpy.ops.export_scene.fbx
                kw = _op_kwargs(op, dict(
                    use_selection=True, object_types={'ARMATURE', 'MESH'},
                    use_mesh_modifiers=False, add_leaf_bones=False, bake_anim=False,
                    use_armature_deform_only=True, primary_bone_axis='X',
                    secondary_bone_axis='-Y', mesh_smooth_type='FACE', colors_type='LINEAR',
                    use_custom_props=False))
            else:
                op = bpy.ops.wm.usd_export
                kw = _op_kwargs(op, USD_EXPORT_OPTIONS)
            res = op(filepath=filepath, **kw)
    finally:
        for o, nm in reversed(renamed):
            o.name = nm
        bpy.data.meshes.remove(export_mesh)
    if fmt == "USD":
        author_usd_materials(filepath, list(skin.data.materials))
        author_usd_joint_axes(filepath)
    counts = dict(arm.get("pgs_counts", {}))
    wind_path = write_wind_json(obj, os.path.splitext(filepath)[0] + "_wind.json")
    return {"result": list(res), "format": fmt, "filepath": filepath, "wind": wind_path,
            "bones": len(arm.data.bones), "verts": len(skin.data.vertices), **counts}


def write_wind_json(obj, path):
    """Write Unreal Dynamic Wind's import file for `obj`'s rig (every bone by
    name, its simulation group; the groups' influences). Returns the path,
    or None for a rig built before the groups existed."""
    from ..geometry import rig_layout as rl
    found = find_rig(obj)
    if found is None or "pgs_wind" not in found[0]:
        return None
    arm = found[0]
    wind = json.loads(arm["pgs_wind"])
    data = rl.wind_json(wind, [b.name for b in arm.data.bones])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return path


def wind_summary(obj):
    """The rig's wind groups for the panel: [(name, settings, bone count)]."""
    found = find_rig(obj)
    if found is None or "pgs_wind" not in found[0]:
        return []
    wind = json.loads(found[0]["pgs_wind"])
    counts = {}
    for i in wind["bones"].values():
        counts[i] = counts.get(i, 0) + 1
    return [(g["name"], g["settings"], counts.get(i, 0)) for i, g in enumerate(wind["groups"])]


# -- Phase 5: locate / replace a plant's baked rig ----------------------------

def _rig_names(obj):
    return obj.name + "_Rig", obj.name + "_skin"


def find_rig(obj):
    """The (armature, skin) objects of `obj`'s baked rig, or None if not baked.
    The armature is found by its deterministic name; the skin is found as the
    armature's mesh child rather than by name, so a `.NNN` suffix that a datablock
    name-collision may have forced onto the skin object does not hide the rig."""
    rig_name, skin_name = _rig_names(obj)
    rig_obj = bpy.data.objects.get(rig_name)
    if rig_obj is None or rig_obj.type != 'ARMATURE':
        return None
    skin_obj = bpy.data.objects.get(skin_name)
    if skin_obj is None:
        skin_obj = next((c for c in rig_obj.children if c.type == 'MESH'), None)
    return (rig_obj, skin_obj) if skin_obj is not None else None


def apply_rig_view(obj):
    """Show `obj` as its baked rig (Rig View on) or as the live plant (off).
    The plant is always the one visible object: in Rig View its GN output IS
    the skin, posed by the bones (`mesh.push_rig_view`); the skin itself stays
    hidden (`hide_set`, so it still evaluates for Object Info), the armature
    shows only with Show Bones. Off, both drop out of the depsgraph
    (`hide_viewport`): an unviewed rig costs nothing. Returns True if a rig
    is shown."""
    from . import mesh
    s = obj.plantgen
    found = find_rig(obj)
    on = bool(s.rig_view and found)
    if found is not None:
        arm, skin = found
        layer = bpy.context.view_layer
        base = arm.name[:-len("_Rig")] if arm.name.endswith("_Rig") else obj.name
        parts = [bpy.data.objects.get(n) for n in _preview_names(base)]
        fast = on and fast_preview_parts(obj) is not None
        # Only the skin in use stays evaluated: with Fast Preview the full
        # skin (250k vertices of realized organs) leaves the depsgraph -- even
        # hidden it would re-deform every frame.
        helpers = [(arm, on, on and s.rig_show_bones), (skin, on and not fast, False)]
        helpers += [(o, fast, False) for o in parts if o is not None]
        for o, evaluated, visible in helpers:
            o.hide_viewport = not evaluated
            o.hide_render = True
            o.hide_select = True
            if o.name in layer.objects:
                o.hide_set(not visible)
        if on:
            arm.show_in_front = True
        arm.data.display_type = s.rig_bone_display
    mesh.push_rig_view(obj)
    return on


def leave_rig_view(obj, stale: bool = True) -> None:
    """`obj` is about to regrow: show the live plant again, and -- for an edit
    (`stale`) -- mark the baked rig out of date (it no longer matches the
    settings). A Grow Over Timeline frame passes `stale=False`: playing the
    timeline changes the Day, not the plant's settings."""
    s = obj.plantgen
    if not s.rig_view:
        return
    if stale:
        obj["plantgen_rig_stale"] = True
    s.rig_view = False                      # its update applies the view


def rig_stale(obj) -> bool:
    """Whether the plant changed since its rig was built: an edit that left
    Rig View (`leave_rig_view`), or any setting a regrow reads differing from
    the fingerprint Build Rig recorded -- the Day among them, unless Grow Over
    Timeline drives it (then the timeline's frame is not a change)."""
    if obj.get("plantgen_rig_stale", False):
        return True
    found = find_rig(obj)
    if found is None or "pgs_source" not in found[0]:
        return False
    from . import bridge
    arm = found[0]
    if obj.plantgen.animate_day and "pgs_source_settings" in arm:
        return arm["pgs_source_settings"] != bridge.growth_signature(obj, include_day=False)
    return arm["pgs_source"] != bridge.growth_signature(obj)


def clear_rig(obj):
    """Remove any previously-baked rig + skin for `obj` -- objects, their data and
    the action -- so a re-bake starts clean. Matches the deterministic base names
    *and* any `.NNN` suffix a datablock collision may have added, and purges
    orphaned rig/skin data (a mesh/armature left behind by a manual delete): a
    lingering datablock of the same name forces the fresh skin object to
    `_skin.001`, which name-based lookup then misses. Returns objects removed."""
    removed = 0
    pats = [re.compile(re.escape(b) + r"(\.\d+)?$")
            for b in (*_rig_names(obj), *_preview_names(obj.name))]
    matches = [o for o in bpy.data.objects if any(p.match(o.name) for p in pats)]
    for o in matches:
        data = o.data
        act = (o.animation_data.action if o.animation_data else None)
        bpy.data.objects.remove(o, do_unlink=True)
        removed += 1
        if act is not None and act.users == 0:
            bpy.data.actions.remove(act)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Armature):
                bpy.data.armatures.remove(data)
    # Purge orphaned rig/skin data whose name would otherwise collide and push the
    # fresh bake to a `.NNN` suffix.
    for datacoll in (bpy.data.meshes, bpy.data.armatures):
        for d in list(datacoll):
            if d.users == 0 and any(p.match(d.name) for p in pats):
                datacoll.remove(d)
    return removed
