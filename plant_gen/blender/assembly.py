"""Nanite Assembly export.

A skeletal-mesh Nanite Assembly in one USD file, built from the plant's rig
(Build Rig): the skinned **base mesh** -- the stems, stalks and any organ that
is unique geometry -- on the rig's skeleton, and every organ that is one of its
component's meshes as an **instance** of that mesh, scattered by a
`UsdGeomPointInstancer` and bound rigidly to the organ's own bone. Epic's
`NaniteAssembly*` schemas are stamped on with `pxr` after Blender's own USD
export (no exporter writes them). The wind JSON names the same joints as
Export Rig's, so Dynamic Wind applies unchanged.

Which mesh an organ is, and where, is read off the geometry, not by replaying
the node graph's pick: the organ's realized vertices are fitted to every
component mesh with as many vertices (a similarity transform, Umeyama). An
exact fit names the prototype and gives the instance transform; no fit (a leaf
still unfolding is drawn deformed) leaves the organ in the base mesh. Flowers
take their discrete bloom stage -- the plant is realized with Smooth Bloom off
for this, so a blended flower snaps to its stage.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile

import bpy
import numpy as np

from ..geometry import rig_layout as rl
from . import rig

FIT_ABS = 1e-5
FIT_REL = 1e-4
"""An organ is a mesh's instance when every vertex lands within FIT_ABS metres
plus FIT_REL of the organ's own size (the leaves fit to ~3e-8 m)."""

BIND_JOINTS = "primvars:unreal:naniteAssembly:bindJoints"
ROOT_API = "NaniteAssemblyRootAPI"
BINDING_API = "NaniteAssemblySkelBindingAPI"
MESH_TYPE = "unreal:naniteAssembly:meshType"
SKELETON_REL = "unreal:naniteAssembly:skeleton"
INSTANCER = "Organs"


def _fit(P, R):
    """The similarity transform taking points P onto R (Umeyama): scale,
    rotation (3x3), translation and the largest residual."""
    mp, mr = P.mean(axis=0), R.mean(axis=0)
    Pc, Rc = P - mp, R - mr
    U, S, Vt = np.linalg.svd(Pc.T @ Rc)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
    D = np.array([1.0, 1.0, d])
    rot = (U * D) @ Vt
    rot = rot.T
    var = float((Pc ** 2).sum())
    s = float((S * D).sum() / var) if var > 0.0 else 1.0
    t = mr - s * rot @ mp
    res = float(np.abs(s * (P @ rot.T) + t - R).max())
    return s, rot, t, res


def _prototype_meshes(s):
    """Every component mesh an organ can be drawn as, one per mesh datablock,
    in a stable order: [(object, local vertex positions)]."""
    seen, out = set(), []
    for c in s.components:
        col = getattr(c, "collection", None)
        if col is None:
            continue
        for ob in sorted(col.all_objects, key=lambda o: o.name):
            if ob.type != 'MESH' or ob.data.name in seen or not len(ob.data.vertices):
                continue
            seen.add(ob.data.name)
            co = np.empty(len(ob.data.vertices) * 3)
            ob.data.vertices.foreach_get("co", co)
            out.append((ob, co.reshape(-1, 3)))
    return out


def _realize_discrete(obj):
    """The live plant realized (never Rig View's skin) with every flower on its
    discrete bloom stage; the plant's sockets are pushed back after."""
    from . import mesh as gn_mesh
    mod = obj.modifiers.get(gn_mesh._MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return None
    ng = mod.node_group
    for socket in (gn_mesh._RIG_VIEW_SOCKET, gn_mesh._BLOOM_CONTINUOUS_SOCKET):
        gn_mesh._set_modifier_input(mod, gn_mesh._socket_identifier(ng, socket), False)
    obj.update_tag()
    try:
        return rig._realize_full(obj)
    finally:
        gn_mesh.push_bloom(obj)
        gn_mesh.push_rig_view(obj)


_ORGAN_BONE = re.compile(r"^(leaf|bract|flower|bud|apex|fruit|organ)_(\d+)$")
"""Build Rig's organ bone names: `<kind>_<organ id>`."""


def _skin_organs(skin, riders=None):
    """organ id -> (bone, skin vertex indices), one entry per organ bone and per
    flower riding a stem bone (`riders`: organ id -> that bone). An organ's
    vertices are those carrying its id -- except that the skin keeps no kind /
    width to tell stems by, and its stems carry organ id 0 too: for that id,
    only the vertices weighted most to the organ's own bone. (By weight alone a
    leaf loses the few vertices its stalk's foot shares.) A rider with id 0 has
    no bone of its own to tell it by, so it stays in the base mesh."""
    me = skin.data
    ids = np.zeros(len(me.vertices), dtype=np.int64)
    attr = me.attributes.get("organ_id")
    if attr is not None and attr.domain == 'POINT':
        attr.data.foreach_get("value", ids)
    out = {}
    verts = rl.group_indices(ids)
    for g in skin.vertex_groups:
        m = _ORGAN_BONE.match(g.name)
        if m is None:
            continue
        oid = int(m.group(2))
        idx = verts.get(oid, np.empty(0, dtype=np.int64))
        if oid == 0:
            idx = np.array([i for i in idx
                            if max(me.vertices[int(i)].groups, key=lambda x: x.weight,
                                   default=None) is not None
                            and max(me.vertices[int(i)].groups,
                                    key=lambda x: x.weight).group == g.index], dtype=np.int64)
        if len(idx):
            out[oid] = (g.name, idx)
    for oid, bone in (riders or {}).items():
        if oid != 0 and oid in verts:
            out[oid] = (bone, verts[oid])
    return out


def instances(obj):
    """The plan of the assembly: [(organ id, prototype index, scale, rotation,
    translation, bone)], the prototypes [(object, points)], the skin vertices
    that stay in the base mesh (a bool mask) and the organs left there."""
    arm, skin = rig.find_rig(obj)
    riders = {int(k): v for k, v in json.loads(arm.get("pgs_riders", "{}")).items()}
    protos = _prototype_meshes(obj.plantgen)
    disc = _realize_discrete(obj)
    if disc is None:
        raise rig.RigError("The plant has no PlantGen modifier")
    try:
        d_arr = rig._skin_arrays(disc)
    finally:
        bpy.data.meshes.remove(disc)
    # The fresh realize still carries kind / width: organs are what is not stem.
    d_organ = (d_arr["kind"] != 0) | (d_arr["width"] <= 1e-9)
    d_ids = d_arr["organ_id"]
    disc_organs = rl.group_indices(d_ids, d_organ)

    by_count: dict[int, list[int]] = {}
    for k, (_ob, pts) in enumerate(protos):
        by_count.setdefault(len(pts), []).append(k)
    keep = np.ones(len(skin.data.vertices), dtype=bool)
    out, unique = [], []
    for oid, (bone, s_idx) in sorted(_skin_organs(skin, riders).items()):
        d_idx = disc_organs.get(oid)
        R = d_arr["co"][d_idx] if d_idx is not None else None
        best = None
        if R is not None and len(d_idx) == len(s_idx):
            for k in by_count.get(len(R), ()):
                fit = _fit(protos[k][1], R)
                if best is None or fit[3] < best[1][3]:
                    best = (k, fit)
        size = float(np.ptp(R, axis=0).max()) if R is not None and len(R) else 0.0
        if best is None or best[1][3] > FIT_ABS + FIT_REL * size:
            unique.append(oid)
            continue
        k, (sc, rot, t, _res) = best
        out.append((oid, k, sc, rot, t, bone))
        keep[s_idx] = False
    return out, protos, keep, unique


def _base_mesh(skin, keep):
    """The skin's export mesh less every instanced organ's vertices."""
    import bmesh
    me = rig._export_mesh(skin)
    if keep.all():
        return me
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v, k in zip(bm.verts, keep, strict=True) if not k],
                     context='VERTS')
    bm.to_mesh(me)
    bm.free()
    # Only the slots it still uses: USD export binds a one-material mesh to the
    # first filled slot, whichever its faces use (the stems wore the flower's).
    used = {p.material_index for p in me.polygons}
    for i in reversed(range(len(me.materials))):
        if i not in used:
            me.materials.pop(index=i)
    return me


def _prototype_object(src, name):
    """A copy of a component mesh at the origin, with only what an engine
    reads (geometry, UVs, materials, Color)."""
    me = src.data.copy()
    me.name = name
    keep = {"position", "material_index", "sharp_face", "sharp_edge", *rig.EXPORT_ATTRIBUTES}
    keep |= {uv.name for uv in me.uv_layers}
    for a in list(me.attributes):
        if a.name in keep or a.name.startswith(".") or a.is_internal or a.is_required:
            continue
        me.attributes.remove(me.attributes[a.name])
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def export_assembly(obj, filepath):
    """Write `obj`'s rig as a skeletal-mesh Nanite Assembly (USD) and the
    Dynamic Wind JSON beside it. A plant with nothing to instance (Corylus: bare
    wood and catkins, which are stem mesh) has no parts, and an assembly needs
    parts, so it stays a plain skinned mesh. Returns a summary (or {"error": ...})."""
    found = rig.find_rig(obj)
    if found is None:
        return {"error": "no rig"}
    if os.path.splitext(filepath)[1].lower() != ".usdc":
        return {"error": f"an assembly is binary USD (.usdc): {filepath}"}
    arm, skin = found
    plan, protos, keep, unique = instances(obj)
    used = sorted({k for _o, k, *_r in plan})
    names = {k: f"PGS_Proto_{i:02d}_{protos[k][0].name}" for i, k in enumerate(used)}
    proto_objs = {k: _prototype_object(protos[k][0], names[k]) for k in used}
    base = _base_mesh(skin, keep)
    try:
        with rig.export_scene(arm, skin, base, extra=list(proto_objs.values())):
            res = bpy.ops.wm.usd_export(filepath=filepath, **rig._op_kwargs(
                bpy.ops.wm.usd_export, rig.USD_EXPORT_OPTIONS))
    finally:
        for ob in proto_objs.values():
            me = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.meshes.remove(me)
        base_verts = len(base.vertices)
        bpy.data.meshes.remove(base)
    if 'FINISHED' not in res:
        return {"error": f"USD export: {set(res)}"}
    if plan:
        author(filepath, plan, [names[k] for k in used], {k: i for i, k in enumerate(used)})
    materials = [*skin.data.materials, *(m for k in used for m in protos[k][0].data.materials)]
    rig.author_usd_materials(filepath, materials)
    rig.author_usd_joint_axes(filepath)
    wind = rig.write_wind_json(obj, os.path.splitext(filepath)[0] + "_wind.json")
    return {"filepath": filepath, "wind": wind, "instances": len(plan),
            "prototypes": len(used), "unique_organs": len(unique),
            "base_verts": base_verts, "bones": len(arm.data.bones)}


def combine(parts, filepath, name) -> int:
    """Flatten written assemblies into one USD file: `parts` [(prim name, path)],
    each its own NaniteAssemblyRoot under one `name` prim. Unreal's asset cache
    lasts one import and shares what hashes alike within it, so one file is
    what lets plants share their leaf and flower meshes, materials and textures
    (10 Rudbeckia: 124 part meshes as files, 13 as one). Texture paths stay
    `./textures/...`. Returns how many assemblies the file holds."""
    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    top = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(top, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(top, 1.0)
    root = UsdGeom.Xform.Define(top, Sdf.Path.absoluteRootPath.AppendChild(name)).GetPrim()
    top.SetDefaultPrim(root)
    for prim_name, path in parts:
        prim = UsdGeom.Xform.Define(top, root.GetPath().AppendChild(prim_name)).GetPrim()
        prim.GetReferences().AddReference(path.replace("\\", "/"))
    layer = top.Flatten()

    def relative(asset):
        marker = "textures/"
        cut = asset.replace("\\", "/").rfind(marker)
        return asset if cut < 0 else "./" + asset.replace("\\", "/")[cut:]

    UsdUtils.ModifyAssetPaths(layer, relative)
    layer.Export(filepath)
    stage = Usd.Stage.Open(filepath)
    return sum(ROOT_API in _api_schemas(p) for p in stage.Traverse())


def export_variations(source, filepath, progress=None, plants=None):
    """A plant and its built variations as one USD file of Nanite Assemblies
    (`combine`), with one wind JSON per plant and `textures/` beside it: each
    has its own skeleton, and Dynamic Wind's import takes one file per mesh.
    A plant without a current rig gets one for the export, and loses it after
    if it had none. `plants`: another set than the source and its variations.
    Returns a summary (or {"error": ...})."""
    from pxr import Tf

    from . import bridge, variations

    if os.path.splitext(filepath)[1].lower() != ".usdc":
        return {"error": f"an assembly is binary USD (.usdc): {filepath}"}
    plants = plants or [source, *variations.generated(source)]
    folder = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filepath))[0]
    work = tempfile.mkdtemp(prefix=f"{stem}_parts_", dir=folder)
    parts, instances_total, failed, assemblies = [], 0, {}, 0
    try:
        for i, plant in enumerate(plants):
            if progress is not None:
                progress(i, len(plants))
            fresh = rig.find_rig(plant) is None
            if fresh or rig.rig_stale(plant):
                bridge.regrow(plant, weld=bridge.WELD_NOW)
                rig.build_rig(plant, preview=not fresh)
            name = Tf.MakeValidIdentifier(plant.name)
            path = os.path.join(work, name + "_assembly.usdc")
            out = export_assembly(plant, path)
            if "error" in out:
                return {"error": f"{plant.name}: {out['error']}"}
            if out["instances"]:
                assemblies += 1
                bad = {k: v[1] for k, v in verify(path, out["instances"]).items() if not v[0]}
                if bad:
                    failed[plant.name] = bad
            instances_total += out["instances"]
            parts.append((name, path))
            if fresh:
                rig.clear_rig(plant)
                bridge.regrow(plant)
        if failed:
            return {"error": f"checks failed: {failed}"}
        held = combine(parts, filepath, Tf.MakeValidIdentifier(stem))
        if held != assemblies:
            return {"error": f"the file holds {held} of {assemblies} assemblies"}
        for entry in os.listdir(work):
            src = os.path.join(work, entry)
            if entry.endswith("_wind.json"):
                shutil.move(src, os.path.join(folder, entry))
        textures = os.path.join(work, "textures")
        if os.path.isdir(textures):             # a plant of plain colours writes none
            shutil.copytree(textures, os.path.join(folder, "textures"), dirs_exist_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"filepath": filepath, "plants": len(plants), "instances": instances_total,
            "assemblies": assemblies}


def _find(stage, type_name, name=None):
    for p in stage.Traverse():
        if p.GetTypeName() == type_name and (name is None or p.GetName() == name):
            return p
    return None


def author(filepath, plan, proto_names, proto_index):
    """Reshape Blender's export into the assembly: each prototype mesh moved
    under the instancer as a SkelRoot with a one-joint skeleton, the
    PointInstancer with every organ's transform, prototype and bone, and the
    NaniteAssembly schemas applied."""
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdSkel, Vt

    stage = Usd.Stage.Open(filepath)
    layer = stage.GetRootLayer()
    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()))
    skel_prim = _find(stage, "Skeleton")
    skel = UsdSkel.Skeleton(skel_prim)
    # Bind by the joint's own name, with `jointNames` authored: UE 5.7's legacy
    # importer matches those first, and Interchange matches only leaf names --
    # a joint it cannot find silently binds its part to the first bone.
    names = [str(j).rsplit("/", 1)[-1] for j in skel.GetJointsAttr().Get()]
    skel.CreateJointNamesAttr(Vt.TokenArray(names))
    known = set(names)

    inst_path = root.GetPath().AppendChild(INSTANCER)
    instancer = UsdGeom.PointInstancer.Define(stage, inst_path)
    protos_path = inst_path.AppendChild("Prototypes")
    UsdGeom.Scope.Define(stage, protos_path)
    targets = []
    for name in proto_names:
        valid = Tf.MakeValidIdentifier(name)
        src = _find(stage, "Mesh", valid) or _find(stage, "Mesh", valid + "_001")
        holder = src.GetParent() if src is not None else None
        part = protos_path.AppendChild(valid)
        UsdSkel.Root.Define(stage, part)
        pivot = UsdSkel.Skeleton.Define(stage, part.AppendChild("Skel"))
        pivot.CreateJointsAttr(Vt.TokenArray(["pivot"]))
        pivot.CreateBindTransformsAttr(Vt.Matrix4dArray([Gf.Matrix4d(1.0)]))
        pivot.CreateRestTransformsAttr(Vt.Matrix4dArray([Gf.Matrix4d(1.0)]))
        mesh_path = part.AppendChild("Mesh")
        Sdf.CopySpec(layer, src.GetPath(), layer, mesh_path)
        if holder is not None and holder != root:
            stage.RemovePrim(holder.GetPath())
        mesh = stage.GetPrimAtPath(mesh_path)
        binding = UsdSkel.BindingAPI.Apply(mesh)
        binding.CreateSkeletonRel().SetTargets([pivot.GetPath()])
        binding.CreateJointIndicesPrimvar(True, 1).Set(Vt.IntArray([0]))
        binding.CreateJointWeightsPrimvar(True, 1).Set(Vt.FloatArray([1.0]))
        targets.append(part)
    instancer.CreatePrototypesRel().SetTargets(targets)

    positions, orients, scales, indices, joints = [], [], [], [], []
    from mathutils import Matrix
    for _oid, k, sc, rot, t, bone in plan:
        q = Matrix([list(r) for r in rot]).to_quaternion()
        positions.append(Gf.Vec3f(*map(float, t)))
        orients.append(Gf.Quath(float(q.w), Gf.Vec3h(float(q.x), float(q.y), float(q.z))))
        scales.append(Gf.Vec3f(float(sc), float(sc), float(sc)))
        indices.append(proto_index[k])
        name = Tf.MakeValidIdentifier(bone)
        if name not in known:
            raise rig.RigError(f"Bone '{bone}' is not in the exported skeleton")
        joints.append(name)
    instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
    instancer.CreateOrientationsAttr(Vt.QuathArray(orients))
    instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
    instancer.CreateProtoIndicesAttr(Vt.IntArray(indices))

    # Epic's schemas are not registered in Blender's USD: applied by name.
    root.AddAppliedSchema(ROOT_API)
    root.CreateAttribute(MESH_TYPE, Sdf.ValueTypeNames.Token, False,
                         Sdf.VariabilityUniform).Set("skeletalMesh")
    root.CreateRelationship(SKELETON_REL, False).SetTargets([skel_prim.GetPath()])
    ip = instancer.GetPrim()
    ip.AddAppliedSchema(BINDING_API)
    pv = UsdGeom.PrimvarsAPI(ip).CreatePrimvar(
        BIND_JOINTS[len("primvars:"):], Sdf.ValueTypeNames.TokenArray, UsdGeom.Tokens.vertex, 1)
    pv.Set(Vt.TokenArray(joints))
    layer.Save()


def _api_schemas(prim) -> list:
    """The API schemas authored on `prim`, registered or not (Blender's USD
    does not know Epic's, so `GetAppliedSchemas` leaves them out)."""
    listop = prim.GetMetadata("apiSchemas")
    return [str(x) for x in listop.GetAddedOrExplicitItems()] if listop else []


def verify(filepath, expected_instances=None):
    """Structural checks on a written assembly: {check: (ok, detail)}."""
    from pxr import Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.Open(filepath)
    out = {}
    root = stage.GetDefaultPrim()
    applied = _api_schemas(root) if root else []
    out["root_api"] = (ROOT_API in applied, applied)
    mt = root.GetAttribute(MESH_TYPE).Get() if root else None
    out["mesh_type"] = (mt == "skeletalMesh", mt)
    rel = root.GetRelationship(SKELETON_REL).GetTargets() if root else []
    skel_ok = bool(rel) and stage.GetPrimAtPath(rel[0]).IsA(UsdSkel.Skeleton) and \
        str(rel[0]).startswith(str(root.GetPath()))
    out["skeleton_rel"] = (skel_ok, [str(r) for r in rel])
    skel = UsdSkel.Skeleton(stage.GetPrimAtPath(rel[0])) if rel else None
    joints = set(map(str, skel.GetJointNamesAttr().Get() or [])) if skel else set()
    full = list(skel.GetJointsAttr().Get()) if skel else []
    out["joint_names"] = (len(joints) == len(full) > 0, (len(joints), len(full)))
    base = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
            and UsdSkel.BindingAPI(p).GetSkeletonRel().GetTargets() == list(rel)]
    base_skinned = bool(base) and all(UsdSkel.BindingAPI(p).GetJointIndicesPrimvar().IsDefined()
                                      for p in base)
    out["base_mesh_skinned"] = (base_skinned, [str(p.GetPath()) for p in base])
    inst = stage.GetPrimAtPath(root.GetPath().AppendChild(INSTANCER))
    pi = UsdGeom.PointInstancer(inst)
    protos = pi.GetPrototypesRel().GetTargets()
    idx = list(pi.GetProtoIndicesAttr().Get() or [])
    n = len(idx)
    lens = (len(pi.GetPositionsAttr().Get() or []), len(pi.GetOrientationsAttr().Get() or []),
            len(pi.GetScalesAttr().Get() or []))
    out["instancer_arrays"] = (all(x == n for x in lens) and n > 0, (n, *lens))
    out["proto_indices"] = (all(0 <= i < len(protos) for i in idx), len(protos))
    parts_ok = all(stage.GetPrimAtPath(p).IsA(UsdSkel.Root)
                   and any(c.IsA(UsdSkel.Skeleton) for c in stage.GetPrimAtPath(p).GetChildren())
                   and any(c.IsA(UsdGeom.Mesh) for c in stage.GetPrimAtPath(p).GetChildren())
                   for p in protos)
    out["prototypes_skelroot"] = (parts_ok, [str(p) for p in protos])
    out["binding_api"] = (BINDING_API in _api_schemas(inst), _api_schemas(inst))
    pv = UsdGeom.PrimvarsAPI(inst).GetPrimvar(BIND_JOINTS[len("primvars:"):])
    bj = list(pv.Get() or []) if pv else []
    out["bind_joints"] = (pv.GetElementSize() == 1 and len(bj) == n
                          and all(str(j) in joints for j in bj),
                          (pv.GetElementSize() if pv else None, len(bj),
                           sum(1 for j in bj if str(j) not in joints)))
    if expected_instances is not None:
        out["instance_count"] = (n == expected_instances, (n, expected_instances))
    return out
