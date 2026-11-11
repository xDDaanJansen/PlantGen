"""Leaf material: the master leaf shader group + the leaf material that feeds it.

A master shader with texture inputs and controls for subsurface scattering, a
two-sided option, thin film, specular/IOR (1.38-1.425) and
transparency/translucency. This is the leaves one, aimed at Cycles:

  * `PlantGen · Leaf` -- a ShaderNodeTree GROUP holding the whole look: ONE
    Principled BSDF, by default in **Thin Wall** mode (a toggle; Blender's model
    for "papers, leaves, and window sheets": a zero-thickness slab, so the Subsurface weight becomes
    light passing THROUGH the blade -- the leaf's translucency, physically, in
    one BSDF), its own Alpha for the cut-out, and a back-face tint for the
    two-sided option. Version-stamped and rebuilt IN PLACE when the builder
    changes (node input values snapshotted and restored -- a rebuild otherwise
    zeroes them), so every material keeps pointing at it with its settings.
  * a leaf MATERIAL: UV Map -> Image Texture nodes (colour + alpha, normal,
    roughness, translucency) -> the group -> output. A texture is linked only
    while an image is picked; otherwise the group's own value applies, so a
    fresh material is a plain green leaf and each map overrides one setting.
    The artist's "PlantGen Leaf Material" is the MAIN one every new leaf shares
    (adopted -- built into -- if it exists as a plain material).

Settings live on the group NODE's inputs inside the material (the panel draws
them straight off the sockets -- one copy of the truth); the picked images live
on `material.plantgen_leaf` (a PointerProperty per map), whose update
callbacks re-sync the nodes.
"""

import bpy

GROUP_NAME = "PlantGen · Leaf"
_GROUP_VERSION = 4
_VERSION_KEY = "plantgen_leaf_shader_version"
MATERIAL_TAG = "plantgen_leaf_material"
MATERIAL_NAME = "PlantGen Leaf Material"

# Node names inside a leaf material (found by name, never by position).
_N_GROUP = "PGS Leaf"
_N_OUT = "PGS Output"
_N_UV = "PGS UV"
_N_NORMAL_MAP = "PGS Normal Map"

# The texture slots: (pg property, node name, colour space, group input fed,
# extra (output, input) links when the map is present).
TEXTURE_SLOTS = (
    ("base_color_image", "PGS Base Color", "sRGB", "Base Color", (("Alpha", "Alpha"),)),
    ("normal_image", "PGS Normal", "Non-Color", None, ()),          # via Normal Map
    ("roughness_image", "PGS Roughness", "Non-Color", "Roughness", ()),
    ("translucency_image", "PGS Translucency", "Non-Color", "Translucency", ()),
)

# Group interface: (name, socket type, default, min, max, description).
_INPUTS = (
    ("Base Color", "NodeSocketColor", (0.10, 0.25, 0.04, 1.0), None, None,
     "Leaf colour, reflected and transmitted (the Base Color map replaces it)"),
    ("Alpha", "NodeSocketFloat", 1.0, 0.0, 1.0,
     "Opacity; the Base Color map's alpha drives it for cut-out leaves"),
    ("Two-Sided", "NodeSocketFloat", 1.0, 0.0, 1.0,
     "How strongly the back of the leaf takes the Back Tint"),
    ("Back Tint", "NodeSocketColor", (0.85, 1.0, 0.6, 1.0), None, None,
     "Multiplied onto the colour on the back face (undersides are usually paler "
     "and yellower)"),
    ("Roughness", "NodeSocketFloat", 0.45, 0.0, 1.0, "Specular roughness"),
    ("IOR", "NodeSocketFloat", 1.40, 1.0, 2.0,
     "Index of refraction of the leaf surface (real leaves: 1.38 - 1.425)"),
    ("Normal", "NodeSocketVector", (0.0, 0.0, 0.0), None, None,
     "Shading normal (the Normal map feeds it)"),
    ("Thin Wall", "NodeSocketBool", True, None, None,
     "Treat the blade as a zero-thickness sheet (Blender's model for leaves and "
     "paper): Translucency is light passing THROUGH it. Off: a solid surface, and "
     "Translucency becomes ordinary subsurface scattering (waxy, little light "
     "through)"),
    ("Translucency", "NodeSocketFloat", 0.4, 0.0, 1.0,
     "How much light passes through the blade (thin-wall subsurface: the back-lit "
     "glow; with Thin Wall off, the subsurface scattering weight). The "
     "Translucency map (greyscale) replaces it -- veins block light"),
    ("Subsurface Scale", "NodeSocketFloat", 0.002, 0.0, 0.1,
     "How far light scatters inside the leaf, in metres -- about the blade's "
     "thickness. Only used with Thin Wall off (thin-wall mode ignores it)"),
    ("Subsurface Radius", "NodeSocketVector", (0.4, 1.0, 0.2), 0.0, 10.0,
     "Scatter distance per colour (R, G, B), times Subsurface Scale: green goes "
     "furthest in a leaf (chlorophyll absorbs red and blue). Thin Wall off only"),
    ("Thin Film Thickness", "NodeSocketFloat", 0.0, 0.0, 100000.0,
     "Thin-film interference layer thickness in nanometres (0 = off): the "
     "iridescent sheen of a waxy cuticle"),
    ("Thin Film IOR", "NodeSocketFloat", 1.33, 1.0, 3.0,
     "Index of refraction of the thin film"),
)
SETTINGS = tuple(name for name, *_ in _INPUTS if name != "Normal")
# Settings that only matter with Thin Wall off (thin-wall mode ignores them).
SOLID_ONLY = ("Subsurface Scale", "Subsurface Radius")

# Shader presets: starting looks for common leaf conditions -- values chosen by eye
# (no measured source), meant to be tuned. Each sets every shading setting except
# Alpha / Two-Sided / Normal, so applying one gives a whole, consistent look.
_C = (0.85, 1.0, 0.6, 1.0)                        # the default back tint
PRESETS = (
    ("FRESH", "Fresh", "A healthy, mid-green leaf (the default look)", {
        "Base Color": (0.10, 0.25, 0.04, 1.0), "Back Tint": _C, "Roughness": 0.45,
        "IOR": 1.40, "Thin Wall": True, "Translucency": 0.4,
        "Thin Film Thickness": 0.0}),
    ("NEW_GROWTH", "New Growth", "Young, tender leaves: light yellow-green, thin and "
     "very translucent, soft", {
        "Base Color": (0.25, 0.45, 0.06, 1.0), "Back Tint": (0.95, 1.0, 0.7, 1.0),
        "Roughness": 0.55, "IOR": 1.38, "Thin Wall": True, "Translucency": 0.65,
        "Thin Film Thickness": 0.0}),
    ("GLOSSY", "Glossy", "A waxy cuticle (laurel, ivy, citrus): dark green, sharp "
     "highlights, less light through", {
        "Base Color": (0.05, 0.16, 0.03, 1.0), "Back Tint": (0.8, 1.0, 0.55, 1.0),
        "Roughness": 0.15, "IOR": 1.425, "Thin Wall": True, "Translucency": 0.25,
        "Thin Film Thickness": 0.0}),
    ("MATTE", "Matte", "A hairy, velvety surface (sage, lamb's ear): grey-green, no "
     "highlight", {
        "Base Color": (0.14, 0.22, 0.10, 1.0), "Back Tint": (0.9, 1.0, 0.8, 1.0),
        "Roughness": 0.9, "IOR": 1.38, "Thin Wall": True, "Translucency": 0.2,
        "Thin Film Thickness": 0.0}),
    ("MATURE", "Mature", "An old, tough leaf: dark, leathery, little light through", {
        "Base Color": (0.05, 0.12, 0.03, 1.0), "Back Tint": (0.8, 0.95, 0.6, 1.0),
        "Roughness": 0.5, "IOR": 1.41, "Thin Wall": True, "Translucency": 0.2,
        "Thin Film Thickness": 0.0}),
    ("AUTUMN", "Autumn", "Turning: yellow-orange, still translucent", {
        "Base Color": (0.55, 0.30, 0.03, 1.0), "Back Tint": (1.0, 0.9, 0.6, 1.0),
        "Roughness": 0.55, "IOR": 1.40, "Thin Wall": True, "Translucency": 0.5,
        "Thin Film Thickness": 0.0}),
    ("DRY", "Dry", "Dead and dried: brown, papery, nearly opaque", {
        "Base Color": (0.22, 0.13, 0.05, 1.0), "Back Tint": (0.9, 0.85, 0.75, 1.0),
        "Roughness": 0.8, "IOR": 1.40, "Thin Wall": True, "Translucency": 0.1,
        "Thin Film Thickness": 0.0}),
    ("SUCCULENT", "Succulent", "Thick and fleshy (echeveria, jade): blue-green, waxy, "
     "real subsurface scattering (Thin Wall off)", {
        "Base Color": (0.15, 0.25, 0.15, 1.0), "Back Tint": (0.95, 1.0, 0.9, 1.0),
        "Roughness": 0.35, "IOR": 1.42, "Thin Wall": False, "Translucency": 0.5,
        "Subsurface Scale": 0.006, "Subsurface Radius": (0.4, 1.0, 0.3),
        "Thin Film Thickness": 0.0}),
)


def apply_preset(mat, key: str) -> bool:
    """Set a leaf material's shader settings to preset `key`."""
    node = group_node(mat)
    values = next((v for k, _l, _d, v in PRESETS if k == key), None)
    if node is None or values is None:
        return False
    for name, value in values.items():
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
    return True


def matching_preset(mat):
    """The (key, label) of the preset the material's settings match, or None
    (the dropdown then shows Custom)."""
    node = group_node(mat)
    if node is None:
        return None
    for key, label, _desc, values in PRESETS:
        if all(_same(node.inputs.get(n), v) for n, v in values.items()):
            return key, label
    return None


def _same(sock, value) -> bool:
    if sock is None:
        return False
    cur = sock.default_value
    if isinstance(value, bool):
        return bool(cur) == value
    if hasattr(value, "__len__"):
        return len(cur) == len(value) and all(
            abs(a - b) < 1e-3 for a, b in zip(cur, value, strict=True))
    return abs(cur - value) < 1e-3


# --- the group ----------------------------------------------------------------
def ensure_group() -> bpy.types.ShaderNodeTree:
    """The master leaf shader group, built or rebuilt in place to this version."""
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None and ng.bl_idname != "ShaderNodeTree":
        ng = None
    if ng is None:
        ng = bpy.data.node_groups.new(GROUP_NAME, "ShaderNodeTree")
    if ng.get(_VERSION_KEY) != _GROUP_VERSION:
        # Rebuilding the interface recreates every group node's input sockets AT
        # ZERO (not the interface default) -- Alpha 0 = an invisible leaf, and the
        # artist's settings lost. Snapshot each node's values by socket name and
        # put them back; a socket new in this version takes its default.
        saved = _snapshot_values(ng)
        _build_group(ng)
        _restore_values(ng, saved)
        ng[_VERSION_KEY] = _GROUP_VERSION          # stamped LAST (landmine)
        for mat in bpy.data.materials:            # rebuilt sockets drop links
            if mat.get(MATERIAL_TAG):
                sync_material(mat)
    return ng


def _group_nodes(ng):
    for owner in list(bpy.data.materials) + list(bpy.data.node_groups):
        tree = owner.node_tree if isinstance(owner, bpy.types.Material) else owner
        if tree is None:
            continue
        for node in tree.nodes:
            if node.bl_idname == "ShaderNodeGroup" and node.node_tree == ng:
                yield node


def _snapshot_values(ng) -> list:
    saved = []
    for node in _group_nodes(ng):
        vals = {}
        for sock in node.inputs:
            if hasattr(sock, "default_value"):
                v = sock.default_value
                vals[sock.name] = tuple(v) if hasattr(v, "__len__") else v
        saved.append((node, vals))
    return saved


def _restore_values(ng, saved) -> None:
    defaults = {name: default for name, _kind, default, *_r in _INPUTS}
    for node, vals in saved:
        for sock in node.inputs:
            if not hasattr(sock, "default_value"):
                continue
            value = vals.get(sock.name, defaults.get(sock.name))
            if value is not None:
                try:
                    sock.default_value = value
                except (TypeError, ValueError):
                    pass


def _build_group(ng) -> None:
    ng.nodes.clear()
    ng.interface.clear()
    for name, kind, default, lo, hi, desc in _INPUTS:
        sock = ng.interface.new_socket(name, in_out="INPUT", socket_type=kind)
        sock.description = desc
        if name == "Normal":
            sock.hide_value = True                 # zero when unlinked (see below)
        else:
            sock.default_value = default
        if lo is not None:
            sock.min_value, sock.max_value = lo, hi
        if kind == "NodeSocketFloat" and hi == 1.0 and lo == 0.0:
            sock.subtype = 'FACTOR'
    ng.interface.new_socket("BSDF", in_out="OUTPUT", socket_type="NodeSocketShader")

    n, link = ng.nodes, ng.links.new
    gin = n.new("NodeGroupInput")
    gout = n.new("NodeGroupOutput")
    geo = n.new("ShaderNodeNewGeometry")

    # Two-sided: on back faces the colour is multiplied by Back Tint.
    side = n.new("ShaderNodeMath")
    side.operation = 'MULTIPLY'
    link(geo.outputs["Backfacing"], side.inputs[0])
    link(gin.outputs["Two-Sided"], side.inputs[1])
    tint = n.new("ShaderNodeMix")
    tint.data_type = 'RGBA'
    tint.blend_type = 'MULTIPLY'
    link(side.outputs[0], tint.inputs["Factor"])
    link(gin.outputs["Base Color"], tint.inputs["A"])
    link(gin.outputs["Back Tint"], tint.inputs["B"])

    # Normal: a shader group's vector input can only default to a value, so an
    # unlinked Normal arrives as (0,0,0). Use it only when it has length, else
    # the surface normal.
    nlen = n.new("ShaderNodeVectorMath")
    nlen.operation = 'LENGTH'
    link(gin.outputs["Normal"], nlen.inputs[0])
    has_n = n.new("ShaderNodeMath")
    has_n.operation = 'GREATER_THAN'
    has_n.inputs[1].default_value = 0.5
    link(nlen.outputs["Value"], has_n.inputs[0])
    normal = n.new("ShaderNodeMix")
    normal.data_type = 'VECTOR'
    link(has_n.outputs[0], normal.inputs["Factor"])
    link(geo.outputs["Normal"], normal.inputs["A"])
    link(gin.outputs["Normal"], normal.inputs["B"])

    # One thin-walled Principled: Subsurface = diffuse transmission through the
    # slab (translucency); Alpha cuts it all out.
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    link(tint.outputs["Result"], bsdf.inputs["Base Color"])
    link(normal.outputs["Result"], bsdf.inputs["Normal"])
    for src, dst in (("Alpha", "Alpha"), ("Roughness", "Roughness"), ("IOR", "IOR"),
                     ("Thin Wall", "Thin Wall"),
                     ("Translucency", "Subsurface Weight"),
                     ("Subsurface Scale", "Subsurface Scale"),
                     ("Subsurface Radius", "Subsurface Radius"),
                     ("Thin Film Thickness", "Thin Film Thickness"),
                     ("Thin Film IOR", "Thin Film IOR")):
        link(gin.outputs[src], bsdf.inputs[dst])
    link(bsdf.outputs["BSDF"], gout.inputs["BSDF"])

    for node, x, y in ((gin, -900, 0), (geo, -900, 300), (side, -650, 300),
                       (tint, -450, 250), (nlen, -650, -150), (has_n, -450, -150),
                       (normal, -300, -100), (bsdf, -50, 100), (gout, 300, 100)):
        node.location = (x, y)


# --- the material ---------------------------------------------------------------
def is_leaf_material(mat) -> bool:
    return bool(mat) and bool(mat.get(MATERIAL_TAG))


def main_material() -> bpy.types.Material:
    """The MAIN leaf material every new leaf shares: "PlantGen Leaf Material".
    Created if missing; if it exists as a plain material (the artist made it),
    the leaf setup is built into it -- keeping its name and every user."""
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        return new_material(MATERIAL_NAME)
    if not is_leaf_material(mat):
        build_material(mat)
    return mat


def new_material(name: str = MATERIAL_NAME) -> bpy.types.Material:
    """A fresh leaf material (an independent look)."""
    mat = bpy.data.materials.new(name)
    build_material(mat)
    return mat


def build_material(mat) -> None:
    """Build the leaf setup into `mat` (replacing its node tree): the master
    group with its default (untextured) look, empty texture slots, cut-out alpha
    and both faces rendered."""
    group = ensure_group()
    mat.use_nodes = True
    mat[MATERIAL_TAG] = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.name = _N_OUT
    out.target = 'ALL'
    node = nt.nodes.new("ShaderNodeGroup")
    node.name = _N_GROUP
    node.node_tree = group
    node.width = 220
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.name = _N_UV
    uv.uv_map = "UVMap"
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.name = _N_NORMAL_MAP
    for i, (_prop, tex_name, _cs, _dst, _extra) in enumerate(TEXTURE_SLOTS):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = tex_name
        tex.label = tex_name[4:]
        tex.location = (-650, 300 - 280 * i)
    uv.location = (-900, 0)
    nmap.location = (-350, -280)
    node.location = (-100, 0)
    out.location = (220, 0)
    # A cut-out leaf seen from both sides. Thin wall wants a zero-thickness
    # SLAB in EEVEE (manual) -- harmless for Cycles, the target.
    mat.surface_render_method = 'DITHERED'
    mat.use_backface_culling = False
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    if hasattr(mat, "thickness_mode"):
        mat.thickness_mode = 'SLAB'
    thick = out.inputs.get("Thickness")
    if thick is not None:
        thick.default_value = 0.0
    mat.diffuse_color = (0.10, 0.25, 0.04, 1.0)   # Solid-view colour
    sync_material(mat)


def group_node(mat):
    """The master-group node inside a leaf material (None if missing)."""
    if mat is None or mat.node_tree is None:
        return None
    return mat.node_tree.nodes.get(_N_GROUP)


def normal_map_node(mat):
    return mat.node_tree.nodes.get(_N_NORMAL_MAP) if mat and mat.node_tree else None


def sync_material(mat) -> None:
    """(Re)wire a leaf material: every texture node shows its picked image and is
    linked into the group only while it has one. Idempotent."""
    nt = mat.node_tree
    node = nt.nodes.get(_N_GROUP)
    out = nt.nodes.get(_N_OUT)
    uv = nt.nodes.get(_N_UV)
    nmap = nt.nodes.get(_N_NORMAL_MAP)
    if node is None or out is None:
        return
    if node.node_tree is None:
        node.node_tree = ensure_group()
    links = nt.links
    _relink(links, node.outputs["BSDF"], out.inputs["Surface"])
    pg = mat.plantgen_leaf
    for prop, tex_name, colorspace, dst, extra in TEXTURE_SLOTS:
        tex = nt.nodes.get(tex_name)
        if tex is None:
            continue
        image = getattr(pg, prop)
        tex.image = image
        targets = [(tex.outputs["Color"], dst)] if dst else []
        targets += [(tex.outputs[o], i) for o, i in extra]
        if image is not None:
            if image.colorspace_settings.name != colorspace:
                image.colorspace_settings.name = colorspace
            if uv is not None:
                _relink(links, uv.outputs["UV"], tex.inputs["Vector"])
            for sock, name in targets:
                if name in node.inputs:
                    _relink(links, sock, node.inputs[name])
        else:
            for _sock, name in targets:
                if name in node.inputs:
                    _unlink(links, node.inputs[name])
    # The normal map goes through the Normal Map node.
    tex = nt.nodes.get("PGS Normal")
    if nmap is not None and tex is not None:
        if pg.normal_image is not None:
            _relink(links, tex.outputs["Color"], nmap.inputs["Color"])
            _relink(links, nmap.outputs["Normal"], node.inputs["Normal"])
        else:
            _unlink(links, node.inputs["Normal"])
    # The active image node is what Solid > Texture colouring shows.
    base = nt.nodes.get("PGS Base Color")
    if base is not None:
        nt.nodes.active = base


def _relink(links, from_sock, to_sock) -> None:
    for lk in list(to_sock.links):
        if lk.from_socket != from_sock:
            links.remove(lk)
    if not to_sock.is_linked:
        links.new(from_sock, to_sock)


def _unlink(links, to_sock) -> None:
    for lk in list(to_sock.links):
        links.remove(lk)


def _image_changed(self, context):
    mat = self.id_data
    if is_leaf_material(mat):
        sync_material(mat)


class PLANTGEN_PG_leaf_material(bpy.types.PropertyGroup):
    """The images a leaf material's texture nodes show (`material.plantgen_leaf`).
    Picking one links its map into the leaf shader; clearing it falls back to
    the shader's own setting."""
    base_color_image: bpy.props.PointerProperty(
        name="Base Color", type=bpy.types.Image, update=_image_changed,
        description="Colour map; its alpha cuts the leaf out")
    normal_image: bpy.props.PointerProperty(
        name="Normal", type=bpy.types.Image, update=_image_changed,
        description="Tangent-space normal map (veins, surface relief)")
    roughness_image: bpy.props.PointerProperty(
        name="Roughness", type=bpy.types.Image, update=_image_changed,
        description="Roughness map")
    translucency_image: bpy.props.PointerProperty(
        name="Translucency", type=bpy.types.Image, update=_image_changed,
        description="Greyscale translucency map: how much light each part of the "
                    "blade lets through (bright = glows when back-lit, dark veins)")


# --- the UV checker override -----------------------------------------------------
def set_uv_checker(obj, on: bool) -> None:
    """Show `obj` with the plant's shared UV checker instead of its material, via
    an OBJECT-linked slot -- the mesh's own (leaf) material is never touched, so
    turning the checker off restores it exactly."""
    from . import mesh as gn_mesh
    if not obj.material_slots:
        obj.data.materials.append(None)
    slot = obj.material_slots[0]
    if on:
        slot.link = 'OBJECT'
        slot.material = gn_mesh.uv_checker_material()
    else:
        if slot.link == 'OBJECT':
            slot.material = None
        slot.link = 'DATA'


def data_material(obj):
    """The leaf's own material (slot 0 of the mesh), even while the checker
    overrides what the object shows."""
    mats = obj.data.materials if obj and obj.data else ()
    return mats[0] if len(mats) else None


def assign(obj, mat) -> None:
    """Make `mat` the leaf's own material (mesh slot 0)."""
    mats = obj.data.materials
    if len(mats):
        mats[0] = mat
    else:
        mats.append(mat)


_classes = (PLANTGEN_PG_leaf_material,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.plantgen_leaf = bpy.props.PointerProperty(
        type=PLANTGEN_PG_leaf_material)


def unregister():
    del bpy.types.Material.plantgen_leaf
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
