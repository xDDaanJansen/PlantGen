"""Stem material: the master stem shader group + the stem materials that feed it.

The bark/stem master shader (texture inputs + subsurface, specular/IOR). The
leaf one's twin (`leaf_material`):

  * `PlantGen · Stem` -- a ShaderNodeTree GROUP holding the look: ONE solid
    Principled BSDF (a green stem lets a little light in -- Subsurface; hairs or
    a waxy bloom catch the light at grazing angles -- Sheen) plus a Foot colour
    that fades up the plant's lowest part (a stem reddening or browning at its
    base), read off the stem mesh's `height_fraction` (0 at the plant's foot, 1
    at its top). Version-stamped and rebuilt in place, values kept.
  * a stem MATERIAL: UV Map -> Image Texture nodes (colour, normal, roughness)
    -> the group -> output. A map is linked only while an image is picked, so a
    fresh material is a plain green stem; the stems' UVs are one absolute UV
    Size (metres per tile) on every tube, so a tiling bark texture keeps its
    scale from trunk to twig.

Each branching level picks its material (`PLANTGEN_PG_branch_level.material`;
none = its parent level's), set per level in the node graph
(`mesh._build_stem_tube`). Settings live on the group NODE's inputs; the picked
images on `material.plantgen_stem`.
"""

import bpy

from . import leaf_material as _lm

GROUP_NAME = "PlantGen · Stem"
_GROUP_VERSION = 1
_VERSION_KEY = "plantgen_stem_shader_version"
MATERIAL_TAG = "plantgen_stem_material"
STEM_NAME = "PlantGen Stem Material"
BRANCH_NAME = "PlantGen Branch Material"
MAX_LEVELS = 8
"""Branching levels with a material socket of their own in the node graph; a
deeper level shares the last one's."""

_N_GROUP = "PGS Stem"
_N_OUT = "PGS Output"
_N_UV = "PGS UV"
_N_NORMAL_MAP = "PGS Normal Map"

# (pg property, node name, colour space, group input fed).
TEXTURE_SLOTS = (
    ("base_color_image", "PGS Base Color", "sRGB", "Base Color"),
    ("normal_image", "PGS Normal", "Non-Color", None),           # via Normal Map
    ("roughness_image", "PGS Roughness", "Non-Color", "Roughness"),
)

# Group interface: (name, socket type, default, min, max, description).
_INPUTS = (
    ("Base Color", "NodeSocketColor", (0.12, 0.26, 0.05, 1.0), None, None,
     "Stem colour (the Base Color map replaces it)"),
    ("Foot Color", "NodeSocketColor", (0.30, 0.06, 0.04, 1.0), None, None,
     "Colour the stems take at the plant's foot, fading into the Base Color "
     "(a red or brown base)"),
    ("Foot Height", "NodeSocketFloat", 0.0, 0.0, 1.0,
     "How far up the plant the Foot Color reaches, as a fraction of its height "
     "(0: no foot colour)"),
    ("Roughness", "NodeSocketFloat", 0.45, 0.0, 1.0, "Specular roughness"),
    ("IOR", "NodeSocketFloat", 1.40, 1.0, 2.0,
     "Index of refraction of the stem's cuticle (plant surfaces: 1.38 - 1.425)"),
    ("Normal", "NodeSocketVector", (0.0, 0.0, 0.0), None, None,
     "Shading normal (the Normal map feeds it)"),
    ("Subsurface", "NodeSocketFloat", 0.15, 0.0, 1.0,
     "Light scattered inside the stem: a soft, fleshy glow for a green stem, "
     "none for bark or straw"),
    ("Subsurface Scale", "NodeSocketFloat", 0.003, 0.0, 0.1,
     "How far light scatters inside the stem, in metres"),
    ("Subsurface Radius", "NodeSocketVector", (0.4, 1.0, 0.2), 0.0, 10.0,
     "Scatter distance per colour (R, G, B), times Subsurface Scale: green goes "
     "furthest in green tissue"),
    ("Sheen", "NodeSocketFloat", 0.0, 0.0, 1.0,
     "A soft rim of light at grazing angles: fine hairs, or a waxy bloom"),
    ("Sheen Tint", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), None, None,
     "Colour of the sheen (bluish-white for a glaucous bloom)"),
)
SETTINGS = tuple(name for name, *_ in _INPUTS if name != "Normal")

# Presets: starting looks for common stems -- values chosen by eye (no measured
# source), meant to be tuned. Each sets every setting except Normal.
_W = (1.0, 1.0, 1.0, 1.0)
_RED = (0.30, 0.06, 0.04, 1.0)
_SSS = (0.4, 1.0, 0.2)


def _p(color, rough, sss, *, foot=_RED, foot_h=0.0, sheen=0.0, tint=_W, ior=1.40,
       scale=0.003, radius=_SSS):
    return {"Base Color": color, "Foot Color": foot, "Foot Height": foot_h,
            "Roughness": rough, "IOR": ior, "Subsurface": sss,
            "Subsurface Scale": scale, "Subsurface Radius": radius,
            "Sheen": sheen, "Sheen Tint": tint}


PRESETS = (
    ("GREEN", "Green Stem", "A healthy herbaceous stem: mid-green, a little light "
     "scattered in (the default look)", _p((0.12, 0.26, 0.05, 1.0), 0.45, 0.15)),
    ("YOUNG", "Young Shoot", "New, tender growth: light yellow-green, soft and "
     "fleshy", _p((0.25, 0.45, 0.08, 1.0), 0.5, 0.3)),
    ("RED_FOOT", "Red Foot", "Green, reddening toward the base (many perennials, "
     "docks, knotweeds)", _p((0.12, 0.26, 0.05, 1.0), 0.45, 0.15, foot_h=0.3)),
    ("PURPLE", "Purple", "A dark purple-flushed stem (verbena, sedum, some salvias)",
     _p((0.14, 0.04, 0.10, 1.0), 0.4, 0.1, radius=(1.0, 0.3, 0.6))),
    ("GLAUCOUS", "Glaucous", "Blue-green under a waxy bloom (euphorbia, sedum, blue "
     "grasses)", _p((0.13, 0.22, 0.17, 1.0), 0.6, 0.1, sheen=0.5,
                    tint=(0.8, 0.9, 1.0, 1.0), ior=1.42)),
    ("HAIRY", "Hairy", "Covered in fine hairs (borage, lamb's ear, poppy): grey-green, "
     "a soft silvery rim", _p((0.15, 0.22, 0.10, 1.0), 0.8, 0.1, sheen=0.7)),
    ("STRAW", "Straw", "Dry, dead grass stems: pale gold, papery, no light through",
     _p((0.45, 0.34, 0.14, 1.0), 0.6, 0.0, foot=(0.30, 0.22, 0.10, 1.0))),
    ("WOODY", "Woody", "Brown, rough bark on older wood (sub-shrubs: lavender, sage)",
     _p((0.12, 0.08, 0.05, 1.0), 0.85, 0.0, foot=(0.08, 0.05, 0.03, 1.0))),
)
DEFAULT_PRESET = "GREEN"


def apply_preset(mat, key: str) -> bool:
    """Set a stem material's shader settings to preset `key`."""
    node = group_node(mat)
    values = next((v for k, _l, _d, v in PRESETS if k == key), None)
    if node is None or values is None:
        return False
    for name, value in values.items():
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
    mat.diffuse_color = values["Base Color"]          # the Solid-view colour
    return True


def matching_preset(mat):
    """The (key, label) of the preset the material matches, or None (Custom)."""
    node = group_node(mat)
    if node is None:
        return None
    for key, label, _desc, values in PRESETS:
        if all(_lm._same(node.inputs.get(n), v) for n, v in values.items()):
            return key, label
    return None


# --- the group ----------------------------------------------------------------
def ensure_group() -> bpy.types.ShaderNodeTree:
    """The master stem shader group, built or rebuilt in place to this version."""
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None and ng.bl_idname != "ShaderNodeTree":
        ng = None
    if ng is None:
        ng = bpy.data.node_groups.new(GROUP_NAME, "ShaderNodeTree")
    if ng.get(_VERSION_KEY) != _GROUP_VERSION:
        # A rebuilt interface zeroes every group node's inputs (landmine):
        # snapshot by socket name, put back; a new socket takes its default.
        saved = _lm._snapshot_values(ng)
        _build_group(ng)
        defaults = {name: default for name, _kind, default, *_r in _INPUTS}
        for node, vals in saved:
            for sock in node.inputs:
                value = vals.get(sock.name, defaults.get(sock.name))
                if value is not None and hasattr(sock, "default_value"):
                    try:
                        sock.default_value = value
                    except (TypeError, ValueError):
                        pass
        ng[_VERSION_KEY] = _GROUP_VERSION          # stamped LAST (landmine)
        for mat in bpy.data.materials:
            if mat.get(MATERIAL_TAG):
                sync_material(mat)
    return ng


def _build_group(ng) -> None:
    ng.nodes.clear()
    ng.interface.clear()
    for name, kind, default, lo, hi, desc in _INPUTS:
        sock = ng.interface.new_socket(name, in_out="INPUT", socket_type=kind)
        sock.description = desc
        if name == "Normal":
            sock.hide_value = True
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

    # Foot colour: 1 at the plant's foot, smoothly 0 by Foot Height (of the
    # plant's height, `height_fraction`); 0 everywhere while Foot Height is 0.
    height = n.new("ShaderNodeAttribute")
    height.attribute_type = 'GEOMETRY'
    height.attribute_name = "height_fraction"
    fade = n.new("ShaderNodeMapRange")
    fade.interpolation_type = 'SMOOTHSTEP'
    fade.clamp = True
    link(height.outputs["Fac"], fade.inputs["Value"])
    fade.inputs["From Min"].default_value = 0.0
    link(gin.outputs["Foot Height"], fade.inputs["From Max"])
    fade.inputs["To Min"].default_value = 1.0
    fade.inputs["To Max"].default_value = 0.0
    on = n.new("ShaderNodeMath")
    on.operation = 'GREATER_THAN'
    on.inputs[1].default_value = 1e-4
    link(gin.outputs["Foot Height"], on.inputs[0])
    weight = n.new("ShaderNodeMath")
    weight.operation = 'MULTIPLY'
    link(fade.outputs["Result"], weight.inputs[0])
    link(on.outputs[0], weight.inputs[1])
    tinted = n.new("ShaderNodeMix")
    tinted.data_type = 'RGBA'
    tinted.blend_type = 'MIX'
    tinted.clamp_factor = True
    link(weight.outputs[0], tinted.inputs["Factor"])
    link(gin.outputs["Base Color"], tinted.inputs["A"])
    link(gin.outputs["Foot Color"], tinted.inputs["B"])

    # Normal: an unlinked group vector arrives as (0,0,0) -- use it only when it
    # has length, else the surface normal (as the leaf group does).
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

    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.subsurface_method = 'RANDOM_WALK'
    link(tinted.outputs["Result"], bsdf.inputs["Base Color"])
    link(normal.outputs["Result"], bsdf.inputs["Normal"])
    for src, dst in (("Roughness", "Roughness"), ("IOR", "IOR"),
                     ("Subsurface", "Subsurface Weight"),
                     ("Subsurface Scale", "Subsurface Scale"),
                     ("Subsurface Radius", "Subsurface Radius"),
                     ("Sheen", "Sheen Weight"), ("Sheen Tint", "Sheen Tint")):
        link(gin.outputs[src], bsdf.inputs[dst])
    link(bsdf.outputs["BSDF"], gout.inputs["BSDF"])

    for node, x, y in ((gin, -1100, 0), (height, -1100, 400), (fade, -850, 400),
                       (on, -850, 150), (weight, -650, 350), (tinted, -450, 200), (geo, -850, -350),
                       (nlen, -650, -350), (has_n, -450, -350), (normal, -300, -250),
                       (bsdf, -50, 100), (gout, 300, 100)):
        node.location = (x, y)


# --- the material ---------------------------------------------------------------
def is_stem_material(mat) -> bool:
    return bool(mat) and bool(mat.get(MATERIAL_TAG))


def new_material(name: str = STEM_NAME, preset: str = DEFAULT_PRESET):
    """A fresh stem material (an independent look) set to `preset`."""
    mat = bpy.data.materials.new(name)
    build_material(mat)
    apply_preset(mat, preset)
    return mat


def build_material(mat) -> None:
    """Build the stem setup into `mat` (replacing its node tree): the master
    group with its default (untextured) look and empty texture slots."""
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
    for i, (_prop, tex_name, _cs, _dst) in enumerate(TEXTURE_SLOTS):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = tex_name
        tex.label = tex_name[4:]
        tex.location = (-650, 300 - 280 * i)
    uv.location = (-900, 0)
    nmap.location = (-350, -280)
    node.location = (-100, 0)
    out.location = (220, 0)
    mat.diffuse_color = _INPUTS[0][2]
    sync_material(mat)


def group_node(mat):
    if mat is None or mat.node_tree is None:
        return None
    return mat.node_tree.nodes.get(_N_GROUP)


def normal_map_node(mat):
    return mat.node_tree.nodes.get(_N_NORMAL_MAP) if mat and mat.node_tree else None


def sync_material(mat) -> None:
    """(Re)wire a stem material: each texture node shows its picked image and is
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
    _lm._relink(links, node.outputs["BSDF"], out.inputs["Surface"])
    pg = mat.plantgen_stem
    for prop, tex_name, colorspace, dst in TEXTURE_SLOTS:
        tex = nt.nodes.get(tex_name)
        if tex is None:
            continue
        image = getattr(pg, prop)
        tex.image = image
        if image is not None:
            if image.colorspace_settings.name != colorspace:
                image.colorspace_settings.name = colorspace
            if uv is not None:
                _lm._relink(links, uv.outputs["UV"], tex.inputs["Vector"])
            if dst:
                _lm._relink(links, tex.outputs["Color"], node.inputs[dst])
        elif dst:
            _lm._unlink(links, node.inputs[dst])
    tex = nt.nodes.get("PGS Normal")
    if nmap is not None and tex is not None:
        if pg.normal_image is not None:
            _lm._relink(links, tex.outputs["Color"], nmap.inputs["Color"])
            _lm._relink(links, nmap.outputs["Normal"], node.inputs["Normal"])
        else:
            _lm._unlink(links, node.inputs["Normal"])
    base = nt.nodes.get("PGS Base Color")
    if base is not None:
        nt.nodes.active = base


def _image_changed(self, context):
    mat = self.id_data
    if is_stem_material(mat):
        sync_material(mat)


class PLANTGEN_PG_stem_material(bpy.types.PropertyGroup):
    """The images a stem material's texture nodes show (`material.plantgen_stem`)."""
    base_color_image: bpy.props.PointerProperty(
        name="Base Color", type=bpy.types.Image, update=_image_changed,
        description="Colour map (bark, stem streaks); tiles at the plant's UV Size")
    normal_image: bpy.props.PointerProperty(
        name="Normal", type=bpy.types.Image, update=_image_changed,
        description="Tangent-space normal map (bark relief, ridges)")
    roughness_image: bpy.props.PointerProperty(
        name="Roughness", type=bpy.types.Image, update=_image_changed,
        description="Roughness map")


# --- per level -----------------------------------------------------------------
def level_materials(s) -> list:
    """Each branching level's material: its own, else its parent level's (up to
    the stems level; None there means no material set)."""
    levels = list(s.branch_levels)
    by_uid = {lv.uid: lv for lv in levels}
    out = []
    for lv in levels:
        seen, cur = set(), lv
        while cur is not None and cur.material is None and cur.uid not in seen:
            seen.add(cur.uid)
            cur = by_uid.get(cur.parent_uid) if cur.parent_uid else None
        out.append(cur.material if cur is not None else None)
    return out


def _active_level(context):
    from . import bridge
    obj = bridge.resolve_plant(context)
    if obj is None:
        return None, None
    s = obj.plantgen
    i = s.branch_levels_active
    return obj, (s.branch_levels[i] if 0 <= i < len(s.branch_levels) else None)


class PLANTGEN_OT_stem_material_new(bpy.types.Operator):
    bl_idname = "plantgen.stem_material_new"
    bl_label = "New Stem Material"
    bl_description = ("Give this level a new stem material of its own (the master "
                      "stem shader, Green Stem preset, untextured)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_level(context)[1] is not None

    def execute(self, context):
        _obj, lv = _active_level(context)
        lv.material = new_material(STEM_NAME if lv.parent_uid == 0 else BRANCH_NAME)
        return {"FINISHED"}


class PLANTGEN_OT_stem_material_preset(bpy.types.Operator):
    bl_idname = "plantgen.stem_material_preset"
    bl_label = "Stem Shader Preset"
    bl_description = ("Set this level's stem material to this preset (every level "
                      "sharing the material changes)")
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.StringProperty(options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        lv = _active_level(context)[1]
        return lv is not None and is_stem_material(lv.material)

    @classmethod
    def description(cls, context, properties):
        for key, _label, desc, _v in PRESETS:
            if key == properties.preset:
                return desc + " (every level sharing the material changes)"
        return cls.bl_description

    def execute(self, context):
        _obj, lv = _active_level(context)
        if not apply_preset(lv.material, self.preset):
            self.report({'ERROR'}, f"Unknown stem shader preset {self.preset!r}")
            return {'CANCELLED'}
        return {"FINISHED"}


class PLANTGEN_MT_stem_material_presets(bpy.types.Menu):
    bl_idname = "PLANTGEN_MT_stem_material_presets"
    bl_label = "Stem Shader Presets"

    def draw(self, context):
        for key, label, _desc, _v in PRESETS:
            op = self.layout.operator("plantgen.stem_material_preset", text=label)
            op.preset = key


def draw_level_material(layout, lv, parent_name: str) -> None:
    """A level's Material box: the picker, then (for a stem material) its
    preset, texture maps and shader settings in a closed-by-default panel."""
    box = layout.box()
    box.template_ID(lv, "material", new="plantgen.stem_material_new")
    mat = lv.material
    if mat is None:
        box.label(text=f"Uses {parent_name}'s material" if parent_name
                  else "No material (click New)", icon="INFO")
        return
    if not is_stem_material(mat):
        box.label(text="Not a PlantGen stem material", icon="INFO")
        return
    node = group_node(mat)
    if node is None:
        box.label(text="Stem shader node missing", icon="ERROR")
        return
    row = box.row(align=True)
    row.label(text="Preset")
    match = matching_preset(mat)
    row.menu("PLANTGEN_MT_stem_material_presets",
             text=match[1] if match else "Custom", icon="PRESET")
    header, body = box.panel("plantgen_stem_material_settings", default_closed=True)
    header.label(text="Textures & Settings")
    if body is None:
        return
    pg = mat.plantgen_stem
    col = body.column(align=True)
    for prop, *_rest in TEXTURE_SLOTS:
        split = col.split(factor=0.32, align=True)
        split.label(text=pg.bl_rna.properties[prop].name)
        split.template_ID(pg, prop, open="image.open")
    if pg.normal_image is not None:
        nmap = normal_map_node(mat)
        if nmap is not None:
            col.prop(nmap.inputs["Strength"], "default_value", text="Normal Strength")
    col = body.column(align=True)
    foot = node.inputs.get("Foot Height")
    for name in SETTINGS:
        sock = node.inputs.get(name)
        if sock is None or sock.is_linked:
            continue
        if name == "Foot Color" and foot is not None and foot.default_value <= 0.0:
            continue                            # no foot: its colour does nothing
        if name in ("Subsurface Scale", "Subsurface Radius") and \
                node.inputs["Subsurface"].default_value <= 0.0:
            continue
        if name == "Sheen Tint" and node.inputs["Sheen"].default_value <= 0.0:
            continue
        col.prop(sock, "default_value", text=name)


_classes = (PLANTGEN_PG_stem_material, PLANTGEN_OT_stem_material_new,
            PLANTGEN_OT_stem_material_preset, PLANTGEN_MT_stem_material_presets)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.plantgen_stem = bpy.props.PointerProperty(
        type=PLANTGEN_PG_stem_material)


def unregister():
    del bpy.types.Material.plantgen_stem
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
