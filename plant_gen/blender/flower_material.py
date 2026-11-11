"""FlowerGen material: one material for the whole flower, and a UV checker.

The flower is unwrapped as one texture laid out like the flower flattened
(`flower._uv_layout`), so one material covers every part. "New" builds the
master LEAF material (`plant_gen`'s `leaf_material`: one thin-wall
Principled for translucency, a back tint, cut-out alpha, texture slots on
`UVMap`) -- a petal is a thin, translucent blade just as a leaf is. That
module is optional here: without the sim add-on (or its leaf tool), New makes
a plain material and FlowerGen works as before.

The UV checker overrides the material with a shared checker to judge the
unwrap.
"""

import bpy

CHECKER_NAME = "PlantGen Flower UV Checker"
_CHECKER_IMAGE = "PlantGen Flower UV Grid"
MATERIAL_NAME = "PlantGen Flower Material"
# A new flower material's starting look (the texture is what really colours it).
_LOOK = {"Base Color": (0.42, 0.10, 0.38, 1.0), "Back Tint": (1.0, 0.85, 1.0, 1.0),
         "Roughness": 0.5, "Translucency": 0.55}


def leaf_module():
    """`plant_gen`'s leaf material module, when it is installed and its
    material settings are registered -- else None (a plain material then)."""
    from . import leaf_material
    if not hasattr(bpy.types.Material, "plantgen_leaf"):
        return None
    return leaf_material


def new_material() -> bpy.types.Material:
    """A fresh flower material: the leaf material when available, else plain."""
    lm = leaf_module()
    if lm is None:
        mat = bpy.data.materials.new(MATERIAL_NAME)
        mat.diffuse_color = _LOOK["Base Color"]
        return mat
    mat = lm.new_material(MATERIAL_NAME)
    node = lm.group_node(mat)
    for socket, value in _LOOK.items():
        if node is not None and socket in node.inputs:
            node.inputs[socket].default_value = value
    mat.diffuse_color = _LOOK["Base Color"]
    return mat


def checker_material() -> bpy.types.Material:
    """The shared checker (Blender's generated colour grid on `UVMap`)."""
    mat = bpy.data.materials.get(CHECKER_NAME)
    if mat is not None and mat.node_tree is not None:
        return mat
    image = bpy.data.images.get(_CHECKER_IMAGE)
    if image is None:
        image = bpy.data.images.new(_CHECKER_IMAGE, 1024, 1024)
        image.generated_type = 'COLOR_GRID'
    mat = mat or bpy.data.materials.new(CHECKER_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    tex.image = image
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    nt.nodes.active = tex                 # what Solid > Texture colouring shows
    uv.location, tex.location, bsdf.location, out.location = (
        (-700, 0), (-450, 0), (-150, 0), (200, 0))
    return mat


def set_texture_shading(context, on: bool) -> None:
    """The checker is an image, which Solid shading only shows with Texture
    colouring: switch Solid 3D views to it (and back to Material when off)."""
    screen = getattr(context, "screen", None) if context is not None else None
    for area in (screen.areas if screen is not None else ()):
        if area.type != 'VIEW_3D':
            continue
        shading = area.spaces.active.shading
        if shading.type != 'SOLID':
            continue
        if on:
            shading.color_type = 'TEXTURE'
        elif shading.color_type == 'TEXTURE':
            shading.color_type = 'MATERIAL'


def _flower(context):
    from . import flower
    obj = context.active_object
    return obj if flower.is_flower(obj) else None


class PLANTGEN_OT_flower_material_new(bpy.types.Operator):
    bl_idname = "plantgen.flower_material_new"
    bl_label = "New Flower Material"
    bl_description = ("Give the flower a new material of its own (the master leaf "
                      "shader -- translucent, two-sided -- in a petal colour)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _flower(context) is not None

    def execute(self, context):
        _flower(context).plantgen_flower.material = new_material()
        return {"FINISHED"}


def draw_materials(layout, obj) -> None:
    """The Material panel: the material picker (New builds a leaf-shader
    one), the UV checker, then the shared leaf-shader editor (its preset and
    Textures & Settings, `ui_materials.draw_editor`)."""
    s = obj.plantgen_flower
    row = layout.row(align=True)
    row.prop(s, "material", text="")
    row.operator("plantgen.flower_material_new", text="", icon="ADD")
    layout.prop(s, "uv_checker", icon="TEXTURE", toggle=True)
    if s.material is not None:
        from . import ui_materials
        ui_materials.draw_editor(layout, s.material)


_classes = (PLANTGEN_OT_flower_material_new,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
