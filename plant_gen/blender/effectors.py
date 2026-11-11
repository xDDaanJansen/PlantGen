"""Effector objects: Blender Empties that bias growth, converted to the plain
`EffectorSpec` data `growth/effectors.py` reads.

The same split as `blender/environment.py`: everything that knows what a
Blender object is lives here, and the growth core is handed numbers. Nothing in
`growth/` ever learns that an effector is an Empty.

**Plant-local space, not world.** This is the Stage 3 coordinate bug, and the
shelved version of this module had it: it read `obj.matrix_world.translation`
straight into the spec. `bridge.regrow` writes the grown skeleton into the
plant object's own mesh, so growth works in object-local coordinates -- an
effector in world space acts somewhere else entirely, and the symptom is not an
error but growth calmly ignoring a magnet sitting visibly next to it.

**Scoped by a collection, not by scanning `bpy.data.objects`.** The shelved
version collected every effector in the file, so every plant in a scene
responded to every effector in it. A collection pointer per plant is both the
scoping mechanism and the on/off switch, and it matches how obstacles are
already chosen.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, PointerProperty
from bpy.types import PropertyGroup
from mathutils import Vector

from ..growth.effectors import EffectorField, EffectorKind, EffectorSpec

_MARKER = "plantgen_effector"

_KIND_ITEMS = [
    (EffectorKind.ATTRACT, "Attract",
     "Pull nearby growth toward this point, fading out at the radius"),
    (EffectorKind.REPEL, "Repel",
     "Push nearby growth away from this point, fading out at the radius"),
    (EffectorKind.FORBID, "Forbid",
     "Keep growth out of this sphere entirely. Tips lean away as they approach "
     "and any that still reach the boundary stop there, the same way they stop "
     "at a wall"),
    (EffectorKind.TARGET, "Target",
     "Aim growth at this point from anywhere, with no falloff. A training goal "
     "for a whole plant rather than a local magnet - Radius is ignored"),
    (EffectorKind.FLATTEN, "Flatten",
     "Grow in the plane this effector faces, for an espalier or a fan. Rotate "
     "the effector to turn the plane. Radius is how far either side of the "
     "plane the flattening reaches"),
    (EffectorKind.ORBIT, "Orbit",
     "Spiral growth around this effector's axis, for a stem twining up a pole. "
     "Rotate the effector to aim the axis. Radius is how far out from the axis "
     "it reaches. Pair it with an Attract on the same spot to keep the stem "
     "against the pole"),
    (EffectorKind.ALIGN, "Align Up",
     "Turn each node's frame to face this effector, so the leaves and flowers "
     "on it swing round - put one where the sun is and the foliage follows. It "
     "turns where branches leave the stem too, so on a bushy plant the whole "
     "shape shifts, not just the foliage. Reads strongest on opposite, "
     "distichous or unilateral species: a spiral keeps its leaves spread all "
     "round the stem and simply rotates the whole set"),
    (EffectorKind.LEVEL, "Level Up",
     "Turn each node's frame to lie flat in this effector's plane. Same reach "
     "as Align Up - foliage and branch angles both. Rotate the effector to "
     "turn the plane"),
]

_DISPLAY_FOR_KIND = {
    # The gizmo has to show what the kind actually reads. A featureless sphere
    # on a Flatten hides that rotating the Empty turns the plane, and a control
    # that silently matters is worse than one that is missing.
    EffectorKind.ATTRACT: 'SPHERE',
    EffectorKind.REPEL: 'SPHERE',
    EffectorKind.FORBID: 'SPHERE',
    EffectorKind.TARGET: 'PLAIN_AXES',   # radius is ignored, so draw no sphere
    EffectorKind.FLATTEN: 'CIRCLE',      # a disc lying in the plane it flattens to
    EffectorKind.ORBIT: 'SINGLE_ARROW',  # the axis growth spirals around
    EffectorKind.ALIGN: 'SPHERE',
    EffectorKind.LEVEL: 'CIRCLE',
}

_AXIS_KINDS = (EffectorKind.FLATTEN, EffectorKind.ORBIT, EffectorKind.LEVEL)
"""Kinds whose result depends on the Empty's *rotation* as well as its
position. Exposed so the panel can say so on the row rather than leaving it to
be discovered."""


def uses_axis(kind: str) -> bool:
    return kind in _AXIS_KINDS


def uses_radius(kind: str) -> bool:
    return kind != EffectorKind.TARGET


def _on_radius_changed(self, context):
    # An Empty's display size for a SPHERE type *is* its drawn radius in
    # Blender units, so this makes the visible sphere the influence zone
    # itself rather than a proportional stand-in for it.
    self.id_data.empty_display_size = self.radius


def _on_kind_changed(self, context):
    self.id_data.empty_display_type = _DISPLAY_FOR_KIND.get(self.kind, 'SPHERE')


class PLANTGEN_PG_effector(PropertyGroup):
    kind: EnumProperty(
        name="Kind", items=_KIND_ITEMS, default=EffectorKind.ATTRACT,
        update=_on_kind_changed,
        description="How this effector biases growth")
    radius: FloatProperty(
        name="Radius", default=0.3, min=0.0, unit='LENGTH', update=_on_radius_changed,
        description="Sphere of influence. The pull fades to nothing at this distance. "
        "Ignored by Target, which reaches everywhere")
    strength: FloatProperty(
        name="Strength", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
        description="How far a growing tip turns toward or away from this effector each "
        "step. 1 aims it straight at the effector every step, which is as strong as this "
        "can get - there is nothing above it")


def is_effector(obj) -> bool:
    return bool(obj) and obj.type == 'EMPTY' and obj.get(_MARKER) is not None


def create_effector(context, collection=None) -> bpy.types.Object:
    """A new effector Empty at the 3D cursor, linked into `collection` if one is
    given so it is already scoped to the plant that will read it."""
    obj = bpy.data.objects.new("PlantGen_Effector", None)
    obj[_MARKER] = True
    obj.empty_display_type = _DISPLAY_FOR_KIND[obj.plantgen_effector.kind]
    target = collection if collection is not None else context.collection
    target.objects.link(obj)
    obj.location = context.scene.cursor.location
    # Property defaults populate silently -- an `update=` callback does not fire
    # on instantiation -- so sync the drawn size explicitly here too.
    obj.empty_display_size = obj.plantgen_effector.radius

    for other in context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


def _uniform_scale(matrix) -> float:
    """One number for a matrix's scale, used to carry a radius between spaces.

    A radius is a sphere and a sphere has one number, so a non-uniformly scaled
    plant or effector can only be approximated -- the mean is the honest
    approximation, and `blender/environment.py` already documents the same
    limitation for the obstacle lattice. Both are strange things to do to a
    plant.
    """
    scale = matrix.to_scale()
    return float(abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3.0


def collect(plant_obj, collection) -> EffectorField:
    """Every effector in `collection`, in the plant's own local space."""
    if collection is None:
        return EffectorField()

    to_plant = plant_obj.matrix_world.inverted()
    plant_scale = _uniform_scale(to_plant)

    specs = []
    for obj in collection.all_objects:
        if not is_effector(obj):
            continue
        e = obj.plantgen_effector
        if e.strength <= 0.0:
            continue
        local = to_plant @ obj.matrix_world.translation
        # The Empty's own object scale counts too: scaling the Empty is the
        # obvious way to resize the sphere you can see, and a radius that
        # ignored it would disagree with the viewport.
        radius = e.radius * _uniform_scale(obj.matrix_world) * plant_scale
        # The Empty's own +Z, carried into plant space. A direction, so the
        # rotation part of both matrices only -- translating a plant must not
        # tip the plane a Flatten flattens to.
        axis = to_plant.to_3x3() @ (obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
        axis = np.array([axis.x, axis.y, axis.z], dtype=float)
        norm = float(np.linalg.norm(axis))
        specs.append(EffectorSpec(
            kind=e.kind,
            position=np.array([local.x, local.y, local.z], dtype=float),
            radius=max(radius, 0.0),
            strength=e.strength,
            axis=axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0]),
        ))
    return EffectorField(specs=tuple(specs))


_classes = (PLANTGEN_PG_effector,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.plantgen_effector = PointerProperty(type=PLANTGEN_PG_effector)


def unregister():
    del bpy.types.Object.plantgen_effector
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
