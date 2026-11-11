"""Stage 3's bpy half: real Blender objects become the plain voxel grid
`growth/environment.py` reads.

This is the same split `bridge.py` uses -- everything that knows what a
Blender object is lives here, and the growth core is handed numbers. It is
also the reason obstacles can be *any* mesh: the core never sees geometry, only
a lattice of "solid here / not solid here", so a rock, a wall and a watering
can are all the same thing by the time growth reads them.

**Everything here is in the plant's own local space, not world space.**
`bridge.regrow` writes the grown skeleton straight into the object's mesh, so
the growth core has always worked in object-local coordinates with the plant's
base at the origin. Voxelising obstacles in world space instead put the two in
different frames: a slab really was found and really was 2352 voxels of solid,
and growth sailed through it because the plant was three metres away in the
only coordinate system that mattered. Obstacle bounds and grid columns are
therefore built in plant-local space and transformed out to world only to
actually cast a ray. A non-uniformly scaled plant object skews the lattice
accordingly, which is a real limitation and a strange thing to do to a plant.

**Solid, not shell.** Filling is by ray parity: one ray per grid column,
straight up, with the spans between successive crossings filled in. A shell
would let a tip come to rest inside a hollow object, and would cast a hollow
shadow. The tradeoff is the usual one for parity filling -- it wants closed
meshes, and an open or self-intersecting one can fill inside out along the
columns where a ray grazes an edge. That is acceptable here: the failure is
local to a column, obvious on screen, and fixed by closing the mesh.
"""

from __future__ import annotations

import numpy as np

from ..growth.environment import Environment, Ground, Lattice, Obstacles
from . import effectors as effector_bridge

OBSTACLE_RESOLUTION = 32
"""Voxels along the longest axis of the obstacle lattice. A solver constant
with no UI: it trades avoidance precision against one-off voxelisation cost,
and neither is a property of the plant."""

_MAX_HITS_PER_COLUMN = 32
"""Guard against a degenerate mesh turning one ray into an unbounded loop."""

_SURFACE_EPSILON = 1e-5

GROUND_RESOLUTION = 128
"""Samples along the longer side of a Ground Object's footprint. A solver
constant, like the obstacle lattice's: a 4 m terrain is sampled every ~3 cm,
read bilinearly between."""


def sample_ground(ground_obj, plant_obj) -> Ground | None:
    """The terrain's top surface under the plant, as a plant-local height grid:
    one ray straight down per grid point over the object's footprint (its
    evaluated mesh, so modifiers count). NaN where a ray misses."""
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    to_plant = plant_obj.matrix_world.inverted()
    bounds = _local_bounds(ground_obj, to_plant)
    if bounds is None:
        return None
    lo, hi = bounds
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    if span <= 0.0:
        return None
    cell = span / (GROUND_RESOLUTION - 1)
    nx = max(2, int(np.ceil((hi[0] - lo[0]) / cell)) + 1)
    ny = max(2, int(np.ceil((hi[1] - lo[1]) / cell)) + 1)
    try:
        tree = BVHTree.FromObject(ground_obj, bpy.context.evaluated_depsgraph_get())
    except ValueError:
        # Not in the active scene's depsgraph (the plant regrown from another
        # scene): its own mesh, without modifiers, in the same object space.
        me = ground_obj.data
        tree = BVHTree.FromPolygons([v.co for v in me.vertices],
                                    [tuple(p.vertices) for p in me.polygons])
    # Rays cast in the ground object's own space: plant-local -> world -> its space.
    to_ground = ground_obj.matrix_world.inverted() @ plant_obj.matrix_world
    from_ground = to_plant @ ground_obj.matrix_world
    down = (to_ground.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    top = float(hi[2]) + 1.0
    heights = np.full((nx, ny), np.nan)
    for i in range(nx):
        for j in range(ny):
            start = to_ground @ Vector((float(lo[0]) + i * cell, float(lo[1]) + j * cell, top))
            hit = tree.ray_cast(start, down)
            if hit[0] is not None:
                heights[i, j] = (from_ground @ hit[0]).z
    if np.isnan(heights).all():
        return None
    return Ground(origin=np.array([float(lo[0]), float(lo[1])]), cell=cell, heights=heights)


def _local_bounds(obj, to_plant) -> tuple[np.ndarray, np.ndarray] | None:
    """An object's bounding box in the plant's local space."""
    from mathutils import Vector

    corners = [to_plant @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    if not corners:
        return None
    stacked = np.array([[c.x, c.y, c.z] for c in corners])
    return stacked.min(axis=0), stacked.max(axis=0)


def _column_hits(obj, start, end, to_plant) -> list[float]:
    """Plant-local z of every surface crossing along one upward column.

    Ray casts happen in the *obstacle's* own space -- `Object.ray_cast` takes
    and returns object-local coordinates -- so the column comes in as two world
    points, is transformed into that space to cast, and each hit is transformed
    back out to the plant's space, which is the one the caller thinks in.
    """
    inverse = obj.matrix_world.inverted()
    cursor = inverse @ start
    finish = inverse @ end
    direction = finish - cursor
    remaining = direction.length
    if remaining <= 0.0:
        return []
    direction = direction.normalized()

    hits: list[float] = []
    for _ in range(_MAX_HITS_PER_COLUMN):
        found, location, _normal, _index = obj.ray_cast(cursor, direction, distance=remaining)
        if not found:
            break
        hits.append(float((to_plant @ (obj.matrix_world @ location)).z))
        travelled = (location - cursor).length + _SURFACE_EPSILON
        remaining -= travelled
        if remaining <= 0.0:
            break
        cursor = location + direction * _SURFACE_EPSILON
    return hits


def voxelise(objects, lattice: Lattice, plant_matrix) -> np.ndarray:
    """A 1.0/0.0 solidity array over `lattice`, filled from every object.
    `lattice` is in the plant's local space; `plant_matrix` takes it to world."""
    from mathutils import Vector

    to_plant = plant_matrix.inverted()
    solid = np.zeros(lattice.shape, dtype=float)
    nx, ny, nz = lattice.shape
    origin, size = lattice.origin, lattice.voxel_size
    # Column centres, so a ray never runs exactly along a voxel face.
    xs = origin[0] + (np.arange(nx) + 0.5) * size
    ys = origin[1] + (np.arange(ny) + 0.5) * size
    z0 = float(origin[2]) - size
    z1 = float(origin[2]) + nz * size + size

    for obj in objects:
        bounds = _local_bounds(obj, to_plant)
        if bounds is None:
            continue
        lo, hi = bounds
        # Only the columns this object could possibly occupy. On a scene with
        # several small objects this is the difference between one full grid
        # sweep and a handful of columns each.
        ix = np.nonzero((xs >= lo[0] - size) & (xs <= hi[0] + size))[0]
        iy = np.nonzero((ys >= lo[1] - size) & (ys <= hi[1] + size))[0]
        for i in ix:
            for j in iy:
                start = plant_matrix @ Vector((float(xs[i]), float(ys[j]), z0))
                end = plant_matrix @ Vector((float(xs[i]), float(ys[j]), z1))
                hits = sorted(_column_hits(obj, start, end, to_plant))
                # Parity: inside between the 1st and 2nd crossing, the 3rd and
                # 4th, and so on. An odd count means the ray left through a
                # hole; the trailing hit is dropped rather than filling to the
                # top of the grid.
                for a, b in zip(hits[0::2], hits[1::2], strict=False):
                    lo_k = int(np.ceil((a - origin[2]) / size - 0.5))
                    hi_k = int(np.floor((b - origin[2]) / size - 0.5))
                    if hi_k >= lo_k:
                        solid[i, j, max(lo_k, 0):min(hi_k + 1, nz)] = 1.0
    return solid


def _obstacle_bounds(meshes, to_plant) -> tuple[np.ndarray, np.ndarray] | None:
    """The box the obstacle lattice covers, in the plant's local space: the
    obstacles themselves, plus the plant's own base at the local origin so the
    space between them is represented too.

    Sized from the obstacles rather than from any estimate of how big the plant
    will get. That estimate was the first version and it was badly wrong in the
    direction that matters: a generic perennial's profile predicts 2.7 m of
    reach for a plant that grows to 0.64 m, which at a fixed voxel count made
    each voxel 12 cm -- so a real 12 cm slab came out one voxel thick, never
    sampled above the 0.5 solidity threshold, and stopped nothing at all. The
    obstacle's own size is the scale that has to be resolved here.

    Anything beyond the lattice reads as open air, which is the correct answer
    there rather than a fallback: obstacles are all inside it by construction.
    """
    lo = hi = None
    for mesh_obj in meshes:
        bounds = _local_bounds(mesh_obj, to_plant)
        if bounds is None:
            continue
        lo = bounds[0] if lo is None else np.minimum(lo, bounds[0])
        hi = bounds[1] if hi is None else np.maximum(hi, bounds[1])
    if lo is None:
        return None
    base = np.zeros(3)  # the plant's own base, by definition, in its own space
    return np.minimum(lo, base), np.maximum(hi, base)


def build_environment(obj, profile) -> Environment | None:
    """The Environment for one plant, or None if Stage 3 is doing nothing.

    None rather than an empty Environment on purpose: `development.grow` treats
    None as "no Stage 3 at all" and skips the per-step resolve entirely, so a
    plant that is not responding to anything costs exactly what it did before
    Stage 3 existed.
    """
    s = obj.plantgen
    wants_obstacles = profile.obstacle_clearance > 0.0 and s.obstacle_collection is not None
    # Effectors are scene furniture rather than a species trait, so unlike
    # light and obstacles there is nothing on the profile to enable first --
    # pointing at a collection with something in it is the whole switch.
    effectors = effector_bridge.collect(obj, s.effector_collection)
    # WS3 shade-death is a post-growth reader of the light field, so it needs an
    # Environment to exist even when nothing else does -- a plant shades itself.
    # `_responds_to` still returns False for a bare Environment (no obstacles, no
    # effectors, `reads_light` False), so growth stays byte-identical: the light
    # field is only built later, by `lifecycle`, against the finished plant.
    wants_shade_death = profile.senescence_shade_threshold > 0.0
    # Heliotropism: a post-growth re-aim of organ facing toward a picked object.
    # Needs both a non-zero strength (species) and an object to track (scene). The
    # direction is that object's position in the plant's own local space, base at
    # the origin -- "move it to place the sun", position not rotation. A
    # heliotropism-only Environment is inert during growth (`_responds_to` sees no
    # obstacles/effectors/light), so growth stays byte-identical; only the final
    # `apply_heliotropism` pass reads it.
    # The plant's Sun Object is the leaves' (and a component-less plant's
    # flowers'); each flower component tracks its own.
    # (Imported here: bridge imports this module at load time.)
    from . import bridge

    to_plant = obj.matrix_world.inverted()
    sun_target = None
    if profile.tracks_sun and bridge.plant_sun_used(s):
        sun_target = _toward(to_plant, s.sun_object)
    flower_suns = ()
    owners = bridge.inflorescence_owners(s)
    if profile.tracks_sun and owners:
        # One per inflorescence, None where it tracks nothing -- never the
        # plant's in its place.
        flower_suns = tuple(_toward(to_plant, c.sun_object) if c.flower_heliotropism > 0.0
                            else None for c in owners)
    wants_helio = sun_target is not None or any(t is not None for t in flower_suns)
    # The Ground Object: read by the gravity pass alone, so a ground-only
    # Environment is inert during growth (`_responds_to` ignores it).
    ground = None
    if s.ground_object is not None and s.ground_object.type == 'MESH':
        ground = sample_ground(s.ground_object, obj)
    if not (profile.reads_light or wants_obstacles or effectors or wants_shade_death
            or wants_helio or ground is not None):
        return None

    obstacles = None
    if wants_obstacles:
        meshes = [o for o in s.obstacle_collection.all_objects
                  if o.type == 'MESH' and o is not obj and o.name != _attachments_name(obj)]
        to_plant = obj.matrix_world.inverted()
        bounds = _obstacle_bounds(meshes, to_plant) if meshes else None
        if bounds is not None:
            lo, hi = bounds
            padding = max(float(np.max(hi - lo)) * 0.08, profile.obstacle_clearance)
            lattice = Lattice.covering(lo, hi, OBSTACLE_RESOLUTION, padding=padding)
            solid = voxelise(meshes, lattice, obj.matrix_world)
            if solid.any():
                obstacles = Obstacles.from_solid(lattice, solid,
                                                 clearance=profile.obstacle_clearance)

    if obstacles is None and not profile.reads_light and not effectors \
            and not wants_shade_death and not wants_helio and ground is None:
        # Obstacle response was asked for but there is nothing to respond to.
        return None
    return Environment(obstacles=obstacles, effectors=effectors or None,
                       sun_target=sun_target, flower_sun_targets=flower_suns,
                       ground=ground)


def _toward(to_plant, target):
    """The plant-local unit direction from the plant's base to `target`'s
    position, or None (no object, or one sitting on the base)."""
    if target is None:
        return None
    local = to_plant @ target.matrix_world.translation
    v = np.array([local.x, local.y, local.z], dtype=float)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def _attachments_name(obj) -> str:
    return obj.get("plantgen_attachments_name", "")


def solid_voxels(environment: Environment | None) -> int:
    """How much solid matter the plant actually found, reported in the panel
    after a grow. Stage 3 is invisible until it changes something, and a
    collection that resolved to nothing looks exactly like one that is
    working -- this is the difference, and it is a stat written by the grow
    rather than anything a redraw could recompute."""
    if environment is None or environment.obstacles is None:
        return 0
    return int(np.count_nonzero(environment.obstacles.solid))
