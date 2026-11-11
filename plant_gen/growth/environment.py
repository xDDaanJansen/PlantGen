"""Stage 3 -- Environment: what the plant is responding to, and the fields it
reads to do it. Pure, bpy-free.

The architecture doc's stage order is the whole design here. Morphology says
what kind of plant this is, Development decides what grows and when, and
Environment *modifies* that. It never drives it. The previous build inverted
exactly this -- space colonization decided where branches went, when they
forked and when they stopped, with morphology reduced to decoration -- and the
rebuild plan's diagnosis of that inversion is the reason nothing in this module
creates, places or terminates anything on its own. Everything here returns a
number that development multiplies into a decision it was already making.

Two fields, one lattice:

  * **occupancy** -- where solid matter is. Supplied from outside (the Blender
    side voxelises whatever objects the user pointed at); this module only
    reads it. Blurred once at construction so it has a usable gradient a short
    way out from every surface, which is what lets a shoot lean away from a
    wall *before* touching it rather than only after.
  * **light** -- rebuilt each growth step from the plant's own foliage plus
    that same occupancy. Sky enters at the top and propagates down.

Both are sampled trilinearly, and both read as "nothing there" outside the
lattice -- full sun, no obstacle. That is the correct answer rather than a
fallback: outside the region we bothered to voxelise there is, by
construction, nothing to shade or block anything.

Solver constants (resolution, sweeps, absorption, blur radius) live here as
module constants with no UI, per the rebuild plan's own rule: a control earns
a panel row if it names a property of the plant, not of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # `effectors` imports `steer` from here, so only for typing.
    from .effectors import EffectorField

_EPS = 1e-9
_UP = np.array([0.0, 0.0, 1.0])

LIGHT_RESOLUTION = 28
"""Voxels along the longest axis of the light lattice. Small on purpose: light
here decides *how much* of a region is shaded, not where a leaf is, and the
grid is rebuilt every step. 28 keeps a full 40-step grow in the tens of
milliseconds while still resolving a clump into lit and shaded sides."""

LIGHT_ABSORPTION = 0.55
"""How much light one voxel's worth of foliage removes. Above ~0.8 a canopy
goes black underneath and nothing below the top branches ever releases."""

FOLIAGE_MASS = 0.35
"""Occlusion deposited per leaf, in units where 1.0 is an opaque voxel. A leaf
is not opaque and there are many per voxel, so this is well under one."""

_LATERAL_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# -- The lattice ----------------------------------------------------------

@dataclass(frozen=True)
class Lattice:
    """A regular voxel grid in world space. Shared by both fields so a sample
    never has to be resampled from one grid onto another."""

    origin: np.ndarray
    voxel_size: float
    shape: tuple[int, int, int]

    @staticmethod
    def covering(lo, hi, resolution: int, padding: float = 0.0) -> Lattice:
        lo = np.asarray(lo, dtype=float) - padding
        hi = np.asarray(hi, dtype=float) + padding
        extent = float(np.max(hi - lo))
        voxel = max(extent / max(resolution, 2), 1e-4)
        shape = tuple(int(max(2, np.ceil((hi[i] - lo[i]) / voxel) + 1)) for i in range(3))
        return Lattice(origin=lo, voxel_size=voxel, shape=shape)

    def coords(self, points: np.ndarray) -> np.ndarray:
        """Continuous voxel coordinates for an (n, 3) array of world points."""
        return (np.asarray(points, dtype=float) - self.origin) / self.voxel_size

    def indices(self, points: np.ndarray) -> np.ndarray:
        coords = np.floor(self.coords(points)).astype(np.int64)
        return np.clip(coords, 0, np.array(self.shape) - 1)

    def inside(self, position: np.ndarray) -> bool:
        coord = self.coords(np.asarray(position, dtype=float))
        return bool(np.all(coord >= 0.0) and np.all(coord <= np.array(self.shape) - 1.0))

    def sample(self, values: np.ndarray, position: np.ndarray, outside: float = 0.0) -> float:
        """Trilinear sample. `outside` is returned beyond the lattice rather
        than the edge value being extended -- a plant that grows past the
        voxelised region is in open air, not pressed against a wall."""
        coord = self.coords(np.asarray(position, dtype=float))
        limit = np.array(self.shape, dtype=float) - 1.0
        if np.any(coord < 0.0) or np.any(coord > limit):
            return float(outside)
        base = np.clip(np.floor(coord).astype(np.int64), 0, np.array(self.shape) - 2)
        frac = np.clip(coord - base, 0.0, 1.0)
        total = 0.0
        for dx in (0, 1):
            wx = frac[0] if dx else 1.0 - frac[0]
            for dy in (0, 1):
                wy = frac[1] if dy else 1.0 - frac[1]
                for dz in (0, 1):
                    wz = frac[2] if dz else 1.0 - frac[2]
                    total += wx * wy * wz * values[base[0] + dx, base[1] + dy, base[2] + dz]
        return float(total)

    def sample_many(self, values: np.ndarray, points: np.ndarray,
                    outside: float = 0.0) -> np.ndarray:
        """Vectorised `sample` over an (n, 3) array of world points. Points
        outside the lattice take `outside`. The scalar `sample` in a Python loop
        was fine for a handful of branch bases, but the mask/debug path samples
        every stem vertex and organ at once."""
        pts = np.asarray(points, dtype=float)
        n = pts.shape[0]
        if n == 0:
            return np.zeros(0)
        coord = self.coords(pts)
        limit = np.array(self.shape, dtype=float) - 1.0
        inside = np.all((coord >= 0.0) & (coord <= limit), axis=1)
        base = np.clip(np.floor(coord).astype(np.int64), 0, np.array(self.shape) - 2)
        frac = np.clip(coord - base, 0.0, 1.0)
        acc = np.zeros(n)
        for dx in (0, 1):
            wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
            for dy in (0, 1):
                wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
                for dz in (0, 1):
                    wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                    acc += wx * wy * wz * values[
                        base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz]
        return np.where(inside, acc, float(outside))

    def gradient(self, values: np.ndarray, position: np.ndarray,
                 outside: float = 0.0) -> np.ndarray:
        """Central difference of `values` at a world position, one voxel apart.
        Returns the direction the field *increases* in, unnormalised."""
        step = self.voxel_size
        out = np.zeros(3)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            ahead = self.sample(values, position + offset, outside)
            behind = self.sample(values, position - offset, outside)
            out[axis] = (ahead - behind) / (2.0 * step)
        return out


def box_blur(values: np.ndarray, passes: int) -> np.ndarray:
    """Repeated 3x3x3 box blur. Cheap stand-in for a distance transform: what
    the obstacle field needs is a smooth ramp reaching a stated distance out
    from every surface, and n passes of a box blur reach exactly n voxels.
    Uses `np.pad(mode="edge")` so a solid touching the lattice boundary stays
    solid at the boundary instead of being eroded by imaginary empty space."""
    out = np.asarray(values, dtype=float)
    for _ in range(max(0, passes)):
        padded = np.pad(out, 1, mode="edge")
        acc = np.zeros_like(out)
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                for dz in (0, 1, 2):
                    acc += padded[dx:dx + out.shape[0],
                                  dy:dy + out.shape[1],
                                  dz:dz + out.shape[2]]
        out = acc / 27.0
    return out


def _shift2d(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    padded = np.pad(a, 1, mode="edge")
    n0, n1 = a.shape
    return padded[1 + dx:1 + dx + n0, 1 + dy:1 + dy + n1]


def propagate_light(occlusion: np.ndarray,
                    absorption: float = LIGHT_ABSORPTION) -> np.ndarray:
    """Sky light entering the top of the lattice and working down through the
    occluders.

    One top-to-bottom raster pass: layer z is computed from layer z+1 *already
    updated in this same pass*, so light reaches the full depth of the grid in
    a single sweep. That cascade is load-bearing and was learned the hard way
    on the previous build -- a Jacobi-style update against the previous
    snapshot spreads light only one voxel deep per sweep, leaving everything
    below the top few layers at exactly zero however many sweeps you run, and
    the symptom is a plant that refuses to branch anywhere rather than an
    obviously wrong grid.

    **The kernel must conserve light in empty space, and getting that wrong is
    subtle.** The first version here averaged the five voxels above *plus* the
    four lateral neighbours in the layer being written, over 9. Those four
    start the pass at zero, so even a completely empty grid decayed by roughly
    0.55x per layer: measured light through an unshaded plant came out at a
    mean of 0.056, every bud read as deeply shaded, and shade tolerance below 1
    collapsed the plant to two axes. Averaging the five above over 5 leaves
    open air at exactly 1.0, and makes each layer depend only on the one above
    -- which is also why there is no sweep count any more: a second identical
    pass cannot change anything.
    """
    light = np.zeros_like(occlusion, dtype=float)
    light[:, :, -1] = 1.0
    for z in range(occlusion.shape[2] - 2, -1, -1):
        above = light[:, :, z + 1]
        total = above.copy()
        for dx, dy in _LATERAL_OFFSETS:
            total += _shift2d(above, dx, dy)
        # /5 keeps open air at 1.0; the lateral terms are what let light spill
        # sideways back under an occluder instead of leaving a hard column of
        # shadow all the way to the ground.
        light[:, :, z] = np.clip(total / 5.0 - occlusion[:, :, z] * absorption, 0.0, 1.0)
    return light


# -- What the plant responds to -------------------------------------------

@dataclass
class Obstacles:
    """Solid matter, voxelised. Built by the Blender side from whatever objects
    the user pointed at; nothing in `growth/` knows what a Blender object is.

    `solid` is 1.0 inside an object and 0.0 outside. `field` is that blurred
    out to the species' stated clearance, and is what growth actually reads:
    its gradient points into the obstacle, so growth steers down it.
    """

    lattice: Lattice
    solid: np.ndarray
    field: np.ndarray

    @staticmethod
    def from_solid(lattice: Lattice, solid: np.ndarray, clearance: float = 0.0) -> Obstacles:
        passes = int(round(max(0.0, clearance) / max(lattice.voxel_size, 1e-6)))
        return Obstacles(lattice=lattice, solid=np.asarray(solid, dtype=float),
                         field=box_blur(solid, min(passes, 8)))

    def blocked(self, position) -> bool:
        """Is this point inside solid matter? Reads `solid`, not the blurred
        field -- the blur exists to give the *approach* a gradient, and using
        it here would make a shoot stop a clearance-width short of anything."""
        return self.lattice.sample(self.solid, position, outside=0.0) > 0.5

    def resampled_onto(self, target: Lattice) -> np.ndarray:
        """This solidity, nearest-neighbour onto another lattice.

        The two grids are deliberately independent: obstacles want voxels sized
        to the *obstacle* (a 12 cm slab needs a few voxels of thickness or
        growth walks through it), and light wants voxels sized to the *plant*
        as it currently stands. Sharing one lattice made each ruin the other --
        sized for the plant's eventual reach, a real slab came out 16 voxels
        big and stopped nothing. Vectorised, so this costs one gather per step
        rather than a sample per voxel.
        """
        out = np.zeros(target.shape, dtype=float)
        axes_idx, axes_valid = [], []
        for axis in range(3):
            n = target.shape[axis]
            centres = target.origin[axis] + (np.arange(n) + 0.5) * target.voxel_size
            idx = np.floor((centres - self.lattice.origin[axis]) / self.lattice.voxel_size)
            idx = idx.astype(np.int64)
            valid = (idx >= 0) & (idx < self.lattice.shape[axis])
            if not valid.any():
                return out
            axes_idx.append(idx)
            axes_valid.append(np.nonzero(valid)[0])
        vx, vy, vz = axes_valid
        out[np.ix_(vx, vy, vz)] = self.solid[np.ix_(axes_idx[0][vx],
                                                    axes_idx[1][vy],
                                                    axes_idx[2][vz])]
        return out

    def proximity(self, position) -> float:
        """0 well clear of anything, rising toward 1 at a surface. This is what
        makes avoidance proportional: a shoot passing a wall at arm's length
        should barely notice it, and one about to hit it should turn hard."""
        return float(np.clip(self.lattice.sample(self.field, position, outside=0.0), 0.0, 1.0))

    def repulsion(self, position) -> np.ndarray:
        """Which way to lean to get clear. Zero where there is nothing nearby,
        so this costs nothing in an empty scene."""
        gradient = self.lattice.gradient(self.field, position, outside=0.0)
        norm = float(np.linalg.norm(gradient))
        if norm < _EPS:
            return np.zeros(3)
        return -gradient / norm


@dataclass
class Ground:
    """A terrain's height under the plant, sampled on a grid in the plant's
    own space: `heights[i, j]` is the surface z at `origin + (i, j) * cell`, NaN
    where the terrain has no surface. Read only by the gravity pass, which
    lands a stem that reaches it on it (`physics.apply_gravity`) -- so it
    never steers growth, and a plant clear of it bends as before."""

    origin: np.ndarray
    cell: float
    heights: np.ndarray

    def height(self, x: float, y: float) -> float | None:
        """The surface z at (x, y), bilinear between samples; None off the
        grid or where a sample has no surface (the flat ground applies)."""
        fx = (x - float(self.origin[0])) / self.cell
        fy = (y - float(self.origin[1])) / self.cell
        nx, ny = self.heights.shape
        if not (0.0 <= fx <= nx - 1 and 0.0 <= fy <= ny - 1):
            return None
        i, j = min(int(fx), nx - 2), min(int(fy), ny - 2)
        u, v = fx - i, fy - j
        h = self.heights[i:i + 2, j:j + 2]
        if np.isnan(h).any():
            return None
        return float((h[0, 0] * (1 - u) + h[1, 0] * u) * (1 - v)
                     + (h[0, 1] * (1 - u) + h[1, 1] * u) * v)


@dataclass
class Environment:
    """The world around one plant, and the light in it.

    Held for the duration of a grow and mutated in place: `obstacles` is fixed
    for the whole run, `light` is rebuilt every step because the plant is its
    own main occluder. A plant with no environment at all is `None` everywhere
    it is consulted, and every reader below returns the neutral value -- which
    is why turning Environment off reproduces the previous output exactly
    rather than approximately.
    """

    obstacles: Obstacles | None = None
    lattice: Lattice | None = None
    light: np.ndarray | None = None
    sun: np.ndarray = field(default_factory=lambda: _UP.copy())
    effectors: EffectorField | None = None
    """Hand-placed attractors and exclusion zones. Held here rather than passed
    separately through `grow` so development reads one object for everything
    about the surroundings, and so an effector automatically obeys the same
    "off means byte-identical" guarantee the rest of Stage 3 does."""
    sun_target: np.ndarray | None = None
    """The plant-local direction to the sun for heliotropism -- from the plant
    toward whatever object the user picked to track, or None when nothing is
    picked. A direction (position-derived, not a rotation), set by the Blender
    side from the tracked object's world position; `development.apply_heliotropism`
    turns organ facing toward it. Deliberately separate from `sun` above, which
    is the light field's fallback direction: these are two different "where is
    the sun" questions, and conflating them would let the tracked object silently
    steer phototropism as well."""
    ground: Ground | None = None
    """The terrain under the plant (its Ground Object), or None: the flat
    ground at the plant's base level. Inert during growth -- only the gravity
    pass reads it, so a ground-only Environment grows byte-identically."""
    flower_sun_targets: tuple = ()
    """Per inflorescence (0 the primary, k the k-th further one), the
    plant-local direction to the Sun Object its flower component tracks, or
    None where it picks none. Empty on a plant with no flower component, whose
    flowers turn to `sun_target` -- which is otherwise the leaves' alone."""

    def light_at(self, position) -> float:
        """0 (fully shaded) to 1 (open sky). Unbuilt or out of lattice reads as
        full sun."""
        if self.light is None or self.lattice is None:
            return 1.0
        return float(np.clip(self.lattice.sample(self.light, position, outside=1.0), 0.0, 1.0))

    def light_at_many(self, points) -> np.ndarray:
        """Vectorised `light_at`: 0 (shaded) to 1 (open sky) for an (n, 3) array.
        All open sky when there is no field to sample."""
        pts = np.asarray(points, dtype=float)
        if self.light is None or self.lattice is None:
            return np.ones(pts.shape[0] if pts.ndim == 2 else 0)
        return np.clip(self.lattice.sample_many(self.light, pts, outside=1.0), 0.0, 1.0)

    def light_direction(self, position) -> np.ndarray:
        """Unit vector toward more light, or up when the field is flat -- which
        it genuinely is out in the open, and up is where the sky is."""
        if self.light is None or self.lattice is None:
            return self.sun.copy()
        gradient = self.lattice.gradient(self.light, position, outside=1.0)
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-6:
            return self.sun.copy()
        return gradient / norm

    def repulsion_at(self, position) -> np.ndarray:
        if self.obstacles is None:
            return np.zeros(3)
        return self.obstacles.repulsion(position)

    def proximity_at(self, position) -> float:
        if self.obstacles is None:
            return 0.0
        return self.obstacles.proximity(position)

    def deflect(self, direction: np.ndarray, position, gain: float) -> np.ndarray:
        """Turn a growth direction away from whatever solid it is heading into.

        **Sliding, not repulsion.** The first version steered along the
        repulsion vector itself, and under a horizontal slab that vector points
        straight down -- so a shoot that met a ceiling grew back toward the
        ground instead of out from under it, which is neither what a plant does
        nor what anyone asks for when they say "grow around it". What actually
        happens is that the tip keeps going, minus whatever part of its heading
        was into the surface: it runs along the underside and emerges at the
        edge. Cancelling the inward component is exactly that, and it leaves a
        tip that is not heading into anything completely untouched.
        """
        if self.obstacles is None or gain <= 0.0:
            return direction
        proximity = self.obstacles.proximity(position)
        if proximity <= 0.0:
            return direction
        away = self.obstacles.repulsion(position)
        into = float(np.dot(direction, away))
        if into >= 0.0:
            return direction  # already heading out; nothing to correct
        tangent = direction - into * away
        norm = float(np.linalg.norm(tangent))
        if norm < _EPS:
            # Dead-on into a surface, with no tangent to slide along. Leaving
            # it is correct: the blocked test stops it at the surface, which is
            # what a shoot driven straight into a wall does.
            return direction
        return steer(direction, tangent / norm, min(1.0, proximity * gain))

    def blocked_at(self, position) -> bool:
        """Solid matter or a Forbid zone -- one question, so `_add_phytomer`
        has a single place a tip can run out of room rather than one per
        source."""
        if self.obstacles is not None and self.obstacles.blocked(position):
            return True
        return self.effectors is not None and self.effectors.blocked(position)

    @property
    def blocks_growth(self) -> bool:
        """Whether anything here can stop an apex at all. Checked once per grow
        so a plant with light response and nothing solid never pays for a
        per-node exclusion test.

        This replaced a gate on `profile.obstacle_clearance > 0`, which was the
        right question when obstacles were the only blocker but is the wrong
        one now -- a Forbid effector is placed in the scene and has no clearance
        to speak of. It is not a behaviour change for obstacles: the Blender
        side only ever builds `obstacles` when the clearance is above zero.
        """
        return self.obstacles is not None or (
            self.effectors is not None and self.effectors.forbids)

    def bias_direction(self, direction: np.ndarray, position) -> np.ndarray:
        if self.effectors is None:
            return direction
        return self.effectors.steer(direction, position)

    def bias_reference(self, reference: np.ndarray, direction: np.ndarray,
                       position) -> np.ndarray:
        """Twist a node's frame without touching where the stem goes. Separate
        from `bias_direction` because it is a separate quantity: the reference
        is what every organ's azimuth is measured from, so this re-aims the
        leaves and flowers on a node and moves the node itself not at all."""
        if self.effectors is None:
            return reference
        return self.effectors.twist(reference, direction, position)

    def update_light(self, plant) -> None:
        """Rebuild the light field for the plant as it stands right now.

        Occluders are the plant's own leaves and flowers -- its foliage is what
        casts the shade a plant competes with itself for -- plus any solid
        matter. Bare stems deposit nothing: a leafless twig shades almost
        nothing, and counting nodes rather than foliage was what made the
        previous build's light field track topology instead of canopy.
        """
        organs = [o for o in plant.all_organs() if o.scale > 0.0]
        if not organs:
            self.lattice = None
            self.light = None
            return

        points = np.array([o.position for o in organs], dtype=float)
        masses = np.array([FOLIAGE_MASS * max(o.scale, 0.0) for o in organs], dtype=float)

        lo = points.min(axis=0)
        hi = points.max(axis=0)
        if self.obstacles is not None:
            # Anything solid overhead has to be inside the light lattice or it
            # casts no shade -- which is most of the point of an obstacle.
            obstacle_lo = self.obstacles.lattice.origin
            obstacle_hi = obstacle_lo + (np.array(self.obstacles.lattice.shape)
                                         * self.obstacles.lattice.voxel_size)
            lo = np.minimum(lo, obstacle_lo)
            hi = np.maximum(hi, obstacle_hi)
        span = float(np.max(hi - lo))
        # Headroom above everything, so the top layer is genuinely open sky
        # rather than the first layer of leaves.
        hi = hi + np.array([0.0, 0.0, span * 0.15 + 1e-3])
        lattice = Lattice.covering(lo, hi, LIGHT_RESOLUTION, padding=span * 0.05 + 1e-3)

        occlusion = (self.obstacles.resampled_onto(lattice)
                     if self.obstacles is not None else np.zeros(lattice.shape, dtype=float))

        idx = lattice.indices(points)
        np.add.at(occlusion, (idx[:, 0], idx[:, 1], idx[:, 2]), masses)

        self.lattice = lattice
        self.light = propagate_light(occlusion)


def steer(direction: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
    """Rotate `direction` a `strength` fraction of the way toward `target`.

    A rotation rather than a blend, so the result is always unit length and
    behaves when the two are nearly opposed -- the same reason
    `development._rotate_toward` is a rotation. Duplicated here rather than
    imported to keep this module free of a development-side dependency; the
    two are three lines and neither is going to drift.
    """
    norm_t = float(np.linalg.norm(target))
    if strength <= 0.0 or norm_t < _EPS:
        return direction
    target = target / norm_t
    axis = np.cross(direction, target)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return direction
    angle = float(np.degrees(np.arccos(np.clip(float(np.dot(direction, target)), -1.0, 1.0))))
    axis = axis / norm
    theta = np.radians(angle * min(1.0, strength))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    rotated = (direction * cos_t
               + np.cross(axis, direction) * sin_t
               + axis * float(np.dot(axis, direction)) * (1.0 - cos_t))
    return rotated / max(float(np.linalg.norm(rotated)), _EPS)
