"""Effectors: the artist's hand on Stage 3. Pure, bpy-free.

Everything else in Environment is something the plant reads off the world --
how much sky a bud can see, what solid matter is in the way. An effector is
the other thing an artist needs: a point placed by hand that growth leans
toward or away from, so a plant can be trained around a doorway or made to
reach across a scene without editing the species.

The eight kinds follow the idea of Natsura's effectors, as its public
documentation describes them; the implementation is PlantGen's own. They are
deliberately *not* a new tropism:
gravitropism and phototropism are properties of the species and live on the
profile, while an effector is a property of the scene and lives on an object
in it.

The eight fall into two families, and the split is the one real structure here:

* **Six steer the heading** -- Attract, Repel, Forbid, Target, Flatten, Orbit.
  They answer "where does this tip go next".
* **Two twist the frame** -- Align and Level. They answer "which way is up on
  the stem that just grew".

**What "twisting the frame" actually reaches, stated exactly, because the
obvious shorter claim is wrong.** `Phytomer.reference` is the zero that every
divergence angle on that node is measured from -- so it sets the azimuth of the
leaves *and* of the axillary buds in their axils, which means it also sets the
azimuth any branch launches at. On an unbranched plant the frame kinds
therefore move the stem by exactly nothing (measured: 0.0000 mm) and only
re-aim foliage. On a branched one they visibly reshape the plant, because every
branch now leaves at a different angle around its parent (measured: up to 335
mm of node displacement on the default perennial).

That is correct rather than a leak. A leaf and the bud in its axil are at the
same azimuth by definition -- that is what "axillary" means -- so a frame twist
has to carry both or the phytomer model stops being true. Worth knowing before
reaching for Align on a bushy species expecting only the leaves to turn.

**A frame twist rotates the phyllotactic set; it does not collapse it.** With a
137.5-degree spiral the leaves stay spread all the way round the stem and the
whole arrangement simply turns, so aggregate "are the leaves facing the sun"
statistics barely move even though individual organs turn by tens of degrees.
It reads strongest on distichous, opposite and unilateral species, where there
are only one or two azimuths to aim in the first place.

Adding six kinds cost six enum entries and no new sliders, which is the shape
this project wants: a categorical trait belongs in a dropdown, and only a
continuous quantity earns a slider.

**Stage 3's contract holds without exception.** An effector turns a direction
the plant was already going to take, and a Forbid stops a tip exactly the way
a wall does -- through the same `blocked` test in `development._add_phytomer`,
not a second termination path. Nothing here creates, places or times anything.

**The old build's tuning trap cannot recur here, and it is worth knowing why.**
There, `strength=3.0` on an attractor collapsed a 500-node plant to 68 clustered
at the origin: the effector vector was *added* to a growth direction, so a big
enough strength drowned everything else out. `steer` is a fractional rotation
instead -- 1.0 means "point exactly at it this step" and there is nothing above
that -- so the strength slider saturates rather than exploding. Combined pulls
are averaged by weight, so two opposed effectors cancel instead of fighting for
whichever one is applied last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import steer

_EPS = 1e-9


class EffectorKind:
    # Kinds that steer where a tip goes.
    ATTRACT = "ATTRACT"
    REPEL = "REPEL"
    FORBID = "FORBID"
    TARGET = "TARGET"
    FLATTEN = "FLATTEN"
    ORBIT = "ORBIT"
    # Kinds that twist the growth frame rather than the heading.
    ALIGN = "ALIGN"
    LEVEL = "LEVEL"


KINDS = (EffectorKind.ATTRACT, EffectorKind.REPEL, EffectorKind.FORBID,
         EffectorKind.TARGET, EffectorKind.FLATTEN, EffectorKind.ORBIT,
         EffectorKind.ALIGN, EffectorKind.LEVEL)

_FRAME_KINDS = (EffectorKind.ALIGN, EffectorKind.LEVEL)
"""Align and Level twist the *frame*, not the heading. See the module docstring
for exactly how far that reaches.

**Two traps in measuring these, both hit for real before either shipped.**

*Look at organ facing, not organ position.* A sessile organ sits exactly on its
node, so a 90-degree frame twist turns 143 of 166 organs by 49 degrees on
average and moves their positions 0.00 mm. Position reports a working feature
as doing nothing whatsoever.

*Pick a target the frame is not already aligned with.* An unbranched upright
plant's frame comes out at almost exactly (0, 1, 0) on every node, so a Level
tested against a +X axis is asking it to fix something already perfect -- which
measures 0.000 before and after and reads as broken. Against +Y the same code
takes the mean |reference . axis| from 1.000 to 0.086.
"""

# How far away an effector's influence is measured from. The three geometries
# are not decoration: a Flatten's influence has to fall off with distance from
# its *plane* (a slab) and an Orbit's with distance from its *axis* (a
# cylinder), because measuring either from the centre point would make an
# effector meant to shape a whole trellis or pole act only in one small ball.
_SPHERE, _SLAB, _CYLINDER, _INFINITE = "sphere", "slab", "cylinder", "infinite"

_FALLOFF_SHAPE = {
    EffectorKind.ATTRACT: _SPHERE,
    EffectorKind.REPEL: _SPHERE,
    EffectorKind.FORBID: _SPHERE,
    EffectorKind.TARGET: _INFINITE,
    EffectorKind.FLATTEN: _SLAB,
    EffectorKind.ORBIT: _CYLINDER,
    EffectorKind.ALIGN: _SPHERE,
    EffectorKind.LEVEL: _SPHERE,
}


@dataclass(frozen=True)
class EffectorSpec:
    """One effector, in the plant's own local space.

    Local, not world, and that is the whole of the Stage 3 coordinate bug this
    project already paid for once: `bridge.regrow` writes the skeleton straight
    into the object's mesh, so growth has always worked in object-local
    coordinates with the plant's base at the origin. An effector converted from
    a Blender Empty in world space sits somewhere else entirely, and the symptom
    is not an error -- it is growth quietly ignoring an effector that is visibly
    right next to it.

    `axis` is the Empty's own +Z, and it makes an effector's *rotation*
    load-bearing for four of the eight kinds -- the plane Flatten flattens to,
    the pole Orbit spirals around, the plane Level lies flat in. The Blender
    side draws those kinds with a gizmo that shows the axis rather than a
    featureless sphere, because a control that silently matters is worse than
    one that is missing.
    """

    kind: str
    position: np.ndarray
    radius: float
    strength: float
    axis: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.axis is None:
            object.__setattr__(self, "axis", np.array([0.0, 0.0, 1.0]))


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return None if norm < _EPS else vector / norm


def _distance(effector: EffectorSpec, offset: np.ndarray) -> float:
    """How far a point is from this effector, measured in the geometry its kind
    actually cares about. `offset` runs from the point to the effector."""
    shape = _FALLOFF_SHAPE[effector.kind]
    if shape == _SLAB:
        return abs(float(np.dot(offset, effector.axis)))
    if shape == _CYLINDER:
        along = float(np.dot(offset, effector.axis)) * effector.axis
        return float(np.linalg.norm(offset - along))
    return float(np.linalg.norm(offset))


def _falloff(effector: EffectorSpec, distance: float) -> float:
    """How much of an effector's strength reaches a point.

    Smoothstep rather than linear, so the influence eases in at the boundary
    instead of switching on -- a tip crossing the edge of a magnet should start
    to lean, not flinch.

    **Smoothstep specifically, not the squared falloff this shipped with
    first.** `(1 - d/r)^2` is smooth at the outer edge only and is brutal
    through the middle: at 80% of the radius it passes 4% of the strength.
    That was survivable for the kinds that steer a heading, because a tip
    re-reads them every node and small turns accumulate over a whole stem. It
    was not survivable for Align and Level, which are re-established from the
    transported frame each node and so converge at whatever rate the falloff
    allows -- measured, a full-strength Align crawled from 90 degrees off
    target to 59 over an entire stem, reading as "barely does anything".
    Smoothstep passes 50% at half the radius where the square passes 25%, and
    is smooth at both ends.

    Target has none: it is a homing goal rather than a local magnet, which is
    the one thing Attract with a very large radius cannot express (at a radius
    that reaches, the falloff at the plant's own distance is already near
    zero, so the pull arrives too weak to train anything).
    """
    if _FALLOFF_SHAPE[effector.kind] == _INFINITE:
        return 1.0
    if effector.radius <= _EPS:
        return 0.0
    near = 1.0 - min(distance / effector.radius, 1.0)
    return near * near * (3.0 - 2.0 * near)


def _rotate_about(vector: np.ndarray, axis: np.ndarray, radians: float) -> np.ndarray:
    """Rodrigues. Written out here rather than imported from `model` for the
    same reason `environment.steer` is: this module stays free of a dependency
    on the data layer, and it is five lines that are not going to drift."""
    cos_t, sin_t = float(np.cos(radians)), float(np.sin(radians))
    return (vector * cos_t
            + np.cross(axis, vector) * sin_t
            + axis * float(np.dot(axis, vector)) * (1.0 - cos_t))


def _twist(reference: np.ndarray, direction: np.ndarray,
           target: np.ndarray, strength: float) -> np.ndarray:
    """Turn `reference` a fraction of the way toward `target` **about
    `direction`**, rather than by the minimal rotation between the two.

    The distinction is the whole point. A minimal rotation would tilt the
    reference off the plane perpendicular to `direction`, and the frame's
    entire job is that it stays perpendicular -- `radial_direction` builds
    every organ's azimuth on that assumption. Rotating about the heading keeps
    it perpendicular by construction instead of relying on a re-normalisation
    to put it back.
    """
    flat = _unit(target - float(np.dot(target, direction)) * direction)
    if flat is None or strength <= 0.0:
        return reference
    angle = float(np.arctan2(
        float(np.dot(np.cross(reference, flat), direction)),
        float(np.dot(reference, flat)),
    ))
    return _rotate_about(reference, direction, angle * min(1.0, strength))


@dataclass(frozen=True)
class EffectorField:
    """Every effector acting on one plant, and the two questions growth asks
    of them."""

    specs: tuple[EffectorSpec, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.specs)

    @property
    def forbids(self) -> bool:
        """Whether any effector can stop a tip. Checked once so a scene with
        only magnets never pays for a per-node exclusion test."""
        return any(e.kind == EffectorKind.FORBID and e.strength > 0.0
                   for e in self.specs)

    def blocked(self, position) -> bool:
        """Inside a Forbid zone. Analytic rather than voxelised: a sphere is
        exactly testable, so this is both cheaper and more accurate than
        putting a sphere mesh in the obstacle collection would be. It reaches
        growth through `Environment.blocked_at`, so obstacles and Forbid share
        one decision point rather than being two ways to stop an apex.
        """
        point = np.asarray(position, dtype=float)
        for effector in self.specs:
            if effector.kind != EffectorKind.FORBID or effector.strength <= 0.0:
                continue
            if float(np.linalg.norm(point - effector.position)) < effector.radius:
                return True
        return False

    def _target(self, effector: EffectorSpec, offset: np.ndarray,
                direction: np.ndarray) -> np.ndarray | None:
        """Where this one effector wants the heading to point, or None if it
        has nothing to say here. `offset` runs from the tip to the effector."""
        kind = effector.kind

        if kind == EffectorKind.FLATTEN:
            # Grow in the plane the effector's axis is normal to: keep only the
            # part of the heading that already lies in it. This bends growth
            # *parallel* to the plane rather than dragging it onto the plane --
            # the espalier/fan case, where what reads is that nothing leaves
            # the plane, not that everything meets it at one height.
            return _unit(direction - float(np.dot(direction, effector.axis)) * effector.axis)

        if kind == EffectorKind.ORBIT:
            # Tangential: perpendicular to both the axis and the spoke out to
            # this tip, which is a circle around the axis. Paired with the
            # plant's own upward gravitropism that circle becomes a helix,
            # which is the spiral; paired with an Attract on the same spot it
            # is a stem twining up a pole.
            along = float(np.dot(offset, effector.axis)) * effector.axis
            spoke = _unit(offset - along)
            if spoke is None:
                return None  # sitting on the axis: no spoke, so no tangent
            return _unit(np.cross(effector.axis, spoke))

        toward = _unit(offset)
        if toward is None:
            # Sitting exactly on it: no direction to pull along, and for a
            # Forbid the `blocked` test above has already stopped this tip.
            return None
        if kind in (EffectorKind.REPEL, EffectorKind.FORBID):
            return -toward
        return toward

    def steer(self, direction: np.ndarray, position) -> np.ndarray:
        """Turn a growth direction by everything acting on this point.

        Accumulated into one weighted pull and applied as a single rotation,
        rather than each effector steering in turn. Applying them in sequence
        would make the result depend on collection order, and two opposed
        effectors would resolve to whichever happened to be last instead of
        cancelling.
        """
        if not self.specs:
            return direction
        point = np.asarray(position, dtype=float)
        pull = np.zeros(3)
        total = 0.0
        for effector in self.specs:
            if effector.strength <= 0.0 or effector.kind in _FRAME_KINDS:
                continue
            offset = effector.position - point
            weight = effector.strength * _falloff(effector, _distance(effector, offset))
            if weight <= 0.0:
                continue
            target = self._target(effector, offset, direction)
            if target is None:
                continue
            pull += target * weight
            total += weight
        if total <= 0.0:
            return direction
        return steer(direction, pull, min(1.0, total))

    def twist(self, reference: np.ndarray, direction: np.ndarray, position) -> np.ndarray:
        """Turn the frame's up vector by every Align and Level acting here.

        A second accumulator rather than a second pass over the first, because
        this is a different quantity: the heading answers where a tip goes next,
        the frame answers which way is up on the stem behind it. Folding them
        together would make a leaf-facing effector steer the apex.
        """
        if not self.specs:
            return reference
        point = np.asarray(position, dtype=float)
        pull = np.zeros(3)
        total = 0.0
        for effector in self.specs:
            if effector.strength <= 0.0 or effector.kind not in _FRAME_KINDS:
                continue
            offset = effector.position - point
            weight = effector.strength * _falloff(effector, _distance(effector, offset))
            if weight <= 0.0:
                continue
            if effector.kind == EffectorKind.ALIGN:
                # Up points at the effector, so every leaf on the node turns to
                # face it. With the effector standing in for the sun this is
                # leaf heliotropism, which nothing else here expresses --
                # `light_seeking` turns shoots, not the organs on them.
                target = offset
            else:
                # Level: up lies flat in the effector's own plane, which is the
                # one vector perpendicular to both its axis and the heading.
                # Signed toward whichever of the two it already sits nearer, so
                # a stem does not flip its whole frame over between nodes.
                flat = _unit(np.cross(effector.axis, direction))
                if flat is None:
                    # Heading parallel to the axis: every perpendicular
                    # reference is already flat to the plane, so there is
                    # nothing to correct rather than nothing defined.
                    continue
                target = flat if float(np.dot(flat, reference)) >= 0.0 else -flat
            pull += target * weight
            total += weight
        if total <= 0.0:
            return reference
        return _twist(reference, direction, pull, min(1.0, total))
