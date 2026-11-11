"""Flat-topped inflorescences: solving each pedicel so every flower in a head
lands on one chosen surface.

A corymb and a flat umbel are defined by their *outcome* -- all the flowers at
one level -- not by any fixed relationship between their pedicels. That makes
them a solve rather than a formula, and it has to happen after growth, once the
flowers' attachment points actually exist. `head_dome` bends the surface the
solve aims at, so "one plane" is the flat case rather than the only case.

`solve_flat_head` is the whole of it, and it is deliberately one pass. An
earlier version was two -- re-aim every pedicel for even spread, resolve, then
solve the lengths against a shared plane -- which needed the caller to
interleave a resolve between them and could only ever choose *one* of the two
things that decide where a flower ends up. Choosing the radius first and
solving the pedicel to reach it settles both at once, and costs nothing extra
because the pedicel was already being solved.

The first attempt at any of this scaled pedicel length by a fixed ratio along
the axis (longest at the base, 2.5x the shortest). That cannot work and the
numbers say why: a corymb's flowering region spans about 0.2 m of axis, while
its pedicels are around 0.03 m, so even a 2.5x ratio moves the lowest flower
0.045 m against the 0.2 m it needs to climb. The flowers stayed exactly where
the axis put them and the head was never flat.

The parametric add-on hit and solved the same thing (`pedicel_target_mode="flat_top"`),
and two of its hard-won findings are reused here rather than rediscovered. Both
are `solve_flat_head`'s own doing now:

* **Flat means flat against world up.** the parametric add-on measured it against the
  carrying axis and this module followed, on the reasoning that a corymb on a
  leaning branch should be flat relative to that branch -- which is *identical*
  on an upright stem and puts a compound corymb's near-horizontal rays' heads
  in a vertical plane. See `_flat_axis` for the measurements that reversed it.
* **A corymb's head is elongated along its axis; a flat umbel's is a disc.**
  Both are solved here, and the difference is where the flowers attach: a
  corymb's run *along* the axis, so those node positions are half the head's
  layout and the radius is an offset from each node. Collapsing them onto one
  centre turns a corymb into a flat umbel, which is a different inflorescence
  -- measured, it stripped a compound corymb's rim and pulled the whole head
  inward. A flat umbel needs no special case: its flowers all attach at the
  apex, so "offset from its own node" already is a disc.
* **Rank runs along the axis, height runs against gravity.** Which flower is
  outermost is a question about the inflorescence (the earliest, lowest one),
  and which plane it lands on is a question about gravity. Ranking by height
  works only while those coincide -- see `solve_flat_head`.
* **The azimuth the head is laid out on is `Organ.head_azimuth_deg`, not
  `divergence_deg`.** The solve overwrites the latter with its own answer, so
  reading it again on a second pass feeds the answer back in as the question.
* **Equal-solid-angle placement stops being uniform once lengths differ.** On a
  flat umbel, inner flowers solve to short pedicels and outer ones to long, so
  a sphere's even angular spacing lands them bunched at the centre and sparse
  at the rim. Radius on the surface has to follow the equal-area disc rule
  instead -- `head_radius * sqrt((rank + 0.5) / count)`, applied per flower
  inside the solve.
"""

from __future__ import annotations

import numpy as np

from .model import OrganKind, Plant
from .morphology import MorphologyProfile

_EPS = 1e-9
_PLANE_CLEARANCE = 0.35
"""How far above the highest attachment the shared plane sits, as a fraction
of head radius. Enough that every pedicel has a real climb to make -- a flower
solving to zero rise would need an infinitely shallow pedicel to reach any
radius at all."""


_UP = np.array([0.0, 0.0, 1.0])


def _flat_axis(axis) -> np.ndarray:
    """The direction 'flat' is measured against -- **world up**.

    **This reverses a decision recorded here, deliberately, and compound forms
    are what changed.** It used to be the axis's own heading where the
    inflorescence starts, on the reasoning that a corymb on a leaning branch
    should be flat relative to that branch. That was written when the only
    thing carrying a flat head was the plant's own stem, which is near enough
    upright that the two answers coincide -- measured, a simple corymb is
    **identical** under either rule, down to the pedicel elevations and the
    1.7 mm it sits from flat.

    A compound corymb is the case it was never tested against. Its rays are
    near-horizontal by design -- that is what spreads the head -- so
    "perpendicular to the ray" is a *vertical* plane, and the umbellet ends up
    standing on edge. Measured on a ray 81.7 degrees off vertical: the
    lowest-attached flowers came out at +24.5, -3.2 and -15.3 degrees of world
    elevation, i.e. pointing sideways and downward, reaching 75-83 mm out.

    World up is also what the word means. A corymb is defined by its flowers
    being at one *level*, and level is a statement about gravity; real
    pedicels are gravitropic and hold the head up whatever the stem does.

    `axis` is unused now and kept so the call reads as a question about this
    head rather than a constant.
    """
    _ = axis
    return _UP.copy()


def _launch_angles(tangent: np.ndarray, reference: np.ndarray,
                   direction: np.ndarray) -> tuple[float | None, float]:
    """The (divergence, elevation) that `model.radial_direction` needs in order
    to aim a pedicel along `direction` from this phytomer's frame.

    **The other half of measuring the head against world up.** The solve
    chooses where a flower should end up in world space, but a pedicel is
    launched from its own phytomer's frame -- `resolve_geometry` calls
    `radial_direction(phytomer.direction, phytomer.reference, ...)`. Writing
    the angle as though those two frames agreed is what made the fix
    incomplete: on a near-vertical stem they do, and on an 81.7-degree ray they
    are 81.7 degrees apart, so the pedicel launched nowhere near where the
    solve had put it.

    Returns `None` for the divergence when the direction lies along the
    tangent, where an azimuth is undefined and the caller should keep whatever
    the flower already had.
    """
    sin_phi = float(np.clip(np.dot(direction, tangent), -1.0, 1.0))
    elevation = float(np.degrees(np.arcsin(sin_phi)))
    radial = direction - tangent * sin_phi
    norm = float(np.linalg.norm(radial))
    if norm <= _EPS:
        return None, elevation
    radial /= norm
    across = float(np.dot(np.cross(reference, radial), tangent))
    along = float(np.dot(reference, radial))
    return float(np.degrees(np.arctan2(across, along))) % 360.0, elevation


def _head_basis(axis, head_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two perpendicular directions spanning the head's own plane, so a
    flower's azimuth can be laid out as a real 360-degree fan in it.

    **Built once per head rather than projected per flower, and that is the
    difference between a fan and a fold.** Flattening each flower's own radial
    into the plane looks equivalent and is not: a phytomer's radial sweeps a
    circle in the plane *perpendicular to its stem*, which for a near-horizontal
    branch is a vertical circle -- so projecting it collapses 360 degrees of
    spread onto a single horizontal line, and every flower ends up on one of two
    opposite sides with none in between. Reported as the branches all sitting on
    the world-Z side.

    Zero degrees is the axis's own frame vector where its inflorescence starts,
    flattened -- so on an upright stem the azimuths are exactly the ones
    phyllotaxis already gave, and the head only starts to differ as the stem
    leans. Falls back to the stem's own heading, then to a fixed axis, for the
    cases where that vector is itself vertical.
    """
    index = axis.flowering_from if axis.flowering_from >= 0 else len(axis.phytomers) - 1
    index = min(max(index, 0), len(axis.phytomers) - 1)
    phytomer = axis.phytomers[index]
    for candidate in (np.array(phytomer.reference, dtype=float),
                      np.array(phytomer.direction, dtype=float),
                      np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0])):
        flat = candidate - head_up * float(np.dot(candidate, head_up))
        norm = float(np.linalg.norm(flat))
        if norm > _EPS:
            e0 = flat / norm
            return e0, np.cross(head_up, e0)
    raise AssertionError("head_up is not a unit vector")


def _in_plane(point: np.ndarray, head_up: np.ndarray) -> np.ndarray:
    """`point` with its height removed, so what is left is where it sits in the
    head's own plane."""
    return point - head_up * float(np.dot(point, head_up))


def _flowers_of(axis):
    """Every flower on this axis paired with the phytomer carrying it. A set
    fruit counts (day path only): the post-bend flat-head solve runs after
    `apply_fruit`, and a fruit sitting where its flower was still needs its
    pedicel re-aimed to keep the head flat."""
    out = []
    for phytomer in axis.phytomers:
        for organ in phytomer.organs:
            if organ.kind in (OrganKind.FLOWER, OrganKind.FRUIT):
                out.append((phytomer, organ))
    return out


def solve_flat_head(plant: Plant, profile: MorphologyProfile, seed: int = 0) -> None:
    """Place every flower of a flat-topped head at a chosen (radius, height)
    and derive the pedicel that reaches it.

    This replaces solving only for height. Solving height alone leaves the
    radius to whatever a fixed launch angle happens to produce: at 55 degrees
    a flower climbing h lands at 1.4h out, so a corymb spanning 0.2 m of axis
    threw its lowest flowers 0.29 m sideways -- the wide sparse fan. Choosing
    the radius first and solving the pedicel to hit it is what makes the head
    compact, and it costs nothing extra because the pedicel was already being
    solved.

    Radius follows the equal-area disc rule, `head_radius * sqrt(rank)`, so
    density is even across the head rather than crowded at one end. Rank runs
    outward from the flowers highest on the axis (shortest climb, innermost)
    to those lowest (longest climb, outermost), which is the arrangement a
    real corymb has and keeps the long pedicels on the outside where they
    belong.

    `head_dome` bends the target surface: the height each flower reaches drops
    with its own distance from the centre, so the head reads as a dome rather
    than a plate. **The radius is held while the height moves**, which is what
    keeps the dome separate from `head_radius`. Scaling the solved lengths
    instead -- the obvious reading of "make the outer stalks shorter" -- pulls
    each flower in *and* down along its own ray, so the head narrows as it
    domes and the two controls fight: at a 0.7 multiplier the head is 30%
    narrower and `head_radius` has to be raised to undo it.
    """
    # Imported here rather than at module scope: `development` reaches into
    # this module the same way, and only from inside a function, for the same
    # reason.
    from .development import _T_HEAD_DOME, _signed

    # Per axis, not once up front: with a lateral inflorescence a plant can be
    # flat-topped on its branches and not on its trunk (a compound corymb is
    # exactly that), so the test belongs inside the loop.
    for axis in plant.axes.values():
        inf = profile.inflorescence_of(axis)
        if not inf.is_flat_topped:
            continue
        flowers = _flowers_of(axis)
        if len(flowers) < 2:
            continue
        flat_axis = _flat_axis(axis)
        e0, e1 = _head_basis(axis, flat_axis)

        # Height of the shared plane: a little above the highest attachment,
        # so every pedicel has a positive climb to make.
        heights = [float(np.dot(p.tip_position, flat_axis)) for p, _ in flowers]
        target = max(heights) + inf.head_radius * _PLANE_CLEARANCE

        # **Innermost is the flower attached nearest the apex, ranked by its
        # position ALONG the axis rather than by its world height.**
        #
        # Those are the same ordering on an upright stem, and only one of them
        # survives a branch that leans. Ranking by height was right while the
        # head was measured in the carrying axis's own frame; once the plane
        # became world up it quietly became noise, because a near-horizontal
        # ray's flowers are all at nearly the same height and what little
        # difference there is comes from how the branch happened to bend. The
        # radius each flower got was therefore close to random, which scrambled
        # the head and pulled its mass inward -- measured on a compound flat
        # umbel, the outermost fifth of the head went from 32 flowers to 8.
        #
        # Position along the axis is well defined at any tilt and is the
        # botanically meaningful order anyway: a corymb's lowest, earliest
        # flower is its outermost, which is exactly what gives it the flat top.
        # `_flowers_of` walks the phytomers in order, so the list index is that
        # position already.
        count = len(flowers)
        order = list(range(count - 1, -1, -1))
        for rank, i in enumerate(order):
            phytomer, organ = flowers[i]
            radius = inf.head_radius * float(np.sqrt((rank + 0.5) / count))
            # The curve's X is the flower's own distance from the centre as a
            # fraction of the head radius, so the graph is a real cross-section
            # of the head rather than a ranking of its flowers.
            out = radius / inf.head_radius if inf.head_radius > _EPS else 0.0
            drop = inf.head_radius * (
                inf.head_dome * inf.head_dome_profile(out)
                + inf.head_dome_variation * _signed(seed, _T_HEAD_DOME, organ.id))
            # **Where this flower goes, as a point in the head, not as an
            # offset from its own node.** Its azimuth is laid out on the head's
            # own basis, its radius is measured from the head's own centre, and
            # its height is the shared plane less whatever the dome drops it
            # by. Nothing here depends on where its node happened to end up,
            # which is the whole fix: the node decides the *pedicel*, not the
            # flower's place in the head.
            # Captured on the first solve and reused after: `divergence_deg`
            # is overwritten below with the launch azimuth, so reading it again
            # on the second pass would feed an answer back in as a question.
            # See `Organ.head_azimuth_deg`.
            if organ.head_azimuth_deg < 0.0:
                organ.head_azimuth_deg = float(organ.divergence_deg) % 360.0
            angle = np.radians(organ.head_azimuth_deg)
            # **Offset from the flower's own node, not from one head centre.**
            # A corymb attaches its flowers *along* its axis, so where those
            # nodes are IS half the head's layout -- collapsing them onto the
            # apex turns a corymb into a flat umbel, which is a different
            # inflorescence. Measured when it was: a compound corymb's whole
            # head went from an even 0/26/46/29/29 flowers per equal-area ring
            # to 35/36/26/26/7, mass pulled inward and the rim stripped.
            #
            # Ranking by position along the axis is what keeps this bounded,
            # and the two only work together: the node nearest the apex -- the
            # one furthest out on a leaning branch -- takes rank 0 and so the
            # *smallest* radius, while the largest radius goes to the node at
            # the base, which is nearest the centre. So no flower can be thrown
            # beyond the head, which is the complaint this pair started from.
            point = (_in_plane(np.array(phytomer.tip_position, dtype=float), flat_axis)
                     + (e0 * np.cos(angle) + e1 * np.sin(angle)) * radius
                     + flat_axis * (target - drop))
            # The stalk that reaches it, in world space, then expressed in this
            # phytomer's own frame. Writing `90 - polar` straight into
            # `elevation_deg` was only ever right because the head's normal and
            # the phytomer's heading used to be the same vector; on a branch
            # that leans they are not.
            #
            # No guard on the height: a flower domed below its own attachment
            # simply gets a pedicel angling down, which this expresses
            # directly.
            stalk = point - np.array(phytomer.tip_position, dtype=float)
            length = float(np.linalg.norm(stalk))
            if length <= _EPS:
                continue
            divergence, elevation = _launch_angles(
                np.array(phytomer.direction, dtype=float),
                np.array(phytomer.reference, dtype=float),
                stalk / length)
            organ.pedicel_length = length
            organ.elevation_deg = elevation
            if divergence is not None:
                organ.divergence_deg = divergence
