"""Stage 4 -- Physical Form: how does the grown structure behave?

Gravity applied as a **post-growth deformation**. The architecture doc is
explicit that this is separate from gravitropism: gravitropism changes how the
plant *grows*, gravity changes how the already-grown structure *bends*. The
topology is untouched -- same nodes, same parents, same counts -- only
directions are rewritten, and `resolve_geometry` derives the new positions.

The model is a cantilever. At each node the bending moment is the weight it
carries beyond that point times the horizontal distance to that weight's
centroid; the section's resistance is its second moment of area, proportional
to r^4. So a thin twig holding a flower head bends hard and a thick base
holding the same load barely moves -- the r^4 term spans four orders of
magnitude across a plant whose radii span one.

Why Python rather than Geometry Nodes, which the plan left open: droop has to
respect the chain. A branch that bends carries its children with it, so the
rotation at a node applies to everything beyond it. That is a parent-to-child
accumulation over an explicit tree, which is natural here -- axes are already
objects with parent links -- and awkward in a node graph. It also has to run
between growth and diameter's consumers, which are both already Python.
"""

from __future__ import annotations

import numpy as np

from .model import Axis, Phytomer, Plant, _cross, parallel_transport, rotate_about
from .morphology import MorphologyProfile

_UP = np.array([0.0, 0.0, 1.0])
_DOWN = -_UP
_EPS = 1e-9
_G = 9.81
"""Standard gravity, m/s**2. The bend is a real Euler-Bernoulli beam in SI
units now, so the acceleration that turns mass into force is explicit rather
than folded into a unitless `droop` multiplier."""
_KINK_RATIO = 15.0
"""A stem's minimum bend radius as a multiple of its own radius: a real tube
kinks once bent tighter than roughly fifteen diameters, so curvature is capped
at ``1 / (_KINK_RATIO * r)``. Radius-aware and resolution-free -- it bounds the
physical tightness of a bend, while `smooth_bent_arcs` bounds how coarsely that
bend is drawn. Together they are the "never a sharp joint" guarantee."""
_T_STIFFNESS = 202
"""Its own random stream, so drawing a per-axis stiffness cannot shift any
existing seeded draw."""


def _signed(seed: int, tag: int, ident: int) -> float:
    """A deterministic value in [-1, 1) keyed on one element, not on visit order
    -- the construction `diameter._signed` uses (a local copy rather than an
    import across the diameter/physics boundary)."""
    return float(np.random.default_rng([int(seed), int(tag), int(ident)]).random()) * 2.0 - 1.0


_LOAD_ECCENTRICITY = 0.02
"""How far off its own axis a load is assumed to sit, as a fraction of its
distance from the node carrying it.

Without this the model cannot bend a straight stem *at all*, and the reason is
worth stating because it looks like a bug and is not: the bending moment is
`mass * horizontal offset to the centre of mass`, and on a dead-vertical stem
that offset is exactly zero. Measured: growth irregularity at 0 gave a tip tilt
of exactly 0.000 degrees at any flower weight, because irregularity was the
only thing making a stem crooked enough to give gravity a lever. A perfectly
balanced vertical column really is in equilibrium -- but a sunflower nods
because its head is heavy, not because its stem happened to wobble, so
depending on the wobble is useless as a tool.

This is the standard engineering treatment of the same problem: a real column
is never perfectly straight or perfectly centred, so it is given a small
initial imperfection. `max` rather than a sum, so a stem that is *already*
leaning past this keeps its own honest geometry and only a near-straight one is
helped.

Calibrated against the goldens rather than picked: at this value the nine tuned
inflorescences drift by 0.77 mm, which is the bending-plane fallback below and
not this, while a dead-straight heavy-headed stem spans upright to fully folded
across Droop's own 0..2. A first pass at 0.15 was measurably too strong -- it
pushed every control past the bend cap, so Peduncle Droop 0.5 and 2.0 both read
as "fully folded" and the two stopped being distinguishable."""

_UPRIGHT_COS = float(np.cos(np.radians(20.0)))
"""How near vertical counts as upright for `_turn_to_lowest`. Twenty degrees is
generous on purpose: the ambiguity it resolves is real anywhere the signed turn
approaches half a circle, not only at exactly vertical, and every stem this
matters for is one that has barely begun to bend."""

_LEAN_MIN = 1e-4
"""Below this a horizontal component is noise rather than a lean, and its
direction must not be trusted. See `_fall_direction`."""

def _organ_mass(organ, profile: MorphologyProfile) -> float:
    # A flower head is a lump (mass ~ scale**3), a leaf is a plate (mass ~
    # scale**2). Both scale with the organ's own expansion, so a bud that has
    # not opened yet does not pull its stem over. The cubed flower is what makes
    # Flower Size a real lever on the nod -- doubling a head's size makes it
    # eight times heavier, so a big-headed stem droops hard where a small-headed
    # one of the same species stands up.
    #
    # A catkin counts as a flower here: it is a hanging reproductive lump, not a
    # plate, and it must ride `flower_weight` rather than `leaf_weight` -- a real
    # catkin is far heavier than a leaf, and coupling its mass to leaf size/weight
    # made a foliage tweak swing the whole twig.
    if organ.kind == "fruit":
        # A ripe fruit is a heavier lump than the flower it set from, on its own
        # weight -- what bows a fruiting branch. Only ever reached on the day
        # path, where `lifecycle.apply_fruit` has flipped set flowers to fruit
        # before the bend; off it no organ is a fruit, so this is byte-identical.
        return organ.scale ** 3 * profile.fruit_weight
    if organ.kind in ("flower", "catkin"):
        return organ.scale ** 3 * profile.flower_weight
    return organ.scale ** 2 * profile.leaf_weight


def _phytomer_mass(phytomer, profile: MorphologyProfile, terminal_organ=None) -> float:
    """Woody mass plus whatever organs hang at this node. Relative units --
    only ratios matter, since `stiffness` is calibrated against them.

    `terminal_organ` is the flower capping an axis, which lives on the Axis
    rather than in any phytomer's own list. Omitting it made a determinate
    plant's terminal flower weightless: flower_weight could be raised by 5x
    with the tip tilt not moving a single degree, which is exactly the case
    this stage is supposed to demonstrate.
    """
    wood = np.pi * phytomer.radius ** 2 * phytomer.length * profile.wood_density
    organs = 0.0
    if terminal_organ is not None:
        organs += _organ_mass(terminal_organ, profile)
    for organ in phytomer.organs:
        organs += _organ_mass(organ, profile)
    return float(wood + organs)


def _axis_level(plant: Plant, axis: Axis) -> int:
    """The branching level an axis counts as: its own, or -- an inflorescence
    branch, which has none -- the nearest ancestor's."""
    while axis is not None:
        if axis.level >= 0:
            return axis.level
        axis = plant.axes.get(axis.parent_axis_id)
    return -1


def _inflorescence_axes(plant: Plant) -> set:
    """Axes that are inflorescence structure (`structural_depth >= 1`, not a
    vegetative fork), with their subtrees -- the bridge's own definition."""
    out: set = set()
    for axis in plant.axes_in_topological_order():
        if axis.parent_axis_id in out or (axis.structural_depth >= 1 and not axis.is_fork):
            out.add(axis.id)
    return out


def _axis_modulus(plant: Plant, profile: MorphologyProfile, axis: Axis) -> float:
    """The Young's modulus an axis bends with: its level's Stiffness
    (`BranchLevel.stiffness`; an inflorescence branch takes the level it grows
    on), else the plant's `stem_stiffness`."""
    level = _axis_level(plant, axis)
    own = profile.levels[level].stiffness if 0 <= level < len(profile.levels) else 0.0
    return max(own if own > 0.0 else profile.stem_stiffness, 1e-6)


def _leaf_load_mass(plant: Plant, profile: MorphologyProfile) -> dict:
    """`(axis id, phytomer index) -> mass` of the leaves components place
    themselves (`profile.leaf_loads`), at the nodes the placement puts them on
    -- the same anchors `instancing._mode_anchors` uses: every node but an axis's
    tip (node / internode), its tip, a branch's base, a stem's base -- foliage
    skipping inflorescence nodes; `random` spreads its total over every node by
    internode length. Empty without loads (byte-identical)."""
    loads = [ld for ld in getattr(profile, "leaf_loads", None) or ()
             if ld.mass > 0.0 and ld.count > 0.0]
    if not loads:
        return {}
    infl = _inflorescence_axes(plant)
    out: dict[tuple[int, int], float] = {}
    for load in loads:
        wanted = set(load.levels)
        spread: list[tuple[tuple[int, int], float]] = []
        for axis in plant.axes.values():
            if wanted and _axis_level(plant, axis) not in wanted:
                continue
            nodes = [i for i, ph in enumerate(axis.phytomers) if not ph.is_split]
            if not nodes:
                continue

            def leafy(i, axis=axis):
                return axis.id not in infl and not (0 <= axis.flowering_from <= i)

            if load.mode in ("node", "internode"):
                anchors = nodes[:-1]
            elif load.mode == "tip":
                anchors = nodes[-1:]
            elif load.mode == "intersection":
                anchors = nodes[:1] if axis.order >= 1 else []
            elif load.mode == "basal":
                anchors = nodes[:1] if axis.parent_axis_id < 0 else []
            elif load.mode == "random":
                spread += [((axis.id, i), float(axis.phytomers[i].length)) for i in nodes
                           if not load.skip_inflorescence or leafy(i)]
                continue
            else:
                anchors = []
            if load.skip_inflorescence:
                anchors = [i for i in anchors if leafy(i)]
            for i in anchors:
                out[(axis.id, i)] = out.get((axis.id, i), 0.0) + load.mass * load.count
        total = sum(length for _key, length in spread)
        if total > _EPS:
            for key, length in spread:
                out[key] = out.get(key, 0.0) + load.mass * load.count * length / total
    return out


def _supported(plant: Plant, profile: MorphologyProfile):
    """For every phytomer, the mass it carries beyond itself and that mass's
    centroid. Walked children-first so a parent can read its children's
    already-finished totals -- the same traversal order diameter.py uses. The
    leaves components place themselves weigh in at their nodes
    (`_leaf_load_mass`)."""
    mass: dict[tuple[int, int], float] = {}
    moment: dict[tuple[int, int], np.ndarray] = {}
    placed = _leaf_load_mass(plant, profile)

    children_at: dict[tuple[int, int], list[int]] = {}
    for axis in plant.axes.values():
        if axis.parent_axis_id >= 0 and axis.phytomers:
            children_at.setdefault(
                (axis.parent_axis_id, axis.parent_phytomer_index), []).append(axis.id)

    for axis in reversed(plant.axes_in_topological_order()):
        for i in range(len(axis.phytomers) - 1, -1, -1):
            phytomer = axis.phytomers[i]
            is_tip = i == len(axis.phytomers) - 1
            own = _phytomer_mass(phytomer, profile, axis.terminal_organ if is_tip else None)
            if placed:
                own += placed.get((axis.id, i), 0.0)
            total = own
            weighted = phytomer.tip_position * own

            if i + 1 < len(axis.phytomers):
                above = (axis.id, i + 1)
                total += mass[above]
                weighted = weighted + moment[above]
            for child_id in children_at.get((axis.id, i), ()):
                key = (child_id, 0)
                total += mass[key]
                weighted = weighted + moment[key]

            mass[(axis.id, i)] = total
            moment[(axis.id, i)] = weighted
    return mass, moment


def plant_center_of_mass(plant: Plant, profile: MorphologyProfile) -> np.ndarray | None:
    """The whole plant's centre of mass, in its own local frame.

    Uses the same mass model the bend does (`_supported`: wood as
    `pi r**2 L * wood_density`, a flower as `scale**3`, a leaf as `scale**2`),
    so this is exactly the point every axis is leaning toward -- which is what
    makes it worth drawing as a marker. Summed over the basal axes, whose base
    totals already fold in every branch and organ hanging off them. `None` on a
    plant with no mass yet.
    """
    mass, moment = _supported(plant, profile)
    total = 0.0
    weighted = np.zeros(3)
    for axis_id in plant.basal_axis_ids:
        key = (axis_id, 0)
        if key in mass:
            total += mass[key]
            weighted = weighted + moment[key]
    return weighted / total if total > _EPS else None


def stem_centers_of_mass(plant: Plant, profile: MorphologyProfile) -> list[np.ndarray]:
    """Each basal stem's own centre of mass -- everything it carries: its stem,
    branches, leaves and flowers -- in the plant's local frame, in basal-axis
    order. The same totals each stem's bend reads (`_axis_fall_direction`), so
    in a clump every stem shows the point it alone leans toward, where the
    plant's single centre (`plant_center_of_mass`) averages them."""
    mass, moment = _supported(plant, profile)
    out = []
    for axis_id in plant.basal_axis_ids:
        carried = mass.get((axis_id, 0), 0.0)
        if carried > _EPS:
            out.append(moment[(axis_id, 0)] / carried)
    return out


def _horizontal(vector: np.ndarray) -> np.ndarray | None:
    """`vector` flattened onto the ground plane and normalised, or None if it
    has no meaningful horizontal component. Below `_LEAN_MIN` the direction is
    floating-point noise and must not be trusted."""
    flat = np.array([vector[0], vector[1], 0.0])
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > _LEAN_MIN else None


def _flower_offset(axis: Axis) -> np.ndarray:
    """Where this axis's flowers sit, relative to its base.

    Needs resolved positions, which `build` guarantees -- it re-resolves right
    before `compute_radii`, so every organ is where it will be when the bend
    reads it. A terminal flower counts: on a solitary head it is the entire
    load.
    """
    # A set fruit counts with the flowers here (day path only): it is the same
    # reproductive load, heavier, and the branch should lean toward it.
    total = np.zeros(3)
    count = 0
    for phytomer in axis.phytomers:
        for organ in phytomer.organs:
            if organ.kind in ("flower", "fruit"):
                total += organ.position
                count += 1
    if axis.terminal_organ is not None and axis.terminal_organ.kind in ("flower", "fruit"):
        total += axis.terminal_organ.position
        count += 1
    if count == 0:
        return np.zeros(3)
    return total / count - axis.base_position


def _axis_lean(axis: Axis) -> np.ndarray:
    """The way the axis already leans, as the chord from its base to its tip.

    The chord, not the tip's own direction vector: a single node's heading is
    the noisiest thing on the axis, while base-to-tip is what anyone looking at
    the plant would call its lean.
    """
    if not axis.phytomers:
        return np.zeros(3)
    return axis.phytomers[-1].tip_position - axis.base_position


_COM_SIGNIF = 0.015
"""How far off-axis the carried-mass centroid must sit -- as a fraction of the
axis's own length -- before it is trusted as the toppling direction rather than
noise. Below it the load is effectively balanced and its horizontal offset is
floating-point junk, so the axis's existing lean holds the plane instead. This
is not a hard switch: the lean enters as a stabiliser of exactly this size (see
below), so a load crossing the threshold turns the plane smoothly rather than
flipping it."""


def _axis_fall_direction(axis: Axis, mass=None, moment=None) -> np.ndarray:
    """The one horizontal direction this whole axis topples toward.

    **The true centre of mass decides the side.** An axis is a cantilever, so it
    bends in the vertical plane through the resultant weight it carries -- which
    is gravity acting through the centroid of *everything beyond its base*: its
    own stem, leaves and flowers, **and the whole of every branch hanging off
    it**. That last part is the point of computing it from `_supported`'s totals
    rather than from this axis's own flowers alone: a trunk whose branches are
    all on one side has its mass on that side and must lean that way, and its
    own flowers (there may be none) cannot see the branches. Measured before
    this, a trunk leaned 90-160 degrees away from a genuinely one-sided load.

    **One per axis, not one per node.** `cross(direction, _DOWN)` is numerically
    unstable exactly where plants are interesting: on a near-vertical stem the
    product is tiny, so once normalised its direction is noise, and doing it per
    node made neighbouring nodes pick different planes and the tip's heading
    jump. Gravity acts in one direction, so an axis sags in one vertical plane;
    the centroid of the whole carried load names that plane once.

    **Balanced loads do not wobble, and that is why the centroid enters as an
    offset rather than a unit vector.** When leaves and wood are spread evenly
    they nearly cancel and the horizontal centroid offset shrinks toward zero,
    where its direction is junk -- the failure an earlier "lead with the total
    centroid" version hit. Here the axis's own lean is added as a stabiliser
    sized to `_COM_SIGNIF` of the axis length, so a real one-sided mass (offset
    well above that) sets the side outright while a near-balanced one leaves the
    lean holding the existing plane. A stem leans because of what it carries, so
    the two agree in practice; the stabiliser only decides the tie.
    """
    lean = _horizontal(_axis_lean(axis))
    if mass is not None and moment is not None:
        carried = mass.get((axis.id, 0), 0.0)
        if carried > _EPS:
            offset = moment[(axis.id, 0)] / carried - axis.base_position
            offset_h = np.array([offset[0], offset[1], 0.0])
            if float(np.linalg.norm(offset_h)) > _LEAN_MIN:
                axis_len = sum(float(p.length) for p in axis.phytomers)
                stabiliser = (lean * (_COM_SIGNIF * axis_len)
                              if lean is not None and axis_len > _EPS else 0.0)
                fall = _horizontal(offset_h + stabiliser)
                if fall is not None:
                    return fall
    # No mass info, or a load too balanced to trust: fall back to the axis's own
    # one-sided flowers plus its lean, the stable pre-centroid behaviour.
    flowers = _horizontal(_flower_offset(axis))
    if flowers is not None and lean is not None:
        combined = _horizontal(flowers + lean)
        return combined if combined is not None else flowers
    for candidate in (flowers, lean, _horizontal(axis.base_reference)):
        if candidate is not None:
            return candidate
    return np.array([1.0, 0.0, 0.0])


def _split_phytomers(plant: Plant, axis: Axis, parts_for: list[int]) -> None:
    """Split each of an axis's internodes into `parts_for[i]` pieces, turning
    smoothly across each split.

    One implementation for both subdivision passes, because the bookkeeping is
    the part that is easy to get silently wrong and there is no reason to have
    it twice. `flowering_from` is an index into the list being rebuilt, and
    child axes remember the index they hang from; both are remapped onto the
    **last** part of each split node, which is where the original node's tip --
    and so its organs, its buds and anything hanging off it -- actually was.

    Directions interpolate from the previous internode to this one across the
    parts, so a split *smooths* the joint rather than drawing the same facet
    with more vertices. The last part keeps the original direction exactly, so
    the joint with the next internode is untouched.
    """
    children_of: list = [c for c in plant.axes.values()
                         if c.parent_axis_id == axis.id]
    rebuilt: list = []
    remap: dict[int, int] = {}
    for i, phytomer in enumerate(axis.phytomers):
        parts = max(1, parts_for[i])
        before = (axis.phytomers[i - 1].direction if i > 0
                  else np.asarray(axis.base_direction, dtype=float))
        before = np.asarray(before, dtype=float)
        norm = float(np.linalg.norm(before))
        before = before / norm if norm > _EPS else phytomer.direction
        # The reference has to travel with the direction. Turning one and
        # copying the other leaves the frame skewed -- measured up to **24
        # degrees** off perpendicular on a pendant, against 0 on an
        # unsubdivided plant -- and `radial_direction` then places organs off
        # that skewed frame, differently at every node. It showed up as some
        # flowers barely responding to Pedicel Angle and others inverting.
        ref_before = (axis.phytomers[i - 1].reference if i > 0
                      else np.asarray(axis.base_reference, dtype=float))
        for k in range(parts):
            last = k == parts - 1
            direction = _slerp(before, phytomer.direction, (k + 1) / parts)
            rebuilt.append(Phytomer(
                index=0,                       # renumbered below
                birth_step=phytomer.birth_step,
                final_length=phytomer.final_length / parts,
                direction=direction,
                # Transported from the frame `before` carried, so the last
                # part lands back on this phytomer's own reference exactly.
                reference=parallel_transport(ref_before, before, direction),
                factor=phytomer.factor,
                organs=phytomer.organs if last else [],
                buds=phytomer.buds if last else [],
                is_split=not last,
            ))
            # Radius is set by `compute_radii` BEFORE the pre-gravity splits (so
            # it reads 0 here and is refilled later) but AFTER the post-gravity
            # one, where the new parts must inherit it or the tube collapses to
            # a thread. Copying the parent's is right for a short split; the
            # bend-display values ride along so the debug views stay continuous.
            rebuilt[-1].radius = phytomer.radius
            rebuilt[-1].debug_curvature = phytomer.debug_curvature
            rebuilt[-1].debug_load = phytomer.debug_load
            rebuilt[-1].debug_stiffness = phytomer.debug_stiffness
            rebuilt[-1].debug_moment = phytomer.debug_moment
            rebuilt[-1].lying = phytomer.lying
        remap[i] = len(rebuilt) - 1

    for j, phytomer in enumerate(rebuilt):
        phytomer.index = j
    axis.phytomers = rebuilt
    if axis.flowering_from >= 0:
        axis.flowering_from = remap.get(axis.flowering_from, axis.flowering_from)
    for child in children_of:
        child.parent_phytomer_index = remap.get(
            child.parent_phytomer_index, child.parent_phytomer_index)


_MAX_ARC_FACET_DEG = 7.0
"""The biggest turn allowed between two nodes of an inflorescence that arced as
it grew. Above this the internode is split until it is met."""


def smooth_oriented_arcs(plant: Plant, profile: MorphologyProfile) -> None:
    """Add nodes wherever an inflorescence's own growth arc turns too sharply
    for its resolution.

    The counterpart to `smooth_bent_arcs`: both split only the joints that turn
    too sharply and leave straight runs alone, but at opposite ends of the
    pipeline. This one smooths the arc an inflorescence took at *growth* time
    (before gravity), where the turn at every joint is already known, so the
    extra nodes land exactly where the bend is and nowhere else; `smooth_bent_arcs`
    does the same for the *gravity* bend, after it.

    That precision matters for cost. Subdividing a pendant end to end to smooth
    its first few joints took it from 15 nodes to 91; raising the shared Bend
    Resolution default instead moved node counts on twelve golden fixtures,
    because it reaches every cyme, panicle and compound form as well. This
    reaches only axes that were aimed, and only their bending joints, so it needs
    no control and changes nothing else.

    Runs before the pipe model and the bend, so the extra nodes are ordinary
    phytomers by the time anything reads them.
    """
    for axis in plant.axes.values():
        if axis.orientation_target_deg < 0.0 or len(axis.phytomers) < 2:
            continue
        parts_for = []
        for i, phytomer in enumerate(axis.phytomers):
            before = (axis.phytomers[i - 1].direction if i > 0
                      else np.asarray(axis.base_direction, dtype=float))
            norm = float(np.linalg.norm(before))
            if norm <= _EPS:
                parts_for.append(1)
                continue
            turn = float(np.degrees(np.arccos(np.clip(
                float(np.dot(before / norm, phytomer.direction)), -1.0, 1.0))))
            parts_for.append(min(8, max(1, int(np.ceil(turn / _MAX_ARC_FACET_DEG)))))
        if all(k == 1 for k in parts_for):
            continue
        _split_phytomers(plant, axis, parts_for)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Unit direction a fraction `t` of the way from `a` to `b`, along the arc
    rather than through the chord, so every step turns by the same angle."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1e-6:
        return b.copy()
    sin = float(np.sin(angle))
    return (a * float(np.sin((1.0 - t) * angle)) + b * float(np.sin(t * angle))) / sin


_NEAR_VERTICAL_SIN = float(np.sin(np.radians(10.0)))
_MAX_BRANCH_BEND_DEG = 90.0
"""What Branch Bend 1.0 means in degrees, base to tip."""


def apply_branch_bend(plant: Plant, profile: MorphologyProfile) -> None:
    """Arch each inflorescence branch by `branch_bend`, as a shape.

    **A post-growth pass, and moving it here from `_add_phytomer` is what makes
    the number on the panel true.** Applied during growth it was fought by
    gravitropism at every node -- each node was steered back toward its
    setpoint and then nudged by one bend increment -- so what the branch
    finally held depended on how those two happened to balance, which in turn
    depended on its node count. Measured over the fix, on a compound corymb
    branch: a stated 18 degrees delivered **54.5**, and 0.6 and 1.0 both
    delivered **87.8** -- over-strong at the bottom of the range and completely
    saturated over its whole upper half, with the whole turn landing on single
    joints of up to 53.9 degrees.

    Run **before** `apply_gravity`, so the branch sags from the shape it holds
    rather than the two fighting for the same joints; its arch is then spread
    over the nodes the axis actually has by `smooth_bent_arcs`, so raising the
    resolution genuinely smooths it instead of leaving the corner where it was.

    Shape rather than gravity, exactly as `pedicel_bend` is: positive lifts,
    negative sags, and nothing turns past vertical either way -- an unclamped
    arch sails over the top and comes down the far side. The pivot is the
    node's own vertical plane with the axis's as a near-vertical fallback,
    which is the hybrid this module already settled on for the same
    degeneracy: `cross(direction, _UP)` vanishes as a branch nears vertical and
    its *direction* becomes floating-point noise, so the arch would swing
    sideways instead of continuing up.
    """
    for axis in plant.axes.values():
        if axis.structural_depth < 1 or abs(axis.inflorescence_bend) <= _EPS:
            continue
        joints = len(axis.phytomers) - 1
        if joints < 1:
            continue
        per = np.radians(axis.inflorescence_bend * _MAX_BRANCH_BEND_DEG / joints)
        fallback = _cross(axis.base_direction, _UP)
        norm = float(np.linalg.norm(fallback))
        fallback = fallback / norm if norm > _EPS else None
        applied = 0.0
        for i in range(1, len(axis.phytomers)):
            phytomer = axis.phytomers[i]
            direction = np.array(phytomer.direction, dtype=float)
            pivot = _cross(direction, _UP)
            norm = float(np.linalg.norm(pivot))
            if norm <= _NEAR_VERTICAL_SIN:
                if fallback is None:
                    continue
                pivot, norm = fallback, 1.0
            else:
                pivot = pivot / norm
            applied += per
            # Straight up is where arching up runs out, and straight down where
            # arching down does -- the same stop `_turn_to_lowest` applies to a
            # sag, stated for a rotation whose pivot is horizontal and so
            # changes elevation by exactly the angle turned.
            elevation = float(np.arcsin(np.clip(float(np.dot(direction, _UP)), -1.0, 1.0)))
            half = np.pi / 2.0
            turn = float(np.clip(applied, -(half + elevation), half - elevation))
            if abs(turn) <= _EPS:
                continue
            rotated = rotate_matrix(pivot, np.degrees(turn)) @ direction
            norm = float(np.linalg.norm(rotated))
            if norm <= _EPS:
                continue
            rotated = rotated / norm
            phytomer.reference = parallel_transport(
                np.array(phytomer.reference, dtype=float), direction, rotated)
            phytomer.direction = rotated


def apply_autotropism(plant: Plant, profile: MorphologyProfile) -> None:
    """Straighten each stem back toward its own gravitropic setpoint as its
    nodes age -- the post-growth half of circumnutation.

    Gravitropism (`development._apply_gravitropism`) corrects a node once, at
    birth, and never revisits it, so a coherent circumnutation wander survives
    as a *uniform* helix all the way to the tip -- measured base and tip at the
    same ~1.5 degrees on a default plant. Real stems do not keep that: older
    tissue stiffens and straightens (autotropism, Firn & Digby), which leaves
    the mature stem tidy while the growing tip still nods -- the look a growth
    timelapse shows. Modelled as an age-scaled extra correction: a node that has
    survived `age` steps has had a fraction ``1 - (1 - k)**age`` of its
    tilt-from-setpoint removed, so the oldest nodes go nearly straight
    (measured 0.03 degrees) and the newest few keep their full nod (10+).

    A post-growth deformation like the gravity bend, and deliberately the
    opposite sign: gravity rotates a node *down* under load, this rotates it
    *toward* the up-setpoint. It only ever shrinks the deviation from setpoint,
    never grows it, so it cannot introduce a bend the growth did not have.

    Vegetative stems and forks only -- an inflorescence axis is aimed by its
    orientation mode and its arcs are deliberate, so straightening it toward a
    setpoint would fight that. `parallel_transport` carries each node's frame
    with its rotated direction, the same discipline `apply_branch_bend` keeps.
    """
    k = profile.autotropism
    if k <= 0.0:
        return
    # A node's age on the plant's own clock: the fractional step on the day
    # path (a timelapse), so each node straightens smoothly as it ages instead
    # of every node snapping at once each whole step (measured: a 5.6 mm
    # sideways jump per step against 0.37 mm per frame between). The age path
    # (no fractional step) is the integer step, byte-identical.
    now = plant.step if plant.continuous_step < 0.0 else plant.continuous_step
    for axis in plant.axes.values():
        if axis.structural_depth >= 0 and not axis.is_fork:
            continue  # a deliberately-oriented inflorescence axis
        setpoint = axis.setpoint_tilt_deg
        for phytomer in axis.phytomers:
            age = phytomer.age_steps(now)
            if age <= 0:
                continue  # the newest node keeps its full nod
            direction = np.array(phytomer.direction, dtype=float)
            pivot = _cross(direction, _UP)
            norm = float(np.linalg.norm(pivot))
            if norm < _EPS:
                continue  # already vertical: nothing to straighten
            tilt = float(np.degrees(np.arccos(
                np.clip(float(np.dot(direction, _UP)), -1.0, 1.0))))
            frac = 1.0 - (1.0 - k) ** age
            delta = (tilt - setpoint) * frac
            if abs(delta) < _EPS:
                continue
            rotated = rotate_about(direction, pivot / norm, delta)
            rnorm = float(np.linalg.norm(rotated))
            if rnorm < _EPS:
                continue
            rotated = rotated / rnorm
            phytomer.reference = parallel_transport(
                np.array(phytomer.reference, dtype=float), direction, rotated)
            phytomer.direction = rotated


def _bend_axis(axis_fall: np.ndarray) -> np.ndarray:
    """The axis every segment of one stem rotates about to fall toward the
    ground: horizontal, square to the direction the stem topples in.

    **One plane for the whole axis, and the orientation problem was this.** An
    earlier version gave each segment its *own* plane through straight down, so
    that it converged on vertical rather than stalling on a cone. Measured, that
    is where all the remaining heading instability lived: as more nodes came into
    the drooping region, their planes fanned out from the axis's own by a mean of
    8.6 degrees at Peduncle Reach 0 rising to 40.8 at 0.5, with individual nodes
    as far as 180 -- bending in the exactly opposite direction. The stem's own
    fall direction stayed pinned at 330.1 degrees through that entire sweep while
    the tip came out anywhere from 305 to 28.

    The cone it was avoiding is real -- rotation about a fixed axis cannot change
    the component *along* that axis, so a segment leaning out of this plane
    cannot reach exactly straight down, and `_turn_to_lowest` stops it at the
    lowest point it can reach. What makes that a non-issue now is *which* plane
    this is: `_axis_fall_direction` folds the stem's own lean in, so the plane
    already contains the lean and the out-of-plane component is small. The old
    45-degree stall was measured back when the plane came from the total load
    centroid, which is very nearly cancelled by leaves and wood and so pointed
    in a direction unrelated to the lean -- put the plane where the stem is
    actually falling and the cone stops being in the way.

    Gravity acts in one direction; a stem sags in one vertical plane. This is
    that plane.
    """
    perpendicular = _cross(_UP, axis_fall)
    norm = float(np.linalg.norm(perpendicular))
    if norm <= _EPS:
        return np.array([0.0, 1.0, 0.0])
    return perpendicular / norm


def _turn_to_lowest(direction: np.ndarray, bend_axis: np.ndarray) -> float:
    """How far `direction` may rotate about `bend_axis` before it is pointing as
    close to straight down as that axis allows, in degrees.

    **Not the angle between `direction` and down**, which is the obvious answer
    and is wrong: rotation about a fixed axis cannot change the component
    *along* that axis, so a segment already leaning out of the bend plane
    cannot reach vertical at all. Clamping to the raw angle-to-down let such a
    segment turn past its own lowest point and start climbing the far side --
    reported as the peduncle curling *up*.

    What can be reached is the point where the part of `direction` in the bend
    plane lines up with down. `bend_axis` is horizontal by construction, so
    down lies wholly in that plane and this is just the signed angle from one
    to the other, taken the way the rotation actually goes.
    """
    in_plane = direction - bend_axis * float(np.dot(direction, bend_axis))
    norm = float(np.linalg.norm(in_plane))
    if norm <= _EPS:
        return 0.0        # pointing along the axis: rotating changes nothing
    in_plane = in_plane / norm
    if float(np.dot(in_plane, _UP)) >= _UPRIGHT_COS:
        # Essentially upright, where the bottom is half a turn away on either
        # side and the signed angle below cannot choose between them -- it
        # returns +180 or -180 on floating-point noise, and on the minus side
        # the segment reads as "already fallen" and refuses to move. That is
        # not a corner case, it is every straight stem: it left a peduncle
        # whose whole turn landed on one vertical node completely unbent.
        # An upright stem under an off-centre load falls toward that load,
        # which is the direction `bend_axis` was built from, so the full turn
        # is the right answer.
        return 180.0
    # Signed, in (-180, 180]: positive means the bottom lies forward along the
    # rotation, negative means it lies behind. Gravity only turns one way here,
    # so a segment already past its lowest point simply stops rather than being
    # hauled the long way round -- which is what left a single node reading 60
    # degrees between two neighbours at 179.
    return max(0.0, _signed_to_lowest(in_plane, bend_axis))


def _signed_to_lowest(in_plane: np.ndarray, bend_axis: np.ndarray) -> float:
    """The signed angle about `bend_axis` from a unit in-plane direction to
    straight down, in (-180, 180]: negative when it is already past its lowest
    point."""
    return float(np.degrees(np.arctan2(
        float(np.dot(_cross(in_plane, _DOWN), bend_axis)),
        float(np.dot(in_plane, _DOWN)))))


def _past_lowest(direction: np.ndarray, bend_axis: np.ndarray) -> float:
    """How far past its lowest point about `bend_axis` a direction is, in
    degrees (0 when it is not, or is upright, or lies along the axis)."""
    in_plane = direction - bend_axis * float(np.dot(direction, bend_axis))
    norm = float(np.linalg.norm(in_plane))
    if norm <= _EPS or float(np.dot(in_plane / norm, _UP)) >= _UPRIGHT_COS:
        return 0.0
    return max(0.0, -_signed_to_lowest(in_plane / norm, bend_axis))


def apply_gravity(plant: Plant, profile: MorphologyProfile, seed: int = 0,
                  terrain=None) -> None:
    """Bend the finished structure under its own weight -- one cantilever beam,
    every axis, in SI units.

    Requires radii (`diameter.compute_radii`) and resolved positions, so the
    call order is grow -> compute_radii -> apply_gravity -> smooth_bent_arcs ->
    resolve_geometry.

    The model is the Euler-Bernoulli cantilever. At each node the bending moment
    is the weight carried beyond it (`m_carried * g`) times the horizontal
    distance to that weight's centroid (`arm`); the section resists with its
    second moment of area `I = pi r**4 / 4`. The angle an internode turns
    through is the curvature `M / (E I)` times its own length::

        theta = 4 * g * m_carried * arm * L / (pi * E_local * r**4)

    **This is the whole of gravity now.** The prescribed-angle terms this used
    to carry -- `droop` as a unitless multiplier, Peduncle Droop and Branch
    Droop as half-circle turns shared across a flowering region, all capped at a
    flat 14 degrees a node -- are gone. A flowering stretch bends more because
    its flowers are heavy and its stem is thin (`r**4`), a branch bends because
    it is a loaded cantilever like any other, and *where* a stem gives way is
    decided by its taper and its stiffness profile, not by a `rigid_base`
    fraction. `stem_stiffness` is a literal Young's modulus in pascals;
    `wood_density`, `leaf_weight` and `flower_weight` are real masses (kg).

    `E_local` is that modulus scaled by two things: `stiffness_profile` read
    base-to-tip, the lignification gradient (woody and stiff at the base, green
    and soft at the tip), and a per-axis `stiffness_variation` so a clump of
    stems differs. Both default to a flat 1.0.

    The turn is capped two ways, neither the old flat per-node ceiling: a
    per-radius **curvature cap** (`_KINK_RATIO`) keeps the physical bend from
    tightening past where a real tube kinks, and the stop against straight down
    (`_turn_to_lowest`) keeps this single pass from overshooting vertical.
    Drawing a hard turn as a smooth arc rather than a facet is
    `smooth_bent_arcs`' job, run straight after this.

    **Single pass, and load-stepping was measured and dropped.** Applying the
    weight in four increments -- repositioning the skeleton between each so the
    moment arm shrinks as the stem bends -- is the textbook large-deflection
    fix, and it was built and measured. It changed the worst raw internode turn
    on the generic from 34.7 to 34.1 degrees and on a compound umbel's rays from
    172.5 to 169.1, for four times the cost: the steep bends here are genuine
    equilibria (a limp 1.2 mm ray really does hang near vertical), not the
    undeformed-arm overshoot load-stepping corrects. What those folds actually
    need is resolution, which is `smooth_bent_arcs`' job, not a better
    integrator.
    """
    mass, moment = _supported(plant, profile)
    E = max(profile.stem_stiffness, 1e-6)
    variation = max(profile.stiffness_variation, 0.0)

    # Rotation accumulated at each attachment point, so a child axis inherits
    # every bend its parent underwent below the branch point. Identity for
    # basal axes.
    inherited: dict[int, np.ndarray] = {}
    # **The ground carries what lies on it.** Each node's bent tip is tracked
    # as the pass goes, and a segment that would end below the plant's own
    # base lifts just enough to land on it -- so a stem that reaches the
    # ground lies along it instead of hanging through it (a clubmoss runner,
    # a strawberry stolon). Nothing changes on a plant that never gets there.
    # The ground is the plant's base level, or `terrain` (an
    # `environment.Ground`, the plant's Ground Object) where it has a surface.
    ground_at = _ground_reader(plant, terrain)
    landed: dict[tuple[int, int], np.ndarray] = {}
    # **A stem's foot is on the ground.** Stems grow from the plant's own
    # level, so on a terrain each basal stem starts where the surface is under
    # it -- else one on a slope began inside the ground or in the air beside
    # it (a runner 6 mm up never touched down). The resolve after this carries
    # the whole stem with its foot.
    if terrain is not None:
        for axis_id in plant.basal_axis_ids:
            axis = plant.axes[axis_id]
            z = terrain.height(float(axis.base_position[0]), float(axis.base_position[1]))
            if z is not None:
                axis.base_position = np.array(
                    [float(axis.base_position[0]), float(axis.base_position[1]), z])

    for axis in plant.axes_in_topological_order():
        rotation = inherited.get(axis.id, np.eye(3))
        pos = landed.get((axis.parent_axis_id, axis.parent_phytomer_index))
        pos = (np.array(axis.base_position, dtype=float) if pos is None else pos.copy())
        # One stiffness draw per axis, not per node: varying it along the length
        # is what a ragged stem looks like, and a clump of stems differing from
        # each other is the variation anyone wants -- the same call
        # `_level_radii` makes for a level's width variation.
        axis_stiff = 1.0 + variation * _signed(seed, _T_STIFFNESS, axis.id)
        E = _axis_modulus(plant, profile, axis)   # its level's Stiffness
        # Gravity acts in one direction, so a whole axis sags in one vertical
        # plane. `_axis_fall_direction` folds the axis's own lean and its
        # flowers' offset into that heading; `_bend_axis` is the horizontal
        # pivot square to it. Doing this per node was measured to fan the planes
        # out and make the tip's compass heading jump -- see both functions.
        bend_axis = _bend_axis(_axis_fall_direction(axis, mass, moment))
        last_index = max(len(axis.phytomers) - 1, 1)

        for i, phytomer in enumerate(axis.phytomers):
            # Everything below this node has already bent; carry that first.
            grown_past = _past_lowest(phytomer.direction, bend_axis)
            grown = phytomer.direction
            phytomer.direction = rotation @ phytomer.direction
            phytomer.reference = rotation @ phytomer.reference
            # **A hanging part swings back to hanging.** Carried rigidly, a
            # segment already pointing down is swung past its lowest point by
            # the sag below it and climbs the far side -- a pendulous raceme on
            # a sagging branch curled up into a loop (measured: a region that
            # grew to 180 degrees ended at 84). It is free to hang, so the
            # overshoot the carry added is undone (what it grew past down with
            # stays), for this node and everything beyond it -- never more than
            # the carry turned it (measured in the bend plane, a segment lying
            # nearly along the pivot read swings of 147 degrees from a few).
            carried = float(np.degrees(np.arccos(np.clip(
                float(np.dot(grown, phytomer.direction))
                / max(float(np.linalg.norm(grown) * np.linalg.norm(phytomer.direction)), _EPS),
                -1.0, 1.0))))
            back = min(_past_lowest(phytomer.direction, bend_axis) - grown_past, carried)
            if back > 0.0:
                rotation = rotate_matrix(bend_axis, -back) @ rotation
                phytomer.direction = rotate_about(phytomer.direction, bend_axis, -back)
                phytomer.reference = rotate_about(phytomer.reference, bend_axis, -back)

            carried = mass[(axis.id, i)]
            centroid = moment[(axis.id, i)] / max(carried, _EPS)
            delta = centroid - phytomer.base_position
            # The moment arm is the horizontal offset to the carried mass, which
            # vanishes on a dead-straight vertical stem. See `_LOAD_ECCENTRICITY`.
            arm = max(float(np.linalg.norm(delta[:2])),
                      _LOAD_ECCENTRICITY * float(np.linalg.norm(delta)))
            radius = max(phytomer.radius, 1e-6)
            length = max(phytomer.length, 0.0)

            # Local modulus: base-to-tip stiffness profile x per-axis variance,
            # floored so a curve dragged to zero cannot divide by zero.
            u = i / last_index
            e_local = E * max(float(profile.stiffness_profile(u)), 0.02) * max(axis_stiff, 0.02)

            # The complete Euler-Bernoulli turn for this internode, in degrees.
            bend = float(np.degrees(
                4.0 * _G * carried * arm * length / (np.pi * e_local * radius ** 4)))

            # Curvature cap: no tighter than a real tube bends before it kinks.
            # Radius-aware, so it is loose on a thin twig (which really does bend
            # tightly) and tight on a thick stem (which does not).
            bend = min(bend, float(np.degrees(length / (_KINK_RATIO * radius))))
            # Nothing turns past straight down. A vertical segment has zero
            # moment arm about `bend_axis` and so stops bending on its own.
            bend = min(bend, _turn_to_lowest(phytomer.direction, bend_axis))

            # Beam readouts for the four gravity debug views (display-only). The
            # curvature is the *realized* turn per length after the cap, so a
            # capped joint reads lower than its moment; the load and modulus are
            # what produced it; the moment is `g * load * arm`.
            phytomer.debug_curvature = bend / length if length > _EPS else 0.0
            phytomer.debug_load = carried
            phytomer.debug_stiffness = e_local
            phytomer.debug_moment = _G * carried * arm
            if bend > 0.0:
                rotation = rotate_matrix(bend_axis, bend) @ rotation
                phytomer.direction = rotate_about(phytomer.direction, bend_axis, bend)
                phytomer.reference = rotate_about(phytomer.reference, bend_axis, bend)

            # The ground carries the stem's underside: its centre lands a
            # radius up, so a lying stem rests on the ground, not half in it.
            # Landed where the ground is under the LANDED end, so the terrain
            # is re-read after each lift (a sagging segment ends nearer its
            # start, where a hill is lower); flat ground settles at once.
            # **A stem lying on the ground roots there** (a clubmoss runner, a
            # strawberry stolon: rooting at frequent intervals), so it follows
            # the ground down a falling slope as well as up a rising one,
            # instead of standing off it as a stiff cantilever. Lying: it
            # starts on the ground (underside within a radius of it) and rises
            # no more than a radius over its length off the line it starts
            # from -- the segment before it, which lies there too (so a stem
            # tilted up a hill does not read as rising past the crest), or
            # level for an axis's first. One rising off the ground (an upright,
            # a decumbent stem curving up) is only kept out of it, as before.
            before = axis.phytomers[i - 1].direction if i else axis.base_direction
            line = float(before[2]) if i and axis.phytomers[i - 1].lying else 0.0
            rooted = (float(pos[2]) - ground_at(pos) <= 2.0 * radius
                      and (float(phytomer.direction[2]) - line) * length <= radius)
            phytomer.lying = rooted
            lift = _land(phytomer.direction, pos, length, radius, ground_at, before,
                         follow=rooted)
            carry = rotation
            if lift is not None:
                if rooted:
                    # What grows off a lying stem takes it as lying flat, not
                    # tilted with the ground (an upright stays upright on a
                    # hillside); a lift off a stem that is not lying is its
                    # own sag undone, which its branches shared.
                    flat = _ground_lift(phytomer.direction, pos, length,
                                        float(pos[2]), before, follow=True)
                    carry = rotation if flat is None else flat @ rotation
                rotation = lift @ rotation
                phytomer.direction = lift @ phytomer.direction
                phytomer.reference = lift @ phytomer.reference
                if not rooted:
                    carry = rotation
            pos = pos + phytomer.direction * length
            landed[(axis.id, i)] = pos

            for child in plant.children_of(axis.id):
                if child.parent_phytomer_index == i:
                    inherited[child.id] = carry

_GROUND_ITERATIONS = 4
"""Lifts per segment onto uneven ground: each re-reads the terrain under the
end the last one landed, converging on the surface (flat ground needs one)."""

_GROUND_TOLERANCE = 1e-6
"""Metres a rooted stem's segment may end above the ground and count as lying
on it -- so the landing stops once it has settled, not on float noise."""


def _ground_reader(plant: Plant, terrain=None):
    """The ground height at a plant-local point: `terrain` (an
    `environment.Ground`, the plant's Ground Object) where it has a surface,
    else the flat ground at the plant's base level."""
    base_level = min((float(plant.axes[i].base_position[2]) for i in plant.basal_axis_ids),
                     default=0.0)

    def ground_at(pos) -> float:
        z = None if terrain is None else terrain.height(float(pos[0]), float(pos[1]))
        return base_level if z is None else z
    return ground_at


def _land(direction, pos, length, radius, ground_at, before,
          follow: bool = False) -> np.ndarray | None:
    """The turn that lands one segment from `pos` with its underside on the
    ground (its centre a radius up), or None when it ends above it -- or,
    with `follow` (a stem lying on the ground, rooted there), only when it
    ends on it: it is lowered onto a ground that falls away too. Landed
    where the ground is under the LANDED end, so the terrain is re-read after
    each turn: a sagging segment ends nearer its start, where a hill is lower.
    `before` (the previous segment's heading) gives a segment hanging straight
    down a compass heading to lie along."""
    lift = None
    for _ in range(_GROUND_ITERATIONS):
        d = direction if lift is None else lift @ direction
        step = _ground_lift(d, pos, length, ground_at(pos + d * length) + radius, before,
                            follow)
        if step is None:
            break
        lift = step if lift is None else step @ lift
    return lift


def settle_on_ground(plant: Plant, terrain=None) -> bool:
    """Land, once more, whatever the arc smoothing set into the ground.

    `smooth_bent_arcs` draws a sharp lift as an arc, which runs inside its
    chord and so ends lower -- about a stem radius per lifted joint, so on a
    slope every node of a lying stem sank half into the terrain. The same
    landing as `apply_gravity`'s, on the resolved polyline (run after a
    resolve: the arc's parts have no length before one); a node above the
    ground is untouched (and with it every plant that never reaches it),
    unless the gravity pass found it lying there (`Phytomer.lying`): a rooted
    stem is laid back along the ground, down a falling one too.
    True if it moved anything, so the caller resolves again only then."""
    ground_at = _ground_reader(plant, terrain)
    landed: dict[tuple[int, int], np.ndarray] = {}
    moved = False
    for axis in plant.axes_in_topological_order():
        pos = landed.get((axis.parent_axis_id, axis.parent_phytomer_index))
        pos = np.array(axis.base_position, dtype=float) if pos is None else pos.copy()
        for i, phytomer in enumerate(axis.phytomers):
            # Each segment on its own: the ones after it already lie as the
            # gravity pass laid them, so only their start moves with it (a
            # carried turn tilted a whole runner up the hill, 18 cm by its end).
            length = max(phytomer.length, 0.0)
            # A part the gravity pass laid on the ground (`lying`) follows it
            # down too while it starts there; read from the flag, not from its
            # heading now -- the arc parts of a steep climb rise, but the stem
            # still lies.
            radius = max(phytomer.radius, 1e-6)
            follow = phytomer.lying and float(pos[2]) - ground_at(pos) <= 2.0 * radius
            lift = _land(phytomer.direction, pos, length, radius, ground_at,
                         axis.phytomers[i - 1].direction if i else axis.base_direction,
                         follow=follow)
            if lift is not None:
                moved = True
                phytomer.direction = lift @ phytomer.direction
                phytomer.reference = lift @ phytomer.reference
            pos = pos + phytomer.direction * length
            landed[(axis.id, i)] = pos
    return moved


def _ground_lift(direction: np.ndarray, pos: np.ndarray, length: float,
                 ground: float, before=None, follow: bool = False) -> np.ndarray | None:
    """The turn that lands a segment from `pos` on the ground, or None when it
    ends above it (with `follow`, a rooted stem's: only when it ends on it):
    the same compass heading (or `before`'s, for a segment hanging straight
    down), its slope eased to exactly reach `ground`."""
    if length <= _EPS:
        return None
    end = float(pos[2] + direction[2] * length)
    if end >= ground and (not follow or end - ground <= _GROUND_TOLERANCE):
        return None
    rise = float(np.clip((ground - pos[2]) / length, -1.0, 1.0))
    flat = np.array([direction[0], direction[1], 0.0])
    norm = float(np.linalg.norm(flat))
    if norm <= 1e-3 and before is not None:
        flat = np.array([before[0], before[1], 0.0])
        norm = float(np.linalg.norm(flat))
    if norm <= _EPS:
        return None                     # no heading anywhere to keep
    flat /= norm
    target = flat * np.sqrt(max(0.0, 1.0 - rise * rise)) + np.array([0.0, 0.0, rise])
    axis = _cross(direction, target)
    sin = float(np.linalg.norm(axis))
    if sin <= _EPS:
        return None
    angle = float(np.degrees(np.arctan2(sin, float(np.dot(direction, target)))))
    return rotate_matrix(axis / sin, angle)


_SUBDIV_MAX_GROWTH = 3.0
"""How many nodes `smooth_bent_arcs` may add, as a fraction of the grown count.
A hard plant-wide ceiling so a dense form cannot explode; spent worst-joint-
first, so the budget goes where the bend is sharpest and a plant too dense to
fully smooth degrades gracefully rather than favouring whichever axis was
walked first.

Set generous on measurement, because the added nodes are cheap -- they carry no
organs or pedicels (those stay on the split's last part), so they are position-
only and barely touch resolve. Most shipped species smooth completely inside
+50% (generic +12%, laburnum +23%, foxtail +49%); the exception is a heavy
terminal flower on a wiry stem -- the nodding anemone -- which concentrates its
whole nod on its top one or two internodes and needs ~+300% to draw that as a
curve rather than a facet, still under a quarter-second to build. The genuinely
pathological case is a synthetic compound umbel on the generic profile, whose
two-node rays hang near vertical: it is capped here rather than smoothed, which
is an authoring artifact (a real umbel species gives its rays more nodes) not a
normal plant. A stem that wants smoothing this hard really wants more phytomers
to bend over, which is a species-authoring choice; this is the fallback."""


def smooth_bent_arcs(plant: Plant, threshold: float = _MAX_ARC_FACET_DEG,
                     max_growth: float = _SUBDIV_MAX_GROWTH) -> None:
    """Split, after the bend, only the joints too sharp to read as a curve --
    those and no others, worst-first, under a plant-wide node budget.

    Earlier builds guessed *before* the bend which stretch would need resolution
    and split it uniformly by a control. The bend is physical now and known at
    every joint the moment `apply_gravity` has run, so the extra nodes can go
    exactly where a joint exceeds `threshold` and nowhere else -- no control, and
    no straight run touched.

    Worst-first under a budget (`max_growth` of the grown node count): a joint
    is handed one more part at a time, always the joint whose current facet is
    largest, until every facet is under `threshold` or the budget is spent. So a
    solitary flower's two-node hook and a compound umbel's hundreds of rays both
    smooth as far as their share allows, and neither runs the node count away.

    Reuses `_split_phytomers`, which slerps the bent directions across each new
    part -- so a 40-degree joint becomes a six-segment arc of the same total
    turn, not the same facet drawn six times.
    """
    import heapq
    axes = [a for a in plant.axes.values() if a.phytomers]
    total = sum(len(a.phytomers) for a in axes)
    if total == 0:
        return
    budget = int(max_growth * total)

    parts: dict[tuple[int, int], int] = {}
    turn_of: dict[tuple[int, int], float] = {}
    heap: list[tuple[float, int, int]] = []
    for axis in axes:
        for i, phytomer in enumerate(axis.phytomers):
            before = (axis.phytomers[i - 1].direction if i > 0
                      else np.asarray(axis.base_direction, dtype=float))
            norm = float(np.linalg.norm(before))
            if norm <= _EPS:
                continue
            before = before / norm
            turn = float(np.degrees(np.arccos(np.clip(
                float(np.dot(before, phytomer.direction)), -1.0, 1.0))))
            if turn <= threshold:
                continue
            key = (axis.id, i)
            parts[key] = 1
            turn_of[key] = turn
            heapq.heappush(heap, (-turn, axis.id, i))

    while heap and budget > 0:
        neg_per_part, aid, i = heapq.heappop(heap)
        if -neg_per_part <= threshold:
            break                       # every remaining facet is already fine
        key = (aid, i)
        parts[key] += 1
        budget -= 1
        per_part = turn_of[key] / parts[key]
        if per_part > threshold:
            heapq.heappush(heap, (-per_part, aid, i))

    for axis in axes:
        pf = [parts.get((axis.id, i), 1) for i in range(len(axis.phytomers))]
        if any(k > 1 for k in pf):
            _split_phytomers(plant, axis, pf)


def _set_pedicel_droop(organ, node_radius: float, E: float, profile: MorphologyProfile) -> None:
    """Store the physical sag of one flower's own stalk on `pedicel_droop_deg`.

    The closed-form tip-loaded cantilever: a pedicel is a short near-uniform
    section with the flower's weight `W` at its far end, so its tip slope is
    `theta = W L**2 / (2 E I)`, `I = pi r**4 / 4`. That angle is *exactly* what
    `model._sag`'s `t(2 - t)` profile already reaches at the tip -- so the whole
    of routing a pedicel through the beam is to compute this number physically
    instead of from a `pedicel_droop` dial, and hand it to `_sag` unchanged.

    Negative because gravity only sags (world down); `_sag`'s sign convention is
    negative-is-down. The pedicel's own weight along its length is left out: a
    stalk carrying a flower several times its own mass makes that a rounding
    error, and both length and radius already enter through `L` and `I`.
    """
    if (organ.kind not in ("flower", "fruit")
            or organ.pedicel_length <= 0.0 or organ.catkin_length > 0.0):
        return
    r_ped = max(node_radius * organ.pedicel_width, 1e-6)
    second_moment = np.pi * r_ped ** 4 / 4.0
    load = _organ_mass(organ, profile) * _G
    theta = float(np.degrees(load * organ.pedicel_length ** 2 / (2.0 * E * second_moment)))
    organ.pedicel_droop_deg = -min(theta, 180.0)


def _set_catkin_droop(organ, node_radius: float, E: float, profile: MorphologyProfile) -> None:
    """Store the physical sag of a catkin bunch's shared stalk on `catkin_droop`.

    A bunch is several heavy floret-tubes on one fine stalk, so the stalk is a
    tip-loaded cantilever exactly as a pedicel is: `theta = W L**2 / (2 E I)`,
    with `W` the whole bunch's weight. Each body's mass is its own volume times
    `catkin_density`; the stalk's radius is a fraction of the twig it leaves,
    like a pedicel. `catkin_droop` now holds this angle in DEGREES (positive),
    which `model.catkin_bunch` sags the stalk by -- replacing the old 0..1
    fraction that came from an Orientation dial. The bodies stay limp; see
    `catkin_density`.
    """
    if organ.catkin_length <= 0.0:
        return
    if organ.pedicel_length <= 0.0:
        organ.catkin_droop = 0.0
        return
    body_mass = profile.catkin_density * np.pi * organ.catkin_width ** 2 * organ.catkin_length
    load = max(1, organ.catkin_count) * body_mass * _G
    r_stalk = max(node_radius * organ.pedicel_width, 1e-6)
    second_moment = np.pi * r_stalk ** 4 / 4.0
    theta = float(np.degrees(load * organ.pedicel_length ** 2 / (2.0 * E * second_moment)))
    organ.catkin_droop = min(theta, 180.0)


def apply_pedicel_droop(plant: Plant, profile: MorphologyProfile) -> None:
    """Every flower's stalk sags under its own flower, and every catkin bunch's
    stalk under its florets, by the same beam the stem uses -- see
    `_set_pedicel_droop` and `_set_catkin_droop`. Requires radii
    (`compute_radii`), so it runs alongside `apply_gravity`; `model._sag` then
    draws the sag at the final resolve. This is what replaces the `pedicel_droop`
    and `catkin_droop` controls: a heavy flower or a heavy catkin on a thin stalk
    hangs, a light one barely moves, and any gradient along an inflorescence
    falls out of the organs' own varying size and stalk length, not a curve."""
    for axis in plant.axes.values():
        E = _axis_modulus(plant, profile, axis)       # the stalk-bearing axis's level's
        for i, phytomer in enumerate(axis.phytomers):
            radius = max(axis.stalk_radius(i), 1e-6)
            for organ in phytomer.organs:
                _set_pedicel_droop(organ, radius, E, profile)
                _set_catkin_droop(organ, radius, E, profile)
        if axis.terminal_organ is not None and axis.phytomers:
            _set_pedicel_droop(axis.terminal_organ,
                               max(axis.stalk_radius(len(axis.phytomers) - 1), 1e-6),
                               E, profile)


def rotate_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation as a 3x3 matrix, so accumulated bends compose by
    multiplication instead of being re-applied vector by vector."""
    theta = np.radians(degrees)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
