"""Stage 2 -- Development: what grows, and when?

The growth loop. Age in [0, 1] becomes a whole number of developmental steps
and the plant is built forward from nothing, one step at a time.

**The trajectory is a single fixed path; Age only says how far along it you
are.** Every curve is evaluated at `step / total_steps` rather than at the
caller's requested age, so growing to Age 0.6 produces exactly the first 60%
of the Age 1.0 run -- a truncation, not a different simulation. Three things
fall out of that for free:

* Phytomer count is monotonic in Age (structural assertion 4) by construction
  rather than by tuning.
* Dragging the Age slider reads as one plant maturing rather than as a series
  of unrelated plants, which is the property the architecture doc is actually
  asking for.
* Nothing needs to be cached or re-based when Age changes.

Determinism is achieved without threading a shared RNG through the loop: every
random quantity is drawn from a stream seeded on `[seed, tag, element_id]`, so
a value depends only on which element it belongs to and never on the order
elements were visited in. Buds in particular draw an intrinsic
`release_threshold` once at creation and are released when the branching-onset
curve rises past it, so a bud can never un-release as Age advances.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from . import maps as growth_maps
from . import masks as growth_masks
from . import signals as growth_signals
from .environment import Environment, steer
from .model import (
    Axis,
    Bud,
    BudBlock,
    BudFate,
    BudPosition,
    BudState,
    LeafForm,
    Organ,
    OrganKind,
    Phytomer,
    Plant,
    _cross,
    orthonormal_reference,
    parallel_transport,
    radial_direction,
    resolve_geometry,
    rotate_about,
    scale_plant,
)
from .morphology import (
    Attachment,
    BloomOrder,
    BranchLevel,
    GrowthPattern,
    Inflorescence,
    InflorescenceKind,
    MorphologyProfile,
    OrientationMode,
    Phyllotaxis,
    PhyllotaxisPattern,
    Topology,
    generic_perennial,
)
from .physics import _MAX_ARC_FACET_DEG

_UP = np.array([0.0, 0.0, 1.0])
_GOLDEN_DEG = 180.0 * (3.0 - np.sqrt(5.0))   # 137.507...: the clump's stems turn by it
_EPS = 1e-9

# _PENDULOUS_SETPOINT_DEG was here, removed 2026-08-20 with the catkin work.
# It turned a pendulous flowering region's gravitropic setpoint past horizontal
# (165 degrees) so the region grew downward on its own. That is a second
# mechanism for the hang that Peduncle Droop already owns, and measured, the
# two fought: with the setpoint a catkin sat at 175 degrees at Droop 0 and then
# went *backwards* as Droop rose -- 172, 163, 155, 143 -- because Droop bends
# the pedicel below the catkin, which levers an already-hanging tube back up.
# Without it the same plant runs 77 -> 122 -> 166 -> 175 degrees across the
# same range: one mechanism, its whole range live, and the angle a catkin
# emerges at is now the plant's own Branch Angle rather than a constant here.
# `Inflorescence.is_pendulous` went with it.


# Distinct stream tags, so two different uses of the same element id never
# draw the same number.
_T_BUD_RELEASE = 1
_T_INTERNODE = 2
# 3 and 4 were the per-node wander (Growth Irregularity, removed
# 2026-10-12): never reuse them.
_T_BRANCH_ANGLE = 5
_T_LEAF_SIZE = 6
_T_AXIS_FLOWER = 7
_T_PHYLLOTAXIS_OFFSET = 8
_T_PEDICEL = 9
_T_LEAF_YAW = 10
_T_LEAF_ROLL = 11
# 12-15 were leaf curl/cup/gravity/twist salts; blade deform was removed
#. The gap is kept so every later
# salt value stays put -- renumbering would reseed unrelated fields.
_T_AXIS_BLOOM = 16
# 17 was _T_LEAF_FORM (heteroblasty roll); removed. Gap kept so later salts stay put.
_T_FLOWER_SIZE = 18
_T_FLOWER_SPREAD = 19
_T_FLOWER_PITCH = 20
_T_PEDICEL_ANGLE = 21
_T_PEDICEL_BEND = 22
_T_FLOWER_COUNT = 23
_T_FLOWER_SPACING = 24
_T_PEDICEL_WIDTH = 25
_T_HEAD_DOME = 27
_T_BRANCH_SPACING = 28
_T_BRANCH_LENGTH = 29
_T_BRANCH_WIDTH = 30
_T_BRANCH_BEND = 31
_T_BRANCH_DROOP = 32
_T_BRANCH_NODES = 33
_T_BRANCH_COUNT = 34
_T_CATKIN_SIZE = 35
_T_CATKIN_STALK = 37
_T_CATKIN_COUNT = 38
_T_ADVENT_AZIMUTH = 39
_T_ADVENT_ELEVATION = 40
_T_ADVENT_ALONG = 41
_T_MAP_RANDOM = 42
"""The growth maps' `random` channel (growth/maps.py). Appended at the end, per
the positional-salt rule: every earlier tag keeps its value, so no existing
draw moves. One tag for every element kind; the kind and the map target ride in
the seed list beside it (`_map_random`) so a bud, a node and an axis sharing an
id -- or a density map and an angle map on one bud -- draw independently."""
_T_FORK_CHANCE = 43
"""Chance forking's per-node roll (`_chance_forks`). Appended, per the rule."""
# 44 was _T_PHYLLO_JITTER (arrangement jitter, removed with the Stem panel's
# leaf arrangement). Gap kept so later salts stay put.
_T_LEVEL_MASK = 45
"""A level mask's survival draw and its `random` entries (`_level_mask_keeps`),
per level. Appended, per the rule."""
_T_LEVEL_SLIDE = 46
"""A level's Slide Variation draw, per node (`_slide_along`). Appended, per the
rule."""

_NEAR_VERTICAL_SIN = float(np.sin(np.radians(10.0)))
"""How close to vertical a heading has to be before its own vertical plane
stops being trustworthy. `|cross(d, up)|` is `sin` of the angle to vertical, so
this is a 10-degree cone."""

_MAX_BRANCH_BEND_DEG = 90.0
"""What a Branch Bend of 1 means: a quarter turn from a branch's base to its
tip. Matched to `pedicel_bend`'s own feel rather than to a physical limit --
this is a shape control, so the ceiling only has to be a round number that
covers the useful range."""


def _rotate_about(vector: np.ndarray, axis: np.ndarray, radians: float) -> np.ndarray:
    """Rodrigues, for the one place that needs a rotation the frame helpers do
    not already provide."""
    c, s = float(np.cos(radians)), float(np.sin(radians))
    return vector * c + _cross(axis, vector) * s + axis * float(np.dot(axis, vector)) * (1.0 - c)
"""Read by `inflorescence.solve_flat_head`, which runs after growth -- the one
tagged stream drawn outside this module. Its own tag either way, so a head's
roughness scatters independently of every other per-flower draw."""

_OBSTACLE_TURN = 2.5
"""How hard a tip turns per step, per unit of obstacle proximity. A solver
constant, not a control: `obstacle_clearance` states the distance the plant
keeps, and this is only the gain that achieves it. Tuned so a tip approaching
a surface head-on is clear of it within two or three nodes -- low enough that
growth curves away rather than kinking, high enough to actually miss."""


def _unit(seed: int, tag: int, ident: int) -> float:
    """A deterministic value in [0, 1) for one element, independent of visit
    order. Constructing a Generator per draw is not free, but it is drawn once
    per element (not per element per step), and order-independence is worth
    considerably more here than the microseconds. Remembered per (seed, tag,
    ident) -- a pure function, and the same draws recur across regrows."""
    key = (int(seed), int(tag), int(ident))
    got = _UNIT_MEMO.get(key)
    if got is None:
        got = float(np.random.default_rng(list(key)).random())
        if len(_UNIT_MEMO) > 200_000:
            _UNIT_MEMO.clear()
        _UNIT_MEMO[key] = got
    return got


_UNIT_MEMO: dict = {}


def _signed(seed: int, tag: int, ident: int) -> float:
    return _unit(seed, tag, ident) * 2.0 - 1.0


def _elevation_from_axis(angle_from_axis_deg: float) -> float:
    """Both `branch_angle_deg` and `leaf_angle_deg` are stated the way a
    person reads them off a plant: the angle away from the stem they grow on,
    0 being pressed flat against it and 90 straight out sideways. The frame
    maths in model.py wants elevation measured up from perpendicular, which is
    the complement."""
    return 90.0 - angle_from_axis_deg


def _rotate_toward(direction: np.ndarray, target: np.ndarray, factor: float) -> np.ndarray:
    """Rotate `direction` a `factor` fraction of the way toward `target`.
    Rotation rather than linear blend, so the result is always unit length and
    the step is well-behaved even when the two are nearly opposed."""
    axis = _cross(direction, target)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS or factor <= 0.0:
        return direction
    angle = np.degrees(np.arccos(np.clip(float(np.dot(direction, target)), -1.0, 1.0)))
    return rotate_about(direction, axis / norm, angle * min(1.0, factor))


def _tilt_from_vertical(direction: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(direction, _UP)), -1.0, 1.0))))


def _apply_gravitropism(direction: np.ndarray, setpoint_tilt_deg: float,
                        strength: float) -> np.ndarray:
    """Steer toward the axis's gravitropic *setpoint angle* rather than toward
    vertical: correct only the angle away from up, leaving azimuth free to
    drift with the wander. A main shoot with a setpoint near zero grows up; a
    lateral with a setpoint of 48 degrees holds 48 degrees and reaches
    outward instead of curling back to parallel with the trunk."""
    if strength <= 0.0:
        return direction
    axis = _cross(direction, _UP)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return direction  # exactly vertical: no well-defined plane to correct in
    delta = (_tilt_from_vertical(direction) - setpoint_tilt_deg) * min(1.0, strength)
    return rotate_about(direction, axis / norm, delta)


def _next_direction(
    previous: np.ndarray,
    profile: MorphologyProfile,
    seed: int,
    axis_id: int,
    index: int,
    setpoint_tilt_deg: float,
    gravitropism: float = -1.0,
    environment=None,
    position=None,
) -> np.ndarray:
    """An apex's direction for its next phytomer: persist, nod (circumnutation),
    then correct back toward the axis's own gravitropic setpoint.

    Note what this deliberately is not. The previous build tried a
    `DirectionalPersistenceBias` that steered growth toward the apex's own
    last direction, and it was correctly rejected as a band-aid -- because
    there it sat on top of an attraction-point-driven direction and merely
    filtered its noise. Here there is no attraction cloud: persistence *is*
    the growth rule, circumnutation the species' own nod, and
    gravitropism is the only thing correcting it. Nothing is being papered
    over.
    """
    direction = np.array(previous, dtype=float)

    # Circumnutation: an off-axis bend whose azimuth *precesses* by a fixed
    # step per node -- so it winds a coherent helix rather than a random walk.
    # Keyed on the node's index along its own axis, so each shoot circumnutates from
    # its own base and the phase is deterministic. Its whole visibility is a
    # balance against gravitropism below; `autotropism` (a post-growth pass) is
    # what keeps the wobble at the tip on a stem gravity would otherwise
    # straighten. Guarded so a zero amplitude is byte-identical to before.
    if profile.circumnutation_deg != 0.0:
        reference = orthonormal_reference(direction)
        phase = index * profile.circumnutation_rate_deg
        tilt_axis = rotate_about(reference, direction, phase)
        direction = rotate_about(direction, tilt_axis, profile.circumnutation_deg)
        direction /= max(float(np.linalg.norm(direction)), _EPS)

    direction = _apply_gravitropism(
        direction, setpoint_tilt_deg,
        profile.gravitropism if gravitropism < 0.0 else gravitropism)

    # Stage 3 gets the last word, deliberately. Gravitropism is a standing
    # correction toward a setpoint; the environment is a reason to depart from
    # it -- a shoot leans out of shade and around a wall *despite* gravity,
    # not within whatever slack gravity leaves. Both terms are zero unless the
    # species asks for them, so this is a no-op for a profile that does not.
    if environment is not None and position is not None:
        if profile.light_seeking > 0.0:
            direction = steer(direction, environment.light_direction(position),
                              profile.light_seeking)
        if profile.obstacle_clearance > 0.0:
            # Proportional to how close the obstacle actually is, so passing
            # one at a distance costs a lean rather than a swerve.
            direction = environment.deflect(direction, position, _OBSTACLE_TURN)
        # Effectors last of all, and not gated on the profile: they are the
        # artist's hand rather than a trait of the species, so an Empty placed
        # in the scene works on any plant without first enabling something on
        # it. A scene with no effectors is a no-op, not a cheaper no-op.
        direction = environment.bias_direction(direction, position)

    return direction / max(float(np.linalg.norm(direction)), _EPS)


def _flower_arrangement(inf, phyllotaxis: Phyllotaxis) -> Phyllotaxis:
    """How the flowers of a flowering region are placed around each node.

    The axis's own phyllotaxis unless the inflorescence overrides it: with
    `whorl_count` (a ring at every node) or `spiral_deg` (its own spiral, one
    flower a node turned that far from the last) -- so for every kind that
    leaves both at 0 this returns the very object the leaves use and the
    placement is bit-for-bit what it was before either existed.

    A whorl is built through `Phyllotaxis` rather than by computing
    `360 / n` here, so the half-step offset between successive whorls comes
    from the one place that rule is written down. Two rings stacked in perfect
    register read as a mistake rather than as a plant, and having discovered
    that once is enough.
    """
    if inf.whorl_count >= 2:
        return Phyllotaxis(pattern=PhyllotaxisPattern.WHORLED, whorl_count=inf.whorl_count)
    if inf.spiral_deg > 0.0:
        return Phyllotaxis(pattern=PhyllotaxisPattern.ALTERNATE, divergence_deg=inf.spiral_deg)
    return phyllotaxis


def _add_bracts(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                phytomer: Phytomer, phyllotaxis: Phyllotaxis, index: int, along: float,
                scale: float, elevation: float) -> None:
    """The leafy bracts subtending a whorl of flowers.

    A flowering node bears no foliage -- an inflorescence's nodes carry flowers
    where a vegetative node carries leaves, which is the phytomer model working
    correctly. But a verticillaster's whole look is that its rings of flowers
    sit in the axils of a pair of leaves, and a raceme's lowest flowers often
    do the same. So a flowering node can carry bracts *as well as* its flowers,
    and nothing else about it changes.

    Three deliberate reuses, each avoiding a control that would have said what
    something already says:

    * **`phyllotaxis` is the axis's own, not the whorl's.** A Lamiaceae node has
      two opposite bracts with many flowers in their axils; passing the leaves'
      arrangement here and the flowers' above is what makes that fall out
      instead of needing a bract count.
    * **Every blade property is the leaf's** -- angle, yaw, roll. How a
      species' foliage sits is one statement about the plant, and a bract is
      foliage. Only the size differs, because that is the one thing a bract is
      actually defined by relative to a leaf.

    No axillary bud: the axil is already occupied by this node's flowers, and
    an inflorescence does not branch vegetatively.
    """
    for angle in phyllotaxis.angles_at(index):
        angle = (angle + axis.phyllotaxis_offset_deg) % 360.0
        organ_id = plant.new_organ_id()
        phytomer.organs.append(Organ(
            id=organ_id,
            kind=OrganKind.LEAF,
            birth_step=plant.step,
            final_scale=max(0.0, scale * (
                1.0 + profile.leaf_size_variation * _signed(seed, _T_LEAF_SIZE, organ_id))),
            divergence_deg=angle,
            elevation_deg=elevation,
            yaw_deg=profile.leaf_yaw_deg
            + profile.leaf_yaw_variation * _signed(seed, _T_LEAF_YAW, organ_id),
            roll_deg=profile.leaf_roll_deg
            + profile.leaf_roll_variation * _signed(seed, _T_LEAF_ROLL, organ_id),
            # A bract is one because of where it sits (a flowering node), so it
            # is written as form 2 outright -- foliage is always form 0.
            form=LeafForm.BRACT,
        ))


def _add_catkins(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                 phytomer: Phytomer, phyllotaxis: Phyllotaxis, index: int,
                 along: float) -> None:
    """The catkins hanging at a vegetative node.

    **A catkin is borne at a node, on its own stalk -- it is not the tip of an
    axis.** That is the whole point of this function, and it is the one thing
    the inflorescence system could not express: every kind in
    `InflorescenceKind` converts an axis's *own apex* into flowers, so a
    hazel's twig would have had to become the catkin rather than carry it.
    Trying it that way makes the branch droop instead of the catkin, and no
    amount of tuning fixes it, because the drooping thing is the branch.

    So this places catkins the way leaves and flowers are placed -- at a node,
    on the phyllotactic frame, gated by a placement window -- and each one is a
    curve that hangs from that node under its own weight.

    Two reuses and one departure, all three deliberate:

    * **`phyllotaxis` is the axis's own**, and decides where round the twig each
      *bunch* sits. How many catkins are in a bunch is `catkin_count`, which is
      a separate fact: an earlier version took it from a whorled arrangement,
      which does put three organs on a node -- but it whorls the *leaves* too,
      and hazel's foliage is spiral. One control cannot say both.
    * **The node keeps its leaf and its axillary bud.** A catkin hangs *beside*
      the foliage on an ordinary vegetative twig; it does not replace it, and
      it does not stop the twig branching. This is the difference from every
      other inflorescence here, all of which convert what a node bears.
    * **Its own placement window**, unlike a bract. Where catkins occur is the
      whole structural statement, and it is not where the leaves are.
    """
    claim = _claimant(profile, axis, Attachment.AXILLARY)
    inf_here = (profile.inflorescence_at(claim, axis.order) if claim is not None
                else _inf(profile, axis))
    if claim is None or inf_here.kind != InflorescenceKind.CATKIN:
        # **The kind decides, not the settings.** `_KIND_DEFAULTS` stamps a
        # catkin length when Catkin is picked and -- like every stamp here --
        # never clears it on switching away, so without this test a plant that
        # had once been a Catkin went on bearing catkins as a raceme, forever.
        # Gating here rather than clearing the stamp is what lets the tuning
        # survive a trip through another kind and come back.
        return
    if profile.catkin_length <= 0.0:
        return
    if not inf_here.placement.allows(along, axis.order):
        return
    for angle in phyllotaxis.angles_at(index):
        angle = (angle + axis.phyllotaxis_offset_deg) % 360.0
        organ_id = plant.new_organ_id()
        length = max(0.0, profile.catkin_length * (
            1.0 + profile.catkin_length_variation
            * _signed(seed, _T_CATKIN_SIZE, organ_id)))
        # The hang is physical now: `physics.apply_pedicel_droop` sets
        # `catkin_droop` (in degrees) from the bunch's own weight on its fine
        # stalk, the same beam a pedicel uses. Set 0 here as a placeholder that
        # pass overwrites -- the old Orientation-angle wiring is gone. See
        # `MorphologyProfile.catkin_density` and `InflorescenceKind.CATKIN`.
        phytomer.organs.append(Organ(
            id=organ_id,
            kind=OrganKind.CATKIN,
            birth_step=plant.step,
            inflorescence=claim,
            final_scale=1.0,
            divergence_deg=angle,
            # Straight out from the node. Which way it ends up pointing is
            # decided by the sag, not by an angle measured off the stem --
            # a catkin hangs, and what it hangs from is at every angle.
            elevation_deg=0.0,
            pedicel_length=max(0.0, profile.catkin_stalk * (
                1.0 + profile.catkin_stalk_variation
                * _signed(seed, _T_CATKIN_STALK, organ_id))),
            catkin_droop=0.0,
            catkin_length=length,
            catkin_count=max(1, int(round(profile.catkin_count * (
                1.0 + profile.catkin_count_variation
                * _signed(seed, _T_CATKIN_COUNT, organ_id))))),
            catkin_noise=max(0.0, profile.catkin_noise),
            catkin_length_variation=max(0.0, profile.catkin_length_variation),
            catkin_width=max(1e-5, profile.catkin_width),
        ))


def _growth_rate(profile: MorphologyProfile, order: int) -> float:
    """Phytomers per step: one per plastochron on every axis. How long an axis
    grows is its level's Steps, not a rate that slows with depth."""
    return 1.0 / max(profile.plastochron, 1)


# --- branching levels ----------------------

def _axis_level_index(profile: MorphologyProfile, axis: Axis) -> int:
    """The index of the branching level this axis grows by, -1 for none: its
    own (`Axis.level`) for a stem, branch or fork; for an axis made outside
    the level machinery, its order's (the deepest level past the stack) --
    what a chain of levels always gave; -1 for an inflorescence branch, which
    keeps the inflorescence's own controls."""
    if (axis.structural_depth >= 1 and not axis.is_fork) or not profile.levels:
        return -1
    if 0 <= axis.level < len(profile.levels):
        return axis.level
    return min(max(axis.order, 0), len(profile.levels) - 1)


def _inflorescences_here(profile: MorphologyProfile, axis: Axis) -> list[int]:
    """The inflorescences placed on this axis's level (`Placement.levels`), in
    priority order (`MorphologyProfile.inflorescence_at` indices). An axis no
    level drives takes every one."""
    i = _axis_level_index(profile, axis)
    out = []
    for k in range(len(profile.extra_inflorescences) + 1):
        levels = profile.inflorescence_at(k, axis.order).placement.levels
        if levels is None or i < 0 or i in levels:
            out.append(k)
    return out


def _claimant(profile: MorphologyProfile, axis: Axis, mode: str) -> int | None:
    """The inflorescence that has this spot (a tip -- TERMINAL -- or one node
    position) on this axis: the first placed on its level that wants it, so
    two inflorescences after the same spot never both grow there.
   None: no inflorescence here wants
    it."""
    for k in _inflorescences_here(profile, axis):
        if mode in profile.inflorescence_at(k, axis.order).attachments:
            return k
    return None


def _inf_index(profile: MorphologyProfile, axis: Axis) -> int:
    """Which inflorescence this axis is: what it stored when it began one or
    was released as a stalk (`Axis.inflorescence`); else, for a vegetative
    axis, its level's tip claimant, or the first placed on its level."""
    if axis.inflorescence >= 0 or not profile.extra_inflorescences:
        return max(axis.inflorescence, 0)
    if axis.structural_depth >= 1 and not axis.is_fork:
        return 0
    k = _claimant(profile, axis, Attachment.TERMINAL)
    if k is None:
        here = _inflorescences_here(profile, axis)
        k = here[0] if here else 0
    return k


def _inf(profile: MorphologyProfile, axis: Axis) -> Inflorescence:
    """This axis's inflorescence (`_inf_index`)."""
    return profile.inflorescence_at(_inf_index(profile, axis), axis.order)


def _attachments(profile: MorphologyProfile, axis: Axis) -> frozenset:
    """Where inflorescences arise on this axis: every attachment an
    inflorescence placed on its level asks for -- none on a level no flower
    component is placed on, so that axis grows as a flowerless plant's
    would. An inflorescence axis has its own inflorescence's."""
    if axis.structural_depth >= 1 and not axis.is_fork:
        return _inf(profile, axis).attachments
    out = frozenset()
    for k in _inflorescences_here(profile, axis):
        out = out | profile.inflorescence_at(k, axis.order).attachments
    return out


_SLIDE_REACH = 0.25
"""How far a branch slides at Slide 1, in internodes: a pair's members then sit
half an internode apart."""


def _slide_along(level: BranchLevel, seed: int, node: int, level_index: int, k: int,
                 count: int) -> float:
    """Where whorl member `k` of `count` leaves, as `Bud.along_internode`: 1 at
    its node, below 1 partway down the internode under it, above 1 partway up
    the one over it. The node's slide is the level's Slide plus its Slide
    Variation drawn per node (clamped to -1..1); a pair's members take it in
    opposite directions, a whorl spread evenly between, a lone branch whole."""
    if level.slide == 0.0 and level.slide_variation <= 0.0:
        return 1.0
    s = level.slide
    if level.slide_variation > 0.0:
        s += level.slide_variation * _signed(_level_seed(seed, level), _T_LEVEL_SLIDE,
                                             node * 16 + level_index)
    s = min(1.0, max(-1.0, s))
    share = 1.0 - 2.0 * k / (count - 1) if count > 1 else 1.0
    return 1.0 + _SLIDE_REACH * s * share


def _level(profile: MorphologyProfile, axis: Axis) -> BranchLevel | None:
    """The branching level this axis grows by, or None (`_axis_level_index`)."""
    i = _axis_level_index(profile, axis)
    return profile.levels[i] if i >= 0 else None


def _level_seed(seed: int, level: BranchLevel | None) -> int:
    """The seed a level's own random draws use: the plant's, or -- when the
    level carries a Seed of its own -- one drawn from both, so reseeding a
    level redraws only its variation. Byte-identical at 0 (`BranchLevel.seed`)."""
    own = int(getattr(level, "seed", 0) or 0) if level is not None else 0
    if not own:
        return seed
    return int(np.random.default_rng([int(seed), _T_LEVEL_SEED, own]).integers(1 << 31))


def _child_levels(profile: MorphologyProfile, axis: Axis) -> list[int]:
    """The levels growing on this axis's level (their indices, list order):
    each branches at every node of it. Empty on a level nothing grows on."""
    i = _axis_level_index(profile, axis)
    return profile.level_children(i) if i >= 0 else []


def _child_level(profile: MorphologyProfile, axis: Axis) -> BranchLevel | None:
    """The first level growing on this axis's level, or None."""
    kids = _child_levels(profile, axis)
    return profile.levels[kids[0]] if kids else None


def _bud_level(profile: MorphologyProfile, axis: Axis, bud: Bud) -> int:
    """The level a bud's branch would be: its own (a branch bud knows it), else
    its axis's first child level, else -1 (no branch to become)."""
    if 0 <= bud.level < len(profile.levels):
        return bud.level
    kids = _child_levels(profile, axis)
    return kids[0] if kids else -1


def _nominal_internode(profile: MorphologyProfile, axis: Axis, index: int) -> float:
    """Axis node `index`'s internode before variation: its base x Length / Step
    ** index on a level axis, the plant's Internode Length otherwise."""
    level = _level(profile, axis)
    if level is None or axis.internode_base < 0.0:
        return profile.internode_length
    return axis.internode_base * level.length_scale_per_step ** max(index, 0)


def _length_variation(plant: Plant, profile: MorphologyProfile, axis: Axis) -> float:
    """The internode length variation for this axis: its level's (Length / Step
    variation); an inflorescence branch takes the level of the axis it grows
    from, so its flowers are spaced as irregularly as its stem's."""
    while True:
        level = _level(profile, axis)
        if level is not None:
            return level.length_variation
        parent = plant.axes.get(axis.parent_axis_id)
        if parent is None:
            return 0.0
        axis = parent


def _spiral_turn(direction: np.ndarray, reference: np.ndarray, level: BranchLevel,
                 index: int, pitch_scale: float = 1.0, roll_scale: float = 1.0) -> np.ndarray:
    """Spiral Pitch / Step and Roll / Step: the heading turns `spiral_pitch_deg`
    toward a direction `spiral_roll_deg * index` round from the frame's
    reference -- a planar arc with no roll, a helix with it. The scales are this
    node's Spiral Pitch / Roll maps."""
    pitch = level.spiral_pitch_deg * pitch_scale
    if pitch == 0.0:
        return direction
    toward = rotate_about(reference, direction, level.spiral_roll_deg * roll_scale * index)
    axis = _cross(direction, toward)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return direction
    turned = rotate_about(direction, axis / norm, pitch)
    return turned / max(float(np.linalg.norm(turned)), _EPS)


def _turn_sideways(direction: np.ndarray, parent_direction: np.ndarray,
                   radial: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Start Yaw: a launch heading turned `yaw_deg` sideways -- toward the
    parent's tangent at its slot, round the parent, by the full angle at any
    pitch. (It used to turn about the outward radial, which for a branch
    leaving near square to its parent is a spin about its own length: the
    sideways swing was only cos(pitch) x yaw, ~0.4 deg of 6 at 86 deg.) Its
    angle to the parent is acos(cos pitch x cos yaw) either way."""
    side = _cross(parent_direction, radial)
    axis = _cross(side, direction)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return direction
    turned = rotate_about(direction, axis / norm, yaw_deg)
    return turned / max(float(np.linalg.norm(turned)), _EPS)


def _launch(parent_direction: np.ndarray, radial: np.ndarray, pitch_deg: float,
            yaw_deg: float) -> np.ndarray:
    """A branch's launch heading: `pitch_deg` from its parent toward the
    outward `radial`, then turned `yaw_deg` sideways (`_turn_sideways`)."""
    direction = rotate_about(parent_direction, _cross(parent_direction, radial), pitch_deg)
    direction = direction / max(float(np.linalg.norm(direction)), _EPS)
    if yaw_deg != 0.0:
        direction = _turn_sideways(direction, parent_direction, radial, yaw_deg)
    return direction


_COMPOUNDS_ALONG_A_REGION = (Topology.BRANCHED, Topology.FLOWERING_REGION)
"""Topologies whose compound form hangs its branches along a rachis.

`BRANCHED` is a panicle, which has always done this. `FLOWERING_REGION` joining
it is what makes a compound corymb and a compound spike reachable, and it is
not a new mechanism -- a panicle *is* a raceme at depth 1, so the machinery for
"this node bears a branch, not a flower" was already general and only the test
in front of it was narrow. `APEX_CLUSTER` compounds too, but hangs its branches
off one point rather than along a region, so it has its own path in
`_begin_inflorescence`. `TERMINAL_FLOWER` has nowhere to put one and `RELAY`
already compounds by relaying."""


_COMPOUNDING_TOPOLOGIES = _COMPOUNDS_ALONG_A_REGION + (Topology.APEX_CLUSTER,)


def _carries_foliage(profile: MorphologyProfile, axis: Axis, along: float) -> bool:
    """Whether this node gets a foliage leaf.

    Every vegetative node does -- placement (which stems, which part of an axis)
    is now a render-time component mask, not a growth-time gate.
   The one growth-level exception is an
    inflorescence branch: "a shoot whose phytomers changed what they carry"
    (`Axis.flowering_from`), entirely inflorescence rather than just at its
    flowering end, so its nodes carry bracts (via `_add_bracts`), never foliage.
    """
    return axis.structural_depth < 1


def _branches_here(profile: MorphologyProfile, axis: Axis) -> bool:
    """True when this axis bears inflorescence *branches* rather than flowers
    -- a panicle's rachis, any compound flowering region, or a compound
    umbel's apex, at every depth above the last.

    One question with one answer, which is why the apex cluster is in here
    even though it puts its branches somewhere quite different: what the two
    share is that `branch_count` rather than `flower_count` is what this axis
    carries, and `_flower_nodes` reads exactly that. The region call sites are
    all guarded by `flowering`, which an apex cluster never sets, so including
    it cannot reach them."""
    inf = _inf(profile, axis)
    return (inf.topology in _COMPOUNDING_TOPOLOGIES
            and 0 <= axis.structural_depth < inf.branching_depth)


def _branch_total_nodes(profile: MorphologyProfile, seed: int, axis: Axis) -> int:
    """Every node an inflorescence branch will build -- its bare basal stretch
    *and* its own flowering region.

    `_inflorescence_branch_nodes` is only the first half, and using it as the
    whole was what made Branch Bend a control that lied. The bend increment is
    the stated turn divided by the span it is spread over, and it is applied at
    every joint the axis has: with the span reading 2 on a corymb branch that
    goes on to build 22 nodes, each of its 21 joints took the *entire* stated
    turn. Measured, before this: a stated 18 degrees delivered 54.5, and 0.6
    and 1.0 both delivered 87.8 -- the vertical clamp absorbing the runaway, so
    the whole upper half of the control was dead.

    An apex cluster's ray adds nothing here, correctly: its flowers hang off
    the apex rather than sitting along it, and `_flower_nodes` already returns
    0 for that case. So this is the branch's real node count for every kind
    without asking which kind it is.
    """
    return max(1, _inflorescence_branch_nodes(profile, axis)
               + max(0, _flower_nodes(profile, seed, axis)))


def _inflorescence_branch_nodes(profile: MorphologyProfile, axis: Axis) -> int:
    """How long an inflorescence branch is, in nodes before it flowers.

    From the inflorescence, graded by where the branch sits along its parent --
    not from a branching level's Steps, which is the plant's own vegetative
    count and on a tree twig would give a panicle branch a woody stem's worth
    of nodes."""
    inf = _inf(profile, axis)
    return max(1, int(round(max(1, inf.branch_nodes) * axis.inflorescence_scale)))


def _expected_phytomers(plant: Plant, profile: MorphologyProfile, axis: Axis) -> int:
    """How many phytomers this axis will end up with, used only to normalise
    its position along it (`Phytomer.factor`): its level's Steps, or an
    inflorescence branch's own node count. Fixed for the axis's whole life, so
    already-grown nodes never re-scale as it extends."""
    if axis.structural_depth >= 1 and not axis.is_fork:
        return max(1, _inflorescence_branch_nodes(profile, axis))
    return _vegetative_nodes(profile, axis)


def _vegetative_nodes(profile: MorphologyProfile, axis: Axis) -> int:
    """The nodes a vegetative axis has when fully grown, before it flowers
    (determinate) or stops (indeterminate): its level's Steps; a fork child's
    share of its parent's budget. On a relaying level, one unit: Unit Steps, or
    what is left of the chain's budget if that is less. Positions along the
    axis are counted over these; Growth % builds part of them
    (`_vegetative_reach`)."""
    level = _level(profile, axis)
    if level is not None and level.relays:
        return _unit_nodes(level, _chain_budget(level, axis))
    if axis.node_budget >= 0:
        return max(2, axis.node_budget)
    steps = level.steps if level is not None else 1
    return max(1, int(steps))


def _chain_budget(level: BranchLevel, axis: Axis) -> int:
    """The nodes a relaying chain has left from this unit on: the level's
    Steps on the first unit, the relay's inheritance after."""
    if axis.node_budget >= 0:
        return axis.node_budget
    return max(1, int(level.steps))


def _chain_reach(level: BranchLevel, axis: Axis) -> float:
    """How many of `_chain_budget`'s nodes the chain builds (Growth %),
    fractional: the first unit's Growth % of the level's Steps, a relay's
    inheritance after -- the whole budget on an uncut chain."""
    if axis.node_reach >= 0.0:
        return axis.node_reach
    if axis.node_budget >= 0:
        return float(axis.node_budget)
    return axis.growth * _chain_budget(level, axis)


_MIN_GROWTH = 0.01
"""The least Growth % an axis takes (1%): a branch exists once its bud breaks
-- its level's Masks decide which branches grow, Growth % only how far."""


def _vegetative_reach(profile: MorphologyProfile, axis: Axis) -> float:
    """How many of `_vegetative_nodes` this axis builds -- Growth %, its
    level's x its map (Natsura's Growth %) -- fractional: the last node built
    grows only that part of its length (`_add_phytomer`). Not renormalised:
    the full count still places everything along the axis and sizes the
    plant's clock, so a cut branch is the lower part of the full one. A fork
    or relay child takes what its parent's chain had left (`node_reach`)."""
    full = _vegetative_nodes(profile, axis)
    level = _level(profile, axis)
    if level is None:
        return float(full)
    if level.relays:
        return min(float(full), _chain_reach(level, axis))
    if axis.node_reach >= 0.0:
        return min(float(full), axis.node_reach)
    if axis.node_budget >= 0:
        return float(full)                      # a child of an uncut chain
    return min(float(full), axis.growth * full)


def _add_emergence_delay(axis: Axis, delay: float) -> None:
    """Emergence Delay: show a just-made stem / branch (and all it grows)
    `delay` steps later than it grows. Growth is untouched -- the axis keeps the
    id, draws and timing it was made with -- `_reveal` holds it back. 0 leaves
    it untouched."""
    steps = max(0, int(round(delay)))
    if steps:
        axis.emergence_shift += steps


def _trim_unborn(plant: Plant, now: float) -> None:
    """Leave out every node whose development has not started by `now` -- a
    flowering region's later nodes (`_preform_region` staggers them) -- with
    what they carry. Each axis's births only rise along it, so this keeps a
    prefix; a no-op (and byte-identical) when nothing is born in the future."""
    for axis in plant.axes.values():
        phys = axis.phytomers
        if not phys or phys[-1].birth_step <= now + 1e-9:
            continue
        keep = len(phys)
        while keep > 0 and phys[keep - 1].birth_step > now + 1e-9:
            keep -= 1
        axis.hidden_nodes += len(phys) - keep
        axis.phytomers = phys[:keep]
        axis.terminal_organ = None           # its tip is not there yet
        if axis.flowering_from > keep:
            axis.flowering_from = -1


def _reveal(plant: Plant, show: float, now: float) -> None:
    """Emergence Delay, shown: the plant at display step `show`, from the plant
    grown to structure step `now` (= min(show, total_steps)). An axis shifted
    by s is at its own step `show - s`: what it built after that is left out
    (nodes, organs; an axis with nothing yet goes, its bud dormant again) and
    what stays is re-clocked so the final resolve ages it from its own step
    (each birth moved by `now - min(show - s, now)`). Unshifted axes are left
    exactly as grown; at `show >= now + every shift` nothing changes at all.

    Exact because a plant grown to step t is the `birth_step <= t` subset of
    the plant grown further (proven for the rig's grow-once bake): per-axis
    state recorded at a later node count is reset (`flowering_from`), and a cut
    tip is a growing apex again."""
    gone: set[int] = set()

    def dormant(bud: Bud) -> None:
        bud.state, bud.fate, bud.child_axis_id, bud.released_step = (
            BudState.DORMANT, BudFate.NONE, -1, -1)

    for aid in sorted(plant.axes):
        axis = plant.axes[aid]
        shift = axis.emergence_shift
        if shift <= 0.0:
            continue
        local = show - shift
        kept = [ph for ph in axis.phytomers if ph.birth_step <= local]
        if not kept:
            gone.add(aid)
            continue
        if len(kept) < len(axis.phytomers):
            axis.hidden_nodes += len(axis.phytomers) - len(kept)
            axis.phytomers = kept
            if axis.flowering_from > len(kept):
                axis.flowering_from = -1
            if axis.apex is not None:
                axis.apex.state = BudState.ACTIVE
        if axis.terminal_organ is not None and axis.terminal_organ.birth_step > local:
            axis.terminal_organ = None
            if axis.apex is not None:
                axis.apex.state = BudState.ACTIVE
        offset = now - min(local, now)
        for ph in axis.phytomers:
            ph.organs = [o for o in ph.organs if o.birth_step <= local]
            for bud in ph.buds:
                if bud.released_step > local:
                    dormant(bud)
                bud.birth_step += offset
            for organ in ph.organs:
                organ.birth_step += offset
            ph.birth_step += offset
        axis.birth_step += offset
        if axis.apex is not None:
            axis.apex.birth_step += offset
        if axis.terminal_organ is not None:
            axis.terminal_organ.birth_step += offset
    if not gone:
        return
    for aid in gone:
        plant.axes.pop(aid)
        if aid in plant.basal_axis_ids:
            plant.basal_axis_ids.remove(aid)
    for axis in plant.axes.values():
        for ph in axis.phytomers:
            for bud in ph.buds:
                if bud.fate == BudFate.SHOOT and bud.child_axis_id in gone:
                    dormant(bud)


def _built_nodes(profile: MorphologyProfile, axis: Axis) -> int:
    """The vegetative nodes this axis actually builds: its reach, the last
    one grown partly (`_vegetative_reach`). `_vegetative_nodes` at 100%."""
    return max(1, math.ceil(_vegetative_reach(profile, axis) - 1e-9))


def _unit_nodes(level: BranchLevel, budget: int) -> int:
    """One sympodial unit's node count: Unit Steps, capped by the budget."""
    return max(1, min(int(level.unit_steps), budget))


# --- growth maps (growth/maps.py) -------------------------------------------
#
# A map multiplies a quantity a decision already computes -- see the module
# docstring there. Everything below is reached only when `profile.maps` has an
# enabled entry on the target in question; every call site is guarded by
# `growth_maps.has_map`, so a plant with no maps runs none of it.

_MAP_FRAME_ATTR = "_growth_map_frame"
"""Where `grow()` hangs the per-step `_MapFrame` on the plant. An instance
attribute rather than a `Plant` field because it is scratch state of one growth
run -- nothing downstream reads it -- and threading it through every signature
that can reach `_add_phytomer` / `_new_axis` would touch a dozen functions for a
value only the map channels use."""

_MAP_KIND_BUD = 0
_MAP_KIND_NODE = 1
_MAP_KIND_AXIS = 2
_MAP_KIND_FLOWER = 3


class _MapFrame:
    """Plant-wide facts the map channels normalise against: the span (for
    `age`), the depth ceiling (for `split_depth`), the plant's height extent at
    the start of this step (for `height`), and the environment (for `shade`).

    The height extent is measured right after `grow()`'s per-step resolve, so
    every position it spans is a real resolved one. Until the first measure --
    or when no map reads a positional channel -- the span is 0 and `height`
    reads 0."""

    __slots__ = ("total_steps", "split_norm", "z_min", "z_span", "environment", "signals",
                 "fixed")

    def __init__(self, profile: MorphologyProfile, environment=None, signals=None,
                 extent: tuple[float, float] | None = None):
        self.total_steps = float(max(profile.total_steps, 1))
        self.split_norm = _split_norm(profile)
        self.z_min, self.z_span = extent if extent is not None else (0.0, 0.0)
        self.fixed = extent is not None
        self.environment = environment
        self.signals = signals

    def measure(self, plant: Plant) -> None:
        if self.fixed:
            return  # `height` reads against the whole plant's extent (`_height_extent`)
        lo, hi = plant.bounds()
        self.z_min = float(lo[2])
        self.z_span = float(hi[2] - lo[2])

    def height(self, position) -> float:
        if self.z_span <= _EPS:
            return 0.0
        return min(1.0, max(0.0, (float(position[2]) - self.z_min) / self.z_span))

    def shade(self, position) -> float:
        if self.environment is None:
            return 0.0
        return 1.0 - self.environment.light_at(position)


def _split_norm(profile: MorphologyProfile) -> float:
    """The deepest `structural_depth` this profile can reach, floor 1: the
    trunk / lateral inflorescence `branching_depth`, and on a forking level the
    most forks its Steps can hold (each fork leaves its children at least two
    nodes). A profile-level ceiling, because at growth time the finished
    plant's deepest split -- what the render-side mask normalises by -- does
    not exist yet."""
    deepest = max(0, profile.inflorescence.branching_depth,
                  *(x.branching_depth for x in profile.extra_inflorescences))
    if profile.lateral_inflorescence is not None:
        deepest = max(deepest, profile.lateral_inflorescence.branching_depth)
    for level in profile.levels:
        if level.forks and level.fork_probability > 0.0:
            deepest = max(deepest, level.steps // 2)
        elif level.relays:
            # one relay generation per unit the level's Steps can hold
            deepest = max(deepest, (level.steps - 1) // max(1, level.unit_steps))
    return float(max(deepest, 1))


def _map_frame(plant: Plant, profile: MorphologyProfile) -> _MapFrame:
    """This run's frame, or a position-less one for a helper called outside
    `grow()` (the structural tests drive `_add_phytomer` directly)."""
    frame = getattr(plant, _MAP_FRAME_ATTR, None)
    if frame is None:
        frame = _MapFrame(profile)
        setattr(plant, _MAP_FRAME_ATTR, frame)
    return frame


def _map_random(seed: int, kind: int, target: str, ident: int) -> float:
    """The `random` channel: one seeded uniform per (element kind, target,
    element id). Same seed-by-id discipline as `_unit`, with two extra words so
    a bud and a node that share an id, or two targets on one bud, never share a
    draw."""
    target_index = growth_maps.RANDOM_SALT[target]
    return float(np.random.default_rng(
        [int(seed), _T_MAP_RANDOM, int(kind), target_index, _draw_word(ident)]).random())


def _draw_word(ident: int) -> int:
    """An element id as a seed word: a stem-to-be's id is negative
    (`_stem_channels`), which a seed word cannot be -- those go above 2^40."""
    ident = int(ident)
    return ident if ident >= 0 else (1 << 40) - ident


def _settled_position(plant: Plant, axis: Axis, index: int) -> np.ndarray:
    """Node `index` of `axis`'s tip where it will stand when mature (its axis
    base for index -1).

    Positions are derived by the start-of-step resolve, which places every node
    at its mature length, so a node born before this step already holds it. One
    built this step still holds a zero placeholder: its place is the last
    resolved one plus each newer node's direction x final length -- exactly what
    the resolve will make of it. (It used to read the nearest resolved point
    below instead, one internode low -- a height mask kept its band one node
    off.) An axis born this step starts from its parent's node, found the same
    way."""
    last = min(index, len(axis.phytomers) - 1)
    position = None
    start = 0
    for i in range(last, -1, -1):
        if axis.phytomers[i].birth_step < plant.step:
            position, start = np.asarray(axis.phytomers[i].tip_position, dtype=float), i + 1
            break
    if position is None:
        parent = plant.axes.get(axis.parent_axis_id)
        if parent is not None and axis.birth_step >= plant.step:
            position = _settled_position(plant, parent, axis.parent_phytomer_index)
        else:
            position = np.asarray(axis.base_position, dtype=float)
    for phytomer in axis.phytomers[start:last + 1]:
        position = position + phytomer.direction * phytomer.final_length
    return position


class _MapChannels:
    """Lazy channel reader for one element (bud / node / new axis), handed to
    `growth_maps.map_factor`. Nothing is computed until a map asks for it, so a
    map reading `factor` never pays for a position lookup or a seeded draw.

    `node_index` and `position` are callables for exactly that reason; the rest
    are plain numbers the call site already has."""

    __slots__ = ("plant", "profile", "seed", "target", "axis", "along",
                 "node_index", "birth_step", "kind", "ident", "position")

    def __init__(self, plant, profile, seed, target, axis, *, along, node_index,
                 birth_step, kind, ident, position):
        self.plant = plant
        self.profile = profile
        self.seed = seed
        self.target = target
        self.axis = axis
        self.along = along
        self.node_index = node_index
        self.birth_step = birth_step
        self.kind = kind
        self.ident = ident
        self.position = position

    def __call__(self, name: str) -> float | None:
        axis = self.axis
        order = axis.order if axis is not None else 0
        if name == "factor":
            return float(self.along)
        if name == "node_index":
            return float(self.node_index())
        if name == "branch_depth":
            return min(1.0, max(order, 0) / max(self.profile.max_level_depth, 1))
        if name == "on_branch":
            return 1.0 if order >= 1 else 0.0
        frame = _map_frame(self.plant, self.profile)
        if growth_signals.is_derived(name):
            return self._derived(frame, name)
        if name == "split_depth":
            depth = axis.structural_depth if axis is not None else -1
            return min(1.0, max(depth, 0) / frame.split_norm)
        if name == "age":
            return min(1.0, max(0.0, self.birth_step / frame.total_steps))
        if name == "random":
            return _map_random(self.seed, self.kind, self.target, self.ident)
        if name == "height":
            return frame.height(self.position())
        if name == "shade":
            return frame.shade(self.position())
        return None  # unknown channel: identity, per `map_factor`

    def seeded_random(self, map_seed: int) -> float:
        """The `random` channel under a map's own seed: the same element words
        plus the seed, so two maps on one target can draw apart."""
        target_index = growth_maps.RANDOM_SALT[self.target]
        return float(np.random.default_rng(
            [int(self.seed), _T_MAP_RANDOM, int(self.kind), target_index,
             _draw_word(self.ident), int(map_seed)]).random())

    def _derived(self, frame: _MapFrame, name: str) -> float | None:
        """A custom attribute / class (growth/signals.py) at this element: its
        position, its own plain channels, its id for the random draws. None
        (identity) when the plant has no such attribute or class."""
        signals = frame.signals
        if signals is None or not signals.defines(name):
            return None

        def base(key: str):
            value = self(key)
            return None if value is None else np.array([value])

        evaluator = growth_signals.Evaluator(
            signals, np.array([self.ident]), base, lambda: np.array([self.position()]))
        return float(evaluator.get(name)[0])


def _node_index(plant: Plant, profile: MorphologyProfile, axis: Axis, index: int) -> float:
    """`index / max(expected - 1, 1)`, clamped -- the `node_index` channel."""
    return min(1.0, max(0, index) / max(_expected_phytomers(plant, profile, axis) - 1, 1))


def _bud_channels(plant: Plant, profile: MorphologyProfile, seed: int, target: str,
                  axis: Axis, bud: Bud) -> _MapChannels:
    """A dormant bud's channels, read where it sits: its node's factor, its own
    birth step (so re-reading it every step never drifts), its node's position."""
    index = bud.phytomer_index
    along = axis.phytomers[index].factor if 0 <= index < len(axis.phytomers) else 0.0
    return _MapChannels(
        plant, profile, seed, target, axis,
        along=along,
        node_index=lambda: _node_index(plant, profile, axis, index),
        birth_step=bud.birth_step, kind=_MAP_KIND_BUD, ident=bud.id,
        position=lambda: _settled_position(plant, axis, index),
    )


def _bud_map_factor(plant: Plant, profile: MorphologyProfile, seed: int, target: str,
                    axis: Axis, bud: Bud, stack=None) -> float:
    """A dormant bud's map factor on `target`, from `stack` (a level's maps) or
    the plant's."""
    maps = profile.maps if stack is None else stack
    return growth_maps.stack_factor(maps, target, _bud_channels(
        plant, profile, seed, target, axis, bud))


def _node_map_factor(plant: Plant, profile: MorphologyProfile, seed: int, target: str,
                     axis: Axis, index: int, along: float, kind: int, ident: int,
                     stack=None, birth_step: int | None = None) -> float:
    """A node being built (or a bud on it): `along` is already index over the
    expected count, which is `node_index` too at growth time. Position is the
    node's base -- the resolved point it grows from. `stack` is a level's maps
    (else the plant's); `birth_step` the node's, when read after it was built."""
    maps = profile.maps if stack is None else stack
    return growth_maps.stack_factor(maps, target, _MapChannels(
        plant, profile, seed, target, axis,
        along=along, node_index=lambda: along,
        birth_step=plant.step if birth_step is None else birth_step, kind=kind, ident=ident,
        position=lambda: _settled_position(plant, axis, index - 1),
    ))


def _attachment_channels(plant: Plant, profile: MorphologyProfile, seed: int, target: str,
                         axis: Axis) -> _MapChannels:
    """A newly born axis's channels: structural ones read the new axis (its
    order, its depth), positional ones where it attaches on its parent -- the
    parent node's factor, index and position -- since the axis has no nodes
    yet. A basal stem reads factor 0 at its own base."""
    parent = plant.axes.get(axis.parent_axis_id)
    index = axis.parent_phytomer_index
    if parent is not None and 0 <= index < len(parent.phytomers):
        along = parent.phytomers[index].factor

        def node_index() -> float:
            return _node_index(plant, profile, parent, index)

        def position():
            return _settled_position(plant, parent, index)
    else:
        along = 0.0

        def node_index() -> float:
            return 0.0

        def position():
            return axis.base_position
    return _MapChannels(
        plant, profile, seed, target, axis,
        along=along, node_index=node_index,
        birth_step=axis.birth_step, kind=_MAP_KIND_AXIS, ident=axis.id,
        position=position,
    )


_FLOWER_ORGAN_TARGETS = (
    growth_maps.PEDICEL_LENGTH, growth_maps.PEDICEL_ANGLE, growth_maps.PEDICEL_BEND,
    growth_maps.PEDICEL_WIDTH, growth_maps.FLOWER_PITCH,
)


def _flower_map_factors(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                        index: int, along: float, organ_id: int, birth_step,
                        inf: Inflorescence) -> dict | None:
    """A flower's inflorescence-map factors, by target (None without a map on
    any): read at its node -- that node's factor and index, its base position
    -- with the flower's own id (its `random`) and birth (its `age`)."""
    stack = inf.maps
    if not stack:
        return None
    out = {}
    for target in _FLOWER_ORGAN_TARGETS:
        if growth_maps.enabled(stack, target):
            out[target] = growth_maps.stack_factor(stack, target, _MapChannels(
                plant, profile, seed, target, axis,
                along=along, node_index=lambda: _node_index(plant, profile, axis, index),
                birth_step=birth_step, kind=_MAP_KIND_FLOWER, ident=organ_id,
                position=lambda: _settled_position(plant, axis, index - 1)))
    return out or None


def _branch_map_factor(plant: Plant, profile: MorphologyProfile, seed: int,
                       inf: Inflorescence, child: Axis, target: str) -> float:
    """An inflorescence branch's map factor on `target`, fixed at its birth and
    read at its attachment; exactly 1.0 without a map on it."""
    if not growth_maps.enabled(inf.maps, target):
        return 1.0
    return growth_maps.stack_factor(inf.maps, target, _attachment_channels(
        plant, profile, seed, target, child))


def _level_index(profile: MorphologyProfile, level: BranchLevel) -> int:
    for i, candidate in enumerate(profile.levels):
        if candidate is level:
            return i
    return 0


def _level_mask_keeps(profile: MorphologyProfile, seed: int, level: BranchLevel,
                      channels: _MapChannels, raw_index: int, bud: Bud | None = None) -> bool:
    """Whether a level's Masks keep one of its branches (a bud) or stems: the
    component-mask fold (`growth.masks.evaluate`) over the element's channels,
    its value the element's seeded survival chance. `raw_index` is the node
    ordinal an index entry's Modulo reads. Always True for an empty stack.

    A `bud` keeps each entry's own window (`Bud.mask_windows`), for the mask
    preview to show what growth decided."""
    entries = [e for e in level.masks]
    if not entries:
        return True
    salt = int(np.random.default_rng(
        [int(_level_seed(seed, level)), _T_LEVEL_MASK,
         _level_index(profile, level)]).integers(1 << 62))
    stack = growth_masks.MaskStack(entries=entries, keep=1.0, seed=salt)
    values: dict[str, np.ndarray] = {}
    for e in entries:
        if e.mute or e.channel == growth_masks.RANDOM_CHANNEL or e.channel in values:
            continue
        value = channels(e.channel)
        if value is not None:
            values[e.channel] = np.array([value])
        raw = growth_masks.ORDINAL_CHANNELS.get(e.channel)
        if raw is not None:
            values[raw] = np.array([raw_index])
    ids = np.array([channels.ident])
    if bud is not None:
        bud.mask_windows = tuple(
            -1.0 if e.mute else float(growth_masks.entry_window(stack, i, values, ids)[0])
            for i, e in enumerate(entries))
    return bool(growth_masks.evaluate(stack, values, ids)[1][0])


def _assign_axis_maps(plant: Plant, profile: MorphologyProfile, seed: int,
                      axis: Axis) -> None:
    """The plant-level per-axis target (gravitropism), evaluated once at the
    axis's birth and held for its whole life (`Axis.map_gravitropism`)."""
    if not growth_maps.has_map(profile, growth_maps.GRAVITROPISM):
        return
    axis.map_gravitropism = growth_maps.map_factor(
        profile, growth_maps.GRAVITROPISM,
        _attachment_channels(plant, profile, seed, growth_maps.GRAVITROPISM, axis))


def _level_axis_factor(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                       level: BranchLevel, target: str) -> float:
    """A level map on a per-branch target (Roll, Steps, Start Length, Start
    Width), read once at the axis's birth -- 1.0 without one."""
    if not growth_maps.level_has_map(level, target):
        return 1.0
    return growth_maps.stack_factor(level.maps, target, _attachment_channels(
        plant, profile, seed, target, axis))


def _add_phytomer(plant: Plant, axis: Axis, profile: MorphologyProfile, seed: int,
                  environment=None) -> Phytomer | None:
    """The next node on this axis, or None if the environment stopped it there.

    None is only ever returned when a tip has run out of room -- solid matter,
    or a Forbid effector, which are one question by the time they get here.
    Neither can happen without Stage 3 -- see `_extend_apices`, the only caller,
    for what it does about it.
    """
    index = len(axis.phytomers)
    level = _level(profile, axis)
    gravitropism = axis.gravitropism
    if axis.map_gravitropism != 1.0:
        # A `gravitropism` growth map: scale whichever strength this axis runs
        # on -- the species' own, or its inflorescence's override. Resolved to a
        # real value here, since the -1 sentinel cannot be multiplied.
        gravitropism = ((profile.gravitropism if gravitropism < 0.0 else gravitropism)
                        * axis.map_gravitropism)
    if axis.phytomers:
        previous = axis.phytomers[-1]
        heading = previous.direction
        setpoint = axis.setpoint_tilt_deg
        if level is not None and axis.flowering_from < 0 and level.maps \
                and growth_maps.level_has_map(level, growth_maps.SETPOINT):
            # A Gravitropic Setpoint map: the angle from vertical this node's
            # growth is corrected toward, x its map -- a shoot's set-point is
            # not fixed along it (Digby & Firn 1995): a tip aiming more upright
            # than the wood behind it turns every branch up, whichever way it
            # faces.
            along_here = min(1.0, index / max(_expected_phytomers(plant, profile, axis) - 1, 1))
            setpoint = min(180.0, max(0.0, setpoint * _node_map_factor(
                plant, profile, seed, growth_maps.SETPOINT, axis, index, along_here,
                _MAP_KIND_NODE, axis.id * 4096 + index, stack=level.maps)))
        if level is not None and axis.flowering_from < 0:
            # Spiral Pitch / Roll per step, before the wander and gravitropism,
            # which then act on the turned heading -- each x its level map.
            pitch_scale = roll_scale = 1.0
            if level.maps and level.spiral_pitch_deg != 0.0:
                along_here = min(1.0, index / max(_expected_phytomers(plant, profile, axis) - 1, 1))
                for target in (growth_maps.SPIRAL_PITCH, growth_maps.SPIRAL_ROLL):
                    if growth_maps.level_has_map(level, target):
                        f = _node_map_factor(plant, profile, seed, target, axis, index,
                                             along_here, _MAP_KIND_NODE,
                                             axis.id * 4096 + index, stack=level.maps)
                        if target == growth_maps.SPIRAL_PITCH:
                            pitch_scale = f
                        else:
                            roll_scale = f
            heading = _spiral_turn(heading, previous.reference, level, index,
                                   pitch_scale, roll_scale)
        if axis.displace_deg != 0.0 and index == axis.flowering_from \
                and axis.displace_normal is not None:
            # An Aligned Relay's spent apex: its inflorescence leans aside for
            # the relay continuing in line (`_relay_unit`).
            heading = rotate_about(heading, axis.displace_normal, axis.displace_deg)
        direction = _next_direction(
            heading, profile, seed, axis.id, index, setpoint,
            gravitropism,
            environment=environment, position=previous.tip_position,
        )
        reference = parallel_transport(previous.reference, previous.direction, direction)
        # Align/Level twist the frame after transport, not before: transport is
        # what carries a divergence spiral consistently from one node to the
        # next, and correcting ahead of it would be corrected away again.
        if environment is not None:
            reference = environment.bias_reference(
                reference, direction, previous.tip_position)
    else:
        direction = np.array(axis.base_direction, dtype=float)
        reference = np.array(axis.base_reference, dtype=float)
        if axis.orientation_target_deg >= 0.0:
            # An inflorescence bends from **where it joins the branch**, so its
            # very first segment is already turning. Every other axis leaves at
            # the angle its bud gave it and is only corrected from the second
            # node on, which for a two-node peduncle means the bend could not
            # start until it was half over -- reported as the peduncle staying
            # straight and stiff before it turned.
            strength = 1.0 if axis.gravitropism < 0.0 else axis.gravitropism
            if axis.map_gravitropism != 1.0:
                strength *= axis.map_gravitropism
            turned = _apply_gravitropism(direction, axis.setpoint_tilt_deg, strength)
            # The frame travels with it. Rotating the direction and keeping the
            # base reference leaves the two no longer perpendicular -- measured
            # 78 degrees out -- and `radial_direction` then places every organ
            # on that skewed frame, which showed up as flowers ignoring or
            # inverting Pedicel Angle.
            reference = parallel_transport(reference, direction, turned)
            direction = turned

    ident = axis.id * 4096 + index
    flowering = axis.flowering_from >= 0 and index >= axis.flowering_from
    along = min(1.0, index / max(_expected_phytomers(plant, profile, axis) - 1, 1))
    # **A flowering node's spacing is the inflorescence's, not its branch's**:
    # the plant's Internode Length, scaled by the region's own Flower/Branch
    # Spacing below -- never the level's taper, or a raceme on a short-noded
    # branch would crowd by that branch's scale. A vegetative node is its
    # level's: the axis's base internode x Length / Step ** index.
    nominal = profile.internode_length if flowering else _nominal_internode(
        profile, axis, index)
    # Length Variation: a vegetative internode takes ONE draw per axis, so each
    # stem / branch is its own length -- longer or shorter whole -- with its
    # internodes still evenly stepped. (Drawn per node, the +/- averaged out
    # over a stem: five stems at 0.18 came out within ~4% of each other, reading
    # as identical copies.) A flowering node keeps a draw per node: its flowers
    # are spaced as irregularly as the variation says.
    draw = (_signed(seed, _T_INTERNODE, ident) if flowering
            else _signed(_level_seed(seed, _level(profile, axis)), _T_AXIS_LENGTH, axis.id))
    length = nominal * (1.0 + _length_variation(plant, profile, axis) * draw)
    if axis.structural_depth >= 1:
        # An inflorescence branch's own internodes. Its length is this times
        # `branch_nodes`, which are deliberately two controls: how far apart its
        # flowers sit, and how many it carries.
        length *= max(0.0, axis.inflorescence_length)
    if flowering:
        # An inflorescence axis is normally more crowded than the leafy stem
        # below it.
        inf_here = _inf(profile, axis)
        # Spacing varies along the flowering region as well as being scaled by
        # it: some racemes crowd at the tip and open out at the base, some do
        # the reverse, and both read very differently from an evenly spaced
        # one. Measured on the same normalised coordinate the flower gradients
        # use, so the two cannot disagree.
        flower_nodes = _flower_nodes(profile, seed, axis)
        flower_up = min(1.0, (index - axis.flowering_from) / max(flower_nodes - 1, 1))
        if _branches_here(profile, axis):
            # A rachis spaces its *branches*, which is a different number from
            # how its branchlets space their flowers -- see
            # `Inflorescence.branch_spacing`. Both read the same normalised
            # coordinate, so the two gradients stay comparable.
            length *= (
                inf_here.branch_spacing
                * inf_here.branch_spacing_profile(flower_up)
                * (1.0 + inf_here.branch_spacing_variation
                   * _signed(seed, _T_BRANCH_SPACING, ident))
            )
            spacing_target = growth_maps.INFL_BRANCH_SPACING
        else:
            length *= (
                inf_here.internode_scale
                * inf_here.flower_spacing_profile(flower_up)
                * (1.0 + inf_here.flower_spacing_variation
                   * _signed(seed, _T_FLOWER_SPACING, ident))
            )
            spacing_target = growth_maps.FLOWER_SPACING
        # The flower component's own spacing map, per flowering node.
        if growth_maps.enabled(inf_here.maps, spacing_target):
            length *= _node_map_factor(plant, profile, seed, spacing_target, axis, index,
                                       along, _MAP_KIND_NODE, ident, stack=inf_here.maps)

    # Last of the multipliers, so a map scales the internode every other rule
    # has already sized, and the obstacle test below checks the length actually
    # built: a vegetative node on a level by that level's Length map, any other
    # (flowering, inflorescence branch) by the plant's Internode Length map.
    if level is not None and not flowering:
        if growth_maps.level_has_map(level, growth_maps.LENGTH):
            length *= _node_map_factor(plant, profile, seed, growth_maps.LENGTH,
                                       axis, index, along, _MAP_KIND_NODE, ident,
                                       stack=level.maps)
    elif growth_maps.has_map(profile, growth_maps.INTERNODE_LENGTH):
        length *= _node_map_factor(plant, profile, seed, growth_maps.INTERNODE_LENGTH,
                                   axis, index, along, _MAP_KIND_NODE, ident)
    if level is not None and not flowering:
        # Growth %: the last node built grows only the part of it the axis
        # reaches (none short of 1 on an uncut axis: byte-identical).
        reach = _vegetative_reach(profile, axis)
        if index + 1 > reach:
            length *= max(reach - index, 0.0)

    if environment is not None and environment.blocks_growth:
        # Where this node would put its tip. Steering has already done what it
        # can; a tip that would still end up inside solid matter has run out of
        # room and stops here, rather than growing through the obstacle.
        #
        # The first node of an axis is checked too, measured from the axis's
        # own base. It has no steering applied -- it launches at the angle
        # phyllotaxis and branch angle gave it -- so it is precisely the node
        # most likely to be aimed straight into something, and leaving it
        # unchecked let five nodes end up inside a slab in testing.
        origin = axis.phytomers[-1].tip_position if axis.phytomers else axis.base_position
        if environment.blocked_at(origin + direction * max(length, 1e-5)):
            return None

    phytomer = Phytomer(
        index=index,
        birth_step=plant.step,
        final_length=max(length, 1e-5),
        direction=direction,
        reference=reference,
        factor=along,
    )

    # Leaves, and one axillary bud per leaf -- the phytomer model's whole
    # point. There is no path here that produces a bare node.
    # Angles and sizes vary along the axis, read at this node's own factor.
    # Every profile curve is a multiplier on the panel's own value, so a flat
    # curve at 1.0 is exactly the behaviour before curves existed.
    phyllotaxis = profile.phyllotaxis_for(axis.order)
    leaf_elevation = _elevation_from_axis(
        profile.leaf_angle_deg * profile.leaf_angle_profile(along))
    leaf_scale_here = profile.leaf_size * profile.leaf_size_profile(along)
    inf = _inf(profile, axis)
    branching = flowering and _branches_here(profile, axis)
    # A flowering region may arrange its flowers differently from the leaves
    # below it -- a verticillaster's six-flowered whorls over decussate foliage.
    # `_flower_arrangement` returns the axis's own phyllotaxis unchanged unless
    # the inflorescence asks otherwise, so this is the same list as before for
    # every kind that does not.
    placement = (_flower_arrangement(inf, phyllotaxis)
                 if flowering and not branching else phyllotaxis)
    vegetative = not flowering and not branching
    # The branches this node bears are laid out by the NEXT level's own
    # phyllotaxis (its spiral, whorl and whorl angle), apart from the leaves,
    # and every one grows out. On the deepest level there is no next level:
    # the leaf-axil buds stay, and can only ever become flowers.
    child_levels = _child_levels(profile, axis) if vegetative else []

    def axillary_bud(angle: float, pitch_deg: float, maps=None, level: int = -1,
                     along_internode: float = 1.0) -> None:
        bud_id = plant.new_bud_id()
        if maps and growth_maps.enabled(maps, growth_maps.PITCH):
            # The branch level's Pitch map, per bud: scales the angle away from
            # the stem (the panel's own units) before it becomes an elevation,
            # so doubling it means twice as far from the stem -- not a doubled
            # elevation, which would push a 48-degree branch *toward* the stem.
            pitch_deg *= _node_map_factor(plant, profile, seed, growth_maps.PITCH,
                                          axis, index, along, _MAP_KIND_BUD, bud_id,
                                          stack=maps)
        if 0 <= level < len(profile.levels):
            var = profile.levels[level].start_pitch_variation_deg
            if var > 0.0:
                # Pitch Variation: this branch's own angle off the stem.
                pitch_deg += var * _signed(_level_seed(seed, profile.levels[level]),
                                           _T_START_PITCH, bud_id)
        phytomer.buds.append(Bud(
            id=bud_id,
            position_kind=BudPosition.AXILLARY,
            birth_step=plant.step,
            axis_id=axis.id,
            phytomer_index=index,
            divergence_deg=angle,
            elevation_deg=_elevation_from_axis(pitch_deg),
            release_threshold=_unit(seed, _T_BUD_RELEASE, bud_id),
            level=level,
            along_internode=along_internode,
        ))

    for slot, angle in enumerate(placement.angles_at(index)):
        angle = (angle + axis.phyllotaxis_offset_deg) % 360.0
        organ_id = plant.new_organ_id()

        if branching:
            # A panicle rachis: this node bears an inflorescence branch where a
            # raceme would bear a flower. `laterals_per_node` caps how many --
            # the same field a cyme uses for its relays, because "how many axes
            # does an inflorescence node start" is one quantity.
            if slot >= max(1, inf.laterals_per_node):
                continue
            # Where this branch sits along the rachis -- the same normalised
            # coordinate `_release_inflorescence_branches` reads its other
            # gradients at, so the angle cannot disagree with them about which
            # end of the inflorescence a branch is at.
            branch_up = min(1.0, (index - axis.flowering_from)
                            / max(_flower_nodes(profile, seed, axis) - 1, 1))
            bud_id = plant.new_bud_id()
            branch_angle = (inf.branch_angle_deg
                            * float(inf.branch_angle_profile(branch_up))
                            * (1.0 + inf.branch_angle_variation
                               * _signed(seed, _T_BRANCH_ANGLE, bud_id)))
            if growth_maps.enabled(inf.maps, growth_maps.INFL_BRANCH_ANGLE):
                branch_angle *= _node_map_factor(
                    plant, profile, seed, growth_maps.INFL_BRANCH_ANGLE, axis, index, along,
                    _MAP_KIND_BUD, bud_id, stack=inf.maps)
            phytomer.buds.append(Bud(
                id=bud_id,
                position_kind=BudPosition.AXILLARY,
                birth_step=plant.step,
                axis_id=axis.id,
                phytomer_index=index,
                divergence_deg=angle,
                # No `_elevation_from_axis` -- the second exception in this
                # module alongside the pedicel: an inflorescence branch angle is
                # stated **from perpendicular** so that it can be signed. 0 is
                # straight out sideways, +90 flat up the rachis, -90 flat down.
                # A panicle branch sweeping below horizontal is ordinary, and
                # from-the-axis cannot say so without running past 90 where the
                # number stops meaning what it says. Same reasoning and same
                # convention as `pedicel_angle_deg`.
                elevation_deg=branch_angle,
                # Released by the inflorescence itself a few lines below, never
                # by the vegetative branching lottery, so the threshold is not
                # consulted -- see `_release_inflorescence_branches`.
                release_threshold=0.0,
            ))
            continue

        if flowering:
            # A flowering node bears a flower on a pedicel and no axillary
            # bud -- an inflorescence does not branch vegetatively.
            position = index - axis.flowering_from
            count = max(1, _flower_nodes(profile, seed, axis))
            # Normalised position in the flowering region: 0 at the first
            # flower, 1 at the last. Every gradient below is read here, so they
            # all share one coordinate and cannot disagree about where "the
            # top of the inflorescence" is.
            organ = Organ(
                id=organ_id,
                kind=OrganKind.FLOWER,
                birth_step=plant.step,
                divergence_deg=_spread(
                    angle, inf.flower_spread_deg,
                    inf.flower_spread_variation * _signed(seed, _T_FLOWER_SPREAD, organ_id)),
            )
            # Everything a flower reads off its position in the region is set
            # in one place, so it can be set again once the region's real
            # length is known -- see `_renormalise_truncated_regions`.
            organ.map_factors = _flower_map_factors(plant, profile, seed, axis, index, along,
                                                    organ_id, organ.birth_step, inf)
            _flower_gradients(profile, seed, axis, organ, position, count)
            phytomer.organs.append(organ)
            continue

        # Foliage on every vegetative node now (`_carries_foliage`); which
        # leaves render is a component mask, not a growth gate. The bud below is
        # created either way, keeping branching independent of foliage.
        if _carries_foliage(profile, axis, along):
            phytomer.organs.append(Organ(
                id=organ_id,
                kind=OrganKind.LEAF,
                birth_step=plant.step,
                final_scale=leaf_scale_here
                * (1.0 + profile.leaf_size_variation * _signed(seed, _T_LEAF_SIZE, organ_id)),
                divergence_deg=angle,
                elevation_deg=leaf_elevation,
                yaw_deg=profile.leaf_yaw_deg
                + profile.leaf_yaw_variation * _signed(seed, _T_LEAF_YAW, organ_id),
                roll_deg=profile.leaf_roll_deg
                + profile.leaf_roll_variation * _signed(seed, _T_LEAF_ROLL, organ_id),
                form=LeafForm.JUVENILE,
            ))
        if not child_levels:
            # Deepest level: a leaf-axil bud, a lateral flower at most.
            axillary_bud(angle, profile.leaf_angle_deg)
        _ = slot

    # An axis's tip is carried on by the axis itself (monopodial), by its
    # relays (a sympodial unit's last node) or by its forks -- never by a side
    # branch: its last vegetative node bears none, so what continues the tip
    # follows the parent's direction. A fork
    # rolled at an earlier node spends that node's buds in `_chance_fork`.
    bare_tip = (bool(child_levels) and level is not None
                and index >= _built_nodes(profile, axis) - 1)
    for ci in ([] if bare_tip else child_levels):
        # Each level growing on this one lays out its own slots at every node,
        # measured from the node's own transported reference, which carries
        # this axis's Start Roll -- so Roll turns where its branches start.
        child_level = profile.levels[ci]
        angles = child_level.angles_at(index)
        for k, angle in enumerate(angles):
            axillary_bud(angle, child_level.start_pitch_deg, child_level.maps, ci,
                         _slide_along(child_level, seed, axis.id * 4096 + index, ci, k,
                                      len(angles)))

    if flowering and not branching and inf.bract_size > 0.0:
        _add_bracts(plant, profile, seed, axis, phytomer, phyllotaxis, index, along,
                    leaf_scale_here * inf.bract_size, leaf_elevation)
    if not flowering and not branching:
        # Vegetative nodes only. A catkin hangs from an ordinary leafy twig --
        # that is what distinguishes it from a spike, which *is* the axis.
        _add_catkins(plant, profile, seed, axis, phytomer, phyllotaxis, index, along)

    axis.phytomers.append(phytomer)
    _update_orientation(plant, profile, seed, axis)
    if not flowering and not branching and index >= 1:
        # The same idea as a catkin, for the kinds that carry real flowers: a
        # stalk released at a node rather than the axis itself becoming the
        # inflorescence.
        #
        # Released **one node behind the apex**, which is the botanical rule
        # and also fixes a visible defect. An inflorescence breaks from a bud
        # the shoot has already grown past, so the newest node never carries
        # one -- and since nothing is created after the last node, a shoot
        # always ends in a bare internode rather than in a stalk. Released at
        # the apex instead, the branch stopped exactly where its topmost
        # pendant began, and the ordinary 48-degree branching angle read as a
        # kink in one continuous stem rather than as a fork.
        _release_node_inflorescences(
            plant, profile, seed, axis, index - 1,
            min(1.0, (index - 1) / max(_expected_phytomers(plant, profile, axis) - 1, 1)))
    if branching:
        # After the append, because releasing reads the phytomer back off the
        # axis by index. Before `_release_buds` runs this step, which is what
        # keeps these out of the vegetative branching gates entirely.
        _release_inflorescence_branches(plant, profile, seed, axis, index)
    return phytomer


def _new_axis(
    plant: Plant,
    profile: MorphologyProfile,
    seed: int,
    order: int,
    base_position: np.ndarray,
    base_direction: np.ndarray,
    base_reference: np.ndarray | None = None,
    parent_axis_id: int = -1,
    parent_phytomer_index: int = -1,
    parent_along_internode: float = 1.0,
    structural_depth: int = -1,
    inflorescence_scale: float = 1.0,
    is_fork: bool = False,
    level: int = -1,
    inflorescence: int = -1,
) -> Axis:
    direction = np.array(base_direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), _EPS)
    axis = Axis(
        id=plant.new_axis_id(),
        order=order,
        birth_step=plant.step,
        parent_axis_id=parent_axis_id,
        parent_phytomer_index=parent_phytomer_index,
        parent_along_internode=parent_along_internode,
        structural_depth=structural_depth,
        is_fork=is_fork,
        level=level,
        inflorescence_scale=inflorescence_scale,
        base_position=np.array(base_position, dtype=float),
        base_direction=direction,
        base_reference=(
            orthonormal_reference(direction) if base_reference is None
            else np.array(base_reference, dtype=float)
        ),
        # The angle it launched at is the angle it will hold.
        setpoint_tilt_deg=_tilt_from_vertical(direction),
    )
    parent = plant.axes.get(parent_axis_id)
    if parent is not None:
        axis.emergence_shift = parent.emergence_shift   # shown when its parent is
    if inflorescence >= 0:
        axis.inflorescence = inflorescence
    elif parent is not None and structural_depth >= 1 and not is_fork:
        # An inflorescence branch is part of the inflorescence it leaves.
        axis.inflorescence = parent.inflorescence
    # Start this axis's spiral somewhere of its own rather than at a shared
    # zero -- see Axis.phyllotaxis_offset_deg. Seeded on the axis id so it
    # stays deterministic and order-independent like every other draw here.
    axis.phyllotaxis_offset_deg = _unit(seed, _T_PHYLLOTAXIS_OFFSET, axis.id) * 360.0
    level = _level(profile, axis)
    if level is not None:
        parent = plant.axes.get(parent_axis_id)
        # The level's per-branch maps (Roll, Growth %, Start Length, Start
        # Width), read once at birth; a fork continues its parent and takes none.
        mapped = bool(level.maps) and not is_fork

        def factor(target: str) -> float:
            return (_level_axis_factor(plant, profile, seed, axis, level, target)
                    if mapped else 1.0)

        if parent is None:
            # A stem: the stems level's Start Length is its first internode,
            # in metres (it has no parent to be relative to).
            axis.internode_base = level.start_length_scale * factor(growth_maps.START_LENGTH)
        elif is_fork:
            # A fork continues its parent's taper from the node it splits at.
            axis.internode_base = _nominal_internode(profile, parent, parent_phytomer_index + 1)
        else:
            axis.internode_base = (_nominal_internode(profile, parent, parent_phytomer_index)
                                   * level.start_length_scale
                                   * factor(growth_maps.START_LENGTH))
        if level.upper_side_up and not is_fork:
            # Upper Side Up: the frame starts at the branch's top -- world up
            # across its heading -- so every branch of a whorl lays its own
            # branches out the same way round (a branch straight up or down
            # has no top: it keeps its transported frame).
            heading = axis.base_direction
            up = _UP - heading * float(np.dot(_UP, heading))
            norm = float(np.linalg.norm(up))
            if norm > 1e-6:
                axis.base_reference = up / norm
        roll = level.start_roll_deg * factor(growth_maps.ROLL)
        if level.start_roll_variation_deg > 0.0 and not is_fork:
            # Roll Variation: this stem / branch's own frame twist.
            roll += level.start_roll_variation_deg * _signed(_level_seed(seed, level),
                                                             _T_START_ROLL, axis.id)
        if roll != 0.0 and not is_fork:
            # Start Roll: the axis's own frame twisted about its heading, which
            # is where its leaves and its own branches start round it.
            axis.base_reference = rotate_about(axis.base_reference, axis.base_direction,
                                               roll)
        if not is_fork:
            # Growth %: the part of its Steps this axis builds (a fork / relay
            # takes its parent chain's share instead, `node_reach`).
            axis.growth = min(1.0, max(_MIN_GROWTH,
                                       level.growth * factor(growth_maps.GROWTH)))
        if mapped:
            axis.map_width = factor(growth_maps.START_WIDTH)
    if profile.maps:
        # The plant-level per-axis map (gravitropism), fixed at birth.
        _assign_axis_maps(plant, profile, seed, axis)
    axis.apex = Bud(
        id=plant.new_bud_id(),
        position_kind=BudPosition.TERMINAL,
        birth_step=plant.step,
        axis_id=axis.id,
        phytomer_index=-1,
        state=BudState.ACTIVE,
    )
    return plant.add_axis(axis)


def _emerge_basal_stems(plant: Plant, profile: MorphologyProfile, seed: int) -> None:
    """The stems level (level 0): its Whorl is the stem count, all emerging at
    once, filling the clump's crown as a phyllotactic disc (Vogel 1979, the
    sunflower-head construction): stem n turned n golden angles round the base
    (or n Whorl Angles, when the level sets one) and `basal_spread` x
    sqrt((n + 1/2) / count) from the centre, so the feet spread evenly over the
    crown, centre to edge, rather than standing on its rim (a single stem
    stands at the centre). Each leaves vertical by the level's Start Pitch (the
    splay) outward from where it stands, turned sideways by Start Yaw; Start
    Roll twists its frame (`_new_axis`)."""
    if plant.basal_axis_ids:
        return
    if not profile.levels:
        _new_axis(plant, profile, seed, 0, np.zeros(3), _UP.copy())
        return
    # Every stems level (a level on the plant) puts up its own stems, in list
    # order; a single one is exactly the stems level of a chain.
    for ri, li in enumerate(profile.level_children(-1)):
        level = profile.levels[li]
        count = max(1, int(level.whorl))
        step = level.whorl_angle_deg if level.whorl_angle_deg > 0.0 else _GOLDEN_DEG
        for n in range(count):
            azimuth = np.radians((n * step) % 360.0)
            radial = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            position = (radial * profile.basal_spread * np.sqrt((n + 0.5) / count)
                        if count > 1 else np.zeros(3))
            if count > 1 and profile.basal_spread_variation > 0.0:
                # Spread Variation: the foot off its place in the disc, by up to
                # the variation x the spread in X and in Y, drawn per stem on the
                # stems level's seed. It still splays the way it faces.
                reach = profile.basal_spread_variation * profile.basal_spread
                own = _level_seed(seed, level)
                position = position + reach * np.array(
                    [_signed(own, _T_BASE_X, li * 1024 + n),
                     _signed(own, _T_BASE_Y, li * 1024 + n), 0.0])
            if level.masks and not _level_mask_keeps(
                    profile, seed, level,
                    _stem_channels(plant, profile, seed, n, count, position, ri), n):
                continue  # the stems level's Masks leave this stem out
            pitch, yaw = level.start_pitch_deg, level.start_yaw_deg
            # The stems level's Pitch / Yaw maps, read at each stem's foot (as a
            # branch level's are per bud), before its variation -- as there.
            if growth_maps.level_has_map(level, growth_maps.PITCH):
                pitch *= growth_maps.stack_factor(level.maps, growth_maps.PITCH, _stem_channels(
                    plant, profile, seed, n, count, position, ri, growth_maps.PITCH))
            if yaw != 0.0 and growth_maps.level_has_map(level, growth_maps.YAW):
                yaw *= growth_maps.stack_factor(level.maps, growth_maps.YAW, _stem_channels(
                    plant, profile, seed, n, count, position, ri, growth_maps.YAW))
            if level.start_pitch_variation_deg > 0.0 or level.start_yaw_variation_deg > 0.0:
                # Splay / Yaw Variation: each stem its own lean out of the clump.
                own = _level_seed(seed, level)
                pitch += level.start_pitch_variation_deg * _signed(own, _T_START_PITCH,
                                                                   li * 1024 + n)
                yaw += level.start_yaw_variation_deg * _signed(own, _T_START_YAW,
                                                               li * 1024 + n)
            direction = _launch(_UP.copy(), radial, pitch, yaw)
            stem = _new_axis(plant, profile, seed, 0, position, direction, level=li)
            delay = float(level.emergence_delay)
            if delay and growth_maps.level_has_map(level, growth_maps.EMERGENCE_DELAY):
                delay *= growth_maps.stack_factor(
                    level.maps, growth_maps.EMERGENCE_DELAY, _stem_channels(
                        plant, profile, seed, n, count, position, ri,
                        growth_maps.EMERGENCE_DELAY))
            _add_emergence_delay(stem, delay)


def _stem_channels(plant: Plant, profile: MorphologyProfile, seed: int, n: int,
                   count: int, position: np.ndarray, root: int = 0,
                   target: str = growth_maps.GROWTH) -> _MapChannels:
    """A stem-to-be's channels for the stems level's Masks (and its Pitch / Yaw
    maps, `target`): at the ground, its place in the clump as `node_index` (0 ..
    1 over the stems), its slot as the random draw's id."""
    return _MapChannels(
        plant, profile, seed, target, None,
        along=0.0, node_index=lambda: n / max(count - 1, 1),
        birth_step=plant.step, kind=_MAP_KIND_AXIS, ident=-(n + 1) - root * 4096,
        position=lambda: position,
    )


def _extend_apices(plant: Plant, profile: MorphologyProfile, seed: int,
                   environment=None) -> None:
    floral = profile.vegetativeness().apex_is_floral()
    for axis in plant.growing_axes():
        if plant.step < axis.birth_step:
            continue
        # A vegetative axis builds its level's Steps and no more: a determinate
        # one then waits for the plant to flower (`_terminate_determinate_axes`
        # opens its inflorescence), an indeterminate one simply ends.
        target = None
        level = _level(profile, axis) if axis.flowering_from < 0 else None
        if level is not None:
            target = _built_nodes(profile, axis)
            if len(axis.phytomers) >= target:
                # A sympodial unit that has built its nodes waits for
                # `_end_units` to relay and end it. Any other tip that will not
                # flower grows on as the next level down (`_continue_tip`) --
                # at once, not held for the plant's flowering as a tip that
                # may flower is.
                if level.relays:
                    continue
                flowers = floral and _tip_flowers(profile, axis)
                if not flowers and not _continue_tip(plant, profile, seed, axis) \
                        and not floral:
                    axis.apex.state = BudState.ABORTED
                continue
        axis.growth_credit += _growth_rate(profile, axis.order)
        # `while`, not `if`: an axis whose rate exceeds one node per step
        # (plastochron under 1) should build several, not silently cap at one.
        while axis.growth_credit >= 1.0:
            if target is not None and len(axis.phytomers) >= target:
                axis.growth_credit = 0.0
                break
            axis.growth_credit -= 1.0
            if _add_phytomer(plant, axis, profile, seed, environment) is None:
                # Ran into something solid. The apex stops for good -- the axis
                # keeps every node it built, and its dormant buds are released
                # from its apical dominance exactly as they would be on any
                # other axis that stopped growing, which is why a plant that
                # meets an obstacle bushes out below it.
                axis.apex.state = BudState.ABORTED
                axis.growth_credit = 0.0
                break


def _release_buds(plant: Plant, profile: MorphologyProfile, seed: int, plant_age: float,
                  environment=None, day: float | None = None) -> None:
    """Every node branches: a bud
    on an axis whose level has a next level grows out as that level's branch
    as soon as its node exists. A bud on the deepest level has no branch to
    become -- it becomes a flower where the lateral meristem has reached floral
    identity and the plant is flowering, else it stays dormant. Thinning is not
    part of the first cut: masks and growth maps do it meanwhile.

    `environment` is unused since density went (light only ever scaled it);
    kept in the signature the loop calls with."""
    del environment
    # The calendar's bloom date on the day path, else the developmental onset
    # curve (byte-identical off the day path). A lateral bud that cannot branch
    # becomes a flower only once the plant is flowering, same gate the apex uses.
    flowering = profile.flowering_permission(plant_age, day)
    veg = profile.vegetativeness()

    def held(bud, reason: str, value: float, limit: float) -> None:
        """Record which gate stopped this bud, for the inspector. Diagnostic
        only -- it decides nothing."""
        bud.blocked_by = reason
        bud.blocked_value = float(value)
        bud.blocked_limit = float(limit)

    for bud in plant.dormant_buds():
        axis = plant.axes[bud.axis_id]
        if 0 <= bud.phytomer_index < len(axis.phytomers) and \
                axis.phytomers[bud.phytomer_index].birth_step > plant.step:
            continue                    # its node has not started developing yet
        child_index = _bud_level(profile, axis, bud)
        child = profile.levels[child_index] if child_index >= 0 else None
        if child is None:
            if _is_inflorescence_axis(axis) and axis.flowering_from < 0:
                # A stalk's bare nodes below its inflorescence: a peduncle
                # bears no flowers of its own, whatever the plant's laterals do.
                held(bud, BudBlock.ORDER, axis.order + 1, len(profile.levels) - 1)
                continue
            if veg.lateral_is_floral(bud.age_steps(plant.step)) \
                    and flowering >= bud.release_threshold:
                _make_flower_bud(plant, profile, seed, axis, bud)
                held(bud, BudBlock.NONE, 0.0, 0.0)
            else:
                held(bud, BudBlock.ORDER, axis.order + 1, len(profile.levels) - 1)
            continue
        if bud.masked:
            held(bud, BudBlock.MASK, 0.0, 0.0)
            continue
        if child.masks and not _level_mask_keeps(
                profile, seed, child,
                _bud_channels(plant, profile, seed, growth_maps.PITCH, axis, bud),
                bud.phytomer_index, bud):
            # The level's Masks reject this branch: decided once, for good.
            bud.masked = True
            held(bud, BudBlock.MASK, 0.0, 0.0)
            continue
        held(bud, BudBlock.NONE, 0.0, 0.0)
        yaw = child.start_yaw_deg
        if yaw != 0.0 and growth_maps.level_has_map(child, growth_maps.YAW):
            yaw *= _bud_map_factor(plant, profile, seed, growth_maps.YAW, axis, bud,
                                   stack=child.maps)
        if child.start_yaw_variation_deg > 0.0:
            # Yaw Variation: this branch swung its own way sideways.
            yaw += child.start_yaw_variation_deg * _signed(_level_seed(seed, child),
                                                           _T_START_YAW, bud.id)
        shoot = _release_as_shoot(plant, profile, seed, axis, bud, axis.order + 1,
                                  yaw_deg=yaw, level=child_index)
        delay = float(child.emergence_delay)
        if delay and growth_maps.level_has_map(child, growth_maps.EMERGENCE_DELAY):
            delay *= _bud_map_factor(plant, profile, seed, growth_maps.EMERGENCE_DELAY,
                                     axis, bud, stack=child.maps)
        _add_emergence_delay(shoot, delay)


def _chance_forks(plant: Plant, profile: MorphologyProfile, seed: int) -> None:
    """A level's Fork: each new node of a growing vegetative tip rolls the
    level's Fork Probability (x its Fork Probability map) and, on a hit,
    the tip splits there into Fork Branches continuations.

    Each node is rolled exactly once (`Axis.fork_checked`), on a hash of the
    axis and node, so the outcome is order-free and byte-deterministic. Not on
    an inflorescence branch, not in the axis's last two nodes (the children
    need at least two of the remaining budget each), and never on a level with
    Fork off or Probability 0 -- an exact no-op that draws nothing."""
    if not any(level.forks and level.fork_probability > 0.0 for level in profile.levels):
        return
    for axis in list(plant.growing_axes()):
        level = _level(profile, axis)
        if level is None or not level.forks or level.fork_probability <= 0.0:
            continue
        mapped = growth_maps.level_has_map(level, growth_maps.FORK_PROBABILITY)
        if axis.apex is None or axis.apex.state != BudState.ACTIVE or axis.flowering_from >= 0:
            continue
        count = len(axis.phytomers)
        start = max(axis.fork_checked, 2)  # never right at a stem's base
        axis.fork_checked = max(axis.fork_checked, count)
        if start >= count:
            continue
        full = _vegetative_nodes(profile, axis)
        target = _built_nodes(profile, axis)
        expected = max(full - 1, 1)
        for k in range(start, count):
            if k >= target - 2:
                break
            chance = level.fork_probability
            ident = axis.id * 4096 + k
            if mapped:
                chance *= _node_map_factor(plant, profile, seed,
                                           growth_maps.FORK_PROBABILITY,
                                           axis, k + 1, min(1.0, k / expected),
                                           _MAP_KIND_NODE, ident, stack=level.maps)
            if _unit(_level_seed(seed, level), _T_FORK_CHANCE, ident) < chance:
                if target - count >= 2:
                    reach = _vegetative_reach(profile, axis)
                    _chance_fork(plant, profile, seed, axis, level, full - count,
                                 reach - count if reach < full else -1.0)
                break


def _chance_fork(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                 level: BranchLevel, remaining: int, reach: float = -1.0) -> None:
    """Split a growing tip into the level's Fork Branches continuations, spread
    evenly round it -- the ring turned Fork Whorl per fork generation -- each
    leaving half the Fork Spread. The children stay on the parent's level (a
    fork continues the axis, it does not subordinate it), go one
    `structural_depth` deeper, and share the parent's remaining node budget.

    An Unequal Fork (pseudomonopodial) makes the first child the leader: it
    keeps the whole budget and deviates only half x (1 - Balance). The others
    leave half x (1 + Balance) with budget and base width x (1 - Balance / 2),
    so the leaders chain into what reads as one trunk (vigour and direction
    from one control).

    `reach` is what Growth % leaves the chain to build of `remaining` (-1: all
    of it); each child builds its share of it."""
    tip = axis.phytomers[-1]
    depth = axis.structural_depth + 1
    n = max(2, int(level.fork_branches))
    spread = level.fork_spread_deg
    if spread != 0.0 and growth_maps.level_has_map(level, growth_maps.FORK_SPREAD):
        k = len(axis.phytomers) - 1
        spread *= _node_map_factor(plant, profile, seed, growth_maps.FORK_SPREAD, axis, k + 1,
                                   tip.factor, _MAP_KIND_NODE, axis.id * 4096 + k,
                                   stack=level.maps, birth_step=tip.birth_step)
    half = spread / 2.0
    ring = rotate_about(tip.reference, tip.direction,
                        level.fork_whorl_deg * (max(depth, 0) + axis.order))
    axis.apex.state = BudState.ABORTED
    # The forks carry the tip on: the forking node grows no side branch as
    # well (its branch buds are spent -- `_chance_forks` runs before
    # `_release_buds`, so none has broken yet). Leaf-axil buds are untouched.
    for bud in tip.buds:
        if bud.state == BudState.DORMANT and bud.level >= 0:
            bud.state = BudState.ABORTED
    unequal = level.growth_pattern == GrowthPattern.PSEUDO_FORK
    balance = min(1.0, max(0.0, level.fork_balance)) if unequal else 0.0
    weak = 1.0 - 0.5 * balance
    for i in range(n):
        normal = rotate_about(ring, tip.direction, 360.0 * i / n)
        leader = not unequal or i == 0
        tilt = half
        if unequal:
            tilt = half * ((1.0 - balance) if leader else (1.0 + balance))
        launch = rotate_about(tip.direction, normal, tilt)
        child = _fork_child(plant, profile, seed, axis, tip, launch, axis.order, depth)
        if leader:
            child.node_budget = max(2, remaining)
            child.node_reach = reach
        else:
            child.node_budget = max(2, int(round(remaining * weak)))
            # The chain's cut, as a share of its own (rounded) budget: scaled
            # apart from it, a budget rounded down under its reach (10.5 -> 10
            # under 10.19) read as whole, and a cut runner's fork grew on.
            child.node_reach = (child.node_budget * reach / remaining if reach >= 0.0
                                else -1.0)
            child.vigour = weak


def _end_units(plant: Plant, profile: MorphologyProfile, seed: int,
               environment=None) -> None:
    """The sympodial levels: a unit that has
    built its nodes ends -- its relays launch from its last node, then its apex
    flowers or aborts (the level's Unit End). Relays continue the level with
    what is left of the chain's budget, so the level's Steps is still the
    length of the whole chain. A no-op, drawing nothing, on a profile with no
    relaying level."""
    if not any(level.relays for level in profile.levels):
        return
    for axis in list(plant.growing_axes()):
        if axis.unit_ended or axis.flowering_from >= 0 or not axis.phytomers:
            continue
        level = _level(profile, axis)
        if level is None or not level.relays:
            continue
        budget = _chain_budget(level, axis)
        unit = _unit_nodes(level, budget)
        if len(axis.phytomers) < _built_nodes(profile, axis):
            continue
        axis.unit_ended = True
        # Growth % may end the chain inside this unit: then it relays no more.
        reach = _chain_reach(level, axis)
        if budget - unit >= 1 and reach > unit:
            _relay_unit(plant, profile, seed, axis, level, budget - unit,
                        reach - unit if reach < budget else -1.0)
        elif (reach >= budget and not _tip_flowers(profile, axis)
              and _continue_tip(plant, profile, seed, axis)):
            # The chain's last unit, run to its whole budget, its tip not
            # flowering: it grows on as the next level down.
            continue
        _end_unit(plant, profile, seed, axis, level, environment)


def _relay_unit(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                level: BranchLevel, remaining: int, reach: float = -1.0) -> None:
    """A finished unit's relays, from its last vegetative node: spread evenly
    round it (the ring turned Whorl per generation), each leaving half the
    Spread -- or, for the Aligned Relay, one relay straight on in line while
    the spent apex is pushed half the Spread aside. Each is a continuation of
    the level (a fork child: same level and order, one `structural_depth`
    deeper) carrying the chain's remaining budget (and Growth %'s `reach` of
    it, -1 for all)."""
    tip = axis.phytomers[-1]
    depth = axis.structural_depth + 1
    spread = level.fork_spread_deg
    if spread != 0.0 and growth_maps.level_has_map(level, growth_maps.FORK_SPREAD):
        k = len(axis.phytomers) - 1
        spread *= _node_map_factor(plant, profile, seed, growth_maps.FORK_SPREAD, axis, k + 1,
                                   tip.factor, _MAP_KIND_NODE, axis.id * 4096 + k,
                                   stack=level.maps, birth_step=tip.birth_step)
    half = spread / 2.0
    ring = rotate_about(tip.reference, tip.direction,
                        level.fork_whorl_deg * (max(depth, 0) + axis.order))
    if level.growth_pattern == GrowthPattern.PSEUDO_RELAY:
        launches = [tip.direction.copy()]
        axis.displace_normal, axis.displace_deg = ring, half
    else:
        n = level.relays
        launches = [rotate_about(tip.direction, rotate_about(ring, tip.direction, 360.0 * i / n),
                                 half) for i in range(n)]
    for launch in launches:
        child = _fork_child(plant, profile, seed, axis, tip, launch, axis.order, depth)
        child.node_budget = remaining
        child.node_reach = reach
        # A relay takes over as the leader, so it keeps the chain's gravitropic
        # setpoint rather than holding its launch angle as a lateral does: it
        # kinks off, then rights itself -- the zig-zag of a sympodial stem.
        child.setpoint_tilt_deg = axis.setpoint_tilt_deg


def _end_unit(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
              level: BranchLevel, environment=None) -> None:
    """A unit's apex ends: with a Terminal inflorescence it becomes that
    inflorescence at once -- not held for the plant's flowering onset, because
    a sympodial unit's flower *is* its end; bloom timing still opens it.
    Without one (flowers only at the nodes, or none) the tip dies back, as
    `_begin_inflorescence` stops a stem."""
    del level
    if Attachment.TERMINAL not in _attachments(profile, axis):
        axis.apex.state = BudState.ABORTED
        return
    _begin_inflorescence(plant, profile, seed, axis)
    organ = axis.terminal_organ
    if organ is not None and axis.displace_deg != 0.0 and axis.displace_normal is not None:
        # An Aligned Relay's spent apex capped by one flower: it leans aside
        # the same way a flowering region's first node does (`_add_phytomer`),
        # as a divergence and elevation in the tip's frame.
        tip = axis.phytomers[-1]
        lean = _cross(axis.displace_normal, tip.direction)
        organ.divergence_deg = float(np.degrees(np.arctan2(
            np.dot(lean, _cross(tip.direction, tip.reference)), np.dot(lean, tip.reference))))
        organ.elevation_deg = 90.0 - axis.displace_deg
    _preform_region(plant, profile, seed, axis, environment)


def _tip_flowers(profile: MorphologyProfile, axis: Axis) -> bool:
    """Whether this axis's tip becomes an inflorescence: an inflorescence
    axis's always does; a vegetative tip when one placed on its level claims
    it as Terminal and makes flowers (`_begin_inflorescence`)."""
    if _is_inflorescence_axis(axis):
        return True
    k = _claimant(profile, axis, Attachment.TERMINAL)
    return k is not None and profile.inflorescence_at(k, axis.order).topology != Topology.NONE


def _continue_tip(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis) -> bool:
    """Tip continuation: a finished stem /
    branch whose tip does not flower grows on, in line, as its level's first
    child level -- which continues in turn -- so every line of the plant ends
    in its lowest level, twig-thin, not at its own last node. The apex stops
    and the continuation takes over; True if it did. A level with nothing
    growing on it, or an axis cut short by Growth %, ends as before."""
    kids = _child_levels(profile, axis)
    if not kids or not axis.phytomers or _is_inflorescence_axis(axis):
        return False
    # Cut short by Growth %: compared on the fractional reach, not the nodes
    # built -- the partly-grown last node counts as built, so a fork off a cut
    # chain (1.43 of its 2 nodes) read as whole and grew on.
    if _vegetative_reach(profile, axis) < _vegetative_nodes(profile, axis) - 1e-9:
        return False
    level = _level(profile, axis)
    if level is not None and level.relays:
        # A sympodial unit continues only as its chain's last: one that relays
        # carries the chain on itself (`_end_units`).
        budget = _chain_budget(level, axis)
        unit = _unit_nodes(level, budget)
        if budget - unit >= 1 and _chain_reach(level, axis) > unit:
            return False
    tip = axis.phytomers[-1]
    child = _new_axis(
        plant, profile, seed, axis.order + 1,
        tip.tip_position.copy(),
        tip.direction.copy(),
        base_reference=tip.reference.copy(),
        parent_axis_id=axis.id,
        parent_phytomer_index=len(axis.phytomers) - 1,
        level=kids[0],
    )
    child.is_continuation = True
    # It leaves in line but holds its own level's angle: the level's Start
    # Pitch, measured from straight up -- so a trunk's top leans out to a
    # limb's angle and a limb's end turns to a branch's, instead of running on
    # dead straight. From vertical, not added onto the parent's: added, the
    # turns stacked down the levels and every line ended pointing down (the
    # Laburnum wept whatever its stiffness).
    child.setpoint_tilt_deg = min(180.0, max(0.0, profile.levels[kids[0]].start_pitch_deg))
    axis.apex.state = BudState.ABORTED
    return True


def _fork_child(plant, profile, seed, axis, tip, launch, order, structural_depth):
    launch = launch / max(float(np.linalg.norm(launch)), _EPS)
    return _new_axis(
        plant, profile, seed, order,
        tip.tip_position.copy(),
        launch,
        base_reference=parallel_transport(tip.reference, tip.direction, launch),
        parent_axis_id=axis.id,
        parent_phytomer_index=len(axis.phytomers) - 1,
        structural_depth=structural_depth,
        is_fork=True,
        level=axis.level,
    )


def _make_flower_bud(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis, bud: Bud,
                     claim: int | None = None) -> None:
    """A flower that is the bud itself, of inflorescence `claim` (the
    axis's own when None)."""
    bud.state = BudState.RELEASED
    bud.fate = BudFate.FLOWER
    bud.released_step = plant.step
    phytomer = axis.phytomers[bud.phytomer_index]
    claim = _inf_index(profile, axis) if claim is None else claim
    organ_id = plant.new_organ_id()
    phytomer.organs.append(Organ(
        id=organ_id,
        kind=OrganKind.FLOWER,
        birth_step=plant.step,
        inflorescence=claim,
        final_scale=_flower_scale(profile, seed, profile.inflorescence_at(claim, axis.order),
                                  organ_id),
        divergence_deg=bud.divergence_deg,
        elevation_deg=bud.elevation_deg,
        along_internode=bud.along_internode,
    ))


def _release_as_shoot(
    plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis, bud: Bud, child_order: int,
    structural_depth: int = -1, inflorescence_scale: float = 1.0, yaw_deg: float = 0.0,
    level: int = -1, inflorescence: int = -1,
) -> Axis:
    parent_phytomer = axis.phytomers[bud.phytomer_index]
    direction = radial_direction(
        parent_phytomer.direction, parent_phytomer.reference, bud.divergence_deg, bud.elevation_deg
    )
    if yaw_deg != 0.0:
        # Start Yaw (vegetative branches -- the one caller that passes it): the
        # launch turned sideways at its slot, round the parent.
        radial = radial_direction(parent_phytomer.direction, parent_phytomer.reference,
                                  bud.divergence_deg, 0.0)
        direction = _turn_sideways(direction, parent_phytomer.direction, radial, yaw_deg)
    # An internodal bud breaks partway up its own internode rather than at the
    # node above it -- the one thing that makes it internodal. Every other
    # meristem leaves `along_internode` at 1 and lands on the node, exactly as
    # before.
    origin = parent_phytomer.tip_position
    if bud.along_internode < 1.0:
        origin = (parent_phytomer.base_position
                  + (parent_phytomer.tip_position - parent_phytomer.base_position)
                  * bud.along_internode)
    elif bud.along_internode > 1.0:
        # Slid up past its node (a level's Slide): partway up the internode
        # above, or on along this one's heading while that is not grown yet --
        # `resolve_geometry` puts it on the real one.
        nxt = (axis.phytomers[bud.phytomer_index + 1]
               if bud.phytomer_index + 1 < len(axis.phytomers) else parent_phytomer)
        origin = (parent_phytomer.tip_position
                  + (nxt.tip_position - nxt.base_position) * (bud.along_internode - 1.0))
    child = _new_axis(
        plant,
        profile,
        seed,
        child_order,
        origin.copy(),
        direction,
        base_reference=parallel_transport(parent_phytomer.reference, parent_phytomer.direction,
                                          direction),
        parent_axis_id=axis.id,
        parent_along_internode=bud.along_internode,
        parent_phytomer_index=bud.phytomer_index,
        structural_depth=structural_depth,
        inflorescence_scale=inflorescence_scale,
        level=level,
        inflorescence=inflorescence,
    )
    bud.state = BudState.RELEASED
    bud.fate = BudFate.SHOOT
    bud.child_axis_id = child.id
    bud.released_step = plant.step
    return child


_MAX_TURN_PER_NODE_DEG = _MAX_ARC_FACET_DEG
"""The most an aimed inflorescence's setpoint turns per node on its way to its
orientation: as many nodes as the turn needs, each turning no more than the
facet a smooth arc may have (`physics._MAX_ARC_FACET_DEG`) -- so a bigger
swing takes more nodes and the bend is smooth whatever the spacing. Faster
only when the inflorescence is too short to arrive by its midpoint
(`_turn_deadline`): a 7-flower pendulous raceme at this rate alone got 49 of
its 180 degrees and never hung.

It replaced a fixed span of 1.5 nodes, which put the whole turn inside one
internode: on a raceme's short flowering nodes (~4.5 mm) a branch's raceme
swung 48 degrees in 4 mm, split into 0.5 mm facets -- smooth facets, but a
kink. A rate, not an exponential approach, so it reaches the target and then
holds (an approach with no end curled the inflorescence -- see
`_update_orientation`)."""


def _orientation_target(plant: Plant, profile: MorphologyProfile,
                        axis: Axis) -> float | None:
    """The tilt from vertical this axis prefers, or None if nothing pulls it.

    A tilt rather than a full direction because the azimuth is already right:
    the growth loop's gravitropism turns an axis in its own vertical plane, so
    a target tilt preserves whatever azimuth the bud had -- radially outward
    from its own stem. That is exactly what `HORIZONTAL` wants, and it keeps
    the phyllotaxis instead of sending every inflorescence to one compass
    point.
    """
    inf = _inf(profile, axis)
    if inf.orientation_strength <= 0.0:
        return None
    mode = inf.orientation_mode

    if mode == OrientationMode.ERECT:
        preferred = 0.0
    elif mode == OrientationMode.HORIZONTAL:
        preferred = 90.0
    elif mode == OrientationMode.PENDULOUS:
        preferred = 180.0
    else:
        # PARENT_ALIGNED and CUSTOM are both stated against the parent, and
        # both re-read it every step so the inflorescence swings when the
        # branch carrying it bends. The parent is what carries it: for a stalk
        # or inflorescence branch the axis it leaves, at the node it leaves;
        # for a terminal inflorescence the stem or branch it ends -- its own
        # last leafy node -- not the axis that branch grew from (which aligned
        # a branch's raceme with the main stem).
        if axis.structural_depth < 1 or axis.is_fork:
            upto = axis.flowering_from if axis.flowering_from >= 0 else len(axis.phytomers)
            leafy = axis.phytomers[:upto]
            heading = leafy[-1].direction if leafy else axis.base_direction
        else:
            parent = plant.axes.get(axis.parent_axis_id)
            if parent is None or not parent.phytomers:
                return None
            index = min(max(axis.parent_phytomer_index, 0), len(parent.phytomers) - 1)
            heading = parent.phytomers[index].direction
        preferred = _tilt_from_vertical(heading)
        if mode == OrientationMode.CUSTOM:
            preferred += inf.orientation_deg

    preferred = min(180.0, max(0.0, preferred))
    # **A preference, not a constraint.** The pull is a blend from the
    # direction the axis actually launched at, so even at full strength the
    # result carries its structural history rather than snapping onto an exact
    # world axis -- and the physics bend then acts on top. Real organs are
    # shaped by their support and their own growth as well as by gravity.
    launch = axis.orientation_from_deg
    strength = min(1.0, max(0.0, inf.orientation_strength))
    return launch + (preferred - launch) * strength


def _update_orientation(plant: Plant, profile: MorphologyProfile, seed: int,
                        axis: Axis) -> None:
    """Move an axis's gravitropic setpoint, for the modes that do not fix it
    once at birth. Called after each phytomer is appended.

    **The setpoint is what curves, not the correction.** Every mode here tracks
    its setpoint hard; what makes the shape is where that setpoint is on any
    given node. Softening the correction instead gives an exponential approach
    with no end, so the turn carries on down the whole inflorescence and curls
    it -- measured chord/arc 0.86 against 1.00, with the early flowers swinging
    back toward the stem. That was Steer To Angle, and it is why it went.

    Two drivers, one shape. `WORLD_ANGLE` and `FOLLOW_PARENT` turn onto their
    angle at most a smooth arc's facet per node (`_MAX_TURN_PER_NODE_DEG`), so
    the stalk arcs over across as many nodes as the turn needs and then
    holds. `DYNAMIC` is driven by flower load instead, so it bows as it
    fills rather than on a schedule.
    """
    if axis.orientation_target_deg < 0.0:
        # **Only an axis that has actually been aimed.** `_aim_inflorescence`
        # sets this, and it is called on the axis carrying the inflorescence --
        # never on the vegetative stems that bear one. Without the test this
        # read the inflorescence's angle for the trunk as well and dragged the
        # whole plant over: measured, a laburnum's main stem curled into a
        # shepherd's crook and its height fell 0.561 -> 0.317 m as the arc was
        # widened.
        return
    inf = _inf(profile, axis)
    if inf.orientation_strength <= 0.0:
        return
    target = _orientation_target(plant, profile, axis)
    if target is None:
        return

    if not inf.orientation_load_driven:
        # Toward the target by at most a smooth arc's facet per node -- more
        # only where the inflorescence is too short to get there by its
        # midpoint otherwise -- from the node about to be built
        # (`_aim_inflorescence` takes the first step before the first phytomer
        # exists, so the bend starts at the join).
        current = axis.setpoint_tilt_deg
        left = _turn_deadline(profile, seed, axis) - (len(axis.phytomers) + 1)
        rate = max(_MAX_TURN_PER_NODE_DEG,
                   abs(target - current) / left if left >= 1.0 else 180.0)
        axis.setpoint_tilt_deg = current + max(-rate, min(rate, target - current))
        return
    # A schedule, not a direction: the same target, reached as the axis fills
    # rather than over a run of nodes at its base.
    if axis.flowering_from < 0:
        return
    built = len(axis.phytomers) - axis.flowering_from
    progress = min(1.0, built / max(1, _flower_nodes(profile, seed, axis)))

    # `t * (2 - t)`: the tip-loaded cantilever, the same profile `_sag` bends a
    # flower stalk with. Curvature is greatest **at the attachment** and tapers
    # to nothing at the far end, which is what a peduncle giving way under its
    # own load actually does.
    #
    # Smoothstep was tried first and is wrong here, because its slope is zero
    # at *both* ends: the peduncle left the branch dead straight, held stiff
    # for several nodes, and only then turned. Reported as staying straight
    # before it starts turning, and as widening the arc moving only the
    # flowers -- with the base pinned, everything that moved was downstream of
    # it. This profile starts turning on the first segment.
    eased = progress * (2.0 - progress)
    axis.setpoint_tilt_deg = (axis.orientation_from_deg
                              + (target - axis.orientation_from_deg) * eased)


def _turn_deadline(profile: MorphologyProfile, seed: int, axis: Axis) -> float:
    """The node by which an aimed inflorescence has reached its orientation:
    its midpoint, so its outer half holds the angle (a pendulous raceme hangs
    straight) -- a stalk's whole length counted, peduncle and flowers; a
    terminal one's flowering region."""
    if axis.flowering_from >= 0:
        return axis.flowering_from + _flower_nodes(profile, seed, axis) / 2.0
    if axis.structural_depth >= 1 and not axis.is_fork:
        return _branch_total_nodes(profile, seed, axis) / 2.0
    return len(axis.phytomers) + _flower_nodes(profile, seed, axis) / 2.0


def _aim_inflorescence(plant: Plant, profile: MorphologyProfile, seed: int,
                       inf, axis: Axis) -> None:
    """Point an axis the way its inflorescence says -- see `OrientationMode`.

    Called in both places an inflorescence gets an axis: when a terminal one
    converts the axis it is already on, and when an axillary one is released
    onto an axis of its own. One helper so the two cannot drift apart about
    what a mode means.

    `INHERITED` writes nothing at all, which is why every existing plant is
    untouched: the axis keeps the launch angle `_new_axis` gave it.
    """
    if axis.orientation_target_deg >= 0.0:
        # Already aimed. An axillary inflorescence passes here twice -- once
        # when its stalk is released, and again when its flowering region
        # opens -- and re-aiming restarts the arc from wherever it had got to,
        # which showed up as a 50 degree jump at exactly the node the flowers
        # begin on.
        return
    setpoint, gravitropism = inf.orientation()
    if setpoint is not None:
        axis.setpoint_tilt_deg = setpoint
    if gravitropism is not None:
        axis.gravitropism = gravitropism
    if inf.orientation_strength > 0.0:
        # Nothing is aimed once. `_update_orientation` moves the setpoint every
        # node, easing onto the preference or following the load, and it needs
        # the launch angle to start that arc from -- and to blend against, so
        # the pull is a preference from where the axis actually left its stem
        # rather than an absolute set.
        axis.orientation_from_deg = axis.setpoint_tilt_deg
        # Non-negative marks this axis as aimed, which is what stops
        # `_update_orientation` reading an inflorescence's angle for the
        # vegetative stems that merely bear one.
        axis.orientation_target_deg = 0.0
        # Take the first eased step now, before the first phytomer exists, so
        # the segment leaving the branch is already part of the arc.
        _update_orientation(plant, profile, seed, axis)


# Which attachments arise on the node at the moment it is made, and are
# therefore released by `_release_node_inflorescences`. The rest each need
# their own pass: the old-wood three because they break seasons later from wood
# that already exists, and BASAL because it arises below every node there is.
# A TUPLE, because the order it is iterated in decides which node-borne mode
# claims a node's real buds when two are selected -- a set would make that
# depend on hash order and the plant would stop being reproducible.
_NODE_BORNE = (Attachment.AXILLARY, Attachment.INTERNODAL, Attachment.LEAF_OPPOSED)
# A frozenset, because this one is only ever intersected with the selection.
_OLD_WOOD = frozenset({Attachment.CAULIFLOROUS, Attachment.RAMIFLOROUS,
                       Attachment.ADVENTITIOUS})
# How wide a fan `attachment_per_node` spreads a cluster over, in degrees. A
# cluster at one axil is several stalks leaving the same point at slightly
# different angles; stacking them on one azimuth would put every stalk inside
# the last one.
_CLUSTER_FAN_DEG = 50.0


def _cluster_buds(plant: Plant, inf, phytomer, bud: Bud) -> list[Bud]:
    """`bud` plus however many extra meristems `attachment_per_node` asks for,
    fanned evenly about its own azimuth.

    A node carries only as many real buds as its phyllotaxis gave it, so a
    cluster of three at one axil has nothing already there to release and the
    extras have to be synthesised. Returning a list including the original --
    rather than releasing it separately -- is what keeps the default of 1 a
    byte-for-byte no-op: one bud in, one bud out, no fan applied.
    """
    count = max(1, int(inf.attachment_per_node))
    if count == 1:
        return [bud]
    members = [bud]
    # Centre the fan on the original azimuth so the cluster straddles the axil
    # the phyllotaxis chose, rather than growing off to one side of it.
    step = _CLUSTER_FAN_DEG / (count - 1)
    start = bud.divergence_deg - _CLUSTER_FAN_DEG * 0.5
    bud.divergence_deg = start
    for i in range(1, count):
        extra = Bud(
            id=plant.new_bud_id(),
            position_kind=bud.position_kind,
            birth_step=plant.step,
            axis_id=bud.axis_id,
            phytomer_index=bud.phytomer_index,
            divergence_deg=start + step * i,
            elevation_deg=bud.elevation_deg,
            along_internode=bud.along_internode,
        )
        phytomer.buds.append(extra)
        members.append(extra)
    return members


def _release_node_inflorescences(
    plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis, index: int,
    along: float,
) -> None:
    """Hang an inflorescence at this vegetative node, on its own short stalk.

    The generalisation of what a catkin does, for the kinds that need real
    flowers on real stems rather than a swept tube. What is released is an
    **inflorescence axis**, not a vegetative one, which is the whole difference
    between a peduncle and a branch: it carries no foliage, it takes its node
    count from `branch_nodes` rather than the plant's own vegetative formula,
    and it is outside the branching budget -- so Branch Density and Branching
    Zone go on describing the plant's woody architecture rather than being
    conscripted to shape an inflorescence.

    Released here rather than through `_release_buds` for the reason
    `_release_inflorescence_branches` already gives: those gates are apical
    dominance, branching onset and `branch_density`, which are right for a
    shoot and wrong for a structure.
    """
    # Ordered rather than taken straight from the set, so which mode claims the
    # node's real buds does not depend on set iteration order -- that would make
    # a plant with two node-borne positions non-reproducible. Each position is
    # its claimant's: the first inflorescence on this level that wants it.
    # The old-wood and basal attachments are not released here: they break
    # from wood laid down long before this node was made, or from below the
    # plant entirely, so each has its own per-step pass.
    claims = [(m, _claimant(profile, axis, m)) for m in _NODE_BORNE]
    claims = [(m, k) for m, k in claims if k is not None]
    if not claims:
        return
    if axis.structural_depth >= 0 and not axis.is_fork:
        # An inflorescence axis bears none of its own. A fork or a sympodial
        # relay is a vegetative shoot continuing its parent -- it shares the
        # depth counter but bears them like any stem (every unit of a chain did
        # not, only the first).
        return
    phytomer = axis.phytomers[index]
    for mode, claim in claims:
        inf = profile.inflorescence_at(claim, axis.order)
        if inf.kind == InflorescenceKind.CATKIN:
            # Same attachment, different body: a catkin is a swept tube placed
            # as organs by `_add_catkins`, not a flowering region on an axis of
            # its own. The control is shared; the implementation is the kind's.
            continue
        if not inf.placement.allows(along, axis.order):
            continue
        # A solitary flower at a node is the bud itself, not a stalk carrying
        # one: a stalk grew on as a shoot of its own (on an indeterminate plant
        # it never capped, and ran to the season's end carrying flowers).
        solitary = inf.topology == Topology.TERMINAL_FLOWER
        buds = [b for b in list(phytomer.buds) if b.state == BudState.DORMANT]
        if not buds:
            # A node carries only as many real meristems as its phyllotaxis gave
            # it, and the first selected mode will have claimed them. A second
            # position at the same node is a real thing -- an axillary raceme
            # and a leaf-opposed one on the same stem -- so it gets a synthesised
            # bud rather than silently doing nothing.
            buds = [_extra_node_bud(plant, axis, phytomer, index)]
        for bud in buds:
            if mode == Attachment.INTERNODAL:
                # Move the meristem off its node and partway down the internode
                # below. That is the whole difference, so it is one field rather
                # than a second release path.
                bud.position_kind = BudPosition.INTERNODAL
                bud.along_internode = min(1.0, max(0.0, inf.internode_position))
            elif mode == Attachment.LEAF_OPPOSED:
                # Half a turn round the stem from the axil the bud sits in,
                # which is the whole of what "opposite the leaf" means. Also one
                # field rather than a second release path, for the same reason.
                bud.position_kind = BudPosition.LEAF_OPPOSED
                bud.divergence_deg += 180.0
            for member in _cluster_buds(plant, inf, phytomer, bud):
                if solitary:
                    # where the mode put the bud: at the node, down the
                    # internode, or opposite the leaf
                    _make_flower_bud(plant, profile, seed, axis, member, claim)
                    continue
                child = _release_as_shoot(plant, profile, seed, axis, member,
                                          axis.order, structural_depth=1,
                                          inflorescence=claim)
                if child is None:
                    continue
                _aim_inflorescence(plant, profile, seed, inf, child)
                _cache_branch_traits(plant, profile, inf, seed, child, along, 1.0)


def _extra_node_bud(plant: Plant, axis: Axis, phytomer, index: int) -> Bud:
    """One more meristem at a node whose own buds are already spoken for.

    Only reached when two node-borne positions are selected at once, which is
    why it copies the node's first bud rather than inventing an azimuth: the
    second position should sit in the same phyllotactic place the first did and
    then be moved by its own rule (half a turn for leaf-opposed, down the
    internode for internodal), not somewhere unrelated."""
    template = phytomer.buds[0] if phytomer.buds else None
    bud = Bud(
        id=plant.new_bud_id(),
        position_kind=BudPosition.AXILLARY,
        birth_step=plant.step,
        axis_id=axis.id,
        phytomer_index=index,
        divergence_deg=template.divergence_deg if template else 0.0,
        elevation_deg=template.elevation_deg if template else 45.0,
    )
    phytomer.buds.append(bud)
    return bud


def _release_old_wood_inflorescences(
    plant: Plant, profile: MorphologyProfile, seed: int
) -> None:
    """Break inflorescences from **old wood** -- cauliflory, ramiflory and the
    adventitious case, which are one mechanism with three different answers to
    "from where".

    Its own per-step pass rather than a branch of `_release_node_inflorescences`
    because of *when* rather than *where*: every node-borne attachment arises on
    the node it is made at, and these arise seasons later on a node that was
    made long ago. There is nothing to hook at node-creation time, so this walks
    the wood that already exists and asks how old it is.

    The three differ only in which wood qualifies, so they share one pass:

    * `CAULIFLOROUS` -- the trunk only (order 0), which is what cauliflory
      means: figs and cacao fruit on the bole, not on twigs.
    * `RAMIFLOROUS` -- the branches only (order 1 and up), as redbud does along
      bare limbs.
    * `ADVENTITIOUS` -- either, and from tissue with no meristem in it, so its
      buds are synthesised at scattered points along the internode rather than
      found at nodes.

    Reuses the same release, aim and trait-caching as the axillary path, so a
    cauline raceme is the same structure hanging off a different meristem.
    """
    for axis in list(plant.axes.values()):
        if axis.structural_depth >= 0 and not axis.is_fork:
            continue   # inflorescence axes; a fork / relay is wood like any stem
        # Each selected member decides for itself whether THIS axis is wood
        # it may break from: cauliflory the trunk, ramiflory the branches,
        # adventitious either. With both picked a plant fruits on its bole
        # and its limbs, which is what several real trees do. Each position
        # is its claimant's (the first inflorescence here that wants it).
        wood = _OLD_WOOD - ({Attachment.CAULIFLOROUS} if axis.order != 0
                            else {Attachment.RAMIFLOROUS})
        by_claim: dict[int, set] = {}
        for mode in sorted(wood):
            claim = _claimant(profile, axis, mode)
            if claim is not None:
                by_claim.setdefault(claim, set()).add(mode)
        for claim, modes in sorted(by_claim.items()):
            _release_old_wood(plant, profile, seed, axis, claim, modes)


def _release_old_wood(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                      claim: int, modes: set) -> None:
    """One inflorescence's old-wood positions on one axis."""
    inf = profile.inflorescence_at(claim, axis.order)
    last = max(len(axis.phytomers) - 1, 1)
    for index, phytomer in enumerate(axis.phytomers):
        if phytomer.age_steps(plant.step) < inf.cauline_min_age:
            continue
        along = index / last
        if not inf.placement.allows(along, axis.order):
            continue
        if modes == {Attachment.ADVENTITIOUS}:
            buds = _adventitious_buds(plant, seed, inf, phytomer, axis, index)
        else:
            buds = [b for b in list(phytomer.buds) if b.state == BudState.DORMANT]
            if Attachment.ADVENTITIOUS in modes:
                buds += _adventitious_buds(plant, seed, inf, phytomer, axis, index)
        for bud in buds:
            if bud.state != BudState.DORMANT:
                continue
            if bud.position_kind != BudPosition.ADVENTITIOUS:
                bud.position_kind = BudPosition.CAULINE
            for member in _cluster_buds(plant, inf, phytomer, bud):
                child = _release_as_shoot(plant, profile, seed, axis, member,
                                          axis.order, structural_depth=1,
                                          inflorescence=claim)
                if child is None:
                    continue
                _aim_inflorescence(plant, profile, seed, inf, child)
                _cache_branch_traits(plant, profile, inf, seed, child, along, 1.0)


def _adventitious_buds(
    plant: Plant, seed: int, inf, phytomer, axis: Axis, index: int
) -> list[Bud]:
    """Synthesise this phytomer's adventitious meristems, once.

    The only attachment that makes its own buds rather than releasing ones the
    plant already grew, because having no pre-existing bud site is precisely
    what adventitious means. Position along the internode and azimuth are both
    drawn from a stream seeded on the phytomer's identity, so they scatter
    without following the phyllotaxis and without needing an RNG threaded
    through the loop.

    Self-limiting by counting what it already made: this pass runs every step
    over wood that stays old, so without that it would keep adding buds to the
    same internode forever.
    """
    want = max(1, int(inf.attachment_per_node))
    existing = sum(1 for b in phytomer.buds
                   if b.position_kind == BudPosition.ADVENTITIOUS)
    if existing >= want:
        return []
    made: list[Bud] = []
    for slot in range(existing, want):
        # `_unit` rather than a Generator seeded on hash(): hashing a tuple that
        # contains a STRING is salted per process, so the first version of this
        # placed its buds somewhere different on every run -- caught by the
        # golden fixture failing to reproduce its own capture. Identity here is
        # the axis, the node and the slot, which is stable across processes and
        # independent of the order phytomers happen to be visited in.
        ident = (axis.id * 8191 + index) * 31 + slot
        bud = Bud(
            id=plant.new_bud_id(),
            position_kind=BudPosition.ADVENTITIOUS,
            birth_step=plant.step,
            axis_id=axis.id,
            phytomer_index=index,
            divergence_deg=360.0 * _unit(seed, _T_ADVENT_AZIMUTH, ident),
            elevation_deg=30.0 + 40.0 * _unit(seed, _T_ADVENT_ELEVATION, ident),
            along_internode=0.15 + 0.70 * _unit(seed, _T_ADVENT_ALONG, ident),
        )
        phytomer.buds.append(bud)
        made.append(bud)
    return made


def _release_basal_inflorescences(
    plant: Plant, profile: MorphologyProfile, seed: int
) -> None:
    """Break inflorescences from the **root crown** -- violets, many bulbs, and
    a primrose's scapes, where the flower stalk comes straight out of the
    ground rather than from anywhere up a stem.

    Its own pass for the same reason the old-wood one has one: there is no node
    to hook, because the position is *below* the lowest node the plant made.
    The lowest phytomer of each first-order axis stands in for the crown, which
    is where that axis meets the ground.

    Unlike the old-wood pass this has no age gate -- a basal scape is the first
    thing a violet does, not something it waits seasons for -- so the buds'
    own release state is what stops it running twice.
    """
    for axis in list(plant.axes.values()):
        if axis.structural_depth >= 0 or axis.order != 0 or not axis.phytomers:
            continue
        claim = _claimant(profile, axis, Attachment.BASAL)
        if claim is None:
            continue
        inf = profile.inflorescence_at(claim, axis.order)
        phytomer = axis.phytomers[0]
        for bud in list(phytomer.buds):
            if bud.state != BudState.DORMANT:
                continue
            bud.position_kind = BudPosition.BASAL
            for member in _cluster_buds(plant, inf, phytomer, bud):
                child = _release_as_shoot(plant, profile, seed, axis, member,
                                          axis.order, structural_depth=1,
                                          inflorescence=claim)
                if child is None:
                    continue
                _aim_inflorescence(plant, profile, seed, inf, child)
                _cache_branch_traits(plant, profile, inf, seed, child, 0.0, 1.0)


def _release_inflorescence_branches(
    plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis, index: int
) -> None:
    """Release this rachis node's branches straight away, rather than leaving
    them to `_release_buds`.

    Exactly the call `_relay_cyme` already makes, and for the reason its own
    docstring gives: `_release_buds` gates on apical dominance, branching onset
    and `branch_density`, which is right for a vegetative shoot and wrong here,
    where branching *is* the structure of the inflorescence and not a
    probabilistic event. It is also what keeps a panicle out of the plant's
    branching budget -- before this, shaping a panicle meant reaching for
    Branch Density and Branching Zone, which are the plant's controls, and its
    branches came out carrying full foliage because they were ordinary
    vegetative shoots.

    The branch inherits its parent's `order` rather than deepening it, so
    `max_branch_order` stays purely vegetative. See `Axis.structural_depth`.
    """
    phytomer = axis.phytomers[index]
    inf = _inf(profile, axis)
    # The same normalised coordinate every other flowering gradient is read on,
    # so the pyramid and the flower gradients cannot disagree about where the
    # top of the inflorescence is.
    position = index - axis.flowering_from
    count = max(1, _flower_nodes(profile, seed, axis))
    up = min(1.0, position / max(count - 1, 1))
    size = max(0.0, float(inf.branch_size_profile(up)))
    for bud in list(phytomer.buds):
        if bud.state != BudState.DORMANT:
            continue
        child = _release_as_shoot(plant, profile, seed, axis, bud, axis.order,
                                  structural_depth=axis.structural_depth + 1,
                                  inflorescence_scale=size)
        if child is None:
            continue
        # Every per-branch quantity is drawn once, here: its own scalar, times
        # its gradient along the rachis, times its own variance. At birth
        # because the stages that read them (`compute_radii`, `apply_gravity`,
        # and `_add_phytomer` on later steps) take no seed -- and because a
        # value that changed as the parent grew would re-scale nodes already
        # drawn, which is the same reason `_expected_phytomers` is fixed for an
        # axis's whole life.
        _cache_branch_traits(plant, profile, inf, seed, child, up, size)


def _cache_branch_traits(plant: Plant, profile: MorphologyProfile, inf, seed: int, child: Axis,
                         up: float, size: float) -> None:
    """Every per-branch quantity an inflorescence branch carries, drawn once at
    birth: its own scalar, times its gradient, times its own variance.

    Its own function because two different structures create these branches --
    a rachis node and a compound umbel's apex -- and they must carry the same
    traits or half the Branching box would silently apply to only one of them.

    `up` is where the branch sits in its parent inflorescence: along the rachis
    for a region, and centre-to-rim for a dome. Different coordinates, one
    meaning -- "where in this inflorescence is this branch" -- which is what
    lets one set of gradients serve both.
    """
    child.inflorescence_scale = max(0.0, size * (
        1.0 + inf.branch_nodes_variation
        * _signed(seed, _T_BRANCH_NODES, child.id)))
    child.inflorescence_length = max(0.0, inf.branch_length
        * float(inf.branch_length_profile(up))
        * (1.0 + inf.branch_length_variation
           * _signed(seed, _T_BRANCH_LENGTH, child.id)))
    # **No `branch_width_profile` here, deliberately.** It is the taper
    # along one branch and `diameter.compute_radii` applies it there.
    # Reading it at `up` as well -- a leftover from when it was a gradient
    # across the rachis -- applied it twice, so the curve's right-hand end
    # silently thinned the topmost branches on top of tapering them.
    # Reported as "the graph is still controlling the width of the top most
    # branches; it should just control taper".
    #
    # Capped at 1.0: a branch is never thicker than the stem it comes off.
    # Clamped after the variance, or scatter would reintroduce exactly the
    # case the cap exists to forbid.
    child.inflorescence_width = min(1.0, max(0.01, inf.branch_width
        * (1.0 + inf.branch_width_variation
           * _signed(seed, _T_BRANCH_WIDTH, child.id))))
    child.inflorescence_bend = (inf.branch_bend
        * float(inf.branch_bend_profile(up))
        * (1.0 + inf.branch_bend_variation
           * _signed(seed, _T_BRANCH_BEND, child.id)))
    # Signed: negative sags, positive lifts. Not clamped to zero, which is
    # what made this control one-sided while Angle and Bend beside it were not.
    child.inflorescence_droop = (inf.branch_droop
        * float(inf.branch_droop_profile(up))
        * (1.0 + inf.branch_droop_variation
           * _signed(seed, _T_BRANCH_DROOP, child.id)))
    if inf.maps:
        _scale_by_branch_maps(plant, profile, seed, inf, child)


def _scale_by_branch_maps(plant: Plant, profile: MorphologyProfile, seed: int,
                          inf: Inflorescence, child: Axis) -> None:
    """An inflorescence branch's maps, each a multiplier on its whole trait
    (the parameter, its curve and its +/- together), its caps kept."""
    def factor(target: str) -> float:
        return _branch_map_factor(plant, profile, seed, inf, child, target)
    child.inflorescence_scale = max(0.0, child.inflorescence_scale
                                    * factor(growth_maps.INFL_BRANCH_NODES))
    child.inflorescence_length = max(0.0, child.inflorescence_length
                                     * factor(growth_maps.INFL_BRANCH_LENGTH))
    child.inflorescence_width = min(1.0, max(0.01, child.inflorescence_width
                                             * factor(growth_maps.INFL_BRANCH_WIDTH)))
    child.inflorescence_bend *= factor(growth_maps.INFL_BRANCH_BEND)
    child.inflorescence_droop *= factor(growth_maps.INFL_BRANCH_DROOP)


def _release_cluster_rays(plant: Plant, profile: MorphologyProfile, seed: int,
                          axis: Axis) -> None:
    """A compound umbel: the apex puts out **rays**, each of which is an umbel
    of its own, instead of putting out flowers.

    The whole of *Apiaceae* -- carrot, cow parsley, fennel, hogweed -- and it
    needs no kind of its own, because it is the same statement a panicle makes
    one level down: at every place this inflorescence would have put a flower,
    put a whole inflorescence instead. What differs is only *where* those
    places are, and `_dome_fill` already answers that for a dome. So the rays
    are laid out on exactly the layout the florets would have used, with
    `branch_count` in place of `flower_count`.

    Each ray is an ordinary axis at its parent's own `order` (the inflorescence
    is not deeper wood -- see `Axis.structural_depth`), so it grows its
    `branch_nodes`, reaches `_begin_inflorescence` again, and at the last depth
    fills its own dome with flowers. Depth 2 gives the triple umbels real cow
    parsley produces, from the same code and no extra case.

    Released directly rather than through `_release_buds`, for the reason
    `_release_inflorescence_branches` already gives: those gates are right for
    a vegetative shoot and wrong for a structure.
    """
    tip = axis.phytomers[-1] if axis.phytomers else None
    if tip is None:
        _cap_with_flower(plant, profile, seed, axis)
        return

    inf = _inf(profile, axis)
    count = max(1, _flower_nodes(profile, seed, axis))
    for polar_deg, azimuth_deg, radial_t in _dome_fill(inf, count):
        # The dome's own angles, exactly as a floret's pedicel would have taken
        # them -- so Head Spread shapes a compound umbel's rays as directly as
        # it shapes a simple one's stalks, and turning Branching Depth on does
        # not silently change the head's shape.
        direction = radial_direction(
            tip.direction, tip.reference, azimuth_deg, _elevation_from_axis(polar_deg))
        child = _new_axis(
            plant,
            profile,
            seed,
            axis.order,
            tip.tip_position.copy(),
            direction,
            base_reference=parallel_transport(tip.reference, tip.direction, direction),
            parent_axis_id=axis.id,
            parent_phytomer_index=len(axis.phytomers) - 1,
            structural_depth=axis.structural_depth + 1,
        )
        # Centre-to-rim rather than base-to-tip: a dome has no along-the-axis
        # position, and the radius is the gradient that means something on one
        # -- an outer ray is longer, which is what brings a compound umbel's
        # umbellets up to one level. Same coordinate `head_dome_profile`
        # already reads, and captioned the same way in the panel.
        size = max(0.0, float(inf.branch_size_profile(radial_t)))
        _cache_branch_traits(plant, profile, inf, seed, child, radial_t, size)
    axis.apex.state = BudState.ABORTED


def _flowering_target(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis) -> int:
    """The node count at which this particular axis terminates in a flower:
    its order's species-typical count, varied per axis so a plant does not
    bloom in lockstep.

    The variation lives in the node *count*, not in a time delay. An earlier
    version staggered blooming by making each axis wait for the flowering
    curve to cross a random per-axis value, which let axes overshoot their
    target by an unbounded amount -- the trunk built 24 nodes against a
    stated target of 15, so the control did not actually control anything."""
    level = _level(profile, axis)
    if level is not None:
        built = _built_nodes(profile, axis)
        if built < _vegetative_nodes(profile, axis):
            return built                          # Growth %: flowers at the cut tip
    if level is not None and level.relays:
        return _vegetative_nodes(profile, axis)   # one sympodial unit
    if axis.node_budget >= 0:
        # A chance-fork child: its parent's remaining budget (`_chance_fork`).
        return max(2, axis.node_budget)
    if axis.structural_depth >= 1 and not axis.is_fork:
        # An inflorescence branch is not a vegetative shoot and does not get the
        # per-axis flowering jitter: a panicle's branches are a structure, and
        # scattering their lengths individually reads as noise rather than as
        # the acropetal gradient `branch_length_profile` states. A vegetative
        # fork shares the depth counter but IS an ordinary shoot, so it falls
        # through to the base length like any axis of its order.
        return _inflorescence_branch_nodes(profile, axis)
    # Steps then flowers: exactly
    # the level's Steps, no per-axis jitter -- Steps is a count the panel
    # states, and a stated count is what the axis builds.
    return max(2, _vegetative_nodes(profile, axis))


def _terminate_determinate_axes(
    plant: Plant, profile: MorphologyProfile, seed: int, plant_age: float,
    day: float | None = None, environment=None
) -> None:
    """A determinate axis is one whose *apex itself* becomes a flower -- that
    conversion is what stops it.

    Two conditions, and the distinction between them is what the earlier
    version got wrong. The axis must have **completed its own development**
    (built its species-typical node count for its order), and the plant must
    be **old enough to flower at all** (the onset curve). Treating the node
    count as a mere floor meant that once the onset curve saturated, every
    axis flowered the moment it crossed the floor -- so the floor silently
    became the node count and 96% of branches were 4-node stubs.

    The second condition was described here but never actually applied:
    `plant_age` came in and was not read, so the onset curve did nothing at
    all on a determinate plant -- which is two of the three reference species.
    Setting the curve flat to zero produced exactly the same 23 flowers, which
    is how it was caught. It is a real gate now.

    **It is a plant-level gate, not a per-axis lottery**, and the difference is
    not subtle. Testing each axis against its own draw was tried first and is
    wrong: a determinate axis that is not allowed to flower does not wait
    politely, it *keeps growing* and keeps releasing buds, so blocking 70% of
    axes at mid-life compounds. It took a generic perennial from 143 nodes to
    307 and a Rudbeckia to 532 flowers. What the curve says is when this
    species is old enough to bloom at all, which is one fact about the plant
    and not a per-branch coin flip.
    """
    # "Determinate" is exactly "the apex reaches floral identity", which is what
    # vegetativeness measures -- so this asks veg rather than the category. With
    # the derived default the two are the same statement; they part company only
    # for a species that sets veg directly, which is what makes a panicle
    # reachable at all. See `Vegetativeness`.
    # An indeterminate plant's shoot tips never flower -- but an inflorescence
    # axis (a stalk released at a node) is an inflorescence whatever the
    # plant's tips do, so on such a plant only those are opened here.
    stalks_only = not profile.vegetativeness().apex_is_floral()
    if stalks_only and not any(_is_inflorescence_axis(a) for a in plant.growing_axes()):
        return
    # Plant-level permission: the calendar's bloom date on the day path (bloom
    # opens on the date, the node-count floor below still decides readiness, so
    # bloom lands at max(date, ready)), else the developmental onset curve --
    # byte-identical off the day path.
    if profile.flowering_permission(plant_age, day) <= 0.0:
        return
    for axis in plant.growing_axes():
        if stalks_only and not _is_inflorescence_axis(axis):
            continue
        if axis.flowering_from >= 0:
            # Already building its inflorescence: stop once it has laid down
            # the flowering nodes its kind calls for. Read per axis, not once
            # up front -- with a lateral inflorescence the trunk and its
            # branches can want different node counts.
            if (len(axis.phytomers) - axis.flowering_from
                    >= _flower_nodes(profile, seed, axis)):
                axis.apex.state = BudState.ABORTED
            continue
        if len(axis.phytomers) >= _flowering_target(plant, profile, seed, axis):
            _begin_inflorescence(plant, profile, seed, axis)
            _preform_region(plant, profile, seed, axis, environment)


def _is_inflorescence_axis(axis: Axis) -> bool:
    """A stalk or branch of an inflorescence (not a vegetative fork / relay,
    which share the depth counter): its tip is always its inflorescence."""
    return axis.structural_depth >= 1 and not axis.is_fork


def _preform_region(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis,
                    environment=None) -> None:
    """Lay down an axis's whole flowering region the moment it opens, then stop
    its apex.

    **The flower count is settled when the inflorescence is initiated, not by
    how much season is left.** Building the region one node per step, as the
    vegetative stem is built, let an axis that began flowering late run out of
    span part way up: measured on a raceme plant at Flower Count 7, the trunk
    and early branches carried 7 while later branches ended on 3, 2, 1 or 0 --
    `_close_out` simply stopped them mid-region. An inflorescence is initiated
    as a unit (its flower primordia are laid down in the bud) and *then*
    elongates, so every node here is born now and the elongation curve opens
    the region out over the following steps. A late inflorescence therefore
    reads as a young, compact one -- with all of its flowers -- rather than as
    a short one.

    Only for the region kinds (raceme, spike, corymb, and a compound rachis,
    whose nodes bear branches): every other topology already places all of its
    flowers at once, and stops its own apex. Obstacles still stop it -- a node
    that `_add_phytomer` refuses ends the region there.
    """
    if axis.flowering_from < 0 or not axis.is_growing:
        return
    want = _flower_nodes(profile, seed, axis)
    while len(axis.phytomers) - axis.flowering_from < want:
        if _add_phytomer(plant, axis, profile, seed, environment) is None:
            break
    axis.apex.state = BudState.ABORTED
    # Laid down as a unit, then developing from the base up: node i of the
    # region starts i x (Elongation Duration / its nodes) steps after the first,
    # so the whole region has started within one elongation and a timelapse
    # shows it come in flower by flower rather than whole (`_trim_unborn`
    # leaves out the ones not started yet; buds on them wait).
    region = axis.phytomers[axis.flowering_from:]
    if len(region) > 1:
        dt = max(profile.elongation_steps, 1) / len(region)
        for i, phytomer in enumerate(region):
            born = phytomer.birth_step + i * dt
            phytomer.birth_step = born
            for organ in phytomer.organs:
                organ.birth_step = born
            for bud in phytomer.buds:
                bud.birth_step = born


def _flower_gradients(profile: MorphologyProfile, seed: int, axis: Axis,
                      organ, position: int, count: int) -> None:
    """Write every field a flower reads off its position in the flowering
    region: its size, and the whole of its pedicel.

    Its own function purely so it can be applied **twice** -- once at creation
    against the region the axis intends to build, and again if the axis ran out
    of span and built a shorter one. Every draw is seeded on the organ own id,
    so re-applying with an unchanged count reproduces the same values exactly
    and only a changed count moves anything.
    """
    inf = _inf(profile, axis)
    organ_id = organ.id
    # Normalised position in the flowering region: 0 at the first flower, 1 at
    # the last. Every gradient below is read here, so they all share one
    # coordinate and cannot disagree about where the top of it is.
    up = min(1.0, position / max(count - 1, 1))
    organ.inflorescence = _inf_index(profile, axis)
    scale = profile.flower_size_of(inf) * inf.flower_size_profile(up) * (
        1.0 + inf.flower_size_variation * _signed(seed, _T_FLOWER_SIZE, organ_id))
    organ.bloom_start = _bloom_start(inf, position, count)
    organ.final_scale = scale
    # `pedicel_droop_deg` is not set here: `physics.apply_pedicel_droop` computes
    # each stalk's sag physically from its flower's mass, length and radius and
    # overwrites it, so an assignment here would only be thrown away.
    # No `_elevation_from_axis`, unlike every other angle in this module: a
    # pedicel angle is already stated from perpendicular so that it can be
    # signed. See `Inflorescence.pedicel_angle_deg` for why it is the exception.
    organ.elevation_deg = (inf.pedicel_angle_deg * inf.pedicel_angle_profile(up)
                           + inf.pedicel_angle_variation
                           * _signed(seed, _T_PEDICEL_ANGLE, organ_id))
    organ.pitch_deg = (inf.flower_pitch_deg * inf.flower_pitch_profile(up)
                       + inf.flower_pitch_variation
                       * _signed(seed, _T_FLOWER_PITCH, organ_id))
    organ.pedicel_length = _pedicel_length(profile, seed, organ_id, position, count, axis.order,
                                           inf)
    organ.pedicel_bend = max(0.0, inf.pedicel_bend * inf.pedicel_bend_profile(up) * (
        1.0 + inf.pedicel_bend_variation * _signed(seed, _T_PEDICEL_BEND, organ_id)))
    organ.pedicel_width = max(0.001, inf.pedicel_width * inf.pedicel_width_profile(up) * (
        1.0 + inf.pedicel_width_variation * _signed(seed, _T_PEDICEL_WIDTH, organ_id)))
    if organ.map_factors:
        _scale_by_flower_maps(organ)


def _scale_by_flower_maps(organ) -> None:
    """A flower's inflorescence maps (`Organ.map_factors`), each a multiplier on
    its whole value -- the parameter, its curve and its +/- together."""
    f = organ.map_factors
    organ.elevation_deg *= f.get(growth_maps.PEDICEL_ANGLE, 1.0)
    organ.pitch_deg *= f.get(growth_maps.FLOWER_PITCH, 1.0)
    organ.pedicel_length = max(0.0, organ.pedicel_length * f.get(growth_maps.PEDICEL_LENGTH, 1.0))
    organ.pedicel_bend = max(0.0, organ.pedicel_bend * f.get(growth_maps.PEDICEL_BEND, 1.0))
    organ.pedicel_width = max(0.001, organ.pedicel_width
                              * f.get(growth_maps.PEDICEL_WIDTH, 1.0))


def _renormalise_truncated_regions(plant: Plant, profile: MorphologyProfile, seed: int) -> None:
    """Re-read every flowering gradient against the region an axis actually
    built, for the axes that did not get to finish.

    A flower gradient is sampled at creation against the count its axis
    *intends*, which is the only number available then. An axis that runs out of
    developmental span before laying them all down therefore samples only the
    first slice of every curve, and its last flower reads as "number 7 of 9"
    rather than as the tip -- measured on a panicle whose upper branches built 7
    and 8 of an intended 9: their tip-most flower came out at scale 0.55 and
    0.42 against 0.28 on the branches that finished. Reported exactly that way,
    as the tip flowers of the short branches being too large.

    Only touches axes where the two counts differ, so a plant whose axes all
    finish is bit-for-bit what it was -- which is what keeps the goldens honest
    about this change rather than absorbing it.
    """
    for axis in plant.axes.values():
        if axis.flowering_from < 0:
            continue
        built = len(axis.phytomers) - axis.flowering_from
        if built < 1 or built >= max(1, _flower_nodes(profile, seed, axis)):
            continue
        for index in range(axis.flowering_from, len(axis.phytomers)):
            for organ in axis.phytomers[index].organs:
                if organ.kind == OrganKind.FLOWER:
                    _flower_gradients(profile, seed, axis, organ,
                                      index - axis.flowering_from, built)


def _flower_nodes(profile: MorphologyProfile, seed: int, axis: Axis) -> int:
    """How many flowering nodes this particular axis builds.

    Drawn once per axis rather than per flower, because the count is a property
    of one inflorescence -- and then it has to agree everywhere. Three places
    read it: the test that stops an axis flowering, the normalised coordinate
    every flower gradient is sampled on, and the bloom order. A value that
    differed between them would put a gradient's "last flower" somewhere other
    than the last flower.
    """
    inf = _inf(profile, axis)
    if _branches_here(profile, axis):
        # A branching region is measured in **branches**, not flowers. One
        # number served both until `branch_count` existed, so a panicle's
        # rachis carried exactly as many branches as each of its branchlets
        # carried flowers -- "ten branches of three flowers" was unsayable.
        base = max(1, inf.branch_count)
        if axis.structural_depth >= 1:
            base = max(1, int(round(base * axis.inflorescence_scale)))
        if inf.branch_count_variation <= 0.0:
            return base
        jitter = inf.branch_count_variation * _signed(seed, _T_BRANCH_COUNT, axis.id)
        return max(1, int(round(base * (1.0 + jitter))))
    base = inf.extra_nodes
    if axis.structural_depth >= 1:
        # "Longer is more divided" is one fact, not two: a panicle's lower
        # branches are both, so the same gradient scales the region as scales
        # the stem below it. An uppermost branch reduced to a single flower
        # falls straight out of this.
        base = max(1, int(round(base * axis.inflorescence_scale)))
    if base <= 0 or inf.flower_count_variation <= 0.0:
        return base
    jitter = inf.flower_count_variation * _signed(seed, _T_FLOWER_COUNT, axis.id)
    return max(1, int(round(base * (1.0 + jitter))))


def _cluster_bloom_start(inf, radial_t: float) -> float:
    """Where a cluster flower sits on the bloom timeline, 0 opening first.

    Bloom order across a dome runs by *radius*, not by index, which is what
    centripetal and centrifugal actually mean: centripetal matures from the
    outside inward, so the rim opens first; centrifugal is the reverse.
    Acropetal and basipetal have no radial meaning on a cluster, so it opens
    evenly.

    Until this existed a cluster's bloom order lived only in a `birth_step`
    offset, which meant that once the stage system started reading
    `bloom_start` every flower in an umbel opened simultaneously. `radial_t` is
    0 at the centre and 1 at the rim.
    """
    if inf.bloom_order == BloomOrder.CENTRIPETAL:
        return 1.0 - radial_t
    if inf.bloom_order == BloomOrder.CENTRIFUGAL:
        return radial_t
    return 0.0


def _dome_fill(inf, count: int) -> list[tuple[float, float, float]]:
    """Place `count` flowers over a spherical cap of half-angle
    `cluster_spread_deg / 2`, returning (polar from the axis, azimuth around
    it, normalised radius) per flower.

    Two things make this read as an umbel rather than a ring:

    * The polar angle *varies*. Putting every flower at `pedicel_angle_deg`
      leaves the middle of the head empty, which is the "nice circle missing
      its centre" this replaces. Here the outermost flower sits at the rim and
      the innermost near the axis.
    * Spacing is by equal **solid angle**, `cos(polar)` linear in rank, not by
      equal polar angle. Equal polar angle crowds the rim, because a cap's area
      grows with sin(polar); equal solid angle gives even density all the way
      across.

    Azimuth advances by the golden angle so successive flowers never stack into
    visible spokes -- the same reason phyllotaxis uses it.

    **The spread runs to a full 360, meaning a complete sphere of flowers.** It
    was capped at 179 because an earlier two-pass flat-top solve derived a
    cluster's plane radius as `climb * tan(polar)`, undefined at 90 degrees and
    so at a 180-degree spread. That solve is gone (`solve_flat_head` chooses the
    radius directly), which left the cap vestigial and Allium's globe and
    Hedera's out of reach. Nothing here degenerates at the top of the range:
    `rank` never reaches 1, since it is offset by half a step, so `polar` stays
    strictly below 180 -- and `organ_normal` falls back to the phytomer's own
    transported reference for an organ that does point along its axis.
    """
    half = np.radians(min(360.0, max(1.0, inf.cluster_spread_deg)) * 0.5)
    cos_min = float(np.cos(half))
    out = []
    for i in range(count):
        rank = (i + 0.5) / count
        cos_polar = 1.0 - rank * (1.0 - cos_min)
        polar = float(np.degrees(np.arccos(np.clip(cos_polar, -1.0, 1.0))))
        azimuth = (i * 137.5) % 360.0
        # Radius as a fraction of the rim, for bloom order.
        #
        # A flat-topped head reads its own equal-area rank instead, and that is
        # what lets Head Spread be absent from its panel honestly: the flat
        # solve overwrites every elevation this function produces, so the only
        # thing the spread still reached on a flat umbel was its bloom order --
        # a control named for a geometry it no longer touches, quietly steering
        # something else. `sqrt(rank)` is exactly the radius fraction
        # `solve_flat_head` places the flower at, so bloom order now follows
        # where the flower actually sits.
        radial_t = (float(np.sqrt(rank)) if inf.is_flat_topped
                    else polar / max(float(np.degrees(half)), 1e-6))
        out.append((polar, azimuth, min(1.0, radial_t)))
    return out


def _begin_inflorescence(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis) -> None:
    """The axis has finished its vegetative development. What happens now is a
    question of *topology* -- what the meristem does -- and the answer is one of
    four, not one of nine.

    This used to test four kind-memberships in sequence, which hid that the
    nine kinds carry only four structures between them: raceme, spike, corymb
    and catkin produce byte-identical skeletons and differ purely in how their
    flowers are placed. Asking `inf.topology` states that directly, and makes
    the fourth structure -- a panicle -- reachable without inventing a tenth
    kind for it. See `morphology.Topology` and `morphology.Vegetativeness`.
    """
    if axis.structural_depth < 0 or axis.is_fork:
        # A vegetative tip: it becomes the inflorescence that has its level's
        # tips (the first placed there with Terminal), from here on.
        claim = _claimant(profile, axis, Attachment.TERMINAL)
        if claim is not None:
            axis.inflorescence = claim
    else:
        claim = _inf_index(profile, axis)
    inf = _inf(profile, axis)
    topology = inf.topology

    if claim is None:
        # A vegetative stem carrying node-borne inflorescences does not flower
        # itself -- the stalks released along it do (a fork or sympodial relay
        # is such a stem too, though it shares the depth counter). Only vegetative axes take
        # this path: the released axis is an inflorescence axis, so it falls
        # through and builds its flowering region normally rather than
        # releasing more of itself. Its tip grows on as the next level down
        # when its level has one (`_continue_tip`).
        if not _continue_tip(plant, profile, seed, axis):
            axis.apex.state = BudState.ABORTED
        return

    if topology == Topology.NONE:
        # This axis makes no inflorescence. It has finished its development,
        # so it stops -- and that is the whole of it: nothing is placed, no
        # flowering region opens, and `axis.flowering_from` stays -1, so it
        # carries no flower load and bends only under its own wood and leaves,
        # like any leafy shoot -- or grows on as the next level down.
        if not _continue_tip(plant, profile, seed, axis):
            axis.apex.state = BudState.ABORTED
        return

    if topology == Topology.TERMINAL_FLOWER:
        _cap_with_flower(plant, profile, seed, axis)
        return

    if topology == Topology.RELAY:
        _relay_cyme(plant, profile, seed, axis)
        return

    if inf.branching_depth > 0 and axis.structural_depth < 0:
        # This axis is beginning a compound inflorescence, so it is depth 0 of
        # one. Set before either branch below reads it.
        axis.structural_depth = 0

    if topology in _COMPOUNDS_ALONG_A_REGION:
        # A panicle, or any compound flowering region. The axis enters a region
        # exactly as a raceme does -- it keeps growing and its remaining nodes
        # stop bearing leaves -- but those nodes bear inflorescence BRANCHES
        # instead of flowers, until the last depth, which falls through below
        # and blooms.
        #
        # What decides that is `structural_depth`, **not** `axis.order`: an
        # inflorescence is a bounded shoot system with its own internal
        # branching, and how far from the root it happens to sit is a separate
        # fact. Keyed on branch order (as this was, briefly) a panicle on a
        # tree twig could only ever be one order deep, which is a raceme. See
        # `Axis.structural_depth`.
        #
        # `BRANCHED` used to be the only topology allowed in here, which is why
        # a compound corymb and a compound spike were unreachable: the
        # machinery was already general and only this test was narrow. A
        # panicle is a raceme at depth 1, and now says so.
        if 0 <= axis.structural_depth < inf.branching_depth:
            axis.flowering_from = len(axis.phytomers)
            return
        # Deepest branch: fall through and flower.

    if topology == Topology.APEX_CLUSTER:
        if 0 <= axis.structural_depth < inf.branching_depth:
            # A compound umbel: the apex puts out RAYS rather than flowers, on
            # the same dome layout, and each ray is an umbel of its own. That
            # is the whole of *Apiaceae*, and it needs no kind of its own --
            # depth 1 on an umbel is one, depth 2 is the triple umbel real cow
            # parsley produces.
            _release_cluster_rays(plant, profile, seed, axis)
            return
        # Umbel, flat umbel and head: every flower attaches at the apex,
        # filling a dome rather than sitting on a ring.
        tip = axis.phytomers[-1] if axis.phytomers else None
        if tip is None:
            _cap_with_flower(plant, profile, seed, axis)
            return
        count = max(1, inf.flower_count)
        tip_index = len(axis.phytomers) - 1
        for polar_deg, azimuth_deg, radial_t in _dome_fill(inf, count):
            organ_id = plant.new_organ_id()
            factors = _flower_map_factors(plant, profile, seed, axis, tip_index, tip.factor,
                                          organ_id, plant.step, inf)
            tip.organs.append(Organ(
                id=organ_id,
                kind=OrganKind.FLOWER,
                birth_step=plant.step,
                bloom_start=_cluster_bloom_start(inf, radial_t),
                inflorescence=_inf_index(profile, axis),
                final_scale=_flower_scale(profile, seed, inf, organ_id),
                divergence_deg=azimuth_deg,
                elevation_deg=_elevation_from_axis(polar_deg),
                # The same angle again, and not a duplicate: elevation aims the
                # *stalk*, pitch aims a flower that has no stalk to be aimed by.
                # A sessile cluster is exactly that case -- `resolve_geometry`
                # discards elevation for an organ with no pedicel and reads
                # `pitch_deg` instead -- so without this the dome's polar angle
                # was computed and thrown away, leaving every floret in a head
                # pointing flat out sideways at any Head Spread. Writing it
                # unconditionally also means an umbel whose Pedicel Length is
                # turned down to nothing degrades into a head correctly rather
                # than collapsing to one facing.
                pitch_deg=_elevation_from_axis(polar_deg),
                pedicel_length=(
                    0.0 if inf.is_sessile
                    else _pedicel_length(profile, seed, organ_id, 0, count, axis.order, inf)
                ),
                # Thickness is the inflorescence's to state here as anywhere:
                # this was simply never wired, and went unnoticed because
                # `Organ.pedicel_width` happens to default to the same 0.3, so
                # the panel's own Pedicel Width did nothing on an umbel while
                # looking correct. The along-the-stalk taper curve is applied in
                # the bridge and so always did reach these kinds.
                pedicel_width=max(0.001, inf.pedicel_width * (
                    1.0 + inf.pedicel_width_variation
                    * _signed(seed, _T_PEDICEL_WIDTH, organ_id))),
                # Variance applies here, the along-the-axis profiles do not:
                # every flower in a dome sits at the apex, so there is no
                # "position along the flowering region" to read a gradient at.
                # Consistent with every other flowering gradient, which the
                # cluster kinds already ignore for the same reason.
                pedicel_bend=max(0.0, inf.pedicel_bend * (
                    1.0 + inf.pedicel_bend_variation
                    * _signed(seed, _T_PEDICEL_BEND, organ_id))),
                # `pedicel_droop_deg` left at its default: `apply_pedicel_droop`
                # sets every stalk's sag physically and overwrites it.
            ))
            if factors:
                # The stalk's maps; its aim is the dome's (no Pedicel Angle or
                # Flower Pitch here, mapped or not).
                organ = tip.organs[-1]
                organ.map_factors = {k: v for k, v in factors.items()
                                     if k not in (growth_maps.PEDICEL_ANGLE,
                                                  growth_maps.FLOWER_PITCH)}
                _scale_by_flower_maps(organ)
        axis.apex.state = BudState.ABORTED
        return

    # Raceme, spike, corymb: the axis keeps going, but its remaining nodes bear
    # flowers instead of leaves.
    axis.flowering_from = len(axis.phytomers)
    # A pendulous inflorescence *grows* downward rather than being bent
    # downward, which is a different shape: bending a grown axis spreads the
    # turn evenly and curls a long region up like a watch spring. Measured on a
    # 0.157 m raceme at full Peduncle Droop, the tip finished 0.035 m *above*
    # where the region started.
    _aim_inflorescence(plant, profile, seed, inf, axis)


def _relay_cyme(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis) -> None:
    """A cyme: the tip blooms and is spent, and the inflorescence carries on
    from shoots just below it, each repeating the pattern.

    This is the one inflorescence built out of *branching* rather than out of a
    flowering region, because that is what it botanically is. Nothing recursive
    happens here: each relay is an ordinary axis with an ordinary apex, so it
    grows, reaches its own node count, and arrives back at
    `_begin_inflorescence` on its own.

    **Two things were wrong with the previous version, and together they meant
    a cyme was Solitary with a couple of extra branches.** Measured on the
    generic perennial: 30 axes and 25 flowers against Solitary's 27 and 23,
    with 18 of 25 relay events spawning *zero* relays and the other 7 spawning
    exactly one -- never the two `laterals_per_node` asked for.

    1. **Depth was bounded by `max_branch_order`** and each relay took
       `axis.order + 1`, so on a default profile a cyme relayed at most twice
       and every relay ate the plant's own woody branching budget. This is the
       identical mistake the panicle work already diagnosed and fixed with
       `Axis.structural_depth`; it was simply never carried back here. A
       relay now **inherits its parent's order** -- it is the inflorescence
       continuing, not a subordinate lateral -- and depth is
       `Inflorescence.branching_depth`.
    2. **Relays were fished out of surviving axillary buds.** `_release_buds`
       has usually consumed them, and alternate phyllotaxis only ever supplies
       one bud per node, so `laterals_per_node` was unreachable above 1 and a
       dichasium was impossible however the panel was set. The tooltip admitted
       this ("a forked cyme needs Opposite or Whorled leaves"), which is a
       hidden coupling reading as a broken control. **The inflorescence places
       its own relays now**, the same call `_flower_arrangement` already makes
       for a verticillaster's whorl: an inflorescence states the arrangement of
       its own parts rather than inheriting the foliage's.

    The relays are placed evenly around the node, and the whole set is turned
    `relay_rotation_deg` from the set that produced this axis. That one number
    is what separates the classical forms -- see `Inflorescence.
    relay_rotation_deg`. Their launch angle is `branch_angle_deg`, which is the
    zigzag: it already states "which way does an inflorescence axis leave the
    one below it", so a separate zigzag control would be the same control
    twice.

    Buds are released directly rather than left to `_release_buds`, which gates
    on dominance, branching onset and `branch_density` -- right for a
    vegetative shoot, wrong here, where relaying *is* the structure of the
    inflorescence and not a probabilistic event.
    """
    _cap_with_flower(plant, profile, seed, axis)

    inf = _inf(profile, axis)
    if axis.structural_depth < 0:
        # The axis that begins the inflorescence. Depth 0 is the terminal
        # flower's own generation; the relays below it are depth 1.
        axis.structural_depth = 0
    axis.terminal_organ.bloom_start = _cyme_bloom_start(inf, axis.structural_depth)

    tip = axis.phytomers[-1] if axis.phytomers else None
    if tip is None:
        return

    # The spent tip's own node is inflorescence tissue: whatever it does not
    # spend on relays is finished, not a future side branch. Without this the
    # buds a shallow cyme leaves behind come back through `_release_buds` as
    # ordinary vegetative shoots once dominance lifts, which is a large part of
    # why the old build read as an ordinary branchy plant.
    spent = [b for b in tip.buds if b.state == BudState.DORMANT]
    for bud in spent:
        bud.state = BudState.ABORTED
    if axis.structural_depth >= 1:
        # A relay is entirely inflorescence, not just at its tip.
        for phytomer in axis.phytomers:
            for bud in phytomer.buds:
                if bud.state == BudState.DORMANT:
                    bud.state = BudState.ABORTED

    if axis.structural_depth >= max(0, inf.branching_depth):
        return

    count = max(1, inf.laterals_per_node)
    # Evenly around the node, with the whole set turned from the set that
    # produced this axis. Accumulated rather than derived from the depth so
    # that a rotation applies in the parent's own transported frame at each
    # step -- see `Axis.relay_azimuth_deg`.
    base = axis.relay_azimuth_deg + inf.relay_rotation_deg
    step = 360.0 / count
    for i in range(count):
        divergence = base + i * step
        # `branch_angle_deg` is already stated from perpendicular and signed,
        # like `pedicel_angle_deg` and unlike `leaf_angle_deg` -- so no
        # `_elevation_from_axis` here. See `Inflorescence.branch_angle_deg`.
        # Seeded on the parent axis and the relay's own slot rather than on
        # the child's id, because the direction is needed before the axis
        # exists. Still order-independent, which is the property that matters.
        elevation = (inf.branch_angle_deg
                     + inf.branch_angle_variation
                     * _signed(seed, _T_BRANCH_ANGLE, axis.id * 16 + i))
        if growth_maps.enabled(inf.maps, growth_maps.INFL_BRANCH_ANGLE):
            # read at the spent tip, as the relay's bud would be
            elevation *= _node_map_factor(
                plant, profile, seed, growth_maps.INFL_BRANCH_ANGLE, axis,
                len(axis.phytomers), tip.factor, _MAP_KIND_BUD, axis.id * 16 + i,
                stack=inf.maps)
        direction = radial_direction(tip.direction, tip.reference, divergence, elevation)
        child = _new_axis(
            plant,
            profile,
            seed,
            axis.order,
            tip.tip_position.copy(),
            direction,
            base_reference=parallel_transport(tip.reference, tip.direction, direction),
            parent_axis_id=axis.id,
            parent_phytomer_index=len(axis.phytomers) - 1,
            structural_depth=axis.structural_depth + 1,
        )
        child.relay_azimuth_deg = divergence
        child.is_cyme_relay = True
        # The same per-branch quantities a panicle's branches carry, drawn once
        # at birth for the same reason: `compute_radii` and `apply_gravity`
        # take no seed, and a value that changed as the parent grew would
        # re-scale nodes already drawn.
        child.inflorescence_length = max(0.0, inf.branch_length * (
            1.0 + inf.branch_length_variation
            * _signed(seed, _T_BRANCH_LENGTH, child.id)))
        child.inflorescence_width = min(1.0, max(0.01, inf.branch_width * (
            1.0 + inf.branch_width_variation
            * _signed(seed, _T_BRANCH_WIDTH, child.id))))
        child.inflorescence_bend = inf.branch_bend * (
            1.0 + inf.branch_bend_variation
            * _signed(seed, _T_BRANCH_BEND, child.id))
        child.inflorescence_droop = (inf.branch_droop * (
            1.0 + inf.branch_droop_variation
            * _signed(seed, _T_BRANCH_DROOP, child.id)))
        if inf.maps:
            _scale_by_branch_maps(plant, profile, seed, inf, child)
        # A bud on the node to record what happened there, where one is spare.
        # Bookkeeping only -- the direction above is the inflorescence's, not
        # the bud's -- but a relay that leaves no trace in the bud counts makes
        # the Diagnostics panel lie about where the plant's meristems went.
        if i < len(spent):
            spent[i].state = BudState.RELEASED
            spent[i].fate = BudFate.SHOOT
            spent[i].child_axis_id = child.id


def _pedicel_length(profile: MorphologyProfile, seed: int, organ_id: int, index: int,
                    count: int, order: int = 0, inf: Inflorescence | None = None) -> float:
    """A corymb's lower pedicels are longer, which is what brings every flower
    up to one level; every other kind uses a constant length."""
    if inf is None:
        inf = profile.inflorescence_for(order)
    if inf.is_sessile:
        return 0.0
    # A corymb or flat umbel's real lengths are solved after growth, against
    # the flowers' actual positions (inflorescence.solve_flat_top_lengths).
    # This is only the starting length that solve measures from -- an earlier
    # attempt to bake the flat top in here as a fixed ratio could not work:
    # the flowering axis spans ~0.2 m while pedicels are ~0.03 m, so even a
    # 2.5x ratio moved the lowest flower 0.045 m of the 0.2 m it needed.
    jitter = 1.0 + inf.pedicel_variation * _signed(seed, _T_PEDICEL, organ_id)
    up = min(1.0, index / max(count - 1, 1))
    return max(0.0, inf.pedicel_length * inf.pedicel_length_profile(up) * jitter)




def _spread(divergence_deg: float, spread_deg: float, jitter_deg: float = 0.0) -> float:
    """Narrow the arc of stem the flowers occupy, and scatter them within it.

    A scaling rather than a replacement, so partial values read as "leaning
    that way" rather than snapping. Each flower's signed divergence spans
    -180..180, so scaling by `spread/360` lands them across exactly
    `spread_deg` of arc: 360 leaves phyllotaxis untouched, 0 puts every flower
    on one line up one side.

    Scaling the angle itself moves the attachment point as well as the facing,
    where a real secund raceme keeps its spiral attachments and twists only the
    stalks. The difference is a few millimetres of node position against a
    completely different-looking plant, and carrying a second frame per organ
    to capture it is not worth that.
    """
    if spread_deg < 360.0:
        # Shortest way round, so a flower at 350 degrees turns 10 degrees
        # forward rather than 350 degrees backward.
        signed = (divergence_deg + 180.0) % 360.0 - 180.0
        divergence_deg = signed * (max(0.0, spread_deg) / 360.0)
    return (divergence_deg + jitter_deg) % 360.0


def _bloom_start(inf, index: int, count: int) -> float:
    """Where this flower sits on the bloom timeline, 0 to 1.

    Purely a display ordering: it decides where a flower falls in a sweep the
    Blender side can scrub without regrowing anything. It deliberately does not
    touch `birth_step` -- an earlier build shifted developmental age to encode
    bloom order, which measured as doing nothing at all on a mature plant
    (making an already-expanded flower older cannot expand it further) and is
    gone.
    """
    if count <= 1:
        return 0.0
    u = index / (count - 1)
    if inf.bloom_order in (BloomOrder.BASIPETAL, BloomOrder.CENTRIFUGAL):
        u = 1.0 - u
    return float(min(1.0, max(0.0, u)))


def _cyme_bloom_start(inf, depth: int) -> float:
    """Where a cyme flower sits on the bloom timeline, 0 opening first: by its
    generation, the terminal flower (depth 0) to the last relays'
    (`branching_depth`).

    A cyme matures from its terminal flower outward -- each flower closes its
    axis, and the relays below it come after -- so Middle first (and Top
    first: the terminal flower is also the highest of its cyme) opens depth 0
    first; Outside / Bottom first reverses it. Until this, every flower in a
    cyme sat at 0 and a dichasium opened all at once, whatever Bloom Order
    said."""
    deepest = max(0, inf.branching_depth)
    if deepest == 0:
        return 0.0
    u = min(1.0, max(0.0, depth / deepest))
    if inf.bloom_order in (BloomOrder.ACROPETAL, BloomOrder.CENTRIPETAL):
        u = 1.0 - u
    return float(u)


def _flower_scale(profile: MorphologyProfile, seed: int, inf, organ_id: int) -> float:
    """A flower's size with its Random Range applied, seeded on its own id."""
    return profile.flower_size_of(inf) * (
        1.0 + inf.flower_size_variation * _signed(seed, _T_FLOWER_SIZE, organ_id))


def _cap_with_flower(plant: Plant, profile: MorphologyProfile, seed: int, axis: Axis) -> None:
    axis.apex.state = BudState.RELEASED
    axis.apex.fate = BudFate.FLOWER
    organ_id = plant.new_organ_id()
    axis.terminal_organ = Organ(
        id=organ_id,
        kind=OrganKind.FLOWER,
        birth_step=plant.step,
        inflorescence=_inf_index(profile, axis),
        final_scale=_flower_scale(profile, seed, _inf(profile, axis), organ_id),
        elevation_deg=90.0,
    )


def _close_out(plant: Plant, profile: MorphologyProfile, seed: int) -> None:
    """At full age every axis must have resolved -- structural assertion 3.
    The previous build left 93 buds active-but-unable-to-grow, so they were
    neither drawn nor capped; nothing here may end in that state."""
    # Whether a tip may flower at all (the default: yes -- whether it does is
    # its inflorescence's Terminal attachment, which `_begin_inflorescence`
    # reads); an inflorescence stalk's tip always is its inflorescence.
    floral = profile.vegetativeness().apex_is_floral()
    for axis in plant.growing_axes():
        if axis.flowering_from >= 0 and (floral or _is_inflorescence_axis(axis)):
            # Mid-inflorescence: it has its flowers already, it just stops.
            axis.apex.state = BudState.ABORTED
        elif axis.phytomers and (floral or _is_inflorescence_axis(axis)):
            # **Every tip still growing at the end of the season is
            # given its inflorescence**, whether or not it reached its node
            # count -- including the branches that "ran out of life before
            # completing", which used to stop bare. Which of them *show* is the
            # flower component's masks' decision (an Age window hides the young
            # ones), not a hard-coded rule the user could not reach. They are
            # laid down now, so they read as what they are: young racemes --
            # compact, short-stalked, small-budded (`model._organ_expansion`).
            # Only the season's end does this: nothing flowers early, so growth
            # itself -- lengths, branching, timing -- is exactly as before.
            _begin_inflorescence(plant, profile, seed, axis)
            _preform_region(plant, profile, seed, axis)
            if axis.is_growing:
                axis.apex.state = BudState.ABORTED
        else:
            # A tip that may not flower (a species' own `veg`): it stops.
            axis.apex.state = BudState.ABORTED


def resolve(plant: Plant, profile: MorphologyProfile, catkins: bool = True) -> None:
    """Re-derive world positions using this profile's own elongation and
    expansion curves. Public because Stage 4's physics pass rewrites
    directions and needs the consequences recomputed, and every caller should
    use the same two curves rather than re-deriving which ones apply."""
    resolve_geometry(
        plant,
        elongation=profile.elongation.over(profile.elongation_steps),
        expansion=profile.leaf_expansion.over(profile.leaf_expansion_steps),
        development=lambda axis, fraction: _inf(profile, axis).development_size(fraction),
        catkins=catkins,
    )


def grow(profile: MorphologyProfile | None = None, seed: int = 0, age: float = 1.0,
         environment: Environment | None = None, day: float | None = None,
         signals: growth_signals.Signals | None = None,
         _clock: tuple[int, float] | None = None, _catkins: bool = True) -> Plant:
    """Build the plant at developmental age `age` in [0, 1].

    `environment` is Stage 3: the surroundings this plant responds to. It is
    consulted only where the profile says the species responds at all -- with a
    neutral profile the loop below is exactly the pre-Stage-3 loop, down to the
    per-step resolve, which is skipped entirely.

    `day` is the real-days calendar entry: when
    given it overrides `age` via `profile.progress_for_day`, so the loop still
    runs on developmental progress and everything downstream is unchanged. The
    age path is left intact for the goldens and every existing caller.

    With an Emergence Delay (`profile.emergence_span` > 0) `age` spans
    `total_steps + emergence_span` display steps: growth runs to
    min(that, `total_steps`) exactly as without the delay, then `_reveal`
    holds each delayed part back. `_clock` = (steps, continuous step) grows to
    that structure step with no reveal (`_height_extent`'s measuring grow).
    """
    profile = profile or generic_perennial()
    if day is not None:
        age = profile.progress_for_day(day)
    # The developmental clock as a real (fractional) step count. The age path --
    # goldens and every existing caller -- keeps the integer node count it always
    # had (`steps_for_age` rounds) and clocks the final resolve on that integer,
    # so it stays byte-identical. The day path (the interactive scrub) grows one
    # whole node per whole step reached (floor) and hands the final resolve the
    # fractional `t`, so the growing tip elongates in smoothly between days rather
    # than a node popping to full size. At a clean age -- t integer, which every
    # golden's age 1.0 is -- floor == round == t, so the two paths and the
    # pre-phase-2 output all coincide to the bit.
    span = profile.emergence_span if _clock is None else 0
    show = None
    if _clock is not None:
        steps, continuous_step = _clock
    elif day is None:
        if span:
            # Emergence Delay: the display clock, rounded as the age path is.
            show = float(round(min(1.0, max(0.0, age)) * (profile.total_steps + span)))
            steps = int(min(show, profile.total_steps))
        else:
            steps = profile.steps_for_age(age)
        continuous_step = -1.0
    else:
        t = min(1.0, max(0.0, age)) * (profile.total_steps + span)
        if span:
            show = t
            t = min(t, float(profile.total_steps))
        steps = int(t)
        continuous_step = t

    responds = _responds_to(profile, environment)
    reads_light = responds and profile.reads_light

    # Growth maps (growth/maps.py). A map reading `height` or `shade` needs the
    # plant's positions each step, so it asks for the same per-step resolve a
    # responding plant already does -- and `shade` for the light field too, even
    # on a plant that does not itself respond to light (the map is the only
    # reader). Both flags are False without such a map, so the loop below is
    # the unchanged one.
    # A custom attribute or class counts by what it reads in turn (a Noise or
    # Distance attribute needs positions too).
    keys = set(growth_maps.channel_keys(profile))
    map_positions = any(growth_signals.reads_position(signals, k) for k in keys)
    map_shade = environment is not None and any(
        growth_signals.reads_position(signals, k, wanted=("shade",)) for k in keys)

    plant = Plant(step=0)
    frame = None
    if _uses_maps(profile):
        # `height` is read against the whole plant at this age, not the plant
        # as it stands mid-growth: a branch is decided the step after its node
        # forms, when that node is always the top of the plant so far -- every
        # bud read ~1 and a Min 0.5 kept them all. Growth places each node where
        # it will be when mature, so against the whole extent a node's height is
        # its place on the finished (pre-bend) plant.
        extent = None
        if any(growth_signals.reads_position(signals, k, wanted=("height",)) for k in keys):
            extent = _height_extent(profile, seed, age, environment, day, signals,
                                    (steps, continuous_step) if show is not None else None)
        frame = _MapFrame(profile, environment, signals, extent)
        setattr(plant, _MAP_FRAME_ATTR, frame)
    _emerge_basal_stems(plant, profile, seed)

    # The Day each step stands for (the day path): flowering opens on the bloom
    # DATE in the plant's own history -- max(date, ready) -- so a plant grown
    # to a later Day is the earlier one grown further. Reading the target Day
    # at every step let a raceme form steps before its date the moment the
    # target passed it, and pop in whole.
    clock = profile.total_steps + profile.emergence_span
    for step in range(1, steps + 1):
        plant.step = step
        plant_age = step / max(profile.total_steps, 1)
        step_day = (None if day is None
                    else profile.day_for_progress(step / max(clock, 1)))
        _emerge_basal_stems(plant, profile, seed)
        # Both the light field and every obstacle test read world positions,
        # and positions are otherwise only derived once at the end. A
        # responding plant therefore resolves each step.
        #
        # Deliberately without the elongation curve -- every part sits where it
        # will be when mature, not where it is mid-stretch. Otherwise a tip
        # cleared against an obstacle gets pushed into it later by the stem
        # below it still elongating, which is precisely how two nodes ended up
        # inside a slab in testing: the check was honest at the time and the
        # answer stopped being true afterwards. Deciding against final
        # positions makes "nothing ends up inside solid matter" an invariant
        # instead of a likelihood. The cost is that a young plant's foliage
        # shades at its mature size; the final resolve below is unaffected,
        # since it recomputes everything with the real curves.
        if responds or map_positions:
            resolve_geometry(plant)
            if reads_light or map_shade:
                environment.update_light(plant)
            if frame is not None and map_positions:
                frame.measure(plant)
        _extend_apices(plant, profile, seed, environment if responds else None)
        # A tip that forks at the node it just built does so before that
        # node's buds are released, so the forks -- not a side branch -- carry
        # the tip on (`_chance_fork` spends its buds). Also before determinate
        # flowering: a forking axis forks while it still can, and only flowers
        # when the depth cap stops the fork -- "determinate lands on the final
        # tips".
        _chance_forks(plant, profile, seed)
        _release_buds(plant, profile, seed, plant_age,
                      environment if responds else None, day=step_day)
        _release_old_wood_inflorescences(plant, profile, seed)
        _release_basal_inflorescences(plant, profile, seed)
        # A sympodial unit that has built its nodes relays and ends -- before
        # determinate flowering, which would otherwise take its apex first.
        _end_units(plant, profile, seed, environment if responds else None)
        _terminate_determinate_axes(plant, profile, seed, plant_age, day=step_day,
                                    environment=environment if responds else None)

    if (age >= 1.0 if show is None and _clock is None
            else steps >= profile.total_steps):
        _close_out(plant, profile, seed)
    # Every axis states its inflorescence from here on, for the passes after
    # growth (`MorphologyProfile.inflorescence_of`).
    for axis in plant.axes.values():
        axis.inflorescence = _inf_index(profile, axis)
    if frame is not None:
        # Scratch state of this run only; don't hand a plant back carrying an
        # environment reference nothing downstream reads.
        delattr(plant, _MAP_FRAME_ATTR)

    _trim_unborn(plant, continuous_step if continuous_step >= 0.0 else float(steps))
    if show is not None:
        _reveal(plant, show, continuous_step if continuous_step >= 0.0 else float(steps))
    # Clock the final resolve on the fractional step (day path) or leave the
    # sentinel so it clocks on the integer step (age path); see the block above
    # and resolve_geometry's `ref_step`.
    plant.continuous_step = continuous_step
    # (`build` sweeps the catkins later, once their droop is known.)
    resolve(plant, profile, catkins=_catkins)
    return plant


def _responds_to(profile: MorphologyProfile, environment: Environment | None) -> bool:
    """Whether Stage 3 does anything at all for this plant. False costs
    nothing: no per-step resolve, no light field, no field sampling -- and
    guarantees identical output to before Stage 3 existed."""
    if environment is None:
        return False
    return profile.reads_light or bool(environment.effectors) or (
        profile.obstacle_clearance > 0.0 and environment.obstacles is not None
    )


def _sun_turn(face, target, strength):
    """The turn that takes `face` `strength` of the way toward the unit
    `target`: (unit axis, degrees), or None when there is nothing to turn
    (already aligned, exactly opposed, or no strength)."""
    face = np.asarray(face, dtype=float)
    axis = _cross(face, target)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS or strength <= 0.0:
        return None
    angle = float(np.degrees(np.arccos(
        np.clip(float(np.dot(face, target)), -1.0, 1.0)))) * min(1.0, strength)
    return axis / norm, angle


def _reaim_organ(face, partner, target, strength):
    """Rotate the rigid (face, partner) frame so `face` turns `strength` of the
    way toward the unit `target`, carrying `partner` with it so the frame stays
    orthonormal. The one operation heliotropism needs -- point an organ's chosen
    axis at the sun without shearing its frame."""
    face = np.asarray(face, dtype=float)
    partner = np.asarray(partner, dtype=float)
    turn = _sun_turn(face, target, strength)
    if turn is None:
        return face, partner
    axis, angle = turn
    return rotate_about(face, axis, angle), rotate_about(partner, axis, angle)


def _turned(value, pivot, axis, angle, to=None):
    """A point -- or a (nested) list of them: a pedicel polyline, a catkin
    bunch -- turned about `pivot`, then moved with it to `to` (default: it
    stays)."""
    if isinstance(value, list):
        return [_turned(v, pivot, axis, angle, to) for v in value]
    return ((pivot if to is None else to)
            + rotate_about(np.asarray(value, dtype=float) - pivot, axis, angle))


def _carry_organ(organ, pivot, axis, angle, to=None) -> None:
    """Move an organ rigidly with the stalk it sits on: where it is, which way
    it faces, and its own stalk -- turned about `pivot`, which goes to `to`."""
    organ.position = _turned(organ.position, pivot, axis, angle, to)
    organ.direction = rotate_about(np.asarray(organ.direction, dtype=float), axis, angle)
    organ.normal = rotate_about(np.asarray(organ.normal, dtype=float), axis, angle)
    if organ.pedicel_points:
        organ.pedicel_points = _turned(organ.pedicel_points, pivot, axis, angle, to)
    if organ.catkin_bodies:
        organ.catkin_bodies = _turned(organ.catkin_bodies, pivot, axis, angle, to)


def _arc_parts(lengths, angle: float) -> list[int]:
    """How many straight pieces each stalk segment is cut into so a bend of
    `angle` along the whole stalk turns no more than `_MAX_ARC_FACET_DEG` at
    any joint -- the facet `physics.smooth_oriented_arcs` holds a grown arc
    to. A segment's share follows its length."""
    total = float(sum(lengths))
    need = max(1, math.ceil(abs(angle) / _MAX_ARC_FACET_DEG - 1e-9))
    if total <= _EPS:
        return [1] * len(lengths)
    return [max(1, math.ceil(need * float(n) / total - 1e-9)) for n in lengths]


def _arc_turns(lengths, angle: float) -> list[float]:
    """The turn each piece of a stalk takes for the stalk to bend by `angle`
    as one even arc: `angle` x the piece's midpoint's share of the length. So
    the piece at the node barely turns (the stalk leaves it as before) and the
    last one almost the full angle (it meets its turned flower), every joint
    turning the same share between."""
    total = float(sum(lengths))
    if total <= _EPS:
        return [0.0] * len(lengths)
    turns, run = [], 0.0
    for n in lengths:
        turns.append(angle * (run + 0.5 * float(n)) / total)
        run += float(n)
    return turns


def _bent_polyline(points, turn_axis, angle):
    """`points` (a stalk from its fixed first point) cut by `_arc_parts` and
    bent by `angle` as one even arc (`_arc_turns`): each piece keeps its
    length and turns about `turn_axis`. Returns (the new points, each piece's
    turn)."""
    points = [np.asarray(p, dtype=float) for p in points]
    pieces = []
    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:], strict=False)]
    for (a, b), parts in zip(zip(points, points[1:], strict=False),
                             _arc_parts(lengths, angle), strict=True):
        pieces.extend([(b - a) / parts] * parts)
    turns = _arc_turns([float(np.linalg.norm(v)) for v in pieces], angle)
    out = [points[0]]
    for v, turn in zip(pieces, turns, strict=True):
        out.append(out[-1] + rotate_about(v, turn_axis, turn))
    return out, turns


def _split_straight(axis, start: int, parts_for: list[int]) -> None:
    """Cut each phytomer from `start` on into `parts_for[k]` straight pieces,
    in place: the same line, more points, so it can bend smoothly. The pieces
    are split arc points (`is_split`) but the last, which keeps the node -- its
    organs, buds and index in `flowering_from`; the radius steps evenly from
    the one below. No branch leaves these (the caller checked)."""
    head, rebuilt, remap = axis.phytomers[:start], [], {}
    for k, phytomer in enumerate(axis.phytomers[start:]):
        parts = max(1, parts_for[k])
        i = start + k
        below = axis.phytomers[i - 1].radius if i > 0 else phytomer.radius
        base = np.asarray(phytomer.base_position, dtype=float)
        span = np.asarray(phytomer.tip_position, dtype=float) - base
        for j in range(parts):
            last = j == parts - 1
            piece = Phytomer(
                index=0, birth_step=phytomer.birth_step,
                final_length=phytomer.final_length / parts,
                direction=np.array(phytomer.direction, dtype=float),
                reference=np.array(phytomer.reference, dtype=float),
                factor=phytomer.factor,
                organs=phytomer.organs if last else [],
                buds=phytomer.buds if last else [],
                is_split=phytomer.is_split or not last,
                base_position=base + span * (j / parts),
                tip_position=base + span * ((j + 1) / parts))
            piece.length = phytomer.length / parts
            piece.radius = below + (phytomer.radius - below) * ((j + 1) / parts)
            piece.debug_curvature = phytomer.debug_curvature
            piece.debug_load = phytomer.debug_load
            piece.debug_stiffness = phytomer.debug_stiffness
            piece.debug_moment = phytomer.debug_moment
            piece.lying = phytomer.lying
            rebuilt.append(piece)
        remap[i] = start + len(rebuilt) - 1
    axis.phytomers = head + rebuilt
    for j, phytomer in enumerate(axis.phytomers):
        phytomer.index = j
        for bud in phytomer.buds:
            if bud.phytomer_index >= start:
                bud.phytomer_index = j
    if axis.flowering_from >= start:
        axis.flowering_from = remap.get(axis.flowering_from, axis.flowering_from)


def _bend_top_internode(axis, branched: set, turn_axis, angle) -> bool:
    """Bend `axis`'s top internode -- its last node's phytomer with any split
    parts leading up to it -- by `angle` as one even arc from the node below
    (`_bent_polyline`), cut into enough pieces first (`_split_straight`) that
    it curves rather than kinks. The node it ends in, and everything on it
    (organs, buds, apex, the terminal flower), moves rigidly with the last
    piece. Only a free top internode bends: if a branch leaves it (`branched`
    holds (axis id, phytomer index)), moving it would tear that branch off or
    swing it, so it stays put and False is returned."""
    if not axis.phytomers:
        return False
    start = len(axis.phytomers) - 1
    while start > 0 and axis.phytomers[start - 1].is_split:
        start -= 1
    if any((axis.id, i) in branched for i in range(start, len(axis.phytomers))):
        return False
    lengths = [float(np.linalg.norm(np.asarray(p.tip_position, dtype=float)
                                    - np.asarray(p.base_position, dtype=float)))
               for p in axis.phytomers[start:]]
    parts = _arc_parts(lengths, angle)
    if any(n > 1 for n in parts):
        _split_straight(axis, start, parts)
    group = axis.phytomers[start:]
    old_tip = np.array(group[-1].tip_position, dtype=float)
    points, turns = _bent_polyline(
        [group[0].base_position] + [p.tip_position for p in group], turn_axis, angle)
    for k, (phytomer, turn) in enumerate(zip(group, turns, strict=True)):
        phytomer.direction = rotate_about(phytomer.direction, turn_axis, turn)
        phytomer.reference = rotate_about(phytomer.reference, turn_axis, turn)
        phytomer.base_position, phytomer.tip_position = points[k], points[k + 1]
    tip, turn = points[-1], turns[-1]
    for organ in group[-1].organs:
        _carry_organ(organ, old_tip, turn_axis, turn, tip)
    for bud in group[-1].buds:
        bud.position = _turned(bud.position, old_tip, turn_axis, turn, tip)
        bud.direction = rotate_about(bud.direction, turn_axis, turn)
    if axis.apex is not None:
        axis.apex.position = _turned(axis.apex.position, old_tip, turn_axis, turn, tip)
        axis.apex.direction = rotate_about(axis.apex.direction, turn_axis, turn)
    if axis.terminal_organ is not None:
        _carry_organ(axis.terminal_organ, old_tip, turn_axis, turn, tip)
    return True


def _bend_pedicel(organ, anchor, turn_axis, angle) -> None:
    """Bend a flower's own stalk by `angle` as one even arc from where it
    leaves its node (`_bent_polyline`): the node stays put, the flower sits on
    the stalk's new end."""
    anchor = np.asarray(anchor, dtype=float)
    points = organ.pedicel_points or [anchor, np.asarray(organ.position, dtype=float)]
    bent, _turns = _bent_polyline(points, turn_axis, angle)
    # A straight stalk left in one piece stays pointless: the bridge draws it
    # from its node with a base stub of its own.
    if organ.pedicel_points or len(bent) > 2:
        organ.pedicel_points = bent
    organ.position = np.array(bent[-1], dtype=float)


def _flower_sun(environment: Environment, index: int):
    """The sun inflorescence `index`'s flowers turn to: its flower component's
    own Sun Object (`flower_sun_targets`), or -- on a plant with no flower
    component -- the plant's."""
    targets = environment.flower_sun_targets
    if targets:
        target = targets[index] if 0 <= index < len(targets) else None
    else:
        target = environment.sun_target
    return None if target is None else np.asarray(target, dtype=float)


def _heliotrope_flower(organ, sun, strength, stalk) -> None:
    """Turn one flower `strength` of the way to `sun`, its stalk curving with
    it. `stalk` bends the stalk by an angle (the free top internode of the
    axis it caps, or its own pedicel), where it can.

    **The stalk bends as one even arc by the flower's whole turn** -- it
    leaves its node as it did and arrives turned with its flower, every joint
    on the way turning the same few degrees (no more than
    `_MAX_ARC_FACET_DEG`). Its end moves as the chord of that arc does, half
    the turn; the flower still ends facing exactly where heliotropism aims it.
    Turning the straight stalk rigidly by that half (the first version) put
    the whole half turn into one kink at the node below."""
    turn = _sun_turn(organ.direction, sun, strength)
    if turn is None:
        return
    axis, angle = turn
    face = rotate_about(np.asarray(organ.direction, dtype=float), axis, angle)
    normal = rotate_about(np.asarray(organ.normal, dtype=float), axis, angle)
    stalk(axis, angle)
    organ.direction, organ.normal = face, normal


def apply_heliotropism(plant: Plant, profile: MorphologyProfile,
                       environment: Environment | None) -> None:
    """Re-aim organ facing toward the sun -- floral and foliar heliotropism.

    A final, post-everything pass: it turns each organ's *face* toward the sun.
    That separation is the point -- a nodding sunflower's stalk hangs where
    gravity left it, but its head still faces the sun -- and it is why this
    runs after physics and the final resolve rather than during growth.
    `light_seeking` turns growing *shoots* toward a light gradient; this turns
    the *organs* on them toward a placed sun.

    The sun is an object the user picks to track: a flower component's own Sun
    Object for its inflorescence's flowers (`environment.flower_sun_targets`),
    the plant's for the leaves (`environment.sun_target`) -- each the
    plant-local direction toward it, a scene reference, not a species trait, so
    no object picked is a no-op; the strengths *are* species traits, so a
    sunflower tracks and a foxglove does not. Zero strength, or no target, is
    byte-identical.

    Flowers align their facing (`direction`); leaves align their blade (`normal`,
    the broad face) -- diaheliotropic, the light-gathering pose. Distinct from
    the `ALIGN` effector, which twists the *growth frame* during growth (the
    phyllotactic azimuth, branches with it) and so rotates the whole leaf set
    rather than pointing each blade at the sun.

    **A flower's own stalk bends with it, and nothing else moves**
    (`_heliotrope_flower`), as an even arc: a flower capping its axis bends
    the axis's top internode from the node below -- but only where no branch
    leaves that internode (a dichasium's central flowers sit where their cyme
    forks, and stay straight); a flower on a pedicel bends the pedicel from
    where it leaves its node. The stem under them, and every node below, stays where
    physics put it. A capped flower is turned before the flowers on its axis's
    nodes, which ride along with a bent internode and then turn on their own.
    Leaves turn last, so a leaf on a bent internode faces the sun from where it
    was carried.
    """
    leaf_k = profile.leaf_heliotropism
    if not profile.tracks_sun or environment is None:
        return
    branched = {(a.parent_axis_id, a.parent_phytomer_index)
                for a in plant.axes.values() if a.parent_axis_id >= 0}
    for axis in plant.axes.values():
        flowers = []
        terminal = axis.terminal_organ
        if terminal is not None and terminal.kind == OrganKind.FLOWER:
            flowers.append((terminal, None))
        for phytomer in axis.phytomers:
            flowers.extend((o, phytomer) for o in phytomer.organs
                           if o.kind == OrganKind.FLOWER)
        for organ, phytomer in flowers:
            if organ.scale <= 0.0:
                continue
            # Each flower at its own inflorescence's strength, toward its sun.
            index = max(organ.inflorescence, 0)
            strength = profile.inflorescence_at(index, axis.order).heliotropism
            sun = _flower_sun(environment, index)
            if strength <= 0.0 or sun is None:
                continue
            if phytomer is None:
                def stalk(turn_axis, angle, axis=axis):
                    _bend_top_internode(axis, branched, turn_axis, angle)
            else:
                anchor = phytomer.tip_position
                if organ.along_internode < 1.0:
                    anchor = (phytomer.base_position + (phytomer.tip_position
                              - phytomer.base_position) * organ.along_internode)
                anchor = np.asarray(anchor, dtype=float)
                has_stalk = (not organ.catkin_bodies and float(np.linalg.norm(
                    np.asarray(organ.position, dtype=float) - anchor)) > _EPS)

                def stalk(turn_axis, angle, organ=organ, anchor=anchor, has_stalk=has_stalk):
                    if has_stalk:
                        _bend_pedicel(organ, anchor, turn_axis, angle)
            _heliotrope_flower(organ, sun, strength, stalk)
    if leaf_k <= 0.0 or environment.sun_target is None:
        return
    sun = np.asarray(environment.sun_target, dtype=float)
    for organ in plant.all_organs():
        if organ.scale > 0.0 and organ.kind == OrganKind.LEAF:
            organ.normal, organ.direction = _reaim_organ(
                organ.normal, organ.direction, sun, leaf_k)


_T_LEVEL_RADIUS = 101
"""A level's width variation, per axis -- the stream the stems' old Radius
Variation was drawn on, so a clump's stems keep their spread across the move."""
_T_AXIS_LENGTH = 102
"""A level's Length Variation, once per axis (`_internode_length`)."""
_T_LEVEL_SEED = 103
"""Mixes a level's own Seed with the plant's (`_level_seed`)."""
_T_BASE_X = 104
_T_BASE_Y = 105
_T_START_PITCH = 106
_T_START_YAW = 107
_T_START_ROLL = 108
"""A level's Start Pitch / Yaw / Roll Variation, per stem or branch."""
"""A stem's foot off its place in the clump (Spread Variation, `_emerge_basal_stems`)."""


def _taper_steps(axis: Axis) -> list[float]:
    """How far down its level's taper (Width / Step) each phytomer steps: one
    per node, a node split into parts (`physics._split_phytomers`) sharing its
    step among them by length, so a split adds no taper of its own."""
    steps: list[float] = []
    group: list = []
    for i, phytomer in enumerate(axis.phytomers):
        group.append(phytomer)
        if phytomer.is_split and i < len(axis.phytomers) - 1:
            continue
        total = sum(p.final_length for p in group)
        for p in group:
            steps.append(p.final_length / total if total > 0.0 else 1.0 / len(group))
        group = []
    return steps


_TAPER_BY_LENGTH = True
"""A level's Width / Step narrowing spread evenly by length between an axis's
first node and its last before any flowers (`_level_radii`); False: per node."""


def _rachis_taper(profile: MorphologyProfile, axis: Axis, start: float,
                  slope: float | None = None) -> list[float]:
    """The flowering region's radii (`flowering_from` on), from `start` -- the
    rachis where the region begins -- down to its top flower's stalk width by
    length, so the rachis ends as thin as the stalk it ends beside.

    `slope` is the stem's taper where the region begins (d ln r / d length):
    the rachis sets off at that rate and bends onto its own -- the log radius a
    cubic Hermite from the stem's slope to the region's even rate -- so the
    taper has no corner where the flowers start. (Even by length from the
    first flower, a branch narrowing 5% a node below its flowers went on at
    2.5% above them: a step in the silhouette a third of the way up.) The
    stem's slope is held within three times the even rate, the most a
    monotone cubic takes.

    The level's Width / Step does not reach it: a region's nodes are the
    inflorescence's spacing, a fraction of the stem's, so a per-node taper ran
    far too fast, and a per-length one barely narrowed and left a rachis
    ending at the stem's width beside stalks half as thick. The stalks are
    read against `start` too (`Axis.stalk_radius`), so each keeps its own
    Pedicel Width and the top one meets the rachis's tip."""
    inf = _inf(profile, axis)
    end = start * min(1.0, max(1e-3, inf.pedicel_width * inf.pedicel_width_profile(1.0)))
    f = axis.flowering_from
    run = [p.final_length for p in axis.phytomers[f:]]
    total = sum(run)
    drop = math.log(end / start)                  # <= 0
    m0 = drop if slope is None else min(0.0, max(3.0 * drop, slope * total))
    out, along = [], 0.0
    for length in run:
        along += length
        u = along / total if total > 0.0 else 1.0
        # log radius: Hermite from (0, slope m0) to (drop, slope = the even rate)
        h10, h01, h11 = u ** 3 - 2.0 * u * u + u, -2.0 * u ** 3 + 3.0 * u * u, u ** 3 - u * u
        out.append(start * math.exp(h10 * m0 + h01 * drop + h11 * drop))
    return out


def _level_radii(plant: Plant, profile: MorphologyProfile, seed: int) -> None:
    """Widths from the branching levels: node i of an axis is its base x Width
    / Step ** i, and its flowering region tapers on to its top stalk's width
    (`_rachis_taper`). A stem's base is the stems level's Start Width, in
    metres; a branch's is its parent's radius at the node it leaves x Start
    Width (either x the level's width variation); a fork continues its
    parent's taper. Topological order, so every parent is final before its
    children read it. Axes no level drives (inflorescence branches) are sized
    after, off these."""
    for axis in plant.axes_in_topological_order():
        level = _level(profile, axis)
        if level is None or not axis.phytomers:
            continue
        parent = plant.axes.get(axis.parent_axis_id)
        if parent is None or not parent.phytomers:
            base = level.start_width_scale        # the stems' base radius, metres
        elif axis.is_continuation:
            # A continuation starts at exactly the tip it grows on from, and
            # tapers on by its own level's Width / Step: no step at the joint.
            base = parent.phytomers[-1].radius
        else:
            at = parent.phytomers[min(max(axis.parent_phytomer_index, 0),
                                      len(parent.phytomers) - 1)].radius
            base = at * (level.width_scale_per_step if axis.is_fork
                         else level.start_width_scale)
        if level.width_variation > 0.0 and not axis.is_fork and not axis.is_continuation:
            # Width / Step's variation: once per axis, on its base, so axes
            # differ from each other and no taper goes ragged (a fork continues
            # its parent's).
            base *= 1.0 + level.width_variation * _signed(_level_seed(seed, level),
                                                          _T_LEVEL_RADIUS, axis.id)
        if not axis.is_continuation:
            base *= axis.map_width                # the level's Start Width map
        base *= axis.vigour                       # an Unequal Fork's weak fork
        f = axis.flowering_from
        radii, stepped = [], -1.0                 # the first node is the base
        for step in _taper_steps(axis)[:f if f >= 0 else None]:
            stepped += step
            radii.append(base * level.width_scale_per_step ** max(0.0, stepped))
        if _TAPER_BY_LENGTH and len(radii) > 2:
            # Width / Step's narrowing, spread by length: the same width at the
            # first node and at the last, evenly per metre between. (Per node,
            # internodes shrinking toward the flowers -- 20, 9, 5, 2, 1, 0.6,
            # 0.3 mm -- crammed a quarter of the width into the last few mm: a
            # sudden step a third of the way up the branch.)
            run = [p.final_length for p in axis.phytomers[1:len(radii)]]
            total = sum(run)
            if total > 0.0:
                ratio, along = radii[-1] / radii[0], 0.0
                for i, length in enumerate(run, start=1):
                    along += length
                    radii[i] = radii[0] * ratio ** (along / total)
        if f >= 0:
            start = radii[-1] if radii else base
            slope = None
            if len(radii) >= 2 and axis.phytomers[f - 1].final_length > 0.0:
                # the stem's taper rate coming into the region, per metre
                slope = (math.log(radii[-1] / radii[-2])
                         / axis.phytomers[f - 1].final_length)
            radii += _rachis_taper(profile, axis, start, slope)
            axis.stalk_base_radius = start
        width_mapped = growth_maps.level_has_map(level, growth_maps.WIDTH)
        for i, (phytomer, radius) in enumerate(zip(axis.phytomers, radii, strict=True)):
            if width_mapped:
                # The level's Width map, per node, read at the node as built.
                radius *= _node_map_factor(plant, profile, seed, growth_maps.WIDTH, axis,
                                           i, phytomer.factor, _MAP_KIND_NODE,
                                           axis.id * 4096 + i, stack=level.maps,
                                           birth_step=phytomer.birth_step)
            phytomer.radius = max(1e-6, radius)
    _thicken(plant, profile)


def _axis_growth_so_far(profile: MorphologyProfile, axis: Axis) -> float:
    """How far a stem / branch has grown, 0-1: its vegetative nodes' elongation
    so far over what it will build (`_vegetative_reach`, the last one counting
    for its part) -- or over what it built, once its apex has stopped. 1 once
    it has turned to flowering or relayed on (its vegetative growth is done)."""
    if axis.flowering_from >= 0 or axis.unit_ended:
        return 1.0
    nodes, group = [], []
    for i, phytomer in enumerate(axis.phytomers):      # a split node is one node
        group.append(phytomer)
        if phytomer.is_split and i < len(axis.phytomers) - 1:
            continue
        final = sum(p.final_length for p in group)
        now = sum(p.length for p in group)
        nodes.append(min(1.0, now / final) if final > 0.0 else 1.0)
        group = []
    growing = axis.apex is not None and axis.apex.state == BudState.ACTIVE
    reach = _vegetative_reach(profile, axis) if growing else float(len(nodes))
    if reach <= 0.0:
        return 1.0
    done = sum(e * min(1.0, reach - j) for j, e in enumerate(nodes))
    return min(1.0, done / reach)


def _thicken(plant: Plant, profile: MorphologyProfile) -> None:
    """Stem Thickening: every stem / branch's stated widths x the curve at how
    far it has grown -- so a young stem thickens as it lengthens (a reveal of
    the widths the levels state, not a load- or age-driven girth). A separate
    pass after the stated widths, so a branch still takes its base off its
    parent's STATED width, never a parent thinned in turn. Byte-identical with
    the curve flat at 1, and on a fully grown axis."""
    curve = profile.stem_thickening
    for axis in plant.axes.values():
        if _level(profile, axis) is None or not axis.phytomers:
            continue
        progress = _axis_growth_so_far(profile, axis)
        if progress >= 1.0:
            continue
        factor = min(1.0, max(0.0, float(curve(progress))))
        if factor == 1.0:
            continue
        for phytomer in axis.phytomers:
            phytomer.radius = max(1e-6, phytomer.radius * factor)


def finish_build(plant: Plant, profile: MorphologyProfile, seed: int, age: float,
                 environment: Environment | None, day: float | None,
                 arc_facet_deg: float | None, mode: str, vigour: float,
                 signals: growth_signals.Signals | None = None) -> None:
    """The post-grow pipeline -- thicken, bend, re-resolve, senesce, aim --
    everything `build` runs after `grow`, in the one order every consumer needs
    (physics must run *between* diameter and the final resolve, because bending
    needs radii and positions have to be recomputed after it).

    Extracted from `build` so a caller holding a pre-grown topology (the rig
    bake's grow-once path) can re-run it on a subset without re-growing. `build`
    still calls it, so the behaviour -- and the goldens -- are unchanged.
    """
    from .diameter import compute_radii
    from .inflorescence import solve_flat_head
    from .lifecycle import apply_fruit, apply_lifecycle, kill_all
    from .physics import (
        apply_autotropism,
        apply_branch_bend,
        apply_gravity,
        apply_pedicel_droop,
        settle_on_ground,
        smooth_bent_arcs,
        smooth_oriented_arcs,
    )
    # Age-stiffening: straighten each stem's growth wander toward its setpoint by
    # node age, so a circumnutation nod ends up a live tip on a tidy stem rather
    # than a uniform helix. A post-growth deformation like the gravity bend, run
    # first so the flat solve and the load bends below see the tidied stem. Off
    # (autotropism 0) is byte-identical -- no call, no extra resolve.
    # Every resolve until the catkins' own droop is known (`apply_pedicel_droop`,
    # below) leaves them unswept: no pass before it reads a catkin's shape
    # (physics weighs them at their node), and a bunch swept with no droop
    # was thrown away -- on a hazel in catkin, ~2/3 of its whole build.
    if profile.autotropism > 0.0:
        apply_autotropism(plant, profile)
        resolve(plant, profile, catkins=False)
    # **Before the flat solve, and that ordering is load-bearing.** Both write
    # `pedicel_length` and `elevation_deg`, and for a flat-topped kind the
    # solve is the one that owns them -- so with this running afterwards it
    # silently undid the solve on exactly the axes that had been truncated.
    #
    # Measured on a compound corymb, where it first became visible: the top
    # three branches built 19, 18 and 17 flowers of an intended 20, and came
    # out 66-76 mm from flat against 4-14 mm on the four that finished, their
    # pedicels scattered 24-36 mm instead of the solved 106->17 mm gradient.
    # Reported exactly that way. It reads only counts and the profile, no
    # geometry, so it is free to run this early.
    _renormalise_truncated_regions(plant, profile, seed)
    # Flat-topped heads are a solve against real attachment positions, so
    # they need resolved geometry and a re-resolve once they have changed it.
    # This first pass is what gravity below sees: a corymb's outer flowers sit
    # far out on long pedicels and that is a real part of the load, so bending
    # against unsolved pedicels would bend against the wrong plant.
    solve_flat_head(plant, profile, seed)
    resolve(plant, profile, catkins=False)
    # An oriented inflorescence has already arced by now, so its extra nodes can
    # go exactly where that growth-time bend is; the gravity bend below is
    # subdivided adaptively by `smooth_bent_arcs` after the fact, so nothing
    # needs to guess before the bend which stretch will need the resolution.
    smooth_oriented_arcs(plant, profile)
    # Before gravity, so a branch sags from the shape it holds rather than the
    # two competing for the same joints. See `physics.apply_branch_bend`.
    apply_branch_bend(plant, profile)
    resolve(plant, profile, catkins=False)
    # Flower -> fruit -> drop on the calendar's fruit window (phase 4 backend).
    # Runs HERE, before the pipe model and the bend, so a set fruit's own
    # `fruit_weight` enters `physics._organ_mass` and a fruiting branch bows --
    # then springs back once the window closes. All the flower-shaping above
    # (renormalise, the first flat-head solve, the arcs, branch-bend) has already
    # run on flowers; the passes below that read reproductive organs
    # (`_organ_mass`, `_flowers_of`, `_set_pedicel_droop`, `_organ_expansion`)
    # each treat a set fruit like the flower it was. Day-path only, so off it
    # nothing flips and every plant is byte-identical.
    apply_fruit(plant, profile, day=day)
    width_mapped = any(growth_maps.level_has_map(level, growth_maps.WIDTH)
                       for level in profile.levels)
    if width_mapped:
        # A Width map reads each node as grown: the frame growth used, measured
        # on the grown plant, for the length of the radii pass.
        frame = _MapFrame(profile, environment, signals)
        frame.measure(plant)
        setattr(plant, _MAP_FRAME_ATTR, frame)
    compute_radii(plant, branch_taper=profile.inflorescence.branch_width_profile,
                  branch_taper_of=lambda axis: _inf(profile, axis).branch_width_profile,
                  level_radii=lambda p: _level_radii(p, profile, seed))
    if width_mapped:
        delattr(plant, _MAP_FRAME_ATTR)
    apply_gravity(plant, profile, seed,
                  terrain=environment.ground if environment is not None else None)
    # Each flower's own stalk sags under its own weight, by the same beam -- the
    # physical replacement for the `pedicel_droop` control. Needs radii, so it
    # runs here; `model._sag` draws it at the resolve below.
    apply_pedicel_droop(plant, profile)
    # Adaptive subdivision AFTER the bend, so the extra nodes land exactly on the
    # joints the bend made too sharp -- a hard turn becomes a smooth arc, only
    # where needed, worst-first under a node budget.
    if arc_facet_deg is None:
        smooth_bent_arcs(plant)
    else:
        smooth_bent_arcs(plant, threshold=arc_facet_deg)
    resolve(plant, profile)
    # The arcs run inside their chords: land again what they set into the
    # ground (on the resolved lengths), and resolve once more only if it did.
    if settle_on_ground(plant, terrain=environment.ground if environment is not None else None):
        resolve(plant, profile)
    # **Solved again, against the bent plant, and this is what makes a corymb
    # actually flat.** The first pass is exact -- measured 0.0000 mm across
    # every head -- and then `apply_gravity` bends the stems underneath it and
    # carries the heads along, which put a compound corymb's umbellets 52-91 mm
    # out of level and tipped their lowest pedicels back below horizontal. A
    # head that is only flat before the plant sags is not flat.
    #
    # Safe to run here because it re-aims pedicels and moves nothing else: no
    # stem is touched, so the bend it was measured against still holds. It is
    # the same argument that lets the bend run before it -- each stage owns
    # different fields.
    solve_flat_head(plant, profile, seed)
    resolve(plant, profile)
    # WS3: the post-growth lifecycle pass. Marks organs and axes dead-or-shed
    # from the profile's senescence triggers, reading the fully-built plant and
    # the environment's already-computed light field -- never feeding back into
    # growth. Inert (leaves everything ALIVE) until a trigger is enabled, so
    # every plant is byte-identical until senescence is asked for.
    apply_lifecycle(plant, profile, seed, age, environment, day=day)
    # (Fruit-set moved earlier, before the bend -- see `apply_fruit` above.)
    # Floral/foliar heliotropism: the last thing, a re-aim of organ facing toward
    # a placed sun after everything that could move an organ. No-op unless a
    # species sets a strength AND a SUN effector is in the scene, so off is
    # byte-identical.
    apply_heliotropism(plant, profile, environment)
    # Phase 5. An annual past its one season is dead whole (trunk included);
    # a perennial's later-year size is a subtle uniform scale on the finished
    # plant (QI-a). Both no-ops for a single-season plant (mode 'grow', vigour
    # 1.0), so lifespan None is byte-identical.
    if mode == 'dead':
        kill_all(plant, profile)
    scale_plant(plant, vigour)
    for organ in plant.all_organs():
        organ.__dict__.pop("_catkin_memo", None)    # resolve's scratch (`_catkin_geometry`)


def _strip_height_readers(profile: MorphologyProfile, signals) -> MorphologyProfile:
    """The profile without the maps and level masks that read height (directly
    or through an attribute or class)."""
    def keep(key: str) -> bool:
        return not growth_signals.reads_position(signals, key, wanted=("height",))

    levels = [replace(level, maps=[m for m in level.maps if keep(m.channel)],
                      masks=[e for e in level.masks if e.mute or keep(e.channel)])
              for level in profile.levels]

    def inf_kept(inf):
        if inf is None or not inf.maps:
            return inf
        return replace(inf, maps=[m for m in inf.maps if keep(m.channel)])
    return replace(profile, maps=[m for m in profile.maps if keep(m.channel)], levels=levels,
                   inflorescence=inf_kept(profile.inflorescence),
                   lateral_inflorescence=inf_kept(profile.lateral_inflorescence),
                   extra_inflorescences=[inf_kept(x) for x in profile.extra_inflorescences])


def _height_extent(profile: MorphologyProfile, seed: int, age: float, environment,
                   day, signals, clock: tuple[int, float] | None = None
                   ) -> tuple[float, float]:
    """(lowest z, height span) of this plant at this age with every node at its
    mature place, grown once without the maps and masks that read height (which
    would need this to be known already). What `height` is read against.

    Kept per input -- the stripped profile, seed, age, day, signals -- so
    adjusting a height-reading map or mask (none of which this grow sees)
    reuses it instead of growing the whole plant a second time."""
    stripped = _strip_height_readers(profile, signals)
    key = None
    if environment is None:
        import hashlib
        import json

        from . import serialize
        blob = json.dumps(serialize.profile_to_dict(stripped), sort_keys=True, default=repr)
        key = hashlib.blake2b("\x1f".join((blob, repr((seed, age, day, clock)),
                                            repr(signals)))
                              .encode("utf-8"), digest_size=16).hexdigest()
        if key in _EXTENT_CACHE:
            return _EXTENT_CACHE[key]
    plant = grow(stripped, seed=seed, age=age, environment=environment, day=day,
                 signals=signals, _clock=clock)
    resolve_geometry(plant)
    lo, hi = plant.bounds()
    extent = (float(lo[2]), float(hi[2] - lo[2]))
    if key is not None:
        _EXTENT_CACHE[key] = extent
        while len(_EXTENT_CACHE) > 16:
            _EXTENT_CACHE.pop(next(iter(_EXTENT_CACHE)))
    return extent


_EXTENT_CACHE: dict = {}
"""`_height_extent` results by input digest (oldest dropped past 16)."""


def _uses_maps(profile: MorphologyProfile) -> bool:
    """Whether any plant map, level map or level mask exists -- the one thing
    that makes growth build a map frame."""
    return (bool(profile.maps) or any(level.maps or level.masks for level in profile.levels)
            or any(inf.maps for inf in growth_maps.inflorescences(profile)))


def build(profile: MorphologyProfile | None = None, seed: int = 0, age: float = 1.0,
          environment: Environment | None = None,
          arc_facet_deg: float | None = None, day: float | None = None,
          signals: growth_signals.Signals | None = None) -> Plant:
    """The whole pipeline in the one order every consumer needs: grow the
    topology, then `finish_build` (thicken, bend under its own weight, re-derive
    positions, senesce, aim).

    `arc_facet_deg` overrides how finely `smooth_bent_arcs` draws each bend into
    an arc (default `None` keeps its own default) -- it changes only the *number*
    of intermediate arc points, never the bend, so the resolution-independence
    check can build the same plant twice at two thresholds. Anything that just
    wants a finished plant should call this rather than re-deriving the order.
    """
    profile = profile or generic_perennial()
    # Calendar phase 5: route the real `day` to this year's season. `grow_day` is
    # the day-of-year, so each year's phenology and progress repeat on the same
    # calendar dates; `vigour`/`stem_factor` are this year's subtle size/clump
    # multipliers; `mode` is 'grow', 'bare' (a perennial's dormant off-season) or
    # 'dead' (an annual past its one season). With lifespan None it passes `day`
    # straight through with neutral multipliers -- exactly the pre-phase-5 season.
    grow_day, vigour, stem_factor, mode = profile.season_state(day)
    if mode == 'bare':
        # Dormancy: nothing above ground. The persistent clump is conceptual
        # (roots are a later feature), so an empty plant is the whole of it.
        return Plant(step=0)
    day = grow_day
    if day is not None:
        age = profile.progress_for_day(day)
    roots = set(profile.level_children(-1)) if profile.levels else set()
    if stem_factor != 1.0 and any(profile.levels[i].whorl > 1 for i in roots):
        # A subtly bigger clump in a later year -- more basal stems (a clump
        # only; a single stem stays one). Year 0 keeps the species' count.
        profile = replace(profile, levels=[
            replace(lv, whorl=max(1, round(lv.whorl * stem_factor)))
            if i in roots and lv.whorl > 1 else lv
            for i, lv in enumerate(profile.levels)])
    # `age` is kept for lifecycle; `day` (the day-of-year) is forwarded so grow()
    # takes the continuous scrub path.
    plant = grow(profile, seed=seed, age=age, day=day, environment=environment,
                 signals=signals, _catkins=False)
    finish_build(plant, profile, seed, age, environment, day, arc_facet_deg, mode, vigour,
                 signals=signals)
    return plant
