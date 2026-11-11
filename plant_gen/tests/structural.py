"""The structural assertions -- the eight of the rebuild's spec, plus one
added with each stage that could break the model quietly.

Every one exists because of a specific measured failure. The first eight
guard against failures of the *previous* build that survived a full day of
work undetected; the ninth guards the one way the gravity pass could corrupt
the model silently, since a rendered plant looks plausible whether or not its
topology survived the bend; the tenth guards Stage 3's own discipline, which
is that the environment changes nothing until a species asks it to. The last
three came with the unified gravity system and guard the beam's own outcomes:
that nothing kinks (11), that gravity only ever lowers the plant and never
lifts it (12), and that the adaptive subdivision is fine enough that the shape
barely moves when its resolution is halved (13). They are cheap -- the whole
suite runs in well under a second on a normal plant -- and they run headless,
with no Blender, so there is no excuse for skipping them after a change.

    python -m plant_gen.tests.structural

Two deliberate design choices:

* The checks take an `Expectations` object rather than a `MorphologyProfile`.
  A profile is a Phase B concept and carries far more than the checks need;
  depending on it here would couple the harness to a class that did not exist
  when these were written and would make the checks harder to point at a
  hand-built fixture. Phase B's profile grows an `expectations()` method that
  returns one of these.

* A check that cannot be evaluated yet reports SKIP rather than passing
  quietly. A green run that silently checked nothing is exactly how the
  previous build's problems went unnoticed.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..growth import curves
from ..growth.model import (
    Axis,
    Bud,
    BudBlock,
    BudPosition,
    BudState,
    LeafForm,
    Organ,
    OrganKind,
    Phytomer,
    Plant,
    orthonormal_reference,
    parallel_transport,
    resolve_geometry,
    rotate_about,
)

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


@dataclass
class Expectations:
    """What a given species' output is supposed to look like structurally.
    Kept small on purpose -- these are guardrails against breakage, not a
    description of the plant."""

    max_branch_order: int = 2
    expects_leaves: bool = True
    expects_flowers: bool = False
    single_leader: bool = True
    """Whether one main axis is meant to reach the crown (check 2). False for a
    forking architecture, where the trunk deliberately splits into co-equal
    halves and no single order-0 axis spans the plant -- that is what dichotomy
    *is*, not the stub-trunk pathology the check guards against."""
    bears_leaf_at: Callable[[float, int], bool] = lambda factor, order: True
    """Whether a node at this (factor, branch order) is one the species places a
    leaf on at all. Growth now emits foliage on every vegetative node (placement
    is a render-time component mask), so this is the permissive default and
    check 5 asserts no vegetative phytomer is missing its leaf -- which is the
    original failure it was written for (2 leaves across 598 nodes)."""
    main_axis_height_share: float = 0.7
    """The tallest order-0 axis must reach at least this fraction of the
    plant's full height.

    An earlier draft of this check used order-0's share of *total length*
    plus "the highest point sits on an order <= 1 axis". Both were the wrong
    metric. In any legitimately bushy plant a single trunk is a small
    fraction of total length (in a tree with a thousand branches it is a
    couple of percent, correctly), and a branch near the top of a stem can
    genuinely overtop the leader by a little. Neither measures what actually
    went wrong: yesterday's main axis reached 0.23 m in a 7.61 m plant --
    3% of its height. Height reach catches that decisively and does not
    punish plants for branching."""


@dataclass
class CheckResult:
    number: int
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != FAIL


# -- 1 ----------------------------------------------------------------------
def check_branch_order(plant: Plant, exp: Expectations) -> CheckResult:
    """Caught: the previous build reaching order 17, because `order` really
    meant 'forks behind me' and diameter was derived from it."""
    worst = plant.max_order
    if worst > exp.max_branch_order:
        offenders = sorted({a.order for a in plant.axes.values() if a.order > exp.max_branch_order})
        return CheckResult(1, "Branch order within declared maximum", FAIL,
                           f"max order {worst} exceeds declared {exp.max_branch_order} "
                           f"(saw orders {offenders})")
    return CheckResult(1, "Branch order within declared maximum", PASS,
                       f"max order {worst} <= {exp.max_branch_order}")


# -- 2 ----------------------------------------------------------------------
def check_main_axis_dominance(plant: Plant, exp: Expectations) -> CheckResult:
    """Caught: a 5-phytomer, 0.23 m main axis in a 7.61 m plant, with 593 of
    598 nodes on laterals. Two independent symptoms, so two sub-tests."""
    if not plant.axes or not plant.basal_axis_ids:
        return CheckResult(2, "Main axis reaches the crown", SKIP, "no axes")
    if not exp.single_leader:
        return CheckResult(2, "Main axis reaches the crown", SKIP,
                           "a forking architecture has no single main axis")

    lo, hi = plant.bounds()
    height = float(hi[2] - lo[2])
    if height <= 1e-6:
        return CheckResult(2, "Main axis reaches the crown", SKIP, "plant has no height yet")

    base_z = float(lo[2])
    # The highest point *along* the axis, not its tip. An axis that arcs over
    # under its own weight, or a catkin whose flowering region deliberately
    # turns downward, still reaches the crown at its apex -- measuring the tip
    # failed a correctly-built catkin.
    main_reach = max(
        (float(p.tip_position[2]) - base_z
         for i in plant.basal_axis_ids for p in plant.axes[i].phytomers),
        default=0.0,
    )
    share = main_reach / height

    if share < exp.main_axis_height_share:
        return CheckResult(2, "Main axis reaches the crown", FAIL,
                           f"tallest order-0 axis reaches {share:.1%} of plant height "
                           f"({main_reach:.3f} m of {height:.3f} m), below "
                           f"{exp.main_axis_height_share:.0%}")
    return CheckResult(2, "Main axis reaches the crown", PASS,
                       f"tallest order-0 axis reaches {share:.1%} of plant height")


# -- 3 ----------------------------------------------------------------------
def check_all_axes_terminate(plant: Plant, at_full_age: bool) -> CheckResult:
    """Caught: 93 buds ending a run still 'active' but unable to grow -- never
    drawn, never capped with a flower, never terminated. Only meaningful at
    full age; mid-development a growing axis is correct."""
    if not at_full_age:
        return CheckResult(3, "Every axis terminates by full age", SKIP,
                           "only checked at Age = 1.0")
    stragglers = [a.id for a in plant.axes.values() if not a.is_terminated]
    if stragglers:
        return CheckResult(3, "Every axis terminates by full age", FAIL,
                           f"{len(stragglers)} axes still growing at Age = 1.0 "
                           f"(ids {stragglers[:8]})")
    return CheckResult(3, "Every axis terminates by full age", PASS,
                       f"all {len(plant.axes)} axes terminated")


# -- 4 ----------------------------------------------------------------------
def check_age_monotonic(grow_fn, ages=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> CheckResult:
    """Caught: the global growth stall. 60 iterations gave 598 nodes and 150
    gave 680, because normalised vigor was tested against an absolute
    threshold, so every bud fell under the bar once enough of them existed."""
    if grow_fn is None:
        return CheckResult(4, "Phytomer count monotonic in Age", SKIP,
                           "no growth function supplied")
    counts = [(a, grow_fn(a).phytomer_count) for a in ages]
    for (a0, c0), (a1, c1) in zip(counts, counts[1:], strict=False):
        if c1 < c0:
            return CheckResult(4, "Phytomer count monotonic in Age", FAIL,
                               f"Age {a0} -> {c0} phytomers but Age {a1} -> {c1}")
    if counts[-1][1] <= counts[0][1]:
        return CheckResult(4, "Phytomer count monotonic in Age", FAIL,
                           f"Age {ages[0]} and Age {ages[-1]} both give {counts[-1][1]} phytomers "
                           "-- the Age dial is inert")
    return CheckResult(4, "Phytomer count monotonic in Age", PASS,
                       " -> ".join(f"{a:g}:{c}" for a, c in counts))


# -- 5 ----------------------------------------------------------------------
def _is_foliage(organ: Organ) -> bool:
    """A leaf this check is actually about.

    `OrganKind.LEAF` is not the question, because three things now share it:
    foliage, bracts and catkins. Bracts never reached here (they sit on
    flowering nodes, which check 5 already excludes) but **catkins do** -- they
    hang at ordinary vegetative nodes, beside the foliage, and they have a
    placement window of their own. Measured before this existed: a hazel whose
    leaves were confined to the lower half **failed** on 9 phytomers "outside
    the leaf placement window carrying a leaf", every one of which carried a
    catkin and no leaf, and the passing case reported 84 leaves where 57 were
    leaves and 27 were catkins."""
    return organ.kind == OrganKind.LEAF and organ.form != LeafForm.BRACT


def check_leaves_present(plant: Plant, exp: Expectations) -> CheckResult:
    """Caught: 2 leaves on a 598-node plant at leaf_probability 0.75. Under the
    phytomer model a leaf is structural, so the check is not merely 'some
    leaves exist' but 'no phytomer is missing one'.

    Scoped to *vegetative* phytomers once inflorescences landed: a flowering
    node bears a flower rather than a leaf, which is what an inflorescence is.
    That is not a loophole -- the original failure was 2 leaves across 598
    nodes that were all vegetative, and it still trips this.

    Scoped again once inflorescences could branch: an inflorescence *branch* is
    entirely inflorescence, so the bare stretch below its flowers carries no
    foliage either. Its phytomers pass the `flowering_from` test above (they
    are before their own flowering region) and would otherwise read as
    leaf-bearing -- which is how this caught the compound forms live, five
    plants failing on exactly the nodes the model had just, deliberately,
    stopped putting leaves on. See `development._carries_foliage`."""
    label = "Leaves on every leaf-bearing phytomer"
    if not exp.expects_leaves:
        return CheckResult(5, label, SKIP, "profile places no leaves")
    vegetative = [
        (axis, p)
        for axis in plant.axes.values()
        for i, p in enumerate(axis.phytomers)
        if axis.structural_depth < 1 and (axis.flowering_from < 0 or i < axis.flowering_from)
        and not p.is_split
    ]
    eligible = [p for axis, p in vegetative if exp.bears_leaf_at(p.factor, axis.order)]
    if not eligible:
        return CheckResult(5, label, SKIP, "no leaf-bearing phytomers yet")
    bare = [p for p in eligible if not any(_is_foliage(o) for o in p.organs)]
    if bare:
        return CheckResult(5, label, FAIL,
                           f"{len(bare)} of {len(eligible)} leaf-bearing phytomers carry no leaf")
    # The converse matters just as much: a node the species excludes must not
    # have grown a leaf anyway, or placement is being ignored rather than
    # applied.
    excluded = [p for axis, p in vegetative if not exp.bears_leaf_at(p.factor, axis.order)]
    stray = [p for p in excluded if any(_is_foliage(o) for o in p.organs)]
    if stray:
        return CheckResult(5, label, FAIL,
                           f"{len(stray)} phytomers outside the leaf placement window carry a leaf")
    # Counted on the eligible phytomers rather than plant-wide, or the number
    # stops matching the sentence it sits in: bracts are `OrganKind.LEAF` too,
    # and they live on exactly the flowering nodes this check has just excluded.
    total = sum(1 for p in eligible for o in p.organs if _is_foliage(o))
    note = f"{total} leaves across {len(eligible)} leaf-bearing phytomers"
    if excluded:
        note += f" ({len(excluded)} bare by placement)"
    flowering = plant.phytomer_count - len(vegetative)
    if flowering:
        note += f" ({flowering} flowering nodes excluded)"
    return CheckResult(5, label, PASS, note)


# -- 6 ----------------------------------------------------------------------
def check_determinate_flowers(plant: Plant, exp: Expectations, at_full_age: bool) -> CheckResult:
    """Caught: FlowerCapInflorescence running zero times in a default run --
    no node ever reached TERMINAL, so Determinate vs Indeterminate was a
    no-op."""
    if not exp.expects_flowers:
        return CheckResult(6, "Determinate plant flowers", SKIP, "profile expects no flowers")
    if not at_full_age:
        return CheckResult(6, "Determinate plant flowers", SKIP, "only checked at Age = 1.0")
    # Catkins count. They are flowers -- a dense sessile spike of them -- and
    # only their *body* differs, being a swept tube rather than instanced
    # blooms. Counting only `FLOWER` failed a hazel for not flowering, which it
    # plainly does. Before the three-layer split this never arose, because
    # `CATKIN` forced `Topology.NONE` and the check skipped the profile
    # entirely; now that attachment says which stems flower, morphology has to
    # stop implying it.
    flowers = len(plant.organs_of_kind(OrganKind.FLOWER))
    catkins = len(plant.organs_of_kind(OrganKind.CATKIN))
    if flowers + catkins == 0:
        return CheckResult(6, "Determinate plant flowers", FAIL,
                           "no flowers or catkins at Age = 1.0 on a flowering profile")
    detail = f"{flowers} flowers" if not catkins else f"{flowers} flowers, {catkins} catkins"
    return CheckResult(6, "Determinate plant flowers", PASS, detail)


# -- 7 ----------------------------------------------------------------------
def check_radius_monotonic(plant: Plant) -> CheckResult:
    """Caught: a 430:1 thickness range and a 0.116 mm thinnest tube, from
    `base_width * taper^order` with order running to 17. Skips until the
    pipe-model pass (Phase C) actually writes radii."""
    phytomers = plant.all_phytomers()
    if not phytomers or all(p.radius <= 0.0 for p in phytomers):
        return CheckResult(7, "Radius non-increasing base to tip", SKIP,
                           "no radii written yet (pipe-model pass is Phase C)")
    for axis in plant.axes.values():
        for prev, nxt in zip(axis.phytomers, axis.phytomers[1:], strict=False):
            if nxt.radius > prev.radius + 1e-9:
                return CheckResult(7, "Radius non-increasing base to tip", FAIL,
                                   f"axis {axis.id}: radius rises from {prev.radius:.5f} to "
                                   f"{nxt.radius:.5f} at phytomer {nxt.index}")
    radii = [p.radius for p in phytomers]
    return CheckResult(7, "Radius non-increasing base to tip", PASS,
                       f"range {min(radii):.5f}-{max(radii):.5f} m "
                       f"({max(radii)/max(min(radii), 1e-9):.0f}:1)")


# -- 8 ----------------------------------------------------------------------
def check_determinism(grow_fn, age: float = 1.0) -> CheckResult:
    """Not a past failure -- the previous build was genuinely deterministic and
    this guards that property through the rewrite."""
    if grow_fn is None:
        return CheckResult(8, "Same seed reproduces the same plant", SKIP,
                           "no growth function supplied")
    a, b = grow_fn(age), grow_fn(age)
    fa, fb = _fingerprint(a), _fingerprint(b)
    if fa != fb:
        return CheckResult(8, "Same seed reproduces the same plant", FAIL,
                           "two runs at the same seed differ")
    return CheckResult(8, "Same seed reproduces the same plant", PASS,
                       f"identical across two runs ({a.phytomer_count} phytomers)")


# -- 9 ----------------------------------------------------------------------
def check_physics_preserves_topology(profile=None, seed: int = 0) -> CheckResult:
    """Stage 4 bends the structure; it must not restructure it. Same axes,
    same parents, same node counts, same organs -- only positions move.

    This is the one way the physics pass could corrupt the model silently, and
    a rendered plant would look plausible either way."""
    if profile is None:
        return CheckResult(9, "Physics bends without restructuring", SKIP, "no profile supplied")

    from ..growth.development import _level_radii, grow, resolve
    from ..growth.diameter import compute_radii
    from ..growth.physics import apply_gravity

    plant = grow(profile, seed=seed, age=1.0)
    compute_radii(plant, branch_taper=profile.inflorescence.branch_width_profile,
                  level_radii=lambda p: _level_radii(p, profile, seed))

    def topology(p):
        return sorted(
            (a.id, a.order, a.parent_axis_id, a.parent_phytomer_index, len(a.phytomers),
             sum(len(ph.organs) for ph in a.phytomers), a.terminal_organ is not None)
            for a in p.axes.values()
        )

    before = topology(plant)
    positions_before = [tuple(ph.tip_position) for ph in plant.all_phytomers()]
    apply_gravity(plant, profile)
    resolve(plant, profile)
    after = topology(plant)
    positions_after = [tuple(ph.tip_position) for ph in plant.all_phytomers()]

    if before != after:
        return CheckResult(9, "Physics bends without restructuring", FAIL,
                           "topology changed across the gravity pass")
    moved = max(
        (float(np.linalg.norm(np.array(a) - np.array(b)))
         for a, b in zip(positions_before, positions_after, strict=True)),
        default=0.0,
    )
    if profile.droop > 0.0 and moved <= 1e-9:
        return CheckResult(9, "Physics bends without restructuring", FAIL,
                           f"droop is {profile.droop} but nothing moved")
    return CheckResult(9, "Physics bends without restructuring", PASS,
                       f"{len(before)} axes identical; largest node displacement "
                       f"{moved * 1000:.1f} mm")


# -- 10 ---------------------------------------------------------------------
def check_environment_is_neutral(profile=None, seed: int = 0) -> CheckResult:
    """Stage 3 must change nothing until a species asks it to.

    Environment is the one stage that reaches into every other one -- it
    modifies growth direction, bud release and where an axis stops -- so a
    mistake there does not announce itself as an environment bug. It shows up
    as every species quietly growing differently than it did, which is exactly
    the kind of regression the previous build shipped repeatedly. A profile
    with neutral responses, handed a real Environment object, has to reproduce
    the no-environment plant node for node.
    """
    if profile is None:
        return CheckResult(10, "Environment is neutral until asked", SKIP, "no profile supplied")
    if profile.reads_light or profile.obstacle_clearance > 0.0:
        return CheckResult(10, "Environment is neutral until asked", SKIP,
                           "this species responds to its surroundings by default")

    from ..growth.development import build
    from ..growth.environment import Environment

    plain = build(profile, seed=seed, age=1.0)
    with_env = build(profile, seed=seed, age=1.0, environment=Environment())
    if _fingerprint(plain) != _fingerprint(with_env):
        return CheckResult(10, "Environment is neutral until asked", FAIL,
                           "a neutral profile grew differently with an Environment attached")
    return CheckResult(10, "Environment is neutral until asked", PASS,
                       f"identical with and without ({plain.phytomer_count} phytomers)")


# -- 11 ---------------------------------------------------------------------
_STEM_KINK_DEG = 30.0
_APPENDAGE_KINK_DEG = 45.0


def _polyline_max_turn(points) -> float:
    """Sharpest joint (deg) between consecutive segments of a position polyline."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    segs = np.diff(pts, axis=0)
    norms = np.linalg.norm(segs, axis=1)
    segs = segs[norms > 1e-9] / norms[norms > 1e-9, None]
    if len(segs) < 2:
        return 0.0
    dots = np.clip(np.einsum("ij,ij->i", segs[:-1], segs[1:]), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())


def _directions_max_turn(directions) -> float:
    """Sharpest joint (deg) in a sequence of unit direction vectors."""
    d = np.asarray(directions, dtype=float)
    if len(d) < 2:
        return 0.0
    dots = np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())


def check_no_kinks(plant: Plant) -> CheckResult:
    """Stage 2's never-kink guarantee, as an outcome: no joint on any stem,
    pedicel or catkin turns sharply enough to read as a corner. This is the
    user's stated requirement -- "never kink, no 90-degree bends" -- made into
    a guardrail.

    Two populations, two bounds, because they are smoothed by different things.
    The stem, and every inflorescence branch (which is an axis), goes through
    `physics.smooth_bent_arcs`, which targets 7 deg per joint under a node
    budget -- so 30 deg passes every shipped species (worst is the nodding
    anemone at 9.4 deg, budget-capped) while still catching a smoothing
    regression or a ray too short to smooth. Pedicels and catkins are drawn by
    their own samplers (`model.pedicel_polyline`, `catkin_bunch`) at a
    resolution their own controls set, not by the beam -- worst measured is a
    hazel catkin stalk at 35 deg -- so they take the looser 45 deg bound, which
    is "no 90-degree bend" with a 2x margin. The point of both is the same:
    nothing kinks."""
    label = "No component kinks"
    stem_worst, stem_where = 0.0, None
    for axis in plant.axes.values():
        turn = _directions_max_turn([p.direction for p in axis.phytomers])
        if turn > stem_worst:
            stem_worst, stem_where = turn, axis.id

    app_worst = 0.0
    for org in plant.all_organs():
        if org.pedicel_points:
            app_worst = max(app_worst, _polyline_max_turn(org.pedicel_points))
        for body in getattr(org, "catkin_bodies", None) or []:
            app_worst = max(app_worst, _polyline_max_turn(body))

    if stem_worst > _STEM_KINK_DEG:
        return CheckResult(11, label, FAIL,
                           f"stem joint of {stem_worst:.1f} deg on axis {stem_where} exceeds "
                           f"{_STEM_KINK_DEG:.0f} deg -- smoothing is under-resolved")
    if app_worst > _APPENDAGE_KINK_DEG:
        return CheckResult(11, label, FAIL,
                           f"pedicel/catkin joint of {app_worst:.1f} deg exceeds "
                           f"{_APPENDAGE_KINK_DEG:.0f} deg")
    return CheckResult(11, label, PASS,
                       f"worst stem joint {stem_worst:.1f} deg, worst pedicel/catkin "
                       f"joint {app_worst:.1f} deg")


# -- 12 ---------------------------------------------------------------------
def check_gravity_lowers(profile=None, seed: int = 0) -> CheckResult:
    """Gravity may only sag: the bending pass must lower the plant, never raise
    it. One line against the whole family of sign bugs this system keeps hitting
    -- catkins that grew upward, a fall direction that pointed the wrong way --
    none of which announce themselves as a gravity bug; the plant just grows the
    wrong way.

    Measured as the mean node height before and after `apply_gravity` alone --
    not the growth pass, which legitimately aims stems upward. Physically exact:
    a stable bend does negative work on the centre of mass, so the mean can only
    fall, and it does across every species (0.5 mm on a stiff twig to 1.3 m on
    the nodding anemone). Deliberately not a per-segment check: an inherited
    parent bend can rigidly lift a child segment, which is correct rather than a
    curl-up, so only the aggregate is a sound invariant."""
    if profile is None:
        return CheckResult(12, "Gravity lowers, never lifts", SKIP, "no profile supplied")

    from ..growth.development import _level_radii, grow, resolve
    from ..growth.diameter import compute_radii
    from ..growth.physics import apply_gravity

    plant = grow(profile, seed=seed, age=1.0)
    compute_radii(plant, branch_taper=profile.inflorescence.branch_width_profile,
                  level_radii=lambda p: _level_radii(p, profile, seed))
    resolve(plant, profile)
    before = float(np.mean([ph.tip_position[2] for ph in plant.all_phytomers()]))
    apply_gravity(plant, profile, seed)
    resolve(plant, profile)
    after = float(np.mean([ph.tip_position[2] for ph in plant.all_phytomers()]))

    if after > before + 1e-6:
        return CheckResult(12, "Gravity lowers, never lifts", FAIL,
                           f"mean node height rose {(after - before) * 1000:.2f} mm across the "
                           "gravity pass -- gravity is pushing the wrong way")
    return CheckResult(12, "Gravity lowers, never lifts", PASS,
                       f"mean node height fell {(before - after) * 1000:.2f} mm")


# -- 13 ---------------------------------------------------------------------
_RESOLUTION_TOL_M = 0.08


def check_resolution_stable(profile=None, seed: int = 0) -> CheckResult:
    """Halving the arc-subdivision angle must barely move the plant. The bend is
    solved once, before subdivision, so a finer threshold only draws each turn
    as a finer arc and never changes the bend -- the original nodes should land
    in nearly the same place. Guards against subdivision breaking outright or
    starving its node budget, either of which makes the shape lurch when the
    resolution changes.

    Compared on the original (non-split) nodes, matched by their *order* along
    each axis: subdivision renumbers the phytomer index, so keying on the index
    compares different nodes and invents a large false delta (this check was
    written to a 600 mm phantom until that was found). Worst real move is the
    dense pendant laburnum at 34 mm, 8% of its height; every other species is
    under 9 mm."""
    if profile is None:
        return CheckResult(13, "Shape stable under subdivision", SKIP, "no profile supplied")

    from ..growth.development import build
    from ..growth.physics import _MAX_ARC_FACET_DEG

    def originals(plant):
        out = {}
        for axis in plant.axes.values():
            k = 0
            for ph in axis.phytomers:
                if not ph.is_split:
                    out[(axis.id, k)] = np.asarray(ph.tip_position, dtype=float)
                    k += 1
        return out

    coarse = originals(build(profile, seed=seed, arc_facet_deg=_MAX_ARC_FACET_DEG))
    fine = originals(build(profile, seed=seed, arc_facet_deg=_MAX_ARC_FACET_DEG / 2.0))
    keys = coarse.keys() & fine.keys()
    move = max((float(np.linalg.norm(coarse[k] - fine[k])) for k in keys), default=0.0)

    if move > _RESOLUTION_TOL_M:
        return CheckResult(13, "Shape stable under subdivision", FAIL,
                           f"halving the subdivision angle moved a node {move * 1000:.1f} mm "
                           f"(over {_RESOLUTION_TOL_M * 1000:.0f} mm) -- subdivision is unstable")
    return CheckResult(13, "Shape stable under subdivision", PASS,
                       f"largest node move on halving resolution {move * 1000:.1f} mm")


# -- 14 ---------------------------------------------------------------------
def check_dormancy_explained(plant: Plant, grow_fn=None) -> CheckResult:
    """Every dormant bud names the gate that is holding it.

    Not a past failure -- it guards the inspector's why-dormant readout against
    the way it would most plausibly break. `_release_buds` records a reason at
    each of its five `continue`s, and a *sixth* gate added later without a
    matching `held(...)` call would leave those buds reading "not held" while
    plainly not having released. Nothing would raise; the readout would just
    quietly start lying about the one thing it exists to say.

    Deliberately not a check that the reason is *correct* -- that is the gate's
    own logic, which every other check here already exercises. This asks only
    that no bud falls through unexplained.

    A bud created on the final step may not have been evaluated yet, so it is
    allowed to read NONE. Anything older than that has been past the gates at
    least once.

    Skipped without a `grow_fn`, on the same signal checks 4 and 8 use: that is
    the Phase A gate, whose buds are hand-built and have never been past
    `_release_buds` at all, so there is no gate that could have recorded
    anything. Asking them to name one fails a fixture that is working.
    """
    if grow_fn is None:
        return CheckResult(14, "Dormant buds name what holds them", SKIP,
                           "no growth loop on this fixture")
    stale = [
        bud for bud in plant.dormant_buds()
        if bud.blocked_by == BudBlock.NONE and bud.birth_step < plant.step
    ]
    dormant = plant.dormant_buds()
    if stale:
        bud = stale[0]
        return CheckResult(14, "Dormant buds name what holds them", FAIL,
                           f"{len(stale)} of {len(dormant)} dormant buds have no reason "
                           f"recorded, e.g. bud {bud.id} on axis {bud.axis_id} "
                           f"(born step {bud.birth_step}, plant at {plant.step})")
    if not dormant:
        return CheckResult(14, "Dormant buds name what holds them", SKIP,
                           "no dormant buds on this plant")
    counts: dict[str, int] = {}
    for bud in dormant:
        counts[bud.blocked_by] = counts.get(bud.blocked_by, 0) + 1
    summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    return CheckResult(14, "Dormant buds name what holds them", PASS,
                       f"{len(dormant)} dormant, all explained -- {summary}")


# -- 15 ---------------------------------------------------------------------
def check_chance_forking(profile=None, seed: int = 0) -> CheckResult:
    """A level's Fork: Probability 0 is an exact no-op; above 0 it is
    deterministic, really forks, and splits a stem's growth rather than adding
    height (children share the remaining node budget, at least two nodes each
    -- the plant may not grow more than 10 % taller)."""
    from dataclasses import replace

    from ..growth.development import build
    from ..growth.morphology import generic_perennial

    name = "Forking: no-op at 0, deterministic, not longer"
    base_profile = profile or generic_perennial()
    # Measured against a monopodial stems level, whatever pattern the profile
    # gives it: the check is of the fork mechanism, not of the species.
    stems = replace(base_profile.levels[0], growth_pattern="monopodial")
    base_profile = replace(base_profile, levels=[stems, *base_profile.levels[1:]])
    # Probability 0 never forks and leaves the stems as they were. Not the whole
    # plant: a monopodial stem's tip node is bare and a forking one's is not.
    # On the grown skeleton: `build`'s arc smoothing subdivides bent internodes,
    # and the tip branches' load changes the bend.
    from ..growth.development import grow
    off = grow(replace(base_profile, levels=[
        replace(stems, growth_pattern="dichotomous", fork_probability=0.0),
        *base_profile.levels[1:]]), seed=seed, age=1.0)
    if any(x.is_fork and x.level == 0 for x in off.axes.values()):
        return CheckResult(15, name, FAIL, "Fork on at Probability 0 forked")
    if _stems_fingerprint(off) != _stems_fingerprint(grow(base_profile, seed=seed, age=1.0)):
        return CheckResult(15, name, FAIL, "Fork on at Probability 0 changed the stems")
    forky = replace(base_profile, levels=[
        replace(stems, growth_pattern="dichotomous", fork_probability=0.2),
        *base_profile.levels[1:]])
    a, b = build(forky, seed=seed, age=1.0), build(forky, seed=seed, age=1.0)
    if _fingerprint(a) != _fingerprint(b):
        return CheckResult(15, name, FAIL, "two runs at the same seed differ")
    # Fork children only: a relaying level's relays are `is_fork` too, and a
    # chain's last unit may rightly be a single node.
    forks = [x for x in a.axes.values() if x.is_fork and 0 <= x.level < len(forky.levels)
             and forky.levels[x.level].forks]
    if not forks:
        return CheckResult(15, name, FAIL, "Probability 0.2 produced no forks")
    deepest = max(x.structural_depth for x in forks)
    short = [x.id for x in forks if x.node_budget < 2]
    if short:
        return CheckResult(15, name, FAIL, f"fork children with under 2 nodes: {short[:6]}")
    # Not longer: along every chain of forks from a stem's base, the vegetative
    # nodes add up to at most the stems level's Steps (each fork shares out
    # the nodes its parent had left).
    # Read on the grown skeleton: `build`'s arc smoothing subdivides bent
    # internodes, so its node counts are not Steps.
    grown = grow(forky, seed=seed, age=1.0)
    steps = stems.steps

    def vegetative(axis) -> int:
        return axis.flowering_from if axis.flowering_from >= 0 else len(axis.phytomers)

    longest = 0
    for x in grown.axes.values():
        if x.order != 0 or x.structural_depth >= 1 and not x.is_fork:
            continue
        total, cur = vegetative(x), x
        while cur.is_fork:
            parent = grown.axes[cur.parent_axis_id]
            total += cur.parent_phytomer_index + 1
            cur = parent
        longest = max(longest, total)
    if longest > steps:
        return CheckResult(15, name, FAIL, f"a fork chain built {longest} nodes > Steps {steps}")
    return CheckResult(15, name, PASS, f"{len(forks)} forks, depth <= {deepest}, "
                       f"longest chain {longest} <= Steps {steps}")


def _stems_fingerprint(plant: Plant) -> tuple:
    """The stems' ids and node counts -- what forking at Probability 0 must not
    change, whatever the branches on the stems' tip nodes do."""
    return tuple(
        (axis.id, len(axis.phytomers))
        for axis in sorted(plant.axes.values(), key=lambda a: a.id)
        if axis.parent_axis_id < 0
    )


def _fingerprint(plant: Plant) -> tuple:
    return tuple(
        (axis.id, axis.order, len(axis.phytomers),
         tuple(round(float(c), 9) for p in axis.phytomers for c in p.tip_position))
        for axis in sorted(plant.axes.values(), key=lambda a: a.id)
    )


# -- runner -----------------------------------------------------------------
def run_checks(plant: Plant, exp: Expectations, grow_fn=None, at_full_age: bool = True,
               profile=None) -> list[CheckResult]:
    return [
        check_branch_order(plant, exp),
        check_main_axis_dominance(plant, exp),
        check_all_axes_terminate(plant, at_full_age),
        check_age_monotonic(grow_fn),
        check_leaves_present(plant, exp),
        check_determinate_flowers(plant, exp, at_full_age),
        check_radius_monotonic(plant),
        check_determinism(grow_fn),
        check_physics_preserves_topology(profile),
        check_environment_is_neutral(profile),
        check_no_kinks(plant),
        check_gravity_lowers(profile),
        check_resolution_stable(profile),
        check_dormancy_explained(plant, grow_fn),
        check_chance_forking(profile),
    ]


def report(results: list[CheckResult], title: str = "structural checks") -> bool:
    marks = {PASS: "PASS", FAIL: "FAIL", SKIP: "skip"}
    print(f"\n{title}")
    print("-" * max(len(title), 64))
    for r in results:
        print(f"  [{marks[r.status]}] {r.number}. {r.name}\n         {r.detail}")
    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    passed = [r for r in results if r.status == PASS]
    print(f"\n  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    return not failed


# -- Phase A fixture --------------------------------------------------------
def hand_built_plant(phytomer_count: int = 10) -> Plant:
    """A single upright axis of `phytomer_count` phytomers, built by hand.

    This is the Phase A gate: it exercises the data model, the transported
    phyllotaxis frame, the elongation/expansion curves and the geometry
    resolver end to end, with no growth loop in existence yet. Alternate
    phyllotaxis at the golden angle, one leaf and one axillary bud per node,
    capped with a terminal flower so the determinate path is covered too.
    """
    plant = Plant(step=phytomer_count)
    axis = Axis(
        id=plant.new_axis_id(),
        order=0,
        birth_step=0,
        base_position=np.zeros(3),
        base_direction=np.array([0.0, 0.0, 1.0]),
    )
    axis.base_reference = orthonormal_reference(axis.base_direction)
    plant.add_axis(axis)

    # Short at the base, longest mid-axis, shorter to the tip (the shape the
    # retired internode-profile default had), so the fixture is unchanged.
    profile = curves.Curve.from_points([(0.0, 0.55), (0.35, 1.0), (0.75, 0.95), (1.0, 0.5)])
    direction = axis.base_direction.copy()
    reference = axis.base_reference.copy()
    divergence = 0.0

    for i in range(phytomer_count):
        # A gentle lean so the transported frame is genuinely exercised: with
        # a perfectly straight axis every transport is the identity and the
        # frame code would be trivially correct without being tested.
        if i > 0:
            new_direction = rotate_about(direction, np.array([0.0, 1.0, 0.0]), 3.0)
            new_direction /= np.linalg.norm(new_direction)
            reference = parallel_transport(reference, direction, new_direction)
            direction = new_direction

        along = i / max(phytomer_count - 1, 1)
        phytomer = Phytomer(
            index=i,
            birth_step=i,
            final_length=0.09 * profile(along),
            direction=direction.copy(),
            reference=reference.copy(),
        )
        divergence = (divergence + 137.5) % 360.0
        phytomer.organs.append(Organ(
            id=plant.new_organ_id(), kind=OrganKind.LEAF, birth_step=i,
            final_scale=1.0, divergence_deg=divergence, elevation_deg=42.0,
        ))
        phytomer.buds.append(Bud(
            id=plant.new_bud_id(), position_kind=BudPosition.AXILLARY, birth_step=i,
            axis_id=axis.id, phytomer_index=i, divergence_deg=divergence, elevation_deg=42.0,
        ))
        axis.phytomers.append(phytomer)

    # A determinate cap: the apex converts to a terminal flower and stops.
    axis.apex = Bud(
        id=plant.new_bud_id(), position_kind=BudPosition.TERMINAL, birth_step=0,
        axis_id=axis.id, phytomer_index=-1,
        state=BudState.RELEASED, fate="flower",
    )
    axis.terminal_organ = Organ(
        id=plant.new_organ_id(), kind=OrganKind.FLOWER,
        birth_step=phytomer_count, final_scale=1.0, elevation_deg=90.0,
    )

    resolve_geometry(
        plant,
        elongation=curves.default_elongation().over(4),
        expansion=curves.default_leaf_expansion().over(6),
    )
    return plant


def _phase_a() -> bool:
    plant = hand_built_plant()
    exp = Expectations(max_branch_order=2, expects_leaves=True, expects_flowers=True)

    lo, hi = plant.bounds()
    print("Phase A fixture: one hand-built axis, 10 phytomers")
    print(f"  height {hi[2] - lo[2]:.3f} m | {plant.phytomer_count} phytomers | "
          f"{len(plant.organs_of_kind(OrganKind.LEAF))} leaves | "
          f"{len(plant.organs_of_kind(OrganKind.FLOWER))} flowers")
    print("  leaf divergence (deg): " + ", ".join(
        f"{o.divergence_deg:.1f}" for o in plant.organs_of_kind(OrganKind.LEAF)[:6]) + " ...")
    print("  internode lengths (m): " + ", ".join(
        f"{p.length:.4f}" for p in plant.all_phytomers()))
    return report(run_checks(plant, exp), "Phase A gate - hand-built structure")


def _phase_b() -> bool:
    """The real growth loop, which makes assertions 4 and 8 run for the first
    time. The developmental sequence is printed alongside so the Age dial can
    be seen doing what the architecture doc asks of it."""
    from ..growth.development import build
    from ..growth.morphology import generic_perennial

    profile = generic_perennial()
    exp = profile.expectations()

    def grow_fn(age: float):
        """The full pipeline, exactly as bridge.regrow() runs it: grow,
        thicken, bend, resolve."""
        return build(profile, seed=0, age=age)

    print("\n\nPhase B: age-driven growth, generic perennial")
    print(f"  {'age':>5} {'phytomers':>10} {'axes':>6} {'height':>8} {'width':>8} "
          f"{'leaves':>7} {'flowers':>8}")
    for age in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p = grow_fn(age)
        if p.phytomer_count:
            lo, hi = p.bounds()
            height, width = hi[2] - lo[2], max(hi[0] - lo[0], hi[1] - lo[1])
        else:
            height = width = 0.0
        print(f"  {age:>5.1f} {p.phytomer_count:>10} {len(p.axes):>6} {height:>7.3f}m "
              f"{width:>7.3f}m {len(p.organs_of_kind(OrganKind.LEAF)):>7} "
              f"{len(p.organs_of_kind(OrganKind.FLOWER)):>8}")

    mature = grow_fn(1.0)
    return report(run_checks(mature, exp, grow_fn=grow_fn, profile=profile),
                  "Phase B/E gate - grown structure")


def main() -> int:
    ok_a = _phase_a()
    ok_b = _phase_b()
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
