"""WS3 -- the post-growth lifecycle pass (senescence).

Runs after `diameter` (the pipe model) and `physics`, so the sim grows the full
plant untouched and this pass only marks what dies -- ALIVE / DEAD (retained) /
SHED. That ordering is the whole reason removal is legal here: nothing is
stopped from *growing*; structure is edited after the fact, and environment-
driven death (later phases) only *reads* the already-built light field rather
than driving growth.

Every trigger is off by default, so with a default profile this pass leaves the
whole plant ALIVE and the build is byte-identical to one without it -- which is
what the goldens guard.
"""

from __future__ import annotations

import numpy as np

from .model import LifecycleState, OrganKind, Plant


def apply_lifecycle(plant: Plant, profile, seed: int = 0, age: float = 1.0,
                    environment=None, day: float | None = None) -> None:
    """Mark organs and axes ALIVE/DEAD/SHED from the profile's senescence
    triggers.

    Phase 1 wires the age death-front; Phase 2 the environment shade threshold.
    The manual (prune set) trigger slots in here in a later phase; each resolves
    a state per item and hands a dying axis to `cascade`.

    `day` is the calendar (age→real-days) entry: when given and the species
    carries a senescence window, the death-front reads the plant's position in
    that window instead of the developmental `senescence_onset(age)` curve. Off
    the day path (`day is None`) or with no calendar it is byte-identical.
    """
    _age_death_front(plant, profile, age, day)
    _shade_death_front(plant, profile, environment)


def _sets_fruit(organ, fruit_set: float) -> bool:
    """Whether one flower sets fruit, rolled per organ so a plant is stable
    across regrows -- the same per-id seeding every other per-organ roll uses.
    `fruit_set` 1.0 (every flower fruits) skips the roll."""
    if fruit_set >= 1.0:
        return True
    return float(np.random.default_rng(int(organ.id)).random()) < fruit_set


def _to_fruit(organ, fruit_size: float) -> None:
    """Turn a set flower into a fruit in place. It keeps the flower's position,
    facing and pedicel -- a fruit develops where its flower was -- but takes its
    own size: `final_scale` so the next resolve keeps it (a fruit does not
    expand -- see `model._organ_expansion`), and `scale` directly so
    `apply_gravity`, which reads scale before that resolve, already sees it."""
    organ.kind = OrganKind.FRUIT
    organ.final_scale = fruit_size
    organ.scale = fruit_size


def apply_fruit(plant: Plant, profile, day: float | None = None) -> None:
    """Flower -> fruit -> drop on the calendar's fruit window (phase 4 backend,
    milestone-2 render).

    Botanically a fruit develops from a pollinated flower in place, so a flower
    whose species carries a fruit window flips to `OrganKind.FRUIT` once
    fruit-set passes -- taking the flower's position, facing and pedicel, but its
    own `fruit_size`. `fruit_set` is the fraction that pollinate (rolled per
    organ); the rest drop unpollinated. Everything is removed once fruit-drop
    passes.

    **Day-path only.** Off the day path (goldens, every non-calendar caller) or
    with no fruit window, nothing here runs and every plant is byte-identical --
    no organ is ever a fruit, so the fruit branches in `physics`/`resolve` are
    never taken either.

    **Runs before the bend** (`development.build` calls it just before
    `compute_radii`), so `fruit_weight` enters `physics._organ_mass` and a
    fruiting branch bows, springing back once the window closes and the fruit is
    gone. That is a change from the milestone-1 placement (it ran last, purely as
    data); the cost is it now runs before `apply_lifecycle`, so a flower on a
    limb the death-front later sheds becomes a fruit first and is shed with the
    limb -- the same bare limb either way.
    """
    cal = getattr(profile, "calendar", None)
    if day is None or cal is None or cal.fruit is None:
        return
    win = cal.fruit
    if not win.opened(day, profile.emergence_day):
        return                                   # nothing has set yet
    dropped = win.closed(day, profile.emergence_day)   # past fruit-drop -> gone
    fruit_set, fruit_size = profile.fruit_set, profile.fruit_size
    for axis in plant.axes.values():
        if axis.lifecycle != LifecycleState.ALIVE:
            continue
        term = axis.terminal_organ
        if term is not None and term.kind == OrganKind.FLOWER:
            if dropped or not _sets_fruit(term, fruit_set):
                axis.terminal_organ = None
            else:
                _to_fruit(term, fruit_size)
        for phy in axis.phytomers:
            kept = []
            for o in phy.organs:
                if o.kind != OrganKind.FLOWER:
                    kept.append(o)
                elif not dropped and _sets_fruit(o, fruit_set):
                    _to_fruit(o, fruit_size)
                    kept.append(o)
                # else: unpollinated, or past fruit-drop -> removed
            phy.organs = kept


def kill_all(plant: Plant, profile) -> None:
    """Mark the whole plant dead (calendar phase 5 -- an annual past its one
    season). Everything, the **trunk included**: an annual dies whole, unlike
    the WS3 death-front which deliberately spares the main axis. Retained
    standing (DEAD, browning) or shed to the ground per the species'
    `senescence_retain_dead_wood`, the same control the death-front honours."""
    state = (LifecycleState.DEAD if profile.senescence_retain_dead_wood
             else LifecycleState.SHED)
    for axis in plant.axes.values():
        axis.lifecycle = state
        for phy in axis.phytomers:
            for o in phy.organs:
                o.lifecycle = state
        if axis.terminal_organ is not None:
            axis.terminal_organ.lifecycle = state


def _age_death_front(plant: Plant, profile, age: float, day: float | None = None) -> None:
    """Kill the oldest lateral branches first as the Age death-front advances.

    `senescence_onset(age)` gives the fraction of primary limbs dead by now,
    **lowest first** -- crown-lifting, the way a maturing plant abandons its
    lower branches. The unit is the **primary limb -- an order-1 branch** -- and
    its whole subtree dies with it (via `cascade`), which is what a limb
    senescing means; selecting deeper laterals directly would let one low
    order-1 branch's cascade take the entire crown. The trunk (order 0) is
    exempt -- a plant does not senesce its own main axis. A dead branch is
    retained wood (or shed, per `senescence_retain_dead_wood`) with its leaves
    and flowers shed: bare.

    **By height, not `birth_step`.** The grill assumed the oldest limb is the
    lowest, but measured on this growth model it is not -- axillary release is
    not bottom-up, so `birth_step` scatters across height (the oldest limb sat
    high, the youngest lowest) and "oldest-first" read as crown *dieback*.
    Height is what "lower branches die" actually means, and it is deterministic.
    """
    fraction = min(1.0, max(0.0, profile.senescence_fraction(age, day)))
    if fraction <= 0.0:
        return
    limbs = sorted(
        (a for a in plant.axes.values()
         if a.order == 1 and a.phytomers and a.lifecycle == LifecycleState.ALIVE),
        key=lambda a: (float(a.base_position[2]), a.id))
    n_dead = int(round(fraction * len(limbs)))
    if n_dead <= 0:
        return

    branch_state = (LifecycleState.DEAD if profile.senescence_retain_dead_wood
                    else LifecycleState.SHED)
    children = children_map(plant)
    for axis in limbs[:n_dead]:
        cascade(plant, axis, branch_state, LifecycleState.SHED, children)


def _shade_death_front(plant: Plant, profile, environment) -> None:
    """Kill lateral branches whose base sits in too little light.

    The deliberately-reopened shade-death decision, legal only because it runs
    *after* growth: the environment never stopped a bud from growing (that would
    be the forbidden inversion), it only reads the finished plant's own light
    field and marks what is too shaded to keep. `senescence_shade_threshold`
    default 0 means never, so this is inert -- and byte-identical -- until a
    species opts in.

    **The light field is rebuilt against the finished plant here**, not reused
    from the last growth step, for two reasons: growth only builds a light field
    at all when the plant *responds* to light (`light_seeking > 0`), and even
    then it reflects the plant one step short of done. A post-growth judgment
    wants the post-growth canopy. `update_light` reads the plant's own foliage
    plus any obstacles, so a plant shades itself even with no obstacle in the
    scene.

    Judged per lateral (order >= 1), parents first, so a branch already taken by
    a shaded parent's cascade is skipped rather than re-walked. The trunk (order
    0) is exempt: a plant does not shade-kill its own main axis. A branch's whole
    subtree dies with it, the same cascade the age front uses.
    """
    threshold = float(getattr(profile, "senescence_shade_threshold", 0.0))
    if threshold <= 0.0 or environment is None:
        return
    # The finished canopy, not the last growth step's -- see the docstring.
    environment.update_light(plant)

    laterals = sorted(
        (a for a in plant.axes.values() if a.order >= 1 and a.phytomers),
        key=lambda a: (a.order, a.id))
    branch_state = (LifecycleState.DEAD if profile.senescence_retain_dead_wood
                    else LifecycleState.SHED)
    children = children_map(plant)
    for axis in laterals:
        if axis.lifecycle != LifecycleState.ALIVE:
            continue  # already taken by a parent's cascade
        if environment.light_at(axis.base_position) < threshold:
            cascade(plant, axis, branch_state, LifecycleState.SHED, children)


def children_map(plant: Plant) -> dict:
    """{parent_id: [child axes]}, so a bulk trigger builds it once instead of
    scanning `plant.axes` per death."""
    m: dict = {}
    for a in plant.axes.values():
        if a.parent_axis_id >= 0:
            m.setdefault(a.parent_axis_id, []).append(a)
    return m


def cascade(plant: Plant, axis, branch_state: str, organ_state: str,
            children: dict | None = None) -> None:
    """Mark an axis and its whole subtree: axes -> `branch_state`, organs ->
    `organ_state`. Death of a branch takes the limb, not one internode.

    The two states are what makes a dead branch *bare*: the wood is retained
    (branch_state DEAD) while its leaves and flowers are shed (organ_state SHED).
    A fully shed limb passes SHED for both.
    """
    if children is None:
        children = children_map(plant)
    stack = [axis]
    while stack:
        a = stack.pop()
        a.lifecycle = branch_state
        for phytomer in a.phytomers:
            for organ in phytomer.organs:
                organ.lifecycle = organ_state
        if a.terminal_organ is not None:
            a.terminal_organ.lifecycle = organ_state
        stack.extend(children.get(a.id, ()))
