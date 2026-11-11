"""Stem diameter -- Stage 4, after growth, before the gravity bend (which
needs radii: curvature goes as 1 / radius ** 4).

Since the branching overhaul the widths are
*stated*, not derived: every stem and branch takes its branching level's Start
Width and Width / Step (`development._level_radii`, passed in as
`level_radii`), and every inflorescence branch a fraction of the rachis it
leaves, tapered by the inflorescence's own curve (below). The pipe model
(da Vinci / Shinozaki: a cross-section equals the sum of those it supports) that
derived them from the load was retired with it -- once every axis had a stated
width it no longer decided any radius, measured bit for bit.
"""

from __future__ import annotations

from .model import Plant


def compute_radii(plant: Plant, branch_taper=None, level_radii=None,
                  branch_taper_of=None) -> None:
    """Write `radius` on every phytomer: the level widths first (topological
    order, so a branch reads its parent's final radius), then the
    inflorescence branches off them."""
    # Before the inflorescence-branch pass below, which reads the rachis it
    # leaves.
    if level_radii is not None:
        level_radii(plant)

    # An inflorescence branch takes its thickness from the rachis it leaves,
    # tapering to its own tip -- `pedicel_width` and `pedicel_width_profile`'s
    # convention exactly, so "1.0" means "as thick as the stem it comes off"
    # rather than naming no particular thickness.
    #
    # Still before `physics.apply_gravity`, so bending sees it: curvature goes
    # as `1 / radius ** 4`, and a branch drawn finer therefore also arches more.
    chains = _cyme_chains(plant)
    for axis in plant.axes_in_topological_order():
        if axis.structural_depth < 1 or axis.is_fork or not axis.phytomers:
            # Only real inflorescence branches take the inflorescence-branch
            # width treatment; a vegetative fork is sized by its level above.
            continue
        parent = plant.axes.get(axis.parent_axis_id)
        if parent is None or not (0 <= axis.parent_phytomer_index < len(parent.phytomers)):
            continue
        # Read *after* the parent has its own final radius, which the pass above
        # guarantees -- and after any width the parent itself took from ITS
        # parent, so a fraction compounds down the orders the way a real
        # branching system narrows.
        # **One reference per inflorescence, not the radius at each branch's own
        # attachment node.** Read at the parent's `flowering_from` -- where its
        # branching region begins -- so every branch off one rachis starts the
        # same thickness whatever height it sits at, and the taper curve is the
        # only thing that varies width along a branch.
        #
        # Per-attachment was the first version, and it makes branch thickness
        # track the rachis's own pipe-model taper: measured 7.778 mm at the base
        # of the region down to 1.527 mm at the top, so branch bases ran 3.889 ->
        # 0.764 mm, a 5x spread nobody asked for. Reported as the graph
        # controlling "the width of the branch along the height of the trunk" --
        # the graph was innocent, this was.
        #
        # It compounds down the orders naturally: a secondary branch references
        # its own primary's base, which already references the rachis's.
        anchor = parent.flowering_from if parent.flowering_from >= 0 else 0
        anchor = min(max(anchor, 0), len(parent.phytomers) - 1)
        reference = parent.phytomers[anchor].radius
        # Never thicker than what actually carries it, which is the local node
        # rather than the reference -- the topmost branches come off a rachis
        # that has thinned below the reference, and a side branch fatter than
        # the stem it leaves reads as broken however uniform it makes the set.
        width = min(1.0, max(0.0, axis.inflorescence_width))
        local = parent.phytomers[axis.parent_phytomer_index].radius
        base = min(reference * width, local)
        last = max(len(axis.phytomers) - 1, 1)
        # **Normalised at its own start**, so the curve shapes the taper and
        # never the base width -- that is `branch_width`'s to state. Otherwise
        # dragging the curve's left end down thins the whole branch, which is
        # the scalar's job, and the two controls fight over one number.
        # Its own inflorescence's curve when the caller says per axis
        # (several inflorescences on one plant), else the one given.
        taper_curve = branch_taper if branch_taper_of is None else branch_taper_of(axis)
        chain = chains.get(axis.id)
        if chain is not None:
            # **A cyme unit tapers as part of its whole chain.** Each unit is
            # a short sympodial piece of one stem, and the curve applied per
            # unit compounded down it: Width 1.0, the taper ending at 0.55, and
            # a Scorpioid Cyme's sixth unit was 0.07 mm carrying 27 flowers
            # (2.51 mm at its first) -- it hung to the ground. Read along the
            # chain instead, from the first unit's base to its farthest tip, a
            # unit going on from its parent's last node exactly as the next
            # node of one axis would. Width likewise sets the chain's start
            # against the stem it leaves, once: per unit it compounded the
            # same way (0.83 a unit took a sixth unit to 40%).
            k0, span = chain
            taper = [1.0 if taper_curve is None else float(taper_curve(k / span))
                     for k in range(k0 - 1, k0 + len(axis.phytomers))]
            if axis.parent_axis_id in chains:
                start, ref = parent.phytomers[-1].radius, taper[0]
            else:
                start, ref = base, taper[1]
            if abs(ref) <= 1e-6:            # the curve has run out: hold the width
                ref, taper = 1.0, [1.0] * len(taper)
            for i, phytomer in enumerate(axis.phytomers):
                phytomer.radius = max(1e-6, start * taper[i + 1] / ref)
            continue
        head = 1.0 if taper_curve is None else float(taper_curve(0.0))
        scale = 1.0 / head if abs(head) > 1e-6 else 1.0
        for i, phytomer in enumerate(axis.phytomers):
            taper = 1.0 if taper_curve is None else float(taper_curve(i / last)) * scale
            phytomer.radius = max(1e-6, base * taper)


def _cyme_chains(plant: Plant) -> dict:
    """Every cyme unit's place in its chain, `{axis id: (k0, span)}`: `k0` the
    chain's node index at the unit's first node, `span` the node count from the
    chain's first node to its farthest tip, minus one (so the taper runs 0..1
    on the longest path; a shorter branch of a dichasium stops partway)."""
    relays = {a.id: a for a in plant.axes.values() if a.is_cyme_relay and a.phytomers}
    if not relays:
        return {}
    children: dict = {}
    for a in relays.values():
        children.setdefault(a.parent_axis_id, []).append(a.id)
    reach: dict = {}
    # deepest first, so each unit's farthest tip is known before its parent's
    for a in reversed(list(plant.axes_in_topological_order())):
        if a.id in relays:
            below = max((reach[c] for c in children.get(a.id, ())), default=0)
            reach[a.id] = len(a.phytomers) + below
    out: dict = {}
    for a in plant.axes_in_topological_order():
        if a.id not in relays:
            continue
        if a.parent_axis_id in relays:
            k0, span = out[a.parent_axis_id]
            out[a.id] = (k0 + len(relays[a.parent_axis_id].phytomers), span)
        else:
            out[a.id] = (0, max(reach[a.id] - 1, 1))
    return out
