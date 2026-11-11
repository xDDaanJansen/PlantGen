"""A custom viewport overlay for inspecting a plant's developmental state.

Blender's own debug surfaces are geometry-shaped: an attribute can be coloured,
a vertex can be selected, and anything that is not geometry cannot be shown at
all. That is the wrong shape for this problem. **The most important things
about a plant here produce no geometry** -- a dormant bud, a terminated apex,
the threshold a bud never crossed -- and the first attempt at WS1 proved the
rest of it too: one selectable point per component, in edit mode, put every one
of the Anemone's 779 components on just **161 distinct positions**, in stacks
of up to nine, because the phytomer model attaches a node's leaves and the buds
in their axils at that node's own point.

So this draws its own. `SpaceView3D.draw_handler_add` gives a GPU pass over the
finished viewport and `blf` gives text at any projected position, which between
them allow a schematic: one coloured glyph per component, shaped by what it is,
and its values readable in the viewport rather than only in the sidebar.

**Screen space, not a point cloud, and the cost was measured before committing
to it.** Projecting the whole cloud in numpy and building a fan per glyph costs
**0.32 ms per redraw at 779 components and 1.09 ms at 3116** -- 2% and 6% of a
60 fps frame. That buys per-kind shapes, constant pixel size at any zoom, and
the screen-space fanning below, none of which a `POINTS` batch can do.

Positions stay **exactly** where the model puts them. Where several components
land on one point the glyphs are fanned apart *on screen*, which is a display
nudge rather than a lie about where anything is -- the earlier build moved the
points themselves in 3D and that is the thing being corrected.

This does not replace the Geometry Nodes viewer taps in `mesh.py`. Those tint
the *real* geometry, which is a different job and the right one for asking how
a quantity varies along a stem. This one exists for everything that has no
geometry to tint.
"""

from __future__ import annotations

from dataclasses import dataclass

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ..growth.model import (
    BudBlock,
    BudFate,
    BudPosition,
    BudState,
    LeafForm,
    LifecycleState,
    OrganKind,
)

# -- What a component is ---------------------------------------------------

NODE = 0
BUD = 1
APEX = 2
LEAF = 3
FLOWER = 4
CATKIN = 5
BRACT = 6
FRUIT = 7  # calendar phase 4: a flower that has set fruit (no geometry, data only)

KIND_LABELS = {
    NODE: "Node", BUD: "Bud", APEX: "Apex", LEAF: "Leaf",
    FLOWER: "Flower", CATKIN: "Catkin", BRACT: "Bract", FRUIT: "Fruit",
}

KIND_ITEMS = (
    ('NODE', "Nodes", "Phytomer nodes -- one per internode, at its distal end"),
    ('BUD', "Buds", "Axillary and other lateral meristems, including the ones that "
     "never released. These have no geometry at all, so no other view can show them"),
    ('APEX', "Apices", "Terminal buds -- one per axis. A terminated apex is a bud "
     "whose axis stopped growing"),
    ('LEAF', "Leaves", "Foliage leaves and bracts"),
    ('FLOWER', "Flowers", "Flowers, catkins, and any flower that has set fruit"),
)

_ITEM_KINDS = {
    'NODE': (NODE,), 'BUD': (BUD,), 'APEX': (APEX,),
    'LEAF': (LEAF, BRACT), 'FLOWER': (FLOWER, CATKIN, FRUIT),
}

# Enum codes derived from the model's own members rather than typed out a
# second time, so a code cannot drift from what it stands for.
BUD_STATES = (BudState.ACTIVE, BudState.DORMANT, BudState.RELEASED, BudState.ABORTED)
BUD_FATES = (BudFate.NONE, BudFate.SHOOT, BudFate.FLOWER)
BUD_POSITIONS = (
    BudPosition.TERMINAL, BudPosition.AXILLARY, BudPosition.INTERNODAL,
    BudPosition.LEAF_OPPOSED, BudPosition.CAULINE, BudPosition.BASAL,
    BudPosition.ADVENTITIOUS,
)
LEAF_FORMS = ("Juvenile", "Adult", "Bract")

LIFECYCLE_STATES = (LifecycleState.ALIVE, LifecycleState.DEAD, LifecycleState.SHED)
"""WS3. A component's lifecycle -- and the one thing that has no geometry when it
matters most: a SHED item is drawn by nothing, so the inspector is the only
place to see what a prune or a death front removed."""

BUD_BLOCKS = (BudBlock.NONE, BudBlock.DENSITY, BudBlock.ZONE,
              BudBlock.DOMINANCE, BudBlock.ONSET, BudBlock.ORDER)

BLOCK_LABELS = {
    BudBlock.NONE: "-",
    BudBlock.DENSITY: "Branch Density",
    BudBlock.ZONE: "Branching Zone",
    BudBlock.DOMINANCE: "Apical Dominance",
    BudBlock.ONSET: "Branching Onset",
    BudBlock.ORDER: "Branching Levels",
}
"""Named after the **panel controls** rather than after the code, deliberately.
The question a held bud raises is "what do I change to release it", and a
readout saying `dominance` answers it one translation step short. Since the
branching overhaul growth only raises ORDER (no level below this one); the
other four are kept so the codes stay positional."""

_BLOCK_DETAIL = {
    BudBlock.DENSITY: "threshold {v:.3f} vs density {l:.3f}",
    BudBlock.ZONE: "at {v:.2f} up the axis, zone from {l:.2f}",
    BudBlock.DOMINANCE: "{v:.0f} {nodes} below the tip, needs {l:.1f}",
    BudBlock.ONSET: "onset {v:.3f}, needs {l:.3f}",
    BudBlock.ORDER: "would be level {v:.0f}, deepest level is {l:.0f}",
}
"""Both sides of the gate's own comparison, phrased per gate -- the two stored
numbers mean something different for each, and a bare pair would be unreadable.
This is what turns "held by Apical Dominance" into "2 nodes below the tip,
needs 4", which says how far off the bud actually is."""

_FIELDS = (
    "axis_id", "order", "node_index", "structural_depth", "birth_step",
    "age_steps", "factor", "radius", "curvature", "bud_state", "bud_fate",
    "position_kind", "release_threshold", "child_id", "organ_form",
    "organ_scale", "bloom_start", "divergence_deg", "elevation_deg",
    "blocked_by", "blocked_value", "blocked_limit", "lifecycle",
)

_UNSET = -1.0
"""What a field reads where it does not apply. -1 rather than 0, so "this means
nothing here" stays distinguishable from a real zero -- order 0 and axis 0 are
both perfectly ordinary values."""


@dataclass
class Snapshot:
    """One grown plant, reduced to what the overlay draws and reports.

    Column arrays rather than a list of objects, because every frame projects
    the whole set at once and a list of dicts cannot be handed to numpy.
    """

    positions: np.ndarray            # (n, 3) in the plant's own local space
    kind: np.ndarray                 # (n,) int
    stack_slot: np.ndarray           # (n,) int, position within a coincident group
    stack_size: np.ndarray           # (n,) int
    fields: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.kind)


_snapshots: dict[str, Snapshot] = {}
"""Keyed by object name, the same convention `live.py` uses for its dirty set.

The previous build stored all of this as mesh attributes, which persist in the
.blend; a Python cache does not survive a file reload. That is an accepted
trade rather than an oversight: `development.build` is deterministic given
seed, age and environment, so a lost snapshot is always exactly recoverable by
regrowing, and dropping the mesh takes an object out of the outliner, edit mode
out of the workflow, and the whole coincident-point problem off the table."""


def capture(obj, plant) -> None:
    """Reduce a freshly grown plant to a snapshot and cache it.

    Called from `bridge.regrow` while the `Plant` is still in hand -- it is a
    local there and goes out of scope immediately after, so anything the
    overlay wants to say has to be taken now.

    **Pins are kept, and only ever clamped back into range.** An earlier
    version dropped them whenever the structure changed, on the reasoning that
    a component index is a position rather than an identity. That became wrong
    the moment pins moved onto the object to make them undoable: undo restores
    the pins *belonging to the state it restored*, which are correct for the
    structure it restored -- so clearing them here threw away exactly what
    Ctrl+Z had just brought back. Measured as pins vanishing on every undo.

    What remains is the one real hazard: an index past the end of a smaller
    plant. The draw pass guards it too, but a stored pin that can never be
    drawn is dead weight.
    """
    fresh = _build_snapshot(plant)
    _snapshots[obj.name] = fresh
    live = [i for i in pinned(obj) if 0 <= i < len(fresh)]
    if len(live) != len(obj.plantgen.inspect_pinned):
        set_pinned(obj, live)
    if _state["object"] == obj.name:
        _state.update(hover=-1, screen=None, visible=None, region=0)


def snapshot_for(obj) -> Snapshot | None:
    return _snapshots.get(obj.name) if obj else None


def ensure_snapshot(obj) -> Snapshot:
    """The snapshot for a plant, growing it once if there is not one cached --
    after a file reload there never is."""
    existing = _snapshots.get(obj.name)
    if existing is not None:
        return existing
    from ..growth.development import build
    from . import bridge
    from . import environment as env_bridge

    s = obj.plantgen
    profile = bridge.profile_from_object(obj)
    plant = build(profile, seed=s.seed, age=s.age,
                  environment=env_bridge.build_environment(obj, profile),
                  signals=bridge.growth_signals_for(obj, profile))
    # Apply manual pruning (and stamp the topology the Prune operator reads)
    # here too, so the inspector on a freshly loaded file shows the same
    # lifecycle states the rendered mesh does.
    bridge.apply_pruning(obj, plant)
    capture(obj, plant)
    return _snapshots[obj.name]


def _build_snapshot(plant) -> Snapshot:
    """Walk the plant once, emitting one record per component.

    Topological order, so a component's index climbs the plant base to tip --
    it is the id the card shows, and one that climbs is far easier to hold in
    your head than one that jumps about.

    Split arc points are skipped. `physics.smooth_bent_arcs` inserts them so a
    hard bend reads as a curve rather than a corner; they carry a position and
    a radius but no botanical identity, bear no organ and hold no bud, so they
    would fill the overlay with glyphs standing for nothing -- and on a heavily
    bent plant they outnumber the real nodes.
    """
    positions: list[tuple] = []
    kinds: list[int] = []
    slot: list[int] = []
    anchor: list[int] = []
    cols: dict[str, list[float]] = {name: [] for name in _FIELDS}

    def emit(position, kind: int, group: int, **values) -> None:
        positions.append(tuple(float(c) for c in position))
        kinds.append(kind)
        anchor.append(group)
        slot.append(0)
        for name in _FIELDS:
            cols[name].append(float(values.get(name, _UNSET)))

    def code(table, value) -> float:
        return float(table.index(value)) if value in table else _UNSET

    def emit_bud(bud, axis, kind: int, group: int) -> None:
        emit(bud.position, kind, group,
             axis_id=axis.id, order=axis.order, node_index=bud.phytomer_index,
             structural_depth=axis.structural_depth,
             lifecycle=code(LIFECYCLE_STATES, axis.lifecycle),
             birth_step=bud.birth_step, age_steps=bud.age_steps(plant.step),
             bud_state=code(BUD_STATES, bud.state),
             bud_fate=code(BUD_FATES, bud.fate),
             position_kind=code(BUD_POSITIONS, bud.position_kind),
             release_threshold=bud.release_threshold,
             blocked_by=code(BUD_BLOCKS, bud.blocked_by),
             blocked_value=bud.blocked_value,
             blocked_limit=bud.blocked_limit,
             child_id=bud.child_axis_id,
             divergence_deg=bud.divergence_deg, elevation_deg=bud.elevation_deg)

    def emit_organ(organ, axis, group: int) -> None:
        if organ.kind == OrganKind.FLOWER:
            kind = FLOWER
        elif organ.kind == OrganKind.FRUIT:
            kind = FRUIT
        elif organ.kind == OrganKind.CATKIN:
            kind = CATKIN
        elif organ.form == LeafForm.BRACT:
            kind = BRACT
        else:
            kind = LEAF
        emit(organ.position, kind, group,
             axis_id=axis.id, order=axis.order,
             structural_depth=axis.structural_depth,
             lifecycle=code(LIFECYCLE_STATES, organ.lifecycle),
             birth_step=organ.birth_step, age_steps=organ.age_steps(plant.step),
             organ_form=int(organ.form), organ_scale=organ.scale,
             bloom_start=organ.bloom_start, child_id=organ.id,
             divergence_deg=organ.divergence_deg,
             elevation_deg=organ.elevation_deg)

    group = 0
    for axis in plant.axes_in_topological_order():
        for phytomer in axis.phytomers:
            if phytomer.is_split:
                continue
            # One group per node: everything attached here shares its position
            # exactly, and the draw pass fans a group apart on screen.
            emit(phytomer.tip_position, NODE, group,
                 axis_id=axis.id, order=axis.order, node_index=phytomer.index,
                 structural_depth=axis.structural_depth,
                 lifecycle=code(LIFECYCLE_STATES, axis.lifecycle),
                 birth_step=phytomer.birth_step,
                 age_steps=phytomer.age_steps(plant.step),
                 factor=phytomer.factor, radius=phytomer.radius,
                 curvature=phytomer.debug_curvature)
            for bud in phytomer.buds:
                emit_bud(bud, axis, BUD, group)
            for organ in phytomer.organs:
                emit_organ(organ, axis, group)
            group += 1
        # The apex last, so it reads as the top of its own axis. It is an
        # ordinary `Bud` -- a terminated apex is simply one whose axis stopped
        # -- which is why emitting buds naturally covers apices too.
        if axis.apex is not None:
            emit_bud(axis.apex, axis, APEX, group)
        if axis.terminal_organ is not None:
            emit_organ(axis.terminal_organ, axis, group)
        group += 1

    n = len(positions)
    anchor_arr = np.asarray(anchor, dtype=np.int32)
    slot_arr = np.zeros(n, dtype=np.int32)
    size_arr = np.ones(n, dtype=np.int32)
    if n:
        # Slot within each coincident group, assigned once here rather than per
        # frame: a slot recomputed from screen distance every redraw would make
        # glyphs swap places as the camera moves, which is worse than overlap.
        order = np.argsort(anchor_arr, kind="stable")
        grouped = anchor_arr[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(grouped)) + 1))
        counts = np.diff(np.concatenate((starts, [n])))
        slot_arr[order] = np.arange(n) - np.repeat(starts, counts)
        size_arr[order] = np.repeat(counts, counts)

    return Snapshot(
        positions=np.asarray(positions, dtype="f").reshape(n, 3),
        kind=np.asarray(kinds, dtype=np.int32),
        stack_slot=slot_arr,
        stack_size=size_arr,
        fields={name: np.asarray(values, dtype="f") for name, values in cols.items()},
    )


# -- Reading one component -------------------------------------------------

def _label(table, code: float) -> str:
    i = int(code)
    if 0 <= i < len(table):
        return str(table[i]).replace("_", " ").title()
    return "-"


def describe(snap: Snapshot, i: int) -> tuple[str, list[tuple[str, str]]]:
    """A component decoded into a title and label/value rows.

    One function for both the viewport card and the sidebar, so the two can
    never disagree about what a component says.

    Laid out per kind rather than as a flat dump of every column: most columns
    do not apply to any given component, and a stack of "-" is what an
    inspector is supposed to spare you.
    """
    f = snap.fields
    kind = int(snap.kind[i])
    title = f"{KIND_LABELS.get(kind, '?')}  #{i}"
    rows: list[tuple[str, str]] = []

    def num(name: str, fmt: str = "{:.0f}") -> str:
        v = float(f[name][i])
        return "-" if v == _UNSET else fmt.format(v)

    if kind == NODE:
        rows += [
            ("Axis", num("axis_id")),
            ("Branch order", num("order")),
            ("Node on axis", num("node_index")),
            ("Along axis", num("factor", "{:.3f}")),
            ("Radius", f"{float(f['radius'][i]) * 1000.0:.2f} mm"),
            ("Curvature", f"{float(f['curvature'][i]):.1f} deg/m"),
        ]
    elif kind in (BUD, APEX):
        node = float(f["node_index"][i])
        rows += [
            ("State", _label(BUD_STATES, f["bud_state"][i])),
            ("Fate", _label(BUD_FATES, f["bud_fate"][i])),
            ("Meristem", _label(BUD_POSITIONS, f["position_kind"][i])),
            ("Axis", num("axis_id")),
            ("Branch order", num("order")),
            ("Node on axis", "terminal" if node < 0 else f"{node:.0f}"),
            # Every release gate in `_release_buds` is a comparison against
            # this one number, so it is the most useful thing to see on a bud
            # that has not gone anywhere.
            ("Release threshold", num("release_threshold", "{:.3f}")),
            ("Divergence", f"{float(f['divergence_deg'][i]):.1f} deg"),
            ("Elevation", f"{float(f['elevation_deg'][i]):.1f} deg"),
        ]
        # Why this bud is still dormant, which is the question a bud raises
        # and the one `release_threshold` alone cannot answer: every gate in
        # `_release_buds` is a comparison against that same number, so it says
        # how reluctant the bud is and this says what it ran into.
        code = int(f["blocked_by"][i])
        block = BUD_BLOCKS[code] if 0 <= code < len(BUD_BLOCKS) else BudBlock.NONE
        if block != BudBlock.NONE:
            rows.append(("Held by", BLOCK_LABELS[block]))
            detail = _BLOCK_DETAIL.get(block)
            if detail:
                # `format` ignores keywords a template does not use, so the
                # noun costs nothing on the four gates that do not count things.
                value = float(f["blocked_value"][i])
                rows.append(("", detail.format(
                    v=value, l=float(f["blocked_limit"][i]),
                    nodes="node" if abs(value) == 1.0 else "nodes")))
        if float(f["child_id"][i]) >= 0:
            rows.append(("Grew axis", num("child_id")))
    else:
        rows.append(("Organ", num("child_id")))
        form = int(f["organ_form"][i])
        if 0 <= form < len(LEAF_FORMS):
            rows.append(("Form", LEAF_FORMS[form]))
        rows.append(("Size", num("organ_scale", "{:.3f}")))
        if kind == FLOWER:
            rows.append(("Bloom order", num("bloom_start", "{:.3f}")))
        rows += [
            ("On axis", num("axis_id")),
            ("Branch order", num("order")),
            ("Divergence", f"{float(f['divergence_deg'][i]):.1f} deg"),
            ("Elevation", f"{float(f['elevation_deg'][i]):.1f} deg"),
        ]

    rows.append(("Born at step", num("birth_step")))
    rows.append(("Age", f"{float(f['age_steps'][i]):.0f} steps"))
    if float(f["structural_depth"][i]) >= 0:
        rows.append(("Structural depth", num("structural_depth")))
    # Only when it is not alive: a dead or shed component is the interesting
    # case, and an "Alive" row on every card is the noise an inspector spares.
    life = int(f["lifecycle"][i])
    if 0 < life < len(LIFECYCLE_STATES):
        rows.append(("Lifecycle", _label(LIFECYCLE_STATES, life)))
    return title, rows


# -- Colour ----------------------------------------------------------------

_OKABE_ITO = (
    (0.902, 0.624, 0.000), (0.337, 0.706, 0.914), (0.000, 0.620, 0.451),
    (0.941, 0.894, 0.259), (0.000, 0.447, 0.698), (0.835, 0.369, 0.000),
    (0.800, 0.475, 0.655),
)
"""The same colour-blind-safe set the Geometry Nodes categorical views use, so
a branch order reads as the same hue whichever surface is showing it."""

_KIND_COLOR = {
    NODE: (0.72, 0.74, 0.78), BUD: (0.902, 0.624, 0.0), APEX: (0.835, 0.369, 0.0),
    LEAF: (0.0, 0.620, 0.451), FLOWER: (0.800, 0.475, 0.655),
    CATKIN: (0.337, 0.706, 0.914), BRACT: (0.35, 0.55, 0.35),
    FRUIT: (0.792, 0.243, 0.208),  # a ripe red, distinct from the pink flower
}

_STATE_COLOR = {
    0: (0.20, 0.80, 0.35),   # active -- growing
    1: (0.95, 0.70, 0.10),   # dormant -- the one everything else cannot show
    2: (0.30, 0.60, 0.95),   # released -- committed, did something
    3: (0.80, 0.47, 0.65),   # aborted -- will never do anything
}

_FATE_COLOR = {
    0: (0.95, 0.70, 0.10),   # none -- undecided, and a dormant bud's fate
    1: (0.20, 0.75, 0.45),   # shoot
    2: (0.85, 0.45, 0.70),   # flower
}
"""Aborted and "none" were grey until the legend put them next to `_GREY`, and
the two were near enough to read as the same swatch -- which conflates *this
bud will never do anything*, a real and interesting state, with *this is not a
bud*, which means nothing at all. Grey belongs to the second, so the first
needs a colour of its own.

Reddish purple rather than a red, deliberately: active is green, and red
against green is the common deficiency. This is Okabe-Ito's own reddish purple,
which separates cleanly from the green, the orange and the blue under protan
and deutan vision alike -- the same standard the Geometry Nodes debug palettes
are held to.

Fate NONE takes dormant's orange rather than a fourth hue, because they are the
same fact seen twice: a bud that has not broken dormancy has not decided a
fate. The two modes then agree at a glance instead of recolouring the same
buds."""

_BLOCK_COLOR = {
    1: (0.902, 0.624, 0.000),   # density
    2: (0.337, 0.706, 0.914),   # zone
    3: (0.000, 0.620, 0.451),   # dominance
    4: (0.941, 0.894, 0.259),   # onset
    5: (0.800, 0.475, 0.655),   # order cap
}
"""Okabe-Ito again, and code 0 (not held) is deliberately absent so a released
bud falls through to grey along with everything that is not a bud."""

# Every item carries an explicit number, so display order is free and a stored
# value can never drift. An `EnumProperty` stores the item's *position*, so
# inserting one in the middle silently rewrites what every saved object means
# -- the lesson `inflorescence_kind` already paid for.
COLOR_ITEMS = (
    ('KIND', "Component", "A colour per component type -- node, bud, apex, leaf, flower",
     'NONE', 0),
    ('STATE', "Bud State", "Active, dormant, released or aborted. Components that are "
     "not buds stay grey. This is the view no Geometry Nodes overlay can produce, "
     "because a dormant bud has no geometry to tint", 'NONE', 1),
    ('FATE', "Bud Fate", "What each bud became -- nothing, a shoot, or a flower",
     'NONE', 2),
    ('BLOCK', "Held By", "Which control is keeping each dormant bud shut - branch "
     "density, branching zone, apical dominance, branching onset, or the branch "
     "order cap. The legend counts them, so this answers 'why is my plant not "
     "branching' directly", 'NONE', 5),
    ('ORDER', "Branch Order", "Which branch order each component sits on", 'NONE', 3),
    ('AXIS', "Axis", "A colour per stem or branch", 'NONE', 4),
    ('LIFECYCLE', "Lifecycle", "Alive, dead (retained wood) or shed (removed). WS3 "
     "senescence and pruning -- and the only view that shows a shed item, which by "
     "definition draws no geometry", 'NONE', 6),
)

_GREY = (0.42, 0.42, 0.45)

_LIFECYCLE_COLOR = {
    0: (0.20, 0.75, 0.40),   # alive -- green
    1: (0.62, 0.47, 0.28),   # dead  -- dead-wood brown
    2: (0.80, 0.30, 0.35),   # shed  -- a warm red, "removed"
}
"""Green/brown/red rather than the Okabe-Ito set the categorical views use,
because these are not arbitrary categories: alive/dead/shed is an ordered
severity, and green-to-red reads as one. Shed is the only red anywhere in the
overlay, so it does not clash with a category hue, and dead-wood brown matches
the render's own Dead Colour so the overlay and the mesh agree at a glance."""


def _colors(snap: Snapshot, mode: str) -> np.ndarray:
    n = len(snap)
    out = np.zeros((n, 3), dtype="f")
    if mode == 'KIND':
        for kind, rgb in _KIND_COLOR.items():
            out[snap.kind == kind] = rgb
        return out
    if mode in ('STATE', 'FATE', 'BLOCK', 'LIFECYCLE'):
        table = {'STATE': _STATE_COLOR, 'FATE': _FATE_COLOR, 'BLOCK': _BLOCK_COLOR,
                 'LIFECYCLE': _LIFECYCLE_COLOR}[mode]
        column = snap.fields[{'STATE': "bud_state", 'FATE': "bud_fate",
                              'BLOCK': "blocked_by", 'LIFECYCLE': "lifecycle"}[mode]]
        out[:] = _GREY
        for code, rgb in table.items():
            out[column.astype(np.int32) == code] = rgb
        return out
    column = snap.fields["order" if mode == 'ORDER' else "axis_id"].astype(np.int32)
    for i, rgb in enumerate(_OKABE_ITO):
        out[np.mod(np.maximum(column, 0), len(_OKABE_ITO)) == i] = rgb
    return out


# -- Glyph shapes ----------------------------------------------------------

def _ngon(sides: int, rotation: float = 0.0, scale: float = 1.0) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False) + rotation
    return np.stack([np.cos(a), np.sin(a)], axis=1).astype("f") * scale


_SHAPE = {
    NODE: _ngon(4, np.pi / 4, 0.78),       # square, on-axis
    BUD: _ngon(14, 0.0, 1.0),              # circle
    APEX: _ngon(3, np.pi / 2, 1.30),       # triangle, pointing up the axis
    LEAF: _ngon(4, np.pi / 2, 1.15),       # diamond
    FLOWER: _ngon(6, np.pi / 6, 1.05),     # hexagon
    CATKIN: _ngon(12, 0.0, 1.0) * np.array([0.62, 1.45], dtype="f"),
    BRACT: _ngon(3, -np.pi / 2, 1.05),     # triangle, inverted
    FRUIT: _ngon(5, np.pi / 2, 1.05),      # pentagon -- a berry
}
"""A shape per component kind, so the overlay is readable in one glance even
in a colour mode that deliberately greys most of it -- and legible to anyone
who cannot separate two of the hues."""

_GLYPH_PX = 5.2
_STACK_PX = 7.5
"""How far apart a coincident group is fanned, in pixels. Everything attached
to a node shares that node's exact position, so without this a bud sits inside
its own node and neither can be pointed at."""


def _fans(centres: np.ndarray, shape: np.ndarray, radius: float) -> np.ndarray:
    """Triangle fans for one glyph shape at many centres, built without a
    Python loop -- this runs every redraw."""
    if not len(centres):
        return np.zeros((0, 2), dtype="f")
    ring = centres[:, None, :] + shape[None, :, :] * radius
    hub = np.repeat(centres[:, None, :], shape.shape[0], axis=1)
    nxt = np.roll(ring, -1, axis=1)
    tris = np.concatenate(
        [hub[:, :, None, :], ring[:, :, None, :], nxt[:, :, None, :]], axis=2)
    return tris.reshape(-1, 2)


# -- Live state ------------------------------------------------------------

_state = {
    "object": "",      # the plant the overlay is showing; "" means off
    "hover": -1,
    "screen": None,    # (n, 2) last drawn screen positions, for hit-testing
    "visible": None,   # (n,) bool, which of them were actually drawn
    "region": 0,       # which region those positions belong to
    "reselect": False, # whether the plant was selected before inspecting
    "session": 0,      # bumped on every start and stop -- see `session()`
}


def active_object():
    name = _state["object"]
    return bpy.data.objects.get(name) if name else None


def is_active(obj) -> bool:
    return bool(obj) and _state["object"] == obj.name


def _set_flag(obj, value: bool) -> None:
    """Mirror the overlay's on/off onto the object, where undo can reach it.

    Guarded, because `stop()` runs from `unregister()` too and the property
    group may already be gone by then.
    """
    try:
        obj.plantgen.inspect_active = value
    except (AttributeError, ReferenceError, TypeError):
        pass


def reconcile() -> None:
    """Bring the running overlay in line with what undo restored.

    `inspect_active` lives on the object so Ctrl+Z can reverse turning the
    inspector on or off. But undo only puts the *flag* back -- the modal
    operator that tracks the cursor is not part of any datablock, so it has to
    be started or stopped to match. Called from the deferred rebuild, which
    already runs after every undo.
    """
    wanted = None
    for candidate in bpy.data.objects:
        if candidate.get("plantgen") is not None and candidate.plantgen.inspect_active:
            wanted = candidate
            break
    if (wanted.name if wanted else "") == _state["object"]:
        return
    if _state["object"]:
        stop()
    if wanted is not None:
        _invoke_inspect(wanted)


def _invoke_inspect(obj) -> None:
    """Restart the modal for a plant undo has just switched back on.

    A modal needs a 3D viewport to run in, and a timer has no useful context
    of its own, so one is found and overridden.
    """
    window = next((w for w in bpy.context.window_manager.windows), None)
    if window is None:
        _set_flag(obj, False)
        return
    area = next((a for a in window.screen.areas if a.type == 'VIEW_3D'), None)
    region = next((r for r in area.regions if r.type == 'WINDOW'), None) if area else None
    if region is None:
        _set_flag(obj, False)
        return
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.plantgen.toggle_inspect('INVOKE_DEFAULT')


def start(obj) -> int:
    """Begin inspecting. The plant is **deselected but left active**, which is
    not tidiness: Blender's bright selection outline traces every silhouette
    edge of the plant, and against a few hundred glyphs it reads as more
    overlay rather than as selection. The panels only ever needed the object to
    be active. Restored on the way out."""
    was_selected = bool(obj.select_get())
    try:
        obj.select_set(False)
    except RuntimeError:
        was_selected = False
    # Exactly one plant is inspected at a time, so any stale flag left on
    # another by an undo is cleared rather than leaving two claiming it.
    for other in bpy.data.objects:
        if other.get("plantgen") is not None and other is not obj:
            _set_flag(other, False)
    _set_flag(obj, True)
    _state.update(object=obj.name, hover=-1, screen=None, visible=None,
                  region=0, reselect=was_selected, session=_state["session"] + 1)
    return _state["session"]


def stop() -> None:
    obj = active_object()
    if obj is not None:
        _set_flag(obj, False)
    if obj is not None and _state["reselect"]:
        try:
            obj.select_set(True)
        except RuntimeError:
            pass
    _state.update(object="", hover=-1, screen=None, visible=None,
                  region=0, reselect=False, session=_state["session"] + 1)


def session() -> int:
    """A token bumped on every start and stop.

    A modal operator cannot end itself from outside, and turning the overlay
    off runs a *fresh* operator instance -- the one already in the event
    stream keeps going until its next event. Worse, starting the overlay on a
    second plant leaves the first modal running with nothing that looks wrong
    to it, so both then handle every mouse move and both toggle the same pin.
    Each modal captures this token at invoke and exits the moment it no longer
    matches, which makes "at most one modal is live" true by construction
    rather than by hoping the events line up.
    """
    return _state["session"]


def set_hover(index: int) -> bool:
    if _state["hover"] == index:
        return False
    _state["hover"] = index
    return True


def pinned(obj) -> list[int]:
    """Which cards are pinned on this plant.

    Read off the object, not out of `_state`: pins live in a `CollectionProperty`
    precisely so **Blender's undo can reach them**, since undo restores
    datablocks and cannot restore a Python dict.
    """
    if obj is None:
        return []
    return [p.index for p in obj.plantgen.inspect_pinned]


def set_pinned(obj, indices) -> None:
    coll = obj.plantgen.inspect_pinned
    coll.clear()
    for i in indices:
        coll.add().index = int(i)


def unpin(obj, index: int) -> bool:
    """Drop one pinned card by component index.

    Separate from `toggle_pin`, which acts on whatever is under the cursor:
    the sidebar has to name the card it is closing, and a pinned component is
    very often not the one being hovered.
    """
    current = pinned(obj)
    if index not in current:
        return False
    set_pinned(obj, [i for i in current if i != index])
    return True


def toggle_pin(obj) -> str:
    """Pin or unpin whatever is currently hovered."""
    i = _state["hover"]
    if i < 0:
        return ""
    current = pinned(obj)
    if i in current:
        set_pinned(obj, [x for x in current if x != i])
        return "unpinned"
    set_pinned(obj, [*current, i])
    return "pinned"


def hovered() -> int:
    return _state["hover"]


def prune_hovered(obj) -> int:
    """Prune the *axis* the hovered component sits on. Returns the pruned axis
    id, or -1 if nothing is hovered or that axis is already pruned.

    Axis, not component: you prune a limb, not one internode, so pointing at any
    node/bud/leaf on a branch prunes the whole branch (its subtree goes too,
    applied in `bridge.apply_pruning`). The set is stamped with the current
    topology so it can be dropped when the plant is regrown into a different
    shape and the id stops naming this limb.
    """
    i = _state["hover"]
    snap = snapshot_for(obj)
    if i < 0 or snap is None or i >= len(snap):
        return -1
    axis_id = int(snap.fields["axis_id"][i])
    if axis_id < 0:
        return -1
    s = obj.plantgen
    if any(p.axis == axis_id for p in s.inspect_pruned):
        return -1
    from . import bridge
    s.inspect_pruned.add().axis = axis_id
    s.prune_signature = obj.get(bridge._TOPO_KEY, "")
    s.prune_stale = False
    return axis_id


def clear_pruned(obj) -> int:
    """Drop every manual prune. Returns how many limbs were un-pruned."""
    s = obj.plantgen
    n = len(s.inspect_pruned)
    s.inspect_pruned.clear()
    s.prune_signature = ""
    s.prune_stale = False
    return n


def pick(x: float, y: float, region_key: int = 0, radius: float = 14.0) -> int:
    """The component nearest a screen position, or -1.

    Hit-tested against the positions the last draw actually used, fanning
    included, so what you point at is what you saw rather than where the raw
    3D position happens to project. `region_key` guards the several-viewports
    case: those positions belong to whichever region drew last, and testing a
    second viewport's cursor against them would report confident nonsense."""
    screen, visible = _state["screen"], _state["visible"]
    if screen is None or visible is None or not visible.any():
        return -1
    if region_key and _state["region"] and region_key != _state["region"]:
        return -1
    d = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
    d = np.where(visible, d, np.inf)
    i = int(np.argmin(d))
    return i if d[i] <= radius else -1


# -- Drawing ---------------------------------------------------------------

def _project(snap: Snapshot, obj, region, r3d):
    """The whole cloud to screen space in one vectorised pass."""
    world = np.asarray(obj.matrix_world.transposed(), dtype="f")
    clip_m = np.asarray(r3d.perspective_matrix.transposed(), dtype="f")
    h = np.hstack([snap.positions, np.ones((len(snap), 1), dtype="f")])
    clip = h @ world @ clip_m
    w = clip[:, 3]
    in_front = w > 1e-6
    safe = np.where(in_front, w, 1.0)
    x = (clip[:, 0] / safe * 0.5 + 0.5) * region.width
    y = (clip[:, 1] / safe * 0.5 + 0.5) * region.height
    return np.stack([x, y], axis=1), in_front


def _fan_offsets(snap: Snapshot) -> np.ndarray:
    """A fixed screen-space nudge per component, spreading a coincident group
    around a small ring. Stable, because the slot was decided at capture time
    rather than from whatever the camera happens to show."""
    size = snap.stack_size
    lone = size <= 1
    angle = 2.0 * np.pi * snap.stack_slot / np.maximum(size, 1)
    reach = np.where(lone, 0.0, _STACK_PX * np.minimum(np.sqrt(size) * 0.8, 2.2))
    return np.stack([np.cos(angle) * reach, np.sin(angle) * reach], axis=1).astype("f")


def _kind_mask(snap: Snapshot, shown: set) -> np.ndarray:
    keep = np.zeros(len(snap), dtype=bool)
    for item in shown:
        for kind in _ITEM_KINDS.get(item, ()):
            keep |= snap.kind == kind
    return keep


def _usable_rect(region) -> tuple[float, float, float, float]:
    """The part of the window region nothing else is drawn on top of.

    The toolbar and the sidebar are **separate regions drawn over this one**,
    so `region.width` counts pixels that are not actually visible. Clamping a
    card against the raw width slides it neatly under the sidebar, where it is
    cut off mid-word -- which is exactly how the Held By detail row was first
    seen, truncated to "would be or".
    """
    left, right = 0.0, float(region.width)
    area = bpy.context.area
    if area is not None:
        for r in area.regions:
            # A collapsed region reports a width of 1, so this is inert when
            # either panel is closed.
            if r.type == 'TOOLS':
                left = max(left, float(r.width))
            elif r.type == 'UI':
                right = min(right, float(region.width) - float(r.width))
    return left, 0.0, max(right, left + 1.0), float(region.height)


def _card_box(font: int, title: str, rows: list[tuple[str, str]],
              scale: float) -> tuple[float, float]:
    """A card's size, measured without drawing it -- placement has to know how
    big every card is before it can stop them landing on each other."""
    pad = 7.0 * scale
    line = 15.0 * scale
    blf.size(font, int(12 * scale))
    label_w = max((blf.dimensions(font, r[0])[0] for r in rows), default=0.0)
    value_w = max((blf.dimensions(font, r[1])[0] for r in rows), default=0.0)
    blf.size(font, int(13 * scale))
    width = max(blf.dimensions(font, title)[0], label_w + value_w + 18.0 * scale) + pad * 2
    return width, line * (len(rows) + 1) + pad * 2


def _place_card(x: float, y: float, width: float, height: float,
                taken: list[tuple[float, float, float, float]],
                bounds: tuple[float, float, float, float]):
    """Somewhere this card fits without covering one already placed.

    Cards were previously offset by a few pixels each, which only staggers the
    corners -- with two components close together the second card sat almost
    entirely on the first and neither could be read. Slides down, then up, then
    gives up and takes the spot: an overlapping card is still better than a
    card pushed off screen.
    """
    x0, y0, x1, y1 = bounds
    x = min(max(x, x0 + 4.0), x1 - width - 4.0)

    def clear(cy: float) -> bool:
        return not any(x < ox + ow and x + width > ox and cy < oy + oh and cy + height > oy
                       for ox, oy, ow, oh in taken)

    step = 6.0
    start = min(max(y, y0 + 4.0), y1 - height - 4.0)
    for candidate in (start, *(start - n * step for n in range(1, 200)),
                      *(start + n * step for n in range(1, 200))):
        if candidate < y0 + 4.0 or candidate > y1 - height - 4.0:
            continue
        if clear(candidate):
            return x, candidate
    return x, start


def _draw_card(font: int, x: float, y: float, title: str,
               rows: list[tuple[str, str]], accent, scale: float) -> None:
    """A value card in the viewport, which is the whole point of not putting
    this in the sidebar -- the numbers sit beside the thing they describe.

    Placed by `_place_card`; this only draws it.
    """
    pad = 7.0 * scale
    line = 15.0 * scale
    # The value column starts past the widest label, so every row's values line
    # up -- measured here rather than passed in, since `_card_box` measures the
    # same text for the same reason and the two must agree.
    blf.size(font, int(12 * scale))
    label_w = max((blf.dimensions(font, r[0])[0] for r in rows), default=0.0)
    width, height = _card_box(font, title, rows, scale)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    quad = np.array([
        (x, y), (x + width, y), (x + width, y + height),
        (x, y), (x + width, y + height), (x, y + height)], dtype="f")
    shader.bind()
    shader.uniform_float("color", (0.06, 0.06, 0.07, 0.88))
    batch_for_shader(shader, 'TRIS', {"pos": quad}).draw(shader)
    # A coloured spine on the left edge, carrying the same colour the glyph has
    # -- which card belongs to which glyph is otherwise guesswork once two are
    # up at once.
    spine = np.array([
        (x, y), (x + 3.0 * scale, y), (x + 3.0 * scale, y + height),
        (x, y), (x + 3.0 * scale, y + height), (x, y + height)], dtype="f")
    shader.uniform_float("color", (*accent, 1.0))
    batch_for_shader(shader, 'TRIS', {"pos": spine}).draw(shader)

    top = y + height - pad - line * 0.82
    blf.size(font, int(13 * scale))
    blf.color(font, *accent, 1.0)
    blf.position(font, x + pad + 5.0 * scale, top, 0)
    blf.draw(font, title)

    blf.size(font, int(12 * scale))
    for n, (label, value) in enumerate(rows):
        row_y = top - line * (n + 1)
        blf.color(font, 0.62, 0.63, 0.66, 1.0)
        blf.position(font, x + pad + 5.0 * scale, row_y, 0)
        blf.draw(font, label)
        blf.color(font, 0.93, 0.93, 0.95, 1.0)
        blf.position(font, x + pad + 5.0 * scale + label_w + 14.0 * scale, row_y, 0)
        blf.draw(font, value)


def _legend_rows(snap: Snapshot, visible: np.ndarray, mode: str):
    """What the legend has to say, as (shape rows, colour rows, note).

    Counted off `visible` rather than the whole snapshot, so the numbers
    describe what is actually on screen -- which turns the legend into a small
    histogram of the thing being diagnosed for no extra cost, and is most of
    why it earns its corner.

    **In Component mode shape and colour are the same axis**, so the two lists
    collapse into one. In every other mode they say different things and the
    shape swatches are drawn grey, because tinting them would imply a
    correspondence that is not there.
    """
    shapes: list[tuple[int, str, int, tuple]] = []
    for kind in _SHAPE:
        count = int((visible & (snap.kind == kind)).sum())
        if count:
            colour = _KIND_COLOR[kind] if mode == 'KIND' else (0.62, 0.63, 0.66)
            shapes.append((kind, KIND_LABELS[kind], count, colour))

    if mode == 'KIND':
        return shapes, [], None

    if mode == 'AXIS':
        # One hue per axis, cycled -- a plant with forty axes would be a legend
        # longer than the viewport, and the colours repeat anyway, so the
        # honest thing is to say what the mode means rather than enumerate it.
        axes = np.unique(snap.fields["axis_id"][visible].astype(np.int32))
        return shapes, [], f"a colour per axis  ({len(axes)} shown)"

    if mode == 'ORDER':
        column = snap.fields["order"][visible].astype(np.int32)
        rows = [(_OKABE_ITO[o % len(_OKABE_ITO)], f"Order {o}", int((column == o).sum()))
                for o in np.unique(column) if o >= 0]
        return shapes, rows, None

    table = {'STATE': _STATE_COLOR, 'FATE': _FATE_COLOR, 'BLOCK': _BLOCK_COLOR,
             'LIFECYCLE': _LIFECYCLE_COLOR}[mode]
    field = {'STATE': "bud_state", 'FATE': "bud_fate", 'BLOCK': "blocked_by",
             'LIFECYCLE': "lifecycle"}[mode]
    names = {'STATE': BUD_STATES, 'FATE': BUD_FATES, 'BLOCK': BUD_BLOCKS,
             'LIFECYCLE': LIFECYCLE_STATES}[mode]
    column = snap.fields[field][visible]
    rows = []
    for code, rgb in table.items():
        count = int((column.astype(np.int32) == code).sum())
        if count:
            label = (BLOCK_LABELS[names[code]] if mode == 'BLOCK'
                     else str(names[code]).title())
            rows.append((rgb, label, count))
    # Everything grey, named so it is not mistaken for a category of its own.
    # In Held By that is both non-buds and buds nothing is holding, which are
    # genuinely the same statement: no gate applies.
    other = int((column == _UNSET).sum())
    if mode == 'BLOCK':
        other += int((column.astype(np.int32) == 0).sum())
    if other:
        rows.append((_GREY, "Not held" if mode == 'BLOCK' else "Not a bud", other))
    return shapes, rows, None


def _draw_legend(font: int, region, snap: Snapshot, visible: np.ndarray,
                 mode: str, scale: float) -> None:
    """A key to the shapes and colours, bottom-left.

    In the viewport rather than the sidebar, for the same reason the value
    cards are: this reads as a diagram, and a diagram carries its own key.

    It has no toggle. One more control to hide a small panel is exactly the
    slider-farm trade this project refuses, and it is drawn only while
    inspecting -- which is the only time it means anything.
    """
    shapes, colours, note = _legend_rows(snap, visible, mode)
    if not shapes and not colours:
        return
    mode_name = next((m[1] for m in COLOR_ITEMS if m[0] == mode), mode)

    line = 16.0 * scale
    pad = 8.0 * scale
    swatch = 9.0 * scale
    gap = swatch * 2.0 + 8.0 * scale
    blf.size(font, int(11 * scale))

    rows_text = ([f"{r[1]}" for r in shapes]
                 + ([mode_name] if (colours or note) else [])
                 + [r[1] for r in colours] + ([note] if note else []))
    counts_text = ([f"{r[2]}" for r in shapes]
                   + ([""] if (colours or note) else [])
                   + [f"{r[2]}" for r in colours] + ([""] if note else []))
    label_w = max((blf.dimensions(font, t)[0] for t in rows_text), default=0.0)
    count_w = max((blf.dimensions(font, t)[0] for t in counts_text), default=0.0)
    n_rows = len(rows_text)

    width = gap + label_w + 12.0 * scale + count_w + pad * 2
    height = line * n_rows + pad * 2

    # Clear of the toolbar and the sidebar, both of which are separate regions
    # drawn *over* this one -- see `_usable_rect`.
    x0, y0, _x1, _y1 = _usable_rect(region)
    x = x0 + 12.0 * scale
    y = y0 + 12.0 * scale

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", (0.06, 0.06, 0.07, 0.85))
    quad = np.array([
        (x, y), (x + width, y), (x + width, y + height),
        (x, y), (x + width, y + height), (x, y + height)], dtype="f")
    batch_for_shader(shader, 'TRIS', {"pos": quad}).draw(shader)

    glyph_shader = gpu.shader.from_builtin('FLAT_COLOR')
    top = y + height - pad - line * 0.78
    row = 0

    def text(label: str, count: str, dim: bool) -> None:
        row_y = top - line * row
        blf.color(font, 0.55, 0.56, 0.60, 1.0) if dim else \
            blf.color(font, 0.90, 0.91, 0.93, 1.0)
        blf.position(font, x + pad + gap, row_y, 0)
        blf.draw(font, label)
        if count:
            blf.color(font, 0.60, 0.61, 0.65, 1.0)
            blf.position(font, x + width - pad - blf.dimensions(font, count)[0], row_y, 0)
            blf.draw(font, count)

    for kind, label, count, colour in shapes:
        centre = np.array([[x + pad + swatch, top - line * row + 3.5 * scale]], dtype="f")
        tris = _fans(centre, _SHAPE[kind], swatch * 0.62)
        rgba = np.tile(np.array([*colour, 1.0], dtype="f"), (len(tris), 1))
        glyph_shader.bind()
        batch_for_shader(glyph_shader, 'TRIS',
                         {"pos": tris, "color": rgba}).draw(glyph_shader)
        text(label, str(count), False)
        row += 1

    if colours or note:
        text(mode_name, "", True)
        row += 1

    for rgb, label, count in colours:
        cx = x + pad + swatch
        cy = top - line * row + 3.5 * scale
        box = np.array([
            (cx - swatch * 0.5, cy - swatch * 0.5), (cx + swatch * 0.5, cy - swatch * 0.5),
            (cx + swatch * 0.5, cy + swatch * 0.5),
            (cx - swatch * 0.5, cy - swatch * 0.5), (cx + swatch * 0.5, cy + swatch * 0.5),
            (cx - swatch * 0.5, cy + swatch * 0.5)], dtype="f")
        rgba = np.tile(np.array([*rgb, 1.0], dtype="f"), (6, 1))
        glyph_shader.bind()
        batch_for_shader(glyph_shader, 'TRIS',
                         {"pos": box, "color": rgba}).draw(glyph_shader)
        text(label, str(count), False)
        row += 1

    if note:
        text(note, "", True)


def _draw() -> None:
    obj = active_object()
    if obj is None:
        return
    snap = snapshot_for(obj)
    if snap is None or not len(snap):
        return
    region = bpy.context.region
    r3d = bpy.context.region_data
    if region is None or r3d is None:
        return
    space = bpy.context.space_data
    if space is None or space.type != 'VIEW_3D':
        return

    s = obj.plantgen
    shown = set(s.inspect_show)
    keep = _kind_mask(snap, shown)
    screen, in_front = _project(snap, obj, region, r3d)
    screen = screen + _fan_offsets(snap)
    on_screen = (
        in_front
        & (screen[:, 0] > -20) & (screen[:, 0] < region.width + 20)
        & (screen[:, 1] > -20) & (screen[:, 1] < region.height + 20)
    )
    visible = keep & on_screen
    # Remembered for hit-testing: pointing at a glyph should test where it was
    # drawn, not where its raw position projects.
    _state["screen"] = screen
    _state["visible"] = visible
    _state["region"] = region.as_pointer()

    colours = _colors(snap, s.inspect_color_by)
    scale = bpy.context.preferences.system.ui_scale
    radius = _GLYPH_PX * scale

    verts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    # Dark outlines first, then the fills over them: with depth test off the
    # later triangles win, so one batch in this order gives every glyph a
    # halo and stays a single draw call.
    for outline in (True, False):
        for kind, shape in _SHAPE.items():
            mask = visible & (snap.kind == kind)
            count = int(mask.sum())
            if not count:
                continue
            tris = _fans(screen[mask], shape, radius * (1.45 if outline else 1.0))
            verts.append(tris)
            if outline:
                rgba = np.tile(np.array([0.03, 0.03, 0.04, 0.9], dtype="f"), (len(tris), 1))
            else:
                rgba = np.repeat(
                    np.hstack([colours[mask], np.ones((count, 1), dtype="f")]),
                    shape.shape[0] * 3, axis=0)
            cols.append(rgba)

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    if verts:
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(
            shader, 'TRIS',
            {"pos": np.concatenate(verts), "color": np.concatenate(cols)})
        shader.bind()
        batch.draw(shader)

    # A ring on whatever is under the cursor, and on anything pinned.
    pins = pinned(obj)
    marks = [(i, 1.0) for i in pins]
    hover = _state["hover"]
    if hover >= 0 and hover not in pins:
        marks.append((hover, 0.75))
    ring_shader = gpu.shader.from_builtin('FLAT_COLOR')
    for i, alpha in marks:
        if i >= len(snap) or not visible[i]:
            continue
        ring = _fans(screen[i:i + 1], _ngon(20), radius * 2.5)
        hole = _fans(screen[i:i + 1], _ngon(20), radius * 2.0)
        rgba = np.tile(np.array([1.0, 1.0, 1.0, alpha], dtype="f"), (len(ring), 1))
        ring_shader.bind()
        batch_for_shader(ring_shader, 'TRIS',
                         {"pos": ring, "color": rgba}).draw(ring_shader)
        # Punching the middle back out with the viewport's own background is not
        # possible, so the inner disc is drawn transparent-black instead, which
        # reads as a ring against anything.
        batch_for_shader(
            ring_shader, 'TRIS',
            {"pos": hole,
             "color": np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype="f"),
                              (len(hole), 1))}).draw(ring_shader)

    font = 0
    cards = list(pins)
    if hover >= 0 and hover not in cards:
        cards.append(hover)
    bounds = _usable_rect(region)
    taken: list[tuple[float, float, float, float]] = []
    for i in cards:
        if i >= len(snap) or not visible[i]:
            continue
        title, rows = describe(snap, i)
        accent = tuple(float(c) for c in colours[i])
        width, height = _card_box(font, title, rows, scale)
        cx, cy = _place_card(screen[i][0] + 18.0 * scale,
                             screen[i][1] - 24.0 * scale - height,
                             width, height, taken, bounds)
        taken.append((cx, cy, width, height))
        _draw_card(font, cx, cy, title, rows, accent, scale)

    _draw_legend(font, region, snap, visible, s.inspect_color_by, scale)

    gpu.state.blend_set('NONE')


def _tag_redraw() -> None:
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _rebuild_inspected() -> None:
    """Re-derive the inspected plant's snapshot, and put the mesh back in step
    with it.

    **`bridge.regrow` rather than `ensure_snapshot`, and that is the fix rather
    than a precaution.** Measured after an undo: the profile and all 27 curves
    read *identical* to before, yet the snapshot grown from them missed the
    restored mesh by **154 mm on average and 342 mm at worst** -- same node
    count, completely different geometry. Growing the plant and writing the
    mesh from that one grow makes the two agree by construction, whatever the
    scene happens to read as mid-undo.
    """
    name = _state["object"]
    if not name:
        return
    obj = bpy.data.objects.get(name)
    if obj is None or not obj.get("plantgen"):
        stop()
        _tag_redraw()
        return
    from . import live

    # `live.recover` settles the evaluated scene and regrows only if the plant
    # is actually out of step -- and it is the ONLY undo-time regrow now. This
    # used to call `bridge.regrow` itself, which meant two grows per undo on
    # separate timers producing different plants from the same stamp.
    if not live.recover(obj.name):
        # Not stale: the restored mesh is right, so only the snapshot is
        # missing. `bridge.regrow` captures one, so this is a no-op when
        # `recover` did grow.
        ensure_snapshot(obj)
    _state.update(hover=-1, screen=None, visible=None, region=0)
    _tag_redraw()


def _deferred_rebuild() -> None:
    # Reconcile first: undo may have turned inspection on or off, and there is
    # no point rebuilding a snapshot for a plant that is no longer inspected.
    reconcile()
    _rebuild_inspected()
    return None          # a timer returning None is not rescheduled


@bpy.app.handlers.persistent
def _refresh_after_undo(*_args) -> None:
    """Undo restores datablocks. It cannot restore a Python dict.

    Measured: undoing back to Age 1.0 put the plant back to 205 vertices while
    the cached snapshot still described the 22-vertex plant at Age 0.4 -- so
    every glyph would have been drawn in the wrong place and every card would
    have described a component that no longer existed. Nothing errored; the
    overlay just quietly stopped being about the plant on screen.

    The whole cache is dropped rather than only the inspected plant's, because
    an undo can revert any of them, and a snapshot costs nothing until it is
    asked for.

    **Pins survive only when the rebuilt plant is structurally identical.** A
    pin is an index into the snapshot, not an identity: if the structure moved,
    index N is a different bud and the pin would silently re-point at it --
    the same reasoning `capture` already uses. When nothing moved, which is the
    common case of undoing a cosmetic change, there is no reason to lose them.
    """
    _snapshots.clear()
    # Scheduled unconditionally, **including when nothing is being inspected**.
    # An early return here was the bug that stopped Ctrl+Z re-opening the
    # inspector: undo had restored `inspect_active` on the object, but with the
    # overlay currently off there was nothing to rebuild, so the timer never
    # ran and nobody noticed the flag had changed. Reconciling is exactly the
    # work that has to happen in that state.
    #
    # **Deferred onto a zero-delay timer rather than done here**, and that is
    # what fixed the reported "the debug points move" -- rebuilding inside the
    # handler reads a scene that is still mid-undo, and produced a plant 154 mm
    # away from the one undo had just restored. By the time a timer fires,
    # everything has settled.
    #
    # It also keeps datablock writes out of an undo handler, which is its own
    # good reason.
    if not bpy.app.timers.is_registered(_deferred_rebuild):
        bpy.app.timers.register(_deferred_rebuild, first_interval=0.0)


@bpy.app.handlers.persistent
def _forget_on_load(*_args) -> None:
    """A new file means every cached snapshot describes plants that are gone,
    and `_state` names an object from the previous one."""
    _snapshots.clear()
    stop()


_handler = None

_HANDLERS = (
    (lambda: bpy.app.handlers.undo_post, _refresh_after_undo),
    (lambda: bpy.app.handlers.redo_post, _refresh_after_undo),
    (lambda: bpy.app.handlers.load_post, _forget_on_load),
)
"""The lists are fetched through a lambda rather than captured, so this table
still resolves after a module reload has replaced them."""

_HANDLER_NAMES = frozenset({"_refresh_after_undo", "_forget_on_load"})


def _drop_handlers() -> None:
    """Removed by **name**, not by identity: a module reload replaces the
    function object, so the one sitting in Blender's handler list is no longer
    the one this module can reach -- and an unremovable handler that keeps
    firing forever is the classic way an addon reload leaves wreckage behind.
    """
    for fetch, _fn in _HANDLERS:
        lst = fetch()
        for existing in [f for f in lst if getattr(f, "__name__", "") in _HANDLER_NAMES]:
            lst.remove(existing)


def register() -> None:
    """Registered for the addon's whole lifetime rather than switched on and
    off with the overlay: a draw handler left behind by a failed teardown
    survives an addon reload and draws forever with no way to reach it. `_draw`
    returns immediately when nothing is being inspected."""
    global _handler
    unregister()
    _handler = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_PIXEL')
    for fetch, fn in _HANDLERS:
        fetch().append(fn)


def unregister() -> None:
    global _handler
    if _handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handler, 'WINDOW')
        except (ValueError, RuntimeError):
            pass
        _handler = None
    _drop_handlers()
    stop()
    _snapshots.clear()
