"""Stage 1 -- Morphology: what kind of plant is this?

A `MorphologyProfile` is a **specification**, not a simulation. It says what
the plant's potential architecture is; `development.py` decides what of that
potential actually gets built, and when. Keeping the two apart is the whole
point of the rebuild -- the previous design let one attraction-point cloud
decide architecture, timing and position all at once, which is why every
plant came out the same shape at a different scale.

Two vocabulary notes, because the terms are close enough to be conflated and
the model needs both as separate controls:

* **Apical dominance** is the apex suppressing the *release* of axillary buds
  below it. It decides whether a branch exists at all. Here: a curve over
  plant age, times a reach in phytomers.
* **Apical control** is the parent axis growing *faster and larger* than the
  laterals it has already released. It decides how big a branch gets. Here:
  a single 0-1 multiplier on a lateral's growth rate and internode length.

The previous build had a rough version of the first and nothing at all for
the second, which is part of why laterals overran the main axis so completely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .curves import (
    Curve,
    default_elongation,
    default_flowering_onset,
    default_head_dome_profile,
    default_leaf_angle_profile,
    default_leaf_expansion,
    default_leaf_size_profile,
    default_senescence,
    default_stem_thickening,
)
from .maps import EMERGENCE_DELAY, ParamMap, stack_bound
from .masks import MaskEntry, MaskStack


class CrossSection:
    """The shape swept along a stem -- the profile curve `Curve to Mesh` uses.

    Ported from the parametric add-on's own set (its `axis_cross_section`), logic not
    code: the original is a pure function over `StemParams` but reaching it
    would make the parametric add-on a hard load-time dependency for something that is
    not cosmetic, and a stem with no profile is not graceful degradation. The
    mapping below is the part worth keeping and it is twenty lines.

    A botanical trait, not an artistic one, which is why it sits on the profile
    rather than being a panel-only control: "sedges have edges, rushes are
    round, grasses have nodes" is a real field mnemonic, and a square stem is
    diagnostic of the mint family.
    """

    CIRCULAR = "circular"
    TRIANGULAR = "triangular"
    SQUARE = "square"
    HEXAGONAL = "hexagonal"
    POLYGONAL = "polygonal"
    FLATTENED = "flattened"
    WINGED = "winged"
    RIBBED = "ribbed"
    STAR = "star"
    CUSTOM = "custom"


_FIXED_SIDES = {CrossSection.TRIANGULAR: 3, CrossSection.SQUARE: 4,
                CrossSection.HEXAGONAL: 6}
_LOBED_SHAPES = (CrossSection.WINGED, CrossSection.RIBBED, CrossSection.STAR)
_CIRCULAR_SIDES = 32
_LOBE_SIDES = 64
"""Enough points to resolve a lobe as a lobe. Not user-facing: it trades
smoothness against vertex count on a curve swept along every stem, and neither
is a property of the plant."""


class Lifespan:
    """How a plant lives across years.

    Consulted only on the day path, and only past year 0's season -- so a plant
    with any lifespan is byte-identical to before at the default mature Day 270
    (year 0), and the age path (goldens, every existing caller) ignores it
    entirely. `None` on the profile is the default and means the pre-phase-5
    behaviour: a single season that clamps mature past its end.

    * ``ANNUAL`` -- one life. Grows, blooms and dies at season end (fully
      senesced, standing dead), and never regrows.
    * ``HERBACEOUS_PERENNIAL`` -- dies back to bare ground each winter and
      regrows a fresh top each spring from a persistent (conceptual, until roots
      exist) clump that expands subtly year over year toward a mature plateau.
    """

    ANNUAL = "annual"
    HERBACEOUS_PERENNIAL = "herbaceous_perennial"


_YEAR_DAYS = 365.0
"""A calendar year (phase 5). The season sits at
``[emergence_day, emergence_day + season_length_days]`` inside each one; the
rest is dormancy. A fixed real year keeps the day counter aligned to the
calendar/month view, and `emergence_day` is already a day-of-year."""


# AxisType (determinate / indeterminate) was removed 2026-09-30: whether a tip
# flowers is the inflorescence's Terminal attachment, and flowers along a stem
# its Axillary / Internodal / Leaf-opposed ones.
# Presets carrying `axis_type` migrate at schema 9 (`serialize`).


class Topology:
    """What a maturing meristem *does* -- the structural half of an
    inflorescence, as opposed to how its flowers are arranged in space.

    These four are not a new taxonomy; they are what the
    `InflorescenceKind` values already collapse to, measured. Captured as
    golden output before this existed: raceme, spike, corymb and catkin are
    byte-identical to each other (263 nodes / 32 axes / 59 flowers), as are
    umbel and flat umbel (143 / 27 / 161). Only solitary and cyme have
    topologies of their own. The kinds differ in *geometry* -- sessile,
    flat-topped, pendulous, how tightly spaced -- which is a separate axis and
    stays exactly where it was.
    """

    TERMINAL_FLOWER = "terminal_flower"
    """The apex becomes one flower and the axis stops. Solitary."""
    FLOWERING_REGION = "flowering_region"
    """The apex stays vegetative; the nodes it goes on making bear flowers
    instead of leaves. Raceme and its geometric variants."""
    APEX_CLUSTER = "apex_cluster"
    """Every flower attaches at the apex at once. Umbel and flat umbel."""
    RELAY = "relay"
    """The apex flowers and is spent; buds below it take over and repeat the
    whole pattern. Cyme -- the sympodial case."""
    BRANCHED = "branched"
    """The apex completes its own development and stops **without flowering**.
    What carries the flowers are the branches below it, each of which repeats
    the pattern one order deeper until the last order, which flowers. Panicle.

    This is the fourth structure, and the one the other three could not
    express: every other topology ends an axis by putting flowers on it, so
    *no axis could mature without flowering*. That gap is what made a panicle
    unrepresentable -- a panicle's main rachis is indeterminate and bears
    branches, not flowers -- and it is the same gap the parked catkin names as
    its own blocker, since a catkin has to hang from a twig that did not itself
    turn into an inflorescence. See `InflorescenceKind.PANICLE`."""
    NONE = "none"
    """The apex completes its own development and stops, bearing **nothing**.

    The fifth structure, and the one the catkin was parked on. `BRANCHED`
    already lets an axis mature without flowering, but it matures into an
    inflorescence anyway -- its nodes carry branches. This carries nothing at
    all: the axis reaches its species-typical node count, its apex aborts, and
    what it has made is an ordinary leafy shoot that has finished.

    That is what a catkin hangs *from*. Everything a catkin needs structurally
    was measured working already; the one thing missing was a twig that could
    stop without turning into an inflorescence itself, because
    `physics._peduncle_from` leaves a flowerless axis alone and so the twig
    would stop drooping for free. Capping it with a single terminal flower
    instead was measured and is worse -- the whole peduncle turn lands on the
    two nodes below that one flower, giving a J-hook rather than an arch.

    **Not the same thing as a plant whose meristems never flower.** Setting
    `Vegetativeness` to neither-floral works by making
    `_terminate_determinate_axes` return early, so nothing ever stops and the
    plant grows until it runs out of span -- measured on the generic perennial,
    **468 nodes / 84 axes / 1.049 m** against the default's 143 / 27 / 0.534,
    and it does not even reliably reach zero flowers (30 of them, from the
    `max_branch_order` fallback in `_release_buds` making an unbranchable
    lateral into a flower). NONE is the same plant minus its flowers: measured
    **143 / 27 / 0.534 m**, node for node the default, with 0 flowers. It also
    routes per order through `inflorescence_for`, which is what lets one order
    bear catkins while another bears nothing.

    See `InflorescenceKind.NONE`."""


@dataclass
class Vegetativeness:
    """Prusinkiewicz, Erasmus, Lane, Harder & Coen, *Science* 316:1452 (2007).

    One continuous quantity per meristem -- **vegetativeness** -- with shoot
    identity at the high end and flower identity at the low end, falling as the
    meristem ages. A meristem whose veg reaches the threshold becomes floral.
    What produces the architecture is not the value but the *difference*
    between an apex and the laterals below it:

    ======================  ==============  =================  ===========
    apex reaches floral?    laterals do?    resulting form     classical
    ======================  ==============  =================  ===========
    yes, immediately        no laterals     terminal flower    solitary
    no                      yes             flowering region   raceme
    yes                     no -- they      relay              cyme
                            stay vegetative
    yes                     yes, but only   branched, then     panicle
                            after growing   flowering
    ======================  ==============  =================  ===========

    The bottom row is the point, and it took two passes to land. This class
    made the *distinction* sayable -- an apex and its laterals having different
    fates is what produces the architecture -- but not the panicle itself: see
    `decline` for why the apex half of the mechanism never actually ran. What
    builds a panicle is `Topology.BRANCHED`, which supplies the piece all four
    of these rows assume away, namely an axis that can mature *without*
    flowering. The table stands as the reasoning; PANICLE is the implementation
    of its last row.

    **This is the mechanism, not the parameter surface.** The panel says where
    flowers arise with the inflorescence's attachments (a tip flowers when it
    is Terminal), and `default` is what that implies: an apex that may flower,
    laterals that never turn into flowers on their own. A species that wants
    something else (a panicle) sets `MorphologyProfile.veg` directly in Python.
    """

    apex: float = 1.0
    """The apex's veg when its axis reaches maturity. At or below `threshold`
    the apex itself becomes a flower."""
    lateral_offset: float = -1.0
    """What a lateral meristem carries relative to its parent's. Negative makes
    laterals floral before the parent (raceme); positive keeps them vegetative
    after the parent has flowered (cyme)."""
    decline: float = 0.0
    """How much veg a meristem loses per developmental step of its own age.
    Zero makes veg a fixed property of position; positive lets a lateral start
    vegetative, age, and *then* reach floral identity.

    **Reaches less than it looks like, measured 2026-08-19.** Only
    `lateral_is_floral` is ever called with a real age (`_release_buds`, on the
    `max_branch_order` fallback); `apex_is_floral` is called with none, so on
    the apex `decline` multiplies by zero and does nothing at all. And the
    lateral path it does reach turns a bud into a *single flower*, never into
    "a branch that later flowers" -- so this cannot build the panicle the table
    above describes, whatever value it is given. `Topology.BRANCHED` is what
    actually builds one. Left in place because the lateral half is real and
    working; do not read the apex half as live."""
    threshold: float = 0.0

    def apex_is_floral(self, meristem_age: int = 0) -> bool:
        return self.apex - self.decline * max(meristem_age, 0) <= self.threshold

    def lateral_is_floral(self, meristem_age: int = 0) -> bool:
        return (self.apex + self.lateral_offset
                - self.decline * max(meristem_age, 0)) <= self.threshold

    @classmethod
    def default(cls) -> Vegetativeness:
        """What a plant is unless its species says otherwise: the apex may
        flower (whether it does is its inflorescence's Terminal attachment) and
        a lateral never turns into a flower on its own (flowers along a stem
        are its Axillary / Internodal / Leaf-opposed attachments). This was the
        old Determinate Axis Type exactly, so every shipped species grows as
        before.

        **The offset is positive on purpose.** Laterals stay vegetative after
        the apex has flowered -- the relay pattern, and why relays make a cyme.
        Negative would make every lateral that cannot branch any deeper turn
        into a flower, which the goldens catch as a sub-millimetre shift in
        bounds (extra organs consume ids, which shifts every later organ's own
        seeded draw) long before it shows."""
        return cls(apex=0.0, lateral_offset=1.0, decline=0.0)


class PhyllotaxisPattern:
    ALTERNATE = "alternate"
    SPIRAL = "spiral"
    DISTICHOUS = "distichous"
    UNILATERAL = "unilateral"
    OPPOSITE = "opposite"
    DECUSSATE = "decussate"
    WHORLED = "whorled"


@dataclass
class Phyllotaxis:
    """Where leaves sit around the stem, and therefore how many axillary buds
    a node carries -- the two are the same question under the phytomer model,
    since every leaf has a bud in its axil.

    `angles_at` returns one divergence angle per leaf at that node, measured
    around the stem from the phytomer's own transported reference vector.
    """

    pattern: str = PhyllotaxisPattern.ALTERNATE
    divergence_deg: float = 137.5
    whorl_count: int = 3

    def angles_at(self, index: int) -> list[float]:
        p = self.pattern
        if p == PhyllotaxisPattern.SPIRAL:
            return [(index * 137.5) % 360.0]
        if p == PhyllotaxisPattern.DISTICHOUS:
            return [(index * 180.0) % 360.0]
        if p == PhyllotaxisPattern.UNILATERAL:
            return [0.0]
        if p == PhyllotaxisPattern.OPPOSITE:
            # A fixed plane up the whole axis -- successive pairs stack rather
            # than rotating. Rotating pairs are Decussate, below.
            return [0.0, 180.0]
        if p == PhyllotaxisPattern.DECUSSATE:
            base = 0.0 if index % 2 == 0 else 90.0
            return [base, base + 180.0]
        if p == PhyllotaxisPattern.WHORLED:
            step = 360.0 / max(self.whorl_count, 2)
            # Successive whorls offset by half a step, the natural
            # generalisation of decussate's alternation -- stacked whorls in
            # perfect register read as a mistake rather than as a plant.
            offset = (index % 2) * step * 0.5
            return [(offset + i * step) % 360.0 for i in range(max(self.whorl_count, 2))]
        return [(index * self.divergence_deg) % 360.0]

    @property
    def leaves_per_node(self) -> int:
        return len(self.angles_at(0))


class LevelPhyllotaxis:
    """The Phyllotaxis dropdown of a branching level: each named pattern is a
    preset of the three numbers a level actually carries (spiral angle, whorl,
    whorl angle); CUSTOM is any hand-set combination."""
    CUSTOM = "CUSTOM"
    ALTERNATE = "ALTERNATE"
    DISTICHOUS = "DISTICHOUS"
    OPPOSITE = "OPPOSITE"
    DECUSSATE = "DECUSSATE"
    WHORLED = "WHORLED"

    PRESETS = {
        ALTERNATE: (137.5, 1, 0.0),     # golden spiral, one per node
        DISTICHOUS: (180.0, 1, 0.0),    # two ranks
        OPPOSITE: (0.0, 2, 0.0),        # pairs stacked up the stem
        DECUSSATE: (90.0, 2, 0.0),      # pairs, each turned 90 deg from the last
        WHORLED: (60.0, 3, 0.0),        # threes, each turned half a step
    }


def slot_angles(index: int, spiral_deg: float, whorl: int, whorl_angle_deg: float) -> list[float]:
    """Slot angles round a stem at node `index`, in degrees: the node turned
    `spiral_deg` from the last, `whorl` slots spaced `whorl_angle_deg` apart
    (evenly when 0). The one arrangement rule the branching levels and the
    leaves share."""
    n = max(1, int(whorl))
    step = whorl_angle_deg if whorl_angle_deg > 0.0 else 360.0 / n
    base = index * spiral_deg
    return [(base + i * step) % 360.0 for i in range(n)]


@dataclass
class LeafArrangement:
    """Where grown leaves sit round each node -- the primary leaf component's
    Phyllotaxis (its Spiral, Whorl and Whorl Angle). Also where a flowering
    node's flowers, bracts and catkins sit unless the inflorescence sets its
    own whorl. The default is a golden-angle spiral, one leaf a node."""

    spiral_angle_deg: float = 137.5
    whorl: int = 1
    whorl_angle_deg: float = 0.0

    def angles_at(self, index: int) -> list[float]:
        return slot_angles(index, self.spiral_angle_deg, self.whorl, self.whorl_angle_deg)

    @property
    def leaves_per_node(self) -> int:
        return max(1, int(self.whorl))


PARENT_PREVIOUS = -2
PARENT_PLANT = -1


class GrowthPattern:
    """How a level's axes carry their growth on:
    the one apex (monopodial), relays from a unit that ends (the sympodial
    family), or a tip that splits (the forks)."""

    MONOPODIAL = "monopodial"
    """One apex builds the whole axis; branches come off its sides."""
    SYMPODIAL = "sympodial"
    """The axis grows in units; each ends and one relay carries on."""
    DICHASIAL = "dichasial"
    """Each unit ends and two relays carry on (a false fork)."""
    PLEIOCHASIAL = "pleiochasial"
    """Each unit ends and a ring of Relays (3+) carries on."""
    PSEUDO_RELAY = "pseudomonopodial_relay"
    """A sympodium whose one relay continues in line, the spent apex pushed
    aside: it reads as one straight stem with side-set flowers."""
    DICHOTOMOUS = "dichotomous"
    """The tip splits into equal forks (the level's old Fork)."""
    PSEUDO_FORK = "pseudomonopodial_fork"
    """The tip splits unequally: a leader carries on, weaker forks peel off."""

    RELAYING = frozenset({SYMPODIAL, DICHASIAL, PLEIOCHASIAL, PSEUDO_RELAY})
    FORKING = frozenset({DICHOTOMOUS, PSEUDO_FORK})


@dataclass
class BranchLevel:
    """One level of the branching stack.

    Level 0 is the stems (on it Whorl is the stem count, filling the base as a
    phyllotactic disc, and Start Pitch is their splay); level n+1 is the
    branches on level n's axes. Angles in degrees; every scale is a plain
    multiplier. Relative quantities are relative to the parent where the
    branch attaches (on level 0, which has no parent, Start Length and Start
    Width are the stems' own first internode and base radius, in metres).

    The levels form a tree: `parent` is the
    level this one grows on. The default, `PARENT_PREVIOUS`, is the level
    before it in the list -- the chain every species and older preset is --
    and `PARENT_PLANT` makes it a stems level growing from the ground. Levels
    sharing a parent each branch at every node of it, with their own
    phyllotaxis; their Masks split the nodes between them."""

    name: str = ""
    """The level's name in the Plant Structure tree ("" = Stems / Branches n)."""
    parent: int = -2
    """The index of the level it grows on; `PARENT_PREVIOUS` (-2) the one
    before it, `PARENT_PLANT` (-1) none -- a stems level."""
    phyllotaxis: str = LevelPhyllotaxis.ALTERNATE
    spiral_angle_deg: float = 137.5
    """How far each node's slots turn round the parent from the last node's."""
    whorl: int = 1
    """Branches per node, spaced round the parent."""
    whorl_angle_deg: float = 0.0
    """Angle between the members of one whorl; 0 spreads them evenly."""
    slide: float = 0.0
    """Slides a node's branches along the stem, -1..1 (+ up, - down): the
    members of a pair move apart, the first up and the second down, half an
    internode apart at 1 (a quarter internode each way; a whorl spread evenly
    over that, a single branch the full quarter)."""
    slide_variation: float = 0.0
    """Random slide per node, added to `slide` (clamped to -1..1): some pairs
    slide up, some down."""

    start_pitch_deg: float = 45.0
    """The angle the branch leaves its parent at (level 0: the stems' splay)."""
    start_yaw_deg: float = 0.0
    """Turn of the branch sideways at launch, about the parent's normal."""
    start_roll_deg: float = 0.0
    """Twist of the branch's own frame about its axis at launch -- where its
    own phyllotaxis starts."""
    upper_side_up: bool = False
    """Start each branch's frame at its upper side (world up, projected across
    the branch) instead of where the parent's frame carries it -- a
    dorsiventral branch, with a top and an underside (a plagiotropic branch
    in Massart's model: Tomlinson 2009). The parent's frame comes round a
    whorl turned differently for each member, so without it a Distichous
    level on its branches lies up/down on some and sideways on others. Start
    Roll turns from the upper side (90: its side). False: as before."""
    start_pitch_variation_deg: float = 0.0
    start_yaw_variation_deg: float = 0.0
    start_roll_variation_deg: float = 0.0
    """Degrees each stem / branch's Start Pitch / Yaw / Roll deviates either way,
    drawn once per stem or branch on the level's seed -- so a clump's stems splay
    and its branches leave at slightly different angles. 0: all alike."""

    steps: int = 12
    """Nodes the axis has when fully grown (vegetative; a determinate axis
    flowers after). Positions along it (`factor`) are counted over them and
    the plant's clock (`total_steps`) is sized from them."""

    emergence_delay: int = 0
    """Emergence Delay: growth steps each branch of this level waits, after the
    node it leaves forms, before it starts growing (Natsura's Delay); on a stems
    level, steps after the plant starts. x its `emergence_delay` map per branch
    / stem. Timing only: growth runs exactly as without it and the delay only
    shifts when each part is SHOWN (`development._reveal`), the plant shown over
    `total_steps + emergence_span` -- so the full-grown plant is the same one,
    byte for byte, and every earlier Day is a part of it. 0 = none."""

    growth: float = 1.0
    """Growth %: how much of its Steps each axis of this level builds, 0-1
    (Natsura's Growth %), x its `growth` map per branch. Fractional -- the last
    node built grows that part of its length -- and NOT renormalised: a branch
    at 0.5 is the lower half of the full one (its maps, taper and clock as at
    1), and still flowers at its cut tip. 1 = all of it, byte-identical."""

    stiffness: float = 0.0
    """This level's stems / branches' Young's modulus, in pascals -- how hard
    they resist bending under their load (`physics._axis_modulus`). 0 takes the
    plant's `stem_stiffness`, as before levels had one."""

    seed: int = 0
    """This level's own seed, mixed with the plant's: changing it redraws only
    this level's variation (its axes' length and width, slide, fork chance and
    mask survival), the rest of the plant held. 0 draws from the plant's seed
    alone, as before levels had one."""

    start_length_scale: float = 1.0
    """First internode relative to the parent's internode where it attaches --
    on the stems level (no parent), the stems' first internode in metres."""
    length_scale_per_step: float = 1.0
    """Each internode is this times the previous one."""
    length_variation: float = 0.0
    """How much each internode's length varies node to node around it, as a
    fraction (drawn per node)."""
    start_width_scale: float = 1.0
    """Base width relative to the parent's width where it attaches -- on the
    stems level (no parent), the stems' base radius in metres."""
    width_scale_per_step: float = 1.0
    """Each node's width is this times the previous one's."""
    width_variation: float = 0.0
    """How much each axis's width varies from its siblings', as a fraction --
    drawn once per axis (its base), so one stem's taper never goes ragged."""

    # The shape swept along this level's axes (the Built mesher traces it).
    cross_section: str = CrossSection.CIRCULAR
    profile_sides: int = 8
    """Sides of a Polygonal section. Ignored by every other shape."""
    lobe_count: int = 5
    lobe_depth: float = 0.35
    """Shared by Winged, Ribbed and Star; the panel relabels the rows per shape
    ("Wings" / "Ribs" / "Points")."""
    flatten: float = 0.6
    """How far a Flattened section is squashed across one axis. 0 is round, 1
    is a ribbon."""

    spiral_pitch_deg: float = 0.0
    """The apex heading bends this many degrees per step."""
    spiral_roll_deg: float = 0.0
    """The bend's direction turns this many degrees per step (with pitch, a
    helix)."""

    growth_pattern: str = GrowthPattern.MONOPODIAL
    """How this level's axes carry their growth on (`GrowthPattern`)."""
    fork_branches: int = 2
    """Continuations a fork splits into."""
    fork_probability: float = 0.0
    """Chance per node that the growing tip forks there."""
    fork_spread_deg: float = 30.0
    """Total spread of a fork or of a unit's relays: each continuation leaves
    by half of it (the aligned relay: how far the spent apex is pushed aside)."""
    fork_whorl_deg: float = 90.0
    """How far each fork / relay generation's plane turns from its parent's."""
    fork_balance: float = 0.5
    """The unequal fork's lopsidedness: 0 an equal fork, 1 a straight leader
    with forks half as long and thick, splayed twice as wide."""
    unit_steps: int = 3
    """Nodes in one sympodial unit before it ends and relays. The unit's tip
    flowers when the inflorescence is Terminal, else it dies back."""
    relay_count: int = 3
    """Relays a pleiochasial unit ends in."""

    masks: list[MaskEntry] = field(default_factory=list)
    """Which of this level's branches grow: the
    folded mask is each branch bud's seeded survival chance, drawn once; a bud
    it rejects stays dormant for good. On the stems level: which stems emerge.
    Empty (the default) keeps every one."""
    maps: list[ParamMap] = field(default_factory=list)
    """Maps on this level's parameters (`growth.maps.LEVEL_TARGETS`), each a
    multiplier read per bud / branch / node / fork. Empty is a no-op."""

    def angles_at(self, index: int) -> list[float]:
        """The slot angles round the parent at node `index` (`slot_angles`)."""
        return slot_angles(index, self.spiral_angle_deg, self.whorl, self.whorl_angle_deg)

    @property
    def forks(self) -> bool:
        """Whether this level's tips split (Dichotomous / Unequal Fork)."""
        return self.growth_pattern in GrowthPattern.FORKING

    @property
    def relays(self) -> int:
        """How many relays a unit ends in; 0 on a level that does not relay."""
        pattern = self.growth_pattern
        if pattern not in GrowthPattern.RELAYING:
            return 0
        if pattern == GrowthPattern.DICHASIAL:
            return 2
        if pattern == GrowthPattern.PLEIOCHASIAL:
            return max(3, int(self.relay_count))
        return 1

    def longest_delay(self) -> int:
        """The most steps Emergence Delay can hold one of this level's branches
        (its maps at their highest) -- what the plant's clock adds for it."""
        if self.emergence_delay <= 0:
            return 0
        bound = stack_bound(self.maps, EMERGENCE_DELAY)
        return max(0, math.ceil(self.emergence_delay * bound - 1e-9))

    def section_inputs(self) -> dict:
        """This level's section as the numbers the Built mesher traces: `sides`
        (a polygon's corners, else the round / lobed resolution), `lobe_count`
        and `lobe_depth` (Winged, Ribbed, Star -- radius x 1 + cos(lobes x angle)
        x depth) and `flatten` (Flattened's squash, 0 for every other shape).
        Custom (a Curve object) is not traced; it builds round."""
        shape = self.cross_section
        if shape == CrossSection.POLYGONAL:
            sides = max(3, self.profile_sides)
        elif shape in _LOBED_SHAPES:
            sides = _LOBE_SIDES
        else:
            sides = _FIXED_SIDES.get(shape, _CIRCULAR_SIDES)
        lobed = shape in _LOBED_SHAPES
        return {
            "mode": 0,
            "sides": sides,
            "lobe_count": max(2, self.lobe_count) if lobed else 0,
            "lobe_depth": self.lobe_depth if lobed else 0.0,
            "flatten": self.flatten if shape == CrossSection.FLATTENED else 0.0,
        }


def default_levels() -> list[BranchLevel]:
    """A single stem with one level of alternate branches -- the generic
    perennial's proportions (12 stem nodes from a 29 mm internode and a 9 mm
    radius; side shoots at 48 deg of 7 nodes)."""
    return [
        BranchLevel(start_pitch_deg=0.0, steps=12, start_length_scale=0.02925,
                    length_scale_per_step=1.05, length_variation=0.18,
                    start_width_scale=0.009, width_scale_per_step=0.90),
        BranchLevel(start_pitch_deg=48.0, steps=7, start_length_scale=0.6,
                    length_variation=0.18, start_width_scale=0.8,
                    width_scale_per_step=0.85),
    ]


class InflorescenceKind:
    """How flowers are arranged where an axis finishes.

    Botanically an inflorescence is a modified shoot -- the same axis, with its
    phytomers bearing flowers and bracts instead of leaves and vegetative buds.
    That is exactly how it is built here: an axis reaching its flowering node
    count switches into a flowering region rather than being replaced by a
    special structure.
    """

    NONE = "none"
    """No inflorescence at all: the axis matures and stops, bearing nothing.

    Two things this is for, and the second is the reason it exists.

    A **foliage species** -- a plant that is grown for its leaves and simply
    never flowers. Reachable before this only by setting `veg` directly in
    Python, which is a different plant: that route makes nothing ever stop, so
    the shoot runs on until the developmental span runs out.

    And **the twig a catkin hangs from**. A catkin is a small tube of florets
    on one tiny pedicel at a node of a young branch, so the branch itself must
    be able to finish without turning into an inflorescence -- otherwise it
    droops too, and the thing that reads as hanging is the branch rather than
    the catkin. Put NONE on the trunk and a catkin on `lateral_inflorescence`
    and the twig stays a twig. See `Topology.NONE`."""
    SOLITARY = "solitary"
    """One terminal flower. A Rudbeckia's head is a single instanced flower at
    this level -- the florets inside it are a flower-asset concern, not an
    axis-level one."""
    RACEME = "raceme"
    """Flowers on pedicels along a continuing terminal axis."""
    SPIKE = "spike"
    """A raceme with no pedicels -- flowers sit directly on the axis."""
    CORYMB = "corymb"
    """A raceme whose lower pedicels are longer, bringing every flower up to
    roughly one level."""
    UMBEL = "umbel"
    """Every pedicel radiating from a single point at the tip, filling a dome.
    Not a ring: an umbel has flowers across the whole cap including its
    centre, which needs an equal-solid-angle fill rather than one polar
    angle."""
    FLAT_UMBEL = "flat_umbel"
    """An umbel whose pedicels are solved to land on one plane, so the head
    reads flat rather than domed (cow parsley, Orlaya)."""
    # HEAD was here, removed 2026-08-17. It placed sessile flowers crowded over
    # the apex -- and "crowded" meant *coincident*: measured nearest-neighbour
    # spacing was 0.00 mm at 28, 60, 120 and 250 florets, because `is_sessile`
    # makes `pedicel_length` zero and every organ then sits exactly on the apex
    # node. Only facing differed. That is not a capitulum, and building one here
    # was ruled out twice already -- the parametric add-on removed its axis-level
    # Receptacle on 2026-07-29 as duplicating the per-flower one, and SOLITARY's
    # own docstring above says a head belongs in the flower asset. So the kind
    # was a placeholder for a feature that is deliberately not at this level.
    CATKIN = "catkin"
    """Catkins: the stems themselves bear no flowers, and hang catkins at their
    nodes instead -- hazel, birch, willow.

    **The one kind here that is not a structure on an axis's own apex**, and
    that is what a catkin is. Every other kind converts a shoot's tip into
    flowers; a hazel's twigs go on being twigs and carry catkins on short
    stalks along their length. Built the other way -- as a flowering region --
    what hangs is the *branch*, and no control can fix that, because the
    drooping thing is the branch. Measured: the whole Peduncle Droop range
    moved such a catkin 1.7 cm, against 50 cm on the same plant as a spike.

    **It no longer forces its bearing stem not to flower** -- that was
    attachment leaking into morphology, and `Attachment.AXILLARY` says it now.
    Picking Catkin stamps that attachment, plus a pendulous orientation and the
    `catkin_*` geometry block, and every one of those stays independently
    settable afterwards. See `properties._KIND_DEFAULTS`.

    What stays specific to this kind is only how its *body* is built: a swept
    tube rather than flowers on phytomers. That is a morphology property and is
    allowed to differ per kind, the same way `is_sessile` and the flat-top
    solve already do -- so the shared Orientation control drives a polyline sag
    here and an axis setpoint everywhere else. One control, two
    implementations.
    """
    CYME = "cyme"
    """The determinate case, and the true opposite of a raceme: the *terminal*
    flower opens first and stops that axis, and growth continues from one or
    more buds below it, each repeating the pattern. A raceme's tip keeps
    growing and adds flowers below it indefinitely; a cyme's tip is spent the
    moment it blooms, so the whole inflorescence is built by relaying.

    `laterals_per_node` distinguishes the classical forms -- one gives a
    monochasium, two a dichasium, three or more a pleiochasium -- and
    `relay_rotation_deg` distinguishes the two monochasia from each other. All
    four are named kinds of their own (`SCORPIOID`, `HELICOID`, `DICHASIUM`,
    `PLEIOCHASIUM`) which stamp those two numbers; this stays the unstamped
    form, for a combination they do not name.

    Depth is `branching_depth`, counted inside the inflorescence -- **not**
    `max_branch_order`, which stays purely vegetative, and which bounding it
    was the main reason a cyme relayed at most twice. A relay inherits its
    parent's `order`, exactly as a panicle's branches do. The recursion is
    free: a relayed axis reaches its own flowering node count and comes back
    through `_begin_inflorescence` on its own.
    """
    PANICLE = "panicle"
    """A branched raceme -- the compound racemose inflorescence of oat, rice,
    *Astilbe* and *Yucca*, and the source of the word (*Panicum*).

    Six structural facts, and the code answers all six:

    1. The main axis is **indeterminate**: it does not end in a flower. Here
       that is `Topology.BRANCHED`, which stops an axis without flowering it.
    2. Its nodes bear **branches, not flowers** -- the defining difference from
       a raceme. Those are inflorescence branches the inflorescence places
       itself (`_release_inflorescence_branches`), not the branching levels'.
    3. Those branches are **themselves panicles**, until the last order, which
       is a raceme. That recursion is free: each branch is an ordinary axis
       that reaches its own flowering target and arrives back at
       `_begin_inflorescence` at its own order.
    4. Branch length and division **decrease acropetally**, giving the
       pyramidal outline -- the inflorescence's own `branch_length_profile`.
    5. Bloom is **acropetal** overall -- the default `BloomOrder`.
    6. Depth is `Inflorescence.branching_depth`, counted inside the
       inflorescence -- **not** the branching levels, which stay purely
       vegetative. At `branching_depth = 0` a panicle is a plain raceme, which
       is the right degradation and is now said by its own control. Measured
       (before the branching overhaul): with no vegetative branching at all,
       depth 1 and 2 still gave 8 and 57 axes -- which is the decoupling.

    Not a **thyrse** (lilac), which has the same indeterminate main axis but
    *cymose* laterals -- that is this kind on the trunk with `CYME` as the
    `lateral_inflorescence`, and needs no kind of its own.

    Appended last on purpose: an `EnumProperty` stores the item's *index*, so
    inserting one anywhere else silently rewrites every saved plant's kind.
    Removing Head cost a repair pass for exactly this.
    """
    VERTICILLASTER = "verticillaster"
    """The false whorl of the mint family -- *Lamium*, *Leonurus*, *Mentha*,
    *Salvia*: rings of flowers spaced up a square stem, each ring subtended by
    a pair of leafy bracts, with bare stem between them.

    **It is a false whorl**, and the name says so: what looks like one ring is
    really two opposite condensed cymes, one in each bract axil. This builds it
    as a *real* whorl instead -- `whorl_count` flowers evenly around the node --
    which is the approximation the user asked for and the one worth making. At
    mint density a ring of single flowers and a ring of tiny cymes read the
    same; on a sparse *Lamium* this will read thinner than the real thing,
    because each half-whorl there is a visible little branched cluster rather
    than a few flowers.

    Two things make it a kind rather than a preset, and both were measured
    before it was built:

    1. **The whorl itself needed no new mechanism** -- `RACEME` with whorled
       phyllotaxis already puts six flowers in a 60-degree ring at every
       flowering node. But one `Phyllotaxis` serves a whole axis, so asking for
       a six-flowered whorl also gave the stem *below* six leaves per node.
       Decussate foliage under many-flowered whorls -- which is what every
       Lamiaceae plant is -- was unreachable. `whorl_count` is the flowering
       region stating its own arrangement, and it is available to `RACEME` and
       `SPIKE` too, since the coupling was never verticillaster-specific.
    2. **A flowering node carried no leaf at all** (measured: `leaves=0` at
       every node of the flowering region), so the leafy whorls that are the
       defining look had nowhere to come from. `bract_size` is that, drawn as
       its own leaf form -- see `Organ.form`.

    Both default to off, so this kind is the only thing that turns them on and
    every other kind is untouched.
    """
    SCORPIOID = "scorpioid"
    """A monochasial cyme whose successive relays alternate sides, so the
    inflorescence zigzags in one plane -- the forget-me-not, comfrey and
    heliotrope pattern (Boraginaceae generally), and *Hemerocallis*.

    One relay per fork, `relay_rotation_deg` 180. Nothing else separates it
    from `HELICOID`: both are monochasia, and which one you get is entirely
    which way the next relay is turned. That is why they are two stamps on one
    mechanism rather than two mechanisms.

    The **zigzag intensity is `branch_angle_deg`**, and deliberately not a
    control of its own: the kink is the angle each relay leaves its spent
    parent at, which is exactly what that field already states. +90 continues
    straight on with no visible kink, lower values fork harder each generation.
    Adding a "zigzag" slider would be a rescaling of a quantity an existing
    control already carries, which is the standing rule here.
    """
    HELICOID = "helicoid"
    """A monochasial cyme whose successive relays all arise on the **same**
    side, so the inflorescence coils -- the bostryx of *Alstroemeria*,
    *Drosera* and many Solanaceae.

    One relay per fork, `relay_rotation_deg` 0. Because the frame is
    parallel-transported from generation to generation, a constant azimuth and
    a constant launch angle trace a curve of constant curvature -- a coil --
    where 180 would trace a zigzag. Gravitropism then pulls the coil out of its
    plane into a real helix.

    Botanically a helicoid and a scorpioid cyme differ in which of the two
    bracts subtends the relay; here they differ in one number, which is the
    same statement made cheaply.
    """
    DICHASIUM = "dichasium"
    """The familiar forked cyme: the spent tip is flanked by **two** relays,
    each of which forks again -- chickweed, campion, *Stellaria*, *Silene*,
    most Caryophyllaceae.

    Two relays per fork, `relay_rotation_deg` 90, so successive forks are
    decussate rather than stacking in one plane -- which is what a real
    dichasium does and what keeps it reading as a three-dimensional spray.

    **This was unreachable before**, and not by a small margin: relays were
    taken from whatever axillary buds happened to survive, and alternate
    phyllotaxis only ever supplies one per node, so `laterals_per_node` could
    not exceed 1 whatever it said. The inflorescence now places its own relays.
    """
    PLEIOCHASIUM = "pleiochasium"
    """Three or more relays below the spent tip, in a whorl around it --
    *Euphorbia*'s cyathia, and the classic pleiochasial umbel-like spray.

    Same mechanism as `DICHASIUM` with a higher `laterals_per_node`; the
    default stamp is three at 60 degrees of rotation per generation, which
    offsets each whorl against the one below rather than stacking them in
    register (the same rule `Phyllotaxis.WHORLED` already applies to leaves,
    and for the same reason -- stacked rings read as a mistake).

    Note the count compounds: `laterals_per_node ** branching_depth` axes, so
    three relays at depth 3 is 27 terminal flowers and depth 5 is 243.
    """



# ======================================================================
# Kind capabilities -- the single source of truth for what each
# inflorescence kind can and cannot do.
#
# These module-level sets and the topology map are the ONE place the per-kind
# facts live. Three consumers read them and can never disagree: the
# `Inflorescence` instance properties below, the panel (`blender/ui.py`, which
# hides the controls a kind makes inert), and the generated LLM/human spec
# (which lists that same inertness). Add a kind in exactly one place here.
# ======================================================================

RELAY_KINDS = frozenset((
    InflorescenceKind.CYME,
    InflorescenceKind.SCORPIOID,
    InflorescenceKind.HELICOID,
    InflorescenceKind.DICHASIUM,
    InflorescenceKind.PLEIOCHASIUM,
))
"""Every kind built by relaying rather than by a flowering region.

The four named forms are one mechanism with two numbers set differently --
`laterals_per_node` and `relay_rotation_deg` -- which is why they are stamps on
`CYME`'s own machinery rather than four branches in `development`. `CYME` stays
as the unstamped form for anyone who wants a combination the four do not name.
"""

TERMINAL_CLUSTER_KINDS = frozenset((
    InflorescenceKind.UMBEL, InflorescenceKind.FLAT_UMBEL))
"""Every flower attaches at the apex (a dome), not along a flowering region, so
the dome states each flower's angle and azimuth. The gradient curves and the
Pedicel Angle/Spread controls that would otherwise set those are inert here.

Measured, fingerprinting facing and scale as well as position: Pedicel Angle and
Spread are byte-for-byte inert on both, because `_dome_fill` writes each flower's
`elevation_deg` and `divergence_deg` outright and never routes the azimuth
through `_spread`."""

FLAT_TOPPED_KINDS = frozenset((
    InflorescenceKind.CORYMB, InflorescenceKind.FLAT_UMBEL))
"""Pedicel lengths and angles are solved after growth so every flower lands on
one plane (`inflorescence.solve_flat_head`); nothing about a pedicel is the
user's to set. Head Radius and Head Dome say how wide and how round instead."""

SESSILE_KINDS = frozenset((InflorescenceKind.SPIKE,))
"""No pedicels, but a real flowering region -- flowers sit directly on the axis.
The model sense of 'sessile': distinct from a capped kind, which has no
flowering region at all."""

CAPPED_KINDS = frozenset((InflorescenceKind.SOLITARY,)) | RELAY_KINDS
"""One terminal flower per axis, placed by `development._cap_with_flower` and
aimed straight up the axis it caps -- so it never passes through the pedicel or
flower-gradient machinery, and Pedicel Length / Flower Pitch / Spread have
nothing to write to. What aims a capped flower is the axis under it (Peduncle
Droop, Rigid Base).

Measured, fingerprinting organ facing as well as position (a position-only
fingerprint reports every orientation control as dead -- a trap this project has
hit three times): on a cyme, Flower Spread, Flower Pitch and Pedicel Length each
change 0 of 37 organ directions across their whole range, and solitary is the
same."""

KIND_TOPOLOGY = {
    InflorescenceKind.NONE: Topology.NONE,
    InflorescenceKind.SOLITARY: Topology.TERMINAL_FLOWER,
    InflorescenceKind.RACEME: Topology.FLOWERING_REGION,
    InflorescenceKind.SPIKE: Topology.FLOWERING_REGION,
    InflorescenceKind.CORYMB: Topology.FLOWERING_REGION,
    InflorescenceKind.VERTICILLASTER: Topology.FLOWERING_REGION,
    InflorescenceKind.CATKIN: Topology.FLOWERING_REGION,
    InflorescenceKind.UMBEL: Topology.APEX_CLUSTER,
    InflorescenceKind.FLAT_UMBEL: Topology.APEX_CLUSTER,
    InflorescenceKind.PANICLE: Topology.BRANCHED,
    InflorescenceKind.CYME: Topology.RELAY,
    InflorescenceKind.SCORPIOID: Topology.RELAY,
    InflorescenceKind.HELICOID: Topology.RELAY,
    InflorescenceKind.DICHASIUM: Topology.RELAY,
    InflorescenceKind.PLEIOCHASIUM: Topology.RELAY,
}
"""The structural topology each kind defaults to -- the mapping the nine kinds
collapse to four (five with NONE), measured as golden output. A per-instance
`topology_override` can still change it; this is only the per-kind default."""


def kind_topology(kind: str) -> str:
    """The default topology for a kind (before any `topology_override`)."""
    return KIND_TOPOLOGY.get(kind, Topology.FLOWERING_REGION)


def kind_has_pedicels(kind: str) -> bool:
    """Whether flowers of this kind hang on pedicels that aim them. False for
    capped kinds (no pedicel) and the sessile flowering region (spike) -- so the
    Pedicel box and Flower Pitch are inert. A flower either takes its facing from
    a pedicel or states its own; there is no third case."""
    return kind not in CAPPED_KINDS and kind not in SESSILE_KINDS


def kind_gradients_apply(kind: str) -> bool:
    """Whether a flowering gradient (a curve sampled along the inflorescence)
    means anything. A terminal cluster has no along-axis position -- every flower
    sits at one point -- so its gradient curves are undefined, not merely
    ineffective. (Pedicel Width is the exception the caller keeps: it runs along
    a single stalk, not along the inflorescence.)"""
    return kind not in TERMINAL_CLUSTER_KINDS


class BloomOrder:
    """Which flowers open first. Implemented purely by offsetting each
    flower's own `birth_step`, so it rides on the organ-age model that already
    exists -- an older flower has simply had longer to expand."""

    ACROPETAL = "acropetal"      # bottom of the axis first (the common case)
    BASIPETAL = "basipetal"      # top first
    CENTRIPETAL = "centripetal"  # outer flowers first, working inward
    CENTRIFUGAL = "centrifugal"  # centre first, working outward


@dataclass
class Placement:
    """Which nodes along a parent axis may carry an organ, as a window in
    `factor` -- position along the axis, 0 at the base and 1 at the tip.

    This is a **morphological gate, not a density control.** It says where an
    organ is anatomically possible on this species; it does not say how many
    appear, which for branches is a resource question that belongs to the
    environment stage. Keeping the two separate is what lets a light model be
    added later without unpicking this one: the gate decides what is eligible,
    the environment decides what wins.

    `slide` shifts the whole window together, so a band can be moved up or down
    the axis without editing both ends. Unlike the parametric add-on's own slide -- a
    signed arc-length nudge on an already-placed attachment -- this one is in
    the same normalised factor units as start and end, because here it moves a
    *window over nodes* rather than an individual organ along a spline.

    The trunk/branch split is the same zoning `phyllotaxis_for` already uses:
    order 0 is the trunk, anything deeper is a branch. It covers the case of a
    perennial that carries leaves on its stems but not on its side shoots, and
    the reverse -- a bare trunk under a leafy crown.
    """

    start: float = 0.0
    end: float = 1.0
    slide: float = 0.0
    on_trunk: bool = True
    on_branches: bool = True
    levels: tuple[int, ...] | None = None
    """The branching levels (indices into `MorphologyProfile.levels`) whose
    axes may carry it; None is every level. For an inflorescence this is
    where the flower component is placed in the Plant Structure: an axis on
    any other level grows as it would on a plant with no flowers -- its tip
    stops or relays, it takes no flower spacing or orientation."""

    def allows(self, factor: float, order: int) -> bool:
        if not (self.on_trunk if order == 0 else self.on_branches):
            return False
        low, high = (self.start, self.end) if self.start <= self.end else (self.end, self.start)
        return (low + self.slide) <= factor <= (high + self.slide)


class OrientationMode:
    """**Which way** an inflorescence points -- the third of the three things a
    kind used to decide on its own, alongside `InflorescenceKind` (what it is)
    and `Attachment` (where it arises).

    Each mode names a **preferred direction**, and `orientation_strength` says
    how far the axis is pulled from the direction it would naturally leave its
    stem at, toward that direction. One meaning for that slider in every mode,
    which is the point of this shape: the previous design had a mode saying
    *how* to reach an angle and a separate `orientation_deg` saying *which*
    angle, where 0/90/180 meant up/horizontal/down -- two things to hold in
    your head, and the mode names described mechanisms rather than directions.

    **A preference, not a constraint.** Even at full strength an organ is not
    forced onto exact +Z or -Z: the pull is applied to the gravitropic setpoint
    the growth loop already steers toward, spread over a run of nodes, and the
    physics bend then acts on the result. Real organs are shaped by their
    support, their own growth direction, stiffness and load as well as by
    gravity, so an exact axis alignment would be the wrong answer even when it
    is what the slider nominally asks for.

    **This reverses a recorded decision, knowingly.** The previous note here
    argued for one number with the named directions as presets over it, since
    "ascending" and "descending" are points on one axis rather than separate
    behaviours. That is still true of Erect/Horizontal/Pendulous -- what
    changed is that in use the angle plus a mechanism-named mode read as
    neither one thing nor the other, and naming the directions is what people
    actually reach for. The angle survives, in `CUSTOM`, for the cases the
    three named points do not cover.
    """

    PARENT_ALIGNED = "parent_aligned"
    """Prefer to lie along the stem that carries it, whatever that stem is
    doing. Re-read every step, so the inflorescence swings when the branch
    bends -- which is what a stalk rigidly joined to a branch does."""
    ERECT = "erect"
    """Prefer to point up, against gravity. Negative gravitropism."""
    HORIZONTAL = "horizontal"
    """Prefer to lie perpendicular to gravity. "Perpendicular" is a whole
    plane, so the azimuth is the one the bud already had -- radially outward
    from its own stem, which keeps the phyllotaxis rather than sending every
    inflorescence to one compass point."""
    PENDULOUS = "pendulous"
    """Prefer to hang, with gravity. Positive gravitropism."""
    CUSTOM = "custom"
    """Prefer `orientation_deg` measured **from the parent's own direction**,
    so it is stated relative to the stem and swings with it. This is where the
    old free angle went."""


class Attachment:
    """**Where** an inflorescence arises. One of the three independent things a
    kind used to decide on its own.

    `InflorescenceKind` says what the inflorescence is, this says where it is
    borne, and `OrientationMode` says which way it points. Keeping them apart is
    what makes "a hanging axillary spike" sayable. A spike is a spike wherever
    it sits, and until this existed the kind silently meant "terminal, upright
    spike" because nothing else could say otherwise.

    That conflation is why the catkin took three rebuilds. Each attempt had to
    smuggle attachment and orientation into the kind, and each produced a plant
    whose *branch* drooped rather than the thing hanging off it.
    """

    TERMINAL = "terminal"
    """At the end of a stem: the axis reaches its node count and its own apex
    becomes the inflorescence. Every kind did this and nothing else."""
    AXILLARY = "axillary"
    """From a bud in a leaf axil, on its own short stalk, while the stem carries
    on being a stem and bears no flowers itself. Wisteria, laburnum, birch,
    hazel."""
    INTERNODAL = "internodal"
    """Between two nodes rather than at one. Carries `Bud.along_internode`."""
    LEAF_OPPOSED = "leaf_opposed"
    """Opposite a leaf at the same node, on the far side of the stem, rather
    than in its axil. Oppositifolious -- grapes and pepper. Mechanically it is
    the axillary path with the bud's azimuth turned half a turn, which is what
    "opposite" means; it is not a second release path."""
    CAULIFLOROUS = "cauliflorous"
    """Straight out of older wood on the TRUNK, well below the leaves.
    Cauliflory, as figs and cacao do. Released by age rather than position,
    because it breaks from wood laid down long before its own node was made."""
    RAMIFLOROUS = "ramiflorous"
    """The same old-wood break, but on the BRANCHES rather than the trunk --
    ramiflory, as redbud does along bare limbs. Shares every mechanism with
    `CAULIFLOROUS` and differs only in which axes it is allowed on, so the two
    are one code path filtered by order rather than two."""
    BASAL = "basal"
    """From the very bottom of the plant, at the root crown, rather than
    anywhere up a stem -- violets, many bulbs, and the scapes of a primrose.
    Released once from the lowest node of each first-order axis."""
    ADVENTITIOUS = "adventitious"
    """From tissue that had no meristem at all: scattered along old wood
    between the nodes, following no phyllotaxis. That is exactly what makes it
    adventitious, and it is why this is the one attachment that has to
    SYNTHESISE its buds rather than release dormant ones -- every other
    position releases a meristem the plant already made."""


@dataclass
class Inflorescence:
    kind: str = InflorescenceKind.SOLITARY
    laterals_per_node: int = 2
    """How many lateral axes **the inflorescence itself** starts at one of its
    nodes.

    For a cyme these are the relay shoots below its spent tip: one gives a
    monochasium, two a dichasium, three or more a pleiochasium. For a panicle
    they are the branches each node of the rachis puts out. Same quantity, and
    deliberately not two fields -- "how many axes does an inflorescence node
    start" is one question, and the standing rule here is that a second control
    for a quantity an existing one already carries is the same control twice.
    Ignored by the kinds that branch at neither level.

    **For a cyme this is now honoured for real.** Relays used to be taken from
    whatever axillary buds `_release_buds` had left on the node below the spent
    tip, and alternate phyllotaxis supplies exactly one -- so a dichasium was
    unreachable however this was set. Measured before the fix: 18 of 25 relay
    events spawned *zero* relays and the other 7 spawned one. The
    inflorescence places its own relays now, so the number asked for is the
    number built."""
    relay_rotation_deg: float = 90.0
    """How far the whole relay set is turned between one generation of a cyme
    and the next, in degrees around the parent axis.

    **This one number is what separates the classical cymose forms**, which is
    why it is a control and not four code paths. With one relay per fork, 180
    puts each successive relay on the opposite side and the inflorescence
    zigzags in a plane (a scorpioid cyme); 0 puts every relay on the same side
    and it coils (a helicoid cyme). With two, 90 makes successive forks
    decussate, which is what stops a dichasium collapsing into one plane.

    Measured in the parent's own **parallel-transported** frame, not against a
    world axis, so the rotations compose down the chain instead of each being
    re-measured from scratch -- the same property that lets phyllotaxis
    divergence angles compose into a real spiral. This is what makes a helicoid
    a genuine coil rather than a wobble.

    Ignored by every kind that does not relay."""
    branching_depth: int = 0
    """How many orders of branching happen **inside** this inflorescence,
    counted from its own base. **This is the compound control**, and it is the
    same number for every kind that has a compound form.

    0 is simple, and is the default: a raceme, a spike, an umbel, a corymb. 1
    makes the inflorescence carry, at each of the places it would have put a
    flower, a whole inflorescence of its own kind instead -- which is exactly
    what a compound form is. 2 gives the triple forms real *Apiaceae* produce.

    | kind at depth 0 | at depth 1 |
    | --------------- | ---------- |
    | raceme          | panicle (a raceme of racemes) |
    | umbel           | compound umbel (carrot, cow parsley, fennel) |
    | corymb          | compound corymb (yarrow, elder) |
    | spike           | compound spike (wheat, rye -- spikelets on a rachis) |
    | cyme            | one relay generation |

    **The recursion is free and always was**: a branch is an ordinary axis that
    reaches its own flowering target and arrives back at `_begin_inflorescence`
    at its own order, so nothing here iterates or special-cases a depth. What
    each depth *does* is decided by the kind's own topology -- a flowering
    region hangs its branches along a rachis, an apex cluster hangs them off one
    point as rays. That is why one control serves every compound form rather
    than each one needing its own.

    For a **cyme** it is how many times the inflorescence relays: each
    generation multiplies the tip count by `laterals_per_node`, so a scorpioid
    chain wants a large value and a dichasium a small one. It used to be
    `max_branch_order` that bounded this, which meant a cyme could relay twice
    on a default profile.

    **The default was 1 while `PANICLE` was the only reader**, which stopped
    being safe the moment every kind read it -- a raceme would have come out a
    panicle. `blender.properties._KIND_DEFAULTS` stamps a depth for the kinds
    whose *name* implies one (Panicle, and the five cymose forms); everything
    else is simple until asked.

    Completely independent of `max_branch_order`, which stays purely
    vegetative. See `Axis.structural_depth` for why they cannot be one
    number: a tree with panicles needs five orders of wood and two of
    inflorescence, and keyed on branch order a panicle on a twig can only be a
    single order deep, which is a raceme."""
    branch_count: int = 7
    """How many **branches** the rachis carries -- nodes in its branching
    region, each bearing `laterals_per_node` of them.

    Its own number, and until it existed `flower_count` silently served both
    levels: the rachis's branch count *and* each branchlet's flower count were
    the same field, so "ten branches of three flowers" was not expressible. The
    two are different facts about a panicle and a real one rarely has them
    equal."""
    branch_count_variation: float = 0.0
    """Random change to the branch count, as a fraction of it. Drawn per axis,
    so on a compound panicle one primary branch carries more secondaries than
    its neighbour."""
    branch_angle_variation: float = 0.0
    branch_angle_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """The launch angle along the rachis -- a panicle whose lower branches
    stand out wide and whose upper ones hug the stem, which is most of them.

    Read at the branch's own position along its parent, the same coordinate
    every other flowering gradient uses."""
    branch_nodes_variation: float = 0.0
    """Random change to a branch's node count, as a fraction of it. Folded into
    `Axis.inflorescence_scale` at birth alongside the acropetal gradient, since
    both say the same thing -- how big is this particular branch."""
    branch_angle_deg: float = 45.0
    """Which way an inflorescence branch leaves the rachis, **measured from
    perpendicular and signed**: 0 straight out sideways, +90 flat up the
    rachis, -90 flat down.

    The same convention as `pedicel_angle_deg` and for the same reason. A leaf
    or a woody branch is read off a plant as an angle *from the stem* and never
    points back down it, so 0..90 from-the-axis covers every real case. A
    panicle branch does sweep below horizontal -- that is what a drooping,
    open panicle is -- and from-the-axis cannot say so without running past 90,
    where the number stops meaning what it says. Measuring from perpendicular
    puts the sign where the eye already sees one.

    Its own field rather than the profile's `branch_angle_deg`, and the same
    tree argument applies: an oat panicle spreads its branches wide off a stem
    whose own vegetative branches leave at a quite different angle, and on a
    tree the two are not remotely related. This is also the main thing
    separating a diffuse panicle from a contracted, spike-like one.

    The default is a true no-op across the re-parameterisation: 45 from the
    axis and 45 from perpendicular are the same direction, since
    `_elevation_from_axis` is `90 - x`."""
    branch_nodes: int = 2
    """Nodes an inflorescence branch lays down before it starts flowering --
    the bare basal stretch of a panicle branch.

    Not `nodes_before_flowering_at(order)`, which is the *plant's* vegetative
    formula and on a tree twig would give an inflorescence branch a woody
    stem's worth of nodes."""
    branch_spacing: float = 1.5
    """Spacing between successive **branches** along the rachis, relative to a
    vegetative internode -- the branching region's counterpart to
    `internode_scale`.

    Its own field because the two are genuinely different numbers on the same
    plant: a panicle's primary branches stand well apart on the rachis while
    the flowers on its branchlets are crowded, and until this existed both read
    `internode_scale`, so one could not be set without the other. Defaults
    above 1.0 rather than below it for the same reason -- branches need room
    where flowers do not."""
    branch_spacing_variation: float = 0.0
    branch_spacing_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Spacing along the rachis, so a panicle can open out at the base and
    crowd toward its tip."""
    branch_length: float = 1.0
    """Internode length **within** an inflorescence branch, relative to a
    vegetative internode.

    Orthogonal to `branch_nodes`, which is how many flowers or sub-branches the
    branch carries: this is how far apart they sit, and so how physically long
    the branch is. Before it existed a branch could only be lengthened by
    making it carry more, which are two different things about a plant."""
    branch_length_variation: float = 0.0
    branch_length_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Length along the rachis, so a panicle's lower branches can be longer.

    Read at the branch's own position along its parent, the same coordinate
    every other flowering gradient uses."""
    branch_width: float = 0.6
    """A branch's radius where it leaves the rachis, as a fraction of the
    rachis's own radius there -- **exactly `pedicel_width`'s convention**, so
    1.0 means "as thick as the stem it comes off".

    **Capped at 1.0**: a branch is never thicker than what carries it. Nothing
    in the pipe model forbade it, and a stem narrower than its own side branch
    reads as broken rather than as a style.

    It was a multiplier on whatever the pipe model gave the branch, which meant
    1.0 named no particular thickness and had no relationship to the parent at
    all. Reported as "a branch width of 1 does not equal parent width", which
    was true.

    The cost, stated because it is real: the pipe model no longer decides an
    inflorescence branch's thickness, so a branch carrying more flowers is no
    longer automatically stouter than one carrying fewer. A panicle's branches
    are a stated trait here rather than a derived one -- the same call the
    pedicel already makes. Still applied before `physics.apply_gravity`, so
    bending sees it: curvature goes as `1 / radius ** 4` and a finer branch
    arches more."""
    branch_width_variation: float = 0.0
    branch_width_profile: Curve = field(
        default_factory=lambda: Curve.from_points([(0.0, 1.0), (1.0, 0.55)]))
    """Thickness along **one branch**, base to tip -- a taper, and the second
    odd X out among these gradients alongside `pedicel_width_profile`, which it
    now matches exactly.

    It read *across* the rachis before, so it set how thick the bottom branch
    was against the top one and no branch tapered at all. Reported as "it
    doesn't control the taper but instead the width from bottom branch to the
    top branch", which is precisely what it did.

    **Defaults to a real taper rather than a flat 1.0**, for the same reason
    `pedicel_width_profile` does: a flat default reinstates the uniform tube
    this exists to remove.

    It shapes the taper and nothing else -- the base width is `branch_width`'s
    to state, so the curve is normalised by its own value at 0 before being
    applied. Without that, dragging the curve's left end down thinned the whole
    branch, which is the scalar's job and not the graph's."""
    branch_bend: float = 0.0
    """A constant arch held in the branch's **own** plane, spread evenly over
    its length -- the branch analogue of `pedicel_bend`, and distinct from
    `branch_droop` in exactly the same way: this is shape, and it turns with the
    branch rather than being measured against gravity. 0 is straight; 1 is a
    quarter turn from base to tip."""
    branch_bend_variation: float = 0.0
    branch_bend_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Arch along the rachis.

    Read at the branch's own position along its parent, the same coordinate
    every other flowering gradient uses."""
    branch_droop: float = 0.0
    """Extra give under gravity for inflorescence branches only, added to
    `flowering_droop`.

    **Scoped, and that scope is the whole justification** -- the standing rule
    here is that a plain multiplier on a quantity an existing control already
    scales is the same control twice. `flowering_droop` finds the flowering
    stretch of *every* axis, rachis included, so it cannot arch a panicle's
    branches while leaving its main stem upright. This can, which is the
    measurement that makes it a second control rather than a second dial."""
    branch_droop_variation: float = 0.0
    branch_droop_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Sag along the rachis.

    Read at the branch's own position along its parent, the same coordinate
    every other flowering gradient uses."""
    branch_size_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """How big an inflorescence branch is, by where it sits along its parent.

    This is the panicle's pyramid: the lowest branches are the longest and the
    most divided, the uppermost are short or reduced to a single flower. One
    curve gives both, because it scales the branch's node count -- longer *is*
    more divided when each node carries a branch or a flower.

    Read at the branch's position along its parent, exactly as `lateral_vigour`
    already is for vegetative branches through `_tonic_factor`. Flat at 1.0 by
    default, so an inflorescence that does not branch never notices it."""
    flower_count: int = 7
    """Flowers in the umbel, or along the raceme. Ignored by Solitary."""
    whorl_count: int = 0
    """How many flowers each node of the flowering region carries, in a ring
    around the stem. **0 means "however many the axis's own phyllotaxis says"**,
    which is one flower per leaf position and is what every kind did before this
    existed -- so it is an exact no-op rather than a value that happens to
    reproduce one.

    Its own field rather than the plant's phyllotaxis, because measurement
    showed the two genuinely cannot be one number: `_add_phytomer` places one
    flower per phyllotaxis slot, so a six-flowered whorl asked for through
    the leaf arrangement also gives the vegetative stem below six *leaves* per
    node. A mint has two. See `InflorescenceKind.VERTICILLASTER`.

    The ring is built by `Phyllotaxis.WHORLED`, so successive whorls are offset
    by half a step for free -- stacked rings in perfect register read as a
    mistake rather than as a plant."""
    bract_size: float = 0.0
    """The leafy bract subtending each whorl, as a fraction of the foliage leaf
    this node would otherwise have carried. **0 means no bracts**, which is what
    every kind did before, so this is an exact no-op too.

    A fraction rather than an absolute size: how much smaller than a leaf a
    bract is, is the botanical fact -- full-sized on *Lamium*, scale-like on
    *Salvia* -- and stating it that way means the bracts follow
    `leaf_size_profile` up the stem instead of needing a gradient of their own.

    Bracts appear at every flowering node they can (placement is now a
    render-time component mask, not a growth gate). *Which mesh* they use is
    `Organ.form == 2`.

    Bracts follow the **axis's own** phyllotaxis, not `whorl_count`: a Lamiaceae
    node has two opposite bracts with many flowers in their axils, and that
    falls straight out of the two arrangements being separate."""
    flower_count_variation: float = 0.0
    """Random change to the flower count, as a fraction of it. Drawn **per
    axis**, not per flower -- the count belongs to one inflorescence, so on a
    branched plant this is what stops every raceme being the same length."""
    pedicel_length: float = 0.03
    pedicel_variation: float = 0.2
    pedicel_angle_deg: float = 30.0
    """How far a pedicel tilts off perpendicular, signed: 0 leaves the axis
    straight out sideways, +90 presses it flat up along the axis, -90 flat
    down. This is `elevation_deg` directly, unlike `leaf_angle_deg` and
    `branch_angle_deg`, which are stated from the axis and complemented by
    `_elevation_from_axis`.

    The odd one out on purpose. A leaf or a branch is read off a plant as an
    angle *from the stem* and never points back down it, so from-the-axis is
    how a person states them and 0..90 covers every real case. A pedicel does
    go down -- a pendulous raceme's stalks hang below horizontal -- and
    from-the-axis has no way to say that: it would need to run past 90, where
    the number stops meaning what it says. Measuring from perpendicular puts
    the sign where the eye already sees one, with 0 at the horizontal the two
    cases divide at.

    Terminal-cluster kinds ignore this entirely: umbels and flat-topped
    heads write their own `elevation_deg` per flower from `cluster_spread_deg`
    or from the flat-top solve, since a dome's rim and its centre cannot share
    one angle."""
    cluster_spread_deg: float = 150.0
    """Full opening angle of an umbel, so 180 is a flat hemisphere, 360 a
    complete sphere of flowers, and
    small values a tight cone. Flowers fill the whole cap by equal solid angle,
    which is what puts flowers in the *centre* as well as the rim. Placing them
    all at one polar angle gives a ring with a hole in it, which is the shape
    an umbel is most often drawn as and least often is."""
    head_radius: float = 0.045
    """Radius of a flat-topped head (corymb, flat umbel). The flowers pack
    evenly inside it, so this is what makes the head compact rather than a
    wide sparse fan.

    Before this existed, a flat-topped head had no width control at all: every
    pedicel left the axis at one fixed angle, so a flower that had to climb h
    landed at radius h*tan(angle) -- at the 55 degrees off the axis that was
    then the default that is 1.4x the climb, and a corymb whose flowering
    region spans 0.2 m of axis threw its lowest flowers 0.29 m out. Wide and
    sparse by construction."""
    head_dome: float = 0.0
    """How far a flat-topped head's rim sits below its centre, as a fraction of
    `head_radius`. 0 is dead flat -- exactly what a corymb did before this
    existed. 1 drops the rim by a full head radius, which over a disc of that
    radius is a hemisphere; negative lifts the rim instead, cupping the head.

    **The one thing a flat-topped kind had no control over.** Measured on the
    generic perennial, a corymb's `pedicel_length`, `pedicel_variation` and
    `pedicel_length_profile` all produce byte-identical output -- the solve
    assigns each length outright from the geometry, so all three are inert and
    the panel rightly hides them. `head_radius` owned the head's *width* and
    nothing owned its *curvature*, so a corymb was flat to 2.07 mm across an
    85.6 mm head at every setting. This is not a second dial on a quantity
    something else already scales; it is the first one on a quantity nothing
    did.

    Botanically a corymb spans flat-topped to convex and stays a corymb, which
    is why this is a control on the kind rather than a kind of its own. An
    umbel's rounded counterpart *is* a separate kind (Umbel against Flat
    Umbel), because there the difference is structural."""
    head_dome_variation: float = 0.0
    """Random height off the head's surface per flower, as a fraction of
    `head_radius`.

    A fraction of the head rather than of `head_dome`, deliberately: the
    commonest want is a flat head that is not machined flat, and a variance
    scaled by the dome would be inert at exactly the setting where roughness
    matters most."""
    head_dome_profile: Curve = field(default_factory=default_head_dome_profile)
    """The head's cross-section, centre to rim -- see
    `curves.default_head_dome_profile` for why this one is not a flat 1.0.

    Read at the flower's own distance from the centre as a fraction of
    `head_radius`, so the graph is the silhouette. That is a different X from
    every other flowering curve, which run along the inflorescence; the panel
    captions it apart for the same reason `pedicel_width_profile` is captioned
    apart."""
    pedicel_width: float = 0.3
    """A stalk's radius where it leaves the axis, as a fraction of the axis's
    own radius there. A real and quite variable trait: some stalks continue the
    stem's thickness and read as short branches, most are conspicuously finer
    than the stem carrying them.

    It was a fixed 0.55 in the bridge, with a comment claiming it had "no
    botanical measurement behind it". That was wrong -- the ratio of a pedicel
    to the axis it hangs from is exactly the sort of thing you read off a plant
    -- and 0.55 made every stalk read as a branch. The pipe model still sizes
    the axis; this only says how much finer the stalk is."""
    pedicel_bend: float = 0.7
    """How a pedicel curves on its way to the flower, signed: **0 is a straight
    spoke, positive sweeps up, negative sags down.** One axis, because that is
    how it reads off a plant -- a stalk either lifts toward its flower or hangs
    under it -- but the two halves are not the same operation and knowing which
    is which matters when tuning:

    * **Positive** sets the exponent in `pedicel_polyline`'s `height ~ radius
      ** n`, an umbrella rib turning the whole way rather than bending once and
      running straight. It is a *shape* in the pedicel's own plane, so it turns
      with the axis, and it **does not move the flower** -- the endpoint stays
      wherever `pedicel_length` and `pedicel_angle_deg` put it. Straight spokes
      are the single thing that most makes a flat head read as diagram rather
      than plant.
    * **Negative** rotates the stalk toward world down under the flower's own
      weight, so it knows nothing about the axis and it **does move the
      flower**, which is what makes one read as hanging rather than as styled.
      Scaled by the flower's mass, so a heavier flower hangs further -- see
      `physics.apply_pedicel_droop`.

    The asymmetry is inherent rather than an oversight: bend has to preserve
    the endpoint or `pedicel_length` stops meaning anything, and droop has to
    move it or nothing is hanging.

    Sign follows `pedicel_angle_deg`, where negative already hangs down and
    positive lifts up -- and that also made the merge free of migration, since
    every value that existed was positive and kept its exact meaning."""
    internode_scale: float = 0.5
    """Spacing between successive flowers along the axis, relative to a
    vegetative internode. Inflorescence axes are normally more crowded than the
    leafy stem below them. Meaningless for `is_terminal_cluster` kinds, whose
    flowers all hang off one apex."""
    bloom_order: str = BloomOrder.ACROPETAL

    # -- Gradients along the flowering region ------------------------------
    # Every one is a multiplier read at a flower's own normalised position in
    # the flowering region -- 0 at the first flower, 1 at the last -- so a flat
    # curve at 1.0 is exactly the behaviour before these existed. Together they
    # are what separates two racemes that share a topology and nothing else:
    # The parametric add-on carries the same idea as `pedicel_length_profile`.
    pedicel_length_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Stalk length along the axis. Lower flowers are older and generally
    longer-stalked; a steep enough gradient here *is* a corymb, which is why
    this belongs on the inflorescence rather than on one kind."""
    flower_size_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Flower size along the axis. Lower flowers opened first and are larger;
    the top of a raceme is a point of tiny undeveloped buds."""
    young_size: float = 0.2
    """How developed an inflorescence is the moment it is laid down, as a
    share of mature: its flower spacing, flower size and pedicel length and
    width all start here and grow to full as its nodes elongate. 0.06 -- the
    raw elongation curve -- buried a late raceme's buds inside its branch tip;
    this is the user's own control over how young a young one looks. One
    number for the start; `young_growth_profile` is the path from it."""
    young_growth_profile: Curve = field(
        default_factory=lambda: Curve.from_points([(0.0, 0.0), (1.0, 1.0)]))
    """The path from `young_size` to full, read at the inflorescence's own
    development (0 just laid down -> 1 mature, the elongation fraction of its
    nodes): the share of the way from Young Size to full size it has come.
    Not a multiplier on anything -- the curve is the progress, so a straight
    line grows it evenly and a curve that rises fast fills a raceme out early.
    A fully developed inflorescence is full size whatever the curve says --
    see `development_size`."""

    def development_size(self, fraction: float) -> float:
        """The size share (spacing, flower, stalk) of an inflorescence whose
        nodes have elongated `fraction` of the way. Mature (>= 1) is exactly
        1.0, so no curve or Young Size can touch a finished inflorescence."""
        if fraction >= 1.0:
            return 1.0
        start = min(1.0, max(0.0, float(self.young_size)))
        progress = min(1.0, max(0.0, float(self.young_growth_profile(max(0.0, fraction)))))
        return start + (1.0 - start) * progress
    flower_spacing_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Internode spacing between successive flowers, on top of
    `internode_scale`. Rising means crowded at the base and open at the tip;
    falling means the reverse. Both occur, and it is one of the strongest cues
    separating one raceme from another."""
    pedicel_angle_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Stalk tilt along the axis. Distinct from length and doing different
    work: lower stalks commonly stand out near horizontal while upper ones hug
    the spire, which narrows the whole silhouette toward the tip without
    changing a single stalk length.

    A multiplier on the signed `pedicel_angle_deg`, so 0 is perpendicular and
    1 is the stated tilt -- which means it scales how far from horizontal the
    stalks sit and cannot flip their sense partway up. A raceme that droops at
    the base and lifts at the tip is not expressible here; nothing observed
    needs it, and the fix would be a second signed curve rather than a
    reinterpretation of this one."""
    pedicel_bend_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Stalk curvature along the axis, as a multiplier on the signed
    `pedicel_bend`. A real gradient in its own right: the lower stalks of a
    corymb sweep hard to bring their flowers up to the head's level while the
    upper ones barely need to, and on a drooping raceme the heavy open flowers
    below hang while the buds above stand out straight.

    A multiplier on a signed value, so it carries exactly the same caveat as
    `pedicel_angle_profile`: it scales the magnitude and cannot flip an
    up-sweep into a sag partway along. Zero is straight, which is the honest
    midpoint between them and the only crossing this can express."""

    # -- Presentation ------------------------------------------------------
    spiral_deg: float = 0.0
    """The flowering region's own spiral: each flowering node's flowers turned
    this many degrees round the axis from the last node's (137.5 the golden
    angle). **0 follows the axis's own phyllotaxis** -- the leaves' -- exactly
    as before this existed. A grass needs the two apart: its leaves are
    two-ranked (180) and its spikelets sit all round the rachis. `whorl_count`
    still wins when it asks for a ring (2 or more); bracts follow the leaves
    either way (`_add_bracts`). (User request 2026-10-18: an Alopecurus spike
    came out in two flat ranks, inheriting the leaves' 180.)"""
    flower_spread_deg: float = 360.0
    """The arc of stem the flowers occupy, in degrees. 360 is the full circle
    the phyllotaxis already gives; 90 crowds them into a quarter turn; 0 puts
    every one on a single line up one side.

    Stated as the arc rather than as a 0..1 "one-sidedness" because the arc is
    what it literally produces and what a person can see: each flower's signed
    divergence spans -180..180 and is scaled by `spread/360`, so the flowers
    end up spread over exactly `spread` degrees. The 0..1 form said the same
    thing backwards, with 1 meaning "no spread at all".

    This is a real and separate mechanism from phyllotaxis, and the reason a
    foxglove is recognisable: its buds are arranged spirally like any other
    raceme, and the *pedicels twist* so the open flowers all present one way.
    Unilateral phyllotaxis produces a similar look by moving the attachment
    points themselves, which is a different plant.

    Implemented as a scaling of each flower's divergence toward a shared
    azimuth, which moves the attachment as well as the facing -- an
    approximation of the twist, visually right and far cheaper than carrying a
    second frame per organ."""
    flower_spread_variation: float = 0.0
    """Random spread of each flower's own azimuth, in degrees either way. Kept
    separate from `flower_spread_deg`: that one narrows the arc every flower is
    placed on, this one blurs the placement within it, and a real one-sided
    raceme wants both."""
    flower_pitch_deg: float = 0.0
    """The flower's own tilt relative to the stalk it sits on, in degrees.
    Negative nods it downward.

    Independent of both stalk length and stalk angle, and the third thing that
    has to be right: a foxglove's flowers hang from near-horizontal stalks, a
    delphinium's face outward from stalks at the same angle, a mullein's face
    outward and up. Same architecture, three different plants."""
    flower_pitch_variation: float = 0.0
    """Random tilt on each flower, in degrees either way."""
    heliotropism: float = 0.0
    """Floral heliotropism, 0-1: how far each of this inflorescence's flowers
    turns its face toward the sun, as a post-growth re-aim. The
    nodding-sunflower case -- the stem stays where gravity left it, the head
    turns up to the sun, and only the flower's own stalk bends with it
    (`development.apply_heliotropism`). The sun is its flower component's Sun
    Object (no object, no effect); the strength is the inflorescence's, so one
    flower component can track while another does not. Distinct from
    `light_seeking`, which turns growing shoots toward a light gradient rather
    than re-aiming the organs on them. Zero is off and byte-identical. Was the
    plant-wide `flower_heliotropism` until schema 11."""
    flower_pitch_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Flower facing along the flowering region, as a multiplier on the signed
    `flower_pitch_deg`. Same shape and same limitation as
    `pedicel_angle_profile`: it scales distance from horizontal, so it cannot
    flip a flower from nodding to upturned partway up. A foxglove wants exactly
    this -- its lowest bells hang hardest and the youngest at the tip barely
    nod at all."""
    flower_size_variation: float = 0.0
    """Random size on each flower, as a fraction of its own size. Applied on
    top of `flower_size_profile`, so a graded inflorescence stays graded and
    only gains scatter."""
    pedicel_angle_variation: float = 0.0
    """Random stalk tilt, in degrees either way."""
    pedicel_bend_variation: float = 0.0
    """Random stalk curvature, as a fraction of `pedicel_bend`."""
    pedicel_width_variation: float = 0.0
    """Random stalk thickness, as a fraction of `pedicel_width`."""
    pedicel_width_profile: Curve = field(
        default_factory=lambda: Curve.from_points([(0.0, 1.0), (1.0, 0.55)]))
    """Thickness along **one stalk**, base to flower -- the odd one out among
    the flowering curves, which are all read across the inflorescence instead.
    It gets its own X caption in the panel for exactly that reason.

    It is read here because this is where the interesting variation is: a stalk
    narrows visibly from the stem to the flower, and until this existed it did
    not narrow at all. Every pedicel vertex took one width, so a stalk was a
    uniform tube; what looked like a taper was an artifact of vertex *count*
    (at Pedicel Bend 0 there were only two, the far one carrying the stalk width
    and the near one carrying the stem's, so Curve to Mesh interpolated
    between them). Raising the bend added vertices all at one width and the
    taper vanished -- reported as "adjusting the bend makes the width uniform",
    which is precisely what happened.

    Defaults to a real taper rather than a flat 1.0, because a flat default
    would reinstate the uniform tube."""
    flower_spacing_variation: float = 0.0
    """Random spacing between successive flowers, as a fraction of it. Drawn
    per internode rather than per axis, so one raceme's own flowers sit
    unevenly instead of every raceme being evenly spaced at its own pitch."""

    # The bloom *timeline* lives entirely on the Blender side (Progress and
    # Spread are modifier sockets, so scrubbing them never regrows the plant).
    # What growth owns is only the bloom *order* -- `bloom_order` above, and the
    # per-flower `bloom_start` it produces.

    attachments: frozenset = field(
        default_factory=lambda: frozenset({Attachment.TERMINAL}))
    """Where this inflorescence arises -- a SET of `Attachment` values, so one
    plant can bear flowers in several places at once.

    Plural because plenty of plants are: a mint carries a terminal spike *and*
    axillary whorls below it, and until this was a set the two were mutually
    exclusive for no reason other than the field's type. Every member is
    evaluated independently, so `{TERMINAL, AXILLARY}` gives both, and the
    controls each position needs (Wood Age, Along Internode) apply to whichever
    members are present.

    An EMPTY set is legal and means no inflorescences anywhere -- the same
    plant `InflorescenceKind.NONE` gives. That overlap is a degenerate state of
    a multi-select rather than a second control: nothing here *offers* two ways
    to say it, you can just also get there by unticking everything.

    `TERMINAL` is what every kind did before this existed, so it is the default
    and every saved plant is unchanged by it. `AXILLARY` releases a short
    *inflorescence* axis at each node in `placement`'s window -- leafless,
    outside the plant's branching budget, `branch_nodes` bare nodes for a
    peduncle and then the flowering region -- and leaves the bearing axis to
    mature and stop without flowering.

    That last part is what a lateral *vegetative* axis could not give: it is a
    branch, with stem radius, internodes and the pipe model, so calling its base
    a peduncle does not make it one.

    Composes with `orientation_mode`, which is what makes the released
    inflorescence hang rather than stand up. Wisteria is this plus that."""
    cauline_min_age: int = 8
    """How many developmental steps a node must have carried before a
    `CAULIFLOROUS`, `RAMIFLOROUS` or `ADVENTITIOUS` inflorescence can break
    from it. All three arise from wood laid down seasons earlier -- a fig
    fruits on old limbs, not on this year's shoot tips -- so age, not position,
    is what defines them.

    Ignored by every other attachment, all of which arise on the node they are
    made at."""
    attachment_per_node: int = 1
    """How many inflorescences arise at each position the attachment picks.

    The answer to "terminal cluster" and "axillary cluster", which are not
    positions but a COUNT applied to one -- folding them into `Attachment`
    would have meant a `..._CLUSTER` twin for every position ever added, and
    would put back exactly the conflation that enum exists to undo. As one
    field it clusters *any* attachment, including the ones a fixed pair of enum
    entries could never have covered.

    Note this is a different question from `laterals_per_node`, which is how
    many axes the inflorescence starts at its OWN nodes once it exists. This is
    how many inflorescences the bearing plant starts at one of ITS nodes.

    Above 1 the extra meristems are synthesised, spaced evenly around the node
    -- a node only carries as many real buds as its phyllotaxis gave it, so
    there is nothing already there to release."""
    internode_position: float = 0.5
    """Where along its internode an `INTERNODAL` inflorescence sits, 0 at the
    node below and 1 at the node above. Ignored by every other attachment,
    which sit at nodes by definition."""
    placement: Placement = field(
        default_factory=lambda: Placement(on_trunk=False, start=0.5, end=1.0))
    """Which nodes bear this inflorescence, when `attachment` is not
    `TERMINAL`. Ignored for a terminal one, which arises where the axis ends
    and has no window to sit in.

    Its own `Placement` rather than the leaves', which a bract shares: a bract
    sits *where its flowers already are*, so it needs no window of its own,
    while where an axillary inflorescence occurs is the entire structural
    statement -- hazel bears catkins near the tips of last year's side twigs,
    not wherever it bears foliage. The default says exactly that."""
    orientation_mode: str = OrientationMode.PARENT_ALIGNED
    """Which direction this inflorescence prefers -- see `OrientationMode`.

    `PARENT_ALIGNED` at strength 0 is what every axis did before orientation
    existed: hold the angle it launched at."""
    orientation_strength: float = 0.0
    """How far the axis is pulled from the direction it would naturally leave
    its stem at, toward `orientation_mode`'s preferred direction. 0 keeps the
    launch direction untouched, 1 pulls fully onto the preference.

    **It also sets how sharply the turn happens, and that is deliberate rather
    than a second job.** The turn is always spread over a fixed run of nodes,
    so a bigger pull covers more degrees over the same span and therefore bends
    harder -- higher is sharper, and no value produces a hard corner. That is
    what retired the separate Turn Arc control: sharpness was never independent
    of how far the thing had to turn, and having both invited setting a large
    angle with a tight arc, which is exactly the 140-degrees-in-one-node corner
    Turn Arc was added to fix.

    Distinct from `gravitropism`, which is how tightly an axis tracks whatever
    setpoint it has. This decides *where the setpoint is*; that decides how
    hard it is held. A soft correction on top of a moving setpoint is an
    unbounded second arc, which is what the removed Steer To Angle was."""
    orientation_deg: float = 0.0
    """`CUSTOM`'s angle, measured **from the parent's own direction** rather
    than from world vertical, so it is stated relative to the stem and swings
    with it when the branch bends.

    Ignored by the four named directions, which state their own preference.
    This is where the old free angle went -- the named points cover the common
    cases and this covers the rest."""
    orientation_load_driven: bool = False
    """Reach the preferred direction as the axis FILLS WITH FLOWERS, rather
    than over a run of nodes at its base. Erect while it is a bud, bowed over
    once laden.

    Kept because it is the one thing post-growth bending genuinely cannot do:
    it moves where the flowers are PLACED as the axis grows, where physics acts
    once on a finished shape. Built first as "turn the correction off and let
    weight decide", which measured **under one degree** different from holding
    the launch angle at every flower weight -- the bend runs regardless of
    mode, so switching correction off changes almost nothing.

    A schedule rather than a direction, which is why it is a flag beside the
    mode and not a sixth mode."""
    flower_size: float | None = None
    """This inflorescence's flowers' size; None is the plant's `flower_size`.
    Set on a further inflorescence (`MorphologyProfile.extra_inflorescences`),
    whose flower component's Size it is, as the primary's is the plant's."""
    maps: list[ParamMap] = field(default_factory=list)
    """This inflorescence's growth maps (`growth.maps.FLOWER_TARGETS`): its
    flower component's per-flower / per-branch maps. Empty: byte-identical."""

    def orientation(self) -> tuple[float | None, float | None]:
        """`(setpoint_tilt_deg, gravitropism)` for an axis carrying this
        inflorescence -- `None` for either meaning "leave it alone".

        The one place the modes turn into numbers, so `development` asks once
        and does not branch on the mode itself.
        """
        if self.orientation_strength <= 0.0:
            return None, None
        # The setpoint itself moves every node -- eased over a run at the base,
        # or driven by flower load -- so `development` owns the value and this
        # only sets how tightly the axis tracks it. Tracking hard is right
        # *because* the setpoint is what curves.
        return None, 1.0

    topology_override: str = ""
    """Force a `Topology` regardless of `kind`, for the forms no kind names.

    **Corrected 2026-08-19.** This used to claim that `RELAY` plus a positive
    `veg.decline` gives a panicle. It does not, and the claim was never
    measured: `_relay_cyme` calls `_cap_with_flower` unconditionally, so the
    apex flowers and is spent -- which is a cyme, and a panicle's main axis is
    indeterminate and never flowers. It also never reads `decline` at all. A
    panicle is `InflorescenceKind.PANICLE` / `Topology.BRANCHED`, which is a
    real fourth structure rather than a re-use of this one.

    What is left here is a genuine escape hatch for a form no kind names, and
    the remaining example is the **thyrse** -- an indeterminate main axis with
    cymose laterals -- though that is better said as PANICLE on the trunk with
    CYME as the `lateral_inflorescence`.

    Left empty by every shipped species, so `topology` reads from `kind` as it
    always did."""

    @property
    def is_terminal_cluster(self) -> bool:
        """True for kinds whose flowers all attach at the apex, false for those
        that build a flowering region of successive nodes. See
        `TERMINAL_CLUSTER_KINDS`."""
        return self.kind in TERMINAL_CLUSTER_KINDS

    @property
    def is_sessile(self) -> bool:
        """No pedicels at all -- the flowers sit directly on the axis. The model
        sense (spike): distinct from a capped kind, which has no flowering region
        at all. See `SESSILE_KINDS`."""
        return self.kind in SESSILE_KINDS

    @property
    def is_flat_topped(self) -> bool:
        """Pedicel lengths are solved so every flower lands on one plane, rather
        than scaled by a fixed ratio (a ratio cannot flatten a corymb, whose axis
        is far longer than any plausible pedicel). See `FLAT_TOPPED_KINDS`."""
        return self.kind in FLAT_TOPPED_KINDS

    @property
    def is_relayed(self) -> bool:
        """True when the axis is spent by its own terminal flower and the
        inflorescence continues from buds below it -- the cyme's defining
        behaviour. Read off `topology`, so a `topology_override` counts."""
        return self.topology == Topology.RELAY

    @property
    def topology(self) -> str:
        """What this kind does structurally, as opposed to how it looks -- the
        four (five with NONE) topologies the kinds collapse to. Read from
        `KIND_TOPOLOGY`, unless a per-instance `topology_override` names one
        directly (which is how a topology no kind names becomes reachable).
        """
        return self.topology_override or kind_topology(self.kind)

    @property
    def extra_nodes(self) -> int:
        """How many more nodes the axis builds after it starts flowering."""
        if (self.kind in (InflorescenceKind.SOLITARY, InflorescenceKind.NONE,
                          InflorescenceKind.CATKIN)
                or self.is_terminal_cluster or self.is_relayed):
            return 0
        return max(1, self.flower_count)

    @property
    def span_nodes(self) -> int:
        """The longest region this inflorescence can ask for, for sizing the
        plant's developmental span.

        A panicle has **two** region lengths -- `branch_count` on its rachis and
        `flower_count` on its branchlets -- and `total_steps` has to hold
        whichever is bigger or the longer one is silently truncated, which is
        the exact failure `total_steps` was written to fix in the first place.
        A cyme has the same problem one level down: it lays `branch_nodes`
        nodes per relay generation and relays `branching_depth` times, and none
        of that is in `extra_nodes` (which is 0 for a relayed kind, since the
        *starting* axis adds no flowering nodes of its own). Without this a
        deep scorpioid silently ran out of span partway down its own chain --
        the same silent truncation this property was written to stop.

        Identical to `extra_nodes` for every kind that does not branch."""
        if self.topology == Topology.RELAY:
            return max(self.extra_nodes,
                       max(0, self.branching_depth) * max(1, self.branch_nodes))
        if self.topology != Topology.BRANCHED:
            return self.extra_nodes
        return max(self.extra_nodes, max(1, self.branch_count))

    def clock_nodes(self, elongation_steps: int) -> int:
        """The steps this inflorescence needs on the plant's clock after its
        axis is grown -- what `MorphologyProfile.total_steps` reserves for it.

        Not `span_nodes`: a flowering region is laid down whole when it opens
        (`development._preform_region`), its nodes staggered over the
        elongation steps the clock already holds, so reserving a step per
        flower was dead season -- an Alopecurus spike of 200 spikelets made
        the clock 214 steps, and its 8-node culm stood full grown 6 days into
        a 90-day season. What does take steps: a relayed kind builds node by
        node (its whole span); any inflorescence axis starts the step after
        its node and lays its region down the next (2), a branched rachis's
        branches first growing their Branch Nodes a step at a time, per
        generation; and the region's nodes are born staggered over
        `elongation_steps`, so its last node starts that many steps after the
        first and must still elongate and reach full size -- without them a
        late raceme ended the season young (flowers ~30% small). Never more
        than `span_nodes`: this only takes back the dead season, so a kind
        whose old reserve was already short (Solitary, the umbels, a compound
        umbel's rays) keeps its clock exactly. (A rachis whose branches need
        more than the old reserve -- Branch Nodes 6 at depth 2 grows 98 of
        its 343 flowers -- is still truncated, as it was before; a separate
        fix.)"""
        if self.topology == Topology.RELAY:
            return self.span_nodes
        build = (max(1, self.branching_depth) * (max(0, self.branch_nodes) + 2)
                 if self.branching_depth > 0 else 2)
        return min(self.span_nodes, build + max(0, int(elongation_steps)))


@dataclass(frozen=True)
class PhenoWindow:
    """One phenological phase as a window in the growing year: it opens `start`
    days after the plant emerges and lasts `duration` days.

    Anchored to emergence, not an absolute date, so the plant's own Emergence
    Day is the single master offset that slides the whole calendar through the
    year -- a colder climate emerges later and carries every phase later with it
    -- while Season Length stays free to stretch growth speed without moving the
    phases. In real days, not season-fractions, so "blooms for two weeks" means
    two weeks whatever the season length.
    """

    start: float = 0.0
    duration: float = 0.0

    def opened(self, day: float, emergence_day: float) -> bool:
        """Whether the window has begun by `day` -- the flowering permission
        gate. A static snapshot at `day` asks the calendar exactly this one
        question, so the permission is a build constant, not a per-step test."""
        return day >= emergence_day + self.start

    def closed(self, day: float, emergence_day: float) -> bool:
        """Whether the window has ended by `day` -- e.g. fruit-drop, past which
        the fruit is gone."""
        return day >= emergence_day + self.start + self.duration

    def progress(self, day: float, emergence_day: float) -> float:
        """Position within the window: 0 before it opens, rising to 1 at its
        close and held there after. Drives Bloom Progress and the senescence
        fraction, so the window width *is* the bloom/senescence duration."""
        if self.duration <= 0.0:
            return 1.0 if self.opened(day, emergence_day) else 0.0
        return min(1.0, max(0.0, (day - emergence_day - self.start) / self.duration))


@dataclass
class PhenologyCalendar:
    """A species' phenological calendar: the per-phase windows (relative to
    emergence) it leafs out, blooms and senesces in, overlapping freely.

    Carried on `MorphologyProfile.calendar`; `None` (the default) means the
    plant has no calendar and every phase stays on its emergent,
    developmental-age driver -- byte-identical to before the calendar existed,
    which is what the goldens guard.

    **Consulted only on the day path** (`build(day=...)`): a calendar is a
    day-of-year concept and has nothing to bind to when a caller asks for a raw
    developmental `age`, so the age path -- goldens and every existing caller --
    ignores it entirely. That is what lets a species carry a real calendar
    without moving a single golden.

    Phase 3 reads `bloom` and `senescence`. `leaf_out`/`leaf_fall` are carried
    as data but inert for the herbaceous plants milestone 1 targets (leaf-out is
    emergence, leaf-fall is senescence); an independent deciduous leaf lifecycle
    is the woody-perennial phase. Fruit joins in phase 4.
    """

    bloom: PhenoWindow | None = None
    senescence: PhenoWindow | None = None
    fruit: PhenoWindow | None = None
    leaf_out: PhenoWindow | None = None
    leaf_fall: PhenoWindow | None = None


@dataclass
class LeafLoad:
    """The weight of leaves a component places itself (at nodes, internodes,
    tips, branch bases, the base, or at random) -- drawn after growth, so the
    bend cannot see them as organs. Physics adds this mass to the nodes they
    will sit on instead (`physics._leaf_load_mass`): which nodes is known from
    the placement and the levels alone, so their exact positions are not
    needed. Masks are not counted (they read the finished plant); Prune is, as
    an average."""

    mass: float = 0.0
    """One leaf's mass: its Weight x Size**2 x (1 - Prune), relative units."""
    mode: str = "node"
    """node / internode / tip / intersection / basal / random."""
    count: float = 1.0
    """Leaves at each anchor (the whorl, x the per-internode or basal count);
    for `random`, the total spread over the plant by internode length."""
    levels: tuple[int, ...] = ()
    """The levels it is placed on; empty = every level."""
    skip_inflorescence: bool = True
    """Foliage skips an inflorescence's nodes (as the placement does)."""


@dataclass
class MorphologyProfile:
    """One species' architecture. Every field is either a categorical
    botanical trait or a botanical measurement -- nothing here names a solver
    internal.
    """

    name: str = "generic perennial"

    # -- Branching levels ----------------
    levels: list[BranchLevel] = field(default_factory=default_levels)
    """The branching levels: stems levels (on the plant) and the branch levels
    growing on them, as a tree -- see `BranchLevel.parent` and the
    `level_parent` / `level_children` / `level_depth` helpers."""

    # -- Axis architecture ------------------------------------------------
    basal_spread: float = 0.04
    """Radius of the basal cluster the stems emerge from, in metres."""
    basal_spread_variation: float = 0.0
    """How irregularly the stems stand in the clump: each stem's foot moves by up
    to this fraction of `basal_spread` in X and in Y, drawn per stem (0.5: up to
    half the spread). 0 keeps the even phyllotactic disc."""

    inflorescence: Inflorescence = field(default_factory=Inflorescence)
    """The trunk's inflorescence, and the fallback for every order."""
    lateral_inflorescence: Inflorescence | None = None
    """A different inflorescence on branches than on the trunk. `None` -- every
    shipped species -- means "the same as the trunk", which is what it always
    silently was.

    **This is what makes compound inflorescences representable.** A panicle is
    a raceme whose branches are themselves racemes; a compound umbel is an
    umbel of umbels; a thyrse is a raceme of cymes. None of those is a new
    kind, and inventing enum entries for them would have been the wrong move --
    they are *compositions* of two orders, and until now `profile.inflorescence`
    was plant-global so every order was forced to agree. The zoning is
    deliberately the same trunk/lateral split `phyllotaxis_for` already uses,
    rather than a second, differently-shaped notion of "where on the plant".
    """

    extra_inflorescences: list[Inflorescence] = field(default_factory=list)
    """Further inflorescences, after `inflorescence` in priority order -- one per
    flower component that grows its own.
    Each grows on its `placement.levels`; where two want the same spot (a
    tip, or one node position) the earlier one has it. Index 0 names the
    primary (`inflorescence_for`), k the k-th of these (`inflorescence_at`)."""

    veg: Vegetativeness | None = None
    """Meristem vegetativeness -- see `Vegetativeness`. `None` means
    `Vegetativeness.default()`, which is what every shipped species uses. Set
    it directly only for a form the attachments cannot express."""

    # -- Instancing masks --------------------------
    masks: dict[str, MaskStack] = field(default_factory=dict)
    """Per-component point masks, keyed by component ("leaf", "flower", "bract",
    "bud", "fruit"). Each `MaskStack` folds attribute -> min/max -> curve ->
    combine maps into a per-point keep-probability deciding which of that
    component's organ points render. Empty -- every shipped species -- is a true
    no-op: the bridge bakes and gates nothing. Render-only: evaluated after
    `build()`, so it never changes the load the physics already sized for. See
    `growth/masks.py`."""

    # -- Growth maps (growth/maps.py) -------------------------------------
    maps: list[ParamMap] = field(default_factory=list)
    """Artist-authored signal -> curve -> multiplier maps on growth decisions
    (branch density / angle, internode length, branch length, gravitropism),
    evaluated per bud / node / axis *while the plant grows*. The growth-time
    sibling of `masks`, which only decide what renders afterwards.

    Each map only multiplies a quantity the decision already computes -- the
    physics stays the base, the way light already scales branch density -- and
    maps on one target multiply together. Empty -- every shipped species -- is
    a true no-op: no channel is read, no extra resolve runs, the build is
    byte-identical. See `growth/maps.py` for targets and channels."""

    # -- Leaf arrangement ---------------------------------------------------
    leaf_arrangement: LeafArrangement = field(default_factory=LeafArrangement)
    """Where grown leaves sit round each node, on every stem and branch alike
    -- owned by the primary leaf component's Phyllotaxis in the panel. (The
    branches have their own arrangement per branching level.)"""

    # Leaf placement (which stems, which part of an axis) was a growth-time
    # gate; it is now a render-time component mask. Growth emits a foliage leaf
    # on every vegetative node; a leaf component's masks decide which render.

    # -- Catkins ----------------------------------------------------------
    catkin_length: float = 0.0
    """Length of the catkin body, in metres. **0 means no catkins**, so every
    existing plant is untouched.

    **A catkin is borne at a node on its own stalk, not at the tip of a
    shoot**, which is what separates it from every `InflorescenceKind` here.
    Those convert an axis's own apex into flowers; a hazel carries its catkins
    on twigs that go on being twigs. Built as a flowering region instead, what
    hangs is the branch, and the tuning that would hide it does not exist --
    measured, the whole Peduncle Droop range moved such a catkin 1.7 cm against
    50 cm on the same plant as a spike.

    It is also **modelled rather than instanced**, and that is the second half
    of the same point: an instanced mesh is rigid, so it can be pointed
    downward but cannot hang, cannot curve, and cannot answer the branch above
    it moving. See `model.Organ.catkin_length` for the curve it is built as."""
    catkin_length_variation: float = 0.0
    """Length scatter between catkins, as a fraction.

    Drawn **per catkin, not per stalk**, so the three on one stalk differ from
    each other rather than every stalk carrying three matching ones. Real
    catkins on one twig are visibly unequal, and identical copies read as
    instancing."""
    catkin_count: int = 3
    """How many catkins hang from one stalk. Hazel, birch and willow all carry
    them in small bunches rather than singly.

    A real control rather than the node's phyllotaxis, which an earlier version
    used. A whorl of three does put three organs on a node -- and whorls the
    leaves with them, which is wrong for every catkin-bearing species here:
    their foliage is spiral. How many catkins share a stalk and how the foliage
    is arranged are independent facts."""
    catkin_count_variation: float = 0.0
    """Scatter in how many catkins a stalk carries, as a fraction of
    `catkin_count`. Hazel's bunches run two to four; a plant whose every stalk
    carries exactly three reads as instanced."""
    catkin_width: float = 0.004
    """Radius of the catkin body, in metres. The body is a capsule, so this is
    also the radius of its rounded ends."""
    catkin_density: float = 320.0
    """Mass per unit volume of a catkin body, in the same relative units as
    `wood_density`. What makes a bunch hang: `physics.apply_pedicel_droop` sizes
    each body's mass as `catkin_density * pi * catkin_width**2 * catkin_length`,
    sums the bunch, and sags the shared stalk under it by the same cantilever
    beam a pedicel uses -- so a hazel catkin hangs from its twig because it is
    heavy on a fine stalk, not because a `catkin_droop` dial was set. The body
    itself stays limp (`model._CATKIN_LIMP_DEG`): a real catkin is a floppy
    chain of florets, not a stiff rod, so it hangs straight down whatever the
    stalk does -- a material fact a solid-tube beam would miss."""
    catkin_stalk_variation: float = 0.0
    """Length scatter between one stalk and the next, as a fraction. Separate
    from `catkin_length_variation`, which scatters the catkins *on* a stalk:
    one varies where the bunches sit, the other what is in them."""
    catkin_stalk: float = 0.008
    """Length of the bare stalk between the node and the catkin body, in metres.
    This is the pedicel the catkin hangs from, and it is why a catkin reads as
    hanging *off* a twig rather than sprouting from it."""
    catkin_droop_variation: float = 0.0
    """Scatter in how far each catkin hangs, as a fraction of the angle
    `orientation_deg` states. A per-organ variance, which an axis-level
    orientation has no equivalent of -- so it stays here rather than moving
    with the angle itself."""
    catkin_noise: float = 0.0
    """How far each catkin bends out of true, as a fraction of its own width.

    Deliberately tiny in effect: two low harmonics with a per-catkin phase,
    growing from nothing at the attachment to full at the free end. Per-vertex
    jitter was not the alternative considered and rejected -- what wants
    breaking up is the *shape*, and a catkin that wobbles vertex to vertex
    reads as a bad mesh rather than as a real one."""

    def phyllotaxis_for(self, order: int) -> LeafArrangement:
        """The leaf arrangement on an axis of this order -- one for the whole
        plant."""
        del order
        return self.leaf_arrangement

    def inflorescence_for(self, order: int) -> Inflorescence:
        """The inflorescence this axis's order carries. Same trunk/lateral
        zoning as `phyllotaxis_for`; falls back to the trunk's when a species
        has not said otherwise, which every shipped one does."""
        if order == 0 or self.lateral_inflorescence is None:
            return self.inflorescence
        return self.lateral_inflorescence

    def inflorescence_at(self, index: int, order: int) -> Inflorescence:
        """Inflorescence `index` (0 the primary -- by order, as
        `inflorescence_for` -- k the k-th of `extra_inflorescences`)."""
        if 0 < index <= len(self.extra_inflorescences):
            return self.extra_inflorescences[index - 1]
        return self.inflorescence_for(order)

    def inflorescence_of(self, axis) -> Inflorescence:
        """The inflorescence a grown axis belongs to (`Axis.inflorescence`;
        undecided reads as the primary)."""
        return self.inflorescence_at(max(axis.inflorescence, 0), axis.order)

    def flower_size_of(self, inf: Inflorescence) -> float:
        """A flower's base size: its inflorescence's own, else the plant's."""
        return self.flower_size if inf.flower_size is None else inf.flower_size

    @property
    def produces_flowers(self) -> bool:
        """Whether any order of this plant makes an inflorescence at all.

        A separate question from the attachments, which say where one arises.
        `InflorescenceKind.NONE` is an apex that matures and stops bearing
        nothing, so a plant carrying it flowers exactly zero times and is
        correct rather than broken.

        Asks every order rather than the trunk's, because the case this exists
        for is precisely a plant that differs between them: NONE on the trunk
        with a catkin on the laterals is a flowering plant whose main stem does
        not flower."""
        laterals = self.lateral_inflorescence
        return (self.inflorescence.topology != Topology.NONE
                or (laterals is not None and laterals.topology != Topology.NONE)
                or any(x.topology != Topology.NONE for x in self.extra_inflorescences))

    # -- Form (botanical measurements) ------------------------------------
    internode_length: float = 0.045
    """The inflorescence's spacing base, in metres: a flowering node's internode
    is this x the inflorescence's Flower Spacing (or Branch Spacing on a rachis),
    and an inflorescence branch's is this x its Branch Length -- the same on
    every stem and branch, so racemes do not crowd on short-noded branches.
    Vegetative internodes are the branching levels' (their Start Length and
    Length / Step)."""
    leaf_angle_deg: float = 42.0
    leaf_size: float = 1.0
    leaf_size_variation: float = 0.15

    # -- Leaf orientation, the other two of the three axes ------------------
    # Pitch is `leaf_angle_deg` above. These turn the blade on its own frame
    # once the stem has placed it; see `model.orient_organ`.
    leaf_yaw_deg: float = 0.0
    leaf_yaw_variation: float = 0.0
    leaf_roll_deg: float = 0.0
    leaf_roll_variation: float = 0.0
    """Variation is the point of these two, more than the base value. A real
    plant's leaves are not all turned identically wrong -- they are turned
    *differently*, and a few degrees of scatter reads as growth where a
    uniform offset reads as a modelling mistake."""

    # Blade shape (curl/cup/gravity/wave/twist) was removed: leaves arrive
    # pre-shaped from the leaf tool.
    flower_size: float = 1.0
    fruit_size: float = 1.0
    """A ripened fruit's size, a multiplier on the fruit asset (mesh-driven,
    like `flower_size`). A fruit is its own organ at its own size, not the
    flower it set from. Day-path only -- fruit exists only inside the calendar
    fruit window."""

    gravitropism: float = 0.35
    """How strongly an apex steers back toward vertical each step, 0-1.
    Negative gravitropism (shoots growing up) is the normal case. Kept here
    beside the other directional traits rather than moved into the Environment
    block below: gravity is not something the surroundings supply, it is
    everywhere, and an apex responds to it with no field to read."""
    circumnutation_deg: float = 0.0
    """Amplitude of the growing apex's coherent helical nod, in degrees per
    step -- circumnutation (Darwin, *The Power of Movement in Plants*, 1880).
    Coherent on purpose, and the difference is measured: at the same per-step
    bend size an *incoherent* wander (the removed Growth Irregularity) sprawls (the tip
    walks off to one side, ~68 mm on a 1.2 m stem) while this *coherent* one
    winds a tight helix that stays close to the axis (~13 mm, 14 turns). Zero
    is off, and off is byte-identical to a plant with no circumnutation. How
    much it reads depends on how hard `gravitropism` corrects it -- a stiff
    upright stem straightens the wobble almost as fast as it is made, which is
    why `autotropism` is usually the control that makes it show on a
    self-supporting plant."""
    circumnutation_rate_deg: float = 137.5
    """How far the nodding azimuth advances each step, in degrees -- the pitch
    of the helix, its sign the chirality (left- vs right-handed). Inert while
    `circumnutation_deg` is zero. The default is the golden angle, which packs
    tightest; nothing depends on that value."""
    autotropism: float = 0.0
    """Age-stiffening, 0-1: how strongly a stem straightens back toward its own
    gravitropic setpoint as each node *ages*, applied once after growth. This
    is what concentrates the wander at the growing tip -- an old node is pulled
    nearly straight while the newest few keep their full nod, the canonical
    'straight stem, live circumnutating tip' of a growth timelapse. Autotropism
    (Firn & Digby): the tendency of an organ to straighten independent of
    gravity. Separate from `gravitropism`, which acts once at a node's birth
    and never revisits it; this is the revisiting. Zero is off and
    byte-identical. It only ever moves a node *toward* its setpoint -- it can
    undo the growth wander but never add to it -- and it is a different pass
    from the post-growth gravity bend, which sags the finished stem *down*
    under load."""

    # -- Environment response (Stage 3) -----------------------------------
    # How this species answers its surroundings. Every default is the neutral
    # value, so a profile that says nothing about the environment grows exactly
    # as it did before Stage 3 existed -- and `growth/environment.py` is never
    # even built for it.
    light_seeking: float = 0.0
    """Phototropism, 0-1: how far an apex turns toward the light each step.
    Applied after gravitropism, so it is the last word -- a shaded shoot leans
    out from under the canopy even though gravity would hold it upright."""
    leaf_heliotropism: float = 0.0
    """Foliar heliotropism, 0-1: how far each leaf turns its blade to face the
    sun (diaheliotropic, the light-gathering pose). Same sun and same
    post-growth re-aim as `Inflorescence.heliotropism`, its own strength because a
    species may track one and not the other. Distinct from an `ALIGN` effector,
    which twists the whole growth frame -- rotating the leaf *set* and the
    branches with it -- rather than pointing each blade at the sun. Zero is off
    and byte-identical."""
    obstacle_clearance: float = 0.0
    """How far from a solid surface, in metres, this plant keeps its growing
    tips. Zero disables obstacle response completely.

    One control rather than a clearance plus a steering strength: the distance
    is the property of the plant a person can actually see and state, and how
    sharply it has to turn to keep that distance follows from it. An apex that
    ends up inside a solid anyway stops growing rather than passing through --
    'Stop' among the collision responses (stop, prune, deflect), the one that
    leaves every profile curve along the axis still meaningful."""

    # -- Development timing ------------------------------------------------

    # -- Season / real-days calendar --
    # The developmental `age`/`steps` sim is unchanged; this only maps a real
    # `day` onto it. `progress_for_day` is the whole re-clocking layer: a day
    # becomes a season-fraction (via the growing-season window) and then the
    # developmental progress the existing loop runs on (via the growth curve).
    emergence_day: float = 90.0
    """Day-of-year the plant emerges and its growing season starts (nominal
    temperate spring). The clock is real days; a month/calendar view is just a
    display of the day counter."""
    season_length_days: float = 180.0
    """Length of the growing season in days, emergence -> the developmental
    plateau. Longer spreads the same growth over more real days."""
    season_growth_curve: Curve = field(
        default_factory=lambda: Curve.from_points([(0.0, 0.0), (1.0, 1.0)]))
    """Maps season-fraction (0 at emergence, 1 at season end) -> developmental
    progress (the `age` the sim runs on). Default is the identity line, so
    `progress_for_day` is a plain linear remap that reproduces the age-driven
    plant exactly -- the byte-identical guard for phase 1. A fast-then-plateau
    curve (plan Q9) is the realistic default a species carries: it tops out
    mid-season and then just matures, blooms and senesces on calendar dates."""
    calendar: PhenologyCalendar | None = None
    """The species' phenological calendar:
    per-phase windows relative to emergence that date-gate flowering and
    senescence when the plant is built by real `day`. `None` (the default, and
    every shipped species until authored) leaves phenology on its
    developmental-age drivers, byte-identical to before. Read only on the day
    path -- the age path (goldens, every existing caller) ignores it, which is
    what lets a species carry a calendar without moving a golden."""
    lifespan: str | None = None
    """How the plant lives across years (phase 5, see `Lifespan`). `None` is the
    default single-season behaviour; ANNUAL dies at season end; HERBACEOUS_-
    PERENNIAL dies back and regrows each year. Day-path only, and inert at year 0
    (Day 270), so byte-identical until scrubbed into a later season."""
    years_to_mature: int = 1
    """Years a perennial takes to reach its mature size and clump (phase 5). 1
    (the default) means mature immediately -- no year-over-year change, so a
    perennial is byte-identical to a single-season plant at year 0."""
    mature_vigour: float = 1.0
    """A perennial's mature *size* multiplier, reached at `years_to_mature`. Year
    0 is always 1.0 (today's plant); this is a subtle plateau above it (e.g. 1.2)
    that the size ramps up to. 1.0 (the default) is no size accumulation."""
    mature_stem_factor: float = 1.0
    """A perennial's mature *stem-count* multiplier, reached at `years_to_mature`
    (a clump only). Year 0 keeps the stems level's own Whorl; this expands
    it subtly year over year (e.g. 1.5). 1.0 (the default) is no accumulation."""

    plastochron: int = 1
    """Steps between successive phytomers on one axis."""
    elongation_steps: int = 5
    """Steps an internode takes to reach its full length along `elongation`
    after it forms (`curves.ELONGATION_MODES`: 1 Sequential, 5 Overlapping)."""
    leaf_expansion_steps: int = 7
    """Steps a leaf takes to reach its full size along `leaf_expansion`."""

    # -- Physical form (Stage 4) ------------------------------------------
    droop: float = 0.35
    """How far the grown structure bends under its own weight. Distinct from
    gravitropism, which changes how the plant *grows*; this changes how the
    already-grown structure *bends*, without touching topology."""
    stem_stiffness: float = 1.5e8
    """Young's modulus, in **pascals** -- a real material stiffness now, not a
    relative number. `physics.apply_gravity` is a complete Euler-Bernoulli beam
    (`theta = 4 g m arm L / (pi E r**4)`), so this is `E` directly and the whole
    thing is in SI: mass in kg, length in m, `g` in m/s**2.

    Provisional default -- 150 MPa sits at the herbaceous/grass-culm boundary
    and gives the generic perennial a ~25-degree tip nod on an upright trunk.
    **Stage 6 finalises this per species.** The old value (2e6, "relative
    units") folded every plant flat once the formula was completed, because the
    completed beam bends ~40x harder at a given number than the truncated one it
    replaced; the recalibration is a tuning within the herbaceous E range
    (1-100 MPa) and the grass-culm range above it, not a rescale by orders of
    magnitude."""
    stiffness_profile: Curve = field(default_factory=lambda: Curve.constant(1.0))
    """Young's modulus along an axis, base (0) to tip (1), as a multiplier on
    `stem_stiffness`. The lignification gradient: a real stem is woody and stiff
    at the base and green and soft at the tip, so a curve held high at the base
    and falling toward the tip concentrates the bend where it belongs -- the
    lamppost, a straight pole with the head hung over at the very end.

    Physically real rather than a correction dial, and it subsumes both of the
    old `where does it bend` controls: held high at the base it *is* `rigid_base`,
    and dropping over the top stretch *is* `peduncle_reach`. Multiplies E in
    `physics.apply_gravity`; a flat 1.0 is one uniform modulus and a true
    no-op."""
    stiffness_variation: float = 0.0
    """Random variation in each axis's stiffness, as a fraction of it. Drawn
    once per axis from its own id (like a level's width variation), so a clump of stems
    bends differently from one another without any one stem's stiffness becoming
    ragged along its length."""
    rigid_base: float = 0.0
    """How much of an axis, from the ground up, stays straight under load.

    The cantilever alone cannot say *where* a stem bends -- it distributes
    curvature by `carried * arm / radius ** 4`, which leans toward the thin
    top but still turns every node below. A real stem is woody at the base and
    green at the tip and puts nearly all its curvature in the top, which is the
    lamppost: a straight pole with the head hung over at the very end.

    Redistributes rather than reduces (see `physics._flexibility`), so this
    stays a control over the *shape* of the bend while `droop` and the organ
    weights own its *amount*. 0 is a true no-op and reproduces every plant that
    predates it byte-for-byte."""
    flowering_droop: float = 0.0
    """Extra bend on the stretch of stem that carries the flowers, **added to**
    `droop` there and stated in the same units.

    Kept alongside `rigid_base` rather than folded into it, because the two
    answer different questions and a plant needs both: this one scopes itself
    to wherever the flowers actually are, per axis and without being told,
    while `rigid_base` names a fraction of the axis directly. On a branched
    plant each axis's flowering region starts at a different node, which a
    single fraction cannot express; conversely sometimes only a short stretch
    should give way, which is what the fraction is for.

    It multiplied `droop` until 2026-08-16, which chained two controls that are
    meant to be independent: either at zero zeroed the other. 0 is a true
    no-op, exactly as the old multiplier's 1.0 was."""
    peduncle_reach: float = 0.0
    """How far *below* the lowest flower the flowering region extends, in
    metres of stem.

    A length rather than a fraction of the axis: a peduncle is a stalk of a
    certain size, and as a fraction the same number reached a different
    distance on every plant because it multiplied by node *count*. It also put
    the bend at the halfway point of a short axis for values that sounded
    small.

    Without it a solitary flower has no peduncle to speak of -- its one flower
    caps the apex, so the region above the lowest flower is a single node and
    there is nothing for `flowering_droop` to bend. Extending the region
    downward gives that flower a real stalk to nod on, and does the same for a
    raceme whose lowest flowers should not sit on a rigid stem."""
    peduncle_resolution: int = 1
    """How many segments each internode in the drooping region is split into.

    The bend can only be as smooth as the stem is dense, and the drooping
    region is the one stretch that is bent hard enough to show it: on a
    solitary flower it is a couple of nodes at the very top, so a large turn
    across them reads as facets rather than a curve. 1 changes nothing."""
    # -- Senescence (WS3): a post-growth pass, off by default --------------
    senescence_onset: Curve = field(default_factory=default_senescence)
    """Age death-front. X = plant Age, Y = the fraction of lateral branches
    (oldest first, i.e. lowest) that have died by then. Flat 0 -- the default --
    means nothing dies, keeping every plant byte-identical until asked. Read by
    `lifecycle.apply_lifecycle`, which runs after growth, so this never affects
    what grew, only what is marked dead afterwards."""
    senescence_retain_dead_wood: bool = True
    """Whether a dead branch stays as grey dead wood (True) or is removed
    entirely (False). Its leaves and flowers are shed either way -- a dead
    branch is bare."""
    senescence_shade_threshold: float = 0.0
    """Shade-death (WS3 Phase 2). After growth, each lateral branch is judged by
    the light at its base: below this threshold it dies and takes its subtree
    with it. 0 -- the default -- means never, keeping every plant byte-identical
    until a species opts in. Read by `lifecycle.apply_lifecycle` against the
    fully-built plant's own light field; it never feeds back into growth, which
    is the whole reason a removal is legal here. 1.0
    would kill anything not in full open sky."""

    wood_density: float = 500.0
    leaf_weight: float = 8.0e-5
    leaf_loads: list[LeafLoad] = field(default_factory=list)
    """The weight of leaves the components place themselves (`LeafLoad`), so
    every leaf drawn loads the stems, not only the grown ones. Empty: none."""
    flower_weight: float = 4.0e-4
    """Organ masses, relative. A flower head outweighs a leaf, which is what
    makes a blooming tip nod."""
    fruit_weight: float = 1.5e-3
    """A ripened fruit's mass (a lump, `scale**3 * fruit_weight`), heavier than
    the flower it set from -- what makes a fruiting branch bow. Day-path only:
    a fruit exists only inside the calendar fruit window, so off the day path
    this never enters `physics._organ_mass` and every plant is byte-identical."""
    fruit_set: float = 1.0
    """Fraction of flowers that set fruit (0-1), rolled per organ. The rest are
    unpollinated and drop at fruit-set. 1.0 = every flower fruits. Day-path
    only, read by `lifecycle.apply_fruit`."""

    # -- Curves ------------------------------------------------------------
    flowering_onset: Curve = field(default_factory=default_flowering_onset)
    leaf_size_profile: Curve = field(default_factory=default_leaf_size_profile)
    """Leaf size along the axis, as a multiplier on `leaf_size`. Closes the
    gradient a real stem has: large stalked leaves low down, small sessile ones
    under the inflorescence."""
    leaf_angle_profile: Curve = field(default_factory=default_leaf_angle_profile)
    """Leaf angle along the axis, as a multiplier on `leaf_angle_deg`."""
    elongation: Curve = field(default_factory=default_elongation)
    leaf_expansion: Curve = field(default_factory=default_leaf_expansion)
    stem_thickening: Curve = field(default_factory=default_stem_thickening)
    """Stem Thickening: each stem / branch's growth so far -> share of its
    stated widths (`development._thicken`). Mature axes are at 1."""

    # -- Derived helpers ---------------------------------------------------
    def level_parent(self, i: int) -> int:
        """The index of the level level `i` grows on, -1 for a stems level. An
        explicit parent that is out of range or would close a loop reads as a
        stems level, so a bad value can never hang growth."""
        p = self.levels[i].parent
        if p == PARENT_PREVIOUS:
            return i - 1
        if not 0 <= p < len(self.levels) or p == i:
            return -1
        seen = {i}
        q = p
        while q >= 0:                      # walk up: a loop back to i is a cycle
            if q in seen:
                return -1
            seen.add(q)
            nxt = self.levels[q].parent
            q = q - 1 if nxt == PARENT_PREVIOUS else (nxt if 0 <= nxt < len(self.levels)
                                                      else -1)
        return p

    def level_children(self, i: int) -> list[int]:
        """The levels growing on level `i` (-1: the stems levels), in list order."""
        return [j for j in range(len(self.levels)) if self.level_parent(j) == i]

    def level_depth(self, i: int) -> int:
        """0 for a stems level, one more per level below it."""
        d = 0
        while True:
            i = self.level_parent(i)
            if i < 0:
                return d
            d += 1

    @property
    def max_level_depth(self) -> int:
        """The deepest level's depth (0 with only stems)."""
        return max((self.level_depth(i) for i in range(len(self.levels))), default=0)

    @property
    def total_steps(self) -> int:
        """The plant's developmental span, in steps: long enough for the
        deepest chain of levels to grow out and flower, and for its last nodes
        to finish elongating.

        A branch starts the step after the node it leaves exists, so the
        deepest tip is complete after every level's Steps in turn (plus one
        step's lag per level); its inflorescence then needs the steps it
        builds in (`Inflorescence.clock_nodes` -- not a step per flower: a
        flowering region is laid down whole), and the newest nodes
        `elongation_steps` more to reach full length. Sized
        from the plant itself, so Age 1.0 is always the finished plant, and
        growing to Age 0.6 is exactly the first 60% of that run -- a
        truncation, not a different simulation."""
        laterals = self.lateral_inflorescence

        def longest(i: int) -> int:     # the steps of the longest path down from i
            own = max(1, self.levels[i].steps) + 1
            return own + max((longest(j) for j in self.level_children(i)), default=0)

        chain = max((longest(i) for i in self.level_children(-1)), default=0) or 1
        e = self.elongation_steps
        span = max(self.inflorescence.clock_nodes(e),
                   laterals.clock_nodes(e) if laterals is not None else 0,
                   *(x.clock_nodes(e) for x in self.extra_inflorescences))
        return chain + span + max(self.elongation_steps, 0)

    @property
    def emergence_span(self) -> int:
        """The most steps Emergence Delay can hold any part of the plant back:
        the largest sum of the levels' longest delays down a chain of levels.
        Growth itself runs on `total_steps`, unchanged; the plant is SHOWN over
        `total_steps + emergence_span`, so the last-delayed branch still
        finishes (`development._reveal`). 0 without a delay."""
        def deepest(i: int) -> int:
            return self.levels[i].longest_delay() + max(
                (deepest(j) for j in self.level_children(i)), default=0)

        return max((deepest(i) for i in self.level_children(-1)), default=0)

    def steps_for_age(self, age: float) -> int:
        return int(round(min(1.0, max(0.0, age)) * self.total_steps))

    def progress_for_day(self, day: float) -> float:
        """Developmental progress (the `age` the sim runs on) for a real `day`.

        The whole re-clocking layer: day -> season-fraction (clamped to the
        growing-season window) -> `season_growth_curve` -> progress. With the
        default identity curve this is a plain linear remap, so a plant grown at
        ``day = emergence_day + a * season_length_days`` is the age-`a` plant to
        the bit -- the byte-identical guard for the calendar rework's phase 1.
        """
        span = max(self.season_length_days, 1e-6)
        frac = min(1.0, max(0.0, (float(day) - self.emergence_day) / span))
        return float(self.season_growth_curve(frac))

    def day_for_progress(self, progress: float) -> float:
        """The real Day at which development reaches `progress` -- the inverse of
        `progress_for_day` (the season curve is non-decreasing; the earliest
        Day reaching it). Lets each growth step read the calendar at its OWN
        Day rather than the Day the plant is grown to."""
        p = min(1.0, max(0.0, float(progress)))
        lo, hi = 0.0, 1.0
        if float(self.season_growth_curve(0.0)) >= p:
            hi = 0.0
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if float(self.season_growth_curve(mid)) >= p:
                    hi = mid
                else:
                    lo = mid
        return self.emergence_day + hi * max(self.season_length_days, 1e-6)

    def flowering_permission(self, plant_age: float, day: float | None = None) -> float:
        """Whether the plant may flower now, as a 0..1 gate (both flowering call
        sites read it as a level, not a bool).

        The calendar's bloom window decides it on the day path: flowering opens
        on the bloom date and the structural node floor still decides
        readiness, so bloom lands at max(date, ready) (plan Q5a). With no
        calendar or no day -- the age path, goldens, every existing caller -- it
        is the developmental `flowering_onset(plant_age)` curve exactly, so
        nothing off the day path moves."""
        cal = self.calendar
        if day is not None and cal is not None and cal.bloom is not None:
            return 1.0 if cal.bloom.opened(day, self.emergence_day) else 0.0
        return float(self.flowering_onset(plant_age))

    def senescence_fraction(self, age: float, day: float | None = None) -> float:
        """The fraction of the plant senesced now. The calendar's senescence
        window drives it on the day path (position within the window); with no
        calendar or no day it is the developmental `senescence_onset(age)`
        curve exactly, byte-identical off the day path."""
        cal = self.calendar
        if day is not None and cal is not None and cal.senescence is not None:
            return cal.senescence.progress(day, self.emergence_day)
        return float(self.senescence_onset(age))

    def bloom_window_progress(self, day: float | None) -> float | None:
        """Position within the bloom window at `day`, or `None` when the plant
        has no bloom calendar to read -- then Bloom's From Age mode uses maturity
        instead. Drives Bloom Progress, so the window width is the bloom
        duration."""
        cal = self.calendar
        if day is None or cal is None or cal.bloom is None:
            return None
        return cal.bloom.progress(day, self.emergence_day)

    def autumn_progress(self, day: float | None) -> float:
        """Per-leaf autumn discoloration stage, 0 (green) to 1 (fully turned),
        from the day's position in the senescence window (phase 4). This is the
        signal a leaf shader mixes its autumn colour on -- the addon supplies the
        stage, the shader does the colour. 0 off the day path or with no
        senescence calendar, so foliage stays green until a species asks for
        autumn (which is the same window that sheds its limbs)."""
        cal = self.calendar
        if day is None or cal is None or cal.senescence is None:
            return 0.0
        return cal.senescence.progress(day, self.emergence_day)

    def fruit_progress(self, day: float | None) -> float:
        """Fruit ripeness stage, 0 (just set) to 1 (ripe, at fruit-drop), from
        the day's position in the fruit window (phase 4). The signal a fruit
        shader mixes its ripening colour on -- the addon supplies the stage, the
        shader does the colour, exactly like `autumn_progress`. 0 off the day
        path or with no fruit calendar. All fruit share it: they set and drop on
        one window, so ripeness is plant-level, not per-fruit."""
        cal = self.calendar
        if day is None or cal is None or cal.fruit is None:
            return 0.0
        return cal.fruit.progress(day, self.emergence_day)

    def _year_ramp(self, year: int, mature: float) -> float:
        """A multiplier rising from 1.0 at year 0 to `mature` at `years_to_mature`
        (phase 5). Year 0 is always 1.0 -- today's plant -- so any accumulation
        leaves the default Day-270 view untouched; `mature` 1.0 is a flat no-op."""
        if year <= 0 or mature == 1.0:
            return 1.0
        t = min(1.0, year / max(1, self.years_to_mature))
        return 1.0 + (mature - 1.0) * t

    def vigour_at_year(self, year: int) -> float:
        """This year's subtle size multiplier (phase 5). 1.0 at year 0."""
        return self._year_ramp(year, self.mature_vigour)

    def stem_factor_at_year(self, year: int) -> float:
        """This year's subtle stem-count multiplier (phase 5). 1.0 at year 0."""
        return self._year_ramp(year, self.mature_stem_factor)

    def season_state(self, day: float | None):
        """Route a real `day` to ``(grow_day, vigour, stem_factor, mode)`` for a
        multi-year build (phase 5). `mode` is 'grow', 'bare' (a perennial's
        dormant off-season -- nothing above ground) or 'dead' (an annual past its
        one season -- fully senesced, standing).

        With `lifespan` None or no day it passes the day straight through with
        neutral multipliers, so the single-season path is exactly as before. For
        a multi-year lifespan, `grow_day` is the *day-of-year*, so each year's
        phenology and progress repeat on the same calendar dates -- the whole of
        'regrow a fresh top each season'."""
        if day is None or self.lifespan is None:
            return day, 1.0, 1.0, 'grow'
        year = int(day // _YEAR_DAYS)
        doy = float(day % _YEAR_DAYS)
        # The above-ground plant is present from emergence through the end of its
        # autumn dieback, then dormant. Growth reaches mature at
        # emergence+season_length; the senescence window, if the species has one,
        # extends the *visible* season past that as the plant dies back, so
        # dormancy begins only once it has fully senesced -- otherwise a perennial
        # would pop from full-mature straight to bare with no autumn. The bound is
        # exclusive, so the mature last day (the default Day 270) still grows.
        active_end = self.emergence_day + self.season_length_days
        cal = self.calendar
        if cal is not None and cal.senescence is not None:
            active_end = max(active_end, self.emergence_day
                             + cal.senescence.start + cal.senescence.duration)
        if self.lifespan == Lifespan.ANNUAL:
            # One life: past its dieback (or any later year) it is dead, built at
            # the end of that dieback and fully killed, and it never regrows.
            if year >= 1 or doy > active_end:
                return float(active_end), 1.0, 1.0, 'dead'
            return doy, 1.0, 1.0, 'grow'
        # HERBACEOUS_PERENNIAL: year-relative day, subtly bigger each year.
        vig = self.vigour_at_year(year)
        stem = self.stem_factor_at_year(year)
        if doy < self.emergence_day or doy > active_end:
            return doy, vig, stem, 'bare'
        return doy, vig, stem, 'grow'

    @property
    def reads_light(self) -> bool:
        """Whether anything on this profile consults the light field. When
        nothing does, the field is never built -- which is both the cost saving
        and the guarantee that Stage 3 off means byte-identical output."""
        return self.light_seeking > 0.0

    @property
    def tracks_sun(self) -> bool:
        """Whether any flower or leaf turns toward the sun (heliotropism) --
        the only thing that reads the Sun Object."""
        infs = [self.inflorescence, self.lateral_inflorescence, *self.extra_inflorescences]
        return self.leaf_heliotropism > 0.0 or any(
            inf is not None and inf.heliotropism > 0.0 for inf in infs)

    def vegetativeness(self) -> Vegetativeness:
        """This profile's veg: `Vegetativeness.default()` unless stated."""
        return self.veg if self.veg is not None else Vegetativeness.default()

    def expectations(self):
        """The structural guardrails this profile should satisfy. Kept here so
        a species declares its own bounds and tests/structural.py needs no
        knowledge of morphology internals."""
        from ..tests.structural import Expectations

        return Expectations(
            max_branch_order=self.max_level_depth,
            expects_leaves=True,
            # A NONE inflorescence means nothing flowers: check 6 would
            # otherwise fail a foliage species for being exactly what it says.
            expects_flowers=self.produces_flowers,
            main_axis_height_share=0.7,
            # A forking stem splits into co-equal continuations, and a relaying
            # one is a chain of units, so no single order-0 axis reaches the
            # crown -- check 2 does not apply.
            single_leader=not (self.levels and (
                (self.levels[0].forks and self.levels[0].fork_probability > 0.0)
                or self.levels[0].relays)),
        )


def generic_perennial() -> MorphologyProfile:
    """The fallback profile. Deliberately a plain single-stem determinate
    perennial: it should look like a plant out of the box, since a default
    that needs tuning to be presentable is how the previous build's parameter
    surface grew.

    Carries a plausible phenological calendar (phase 3): mid-summer bloom,
    senescence at the season's end. Read only on the day path, so this leaves
    the age-path fixtures every golden is built from byte-identical, while a
    new plant blooms and senesces on real dates out of the box."""
    return MorphologyProfile(
        calendar=PhenologyCalendar(
            bloom=PhenoWindow(start=55.0, duration=75.0),
            senescence=PhenoWindow(start=180.0, duration=60.0),
            fruit=PhenoWindow(start=185.0, duration=95.0)),
        # A herbaceous perennial: dies back each winter, regrows each spring, a
        # touch bigger each year to a subtle plateau (phase 5). Year 0 -- the
        # default Day 270 -- is unchanged; scrubbing into later years shows it.
        lifespan=Lifespan.HERBACEOUS_PERENNIAL, years_to_mature=4, mature_vigour=1.2)
