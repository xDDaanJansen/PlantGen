"""The three reference species the rebuild plan names as its acceptance test:
a grass, *Rudbeckia fulgida* and *Anemone hupehensis*.

The gate is that these read as **three different plants at the same Age
setting** -- the property the previous build most conspicuously lacked, where
every species came out as the same blob at a different size.

Each profile is a plain `MorphologyProfile`, so a species is data rather than
code: no branching on species name anywhere in the growth loop. That matters
for the eventual preset format, which has to serialise a species to JSON with
no bpy pointers in it.

Two kinds of field live here, and the split is deliberate:

* Fields the panel exposes, which a species merely *seeds* -- the dropdown
  writes them into the object's properties and the user edits from there.
* Fields the panel does not expose (curves, `plastochron`, organ masses),
  which the species owns outright. `bridge.profile_from_object` builds on the
  chosen species rather than on a bare default, so these survive without
  needing a slider each. This is what keeps a species richer than the parameter
  surface without growing the parameter surface.

**Stems and branches are a stack of `BranchLevel`s**: level 0 is the stems (its Whorl is the stem
count, its Start Pitch their splay), level k+1 the branches on level k, and
*every* node of a level with a next level branches. Each species below was
re-authored from its old branching controls (stem count, splay, branch angle,
branch order, flowering node count, apical control, the pipe-model widths) by
measuring the plant the old controls built -- node counts per order, first
internode and internode taper fitted as a geometric series, base radius and
width taper. Where the old plant branched only sparsely or only in part of the
stem, a level has fewer Steps (with longer internodes, so the height holds)
rather than a new control; each docstring says where. The stems level's Start
Length and Start Width are the stems' own first internode and base radius, in
metres; `internode_length` is only the inflorescence's spacing base.

The grass was left as "which grass?" in the plan's own unresolved list. Meadow
foxtail answers it: it is a real named species, its culms are unbranched (so it
is a single stems level with no branch level), it tillers into a real clump,
and its dense cylindrical spike is reachable with the inflorescence kinds that
exist rather than needing the deferred recursive ones.
"""

from __future__ import annotations

from .curves import Curve
from .morphology import (
    Attachment,
    BranchLevel,
    Inflorescence,
    InflorescenceKind,
    LeafArrangement,
    LevelPhyllotaxis,
    Lifespan,
    MorphologyProfile,
    OrientationMode,
    PhenologyCalendar,
    PhenoWindow,
    Placement,
    generic_perennial,
)


def _calendar(bloom_start: float, bloom_days: float,
              sen_start: float = 180.0, sen_days: float = 60.0,
              fruit_start: float = 185.0, fruit_days: float = 95.0) -> PhenologyCalendar:
    """A species' phenology as windows in days after emergence (see
    `PhenologyCalendar`). Plausible botanical defaults, tuned so the default
    mature view (Day 270, season end) is unchanged: bloom opens well before it,
    while senescence and fruit-set open *at or after* it (`sen_start` 180 = the
    180-day season's end; `fruit_start` 185 a touch later), so a day-270 plant is
    in flower with no die-back and no fruit exactly as before the calendar, and
    scrubbing later reveals autumn colour, fruit-set and drop. Read only on the
    day path. Fruit-set after Day 270 is a simplification for the unchanged
    default view -- real fruit often sets in late summer -- the same compromise
    senescence already makes; it is a species-authoring detail, not a limit of
    the model, which gates on whatever dates a window carries."""
    return PhenologyCalendar(
        bloom=PhenoWindow(start=bloom_start, duration=bloom_days),
        senescence=PhenoWindow(start=sen_start, duration=sen_days),
        fruit=PhenoWindow(start=fruit_start, duration=fruit_days),
    )


def _clump(count: int, splay_deg: float, **growth) -> BranchLevel:
    """A stems level of `count` stems filling the base (a single
    stem is the Alternate default), splayed `splay_deg` from vertical."""
    if count == 1:
        return BranchLevel(start_pitch_deg=splay_deg, **growth)
    return BranchLevel(phyllotaxis=LevelPhyllotaxis.CUSTOM, spiral_angle_deg=0.0,
                       whorl=count, start_pitch_deg=splay_deg, **growth)


def meadow_foxtail() -> MorphologyProfile:
    """*Alopecurus pratensis* -- a tufted meadow grass, 0.5-1.0 m.

    The unbranched case. A grass culm makes no side branches at all, so the
    whole silhouette comes from tillering: many stems from one crown, each
    ending in a dense spike. One level only -- the stems -- so there is no
    branch level for the leaf-axil buds to grow into. Leaves are two-ranked,
    which is the one place `DISTICHOUS` is exactly right rather than an
    approximation.

    Grass internodes are the textbook case of lengthening upward: short and
    crowded at the crown, longest just under the spike. That is the stems'
    Length / Step (1.27 a node from a 27 mm first internode, fitted to the old
    internode curve's culm). The old clump emerged one tiller every two steps
    on a disc, splayed in proportion to how far out each sat; the stems level
    emerges them together on one ring at full splay, so the tillers no longer
    differ in age and the tuft leans a little wider (culms ~0.67 m against the
    old 0.67-0.84 m).
    """
    return MorphologyProfile(
        name="meadow foxtail",
        # Flowers in late spring / early summer (day ~120), a short shedding window.
        calendar=_calendar(bloom_start=30.0, bloom_days=40.0),
        # A perennial tussock grass: the clump thickens year over year (phase 5).
        lifespan=Lifespan.HERBACEOUS_PERENNIAL, years_to_mature=4,
        mature_vigour=1.15, mature_stem_factor=1.5,
        basal_spread=0.055,
        levels=[
            _clump(11, 8.0, steps=8,
                   start_length_scale=0.0272, length_scale_per_step=1.27,
                   width_scale_per_step=0.968,
                   start_width_scale=0.0048, length_variation=0.14),
        ],
        inflorescence=Inflorescence(
            kind=InflorescenceKind.SPIKE,
            flower_count=45,
            # Was 0.055 under the old internode curve, which also scaled the
            # spike's internodes (by ~1.3 up there). A flowering node is now
            # spaced by Internode Length x this alone, so this carries that
            # factor and the spike keeps its old length (~0.28 m).
            internode_scale=0.072,
        ),
        leaf_arrangement=LeafArrangement(spiral_angle_deg=180.0, whorl=1),
        internode_length=0.085,
        leaf_angle_deg=24.0,
        leaf_size_variation=0.2,
        gravitropism=0.62,
        droop=0.3,
        # E in pascals (the unified beam's real modulus). Raised from 1.1e6 when
        # gravity became one correct Euler-Bernoulli beam: the old value was
        # tuned against the pre-unification formula that under-bent by ~40x, so
        # at it a foxtail culm folded to the ground (425 nodes below z=0).
        # 1.5e7 stands the culm near-upright with a gentle nod at the spike.
        stem_stiffness=1.5e7,
        # A culm carries its spike at the very top of a thin stem, so the head
        # has to be light or the whole plant folds over.
        # Leaf/flower size default to 1.0 (a mesh-driven multiplier); the weights
        # carry the old size**3 / size**2 so mass -- and so droop -- is unchanged.
        flower_weight=1.2e-4 * 0.3 ** 3,   # was flower_size 0.3
        leaf_weight=6.0e-5 * 1.45 ** 2,    # was leaf_size 1.45
        # Flowering onset is left at the default. These profiles were tuned
        # against a build where the onset gate was never applied (see
        # `_terminate_determinate_axes`), so the late curve each of them
        # used to declare was decorative -- honouring it now would take
        # this species somewhere it was never tuned for. Late flowering is
        # a real control again; a species that wants it should say so
        # deliberately and be retuned around it.
    )


def rudbeckia_fulgida() -> MorphologyProfile:
    """*Rudbeckia fulgida* -- orange coneflower, 0.6-0.9 m.

    A clump-forming perennial whose stems branch in their upper part, each
    branch ending in one daisy-like head.

    That head is a capitulum, and it is **`SOLITARY`**: one instanced flower per
    branch tip, with the florets inside it left to the flower asset. This is
    what `InflorescenceKind.SOLITARY`'s own docstring has said all along, and
    what the parametric add-on decided on 2026-07-29 when it removed its axis-level
    Receptacle.

    **Branching compromise.** The old plant branched sparsely (density 0.35)
    and only in the top 40% of each stem, reading as a handful of heads held at
    about one level. Every node of a level now branches, so the stems have 4
    Steps rather than the old 13 -- lengthening upward (x1.3 a node), which
    keeps the lower nodes near the crown and leaves the upper ones long. A
    branch's internode follows its parent's internode where it attaches, so the
    low branches stay short and the upper ones reach up toward the leader's
    head. The old third order (a few 2-node branchlets) is dropped rather than
    multiplied onto every branch node. The cost: 25 heads where the old clump
    had 16, some of them low, and 4 leaves a stem where it had 13.
    """
    return MorphologyProfile(
        name="Rudbeckia fulgida",
        # A late-summer daisy: blooms day ~180 for a long stretch into autumn.
        calendar=_calendar(bloom_start=90.0, bloom_days=70.0),
        # A clump-forming herbaceous perennial, spreading year over year (phase 5).
        lifespan=Lifespan.HERBACEOUS_PERENNIAL, years_to_mature=4,
        mature_vigour=1.2, mature_stem_factor=1.6,
        basal_spread=0.05,
        levels=[
            _clump(5, 7.0, steps=4,
                   start_length_scale=0.09, length_scale_per_step=1.3,
                   width_scale_per_step=0.8,
                   start_width_scale=0.0044, length_variation=0.2),
            BranchLevel(start_pitch_deg=38.0, steps=4,
                        start_length_scale=0.38,
                        start_width_scale=0.8, width_scale_per_step=0.85, length_variation=0.2),
        ],
        inflorescence=Inflorescence(kind=InflorescenceKind.SOLITARY),
        internode_length=0.05,
        leaf_angle_deg=46.0,
        gravitropism=0.55,
        droop=0.18,
        # E in pascals; raised hard from 4.0e6 for the unified beam (see meadow
        # foxtail). A coneflower holds its heads stiffly upright with barely any
        # droop: at 3.0e8 the gravity sag is ~2 degrees, so each head sits at its
        # own branch angle rather than nodding. The old value flopped 257 nodes
        # below z=0.
        stem_stiffness=3.0e8,
        # Size defaults to 1.0 (a mesh-driven multiplier); the weights carry the
        # old size**3 / size**2 so mass -- and so droop -- is unchanged.
        flower_weight=6.0e-4 * 0.34 ** 3,   # was flower_size 0.34
        leaf_weight=8.0e-5 * 1.2 ** 2,      # default weight, was leaf_size 1.2
        # Flowering onset is left at the default. These profiles were tuned
        # against a build where the onset gate was never applied (see
        # `_terminate_determinate_axes`), so the late curve each of them
        # used to declare was decorative -- honouring it now would take
        # this species somewhere it was never tuned for. Late flowering is
        # a real control again; a species that wants it should say so
        # deliberately and be retuned around it.
    )


def anemone_hupehensis() -> MorphologyProfile:
    """*Anemone hupehensis* -- Japanese anemone, 0.9-1.2 m.

    Tall wiry flowering stems rising well clear of a basal clump, branching
    into a loose, airy spray with one large flower at the end of every branch.

    That spray is botanically a cyme, and this profile builds it the long way
    round: a determinate axis carrying a `SOLITARY` flower with lateral shoots
    below it, each ending in its own flower -- the real structure rather than
    an imitation of it. The branching is the inflorescence here, which is why
    this profile has by far the longest internodes and the widest branch angle.
    **`DICHASIUM` says the same thing directly** and would be the honest way to
    author this species today; left as it is deliberately, since switching is a
    retuning and a look call, not a port.

    **Branching compromise.** The old plant branched only in the top 30% of
    the stem, sparsely, to two orders. Every node now branches, so the stems
    have just 3 Steps, the first one very long (a bare wiry stalk) and each next
    one shorter (x0.6) -- which puts all three branching nodes in the upper part
    of the stem, the lower branches longer than the upper ones as in a cyme.
    Branches sit in opposite pairs (the old lateral arrangement). The old second
    order is dropped: on every node of every branch it would multiply the spray
    fourfold.
    """
    return MorphologyProfile(
        name="Anemone hupehensis",
        # Japanese anemone: an autumn bloomer (day ~200), senescing a touch later.
        calendar=_calendar(bloom_start=110.0, bloom_days=60.0, sen_start=185.0, sen_days=55.0),
        # Spreads into a wider clump each year, a slow-establishing perennial (phase 5).
        lifespan=Lifespan.HERBACEOUS_PERENNIAL, years_to_mature=5,
        mature_vigour=1.2, mature_stem_factor=1.8,
        basal_spread=0.07,
        levels=[
            _clump(4, 10.0, steps=3,
                   start_length_scale=0.475, length_scale_per_step=0.6,
                   width_scale_per_step=0.8,
                   start_width_scale=0.0048, length_variation=0.24),
            BranchLevel(phyllotaxis=LevelPhyllotaxis.OPPOSITE, spiral_angle_deg=0.0, whorl=2,
                        start_pitch_deg=54.0, steps=2,
                        start_length_scale=0.3,
                        start_width_scale=0.8, width_scale_per_step=0.7, length_variation=0.24),
        ],
        inflorescence=Inflorescence(kind=InflorescenceKind.SOLITARY),
        # Anemone's stem leaves sit in threes at the nodes, so a whorl on the
        # main stem and pairs on the branches is the leaf arrangement, not a
        # spiral.
        leaf_arrangement=LeafArrangement(spiral_angle_deg=60.0, whorl=3),
        internode_length=0.095,
        leaf_angle_deg=52.0,
        leaf_size_variation=0.22,
        gravitropism=0.32,
        droop=0.2,
        # E in pascals for the unified beam. A big flower on a thin stem is what
        # makes an anemone nod -- but under the corrected beam the old 8.0e6 with
        # a 1.1e-3 flower folded the flower straight down (main tip 178 deg) and
        # dropped 281 nodes below z=0. A real anemone flower is papery, not
        # heavy: at flower_weight 3.0e-4 on a stiff 1.5e9 stem it nods a gentle
        # ~35 deg, facing up-and-out, which is the look this species is here for.
        stem_stiffness=1.5e9,
        # Size defaults to 1.0 (a mesh-driven multiplier); the weights carry the
        # old size**3 / size**2 so mass -- and so droop -- is unchanged.
        flower_weight=3.0e-4 * 1.5 ** 3,   # was flower_size 1.5
        leaf_weight=1.4e-4 * 1.5 ** 2,     # was leaf_size 1.5
        # Flowering onset is left at the default. These profiles were tuned
        # against a build where the onset gate was never applied (see
        # `_terminate_determinate_axes`), so the late curve each of them
        # used to declare was decorative -- honouring it now would take
        # this species somewhere it was never tuned for. Late flowering is
        # a real control again; a species that wants it should say so
        # deliberately and be retuned around it.
    )


def digitalis_purpurea() -> MorphologyProfile:
    """*Digitalis purpurea* -- foxglove, 1.0-1.5 m.

    The reason this species is here is the **rosette**. A large share of
    perennials spend their juvenile phase as a basal rosette -- leaves stacked
    on internodes of almost no length -- and then bolt, throwing a tall
    flowering stem in a short burst. Foxglove is the textbook case, and none of
    the other profiles does it: the grass tillers, and both the Rudbeckia and
    the Anemone spread their leaves up the stem.

    It needs **no special mechanism**: the stems level's Length / Step. Its
    first internode is 4.5 mm and each next one is 1.24x the
    last, so the lowest nodes stack into a rosette and the stem bolts above
    them -- fitted to the old internode rosette curve (flat near 0.06,
    bolting at ~35% of the axis). The geometric series bolts more gradually
    than that curve did and its last vegetative internodes run longer, but the
    vegetative height (~0.89 m) and node count are the old plant's.

    Two supporting choices follow from the rosette rather than being separate
    decisions. `leaf_size_profile` decays hard, because a foxglove's basal
    leaves are far larger than the few small bracts up the spike; and
    `leaf_angle_profile` opens at the base, because rosette leaves lie almost
    flat while stem leaves clasp.

    Unbranched (a single stems level, no branch level), which with the grass
    makes two species exercise that path, and a `RACEME` -- the tall one-sided
    spire of tubular flowers. Single-stem, unlike the clump-formers.
    """
    return MorphologyProfile(
        name="Digitalis purpurea",
        # Foxglove: early-to-mid summer spires (day ~135) over a few weeks.
        calendar=_calendar(bloom_start=45.0, bloom_days=50.0),
        levels=[
            _clump(1, 0.0, steps=18,
                   start_length_scale=0.004505, length_scale_per_step=1.24,
                   width_scale_per_step=0.967,
                   start_width_scale=0.0094, length_variation=0.15),
        ],
        inflorescence=Inflorescence(
            kind=InflorescenceKind.RACEME,
            flower_count=26,
            pedicel_length=0.018,
            pedicel_angle_deg=50.4,
            pedicel_bend=0.45,
            internode_scale=0.34,
            bloom_order="acropetal",
            # What actually makes a foxglove read as one, beyond the raceme
            # topology it shares with a dozen other plants.
            #
            # Stalks shorten and swing up toward the tip, which is what tapers
            # the spire; flowers shrink the same way, so the apex is a point of
            # small buds rather than a blunt end; and the spacing tightens
            # upward, crowding the young flowers together.
            pedicel_length_profile=Curve.from_points(
                [(0.0, 1.0), (0.55, 0.72), (1.0, 0.45)]),
            # Stalk tilt runs 18 degrees above horizontal at the base to 50.4
            # at the tip. Stated as a fraction of the 50.4 rather than as
            # round numbers because Pedicel Angle changed sense (from-the-axis
            # to signed-from-perpendicular) and these reproduce the tuned
            # foxglove to four decimals rather than approximately.
            pedicel_angle_profile=Curve.from_points(
                [(0.0, 0.3571), (0.5, 0.6143), (1.0, 1.0)]),
            flower_size_profile=Curve.from_points(
                [(0.0, 1.0), (0.45, 0.86), (0.75, 0.55), (1.0, 0.28)]),
            flower_spacing_profile=Curve.from_points(
                [(0.0, 1.0), (0.6, 0.78), (1.0, 0.5)]),
            # The two presentation traits. Strongly one-sided, and the bells
            # hang well below their stalks -- both are what a person actually
            # recognises, and neither is a branching decision.
            # 0.75 "one-sidedness" under the old parameterisation, exactly:
            # the flowers occupy a quarter of the stem's circumference.
            flower_spread_deg=90.0,
        ),
        internode_length=0.085,
        leaf_angle_deg=42.0,
        leaf_size_variation=0.18,
        # A foxglove spire is famously straight, so it corrects hard.
        gravitropism=0.78,
        droop=0.1,
        stem_stiffness=1.4e7,
        # Size defaults to 1.0 (a mesh-driven multiplier); the weights carry the
        # old size**3 / size**2 so mass -- and so droop -- is unchanged.
        flower_weight=3.0e-4 * 0.5 ** 3,   # was flower_size 0.5
        leaf_weight=2.6e-4 * 2.1 ** 2,     # was leaf_size 2.1
        # Big basal leaves, small bracts up the spike.
        leaf_size_profile=Curve.from_points(
            [(0.0, 1.0), (0.3, 0.72), (0.6, 0.3), (1.0, 0.16)]),
        # Rosette leaves lie almost flat; stem leaves clasp.
        leaf_angle_profile=Curve.from_points(
            [(0.0, 1.6), (0.35, 1.15), (1.0, 0.8)]),
        # (Heteroblasty -- a broad basal rosette leaf vs a narrow stem leaf --
        # was expressed here as a leaf_form_profile; that built-in is gone. It
        # is now two leaf components with an age mask. leaf_size_profile and
        # leaf_angle_profile above still shrink and flatten the stem leaves.)
    )


def corylus_avellana() -> MorphologyProfile:
    """*Corylus avellana* -- hazel, as a young twig bearing catkins.

    The catkin case, and the only species here whose stems bear **no flowers of
    their own**: every axis matures and stops without flowering, and what
    carries the flowering is `catkin_length` -- a hanging curve borne at the
    nodes of the side twigs.

    * **Catkins are borne at nodes, not at shoot tips.** See
      `MorphologyProfile.catkin_length` -- this is the one structure here that
      is neither an inflorescence on an axis nor an instanced organ.
    * **Its placement is branches only, upper half.** Hazel bears its
      catkins near the tips of the previous season's side shoots, not on the
      leader and not at the base.

    **Branching compromise.** The old twig branched in its upper part only
    (density 0.7 over the top half: three side shoots on a 12-node leader).
    Every node now branches, so the leader has 6 Steps of twice the old
    internode, and each side shoot 5 short nodes -- six side shoots rather than
    three, each long enough to hang catkins from its upper half.

    Nothing has to be assigned for the catkins to appear -- they are modelled,
    not instanced -- so this species reads correctly out of the box, unlike
    every other one here, which needs its leaf and flower collections set.
    """
    return MorphologyProfile(
        name="Corylus avellana",
        # Hazel catkins come very early; simplified here to early in the modelled
        # season (day ~105), since the sim only grows structure from emergence.
        calendar=_calendar(bloom_start=15.0, bloom_days=45.0),
        levels=[
            _clump(1, 0.0, steps=6,
                   start_length_scale=0.057, length_scale_per_step=1.05,
                   width_scale_per_step=0.874,
                   start_width_scale=0.0045, length_variation=0.18),
            BranchLevel(start_pitch_deg=48.0, steps=5,
                        start_length_scale=0.3,
                        start_width_scale=0.8, width_scale_per_step=0.8, length_variation=0.18),
        ],
        # The catkins below are the whole inflorescence; the stems bear none.
        inflorescence=Inflorescence(
            kind=InflorescenceKind.CATKIN,
            # The three layers, stated apart: what it is, where it is borne,
            # which way it points.
            attachments=frozenset({Attachment.AXILLARY}),
            # Was WORLD_ANGLE at 158 degrees from vertical. Pendulous at a
            # strength just short of 1 says the same thing in the new terms --
            # a catkin hangs, but off a stalk that still shows where it left
            # the twig rather than dropping to exact vertical.
            orientation_mode=OrientationMode.PENDULOUS,
            orientation_strength=0.85,
            placement=Placement(on_trunk=False, start=0.45, end=1.0)),
        # Spiral foliage, as hazel actually has. How many catkins share a stalk
        # is `catkin_count`, not the leaf arrangement.
        catkin_length=0.075,
        catkin_length_variation=0.25,
        catkin_width=0.0045,
        catkin_stalk=0.018,
        catkin_noise=0.3,
        catkin_droop_variation=0.08,
        # A twig, not a tree: short internodes and a fine stem.
        internode_length=0.038,
        # Leaf size defaults to 1.0 (a mesh-driven multiplier); leaf_weight
        # carries the old size**2 so leaf mass -- and so droop -- is unchanged.
        leaf_weight=8.0e-5 * 0.055 ** 2,   # default weight, was leaf_size 0.055
    )


def laburnum_anagyroides() -> MorphologyProfile:
    """*Laburnum anagyroides* -- golden chain, as a flowering branch.

    The species the three-layer split exists for, and the first here whose
    inflorescences are **borne along its stems rather than at their tips**. Its
    racemes hang in long chains from buds in last year's leaf axils, while the
    shoots carrying them go on being shoots and bear no flowers themselves.

    Stated as the three layers, which is the whole point -- none of them is a
    property of the others:

    * **Morphology** `RACEME`: flowers on short pedicels along a continuing
      axis. Exactly a foxglove's, and nothing about it says which way up.
    * **Attachment** `AXILLARY`: released at nodes on its own short peduncle.
      Before this existed a raceme could only terminate a shoot, so the only
      way to get a hanging one was to hang the *shoot* -- which measured as the
      branch drooping, not the flowers.
    * **Orientation** `PENDULOUS` at full strength: the raceme hangs from the
      moment it emerges rather than starting erect and bowing as it fills.

    **Branching compromise.** The old branch had four side shoots on a 12-node
    leader (density 0.6 over the upper 60%). Every node now branches, so the
    leader has 6 Steps of about twice the old internode and each side shoot 8
    nodes -- six side shoots, with racemes along the upper part of all of them
    (27 racemes, as before).

    Not modelled: the trifoliate leaf, which is a leaf-asset matter, and the
    fact that every part of the plant is poisonous, which is not a geometry
    problem.
    """
    return MorphologyProfile(
        name="Laburnum anagyroides",
        # Golden chain: a short, showy late-spring bloom (day ~115) of hanging racemes.
        calendar=_calendar(bloom_start=25.0, bloom_days=35.0),
        levels=[
            _clump(1, 0.0, steps=6,
                   start_length_scale=0.0374, length_scale_per_step=1.05,
                   width_scale_per_step=0.87,
                   start_width_scale=0.019, length_variation=0.18),
            BranchLevel(start_pitch_deg=44.0, steps=8,
                        start_length_scale=0.4,
                        start_width_scale=0.8, width_scale_per_step=0.85, length_variation=0.18),
        ],
        inflorescence=Inflorescence(
            kind=InflorescenceKind.RACEME,
            attachments=frozenset({Attachment.AXILLARY}),
            # Was WORLD_ANGLE at 180. A laburnum chain hangs from the moment
            # it emerges, so it takes the full pull.
            orientation_mode=OrientationMode.PENDULOUS,
            orientation_strength=1.0,
            placement=Placement(on_trunk=True, on_branches=True, start=0.35, end=1.0),
            flower_count=24,
            # A chain is long and closely set: many flowers, short internodes
            # between them, and only a couple of bare nodes before they start.
            internode_scale=0.26,
            branch_nodes=2,
            pedicel_length=0.011,
            pedicel_angle_deg=-25.0,
            pedicel_width=0.22,
            flower_size_profile=Curve.from_points([(0.0, 1.0), (1.0, 0.72)]),
        ),
        internode_length=0.034,
        # Sizes default to 1.0 (a mesh-driven multiplier); the weights carry the
        # old size**3 / size**2 so mass -- and so droop -- is unchanged.
        flower_weight=4.0e-4 * 0.32 ** 3,   # default weight, was flower_size 0.32
        leaf_weight=8.0e-5 * 0.05 ** 2,     # default weight, was leaf_size 0.05
    )


def lycopodium() -> MorphologyProfile:
    """*Lycopodium* -- a clubmoss, and the reason WS2 exists.

    The one worked species whose branching is **apical dichotomy**: the growing
    tip forks into two equal halves, over and over, which no amount of lateral
    branching can imitate. It is also the only species here that flowers
    **never** -- clubmosses are fern-allies with no flowers at all -- so its
    `inflorescence` is `NONE` and its whole structure is the fork.

    One level, the stem, with Fork on -- and no branch level, because
    clubmosses have no axillary meristems (the seed plants' invention), so
    apical forking is the *only* branching they have. The values that carry the
    character:

    * **Fork Probability 1.0.** A tip forks as soon as it may (its third node),
      so the stem divides again and again rather than running as a long stalk
      that splits a few times near the top. The old fork interval was 5 nodes
      of the axis budget; each fork now needs three nodes of its own.
    * **Fork Whorl 75.** Enough that successive fork planes rotate into a
      three-dimensional bush (*Lycopodium*) rather than a flat fan
      (*Selaginella*), which a whorl near 0 would give.
    * **Steps 20** is the whole budget the forks share: six fork generations
      (2+4+...+64 = 126 fork axes), as the old branch-order cap of 5 did.
      A fork continues its parent's internode taper, so Length / Step 0.93
      shortens each generation's internodes, as the old forks' did.

    Foliage-only, so unlike the flowering species it needs no flower
    collection -- only a leaf collection for its microphylls.
    """
    return MorphologyProfile(
        name="Lycopodium",
        levels=[
            _clump(1, 0.0, steps=20,
                   start_length_scale=0.02, length_scale_per_step=0.93,
                   width_scale_per_step=0.894,
                   growth_pattern="dichotomous", fork_branches=2, fork_probability=1.0,
                   fork_spread_deg=28.0, fork_whorl_deg=75.0,
                   start_width_scale=0.025, length_variation=0.18),
        ],
        # NONE: the stems bear nothing and simply stop -- a clubmoss never
        # flowers. `produces_flowers` is False, so the whole plant is its fork.
        inflorescence=Inflorescence(kind=InflorescenceKind.NONE),
        # A fine, upright forking stem carrying small microphylls.
        # Leaf size defaults to 1.0 (a mesh-driven multiplier); leaf_weight
        # carries the old size**2 so leaf mass -- and so droop -- is unchanged.
        leaf_weight=8.0e-5 * 0.02 ** 2,   # default weight, was leaf_size 0.02
    )


def araucaria() -> MorphologyProfile:
    """*Araucaria* -- a monkey-puzzle: a **reliable whorl** of branches.

    Every tier up the trunk throws exactly the same number of branches. Three
    controls carry the whole tiered habit, and each is a real choice:

    * **The branch level's Whorl is 4** -- four branches a node, spaced 90
      degrees apart, each tier turned 45 degrees from the one below, so every
      whorl is a clean tier of four.
    * **The leaf arrangement is a whorl of 5** -- the foliage: five leaves a
      node. The branch level lays its branches out on its own slots, so the
      leaf whorl stays full regardless.
    * **The stems' first internode is 0.18 m** (Start Length), lengthening
      1.05x a node -- long, and this is the tier *spacing*. There is no
      separate tier control: whorls sit at every node, and a long internode is
      what spaces them up a bare trunk into Araucaria's tiers.

    Foliage-only (`InflorescenceKind.NONE`, `produces_flowers` False) -- a
    conifer, and the model does not do cones, so like the clubmoss it simply
    never flowers and its whole character is the form. Two levels keep the
    branches bare rather than bearing branchlets; a third would give each
    branch its own sprays at an exponential cost. The old plant's top tiers
    were cut short by its fixed growth span; now every branch builds its full
    7 Steps, so the crown stands ~0.4 m taller (3.66 m against 3.23 m).
    """
    return MorphologyProfile(
        name="Araucaria",
        levels=[
            _clump(1, 0.0, steps=12,
                   start_length_scale=0.1806, length_scale_per_step=1.05,
                   width_scale_per_step=0.883,
                   start_width_scale=0.1, length_variation=0.18),
            BranchLevel(phyllotaxis=LevelPhyllotaxis.CUSTOM, spiral_angle_deg=45.0, whorl=4,
                        start_pitch_deg=48.0, steps=7,
                        start_length_scale=0.727,
                        start_width_scale=0.8, width_scale_per_step=0.72, length_variation=0.18),
        ],
        inflorescence=Inflorescence(kind=InflorescenceKind.NONE),
        leaf_arrangement=LeafArrangement(spiral_angle_deg=36.0, whorl=5),
        # Leaf size defaults to 1.0 (a mesh-driven multiplier); leaf_weight
        # carries the old size**2 so leaf mass -- and so droop -- is unchanged.
        leaf_weight=8.0e-5 * 0.03 ** 2,   # default weight, was leaf_size 0.03
    )


SPECIES = {
    "GENERIC": generic_perennial,
    "MEADOW_FOXTAIL": meadow_foxtail,
    "RUDBECKIA": rudbeckia_fulgida,
    "ANEMONE": anemone_hupehensis,
    "FOXGLOVE": digitalis_purpurea,
    "HAZEL": corylus_avellana,
    "LABURNUM": laburnum_anagyroides,
    "LYCOPODIUM": lycopodium,
    "ARAUCARIA": araucaria,
}
"""Every species by key. `bridge` and the panel both read this, so adding one
here is the only edit a new species needs on the growth side."""


def profile(key: str) -> MorphologyProfile:
    """A fresh profile for `key`, falling back to the generic perennial for an
    unknown key rather than raising -- a stale key on an object from an older
    file should degrade, not break the build."""
    return SPECIES.get(key, generic_perennial)()
