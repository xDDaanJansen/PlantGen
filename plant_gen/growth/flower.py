"""Flower morphology skeleton — Milestone 1: Actinomorphic/Rotate, Solitary.

N petals per layer, 1+ vertically-layered whorls radially symmetric around a
central disc (forget-me-not/buttercup-style, or a denser double-flower look with
multiple petal layers), no corolla tube yet -- each component (petal placement,
petal shape, disc) is independently parameterized, following the "parameterize
each component in its individual category" direction: petals get their own
SpeedTree-style shape controls (a full 3-axis Curl + Gravity + Twist, mirroring
`skeleton/stem.py`'s own Curl/Gravity/Twist split almost exactly) and the disc
gets its own radius-vs-height profile curve (mirroring the stem's own taper)
instead of a fixed primitive.

Sepals (the calyx) are a SECOND, independent whorl of the same kind of ribbon
appendage as petals -- same full Curl/Gravity/Twist/Rotation/Layers/Scale
parameter suite, just its own count/size/shape/position, usually sitting below
the petal ring. Rather than duplicating the whorl-building math, both whorls are
built by `_build_petal_group`, parameterized by a `_PetalGroupSpec` bundle (see
below) -- `build_flower_mesh` just calls it twice, once per whorl.

The Receptacle is a SECOND, independent revolved body of the same kind as the
Disc (radius-vs-height profile, see `_build_disc`/`DiscSpine`) -- it reuses
`_build_disc` directly rather than duplicating that math too, always swept
downward from the shared Z=0 attachment rim (a negative height already means
"sweep downward" for the Disc's own concave case, so no new geometry code is
needed, just a different caller).

Reuses the exact stem seam (see `blender/flower.py`'s module docstring) for
petals/sepals/disc/receptacle alike: each
is authored as a poly spline whose native per-point `radius` is repurposed as
ribbon width / revolved-body radius, exactly like `stem.py` computes taper. Petal
(and sepal) twist rides on the spline's native per-point `tilt` (the built-in
Blender mechanism for rotating a swept profile's frame along a curve) rather than
a hand-rolled transported frame.

Pure module: no bpy.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field

from .flower_params import Parameter, Ramp
from .flower_params import varied as _varied
from .flower_vecmath import (
    Vec,
    add,
    cross,
    length,
    normalize,
    ring_azimuths,
    rotate_axis,
    rotate_toward,
    scale,
)

_TURN = math.pi  # 1.0 param unit == 180 degrees, same convention as skeleton/stem.py
_MIN_PETAL_COUNT = 3.0
_MIN_LENGTH = 0.0001
_MIN_RADIUS = 0.0001
# Same constant `skeleton/phyllotaxis.py` defines for stem-level "golden" phyllotaxis
# (~137.5°) -- duplicated locally rather than importing that sibling module, since
# the two skeleton modules otherwise have no dependency on each other.
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
# Point count of the ribbon cross-section profile (the arc `blender/flower.py`'s
# `_ribbon_mesh` sweeps for Petals/Sepals) -- MUST match that profile's own
# `Resolution` input exactly, since `_build_shape_carrier` below authors one
# value per profile point for the Top/Bottom Shape mechanism (see
# `FlowerMeshParams.petal_top_shape_profile`'s docstring), and Geometry Nodes
# reads that array positionally (by index), not by any shared symbolic key.
_RIBBON_PROFILE_RESOLUTION = 16
# Default Top/Bottom Shape cap curve -- a sampled semicircle over width
# position (`sqrt(1 - 4u^2)` at u = +/-0.25 is 0.866), so amount 0.5 gives a
# truly circular cap rather than the pointier parabola a 3-point curve gives.
# Defined HERE, the lowest bpy-free module in the stack, and imported by both
# `blender/flower.py` (as the curve widget's default) and `operators.py` (as
# FLOWER_CAP_SHAPES' own "dome" entry) so the three can't drift apart.
DOME_CAP = [(0.0, 0.0), (0.25, 0.87), (0.5, 1.0), (0.75, 0.87), (1.0, 0.0)]


@dataclass
class FlowerMeshParams:
    # --- development: two independent flower-global "over time" master axes,
    # distinct from each other AND from the existing per-layer Age (which stays a
    # within-flower layer-index thing, see the petal_layers docstring):
    #
    # * Bloom (0 = closed bud -> 1 = open flower): blends the CURL UP/DOWN bend
    #   amount that feeds `_build_petal_spine`'s integrator between a "bud" value
    #   (this whorl's own `{w}_bud_curl` + `{w}_bud_curl_profile`) and the
    #   authored open-pose value (`{w}_curl_updown` + its profile) -- see
    #   `_build_petal_group`'s `local_bloom`/bend blending. Only ONE centerline is
    #   ever built (unlike the earlier design, which lerped POSITIONS between two
    #   independently-built curves -- an envelope and the open pose -- and caused
    #   clipping in the transition zone since the two curves don't correspond).
    #   Blending the bend itself instead means Bloom rising IS the petal/sepal
    #   physically uncurling, continuously, with no second geometry involved.
    #   Each whorl maps the shared `bloom` through its own `{w}_bloom_start`/
    #   `{w}_bloom_end` window (see those fields below) so petals and sepals
    #   close/open TOGETHER (one slider) while opening on their own schedule
    #   (sepals first, by default) -- that's the "linked but still independently
    #   tunable" ask. Default petal window (0, 1) reproduces the pre-Bloom
    #   behavior exactly at bloom == 1.0 (backward-compatible default).
    # * Maturity (0 -> 1): a global uniform SIZE multiplier, shaped by its own
    #   curve, applied as a single final scale pass over the whole flower (see
    #   `build_flower_mesh`) -- independent of how open the flower is. Default
    #   curve rises 0.8 -> 1.0 (same convention as `petal_scale_age_profile`),
    #   so the Maturity slider visibly does something out of the box -- an
    #   earlier flat default (a ×1.0 no-op everywhere) was "safe" but read as
    #   the slider simply being broken, since nothing about a flat curve hints
    #   that it's the thing you're meant to reshape.
    bloom: float = 1.0
    maturity: float = 1.0
    maturity_scale_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.8), (1.0, 1.0)]))

    # Unlike sepal_enabled/receptacle_enabled below, Petals default ON -- they're
    # the flower's primary/always-present structure, not an optional add-on.
    petal_enabled: bool = True
    # --- petals: placement + width -------------------------------------------
    petal_count: int = 5                     # petals PER LAYER (see petal_layers)
    petal_count_variance: float = 0.0        # absolute, global reroll (varied()), floor=3
    petal_length: float = 0.02
    petal_length_variance: float = 0.0       # per-petal (element_id=idx)
    petal_width: float = 0.01                # full width at the profile's peak
    petal_width_variance: float = 0.0        # per-petal (element_id=idx)
    petal_width_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.3), (0.35, 1.0), (1.0, 0.0)]))  # 0=base (disc edge) -> 1=tip
    # Bows the CROSS-SECTION (across the width, not along the length) into a
    # shallow channel/cup/dome -- a real distance push on the already-meshed
    # geometry (see `blender/flower.py`'s `_apply_cup`), scaled by the ring's
    # own ACTUAL width so it fades to exactly 0 wherever the ribbon itself
    # tapers to nothing (the tip, and a narrow base), the same single-dial
    # "round the whole thing off" role SpeedTree-style Cup plays on a leaf
    # blade (`blender/plant.py`'s `leaf_cup`) -- kept under this SAME name for
    # that reason (was briefly split into a separate width-INDEPENDENT
    # "Curvature" field and a "Cup" field; merged back into one, since once
    # Cup was fixed to taper correctly with width there was no longer a
    # meaningful difference between the two, just one control instead of two
    # that did almost the same thing). Deliberately NOT a Ramp/hand-drawn
    # profile like Top/Bottom Shape -- a fixed symmetric parabola (edges curl
    # more than the midline), one shared value for every petal (a GLOBAL
    # reroll like disc_radius_variance, not per-petal -- the width profile is
    # one shared GN Curve-to-Mesh call across every petal spline at once, so
    # only a per-OBJECT value, not per-petal, is reachable without a separate
    # meshing call per petal).
    petal_cup: float = 0.0
    petal_cup_variance: float = 0.0
    # With the disc rim effectively pinched down to a point in a closed bud (see
    # `petal_bud_sink`), a petal's own cross-section reads flatter than it
    # should -- the SAME cup amount that looks right laid out flat at full
    # radius looks under-curved once everything's compressed toward the axis.
    # A true pair with `petal_cup` now (this whorl's own local bloom -- see
    # `_local_bloom` -- lerps FROM this value TO the authored open-pose Cup),
    # the same pattern as Bud Pitch/Bud Curl/Bud Sink, rather than a
    # multiplier on top of it -- an independently authored value composes
    # more predictably than a multiplier that silently changes sign/scale
    # behaviour whenever the open Cup itself changes. 0 = flat at full closure.
    petal_bud_cup: float = 0.0
    resolution: int = 8                      # samples per petal spine (>= 2)
    # Points ACROSS each petal (the swept profile's resolution) -- `resolution`
    # above is along it. The tube a welded whorl is revolved with takes
    # `count * (width_resolution - 1)` sides to match.
    petal_width_resolution: int = _RIBBON_PROFILE_RESOLUTION
    # signed radial offset of each petal's BASE from the disc rim -- + pulls it
    # inward toward the disc's own axis, - pushes it outward past the rim (like a
    # pedicel); same signed-distance convention as PhyllotaxisParams.sink.
    petal_sink: float = 0.0
    petal_sink_variance: float = 0.0         # per-petal (element_id=idx)
    # signed Z offset of each petal's BASE -- + raises it, - lowers it. A rigid
    # translation of the WHOLE petal (the base moves, then every subsequent
    # spine point follows since the spine is integrated forward from the base).
    # Exists to let the base meet a Fusion tube's own top rim, or the Disc's
    # own surface, exactly -- `_build_petal_group` already places the base at
    # the tube's rim height (`fusion_height`) automatically when Fusion > 0,
    # but curvature/width/pitch differences between the tube's own profile and
    # the lobe's cross-section can still leave a visible gap or overlap there;
    # this is the by-hand fix for that, the same role Radial Offset plays for
    # gaps in the horizontal plane.
    petal_vertical_offset: float = 0.0
    petal_vertical_offset_variance: float = 0.0  # per-petal (element_id=idx)
    # random scatter of each petal's BASE azimuth around its ideal evenly-spaced
    # ring position -- same "spread_horizontal" name/mechanism as
    # PhyllotaxisParams (a magnitude Parameter + its own deterministic RNG per
    # attachment), and distinct from petal_rotation_yaw below: this jitters WHERE
    # the base sits on the ring, yaw jitters which way the petal POINTS from
    # wherever its base ends up.
    petal_spread_horizontal: float = 0.0     # radians, max random azimuth scatter
    petal_spread_horizontal_variance: float = 0.0  # per-petal variance of that magnitude
    # Around the Whorl: multipliers over a petal's place on the ring, measured
    # both ways from the first petal (0 = that petal, 1 = straight opposite),
    # so any shape is mirror-symmetric -- a bilateral (zygomorphic) corolla:
    # a pea flower's standard at 0, wings and keel further round. Flat at 1
    # (the default) builds the radial whorl unchanged.
    petal_whorl_length_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    petal_whorl_width_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    petal_whorl_pitch_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # Divides the width at which neighbouring petals' edges exactly meet (see
    # `_touching_width`), so it keeps meaning something past 1: 0 = off (they
    # overlap freely, as many real polypetalous flowers do), 1 = edges land
    # exactly on each other, below 1 permits proportionally more overlap, above
    # 1 pulls them narrower than touching and opens a deliberate gap. Applied
    # along the WHOLE length, not only at the base, since the available room
    # grows with distance from the axis -- see `_build_petal_group`. This is
    # the OPEN-pose value; see `petal_bud_clamp_overlap` below for its own
    # bud-pose counterpart.
    petal_clamp_overlap: float = 0.0
    # A shared corolla TUBE (Sympetalous/Gamopetalous) this whorl's lobes
    # emerge from, before continuing as free ribbons -- disabled by default
    # (today's Polypetalous-only free-petal behavior). The tube is a revolved
    # body built the SAME way as the Disc/Receptacle (see
    # `_build_disc`/`DiscSpine`): one shared surface for the whole whorl, not
    # per-petal. Deliberately given its OWN independent Length/Radius/
    # Resolution/Profile (mirroring Disc/Receptacle's own field group exactly)
    # rather than being derived from `petal_length`/`disc_radius` (an earlier
    # design, changed after the user pointed out it coupled the tube's size to
    # the petal's own -- adjusting one shouldn't move the other). Applied
    # uniformly across every layer when `petal_layers` > 1 (a known
    # simplification: only one tube is built, spanning layer 0's own Z range,
    # so upper layers sit above it rather than each growing their own tube).
    # See `_build_petal_group` for exactly how each lobe's base is raised to
    # meet the tube's own top rim. `petal_tube_enabled == False` reproduces
    # today's free-petal geometry exactly -- a strict generalization, not a
    # new mode: `tube_top_radius` bypasses the tube fields entirely and equals
    # the disc rim radius directly in that case (see `_build_petal_group`).
    petal_tube_enabled: bool = False
    petal_tube_length: float = 0.008
    petal_tube_length_variance: float = 0.0  # whorl-wide, global reroll (like disc_radius)
    # When True (the default), the tube's base radius IS `disc_radius` -- the
    # tube meets the disc rim seamlessly and the whole flower resizes off one
    # master dimension, which is what you want essentially every time. The
    # independent `petal_tube_radius` below is still there for the cases where
    # you deliberately want them apart, it just isn't the thing you have to
    # keep re-matching by hand. NOTE this links only the tube's BASE: its own
    # `petal_tube_profile` still shapes the radius from there up, so a funnel
    # still flares. Old presets predate this flag and carry a radius they chose
    # on purpose, so `model/io.py`'s flower v1->v2 migration turns it OFF for
    # them rather than letting them silently snap to the disc.
    petal_tube_match_disc: bool = True
    petal_tube_radius: float = 0.006
    petal_tube_radius_variance: float = 0.0
    petal_tube_resolution: int = 8           # samples along the tube's own profile (>= 2)
    # Slow radius wobble along the tube (0 = a perfectly machined extrusion) --
    # see `_build_disc`/`_smooth_noise`. Rerolls with the flower's Seed, so a
    # field of the same species varies instead of repeating one silhouette.
    petal_tube_noise: float = 0.0
    # Start the free lobes off at the tube's own mouth angle, so they read as a
    # continuation of the wall rather than straps stuck on at an unrelated
    # tilt -- and so reshaping the top of the Tube Profile carries the lobes
    # with it instead of needing Pitch re-eyeballed. `petal_rotation_pitch`
    # becomes an OFFSET from that angle when this is on (see
    # `_build_petal_group`); the corolla presets' own pitch values are
    # authored as such offsets.
    petal_tube_match_angle: bool = False
    # Widen each lobe's BASE until `petal_count` of them tile the tube's mouth
    # exactly, closing the gaps between lobes so a sympetalous corolla reads as
    # one piece. Fades back into the authored outline over `_FUSE_SPAN`.
    petal_tube_fuse: bool = False
    # Tube's own radius-vs-height shape (Tubular/Funnel/Bell/Urn/etc -- see
    # blender/flower.py's FLOWER_COROLLA_SHAPES), same convention as
    # `disc_profile`/`receptacle_profile`.
    petal_tube_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # The tube through Bloom (the bud pose, like Bud Pitch / Bud Curl): its
    # length, and how much of the Tube Profile's widening it has, as fractions
    # of the open tube's -- 1.0 (the default) keeps the tube the same at every
    # Bloom. Flare scales the profile's deviation from 1, so the base keeps its
    # radius (disc, receptacle and welds stay matched) while a slim bud swells
    # into the bell. Blended over the whole Bloom slider (0 = bud, 1 = open),
    # not this whorl's lobe window: a corolla lengthens before it opens.
    petal_tube_sides: int = 16              # around the tube (unwelded; welded: from the lobes)
    petal_bud_tube_length: float = 1.0
    petal_bud_tube_flare: float = 1.0
    # Overall size multiplier (scales both Length and Width together), driven by
    # `petal_scale_age_profile` vs this petal's own Age (see the layering section
    # below for how Age is derived from layer index) -- e.g. lets the topmost/
    # innermost layer of a multi-layer flower read as younger and smaller, the
    # way inner whorls often are on a real many-petaled flower.
    petal_scale: float = 1.0
    petal_scale_variance: float = 0.0        # per-petal (element_id=idx)
    petal_scale_age_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.8), (1.0, 1.0)]))  # 0=youngest (top/inner) -> 1=oldest (bottom/outer)
    # EXPERIMENTAL (2026-07-20): a real flower's bloom reads as mostly a rigid
    # PITCH hinge (the whole whorl swinging from pointing up to pointing out,
    # around its own base) with only a minor amount of actual Curl unfurling on
    # top -- maybe a 70-80/20-30 split, not the ~50/50 the Curl-driven design
    # above defaulted to. Bud Pitch is that hinge's closed-bud angle (see
    # `_build_petal_group`'s pitch blend, mirroring the Curl blend exactly:
    # lerped with the authored open-pose Rotation Pitch by local_bloom, 1.3 rad
    # (~74 degrees, near-vertical) at local_bloom 0 fading to the open value at
    # 1). Unlike Curl's per-step integration, a rigid rotation from ~74 degrees
    # down to the open pitch is INHERENTLY monotonic and well-behaved -- the
    # base stays fixed on the rim, so there's no "wrap past itself and swing
    # back toward the axis" failure mode at all, regardless of the angle. With
    # Pitch now doing the bulk of the closing motion, Bud Curl only needs to
    # add a little extra shape on top -- reduced accordingly (was 0.5).
    petal_bud_rotation_pitch: float = 1.3
    # Bud closure for petals (see the `bloom` docstring at the top): extra Curl
    # Up/Down bend fully applied when this whorl's own local bloom is 0, fading
    # to zero extra (i.e. exactly `petal_curl_updown`) as local bloom reaches 1.
    # Plain scalar, no variance/age -- same scope as `petal_curl_roll`. Keep
    # this under roughly 0.6-0.7 with the default `petal_bud_curl_profile` --
    # much higher and the accumulated bend overshoots 180 degrees, so the tip
    # swings PAST vertical and back toward/through the central axis as bloom
    # starts to relax it, which reads as the tip pulling back before it
    # unfurls outward rather than a clean monotonic unfold (confirmed
    # numerically: at 1.0 the tip's radial distance from the axis is non-
    # monotonic across the bloom sweep; at 0.5 it isn't). `petal_bud_tip_*`
    # below is the actual pointed-vs-round lever now, so this only needs to
    # fold the whorl into a rounded body, not wrap it past itself. Now that
    # Bud Pitch (above) carries most of the closing motion, this is reduced
    # further still (was 0.5) -- just a little extra curve on top of the hinge.
    petal_bud_curl: float = 0.15
    # Dedicated along-length profile for the bud bend ONLY (independent of
    # `petal_curl_updown_profile`, which stays tuned for the open look). This is
    # the actual pointed-vs-ball lever: the default keeps the base at roughly the
    # same bend strength as a flat curl_updown of 1.0 (so the whole petal still
    # folds all the way up, not just the tip -- a weak base leaves the rim loose
    # even at bloom == 0) while biasing EXTRA bend toward the tip on top of that,
    # so the tip whips inward and converges to a point a plain uniform bend can't
    # reach (constant curvature traces a circular arc, which is inherently round
    # -- the reason pushing Curl directly always "ended up a ball").
    petal_bud_curl_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 1.0), (0.5, 1.1), (1.0, 1.5)]))  # 0=base -> 1=tip
    # Radial pull on the petal's own BASE (same units/sign as `petal_sink`: +
    # pulls inward toward the disc axis) fully applied at local bloom 0, fading
    # to zero extra (i.e. exactly `petal_sink`) as local bloom reaches 1. Curl
    # alone converges each petal's TIP to a point but leaves its base at the full
    # open-flower rim radius -- a ring of individually-pointed petals still reads
    # as a wide spiky crown, not a slender bud, because a real bud also presses
    # the bases close together near the axis. Blended the same way as
    # `petal_bud_curl` (see `_build_petal_group`), just a plain lerp since sink
    # is a scalar, not a length-profile.
    petal_bud_sink: float = 0.005
    # `petal_clamp_overlap`'s own bud-pose counterpart -- a true pair, same
    # lerp-by-local-bloom pattern as Bud Sink above (see `_build_petal_group`).
    # Bud Sink already pulls every base toward the axis as the whorl closes,
    # which shrinks the touching-width ceiling `clamp_overlap` measures
    # against; without its own bud value, the single open-pose Clamp Overlap
    # stays fixed while that ceiling shrinks under it, so widths authored for
    # the open radius can pile up and cover the axis instead of reading as
    # separate furled petals. 0 = off (matches the open default).
    petal_bud_clamp_overlap: float = 0.0
    # Where along the spine (0=base -> 1=tip) the tip-shaping zone -- everything
    # below -- starts. `petal_bud_tip_height_profile`/`petal_bud_tip_radius_profile`
    # are evaluated on a LOCAL 0..1 parameter rescaled to just this zone (0 at
    # this point, 1 at the very tip), so the curve widgets always get full
    # resolution to shape the tip regardless of where the zone starts, rather
    # than being squeezed into a sliver of a 0..1 editor.
    petal_bud_tip_start: float = 0.5
    # Overall Z magnitude raised into the tip zone, fully applied at local bloom
    # 0, fading to zero as local bloom reaches 1 -- multiplies
    # `petal_bud_tip_height_profile`'s own 0..1 shape (same "magnitude scalar x
    # normalized profile" convention as Curl/Gravity/etc. above). Curl alone
    # produces a constant-curvature circular arc no matter how it's shaped by
    # `petal_bud_curl_profile` -- inherently round, since every spine point sits
    # right on that arc. 0 = off (this whorl's tip stays wherever Curl leaves it).
    petal_bud_tip_lift: float = 0.0
    # How much of `petal_bud_tip_lift` applies at each point in the tip zone (0
    # at the zone's start, 1 at the tip) -- e.g. a smooth ease-in (the default)
    # gives a curved, not creased, lead-in into the raised apex; a curve that
    # jumps early gives a blunter tip; one that stays near 0 until near the very
    # end gives a sharper, more sudden point.
    petal_bud_tip_height_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.0), (0.3, 0.06), (0.6, 0.3), (1.0, 1.0)]))
    # Radius MULTIPLIER at each point in the tip zone (1.0 = unchanged -- exactly
    # where Curl/Sink already put it; the default, a no-op). This is the actual
    # general-purpose tip-SHAPE control: Height alone can only raise the
    # existing curl-driven arc, which still reads as round in cross-section --
    # pulling the radius in TOWARD the axis (< 1) as it rises gives an actual
    # narrowing neck (a teardrop/bulb silhouette instead of a raised bump), and
    # pushing it OUT first (> 1) before narrowing gives a bulging teardrop belly.
    # Fully applied at local bloom 0, fading to no-op (1.0) as local bloom
    # reaches 1, same as everything else here. Shares the Disc shape-preset
    # library (see blender/flower.py's FLOWER_CURVE_SPECS) since round/pointed/
    # bulging/waisted are exactly the same silhouette vocabulary.
    petal_bud_tip_radius_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # How fast the Height/Radius shaping above releases as this whorl opens:
    # `tip_fade = (1 - local_bloom) ** this`, so 1.0 is a plain linear release
    # across the FULL local bloom range (matching Curl's own pace) and higher
    # values front-load the release -- most of it happens early, easing
    # smoothly (not snapping) toward zero as local bloom approaches 1.
    #
    # Earlier attempts used a SEPARATE, harder release window (a fixed local-
    # bloom fraction where the effect hit exactly 0) instead of this. Two
    # things went wrong with that: (1) even smoothstep-eased, a window that
    # ends before local bloom 1 is a genuinely DIFFERENT timeline than Curl's
    # own 0->1 unwind, so the tip's "un-pinching" motion and the rest of the
    # petal's Curl-driven swing outward were paced independently of each
    # other -- with the window compressed into ~30% of the sweep, the tip
    # visibly snapped from "sharp bud point" toward "wherever Curl currently
    # has it" (which, that early, is still mostly closed) largely BEFORE Curl
    # itself had swung outward at all, reading as the tip pulling back/
    # retreating rather than unfurling. (2) a hard window, however eased,
    # still forces the SAME clamp-at-zero shape. A power curve is smooth
    # (C-infinity) everywhere over the single 0->1 range Curl already uses --
    # no separate window to desync, and no clamp to kink at.
    petal_bud_tip_falloff: float = 3.0
    # This whorl's own window over the shared `bloom` master: local bloom is 0
    # below `bloom_start`, 1 above `bloom_end`, linearly in between. Petals
    # default to NOT starting until the calyx is 80% of the way through its own
    # opening -- `bloom_start = 0.44` = 0.8 * `sepal_bloom_end` (0.55), so petals
    # sit fully closed while sepals open, then unfurl over the remainder of the
    # sweep up to `bloom_end = 1.0` -- "sepals first, then petals follow", with
    # petals waiting for sepals to be almost done rather than starting alongside
    # them.
    petal_bloom_start: float = 0.44
    petal_bloom_end: float = 1.0

    # --- petals: layering (stacked whorls, e.g. "3 layers top/middle/bottom") -
    # A botanical stand-in for double/many-petaled flowers: each layer is its own
    # full ring of `petal_count` petals, offset vertically from the previous layer
    # and rotated by `petal_layer_divergence` (defaults to the golden angle, same
    # constant `skeleton/phyllotaxis.py` uses for "golden" stem phyllotaxis) so
    # successive layers interleave rather than stack directly on top of each other
    # -- the same "stagger by an angle between whorls" idea as PhyllotaxisParams'
    # whorled mode, just applied radially around a disc instead of along a stem.
    petal_layers: int = 1                    # number of stacked layers (1 = off)
    petal_layer_spacing: float = 0.0         # Z distance between successive layers
    petal_layer_spacing_variance: float = 0.0  # per-petal random Z jitter within its layer
    petal_layer_divergence: float = _GOLDEN_ANGLE  # azimuth step applied per layer
    # Every petal's Age is derived from its OWN layer index -- 1.0 (mature) for
    # the bottom/outermost layer, fading to 0.0 (youngest) for the top/innermost
    # one (a single-layer flower is just always 1.0, fully mature). This reuses
    # `model.parameter.Parameter`'s EXISTING `age_ramp` mechanism (the same
    # 0=seedling/1=mature convention `model/parameter.py` already documents for
    # stems/leaves) rather than inventing a new one -- see the `age_profile`
    # fields below and in the Shape section, and `_build_petal_group`'s `age_i`.
    # No public field here: Age itself isn't user-authored, only how much each
    # trait responds to it (via that trait's own `_age_profile` curve).

    # --- petals: shape (SpeedTree-style, mirrors StemParams almost exactly) --
    # Three FIXED (not co-rotating) bend axes, same model as the stem's own Parent
    # Curl: Up/Down = perpendicular to the petal's own initial direction and world
    # Z: Left/Right = the petal's own initial direction; Out/In = world Z (there's
    # no parent axis yet, same stand-in stem.py uses). Because the axes are fixed
    # rather than tracking the bent tangent, applying all three in sequence still
    # produces genuinely different (non-commuting) shapes, not a redundant 2-axis
    # system -- see skeleton/stem.py's module docstring for the same reasoning.
    # curl/twist are a total angle over the WHOLE petal (1.0 == 180 degrees),
    # independent of Petal Length -- same "Units" convention as the stem's own
    # curl/twist/noise. Gravity is different (bends toward a fixed world target,
    # clamps there) -- also the same convention as the stem.
    petal_curl_updown: float = 0.0           # bends up(+)/down(-) out of the flat
    petal_curl_updown_variance: float = 0.0  # disc plane; uncapped, keeps coiling
    petal_curl_updown_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # Multiplies Curl Up/Down by this petal's own Age (see the layering section
    # above) -- e.g. a gentle downward slope (1.0 at age 0 -> 0.85 at age 1) makes
    # the bottom/outer/mature layer curl ever so slightly LESS than the top.
    petal_curl_updown_age_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    petal_curl_leftright: float = 0.0        # curves sideways (a sickle/scimitar
    petal_curl_leftright_variance: float = 0.0  # shape) as the petal extends outward
    petal_curl_leftright_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    petal_curl_outin: float = 0.0            # corkscrews the lean around the
    petal_curl_outin_variance: float = 0.0   # disc's own central (world Z) axis
    petal_curl_outin_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # A SECOND, independent kind of roll (see petal_rotation_roll below for the
    # first): a one-time rotation of the CURL BEND-AXIS FRAME around the petal's
    # own direction, before any per-step bending starts. Doesn't move the petal
    # by itself (so it has no visible effect at Curl/Gravity == 0, unlike
    # Rotation's own Roll) -- it only redirects which plane the three Curl axes
    # (and Gravity) bend THROUGH afterward, the same role the stem's own Roll
    # plays for its Gravity. The two rolls are independent and can both be
    # nonzero at once: this one reshapes the bend, Rotation's Roll spins the
    # already-built cross-section.
    petal_curl_roll: float = 0.0
    petal_curl_roll_variance: float = 0.0
    petal_gravity: float = 0.0               # bends toward world up(+)/down(-),
    petal_gravity_variance: float = 0.0      # clamps at vertical, can't overshoot
    petal_gravity_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    petal_twist: float = 0.0                 # rotates the swept cross-section
    petal_twist_variance: float = 0.0        # around the petal's own spine (native
    petal_twist_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))  # curve `tilt`)
    # Top/Bottom Shape: the actual shape of the TIP and BASE cap edge -- a
    # magnitude scalar times a curve over WIDTH position (0=one edge -> 1=other
    # edge), a genuinely different axis from `petal_width_profile` ("Sides",
    # which curves the outline over LENGTH position instead). Each width
    # position is pushed OUTWARD beyond today's flat cap along that point's own
    # natural continuation of the ribbon, so a `dome` curve at a positive
    # amount rounds the cap off, a negative amount notches it inward, and an
    # `emarginate` curve gives the bilobed heart-shaped tip the Sides curve
    # alone can never produce. Amount 0 (the default) is a no-op: today's flat,
    # straight-cut cap.
    #
    # The amount is a MULTIPLE OF THAT RING'S OWN WIDTH, not an absolute
    # distance -- 0.5 with a `dome` curve is a semicircular cap at any petal
    # size. That matters for two reasons the absolute version got wrong: cap
    # shaping used to silently vanish when Petal Length/Width grew (and to
    # blow the petal apart at small sizes, since its fixed +/-0.01m range was
    # +/-1x the whole default petal), and it used to fight the width profile
    # at a tapered tip, where forcing a fixed push onto a ring whose width has
    # collapsed to zero tears the geometry rather than shaping it. Ring width
    # is measured LIVE off the meshed ring in Geometry Nodes, so it tracks
    # per-petal Width/Scale variance and per-layer Age scaling for free -- see
    # `blender/flower.py`'s `_push_boundary_ring`.
    #
    # This is NOT computed per-petal in `_build_petal_group` at all -- see
    # `_build_shape_carrier`/`FlowerSkeleton.petal_top_shape`: the curve is
    # sampled once, whorl-wide, at the ribbon cross-section's own fixed point
    # count, and the actual push happens in Geometry Nodes
    # (`blender/flower.py`'s `_ribbon_mesh`) directly on the meshed boundary
    # ring, since Curve to Mesh has no way to vary its swept profile's SHAPE
    # along the sweep -- only Python-baked per-point Scale/tilt survive that
    # conversion (confirmed live: a custom curve-point attribute does not).
    #
    # Bottom Shape is an ADDITIVE override on top of the automatic base fit
    # (see `_base_wrap_carrier`), not a replacement for it -- at amount 0 the
    # base still lands flush on its attachment circle.
    petal_top_shape: float = 0.0
    petal_top_shape_variance: float = 0.0    # whorl-wide, like petal_cup
    petal_top_shape_profile: Ramp = field(default_factory=lambda: Ramp(DOME_CAP))
    petal_bottom_shape: float = 0.0
    petal_bottom_shape_variance: float = 0.0
    petal_bottom_shape_profile: Ramp = field(default_factory=lambda: Ramp(DOME_CAP))

    # --- petals: rotation -- a ONE-TIME base orientation (aviation-style
    # Yaw/Pitch/Roll: around the vertical, lateral, and longitudinal axes),
    # fully distinct from Curl above (which bends PROGRESSIVELY along the
    # petal's length). Yaw turns the petal's pointing DIRECTION around the disc
    # normal, independent of where its base attaches (still evenly spaced on the
    # rim) -- the "rotation from the base"/pinwheel-overlap look. Pitch statically
    # tilts that direction up/down out of the flat plane (the one-time counterpart
    # to Curl Up/Down). Roll spins the cross-section around the petal's own
    # LOCAL direction (its length axis, "sideways") -- a constant offset added to
    # every point's `tilt` (see PetalSpine.tilts), so unlike the old bend-axis-
    # redirect idea this is directly visible even on a perfectly flat petal with
    # zero Curl, exactly like a fixed Twist-from-the-base.
    petal_rotation_yaw: float = 0.0
    petal_rotation_yaw_variance: float = 0.0
    petal_rotation_pitch: float = 0.0
    petal_rotation_pitch_variance: float = 0.0
    # Multiplies Pitch by this petal's own Age (see the layering section above).
    # Since the multiplier can go NEGATIVE, one base Pitch magnitude plus a
    # profile running from +1 (age 0, top/inner) to -1 (age 1, bottom/outer)
    # flips its sign across layers -- e.g. "top layer pitches up a bit, bottom
    # layer pitches down a bit" -- while per-petal Pitch Variance still adds
    # independent random scatter within each layer on top (a symmetric jitter's
    # distribution is unchanged by a sign flip, so it still reads as random,
    # just centered on that layer's own biased mean instead of zero).
    #
    # This field always holds already-remapped -1..1 multiplier values, the
    # same as every other Ramp here. The BLENDER WIDGET for this one curve is
    # the odd one out: it edits in a normal 0..1 range (so the curve widget's
    # grid/zoom stays at its usual scale, editable), and
    # `blender/flower.py`'s `flower_params_from_object` remaps that 0..1 edit
    # range to -1..1 before it ever reaches this field -- see
    # `FLOWER_CURVE_SPECS`'s `y_remap` column. Nothing else in this module
    # needs to know that; it only ever sees the remapped -1..1 form.
    petal_rotation_pitch_age_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    petal_rotation_roll: float = 0.0
    petal_rotation_roll_variance: float = 0.0

    # --- sepals (calyx): a SECOND, independent whorl of the exact same kind of
    # ribbon appendage as petals -- full parameter parity (Count/Length/Width+
    # profile/Curvature/Scale+age/Sink/Spread, Layers, 3-axis Curl+profiles+age+
    # Curl Roll, Gravity, Twist, Rotation Yaw/Pitch+age/Roll). See each `petal_*`
    # field above for the full reasoning -- every `sepal_*` field below means
    # exactly the same thing, just for the calyx whorl instead. Built by the
    # SAME `_build_petal_group` function (see below), just called a second time
    # with these fields bundled into their own `_PetalGroupSpec` -- not a
    # separate implementation. Disabled by default (`sepal_enabled=False`, same
    # convention as the stem's own Leaves/Flowers `enable` checkbox) so adding
    # this field set doesn't change any existing flower's current look.
    sepal_enabled: bool = False
    sepal_count: int = 5
    sepal_count_variance: float = 0.0
    sepal_length: float = 0.008
    sepal_length_variance: float = 0.0
    sepal_width: float = 0.006
    sepal_width_variance: float = 0.0
    sepal_width_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.3), (0.35, 1.0), (1.0, 0.0)]))
    # Mirrors petal_cup -- see its docstring.
    sepal_cup: float = 0.0
    sepal_cup_variance: float = 0.0
    # Mirrors petal_bud_cup -- see its docstring.
    sepal_bud_cup: float = 0.0
    sepal_resolution: int = 6
    sepal_width_resolution: int = _RIBBON_PROFILE_RESOLUTION   # see petal_width_resolution
    sepal_sink: float = 0.0
    sepal_sink_variance: float = 0.0
    # Mirrors petal_vertical_offset -- see its docstring.
    sepal_vertical_offset: float = 0.0
    sepal_vertical_offset_variance: float = 0.0
    # One-time rotation of the WHOLE sepal ring relative to the petal ring --
    # defaults to alternating with a 5-petal default (half the petal spacing),
    # the classic real-flower look where sepals show between petals rather than
    # directly behind them. Distinct from `sepal_spread_horizontal` (per-sepal
    # random scatter) below.
    sepal_azimuth_offset: float = math.pi / 5.0
    # Signed vertical offset seating the sepal ring below (-) or above (+) the
    # petal attachment plane (Z=0) -- petal layering then stacks upward from
    # THIS base, same as petals stack upward from Z=0.
    sepal_z_offset: float = -0.002
    sepal_spread_horizontal: float = 0.0
    sepal_spread_horizontal_variance: float = 0.0
    # Mirrors petal_tube_enabled/_length/_radius/_resolution/_profile -- see
    # those docstrings. Disabled by default = free/Polysepalous (today's only
    # behavior); enabled = a fused Gamosepalous calyx.
    sepal_tube_enabled: bool = False
    sepal_tube_length: float = 0.006
    sepal_tube_length_variance: float = 0.0
    sepal_tube_match_disc: bool = True       # see petal_tube_match_disc
    sepal_tube_radius: float = 0.006
    sepal_tube_radius_variance: float = 0.0
    sepal_tube_resolution: int = 6
    sepal_tube_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_tube_sides: int = 16
    sepal_bud_tube_length: float = 1.0       # see petal_bud_tube_length
    sepal_bud_tube_flare: float = 1.0
    # Mirror petal_tube_noise/_match_angle/_fuse and petal_clamp_overlap --
    # see those docstrings.
    sepal_tube_noise: float = 0.0
    sepal_tube_match_angle: bool = False
    sepal_tube_fuse: bool = False
    sepal_clamp_overlap: float = 0.0
    sepal_scale: float = 1.0
    sepal_scale_variance: float = 0.0
    sepal_scale_age_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.8), (1.0, 1.0)]))
    # Mirrors petal_bud_rotation_pitch -- see its docstring. Slightly more
    # vertical than petals' default (the calyx wraps tighter, as the outer
    # protective layer).
    sepal_bud_rotation_pitch: float = 1.4
    # Mirrors petal_bud_curl/petal_bud_curl_profile -- see those docstrings.
    sepal_bud_curl: float = 0.15
    sepal_bud_curl_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 1.0), (0.5, 1.1), (1.0, 1.5)]))
    # Mirrors petal_bud_sink -- see its docstring.
    sepal_bud_sink: float = 0.0045
    # Mirrors petal_bud_clamp_overlap -- see its docstring.
    sepal_bud_clamp_overlap: float = 0.0
    # Mirrors petal_bud_tip_start -- see its docstring.
    sepal_bud_tip_start: float = 0.5
    # Mirrors petal_bud_tip_lift -- see its docstring. Nonzero by default (unlike
    # the petal side): the calyx is the outermost/visible layer of a closed bud,
    # so this is where the pointed apex actually needs to read.
    sepal_bud_tip_lift: float = 0.004
    # Mirrors petal_bud_tip_height_profile -- see its docstring.
    sepal_bud_tip_height_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 0.0), (0.3, 0.06), (0.6, 0.3), (1.0, 1.0)]))
    # Mirrors petal_bud_tip_radius_profile -- see its docstring.
    sepal_bud_tip_radius_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # Mirrors petal_bud_tip_falloff -- see its docstring.
    sepal_bud_tip_falloff: float = 3.0
    # Sepals default to finishing their uncurl EARLIER than petals (0.55, not
    # 1.0) -- "sepals first, then petals follow" -- see `petal_bloom_start`.
    sepal_bloom_start: float = 0.0
    sepal_bloom_end: float = 0.55
    sepal_layers: int = 1
    sepal_layer_spacing: float = 0.0
    sepal_layer_spacing_variance: float = 0.0
    sepal_layer_divergence: float = _GOLDEN_ANGLE
    sepal_curl_updown: float = 0.0
    sepal_curl_updown_variance: float = 0.0
    sepal_curl_updown_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_curl_updown_age_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_curl_leftright: float = 0.0
    sepal_curl_leftright_variance: float = 0.0
    sepal_curl_leftright_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_curl_outin: float = 0.0
    sepal_curl_outin_variance: float = 0.0
    sepal_curl_outin_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_curl_roll: float = 0.0
    sepal_curl_roll_variance: float = 0.0
    sepal_gravity: float = 0.0
    sepal_gravity_variance: float = 0.0
    sepal_gravity_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_twist: float = 0.0
    sepal_twist_variance: float = 0.0
    sepal_twist_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    # Mirrors petal_top_shape/petal_bottom_shape and their profiles -- see
    # those docstrings.
    sepal_top_shape: float = 0.0
    sepal_top_shape_variance: float = 0.0
    sepal_top_shape_profile: Ramp = field(default_factory=lambda: Ramp(DOME_CAP))
    sepal_bottom_shape: float = 0.0
    sepal_bottom_shape_variance: float = 0.0
    sepal_bottom_shape_profile: Ramp = field(default_factory=lambda: Ramp(DOME_CAP))
    sepal_rotation_yaw: float = 0.0
    sepal_rotation_yaw_variance: float = 0.0
    sepal_rotation_pitch: float = 0.0
    sepal_rotation_pitch_variance: float = 0.0
    # Same 0..1-edit / -1..1-eval remap as petal_rotation_pitch_age_profile (see
    # its docstring above) -- this field also always holds already-remapped
    # -1..1 values.
    sepal_rotation_pitch_age_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    sepal_rotation_roll: float = 0.0
    sepal_rotation_roll_variance: float = 0.0

    # --- disc: a body of revolution, radius-vs-height profile (like the stem's
    # own taper, just revolved instead of extruded along a spine) --------------
    # Unlike sepal_enabled/receptacle_enabled below, the Disc defaults ON, same
    # reasoning as petal_enabled above.
    disc_enabled: bool = True
    disc_radius: float = 0.006
    disc_radius_variance: float = 0.0        # global reroll
    disc_height: float = 0.003               # may be negative -- a downward-
    disc_height_variance: float = 0.0        # pointing (concave/sunken) dome; see
    disc_profile: Ramp = field(default_factory=lambda: Ramp(                # blender/flower.py's
        [(0.0, 1.0), (0.3, 0.97), (0.6, 0.85), (0.85, 0.55), (1.0, 0.0)]))  # module docstring
    disc_resolution: int = 8                 # samples along the disc's profile (>= 2)
    disc_sides: int = 16                     # around the disc

    # --- receptacle: a SECOND revolved body (same `_build_disc`/`DiscSpine` as
    # the Disc above, reused directly -- see module docstring), always swept
    # DOWNWARD from the shared Z=0 rim to meet the stem/pedicel below. `height`
    # is a plain positive magnitude (unlike `disc_height`, it has no "negative =
    # concave" toggle -- it always sweeps down by construction, so
    # `blender/flower.py` always passes `-abs(receptacle_height)` into
    # `_build_disc`). Disabled by default, same reasoning as `sepal_enabled`.
    receptacle_enabled: bool = False
    receptacle_match_disc: bool = True       # see petal_tube_match_disc
    # Matches `disc_radius`'s own default, so unticking Match Disc reveals the
    # value it was already showing rather than snapping the receptacle narrower.
    receptacle_radius: float = 0.006
    receptacle_radius_variance: float = 0.0
    receptacle_height: float = 0.003
    receptacle_height_variance: float = 0.0
    receptacle_profile: Ramp = field(default_factory=lambda: Ramp(
        [(0.0, 1.0), (1.0, 0.0)]))
    receptacle_resolution: int = 6
    receptacle_sides: int = 16               # around (welded to the tube: the tube's)

    # --- Flip Normals, per component (GN only: Flip Faces after the welds).
    # A part welded into another follows that one's flip (a fused tube the
    # petals' / sepals', a welded receptacle the corolla tube's), so the
    # surface keeps one winding.
    petal_flip_normals: bool = False
    sepal_flip_normals: bool = False
    disc_flip_normals: bool = False
    receptacle_flip_normals: bool = False
    petal_tube_flip_normals: bool = False
    sepal_tube_flip_normals: bool = False

    seed: int = 0

    # (Ramp field name) for every curve-enabled parameter — the single source of
    # truth `blender/flower.py`'s curve registry builds from, mirroring
    # `StemParams.RAMP_FIELDS`.
    RAMP_FIELDS = (
        "maturity_scale_profile",
        "petal_width_profile", "petal_tube_profile", "petal_scale_age_profile",
        "petal_curl_updown_profile", "petal_curl_updown_age_profile",
        "petal_curl_leftright_profile", "petal_curl_outin_profile",
        "petal_gravity_profile", "petal_twist_profile", "petal_rotation_pitch_age_profile",
        "petal_bud_curl_profile",
        "petal_bud_tip_height_profile", "petal_bud_tip_radius_profile",
        "petal_top_shape_profile", "petal_bottom_shape_profile",
        "petal_whorl_length_profile", "petal_whorl_width_profile",
        "petal_whorl_pitch_profile",
        "sepal_width_profile", "sepal_tube_profile", "sepal_scale_age_profile",
        "sepal_curl_updown_profile", "sepal_curl_updown_age_profile",
        "sepal_curl_leftright_profile", "sepal_curl_outin_profile",
        "sepal_gravity_profile", "sepal_twist_profile", "sepal_rotation_pitch_age_profile",
        "sepal_bud_curl_profile",
        "sepal_bud_tip_height_profile", "sepal_bud_tip_radius_profile",
        "sepal_top_shape_profile", "sepal_bottom_shape_profile",
        "disc_profile", "receptacle_profile",
    )

    # --- serialization ---
    # Generic over dataclasses.fields(), same pattern as StemParams/PhyllotaxisParams.
    def to_dict(self) -> dict:
        d = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            d[f.name] = value.to_dict() if f.type == "Ramp" else value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FlowerMeshParams:
        kwargs = dict(d)
        for f in dataclasses.fields(cls):
            if f.type == "Ramp" and f.name in kwargs:
                kwargs[f.name] = Ramp.from_dict(kwargs[f.name])
        return cls(**kwargs)


@dataclass
class PetalSpine:
    points: list[Vec]      # base (disc edge) -> tip
    widths: list[float]    # full width at each point (not half-width, not radius)
    tilts: list[float]     # cumulative twist angle (radians) at each point


@dataclass
class DiscSpine:
    points: list[Vec]      # straight vertical spine, base (Z=0) -> top (Z=disc_height)
    radii: list[float]     # disc_radius * disc_profile.evaluate(t) at each point


@dataclass
class FlowerSkeleton:
    petals: list[PetalSpine]
    sepals: list[PetalSpine]  # the calyx whorl -- see `_build_petal_group`; empty
                               # list when `sepal_enabled` is False
    disc: DiscSpine
    receptacle: DiscSpine      # empty (points=[], radii=[]) when
                               # `receptacle_enabled` is False
    petal_cup: float  # see FlowerMeshParams.petal_cup -- one shared whole-
                       # ribbon dome/bowl push, already Maturity-scaled
    sepal_cup: float   # same, for the sepal whorl
    petal_fusion: DiscSpine  # the petal whorl's shared corolla-tube body -- see
                              # `petal_fusion`'s docstring; empty (points=[],
                              # radii=[]) when that whorl's fusion is 0
    sepal_fusion: DiscSpine  # same, for the sepal (calyx) whorl
    # One DIMENSIONLESS push value (a multiple of the boundary ring's own live
    # width) per ribbon cross-section profile point, whorl-wide, not per-petal
    # -- see `_build_shape_carrier`. The two `bottom` ones additionally carry
    # the automatic base fit (`_base_wrap_carrier`) summed underneath.
    petal_top_shape: list[float]
    petal_bottom_shape: list[float]
    sepal_top_shape: list[float]
    sepal_bottom_shape: list[float]


def _build_petal_spine(
    direction: Vec, curl_updown_axis: Vec, curl_leftright_axis: Vec,
    base: Vec, n: int, dl: float, turn_scale: float, roll: float,
    curl_updown: float, curl_updown_profile: Ramp,
    curl_leftright: float, curl_leftright_profile: Ramp,
    curl_outin: float, curl_outin_profile: Ramp,
    gravity: float, gravity_profile: Ramp,
    twist: float, twist_profile: Ramp,
) -> tuple[list[Vec], list[float]]:
    """Walk `direction` forward `n-1` steps of length `dl`, bending it via three
    FIXED curl axes (Up/Down, Left/Right, Out/In -- see FlowerMeshParams' shape
    fields docstring) plus Gravity (bends toward world down/up, clamped) -- the
    same multi-axis split `stem.py`'s `build_stem` uses for Parent Curl + Gravity,
    simplified (no Straightness/Noise/Zigzag; petals are short enough not to need
    them yet). Twist doesn't bend the centerline -- it only accumulates into
    per-point `tilt`, which `Curve to Mesh` reads directly off the authored curve;
    `roll` seeds that same accumulator with a constant one-time offset (see
    FlowerMeshParams' rotation fields docstring), so it's visible along the
    WHOLE petal, including the base, even with zero Twist/Curl.

    `curl_updown_axis`/`curl_leftright_axis` are computed by the caller (not
    derived here) because Petal Rotation's Yaw/Pitch (see FlowerMeshParams'
    rotation fields docstring) already establish `direction` and this frame
    once, statically, before any per-step bending starts."""
    curl_outin_axis: Vec = (0.0, 0.0, 1.0)

    d = direction
    pos = base
    points: list[Vec] = [pos]
    tilts: list[float] = [roll]
    tilt_accum = roll

    for i in range(1, n):
        f = i / (n - 1)
        if gravity != 0.0:
            # Inverted from the stem's own Gravity (which is + = down): negative
            # bends toward world-down here, positive toward world-up.
            target = (0.0, 0.0, -1.0) if gravity < 0.0 else (0.0, 0.0, 1.0)
            g = abs(gravity) * turn_scale * gravity_profile.evaluate(f) * dl
            if g > 1e-9:
                d = rotate_toward(d, target, g, curl_updown_axis)

        c_ud = curl_updown * curl_updown_profile.evaluate(f) * turn_scale * dl
        if abs(c_ud) > 1e-9:
            d = rotate_axis(d, curl_updown_axis, c_ud)
        c_lr = curl_leftright * curl_leftright_profile.evaluate(f) * turn_scale * dl
        if abs(c_lr) > 1e-9:
            d = rotate_axis(d, curl_leftright_axis, c_lr)
        c_oi = curl_outin * curl_outin_profile.evaluate(f) * turn_scale * dl
        if abs(c_oi) > 1e-9:
            d = rotate_axis(d, curl_outin_axis, c_oi)
        d = normalize(d)

        pos = add(pos, scale(d, dl))
        points.append(pos)
        tilt_accum += twist * twist_profile.evaluate(f) * turn_scale * dl
        tilts.append(tilt_accum)

    return points, tilts


_NOISE_CONTROL_POINTS = 5
# How far up the lobe Fuse's mouth-filling width fades back into the petal's
# own authored outline (see `_build_petal_group`). Short on purpose: Fuse is
# meant to close the seam at the mouth, not to override the shape the Petal
# Shape preset just set.
_FUSE_SPAN = 0.22


def _steady_frame(points: list[Vec], tilts: list[float], yaw_dir: Vec) -> list[float]:
    """`tilts` with a half turn added when Blender would hand this lobe its
    cross-section upside down.

    A lobe is swept along a curve whose normals are Blender's Minimum Twist,
    which starts the first point's normal at `cross(first tangent, Z)`. That
    turns over by 180 degrees the moment the first tangent passes straight up
    -- and Bloom carries every lobe from its Bud Pitch to its open Pitch, so a
    bud leaning in past vertical opens through exactly that: the lobe's
    points came out numbered the other way across it (and Cup bowed to the
    other side), so stages baked either side of it could not be blended.

    `yaw_dir` is the lobe's own outward heading before Pitch -- horizontal, so
    it never turns over. A lobe whose first tangent leans along it (every
    outward-leaning lobe, so every open pose) is left exactly as it was; one
    leaning back past vertical gets the half turn, which puts its normal where
    the outward lean would: one frame through the whole opening."""
    if len(points) < 2:
        return tilts
    tx = points[1][0] - points[0][0]
    ty = points[1][1] - points[0][1]
    if abs(tx) + abs(ty) < 1e-9:
        return tilts                    # straight up: Blender's own fallback frame
    if tx * yaw_dir[0] + ty * yaw_dir[1] >= 0.0:
        return tilts
    return [t + math.pi for t in tilts]


def _smooth_noise(t: float, controls: list[float]) -> float:
    """Value noise: linearly interpolate between `controls` across t in 0..1.

    Deliberately not per-sample independent randomness -- that reads as a
    spiky, digital jitter rather than the gentle irregularity a real stem or
    corolla tube has. A handful of control values stretched across the whole
    length gives a slow wander instead, and stays identical regardless of how
    many samples the profile happens to be evaluated at (so raising Tube
    Resolution refines the same shape rather than rerolling it)."""
    if len(controls) < 2:
        return controls[0] if controls else 0.0
    x = min(max(t, 0.0), 1.0) * (len(controls) - 1)
    i = min(int(x), len(controls) - 2)
    return controls[i] + (controls[i + 1] - controls[i]) * (x - i)


def _build_disc(
    disc_radius: float, disc_height: float, profile: Ramp, n: int,
    noise: float = 0.0, noise_key: str = "", seed: int = 0,
) -> DiscSpine:
    """A revolved body's radius-vs-height spine. `noise` (0 = off, the default
    for Disc/Receptacle) wobbles the radius along the length by up to that
    fraction, so a corolla tube doesn't read as a machined extrusion -- see
    `_smooth_noise` for why it's slow value noise rather than per-sample
    jitter. The rim end (t=0) is deliberately left UNTOUCHED: that's where the
    body has to meet the disc or the stem, and wobbling it would reopen the
    seam every other part of this module works to close."""
    dl = disc_height / (n - 1) if n > 1 else 0.0
    controls: list[float] = []
    if noise > 0.0:
        rng = random.Random(f"{seed}:{noise_key}")
        controls = [rng.uniform(-1.0, 1.0) for _ in range(_NOISE_CONTROL_POINTS)]
        controls[0] = 0.0
    points: list[Vec] = []
    radii: list[float] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        points.append((0.0, 0.0, dl * i))
        r = disc_radius * profile.evaluate(t)
        if controls:
            r *= 1.0 + noise * _smooth_noise(t, controls)
        radii.append(max(0.0, r))
    return DiscSpine(points=points, radii=radii)


def _build_shape_carrier(
    profile: Ramp, amount: float, resolution: int = _RIBBON_PROFILE_RESOLUTION,
) -> list[float]:
    """Sample `profile` at `resolution` evenly-spaced points across 0..1 (one
    value per ribbon cross-section profile point, see
    `_RIBBON_PROFILE_RESOLUTION`'s docstring), scaled by `amount` -- the
    whorl-wide "Top"/"Bottom Shape" push per width-position, for
    `blender/flower.py`'s GN-side boundary-ring push (see
    `FlowerMeshParams.petal_top_shape_profile`'s docstring for why this can't
    be evaluated inside Geometry Nodes itself). Whorl-wide, not per-petal --
    called once per whorl in `build_flower_mesh`, unlike per-petal skeleton
    data.

    Values are DIMENSIONLESS (multiples of the boundary ring's own width), not
    meters -- the GN side multiplies by the ring's live measured width. See the
    `petal_top_shape` docstring for why that indirection is the whole point."""
    return [amount * profile.evaluate(i / (resolution - 1)) for i in range(resolution)]


def _touching_width(radius: float, count: int) -> float:
    """The ribbon width at which `count` evenly-spaced petals' EDGES land
    exactly on each other's, at distance `radius` from the flower's axis.

    `2 * radius * tan(pi / count)`, and the tangent matters: the obvious
    candidate is the CHORD between adjacent centrelines, `2 * r * sin(pi/n)`,
    but that's the distance between the two centre POINTS, not the width at
    which their edges meet. A petal is a flat ribbon spanning +/- w/2
    perpendicular to its own radial direction, so its corner sits off the
    centreline tangentially while the neighbour's corner is rotated away by
    2*pi/n; solving for the two corners coinciding gives w/2 = r*tan(pi/n) on
    both axes. Since tan > sin always, the chord version stops the petals
    slightly SHORT of touching and leaves a permanent hairline gap -- which is
    exactly what it did before this was worked out.

    Checkable by hand at n=4, r=1: tan(45 deg) = 1, so w = 2. The petal along
    +X then spans Y from -1 to +1 and has a corner at (1, 1); the petal along
    +Y spans X from -1 to +1 and has its corner at (1, 1) too. They meet."""
    if count < 3:
        return float("inf")
    return 2.0 * radius * math.tan(math.pi / count)


def _add_carriers(a: list[float], b: list[float]) -> list[float]:
    """Sum two shape carriers position-by-position. Both are the same
    dimensionless per-cross-section-point channel (multiples of the ring's own
    width), so a plain add composes them -- which is what makes Bottom Shape an
    additive override on the automatic base fit rather than a replacement."""
    return [x + y for x, y in zip(a, b, strict=True)]


def _base_wrap_carrier(
    attach_radius: float, base_width: float, resolution: int = _RIBBON_PROFILE_RESOLUTION,
) -> list[float]:
    """The automatic base fit: how far each cross-section point of a ribbon's
    BASE ring has to be pulled back along its own growth direction for the
    whole base edge to land flush on the circle it attaches to.

    A ribbon's base ring is a straight CHORD of length `base_width`, centered
    at radius `attach_radius` on that circle and lying tangent to it -- so the
    point sitting `x` to the side of center is actually `sqrt(R^2 + x^2)` from
    the flower's axis, not `R`. Pulling it back by the difference puts it
    exactly on the circle. That difference is zero at the center and grows
    toward both edges, i.e. the flat guillotine cut becomes a shallow arc
    hugging the disc rim (or a corolla/calyx tube's top rim -- `attach_radius`
    is whichever one this whorl actually attaches to). Without it the base
    corners always float a little proud of a round attachment surface, which
    is what made Radial Offset feel mandatory on every new flower.

    Returned DIMENSIONLESS, in multiples of `base_width`, to ride the same
    carrier channel as `_build_shape_carrier` above -- dividing through by w
    leaves `sqrt(k^2 + t^2) - k` with `k = R/w` and `t` in -0.5..0.5, which
    needs only the RATIO of radius to width. Two known approximations, neither
    worth extra machinery: `k` uses the whorl's nominal width, so per-petal
    Width/Scale variance and per-layer Age scaling leave a second-order depth
    error; and with nonzero Pitch the growth direction tilts out of the radial
    plane, so this approximates the true cylinder intersection rather than
    matching it exactly."""
    if base_width <= 1e-9 or attach_radius <= 1e-9:
        return [0.0] * resolution
    k = attach_radius / base_width
    return [
        math.sqrt(k * k + t * t) - k
        for t in (i / (resolution - 1) - 0.5 for i in range(resolution))
    ]


@dataclass
class _PetalGroupSpec:
    """Internal, non-serialized bundle of one whorl-of-ribbon-appendages' worth
    of parameters -- built fresh from either the `petal_*` or `sepal_*` fields
    inside `build_flower_mesh`, so `_build_petal_group`'s whorl-building loop
    (shared by Petals and Sepals) only has to be written once. Not a public type,
    not JSON-serialized -- `FlowerMeshParams`'s flat `petal_*`/`sepal_*` fields
    remain the actual serialized/UI-facing shape."""
    name_prefix: str   # "petal"/"sepal" -- Parameter RNG naming, keeps the two
                        # whorls' random streams independent even under the same seed
    base_radius: float  # shared attachment-ring radius (the varied disc_radius)
    seed: int
    count: float
    count_variance: float
    length: float
    length_variance: float
    width: float
    width_variance: float
    width_profile: Ramp
    scale: float
    scale_variance: float
    scale_age_profile: Ramp
    resolution: int
    width_resolution: int  # points across a ribbon (its profile's resolution)
    sink: float
    sink_variance: float
    vertical_offset: float
    vertical_offset_variance: float
    spread_horizontal: float
    spread_horizontal_variance: float
    tube_enabled: bool
    tube_length: float    # already resolved/varied by the caller (same tier as cup)
    tube_radius: float    # already resolved/varied by the caller
    tube_resolution: int
    tube_profile: Ramp
    tube_noise: float        # radius wobble along the tube -- see `_build_disc`
    tube_match_angle: bool   # derive the lobes' base pitch from the tube's mouth
    tube_fuse: bool          # widen lobe bases to tile the tube's mouth
    bud_tube_length: float   # the tube's bud length / flare -- see petal_bud_tube_length
    bud_tube_flare: float
    clamp_overlap: float     # OPEN-pose value: 0 = neighbours may overlap freely, 1 = never
    azimuth_offset: float  # one-time rotation of the whole whorl (0 for petals)
    z_offset: float        # base Z of layer 0 (0 for petals, below rim for sepals)
    layers: int
    layer_spacing: float
    layer_spacing_variance: float
    layer_divergence: float
    curl_updown: float
    curl_updown_variance: float
    curl_updown_profile: Ramp
    curl_updown_age_profile: Ramp
    curl_leftright: float
    curl_leftright_variance: float
    curl_leftright_profile: Ramp
    curl_outin: float
    curl_outin_variance: float
    curl_outin_profile: Ramp
    curl_roll: float
    curl_roll_variance: float
    gravity: float
    gravity_variance: float
    gravity_profile: Ramp
    twist: float
    twist_variance: float
    twist_profile: Ramp
    rotation_yaw: float
    rotation_yaw_variance: float
    rotation_pitch: float
    rotation_pitch_variance: float
    rotation_pitch_age_profile: Ramp
    rotation_roll: float
    rotation_roll_variance: float
    # bud/bloom (see FlowerMeshParams' `bloom` docstring): `bloom` is the shared
    # global master; `bloom_start`/`bloom_end` are THIS whorl's own window over
    # it; `bud_curl`/`bud_curl_profile` are THIS whorl's own bud-closed bend.
    bloom: float
    bloom_start: float
    bloom_end: float
    bud_rotation_pitch: float
    bud_curl: float
    bud_curl_profile: Ramp
    bud_sink: float
    bud_clamp_overlap: float  # this whorl's own bud-pose Clamp Overlap, see FlowerMeshParams
    bud_tip_start: float
    bud_tip_lift: float
    bud_tip_height_profile: Ramp
    bud_tip_radius_profile: Ramp
    bud_tip_falloff: float
    # Around the Whorl (petals only; see FlowerMeshParams.petal_whorl_*).
    whorl_length_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    whorl_width_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))
    whorl_pitch_profile: Ramp = field(default_factory=lambda: Ramp.constant(1.0))


_BEND_BLEND_SAMPLES = 9


def _local_bloom(bloom: float, bloom_start: float, bloom_end: float) -> float:
    """A whorl's own window over the shared `bloom` master (see FlowerMeshParams'
    `bloom` docstring): 0 below `bloom_start`, 1 above `bloom_end`, linearly in
    between. Shared by `_build_petal_group` (per-petal bend/sink/tip-lift blend)
    and `build_flower_mesh` (the cross-section Curvature blend) so the two never
    drift out of sync. Guards `bloom_end == bloom_start` as a hard step rather
    than dividing by zero."""
    span = bloom_end - bloom_start
    if abs(span) < 1e-9:
        return 1.0 if bloom >= bloom_end else 0.0
    return min(1.0, max(0.0, (bloom - bloom_start) / span))


def _blend_curl_profile(
    local_bloom: float,
    open_amount: float, open_profile: Ramp,
    bud_amount: float, bud_profile: Ramp,
) -> tuple[float, Ramp]:
    """Blend two (amount, along-length-profile) bend specs into one, by sampling
    `bend(f) = lerp(bud_amount*bud_profile(f), open_amount*open_profile(f),
    local_bloom)` at a handful of points and packing the result as a plain Ramp
    with a unit scalar -- so `_build_petal_spine`'s (float, Ramp) signature needs
    no changes at all. Internal, non-serialized (like `_PetalGroupSpec` itself);
    called once per whorl, not per petal, since neither input depends on
    per-petal Age/idx. At local_bloom == 1 this reproduces `open_profile` scaled
    by `open_amount` exactly (the backward-compatible bloom==1 case)."""
    pts = []
    for i in range(_BEND_BLEND_SAMPLES):
        f = i / (_BEND_BLEND_SAMPLES - 1)
        bud_bend = bud_amount * bud_profile.evaluate(f)
        open_bend = open_amount * open_profile.evaluate(f)
        pts.append((f, bud_bend + (open_bend - bud_bend) * local_bloom))
    return 1.0, Ramp(pts)


def _build_petal_group(
    spec: _PetalGroupSpec,
) -> tuple[list[PetalSpine], DiscSpine, list[float]]:
    """Build one whole whorl (all layers) of ribbon appendages -- the actual
    per-petal math (placement, Curl/Gravity/Twist/Rotation, Age-driven Scale/
    Curl/Pitch) shared by both Petals and Sepals; see `_PetalGroupSpec`'s
    docstring and `FlowerMeshParams`' petal/sepal field docstrings for the full
    reasoning behind each step.

    Also builds this whorl's shared corolla/calyx FUSION tube (see
    `FlowerMeshParams.petal_fusion`'s docstring) -- an empty `DiscSpine` when
    `spec.fusion` is 0, so callers don't need a separate enabled check. Every
    lobe's base is raised to the tube's own top rim (`fusion_height` up, at
    `tube_top_radius`) and shortened by the same fraction, rather than starting
    at the full disc rim -- the tube and the lobes together always meet exactly
    where the tube ends, with no separate seam-closing step needed.

    Third return is this whorl's automatic base-fit carrier (see
    `_base_wrap_carrier`). Computed HERE rather than at the call site precisely
    because the radius the base has to fit is `tube_top_radius`, whose
    disc-rim-vs-tube-rim branch lives in this function -- deriving it again
    outside would mean keeping a second copy of that branch in sync."""
    n = max(2, spec.resolution)
    count = max(3, round(_varied(
        f"{spec.name_prefix}_count", float(spec.count), spec.count_variance, spec.seed,
        floor=_MIN_PETAL_COUNT)))
    layers = max(1, spec.layers)
    # This whorl's own window over the shared `bloom` master -- computed once,
    # not per petal, since it doesn't depend on Age/idx. See `_local_bloom`.
    local_bloom = _local_bloom(spec.bloom, spec.bloom_start, spec.bloom_end)
    # Clamp Overlap's own bud-pose pair -- same lerp as Sink/Curl/Pitch above,
    # computed once here (whorl-wide, not per-petal) and used in place of
    # `spec.clamp_overlap` in the width-clamp block below.
    effective_clamp_overlap = (
        spec.bud_clamp_overlap + (spec.clamp_overlap - spec.bud_clamp_overlap) * local_bloom)

    # Corolla/calyx tube: see this function's own docstring above and
    # `FlowerMeshParams.petal_tube_enabled`'s docstring for the full
    # reasoning. Its own independent Length/Radius/Profile, NOT derived from
    # this whorl's Length or the shared disc radius. `tube_top_radius`
    # BYPASSES `tube_profile` entirely (rather than evaluating it at 1.0)
    # whenever the tube is disabled, guaranteeing `tube_enabled == False`
    # reproduces today's free-petal geometry byte-for-byte regardless of what
    # an unrelated tube_profile happens to be authored as.
    # Through Bloom: the bud's length and flare blend to the open tube's
    # (see `petal_bud_tube_length`); 1.0 / 1.0 leaves the tube exactly as is.
    grown = min(1.0, max(0.0, spec.bloom))
    length_k = spec.bud_tube_length + (1.0 - spec.bud_tube_length) * grown
    flare_k = spec.bud_tube_flare + (1.0 - spec.bud_tube_flare) * grown
    tube_profile = spec.tube_profile
    if flare_k != 1.0:
        tube_profile = Ramp([(x, 1.0 + (y - 1.0) * flare_k) for x, y in tube_profile.points])
    fusion_height = spec.tube_length * length_k if spec.tube_enabled else 0.0
    if fusion_height > 1e-9:
        tube = _build_disc(
            spec.tube_radius, fusion_height, tube_profile, max(2, spec.tube_resolution),
            noise=spec.tube_noise, noise_key=f"{spec.name_prefix}_tube_noise", seed=spec.seed)
        tube.points = [add(pt, (0.0, 0.0, spec.z_offset)) for pt in tube.points]
        # The lobes attach to the tube's ACTUAL top rim, read off the built
        # spine rather than recomputed from the profile -- otherwise Tube Noise
        # would move the rim and leave the lobes attached to where it used to be.
        tube_top_radius = tube.radii[-1]
    else:
        tube = DiscSpine(points=[], radii=[])
        tube_top_radius = spec.base_radius

    # Match Tube Angle: the pitch that makes a lobe continue the tube's wall
    # instead of starting from horizontal. The wall's direction at the mouth is
    # (dr, dz) in the radius-vs-height plane, so the angle above horizontal is
    # atan2(dz, dr) -- pi/2 for a dead-straight tube (lobes carry straight on
    # up), shallower as the mouth flares, and PAST pi/2 for an urn that pinches
    # back inward (dr < 0), which is exactly the continuation an urn wants.
    # Measured from the built spine for the same reason `tube_top_radius` is:
    # it has to describe the rim that actually exists, noise and all.
    tube_mouth_pitch = 0.0
    if spec.tube_match_angle and fusion_height > 1e-9 and len(tube.radii) >= 2:
        dr = tube.radii[-1] - tube.radii[-2]
        dz = tube.points[-1][2] - tube.points[-2][2]
        if abs(dr) > 1e-12 or abs(dz) > 1e-12:
            tube_mouth_pitch = math.atan2(dz, dr)

    # Fuse: the lobe base width that exactly tiles the tube's mouth -- the
    # chord between adjacent petal centrelines AT the rim, so `count` of them
    # laid side by side close the circle with no gaps. Only meaningful with a
    # tube; a free/Polypetalous whorl has no mouth to fill.
    fuse_base_width = 0.0
    if spec.tube_fuse and fusion_height > 1e-9 and tube_top_radius > 1e-9:
        fuse_base_width = _touching_width(tube_top_radius, count)

    length_param = Parameter(
        name=f"{spec.name_prefix}_length", base=spec.length, variance=spec.length_variance)
    width_param = Parameter(
        name=f"{spec.name_prefix}_width", base=spec.width, variance=spec.width_variance,
        factor_ramp=spec.width_profile)
    scale_param = Parameter(
        name=f"{spec.name_prefix}_scale", base=spec.scale, variance=spec.scale_variance,
        age_ramp=spec.scale_age_profile)
    sink_param = Parameter(
        name=f"{spec.name_prefix}_sink", base=spec.sink, variance=spec.sink_variance)
    vertical_offset_param = Parameter(
        name=f"{spec.name_prefix}_vertical_offset", base=spec.vertical_offset,
        variance=spec.vertical_offset_variance)
    spread_h_param = Parameter(
        name=f"{spec.name_prefix}_spread_horizontal", base=spec.spread_horizontal,
        variance=spec.spread_horizontal_variance)
    layer_jitter_param = Parameter(
        name=f"{spec.name_prefix}_layer_jitter", base=0.0,
        variance=spec.layer_spacing_variance)
    curl_updown_param = Parameter(
        name=f"{spec.name_prefix}_curl_updown", base=spec.curl_updown,
        variance=spec.curl_updown_variance, age_ramp=spec.curl_updown_age_profile)
    curl_leftright_param = Parameter(
        name=f"{spec.name_prefix}_curl_leftright", base=spec.curl_leftright,
        variance=spec.curl_leftright_variance)
    curl_outin_param = Parameter(
        name=f"{spec.name_prefix}_curl_outin", base=spec.curl_outin,
        variance=spec.curl_outin_variance)
    curl_roll_param = Parameter(
        name=f"{spec.name_prefix}_curl_roll", base=spec.curl_roll,
        variance=spec.curl_roll_variance)
    gravity_param = Parameter(
        name=f"{spec.name_prefix}_gravity", base=spec.gravity, variance=spec.gravity_variance)
    twist_param = Parameter(
        name=f"{spec.name_prefix}_twist", base=spec.twist, variance=spec.twist_variance)
    yaw_param = Parameter(
        name=f"{spec.name_prefix}_rotation_yaw", base=spec.rotation_yaw,
        variance=spec.rotation_yaw_variance)
    pitch_param = Parameter(
        name=f"{spec.name_prefix}_rotation_pitch", base=spec.rotation_pitch,
        variance=spec.rotation_pitch_variance, age_ramp=spec.rotation_pitch_age_profile)
    roll_param = Parameter(
        name=f"{spec.name_prefix}_rotation_roll", base=spec.rotation_roll,
        variance=spec.rotation_roll_variance)

    petals: list[PetalSpine] = []
    idx = 0
    for layer in range(layers):
        layer_start_az = spec.azimuth_offset + layer * spec.layer_divergence
        layer_z = spec.z_offset + layer * spec.layer_spacing
        # 1.0 (mature) at the bottom/outer layer, fading to 0.0 (youngest) at the
        # top/inner one; a single layer is always fully mature. See the
        # `petal_layers` field docstring above for the full Age reasoning.
        age_i = 1.0 if layers <= 1 else 1.0 - layer / (layers - 1)
        for k, az in enumerate(ring_azimuths(count, start=layer_start_az)):
            # Around the Whorl: this petal's place on the ring, both ways
            # from the first (0) to straight opposite (1) -- mirror-symmetric.
            whorl_x = 2.0 * min(k, count - k) / count
            whorl_length_k = spec.whorl_length_profile.evaluate(whorl_x)
            whorl_width_k = spec.whorl_width_profile.evaluate(whorl_x)
            whorl_pitch_k = spec.whorl_pitch_profile.evaluate(whorl_x)
            # spread_horizontal scatters the BASE's azimuth on the ring (same
            # mechanism/naming as PhyllotaxisParams.spread_horizontal: a magnitude
            # Parameter + its own deterministic RNG) -- distinct from Yaw below,
            # which only rotates the pointing direction, not the base position.
            spread_h_i = spread_h_param.evaluate(element_id=idx, seed=spec.seed)
            if spread_h_i:
                rng = random.Random(f"{spec.seed}:{idx}:{spec.name_prefix}_spread_horizontal")
                az += rng.uniform(-spread_h_i, spread_h_i)
            radial_dir: Vec = (math.cos(az), math.sin(az), 0.0)

            # Rotation: a ONE-TIME base orientation (Yaw/Pitch/Roll), fully
            # established before any per-step Curl/Gravity bending begins -- see
            # FlowerMeshParams' rotation fields docstring for why this is
            # distinct from Curl. Yaw turns `direction` around the disc normal;
            # Pitch then statically tilts it out of the flat plane around the
            # resulting side axis. Roll doesn't touch `direction`/`side_axis` at
            # all -- it's handed straight to `_build_petal_spine` as the initial
            # `tilt` value instead (see that function's docstring), since it
            # spins the cross-section, not the centerline.
            yaw_i = yaw_param.evaluate(element_id=idx, seed=spec.seed)
            direction = (
                normalize(rotate_axis(radial_dir, (0.0, 0.0, 1.0), yaw_i))
                if yaw_i else radial_dir
            )
            yaw_dir = direction                 # horizontal: never flips (see below)
            side_axis = cross(direction, (0.0, 0.0, 1.0))
            side_axis = normalize(side_axis) if length(side_axis) > 1e-9 else (1.0, 0.0, 0.0)
            # Bloom: blend this whorl's Bud Pitch (near-vertical, closed) UNDER
            # the authored open-pose Pitch -- the SAME lerp-by-local_bloom idea
            # as Curl's own bud blend (`_blend_curl_profile`), just for a
            # one-time rigid rotation instead of a per-step integrated bend, so
            # no profile sampling is needed. At local_bloom == 1 this reproduces
            # `pitch_i` exactly (the pre-Bud-Pitch, backward-compatible case).
            # See `petal_bud_rotation_pitch`'s docstring for why this -- not
            # Curl -- is meant to carry most of the visual opening motion.
            # With Match Tube Angle on, the authored Pitch is an OFFSET from
            # the tube's own mouth angle rather than an absolute tilt, so a
            # lobe at Pitch 0 continues the wall exactly and reshaping the top
            # edge carries the lobes with it. Added BEFORE the bud blend, not
            # after: Bud Pitch is a genuine absolute (near-vertical, closed)
            # pose, and the whole point of the blend is to travel from that to
            # whatever the open pose turns out to be.
            pitch_i = (
                tube_mouth_pitch
                + pitch_param.evaluate(element_id=idx, age=age_i, seed=spec.seed)
                * whorl_pitch_k)
            effective_pitch_i = (spec.bud_rotation_pitch
                                 + (pitch_i - spec.bud_rotation_pitch) * local_bloom)
            if effective_pitch_i:
                direction = normalize(rotate_axis(direction, side_axis, effective_pitch_i))
                side_axis = cross(direction, (0.0, 0.0, 1.0))
                side_axis = (
                    normalize(side_axis) if length(side_axis) > 1e-9 else (1.0, 0.0, 0.0)
                )
            # Curl Roll: redirects the bend-axis frame itself around `direction`
            # (see petal_curl_roll's docstring) -- independent of, and applied
            # after, Rotation's own Roll-equivalent handling above (Yaw/Pitch).
            curl_roll_i = curl_roll_param.evaluate(element_id=idx, seed=spec.seed)
            if curl_roll_i:
                side_axis = normalize(rotate_axis(side_axis, direction, curl_roll_i))
            roll_i = roll_param.evaluate(element_id=idx, seed=spec.seed)

            scale_i = scale_param.evaluate(element_id=idx, age=age_i, seed=spec.seed)
            # Length is the free lobe's OWN length -- independent of the tube
            # (see FlowerMeshParams.petal_tube_enabled's docstring), unlike
            # the earlier design where enabling Fusion silently shortened it.
            length_i = max(
                _MIN_LENGTH,
                length_param.evaluate(element_id=idx, seed=spec.seed) * scale_i * whorl_length_k)
            sink_i = sink_param.evaluate(element_id=idx, seed=spec.seed)
            z_i = (
                layer_z + fusion_height
                + vertical_offset_param.evaluate(element_id=idx, seed=spec.seed)
                + layer_jitter_param.evaluate(element_id=idx, seed=spec.seed))
            cu_i = curl_updown_param.evaluate(element_id=idx, age=age_i, seed=spec.seed)
            cl_i = curl_leftright_param.evaluate(element_id=idx, seed=spec.seed)
            co_i = curl_outin_param.evaluate(element_id=idx, seed=spec.seed)
            gravity_i = gravity_param.evaluate(element_id=idx, seed=spec.seed)
            twist_i = twist_param.evaluate(element_id=idx, seed=spec.seed)

            # the base attachment stays at the (spread_horizontal-jittered) rim
            # azimuth -- only the petal's own pointing direction rotates further
            # (Yaw), so petals still leave the disc rim cleanly spaced even while
            # fanning unevenly outward. Sink moves it radially in/out; layering
            # moves it up/down. Bloom additionally pulls the base toward
            # `bud_sink` as local_bloom falls toward 0 (see `petal_bud_sink`
            # docstring) -- a plain lerp, since sink is a scalar, not a profile.
            effective_sink_i = spec.bud_sink + (sink_i - spec.bud_sink) * local_bloom
            base_radius_i = max(0.0, tube_top_radius - effective_sink_i)
            base = (radial_dir[0] * base_radius_i, radial_dir[1] * base_radius_i, z_i)
            dl = length_i / (n - 1)
            turn_scale = _TURN / length_i
            # Bloom: blend this whorl's bud bend UNDER the authored open-pose
            # Curl Up/Down (see `_blend_curl_profile` + FlowerMeshParams' `bloom`
            # docstring) before the ONE spine integration below -- at
            # local_bloom == 1 this reproduces (cu_i, spec.curl_updown_profile)
            # exactly (the pre-Bloom, backward-compatible case).
            bend_amount, bend_profile = _blend_curl_profile(
                local_bloom, cu_i, spec.curl_updown_profile,
                spec.bud_curl, spec.bud_curl_profile)
            points, tilts = _build_petal_spine(
                direction, side_axis, direction, base, n, dl, turn_scale, roll_i,
                bend_amount, bend_profile,
                cl_i, spec.curl_leftright_profile,
                co_i, spec.curl_outin_profile,
                gravity_i, spec.gravity_profile,
                twist_i, spec.twist_profile,
            )
            # Bud tip shaping: within the tip zone (spec.bud_tip_start -> 1.0),
            # raise Z (Height profile x Lift magnitude) AND pull the point's
            # radial distance from the central axis in/out (Radius profile) --
            # see FlowerMeshParams' `petal_bud_tip_*` docstrings for the full
            # reasoning. Evaluated on a LOCAL 0..1 parameter rescaled to the tip
            # zone, not the whole spine's own `f`, so the curve widgets always
            # get full resolution over just the tip regardless of where the
            # zone starts. Fades out over the SAME local_bloom 0->1 range Curl
            # itself unwinds over, via a power curve (see `bud_tip_falloff`'s
            # docstring for why this replaced an earlier separate-window
            # approach) -- smooth everywhere, and always exactly 0 at
            # local_bloom == 1, never later.
            tip_fade = (1.0 - local_bloom) ** max(0.1, spec.bud_tip_falloff)
            if tip_fade > 0.0:
                tip_start = min(1.0, max(0.0, spec.bud_tip_start))
                zone_span = 1.0 - tip_start
                for j in range(n):
                    f = j / (n - 1) if n > 1 else 1.0
                    if f < tip_start:
                        continue
                    t = (f - tip_start) / zone_span if zone_span > 1e-9 else 1.0
                    h = spec.bud_tip_height_profile.evaluate(t)
                    r_mult = spec.bud_tip_radius_profile.evaluate(t)
                    px, py, pz = points[j]
                    radial_scale = 1.0 + (r_mult - 1.0) * tip_fade
                    z_add = spec.bud_tip_lift * h * tip_fade
                    points[j] = (px * radial_scale, py * radial_scale, pz + z_add)
            widths = [
                max(0.0, width_param.evaluate(factor=j / (n - 1), element_id=idx, seed=spec.seed)
                    * scale_i * whorl_width_k)
                for j in range(n)
            ]
            # Clamp Overlap and Fuse are two halves of the same measurement:
            # `touch_j` is the width at which this point's two EDGES land
            # exactly on its neighbours' edges -- see `_touching_width`. Clamp
            # is a ceiling on that (neighbours stop intersecting); Fuse is a
            # floor near the base (lobe bases widen until they tile the tube's
            # mouth with no gaps, which is what makes a sympetalous corolla
            # read as one piece rather than separate straps on a cylinder).
            #
            # `clamp_overlap` divides that limit rather than lerping toward it,
            # so the control keeps going past 1: 1 = edges exactly meet, below
            # 1 permits proportionally more overlap, above 1 pulls the petals
            # narrower than touching and opens a deliberate gap between them.
            # Uses `effective_clamp_overlap` (this whorl's own Bud Clamp
            # Overlap blended toward the open value by local bloom, computed
            # once above) rather than `spec.clamp_overlap` directly, so the
            # ceiling can tighten alongside Bud Sink as the whorl closes.
            #
            # Approximation worth naming: `touch_j` assumes the ring is evenly
            # spaced at 2*pi/count, so Spread Horizontal's own azimuth jitter
            # can still bring a pair closer than this predicts.
            if effective_clamp_overlap > 1e-6 or fuse_base_width > 0.0:
                for j in range(n):
                    px, py, _pz = points[j]
                    w = widths[j]
                    if effective_clamp_overlap > 1e-6:
                        allowed = (_touching_width(math.hypot(px, py), count)
                                   / effective_clamp_overlap)
                        w = min(w, allowed)
                    if fuse_base_width > 0.0:
                        f = j / (n - 1) if n > 1 else 0.0
                        # Fades out over the first _FUSE_SPAN of the length, so
                        # the mouth is fully covered and the petal's own
                        # authored outline takes back over immediately above it
                        # rather than the whole lobe being forced wide.
                        if f < _FUSE_SPAN:
                            w = max(w, fuse_base_width * (1.0 - f / _FUSE_SPAN))
                    widths[j] = max(0.0, w)
            tilts = _steady_frame(points, tilts, yaw_dir)
            petals.append(PetalSpine(points=points, widths=widths, tilts=tilts))
            idx += 1

    # Automatic base fit, from this whorl's NOMINAL (un-varied) attachment
    # radius and base width -- see `_base_wrap_carrier` for the geometry and
    # for why sampling the nominal values rather than each petal's own is an
    # acceptable second-order error. `spec.scale`'s own Age ramp is skipped for
    # the same reason: the carrier is whorl-wide, so there's one value to pick
    # and the mature/outer layer is the right one to key it to.
    nominal_sink = spec.bud_sink + (spec.sink - spec.bud_sink) * local_bloom
    base_wrap = _base_wrap_carrier(
        max(0.0, tube_top_radius - nominal_sink),
        max(0.0, spec.width * spec.width_profile.evaluate(0.0) * spec.scale),
        resolution=max(2, spec.width_resolution))
    return petals, tube, base_wrap


def build_flower_mesh(p: FlowerMeshParams) -> FlowerSkeleton:
    disc_radius = max(_MIN_RADIUS, _varied(
        "disc_radius", p.disc_radius, p.disc_radius_variance, p.seed))
    # NOT floored at 0 -- a negative height points the disc's revolve DOWNWARD
    # (a sunken/concave bowl instead of a mound); see blender/flower.py.
    disc_height = _varied("disc_height", p.disc_height, p.disc_height_variance, p.seed)

    # Corolla tube: independent Length/Radius, varied ONCE here (same
    # "enabled -> vary" pattern as Receptacle above) -- not per-petal, since
    # it drives one shared tube surface.
    if p.petal_tube_enabled:
        petal_tube_length = max(0.0, _varied(
            "petal_tube_length", p.petal_tube_length, p.petal_tube_length_variance, p.seed))
        # Match Disc Radius (the default) takes the ALREADY-VARIED disc radius,
        # not the authored one, so the whole flower rerolls off one master
        # dimension instead of the tube drifting away from the disc on every
        # new Seed. See `petal_tube_match_disc`'s docstring.
        petal_tube_radius = disc_radius if p.petal_tube_match_disc else max(
            _MIN_RADIUS, _varied(
                "petal_tube_radius", p.petal_tube_radius, p.petal_tube_radius_variance, p.seed))
    else:
        petal_tube_length = 0.0
        petal_tube_radius = 0.0

    petal_spec = _PetalGroupSpec(
        name_prefix="petal", base_radius=disc_radius, seed=p.seed,
        count=p.petal_count, count_variance=p.petal_count_variance,
        length=p.petal_length, length_variance=p.petal_length_variance,
        width=p.petal_width, width_variance=p.petal_width_variance,
        width_profile=p.petal_width_profile,
        scale=p.petal_scale, scale_variance=p.petal_scale_variance,
        scale_age_profile=p.petal_scale_age_profile,
        resolution=p.resolution, width_resolution=p.petal_width_resolution,
        sink=p.petal_sink, sink_variance=p.petal_sink_variance,
        vertical_offset=p.petal_vertical_offset,
        vertical_offset_variance=p.petal_vertical_offset_variance,
        spread_horizontal=p.petal_spread_horizontal,
        spread_horizontal_variance=p.petal_spread_horizontal_variance,
        tube_enabled=p.petal_tube_enabled, tube_length=petal_tube_length,
        tube_radius=petal_tube_radius, tube_resolution=p.petal_tube_resolution,
        tube_profile=p.petal_tube_profile, tube_noise=p.petal_tube_noise,
        tube_match_angle=p.petal_tube_match_angle, tube_fuse=p.petal_tube_fuse,
        bud_tube_length=p.petal_bud_tube_length, bud_tube_flare=p.petal_bud_tube_flare,
        clamp_overlap=p.petal_clamp_overlap,
        azimuth_offset=0.0, z_offset=0.0,
        layers=p.petal_layers, layer_spacing=p.petal_layer_spacing,
        layer_spacing_variance=p.petal_layer_spacing_variance,
        layer_divergence=p.petal_layer_divergence,
        curl_updown=p.petal_curl_updown, curl_updown_variance=p.petal_curl_updown_variance,
        curl_updown_profile=p.petal_curl_updown_profile,
        curl_updown_age_profile=p.petal_curl_updown_age_profile,
        curl_leftright=p.petal_curl_leftright,
        curl_leftright_variance=p.petal_curl_leftright_variance,
        curl_leftright_profile=p.petal_curl_leftright_profile,
        curl_outin=p.petal_curl_outin, curl_outin_variance=p.petal_curl_outin_variance,
        curl_outin_profile=p.petal_curl_outin_profile,
        curl_roll=p.petal_curl_roll, curl_roll_variance=p.petal_curl_roll_variance,
        gravity=p.petal_gravity, gravity_variance=p.petal_gravity_variance,
        gravity_profile=p.petal_gravity_profile,
        twist=p.petal_twist, twist_variance=p.petal_twist_variance,
        twist_profile=p.petal_twist_profile,
        rotation_yaw=p.petal_rotation_yaw, rotation_yaw_variance=p.petal_rotation_yaw_variance,
        rotation_pitch=p.petal_rotation_pitch,
        rotation_pitch_variance=p.petal_rotation_pitch_variance,
        rotation_pitch_age_profile=p.petal_rotation_pitch_age_profile,
        rotation_roll=p.petal_rotation_roll, rotation_roll_variance=p.petal_rotation_roll_variance,
        bloom=p.bloom, bloom_start=p.petal_bloom_start, bloom_end=p.petal_bloom_end,
        bud_rotation_pitch=p.petal_bud_rotation_pitch,
        bud_curl=p.petal_bud_curl, bud_curl_profile=p.petal_bud_curl_profile,
        bud_sink=p.petal_bud_sink, bud_clamp_overlap=p.petal_bud_clamp_overlap,
        bud_tip_start=p.petal_bud_tip_start,
        bud_tip_lift=p.petal_bud_tip_lift,
        bud_tip_height_profile=p.petal_bud_tip_height_profile,
        bud_tip_radius_profile=p.petal_bud_tip_radius_profile,
        bud_tip_falloff=p.petal_bud_tip_falloff,
        whorl_length_profile=p.petal_whorl_length_profile,
        whorl_width_profile=p.petal_whorl_width_profile,
        whorl_pitch_profile=p.petal_whorl_pitch_profile,
    )
    petals, petal_fusion_tube, petal_base_wrap = (
        _build_petal_group(petal_spec) if p.petal_enabled
        else ([], DiscSpine(points=[], radii=[]), [0.0] * max(2, p.petal_width_resolution)))

    sepals: list[PetalSpine] = []
    sepal_fusion_tube = DiscSpine(points=[], radii=[])
    sepal_base_wrap = [0.0] * max(2, p.sepal_width_resolution)
    if p.sepal_enabled:
        # See the petal tube above.
        if p.sepal_tube_enabled:
            sepal_tube_length = max(0.0, _varied(
                "sepal_tube_length", p.sepal_tube_length, p.sepal_tube_length_variance, p.seed))
            sepal_tube_radius = disc_radius if p.sepal_tube_match_disc else max(
                _MIN_RADIUS, _varied(
                    "sepal_tube_radius", p.sepal_tube_radius,
                    p.sepal_tube_radius_variance, p.seed))
        else:
            sepal_tube_length = 0.0
            sepal_tube_radius = 0.0
        sepal_spec = _PetalGroupSpec(
            name_prefix="sepal", base_radius=disc_radius, seed=p.seed,
            count=p.sepal_count, count_variance=p.sepal_count_variance,
            length=p.sepal_length, length_variance=p.sepal_length_variance,
            width=p.sepal_width, width_variance=p.sepal_width_variance,
            width_profile=p.sepal_width_profile,
            scale=p.sepal_scale, scale_variance=p.sepal_scale_variance,
            scale_age_profile=p.sepal_scale_age_profile,
            resolution=p.sepal_resolution, width_resolution=p.sepal_width_resolution,
            sink=p.sepal_sink, sink_variance=p.sepal_sink_variance,
            vertical_offset=p.sepal_vertical_offset,
            vertical_offset_variance=p.sepal_vertical_offset_variance,
            spread_horizontal=p.sepal_spread_horizontal,
            spread_horizontal_variance=p.sepal_spread_horizontal_variance,
            tube_enabled=p.sepal_tube_enabled, tube_length=sepal_tube_length,
            tube_radius=sepal_tube_radius, tube_resolution=p.sepal_tube_resolution,
            tube_profile=p.sepal_tube_profile, tube_noise=p.sepal_tube_noise,
            tube_match_angle=p.sepal_tube_match_angle, tube_fuse=p.sepal_tube_fuse,
            bud_tube_length=p.sepal_bud_tube_length, bud_tube_flare=p.sepal_bud_tube_flare,
            clamp_overlap=p.sepal_clamp_overlap,
            azimuth_offset=p.sepal_azimuth_offset, z_offset=p.sepal_z_offset,
            layers=p.sepal_layers, layer_spacing=p.sepal_layer_spacing,
            layer_spacing_variance=p.sepal_layer_spacing_variance,
            layer_divergence=p.sepal_layer_divergence,
            curl_updown=p.sepal_curl_updown, curl_updown_variance=p.sepal_curl_updown_variance,
            curl_updown_profile=p.sepal_curl_updown_profile,
            curl_updown_age_profile=p.sepal_curl_updown_age_profile,
            curl_leftright=p.sepal_curl_leftright,
            curl_leftright_variance=p.sepal_curl_leftright_variance,
            curl_leftright_profile=p.sepal_curl_leftright_profile,
            curl_outin=p.sepal_curl_outin, curl_outin_variance=p.sepal_curl_outin_variance,
            curl_outin_profile=p.sepal_curl_outin_profile,
            curl_roll=p.sepal_curl_roll, curl_roll_variance=p.sepal_curl_roll_variance,
            gravity=p.sepal_gravity, gravity_variance=p.sepal_gravity_variance,
            gravity_profile=p.sepal_gravity_profile,
            twist=p.sepal_twist, twist_variance=p.sepal_twist_variance,
            twist_profile=p.sepal_twist_profile,
            rotation_yaw=p.sepal_rotation_yaw,
            rotation_yaw_variance=p.sepal_rotation_yaw_variance,
            rotation_pitch=p.sepal_rotation_pitch,
            rotation_pitch_variance=p.sepal_rotation_pitch_variance,
            rotation_pitch_age_profile=p.sepal_rotation_pitch_age_profile,
            rotation_roll=p.sepal_rotation_roll,
            rotation_roll_variance=p.sepal_rotation_roll_variance,
            bloom=p.bloom, bloom_start=p.sepal_bloom_start, bloom_end=p.sepal_bloom_end,
            bud_rotation_pitch=p.sepal_bud_rotation_pitch,
            bud_curl=p.sepal_bud_curl, bud_curl_profile=p.sepal_bud_curl_profile,
            bud_sink=p.sepal_bud_sink, bud_clamp_overlap=p.sepal_bud_clamp_overlap,
            bud_tip_start=p.sepal_bud_tip_start,
            bud_tip_lift=p.sepal_bud_tip_lift,
            bud_tip_height_profile=p.sepal_bud_tip_height_profile,
            bud_tip_radius_profile=p.sepal_bud_tip_radius_profile,
            bud_tip_falloff=p.sepal_bud_tip_falloff,
        )
        sepals, sepal_fusion_tube, sepal_base_wrap = _build_petal_group(sepal_spec)

    disc = (
        _build_disc(disc_radius, disc_height, p.disc_profile, max(2, p.disc_resolution))
        if p.disc_enabled else DiscSpine(points=[], radii=[])
    )

    if p.receptacle_enabled:
        # See petal_tube_radius above for why Match Disc uses the varied radius.
        receptacle_radius = disc_radius if p.receptacle_match_disc else max(
            _MIN_RADIUS, _varied(
                "receptacle_radius", p.receptacle_radius,
                p.receptacle_radius_variance, p.seed))
        receptacle_height = abs(_varied(
            "receptacle_height", p.receptacle_height, p.receptacle_height_variance, p.seed))
        receptacle = _build_disc(
            receptacle_radius, -receptacle_height, p.receptacle_profile,
            max(2, p.receptacle_resolution))
    else:
        receptacle = DiscSpine(points=[], radii=[])

    # Bud Cup: a true pair with the open-pose Cup (see `petal_bud_cup`
    # docstring), lerped in as this whorl's own local bloom falls, using the
    # SAME per-whorl window as the bend/sink/tip-lift blends above
    # (`_local_bloom`), not a separate one.
    petal_cup = _varied("petal_cup", p.petal_cup, p.petal_cup_variance, p.seed)
    petal_cup_local_bloom = _local_bloom(
        p.bloom, p.petal_bloom_start, p.petal_bloom_end)
    petal_cup = p.petal_bud_cup + (petal_cup - p.petal_bud_cup) * petal_cup_local_bloom
    # Cap amounts: whorl-wide global rerolls, exactly like Cup above and for
    # the same reason (the carrier is one shared array per whorl, so per-petal
    # values aren't reachable). No Maturity scale and no bud multiplier -- these
    # are already relative to the ring's own live width, so they follow every
    # resize for free.
    petal_top_shape = _varied(
        "petal_top_shape", p.petal_top_shape, p.petal_top_shape_variance, p.seed)
    petal_bottom_shape = _varied(
        "petal_bottom_shape", p.petal_bottom_shape, p.petal_bottom_shape_variance, p.seed)
    sepal_top_shape = _varied(
        "sepal_top_shape", p.sepal_top_shape, p.sepal_top_shape_variance, p.seed)
    sepal_bottom_shape = _varied(
        "sepal_bottom_shape", p.sepal_bottom_shape, p.sepal_bottom_shape_variance, p.seed)

    sepal_cup = _varied("sepal_cup", p.sepal_cup, p.sepal_cup_variance, p.seed)
    sepal_cup_local_bloom = _local_bloom(
        p.bloom, p.sepal_bloom_start, p.sepal_bloom_end)
    sepal_cup = p.sepal_bud_cup + (sepal_cup - p.sepal_bud_cup) * sepal_cup_local_bloom

    # Maturity: one global uniform size multiplier (see FlowerMeshParams' `bloom`/
    # `maturity` docstring) baked into the geometry as a final scale-about-origin
    # pass -- curl/twist are angles (scale-invariant) so this is exactly a uniform
    # resize of the whole built flower, fully decoupled from every shape control.
    # A flat default profile at maturity 1.0 gives ×1.0 (a no-op), so this doesn't
    # touch an existing flower.
    mscale = p.maturity_scale_profile.evaluate(p.maturity)
    # Cup is GN-side (a post-mesh push, see `blender/flower.py`'s `_apply_cup`),
    # not baked into these skeleton points/widths, so it needs its own explicit
    # Maturity scale here.
    petal_cup *= mscale
    sepal_cup *= mscale
    if abs(mscale - 1.0) > 1e-9:
        for group in (petals, sepals):
            for spine in group:
                spine.points = [scale(pt, mscale) for pt in spine.points]
                spine.widths = [w * mscale for w in spine.widths]
        for body in (disc, receptacle, petal_fusion_tube, sepal_fusion_tube):
            body.points = [scale(pt, mscale) for pt in body.points]
            body.radii = [r * mscale for r in body.radii]

    # Attachment origin: this object's local (0,0,0) is what actually lands on
    # a stem's attachment point once instanced there (see
    # plant._write_attachment_points' rotation convention) -- it needs to be
    # the flower's true BASE (the receptacle's lowest point, or the rim if
    # there's no receptacle), not the rim itself. Petals/Sepals/Disc/Receptacle
    # are all built with Z=0 at the rim (see module docstring), so shift every
    # point up by however far the receptacle actually reaches below it -- a
    # no-op (0.0) when there's no receptacle, leaving that case exactly as it
    # was (the rim IS already the attach point then). Computed AFTER the
    # Maturity pass above so it accounts for that pass's own resize of the
    # receptacle's depth.
    base_offset = abs(receptacle.points[-1][2]) if receptacle.points else 0.0
    if base_offset > 1e-9:
        shift = (0.0, 0.0, base_offset)
        for group in (petals, sepals):
            for spine in group:
                spine.points = [add(pt, shift) for pt in spine.points]
        for body in (disc, receptacle, petal_fusion_tube, sepal_fusion_tube):
            body.points = [add(pt, shift) for pt in body.points]

    return FlowerSkeleton(
        petals=petals, sepals=sepals, disc=disc, receptacle=receptacle,
        petal_cup=petal_cup, sepal_cup=sepal_cup,
        petal_fusion=petal_fusion_tube, sepal_fusion=sepal_fusion_tube,
        # Top: the authored cap alone. Bottom: the automatic base fit, PLUS
        # whatever the user authored on top of it -- Bottom Shape is an
        # additive override, not a replacement, so amount 0 still lands the
        # base flush on its attachment circle (see `_base_wrap_carrier`).
        petal_top_shape=_build_shape_carrier(
            p.petal_top_shape_profile, petal_top_shape, max(2, p.petal_width_resolution)),
        petal_bottom_shape=_add_carriers(
            petal_base_wrap,
            _build_shape_carrier(p.petal_bottom_shape_profile, petal_bottom_shape,
                                 max(2, p.petal_width_resolution))),
        sepal_top_shape=_build_shape_carrier(
            p.sepal_top_shape_profile, sepal_top_shape, max(2, p.sepal_width_resolution)),
        sepal_bottom_shape=_add_carriers(
            sepal_base_wrap,
            _build_shape_carrier(p.sepal_bottom_shape_profile, sepal_bottom_shape,
                                 max(2, p.sepal_width_resolution))))
