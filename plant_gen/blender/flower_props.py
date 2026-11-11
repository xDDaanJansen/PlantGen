"""A FlowerGen flower's settings (`Object.plantgen_flower`), every field
rebuilding the flower live. Moved from the parametric add-on's `properties.py`; the
attribute keeps its name, so flowers saved before the move load unchanged."""

import math

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import Material, PropertyGroup

from . import flower


def _flower_live_update(self, context):
    obj = flower.owner_of_flower(self)
    if obj is not None:
        flower.rebuild_flower(obj)


def _flower_checker_update(self, context):
    from . import flower_material
    flower_material.set_texture_shading(context, self.uv_checker)
    _flower_live_update(self, context)


def _variance_kwargs(
    label: str, soft_max: float, subtype: str = "NONE", update=None,
) -> dict:
    """Kwargs for an absolute-variance slider (see `model.parameter.Parameter.variance`)
    paired with a base field of the same name + `_variance`, in the SAME units/range
    as that base field (e.g. Length=3 with variance=1 lands anywhere from 2 to 4, not
    a fraction of 3). Edited via the small popup button `_slider_with_variance` draws
    next to its base slider (`ui.py`), not shown inline — keeps the panel from
    doubling in height. `update` defaults to `_live_update` (stem fields); Leaves/
    Flowers fields pass `_phyllotaxis_owner_update` instead."""
    return dict(
        name=f"{label} Variance",
        description=f"Random +/- deviation of {label} per re-roll of Seed, in the "
        "same units as its base value (0 = none)",
        default=0.0, min=0.0, soft_max=soft_max, subtype=subtype,
        update=update or _flower_live_update)


class PLANTGEN_PG_flower(PropertyGroup):
    """A standalone FlowerGen flower's own settings, on `obj.plantgen_flower` --
    mirrors `PLANTGEN_PG_stem`'s per-object-state shape (see `blender/flower.py`).
    Milestone 1 (Actinomorphic/Rotate, Solitary): each component -- petals, disc --
    is independently parameterized, mirroring `PLANTGEN_PG_stem`'s own
    Orientation/Shape/Curl split (see `skeleton/flower.py`'s module docstring)."""

    # --- development: two flower-global "over time" masters (see
    # skeleton/flower.py's `bloom`/`maturity` docstring). Defaults are no-ops so
    # existing flowers are unchanged. ---
    bloom: FloatProperty(
        name="Bloom", description="Open the flower (1) or close it into a bud (0). "
        "Blends each whorl's own Bud Curl in underneath its authored open-pose "
        "Curl Up/Down -- Bloom rising IS the petals/sepals physically uncurling. "
        "Each whorl maps this through its own Bloom Start/End window (see Bud "
        "under Petals/Sepals), so petals and sepals close/open together while "
        "opening on their own schedule. Keyframe it for a bud-to-bloom animation. "
        "0 here is remapped internally onto the best-looking closed pose found "
        "by testing (not literally zero bend) -- see blender/flower.py's "
        "_BLOOM_UI_FLOOR",
        default=1.0, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)
    maturity: FloatProperty(
        name="Maturity", description="Global development/size 'time' for the whole "
        "flower, shaped by the Maturity Scale profile into an overall size "
        "multiplier -- independent of Bloom (how open it is) and of per-layer Age",
        default=1.0, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)

    # Unlike sepal_enabled/receptacle_enabled below, Petals default ON -- they're
    # the flower's primary/always-present structure, not an optional add-on.
    petal_enabled: BoolProperty(
        name="Enable", description="Generate petals", default=True,
        update=_flower_live_update)

    # --- petals: placement + width ---
    petal_count: IntProperty(
        name="Count", description="Number of petals per layer, evenly spaced "
        "around the disc", default=5, min=3, max=32, update=_flower_live_update)
    petal_count_variance: FloatProperty(
        **_variance_kwargs("Count", soft_max=10.0, update=_flower_live_update))
    petal_length: FloatProperty(
        name="Length", description="Petal length from the disc edge to its tip",
        default=0.02, min=0.0001, soft_max=1.0, subtype="DISTANCE", update=_flower_live_update)
    petal_length_variance: FloatProperty(
        **_variance_kwargs("Length", soft_max=1.0, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_width: FloatProperty(
        name="Width", description="Full petal width at the Width Profile curve's peak",
        default=0.01, min=0.0001, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    petal_width_variance: FloatProperty(
        **_variance_kwargs("Width", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_top_shape: FloatProperty(
        name="Top Shape", description="Shapes the petal's TIP cap edge, using "
        "the Top Shape curve as a profile across the petal's WIDTH -- a "
        "different axis from the Width Profile, which shapes the sides along "
        "its LENGTH. Measured in multiples of the tip's own width, so a Dome "
        "curve at 0.5 is a semicircular tip at any petal size; negative "
        "notches the cap inward instead. 0 leaves the flat, straight-cut tip",
        default=0.0, soft_min=-1.0, soft_max=1.0, update=_flower_live_update)
    petal_top_shape_variance: FloatProperty(
        **_variance_kwargs("Top Shape", soft_max=1.0, update=_flower_live_update))
    petal_bottom_shape: FloatProperty(
        name="Bottom Shape", description="Shapes the petal's BASE cap edge, "
        "same units and curve convention as Top Shape. Added ON TOP of the "
        "automatic fit that already curves the base to match whatever it "
        "attaches to (the disc rim, or a corolla tube's mouth), so 0 is not a "
        "flat cut here -- it's the shape that meets that surface cleanly",
        default=0.0, soft_min=-1.0, soft_max=1.0, update=_flower_live_update)
    petal_bottom_shape_variance: FloatProperty(
        **_variance_kwargs("Bottom Shape", soft_max=1.0, update=_flower_live_update))
    petal_cup: FloatProperty(
        name="Cup", description="Bows the petal's CROSS-SECTION (across the "
        "width, not along its length) into a shallow channel/dome, as a real "
        "distance push scaled by the ring's own local width -- fades to 0 "
        "wherever the petal itself tapers to nothing (the tip, and a narrow "
        "base). One shared value for every petal on this flower, not per-petal "
        "like the fields above",
        default=0.0, soft_min=-0.01, soft_max=0.01, subtype="DISTANCE",
        update=_flower_live_update)
    petal_cup_variance: FloatProperty(
        **_variance_kwargs("Cup", soft_max=0.01, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_bud_cup: FloatProperty(
        name="Bud Cup", description="This whorl's Cup at local Bloom 0 "
        "(closed), fading to the authored open-pose Cup above as it opens -- "
        "the same true-pair pattern as Bud Pitch/Bud Curl/Bud Sink. The same "
        "cross-section cup reads flatter once everything's pinched down "
        "toward the axis, so the bud pose typically wants noticeably MORE "
        "cup than the open pose to still read as cupped -- authored "
        "independently here rather than derived from it",
        default=0.0, soft_min=-0.01, soft_max=0.01, subtype="DISTANCE",
        update=_flower_live_update)
    petal_scale: FloatProperty(
        name="Scale", description="Overall size multiplier (scales Length and "
        "Width together), shaped across layers by its own Age Profile curve -- "
        "e.g. lets the topmost/innermost layer read as younger and smaller",
        default=1.0, min=0.0001, soft_max=2.0, update=_flower_live_update)
    petal_scale_variance: FloatProperty(
        **_variance_kwargs("Scale", soft_max=2.0, update=_flower_live_update))
    resolution: IntProperty(
        name="Resolution", description="Points along each petal's spine", default=8,
        min=2, max=64, update=_flower_live_update)
    petal_width_resolution: IntProperty(
        name="Width Resolution", description="Points across each petal (Resolution is "
        "along it). With Fuse To Tube on, the tube is revolved with Count x (this - 1) "
        "sides to weld to it", default=16, min=3, max=64, update=_flower_live_update)
    petal_sink: FloatProperty(
        name="Radial Offset", description="Signed distance moving each petal's "
        "BASE along the disc's own radius. Positive pulls it inward, toward the "
        "disc's axis; negative pushes it OUTWARD instead, past the disc rim, "
        "like a pedicel -- same convention as Leaves/Flowers' own Radial Offset "
        "on a stem. This is a deliberate placement offset, NOT a seam fix: the "
        "base cap is already fitted to whatever it attaches to automatically, "
        "so 0 should look correct on its own",
        default=0.0, soft_min=-0.02, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    petal_sink_variance: FloatProperty(
        **_variance_kwargs("Radial Offset", soft_max=0.02, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_vertical_offset: FloatProperty(
        name="Vertical Offset", description="Signed Z distance moving each "
        "petal's BASE up(+) or down(-) -- lets the base meet a Fusion tube's "
        "own rim, or the Disc's surface, exactly when Curvature/Pitch "
        "differences leave a visible gap or overlap there",
        default=0.0, soft_min=-0.02, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    petal_vertical_offset_variance: FloatProperty(
        **_variance_kwargs("Vertical Offset", soft_max=0.02, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_spread_horizontal: FloatProperty(
        name="Spread Horizontal", description="Random scatter of each petal's "
        "BASE azimuth around its ideal evenly-spaced ring position -- unlike "
        "Yaw (under Shape), this moves WHERE the base sits rather than which "
        "way it points",
        default=0.0, min=0.0, soft_max=0.5, subtype="ANGLE", update=_flower_live_update)
    petal_spread_horizontal_variance: FloatProperty(
        **_variance_kwargs("Spread Horizontal", soft_max=0.5, subtype="ANGLE",
                            update=_flower_live_update))
    petal_clamp_overlap: FloatProperty(
        name="Clamp Overlap", description="Limits how wide neighbouring petals "
        "may get before they meet. 0 is off -- they overlap freely, which many "
        "real polypetalous flowers do. 1 makes their edges land exactly on each "
        "other, with no gap and no intersection. Below 1 allows proportionally "
        "more overlap; above 1 pulls them narrower than touching, opening a "
        "deliberate gap between petals. Applied along the whole length, not "
        "just at the base -- the room available widens as the petal reaches "
        "further from the axis",
        default=0.0, min=0.0, soft_max=2.0, update=_flower_live_update)

    # --- corolla tube: a body of revolution (same mechanism as Disc/
    # Receptacle), its own independent Length/Radius/Resolution/Profile --
    # see skeleton/flower.py's petal_tube_enabled docstring for why this
    # isn't derived from Petal Length/Disc Radius. Disabled by default, same
    # reasoning as Receptacle.
    petal_tube_enabled: BoolProperty(
        name="Enable", description="Generate a shared corolla tube the petal "
        "lobes emerge from (Sympetalous/Gamopetalous)",
        default=False, update=_flower_live_update)
    petal_tube_length: FloatProperty(
        name="Length", description="Length of the corolla tube, independent "
        "of Petal Length (the free lobe's own length)",
        default=0.014, min=0.0, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    petal_tube_length_variance: FloatProperty(
        **_variance_kwargs("Length", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_tube_match_disc: BoolProperty(
        name="Match Disc Radius", description="Take the tube's base radius "
        "straight from Disc Radius, so the tube meets the disc rim seamlessly "
        "and the whole flower resizes off one master dimension. Only the "
        "BASE is linked -- the Tube Profile curve still shapes the radius from "
        "there up, so a funnel or bell still flares. Untick to set it "
        "independently",
        default=True, update=_flower_live_update)
    petal_tube_radius: FloatProperty(
        name="Radius", description="Radius of the corolla tube at its base, "
        "independent of Disc Radius",
        default=0.006, min=0.0001, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    petal_tube_radius_variance: FloatProperty(
        **_variance_kwargs("Radius", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    petal_tube_resolution: IntProperty(
        name="Tube Resolution", description="Points along the tube's own "
        "profile (more = smoother curvature)", default=8, min=2, max=64,
        update=_flower_live_update)
    petal_tube_sides: IntProperty(
        name="Sides", description="Points around the tube (with Fuse To Tube on, set "
        "by the petals instead)", default=16, min=3,
        max=256, update=_flower_live_update)
    petal_tube_noise: FloatProperty(
        name="Noise", description="Irregularity along the tube's length -- a "
        "slow wobble in its radius, so it doesn't read as a machined "
        "extrusion. Rerolls with the flower's Seed, so a field of one species "
        "varies instead of repeating a single silhouette. The base is left "
        "untouched, so this can't reopen the join with the disc",
        default=0.0, min=0.0, soft_max=0.3, subtype="FACTOR",
        update=_flower_live_update)
    petal_tube_match_angle: BoolProperty(
        name="Match Tube Angle", description="Start the petals off at the "
        "angle of the tube's own mouth, so they read as a continuation of the "
        "wall instead of straps attached at an unrelated tilt -- and so "
        "reshaping the top of the Tube Profile carries the petals with it. "
        "Pitch (under Shape) becomes an offset from that angle rather than an "
        "absolute tilt while this is on",
        default=False, update=_flower_live_update)
    petal_bud_tube_length: FloatProperty(
        name="Bud Length", description="The tube's length in bud, as a fraction of "
        "its open Length -- it grows to full length over the Bloom slider. 1 keeps "
        "it the same at every Bloom", default=1.0, min=0.05, max=1.0,
        subtype='FACTOR', update=_flower_live_update)
    petal_bud_tube_flare: FloatProperty(
        name="Bud Flare", description="How much of the Tube Profile's widening the "
        "tube has in bud: 0 a straight tube at the base radius, 1 the full open "
        "shape -- it swells into it over the Bloom slider. The base keeps its "
        "radius, so the disc and receptacle stay matched", default=1.0, min=0.0,
        max=1.0, subtype='FACTOR', update=_flower_live_update)
    petal_tube_fuse: BoolProperty(
        name="Fuse Petals To Tube", description="Widen each petal's base until "
        "they tile the tube's mouth exactly, closing the gaps between them so "
        "the corolla reads as one piece rather than separate lobes stuck to a "
        "cylinder. Fades back into the petal's own outline just above the "
        "mouth, so it doesn't override the Petal Shape",
        default=False, update=_flower_live_update)

    # --- petals: layering (stacked whorls) -- see skeleton/flower.py's
    # FlowerMeshParams.petal_layers docstring for the golden-angle reasoning ---
    petal_layers: IntProperty(
        name="Layers", description="Number of stacked petal rings (whorls), each "
        "offset up/down by Spacing and rotated by Spiral from the last -- a "
        "double-flower / many-petaled look. 1 = a single ring",
        default=1, min=1, max=12, update=_flower_live_update)
    petal_layer_spacing: FloatProperty(
        name="Spacing", description="Vertical distance between successive petal "
        "layers -- typically well under 1mm",
        default=0.0, soft_min=-0.001, soft_max=0.001, subtype="DISTANCE",
        precision=5, update=_flower_live_update)
    petal_layer_spacing_variance: FloatProperty(
        **_variance_kwargs("Spacing", soft_max=0.001, subtype="DISTANCE",
                            update=_flower_live_update), precision=5)
    petal_layer_divergence: FloatProperty(
        name="Spiral", description="Azimuthal rotation applied to each successive "
        "layer, so layers interleave instead of stacking directly on top of each "
        "other. Defaults to the golden angle (~137.5°), the same botanical spiral "
        "stems use for Golden phyllotaxis",
        default=math.pi * (3.0 - math.sqrt(5.0)), min=0.0, max=6.2832,
        subtype="ANGLE", update=_flower_live_update)

    # --- petals: shape (SpeedTree-style 3-axis Curl + Gravity + Twist, mirrors
    # PLANTGEN_PG_stem's own fields almost exactly -- see skeleton/flower.py) ---
    petal_curl_updown: FloatProperty(
        name="Curl Up/Down", description="Steady bend up (+) or down (-) out of "
        "the flat disc plane, shaped along the petal by its own profile curve. "
        "1.0 = 180° total, independent of Length -- keeps coiling tighter the "
        "higher you push it, no clamp (unlike Gravity)",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    petal_curl_updown_variance: FloatProperty(
        **_variance_kwargs("Curl Up/Down", soft_max=4.0, update=_flower_live_update))
    petal_curl_leftright: FloatProperty(
        name="Curl Left/Right", description="Steady sideways bend, curving the "
        "petal into a sickle/scimitar shape as it extends outward. 1.0 = 180° "
        "total, independent of Length",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    petal_curl_leftright_variance: FloatProperty(
        **_variance_kwargs("Curl Left/Right", soft_max=4.0, update=_flower_live_update))
    petal_curl_outin: FloatProperty(
        name="Curl Out/In", description="Corkscrews the petal's lean around the "
        "disc's own central (world Z) axis. 1.0 = 180° total, independent of Length",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    petal_curl_outin_variance: FloatProperty(
        **_variance_kwargs("Curl Out/In", soft_max=4.0, update=_flower_live_update))
    petal_curl_roll: FloatProperty(
        name="Curl Roll", description="Redirects which plane Curl Up/Down and "
        "Gravity bend the petal in -- doesn't move a flat petal by itself. A "
        "second, independent Roll from Rotation's own (which instead spins the "
        "already-built cross-section, visible even at zero Curl)",
        default=0.0, soft_min=-6.2832, soft_max=6.2832, subtype="ANGLE",
        update=_flower_live_update)
    petal_curl_roll_variance: FloatProperty(
        **_variance_kwargs("Curl Roll", soft_max=6.2832, subtype="ANGLE",
                            update=_flower_live_update))
    petal_gravity: FloatProperty(
        name="Gravity", description="Bends each petal toward world-up (+) or "
        "world-down (-), shaped by the Gravity profile curve. Settles at vertical "
        "and can't overshoot past it, however high the value",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    petal_gravity_variance: FloatProperty(**_variance_kwargs("Gravity", soft_max=4.0,
                                                               update=_flower_live_update))
    petal_twist: FloatProperty(
        name="Twist", description="Total rotation of the petal's cross-section "
        "around its own spine, shaped by the Twist profile curve; doesn't bend the "
        "spine itself. 1.0 = 180°, independent of Length",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    petal_twist_variance: FloatProperty(**_variance_kwargs("Twist", soft_max=4.0,
                                                             update=_flower_live_update))
    # Rotation: a ONE-TIME base orientation (like the stem's own Start Angle +
    # Roll), distinct from Curl above -- see skeleton/flower.py's
    # FlowerMeshParams rotation-fields docstring for the full Yaw/Pitch/Roll
    # reasoning (Yaw moves which way the petal points; Pitch statically tilts it
    # up/down once; Roll redirects which plane the Curl axes bend in afterward).
    petal_rotation_yaw: FloatProperty(
        name="Yaw", description="One-time rotation of each petal's pointing "
        "direction around the disc's own normal, independent of where its base "
        "attaches (which stays evenly spaced on the rim) -- fans petals unevenly "
        "for an overlapping, pinwheel look. Positive Variance alone is often enough",
        default=0.0, soft_min=-1.5708, soft_max=1.5708, subtype="ANGLE",
        update=_flower_live_update)
    petal_rotation_yaw_variance: FloatProperty(
        **_variance_kwargs("Yaw", soft_max=1.5708, subtype="ANGLE",
                            update=_flower_live_update))
    petal_rotation_pitch: FloatProperty(
        name="Pitch", description="One-time tilt of each petal's starting "
        "direction up/down out of the flat disc plane, before any Curl/Gravity "
        "bending is applied on top",
        default=0.0, soft_min=-1.5708, soft_max=1.5708, subtype="ANGLE",
        update=_flower_live_update)
    petal_rotation_pitch_variance: FloatProperty(
        **_variance_kwargs("Pitch", soft_max=1.5708, subtype="ANGLE",
                            update=_flower_live_update))
    petal_rotation_roll: FloatProperty(
        name="Roll", description="Spins the petal's cross-section around its own "
        "local direction (length) axis, sideways -- visible along the whole "
        "petal, including the base, even with zero Curl/Twist",
        default=0.0, soft_min=-6.2832, soft_max=6.2832, subtype="ANGLE",
        update=_flower_live_update)
    petal_rotation_roll_variance: FloatProperty(
        **_variance_kwargs("Roll", soft_max=6.2832, subtype="ANGLE",
                            update=_flower_live_update))

    # --- petals: bud closure (see skeleton/flower.py's `bloom`/`petal_bud_curl`
    # docstring) -- how THIS whorl responds to the flower-wide Bloom master. ---
    petal_bud_rotation_pitch: FloatProperty(
        name="Bud Pitch", description="EXPERIMENTAL: this whorl's Rotation "
        "Pitch at local Bloom 0 (near-vertical, closed), fading to the "
        "authored open-pose Pitch above as it opens -- a rigid hinge swing "
        "from pointing up to pointing out, which is how a real flower's bloom "
        "reads mostly (~70-80%), with Curl only providing a minor unfurl "
        "(~20-30%) on top. Unlike Curl, a rigid rotation from vertical can't "
        "wrap past itself, so this is inherently monotonic regardless of angle",
        default=1.3, soft_min=0.0, soft_max=1.5708, subtype="ANGLE",
        update=_flower_live_update)
    petal_bud_curl: FloatProperty(
        name="Bud Curl", description="Extra Curl Up/Down bend fully applied when "
        "this whorl's own local Bloom is 0, fading to zero extra (i.e. exactly "
        "Curl Up/Down above) as it opens -- Bloom rising IS this uncurling. Now "
        "that Bud Pitch (above) carries most of the closing motion, keep this "
        "small (well under 0.3 with the default Bud Curl Profile) -- much "
        "higher and the bend overshoots 180 degrees, so the tip swings past "
        "vertical and back toward/through the axis as it starts to relax, "
        "reading as the tip pulling back before it unfurls outward. The Tip "
        "Shape controls below are the actual pointed-vs-round lever now, so "
        "Bud Curl only needs to add a little extra curve on top of the hinge",
        default=0.15, soft_min=-6.0, soft_max=6.0, update=_flower_live_update)
    petal_bud_sink: FloatProperty(
        name="Bud Sink", description="Radial pull on each petal's own BASE "
        "toward the disc axis, fully applied at local Bloom 0, fading to zero "
        "extra (i.e. exactly Radial Offset) as it opens. Curl alone converges "
        "each petal's TIP to a point but leaves its base at the full open-flower "
        "radius, which still reads as a wide spiky crown rather than a slender "
        "bud -- this presses the bases together near the axis too",
        default=0.005, min=0.0, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    petal_bud_clamp_overlap: FloatProperty(
        name="Bud Clamp Overlap", description="This whorl's own Clamp "
        "Overlap at local Bloom 0 (closed), fading to the authored open-pose "
        "Clamp Overlap (under Shape) as it opens. Bud Sink above already "
        "pulls every base toward the axis as the whorl closes, which shrinks "
        "the touching-width ceiling Clamp Overlap measures against -- "
        "without its own bud value, widths authored for the open radius can "
        "pile up and cover the axis instead of reading as separate furled "
        "petals. 0 = off (matches the open default)",
        default=0.0, min=0.0, soft_max=2.0, update=_flower_live_update)
    petal_bud_tip_start: FloatProperty(
        name="Tip Zone Start", description="Where along the petal (0=base, "
        "1=tip) the tip-shaping zone starts. Height/Radius profiles below are "
        "evaluated across just this zone (0 here -> 1 at the tip), so they "
        "always get full curve resolution regardless of where the zone starts",
        default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)
    petal_bud_tip_lift: FloatProperty(
        name="Bud Tip Lift", description="Overall height magnitude raised into "
        "the tip zone (shaped by the Height profile below), fully applied at "
        "local Bloom 0, fading to zero as it opens. Curl alone always rounds "
        "the apex off into a dome (a constant-curvature arc); this pinches it "
        "up past that arc. 0 = off",
        default=0.0, min=0.0, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    petal_bud_tip_falloff: FloatProperty(
        name="Tip Falloff", description="How fast tip shaping releases as this "
        "whorl opens: tip_fade = (1 - local Bloom) ^ this. 1.0 = plain linear "
        "release across the same range Curl itself unwinds over; higher values "
        "front-load the release (most of it happens early) while still easing "
        "smoothly to exactly 0 at fully open -- no separate cutoff point to "
        "desync from Curl's own pace",
        default=3.0, min=0.1, soft_max=10.0, update=_flower_live_update)
    petal_bloom_start: FloatProperty(
        name="Bloom Start", description="Bloom value below which this whorl is "
        "fully closed (local bloom 0). Defaults to 0.44 -- 80% of the way "
        "through Sepals' own Bloom End (0.55) -- so petals stay closed until "
        "the calyx is nearly done opening, then follow",
        default=0.44, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)
    petal_bloom_end: FloatProperty(
        name="Bloom End", description="Bloom value above which this whorl is "
        "fully open (local bloom 1) -- a narrower window than the sibling whorl's "
        "makes this one finish opening earlier",
        default=1.0, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)

    # --- sepals (calyx): a second, independent whorl of the same kind of ribbon
    # appendage as petals -- full parameter parity, see skeleton/flower.py's
    # sepal_* field docstrings. Disabled by default so adding this field set
    # doesn't change any existing flower's current look.
    sepal_enabled: BoolProperty(
        name="Enable", description="Generate a calyx (sepal whorl) beneath the petals",
        default=False, update=_flower_live_update)
    sepal_count: IntProperty(
        name="Count", description="Number of sepals per layer, evenly spaced "
        "around the disc", default=5, min=3, max=32, update=_flower_live_update)
    sepal_count_variance: FloatProperty(
        **_variance_kwargs("Count", soft_max=10.0, update=_flower_live_update))
    sepal_length: FloatProperty(
        name="Length", description="Sepal length from its base to its tip",
        default=0.008, min=0.0001, soft_max=1.0, subtype="DISTANCE", update=_flower_live_update)
    sepal_length_variance: FloatProperty(
        **_variance_kwargs("Length", soft_max=1.0, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_width: FloatProperty(
        name="Width", description="Full sepal width at the Width Profile curve's peak",
        default=0.006, min=0.0001, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    sepal_width_variance: FloatProperty(
        **_variance_kwargs("Width", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_top_shape: FloatProperty(
        name="Top Shape", description="Mirrors Petals' own Top Shape",
        default=0.0, soft_min=-1.0, soft_max=1.0, update=_flower_live_update)
    sepal_top_shape_variance: FloatProperty(
        **_variance_kwargs("Top Shape", soft_max=1.0, update=_flower_live_update))
    sepal_bottom_shape: FloatProperty(
        name="Bottom Shape", description="Mirrors Petals' own Bottom Shape",
        default=0.0, soft_min=-1.0, soft_max=1.0, update=_flower_live_update)
    sepal_bottom_shape_variance: FloatProperty(
        **_variance_kwargs("Bottom Shape", soft_max=1.0, update=_flower_live_update))
    sepal_cup: FloatProperty(
        name="Cup", description="Mirrors Petals' own Cup",
        default=0.0, soft_min=-0.01, soft_max=0.01, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_cup_variance: FloatProperty(
        **_variance_kwargs("Cup", soft_max=0.01, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_bud_cup: FloatProperty(
        name="Bud Cup", description="Mirrors Petals' own Bud Cup",
        default=0.0, soft_min=-0.01, soft_max=0.01, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_scale: FloatProperty(
        name="Scale", description="Overall size multiplier (scales Length and "
        "Width together), shaped across layers by its own Age Profile curve",
        default=1.0, min=0.0001, soft_max=2.0, update=_flower_live_update)
    sepal_scale_variance: FloatProperty(
        **_variance_kwargs("Scale", soft_max=2.0, update=_flower_live_update))
    sepal_bud_rotation_pitch: FloatProperty(
        name="Bud Pitch", description="Mirrors Petals' own Bud Pitch -- "
        "slightly more vertical by default (the calyx wraps tighter)",
        default=1.4, soft_min=0.0, soft_max=1.5708, subtype="ANGLE",
        update=_flower_live_update)
    sepal_bud_curl: FloatProperty(
        name="Bud Curl", description="Mirrors Petals' own Bud Curl -- extra Curl "
        "Up/Down applied as this whorl closes into a bud. Same overshoot caveat "
        "applies -- see Petals' own Bud Curl",
        default=0.15, soft_min=-6.0, soft_max=6.0, update=_flower_live_update)
    sepal_bud_sink: FloatProperty(
        name="Bud Sink", description="Mirrors Petals' own Bud Sink -- radial pull "
        "on each sepal's own base toward the disc axis as this whorl closes",
        default=0.0045, min=0.0, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_bud_clamp_overlap: FloatProperty(
        name="Bud Clamp Overlap", description="Mirrors Petals' own Bud Clamp Overlap",
        default=0.0, min=0.0, soft_max=2.0, update=_flower_live_update)
    sepal_bud_tip_start: FloatProperty(
        name="Tip Zone Start", description="Mirrors Petals' own Tip Zone Start",
        default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)
    sepal_bud_tip_lift: FloatProperty(
        name="Bud Tip Lift", description="Mirrors Petals' own Bud Tip Lift -- "
        "extra height pinched into the sepal's tip zone as this whorl closes. "
        "Nonzero by default: the calyx is the outermost, visible layer of a "
        "closed bud, so this is where the pointed apex needs to read",
        default=0.004, min=0.0, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_bud_tip_falloff: FloatProperty(
        name="Tip Falloff", description="Mirrors Petals' own Tip Falloff",
        default=3.0, min=0.1, soft_max=10.0, update=_flower_live_update)
    sepal_bloom_start: FloatProperty(
        name="Bloom Start", description="Mirrors Petals' own Bloom Start",
        default=0.0, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)
    sepal_bloom_end: FloatProperty(
        name="Bloom End", description="Mirrors Petals' own Bloom End -- defaults "
        "narrower than Petals' so the calyx finishes opening first",
        default=0.55, min=0.0, max=1.0, subtype="FACTOR", update=_flower_live_update)
    sepal_resolution: IntProperty(
        name="Resolution", description="Points along each sepal's spine", default=6,
        min=2, max=64, update=_flower_live_update)
    sepal_width_resolution: IntProperty(
        name="Width Resolution", description="Points across each sepal (Resolution is "
        "along it). With Fuse To Tube on, the tube is revolved with Count x (this - 1) "
        "sides to weld to it", default=16, min=3, max=64, update=_flower_live_update)
    sepal_sink: FloatProperty(
        name="Radial Offset", description="Signed distance moving each sepal's "
        "BASE along the disc's own radius, same convention as Petals' own Radial Offset",
        default=0.0, soft_min=-0.02, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_sink_variance: FloatProperty(
        **_variance_kwargs("Radial Offset", soft_max=0.02, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_vertical_offset: FloatProperty(
        name="Vertical Offset", description="Mirrors Petals' own Vertical "
        "Offset -- see its docstring",
        default=0.0, soft_min=-0.02, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_vertical_offset_variance: FloatProperty(
        **_variance_kwargs("Vertical Offset", soft_max=0.02, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_azimuth_offset: FloatProperty(
        name="Azimuth Offset", description="One-time rotation of the WHOLE sepal "
        "ring relative to the petal ring -- defaults to alternating with a "
        "5-petal flower (sepals peeking out between petals, not directly behind them)",
        default=math.pi / 5.0, soft_min=-3.1416, soft_max=3.1416, subtype="ANGLE",
        update=_flower_live_update)
    sepal_z_offset: FloatProperty(
        name="Ring Height", description="Signed distance seating the whole sepal "
        "ring below (-) or above (+) the petal attachment plane (each sepal's own "
        "base: Vertical Offset)",
        default=-0.002, soft_min=-0.02, soft_max=0.02, subtype="DISTANCE",
        update=_flower_live_update)
    sepal_spread_horizontal: FloatProperty(
        name="Spread Horizontal", description="Random scatter of each sepal's "
        "BASE azimuth around its ideal evenly-spaced ring position",
        default=0.0, min=0.0, soft_max=0.5, subtype="ANGLE", update=_flower_live_update)
    sepal_spread_horizontal_variance: FloatProperty(
        **_variance_kwargs("Spread Horizontal", soft_max=0.5, subtype="ANGLE",
                            update=_flower_live_update))

    # --- calyx tube: mirrors the petal Corolla Tube above -- see its docstring.
    sepal_tube_enabled: BoolProperty(
        name="Enable", description="Generate a shared calyx tube the sepal "
        "lobes emerge from (Gamosepalous)",
        default=False, update=_flower_live_update)
    sepal_tube_length: FloatProperty(
        name="Length", description="Length of the calyx tube, independent of "
        "Sepal Length (the free lobe's own length)",
        default=0.006, min=0.0, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    sepal_tube_length_variance: FloatProperty(
        **_variance_kwargs("Length", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_tube_match_disc: BoolProperty(
        name="Match Disc Radius", description="Mirrors the Corolla Tube's own "
        "Match Disc Radius -- see its tooltip",
        default=True, update=_flower_live_update)
    sepal_tube_radius: FloatProperty(
        name="Radius", description="Radius of the calyx tube at its base, "
        "independent of Disc Radius",
        default=0.006, min=0.0001, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    sepal_tube_radius_variance: FloatProperty(
        **_variance_kwargs("Radius", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    sepal_clamp_overlap: FloatProperty(
        name="Clamp Overlap", description="Mirrors Petals' own Clamp Overlap",
        default=0.0, min=0.0, soft_max=2.0, update=_flower_live_update)
    sepal_tube_noise: FloatProperty(
        name="Noise", description="Mirrors the Corolla Tube's own Noise",
        default=0.0, min=0.0, soft_max=0.3, subtype="FACTOR",
        update=_flower_live_update)
    sepal_tube_match_angle: BoolProperty(
        name="Match Tube Angle", description="Mirrors the Corolla Tube's own "
        "Match Tube Angle", default=False, update=_flower_live_update)
    sepal_bud_tube_length: FloatProperty(
        name="Bud Length", description="The tube's length in bud, as a fraction of "
        "its open Length -- it grows to full length over the Bloom slider. 1 keeps "
        "it the same at every Bloom", default=1.0, min=0.05, max=1.0,
        subtype='FACTOR', update=_flower_live_update)
    sepal_bud_tube_flare: FloatProperty(
        name="Bud Flare", description="How much of the Tube Profile's widening the "
        "tube has in bud: 0 a straight tube at the base radius, 1 the full open "
        "shape -- it swells into it over the Bloom slider. The base keeps its "
        "radius, so the disc and receptacle stay matched", default=1.0, min=0.0,
        max=1.0, subtype='FACTOR', update=_flower_live_update)
    sepal_tube_fuse: BoolProperty(
        name="Fuse Sepals To Tube", description="Mirrors the Corolla Tube's "
        "own Fuse Petals To Tube", default=False, update=_flower_live_update)
    sepal_tube_resolution: IntProperty(
        name="Tube Resolution", description="Points along the tube's own "
        "profile (more = smoother curvature)", default=6, min=2, max=64,
        update=_flower_live_update)
    sepal_tube_sides: IntProperty(
        name="Sides", description="Points around the tube (with Fuse To Tube on, set "
        "by the sepals instead)", default=16, min=3,
        max=256, update=_flower_live_update)

    # --- sepals: layering ---
    sepal_layers: IntProperty(
        name="Layers", description="Number of stacked sepal rings (whorls), each "
        "offset up/down by Spacing and rotated by Spiral from the last",
        default=1, min=1, max=12, update=_flower_live_update)
    sepal_layer_spacing: FloatProperty(
        name="Spacing", description="Vertical distance between successive sepal "
        "layers -- typically well under 1mm",
        default=0.0, soft_min=-0.001, soft_max=0.001, subtype="DISTANCE",
        precision=5, update=_flower_live_update)
    sepal_layer_spacing_variance: FloatProperty(
        **_variance_kwargs("Spacing", soft_max=0.001, subtype="DISTANCE",
                            update=_flower_live_update), precision=5)
    sepal_layer_divergence: FloatProperty(
        name="Spiral", description="Azimuthal rotation applied to each successive "
        "sepal layer, so layers interleave instead of stacking directly on top "
        "of each other. Defaults to the golden angle (~137.5°)",
        default=math.pi * (3.0 - math.sqrt(5.0)), min=0.0, max=6.2832,
        subtype="ANGLE", update=_flower_live_update)

    # --- sepals: shape (same 3-axis Curl + Gravity + Twist as Petals) ---
    sepal_curl_updown: FloatProperty(
        name="Curl Up/Down", description="Steady bend up (+) or down (-) out of "
        "the flat disc plane, shaped along the sepal by its own profile curve",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    sepal_curl_updown_variance: FloatProperty(
        **_variance_kwargs("Curl Up/Down", soft_max=4.0, update=_flower_live_update))
    sepal_curl_leftright: FloatProperty(
        name="Curl Left/Right", description="Steady sideways bend, curving the "
        "sepal into a sickle/scimitar shape as it extends outward",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    sepal_curl_leftright_variance: FloatProperty(
        **_variance_kwargs("Curl Left/Right", soft_max=4.0, update=_flower_live_update))
    sepal_curl_outin: FloatProperty(
        name="Curl Out/In", description="Corkscrews the sepal's lean around the "
        "disc's own central (world Z) axis",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    sepal_curl_outin_variance: FloatProperty(
        **_variance_kwargs("Curl Out/In", soft_max=4.0, update=_flower_live_update))
    sepal_curl_roll: FloatProperty(
        name="Curl Roll", description="Redirects which plane Curl Up/Down and "
        "Gravity bend the sepal in -- doesn't move a flat sepal by itself",
        default=0.0, soft_min=-6.2832, soft_max=6.2832, subtype="ANGLE",
        update=_flower_live_update)
    sepal_curl_roll_variance: FloatProperty(
        **_variance_kwargs("Curl Roll", soft_max=6.2832, subtype="ANGLE",
                            update=_flower_live_update))
    sepal_gravity: FloatProperty(
        name="Gravity", description="Bends each sepal toward world-up (+) or "
        "world-down (-), shaped by the Gravity profile curve",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    sepal_gravity_variance: FloatProperty(**_variance_kwargs("Gravity", soft_max=4.0,
                                                               update=_flower_live_update))
    sepal_twist: FloatProperty(
        name="Twist", description="Total rotation of the sepal's cross-section "
        "around its own spine, shaped by the Twist profile curve",
        default=0.0, soft_min=-4.0, soft_max=4.0, update=_flower_live_update)
    sepal_twist_variance: FloatProperty(**_variance_kwargs("Twist", soft_max=4.0,
                                                             update=_flower_live_update))

    # --- sepals: rotation ---
    sepal_rotation_yaw: FloatProperty(
        name="Yaw", description="One-time rotation of each sepal's pointing "
        "direction around the disc's own normal",
        default=0.0, soft_min=-1.5708, soft_max=1.5708, subtype="ANGLE",
        update=_flower_live_update)
    sepal_rotation_yaw_variance: FloatProperty(
        **_variance_kwargs("Yaw", soft_max=1.5708, subtype="ANGLE",
                            update=_flower_live_update))
    sepal_rotation_pitch: FloatProperty(
        name="Pitch", description="One-time tilt of each sepal's starting "
        "direction up/down out of the flat disc plane",
        default=0.0, soft_min=-1.5708, soft_max=1.5708, subtype="ANGLE",
        update=_flower_live_update)
    sepal_rotation_pitch_variance: FloatProperty(
        **_variance_kwargs("Pitch", soft_max=1.5708, subtype="ANGLE",
                            update=_flower_live_update))
    sepal_rotation_roll: FloatProperty(
        name="Roll", description="Spins the sepal's cross-section around its own "
        "local direction (length) axis, sideways",
        default=0.0, soft_min=-6.2832, soft_max=6.2832, subtype="ANGLE",
        update=_flower_live_update)
    sepal_rotation_roll_variance: FloatProperty(
        **_variance_kwargs("Roll", soft_max=6.2832, subtype="ANGLE",
                            update=_flower_live_update))

    # --- disc: a body of revolution (radius-vs-height profile, like the stem's
    # own taper) instead of a fixed primitive -- see skeleton/flower.py ---
    # Unlike sepal_enabled/receptacle_enabled below, the Disc defaults ON, same
    # reasoning as petal_enabled above.
    disc_enabled: BoolProperty(
        name="Enable", description="Generate the central disc", default=True,
        update=_flower_live_update)
    disc_radius: FloatProperty(
        name="Radius", description="Radius of the central disc; also where "
        "every petal's base attaches", default=0.006, min=0.0001, soft_max=0.5,
        subtype="DISTANCE", update=_flower_live_update)
    disc_radius_variance: FloatProperty(
        **_variance_kwargs("Radius", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    disc_height: FloatProperty(
        name="Height", description="Height of the central disc. Negative points "
        "the disc downward instead -- a sunken/concave bowl rather than a mound, "
        "since petals still attach at the rim (Z=0) either way",
        default=0.003, soft_min=-0.5, soft_max=0.5, subtype="DISTANCE",
        update=_flower_live_update)
    disc_height_variance: FloatProperty(
        **_variance_kwargs("Height", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    disc_resolution: IntProperty(
        name="Disc Resolution", description="Points along the disc's own profile "
        "(more = smoother dome/bowl curvature)", default=8, min=2, max=64,
        update=_flower_live_update)
    disc_sides: IntProperty(
        name="Sides", description="Points around the disc", default=16, min=3,
        max=256, update=_flower_live_update)

    # --- receptacle: a second revolved body (same mechanism as Disc), always
    # swept downward from the shared rim toward where a stem would attach --
    # see skeleton/flower.py's receptacle_* field docstring. Disabled by default.
    receptacle_enabled: BoolProperty(
        name="Enable", description="Generate a receptacle beneath the disc, "
        "tapering down toward where a stem would attach",
        default=False, update=_flower_live_update)
    receptacle_match_disc: BoolProperty(
        name="Match Disc Radius", description="Take the receptacle's top "
        "radius straight from Disc Radius, so it meets the rim without a step "
        "-- see the Corolla Tube's own Match Disc Radius tooltip",
        default=True, update=_flower_live_update)
    receptacle_radius: FloatProperty(
        name="Radius", description="Radius of the receptacle's top, at the shared rim",
        default=0.006, min=0.0001, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    receptacle_radius_variance: FloatProperty(
        **_variance_kwargs("Radius", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    receptacle_height: FloatProperty(
        name="Height", description="How far the receptacle tapers down below "
        "the rim -- always downward, unlike Disc Height",
        default=0.003, min=0.0, soft_max=0.5, subtype="DISTANCE", update=_flower_live_update)
    receptacle_height_variance: FloatProperty(
        **_variance_kwargs("Height", soft_max=0.5, subtype="DISTANCE",
                            update=_flower_live_update))
    receptacle_resolution: IntProperty(
        name="Receptacle Resolution", description="Points along the receptacle's "
        "own profile (more = smoother curvature)", default=6, min=2, max=64,
        update=_flower_live_update)
    receptacle_sides: IntProperty(
        name="Sides", description="Points around the receptacle (welded to the corolla "
        "tube, it takes the tube's)", default=16, min=3,
        max=256, update=_flower_live_update)

    # --- Flip Normals per component (see skeleton/flower.py) ---
    petal_flip_normals: BoolProperty(
        name="Flip Normals",
        description="Turn the petals' faces the other way round "
        "(the front becomes the back)",
        default=False, update=_flower_live_update)
    sepal_flip_normals: BoolProperty(
        name="Flip Normals",
        description="Turn the sepals' faces the other way round "
        "(the front becomes the back)",
        default=False, update=_flower_live_update)
    disc_flip_normals: BoolProperty(
        name="Flip Normals",
        description="Turn the disc's faces the other way round "
        "(the front becomes the back)",
        default=False, update=_flower_live_update)
    receptacle_flip_normals: BoolProperty(
        name="Flip Normals",
        description="Turn the receptacle's faces the other way round "
        "(the front becomes the back)",
        default=False, update=_flower_live_update)
    petal_tube_flip_normals: BoolProperty(
        name="Flip Normals",
        description="Turn the corolla tube's faces the other way round "
        "(the front becomes the back)",
        default=False, update=_flower_live_update)
    sepal_tube_flip_normals: BoolProperty(
        name="Flip Normals",
        description="Turn the calyx tube's faces the other way round "
        "(the front becomes the back)",
        default=False, update=_flower_live_update)

    seed: IntProperty(
        name="Seed", description="Random seed for variance", default=0, min=0,
        update=_flower_live_update)

    # The flower's material (blender/flower_material.py): one for every part,
    # on one flattened-flower unwrap; set inside the flower's node graph so it
    # travels with it into any plant.
    material: PointerProperty(
        name="Material", type=Material, update=_flower_live_update,
        description="The whole flower's material -- one texture laid out as the flower "
        "flattened (see the UV Checker)")
    uv_checker: BoolProperty(
        name="UV Checker", default=False, update=_flower_checker_update,
        description="Show the whole flower with a checker texture instead of its "
        "materials, to judge the UVs (switches Solid views to Texture colouring)")

    # "Bake Bloom Stages" settings (see blender.flower.bake_bloom_stages) -- NOT
    # live (no update callback): these only take effect the next time that
    # button is clicked, unlike every other field on this PropertyGroup.
    bloom_stage_count: IntProperty(
        name="Bloom Stages", description="How many fixed-Bloom copies \"Bake "
        "Bloom Stages\" creates, evenly spaced from fully closed to fully open",
        default=3, min=2, max=16)
    bloom_stage_variance: FloatProperty(
        name="Bloom Stages Variance", description="Random +/- jitter applied to "
        "each stage's evenly-spaced Bloom value (re-rolled by Seed above), so "
        "stages land close to but not exactly on an even split",
        default=0.0, min=0.0, max=0.5, subtype="FACTOR")

    # this flower's private curve node group (Petal Width/Curl/Gravity/Twist
    # Profile + Disc Profile -- see blender.flower.FLOWER_CURVE_SPECS)
    curve_group: PointerProperty(type=bpy.types.NodeTree)


def register():
    bpy.utils.register_class(PLANTGEN_PG_flower)
    bpy.types.Object.plantgen_flower = PointerProperty(type=PLANTGEN_PG_flower)


def unregister():
    del bpy.types.Object.plantgen_flower
    bpy.utils.unregister_class(PLANTGEN_PG_flower)
