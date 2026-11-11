"""FlowerGen's operators: add a flower, bake its bloom stages or a mesh, its
pool, curve-shape and corolla / petal-shape presets, save / load a flower
preset, and the popup that edits a field's variance. Moved from the parametric add-on's
`operators.py` with the shape tables they use."""

import math

import bpy
from bpy.props import FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..growth import flower_io as model_io
from ..growth.flower import DOME_CAP
from ..growth.flower_params import Ramp
from . import flower

# SpeedTree-style quick shapes for the Taper/Gravity curve widgets.
# (identifier, menu label, control points, smooth) — `smooth` requests
# 'AUTO_CLAMPED' handles (see plant.set_curve) instead of the default 'AUTO'.
# Rise/Droop stay within the upper half (0->0.5 / 1->0.5); Linear Growth/Decay are
# the full-range (0->1 / 1->0) straight versions.
CURVE_SHAPES = [
    ("max", "Max", [(0.0, 1.0), (1.0, 1.0)], False),
    ("min", "Min", [(0.0, 0.0), (1.0, 0.0)], False),
    ("half", "Half", [(0.0, 0.5), (1.0, 0.5)], False),
    ("rise", "Rise", [(0.0, 0.0), (1.0, 0.5)], False),
    ("droop", "Droop", [(0.0, 1.0), (1.0, 0.5)], False),
    ("linear_growth", "Linear Growth", [(0.0, 0.0), (1.0, 1.0)], False),
    ("linear_decay", "Linear Decay", [(0.0, 1.0), (1.0, 0.0)], False),
    ("exp_growth", "Exponential Growth",
     [(0.0, 0.0), (0.25, 0.032), (0.5, 0.119), (0.75, 0.356), (1.0, 1.0)], True),
    ("exp_decay", "Exponential Decay",
     [(0.0, 1.0), (0.25, 0.356), (0.5, 0.119), (0.75, 0.032), (1.0, 0.0)], True),
    ("round", "Round", [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], False),
    ("s_curve", "S-Curve", [(0.0, 0.0), (0.3, 0.1), (0.7, 0.9), (1.0, 1.0)], True),
    ("fade_in", "Fade In", [(0.0, 0.0), (0.3, 0.0), (0.7, 1.0), (1.0, 1.0)], True),
    ("fade_out", "Fade Out", [(0.0, 1.0), (0.3, 1.0), (0.7, 0.0), (1.0, 0.0)], True),
]


_CURVE_SHAPE_DATA = {key: (points, smooth) for key, _label, points, smooth in CURVE_SHAPES}


# Cap-edge shapes for the Petal/Sepal Top & Bottom Shape curves -- a curve over
# WIDTH position (0=one edge -> 1=other edge), NOT length, scaled by the Top/
# Bottom Shape scalar (a multiple of that ring's own width, so the same entry
# reads identically at any petal size). See skeleton/flower.py's
# `petal_top_shape_profile` docstring for the mechanism.
#
# This is where bilaterally-notched TIPS became reachable: `emarginate` below
# dips in the middle and peaks either side of it, producing the two-lobed
# heart/obcordate tip that FLOWER_PETAL_SHAPES' own width ramp (rotationally
# symmetric about the spine, see its comment) genuinely cannot make.
#
# `dome` is `skeleton.flower.DOME_CAP` -- a sampled true semicircle, defined
# there (the lowest bpy-free module) so this library, the curve widget's
# default, and `FlowerMeshParams`' own field default can't drift apart.
# `acute` gets its straightness from collinear intermediate points rather than
# a handle type: `plant._set_curve_points` only offers 'AUTO'/'AUTO_CLAMPED'.
# `emarginate` carries extra shoulder points either side of its notch for the
# same reason -- going straight from 0.0 at the edge to the lobe peak leaves
# the two lobes reading as spikes rather than the rounded pair a real
# obcordate tip has.
FLOWER_CAP_SHAPES = [
    ("flat", "Flat / Truncate", [(0.0, 0.0), (1.0, 0.0)], False),
    ("dome", "Dome / Rounded", DOME_CAP, True),
    ("parabola", "Parabola", [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], True),
    ("acute", "Acute / Pointed",
     [(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, 0.5), (1.0, 0.0)], False),
    ("emarginate", "Emarginate / Notched",
     [(0.0, 0.0), (0.12, 0.72), (0.28, 0.98), (0.5, 0.42), (0.72, 0.98), (0.88, 0.72),
      (1.0, 0.0)], True),
    ("square", "Square / Squared-off",
     [(0.0, 0.0), (0.12, 1.0), (0.88, 1.0), (1.0, 0.0)], False),
    ("crenate", "Crenate / Scalloped",
     [(0.0, 0.0), (0.17, 0.9), (0.33, 0.35), (0.5, 1.0), (0.67, 0.35), (0.83, 0.9),
      (1.0, 0.0)], True),
]


_FLOWER_CAP_SHAPE_DATA = {
    key: (points, smooth) for key, _label, points, smooth in FLOWER_CAP_SHAPES}


# Botanical petal-outline shapes -- the petal's SIDES (the Petal Width Profile
# curve, 0=disc edge -> 1=tip) paired with the TIP CAP that finishes them off
# (a FLOWER_CAP_SHAPES key + how far it pushes, in multiples of the tip ring's
# own width). CURVE_SHAPES' abstract Rise/Droop/S-Curve etc. aren't petal-like,
# so these get their own library instead (see FLOWER_SHAPE_LIBRARIES).
#
# The cap columns are what make Obovate/Spatulate/Oblong/Rounded Tip work at
# all: a width ramp alone can only end at 0.0 (a sharp point) or above it (a
# flat guillotine cut straight across the width) -- it has NO way to round a
# broad end off. Those four used to fake it by collapsing the width to a pinch
# or leaving the flat cut, which is why they read as wrong. They now keep real
# width at the tip and get their roundness from the cap instead.
#
# Shapes that genuinely end in a point (Ovate/Lanceolate/Pointed) keep a `flat`
# cap at amount 0: their tip ring has no width left for a cap to shape, and the
# width-relative scaling makes the push correctly vanish there anyway.
#
# Still out of reach here: shapes needing a genuinely ASYMMETRIC outline (one
# side different from the other), since the width ramp is one radial value per
# length position, rotationally symmetric about the petal's spine. Notched TIPS
# are no longer in that category -- see FLOWER_CAP_SHAPES' `emarginate` above.
FLOWER_PETAL_SHAPES = [
    ("ovate", "Ovate",
     [(0.0, 0.55), (0.25, 0.95), (0.4, 1.0), (0.75, 0.6), (1.0, 0.0)], True, "flat", 0.0),
    ("obovate", "Obovate",
     [(0.0, 0.18), (0.3, 0.5), (0.6, 0.85), (0.8, 1.0), (1.0, 0.8)], True, "dome", 0.45),
    ("lanceolate", "Lanceolate",
     [(0.0, 0.15), (0.3, 1.0), (0.7, 0.55), (1.0, 0.0)], True, "flat", 0.0),
    ("elliptic", "Elliptic", [(0.0, 0.25), (0.5, 1.0), (1.0, 0.2)], True, "dome", 0.15),
    # Spatulate is Obovate at half the width -- literally the same profile
    # halved, and the same dome cap. Despite the name's echo of "spatula" it is
    # NOT a long-clawed spoon; earlier attempts here modelled the claw and read
    # as the wrong plant entirely. Keep the two in step if Obovate is retuned.
    ("spatulate", "Spatulate",
     [(0.0, 0.09), (0.3, 0.25), (0.6, 0.425), (0.8, 0.5), (1.0, 0.4)], True, "dome", 0.45),
    ("oblong", "Oblong",
     [(0.0, 0.7), (0.15, 0.95), (0.85, 0.95), (1.0, 0.85)], True, "dome", 0.45),
    ("pointed", "Pointed", [(0.0, 0.5), (1.0, 0.0)], False, "flat", 0.0),
    ("rounded_tip", "Rounded Tip",
     [(0.0, 0.45), (0.25, 0.9), (0.85, 1.0), (1.0, 0.95)], True, "dome", 0.5),
    # The long-clawed form: a narrow near-parallel stalk for well past half the
    # length, then an abrupt blade. This is what Spatulate above was wrongly
    # modelling -- the shape is real and worth having, it's just called
    # unguiculate ("clawed"). `_corolla_caryophyllaceous` needs it: a pink /
    # carnation petal genuinely is clawed.
    ("unguiculate", "Unguiculate / Clawed",
     [(0.0, 0.10), (0.32, 0.13), (0.56, 0.18), (0.72, 0.6), (0.85, 0.95), (1.0, 1.0)],
     True, "dome", 0.5),
    ("obcordate", "Obcordate / Notched",
     [(0.0, 0.18), (0.3, 0.5), (0.6, 0.85), (0.8, 1.0), (1.0, 0.85)], True,
     "emarginate", 0.5),
]


# The width curve alone, for the Width popover's own "Shape Presets" submenu --
# a quick outline swap that deliberately leaves the cap alone. The full
# width+cap application is `PLANTGEN_OT_flower_petal_shape_preset` below. One
# table, two levels of application.
_FLOWER_PETAL_SHAPE_DATA = {
    key: (points, smooth) for key, _label, points, smooth, _cap, _amount in FLOWER_PETAL_SHAPES}


_FLOWER_PETAL_CAP_DATA = {
    key: (cap, amount) for key, _label, _points, _smooth, cap, amount in FLOWER_PETAL_SHAPES}


_FLOWER_PETAL_LABEL_BY_KEY = {row[0]: row[1] for row in FLOWER_PETAL_SHAPES}


# Disc-body shapes for the Disc Profile curve (0=rim/base, 1=top) -- radius
# multiplier vs. height along the disc's own revolve spine (see
# skeleton/flower.py's `_build_disc`). Petals always attach at the rim (t=0, full
# radius, Z=0), so every shape here keeps that end fixed and only varies how the
# radius closes toward the top -- Convex bulges past full radius (an overhang)
# before closing; Concave pinches narrower than a straight taper (a waisted/
# goblet silhouette) before closing, the two recognizable "rounds outward" vs.
# "pinches inward" readings of a disc's profile curve.
FLOWER_DISC_SHAPES = [
    ("hemisphere", "Hemisphere",
     [(0.0, 1.0), (0.3, 0.97), (0.6, 0.85), (0.85, 0.55), (1.0, 0.0)], True),
    ("flat", "Flat / Button", [(0.0, 1.0), (0.85, 0.98), (1.0, 0.0)], False),
    ("cone", "Cone", [(0.0, 1.0), (1.0, 0.0)], False),
    ("convex", "Convex / Bulging",
     [(0.0, 1.0), (0.2, 1.15), (0.5, 1.05), (0.8, 0.6), (1.0, 0.0)], True),
    ("concave", "Concave / Waisted",
     [(0.0, 1.0), (0.3, 0.5), (0.6, 0.35), (0.85, 0.15), (1.0, 0.0)], True),
]


_FLOWER_DISC_SHAPE_DATA = {key: (points, smooth)
                           for key, _label, points, smooth in FLOWER_DISC_SHAPES}


# Corolla/calyx TUBE shapes for the Petal/Sepal Tube Profile curves
# (0=rim, 1=where the free lobes start) -- radius multiplier vs. height along
# the shared tube's own revolve spine (see skeleton/flower.py's
# `_build_petal_group`/`petal_tube_enabled`). Every shape here is fixed at
# 1.0 at t=0 (same convention as FLOWER_DISC_SHAPES) so the tube always meets
# the disc rim without a visible seam by default -- only how the radius moves
# from there toward the mouth differs. A SEPARATE library from FLOWER_DISC_SHAPES:
# the disc's own vocabulary (hemisphere/flat/cone/convex/concave) has no
# equivalent for a funnel's steady widen, an urn's bulge-then-pinch, or a
# salverform's abrupt late flare, so reusing it would read as a mismatched
# dropdown rather than a genuine corolla-shape library.
#
# NOTE on scope: Papilionaceous (pea-flower) corollas -- an asymmetric banner/
# wings/keel, each a DIFFERENT shape at a different ring position -- are
# deliberately NOT reachable here or anywhere else in this system. Every whorl
# is one rotationally-symmetric ring of identical petals (see
# skeleton/flower.py's module docstring); giving different petal positions
# their own distinct shapes needs a genuine per-position generator, well
# beyond a shared Fusion/profile mechanism. Same tier as the existing
# Cordate/Obcordate caveat on FLOWER_PETAL_SHAPES above.
FLOWER_COROLLA_SHAPES = [
    # Tubular is NOT a perfect cylinder: a real one narrows slightly toward the
    # mouth so the wall eases into the lobes instead of meeting them at a hard
    # step. The taper lives in the last third; the first two thirds stay
    # essentially parallel, which is what still makes it read as "a tube".
    # Per-flower irregularity along the length is a separate control (Tube
    # Noise) rather than baked here -- points baked into a preset would give
    # every flower in a field the identical wobble, which is the opposite of
    # what randomness is for.
    ("tubular", "Tubular",
     [(0.0, 1.0), (0.35, 0.99), (0.62, 0.96), (0.82, 0.88), (1.0, 0.74)], True),
    ("infundibuliform", "Infundibuliform / Funnel",
     [(0.0, 1.0), (0.3, 1.05), (0.6, 1.5), (0.85, 2.3), (1.0, 3.0)], True),
    # Campanulate ("bell-shaped"): botanically, rounded at the base then
    # straight or gradually widening toward the mouth -- the curvature is
    # front-loaded, not back-loaded. An earlier version put the kick at the
    # very top (a straight-sided cone that flares hard at the mouth), which
    # reads as a plant pot with a lip rather than a bell. This one swells
    # quickly right off the rim (the rounded shoulder) and then eases into a
    # near-constant, gently-widening slope for the rest of the tube -- no
    # extra flare at the mouth. The distinction from Bell below is width, not
    # curvature: Bell shares this same rounded-base-then-straight silhouette,
    # just blown out much wider starting right at the base.
    ("campanulate", "Campanulate",
     [(0.0, 1.0), (0.12, 1.32), (0.28, 1.55), (0.5, 1.68), (0.75, 1.8), (1.0, 1.95)], True),
    # Bell: the same rounded-base-then-straight profile as Campanulate above,
    # but scaled up a lot wider, with the extra width concentrated at the
    # base rather than spread evenly -- a fuller, rounder belly right off the
    # rim, giving the hanging-bluebell look instead of Campanulate's narrower
    # shoulder.
    ("bell", "Bell",
     [(0.0, 1.0), (0.12, 1.7), (0.28, 2.1), (0.5, 2.3), (0.75, 2.45), (1.0, 2.6)], True),
    ("urceolate", "Urceolate / Urn",
     [(0.0, 1.0), (0.35, 1.4), (0.7, 1.35), (0.9, 0.9), (1.0, 0.75)], True),
    # Salverform's whole point is an abrupt right-angle flare at the very
    # mouth ("salver" = a flat tray on a stem). The old version spread that
    # over the last 15% of the height, which read as a gentle funnel; it now
    # holds a parallel shaft to 90% and does nearly all the flaring in the
    # last 2%.
    ("salverform", "Salverform",
     [(0.0, 1.0), (0.90, 1.0), (0.955, 1.3), (0.98, 2.2), (1.0, 3.3)], True),
]


_FLOWER_COROLLA_SHAPE_DATA = {
    key: (points, smooth) for key, _label, points, smooth in FLOWER_COROLLA_SHAPES}


# Which shape library each flower curve's "Shape Presets" dropdown draws from --
# keyed by the `library` column in `blender.flower.FLOWER_CURVE_SPECS`. Two forms:
# the `(points, smooth)` lookup `PLANTGEN_OT_flower_curve_shape.execute` applies,
# and a (key, label) list `ui_flower.py` draws menu buttons from (needs the
# label, which the lookup form drops). The menu form is deliberately just those
# two columns rather than each source table's own rows -- FLOWER_PETAL_SHAPES
# carries extra cap columns the menu has no use for, and unpacking a fixed
# arity there would break the moment any one library grows a column.
FLOWER_SHAPE_LIBRARIES = {
    "petal": _FLOWER_PETAL_SHAPE_DATA,
    "disc": _FLOWER_DISC_SHAPE_DATA,
    "corolla": _FLOWER_COROLLA_SHAPE_DATA,
    "cap": _FLOWER_CAP_SHAPE_DATA,
    "generic": _CURVE_SHAPE_DATA,
}


FLOWER_SHAPE_LIBRARIES_LIST = {
    "petal": [(row[0], row[1]) for row in FLOWER_PETAL_SHAPES],
    "disc": [(row[0], row[1]) for row in FLOWER_DISC_SHAPES],
    "corolla": [(row[0], row[1]) for row in FLOWER_COROLLA_SHAPES],
    "cap": [(row[0], row[1]) for row in FLOWER_CAP_SHAPES],
    "generic": [(row[0], row[1]) for row in CURVE_SHAPES],
}


class PLANTGEN_OT_flower_edit_variance(Operator):
    bl_idname = "plantgen.flower_edit_variance"
    bl_label = "Edit Variance"
    bl_description = "Set how much this parameter randomly deviates on each Seed re-roll"
    bl_options = {"REGISTER", "UNDO"}

    prop_name: StringProperty(options={"HIDDEN"})  # e.g. "base_radius"
    label: StringProperty(options={"HIDDEN"})
    # Which per-object PropertyGroup this targets -- "plantgen" (the default, a
    # stem) or "plantgen_flower" (a standalone FlowerGen flower, see
    # `blender/flower.py`). Two independent object kinds share this one operator
    # rather than each needing its own copy.
    root: StringProperty(options={"HIDDEN"}, default="plantgen_flower")
    # Set by ui._slider_with_variance to "" (prop_name lives directly on the root
    # PropertyGroup), "leaves", or "flowers" (prop_name lives one level deeper, on
    # that nested PLANTGEN_PG_phyllotaxis) -- an explicit string rather than a bool,
    # since Leaves and Flowers share one PropertyGroup class and can't be told apart
    # by `bl_rna.identifier`. Dot-separated for anything nested further (e.g.
    # "flowers.inflorescence.pedicel" for a Pedicel field) -- a single segment
    # behaves exactly as before.
    group: StringProperty(options={"HIDDEN"}, default="")
    # Absolute, same units as the target field itself (see properties._variance_kwargs)
    # -- e.g. 1.0 here means +/-1 meter for Length, not +/-100%. No hard max: some
    # fields (Curl, Twist) reasonably want up to 8.
    variance: FloatProperty(name="Variance", min=0.0, soft_max=8.0)

    def _target(self, context):
        s = getattr(context.active_object, self.root)
        for part in self.group.split(".") if self.group else ():
            s = getattr(s, part)
        return s

    def invoke(self, context, event):
        self.variance = getattr(self._target(context), f"{self.prop_name}_variance")
        return context.window_manager.invoke_props_dialog(self, width=250)

    def draw(self, context):
        self.layout.prop(self, "variance", text=self.label or self.prop_name)

    def execute(self, context):
        # setting the *_variance property fires its own update=_live_update/
        # _leaves_owner_update callback (see properties.py's _variance_kwargs), so
        # no explicit rebuild here -- matches every other slider in the panel.
        setattr(self._target(context), f"{self.prop_name}_variance", self.variance)
        return {"FINISHED"}


# Each Corolla Type preset (`FLOWER_COROLLA_PRESETS`) is the SAME
# `fields_fn(obj) -> dict` idiom as `FLOWER_PRESETS` above, just targeting a
# flower's own SHAPE (`obj.plantgen_flower`) instead of its placement on a
# stem -- one whole named botanical corolla form (Cruciform/Tubular/etc) in
# one click, covering both the free/Polypetalous forms (Petal Count + a named
# width-profile outline, Fusion left at 0) and the fused/Sympetalous forms
# (Fusion + its own tube-profile shape, reused from FLOWER_COROLLA_SHAPES,
# + Rotation Pitch for how far the free lobes flare open at the mouth).
#
# Unlike FLOWER_PRESETS, a returned dict here may hold `Ramp` VALUES (the
# Width/Fusion Profile curves), not just plain scalars -- those curves aren't
# `obj.plantgen_flower` bpy properties at all, they live in a separate node
# group (see `flower.set_flower_curve`/`get_flower_curves`), so
# `PLANTGEN_OT_flower_corolla_preset.execute` below dispatches on
# `isinstance(value, Ramp)` to route each item correctly. Ramp values ARE
# compared for "currently active preset" label-matching, same as
# `_matching_flower_position_label` does for plain scalars -- see
# `ui._fields_close`, which compares a Ramp by its authored control points.
# (This used to say the comparison wasn't worth it; showing the live shape's
# real name on the dropdown turned out to be worth it.)
#
# Polysepalous vs. Gamosepalous needs no preset list of its own: it's just
# Sepal Fusion > 0 on the existing slider (see ui_flower.py), directly
# tunable, mirroring these same Corolla shapes if wanted -- only the Corolla
# (petal) side gets a named one-click list, per the botanical taxonomy this
# was scoped from.
def _petal_shape_fields(shape_key: str) -> dict:
    """The full width+cap application of one FLOWER_PETAL_SHAPES entry -- the
    Sides outline AND the tip cap that finishes it (see that table's comment
    for why the cap is not optional for the broad-tipped shapes). Shared by
    `PLANTGEN_OT_flower_petal_shape_preset` and the free/Polypetalous corolla
    presets below, so a Rosaceous corolla can't silently ship a differently
    shaped petal than the Rounded Tip it names."""
    points, _smooth = _FLOWER_PETAL_SHAPE_DATA[shape_key]
    cap_key, cap_amount = _FLOWER_PETAL_CAP_DATA[shape_key]
    cap_points, _cap_smooth = _FLOWER_CAP_SHAPE_DATA[cap_key]
    return {
        "petal_width_profile": Ramp(points=list(points)),
        "petal_top_shape_profile": Ramp(points=list(cap_points)),
        "petal_top_shape": cap_amount,
    }


# `(obj) -> dict` wrappers, one per FLOWER_PETAL_SHAPES entry, purely so
# `_apply_flower_preset_fields`'s domain-reset can treat the Petal Shape
# family the same way as every other preset family (a plain list of
# `fields_fn(obj)` callables) even though `_petal_shape_fields` itself takes
# a shape key, not an object -- `k=k` binds each lambda to its own key
# instead of late-binding to whatever `k` is by the time the list is walked.
_PETAL_SHAPE_FIELDS_FNS = [
    (lambda obj, k=k: _petal_shape_fields(k)) for k in _FLOWER_PETAL_SHAPE_DATA
]


def _corolla_free_fields(petal_count: int, shape_key: str) -> dict:
    """Shared shape for the free/Polypetalous presets below -- Corolla Tube
    off, just Petal Count + a named outline reused from FLOWER_PETAL_SHAPES
    (rather than inventing new control points)."""
    return dict(
        petal_tube_enabled=False, petal_count=petal_count,
        **_petal_shape_fields(shape_key),
    )


# Absolute tube dimensions below are expressed relative to the DEFAULT Petal
# Length/Disc Radius (0.02m/0.006m respectively) -- since the Corolla Tube's
# own Length/Radius are now fully independent fields (see
# skeleton/flower.py's petal_tube_enabled docstring: an earlier design
# derived them from Petal Length x a 0..1 Fusion fraction and Disc Radius,
# which coupled the tube's size to the petal's own; the user asked for these
# to be genuinely separate controls), a preset has to pick SOME absolute
# starting values rather than a fraction -- these keep the same relative
# proportions the old Fusion-fraction presets had, so the shapes read the
# same on a flower still at its own default Petal Length/Disc Radius.
def _tube_mouth_pitch(obj, profile_points, tube_length: float) -> float:
    """The angle a lobe would sit at if it simply continued this tube's wall --
    the same quantity `skeleton.flower._build_petal_group` derives from the
    built tube, recomputed here so a preset can express itself as "put the
    lobes at THIS angle" and let the offset fall out.

    Mirrors that function deliberately, including sampling at the flower's own
    Tube Resolution and using its own radius: the angle depends on both (dr
    scales with radius, dz doesn't), so a preset applied to a big flower needs
    a different offset than the same preset on a small one to land the lobes in
    the same place."""
    s = obj.plantgen_flower if obj else None
    n = max(2, s.petal_tube_resolution if s else 8)
    radius = (s.disc_radius if (s and s.petal_tube_match_disc)
              else (s.petal_tube_radius if s else 0.006))
    prof = Ramp(points=list(profile_points))
    r_last = radius * prof.evaluate(1.0)
    r_prev = radius * prof.evaluate((n - 2) / (n - 1))
    dz = tube_length / (n - 1)
    dr = r_last - r_prev
    if abs(dr) < 1e-12 and abs(dz) < 1e-12:
        return math.pi / 2
    return math.atan2(dz, dr)


def _corolla_fused_fields(
    obj, tube_length: float, profile_key: str, lobe_angle: float,
    noise: float = 0.0, match_angle: bool = True,
) -> dict:
    """Shared shape for the fused/Sympetalous presets below -- an enabled
    Corolla Tube (its own Length + tube profile, reused from
    FLOWER_COROLLA_SHAPES) + how far the free lobes flare open past the tube's
    own mouth. Radius is left to Match Disc Radius (so the tube/disc join is
    seamless out of the box) -- only Length, the profile shape, the pitch
    offset and the irregularity vary per preset.

    `lobe_angle` is where the lobes should actually END UP, measured above
    horizontal -- what the preset is really trying to say. With Match Tube
    Angle on, the stored Pitch is an OFFSET from the tube's own mouth angle,
    so the offset is derived here rather than hand-written: mouth angles across
    these shapes range from about 10 degrees (Salverform's near-horizontal
    flare) to over 120 (an urn pinching back inward), so the same stored Pitch
    means something completely different from one preset to the next, and a
    hand-picked set of offsets silently goes stale the moment a profile is
    retuned. Deriving it keeps the authored intent readable and self-correcting.

    Pitch stays a live offset afterwards, which is the point: reshaping the top
    of the Tube Profile moves the mouth angle, and the lobes follow it instead
    of needing to be re-eyeballed."""
    profile_points = _FLOWER_COROLLA_SHAPE_DATA[profile_key][0]
    offset = (
        lobe_angle - _tube_mouth_pitch(obj, profile_points, tube_length)
        if match_angle else lobe_angle)
    return dict(
        petal_tube_enabled=True, petal_tube_length=tube_length,
        petal_tube_profile=Ramp(points=list(profile_points)),
        petal_tube_match_angle=match_angle,
        petal_tube_noise=noise,
        # Every preset here is a SYMPETALOUS form -- a corolla that is one
        # fused piece by definition -- so the lobes tile the mouth rather than
        # leaving gaps between separate straps stuck to a cylinder. Off would
        # be quietly wrong for the whole category; it's one checkbox to undo
        # for the rare case where separated lobes are wanted.
        petal_tube_fuse=True,
        petal_rotation_pitch=offset,
    )


def _corolla_cruciform(obj) -> dict:
    return _corolla_free_fields(4, "obovate")


def _corolla_caryophyllaceous(obj) -> dict:
    return _corolla_free_fields(5, "unguiculate")


def _corolla_rosaceous(obj) -> dict:
    return _corolla_free_fields(5, "rounded_tip")


# Where each fused corolla's free lobes should sit, in degrees above
# horizontal -- the readable half of `_corolla_fused_fields`' contract.
_DEG = math.pi / 180.0


def _corolla_tubular(obj) -> dict:
    # The one preset that ships with irregularity on: a real tubular corolla is
    # the shape where a dead-straight extrusion reads most obviously synthetic.
    return _corolla_fused_fields(obj, 0.014, "tubular", 35 * _DEG, noise=0.06)


def _corolla_infundibuliform(obj) -> dict:
    # A funnel's lobes are the flare carrying straight on, so this sits barely
    # off its own mouth angle.
    return _corolla_fused_fields(obj, 0.013, "infundibuliform", 22 * _DEG)


def _corolla_campanulate(obj) -> dict:
    # Tuned live against the reworked (rounded-base, straight-top) Campanulate
    # profile, same reasoning as Bell below: the old 28-degree target fought
    # the new profile's near-vertical mouth wall (~83 degrees) and curled the
    # lobes shut into a pinched, half-closed look. 65.6 degrees is the same
    # gentle curl-back-from-the-wall offset used for Bell -- appropriate here
    # too since the two profiles share the same mouth-wall angle by design,
    # just narrower overall.
    return _corolla_fused_fields(obj, 0.014, "campanulate", 65.6 * _DEG)


def _corolla_bell(obj) -> dict:
    # Tuned live against the reworked (wider, base-rounded) Bell profile: this
    # profile is 2.6x radius at the mouth (5.2x across) before the lobes even
    # start, and 65.6 degrees keeps the lobes curling back close to the tube's
    # own near-vertical mouth wall rather than flaring wide open, which is what
    # actually reads as a bell/bluebell silhouette on this shape.
    return _corolla_fused_fields(obj, 0.014, "bell", 65.6 * _DEG)


def _corolla_urceolate(obj) -> dict:
    # An urn PINCHES inward at the mouth, so continuing its wall points up and
    # back over the axis -- these lobes travel further from their own mouth
    # angle than any other preset's just to get back out past vertical.
    return _corolla_fused_fields(obj, 0.017, "urceolate", 45 * _DEG)


def _corolla_salverform(obj) -> dict:
    # The mouth already flares almost to horizontal, so the lobes barely need
    # to turn any further to lie flat -- that abrupt tray is the whole shape.
    return _corolla_fused_fields(obj, 0.017, "salverform", 2 * _DEG)


def _corolla_rotate(obj) -> dict:
    # The one fused preset with Match Tube Angle deliberately OFF: its tube is
    # a 2.4mm stub, so the "mouth angle" there is dominated by the numerical
    # difference between two nearly-identical radii rather than by any shape
    # the user can see or steer. Following it would make a flat wheel's lobes
    # jump around for no visible reason, so this one pins Pitch absolutely
    # instead. Reuses "tubular" rather than a dedicated profile (and drops its
    # irregularity, which would only read as jitter on an invisible stub).
    return _corolla_fused_fields(obj, 0.0024, "tubular", 0.0, match_angle=False)


FLOWER_COROLLA_PRESETS = [
    ("cruciform", "Cruciform",
     "Four free petals arranged in a cross (mustard-family look)",
     _corolla_cruciform),
    ("caryophyllaceous", "Caryophyllaceous",
     "Five free petals with a long narrow claw widening into the blade "
     "(pink-family look)",
     _corolla_caryophyllaceous),
    ("rosaceous", "Rosaceous",
     "Five free, broad rounded overlapping petals (rose-family look)",
     _corolla_rosaceous),
    ("tubular", "Tubular",
     "A long, mostly cylindrical fused corolla tube with short free lobes at the mouth",
     _corolla_tubular),
    ("infundibuliform", "Infundibuliform / Funnel",
     "A corolla tube widening gradually into a funnel before the free lobes",
     _corolla_infundibuliform),
    ("campanulate", "Campanulate",
     "A corolla tube rounded at the base, easing into straight, gently "
     "widening sides toward the mouth (classic bellflower look)",
     _corolla_campanulate),
    ("bell", "Bell",
     "Campanulate's rounded-base-then-straight profile, blown out a lot wider "
     "right from the base (hanging bluebell look)",
     _corolla_bell),
    ("urceolate", "Urceolate / Urn",
     "A corolla tube that bulges outward then pinches narrow again at the mouth",
     _corolla_urceolate),
    ("salverform", "Salverform",
     "A long narrow corolla tube that flares abruptly into a flat rim at the very mouth",
     _corolla_salverform),
    ("rotate", "Rotate / Wheel",
     "A very short fused base with wide, nearly flat free lobes (a wheel/star shape)",
     _corolla_rotate),
]


_FLOWER_COROLLA_PRESET_BY_KEY = {
    key: (label, description, fields_fn)
    for key, label, description, fields_fn in FLOWER_COROLLA_PRESETS
}


_FLOWER_CURVE_DEFAULTS = {spec[0]: spec[4] for spec in flower.FLOWER_CURVE_SPECS}


def _apply_flower_preset_fields(obj, fields: dict, all_fields_fns) -> None:
    """Write one preset's {field name -> value} onto a flower, routing curve
    values into its own curve node group and everything else onto the
    PropertyGroup -- dispatching on `isinstance(value, Ramp)`, since the two
    live in completely different places (see `flower.set_flower_curve`). One
    `suppress_rebuild` around the whole batch, then a single rebuild.

    Before applying, resets every field ANY preset in `all_fields_fns` (the
    caller's own family -- Corolla presets and Petal Shape presets are
    deliberately DIFFERENT families with different domains, so a Petal Shape
    preset can't reset Corolla-only fields like `petal_tube_enabled`) can
    set -- same `_reset_preset_domain` idea as the Position/Inflorescence
    presets, just split by target: `property_unset()` for plain
    `obj.plantgen_flower` fields, and `set_flower_curve` back to
    `FLOWER_CURVE_SPECS`' own authored default for Ramp-valued ones, since
    those live in a curve node group rather than as a real RNA property that
    `property_unset()` can reach."""
    with flower.suppress_rebuild(obj):
        domain = set()
        for fields_fn in all_fields_fns:
            domain.update(fields_fn(obj).keys())
        for name in domain:
            defaults = _FLOWER_CURVE_DEFAULTS.get(name)
            if defaults is not None:
                flower.set_flower_curve(obj, name, Ramp(points=list(defaults)))
            else:
                obj.plantgen_flower.property_unset(name)
        for name, value in fields.items():
            if isinstance(value, Ramp):
                # Ramp values here are real (post-remap) units, same as
                # apply_flower_params_to_object -- invert any y_remap
                # before writing into the (always 0..1) widget.
                y_remap = flower._FLOWER_CURVE_REMAPS.get(name)
                if y_remap is not None:
                    lo, hi = y_remap
                    value = Ramp(points=[
                        (x, (y - lo) / (hi - lo)) for x, y in value.points])
                flower.set_flower_curve(obj, name, value)
            else:
                setattr(obj.plantgen_flower, name, value)
    flower.rebuild_flower(obj)


class PLANTGEN_OT_flower_corolla_preset(Operator):
    bl_idname = "plantgen.flower_corolla_preset"
    bl_label = "Corolla Type Preset"
    bl_description = "Set Petals' Count/Fusion/Width or Fusion Profile to a named corolla type"
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        entry = _FLOWER_COROLLA_PRESET_BY_KEY.get(properties.preset)
        return entry[1] if entry else cls.bl_description

    def execute(self, context):
        obj = context.active_object
        entry = _FLOWER_COROLLA_PRESET_BY_KEY.get(self.preset)
        if not flower.is_flower(obj) or entry is None:
            return {"CANCELLED"}
        _label, _description, fields_fn = entry
        _apply_flower_preset_fields(
            obj, fields_fn(obj),
            [fn for _k, _l, _d, fn in FLOWER_COROLLA_PRESETS])
        return {"FINISHED"}


class PLANTGEN_OT_flower_petal_shape_preset(Operator):
    bl_idname = "plantgen.flower_petal_shape_preset"
    bl_label = "Petal Shape Preset"
    bl_description = (
        "Set the petal outline to a named botanical shape -- both its Sides "
        "(Width Profile) and the Top Shape cap that finishes the tip")
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        label = _FLOWER_PETAL_LABEL_BY_KEY.get(properties.preset)
        return f"Shape the petals as {label}" if label else cls.bl_description

    def execute(self, context):
        obj = context.active_object
        if not flower.is_flower(obj) or self.preset not in _FLOWER_PETAL_SHAPE_DATA:
            return {"CANCELLED"}
        _apply_flower_preset_fields(
            obj, _petal_shape_fields(self.preset), _PETAL_SHAPE_FIELDS_FNS)
        return {"FINISHED"}


class PLANTGEN_OT_add_flower(Operator):
    bl_idname = "plantgen.add_flower"
    bl_label = "Add Flower"
    bl_description = (
        "Create a new standalone flower in its own Collection. Select any flower "
        "later to edit it"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = flower.create_flower(context)
        self.report({"INFO"}, f"Added {obj.name}")
        return {"FINISHED"}


class PLANTGEN_OT_flower_bake_bloom_stages(Operator):
    bl_idname = "plantgen.flower_bake_bloom_stages"
    bl_label = "Bake Bloom Stages"
    bl_description = (
        "Create several fixed-Bloom copies of the active/selected flower(s) "
        "(e.g. bud/half-open/full-bloom, per each one's own Bloom Stages/"
        "Variance fields), named bud to open, into the shared pool collection: "
        "point a flower component's Collection at them and each flower shows "
        "the stage its own openness calls for. Runs on every selected flower, "
        "not just the active one"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        flowers = [o for o in context.selected_objects if flower.is_flower(o)]
        if not flowers:
            self.report({"ERROR"}, "Select at least one flower first")
            return {"CANCELLED"}
        total = 0
        for obj in flowers:
            s = obj.plantgen_flower
            stops = flower.bake_bloom_stages(
                obj, context, stages=s.bloom_stage_count, variance=s.bloom_stage_variance)
            total += len(stops)
        self.report(
            {"INFO"},
            f"Added {total} bloom stages to {flower.FLOWER_POOL_NAME} "
            f"from {len(flowers)} flower(s)")
        return {"FINISHED"}


class PLANTGEN_OT_flower_bake_to_mesh(Operator):
    bl_idname = "plantgen.flower_bake_to_mesh"
    bl_label = "Bake To Mesh"
    bl_description = (
        "Freeze the active/selected flower(s) into a single static Mesh "
        "object each, with no modifiers, properties, or Sepals/Disc/"
        "Receptacle children -- in the shared baked-assets collection, for "
        "when you want a lightweight, no-longer-editable asset. Runs on every "
        "selected flower, not just the active one; the flowers themselves are "
        "left untouched"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        flowers = [o for o in context.selected_objects if flower.is_flower(o)]
        if not flowers:
            self.report({"ERROR"}, "Select at least one flower first")
            return {"CANCELLED"}
        baked = [flower.bake_to_mesh(obj, context) for obj in flowers]
        self.report({"INFO"}, f"Baked {len(baked)} flower(s) to {flower.FLOWER_BAKED_NAME}")
        return {"FINISHED"}


class PLANTGEN_OT_flower_curve_shape(Operator):
    bl_idname = "plantgen.flower_curve_shape"
    bl_label = "Curve Shape"
    bl_description = "Replace this curve's control points with a common shape"
    bl_options = {"REGISTER", "UNDO"}

    curve_key: StringProperty(options={"HIDDEN"})  # a flower.FLOWER_CURVE_SPECS key
    # Which FLOWER_SHAPE_LIBRARIES entry `shape` is looked up in -- set from
    # flower.FLOWER_CURVE_SPECS' own `library` column, so each curve only ever
    # offers shapes that make sense for it (petal outlines, disc bodies, or the
    # generic abstract curves for the shape-only Curl/Gravity/Twist profiles).
    library: StringProperty(options={"HIDDEN"}, default="generic")
    shape: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        obj = context.active_object
        data = FLOWER_SHAPE_LIBRARIES.get(self.library, {}).get(self.shape)
        if not flower.is_flower(obj) or data is None:
            return {"CANCELLED"}
        points, smooth = data
        handle_type = 'AUTO_CLAMPED' if smooth else 'AUTO'
        flower.set_flower_curve(obj, self.curve_key, Ramp(points=list(points)), handle_type)
        flower.rebuild_flower(obj)
        return {"FINISHED"}


class PLANTGEN_OT_flower_add_to_pool(Operator):
    bl_idname = "plantgen.flower_add_to_pool"
    bl_label = "Add To Pool"
    bl_description = (
        f"Toggle whether this flower's main object is in the shared "
        f"{flower.FLOWER_POOL_NAME} Collection (created on first use), where "
        "baked bloom stages also land -- a collection a flower component can "
        "draw on. Depressed when already in the pool; click again to remove it"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if not flower.is_flower(obj):
            self.report({"ERROR"}, "Select a flower first")
            return {"CANCELLED"}
        if flower.is_in_pool(obj):
            flower.remove_from_pool(obj)
            self.report({"INFO"}, f"Removed {obj.name} from {flower.FLOWER_POOL_NAME}")
        else:
            flower.add_to_pool(obj)
            self.report({"INFO"}, f"Added {obj.name} to {flower.FLOWER_POOL_NAME}")
        return {"FINISHED"}


class PLANTGEN_OT_save_flower_preset(Operator, ExportHelper):
    bl_idname = "plantgen.save_flower_preset"
    bl_label = "Save Flower Preset"
    bl_description = "Save the active flower's settings as a .json preset"
    bl_options = {"REGISTER"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        obj = context.active_object
        if not flower.is_flower(obj):
            self.report({"ERROR"}, "Select a flower first")
            return {"CANCELLED"}
        params = flower.flower_params_for_preset(obj)
        model_io.save_flower_json(self.filepath, obj.name, params)
        self.report({"INFO"}, f"Saved {self.filepath}")
        return {"FINISHED"}


class PLANTGEN_OT_load_flower_preset(Operator, ImportHelper):
    bl_idname = "plantgen.load_flower_preset"
    bl_label = "Load Flower Preset"
    bl_description = "Load a .json preset onto the active flower"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        obj = context.active_object
        if not flower.is_flower(obj):
            self.report({"ERROR"}, "Select a flower first")
            return {"CANCELLED"}
        try:
            params = model_io.load_flower_json(self.filepath)
        except model_io.FlowerFileError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        flower.apply_flower_params_to_object(obj, params)
        self.report({"INFO"}, f"Loaded {self.filepath}")
        return {"FINISHED"}


_classes = (
    PLANTGEN_OT_flower_edit_variance, PLANTGEN_OT_add_flower,
    PLANTGEN_OT_flower_curve_shape, PLANTGEN_OT_flower_corolla_preset,
    PLANTGEN_OT_flower_petal_shape_preset, PLANTGEN_OT_flower_add_to_pool,
    PLANTGEN_OT_flower_bake_bloom_stages, PLANTGEN_OT_flower_bake_to_mesh,
    PLANTGEN_OT_save_flower_preset, PLANTGEN_OT_load_flower_preset,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
