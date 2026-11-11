"""Per-object curve widgets: the bpy half of `growth/curves.py`.

Blender has no property type for a curve, so a `CurveMapping` has to be owned
by something that does. The parametric add-on solved this by giving each object its own
shader node group holding one `ShaderNodeFloatCurve` per curve, tagged with a
key -- `node.mapping` is then a real CurveMapping that
`layout.template_curve_mapping` draws and `mapping.evaluate()` reads. That
pattern is reused here rather than re-derived; it is the only approach that
survives a file save and reload.

There are two kinds here and the difference matters. A **shape** curve is a
multiplier on the panel value beside it, read at a node's own position along
its axis -- a flat curve at 1.0 is exactly the behaviour before curves existed,
which is what let them be added without changing any species' output. A
**development** curve is read at the plant's current age and *is* the value;
there is no panel number beside it, because the shape is the whole statement.
`CurveSpec.x_label` is what tells the two apart in the panel.

One trap worth stating because it costs an hour when it bites, and the parametric add-on
paid for it already: a fresh curve mapping is `use_clip=True` with
`clip_max_y = 1.0`. A multiplier capped at 1.0 can only ever make things
*smaller* -- author a point at 1.4 and it silently clamps back. Every curve
here raises the ceiling at creation.
"""

from __future__ import annotations

from typing import NamedTuple

import bpy

from ..growth.curves import (
    DOME_POINTS,
    ELONGATION_MODES,
    STEM_THICKENING_POINTS,
    Curve,
    default_leaf_expansion,
    default_stem_thickening,
    elongation_mode,
)

# The procedurally drawn glyphs (`icons.py`, shared with FlowerGen): a live
# sparkline of each curve's own shape for its popover button.
from . import icons as pg_icons

ICON_PREFIX = "pgs_"
"""Icon-cache keys are shared with FlowerGen's curves in one preview
collection; the prefix keeps the plant's sparklines apart from a flower's."""

_GROUP_NAME = "PlantGen · Curves"
_KEY = "pgs_key"

_FLAT = ((0.0, 1.0), (1.0, 1.0))

_DEFAULT_POINTS = {"young_growth": ((0.0, 0.0), (1.0, 1.0)),
                   "elongation": ELONGATION_MODES["OVERLAPPING"][0],
                   "leaf_expansion": default_leaf_expansion().points,
                   "stem_thickening": STEM_THICKENING_POINTS}
"""The shape a curve is created at when there is no profile to seed it from,
for the few that are not multipliers: flat 1.0 is "no change" for a
multiplier, but for Young Growth it would mean "already full grown" and switch
the young-inflorescence sizing off. Its no-profile default is the profile's."""


_DEFAULT_HANDLES = {"elongation": 'VECTOR', "leaf_expansion": 'VECTOR',
                    "stem_thickening": 'VECTOR'}
"""Handles a curve is created with when there is no profile: the development
curves are piecewise-linear in the model, so their widgets are too."""

_EXACT = {
    "elongation": lambda: [elongation_mode(m)[0] for m in ELONGATION_MODES],
    "leaf_expansion": lambda: [default_leaf_expansion()],
    "stem_thickening": lambda: [default_stem_thickening()],
}
"""Curves whose widget, while it still holds one of these exact shapes, hands
growth that exact curve rather than its re-sampled self -- so a plant nobody
has drawn on grows exactly as before the widget existed."""


def _exact_curve(key: str, node) -> Curve | None:
    """The exact model curve this widget holds untouched, else None."""
    make = _EXACT.get(key)
    if make is None:
        return None
    pts = node.mapping.curves[0].points
    if any(p.handle_type != 'VECTOR' for p in pts):
        return None
    got = [(p.location[0], p.location[1]) for p in pts]
    for curve in make():
        if len(curve.points) == len(got) and all(
                abs(x - a) < 1e-5 and abs(y - b) < 1e-5
                for (x, y), (a, b) in zip(curve.points, got, strict=True)):
            return curve
    return None


def elongation_mode_of(node, steps: int) -> str:
    """The Internode Elongation mode a widget + duration match, else CUSTOM. No
    widget yet (a plant not regrown since it existed) is the default curve."""
    exact = (_exact_curve("elongation", node) if node is not None
             else Curve.from_points(_DEFAULT_POINTS["elongation"]))
    if exact is not None:
        for mode, (points, mode_steps) in ELONGATION_MODES.items():
            if exact.points == tuple(points) and mode_steps == steps:
                return mode
    return 'CUSTOM'


def _default_points(key: str):
    """What a curve node starts as with nothing to seed it: the curve's own
    default, else flat."""
    return _DEFAULT_POINTS.get(key, _FLAT)


class CurveSpec(NamedTuple):
    key: str
    label: str
    x_label: str
    y_label: str
    attr: str


ALONG_AXIS = "0 (base) → 1 (tip)"
OVER_AGE = "0 (seedling) → 1 (mature)"
"""The two things a curve's X axis can mean here, and they are not
interchangeable. A shape curve is read at a node's own position along its axis;
a development curve is read at the plant's current age. Showing the wrong
caption on a development curve would invite exactly the wrong edit -- a
branching curve is not "more branches at the top", it is "branching starts
later"."""

ALONG_BRANCH = "0 (stem) -> 1 (branch tip)"
"""A sixth, and the second of its kind: read along a single inflorescence
branch rather than across the inflorescence. Captioned apart for exactly the
reason `ALONG_PEDICEL` is -- Branch Width read as "bottom branch to top branch"
is what it used to do and is what was reported as wrong."""

ALONG_PEDICEL = "0 (stem) -> 1 (flower)"
"""A fourth, for the one flowering curve read along a single pedicel rather
than across the inflorescence. Captioned separately for the same reason the
other three are: read as "base of the raceme to its tip" it would invite
exactly the wrong edit."""

ACROSS_HEAD = "0 (centre) -> 1 (rim)"
"""A fifth, for the flat-topped head's own cross-section. Read at a flower's
distance from the middle of the head, not at its position along the axis --
which on a corymb runs the *opposite* way, since its lowest flowers are the
outermost ones. Captioned apart so the graph is read as the silhouette it is."""

DEVELOPING = "0 (just laid down) -> 1 (mature)"
"""An inflorescence's own development -- how far its nodes have elongated --
not the plant's age and not a position along anything. Only the youngest
racemes sit left of 1."""

AXIS_GROWTH = "0 (just started) -> 1 (grown)"
"""A stem's or branch's own growth so far, over what it will build."""

AFTER_FORMING = "0 (just formed) -> 1 (Duration later)"
"""An internode's or leaf's own time since it formed, over its Duration."""

FLOWERING = "0 (first flower) -> 1 (last)"
"""A third thing an X axis can mean, and it needed its own caption: these are
read over the flowering region, so 1 is the tip of the inflorescence rather
than the tip of the stem or the end of the plant's life."""

CURVE_SPECS = (
    # Shape along an axis. Each is a multiplier on the panel value beside it.
    CurveSpec("leaf_size", "Leaf Size", ALONG_AXIS, "size multiplier", "leaf_size_profile"),
    CurveSpec("leaf_angle", "Leaf Angle", ALONG_AXIS, "angle multiplier", "leaf_angle_profile"),
    CurveSpec("stiffness", "Stiffness", ALONG_AXIS,
              "1 = full stiffness, low = soft", "stiffness_profile"),
    # Stem radius, internode length, branch angle and branch length along an
    # axis are the branching stack's per-level scales now, not curves.

    # Development over the plant's life. These are not multipliers on anything
    # -- the curve *is* the value, read at the current age -- and they are the
    # mechanism the rebuild plan names as the point of the whole thing: one Age
    # dial cannot produce a species-specific developmental trajectory unless
    # each species says how it interprets that dial.
    CurveSpec("flowering_onset", "Flowering Onset", OVER_AGE, "readiness to flower",
              "flowering_onset"),
    # WS3 senescence death-front: as Age rises, this share of the plant's
    # primary limbs (oldest/lowest first) has died. Flat 0 -- the default --
    # keeps every plant byte-identical.
    CurveSpec("senescence", "Age Senescence", OVER_AGE, "share of limbs dead",
              "senescence_onset"),
    # Each internode's / leaf's own development after it forms (the panel's
    # Internode Elongation modes stamp the first; `ELONGATION_MODES`).
    CurveSpec("elongation", "Internode Elongation", AFTER_FORMING, "share of full length",
              "elongation"),
    CurveSpec("leaf_expansion", "Leaf Expansion", AFTER_FORMING, "share of full size",
              "leaf_expansion"),
    # A stem / branch's width while it grows: its growth so far -> share of its
    # stated widths (`development._thicken`).
    CurveSpec("stem_thickening", "Stem Thickening", AXIS_GROWTH, "share of stated width",
              "stem_thickening"),

    # Gradients along the flowering region. Together these are what separates
    # two racemes that share a topology and nothing else.
    CurveSpec("flower_size", "Flower Size", FLOWERING, "size multiplier",
              "inflorescence.flower_size_profile"),
    CurveSpec("pedicel_length", "Pedicel Length", FLOWERING, "length multiplier",
              "inflorescence.pedicel_length_profile"),
    CurveSpec("pedicel_angle", "Pedicel Angle", FLOWERING, "tilt multiplier",
              "inflorescence.pedicel_angle_profile"),
    CurveSpec("flower_spacing", "Flower Spacing", FLOWERING, "spacing multiplier",
              "inflorescence.flower_spacing_profile"),
    CurveSpec("flower_pitch", "Flower Pitch", FLOWERING, "tilt multiplier",
              "inflorescence.flower_pitch_profile"),
    CurveSpec("pedicel_bend", "Pedicel Bend", FLOWERING, "curvature multiplier",
              "inflorescence.pedicel_bend_profile"),
    CurveSpec("pedicel_width", "Pedicel Width", ALONG_PEDICEL, "thickness multiplier",
              "inflorescence.pedicel_width_profile"),
    # Not a multiplier: the share of the way from Young Size to full size, read
    # at the inflorescence's own development. Its default is a rising line, not
    # flat -- see `_DEFAULT_POINTS`.
    CurveSpec("young_growth", "Young Growth", DEVELOPING,
              "share of the way to full size", "inflorescence.young_growth_profile"),
    # The other odd X out, and unlike its neighbours this one is not a
    # multiplier on the value beside it -- the curve *is* the head's shape, and
    # Dome only says how deep. See `growth.curves.default_head_dome_profile`.
    CurveSpec("head_dome", "Head Dome", ACROSS_HEAD, "drop, share of Dome",
              "inflorescence.head_dome_profile"),
    # The panicle's pyramid. Shares the FLOWERING X with its neighbours -- it is
    # read at a branch's own position along the rachis, which is the same
    # coordinate a flower's position is read at -- but it scales a whole
    # branch's length and division rather than one organ.
    # The panicle's own gradients, all read at a branch's position along the
    # rachis -- the same coordinate the flower gradients above use, so a
    # panicle's two levels stay comparable.
    #
    # **`infl_` prefixed, and that is not decoration.** The plant's own
    # vegetative `branch_angle` and `branch_length` curves were in this
    # registry when these were added (the branching stack has since replaced
    # them), and an inflorescence branch is a different thing from a woody
    # one -- the same distinction `Axis.structural_depth` exists for. The
    # unprefixed names collided: `branch_length` was defined twice, so the
    # panel's own Length graph silently opened and edited the *plant's*
    # vegetative branch-length curve. `CURVE_KEYS` is a tuple, so nothing
    # raised; it showed up only as this object reporting 23 curves of 24 with
    # nothing missing.
    CurveSpec("infl_branch_angle", "Branch Angle", FLOWERING, "angle multiplier",
              "inflorescence.branch_angle_profile"),
    CurveSpec("infl_branch_size", "Branch Size", FLOWERING, "size multiplier",
              "inflorescence.branch_size_profile"),
    CurveSpec("infl_branch_spacing", "Branch Spacing", FLOWERING, "spacing multiplier",
              "inflorescence.branch_spacing_profile"),
    CurveSpec("infl_branch_length", "Branch Length", FLOWERING, "length multiplier",
              "inflorescence.branch_length_profile"),
    CurveSpec("infl_branch_width", "Branch Width", ALONG_BRANCH, "thickness multiplier",
              "inflorescence.branch_width_profile"),
    CurveSpec("infl_branch_bend", "Branch Bend", FLOWERING, "arch multiplier",
              "inflorescence.branch_bend_profile"),
    CurveSpec("infl_branch_droop", "Branch Droop", FLOWERING, "sag multiplier",
              "inflorescence.branch_droop_profile"),
)

_CLIP_MAX_Y = {"infl_branch_size": 2.0, "infl_branch_spacing": 2.0, "infl_branch_length": 2.0}
"""Ceiling per curve, 1.0 for anything not listed -- the parametric add-on's own rule
(widen the one curve that needs it, not the whole set), because a 0..1 widget
reads far better than one where every preset sits in its lower third. The
panicle's size / spacing / length gradients are the exceptions, as multipliers
that may need to exceed 1. The development curves genuinely are 0..1 shares
and want the default ceiling."""

CURVE_KEYS = tuple(spec.key for spec in CURVE_SPECS)
if len(set(CURVE_KEYS)) != len(CURVE_KEYS):  # pragma: no cover - a registry typo
    _dupes = sorted({k for k in CURVE_KEYS if CURVE_KEYS.count(k) > 1})
    raise RuntimeError(
        f"duplicate CurveSpec key(s): {_dupes}. Two specs sharing a key silently "
        "collapse -- both rows open the same widget and one profile never reaches "
        "the plant. This fired for real when the panicle's gradients were added "
        "next to the plant's own vegetative branch curves.")
_ATTR_BY_KEY = {spec.key: spec.attr for spec in CURVE_SPECS}


def _get_attr(profile, path: str):
    """Read a possibly-dotted attribute path off a profile.

    The flowering gradients live on `profile.inflorescence`, not on the profile
    itself -- which is right, since they belong to the inflorescence and vary
    per kind and per order with it.
    """
    target = profile
    for part in path.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def _set_attr(profile, path: str, value) -> None:
    parts = path.split(".")
    target = profile
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
LABEL_BY_KEY = {spec.key: spec.label for spec in CURVE_SPECS}

SHAPES = (
    # The parametric add-on's `operators.CURVE_SHAPES`, entry for entry: same keys, same
    # labels, same points, same smooth flag. Copied rather than imported so a
    # missing the parametric add-on costs the icon glyphs and nothing structural -- but it
    # is a copy, so the two want changing together.
    ("max", "Max", ((0.0, 1.0), (1.0, 1.0)), False),
    ("min", "Min", ((0.0, 0.0), (1.0, 0.0)), False),
    ("half", "Half", ((0.0, 0.5), (1.0, 0.5)), False),
    ("rise", "Rise", ((0.0, 0.0), (1.0, 0.5)), False),
    ("droop", "Droop", ((0.0, 1.0), (1.0, 0.5)), False),
    ("linear_growth", "Linear Growth", ((0.0, 0.0), (1.0, 1.0)), False),
    ("linear_decay", "Linear Decay", ((0.0, 1.0), (1.0, 0.0)), False),
    ("exp_growth", "Exponential Growth",
     ((0.0, 0.0), (0.25, 0.032), (0.5, 0.119), (0.75, 0.356), (1.0, 1.0)), True),
    ("exp_decay", "Exponential Decay",
     ((0.0, 1.0), (0.25, 0.356), (0.5, 0.119), (0.75, 0.032), (1.0, 0.0)), True),
    ("round", "Round", ((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)), False),
    ("s_curve", "S-Curve", ((0.0, 0.0), (0.3, 0.1), (0.7, 0.9), (1.0, 1.0)), True),
    ("fade_in", "Fade In", ((0.0, 0.0), (0.3, 0.0), (0.7, 1.0), (1.0, 1.0)), True),
    ("fade_out", "Fade Out", ((0.0, 1.0), (0.3, 1.0), (0.7, 0.0), (1.0, 0.0)), True),
    # Beyond the parametric add-on's set, so SHAPES is no longer a strict 1:1 copy.
    # Distinct from Fade In, which starts at exactly zero (stacking every
    # rosette leaf on one point) and transitions gently: bolting is abrupt,
    # and a rosette's internodes are short rather than absent.
    ("rosette", "Rosette", ((0.0, 0.06), (0.28, 0.08), (0.42, 1.0), (1.0, 1.0)), True),
    # A parabola, and the default Head Dome cross-section. Distinct from
    # Exponential Growth, which is near enough a sphere cap: that is flat across
    # the middle and turns hard at the rim, where this curves gently throughout.
    # Points shared with `growth.curves.DOME_POINTS` so the widget names it.
    ("dome", "Dome", DOME_POINTS, True),
)
_SHAPE_DATA = {key: (points, smooth) for key, _label, points, smooth in SHAPES}


def ensure_curves(obj, profile=None) -> dict:
    """Every curve node for this object, creating the group and any missing
    node on the way. Safe to call repeatedly.

    A node that is created here is seeded from `profile`, if one is given, and
    a node that already exists is never touched. That distinction is what makes
    adding a curve to `CURVE_SPECS` safe: the object gains the new one at its
    species' shape and keeps every curve its owner already authored. The first
    version re-seeded all of them whenever any were missing, which would have
    silently reset five hand-drawn curves the moment the development curves
    were added.
    """
    s = obj.plantgen
    group = s.curve_group
    if group is None:
        group = bpy.data.node_groups.new(_GROUP_NAME, 'ShaderNodeTree')
        group.use_fake_user = True
        s.curve_group = group

    nodes = {}
    for key in CURVE_KEYS:
        node = next((n for n in group.nodes if n.get(_KEY) == key), None)
        if node is None:
            node = group.nodes.new('ShaderNodeFloatCurve')
            node[_KEY] = key
            node.label = key
            seed = _get_attr(profile, _ATTR_BY_KEY[key]) if profile is not None else None
            set_points(node, (seed.points or _FLAT) if seed is not None
                       else _default_points(key),
                       (seed.handles or 'VECTOR') if seed is not None
                       else _DEFAULT_HANDLES.get(key, 'AUTO'))
        # Re-asserted rather than set once at creation, so a change to
        # `_CLIP_MAX_Y` reaches plants that already exist. Clipping applies on
        # edit, so this never rewrites an authored point.
        ceiling = _CLIP_MAX_Y.get(key, 1.0)
        if node.mapping.clip_max_y != ceiling:
            node.mapping.clip_max_y = ceiling
            # The widget's view is independent of the clip rect, so a raised
            # ceiling would otherwise leave a flat 1.0 curve drawn exactly on
            # the top edge, reading as an empty graph.
            node.mapping.reset_view()
        nodes[key] = node
    return nodes


def get_curves(obj) -> dict:
    """Whatever curve nodes this object already has -- for draw code, which
    must not create datablocks.

    **Returns a partial set rather than nothing, and that is a bug fix.** It
    used to be all-or-nothing (`found if len(found) == len(CURVE_SPECS)`), so
    the moment a CurveSpec was added *every* graph on *every* existing plant
    went blank at once -- pedicels, leaves, development, all of them -- because
    each object's group was built when the registry was shorter. The object is
    not broken and nothing is lost; it simply has not been regrown since, and
    `ensure_curves` backfills on the next regrow. Blanking the other twenty
    curves to say so was the wrong report.

    Callers must therefore treat a missing key as "not created yet" rather than
    indexing it -- see `repaint_icons` and `signature` below, and
    `ui._make_curve_panel`, which says so in the popover instead of drawing an
    empty box."""
    group = obj.plantgen.curve_group if obj else None
    if group is None:
        return {}
    return {n.get(_KEY): n for n in group.nodes if n.get(_KEY) in CURVE_KEYS}


# The inflorescence curves that move onto a flower component in the port
# -- every curve whose profile attribute is
# on `inflorescence.*` (flower size/pitch, the pedicel group, head dome, and the
# compound-branch group). Leaf/stem/development curves stay on the plant.
_FLOWER_CURVE_KEYS = tuple(
    spec.key for spec in CURVE_SPECS if spec.attr.startswith("inflorescence."))
_FLOWER_CURVE_ATTR = {
    spec.key: spec.attr for spec in CURVE_SPECS if spec.attr.startswith("inflorescence.")}


def _component_curve_key(uid: int, name: str = "size") -> str:
    """The `_KEY` tag for one of a component's own curves (`name`). Per-component
    curves share the plant's one curve group but sit outside `CURVE_KEYS`, so
    `get_curves` and the fixed-spec machinery ignore them and they ignore it. A
    leaf component uses just "size"; a flower component adds one per flower curve."""
    return f"comp:{int(uid)}:{name}"


def ensure_component_curve(obj, uid: int, name: str = "size"):
    """A named curve node for one component (by its stable uid), created flat
    (1.0 everywhere = no change) if missing, so it defaults byte-identical to no
    curve at all."""
    s = obj.plantgen
    group = s.curve_group
    if group is None:
        group = bpy.data.node_groups.new(_GROUP_NAME, 'ShaderNodeTree')
        group.use_fake_user = True
        s.curve_group = group
    key = _component_curve_key(uid, name)
    node = next((n for n in group.nodes if n.get(_KEY) == key), None)
    if node is None:
        node = group.nodes.new('ShaderNodeFloatCurve')
        node[_KEY] = key
        node.label = key
        set_points(node, _default_points(name), 'AUTO')
    return node


def get_component_curve(obj, uid: int, name: str = "size"):
    """A component's named curve node if it exists -- for draw code and the bake,
    neither of which may create datablocks."""
    group = obj.plantgen.curve_group if obj else None
    if group is None:
        return None
    key = _component_curve_key(uid, name)
    return next((n for n in group.nodes if n.get(_KEY) == key), None)


def remove_component_curve(obj, uid: int) -> None:
    """Drop ALL of a component's curve nodes when it is removed, so the group does
    not accrue orphans as components come and go."""
    group = obj.plantgen.curve_group if obj else None
    if group is None:
        return
    prefix = _component_curve_key(uid, "")
    for node in [n for n in group.nodes if str(n.get(_KEY) or "").startswith(prefix)]:
        group.nodes.remove(node)


def copy_component_curves(obj, src_uid: int, dst_uid: int) -> None:
    """Copy every curve node of one component onto another (a component copy):
    its size / flower curves and its instance maps' curves."""
    group = obj.plantgen.curve_group if obj else None
    if group is None:
        return
    prefix = _component_curve_key(src_uid, "")
    for node in [n for n in group.nodes if str(n.get(_KEY) or "").startswith(prefix)]:
        name = str(node.get(_KEY))[len(prefix):]
        dst = ensure_component_curve(obj, dst_uid, name)
        if name.startswith(("imap", "fmap")):
            dst.mapping.clip_max_y = node.mapping.clip_max_y
        copy_curve(node, dst)


GROWTH_MAP_OWNER = 0
"""Growth maps are plant-level, not a component's: their curves live under this
pseudo component uid (real components start at 1)."""


def map_curve_name(kind: str, uid: int) -> str:
    """The curve-group node name of one map: `imap<uid>` under its component for
    an instance map, `gmap<uid>` under `GROWTH_MAP_OWNER` for a growth map."""
    return f"{kind}{uid}"


MAP_CLIP_MAX_Y = 2.0
"""A map's curve is a multiplier that must be able to raise a value as well as
lower it (denser, longer, bigger), so its ceiling is 2 -- a fresh curve mapping
clips at 1.0, which would have let a map only ever reduce."""


def ensure_map_curve(obj, owner_uid: int, name: str):
    """A map's curve node (created flat = no-op), with the map ceiling."""
    node = ensure_component_curve(obj, owner_uid, name)
    mapping = node.mapping
    if mapping.clip_max_y != MAP_CLIP_MAX_Y:
        mapping.clip_max_y = MAP_CLIP_MAX_Y
    return node


def remove_named_component_curve(obj, uid: int, name: str) -> None:
    """Drop one named curve node (a removed map's)."""
    node = get_component_curve(obj, uid, name)
    if node is not None:
        obj.plantgen.curve_group.nodes.remove(node)


def sample_node(node, x):
    """Evaluate a curve node at `x` (array of 0-1 values) via its baked samples
    -- the same `to_curve(node)` the growth side reads, vectorised with interp."""
    import numpy as np
    samp = np.asarray(to_curve(node).samples, dtype=np.float64)
    if samp.size < 2:
        return np.full(np.shape(x), float(samp[0]) if samp.size else 1.0)
    xs = np.linspace(0.0, 1.0, samp.size)
    return np.interp(np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0), xs, samp)


def copy_curve(src, dst) -> None:
    """Copy src's control points (locations and handle types) onto dst, so dst
    samples identically -- used to seed a flower component's curves from the
    plant's."""
    pts = [(p.location[0], p.location[1], p.handle_type)
           for p in src.mapping.curves[0].points]
    curve = dst.mapping.curves[0]
    while len(curve.points) > max(len(pts), 2):
        curve.points.remove(curve.points[-1])
    while len(curve.points) < len(pts):
        curve.points.new(0.5, 0.5)
    for point, (x, y, h) in zip(curve.points, pts, strict=False):
        point.location = (x, y)
        point.handle_type = h
    dst.mapping.update()


def ensure_component_flower_curves(obj, uid: int) -> None:
    """Ensure every inflorescence curve node exists on a flower component."""
    for key in _FLOWER_CURVE_KEYS:
        ensure_component_curve(obj, uid, key)


def apply_component_flower_curves(obj, comp, profile) -> None:
    """Override the profile's inflorescence curves from a flower component's own
    curve nodes, leaving any it lacks as the plant set them."""
    for key in _FLOWER_CURVE_KEYS:
        node = get_component_curve(obj, comp.uid, key)
        if node is not None:
            _set_attr(profile, _FLOWER_CURVE_ATTR[key], to_curve(node))


def seed_component_flower_curves(obj, comp) -> None:
    """Copy the plant's inflorescence curves onto a flower component's own curves,
    so it inherits them (the additive seed). A missing plant curve leaves the
    component's flat, which is the plant's default too."""
    ensure_curves(obj)
    plant_nodes = get_curves(obj)
    for key in _FLOWER_CURVE_KEYS:
        dst = ensure_component_curve(obj, comp.uid, key)
        src = plant_nodes.get(key)
        if src is not None:
            copy_curve(src, dst)


def set_points(node, points, handle_type: str = 'AUTO') -> None:
    """Replace a curve's control points. A CurveMap always keeps at least two,
    so the list is resized around that floor rather than emptied and refilled.

    `handle_type` follows the parametric add-on's own rule: 'AUTO' normally,
    'AUTO_CLAMPED' for the curvier shape presets (a CurveMapPoint exposes no
    numeric handle offsets, so clamping is as close as a scripted curve gets to
    hand-tuned handles without overshooting its own points). Seeding a species'
    curves passes 'VECTOR' instead -- see `seed_from_profile`.
    """
    points = [(float(x), float(y)) for x, y in points] or list(_FLAT)
    # one handle type for every point, or one per point (a saved widget curve's)
    types = ([str(handle_type)] * len(points) if isinstance(handle_type, str)
             else list(handle_type) + ['VECTOR'] * (len(points) - len(handle_type)))
    mapping = node.mapping
    curve = mapping.curves[0]
    while len(curve.points) > max(len(points), 2):
        curve.points.remove(curve.points[-1])
    while len(curve.points) < len(points):
        curve.points.new(0.5, 0.5)
    for point, (x, y), kind in zip(curve.points, points, types, strict=False):
        point.location = (x, y)
        point.handle_type = kind
    mapping.update()


def apply_shape(node, shape: str) -> bool:
    """Write one of `SHAPES` onto a curve. False if `shape` is not a known key."""
    data = _SHAPE_DATA.get(shape)
    if data is None:
        return False
    points, smooth = data
    set_points(node, points, 'AUTO_CLAMPED' if smooth else 'AUTO')
    return True


def read_points(node) -> tuple:
    return tuple((round(p.location[0], 6), round(p.location[1], 6))
                 for p in node.mapping.curves[0].points)


def to_curve(node) -> Curve:
    return Curve.from_blender(node.mapping)


def apply_to_profile(obj, profile) -> None:
    """Point every profile curve at what the widgets currently hold."""
    nodes = get_curves(obj)
    for key, attr in _ATTR_BY_KEY.items():
        node = nodes.get(key)
        if node is not None:
            exact = _exact_curve(key, node)
            _set_attr(profile, attr, exact if exact is not None else to_curve(node))


def seed_from_profile(obj, profile) -> None:
    """Write a species' own curves into the widgets, so switching species moves
    the curves too rather than leaving them silently disagreeing with the
    plant. Uses each Curve's authoring points -- see `Curve.points`; handing
    back 64 samples would produce an uneditable widget."""
    nodes = ensure_curves(obj)
    for key, attr in _ATTR_BY_KEY.items():
        curve = _get_attr(profile, attr)
        if curve is not None:
            # VECTOR, not AUTO: a species authors its curves with
            # `Curve.from_points`, which is piecewise-linear, and AUTO's
            # smoothing pulled Rudbeckia's internode profile off its own
            # authored values by up to 0.11. Seeding a widget must not quietly
            # restyle the species; the widget's handle buttons still let anyone
            # smooth it afterwards.
            node = nodes.get(key)
            if node is not None:
                # (a widget curve saved in a preset: its own handles, exactly)
                set_points(node, curve.points or _FLAT, curve.handles or 'VECTOR')


def matching_shape(node) -> str | None:
    """The label of the preset this curve currently matches, if any -- so the
    shape dropdown can show a real "Custom" state rather than always naming
    whichever preset was picked last. Purely derived: a hand-dragged point
    simply stops matching."""
    current = [(round(x, 4), round(y, 4)) for x, y in read_points(node)]
    for _key, label, points, _smooth in SHAPES:
        if current == [(round(x, 4), round(y, 4)) for x, y in points]:
            return label
    return None


def icon_id(key: str) -> int:
    """The sparkline icon for one curve's popover button."""
    return pg_icons.curve_icon_id(ICON_PREFIX + key)


def component_icon_id(uid: int, name: str = "size") -> int:
    """The live sparkline for one of a component's curves."""
    return pg_icons.curve_icon_id(ICON_PREFIX + _component_curve_key(uid, name))


def _component_curve_nodes(obj):
    """Every per-component curve node on this object (`comp:*` keys), covering a
    leaf component's size curve and a flower component's curves alike."""
    group = obj.plantgen.curve_group if obj else None
    if group is None:
        return []
    return [n for n in group.nodes if str(n.get(_KEY) or "").startswith("comp:")]


def repaint_component_icons(obj) -> None:
    """Repaint every per-component curve sparkline. Called at regrow (once the
    nodes are ensured) so a freshly-created curve shows its shape immediately,
    rather than a blank preview until the watcher's next tick."""
    for node in _component_curve_nodes(obj):
        pg_icons.set_curve_icon(ICON_PREFIX + node.get(_KEY), list(read_points(node)))


def repaint_icons(obj) -> None:
    """Redraw every sparkline from the curves as they now are. Called by the
    live watcher, which is already comparing them for changes."""
    nodes = get_curves(obj)
    for key in CURVE_KEYS:
        node = nodes.get(key)
        if node is not None:
            pg_icons.set_curve_icon(ICON_PREFIX + key, list(read_points(node)))
    repaint_component_icons(obj)


def signature(obj) -> tuple:
    """A cheap hash of every curve on this object, for the live-update watcher.
    Curve widgets fire no property callbacks, so a change is only ever visible
    by comparing against what was there last tick."""
    def _pts(node):
        return tuple((round(p.location[0], 5), round(p.location[1], 5), p.handle_type)
                     for p in node.mapping.curves[0].points)

    nodes = get_curves(obj)
    fixed = tuple((key, _pts(nodes[key])) for key in CURVE_KEYS if key in nodes)
    # Component curves live in the same group but outside CURVE_KEYS -- fold them
    # in so an edit to one is seen by the watcher (repaint + regrow) too.
    comp = tuple((node.get(_KEY), _pts(node)) for node in _component_curve_nodes(obj))
    return fixed + comp
