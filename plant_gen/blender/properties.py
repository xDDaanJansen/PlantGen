"""The parameter surface, held to one rule: **a control earns a place here if
it names a property of the plant, not a property of the solver.**

Internode length, branch angle, leaf size, stem count are botanical
measurements and belong. Perception radius, kill radius, vigor lambda/alpha,
light sweeps, attraction point count and iteration count are solver internals
and are simply gone -- thirteen controls removed against the previous surface,
which was the direct cause of "a lot of parameters that are reasonably
unclear".

Age is the master dial. It is not a scale factor: it selects how far along a
single fixed developmental trajectory the plant is, and each species
interprets it through its own curves. Everything the previous build exposed as
`iterations` is now a species constant behind it.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Collection, Material, NodeTree, PropertyGroup

from . import live, overlay


def _changed(self, context):
    live.mark_dirty(self.id_data)


def _is_mesh(self, obj) -> bool:
    return obj.type == 'MESH'


def _gn_display_changed(self, context):
    """A setting only the node graph reads: push its socket, never regrow
    (`mesh.push_display`)."""
    from . import mesh as gn_mesh

    gn_mesh.push_display(self.id_data)


def _uv_checker_changed(self, context):
    """The checker is an image on a material, which Solid shading only shows in
    Texture colouring -- so turning it on switches every Solid 3D view on the
    screen to Texture (and back to Material when turned off, if still Texture).
    Then the ordinary rebuild pushes the material override."""
    screen = getattr(context, "screen", None) if context is not None else None
    for area in (screen.areas if screen is not None else ()):
        if area.type != 'VIEW_3D':
            continue
        shading = area.spaces.active.shading
        if shading.type != 'SOLID':
            continue
        if self.uv_checker:
            shading.color_type = 'TEXTURE'
        elif shading.color_type == 'TEXTURE':
            shading.color_type = 'MATERIAL'
    _gn_display_changed(self, context)


def _animate_day_changed(self, context):
    """Timelapse (phase 6): when Animate Day is on -- or its range changes while
    on -- drive Day from the current frame right away, so the panel and viewport
    reflect it without waiting for the next frame change. Off is a plain edit
    that marks the plant dirty like any other, so a manual Day scrub resumes."""
    if self.animate_day and context is not None and context.scene is not None:
        live.drive_timelapse_day(self.id_data, context.scene)
    else:
        from . import timelapse_cache
        timelapse_cache.hide(self.id_data)   # the live plant, not a recorded frame
        live.mark_dirty(self.id_data)


def _rig_view_changed(self, context):
    """Show the plant as its baked rig, or as the live plant: visibility and a
    socket push, never a regrow."""
    from . import rig
    rig.apply_rig_view(self.id_data)


def _wind_changed(self, context):
    """A wind control changed. Wind is a rig-side pose/debug layer, not a growth
    input (not in `_FIELDS`, not in the stamp). If the live sway is on, re-evaluate
    the frame so the handler re-applies it on a fresh growth pose; otherwise just
    refresh the debug map in place."""
    from . import rig, wind
    wind.ensure_leaf_wind(self.id_data)   # sync the leaf-wind WPO layer (self-gates on wind_enable)
    rig.apply_rig_view(self.id_data)      # Vertex Wind needs the full skin (Fast Preview off)
    found = rig.find_rig(self.id_data)
    if found is not None and not getattr(self, "wind_enable", False):
        ad = found[0].animation_data
        if ad is None or ad.action is None:
            wind.rest_pose(found[0])          # Bone Wind off: the rig back at rest
    if getattr(self, "wind_enable", False) and context is not None and context.scene is not None:
        context.scene.frame_set(context.scene.frame_current)
    else:
        wind.refresh_debug(self.id_data)


def _wind_debug_changed(self, context):
    """Create / rebuild / remove the wind debug object(s) to match the enum + res."""
    from . import wind
    wind.ensure_debug(self.id_data)


_LOBE_DEFAULTS = {
    'winged': (4, 0.60),
    'ribbed': (10, 0.15),
    'star': (5, 0.55),
}
"""What each lobed shape means, as numbers. The parametric add-on gets this structurally
by carrying a separate count/depth pair per shape, at the cost of six fields;
this addon shares two and stamps the defaults on switch instead.

**Not optional polish.** Without it the three are one generator with identical
inputs, so Winged, Ribbed and Star produce byte-identical stems until someone
edits the numbers -- measured, all three came out at exactly 3.403-7.067 mm.
Three dropdown entries that do the same thing is precisely the defect the
parameter sweeps just found in Alternate vs Spiral, and shipping a second one
knowingly would be worse than having found the first.

The tradeoff is real and is the price of two fields instead of six: switching
Winged -> Ribbed -> Winged does not restore the wing numbers.
"""




def _cross_section_changed(self, context):
    defaults = _LOBE_DEFAULTS.get(self.cross_section)
    if defaults is not None:
        self.lobe_count, self.lobe_depth = defaults
    _changed(self, context)


_ATTACHMENT_DEFAULTS = {
    # Same reasoning as `_KIND_DEFAULTS` below, one layer down. The inflorescence
    # Placement defaults to `on_trunk=False`, which is right for the node-borne
    # positions but silently empties the two that are defined by WHICH wood they
    # break from: Cauliflorous is the trunk by definition, so with the trunk
    # excluded it produces nothing at all and the dropdown entry looks broken.
    # Measured before this existed: picking Cauliflorous on a default plant gave
    # 0 flowers.
    #
    # Stamped rather than forced, so the window stays editable afterwards -- and
    # not cleared on switching away, exactly like every other stamp here.
    'CAULIFLOROUS': {"inflorescence_on_trunk": True},
    'RAMIFLOROUS': {"inflorescence_on_branches": True},
    # Basal arises at the root crown of a first-order axis, so it needs the
    # trunk allowed for the same reason.
    'BASAL': {"inflorescence_on_trunk": True},
    'ADVENTITIOUS': {"inflorescence_on_trunk": True, "inflorescence_on_branches": True},
}


def _attachment_changed(self, context):
    """Stamp the placement an attachment needs, then regrow.

    Without this, three of the five new positions are gated out by a default
    window that predates them."""
    # Union of what every selected position needs: with two picked, both
    # windows have to be open or one of them silently produces nothing.
    for mode in self.inflorescence_attachments:
        for prop, value in _ATTACHMENT_DEFAULTS.get(mode, {}).items():
            setattr(self, prop, value)
    _changed(self, context)


_KIND_DEFAULTS = {
    # Its bracts are a Bract component's now (add one to bract the whorls).
    'VERTICILLASTER': {"inflorescence_whorl_count": 6},
    # Branching Depth defaults to 0 -- simple -- now that every kind reads it
    # as its compound control. The kinds whose *name* already implies a depth
    # have to say so, or picking Panicle would give a raceme. Everything else
    # stays simple until asked, which is what keeps the goldens exact.
    # Catkin is a kind whose whole character lives in another box, so picking it
    # has to stamp that box or the dropdown entry does nothing at all -- the
    # defect this project has now found three times (Winged/Ribbed/Star,
    # Alternate/Spiral, Verticillaster).
    # All three layers, because Catkin names a typical combination of them
    # rather than being one: an axillary, pendulous, tube-bodied inflorescence.
    # Each stays independently settable afterwards.
    'CATKIN': {"inflorescence_attachments": {'AXILLARY'},
               "inflorescence_orientation": 'PENDULOUS',
               "inflorescence_orientation_strength": 0.85,
               "inflorescence_on_trunk": False, "inflorescence_start": 0.45,
               "catkin_length": 0.075, "catkin_length_variation": 0.25,
               "catkin_count": 3, "catkin_width": 0.0045,
               "catkin_stalk": 0.018, "catkin_noise": 0.3},
    'PANICLE': {"inflorescence_branching_depth": 1},
    # A cyme opens from its terminal flower outward (`development._cyme_bloom_start`),
    # so the five cymose kinds stamp Middle first.
    'CYME': {"inflorescence_branching_depth": 3, "bloom_order": 'CENTRIFUGAL'},
    # The four cymose forms are one mechanism with two numbers set differently,
    # so the dropdown entry has to set them or every one of them is the same
    # plant -- the Alternate/Spiral defect. Depth is stamped too, because it is
    # what makes each form read as itself: a scorpioid wants a long chain, a
    # dichasium a short one (its count doubles each generation).
    'SCORPIOID': {"inflorescence_laterals": 1, "inflorescence_relay_rotation": 180.0,
                  "inflorescence_branching_depth": 8, "inflorescence_branch_angle": 45.0,
                  "bloom_order": 'CENTRIFUGAL'},
    # All four stamp the zigzag angle, including the three that keep the shared
    # default of 45 -- the same reason `_LOBE_DEFAULTS` stamps a count and depth
    # for every one of Winged/Ribbed/Star rather than only the odd one out. A
    # stamp is never cleared on switching away, so Helicoid -> Scorpioid would
    # otherwise leave the zigzag at Helicoid's 60 and read as a weak scorpioid.
    #
    # Helicoid's gentler kink is measured rather than taste: the coil's total
    # turn is exactly (90 - Zigzag Angle) per relay, so
    # the shared default of 45 turns 45 x 7 = 267 degrees over eight relays and
    # closes into a ring, which reads as a mistake. 60 gives a half turn -- an
    # open, uncoiling spray, which is what a bostryx looks like.
    'HELICOID': {"inflorescence_laterals": 1, "inflorescence_relay_rotation": 0.0,
                 "inflorescence_branching_depth": 8, "inflorescence_branch_angle": 60.0,
                 "bloom_order": 'CENTRIFUGAL'},
    'DICHASIUM': {"inflorescence_laterals": 2, "inflorescence_relay_rotation": 90.0,
                  "inflorescence_branching_depth": 3, "inflorescence_branch_angle": 45.0,
                  "bloom_order": 'CENTRIFUGAL'},
    'PLEIOCHASIUM': {"inflorescence_laterals": 3, "inflorescence_relay_rotation": 60.0,
                     "inflorescence_branching_depth": 2, "inflorescence_branch_angle": 45.0,
                     "bloom_order": 'CENTRIFUGAL'},
}
"""What picking a kind means, where the kind's own character is carried by
controls that are also reachable from other kinds.

Same job as `_LOBE_DEFAULTS` and the same reason: Flowers per Whorl and Bract
Size both default to off, because off is what every other kind has always done
and is what keeps the goldens exact. So a Verticillaster picked from the
dropdown would be a plain spike until someone found two rows and guessed at
them -- a dropdown entry that does nothing is the defect this project has now
found three times (Winged/Ribbed/Star, Alternate/Spiral).

Stamped on switching *to* a kind and never cleared on switching away, again
matching `_LOBE_DEFAULTS`. The difference is worth stating: a lobe count means
nothing to a circular stem, but a whorl count means something to a raceme, so
going Verticillaster -> Raceme leaves a whorled, bracted raceme. That is a real
plant and both controls stay visible saying why -- clearing them would instead
destroy the tuning of anyone deliberately whorling a raceme.
"""


def _branching(angle, nodes, count, spacing, length, width, bend, droop):
    """One Branching-box setting, written out in full.

    **Every preset states every field, including the ones it leaves at the
    default.** A stamp is never cleared on switching away, so a preset that
    named only its own departures would inherit whichever of the others was
    picked before it -- Compound Corymb then Compound Umbel would give an umbel
    carrying the corymb's short, hard-arched branches. That is the same reason
    `_LOBE_DEFAULTS` stamps a count and depth for all three lobed shapes and all
    four cymose kinds stamp the zigzag angle.
    """
    return {
        "inflorescence_branch_angle": angle,
        "inflorescence_branch_nodes": nodes,
        "inflorescence_branch_count": count,
        "inflorescence_branch_spacing": spacing,
        "inflorescence_branch_length": length,
        "inflorescence_branch_width": width,
        "inflorescence_branch_bend": bend,
        "inflorescence_branch_droop": droop,
    }


_COMPOUND_PRESETS = {
    'COMPOUND_SPIKE': ('SPIKE', _branching(45.0, 2, 7, 1.5, 1.0, 0.60, 0.0, 0.0)),
    'COMPOUND_CORYMB': ('CORYMB', _branching(30.0, 3, 7, 0.5, 1.0, 0.35, 1.0, 0.2)),
    'COMPOUND_UMBEL': ('UMBEL', _branching(45.0, 2, 7, 1.5, 3.0, 0.60, 0.0, 0.0)),
    'COMPOUND_FLAT_UMBEL': ('FLAT_UMBEL', _branching(45.0, 2, 10, 1.5, 5.0, 0.39, 0.5, 0.0)),
}
"""The four compound forms, as *presets over an existing kind* rather than kinds
of their own.

Phase 4A ruled that a compound inflorescence is a composition of two orders and
that inventing enum entries for it would be wrong, and the 2026-08-19
generalisation measured that ruling holding: Umbel + Depth 1 **is** a compound
umbel, and Raceme + Depth 1 is byte-for-byte identical to Panicle. Nothing here
reopens that. `branching_depth` is still the one control, and these four write
it rather than standing beside it.

So picking one of these **snaps the dropdown back to its base kind** with
Branching Depth set. That is deliberate and it is the whole reason this is safe:
the panel can never end up saying "Compound Umbel" while Branching Depth reads 0,
which is exactly the lie a real enum entry would have made possible. What you
picked is visible in Branching Depth, not in the Type field.

What each one stamps is the **Branching box** -- the controls that mean nothing
at depth 0, so stamping them cannot destroy tuning that was doing anything. The
numbers are the ones measured off the four hand-tuned demo plants in
`PGS_Sweep_inflorescence_kind2`. Deliberately *not* stamped: Flower Count, Head
Radius, Head Dome, Flower Spacing. Those are live on the simple kind too, so
writing them would wipe real tuning to make a preset look like its demo.
"""


def _inflorescence_kind_changed(self, context):
    # Reset the two cross-contaminating "compound-ness" controls to base first, so
    # a kind that does not stamp them (a plain raceme/umbel/spike/head) comes out
    # clean rather than inheriting the previous kind's whorl or depth -- otherwise
    # Verticillaster -> Raceme leaves whorl at 6 and the raceme reads as whorled,
    # and Panicle -> anything leaves depth at 1. Stamping kinds below override.
    self.inflorescence_whorl_count = 1
    self.inflorescence_branching_depth = 0
    preset = _COMPOUND_PRESETS.get(self.inflorescence_kind)
    if preset is not None:
        base, values = preset
        # Base kind first: assigning it re-enters this callback, which stamps
        # `_KIND_DEFAULTS` for the base. Doing it first means those stamps
        # cannot land on top of the compound values set below.
        self.inflorescence_kind = base
        self.inflorescence_branching_depth = 1
        for prop, value in values.items():
            setattr(self, prop, value)
        _changed(self, context)
        return
    for prop, value in _KIND_DEFAULTS.get(self.inflorescence_kind, {}).items():
        setattr(self, prop, value)
    _changed(self, context)


# Structural presets -- a named starting point for the physical-form controls,
# the same stamp-then-snap-to-Custom idea as `_COMPOUND_PRESETS`. Each is
# (stem_stiffness factor 0-1, stiffness_variation, leaf_weight, flower_weight,
# stiffness-curve shape). The curve is what separates a Weeping preset (soft tip
# = `fade_out`) from an upright one (`max`, uniform); the rest are scalars.
#
# Stiffness factors read against `bridge._stiffness_from_factor`'s new real-Pa
# map (0.5 = 1.5e8, 0 = 5 MPa weeping, 1 = 5 GPa hardwood), so 0.45 is a normal
# herbaceous stem, 0.6-0.75 grass-culm-to-shrub, 0.9 near hardwood.
_STRUCTURAL_PRESETS = {
    'HERB_UPRIGHT':  (0.46, 0.10, 8.0e-5, 4.0e-4, 'max'),
    'HERB_LAX':      (0.26, 0.15, 8.0e-5, 4.0e-4, 'droop'),
    'CLIMBING':      (0.12, 0.20, 6.0e-5, 3.0e-4, 'max'),
    'GRASS_CULM':    (0.62, 0.10, 4.0e-5, 2.0e-4, 'max'),
    'REED':          (0.72, 0.08, 5.0e-5, 2.0e-4, 'max'),
    'SUCCULENT':     (0.28, 0.05, 3.0e-4, 8.0e-4, 'max'),
    'SAPLING':       (0.60, 0.15, 1.0e-4, 5.0e-4, 'max'),
    'SHRUB':         (0.78, 0.18, 1.2e-4, 6.0e-4, 'max'),
    'RIGID':         (0.92, 0.08, 1.0e-4, 5.0e-4, 'max'),
    'WEEPING':       (0.42, 0.15, 1.0e-4, 6.0e-4, 'fade_out'),
}


def _authored(self, context, flag: str) -> None:
    """A value seeded once from the plant (a leaf component's Weight, a level's
    Stiffness): an edit marks it set, so the seeding never overwrites it."""
    if not self.get(flag):
        self[flag] = True
    _changed(self, context)


def _leaf_weight_changed(self, context):
    _authored(self, context, "leaf_weight_authored")


def _level_stiffness_changed(self, context):
    _authored(self, context, "stiffness_authored")


ELONGATION_MODE_ITEMS = (
    ('SEQUENTIAL', "Sequential", "Each internode reaches its full length before the next "
     "one forms -- one stretching at a time", 0),
    ('OVERLAPPING', "Overlapping", "Several internodes stretch at once: the lower ones keep "
     "growing for a while after new ones form above (most herbs)", 1),
    ('DELAYED', "Delayed", "A new internode stays short while the next nodes form, then "
     "stretches -- compressed young growth that spreads out later", 2),
    ('CUSTOM', "Custom", "Your own curve and Duration", 3),
)


def _get_elongation_mode(self) -> int:
    """The mode the Internode Elongation curve + Duration match (Custom if
    none): read off the values, never stored, so it can't claim a preset the
    curve has since been drawn away from."""
    from . import curves
    node = curves.get_curves(self.id_data).get("elongation")
    mode = curves.elongation_mode_of(node, int(self.elongation_steps))
    return next(n for ident, _l, _d, n in ELONGATION_MODE_ITEMS if ident == mode)


def _set_elongation_mode(self, value: int) -> None:
    """Stamp a mode's curve + Duration (Custom stamps nothing: draw your own)."""
    from ..growth.curves import elongation_mode
    from . import curves
    ident = next(i for i, _l, _d, n in ELONGATION_MODE_ITEMS if n == value)
    if ident == 'CUSTOM':
        return
    curve, steps = elongation_mode(ident)
    node = curves.ensure_curves(self.id_data).get("elongation")
    if node is not None:
        curves.set_points(node, curve.points, 'VECTOR')
        curves.repaint_icons(self.id_data)
    self.elongation_steps = steps            # its update regrows


def _structural_preset_changed(self, context):
    """Stamp a preset's stiffness, variance, weights and stiffness-curve shape,
    then snap the dropdown back to Custom -- so the panel can never claim a
    preset the numbers have since drifted from, exactly like `_COMPOUND_PRESETS`.
    Custom stamps nothing."""
    preset = _STRUCTURAL_PRESETS.get(self.structural_preset)
    if preset is None:
        return
    stiffness, variance, leaf_w, flower_w, shape = preset
    self.stem_stiffness = stiffness
    self.stiffness_variation = variance
    self.leaf_weight = leaf_w
    self.flower_weight = flower_w
    # Stiffness lives on the levels now, and the weights on the components (each
    # copy overrides the plant's), so the preset stamps them there too or it
    # would not reach them.
    for lv in self.branch_levels:
        lv.stiffness = stiffness
        lv.stiffness_authored = True
    for c in self.components:
        if c.type == 'leaf':
            c.leaf_weight = leaf_w
            c.leaf_weight_authored = True
        elif c.type == 'flower':
            c.flower_weight = flower_w
    # The stiffness curve, if the object already has one (a fresh object gets it
    # on its first regrow; get_curves returns a partial set until then).
    from . import curves
    node = curves.get_curves(self.id_data).get('stiffness')
    if node is not None:
        curves.apply_shape(node, shape)
    # Snap to Custom -- re-enters this callback, which returns on the Custom key.
    self.structural_preset = 'CUSTOM'
    _changed(self, context)


def _bloom_display_changed(self, context):
    """Push sockets, never regrow -- see `mesh.push_bloom`. Every bloom control
    below this line is a display setting; Order and Stagger are not, because
    they are baked into the plant at grow time and keep `_changed`."""
    from . import mesh as gn_mesh

    gn_mesh.push_bloom(self.id_data)


DEBUG_VIEW_MODES = (
    ('off', "Off", "No debug overlay - normal viewport shading", "RESTRICT_VIEW_ON", 0),
    ('color', "Vertex Colors", "The plant's real exported vertex colors, all four "
     "channels combined", "COLOR", 1),
    ('vcol_r', "Red: Leaf Gradient", "Red alone - 0 at each leaf's own attachment "
     "point rising to 1 at its own tip, flat on stems and flowers", "NONE", 2),
    ('vcol_g', "Green: Per Organ", "Green alone - one random value per leaf or "
     "flower", "NONE", 3),
    ('vcol_b', "Blue: Per Plant", "Blue alone - one random value for this whole "
     "object, identical everywhere. Flat by design", "NONE", 4),
    ('vcol_a', "Alpha: Height", "Alpha alone - height fraction of the whole plant, "
     "root to tip", "NONE", 5),
    ('height', "Height", "How far up the plant each part sits, 0 at the base rising "
     "to 1 at the highest point - the same value the Height instancing mask reads. "
     "Root to tip over stem and organs together", "NONE", 24),
    ('shade', "Shade", "How shaded each part is by the plant's own canopy, 0 in open "
     "sky to 1 fully shaded - the same value the Shade instancing mask reads. Only a "
     "plant with foliage casts any; switching this view on recomputes it", "NONE", 25),
    ('mask_preview', "Mask Preview", "What the instancing masks keep, red (dropped) to "
     "green (kept), per organ. The windowed value, so it moves as you drag a map's "
     "Min/Max and follows whichever attribute the map reads. Stem reads as kept", "NONE", 26),
    ('component', "Component Colours", "Every leaf, flower, bud and fruit in the colour "
     "of the component that draws it (its swatch in the Components list). Stems and "
     "organs no component takes read grey", "NONE", 27),
    ('order', "Branch Order", "Which branch order each part belongs to. The first "
     "thing to look at when a plant branches more than it should", "NONE", 6),
    ('axis_id', "Axis", "A different colour per stem or branch, so it is visible "
     "which vertices belong to one axis", "NONE", 7),
    ('node_index', "Node Number", "Position of each node along its own axis, base "
     "to tip", "NONE", 8),
    ('birth_step', "Age", "When each part was created - dark is old, bright is "
     "new growth", "NONE", 9),
    ('width', "Stem Width", "Stem radius at every point, so taper is visible "
     "directly rather than inferred", "NONE", 10),
    ('kind', "Organ Kind", "Leaves against flowers", "NONE", 11),
    ('bloom_order', "Bloom Order", "Which flower opens first - dark opens first, bright "
     "last. This is what Bloom Order sets and what Bloom Spread staggers over", "NONE", 12),
    ('bloom', "Bloom Openness", "How open each flower is right now, from Bloom Progress "
     "and Spread. Dark is a bud, bright is fully out", "NONE", 13),
    ('organ_scale', "Organ Size", "Each leaf and flower's own expansion, 0 to 1. Leaves "
     "inflate as they mature; flowers do not, because the bloom stages already show an "
     "immature flower as an actual bud rather than a shrunken open one", "NONE", 14),
    ('bloom_birth', "Bloom Birth", "When each flower actually appeared, on the same "
     "timeline Bloom Progress runs on. Compare against Bloom Order: where a flower's "
     "birth is later than its order, From Age holds it shut until it exists", "NONE", 15),
    ('dead', "Senescence", "Alive vs dead: living tissue in green, dead in a grey-brown. "
     "The viewport counterpart to the Dead Colour tint, which only a Color-sampling "
     "shader shows. Shed parts have no geometry, so they simply are not here", "NONE", 16),
    ('autumn', "Autumn", "How far each leaf has turned as the plant enters autumn, from "
     "the day's position in the senescence window: green through amber to brown. This is "
     "the viewport counterpart to the exported 'autumn' attribute a leaf shader mixes its "
     "own colour on", "NONE", 22),
    ('ripeness', "Ripeness", "How ripe the fruit is, from the day's position in the fruit "
     "window: green through amber to red. Every fruit shows one stage at a given Day (the "
     "window is plant-level), and it is 0 on everything that is not fruit. The viewport "
     "counterpart to the exported 'ripeness' attribute a fruit shader mixes its colour on",
     "NONE", 23),
    ('structural_depth', "Structural Depth", "How deep inside its own inflorescence "
     "each part sits, which is a different count from Branch Order - vegetative stem reads "
     "as its own colour. This is what Branching Depth sets, and comparing the two views is "
     "how you see that a panicle's depth is independent of the plant's branching", "NONE", 17),
    # The one beam that now bends every organ, shown as its four ingredients.
    # 16 is deliberately left unused (the retired Peduncle Bend view) so no
    # other stored index shifts.
    ('curvature', "Bend: Curvature", "How sharply each node actually bends, in degrees per "
     "metre. This is the realized turn: where the never-kink cap holds a joint back it reads "
     "lower here than Moment alone would imply", "NONE", 18),
    ('load', "Bend: Carried Load", "How much mass hangs beyond each node, in kilograms - every "
     "flower, leaf and length of wood above it. This is what Flower Weight and Leaf Weight feed "
     "into the bend", "NONE", 19),
    ('stiffness', "Bend: Stiffness", "The local Young's modulus at each node, after the "
     "Stiffness curve and its variance. Bright is stiff, dark is soft - this is the "
     "lignification gradient the base-to-tip curve draws", "NONE", 20),
    ('moment', "Bend: Moment", "The bending moment at each node - the load times its horizontal "
     "reach, in newton-metres. High moment where curvature is low means a stiff or thick section "
     "is resisting the load", "NONE", 21),
)


def _redraw_viewports(self, context):
    """Repaint every 3D view. The inspector overlay is a draw handler rather
    than geometry, so nothing in the depsgraph changes when one of its settings
    does and Blender has no reason to redraw on its own."""
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _debug_view_mode_update(self, context):
    """Show the new debug view: the Debug Mode socket pushed (one evaluation,
    no regrow, no Node Editor needed), and Solid views switched to Attribute
    colouring while a view is on (back to what they were when it goes off)."""
    from . import mesh as gn_mesh

    gn_mesh.push_debug(self.id_data)
    gn_mesh.set_debug_shading(context, self.debug_view_mode != 'off')


def _preview_target(s):
    """The component + filter entry the Mask Preview visualises: the active
    component's active filter, or (None, None) when there is nothing to preview."""
    if not (0 <= s.components_active < len(s.components)):
        return None, None
    c = s.components[s.components_active]
    if 0 <= c.filters_active < len(c.filters):
        return c, c.filters[c.filters_active]
    return c, None


def _component_context_changed(self, context):
    """Keep the Filter Preview honest when the active component or filter changes.
    With the preview on: turn it off when there is no valid active filter,
    otherwise re-bake so the gradient follows the newly selected filter."""
    if self.debug_view_mode != 'mask_preview' or self.mask_preview_level >= 0:
        return  # off, or showing a branching level's mask (not the component's)
    _c, entry = _preview_target(self)
    if entry is not None:
        _changed(self, context)
    else:
        self.debug_view_mode = 'off'


def _level_filter_active_changed(self, context):
    """A level's active mask changed: re-bake the preview if it shows this level."""
    s = self.id_data.plantgen
    if s.debug_view_mode == 'mask_preview' and s.mask_preview_level >= 0:
        _changed(self, context)


def _structure_changed(self, context):
    """A tree row expanded / collapsed: redraw the Plant Structure rows only."""
    from . import structure
    structure.rebuild_tree(self.id_data.plantgen)


def _swatch_changed(self, context):
    """A Plant Structure swatch: its icons in that colour (`map_icons`), then the
    Component Colours view and anything else that paints it."""
    from . import map_icons
    map_icons.refresh(self.id_data.plantgen)
    _changed(self, context)


def _map_folds_changed(self, context):
    """An owner folded / unfolded in the Mapping list: rebuild its rows."""
    from . import mapping
    mapping.rebuild_rows(self.id_data.plantgen)


def _map_range_changed(self, context):
    """A Random Range map's range: redraw its curve from it, then regrow."""
    from . import mapping
    mapping.range_changed(self)
    _changed(self, context)


def _map_preset_changed(self, context):
    """A map switched preset: its driver and curve to that preset's."""
    from . import mapping
    mapping.preset_changed(self)
    mapping.rebuild_rows(self.id_data.plantgen)
    _changed(self, context)


def _map_row_selected(self, context):
    from . import mapping
    mapping.row_selected(self)


def _map_row_enabled_changed(self, context):
    """A Mapping row's box on a Random Range (+/- value) or a profile curve:
    switch it off / on."""
    from . import mapping
    s = self.id_data.plantgen
    if self.kind == "VAR":
        mapping.set_var_enabled(s, self.owner, self.owner_uid, self.param, self.enabled)
    elif self.kind == "CURVE":
        mapping.set_curve_enabled(s, self.owner, self.owner_uid, self.param, self.enabled)
        _changed(self, context)       # a curve edit fires nothing of its own


def _level_active_changed(self, context):
    """A level picked outside the Components tree (the Texturing panel's list):
    with a level selected there, the tree follows."""
    if self.structure_focus == 'LEVEL':
        from . import structure
        structure.rebuild_tree(self)


def _tree_active_changed(self, context):
    """A click on a Plant Structure row: select that level / component."""
    from . import structure
    structure.tree_row_selected(self)


def _component_label_get(self):
    if self.name_override:
        return self.name_override
    if self.collection is not None:
        return self.collection.name
    return f"{self.type.title()} (no collection)"


def _component_label_set(self, value):
    value = value.strip()
    own = self.collection.name if self.collection is not None else ""
    self.name_override = "" if value in ("", own) else value


def _center_of_mass_changed(self, context):
    """Place or remove the centre-of-mass marker at once, with no regrow: the
    COM was stored on the object at the last grow, so toggling only reads it."""
    from . import bridge

    bridge._sync_com_empty(self.id_data)


class PLANTGEN_PG_check(PropertyGroup):
    """One structural-assertion result, so the panel can draw the same eight
    (now ten) guardrails `tests/structural.py` runs headless. The point is not
    test coverage for its own sake -- every failure that triggered the rebuild
    survived a full day undetected, and each one trips a check here in under a
    second."""

    number: IntProperty(default=0)
    name: StringProperty(default="")
    status: StringProperty(default="")
    detail: StringProperty(default="")


class PLANTGEN_PG_pin(PropertyGroup):
    """One pinned inspector card, held as a component index.

    On the object rather than in the overlay's own Python state, and that is
    the whole reason it exists: **Blender's undo restores datablocks and cannot
    restore a module-level dict**, so pins kept in Python are invisible to
    Ctrl+Z. Living here, they are undone and redone for free, and they survive
    a file save as a bonus.
    """

    index: IntProperty(default=-1)


class PLANTGEN_PG_pruned(PropertyGroup):
    """One manually pruned limb, held as an **axis id** (not a component index
    like a pin). WS3 Phase 3: the user points at a component in the inspector and
    prunes the whole axis it sits on; its subtree goes with it.

    On the object for the same reason pins are -- Ctrl+Z restores datablocks,
    not a module dict -- and stamped alongside `prune_signature` so it can be
    cleared when the plant's topology changes and the id no longer names the
    same limb."""

    axis: IntProperty(default=-1)


# The mask channels, shared so the add menu can list them without reaching into a
# collection-element type's bl_rna (which is not reliably exposed on bpy.types).
MASK_CHANNEL_ITEMS = (
    ("height", "Height", "How far up the plant the organ sits (0 base, 1 top)"),
    ("branch_depth", "Branch Depth", "How many levels of branching from the main "
     "stem -- 0 on the main stem, rising with each branch order (normalised to the "
     "plant's deepest branch)"),
    ("branch_index", "Branch Index", "A distinct value per branch, so you can pick "
     "out particular branches (normalised across the plant's branches)"),
    ("on_branch", "On Branch", "0 on the main stem, 1 on any branch -- the simple "
     "stem-vs-branch split"),
    ("node_index", "Node Index", "Which node up its own axis the organ sits at, "
     "0 at the base to 1 at the tip"),
    ("factor", "Factor", "Position along the organ's own axis, 0 at the "
     "axis base to 1 at its tip"),
    ("age", "Age", "When the organ was made, 0 with the plant's first growth "
     "to 1 at the last -- low is old, high is new"),
    ("split_depth", "Split Depth", "How deep inside a fork / inflorescence "
     "the organ sits, 0 on vegetative stem rising with each split"),
    ("shade", "Shade", "How shaded the organ is by the plant's own canopy, "
     "0 in open sky to 1 fully shaded -- sampled from the self-shading light "
     "field, so a dense clump reads its shaded interior"),
    ("bloom", "Bloom", "A flower's place on the bloom order (flowers only)"),
    ("random", "Random", "Seeded per-organ noise, for scattered thinning"),
    # Appended, never inserted: the enum stores an index, so anything added
    # before an existing entry would re-point every saved filter after it.
    ("apex", "Apex", "1 on a shoot's own tip bud (its apex), 0 on axillary buds and "
     "every other organ -- keep only tip buds, or invert to keep only side buds"),
    ("custom", "Custom Attribute", "One of this plant's own attributes (Mapping & "
     "Classification panel): a distance to an object, noise, or two channels combined"),
    ("class", "Class", "One of this plant's named classes (Mapping & Classification "
     "panel): 1 where that class's own mask stack keeps the point, 0 elsewhere -- "
     "build a class once, reuse and chain it"),
    ("inflorescence", "On Inflorescence", "1 on an inflorescence -- the flowering "
     "nodes past an axis's leafy part, and the branches and stalks the inflorescence "
     "releases -- 0 on the leafy stem. Invert to keep only the vegetative part"),
)

CUSTOM_ATTR_KINDS = (
    ("DISTANCE", "Distance", "Closeness to an object: 1 at the object, falling to 0 at "
     "Radius (measured in the plant's own space)", 0),
    ("NOISE", "Noise", "Seeded 3D noise over the plant, 0-1", 1),
    ("COMBINE", "Combine", "Two channels combined into one", 2),
)

COMBINE_OPS = (
    ("MULTIPLY", "Multiply", "A x B", 0),
    ("ADD", "Add", "A + B (clamped to 1)", 1),
    ("SUBTRACT", "Subtract", "A - B (clamped to 0)", 2),
    ("DIFFERENCE", "Difference", "|A - B|", 3),
    ("MIN", "Min", "The smaller of A and B", 4),
    ("MAX", "Max", "The larger of A and B", 5),
    ("AVERAGE", "Average", "(A + B) / 2", 6),
)


def _channel_ref_props():
    """The two name references a `custom` / `class` channel needs -- shared by
    mask entries, maps and custom attributes' own inputs."""
    return {
        "attr_name": StringProperty(
            name="Attribute", default="", update=_changed,
            description="Which of this plant's custom attributes to read"),
        "class_name": StringProperty(
            name="Class", default="", update=_changed,
            description="Which of this plant's classes to read"),
    }


def _probability_get(self):
    return max(0.0, float(self.hi) - float(self.lo))


def _probability_set(self, value):
    """Probability is the band's width: Max moves to Min + it (and Min down,
    when that would run past 1). Written to the stored values directly so the
    one change regrows once, by Probability's own update."""
    value = min(1.0, max(0.0, float(value)))
    lo = min(float(self.lo), 1.0 - value)
    self["lo"] = lo
    self["hi"] = lo + value


class PLANTGEN_PG_mask_entry(PropertyGroup):
    """One entry in a map's stack.

    Reads a 0-1 point attribute (`channel`), remaps the window [`lo`, `hi`] to
    0-1 (a hard step when lo == hi), optionally inverts, and folds onto the stack
    by `op`. The bpy-free evaluator (`growth/masks.py`) turns the folded value
    into a per-point keep probability; the bridge bakes the result and one GN gate
    drops the losers. v1 exposes the channels already on the organ points; more
    arrive with the candidate-point gather."""

    channel: EnumProperty(
        name="Attribute", items=MASK_CHANNEL_ITEMS,
        default="height", update=_changed,
    )
    lo: FloatProperty(
        name="Min", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Low edge of the keep band. Points below Min are dropped")
    hi: FloatProperty(
        name="Max", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="High edge of the keep band. Points above Max are dropped -- so "
        "Min 0.4, Max 0.5 keeps only the 0.4-0.5 slice")
    probability: FloatProperty(
        name="Probability", min=0.0, max=1.0, subtype='FACTOR', options={'SKIP_SAVE'},
        get=_probability_get, set=_probability_set, update=_changed,
        description="On a Random mask: the share of the buds kept, each by a seeded "
        "random draw (the band's width, Max - Min; the level's Seed decides which)")
    invert: BoolProperty(
        name="Invert", default=False, update=_changed,
        description="Keep the OUTSIDE of the band instead of the inside")
    op: EnumProperty(
        name="Combine",
        items=[
            ("replace", "Replace", "Override the result so far with this mask"),
            ("add", "Add", "Union: keep points either keeps (OR)"),
            ("subtract", "Subtract", "Remove the points this mask keeps from the "
             "result so far"),
            ("multiply", "Multiply", "Intersect: keep only points both keep, the "
             "default (AND)"),
            ("divide", "Divide", "Divide the result so far by this mask"),
            ("max", "Max", "Keep the larger of the two (union, for hard masks)"),
            ("min", "Min", "Keep the smaller of the two (intersect, for hard masks)"),
        ],
        default="multiply", update=_changed,
        description="How this mask combines with the ones above it (ignored for the "
        "first). The same set as Natsura's Map node")
    mute: BoolProperty(
        name="Mute", default=False, update=_changed,
        description="Temporarily skip this filter -- evaluated as if it were not in "
        "the stack, without deleting it")
    math_op: EnumProperty(
        name="Math",
        items=[
            ("NONE", "None", "No math on the index"),
            ("MODULO", "Modulo", "Keep only every Nth index -- 2 every other, 3 every "
             "third. Combines with the Min/Max band; Invert flips it"),
        ],
        default="NONE", update=_changed,
        description="A math operation on the raw index (Node / Branch index), for "
        "picking out a regular subset. More operations may be added here")
    math_value: IntProperty(
        name="modulo", default=2, min=2, soft_max=8, update=_changed,
        description="The operand for the math operation (for Modulo: keep every Nth)")
    color: FloatVectorProperty(
        name="Colour", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.85, 0.35, 0.30),
        description="A label colour for this entry, to tell the stack apart in the list")


PLANTGEN_PG_mask_entry.__annotations__.update(_channel_ref_props())


class PLANTGEN_PG_custom_attr(PropertyGroup):
    """A plant-level derived channel, readable by any mask or map as
    Custom Attribute > <name>: a new dimension to classify / map on."""

    name: StringProperty(name="Name", default="Attribute", update=_changed)
    kind: EnumProperty(name="Kind", items=CUSTOM_ATTR_KINDS, default="DISTANCE",
                       update=_changed)
    target: PointerProperty(type=bpy.types.Object, name="Object", update=_changed,
                            description="The object to measure closeness to")
    radius: FloatProperty(name="Radius", default=0.3, min=1e-4, soft_max=5.0,
                          unit='LENGTH', update=_changed,
                          description="Distance at which the attribute reaches 0")
    noise_scale: FloatProperty(name="Scale", default=10.0, min=0.01, soft_max=100.0,
                               update=_changed,
                               description="Noise frequency: higher is finer grain")
    seed: IntProperty(name="Seed", default=0, update=_changed)
    op: EnumProperty(name="Operation", items=COMBINE_OPS, default="MULTIPLY",
                     update=_changed)
    channel_a: EnumProperty(name="A", items=MASK_CHANNEL_ITEMS, default="height",
                            update=_changed)
    channel_b: EnumProperty(name="B", items=MASK_CHANNEL_ITEMS, default="shade",
                            update=_changed)
    attr_a: StringProperty(name="A Attribute", default="", update=_changed)
    attr_b: StringProperty(name="B Attribute", default="", update=_changed)
    class_a: StringProperty(name="A Class", default="", update=_changed)
    class_b: StringProperty(name="B Class", default="", update=_changed)


class PLANTGEN_PG_class(PropertyGroup):
    """A plant-level named classification: a mask stack defined once, readable by
    any component's mask or map as Class > <name>, and by other classes (chaining).
    """

    name: StringProperty(name="Name", default="Class", update=_changed)
    filters: CollectionProperty(type=PLANTGEN_PG_mask_entry)
    filters_active: IntProperty(default=0)
    color: FloatVectorProperty(
        name="Colour", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.35, 0.55, 0.85))


def _component_filter_active_changed(self, context):
    """`filters_active` lives on the component, so reach up to the growth PG to
    keep the Filter Preview pointed at the newly-selected filter."""
    _component_context_changed(self.id_data.plantgen, context)


# Phyllotaxis presets for generated placement: (spiral divergence rad, whorl count,
# whorl spacing rad). They just seed the three controls below; editing any by hand
# flips the dropdown to Custom.
_PHYLLO_PRESETS = {
    "ALTERNATE": (2.39996323, 1, 0.0),   # golden spiral, one per node
    "DISTICHOUS": (3.14159265, 1, 0.0),  # 180 deg, two-ranked
    "OPPOSITE": (0.0, 2, 0.0),           # superposed pairs
    "DECUSSATE": (1.57079633, 2, 0.0),   # pairs, each rotated 90 deg from the last
    "WHORLED": (0.0, 3, 0.0),            # three per node, stacked
}
_applying_phyllo = False


def _apply_phyllo_preset(self, context):
    """Selecting a preset writes the spiral/whorl controls (guarded so those writes
    do not flip the dropdown back to Custom); Custom writes nothing."""
    global _applying_phyllo
    vals = _PHYLLO_PRESETS.get(self.gen_phyllotaxis)
    if vals is not None:
        _applying_phyllo = True
        try:
            self.gen_divergence, self.gen_whorl_count, self.gen_whorl_angle = vals
        finally:
            _applying_phyllo = False
    _changed(self, context)


def _phyllo_value_changed(self, context):
    """Editing a spiral/whorl control by hand marks the preset Custom (one regrow);
    during a preset apply it stays quiet -- the preset regrows once at the end."""
    if _applying_phyllo:
        return
    if self.gen_phyllotaxis != "CUSTOM":
        self.gen_phyllotaxis = "CUSTOM"   # fires _apply_phyllo_preset(CUSTOM) -> _changed
    else:
        _changed(self, context)


# ---------------------------------------------------------------------------
# Maps (Natsura-style "mapping every parameter", clean-room): a map reads one
# per-point channel -- the same list masks read -- through its own curve and
# multiplies a parameter by the result. Several maps on one target chain
# (multiply). Instance maps act on a component's rendered instances (render-only,
# they ride the growth cache); growth maps act on growth decisions (the plant
# grows differently). Both default to a no-op: a flat curve / zero amount.
# The curve is a per-map node in the plant's curve group (see `map_curve_name`).
# ---------------------------------------------------------------------------

INSTANCE_MAP_TARGETS = (
    ("SIZE", "Size", "Multiply each instance's size by the curve (blended by Amount): "
     "flat at 1 changes nothing, 0 hides it", 0),
    ("PITCH", "Pitch", "Tilt each instance toward its own up axis by up to +/- Angle: "
     "the curve's middle (0.5) is no tilt, 1 is +Angle, 0 is -Angle", 1),
    ("YAW", "Yaw", "Turn each instance about its own up axis by up to +/- Angle "
     "(curve middle = no turn)", 2),
    ("ROLL", "Roll", "Spin each instance about its own facing by up to +/- Angle "
     "(curve middle = no spin)", 3),
    # A parameter's own per-instance map, read where the parameter is.
    ("FOLD", "Fold", "Add up to +/- Angle to each placed instance's Fold (curve middle "
     "= none)", 4),
    ("ROLL_PARENT", "Parent Roll", "Add up to +/- Angle to each placed instance's Parent "
     "Roll (curve middle = none)", 5),
    ("PETIOLE_LENGTH", "Petiole Length", "Multiply each leaf's petiole length", 6),
    ("PETIOLE_WIDTH", "Petiole Width", "Multiply each leaf's petiole width", 7),
    ("PETIOLE_BEND", "Petiole Bend", "Add up to +/- Angle to each leaf's petiole bend "
     "(curve middle = none)", 8),
    ("PETIOLE_DROOP", "Petiole Droop", "Add up to +/- Angle to each leaf's petiole droop "
     "(curve middle = none)", 9),
    ("EMERGING_CURL", "Emerging Curl", "Multiply each young leaf's curl", 10),
    ("EMERGING_FOLD", "Emerging Midrib Fold", "Multiply each young leaf's midrib fold", 11),
    ("BLOOM", "Bloom Timing", "Shift each flower earlier or later on the bloom timeline by "
     "(curve - 1) x Amount (curve at 1 = no shift; 0 opens first, 1 last)", 12),
)

GROWTH_MAP_TARGETS = (
    # Explicit numbers are the stored values: 0 was Branch Density (removed with
    # the branching overhaul); 1 Branch Angle, 3 Branch Length and 5 Forking
    # moved onto the branching levels -- all stay
    # free, so no saved map changes target (the object migration reads them).
    ("internode_length", "Flower Spacing", "Multiply each flowering node's internode "
     "length -- the inflorescence's spacing. A branch's own internodes are its "
     "level's Length map", 2),
    ("gravitropism", "Gravitropism", "Multiply how strongly a branch turns toward "
     "its gravitropic setpoint (fixed at the branch's birth)", 4),
)

LEVEL_MAP_TARGETS = (
    # (id, name, description, stored number) -- numbers pinned, never reused.
    ("pitch", "Pitch", "Multiply each branch's angle from its parent (per branch, "
     "read where it leaves the parent)", 0),
    ("yaw", "Yaw", "Multiply each branch's sideways turn at launch (per branch)", 1),
    ("roll", "Roll", "Multiply each branch's twist at launch (per branch)", 2),
    # 3 was Steps: its maps became Growth % maps (Steps is level-wide now).
    ("growth", "Growth", "Multiply how much of its Steps each branch builds (per "
     "branch, read where it leaves the parent; at most 100%)", 3),
    ("start_length", "Start Length", "Multiply each branch's first internode (per "
     "branch, so the whole branch scales)", 4),
    ("length", "Length", "Multiply each internode of this level (per node)", 5),
    ("start_width", "Start Width", "Multiply each branch's base width (per branch, so "
     "the whole branch scales)", 6),
    ("width", "Width", "Multiply the width at each node of this level (per node)", 7),
    ("spiral_pitch", "Spiral Pitch", "Multiply the bend each node takes (per node; "
     "needs Pitch / Step above 0)", 8),
    ("spiral_roll", "Spiral Roll", "Multiply how far the bend's direction turns (per "
     "node)", 9),
    ("fork_probability", "Fork Probability", "Multiply each node's chance to fork "
     "(needs a forking Growth Pattern, with Probability above 0)", 10),
    ("fork_spread", "Fork Spread", "Multiply each fork's or relay's spread (needs a "
     "forking or relaying Growth Pattern)", 11),
    ("emergence_delay", "Emergence Delay", "Multiply how many steps each branch waits "
     "before it starts growing (per branch, read where it leaves the parent; needs "
     "Emergence Delay above 0)", 12),
    ("setpoint", "Gravitropic Setpoint", "Multiply the angle from vertical each node "
     "grows toward (per node, in the world's frame: below 1 the node aims more upright, "
     "whichever way the branch faces; needs Gravitropism above 0)", 13),
)


FLOWER_MAP_TARGETS = (
    # (id, name, description, stored number) -- numbers pinned, never reused.
    ("flower_spacing", "Flower Spacing", "Multiply each flowering node's internode "
     "(per flowering node)", 0),
    ("pedicel_length", "Pedicel Length", "Multiply each flower's stalk length (per "
     "flower)", 1),
    ("pedicel_angle", "Pedicel Angle", "Multiply each flower's stalk angle (per flower)", 2),
    ("pedicel_bend", "Pedicel Bend", "Multiply each flower's stalk bend (per flower)", 3),
    ("pedicel_width", "Pedicel Width", "Multiply each flower's stalk width (per flower)", 4),
    ("flower_pitch", "Flower Pitch", "Multiply each flower's facing (per flower)", 5),
    ("infl_branch_angle", "Branch Angle", "Multiply each inflorescence branch's angle "
     "(per branch)", 6),
    ("infl_branch_spacing", "Branch Spacing", "Multiply each rachis node's internode "
     "(per node)", 7),
    ("infl_branch_nodes", "Branch Size", "Multiply each inflorescence branch's node "
     "count (per branch)", 8),
    ("infl_branch_length", "Branch Length", "Multiply each inflorescence branch's "
     "internodes (per branch)", 9),
    ("infl_branch_width", "Branch Width", "Multiply each inflorescence branch's width "
     "(per branch, at most its stem's)", 10),
    ("infl_branch_bend", "Branch Bend", "Multiply each inflorescence branch's bend "
     "(per branch)", 11),
    ("infl_branch_droop", "Branch Droop", "Multiply each inflorescence branch's droop "
     "(per branch)", 12),
)


MAP_PRESET_ITEMS = (
    # Pinned numbers: the enum stores the index.
    ('CURVE', "Custom Curve", "A curve from any driver", 0),
    ('RANGE', "Random Range", "A random value per element within plus or minus Range", 1),
    ('HEIGHT', "Along Height", "Changes from the bottom of the plant to the top", 2),
    ('AGE', "By Age", "Changes with when the element was made", 3),
)

_MAP_COMMON = dict(
    uid=IntProperty(default=0, options={'HIDDEN'}),
    enabled=BoolProperty(name="Enabled", default=True, update=_changed,
                         description="Apply this map. Off keeps it without effect"),
    channel=EnumProperty(name="Driver", items=MASK_CHANNEL_ITEMS, default="height",
                         update=_changed,
                         description="The per-point attribute this map reads -- the "
                         "curve's X axis"),
    amount=FloatProperty(name="Amount", default=1.0, min=0.0, soft_max=1.0,
                         subtype='FACTOR', update=_changed,
                         description="How much of the curve applies: 0 is no effect, "
                         "1 is the curve in full"),
    # Parameter mapping: the preset a map was
    # made from (its name in the Mapping list), a Random Range's spread, and the
    # map's own seed (0: the draws it always had).
    preset=EnumProperty(name="Preset", items=MAP_PRESET_ITEMS, default='CURVE',
                        update=_map_preset_changed,
                        description="What this map was made as"),
    range=FloatProperty(name="Range", default=0.2, min=0.0, soft_max=1.0,
                        update=_map_range_changed,
                        description="Random Range: values from (1 - Range) to (1 + "
                        "Range) times the parameter (an angle: plus or minus Range "
                        "degrees)"),
    seed=IntProperty(name="Seed", default=0, min=0, update=_changed,
                     description="This map's own seed, mixed into its Random driver "
                     "(0: the draws it always had)"),
)


class PLANTGEN_PG_instance_map(PropertyGroup):
    """One instance map on a component: Driver channel -> curve -> Size or a
    rotation, per rendered instance."""
    __annotations__ = {
        **_MAP_COMMON,
        **_channel_ref_props(),
        "target": EnumProperty(name="Target", items=INSTANCE_MAP_TARGETS,
                               default="SIZE", update=_changed),
        "angle": FloatProperty(name="Angle", default=45.0, min=0.0, soft_max=180.0,
                               update=_changed,
                               description="The largest rotation, in degrees, at the "
                               "curve's ends (0 or 1); its middle (0.5) is none"),
    }


class PLANTGEN_PG_flower_map(PropertyGroup):
    """One inflorescence map on a flower component: Driver channel -> curve ->
    one of its inflorescence's parameters, per flower / branch / flowering node
    as it grows. Its curve is `fmap<uid>` under the component."""
    __annotations__ = {
        **_MAP_COMMON,
        **_channel_ref_props(),
        "target": EnumProperty(name="Target", items=FLOWER_MAP_TARGETS,
                               default="pedicel_length", update=_changed),
    }


class PLANTGEN_PG_growth_map(PropertyGroup):
    """One growth map on the plant: Driver channel -> curve -> a growth parameter,
    per bud / node / branch as the plant grows."""
    __annotations__ = {
        **_MAP_COMMON,
        **_channel_ref_props(),
        "target": EnumProperty(name="Target", items=GROWTH_MAP_TARGETS,
                               default="internode_length", update=_changed),
    }


class PLANTGEN_PG_level_map(PropertyGroup):
    """One map on a branching level: Driver channel -> curve -> one of the
    level's parameters, per branch / node / fork as the plant grows. Its curve
    is `lmap<uid>` under the growth-map owner; uids are unique over all levels."""
    __annotations__ = {
        **_MAP_COMMON,
        **_channel_ref_props(),
        "target": EnumProperty(name="Target", items=LEVEL_MAP_TARGETS,
                               default="pitch", update=_changed),
    }


class PLANTGEN_PG_param_map(PropertyGroup):
    """A map on any parameter without a per-element evaluator of its own:
    evaluated once per plant (a seeded Random draw through its curve) where
    the panel becomes the model (`mapping.plant_value`). Its curve is
    `pmap<uid>` under the growth-map owner."""
    __annotations__ = {
        **_MAP_COMMON,
        **_channel_ref_props(),
        "owner": EnumProperty(name="Owner", items=(
            ('PLANT', "Plant", "", 0), ('LEVEL', "Level", "", 1),
            ('COMPONENT', "Component", "", 2)), default='PLANT'),
        "owner_uid": IntProperty(default=0),
        "param": StringProperty(default=""),
        "angle": FloatProperty(name="Angle", default=10.0, min=0.0, soft_max=180.0,
                               update=_changed,
                               description="An angle's map: the largest offset in "
                               "degrees, at the curve's ends (its middle, 0.5, adds none)"),
    }


class PLANTGEN_PG_texture_row(PropertyGroup):
    """One row of Texturing's Materials list (`ui_materials.rebuild_rows`): a
    branching level, a leaf / bract / flower / bud / fruit component, or one
    material that component's meshes wear. UI state, rebuilt from the plant."""
    kind: StringProperty()        # LEVEL / COMPONENT / MATERIAL
    uid: IntProperty()            # the level's or component's uid
    material: PointerProperty(type=bpy.types.Material)
    count: IntProperty()          # a material row: how many meshes wear it


class PLANTGEN_PG_map_row(PropertyGroup):
    """One row of the Mapping list (`mapping.rebuild_rows`): an owner's header, or
    one map -- which store it lives in (`kind`) and where."""
    kind: StringProperty()        # OWNER / VAR / LMAP / GMAP / IMAP / PMAP
    owner: StringProperty()       # PLANT / LEVEL / COMPONENT
    owner_uid: IntProperty()
    param: StringProperty()
    index: IntProperty(default=-1)
    guides: StringProperty()      # the tree line: 'T' a branch, 'L' the last
    # A Random Range on a +/- value has no map of its own to switch off: its
    # box switches the value to 0 and back (`mapping.set_var_enabled`).
    enabled: BoolProperty(name="Enabled", default=True, update=_map_row_enabled_changed,
                          description="Use this map")


class PLANTGEN_PG_var_mute(PropertyGroup):
    """A Random Range (+/- value) switched off in the Mapping list: its value,
    kept while the +/- itself is 0, to come back when it is switched on."""
    owner: StringProperty()
    owner_uid: IntProperty()
    param: StringProperty()
    value: FloatProperty(name="Range", description="The range it comes back with")
    # A switched-off profile curve (`s.curve_mutes`): its points and handles,
    # "x,y,HANDLE;..." -- the curve itself is flat at 1 meanwhile.
    points: StringProperty()


class PLANTGEN_PG_map_fold(PropertyGroup):
    """Whether an owner's maps are shown in the Mapping list."""
    key: StringProperty()
    expanded: BoolProperty(name="Expanded", default=True, update=_map_folds_changed)


# Component types, grouped in the list and (for now) only 'leaf' is wired through
# the uniform instancing seam; the rest arrive in later Phase 2 slices.
COMPONENT_TYPE_ITEMS = (
    ("leaf", "Leaf", "Foliage leaves"),
    ("bract", "Bract", "Bracts (leaves on flowering nodes)"),
    ("flower", "Flower", "Flowers"),
    ("bud", "Bud", "Buds and apices"),
    ("fruit", "Fruit", "Fruit"),
)


class PLANTGEN_PG_placement(PropertyGroup):
    """A component placed on a branching level (the Plant Structure tree): the
    component takes the organs of that level's stems or branches. `level_uid`
    0 is the Plant row -- every level."""

    level_uid: IntProperty(default=0)
    comp_uid: IntProperty(default=0)


class PLANTGEN_PG_tree_row(PropertyGroup):
    """One row of the Plant Structure list, rebuilt from the levels and
    placements (`structure.rebuild_tree`): what it shows and its guide lines."""

    kind: StringProperty()        # PLANT / LEVEL / COMPONENT
    ref: IntProperty()            # the level index / the placement index
    guides: StringProperty()      # per indent column: '|' line, 'T' branch, 'L' last


class PLANTGEN_PG_level_link(PropertyGroup):
    """A component's link to one branching level, by the level's stable uid."""

    uid: IntProperty(default=0)


# The Traits sheet (`growth/traits.py`, the Traits panel): per component, a row
# per mesh of its collection and a column per point channel. Each cell is that
# mesh's preference for the channel; the pick stays random, weighted by them.

class PLANTGEN_PG_trait_column(PropertyGroup):
    """One column of a component's Traits sheet: the point channel its cells
    weigh the meshes by."""
    __annotations__ = {
        "channel": EnumProperty(
            name="Property", items=MASK_CHANNEL_ITEMS, default="age", update=_changed,
            description="The per-point value this column's preferences read"),
        **_channel_ref_props(),
    }


class PLANTGEN_PG_trait_cell(PropertyGroup):
    value: FloatProperty(
        name="Preference", default=0.0, min=-1.0, max=1.0, update=_changed,
        description="How this mesh leans on the column's property: +1 its chance "
        "follows the property (taken where it is high), -1 the opposite (where it is "
        "low), 0 no effect; between, a blend")


class PLANTGEN_PG_trait_row(PropertyGroup):
    """One mesh of the component's collection in its Traits sheet."""
    asset: PointerProperty(type=bpy.types.Object, name="Mesh")
    weight: FloatProperty(
        name="Weight", default=1.0, min=0.0, soft_max=4.0, update=_changed,
        description="How common this mesh is overall, relative to the others in the "
        "collection: 2 is twice as likely as 1, 0 never (unless no mesh fits a point)")
    cells: CollectionProperty(type=PLANTGEN_PG_trait_cell)


class PLANTGEN_PG_component(PropertyGroup):
    """One component instance: a collection
    to instance, of a given `type`, filtered by its own stack of attribute windows.
    Multiple of a type may coexist -- each claims the candidate points its filters
    keep (first-claim, in list order), and the bridge routes them all through one
    master collection + one Instance-on-Points. v1 wires the leaf type; the others
    follow.

    The filters are the point-mask stack, now inline on the component rather than a
    separate reusable Maps layer -- one flat list, no gate-by-name indirection. Add
    a Branch Depth / Height / ... window and the component claims only the points it
    keeps; `prune` thins what survives at random."""

    type: EnumProperty(
        name="Type", items=COMPONENT_TYPE_ITEMS, default="leaf", update=_changed,
        description="Which kind of organ this component instances onto")
    # Before the Plant Structure placements (object branch-levels schema < 6):
    # read once by the migration, then unused.
    all_levels: BoolProperty(default=True, options={'HIDDEN'})
    level_links: CollectionProperty(type=PLANTGEN_PG_level_link)
    name_override: StringProperty(
        name="Name", default="",
        description="The component's own name; empty uses its collection's")
    label: StringProperty(
        name="Name", get=_component_label_get, set=_component_label_set,
        description="The collection's name, unless renamed (double-click or F2). "
        "Clear it to go back to the collection's name")
    placement: EnumProperty(
        name="Placement",
        items=[
            ("ORGAN", "On organs", "Instance onto the organ points growth already "
             "makes -- leaves at nodes, flowers on the inflorescence, etc.", 'NONE', 1),
            ("NODE", "Nodes", "At every phytomer node, whether or not growth put an "
             "organ there", 'NONE', 8),
            ("INTERNODE", "Internodes", "Spaced along the internodes between nodes -- "
             "leaflets, thorns, moss", 'NONE', 2),
            ("TIP", "Terminal", "At the apex of every axis", 'NONE', 16),
            ("INTERSECTION", "Intersections", "At every branch/fork junction", 'NONE', 32),
            ("EXTRA", "Random", "A random count over the whole plant's stems, "
             "unstructured (vs the even arrangement of the others)", 'NONE', 4),
            ("BASAL", "Basal", "A rosette at the base of every stem -- basal leaves, "
             "turning by the Phyllotaxis below, Basal Count of them per stem", 'NONE', 64),
        ],
        default={"ORGAN"}, options={'ENUM_FLAG'}, update=_changed,
        description="Where this component's instances go -- several at once: on grown "
        "organs, at nodes/terminals/junctions, along the stems, and/or scattered at random")
    internode_count: IntProperty(
        name="Instances per Internode", default=3, min=1, soft_max=12,
        update=_changed,
        description="Internodes placement: how many instances to space evenly along "
        "each internode")
    extra_count: FloatProperty(
        name="Random Count", default=20.0, min=0.0, soft_max=200.0, update=_changed,
        description="Random placement: how many instances to scatter at random over "
        "the whole plant's stems")
    basal_count: IntProperty(
        name="Basal Count", default=8, min=1, soft_max=40, update=_changed,
        description="Basal placement: how many leaves in the rosette at the base of each "
        "stem (each turning by the Phyllotaxis spiral; a whorl puts that many at each)")
    petiole_length: FloatProperty(
        name="Petiole Length", default=0.0, min=0.0, soft_max=0.3, subtype='DISTANCE',
        unit='LENGTH', precision=4, update=_changed,
        description="A stalk this long between the stem and each of this component's "
        "leaves (grown or placed): the leaf sits at its end. Basal leaves usually take a "
        "longer one than the leaves up the stem -- give them their own component. 0: none")
    petiole_width: FloatProperty(
        name="Petiole Width", default=0.0008, min=0.00005, soft_max=0.01, subtype='DISTANCE',
        unit='LENGTH', precision=5, update=_changed,
        description="The petiole's radius")
    petiole_length_variation: FloatProperty(
        name="Petiole Length Variation", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="How much each petiole's length varies, as a fraction, drawn per leaf")
    petiole_angle_variation: FloatProperty(
        name="Petiole Angle Variation", default=0.0, min=0.0, max=90.0, update=_changed,
        description="Random tilt on each petiole where it leaves the stem, in degrees, in a "
        "random direction round it -- up, down or to either side -- drawn per leaf")
    petiole_bend: FloatProperty(
        name="Petiole Bend", default=0.0, min=-3.14159, max=3.14159, subtype='ANGLE',
        update=_changed,
        description="How far the petiole curves along its length, toward the leaf's face "
        "(up when the leaf faces the sky); negative curves it the other way")
    petiole_bend_variation: FloatProperty(
        name="Petiole Bend Variation", default=0.0, min=0.0, max=90.0, update=_changed,
        description="Degrees each petiole's Bend randomly deviates either way, per leaf")
    petiole_droop: FloatProperty(
        name="Petiole Droop", default=0.0, min=0.0, max=3.14159, subtype='ANGLE',
        update=_changed,
        description="How far the petiole sags toward the ground along its length (never "
        "past straight down) -- a long stalk carrying its leaf low")
    petiole_droop_variation: FloatProperty(
        name="Petiole Droop Variation", default=0.0, min=0.0, max=90.0, update=_changed,
        description="Degrees each petiole's Droop randomly deviates either way, per leaf")
    # Collision (geometry/leaf_collision.py): this component's leaves turn out of
    # the stems and the older leaves, then are pruned if still touching.
    collide: BoolProperty(
        name="Avoid Collisions", default=False, update=_changed,
        description="Turn this component's leaves out of the stems and out of other "
        "leaves: the younger leaf yields, turned about where it attaches (with its "
        "petiole) by the least that clears it. Off: its leaves stay put, an obstacle "
        "the others avoid")
    collide_max_turn: FloatProperty(
        name="Max Turn", default=30.0, min=0.0, soft_max=60.0, max=90.0, update=_changed,
        description="The most degrees a leaf is turned to clear a collision (tried in "
        "5 degree steps, sideways, up / down and rolled)")
    collide_prune: BoolProperty(
        name="Prune the Rest", default=True, update=_changed,
        description="Remove a leaf no turn up to Max Turn clears. Off: it keeps the pose "
        "that touches least")
    # Arrangement (phyllotaxis) of the generated points around/along the stem. A
    # preset seeds the three controls; the golden-angle default is Alternate.
    gen_phyllotaxis: EnumProperty(
        name="Phyllotaxis",
        items=[
            ("CUSTOM", "Custom", "Spiral and whorl set by hand below"),
            ("ALTERNATE", "Alternate (spiral)", "One instance per node on a golden "
             "spiral (137.5 deg)"),
            ("DISTICHOUS", "Distichous", "One per node, alternating 180 deg -- two ranks"),
            ("OPPOSITE", "Opposite", "A pair at each node, aligned up the stem"),
            ("DECUSSATE", "Decussate", "A pair at each node, each rotated 90 deg from "
             "the last"),
            ("WHORLED", "Whorled", "Three per node, evenly spaced"),
        ],
        default="ALTERNATE", update=_apply_phyllo_preset,
        description="Phyllotaxis preset -- sets the spiral and whorl below. Reads "
        "'Custom' once you tune them by hand")
    gen_divergence: FloatProperty(
        name="Phyllotaxis Spiral", default=2.39996323, min=0.0, max=6.2831853,
        subtype='ANGLE', update=_phyllo_value_changed,
        description="Divergence angle -- how far each successive generated point turns "
        "around the stem from the last. 137.5 deg is the natural golden-spiral default")
    gen_whorl_count: IntProperty(
        name="Whorl", default=1, min=1, soft_max=8, update=_phyllo_value_changed,
        description="How many instances sit together at each position, spaced around "
        "the stem (a whorl). 1 is a single spiral")
    gen_whorl_angle: FloatProperty(
        name="Whorl Spacing", default=0.0, min=0.0, max=6.2831853, subtype='ANGLE',
        update=_phyllo_value_changed,
        description="Angle between the members of one whorl. 0 spreads them evenly "
        "(360 / whorl); set it to cluster them tighter or wider")
    gen_pitch: FloatProperty(
        name="Pitch", default=0.0, min=-90.0, max=90.0, subtype='ANGLE', update=_changed,
        description="Generated placement: tilt each instance up from straight-out "
        "toward the stem tip. 0 points straight out, positive angles it up the stem")
    gen_roll_self: FloatProperty(
        name="Roll Self", default=0.0, min=-3.14159, max=3.14159, subtype='ANGLE',
        update=_changed,
        description="Generated placement: spin each instance about its own length. On a "
        "leaning branch, positive turns the face toward the sky on both sides")
    gen_roll_parent: FloatProperty(
        name="Parent Roll", default=0.0, min=-3.14159, max=3.14159, subtype='ANGLE',
        update=_changed,
        description="Generated placement: turn each instance around the parent stem's "
        "axis, every one the same way round")
    gen_fold: FloatProperty(
        name="Fold", default=0.0, min=-3.14159, max=3.14159, subtype='ANGLE',
        update=_changed,
        description="Generated placement: raise or lower each instance around the parent "
        "stem's axis. On a leaning branch positive raises it toward the sky on both "
        "sides, so the two leaves of a pair lift together")
    seed: IntProperty(
        name="Seed", default=0, min=0, update=_changed,
        description="This component's own seed, on top of the plant's master seed: "
        "reshuffles only its random choices -- which asset each point gets, Prune, size "
        "and angle variation, its masks' Random channel, where it scatters. 0 follows the "
        "master seed alone")
    emerging_curl: FloatProperty(
        name="Emerging Curl", default=0.0, min=-6.2831853, max=6.2831853,
        soft_min=-3.14159, soft_max=3.14159, subtype='ANGLE', update=_changed,
        description="How far a young leaf is rolled up along its length as it appears "
        "(tip curled in toward its face; negative curls it back), uncurling as it "
        "grows until it is flat at full size. 0 = appears flat")
    emerging_midrib_fold: FloatProperty(
        name="Emerging Midrib Fold", default=0.0, min=0.0, max=1.5707963,
        subtype='ANGLE', update=_changed,
        description="How far a young leaf is folded shut along its midrib as it appears "
        "(90 deg = the halves closed together), opening as it grows until it is flat at "
        "full size. 0 = appears open")
    sky_roll: FloatProperty(
        name="Sky", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Turn each leaf about its own length toward facing the sky -- 1 "
        "twists every face as close to straight up as it can get, without changing "
        "where the leaf points (the leaves twisting on their stalks to face the light)")
    gen_fold_variance: FloatProperty(
        name="Fold Variation", default=0.0, min=0.0, soft_max=90.0, update=_changed,
        description="Generated placement: degrees each instance's Fold randomly "
        "deviates either way, seeded per instance")
    # Per-instance orientation scatter, in degrees (like the leaf panel's yaw/roll
    # variation), seeded per instance and applied on top of each base angle above.
    gen_pitch_variance: FloatProperty(
        name="Pitch Variation", default=0.0, min=0.0, soft_max=90.0, update=_changed,
        description="Generated placement: degrees each instance's Pitch randomly "
        "deviates either way, seeded per instance")
    gen_roll_self_variance: FloatProperty(
        name="Roll Self Variation", default=0.0, min=0.0, soft_max=90.0, update=_changed,
        description="Generated placement: degrees each instance's Roll Self randomly "
        "deviates either way, seeded per instance")
    gen_roll_parent_variance: FloatProperty(
        name="Parent Roll Variation", default=0.0, min=0.0, soft_max=90.0, update=_changed,
        description="Generated placement: degrees each instance's Parent Roll randomly "
        "deviates either way, seeded per instance")
    collection: PointerProperty(
        type=Collection, name="Collection", update=_changed,
        description="The collection of assets to instance; one is picked at random "
        "per point (seeded)")
    catkin_material: PointerProperty(
        type=Material, name="Material", update=_changed,
        description="Catkins are swept into the stem mesh, so they wear their twig's "
        "material; this one is put on the catkins instead (a hazel's yellow lamb's tails "
        "on brown twigs). None: the twig's")
    size: FloatProperty(
        name="Size", default=1.0, min=0.0, soft_max=4.0, subtype='FACTOR',
        update=_changed,
        description="Scales this component's instances on top of the organ's own size. "
        "On a Flower it is the flower size; on a Bract (On Organs) it is the bract's "
        "size as a fraction of the foliage leaf its flowering node would have carried "
        "- full-sized on dead-nettle, scale-like on salvia - so bracts follow the Leaf "
        "Size curve up the stem. On the first Fruit component (On Organs) it is the "
        "fruit size, and fruit weigh with its cube. On a Bud it is the bud size")
    gen_size_variance: FloatProperty(
        name="Size Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Seeded per-instance random size, on grown and placed points alike. "
        "0.3 varies each instance +/-30%; 0 makes them all the same size")
    # A stable per-component id, so its own size curve (keyed by this) follows it
    # across reorders and removals. 0 = not yet assigned; backfilled at regrow.
    uid: IntProperty(default=0, options={'HIDDEN'})
    # Flower components own the inflorescence spec (docs/flower-component-port-plan
    # .md). Until this is True the component's flower fields are seeded from the
    # plant at regrow, so an existing/new flower component inherits the plant's
    # inflorescence rather than resetting it to component defaults.
    flower_authored: BoolProperty(default=False, options={'HIDDEN'})
    leaf_weight: FloatProperty(
        name="Leaf Weight", default=8.0e-5, min=0.0, soft_max=2.0e-3, precision=5,
        update=_leaf_weight_changed,
        description="How heavy one full-size leaf of this component is, on the same "
        "relative scale as Flower Weight. It loads the stems for the grown leaves this "
        "component owns (the first leaf component placed On Organs); leaves placed at "
        "nodes, internodes or the base are drawn after the bend and weigh nothing")
    leaf_weight_authored: BoolProperty(default=False, options={'HIDDEN'})
    # The same once-only seed guard for a fruit component (Fruit Set, Weight, and
    # Size = the plant's old Fruit Size), cleared by a preset apply so it re-seeds.
    fruit_authored: BoolProperty(default=False, options={'HIDDEN'})
    # Flower Heliotropism moved onto the flower component after `flower_authored`
    # had already seeded most of them: this seeds just it, once, from the plant's.
    heliotropism_authored: BoolProperty(default=False, options={'HIDDEN'})
    sun_object: PointerProperty(
        type=bpy.types.Object, update=_changed, name="Sun Object",
        description="The object this component's flowers turn to face (Heliotropism). They "
        "aim at wherever it sits relative to the plant, so move it to place the sun -- any "
        "object works (an empty, a lamp, a mesh). Empty means its flowers do not turn")
    # The same once-only seed: the plant's Sun Object, which flowers tracked
    # before each flower component had its own.
    sun_authored: BoolProperty(default=False, options={'HIDDEN'})
    mute: BoolProperty(
        name="Mute", default=False, update=_changed,
        description="Temporarily skip this component -- it claims no points")
    own_inflorescence: BoolProperty(
        name="Own Inflorescence", default=True, update=_changed,
        description="Grow an inflorescence of its own on the levels it is placed on "
        "(its Type, Attachment Location, spacing, Size...). Off: a variant -- it draws "
        "on the flowers the inflorescences on its levels grow, shared with them by "
        "Density. Where two inflorescences want the same tip or node position, the "
        "one higher in the list has it. The first flower component always has its own")
    color: FloatVectorProperty(
        name="Colour", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.20, 0.70, 0.25), update=_swatch_changed,
        description="A label colour for this component -- shown in the list, painted "
        "on its organs by the Component Colours view, and used as the 'kept' colour in "
        "this component's filter preview")
    filters: CollectionProperty(type=PLANTGEN_PG_mask_entry)
    filters_active: IntProperty(default=0, update=_component_filter_active_changed)
    # Instance maps: per-point Size / rotation driven by a channel through a curve.
    maps: CollectionProperty(type=PLANTGEN_PG_instance_map)
    maps_active: IntProperty(default=0)
    # A flower component's inflorescence maps: per flower / branch / flowering
    # node, as its inflorescence grows (`Inflorescence.maps`).
    growth_maps: CollectionProperty(type=PLANTGEN_PG_flower_map)
    density: FloatProperty(
        name="Density", default=1.0, min=0.0, soft_max=4.0, update=_changed,
        description="Relative weight for sharing points with other components of the "
        "same type. Where several are eligible for a point, one is picked at random "
        "in proportion to their weights -- two at 1 mix 50/50, 2 vs 1 leans 2:1")
    prune: FloatProperty(
        name="Prune", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Fraction of this component's instances to delete at random "
        "(seeded). 0 deletes none, 1 deletes all")
    # Traits: which mesh of the collection each point takes (empty: uniform).
    trait_columns: CollectionProperty(type=PLANTGEN_PG_trait_column)
    trait_rows: CollectionProperty(type=PLANTGEN_PG_trait_row)


# ---------------------------------------------------------------------------
# Branching levels: one full set of branching
# controls per level of the stack -- level 0 the stems, each next level the
# branches on the one before. Mirrors `growth.morphology.BranchLevel` field for
# field (angles in degrees, as the growth side keeps them).
# ---------------------------------------------------------------------------

_LEVEL_PHYLLO_ITEMS = (
    ("CUSTOM", "Custom", "Spiral and whorl set by hand"),
    ("ALTERNATE", "Alternate (spiral)", "One per node on a golden spiral (137.5 deg)"),
    ("DISTICHOUS", "Distichous", "One per node, alternating 180 deg -- two ranks"),
    ("OPPOSITE", "Opposite", "A pair at each node, stacked up the stem"),
    ("DECUSSATE", "Decussate", "A pair at each node, each turned 90 deg from the last"),
    ("WHORLED", "Whorled", "Three per node, each whorl turned half a step"),
)
_applying_level_phyllo = False


def _level_phyllo_preset(self, context):
    """A named pattern writes the spiral / whorl / whorl angle (guarded, so those
    writes do not flip the dropdown back to Custom); Custom writes nothing."""
    global _applying_level_phyllo
    from ..growth.morphology import LevelPhyllotaxis
    vals = LevelPhyllotaxis.PRESETS.get(self.phyllotaxis)
    if vals is not None:
        _applying_level_phyllo = True
        try:
            self.spiral_angle, self.whorl, self.whorl_angle = vals
        finally:
            _applying_level_phyllo = False
    _changed(self, context)


def _level_phyllo_value(self, context):
    """Tuning the spiral / whorl by hand marks the pattern Custom."""
    if _applying_level_phyllo:
        return
    if self.phyllotaxis != "CUSTOM":
        self.phyllotaxis = "CUSTOM"   # fires _level_phyllo_preset(CUSTOM) -> _changed
    else:
        _changed(self, context)


class PLANTGEN_PG_branch_level(PropertyGroup):
    """One level of the branching stack."""

    name: StringProperty(
        name="Name", default="Level",
        description="The level's name in the Components tree (double-click or F2)")
    parent_uid: IntProperty(
        default=0, options={'HIDDEN'},
        description="The level this one grows on (its uid); 0: a stems level")
    uid: IntProperty(
        default=0, options={'HIDDEN'},
        description="Stable id the components link to (survives moving the level)")
    expanded: BoolProperty(update=_structure_changed,
        name="Expanded", default=True,
        description="Show this level's components and the levels on it in the tree")
    color: FloatVectorProperty(
        name="Colour", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.53, 0.70, 0.37), update=_swatch_changed,
        description="A label colour for this level: its row in the Components tree and "
        "the maps on its parameters")
    color_authored: BoolProperty(
        default=False, options={'HIDDEN'},
        description="The colour was set (else it takes its depth's default once)")
    show: BoolProperty(
        name="Show", default=True, update=_changed,
        description=(
            "Build this level's stems or branches. Off hides just them: the "
            "branches, leaves and flowers growing on them stay where they are. "
            "View-only: the plant grows the same"))
    phyllotaxis: EnumProperty(
        name="Phyllotaxis", items=_LEVEL_PHYLLO_ITEMS, default="ALTERNATE",
        update=_level_phyllo_preset,
        description="How this level's branches are arranged round their parent -- sets "
        "Spiral, Whorl and Whorl Angle. On the stems level the stems fill the base as a "
        "phyllotactic disc, each turned the golden angle (or Whorl Angle) from the last")
    spiral_angle: FloatProperty(
        name="Spiral Angle", default=137.5, min=0.0, max=360.0, update=_level_phyllo_value,
        description="Degrees each node turns round the parent from the last")
    whorl: IntProperty(
        name="Whorl", default=1, min=1, soft_max=8, update=_level_phyllo_value,
        description="Branches per node, spaced round the parent. On the stems level: "
        "how many stems")
    whorl_angle: FloatProperty(
        name="Whorl Angle", default=0.0, min=0.0, max=360.0, update=_level_phyllo_value,
        description="Degrees between the members of one whorl. 0 spreads them evenly")
    slide: FloatProperty(
        name="Slide", default=0.0, min=-1.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Slide a node's branches along the stem: + up, - down. A pair's "
        "members move apart, the first up and the second down, half an internode apart "
        "at 1")
    slide_variation: FloatProperty(
        name="Slide Variation", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random slide per node, added to Slide: some pairs slide up, some down")
    start_pitch: FloatProperty(
        name="Pitch", default=45.0, min=-180.0, max=180.0, update=_changed,
        description="Degrees the branch leaves its parent at. On the stems level: how "
        "far the stems splay from vertical")
    start_yaw: FloatProperty(
        name="Yaw", default=0.0, min=-180.0, max=180.0, update=_changed,
        description="Degrees the branch turns sideways as it leaves its parent")
    start_roll: FloatProperty(
        name="Roll", default=0.0, min=-180.0, max=180.0, update=_changed,
        description="Degrees the branch twists about its own axis as it leaves -- where "
        "its own branches start round it")
    upper_side_up: BoolProperty(
        name="Upper Side Up", default=False, update=_changed,
        description="Give each branch a top and an underside: its own branches are laid "
        "out from its upper side (Roll turns from there, 90 = its side), the same way "
        "round on every branch of a whorl. Off: from wherever the parent's frame "
        "carries it, which differs branch to branch")
    steps: IntProperty(
        name="Steps", default=12, min=1, soft_max=60, update=_changed,
        description="Nodes the axis has when fully grown. A flowering axis flowers "
        "after them")
    growth: FloatProperty(
        name="Growth", default=100.0, min=1.0, max=100.0, subtype='PERCENTAGE',
        precision=0, update=_changed,
        description="How much of its Steps each axis of this level builds. A branch at "
        "50% is the lower half of the full one, not a smaller copy (its taper and maps "
        "stay put), the last node grown partly, and it still flowers at its cut tip. "
        "Map it to vary it per branch")
    emergence_delay: IntProperty(
        name="Emergence Delay", default=0, min=0, soft_max=30, update=_changed,
        description="Growth steps each branch of this level waits after the node it "
        "leaves forms before it starts growing (on a stems level: steps after the plant "
        "starts). Timing only: the full-grown plant is unchanged -- the plant's season "
        "lengthens to fit -- so it shows at earlier Days and in Grow Over Timeline. Map "
        "it to stagger branches")
    start_length_scale: FloatProperty(
        name="Start Length", default=1.0, min=0.0, soft_max=2.0, update=_changed,
        description="First internode, as a fraction of the parent's internode where "
        "the branch attaches")
    length_scale_per_step: FloatProperty(
        name="Length / Step", default=1.0, min=0.0, soft_max=1.5, update=_changed,
        description="Each internode is this times the one before -- below 1 shortens "
        "toward the tip")
    length_variation: FloatProperty(
        name="Length Variation", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="How much each stem or branch of this level varies in length from its "
        "siblings, as a fraction -- drawn once per stem or branch, so its internodes stay "
        "evenly stepped (on a flowering stretch, the flower spacing varies node to node)")
    start_width_scale: FloatProperty(
        name="Start Width", default=1.0, min=0.0, soft_max=2.0, update=_changed,
        description="Base width, as a fraction of the parent's width where the branch "
        "attaches")
    width_scale_per_step: FloatProperty(
        name="Width / Step", default=1.0, min=0.0, soft_max=1.5, update=_changed,
        description="Each node's width is this times the one before -- below 1 tapers")
    width_variation: FloatProperty(
        name="Width Variation", default=0.0, min=0.0, soft_max=0.5, subtype='FACTOR',
        update=_changed,
        description="How much each stem or branch of this level varies in thickness from "
        "its siblings, as a fraction. Drawn once per stem or branch, so no taper goes "
        "ragged along its length")
    start_pitch_variation: FloatProperty(
        name="Pitch Variation", default=0.0, min=0.0, soft_max=45.0, max=90.0,
        update=_changed,
        description="Degrees each stem or branch's Pitch (on the stems level, its Splay) "
        "deviates either way, drawn per stem or branch")
    start_yaw_variation: FloatProperty(
        name="Yaw Variation", default=0.0, min=0.0, soft_max=45.0, max=180.0,
        update=_changed,
        description="Degrees each stem or branch's Yaw deviates either way, drawn per stem "
        "or branch")
    start_roll_variation: FloatProperty(
        name="Roll Variation", default=0.0, min=0.0, soft_max=90.0, max=180.0,
        update=_changed,
        description="Degrees each stem or branch's Roll deviates either way, drawn per stem "
        "or branch -- where its own leaves and branches start round it")
    stiffness: FloatProperty(
        name="Stiffness", default=0.75, min=0.0, max=1.0, subtype='FACTOR',
        update=_level_stiffness_changed,
        description="How strongly this level's stems or branches resist bending under their "
        "load: a real material stiffness (Young's modulus) on a log scale, 0 is 5 MPa, 0.5 "
        "is 160 MPa, 1 is 5 GPa. Soft herbaceous stems sit around 0.1-0.4, grass culms "
        "and woody stems around 0.8-1. Thick parts already resist far more than thin ones, "
        "so raising this mostly straightens the tips")
    stiffness_authored: BoolProperty(default=False, options={'HIDDEN'})
    rig_bone_angle: FloatProperty(
        name="Bone Angle", default=0.2617993877991494, min=0.0, max=1.0471975511965976,
        subtype='ANGLE',
        description="How far this level's stems or branches may bend along one bone before "
        "the rig starts the next: straight runs share a bone, bends get more. 0 = a bone "
        "per node. Read when the rig is built")
    seed: IntProperty(
        name="Seed", default=0, min=0, update=_changed,
        description="This level's own seed, on top of the plant's: changing it redraws only "
        "this level's variation (length, width, slide, fork chance, mask survival). 0 "
        "follows the plant's seed alone")
    material: PointerProperty(
        name="Material", type=Material, update=_gn_display_changed,
        description="The material this level's stems or branches wear (none: its parent "
        "level's). Set in the node graph -- never regrows")
    cross_section: EnumProperty(
        name="Cross Section", update=_cross_section_changed,
        items=(
            ('circular', "Circular", "A round stem, the usual case"),
            ('triangular', "Triangular", "Three-sided - sedges have edges"),
            ('square', "Square", "Four-sided, diagnostic of the mint family"),
            ('hexagonal', "Hexagonal", "Six-sided"),
            ('polygonal', "Polygonal", "A regular polygon with a side count you set"),
            ('flattened', "Flattened", "Squashed across one axis, like a ribbon or a "
             "flattened grass culm"),
            ('winged', "Winged", "A few deep ridges running the length of the stem"),
            ('ribbed', "Ribbed", "Many shallow ridges, like celery or angelica"),
            ('star', "Star", "Deep sharp points"),
        ),
        default='circular',
        description="The shape swept along this level's stems or branches. A botanical "
        "trait as much as an artistic one - a square stem says mint family, a triangular "
        "one says sedge")
    profile_sides: IntProperty(
        name="Sides", default=8, min=3, max=32, update=_changed,
        description="How many flat sides a Polygonal section has")
    lobe_count: IntProperty(
        name="Count", default=5, min=2, max=32, update=_changed,
        description="How many ridges or points run up the stem")
    lobe_depth: FloatProperty(
        name="Depth", default=0.35, min=0.0, soft_max=1.0, subtype='FACTOR', update=_changed,
        description="How far the ridges stand out from the stem's own radius")
    flatten: FloatProperty(
        name="Flatten", default=0.6, min=0.0, max=0.98, subtype='FACTOR', update=_changed,
        description="How far a Flattened section is squashed. 0 is round, 1 is a ribbon")
    spiral_pitch: FloatProperty(
        name="Pitch / Step", default=0.0, min=-90.0, max=90.0, update=_changed,
        description="Degrees the growing tip bends each step")
    spiral_roll: FloatProperty(
        name="Roll / Step", default=0.0, min=-180.0, max=180.0, update=_changed,
        description="Degrees the bend's direction turns each step. With Pitch / Step, "
        "a helix")
    # Before the Growth Pattern (object branch-levels schema < 7): read once by
    # the migration (On -> Dichotomous), then unused.
    fork: BoolProperty(default=False, options={'HIDDEN'})
    growth_pattern: EnumProperty(
        name="Growth Pattern", update=_changed,
        # Numbers pinned: an EnumProperty stores the index, so the display order
        # stays free to change without shifting saved plants.
        items=(
            ('monopodial', "Monopodial", "One apex builds the whole axis; its branches "
             "come off its sides", 0),
            ('sympodial', "Sympodial", "The axis grows in units: each ends (Unit End) "
             "and one bud below it carries on -- a zig-zag chain", 1),
            ('dichasial', "Dichasial", "Each unit ends and two buds carry on: a false "
             "fork with the spent tip in every crotch", 2),
            ('pleiochasial', "Pleiochasial", "Each unit ends and a ring of Relays buds "
             "carries on", 3),
            ('pseudomonopodial_relay', "Pseudomonopodial (Aligned Relay)", "A sympodium "
             "whose relay continues in line, the spent tip pushed aside: one straight "
             "stem with side-set flowers", 4),
            ('dichotomous', "Dichotomous", "The growing tip splits into equal forks", 5),
            ('pseudomonopodial_fork', "Pseudomonopodial (Unequal Fork)", "The tip "
             "splits unequally: a leader carries on, weaker forks peel off -- a fake "
             "trunk", 6),
        ),
        default='monopodial',
        description="How this level's axes carry their growth on: one apex, relays "
        "from units that end, or a tip that splits")
    unit_steps: IntProperty(
        name="Unit Steps", default=3, min=1, soft_max=12, update=_changed,
        description="Nodes in one unit before it ends and relays. The level's Steps "
        "is the whole chain's length")
    relay_count: IntProperty(
        name="Relays", default=3, min=3, soft_max=6, update=_changed,
        description="Buds that carry on from each spent unit")
    fork_balance: FloatProperty(
        name="Balance", default=0.5, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How unequal the fork is: 0 equal; higher keeps the leader "
        "straighter and makes the other forks shorter, thinner and wider-splayed")
    fork_branches: IntProperty(
        name="Branches", default=2, min=2, soft_max=5, update=_changed,
        description="How many continuations a fork splits into")
    fork_probability: FloatProperty(
        name="Fork Probability", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed, description="Chance, per node, that the growing tip forks there")
    fork_spread: FloatProperty(
        name="Spread", default=30.0, min=0.0, max=180.0, update=_changed,
        description="Total degrees between the continuations (forks or relays). On an "
        "Aligned Relay, twice how far the spent tip is pushed aside")
    fork_whorl: FloatProperty(
        name="Whorl", default=90.0, min=0.0, max=360.0, update=_changed,
        description="Degrees each fork / relay generation's plane turns from its "
        "parent's (180 zig-zags in one plane)")
    # Masks: which of this level's branches grow (a rejected one never does);
    # Maps: how its parameters vary per branch / node.
    filters: CollectionProperty(type=PLANTGEN_PG_mask_entry)
    filters_active: IntProperty(default=0, update=_level_filter_active_changed)
    maps: CollectionProperty(type=PLANTGEN_PG_level_map)
    maps_active: IntProperty(default=0)


def _variation_param_changed(self, context):
    """An axis picked a parameter: start it at the source plant's own value."""
    sync_variation_starts(self)


def sync_variation_starts(v) -> None:
    """Start each axis at the plant's own value of its parameter, once per
    parameter picked (`<axis>_seen` remembers which). Also run at regrow and
    before a build, so the default Day axis starts at the plant's Day rather
    than 0 (user decision 2026-10-16)."""
    obj = v.id_data
    for axis in ("x", "y"):
        ident = getattr(v, f"{axis}_param")
        if not ident or getattr(v, f"{axis}_seen") == ident:
            continue
        if ident == "plant_size":              # variations.PLANT_SIZE: a factor, 1 = as is
            v[f"{axis}_start"] = 1.0
            v[f"{axis}_seen"] = ident
        elif hasattr(obj.plantgen, ident):
            v[f"{axis}_start"] = float(getattr(obj.plantgen, ident))
            v[f"{axis}_seen"] = ident


class PLANTGEN_PG_variations(PropertyGroup):
    """A row or grid of copies of this plant (`variations.py`): each its own
    seed, one or two plant-wide parameters stepped across them. Settings only --
    nothing here changes the plant itself."""

    layout: EnumProperty(
        name="Layout", default='ROW',
        items=[('ROW', "Row", "One row of plants, one parameter stepped along it", 0),
               ('GRID', "Grid", "Rows of rows: a second parameter (or only the seeds) "
                "varies from row to row", 1)])
    count_x: IntProperty(name="Plants", default=10, min=1, soft_max=24,
                         description="Plants in a row")
    count_y: IntProperty(name="Rows", default=3, min=1, soft_max=12,
                         description="Rows in the grid")
    seed_step: IntProperty(
        name="Seed Step", default=100, min=1,
        description="The gap between the plants' seeds: the plant's Seed + this, + twice "
        "this, ... (Step 100 on Seed 0: 100, 200, 300...). Change it for a fresh set")
    x_param: StringProperty(name="Along Rows", default="day", update=_variation_param_changed,
                            description="The plant-wide parameter stepped along each row")
    x_start: FloatProperty(name="Start", default=0.0,
                           description="Its value at the first plant of a row")
    # -18: with Start at the plant's own Day, a new plant's default row is a
    # growth series back from it (270 -> 108); +36 ran it into the winter,
    # where a perennial shows nothing (user decision 2026-10-18)
    x_step: FloatProperty(name="Step", default=-18.0,
                          description="Added at each next plant along a row")
    x_seen: StringProperty(default="", options={'HIDDEN'})   # "": Day not yet synced
    y_param: StringProperty(name="Across Rows", update=_variation_param_changed,
                            description="The plant-wide parameter stepped from row to row "
                            "(none: the rows differ only by seed)")
    y_start: FloatProperty(name="Start", default=0.0,
                           description="Its value in the first row")
    y_step: FloatProperty(name="Step", default=1.0,
                          description="Added at each next row")
    y_seen: StringProperty(options={'HIDDEN'})
    auto_spacing: BoolProperty(
        name="Auto Spacing", default=True,
        description="Space plants by the widest one's footprint, a quarter to spare")
    spacing_x: FloatProperty(name="Spacing X", default=1.0, min=0.0, unit='LENGTH',
                             description="Distance between plants in a row")
    spacing_y: FloatProperty(name="Spacing Y", default=1.0, min=0.0, unit='LENGTH',
                             description="Distance between rows")
    labels: BoolProperty(name="Labels", default=True,
                         description="A caption under each plant: its values and seed")
    weld: BoolProperty(
        name="Weld", default=False,
        description="Weld every plant's junctions as it is built (slow). Off: the quick "
        "unwelded preview; Bake welds anyway")


class PLANTGEN_PG_growth(PropertyGroup):
    checks: CollectionProperty(type=PLANTGEN_PG_check)
    checks_ran: BoolProperty(default=False)

    debug_view_mode: EnumProperty(
        name="Debug View", items=DEBUG_VIEW_MODES, default='off',
        description="Colour the plant by one of its own attributes, as a viewport "
        "overlay only - never exported or baked",
        update=_debug_view_mode_update,
    )

    # -- Senescence (WS3): off by default; the curve is the Age Senescence
    # widget (senescence_onset), these two are the rest of the box.
    senescence_dead_color: FloatVectorProperty(
        name="Dead Colour", subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.35, 0.28, 0.20, 1.0), update=_gn_display_changed,
        description="What dead tissue is tinted toward. A shader sampling the Color "
        "vertex attribute shows it in a render; the Senescence debug view shows it in "
        "the viewport regardless of material",
    )
    senescence_retain_dead_wood: BoolProperty(
        name="Keep Dead Wood", default=True, update=_changed,
        description="Whether a dead branch stays as grey dead wood, or is removed "
        "entirely. Its leaves and flowers are shed either way -- a dead branch is bare",
    )
    senescence_shade_threshold: FloatProperty(
        name="Shade Death", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="After growth, each branch is judged by the light at its base; "
        "below this it dies and takes its subtree with it. 0 (default) never kills. "
        "The plant shades itself, so this works with no obstacle in the scene; add "
        "obstacles overhead to shade a lower crown. 1.0 kills anything out of full sky",
    )

    # -- The inspector overlay ------------------------------------------
    # Two controls, both categorical, both drawn as one row: a plant has
    # hundreds of components and drawing every one at once is the thing that
    # makes an overlay unreadable, so what is shown and how it is coloured are
    # the whole of its parameter surface.
    #
    # Deliberately NOT folded into `debug_view_mode`. That colours the *real
    # geometry* (the node graph's debug branch), and its modes (vertex
    # colours, curvature, carried load) mean nothing on a glyph; these modes
    # colour things that have no geometry to tint, which is exactly what no
    # geometry colouring can do. Different surfaces, different questions.
    inspect_show: EnumProperty(
        name="Show", items=overlay.KIND_ITEMS, default={'BUD', 'APEX'},
        options={'ENUM_FLAG'},
        description="Which components the inspector draws. Buds and apices by default, "
        "because those are the ones with no geometry of their own",
        update=_redraw_viewports,
    )

    inspect_color_by: EnumProperty(
        name="Colour By", items=overlay.COLOR_ITEMS, default='KIND',
        description="What the glyph colours mean",
        update=_redraw_viewports,
    )

    # Buds render through Bud components now: a component placed On Organs is
    # what shows the plant's dormant buds and apices (split them with the Apex
    # mask channel), its Size is the bud size. The old `enable_buds` /
    # `enable_apices` / `bud_size` are gone; an old file's stored values are
    # read once, raw, to seed its components (`bridge._seed_bud_components`).
    bud_asset_collection: PointerProperty(
        type=bpy.types.Collection, name="Bud Assets",
        description=(
            "Legacy: the single bud collection from before Bud components. Kept only "
            "as the modifier's fallback source; add a Bud component instead"),
        update=_changed,
    )

    enable_leaves: BoolProperty(
        name="Leaves", default=True, update=_changed,
        description="Generate leaves. Off skips them entirely rather than scaling them away",
    )
    enable_flowers: BoolProperty(
        name="Flowers", default=True, update=_changed,
        description="Generate flowers. Off skips them entirely rather than scaling them away",
    )
    enable_catkins: BoolProperty(
        name="Catkins", default=True, update=_changed,
        description=(
            "Generate catkins. Off skips them entirely. They also need the Stem "
            "on, being swept with it rather than instanced"),
    )
    enable_fruit: BoolProperty(
        name="Fruit", default=True, update=_changed,
        description="Generate fruit. Off skips them entirely rather than scaling them away. "
        "Fruit only appears on the day path inside the calendar fruit window anyway",
    )
    """The uniform component gates of WS1 section 5.5. Each is redundant with
    its own size control at zero -- measured, `leaf_size` 0 already drops organ
    points 166 -> 23 and `catkin_length` 0 drops skeleton vertices 6906 -> 144,
    because `_organ_arrays` skips any organ whose scale is zero. They ship for
    uniformity anyway, on the user's explicit call after that measurement: one
    obvious way to say "none of these" without having to know that a size of
    zero is what does it."""

    inspect_active: BoolProperty(
        name="Inspecting", default=False,
        description="Whether the inspector overlay is showing this plant",
    )
    """On the object so Ctrl+Z can reverse turning the inspector on or off.
    The modal that tracks the cursor is not part of any datablock, so
    `overlay.reconcile` starts or stops it to match whatever undo restored."""

    inspect_pinned: CollectionProperty(type=PLANTGEN_PG_pin)
    """Which inspector cards stay up. Here rather than in the overlay module
    so Ctrl+Z reaches them -- see `PLANTGEN_PG_pin`."""

    # -- Manual pruning (WS3 Phase 3) --------------------------------------
    inspect_pruned: CollectionProperty(type=PLANTGEN_PG_pruned)
    """Axis ids the user has pruned. Applied at regrow (bridge) -- the pruned
    axis and its whole subtree are marked shed or dead. Valid only while
    `prune_signature` still matches the plant's topology."""
    prune_signature: StringProperty(default="")
    """The topology signature the pruned axis ids were captured against. When the
    plant's topology changes (seed, Age, the branching levels -- anything
    that reassigns axis ids), an id no longer names the same limb, so the bridge
    clears the set. A **topology** signature rather than the full growth
    signature on purpose: the growth signature also changes on render-only tweaks
    (Dead Colour, bloom, tip radius), which do not move a single axis, and
    clearing pruning on those would break the plan's own 'cosmetic tweaks don't
    disturb it'."""
    prune_stale: BoolProperty(default=False)
    """Set by the bridge when it had to clear the prune set because the topology
    changed. The panel shows a one-line warning until the next prune or clear."""
    prune_retain_dead_wood: BoolProperty(
        name="Prune Keeps Dead Wood", default=False, update=_changed,
        description="Whether a pruned limb stays as grey dead wood, or is removed "
        "entirely (the default). Distinct from the Age/Shade default, which keeps dead "
        "wood standing -- pruning is a gardener's cut, so it removes by default",
    )

    show_center_of_mass: BoolProperty(
        name="Center of Mass", default=False, update=_center_of_mass_changed,
        description="Show a small empty at each stem's own centre of mass - the "
        "mass-weighted average of everything that stem carries (its branches, leaves and "
        "flowers), using the same masses the bend does. It is the point that stem leans "
        "toward under gravity, so a stem hanging to one side has its marker on that side",
    )

    live_update: BoolProperty(
        name="Live Update", default=True,
        description="Regrow automatically a moment after changing a value, instead of only "
        "on the Regrow button",
    )

    # LEGACY, retired from the UI. The species dropdown was replaced by the
    # preset Library (bridge.ensure_migrated / presets). This property is kept,
    # unshown and with no update callback, ONLY so an old plant's stored value
    # can be read once and migrated to an embedded preset on load. Its items are
    # frozen -- do not reorder or remove (an EnumProperty stores the item index).
    species: EnumProperty(
        name="Species",
        items=(
            ('GENERIC', "Generic perennial", "A plain single-stem determinate perennial - the "
             "starting point, not a real species"),
            ('MEADOW_FOXTAIL', "Meadow foxtail", "Alopecurus pratensis - a tufted grass, many "
             "unbranched culms each ending in a dense spike"),
            ('RUDBECKIA', "Rudbeckia", "Rudbeckia fulgida, orange coneflower - a clump branching "
             "only near the top, each branch ending in one daisy head"),
            ('ANEMONE', "Japanese anemone", "Anemone hupehensis - tall wiry stems forking into an "
             "airy spray with one large flower per branch"),
            ('FOXGLOVE', "Foxglove", "Digitalis purpurea - a flat basal rosette that bolts into "
             "one tall unbranched spire of tubular flowers"),
            ('HAZEL', "Hazel", "Corylus avellana - a leafy twig whose stems bear no flowers at "
             "all, hung instead with catkins in bunches near the tips of its side shoots. "
             "Needs a catkin mesh assigned under Flowers > Catkins > Shape before the catkins "
             "appear"),
            ('LABURNUM', "Laburnum", "Laburnum anagyroides, golden chain - long racemes "
             "hanging from buds along last year's shoots, while the shoots themselves bear "
             "no flowers. The worked example of an inflorescence borne along a stem rather "
             "than at its tip"),
            ('LYCOPODIUM', "Clubmoss", "Lycopodium - a fern-ally whose growing tip forks into "
             "two equal halves over and over, the one worked example of apical dichotomy. "
             "Foliage-only, it never flowers; needs only a leaf mesh for its microphylls"),
            ('ARAUCARIA', "Araucaria", "Araucaria, monkey-puzzle - a conifer with a whorl of "
             "exactly four branches at every tier up a bare trunk, the one worked example of the "
             "sibling quota (Branches Per Node). Foliage-only, it never flowers; the long "
             "internode is what spaces the whorls into tiers"),
        ),
        default='GENERIC',
        description="Loads a whole species at once. Every control below is set from it, and "
        "editing them afterwards is how you vary from it - the species also carries traits "
        "that have no control of their own, such as its developmental curves",
    )
    # -- Preset (the species dropdown's successor) -------------------------
    # A preset is a full serialized profile, applied from the Library or a file.
    # `preset_embedded` is the self-contained base the plant resolves from (so a
    # .blend needs no external file and the base never changes unless re-applied);
    # `preset_id` remembers which preset it came from, powering Reload. Neither
    # takes an update callback: they are written by the apply path, not the user,
    # and editing the base is what applying a preset *is*.
    preset_id: StringProperty(
        name="Preset", default="",
        description="Which species preset this plant was last built from. Blank "
        "means it has been hand-built or detached. Powers 'Reload from preset'",
    )
    preset_name: StringProperty(
        name="Preset Name", default="",
        description="Display name of the last-applied preset, for the panel "
        "label. Kept on the object so drawing it needs no disk read",
    )
    preset_embedded: StringProperty(
        name="Embedded profile", default="", options={'HIDDEN'},
        description="The full serialized profile this plant resolves from, kept "
        "on the object so the .blend is self-contained. Written by the preset "
        "apply path, not meant to be edited by hand",
    )

    # -- Development -------------------------------------------------------
    # Real-days calendar: Day drives growth; Age is
    # now the derived internal (day -> season-fraction -> developmental progress).
    day: FloatProperty(
        name="Day", default=270.0, min=0.0, precision=0, step=100,
        update=_changed,
        description="The day to grow the plant to, counting from day 0. Unbounded - a "
        "perennial lives many years, so this runs past 365 into later seasons. Whole "
        "days here, but float-typed so the smooth timelapse can drive fractional days "
        "later. A calendar month/year view is a display of this",
    )
    emergence_day: FloatProperty(
        name="Emergence Day", default=90.0, min=0.0, soft_max=365.0, precision=0, step=100,
        update=_changed,
        description="Day of the year the plant emerges and its growing season starts "
        "(around 90 = spring). Shifts where the whole life sits on the calendar. Moves "
        "into the calendar widget later",
    )
    season_length_days: FloatProperty(
        name="Season Length", default=180.0, min=1.0, soft_max=365.0, precision=0, step=100,
        update=_changed,
        description="Length of the growing season in days, emergence to full maturity. "
        "Longer spreads the same growth over more real days. Moves into the calendar "
        "widget later",
    )
    lifespan: EnumProperty(
        name="Lifespan",
        items=(
            ('NONE', "Single season",
             "One season that stays mature past its end -- the original behaviour. "
             "Nothing changes when the Day is scrubbed into a later year"),
            ('ANNUAL', "Annual",
             "Lives one season, then dies at season end (fully senesced, standing) and "
             "never comes back"),
            ('PERENNIAL', "Herbaceous perennial",
             "Dies back to bare ground each winter and regrows a fresh top each spring, "
             "a touch bigger year over year to a mature plateau"),
        ),
        default='NONE',
        update=_changed,
        description="How the plant lives across years. Only affects Days scrubbed past "
        "the first season -- the mature Day-270 view is the same for all three",
    )

    # -- Phenology calendar windows (Milestone 2 calendar widget) -------------
    # Authored by dragging bands in the Phenology Calendar; each is days after
    # emergence. "Autumn" is the senescence window (it drives the per-leaf autumn
    # attribute and die-back). A plant seeds these from its own species/preset
    # calendar on apply (bridge.apply_profile) and they only drive growth once
    # `calendar_authored` is set, so a plant predating this stays byte-identical.
    # Consulted only on the day path.
    calendar_authored: BoolProperty(
        default=False,
        description="Internal: whether this plant's phenology windows are driven by "
        "these per-object values (set on preset apply and on first calendar edit) "
        "rather than by its embedded species calendar",
    )
    bloom_enabled: BoolProperty(
        name="Bloom", default=True, update=_changed,
        description="Whether the plant flowers on a calendar window (the Bloom lane). "
        "Off leaves flowering on its developmental driver",
    )
    bloom_start: FloatProperty(
        name="Bloom Start", default=55.0, min=0.0, precision=0, step=100, update=_changed,
        description="Days after emergence the bloom window opens",
    )
    bloom_days: FloatProperty(
        name="Bloom Days", default=75.0, min=0.0, precision=0, step=100, update=_changed,
        description="How many days the bloom window lasts",
    )
    fruit_enabled: BoolProperty(
        name="Fruit", default=True, update=_changed,
        description="Whether flowers set fruit on a calendar window (the Fruit lane)",
    )
    fruit_start: FloatProperty(
        name="Fruit Start", default=185.0, min=0.0, precision=0, step=100, update=_changed,
        description="Days after emergence the fruit window opens",
    )
    fruit_days: FloatProperty(
        name="Fruit Days", default=95.0, min=0.0, precision=0, step=100, update=_changed,
        description="How many days the fruit window lasts before drop",
    )
    autumn_enabled: BoolProperty(
        name="Autumn", default=True, update=_changed,
        description="Whether the plant senesces -- autumn colour and die-back -- on a "
        "calendar window (the Autumn lane). Drives the per-leaf autumn attribute",
    )
    autumn_start: FloatProperty(
        name="Autumn Start", default=180.0, min=0.0, precision=0, step=100, update=_changed,
        description="Days after emergence the autumn/senescence window opens",
    )
    autumn_days: FloatProperty(
        name="Autumn Days", default=60.0, min=0.0, precision=0, step=100, update=_changed,
        description="How many days the autumn/senescence window lasts",
    )

    # Per-plant Phenology Calendar palette. Overlay display only -- the calendar
    # widget's legend swatches edit these -- so they redraw the view, never
    # regrow, and are not a growth input.
    cal_col_season: FloatVectorProperty(
        name="Season", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.373, 0.541, 0.290), update=_redraw_viewports,
        description="This plant's Season colour in the Phenology Calendar")
    cal_col_bloom: FloatVectorProperty(
        name="Bloom", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.753, 0.416, 0.596), update=_redraw_viewports,
        description="This plant's Bloom colour in the Phenology Calendar")
    cal_col_fruit: FloatVectorProperty(
        name="Fruit", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.690, 0.271, 0.239), update=_redraw_viewports,
        description="This plant's Fruit colour in the Phenology Calendar")
    cal_col_autumn: FloatVectorProperty(
        name="Autumn", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.757, 0.502, 0.227), update=_redraw_viewports,
        description="This plant's Autumn colour in the Phenology Calendar")
    cal_col_dormant: FloatVectorProperty(
        name="Dormant", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.216, 0.216, 0.216), update=_redraw_viewports,
        description="This plant's Dormant colour in the Phenology Calendar")
    calendar_southern: BoolProperty(
        name="Southern Hemisphere", default=False, update=_redraw_viewports,
        description="Label the calendar's season band for the Southern hemisphere "
        "(December-February is summer). Display only -- the day-of-year and growth are "
        "unchanged; toggle it from the ◐ button in the calendar")

    # -- Timelapse (calendar phase 6): drive Day from the timeline frame -------
    animate_day: BoolProperty(
        name="Grow Over Timeline", default=False, update=_animate_day_changed,
        description="Play this plant's growth on the timeline: every frame sets the Day "
        "(Start Day on the scene's first frame, End Day on its last) and regrows the real "
        "plant. Exact -- every stem, leaf and flower as it is on that day -- but a full "
        "regrow per frame, so it is for rendering a growth timelapse, not for real-time "
        "playback. While on, the Day field follows the timeline",
    )
    timelapse_start_day: FloatProperty(
        name="Start Day", default=90.0, min=0.0, precision=0, step=100,
        update=_animate_day_changed,
        description="The Day the growth timelapse shows on the scene's first frame",
    )
    timelapse_end_day: FloatProperty(
        name="End Day", default=300.0, min=0.0, precision=0, step=100,
        update=_animate_day_changed,
        description="The Day the growth timelapse shows on the scene's last frame. Past a "
        "perennial's first season this runs into dormancy and next year's regrowth",
    )
    rig_view: BoolProperty(
        name="Rig View", default=False, update=_rig_view_changed,
        description="Show this plant as its baked rig -- the skinned mesh, posed and "
        "animated by the bones -- instead of the live build. Still the one object you "
        "select and edit; any edit that regrows drops back to the live plant and marks "
        "the rig out of date",
    )
    rig_fast_preview: BoolProperty(
        name="Fast Preview", default=True, update=_rig_view_changed,
        description="Play the rig fast in Rig View: only the stems deform, and every leaf "
        "and flower stays a GPU instance that follows its bone -- the same look, a fraction "
        "of the work per frame. Off: the full skinned mesh deforms (slow with many flowers). "
        "Falls back to the full mesh while Vertex Wind is on. The export always writes the "
        "full mesh",
    )
    rig_show_bones: BoolProperty(
        name="Show Bones", default=False, update=_rig_view_changed,
        description="Draw the rig's bones over the plant in Rig View",
    )
    rig_bone_display: EnumProperty(
        name="Bone Display", default='OCTAHEDRAL', update=_rig_view_changed,
        items=[('OCTAHEDRAL', "Octahedral", "Bones as octahedra (Blender's default)", 0),
               ('STICK', "Stick", "Bones as thin lines -- the least clutter on a dense rig", 1),
               ('BBONE', "B-Bone", "Bones as boxes", 2),
               ('ENVELOPE', "Envelope", "Bones as their deform envelopes", 3),
               ('WIRE', "Wire", "Bones as wireframe lines", 4)],
        description="How the rig's bones are drawn in the viewport. Display only -- the "
        "skeleton and an export are the same either way",
    )
    # -- Wind. NOT in _FIELDS: a
    #    pose/debug layer over the baked rig, never a growth input.
    wind_enable: BoolProperty(
        name="Bone Wind", default=False, update=_wind_changed,
        description="An approximation of Unreal's wind, to preview in Blender: the rig's "
        "bones sway with the gusts, thin limp parts most, everything on them riding along. "
        "Nothing of it is exported -- Unreal simulates the real sway itself, from the wind "
        "groups Export Rig writes (made from the same stem flexibility this preview uses). "
        "Plays on the timeline in Rig View; needs a rig (Build Rig)",
    )
    wind_secondary_enable: BoolProperty(
        name="Vertex Wind", default=False, update=_wind_changed,
        description="Flutter leaves and petals with a wave driven by the vertex colours "
        "(Red: still at the base, free at the tip; Green / Blue: each leaf and plant out of "
        "step). A viewport preview of what an engine's foliage material does at runtime "
        "from the same colours -- never exported or baked. Needs a rig (Build Rig)",
    )
    wind_secondary_strength: FloatProperty(
        name="Vertex Wind Strength", default=1.0, min=0.0, soft_max=4.0, update=_wind_changed,
        description="How far leaf and petal tips flutter. Shares Direction, Gust Speed and "
        "Turbulence with Bone Wind",
    )
    wind_strength: FloatProperty(
        name="Bone Wind Strength", default=1.0, min=0.0, soft_max=4.0, update=_wind_changed,
        description="How hard the preview wind pushes the bones. Preview only -- Unreal's "
        "wind strength comes from its own wind source in the level",
    )
    wind_direction_deg: FloatProperty(
        name="Wind Direction", default=120.0, min=0.0, max=360.0, update=_wind_changed,
        description="Preview wind heading in degrees (0-360) on the ground plane; gusts "
        "swirl around it. Preview only -- Unreal takes its wind direction from the level",
    )
    wind_stiffness: FloatProperty(
        name="Bone Stiffness", default=1.0, min=0.01, soft_max=8.0, update=_wind_changed,
        description="Trims how far the preview bends -- higher sways less. Each bone's own "
        "give still comes from the stem's flexibility, so thin limp parts sway while thick "
        "ones barely move. Preview only -- it does not change what Unreal gets",
    )
    wind_gust_scale: FloatProperty(
        name="Gust Scale", default=2.5, min=0.001, soft_max=4.0, update=_wind_changed,
        description="Gust spatial frequency -- higher gives smaller, tighter gusts",
    )
    wind_gust_speed: FloatProperty(
        name="Gust Speed", default=1.0, min=0.0, soft_max=3.0, update=_wind_changed,
        description="How fast the gust field drifts and evolves over time",
    )
    wind_turbulence: FloatProperty(
        name="Turbulence", default=1.5, min=0.0, soft_max=2.0, update=_wind_changed,
        description="Swirl amount -- curl-noise eddying added to the wind heading",
    )
    wind_loop_length: FloatProperty(
        name="Loop Length", default=10.0, min=0.5, soft_max=30.0, update=_wind_changed,
        description="Seconds for one seamless loop of the gust field (the mature-idle "
        "clip loops on this)",
    )
    wind_seed: IntProperty(
        name="Wind Seed", default=0, min=0, update=_wind_changed,
        description="Randomizes the whole gust pattern -- each value is a wholly different "
        "field (and decorrelates this plant from others)",
    )
    wind_debug: EnumProperty(
        name="Wind Debug", default='OFF', update=_wind_debug_changed,
        items=[('OFF', "Off", "No wind debug view"),
               ('PLANE', "Plane", "A 2D plane coloured by the animated gust field"),
               ('VOLUME', "Volume", "A 3D voxel grid coloured by the gust field"),
               ('BOTH', "Both", "Both the plane and the volume")],
        description="Visualise the wind noise field -- blue = lull, green = neutral, red = gust",
    )
    wind_debug_res: IntProperty(
        name="Debug Resolution", default=48, min=2, max=128, update=_wind_debug_changed,
        description="Grid resolution of the wind debug view(s)",
    )
    # Skin vertex-color debug: viewers on the SKINNED mesh (the plant's own Debug
    # View taps the static, hidden plant mesh). Driven by the toggle_skin_debug
    # operator, not an update callback -- activating a viewer needs editor context.
    wind_skin_debug: EnumProperty(
        name="Skin Vertex Colors", default='off',
        items=[('off', "Off", "No skin vertex-color overlay"),
               ('color', "Combined", "R, G and B together on the skinned mesh"),
               ('vcol_r', "Red (base->tip)", "The wave-amplitude mask the Secondary wind reads"),
               ('vcol_g', "Green (per-leaf)", "The per-leaf random the Secondary wind phases by"),
               ('vcol_b', "Blue (per-plant)", "The per-plant random"),
               ('vcol_a', "Alpha (height)", "Whole-plant height fraction")],
        description="Tint the skinned mesh by a vertex-color channel -- the same data the "
        "Secondary wind reads, shown on the mesh exactly like the plant's own Debug View",
    )
    age: FloatProperty(
        name="Age", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Derived developmental progress, 0 seedling to 1 mature. The Day "
        "counter above now drives growth; this is kept as the internal value the sim "
        "runs on and is no longer the control you set",
    )
    calendar_migrated: BoolProperty(
        default=False, options={'HIDDEN'},
        description="Internal: whether this plant's Day has been derived once from its "
        "legacy Age. Not user-facing",
    )
    # -- Morphology --------------------------------------------------------
    # Stem count, splay, branch orders and forking are the branching stack's
    # (`branch_levels`, the Branching panel).
    basal_spread: FloatProperty(
        name="Basal Spread", default=0.04, min=0.0, unit='LENGTH', update=_changed,
        description="Radius of the clump at ground level: the stems fill it as a "
        "phyllotactic disc, evenly from the centre to this distance",
    )
    basal_spread_variation: FloatProperty(
        name="Spread Variation", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="How irregularly the stems stand: each stem's foot moves by up to "
        "this fraction of the Spread in X and in Y, drawn per stem (0.5: up to half the "
        "spread). 0 keeps the even phyllotactic disc",
    )
    # Removed 2026-09-30 (a tip flowers when the flower component is Terminal);
    # read once by the migration (object branch-levels schema < 8), then unused.
    axis_type: EnumProperty(
        items=(('DETERMINATE', "Determinate", ""), ('INDETERMINATE', "Indeterminate", "")),
        default='DETERMINATE', options={'HIDDEN'},
    )
    inflorescence_kind: EnumProperty(
        name="Inflorescence Kind", update=_inflorescence_kind_changed,
        items=(
            # **Every item carries an explicit number**, which is what makes the
            # headings and the grouping below safe. An EnumProperty otherwise
            # stores the item's *position*, so inserting one -- a heading very
            # much included, they take an index of their own -- silently
            # rewrites the kind on every saved plant. Removing Head cost a
            # repair pass for exactly that, and this list has grown three times
            # since. Pinned to the numbers the list already had, so nothing
            # stored moved; display order is now free, and new entries take the
            # next free number rather than having to be appended.
            #
            # An empty identifier draws as a heading. It is not selectable and
            # never appears in `enum_items`.
            # Ungrouped and first, because it is the absence of the thing the
            # three headings below classify rather than one more kind of it.
            ('NONE', "None", "No flowers at all. Stems finish their development and stop, "
             "the same plant as any other kind minus its inflorescence - for foliage "
             "species, and for the branch a catkin hangs from, which has to be able to "
             "mature without turning into an inflorescence itself.\n\n"
             "Different from an indeterminate plant with nothing to flower with: those "
             "stems never stop, and grow on until the plant runs out of developmental "
             "span", 'NONE', 18),
            ("", "Regular", ""),
            ('SOLITARY', "Solitary", "One flower at the tip of each stem", 'NONE', 0),
            ('RACEME', "Raceme", "Flowers on short pedicels along a continuing tip (foxglove, "
             "delphinium)", 'NONE', 1),
            ('SPIKE', "Spike", "Like a raceme but the flowers sit directly on the stem, no "
             "pedicels (plantain, many grasses)", 'NONE', 2),
            ('CORYMB', "Corymb", "A raceme whose lower pedicels are longer, bringing every "
             "flower up to about one level (yarrow)", 'NONE', 3),
            ('UMBEL', "Umbel", "Every pedicel radiating from one point at the tip, filling a "
             "dome right through its centre (allium)", 'NONE', 4),
            ('FLAT_UMBEL', "Flat Umbel", "An umbel whose pedicels are solved to land on one "
             "plane, so the head reads flat (cow parsley, Orlaya)", 'NONE', 5),
            ('CATKIN', "Catkin", "The stems bear no flowers themselves and hang catkins "
             "at their nodes instead - hazel, birch, willow. The one type here that is not "
             "something a stem turns into at its tip: a hazel's twigs go on being twigs and "
             "carry catkins along their length, on short stalks.\n\n"
             "Shape them in the Catkins box below. They are modelled rather than instanced, "
             "so they hang under their own weight and need no asset assigned", 'NONE', 6),
            ('PANICLE', "Panicle", "A branched raceme: the stem carries branches rather than "
             "flowers and does not flower itself, and each branch repeats that until the last "
             "one, which flowers (oat, astilbe, yucca). Its depth is Branching Depth below, "
             "counted inside the inflorescence and independent of Max Branch Order - so a "
             "tree can carry panicles on twigs that are already several branch orders deep",
             'NONE', 8),
            ('VERTICILLASTER', "Verticillaster", "Rings of flowers spaced up the stem, "
             "each subtended by a pair of leafy bracts, with bare stem between them - the "
             "false whorl of the mint family (dead-nettle, motherwort, mint, salvia). Sets "
             "Flowers per Whorl and Bract Size, which are ordinary controls the other "
             "flowering kinds can use too. Botanically each ring is two condensed cymes "
             "rather than a true whorl; this builds a real ring, which reads the same at "
             "mint density", 'NONE', 9),
            # One mechanism, five entries: the four named forms differ from Cyme
            # only in Relays per Node and Relay Rotation, which is why they are
            # stamps rather than branches in the growth loop. See
            # `morphology._RELAY_KINDS`.
            ("", "Cymose", ""),
            ('CYME', "Cyme (custom)", "The tip flowers first and stops, and growth carries "
             "on from shoots just below it, each repeating the pattern. This is the "
             "unstamped form: set Relays per Node and Relay Rotation yourself. The four "
             "named cymes below - Scorpioid, Helicoid, Dichasium, Pleiochasium - are this "
             "same mechanism with those two numbers already set, and are what you want "
             "unless you are after a combination they do not name", 'NONE', 7),
            ('SCORPIOID', "Scorpioid Cyme", "A zigzag chain: the tip flowers and one shoot "
             "below it carries on, alternating sides each time, so the whole spray kinks "
             "from side to side in one plane (forget-me-not, comfrey, heliotrope). Sets "
             "Relays per Node to 1 and Relay Rotation to 180. How hard it zigzags is "
             "Branch Angle in the Branching box - +90 grows straight on with no kink, "
             "lower values fork harder every relay", 'NONE', 10),
            ('HELICOID', "Helicoid Cyme", "Like a scorpioid cyme but every relay arises on "
             "the same side rather than alternating, so instead of zigzagging the spray "
             "coils (Alstroemeria, sundew, many nightshades). Sets Relays per Node to 1 and "
             "Relay Rotation to 0 - that single number is the whole difference between this "
             "and a scorpioid cyme", 'NONE', 11),
            ('DICHASIUM', "Dichasium", "The familiar forked cyme: the spent tip is flanked "
             "by two shoots, each of which forks again, doubling every generation "
             "(chickweed, campion, most pinks). Sets Relays per Node to 2 and Relay Rotation "
             "to 90, so successive forks cross rather than stacking in one plane. Note the "
             "count doubles per Branching Depth", 'NONE', 12),
            ('PLEIOCHASIUM', "Pleiochasium", "Three or more shoots in a whorl below each "
             "spent tip, giving a broad umbel-like spray (spurge). Sets Relays per Node to 3 "
             "and Relay Rotation to 60. The count multiplies by Relays per Node every "
             "Branching Depth, so it grows very fast - three at depth 3 is 27 flowers", 'NONE', 13),
            # Presets over a kind above, not kinds -- see `_COMPOUND_PRESETS`.
            # Picking one snaps Type back to its base kind with Branching Depth
            # set, because that is what a compound inflorescence *is* here.
            ("", "Compound", ""),
            ('COMPOUND_SPIKE', "Compound Spike", "A spike of spikes: every node of the "
             "flowering stem carries a whole spike of its own instead of one flower. Sets "
             "Type to Spike and Branching Depth to 1 - the dropdown then reads Spike, "
             "because that is what this is", 'NONE', 14),
            ('COMPOUND_CORYMB', "Compound Corymb", "A corymb whose flowers are themselves "
             "corymbs, so the head reads as a broad flat spray of little flat sprays "
             "(elder, yarrow). Sets Type to Corymb, Branching Depth to 1, and the Branching "
             "box - branches short, well spaced, finely tapered and arched", 'NONE', 15),
            ('COMPOUND_UMBEL', "Compound Umbel", "An umbel of umbels: each ray of the head "
             "ends in a whole umbel of its own. Sets Type to Umbel, Branching Depth to 1 "
             "and a long branch length, so the umbellets stand clear of one another", 'NONE', 16),
            ('COMPOUND_FLAT_UMBEL', "Compound Flat Umbel", "The cow parsley and hogweed "
             "form: rays from one point, each ending in its own flat-topped umbel. Sets "
             "Type to Flat Umbel, Branching Depth to 1, and the Branching box - ten long, "
             "slender, gently arched rays", 'NONE', 17),
        ),
        default='SOLITARY',
        description="How the flowers are arranged where a stem finishes",
    )
    inflorescence_attachments: EnumProperty(
        name="Attachment Location", update=_attachment_changed, options={'ENUM_FLAG'},
        default={'TERMINAL'},
        items=(
            # ENUM_FLAG: the stored value is a BITMASK, so every number here
            # must be a distinct power of two and none may be 0. That is also
            # why this property was RENAMED when it became multi-select rather
            # than keeping its old name -- the old single-value numbers were
            # 0..7, and reusing the name would have silently reinterpreted a
            # saved 3 (Cauliflorous) as the bitmask {Terminal, Axillary}. A new
            # name orphans the old value harmlessly instead of corrupting it.
            #
            # Explicit numbers, and the NUMBER is what actually gets stored --
            # verified live: renaming an identifier while keeping its number
            # reads back fine (CAULINE -> CAULIFLOROUS at 3 survives untouched),
            # while moving a number returns '' on every plant that held the old
            # one. So labels and identifiers are free to change and only the
            # numbers are load-bearing; a new entry takes the next unused number
            # rather than being appended in display order.
            # Original note: this list is expected to grow
            # (Internodal and Cauline are named in the model already), and an
            # EnumProperty stores the item's position, so appending later would
            # otherwise rewrite the value on every saved plant.
            ('TERMINAL', "Terminal", "Each stem finishes by becoming the inflorescence - "
             "the usual case, and what a foxglove, a yarrow or an allium does", 'NONE', 1),
            ('AXILLARY', "Axillary", "Inflorescences arise at nodes along the stems, each "
             "on its own short stalk, while the stems go on being stems and bear no flowers "
             "themselves. Wisteria, laburnum, birch and hazel.\n\n"
             "The stalks are not branches: they carry no leaves and sit outside Branch "
             "Density and Branching Zone, so those keep describing the plant's woody shape. "
             "Pair with Orientation to make them hang", 'NONE', 2),
            ('INTERNODAL', "Internodal", "From partway along an internode rather than at "
             "a node. Uncommon, but real - and the position along the internode is settable "
             "below", 'NONE', 4),
            ('LEAF_OPPOSED', "Leaf-opposed", "At a node, but on the far side of the stem "
             "from the leaf rather than in its axil - oppositifolious, as grapes and pepper "
             "are.\n\nHalf a turn round from where an axillary one would sit; everything "
             "else about it matches Axillary", 'NONE', 16),
            ('CAULIFLOROUS', "Cauliflorous", "Straight out of wood laid down seasons "
             "earlier on the TRUNK, well below the leaves - cauliflory, as figs and cacao "
             "do.\n\nDefined by age rather than position: nothing appears until a node has "
             "carried for the stated number of steps, so a young plant shows none", 'NONE', 8),
            ('RAMIFLOROUS', "Ramiflorous", "The same old-wood break as Cauliflorous, but "
             "along the BRANCHES instead of the trunk - ramiflory, as redbud does on bare "
             "limbs.\n\nShares the Wood Age gate; the two differ only in which stems may "
             "bear it", 'NONE', 32),
            ('BASAL', "Basal", "From the root crown, straight out of the ground rather than "
             "from anywhere up a stem - violets, many bulbs, and a primrose's scapes.\n\n"
             "Arises at once rather than waiting on Wood Age: a basal scape is the first "
             "thing such a plant does", 'NONE', 64),
            ('ADVENTITIOUS', "Adventitious", "From old wood that never carried a meristem "
             "at all - scattered between the nodes, following no phyllotaxis.\n\nThe one "
             "position whose buds are made rather than found, which is what makes it "
             "adventitious. Shares the Wood Age gate; use Per Node to set how many break "
             "from each internode", 'NONE', 128),
        ),
        description="Where the inflorescences arise - at the tips of stems, along them, "
        "out of old wood, or from the base of the plant",
    )
    inflorescence_per_node: IntProperty(
        name="Per Node", default=1, min=1, soft_max=6, update=_changed,
        description="How many inflorescences arise at each position the attachment "
        "picks - 1 is a single one, higher fans a cluster out from the same point.\n\n"
        "This is what a 'terminal cluster' or an 'axillary cluster' is: a count "
        "applied to a position, not a position of its own. As one control it clusters "
        "any attachment, including ones a fixed pair of cluster entries could never "
        "have covered",
    )
    inflorescence_internode_position: FloatProperty(
        name="Along Internode", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Where along the internode it sits, 0 at the node below and 1 at the "
        "node above. Only used by Between Nodes",
    )
    inflorescence_cauline_age: IntProperty(
        name="Wood Age", default=8, min=0, max=60, update=_changed,
        description="How many growth steps a node must have carried before an inflorescence "
        "can break from it. Only used by From Old Wood - it is what makes cauliflory a "
        "matter of age rather than of position",
    )
    inflorescence_orientation: EnumProperty(
        name="Orientation", update=_changed,
        items=(
            # Numbers pinned. 0 is reused deliberately: it was Inherited, and
            # Parent-aligned at strength 0 IS inherited -- hold the launch
            # angle -- so every saved plant on 0 keeps its behaviour untouched.
            # 1/3/4 were World Angle / Bend Under Load / Follow Stem and are
            # left unused rather than reassigned, so a plant holding one reads
            # back as invalid and gets migrated rather than silently becoming a
            # different direction. 2 was Steer To Angle, removed 2026-08-20.
            ('PARENT_ALIGNED', "Parent-aligned", "Prefer to lie along the stem that carries "
             "it, whatever that stem is doing.\n\nRe-read as the plant grows, so the "
             "inflorescence swings when the branch bends. At Strength 0 this is simply the "
             "angle it left the stem at, which is what every plant did before orientation "
             "existed", 'NONE', 0),
            ('ERECT', "Erect", "Prefer to point up, against gravity", 'NONE', 5),
            ('HORIZONTAL', "Horizontal", "Prefer to lie level, perpendicular to gravity.\n\n"
             "Level is a whole plane, so it keeps the compass direction the bud already had "
             "- radially out from its own stem - rather than sending every inflorescence to "
             "one side of the plant", 'NONE', 6),
            ('PENDULOUS', "Pendulous", "Prefer to hang, with gravity", 'NONE', 7),
            ('CUSTOM', "Custom", "Prefer the angle set below, measured from the parent "
             "stem's own direction rather than from world vertical - so it is stated "
             "relative to the stem and swings with it", 'NONE', 8),
        ),
        default='PARENT_ALIGNED',
        description="Which direction this inflorescence prefers to point",
    )
    inflorescence_orientation_strength: FloatProperty(
        name="Strength", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How far the inflorescence is pulled from the direction it would "
        "naturally leave the stem at, toward the preferred direction above.\n\n"
        "0 leaves it exactly as it grew; 1 pulls it fully onto the preference. It is a "
        "preference rather than a rule - even at 1 the result keeps some of where it came "
        "from, and the stem's own weight then bends it further, so an organ is not forced "
        "onto exact vertical.\n\n"
        "It also sets how sharply the turn happens: the turn is always spread over the same "
        "short run of nodes, so a bigger pull bends harder over that run. Higher is "
        "sharper, and no value gives a hard corner",
    )
    inflorescence_orientation_deg: FloatProperty(
        name="Angle", default=0.0, min=-180.0, max=180.0, update=_changed,
        description="Custom's angle, measured from the parent stem's own direction. "
        "0 continues straight along the stem, positive tips away from vertical, negative "
        "back toward it.\n\n"
        "Only used by Custom - the other four state their own direction",
    )
    inflorescence_orientation_load_driven: BoolProperty(
        name="Reach It As Flowers Open", default=False, update=_changed,
        description="Arrive at the preferred direction as the inflorescence fills with "
        "flowers, rather than over its first few nodes - upright while it is a bud, bowed "
        "over once laden.\n\n"
        "Different from Droop, which bends the finished stem: this moves while the "
        "inflorescence is still growing, so it changes where the flowers are actually placed",
    )
    inflorescence_branching_depth: IntProperty(
        name="Branching Depth", default=0, min=0, max=12, update=_changed,
        description="Makes the inflorescence COMPOUND. At each place it would have put a "
        "flower it puts a whole inflorescence of its own kind instead, this many times "
        "over. 0 is simple.\n\n"
        "1 turns a raceme into a panicle, an umbel into a compound umbel (carrot, cow "
        "parsley), a corymb into a compound corymb (yarrow, elder) and a spike into a "
        "compound spike (wheat). 2 gives the triple umbels real cow parsley makes. On a "
        "cyme it is instead how many times the tip relays, so a long zigzag chain wants "
        "8 or more.\n\n"
        "Independent of the Branching levels, which stay the plant's own vegetative "
        "branching - a plant can have several levels of branch under two of "
        "inflorescence, and one number cannot say both.\n\n"
        "Watch the count: every level multiplies it by Rays (or by Relays per Node on a "
        "cyme), and it multiplies again by however many branch tips the plant itself has. "
        "Depth 2 on a bushy plant is thousands of flowers",
    )
    inflorescence_branch_angle: FloatProperty(
        name="Branch Angle", default=45.0, min=-90.0, max=90.0, update=_changed,
        description="Which way an inflorescence branch leaves the main stem, measured from "
        "perpendicular: 0 is straight out sideways, +90 flat up along the stem, -90 flat down. "
        "Separate from the plant's own Branch Angle, which describes its woody branches. This "
        "is the main thing separating a diffuse, open panicle from a contracted, spike-like "
        "one.\n\n"
        "On a cyme this is the zigzag: it is the angle each relay leaves the spent tip "
        "above it at, so +90 carries straight on with no visible kink and lower values "
        "fork harder every generation. The kink is exactly 90 minus this",
    )
    inflorescence_branch_nodes: IntProperty(
        name="Branch Nodes", default=2, min=1, max=20, update=_changed,
        description="Nodes an inflorescence branch lays down before it starts flowering - the "
        "bare basal stretch of a panicle branch. Only used by Panicle",
    )
    inflorescence_branch_count: IntProperty(
        name="Branch Count", default=7, min=1, max=200, update=_changed,
        description="How many branches the main stem of the inflorescence carries. Separate "
        "from Flower Count, which is how many flowers each of those branches ends up with - "
        "one number used to serve both, so ten branches of three flowers was not expressible. "
        "Only used by Panicle",
    )
    inflorescence_branch_count_variation: FloatProperty(
        name="Branch Count Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to the branch count, as a fraction of it. Drawn per axis, "
        "so one primary branch carries more than its neighbour",
    )
    inflorescence_branch_angle_variation: FloatProperty(
        name="Branch Angle Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to the angle each branch leaves at, as a fraction of it",
    )
    inflorescence_branch_nodes_variation: FloatProperty(
        name="Branch Nodes Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to how many nodes each branch carries, as a fraction of it",
    )
    inflorescence_branch_spacing: FloatProperty(
        name="Branch Spacing", default=1.5, min=0.0, max=8.0, soft_max=4.0, update=_changed,
        description="Gap between successive branches along the main stem of the inflorescence, "
        "relative to an ordinary internode. Separate from Flower Spacing on purpose: a panicle "
        "stands its branches well apart while crowding the flowers on them, and one number "
        "cannot say both. Only used by Panicle",
    )
    inflorescence_branch_spacing_variation: FloatProperty(
        name="Branch Spacing Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to the gap between branches, as a fraction of it",
    )
    inflorescence_branch_length: FloatProperty(
        name="Branch Length", default=1.0, min=0.0, soft_max=4.0, update=_changed,
        description="How far apart the nodes of an inflorescence branch sit, and so how long "
        "the branch is. Separate from Branch Nodes, which is how many flowers it carries - "
        "before this a branch could only be lengthened by making it carry more. Only used by "
        "Panicle",
    )
    inflorescence_branch_length_variation: FloatProperty(
        name="Branch Length Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to each branch's length, as a fraction of it",
    )
    inflorescence_branch_width: FloatProperty(
        name="Branch Width", default=0.6, min=0.01, max=1.0, subtype='FACTOR',
        update=_changed,
        description="A branch's thickness where it leaves the main stem, as a fraction of that "
        "stem's own thickness there - so 1.0 is exactly as thick as the stem it comes off, and it "
        "cannot go above that. Same as Pedicel Width. Its graph is the taper along one branch, "
        "base to tip, and shapes only the taper - the base width is this number's to set. Thinner "
        "branches also arch further, because bending resists as the fourth power of radius. "
        "Only used by Panicle",
    )
    inflorescence_branch_width_variation: FloatProperty(
        name="Branch Width Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to each branch's thickness, as a fraction of it",
    )
    inflorescence_branch_bend: FloatProperty(
        name="Branch Bend", default=0.0, min=-1.0, max=1.0, subtype='FACTOR', update=_changed,
        description="A steady arch along each inflorescence branch, held in the branch's own "
        "plane - so it turns with the branch rather than always heading for the ground. 1 is a "
        "quarter turn from base to tip. This is shape; Branch Droop below is weight. Only used "
        "by Panicle",
    )
    inflorescence_branch_bend_variation: FloatProperty(
        name="Branch Bend Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to each branch's arch, as a fraction of it",
    )
    inflorescence_branch_droop: FloatProperty(
        name="Branch Droop", default=0.0, min=-1.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="How far an inflorescence branch turns out of the way it was aimed, over "
        "its whole length. Negative sags it toward the ground, -1 hanging straight down; positive "
        "lifts it toward vertical - the same sign as Angle and Bend above it, and what an "
        "erect panicle's upswept primaries need.\n\n"
        "Distinct from Peduncle Droop, which only bends the stretch below the lowest flower "
        "and therefore cannot touch a branch that carries branches rather than flowers. "
        "Unlike Bend it is measured against gravity, so it does not turn with the branch",
    )
    inflorescence_branch_droop_variation: FloatProperty(
        name="Branch Droop Variance", default=0.0, min=0.0, max=1.0, update=_changed,
        description="Random change to how far each branch turns, as a fraction of it. Scatters "
        "the magnitude, not the direction - a branch does not flip sides",
    )
    flower_count: IntProperty(
        name="Flower Count", default=7, min=1, max=200, update=_changed,
        description="Flowers per inflorescence. Ignored by Solitary and by the cymose "
        "kinds, which get their count from how many times they relay instead - that is "
        "Branching Depth times Relays per Node",
    )
    inflorescence_spiral: FloatProperty(
        name="Spiral", default=0.0, min=0.0, max=360.0, update=_changed,
        description="The flowers' own spiral up the inflorescence: each flowering node's "
        "flower turned this many degrees round the stem from the last (137.5 is the golden "
        "angle). 0 follows the plant's leaf arrangement. Lets a spike spiral while its "
        "leaves stay two-ranked, as a grass's do. Flowers per Whorl (2 or more) "
        "replaces it",
    )
    inflorescence_whorl_count: IntProperty(
        name="Flowers per Whorl", default=0, min=0, max=16, update=_changed,
        description="How many flowers each node of the inflorescence carries, in a ring "
        "around the stem. 0 follows the plant's own leaf arrangement, which is one flower "
        "per leaf position - so 0 is exactly what every kind did before this "
        "existed.\n\n"
        "Separate from Trunk Pattern on purpose: one flower is placed per leaf position, so "
        "asking for a six-flowered whorl through the phyllotaxis also gives the leafy stem "
        "below six leaves per node. A mint has two. This is the flowering region stating "
        "its own arrangement",
    )
    inflorescence_laterals: IntProperty(
        name="Relays per Node", default=2, min=1, max=6, update=_changed,
        description="How many shoots a cyme starts below its spent tip. 1 gives a "
        "monochasium, a single chain that zigzags or coils depending on Relay Rotation; "
        "2 the familiar forked cyme (dichasium); 3 or more a whorled spray "
        "(pleiochasium).\n\n"
        "The count multiplies by this every Branching Depth, so 2 at depth 4 is 16 tips "
        "and 3 at depth 4 is 81. On a panicle this is instead how many branches each "
        "node of the main stem puts out",
    )
    inflorescence_relay_rotation: FloatProperty(
        name="Relay Rotation", default=90.0, min=-180.0, max=180.0, update=_changed,
        description="How far the relays are turned around the stem from one generation of "
        "a cyme to the next.\n\n"
        "This one number is what separates the classical cymose forms. With 1 relay per "
        "node, 180 puts each successive shoot on the opposite side and the spray zigzags "
        "in a plane (scorpioid); 0 puts every shoot on the same side and it coils "
        "(helicoid). With 2, 90 makes successive forks cross instead of stacking flat.\n\n"
        "Measured in the parent stem's own frame, so the rotations compose down the "
        "chain rather than each being measured afresh - which is what makes a helicoid "
        "cyme a real coil. Only used by the cymose kinds",
    )
    pedicel_length: FloatProperty(
        name="Pedicel Length", default=0.03, min=0.0, unit='LENGTH', update=_changed,
        description="Length of the pedicel - the little stalk between the stem and each "
        "flower. A corymb lengthens the lower ones automatically; a spike and a catkin ignore "
        "this, because their flowers are sessile and have no pedicel at all",
    )
    flower_count_variation: FloatProperty(
        name="Flower Count Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random change to how many flowers each inflorescence carries, as a "
        "fraction of the count. Drawn once per stem, not per flower, so on a branched plant "
        "this is what stops every raceme being exactly the same length",
    )
    flower_spacing_variation: FloatProperty(
        name="Flower Spacing Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random change to the gap between one flower and the next, as a fraction "
        "of it. Drawn per gap, so a single raceme's own flowers sit unevenly",
    )
    pedicel_variation: FloatProperty(
        name="Pedicel Length Variance", default=0.2, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random length on each pedicel, as a fraction of its own length",
    )
    pedicel_angle_variation: FloatProperty(
        name="Pedicel Angle Variance", default=0.0, min=0.0, max=90.0, update=_changed,
        description="Random tilt on each pedicel, in degrees either way",
    )
    pedicel_bend_variation: FloatProperty(
        name="Pedicel Bend Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random curvature on each pedicel, as a fraction of its own bend",
    )
    pedicel_angle: FloatProperty(
        name="Pedicel Angle", default=30.0, min=-90.0, max=90.0, update=_changed,
        description="How far a pedicel tilts off horizontal. 0 leaves the stem straight "
        "out sideways, positive swings it up toward the tip, negative hangs it down for a "
        "drooping raceme. Plus or minus 90 presses it flat against the stem",
    )
    cluster_spread: FloatProperty(
        name="Head Spread", default=150.0, min=10.0, max=360.0, update=_changed,
        description="How wide an umbel opens. Small values give a tight cone, 180 a flat "
        "hemisphere, and 360 a complete sphere of flowers - an Allium globe or an ivy head. "
        "Flowers fill the whole dome including its centre, not just its rim",
    )
    head_radius: FloatProperty(
        name="Head Radius", default=0.045, min=0.001, unit='LENGTH', update=_changed,
        description="How wide a flat-topped head is. The flowers pack evenly inside it, so "
        "this is what keeps a corymb or flat umbel compact instead of a wide sparse fan. Only "
        "used by Corymb and Flat Umbel",
    )
    head_dome: FloatProperty(
        name="Head Dome", default=0.0, min=-1.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How far the outer flowers of a flat-topped head sit below the middle "
        "ones, as a fraction of Head Radius. 0 is dead flat; 1 drops the rim by a full radius, "
        "which is a dome as tall as it is wide; negative cups the head instead. The graph is "
        "the head's own cross-section from centre to rim. Only used by Corymb and Flat Umbel",
    )
    head_dome_variation: FloatProperty(
        name="Head Dome Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random height on each flower of a flat-topped head, as a fraction of Head "
        "Radius. Measured against the head rather than against Dome, so it roughens a flat top "
        "as readily as a domed one - which is where an even surface reads most artificial",
    )
    pedicel_width: FloatProperty(
        name="Pedicel Width", default=0.3, min=0.01, max=1.0, subtype='FACTOR',
        update=_changed,
        description="How thick each pedicel is where it leaves the stem, as a fraction "
        "of the stem's own thickness there. 1 continues the stem's width, which reads as a "
        "short branch; most plants carry conspicuously finer pedicels than that",
    )
    pedicel_width_variation: FloatProperty(
        name="Pedicel Width Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random thickness on each pedicel, as a fraction of its own",
    )
    pedicel_bend: FloatProperty(
        name="Pedicel Bend", default=0.7, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How strongly each pedicel curves. 0 gives straight spokes; higher "
        "makes it leave the stem sideways and turn up harder on its way to the flower, "
        "the way a real corymb does. A pedicel never swings wider than its own flower. "
        "This is shape only and does NOT move the flower - the pedicel arcs between the "
        "same two endpoints, so Length and Angle still decide where the flower sits",
    )
    inflorescence_young_size: FloatProperty(
        name="Young Size", default=0.2, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="How big an inflorescence is the moment it is laid down, as a share "
        "of mature - its flower spacing, flower size and pedicel length and width all "
        "start here and grow to full as it develops. Only the youngest racemes (started "
        "near the end of the season) show it; a mature one is always full size. The "
        "curve beside it shapes the growth from this start to full",
    )
    inflorescence_internode_scale: FloatProperty(
        name="Flower Spacing", default=0.5, min=0.02, max=2.0, subtype='FACTOR', update=_changed,
        description="Spacing between flowers along the axis, relative to the leafy stem's own "
        "internode length. Below 1 packs the flowering region tighter than the stem below it, "
        "which is how most inflorescences sit",
    )
    bloom_order: EnumProperty(
        name="Bloom Order", update=_changed,
        items=(
            ('ACROPETAL', "Bottom first", "Lowest flowers open first and are furthest along"),
            ('BASIPETAL', "Top first", "Highest flowers open first"),
            ('CENTRIPETAL', "Outside first", "Outer flowers open first, working inward"),
            ('CENTRIFUGAL', "Middle first", "Central flowers open first, working outward"),
        ),
        default='ACROPETAL',
        description="Which flowers open first. Later ones are still opening, so they read "
        "smaller and tighter",
    )


    # Leaf placement (on stems/branches, start/end/slide along an axis) was a
    # growth-time gate; it is now a render-time component mask (e.g. the
    # On Branch channel), so growth emits foliage on every vegetative node.

    # -- Form --------------------------------------------------------------
    internode_length: FloatProperty(
        name="Spacing Base", default=0.045, min=0.001, unit='LENGTH', update=_changed,
        description="The inflorescence's spacing base: the distance between flowers is this "
        "times Flower Spacing (Branch Spacing on a branched rachis), the same on every stem "
        "and branch. Stem and branch internodes are the Branching levels' Length instead",
    )
    leaf_angle: FloatProperty(
        name="Leaf Angle", default=42.0, min=0.0, max=90.0, update=_changed,
        description="Angle a leaf leaves the stem at, measured from the stem",
    )
    leaf_size: FloatProperty(
        name="Leaf Size", default=1.0, min=0.0, update=_changed,
        description="Size of a fully expanded leaf. Young leaves near a growing tip are "
        "automatically smaller",
    )
    leaf_size_variation: FloatProperty(
        name="Variation", default=0.15, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How much leaf size varies leaf to leaf",
    )
    stem_smoothness: IntProperty(
        name="Smoothness", default=4, min=1, soft_max=8, max=16, update=_changed,
        description="How densely rings are placed along the stems. The stem is a smooth "
        "curve through every node; at 4 its faces are close to square, so bends read "
        "round and the welded junctions shade cleanly. Higher costs geometry",
    )
    stem_weld: BoolProperty(
        name="Weld", default=True, update=_changed,
        description="Merge every stem and branch into one watertight mesh, joined at the "
        "junctions, instead of separate tubes overlapping each other",
    )
    light_stalks: BoolProperty(
        name="Light Stalks", default=False, update=_changed,
        description="Build pedicels with four sides, and give stems too thin to show bark "
        "fewer rings on their straight runs. For a plant with thousands of flowers. Branch "
        "welds and bends keep their density",
    )
    trunk_flare: FloatProperty(
        name="Flare", default=0.5, min=0.0, soft_max=2.0, subtype='FACTOR',
        update=_changed,
        description="How much the base of each stem widens into the ground, as a fraction of "
        "its radius. A stronger flare also runs higher up the stem",
    )
    flare_lobes: IntProperty(
        name="Lobes", default=5, min=0, soft_max=8, max=16, update=_gn_display_changed,
        description="Buttress lobes around the flare, like the roots swelling out of a "
        "trunk's base. 0 keeps the flare round",
    )
    stem_gnarl: FloatProperty(
        name="Gnarl", default=0.3, min=0.0, soft_max=1.0, subtype='FACTOR',
        update=_gn_display_changed,
        description="How much the stems wander and kink between nodes instead of running "
        "straight, at a scale of a few trunk widths. Leaves and flowers move with them",
    )
    bark_noise: FloatProperty(
        name="Bark", default=0.3, min=0.0, soft_max=1.0, subtype='FACTOR',
        update=_changed,
        description="Surface unevenness of the stems: broad lumps sized to each stem's own "
        "thickness. Fine bark grain belongs in the bark texture",
    )
    stem_twist: FloatProperty(
        name="Twist", default=0.1, min=-2.0, max=2.0, soft_min=-1.0, soft_max=1.0,
        update=_changed,
        description="How far the stem turns about its own axis as it grows, in turns per 20 "
        "radii of length. Spirals the bark texture, ribbed or winged sections and the flare "
        "lobes. Negative turns the other way",
    )
    uv_tiles_around: IntProperty(
        name="Tiles Around", default=2, min=1, soft_max=8, max=64, update=_changed,
        description="Retired (UV Size replaced it); kept so older files still load",
    )
    uv_size: FloatProperty(
        name="UV Size", default=0.025, min=0.0005, soft_max=0.5, subtype='DISTANCE',
        unit='LENGTH', precision=4, update=_changed,
        description="The world size of one UV tile. Every stem, branch and stalk is "
        "textured at this one scale -- square texels everywhere, so a thin stalk shows the "
        "same size checker squares as the thickest stem",
    )
    uv_checker: BoolProperty(
        name="UV Checker", default=False, update=_uv_checker_changed,
        description="Show the stems with a checker texture instead of their material, to "
        "judge the UVs. Switches the viewport to texture colouring",
    )
    stem_fillet: FloatProperty(
        name="Weld Range", default=0.5, min=0.0, soft_max=2.0, update=_changed,
        description="How round each weld is: the radius of the fillet that blends a branch "
        "into its parent, in branch radii (smaller in a shallow crotch). Small keeps the join "
        "tight; large blends it out over more of the stem. 0 leaves a hard crease",
    )
    droop: FloatProperty(
        name="Droop", default=0.35, min=0.0, max=2.0, update=_changed,
        description="How far the grown plant bends under its own weight. This is separate "
        "from Gravitropism, which changes how the plant grows - this bends the structure "
        "after it has grown, without changing what is attached to what. A heavier flower "
        "makes its own stem nod further",
    )
    structural_preset: EnumProperty(
        name="Preset", default='CUSTOM', update=_structural_preset_changed,
        description="A named starting point for the physical controls below. Picking one "
        "stamps its stiffness, variance, leaf and flower weight and stiffness curve, then "
        "snaps back to Custom - so the dropdown never claims a preset you have since edited",
        items=(
            ('CUSTOM', "Custom", "Your own values, below"),
            ('HERB_UPRIGHT', "Herbaceous - Upright", "A normal upright perennial stem"),
            ('HERB_LAX', "Herbaceous - Lax", "Soft, lax stem with a gently softening tip"),
            ('CLIMBING', "Herbaceous - Climbing", "Very flexible, a vine or scrambler"),
            ('GRASS_CULM', "Grass Culm", "Stiff, light, upright"),
            ('REED', "Reed", "Stiffer than a grass culm"),
            ('SUCCULENT', "Succulent", "Soft tissue but heavy - a fleshy stem"),
            ('SAPLING', "Woody - Sapling", "Young wood, stiff but still bends"),
            ('SHRUB', "Woody - Shrub", "A firm woody shrub"),
            ('RIGID', "Woody - Rigid", "Near-hardwood, barely bends"),
            ('WEEPING', "Weeping", "Rigid base, soft tip - the lamppost / weeping habit"),
        ),
    )
    stem_stiffness: FloatProperty(
        name="Stem Stiffness", default=0.75, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How strongly stems resist bending: a real material stiffness (Young's "
        "modulus) on a log scale, 0 is 5 MPa, 0.5 is 160 MPa, 1 is 5 GPa. Soft herbaceous "
        "stems sit around 0.1-0.4, grass culms and woody stems around 0.8-1. Thick parts "
        "already resist far more than thin ones, so raising this mostly straightens the tips "
        "and the outer branches",
    )
    flowering_droop: FloatProperty(
        name="Peduncle Droop", default=0.0, min=0.0, max=1.0, update=_changed,
        description="How far the stretch of stem carrying the flowers turns over, on top of "
        "Droop. 0.5 brings it to horizontal and 1 hangs it straight down - it is an angle, so "
        "it means the same thing on every plant whatever the flowers weigh. Raising it "
        "lengthens the arc rather than tightening it: the stretch that has given way grows "
        "and the turn grows with it, so the bend holds one radius and the flower swings "
        "further out and down. Where along the stem it happens is Rigid Base; how far it may "
        "reach at most is Peduncle Reach",
    )
    peduncle_reach: FloatProperty(
        name="Peduncle Reach", default=0.0, min=0.0, soft_max=0.5,
        unit='LENGTH', update=_changed,
        description="How far below the lowest flower the drooping stretch reaches, measured "
        "as real stem length. This spreads the turn rather than adding to it: at 0 the whole "
        "bend happens in one node and the stem hooks sharply, and longer values curve it over "
        "a longer run to the same final angle. A solitary flower sits on the very tip, so it "
        "needs some of this to have a stalk to bend at all",
    )
    peduncle_resolution: IntProperty(
        name="Bend Resolution", default=1, min=1, soft_max=8, update=_changed,
        description="How many segments each internode in a hard-bending stretch is split "
        "into. A bend can only be as smooth as the stem is dense, and on a solitary flower "
        "the drooping stretch is a couple of nodes at the very top - so a big turn across "
        "them reads as two sharp facets rather than a curve.\n\n"
        "Two stretches get it: the drooping region below the lowest flower, and the whole "
        "of every inflorescence BRANCH - a compound umbel's ray is only a couple of nodes "
        "long and is bent by its own Bend, its Droop and the plant's, all landing on the "
        "same one or two joints.\n\n"
        "The rest of the plant is untouched. 1 changes nothing",
    )
    rigid_base: FloatProperty(
        name="Rigid Base", default=0.0, min=0.0, max=0.99, subtype='FACTOR',
        update=_changed,
        description="How much of each stem, from the ground up, stays stiff and unbending. 0 "
        "lets the whole stem curve, which is a plant folding over; 0.9 holds the bottom 90% "
        "rigid and puts the entire bend in the top tenth, so a heavy head nods only at the very "
        "top, like a poppy on its stalk. This only moves the bend, it does not reduce it - the "
        "same curvature is "
        "packed into a shorter stretch, so Droop and the organ weights still set how far it "
        "goes",
    )
    leaf_weight: FloatProperty(
        name="Leaf Weight", default=8.0e-5, min=0.0, soft_max=2.0e-3, precision=5,
        update=_changed,
        description="How heavy a full-size leaf is, on the same relative scale as Flower "
        "Weight. Leaf mass is small against stem stiffness on most plants, so this reads far "
        "more subtly than the flower's does",
    )
    flower_weight: FloatProperty(
        name="Flower Weight", default=4.0e-4, min=0.0, soft_max=1.0e-2, precision=5,
        update=_changed,
        description="How heavy a full-size flower or head is. This is now the main thing that "
        "nods a blooming tip: a heavy head on a thin stem hangs, a light one stands up. Flower "
        "mass grows with the CUBE of Flower Size, so doubling a head's size makes it eight "
        "times heavier - size and weight together decide the droop",
    )
    fruit_weight: FloatProperty(
        name="Fruit Weight", default=1.5e-3, min=0.0, soft_max=2.0e-2, precision=5,
        update=_changed,
        description="How heavy a full-size ripe fruit is, on the same relative scale as Flower "
        "Weight - heavier, which is what bows a fruiting branch. Like the flower, mass grows "
        "with the CUBE of Fruit Size. Fruit exists only inside the calendar fruit window, so a "
        "branch droops as its fruit ripens and springs back after fruit-drop",
    )
    stiffness_variation: FloatProperty(
        name="Stiffness Variance", default=0.0, min=0.0, soft_max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random stem-to-stem variation in stiffness across a clump, as a fraction. "
        "Drawn once per stem, so a stand of the same plant bends differently one to the next "
        "without any single stem becoming ragged along its own length",
    )
    gravitropism: FloatProperty(
        name="Gravitropism", default=0.35, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How strongly a stem corrects back toward the angle it is trying to hold. "
        "Higher makes stems stiff and true; lower lets them wander",
    )
    circumnutation: FloatProperty(
        name="Circumnutation", default=0.0, min=0.0, soft_max=15.0, update=_changed,
        description="How far the growing tip nods off-axis each step in a coiling, helical "
        "sweep -- the spiralling seen in growth timelapses. Unlike Irregularity, which "
        "wanders at random, this winds a coherent helix. Its size is fought by "
        "Gravitropism, so raise Autotropism to see it on an upright plant. 0 is off",
    )
    circumnutation_rate: FloatProperty(
        name="Circumnutation Rate", default=137.5, min=-360.0, max=360.0, update=_changed,
        description="How far the nod turns around the stem each step -- the pitch of the "
        "helix, its sign the handedness (left- vs right-winding). Does nothing while "
        "Circumnutation is 0",
    )
    elongation_mode: EnumProperty(
        name="Internode Elongation", items=ELONGATION_MODE_ITEMS,
        get=_get_elongation_mode, set=_set_elongation_mode,
        description="How each internode stretches to its full length after it forms. "
        "Picking one sets the Internode Elongation curve and Duration; edit either and "
        "it reads Custom")
    elongation_steps: IntProperty(
        name="Elongation Duration", default=5, min=1, soft_max=20, update=_changed,
        description="Growth steps an internode takes to reach its full length after it "
        "forms (one node forms per step): 1 = done before the next forms, higher = "
        "several stretching at once")
    leaf_expansion_steps: IntProperty(
        name="Leaf Expansion Duration", default=7, min=1, soft_max=30, update=_changed,
        description="Growth steps a leaf takes to grow from a tiny primordium to its full "
        "size, along the Leaf Expansion curve -- longer keeps young leaves small for "
        "longer")
    autotropism: FloatProperty(
        name="Autotropism", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How much older stem straightens back to true as it ages, so the wander "
        "stays a live nodding tip on a tidy stem instead of a wobble all the way down. "
        "Higher keeps a shorter, tidier live zone at the very tip. 0 is off",
    )

    # -- Leaf orientation ---------------------------------------------------
    leaf_yaw: FloatProperty(
        name="Yaw", default=0.0, min=-180.0, max=180.0, update=_changed,
        description="Swings each leaf left or right within its own plane, without "
        "tilting it. The blade stays as flat as it was and simply no longer points "
        "straight out from the stem",
    )
    leaf_yaw_variation: FloatProperty(
        name="Yaw Variation", default=0.0, min=0.0, soft_max=90.0, update=_changed,
        description="How far individual leaves swing either side of the Yaw angle. "
        "Real leaves are not all turned the same way, and a few degrees of scatter is "
        "what separates a grown plant from a modelled one",
    )
    leaf_roll: FloatProperty(
        name="Roll", default=0.0, min=-180.0, max=180.0, update=_changed,
        description="Turns each leaf about its own length, pivoting on the midrib, so "
        "the face stops being flat to the sky without the blade pointing anywhere new",
    )
    leaf_roll_variation: FloatProperty(
        name="Roll Variation", default=0.0, min=0.0, soft_max=90.0, update=_changed,
        description="How far individual leaves roll either side of the Roll angle",
    )

    # Blade shape (curl/cup/gravity/wave/twist) was removed: leaves arrive
    # pre-shaped from the leaf tool.

    # -- Environment (Stage 3) ---------------------------------------------
    # Every default here is the neutral value, so a plant responds to nothing
    # until asked and grows exactly as it did before Stage 3 existed.
    light_seeking: FloatProperty(
        name="Light Seeking", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How far growing tips turn toward the light each step. Applied after "
        "Gravitropism, so a shaded shoot leans out from under whatever is shading it even "
        "though gravity would hold it upright. 0 ignores light direction",
    )
    flower_heliotropism: FloatProperty(
        name="Flower Heliotropism", default=0.0, min=0.0, max=1.0,
        subtype='FACTOR', update=_changed,
        description="How far each flower turns its face toward the Sun Object, after growth. "
        "The nodding-sunflower case: the stem stays where gravity left it, the head turns up "
        "to the sun and its own stalk curves with it (a stalk a branch leaves stays "
        "straight); 0 is off",
    )
    leaf_heliotropism: FloatProperty(
        name="Leaf Heliotropism", default=0.0, min=0.0, max=1.0,
        subtype='FACTOR', update=_changed,
        description="How far each leaf turns its blade to face the sun, after growth. Set Sun "
        "Object below to the thing to track. Unlike an Align Up effector this aims each blade "
        "at the sun rather than rotating the whole leaf set; 0 is off",
    )
    sun_object: PointerProperty(
        type=bpy.types.Object, update=_changed, name="Sun Object",
        description="The object leaves turn to face, for Leaf Heliotropism (each flower "
        "component picks its own). They aim at wherever it sits relative to the plant, so "
        "move it to place the sun -- any object works (an empty, a lamp, a mesh). Empty "
        "means no heliotropism",
    )
    obstacle_collection: PointerProperty(
        type=Collection, update=_changed,
        description="Objects this plant grows around. Any mesh will do, and closed shapes "
        "work best. They block growth and cast shade. Empty means nothing is in the way",
    )
    obstacle_clearance: FloatProperty(
        name="Clearance", default=0.0, min=0.0, soft_max=0.5, unit='LENGTH', update=_changed,
        description="How far from a surface this plant keeps its growing tips. Tips curve "
        "away as they approach, and one that runs out of room stops there instead of growing "
        "through. 0 turns obstacle response off entirely",
    )
    ground_object: PointerProperty(
        type=bpy.types.Object, update=_changed, name="Ground",
        poll=_is_mesh,
        description="A terrain mesh the plant rests on: a stem that bends down to it lies "
        "along its surface instead of the flat ground at the plant's base. Only the bend "
        "reads it -- growth is not steered. Empty means flat ground",
    )
    effector_collection: PointerProperty(
        type=Collection, update=_changed,
        description="Effector empties that steer this plant by hand - attract, repel, "
        "forbid or aim. Unlike Light and Obstacles there is nothing to enable first: an "
        "effector in this collection acts straight away. Empty means nothing steers it",
    )

    # -- Variation ---------------------------------------------------------
    seed: IntProperty(
        name="Seed", default=0, min=0, update=_changed,
        description="Master seed: changes the individual plant without changing the "
        "species, every level's and component's own Seed mixed with it. The same seeds "
        "and settings always grow the same plant",
    )

    # -- Foliage assets ----------------------------------------------------
    leaf_asset_collection: PointerProperty(
        type=Collection, update=_changed,
        description="Collection of leaf objects to place at leaf positions. One is picked per "
        "leaf. Empty means no leaves are placed",
    )
    bract_asset_collection: PointerProperty(
        type=Collection, update=_changed,
        description="A third leaf shape, used for the bracts that subtend each whorl of "
        "flowers. Bracts are modified leaves, so they take this plant's own leaf angle and "
        "every blade deformer - only the mesh and the size differ. Empty means bracts are "
        "drawn from the ordinary Leaf collection instead of vanishing",
    )
    catkin_length: FloatProperty(
        name="Catkin Length", default=0.0, min=0.0, soft_max=0.3, unit='LENGTH',
        update=_changed,
        description="How long each catkin is. 0 means none.\n\n"
        "A catkin is built as a hanging curve rather than placed as a model, so it sags "
        "under its own weight, keeps the length you give it, and bends when the branch "
        "carrying it moves. It hangs from a node on its own stalk - it is not the tip of "
        "the branch",
    )
    catkin_length_variation: FloatProperty(
        name="Catkin Length Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Length scatter between catkins. Drawn per catkin rather than per "
        "stalk, so the ones sharing a stalk differ from each other and not just from the "
        "next stalk's. Real catkins on one twig are visibly unequal",
    )
    catkin_count: IntProperty(
        name="Catkins per Stalk", default=3, min=1, max=12, update=_changed,
        description="How many catkins hang from one stalk. Hazel, birch and willow all "
        "carry them in small bunches rather than singly.\n\n"
        "Separate from the leaf arrangement above, deliberately: how many catkins share a "
        "stalk and how the foliage is arranged are independent, and a whorled arrangement "
        "used to be the only way to get bunches - which whorled the leaves too",
    )
    catkin_count_variation: FloatProperty(
        name="Catkins per Stalk Variance", default=0.0, min=0.0, max=1.0,
        subtype='FACTOR', update=_changed,
        description="Scatter in how many catkins each stalk carries. Hazel's bunches run "
        "two to four, and a plant whose every stalk carries exactly three reads as instanced",
    )
    catkin_width: FloatProperty(
        name="Catkin Width", default=0.004, min=0.0001, soft_max=0.02, unit='LENGTH',
        update=_changed,
        description="How thick the catkin body is at its fattest. It swells from nothing at "
        "the stalk to this in the middle and closes again at the tip, which is what gives it "
        "rounded ends rather than a tube cut off flat",
    )
    catkin_stalk: FloatProperty(
        name="Catkin Stalk", default=0.008, min=0.0, soft_max=0.06, unit='LENGTH',
        update=_changed,
        description="The bare stalk between the node and the start of the catkin body. This "
        "is what makes a catkin read as hanging off a twig rather than sprouting from it",
    )
    catkin_noise: FloatProperty(
        name="Catkin Noise", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Bends each catkin slightly out of true, so a bunch does not read as "
        "several copies of one tube. Subtle by design - it bends the whole shape gently "
        "rather than roughening the surface, and grows from nothing where the catkin "
        "attaches to full at its free end",
    )
    catkin_stalk_variation: FloatProperty(
        name="Catkin Stalk Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Length scatter between one stalk and the next, so the bunches sit at "
        "different distances from the twig. Separate from Length's own variance, which "
        "scatters the catkins on a stalk rather than the stalks themselves",
    )
    catkin_droop_variation: FloatProperty(
        name="Catkin Hang Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random scatter in how far each catkin hangs, around the Orientation "
        "angle above. A per-catkin variance, which an angle stated for a whole "
        "inflorescence has no equivalent of",
    )
    inflorescence_on_trunk: BoolProperty(
        name="On Trunk", default=False, update=_changed,
        description="Whether the main stem bears inflorescences",
    )
    inflorescence_on_branches: BoolProperty(
        name="On Branches", default=True, update=_changed,
        description="Whether side branches bear them. Hazel and birch carry theirs near the "
        "tips of the previous year's side twigs, which is what the default says",
    )
    inflorescence_start: FloatProperty(
        name="Start", default=0.5, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How far up a stem they begin, 0 at its base and 1 at its tip",
    )
    inflorescence_end: FloatProperty(
        name="End", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="How far up a stem they stop",
    )
    flower_asset_collection: PointerProperty(
        type=Collection, update=_changed,
        description="Collection of flower objects to place at flower positions",
    )
    bloom_mode: EnumProperty(
        name="Bloom", default='AGE', update=_bloom_display_changed,
        items=(
            ('AGE', "From Age", "Flowers open as the plant matures: Progress is worked out "
             "from Age and the point in its life this species starts flowering. Scrub Age "
             "and the plant grows and blooms together"),
            ('MANUAL', "Manual", "Progress is the slider below, independent of Age. This is "
             "the one to keyframe when the bloom needs its own timing"),
            ('STATIC', "Static", "No timeline at all: each flower shows the stage its own "
             "bloom order calls for, so the first to open is out and the last is a bud"),
        ),
        description="Where each flower's openness comes from",
    )
    bloom_progress: FloatProperty(
        name="Bloom Progress", default=0.35, min=0.0, max=1.0, subtype='FACTOR',
        update=_bloom_display_changed,
        description="Where the band of open flowers has reached, 0 to 1. Keyframe this to "
        "animate the plant coming into flower - it updates without regrowing, so it plays "
        "back at full speed",
    )
    bloom_spread: FloatProperty(
        name="Bloom Spread", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
        update=_bloom_display_changed,
        description="How staggered the opening is. 0 opens every flower in lockstep; 1 means "
        "the last flower in the bloom order does not begin until Progress reaches 1, so the "
        "opening travels up the inflorescence as a wave. Under From Age the stagger already "
        "comes from when each flower was actually born, so this reads instead as how quickly "
        "a flower opens once it appears - higher leaves fewer buds at the tip",
    )
    bloom_continuous: BoolProperty(
        name="Smooth bloom stage blending", default=False, update=_bloom_display_changed,
        description="Blend smoothly between neighbouring bloom stages instead of snapping "
        "from one to the next. Every stage must have identical topology - stages baked from "
        "one flower do; a set of separately modelled flowers will not, and will scramble",
    )
    flower_spread: FloatProperty(
        name="Spread", default=360.0, min=0.0, max=360.0, update=_changed,
        description="How much of the stem's circumference the flowers occupy. 360 is the "
        "full circle the leaf arrangement already gives; narrow it to crowd them onto one "
        "side, the way a foxglove does, and 0 puts every flower on a single line",
    )
    flower_spread_variation: FloatProperty(
        name="Spread Variance", default=0.0, min=0.0, max=180.0, update=_changed,
        description="Random scatter on each flower's own position round the stem, in "
        "degrees either way",
    )
    flower_pitch: FloatProperty(
        name="Flower Pitch", default=0.0, min=-90.0, max=90.0, update=_changed,
        description="Which way each flower faces, stated like Pedicel Angle: 0 straight out "
        "sideways, positive turned up toward the tip, negative nodding down, which is what "
        "makes a foxglove bell hang. Set it equal to Pedicel Angle to point straight along "
        "the pedicel. Separate from the pedicel's own angle, so a hanging flower and an "
        "outward-facing one can share one pedicel shape",
    )
    flower_pitch_variation: FloatProperty(
        name="Flower Pitch Variance", default=0.0, min=0.0, max=90.0, update=_changed,
        description="Random tilt on each flower, in degrees either way",
    )
    flower_scale: FloatProperty(
        name="Flower Size", default=1.0, min=0.0, update=_changed,
        description="Size of each flower. The curve beside it grades the size along the "
        "inflorescence and each flower's own opening is folded in on top, so this is the "
        "one size control -- the same way Leaf Size works",
    )
    flower_size_variation: FloatProperty(
        name="Flower Size Variance", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_changed,
        description="Random size on each flower, as a fraction of its own size",
    )

    fruit_asset_collection: PointerProperty(
        type=Collection, update=_changed,
        description="Collection of fruit objects to place where flowers have set fruit. One "
        "is picked per fruit, the way leaves are. A fruit appears only inside the calendar "
        "fruit window; empty means no fruit is placed",
    )
    fruit_size: FloatProperty(
        name="Fruit Size", default=1.0, min=0.0, update=_changed,
        description="Size of each fruit, a multiplier on the fruit asset (mesh-driven, like "
        "Flower Size). A fruit is its own organ at its own size, not the flower it set from",
    )
    fruit_set: FloatProperty(
        name="Fruit Set", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_changed,
        description="Fraction of flowers that set fruit, rolled per flower. The rest are "
        "unpollinated and drop at fruit-set. 1 means every flower fruits",
    )

    # -- Components. Multiple instances per
    # type, each a collection filtered by its own inline stack of attribute windows.
    # Each component claims the candidate points its filters keep (first-claim, in
    # list order); the bridge routes a type through one master collection + one
    # Instance-on-Points and bakes `render_drop` for points no component claims.
    components: CollectionProperty(type=PLANTGEN_PG_component)
    components_active: IntProperty(default=0, update=_component_context_changed)
    # Growth maps: per-point multipliers on growth parameters (branch density /
    # angle / length, internode length, gravitropism), each a Driver channel
    # through its own curve -- they change how the plant grows.
    growth_maps: CollectionProperty(type=PLANTGEN_PG_growth_map)
    growth_maps_active: IntProperty(default=0)
    # Parameter mapping: maps on parameters with
    # no per-element evaluator (once per plant), the Mapping list's rows over
    # every kind of map, and each owner's fold in it.
    param_maps: CollectionProperty(type=PLANTGEN_PG_param_map)
    map_rows: CollectionProperty(type=PLANTGEN_PG_map_row)
    map_rows_active: IntProperty(default=0, update=_map_row_selected)
    map_folds: CollectionProperty(type=PLANTGEN_PG_map_fold)
    # Texturing's Materials list: levels, then each asset component and its
    # materials (`ui_materials`; UI state, rebuilt from the plant).
    texture_rows: CollectionProperty(type=PLANTGEN_PG_texture_row)
    texture_rows_active: IntProperty(default=0)
    variations: PointerProperty(type=PLANTGEN_PG_variations)
    var_mutes: CollectionProperty(type=PLANTGEN_PG_var_mute)
    curve_mutes: CollectionProperty(type=PLANTGEN_PG_var_mute)
    # Random Range rows kept in the Mapping list at a Range of 0 (only the
    # list's - removes one): owner / owner_uid / param, `value` unused.
    var_keeps: CollectionProperty(type=PLANTGEN_PG_var_mute)
    mapping_details_open: BoolProperty(
        name="Map Settings", default=True, options={'HIDDEN'},
        description="Show the selected map's settings")
    plant_color: FloatVectorProperty(
        name="Colour", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.60, 0.64, 0.71), update=_swatch_changed,
        description="The plant's own label colour: its row in the Components tree and "
        "the maps on its parameters")
    # The branching stack: level 0 the stems,
    # each next level the branches on the one before. Empty until the plant's
    # profile first seeds it (`bridge.profile_from_object` / `apply_profile`).
    branch_levels: CollectionProperty(type=PLANTGEN_PG_branch_level)
    branch_levels_active: IntProperty(default=0, update=_level_active_changed)
    # The Plant Structure tree: which kind of row is selected (the plant root,
    # a level -- `branch_levels_active` -- or a component -- `components_active`),
    # and for a component the level row it was picked under (a component linked
    # to several levels shows under each; 0 = the plant root). View state only.
    structure_focus: EnumProperty(
        items=[('PLANT', "Plant", "", 0), ('LEVEL', "Level", "", 1),
               ('COMPONENT', "Component", "", 2),
               ('LIBRARY', "Unplaced Component", "", 3)],   # placed on no level
        default='LEVEL', options={'HIDDEN'})
    structure_parent_uid: IntProperty(default=0, options={'HIDDEN'})
    placements: CollectionProperty(type=PLANTGEN_PG_placement)
    structure_place: IntProperty(default=-1, options={'HIDDEN'})
    tree_rows: CollectionProperty(type=PLANTGEN_PG_tree_row)
    tree_active: IntProperty(default=0, update=_tree_active_changed)
    structure_plant_expanded: BoolProperty(update=_structure_changed,
        name="Expanded", default=True,
        description="Show the plant's components and levels in the tree")
    # The Components panel's settings boxes below the tree, each foldable.
    structure_level_open: BoolProperty(
        name="Level Settings", default=True, options={'HIDDEN'},
        description="Show the selected level's settings")
    structure_component_open: BoolProperty(
        name="Component Settings", default=True, options={'HIDDEN'},
        description="Show the selected component's settings")
    mask_preview_level: IntProperty(
        default=-1, options={'HIDDEN'},
        description="The branching level whose active mask the Mask Preview shows "
        "(-1: the active component's)")
    # Classification: plant-level custom attributes (derived channels) and named
    # classes (reusable, chainable mask stacks) any mask or map can read.
    custom_attrs: CollectionProperty(type=PLANTGEN_PG_custom_attr)
    custom_attrs_active: IntProperty(default=0)
    classes: CollectionProperty(type=PLANTGEN_PG_class)
    classes_active: IntProperty(default=0)
    # The Classify view's two axes (`classify_overlay`). View-only: they change
    # what the scatter plots, never the plant, so no `_changed` and no place in
    # `bridge.growth_signature`.
    classify_x: EnumProperty(
        name="X Axis", items=MASK_CHANNEL_ITEMS, default="height",
        update=_redraw_viewports,
        description="The attribute the Classify view plots across")
    classify_y: EnumProperty(
        name="Y Axis", items=MASK_CHANNEL_ITEMS, default="branch_depth",
        update=_redraw_viewports,
        description="The attribute the Classify view plots up")
    classify_x_attr: StringProperty(
        name="X Attribute", default="", update=_redraw_viewports,
        description="Which of this plant's custom attributes the X axis plots")
    classify_y_attr: StringProperty(
        name="Y Attribute", default="", update=_redraw_viewports,
        description="Which of this plant's custom attributes the Y axis plots")
    classify_x_class: StringProperty(
        name="X Class", default="", update=_redraw_viewports,
        description="Which of this plant's classes the X axis plots")
    classify_y_class: StringProperty(
        name="Y Class", default="", update=_redraw_viewports,
        description="Which of this plant's classes the Y axis plots")

    curve_group: PointerProperty(
        type=NodeTree,
        description="Holds this plant's own curve widgets. Blender has no property type "
        "for a curve, so each object owns a small node group whose Float Curve nodes "
        "carry the real curve data",
    )

    # -- Read-only stats, written by bridge.regrow(). Deliberately no
    # update= : regrow() writes these, so a callback would re-mark the object
    # dirty and loop forever.
    last_phytomers: IntProperty(name="Nodes", default=0)
    last_axes: IntProperty(name="Stems & Branches", default=0)
    last_leaves: IntProperty(name="Leaves", default=0)
    last_flowers: IntProperty(name="Flowers", default=0)
    last_fruit: IntProperty(name="Fruit", default=0)
    last_height: FloatProperty(name="Height", default=0.0, unit='LENGTH')
    last_width: FloatProperty(name="Width", default=0.0, unit='LENGTH')
    # -- Diagnostics. Buds and per-order breakdowns are the half of the picture
    # the viewport overlay structurally cannot show: a dormant or aborted bud
    # never becomes geometry, so "93 buds frozen forever" -- one of the
    # failures that triggered the rebuild -- is invisible to any attribute
    # colouring and only ever shows up as a number.
    last_buds_active: IntProperty(name="Active", default=0)
    last_buds_dormant: IntProperty(name="Dormant", default=0)
    last_buds_released: IntProperty(name="Released", default=0)
    last_buds_aborted: IntProperty(name="Aborted", default=0)
    last_axes_by_order: StringProperty(name="Axes by Order", default="")
    last_phytomers_by_order: StringProperty(name="Nodes by Order", default="")
    last_flowering_axes: IntProperty(name="Flowering Axes", default=0)
    last_flowering_age: FloatProperty(
        name="Flowering Age", default=0.0, min=0.0, max=1.0,
        description="The Age at which this species starts flowering, read off its own "
        "Flowering Onset curve. Bloom's From Age mode measures Progress from here",
    )
    last_bloom_window: FloatProperty(
        name="Bloom Window Position", default=-1.0,
        description="Position of the current Day within this species' calendar bloom "
        "window (0 as it opens, 1 as it closes), or -1 when the species has no bloom "
        "calendar. From Age drives Bloom Progress from this when it is set, so the bloom "
        "opens across the calendar window -- 'blooms for two weeks' is the window width",
    )
    last_bracts: IntProperty(name="Bracts", default=0)
    last_flower_spacing: FloatProperty(name="Flower Spacing", default=0.0)
    last_catkins: IntProperty(name="Catkins", default=0)

    last_solid_voxels: IntProperty(name="Obstacle Voxels", default=0)
    """How much solid matter the last grow actually found. An obstacle
    collection that resolved to nothing looks identical to one that is working
    until you can see this number."""

    # -- Senescence counts (WS3 Phase 4). Zero unless a plant senesces or is
    # pruned, so the panel rows appear only when there is something to report --
    # and a SHED item is invisible in the viewport, so a number is the only way
    # to see it, the same reason the bud counts exist.
    last_dead_axes: IntProperty(name="Dead Branches", default=0)
    last_shed_axes: IntProperty(name="Shed Branches", default=0)
    last_shed_organs: IntProperty(name="Shed Leaves/Flowers", default=0)


# --- Flower authoring, ported onto the flower component (additive) ----------
# The flower controls are being moved from the plant (PLANTGEN_PG_growth)
# onto the flower component type. The
# plant keeps its own copies as the legacy fallback until the port completes;
# the component's copies are the SAME property annotations, shared here so the
# two can never drift. When a plant has a flower component,
# bridge.profile_from_object sources the inflorescence spec from it instead of
# the plant props (only for the props listed here, so this grows field by field).
# The growth PG is defined after the component PG, so the copy runs here, once
# both classes exist and before registration.
# Every profile-mapped flower field (matches bridge._FLOWER_FIELDS): the whole
# `inflorescence_*` spec, the pedicel/catkin/head/cluster groups, and a handful
# of `flower_*`/`bloom_order` scalars. (Bract size is not here: it is a Bract
# component's own Size.) Bloom *display* (mode/progress/spread) is read by the
# GN layer rather than the profile, so it is ported in a later phase.
_FLOWER_PREFIXES = ("inflorescence_", "pedicel_", "catkin_", "head_", "cluster_")
_FLOWER_EXTRA_PROPS = (
    "flower_count", "flower_count_variation", "flower_size_variation",
    "flower_pitch", "flower_pitch_variation", "flower_spread",
    "flower_spread_variation", "flower_spacing_variation", "flower_weight",
    "bloom_order", "flower_heliotropism",
)
_PORTED_FLOWER_PROPS = tuple(
    n for n in PLANTGEN_PG_growth.__annotations__
    if n.startswith(_FLOWER_PREFIXES) or n in _FLOWER_EXTRA_PROPS)
for _name in _PORTED_FLOWER_PROPS:
    PLANTGEN_PG_component.__annotations__[_name] = \
        PLANTGEN_PG_growth.__annotations__[_name]

# Fruit port: the same shared-annotation copy for the fruit growth scalars. The
# primary fruit component (first placed On Organs) owns them; its Size is the
# fruit size (the plant's Fruit Size merged into it). The plant copies stay as
# the seed source and as the values a plant with no fruit component grows with.
_PORTED_FRUIT_PROPS = ("fruit_set", "fruit_weight")
for _name in _PORTED_FRUIT_PROPS:
    PLANTGEN_PG_component.__annotations__[_name] = \
        PLANTGEN_PG_growth.__annotations__[_name]


_classes = (PLANTGEN_PG_check, PLANTGEN_PG_pin, PLANTGEN_PG_pruned,
            PLANTGEN_PG_mask_entry, PLANTGEN_PG_custom_attr, PLANTGEN_PG_class,
            PLANTGEN_PG_instance_map, PLANTGEN_PG_flower_map,
            PLANTGEN_PG_growth_map, PLANTGEN_PG_level_map,
            PLANTGEN_PG_param_map, PLANTGEN_PG_texture_row, PLANTGEN_PG_map_row,
            PLANTGEN_PG_var_mute,
            PLANTGEN_PG_map_fold,
            PLANTGEN_PG_level_link, PLANTGEN_PG_placement, PLANTGEN_PG_tree_row,
            PLANTGEN_PG_trait_column, PLANTGEN_PG_trait_cell, PLANTGEN_PG_trait_row,
            PLANTGEN_PG_component,
            PLANTGEN_PG_branch_level, PLANTGEN_PG_variations,
            PLANTGEN_PG_growth)


def register():
    from . import showcase

    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.plantgen = PointerProperty(type=PLANTGEN_PG_growth)
    # On the Scene, not an Object: a sweep makes plants rather than editing
    # one, so there is nothing for it to hang off. The items callback is
    # deliberately a function -- it reads the PropertyGroup's own RNA, which
    # does not exist until the line above has run.
    bpy.types.Scene.plantgen_showcase = EnumProperty(
        name="Parameter", items=showcase.property_items,
        description="Which control to sweep across its own range")
    bpy.types.Scene.plantgen_showcase_steps = IntProperty(
        name="Variations", default=10, min=2, max=24,
        description="How many plants to generate across the range. Ignored for "
        "on/off and dropdown controls, which use every state they have")


def unregister():
    del bpy.types.Scene.plantgen_showcase_steps
    del bpy.types.Scene.plantgen_showcase
    del bpy.types.Object.plantgen
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
