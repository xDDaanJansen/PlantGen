"""Parameter sweeps: one control walked across its own range, laid out as a
labelled row so a whole axis of the parameter surface can be judged at a look.

This exists because of a specific gap. `tests/structural.py` asserts a plant is
*sound* and `tests/inflorescence_golden.py` asserts it is *the same*; neither
has anything to say about whether it looks right, and "looks right" is the only
question left once both pass. Ten plants differing in exactly one number, side
by side, is the cheapest instrument for it: a slider whose top half does
nothing, or two enum entries that produce the same plant, are obvious in a
second and invisible one plant at a time.

**Every parameter is discovered from the PropertyGroup, and its range is read
off the property itself.** Nothing here hand-writes a list of values, and that
is not just less code -- an earlier hand-written version passed `3e6` for
`stem_stiffness`, which is the modulus physics really uses but not what the
*panel* declares (a 0-1 factor). All three values clamped to 1.0, so the grid
compared a plant against itself and read exactly like "stiffness does nothing":
a physics bug that did not exist. Reading `soft_min`/`soft_max` off the RNA
makes that class of mistake impossible rather than merely unlikely.

**The sweep varies one control on a copy of a real plant**, not on a bare
default. Tuning means asking what a slider does to *the plant you are working
on*, so every setting is copied from a template first -- including its asset
collections, without which a fresh object grows a bare stem tube that reads as
a broken species and is not one.
"""

from __future__ import annotations

import math

import bpy

from . import bridge

_MARKER = "plantgen_showcase"
_LABEL_HEIGHT = 0.055
"""Text size as a fraction of the tallest plant in the row, so a label is
legible next to a 1.6 m foxglove and next to a 0.2 m seedling without being
retuned for either."""

_MAX_PER_ROW = 12
"""A sweep is a one-dimensional axis and reads best as one row. Past this it
wraps, purely so a 20-entry enum does not run off into the distance."""

_SKIP_EXACT = {
    "rna_type", "name",
    # Written by a grow, not read by one.
    "checks", "checks_ran",
    # Not properties of the plant: a viewport overlay, a rebuild policy, and
    # the datablock holding the curves.
    "debug_view_mode", "live_update", "curve_group",
}

# `classify_` are the Classify view's axes -- a viewport overlay, not the plant.
_SKIP_PREFIX = ("last_", "classify_")

_ASSET_PROPS = ("leaf_asset_collection", "flower_asset_collection")

_UNBOUNDED_SPAN = 3.0
"""When a property declares no usable upper bound, sweep from its floor to this
multiple of the template's current value. Anchoring on the current value rather
than picking a constant is what keeps the sweep in the region the plant
actually lives in -- a plant with 0.045 m internodes and one with 0.5 m
internodes want very different ranges from the same unbounded property."""


def _is_sweepable(prop) -> bool:
    if prop.identifier in _SKIP_EXACT or prop.identifier.startswith(_SKIP_PREFIX):
        return False
    if prop.is_readonly:
        return False
    # Collections and datablock pointers are references, not quantities.
    if prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM'}:
        return False
    return getattr(prop, "array_length", 0) <= 1


def sweepable_properties():
    """Every panel control a sweep can walk, in the PropertyGroup's own
    declaration order -- which is grouped by panel section, so the dropdown
    reads in the same order as the panel rather than alphabetically."""
    rna = bpy.types.Object.bl_rna.properties["plantgen"].fixed_type
    return [p for p in rna.properties if _is_sweepable(p)]


def property_items(self=None, context=None):
    """Enum items for the picker. A callable rather than a fixed list because
    it is built from the RNA, which does not exist until registration."""
    items = []
    for prop in sweepable_properties():
        label = prop.name or prop.identifier
        detail = (prop.description or "").split(". ")[0]
        items.append((prop.identifier, label, detail or label))
    return items or [('NONE', "None", "No sweepable properties")]


_BOUND_SANITY = 1e12
"""Above this a declared limit is Blender's own placeholder, not a real one.

`math.isfinite` is *not* the test, and assuming it was produced captions like
"340282346638528859811704183484516925440.0" on the first run. A property
declared with only `min=0.0` comes back with `soft_max` at FLT_MAX (3.4e38),
which is a perfectly finite float. No control on a plant is legitimately
anywhere near this, so anything above it means "unbounded".
"""


def _bounded(value: float) -> bool:
    return math.isfinite(value) and abs(value) < _BOUND_SANITY


def _span(prop, current):
    """The low and high ends of a numeric property's sweep.

    `soft_min`/`soft_max` first, because those are precisely the range Blender
    draws the slider across -- the range a person would actually drag through,
    which is the question a sweep is asking. Hard limits second. An unbounded
    property falls back to a span around the template's own value, since
    sweeping to FLT_MAX is not a thing.
    """
    lo = prop.soft_min
    hi = prop.soft_max
    if not _bounded(lo):
        lo = prop.hard_min if _bounded(prop.hard_min) else 0.0
    if not _bounded(hi) or hi <= lo:
        base = abs(float(current)) or 1.0
        hi = lo + base * _UNBOUNDED_SPAN
    return float(lo), float(hi)


def sweep_values(prop, current, steps: int):
    """`(caption, value)` for every plant in this sweep.

    Booleans and enums ignore `steps` and use every state they have -- ten
    samples of a two-state control would be eight duplicates, and the whole
    point is that each cell differs from its neighbours.
    """
    if prop.type == 'BOOLEAN':
        return [("Off", False), ("On", True)]

    if prop.type == 'ENUM':
        return [(item.name, item.identifier) for item in prop.enum_items]

    lo, hi = _span(prop, current)
    if prop.type == 'INT':
        # One plant per distinct integer when the range is narrower than the
        # step count, rather than the same value several times over.
        span = int(round(hi)) - int(round(lo))
        count = min(steps, max(span + 1, 2))
        values = [int(round(lo + (hi - lo) * i / (count - 1))) for i in range(count)]
        seen, unique = set(), []
        for v in values:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return [(str(v), v) for v in unique]

    values = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    # Decimals from the step size, not a fixed count. A fixed 3 was fine for
    # Age but labelled a Tip Radius sweep "0.001, 0.001, 0.002" -- neighbouring
    # cells reading as the same number, which is the one thing a caption in a
    # sweep must never do.
    step = (hi - lo) / (steps - 1)
    decimals = 1 if step >= 1.0 else min(6, int(math.ceil(-math.log10(step))) + 1)
    return [(f"{v:.{decimals}f}", v) for v in values]


def find_template(context):
    """A plant to sweep around: the active one if it is a plant, otherwise any
    plant in the file that is not itself part of a sweep."""
    active = context.active_object
    # The marker test applies to the active object too, and leaving it off was
    # a real bug: after building one grid the active object is one of its own
    # plants, so the next build picked it as the template and then deleted it
    # while clearing the collection, leaving a dangling StructRNA.
    if bridge.is_plant(active) and not active.get(_MARKER):
        return active
    for obj in bpy.data.objects:
        if bridge.is_plant(obj) and not obj.get(_MARKER):
            return obj
    return None


def _capture(template):
    """Every sweepable setting on the template, as plain values.

    Read out before anything is deleted, so no Object reference is held across
    a collection clear that might include the template itself.
    """
    state = {p.identifier: getattr(template.plantgen, p.identifier)
             for p in sweepable_properties()}
    assets = {prop: getattr(template.plantgen, prop) for prop in _ASSET_PROPS}
    return state, assets, template.plantgen.curve_group


def _collection_for(name: str):
    """A collection of this sweep's own, emptied if it already exists.

    Rebuilt rather than added to, so re-running a sweep after a tuning change
    replaces it instead of stacking a second one on top -- comparing a grid
    against a stale copy of itself is worse than having no grid.
    """
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        return col
    for obj in list(col.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    return col


def _label(text: str, collection, size: float):
    curve = bpy.data.curves.new("PGS_Label", type='FONT')
    curve.body = text
    curve.size = size
    curve.align_x = 'CENTER'
    obj = bpy.data.objects.new(f"PGS_Label_{text}", curve)
    obj[_MARKER] = True
    # Standing up in the XZ plane, so captions read from the same front view
    # the plants are judged in rather than only from above.
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    collection.objects.link(obj)
    return obj


def build(context, prop_id: str, steps: int):
    """Sweep one property. Returns its collection, the plant count, and a note
    for the caller to report."""
    template = find_template(context)
    if template is None:
        return None, 0, "No plant to sweep from - add one first"

    prop = next((p for p in sweepable_properties() if p.identifier == prop_id), None)
    if prop is None:
        return None, 0, f"{prop_id} is not a sweepable property"

    state, assets, curve_group = _capture(template)
    variants = sweep_values(prop, state[prop_id], steps)
    collection = _collection_for(f"PGS_Sweep_{prop_id}")

    plants = []
    for caption, value in variants:
        obj = bpy.data.objects.new("PGS_Sweep", bpy.data.meshes.new("PGS_Sweep"))
        obj["plantgen"] = True
        obj[_MARKER] = True
        collection.objects.link(obj)
        s = obj.plantgen

        # Species first and alone: assigning it reseeds every mapped control
        # from the profile, so anything set before it is thrown away. This is
        # the documented hazard that once destroyed real hand-tuning, and the
        # ordering here is the whole defence against it. Everything the
        # template actually holds is restored immediately afterwards.
        s.species = state["species"]
        for ident, held in state.items():
            if ident != "species":
                setattr(s, ident, held)
        for asset_prop, held in assets.items():
            setattr(s, asset_prop, held)
        # Shared, not copied: the curves are held constant across a sweep by
        # definition, and one group for the row keeps them that way.
        s.curve_group = curve_group

        setattr(s, prop_id, value)
        bridge.regrow(obj)
        obj.name = f"PGS_{prop_id}_{caption}"
        plants.append((obj, caption))

    _lay_out(plants, collection)
    return collection, len(plants), ""


def _rendered_size(obj, depsgraph) -> tuple[float, float]:
    """The plant's width and height *as rendered*, from its evaluated bounding
    box.

    Not `last_width`/`last_height`, and the difference is not small. Those
    measure the skeleton and its attachment points, which is right for what the
    Diagnostics panel uses them for but wrong here, because a blade extends far
    past the point it hangs from and only Geometry Nodes knows how far.
    Measured on the first grid built with this tool: 0.403 m reported against
    1.598 m actually rendered, and the row laid out on the reported number had
    every plant growing through its neighbours.
    """
    from mathutils import Vector

    evaluated = obj.evaluated_get(depsgraph)
    corners = [Vector(corner) for corner in evaluated.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    return (max(max(xs) - min(xs), max(ys) - min(ys)), max(zs) - min(zs))


def _lay_out(plants, collection) -> None:
    """Place the row on measured sizes rather than a fixed pitch.

    A fixed pitch cannot work: an Age sweep spans a seedling and a mature
    plant. Spacing every column by the widest plant in the sweep keeps the gaps
    even and guarantees nothing overlaps, at the cost of some air around the
    small ones -- the right trade, since two plants growing into each other
    read as one plant and that is exactly the misjudgement a sweep exists to
    prevent.
    """
    if not plants:
        return
    # The plants were built moments ago, so the depsgraph has to be brought up
    # to date before any evaluated bound box means anything.
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sizes = [_rendered_size(obj, depsgraph) for obj, _ in plants]
    # A degenerate value is real here, not a guard against a bug: sweeping Age
    # or Max Branch Order to its floor legitimately produces a plant with no
    # geometry at all, and that cell is informative rather than broken.
    widths = [max(w, 1e-3) for w, _ in sizes]
    heights = [max(h, 1e-3) for _, h in sizes]
    pitch_x = max(widths) * 1.25
    pitch_y = max(max(heights) * 0.8, pitch_x)
    label_size = max(heights) * _LABEL_HEIGHT
    columns = min(len(plants), _MAX_PER_ROW)

    for index, (obj, caption) in enumerate(plants):
        row, column = divmod(index, columns)
        x = (column - (columns - 1) / 2.0) * pitch_x
        y = -row * pitch_y
        obj.location = (x, y, 0.0)
        label = _label(caption, collection, label_size)
        # Below the base, so a caption never sits inside the plant it names.
        label.location = (x, y, -label_size * 1.6)
