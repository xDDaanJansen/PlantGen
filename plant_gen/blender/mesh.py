"""Geometry Nodes bridge: turns skeleton.py's stem + attachment meshes into
real tube+foliage geometry. Own-namespaced group ("PlantGen · Plant") --
reuses the parametric add-on's own proven GN patterns (Named Attribute straight into
Curve to Mesh's Scale input for tube width; Collection Info + seeded random
pick for asset instancing). The patterns are borrowed, not the code: a GN
group is built socket by socket against this approach's own attribute set,
so there is nothing here an import could have saved.

**Skeleton is a plain Mesh (vertices/edges + point attributes), not a
Curve** -- legacy Curve has no attribute API at all in Blender 5.2
(`curve_data.attributes` doesn't exist), and `plant_gen`'s per-point
data (order/vigor/width/...) genuinely varies along the skeleton, unlike
the parametric add-on's own constant-per-stem case. Verified live before writing this:
Mesh -> `Mesh to Curve` -> `Curve to Mesh` correctly handles a branching
edge mesh (produces one continuous tube across a fork, not a broken/zero
result), and custom point attributes survive that round trip intact -- so
GN's own `Mesh to Curve` does the tree-to-splines conversion natively, no
Python-side segmentation needed.
"""

from __future__ import annotations

import math

import bpy

_MESH_GROUP_NAME = "PlantGen · Plant"
_MESH_GROUP_VERSION = 84  # v84: a flower born late in the bloom window opens in what is left of it
# v76: flower / bud / fruit Red peaks at a quarter
# v75: flower / bud / fruit Red peaks at half a leaf's
# v74: Rig View -- the plant shows its baked, animated skin
# v73: organ Red in arcs from the attachment, cantilever-eased
# v72: Red on every organ from a floor above the stems'
# v71: the leaf Red gradient back; Color only on the stem mesh
# v70: categorical debug ramps in equal bands
# v69: debug views by a Debug Mode socket, vertex colours fixed
# v68: per-level stem materials
# v67: UV Size (absolute) replaces Tiles Around in the preview UVs
# v65: Weld Method Built (Python stem mesher) passes through
# Bump this whenever `build_mesh_group` changes. The group is cached by it, so
# an unbumped edit silently reuses the old graph -- which reads exactly like
# the new nodes not working (gravity, twist and wave all measured zero
# movement once, for no reason other than this).

# Which local axes an asset is modelled on. Leaves and flowers do not share a
# convention, and treating them as if they did is what put heads and solitary
# flowers on their sides: a leaf is a blade that *runs along* an axis, a flower
# is a face that *looks down* one. Measured off this file's own library, which
# The parametric add-on instances too:
#
#   grass blade   0.024 x 0.600 x 0.000  -- length +Y, flat in XY, face +Z
#   flower        0.019 x 0.020 x 0.011  -- a disc, thinnest in Z, opening +Z
#
# The primary axis is aligned to the organ's direction and the secondary to its
# normal, so both are pinned and no roll is left to chance.
_LEAF_PRIMARY_AXIS, _LEAF_SECONDARY_AXIS = 'Y', 'Z'
_FLOWER_PRIMARY_AXIS, _FLOWER_SECONDARY_AXIS = 'Z', 'Y'
_PG_VERSION_KEY = "pg_version"

_ATTACHMENTS_SOCKET = "Attachments"
_PROFILE_SIDES_SOCKET = "Profile Sides"
_PROFILE_MODE_SOCKET = "Profile Mode"
_PROFILE_LOBE_COUNT_SOCKET = "Profile Lobe Count"
_PROFILE_LOBE_DEPTH_SOCKET = "Profile Lobe Depth"
_PROFILE_FLATTEN_SOCKET = "Profile Flatten"
_PROFILE_OBJECT_SOCKET = "Profile Object"
# The stem's look: Bezier (Auto handles) resolution
# through the nodes, whether the per-axis tubes are unioned into one welded
# mesh, and the fillet radius along the weld seams in local stem radii.
_STEM_SMOOTH_SOCKET = "Stem Smoothness"
# UVs (phase 2): whole tiles of texture around every stem (V keeps texels square
# at each axis's mean width), and a checker override to preview them.
_STEM_UV_SIZE_SOCKET = "Stem UV Size"
"""The world size of one UV tile (metres): the same texel size on every stem,
branch and stalk, whatever its thickness."""
_STEM_CHECKER_SOCKET = "Stem UV Checker"
_STEM_CHECKER_MAT_SOCKET = "Stem Checker Material"
# Natural surface (phase 3), all render-side: bark noise, twist of the section
# along each axis, trunk flare lobes; Gnarl warps skeleton + organs together.
_STEM_BARK_SOCKET = "Stem Bark"
_STEM_TWIST_SOCKET = "Stem Twist"
_STEM_FLARE_SOCKET = "Stem Flare"
_STEM_LOBES_SOCKET = "Stem Flare Lobes"
_GNARL_SOCKET = "Stem Gnarl"
# Weld Method: 2 Built (the skeleton input IS the welded stem mesh, passed
# through), 0 a preview of plain swept tubes while the weld is built.
_STEM_WELD_METHOD_SOCKET = "Stem Weld Method"
# Each branching level's material (`stem_material.level_materials`), set on the
# stems by their `level` attribute; off (no level has one) leaves them as they
# were. A level past the last socket takes the last one's.
_STEM_MATERIALS_ON_SOCKET = "Stem Materials"
_STEM_MATERIAL_SOCKETS = tuple(f"Stem Material {i}" for i in range(8))
# A flower component's Catkin Material, on the stem vertices its bridge marks
# `catkin` (only written while one is set): catkins are swept into this mesh.
_CATKIN_MATERIAL_ON_SOCKET = "Catkin Material On"
_CATKIN_MATERIAL_SOCKET = "Catkin Material"
_LEAF_ASSET_SOCKET = "Leaf Asset Collection"
_LEAF_ASSET_COUNT_SOCKET = "Leaf Asset Count"
# Phase 2: when on, the Leaf Asset Collection is the master union of every leaf
# component's collection and each juvenile leaf point picks from it by a baked
# `instance_index`; off keeps the legacy seeded-random pick over a single
# collection (byte-identical). One switch, so no components = no change.
_LEAF_USE_COMPONENTS_SOCKET = "Leaf Use Components"
_BRACT_ASSET_SOCKET = "Bract Asset Collection"
_BRACT_ASSET_COUNT_SOCKET = "Bract Asset Count"
_BRACT_USE_COMPONENTS_SOCKET = "Bract Use Components"
_BUD_ASSET_SOCKET = "Bud Asset Collection"
_BUD_ASSET_COUNT_SOCKET = "Bud Asset Count"
_BUD_USE_COMPONENTS_SOCKET = "Bud Use Components"
_FLOWER_ASSET_SOCKET = "Flower Asset Collection"
_FLOWER_ASSET_COUNT_SOCKET = "Flower Asset Count"
_FLOWER_USE_COMPONENTS_SOCKET = "Flower Use Components"
_FRUIT_ASSET_SOCKET = "Fruit Asset Collection"
_FRUIT_ASSET_COUNT_SOCKET = "Fruit Asset Count"
_FRUIT_USE_COMPONENTS_SOCKET = "Fruit Use Components"
_RIPENESS_SOCKET = "Ripeness"
_BLOOM_CONTINUOUS_SOCKET = "Bloom Continuous"
_BLOOM_ANIMATE_SOCKET = "Bloom Animate"
_BLOOM_PROGRESS_SOCKET = "Bloom Progress"
_BLOOM_SPREAD_SOCKET = "Bloom Spread"
_BLOOM_FROM_AGE_SOCKET = "Bloom From Age"
_SEED_SOCKET = "Seed"
_DEAD_COLOR_SOCKET = "Dead Colour"  # WS3: what a dead vertex is tinted toward
# The debug view: 0 off (the plant as it ships), else 1 + its index in
# `DEBUG_MODES`; and the material the debug branch wears.
_DEBUG_MODE_SOCKET = "Debug Mode"
_DEBUG_MAT_SOCKET = "Debug Material"
# Rig View: the plant's output is its baked rig's skin (animated by the bones),
# read in by Object Info, instead of the live build -- one visible plant.
_RIG_VIEW_SOCKET = "Rig View"
_RIG_SKIN_SOCKET = "Rig Skin"
# Fast Preview: only the stems skin deforms; every organ stays the live plant's
# GPU instance, moved by its three tracker points (bound to its organ bone).
_RIG_FAST_SOCKET = "Rig Fast Preview"
_RIG_STEMS_SOCKET = "Rig Stems"
_RIG_TRACKERS_SOCKET = "Rig Trackers"
# Timelapse Cache: a recorded Grow Over Timeline frame (`timelapse_cache`) --
# its mesh + instance points, read in by Object Info and re-instanced from the
# assets -- instead of everything above, so playback skips the regrow.
_TL_CACHE_SOCKET = "Timelapse Cache"
_TL_MESH_SOCKET = "Timelapse Mesh"
_TL_POINTS_SOCKET = "Timelapse Instances"
_TL_REFS_SOCKET = "Timelapse Assets"
TL_REF_ATTR = "tl_ref"
TL_XF_ATTR = "tl_xf"
"""On the cache's instance points: the asset each instance picks (its rank in
the assets collection's name order -- how Collection Info orders separated
children) and its full transform."""


def _socket_identifier(ng, name: str) -> str:
    return next(
        item.identifier for item in ng.interface.items_tree
        if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == name
    )


def _set_modifier_input(mod, ident: str, value) -> None:
    """Blender 5.2 replaced the old `mod[identifier]` IDProperty dict
    access for GN modifier inputs with `mod.properties.inputs.<identifier>.
    value` -- the break the parametric add-on already hit and fixed this session
    (`mod[ident]` raises "this type doesn't support IDProperties"
    unconditionally). Applied here from day one, not rediscovered.
    """
    input_prop = getattr(mod.properties.inputs, ident)
    if input_prop.value != value:
        input_prop.value = value
        return True
    return False


def _store(nodes, links, geo, name: str, data_type: str, value, x: int, y: int):
    node = nodes.new("GeometryNodeStoreNamedAttribute")
    node.data_type = data_type
    node.domain = 'POINT'
    node.inputs["Name"].default_value = name
    links.new(geo, node.inputs["Geometry"])
    links.new(value, node.inputs["Value"])
    node.location = (x, y)
    return node.outputs["Geometry"]


def _named(nodes, name: str, data_type: str, x: int, y: int):
    node = nodes.new("GeometryNodeInputNamedAttribute")
    node.data_type = data_type
    node.inputs["Name"].default_value = name
    node.location = (x, y)
    return node.outputs["Attribute"]


def _bloom_openness(nodes, links, gin, x: int, y: int):
    """How open each flower is, 0 (bud) to 1 (fully out).

    Ported from the parametric add-on's own `_effective_bloom`, formula for formula,
    because that system works and this one's first attempt did not. Two modes,
    switched by Animate:

    * **Off** -- openness is `1 - bloom_start`, so the inflorescence shows its
      own gradient standing still: the flower that opens first is fully out,
      the last is a bud. A raceme reads correctly with no animation at all.
    * **On** -- each flower's own start on the timeline is `bloom_start *
      Spread` and it opens over the remaining `1 - Spread`. Spread 0 opens the
      whole inflorescence in lockstep; Spread 1 means the last flower does not
      begin until Progress reaches 1. Scrubbing Progress sweeps the bloom order
      up the stem as a wave.

    `bloom_start` is 0 for the first flower to open and 1 for the last, which
    is the opposite sense to the parametric add-on's `bloom_order` -- so the multiply
    here has no `1 -` in front of it where the parametric add-on's does. Same formula,
    stated on this codebase's own attribute.

    **The one addition to the parametric add-on's formula, and it is a correctness fix
    rather than a feature: a flower cannot be open before it exists.** That
    problem is this addon's alone, because the parametric add-on has no growth
    simulation -- every flower on a parametric plant is present from the
    start, so an order-based timeline is the only timeline there is. Here
    From Age makes Progress mean *the plant's own life*, and the two timelines
    then disagree badly. Measured on the foxglove: the topmost flower is born
    at Progress 0.862 but its order puts its start at 0.500, so it spent 36%
    of the timeline opening before it existed and arrived fully out. Every
    flower on the plant read as mature at Age 1.0.

    So in From Age mode a flower's start is pushed to whichever is later, its
    order or its birth. `max` rather than a replacement, because bloom order
    is still real -- a basipetal raceme opens top-down, and that must survive;
    birth only ever delays a flower, never reorders one. Costs no new control:
    Spread already says how long a flower takes to open once it starts.
    """
    start_attr = nodes.new("GeometryNodeInputNamedAttribute")
    start_attr.data_type = 'FLOAT'
    start_attr.inputs["Name"].default_value = "bloom_start"
    start_attr.location = (x, y)

    birth_attr = nodes.new("GeometryNodeInputNamedAttribute")
    birth_attr.data_type = 'FLOAT'
    birth_attr.inputs["Name"].default_value = "bloom_birth"
    birth_attr.location = (x, y + 200)

    # start = bloom_start * Spread
    order_start = nodes.new("ShaderNodeMath")
    order_start.operation = 'MULTIPLY'
    links.new(start_attr.outputs["Attribute"], order_start.inputs[0])
    links.new(gin.outputs[_BLOOM_SPREAD_SOCKET], order_start.inputs[1])
    order_start.location = (x + 190, y)

    born_start = nodes.new("ShaderNodeMath")
    born_start.operation = 'MAXIMUM'
    links.new(order_start.outputs["Value"], born_start.inputs[0])
    links.new(birth_attr.outputs["Attribute"], born_start.inputs[1])
    born_start.location = (x + 190, y + 200)

    # Only From Age makes Progress mean the plant's age, so only there does a
    # birth time on that timeline mean anything. Manual is deliberately free
    # of Age, so clamping it to one would tie back together exactly what that
    # mode exists to separate.
    start = nodes.new("ShaderNodeMix")
    start.data_type = 'FLOAT'
    links.new(gin.outputs[_BLOOM_FROM_AGE_SOCKET], start.inputs["Factor"])
    links.new(order_start.outputs["Value"], start.inputs["A"])
    links.new(born_start.outputs["Value"], start.inputs["B"])
    start.location = (x + 380, y + 100)

    # duration = max(min(1 - Spread, 1 - start), eps): Spread sets how fast a
    # flower opens, but never so slowly that it is still opening when Progress
    # reaches 1 -- a flower born late in the bloom window opens in what is left
    # of it, so the whole inflorescence is out when the window closes. Under
    # Manual (and order alone) start <= Spread, so 1 - start >= 1 - Spread and
    # this is 1 - Spread exactly, as before. eps: Spread = 1 is a snap rather
    # than a divide by zero.
    duration_raw = nodes.new("ShaderNodeMath")
    duration_raw.operation = 'SUBTRACT'
    duration_raw.inputs[0].default_value = 1.0
    links.new(gin.outputs[_BLOOM_SPREAD_SOCKET], duration_raw.inputs[1])
    duration_raw.location = (x + 190, y - 160)

    remaining = nodes.new("ShaderNodeMath")
    remaining.operation = 'SUBTRACT'
    remaining.inputs[0].default_value = 1.0
    links.new(start.outputs["Result"], remaining.inputs[1])
    remaining.location = (x + 570, y - 320)

    within = nodes.new("ShaderNodeMath")
    within.operation = 'MINIMUM'
    links.new(duration_raw.outputs["Value"], within.inputs[0])
    links.new(remaining.outputs["Value"], within.inputs[1])
    within.location = (x + 570, y - 160)

    duration = nodes.new("ShaderNodeMath")
    duration.operation = 'MAXIMUM'
    duration.inputs[1].default_value = 0.0001
    links.new(within.outputs["Value"], duration.inputs[0])
    duration.location = (x + 650, y - 160)

    since = nodes.new("ShaderNodeMath")
    since.operation = 'SUBTRACT'
    links.new(gin.outputs[_BLOOM_PROGRESS_SOCKET], since.inputs[0])
    links.new(start.outputs["Result"], since.inputs[1])
    since.location = (x + 570, y)

    animated = nodes.new("ShaderNodeMath")
    animated.operation = 'DIVIDE'
    animated.use_clamp = True
    links.new(since.outputs["Value"], animated.inputs[0])
    links.new(duration.outputs["Value"], animated.inputs[1])
    animated.location = (x + 760, y)

    static = nodes.new("ShaderNodeMath")
    static.operation = 'SUBTRACT'
    static.inputs[0].default_value = 1.0
    links.new(start_attr.outputs["Attribute"], static.inputs[1])
    static.location = (x + 760, y - 320)

    gate = nodes.new("ShaderNodeMix")
    gate.data_type = 'FLOAT'
    links.new(gin.outputs[_BLOOM_ANIMATE_SOCKET], gate.inputs["Factor"])
    links.new(static.outputs["Value"], gate.inputs["A"])
    links.new(animated.outputs["Value"], gate.inputs["B"])
    gate.location = (x + 950, y - 160)
    return gate.outputs["Result"]


def _stage_indices(nodes, links, openness, count_socket, x: int, y: int):
    """Turn openness into a position in an ordered set of bloom stages.

    `count_socket` is already "member count - 1" (the caller wires it that way
    for the random pick), so `openness * count_socket` is the continuous stop
    index directly. Returns (rounded, lo, hi, frac): the discrete pick, the two
    stages bracketing it, and how far between them.
    """
    stop = nodes.new("ShaderNodeMath")
    stop.operation = 'MULTIPLY'
    links.new(openness, stop.inputs[0])
    links.new(count_socket, stop.inputs[1])
    stop.location = (x, y)

    rounded = nodes.new("ShaderNodeMath")
    rounded.operation = 'ROUND'
    links.new(stop.outputs["Value"], rounded.inputs[0])
    rounded.location = (x + 190, y + 160)

    lo = nodes.new("ShaderNodeMath")
    lo.operation = 'FLOOR'
    links.new(stop.outputs["Value"], lo.inputs[0])
    lo.location = (x + 190, y)

    frac = nodes.new("ShaderNodeMath")
    frac.operation = 'SUBTRACT'
    links.new(stop.outputs["Value"], frac.inputs[0])
    links.new(lo.outputs["Value"], frac.inputs[1])
    frac.location = (x + 380, y - 160)

    hi_raw = nodes.new("ShaderNodeMath")
    hi_raw.operation = 'ADD'
    hi_raw.inputs[1].default_value = 1.0
    links.new(lo.outputs["Value"], hi_raw.inputs[0])
    hi_raw.location = (x + 380, y)

    hi = nodes.new("ShaderNodeMath")
    hi.operation = 'MINIMUM'
    links.new(hi_raw.outputs["Value"], hi.inputs[0])
    links.new(count_socket, hi.inputs[1])
    hi.location = (x + 570, y)

    return (rounded.outputs["Value"], lo.outputs["Value"],
            hi.outputs["Value"], frac.outputs["Value"])


def _blend_stages(nodes, links, lo_instances, hi_instances, frac, x: int, y: int):
    """Blend two instanced stage streams vertex by vertex.

    Both streams are realized and paired by plain global vertex index -- no
    nearest-neighbour search -- which is only valid because every stage shares
    identical topology. The parametric add-on gets that for free by baking its stages from
    one FlowerGen flower whose resolution does not depend on Bloom; here it is
    a real precondition on the collection the user points at, and the reason
    Continuous is a toggle rather than always-on.

    Mismatched topology does not error, it silently scrambles: vertex i of a
    28-vertex bud gets paired with vertex i of a 96-vertex open flower.
    """
    realize_lo = nodes.new("GeometryNodeRealizeInstances")
    links.new(lo_instances, realize_lo.inputs["Geometry"])
    realize_lo.location = (x, y)

    realize_hi = nodes.new("GeometryNodeRealizeInstances")
    links.new(hi_instances, realize_hi.inputs["Geometry"])
    realize_hi.location = (x, y - 200)

    index_node = nodes.new("GeometryNodeInputIndex")
    index_node.location = (x + 190, y - 400)

    hi_position = nodes.new("GeometryNodeInputPosition")
    hi_position.location = (x + 190, y - 540)

    sample_hi = nodes.new("GeometryNodeSampleIndex")
    sample_hi.data_type = 'FLOAT_VECTOR'
    sample_hi.domain = 'POINT'
    links.new(realize_hi.outputs["Geometry"], sample_hi.inputs["Geometry"])
    links.new(hi_position.outputs["Position"], sample_hi.inputs["Value"])
    links.new(index_node.outputs["Index"], sample_hi.inputs["Index"])
    sample_hi.location = (x + 380, y - 400)

    lo_position = nodes.new("GeometryNodeInputPosition")
    lo_position.location = (x + 380, y - 700)

    # Mix's sockets are duplicated per data_type and the disabled set answers to
    # the same names, so these are indexed positionally -- confirmed by
    # The parametric add-on live for VECTOR: Factor(uniform)=0, A=4, B=5, Result=1.
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = 'VECTOR'
    links.new(frac, mix.inputs[0])
    links.new(lo_position.outputs["Position"], mix.inputs[4])
    links.new(sample_hi.outputs["Value"], mix.inputs[5])
    mix.location = (x + 570, y - 400)

    set_pos = nodes.new("GeometryNodeSetPosition")
    links.new(realize_lo.outputs["Geometry"], set_pos.inputs["Geometry"])
    links.new(mix.outputs[1], set_pos.inputs["Position"])
    set_pos.location = (x + 760, y)
    return set_pos.outputs["Geometry"]


_STAGE_TOPOLOGY: dict = {}
"""Mesh name -> (vertex count, loop count, topology digest), per mesh state."""


def _stage_topology(o):
    """A bloom stage's topology as Continuous pairs it -- its mesh's vertex
    count and face-corner order, and its modifier stack (types only) -- or None
    when it is not a mesh object."""
    import array
    import hashlib

    me = o.data if o.type == 'MESH' else None
    if me is None:
        return None
    key = (me.name, len(me.vertices), len(me.loops))
    hit = _STAGE_TOPOLOGY.get(key)
    if hit is None:
        corners = array.array('i', [0]) * len(me.loops)
        me.loops.foreach_get("vertex_index", corners)
        hit = hashlib.blake2b(corners.tobytes(), digest_size=8).hexdigest()
        if len(_STAGE_TOPOLOGY) > 512:
            _STAGE_TOPOLOGY.clear()
        _STAGE_TOPOLOGY[key] = hit
    return (len(me.vertices), hit, tuple(m.type for m in o.modifiers))


def bloom_stage_mismatch(collection) -> list[str]:
    """The stages in `collection` whose topology differs from its first
    (in name order, the order the stages play) -- which Bloom Continuous would
    pair vertex by vertex with the wrong vertices (`_blend_stages`). Empty when
    every stage matches or the collection holds no meshes."""
    if collection is None:
        return []
    stages = sorted(collection.objects, key=lambda o: o.name)
    tops = [(o.name, _stage_topology(o)) for o in stages]
    tops = [(name, top) for name, top in tops if top is not None]
    if len(tops) < 2:
        return []
    first = tops[0][1]
    return [name for name, top in tops[1:] if top != first]


def _stem_profile(nodes, links, gin, x: int, y: int):
    """The cross-section curve swept along every stem.

    Technique ported from the parametric add-on's own profile builder, not the code --
    that is a closure over its graph builder, and the sharing rule forbids
    editing the parametric add-on to expose it. One real simplification on the way: it
    keeps a polygon branch and a lobed branch behind separate Switches because
    its lobe resolution is baked at 64 while its polygon side count is
    user-set. Driving the circle's resolution from the same `sides` input makes
    them one graph -- a lobed profile with depth 0 *is* a regular polygon,
    exactly -- so circular, the fixed polygons and polygonal all come out of
    the lobed path with no lobes. See `BranchLevel.section_inputs`, which
    is what decides those numbers.

    The radius modulation is `1 + cos(lobe_count * angle) * lobe_depth`, with
    the angle taken from the point's own index rather than a spline parameter,
    since a Curve Circle's points are evenly spaced by construction and the
    index is exact where a factor would carry float error at the seam.
    """
    circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
    circle.mode = 'RADIUS'
    links.new(gin.outputs[_PROFILE_SIDES_SOCKET], circle.inputs["Resolution"])
    circle.inputs["Radius"].default_value = 1.0
    circle.location = (x, y)

    # angle = index * 2pi / sides. The divide is a node rather than a constant
    # because `sides` is a socket -- baking 2pi/64 the way the parametric add-on can
    # would put every non-lobed shape's lobes at the wrong frequency.
    index = nodes.new("GeometryNodeInputIndex")
    index.location = (x, y - 120)

    step = nodes.new("ShaderNodeMath")
    step.operation = 'DIVIDE'
    step.inputs[0].default_value = 2.0 * math.pi
    links.new(gin.outputs[_PROFILE_SIDES_SOCKET], step.inputs[1])
    step.location = (x, y - 220)

    angle = nodes.new("ShaderNodeMath")
    angle.operation = 'MULTIPLY'
    links.new(index.outputs["Index"], angle.inputs[0])
    links.new(step.outputs["Value"], angle.inputs[1])
    angle.location = (x + 180, y - 170)

    phase = nodes.new("ShaderNodeMath")
    phase.operation = 'MULTIPLY'
    links.new(angle.outputs["Value"], phase.inputs[0])
    links.new(gin.outputs[_PROFILE_LOBE_COUNT_SOCKET], phase.inputs[1])
    phase.location = (x + 360, y - 170)

    wave = nodes.new("ShaderNodeMath")
    wave.operation = 'COSINE'
    links.new(phase.outputs["Value"], wave.inputs[0])
    wave.location = (x + 540, y - 170)

    scaled = nodes.new("ShaderNodeMath")
    scaled.operation = 'MULTIPLY'
    links.new(wave.outputs["Value"], scaled.inputs[0])
    links.new(gin.outputs[_PROFILE_LOBE_DEPTH_SOCKET], scaled.inputs[1])
    scaled.location = (x + 720, y - 170)

    radius = nodes.new("ShaderNodeMath")
    radius.operation = 'ADD'
    radius.inputs[1].default_value = 1.0
    links.new(scaled.outputs["Value"], radius.inputs[0])
    radius.location = (x + 900, y - 170)

    position = nodes.new("GeometryNodeInputPosition")
    position.location = (x + 900, y - 300)

    moved = nodes.new("ShaderNodeVectorMath")
    moved.operation = 'SCALE'
    links.new(position.outputs["Position"], moved.inputs[0])
    links.new(radius.outputs["Value"], moved.inputs["Scale"])
    moved.location = (x + 1080, y - 220)

    lobed = nodes.new("GeometryNodeSetPosition")
    links.new(circle.outputs["Curve"], lobed.inputs["Geometry"])
    links.new(moved.outputs["Vector"], lobed.inputs["Position"])
    lobed.location = (x + 1260, y)

    # Custom: the user's own Curve object, read straight. 'RELATIVE' so the
    # object's own transform is applied -- scaling the profile object scales
    # the stem's section, which is what anyone drawing one expects.
    custom = nodes.new("GeometryNodeObjectInfo")
    custom.transform_space = 'RELATIVE'
    links.new(gin.outputs[_PROFILE_OBJECT_SOCKET], custom.inputs["Object"])
    custom.location = (x + 1260, y - 400)

    is_custom = nodes.new("FunctionNodeCompare")
    is_custom.data_type = 'INT'
    is_custom.operation = 'EQUAL'
    links.new(gin.outputs[_PROFILE_MODE_SOCKET], is_custom.inputs["A"])
    is_custom.inputs["B"].default_value = 1
    is_custom.location = (x + 1440, y - 300)

    pick = nodes.new("GeometryNodeSwitch")
    pick.input_type = 'GEOMETRY'
    links.new(is_custom.outputs["Result"], pick.inputs["Switch"])
    links.new(lobed.outputs["Geometry"], pick.inputs["False"])
    links.new(custom.outputs["Geometry"], pick.inputs["True"])
    pick.location = (x + 1620, y)

    # Flatten squashes whichever profile was picked, rather than being a third
    # branch: it is 0 for every shape except Flattened, so this is a no-op the
    # rest of the time.
    squash = nodes.new("ShaderNodeMath")
    squash.operation = 'SUBTRACT'
    squash.inputs[0].default_value = 1.0
    links.new(gin.outputs[_PROFILE_FLATTEN_SOCKET], squash.inputs[1])
    squash.location = (x + 1620, y - 300)

    squash_vec = nodes.new("ShaderNodeCombineXYZ")
    squash_vec.inputs["X"].default_value = 1.0
    squash_vec.inputs["Z"].default_value = 1.0
    links.new(squash.outputs["Value"], squash_vec.inputs["Y"])
    squash_vec.location = (x + 1800, y - 300)

    flatten = nodes.new("GeometryNodeTransform")
    links.new(pick.outputs["Output"], flatten.inputs["Geometry"])
    links.new(squash_vec.outputs["Vector"], flatten.inputs["Scale"])
    flatten.location = (x + 1980, y)
    return flatten.outputs["Geometry"]


# Red's band reserved for the stems (Quixel / Megascans, Unreal Fest 2025:
# "all the branches are fairly dark", and their branch mask is a threshold on
# Red). Stems are flat 0; every organ's gradient starts here, so a threshold
# anywhere below it separates stems from leaves / flowers / buds / fruit.
_ORGAN_RED_FLOOR = 0.1
# Red's peak per asset kind. Red is the wind-flutter weight; a flower / bud /
# fruit fluttering as freely as a leaf blade read far too strong (user
# 2026-10-09, twice), so a disc asset's gradient tops out at a quarter.
_BLADE_RED_PEAK = 1.0
_DISC_RED_PEAK = 0.25


def _asset_gradient(nodes, links, assets, peak: float = 1.0):
    """Vertex-colour Red on each asset: `_ORGAN_RED_FLOOR` where it attaches
    rising to `peak` at its farthest point (`_BLADE_RED_PEAK` on a leaf,
    `_DISC_RED_PEAK` on a flower / bud / fruit) -- Red is the wind-flutter
    weight.

    * **Arcs, not bands:** u = distance from the asset's origin (its
      attachment point: leaf assets grow from theirs, FlowerGen builds on
      its) over the farthest vertex's. A leaf's edges read brighter than its
      midrib at the same length (the outsides flutter -- the talk's leaf
      wobble), a drooping tip still reads 1, and a bell's mouth / an open
      flower's petal tips both reach 1.
    * **Cantilever-eased:** Red follows the deflection of a uniformly loaded
      cantilever, (6u^2 - 4u^3 + u^4) / 3 -- 0 slope at the root (stiff where
      it attaches), 1 at the tip -- not a straight ramp.

    A point store on the collection's instances writes into every instanced
    mesh, the field evaluated in that mesh -- so Field Min & Max normalises
    each asset by its own extent, and every instance of it carries its own
    gradient into a realize."""
    position = nodes.new("GeometryNodeInputPosition")
    dist = nodes.new("ShaderNodeVectorMath")
    dist.operation = 'LENGTH'
    links.new(position.outputs["Position"], dist.inputs[0])
    extent = nodes.new("GeometryNodeFieldMinAndMax")
    extent.data_type = 'FLOAT'
    extent.domain = 'POINT'
    links.new(dist.outputs["Value"], extent.inputs["Value"])
    u = nodes.new("ShaderNodeMapRange")              # 0 at the origin, 1 farthest
    u.clamp = True
    links.new(dist.outputs["Value"], u.inputs["Value"])
    u.inputs["From Min"].default_value = 0.0
    links.new(extent.outputs["Max"], u.inputs["From Max"])
    u = u.outputs["Result"]

    def math(op, a, b=None, c=None):
        node = nodes.new("ShaderNodeMath")
        node.operation = op
        for i, value in enumerate((a, b, c)):
            if value is None:
                continue
            if isinstance(value, float):
                node.inputs[i].default_value = value
            else:
                links.new(value, node.inputs[i])
        return node.outputs["Value"]

    # (6u^2 - 4u^3 + u^4) / 3 = u^2 * (u * (u - 4) + 6) / 3
    inner = math('MULTIPLY_ADD', u, math('SUBTRACT', u, 4.0), 6.0)
    bend = math('DIVIDE', math('MULTIPLY', math('MULTIPLY', u, u), inner), 3.0)
    red = math('MULTIPLY_ADD', bend, peak - _ORGAN_RED_FLOOR, _ORGAN_RED_FLOOR)
    return _store(nodes, links, assets, _VCOL_R_ATTR, 'FLOAT', red, 0, 0)


def _asset_pick_instances(nodes, links, points_geo, collection_socket, count_socket, seed_socket,
                           scale_socket, direction_attr, normal_attr, id_attr, x: int, y: int,
                           primary_axis: str, secondary_axis: str, scale_attr=None,
                           index_socket=None, gradient=None):
    """The parametric add-on-proven "random object from a collection" recipe:
    Collection Info (Separate + Reset Children) -> a seeded per-point
    random index -> Instance on Points (Pick Instance). An empty/unresolved
    collection (count=0) just yields zero instances -- Instance on Points
    with an empty Instance source is a graceful no-op, so unlike
    the parametric add-on this doesn't need an explicit Switch/placeholder-quad gate.
    """
    collection_info = nodes.new("GeometryNodeCollectionInfo")
    collection_info.inputs["Separate Children"].default_value = True
    collection_info.inputs["Reset Children"].default_value = True
    links.new(collection_socket, collection_info.inputs["Collection"])
    collection_info.location = (x, y + 200)

    # The parametric add-on's own hard-won note (a different Blender version) found
    # FunctionNodeRandomValue's Min/Max/ID/Seed/Value sockets duplicated
    # per data_type, needing positional indices. Re-verified live in this
    # 5.2 build: that's no longer true here -- data_type='INT' leaves a
    # plain 4-input/1-output node (Min, Max, ID, Seed / Value), name lookup
    # works directly. Don't assume either behavior without checking again
    # if this breaks on a future Blender version.
    if index_socket is None:
        random_index = nodes.new("FunctionNodeRandomValue")
        random_index.data_type = 'INT'
        random_index.inputs["Min"].default_value = 0
        links.new(count_socket, random_index.inputs["Max"])  # asset count - 1, by caller
        links.new(id_attr, random_index.inputs["ID"])
        links.new(seed_socket, random_index.inputs["Seed"])
        random_index.location = (x, y + 100)
        index_socket = random_index.outputs["Value"]

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.inputs["Pick Instance"].default_value = True
    assets = collection_info.outputs["Instances"]
    if gradient:
        assets = _asset_gradient(nodes, links, assets, gradient)
    links.new(assets, instance.inputs["Instance"])
    links.new(index_socket, instance.inputs["Instance Index"])
    instance.location = (x, y)

    # Two axes, not one. Align Rotation to Vector constrains only where the
    # organ points and leaves the roll about that axis to the node's own
    # choice, which for the many near-vertical leaves on a plant lands on
    # near-identical rolls -- every blade facing one way in world space. Axes
    # to Rotation pins both: length along the organ's direction, face along the
    # organ's own normal, which is derived from the stem it grows on.
    align = nodes.new("FunctionNodeAxesToRotation")
    align.primary_axis = primary_axis
    align.secondary_axis = secondary_axis
    links.new(direction_attr, align.inputs["Primary Axis"])
    links.new(normal_attr, align.inputs["Secondary Axis"])
    links.new(align.outputs["Rotation"], instance.inputs["Rotation"])
    align.location = (x - 200, y - 100)

    # The user's uniform Leaf/Flower Scale times the organ's own `scale`
    # attribute, which carries how far that individual organ has expanded on
    # its own clock. Without the second factor every leaf renders at full size
    # regardless of age, and the developmental sequence loses the single most
    # visible cue that the plant is maturing rather than just getting bigger.
    # `scale_socket` is a uniform multiplier the panel supplies, and is None for
    # leaves -- Leaf Size already says how big a leaf is, and a second slider
    # multiplying it was the same duplication Flower Size had. `scale_attr` is
    # the organ's own expansion on its own clock; without it every leaf renders
    # full-size regardless of age, losing the clearest cue that the plant is
    # maturing rather than just getting bigger.
    if scale_socket is not None and scale_attr is not None:
        organ_scale = nodes.new("ShaderNodeMath")
        organ_scale.operation = 'MULTIPLY'
        links.new(scale_socket, organ_scale.inputs[0])
        links.new(scale_attr, organ_scale.inputs[1])
        organ_scale.location = (x - 400, y - 250)
        effective_scale = organ_scale.outputs["Value"]
    else:
        effective_scale = scale_attr if scale_attr is not None else scale_socket

    combine_scale = nodes.new("ShaderNodeCombineXYZ")
    links.new(effective_scale, combine_scale.inputs["X"])
    links.new(effective_scale, combine_scale.inputs["Y"])
    links.new(effective_scale, combine_scale.inputs["Z"])
    links.new(combine_scale.outputs["Vector"], instance.inputs["Scale"])
    combine_scale.location = (x - 200, y - 250)

    links.new(points_geo, instance.inputs["Points"])
    return instance.outputs["Instances"]


def _component_index(nodes, links, gin, max_index_socket, id_attr, use_socket, x, y):
    """The index a type's pick uses, switched by its use-components toggle (Phase 2).
    Off (no components) -> the legacy seeded-random index over a single collection,
    reproduced node-for-node so it is byte-identical. On -> the `instance_index`
    baked per point by the bridge, indexing the master union collection fed into the
    type's socket. Returns the switch output to pass as `_asset_pick_instances`'s
    `index_socket`."""
    legacy = nodes.new("FunctionNodeRandomValue")
    legacy.data_type = 'INT'
    legacy.inputs["Min"].default_value = 0
    links.new(max_index_socket, legacy.inputs["Max"])
    links.new(id_attr, legacy.inputs["ID"])
    links.new(gin.outputs[_SEED_SOCKET], legacy.inputs["Seed"])
    legacy.location = (x, y + 40)
    baked = _named(nodes, "instance_index", 'INT', x, y + 160)
    switch = nodes.new("GeometryNodeSwitch")
    switch.input_type = 'INT'
    links.new(gin.outputs[use_socket], switch.inputs["Switch"])
    links.new(legacy.outputs["Value"], switch.inputs["False"])
    links.new(baked, switch.inputs["True"])
    switch.location = (x + 160, y + 80)
    return switch.outputs["Output"]


_VCOL_R_ATTR = "vcol_r"


def _constant_vcolor_r(nodes, links, geo, x: int, y: int):
    """The stems carry a flat Red (0, below every organ's gradient -- see
    `_ORGAN_RED_FLOOR`). Stored explicitly rather than left
    to Join Geometry's implicit zero-fill: a channel a shader depends on should
    be authored on every branch that reaches the output, not present by
    accident of which branch happened to define it."""
    zero = nodes.new("ShaderNodeValue")
    zero.outputs[0].default_value = 0.0
    zero.location = (x, y - 160)
    return _store(nodes, links, geo, _VCOL_R_ATTR, 'FLOAT', zero.outputs[0], x + 200, y)


def _vertex_colors(nodes, links, gin, geo, x: int, y: int):
    """The Quixel/Megascans vegetation vertex-color convention, packed into one
    `Color` attribute -- matching the parametric add-on's assembly 1:1 so a shader written
    against either addon behaves the same on the other.

    WS3: the finished `Color` is then tinted toward the Dead Colour input by the
    `dead` attribute, so a dead branch (dead 1.0) reads as dead wood and foliage
    browning toward death (0..1) ramps into it. Live geometry (dead 0) is
    untouched, so this is byte-identical until senescence is asked for. Only a
    shader that samples `Color` shows it; the Dead debug view is the viewport
    counterpart.

    * **R** -- 0 on the stems; on every organ its own gradient from
      `_ORGAN_RED_FLOOR` where it attaches to its peak at its farthest point
      (1 on a leaf, a quarter on a flower / bud / fruit), in
      arcs round the attachment, cantilever-eased (`_asset_gradient`, on the
      assets as `vcol_r`). A threshold below the floor is a branch mask
      (Quixel).
      Per vertex, so it joins Color only where the plant is realized
      (`_build_compose_vertex_colors`: the debug view, the add-on's bakes).
    * **G** -- one random value per leaf/flower, hashed from `organ_id`.
    * **B** -- one random value for the whole plant object, identical on every
      vertex, for per-instance variation once scattered.
    * **A** -- height fraction of the whole plant, root to tip.

    Always authored, with no opt-out: a shader downstream cannot conditionally
    expect a channel to exist.

    `organ_id`, not `id`: confirmed live that a plain `id` read in a realized
    context returns Blender's own instance-identity hashes rather than the
    value written to it. Green is exactly that read. See `bridge._ORGAN_ATTRS`.
    """
    r = _named(nodes, _VCOL_R_ATTR, 'FLOAT', x, y + 200)
    organ_id = _named(nodes, "organ_id", 'INT', x, y)

    # Sockets addressed by NAME. The parametric add-on's note that Random Value's
    # Min/Max/ID/Seed are duplicated per data_type and must be indexed
    # positionally (ID=7, Seed=8) is **not true in this 5.2 build** -- probed
    # live 2026-08-15: both FLOAT and INT leave a plain 4-input/1-output node,
    # so indices 7 and 8 do not exist at all. Re-verify before trusting either
    # form; this has now changed once already.
    g_random = nodes.new("FunctionNodeRandomValue")
    g_random.data_type = 'FLOAT'
    g_random.inputs["Min"].default_value = 0.0
    g_random.inputs["Max"].default_value = 1.0
    g_random.inputs["Seed"].default_value = 0
    g_random.location = (x + 200, y)
    links.new(organ_id, g_random.inputs["ID"])

    b = _named(nodes, "instance_random", 'FLOAT', x, y - 200)
    a = _named(nodes, "height_fraction", 'FLOAT', x, y - 340)

    combine = nodes.new("FunctionNodeCombineColor")
    combine.mode = 'RGB'
    combine.location = (x + 450, y)
    links.new(r, combine.inputs["Red"])
    links.new(g_random.outputs["Value"], combine.inputs["Green"])
    links.new(b, combine.inputs["Blue"])
    links.new(a, combine.inputs["Alpha"])

    # WS3 dead tint: mix the Quixel colour toward the Dead Colour by `dead`.
    # Sockets are picked by IDENTIFIER: ShaderNodeMix carries a Float, a
    # Vector, a Color and a Rotation pair, the names "Factor" / "A" / "B"
    # repeat across them, and in 5.2 the float factor's type is
    # NodeSocketFloatFactor -- picking by type sent `dead` into the hidden
    # float A, so the factor was never driven and every vertex came out the
    # Dead Colour (found 2026-10-08).
    dead = _named(nodes, "dead", 'FLOAT', x + 450, y - 400)
    mix = _mix_rgba(nodes)
    mix.clamp_factor = True
    mix.location = (x + 620, y - 200)
    links.new(dead, _mix_in(mix, "Factor_Float"))
    links.new(combine.outputs["Color"], _mix_in(mix, "A_Color"))         # the live colour
    links.new(gin.outputs[_DEAD_COLOR_SOCKET], _mix_in(mix, "B_Color"))  # the Dead Colour
    result = _mix_out(mix)

    # On the stem MESH only: a point store on geometry holding instances also
    # writes into every instanced asset, with no organ to read there -- every
    # leaf and flower got one constant Color, which then beat the per-organ
    # one below on realize (found 2026-10-08).
    parts = nodes.new("GeometryNodeSeparateComponents")
    links.new(geo, parts.inputs["Geometry"])
    on_mesh = _store(nodes, links, parts.outputs["Mesh"], "Color", 'FLOAT_COLOR', result,
                     x + 850, y)
    rejoin = nodes.new("GeometryNodeJoinGeometry")
    for sock in (on_mesh, parts.outputs["Curve"], parts.outputs["Point Cloud"],
                 parts.outputs["Instances"]):
        links.new(sock, rejoin.inputs["Geometry"])
    stored = rejoin.outputs["Geometry"]
    # And on the instances (leaves, flowers, buds, fruit): every field above
    # reads the instance's own attributes there, so a realized export (or a
    # bake) carries each organ's colour, not only the stems'.
    on_instances = nodes.new("GeometryNodeStoreNamedAttribute")
    on_instances.data_type = 'FLOAT_COLOR'
    on_instances.domain = 'INSTANCE'
    on_instances.inputs["Name"].default_value = "Color"
    on_instances.location = (x + 1050, y)
    links.new(stored, on_instances.inputs["Geometry"])
    links.new(result, on_instances.inputs["Value"])
    return on_instances.outputs["Geometry"], result


def _mix_rgba(nodes):
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    return mix


def _mix_in(mix, identifier: str):
    """A Mix node input by identifier (`Factor_Float`, `A_Color`, `B_Color`)."""
    return next(s for s in mix.inputs if s.identifier == identifier)


def _mix_out(mix):
    return next(s for s in mix.outputs if s.identifier == "Result_Color")


_WHITE = (1.0, 1.0, 1.0, 1.0)
_BLACK = (0.0, 0.0, 0.0, 1.0)

# -- Debug palettes -------------------------------------------------------
# Three encodings, because these views carry three different kinds of number
# and one ramp cannot serve all of them.
#
# **Not red-to-green**, which is the one obvious alternative worth ruling out
# explicitly: red-green deficiency is the common one (around 8% of men), and a
# red-to-green ramp collapses to a single muddy band for them -- it is the
# textbook example of what not to use for data. Blue-to-yellow is the axis that
# survives both protan and deutan vision, which is why the first version
# reached for it and why the replacement keeps its endpoints.

_SEQUENTIAL = (
    (0.267, 0.005, 0.329, 1.0),
    (0.229, 0.322, 0.545, 1.0),
    (0.128, 0.567, 0.551, 1.0),
    (0.369, 0.789, 0.383, 1.0),
    (0.993, 0.906, 0.144, 1.0),
)
"""Viridis, for a continuous magnitude. It keeps blue-to-yellow's endpoints and
its colour-blind safety, and adds the thing a two-stop ramp cannot have:
**lightness rises monotonically from one end to the other**. That is what lets
you rank two patches by eye rather than only tell them apart -- on a plain
blue-to-yellow ramp the middle is a mid-grey ambiguity that reads equally like
either end."""

_CATEGORICAL = (
    (0.902, 0.624, 0.000, 1.0),   # orange
    (0.337, 0.706, 0.914, 1.0),   # sky blue
    (0.000, 0.620, 0.451, 1.0),   # bluish green
    (0.941, 0.894, 0.259, 1.0),   # yellow
    (0.000, 0.447, 0.698, 1.0),   # blue
    (0.835, 0.369, 0.000, 1.0),   # vermillion
    (0.800, 0.475, 0.655, 1.0),   # reddish purple
    (0.700, 0.700, 0.700, 1.0),   # grey
)
"""Okabe-Ito, the standard colour-blind-safe qualitative set, cycled.

For a *category* a gradient is simply the wrong encoding, and this was a real
defect rather than a preference: Branch Order and Axis normalise their value
across the whole plant and then ramp it, so on a plant with thirty axes two
neighbouring axes differ by a thirtieth of a ramp -- indistinguishable, while
the tooltip promises "a different colour per stem or branch". Distinct hues,
repeating every eight, actually deliver that. Okabe-Ito's own black is replaced
by grey, since black is invisible against Blender's dark viewport."""

_BLOOM_BANDS = 10
"""Bloom is banded rather than smooth, because what you actually see is one of
a discrete set of stages -- a smooth ramp implies a continuum the renderer does
not have. Ten bands matches a typical baked stage set, so one flat colour is a
set of flowers showing literally the same mesh."""

_DEBUG_RAMPS = {
    # The four vertex-colour channels stay literal: a channel shown in its own
    # colour is the one case where the encoding should not be reinterpreted.
    "vcol_r": ((_BLACK, (1.0, 0.0, 0.0, 1.0)), 'LINEAR'),
    "vcol_g": ((_BLACK, (0.0, 1.0, 0.0, 1.0)), 'LINEAR'),
    "vcol_b": ((_BLACK, (0.0, 0.0, 1.0, 1.0)), 'LINEAR'),
    "vcol_a": ((_BLACK, _WHITE), 'LINEAR'),
    # Height reads the same 0-1 the Height instancing mask does; a colourful
    # sequential ramp shows the gradient better than Alpha's greyscale.
    "height": (_SEQUENTIAL, 'LINEAR'),
    # Shade is 0-1 already (0 open sky, 1 fully shaded); its own sequential ramp,
    # no per-plant normalize, so open air reads the same on every plant.
    "shade": (_SEQUENTIAL, 'LINEAR'),
    # Mask preview: the windowed keep-probability, 0 dropped to 1 kept. Red at 0
    # and green at 1 so "this organ is being cut" reads at a glance, moving as the
    # min/max window is dragged.
    "mask_preview": (((0.70, 0.10, 0.10, 1.0), (0.15, 0.62, 0.20, 1.0)), 'LINEAR'),
    # Categories get distinct hues, magnitudes get the sequential ramp.
    "order": (_CATEGORICAL, 'CONSTANT'),
    "structural_depth": (_CATEGORICAL, 'CONSTANT'),
    "axis_id": (_CATEGORICAL, 'CONSTANT'),
    "kind": (_CATEGORICAL[:5], 'CONSTANT'),   # leaf / flower / bud / apex / fruit
    "node_index": (_SEQUENTIAL, 'LINEAR'),
    "birth_step": (_SEQUENTIAL, 'LINEAR'),
    "width": (_SEQUENTIAL, 'LINEAR'),
    # The beam solver's four readouts. All continuous magnitudes, so all the
    # sequential ramp; all normalized per plant at the tap, since their raw
    # ranges span orders of magnitude (deg/m, kg, Pa, N*m) and differ wildly
    # between a wiry stem and a woody one.
    "curvature": (_SEQUENTIAL, 'LINEAR'),
    "load": (_SEQUENTIAL, 'LINEAR'),
    "stiffness": (_SEQUENTIAL, 'LINEAR'),
    "moment": (_SEQUENTIAL, 'LINEAR'),
    "bloom_order": (_SEQUENTIAL, 'LINEAR'),
    "bloom": (_SEQUENTIAL, 'CONSTANT'),
    "organ_scale": (_SEQUENTIAL, 'LINEAR'),
    "bloom_birth": (_SEQUENTIAL, 'LINEAR'),
    # WS3: alive -> dead. Green to a grey-brown, its own two-colour ramp rather
    # than the sequential one so "alive" reads as living tissue, not as the low
    # end of a magnitude. `dead` is already 0..1, so it is not normalized.
    "dead": (((0.16, 0.5, 0.16, 1.0), (0.42, 0.34, 0.22, 1.0)), 'LINEAR'),
    # Calendar phase 4: autumn discoloration stage on foliage, 0 (green) to 1
    # (turned). Its own green -> amber -> brown ramp, the actual progression a
    # leaf goes through, so the debug view reads like autumn rather than like a
    # magnitude. Already 0..1, so not normalized.
    "autumn": (((0.16, 0.5, 0.16, 1.0), (0.85, 0.55, 0.13, 1.0),
                (0.45, 0.26, 0.12, 1.0)), 'LINEAR'),
    # Calendar phase 4: fruit ripeness, 0 (just set) to 1 (ripe). Its own green ->
    # amber -> red ramp, the progression a ripening fruit reads as, rather than a
    # magnitude. Already 0..1, so not normalized. Plant-level, so every fruit
    # shows one colour at a given Day -- what the ripeness socket carries.
    "ripeness": (((0.20, 0.5, 0.16, 1.0), (0.90, 0.75, 0.10, 1.0),
                  (0.75, 0.12, 0.10, 1.0)), 'LINEAR'),
}
"""Which palette each debug view is drawn in, as data rather than as arguments
at the call site, so `_debug_viewers` and `reset_debug_palettes` cannot drift
apart. `color` has no entry because it is shown straight through with no ramp
-- it is already the thing being looked at."""


def _ramp_positions(palette, mode: str) -> list:
    """Where a palette's stops sit. A smooth ramp spans 0..1 end to end; a
    CONSTANT (categorical / banded) one gives each colour an equal band,
    stops at i / n -- at i / (n - 1) the last colour was only reached at
    exactly 1.0, and category 1 fell into category 0's band (Kind, Axis and
    Structural Depth showed one or two colours; found 2026-10-08)."""
    n = len(palette)
    if mode == 'CONSTANT':
        return [i / n for i in range(n)]
    return [0.0 if n == 1 else i / (n - 1) for i in range(n)]


def _stamp_ramp(ramp, palette, mode: str) -> None:
    """Write a palette onto a Color Ramp, replacing whatever is on it."""
    ramp.color_ramp.interpolation = mode
    elements = ramp.color_ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])
    for i, (color, position) in enumerate(zip(palette, _ramp_positions(palette, mode),
                                              strict=True)):
        element = elements[0] if i == 0 else elements.new(position)
        element.position = position
        element.color = color


DEBUG_MODES = (
    "color", "vcol_r", "vcol_g", "vcol_b", "vcol_a", "height", "shade", "mask_preview",
    "component", "order", "structural_depth", "axis_id", "node_index", "birth_step",
    "width", "kind", "bloom_order", "bloom", "organ_scale", "curvature", "load",
    "stiffness", "moment", "bloom_birth", "dead", "autumn", "ripeness",
)
"""`PLANTGEN_PG_growth.debug_view_mode`'s identifiers, in the Debug View
group's switch order: the Debug Mode socket is 1 + a mode's index here (0 off)."""

_DEBUG_GROUP = "Debug View"
_DEBUG_RAMP_LABEL = "Debug Ramp: "


def debug_mode_index(mode: str) -> int:
    """The Debug Mode socket's value for a mode (0: off or unknown)."""
    return DEBUG_MODES.index(mode) + 1 if mode in DEBUG_MODES else 0


def reset_debug_palettes(ng=None, mask_preview_keep=None) -> int:
    """Re-stamp every debug ramp (in the Debug View group) from `_DEBUG_RAMPS`,
    and return how many were found wrong.

    `mask_preview_keep` (an RGBA tuple) overrides the Mask Preview's 'kept' colour
    -- the active component's own colour, so its preview reads in that colour. It
    goes through the same compare-then-stamp path, so an unchanged colour is not
    needlessly re-stamped (which would dirty the group and re-evaluate it).

    A ramp is a normal, editable node, so a stray drag silently rewrites a
    palette; running this whenever a debug view is switched on lets that heal
    itself, rather than needing a version bump to rebuild the whole group.
    """
    ng = ng or bpy.data.node_groups.get(_SUB_PREFIX + _DEBUG_GROUP)
    if ng is None:
        return 0
    ramps = {n.label[len(_DEBUG_RAMP_LABEL):]: n for n in ng.nodes
             if n.bl_idname == "ShaderNodeValToRGB" and n.label.startswith(_DEBUG_RAMP_LABEL)}
    wrong = 0
    for key, (palette, mode) in _DEBUG_RAMPS.items():
        if key == "mask_preview" and mask_preview_keep is not None:
            palette = (palette[0], tuple(mask_preview_keep))
        ramp = ramps.get(key)
        if ramp is None:
            continue
        want = _ramp_positions(palette, mode)
        elements = ramp.color_ramp.elements
        have = [e.position for e in elements]
        positions_ok = (len(have) == len(want)
                        and all(abs(a - b) <= 1e-4
                                for a, b in zip(have, want, strict=True)))
        colors_ok = len(elements) == len(palette) and all(
            all(abs(c - d) <= 1e-4 for c, d in zip(tuple(e.color), col, strict=True))
            for e, col in zip(elements, palette, strict=True))
        # Only re-stamp a ramp that is actually wrong: a ColorRamp edit dirties
        # the group and re-evaluates every plant using it.
        if ramp.color_ramp.interpolation != mode or not positions_ok or not colors_ok:
            wrong += 1
            _stamp_ramp(ramp, palette, mode)
    return wrong


def _debug_fields(nodes, links, geo):
    """Every debug view's colour field, in `DEBUG_MODES` order, read on `geo`
    (the plant REALIZED -- leaves, flowers and fruit are instances until then,
    and a field on the unrealized plant reads their organ attributes as 0).

    Most attributes here are not 0..1 -- `order` is a small integer, `axis_id`
    runs to the axis count, `width` is in metres -- so a colour ramp over them
    raw would be almost entirely one colour. `normalize` runs the attribute
    through its own Attribute Statistic min/max into a Map Range first, so each
    view uses its full contrast range whatever the plant. The colour channels
    are already 0..1 and skip it."""
    fields = {}

    def tap(key: str, value_socket, normalize: bool = True,
            cycle: int = 0, bands: int = 0):
        palette, mode = _DEBUG_RAMPS[key]
        if cycle:
            # value mod cycle, then the middle of its own band: (k + 0.5) / cycle
            # against stops at i / n (`_ramp_positions`)
            wrap = nodes.new("ShaderNodeMath")
            wrap.operation = 'WRAP'
            wrap.inputs[1].default_value = float(cycle)
            wrap.inputs[2].default_value = 0.0
            links.new(value_socket, wrap.inputs[0])
            centred = nodes.new("ShaderNodeMath")
            centred.operation = 'ADD'
            centred.inputs[1].default_value = 0.5
            links.new(wrap.outputs["Value"], centred.inputs[0])
            spread = nodes.new("ShaderNodeMath")
            spread.operation = 'DIVIDE'
            spread.inputs[1].default_value = float(cycle)
            links.new(centred.outputs["Value"], spread.inputs[0])
            value_socket = spread.outputs["Value"]
            normalize = False
        if bands:
            step = nodes.new("ShaderNodeMath")
            step.operation = 'SNAP'
            step.inputs[1].default_value = 1.0 / float(bands)
            links.new(value_socket, step.inputs[0])
            value_socket = step.outputs["Value"]
        if normalize:
            stat = nodes.new("GeometryNodeAttributeStatistic")
            stat.data_type = 'FLOAT'
            links.new(geo, stat.inputs["Geometry"])
            links.new(value_socket, stat.inputs["Attribute"])
            remap = nodes.new("ShaderNodeMapRange")
            links.new(value_socket, remap.inputs["Value"])
            links.new(stat.outputs["Min"], remap.inputs["From Min"])
            links.new(stat.outputs["Max"], remap.inputs["From Max"])
            value_socket = remap.outputs["Result"]
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.label = _DEBUG_RAMP_LABEL + key
        _stamp_ramp(ramp, palette, mode)
        links.new(value_socket, ramp.inputs["Fac"])
        fields[key] = ramp.outputs["Color"]

    def as_float(name):
        return _named(nodes, name, 'FLOAT', 0, 0)

    def as_int(name):
        return _named(nodes, name, 'INT', 0, 0)

    # The combined Color, straight through: it is the thing being looked at.
    fields["color"] = _named(nodes, "Color", 'FLOAT_COLOR', 0, 0)
    # Component Colours: each organ in its component's swatch colour; stems and
    # unclaimed organs carry none (alpha 0), mixed to a neutral grey.
    colour = _named(nodes, "component_color", 'FLOAT_COLOR', 0, 0)
    alpha = nodes.new("FunctionNodeSeparateColor")
    links.new(colour, alpha.inputs["Color"])
    mix = _mix_rgba(nodes)
    _mix_in(mix, "A_Color").default_value = (0.35, 0.35, 0.35, 1.0)
    links.new(alpha.outputs["Alpha"], _mix_in(mix, "Factor_Float"))
    links.new(colour, _mix_in(mix, "B_Color"))
    fields["component"] = _mix_out(mix)

    # Channels: already 0..1, each shown in its own colour.
    tap("vcol_r", as_float(_VCOL_R_ATTR), normalize=False)
    tap("vcol_g", _g_random_socket(nodes, links, 0, 0), normalize=False)
    tap("vcol_b", as_float("instance_random"), normalize=False)
    tap("vcol_a", as_float("height_fraction"), normalize=False)
    # Height, 0-1 root to tip over stem and organs together (the Height mask).
    tap("height", as_float("height_fraction"), normalize=False)
    # Self-shading and the mask preview: present after a regrow that computed
    # them (this view on); 0 and 1 are fixed meanings, so not normalized.
    tap("shade", as_float("shade"), normalize=False)
    tap("mask_preview", as_float("mask_preview"), normalize=False)
    # Structure: categories in distinct hues, magnitudes on the sequential ramp.
    tap("order", as_int("order"), cycle=len(_CATEGORICAL))
    tap("structural_depth", as_int("structural_depth"), cycle=len(_CATEGORICAL))
    tap("axis_id", as_int("axis_id"), cycle=len(_CATEGORICAL))
    tap("kind", as_int("kind"), cycle=5)  # leaf/flower/bud/apex/fruit
    tap("node_index", as_int("node_index"))
    tap("birth_step", as_float("birth_step"))
    tap("width", as_float("width"))
    # The gravity solver's readouts, normalized per plant (raw ranges span
    # orders of magnitude between a wiry stem and a woody one).
    tap("curvature", as_float("curvature"))
    tap("load", as_float("load"))
    tap("stiffness", as_float("stiffness"))
    tap("moment", as_float("moment"))
    # The bloom system: all three already 0..1 by construction.
    tap("bloom_order", as_float("bloom_start"), normalize=False)
    tap("bloom", as_float("bloom"), normalize=False, bands=_BLOOM_BANDS)
    tap("organ_scale", as_float("scale"), normalize=False)
    tap("bloom_birth", as_float("bloom_birth"), normalize=False)
    # Lifecycle: 0..1 already.
    tap("dead", as_float("dead"), normalize=False)
    tap("autumn", as_float("autumn"), normalize=False)
    tap("ripeness", as_float("ripeness"), normalize=False)
    return [fields[key] for key in DEBUG_MODES]


def _build_debug_view():
    """The debug view, taken only while a mode is on (a Switch's untaken branch
    is never evaluated, so off costs nothing): the plant realized, the chosen
    mode's colour written as `Color` -- what Solid's Attribute colouring and the
    debug material show -- and the debug material on every face.

    It replaces 27 Viewer nodes: activating one ran `node.activate_viewer`, a
    2.5 s synchronous re-evaluation with node-editor logging (called twice per
    click, and again on every regrow while a view was on), and showed nothing
    without a Node Editor open on the tree. A mode switch is now one socket
    push. The debug colour lives only in this branch's output, so the plant
    with no mode on is exactly what ships; the add-on's own bakes and exports
    switch the mode off first (`push_debug(obj, off=True)`)."""
    sub, nodes, links, gin, gout = _new_group(
        _DEBUG_GROUP,
        (("Geometry", 'NodeSocketGeometry'), ("Mode", 'NodeSocketInt'),
         ("Material", 'NodeSocketMaterial'),
         (_DEAD_COLOR_SOCKET, 'NodeSocketColor', (0.35, 0.28, 0.20, 1.0))),
        (("Geometry", 'NodeSocketGeometry'),))
    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(gin.outputs["Geometry"], realize.inputs[0])
    composed = _group_node(nodes, links, _build_compose_vertex_colors(), {
        "Geometry": realize.outputs[0], _DEAD_COLOR_SOCKET: gin.outputs[_DEAD_COLOR_SOCKET]})
    geo = composed.outputs["Geometry"]
    before = set(nodes)
    fields = _debug_fields(nodes, links, geo)
    field_nodes = [n for n in nodes if n not in before]
    pick = nodes.new("GeometryNodeIndexSwitch")
    pick.data_type = 'RGBA'
    while len(pick.index_switch_items) < len(fields):
        pick.index_switch_items.new()
    for i, field in enumerate(fields):
        links.new(field, pick.inputs[i + 1])
    index = nodes.new("ShaderNodeMath")
    index.operation = 'SUBTRACT'
    links.new(gin.outputs["Mode"], index.inputs[0])
    index.inputs[1].default_value = 1.0
    links.new(index.outputs["Value"], pick.inputs["Index"])
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = 'FLOAT_COLOR'
    store.domain = 'POINT'
    store.inputs["Name"].default_value = "Color"
    links.new(geo, store.inputs["Geometry"])
    links.new(pick.outputs[0], store.inputs["Value"])
    dressed = nodes.new("GeometryNodeSetMaterial")
    links.new(store.outputs["Geometry"], dressed.inputs["Geometry"])
    links.new(gin.outputs["Material"], dressed.inputs["Material"])
    links.new(dressed.outputs["Geometry"], gout.inputs["Geometry"])
    _tidy(sub, [
        ("Realize: organ attributes onto every vertex; Red composed",
         [realize, composed]),
        ("Every view's colour (ramps restored by reset_debug_palettes)", field_nodes),
        ("Pick the mode's colour (Mode - 1)", [index, pick]),
        ("Write it as Color + the debug material", [store, dressed]),
    ])
    return sub


def _g_random_socket(nodes, links, x: int, y: int):
    """A second Random Value matching `_vertex_colors`' own, so the Green debug
    view shows the same per-organ value the shader will read. Cheaper than
    threading the original's socket out through the call chain, and identical
    by construction: same data type, same range, same seed, same ID."""
    organ_id = _named(nodes, "organ_id", 'INT', x - 200, y)
    node = nodes.new("FunctionNodeRandomValue")
    node.data_type = 'FLOAT'
    node.inputs["Min"].default_value = 0.0
    node.inputs["Max"].default_value = 1.0
    node.inputs["Seed"].default_value = 0
    links.new(organ_id, node.inputs["ID"])
    node.location = (x, y)
    return node.outputs["Value"]


def _find_node_editor(ng):
    """A window/area/region showing `ng` in a node editor, or all-None. The
    `node.activate_viewer` operator genuinely requires a real editor context --
    it cannot be run data-only."""
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type != 'NODE_EDITOR':
                continue
            for space in area.spaces:
                if space.type == 'NODE_EDITOR' and space.node_tree == ng:
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    return win, area, region, space
    return None, None, None, None


# ---------------------------------------------------------------------------
# Sub-groups. The main tree is assembled from named node groups, one per
# separable stage, plus a few small reusable ones. Each wraps the SAME helper the
# flat tree used to call, with interface sockets named exactly as that helper
# reads them off `gin` -- so the computation is node-for-node what it was, only
# encapsulated (verified by fingerprinting every plant's evaluated output before
# and after). Rebuilt with the main group on every version bump.
# ---------------------------------------------------------------------------

_SUB_PREFIX = "PlantGen · "


def _new_group(name: str, inputs, outputs):
    """A fresh sub-group `PlantGen · <name>` (replacing a stale one of that
    name) with the given interface; returns (tree, nodes, links, gin, gout).
    `inputs` items are (name, socket_type) or (name, socket_type, default)."""
    full = _SUB_PREFIX + name
    old = bpy.data.node_groups.get(full)
    if old is not None:
        bpy.data.node_groups.remove(old, do_unlink=True)
    sub = bpy.data.node_groups.new(full, 'GeometryNodeTree')
    for item in inputs:
        sock = sub.interface.new_socket(item[0], in_out='INPUT', socket_type=item[1])
        if len(item) > 2:
            sock.default_value = item[2]
    for sock_name, sock_type in outputs:
        sub.interface.new_socket(sock_name, in_out='OUTPUT', socket_type=sock_type)
    gin = sub.nodes.new("NodeGroupInput")
    gout = sub.nodes.new("NodeGroupOutput")
    return sub, sub.nodes, sub.links, gin, gout


def _group_node(nodes, links, sub, feeds=None):
    """A Group node running `sub`, its inputs linked by socket name from `feeds`."""
    node = nodes.new("GeometryNodeGroup")
    node.node_tree = sub
    for sock_name, socket in (feeds or {}).items():
        links.new(socket, node.inputs[sock_name])
    return node


_PROFILE_INPUTS = (
    (_PROFILE_SIDES_SOCKET, 'NodeSocketInt', 32),
    (_PROFILE_MODE_SOCKET, 'NodeSocketInt'),
    (_PROFILE_LOBE_COUNT_SOCKET, 'NodeSocketInt'),
    (_PROFILE_LOBE_DEPTH_SOCKET, 'NodeSocketFloat'),
    (_PROFILE_FLATTEN_SOCKET, 'NodeSocketFloat'),
    (_PROFILE_OBJECT_SOCKET, 'NodeSocketObject'),
)
_STEM_INPUTS = (
    (_STEM_SMOOTH_SOCKET, 'NodeSocketInt', 4),
    (_STEM_UV_SIZE_SOCKET, 'NodeSocketFloat', 0.025),
    (_STEM_CHECKER_SOCKET, 'NodeSocketBool', False),
    (_STEM_CHECKER_MAT_SOCKET, 'NodeSocketMaterial'),
    (_STEM_BARK_SOCKET, 'NodeSocketFloat', 0.3),
    (_STEM_TWIST_SOCKET, 'NodeSocketFloat', 0.1),
    (_STEM_FLARE_SOCKET, 'NodeSocketFloat', 0.5),
    (_STEM_LOBES_SOCKET, 'NodeSocketInt', 5),
    (_STEM_WELD_METHOD_SOCKET, 'NodeSocketInt', 0),
    (_STEM_MATERIALS_ON_SOCKET, 'NodeSocketBool', False),
    *((name, 'NodeSocketMaterial') for name in _STEM_MATERIAL_SOCKETS),
    (_CATKIN_MATERIAL_ON_SOCKET, 'NodeSocketBool', False),
    (_CATKIN_MATERIAL_SOCKET, 'NodeSocketMaterial'),
)
_BLOOM_INPUTS = (
    (_BLOOM_SPREAD_SOCKET, 'NodeSocketFloat', 0.5),
    (_BLOOM_FROM_AGE_SOCKET, 'NodeSocketBool'),
    (_BLOOM_PROGRESS_SOCKET, 'NodeSocketFloat', 0.0),
    (_BLOOM_ANIMATE_SOCKET, 'NodeSocketBool'),
)


def _tidy(ng, frames=None) -> None:
    """Lay a tree out cleanly (framed, no overlaps, collapsed sources); purely
    cosmetic, so a missing/failed layout never breaks the build."""
    try:
        from . import gn_layout
    except ImportError:
        return
    gn_layout.tidy(ng, frames)


def _upstream(node, stop=()) -> list:
    """`node` and every node feeding it, not crossing a Group Input or `stop` --
    how a helper's internal parts are picked out for a frame without the helper
    having to return them."""
    stop_ids = {n.as_pointer() for n in stop}
    seen, out, stack = set(), [], [node]
    while stack:
        n = stack.pop()
        key = n.as_pointer()
        if key in seen or key in stop_ids or n.bl_idname == 'NodeGroupInput':
            continue
        seen.add(key)
        out.append(n)
        for sock in n.inputs:
            stack.extend(link.from_node for link in sock.links)
    return out


def _feeder(node, socket):
    """The node linked into `node`'s input `socket` (name or index)."""
    return node.inputs[socket].links[0].from_node


def _build_cross_section():
    sub, nodes, links, gin, gout = _new_group(
        "Stem Cross-Section", _PROFILE_INPUTS, (("Curve", 'NodeSocketGeometry'),))
    flatten = _stem_profile(nodes, links, gin, 0, 0).node
    links.new(flatten.outputs["Geometry"], gout.inputs["Curve"])
    pick = _feeder(flatten, "Geometry")
    lobed = _feeder(pick, "False")
    _tidy(sub, [
        ("Lobed circle: radius 1 + cos(lobes x angle) x depth", _upstream(lobed)),
        ("Custom profile object", [_feeder(pick, "True"), _feeder(pick, "Switch")]),
        ("Pick the shape, then Flatten", [pick, flatten] + _upstream(
            _feeder(flatten, "Scale"))),
    ])
    return sub


def _math(nodes, links, operation: str, a, b):
    """A Math node `a <operation> b` (either a socket or a number); its output."""
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    for slot, value in enumerate((a, b)):
        if isinstance(value, (int, float)):
            node.inputs[slot].default_value = float(value)
        else:
            links.new(value, node.inputs[slot])
    return node.outputs[0]


def _build_stem_uv():
    """The swept tubes' `UVMap`: U around from the cross-section's own `pg_u`
    (its spline factor, 0..(n-1)/n on a cyclic profile) x the circumference there
    over the UV Size, V along
    from the axis's `pg_v` (see the Stem Tube). Written per face corner so the
    seam can wrap: on the one face that closes the ring, the corners at u = 0
    belong at 1 -- found as a u = 0 corner on a face whose mean u is large.
    Works inside the per-axis instances, before the weld, which carries it."""
    sub, nodes, links, gin, gout = _new_group(
        "Stem UVs", (("Mesh", 'NodeSocketGeometry'),
                     (_STEM_UV_SIZE_SOCKET, 'NodeSocketFloat', 0.025)),
        (("Mesh", 'NodeSocketGeometry'),))
    u = _named(nodes, "pg_u", 'FLOAT', 0, 0)
    face_u = nodes.new("GeometryNodeFieldOnDomain")
    face_u.domain = 'FACE'
    face_u.data_type = 'FLOAT'
    links.new(u, face_u.inputs[0])
    # A corner well below its face's mean u sits across the ring's 0|1 seam.
    # (Was "u == 0 on a face with mean > 0.25", which only held for the swept
    # section's exact 0; sampled u -- the volume weld -- is never exactly 0.)
    wrap = _math(nodes, links, 'GREATER_THAN', _math(
        nodes, links, 'SUBTRACT', face_u.outputs[0], u), 0.35)
    # U around: the circumference there over one tile, so a thin stalk takes a
    # fraction of a tile and a trunk several -- the same texel size on both.
    span = _math(nodes, links, 'DIVIDE', _math(
        nodes, links, 'MULTIPLY', _named(nodes, "width", 'FLOAT', 0, 0), 2.0 * math.pi),
        gin.outputs[_STEM_UV_SIZE_SOCKET])
    around = _math(nodes, links, 'MULTIPLY', _math(nodes, links, 'ADD', u, wrap), span)
    uv = nodes.new("ShaderNodeCombineXYZ")
    links.new(around, uv.inputs["X"])
    links.new(_named(nodes, "pg_v", 'FLOAT', 0, 0), uv.inputs["Y"])
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = 'FLOAT2'
    store.domain = 'CORNER'
    store.inputs["Name"].default_value = "UVMap"
    links.new(gin.outputs["Mesh"], store.inputs["Geometry"])
    links.new(uv.outputs["Vector"], store.inputs["Value"])
    links.new(store.outputs["Geometry"], gout.inputs["Mesh"])
    _tidy(sub, [
        ("U around: profile factor, seam face wrapped to 1, x circumference / UV Size",
         _upstream(around.node)),
        ("Store UVMap (face corner) from U + the axis's V", [uv, store] + _upstream(
            _feeder(uv, "Y"))),
    ])
    return sub


def _vmath(nodes, links, operation: str, a, b=None, scale=None):
    """A Vector Math node; `scale` feeds the Scale input (for 'SCALE')."""
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    links.new(a, node.inputs[0])
    if b is not None:
        if isinstance(b, (tuple, list)):
            node.inputs[1].default_value = b
        else:
            links.new(b, node.inputs[1])
    if scale is not None:
        if isinstance(scale, (int, float)):
            node.inputs["Scale"].default_value = float(scale)
        else:
            links.new(scale, node.inputs["Scale"])
    return node.outputs["Vector"]


def _noise(nodes, links, vector, seed, detail: float):
    """4-D noise of `vector` (W = the plant seed, so each seed gnarls its own
    way); its node, for Fac (0..1) or Color (three 0..1 channels)."""
    node = nodes.new("ShaderNodeTexNoise")
    node.noise_dimensions = '4D'
    links.new(vector, node.inputs["Vector"])
    links.new(_math(nodes, links, 'MULTIPLY', seed, 0.6180339), node.inputs["W"])
    node.inputs["Scale"].default_value = 1.0
    node.inputs["Detail"].default_value = detail
    return node


def _build_gnarl():
    """Gnarl: a smooth warp of space at a few trunk diameters' wavelength, so
    stems wander and kink a little instead of running ruler-straight between
    nodes. It is a function of *position only*, applied identically to the
    skeleton and to the organ points (by the main graph), so a leaf stays
    exactly on its node however the stem moves. Sized by the plant's widest
    stem (`Reference`), and eased in over the bottom few diameters so the base
    stays planted."""
    sub, nodes, links, gin, gout = _new_group(
        "Gnarl", (("Geometry", 'NodeSocketGeometry'), ("Reference", 'NodeSocketGeometry'),
                  ("Gnarl", 'NodeSocketFloat', 0.3), (_SEED_SOCKET, 'NodeSocketInt')),
        (("Geometry", 'NodeSocketGeometry'),))
    widest = nodes.new("GeometryNodeAttributeStatistic")
    widest.domain = 'POINT'
    links.new(gin.outputs["Reference"], widest.inputs["Geometry"])
    links.new(_named(nodes, "width", 'FLOAT', 0, 0), widest.inputs["Attribute"])
    links.new(_math(nodes, links, 'LESS_THAN', _named(nodes, "flare", 'FLOAT', 0, 0), 1e-4),
              widest.inputs["Selection"])  # the flare is not the stem's width
    ref_z = nodes.new("ShaderNodeSeparateXYZ")
    links.new(nodes.new("GeometryNodeInputPosition").outputs[0], ref_z.inputs[0])
    lowest = nodes.new("GeometryNodeAttributeStatistic")
    lowest.domain = 'POINT'
    links.new(gin.outputs["Reference"], lowest.inputs["Geometry"])
    links.new(ref_z.outputs["Z"], lowest.inputs["Attribute"])
    position = nodes.new("GeometryNodeInputPosition").outputs[0]
    wavelength = _math(nodes, links, 'MULTIPLY', widest.outputs["Max"], 6.0)
    coords = _vmath(nodes, links, 'SCALE', position,
                    scale=_math(nodes, links, 'DIVIDE', 1.0, wavelength))
    noise = _noise(nodes, links, coords, gin.outputs[_SEED_SOCKET], 2.0)
    centred = _vmath(nodes, links, 'SUBTRACT', noise.outputs["Color"], (0.5, 0.5, 0.5))
    here_z = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position, here_z.inputs[0])
    anchor = nodes.new("ShaderNodeMapRange")
    anchor.clamp = True
    anchor.interpolation_type = 'SMOOTHSTEP'
    links.new(_math(nodes, links, 'SUBTRACT', here_z.outputs["Z"], lowest.outputs["Min"]),
              anchor.inputs["Value"])
    links.new(wavelength, anchor.inputs["From Max"])
    reach = _math(nodes, links, 'MULTIPLY', _math(
        nodes, links, 'MULTIPLY', gin.outputs["Gnarl"], widest.outputs["Max"]), 3.0)
    offset = _vmath(nodes, links, 'SCALE', centred,
                    scale=_math(nodes, links, 'MULTIPLY', reach, anchor.outputs["Result"]))
    moved = nodes.new("GeometryNodeSetPosition")
    links.new(gin.outputs["Geometry"], moved.inputs["Geometry"])
    links.new(offset, moved.inputs["Offset"])
    links.new(moved.outputs["Geometry"], gout.inputs["Geometry"])
    _tidy(sub, [
        ("Plant scale: widest stem, lowest point", [widest, lowest, ref_z]
         + _upstream(_feeder(widest, "Attribute"))
         + _upstream(_feeder(widest, "Selection"))
         + _upstream(_feeder(ref_z, 0))),
        ("Noise at a few trunk diameters, eased in above the base",
         _upstream(offset.node, stop=(widest, lowest))),
        ("Warp", [moved]),
    ])
    return sub


def _build_stem_surface():
    """The stem's surface detail, displaced along the normal after the weld (so
    it never moves the centre line or detaches an organ):

    - **Bark** -- noise lumps scaled to the *local* radius, about a radius and a
      half across (+-25 % at Bark 1).
    - **Flare lobes** -- buttresses round the trunk flare: (0.5 + 0.5 cos(n
      theta))^2 around the section (theta from the swept `pg_u`, so Twist spirals
      them), weighted by the skeleton's `flare` falloff and the Flare amount,
      mean-centred so the flare's volume stays what the width says."""
    sub, nodes, links, gin, gout = _new_group(
        "Stem Surface", (("Mesh", 'NodeSocketGeometry'), ("Bark", 'NodeSocketFloat', 0.3),
                         ("Flare", 'NodeSocketFloat', 0.5), ("Flare Lobes", 'NodeSocketInt', 5),
                         (_SEED_SOCKET, 'NodeSocketInt')),
        (("Mesh", 'NodeSocketGeometry'),))
    width = _named(nodes, "width", 'FLOAT', 0, 0)
    bark_width = _named(nodes, "width", 'FLOAT', 0, 0)
    position = nodes.new("GeometryNodeInputPosition").outputs[0]

    def band(across: float, amplitude: float, detail: float):
        coords = _vmath(nodes, links, 'SCALE', position, scale=_math(
            nodes, links, 'DIVIDE', 1.0, _math(nodes, links, 'MULTIPLY', bark_width, across)))
        fac = _noise(nodes, links, coords, gin.outputs[_SEED_SOCKET], detail).outputs["Fac"]
        return _math(nodes, links, 'MULTIPLY', _math(nodes, links, 'SUBTRACT', fac, 0.5),
                     2.0 * amplitude)

    # One band only. A fine grain band (0.3 radii) was tried and removed: it is
    # far finer than the ring spacing along a stem, so it aliased into long
    # shading streaks -- grain is a normal-map / texture job, not geometry.
    bark = _math(nodes, links, 'MULTIPLY', band(1.5, 0.25, 2.0), gin.outputs["Bark"])

    theta = _math(nodes, links, 'MULTIPLY', _math(
        nodes, links, 'MULTIPLY', _named(nodes, "pg_u", 'FLOAT', 0, 0), 2.0 * math.pi),
        gin.outputs["Flare Lobes"])
    ridge = _math(nodes, links, 'ADD', _math(
        nodes, links, 'MULTIPLY', _math(nodes, links, 'COSINE', theta, 0.0), 0.5), 0.5)
    lobe = _math(nodes, links, 'SUBTRACT', _math(nodes, links, 'MULTIPLY', ridge, ridge), 0.375)
    has_lobes = _math(nodes, links, 'GREATER_THAN', gin.outputs["Flare Lobes"], 0.5)
    lobes = _math(nodes, links, 'MULTIPLY', _math(
        nodes, links, 'MULTIPLY', lobe, _named(nodes, "flare", 'FLOAT', 0, 0)), _math(
        nodes, links, 'MULTIPLY', gin.outputs["Flare"], _math(
            nodes, links, 'MULTIPLY', has_lobes, 0.45)))

    push = _math(nodes, links, 'MULTIPLY', _math(nodes, links, 'ADD', bark, lobes), width)
    offset = _vmath(nodes, links, 'SCALE', nodes.new("GeometryNodeInputNormal").outputs[0],
                    scale=push)
    moved = nodes.new("GeometryNodeSetPosition")
    links.new(gin.outputs["Mesh"], moved.inputs["Geometry"])
    links.new(offset, moved.inputs["Offset"])
    links.new(moved.outputs["Geometry"], gout.inputs["Mesh"])
    _tidy(sub, [
        ("Bark: lumps sized to the local radius", _upstream(bark.node)),
        ("Flare lobes: (0.5 + 0.5 cos n theta)^2, by the flare weight",
         _upstream(lobes.node)),
        ("Push along the normal x local radius", [moved] + _upstream(
            offset.node, stop=(bark.node, lobes.node))),
    ])
    return sub


def _uv_factor(nodes, links, curve):
    """`curve` with its own spline factor stored as `pg_u` on its points --
    the U a cross-section contributes to the swept tube (see `_build_stem_uv`)."""
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = 'FLOAT'
    store.domain = 'POINT'
    store.inputs["Name"].default_value = "pg_u"
    links.new(curve, store.inputs["Geometry"])
    links.new(nodes.new("GeometryNodeSplineParameter").outputs["Factor"], store.inputs["Value"])
    return store.outputs["Geometry"]


def _build_stem_tube(cross_section, stem_uv, surface):
    """Skeleton mesh -> Mesh to Curve -> Bezier (Auto) through the nodes -> Curve
    to Mesh with the cross-section, scaled by the `width` attribute straight into
    Curve to Mesh's Scale: the preview of a plant whose weld is still being
    built (Weld Method 0). A built plant's skeleton input IS its welded stem
    mesh (`stem_mesher`, Weld Method 2) and passes straight through.

    The skeleton writes one unbroken chain per axis (`bridge._stem_arrays`), so
    each axis is one spline and one tube. Smoothness 1 is the old polyline;
    higher rounds every bend while still passing through each node, so organs
    stay on the stem; the tubes are simply overlapping solids."""
    sub, nodes, links, gin, gout = _new_group(
        "Stem Tube", (("Skeleton", 'NodeSocketGeometry'), *_PROFILE_INPUTS, *_STEM_INPUTS,
                      (_SEED_SOCKET, 'NodeSocketInt')),
        (("Mesh", 'NodeSocketGeometry'),))
    width = _named(nodes, "width", 'FLOAT', 0, 0)
    to_curve = nodes.new("GeometryNodeMeshToCurve")
    links.new(gin.outputs["Skeleton"], to_curve.inputs["Mesh"])
    # Bezier with Auto handles, not Catmull-Rom: Blender's uniform Catmull-Rom
    # kinks where control points are unevenly spaced (the flare's close points
    # under a long internode bunched the rings into a crease), while Auto
    # handles scale with the neighbouring segments.
    bezier = nodes.new("GeometryNodeCurveSplineType")
    bezier.spline_type = 'BEZIER'
    links.new(to_curve.outputs["Curve"], bezier.inputs["Curve"])
    rounded = nodes.new("GeometryNodeCurveSetHandles")
    rounded.handle_type = 'AUTO'
    links.new(bezier.outputs["Curve"], rounded.inputs["Curve"])
    # Resolution per spline, by how thick the axis is: the trunk gets the full
    # Smoothness, a twig or pedicel (whose facets nobody can see) scales down --
    # full everywhere quadrupled the heaviest test plant to ~0.5 M verts.
    # Share = the spline's mean width over the plant's widest point (a perennial's
    # pedicel is still ~1/3 of its stem, so this is a gentle ramp, not a cut).
    axis_width = _named(nodes, "width", 'FLOAT', 0, 0)
    spline_width = nodes.new("GeometryNodeFieldOnDomain")
    spline_width.domain = 'CURVE'
    spline_width.data_type = 'FLOAT'
    links.new(axis_width, spline_width.inputs[0])
    widest = nodes.new("GeometryNodeAttributeStatistic")
    widest.domain = 'POINT'
    links.new(to_curve.outputs["Curve"], widest.inputs["Geometry"])
    links.new(axis_width, widest.inputs["Attribute"])
    # The flared base is not the plant's width: measured without it, or the
    # flare would shrink every other axis's smoothness and tier.
    links.new(_math(nodes, links, 'LESS_THAN', _named(nodes, "flare", 'FLOAT', 0, 0), 1e-4),
              widest.inputs["Selection"])
    share = _math(nodes, links, 'DIVIDE', spline_width.outputs[0], widest.outputs["Max"])
    share = _math(nodes, links, 'MINIMUM', share, 1.0)
    # Rings by length, not per control point: each axis is resampled so its
    # ring spacing is its own section-side length x 6/Smoothness (1.5 at the
    # default 4 -- near-square quads). Rings placed only at the nodes (x a
    # resolution) made a trunk's quads ~10x taller than wide, and the weld's
    # boolean sliced those into slivers that streaked the shading at every seam.
    # The Bezier is evaluated finely first so the resample follows the curve.
    resolution = nodes.new("GeometryNodeSetSplineResolution")
    links.new(rounded.outputs["Curve"], resolution.inputs["Geometry"])
    resolution.inputs["Resolution"].default_value = 8
    # Three tiers of cross-section by the axis's share of the widest: all the
    # profile's points from 0.6 up, two-thirds from 0.3, a third below (min 6).
    # Two tiers (cut at 0.5) put a branch just under half the trunk on a third
    # of the sides -- facets 3x the parent's, a visible step at its weld.
    tier = _math(nodes, links, 'ADD', _math(nodes, links, 'LESS_THAN', share, 0.6),
                 _math(nodes, links, 'LESS_THAN', share, 0.3))

    def tier_sides(fraction):
        return _math(nodes, links, 'MAXIMUM', _math(
            nodes, links, 'CEIL', _math(nodes, links, 'MULTIPLY',
                                         gin.outputs[_PROFILE_SIDES_SOCKET], fraction), 0.0), 6.0)

    mid_count, coarse_count = tier_sides(2.0 / 3.0), tier_sides(1.0 / 3.0)
    sides = nodes.new("GeometryNodeIndexSwitch")
    sides.data_type = 'FLOAT'
    links.new(tier, sides.inputs["Index"])
    links.new(gin.outputs[_PROFILE_SIDES_SOCKET], sides.inputs[1])
    links.new(mid_count, sides.inputs[2])
    sides.index_switch_items.new()
    links.new(coarse_count, sides.inputs[3])
    side_length = _math(nodes, links, 'DIVIDE', _math(
        nodes, links, 'MULTIPLY', spline_width.outputs[0], 2.0 * math.pi), sides.outputs["Output"])
    spacing = _math(nodes, links, 'MULTIPLY', side_length, _math(
        nodes, links, 'DIVIDE', 6.0, _math(
            nodes, links, 'MAXIMUM', gin.outputs[_STEM_SMOOTH_SOCKET], 1.0)))
    # ...but never closer than the plant's height / 300: square quads on a
    # grass's metre of 1 mm culm put a ring every 0.9 mm (Meadow foxtail 309k
    # verts), where a long quad on a straight thin culm costs nothing visible.
    height_z = nodes.new("ShaderNodeSeparateXYZ")
    links.new(nodes.new("GeometryNodeInputPosition").outputs[0], height_z.inputs[0])
    height = nodes.new("GeometryNodeAttributeStatistic")
    height.domain = 'POINT'
    links.new(to_curve.outputs["Curve"], height.inputs["Geometry"])
    links.new(height_z.outputs["Z"], height.inputs["Attribute"])
    spacing = _math(nodes, links, 'MAXIMUM', spacing,
                    _math(nodes, links, 'DIVIDE', height.outputs["Range"], 300.0))
    rings = nodes.new("GeometryNodeResampleCurve")
    rings.keep_last_segment = True
    rings.inputs["Mode"].default_value = "Length"
    links.new(resolution.outputs["Geometry"], rings.inputs["Curve"])
    links.new(spacing, rings.inputs["Length"])
    # V along each axis: its arc length over the UV Size -- one texel size over
    # the whole plant, matching U (`Stem UVs`) and the built mesh (it was one
    # tile per the axis's mean circumference: a tapering branch's checker came
    # out squashed at its base and stretched at its tip). Stored on the curve;
    # the sweep carries it.
    along = nodes.new("GeometryNodeSplineParameter")
    v = _math(nodes, links, 'DIVIDE', along.outputs["Length"],
              gin.outputs[_STEM_UV_SIZE_SOCKET])
    with_v = nodes.new("GeometryNodeStoreNamedAttribute")
    with_v.data_type = 'FLOAT'
    with_v.domain = 'POINT'
    with_v.inputs["Name"].default_value = "pg_v"
    links.new(rings.outputs["Curve"], with_v.inputs["Geometry"])
    links.new(v, with_v.inputs["Value"])
    # Twist: the section turns about the axis as it runs -- Twist full turns per
    # 20 mean radii of length -- which spirals the UVs, any lobed section and
    # the flare lobes (they are read off the swept section's own angle).
    tilt = _math(nodes, links, 'MULTIPLY', _math(
        nodes, links, 'DIVIDE', along.outputs["Length"], _math(
            nodes, links, 'MULTIPLY', spline_width.outputs[0], 20.0)), _math(
        nodes, links, 'MULTIPLY', gin.outputs[_STEM_TWIST_SOCKET], 2.0 * math.pi))
    twisted = nodes.new("GeometryNodeSetCurveTilt")
    links.new(with_v.outputs["Geometry"], twisted.inputs["Curve"])
    links.new(tilt, twisted.inputs["Tilt"])
    # The tiers (above), each swept with its own section -- 32 sides on every
    # pedicel was most of the stem's vertex count.
    profile = _group_node(nodes, links, cross_section,
                          {name: gin.outputs[name] for name, *_ in _PROFILE_INPUTS})
    tier_parts, sections = [], []
    for index, count in ((0, None), (1, mid_count), (2, coarse_count)):
        part = nodes.new("GeometryNodeSeparateGeometry")
        part.domain = 'CURVE'
        links.new(twisted.outputs["Curve"], part.inputs["Geometry"])
        is_tier = nodes.new("ShaderNodeMath")
        is_tier.operation = 'COMPARE'
        links.new(tier, is_tier.inputs[0])
        is_tier.inputs[1].default_value = float(index)
        is_tier.inputs[2].default_value = 0.5
        links.new(is_tier.outputs[0], part.inputs["Selection"])
        section = profile.outputs["Curve"]
        if count is not None:
            fewer = nodes.new("GeometryNodeResampleCurve")
            links.new(section, fewer.inputs["Curve"])
            links.new(count, fewer.inputs["Count"])
            section = fewer.outputs["Curve"]
        tier_parts.append(part)
        sections.append(_uv_factor(nodes, links, section))

    def sweep(curve, section):
        tube = nodes.new("GeometryNodeCurveToMesh")
        tube.inputs["Fill Caps"].default_value = True
        links.new(curve, tube.inputs["Curve"])
        links.new(section, tube.inputs["Profile Curve"])
        links.new(width, tube.inputs["Scale"])
        return tube

    swept = [sweep(part.outputs["Selection"], section)
             for part, section in zip(tier_parts, sections, strict=True)]
    tubes = nodes.new("GeometryNodeJoinGeometry")
    for tube in reversed(swept):
        links.new(tube.outputs["Mesh"], tubes.inputs["Geometry"])
    uvs = _group_node(nodes, links, stem_uv, {
        "Mesh": tubes.outputs["Geometry"],
        _STEM_UV_SIZE_SOCKET: gin.outputs[_STEM_UV_SIZE_SOCKET]})
    # UV Checker: the stem wears the generated checker material instead of its
    # own, to judge the UVs. Everything else about the stem is unchanged.
    # Weld Method = Built (2): the bridge already swept and welded the stems in
    # Python -- the "skeleton" input *is* the stem mesh; pass it straight on.
    # Else (0, a preview while the weld is built) these plain swept tubes.
    is_built = nodes.new("FunctionNodeCompare")
    is_built.data_type = 'INT'
    is_built.operation = 'EQUAL'
    links.new(gin.outputs[_STEM_WELD_METHOD_SOCKET], is_built.inputs["A"])
    is_built.inputs["B"].default_value = 2
    built = nodes.new("GeometryNodeSwitch")
    built.input_type = 'GEOMETRY'
    links.new(is_built.outputs["Result"], built.inputs["Switch"])
    links.new(uvs.outputs["Mesh"], built.inputs["False"])
    links.new(gin.outputs["Skeleton"], built.inputs["True"])
    detail = _group_node(nodes, links, surface, {
        "Mesh": built.outputs["Output"], "Bark": gin.outputs[_STEM_BARK_SOCKET],
        "Flare": gin.outputs[_STEM_FLARE_SOCKET],
        "Flare Lobes": gin.outputs[_STEM_LOBES_SOCKET],
        _SEED_SOCKET: gin.outputs[_SEED_SOCKET]})
    # Each level's material, by the stems' own `level` (the cross-section
    # level: a pedicel takes the branches' look), before the checker.
    level = _named(nodes, "level", 'INT', 0, 0)
    dressed = detail.outputs["Mesh"]
    level_mats = [level.node]
    last = len(_STEM_MATERIAL_SOCKETS) - 1
    for i, name in enumerate(_STEM_MATERIAL_SOCKETS):
        is_level = nodes.new("FunctionNodeCompare")
        is_level.data_type = 'INT'
        is_level.operation = 'GREATER_EQUAL' if i == last else 'EQUAL'
        links.new(level, is_level.inputs["A"])
        is_level.inputs["B"].default_value = i
        use = nodes.new("FunctionNodeBooleanMath")
        use.operation = 'AND'
        links.new(is_level.outputs["Result"], use.inputs[0])
        links.new(gin.outputs[_STEM_MATERIALS_ON_SOCKET], use.inputs[1])
        dress = nodes.new("GeometryNodeSetMaterial")
        links.new(dressed, dress.inputs["Geometry"])
        links.new(use.outputs["Boolean"], dress.inputs["Selection"])
        links.new(gin.outputs[name], dress.inputs["Material"])
        dressed = dress.outputs["Geometry"]
        level_mats += [is_level, use, dress]
    # The catkins' own material over their twig's (`catkin` > 0, only written
    # while a flower component has a Catkin Material).
    catkin = _named(nodes, "catkin", 'INT', 0, 0)
    is_catkin = nodes.new("FunctionNodeCompare")
    is_catkin.data_type = 'INT'
    is_catkin.operation = 'GREATER_THAN'
    links.new(catkin, is_catkin.inputs["A"])
    is_catkin.inputs["B"].default_value = 0
    use_catkin = nodes.new("FunctionNodeBooleanMath")
    use_catkin.operation = 'AND'
    links.new(is_catkin.outputs["Result"], use_catkin.inputs[0])
    links.new(gin.outputs[_CATKIN_MATERIAL_ON_SOCKET], use_catkin.inputs[1])
    dress_catkin = nodes.new("GeometryNodeSetMaterial")
    links.new(dressed, dress_catkin.inputs["Geometry"])
    links.new(use_catkin.outputs["Boolean"], dress_catkin.inputs["Selection"])
    links.new(gin.outputs[_CATKIN_MATERIAL_SOCKET], dress_catkin.inputs["Material"])
    dressed = dress_catkin.outputs["Geometry"]
    level_mats += [catkin.node, is_catkin, use_catkin, dress_catkin]
    checker = nodes.new("GeometryNodeSetMaterial")
    links.new(dressed, checker.inputs["Geometry"])
    links.new(gin.outputs[_STEM_CHECKER_SOCKET], checker.inputs["Selection"])
    links.new(gin.outputs[_STEM_CHECKER_MAT_SOCKET], checker.inputs["Material"])
    links.new(checker.outputs["Geometry"], gout.inputs["Mesh"])
    _tidy(sub, [
        ("Skeleton -> one smooth spline per axis", [to_curve, bezier, rounded, resolution]),
        ("Per-axis share of the widest -> tier, ring spacing", _upstream(
            spacing.node, stop=(to_curve,)) + [rings]),
        ("Cross-section: all, 2/3 and 1/3 of its points", [profile] + [
            n for section in sections for n in _upstream(
                section.node, stop=(profile, mid_count.node, coarse_count.node))]),
        ("Sweep each tier", [
            n for part, tube, section in zip(tier_parts, swept, sections, strict=True)
            for n in _upstream(tube, stop=(section.node, width.node, twisted, tier.node))]),
        ("V along each axis: square texels at its mean width", _upstream(
            with_v, stop=(resolution, spline_width))),
        ("Twist: the section turns along the axis", [twisted] + _upstream(
            tilt.node, stop=(along, spline_width))),
        ("Join, then UV", [width.node, tubes, uvs]),
        ("Built (2): the bridge's stem mesh passes through", [is_built, built]),
        ("Bark + flare lobes", [detail]),
        ("Each level's material, then the UV Checker override", level_mats + [checker]),
    ])
    return sub


def _build_organ_points():
    """The organs object's points, minus the ones the bridge masked out
    (`render_drop` > 0.5), with `mask_preview` re-stored so it rides onto every
    instance (see the note in the stage body)."""
    sub, nodes, links, gin, gout = _new_group(
        "Organ Points", ((_ATTACHMENTS_SOCKET, 'NodeSocketObject'),),
        (("Points", 'NodeSocketGeometry'),))
    info = nodes.new("GeometryNodeObjectInfo")
    info.transform_space = 'ORIGINAL'
    links.new(gin.outputs[_ATTACHMENTS_SOCKET], info.inputs["Object"])
    # Instancing masks: drop the organ points the bridge marked, before anything
    # is split by kind or instanced. `render_drop` is 0/1 (0 keep); a plant with no
    # active mask has no such attribute, and a missing Named Attribute reads 0 ->
    # nothing deleted, so this is a transparent pass-through when unused.
    dropped = nodes.new("FunctionNodeCompare")
    dropped.data_type = 'FLOAT'
    dropped.operation = 'GREATER_THAN'
    links.new(_named(nodes, "render_drop", 'FLOAT', 0, 0), dropped.inputs["A"])
    dropped.inputs["B"].default_value = 0.5
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = 'POINT'
    links.new(info.outputs["Geometry"], delete.inputs["Geometry"])
    links.new(dropped.outputs["Result"], delete.inputs["Selection"])
    # Re-store `mask_preview` on the points here, inside the graph. A point
    # attribute that merely arrives on the input mesh does NOT survive Instance on
    # Points -- read on an instanced organ it resolves to the asset's own (absent)
    # attribute, 0. An attribute *stored on the points here* is broadcast onto
    # every vertex of that organ's instance when it is realized, so the Mask
    # Preview view reads the right per-organ value on every organ.
    points = _store(nodes, links, delete.outputs["Geometry"], "mask_preview", 'FLOAT',
                    _named(nodes, "mask_preview", 'FLOAT', 0, 0), 0, 0)
    # The Component Colours view's colour, carried the same way and for the
    # same reason (absent unless that view is on: reads transparent black).
    colours = _store(nodes, links, points, "component_color", 'FLOAT_COLOR',
                     _named(nodes, "component_color", 'FLOAT_COLOR', 0, 0), 0, 0)
    links.new(colours, gout.inputs["Points"])
    _tidy(sub, [("Drop masked-out points", [dropped, delete]),
                ("Carry Mask Preview onto the organs", [points.node]),
                ("Carry Component Colours onto the organs", [colours.node])])
    return sub


def _build_split_by_kind():
    """One point cloud in, one per organ type out, by `kind` (0 leaf, 1 flower,
    2/3 bud/apex, 4 fruit) and a leaf's `form` (0 foliage, 2 bract)."""
    sub, nodes, links, gin, gout = _new_group(
        "Split Organs by Kind", (("Points", 'NodeSocketGeometry'),),
        (("Foliage", 'NodeSocketGeometry'), ("Bracts", 'NodeSocketGeometry'),
         ("Flowers", 'NodeSocketGeometry'), ("Buds", 'NodeSocketGeometry'),
         ("Fruit", 'NodeSocketGeometry')))
    kind = _named(nodes, "kind", 'INT', 0, 0)

    def compare(value, op, b):
        node = nodes.new("FunctionNodeCompare")
        node.data_type = 'INT'
        node.operation = op
        links.new(value, node.inputs["A"])
        node.inputs["B"].default_value = b
        return node.outputs["Result"]

    def separate(geo, selection):
        node = nodes.new("GeometryNodeSeparateGeometry")
        node.domain = 'POINT'
        links.new(geo, node.inputs["Geometry"])
        links.new(selection, node.inputs["Selection"])
        return node

    leaf = separate(gin.outputs["Points"], compare(kind, 'EQUAL', 0))
    # Fruit (`kind` 4) comes off the non-leaf side FIRST, before the bud test --
    # which is `>= 2` and would otherwise swallow it.
    fruit = separate(leaf.outputs["Inverted"], compare(kind, 'EQUAL', 4))
    # Buds and apices (2, 3) share one collection; the rest are flowers.
    bud = separate(fruit.outputs["Inverted"], compare(kind, 'GREATER_EQUAL', 2))
    # Leaves split by `form`: foliage (0) and bracts (2), each its own collection.
    # (A catkin rides no instancing path -- it is a curve in the skeleton.)
    form = separate(leaf.outputs["Selection"],
                    compare(_named(nodes, "form", 'INT', 0, 0), 'EQUAL', 0))
    links.new(form.outputs["Selection"], gout.inputs["Foliage"])
    links.new(form.outputs["Inverted"], gout.inputs["Bracts"])
    links.new(bud.outputs["Inverted"], gout.inputs["Flowers"])
    links.new(bud.outputs["Selection"], gout.inputs["Buds"])
    links.new(fruit.outputs["Selection"], gout.inputs["Fruit"])
    kind_node = kind.node

    def stage(separate_node):
        return [separate_node] + [n for n in _upstream(_feeder(separate_node, "Selection"))
                                  if n.as_pointer() != kind_node.as_pointer()]

    _tidy(sub, [
        ("Read kind", [kind_node]),
        ("Leaves (kind 0) vs the rest", stage(leaf)),
        ("Fruit (kind 4) off the rest", stage(fruit)),
        ("Buds + apices (kind 2/3) vs flowers", stage(bud)),
        ("Foliage (form 0) vs bracts", stage(form)),
    ])
    return sub


def _build_asset_index():
    """Reusable: the asset index a type's instances pick -- the legacy seeded
    random over `Asset Count` (no components), or the bridge-baked
    `instance_index` into the master union (components on). Also returns the
    max index (count - 1)."""
    sub, nodes, links, gin, gout = _new_group(
        "Asset Index",
        (("Asset Count", 'NodeSocketInt'), ("Use Components", 'NodeSocketBool'),
         (_SEED_SOCKET, 'NodeSocketInt')),
        (("Index", 'NodeSocketInt'), ("Max Index", 'NodeSocketInt')))
    max_index = nodes.new("ShaderNodeMath")
    max_index.operation = 'SUBTRACT'
    links.new(gin.outputs["Asset Count"], max_index.inputs[0])
    max_index.inputs[1].default_value = 1
    index = _component_index(nodes, links, gin, max_index.outputs["Value"],
                             _named(nodes, "organ_id", 'INT', 0, 0), "Use Components", 0, 0)
    links.new(index, gout.inputs["Index"])
    links.new(max_index.outputs["Value"], gout.inputs["Max Index"])
    _tidy(sub)
    return sub


def _build_instance_assets(label: str, primary_axis: str, secondary_axis: str):
    """Reusable: instance a collection's objects on points, picked by index,
    oriented by the organ's `direction`/`normal` (two pinned axes) and sized by
    its own `scale`. `Blade` assets run along +Y (leaves), `Disc` assets look
    down +Z (flowers, buds, fruit). Each asset carries its own Red gradient
    (`_asset_gradient`)."""
    sub, nodes, links, gin, gout = _new_group(
        f"Instance Assets ({label})",
        (("Points", 'NodeSocketGeometry'), ("Collection", 'NodeSocketCollection'),
         ("Instance Index", 'NodeSocketInt')),
        (("Instances", 'NodeSocketGeometry'),))
    instances = _asset_pick_instances(
        nodes, links, gin.outputs["Points"], gin.outputs["Collection"], None, None, None,
        _named(nodes, "direction", 'FLOAT_VECTOR', 0, 0),
        _named(nodes, "normal", 'FLOAT_VECTOR', 0, 0), None, 0, 0,
        primary_axis, secondary_axis,
        scale_attr=_named(nodes, "scale", 'FLOAT', 0, 0),
        index_socket=gin.outputs["Instance Index"],
        gradient=_BLADE_RED_PEAK if label == "Blade" else _DISC_RED_PEAK)
    links.new(instances, gout.inputs["Instances"])
    _tidy(sub)
    return sub


def _build_flat_red():
    """Reusable: author the stems' flat vertex-colour Red. Never on organs:
    a point store on instances writes into every instanced asset, over its
    own gradient."""
    sub, nodes, links, gin, gout = _new_group(
        "Flat Vertex Red", (("Geometry", 'NodeSocketGeometry'),),
        (("Geometry", 'NodeSocketGeometry'),))
    links.new(_constant_vcolor_r(nodes, links, gin.outputs["Geometry"], 0, 0),
              gout.inputs["Geometry"])
    _tidy(sub)
    return sub


def _build_simple_organ(name: str, instance_assets, asset_index, ripeness=False):
    """Buds / Fruit: index -> instance (disc, its Red gradient) -> [ripeness]."""
    inputs = [("Points", 'NodeSocketGeometry'), ("Collection", 'NodeSocketCollection'),
              ("Asset Count", 'NodeSocketInt'), ("Use Components", 'NodeSocketBool'),
              (_SEED_SOCKET, 'NodeSocketInt')]
    if ripeness:
        inputs.append((_RIPENESS_SOCKET, 'NodeSocketFloat'))
    sub, nodes, links, gin, gout = _new_group(name, inputs,
                                              (("Instances", 'NodeSocketGeometry'),))
    index = _group_node(nodes, links, asset_index, {
        "Asset Count": gin.outputs["Asset Count"],
        "Use Components": gin.outputs["Use Components"],
        _SEED_SOCKET: gin.outputs[_SEED_SOCKET]})
    inst = _group_node(nodes, links, instance_assets, {
        "Points": gin.outputs["Points"], "Collection": gin.outputs["Collection"],
        "Instance Index": index.outputs["Index"]})
    geo = inst.outputs["Instances"]
    if ripeness:
        # Ripeness from a uniform socket, not a per-organ point attribute: it is
        # plant-level, and a point attribute cannot cross Instance on Points.
        geo = _store(nodes, links, geo, "ripeness", 'FLOAT', gin.outputs[_RIPENESS_SOCKET],
                     0, 0)
    links.new(geo, gout.inputs["Instances"])
    _tidy(sub)
    return sub


LEAF_CURL_ATTR = "leaf_curl"
LEAF_FOLD_ATTR = "leaf_fold"
"""Per leaf point (radians, NOW): its component's Emerging Curl / Emerging
Midrib Fold x (1 - how far it has unfolded) -- written by the bridge only when
some leaf / bract component has one (`instancing._emerging_shapes`)."""


def _build_leaf_unfolding():
    """Leaf Unfolding: young leaf instances bent
    by their emerging shape, realized. Each vertex's rest position in its own
    asset (+Y along the blade, +Z its face) is stored on the instances before
    the realize -- a point store there writes into every instanced mesh, the
    field evaluated in it -- with the asset's length (its Field Min & Max).
    Then, in that space: the halves turned up about the midrib (Y) by
    `leaf_fold`, then the midrib bent into an arc in Y-Z by `leaf_curl` (tip
    toward the face). The change is carried back through the leaf's own frame
    (`direction` / `normal`) and `scale`, so the leaf's origin is never needed."""
    sub, nodes, links, gin, gout = _new_group(
        "Leaf Unfolding", (("Instances", 'NodeSocketGeometry'),),
        (("Geometry", 'NodeSocketGeometry'),))
    position = nodes.new("GeometryNodeInputPosition")
    rest_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position.outputs["Position"], rest_xyz.inputs["Vector"])
    extent = nodes.new("GeometryNodeFieldMinAndMax")
    extent.data_type = 'FLOAT'
    extent.domain = 'POINT'
    links.new(rest_xyz.outputs["Y"], extent.inputs["Value"])
    stored = _store(nodes, links, gin.outputs["Instances"], "leaf_rest", 'FLOAT_VECTOR',
                    position.outputs["Position"], 0, 0)
    stored = _store(nodes, links, stored, "leaf_len", 'FLOAT', extent.outputs["Max"], 0, 0)
    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(stored, realize.inputs["Geometry"])
    store_nodes = [position, rest_xyz, extent, stored.node, realize] + _upstream(stored.node)

    rest = _named(nodes, "leaf_rest", 'FLOAT_VECTOR', 0, 0)
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(rest, xyz.inputs["Vector"])
    x, y, z = xyz.outputs["X"], xyz.outputs["Y"], xyz.outputs["Z"]

    def m(op, a, b=None):
        node = nodes.new("ShaderNodeMath")
        node.operation = op
        for slot, value in enumerate((a, b)):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                node.inputs[slot].default_value = float(value)
            else:
                links.new(value, node.inputs[slot])
        return node.outputs[0]

    # Fold: each half turned up about the midrib, +x and -x alike.
    phi = m('MULTIPLY', _named(nodes, LEAF_FOLD_ATTR, 'FLOAT', 0, 0), m('SIGN', x))
    cphi, sphi = m('COSINE', phi), m('SINE', phi)
    x1 = m('SUBTRACT', m('MULTIPLY', x, cphi), m('MULTIPLY', z, sphi))
    z1 = m('ADD', m('MULTIPLY', x, sphi), m('MULTIPLY', z, cphi))
    fold_nodes = _upstream(x1.node) + _upstream(z1.node)

    # Curl: the midrib bent into an arc of total angle `leaf_curl` over the
    # blade's length -- curvature k = curl / length, theta = k y,
    # y' = sin(theta) / k - z sin(theta), z' = (1 - cos(theta)) / k + z cos(theta).
    # k kept off zero (its sign kept) so a leaf with a fold and no curl stays put.
    k = m('DIVIDE', _named(nodes, LEAF_CURL_ATTR, 'FLOAT', 0, 0),
          m('MAXIMUM', _named(nodes, "leaf_len", 'FLOAT', 0, 0), 1e-6))
    sign = m('SIGN', k)
    sign_nz = m('ADD', sign, m('SUBTRACT', 1.0, m('ABSOLUTE', sign)))
    k_safe = m('MULTIPLY', m('MAXIMUM', m('ABSOLUTE', k), 1e-6), sign_nz)
    theta = m('MULTIPLY', k_safe, y)
    s_t, c_t = m('SINE', theta), m('COSINE', theta)
    y2 = m('SUBTRACT', m('DIVIDE', s_t, k_safe), m('MULTIPLY', z1, s_t))
    z2 = m('ADD', m('DIVIDE', m('SUBTRACT', 1.0, c_t), k_safe), m('MULTIPLY', z1, c_t))
    curl_nodes = [n for n in _upstream(y2.node) + _upstream(z2.node)
                  if n not in fold_nodes]

    bent = nodes.new("ShaderNodeCombineXYZ")
    links.new(x1, bent.inputs["X"])
    links.new(y2, bent.inputs["Y"])
    links.new(z2, bent.inputs["Z"])
    delta = _vmath(nodes, links, 'SCALE', _vmath(nodes, links, 'SUBTRACT',
                                                 bent.outputs["Vector"], rest),
                   scale=_named(nodes, "scale", 'FLOAT', 0, 0))
    frame = nodes.new("FunctionNodeAxesToRotation")
    frame.primary_axis = _LEAF_PRIMARY_AXIS
    frame.secondary_axis = _LEAF_SECONDARY_AXIS
    links.new(_named(nodes, "direction", 'FLOAT_VECTOR', 0, 0), frame.inputs["Primary Axis"])
    links.new(_named(nodes, "normal", 'FLOAT_VECTOR', 0, 0), frame.inputs["Secondary Axis"])
    turned = nodes.new("FunctionNodeRotateVector")
    links.new(delta, turned.inputs["Vector"])
    links.new(frame.outputs["Rotation"], turned.inputs["Rotation"])
    move = nodes.new("GeometryNodeSetPosition")
    links.new(realize.outputs["Geometry"], move.inputs["Geometry"])
    links.new(turned.outputs["Vector"], move.inputs["Offset"])
    out = move.outputs["Geometry"]
    for name in ("leaf_rest", "leaf_len"):          # working data, not output
        drop = nodes.new("GeometryNodeRemoveAttribute")
        drop.inputs["Name"].default_value = name
        links.new(out, drop.inputs["Geometry"])
        out = drop.outputs["Geometry"]
    links.new(out, gout.inputs["Geometry"])
    _tidy(sub, _distinct_frames([
        ("Rest position + length per asset (stored on the instances), realize",
         store_nodes),
        ("Fold: the halves up about the midrib by leaf_fold", fold_nodes),
        ("Curl: the midrib bent into an arc by leaf_curl", curl_nodes),
        ("Back through the leaf's frame and scale, as an offset; working data dropped",
         _upstream(out.node)),
    ]))
    return sub


def _distinct_frames(frames):
    """`_tidy` frames with every node in the first frame that lists it (and
    no Group Input / Output) -- for frames gathered by `_upstream`, which
    overlap where branches share a source."""
    seen, out = set(), []
    for title, members in frames:
        keep = []
        for n in members:
            key = n.as_pointer()
            if key in seen or n.bl_idname in ("NodeGroupInput", "NodeGroupOutput"):
                continue
            seen.add(key)
            keep.append(n)
        if keep:
            out.append((title, keep))
    return out


def _build_leaves(blade_assets, asset_index, unfolding=None):
    """Foliage and bracts, each from its own collection, joined. With
    `unfolding`: a leaf still unfolding (`leaf_curl` / `leaf_fold` non-zero) is
    instanced on its own and bent by it, realized; the rest stay instances --
    and a plant with none takes the plain path (a Switch, so nothing extra is
    evaluated and its output is unchanged)."""
    sub, nodes, links, gin, gout = _new_group(
        "Leaves & Bracts",
        (("Foliage", 'NodeSocketGeometry'), ("Bracts", 'NodeSocketGeometry'),
         ("Leaf Collection", 'NodeSocketCollection'), ("Leaf Count", 'NodeSocketInt'),
         ("Leaf Use Components", 'NodeSocketBool'),
         ("Bract Collection", 'NodeSocketCollection'), ("Bract Count", 'NodeSocketInt'),
         ("Bract Use Components", 'NodeSocketBool'), (_SEED_SOCKET, 'NodeSocketInt')),
        (("Instances", 'NodeSocketGeometry'),))
    streams, frames = [], []
    for part, pts in (("Leaf", "Foliage"), ("Bract", "Bracts")):
        index = _group_node(nodes, links, asset_index, {
            "Asset Count": gin.outputs[f"{part} Count"],
            "Use Components": gin.outputs[f"{part} Use Components"],
            _SEED_SOCKET: gin.outputs[_SEED_SOCKET]})
        inst = _group_node(nodes, links, blade_assets, {
            "Points": gin.outputs[pts], "Collection": gin.outputs[f"{part} Collection"],
            "Instance Index": index.outputs["Index"]})
        if unfolding is None:
            streams.append(inst.outputs["Instances"])
            frames.append((f"{pts}: pick + instance", [index, inst]))
            continue
        # Leaf Unfolding: the young (still curled / folded) apart, bent.
        bend = _math(nodes, links, 'ADD', _math(nodes, links, 'ABSOLUTE', _named(
            nodes, LEAF_CURL_ATTR, 'FLOAT', 0, 0), 0.0), _math(nodes, links, 'ABSOLUTE',
            _named(nodes, LEAF_FOLD_ATTR, 'FLOAT', 0, 0), 0.0))
        young = _math(nodes, links, 'GREATER_THAN', bend, 1e-6)
        split = nodes.new("GeometryNodeSeparateGeometry")
        split.domain = 'POINT'
        links.new(gin.outputs[pts], split.inputs["Geometry"])
        links.new(young, split.inputs["Selection"])
        parts = {}
        for side, sock in (("mature", "Inverted"), ("young", "Selection")):
            parts[side] = _group_node(nodes, links, blade_assets, {
                "Points": split.outputs[sock],
                "Collection": gin.outputs[f"{part} Collection"],
                "Instance Index": index.outputs["Index"]})
        bent = _group_node(nodes, links, unfolding, {
            "Instances": parts["young"].outputs["Instances"]})
        joined = nodes.new("GeometryNodeJoinGeometry")
        links.new(parts["mature"].outputs["Instances"], joined.inputs["Geometry"])
        links.new(bent.outputs["Geometry"], joined.inputs["Geometry"])
        any_young = nodes.new("GeometryNodeAttributeStatistic")
        any_young.data_type = 'FLOAT'
        any_young.domain = 'POINT'
        links.new(gin.outputs[pts], any_young.inputs["Geometry"])
        links.new(young, any_young.inputs["Attribute"])
        switch = nodes.new("GeometryNodeSwitch")
        switch.input_type = 'GEOMETRY'
        links.new(_math(nodes, links, 'GREATER_THAN', any_young.outputs["Max"], 0.5),
                  switch.inputs["Switch"])
        links.new(inst.outputs["Instances"], switch.inputs["False"])
        links.new(joined.outputs["Geometry"], switch.inputs["True"])
        streams.append(switch.outputs["Output"])
        frames.append((f"{pts}: pick + instance", [index, inst]))
        frames.append((f"{pts}: unfolding leaves apart, bent (only when any)",
                       [split, parts["mature"], parts["young"], bent, joined, any_young,
                        _feeder(switch, "Switch"), switch] + _upstream(young.node)))
    join = nodes.new("GeometryNodeJoinGeometry")
    for geo in streams:                       # foliage first, then bracts
        links.new(geo, join.inputs["Geometry"])
    # No flat Red here: each leaf asset carries its own base-to-tip Red
    # (`_asset_gradient`), which a flat store would overwrite in every asset.
    links.new(join.outputs["Geometry"], gout.inputs["Instances"])
    _tidy(sub, _distinct_frames(frames + [("Join", [join])]))
    return sub


def _build_bloom_openness():
    sub, nodes, links, gin, gout = _new_group(
        "Bloom Openness", _BLOOM_INPUTS, (("Openness", 'NodeSocketFloat'),))
    gate = _bloom_openness(nodes, links, gin, 0, 0).node
    links.new(gate.outputs["Result"], gout.inputs["Openness"])
    static = _feeder(gate, "A")
    animated = _feeder(gate, "B")
    start = _feeder(_feeder(animated, 0), 1)       # since = progress - start
    start_nodes = _upstream(start)
    _tidy(sub, [
        ("When it starts: order x Spread (not before its birth, From Age)",
         start_nodes),
        ("Animate: open over the remaining (1 - Spread)",
         _upstream(animated, stop=start_nodes)),
        ("Static: 1 - bloom order", _upstream(static, stop=start_nodes)),
        ("Animate on/off", [gate]),
    ])
    return sub


def _build_bloom_stages():
    sub, nodes, links, gin, gout = _new_group(
        "Bloom Stage Index",
        (("Openness", 'NodeSocketFloat'), ("Stage Count", 'NodeSocketInt')),
        (("Nearest", 'NodeSocketFloat'), ("Lower", 'NodeSocketFloat'),
         ("Upper", 'NodeSocketFloat'), ("Blend", 'NodeSocketFloat')))
    staged = _stage_indices(nodes, links, gin.outputs["Openness"],
                            gin.outputs["Stage Count"], 0, 0)
    for socket, name in zip(staged, ("Nearest", "Lower", "Upper", "Blend"), strict=True):
        links.new(socket, gout.inputs[name])
    _tidy(sub)
    return sub


def _build_bloom_blend():
    sub, nodes, links, gin, gout = _new_group(
        "Bloom Stage Blend",
        (("Lower Instances", 'NodeSocketGeometry'), ("Upper Instances", 'NodeSocketGeometry'),
         ("Blend", 'NodeSocketFloat')),
        (("Geometry", 'NodeSocketGeometry'),))
    links.new(_blend_stages(nodes, links, gin.outputs["Lower Instances"],
                            gin.outputs["Upper Instances"], gin.outputs["Blend"], 0, 0),
              gout.inputs["Geometry"])
    _tidy(sub)
    return sub


def _build_flowers(disc_assets):
    """Flowers as a bud-to-open bloom sequence: openness per flower, the stage it
    shows (inside its component's block of the master when components are on),
    optionally blended between the two nearest stages (Continuous), `bloom`
    published. Red is each asset's own gradient (`_asset_gradient`)."""
    sub, nodes, links, gin, gout = _new_group(
        "Flowers & Bloom",
        (("Points", 'NodeSocketGeometry'), ("Collection", 'NodeSocketCollection'),
         ("Asset Count", 'NodeSocketInt'), ("Use Components", 'NodeSocketBool'),
         (_BLOOM_CONTINUOUS_SOCKET, 'NodeSocketBool'), *_BLOOM_INPUTS),
        (("Instances", 'NodeSocketGeometry'),))
    openness_group = _build_bloom_openness()
    stages_group = _build_bloom_stages()
    blend_group = _build_bloom_blend()

    max_index = nodes.new("ShaderNodeMath")
    max_index.operation = 'SUBTRACT'
    links.new(gin.outputs["Asset Count"], max_index.inputs[0])
    max_index.inputs[1].default_value = 1
    # The collection is read as a bud-to-open sequence -- the index is "how
    # open", not "which variety". With components, each flower stages inside its
    # own component's contiguous block of the master (`flower_block_*`).
    stage_count = nodes.new("GeometryNodeSwitch")
    stage_count.input_type = 'INT'
    links.new(gin.outputs["Use Components"], stage_count.inputs["Switch"])
    links.new(max_index.outputs["Value"], stage_count.inputs["False"])
    links.new(_named(nodes, "flower_block_max", 'INT', 0, 0), stage_count.inputs["True"])
    stage_base = nodes.new("GeometryNodeSwitch")
    stage_base.input_type = 'INT'
    links.new(gin.outputs["Use Components"], stage_base.inputs["Switch"])
    stage_base.inputs["False"].default_value = 0
    links.new(_named(nodes, "flower_block_offset", 'INT', 0, 0), stage_base.inputs["True"])

    openness = _group_node(nodes, links, openness_group,
                           {name: gin.outputs[name] for name, *_ in _BLOOM_INPUTS})
    stages = _group_node(nodes, links, stages_group, {
        "Openness": openness.outputs["Openness"],
        "Stage Count": stage_count.outputs["Output"]})

    def flowers_at(stage_name, points=None):
        add = nodes.new("ShaderNodeMath")
        add.operation = 'ADD'
        links.new(stage_base.outputs["Output"], add.inputs[0])
        links.new(stages.outputs[stage_name], add.inputs[1])
        return _group_node(nodes, links, disc_assets, {
            "Points": points or gin.outputs["Points"],
            "Collection": gin.outputs["Collection"],
            "Instance Index": add.outputs["Value"]}), add

    discrete, add_n = flowers_at("Nearest")
    # Continuous: only a flower BETWEEN two stages is blended (realized); one
    # sitting on a stage -- a bud still waiting, a flower fully open -- is that
    # stage's instance, the blend's own value there. So the cost follows the
    # flowers opening now, not every flower on the plant.
    between = _math(nodes, links, 'MULTIPLY',
                    _math(nodes, links, 'GREATER_THAN', stages.outputs["Blend"], 1e-3),
                    _math(nodes, links, 'LESS_THAN', stages.outputs["Blend"], 1.0 - 1e-3))
    split = nodes.new("GeometryNodeSeparateGeometry")
    split.domain = 'POINT'
    links.new(gin.outputs["Points"], split.inputs["Geometry"])
    links.new(between, split.inputs["Selection"])
    steady, add_s = flowers_at("Nearest", split.outputs["Inverted"])
    lower, add_l = flowers_at("Lower", split.outputs["Selection"])
    upper, add_u = flowers_at("Upper", split.outputs["Selection"])
    blend = _group_node(nodes, links, blend_group, {
        "Lower Instances": lower.outputs["Instances"],
        "Upper Instances": upper.outputs["Instances"],
        "Blend": stages.outputs["Blend"]})
    blended = nodes.new("GeometryNodeJoinGeometry")
    links.new(steady.outputs["Instances"], blended.inputs["Geometry"])
    links.new(blend.outputs["Geometry"], blended.inputs["Geometry"])
    # A Switch's untaken geometry branch is not evaluated, so the blend's
    # extra streams cost nothing while Continuous is off.
    continuous = nodes.new("GeometryNodeSwitch")
    continuous.input_type = 'GEOMETRY'
    links.new(gin.outputs[_BLOOM_CONTINUOUS_SOCKET], continuous.inputs["Switch"])
    links.new(discrete.outputs["Instances"], continuous.inputs["False"])
    links.new(blended.outputs["Geometry"], continuous.inputs["True"])
    # Published so a shader can recolour or fade a flower by how open it is.
    bloomed = _store(nodes, links, continuous.outputs["Output"], "bloom", 'FLOAT',
                     openness.outputs["Openness"], 0, 0)
    links.new(bloomed, gout.inputs["Instances"])
    _tidy(sub, [
        ("How open each flower is", [openness]),
        ("Which bloom stage it shows", [max_index, stage_count, stage_base, stages,
                                        stage_count.inputs["True"].links[0].from_node,
                                        stage_base.inputs["True"].links[0].from_node]),
        ("Instance the stage(s)", [discrete, add_n]),
        ("Continuous: flowers between stages blended, the rest their stage",
         [split, steady, lower, upper, add_s, add_l, add_u, blend, blended, continuous]
         + _upstream(between.node, stop=[stages])),
        ("Publish bloom", [bloomed.node]),
    ])
    return sub


_COMPOSE_GROUP = "Compose Vertex Colours"


def _build_rig_fast_preview():
    """Rig View's Fast Preview: the rig's stems-only skin (`Stems`, posed by
    the armature) joined with the live plant's organ INSTANCES, each moved
    rigidly as its organ bone moves. The bone's motion comes from `Trackers`
    -- three points per organ (`3 x organ_id` + 0 attach / 1 along / 2 across)
    bound to its organ bone, with their rest positions in `rest` -- so an
    instance turns about its rest attach point by the change in that frame
    and moves with it. Organs stay GPU instances: per frame only the stems
    deform, never the 200k vertices of realized flowers."""
    sub, nodes, links, gin, gout = _new_group(
        "Rig Fast Preview",
        (("Plant", 'NodeSocketGeometry'), ("Stems", 'NodeSocketObject'),
         ("Trackers", 'NodeSocketObject')),
        (("Geometry", 'NodeSocketGeometry'),))
    parts = nodes.new("GeometryNodeSeparateComponents")
    links.new(gin.outputs["Plant"], parts.inputs["Geometry"])
    stems = nodes.new("GeometryNodeObjectInfo")
    stems.transform_space = 'RELATIVE'
    links.new(gin.outputs["Stems"], stems.inputs["Object"])
    trackers = nodes.new("GeometryNodeObjectInfo")
    trackers.transform_space = 'RELATIVE'
    links.new(gin.outputs["Trackers"], trackers.inputs["Object"])
    position = nodes.new("GeometryNodeInputPosition")
    rest = _named(nodes, "rest", 'FLOAT_VECTOR', 0, 0)
    oid = _named(nodes, "organ_id", 'INT', 0, 0)

    def math(op, a, b):
        node = nodes.new("ShaderNodeMath")
        node.operation = op
        links.new(a, node.inputs[0])
        node.inputs[1].default_value = b
        return node.outputs["Value"]

    def vec(op, a, b):
        node = nodes.new("ShaderNodeVectorMath")
        node.operation = op
        links.new(a, node.inputs[0])
        links.new(b, node.inputs[1])
        return node.outputs["Vector"]

    def sample(value, index):
        node = nodes.new("GeometryNodeSampleIndex")
        node.data_type = 'FLOAT_VECTOR'
        node.domain = 'POINT'
        links.new(trackers.outputs["Geometry"], node.inputs["Geometry"])
        links.new(value, node.inputs["Value"])
        links.new(index, node.inputs["Index"])
        return node.outputs["Value"]

    base = math('MULTIPLY', oid, 3.0)
    index = [base, math('ADD', base, 1.0), math('ADD', base, 2.0)]
    posed = [sample(position.outputs["Position"], i) for i in index]
    at_rest = [sample(rest, i) for i in index]

    def frame(points):
        node = nodes.new("FunctionNodeAxesToRotation")
        node.primary_axis = 'Y'
        node.secondary_axis = 'X'
        links.new(vec('SUBTRACT', points[1], points[0]), node.inputs["Primary Axis"])
        links.new(vec('SUBTRACT', points[2], points[0]), node.inputs["Secondary Axis"])
        return node.outputs["Rotation"]

    undo = nodes.new("FunctionNodeInvertRotation")
    links.new(frame(at_rest), undo.inputs["Rotation"])
    delta = nodes.new("FunctionNodeRotateRotation")
    delta.rotation_space = 'GLOBAL'
    links.new(undo.outputs["Rotation"], delta.inputs["Rotation"])
    links.new(frame(posed), delta.inputs["Rotate By"])
    turn = nodes.new("GeometryNodeRotateInstances")
    links.new(parts.outputs["Instances"], turn.inputs["Instances"])
    links.new(delta.outputs["Rotation"], turn.inputs["Rotation"])
    links.new(at_rest[0], turn.inputs["Pivot Point"])
    turn.inputs["Local Space"].default_value = False
    move = nodes.new("GeometryNodeTranslateInstances")
    links.new(turn.outputs["Instances"], move.inputs["Instances"])
    links.new(vec('SUBTRACT', posed[0], at_rest[0]), move.inputs["Translation"])
    move.inputs["Local Space"].default_value = False
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(stems.outputs["Geometry"], join.inputs["Geometry"])
    links.new(move.outputs["Instances"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], gout.inputs["Geometry"])
    _tidy(sub)
    return sub


def _build_compose_vertex_colors():
    """On a REALIZED plant: Color's Red from each vertex's own `vcol_r` (the
    leaf gradient, which only a leaf's own vertices have), dead-tinted like the
    other three channels. Green / Blue / Alpha are per organ / plant and
    arrive on the instances' Color. Used by the debug view and the add-on's
    own bakes / exports (`rig._realize_full`); a plain File > Export realizes
    outside the graph, so there the gradient stays in `vcol_r` beside Color."""
    sub, nodes, links, gin, gout = _new_group(
        _COMPOSE_GROUP,
        (("Geometry", 'NodeSocketGeometry'),
         (_DEAD_COLOR_SOCKET, 'NodeSocketColor', (0.35, 0.28, 0.20, 1.0))),
        (("Geometry", 'NodeSocketGeometry'),))
    colour = nodes.new("FunctionNodeSeparateColor")
    links.new(_named(nodes, "Color", 'FLOAT_COLOR', 0, 0), colour.inputs["Color"])
    dead_rgb = nodes.new("FunctionNodeSeparateColor")
    links.new(gin.outputs[_DEAD_COLOR_SOCKET], dead_rgb.inputs["Color"])
    red = nodes.new("ShaderNodeMix")
    red.data_type = 'FLOAT'
    red.clamp_factor = True
    links.new(_named(nodes, "dead", 'FLOAT', 0, 0), _mix_in(red, "Factor_Float"))
    links.new(_named(nodes, _VCOL_R_ATTR, 'FLOAT', 0, 0), _mix_in(red, "A_Float"))
    links.new(dead_rgb.outputs["Red"], _mix_in(red, "B_Float"))
    combine = nodes.new("FunctionNodeCombineColor")
    links.new(next(s for s in red.outputs if s.identifier == "Result_Float"),
              combine.inputs["Red"])
    for channel in ("Green", "Blue", "Alpha"):
        links.new(colour.outputs[channel], combine.inputs[channel])
    stored = _store(nodes, links, gin.outputs["Geometry"], "Color", 'FLOAT_COLOR',
                    combine.outputs["Color"], 0, 0)
    links.new(stored, gout.inputs["Geometry"])
    _tidy(sub, [("Red from vcol_r (dead-tinted), the rest from Color",
                 [n for n in nodes if n.bl_idname not in ("NodeGroupInput", "NodeGroupOutput")])])
    return sub


def _build_vertex_colors():
    sub, nodes, links, gin, gout = _new_group(
        "Quixel Vertex Colours",
        (("Geometry", 'NodeSocketGeometry'),
         (_DEAD_COLOR_SOCKET, 'NodeSocketColor', (0.35, 0.28, 0.20, 1.0))),
        (("Geometry", 'NodeSocketGeometry'), ("Color", 'NodeSocketColor')))
    stored, color = _vertex_colors(nodes, links, gin, gin.outputs["Geometry"], 0, 0)
    links.new(stored, gout.inputs["Geometry"])
    links.new(color, gout.inputs["Color"])
    mix = color.node
    combine = next(link.from_node for sock in mix.inputs for link in sock.links
                   if link.from_node.bl_idname == "FunctionNodeCombineColor")
    rgba = _upstream(combine)
    _tidy(sub, [
        ("R flat / G per organ / B per plant / A height", rgba),
        ("Tint toward Dead Colour by `dead`", _upstream(mix, stop=rgba)),
        ("Store as the Color attribute", [stored.node]),
    ])
    return sub


def build_mesh_group() -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.new(_MESH_GROUP_NAME, 'GeometryNodeTree')
    # NOTE: the version is stamped at the *end* of this function, not here.
    # The parametric add-on learned this live and it caught us too: if the builder raises
    # partway through, a group stamped up front is left half-built but looking
    # current, so `ensure_gn_modifier` reuses the wreckage on every subsequent
    # call and the failure looks like "my edit did nothing" rather than like a
    # crash. Stamping last makes a partial build self-healing -- it simply has
    # no version and gets rebuilt.
    iface = ng.interface

    iface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    iface.new_socket(_ATTACHMENTS_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    s_sides = iface.new_socket(_PROFILE_SIDES_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    s_sides.default_value = 32
    s_sides.min_value = 3
    # The cross-section. Four numbers and one object cover all ten named
    # shapes -- see BranchLevel.section_inputs for the collapse.
    iface.new_socket(_PROFILE_MODE_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_PROFILE_LOBE_COUNT_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_PROFILE_LOBE_DEPTH_SOCKET, in_out='INPUT', socket_type='NodeSocketFloat')
    iface.new_socket(_PROFILE_FLATTEN_SOCKET, in_out='INPUT', socket_type='NodeSocketFloat')
    iface.new_socket(_PROFILE_OBJECT_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(_GNARL_SOCKET, in_out='INPUT', socket_type='NodeSocketFloat')
    for sock_name, sock_type, *default in _STEM_INPUTS:
        sock = iface.new_socket(sock_name, in_out='INPUT', socket_type=sock_type)
        if default:
            sock.default_value = default[0]
    iface.new_socket(_LEAF_ASSET_SOCKET, in_out='INPUT', socket_type='NodeSocketCollection')
    iface.new_socket(_LEAF_ASSET_COUNT_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_LEAF_USE_COMPONENTS_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_BRACT_ASSET_SOCKET, in_out='INPUT', socket_type='NodeSocketCollection')
    iface.new_socket(_BRACT_ASSET_COUNT_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_BRACT_USE_COMPONENTS_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_BUD_ASSET_SOCKET, in_out='INPUT', socket_type='NodeSocketCollection')
    iface.new_socket(_BUD_ASSET_COUNT_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_BUD_USE_COMPONENTS_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_FLOWER_ASSET_SOCKET, in_out='INPUT', socket_type='NodeSocketCollection')
    iface.new_socket(_FLOWER_ASSET_COUNT_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_FLOWER_USE_COMPONENTS_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_FRUIT_ASSET_SOCKET, in_out='INPUT', socket_type='NodeSocketCollection')
    iface.new_socket(_FRUIT_ASSET_COUNT_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_FRUIT_USE_COMPONENTS_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_RIPENESS_SOCKET, in_out='INPUT', socket_type='NodeSocketFloat')
    # The bloom timeline. Progress is the one socket meant to be keyframed, and
    # is pushed on its own by `push_bloom_progress` rather than through a
    # regrow -- Blender fires a property's update callback every frame during
    # playback, so rebuilding the plant there would make the animation
    # unusable. The parametric add-on learned this and split the same two paths.
    iface.new_socket(_BLOOM_CONTINUOUS_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_BLOOM_ANIMATE_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    s_progress = iface.new_socket(_BLOOM_PROGRESS_SOCKET, in_out='INPUT',
                                  socket_type='NodeSocketFloat')
    s_progress.default_value = 0.0
    s_progress.min_value = 0.0
    s_progress.max_value = 1.0
    s_spread = iface.new_socket(_BLOOM_SPREAD_SOCKET, in_out='INPUT',
                                socket_type='NodeSocketFloat')
    s_spread.default_value = 0.5
    s_spread.min_value = 0.0
    s_spread.max_value = 1.0
    # Whether Progress currently *means* the plant's age. Only then can a
    # flower's own birth be compared against it -- see `_bloom_openness`.
    iface.new_socket(_BLOOM_FROM_AGE_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_SEED_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    s_dead_color = iface.new_socket(_DEAD_COLOR_SOCKET, in_out='INPUT',
                                    socket_type='NodeSocketColor')
    s_dead_color.default_value = (0.35, 0.28, 0.20, 1.0)  # a botanical grey-brown
    iface.new_socket(_DEBUG_MODE_SOCKET, in_out='INPUT', socket_type='NodeSocketInt')
    iface.new_socket(_DEBUG_MAT_SOCKET, in_out='INPUT', socket_type='NodeSocketMaterial')
    iface.new_socket(_RIG_VIEW_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_RIG_SKIN_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(_RIG_FAST_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_RIG_STEMS_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(_RIG_TRACKERS_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(_TL_CACHE_SOCKET, in_out='INPUT', socket_type='NodeSocketBool')
    iface.new_socket(_TL_MESH_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(_TL_POINTS_SOCKET, in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(_TL_REFS_SOCKET, in_out='INPUT', socket_type='NodeSocketCollection')

    nodes = ng.nodes
    links = ng.links
    gin = nodes.new("NodeGroupInput")
    gout = nodes.new("NodeGroupOutput")

    # The stages, each its own named node group (see the sub-group builders
    # above); the small reusable ones are built once and shared.
    asset_index = _build_asset_index()
    blade_assets = _build_instance_assets("Blade", _LEAF_PRIMARY_AXIS, _LEAF_SECONDARY_AXIS)
    disc_assets = _build_instance_assets("Disc", _FLOWER_PRIMARY_AXIS, _FLOWER_SECONDARY_AXIS)
    flat_red = _build_flat_red()

    def feed(*names):
        return {name: gin.outputs[name] for name in names}

    # --- Stem: the skeleton swept into one tapered tube.
    # Gnarl warps the skeleton and (below) the organ points by the same field of
    # position, measured against the unwarped skeleton, so organs stay put.
    gnarl = _build_gnarl()
    skeleton = _group_node(nodes, links, gnarl, {
        "Geometry": gin.outputs["Geometry"], "Reference": gin.outputs["Geometry"],
        "Gnarl": gin.outputs[_GNARL_SOCKET], **feed(_SEED_SOCKET)})
    stem_uv = _build_stem_uv()
    stem = _group_node(nodes, links, _build_stem_tube(_build_cross_section(), stem_uv,
                                                      _build_stem_surface()),
                       {"Skeleton": skeleton.outputs["Geometry"],
                        **feed(*(name for name, *_ in _PROFILE_INPUTS + _STEM_INPUTS),
                               _SEED_SOCKET)})
    stem_red = _group_node(nodes, links, flat_red, {"Geometry": stem.outputs["Mesh"]})

    # --- Organs: the attachment points, masked, then split by kind.
    organs = _group_node(nodes, links, _build_organ_points(), feed(_ATTACHMENTS_SOCKET))
    organs_gnarled = _group_node(nodes, links, gnarl, {
        "Geometry": organs.outputs["Points"], "Reference": gin.outputs["Geometry"],
        "Gnarl": gin.outputs[_GNARL_SOCKET], **feed(_SEED_SOCKET)})
    split = _group_node(nodes, links, _build_split_by_kind(),
                        {"Points": organs_gnarled.outputs["Geometry"]})

    # --- Instancing: one group per organ type. Each picks its asset one of two
    # ways by its use-components switch (Phase 2): the legacy seeded-random index
    # over one collection, or the bridge-baked `instance_index` into the type's
    # master union. Flowers index a bloom stage instead.
    leaves = _group_node(nodes, links, _build_leaves(blade_assets, asset_index,
                                                     _build_leaf_unfolding()), {
        "Foliage": split.outputs["Foliage"], "Bracts": split.outputs["Bracts"],
        "Leaf Collection": gin.outputs[_LEAF_ASSET_SOCKET],
        "Leaf Count": gin.outputs[_LEAF_ASSET_COUNT_SOCKET],
        "Leaf Use Components": gin.outputs[_LEAF_USE_COMPONENTS_SOCKET],
        "Bract Collection": gin.outputs[_BRACT_ASSET_SOCKET],
        "Bract Count": gin.outputs[_BRACT_ASSET_COUNT_SOCKET],
        "Bract Use Components": gin.outputs[_BRACT_USE_COMPONENTS_SOCKET],
        **feed(_SEED_SOCKET)})
    flowers = _group_node(nodes, links, _build_flowers(disc_assets), {
        "Points": split.outputs["Flowers"],
        "Collection": gin.outputs[_FLOWER_ASSET_SOCKET],
        "Asset Count": gin.outputs[_FLOWER_ASSET_COUNT_SOCKET],
        "Use Components": gin.outputs[_FLOWER_USE_COMPONENTS_SOCKET],
        **feed(_BLOOM_CONTINUOUS_SOCKET, *(name for name, *_ in _BLOOM_INPUTS))})
    buds = _group_node(nodes, links,
                       _build_simple_organ("Buds", disc_assets, asset_index), {
                           "Points": split.outputs["Buds"],
                           "Collection": gin.outputs[_BUD_ASSET_SOCKET],
                           "Asset Count": gin.outputs[_BUD_ASSET_COUNT_SOCKET],
                           "Use Components": gin.outputs[_BUD_USE_COMPONENTS_SOCKET],
                           **feed(_SEED_SOCKET)})
    fruit = _group_node(nodes, links,
                        _build_simple_organ("Fruit", disc_assets, asset_index,
                                            ripeness=True), {
                            "Points": split.outputs["Fruit"],
                            "Collection": gin.outputs[_FRUIT_ASSET_SOCKET],
                            "Asset Count": gin.outputs[_FRUIT_ASSET_COUNT_SOCKET],
                            "Use Components": gin.outputs[_FRUIT_USE_COMPONENTS_SOCKET],
                            **feed(_SEED_SOCKET, _RIPENESS_SOCKET)})

    # --- Assemble (stem, leaves, flowers, buds, fruit -- the order the flat
    # graph joined them in) and author the vertex colours.
    join = nodes.new("GeometryNodeJoinGeometry")
    for geo in (stem_red.outputs["Geometry"], leaves.outputs["Instances"],
                flowers.outputs["Instances"], buds.outputs["Instances"],
                fruit.outputs["Instances"]):
        links.new(geo, join.inputs["Geometry"])
    colors = _group_node(nodes, links, _build_vertex_colors(), {
        "Geometry": join.outputs["Geometry"], **feed(_DEAD_COLOR_SOCKET)})
    colored = colors.outputs["Geometry"]

    # Debug view: the finished plant recoloured by one mode, only while a mode
    # is on (Debug Mode > 0); off, the Switch never evaluates that branch.
    debug = _group_node(nodes, links, _build_debug_view(), {
        "Geometry": colored, "Mode": gin.outputs[_DEBUG_MODE_SOCKET],
        "Material": gin.outputs[_DEBUG_MAT_SOCKET],
        _DEAD_COLOR_SOCKET: gin.outputs[_DEAD_COLOR_SOCKET]})
    debug_on = nodes.new("FunctionNodeCompare")
    debug_on.data_type = 'INT'
    debug_on.operation = 'GREATER_THAN'
    links.new(gin.outputs[_DEBUG_MODE_SOCKET], debug_on.inputs["A"])
    debug_on.inputs["B"].default_value = 0
    shown = nodes.new("GeometryNodeSwitch")
    shown.input_type = 'GEOMETRY'
    links.new(debug_on.outputs["Result"], shown.inputs["Switch"])
    links.new(colored, shown.inputs["False"])
    links.new(debug.outputs["Geometry"], shown.inputs["True"])
    debug_nodes = [debug, debug_on, shown]

    # Rig View: the baked skin, posed by its armature, in place of everything
    # above -- which the Switch then never evaluates. Relative, so the skin
    # (parented under the plant via the rig) lands in the plant's own frame.
    skin = nodes.new("GeometryNodeObjectInfo")
    skin.transform_space = 'RELATIVE'
    links.new(gin.outputs[_RIG_SKIN_SOCKET], skin.inputs["Object"])
    # Fast Preview: the stems skin deformed + the live organ instances moved
    # rigidly by their trackers -- 20k deforming vertices, not 250k.
    fast = _group_node(nodes, links, _build_rig_fast_preview(), {
        "Plant": colored, "Stems": gin.outputs[_RIG_STEMS_SOCKET],
        "Trackers": gin.outputs[_RIG_TRACKERS_SOCKET]})
    which = nodes.new("GeometryNodeSwitch")
    which.input_type = 'GEOMETRY'
    links.new(gin.outputs[_RIG_FAST_SOCKET], which.inputs["Switch"])
    links.new(skin.outputs["Geometry"], which.inputs["False"])
    links.new(fast.outputs["Geometry"], which.inputs["True"])
    rigged = nodes.new("GeometryNodeSwitch")
    rigged.input_type = 'GEOMETRY'
    links.new(gin.outputs[_RIG_VIEW_SOCKET], rigged.inputs["Switch"])
    links.new(shown.outputs["Output"], rigged.inputs["False"])
    links.new(which.outputs["Output"], rigged.inputs["True"])

    # Timelapse Cache: a recorded frame in place of all of it (playback only).
    cached = _group_node(nodes, links, _build_timelapse_cache(), {
        "Mesh": gin.outputs[_TL_MESH_SOCKET], "Instances": gin.outputs[_TL_POINTS_SOCKET],
        "Assets": gin.outputs[_TL_REFS_SOCKET]})
    playback = nodes.new("GeometryNodeSwitch")
    playback.input_type = 'GEOMETRY'
    links.new(gin.outputs[_TL_CACHE_SOCKET], playback.inputs["Switch"])
    links.new(rigged.outputs["Output"], playback.inputs["False"])
    links.new(cached.outputs["Geometry"], playback.inputs["True"])

    links.new(playback.outputs["Output"], gout.inputs["Geometry"])

    _tidy(ng, [
        ("Stem (gnarled skeleton -> welded tube)", [skeleton, stem, stem_red]),
        ("Organ points (gnarled alike), split by kind", [organs, organs_gnarled, split]),
        ("Leaves & bracts", [leaves]),
        ("Flowers & bloom", [flowers]),
        ("Buds", [buds]),
        ("Fruit", [fruit]),
        ("Assemble + vertex colours", [join, colors]),
        ("Debug view (only while a mode is on)", debug_nodes),
        ("Rig View: the baked skin, animated, instead of the live plant "
         "(Fast Preview: stems skin + organ instances on their trackers)",
         [skin, fast, which, rigged]),
        ("Timelapse Cache: a recorded frame instead (playback)", [cached, playback]),
    ])

    # Last, so a build that raised partway through is left unstamped and gets
    # rebuilt rather than cached. See the note at the top of this function.
    ng[_PG_VERSION_KEY] = _MESH_GROUP_VERSION
    return ng


def effective_bloom_progress(s) -> float:
    """The Progress value the node group is actually given.

    Two ways to drive it, and both are kept because they answer different
    questions. **From Age** ties bloom to the plant's own maturity: Progress is
    how far the plant is past the point in its life it starts flowering, so
    scrubbing Age grows and blooms it together and no keyframes are needed.
    **Manual** leaves Progress free, which is what you want when the bloom has
    its own timing independent of how mature the plant is -- and it is the mode
    to keyframe, since a keyframe on Age would regrow the plant every frame.

    Deriving rather than adding a curve: the grown plant already knows when it
    first flowered, and `bridge._flowering_age` measures it. A separate
    bloom-timing curve would be a second place to say the same thing and a
    second place for the two to disagree.

    This maps the *plant's* timeline. What it deliberately does not do is
    decide when an individual flower may open -- that needs each flower's own
    birth, which lives on the `bloom_birth` attribute and is applied in
    `_bloom_openness`. Progress reaching 1.0 at Age 1.0 is correct and is not
    the same claim as "every flower is open".
    """
    if s.bloom_mode != 'AGE':
        return float(s.bloom_progress)
    # Calendar species (phase 3): Progress is the Day's position in the bloom
    # window, so the bloom opens across the real dates the species carries and
    # 'blooms for two weeks' is the window width. `bridge.regrow` stores it; -1
    # means no bloom calendar, so fall back to the maturity measure below.
    window = float(s.last_bloom_window)
    if window >= 0.0:
        return min(1.0, max(0.0, window))
    onset = float(s.last_flowering_age)
    return min(1.0, max(0.0, (float(s.age) - onset) / max(1.0 - onset, 1e-4)))


def push_display(obj) -> None:
    """Update the sockets only the node graph reads -- Flare Lobes, Gnarl, the
    dead colour and the UV checker -- and never regrow. None of them reaches
    growth, the skeleton or the stem mesher (Trunk Flare does: it widens the
    stem's base in the skeleton, so it is not here), so a regrow would only
    redo work whose every cache hits, then re-evaluate the graph anyway."""
    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return
    s = obj.plantgen
    for socket, value in ((_STEM_LOBES_SOCKET, int(s.flare_lobes)),
                          (_GNARL_SOCKET, float(s.stem_gnarl)),
                          (_DEAD_COLOR_SOCKET, tuple(s.senescence_dead_color)),
                          (_STEM_CHECKER_SOCKET, bool(s.uv_checker)),
                          (_STEM_CHECKER_MAT_SOCKET,
                           uv_checker_material() if s.uv_checker else None),
                          *stem_material_values(s).items()):
        _set_modifier_input(mod, _socket_identifier(mod.node_group, socket), value)


def stem_material_values(s) -> dict:
    """The per-level material sockets: each level's (own or inherited), the
    last socket for every level past it; off while no level has one."""
    from . import stem_material

    mats = stem_material.level_materials(s)
    values = {_STEM_MATERIALS_ON_SOCKET: any(m is not None for m in mats)}
    for i, name in enumerate(_STEM_MATERIAL_SOCKETS):
        values[name] = mats[i] if i < len(mats) else None
    catkin = catkin_material(s)
    values[_CATKIN_MATERIAL_ON_SOCKET] = catkin is not None
    values[_CATKIN_MATERIAL_SOCKET] = catkin
    return values


def catkin_material(s):
    """The catkins' material: the first live flower component's Catkin
    Material, or None (the catkins wear their twig's)."""
    for c in s.components:
        if c.type == 'flower' and not c.mute and getattr(c, "catkin_material", None):
            return c.catkin_material
    return None


def push_stem_mode(obj) -> None:
    """Point the Stem Tube at what the plant's mesh holds now: the built stem
    mesh (passed through), or the bare skeleton of a preview (swept here) --
    see `ensure_gn_modifier`. For a worker's mesh landing between regrows."""
    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return
    from . import bridge

    preview = bool(obj.get(bridge._STEM_PREVIEW_KEY))
    _set_modifier_input(mod, _socket_identifier(mod.node_group, _STEM_WELD_METHOD_SOCKET),
                        0 if preview else 2)


def push_bloom(obj) -> None:
    """Update the bloom display sockets, and never regrow.

    None of these five touches the grown skeleton -- which stage a flower shows
    is decided entirely in the node group from the `bloom_start` the grow
    already baked. So a regrow here would be pure waste, and for Progress it
    would be worse than waste: Blender fires a property's update callback on
    every frame during playback, exactly as it does while dragging the slider,
    so rebuilding the plant would make the one control meant to be keyframed
    the one control that cannot be played back.

    Spread learned this the hard way. It sat on the regrow path at first, so it
    only took effect once the debounce timer fired -- which never happens
    inside a running script, and reads as "Spread does nothing" from the panel
    if you drag it and look immediately.
    """
    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return
    s = obj.plantgen
    for socket, value in ((_BLOOM_CONTINUOUS_SOCKET, s.bloom_continuous),
                          (_BLOOM_ANIMATE_SOCKET, s.bloom_mode != 'STATIC'),
                          (_BLOOM_PROGRESS_SOCKET, effective_bloom_progress(s)),
                          (_BLOOM_SPREAD_SOCKET, s.bloom_spread),
                          (_BLOOM_FROM_AGE_SOCKET, s.bloom_mode == 'AGE')):
        _set_modifier_input(mod, _socket_identifier(mod.node_group, socket), value)


_DEBUG_MAT_NAME = "PlantGen Debug View"


def compose_vertex_colors_group():
    """The Compose Vertex Colours group (built with the mesh group), or None."""
    return bpy.data.node_groups.get(_SUB_PREFIX + _COMPOSE_GROUP)


def debug_material() -> bpy.types.Material:
    """The material the debug branch wears: its `Color` as emission, so the
    view reads flat and unlit in Material Preview and Rendered (Solid shows the
    same `Color` through Attribute colouring). Made once, shared."""
    mat = bpy.data.materials.get(_DEBUG_MAT_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(_DEBUG_MAT_NAME)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = "Color"
    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    attr.location, emit.location, out.location = (-400, 0), (-150, 0), (100, 0)
    return mat


def debug_values(s, off: bool = False) -> dict:
    """The debug sockets for a plant's current view (or off)."""
    index = 0 if off else debug_mode_index(s.debug_view_mode)
    return {_DEBUG_MODE_SOCKET: index,
            _DEBUG_MAT_SOCKET: debug_material() if index else None}


def _build_timelapse_cache():
    """A recorded Grow Over Timeline frame (`timelapse_cache`): its mesh (stems
    and anything realized) as it was, plus every instance re-made from its
    asset -- Collection Info (separated, in name order) picked by `tl_ref`,
    then given its recorded transform `tl_xf`. Both objects hold the plant's
    own space, so Object Info reads them Original."""
    sub, nodes, links, gin, gout = _new_group(
        "Timelapse Cache",
        (("Mesh", 'NodeSocketObject'), ("Instances", 'NodeSocketObject'),
         ("Assets", 'NodeSocketCollection')),
        (("Geometry", 'NodeSocketGeometry'),))
    mesh_info = nodes.new("GeometryNodeObjectInfo")
    mesh_info.transform_space = 'ORIGINAL'
    links.new(gin.outputs["Mesh"], mesh_info.inputs["Object"])
    points_info = nodes.new("GeometryNodeObjectInfo")
    points_info.transform_space = 'ORIGINAL'
    links.new(gin.outputs["Instances"], points_info.inputs["Object"])
    assets = nodes.new("GeometryNodeCollectionInfo")
    assets.inputs["Separate Children"].default_value = True
    assets.inputs["Reset Children"].default_value = True
    links.new(gin.outputs["Assets"], assets.inputs["Collection"])
    place = nodes.new("GeometryNodeInstanceOnPoints")
    place.inputs["Pick Instance"].default_value = True
    links.new(points_info.outputs["Geometry"], place.inputs["Points"])
    links.new(assets.outputs["Instances"], place.inputs["Instance"])
    links.new(_named(nodes, TL_REF_ATTR, 'INT', 0, 0), place.inputs["Instance Index"])
    xf = nodes.new("GeometryNodeSetInstanceTransform")
    links.new(place.outputs["Instances"], xf.inputs["Instances"])
    links.new(_named(nodes, TL_XF_ATTR, 'FLOAT4X4', 0, 0), xf.inputs["Transform"])
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(mesh_info.outputs["Geometry"], join.inputs["Geometry"])
    links.new(xf.outputs["Instances"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], gout.inputs["Geometry"])
    _tidy(sub, _distinct_frames([
        ("The recorded mesh", [mesh_info]),
        ("Instances re-made: asset by rank, then the recorded transform",
         _upstream(xf)),
        ("Join", [join]),
    ]))
    return sub


def timelapse_values(obj) -> dict:
    """The Timelapse Cache sockets: on only while a recorded frame is shown
    (`timelapse_cache.show`); its helpers by remembered name."""
    on = bool(obj.get("plantgen_tl_showing", False))
    found = (bpy.data.objects.get(obj.get("plantgen_tl_mesh", "")),
             bpy.data.objects.get(obj.get("plantgen_tl_points", "")),
             bpy.data.collections.get(obj.get("plantgen_tl_assets", "")))
    on = on and all(x is not None for x in found)
    return {_TL_CACHE_SOCKET: on, _TL_MESH_SOCKET: found[0] if on else None,
            _TL_POINTS_SOCKET: found[1] if on else None,
            _TL_REFS_SOCKET: found[2] if on else None}


def push_timelapse(obj) -> None:
    """Show a recorded timelapse frame (or the plant): a socket push."""
    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return
    for socket, value in timelapse_values(obj).items():
        _set_modifier_input(mod, _socket_identifier(mod.node_group, socket), value)
    obj.update_tag()


def rig_values(obj) -> dict:
    """The Rig View sockets: on only while the plant's Rig View is on AND its
    baked skin exists (a missing skin would show an empty plant)."""
    from . import rig
    found = rig.find_rig(obj) if obj.plantgen.rig_view else None
    skin = found[1] if found else None
    fast = rig.fast_preview_parts(obj) if skin is not None else None
    return {_RIG_VIEW_SOCKET: skin is not None, _RIG_SKIN_SOCKET: skin,
            _RIG_FAST_SOCKET: fast is not None,
            _RIG_STEMS_SOCKET: fast[0] if fast else None,
            _RIG_TRACKERS_SOCKET: fast[1] if fast else None}


def push_rig_view(obj) -> None:
    """Show the plant's baked rig (or its live build): a socket push."""
    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return
    for socket, value in rig_values(obj).items():
        _set_modifier_input(mod, _socket_identifier(mod.node_group, socket), value)
    # A fresh skin (Build Rig makes a new object under the old name) is not
    # picked up by the socket write alone -- the plant kept showing stale
    # geometry until something else re-evaluated it.
    obj.update_tag()


def push_debug(obj, off: bool = False) -> None:
    """Show the plant's debug view (or, `off`, the plant as it ships): a socket
    push, never a regrow. The Mask Preview's ramp takes the active component's
    colour first."""
    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None or mod.node_group is None:
        return
    s = obj.plantgen
    keep = None
    if s.debug_view_mode == 'mask_preview' and s.mask_preview_level < 0 and \
            0 <= s.components_active < len(s.components):
        col = s.components[s.components_active].color
        keep = (col[0], col[1], col[2], 1.0)
    if not off and s.debug_view_mode != 'off':
        reset_debug_palettes(mask_preview_keep=keep)
    for socket, value in debug_values(s, off).items():
        _set_modifier_input(mod, _socket_identifier(mod.node_group, socket), value)


_SOLID_BEFORE: dict = {}
"""Each Solid 3D view's colouring before a debug view switched it to Attribute,
by the view's pointer, to put back when the view goes off (session only)."""


def set_debug_shading(context, on: bool) -> None:
    """Solid views show the debug colour through Attribute colouring (Material
    Preview / Rendered show the debug material as they are): switch them in,
    and back to what they were when the view goes off."""
    wm = getattr(context, "window_manager", None) or bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                space.overlay.show_viewer_attribute = False   # the retired Viewer overlay
                shading = space.shading
                key = space.as_pointer()
                if on and shading.color_type != 'VERTEX':
                    _SOLID_BEFORE[key] = shading.color_type
                    shading.color_type = 'VERTEX'
                elif not on and key in _SOLID_BEFORE:
                    shading.color_type = _SOLID_BEFORE.pop(key)


_CHECKER_NAME = "PlantGen UV Checker"


def uv_checker_material() -> bpy.types.Material:
    """The shared checker material the UV Checker puts on the stems: Blender's
    generated colour grid (numbered cells, so stretching and seams read at a
    glance) on the stems' `UVMap`. Made once, reused by every plant."""
    mat = bpy.data.materials.get(_CHECKER_NAME)
    if mat is not None:
        return mat
    image = bpy.data.images.get(_CHECKER_NAME)
    if image is None:
        image = bpy.data.images.new(_CHECKER_NAME, 1024, 1024)
        image.generated_type = 'COLOR_GRID'
    mat = bpy.data.materials.new(_CHECKER_NAME)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    if bsdf is not None:
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nodes.active = tex  # Solid > Texture colouring shows the active image node
    tex.location = (-300, 300)
    uv.location = (-500, 300)
    return mat


def ensure_gn_modifier(obj: bpy.types.Object, attachments_obj: bpy.types.Object) -> None:
    ng = bpy.data.node_groups.get(_MESH_GROUP_NAME)
    if ng is not None and ng.get(_PG_VERSION_KEY) != _MESH_GROUP_VERSION:
        bpy.data.node_groups.remove(ng, do_unlink=True)
        ng = None
    if ng is None:
        ng = build_mesh_group()

    mod = obj.modifiers.get(_MESH_GROUP_NAME)
    if mod is None:
        mod = obj.modifiers.new(_MESH_GROUP_NAME, 'NODES')
    if mod.node_group is not ng:
        mod.node_group = ng

    s = obj.plantgen
    from . import bridge, instancing
    # Phase 2: when a type has components, its asset socket carries the master union
    # collection and its switch reads the baked index; with none it stays the legacy
    # single collection (byte-identical). Flowers use the same master, but their
    # index is a per-point bloom-stage block, not a variety pick.
    # A type whose components are all muted has no master and draws nothing --
    # never the legacy collection, which would bring its assets back.
    def _asset_for(tkey, legacy_collection):
        master = instancing.master_collection(obj, tkey)
        if master is None and instancing.has_components(s, tkey):
            return None, True
        return (master if master is not None else legacy_collection), master is not None
    leaf_collection, use_leaf_components = _asset_for("leaf", s.leaf_asset_collection)
    bract_collection, use_bract_components = _asset_for("bract", s.bract_asset_collection)
    bud_collection, use_bud_components = _asset_for("bud", s.bud_asset_collection)
    fruit_collection, use_fruit_components = _asset_for("fruit", s.fruit_asset_collection)
    flower_collection, use_flower_components = _asset_for("flower", s.flower_asset_collection)
    leaf_count = len(s.leaf_asset_collection.objects) if s.leaf_asset_collection else 0
    bract_count = len(s.bract_asset_collection.objects) if s.bract_asset_collection else 0
    bud_count = len(s.bud_asset_collection.objects) if s.bud_asset_collection else 0
    flower_count = len(flower_collection.objects) if flower_collection else 0
    fruit_count = len(s.fruit_asset_collection.objects) if s.fruit_asset_collection else 0
    # The cross-section is decided in the growth core, not here: it is a
    # botanical trait a species carries, so it lives on the profile and the
    # panel is only one way to reach it.
    full_profile = bridge.profile_from_object(obj)
    # The stems are always the Built mesher's (each level its own section, traced
    # there); the node graph's own sweep only draws a preview, so its profile
    # inputs are held at a plain circle.
    profile = {"sides": 32, "mode": 0, "lobe_count": 0, "lobe_depth": 0.0, "flatten": 0.0}
    # Fruit ripeness is a single plant-level value (the fruit window's progress
    # at this Day), passed as a uniform and stored on the fruit instances.
    ripeness = full_profile.fruit_progress(s.day)
    stem_preview = bool(obj.get(bridge._STEM_PREVIEW_KEY))

    values = {
        _ATTACHMENTS_SOCKET: attachments_obj,
        _PROFILE_SIDES_SOCKET: profile["sides"],
        _PROFILE_MODE_SOCKET: profile["mode"],
        _PROFILE_LOBE_COUNT_SOCKET: profile["lobe_count"],
        _PROFILE_LOBE_DEPTH_SOCKET: profile["lobe_depth"],
        _PROFILE_FLATTEN_SOCKET: profile["flatten"],
        _PROFILE_OBJECT_SOCKET: None,
        _STEM_SMOOTH_SOCKET: int(s.stem_smoothness),
        _STEM_UV_SIZE_SOCKET: float(s.uv_size),
        _STEM_CHECKER_SOCKET: bool(s.uv_checker),
        _STEM_CHECKER_MAT_SOCKET: uv_checker_material() if s.uv_checker else None,
        _STEM_BARK_SOCKET: float(s.bark_noise),
        _STEM_TWIST_SOCKET: float(s.stem_twist),
        _STEM_FLARE_SOCKET: float(s.trunk_flare),
        _STEM_LOBES_SOCKET: int(s.flare_lobes),
        _GNARL_SOCKET: float(s.stem_gnarl),
        # 2 (Built): the bridge wrote the finished stem mesh (welded or not, per
        # Weld Junctions) and the Stem Tube passes it through -- or, while a
        # worker builds it (`stem_preview`), 0: the graph's own sweep turns
        # the bare skeleton into plain tubes meanwhile.
        _STEM_WELD_METHOD_SOCKET: 0 if stem_preview else 2,
        **stem_material_values(s),
        _LEAF_ASSET_SOCKET: leaf_collection,
        _LEAF_ASSET_COUNT_SOCKET: leaf_count,
        _LEAF_USE_COMPONENTS_SOCKET: use_leaf_components,
        _BRACT_ASSET_SOCKET: bract_collection,
        _BRACT_ASSET_COUNT_SOCKET: bract_count,
        _BRACT_USE_COMPONENTS_SOCKET: use_bract_components,
        _BUD_ASSET_SOCKET: bud_collection,
        _BUD_ASSET_COUNT_SOCKET: bud_count,
        _BUD_USE_COMPONENTS_SOCKET: use_bud_components,
        _FLOWER_ASSET_SOCKET: flower_collection,
        _FLOWER_ASSET_COUNT_SOCKET: flower_count,
        _FLOWER_USE_COMPONENTS_SOCKET: use_flower_components,
        _FRUIT_ASSET_SOCKET: fruit_collection,
        _FRUIT_ASSET_COUNT_SOCKET: fruit_count,
        _FRUIT_USE_COMPONENTS_SOCKET: use_fruit_components,
        _RIPENESS_SOCKET: ripeness,
        _BLOOM_CONTINUOUS_SOCKET: s.bloom_continuous,
        _BLOOM_ANIMATE_SOCKET: s.bloom_mode != 'STATIC',
        _BLOOM_PROGRESS_SOCKET: effective_bloom_progress(s),
        _BLOOM_SPREAD_SOCKET: s.bloom_spread,
        _BLOOM_FROM_AGE_SOCKET: s.bloom_mode == 'AGE',
        _SEED_SOCKET: s.seed,
        _DEAD_COLOR_SOCKET: tuple(s.senescence_dead_color),
        **debug_values(s),
        **rig_values(obj),
        **timelapse_values(obj),
    }
    relink = instancing.take_relinked(obj)
    for socket_name, value in values.items():
        ident = _socket_identifier(ng, socket_name)
        if _set_modifier_input(mod, ident, value) and (
                value is None or isinstance(value, bpy.types.ID)):
            relink = True
    if relink:
        # Pointing an input at another collection / object -- or a master
        # collection gaining or losing objects -- only re-evaluates the
        # modifier; the depsgraph's relations are not rebuilt, so the instanced
        # assets of a plant set up this session (a preset applied, a component
        # added) were never drawn until the file was reopened. Re-setting the
        # node group rebuilds them.
        mod.node_group = mod.node_group
