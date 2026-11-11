"""Wind -- a bone-driven, additive sway layer on the baked rig.

The target (Epic Nanite Foliage) drives wind through a
bone hierarchy, so wind here rotates the baked rig's *bones* on top of the growth
pose -- it is NOT Geometry Nodes / vertex-shader wind. This module is the
**field + per-bone core**: an animated 3D noise field (a scalar gust strength +
a divergence-free curl swirl), and the per-bone flexibility derived from the stem
radius. It is deliberately kept to pure functions of (position, time, params) so
it can be sampled identically by three consumers -- the live pose handler, the
Bake-Wind step, and the debug views (plane + volume) -- and so the debug views
show *exactly* what moves the bones.

Seamless loop: the time term traces a closed circle (period = `loop_length`), so
the field at `t == loop_length` equals the field at `t == 0`. That makes the
mature-idle clip loop with no hitch; the throughout-growth clip is one-shot and
simply does not care where the loop falls.

Stateless by design (grill Q2): every value is a fresh function of (p, t), so
scrubbing to any frame reproduces the same field -- no integration, no history.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector
from mathutils import noise as _noise

_TWO_PI = 2.0 * math.pi
_MIN_WIDTH = 1e-4          # m; guard the 1/width**4 stiffness against zero-radius
_FLEX_CAP = 24.0           # cap the thin-tip flexibility (lower = calmer tips, less whip)
_UP = Vector((0.0, 0.0, 1.0))
_WIND_GAIN = 0.05       # radians of bend per (flex * perpendicular gust), per _GAIN_LENGTH
_MAX_BONE_ANGLE = 0.15  # cap on that bend, per _GAIN_LENGTH; the chain accumulates to the tips
_GAIN_LENGTH = 0.01     # m: the stem length the two above are per (the old per-node bone) --
#                         so the bend follows the stem's length, never how many bones it has
#                         (the fallback for a rig without `pgs_flex`)
_WIND_LOAD = 1.0 / 6.0  # the wind's push at Strength 1, as a fraction of the plant's own
#                         weight pushed sideways (`rig_layout.sideways_turn`, 90 deg at most
#                         an axis): the limpest stem leans ~15 deg on an average gust
_SWIRL_WEIGHT = 0.18    # turbulence -> eddy tilt (lower = steadier lean, less direction snap)

# -- Leaf wind (the vertex-color WPO layer, grill 2026-09-10) ------------------
# The bones carry the coarse structural sway (and bake/export). The fine leaf
# flutter is a Geometry-Nodes World-Position-Offset on the SKIN mesh, layered
# AFTER the Armature modifier -- exactly how Unreal's Nanite-foliage wind reads
# these same vertex-color roles at runtime, so this is a faithful preview, not a
# second animation to bake. Reads point attributes the bake already carries:
#   vcol_r         -> R: base->tip amplitude mask (0 on stems/flowers, 1 at tips)
#   organ_id       -> G: per-leaf phase (hashed) so leaves desync
#   instance_random-> B: per-plant phase (constant per object, differs per plant)
_LEAF_WIND_GROUP = "PGS_LeafWind"
_LEAF_WIND_MOD = "PGS Leaf Wind"
_LEAF_WIND_VERSION = 2   # v2: Red read above the stems' floor
_LEAF_WIND_VERSION_KEY = "pgs_leafwind_version"
_LEAF_BASE_AMP = 0.03        # m of leaf-tip sway at strength 1, vcol_r 1
_LEAF_FLUTTER_HZ = 1.2       # base flutter frequency (scaled by gust_speed)
_LEAF_TRAVEL_FREQ = 3.0      # traveling-wave spatial frequency across the plant
_LEAF_VERT_FRAC = 0.4        # vertical flutter as a fraction of the horizontal sway


@dataclass
class WindParams:
    """The wind control surface. `direction` is a world vector (built from the
    0-360 heading in `params_from`). contrast / octaves / along_stretch /
    normalize are the temporary noise-tuning surface (baked into constants once
    the look is settled)."""

    strength: float = 1.0         # base wind magnitude
    direction: tuple = (1.0, 0.0, 0.0)   # world-space mean wind heading (normalized on use)
    stiffness: float = 1.0        # global responsiveness trim (higher = stiffer = less sway)
    gust_scale: float = 2.5       # gust spatial frequency (bigger = smaller, tighter gusts)
    gust_speed: float = 1.0       # how fast the gust field drifts/evolves
    turbulence: float = 1.5       # swirl amount (curl-noise weight on the heading)
    loop_length: float = 10.0     # seconds for one seamless loop of the field
    seed_offset: tuple = (0.0, 0.0, 0.0)   # large per-seed offset -> a wholly different pattern
    # Noise shape -- tuned on the debug map, then baked (was the temp tuning surface):
    contrast: float = 1.75        # >1 spreads toward extremes (more red/blue, less green)
    octaves: int = 1              # fBm octaves -> edge detail
    along_stretch: float = 0.3    # <1 elongates gusts along the wind (streaks)
    normalize: bool = False       # divide fBm by its amplitude sum; off widens the range


def _dir(params: WindParams) -> Vector:
    d = Vector(params.direction)
    return d.normalized() if d.length > 1e-9 else Vector((1.0, 0.0, 0.0))


def _seed_offset(seed_int) -> tuple:
    """A large, well-separated sample offset per integer seed, so different seeds
    land in far-apart (uncorrelated) regions of the noise field -- a wholly
    different pattern, not a small slide of the same one."""
    r = random.Random(int(seed_int))
    return (r.uniform(-1.0e4, 1.0e4), r.uniform(-1.0e4, 1.0e4), r.uniform(-1.0e4, 1.0e4))


def _loop_offset(t: float, params: WindParams) -> Vector:
    """A periodic-in-t sample offset: trace a circle so the field returns to
    itself at t == loop_length (seamless). The circle radius is chosen so the
    apparent drift speed tracks `gust_speed`."""
    loop = params.loop_length if params.loop_length > 1e-6 else 1.0
    theta = _TWO_PI * (t / loop)
    r = params.gust_speed * loop / _TWO_PI
    return Vector((math.cos(theta) * r, math.sin(theta) * r, 0.0))


def _fbm(coord: Vector, octaves: int, normalize: bool) -> float:
    """Fractal Brownian motion: summed Perlin octaves for edge detail. Dividing by
    the amplitude sum (`normalize`) keeps it ~[-1, 1] but bunches values near 0;
    OFF widens the spread toward the extremes."""
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(int(octaves)):
        total += amp * _noise.noise(coord * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm if (normalize and norm) else total


def gust(p: Vector, t: float, params: WindParams) -> float:
    """Scalar gust strength in [0, 1], spatial-mean ~0.5 ('neutral', green on the
    jet ramp). Directional: features are elongated along the wind and PAN downwind
    (a crossfade of two phase-shifted copies keeps it seamless over `loop_length`);
    fBm adds edge detail; `contrast` spreads it toward the strong/weak extremes."""
    d = _dir(params)
    loop = params.loop_length if params.loop_length > 1e-6 else 1.0
    c = p * params.gust_scale
    along = c.dot(d)
    c = c - d * (along * (1.0 - params.along_stretch))  # elongate features along the wind
    off_seed = Vector(params.seed_offset)
    tt = t % loop
    phase = tt / loop
    # Shift the SAMPLE coordinate in -d so the visible features pan toward +d
    # (downwind), matching the bone lean and the heading slider. Shifting it in
    # +d makes the pattern appear to travel upwind -- that was the 180-deg bug.
    fwd = d * -(params.gust_speed * tt)                 # features pan downwind (+d)
    back = d * -(params.gust_speed * (tt - loop))       # a full loop apart -> seamless crossfade
    n = ((1.0 - phase) * _fbm(c + fwd + off_seed, params.octaves, params.normalize)
         + phase * _fbm(c + back + off_seed, params.octaves, params.normalize))
    n = max(-1.0, min(1.0, n * params.contrast))        # spread toward strong/weak
    return min(1.0, max(0.0, 0.5 + 0.5 * n))


def _potential(p: Vector, t: float, params: WindParams) -> Vector:
    # A vector potential for the curl; a coarser scale than the gust so the swirl
    # is a larger-scale rotation. Offset in the seed so it decorrelates from gust.
    coord = (p * params.gust_scale * 0.5 + _loop_offset(t, params)
             + Vector(params.seed_offset) + Vector((0.0, 0.0, 131.0)))
    return _noise.noise_vector(coord)


def swirl(p: Vector, t: float, params: WindParams) -> Vector:
    """A divergence-free swirl vector = curl of a noise vector-potential. This is
    what gives gusts their rotating/eddying motion (grill Q1). Roughly unit-scale;
    the caller weights it by `turbulence`."""
    e = 0.1 / max(params.gust_scale, 1e-6)        # finite-difference step, world units
    ex, ey, ez = Vector((e, 0, 0)), Vector((0, e, 0)), Vector((0, 0, e))
    fy_z = _potential(p + ez, t, params).y - _potential(p - ez, t, params).y
    fz_y = _potential(p + ey, t, params).z - _potential(p - ey, t, params).z
    fx_z = _potential(p + ez, t, params).x - _potential(p - ez, t, params).x
    fz_x = _potential(p + ex, t, params).z - _potential(p - ex, t, params).z
    fx_y = _potential(p + ey, t, params).x - _potential(p - ey, t, params).x
    fy_x = _potential(p + ex, t, params).y - _potential(p - ex, t, params).y
    inv = 1.0 / (2.0 * e)
    return Vector(((fz_y - fy_z) * inv, (fx_z - fz_x) * inv, (fy_x - fx_y) * inv))


def local_wind(p: Vector, t: float, params: WindParams) -> Vector:
    """The wind vector at world point `p`, time `t`: the base **direction** tilted
    by a BOUNDED curl-noise eddy (turbulence) -- so the plant clearly sways along
    the wind while gusts wander it a little -- and gusted in magnitude by the
    scalar field (~0.5x lull to ~1.5x gust). The eddy is capped below the base's
    unit length, so the base wind direction always dominates (earlier it was added
    unbounded and swamped the direction, giving near-random sway)."""
    d = _dir(params)
    sw = swirl(p, t, params)
    if sw.length > 1e-9:
        d = (d + sw.normalized() * min(params.turbulence * _SWIRL_WEIGHT, 0.7)).normalized()
    return d * (params.strength * (0.5 + gust(p, t, params)))


def bone_flex(width: float, ref_width: float, params: WindParams) -> float:
    """Per-bone flexibility from the stem radius, using the same 1/radius**4 beam
    relation the gravity bend uses (`physics`), so a plant that droops correctly
    also flexes correctly under wind -- one stiffness model. Normalized against
    `ref_width` (the axis/trunk base radius) so the thick base is ~1 and thin tips
    are larger (up to `_FLEX_CAP`); the global `stiffness` trims it (higher
    stiffness = less flex)."""
    w = max(width, _MIN_WIDTH)
    ref = max(ref_width, _MIN_WIDTH)
    raw = (ref / w) ** 4
    return min(_FLEX_CAP, raw) / max(params.stiffness, 1e-6)


def sway_axis(bone_dir: Vector, wind: Vector) -> Vector:
    """The rotation axis a bone bends about under `wind`: perpendicular to both the
    bone and the wind, so the bone leans downwind. Degenerates gracefully (bone
    parallel to wind) to a stable perpendicular."""
    axis = bone_dir.cross(wind)
    if axis.length < 1e-6:
        axis = bone_dir.cross(_UP)
    if axis.length < 1e-6:
        axis = Vector((1.0, 0.0, 0.0))
    return axis.normalized()


# -- bpy debug views: visualise the field (plane now; volume is the next slice) --

def params_from(s) -> WindParams:
    """Build a WindParams from a plant's PropertyGroup. The 0-360 heading becomes a
    horizontal world vector. The noise-shape fields (contrast / octaves /
    along_stretch / normalize) are baked WindParams defaults now, not panel props."""
    rad = math.radians(s.wind_direction_deg)
    return WindParams(
        strength=s.wind_strength,
        direction=(math.cos(rad), math.sin(rad), 0.0),
        stiffness=s.wind_stiffness, gust_scale=s.wind_gust_scale,
        gust_speed=s.wind_gust_speed, turbulence=s.wind_turbulence,
        loop_length=s.wind_loop_length, seed_offset=_seed_offset(s.wind_seed))


_JET = ((0.00, (0.0, 0.0, 1.0)),   # blue   - lull
        (0.25, (0.0, 1.0, 1.0)),   # cyan
        (0.50, (0.0, 1.0, 0.0)),   # green  - neutral
        (0.75, (1.0, 1.0, 0.0)),   # yellow
        (1.00, (1.0, 0.0, 0.0)))   # red    - gust  (yellow->red passes through orange)


def ramp(v: float):
    """The debug palette: a jet/turbo ramp -- blue (lull) through cyan, green
    (neutral), yellow/orange, to red (gust). Keeps the blue/green/red anchors with
    smooth gradation, matching the reference windmap."""
    v = min(1.0, max(0.0, v))
    for i in range(len(_JET) - 1):
        v0, c0 = _JET[i]
        v1, c1 = _JET[i + 1]
        if v <= v1:
            u = (v - v0) / (v1 - v0) if v1 > v0 else 0.0
            return (c0[0] + (c1[0] - c0[0]) * u,
                    c0[1] + (c1[1] - c0[1]) * u,
                    c0[2] + (c1[2] - c0[2]) * u, 1.0)
    return (*_JET[-1][1], 1.0)


def _debug_plane_name(obj):
    return obj.name + "_WindDebugPlane"


def _remove(o):
    if o is None:
        return
    data = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)


def _debug_material():
    """A shared emission material that shows the point 'Color' attribute (view the
    plane in Material Preview / Rendered shading)."""
    mat = bpy.data.materials.get("PGS_WindDebug")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("PGS_WindDebug")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Color"
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    return mat


def build_debug_plane(obj, res: int):
    """A res x res grid, world-axis-aligned, sized to the plant's footprint, with a
    point 'Color' attribute the handler fills from the gust field each frame."""
    res = max(2, int(res))
    size = max(max(obj.dimensions.x, obj.dimensions.y) * 3.0, 5.0)   # a ground-spanning map
    half = size * 0.5
    xs = np.linspace(-half, half, res)
    gx, gy = np.meshgrid(xs, xs)
    verts = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(res * res)))
    faces = [(j * res + i, j * res + i + 1, (j + 1) * res + i + 1, (j + 1) * res + i)
             for j in range(res - 1) for i in range(res - 1)]
    me = bpy.data.meshes.new(_debug_plane_name(obj))
    me.from_pydata(verts.tolist(), [], faces)
    me.update()
    me.attributes.new("Color", 'FLOAT_COLOR', 'POINT')
    plane = bpy.data.objects.new(_debug_plane_name(obj), me)
    plane.data.materials.append(_debug_material())
    plane.hide_render = True
    coll = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    coll.objects.link(plane)
    return plane


def update_debug_plane(obj, plane, t: float, params: WindParams):
    plane.location = obj.matrix_world.translation      # follow the plant, world-aligned
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.scale = (1.0, 1.0, 1.0)
    me = plane.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    base = plane.location
    cols = np.empty(n * 4)
    for i in range(n):
        wp = Vector((base.x + co[i, 0], base.y + co[i, 1], base.z + co[i, 2]))
        cols[i * 4:i * 4 + 4] = ramp(gust(wp, t, params))
    me.attributes["Color"].data.foreach_set("color", cols)
    me.update_tag()


def _vol_res(res: int) -> int:
    """The volume's per-axis resolution (capped -- it costs res**3 cubes)."""
    return max(2, min(int(res) // 3, 16))


def _debug_volume_name(obj):
    return obj.name + "_WindDebugVolume"


_CUBE = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
_CUBE_FACES = ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
               (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))


def _volume_material():
    """Emission tinted by the 'Color' attribute, faded by its alpha, so the gust
    structure reads in 3D. View in Material Preview / Rendered."""
    mat = bpy.data.materials.get("PGS_WindDebugVol")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("PGS_WindDebugVol")
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = 'BLEND'
        except (TypeError, AttributeError):
            pass
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Color"
    emit = nt.nodes.new("ShaderNodeEmission")
    trans = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(attr.outputs["Alpha"], mix.inputs[0])
    nt.links.new(trans.outputs[0], mix.inputs[1])
    nt.links.new(emit.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def build_debug_volume(obj, res: int):
    """A grid of small cubes over the plant's bounding box, each coloured by the
    gust field with alpha fading toward neutral -- a faithful, cheap 'volume'."""
    vres = _vol_res(res)
    dim = obj.dimensions
    size = Vector((max(dim.x, 0.5), max(dim.y, 0.5), max(dim.z, 0.5))) * 1.3
    cell = min(size.x, size.y, size.z) / vres
    xs = np.linspace(-size.x / 2, size.x / 2, vres)
    ys = np.linspace(-size.y / 2, size.y / 2, vres)
    zs = np.linspace(0.0, size.z, vres)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    verts = (centers[:, None, :] + _CUBE[None, :, :] * (cell * 0.35)).reshape(-1, 3)
    faces = [tuple(k * 8 + i for i in f)
             for k in range(centers.shape[0]) for f in _CUBE_FACES]
    me = bpy.data.meshes.new(_debug_volume_name(obj))
    me.from_pydata(verts.tolist(), [], faces)
    me.update()
    me.attributes.new("Color", 'FLOAT_COLOR', 'POINT')
    vol = bpy.data.objects.new(_debug_volume_name(obj), me)
    vol.data.materials.append(_volume_material())
    vol.hide_render = True
    coll = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    coll.objects.link(vol)
    return vol


def update_debug_volume(obj, vol, t: float, params: WindParams):
    vol.location = obj.matrix_world.translation
    vol.rotation_euler = (0.0, 0.0, 0.0)
    vol.scale = (1.0, 1.0, 1.0)
    me = vol.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    centers = co.reshape(-1, 8, 3).mean(axis=1)        # local cube centres
    base = vol.location
    cols = np.empty((n, 4))
    for k in range(centers.shape[0]):
        wp = Vector((base.x + centers[k, 0], base.y + centers[k, 1], base.z + centers[k, 2]))
        g = gust(wp, t, params)
        r, gr, b, _a = ramp(g)
        alpha = 0.12 + 0.88 * abs(2.0 * (g - 0.5))     # neutral faint, gust/lull opaque
        cols[k * 8:(k + 1) * 8] = (r, gr, b, alpha)
    me.attributes["Color"].data.foreach_set("color", cols.ravel())
    me.update_tag()


def _scene_time(scene) -> float:
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    return scene.frame_current / max(fps, 1e-6)


def refresh_debug(obj):
    """Update `obj`'s debug object(s) at the current frame. No rig needed -- this
    visualises the field itself. A no-op when debug is Off."""
    s = getattr(obj, "plantgen", None)
    if s is None or s.wind_debug == 'OFF':
        return
    t = _scene_time(bpy.context.scene)
    params = params_from(s)
    plane = bpy.data.objects.get(_debug_plane_name(obj))
    if plane is not None:
        update_debug_plane(obj, plane, t, params)
    vol = bpy.data.objects.get(_debug_volume_name(obj))
    if vol is not None:
        update_debug_volume(obj, vol, t, params)


def ensure_debug(obj):
    """Create / rebuild / remove the debug object(s) to match `wind_debug` + res."""
    s = obj.plantgen
    plane = bpy.data.objects.get(_debug_plane_name(obj))
    if s.wind_debug in ('PLANE', 'BOTH'):
        if plane is None or len(plane.data.vertices) != int(s.wind_debug_res) ** 2:
            _remove(plane)
            build_debug_plane(obj, s.wind_debug_res)
    else:
        _remove(plane)
    vol = bpy.data.objects.get(_debug_volume_name(obj))
    if s.wind_debug in ('VOLUME', 'BOTH'):
        if vol is None or len(vol.data.vertices) != _vol_res(s.wind_debug_res) ** 3 * 8:
            _remove(vol)
            build_debug_volume(obj, s.wind_debug_res)
    else:
        _remove(vol)
    refresh_debug(obj)


def _lw_named(nodes, name, data_type, x, y):
    n = nodes.new("GeometryNodeInputNamedAttribute")
    n.data_type = data_type
    n.inputs["Name"].default_value = name
    n.location = (x, y)
    return n.outputs["Attribute"]


def _lw_socket_id(ng, name):
    return next(it.identifier for it in ng.interface.items_tree
                if it.item_type == 'SOCKET' and it.in_out == 'INPUT' and it.name == name)


def _lw_set_input(mod, ng, name, value):
    getattr(mod.properties.inputs, _lw_socket_id(ng, name)).value = value


_TAU = 6.283185307179586


def build_leaf_wind_group():
    """A Geometry-Nodes World-Position-Offset group: each leaf vertex oscillates
    around its (already skinned) rest position by `vcol_r * strength` in the wind
    direction, with a per-leaf phase from `organ_id`, a per-plant phase from
    `instance_random`, and a traveling-wave term across the plant. Zero-mean, so
    it flutters rather than drifts -- the coarse downwind lean is the bones' job.
    Red is read above the stems' floor: stems stay still, an organ is still
    where it attaches and freest at its tip (a flower's Red peaks at a
    quarter, so it flutters about a sixth as far as a leaf)."""
    ng = bpy.data.node_groups.get(_LEAF_WIND_GROUP)
    if ng is not None and ng.get(_LEAF_WIND_VERSION_KEY) == _LEAF_WIND_VERSION:
        return ng
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(_LEAF_WIND_GROUP, "GeometryNodeTree")
    it = ng.interface
    it.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    wd = it.new_socket("Wind Dir", in_out='INPUT', socket_type='NodeSocketVector')
    wd.default_value = (1.0, 0.0, 0.0)
    it.new_socket("Strength", in_out='INPUT', socket_type='NodeSocketFloat').default_value = 1.0
    it.new_socket("Speed", in_out='INPUT', socket_type='NodeSocketFloat').default_value = 1.0
    it.new_socket("Turbulence", in_out='INPUT', socket_type='NodeSocketFloat').default_value = 1.5
    it.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = ng.nodes, ng.links

    def wire(node, idx, val):
        # `val` is a socket (linked) or a plain default; None leaves the input.
        if isinstance(val, bpy.types.NodeSocket):
            links.new(val, node.inputs[idx])
        elif val is not None:
            node.inputs[idx].default_value = val

    def mth(op, x, y, a=None, b=None):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.location = (x, y)
        wire(n, 0, a)
        wire(n, 1, b)
        return n

    def vmath(op, x, y, a=None, b=None):
        n = nodes.new("ShaderNodeVectorMath")
        n.operation = op
        n.location = (x, y)
        wire(n, 0, a)
        wire(n, 1, b)
        return n

    def new(kind, x, y):
        n = nodes.new(kind)
        n.location = (x, y)
        return n

    gin = new("NodeGroupInput", -1400, 0)
    gout = new("NodeGroupOutput", 1300, 0)
    time = new("GeometryNodeInputSceneTime", -1400, -520)
    pos = new("GeometryNodeInputPosition", -1400, -680)

    r = _lw_named(nodes, "vcol_r", 'FLOAT', -1400, 320)
    # Red above the stems' band (`mesh._ORGAN_RED_FLOOR`): 0 where an organ
    # attaches, so it never slides off its stalk; stems (0) stay still.
    from . import mesh as _mesh
    floor = _mesh._ORGAN_RED_FLOOR
    above = mth('SUBTRACT', -1250, 460, r, floor)
    scaled = mth('DIVIDE', -1100, 460, above.outputs[0], 1.0 - floor)
    r = mth('MAXIMUM', -950, 460, scaled.outputs[0], 0.0).outputs[0]
    oid = _lw_named(nodes, "organ_id", 'INT', -1400, 180)
    binst = _lw_named(nodes, "instance_random", 'FLOAT', -1400, 40)

    # per-leaf phase: a stable per-organ random in [0, 2pi) (sockets by name:
    # Random Value's socket order has changed between Blender versions)
    rv = new("FunctionNodeRandomValue", -1150, 180)
    rv.data_type = 'FLOAT'
    rv.inputs["Min"].default_value = 0.0
    rv.inputs["Max"].default_value = _TAU
    links.new(oid, rv.inputs["ID"])
    pl_phase = rv.outputs["Value"]

    pp = mth('MULTIPLY', -1150, 40, binst, _TAU)          # per-plant phase = binst * 2pi
    ndir = vmath('NORMALIZE', -1150, -300, gin.outputs["Wind Dir"])
    dot = vmath('DOT_PRODUCT', -950, -460, pos.outputs["Position"], ndir.outputs["Vector"])
    trav = mth('MULTIPLY', -750, -460, dot.outputs["Value"], _LEAF_TRAVEL_FREQ)

    wf = mth('MULTIPLY', -950, -250, gin.outputs["Speed"], _TAU * _LEAF_FLUTTER_HZ)
    tw = mth('MULTIPLY', -750, -250, time.outputs["Seconds"], wf.outputs[0])

    a1 = mth('ADD', -500, -120, tw.outputs[0], pl_phase)
    a2 = mth('ADD', -320, -120, a1.outputs[0], pp.outputs[0])
    a3 = mth('ADD', -140, -120, a2.outputs[0], trav.outputs[0])
    wave = mth('SINE', 40, -120, a3.outputs[0])

    tw2 = mth('MULTIPLY', -500, -360, tw.outputs[0], 2.3)
    pl17 = mth('MULTIPLY', -500, -520, pl_phase, 1.7)
    a2b = mth('ADD', -300, -440, tw2.outputs[0], pl17.outputs[0])
    wave2 = mth('SINE', -100, -440, a2b.outputs[0])

    amp1 = mth('MULTIPLY', -500, 320, r, gin.outputs["Strength"])
    amp = mth('MULTIPLY', -300, 320, amp1.outputs[0], _LEAF_BASE_AMP)

    hmag = mth('MULTIPLY', 260, 80, amp.outputs[0], wave.outputs[0])
    hvec = vmath('SCALE', 460, 80, ndir.outputs["Vector"])
    links.new(hmag.outputs[0], hvec.inputs["Scale"])

    vmag = mth('MULTIPLY', 260, -160, amp.outputs[0], wave2.outputs[0])
    vmag2 = mth('MULTIPLY', 440, -160, vmag.outputs[0], _LEAF_VERT_FRAC)
    upv = new("FunctionNodeInputVector", 260, -320)
    upv.vector = (0.0, 0.0, 1.0)
    vvec = vmath('SCALE', 620, -220, upv.outputs[0])
    links.new(vmag2.outputs[0], vvec.inputs["Scale"])

    off = vmath('ADD', 820, 0, hvec.outputs["Vector"], vvec.outputs["Vector"])

    setp = new("GeometryNodeSetPosition", 1050, 0)
    links.new(gin.outputs["Geometry"], setp.inputs["Geometry"])
    links.new(off.outputs["Vector"], setp.inputs["Offset"])
    links.new(setp.outputs["Geometry"], gout.inputs["Geometry"])

    ng[_LEAF_WIND_VERSION_KEY] = _LEAF_WIND_VERSION
    return ng


def ensure_leaf_wind(obj):
    """Add / update / gate the leaf-wind GN modifier on `obj`'s skin. Placed after
    the Armature modifier so it flutters the skinned result. Preview-only:
    show_render is off, and it never bakes -- the export carries only the bones."""
    from . import rig
    found = rig.find_rig(obj)
    skin = found[1] if found and len(found) > 1 else None
    if skin is None:
        return
    s = obj.plantgen
    mod = skin.modifiers.get(_LEAF_WIND_MOD)
    if getattr(s, "wind_secondary_enable", False):
        ng = build_leaf_wind_group()
        if mod is None or mod.type != 'NODES':
            if mod is not None:
                skin.modifiers.remove(mod)
            mod = skin.modifiers.new(_LEAF_WIND_MOD, 'NODES')   # appended -> after Armature
        mod.node_group = ng
        rad = math.radians(s.wind_direction_deg)
        world_dir = Vector((math.cos(rad), math.sin(rad), 0.0))
        local_dir = skin.matrix_world.to_3x3().inverted() @ world_dir
        if local_dir.length > 1e-9:
            local_dir.normalize()
        _lw_set_input(mod, ng, "Wind Dir", local_dir)
        _lw_set_input(mod, ng, "Strength", float(s.wind_secondary_strength))
        _lw_set_input(mod, ng, "Speed", float(s.wind_gust_speed))
        _lw_set_input(mod, ng, "Turbulence", float(s.wind_turbulence))
        mod.show_render = False
        mod.show_viewport = True
    elif mod is not None:
        skin.modifiers.remove(mod)


# -- Skin vertex-color debug (grill 2026-09-10) --------------------------------
# Shows the skin's vertex-color channels tinted ON THE MESH, exactly like the
# plant's own Debug View -- same mechanism (a GN Viewer node + the viewport
# viewer-attribute overlay), which is what makes it work in SOLID shading (an
# emission material would only show in Material Preview / Rendered). The plant's
# viewers live in the *static* plant's mesh group, invisible once the plant is
# hidden; these live on the SKIN. One named viewer per channel, activated by
# name via `mesh._find_node_editor` + node.activate_viewer -- the skin's copy of
# `mesh.set_active_viewer`. Enabling also turns the overlay on in every 3D
# viewport so the tint is never silently invisible. Non-destructive; render off.
_SKIN_DEBUG_GROUP = "PGS_SkinVColDebug"
_SKIN_DEBUG_MOD = "PGS Skin VCol Debug"
_SKIN_DEBUG_VERSION = 4
_SKIN_DEBUG_VERSION_KEY = "pgs_skindebug_version"
_SKIN_VIEWER_NAMES = {
    "color": "PGS_SkinVCol_color",   # R,G,B together
    "vcol_r": "PGS_SkinVCol_r",      # base->tip amplitude (grey)
    "vcol_g": "PGS_SkinVCol_g",      # per-leaf random (grey)
    "vcol_b": "PGS_SkinVCol_b",      # per-plant random (grey)
    "vcol_a": "PGS_SkinVCol_a",      # height fraction (grey)
}


def build_skin_vcol_group():
    """One Viewer node per vertex-color channel (greyscale for the isolated ones,
    RGB for combined), reading the same roles the secondary wind does: R vcol_r,
    G per-leaf (organ_id), B instance_random, A height_fraction. Geometry passes
    straight through -- the viewers are taps, so the mesh itself is unchanged."""
    ng = bpy.data.node_groups.get(_SKIN_DEBUG_GROUP)
    if ng is not None and ng.get(_SKIN_DEBUG_VERSION_KEY) == _SKIN_DEBUG_VERSION:
        return ng
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(_SKIN_DEBUG_GROUP, "GeometryNodeTree")
    it = ng.interface
    it.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    it.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = ng.nodes, ng.links
    gin = nodes.new("NodeGroupInput")
    gin.location = (-900, 0)
    gout = nodes.new("NodeGroupOutput")
    gout.location = (900, 0)
    geo = gin.outputs["Geometry"]
    links.new(geo, gout.inputs["Geometry"])   # pass-through; viewers are taps

    r = _lw_named(nodes, "vcol_r", 'FLOAT', -900, 320)
    oid = _lw_named(nodes, "organ_id", 'INT', -900, 180)
    binst = _lw_named(nodes, "instance_random", 'FLOAT', -900, 40)
    hf = _lw_named(nodes, "height_fraction", 'FLOAT', -900, -100)
    rv = nodes.new("FunctionNodeRandomValue")
    rv.data_type = 'FLOAT'
    rv.location = (-680, 180)
    rv.inputs["Min"].default_value = 0.0
    rv.inputs["Max"].default_value = 1.0
    links.new(oid, rv.inputs["ID"])
    g = rv.outputs["Value"]

    def comb(cr, cg, cb, y):
        # A None channel is left at 0, so an isolated channel reads in its own
        # hue (Red -> red, Green -> green, Blue -> blue) rather than greyscale.
        n = nodes.new("FunctionNodeCombineColor")
        n.mode = 'RGB'
        n.location = (-400, y)
        for name, src in (("Red", cr), ("Green", cg), ("Blue", cb)):
            if src is None:
                n.inputs[name].default_value = 0.0
            else:
                links.new(src, n.inputs[name])
        return n.outputs["Color"]

    colors = {
        "color": comb(r, g, binst, 320),        # combined R/G/B
        "vcol_r": comb(r, None, None, 160),     # red channel, shown red
        "vcol_g": comb(None, g, None, 0),       # green channel, shown green
        "vcol_b": comb(None, None, binst, -160),  # blue channel, shown blue
        "vcol_a": comb(hf, hf, hf, -320),       # height has no colour -> greyscale
    }
    y = 320
    for key, col in colors.items():
        v = nodes.new("GeometryNodeViewer")
        v.name = v.label = _SKIN_VIEWER_NAMES[key]
        v.viewer_items.new('GEOMETRY', "Geometry")   # 5.2: a fresh viewer has no sockets
        v.viewer_items.new('RGBA', "Debug")
        links.new(geo, v.inputs["Geometry"])
        links.new(col, v.inputs["Debug"])
        v.location = (300, y)
        y -= 180
    ng[_SKIN_DEBUG_VERSION_KEY] = _SKIN_DEBUG_VERSION
    return ng


def _show_viewer_overlay():
    """Turn the viewer-attribute overlay on in every 3D viewport, so an active
    viewer actually paints. The plant debug relies on the user's own overlay
    setting; here it is forced on, because a skin debug that silently shows
    nothing is exactly the failure being fixed."""
    for scr in bpy.data.screens:
        for area in scr.areas:
            if area.type != 'VIEW_3D':
                continue
            for sp in area.spaces:
                if sp.type == 'VIEW_3D':
                    sp.overlay.show_viewer_attribute = True


def _activate_skin_viewer(skin, viewer_name):
    """Make one skin-debug viewer the live viewport overlay -- the skin's copy of
    `mesh.set_active_viewer`.

    Activation genuinely needs a node-editor context, and requiring one to
    already show this (possibly just-rebuilt) group is what made the tint fail
    silently. So if none does, borrow ANY Geometry Nodes editor, point it at the
    group just long enough to run `node.activate_viewer`, then restore it -- the
    3D-viewport overlay persists after the editor moves on (verified live), so
    the user's node editor is left exactly as it was and the mesh stays tinted."""
    from .mesh import _find_node_editor
    mod = skin.modifiers.get(_SKIN_DEBUG_MOD)
    if mod is None or mod.node_group is None:
        return "no_modifier"
    ng = mod.node_group
    viewer = ng.nodes.get(viewer_name)
    if viewer is None:
        return "no_viewer_node"
    skin.modifiers.active = mod
    for n in ng.nodes:
        n.select = False
    viewer.select = True
    ng.nodes.active = viewer
    _show_viewer_overlay()

    win, area, region, space = _find_node_editor(ng)
    if win is not None:
        with bpy.context.temp_override(window=win, area=area, region=region, space_data=space):
            bpy.ops.node.activate_viewer()
        return "activated"

    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type != 'NODE_EDITOR':
                continue
            sp = next((s for s in a.spaces
                       if s.type == 'NODE_EDITOR' and s.tree_type == 'GeometryNodeTree'), None)
            if sp is None:
                continue
            region = next((r for r in a.regions if r.type == 'WINDOW'), None)
            saved_tree, saved_pin = sp.node_tree, sp.pin
            try:
                sp.node_tree = ng
                with bpy.context.temp_override(window=w, area=a, region=region, space_data=sp):
                    bpy.ops.node.activate_viewer()
            finally:
                sp.node_tree = saved_tree
                sp.pin = saved_pin
            return "activated"
    return "no_editor_open"


def ensure_skin_debug(obj):
    """Add / update / remove the skin vertex-color debug modifier to match
    `wind_skin_debug`, and activate the chosen channel's viewer so the mesh tints
    in the viewport (Solid included). Returns a short status for the operator."""
    from . import rig
    found = rig.find_rig(obj)
    skin = found[1] if found and len(found) > 1 else None
    if skin is None:
        return "no_skin"
    mode = getattr(obj.plantgen, "wind_skin_debug", 'off')
    mod = skin.modifiers.get(_SKIN_DEBUG_MOD)
    if mode == 'off':
        if mod is not None:
            skin.modifiers.remove(mod)
        return "off"
    ng = build_skin_vcol_group()
    if mod is None or mod.type != 'NODES':
        if mod is not None:
            skin.modifiers.remove(mod)
        mod = skin.modifiers.new(_SKIN_DEBUG_MOD, 'NODES')
    mod.node_group = ng
    mod.show_render = False
    return _activate_skin_viewer(skin, _SKIN_VIEWER_NAMES[mode])


def _ref_width(arm) -> float:
    """The reference (thickest) node-bone radius, for normalizing flexibility."""
    ws = [b.get("pgs_width", 0.0) for b in arm.data.bones if "_o" not in b.name]
    ws = [w for w in ws if w and w > 0.0]
    return max(ws) if ws else 1.0


def apply_wind_pose(arm, t, params, from_rest: bool = False):
    """Add the wind bend ON TOP of the growth pose. Each NODE bone rotates about the
    axis perpendicular to (bone, wind) by an angle scaled by its flexibility (from
    the stored stem radius) and the local gust. Applied in the bone's local basis so
    it composes down the chain -- tips accumulate the stem's whole bend -- and organ
    bones inherit it (inherit_scale='NONE' still inherits rotation). Order-free: it
    reads each bone's growth pose and post-multiplies its local rotation ONCE, so it
    must run on a *fresh* growth pose (the frame handler resets it each frame).

    The bend is per metre of stem (`_GAIN_LENGTH`), so a rig with fewer,
    longer bones bends exactly as far. `from_rest`: a static rig (no growth
    action) works from its rest matrices -- the evaluated pose is still the
    previous frame's, and reading it made one frame differ visit to visit."""
    ref = _ref_width(arm)
    # ONE coherent wind for the whole plant, sampled at its base: at a fine gust
    # scale, per-bone sampling made adjacent bones pull opposite ways and cancel
    # (weak, non-directional jitter). A single gust vector makes the plant lean
    # together in the wind direction; the bend still accumulates down the chain so
    # tips whip, and each bone bends about its own perpendicular to that wind.
    wind_arm = arm.matrix_world.to_3x3().inverted() @ local_wind(
        arm.matrix_world.translation, t, params)
    if wind_arm.length < 1e-9:
        return
    yb = Vector((0.0, 1.0, 0.0))
    for pb in arm.pose.bones:
        if "_o" in pb.name or pb.bone.get("pgs_organ") or pb.parent is None:
            continue                              # organs ride their node; the root stays
        pm3 = (pb.bone.matrix_local if from_rest else pb.matrix).to_3x3()
        dir_arm = (pm3 @ yb).normalized()
        wperp = wind_arm - dir_arm * wind_arm.dot(dir_arm)
        mag = wperp.length
        if mag < 1e-6:
            continue
        give = pb.bone.get("pgs_flex")
        if give is not None:
            # The rig's own beam model: how far this bone's internodes turn
            # under the plant's weight pushed sideways, scaled by the wind.
            angle = math.radians(float(give)) * _WIND_LOAD * mag / max(params.stiffness, 1e-6)
        else:
            w = pb.bone.get("pgs_width", 0.0)
            flex = (bone_flex(w, ref, params) if w and w > 0.0
                    else 1.0 / max(params.stiffness, 1e-6))
            per = max(pb.bone.length, 1e-6) / _GAIN_LENGTH
            cap = _MAX_BONE_ANGLE * per
            angle = max(-cap, min(cap, flex * _WIND_GAIN * mag * per))
        axis = dir_arm.cross(wind_arm)
        if axis.length < 1e-6:
            continue
        rot = Matrix.Rotation(angle, 3, axis.normalized())
        r_local = (pm3.inverted() @ rot @ pm3).to_quaternion()
        pb.rotation_quaternion = pb.rotation_quaternion @ r_local


def rest_pose(arm) -> None:
    """Every bone back to rest -- the base Bone Wind adds to on a rig with no
    baked action (it would otherwise pile up frame on frame)."""
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


@persistent
def _drive_wind(scene, _depsgraph=None):
    """frame_change_post: apply the live wind sway to each enabled rig (on top of the
    frame's fresh growth pose) and keep debug views current. Cheap -- only touches
    plants with wind enabled or a debug view on."""
    from . import bridge, rig
    t = _scene_time(scene)
    for obj in scene.objects:
        s = getattr(obj, "plantgen", None)
        if s is None or not bridge.is_plant(obj):
            continue
        if getattr(s, "wind_enable", False):
            found = rig.find_rig(obj)
            if found is not None and not found[0].hide_viewport:
                ad = found[0].animation_data
                static = ad is None or ad.action is None
                if static:
                    rest_pose(found[0])           # a static rig: wind on a fresh rest pose
                apply_wind_pose(found[0], t, params_from(s), from_rest=static)
        if getattr(s, "wind_debug", 'OFF') != 'OFF':
            refresh_debug(obj)
