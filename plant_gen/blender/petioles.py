"""Leaf stalks and leaf collision: each drawn leaf's petiole (shape, maps,
growth with the leaf), the collision resolve that turns or prunes leaves,
and the reference resolve a timelapse reuses (`ensure_leaf_reference`).
Split out of `bridge.py` (2026-10-13)."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import bpy
import numpy as np

from ..growth import masks
from ..growth.development import build
from ..growth.model import OrganKind
from . import bridge, curves, instancing, mapping
from . import environment as env_bridge

_PETIOLE_SEGMENTS = 8
_DOWN = np.array([0.0, 0.0, -1.0])


def _turn(v, axis, theta: float):
    """`v` rotated about the unit `axis` by `theta` radians (Rodrigues)."""
    c, s_ = np.cos(theta), np.sin(theta)
    return v * c + np.cross(axis, v) * s_ + axis * float(axis @ v) * (1.0 - c)


def _petiole_shape(c, oid: int, cseed: int):
    """One petiole's own draws -- length factor, start tilt (radians) and the
    tilt's direction round the leaf (a cone), bend and droop (radians) -- each
    seeded per leaf on the component's seed, its own salt per field."""
    def signed(salt):
        key = np.array([int(oid)], dtype=np.int64)
        word = (cseed * 2654435761 + salt) & instancing._U64
        return float(masks._hash01(key, word)[0]) * 2.0 - 1.0
    length = (float(mapping.cval(c, "petiole_length"))
              * (1.0 + float(c.petiole_length_variation) * signed(11)))
    tilt = np.radians(float(c.petiole_angle_variation)) * abs(signed(13))
    around = np.pi * signed(17)
    bend = (float(mapping.cval(c, "petiole_bend"))
            + np.radians(float(c.petiole_bend_variation)) * signed(19))
    droop = max(0.0, float(mapping.cval(c, "petiole_droop"))
                + np.radians(float(c.petiole_droop_variation)) * signed(23))
    return max(length, 0.0), tilt, around, bend, droop


_PETIOLE_TARGETS = ("PETIOLE_LENGTH", "PETIOLE_WIDTH", "PETIOLE_BEND", "PETIOLE_DROOP")


def _petiole_maps(obj, comps, owner, organ_attrs) -> dict:
    """{point: (length x, width x, bend +, droop + radians)} from each leaf /
    bract component's petiole instance maps, over the points it owns. Empty
    without such a map: every petiole is exactly as before."""
    out = {}
    chan = None
    ids = None
    for ci, c in enumerate(comps):
        if c.type not in ("leaf", "bract") or not instancing._has_imap(c, _PETIOLE_TARGETS):
            continue
        take = np.nonzero(owner == ci)[0]
        if not len(take):
            continue
        if chan is None:
            chan = instancing._channels_from(organ_attrs)
            ids = np.asarray(organ_attrs.get("vkey") or organ_attrs["organ_id"], dtype=np.int64)
        sub = {k: np.asarray(v)[take] for k, v in chan.items()}
        vals = [instancing.instance_map_values(obj, c, target, sub, ids[take])
                for target in _PETIOLE_TARGETS]
        neutral = (1.0, 1.0, 0.0, 0.0)
        for j, i in enumerate(take):
            out[int(i)] = tuple(n if v is None else float(v[j])
                                for v, n in zip(vals, neutral, strict=True))
    return out


def _petiole_path(start, direction, normal, length, tilt, around, bend, droop):
    """The petiole as points from the stem out, and the leaf's frame at its end.

    It leaves along the leaf's own facing, tilted `tilt` toward a direction
    `around` it (Angle Variation: a cone, so petioles splay up, down and to the
    sides alike); bends `bend` in total toward the leaf's face (about its
    width: positive arches it up when the face is up); and sags `droop` in
    total toward the ground, never past straight down. Both turns build up
    along it, so it curves smoothly. The leaf is turned with the stalk's end,
    so it carries on the way the petiole points."""
    width = np.cross(direction, normal)
    wn = float(np.linalg.norm(width))
    width = width / wn if wn > 1e-9 else np.array([1.0, 0.0, 0.0])
    d0, n0 = direction, normal
    if tilt > 0.0:
        side = _turn(width, direction, around)                  # a direction round it
        d0, n0 = _turn(direction, side, tilt), _turn(normal, side, tilt)
    width0 = np.cross(d0, n0)
    width0 = width0 / max(float(np.linalg.norm(width0)), 1e-9)

    def tangent(f):
        t = _turn(d0, width0, bend * f)
        if droop > 0.0:
            axis = np.cross(t, _DOWN)
            na = float(np.linalg.norm(axis))
            if na > 1e-9:
                to_down = float(np.arccos(np.clip(t @ _DOWN, -1.0, 1.0)))
                t = _turn(t, axis / na, min(droop * f, to_down))
        return t / max(float(np.linalg.norm(t)), 1e-12)

    step = length / _PETIOLE_SEGMENTS
    points = [np.asarray(start, dtype=np.float64)]
    for k in range(_PETIOLE_SEGMENTS):
        points.append(points[-1] + tangent((k + 0.5) / _PETIOLE_SEGMENTS) * step)
    end = tangent(1.0)
    # the leaf's frame, turned the least way from where it pointed to the end
    axis = np.cross(direction, end)
    na = float(np.linalg.norm(axis))
    if na > 1e-9:
        theta = float(np.arctan2(na, float(direction @ end)))
        leaf_d, leaf_n = _turn(direction, axis / na, theta), _turn(normal, axis / na, theta)
    elif float(direction @ end) < 0.0:
        leaf_d, leaf_n = -direction, normal
    else:
        leaf_d, leaf_n = direction, normal
    return points, leaf_d, leaf_n


LAST_COLLISIONS: dict = {}

_PROXY_CACHE: dict = {}
_SETTLE_CACHE: dict = {}


def _leaf_grown(organ_attrs, i: int) -> float:
    """How far leaf point `i` has expanded (0-1; `unfolded`, 1 when absent)."""
    grown = organ_attrs.get("unfolded")
    return 1.0 if grown is None else min(1.0, max(0.0, float(grown[i])))


def _add_petioles(obj, s, plant, positions, organ_attrs, verts, edges, attrs,
                  hidden_tubes=frozenset()) -> int:
    """A stalk for every leaf / bract a component with a Petiole Length drew
    (grown or placed, not dropped), shaped like a flower's pedicel: Length and
    Width with its Width curve along it (the taper; flat by default), Bend
    toward the leaf's face, Droop toward the ground, and
    variation on its length, start angle (a cone round the leaf's facing),
    bend and droop -- each drawn per leaf on the component's seed
    (`_petiole_shape`, `_petiole_path`); Length and Width grow with the leaf's
    expansion (`unfolded`), full at full size. The leaf moves to the stalk's end and
    turns with it. Appended to the stem skeleton in place as a chain of its
    own -- a tube with no parent, so the mesher caps it rather than welding it
    (a petiole meets the stem at a joint, not a fillet) and the preview sweeps
    it like any stem. Each stalk vertex takes its leaf's channels (height,
    level, the structural ones), so masks and the debug views read it as the
    leaf's.

    Before anything is written, a component with Avoid Collisions turns its
    leaves -- stalk and blade together, about the stalk's foot -- out of the
    stems and the older leaves, and prunes those it cannot clear
    (`_settle_leaves`). Returns the stalks added."""
    lp = instancing.LAST_POINTS.get(obj.name)
    LAST_COLLISIONS.pop(obj.name, None)
    if not verts or lp is None:
        return 0
    owner, drop = np.asarray(lp["owner"]), np.asarray(lp["drop"])
    kind = organ_attrs["kind"]
    if len(owner) != len(positions):
        return 0
    comps = list(s.components)
    seed = int(s.seed)
    mapped = _petiole_maps(obj, comps, owner, organ_attrs)
    # Every leaf / bract a component drew: [index, component, stalk points (or
    # None), the blade's base, direction, normal, changed, width factor].
    drawn = []
    for i, comp_index in enumerate(owner):
        ci = int(comp_index)
        if ci < 0 or ci >= len(comps) or drop[i] >= 0.5 or int(kind[i]) != 0:
            continue
        c = comps[ci]
        if c.type not in ("leaf", "bract") or c.mute:
            continue
        if not c.collide and float(mapping.cval(c, "petiole_length")) <= 0.0 and \
                not any(o.collide for o in comps):
            continue
        direction = np.asarray(organ_attrs["direction"][i], dtype=np.float64)
        normal = np.asarray(organ_attrs["normal"][i], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction = direction / norm
        normal = normal - direction * float(normal @ direction)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
        base = np.asarray(positions[i], dtype=np.float64)
        points, changed = None, False
        if float(mapping.cval(c, "petiole_length")) > 0.0:
            oid = int((organ_attrs.get("vkey") or organ_attrs["organ_id"])[i])
            length, tilt, around, bend, droop = _petiole_shape(c, oid, instancing._cseed(seed, c))
            terms = mapped.get(i)
            if terms is not None:
                # its instance maps: Length x, Bend / Droop + (radians)
                lf, _wf, bo, do = terms
                length = max(length * lf, 0.0)
                bend += bo
                droop = max(0.0, droop + do)
            # It grows with its leaf, like a flower's pedicel with its flower.
            length *= _leaf_grown(organ_attrs, i)
            if length > 0.0:
                points, direction, normal = _petiole_path(base, direction, normal, length,
                                                          tilt, around, bend, droop)
                points = np.asarray(points, dtype=np.float64)
                base, changed = points[-1], True
        drawn.append([i, c, points, base, direction, normal, changed,
                      mapped[i][1] if i in mapped else 1.0])
    if any(row[1].collide for row in drawn):
        _settle_leaves(obj, positions, organ_attrs, verts, edges, attrs, hidden_tubes,
                       drawn, lp)

    n0 = len(verts)
    keys = [k for k, v in attrs.items() if isinstance(v, list) and len(v) == n0]
    ints = {k for k in keys if n0 and isinstance(attrs[k][0], (int, np.integer))
            and not isinstance(attrs[k][0], bool)}
    point_keys = [k for k in keys if k in organ_attrs and len(organ_attrs[k]) == len(positions)]
    ranked = sorted(plant.axes)
    next_tube = max((int(t) for t in attrs["tube"]), default=-1) + 1
    added = 0
    for i, c, points, base, leaf_d, leaf_n, changed, width_f in drawn:
        if changed:
            positions[i] = (float(base[0]), float(base[1]), float(base[2]))
            organ_attrs["direction"][i] = tuple(float(x) for x in leaf_d)
            organ_attrs["normal"][i] = tuple(float(x) for x in leaf_n)
        if points is None:
            continue
        rank = int(organ_attrs["branch_ordinal"][i]) if "branch_ordinal" in organ_attrs else 0
        aid = ranked[min(max(rank, 0), len(ranked) - 1)] if ranked else -1
        axis = plant.axes.get(aid)
        base_w = (float(mapping.cval(c, "petiole_width")) * _leaf_grown(organ_attrs, i)
                  * width_f)
        node = curves.get_component_curve(obj, c.uid, "petiole_width")
        taper = (curves.sample_node(node, np.linspace(0.0, 1.0, _PETIOLE_SEGMENTS + 1))
                 if node is not None else np.ones(_PETIOLE_SEGMENTS + 1))
        fixed = {"tube": next_tube, "parent_tube": -1, "axis_id": int(aid), "node_index": -1,
                 "structural_depth": -1, "order": (axis.order + 1) if axis is not None else 1,
                 "flare": 0.0}
        first = len(verts)
        for k, p in enumerate(points):
            verts.append((float(p[0]), float(p[1]), float(p[2])))
            fixed["width"] = base_w * max(float(taper[k]), 0.02)
            for key in keys:
                if key in fixed:
                    value = fixed[key]
                elif key in point_keys:
                    value = organ_attrs[key][i]
                else:
                    value = 0 if key in ints else 0.0
                attrs[key].append(value)
            if k:
                edges.append((first + k - 1, first + k))
        next_tube += 1
        added += 1
    return added


def _leaf_proxy(o):
    """The collision proxy of a leaf asset object (its mesh, decimated), cached."""
    from ..geometry import leaf_collision

    me = o.data if o.type == 'MESH' else None
    if me is None or not len(me.polygons):
        return None
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    key = (o.name, me.name, n, hashlib.blake2b(co.tobytes(), digest_size=8).hexdigest())
    hit = _PROXY_CACHE.get(key)
    if hit is not None:
        return hit
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    shape = leaf_collision.simplify(co.reshape(-1, 3), tri.reshape(-1, 3))
    if len(_PROXY_CACHE) > 256:
        _PROXY_CACHE.clear()
    _PROXY_CACHE[key] = shape
    return shape


_FLOWER_PROXY_CELLS = (5, 5, 6)


def _flower_proxy(o):
    """The collision proxy of a flower asset object: its EVALUATED mesh (a
    FlowerGen flower is built by its modifier), decimated, cached on a digest
    of its vertices. None when the object has no evaluated mesh."""
    from ..geometry import leaf_collision

    try:
        ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
    except (RuntimeError, ReferenceError, AttributeError):
        return None
    try:
        if me is None or not len(me.polygons):
            return None
        n = len(me.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        key = ("flower", o.name, n, hashlib.blake2b(co.tobytes(), digest_size=8).hexdigest())
        hit = _PROXY_CACHE.get(key)
        if hit is not None:
            return hit
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        shape = leaf_collision.simplify(co.reshape(-1, 3), tri.reshape(-1, 3),
                                        cells=_FLOWER_PROXY_CELLS)
        if len(_PROXY_CACHE) > 256:
            _PROXY_CACHE.clear()
        _PROXY_CACHE[key] = shape
        return shape
    finally:
        ev.to_mesh_clear()


def _flower_obstacles(obj, positions, organ_attrs, stems, shapes, shape_of) -> list:
    """Every drawn flower as a fixed obstacle the leaves turn away from: its
    OPEN stage (the last of its component's bloom block) -- the most room it
    takes, so a leaf clears it at any Bloom and does not move as it opens --
    posed as the node graph instances it (+Z along `direction`, `scale`)."""
    from ..geometry import leaf_collision as lc

    master = instancing.master_collection(obj, "flower")
    offset = organ_attrs.get("flower_block_offset")
    top = organ_attrs.get("flower_block_max")
    if master is None or offset is None or top is None:
        return []
    kinds, scales = organ_attrs["kind"], organ_attrs["scale"]
    render_drop = organ_attrs.get("render_drop")
    out = []
    for i in range(len(positions)):
        if int(kinds[i]) != 1 or (render_drop is not None and render_drop[i] >= 0.5):
            continue
        k = int(offset[i]) + int(top[i])
        if not 0 <= k < len(master.objects):
            continue
        o = master.objects[k]
        tag = "flower:" + o.name
        if tag not in shape_of:
            shape = _flower_proxy(o)
            shape_of[tag] = None if shape is None else len(shapes)
            if shape is not None:
                shapes.append(shape)
        if shape_of[tag] is None:
            continue
        direction = np.asarray(organ_attrs["direction"][i], dtype=np.float64)
        if float(np.linalg.norm(direction)) < 1e-9:
            continue
        pivot = np.asarray(positions[i], dtype=np.float64)
        r_near = 0.0
        if len(stems.a):
            dist = lc.points_to_segments(pivot[None], stems.a, stems.b)[0]
            r_near = float(stems.radius[int(np.argmin(dist - stems.radius))])
        out.append(lc.Leaf(
            shape=shape_of[tag], base=pivot, direction=direction,
            normal=np.asarray(organ_attrs["normal"][i], dtype=np.float64),
            scale=float(scales[i]), pivot=pivot, exempt=1.5 * r_near, upright=True))
    return out


def _settle_leaves(obj, positions, organ_attrs, verts, edges, attrs, hidden_tubes,
                   drawn, lp) -> None:
    """Leaf collision (`geometry/leaf_collision.py`): every drawn leaf posed as
    the node graph will draw it (its asset at `instance_index`, `scale`,
    direction / normal), the stems as capsules of their own width (hidden
    levels left out). Leaves of a component with Avoid Collisions turn --
    oldest settling first (`age`), stalk and blade about the stalk's foot --
    and are pruned when no turn clears them; the rest are obstacles, and so
    is every drawn flower (`_flower_obstacles`). No turn takes a leaf below
    the stems' lowest point. Updates
    `drawn` in place (pose, changed) and drops the pruned (`render_drop`,
    `LAST_POINTS`)."""
    from ..geometry import leaf_collision as lc

    index = organ_attrs.get("instance_index")
    if index is None or not edges:
        return
    v = np.asarray(verts, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    width = np.asarray(attrs["width"], dtype=np.float64)
    if hidden_tubes:
        tube = np.asarray(attrs["tube"])
        shown = ~np.isin(tube, list(hidden_tubes))
        e = e[shown[e[:, 0]] & shown[e[:, 1]]]
    stems = lc.Stems(v[e[:, 0]], v[e[:, 1]], np.maximum(width[e[:, 0]], width[e[:, 1]]))
    shapes, shape_of, leaves, rows = [], {}, [], []
    ages = organ_attrs.get("age")
    for row in drawn:
        i, c, points, base, direction, normal, _changed, width_f = row
        master = instancing.master_collection(obj, c.type)
        k = int(index[i])
        if master is None or not 0 <= k < len(master.objects):
            continue
        o = master.objects[k]
        if o.name not in shape_of:
            shape = _leaf_proxy(o)
            shape_of[o.name] = None if shape is None else len(shapes)
            if shape is not None:
                shapes.append(shape)
        if shape_of[o.name] is None:
            continue
        pivot = np.asarray(positions[i], dtype=np.float64)
        # The attachment zone: the stem it grows from, 1.5 x that stem's radius.
        r_near = 0.0
        if len(stems.a):
            dist = lc.points_to_segments(pivot[None], stems.a, stems.b)[0]
            r_near = float(stems.radius[int(np.argmin(dist - stems.radius))])
        leaves.append(lc.Leaf(
            shape=shape_of[o.name], base=base, direction=direction, normal=normal,
            scale=float(organ_attrs["scale"][i]), pivot=pivot, stalk=points,
            stalk_radius=(float(mapping.cval(c, "petiole_width")) * _leaf_grown(organ_attrs, i)
                          * width_f),
            exempt=1.5 * r_near,
            order=float(ages[i]) if ages is not None else 0.0,
            movable=bool(c.collide), max_turn=float(np.radians(c.collide_max_turn)),
            prune=bool(c.collide_prune)))
        rows.append(row)
    if not leaves:
        return
    # The flowers: fixed obstacles (after the leaves, so `rows` still indexes them)
    leaves += _flower_obstacles(obj, positions, organ_attrs, stems, shapes, shape_of)
    digest = hashlib.blake2b(digest_size=16)
    for arr in (stems.a, stems.b, stems.radius):
        digest.update(np.ascontiguousarray(arr).tobytes())
    for shape in shapes:
        digest.update(shape.verts.tobytes())
        digest.update(shape.tris.tobytes())
    for leaf in leaves:
        digest.update(np.array([leaf.shape, leaf.scale, leaf.stalk_radius, leaf.exempt,
                                leaf.order, leaf.movable, leaf.max_turn, leaf.prune,
                                leaf.upright], dtype=np.float64).tobytes())
        for arr in (leaf.base, leaf.direction, leaf.normal, leaf.pivot):
            digest.update(np.asarray(arr, dtype=np.float64).tobytes())
        if leaf.stalk is not None:
            digest.update(np.ascontiguousarray(leaf.stalk, dtype=np.float64).tobytes())
    key = digest.digest()
    vkeys = organ_attrs.get("vkey") or organ_attrs["organ_id"]
    reference = _leaf_reference_for(obj)
    if reference is not None:
        out = _referenced_outcome(reference, rows, leaves, vkeys)
    else:
        cached = _SETTLE_CACHE.get(obj.name)
        if cached is not None and cached[0] == key:
            out = cached[1]
        elif obj.name in bridge._PROBING:
            raise bridge._CollisionNeeded(key, leaves, shapes, stems)   # the worker resolves it
        else:
            with bridge._stage("Leaf collision"):
                out = lc.resolve(leaves, shapes, stems)
            _SETTLE_CACHE[obj.name] = (key, out)
        if obj.name in _CAPTURING:
            _CAPTURED[obj.name] = _capture_outcome(out, rows, leaves, vkeys)
    stats: dict = {}
    drops = np.asarray(lp["drop"], dtype=np.float64).copy()
    render_drop = organ_attrs.get("render_drop")
    for k, row in enumerate(rows):
        i, c = row[0], row[1]
        tally = stats.setdefault(c.uid, [0, 0, 0]) if c.collide else [0, 0, 0]
        if out.pruned[k]:
            drops[i] = 1.0
            if render_drop is not None:
                render_drop[i] = 1.0
            row[2] = None
            row[6] = False
            tally[1] += 1
            continue
        if out.angle[k] > 0.0:
            rot, pivot = out.turn[k], leaves[k].pivot
            row[3] = pivot + rot @ (np.asarray(row[3], dtype=np.float64) - pivot)
            row[4], row[5] = rot @ row[4], rot @ row[5]
            if row[2] is not None:
                row[2] = pivot + (row[2] - pivot) @ rot.T
            row[6] = True
            tally[0] += 1
        tally[2] += int(out.unresolved[k])
    # A pruned leaf draws nothing: drop it from the stalk list too.
    drawn[:] = [row for row in drawn if drops[row[0]] < 0.5]
    lp["drop"] = drops
    LAST_COLLISIONS[obj.name] = {uid: tuple(t) for uid, t in stats.items()}


# --- the timelapse's leaf reference -----------------------------------------------
#
# Resolved frame by frame, leaf collision prunes a leaf that grew into its
# neighbour and turns one away the moment it touches -- in a timelapse leaves
# grew, vanished and swung about. Under Grow Over Timeline the collision is
# instead resolved once, on the plant at its fullest within Start Day..End Day
# (`_reference_day`), and every frame takes each leaf's outcome from there by
# its own place (component + `vkey`): a leaf pruned there never grows, one
# turned there grows in turned -- the same rule the Day already follows (every
# earlier Day a subset of the grown plant). A leaf the reference does not hold
# is drawn as it is.

_LEAF_REFERENCE: dict = {}

_CAPTURING: set = set()
_CAPTURED: dict = {}

_REFERENCE_SAMPLES = 12


def _leaf_frame(direction, normal) -> np.ndarray:
    """A leaf's own frame as columns: its length, its face, their cross."""
    d = np.asarray(direction, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    return np.column_stack((d, n, np.cross(d, n)))


def _capture_outcome(out, rows, leaves, vkeys) -> dict:
    """The resolved outcome by leaf, each turn stated in the leaf's own frame
    so it carries to a Day where the stem holds the leaf a little differently."""
    record = {}
    for k, row in enumerate(rows):
        key = (row[1].uid, int(vkeys[row[0]]))
        if out.pruned[k]:
            record[key] = None
        elif out.angle[k] > 0.0:
            frame = _leaf_frame(leaves[k].direction, leaves[k].normal)
            record[key] = frame.T @ np.asarray(out.turn[k], dtype=np.float64) @ frame
        else:
            record[key] = np.eye(3)
    return record


def _referenced_outcome(reference: dict, rows, leaves, vkeys):
    """An `Outcome`-shaped result for this frame's leaves from the reference:
    pruned there -> pruned, turned there -> the same turn in today's frame."""
    n = len(leaves)
    turn, angle = [np.eye(3)] * n, np.zeros(n)
    pruned, unresolved = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for k, row in enumerate(rows):
        key = (row[1].uid, int(vkeys[row[0]]))
        if key not in reference:
            continue
        local = reference[key]
        if local is None:
            pruned[k] = True
        elif not np.allclose(local, np.eye(3)):
            frame = _leaf_frame(leaves[k].direction, leaves[k].normal)
            turn[k] = frame @ local @ frame.T
            angle[k] = float(np.arccos(np.clip((np.trace(local) - 1.0) / 2.0, -1.0, 1.0)))
    return SimpleNamespace(turn=turn, angle=angle, pruned=pruned, unresolved=unresolved)


def _leaf_reference_key(obj) -> str:
    s = obj.plantgen
    return bridge.growth_signature(obj, include_day=False) + repr(
        (round(float(s.timelapse_start_day), 6), round(float(s.timelapse_end_day), 6)))


def _wants_leaf_reference(obj) -> bool:
    s = obj.plantgen
    return bool(s.animate_day) and any(
        c.collide and not c.mute and c.type in ("leaf", "bract") for c in s.components)


def _leaf_reference_for(obj):
    """This plant's leaf reference when it is in use now, else None -- never
    while the reference itself is being resolved."""
    if obj.name in _CAPTURING or not _wants_leaf_reference(obj):
        return None
    hit = _LEAF_REFERENCE.get(obj.name)
    if hit is None or hit[0] != _leaf_reference_key(obj):
        return None
    return hit[1]


def _reference_day(obj) -> float:
    """The Day in Start Day..End Day where the plant is at its fullest: the
    most leaves, then the most flowers (each an obstacle a leaf turns from),
    then the latest -- from the growth model alone, sampled."""
    s = obj.plantgen
    lo, hi = sorted((float(s.timelapse_start_day), float(s.timelapse_end_day)))
    profile = bridge.profile_from_object(obj)
    environment = env_bridge.build_environment(obj, profile)
    signals = bridge.growth_signals_for(obj, profile)
    best = None
    for day in np.linspace(lo, hi, _REFERENCE_SAMPLES):
        plant = build(profile, seed=s.seed, day=float(day), environment=environment,
                      signals=signals)
        score = (len(plant.organs_of_kind(OrganKind.LEAF)),
                 len(plant.organs_of_kind(OrganKind.FLOWER)), float(day))
        if best is None or score > best:
            best = score
    return best[2] if best is not None else hi


def ensure_leaf_reference(obj) -> None:
    """Resolve the timelapse's leaf reference if it is wanted and out of date
    (`_leaf_reference_key`: every growth and component input, Start / End Day):
    the organ points built once at `_reference_day`, their collision recorded.
    Writes nothing to the object; the per-plant records it passes through are
    put back."""
    if not _wants_leaf_reference(obj):
        _LEAF_REFERENCE.pop(obj.name, None)
        return
    key = _leaf_reference_key(obj)
    hit = _LEAF_REFERENCE.get(obj.name)
    if hit is not None and hit[0] == key:
        return
    day = _reference_day(obj)
    saved = {name: rec.get(obj.name) for name, rec in
             (("points", instancing.LAST_POINTS), ("collisions", LAST_COLLISIONS))}
    _CAPTURING.add(obj.name)
    _CAPTURED.pop(obj.name, None)
    try:
        bridge._organ_points(obj, obj.plantgen, day)
    finally:
        _CAPTURING.discard(obj.name)
        for name, rec in (("points", instancing.LAST_POINTS), ("collisions", LAST_COLLISIONS)):
            if saved[name] is None:
                rec.pop(obj.name, None)
            else:
                rec[obj.name] = saved[name]
    _LEAF_REFERENCE[obj.name] = (key, _CAPTURED.pop(obj.name, {}), day)
