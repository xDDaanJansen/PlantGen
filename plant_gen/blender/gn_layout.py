"""Automatic, semantics-preserving layout for generated Geometry Nodes trees.

`tidy(ng, frames)` takes any GeometryNodeTree (the main mesh group or one of
its sub-groups) and makes it readable: nodes are grouped into labelled
frames, each frame is laid out left-to-right as a layered graph
(Sugiyama-lite), and the frames themselves are laid out the same way as
super-nodes, Group Input far left and Group Output far right.

It never changes what the graph computes. The only things it touches are:
node locations, frame nodes it creates (and, when `frames` is given, any
pre-existing frames, which carry no computation), `node.parent`,
`node.hide`, Group Input output `socket.hide`, and Group Input nodes: a
frame-local Group Input that exposes the sockets used inside a frame is
semantically identical to the shared one, so links are moved onto it.
A link into a multi-input socket (Join Geometry) keeps its slot, because that
order is the element order of the result.

Facts probed live in Blender 5.2 (recorded because they decide the code):

* `node.location` of a parented node is RELATIVE to its frame; the RNA also
  exposes `node.location_absolute` (read/write, global view space).
  Parenting a node keeps its absolute position (location is rewritten).
  Moving a frame moves its children. So: parent first, then write
  `location_absolute` -- never compute frame offsets by hand.
* `node.dimensions` is (0, 0) until a node editor has drawn the tree, and is
  stale after `hide` / socket visibility changes until the next redraw. At
  UI scale 1.0, `dimensions.x == node.width`; the scale factor is taken
  empirically as `dimensions.x / node.width` so a scaled UI still divides
  out. Measured heights of the drawn PlantGen · Plant tree calibrated the
  estimator below (header 28 px, 22 px per socket row, +60 for an unlinked
  vector / rotation input, per-type button allowance); the Group Input
  estimate (28 + 22 * 32 = 732) matched its drawn height exactly.
* A frame with `shrink = True` refits to its children when drawn; before
  that its own location/width/height are whatever was set, so we set them to
  the computed padded bounds too (Blender pads a frame by 1.5 widget units
  = 30 px plus the label).
"""

from __future__ import annotations

import colorsys
import heapq

import bpy

# -- spacing (view units at UI scale 1) --------------------------------------
_GAP_X = 60.0  # between node columns inside a frame
_GAP_Y = 30.0  # between nodes in a column
_BLOCK_GAP_X = 100.0  # between frame columns
_BLOCK_GAP_Y = 80.0  # between frames stacked in a column
_FRAME_PAD = 30.0  # Blender's own frame margin (1.5 * widget unit)
_FRAME_LABEL = 36.0  # extra top room for the frame label
_SAFETY = 4.0  # added to every estimated node height

# -- node size estimate ------------------------------------------------------
_HEADER = 28.0
_ROW = 22.0
_VECTOR_EXTRA = 60.0  # an unlinked vector/rotation input draws 3 value fields
_COLLAPSED = 32.0
_REROUTE = 20.0

# Height of each node type's own buttons (dropdowns, checkboxes, widgets),
# measured on drawn nodes in 5.2 where available, conservative otherwise.
_BUTTONS = {
    "ShaderNodeMath": 54,
    "ShaderNodeVectorMath": 27,
    "FunctionNodeCompare": 52,
    "FunctionNodeBooleanMath": 27,
    "FunctionNodeIntegerMath": 27,
    "ShaderNodeClamp": 27,
    "ShaderNodeMapRange": 79,
    "GeometryNodeSwitch": 27,
    "GeometryNodeIndexSwitch": 27,
    "GeometryNodeMenuSwitch": 27,
    "GeometryNodeInputNamedAttribute": 27,
    "GeometryNodeStoreNamedAttribute": 27,
    "GeometryNodeCaptureAttribute": 27,
    "GeometryNodeAttributeStatistic": 52,
    "FunctionNodeRandomValue": 27,
    "GeometryNodeSampleIndex": 79,
    "GeometryNodeSampleNearest": 27,
    "GeometryNodeSampleNearestSurface": 27,
    "GeometryNodeRaycast": 54,
    "ShaderNodeValToRGB": 112,
    "ShaderNodeFloatCurve": 190,
    "ShaderNodeRGBCurve": 220,
    "ShaderNodeVectorCurve": 220,
    "FunctionNodeAxesToRotation": 52,
    "FunctionNodeAlignRotationToVector": 54,
    "FunctionNodeRotateRotation": 27,
    "GeometryNodeCollectionInfo": 27,
    "GeometryNodeObjectInfo": 27,
    "GeometryNodeSeparateGeometry": 27,
    "GeometryNodeDeleteGeometry": 27,
    "GeometryNodeMeshToCurve": 27,
    "GeometryNodeCurvePrimitiveCircle": 27,
    "FunctionNodeCombineColor": 27,
    "FunctionNodeSeparateColor": 27,
    "GeometryNodeViewer": 27,
    "GeometryNodeGroup": 27,
    "GeometryNodeTransform": 27,
    "ShaderNodeValue": 24,
    "FunctionNodeInputInt": 24,
    "FunctionNodeInputBool": 24,
    "FunctionNodeInputString": 24,
    "FunctionNodeInputVector": 70,
    "ShaderNodeCombineXYZ": 0,
    "ShaderNodeSeparateXYZ": 0,
    "GeometryNodeSetPosition": 0,
    "GeometryNodeInstanceOnPoints": 0,
    "GeometryNodeCurveToMesh": 0,
    "GeometryNodeRealizeInstances": 0,
    "GeometryNodeJoinGeometry": 0,
    "GeometryNodeInputIndex": 0,
    "GeometryNodeInputPosition": 0,
    "NodeGroupInput": 0,
    "NodeGroupOutput": 0,
}
_MIX_BUTTONS = {"RGBA": 104, "VECTOR": 79}

# Nodes that stay expanded under `collapse_sources` even with no linked input:
# their body is the visible state (a ramp / curve widget), or they are IO.
_KEEP_EXPANDED = {
    "NodeGroupInput",
    "NodeGroupOutput",
    "NodeFrame",
    "NodeReroute",
    "GeometryNodeViewer",
    "ShaderNodeValToRGB",
    "ShaderNodeFloatCurve",
    "ShaderNodeRGBCurve",
    "ShaderNodeVectorCurve",
}

_base_props_cache: set[str] | None = None


def _base_props() -> set[str]:
    global _base_props_cache
    if _base_props_cache is None:
        _base_props_cache = {p.identifier for p in bpy.types.Node.bl_rna.properties}
    return _base_props_cache


def _button_height(node) -> float:
    idn = node.bl_idname
    if idn == "ShaderNodeMix":
        return _MIX_BUTTONS.get(getattr(node, "data_type", ""), 54)
    if idn in _BUTTONS:
        h = float(_BUTTONS[idn])
        if idn == "GeometryNodeGroup" and node.node_tree is not None:
            # One header row per interface panel.
            n_panels = sum(
                1 for it in node.node_tree.interface.items_tree if it.item_type == "PANEL"
            )
            h += _ROW * n_panels
        return h
    # Unknown type: one row per own enum property, capped at three.
    base = _base_props()
    n_enum = sum(
        1
        for p in node.bl_rna.properties
        if p.identifier not in base and p.type == "ENUM" and not p.is_readonly
    )
    return 27.0 * min(n_enum, 3)


def _visible(sockets):
    return [s for s in sockets if s.enabled and not s.hide]


class _Size:
    """Estimated size of one node plus its socket y-offsets from the top."""

    __slots__ = ("w", "h", "sock")

    def __init__(self, w, h, sock):
        self.w = w
        self.h = h
        self.sock = sock  # socket.as_pointer() -> offset from node top


def _measure(node, touched: bool) -> _Size:
    sock: dict[int, float] = {}
    if node.bl_idname == "NodeReroute":
        for s in list(node.inputs) + list(node.outputs):
            sock[s.as_pointer()] = _REROUTE / 2
        return _Size(_REROUTE, _REROUTE, sock)
    w = float(node.width)
    vin = _visible(node.inputs)
    vout = _visible(node.outputs)
    if node.hide:
        tot = max(len([s for s in vin if s.is_linked]), len([s for s in vout if s.is_linked]))
        h = _COLLAPSED + max(0, tot - 4) * 10.0
        for s in list(node.inputs) + list(node.outputs):
            sock[s.as_pointer()] = h / 2
        return _Size(w, h + _SAFETY, sock)
    y = _HEADER
    for s in vout:
        sock[s.as_pointer()] = y + _ROW / 2
        y += _ROW
    y += _button_height(node)
    for s in vin:
        sock[s.as_pointer()] = y + _ROW / 2
        y += _ROW
        if (not s.is_linked) and s.type in ("VECTOR", "ROTATION") and not s.hide_value:
            y += _VECTOR_EXTRA
    h = y
    # A drawn, untouched node knows its real size; trust it when it is larger.
    dx, dy = node.dimensions
    if not touched and dx > 0 and dy > 0 and w > 0:
        scale = dx / w
        h = max(h, dy / scale) if scale > 0 else h
    # Hidden-but-still-listed sockets (disabled, hidden) sit at the top.
    for s in list(node.inputs) + list(node.outputs):
        sock.setdefault(s.as_pointer(), _HEADER / 2)
    return _Size(w, h + _SAFETY, sock)


def _abs_loc(node):
    loc = getattr(node, "location_absolute", None)
    if loc is not None:
        return float(loc[0]), float(loc[1])
    x, y = node.location
    p = node.parent
    while p is not None:
        x += p.location[0]
        y += p.location[1]
        p = p.parent
    return x, y


def _set_abs_loc(node, x, y):
    if hasattr(node, "location_absolute"):
        node.location_absolute = (x, y)
        return
    px = py = 0.0
    p = node.parent
    while p is not None:
        px += p.location[0]
        py += p.location[1]
        p = p.parent
    node.location = (x - px, y - py)


def _top_frame(node):
    p = node.parent
    top = None
    while p is not None:
        top = p
        p = p.parent
    return top


# -- 1D packing --------------------------------------------------------------
def _pack(targets, weights, sizes, gaps):
    """Place items (fixed order) at downward coords as close to `targets` as
    possible (weighted least squares) with item i+1 at least sizes[i]+gaps[i]
    below item i. Substituting out the offsets turns this into isotonic
    regression, solved exactly by pool-adjacent-violators."""
    n = len(targets)
    off = []
    acc = 0.0
    for i in range(n):
        off.append(acc)
        acc += sizes[i] + gaps[i]
    blocks = []  # [sum_wz, sum_w, count]
    for i in range(n):
        wz = weights[i] * (targets[i] - off[i])
        blocks.append([wz, weights[i], 1])
        while len(blocks) > 1 and (blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]):
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = []
    i = 0
    for swz, sw, cnt in blocks:
        v = swz / sw
        for _ in range(cnt):
            out.append(v + off[i])
            i += 1
    return out


# -- layered layout ----------------------------------------------------------
def _layered(items, sizes, edges, order_key, gap_x, gap_y, first=(), last=()):
    """Sugiyama-lite layout of `items` (hashable keys).

    sizes: key -> (w, h). edges: list of (u, v, port_u, port_v), ports being
    y-offsets from each item's top. gap_x / gap_y: a number, or a dict giving
    each item its own gap (two neighbours / columns use the larger). Returns
    ({key: (x, d)}, width, height) in a downward-positive frame whose origin
    is the top-left of the bounds.
    """
    gx = gap_x if isinstance(gap_x, dict) else dict.fromkeys(items, float(gap_x))
    gy = gap_y if isinstance(gap_y, dict) else dict.fromkeys(items, float(gap_y))
    idx = {k: i for i, k in enumerate(items)}
    preds = {k: [] for k in items}
    succs = {k: [] for k in items}
    for u, v, pu, pv in edges:
        if u == v or u not in idx or v not in idx:
            continue
        preds[v].append((u, pu, pv))
        succs[u].append((v, pu, pv))

    # Topological order (deterministic: ties broken by item order).
    indeg = {k: len({u for u, _, _ in preds[k]}) for k in items}
    heap = [(idx[k], k) for k in items if indeg[k] == 0]
    heapq.heapify(heap)
    topo = []
    seen = set()
    while heap:
        _, k = heapq.heappop(heap)
        if k in seen:
            continue
        seen.add(k)
        topo.append(k)
        for v in sorted({v for v, _, _ in succs[k]}, key=idx.get):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (idx[v], v))
    topo += [k for k in items if k not in seen]  # cycles (should not happen)

    first = set(first)
    last = set(last)
    col = {}
    for k in topo:
        c = 0
        for u, _, _ in preds[k]:
            if u in col:
                c = max(c, col[u] + 1)
        col[k] = 0 if k in first else c
    # Pull nodes right, next to their nearest consumer (shortens long links).
    for k in reversed(topo):
        if k in first or k in last:
            continue
        cs = [col[v] for v, _, _ in succs[k] if v not in last]
        if cs:
            col[k] = max(col[k], min(cs) - 1)
    if last:
        mx = max([col[k] for k in items if k not in last] or [-1])
        for k in last:
            col[k] = mx + 1
    ncol = max(col.values()) + 1 if col else 0
    layers = [[] for _ in range(ncol)]
    for k in sorted(items, key=order_key):
        layers[col[k]].append(k)

    # Crossing reduction: barycenter sweeps on normalised in-layer position.
    def positions():
        p = {}
        for layer in layers:
            n = len(layer)
            for i, k in enumerate(layer):
                p[k] = (i + 0.5) / n
        return p

    for sweep in range(4):
        rng = range(1, ncol) if sweep % 2 == 0 else range(ncol - 2, -1, -1)
        for c in rng:
            pos = positions()
            nbr = preds if sweep % 2 == 0 else succs
            layer = layers[c]
            keyed = []
            for i, k in enumerate(layer):
                ns = [pos[n] for n, _, _ in nbr[k]]
                b = sum(ns) / len(ns) if ns else pos[k]
                keyed.append((b, i, k))
            keyed.sort(key=lambda t: (t[0], t[1]))
            layers[c] = [k for _, _, k in keyed]

    # x per column.
    xs = []
    x = 0.0
    for c, layer in enumerate(layers):
        xs.append(x)
        x += max((sizes[k][0] for k in layer), default=0.0)
        if c + 1 < ncol:
            x += max(gx[k] for k in layer + layers[c + 1])
    width = x

    def gaps(layer):
        return [max(gy[a], gy[b]) for a, b in zip(layer, layer[1:], strict=False)] + [0.0]

    # y: stack, then relax toward socket-aligned targets, never overlapping.
    d = {}
    for layer in layers:
        acc = 0.0
        for k, g in zip(layer, gaps(layer), strict=True):
            d[k] = acc
            acc += sizes[k][1] + g

    def relax(c, use_preds, use_succs):
        layer = layers[c]
        tg, wt, hs = [], [], []
        for k in layer:
            vals = []
            if use_preds:
                vals += [d[u] + pu - pv for u, pu, pv in preds[k]]
            if use_succs:
                vals += [d[v] + pv - pu for v, pu, pv in succs[k]]
            if vals:
                tg.append(sum(vals) / len(vals))
                wt.append(float(len(vals)))
            else:
                tg.append(d[k])
                wt.append(1e-3)
            hs.append(sizes[k][1])
        for k, v in zip(layer, _pack(tg, wt, hs, gaps(layer)), strict=True):
            d[k] = v

    for c in range(ncol):
        relax(c, True, False)
    for c in range(ncol - 1, -1, -1):
        relax(c, True, True)
    for c in range(ncol):
        relax(c, True, True)

    pos = {}
    if items:
        dmin = min(d.values())
        height = max(d[k] - dmin + sizes[k][1] for k in items)
        for c, layer in enumerate(layers):
            for k in layer:
                pos[k] = (xs[c], d[k] - dmin)
    else:
        height = 0.0
    return pos, width, height


# -- graph edits -------------------------------------------------------------
def _collapse_sources(ng, touched):
    for n in ng.nodes:
        if n.bl_idname in _KEEP_EXPANDED or n.type == "FRAME":
            continue
        if any(s.is_linked for s in n.inputs):
            continue
        if not n.hide:
            n.hide = True
            touched.add(n.name)


def _hide_unused_gi_outputs(ng, touched):
    for n in ng.nodes:
        if n.bl_idname != "NodeGroupInput":
            continue
        for s in n.outputs:
            want = not s.is_linked
            if s.hide != want:
                try:
                    s.hide = want
                except (AttributeError, TypeError, RuntimeError):
                    continue
                touched.add(n.name)


def _relink(ng, link, new_from):
    """Replace one link's source socket, keeping its place in a multi-input.

    Probed in 5.2: a new link into a multi-input socket is appended (sort id
    = count), and removing a link does not renumber the others. So for a
    multi-input we add the new link, swap sort ids with the old one (the new
    link takes the old slot, the old goes last), then remove the old one --
    the order of every other link is untouched.
    """
    to_sock = link.to_socket
    muted = link.is_muted
    if to_sock.is_multi_input:
        nl = ng.links.new(new_from, to_sock, verify_limits=False)
        nl.swap_multi_input_sort_id(link)
        ng.links.remove(link)
    else:
        ng.links.remove(link)
        nl = ng.links.new(new_from, to_sock)
    nl.is_muted = muted


def _split_group_inputs(ng, frame_of, touched):
    """Give every frame fed by an outside Group Input its own Group Input."""
    gis = [n for n in ng.nodes if n.bl_idname == "NodeGroupInput"]
    if not gis:
        return
    needs: dict[str, list] = {}
    for lk in list(ng.links):
        fn = lk.from_node
        if fn.bl_idname != "NodeGroupInput":
            continue
        tf = frame_of.get(lk.to_node.name)
        if tf is None or frame_of.get(fn.name) == tf:
            continue
        needs.setdefault(tf, []).append(lk)
    if not needs:
        return
    order = {}
    for key in frame_of.values():
        order.setdefault(key, len(order))
    for tf in sorted(needs, key=lambda k: order.get(k, 0)):
        links = needs[tf]
        new_gi = ng.nodes.new("NodeGroupInput")
        new_gi.select = False
        frame_of[new_gi.name] = tf
        touched.add(new_gi.name)
        by_ident = {s.identifier: s for s in new_gi.outputs}
        for lk in links:
            _relink(ng, lk, by_ident[lk.from_socket.identifier])
    # Drop originals nothing reads from any more -- but never the last one.
    for gi in gis:
        if any(s.is_linked for s in gi.outputs):
            continue
        remaining = [n for n in ng.nodes if n.bl_idname == "NodeGroupInput"]
        if len(remaining) > 1:
            frame_of.pop(gi.name, None)
            ng.nodes.remove(gi)


# -- main entry --------------------------------------------------------------
def _frame_color(i):
    h = (0.08 + i * 0.618034) % 1.0
    return colorsys.hsv_to_rgb(h, 0.35, 0.42)


def tidy(ng, frames=None, *, split_group_inputs=True, collapse_sources=True) -> None:
    """Lay out `ng` tidily without changing what it computes.

    frames: optional ordered list of (label, [nodes]). One shrink-to-fit frame
    is created per non-empty label and those nodes are parented to it (Group
    Input / Output are never framed by the caller; frame-local Group Inputs
    are added by `split_group_inputs`). When given, any pre-existing frames
    are removed first; when None, the existing top-level frames are kept and
    used as the grouping.
    """
    nodes = ng.nodes
    # Snapshot original positions for a stable initial ordering.
    orig = {n.name: _abs_loc(n) for n in nodes}
    order_index = {n.name: i for i, n in enumerate(nodes)}

    # 1. Grouping: frame_of[name] = frame key (int index into frame_specs).
    frame_specs: list[tuple[str, object]] = []  # (label, existing frame or None)
    frame_of: dict[str, int] = {}
    io_types = ("NodeGroupInput", "NodeGroupOutput")
    if frames is not None:
        for n in nodes:
            if n.parent is not None:
                n.parent = None
        for f in [n for n in nodes if n.type == "FRAME"]:
            nodes.remove(f)
        for label, members in frames:
            names = []
            for m in members:
                if m is None or m.name not in nodes or m.name in frame_of:
                    continue
                if m.bl_idname in io_types or m.type == "FRAME":
                    continue
                names.append(m.name)
            if not names:
                continue
            key = len(frame_specs)
            frame_specs.append((str(label), None))
            for nm in names:
                frame_of[nm] = key
    else:
        tops = {}
        for n in nodes:
            if n.type == "FRAME":
                continue
            top = _top_frame(n)
            if top is None:
                continue
            if top.name not in tops:
                tops[top.name] = len(frame_specs)
                frame_specs.append((top.label or top.name, top))
            frame_of[n.name] = tops[top.name]

    touched: set[str] = set()
    if collapse_sources:
        _collapse_sources(ng, touched)
    if split_group_inputs:
        _split_group_inputs(ng, frame_of, touched)
    _hide_unused_gi_outputs(ng, touched)

    # 2. Measure.
    real = [n for n in nodes if n.type != "FRAME"]
    by_name = {n.name: n for n in real}
    size = {n.name: _measure(n, n.name in touched) for n in real}
    links = []
    for lk in ng.links:
        a, b = lk.from_node.name, lk.to_node.name
        if a not in by_name or b not in by_name:
            continue
        links.append(
            (
                a,
                b,
                size[a].sock.get(lk.from_socket.as_pointer(), 0.0),
                size[b].sock.get(lk.to_socket.as_pointer(), 0.0),
            )
        )

    def okey(name):
        x, y = orig.get(name, (0.0, 0.0))
        return (-y, x, order_index.get(name, 1 << 30), name)

    # 3. Blocks: one per frame, one per unframed node.
    block_of: dict[str, object] = {}
    members: dict[object, list[str]] = {}
    for name in sorted(by_name, key=lambda nm: order_index.get(nm, 1 << 30)):
        key = ("F", frame_of[name]) if name in frame_of else ("N", name)
        block_of[name] = key
        members.setdefault(key, []).append(name)

    inner: dict[object, dict[str, tuple[float, float]]] = {}
    bsize: dict[object, tuple[float, float]] = {}
    pad: dict[object, tuple[float, float]] = {}  # (left, top) inset of content
    for key, names in members.items():
        if key[0] == "N":
            inner[key] = {names[0]: (0.0, 0.0)}
            s = size[names[0]]
            bsize[key] = (s.w, s.h)
            pad[key] = (0.0, 0.0)
            continue
        sub = [e for e in links if block_of[e[0]] == key and block_of[e[1]] == key]
        gis = [nm for nm in names if by_name[nm].bl_idname == "NodeGroupInput"]
        pos, w, h = _layered(
            names,
            {nm: (size[nm].w, size[nm].h) for nm in names},
            sub,
            okey,
            _GAP_X,
            _GAP_Y,
            first=gis,
        )
        inner[key] = pos
        top = _FRAME_PAD + _FRAME_LABEL
        bsize[key] = (w + 2 * _FRAME_PAD, h + top + _FRAME_PAD)
        pad[key] = (_FRAME_PAD, top)

    # 4. Blocks as super-nodes.
    bedges = []
    for a, b, pa, pb in links:
        ka, kb = block_of[a], block_of[b]
        if ka == kb:
            continue
        bedges.append(
            (ka, kb, pad[ka][1] + inner[ka][a][1] + pa, pad[kb][1] + inner[kb][b][1] + pb)
        )
    bkeys = list(members)
    firsts = [k for k in bkeys if k[0] == "N" and by_name[k[1]].bl_idname == "NodeGroupInput"]
    lasts = [k for k in bkeys if k[0] == "N" and by_name[k[1]].bl_idname == "NodeGroupOutput"]

    def bkey_order(k):
        if k[0] == "F":
            return (0, k[1], 0.0, "")
        x, y = orig.get(k[1], (0.0, 0.0))
        return (1, -y, x, k[1])

    # Frames keep frame-sized gaps; two loose nodes sit at node spacing.
    bgx = {k: _BLOCK_GAP_X if k[0] == "F" else _GAP_X for k in bkeys}
    bgy = {k: _BLOCK_GAP_Y if k[0] == "F" else _GAP_Y for k in bkeys}
    bpos, _, _ = _layered(bkeys, bsize, bedges, bkey_order, bgx, bgy, first=firsts, last=lasts)

    # 5. Frames: create (or reuse), parent, then write absolute locations.
    frame_nodes = {}
    for i, (label, existing) in enumerate(frame_specs):
        key = ("F", i)
        if key not in members:
            continue
        if existing is None:
            fr = nodes.new("NodeFrame")
            fr.label = label
            fr.name = "Frame " + label
            fr.use_custom_color = True
            fr.color = _frame_color(i)
            fr.select = False
        else:
            fr = existing
        fr.shrink = True
        frame_nodes[key] = fr

    # Anchor the layout at the old top-left so the tree does not jump away.
    ox = min((x for x, _ in orig.values()), default=0.0)
    oy = max((y for _, y in orig.values()), default=0.0)

    for key, fr in frame_nodes.items():
        bx, bd = bpos[key]
        _set_abs_loc(fr, ox + bx, oy - bd)
        fr.width, fr.height = bsize[key]
    for key, names in members.items():
        bx, bd = bpos[key]
        lx, ld = pad[key]
        fr = frame_nodes.get(key)
        for nm in names:
            n = by_name[nm]
            if fr is not None:
                if _top_frame(n) is not fr and n.parent is not fr:
                    n.parent = fr
            elif n.parent is not None and frames is not None:
                n.parent = None
            x, dd = inner[key][nm]
            _set_abs_loc(n, ox + bx + lx + x, oy - (bd + ld + dd))
