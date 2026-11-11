"""Rig layout -- the bones and skin weights of a grown plant, bpy-free.

The deform skeleton is built for engines first (one root, a clean branch
hierarchy, a bounded bone count, at most two influences per vertex) and
Blender's conveniences (bone collections, display) ride on top without
changing it.

- **Stems and branches:** each axis's node polyline is simplified -- a
  bone runs on while every internode along it stays within its level's Bone
  Angle of the bone's own chord AND the internodes it spans would turn no
  more than that between them under a sideways load of the plant's own
  weight (`sideways_turn`: thin, limp, loaded stems get joints, stiff ones
  stay long), and starts afresh past either or where a branch leaves
  (`chain_breaks`). 0 deg = a bone per node. Bones down an axis are
  connected; an axis's first bone hangs off the parent bone covering the node
  it grew from, a stem's off the root.
- **Organs:** every drawn leaf, bract, flower, bud and fruit has a bone of its
  own, from where it attaches (its stalk's root on the stem, when it has a
  petiole / pedicel) to its farthest point, parented to the stem bone that
  carries its node. The stalk and the organ ride that bone rigidly.
- **Weights:** a stem vertex goes to the bone of its own axis it lies along,
  blended with the neighbouring bone across each joint (a smoothstep over
  `BLEND` of the shorter bone's length either side) -- also across where a
  branch leaves its parent -- so fewer bones still bend the tube smoothly.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

ROOT = "root"
BLEND = 0.3              # each joint blends over this fraction of the shorter bone, per side
MIN_BONE = 1e-3          # a stub length for a degenerate bone (Blender drops zero-length ones)
SATURATED_TURN = 90.0    # degrees: the most a stem turns under a sideways push (it lies down)
ORGAN_GROUPS = {"leaf": "Leaves", "bract": "Bracts", "flower": "Flowers", "bud": "Buds",
                "apex": "Buds", "fruit": "Fruit"}


@dataclass
class Bone:
    name: str
    head: np.ndarray
    tail: np.ndarray
    parent: str | None
    connect: bool
    group: str
    roll_ref: np.ndarray | None = None      # the bone's Z is aligned toward this
    width: float = 0.0                      # stem radius at the bone's head
    flex: float = 0.0                       # degrees its internodes turn under a sideways push
    organ: bool = False                     # an organ's bone (rides its node in Bone Wind)


@dataclass
class Organ:
    """A drawn organ, as the rig sees it."""
    oid: int
    kind: str                     # a key of ORGAN_GROUPS
    base: np.ndarray              # where it is drawn from (the instance point)
    tip: np.ndarray               # its farthest drawn point from `base`


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def chain_breaks(points: np.ndarray, tol_deg: float, forced=(), flex=None) -> list[int]:
    """Node indices where bones start and end along a polyline (first and
    last always): a bone runs on while every segment along it stays within
    `tol_deg` of the bone's chord AND (given `flex`, each internode's turn
    under a sideways load, `sideways_turn`) the internodes it spans would
    turn no more than `tol_deg` between them -- so a straight but limp stem
    still gets the joints it bends at. 0 = every node. A node in `forced`
    (where a branch leaves) always ends a bone, so the branch's joint meets
    one."""
    n = len(points) - 1
    cuts = sorted({int(k) for k in forced if 0 < int(k) < n})
    if cuts:
        out, start = [0], 0
        for end in [*cuts, n]:
            part = None if flex is None else flex[start:end]
            out += [start + k for k in chain_breaks(points[start:end + 1], tol_deg,
                                                     flex=part)[1:]]
            start = end
        return out
    if n <= 0:
        return [0]
    if tol_deg <= 0.0:
        return list(range(n + 1))
    cos_tol = math.cos(math.radians(min(tol_deg, 89.0)))
    segs = [_unit(points[k + 1] - points[k]) for k in range(n)]
    breaks, start, end = [0], 0, 1
    while end < n:
        chord = _unit(points[end + 1] - points[start])
        fits = chord is None or all(
            s is None or float(np.dot(s, chord)) >= cos_tol for s in segs[start:end + 1])
        if fits and flex is not None:
            fits = float(np.sum(flex[start:end + 1])) <= tol_deg
        if fits:
            end += 1
        else:
            breaks.append(end)
            start, end = end, end + 1
    breaks.append(n)
    return breaks


@dataclass
class AxisChain:
    axis_id: int
    level: int
    names: list[str]              # bone names, base to tip
    breaks: list[int]             # node indices bounding them
    points: np.ndarray            # the axis's nodes

    def covering(self, phytomer_index: int) -> str:
        """The bone carrying phytomer `phytomer_index` (whose tip node is
        phytomer_index + 1), clamped to the axis."""
        node = min(max(phytomer_index, 0), len(self.points) - 2) + 1
        for i in range(len(self.names)):
            if node <= self.breaks[i + 1]:
                return self.names[i]
        return self.names[-1]


def sideways_turn(plant, profile) -> dict[int, np.ndarray]:
    """How far, in degrees, each internode would turn if the plant's own
    weight pushed sideways (1 g across it -- the rig's reference gust): the
    gravity pass's own Euler-Bernoulli turn, 4 g M a L / (pi E r^4), with the
    lever arm the full distance to the carried mass's centre (a sideways load
    has no vertical stem to hide behind) and the same kink cap. Thin, limp,
    loaded stems turn far, thick stiff ones barely: the rig puts its joints
    where the plant can actually bend. {axis id: per-phytomer degrees}."""
    from ..growth import physics

    mass, moment = physics._supported(plant, profile)
    out = {}
    for axis in plant.axes.values():
        if not axis.phytomers:
            continue
        modulus = physics._axis_modulus(plant, profile, axis)
        turns = np.zeros(len(axis.phytomers))
        for i, ph in enumerate(axis.phytomers):
            carried = mass.get((axis.id, i), 0.0)
            if carried <= 0.0:
                continue
            centroid = moment[(axis.id, i)] / carried
            arm = float(np.linalg.norm(centroid - np.asarray(ph.base_position)))
            radius = max(float(ph.radius), 1e-6)
            length = max(float(ph.length), 0.0)
            e_local = float(ph.debug_stiffness) or modulus
            bend = math.degrees(4.0 * physics._G * carried * arm * length
                                / (math.pi * max(e_local, 1e-6) * radius ** 4))
            turns[i] = min(bend, math.degrees(length / (physics._KINK_RATIO * radius)))
        # A stem pushed sideways at most lies down along the push: its turn
        # saturates at 90 degrees in all, where linear beam theory runs on (a
        # slender spike summed to 1466). Kept where it bends, scaled to that.
        total = float(turns.sum())
        if total > SATURATED_TURN:
            turns *= SATURATED_TURN / total
        out[axis.id] = turns
    return out


def stem_bones(plant, tol_of_level, *, skip=lambda axis: False, flex=None):
    """The root and every axis's bones (`flex`: `sideways_turn`, so bones
    follow where each stem can bend). Returns (bones, chains by axis id)."""
    bones: list[Bone] = []
    chains: dict[int, AxisChain] = {}
    tops = [p.tip_position for a in plant.axes.values() for p in a.phytomers]
    height = max((float(t[2]) for t in tops), default=0.0)
    root_len = max(0.01, 0.05 * height)
    branch_nodes: dict[int, set] = {}
    for axis in plant.axes.values():
        if axis.parent_axis_id >= 0 and not skip(axis):
            branch_nodes.setdefault(axis.parent_axis_id, set()).add(
                int(axis.parent_phytomer_index) + 1)
    bones.append(Bone(ROOT, np.zeros(3), np.array([0.0, 0.0, root_len]), None, False, "Root",
                      np.array([0.0, 1.0, 0.0])))
    for axis in plant.axes_in_topological_order():
        if skip(axis) or not axis.phytomers:
            continue
        pts = np.array([axis.phytomers[0].base_position]
                       + [p.tip_position for p in axis.phytomers], dtype=np.float64)
        level = int(getattr(axis, "level", 0) or 0)
        parent = chains.get(axis.parent_axis_id)
        if level < 0:
            # An inflorescence axis has no level of its own: it takes the Bone
            # Angle of the level it hangs off.
            prefix, group = "Infl", "Inflorescence"
            tol = tol_of_level(parent.level if parent is not None else 0)
        else:
            prefix, group = f"L{level}", f"Level {level}"
            tol = tol_of_level(level)
        breaks = chain_breaks(pts, float(tol), branch_nodes.get(axis.id, ()),
                              None if flex is None else flex.get(axis.id))
        names = [f"{prefix}_A{axis.id}_{i:02d}" for i in range(len(breaks) - 1)]
        for i, name in enumerate(names):
            head, tail = pts[breaks[i]].copy(), pts[breaks[i + 1]].copy()
            if np.linalg.norm(tail - head) < MIN_BONE:
                d = _unit(np.asarray(axis.phytomers[min(breaks[i], len(axis.phytomers) - 1)]
                                     .direction, dtype=np.float64))
                tail = head + (d if d is not None else np.array([0.0, 0.0, 1.0])) * MIN_BONE
            if i:
                bparent, connect = names[i - 1], True
            elif parent is not None:
                bparent, connect = parent.covering(axis.parent_phytomer_index), False
            else:
                bparent, connect = ROOT, False
            ref = axis.phytomers[min(breaks[i], len(axis.phytomers) - 1)].reference
            bones.append(Bone(name, head, tail, bparent, connect, group,
                              np.asarray(ref, dtype=np.float64) if ref is not None else None,
                              width=float(axis.phytomers[min(breaks[i],
                                                             len(axis.phytomers) - 1)].radius),
                              flex=(float(np.sum(flex[axis.id][breaks[i]:breaks[i + 1]]))
                                    if flex is not None and axis.id in flex else 0.0)))
        chains[axis.id] = AxisChain(axis.id, level if level >= 0 else
                                    (parent.level if parent is not None else 0),
                                    names, breaks, pts)
    return bones, chains


TRUNK_GROUP = "Trunk"
RIDER_GROUP = "Branches"
ORGAN_GROUP = "Organs"
ORGAN_WIND_BINS = 4
"""Organ simulation groups, by how free the stem bone they hang from is. Dynamic
Wind reads a dual influence along a bone CHAIN, and every organ bone is a chain
of one, which always takes the Min: flexibility per organ only reaches Unreal
as a group's single influence."""
TRUNK_LINE_BONES = 16
"""How many bones down a root-to-tip line take Dynamic Wind's trunk path. It is
smooth (one slow wave about one axis for the whole plant), but every bone adds
the same turn: Rudbeckia's lines are 8 bones long, a hazel's 76, which would
fold over. Past this a stem rides its parent (`RIDER_GROUP`): Dynamic Wind's
branch path jitters at any Influence above 0 (Laburnum and Corylus, down to a
Max of 0.01), so a stem never takes it."""
BRANCH_WIND_SCALE = 0.35
"""Our flexibility (0..1 over the limpest stem bone) to Dynamic Wind's branch
Influence. That path adds a fast oscillation and per-bone noise straight onto
each bone, in proportion, which is too much for one bone on its own: calibrated
by eye in UE 5.7 on Rudbeckia (flowers shook at 1, sway cleanly here)."""
TRUNK_SWAY_BONES = 10
"""The longest line of trunk-path bones a plant sways over in full: the perennials
it was calibrated on reach 10. Every bone adds the same turn, so a longer line
would sway further for having more bones (Araucaria, 17 bones, outswayed them
all): its gust is shared out over its bones instead."""
BRANCH_MIN_BONES = 8
"""A run of bones past `TRUNK_LINE_BONES` shorter than this stays on the trunk
path: so few bones add little turn to the line, and a twig keeps a sway of its
own instead of riding stiffly."""


def branch_chains(parent: dict, group: dict) -> list[list[str]]:
    """Runs of bones: `parent`: bone -> parent; `group`: bone -> its group, for
    the bones in question. Bones joined within one group run on as the longest
    line down; each side twig off it is a run of its own."""
    kids: dict[str, list[str]] = {}
    heads = []
    for name, g in group.items():
        up = parent.get(name)
        if up is not None and group.get(up) == g:
            kids.setdefault(up, []).append(name)
        else:
            heads.append(name)
    down = []
    stack = list(heads)
    while stack:
        name = stack.pop()
        down.append(name)
        stack.extend(kids.get(name, ()))
    height: dict[str, int] = {}
    for name in reversed(down):
        height[name] = 1 + max((height[k] for k in kids.get(name, ())), default=0)
    chains = []
    stack = list(heads)
    while stack:
        name = stack.pop()
        chain = []
        while name is not None:
            chain.append(name)
            below = kids.get(name, ())
            best = max(below, key=height.__getitem__, default=None)
            stack.extend(k for k in below if k != best)
            name = best
        chains.append(chain)
    return chains


def _line_depths(bones: list[Bone]) -> dict[str, int]:
    """Each bone's place on its root-to-tip line: the root 0, its children 1..."""
    parent = {b.name: b.parent for b in bones}
    depths: dict[str, int] = {}
    for name in parent:
        walk = []
        node = name
        while node is not None and node not in depths and node in parent:
            walk.append(node)
            node = parent[node]
        depth = depths.get(node, -1)
        for n in reversed(walk):
            depth += 1
            depths[n] = depth
    return depths


def wind_groups(bones: list[Bone]) -> dict:
    """Unreal Dynamic Wind's classification of the skeleton
   : every bone in one simulation group, each
    group's wind influence, and the trunk's gust attenuation -- what Unreal
    simulates the sway from in engine. Groups, in order:

    * **Trunk** -- the root and every stem bone within `TRUNK_LINE_BONES` of
      it, whatever its branching level: `bIsTrunkGroup` (the trunk path has no
      use for an Influence), and any run past that depth shorter than
      `BRANCH_MIN_BONES`.
    * **Branches** -- the other stem bones past that depth, at Influence 0:
      they ride their parent, and a tree's outer wood moves as a whole.
    * **Organs 1..n** (every leaf / bract / flower / bud / fruit bone): an
      organ as free as the stem bone it hangs from, binned by that freedom
      (`ORGAN_WIND_BINS`), each bin one influence, its mean x
      `BRANCH_WIND_SCALE`.

    `GustAttenuation` (Unreal: attenuates gusts on trunk groups uniformly) is
    1 less the stems' (level 0) mean flexibility over the most flexible, the
    part left shared over the trunk path's longest line past
    `TRUNK_SWAY_BONES`: a stiff trunk shrugs gusts off. Returns {"groups": [{name, settings}],
    "bones": {bone name: group index}, "gust_attenuation": float}."""
    stem = [b for b in bones if not b.organ and b.name != ROOT]
    top = max((b.flex for b in stem), default=0.0)

    def rel(v):
        return float(min(1.0, max(0.0, v / top))) if top > 0.0 else 0.0

    depths = _line_depths(bones)
    outer = {b.name: b.group for b in stem if depths[b.name] > TRUNK_LINE_BONES}
    parents = {b.name: b.parent for b in bones}
    riders = {n for chain in branch_chains(parents, outer) if len(chain) >= BRANCH_MIN_BONES
              for n in chain}
    by_name = {b.name: b for b in stem}
    freedom = {b.name: rel(by_name[b.parent].flex) if b.parent in by_name else 1.0
               for b in bones if b.organ}

    def organ_group(name):
        k = min(int(freedom[name] * ORGAN_WIND_BINS), ORGAN_WIND_BINS - 1)
        return f"{ORGAN_GROUP} {k + 1}"

    order = [TRUNK_GROUP] + ([RIDER_GROUP] if riders else [])
    order += sorted({organ_group(n) for n in freedom})
    index = {g: i for i, g in enumerate(order)}

    def group_of(b):
        if b.organ:
            return organ_group(b.name)
        return RIDER_GROUP if b.name in riders else TRUNK_GROUP

    members: dict[str, list] = {g: [] for g in order}
    assign = {}
    for b in bones:
        g = group_of(b)
        assign[b.name] = index[g]
        members[g].append(b)
    groups = []
    for g in order:
        if g == TRUNK_GROUP:
            groups.append({"name": g, "settings": {
                "bIsTrunkGroup": 1, "bUseDualInfluence": 0, "Influence": 1.0}})
        elif g == RIDER_GROUP:
            groups.append({"name": g, "settings": {
                "bIsTrunkGroup": 0, "bUseDualInfluence": 1,
                "MinInfluence": 0.0, "MaxInfluence": 0.0, "ShiftTop": 0.0}})
        else:
            gives = [freedom[b.name] for b in members[g]]
            groups.append({"name": g, "settings": {
                "bIsTrunkGroup": 0, "bUseDualInfluence": 0,
                "Influence": round(BRANCH_WIND_SCALE * sum(gives) / len(gives), 3)}})
    trunk = [rel(b.flex) for b in stem if b.group == "Level 0"]
    line = {}
    for b in sorted(bones, key=lambda b: depths[b.name]):
        line[b.name] = line.get(b.parent, 0) + (group_of(b) == TRUNK_GROUP)
    share = min(1.0, TRUNK_SWAY_BONES / max(max(line.values(), default=1), 1))
    gust = round(1.0 - (sum(trunk) / len(trunk) if trunk else 0.0) * share, 3)
    return {"groups": groups, "bones": assign, "gust_attenuation": gust}


def wind_json(wind: dict, joint_names, ground_cover: bool = False) -> dict:
    """The Dynamic Wind import file for a skeleton whose joints are
    `joint_names` (every bone, as exported), from `wind_groups`."""
    return {
        "GustAttenuation": wind["gust_attenuation"],
        "bIsGroundCover": bool(ground_cover),
        "SimulationGroups": [g["settings"] for g in wind["groups"]],
        "Joints": [{"JointName": name, "SimulationGroupIndex": int(wind["bones"].get(name, 0))}
                   for name in joint_names],
    }


def flowers_on_shared_axes(organs: list[Organ], axis_of: dict[int, int]) -> set[int]:
    """The flowers that ride their stem bone instead of having their own: those
    on an axis carrying several flowers (`axis_of`: organ id -> axis id). A
    raceme's flowers sway with it, and a bone each is most of a flowering
    tree's rig (Laburnum: 29,595 of ~47,400). A lone flower head keeps its bone,
    since it nods on its own stalk."""
    on_axis: dict[int, list[int]] = {}
    for o in organs:
        if o.kind == "flower" and o.oid in axis_of:
            on_axis.setdefault(axis_of[o.oid], []).append(o.oid)
    return {oid for oids in on_axis.values() if len(oids) > 1 for oid in oids}


MAX_BONES = 32767
"""Unreal keeps a bone index in 16 signed bits: importing bone 32,768 overflowed
to a negative index and took the editor down (Araucaria: 35,078 bones)."""


def organs_riding(organs: list[Organ], axis_of: dict[int, int], stem_bones: int) -> set[int]:
    """The organs that ride their stem bone: flowers sharing an axis, and, when
    the plant would still pass `MAX_BONES`, every organ sharing an axis. A needle
    tree or a clubmoss is tens of thousands of small leaves that do not move on
    their own; a plant within the limit keeps its leaves' bones."""
    riders = flowers_on_shared_axes(organs, axis_of)
    if stem_bones + len(organs) - len(riders) <= MAX_BONES:
        return riders
    on_axis: dict[int, list[int]] = {}
    for o in organs:
        if o.oid in axis_of:
            on_axis.setdefault(axis_of[o.oid], []).append(o.oid)
    return riders | {oid for oids in on_axis.values() if len(oids) > 1 for oid in oids}


def organ_bones(organs: list[Organ], heads: dict[int, np.ndarray], parents: dict[int, str],
                fallback_parent):
    """A bone per drawn organ: from `heads[oid]` (its stalk's root) or its
    base, to its tip; parented to `parents[oid]`, else `fallback_parent(head)`."""
    bones = []
    for o in organs:
        head = heads.get(o.oid, o.base)
        tail = o.tip
        if np.linalg.norm(tail - head) < MIN_BONE:
            d = _unit(o.tip - o.base)
            tail = head + (d if d is not None else np.array([0.0, 0.0, 1.0])) * MIN_BONE
        parent = parents.get(o.oid) or fallback_parent(head)
        bones.append(Bone(f"{o.kind}_{o.oid}", np.asarray(head, dtype=np.float64),
                          np.asarray(tail, dtype=np.float64), parent, False,
                          ORGAN_GROUPS.get(o.kind, "Organs"), organ=True))
    return bones


def _segment_project(co, head, tail):
    """Per point: (distance to the segment, distance along it from head)."""
    d = tail - head
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        return np.linalg.norm(co - head, axis=1), np.zeros(len(co))
    u = d / length
    s = np.clip((co - head) @ u, 0.0, length)
    return np.linalg.norm(co - (head + np.outer(s, u)), axis=1), s


def _polyline_distance(co, points):
    best = np.full(len(co), np.inf)
    for k in range(len(points) - 1):
        dist, _ = _segment_project(co, points[k], points[k + 1])
        best = np.minimum(best, dist)
    return best


def axis_segments(chains: dict[int, AxisChain]):
    """Every internode of every axis as arrays, for `nearest_on_axes`: starts,
    ends, axis id and internode index per segment, in chain order."""
    a, b, aids, ks = [], [], [], []
    for aid, chain in chains.items():
        pts = np.asarray(chain.points, dtype=np.float64)
        n = len(pts) - 1
        if n > 0:
            a.append(pts[:-1])
            b.append(pts[1:])
            aids.append(np.full(n, aid, dtype=np.int64))
            ks.append(np.arange(n))
    if not a:
        empty = np.empty((0, 3))
        return empty, empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(a), np.concatenate(b), np.concatenate(aids), np.concatenate(ks)


def nearest_many_on_axes(points, segments) -> list:
    """`nearest_on_axes` for many points, through the same octree as the stems,
    so the cost is not points x segments."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out: list = [(None, 0, None)] * len(points)
    starts, ends, aids, ks = segments
    if not len(points) or not len(starts):
        return out

    def gap(c, i):
        return _segments_distance(c[None, :], starts[i], ends[i])[0]

    def solve(cell, near_sets):
        i = near_sets[0]
        local = (starts[i], ends[i], aids[i], ks[i])
        for v in cell.tolist():
            out[v] = nearest_on_axes(points[v], local)

    _over_cells(points, [len(starts)], [gap], solve)
    return out


def nearest_on_axes(point, segments):
    """The nearest point on any axis's centre line to `point`: (axis id,
    phytomer index of that internode, the point on the line). `segments`:
    `axis_segments`. Neighbouring internodes share a node, so a tie is common:
    the near-ties are walked in chain order and the first nearest wins."""
    starts, ends, aids, ks = segments
    if not len(starts):
        return None, 0, None
    d = ends - starts
    l2 = (d * d).sum(1)
    rel = point - starts
    t = np.clip((rel * d).sum(1) / np.maximum(l2, 1e-24), 0.0, 1.0)
    t[l2 < 1e-24] = 0.0
    gap = np.linalg.norm(rel - t[:, None] * d, axis=1)
    best = (None, 0, None, np.inf)
    for i in np.nonzero(gap <= gap.min() * (1.0 + 1e-9) + 1e-15)[0]:
        seg = ends[i] - starts[i]
        seg2 = float(seg @ seg)
        u = 0.0 if seg2 < 1e-24 else min(max(float((point - starts[i]) @ seg) / seg2, 0.0), 1.0)
        foot = starts[i] + u * seg
        dist = float(np.linalg.norm(point - foot))
        if dist < best[3]:
            best = (int(aids[i]), int(ks[i]), foot, dist)
    return best[:3]


def _segments_distance(co, a, b):
    """(points x segments) distance from each point to each segment a -> b."""
    d = b - a
    l2 = np.maximum((d * d).sum(1), 1e-24)
    rel = co[:, None, :] - a[None, :, :]
    t = np.clip((rel * d[None]).sum(-1) / l2[None], 0.0, 1.0)
    return np.linalg.norm(rel - t[..., None] * d[None], axis=-1)


def _segment_arrays(polylines):
    """Flatten [(key, points, radii|None)] into segment arrays + keys + radii."""
    a, b, key, rad = [], [], [], []
    for k, pts, radii in polylines:
        for s in range(len(pts) - 1):
            a.append(pts[s])
            b.append(pts[s + 1])
            key.append(k)
            rad.append(float(radii[s]) if radii is not None else 0.0)
    return (np.array(a, dtype=np.float64).reshape(-1, 3), np.array(b, dtype=np.float64)
            .reshape(-1, 3), np.array(key, dtype=np.int64), np.array(rad))


def assign_stem(co, axes, stalks, chunk: int = 1024):
    """Which axis, or which organ's stalk, each stem-mesh vertex belongs to --
    from the geometry and the model alone (the welded mesh's tube / axis
    labels are not reliable at junctions).

    `axes`: [(axis id, node points, per-internode radius)]; `stalks`: [(organ
    id, stalk centre line from its node out to the organ)]. A vertex is a
    stalk's when it lies nearer that stalk's centre line than the stem's
    surface (the stem centre line less its radius); else the nearest axis's.
    Returns (axis id per vertex, owning organ id per vertex or -1, nearest
    stalk's organ id per vertex or -1, distance to that stalk's centre line)."""
    n = len(co)
    axis_of = np.full(n, -1, dtype=np.int64)
    owner = np.full(n, -1, dtype=np.int64)
    near = np.full(n, -1, dtype=np.int64)
    near_d = np.full(n, np.inf)
    aa, ab, akey, arad = _segment_arrays(axes)
    sa, sb, skey, _ = _segment_arrays([(oid, pts, None) for oid, pts in stalks])
    if not len(aa):
        return axis_of, owner, near, near_d
    measures = [lambda c, i: _segments_distance(c[None, :], aa[i], ab[i])[0]]
    if len(sa):
        measures.append(lambda c, i: _segments_distance(c[None, :], sa[i], sb[i])[0])

    def place(cell, near_sets):
        axis_near = near_sets[0]
        stalk_near = near_sets[1] if len(near_sets) > 1 else None
        for lo in range(0, len(cell), chunk):
            at = cell[lo:lo + chunk]
            pts = co[at]
            d = _segments_distance(pts, aa[axis_near], ab[axis_near])
            hit = np.argmin(d, axis=1)
            j = axis_near[hit]
            dmin = d[np.arange(len(pts)), hit]
            axis_of[at] = akey[j]
            if stalk_near is not None:
                ds = _segments_distance(pts, sa[stalk_near], sb[stalk_near])
                hit = np.argmin(ds, axis=1)
                k = stalk_near[hit]
                dk = ds[np.arange(len(pts)), hit]
                on_stalk = dk < dmin - arad[j]
                owner[at] = np.where(on_stalk, skey[k], -1)
                near[at] = skey[k]
                near_d[at] = dk

    _over_cells(co, [len(aa), *([len(sa)] if len(sa) else [])], measures, place)
    return axis_of, owner, near, near_d


CELL_VERTICES = 400
"""About how many vertices `assign_stem` takes per grid cell (they cluster on the
stems, so a real cell holds more)."""


def _over_cells(co, sizes, measures, work) -> None:
    """Run `work(cell, candidates)` over `co` split into an octree of about
    `CELL_VERTICES` vertices a cell, each with the items that can be nearest to
    it, on every core. `sizes`: how many items in each set; `measures`: per set,
    (point, indices) -> those items' distances from the point.

    Moving by a node's radius changes a distance by at most that radius, so the
    nearest item to any vertex in a node is within 2 x radius of the node
    centre's nearest. Each node filters its parent's list by that bound, so the
    lists shrink on the way down, and `argmin` over a cell's list finds what it
    would over every item, ties included (the lists stay ascending). Cells share
    no vertices, so the threads write apart."""
    leaves = []
    todo = [(np.arange(len(co)), [np.arange(n) for n in sizes])]
    while todo:
        idx, lists = todo.pop()
        pts = co[idx]
        low, high = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * (low + high)
        radius = float(np.linalg.norm(pts - centre, axis=1).max())
        kept = []
        for measure, items in zip(measures, lists, strict=True):
            gap = measure(centre, items)
            kept.append(items[gap <= gap.min() + 2.0 * radius + 1e-9])
        if len(idx) <= CELL_VERTICES or float((high - low).max()) < 1e-6:
            leaves.append((idx, kept))
            continue
        side = (pts > centre).astype(np.int64) @ np.array([1, 2, 4])
        todo.extend((idx[g], kept) for g in group_indices(side).values())
    with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        list(pool.map(lambda leaf: work(*leaf), leaves))


def group_indices(ids, where=None) -> dict[int, np.ndarray]:
    """{id: ascending indices holding it}, ids ascending: `nonzero(ids == id)`
    for every id at once, since one mask per organ is organs x vertices (a
    Laburnum: 36,000 x 19 million). `where`: only these positions count."""
    ids = np.asarray(ids)
    idx = np.arange(len(ids)) if where is None else np.nonzero(where)[0]
    if not len(idx):
        return {}
    picked = ids[idx]
    order = np.argsort(picked, kind="stable")
    picked, idx = picked[order], idx[order]
    cuts = np.nonzero(np.diff(picked))[0] + 1
    starts = np.concatenate(([0], cuts))
    return dict(zip((int(v) for v in picked[starts]), np.split(idx, cuts), strict=True))


def components(n: int, edges) -> np.ndarray:
    """The connected piece of the mesh each of `n` vertices is in, named by its
    lowest vertex. Every edge hands both ends its lower name, and names follow
    each other to their end, until nothing moves."""
    label = np.arange(n)
    if not len(edges):
        return label
    edges = np.asarray(edges)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        low = np.minimum(label[a], label[b])
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        np.minimum.at(label, before[a], low)
        np.minimum.at(label, before[b], low)
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
        if np.array_equal(before, label):
            return label


def settle_pieces(labels, piece) -> np.ndarray:
    """A separate mesh piece (an unwelded petiole, a catkin) whose vertices
    mostly carry one stalk's label (<= -2) is that stalk's, whole -- a
    straight stalk line misses a bent petiole beside another one."""
    labels = np.asarray(labels).copy()
    if not len(labels):
        return labels
    order = np.argsort(piece, kind="stable")
    cuts = np.nonzero(np.diff(piece[order]))[0] + 1
    for group in np.split(order, cuts):
        values, counts = np.unique(labels[group], return_counts=True)
        k = int(np.argmax(counts))
        if values[k] <= -2:
            labels[group] = values[k]
    return labels


def vote_labels(labels, edges, iterations: int = 4):
    """Each vertex takes the label most of its mesh neighbours carry, when
    more than half agree on one -- `iterations` passes. Mends the stray
    vertices a nearest-centre-line assignment gets wrong where the drawn mesh
    sits off the model's lines (gnarl, a raceme hanging past a stem).
    Returns the new labels (an int array; the input is not edited)."""
    labels = np.asarray(labels).copy()
    if not len(edges):
        return labels
    order = np.concatenate([edges[:, 0], edges[:, 1]])
    other = np.concatenate([edges[:, 1], edges[:, 0]])
    sort = np.argsort(order, kind="stable")
    order, other = order[sort], other[sort]
    starts = np.searchsorted(order, np.arange(len(labels) + 1))
    degree = np.diff(starts)
    for _ in range(iterations):
        fresh = labels.copy()
        changed = 0
        # A majority against a vertex needs more than half its ring to differ.
        differ = np.bincount(order, labels[other] != labels[order], len(labels))
        for v in np.nonzero(differ * 2 > degree)[0].tolist():
            ring = labels[other[starts[v]:starts[v + 1]]]
            values, counts = np.unique(ring, return_counts=True)
            k = int(np.argmax(counts))
            if values[k] != labels[v] and counts[k] * 2 > len(ring):
                fresh[v] = values[k]
                changed += 1
        labels = fresh
        if not changed:
            break
    return labels


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def surface_handover(co, parent_points, parent_radius: float, reach: float):
    """How much of a branch's own weight each vertex takes, 0..1, by its
    distance from the PARENT's surface: 0 on it, rising (smoothstep) to 1
    `reach` out. A crotch fillet hugging the parent stays the parent's, the
    branch takes over as it pulls away -- however steeply it rises."""
    gap = _polyline_distance(co, parent_points) - parent_radius
    return _smooth(gap / max(reach, 1e-6))


def axis_weights(co, chain: AxisChain, bone_len: dict[str, float],
                 parent_bone: str | None, root_offset: float = 0.0):
    """Two-bone weights for one axis's tube vertices: (bone a, bone b, weight
    of b) per vertex -- b == a where no blend applies.

    A joint inside the chain blends 50/50 at the joint, each side handing
    over to the other. Where the axis leaves its parent the parent's own
    surface carries none of it, so that hand-over is one-sided: all parent
    out to `root_offset` (the parent's radius -- where the branch's tube
    leaves its surface), then the branch's own bone over the blend length.
    (The rig builds without a parent bone and hands a branch over by
    `surface_handover` instead.)"""
    names = chain.names
    pts = chain.points
    dist = np.full(len(co), np.inf)
    which = np.zeros(len(co), dtype=np.int64)
    along = np.zeros(len(co))
    for i in range(len(names)):
        d, s = _segment_project(co, pts[chain.breaks[i]], pts[chain.breaks[i + 1]])
        closer = d < dist
        dist = np.where(closer, d, dist)
        which = np.where(closer, i, which)
        along = np.where(closer, s, along)
    a = [names[i] for i in which]
    b = list(a)
    wb = np.zeros(len(co))
    for v in range(len(co)):
        i = int(which[v])
        length = bone_len[names[i]]
        s = float(along[v])
        prev = names[i - 1] if i else parent_bone
        nxt = names[i + 1] if i + 1 < len(names) else None
        if prev is not None:
            hw = BLEND * min(length, bone_len.get(prev, length))
            if i == 0:                       # off the parent: one-sided
                if hw > 0.0 and s < root_offset + hw:
                    b[v], wb[v] = prev, 1.0 - float(_smooth((s - root_offset) / hw))
                    continue
            elif hw > 0.0 and s < hw:
                b[v], wb[v] = prev, 0.5 - 0.5 * float(_smooth(s / hw))
                continue
        if nxt is not None:
            hw = BLEND * min(length, bone_len[nxt])
            if hw > 0.0 and length - s < hw:
                b[v], wb[v] = nxt, 0.5 - 0.5 * float(_smooth((length - s) / hw))
    return a, b, wb


def near_points(co, targets, reach) -> np.ndarray:
    """Which of `co` lie within `reach[v]` of any of `targets`. Each grid cell
    keeps only the targets that can be nearest to it (`_over_cells`), so the cost
    is not vertices x targets."""
    hit = np.zeros(len(co), dtype=bool)

    def mark(cell, near_sets):
        close = targets[near_sets[0]]
        for lo in range(0, len(cell), 2048):
            at = cell[lo:lo + 2048]
            pts = co[at]
            nearest = np.full(len(pts), np.inf)
            for s0 in range(0, len(close), 2048):
                d = np.linalg.norm(pts[:, None, :] - close[None, s0:s0 + 2048, :], axis=-1)
                nearest = np.minimum(nearest, d.min(axis=1))
            hit[at] = nearest <= reach[at]

    _over_cells(co, [len(targets)], [lambda c, i: np.linalg.norm(targets[i] - c, axis=1)], mark)
    return hit


def _weight_table(weights: list[dict]):
    """Per vertex a row of (bone number, weight) pairs: bones -1 where unused."""
    index: dict[str, int] = {}
    width = max((len(d) for d in weights), default=0)
    bones = np.full((len(weights), width), -1, dtype=np.int64)
    values = np.zeros((len(weights), width))
    for v, d in enumerate(weights):
        for k, (name, w) in enumerate(d.items()):
            bones[v, k] = index.setdefault(name, len(index))
            values[v, k] = w
    return bones, values, list(index)


def _seam_edges(bones, values, edges, threshold: float, chunk: int = 200_000):
    """Which edges' two ends differ by more than `threshold` (L1): |a - b| summed
    over the bones is sum(a) + sum(b) - 2 sum(min(a, b)) over the bones they share."""
    total = values.sum(axis=1)
    out = np.zeros(len(edges), dtype=bool)
    for lo in range(0, len(edges), chunk):
        a, b = edges[lo:lo + chunk, 0], edges[lo:lo + chunk, 1]
        same = (bones[a][:, :, None] == bones[b][:, None, :]) & (bones[a][:, :, None] >= 0)
        low = np.minimum(values[a][:, :, None], values[b][:, None, :])
        out[lo:lo + chunk] = total[a] + total[b] - 2.0 * (low * same).sum(axis=(1, 2)) > threshold
    return out


def _relax_pieces(piece, source, target, zone, bones, values, iterations, max_influences):
    """Jacobi on every connected piece of the zone at once, in one table: a row
    per (piece, vertex), a column per bone of that piece. A vertex feeding two
    pieces gets a row in each. `source` -> `target` are directed edges; only
    targets move. Returns (vertices, bones, weights) of the moved rows' kept
    bones, one flat triple per weight."""
    of_edge = piece[target]
    width = int(max(source.max(), target.max())) + 1
    pair_s, pair_t = of_edge * width + source, of_edge * width + target
    rows, inverse = np.unique(np.concatenate([pair_s, pair_t]), return_inverse=True)
    src, dst = inverse[:len(pair_s)], inverse[len(pair_s):]
    row_vert = rows % width
    _pieces, row_piece = np.unique(rows // width, return_inverse=True)   # 0, 1, 2...
    # Columns: each piece numbers its own bones, ascending.
    present = bones[row_vert]
    r, k = np.nonzero(present >= 0)
    bone_at = present[r, k]
    span = int(bone_at.max()) + 1
    keys = row_piece[r] * span + bone_at
    uniq, col_of = np.unique(keys, return_inverse=True)
    uniq_piece = uniq // span
    first = np.searchsorted(uniq_piece, uniq_piece, side="left")
    local = np.arange(len(uniq)) - first
    columns = int(local.max()) + 1
    grid = np.zeros((len(rows), columns))
    grid[r, local[col_of]] = values[row_vert[r], k]
    by_piece_col = np.full((int(row_piece.max()) + 1, columns), -1, dtype=np.int64)
    by_piece_col[uniq_piece, local] = uniq % span
    order = np.argsort(dst, kind="stable")
    src, dst = src[order], dst[order]
    heads = np.nonzero(np.r_[True, dst[1:] != dst[:-1]])[0]
    tails = dst[heads]
    scale = 1.0 / (np.diff(np.r_[heads, len(dst)]) + 1.0)[:, None]
    for _ in range(iterations):
        grid[tails] = (grid[tails] + np.add.reduceat(grid[src], heads, axis=0)) * scale
    keep = min(max_influences, columns)
    top = np.argsort(-grid[tails], axis=1, kind="stable")[:, :keep]
    w = np.take_along_axis(grid[tails], top, axis=1)
    total = w.sum(axis=1, keepdims=True)
    w = w / np.where(total > 0.0, total, 1.0)
    v = np.repeat(row_vert[tails], keep)
    b = by_piece_col[np.repeat(row_piece[tails], keep), top.ravel()]
    w = w.ravel()
    mask = (w > 1e-4) & (b >= 0)
    return v[mask], b[mask], w[mask], row_vert[tails]


def smooth_seams(co, edges, weights: list[dict], reach, *, threshold: float = 1.0,
                 iterations: int = 30, max_influences: int = 4) -> int:
    """Smooth the weights across seams -- where welded stems (or a stalk and
    its stem) meet with different bones on either side. A seam is a mesh
    edge whose two ends' weights differ by more than `threshold` (L1, 2 = no
    bone shared; a gentler step is a blend doing its job, and at 0.3 most of a
    thin stem counted, one piece 25,824 vertices x 696 bones); every
    vertex within `reach[v]` of a seam vertex is relaxed (Jacobi, the
    vertex and its neighbours averaged) while the rest stay fixed, then
    capped to `max_influences` and renormalised. `weights` is edited in
    place; returns the number of vertices relaxed. Each connected piece of the
    relaxed zone shares one table (`_relax_pieces`), so this is arrays, not dicts."""
    if not len(edges):
        return 0
    edges = np.asarray(edges, dtype=np.int64)
    bones, values, names = _weight_table(weights)
    seam = np.unique(edges[_seam_edges(bones, values, edges, threshold)])
    if not len(seam):
        return 0
    zone = near_points(co, co[seam], reach)
    touching = edges[zone[edges[:, 0]] | zone[edges[:, 1]]]
    if not len(touching):
        return 0
    # Moving vertices connect pieces; a fixed ring vertex only feeds them.
    inner = touching[zone[touching[:, 0]] & zone[touching[:, 1]]]
    piece = components(len(weights), inner)
    source = np.concatenate([touching[:, 0], touching[:, 1]])
    target = np.concatenate([touching[:, 1], touching[:, 0]])
    moved = zone[target]
    source, target = source[moved], target[moved]
    if not len(target):
        return 0
    # One table per size of piece: its width is its piece with the most bones,
    # so a joint of hundreds of bones must not widen every two-bone stalk's row.
    of_edge = piece[target]
    span = int(bones.max()) + 2
    keys = np.unique((of_edge[:, None] * span + bones[source]).ravel()[bones[source].ravel() >= 0])
    named, count = np.unique(keys // span, return_counts=True)
    size = np.ceil(np.log2(count[np.searchsorted(named, of_edge)] + 1)).astype(np.int64)
    relaxed = 0
    for tier in group_indices(size).values():
        v, b, w, moved = _relax_pieces(piece, source[tier], target[tier], zone, bones, values,
                                       iterations, max_influences)
        for vert in moved.tolist():
            weights[vert] = {}
        for vert, bone, weight in zip(v.tolist(), b.tolist(), w.tolist(), strict=True):
            weights[vert][names[bone]] = weight
        relaxed += len(moved)
    return relaxed
