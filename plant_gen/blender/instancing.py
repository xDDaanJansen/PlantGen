"""Components on the organ points: which component takes each point
(masks, Density, levels), the points placed along stems (Node /
Internode / Basal / Extra...), and how each instance is drawn -- instance
maps, Sky, Fold / Parent Roll, a young leaf's shape. Split out of
`bridge.py` (2026-10-13)."""

from __future__ import annotations

import bpy
import numpy as np

from ..growth import masks, traits
from ..growth.model import LeafForm
from . import bridge, curves, mapping, structure
from . import mesh as gn_mesh

_BUD_KIND = 2
_APEX_KIND = 3
_FRUIT_KIND = 4


def _filters_stack(component, seed):
    """A bpy-free `MaskStack` (the mask fold only -- `keep`=1) from a component's
    inline mask list. It decides *eligibility* (which points this component may
    take); the mix weight (Density) and Prune are applied separately, per point.
    Its Random channel draws from the component's own seed (`_cseed`)."""
    return masks.MaskStack(entries=bridge._mask_entries(component.filters), keep=1.0,
                           seed=int(_cseed(seed, component)))


def _pick_assets(seed: int, c, objs, ids, draw, channels) -> np.ndarray:
    """Which of `objs` (the component's collection) each point takes, from its
    seeded uniform `draw`: uniform, or -- with a Traits sheet that can change
    it -- weighted by the sheet (`growth/traits.py`). `channels()` gives the
    points' channel dict, read only when the sheet is in use. A neutral sheet
    keeps the uniform pick's own expression: byte-identical."""
    k = len(objs)
    sheet = _trait_sheet(c, objs)
    if sheet is None:
        return np.clip((draw * k).astype(np.int64), 0, k - 1)
    weights, prefs = sheet
    chan = channels()
    values = np.full((len(ids), len(c.trait_columns)), np.nan)
    for ci, col in enumerate(c.trait_columns):
        v = map_channel_values(bridge.channel_key(col), chan, ids,
                               _cseed(seed, c) * 7919 + (ci + 1) * 15485863)
        if v is not None:
            values[:, ci] = v
    return traits.pick(weights, prefs, values, draw)


def _trait_sheet(c, objs):
    """The component's Traits sheet over `objs`, in their order: (Weights,
    preferences), or None when it cannot change the uniform pick. A mesh with
    no row (added to the collection since) takes Weight 1, no preferences."""
    if not len(c.trait_rows):
        return None
    ncol = len(c.trait_columns)
    rows = {r.asset.name: r for r in c.trait_rows if r.asset is not None}
    weights = np.ones(len(objs))
    prefs = np.zeros((len(objs), ncol))
    for j, o in enumerate(objs):
        row = rows.get(o.name)
        if row is None:
            continue
        weights[j] = float(row.weight)
        for ci, cell in enumerate(list(row.cells)[:ncol]):
            prefs[j, ci] = float(cell.value)
    if traits.is_neutral(weights, prefs):
        return None
    return weights, prefs


def _cseed(seed: int, component) -> int:
    """The seed a component's own random choices use -- which asset of its
    collection a point gets, Prune, its size / angle variation, its masks'
    Random channel, where it scatters -- the plant's (master) seed, or one drawn
    from it and the component's own Seed when that is set, so reseeding a
    component reshuffles only it. Byte-identical at 0."""
    own = int(getattr(component, "seed", 0) or 0)
    if not own:
        return int(seed)
    return int(np.random.default_rng([int(seed), 7907, own]).integers(1 << 31))


# Which per-point attribute each filter channel reads. The organs carry every
# channel; the stem carries only the whole-plant ones (`height`, and `shade` once
# computed), so a structural/organ-only channel leaves the stem at identity (1.0,
# kept/green) in both the evaluator and the preview.
_CHANNEL_ATTR = {
    "height": "height_fraction",
    "branch_depth": "branch_depth",
    "branch_index": "branch_index",
    "on_branch": "on_branch",
    "node_index": "node_frac",
    "node_ordinal": "node_ordinal",
    "branch_ordinal": "branch_ordinal",
    "bloom": "bloom_start",
    "factor": "factor",
    "age": "age",
    "split_depth": "split_depth",
    "shade": "shade",
    "apex": "apex",
    "inflorescence": "on_inflorescence",
}


def _channels_from(attrs) -> dict:
    """The channel -> value-array dict the evaluator reads, from whichever channel
    attributes this domain actually carries (an absent channel is left out and the
    evaluator treats it as identity)."""
    out = {ch: np.asarray(attrs[key], dtype=np.float64)
           for ch, key in _CHANNEL_ATTR.items() if key in attrs}
    # The plant's custom attributes and classes, when `derived_channels` put
    # them on these points, pass through under their own keys.
    for key, values in attrs.items():
        if key.startswith((masks.CUSTOM_PREFIX, masks.CLASS_PREFIX)):
            out[key] = np.asarray(values, dtype=np.float64)
    return out


# The organ types routed through the component seam now. Flowers are excluded: a
# flower's instance index is its BLOOM STAGE, not a variety pick, so they need a
# different mechanism (their own slice). A fixed salt index per type decorrelates
# the seeded draws and is stable if flower is wired later; leaf is 0, so its salts
# are the old leaf-only ones -- a leaf-only plant stays byte-identical.
_WIRED_COMPONENT_TYPES = ("leaf", "bract", "bud", "fruit")
_TYPE_SALT_INDEX = {"leaf": 0, "bract": 1, "flower": 2, "bud": 3, "fruit": 4}
_U64 = 0xFFFFFFFFFFFFFFFF


def _type_point_mask(tkey, kind, form):
    """Boolean index into the organ arrays for one component type's points."""
    if tkey == "leaf":
        # Juvenile only for now -- adult (form 1) keeps its legacy GN pick until
        # the leaf-cleanup slice removes heteroblasty.
        return (kind == 0) & (form == int(LeafForm.JUVENILE))
    if tkey == "bract":
        return (kind == 0) & (form == int(LeafForm.BRACT))
    if tkey == "flower":
        return kind == 1
    if tkey == "bud":
        return (kind == _BUD_KIND) | (kind == _APEX_KIND)
    if tkey == "fruit":
        return kind == _FRUIT_KIND
    return np.zeros(len(kind), dtype=bool)


def _components_of_type(s, tkey):
    """A type's components that can instance -- right type, not muted, a collection
    with at least one object. List order sets the master collection's object order
    (the instance-index space); which component takes a point is by Density weight."""
    return [c for c in s.components
            if c.type == tkey and not c.mute
            and c.collection is not None and len(c.collection.objects) > 0]


def has_components(s, tkey) -> bool:
    """Whether the plant has any component of a type, muted or empty ones
    included: then the components own that type, and the plant's own legacy
    asset collection is never drawn from."""
    return any(c.type == tkey for c in s.components)


_RELINKED: set = set()
"""Plants (by name) whose master collection changed at this regrow: the
modifier must rebuild its depsgraph relations (`mesh.ensure_gn_modifier`)."""


def take_relinked(obj) -> bool:
    """Whether `obj`'s master collections changed since the last ask."""
    if obj.name in _RELINKED:
        _RELINKED.discard(obj.name)
        return True
    return False


def _master_key(tkey):
    return f"plantgen_{tkey}_master_name"


def _ensure_master(obj, tkey, comps):
    """The hidden master collection a type's components instance through: the union
    of their collections' objects, deduplicated, in list order. Not linked into any
    scene (a GN Collection Info input), kept alive by a fake user; rebuilt each
    regrow so it always matches the components."""
    key = _master_key(tkey)
    name = obj.get(key) or f"{obj.name}_{tkey.capitalize()}Master"
    master = bpy.data.collections.get(name)
    if master is not None and any(o.get(key) == name for o in bpy.data.objects
                                  if o is not obj):
        # **Another plant's master.** Found by name, it was shared: a plant made
        # under a name another plant had first ("Stem", renamed after), or a
        # duplicate carrying the original's key, rebuilt it with its own assets
        # and every plant on it drew whichever regrew last (18 trailer stems,
        # one flower). A plant always gets a master of its own.
        master = None
    if master is None:
        master = bpy.data.collections.new(name)
        master.use_fake_user = True
    want, seen = [], set()
    for c in comps:
        for o in c.collection.objects:
            if o.name not in seen:
                want.append(o)
                seen.add(o.name)
    # Relinked only when it changes. Unlinking and relinking the same asset
    # objects every regrow tagged them, and through them every plant in the
    # file instancing them re-evaluated: ~1 s after each regrow in a scene of
    # 28 plants (the 739k-polygon sweep plants among them), for nothing.
    if list(master.objects) != want:
        for o in list(master.objects):
            master.objects.unlink(o)
        for o in want:
            master.objects.link(o)
        _RELINKED.add(obj.name)
    if obj.get(key) != master.name:
        obj[key] = master.name
    return master


def master_collection(obj, tkey):
    """The hidden master collection for a type built at the last regrow, or None
    when the plant has no components of that type (modifier feeds the legacy one)."""
    name = obj.get(_master_key(tkey))
    return bpy.data.collections.get(name) if name else None


def _assign_type(obj, s, organ_attrs, tkey, kind, form, ids, idxs, scale, drop,
                 generated=None, owned=None) -> bool:
    """Route one type's points through its components. Each candidate point goes to
    ONE component chosen at random in proportion to the Density weights of the
    components eligible for it (masks keep it) -- so eligible components mix by
    weight, not first-come. A point no component wants renders nothing; the chosen
    component then deletes a Prune fraction of its own instances. Mutates `idxs`
    (into the type's master), `scale` (Size) and `drop`. Returns False -- the legacy
    single-collection path -- only when the plant has no components of this type
    at all: with every one muted (or empty) its points render nothing, rather
    than falling back to the plant's own asset collection.

    The claim is only among ORGAN-placement components; along-stems ones still join
    the master (so their assets share the index space) but own their generated
    points instead of claiming grown ones, and those points (`generated`) are held
    out of the claim here."""
    comps = _components_of_type(s, tkey)
    if not comps:
        obj.pop(_master_key(tkey), None)
        if not has_components(s, tkey):
            return False
        drop[np.nonzero(_type_point_mask(tkey, kind, form))[0]] = 1.0
        return True
    master = _ensure_master(obj, tkey, comps)
    global_of = {o.name: i for i, o in enumerate(master.objects)}
    seed = int(s.seed)
    ti = _TYPE_SALT_INDEX[tkey]                # ti == 0 (leaf) keeps the old salts

    comps = [c for c in comps if 'ORGAN' in c.placement]
    if not comps:
        # Only along-stems components of this type: nothing claims the grown organ
        # points, so they render nothing (their default instance_index would other-
        # wise read as master object 0). The along-stems points are baked separately.
        sel = _type_point_mask(tkey, kind, form)
        if generated is not None:
            sel = sel & ~generated
        drop[np.nonzero(sel)[0]] = 1.0
        return True
    pts = _type_point_mask(tkey, kind, form)
    if generated is not None:
        pts = pts & ~generated
    pidx = np.nonzero(pts)[0]
    if not len(pidx):
        return True
    pids = ids[pidx]
    pchan = {k: np.asarray(v)[pidx] for k, v in _channels_from(organ_attrs).items()}
    plevels = np.asarray(organ_attrs.get("level", np.zeros(len(ids))))[pidx]
    weighted = np.zeros((len(pidx), len(comps)), dtype=np.float64)
    for ci, c in enumerate(comps):
        stack = _filters_stack(c, seed)
        if stack.is_empty():
            keeps = np.ones(len(pidx), dtype=bool)   # unmasked -> eligible everywhere
        else:
            keeps = masks.evaluate(stack, pchan, pids)[0] > 0.5
        keeps &= structure.component_level_ok(s, c, plevels)   # only on its linked levels
        weighted[:, ci] = keeps * max(float(c.density), 0.0)
    total = weighted.sum(axis=1)
    has = total > 0.0
    draw = masks._hash01(pids, (seed * 2246822519 + 12345 + ti * 1013904223) & _U64) * total
    assigned = np.where(has, (np.cumsum(weighted, axis=1) > draw[:, None]).argmax(axis=1), -1)

    # A grown bract / fruit already carries its primary component's Size (it is
    # `inflorescence.bract_size` / `fruit_size` in growth), so each component of
    # the type scales relative to that -- the primary by exactly 1, never twice.
    bract_base = 1.0
    primary = (bridge._primary_bract_component(s) if tkey == "bract"
               else bridge._primary_fruit_component(s) if tkey == "fruit" else None)
    if primary is not None and primary.size > 0.0:
        bract_base = float(primary.size)
    for ci, c in enumerate(comps):
        take = pidx[assigned == ci]
        if not len(take):
            continue
        objs = list(c.collection.objects)
        osalt = (_cseed(seed, c) * 2654435761 + (ci + 1) * 40503 + ti * 2246822519) & _U64
        mine = assigned == ci
        local = _pick_assets(seed, c, objs, ids[take], masks._hash01(ids[take], osalt),
                             lambda mine=mine: {k: v[mine] for k, v in pchan.items()})
        idxs[take] = [global_of[objs[li].name] for li in local]
        if owned is not None:
            owned.append((c, take))
        scale[take] *= float(c.size) / bract_base
        var = float(c.gen_size_variance)
        if var > 0.0:
            # Size Variation on the grown points it claims too (a placed point
            # takes it in the generated bake): a seeded +/- fraction per point.
            zsalt = (_cseed(seed, c) * 2654435761 + (ci + 1) * 2246822519 + ti * 40503
                     + 17) & _U64
            scale[take] *= 1.0 + (masks._hash01(ids[take], zsalt) * 2.0 - 1.0) * var
        if c.prune > 0.0:                            # delete a random fraction
            psalt = (_cseed(seed, c) * 40499 + (ci + 1) * 2971215073 + ti * 1597334677) & _U64
            drop[take[masks._hash01(ids[take], psalt) < float(c.prune)]] = 1.0
    drop[pidx[~has]] = 1.0                            # a point no component wanted
    return True


def _assign_flowers(obj, s, organ_attrs, kind, ids, block_offset, block_max, scale, drop,
                    owned=None) -> bool:
    """Route flower points through flower components. The claim is `_assign_type`'s
    -- one component per point by mask + Density weight -- but a flower's index is
    its BLOOM STAGE, not a variety, so instead of an instance_index it bakes the
    point's BLOCK in the flower master: `block_offset` (its component's first frame)
    and `block_max` (its frame count - 1, the form `_stage_indices` wants). The GN
    adds the openness-driven stage within that block. Returns False (legacy single
    collection) only when the plant has no flower component at all.

    Each flower component's collection is one bloom sequence and becomes a
    contiguous block in the master, in list order -- so components must not share
    objects (a collection cannot hold one twice, which would break a block)."""
    comps = _components_of_type(s, "flower")
    if not comps:
        obj.pop(_master_key("flower"), None)
        if not has_components(s, "flower"):
            return False
        drop[np.nonzero(kind == 1)[0]] = 1.0   # all muted / empty: none drawn
        return True
    master = _ensure_master(obj, "flower", comps)
    global_of = {o.name: i for i, o in enumerate(master.objects)}
    seed = int(s.seed)
    ti = _TYPE_SALT_INDEX["flower"]

    pidx = np.nonzero(kind == 1)[0]
    if not len(pidx):
        return True
    pids = ids[pidx]
    pchan = {k: np.asarray(v)[pidx] for k, v in _channels_from(organ_attrs).items()}
    plevels = np.asarray(organ_attrs.get("level", np.zeros(len(ids))))[pidx]
    pinfl = np.asarray(organ_attrs.get("inflorescence", np.zeros(len(ids))))[pidx]
    weighted = np.zeros((len(pidx), len(comps)), dtype=np.float64)
    for ci, c in enumerate(comps):
        stack = _filters_stack(c, seed)
        keeps = (np.ones(len(pidx), dtype=bool) if stack.is_empty()
                 else masks.evaluate(stack, pchan, pids)[0] > 0.5)
        keeps &= structure.component_level_ok(s, c, plevels)   # only on its linked levels
        keeps &= structure.component_inflorescence_ok(s, c, pinfl)   # its own inflorescence's
        weighted[:, ci] = keeps * max(float(c.density), 0.0)
    total = weighted.sum(axis=1)
    has = total > 0.0
    draw = masks._hash01(pids, (seed * 2246822519 + 12345 + ti * 1013904223) & _U64) * total
    assigned = np.where(has, (np.cumsum(weighted, axis=1) > draw[:, None]).argmax(axis=1), -1)

    for ci, c in enumerate(comps):
        take = pidx[assigned == ci]
        if not len(take):
            continue
        objs = list(c.collection.objects)
        block_offset[take] = global_of[objs[0].name]
        if owned is not None:
            owned.append((c, take))
        block_max[take] = len(objs) - 1
        # No `scale *= size` here: a flower's size is applied in growth via
        # profile.flower_size (= the component Size), so multiplying again would
        # double it. Size is the one flower-size control.
        if c.prune > 0.0:
            psalt = (_cseed(seed, c) * 40499 + (ci + 1) * 2971215073 + ti * 1597334677) & _U64
            drop[take[masks._hash01(ids[take], psalt) < float(c.prune)]] = 1.0
    drop[pidx[~has]] = 1.0
    return True


# Generated placement modes (Phase 3): kind + form the point takes for a
# component of each type. Flower is excluded -- its index is a bloom stage, not a
# variety, so it has no meaning as a scattered point.
_INTERNODE_KIND_FORM = {
    "leaf": (0, int(LeafForm.JUVENILE)),
    "bract": (0, int(LeafForm.BRACT)),
    "bud": (_BUD_KIND, 0),
    "fruit": (_FRUIT_KIND, 0),
}


# The generated placement modes (all but ORGAN) and the subset that arranges by
# phyllotaxis (a whorl of instances at each anchor, the spiral advancing between
# anchors). EXTRA is random scatter, so it takes no phyllotaxis.
_GENERATED_MODES = ('NODE', 'INTERNODE', 'TIP', 'INTERSECTION', 'EXTRA', 'BASAL')
_STRUCTURED_MODES = ('NODE', 'INTERNODE', 'TIP', 'INTERSECTION', 'BASAL')


def _generated_placement_components(s):
    """Components that generate their own points -- any generated mode in the
    placement set, not muted, a non-flower type with a collection."""
    gm = set(_GENERATED_MODES)
    return [c for c in s.components
            if not c.mute and c.type in _INTERNODE_KIND_FORM
            and c.collection is not None and len(c.collection.objects) > 0
            and (gm & set(c.placement))]


def _placement_frames(plant, hidden_axes=frozenset()):
    """Per axis, its non-split phytomers as frames, as `(axis, [frame, ...])` where a
    frame is `(struct, birth_age, base, tip, tangent, ref, binormal, radius, arc)`. The
    axis grouping is what lets Tip take the last phytomer and Junction the base of a
    branch; `struct` carries the axis structural-channel values so a generated point
    masks the same as a grown organ on that axis.

    A frame spans its whole internode, arc points included: a bent internode is
    split into arc points (`physics.smooth_bent_arcs`, `is_split`) that bear no
    organ, so they make no frame of their own -- but `base` is where the
    internode starts (its first arc point's base) and `arc` its centre line,
    base -> each arc point -> tip. Taking `base` from the last part instead put a
    stem's foot part-way up its first internode whenever that bent enough to
    split (a Basal rosette floated 5 cm up the stem below Stiffness ~0.75), a
    branch's Junction likewise, and squeezed Internode points onto the last
    part."""
    axes = plant.axes
    max_depth = max((max(a.structural_depth, 0) for a in axes.values()), default=0)
    depth_norm = float(max_depth) if max_depth > 0 else 1.0
    max_order = max((a.order for a in axes.values()), default=0)
    order_norm = float(max_order) if max_order > 0 else 1.0
    rank = {aid: i for i, aid in enumerate(sorted(axes))}
    rank_norm = float(len(rank) - 1) if len(rank) > 1 else 1.0
    total_steps = max(plant.step, 1)
    infl_axes = bridge._inflorescence_axes(plant)
    groups = []
    for aid, axis in axes.items():
        if aid in hidden_axes:
            continue  # a muted flower component's inflorescence axes
        split = max(axis.structural_depth, 0) / depth_norm
        branch_depth = max(axis.order, 0) / order_norm
        branch_index = rank[aid] / rank_norm
        on_branch = 1.0 if axis.order >= 1 else 0.0
        node_norm = float(len(axis.phytomers) - 1) if len(axis.phytomers) > 1 else 1.0
        frames = []
        arc = []
        for ph in axis.phytomers:
            if not arc:
                arc.append(np.asarray(ph.base_position, dtype=np.float64))
            arc.append(np.asarray(ph.tip_position, dtype=np.float64))
            if ph.is_split:
                continue                      # an arc point: part of the next frame
            base = arc[0]
            tip = arc[-1]
            span, arc = np.array(arc), []
            d = np.asarray(ph.direction, dtype=np.float64)
            ref = np.asarray(ph.reference, dtype=np.float64)
            struct = (float(ph.factor), split, branch_depth, branch_index,
                      on_branch, ph.index / node_norm, float(ph.index), float(rank[aid]),
                      bridge._on_inflorescence(axis, ph.index, infl_axes))
            birth_age = min(1.0, max(0.0, ph.birth_step / total_steps))
            frames.append((struct, birth_age, base, tip, d, ref, np.cross(d, ref),
                           ph.radius if ph.radius > 0.0 else 0.001, span))
        if frames:
            groups.append((axis, frames))
    return groups


def _along(arc, f: float, lengths=None) -> np.ndarray:
    """The point a fraction `f` (0..1) of the way along a frame's centre line
    `arc` (base -> arc points -> tip), by length -- on the bent internode, not
    its chord. An unsplit internode is its straight base -> tip. `lengths` is
    the arc's `_arc_lengths`, passed when many points share one arc."""
    seg, total, cum = lengths if lengths is not None else _arc_lengths(arc)
    if total <= 0.0:
        return arc[0].copy()
    at = min(max(float(f), 0.0), 1.0) * total
    k = int(np.searchsorted(cum, at, side="left"))
    k = min(k, len(seg) - 1)
    before = float(seg[:k].sum())
    t = (at - before) / float(seg[k]) if seg[k] > 0.0 else 0.0
    return arc[k] + t * (arc[k + 1] - arc[k])


def _arc_lengths(arc):
    """An arc's segment lengths, their total and running sum, for `_along`."""
    seg = np.linalg.norm(np.diff(arc, axis=0), axis=1)
    return seg, float(seg.sum()), np.cumsum(seg)


def _mode_anchors(groups, mode, internode_count):
    """The anchor points for one structured mode, each `(struct, birth, pos, tangent,
    ref, binormal, radius)`. NODE = every node (a phytomer's distal end); INTERNODE =
    `internode_count` points spaced evenly up each internode; TERMINAL = each axis's
    last node; INTERSECTION = the base of every branch (order >= 1). A whorl is placed
    at each. Each carries its place last, `(axis id, node / rosette index, sub,
    spiral index)` -- its spiral turn counted along its own stem, so a node added
    elsewhere moves nothing. (To thin -- every other node etc. -- add a mask with a
    modulo on the Node or Branch index; it applies here just as it does to grown
    organs.)

    NODE excludes each axis's terminal phytomer -- its tip node belongs to the
    Terminal mode, so the two never double up on the growing tip. INTERNODE
    takes it: its points sit partway up the internode, never on the tip, and
    without it every axis ended in a bare internode (a monkey puzzle's leader
    14 cm bare)."""
    out = []
    n = int(internode_count)
    for axis, frames in groups:
        if mode == 'TIP':
            struct, birth, _b, tip, d, ref, bi, r, _arc = frames[-1]
            out.append((struct, birth, tip, d, ref, bi, r,
                        (axis.id, 0, 0, int(struct[_NODE_INDEX]))))
            continue
        if mode == 'INTERSECTION':
            if axis.order >= 1:
                struct, birth, base, _t, d, ref, bi, r, _arc = frames[0]
                out.append((struct, birth, base, d, ref, bi, r, (axis.id, 0, 0, 0)))
            continue
        if mode == 'BASAL':
            # A rosette at the stem's foot: `internode_count` (the Basal Count)
            # anchors at its base, the whorl / spiral turning between them --
            # the compressed basal internodes a rosette is, laid flat.
            if axis.parent_axis_id < 0:
                struct, birth, base, _t, d, ref, bi, r, _arc = frames[0]
                out.extend((struct, birth, base, d, ref, bi, r, (axis.id, k, 0, k))
                           for k in range(n))
            continue
        if mode == 'NODE':
            for struct, birth, _base, tip, d, ref, bi, r, _arc in frames[:-1]:
                node = int(struct[_NODE_INDEX])
                out.append((struct, birth, tip, d, ref, bi, r, (axis.id, node, 0, node)))
            continue
        # INTERNODE: each point reads where it sits along its axis -- its factor
        # and node index run from the node below to its own, so a map along the
        # axis (smaller leaves toward a tip) is smooth, not a step per node.
        for k, (struct, birth, _base, _tip, d, ref, bi, r, arc) in enumerate(frames):
            node = int(struct[_NODE_INDEX])
            below = _channels_below(frames, k)
            lengths = _arc_lengths(arc)
            for i in range(n):
                t = (i + 0.5) / n
                at = list(struct)
                for ch in _ALONG_CHANNELS:
                    at[ch] = max(0.0, below[ch] + (struct[ch] - below[ch]) * t)
                pos = _along(arc, t, lengths)
                out.append((tuple(at), birth, pos, d, ref, bi, r,
                            (axis.id, node, i, node * n + i)))
    return out


_NODE_INDEX = 6
_ALONG_CHANNELS = (0, 5)
"""`factor` and `node_frac` in a frame's struct: the channels that run along an
axis, interpolated over an internode for the points placed on it."""


def _channels_below(frames, k: int) -> tuple:
    """A frame's struct at the node below it: the previous frame's, or -- the
    first internode, from the axis's foot -- its own stepped back by the next
    one's increment."""
    if k > 0:
        return frames[k - 1][0]
    here = list(frames[k][0])
    if len(frames) > 1:
        nxt = frames[k + 1][0]
        for ch in _ALONG_CHANNELS:
            here[ch] = here[ch] - (nxt[ch] - here[ch])
    return tuple(here)

_MODE_CODES = {'NODE': 1, 'INTERNODE': 2, 'TIP': 3, 'INTERSECTION': 4, 'BASAL': 5,
               'EXTRA': 6}


def _placed_key(comp_index: int, mode: str, axis_id: int, node: int, slot: int) -> int:
    """A placed point's variation key -- what its asset pick, angle / fold /
    size jitter, Prune and Random masks hash -- from its PLACE (component,
    mode, stem, node, slot), so it keeps its look while the plant grows. (Its
    `organ_id` stays a running count: the rig sizes arrays by it.) Above every
    organ id, so a key never repeats a grown organ's."""
    return ((1 << 40) | ((comp_index & 0xFF) << 32) | (_MODE_CODES[mode] << 29)
            | ((axis_id & 0x3FFF) << 15) | ((node & 0x3FF) << 5) | (slot & 0x1F))


_UP = np.array([0.0, 0.0, 1.0])
_SIDE_LEAN = (np.sin(np.radians(5.0)), np.sin(np.radians(20.0)))


def _cross(a, b) -> np.ndarray:
    """`np.cross` of two 3-vectors, the same float64 products and differences
    (byte-identical) without its general-axis overhead: ~50 us a call there,
    run three times for each of a clubmoss's 80k placed leaves."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0))


def _azimuth_zero(d, ref, bi, first: float) -> float:
    """The turn that puts a generated whorl's first member (`first` from its
    whorl's base, `_whorl_azimuths`) around its stem, as an angle from the
    node's reference `ref` (toward `bi`): on a leaning branch, at its
    horizontal side -- so the first pair of an opposite / decussate arrangement
    sits left and right of the branch, not on its top and bottom (user decision
    2026-10-03). The reference alone cannot say that: it is carried down each
    branch from its parent's phyllotaxis turn, so it points up on one branch and
    sideways on the next. A near-vertical stem has no left or right and keeps
    its reference; between 5 and 20 deg of lean the two blend, so nothing
    jumps. The position and the facing turn together; Parent Roll still turns
    the facing from there."""
    up_perp = _UP - d * float(d @ _UP)
    lean = float(np.linalg.norm(up_perp))
    lo, hi = _SIDE_LEAN
    t = min(max((lean - lo) / (hi - lo), 0.0), 1.0)
    w = t * t * (3.0 - 2.0 * t)
    if w <= 0.0:
        return 0.0
    side = _cross(_UP, d)                  # horizontal, square to the branch
    turn = float(np.arctan2(side @ bi, side @ ref)) - first
    return w * float(np.arctan2(np.sin(turn), np.cos(turn)))     # the short way round


def _rosette_zero(d, foot, ref, bi, count: int, divergence: float):
    """The turn that centres a basal rosette's fan on the way its stem faces
    out of the clump (user decision 2026-10-18): the stem's own lean (its
    splay), or for an upright stem the line from the clump's centre to its
    foot. So a narrow spiral -- 30 deg x 6 leaves -- fans out of the clump on
    every stem, never into it. As an angle from the node's reference `ref`
    (toward `bi`), minus half the fan (`count` leaves `divergence` apart).
    A crown stem's node leaves use it too, with `count` 1 (2026-10-21).
    The lean only counts once it is clear: under 5 deg the line from the
    centre decides, over 20 deg the lean, blended between (`_SIDE_LEAN`) --
    a near-upright stem's lean of a degree points anywhere.
    None for a lone stem at the centre standing straight up: it has no
    outward, and keeps `_azimuth_zero`."""
    lean_v = d - _UP * float(d @ _UP)
    lean = float(np.linalg.norm(lean_v))
    centre = np.array([float(foot[0]), float(foot[1]), 0.0])
    cn = float(np.linalg.norm(centre))
    lo, hi = _SIDE_LEAN
    t = min(max((lean - lo) / (hi - lo), 0.0), 1.0)
    w = t * t * (3.0 - 2.0 * t)
    out = (w * lean_v / lean if lean > 1e-9 else 0.0 * lean_v) + (
        (1.0 - w) * centre / cn if cn > 1e-9 else 0.0 * centre)
    if float(np.linalg.norm(out)) < 1e-6:
        return None
    return float(np.arctan2(out @ bi, out @ ref)) - 0.5 * (count - 1) * divergence


def _whorl_azimuths(anchor_index, divergence, whorl_count, whorl_spacing):
    """The azimuths (radians) of one anchor's whorl: the spiral base
    `anchor_index * divergence`, plus `whorl_count` members centred on it, spaced by
    `whorl_spacing` (0 -> even 360/whorl). Whorl 1 is a single spiralling point."""
    base = anchor_index * divergence
    if whorl_count <= 1:
        return (base,)
    spacing = whorl_spacing if whorl_spacing > 0.0 else (2.0 * np.pi / whorl_count)
    centre = (whorl_count - 1) / 2.0
    return tuple(base + (j - centre) * spacing for j in range(whorl_count))


def _generated_orientation(radial, d, pitch, roll_self, roll_parent, fold=0.0, sky=0.0):
    """The facing of a generated instance, from `radial` (straight out from the stem)
    and the tangent `d`. In order: `roll_parent` turns it around the stem axis
    (every instance the same way round) and `fold` raises it up or down around
    that axis (mirrored, below) -- where it points out from; `sky` turns its face
    toward the sky (`_sky_turn`) -- the face it starts from; then about the
    instance's **own** axes: `pitch` tips it toward its face (about its width:
    with the face along the stem that is toward the stem's tip, with Sky on it
    lifts the tip toward the sky) and `roll_self` spins it about its length.
    So Sky only sets where the face starts, and Pitch and Roll Self still do
    exactly one thing each on top of it.
    Returns (direction, normal), the normal kept perpendicular to direction so the
    instance frame is square. All-zero -> direction is `radial` unchanged.

    The normal is a grown organ's (`growth.model.organ_normal`): the stem's
    direction squared up against the instance's, so a leaf blade faces along its
    stem and lies held out flat -- at Pitch 90 (pointing up the stem) its face
    turns to the stem. It was `cross(d, radial)`, which is that frame turned 90
    deg about the blade's length: every generated leaf stood on edge like a fin
    and needed Roll Self 90 to lie flat (the bug growth's leaves once had).

    On a leaning branch Fold and Roll Self are **mirrored across it**: a
    positive Fold raises the instance toward the sky and a positive Roll Self
    turns its face up, on either side -- so the two leaves of an opposite pair
    lift and roll together (SpeedTree's Fold / side-aware Face). Parent Roll is
    not: it turns the whole set around the branch the one way. A near-vertical
    stem has no up side and folds every instance the one way round. Which side
    an instance is on is read from where it sits (`radial`), not from its
    current face, so the mirroring holds whatever Sky and Pitch did to it."""
    mirrored = float(np.linalg.norm(_UP - d * float(d @ _UP))) >= _SIDE_LEAN[0]
    lateral = _cross(d, radial)
    side = -1.0 if mirrored and float(lateral @ _UP) < 0.0 else 1.0   # + Fold raises it
    turn = roll_parent + side * fold
    direction = np.cos(turn) * radial + np.sin(turn) * lateral        # out, around the tangent
    normal = np.array(d, dtype=np.float64)                             # face along the stem
    if sky > 0.0:
        normal = _sky_turn(direction, normal, sky)                    # face toward the sky
    if pitch != 0.0:                                                  # tip toward its face
        direction, normal = (np.cos(pitch) * direction + np.sin(pitch) * normal,
                             np.cos(pitch) * normal - np.sin(pitch) * direction)
    if roll_self != 0.0:                                              # spin about its length
        theta = -side * roll_self if mirrored else roll_self          # + turns the face up
        normal = np.cos(theta) * normal + np.sin(theta) * _cross(direction, normal)
    return direction, normal


def _sky_turn(direction, normal, sky: float):
    """`normal` turned about `direction` `sky` of the way to the face closest to
    world up it can have -- up with the direction's own part removed. A
    direction straight up or down has no such face: unchanged."""
    target = _UP - direction * float(direction @ _UP)
    reach = float(np.linalg.norm(target))
    if reach < 1e-6:
        return normal
    target = target / reach
    phi = float(np.arctan2(_cross(normal, target) @ direction, normal @ target))
    phi *= min(float(sky), 1.0)
    return np.cos(phi) * normal + np.sin(phi) * _cross(direction, normal)


def _append_generated(positions, organ_attrs, kind_code, form_code, oid,
                      pos, direction, normal, struct, birth_age, scale=1.0, vkey=None):
    """Append one generated candidate point, filling every organ attribute so the
    arrays stay square (`struct` is the `_STRUCT_CHANNELS` tuple for its axis;
    `scale` its expansion so far)."""
    positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
    organ_attrs["kind"].append(kind_code)
    organ_attrs["organ_id"].append(oid)
    organ_attrs["vkey"].append(int(oid if vkey is None else vkey))
    organ_attrs["direction"].append((float(direction[0]), float(direction[1]), float(direction[2])))
    organ_attrs["normal"].append((float(normal[0]), float(normal[1]), float(normal[2])))
    organ_attrs["scale"].append(float(scale))
    organ_attrs["bloom_start"].append(0.0)
    organ_attrs["bloom_birth"].append(0.0)
    organ_attrs["form"].append(form_code)
    organ_attrs["dead"].append(0.0)
    organ_attrs["autumn"].append(0.0)
    for name, val in zip(bridge._STRUCT_CHANNELS, struct, strict=True):
        organ_attrs[name].append(float(val))
    organ_attrs["age"].append(birth_age)
    organ_attrs["apex"].append(0.0)
    organ_attrs["unfolded"].append(min(1.0, float(scale)))   # its expansion so far


def _gen_angles(oid, base, var, vsalt):
    """The three base angles (pitch, roll_self, roll_parent) with a seeded
    per-instance deviation added -- one hash draw per instance, the axes
    decorrelated by xored keys, each mapped to +/- its own variance in radians.
    No variance -> the base angles unchanged (byte-identical to before)."""
    if not any(var):
        return base
    keys = np.array([oid, oid ^ 0x85EBCA6B, oid ^ 0xC2B2AE35], dtype=np.int64)
    h = masks._hash01(keys, vsalt) * 2.0 - 1.0
    return tuple(b + hi * vi for b, hi, vi in zip(base, h, var, strict=True))


def _generate_placement(obj, s, plant, positions, organ_attrs, hidden_axes=frozenset(),
                        profile=None):
    """Append generated candidate points for every generated-placement component
    (Phase 3) and return `[(component, point_indices)]`. Geometry only -- the
    instance_index / scale / prune bake happens later in `_assign_components`, owned
    by the generating component (these points bypass the shared weighted claim).

    The structured modes (Node / Internode / Tip / Junction) put a whorl of
    `gen_whorl_count` instances at each anchor, spaced by `gen_whorl_angle` (0 = even),
    with the spiral advancing `gen_divergence` between anchors -- the same
    arrangement growth uses, driven here off the component's controls. EXTRA scatters
    `extra_count` points at random over all stems, internode-length-weighted, seeded.

    **Each generated leaf grows in like a grown one**: it forms with its node --
    a Basal rosette's leaves one plastochron apart, in their spiral order, the
    first with the stem's foot -- is left out until then, and inflates along
    the profile's Leaf Expansion from there. (They used to be drawn full-size
    from the moment their node existed, the whole rosette at once.)"""
    comps = _generated_placement_components(s)
    if not comps:
        return []
    groups = _placement_frames(plant, hidden_axes)
    if not groups:
        return []
    flat = [fr for _axis, frames in groups for fr in frames]
    lengths = np.array([float(np.linalg.norm(np.diff(fr[8], axis=0), axis=1).sum())
                        for fr in flat], dtype=np.float64)
    # Foliage leaves sit on the leafy stem: an inflorescence's nodes carry
    # flowers and bracts (the Bract component's), so a leaf component skips them
    # -- a raceme on every branch otherwise grew a leaf pair at each flower.
    infl = bridge._STRUCT_CHANNELS.index("on_inflorescence")
    depth = bridge._STRUCT_CHANNELS.index("branch_depth")
    leafy = np.array([fr[0][infl] < 0.5 for fr in flat], dtype=bool)
    seed = int(s.seed)
    next_id = max((int(v) for v in organ_attrs["organ_id"]), default=-1) + 1

    now = float(plant.step if plant.continuous_step < 0.0 else plant.continuous_step)
    span = float(max(plant.step, 1))             # `_placement_frames`' birth_age scale
    expand = (profile.leaf_expansion.over(profile.leaf_expansion_steps)
              if profile is not None else None)
    plastochron = float(max(1, profile.plastochron)) if profile is not None else 1.0

    def grown(birth_age: float, later: float = 0.0):
        """(born yet, scale): a leaf formed `later` steps after its node."""
        if expand is None:
            return True, 1.0
        age = now - (birth_age * span + later)
        if age < -1e-9:
            return False, 0.0
        return True, float(expand(max(age, 0.0)))

    specs = []
    level_of = None
    for gk, comp in enumerate(comps):
        kind_code, form_code = _INTERNODE_KIND_FORM[comp.type]
        skip_infl = comp.type == 'leaf'
        pitch = float(mapping.cval(comp, "gen_pitch"))
        roll_self = float(mapping.cval(comp, "gen_roll_self"))
        roll_parent = float(mapping.cval(comp, "gen_roll_parent"))
        fold = float(mapping.cval(comp, "gen_fold"))
        sky = float(mapping.cval(comp, "sky_roll")) if comp.type in ("leaf", "bract") else 0.0
        divergence = float(mapping.cval(comp, "gen_divergence"))
        whorl = max(1, int(mapping.cval(comp, "gen_whorl_count")))
        whorl_angle = float(mapping.cval(comp, "gen_whorl_angle"))
        icount = max(1, int(mapping.cval(comp, "internode_count")))
        # Per-instance orientation scatter (degrees on the panel -> radians here).
        base_angles = (pitch, roll_self, roll_parent)
        angle_var = (np.deg2rad(float(comp.gen_pitch_variance)),
                     np.deg2rad(float(comp.gen_roll_self_variance)),
                     np.deg2rad(float(comp.gen_roll_parent_variance)))
        fold_var = np.deg2rad(float(comp.gen_fold_variance))
        vsalt = (_cseed(seed, comp) * 2654435761 + (gk + 1) * 1013904223) & _U64
        fsalt = (vsalt * 6364136223846793005 + 1442695040888963407) & _U64
        start_n = len(organ_attrs["organ_id"])
        # Each point's facing inputs, kept for its Fold / Parent Roll maps
        # (`_reorient_placed`) -- only on a component that has one.
        frames = [] if _has_imap(comp, ("FOLD", "ROLL_PARENT")) else None

        def fold_of(oid, fold=fold, fold_var=fold_var, fsalt=fsalt):
            if not fold_var:
                return fold
            h = float(masks._hash01(np.array([oid], dtype=np.int64), fsalt)[0])
            return fold + (2.0 * h - 1.0) * fold_var

        # Only the axes on the levels it is placed on: the rest would be built
        # here and dropped in `_assign_components` (two thirds of a monkey
        # puzzle's 71k leaf points). Every point keys its look on its place,
        # not its index, so the ones kept come out the same.
        on_levels = groups
        if not any(q.comp_uid == comp.uid and q.level_uid == 0 for q in s.placements):
            if level_of is None:
                level_of = structure._axis_levels(plant, len(s.branch_levels))
            allowed = set(structure.component_level_indices(s, comp))
            on_levels = [(axis, fr) for axis, fr in groups
                         if level_of.get(axis.id, 0) in allowed]
        for mode in _STRUCTURED_MODES:
            if mode not in comp.placement:
                continue
            basal_n = max(1, int(mapping.cval(comp, "basal_count")))
            anchors = _mode_anchors(on_levels, mode, basal_n if mode == 'BASAL' else icount)
            first = _whorl_azimuths(0, 0.0, whorl, whorl_angle)[0]
            # An internode's points share its frame (the same d / ref / bi
            # arrays): its side turn is worked out once, not once a point.
            side_zero = (None, None, None, 0.0)
            for struct, birth_age, pos0, d, ref, bi, radius, place in anchors:
                if skip_infl and struct[infl] >= 0.5:
                    continue
                axis_id, node, sub, spiral = place
                # A rosette's leaves form one after another round the spiral,
                # one plastochron apart; every other anchor forms with its node.
                later = node * plastochron if mode == 'BASAL' else 0.0
                born, size = grown(birth_age, later)
                if not born:
                    continue
                if mode == 'BASAL':
                    zero = _rosette_zero(d, pos0, ref, bi, basal_n, divergence)
                elif struct[depth] <= 0.0:
                    # A crown stem (a culm, a tiller): its leaves line up with the way
                    # it faces out of the clump, as its basal rosette does -- one
                    # rank of a two-ranked stem leans out, the other back over it
                    # (user decision 2026-10-21). Branches keep `_azimuth_zero`.
                    zero = _rosette_zero(d, pos0, ref, bi, 1, divergence)
                    if zero is not None:
                        zero -= first
                else:
                    zero = None
                if zero is None:
                    if not (side_zero[0] is d and side_zero[1] is ref and side_zero[2] is bi):
                        side_zero = (d, ref, bi, _azimuth_zero(d, ref, bi, first))
                    zero = side_zero[3]
                for j, az in enumerate(_whorl_azimuths(spiral, divergence, whorl,
                                                       whorl_angle)):
                    key = _placed_key(gk, mode, axis_id, node, sub * whorl + j)
                    az += zero
                    radial = np.cos(az) * ref + np.sin(az) * bi
                    pos = pos0 + radial * radius
                    p, rs, rp = _gen_angles(key, base_angles, angle_var, vsalt)
                    direction, normal = _generated_orientation(radial, d, p, rs, rp,
                                                               fold_of(key), sky)
                    if frames is not None:
                        frames.append((radial, d, p, rs, rp, fold_of(key), sky))
                    _append_generated(positions, organ_attrs, kind_code, form_code,
                                      next_id, pos, direction, normal, struct,
                                      birth_age + later / span, size, vkey=key)
                    next_id += 1

        if 'EXTRA' in comp.placement:
            total = int(round(max(0.0, float(mapping.cval(comp, "extra_count")))))
            w = lengths * leafy if skip_infl else lengths
            weights = w / w.sum() if w.sum() > 0 else None
            if total > 0 and weights is not None:
                rng = np.random.default_rng(
                    (_cseed(seed, comp) * 6364136223846793005 + gk + 1) & _U64)
                pick = rng.choice(len(flat), size=total, p=weights)
                fs = rng.random(total)
                thetas = rng.random(total) * (2.0 * np.pi)
                for j in range(total):
                    struct, birth_age, base, tip, d, ref, bi, radius, arc = flat[pick[j]]
                    _born, size = grown(birth_age)
                    radial = np.cos(thetas[j]) * ref + np.sin(thetas[j]) * bi
                    pos = _along(arc, fs[j]) + radial * radius
                    key = _placed_key(gk, 'EXTRA', 0, 0, 0) | j
                    p, rs, rp = _gen_angles(key, base_angles, angle_var, vsalt)
                    direction, normal = _generated_orientation(radial, d, p, rs, rp,
                                                               fold_of(key), sky)
                    if frames is not None:
                        frames.append((radial, d, p, rs, rp, fold_of(key), sky))
                    _append_generated(positions, organ_attrs, kind_code, form_code,
                                      next_id, pos, direction, normal, struct, birth_age,
                                      size, vkey=key)
                    next_id += 1

        n_after = len(organ_attrs["organ_id"])
        specs.append((comp, np.arange(start_n, n_after, dtype=np.int64), frames))
    return specs


def map_channel_values(channel, chan: dict, ids, salt: int):
    """One channel's 0-1 values for a set of points, as a map reads them: the
    baked channel arrays (`_channels_from`), `random` as seeded per-point noise
    (its own salt per map so two random maps do not move together). None when
    the channel is not on these points (a map on it is then a no-op)."""
    if channel == "random":
        return masks._hash01(np.asarray(ids), salt & _U64)
    values = chan.get(channel)
    return None if values is None else np.asarray(values, dtype=np.float64)


def _rotate(v, k, theta):
    """Rodrigues: rotate row vectors `v` about unit row axes `k` by `theta` rad."""
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    return v * c + np.cross(k, v) * s + k * np.sum(k * v, axis=1, keepdims=True) * (1.0 - c)


def _unit_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


def instance_map_values(obj, c, target: str, chan: dict, ids):
    """Component `c`'s instance maps on `target` over some points (their
    channels `chan`, variation keys `ids`): per point a multiplier, or -- a
    rotation target (`plant_maps.ROTATION_TARGETS`) -- an offset in radians, or
    -- an offset target (`plant_maps.OFFSET_TARGETS`) -- a plain shift.
    None without an enabled map on it: the caller then changes nothing."""
    rotation = target in mapping.pm.ROTATION_TARGETS
    offset = target in mapping.pm.OFFSET_TARGETS
    out = None
    for m in c.maps:
        if not m.enabled or m.amount <= 0.0 or m.target != target:
            continue
        node = curves.get_component_curve(obj, c.uid, curves.map_curve_name("imap", m.uid))
        if node is None:
            continue
        x = map_channel_values(bridge.channel_key(m), chan, ids,
                               _cseed(int(obj.plantgen.seed), c) * 7919 + m.uid * 104729)
        if x is None:
            continue
        y = curves.sample_node(node, x)
        if rotation:
            term = np.radians(float(m.angle) * float(m.amount) * (2.0 * y - 1.0))
            out = term if out is None else out + term
        elif offset:
            term = float(m.amount) * (y - 1.0)
            out = term if out is None else out + term
        else:
            term = 1.0 + float(m.amount) * (y - 1.0)
            out = term if out is None else out * term
    return out


def _has_imap(c, targets) -> bool:
    return any(m.enabled and m.amount > 0.0 and m.target in targets for m in c.maps)


def _apply_instance_maps(obj, organ_attrs, owned, ids, scale) -> None:
    """Each component's instance maps on the points it owns: a Driver channel
    through the map's curve, times Amount, onto Size or a rotation.

    The Natsura-style "every parameter can vary per point" (clean-room) for the
    rendered instances: masks decide *which* component takes a point, maps decide
    *how* it is drawn there. Several maps chain by multiplying (Size) or adding
    (angles). Render-only -- growth never sees them, so they ride the growth
    cache. No enabled map anywhere: nothing is touched (byte-identical).

    Rotations are about the instance's own frame: Pitch tilts its facing toward
    its up (`normal`), Yaw turns it about its up, Roll spins it about its facing.
    Bloom Timing shifts a flower's `bloom_start` (and delays its `bloom_birth`).
    """
    work = [(c, take) for c, take in owned
            if len(take) and any(m.enabled and m.target in mapping.pm.ORIENTATION_TARGETS
                                 for m in c.maps)]
    if not work:
        return
    chan = _channels_from(organ_attrs)
    direction = normal = bloom = birth = None
    for c, take in work:
        sub = {k: np.asarray(v)[take] for k, v in chan.items()}
        for m in c.maps:
            if (not m.enabled or m.amount <= 0.0
                    or m.target not in mapping.pm.ORIENTATION_TARGETS):
                continue
            node = curves.get_component_curve(
                obj, c.uid, curves.map_curve_name("imap", m.uid))
            if node is None:
                continue
            x = map_channel_values(bridge.channel_key(m), sub, ids[take],
                                   _cseed(int(obj.plantgen.seed), c) * 7919
                                   + m.uid * 104729)
            if x is None:
                continue
            y = curves.sample_node(node, x)
            if m.target == 'SIZE':
                scale[take] *= 1.0 + float(m.amount) * (y - 1.0)
                continue
            if m.target == 'BLOOM':
                # Each flower's place on the bloom timeline (0 opens first, 1
                # last) shifted earlier or later, kept on it; Spread then says
                # how much a place matters, as it does for the order itself.
                # From Age opens a flower at the later of that place and its
                # birth, so a delay moves the birth too -- else flowers born
                # together open together whatever their timing. Only a delay:
                # a flower still cannot open before it exists.
                if bloom is None:
                    bloom = np.asarray(organ_attrs["bloom_start"], dtype=np.float64)
                    birth = np.asarray(organ_attrs["bloom_birth"], dtype=np.float64)
                shift = float(m.amount) * (y - 1.0)
                bloom[take] = np.clip(bloom[take] + shift, 0.0, 1.0)
                birth[take] = np.clip(birth[take] + np.maximum(shift, 0.0), 0.0, 1.0)
                continue
            theta = np.radians(float(m.angle) * float(m.amount) * (2.0 * y - 1.0))
            if not np.any(theta):
                continue
            if direction is None:
                direction = np.asarray(organ_attrs["direction"], dtype=np.float64)
                normal = np.asarray(organ_attrs["normal"], dtype=np.float64)
            d, nrm = direction[take], normal[take]
            if m.target == 'PITCH':
                axis = _unit_rows(np.cross(d, nrm))
                d, nrm = _rotate(d, axis, theta), _rotate(nrm, axis, theta)
            elif m.target == 'YAW':
                d = _rotate(d, _unit_rows(nrm), theta)
            else:  # ROLL
                nrm = _rotate(nrm, _unit_rows(d), theta)
            direction[take], normal[take] = d, nrm
    if direction is not None:
        organ_attrs["direction"] = [tuple(r) for r in direction.tolist()]
        organ_attrs["normal"] = [tuple(r) for r in normal.tolist()]
    if bloom is not None:
        organ_attrs["bloom_start"] = bloom.tolist()
        organ_attrs["bloom_birth"] = birth.tolist()


def _reorient_placed(obj, organ_attrs, gen_specs, ids) -> None:
    """Placed points' Fold / Parent Roll instance maps: each facing rebuilt
    (`_generated_orientation`) with the maps' offsets added to its own Fold and
    Parent Roll -- before Sky, Pitch and Roll Self, where the parameters act.
    A component with no such map was never recorded: untouched."""
    work = [(c, gi, frames) for c, gi, frames in gen_specs if frames and len(gi)]
    if not work:
        return
    chan = _channels_from(organ_attrs)
    direction, normal = organ_attrs["direction"], organ_attrs["normal"]
    for c, gi, frames in work:
        sub = {k: np.asarray(v)[gi] for k, v in chan.items()}
        dfold = instance_map_values(obj, c, "FOLD", sub, ids[gi])
        droll = instance_map_values(obj, c, "ROLL_PARENT", sub, ids[gi])
        if dfold is None and droll is None:
            continue
        for j, i in enumerate(gi):
            radial, d, p, rs, rp, fold, sky = frames[j]
            if droll is not None:
                rp += float(droll[j])
            if dfold is not None:
                fold += float(dfold[j])
            dn, nn = _generated_orientation(radial, d, p, rs, rp, fold, sky)
            direction[i] = tuple(float(x) for x in dn)
            normal[i] = tuple(float(x) for x in nn)


def _apply_sky_roll(organ_attrs, owned, generated) -> None:
    """Each component's Sky on the grown points it owns: turn every instance
    about its own length so its face (`normal`) goes `sky_roll` of the way to as
    close to world up as it can get -- leaves twisting on their stalk to face
    the light, SpeedTree's Sky / Natsura's Unroll (clean-room, from their docs).
    Where it points is untouched. The target is up with the facing's own part
    removed; an instance pointing straight up or down has no such face and is
    left. Before the instance maps, so a map's Roll turns it from there.
    Placed (`generated`) points take Sky in their own frame instead, before
    their Pitch and Roll Self (`_generated_orientation`) -- applied after, it
    erased Roll Self and made Pitch swing as well as tip. Sky 0: untouched."""
    work = [(c, take[~generated[take]]) for c, take in owned
            if len(take) and float(getattr(c, "sky_roll", 0.0)) > 0.0]
    work = [(c, take) for c, take in work if len(take)]
    if not work:
        return
    direction = np.asarray(organ_attrs["direction"], dtype=np.float64)
    normal = np.asarray(organ_attrs["normal"], dtype=np.float64)
    for c, take in work:
        d = _unit_rows(direction[take])
        nrm = normal[take]
        target = _UP[None, :] - d * (d @ _UP)[:, None]
        reach = np.linalg.norm(target, axis=1)
        target = target / np.where(reach > 1e-6, reach, 1.0)[:, None]
        phi = np.arctan2(np.sum(np.cross(nrm, target) * d, axis=1), np.sum(nrm * target, axis=1))
        phi = np.where(reach > 1e-6, phi * min(float(mapping.cval(c, "sky_roll")), 1.0), 0.0)
        normal[take] = _rotate(nrm, d, phi)
    organ_attrs["normal"] = [tuple(r) for r in normal.tolist()]


def _emerging_shapes(obj, organ_attrs, owned, n: int, ids) -> None:
    """Leaf Unfolding: each leaf / bract point's
    curl and midrib fold NOW -- its component's Emerging Curl / Emerging Midrib
    Fold (each x its instance maps, if any) x (1 - how far it has unfolded,
    its expansion so far), so a leaf is flat exactly when it is full size.
    Written (`gn_mesh.LEAF_CURL_ATTR` / `LEAF_FOLD_ATTR`) only when some
    component has one: every other plant's points are unchanged."""
    shaped = [(c, take) for c, take in owned
              if c.type in ("leaf", "bract") and len(take)
              and (c.emerging_curl != 0.0 or c.emerging_midrib_fold != 0.0)]
    if not shaped:
        return
    young = 1.0 - np.clip(np.asarray(organ_attrs.get("unfolded", np.ones(n)),
                                     dtype=np.float64), 0.0, 1.0)
    curl, fold = np.zeros(n), np.zeros(n)
    chan = None
    for c, take in shaped:
        curl[take] = float(c.emerging_curl) * young[take]
        fold[take] = float(c.emerging_midrib_fold) * young[take]
        if _has_imap(c, ("EMERGING_CURL", "EMERGING_FOLD")):
            if chan is None:
                chan = _channels_from(organ_attrs)
            sub = {k: np.asarray(v)[take] for k, v in chan.items()}
            for target, values in (("EMERGING_CURL", curl), ("EMERGING_FOLD", fold)):
                f = instance_map_values(obj, c, target, sub, ids[take])
                if f is not None:
                    values[take] *= f
    organ_attrs[gn_mesh.LEAF_CURL_ATTR] = curl.tolist()
    organ_attrs[gn_mesh.LEAF_FOLD_ATTR] = fold.tolist()


def _assign_components(obj, s, organ_attrs, gen_specs=()) -> set:
    """Route every wired organ type through its components, in one pass over
    instance_index / flower block / scale / render_drop. Returns the set of types
    that have active components (the modifier switches those to the master-collection
    pick). No components of any type -> nothing written, the legacy single-collection
    path byte-identical.

    `gen_specs` are the along-stems generated points (Phase 3), each owned by the
    component that made it -- they skip the shared claim and are baked directly."""
    kind = np.asarray(organ_attrs["kind"])
    form = np.asarray(organ_attrs["form"])
    # The variation key every pick / jitter / Prune / Random mask hashes: the
    # organ id of a grown organ (unchanged), a placed point's place
    # (`_placed_key`) -- so a placed leaf keeps its look as the plant grows.
    ids = np.asarray(organ_attrs.get("vkey") or organ_attrs["organ_id"], dtype=np.int64)
    n = len(ids)
    idxs = np.zeros(n, dtype=np.int64)
    block_offset = np.zeros(n, dtype=np.int64)
    block_max = np.zeros(n, dtype=np.int64)
    scale = np.asarray(organ_attrs["scale"], dtype=np.float64).copy()
    drop = np.asarray(organ_attrs.get("render_drop", np.zeros(n)), dtype=np.float64).copy()
    generated = np.zeros(n, dtype=bool)
    for _c, gi, _frames in gen_specs:
        generated[gi] = True
    owned: list = []          # (component, point indices) -- for its instance maps

    active = set()
    for tkey in _WIRED_COMPONENT_TYPES:
        if _assign_type(obj, s, organ_attrs, tkey, kind, form, ids, idxs, scale, drop, generated,
                        owned):
            active.add(tkey)
    flowers = _assign_flowers(obj, s, organ_attrs, kind, ids, block_offset, block_max, scale, drop,
                              owned)
    if flowers:
        active.add("flower")
        organ_attrs["flower_block_offset"] = block_offset.tolist()
        organ_attrs["flower_block_max"] = block_max.tolist()

    # Bake the generated points, each into its owning component's collection through
    # that type's master (built by `_assign_type` above).
    seed = int(s.seed)
    for gk, (comp, gi, _frames) in enumerate(gen_specs):
        if not len(gi):
            continue
        master = master_collection(obj, comp.type)
        if master is None:
            continue
        active.add(comp.type)
        global_of = {o.name: i for i, o in enumerate(master.objects)}
        objs = list(comp.collection.objects)
        ti = _TYPE_SALT_INDEX.get(comp.type, 0)
        osalt = (_cseed(seed, comp) * 2654435761 + (gk + 1) * 2971215073 + ti * 40503) & _U64
        local = _pick_assets(seed, comp, objs, ids[gi], masks._hash01(ids[gi], osalt),
                             lambda gi=gi: {k: np.asarray(v)[gi] for k, v in
                                            _channels_from(organ_attrs).items()})
        idxs[gi] = [global_of[objs[li].name] for li in local]
        owned.append((comp, gi))
        scale[gi] *= float(comp.size)
        var = float(comp.gen_size_variance)
        if var > 0.0:
            # Seeded per-instance +/- fraction, its own salt so it does not track the
            # collection pick above; hash01 in [0,1) -> [1-var, 1+var).
            vsalt = (_cseed(seed, comp) * 2654435761 + (gk + 1) * 2246822519 + ti * 40503) & _U64
            scale[gi] *= 1.0 + (masks._hash01(ids[gi], vsalt) * 2.0 - 1.0) * var
        # Size curve read at each instance's OWN age (1 - birth age): points born
        # early -> full, points born late near the tips -> the curve's low end. A
        # flat curve (the default) is all 1.0, so this stays byte-identical off.
        node = curves.get_component_curve(obj, comp.uid)
        if node is not None:
            samp = np.asarray(curves.to_curve(node).samples, dtype=np.float64)
            if samp.size >= 2 and not np.allclose(samp, 1.0):
                inst_age = np.clip(1.0 - np.asarray(organ_attrs["age"])[gi], 0.0, 1.0)
                scale[gi] *= np.interp(inst_age, np.linspace(0.0, 1.0, samp.size), samp)
        # Only on the levels it is linked to.
        if "level" in organ_attrs:
            glevels = np.asarray(organ_attrs["level"])[gi]
            drop[gi[~structure.component_level_ok(s, comp, glevels)]] = 1.0
        # Masks apply to generated points too: a filter that rejects a point drops
        # it, exactly as it gates a grown organ point in `_assign_type`.
        stack = _filters_stack(comp, seed)
        if not stack.is_empty():
            gchan = {ch: np.asarray(v)[gi] for ch, v in _channels_from(organ_attrs).items()}
            keeps = masks.evaluate(stack, gchan, ids[gi])[0] > 0.5
            drop[gi[~keeps]] = 1.0
        if comp.prune > 0.0:
            psalt = (_cseed(seed, comp) * 40499 + (gk + 1) * 2246822519 + ti * 1597334677) & _U64
            drop[gi[masks._hash01(ids[gi], psalt) < float(comp.prune)]] = 1.0

    _reorient_placed(obj, organ_attrs, gen_specs, ids)
    _apply_sky_roll(organ_attrs, owned, generated)
    _apply_instance_maps(obj, organ_attrs, owned, ids, scale)
    _emerging_shapes(obj, organ_attrs, owned, n, ids)
    _stash_points(obj, s, organ_attrs, kind, form, ids, drop, owned)

    if active:
        organ_attrs["instance_index"] = idxs.tolist()
        organ_attrs["scale"] = scale.tolist()
        organ_attrs["render_drop"] = drop.tolist()
    return active


LAST_POINTS: dict = {}


def _stash_points(obj, s, organ_attrs, kind, form, ids, drop, owned) -> None:
    # RNA hands back a fresh wrapper per access, so components are matched by
    # their stable `uid`, never by identity.
    index_of = {c.uid: i for i, c in enumerate(s.components)}
    owner = np.full(len(ids), -1, dtype=np.int64)
    for c, take in owned:
        owner[take] = index_of.get(c.uid, -1)
    LAST_POINTS[obj.name] = {
        "channels": _channels_from(organ_attrs), "kind": kind, "form": form,
        "ids": ids, "drop": drop.copy(), "owner": owner, "seed": int(s.seed),
    }


def _component_colours(obj, s, n: int) -> list:
    """The Component Colours view: each organ point in the swatch colour of the
    component that took it (alpha 1), or none (alpha 0 -- the view mixes that
    to grey) where no component did. Read from this regrow's assignment
    (`LAST_POINTS`' owner), so it is exactly the component drawing each point."""
    owner = LAST_POINTS.get(obj.name, {}).get("owner")
    if owner is None or len(owner) != n:
        return [(0.0, 0.0, 0.0, 0.0)] * n
    swatch = [(*tuple(c.color)[:3], 1.0) for c in s.components]
    return [swatch[i] if 0 <= i < len(swatch) else (0.0, 0.0, 0.0, 0.0) for i in owner]


def _active_filter(s):
    """The filter the preview visualises: the active component's active filter
    entry, or None when there is nothing selected (preview then reads all-green)."""
    if not (0 <= s.components_active < len(s.components)):
        return None
    c = s.components[s.components_active]
    return c.filters[c.filters_active] if 0 <= c.filters_active < len(c.filters) else None


def _level_preview_values(s, plant, attrs, n) -> list:
    """The Mask Preview of a branching level's active mask: on the axes that
    bear that level's branches, each node's value is the mask entry's window as
    growth judged its buds there (`Bud.mask_windows`; the mean over a whorl) --
    green where it kept them, red where it removed them. Everything else reads
    1.0. The stems level has nothing to show: a stem its masks drop never
    exists, and there is no parent to colour."""
    li = s.mask_preview_level
    if n == 0 or plant is None or not (0 <= li < len(s.branch_levels)):
        return [1.0] * n
    k = s.branch_levels[li].filters_active
    by_node: dict = {}
    for axis in plant.axes.values():
        for ph in axis.phytomers:
            for bud in ph.buds:
                if bud.level != li:
                    continue
                if len(bud.mask_windows) > k and bud.mask_windows[k] >= 0.0:
                    by_node.setdefault((axis.id, ph.index), []).append(bud.mask_windows[k])
    if not by_node or "axis_id" not in attrs or "node_index" not in attrs:
        return [1.0] * n
    mean = {key: float(np.mean(v)) for key, v in by_node.items()}
    return [mean.get((int(a), int(i)), 1.0)
            for a, i in zip(attrs["axis_id"], attrs["node_index"], strict=True)]


def _preview_values(s, attrs, n, plant=None) -> list:
    """What the Filter Preview bakes per point (0 masked out/red -> 1 kept/green):
    the active component's active filter's own attribute window -- `masks._remap`,
    the same remap the filter applies, with its invert. A point whose channel this
    domain does not carry reads 1.0 (kept/green), so a structural/organ-only channel
    colours only the organs. Prune is excluded: the preview is the deterministic
    window, not the random thinning."""
    if n == 0:
        return []
    if s.mask_preview_level >= 0:
        return _level_preview_values(s, plant, attrs, n)
    entry = _active_filter(s)
    if entry is None:
        return [1.0] * n
    if entry.channel == "random":
        ids = attrs.get("organ_id")
        if ids is None:
            return [1.0] * n
        values = masks._hash01(np.asarray(ids), 0)
    else:
        key = (bridge.channel_key(entry) if entry.channel in ("custom", "class")
               else _CHANNEL_ATTR.get(entry.channel))
        if key is None or key not in attrs:
            return [1.0] * n
        values = np.asarray(attrs[key], dtype=np.float64)
    lo, hi = masks.effective_window(bridge.channel_key(entry), float(entry.lo), float(entry.hi))
    w = masks._remap(values, lo, hi)
    if entry.invert:
        w = 1.0 - w
    return np.asarray(w, dtype=np.float64).tolist()
