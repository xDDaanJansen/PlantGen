"""The plant structure -- branching levels and where components are
placed on them: uids, the Components tree's rows and selection (kept in
step with the Mapping list), placements, and which level each point or
axis is on. Split out of `bridge.py` (2026-10-13)."""

from __future__ import annotations

import numpy as np

from . import bridge, mapping


def ensure_component_uids(s) -> None:
    """Give every component a unique uid (placements refer to it)."""
    seen: set = set()
    nxt = max((c.uid for c in s.components), default=0) + 1
    for c in s.components:
        if c.uid <= 0 or c.uid in seen:
            c.uid, nxt = nxt, nxt + 1
        seen.add(c.uid)


def component_by_uid(s, uid):
    """(index, component) for a uid, (-1, None) if there is none."""
    for i, c in enumerate(s.components):
        if c.uid == uid:
            return i, c
    return -1, None


def level_index_of(s, uid) -> int:
    return next((i for i, lv in enumerate(s.branch_levels) if lv.uid == uid), -1)


def child_levels_of(s, parent_uid) -> list:
    """The levels growing on `parent_uid` (0: the stems levels), list order."""
    return [i for i, lv in enumerate(s.branch_levels) if lv.parent_uid == parent_uid]


def level_subtree(s, uid) -> set:
    """`uid` and the uid of every level growing on it, at any depth."""
    out, todo = set(), [uid]
    while todo:
        u = todo.pop()
        out.add(u)
        todo += [lv.uid for lv in s.branch_levels if lv.parent_uid == u and lv.uid not in out]
    return out


def level_can_move(s, i, parent_uid) -> bool:
    """May level `i` grow on `parent_uid`? The hierarchy is locked by kind: a
    stems level stays on the Plant, a branch level stays on a stems or branch
    level (never the Plant) -- and never on itself or a level growing on it."""
    lv = s.branch_levels[i]
    if lv.parent_uid == 0:
        return parent_uid == 0
    return parent_uid != 0 and parent_uid not in level_subtree(s, lv.uid)


def place(s, level_uid, comp_uid) -> int:
    """Place a component on a level (0: the Plant, every level); the placement's
    index -- the existing one if it is already there."""
    for k, q in enumerate(s.placements):
        if q.level_uid == level_uid and q.comp_uid == comp_uid:
            return k
    q = s.placements.add()
    q.level_uid, q.comp_uid = level_uid, comp_uid
    return len(s.placements) - 1


def component_users(s, comp_uid) -> int:
    return sum(1 for q in s.placements if q.comp_uid == comp_uid)


def component_level_names(s, comp_uid) -> list:
    names = {lv.uid: lv.name for lv in s.branch_levels}
    return ["Plant" if q.level_uid == 0 else names.get(q.level_uid, "?")
            for q in s.placements if q.comp_uid == comp_uid]


_TREE_GUARD = [False]


def rebuild_tree(s) -> None:
    """Flatten the levels and placements into the Plant Structure rows: the
    Plant, its every-level components, then each stems level with its
    components and the levels growing on it, and so on -- each row with its
    guide lines (`'|'` a line, `'T'` a branch, `'L'` the last). Keeps the
    selection on the focused row. Runs in operators and at regrow, never in
    draw."""
    rows = []

    def walk(kind, ref, prefix, last, root):
        rows.append((kind, ref, "" if root else prefix + ("L" if last else "T")))
        below = "" if root else prefix + (" " if last else "|")
        if kind == 'COMPONENT':
            return
        if kind == 'PLANT':
            if not s.structure_plant_expanded:
                return
            uid = 0
        else:
            lv = s.branch_levels[ref]
            if not lv.expanded:
                return
            uid = lv.uid
        kids = [('COMPONENT', k) for k, q in enumerate(s.placements) if q.level_uid == uid]
        kids += [('LEVEL', i) for i in child_levels_of(s, uid)]
        for i, (k, r) in enumerate(kids):
            walk(k, r, below, i == len(kids) - 1, False)

    walk('PLANT', 0, "", True, True)
    # Components placed on no level yet (new, copied, or taken off their last
    # level) stay in the library: listed last, under their own header.
    placed = {q.comp_uid for q in s.placements}
    loose = [i for i, c in enumerate(s.components) if c.uid not in placed]
    if loose:
        rows.append(('UNPLACED', 0, ""))
        for n, i in enumerate(loose):
            rows.append(('LIBRARY', i, "L" if n == len(loose) - 1 else "T"))
    _TREE_GUARD[0] = True
    try:
        s.tree_rows.clear()
        active = 0
        for i, (kind, ref, guides) in enumerate(rows):
            r = s.tree_rows.add()
            r.kind, r.ref, r.guides = kind, ref, guides
            if kind == s.structure_focus and (
                    kind == 'PLANT'
                    or (kind == 'LEVEL' and ref == s.branch_levels_active)
                    or (kind == 'COMPONENT' and ref == s.structure_place)
                    or (kind == 'LIBRARY' and ref == s.components_active)):
                active = i
        s.tree_active = active
    finally:
        _TREE_GUARD[0] = False


def tree_row_selected(s) -> None:
    """A Plant Structure row was clicked: select its level / component (the
    component also becomes the library's active one)."""
    if _TREE_GUARD[0] or not 0 <= s.tree_active < len(s.tree_rows):
        return
    row = s.tree_rows[s.tree_active]
    if row.kind == 'UNPLACED':
        rebuild_tree(s)         # the header selects nothing: back to the selection
        return
    s.structure_focus = row.kind
    owner = (mapping.PLANT, 0) if row.kind == 'PLANT' else None
    if row.kind == 'LIBRARY' and 0 <= row.ref < len(s.components):
        s.components_active = row.ref
        owner = (mapping.COMPONENT, int(s.components[row.ref].uid))
    elif row.kind == 'LEVEL' and 0 <= row.ref < len(s.branch_levels):
        s.branch_levels_active = row.ref
        owner = (mapping.LEVEL, int(s.branch_levels[row.ref].uid))
    elif row.kind == 'COMPONENT' and 0 <= row.ref < len(s.placements):
        s.structure_place = row.ref
        q = s.placements[row.ref]
        s.structure_parent_uid = q.level_uid
        i, _c = component_by_uid(s, q.comp_uid)
        if i >= 0:
            s.components_active = i
            owner = (mapping.COMPONENT, int(q.comp_uid))
    if owner is not None:
        mapping.follow_owner(s, *owner)        # the Mapping list follows


def focus_owner(s, owner: str, uid: int) -> None:
    """Select a map's owner in the Components tree (a Mapping list pick): the
    Plant, a level, or a component -- its row on the level already selected if
    it is placed there, else its first placement, else the library -- with
    every level above it unfolded so the row shows."""
    if owner == mapping.PLANT:
        s.structure_focus = 'PLANT'
    elif owner == mapping.LEVEL:
        i = level_index_of(s, uid)
        if i < 0:
            return
        _unfold_to(s, s.branch_levels[i].parent_uid)
        s.structure_focus = 'LEVEL'
        s.branch_levels_active = i
    else:
        i, _c = component_by_uid(s, uid)
        if i < 0:
            return
        s.components_active = i
        here = (s.structure_focus == 'COMPONENT' and 0 <= s.structure_place < len(s.placements)
                and s.placements[s.structure_place].comp_uid == uid)
        if not here:
            k = next((k for k, q in enumerate(s.placements) if q.comp_uid == uid), -1)
            if k < 0:
                s.structure_focus = 'LIBRARY'
            else:
                s.structure_focus = 'COMPONENT'
                s.structure_place = k
                s.structure_parent_uid = s.placements[k].level_uid
                _unfold_to(s, s.placements[k].level_uid)
    rebuild_tree(s)


def _unfold_to(s, level_uid: int) -> None:
    """Unfold the tree down to `level_uid` (0: the Plant) so its rows show."""
    s.structure_plant_expanded = True
    seen = set()
    while level_uid and level_uid not in seen:
        seen.add(level_uid)
        i = level_index_of(s, level_uid)
        if i < 0:
            return
        s.branch_levels[i]["expanded"] = True
        level_uid = s.branch_levels[i].parent_uid


def structure_target(s) -> int:
    """The level uid a new component / level goes on: the selected level, a
    selected component's level, else the Plant (0)."""
    if s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(s.branch_levels):
        return s.branch_levels[s.branch_levels_active].uid
    if s.structure_focus == 'COMPONENT' and 0 <= s.structure_place < len(s.placements):
        return s.placements[s.structure_place].level_uid
    return 0


def ensure_level_uids(s) -> None:
    """Give every branching level a unique uid (a plant from before the tree has
    them all 0). Runs at regrow, never in draw."""
    seen: set = set()
    nxt = max((lv.uid for lv in s.branch_levels), default=0) + 1
    for lv in s.branch_levels:
        if lv.uid <= 0 or lv.uid in seen:
            lv.uid, nxt = nxt, nxt + 1
        seen.add(lv.uid)


def _axis_level(axis, n_levels: int) -> int:
    """A vegetative axis's level: its own, else its order's (clamped)."""
    if 0 <= axis.level < n_levels:
        return axis.level
    return min(max(axis.order, 0), max(n_levels - 1, 0))


def _axis_levels(plant, n_levels: int) -> dict:
    """axis id -> the branching level it is on: a vegetative axis (stem, branch,
    fork) its own, an inflorescence branch the level of the axis it grows
    from."""
    out: dict = {}

    def level_of(axis) -> int:
        if axis.id in out:
            return out[axis.id]
        if axis.structural_depth >= 1 and not axis.is_fork:
            parent = plant.axes.get(axis.parent_axis_id)
            value = level_of(parent) if parent is not None else 0
        else:
            value = _axis_level(axis, n_levels)
        out[axis.id] = value
        return value

    for axis in plant.axes_in_topological_order():
        level_of(axis)
    return out


def _section_levels(plant, profile, attrs) -> list:
    """Each stem vertex's cross-section level: a stem / branch / fork its own
    level's; a pedicel, catkin or inflorescence branch the first level growing
    on the level it grows from (what a chain always gave it), else that level."""
    n = len(profile.levels)
    if n == 0:
        return [0] * len(attrs["axis_id"])
    by_axis = _axis_levels(plant, n)
    parents = [profile.level_parent(i) for i in range(n)]
    first_child = {}
    for i, p in enumerate(parents):
        first_child.setdefault(p, i)
    out = []
    for aid, order in zip(attrs["axis_id"], attrs["order"], strict=True):
        axis = plant.axes.get(int(aid))
        if axis is None:
            out.append(0)
            continue
        own = by_axis.get(axis.id, 0)
        vegetative = axis.structural_depth < 1 or axis.is_fork
        if vegetative and int(order) == axis.order:
            out.append(own)
        else:
            out.append(first_child.get(own, own))
    return out


def point_levels(plant, n_levels: int, branch_ordinal) -> np.ndarray:
    """Each point's branching level, from its axis's rank (`branch_ordinal`,
    the index in the sorted axis ids every point carries)."""
    ranked = sorted(plant.axes)
    by_axis = _axis_levels(plant, n_levels)
    table = np.array([by_axis.get(aid, 0) for aid in ranked] or [0], dtype=np.int64)
    ranks = np.clip(np.asarray(branch_ordinal, dtype=np.int64), 0, len(table) - 1)
    return table[ranks]


def component_level_indices(s, c) -> list:
    """The level indices a component is placed on (every level when it is on
    the Plant)."""
    at = {q.level_uid for q in s.placements if q.comp_uid == c.uid}
    if 0 in at:
        return list(range(len(s.branch_levels)))
    return sorted(i for i, lv in enumerate(s.branch_levels) if lv.uid in at)


def component_inflorescence_ok(s, c, inflorescences) -> np.ndarray:
    """Which flower points component `c` may take by inflorescence: only its
    own's when it grows one, any when it is a variant."""
    infl = np.asarray(inflorescences)
    own = bridge.component_inflorescence(s, c)
    if own is None:
        return np.ones(infl.shape, dtype=bool)
    return infl == own


def component_level_ok(s, c, levels) -> np.ndarray:
    """Which points a component may take by level: those on a level it is
    placed on (all of them when it is on the Plant; none when it is placed
    nowhere)."""
    levels = np.asarray(levels)
    if levels.size == 0:
        return np.ones(levels.shape, dtype=bool)
    if any(q.comp_uid == c.uid and q.level_uid == 0 for q in s.placements):
        return np.ones(levels.shape, dtype=bool)
    return np.isin(levels, component_level_indices(s, c))
