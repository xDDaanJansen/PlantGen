"""Parameter mapping, the Blender side: which
parameters are mappable, a parameter's maps across the kinds that evaluate
them (growth/plant_maps.py), the Mapping list's rows, adding / removing /
presets, and the per-plant evaluation at the panel -> model boundary."""

from __future__ import annotations

from ..growth import plant_maps as pm
from . import curves, map_icons

PLANT, LEVEL, COMPONENT = pm.OWNER_PLANT, pm.OWNER_LEVEL, pm.OWNER_COMPONENT

# Store -> (curve name prefix, curve owner uid (None: the component's own)).
_CURVE_KIND = {"LMAP": "lmap", "GMAP": "gmap", "IMAP": "imap", "FMAP": "fmap", "PMAP": "pmap"}
_COMPONENT_KINDS = ("IMAP", "FMAP")
_OWNER_KINDS = {LEVEL: ("LMAP",), PLANT: ("GMAP",), COMPONENT: _COMPONENT_KINDS}

DEPTH_COLOURS = ((0.53, 0.70, 0.37), (0.82, 0.64, 0.23), (0.65, 0.54, 0.85),
                 (0.89, 0.54, 0.29), (0.45, 0.62, 0.92))
"""A level's default swatch, by its depth in the tree (stems green, ochre...)."""
COMPONENT_COLOURS = ((0.83, 0.40, 0.62), (0.27, 0.69, 0.64), (0.35, 0.75, 0.40),
                     (0.95, 0.74, 0.25), (0.45, 0.55, 0.95), (0.85, 0.35, 0.30),
                     (0.62, 0.45, 0.85), (0.55, 0.80, 0.85))
"""New components take the first of these no other component uses."""

_EXCLUDE = frozenset({
    "seed", "uid", "parent_uid", "owner_uid", "comp_uid", "level_uid", "ref",
    "collide_max_turn", "density", "prune", "flower_count_display", "rig_bone_angle",
})
_EXCLUDE_SUFFIXES = ("_active", "_authored", "_schema", "_index_cache")
# Animation settings, not growth parameters: the timelapse's days, the rig and
# the wind act on the finished plant, so a map over them means nothing.
_EXCLUDE_PREFIXES = ("wind_", "timelapse_", "rig_")
_ANGLE_WORDS = ("pitch", "yaw", "roll", "angle", "rotation", "divergence", "twist")


# --- owners ---------------------------------------------------------------------
def owner_of(pg) -> tuple[str, int] | None:
    """(owner kind, uid) of a property group whose parameters can be mapped."""
    try:
        ident = pg.bl_rna.identifier
    except AttributeError:
        return None
    if ident == "PLANTGEN_PG_branch_level":
        return LEVEL, int(pg.uid)
    if ident == "PLANTGEN_PG_component":
        return COMPONENT, int(pg.uid)
    if ident == "PLANTGEN_PG_growth":
        return PLANT, 0
    return None


def owner_pg(s, owner: str, uid: int):
    if owner == PLANT:
        return s
    if owner == LEVEL:
        return next((lv for lv in s.branch_levels if lv.uid == uid), None)
    return next((c for c in s.components if c.uid == uid), None)


def component_type(pg) -> str:
    return getattr(pg, "type", "") if owner_of(pg) and owner_of(pg)[0] == COMPONENT else ""


def owner_color(s, owner: str, uid: int):
    pg = owner_pg(s, owner, uid)
    if pg is None:
        return (0.5, 0.5, 0.5)
    return tuple(pg.plant_color) if owner == PLANT else tuple(pg.color)


def owner_title(s, owner: str, uid: int) -> str:
    pg = owner_pg(s, owner, uid)
    if pg is None:
        return "(gone)"
    if owner == PLANT:
        return "Plant"
    if owner == LEVEL:
        return pg.name or "Level"
    return (getattr(pg, "name_override", "") or getattr(pg, "label", "")
            or (pg.collection.name if pg.collection else pg.type.title()))


def owner_seed(s, owner: str, uid: int) -> int:
    pg = owner_pg(s, owner, uid)
    return int(getattr(pg, "seed", 0)) if pg is not None and owner != PLANT else 0


def owners(s) -> list[tuple[str, int]]:
    """Every owner in Plant Structure order: the Plant, the levels, the components."""
    return ([(PLANT, 0)] + [(LEVEL, int(lv.uid)) for lv in s.branch_levels]
            + [(COMPONENT, int(c.uid)) for c in s.components])


def icon(s, owner: str, uid: int, style: str) -> int:
    return map_icons.icon(owner_color(s, owner, uid), style)


def ensure_level_colours(s) -> None:
    """A level whose colour was never set takes its depth's default, once."""
    by_uid = {lv.uid: lv for lv in s.branch_levels}
    for lv in s.branch_levels:
        if lv.color_authored:
            continue
        depth, p, seen = 0, lv, set()
        while p is not None and p.parent_uid in by_uid and p.uid not in seen:
            seen.add(p.uid)
            p = by_uid[p.parent_uid]
            depth += 1
        lv["color"] = DEPTH_COLOURS[min(depth, len(DEPTH_COLOURS) - 1)]
        lv["color_authored"] = True


def next_component_colour(s):
    """The first component colour no component uses yet."""
    used = {tuple(round(c, 2) for c in comp.color) for comp in s.components}
    for colour in COMPONENT_COLOURS:
        if tuple(round(c, 2) for c in colour) not in used:
            return colour
    return COMPONENT_COLOURS[len(s.components) % len(COMPONENT_COLOURS)]


# --- which parameters ------------------------------------------------------------
def is_mappable(pg, param: str) -> bool:
    """A numeric parameter of a level, a component or the plant -- not a +/-
    value (that is a map now), not an id, seed or view state."""
    found = owner_of(pg)
    if found is None:
        return False
    if param in pm.PSEUDO_PARAMS:
        return True
    if param in pm.VARIANCE_PROPS or param in _EXCLUDE or param.endswith(_EXCLUDE_SUFFIXES) \
            or param.startswith(_EXCLUDE_PREFIXES):
        return False
    prop = pg.bl_rna.properties.get(param)
    if prop is None or prop.type not in {'FLOAT', 'INT'}:
        return False
    if getattr(prop, "is_array", False) or getattr(prop, "is_hidden", False):
        return False
    return True


_FLOWER_WORDS = ("inflorescence_", "pedicel_", "flower_", "catkin_", "head_", "bloom",
                 "cluster_")


def searchable_params(pg, owner: str) -> list[str]:
    """The parameters the Mapping panel's + offers for one owner: a level's own,
    a component's that its type uses (a leaf component carries the flower ones
    too, unused), the plant's model fields that no component owns."""
    if pg is None:
        return []
    ctype = component_type(pg)
    if owner == PLANT:
        from .bridge import _FIELDS
        names = [f[0] for f in _FIELDS if not f[0].startswith(_FLOWER_WORDS)]
    else:
        names = [p.identifier for p in pg.bl_rna.properties]
        if owner == COMPONENT and ctype != "flower":
            names = [n for n in names if not n.startswith(_FLOWER_WORDS)]
    out = [n for n in names if is_mappable(pg, n)]
    if owner == COMPONENT:
        out += ["instance_pitch", "instance_yaw", "instance_roll"]
        if ctype in ("leaf", "bract"):
            out.append("petiole_angle")
        if ctype == "flower":
            out += ["catkin_hang", "bloom_timing"]
    elif owner == LEVEL:
        out.append("gravitropic_setpoint")
    return sorted(dict.fromkeys(out), key=lambda p: param_label(pg, p))


def param_label(pg, param: str) -> str:
    if param in pm.PSEUDO_PARAMS:
        return pm.PSEUDO_PARAMS[param]
    prop = pg.bl_rna.properties.get(param) if pg is not None else None
    return prop.name if prop is not None else param.replace("_", " ").title()


def is_additive(pg, param: str) -> bool:
    """Whether a parameter's maps add (an angle, which may be 0) or multiply."""
    if param in ("instance_pitch", "instance_yaw", "instance_roll", "petiole_angle"):
        return True
    prop = pg.bl_rna.properties.get(param) if pg is not None else None
    if prop is not None and getattr(prop, "subtype", "") == 'ANGLE':
        return True
    return any(word in param for word in _ANGLE_WORDS)


def is_integer(pg, param: str) -> bool:
    prop = pg.bl_rna.properties.get(param) if pg is not None else None
    return prop is not None and prop.type == 'INT'


def variance_prop(pg, owner: str, param: str) -> str | None:
    var = pm.variance_table(owner, component_type(pg)).get(param)
    return var if var is not None and pg is not None and hasattr(pg, var) else None


# --- the profile curves (a "Curve" row IS the curve, like a +/- value's range) ----
def curve_entry(s, owner: str, uid: int, param: str):
    """(curve key, short name, driver caption) of the multiplier curve on a
    parameter, or None -- where its owner's kind reads it."""
    pg = owner_pg(s, owner, uid)
    if pg is None:
        return None
    found = pm.curve_table(owner, component_type(pg)).get(param)
    if found is None:
        return None
    key, (short, caption) = found
    if owner == COMPONENT and component_type(pg) == "flower":
        # where the inflorescence reads it (the old panel's button showed only there)
        from ..growth import morphology
        kind = str(pg.inflorescence_kind).lower()
        if key in pm.FLOWERING_CURVES and not morphology.kind_gradients_apply(kind):
            return None             # a terminal cluster has no along-axis position
        if key.startswith("pedicel_") and not morphology.kind_has_pedicels(kind):
            return None
        if key == "flower_pitch" and morphology.kind_has_pedicels(kind):
            return None             # a stalked flower's tilt is its pedicel's
        if key.startswith("infl_branch_") and (
                kind in morphology.RELAY_KINDS or pg.inflorescence_branching_depth <= 0):
            return None
    return key, short, caption


def param_curve_node(s, owner: str, uid: int, param: str, *, create: bool = False):
    """The curve node behind a parameter's Curve row (None if not built yet;
    `create` builds it -- operators only, never a draw)."""
    entry = curve_entry(s, owner, uid, param)
    if entry is None:
        return None
    obj, key = s.id_data, entry[0]
    if owner == PLANT:
        return (curves.ensure_curves(obj) if create else curves.get_curves(obj)).get(key)
    if create:
        return curves.ensure_component_curve(obj, uid, key)
    return curves.get_component_curve(obj, uid, key)


def _curve_key(node) -> str:
    """A curve node's key ("pedicel_length"), its component prefix dropped."""
    return str(node.get("pgs_key") or "").rsplit(":", 1)[-1]


def _flat(node) -> bool:
    """At its neutral -- flat at 1, or a non-multiplier's own default line --
    so the curve changes nothing: not mapped."""
    key = _curve_key(node)
    return all(abs(p.location[1] - pm.curve_neutral_y(key, p.location[0])) < 1e-6
               for p in node.mapping.curves[0].points)


def _set_neutral(node) -> None:
    """Back to its neutral shape (no curve)."""
    curves.set_points(node, pm.CURVE_NEUTRAL.get(_curve_key(node), ((0.0, 1.0), (1.0, 1.0))),
                      'VECTOR')


def _curve_mute(s, owner: str, uid: int, param: str):
    for k, x in enumerate(s.curve_mutes):
        if x.owner == owner and x.owner_uid == uid and x.param == param:
            return k, x
    return -1, None


def curve_muted(s, owner: str, uid: int, param: str) -> bool:
    """Is this parameter's curve switched off? (Flat meanwhile, its points kept;
    one drawn again since is on.)"""
    node = param_curve_node(s, owner, uid, param)
    return (node is not None and _flat(node)
            and _curve_mute(s, owner, uid, param)[1] is not None)


def _has_curve(s, owner: str, uid: int, param: str) -> bool:
    node = param_curve_node(s, owner, uid, param)
    return node is not None and (not _flat(node)
                                 or _curve_mute(s, owner, uid, param)[1] is not None)


def set_curve_enabled(s, owner: str, uid: int, param: str, enabled: bool) -> None:
    """Switch a parameter's curve off (flat at 1 -- exactly no curve -- its points
    kept) or back on (its points restored)."""
    node = param_curve_node(s, owner, uid, param)
    if node is None:
        return
    k, kept = _curve_mute(s, owner, uid, param)
    if enabled:
        if kept is not None:
            points = [p.split(",") for p in kept.points.split(";") if p]
            s.curve_mutes.remove(k)
            if _flat(node) and points:
                curve = node.mapping.curves[0]
                while len(curve.points) > max(len(points), 2):
                    curve.points.remove(curve.points[-1])
                while len(curve.points) < len(points):
                    curve.points.new(0.5, 0.5)
                for point, (x, y, handle) in zip(curve.points, points, strict=False):
                    point.location = (float(x), float(y))
                    point.handle_type = handle
                node.mapping.update()
    elif not _flat(node):
        if kept is None:
            kept = s.curve_mutes.add()
            kept.owner, kept.owner_uid, kept.param = owner, uid, param
        kept.points = ";".join(f"{p.location[0]!r},{p.location[1]!r},{p.handle_type}"
                               for p in node.mapping.curves[0].points)
        _set_neutral(node)


def per_label(s, owner: str, uid: int, param: str) -> str:
    pg = owner_pg(s, owner, uid)
    stems = owner == LEVEL and pg is not None and not pg.parent_uid
    return "per " + pm.evaluated_per(owner, param, stems_level=stems,
                                     component_type=component_type(pg))


# --- a parameter's maps -----------------------------------------------------------
def _store(s, kind: str, owner: str, uid: int):
    if kind == "LMAP":
        lv = owner_pg(s, LEVEL, uid)
        return lv.maps if lv is not None else None
    if kind == "GMAP":
        return s.growth_maps
    if kind == "IMAP":
        c = owner_pg(s, COMPONENT, uid)
        return c.maps if c is not None else None
    if kind == "FMAP":
        c = owner_pg(s, COMPONENT, uid)
        return c.growth_maps if c is not None else None
    return s.param_maps


def _param_of(kind: str, m) -> str:
    """The parameter a stored map is on."""
    if kind == "LMAP":
        return next((p for p, t in pm.LEVEL_HOOKS.items() if t == m.target), m.target)
    if kind == "GMAP":
        return next((p for p, t in pm.PLANT_HOOKS.items() if t == m.target), m.target)
    if kind == "IMAP":
        return next((p for p, t in pm.INSTANCE_HOOKS.items() if t == m.target), m.target)
    if kind == "FMAP":
        return next((p for p, t in pm.FLOWER_HOOKS.items() if t == m.target), m.target)
    return m.param


def _mute(s, owner: str, uid: int, param: str):
    """(index, entry) of a switched-off Random Range's kept value, or (-1, None)."""
    for k, x in enumerate(s.var_mutes):
        if x.owner == owner and x.owner_uid == uid and x.param == param:
            return k, x
    return -1, None


def _keep(s, owner: str, uid: int, param: str):
    """(index, record) of a Random Range row kept at 0, or (-1, None)."""
    for k, x in enumerate(s.var_keeps):
        if x.owner == owner and x.owner_uid == uid and x.param == param:
            return k, x
    return -1, None


def _keep_var(s, owner: str, uid: int, param: str) -> None:
    """Keep this Random Range row in the list, whatever its Range."""
    if _keep(s, owner, uid, param)[1] is None:
        rec = s.var_keeps.add()
        rec.owner, rec.owner_uid, rec.param = owner, uid, param


def _has_var(s, owner: str, uid: int, param: str, pg, var: str) -> bool:
    """Is there a Random Range row for this +/- value: not 0, switched off, or
    kept -- a Range set to 0 keeps its row; only the list's - removes it."""
    return (abs(float(getattr(pg, var))) > 0.0 or _mute(s, owner, uid, param)[1] is not None
            or _keep(s, owner, uid, param)[1] is not None)


def var_muted(s, owner: str, uid: int, param: str) -> bool:
    """Is this +/- value's Random Range switched off? (Its +/- is 0 and its value
    kept; a +/- set again since -- a preset, say -- is on.)"""
    pg = owner_pg(s, owner, uid)
    var = variance_prop(pg, owner, param)
    return (var is not None and float(getattr(pg, var)) == 0.0
            and _mute(s, owner, uid, param)[1] is not None)


def var_field(s, owner: str, uid: int, param: str):
    """(data, property) of a Random Range's value: the +/- itself, or the kept
    value while it is switched off."""
    if var_muted(s, owner, uid, param):
        return _mute(s, owner, uid, param)[1], "value"
    pg = owner_pg(s, owner, uid)
    return pg, variance_prop(pg, owner, param)


def set_var_enabled(s, owner: str, uid: int, param: str, enabled: bool) -> None:
    """Switch a Random Range on a +/- value off (the +/- goes to 0, its value
    kept) or back on (the kept value restored). Off is exactly a +/- of 0."""
    pg = owner_pg(s, owner, uid)
    var = variance_prop(pg, owner, param)
    if var is None:
        return
    k, kept = _mute(s, owner, uid, param)
    kind = type(getattr(pg, var))
    if enabled:
        if kept is not None:
            value = kept.value
            s.var_mutes.remove(k)
            if float(getattr(pg, var)) == 0.0:
                setattr(pg, var, kind(value))
    elif float(getattr(pg, var)) != 0.0:
        if kept is None:
            kept = s.var_mutes.add()
            kept.owner, kept.owner_uid, kept.param = owner, uid, param
        kept.value = float(getattr(pg, var))
        setattr(pg, var, kind(0))


def owner_entries(s, owner: str, uid: int) -> list[tuple[str, int, str]]:
    """Every map on one owner: (store kind, index, parameter); 'VAR' (index -1)
    for a +/- value that is not 0, or one switched off."""
    pg = owner_pg(s, owner, uid)
    if pg is None:
        return []
    out = []
    for param, var in pm.variance_table(owner, component_type(pg)).items():
        if hasattr(pg, var) and _has_var(s, owner, uid, param, pg, var):
            out.append(("VAR", -1, param))
    for param in pm.curve_table(owner, component_type(pg)):
        if _has_curve(s, owner, uid, param):
            out.append(("CURVE", -1, param))
    for kind in _OWNER_KINDS[owner]:
        store = _store(s, kind, owner, uid)
        for i, m in enumerate(store or ()):
            out.append((kind, i, _param_of(kind, m)))
    for i, m in enumerate(s.param_maps):
        if m.owner == owner and m.owner_uid == uid:
            out.append(("PMAP", i, m.param))
    return out


def entries(s, owner: str, uid: int, param: str) -> list[tuple[str, int]]:
    """One parameter's maps (owner_entries narrowed to it, without visiting the
    owner's other parameters -- every map icon asks this on every redraw)."""
    pg = owner_pg(s, owner, uid)
    if pg is None:
        return []
    out = []
    var = variance_prop(pg, owner, param)
    if var is not None and _has_var(s, owner, uid, param, pg, var):
        out.append(("VAR", -1))
    if param in pm.curve_table(owner, component_type(pg)) and _has_curve(s, owner, uid, param):
        out.append(("CURVE", -1))
    for kind in _OWNER_KINDS[owner]:
        for i, m in enumerate(_store(s, kind, owner, uid) or ()):
            if _param_of(kind, m) == param:
                out.append((kind, i))
    for i, m in enumerate(s.param_maps):
        if m.owner == owner and m.owner_uid == uid and m.param == param:
            out.append(("PMAP", i))
    return out


def stored(s, kind: str, owner: str, uid: int, index: int):
    """The map PG of a stored entry (None for a +/- value or a curve)."""
    if kind in ("VAR", "CURVE"):
        return None
    store = _store(s, kind, owner, uid)
    return store[index] if store is not None and 0 <= index < len(store) else None


def entry_enabled(s, kind: str, owner: str, uid: int, param: str, index: int) -> bool:
    if kind == "VAR":
        return not var_muted(s, owner, uid, param)
    if kind == "CURVE":
        return not curve_muted(s, owner, uid, param)
    m = stored(s, kind, owner, uid, index)
    return m is not None and m.enabled


def is_mapped(s, owner: str, uid: int, param: str) -> bool:
    return any(entry_enabled(s, kind, owner, uid, param, i)
               for kind, i in entries(s, owner, uid, param))


PRESET_NAMES = {'RANGE': "Random Range", 'HEIGHT': "Along Height", 'AGE': "By Age",
                'CURVE': "Custom Curve"}
PER_PLANT_CURVE = "Per Plant (random)"
"""What a Custom Curve is on a parameter read once for the whole plant: its
curve over one random draw per plant -- there is no attribute to vary along."""


def per_plant(owner: str, param: str) -> bool:
    """A parameter with no per-element evaluator: one value for the whole plant."""
    return pm.hook_target(owner, param) is None


def preset_label(owner: str, param: str, preset: str) -> str:
    """A mode's name for this parameter (`PER_PLANT_CURVE` where a Custom Curve
    can only be one random value per plant)."""
    if preset == 'CURVE' and per_plant(owner, param):
        return PER_PLANT_CURVE
    return PRESET_NAMES.get(preset, PRESET_NAMES['CURVE'])


def map_name(kind: str, m) -> str:
    if kind == "VAR":
        return "Random Range"
    preset = getattr(m, "preset", 'CURVE')
    if kind == "PMAP" and preset == 'CURVE':
        return PER_PLANT_CURVE
    return PRESET_NAMES.get(preset, PRESET_NAMES['CURVE'])


def entry_name(s, kind: str, owner: str, uid: int, param: str, index: int) -> str:
    """A map's name in the list: its preset, or a profile curve's axis."""
    if kind == "CURVE":
        entry = curve_entry(s, owner, uid, param)
        return entry[1] if entry is not None else "Curve"
    return map_name(kind, stored(s, kind, owner, uid, index))


def describe(s, owner: str, uid: int, param: str) -> str:
    """The map icon's tooltip."""
    pg = owner_pg(s, owner, uid)
    label = param_label(pg, param)
    found = entries(s, owner, uid, param)
    if not found:
        return f"{label}: not mapped. Click to add a map ({per_label(s, owner, uid, param)})"
    names = []
    for kind, i in found:
        names.append(entry_name(s, kind, owner, uid, param, i)
                     + ("" if entry_enabled(s, kind, owner, uid, param, i) else " (off)"))
    n = len(found)
    return f"{label}: {n} map{'s' if n != 1 else ''} ({', '.join(names)}). Click to edit"


# --- the Mapping list ------------------------------------------------------------
def _fold(s, owner: str, uid: int):
    key = f"{owner}:{uid}"
    for f in s.map_folds:
        if f.key == key:
            return f
    f = s.map_folds.add()
    f.key = key
    f["expanded"] = True
    return f


def rebuild_rows(s) -> None:
    """The Mapping list's rows: per owner with maps, a header, then -- unfolded
    -- one row per map, by parameter name. Keeps the selected map selected."""
    rows = s.map_rows
    keep = None
    if 0 <= s.map_rows_active < len(rows):
        r = rows[s.map_rows_active]
        keep = (r.kind, r.owner, r.owner_uid, r.param, r.index)
    rows.clear()
    pick = -1
    for owner, uid in owners(s):
        found = owner_entries(s, owner, uid)
        if not found:
            continue
        for kind, _index, param in found:
            if kind == "VAR":
                _keep_var(s, owner, uid, param)   # listed once: kept, even at Range 0
        fold = _fold(s, owner, uid)
        head = rows.add()
        head.kind, head.owner, head.owner_uid = "OWNER", owner, uid
        if keep is not None and keep[0] == "OWNER" and keep[1:3] == (owner, uid):
            pick = len(rows) - 1
        if not fold.expanded:
            continue
        pg = owner_pg(s, owner, uid)
        found = sorted(found, key=lambda e: (param_label(pg, e[2]), e[1]))
        for n, (kind, index, param) in enumerate(found):
            row = rows.add()
            row.kind, row.owner, row.owner_uid = kind, owner, uid
            row.param, row.index = param, index
            row.guides = "L" if n == len(found) - 1 else "T"
            row["enabled"] = entry_enabled(s, kind, owner, uid, param, index)
            if keep == (kind, owner, uid, param, index):
                pick = len(rows) - 1
    if pick >= 0:
        s["map_rows_active"] = pick
    elif s.map_rows_active >= len(rows):
        s["map_rows_active"] = max(len(rows) - 1, 0)


def row_selected(s) -> None:
    """A click on a list row: the Components tree selects its owner (the
    details follow on redraw)."""
    row = active_row(s)
    if row is None:
        return
    from . import structure
    structure.focus_owner(s, row.owner, int(row.owner_uid))


def follow_owner(s, owner: str, uid: int) -> None:
    """The Components tree picked an owner: select its first map in the list
    (its header while folded) -- unless one of its maps is selected already.
    Written past the update, so the tree is not re-focused in turn."""
    row = active_row(s)
    if row is not None and row.owner == owner and int(row.owner_uid) == uid:
        return
    rows = s.map_rows
    head = next((r for r, x in enumerate(rows)
                 if x.kind == "OWNER" and x.owner == owner and int(x.owner_uid) == uid), -1)
    if head < 0:
        return                                    # no maps on it
    first = head + 1
    pick = first if (first < len(rows) and rows[first].kind != "OWNER"
                     and rows[first].owner == owner and int(rows[first].owner_uid) == uid) \
        else head
    s["map_rows_active"] = pick


def select(s, owner: str, uid: int, param: str, kind: str = "", index: int = -2) -> None:
    """Select a parameter's map in the list (unfolding its owner)."""
    _fold(s, owner, uid)["expanded"] = True
    rebuild_rows(s)
    for r, row in enumerate(s.map_rows):
        if row.owner == owner and row.owner_uid == uid and row.param == param and \
                (not kind or (row.kind == kind and (index == -2 or row.index == index))):
            s["map_rows_active"] = r
            return


def active_row(s):
    rows = s.map_rows
    return rows[s.map_rows_active] if 0 <= s.map_rows_active < len(rows) else None


# --- presets, curves ---------------------------------------------------------------
def curve_node(obj, kind: str, owner: str, uid: int, m):
    if kind not in _CURVE_KIND or m is None:
        return None
    holder = uid if kind in _COMPONENT_KINDS else curves.GROWTH_MAP_OWNER
    return curves.get_component_curve(obj, holder, curves.map_curve_name(_CURVE_KIND[kind],
                                                                         m.uid))


def _ensure_curve(obj, kind: str, uid: int, m):
    holder = uid if kind in _COMPONENT_KINDS else curves.GROWTH_MAP_OWNER
    return curves.ensure_map_curve(obj, holder, curves.map_curve_name(_CURVE_KIND[kind], m.uid))


def _rotation(kind: str, m) -> bool:
    return kind == "IMAP" and m.target in pm.ROTATION_TARGETS


def apply_preset(obj, kind: str, owner: str, uid: int, m, pg, param: str) -> None:
    """Set a stored map's driver and curve to its preset."""
    additive = _rotation(kind, m) or (kind == "PMAP" and is_additive(pg, param))
    preset = m.preset
    if preset == 'RANGE':
        m.channel = "random"
        points = ((0.0, 0.0), (1.0, 1.0)) if additive else \
            ((0.0, 1.0 - m.range), (1.0, 1.0 + m.range))
    elif preset == 'HEIGHT':
        m.channel = "height"
        points = ((0.0, 0.75), (1.0, 0.25)) if additive else \
            ((0.0, 1.25), (0.5, 1.0), (1.0, 0.75))
    elif preset == 'AGE':
        m.channel = "age"
        points = ((0.0, 0.3), (1.0, 0.7)) if additive else \
            ((0.0, 0.75), (0.6, 1.05), (1.0, 1.15))
    else:
        points = ((0.0, 0.5), (1.0, 0.5)) if additive else ((0.0, 1.0), (1.0, 1.0))
    if kind == "PMAP":
        m.channel = "random"                     # a plant has one of everything else
    node = _ensure_curve(obj, kind, uid, m)
    curves.set_points(node, points, 'VECTOR' if preset in ('RANGE', 'CURVE') else 'AUTO')


def _locate(m):
    """(store kind, owner, uid) of a stored map PG, from its RNA path."""
    path = m.path_from_id()
    s = m.id_data.plantgen
    if ".branch_levels[" in path and ".maps[" in path:
        i = int(path.split(".branch_levels[")[1].split("]")[0])
        return "LMAP", LEVEL, int(s.branch_levels[i].uid)
    if ".components[" in path and (".maps[" in path or ".growth_maps[" in path):
        i = int(path.split(".components[")[1].split("]")[0])
        return ("IMAP" if ".maps[" in path else "FMAP"), COMPONENT, int(s.components[i].uid)
    if ".growth_maps[" in path:
        return "GMAP", PLANT, 0
    if ".param_maps[" in path:
        return "PMAP", m.owner, int(m.owner_uid)
    return None


def range_changed(m) -> None:
    """A Random Range map's range moved: redraw its curve from it."""
    if getattr(m, "preset", 'CURVE') != 'RANGE':
        return
    found = _locate(m)
    if found is None:
        return
    kind, owner, uid = found
    s = m.id_data.plantgen
    pg = owner_pg(s, owner, uid)
    param = _param_of(kind, m)
    apply_preset(m.id_data, kind, owner, uid, m, pg, param)


def preset_changed(m) -> None:
    found = _locate(m)
    if found is None:
        return
    kind, owner, uid = found
    s = m.id_data.plantgen
    apply_preset(m.id_data, kind, owner, uid, m, owner_pg(s, owner, uid), _param_of(kind, m))


def presets_for(owner: str, param: str) -> tuple:
    """The presets a parameter's map can start from: every one where a
    per-element evaluator reads it (or it had a +/-), Random Range and a
    Random curve where it maps once per plant."""
    if pm.hook_target(owner, param) is not None:
        return ('RANGE', 'HEIGHT', 'AGE', 'CURVE')
    return ('RANGE', 'CURVE')


def add_map(obj, owner: str, uid: int, param: str, preset: str):
    """Add a map on a parameter from a preset; returns (store kind, index)."""
    s = obj.plantgen
    pg = owner_pg(s, owner, uid)
    if pg is None:
        return None
    if preset == 'PROFILE':
        # the parameter's own profile curve: back on, or a first shape on it
        node = param_curve_node(s, owner, uid, param, create=True)
        if node is None:
            return None
        set_curve_enabled(s, owner, uid, param, True)
        if _flat(node):
            # a first shape: a decline along the axis; a growth curve that
            # fills out early instead of evenly
            curves.apply_shape(node, "s_curve" if _curve_key(node) in pm.CURVE_NEUTRAL
                               else "droop")
        select(s, owner, uid, param, "CURVE")
        return "CURVE", -1
    if preset == 'RANGE':
        var = variance_prop(pg, owner, param)
        if var is not None:
            set_var_enabled(s, owner, uid, param, True)      # a switched-off one: back on
            if abs(float(getattr(pg, var))) == 0.0:
                setattr(pg, var, 10.0 if is_additive(pg, param) else 0.2)
            _keep_var(s, owner, uid, param)
            select(s, owner, uid, param, "VAR")
            return "VAR", -1
    hook = pm.hook_target(owner, param)
    if hook is not None:
        kind, target = hook
        store = _store(s, kind, owner, uid)
        pool = ([x for lv in s.branch_levels for x in lv.maps] if kind == "LMAP"
                else list(store))
        m = store.add()
        m.uid = max((x.uid for x in pool), default=0) + 1
        m.target = target
    else:
        kind, store = "PMAP", s.param_maps
        m = store.add()
        m.uid = max((x.uid for x in store), default=0) + 1
        m.owner, m.owner_uid, m.param = owner, uid, param
        m.channel = "random"
        if is_additive(pg, param):
            m.angle = 10.0
    m["preset"] = {'CURVE': 0, 'RANGE': 1, 'HEIGHT': 2, 'AGE': 3}[preset]
    if _rotation(kind, m) and hasattr(m, "angle"):
        m.angle = 15.0
    apply_preset(obj, kind, owner, uid, m, pg, param)
    index = len(store) - 1
    select(s, owner, uid, param, kind, index)
    return kind, index


def convert_var(obj, owner: str, uid: int, param: str, preset: str):
    """A Random Range on a +/- value switched to another mode: the +/- goes to 0
    (its row with it) and a map of that preset takes its place. Returns
    (store kind, index) of the new map, or None."""
    if preset == 'RANGE':
        return "VAR", -1
    s = obj.plantgen
    pg = owner_pg(s, owner, uid)
    var = variance_prop(pg, owner, param)
    if var is not None:
        setattr(pg, var, type(getattr(pg, var))(0))
    for coll, (k, _rec) in ((s.var_mutes, _mute(s, owner, uid, param)),
                            (s.var_keeps, _keep(s, owner, uid, param))):
        if k >= 0:
            coll.remove(k)
    return add_map(obj, owner, uid, param, preset)


def remove(obj, row) -> None:
    """Remove a list row's map: a +/- value goes to 0, a stored map (and its
    curve) goes."""
    s = obj.plantgen
    pg = owner_pg(s, row.owner, row.owner_uid)
    if row.kind == "VAR":
        var = variance_prop(pg, row.owner, row.param)
        if var is not None:
            setattr(pg, var, 0)
        k, _kept = _mute(s, row.owner, row.owner_uid, row.param)
        if k >= 0:
            s.var_mutes.remove(k)
        k, _kept = _keep(s, row.owner, row.owner_uid, row.param)
        if k >= 0:
            s.var_keeps.remove(k)
    elif row.kind == "CURVE":
        node = param_curve_node(s, row.owner, row.owner_uid, row.param)
        if node is not None:
            _set_neutral(node)                     # neutral: no curve
        k, _kept = _curve_mute(s, row.owner, row.owner_uid, row.param)
        if k >= 0:
            s.curve_mutes.remove(k)
    else:
        store = _store(s, row.kind, row.owner, row.owner_uid)
        if store is not None and 0 <= row.index < len(store):
            m = store[row.index]
            holder = row.owner_uid if row.kind in _COMPONENT_KINDS else curves.GROWTH_MAP_OWNER
            curves.remove_named_component_curve(
                obj, holder, curves.map_curve_name(_CURVE_KIND[row.kind], m.uid))
            store.remove(row.index)
    rebuild_rows(s)


# --- per-plant evaluation ----------------------------------------------------------
def plant_index(obj) -> dict:
    """{(owner, uid, param): [PlantMap]} of the plant's per-plant maps."""
    s = obj.plantgen
    out: dict = {}
    for m in s.param_maps:
        if not m.enabled or not m.param:
            continue
        pg = owner_pg(s, m.owner, m.owner_uid)
        if pg is None:
            continue
        node = curves.get_component_curve(obj, curves.GROWTH_MAP_OWNER,
                                          curves.map_curve_name("pmap", m.uid))
        out.setdefault((m.owner, int(m.owner_uid), m.param), []).append(pm.PlantMap(
            param=m.param, preset=m.preset,
            curve=curves.to_curve(node) if node is not None else None,
            amount=float(m.amount), range=float(m.range), angle=float(m.angle),
            seed=int(m.seed), additive=is_additive(pg, m.param)))
    return out


def mapped(obj, index: dict, owner: str, uid: int, param: str, value):
    """`value` of a parameter under its per-plant maps (itself without any)."""
    maps = index.get((owner, int(uid), param))
    if not maps or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    s = obj.plantgen
    return pm.apply(value, maps, int(s.seed), owner_seed(s, owner, uid),
                    integer=isinstance(value, int))


def cval(c, param: str):
    """A component's render-side value under its per-plant maps."""
    value = getattr(c, param)
    obj = c.id_data
    s = obj.plantgen
    if not len(s.param_maps) or not any(m.owner == COMPONENT and m.owner_uid == c.uid
                                        and m.param == param for m in s.param_maps):
        return value
    return mapped(obj, plant_index(obj), COMPONENT, c.uid, param, value)
