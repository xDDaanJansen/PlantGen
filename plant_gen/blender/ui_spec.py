"""Generate the manual's panel reference from the panels themselves.

Every PlantGen panel is drawn into a recording layout -- the same draw code the
N-panel runs -- over a fixed set of states, so the reference lists every
control in the order and grouping the artist sees, with its label, tooltip,
type, default, range, unit, enum values, and whether it is mappable or a Debug
Tool. It cannot drift from the UI: regenerate after a panel change (Debug Tools
> Documentation > Regenerate Docs).

States: the plant panels on a plant with a rig, components and maps, then each
component type, the stems and a branch level, each kind of map and texture row,
with every gating toggle on; FlowerGen on a flower with every whorl on; the leaf
tool in both modes. A state is set with RAW property writes (no update fires:
nothing regrows, rebuilds or pushes an undo step) and every one is restored in
a `finally`. Controls a state reveals carry its tag ("component:bract"), so the
reference can say "shown for Bract" where a control is not universal.

The output records a digest of the UI source files (`source_digest`), which
`tools/build_docs.py` recomputes without Blender: a reference older than the
panels it describes is refused, never published.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import bpy

from . import bridge, effectors, mapping, operators

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "ui-reference.json"

def source_digest() -> str:
    """The UI source digest (`plant_gen/docs_digest.py`, shared with the docs build)."""
    from .. import docs_digest
    return docs_digest.source_digest()


# --- owners: what a property belongs to, in the artist's words -----------------

_OWNERS = {
    "PLANTGEN_PG_growth": "plant", "PLANTGEN_PG_branch_level": "level",
    "PLANTGEN_PG_component": "component", "PLANTGEN_PG_mask_entry": "mask",
    "PLANTGEN_PG_instance_map": "map", "PLANTGEN_PG_flower_map": "map",
    "PLANTGEN_PG_growth_map": "map", "PLANTGEN_PG_level_map": "map",
    "PLANTGEN_PG_param_map": "map", "PLANTGEN_PG_custom_attr": "custom attribute",
    "PLANTGEN_PG_class": "class", "PLANTGEN_PG_trait_column": "traits",
    "PLANTGEN_PG_trait_row": "traits", "PLANTGEN_PG_trait_cell": "traits",
    "PLANTGEN_PG_flower": "flower", "PLANTGEN_PG_leaf_gen": "leaf",
    "PLANTGEN_PG_leaf": "leaf tool", "PLANTGEN_PG_variations": "variations",
    "PLANTGEN_PG_effector": "effector", "PLANTGEN_PG_stem_material": "material",
    "PLANTGEN_PG_leaf_material": "material", "Scene": "scene", "Object": "object",
}


def _owner(data) -> str:
    if isinstance(data, bpy.types.NodeSocket):
        return "shader"
    name = type(data).__name__
    return _OWNERS.get(name, name)


_TITLE_RE = re.compile(r"[0-9:.!?\-=>/]|--")


_MINOR = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "per",
          "the", "to", "&"}


def _title(text) -> str | None:
    """A group title, normalised: counts and parentheticals dropped ("Traits  (2
    properties)" -> "Traits"); None for a sentence or a status line. The
    panels write their titles in title case ("Internode Elongation", "Along
    the Row"); a status line is a sentence ("Stems not welded", "Evaluated
    per plant"), and a datablock's own name is never a title."""
    if not text:
        return None
    text = re.sub(r"\s*\(.*\)\s*$", "", str(text)).strip()
    if not text or len(text) > 32 or _TITLE_RE.search(text):
        return None
    for word in text.split()[1:]:
        if word.lower() not in _MINOR and word[:1].islower():
            return None
    for coll in (bpy.data.materials, bpy.data.objects, bpy.data.collections,
                 bpy.data.node_groups, bpy.data.images):
        if text in coll:
            return None
    return text


_STATE_SUFFIX = re.compile(r"\s*\([^)]*[0-9:][^)]*\)\s*$")
"""A parenthetical that reports state -- "Sides (35, from the petals)",
"Flip Normals (welded: follows Petals)" -- dropped from a label; a unit like
"Start (m)" stays."""


_PANEL_PROP_TITLES = {"structure_level_open": "Level settings",
                      "structure_component_open": "Component settings",
                      "mapping_details_open": "Map settings"}
_EXTRA = {"plantgen.map_param": "map", "plantgen.toggle_debug_view": "debug view",
          "plantgen.leaf_edit_variance": "variance", "plantgen.edit_variance": "variance",
          "plantgen.flower_edit_variance": "variance"}


def _rna_record(data, name: str, shown: str | None) -> dict:
    rna = data.bl_rna.properties[name]
    rec = {"kind": "prop", "owner": _owner(data),
           "id": data.name if isinstance(data, bpy.types.NodeSocket) else name,
           "label": _STATE_SUFFIX.sub("", shown) if shown else (
               data.name if isinstance(data, bpy.types.NodeSocket) else rna.name),
           "name": data.name if isinstance(data, bpy.types.NodeSocket) else rna.name,
           "desc": rna.description, "type": rna.type.lower()}
    if getattr(rna, "subtype", "NONE") not in ("NONE", ""):
        rec["subtype"] = rna.subtype.lower()
    unit = getattr(rna, "unit", "NONE")
    if unit not in ("NONE", None):
        rec["unit"] = unit.lower()
    if rna.type in ("FLOAT", "INT"):
        arr = getattr(rna, "is_array", False) and rna.array_length > 0
        rec["default"] = (list(rna.default_array) if arr else rna.default)
        for key, attr in (("min", "hard_min"), ("max", "hard_max"),
                          ("soft_min", "soft_min"), ("soft_max", "soft_max")):
            v = getattr(rna, attr)
            if abs(v) < 1e12:
                rec[key] = round(v, 6) if isinstance(v, float) else v
    elif rna.type == "BOOLEAN":
        rec["default"] = bool(rna.default)
    elif rna.type == "ENUM":
        rec["items"] = [[e.identifier, e.name, e.description] for e in rna.enum_items]
        rec["flag"] = bool(rna.is_enum_flag)
        if rec["items"]:
            rec["default"] = (sorted(rna.default_flag) if rna.is_enum_flag else rna.default)
    elif rna.type == "STRING":
        rec["default"] = rna.default
    elif rna.type == "POINTER":
        rec["points_to"] = rna.fixed_type.identifier
    return rec


class _Rec:
    """A stand-in UILayout that records what a panel draws (it answers every
    layout call; the ones that carry a control append an entry)."""

    def __init__(self, node, header=False):
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_header", header)

    # sub-layouts share the node they belong to
    def _child(self):
        return _Rec(self._node, self._header)

    def __setattr__(self, name, value):
        pass

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._child()

    def box(self, *a, **k):
        # a box is a visual group: its first title label becomes a heading
        return _Rec(self._node, self._header)

    def panel(self, idname, *a, **k):
        fold = _node("fold", None)
        fold["closed"] = bool(k.get("default_closed", False))
        fold["_fold_id"] = idname
        self._node["children"].append(fold)
        return _Rec(fold, header=True), _Rec(fold)

    def panel_prop(self, data, prop, *a, **k):
        fold = _node("fold", _PANEL_PROP_TITLES.get(prop))
        fold["_fold_id"] = prop
        self._node["children"].append(fold)
        return _Rec(fold, header=True), _Rec(fold)

    def icon(self, *a, **k):
        return 0

    def label(self, *a, text="", **k):
        text = k.get("text", a[0] if a else text)
        t = _title(text)
        if self._header:
            if t and not self._node["title"]:
                self._node["title"] = t
        elif t:
            kids = self._node["children"]
            if not (kids and kids[-1].get("kind") == "heading" and kids[-1]["text"] == t):
                kids.append({"kind": "heading", "text": t})
        return self._child()

    def _add(self, entry):
        if not self._header:
            self._node["children"].append(entry)

    def _attach(self, extra) -> bool:
        """An icon-only button beside a control (its map, curve, variance or
        debug view): recorded on that control, not as a control of its own."""
        kids = self._node["children"]
        if kids and kids[-1].get("kind") in ("prop", "op", "picker"):
            ex = kids[-1].setdefault("extras", [])
            if extra not in ex:
                ex.append(extra)
            return True
        return False

    def prop(self, data, name, *a, **k):
        if self._header:
            return self._child()
        try:
            rec = _rna_record(data, name, k.get("text"))
        except (AttributeError, KeyError):
            return self._child()
        if rec["owner"] in ("plant", "level", "component") and mapping.is_mappable(data, name):
            rec["mappable"] = True
        self._add(rec)
        return self._child()

    def prop_search(self, data, name, *a, **k):
        return self.prop(data, name, **{x: v for x, v in k.items() if x == "text"})

    def prop_enum(self, *a, **k):
        return self._child()

    def template_ID(self, data, name, *a, **k):
        try:
            rna = data.bl_rna.properties[name]
            self._add({"kind": "picker", "owner": _owner(data), "id": name,
                       "label": rna.name, "desc": rna.description,
                       "points_to": rna.fixed_type.identifier})
        except (AttributeError, KeyError):
            pass
        return self._child()

    def operator(self, idname, *a, **k):
        text = k.get("text")
        if text == "" and idname in _EXTRA:
            self._attach(_EXTRA[idname])
            return self._child()
        try:
            mod, op = idname.split(".")
            rna = getattr(getattr(bpy.ops, mod), op).get_rna_type()
            name, desc = rna.name, rna.description
        except (AttributeError, ValueError, KeyError):
            name, desc = idname, ""
        if text == "":
            return self._child()           # list +/- / move arrows: part of the list
        if text is not None and (re.search(r"[0-9]", text) or text in ("None", "Custom")):
            text = None
        self._add({"kind": "op", "id": idname, "label": text or name, "desc": desc})
        return self._child()

    def menu(self, menu, *a, **k):
        cls = getattr(bpy.types, menu, None)
        text = k.get("text")
        label = getattr(cls, "bl_label", menu) if cls is not None else menu
        doc = inspect.getdoc(cls) if cls is not None else ""
        if text == "":
            return self._child()
        self._add({"kind": "menu", "id": menu, "label": label,
                   "desc": (doc or "").split("\n\n")[0].replace("\n", " ")})
        return self._child()

    def popover(self, *a, **k):
        self._attach("curve")
        return self._child()

    def template_list(self, listtype, *a, **k):
        cls = getattr(bpy.types, listtype, None)
        doc = inspect.getdoc(cls) if cls is not None else ""
        self._add({"kind": "list", "id": listtype,
                   "desc": (doc or "").split("\n\n")[0].replace("\n", " ")})
        return self._child()

    def template_curve_mapping(self, *a, **k):
        self._add({"kind": "curve", "id": "curve"})
        return self._child()

    def template_curveprofile(self, *a, **k):
        self._add({"kind": "curve", "id": "profile curve"})
        return self._child()


def _node(kind, title):
    return {"node": kind, "title": title, "children": []}


class _Proxy:
    """A panel instance stand-in: the panel's own methods, a recording layout."""

    def __init__(self, cls, layout):
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "layout", layout)

    def __getattr__(self, name):
        raw = inspect.getattr_static(self._cls, name)
        value = getattr(self._cls, name)
        import types
        return types.MethodType(value, self) if isinstance(raw, types.FunctionType) else value

    def __setattr__(self, name, value):
        pass


# --- panels, in the order the tab shows them --------------------------------------

def _panel_classes():
    from . import ui, ui_flower
    mods = [ui, ui_flower]
    try:
        from . import ui_leaf
        mods.append(ui_leaf)
    except ImportError:
        pass
    out = []
    for mod in mods:
        for cls in getattr(mod, "_classes", ()):
            if (isinstance(cls, type) and issubclass(cls, bpy.types.Panel)
                    and getattr(bpy.types, cls.__name__, None) is cls
                    and getattr(cls, "bl_region_type", "") == "UI"):
                out.append(cls)
    return out


def _visible(cls, ctx) -> bool:
    try:
        if hasattr(cls, "poll") and not cls.poll(ctx):
            return False
    except Exception:  # noqa: BLE001
        return False
    parent = getattr(cls, "bl_parent_id", "")
    return _visible(getattr(bpy.types, parent), ctx) if parent else True


def _draw_tree(cls, classes, ctx) -> dict:
    node = _node("subpanel" if getattr(cls, "bl_parent_id", "") else "panel", cls.bl_label)
    node["closed"] = "DEFAULT_CLOSED" in getattr(cls, "bl_options", set())
    rec = _Rec(node)
    proxy = _Proxy(cls, rec)
    if hasattr(cls, "draw_header"):
        hdr = _Rec(node)
        try:
            cls.draw_header(_Proxy(cls, hdr), ctx)
        except Exception:  # noqa: BLE001
            pass
    cls.draw(proxy, ctx)
    for sub in classes:
        if getattr(sub, "bl_parent_id", "") == cls.bl_idname and _visible(sub, ctx):
            node["children"].append(_draw_tree(sub, classes, ctx))
    return node


# --- merging states ------------------------------------------------------------------

def _key(item):
    if "node" in item:
        return ("node", item["node"], item.get("title"))
    kind = item["kind"]
    if kind == "heading":
        return ("heading", item["text"])
    if kind == "op":
        return ("op", item["id"], item["label"])
    return (kind, item.get("owner"), item.get("id"))


def _merge(base: list, new: list, tag, debug: bool) -> None:
    """Merge `new` into `base` in draw order, tagging what each state showed."""
    index = {_key(x): i for i, x in enumerate(base)}
    last = -1
    for item in new:
        k = _key(item)
        if k in index:
            i = index[k]
            target = base[i]
            if "children" in item:
                _merge(target["children"], item["children"], tag, debug)
            for ex in item.get("extras", ()):
                target.setdefault("extras", [])
                if ex not in target["extras"]:
                    target["extras"].append(ex)
            last = i
        else:
            last += 1
            base.insert(last, _tag_all(item, tag, debug))
            index = {_key(x): j for j, x in enumerate(base)}
            continue
        _tag(base[last], tag, debug)


def _tag(item, tag, debug):
    if tag:
        item.setdefault("tags", [])
        if tag not in item["tags"]:
            item["tags"].append(tag)
    if not debug:
        item["seen_off"] = True


def _tag_all(item, tag, debug):
    _tag(item, tag, debug)
    for child in item.get("children", ()):
        _tag_all(child, tag, debug)
    return item


# --- states ----------------------------------------------------------------------------

class _Raw:
    """Raw property writes (no update callback fires) restored on exit."""

    def __init__(self):
        self._saved = []

    def set(self, pg, name, value):
        missing = name not in pg.keys()
        self._saved.append((pg, name, None if missing else pg[name], missing))
        pg[name] = value

    def restore(self):
        for pg, name, value, missing in reversed(self._saved):
            try:
                if missing:
                    del pg[name]
                else:
                    pg[name] = value
            except (KeyError, TypeError):
                pass
        self._saved.clear()


def _enum_value(pg, name, ident):
    for e in pg.bl_rna.properties[name].enum_items:
        if e.identifier == ident:
            return e.value
    raise KeyError(ident)


_PLANT_TOGGLES = {"circumnutation": 0.2, "flower_heliotropism": 0.2,
                  "leaf_heliotropism": 0.2, "stem_weld": True, "animate_day": True,
                  "wind_enable": True, "wind_secondary_enable": True}
_FLOWER_TOGGLES = {"petal_enabled": True, "petal_tube_enabled": True,
                   "sepal_enabled": True, "sepal_tube_enabled": True,
                   "disc_enabled": True, "receptacle_enabled": True,
                   "petal_layers": 2, "sepal_layers": 2}


_STATE_PANEL = {"component": "PLANTGEN_PT_structure", "level": "PLANTGEN_PT_structure",
                "map": "PLANTGEN_PT_mapping", "texture": "PLANTGEN_PT_texturing"}
"""Which panel a tagged state varies -- the only one drawn in it, so a tag
never lands on a panel the state does not change."""


def _plant_states(raw, plant, levels_plant):
    """Yield (object, tag) for every plant-panel state; the caller draws."""
    s = plant.plantgen
    for name, value in _PLANT_TOGGLES.items():
        if name in s.bl_rna.properties:
            raw.set(s, name, value)
    yield plant, None
    # each component, as each organ type
    for i, comp in enumerate(s.components):
        raw.set(s, "structure_focus", _enum_value(s, "structure_focus", "COMPONENT"))
        raw.set(s, "components_active", i)
        raw.set(s, "structure_component_open", True)
        if comp.type == "flower":
            yield plant, "component:flower"
            continue
        for kind in ("leaf", "bract", "bud", "fruit"):
            raw.set(comp, "type", _enum_value(comp, "type", kind))
            yield plant, f"component:{kind}"
    # each level of a plant with branch levels
    ls = levels_plant.plantgen
    for i, lv in enumerate(ls.branch_levels):
        raw.set(ls, "structure_focus", _enum_value(ls, "structure_focus", "LEVEL"))
        raw.set(ls, "branch_levels_active", i)
        raw.set(ls, "structure_level_open", True)
        yield levels_plant, "level:stems" if lv.parent_uid == 0 else "level:branches"
    # each kind of map, each kind of texture row
    for i, row in enumerate(s.map_rows):
        if row.kind != "OWNER":
            raw.set(s, "map_rows_active", i)
            raw.set(s, "mapping_details_open", True)
            yield plant, "map:" + row.kind.lower()
    for i, row in enumerate(s.texture_rows):
        raw.set(s, "texture_rows_active", i)
        yield plant, "texture:" + row.kind.lower()


def generate() -> dict:
    classes = _panel_classes()
    tops = [c for c in classes if not getattr(c, "bl_parent_id", "")
            and getattr(c, "bl_category", "") == "PlantGen"]
    tops.sort(key=lambda c: getattr(c, "bl_order", 0))
    plants = [o for o in bpy.data.objects if bridge.is_plant(o)]
    plant = max(plants, key=lambda o: (len(o.plantgen.components), len(o.plantgen.map_rows)))
    levels_plant = max(plants, key=lambda o: len(o.plantgen.branch_levels))
    from . import flower, leaf
    flowers = [o for o in bpy.data.objects if flower.is_flower(o)]
    cutouts = [o for o in bpy.data.objects if leaf.is_cutout_leaf(o)]

    prefs = bpy.context.preferences.addons[operators._ADDON].preferences
    keep_debug = prefs.debug_tools
    tree: dict = {}
    order: list = []
    raw = _Raw()

    def draw_all(obj, tag, debug, only=None):
        with bpy.context.temp_override(active_object=obj, object=obj):
            ctx = bpy.context
            for cls in tops:
                if only and cls.__name__ not in only:
                    continue
                if not _visible(cls, ctx):
                    continue
                try:
                    node = _draw_tree(cls, classes, ctx)
                except Exception as exc:  # noqa: BLE001
                    print(f"[ui_spec] {cls.__name__} on {obj.name}: {exc}")
                    continue
                key = cls.bl_label
                if key not in tree:
                    tree[key] = _tag_all(node, tag, debug)
                    order.append(key)
                else:
                    _tag(tree[key], tag, debug)
                    _merge(tree[key]["children"], node["children"], tag, debug)

    # The panels whose contents follow a selection are drawn once per state
    # (tagged); the base, untagged pass draws every other panel.
    by_selection = {"PLANTGEN_PT_structure", "PLANTGEN_PT_mapping", "PLANTGEN_PT_texturing"}
    plant_panels = {c.__name__ for c in tops} - by_selection - {
        "PLANTGEN_PT_flower", "PLANTGEN_PT_leaf"}

    def set_debug(value):
        try:
            prefs["debug_tools"] = value          # raw: switching off turns no view off
        except (TypeError, KeyError):
            prefs.debug_tools = value

    try:
        for debug in (True, False):
            set_debug(debug)
            try:
                for obj, tag in _plant_states(raw, plant, levels_plant):
                    draw_all(obj, tag, debug,
                             only=plant_panels if tag is None
                             else {_STATE_PANEL[tag.split(":")[0]]})
            finally:
                raw.restore()
            if flowers:
                fl = flowers[0]
                try:
                    for name, value in _FLOWER_TOGGLES.items():
                        raw.set(fl.plantgen_flower, name, value)
                    draw_all(fl, None, debug, only={"PLANTGEN_PT_flower"})
                finally:
                    raw.restore()
            # The Plant Effector panel draws for an effector Empty: an existing
            # one, else a temporary one never linked to a scene, once per kind.
            eff = next((o for o in bpy.data.objects if effectors.is_effector(o)), None)
            temp = None
            if eff is None:
                temp = eff = bpy.data.objects.new("_ui_spec_effector", None)
                eff[effectors._MARKER] = True
            try:
                for kind in [item[0] for item in effectors._KIND_ITEMS]:
                    raw.set(eff.plantgen_effector, "kind", _enum_value(
                        eff.plantgen_effector, "kind", kind))
                    draw_all(eff, f"effector:{kind.lower()}", debug,
                             only={"PLANTGEN_PT_effector"})
            finally:
                raw.restore()
                if temp is not None:
                    bpy.data.objects.remove(temp, do_unlink=True)
            sl = bpy.context.scene.plantgen_leaf
            for mode in ("generate", "cutout"):
                try:
                    raw.set(sl, "mode", _enum_value(sl, "mode", mode))
                    target = cutouts[0] if (mode == "cutout" and cutouts) else plant
                    draw_all(target, f"leaf:{mode}", debug, only={"PLANTGEN_PT_leaf"})
                finally:
                    raw.restore()
    finally:
        set_debug(keep_debug)

    panels = [tree[k] for k in order]
    panels.sort(key=lambda n: next((getattr(c, "bl_order", 0) for c in tops
                                    if c.bl_label == n["title"]), 0))
    for p in panels:
        _finish(p)
    return {"version": 1, "source_digest": source_digest(),
            "blender": bpy.app.version_string, "panels": panels}


def _finish(item, under_debug=False) -> None:
    """Mark Debug Tools (never seen with the preference off) and drop the
    working keys."""
    debug = under_debug or not item.pop("seen_off", False)
    if debug:
        item["debug"] = True
    item.pop("_fold_id", None)
    if item.get("tags") is not None and not item["tags"]:
        item.pop("tags")
    for child in item.get("children", ()):
        _finish(child, debug)


def write(path: Path = OUT_PATH, staged: bool = False) -> Path:
    """Write the reference. `staged`: first build the examples it draws from
    (`stage_examples`) -- for a factory-fresh Blender; otherwise it draws from
    the plants already in the file."""
    if staged:
        stage_examples()
    doc = generate()
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_isolated(timeout: float = 300.0) -> Path:
    """Write the reference from a factory-fresh Blender in a subprocess, on
    staged examples: the same reference whatever file is open, in seconds,
    and nothing in the open file is touched."""
    import subprocess
    # the add-on registered from the repo, with an add-on entry of its own so
    # its preferences (Debug Tools, which the reference switches) exist
    repo = str(ROOT.parent)
    code = (f"import sys, bpy; sys.path.insert(0, {repo!r}); "
            "bpy.context.preferences.addons.new().module = 'plant_gen'; "
            "import plant_gen; plant_gen.register(); "
            "from plant_gen.blender import ui_spec; "
            "print('[ui_spec] wrote', ui_spec.write(staged=True))")
    out = subprocess.run([bpy.app.binary_path, "-b", "--factory-startup",
                          "--python-expr", code],
                         capture_output=True, text=True, timeout=timeout)
    if "[ui_spec] wrote" not in out.stdout:
        tail = "\n".join((out.stdout + out.stderr).splitlines()[-15:])
        raise RuntimeError(f"ui_spec subprocess failed:\n{tail}")
    return OUT_PATH


# --- staged examples -------------------------------------------------------------

# One map of each kind: (owner, which, parameter, preset) -- `which` picks the
# owner: "branches" (the branch level), "flower" / "leaf" (the components),
# "" (the plant). Several candidates per kind, the first that takes is kept.
_STAGED_MAPS = {
    "VAR": [("LEVEL", "branches", "start_pitch", 'RANGE')],
    "CURVE": [("COMPONENT", "flower", "pedicel_length", 'PROFILE'),
              ("COMPONENT", "flower", "inflorescence_internode_scale", 'PROFILE')],
    "LMAP": [("LEVEL", "branches", "start_length_scale", 'HEIGHT')],
    "GMAP": [("PLANT", "", "gravitropism", 'HEIGHT')],
    "FMAP": [("COMPONENT", "flower", "pedicel_angle", 'HEIGHT')],
    "IMAP": [("COMPONENT", "leaf", "size", 'HEIGHT')],
    "PMAP": [("PLANT", "", "light_seeking", 'RANGE'), ("PLANT", "", "circumnutation", 'RANGE')],
}


def _clear_factory_scene() -> None:
    """Remove the factory file's own datablocks (Cube, Light, Camera,
    Material, Collection...): `_title` never takes a datablock's name for a
    heading, so a stray "Light" object or "Material" material would hide the
    folds of those names."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.collections):
        for block in list(coll):
            coll.remove(block)


def _staged_mesh(name, verts, faces, material):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.materials.append(material)
    return bpy.data.objects.new(name, me)


def stage_examples():
    """Build, in this (factory-fresh) file, one example of every state the
    panels are drawn in, so the reference never depends on the open file: a
    new plant (stems and a branch level, Add Plant's flower component, set to
    a Panicle so its inflorescence controls show) with a leaf component
    placed on organs, nodes and at the base, each with a mask; leaf and
    flower meshes in leaf-shader materials; a stem material with a normal map
    and sheen; one map of every kind; welded stems and a rig; a FlowerGen
    flower with its material; a cut-out leaf with its material; the Leaf
    Generator with an atlas and a margin. Names are neutral ("Ref ...") so no
    datablock name collides with a panel label. Returns the plant."""
    import bmesh

    from . import leaf, leaf_material, stem_material, structure, ui_materials
    ctx = bpy.context
    _clear_factory_scene()
    plant = bridge.create_plant(ctx, name="Ref Plant", preset=None)
    s = plant.plantgen
    s.live_update = False
    s.leaf_asset_collection.objects.link(_staged_mesh(
        "Ref Leaf", [(0, 0, 0), (0.008, 0.025, 0), (0, 0.07, 0), (-0.008, 0.025, 0)],
        [(0, 1, 2, 3)], leaf_material.new_material("Ref Leaf Material")))
    cup = bpy.data.meshes.new("Ref Flower")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=8, radius1=0.004, radius2=0.016,
                          depth=0.014)
    bm.to_mesh(cup)
    bm.free()
    cup.materials.append(leaf_material.new_material("Ref Flower Material"))
    s.flower_asset_collection.objects.link(bpy.data.objects.new("Ref Flower", cup))

    # welded stems and a rig, built on the plain new plant: on the Panicle
    # below its many flower bones make Build Rig take minutes
    bridge.regrow(plant, weld=bridge.WELD_NOW)
    with ctx.temp_override(active_object=plant, object=plant):
        bpy.ops.plantgen.build_rig()

    flower_c = next(c for c in s.components if c.type == 'flower')
    flower_c.inflorescence_kind = 'PANICLE'
    flower_c.inflorescence_orientation_strength = 0.5
    s.structure_focus, s.structure_place = 'PLANT', -1
    with ctx.temp_override(active_object=plant, object=plant):
        bpy.ops.plantgen.component_add(type="leaf", place=True)
    leaf_c = s.components[-1]
    leaf_c.collection = s.leaf_asset_collection
    leaf_c.placement = {'ORGAN', 'NODE', 'BASAL'}
    branches = next(lv for lv in s.branch_levels if lv.parent_uid != 0)
    # a mask on the leaf component and on the branch level
    structure.rebuild_tree(s)
    with ctx.temp_override(active_object=plant, object=plant):
        s.structure_focus = 'COMPONENT'
        s.components_active = len(s.components) - 1
        bpy.ops.plantgen.mask_add(channel="height")
        s.structure_focus = 'LEVEL'
        s.branch_levels_active = list(s.branch_levels).index(branches)
        bpy.ops.plantgen.mask_add(channel="height", owner="LEVEL")

    mat = stem_material.new_material()
    s.branch_levels[0].material = mat
    mat.plantgen_stem.normal_image = bpy.data.images.new("Ref Normal", 8, 8)
    node = stem_material.group_node(mat)
    if node is not None and "Sheen" in node.inputs:
        node.inputs["Sheen"].default_value = 0.5

    owners = {"branches": branches, "flower": flower_c, "leaf": leaf_c}
    made = set()
    for kind, candidates in _STAGED_MAPS.items():
        for owner, which, param, preset in candidates:
            uid = int(owners[which].uid) if which else 0
            got = mapping.add_map(plant, owner, uid, param, preset)
            if got is not None and got[0] == kind:
                made.add(kind)
                break
    missing = set(_STAGED_MAPS) - made
    if missing:
        print(f"[ui_spec] staged maps missing: {sorted(missing)}")
    mapping.rebuild_rows(s)

    bridge.regrow(plant)
    structure.rebuild_tree(s)
    ui_materials.rebuild_rows(s)

    bpy.ops.plantgen.add_flower()
    flower_obj = ctx.view_layer.objects.active
    if flower_obj is not None and hasattr(flower_obj, "plantgen_flower"):
        flower_obj.plantgen_flower.material = leaf_material.new_material("Ref Petal Material")

    cut = bpy.data.objects.new("Ref Cutout", bpy.data.meshes.new("Ref Cutout"))
    cut[leaf._LEAF_TAG] = True
    cut[leaf._LEAF_CUT_TAG] = True
    ctx.scene.collection.objects.link(cut)
    cut_mat = leaf_material.new_material("Ref Cutout Material")
    leaf_material.assign(cut, cut_mat)
    cut_mat.plantgen_leaf.normal_image = bpy.data.images.new("Ref Leaf Normal", 8, 8)
    leaf.profile_carrier(create=True)            # the Profile Curve widget's host
    sl = ctx.scene.plantgen_leaf
    sl.image = bpy.data.images.new("Ref Atlas", 64, 64, alpha=True)
    holder = leaf.active_holder(ctx)
    margins = holder.bl_rna.properties["gen_margin"].enum_items
    if len(margins) > 1:
        holder.gen_margin = margins[1].identifier
    holder.gen_noise = 0.1
    return plant
