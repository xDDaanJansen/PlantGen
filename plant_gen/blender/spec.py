"""Generate the LLM-facing parameter spec from the model itself, in Blender.

A build-time generator (not part of normal add-on use) that reads the SAME
sources the add-on runs on, so the spec cannot drift from the model:

  * the `MorphologyProfile` dataclasses -- structure, nesting, types, defaults
    (walked the way `growth/serialize.py` walks them);
  * the panel property RNA -- ranges, useful (soft) ranges, units, tooltips and
    enum values, reached through `bridge._FIELDS`;
  * the shared kind-capabilities table in `growth/morphology.py` -- which kinds
    make which controls inert;
  * the curve-shape vocabulary in `blender/curves.py` -- so a curve field is a
    small named-shape choice, not a point list.

It emits one JSON document (`docs/llm/plantgen-spec.json`). An LLM/agent reads it
to author a full preset; the human reference is rendered from the same JSON
later. A companion notes file supplies the endpoint-visual range descriptions a
tooltip does not carry (Step 2); it is merged here when present.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from pathlib import Path

import bpy

from ..growth import morphology, serialize
from ..growth.curves import Curve
from ..growth.morphology import MorphologyProfile
from . import bridge
from . import curves as ui_curves

SCHEMA_VERSION = 1

# A range endpoint at or beyond this is Blender's FLT_MAX sentinel -- an
# unbounded property, not a real limit. (Measured 3.4e38 on leaf_size/tip_radius.)
_BOUND_SANITY = 1e12

SPEC_DIR = Path(__file__).parent.parent / "docs" / "llm"
SPEC_PATH = SPEC_DIR / "plantgen-spec.json"
NOTES_PATH = SPEC_DIR / "field-notes.json"
"""Hand-authored endpoint-visual descriptions, keyed by dotted field path, that
override the harvested tooltip for look-shaping fields (Step 2)."""


# --- the panel <-> profile bridge, inverted -------------------------------

def _panel_by_path() -> dict:
    """profile path -> (panel property name, to-profile converter or None), from
    `bridge._FIELDS`. The one map that ties a profile field to the RNA that
    carries its range/unit/tooltip/enum."""
    out = {}
    for prop, path, *converters in bridge._FIELDS:
        out[path] = (prop, converters[0] if converters else None)
    return out


def _rna_props():
    return bpy.types.Object.bl_rna.properties["plantgen"].fixed_type.properties


LEVELS_PATH = "levels"
"""The profile's list of `BranchLevel`s. Its fields are not in `bridge._FIELDS`:
each level is one `PLANTGEN_PG_branch_level` in the `branch_levels`
collection, mapped field for field by `bridge._LEVEL_FIELDS`."""


def _level_panel() -> tuple[dict, object]:
    """(profile path -> (level property, None), the level property RNA) for the
    fields of one branching level, keyed `levels[].<field>` like the walk."""
    panel = {f"{LEVELS_PATH}[].{fld}": (prop, None)
             for prop, fld in (("name", "name"), *bridge._LEVEL_FIELDS)}
    return panel, _rna_props()["branch_levels"].fixed_type.properties


# --- dataclass walk (mirrors serialize.py) --------------------------------

_HINTS: dict[type, dict] = {}


def _hints(cls) -> dict:
    cached = _HINTS.get(cls)
    if cached is None:
        cached = _HINTS[cls] = typing.get_type_hints(cls)
    return cached


def _unwrap_optional(hint):
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return None


def _collection_of(hint):
    """("list" | "dict", element dataclass) for a list/tuple of dataclasses or a
    str-keyed dict of them (the branching levels, mask stacks, map lists), else
    None. Such a field is a container of records, not a leaf value."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (list, tuple) and args:
        kind, elem = "list", args[0]
    elif origin is dict and len(args) == 2:
        kind, elem = "dict", args[1]
    else:
        return None
    if dataclasses.is_dataclass(elem) and elem is not Curve:
        return kind, elem
    return None


def _field_default(f: dataclasses.Field):
    """A field's class default (running its factory), or MISSING for a required
    field -- used when walking an element type that has no default instance
    (a `MaskEntry` needs its channel, a `ParamMap` its target and channel)."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _default_instance(cls):
    """`cls()` if every field has a default, else None (walk its field defaults)."""
    try:
        return cls()
    except TypeError:
        return None


# --- curve helpers --------------------------------------------------------

_SHAPE_KEYS = [key for key, _label, _pts, _smooth in ui_curves.SHAPES]
_SHAPE_LABELS = {key: label for key, label, _pts, _smooth in ui_curves.SHAPES}
_SHAPE_POINTS = {key: [[round(x, 4), round(y, 4)] for x, y in pts]
                 for key, _label, pts, _smooth in ui_curves.SHAPES}
_SHAPE_SMOOTH = {key: bool(smooth) for key, _label, _pts, smooth in ui_curves.SHAPES}
# profile attr -> (curve key, x-axis caption), from the panel curve registry.
_CURVE_BY_ATTR = {spec.attr: (spec.key, spec.x_label) for spec in ui_curves.CURVE_SPECS}


def _default_shape(curve: Curve) -> str | None:
    """The named shape a default curve matches, by comparing authoring points to
    `SHAPES` (rounded). None -> not a named shape (its points are reported)."""
    pts = tuple((round(x, 4), round(y, 4)) for x, y in curve.points)
    for key, _label, shape_pts, _smooth in ui_curves.SHAPES:
        if pts == tuple((round(x, 4), round(y, 4)) for x, y in shape_pts):
            return key
    return None


# --- field records --------------------------------------------------------

def _clean_bound(v):
    """A range endpoint, or None if it is the FLT_MAX 'unbounded' sentinel."""
    return None if (v is None or abs(v) >= _BOUND_SANITY) else round(float(v), 6)


def _numeric_meta(record, rna, to_profile) -> None:
    """Hard range, useful (soft) range and unit for a float/int field, in MODEL
    units. Where a monotonic converter maps the panel value to a different model
    value (only Stem Stiffness does), the range is converted too, so the number
    is the one that goes in the preset JSON -- not the panel's own."""
    lo, hi = rna.hard_min, rna.hard_max
    slo, shi = rna.soft_min, rna.soft_max
    if to_profile is not None:
        try:
            lo, hi, slo, shi = (to_profile(lo), to_profile(hi),
                                to_profile(slo), to_profile(shi))
        except (TypeError, ValueError):
            pass
    record["min"] = _clean_bound(lo)
    record["max"] = _clean_bound(hi)
    useful_lo, useful_hi = _clean_bound(slo), _clean_bound(shi)
    # Only worth stating when it narrows the hard range -- the useful number an
    # LLM should reach for rather than the extreme.
    if (useful_lo, useful_hi) != (record["min"], record["max"]):
        record["useful_range"] = [useful_lo, useful_hi]
    if rna.unit and rna.unit != "NONE":
        record["unit"] = rna.unit.lower()


def _enum_values(path: str, rna, to_profile) -> list:
    """The MODEL enum values (what a preset stores), not the panel identifiers.
    Inflorescence Kind is taken from the model directly -- the RNA enum also
    lists the UI-only COMPOUND_* presets, which are not real kinds."""
    if path == "inflorescence.kind":
        return list(morphology.KIND_TOPOLOGY)
    conv = to_profile or (lambda v: v)
    seen, out = set(), []
    for item in rna.enum_items:
        v = conv(item.identifier)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _note_key(v) -> str:
    """`value_notes` is keyed by the profile value, but a JSON object key must be a
    string -- so a profile value like the lifespan `None` cannot survive a JSON
    round-trip as a key (json.dumps coerces it to "null"). Coerce it here, exactly
    as json.dumps would, so generate_spec()'s dict already equals its own round-trip
    -- the invariant write_spec asserts and is_stale compares against. The matching
    `values` list keeps the raw value (a list element, which round-trips fine)."""
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def _field_record(path: str, value, hint, panel, rna_props, notes,
                  top_level: bool = True) -> dict | None:
    """One field's spec entry, or None for a container handled by recursion.
    `value` is MISSING for a required field of a collection element (no
    default); `top_level` is False inside a collection's elements, whose
    panel-less fields are not species-level traits."""
    hint = _unwrap_optional(hint) or hint
    required = value is dataclasses.MISSING
    if required:
        value = None
    if isinstance(value, tuple):
        # A tuple default is a list in the JSON: give it as one, or the written
        # spec no longer reads back equal (`write_spec`'s round-trip check).
        value = list(value)

    rec: dict = {"path": path}
    prop_name, to_profile = panel.get(path, (None, None))
    rna = rna_props.get(prop_name) if prop_name else None

    # The human label the panel shows (e.g. flowering_droop -> "Peduncle Droop",
    # which a prettified path would get wrong). From the property; for a
    # species-only field with no panel prop, a title-cased path segment.
    rec["label"] = (rna.name if rna is not None and rna.name
                    else path.split(".")[-1].replace("_", " ").title())

    if isinstance(value, Curve) or hint is Curve:
        # Every curve accepts every shape in the top-level `curve_shapes`, so the
        # list is documented once there, not repeated on all ~30 curve fields.
        rec["type"] = "curve"
        _key, x_label = _CURVE_BY_ATTR.get(path.split(".")[-1], (None, None))
        if x_label:
            rec["x_axis"] = x_label
        if isinstance(value, Curve):
            shape = _default_shape(value)
            rec["default"] = ({"shape": shape} if shape
                              else {"points": [[round(x, 4), round(y, 4)]
                                               for x, y in value.points]})
    elif isinstance(value, (set, frozenset)):
        # A multi-select: any subset of the values may be set at once. The
        # per-item model value is the identifier lowercased (its converter maps
        # the whole set, not one item, so it cannot be applied per identifier).
        rec["type"] = "multi_enum"
        rec["values"] = ([e.identifier.lower() for e in rna.enum_items] if rna
                         else sorted(value))
        rec["default"] = sorted(value)
        if rna is not None:
            rec["value_notes"] = {e.identifier.lower(): {"label": e.name,
                                                         "desc": e.description}
                                  for e in rna.enum_items if e.description}
    elif rna is not None and rna.type == "ENUM":
        rec["type"] = "enum"
        rec["values"] = _enum_values(path, rna, to_profile)
        rec["default"] = value
        if path != "inflorescence.kind":  # kinds carry their own reference cards
            conv = to_profile or (lambda v: v)
            rec["value_notes"] = {_note_key(conv(e.identifier)): {"label": e.name,
                                                                  "desc": e.description}
                                  for e in rna.enum_items if e.description}
    elif isinstance(value, bool):
        rec["type"] = "bool"
        rec["default"] = value
    elif isinstance(value, int):
        rec["type"] = "int"
        rec["default"] = value
        if rna is not None:
            _numeric_meta(rec, rna, to_profile)
    elif isinstance(value, float):
        rec["type"] = "float"
        rec["default"] = round(value, 6)
        if rna is not None:
            _numeric_meta(rec, rna, to_profile)
    else:
        rec["type"] = {bool: "bool", int: "int", float: "float"}.get(hint, "str")
        rec["default"] = value
    if required:
        del rec["default"]
        rec["required"] = True

    # Description + enrichment. A note is either a plain string (replaces the
    # tooltip) or an object carrying `description` (replaces the tooltip),
    # `endpoints` (a value->visual map for look-shaping fields) and `see_also`
    # (coupled fields). Whatever the note leaves unset falls back to the panel
    # tooltip. Species-level fields (no panel prop) rely on the note or go bare.
    note = notes.get(path)
    if isinstance(note, str):
        rec["description"] = note
    elif isinstance(note, dict):
        if note.get("description"):
            rec["description"] = note["description"]
        if note.get("endpoints"):
            rec["endpoints"] = note["endpoints"]
        if note.get("see_also"):
            rec["see_also"] = note["see_also"]
    if "description" not in rec and rna is not None and rna.description:
        rec["description"] = rna.description
    if top_level and rna is None and not isinstance(value, Curve):
        # Not on any panel -- an advanced, species-level trait. Flag it so an
        # agent knows it is rarely set by hand and can copy it from a preset.
        rec["scope"] = "species"
    return rec


def _walk(cls, obj, prefix: str, panel, rna_props, notes, out: list,
          seen_types: set, top_level: bool = True) -> None:
    """Recurse the profile's dataclasses, emitting one record per leaf field and
    descending into nested dataclasses. A nested type is documented once: a
    second field of the same type (lateral_inflorescence) gets a pointer.

    `obj` is an instance of `cls`, or None to read the class's field defaults
    (an element type with required fields has no default instance). A field
    holding a list / dict of dataclasses (the branching levels, masks, maps)
    is a container: one record carrying its default -- encoded exactly as a
    preset stores it (`serialize._encode`) -- then its element type's fields
    under `<path>[].` (list) or `<path>{}.` (dict)."""
    hints = _hints(cls)
    for f in dataclasses.fields(cls):
        path = f"{prefix}{f.name}"
        value = getattr(obj, f.name) if obj is not None else _field_default(f)
        hint = hints.get(f.name, type(value))
        inner = _unwrap_optional(hint)
        target_type = inner or hint

        collection = _collection_of(target_type)
        if collection is not None:
            kind, elem = collection
            rec = {"path": path, "type": "object", "container": kind,
                   "optional": inner is not None}
            if value is not dataclasses.MISSING:
                rec["default"] = serialize._encode(value)
            if elem in seen_types:
                out.append({**rec, "same_as": elem.__name__.lower()})
                continue
            seen_types.add(elem)
            out.append({**rec, "object": elem.__name__})
            child_prefix = f"{path}{'[]' if kind == 'list' else '{}'}."
            child_panel, child_rna = panel, rna_props
            if top_level and path == LEVELS_PATH:
                child_panel, child_rna = _level_panel()
            _walk(elem, _default_instance(elem), child_prefix, child_panel,
                  child_rna, notes, out, seen_types, top_level=False)
            continue

        # A nested dataclass (possibly optional and None by default): recurse
        # into a default instance so its own defaults are documented.
        if dataclasses.is_dataclass(target_type) and not isinstance(value, Curve):
            if target_type in seen_types:
                out.append({"path": path, "type": "object",
                            "same_as": target_type.__name__.lower(),
                            "optional": inner is not None})
                continue
            seen_types.add(target_type)
            out.append({"path": path, "type": "object",
                        "object": target_type.__name__,
                        "optional": inner is not None})
            child = (value if dataclasses.is_dataclass(value)
                     else _default_instance(target_type))
            _walk(target_type, child, f"{path}.", panel, rna_props, notes, out,
                  seen_types, top_level)
            continue

        rec = _field_record(path, value, hint, panel, rna_props, notes, top_level)
        if rec is not None:
            out.append(rec)


# --- assembly -------------------------------------------------------------

def _kind_capabilities() -> dict:
    """Per inflorescence kind: a description (harvested from the panel enum
    item, so it cannot drift), its topology, and which control groups it makes
    inert -- the same facts the panel hides controls by."""
    rna = _rna_props().get("inflorescence_kind")
    descs = {it.identifier.lower(): it.description
             for it in rna.enum_items} if rna is not None else {}
    out = {}
    for kind in morphology.KIND_TOPOLOGY:
        out[kind] = {
            "description": descs.get(kind, ""),
            "topology": morphology.kind_topology(kind),
            "has_pedicels": morphology.kind_has_pedicels(kind),
            "flowering_gradients_apply": morphology.kind_gradients_apply(kind),
            "terminal_cluster": kind in morphology.TERMINAL_CLUSTER_KINDS,
            "flat_topped": kind in morphology.FLAT_TOPPED_KINDS,
            "relayed": kind in morphology.RELAY_KINDS,
        }
    return out


def generate_spec() -> dict:
    notes = {}
    if NOTES_PATH.exists():
        notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    panel = _panel_by_path()
    rna_props = _rna_props()
    fields: list = []
    _walk(MorphologyProfile, MorphologyProfile(), "", panel, rna_props, notes,
          fields, set())
    return {
        "schema_version": SCHEMA_VERSION,
        "about": ("Parameter spec for the PlantGen Blender add-on. Author a full "
                  "species preset (the same JSON the add-on loads) by setting "
                  "these fields. Generated from the model; do not edit by hand."),
        "scope": ("The AI sets these PARAMETERS. The user provides the asset "
                  "MESHES -- leaf blade shape, flower/petal form, bud/bract/catkin "
                  "geometry -- and does final tweaking. Sizes are multipliers on "
                  "those meshes (1.0 = the mesh as modelled), so set relative "
                  "proportions, not absolute size."),
        "curves": ("A field of type 'curve' is a small graph over a normalised x "
                   "(its x_axis says what x means, e.g. base->tip or "
                   "seedling->mature). Set it to one of the named shapes below as "
                   "{\"shape\": \"<key>\"}; every curve field accepts every shape. "
                   "Raw {\"points\": [[x, y], ...]} is a rare escape hatch for a "
                   "shape none of the names fit."),
        "curve_shapes": [{"key": k, "label": _SHAPE_LABELS[k],
                          "points": _SHAPE_POINTS[k], "smooth": _SHAPE_SMOOTH[k]}
                         for k in _SHAPE_KEYS],
        "inflorescence_kinds": _kind_capabilities(),
        "fields": fields,
    }


def _serialize(doc: dict) -> str:
    """Lean JSON: the header stays readable, but every table entry and every
    field record goes on ONE line -- greppable and diff-friendly, and ~20%
    smaller than full indentation on a content-heavy file, which matters when an
    agent reads the whole thing. Round-trips to the same object."""
    def line(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ": "))

    scalars = {k: v for k, v in doc.items()
               if k not in ("curve_shapes", "inflorescence_kinds", "fields")}
    head = json.dumps(scalars, ensure_ascii=False, indent=2)[:-2]  # drop "\n}"

    def block(key, entries):
        return f'  {json.dumps(key)}: [\n    ' + ",\n    ".join(entries) + "\n  ]"

    parts = [
        head,
        block("curve_shapes", [line(s) for s in doc["curve_shapes"]]),
        '  "inflorescence_kinds": {\n    ' + ",\n    ".join(
            f"{json.dumps(k)}: {line(v)}"
            for k, v in doc["inflorescence_kinds"].items()) + "\n  }",
        block("fields", [line(f) for f in doc["fields"]]),
    ]
    return ",\n".join(parts) + "\n}\n"


def write_spec(path: Path = SPEC_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _serialize(generate_spec())
    assert json.loads(text) == generate_spec(), "lean serialization must round-trip"
    path.write_text(text, encoding="utf-8")
    return path


def is_stale(path: Path = SPEC_PATH) -> tuple[bool, str]:
    """Whether the committed spec differs from one freshly generated from the
    model. Returns `(stale, summary)`; the summary names the drift by field path
    so a regenerate says what moved. The anti-drift guard -- Blender-only, since
    `generate_spec` reads the property RNA. Compares parsed objects, so
    formatting never counts as drift."""
    fresh = generate_spec()
    if not path.exists():
        return True, "spec file does not exist yet"
    try:
        on_disk = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return True, f"spec file unreadable: {e}"
    if on_disk == fresh:
        return False, "up to date"

    def by_path(doc):
        return {f["path"]: f for f in doc.get("fields", [])}

    old, new = by_path(on_disk), by_path(fresh)

    def sample(paths):
        paths = sorted(paths)
        head = ", ".join(paths[:5])
        return f"{head}{', ...' if len(paths) > 5 else ''}" if paths else ""

    bits = []
    header = sorted(k for k in set(on_disk) | set(fresh)
                    if k != "fields" and on_disk.get(k) != fresh.get(k))
    if header:
        bits.append(f"header: {', '.join(header)}")
    for label, paths in (("+", set(new) - set(old)),
                         ("-", set(old) - set(new)),
                         ("~", {p for p in set(old) & set(new) if old[p] != new[p]})):
        if paths:
            bits.append(f"{label}{len(paths)} fields ({sample(paths)})")
    return True, "; ".join(bits) or "content differs"
