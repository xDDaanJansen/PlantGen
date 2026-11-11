"""Species presets: a full `MorphologyProfile` <-> JSON, plus the file envelope.

A species preset is a **baked, self-contained** profile: every field, including
the ones the panel cannot show (`plastochron`, `veg`, organ masses,
`topology_override`, `lateral_inflorescence`), serialised outright rather than
referenced by a base-species name. The 9 built-in species become the first
bundled presets, and "Save Preset" is this same serialisation of whatever a
plant resolves to.

Pure module: no bpy. The profile lives in `growth/`, so its serialisation does
too; the bpy bridge (`blender/presets.py`) owns save/load/apply.

The heart of it is a **generic recursive walk** rather than a hand-written
per-field mapping. The profile is entirely plain data -- nested dataclasses,
string-valued enum constants, scalars, one `frozenset`, two `X | None`
optionals, and `Curve` -- with no callables, bpy pointers or strategy objects,
so the encoder reads `dataclasses.fields()` and the decoder is driven by the
resolved type hints. This is deliberate: a hand-kept mapping of ~200 fields
drifts (see `bridge._FIELDS`'s own note about that), and this project adds
profile fields constantly. The walk keeps serialising them with no edit here.
`Curve` is the one type the walk special-cases.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from pathlib import Path

from .curves import Curve
from .maps import LEGACY_TO_LEVEL
from .morphology import MorphologyProfile

SCHEMA_VERSION = 11
KIND = "species"


class SpeciesFileError(ValueError):
    """A file loaded as a species preset is not one -- wrong or missing `kind`.

    Disambiguated by the `kind` field rather than the filename, so loading an
    unrelated `.json` (or a future sub-preset envelope) gives a clear error
    instead of a confusing partial parse."""


class SpeciesPreset(typing.NamedTuple):
    """A loaded preset: the reconstructed profile plus its envelope metadata."""

    profile: MorphologyProfile
    name: str = ""
    description: str = ""
    preset_id: str = ""


# --- Curve <-> plain data --------------------------------------------------

def _encode_curve(curve: Curve):
    # Authoring points when it has them -- the handful of (x, y) a person drew,
    # which stays small, reads in a diff, and re-seeds the widget as points
    # rather than 64 uneditable ones. A curve that came *from* a widget keeps
    # its exact table AND the widget's own points + handle types: its smooth
    # (Auto) shape is not the piecewise line through its points, so the table
    # is what grows and the points / handles are what re-seed the widget
    # exactly. (Without them a saved widget curve came back flat on apply.)
    if curve.handles:
        return {"samples": list(curve.samples),
                "points": [[x, y] for x, y in curve.points],
                "handles": list(curve.handles)}
    if curve.points:
        return {"points": [[x, y] for x, y in curve.points]}
    return {"samples": list(curve.samples)}


def _decode_curve(data) -> Curve:
    if "samples" in data:
        return Curve(data["samples"], [(x, y) for x, y in data.get("points", ())],
                     data.get("handles", ()))
    if "points" in data:
        return Curve.from_points([(x, y) for x, y in data["points"]])
    return Curve(data["samples"])


# --- generic profile <-> plain data ---------------------------------------

def _encode(value):
    if isinstance(value, Curve):
        return _encode_curve(value)
    if dataclasses.is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, (set, frozenset)):
        return sorted(value)  # only ever strings (Attachment values); stable order
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}  # keys are plain strings
    return value  # str / int / float / bool / None round-trip through JSON as-is


def _unwrap_optional(hint):
    """The non-None member of an `X | None` hint, else None.

    Handles both PEP 604 (`X | None`, a `types.UnionType`) and `typing.Optional`,
    since `get_type_hints` can return either depending on how the annotation was
    written. The profile has two such fields: `veg` and `lateral_inflorescence`.
    """
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return None


_HINTS: dict[type, dict] = {}


def _hints(cls) -> dict:
    """Resolved type hints for a dataclass, cached. Needed because the growth
    modules use `from __future__ import annotations`, so `field.type` is a bare
    string; `get_type_hints` evaluates it against the class's own module."""
    cached = _HINTS.get(cls)
    if cached is None:
        cached = _HINTS[cls] = typing.get_type_hints(cls)
    return cached


def _decode(data, hint):
    if data is None:
        return None
    inner = _unwrap_optional(hint)
    if inner is not None:
        hint = inner
    if hint is Curve:
        return _decode_curve(data)
    if dataclasses.is_dataclass(hint):
        return _decode_dataclass(data, hint)
    origin = typing.get_origin(hint)
    if hint in (set, frozenset) or origin in (set, frozenset):
        return (frozenset if hint is frozenset or origin is frozenset else set)(data)
    if origin in (list, tuple):
        args = typing.get_args(hint)
        elem = args[0] if args else object
        decoded = [_decode(v, elem) for v in data]
        return tuple(decoded) if origin is tuple else decoded
    if origin is dict:
        args = typing.get_args(hint)
        val = args[1] if len(args) == 2 else object
        return {k: _decode(v, val) for k, v in data.items()}
    # Light coercion so a hand-edited preset that writes `45` for a float field
    # (or `3.0` for an int one) still reconstructs the right type. bool is a
    # subclass of int and already round-trips exactly, so leave it alone.
    if hint is float:
        return float(data)
    if hint is int and not isinstance(data, bool):
        return int(data)
    return data  # str, and the string-valued enum-constant classes, pass through


def _decode_dataclass(data: dict, cls):
    hints = _hints(cls)
    kwargs = {f.name: _decode(data[f.name], hints[f.name])
              for f in dataclasses.fields(cls) if f.name in data}
    return cls(**kwargs)


# --- public: profile body --------------------------------------------------

def profile_to_dict(profile: MorphologyProfile) -> dict:
    return _encode(profile)


def profile_from_dict(data: dict) -> MorphologyProfile:
    return _decode_dataclass(data, MorphologyProfile)


# --- public: the file envelope ---------------------------------------------

def _v1_drop_leaf_deform(d: dict) -> dict:
    """v1 -> v2: leaf blade deform (curl/cup/gravity/wave/twist and their
    variations) was removed from the sim add-on -- leaves arrive pre-shaped from
    the leaf tool. The keys are dropped;
    `_decode` already ignores unknown ones, so this is explicit bookkeeping."""
    profile = d.get("profile")
    if isinstance(profile, dict):
        for k in ("leaf_curl", "leaf_curl_variation", "leaf_cup", "leaf_cup_variation",
                  "leaf_gravity", "leaf_gravity_variation", "leaf_wave_amplitude",
                  "leaf_wave_frequency", "leaf_twist", "leaf_twist_variation"):
            profile.pop(k, None)
    return d


def _v2_drop_leaf_placement(d: dict) -> dict:
    """v2 -> v3: leaf placement (on_trunk/on_branches/start/end/slide) stopped
    being a growth-time gate -- growth emits a foliage leaf on every vegetative
    node and a render-time component mask decides which show.
   The nested `leaf_placement` object is
    dropped; `_decode` already ignores unknown keys."""
    profile = d.get("profile")
    if isinstance(profile, dict):
        profile.pop("leaf_placement", None)
    return d


def _v3_drop_leaf_form(d: dict) -> dict:
    """v3 -> v4: heteroblasty (the juvenile/adult leaf split) was removed -- one
    leaf collection now, and two leaf shapes are two leaf components with a mask.
   The `leaf_form_profile` curve is
    dropped; `_decode` already ignores unknown keys."""
    profile = d.get("profile")
    if isinstance(profile, dict):
        profile.pop("leaf_form_profile", None)
    return d


# The old growth patterns as the leaf arrangement's (spiral, whorl): the same
# slot set at every node (only the slot order differs).
_V4_LEAF_PATTERNS = {"spiral": (137.5, 1), "distichous": (180.0, 1),
                     "unilateral": (0.0, 1), "opposite": (0.0, 2), "decussate": (90.0, 2)}


def _v4_leaf_arrangement_absolute_stems(d: dict) -> dict:
    """v4 -> v5: the zoned leaf
    phyllotaxis became one `leaf_arrangement` (the trunk's, as spiral / whorl /
    whorl angle; the arrangement jitter is gone), and the stems level's Start
    Length / Start Width became the stems' own first internode and base radius
    in metres -- the old `internode_length` x Start Length and `base_radius` x
    Start Width, so the plant is the same. `base_radius`, `tip_radius` and
    `pipe_per_leaf` are dropped (the pipe model sized nothing any more)."""
    profile = d.get("profile")
    if not isinstance(profile, dict):
        return d
    trunk = profile.pop("trunk_phyllotaxis", None)
    profile.pop("lateral_phyllotaxis", None)
    profile.pop("phyllotaxis_jitter_deg", None)
    if isinstance(trunk, dict):
        pattern = str(trunk.get("pattern", "alternate"))
        if pattern == "whorled":
            n = max(2, int(trunk.get("whorl_count", 3)))
            spiral, whorl = 180.0 / n, n
        elif pattern in _V4_LEAF_PATTERNS:
            spiral, whorl = _V4_LEAF_PATTERNS[pattern]
        else:  # alternate: its own divergence
            spiral, whorl = float(trunk.get("divergence_deg", 137.5)), 1
        profile["leaf_arrangement"] = {"spiral_angle_deg": spiral, "whorl": whorl,
                                       "whorl_angle_deg": 0.0}
    base_radius = profile.pop("base_radius", None)
    levels = profile.get("levels")
    if isinstance(levels, list) and levels and isinstance(levels[0], dict):
        stems = levels[0]
        stems["start_length_scale"] = (float(stems.get("start_length_scale", 1.0))
                                       * float(profile.get("internode_length", 0.045)))
        stems["start_width_scale"] = (float(stems.get("start_width_scale", 1.0))
                                      * float(0.005 if base_radius is None else base_radius))
    for key in ("tip_radius", "pipe_per_leaf"):
        profile.pop(key, None)
    return d


_V5_SECTION_KEYS = ("cross_section", "profile_sides", "lobe_count", "lobe_depth", "flatten")


def _v5_variation_and_section_per_level(d: dict) -> dict:
    """v5 -> v6: the plant-level internode and radius variations and the cross
    section moved onto the branching levels. `internode_variation` becomes every
    level's Length / Step variation and `radius_variation` the stems level's
    Width / Step variation (it only ever varied the stems); the cross-section
    fields are copied onto every level, so each starts as the plant was."""
    profile = d.get("profile")
    if not isinstance(profile, dict):
        return d
    internode = profile.pop("internode_variation", 0.18)
    radius = profile.pop("radius_variation", 0.0)
    section = {k: profile.pop(k) for k in _V5_SECTION_KEYS if k in profile}
    levels = profile.get("levels")
    if isinstance(levels, list):
        for i, level in enumerate(levels):
            if not isinstance(level, dict):
                continue
            level.setdefault("length_variation", internode)
            level.setdefault("width_variation", radius if i == 0 else 0.0)
            for k, v in section.items():
                level.setdefault(k, v)
    return d


def _v6_maps_onto_levels(d: dict) -> dict:
    """v6 -> v7: the plant-level growth maps that were really level parameters
    moved onto the branching levels --
    `maps.LEGACY_TO_LEVEL` says where each lands. A copy per level, so every
    level scales exactly as the one plant map did."""
    profile = d.get("profile")
    if not isinstance(profile, dict):
        return d
    old = profile.get("maps")
    levels = profile.get("levels")
    if not isinstance(old, list) or not isinstance(levels, list):
        return d
    kept = []
    for m in old:
        rule = LEGACY_TO_LEVEL.get(m.get("target")) if isinstance(m, dict) else None
        if rule is None:
            kept.append(m)
            continue
        target, first, stays = rule
        for level in levels[first:]:
            if isinstance(level, dict):
                level.setdefault("maps", []).append(dict(m, target=target))
        if stays:
            kept.append(m)
    profile["maps"] = kept
    return d


def _v7_fork_to_growth_pattern(d: dict) -> dict:
    """v7 -> v8: a level's Fork toggle became its Growth Pattern
    -- Fork on is Dichotomous, off Monopodial."""
    profile = d.get("profile")
    levels = profile.get("levels") if isinstance(profile, dict) else None
    if isinstance(levels, list):
        for level in levels:
            if isinstance(level, dict) and "fork" in level:
                level.setdefault("growth_pattern",
                                 "dichotomous" if level.pop("fork") else "monopodial")
    return d


def _v8_axis_type_to_attachments(d: dict) -> dict:
    """v8 -> v9: Axis Type and a level's Unit End were removed.
   An Indeterminate plant's tips never
    flowered and its laterals did: its inflorescences become Axillary instead
    of Terminal. Unit End is dropped (a unit's tip flowers exactly when the
    inflorescence is Terminal)."""
    profile = d.get("profile")
    if not isinstance(profile, dict):
        return d
    if profile.pop("axis_type", "determinate") == "indeterminate":
        for key in ("inflorescence", "lateral_inflorescence"):
            inf = profile.get(key)
            if isinstance(inf, dict):
                attached = set(inf.get("attachments", ["terminal"]))
                attached.discard("terminal")
                attached.add("axillary")
                inf["attachments"] = sorted(attached)
    for level in profile.get("levels") or []:
        if isinstance(level, dict):
            level.pop("unit_end", None)
    return d


def _v9_steps_map_to_growth(d: dict) -> dict:
    """v9 -> v10: a level's Steps map became its Growth % map -- Steps is the
    full-grown node count, Growth % the part of it each branch builds. The map
    keeps its curve (multipliers above 1 now cap at 100%)."""
    profile = d.get("profile")
    for level in (profile.get("levels") or []) if isinstance(profile, dict) else []:
        for m in (level.get("maps") or []) if isinstance(level, dict) else []:
            if isinstance(m, dict) and m.get("target") == "steps":
                m["target"] = "growth"
    return d


def _v10_heliotropism_onto_inflorescence(d: dict) -> dict:
    """v10 -> v11: Flower Heliotropism moved from the plant onto the flower
    component, so the profile's `flower_heliotropism` became the primary
    inflorescence's `heliotropism` -- and every other inflorescence's, as
    they all shared the plant's value before, so the plant is the same)."""
    profile = d.get("profile")
    if isinstance(profile, dict) and "flower_heliotropism" in profile:
        value = profile.pop("flower_heliotropism")
        inf = profile.get("inflorescence")
        if isinstance(inf, dict):
            inf["heliotropism"] = value
            for extra in profile.get("extra_inflorescences") or ():
                if isinstance(extra, dict):
                    extra["heliotropism"] = value
            lateral = profile.get("lateral_inflorescence")
            if isinstance(lateral, dict):
                lateral["heliotropism"] = value
    return d


_MIGRATIONS: dict[int, typing.Callable[[dict], dict]] = {
    1: _v1_drop_leaf_deform,
    2: _v2_drop_leaf_placement,
    3: _v3_drop_leaf_form,
    4: _v4_leaf_arrangement_absolute_stems,
    5: _v5_variation_and_section_per_level,
    6: _v6_maps_onto_levels,
    7: _v7_fork_to_growth_pattern,
    8: _v8_axis_type_to_attachments,
    9: _v9_steps_map_to_growth,
    10: _v10_heliotropism_onto_inflorescence,
}
"""Migrations from schema_version N -> N+1, keyed by N. Additive field changes
need no entry (`_decode` fills missing fields from their dataclass defaults and
ignores unknown ones); only renames/removals/semantic shifts do."""


def _migrate(d: dict) -> dict:
    version = d.get("schema_version", SCHEMA_VERSION)
    while version < SCHEMA_VERSION:
        d = _MIGRATIONS[version](d)
        version = d["schema_version"] = version + 1
    return d


def to_json(name: str, profile: MorphologyProfile, description: str = "",
            *, preset_id: str = "") -> str:
    d = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "name": name,
        "description": description,
        "profile": profile_to_dict(profile),
    }
    if preset_id:
        d["id"] = preset_id
    return json.dumps(d, indent=2)


def from_json(text: str) -> SpeciesPreset:
    d = json.loads(text)
    if d.get("kind") != KIND:
        raise SpeciesFileError(
            f'not a species preset (expected "kind": "{KIND}", got {d.get("kind")!r})')
    d = _migrate(d)
    return SpeciesPreset(
        profile=profile_from_dict(d["profile"]),
        name=d.get("name", ""),
        description=d.get("description", ""),
        preset_id=d.get("id", ""),
    )


def save_json(path, name: str, profile: MorphologyProfile, description: str = "",
              *, preset_id: str = "") -> None:
    Path(path).write_text(
        to_json(name, profile, description, preset_id=preset_id), encoding="utf-8")


def load_json(path) -> SpeciesPreset:
    return from_json(Path(path).read_text(encoding="utf-8"))
