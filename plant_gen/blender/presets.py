"""bpy bridge for species presets: the library folders, and apply / save / load.

Pure serialisation is `growth/serialize.py`; this is the Blender side of it --
where the bundled and user preset folders live, applying a preset onto an object
(embed + reflect its controls, via `bridge`), and saving the active plant out to
a file. Keeping it thin and bpy-only keeps the serialiser headless-testable.
"""

from __future__ import annotations

import json
from pathlib import Path

import bpy

from ..growth import serialize
from . import bridge

# The addon registers under its top package name; the AddonPreferences lookup
# needs that exact string, not this module's `__package__`.
_ADDON = __package__.split(".")[0] if __package__ else "plant_gen"

BUNDLED_DIR = Path(__file__).parent.parent / "presets"
"""The presets that ship with the addon -- generated from the built-in species
by `python -m plant_gen.tests.preset_roundtrip --generate`."""


def user_presets_dir(create: bool = False) -> Path:
    """Where the user's own presets live: the AddonPreference path if set, else a
    default folder under Blender's user config so saved presets persist across
    .blends and show up in the Library."""
    addon = bpy.context.preferences.addons.get(_ADDON)
    configured = getattr(addon.preferences, "user_presets_path", "") if addon else ""
    if configured:
        path = Path(bpy.path.abspath(configured))
    else:
        path = Path(bpy.utils.user_resource('CONFIG')) / "plantgen_presets"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def slug(name: str) -> str:
    """A filesystem-safe stem for a preset file, from a display name."""
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    return keep.strip("_").lower() or "preset"


def _read_entry(path: Path, source: str) -> dict | None:
    """Envelope-only metadata for one preset file (name / description / id), or
    None if it is not a species preset. The full profile is parsed only when the
    preset is actually applied, so listing the Library stays cheap."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("kind") != serialize.KIND:
        return None
    return {
        "id": d.get("id") or path.stem,
        "name": d.get("name") or path.stem,
        "description": d.get("description", ""),
        "path": str(path),
        "source": source,
    }


def list_presets() -> list[dict]:
    """Every available preset -- bundled first, then the user folder -- each
    {id, name, description, path, source}. Files that do not parse as a species
    preset are skipped rather than shown as broken entries."""
    out: list[dict] = []
    for folder, source in ((BUNDLED_DIR, "bundled"), (user_presets_dir(), "user")):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            entry = _read_entry(path, source)
            if entry is not None:
                out.append(entry)
    return out


def find_preset(preset_id: str) -> dict | None:
    """The preset whose id matches, user folder shadowing bundled. None if the
    id is blank or no file matches (a preset that has since been deleted)."""
    if not preset_id:
        return None
    matches = [e for e in list_presets() if e["id"] == preset_id]
    matches.sort(key=lambda e: e["source"] != "user")  # user before bundled
    return matches[0] if matches else None


def apply_preset(obj, preset: serialize.SpeciesPreset) -> None:
    """Make `preset` the plant's base: stash it (self-contained), reflect its
    panel controls and seed its curves. The caller regrows."""
    preset_id = preset.preset_id or slug(preset.name)
    bridge.stash_profile(obj, preset.name, preset.profile, preset.description,
                         preset_id=preset_id)
    bridge.apply_profile(obj, preset.profile)


LAST_MISSING: list = []
"""What the last full preset applied could not find (assets neither in the
file nor in the preset's .blend) -- for the operator to report."""


def apply_preset_file(obj, path) -> serialize.SpeciesPreset:
    """Apply a preset file. A full preset (one with a `plant` section) makes
    `obj` a 1-to-1 copy of the plant it was saved from -- assets appended,
    every setting and curve restored; a growth-only one (the bundled species,
    older presets) sets the growth and leaves the components as they are."""
    from . import plant_state
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    preset = serialize.from_json(text)
    apply_preset(obj, preset)
    LAST_MISSING.clear()
    raw = json.loads(text)
    if isinstance(raw.get("plant"), dict):
        LAST_MISSING.extend(plant_state.restore(obj, raw["plant"], path.parent,
                                                raw.get("assets")))
    return preset


def reload(obj) -> serialize.SpeciesPreset | None:
    """Re-apply the current preset's library version, pulling any updates. None
    if the plant has no remembered preset or its file can no longer be found."""
    entry = find_preset(obj.plantgen.preset_id)
    if entry is None:
        return None
    return apply_preset_file(obj, entry["path"])


def save_preset(obj, path, name: str, description: str = "", *,
                full: bool = True) -> Path:
    """Save the plant as a preset file: its resolved profile (base + panel +
    curves) and -- `full` -- the whole plant (every setting and curve,
    `plant_state`) with its assets in a .blend beside it (same stem), so that
    loading it makes a 1-to-1 copy. The saved id is the file's own stem, so it
    is stable and unique within the folder and 'Reload from preset' finds it."""
    from . import plant_state
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    profile = bridge.profile_from_object(obj)
    data = json.loads(serialize.to_json(name or obj.name, profile, description,
                                        preset_id=path.stem))
    if full:
        data["plant"] = plant_state.dump(obj)
        assets = plant_state.write_assets(path.with_suffix(".blend"), data["plant"])
        if assets is not None:
            data["assets"] = assets
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path
