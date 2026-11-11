"""FlowerGen presets: a flower's `FlowerMeshParams` <-> schema-versioned JSON,
with migrations for older files. Moved from `the parametric add-on/model/io.py`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .flower import FlowerMeshParams

FLOWER_SCHEMA_VERSION = 3


_FLOWER_KIND = "flower_mesh"


class FlowerFileError(ValueError):
    """Raised when a file loaded as a Flower preset isn't one (e.g. it's a whole-
    Species or Distribution preset, or an unrelated .json file) -- distinguished by
    the `"kind"` field, not the filename, same as `DistributionFileError`."""


def _migrate_flower_v1_to_v2(d: dict) -> dict:
    """v2 adds `petal_tube_match_disc`/`sepal_tube_match_disc`/
    `receptacle_match_disc`, which link those three radii to Disc Radius and
    default to True for new flowers. A v1 preset predates the flags and carries
    an explicit radius it was authored with on purpose, so turn them OFF rather
    than letting the preset silently snap its tube and receptacle to the disc
    the first time it's reloaded.

    Also renames nothing and rescales nothing for Top/Bottom Shape, despite
    those changing meaning in the same release (absolute meters -> multiples of
    the ring's own width, and the widget's 0..1 range no longer remapped to
    -0.01..0.01). That is deliberate: the old values were a push distance
    stored against a curve whose Y axis has also changed, and the new scalar
    amount they'd need has no v1 counterpart at all -- there's nothing to carry
    forward, so a v1 preset reloads with cap shaping at 0 (a flat cut) plus the
    new automatic base fit, and its tip cap is re-picked from the Petal Shape
    menu. Faking a conversion would produce arbitrary geometry, not the old
    silhouette."""
    flower = d.get("flower_mesh")
    if flower:
        for key in (
            "petal_tube_match_disc", "sepal_tube_match_disc", "receptacle_match_disc",
        ):
            flower.setdefault(key, False)
    return d


def _migrate_flower_v2_to_v3(d: dict) -> dict:
    """v3 turns Bud Cup from a MULTIPLIER on the open Cup value into a true
    authored pair (see `skeleton/flower.py`'s `petal_bud_cup` docstring), and
    adds a new `petal_bud_clamp_overlap`/`sepal_bud_clamp_overlap` pairing for
    Clamp Overlap that didn't exist at all in v2. Both renames have an EXACT
    equivalent to carry forward (unlike v1->v2's Top/Bottom Shape, which
    didn't): the old multiplier's effective closed-bud cup was
    `multiplier * cup`, so that product becomes the new independent
    `bud_cup` field, reproducing the old fully-closed silhouette byte-for-
    byte. Clamp Overlap had no bud-specific value at all in v2 -- it applied
    uniformly across every bloom stage -- so a v2 preset's existing
    `clamp_overlap` is copied forward as its own bud value too, reproducing
    that same "one value everywhere" behavior rather than silently turning
    the clamp off at full closure."""
    flower = d.get("flower_mesh")
    if flower:
        for prefix in ("petal", "sepal"):
            multiplier = flower.pop(f"{prefix}_bud_cup_multiplier", 3.0)
            cup = flower.get(f"{prefix}_cup", 0.0)
            flower[f"{prefix}_bud_cup"] = multiplier * cup
            flower.setdefault(
                f"{prefix}_bud_clamp_overlap", flower.get(f"{prefix}_clamp_overlap", 0.0))
    return d


_FLOWER_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    1: _migrate_flower_v1_to_v2,
    2: _migrate_flower_v2_to_v3,
}


def _migrate_flower(d: dict) -> dict:
    version = d.get("schema_version", FLOWER_SCHEMA_VERSION)
    while version < FLOWER_SCHEMA_VERSION:
        d = _FLOWER_MIGRATIONS[version](d)
        version = d["schema_version"] = version + 1
    return d


def to_flower_json(name: str, params: FlowerMeshParams) -> str:
    d = {
        "schema_version": FLOWER_SCHEMA_VERSION, "kind": _FLOWER_KIND,
        "name": name, "flower_mesh": params.to_dict(),
    }
    return json.dumps(d, indent=2)


def from_flower_json(text: str) -> FlowerMeshParams:
    d = json.loads(text)
    if d.get("kind") != _FLOWER_KIND:
        raise FlowerFileError(
            "This file isn't a Flower preset (expected "
            f'"kind": "{_FLOWER_KIND}", got {d.get("kind")!r}).'
        )
    return FlowerMeshParams.from_dict(_migrate_flower(d)["flower_mesh"])


def save_flower_json(path, name: str, params: FlowerMeshParams) -> None:
    Path(path).write_text(to_flower_json(name, params), encoding="utf-8")


def load_flower_json(path) -> FlowerMeshParams:
    return from_flower_json(Path(path).read_text(encoding="utf-8"))
