"""PlantGen's own UI icons -- plant, stems, leaf, flower, phyllotaxis ... -- for
the concepts Blender's stock icons don't show.

The art is plant_gen/icons/plant/<name>.svg (16 px grid, one colour, dimmed
parts as opacity), exported to <name>.png -- white, the coverage in alpha -- by
tools/export_icons.py. A custom preview draws at its own pixel colours (Blender
never tints it the way it tints its own icons), so each mask is written out in
the theme's text colour, keyed by that colour, and loaded into one preview
collection: a theme change just makes the next set. The same route as
`map_icons`: no image datablock, safe to call from a panel's draw.

Wherever an icon name is passed around (`ui._COMPONENT_ICONS`, `_level_box`,
`_owner_icon` ...), "PG:<name>" stands for one of these; `kw(icon)` turns either
kind into the keyword argument a layout call takes, falling back to a stock
icon if a PlantGen one can't be made."""

from __future__ import annotations

import os
import struct
import tempfile
import zlib

PREFIX = "PG:"
_SIZE = 32
_ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "icons", "plant")
_FOLDER = os.path.join(tempfile.gettempdir(), "plantgen_plant_icons")
_PREVIEWS = None
_IDS: dict = {}
_MASKS: dict = {}

# the stock icon each one replaced -- drawn if the PlantGen icon can't be made
FALLBACK = {
    "plant": "OUTLINER_COLLECTION", "stems": "OUTLINER_OB_CURVES",
    "branches": "OUTLINER_DATA_CURVES", "inflorescence": "OUTLINER_OB_LIGHTPROBE",
    "leaf": "OUTLINER_OB_MESH", "bract": "OUTLINER_OB_MESH",
    "flower": "OUTLINER_OB_POINTCLOUD", "bud": "OUTLINER_OB_EMPTY",
    "fruit": "OUTLINER_OB_META", "phyllotaxis": "PHYSICS", "growth": "FORCE_FORCE",
    "bloom": "TIME", "senescence": "GHOST_ENABLED", "pruning": "UNLINKED",
    "petiole": "CURVE_PATH", "compound_leaf": "OUTLINER_OB_POINTCLOUD",
    "calendar": "PREVIEW_RANGE",
}


def _read_mask(path: str) -> list:
    """The alpha of an 8-bit RGBA / grey+alpha PNG, rows top first, 0..1."""
    with open(path, "rb") as f:
        data = f.read()
    pos, idat = 8, b""
    w = h = depth = ctype = 0
    while pos < len(data):
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    if depth != 8 or ctype not in (4, 6):
        raise ValueError(f"{path}: need 8-bit RGBA or grey+alpha")
    bpp = 4 if ctype == 6 else 2
    raw = zlib.decompress(idat)
    stride = w * bpp
    rows, prev = [], bytearray(stride)
    for y in range(h):
        kind = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 1:
                line[i] = (line[i] + a) & 255
            elif kind == 2:
                line[i] = (line[i] + b) & 255
            elif kind == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append([line[x * bpp + bpp - 1] / 255.0 for x in range(w)])
        prev = line
    return rows


def _theme_rgb() -> tuple:
    """The colour Blender draws its own icons in: the theme's text colour
    (an sRGB colour, as the PNG wants it), 0-255."""
    try:
        import bpy
        c = bpy.context.preferences.themes[0].user_interface.wcol_regular.text
        return tuple(int(round(max(0.0, min(1.0, v)) * 255)) for v in c[:3])
    except Exception:  # noqa: BLE001 -- no context yet: Blender's default text grey
        return (230, 230, 230)


def _collection():
    global _PREVIEWS
    if _PREVIEWS is None:
        import bpy.utils.previews
        _PREVIEWS = bpy.utils.previews.new()
    return _PREVIEWS


def icon_id(name: str) -> int:
    """The preview icon id of PlantGen icon `name` in the theme's colour, made
    the first time it is asked for. 0 if it can't be made."""
    rgb = _theme_rgb()
    key = (name, rgb)
    found = _IDS.get(key)
    if found is not None:
        return found
    try:
        from .map_icons import _png
        os.makedirs(_FOLDER, exist_ok=True)
        tag = "pgicon_{}_{}_{}_{}".format(name, *rgb)
        path = os.path.join(_FOLDER, tag + ".png")
        if not os.path.exists(path):
            mask = _MASKS.get(name)
            if mask is None:
                mask = _MASKS[name] = _read_mask(os.path.join(_ART, name + ".png"))
            col = [v / 255.0 for v in rgb]
            _png(path, [[(col[0], col[1], col[2], a) for a in row] for row in mask])
        pcoll = _collection()
        item = pcoll.get(tag) or pcoll.load(tag, path, 'IMAGE')
        _IDS[key] = item.icon_id
    except Exception:  # noqa: BLE001 -- an icon is cosmetic; never break a draw
        _IDS[key] = 0
    return _IDS[key]


def kw(icon: str) -> dict:
    """The layout keyword for `icon` -- a stock name, or "PG:<name>"."""
    if icon and icon.startswith(PREFIX):
        name = icon[len(PREFIX):]
        found = icon_id(name)
        return {"icon_value": found} if found else {"icon": FALLBACK.get(name, 'NONE')}
    return {"icon": icon or 'NONE'}


def unregister() -> None:
    global _PREVIEWS
    if _PREVIEWS is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_PREVIEWS)
        _PREVIEWS = None
    _IDS.clear()
    _MASKS.clear()
