"""Icons in a Plant Structure swatch's colour:
a swatch, the map glyph filled (the parameter is mapped) and outlined (not).

Written as PNGs in pure Python -- no image datablock, so nothing here writes
Blender data and a missing icon can be made from a panel's draw -- into one
preview collection, keyed by the colour as an 8-bit sRGB value.

The swatches are `subtype='COLOR'` properties -- scene-linear values the
colour button shows through the sRGB transform -- so an icon's pixels are the
same linear -> sRGB conversion, or it reads darker than its swatch."""

from __future__ import annotations

import math
import os
import struct
import tempfile
import zlib

_SIZE = 32
_PREVIEWS = None
_IDS: dict = {}
_FOLDER = os.path.join(tempfile.gettempdir(), "plantgen_map_icons")


def _srgb(c: float) -> float:
    """Scene-linear -> sRGB (the standard transfer curve)."""
    c = max(0.0, min(1.0, float(c)))
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


def _key(color) -> tuple:
    """The colour as the swatch shows it: sRGB, 0-255."""
    return tuple(int(round(_srgb(c) * 255)) for c in color[:3])


def _png(path: str, rows) -> None:
    """Write an RGBA image (rows of (r, g, b, a) floats, top row first)."""
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for r, g, b, a in row:
            raw += bytes(int(round(max(0.0, min(1.0, v)) * 255)) for v in (r, g, b, a))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    header = struct.pack(">IIBBBBB", _SIZE, _SIZE, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def _draw(color, style: str):
    """One icon's pixels: a rounded square (filled, or just its edge) and, for
    the map styles, a curve glyph."""
    fg = (color[0], color[1], color[2], 1.0)
    dark = (0.09, 0.09, 0.10, 1.0)
    clear = (0.0, 0.0, 0.0, 0.0)
    px = [[clear] * _SIZE for _ in range(_SIZE)]
    pad = 5 if style == "swatch" else 2
    for y in range(pad, _SIZE - pad):
        for x in range(pad, _SIZE - pad):
            cx = min(x - pad, _SIZE - 1 - pad - x)
            cy = min(y - pad, _SIZE - 1 - pad - y)
            if cx < 3 and cy < 3 and math.hypot(3 - cx, 3 - cy) > 3.2:
                continue
            if style != "off" or min(cx, cy) < 2:
                px[y][x] = fg
    if style != "swatch":
        stroke = dark if style == "on" else fg
        for k in range(240):
            t = k / 239.0
            s = t * t * (3.0 - 2.0 * t)
            x = 7 + 18 * t
            y = _SIZE - 1 - (8 + 16 * s + 3.0 * math.sin(t * math.pi * 2.0) * (1.0 - t))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx * dx + dy * dy <= 1:
                        xi, yi = int(round(x)) + dx, int(round(y)) + dy
                        if 0 <= xi < _SIZE and 0 <= yi < _SIZE:
                            px[yi][xi] = stroke
    return px


def _collection():
    global _PREVIEWS
    if _PREVIEWS is None:
        import bpy.utils.previews
        _PREVIEWS = bpy.utils.previews.new()
    return _PREVIEWS


def icon(color, style: str) -> int:
    """The icon id of `style` ('swatch' / 'on' / 'off') in `color`, made the
    first time it is asked for. 0 (no icon) if it cannot be made."""
    key = (_key(color), style)
    found = _IDS.get(key)
    if found is not None:
        return found
    try:
        os.makedirs(_FOLDER, exist_ok=True)
        name = "pgsmap2_{}_{}_{}_{}".format(*key[0], style)
        path = os.path.join(_FOLDER, name + ".png")
        if not os.path.exists(path):
            _png(path, _draw([c / 255.0 for c in key[0]], style))   # sRGB already
        pcoll = _collection()
        item = pcoll.get(name) or pcoll.load(name, path, 'IMAGE')
        _IDS[key] = item.icon_id
    except Exception:  # noqa: BLE001 -- an icon is cosmetic; never break a draw
        _IDS[key] = 0
    return _IDS[key]


def refresh(s) -> None:
    """Make the icons of every swatch on a plant (a colour just changed)."""
    colors = [s.plant_color] + [lv.color for lv in s.branch_levels] + \
        [c.color for c in s.components]
    for color in colors:
        for style in ("swatch", "on", "off"):
            icon(color, style)


def unregister() -> None:
    global _PREVIEWS
    if _PREVIEWS is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_PREVIEWS)
        _PREVIEWS = None
    _IDS.clear()
