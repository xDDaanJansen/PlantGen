"""Procedurally-generated custom icons — no external image files, no network.

Two things:
  * a flat "+/-" glyph for the per-parameter variance buttons (replaces Blender's
    generic PLUS icon).
  * live thumbnails of the Taper/Gravity curve shapes, for their popover buttons —
    a tiny sparkline of the actual authored curve instead of a static FCURVE icon.
    Only the ACTIVE object's curves are ever visible in the panel at once, so these
    are cached under fixed keys ("taper"/"gravity") and repainted in place whenever
    `live.py`'s existing curve-watcher timer detects the active object's curve
    points changed — no separate polling loop needed.

Custom preview icons render at their own true pixel colors (Blender does not tint
them to the theme like its own vector icons), so everything here is drawn light
grey-on-transparent to sit reasonably naturally on a button.
"""

import bpy.utils.previews as _previews

_SIZE = 32
_WHITE = (0.92, 0.92, 0.92, 1.0)
_pcoll = None


def _blank():
    return [0.0] * (_SIZE * _SIZE * 4)


def _set_px(pixels, x, y, rgba):
    """`x, y` are top-down authoring coordinates (0,0 = top-left, like every other
    image convention) — `icon_pixels_float` itself is bottom-up (row 0 = the
    bottom of the displayed icon, same as `Image.pixels`), so this is the one
    place that flips row order. Every caller above this may as well be drawing
    into a normal top-down canvas."""
    x, y = round(x), round(y)
    if 0 <= x < _SIZE and 0 <= y < _SIZE:
        row = _SIZE - 1 - y
        i = (row * _SIZE + x) * 4
        pixels[i:i + 4] = rgba


def _rect(pixels, x0, y0, x1, y1, rgba):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            _set_px(pixels, x, y, rgba)


def _line(pixels, x0, y0, x1, y1, rgba, thickness=1):
    steps = int(round(max(abs(x1 - x0), abs(y1 - y0), 1)))
    for i in range(steps + 1):
        t = i / steps
        cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        for dy in range(-(thickness // 2), thickness - thickness // 2):
            _set_px(pixels, cx, cy + dy, rgba)


def _bar(pixels, cx, cy, length, thickness, vertical, rgba):
    half_l, half_t = length / 2, thickness / 2
    if vertical:
        _rect(pixels, round(cx - half_t), round(cy - half_l),
              round(cx + half_t), round(cy + half_l), rgba)
    else:
        _rect(pixels, round(cx - half_l), round(cy - half_t),
              round(cx + half_l), round(cy + half_t), rgba)


def _variance_pixels():
    pixels = _blank()
    cx = _SIZE / 2
    arm, thickness = 11, 1.5
    plus_cy, minus_cy = 10, 22  # "+" above "-", both a similar-weight 11px/3px bar
    _bar(pixels, cx, plus_cy, arm, thickness, vertical=False, rgba=_WHITE)
    _bar(pixels, cx, plus_cy, arm, thickness, vertical=True, rgba=_WHITE)
    _bar(pixels, cx, minus_cy, arm, thickness, vertical=False, rgba=_WHITE)
    return pixels


def _dice_pixels():
    lo, hi = 5, 26
    pixels = _blank()
    _line(pixels, lo, lo, hi, lo, _WHITE, thickness=2)
    _line(pixels, hi, lo, hi, hi, _WHITE, thickness=2)
    _line(pixels, hi, hi, lo, hi, _WHITE, thickness=2)
    _line(pixels, lo, hi, lo, lo, _WHITE, thickness=2)
    for px, py in ((10, 10), (21, 10), (16, 16), (10, 21), (21, 21)):  # a die's "5" face
        _rect(pixels, px - 1, py - 1, px + 1, py + 1, _WHITE)
    return pixels


def _weight_pixels():
    """A simple balance scale: a vertical post, a horizontal beam, and a pan
    hanging at each end -- the Weight popup button next to Age/Health/Season."""
    pixels = _blank()
    cx, top, bottom = 16, 8, 24
    _bar(pixels, cx, (top + bottom) / 2, bottom - top, 2, vertical=True, rgba=_WHITE)
    _bar(pixels, cx, top, 20, 2, vertical=False, rgba=_WHITE)
    for px in (6, 26):
        _line(pixels, cx, top, px, top + 8, _WHITE, thickness=1)
        _rect(pixels, px - 4, top + 7, px + 4, top + 10, _WHITE)
    return pixels


def _curve_pixels(points: list[tuple[float, float]]) -> list[float]:
    pixels = _blank()
    pad = 3
    span = _SIZE - 2 * pad - 1

    def to_px(pt):
        x, y = pt
        return (pad + x * span, pad + (1.0 - min(max(y, 0.0), 1.0)) * span)

    pts = sorted(points)
    if len(pts) >= 2:
        prev = to_px(pts[0])
        for pt in pts[1:]:
            cur = to_px(pt)
            _line(pixels, prev[0], prev[1], cur[0], cur[1], _WHITE, thickness=2)
            prev = cur
    return pixels


def _ensure_pcoll():
    global _pcoll
    if _pcoll is None:
        _pcoll = _previews.new()
    return _pcoll


def _get_or_create(key: str):
    pcoll = _ensure_pcoll()
    prev = pcoll.get(key)
    if prev is None:
        prev = pcoll.new(key)
        prev.icon_size = (_SIZE, _SIZE)
    return prev


_variance_painted = False
_dice_painted = False
_weight_painted = False


def variance_icon_id() -> int:
    # `is_icon_custom` isn't a reliable "already painted" flag — it flips true the
    # moment `icon_size` is set in `_get_or_create`, before any pixels are written —
    # so track it ourselves instead.
    global _variance_painted
    prev = _get_or_create("variance")
    if not _variance_painted:
        prev.icon_pixels_float = _variance_pixels()
        _variance_painted = True
    return prev.icon_id


def dice_icon_id() -> int:
    """A die's "5" face -- the Fully Random toggle next to Age/Health/Season
    (see `ui._slider_with_variance`'s `random_prop`), same procedural/no-tint
    approach as `variance_icon_id` instead of a stock Blender icon."""
    global _dice_painted
    prev = _get_or_create("dice")
    if not _dice_painted:
        prev.icon_pixels_float = _dice_pixels()
        _dice_painted = True
    return prev.icon_id


def weight_icon_id() -> int:
    """A balance scale -- the Weight popup button next to Age/Health/Season
    (see `ui._slider_with_variance`'s `weight_prop`), same procedural/no-tint
    approach as `variance_icon_id` instead of a stock Blender icon."""
    global _weight_painted
    prev = _get_or_create("weight")
    if not _weight_painted:
        prev.icon_pixels_float = _weight_pixels()
        _weight_painted = True
    return prev.icon_id


def curve_icon_id(key: str) -> int:
    """`key` is "taper" or "gravity" — see `set_curve_icon` for repainting it."""
    return _get_or_create(key).icon_id


def set_curve_icon(key: str, points: list[tuple[float, float]]) -> None:
    """Repaint the `key` curve thumbnail from its current authored control points
    (`plant.raw_ramp(node).points`) — called by `live.py`'s watcher whenever the
    active object's curve changes."""
    _get_or_create(key).icon_pixels_float = _curve_pixels(points)


def register() -> None:
    _ensure_pcoll()


def unregister() -> None:
    global _pcoll, _variance_painted, _dice_painted, _weight_painted
    if _pcoll is not None:
        _previews.remove(_pcoll)
        _pcoll = None
    _variance_painted = False
    _dice_painted = False
    _weight_painted = False
