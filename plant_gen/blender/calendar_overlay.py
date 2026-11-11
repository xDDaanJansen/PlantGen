"""The Phenology Calendar -- a GPU-drawn, theme-styled floating window that
authors a plant's within-year phenology by dragging, and scrubs its Day.

Why GPU, not a native popup (the decision, recorded so it is not re-litigated):
a native-widget popup (`invoke_props_dialog` / popover) is Blender's own buttons
and so is perfectly native, but its layout engine has **no canvas** -- it cannot
draw a continuous ruler, coloured phenophase bars, week/month gridlines, or a
draggable playhead, and clicks land on widgets, not on pixels you could drag. A
calendar *is* a continuous time axis with a playhead, so it has to be drawn. The
"native look" is met by sourcing the chrome colours from `preferences.themes`,
the font size from `ui_styles`, following Blender's grey hierarchy (light panel,
darker graph, darkest fields) and its embossed, gradient widgets.

**Drawn in area space across every region so it is never hidden behind the
N-panel.** The draw handler is registered on WINDOW *and* the overlapping UI /
TOOLS / HEADER regions; each pass offsets the (area-local) geometry by its own
region origin, so the part of the window over a panel is redrawn *on top of*
that panel. Hit-testing is likewise area-local (`event.mouse_x - area.x`), so a
drag works even with the cursor over the panel.

Mechanically this mirrors the WS1 inspector (`overlay.py`): a modal operator
(`plantgen.toggle_calendar`) owns the mouse while open, and a session token
makes "at most one calendar live" true by construction.
"""

from __future__ import annotations

import math

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import bridge

# -- Calendar constants -------------------------------------------------------
_YEAR = 365.0
_MSTART = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
_MNAME = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Season band segments: the astronomical equinox/solstice boundaries (fixed in
# day-of-year), with the season NAMES rotating a half-year for the Southern
# hemisphere (same dates, Dec-Feb becomes summer). Tint follows the name.
_SEASON_BOUNDS = (0, 79, 172, 265, 355)
_SEASON_TINTS = {"Winter": (0.176, 0.196, 0.235), "Spring": (0.188, 0.227, 0.184),
                 "Summer": (0.235, 0.227, 0.176), "Autumn": (0.235, 0.200, 0.176)}
_SEASON_NAMES_N = ("Winter", "Spring", "Summer", "Autumn", "Winter")
_SEASON_NAMES_S = ("Summer", "Autumn", "Winter", "Spring", "Summer")
_SEASON_STARTS = (79, 172, 265, 355)

_C_SEASON = (0.373, 0.541, 0.290, 1.0)
_C_BLOOM = (0.753, 0.416, 0.596, 1.0)
_C_FRUIT = (0.690, 0.271, 0.239, 1.0)
_C_AUTUMN = (0.757, 0.502, 0.227, 1.0)
_C_DORM = (0.216, 0.216, 0.216, 1.0)
_C_DISABLED = (0.42, 0.42, 0.42, 0.5)
_PLAYHEAD = (0.357, 0.561, 0.839, 1.0)
_GRIP = (0.86, 0.86, 0.865, 0.95)
# (label, phase-key) -- the phase swatches, click-to-recolour in the legend.
_LEGEND = (("Season", "season"), ("Bloom", "bloom"), ("Fruit", "fruit"),
           ("Autumn", "autumn"), ("Dormant", "dormant"))
_DEFAULT_COLORS = {"season": _C_SEASON, "bloom": _C_BLOOM, "fruit": _C_FRUIT,
                   "autumn": _C_AUTUMN, "dormant": _C_DORM}

_W, _H, _WMIN, _HMIN, _H_COMPACT = 730.0, 352.0, 480.0, 210.0, 226.0
_TITLE_H, _TOOL_H, _GUTTER = 26.0, 30.0, 88.0

_LANES = (("season", "Season", _C_SEASON), ("bloom", "Bloom", _C_BLOOM),
          ("fruit", "Fruit", _C_FRUIT), ("autumn", "Autumn", _C_AUTUMN),
          ("dormant", "Dormant", _C_DORM))
_EVENTS = ("bloom", "fruit", "autumn")
_EVENT_DEFAULT = {"bloom": (55.0, 75.0), "fruit": (185.0, 95.0), "autumn": (180.0, 60.0)}

_state = {
    "session": 0, "object": "",
    "win": None, "size": None, "exp_h": None,
    "view": "expanded", "year": 0,
    "windows": None, "drag": None, "shift": False,
    "cursor": (0.0, 0.0), "drag_label": "", "L": None,
    "color_phase": "", "legend_rects": {},
}
_OFF = [0.0, 0.0]


# -- Theme (Blender's grey hierarchy, widget gradients, UI font size) ----------
def _lift(c, d):
    return (min(1.0, max(0.0, c[0] + d)), min(1.0, max(0.0, c[1] + d)),
            min(1.0, max(0.0, c[2] + d)), 1.0)


def _fs() -> float:
    try:
        return float(bpy.context.preferences.ui_styles[0].widget.points)
    except Exception:
        return 11.0


def _palette() -> dict:
    def wcol(name, fb_inner, fb_sel, fb_out, fb_top=15, fb_down=-15):
        try:
            w = getattr(bpy.context.preferences.themes[0].user_interface, name)
            return (tuple(w.inner)[:3], tuple(w.inner_sel)[:3], tuple(w.outline)[:3],
                    w.shadetop, w.shadedown)
        except Exception:
            return (fb_inner, fb_sel, fb_out, fb_top, fb_down)

    try:
        ui = bpy.context.preferences.themes[0].user_interface
        back = tuple(ui.wcol_menu_back.inner)[:3]
        txt = tuple(ui.wcol_menu_back.text)[:3]
        text = (txt[0], txt[1], txt[2], 1.0)
    except Exception:
        back = (0.19, 0.19, 0.19)
        text = (0.82, 0.82, 0.82, 1.0)
    # Buttons follow Blender's Tool widget (a mid-grey, embossed via shadetop/
    # shadedown), number fields the Num widget (dark), toggles the Option widget.
    t_in, t_sel, t_out, t_top, t_dn = wcol("wcol_tool", (0.33, 0.33, 0.33),
                                           (0.27, 0.44, 0.72), (0.10, 0.10, 0.10))
    n_in, _ns, _no, n_top, n_dn = wcol("wcol_num", (0.13, 0.13, 0.13),
                                       (0.27, 0.44, 0.72), (0.10, 0.10, 0.10))
    o_in, o_sel, o_out, _ot, _od = wcol("wcol_option", (0.10, 0.10, 0.10),
                                        (0.27, 0.44, 0.72), (0.10, 0.10, 0.10))
    try:
        t_text = tuple(bpy.context.preferences.themes[0].user_interface.wcol_tool.text)[:3]
    except Exception:
        t_text = (0.96, 0.96, 0.96)
    return {
        "panel": _lift(back, 0.10), "canvas": _lift(back, -0.045), "field": _lift(back, -0.10),
        "border": _lift(back, 0.22), "grid_week": _lift(back, 0.02),
        "grid_month": _lift(back, 0.14),
        # White text throughout (matching native Blender), with only the
        # disabled-phase label kept dim so a greyed lane still reads as off.
        "text": (*t_text, 1.0), "text_mute": (*t_text, 1.0),
        "text_off": (text[0] * 0.45, text[1] * 0.45, text[2] * 0.45, 1.0),
        "accent": _PLAYHEAD,
        "tool_top": _lift(t_in, t_top / 255.0), "tool_bot": _lift(t_in, t_dn / 255.0),
        "sel_top": _lift(t_sel, t_top / 255.0), "sel_bot": _lift(t_sel, t_dn / 255.0),
        "tool_out": (*t_out, 1.0), "tool_text": (*t_text, 1.0),
        "num_top": _lift(n_in, n_top / 255.0), "num_bot": _lift(n_in, n_dn / 255.0),
        "option": (*o_in, 1.0), "option_sel": (*o_sel, 1.0), "option_out": (*o_out, 1.0),
    }


def _colors() -> dict:
    """This plant's phase palette -- the per-object overrides the legend swatches
    edit, else the built-in defaults. Per-plant, so two plants in a scene can be
    tinted differently."""
    obj = active_object()
    if obj is None:
        return dict(_DEFAULT_COLORS)
    s = obj.plantgen
    out = {}
    for key, default in _DEFAULT_COLORS.items():
        v = getattr(s, "cal_col_" + key, None)
        out[key] = (v[0], v[1], v[2], 1.0) if v is not None else default
    return out


# -- Session / lifecycle ------------------------------------------------------
def active_object():
    name = _state["object"]
    return bpy.data.objects.get(name) if name else None


def is_active(obj) -> bool:
    return obj is not None and _state["object"] == obj.name


def _read_windows(obj) -> dict:
    s = obj.plantgen
    if s.calendar_authored:
        return {"bloom": (s.bloom_start, s.bloom_days, s.bloom_enabled),
                "autumn": (s.autumn_start, s.autumn_days, s.autumn_enabled),
                "fruit": (s.fruit_start, s.fruit_days, s.fruit_enabled)}
    cal = bridge._base_profile(obj).calendar

    def wc(win, key):
        if win is not None:
            return (win.start, win.duration, True)
        d = _EVENT_DEFAULT[key]
        return (d[0], d[1], False)

    if cal is None:
        return {k: (*_EVENT_DEFAULT[k], False) for k in _EVENTS}
    return {"bloom": wc(cal.bloom, "bloom"), "autumn": wc(cal.senescence, "autumn"),
            "fruit": wc(cal.fruit, "fruit")}


def refresh(obj) -> None:
    if is_active(obj):
        _state["windows"] = _read_windows(obj)


def _ensure_authored(obj) -> None:
    s = obj.plantgen
    if not s.calendar_authored:
        bridge._calendar_to_object(s, bridge._base_profile(obj).calendar)
        s.calendar_authored = True
    refresh(obj)


def start(obj) -> int:
    _state.update(object=obj.name, win=None, size=None, exp_h=None, drag=None, drag_label="",
                  year=int(obj.plantgen.day // _YEAR),
                  windows=_read_windows(obj), session=_state["session"] + 1)
    return _state["session"]


def stop() -> None:
    _state.update(object="", win=None, drag=None, windows=None, drag_label="",
                  session=_state["session"] + 1)


def session() -> int:
    return _state["session"]


def dragging() -> bool:
    return _state["drag"] is not None


def _date_label(doy) -> str:
    doy = int(max(0, min(364, round(doy))))
    m = 0
    while m < 11 and _MSTART[m + 1] <= doy:
        m += 1
    return f"{_MNAME[m]} {doy - _MSTART[m] + 1}"


# -- Geometry (area-local coords) ---------------------------------------------
def _layout(obj, area) -> dict:
    sc = bpy.context.preferences.system.ui_scale
    if _state["size"] is None:
        _state["size"] = (_W * sc, (_H_COMPACT if _state["view"] == "compact" else _H) * sc)
    W, H = _state["size"]
    if _state["win"] is None:
        _state["win"] = ((area.width - W) * 0.5, (area.height - H) * 0.5)
    x, yb = _state["win"]
    x = max(2.0, min(x, area.width - W - 2.0))
    yb = max(2.0, min(yb, area.height - H - 2.0))
    _state["win"] = (x, yb)
    top = yb + H

    px0, px1 = x + _GUTTER * sc, x + W - 14.0 * sc
    pw = px1 - px0

    def d2x(day):
        return px0 + (day / _YEAR) * pw

    canvas_top = top - (_TITLE_H + _TOOL_H) * sc
    head_y = canvas_top - 13 * sc
    graph_top = canvas_top - 19 * sc
    band_top = graph_top - 3 * sc
    band_bot = band_top - 13 * sc
    grid_top = band_bot - 5 * sc
    legend_y = yb + 15 * sc
    labels_y = legend_y + 24 * sc
    graph_bot = labels_y + 11 * sc
    grid_bot = graph_bot + 4 * sc

    s = obj.plantgen
    emergence = float(s.emergence_day)
    season_len = float(s.season_length_days)
    mature = min(_YEAR, emergence + season_len)
    windows = _state["windows"] or {k: (*_EVENT_DEFAULT[k], False) for k in _EVENTS}

    day = float(s.day)
    doy = day - _state["year"] * _YEAR
    play_x = d2x(doy) if 0.0 <= doy <= _YEAR else None
    play_label = None
    if play_x is not None:
        lw = _tw(f"Day {int(round(day))}", (_fs() - 1) * sc) + 14 * sc
        play_label = (play_x - lw * 0.5, head_y - 1 * sc, lw, 15 * sc)

    # Toolbar: a joined 3-segment year stepper and a 2-segment view toggle.
    tby = canvas_top + (_TOOL_H * sc - 21 * sc) * 0.5
    aw, fw = 22 * sc, 66 * sc
    sx = x + 12 * sc
    step = [(sx, tby, aw, 21 * sc), (sx + aw, tby, fw, 21 * sc), (sx + aw + fw, tby, aw, 21 * sc)]
    vx = sx + 2 * aw + fw + 16 * sc
    ev_w, cp_w = 76 * sc, 66 * sc
    view = [(vx, tby, ev_w, 21 * sc), (vx + ev_w, tby, cp_w, 21 * sc)]
    hemi = (vx + ev_w + cp_w + 14 * sc, tby, 46 * sc, 21 * sc)
    southern = bool(getattr(s, "calendar_southern", False))

    L = {
        "sc": sc, "fs": _fs(), "x": x, "yb": yb, "W": W, "H": H, "top": top, "view": _state["view"],
        "px0": px0, "px1": px1, "d2x": d2x, "emergence": emergence, "mature": mature,
        "season_len": season_len, "emergence_x": d2x(emergence), "mature_x": d2x(mature),
        "canvas_top": canvas_top, "head_y": head_y, "graph_top": graph_top,
        "band_top": band_top, "band_bot": band_bot, "grid_top": grid_top,
        "grid_bot": grid_bot, "graph_bot": graph_bot, "labels_y": labels_y, "legend_y": legend_y,
        "play_x": play_x, "play_label": play_label, "day": day, "doy": doy,
        "year": _state["year"],
        "close": (x + W - 27 * sc, top - _TITLE_H * sc, 27 * sc, _TITLE_H * sc),
        "title": (x, top - _TITLE_H * sc, W - 27 * sc, _TITLE_H * sc),
        "step": step, "view_btn": view, "hemi": hemi, "southern": southern,
        "colors": _colors(),
    }
    cols = L["colors"]

    lane_h = (grid_top - grid_bot) / len(_LANES)
    lanes, bands, checks = [], [], {}
    for i, (key, label, _default_col) in enumerate(_LANES):
        colour = cols[key]
        cy = grid_top - i * lane_h - lane_h * 0.5
        bh = min(lane_h * 0.5, 22.0 * sc)
        y0, y1 = cy - bh * 0.5, cy + bh * 0.5
        if key == "season":
            spans, enabled = [(emergence, mature)], True
        elif key == "dormant":
            spans = []
            if emergence > 0:
                spans.append((0.0, emergence))
            if emergence + season_len < _YEAR:
                spans.append((min(_YEAR, emergence + season_len), _YEAR))
            enabled = True
        else:
            st, du, enabled = windows[key]
            spans = [(emergence + st, emergence + st + du)]
        lanes.append({"key": key, "label": label, "colour": colour, "enabled": enabled,
                      "cy": cy, "y0": y0, "y1": y1, "spans": spans})
        if key in _EVENTS:
            checks[key] = (x + 11 * sc, cy - 7 * sc, 14 * sc, 14 * sc)
        if key != "dormant":
            a, b = spans[0]
            bands.append({"key": key, "a": a, "b": b, "xa": d2x(a), "xb": min(d2x(b), px1),
                          "y0": y0, "y1": y1, "enabled": enabled})
    L["lanes"], L["bands"], L["checks"] = lanes, bands, checks
    return L


# -- Draw primitives (offset into the current region) -------------------------
_UNIFORM = None
_SMOOTH = None


def _shader():
    global _UNIFORM
    if _UNIFORM is None:
        _UNIFORM = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _UNIFORM


def _tris(verts, rgba):
    verts = [(vx - _OFF[0], vy - _OFF[1]) for vx, vy in verts]
    sh = _shader()
    batch = batch_for_shader(sh, 'TRIS', {"pos": verts})
    sh.bind()
    sh.uniform_float("color", rgba)
    batch.draw(sh)


def _quad(x, y, w, h, rgba):
    _tris(((x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)), rgba)


_ALL = ("tr", "tl", "bl", "br")


def _round_pts(x, y, w, h, r, corners=_ALL):
    r = max(0.0, min(r, w * 0.5, h * 0.5))
    if r <= 0.5:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    pts = []
    specs = (("tr", x + w - r, y + h - r, 0.0, (x + w, y + h)),
             ("tl", x + r, y + h - r, 90.0, (x, y + h)),
             ("bl", x + r, y + r, 180.0, (x, y)),
             ("br", x + w - r, y + r, 270.0, (x + w, y)))
    for key, cx, cy, a0, sharp in specs:
        if key in corners:
            for k in range(5):
                a = math.radians(a0 + 90.0 * k / 4)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        else:
            pts.append(sharp)
    return pts


def _fill_poly(pts, rgba):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    tris = []
    for i in range(len(pts)):
        tris += [(cx, cy), pts[i], pts[(i + 1) % len(pts)]]
    _tris(tris, rgba)


def _round_quad(x, y, w, h, r, rgba, corners=_ALL):
    _fill_poly(_round_pts(x, y, w, h, r, corners), rgba)


def _grad_round(x, y, w, h, r, top, bot, corners=_ALL):
    global _SMOOTH
    if _SMOOTH is None:
        _SMOOTH = gpu.shader.from_builtin('SMOOTH_COLOR')
    pts = _round_pts(x, y, w, h, r, corners)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    def col(py):
        t = (py - y) / h if h > 0 else 0.0
        return tuple(bot[k] + (top[k] - bot[k]) * max(0.0, min(1.0, t)) for k in range(4))

    verts, cols = [], []
    for i in range(len(pts)):
        for pnt in ((cx, cy), pts[i], pts[(i + 1) % len(pts)]):
            verts.append((pnt[0] - _OFF[0], pnt[1] - _OFF[1]))
            cols.append(col(pnt[1]))
    batch = batch_for_shader(_SMOOTH, 'TRIS', {"pos": verts, "color": cols})
    _SMOOTH.bind()
    batch.draw(_SMOOTH)


def _bordered(x, y, w, h, r, fill, border, bw):
    _round_quad(x - bw, y - bw, w + 2 * bw, h + 2 * bw, r + bw, border)
    _round_quad(x, y, w, h, r, fill)


def _line(x1, y1, x2, y2, rgba, width=1.0):
    gpu.state.line_width_set(width)
    sh = _shader()
    batch = batch_for_shader(sh, 'LINES', {"pos": ((x1 - _OFF[0], y1 - _OFF[1]),
                                                   (x2 - _OFF[0], y2 - _OFF[1]))})
    sh.bind()
    sh.uniform_float("color", rgba)
    batch.draw(sh)
    gpu.state.line_width_set(1.0)


def _lines(segs, rgba, width=1.0):
    if not segs:
        return
    pts = []
    for (a, b) in segs:
        pts += [(a[0] - _OFF[0], a[1] - _OFF[1]), (b[0] - _OFF[0], b[1] - _OFF[1])]
    gpu.state.line_width_set(width)
    sh = _shader()
    batch = batch_for_shader(sh, 'LINES', {"pos": pts})
    sh.bind()
    sh.uniform_float("color", rgba)
    batch.draw(sh)
    gpu.state.line_width_set(1.0)


def _diamond(cx, cy, r, rgba):
    _fill_poly([(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)], rgba)


def _hatch(rx, ry, rw, rh, rgba, sc):
    g = 6.0 * sc
    segs = []
    c = (ry - (rx + rw))
    c -= c % g
    while c <= ry + rh - rx:
        x_lo = max(rx, ry - c)
        x_hi = min(rx + rw, ry + rh - c)
        if x_lo < x_hi:
            segs.append(((x_lo, x_lo + c), (x_hi, x_hi + c)))
        c += g
    _lines(segs, rgba, 3.0 * sc)


_FONT = 0


def _text(x, y, s, rgba, size, anchor="left"):
    blf.size(_FONT, max(8, int(size)))
    if anchor != "left":
        w = blf.dimensions(_FONT, s)[0]
        x = x - w if anchor == "right" else x - w * 0.5
    blf.color(_FONT, *rgba)
    blf.position(_FONT, x - _OFF[0], y - _OFF[1], 0)
    blf.draw(_FONT, s)


def _tw(s, size):
    blf.size(_FONT, max(8, int(size)))
    return blf.dimensions(_FONT, s)[0]


def _checkbox(rect, on, p, sc):
    inner = p["option_sel"] if on else p["option"]
    _bordered(rect[0], rect[1], rect[2], rect[3], 3 * sc, inner, p["option_out"], 1.0 * sc)
    if on:
        s0, x0, y0 = rect[3], rect[0], rect[1]
        a = (x0 + 0.24 * s0, y0 + 0.48 * s0)
        b = (x0 + 0.43 * s0, y0 + 0.28 * s0)
        cc = (x0 + 0.76 * s0, y0 + 0.70 * s0)
        _lines([(a, b), (b, cc)], (0.95, 0.95, 0.96, 1.0), 2.0 * sc)


def _seg_group(rects, kinds, labels, p, sc, fs):
    """A joined segmented control (the year stepper, the view toggle): one
    rounded outer outline, straight *shared* internal edges (so abutting buttons
    are not each individually rounded), a per-segment gradient and a divider --
    Blender's own toggle-row look."""
    x0, yb, h = rects[0][0], rects[0][1], rects[0][3]
    tot = sum(r[2] for r in rects)
    rr = 4 * sc
    _round_quad(x0 - 1 * sc, yb - 1 * sc, tot + 2 * sc, h + 2 * sc, rr + 1 * sc, p["tool_out"])
    for i, (rc, kind, label) in enumerate(zip(rects, kinds, labels, strict=True)):
        corners = set()
        if i == 0:
            corners |= {"tl", "bl"}
        if i == len(rects) - 1:
            corners |= {"tr", "br"}
        if kind == "sel":
            top, bot = p["sel_top"], p["sel_bot"]
        elif kind == "field":
            top, bot = p["num_top"], p["num_bot"]
        else:
            top, bot = p["tool_top"], p["tool_bot"]
        _grad_round(rc[0], rc[1], rc[2], rc[3], rr, top, bot, tuple(corners))
        if i > 0:
            _line(rc[0], yb + 2 * sc, rc[0], yb + h - 2 * sc, p["tool_out"], 1.0 * sc)
        col = (1, 1, 1, 1) if kind == "sel" else p["tool_text"]
        _text(rc[0] + rc[2] * 0.5, rc[1] + rc[3] * 0.5 - fs * 0.36, label, col, fs, "center")


# -- Draw ---------------------------------------------------------------------
_REGIONS = ('WINDOW', 'UI', 'TOOLS', 'HEADER')


def _draw() -> None:
    obj = active_object()
    if obj is None:
        return
    region = bpy.context.region
    area = bpy.context.area
    space = bpy.context.space_data
    if (region is None or area is None or space is None or space.type != 'VIEW_3D'
            or region.type not in _REGIONS):
        return
    _OFF[0] = region.x - area.x
    _OFF[1] = region.y - area.y
    p = _palette()
    L = _layout(obj, area)
    _state["L"] = L
    sc, fs = L["sc"], L["fs"]
    gpu.state.blend_set('ALPHA')

    _bordered(L["x"], L["yb"], L["W"], L["H"], 6 * sc, p["panel"], p["border"], 1.2 * sc)
    _line(L["x"] + 6 * sc, L["canvas_top"], L["x"] + L["W"] - 6 * sc, L["canvas_top"],
          p["border"], 1.0 * sc)
    _round_quad(L["px0"] - 6 * sc, L["graph_bot"], (L["px1"] - L["px0"]) + 12 * sc,
                L["graph_top"] - L["graph_bot"], 5 * sc, p["canvas"])

    title_y = L["top"] - _TITLE_H * sc + (_TITLE_H * sc - fs * sc) * 0.5
    _text(L["x"] + 12 * sc, title_y, "Phenology Calendar", p["text"], fs * sc)
    tw = _tw("Phenology Calendar", fs * sc)
    _text(L["x"] + 12 * sc + tw + 8 * sc, title_y, "· " + obj.name, p["text_mute"], (fs - 1) * sc)
    _text(L["x"] + L["W"] - 13 * sc, title_y, "✕", p["text_mute"], (fs + 2) * sc, "right")

    _seg_group(L["step"], ("tool", "field", "tool"),
               ("◀", f"Year {L['year']}", "▶"), p, sc, fs * sc)
    view_kinds = ("sel" if L["view"] == "expanded" else "tool",
                  "sel" if L["view"] == "compact" else "tool")
    _seg_group(L["view_btn"], view_kinds, ("Expanded", "Compact"), p, sc, fs * sc)
    _seg_group([L["hemi"]], ("tool",), ("◒ S" if L["southern"] else "◓ N",), p, sc, fs * sc)
    if _state["drag"] and _state["drag_label"]:
        readout = _state["drag_label"]
    else:
        readout = (f"Emergence {_date_label(L['emergence'])} · "
                   f"Season {int(L['season_len'])} d")
    _text(L["x"] + L["W"] - 13 * sc, L["step"][0][1] + 6 * sc, readout,
          p["text_mute"], (fs - 1) * sc, "right")

    d2x = L["d2x"]
    names = _SEASON_NAMES_S if L["southern"] else _SEASON_NAMES_N
    for i, s0 in enumerate(_SEASON_BOUNDS):
        end = _SEASON_BOUNDS[i + 1] if i + 1 < len(_SEASON_BOUNDS) else _YEAR
        name = names[i]
        xa, xb = d2x(s0), d2x(end)
        _quad(xa, L["band_bot"], xb - xa, L["band_top"] - L["band_bot"],
              (*_SEASON_TINTS[name], 1.0))
        if end - s0 > 20:
            _text((xa + xb) * 0.5, L["band_bot"] + 4 * sc, name,
                  p["text_mute"], (fs - 2) * sc, "center")

    _lines([((d2x(d), L["grid_bot"]), (d2x(d), L["grid_top"]))
            for d in range(7, 365, 7) if d not in _MSTART], p["grid_week"], 1.0 * sc)
    _lines([((d2x(m), L["grid_bot"]), (d2x(m), L["grid_top"])) for m in _MSTART],
           p["grid_month"], 1.6 * sc)
    _lines([((d2x(t), L["grid_bot"]), (d2x(t), L["band_top"])) for t in _SEASON_STARTS],
           p["grid_month"], 3.0 * sc)
    for i in range(12):
        cx = d2x((_MSTART[i] + _MSTART[i + 1]) * 0.5)
        _text(cx, L["labels_y"], _MNAME[i], p["text_mute"], (fs - 1.5) * sc, "center")

    if L["view"] == "compact":
        _draw_compact(L, p, sc)
    else:
        _draw_expanded(L, p, sc)
    _draw_legend(L, p, sc, L["view"] == "compact")

    if L["play_label"] is not None:
        hx, pl = L["play_x"], L["play_label"]
        _round_quad(pl[0], pl[1], pl[2], pl[3], 3 * sc, p["accent"])
        _text(hx, L["head_y"] + 3 * sc, f"Day {int(round(L['day']))}",
              (1, 1, 1, 1), (fs - 1) * sc, "center")
        _line(hx, L["grid_bot"], hx, L["head_y"], p["accent"], 2.0)

    if _state["drag"] and _state["drag_label"] and _state["drag"][0] == "band":
        mx, my = _state["cursor"]
        tw2 = _tw(_state["drag_label"], (fs - 1) * sc)
        _bordered(mx + 14 * sc, my + 14 * sc, tw2 + 12 * sc, 18 * sc, 3 * sc,
                  p["field"], p["border"], 1.0 * sc)
        _text(mx + 20 * sc, my + 19 * sc, _state["drag_label"], p["text"], (fs - 1) * sc)

    gpu.state.blend_set('NONE')


def _draw_expanded(L, p, sc) -> None:
    d2x = L["d2x"]
    fs = L["fs"]
    for lane in L["lanes"]:
        enabled = lane["enabled"]
        cy = lane["cy"]
        lab = p["text"] if enabled else p["text_off"]
        _text(L["x"] + 32 * sc, cy - fs * sc * 0.36, lane["label"], lab, fs * sc)
        if lane["key"] in _EVENTS:
            _checkbox(L["checks"][lane["key"]], enabled, p, sc)
        col = lane["colour"] if enabled else _C_DISABLED
        for (a, b) in lane["spans"]:
            xa, xb = d2x(a), min(d2x(b), L["px1"])
            _round_quad(xa, lane["y0"], max(2 * sc, xb - xa), lane["y1"] - lane["y0"], 3 * sc, col)
            if enabled and lane["key"] != "dormant":
                for gx in (xa, xb):
                    _round_quad(gx - 1.5 * sc, cy - 9 * sc, 3 * sc, 18 * sc, 1.2 * sc, _GRIP)
    ex = L["emergence_x"]
    _lines([((ex, y), (ex, y + 4 * sc)) for y in
            range(int(L["grid_bot"]), int(L["band_top"]), max(1, int(8 * sc)))],
           (*_GRIP[:3], 0.45), 1.0 * sc)
    _diamond(ex, (L["band_top"] + L["band_bot"]) * 0.5, 6 * sc, _GRIP)


def _draw_compact(L, p, sc) -> None:
    d2x = L["d2x"]
    bar_h = 30 * sc
    by = (L["grid_top"] + L["grid_bot"]) * 0.5 - bar_h * 0.5
    _round_quad(d2x(0), by, d2x(_YEAR) - d2x(0), bar_h, 6 * sc, L["colors"]["dormant"])
    _quad(d2x(L["emergence"]), by, d2x(L["mature"]) - d2x(L["emergence"]),
          bar_h, L["colors"]["season"])
    evs = []
    for lane in L["lanes"]:
        if lane["key"] in _EVENTS and lane["enabled"] and lane["spans"]:
            evs.append((lane["spans"][0][0], lane["spans"][0][1], lane["colour"]))
    evs.sort(key=lambda e: e[0])
    inset = 5 * sc
    for a, b, c in evs:
        _round_quad(d2x(a), by + inset, max(2 * sc, d2x(min(b, _YEAR)) - d2x(a)),
                    bar_h - 2 * inset, 3 * sc, c)
    for i in range(len(evs)):
        for j in range(i + 1, len(evs)):
            lo, hi = max(evs[i][0], evs[j][0]), min(evs[i][1], evs[j][1])
            if lo < hi:
                _hatch(d2x(lo), by + inset, d2x(min(hi, _YEAR)) - d2x(lo),
                       bar_h - 2 * inset, evs[i][2], sc)
    # Emergence diamond on the Season/Dormant boundary, just above the bar.
    _diamond(L["emergence_x"], by + bar_h + 6 * sc, 6 * sc, _GRIP)


def _draw_legend(L, p, sc, compact) -> None:
    fs = (L["fs"] - 1.5) * sc
    x, y = L["x"] + 14 * sc, L["legend_y"]
    gap, sw = 22 * sc, 13 * sc
    # Day playhead + Emergence markers, then the phase colours.
    _round_quad(x, y - 1 * sc, 3 * sc, 12 * sc, 1 * sc, p["accent"])
    _text(x + 9 * sc, y + 1 * sc, "Day", p["text_mute"], fs)
    x += 9 * sc + _tw("Day", fs) + gap
    _diamond(x + 4 * sc, y + 5 * sc, 5 * sc, _GRIP)
    _text(x + 13 * sc, y + 1 * sc, "Emergence", p["text_mute"], fs)
    x += 13 * sc + _tw("Emergence", fs) + gap
    cols = L["colors"]
    rects = {}
    for label, key in _LEGEND:
        _bordered(x, y, sw, 11 * sc, 2 * sc, cols[key], p["border"], 1.0 * sc)
        rects[key] = (x - 2 * sc, y - 2 * sc, sw + 4 * sc, 15 * sc)
        _text(x + sw + 6 * sc, y + 1 * sc, label, p["text_mute"], fs)
        x += sw + 6 * sc + _tw(label, fs) + gap
    _state["legend_rects"] = rects
    if compact:
        _round_quad(x, y, sw, 11 * sc, 2 * sc, cols["autumn"])
        _hatch(x, y, sw, 11 * sc, cols["fruit"], sc)
        _text(x + sw + 6 * sc, y + 1 * sc, "overlap", p["text_mute"], fs)


# -- Interaction (area-local coords from the operator) ------------------------
def _in(mx, my, rect):
    return rect[0] <= mx <= rect[0] + rect[2] and rect[1] <= my <= rect[1] + rect[3]


def _x2day(mx, L):
    return max(0.0, min(_YEAR, (mx - L["px0"]) / (L["px1"] - L["px0"]) * _YEAR))


def _snap(day, shift):
    step = 1.0 if shift else 7.0
    return round(day / step) * step


def _scrub_day(obj, mx, L) -> None:
    day = _snap(_x2day(mx, L), _state.get("shift", False))
    obj.plantgen.day = _state["year"] * _YEAR + day
    _state["drag_label"] = f"Day {int(obj.plantgen.day)} · {_date_label(day)}"


def _resize_zone(mx, my, L):
    m = 6 * L["sc"]
    x, yb, W, H = L["x"], L["yb"], L["W"], L["H"]
    if not (x - m <= mx <= x + W + m and yb - m <= my <= yb + H + m):
        return ""
    z = ""
    if abs(mx - x) <= m:
        z += "l"
    if abs(mx - (x + W)) <= m:
        z += "r"
    if abs(my - yb) <= m:
        z += "b"
    return z


def cursor_hint(mx, my):
    L = _state["L"]
    if L is None or _state["drag"]:
        return None
    z = _resize_zone(mx, my, L)
    if z in ("l", "r"):
        return 'MOVE_X'
    if z == "b":
        return 'MOVE_Y'
    if z:
        return 'SCROLL_XY'
    if L["x"] <= mx <= L["x"] + L["W"] and L["yb"] <= my <= L["yb"] + L["H"]:
        return 'DEFAULT'
    return None


def _set_view(view) -> None:
    if _state["view"] == view or _state["size"] is None:
        _state["view"] = view
        return
    w, h = _state["size"]
    if view == "compact":
        _state["exp_h"] = h
        _state["win"] = (_state["win"][0], _state["win"][1] + (h - _H_COMPACT * _sc()))
        _state["size"] = (w, _H_COMPACT * _sc())
    else:
        newh = _state.get("exp_h") or _H * _sc()
        _state["win"] = (_state["win"][0], _state["win"][1] + (h - newh))
        _state["size"] = (w, newh)
    _state["view"] = view


def _sc():
    return bpy.context.preferences.system.ui_scale


def press(obj, mx, my, shift) -> str:
    L = _state["L"]
    if L is None:
        return "pass"
    sc = L["sc"]
    zone = _resize_zone(mx, my, L)
    if zone:
        _state["drag"] = ("resize", zone)
        return "consume"
    if not (L["x"] <= mx <= L["x"] + L["W"] and L["yb"] <= my <= L["yb"] + L["H"]):
        return "pass"
    if _in(mx, my, L["close"]):
        return "close"
    if _in(mx, my, L["step"][0]):
        _step_year(obj, -1)
        return "consume"
    if _in(mx, my, L["step"][2]):
        _step_year(obj, +1)
        return "consume"
    if _in(mx, my, L["step"][1]):
        return "year"
    if _in(mx, my, L["view_btn"][0]):
        _set_view("expanded")
        return "consume"
    if _in(mx, my, L["view_btn"][1]):
        _set_view("compact")
        return "consume"
    if _in(mx, my, L["hemi"]):
        s = obj.plantgen
        s.calendar_southern = not s.calendar_southern
        return "consume"
    for key, rect in _state.get("legend_rects", {}).items():
        if _in(mx, my, rect):
            _state["color_phase"] = key
            return "color"
    if L["view"] == "expanded":
        for key, cb in L["checks"].items():
            if _in(mx, my, cb):
                _toggle_phase(obj, key)
                return "consume"
    if _in(mx, my, L["title"]):
        _state["drag"] = ("move", mx - L["x"], my - L["yb"])
        return "consume"
    if L["view"] == "expanded" and abs(mx - L["emergence_x"]) <= 8 * sc \
            and L["band_bot"] - 6 * sc <= my <= L["band_top"] + 4 * sc:
        _state["drag"] = ("band", "season", "start", 0.0)
        return "consume"
    if L["view"] == "expanded":
        for b in L["bands"]:
            # A disabled (greyed) phase is display-only -- not draggable -- until
            # its checkbox re-enables it. Season is always enabled.
            if b["enabled"] and b["y0"] - 4 * sc <= my <= b["y1"] + 4 * sc:
                if abs(mx - b["xa"]) <= 7 * sc:
                    _state["drag"] = ("band", b["key"], "start", 0.0)
                    return "consume"
                if abs(mx - b["xb"]) <= 7 * sc:
                    _state["drag"] = ("band", b["key"], "end", 0.0)
                    return "consume"
                if b["xa"] < mx < b["xb"]:
                    _state["drag"] = ("band", b["key"], "body", _x2day(mx, L) - b["a"])
                    return "consume"
    # The "Day" label box is a grab handle for the playhead too (it sits above
    # the track), so grabbing it drags without jumping.
    if L["play_label"] is not None and _in(mx, my, L["play_label"]):
        _state["drag"] = ("day", 0, 0)
        return "consume"
    # Day scrub -- click anywhere on the ruler/track (not on a band) jumps the
    # playhead there and starts a drag, so Day is a real slider, including in the
    # compact view where every other element is display-only.
    if L["px0"] - 4 * sc <= mx <= L["px1"] + 4 * sc \
            and L["grid_bot"] <= my <= L["band_top"]:
        _state["drag"] = ("day", 0, 0)
        _scrub_day(obj, mx, L)
        return "consume"
    return "consume"


def move(obj, mx, my) -> bool:
    L = _state["L"]
    _state["cursor"] = (mx, my)
    drag = _state["drag"]
    if drag is None:
        if L is None:
            return False
        if _resize_zone(mx, my, L):
            return True
        return L["x"] <= mx <= L["x"] + L["W"] and L["yb"] <= my <= L["yb"] + L["H"]
    kind = drag[0]
    sc = L["sc"] if L else _sc()
    if kind == "move":
        _state["win"] = (mx - drag[1], my - drag[2])
        return True
    if kind == "resize":
        zone = drag[1]
        x0, yb0 = _state["win"]
        w0, h0 = _state["size"]
        left, right, bottom, top = x0, x0 + w0, yb0, yb0 + h0
        if "l" in zone:
            left = min(mx, right - _WMIN * sc)
        if "r" in zone:
            right = max(mx, left + _WMIN * sc)
        if "b" in zone:
            bottom = min(my, top - _HMIN * sc)
        _state["win"] = (left, bottom)
        _state["size"] = (right - left, top - bottom)
        return True
    if L is None:
        return True
    if kind == "day":
        _scrub_day(obj, mx, L)
        return True
    if kind == "band":
        _apply_band(obj, drag, mx, L)
        return True
    return True


def _apply_band(obj, drag, mx, L) -> None:
    _, key, mode, anchor = drag
    day = _snap(_x2day(mx, L), _state.get("shift", False))
    s = obj.plantgen
    if key == "season":
        if mode == "start":
            s.emergence_day = day
            _state["drag_label"] = f"Emergence {_date_label(day)}"
        elif mode == "end":
            s.season_length_days = max(1.0, day - float(s.emergence_day))
            _state["drag_label"] = f"Season {int(s.season_length_days)} d"
        else:
            s.emergence_day = max(0.0, day - anchor)
            _state["drag_label"] = f"Emergence {_date_label(s.emergence_day)}"
        return
    _ensure_authored(obj)
    emergence = float(s.emergence_day)
    if mode == "start":
        setattr(s, key + "_start", max(0.0, day - emergence))
        _state["drag_label"] = f"{key.title()} start {_date_label(day)}"
    elif mode == "end":
        start = getattr(s, key + "_start")
        setattr(s, key + "_days", max(1.0, day - emergence - start))
        _state["drag_label"] = f"{key.title()} {int(getattr(s, key + '_days'))} d"
    else:
        setattr(s, key + "_start", max(0.0, (day - anchor) - emergence))
        _state["drag_label"] = f"{key.title()} start {_date_label(day - anchor)}"
    refresh(obj)


def release() -> None:
    _state["drag"] = None
    _state["drag_label"] = ""


def set_shift(on: bool) -> None:
    _state["shift"] = bool(on)


def _toggle_phase(obj, key) -> None:
    _ensure_authored(obj)
    s = obj.plantgen
    setattr(s, key + "_enabled", not getattr(s, key + "_enabled"))
    refresh(obj)


def _step_year(obj, delta) -> None:
    _state["year"] = max(0, _state["year"] + delta)
    s = obj.plantgen
    s.day = _state["year"] * _YEAR + (float(s.day) % _YEAR)


def current_year() -> int:
    return _state["year"]


def color_phase() -> str:
    return _state["color_phase"]


def set_year(year) -> None:
    """Jump to a typed year (from the year field's popup), keeping day-of-year."""
    _state["year"] = max(0, int(year))
    obj = active_object()
    if obj is not None:
        s = obj.plantgen
        s.day = _state["year"] * _YEAR + (float(s.day) % _YEAR)


# -- Registration (one handler per overlapping region) ------------------------
_handlers = []


def register() -> None:
    unregister()
    for rt in _REGIONS:
        _handlers.append((rt, bpy.types.SpaceView3D.draw_handler_add(_draw, (), rt, 'POST_PIXEL')))


def unregister() -> None:
    for rt, h in _handlers:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, rt)
        except (ValueError, RuntimeError):
            pass
    _handlers.clear()
    stop()
