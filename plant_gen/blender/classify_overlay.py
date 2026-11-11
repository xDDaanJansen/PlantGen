"""The Classify view -- a GPU-drawn 2D scatter of the plant's organ points, one
channel across and one up, each point coloured by the component that owns it,
with the ACTIVE component's mask window on those two channels drawn as a
rectangle you drag. Interactive classification (Natsura-parity, clean-room):
instead of typing Min/Max for two filters and regrowing to see who went where,
you see the point cloud in the two dimensions you are classifying by and draw
the window around the points you want.

What the view shows and edits:
- **Points** are the active component's *candidate pool* -- the organ points of
  its type (leaf / bract / bud / fruit / flower), the ones it competes for. A
  point is drawn in the colour of the component that claimed it at the last
  regrow (`instancing.LAST_POINTS`), faint when that component then dropped it
  (Prune, or a mask on a generated point), grey when nothing claimed it. Over
  ~5000 points a seeded, stable subsample is drawn (counts use every point).
- **The rectangle** is the window [lo, hi] of the active component's first
  un-muted mask entry on each axis channel ([0, 1] when it has none). Dragging
  an edge / corner / the body, or dragging out a new rectangle on empty plot,
  writes that entry's `lo` / `hi` -- creating a Multiply entry for a channel the
  component has no entry for. Those props carry `_changed`, so the live
  debounce regrows the plant and the scatter recolours; the view itself never
  regrows or touches growth.
- **Binary axes** (`masks.BINARY_CHANNELS`, a class) are offered -- the split
  they draw is useful to see -- but their rectangle edges are inert: those
  channels ignore lo/hi (`masks.effective_window`), so the rect shows the pinned
  window and cannot be dragged on that axis. Only the window is edited -- an
  entry's invert / combine op / curve stay as set in the panel (the footer notes
  an inverted window, which keeps the OUTSIDE of the rect).

Mechanically this mirrors the Phenology Calendar (`calendar_overlay`): drawn in
area space over every region so the N-panel never hides it, hit-tested in
area-local coordinates, owned by a modal operator (`plantgen.classify_view`)
with a session token so at most one is live.
"""

from __future__ import annotations

import math

import blf
import bpy
import gpu
import numpy as np
from bpy.types import Operator
from gpu_extras.batch import batch_for_shader

from ..growth import masks
from . import bridge, calendar_overlay, instancing, live

_W, _H = 340.0, 340.0
_TITLE_H = 24.0
_MAX_POINTS = 5000
_SUBSAMPLE_SALT = 0x5EED_C1A5
_GREY = (0.52, 0.52, 0.52, 0.55)
_HANDLE = (0.95, 0.95, 0.96, 1.0)
_MASK_COLORS = (
    (0.85, 0.35, 0.30), (0.30, 0.55, 0.85), (0.35, 0.75, 0.40),
    (0.85, 0.70, 0.25), (0.65, 0.40, 0.80), (0.30, 0.72, 0.72),
)

_state = {"session": 0, "object": "", "win": None, "drag": None, "L": None,
          "edited": False}
_OFF = [0.0, 0.0]


# -- Session / lifecycle ------------------------------------------------------
def active_object():
    name = _state["object"]
    return bpy.data.objects.get(name) if name else None


def is_active(obj) -> bool:
    return obj is not None and _state["object"] == obj.name


def start(obj) -> int:
    _state.update(object=obj.name, drag=None, L=None, edited=False,
                  session=_state["session"] + 1)
    return _state["session"]


def stop() -> None:
    _state.update(object="", drag=None, L=None, session=_state["session"] + 1)


def session() -> int:
    return _state["session"]


def dragging() -> bool:
    return _state["drag"] is not None


# -- Axes and the active component's window -----------------------------------
def active_component(s):
    i = s.components_active
    return s.components[i] if 0 <= i < len(s.components) else None


def axis(s, which: str) -> dict:
    """One axis: its channel, the channel-dict key it reads, a label, and whether
    its window can be dragged (a binary / class channel ignores lo/hi)."""
    ch = getattr(s, f"classify_{which}")
    attr = getattr(s, f"classify_{which}_attr")
    cls = getattr(s, f"classify_{which}_class")
    label = s.bl_rna.properties[f"classify_{which}"].enum_items[ch].name
    if ch == "custom":
        key, label = masks.CUSTOM_PREFIX + attr, attr or "(no attribute)"
    elif ch == "class":
        key, label = masks.CLASS_PREFIX + cls, (cls or "(no class)")
    else:
        key = ch
    windowable = (ch not in masks.BINARY_CHANNELS and ch != "class"
                  and not (ch == "custom" and not attr))
    return {"channel": ch, "attr": attr, "cls": cls, "key": key, "label": label,
            "windowable": windowable}


def _matches(e, ax) -> bool:
    if e.channel != ax["channel"]:
        return False
    if ax["channel"] == "custom":
        return e.attr_name == ax["attr"]
    if ax["channel"] == "class":
        return e.class_name == ax["cls"]
    return True


def find_entry(c, ax):
    """(index, entry) of the component's first un-muted mask entry on this axis's
    channel -- the one the rectangle shows and edits -- or (-1, None)."""
    if c is None:
        return -1, None
    for i, e in enumerate(c.filters):
        if not e.mute and _matches(e, ax):
            return i, e
    return -1, None


def window(c, ax) -> tuple[float, float, bool]:
    """(lo, hi, inverted) the rectangle shows on this axis."""
    _i, e = find_entry(c, ax)
    if e is None:
        return 0.0, 1.0, False
    lo, hi = masks.effective_window(ax["key"], float(e.lo), float(e.hi))
    return min(lo, hi), max(lo, hi), bool(e.invert)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def set_window(obj, which: str, lo: float, hi: float) -> bool:
    """THE write path every drag goes through: put [lo, hi] on the active
    component's entry for axis `which`, creating a Multiply entry when it has
    none (and the window is not the no-op [0, 1]). Only writes on a change, so a
    drag that does not move an axis never marks the plant dirty. Returns whether
    the axis is editable."""
    s = obj.plantgen
    c = active_component(s)
    ax = axis(s, which)
    if c is None or not ax["windowable"]:
        return False
    lo, hi = sorted((round(_clamp01(lo), 3), round(_clamp01(hi), 3)))
    _i, e = find_entry(c, ax)
    if e is None:
        if lo <= 0.0 and hi >= 1.0:
            return True
        e = c.filters.add()
        e.channel = ax["channel"]
        if ax["channel"] == "custom":
            e.attr_name = ax["attr"]
        e.op = "multiply"
        e.color = _MASK_COLORS[(len(c.filters) - 1) % len(_MASK_COLORS)]
    if abs(float(e.lo) - lo) > 1e-6:
        e.lo = lo
        _state["edited"] = True
    if abs(float(e.hi) - hi) > 1e-6:
        e.hi = hi
        _state["edited"] = True
    return True


# -- The points ---------------------------------------------------------------
def _axis_values(st, ax, c, idx):
    """This axis's value per candidate point, and whether the channel exists on
    them. `random` is regenerated with the salt the component's own random entry
    uses (or would use, appended), so the scatter matches what its mask reads."""
    if ax["channel"] == "random":
        i, _e = find_entry(c, ax)
        if i < 0:
            i = len(c.filters) if c is not None else 0
        return masks._hash01(st["ids"][idx], masks._random_salt(st["seed"], i)), True
    v = st["channels"].get(ax["key"])
    if v is None:
        return np.zeros(len(idx)), False
    return np.clip(np.asarray(v, dtype=np.float64)[idx], 0.0, 1.0), True


def view_points(obj):
    """Everything the scatter needs, from the last regrow's stash, or None when
    there is no stash. Counts are over every candidate point; `draw` is the
    (sub)sample actually plotted."""
    st = instancing.LAST_POINTS.get(obj.name)
    if st is None:
        return None
    s = obj.plantgen
    c = active_component(s)
    n = len(st["ids"])
    sel = (instancing._type_point_mask(c.type, st["kind"], st["form"]) if c is not None
           else np.ones(n, dtype=bool))
    idx = np.nonzero(sel)[0]
    ax, ay = axis(s, "x"), axis(s, "y")
    xs, x_ok = _axis_values(st, ax, c, idx)
    ys, y_ok = _axis_values(st, ay, c, idx)
    owner = st["owner"][idx]
    dropped = st["drop"][idx] > 0.5
    ci = s.components_active if c is not None else -2
    (xlo, xhi, _), (ylo, yhi, _) = window(c, ax), window(c, ay)
    inside = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
    draw = np.arange(len(idx))
    if len(idx) > _MAX_POINTS:
        keep = masks._hash01(st["ids"][idx], _SUBSAMPLE_SALT) < _MAX_POINTS / len(idx)
        draw = np.nonzero(keep)[0]
    return {
        "idx": idx, "xs": xs, "ys": ys, "owner": owner, "dropped": dropped,
        "inside": inside, "draw": draw, "x_ok": x_ok, "y_ok": y_ok,
        "n": len(idx), "n_inside": int(inside.sum()),
        "n_owned": int((owner == ci).sum()),
        "n_kept": int(((owner == ci) & ~dropped).sum()),
        "n_other": int(((owner >= 0) & (owner != ci)).sum()),
        "n_unclaimed": int((owner < 0).sum()),
    }


# -- Geometry (area-local coords) ---------------------------------------------
def _sc() -> float:
    return bpy.context.preferences.system.ui_scale


def _layout(obj, area) -> dict:
    sc = _sc()
    W, H = _W * sc, _H * sc
    if _state["win"] is None:
        _state["win"] = (64.0 * sc, 40.0 * sc)
    x, yb = _state["win"]
    x = max(2.0, min(x, area.width - W - 2.0))
    yb = max(2.0, min(yb, area.height - H - 2.0))
    _state["win"] = (x, yb)
    top = yb + H
    s = obj.plantgen
    c = active_component(s)
    ax, ay = axis(s, "x"), axis(s, "y")
    L = {
        "sc": sc, "fs": calendar_overlay._fs(), "x": x, "yb": yb, "W": W, "H": H, "top": top,
        "px0": x + 42 * sc, "px1": x + W - 16 * sc,
        "py0": yb + 70 * sc, "py1": top - (_TITLE_H + 14) * sc,
        "close": (x + W - 26 * sc, top - _TITLE_H * sc, 26 * sc, _TITLE_H * sc),
        "title": (x, top - _TITLE_H * sc, W - 26 * sc, _TITLE_H * sc),
        "ax": ax, "ay": ay, "comp": c,
    }
    L["wx"], L["wy"] = window(c, ax), window(c, ay)
    return L


def _d2x(v, L):
    return L["px0"] + v * (L["px1"] - L["px0"])


def _d2y(v, L):
    return L["py0"] + v * (L["py1"] - L["py0"])


def _x2d(mx, L):
    return _clamp01((mx - L["px0"]) / max(1e-6, L["px1"] - L["px0"]))


def _y2d(my, L):
    return _clamp01((my - L["py0"]) / max(1e-6, L["py1"] - L["py0"]))


def _rect_px(L):
    (xlo, xhi, _), (ylo, yhi, _) = L["wx"], L["wy"]
    return _d2x(xlo, L), _d2x(xhi, L), _d2y(ylo, L), _d2y(yhi, L)


# -- Draw primitives (offset into the current region) -------------------------
# Every primitive re-arms ALPHA blending right before it draws: `blf.draw` leaves
# blending OFF behind it, and a translucent fill drawn after text then writes its
# low alpha straight into the overlay buffer -- the viewport composites that as a
# hole, the scene showing through a fully saturated tint (seen live).
def _uniform():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


def _tris(verts, rgba):
    verts = [(vx - _OFF[0], vy - _OFF[1]) for vx, vy in verts]
    sh = _uniform()
    batch = batch_for_shader(sh, 'TRIS', {"pos": verts})
    sh.bind()
    sh.uniform_float("color", rgba)
    gpu.state.blend_set('ALPHA')
    batch.draw(sh)


def _quad(x, y, w, h, rgba):
    _tris(((x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)), rgba)


def _lines(segs, rgba, width=1.0):
    if not segs:
        return
    pts = []
    for (a, b) in segs:
        pts += [(a[0] - _OFF[0], a[1] - _OFF[1]), (b[0] - _OFF[0], b[1] - _OFF[1])]
    gpu.state.line_width_set(width)
    sh = _uniform()
    batch = batch_for_shader(sh, 'LINES', {"pos": pts})
    sh.bind()
    sh.uniform_float("color", rgba)
    gpu.state.blend_set('ALPHA')
    batch.draw(sh)
    gpu.state.line_width_set(1.0)


def _text(x, y, s, rgba, size, anchor="left", angle=0.0):
    font = 0
    blf.size(font, max(8, int(size)))
    if anchor != "left":
        w = blf.dimensions(font, s)[0]
        if angle:
            y = y - w * 0.5
        else:
            x = x - w if anchor == "right" else x - w * 0.5
    blf.color(font, *rgba)
    if angle:
        blf.enable(font, blf.ROTATION)
        blf.rotation(font, angle)
    blf.position(font, x - _OFF[0], y - _OFF[1], 0)
    blf.draw(font, s)
    if angle:
        blf.disable(font, blf.ROTATION)


def _points(px, py, cols, r):
    """Many small squares in ONE flat-colour batch (numpy-built)."""
    if not len(px):
        return
    x0, x1 = px - r - _OFF[0], px + r - _OFF[0]
    y0, y1 = py - r - _OFF[1], py + r - _OFF[1]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
    verts = np.stack([np.stack(c, axis=-1) for c in corners], axis=1)
    verts = np.ascontiguousarray(verts.reshape(-1, 2), dtype=np.float32)
    colors = np.ascontiguousarray(np.repeat(cols, 6, axis=0), dtype=np.float32)
    sh = gpu.shader.from_builtin('FLAT_COLOR')
    batch = batch_for_shader(sh, 'TRIS', {"pos": verts, "color": colors})
    gpu.state.blend_set('ALPHA')
    batch.draw(sh)


# -- Draw ---------------------------------------------------------------------
_REGIONS = ('WINDOW', 'UI', 'TOOLS', 'HEADER')


def _draw() -> None:
    obj = active_object()
    if obj is None:
        return
    region, area, space = bpy.context.region, bpy.context.area, bpy.context.space_data
    if (region is None or area is None or space is None or space.type != 'VIEW_3D'
            or region.type not in _REGIONS):
        return
    try:
        _draw_view(obj, region, area)
    except ReferenceError:
        pass


def _draw_view(obj, region, area) -> None:
    _OFF[0] = region.x - area.x
    _OFF[1] = region.y - area.y
    p = calendar_overlay._palette()
    L = _layout(obj, area)
    _state["L"] = L
    sc, fs = L["sc"], L["fs"]
    s = obj.plantgen
    c = L["comp"]
    gpu.state.blend_set('ALPHA')

    x, yb, W, H = L["x"], L["yb"], L["W"], L["H"]
    _quad(x - 1.2 * sc, yb - 1.2 * sc, W + 2.4 * sc, H + 2.4 * sc, p["border"])
    _quad(x, yb, W, H, p["panel"])
    title_y = L["top"] - _TITLE_H * sc + (_TITLE_H * sc - fs * sc) * 0.5
    title = "Classify"
    if c is not None:
        title += f" · {c.name or c.type.title()} {s.components_active + 1}"
    _text(x + 10 * sc, title_y, title, p["text"], fs * sc)
    _text(x + W - 10 * sc, title_y, "✕", p["text_mute"], (fs + 2) * sc, "right")

    px0, px1, py0, py1 = L["px0"], L["px1"], L["py0"], L["py1"]
    _quad(px0, py0, px1 - px0, py1 - py0, p["canvas"])
    grid = []
    for t in (0.25, 0.5, 0.75):
        grid.append(((_d2x(t, L), py0), (_d2x(t, L), py1)))
        grid.append(((px0, _d2y(t, L)), (px1, _d2y(t, L))))
    _lines(grid, p["grid_month"], 1.0)
    small = (fs - 2) * sc
    for t, lab in ((0.0, "0"), (0.5, ".5"), (1.0, "1")):
        _text(_d2x(t, L), py0 - 12 * sc, lab, p["text_mute"], small, "center")
        _text(px0 - 5 * sc, _d2y(t, L) - small * 0.35, lab, p["text_mute"], small, "right")

    ax, ay = L["ax"], L["ay"]
    V = view_points(obj) if c is not None else None
    xl = ax["label"] + ("" if V is None or V["x_ok"] else " (not on these points)")
    yl = ay["label"] + ("" if V is None or V["y_ok"] else " (not on these points)")
    _text((px0 + px1) * 0.5, py0 - 26 * sc, xl, p["text"], (fs - 1) * sc, "center")
    _text(x + 14 * sc, (py0 + py1) * 0.5, yl, p["text"], (fs - 1) * sc, "center",
          angle=math.pi * 0.5)

    lines = []
    if c is None:
        lines.append("No components -- add one in the Components panel")
    elif V is None:
        lines.append("Regrow the plant to fill the view")
    else:
        _draw_scatter(V, L, s, sc)
        _draw_rect(L, c, sc)
        if _state["drag"] is not None:
            (xlo, xhi, _), (ylo, yhi, _) = L["wx"], L["wy"]
            lines.append(f"{ax['label']} {xlo:.2f}-{xhi:.2f} · "
                         f"{ay['label']} {ylo:.2f}-{yhi:.2f}")
        else:
            lines.append(f"{V['n']} {c.type} pts · in window {V['n_inside']} · "
                         f"owns {V['n_kept']}")
        notes = []
        if c.mute:
            notes.append("muted: claims nothing")
        elif c.collection is None or not len(c.collection.objects):
            notes.append("no collection: claims nothing")
        if L["wx"][2] or L["wy"][2]:
            notes.append("inverted: keeps outside")
        if not s.live_update:
            notes.append("Live Update off: regrow to recolour")
        if not notes:
            notes.append(f"others {V['n_other']} · unclaimed {V['n_unclaimed']}")
        lines.append(" · ".join(notes))
    for k, line in enumerate(lines[:2]):
        _text(x + 10 * sc, yb + (24 - 14 * k) * sc, line,
              p["text"] if k == 0 else p["text_mute"], (fs - 1.5) * sc)
    gpu.state.blend_set('NONE')


def _draw_scatter(V, L, s, sc) -> None:
    d = V["draw"]
    if not len(d):
        return
    px = _d2x(V["xs"][d], L)
    py = _d2y(V["ys"][d], L)
    # A pixel-level jitter (hashed per point, stable) so the discrete channels
    # (branch depth, binary flags) read as clumps rather than one dot each. It
    # moves the dot, never the value the window tests.
    ids = instancing.LAST_POINTS[active_object().name]["ids"][V["idx"][d]]
    j = 2.5 * sc
    px = px + (masks._hash01(ids, 0xA11CE) * 2.0 - 1.0) * j
    py = py + (masks._hash01(ids, 0xB0B) * 2.0 - 1.0) * j
    owner, dropped = V["owner"][d], V["dropped"][d]
    palette = np.array([(*c.color, 1.0) for c in s.components] or [(0, 0, 0, 1)],
                       dtype=np.float64)
    cols = np.tile(np.array(_GREY), (len(d), 1))
    has = owner >= 0
    cols[has] = palette[np.clip(owner[has], 0, len(palette) - 1)]
    cols[has & dropped, 3] = 0.25
    # Draw order: unclaimed, other components, then the active one on top.
    act = owner == s.components_active
    for sel in (~has, has & ~act, act):
        if sel.any():
            _points(px[sel], py[sel], cols[sel], 2.0 * sc)


def _draw_rect(L, c, sc) -> None:
    rx0, rx1, ry0, ry1 = _rect_px(L)
    col = (*c.color, 1.0)
    _quad(rx0, ry0, rx1 - rx0, ry1 - ry0, (*c.color, 0.06))
    ex, ey = L["ax"]["windowable"], L["ay"]["windowable"]
    live_w, dead_w = 2.0 * sc, 1.0 * sc
    dead = (*c.color, 0.45)
    _lines([((rx0, ry0), (rx0, ry1)), ((rx1, ry0), (rx1, ry1))], col if ex else dead,
           live_w if ex else dead_w)
    _lines([((rx0, ry0), (rx1, ry0)), ((rx0, ry1), (rx1, ry1))], col if ey else dead,
           live_w if ey else dead_w)
    h = 3.0 * sc
    handles = []
    if ex:
        handles += [(rx0, (ry0 + ry1) * 0.5), (rx1, (ry0 + ry1) * 0.5)]
    if ey:
        handles += [((rx0 + rx1) * 0.5, ry0), ((rx0 + rx1) * 0.5, ry1)]
    if ex and ey:
        handles += [(rx0, ry0), (rx1, ry0), (rx0, ry1), (rx1, ry1)]
    for hx, hy in handles:
        _quad(hx - h - 1, hy - h - 1, 2 * h + 2, 2 * h + 2, (0.08, 0.08, 0.08, 1.0))
        _quad(hx - h, hy - h, 2 * h, 2 * h, _HANDLE)


# -- Interaction (area-local coords from the operator) ------------------------
def _in(mx, my, rect):
    return rect[0] <= mx <= rect[0] + rect[2] and rect[1] <= my <= rect[1] + rect[3]


def _over_window(mx, my, L) -> bool:
    return L["x"] <= mx <= L["x"] + L["W"] and L["yb"] <= my <= L["yb"] + L["H"]


def _in_plot(mx, my, L, m=0.0) -> bool:
    return L["px0"] - m <= mx <= L["px1"] + m and L["py0"] - m <= my <= L["py1"] + m


def _edges_at(mx, my, L) -> str:
    """Which draggable rect edges the cursor is on: any of 'l' 'r' 'b' 't'."""
    rx0, rx1, ry0, ry1 = _rect_px(L)
    m = 7.0 * L["sc"]
    z = ""
    if L["ax"]["windowable"] and ry0 - m <= my <= ry1 + m:
        if abs(mx - rx0) <= m:
            z += "l"
        elif abs(mx - rx1) <= m:
            z += "r"
    if L["ay"]["windowable"] and rx0 - m <= mx <= rx1 + m:
        if abs(my - ry0) <= m:
            z += "b"
        elif abs(my - ry1) <= m:
            z += "t"
    return z


def cursor_hint(mx, my):
    L = _state["L"]
    if L is None or _state["drag"] is not None or not _over_window(mx, my, L):
        return None
    if L["comp"] is not None and _in_plot(mx, my, L, 7.0 * L["sc"]):
        z = _edges_at(mx, my, L)
        if len(z) == 2:
            return 'SCROLL_XY'
        if z in ("l", "r"):
            return 'MOVE_X'
        if z in ("b", "t"):
            return 'MOVE_Y'
        rx0, rx1, ry0, ry1 = _rect_px(L)
        if rx0 <= mx <= rx1 and ry0 <= my <= ry1:
            return 'HAND'
        return 'CROSSHAIR'
    return 'DEFAULT'


def press(obj, mx, my) -> str:
    L = _state["L"]
    if L is None or not _over_window(mx, my, L):
        return "pass"
    if _in(mx, my, L["close"]):
        return "close"
    if _in(mx, my, L["title"]):
        _state["drag"] = ("win", mx - L["x"], my - L["yb"])
        return "consume"
    if L["comp"] is None or not _in_plot(mx, my, L, 7.0 * L["sc"]):
        return "consume"
    snap = (L["wx"][0], L["wx"][1], L["wy"][0], L["wy"][1])
    z = _edges_at(mx, my, L)
    if z:
        _state["drag"] = ("edge", z, snap)
        return "consume"
    rx0, rx1, ry0, ry1 = _rect_px(L)
    if rx0 <= mx <= rx1 and ry0 <= my <= ry1:
        _state["drag"] = ("move", _x2d(mx, L), _y2d(my, L), snap)
        return "consume"
    # A new rectangle only takes effect once the cursor has actually moved (see
    # `move`), so a stray click on the plot never collapses the window to a point.
    _state["drag"] = ("new", _x2d(mx, L), _y2d(my, L), mx, my)
    return "consume"


def move(obj, mx, my) -> bool:
    L = _state["L"]
    drag = _state["drag"]
    if drag is None:
        return L is not None and _over_window(mx, my, L)
    if drag[0] == "win":
        _state["win"] = (mx - drag[1], my - drag[2])
        return True
    if L is None:
        return True
    dx, dy = _x2d(mx, L), _y2d(my, L)
    if drag[0] == "edge":
        z, (xlo, xhi, ylo, yhi) = drag[1], drag[2]
        if "l" in z:
            set_window(obj, "x", dx, xhi)
        if "r" in z:
            set_window(obj, "x", xlo, dx)
        if "b" in z:
            set_window(obj, "y", dy, yhi)
        if "t" in z:
            set_window(obj, "y", ylo, dy)
    elif drag[0] == "move":
        x0, y0, (xlo, xhi, ylo, yhi) = drag[1], drag[2], drag[3]
        nx = max(0.0, min(1.0 - (xhi - xlo), xlo + dx - x0))
        ny = max(0.0, min(1.0 - (yhi - ylo), ylo + dy - y0))
        set_window(obj, "x", nx, nx + (xhi - xlo))
        set_window(obj, "y", ny, ny + (yhi - ylo))
    elif drag[0] == "new":
        if math.hypot(mx - drag[3], my - drag[4]) < 4.0 * L["sc"]:
            return True
        set_window(obj, "x", drag[1], dx)
        set_window(obj, "y", drag[2], dy)
    return True


def release() -> bool:
    """End a drag; True when it edited a mask (the caller pushes an undo step)."""
    edited = _state["edited"]
    _state["drag"] = None
    _state["edited"] = False
    return edited


# -- Operator -----------------------------------------------------------------
def _redraw(context) -> None:
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class PLANTGEN_OT_classify_view(Operator):
    """Open (or close) the Classify view for the active plant.

    A 2D scatter of the active component's candidate points on two attributes
    (the X / Y dropdowns), coloured by which component owns each. Drag the
    rectangle -- its edges, corners or body, or drag out a new one -- to set the
    active component's mask window on both attributes; the plant regrows and the
    scatter recolours. The modal owns the mouse over the view and passes
    everything else through; it follows the active plant and ends on Esc, a
    right-click on the view, the close box, or this button again.
    """

    bl_idname = "plantgen.classify_view"
    bl_label = "Classify View"
    bl_description = ("Open a 2D scatter of the active component's points on two "
                      "attributes, and drag its mask window around the ones it should "
                      "take. Click again to close")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def invoke(self, context, event):
        obj = bridge.resolve_plant(context)
        if is_active(obj):
            stop()
            _redraw(context)
            return {"FINISHED"}
        if context.area is None or context.area.type != 'VIEW_3D':
            area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
            if area is None:
                self.report({"WARNING"}, "Needs a 3D viewport")
                return {"CANCELLED"}
        else:
            area = context.area
        self._area = area
        self._session = start(obj)
        if obj.name not in instancing.LAST_POINTS:
            live.mark_dirty(obj)          # fills the view on the next debounce tick
        context.window_manager.modal_handler_add(self)
        _redraw(context)
        return {"RUNNING_MODAL"}

    def _end(self, context):
        stop()
        try:
            context.window.cursor_set('DEFAULT')
        except (AttributeError, RuntimeError):
            pass
        _redraw(context)
        return {"FINISHED"}

    def modal(self, context, event):
        obj = active_object()
        if session() != self._session:
            return {"FINISHED"}
        if obj is None:
            return self._end(context)
        current = bridge.resolve_plant(context)
        if current is not None and current.name != obj.name:
            return self._end(context)
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._end(context)
        try:
            ax, ay = self._area.x, self._area.y
        except (ReferenceError, AttributeError):
            return self._end(context)
        mx, my = event.mouse_x - ax, event.mouse_y - ay
        L = _state["L"]

        if event.type == 'MOUSEMOVE':
            over = move(obj, mx, my)
            if dragging():
                _redraw(context)
                return {"RUNNING_MODAL"}
            hint = cursor_hint(mx, my)
            if hint is not None:
                context.window.cursor_set(hint)
            return {"RUNNING_MODAL"} if over else {"PASS_THROUGH"}

        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            if L is not None and _over_window(mx, my, L):
                return self._end(context)
            return {"PASS_THROUGH"}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            action = press(obj, mx, my)
            if action == "close":
                return self._end(context)
            if action == "pass":
                return {"PASS_THROUGH"}
            _redraw(context)
            return {"RUNNING_MODAL"}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if dragging():
                if release():
                    try:
                        bpy.ops.ed.undo_push(message="Classify window")
                    except RuntimeError:
                        pass
                _redraw(context)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if L is not None and _over_window(mx, my, L) and event.type in (
                'WHEELUPMOUSE', 'WHEELDOWNMOUSE'):
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


# -- Registration -------------------------------------------------------------
# Draw-handler handles are kept under a fixed NAME in `driver_namespace` (which
# survives a module reload), not in a module list: a reload replaces this module,
# and a handle held only in the old module's globals could never be removed.
_HANDLES_KEY = "plantgen_classify_draw_handlers"


def _remove_handlers() -> None:
    ns = bpy.app.driver_namespace
    for rt, h in ns.pop(_HANDLES_KEY, ()) or ():
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, rt)
        except (ValueError, RuntimeError):
            pass


def register() -> None:
    _remove_handlers()
    bpy.utils.register_class(PLANTGEN_OT_classify_view)
    bpy.app.driver_namespace[_HANDLES_KEY] = [
        (rt, bpy.types.SpaceView3D.draw_handler_add(_draw, (), rt, 'POST_PIXEL'))
        for rt in _REGIONS]


def unregister() -> None:
    _remove_handlers()
    stop()
    try:
        bpy.utils.unregister_class(PLANTGEN_OT_classify_view)
    except RuntimeError:
        pass
