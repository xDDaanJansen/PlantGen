"""Live update for FlowerGen flowers: curve widgets (CurveMapping) fire no
property `update=`, so the active flower's curves are polled on a timer and the
flower rebuilt when they change, its curve popover thumbnails repainted with
it. FlowerGen's half of the parametric add-on's old `blender/live.py` watcher."""

import bpy

from . import flower, icons

_INTERVAL = 0.12  # seconds between polls
_last_sig = None


def _active():
    try:
        return bpy.context.view_layer.objects.active
    except Exception:
        return None


def _signature(obj, nodes, keys):
    parts = [obj.as_pointer()]
    for key in keys:
        for p in nodes[key].mapping.curves[0].points:
            parts.append((round(p.location[0], 5), round(p.location[1], 5), p.handle_type))
        parts.append("|")
    return tuple(parts)


def _redraw_view3d():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _watch():
    global _last_sig
    try:
        obj = _active()
        if flower.is_flower(obj):
            nodes = flower.get_flower_curves(obj)
            if nodes:
                sig = _signature(obj, nodes, flower._FLOWER_CURVE_KEYS)
                if sig != _last_sig:
                    _last_sig = sig
                    flower.rebuild_flower(obj)
                    for key in flower._FLOWER_CURVE_KEYS:
                        icons.set_curve_icon(f"flower_{key}", flower.raw_ramp(nodes[key]).points)
                    _redraw_view3d()
    except Exception:
        pass        # a timer that raises is silently unregistered
    return _INTERVAL


def register():
    if not bpy.app.timers.is_registered(_watch):
        bpy.app.timers.register(_watch, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_watch):
        bpy.app.timers.unregister(_watch)
