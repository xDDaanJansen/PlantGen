"""Live update's regrows reported in Blender's status bar, like a manual
Regrow's: "Regrowing <plant>..." when one is handed to the compute worker,
then "Grew N nodes, M stems & branches" when it lands (user request
2026-10-18). The Plant Generator panel shows the stage under Regrow
(`bridge.regrow_state`).

Live update regrows from a timer, and a report from an operator *called* by a
script only reaches the console -- Blender keeps reports for the status bar
only from operators the window manager runs. So `start` invokes this small
modal operator, whose reports the window manager does show: it reports on its
first tick, watches `bridge.regrow_state` and reports again once every plant
it saw has landed. (Measured 2026-10-18 through the Info log: a timer's plain
operator call left no report there, the modal one's did.)

At most one runs at a time (`_running`); a regrow that starts while it is
watching joins the same report.
"""

from __future__ import annotations

import bpy
from bpy.types import Operator

from . import bridge

_running = False
_POLL_SECONDS = 0.1


def _busy() -> list:
    out = []
    for obj in bpy.data.objects:
        try:
            if bridge.is_plant(obj) and bridge.regrow_state(obj) is not None:
                out.append(obj)
        except (ReferenceError, AttributeError):
            continue
    return out


class PLANTGEN_OT_regrow_report(Operator):
    bl_idname = "plantgen.regrow_report"
    bl_label = "Report Live Regrow"
    bl_description = "Report a live update's regrow in the status bar while it runs"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        global _running
        if _running or context.window is None:
            return {'CANCELLED'}
        _running = True
        self._seen = []
        self._started = False
        wm = context.window_manager
        self._timer = wm.event_timer_add(_POLL_SECONDS, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            busy = _busy()
            for obj in busy:
                if obj.name not in self._seen:
                    self._seen.append(obj.name)
            if not self._started:
                self._started = True
                if busy:
                    self.report({'INFO'}, f"Regrowing {', '.join(self._seen)}...")
            if busy:
                return {'PASS_THROUGH'}
            self._finish(context)
            for name in self._seen:
                obj = bpy.data.objects.get(name)
                if obj is not None and bridge.is_plant(obj):
                    s = obj.plantgen
                    self.report({'INFO'}, f"Grew {s.last_phytomers} nodes, "
                                          f"{s.last_axes} stems & branches")
            return {'FINISHED'}
        except Exception:  # noqa: BLE001 -- a modal that raises is dropped mid-watch
            self._finish(context)
            return {'CANCELLED'}

    def cancel(self, context):
        self._finish(context)

    def _finish(self, context):
        global _running
        _running = False
        timer = getattr(self, "_timer", None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self._timer = None


def start() -> None:
    """Report the live regrow that was just handed to the worker -- from a
    timer, so the operator gets a window of its own to run in."""
    if _running:
        return                       # the watching one picks the new plant up
    try:
        windows = bpy.context.window_manager.windows
        if not windows:
            return
        win = windows[0]
        with bpy.context.temp_override(window=win, screen=win.screen):
            bpy.ops.plantgen.regrow_report('INVOKE_DEFAULT')
    except Exception:  # noqa: BLE001 -- a report must never break live update
        pass


def register() -> None:
    bpy.utils.register_class(PLANTGEN_OT_regrow_report)


def unregister() -> None:
    global _running
    _running = False
    bpy.utils.unregister_class(PLANTGEN_OT_regrow_report)
