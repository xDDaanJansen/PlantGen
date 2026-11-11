"""Debounced live-update: a regrow() call costs ~1-3 seconds (measured live:
2.77s for the default 40-iteration/48-res settings), far too slow to run on
every property-changed event -- a raw `update=` callback fires on every
intermediate value during a slider drag, which would hang Blender
repeatedly mid-drag. Instead, an edit just marks the object dirty with a
timestamp; a `bpy.app.timers` poll only regrows an object once it's been
quiet for DEBOUNCE_SECONDS, so a whole drag costs exactly one regrow, fired
shortly after you let go -- not a regrow per tick. It keeps the "explicit
regenerate, not a depsgraph-handler write" shape; this just decides *when*
to call it automatically instead of only on a button press.
"""

from __future__ import annotations

import time

import bpy

from . import bridge, curves, wind

DEBOUNCE_SECONDS = 0.5
POLL_INTERVAL = 0.15
FAST_DEBOUNCE_SECONDS = 0.05
"""The wait when the regrow will come from the caches (`bridge.regrow_is_cheap`,
~0.1 s on a 460-tube plant): short enough that dragging a render-side slider
updates as it moves."""
FAST_POLL_INTERVAL = 0.03

TRACE_LIMIT = 600
_trace: list[str] = []
_trace_on = True
"""A ring buffer of what happened around each undo. Cheap enough to leave on:
one formatted line per event, and events only fire on undo, redo and regrow.
Written out by `plantgen.save_undo_log`."""


def log_event(event: str, obj=None, **extra) -> None:
    if not _trace_on:
        return
    from . import bridge

    fields = [f"{time.time() % 10000:9.3f}", f"{event:<16}"]
    if obj is not None:
        try:
            s = obj.plantgen
            fields.append(f"{obj.name:<24}")
            fields.append(f"verts={len(obj.data.vertices):<6}")
            fields.append(f"h={s.last_height * 1000:8.2f}mm")
            fields.append(f"age={s.age:.3f}")
            fields.append(f"levels={len(s.branch_levels)}")
            fields.append(f"seed={s.seed}")
            stamp = obj.get(bridge._GROWN_KEY)
            live_sig = bridge.growth_signature(obj)
            fields.append("stamp=" + ("none" if stamp is None
                                      else ("MATCH" if stamp == live_sig else "STALE")))
            fields.append(f"stamp8={str(stamp)[:8]}")
            fields.append(f"now8={live_sig[:8]}")
            fields.append(f"dirty={obj.name in _dirty}")
            fields.append(f"liveupd={s.live_update}")
            fields.append(f"inspect={s.inspect_active}")
        except (ReferenceError, AttributeError) as exc:
            fields.append(f"<unreadable: {exc!r}>")
    for key, value in extra.items():
        fields.append(f"{key}={value}")
    _trace.append("  ".join(fields))
    del _trace[:-TRACE_LIMIT]


def trace_lines() -> list[str]:
    return list(_trace)


def clear_trace() -> None:
    _trace.clear()


_dirty: dict[str, float] = {}
_curve_state: tuple | None = None
"""(object pointer, curve signature) for the active object only -- one global
rather than a per-object cache, so that selecting a different plant always
repaints the sparklines instead of leaving the last one's on screen."""


def mark_dirty(obj) -> None:
    _dirty[obj.name] = time.time()


def clear_dirty(name: str) -> None:
    _dirty.pop(name, None)


# -- Timelapse (calendar phase 6): drive Day from the timeline frame -----------

def frame_to_day(frame: float, frame_start: float, frame_end: float,
                 day_start: float, day_end: float) -> float:
    """Map a timeline frame to a Day, the scene's whole frame range sweeping the
    Day range (phase 6). Clamped, so frames outside the range hold at the ends,
    and a single-frame range is just the start Day. Pure, so it is unit-tested."""
    span = frame_end - frame_start
    t = 0.0 if span <= 0 else (frame - frame_start) / span
    t = min(1.0, max(0.0, t))
    return day_start + (day_end - day_start) * t


def drive_timelapse_day(obj, scene) -> None:
    """Set `obj`'s Day from the scene's current frame and regrow it now.

    The regrow is direct (not the debounced timer), because a timelapse renders
    a frame right after this and cannot wait 0.5s -- and `clear_dirty` cancels
    the debounce the Day write would otherwise queue, so the plant is not regrown
    twice. Only regrows when the Day actually changed, so a frame that maps to
    the same whole Day as the last is free."""
    from . import timelapse_cache
    s = obj.plantgen
    if s.rig_view:
        # Leaving Rig View takes the rig, skin and preview helpers in or out of
        # the depsgraph (`hide_viewport`); done from inside frame_change_post,
        # mid-evaluation, it crashed Blender (EXCEPTION_ACCESS_VIOLATION in
        # BKE_object_eval_eval_base_flags, twice). So it waits for a timer --
        # between frames -- which then grows this frame.
        _defer_leave_rig_view(obj.name, scene.name)
        return
    frame = int(scene.frame_current)
    day = frame_to_day(frame, scene.frame_start, scene.frame_end,
                       s.timelapse_start_day, s.timelapse_end_day)
    rendering = bpy.app.is_job_running('RENDER')
    # Timelapse Cache: a frame already grown once is shown from its record --
    # no regrow (playback). Renders always regrow exactly.
    if not rendering and timelapse_cache.show(obj, scene, frame):
        if abs(float(s.day) - day) >= 1e-9:
            s.day = day                          # the panel's Day keeps step
            clear_dirty(obj.name)
        return
    if abs(float(s.day) - day) < 1e-9 and not rendering and not bridge.is_stale(obj):
        # Same Day, and the mesh was grown at it (not merely the Day a recorded
        # frame set): record what is shown.
        timelapse_cache.record(obj, scene, frame)
        return
    s.day = day
    clear_dirty(obj.name)
    # A rendered frame is welded now; viewport playback takes the fast preview
    # (a weld already cached for that Day, else the GN-swept stems), so play
    # runs at regrow speed rather than regrow + weld -- and is recorded, so the
    # next time round it plays from the cache.
    bridge.regrow(obj, weld=bridge.WELD_NOW if rendering else bridge.WELD_LATER)
    if not rendering:
        timelapse_cache.record(obj, scene, frame)


_LEAVING_RIG_VIEW: set = set()


def _defer_leave_rig_view(obj_name: str, scene_name: str) -> None:
    """Leave Rig View for a timeline frame from a timer, not the frame handler
    (`drive_timelapse_day`), then grow the frame the timeline is on. A
    timeline frame is no edit: the rig is not marked out of date."""
    if obj_name in _LEAVING_RIG_VIEW:
        return
    _LEAVING_RIG_VIEW.add(obj_name)

    def run():
        _LEAVING_RIG_VIEW.discard(obj_name)
        try:
            from . import rig
            obj = bpy.data.objects.get(obj_name)
            scene = bpy.data.scenes.get(scene_name)
            if obj is None or scene is None or not bridge.is_plant(obj):
                return None
            rig.leave_rig_view(obj, stale=False)
            if obj.plantgen.animate_day and not obj.plantgen.rig_view:
                drive_timelapse_day(obj, scene)
        except Exception:  # noqa: BLE001 -- a raising timer is silently unregistered
            _report("leave-rig-view-failed", obj_name)
        return None

    bpy.app.timers.register(run, first_interval=0.0)


@bpy.app.handlers.persistent
def _drive_timelapse(scene, _depsgraph=None) -> None:
    """frame_change_post: drive every timelapse-animated plant's Day from the new
    frame and regrow it, before the frame is rendered. No-op (and cheap) when no
    plant has Animate Day on, so it is byte-identical off."""
    for obj in scene.objects:
        s = getattr(obj, "plantgen", None)
        if s is not None and getattr(s, "animate_day", False) and bridge.is_plant(obj):
            drive_timelapse_day(obj, scene)


@bpy.app.handlers.persistent
def _arm_on_load(*_args) -> None:
    """Migrate and baseline-stamp every plant in a freshly opened file.

    Migration: a plant made before the preset system carried its base by a live
    lookup on the retired `species` dropdown. `ensure_migrated` stamps that same
    species as a self-contained embedded preset, without changing how the plant
    looks (see `bridge.ensure_migrated`), so old files open self-contained.

    Stamp: without a baseline the *first* edit to each plant is unprotected --
    the undo step Blender pushes on that edit carries no stamp, so nothing can
    tell that its mesh belongs to the settings from before. (`ensure_migrated`
    re-stamps the plants it migrates, so this only covers the rest.)

    Any weld still building belonged to the last file's plants: it is killed.
    (A plant saved mid-weld simply opens unwelded; Weld finishes it, from the
    disk cache if its worker got that far.)
    """
    bridge.cancel_stem_jobs()
    from . import timelapse_cache
    timelapse_cache.clear_all()              # the last file's records
    for obj in bpy.data.objects:
        try:
            if not bridge.is_plant(obj):
                continue
            if obj.get("plantgen_tl_showing", False):
                timelapse_cache.hide(obj)    # saved mid-playback: show the plant
            bridge.ensure_migrated(obj)
            bridge.ensure_day_migrated(obj)
            if not bridge.is_stamped(obj):
                bridge.stamp(obj)
        except (ReferenceError, AttributeError):
            continue


def _arm_once() -> None:
    _arm_on_load()
    return None          # a timer returning None is not rescheduled


@bpy.app.handlers.persistent
def _resync_after_undo(*_args) -> None:
    """Regrow a plant whose restored mesh was never grown from its restored
    settings.

    Undo restores the mesh and the settings from one snapshot, but that
    snapshot can disagree with itself: Blender pushes an undo step the instant
    a slider is released, while the regrow that answers it is a timer 0.5s
    later and records nothing. So the step holds the new settings beside the
    mesh grown from the old ones. Measured on a scratch plant -- undoing onto
    such a step gave Age 0.4 with the 146-vertex Age-1.0 mesh, where 0.4 grows
    15 vertices -- and that is why a manual Regrow appeared to be needed after
    every Ctrl+Z.

    Healed on the **next tick**, not through the debounce. Marking dirty left
    the plant visibly wrong for at least DEBOUNCE_SECONDS after every undo, and
    because `mark_dirty` restamps the timestamp, holding Ctrl+Z pushed the heal
    out for the entire burst -- which is what "the state in between undoes
    stays broken" was. Still deferred to a timer rather than written here:
    a datablock write inside Blender's undo lands in a scene that is still
    settling, the trap that put the inspector's glyphs 342 mm out of place.
    """
    stale = []
    for name in bridge.recently_grown():
        obj = bpy.data.objects.get(name)
        try:
            if obj is None or not bridge.is_plant(obj):
                continue
            # Logged, but NOT used to decide: read here the signature can still
            # be moving -- measured 2fe7ea85 in this handler against 0aabbb40 a
            # moment later. Every tracked plant is queued and `recover` decides
            # once the scene has settled.
            log_event("undo/redo", obj, stale=bridge.is_stale(obj), note="pre-settle")
            if obj.plantgen.live_update:
                stale.append(name)
        except (ReferenceError, AttributeError) as exc:
            log_event("undo/redo", None, name=name, error=repr(exc))
            continue
    if stale:
        _pending.clear()
        _pending.extend(stale)
        if not bpy.app.timers.is_registered(_heal_now):
            bpy.app.timers.register(_heal_now, first_interval=0.0)


_pending: list[str] = []


def settle_scene() -> None:
    """Force the evaluated scene up to date before anything grows against it.

    Obstacles are voxelised with `Object.ray_cast`, which reads **evaluated**
    geometry -- and that lags an undo even after the object's own matrix has
    settled. Measured: two regrows moments apart, same stamp, gave 207 verts /
    590.63 mm and 205 / 596.14, because the earlier one cast rays at obstacles
    the depsgraph had not caught up with yet.
    """
    try:
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get()
    except (AttributeError, RuntimeError):
        pass


def recover(name: str, settled: bool = False) -> bool:
    """Put one plant back in step with its settings. Returns whether it grew.

    **The single place an undo-time regrow happens.** There were two -- this
    module's staleness heal and the overlay's snapshot rebuild -- on separate
    timers, and they raced: same stamp, different plants, and whichever landed
    last decided what you were left looking at.
    """
    obj = bpy.data.objects.get(name)
    try:
        if obj is None or not bridge.is_plant(obj):
            return False
        if not settled:
            settle_scene()
        if not bridge.is_stale(obj):
            return False
        bridge.regrow(obj)
        # Undo can leave a rebuilt mesh being evaluated as it was before, so
        # tag rather than trust the write to have done it.
        obj.update_tag()
        organs = bpy.data.objects.get(obj.get("plantgen_organs_name") or "")
        if organs is not None:
            organs.update_tag()
        log_event("healed", obj)
        return True
    except (ReferenceError, AttributeError) as exc:
        log_event("heal-failed", None, name=name, error=repr(exc))
        return False


def _heal_now() -> None:
    names, _pending[:] = list(_pending), []
    settle_scene()          # once for the batch, not once per plant
    for name in names:
        recover(name, settled=True)
    for area in getattr(bpy.context.screen, "areas", ()):
        area.tag_redraw()
    return None          # a timer returning None is not rescheduled


def _watch_curves() -> None:
    """Dragging a curve point fires no property callback -- a CurveMapping is
    not a property, so there is no `update=` to hang anything off. The only way
    to notice an edit is to compare the points against what they were last
    tick, which is cheap enough at a handful of points per curve.

    The same check repaints each curve button's sparkline icon: a thumbnail
    only ever has to reflect the active object's curves, so it needs no poll of
    its own. Straight off the parametric add-on's own watcher."""
    global _curve_state
    obj = getattr(bpy.context, "active_object", None)
    if obj is None or not bridge.is_plant(obj):
        _curve_state = None
        return
    signature = curves.signature(obj)
    if not signature:
        return
    current = (obj.as_pointer(), signature)
    previous = _curve_state
    if previous == current:
        return
    _curve_state = current
    curves.repaint_icons(obj)
    # Only an edit to the *same* plant's curves is a reason to regrow -- simply
    # selecting another plant changes the signature too, and must not.
    if previous is not None and previous[0] == current[0]:
        mark_dirty(obj)


def _tick() -> float:
    """One poll. **Never lets an exception out**: a `bpy.app.timers` callback
    that raises is unregistered by Blender on the spot, silently, so a single
    failed regrow used to switch live update off for the rest of the session --
    every later edit (a mute, a slider) then did nothing, reading as whatever
    was touched last being broken. The failure is printed and logged instead,
    and the next edit gets its regrow."""
    try:
        _watch_curves()
    except Exception:  # noqa: BLE001 -- see docstring
        _report("curve-watch-failed")
    now = time.time()
    settled = []
    for name, t in list(_dirty.items()):
        quiet = now - t
        if quiet >= DEBOUNCE_SECONDS:
            settled.append(name)
        elif quiet >= FAST_DEBOUNCE_SECONDS:
            obj = bpy.data.objects.get(name)
            try:
                if obj is not None and bridge.is_plant(obj) and bridge.regrow_is_cheap(obj):
                    settled.append(name)
            except Exception:  # noqa: BLE001 -- see docstring
                pass
    grew = False
    for name in settled:
        del _dirty[name]
        obj = bpy.data.objects.get(name)
        try:
            if obj is not None and bridge.is_plant(obj) and obj.plantgen.live_update:
                log_event("debounce-grow", obj)
                from . import rig
                rig.leave_rig_view(obj)     # an edit: show it live, the rig is stale
                # Never welds (the Weld button's); its growth and leaf collision
                # run in the compute worker, so Blender stays responsive.
                bridge.regrow_live(obj)
                grew = True
                if bridge.regrow_state(obj) is not None:
                    from . import regrow_report
                    regrow_report.start()     # "Regrowing ..." in the status bar
        except Exception:  # noqa: BLE001 -- see docstring
            _report("regrow-failed", name)
    if grew:
        # `context.screen` can be None in a timer (no window under the
        # cursor), so reading its areas can raise; walk the windows instead.
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return FAST_POLL_INTERVAL if _dirty else POLL_INTERVAL


def _report(event: str, name: str = "") -> None:
    import traceback
    text = traceback.format_exc()
    print(f"PlantGen live update: {event} {name}\n{text}")
    log_event(event, None, name=name, error=text.strip().splitlines()[-1])


_HANDLERS = (
    (lambda: bpy.app.handlers.undo_post, _resync_after_undo),
    (lambda: bpy.app.handlers.redo_post, _resync_after_undo),
    (lambda: bpy.app.handlers.load_post, _arm_on_load),
    (lambda: bpy.app.handlers.frame_change_post, _drive_timelapse),
    (lambda: bpy.app.handlers.frame_change_post, wind._drive_wind),
)
_HANDLER_NAMES = frozenset({"_resync_after_undo", "_arm_on_load", "_drive_timelapse",
                            "_drive_wind"})


def _drop_handlers() -> None:
    """Removed by NAME, not identity: a module reload replaces the function
    object, so the one sitting in Blender's list is no longer the one this
    module can reach -- and a handler that cannot be removed keeps firing."""
    for get in (h for h, _ in _HANDLERS):
        handlers = get()
        for fn in list(handlers):
            if fn is not None and getattr(fn, "__name__", "") in _HANDLER_NAMES:
                handlers.remove(fn)


def register():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, persistent=True)
    _drop_handlers()
    for get, fn in _HANDLERS:
        get().append(fn)
    # The file already open when the addon loads never fires `load_post`, so
    # arm it here -- but on a timer, because `bpy.data` is restricted during
    # register() and reading it raises `_RestrictData has no attribute
    # 'objects'`.
    bpy.app.timers.register(_arm_once, first_interval=0.0)


def unregister():
    global _curve_state
    _dirty.clear()
    _curve_state = None
    _drop_handlers()
    bridge.cancel_stem_jobs()
    bridge.stop_compute_worker()
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
