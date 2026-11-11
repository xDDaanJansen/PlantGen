"""Take the manual's panel screenshots (plant_gen/docs/shots.md) in a separate,
factory-settings Blender WINDOW -- never the user's session or files.

    python tools/panel_shots.py [shot ...]          # every shot, or the named ones

runs, once per shot (each from a clean Blender),

    blender --factory-startup --enable-event-simulate --python tools/panel_shots.py -- <shot>

Inside Blender the script registers plant_gen from the repo and stages the
shot: a plant with stand-in leaf / flower meshes, a selection, the folds it
wants open (every other fold closed -- a fresh Blender makes each fold from its
draw call's `default_closed`, so the panel's draw is wrapped to pass the shot's
choice). Then, for a SIDEBAR shot, only that panel is left in the PlantGen tab,
the tab is clicked (the active tab is read-only to Python, hence
--enable-event-simulate), and the window is screenshot and cropped to the panel.
A VIEWPORT shot (a menu, the calendar) hides the sidebar, the plant and the
overlays on a flat grey and crops to what is drawn over it.
Writes plant_gen/docs/images/<shot>.png.
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "plant_gen" / "docs" / "images"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

try:
    import bpy
    import numpy as np
except ImportError:                              # the driver, outside Blender
    bpy = None

if bpy is not None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))


# --- staging -------------------------------------------------------------------

def _plant(leaves: int = 1):
    """The bare new plant (Add Plant's own is a full preset), with stand-in meshes in its folders
    (`leaves` leaf meshes, named apart) so its components show something."""
    import render_figures as rf

    from plant_gen.blender import bridge
    obj = bridge.create_plant(bpy.context, name="Plant", preset=None)
    s = obj.plantgen
    s.live_update = False
    for k in range(leaves):
        leaf = rf._stand_in_leaf()
        leaf.name = ("Leaf", "Leaf_Young", "Leaf_Shade")[k] if leaves > 1 else "Leaf"
        s.leaf_asset_collection.objects.link(leaf)
    flower = rf._stand_in_flower()
    flower.name = "Flower"
    s.flower_asset_collection.objects.link(flower)
    bridge.regrow(obj)
    return obj


def _select_row(s, kind: str, test) -> None:
    """Select the Components tree row of `kind` whose ref passes `test`."""
    from plant_gen.blender import structure
    structure.rebuild_tree(s)
    for i, row in enumerate(s.tree_rows):
        if row.kind == kind and test(row.ref):
            s.tree_active = i
            return
    raise LookupError(f"no {kind} row")


def _level_ref(s, name_prefix: str):
    return lambda ref: s.branch_levels[ref].name.startswith(name_prefix)


def _component_ref(s, ctype: str):
    return lambda ref: s.components[[c.uid for c in s.components].index(
        s.placements[ref].comp_uid)].type == ctype


def _add_leaf_component(obj):
    """A leaf component on the Plant row drawing from the leaf folder."""
    s = obj.plantgen
    s.structure_focus, s.structure_place = 'PLANT', -1
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.plantgen.component_add(type="leaf", place=True)
    s.components[-1].collection = s.leaf_asset_collection


def stage_plain():
    _plant()


def stage_timelapse():
    _plant().plantgen.animate_day = True


def stage_components_tree():
    obj = _plant()
    _add_leaf_component(obj)
    s = obj.plantgen
    s.structure_component_open = False           # the tree, not the settings
    _select_row(s, "COMPONENT", _component_ref(s, "leaf"))


def stage_components_level():
    s = _plant().plantgen
    s.structure_level_open = True
    _select_row(s, "LEVEL", _level_ref(s, "Branches"))


def stage_components_flower():
    s = _plant().plantgen
    s.components[0].inflorescence_kind = 'RACEME'
    s.structure_component_open = True
    _select_row(s, "COMPONENT", _component_ref(s, "flower"))


def stage_traits():
    obj = _plant(leaves=3)
    _add_leaf_component(obj)
    s = obj.plantgen
    s.structure_component_open = True
    _select_row(s, "COMPONENT", _component_ref(s, "leaf"))
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.plantgen.trait_column_add()      # an Age column, rows synced


def stage_mapping():
    from plant_gen.blender import bridge, mapping
    obj = _plant()
    s = obj.plantgen
    lv = next(lv for lv in s.branch_levels if lv.name.startswith("Branches"))
    mapping.add_map(obj, mapping.LEVEL, lv.uid, "start_length_scale", 'HEIGHT')
    mapping.select(s, mapping.LEVEL, lv.uid, "start_length_scale")
    bridge.regrow(obj)


def stage_welded():
    from plant_gen.blender import bridge
    obj = _plant()
    bridge.regrow(obj, weld=bridge.WELD_NOW)


def stage_rig(wind: bool = False):
    obj = _plant()
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.plantgen.build_rig()
    if wind:
        obj.plantgen.wind_enable = True


def stage_effector():
    from plant_gen.blender import effectors
    obj = _plant()
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.plantgen.add_effector()
    eff = next(o for o in bpy.data.objects if effectors.is_effector(o))
    eff.plantgen_effector.kind = 'FLATTEN'
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    eff.select_set(True)
    bpy.context.view_layer.objects.active = eff


def stage_texturing():
    from plant_gen.blender import stem_material, ui_materials
    s = _plant().plantgen
    s.branch_levels[0].material = stem_material.new_material()
    ui_materials.rebuild_rows(s)
    s.texture_rows_active = 0


def stage_flower_generator():
    bpy.ops.plantgen.add_flower()


def stage_flower(**values):
    """A new FlowerGen flower, selected, with `values` set on its settings."""
    bpy.ops.plantgen.add_flower()
    s = bpy.context.view_layer.objects.active.plantgen_flower
    for name, value in values.items():
        setattr(s, name, value)


def stage_leaf_generator():
    _plant()
    sl = bpy.context.scene.plantgen_leaf
    sl.mode = 'generate'


def stage_leaf(**values):
    """The Leaf Generator in Generate mode, `values` set on its template."""
    sl = bpy.context.scene.plantgen_leaf
    sl.mode = 'generate'
    for name, value in values.items():
        setattr(sl, name, value)


def stage_leaf_cutout():
    sl = bpy.context.scene.plantgen_leaf
    sl.mode = 'cutout'
    sl.image = bpy.data.images.new("Leaf_Atlas", 512, 512, alpha=True)


def stage_leaf_edit():
    """A generated leaf, selected: the panel edits it live."""
    stage_leaf()
    bpy.ops.plantgen.leaf_generate()


def stage_calendar():
    from plant_gen.blender import calendar_overlay
    obj = _plant()
    calendar_overlay.start(obj)


def stage_preset_menu():
    _plant()


def stage_map_menu():
    s = _plant().plantgen
    s.structure_level_open = True
    _select_row(s, "LEVEL", _level_ref(s, "Branches"))


def open_preset_menu(obj):
    bpy.ops.wm.call_menu(name="PLANTGEN_MT_preset_library")


def open_map_menu(obj):
    from plant_gen.blender import mapping
    s = obj.plantgen
    lv = next(lv for lv in s.branch_levels if lv.name.startswith("Branches"))
    bpy.ops.plantgen.map_param('INVOKE_DEFAULT', owner=mapping.LEVEL, owner_uid=lv.uid,
                               param="start_length_scale")


# shot -> (panel class, stage, folds to open, viewport opener or None).
# A sidebar shot leaves its panel alone in the tab; a viewport shot (opener
# given, or "VIEWPORT" as the panel) is cropped from the 3D view instead.
SHOTS = {
    "plant-generator": ("PLANTGEN_PT_plant", stage_plain, set(), None),
    "timelapse-fold": ("PLANTGEN_PT_plant", stage_timelapse, {"PLANTGEN_plant_timelapse"}, None),
    "variations-fold": ("PLANTGEN_PT_plant", stage_plain, {"PLANTGEN_plant_variations"}, None),
    "components-tree": ("PLANTGEN_PT_structure", stage_components_tree, set(), None),
    "components-level": ("PLANTGEN_PT_structure", stage_components_level, set(), None),
    "components-flower": ("PLANTGEN_PT_structure", stage_components_flower, set(), None),
    "traits-sheet": ("PLANTGEN_PT_structure", stage_traits, {"PLANTGEN_component_traits"}, None),
    "mapping-panel": ("PLANTGEN_PT_mapping", stage_mapping, set(), None),
    "stiffness-fold": ("PLANTGEN_PT_stem", stage_plain, {"PLANTGEN_stem_stiffness"}, None),
    "pruning-fold": ("PLANTGEN_PT_stem", stage_plain, {"PLANTGEN_stem_pruning"}, None),
    "surface-fold": ("PLANTGEN_PT_stem", stage_welded, {"PLANTGEN_stem_surface"}, None),
    "rig-fold": ("PLANTGEN_PT_stem", stage_rig, {"PLANTGEN_dev_rig"}, None),
    "wind-fold": ("PLANTGEN_PT_stem", lambda: stage_rig(wind=True), {"PLANTGEN_dev_wind"}, None),
    "effector-panel": ("PLANTGEN_PT_effector", stage_effector, set(), None),
    "texturing-materials": ("PLANTGEN_PT_texturing", stage_texturing,
                            {"PLANTGEN_texturing_materials"}, None),
    "flower-generator": ("PLANTGEN_PT_flower", stage_flower_generator, set(), None),
    "flower-bloom": ("PLANTGEN_PT_flower", stage_flower, {"PLANTGEN_PT_flower_bloom"}, None),
    "flower-petals": ("PLANTGEN_PT_flower", stage_flower,
                      {"PLANTGEN_PT_flower_petals", "PLANTGEN_flower_petal_outline",
                       "PLANTGEN_flower_petal_placement"}, None),
    "flower-petal-layers": ("PLANTGEN_PT_flower", stage_flower,
                            {"PLANTGEN_PT_flower_petals", "PLANTGEN_flower_petal_mesh",
                             "PLANTGEN_PT_flower_petal_layers"}, None),
    "flower-petal-deform": ("PLANTGEN_PT_flower", stage_flower,
                            {"PLANTGEN_PT_flower_petals", "PLANTGEN_PT_flower_petal_shape",
                             "PLANTGEN_PT_flower_petal_shape_curl",
                             "PLANTGEN_PT_flower_petal_shape_rotation"}, None),
    "flower-petal-bud": ("PLANTGEN_PT_flower", stage_flower,
                         {"PLANTGEN_PT_flower_petals", "PLANTGEN_PT_flower_petal_bud"}, None),
    "flower-corolla-tube": ("PLANTGEN_PT_flower",
                            lambda: stage_flower(petal_tube_enabled=True),
                            {"PLANTGEN_PT_flower_petals", "PLANTGEN_PT_flower_petal_tube"},
                            None),
    "flower-sepals": ("PLANTGEN_PT_flower", lambda: stage_flower(sepal_enabled=True),
                      {"PLANTGEN_PT_flower_sepals", "PLANTGEN_flower_sepal_outline",
                       "PLANTGEN_flower_sepal_placement"}, None),
    "flower-disc-receptacle": ("PLANTGEN_PT_flower",
                               lambda: stage_flower(receptacle_enabled=True),
                               {"PLANTGEN_PT_flower_disc",
                                "PLANTGEN_PT_flower_receptacle"}, None),
    "flower-material": ("PLANTGEN_PT_flower", lambda: (stage_flower(),
                                                       bpy.ops.plantgen.flower_material_new()),
                        {"PLANTGEN_PT_flower_material",
                         "plantgen_asset_material_settings"}, None),
    "leaf-generator": ("PLANTGEN_PT_leaf", stage_leaf_generator, set(), None),
    "leaf-profile-margin": ("PLANTGEN_PT_leaf",
                            lambda: stage_leaf(gen_margin='serrate', gen_margin_teeth=14),
                            {"PLANTGEN_leaf_profile", "PLANTGEN_leaf_margin"}, None),
    "leaf-deform": ("PLANTGEN_PT_leaf",
                    lambda: stage_leaf(gen_gravity=0.4, gen_cup=0.2, gen_wave=0.1,
                                       gen_noise=0.1),
                    {"PLANTGEN_leaf_deform"}, None),
    "leaf-compound": ("PLANTGEN_PT_leaf", lambda: stage_leaf(gen_compound='pinnate'),
                      {"PLANTGEN_leaf_compound"}, None),
    "leaf-cutout": ("PLANTGEN_PT_leaf", stage_leaf_cutout, set(), None),
    "leaf-edit": ("PLANTGEN_PT_leaf", stage_leaf_edit, {"PLANTGEN_leaf_material"}, None),
    "calendar": ("VIEWPORT", stage_calendar, set(), None),
    "preset-menu": ("VIEWPORT", stage_preset_menu, set(), open_preset_menu),
    "map-menu": ("VIEWPORT", stage_map_menu, set(), open_map_menu),
}


# --- fold control ----------------------------------------------------------------

class _Layout:
    """A UILayout stand-in that opens the shot's folds and closes every other:
    `panel()` is passed the shot's choice; sub-layouts are wrapped alike."""

    _WRAP = {"row", "column", "box", "split", "column_flow", "grid_flow"}

    def __init__(self, layout, folds):
        object.__setattr__(self, "_l", layout)
        object.__setattr__(self, "_f", folds)

    def panel(self, idname, default_closed=False):
        head, body = self._l.panel(idname, default_closed=idname not in self._f)
        return head, (None if body is None else _Layout(body, self._f))

    def panel_prop(self, *args, **kw):
        head, body = self._l.panel_prop(*args, **kw)
        return head, (None if body is None else _Layout(body, self._f))

    def __getattr__(self, name):
        attr = getattr(self._l, name)
        if name in self._WRAP:
            return lambda *a, **k: _Layout(attr(*a, **k), self._f)
        return attr

    def __setattr__(self, name, value):
        setattr(self._l, name, value)


class _PanelView:
    """The panel as its draw sees it, with the wrapped layout."""

    def __init__(self, panel, folds):
        object.__setattr__(self, "_p", panel)
        object.__setattr__(self, "layout", _Layout(panel.layout, folds))

    def __getattr__(self, name):
        return getattr(self._p, name)


def _control_folds(cls_name: str, folds: set) -> None:
    cls = next(c for c in _every_panel() if c.__name__ == cls_name)
    draw = cls.draw

    def wrapped(self, context):
        draw(_PanelView(self, folds), context)
    cls.draw = wrapped


def _subpanels(target: str) -> list:
    """The registered sub-panels under `target`, any depth, in their module's
    registration order (`_classes`): the order Blender stacks siblings in."""
    panels = [c for c in _every_panel() if getattr(c, "is_registered", False)]
    found, names = [], {target}
    grew = True
    while grew:
        grew = False
        for c in panels:
            if getattr(c, "bl_parent_id", "") in names and c.__name__ not in names:
                names.add(c.__name__)
                found.append(c)
                grew = True

    def order(c):
        classes = getattr(sys.modules.get(c.__module__), "_classes", ())
        return classes.index(c) if c in classes else len(classes)
    return sorted(found, key=order)


def _open_subpanels(target: str, folds: set) -> None:
    """Open the sub-panels named in `folds` and close the rest, and give every
    sub-panel's own folds the shot's choice. A sub-panel's first open state is
    its `DEFAULT_CLOSED` option, read when it is registered, so the sub-panels
    are registered again with the option set per the shot."""
    subs = _subpanels(target)
    for c in reversed(subs):
        bpy.utils.unregister_class(c)
    for c in subs:
        opts = set(getattr(c, "bl_options", set())) - {'DEFAULT_CLOSED'}
        c.bl_options = opts if c.__name__ in folds else opts | {'DEFAULT_CLOSED'}
        bpy.utils.register_class(c)
        _control_folds(c.__name__, folds)


# --- the sidebar -----------------------------------------------------------------

_LOG: list[str] = []


def _log(*a) -> None:
    _LOG.append(" ".join(str(x) for x in a))
    print("[shots]", *a)


def _view3d():
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    return (win, area, next(r for r in area.regions if r.type == 'UI'),
            next(r for r in area.regions if r.type == 'WINDOW'))


def _every_panel():
    """Every Panel class, however deep (panels sharing a mixin sit below it)."""
    seen, todo = set(), [bpy.types.Panel]
    while todo:
        for sub in todo.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                todo.append(sub)
    return seen


def _only_panel(target: str) -> None:
    """Leave `target` (and its sub-panels) the only PlantGen panel: the others
    are unregistered, children first. (A poll added after registration is
    never called, so hiding by poll does not work.)"""
    # Every registered panel: a sub-panel names no tab of its own (it takes
    # its parent's), so it is found through its parent, not its category.
    panels = [c for c in _every_panel() if getattr(c, "is_registered", False)]
    drop = {c.__name__ for c in panels if not getattr(c, "bl_parent_id", "")
            and getattr(c, "bl_category", "") == "PlantGen"} - {target}
    grew = True
    while grew:                                  # sub-panels of dropped panels, any depth
        grew = False
        for c in panels:
            if getattr(c, "bl_parent_id", "") in drop and c.__name__ not in drop:
                drop.add(c.__name__)
                grew = True
    for c in sorted(panels, key=lambda c: bool(getattr(c, "bl_parent_id", "")), reverse=True):
        if c.__name__ in drop:
            bpy.utils.unregister_class(c)


_TABS_PX = 26          # the tab strip at the sidebar's right edge, left out of the test
_GAP_ROWS = 40         # this many empty rows in a row end the panel
_GREY = (0.027, 0.027, 0.027)   # a viewport shot's flat background: the sidebar's grey
_EDGE_PX = 24          # a viewport's edge strips, where its region-toggle arrows sit


def _panel_bottom(crop) -> int:
    """The row to cut the sidebar at, just below the panel: walking down from
    the top, the first run of empty rows (the sidebar's own background, the
    tab strip ignored) long enough to be past the panel, less a margin."""
    body = crop[:, :crop.shape[1] - _TABS_PX, :3]
    bg = body[20, body.shape[1] // 2]            # well below any panel
    empty = np.abs(body - bg).max(axis=(1, 2)) <= 0.01
    run = 0
    for row in range(len(empty) - 1, -1, -1):    # top of the sidebar downward
        run = run + 1 if empty[row] else 0
        if run >= _GAP_ROWS:
            return min(len(empty) - 1, row + _GAP_ROWS - 8)
    return 0


def _drawn_box(crop, margin: int = 10):
    """The part of a flat-grey viewport something is drawn on. The grey is
    read off the image (a screenshot holds display colours, not the linear
    value set), and the region-toggle arrows at the left and right edges are
    left out."""
    bg = np.median(crop[:, _EDGE_PX, :3], axis=0)
    # the viewport's background is dithered: a pixel counts only well clear of
    # it, and a row / column only with a run of such pixels
    used = np.abs(crop[:, :, :3] - bg).max(axis=2) > 0.08
    used[:, :_EDGE_PX] = used[:, -_EDGE_PX:] = False
    rows = np.where(used.sum(axis=1) > 6)[0]
    cols = np.where(used.sum(axis=0) > 6)[0]
    if not len(rows) or not len(cols):
        return crop
    r0, r1 = max(0, rows.min() - margin), min(crop.shape[0], rows.max() + margin)
    c0, c1 = max(0, cols.min() - margin), min(crop.shape[1], cols.max() + margin)
    return crop[r0:r1, c0:c1]


def _flat_viewport(area) -> None:
    """Nothing in the 3D view but its flat grey: no sidebar, toolbar,
    overlays or gizmos, and every object hidden."""
    space = area.spaces[0]
    space.show_region_ui = False
    space.show_region_toolbar = False
    space.overlay.show_overlays = False
    space.show_gizmo = False
    shading = space.shading
    shading.type = 'SOLID'
    shading.background_type = 'VIEWPORT'
    shading.background_color = _GREY
    for o in bpy.context.view_layer.objects:
        o.hide_set(True)


class _Run:
    def __init__(self, shot: str):
        self.shot = shot
        self.target, self.stage, self.folds, self.opener = SHOTS[shot]
        self.viewport = self.target == "VIEWPORT"
        self.tries = 0

    def setup(self):
        try:
            import plant_gen
            plant_gen.register()
            prefs = bpy.context.preferences
            prefs.system.use_region_overlap = False   # a solid sidebar
            prefs.view.show_tooltips = False          # none over a hovered item
            win, area, ui, view = _view3d()
            # a factory start opens the splash screen over the viewport: Esc it
            win.event_simulate(type='MOUSEMOVE', value='NOTHING',
                               x=view.x + view.width // 2, y=view.y + view.height // 2)
            win.event_simulate(type='ESC', value='PRESS')
            win.event_simulate(type='ESC', value='RELEASE')
            with bpy.context.temp_override(window=win, area=area):
                self.stage()
            if self.viewport:
                self.plant = bpy.context.view_layer.objects.active
                _flat_viewport(area)
                bpy.app.timers.register(self.open_menu, first_interval=0.8)
                return None
            _control_folds(self.target, self.folds)
            _only_panel(self.target)
            if any(f.startswith("PLANTGEN_PT_") for f in self.folds):
                _open_subpanels(self.target, self.folds)
            area.spaces[0].show_region_ui = True
        except Exception:                        # noqa: BLE001 -- report, then quit
            traceback.print_exc()
            return self.quit()
        bpy.app.timers.register(self.pick_tab, first_interval=0.8)
        return None

    def open_menu(self):
        try:
            if self.opener is not None:
                win, area, ui, view = _view3d()
                x, y = view.x + view.width // 3, view.y + view.height * 2 // 3
                win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
                with bpy.context.temp_override(window=win, area=area, region=view,
                                               active_object=self.plant, object=self.plant):
                    self.opener(self.plant)
        except Exception:                        # noqa: BLE001
            traceback.print_exc()
            return self.quit()
        bpy.app.timers.register(self.shoot, first_interval=1.0)
        return None

    def pick_tab(self):
        """Click down the tab strip at the sidebar's right edge until PlantGen
        is the active tab."""
        win, area, ui, view = _view3d()
        if ui.active_panel_category == "PlantGen" or self.tries >= 80:
            bpy.app.timers.register(self.shoot, first_interval=0.8)
            return None
        x = ui.x + ui.width - 8
        y = ui.y + ui.height - 10 - 12 * self.tries
        self.tries += 1
        win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
        win.event_simulate(type='LEFTMOUSE', value='PRESS', x=x, y=y)
        win.event_simulate(type='LEFTMOUSE', value='RELEASE', x=x, y=y)
        return 0.15

    def shoot(self):
        try:
            win, area, ui, view = _view3d()
            if not self.viewport:
                if ui.active_panel_category != "PlantGen":
                    raise RuntimeError(f"sidebar stuck on {ui.active_panel_category!r}")
                # park the pointer off the sidebar so no button is hovered
                win.event_simulate(type='MOUSEMOVE', value='NOTHING',
                                   x=area.x + 40, y=area.y + 40)
            with bpy.context.temp_override(window=win, area=area):
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=2)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            full = OUT_DIR / f"_{self.shot}_window.png"
            with bpy.context.temp_override(window=win, area=area):
                bpy.ops.screen.screenshot(filepath=str(full), check_existing=False)
            img = bpy.data.images.load(str(full))
            w, h = img.size
            px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
            bpy.data.images.remove(img)
            full.unlink()
            # (pixels run bottom-up: row 0 is the region's bottom edge)
            if self.viewport:
                crop = _drawn_box(px[view.y:view.y + view.height, view.x:view.x + view.width])
            else:
                crop = px[ui.y:ui.y + ui.height, ui.x:ui.x + ui.width]
                crop = crop[_panel_bottom(crop):]
            out = bpy.data.images.new(self.shot, crop.shape[1], crop.shape[0], alpha=False)
            out.pixels[:] = np.ascontiguousarray(crop).ravel()
            out.filepath_raw = str(OUT_DIR / f"{self.shot}.png")
            out.file_format = 'PNG'
            out.save()
            _log("wrote", out.filepath_raw, f"{crop.shape[1]} x {crop.shape[0]}")
        except Exception:                        # noqa: BLE001
            traceback.print_exc()
        return self.quit()

    def quit(self):
        bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender() and None, first_interval=0.3)
        return None


def _in_blender(argv) -> None:
    if not argv or argv[0] not in SHOTS:
        print("[shots] name one shot:", ", ".join(SHOTS))
        return
    if bpy.app.background:
        print("[shots] needs a window: run without -b")
        return
    run = _Run(argv[0])
    bpy.app.timers.register(run.setup, first_interval=1.0)


def _driver(argv) -> int:
    """Outside Blender: one Blender window per shot, in turn."""
    shots = argv or list(SHOTS)
    failed = []
    for shot in shots:
        out = subprocess.run([str(BLENDER), "--factory-startup", "--enable-event-simulate",
                              "--python", __file__, "--", shot],
                             capture_output=True, text=True, timeout=300)
        lines = [ln for ln in (out.stdout + out.stderr).splitlines()
                 if "[shots]" in ln or "Error" in ln or "Traceback" in ln]
        ok = any("[shots] wrote" in ln for ln in lines)
        print(f"{shot}: {'ok' if ok else 'FAILED'}")
        if not ok:
            failed.append(shot)
            print("  " + "\n  ".join(lines[-12:]))
    return 1 if failed else 0


if __name__ == "__main__":
    if bpy is not None:
        _in_blender(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    else:
        sys.exit(_driver(sys.argv[1:]))
