from __future__ import annotations

import contextlib
import dataclasses
import os
import time

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Menu, Operator

from . import (
    bridge,
    calendar_overlay,
    curves,
    effectors,
    live,
    mapping,
    overlay,
    presets,
    rig,
    showcase,
    structure,
)


@contextlib.contextmanager
def _undo_step(message: str):
    """Record ONE undo step for an inspector change, after it has been made.

    `bl_options = {'UNDO'}` alone was not enough: after pinning two components
    in turn, one undo jumped all the way back past both. Pushing here -- once
    the CollectionProperty has actually been written -- puts the boundary
    exactly where the change is.

    **A leading push was tried and removed.** It was meant to stop an undo
    walking back past the user's own last edit, on a measurement that turned
    out to be an artifact of the test: setting a property from a script pushes
    nothing, while from the UI Blender pushes for you, so the boundary it was
    adding already existed. What it did in real use was duplicate the previous
    step -- same settings, differing only in a mesh that `bridge.is_stale` now
    repairs on arrival -- so every inspector action cost two Ctrl+Z presses,
    the second of them visibly doing nothing.
    """
    yield
    bpy.ops.ed.undo_push(message=message)


def _redraw(context) -> None:
    """The overlay is a draw handler, not geometry, so nothing in the
    depsgraph changes when its state does and Blender has no reason to
    repaint on its own."""
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# The addon registers under its top package name; the AddonPreferences
# `bl_idname` has to match that exact string, not this module's `__package__`
# (which is `plant_gen.blender`).
_ADDON = __package__.split(".")[0] if __package__ else "plant_gen"


ARTIST_VIEWS = ('color', 'vcol_r', 'vcol_g', 'vcol_b', 'vcol_a', 'mask_preview')
"""The views shown to everyone: Texturing's colour views (real data on the
exported Color attribute) and a mask's preview -- every other debug view is a
Debug Tool."""


def debug_tools(context=None) -> bool:
    """Whether the Debug Tools panel, the debug views beside their parameters
    and wind's Show Gusts are shown (the add-on preference; off when the add-on
    entry cannot be found)."""
    addon = (context or bpy.context).preferences.addons.get(_ADDON)
    return bool(addon and addon.preferences.debug_tools)


def _debug_tools_changed(self, context):
    """Hiding the debug tools also switches off any that are on: their off
    buttons are hidden with them."""
    if self.debug_tools:
        return
    if overlay.active_object() is not None:
        overlay.stop()
    for obj in bpy.data.objects:
        if not bridge.is_plant(obj):
            continue
        s = obj.plantgen
        if s.debug_view_mode != "off" and s.debug_view_mode not in ARTIST_VIEWS:
            s.debug_view_mode = "off"
        if s.wind_debug != 'OFF':
            s.wind_debug = 'OFF'
        if s.show_center_of_mass:
            s.show_center_of_mass = False


class PLANTGEN_AddonPreferences(AddonPreferences):
    bl_idname = _ADDON

    debug_tools: BoolProperty(
        name="Debug Tools",
        default=False,
        update=_debug_tools_changed,
        description=(
            "Show the Debug Tools panel (debug views, Inspector, Diagnostics, "
            "Parameter Sweep, Documentation), the debug views beside their "
            "parameters and wind's Show Gusts. Off hides them all, and switches "
            "off any that are on"),
    )

    user_presets_path: StringProperty(
        name="User Presets Folder",
        subtype='DIR_PATH',
        default="",
        description=(
            "Where Save Preset writes and the Library reads your own species "
            "presets. Leave blank to use a default folder in your Blender config"),
    )
    leaf_shapes_path: StringProperty(
        name="Leaf Shapes Folder",
        subtype='DIR_PATH',
        default="",
        description=(
            "Folder of your own leaf shapes (.json files written by the Leaf "
            "panel's Save Preset). Each one appears under Custom Shapes in the "
            "leaf Shape list, and Save Preset writes here by default"),
    )

    def draw(self, context):
        self.layout.prop(self, "user_presets_path")
        self.layout.prop(self, "leaf_shapes_path")
        self.layout.prop(self, "debug_tools")


class PLANTGEN_OT_add_plant(Operator):
    bl_idname = "plantgen.add_plant"
    bl_label = "Add Plant"
    bl_description = ("Grow a new plant at the 3D cursor, from the Rudbeckia fulgida "
                      "preset. Select it later to change it")
    bl_options = {"REGISTER", "UNDO"}

    plant_name: StringProperty(
        name="Name",
        default="Plant",
        description=(
            "Names the plant and its collection: <Name>_Plant holds the plant "
            "and its organs helper"),
    )

    def invoke(self, context, event):
        # Prompt for a name before building anything -- the structure is named
        # after it.
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "plant_name")
        layout.label(text="Grows the Rudbeckia fulgida preset into <Name>_Plant.",
                     icon='OUTLINER_COLLECTION')

    def execute(self, context):
        obj = bridge.create_plant(context, name=self.plant_name)
        s = obj.plantgen
        self.report({"INFO"}, f"Grew {s.last_phytomers} nodes into {obj.name}")
        return {"FINISHED"}


class PLANTGEN_OT_regrow(Operator):
    bl_idname = "plantgen.regrow"
    bl_label = "Regrow"
    bl_description = ("Rebuild the active plant with its current settings. The stems "
                      "come back unwelded unless this exact plant was welded before -- "
                      "welding is the Weld button")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        obj = context.active_object
        rig.leave_rig_view(obj)
        bridge.regrow(obj)
        s = obj.plantgen
        self.report({"INFO"}, f"Grew {s.last_phytomers} nodes, {s.last_axes} stems & branches")
        return {"FINISHED"}


class PLANTGEN_OT_weld(Operator):
    """Weld the stems: the one step that takes seconds, so it is its own button.
    Every edit shows the stems unwelded (the node graph sweeping the bare
    skeleton) until this is pressed; it then builds in a worker process, Blender
    usable meanwhile, and lands by itself (`bridge.WELD_BACKGROUND`). A weld
    already built for this exact plant comes back at once."""

    bl_idname = "plantgen.weld"
    bl_label = "Weld"
    bl_description = ("Weld the active plant's stems: every junction joined and filleted "
                      "into one mesh, with UVs. Builds in the background -- keep working "
                      "meanwhile. Changing parameters after welding will require "
                      "rebuilding the welds")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not bridge.is_plant(obj):
            return False
        state = bridge.weld_state(obj)
        if state == 'OFF':
            cls.poll_message_set("Weld Junctions is off (Plant-wide > Surface)")
            return False
        if state == 'WELDING':
            cls.poll_message_set("Welding in the background")
            return False
        if state == 'NONE':
            cls.poll_message_set("No stems to weld")
            return False
        if state == 'WELDED' and obj.plantgen.live_update:
            # (With Live Update off the settings may have moved on since.)
            cls.poll_message_set("Already welded")
            return False
        return True

    def execute(self, context):
        obj = context.active_object
        bridge.regrow(obj, weld=bridge.WELD_BACKGROUND)
        state = bridge.weld_state(obj)
        self.report({"INFO"}, "Welded" if state == 'WELDED' else "Welding in the background...")
        return {"FINISHED"}


class PLANTGEN_OT_bake_timelapse(Operator):
    bl_idname = "plantgen.bake_timelapse"
    bl_label = "Render Timelapse"
    bl_description = ("Render the growth animation across the scene's frame range "
                     "(calendar phase 6). Each frame drives Day from Timelapse Start "
                     "to End and regrows the plant, then renders to the scene's output "
                     "path. Needs a camera and Animate Day on; slow -- a full regrow "
                     "per frame -- which is what an offline timelapse render is")

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        scene = context.scene
        if scene.camera is None:
            self.report({'ERROR'}, "Add a camera before rendering the timelapse")
            return {'CANCELLED'}
        animated = [o for o in scene.objects
                    if bridge.is_plant(o) and o.plantgen.animate_day]
        if not animated:
            self.report({'ERROR'}, "Turn on Animate Day on a plant first")
            return {'CANCELLED'}
        # The live plant, not its rig -- left here, before the render steps a
        # frame: the frame handler must not (it defers that to a timer, which a
        # render job gives no chance to run).
        for o in animated:
            rig.leave_rig_view(o, stale=False)
        self.report(
            {'INFO'},
            f"Rendering frames {scene.frame_start}-{scene.frame_end} to "
            f"{scene.render.filepath} -- a regrow per frame, please wait")
        # Blender's own animation render steps each frame (firing the
        # `live._drive_timelapse` frame handler, which sets Day and regrows every
        # animated plant) then renders it, with a progress bar and Esc to cancel.
        # It uses the scene's output path and format, so an image sequence or a
        # video file both just work.
        return bpy.ops.render.render('INVOKE_DEFAULT', animation=True)


class PLANTGEN_OT_build_rig(Operator):
    bl_idname = "plantgen.build_rig"
    bl_label = "Build Rig"
    bl_description = ("Build a skeletal rig for the plant as it stands: a bone chain down "
                      "every stem and branch (straight runs share a bone -- each level's Bone "
                      "Angle), a bone per leaf, bract, flower, bud and fruit, and the welded "
                      "mesh skinned to it with smooth weights. The plant then shows the rig "
                      "(Rig View). Rebuilding replaces the previous rig")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        obj = context.active_object
        s = obj.plantgen
        t0 = time.perf_counter()
        bridge.regrow(obj, weld=bridge.WELD_NOW)     # the skin is the welded mesh
        try:
            counts = rig.build_rig(obj)
        except rig.RigError as err:
            self.report({'ERROR'}, str(err))
            return {'CANCELLED'}
        obj.pop("plantgen_rig_stale", None)
        if s.rig_view:
            rig.apply_rig_view(obj)
        else:
            s.rig_view = True
        self.report({'INFO'}, f"Rig: {counts['stem']} stem + {counts['organ']} organ bones "
                    f"+ root, {counts['verts']} vertices skinned, in "
                    f"{time.perf_counter() - t0:.1f} s")
        return {'FINISHED'}


class PLANTGEN_OT_cache_timelapse(Operator):
    bl_idname = "plantgen.cache_timelapse"
    bl_label = "Bake Timelapse"
    bl_description = ("Grow every frame of the timeline once and record it, so Grow Over "
                      "Timeline plays back smoothly (frames already recorded are skipped). "
                      "Esc stops; the frames done so far are kept")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not bridge.is_plant(obj) or not obj.plantgen.animate_day:
            cls.poll_message_set("Turn on Grow Over Timeline first")
            return False
        return True

    def invoke(self, context, event):
        scene = context.scene
        self._frames = list(range(int(scene.frame_start), int(scene.frame_end) + 1))
        self._i = 0
        self._return_to = scene.frame_current
        wm = context.window_manager
        wm.progress_begin(0, len(self._frames))
        self._timer = wm.event_timer_add(0.001, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context, cancelled: bool):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        context.workspace.status_text_set(None)
        context.scene.frame_set(self._return_to)
        from . import timelapse_cache
        done, total = timelapse_cache.stats(context.active_object, context.scene)
        self.report({'WARNING' if cancelled else 'INFO'},
                    f"Timelapse: {done} / {total} frames recorded"
                    + (" (stopped)" if cancelled else ""))
        return {'CANCELLED' if cancelled else 'FINISHED'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(context, True)
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}
        if self._i >= len(self._frames):
            return self._finish(context, False)
        frame = self._frames[self._i]
        # The timelapse handler regrows and records each plant growing over the
        # timeline (a frame already recorded is only shown).
        context.scene.frame_set(frame)
        self._i += 1
        context.window_manager.progress_update(self._i)
        context.workspace.status_text_set(
            f"Baking timelapse: frame {frame} ({self._i} / {len(self._frames)}) -- Esc to stop")
        return {'RUNNING_MODAL'}


class PLANTGEN_OT_clear_timelapse(Operator):
    bl_idname = "plantgen.clear_timelapse"
    bl_label = "Clear Timelapse Cache"
    bl_description = "Drop the recorded timelapse frames and show the live plant"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        from . import timelapse_cache
        timelapse_cache.clear(context.active_object)
        return {'FINISHED'}


class PLANTGEN_OT_export_rig(Operator):
    bl_idname = "plantgen.export_rig"
    bl_label = "Export Rig"
    bl_description = ("Export the rig as a skeletal mesh -- the bones and the skinned mesh "
                      "in their bind pose, with weights, UVs, materials and the vertex "
                      "colours -- to FBX or USD (by the file's extension), the plant's root "
                      "at the origin; and beside it <name>_wind.json, the simulation groups "
                      "Unreal's Dynamic Wind imports (Import Dynamic Wind Data). Build Rig "
                      "first")
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.fbx;*.usdc", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not bridge.is_plant(obj) or rig.find_rig(obj) is None:
            cls.poll_message_set("Build the rig first")
            return False
        return True

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.clean_name(context.active_object.name) + ".fbx"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = context.active_object
        out = rig.export_rig(obj, self.filepath)
        if "error" in out:
            self.report({'ERROR'}, f"Export failed: {out['error']} (FBX or USD: "
                        ".fbx / .usdc)")
            return {'CANCELLED'}
        stale = " -- the rig is out of date with the plant" if rig.rig_stale(obj) else ""
        wind = f" + Unreal wind data {os.path.basename(out['wind'])}" if out.get("wind") else \
            " (no wind data: Build Rig again to add it)"
        self.report({'WARNING' if stale else 'INFO'},
                    f"Exported {out['bones']} bones, {out['verts']} vertices as "
                    f"{out['format']} to {out['filepath']}{wind}{stale}")
        return {'FINISHED'}


class PLANTGEN_OT_export_nanite_assembly(Operator):
    bl_idname = "plantgen.export_nanite_assembly"
    bl_label = "Export Nanite Assembly"
    bl_description = ("Export the rig as a skeletal-mesh Nanite Assembly (USD) for Unreal "
                      "5.7+ Nanite Foliage: the stems as a skinned base mesh, every leaf and "
                      "flower that is one of its component's meshes as an instance of it, "
                      "bound to its own bone; flowers on their bloom stage. Beside it "
                      "<name>_wind.json for Dynamic Wind. Build Rig first")
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.usdc", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not bridge.is_plant(obj) or rig.find_rig(obj) is None:
            cls.poll_message_set("Build the rig first")
            return False
        return True

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.clean_name(context.active_object.name) + "_assembly.usdc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from . import assembly
        obj = context.active_object
        out = assembly.export_assembly(obj, self.filepath)
        if "error" in out:
            self.report({'ERROR'}, f"Export failed: {out['error']}")
            return {'CANCELLED'}
        stale = " -- the rig is out of date with the plant" if rig.rig_stale(obj) else ""
        self.report({'WARNING' if stale else 'INFO'},
                    f"Exported {out['instances']} instances of {out['prototypes']} meshes, "
                    f"a {out['base_verts']}-vertex base mesh and {out['bones']} bones to "
                    f"{out['filepath']}{stale}")
        return {'FINISHED'}


def _report_preset(op, done: str) -> None:
    """A preset applied: say so, and name whatever its assets lacked."""
    missing = presets.LAST_MISSING
    if missing:
        op.report({"WARNING"}, f"{done}, but could not find: {', '.join(missing)}")
    else:
        op.report({"INFO"}, done)


class PLANTGEN_OT_apply_preset(Operator):
    """Apply a species preset onto the active plant and regrow. The preset
    becomes the plant's self-contained base (embedded on the object), and every
    panel control and curve is seeded from it. Used by the Library menu, which
    sets `filepath` to a known preset file."""

    bl_idname = "plantgen.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Load this species preset onto the active plant and regrow"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype='FILE_PATH')

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        obj = context.active_object
        try:
            preset = presets.apply_preset_file(obj, self.filepath)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Could not load preset: {exc}")
            return {"CANCELLED"}
        bridge.regrow(obj)
        _report_preset(self, f"Applied preset '{preset.name}'")
        return {"FINISHED"}


class PLANTGEN_OT_reload_preset(Operator):
    """Re-apply the current preset's library version, pulling any edits made to
    the file since. Only offered when the plant remembers a preset."""

    bl_idname = "plantgen.reload_preset"
    bl_label = "Reload from Preset"
    bl_description = ("Re-apply the current preset's library version, pulling any "
                      "changes made to the preset file since it was applied")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bridge.is_plant(obj) and bool(obj.plantgen.preset_id)

    def execute(self, context):
        obj = context.active_object
        preset = presets.reload(obj)
        if preset is None:
            self.report({"WARNING"},
                        f"No preset file found for '{obj.plantgen.preset_id}'")
            return {"CANCELLED"}
        bridge.regrow(obj)
        _report_preset(self, f"Reloaded preset '{preset.name}'")
        return {"FINISHED"}


class PLANTGEN_OT_save_preset(Operator):
    """Save the active plant as a preset: everything on it -- its growth, every
    level, component, mask, map and curve -- with its assets (collections,
    materials, textures) in a .blend beside the .json, so loading it anywhere
    makes a 1-to-1 copy. Defaults into the user presets folder, where it shows
    up in the Library."""

    bl_idname = "plantgen.save_preset"
    bl_label = "Save Preset"
    bl_description = "Save the active plant as a reusable species preset"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    preset_name: StringProperty(
        name="Name", default="",
        description="Display name shown in the Library")
    preset_description: StringProperty(
        name="Description", default="",
        description="A one-line note about this preset (optional)")

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def invoke(self, context, event):
        obj = context.active_object
        if not self.preset_name:
            self.preset_name = obj.name
        folder = presets.user_presets_dir(create=True)
        self.filepath = str(folder / f"{presets.slug(obj.name)}.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        self.layout.prop(self, "preset_name")
        self.layout.prop(self, "preset_description")

    def execute(self, context):
        obj = context.active_object
        try:
            path = presets.save_preset(
                obj, self.filepath, self.preset_name, self.preset_description)
        except OSError as exc:
            self.report({"ERROR"}, f"Could not save preset: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved preset to {path.name}")
        return {"FINISHED"}


class PLANTGEN_OT_load_preset(Operator):
    """Apply a species preset .json from anywhere on disk -- e.g. one someone
    shared with you, or a file outside the Library folders."""

    bl_idname = "plantgen.load_preset"
    bl_label = "Load Preset from File"
    bl_description = "Apply a species preset .json from anywhere on disk"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def invoke(self, context, event):
        self.filepath = str(presets.user_presets_dir(create=True)) + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        obj = context.active_object
        try:
            preset = presets.apply_preset_file(obj, self.filepath)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Could not load preset: {exc}")
            return {"CANCELLED"}
        bridge.regrow(obj)
        _report_preset(self, f"Applied preset '{preset.name}'")
        return {"FINISHED"}


class PLANTGEN_OT_add_effector(Operator):
    """Add an effector Empty and, if a plant is active, wire it up in one go.

    Wiring it up is the point. An effector that is not in the active plant's
    effector collection does nothing at all, and an Empty that visibly does
    nothing is the most confusing possible first impression of the feature --
    so the operator creates the collection if there is not one yet, links the
    Empty into it, and points the plant at it.
    """

    bl_idname = "plantgen.add_effector"
    bl_label = "Add Effector"
    bl_description = ("Add an effector at the 3D cursor that steers the active plant's "
                      "growth, and link it to that plant")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        plant = context.active_object if bridge.is_plant(context.active_object) else None
        collection = None
        if plant is not None:
            s = plant.plantgen
            collection = s.effector_collection
            if collection is None:
                collection = bpy.data.collections.new(f"{plant.name}_Effectors")
                # Into the plant's own collection, not `context.collection` --
                # the active Outliner collection is not necessarily where the
                # plant lives.
                parent = (plant.users_collection[0] if plant.users_collection
                          else context.scene.collection)
                parent.children.link(collection)
                s.effector_collection = collection

        obj = effectors.create_effector(context, collection)
        if plant is not None:
            live.mark_dirty(plant)
            self.report({"INFO"}, f"{obj.name} now steers {plant.name}")
        else:
            self.report({"INFO"}, f"{obj.name} created. Point a plant's Effectors "
                                  "collection at it to use it")
        return {"FINISHED"}


class PLANTGEN_OT_toggle_debug_view(Operator):
    """One button per mode rather than a dropdown, so switching between two
    views is one click each way and the active one is visible without opening
    anything. Clicking the mode that is already on turns it back off, which is
    what the button being lit implies it will do."""

    bl_idname = "plantgen.toggle_debug_view"
    bl_label = "Debug View"
    bl_description = "Colour the plant by this attribute in the viewport. Click again to turn off"
    bl_options = {"REGISTER", "UNDO"}

    mode: StringProperty(default="off")

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        s = context.active_object.plantgen
        # Plain assignment, not property_unset: the update callback is what
        # actually flips the overlay, and it has to run.
        s.debug_view_mode = "off" if s.debug_view_mode == self.mode else self.mode
        # Shade, Mask Preview and Component Colours are computed lazily and
        # only land on the organs when a regrow sees them wanted -- which they
        # now are. (Every other view is a socket push, done by the update.)
        if s.debug_view_mode in {"shade", "mask_preview", "component"}:
            bridge.regrow(context.active_object)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_toggle_skin_debug(Operator):
    """Show one vertex-color channel on the SKINNED mesh, in the viewport.

    The plant's own Debug View taps viewers in the static plant's mesh group,
    which is invisible once the plant is hidden and you are looking at the rig.
    This is the same idea on the skin. One button per channel; clicking the lit
    one turns it off. Resolves the plant from whatever component is selected
    (the skin itself, typically), so it works while inspecting the rig."""

    bl_idname = "plantgen.toggle_skin_debug"
    bl_label = "Skin Vertex Color Debug"
    bl_description = "Colour the skinned mesh by this vertex-color channel. Click again to turn off"
    bl_options = {"REGISTER", "UNDO"}

    mode: StringProperty(default="off")

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        from . import wind
        plant = bridge.resolve_plant(context)
        if plant is None:
            self.report({"ERROR"}, "Select a plant (or its rig / skin)")
            return {"CANCELLED"}
        s = plant.plantgen
        s.wind_skin_debug = "off" if s.wind_skin_debug == self.mode else self.mode
        status = wind.ensure_skin_debug(plant)
        if s.wind_skin_debug != "off":
            if status == "no_skin":
                self.report({"WARNING"}, "No baked rig/skin to show vertex colors on")
            elif status == "no_editor_open":
                self.report({"INFO"}, "Skin colors set -- they will tint the mesh once a "
                            "Geometry Nodes editor on the skin is open once to activate")
            elif status != "activated":
                self.report({"WARNING"}, f"Could not show skin debug: {status}")
        return {"FINISHED"}


class PLANTGEN_OT_toggle_inspect(Operator):
    """Turn the inspector overlay on or off.

    Modal, and it has to be: a draw handler cannot ask where the mouse is, so
    hovering to read a component means an operator sitting in the event stream.
    It passes every event through except its own pin key, so navigation,
    selection and every other tool keep working underneath it -- the overlay is
    something you look through, not a mode you are trapped in.

    It ends when the overlay is switched off, when the plant stops being the
    active object, or on Escape.
    """

    bl_idname = "plantgen.toggle_inspect"
    bl_label = "Inspect"
    bl_description = ("Draw a labelled glyph for every node, bud, apex, leaf and flower, "
                      "and read one by pointing at it. P pins a card, X prunes the limb. "
                      "Click again to turn off")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def invoke(self, context, event):
        obj = context.active_object
        if overlay.is_active(obj):
            with _undo_step("Stop inspecting"):
                overlay.stop()
            _redraw(context)
            return {"FINISHED"}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({"WARNING"}, "Run this from a 3D viewport")
            return {"CANCELLED"}

        snapshot = overlay.ensure_snapshot(obj)
        with _undo_step("Inspect"):
            self._session = overlay.start(obj)
        context.window_manager.modal_handler_add(self)
        _redraw(context)
        self.report({"INFO"},
                    f"Inspecting {len(snapshot)} components of {obj.name}. "
                    "Point at one to read it, P to pin, X to prune")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        obj = overlay.active_object()
        # Ending conditions checked first, so a stale modal can never outlive
        # the thing it was drawing. The session token is the one that matters:
        # without it, a second Inspect leaves the first modal running happily,
        # since everything else it can see still looks correct.
        if overlay.session() != self._session:
            return {"FINISHED"}
        if obj is None or obj is not context.active_object:
            overlay.stop()
            _redraw(context)
            return {"FINISHED"}
        if event.type in {'ESC'} and event.value == 'PRESS':
            overlay.stop()
            _redraw(context)
            return {"FINISHED"}

        if event.type == 'MOUSEMOVE':
            region = context.region
            if region is not None and region.type == 'WINDOW':
                hit = overlay.pick(event.mouse_region_x, event.mouse_region_y,
                                   region.as_pointer())
                if overlay.set_hover(hit):
                    _redraw(context)
            return {"PASS_THROUGH"}

        # The one event that is consumed. A modifier-free letter cannot collide
        # with navigation, and taking the mouse button instead would break
        # ordinary selection for as long as the overlay is up.
        if event.type == 'P' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            if overlay.hovered() < 0:
                return {"PASS_THROUGH"}
            # Through the operator, not by touching state: a modal that returns
            # RUNNING_MODAL pushes no undo step, so pinning inline would be the
            # one inspector action Ctrl+Z could not reach.
            bpy.ops.plantgen.toggle_pin()
            return {"RUNNING_MODAL"}

        # X prunes the limb under the cursor -- same operator-not-state reasoning
        # as P, so Ctrl+Z reaches it. Modifier-free so it cannot collide with a
        # navigation gesture, and it regrows, so the pruned branch vanishes at
        # once rather than on the next edit.
        if event.type == 'X' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            if overlay.hovered() < 0:
                return {"PASS_THROUGH"}
            bpy.ops.plantgen.prune()
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


class PLANTGEN_OT_calendar_color(Operator):
    """Recolour a phenophase from the calendar legend -- opens the colour picker
    for that phase's palette colour, a global addon preference (so retheming the
    calendar once applies to every plant)."""

    bl_idname = "plantgen.calendar_color"
    bl_label = "Phase Colour"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(calendar_overlay.color_phase())

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=220)

    def draw(self, context):
        phase = calendar_overlay.color_phase()
        obj = calendar_overlay.active_object()
        col = self.layout.column()
        col.label(text=f"{phase.title()} colour")
        if obj is not None and phase:
            col.prop(obj.plantgen, "cal_col_" + phase, text="")
        col.operator("plantgen.calendar_color_reset", text="Reset to Default", icon="LOOP_BACK")

    def execute(self, context):
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_calendar_color_reset(Operator):
    """Reset this plant's phenophase colour to its built-in default."""

    bl_idname = "plantgen.calendar_color_reset"
    bl_label = "Reset to Default"
    bl_options = {"REGISTER"}

    def execute(self, context):
        phase = calendar_overlay.color_phase()
        obj = calendar_overlay.active_object()
        if obj is not None and phase:
            obj.plantgen.property_unset("cal_col_" + phase)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_calendar_year(Operator):
    """Type the calendar year to view, invoked from the Phenology Calendar's
    year field (a GPU overlay has no native text entry, so the field opens this
    small native popup to type into)."""

    bl_idname = "plantgen.calendar_year"
    bl_label = "Set Calendar Year"
    bl_description = "Type the year to view (0 = the plant's first year)"
    bl_options = {"REGISTER"}

    year: IntProperty(name="Year", min=0, default=0)

    def invoke(self, context, event):
        self.year = calendar_overlay.current_year()
        return context.window_manager.invoke_props_dialog(self, width=150)

    def draw(self, context):
        self.layout.prop(self, "year")

    def execute(self, context):
        calendar_overlay.set_year(self.year)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_toggle_calendar(Operator):
    """Open (or close) the Phenology Calendar for the active plant.

    A GPU-drawn floating window (theme-styled, so it reads as native) that
    authors the plant's within-year phenology by dragging and scrubs its Day.
    The modal owns the mouse while the window is open -- consuming events over
    the window, passing everything else through so viewport navigation and
    selection still work -- and a session token keeps at most one live (see
    `calendar_overlay`). Ends when a *different* plant becomes active (it follows
    the active plant), never on a stray deselect, so a click in empty space does
    not close a calendar you are working in.
    """

    bl_idname = "plantgen.toggle_calendar"
    bl_label = "Phenology Calendar"
    bl_description = ("Open a calendar to author the plant's bloom / fruit / autumn "
                     "timing by dragging, and scrub the Day. Click again to close")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def invoke(self, context, event):
        obj = context.active_object
        if calendar_overlay.is_active(obj):
            calendar_overlay.stop()
            _redraw(context)
            return {"FINISHED"}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({"WARNING"}, "Run this from a 3D viewport")
            return {"CANCELLED"}
        self._area = context.area
        self._session = calendar_overlay.start(obj)
        context.window_manager.modal_handler_add(self)
        _redraw(context)
        self.report({"INFO"}, f"Phenology Calendar open for {obj.name}. "
                    "Drag the playhead to scrub Day, Esc or ✕ to close")
        return {"RUNNING_MODAL"}

    def _end(self, context):
        calendar_overlay.stop()
        try:
            context.window.cursor_set('DEFAULT')
        except (AttributeError, RuntimeError):
            pass
        _redraw(context)
        return {"FINISHED"}

    def modal(self, context, event):
        obj = calendar_overlay.active_object()
        # Ending conditions first, so a stale modal never outlives its window.
        if calendar_overlay.session() != self._session:
            return {"FINISHED"}
        if obj is None:
            return self._end(context)
        active = context.active_object
        if bridge.is_plant(active) and active is not obj:
            return self._end(context)
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._end(context)

        # Area-local coordinates (event.mouse_* is window-absolute), so the
        # window can be hit-tested even with the cursor over the N-panel.
        try:
            ax, ay = self._area.x, self._area.y
        except (ReferenceError, AttributeError):
            return self._end(context)
        mx, my = event.mouse_x - ax, event.mouse_y - ay
        calendar_overlay.set_shift(event.shift)

        if event.type == 'MOUSEMOVE':
            over = calendar_overlay.move(obj, mx, my)
            if calendar_overlay.dragging():
                _redraw(context)
                return {"RUNNING_MODAL"}
            hint = calendar_overlay.cursor_hint(mx, my)
            if hint is not None:
                context.window.cursor_set(hint)
            return {"RUNNING_MODAL"} if over else {"PASS_THROUGH"}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            action = calendar_overlay.press(obj, mx, my, event.shift)
            if action == "close":
                return self._end(context)
            if action == "year":
                try:
                    bpy.ops.plantgen.calendar_year('INVOKE_DEFAULT')
                except RuntimeError:
                    pass
                _redraw(context)
                return {"RUNNING_MODAL"}
            if action == "color":
                try:
                    bpy.ops.plantgen.calendar_color('INVOKE_DEFAULT')
                except RuntimeError:
                    pass
                _redraw(context)
                return {"RUNNING_MODAL"}
            if action == "pass":
                return {"PASS_THROUGH"}
            _redraw(context)
            return {"RUNNING_MODAL"}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if calendar_overlay.dragging():
                calendar_overlay.release()
                _redraw(context)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        return {"PASS_THROUGH"}


class PLANTGEN_OT_toggle_pin(Operator):
    """Pin or unpin whatever the cursor is over.

    An operator rather than a line in the modal, and that is what makes Ctrl+Z
    reach it: a modal returning RUNNING_MODAL pushes no undo step, so the P key
    calls this instead of touching state directly. `UNDO` does the rest, since
    the pins themselves live on the object.
    """

    bl_idname = "plantgen.toggle_pin"
    bl_label = "Pin Component"
    bl_description = "Pin the component under the cursor, or unpin it if it is already pinned"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return overlay.active_object() is not None

    def execute(self, context):
        obj = overlay.active_object()
        if obj is None:
            return {"CANCELLED"}
        with _undo_step("Pin component"):
            what = overlay.toggle_pin(obj)
        if not what:
            return {"CANCELLED"}
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_unpin(Operator):
    """Close one pinned inspector card from the sidebar.

    The viewport already unpins with P, but only what is under the cursor --
    and a card you want rid of is usually one you have since moved away from.
    """

    bl_idname = "plantgen.unpin"
    bl_label = "Unpin"
    bl_description = "Close this pinned card"
    bl_options = {"REGISTER"}

    index: IntProperty(options={"HIDDEN"})

    def execute(self, context):
        obj = overlay.active_object() or context.active_object
        if obj is None or not bridge.is_plant(obj):
            return {"CANCELLED"}
        removed = False
        with _undo_step("Unpin component"):
            removed = overlay.unpin(obj, self.index)
        if not removed:
            return {"CANCELLED"}
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_prune(Operator):
    """Prune the limb under the cursor -- WS3 Phase 3 manual removal.

    Cursor-based, so it lives with the inspector: the X key in the overlay calls
    it, the same way P calls the pin operator, and for the same reason -- a modal
    returning RUNNING_MODAL pushes no undo step, so pruning inline would be the
    one prune Ctrl+Z could not reach. The prune set lives on the object, so
    `UNDO` plus the push here undoes both the set change and the regrown mesh.
    """

    bl_idname = "plantgen.prune"
    bl_label = "Prune Limb"
    bl_description = "Remove the branch under the cursor and everything it carries"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return overlay.active_object() is not None

    def execute(self, context):
        obj = overlay.active_object()
        if obj is None:
            return {"CANCELLED"}
        axis_id = -1
        with _undo_step("Prune limb"):
            axis_id = overlay.prune_hovered(obj)
            if axis_id >= 0:
                bridge.regrow(obj)
        if axis_id < 0:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Pruned axis {axis_id} and its subtree")
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_clear_pruning(Operator):
    """Restore every manually pruned limb.

    A panel button rather than a cursor action: pruning is per-limb and aimed,
    but un-pruning is nearly always all-or-nothing, and there is no click target
    for a limb that is no longer drawn. Also the way to dismiss the 'pruning
    cleared' warning after a topology change wiped the set.
    """

    bl_idname = "plantgen.clear_pruning"
    bl_label = "Clear Pruning"
    bl_description = "Restore every pruned limb on this plant"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = overlay.active_object() or context.active_object
        if obj is None or not bridge.is_plant(obj):
            return False
        s = obj.plantgen
        return bool(len(s.inspect_pruned) or s.prune_stale)

    def execute(self, context):
        obj = overlay.active_object() or context.active_object
        if obj is None or not bridge.is_plant(obj):
            return {"CANCELLED"}
        with _undo_step("Clear pruning"):
            n = overlay.clear_pruned(obj)
            bridge.regrow(obj)
        self.report({"INFO"}, f"Restored {n} pruned limbs")
        _redraw(context)
        return {"FINISHED"}


# Distinct label colours handed to new maps/entries in turn, so a fresh list reads
# as separable at a glance; the user recolours any of them.
_MASK_COLORS = (
    (0.85, 0.35, 0.30), (0.30, 0.55, 0.85), (0.35, 0.75, 0.40),
    (0.85, 0.70, 0.25), (0.65, 0.40, 0.80), (0.30, 0.72, 0.72),
)


def _active_component(context):
    """The plant and its active component, or (obj, None) / (None, None). The
    component owns the filter list the filter operators edit."""
    obj = bridge.resolve_plant(context)
    if obj is None:
        return None, None
    s = obj.plantgen
    if 0 <= s.components_active < len(s.components):
        return obj, s.components[s.components_active]
    return obj, None


class PLANTGEN_OT_component_add(Operator):
    """Add a component of the given type to the library, and select it. With
    `place`, also place it on the Plant Structure's selection."""

    bl_idname = "plantgen.component_add"
    bl_label = "Add Component"
    bl_description = "Add a component (a collection to instance onto the plant)"
    bl_options = {"REGISTER"}

    type: StringProperty(default="leaf", options={'HIDDEN'})
    place: BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj = bridge.resolve_plant(context)
        if obj is None:
            return {"CANCELLED"}
        s = obj.plantgen
        with _undo_step("Add component"):
            colour = mapping.next_component_colour(s)
            c = s.components.add()
            if self.type:
                c.type = self.type
            c.color = colour
            c.uid = max((x.uid for x in s.components), default=0) + 1
            s.components_active = len(s.components) - 1
            if self.place:
                s.structure_place = structure.place(s, structure.structure_target(s), c.uid)
                s.structure_focus = 'COMPONENT'
            structure.rebuild_tree(s)
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


def _component_selected(s) -> bool:
    """Is a component the Components tree's selection (placed or not)?"""
    return (s.structure_focus in ('COMPONENT', 'LIBRARY')
            and 0 <= s.components_active < len(s.components))


def _delete_component(obj, s, active) -> None:
    """Delete component `active` with its curves and every placement of it."""
    uid = s.components[active].uid
    curves.remove_component_curve(obj, uid)
    s.components.remove(active)
    for k in reversed(range(len(s.placements))):
        if s.placements[k].comp_uid == uid:
            s.placements.remove(k)
    s.components_active = min(active, len(s.components) - 1)
    s.structure_focus, s.structure_place = 'PLANT', -1


class PLANTGEN_OT_component_remove(Operator):
    """Delete the selected component, from every level it is on."""

    bl_idname = "plantgen.component_remove"
    bl_label = "Delete Component"
    bl_description = "Delete the selected component, from every level it is placed on"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = bridge.resolve_plant(context)
        return obj is not None and _component_selected(obj.plantgen)

    def execute(self, context):
        obj = bridge.resolve_plant(context)
        if obj is None:
            return {"CANCELLED"}
        s = obj.plantgen
        active = s.components_active
        if not (0 <= active < len(s.components)):
            return {"CANCELLED"}
        with _undo_step("Delete component"):
            _delete_component(obj, s, active)
            structure.rebuild_tree(s)
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


def _copy_rna(src, dst) -> None:
    """Copy every stored RNA property of a property group (collections item by
    item), for a component copy."""
    for prop in src.bl_rna.properties:
        name = prop.identifier
        if name in ("rna_type", "uid", "label") or prop.is_readonly and prop.type != 'COLLECTION':
            continue
        if prop.type == 'COLLECTION':
            items = getattr(dst, name)
            items.clear()
            for item in getattr(src, name):
                _copy_rna(item, items.add())
        elif prop.type == 'POINTER':
            if not prop.is_readonly:
                setattr(dst, name, getattr(src, name))
        else:
            try:
                setattr(dst, name, getattr(src, name))
            except (AttributeError, TypeError, ValueError):
                pass


class PLANTGEN_OT_component_copy(Operator):
    """Duplicate the selected component: a new, independent component with the
    same settings, masks, maps and curves, placed on the same level as the
    selected row (an unplaced one's copy stays unplaced)."""

    bl_idname = "plantgen.component_copy"
    bl_label = "Duplicate Component"
    bl_description = ("Duplicate the selected component: an independent one with the same "
                      "settings, on the same level")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = bridge.resolve_plant(context)
        return obj is not None and _component_selected(obj.plantgen)

    def execute(self, context):
        obj, src = _active_component(context)
        if src is None:
            return {"CANCELLED"}
        s = obj.plantgen
        i = s.components_active
        level = (s.placements[s.structure_place].level_uid
                 if s.structure_focus == 'COMPONENT'
                 and 0 <= s.structure_place < len(s.placements) else None)
        with _undo_step("Duplicate component"):
            colour = mapping.next_component_colour(s)
            c = s.components.add()
            src = s.components[i]
            _copy_rna(src, c)
            c.color = colour                     # its own swatch, not its source's
            c.uid = max((x.uid for x in s.components), default=0) + 1
            c.name_override = src.label + " copy"
            curves.copy_component_curves(obj, src.uid, c.uid)
            uid = c.uid                          # (the move invalidates `c`)
            s.components.move(len(s.components) - 1, i + 1)
            s.components_active = i + 1
            if level is None:
                s.structure_focus = 'LIBRARY'
            else:
                # right after its source among that level's components
                k = structure.place(s, level, uid)
                s.placements.move(k, s.structure_place + 1)
                s.structure_place += 1
            structure.rebuild_tree(s)
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


# --- the Plant Structure tree -------------------------------------------------------------

def _structure(context):
    obj = bridge.resolve_plant(context)
    return obj, (obj.plantgen if obj is not None else None)


def _select_level(s, uid) -> None:
    s.structure_focus = 'LEVEL'
    s.branch_levels_active = max(structure.level_index_of(s, uid), 0)


def _after_tree_edit(context, obj, s, regrow=True) -> None:
    structure.rebuild_tree(s)
    if regrow:
        bridge.regrow(obj)
    _redraw(context)


class PLANTGEN_OT_tree_add_level(Operator):
    """Add a stems level on the Plant, or a level of branches growing on the
    Plant Structure's selected level."""

    bl_idname = "plantgen.tree_add_level"
    bl_label = "Branch Level"
    bl_description = "Add a level of branches growing on the selected level"
    bl_options = {"REGISTER"}

    stems: BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    @classmethod
    def description(cls, context, properties):
        if properties.stems:
            return "Add a stems level: another set of stems growing from the plant's base"
        return cls.bl_description

    def execute(self, context):
        from ..growth.morphology import default_levels
        obj, s = _structure(context)
        parent_uid = 0 if self.stems else structure.structure_target(s)
        if parent_uid == 0 and not self.stems:
            self.report({'INFO'}, "Branches grow on a level: select a stems or branch level")
            return {"CANCELLED"}
        with _undo_step("Add stems level" if self.stems else "Add branching level"):
            levels = bridge._levels_from_object(s)
            parent = structure.level_index_of(s, parent_uid)
            siblings = structure.child_levels_of(s, parent_uid)
            # like its last sibling; the first one on a level like a default branch
            template = (levels[siblings[-1]] if siblings
                        else default_levels()[0 if parent < 0 else 1])
            levels.append(dataclasses.replace(template, name="", parent=parent,
                                              masks=list(template.masks),
                                              maps=list(template.maps)))
            bridge._levels_to_object(s, levels)
            if parent >= 0:
                s.branch_levels[parent].expanded = True
            _select_level(s, s.branch_levels[-1].uid)
            _after_tree_edit(context, obj, s)
        return {"FINISHED"}


def _remove_levels(obj, s, gone: set) -> None:
    """Remove levels (by uid) with their map curves and placements."""
    for i in reversed(range(len(s.branch_levels))):
        lv = s.branch_levels[i]
        if lv.uid in gone:
            for m in lv.maps:
                curves.remove_named_component_curve(
                    obj, curves.GROWTH_MAP_OWNER, curves.map_curve_name(bridge.LEVEL_MAP_KIND,
                                                                       m.uid))
            s.branch_levels.remove(i)
    for k in reversed(range(len(s.placements))):
        if s.placements[k].level_uid in gone:
            s.placements.remove(k)


class PLANTGEN_OT_tree_remove(Operator):
    """Remove the Components tree's selection: a component from that level only
    (it stays, under Not Placed, if that was its last level), an unplaced
    component altogether, or a level with every level growing on it."""

    bl_idname = "plantgen.tree_remove"
    bl_label = "Remove"
    bl_description = ("Remove the selected component from this level (it is kept under "
                      "Not Placed), or the selected level with the levels on it")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = bridge.resolve_plant(context)
        return obj is not None and obj.plantgen.structure_focus in (
            'LEVEL', 'COMPONENT', 'LIBRARY')

    @classmethod
    def description(cls, context, properties):
        obj = bridge.resolve_plant(context)
        if obj is not None and obj.plantgen.structure_focus == 'LIBRARY':
            return "Delete the selected component (it is placed on no level)"
        return cls.bl_description

    def execute(self, context):
        obj, s = _structure(context)
        with _undo_step("Remove from structure"):
            if s.structure_focus == 'LIBRARY':
                if not 0 <= s.components_active < len(s.components):
                    return {"CANCELLED"}
                _delete_component(obj, s, s.components_active)
            elif s.structure_focus == 'COMPONENT':
                if not 0 <= s.structure_place < len(s.placements):
                    return {"CANCELLED"}
                s.placements.remove(s.structure_place)
            else:
                i = s.branch_levels_active
                if not 0 <= i < len(s.branch_levels):
                    return {"CANCELLED"}
                lv = s.branch_levels[i]
                if lv.parent_uid == 0 and len(structure.child_levels_of(s, 0)) == 1:
                    self.report({'INFO'}, "The plant keeps one stems level")
                    return {"CANCELLED"}
                _remove_levels(obj, s, structure.level_subtree(s, lv.uid))
            s.structure_focus, s.structure_place = 'PLANT', -1
            _after_tree_edit(context, obj, s)
        return {"FINISHED"}


class PLANTGEN_OT_tree_move(Operator):
    """Move the Plant Structure's selection up or down among its siblings: a
    component among the components on its level, a level among the levels on
    its parent (its subtree comes along)."""

    bl_idname = "plantgen.tree_move"
    bl_label = "Move"
    bl_description = "Move the selection up or down among its siblings"
    bl_options = {"REGISTER"}

    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")],
                            options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, s = _structure(context)
        step = -1 if self.direction == "UP" else 1
        with _undo_step("Move in structure"):
            if s.structure_focus == 'COMPONENT' and 0 <= s.structure_place < len(s.placements):
                k = s.structure_place
                lv = s.placements[k].level_uid
                sib = [i for i, q in enumerate(s.placements) if q.level_uid == lv]
                at = sib.index(k) + step
                if not 0 <= at < len(sib):
                    return {"CANCELLED"}
                s.placements.move(k, sib[at])
                s.structure_place = sib[at]
                _after_tree_edit(context, obj, s, regrow=False)
                return {"FINISHED"}
            if s.structure_focus != 'LEVEL':
                return {"CANCELLED"}
            i = s.branch_levels_active
            uid = s.branch_levels[i].uid
            sib = structure.child_levels_of(s, s.branch_levels[i].parent_uid)
            at = sib.index(i) + step
            if not 0 <= at < len(sib):
                return {"CANCELLED"}
            s.branch_levels.move(i, sib[at])
            _select_level(s, uid)
            _after_tree_edit(context, obj, s)
        return {"FINISHED"}


def _move_placement(s, k, level_uid) -> bool:
    """Move placement `k` onto `level_uid` (last among the components there)."""
    q = s.placements[k]
    if q.level_uid == level_uid or any(
            x.level_uid == level_uid and x.comp_uid == q.comp_uid for x in s.placements):
        return False
    q.level_uid = level_uid
    s.placements.move(k, len(s.placements) - 1)
    s.structure_place = len(s.placements) - 1
    li = structure.level_index_of(s, level_uid)
    if li >= 0:
        s.branch_levels[li].expanded = True
    return True


def _reparent_level(s, i, parent_uid, after_uid=None) -> None:
    """Make level `i` grow on `parent_uid`: last among its levels, or right after
    the level `after_uid`."""
    uid = s.branch_levels[i].uid
    s.branch_levels[i].parent_uid = parent_uid
    if after_uid is not None:
        a = structure.level_index_of(s, after_uid)
        s.branch_levels.move(i, a + 1 if a < i else a)
    else:
        s.branch_levels.move(i, len(s.branch_levels) - 1)
    pi = structure.level_index_of(s, parent_uid)
    if pi >= 0:
        s.branch_levels[pi].expanded = True
    _select_level(s, uid)


class PLANTGEN_OT_tree_indent(Operator):
    """Indent the selection -- a child of the level above it -- or outdent it,
    up a level beside its parent. A component moves into the first level
    growing on its level, or up to the level its level grows on."""

    bl_idname = "plantgen.tree_indent"
    bl_label = "Indent"
    bl_description = ("Indent: make it grow on the level above it. Outdent: move it up "
                      "a level, beside its parent")
    bl_options = {"REGISTER"}

    direction: EnumProperty(items=[("IN", "In", ""), ("OUT", "Out", "")],
                            options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, s = _structure(context)
        inward = self.direction == "IN"
        with _undo_step("Indent" if inward else "Outdent"):
            if s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(
                    s.branch_levels):
                i = s.branch_levels_active
                lv = s.branch_levels[i]
                if lv.parent_uid == 0:
                    self.report({'INFO'}, "A stems level stays on the Plant")
                    return {"CANCELLED"}
                if inward:
                    sib = structure.child_levels_of(s, lv.parent_uid)
                    at = sib.index(i)
                    if at == 0:
                        self.report({'INFO'}, "No level above it to grow on")
                        return {"CANCELLED"}
                    _reparent_level(s, i, s.branch_levels[sib[at - 1]].uid)
                else:
                    parent = s.branch_levels[structure.level_index_of(s, lv.parent_uid)]
                    if not structure.level_can_move(s, i, parent.parent_uid):
                        self.report({'INFO'}, "Branches stay on a stems or branch level")
                        return {"CANCELLED"}
                    _reparent_level(s, i, parent.parent_uid, after_uid=parent.uid)
                _after_tree_edit(context, obj, s)
                return {"FINISHED"}
            if s.structure_focus == 'COMPONENT' and 0 <= s.structure_place < len(s.placements):
                here = s.placements[s.structure_place].level_uid
                if inward:
                    kids = structure.child_levels_of(s, here)
                    target = s.branch_levels[kids[0]].uid if kids else None
                else:
                    target = (None if here == 0 else
                              s.branch_levels[structure.level_index_of(s, here)].parent_uid)
                if target is None or not _move_placement(s, s.structure_place, target):
                    self.report({'INFO'}, "Nowhere to move it that way")
                    return {"CANCELLED"}
                _after_tree_edit(context, obj, s)
                return {"FINISHED"}
        return {"CANCELLED"}


class PLANTGEN_OT_tree_move_to(Operator):
    """Move the selection onto a level: a level then grows on it (on the Plant:
    a stems level), a component is placed there instead."""

    bl_idname = "plantgen.tree_move_to"
    bl_label = "Move To"
    bl_description = "Move the selection onto this level"
    bl_options = {"REGISTER"}

    uid: IntProperty(default=0, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, s = _structure(context)
        with _undo_step("Move to level"):
            if s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(
                    s.branch_levels):
                i = s.branch_levels_active
                if (self.uid == s.branch_levels[i].parent_uid
                        or not structure.level_can_move(s, i, self.uid)):
                    return {"CANCELLED"}
                _reparent_level(s, i, self.uid)
            elif s.structure_focus == 'COMPONENT' and 0 <= s.structure_place < len(
                    s.placements):
                if not _move_placement(s, s.structure_place, self.uid):
                    self.report({'INFO'}, "Already on that level")
                    return {"CANCELLED"}
            elif s.structure_focus == 'LIBRARY' and 0 <= s.components_active < len(
                    s.components):
                # an unplaced component: placed there
                s.structure_place = structure.place(s, self.uid,
                                                 s.components[s.components_active].uid)
                s.structure_focus = 'COMPONENT'
                li = structure.level_index_of(s, self.uid)
                if li >= 0:
                    s.branch_levels[li].expanded = True
                else:
                    s.structure_plant_expanded = True
            else:
                return {"CANCELLED"}
            _after_tree_edit(context, obj, s)
        return {"FINISHED"}


class PLANTGEN_OT_tree_place(Operator):
    """Place a library component on the Plant Structure's selection."""

    bl_idname = "plantgen.tree_place"
    bl_label = "Place"
    bl_description = "Place this component on the selected level (it shares all its settings)"
    bl_options = {"REGISTER"}

    comp_uid: IntProperty(default=0, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, s = _structure(context)
        with _undo_step("Place component"):
            s.structure_place = structure.place(s, structure.structure_target(s), self.comp_uid)
            s.structure_focus = 'COMPONENT'
            i, _c = structure.component_by_uid(s, self.comp_uid)
            if i >= 0:
                s.components_active = i
            _after_tree_edit(context, obj, s)
        return {"FINISHED"}


class PLANTGEN_OT_tree_expand(Operator):
    """Expand or collapse every row of the Plant Structure."""

    bl_idname = "plantgen.tree_expand"
    bl_label = "Expand"
    bl_options = {"INTERNAL"}

    expand: BoolProperty(default=True, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, s = _structure(context)
        s.structure_plant_expanded = self.expand
        for lv in s.branch_levels:
            lv.expanded = self.expand
        structure.rebuild_tree(s)
        _redraw(context)
        return {"FINISHED"}


def _rename_target(s):
    """What F2 renames: the Plant Structure's selected level, else the selected
    (tree or library) component. (owner, prop, hint)"""
    if s.structure_focus == 'LEVEL' and 0 <= s.branch_levels_active < len(s.branch_levels):
        return s.branch_levels[s.branch_levels_active], "name", None
    if 0 <= s.components_active < len(s.components):
        return (s.components[s.components_active], "label",
                "Empty: back to the collection's name")
    return None, None, None


class PLANTGEN_OT_rename(Operator):
    """Rename the selected level or component (F2 over the PlantGen sidebar)."""

    bl_idname = "plantgen.rename"
    bl_label = "Rename"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        region = context.region
        if region is None or region.type != 'UI' or getattr(
                region, "active_panel_category", "") != "PlantGen":
            return False
        obj = bridge.resolve_plant(context)
        return obj is not None and _rename_target(obj.plantgen)[0] is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=260)

    def draw(self, context):
        owner, prop, hint = _rename_target(bridge.resolve_plant(context).plantgen)
        if owner is None:
            return
        self.layout.activate_init = True
        self.layout.prop(owner, prop, text="")
        if hint:
            self.layout.label(text=hint, icon='INFO')

    def execute(self, context):
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_component_move(Operator):
    """Move the active component up or down the list. Which component takes a point
    is by Density weight, not order, so this only reorders the list."""

    bl_idname = "plantgen.component_move"
    bl_label = "Move Component"
    bl_description = "Move the active component up or down the list"
    bl_options = {"REGISTER"}

    direction: EnumProperty(
        items=[("UP", "Up", ""), ("DOWN", "Down", "")], options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        obj = bridge.resolve_plant(context)
        return obj is not None and len(obj.plantgen.components) > 1

    def execute(self, context):
        obj = bridge.resolve_plant(context)
        if obj is None:
            return {"CANCELLED"}
        s = obj.plantgen
        active = s.components_active
        target = active - 1 if self.direction == "UP" else active + 1
        if not (0 <= target < len(s.components)):
            return {"CANCELLED"}
        with _undo_step("Move component"):
            s.components.move(active, target)
            s.components_active = target
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_mask_debug(Operator):
    """Preview one filter's attribute across the plant: colour it by this filter's
    own window, red where it masks out to green where it keeps. Click again to
    turn it off."""

    bl_idname = "plantgen.mask_debug"
    bl_label = "Preview Attribute"
    bl_description = ("Show this mask's attribute window across the plant, red "
                      "(masked out) to green (kept). Click again to turn off")
    bl_options = {"REGISTER"}

    index: IntProperty(default=-1, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        _obj, c = _active_component(context)
        return c is not None

    def execute(self, context):
        obj, c = _active_component(context)
        if c is None:
            return {"CANCELLED"}
        s = obj.plantgen
        if not (0 <= self.index < len(c.filters)):
            return {"CANCELLED"}
        on = (s.debug_view_mode == 'mask_preview' and s.mask_preview_level < 0
              and c.filters_active == self.index)
        with _undo_step("Toggle attribute preview"):
            s.mask_preview_level = -1
            c.filters_active = self.index
            s.debug_view_mode = 'off' if on else 'mask_preview'
            bridge.regrow(obj)  # mask_preview is lazy -- regrow bakes it
        _redraw(context)
        return {"FINISHED"}


def _mask_owner(context, owner: str):
    """(plant, the PG owning the mask stack) -- the active component, with owner
    "CLASS" the plant's active class, with "LEVEL" the active branching level --
    or (plant, None)."""
    if owner in ("CLASS", "LEVEL"):
        obj = bridge.resolve_plant(context)
        if obj is None:
            return None, None
        s = obj.plantgen
        items, active = ((s.classes, s.classes_active) if owner == "CLASS"
                         else (s.branch_levels, s.branch_levels_active))
        if 0 <= active < len(items):
            return obj, items[active]
        return obj, None
    return _active_component(context)


class PLANTGEN_OT_mask_add(Operator):
    """Add a filter of this attribute to the active component, and regrow."""

    bl_idname = "plantgen.mask_add"
    bl_label = "Add Mask"
    bl_description = "Add a mask of this attribute to the active component"
    bl_options = {"REGISTER"}

    owner: StringProperty(default="", options={'HIDDEN'})
    """Whose mask stack: "" the active component's, "CLASS" the active class's,
    "LEVEL" the active branching level's."""

    channel: StringProperty(default="", options={'HIDDEN'})
    """Which attribute the new filter reads, chosen from the add menu. Empty falls
    back to the property's own default."""

    @classmethod
    def poll(cls, context):
        # owner is per-invocation (not visible to a classmethod poll); an
        # empty / missing stack cancels in execute instead.
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, c = _mask_owner(context, self.owner)
        if c is None:
            return {"CANCELLED"}
        with _undo_step("Add filter"):
            entry = c.filters.add()
            if self.channel:
                entry.channel = self.channel
            entry.color = _MASK_COLORS[(len(c.filters) - 1) % len(_MASK_COLORS)]
            c.filters_active = len(c.filters) - 1
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_mask_remove(Operator):
    """Remove the active filter from the active component."""

    bl_idname = "plantgen.mask_remove"
    bl_label = "Remove Mask"
    bl_description = "Remove the active mask"
    bl_options = {"REGISTER"}

    owner: StringProperty(default="", options={'HIDDEN'})
    """Whose mask stack: "" the active component's, "CLASS" the active class's,
    "LEVEL" the active branching level's."""

    @classmethod
    def poll(cls, context):
        # owner is per-invocation (not visible to a classmethod poll); an
        # empty / missing stack cancels in execute instead.
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, c = _mask_owner(context, self.owner)
        if c is None:
            return {"CANCELLED"}
        active = c.filters_active
        if not (0 <= active < len(c.filters)):
            return {"CANCELLED"}
        with _undo_step("Remove filter"):
            c.filters.remove(active)
            c.filters_active = min(active, len(c.filters) - 1)
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_mask_move(Operator):
    """Move the active filter up or down -- order matters, the first filter is the
    base the rest fold onto."""

    bl_idname = "plantgen.mask_move"
    bl_label = "Move Mask"
    bl_description = "Move the active mask up or down"
    bl_options = {"REGISTER"}

    owner: StringProperty(default="", options={'HIDDEN'})
    """Whose mask stack: "" the active component's, "CLASS" the active class's,
    "LEVEL" the active branching level's."""

    direction: EnumProperty(
        items=[("UP", "Up", ""), ("DOWN", "Down", "")], options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        # owner is per-invocation (not visible to a classmethod poll); an
        # empty / missing stack cancels in execute instead.
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj, c = _mask_owner(context, self.owner)
        if c is None:
            return {"CANCELLED"}
        active = c.filters_active
        target = active - 1 if self.direction == "UP" else active + 1
        if not (0 <= target < len(c.filters)):
            return {"CANCELLED"}
        with _undo_step("Move filter"):
            c.filters.move(active, target)
            c.filters_active = target
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


_PLANT_LISTS = [("custom_attrs", "Custom Attributes", "The plant's derived channels"),
                ("classes", "Classes", "The plant's named, reusable mask stacks")]


def _unique_name(existing, base: str) -> str:
    """`base`, or `base.001` ... -- masks and maps reference these by name, so
    two with one name would be ambiguous."""
    if base not in existing:
        return base
    i = 1
    while f"{base}.{i:03d}" in existing:
        i += 1
    return f"{base}.{i:03d}"


class PLANTGEN_OT_plant_list_add(Operator):
    """Add a custom attribute or a class to the plant."""

    bl_idname = "plantgen.plant_list_add"
    bl_label = "Add"
    bl_description = "Add a custom attribute / class to this plant"
    bl_options = {"REGISTER"}

    list_name: EnumProperty(items=_PLANT_LISTS, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj = bridge.resolve_plant(context)
        s = obj.plantgen
        coll = getattr(s, self.list_name)
        base = "Attribute" if self.list_name == "custom_attrs" else "Class"
        with _undo_step("Add " + base.lower()):
            item = coll.add()
            item.name = _unique_name({x.name for x in coll if x != item}, base)
            setattr(s, self.list_name + "_active", len(coll) - 1)
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_plant_list_remove(Operator):
    """Remove the active custom attribute or class. Masks and maps still naming
    it read as identity until pointed elsewhere."""

    bl_idname = "plantgen.plant_list_remove"
    bl_label = "Remove"
    bl_description = "Remove the active custom attribute / class"
    bl_options = {"REGISTER"}

    list_name: EnumProperty(items=_PLANT_LISTS, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        obj = bridge.resolve_plant(context)
        s = obj.plantgen
        coll = getattr(s, self.list_name)
        active = getattr(s, self.list_name + "_active")
        if not (0 <= active < len(coll)):
            return {"CANCELLED"}
        with _undo_step("Remove"):
            coll.remove(active)
            setattr(s, self.list_name + "_active", min(active, len(coll) - 1))
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_level_mask_debug(Operator):
    """Preview one of the active branching level's masks: the parent stems
    coloured where growth kept (green) or rejected (red) this level's branches
    by this mask's own window. Click again to turn it off."""

    bl_idname = "plantgen.level_mask_debug"
    bl_label = "Preview Level Mask"
    bl_description = ("Show where this mask kept (green) or removed (red) this level's "
                      "branches, on the axes they grow from. Click again to turn off")
    bl_options = {"REGISTER"}

    index: IntProperty(default=-1, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        obj = bridge.resolve_plant(context)
        return obj is not None and len(obj.plantgen.branch_levels) > 0

    def execute(self, context):
        obj = bridge.resolve_plant(context)
        if obj is None:
            return {"CANCELLED"}
        s = obj.plantgen
        li = s.branch_levels_active
        if not (0 <= li < len(s.branch_levels)):
            return {"CANCELLED"}
        lv = s.branch_levels[li]
        if not (0 <= self.index < len(lv.filters)):
            return {"CANCELLED"}
        on = (s.debug_view_mode == 'mask_preview' and s.mask_preview_level == li
              and lv.filters_active == self.index)
        with _undo_step("Toggle level mask preview"):
            s.mask_preview_level = -1 if on else li
            lv.filters_active = self.index
            s.debug_view_mode = 'off' if on else 'mask_preview'
            bridge.regrow(obj)
        _redraw(context)
        return {"FINISHED"}


_MAP_OWNERS = [("COMPONENT", "Component", "The active component's instance maps"),
               ("GROWTH", "Growth", "The plant's growth maps"),
               ("LEVEL", "Level", "The active branching level's maps")]


def _map_owner(context, owner: str):
    """(plant, collection owner PG, maps collection prop name, curve owner uid,
    curve name prefix) for the map list `owner` names, or None."""
    if owner == "GROWTH":
        plant = bridge.resolve_plant(context)
        if plant is None:
            return None
        return (plant, plant.plantgen, "growth_maps", curves.GROWTH_MAP_OWNER, "gmap")
    if owner == "LEVEL":
        plant = bridge.resolve_plant(context)
        if plant is None:
            return None
        s = plant.plantgen
        if not (0 <= s.branch_levels_active < len(s.branch_levels)):
            return None
        return (plant, s.branch_levels[s.branch_levels_active], "maps",
                curves.GROWTH_MAP_OWNER, bridge.LEVEL_MAP_KIND)
    plant, c = _active_component(context)
    if c is None:
        return None
    return plant, c, "maps", c.uid, "imap"


class PLANTGEN_OT_map_add(Operator):
    """Add a map -- a Driver channel through its own curve onto one parameter."""

    bl_idname = "plantgen.map_add"
    bl_label = "Add Map"
    bl_description = ("Add a map: a per-point attribute read through a curve that "
                      "drives one parameter")
    bl_options = {"REGISTER"}

    owner: EnumProperty(items=_MAP_OWNERS, default="COMPONENT", options={"HIDDEN"})
    target: StringProperty(default="", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        found = _map_owner(context, self.owner)
        if found is None:
            return {"CANCELLED"}
        plant, pg, prop, curve_uid, prefix = found
        maps = getattr(pg, prop)
        with _undo_step("Add map"):
            # A level map's uid is unique over every level's maps (they share
            # one curve owner); any other map's over its own list.
            pool = ([x for lv in plant.plantgen.branch_levels for x in lv.maps]
                    if self.owner == "LEVEL" else list(maps))
            m = maps.add()
            m.uid = max((x.uid for x in pool), default=0) + 1
            if self.target:
                m.target = self.target
            node = curves.ensure_map_curve(
                plant, curve_uid, curves.map_curve_name(prefix, m.uid))
            # A rotation map is "no turn" at the curve's middle, so it starts flat
            # at 0.5 (a no-op, like every other new control); a multiplier at 1.
            rotation = prop == "maps" and m.target in mapping.pm.ROTATION_TARGETS
            curves.set_points(node, ((0.0, 0.5), (1.0, 0.5)) if rotation
                              else ((0.0, 1.0), (1.0, 1.0)))
            setattr(pg, prop + "_active", len(maps) - 1)
            bridge.regrow(plant)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_map_remove(Operator):
    """Remove the active map (and its curve)."""

    bl_idname = "plantgen.map_remove"
    bl_label = "Remove Map"
    bl_description = "Remove the active map"
    bl_options = {"REGISTER"}

    owner: EnumProperty(items=_MAP_OWNERS, default="COMPONENT", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def execute(self, context):
        found = _map_owner(context, self.owner)
        if found is None:
            return {"CANCELLED"}
        plant, pg, prop, curve_uid, prefix = found
        maps = getattr(pg, prop)
        active = getattr(pg, prop + "_active")
        if not (0 <= active < len(maps)):
            return {"CANCELLED"}
        with _undo_step("Remove map"):
            curves.remove_named_component_curve(
                plant, curve_uid, curves.map_curve_name(prefix, maps[active].uid))
            maps.remove(active)
            setattr(pg, prop + "_active", min(active, len(maps) - 1))
            bridge.regrow(plant)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_showcase(Operator):
    """Generate a labelled grid of plants differing in exactly one thing.

    A tuning instrument rather than an authoring one. The structural checks say
    a plant is sound and the goldens say it is unchanged; neither says it looks
    right, and side by side is the only way that question gets answered.
    """

    bl_idname = "plantgen.showcase"
    bl_label = "Sweep Parameter"
    bl_description = ("Build a labelled row of plants walking one control across its own "
                      "range, in its own collection. Every other setting is copied from the "
                      "active plant. Re-running replaces the row rather than adding a second")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        collection, count, note = showcase.build(
            context, scene.plantgen_showcase, scene.plantgen_showcase_steps)
        if note:
            self.report({"WARNING"}, note)
            return {"CANCELLED"}
        self.report({"INFO"}, f"Built {count} plants in {collection.name}")
        return {"FINISHED"}


class PLANTGEN_OT_run_checks(Operator):
    """Run the structural assertions against the active plant and show the
    result in the panel.

    On demand rather than on every regrow, deliberately: two of the ten checks
    (`check_age_monotonic` and `check_determinism`) re-grow the plant several
    times over, so wiring this into `regrow` would multiply the cost of every
    slider drag. This is the "is my plant actually sound" button, not a
    background monitor.
    """

    bl_idname = "plantgen.run_checks"
    bl_label = "Run Structural Checks"
    bl_description = ("Check the active plant against the ten structural assertions - "
                      "branch order, trunk dominance, termination, monotonic growth, "
                      "leaves, flowers, taper, determinism, physics and environment")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        from ..growth.development import build
        from ..tests import structural

        obj = context.active_object
        s = obj.plantgen
        profile = bridge.profile_from_object(obj)

        def grow_fn(age: float):
            return build(profile, seed=s.seed, age=age)

        plant = grow_fn(s.age)
        results = structural.run_checks(
            plant, profile.expectations(), grow_fn=grow_fn,
            # Several checks only hold at maturity -- an axis still growing at
            # Age 0.4 has not failed to terminate, it simply has not finished.
            at_full_age=s.age >= 1.0, profile=profile)

        s.checks.clear()
        for r in results:
            entry = s.checks.add()
            entry.number, entry.name = r.number, r.name
            entry.status, entry.detail = r.status, r.detail
        s.checks_ran = True

        failed = [r for r in results if r.status == structural.FAIL]
        if failed:
            self.report({"WARNING"},
                        f"{len(failed)} of {len(results)} checks failed: "
                        + ", ".join(str(r.number) for r in failed))
        else:
            self.report({"INFO"}, f"All {len(results)} structural checks passed")
        return {"FINISHED"}


class PLANTGEN_OT_edit_variance(Operator):
    """Variance lives behind a small button on the row it varies rather than
    taking a row of its own -- the parametric add-on's pattern, and the reason its panel
    stays readable at thirty-odd controls. Same popup dialog, same "one
    operator, named field" shape as `plantgen.edit_variance`."""

    bl_idname = "plantgen.edit_variance"
    bl_label = "Edit Variance"
    bl_description = "Set how much this parameter randomly deviates on each Seed re-roll"
    bl_options = {"REGISTER", "UNDO"}

    prop_name: StringProperty(options={"HIDDEN"})  # e.g. "leaf_size_variation"
    label: StringProperty(options={"HIDDEN"})
    # "" -> the property lives on the plant; "component" -> on the active component
    # (the per-component instancing variances share this one dialog); "level" ->
    # on the active branching level (its Length / Width variations).
    target: StringProperty(options={"HIDDEN"})
    variance: FloatProperty(name="Variance", min=0.0, soft_max=8.0)

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def _owner(self, context):
        s = context.active_object.plantgen
        if self.target == "component" and 0 <= s.components_active < len(s.components):
            return s.components[s.components_active]
        if self.target == "level" and 0 <= s.branch_levels_active < len(s.branch_levels):
            return s.branch_levels[s.branch_levels_active]
        return s

    def invoke(self, context, event):
        self.variance = getattr(self._owner(context), self.prop_name)
        return context.window_manager.invoke_props_dialog(self, width=250)

    def draw(self, context):
        self.layout.prop(self, "variance", text=self.label or self.prop_name)

    def execute(self, context):
        # The property's own update= callback marks the plant dirty, same as
        # every other slider in the panel -- no explicit regrow here.
        setattr(self._owner(context), self.prop_name, self.variance)
        return {"FINISHED"}


class PLANTGEN_OT_curve_shape(Operator):
    bl_idname = "plantgen.curve_shape"
    bl_label = "Curve Shape"
    bl_description = "Replace this curve's control points with a common shape"
    bl_options = {"REGISTER", "UNDO"}

    curve_key: StringProperty(options={"HIDDEN"})
    shape: StringProperty(options={"HIDDEN"})  # a curves.SHAPES key
    # True -> the active component's own size curve, not a fixed plant curve.
    component: BoolProperty(default=False, options={"HIDDEN"})
    # A component's curve by uid (the Mapping panel's owner), else the active one.
    comp_uid: IntProperty(default=0, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bridge.is_plant(context.active_object)

    def execute(self, context):
        obj = context.active_object
        if self.component:
            s = obj.plantgen
            uid = self.comp_uid
            if not uid:
                if not (0 <= s.components_active < len(s.components)):
                    return {"CANCELLED"}
                uid = s.components[s.components_active].uid
            node = curves.ensure_component_curve(obj, uid, self.curve_key or "size")
        else:
            node = curves.ensure_curves(obj).get(self.curve_key)
        if node is None or not curves.apply_shape(node, self.shape):
            return {"CANCELLED"}
        live.mark_dirty(obj)
        return {"FINISHED"}


class PLANTGEN_OT_save_undo_log(Operator):
    """Write the undo trace to a text file next to the .blend.

    A debugging aid for one specific class of problem: the plant and the panel
    disagreeing after Ctrl+Z. The trace is the only way to see it, because by
    the time anyone looks the state has usually healed.
    """

    bl_idname = "plantgen.save_undo_log"
    bl_label = "Save Undo Log"
    bl_description = "Write what happened around each undo to a text file, for diagnosis"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from . import live

        lines = live.trace_lines()
        folder = bpy.path.abspath("//") if bpy.data.filepath else bpy.app.tempdir
        path = os.path.join(folder, "plantgen_undo_log.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.report({"INFO"}, f"{len(lines)} events -> {path}")
        print(f"[plantgen] undo log ({len(lines)} events) -> {path}")
        return {"FINISHED"}


class PLANTGEN_OT_clear_undo_log(Operator):
    """Empty the trace, so the next reproduction starts from nothing."""

    bl_idname = "plantgen.clear_undo_log"
    bl_label = "Clear"
    bl_description = "Empty the undo trace before reproducing a problem"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from . import live

        live.clear_trace()
        self.report({"INFO"}, "Undo trace cleared")
        return {"FINISHED"}


class PLANTGEN_OT_generate_docs(Operator):
    """Regenerate the documentation from the current model and panels.

    A build-time tool, not a plant operation. Run it after changing a panel, a
    property's label / range / tooltip / enum, the profile dataclasses, the
    kind-capability table, or field-notes.json: it rewrites the LLM spec
    (docs/llm/plantgen-spec.json), the manual's panel reference
    (docs/ui-reference.json, drawn from the panels themselves -- `ui_spec`) and
    the HTML (docs/plantgen-docs.html), so none can drift. It reports whether
    the spec was already stale -- that report is the anti-drift guard. The HTML
    build is pure Python (no bpy), reached through the repo's tools/ (resolve()
    follows the install junction, so the real repo path is found).
    """

    bl_idname = "plantgen.generate_docs"
    bl_label = "Regenerate Docs"
    bl_description = ("Rebuild the LLM spec, the panel reference and the HTML docs "
                      "from the model and the panels, and report whether the spec "
                      "had drifted")
    bl_options = {"REGISTER"}

    def execute(self, context):
        import importlib
        import sys
        from pathlib import Path

        from . import spec, ui_spec

        was_stale, summary = spec.is_stale()
        spec_path = spec.write_spec()
        ui_spec.write_isolated()       # a factory Blender on staged examples, not this file

        tools = str(Path(spec.__file__).resolve().parents[2] / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        try:
            import _docs_assets
            import build_docs
            import ui_reference
            importlib.reload(_docs_assets)
            importlib.reload(ui_reference)
            importlib.reload(build_docs)
            html_path = build_docs.build()
        except Exception as exc:  # noqa: BLE001 -- surface any build error to the user
            self.report({"WARNING"}, f"Spec written, but HTML build failed: {exc}")
            print(f"[plantgen] HTML build failed: {exc}")
            return {"FINISHED"}

        verb = f"was STALE ({summary}), regenerated" if was_stale else "already up to date"
        msg = (f"Docs {verb}: {spec_path.name} + {html_path.name} "
               f"({html_path.stat().st_size // 1024} KB)")
        self.report({"WARNING" if was_stale else "INFO"}, msg)
        print(f"[plantgen] {msg}")
        return {"FINISHED"}


class PLANTGEN_OT_open_docs(Operator):
    """Open the offline PlantGen documentation in the default browser.

    docs/plantgen-docs.html is a self-contained file generated from the model
    (tools/build_docs.py), so it works with no network. Opened with wm.path_open
    -- the reliable way to hand a local file to the OS default app -- rather than
    a file:// URL, whose handling varies.
    """

    bl_idname = "plantgen.open_docs"
    bl_label = "Open Documentation"
    bl_description = "Open the offline PlantGen documentation in your browser"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from pathlib import Path

        docs = Path(__file__).resolve().parent.parent / "docs" / "plantgen-docs.html"
        if not docs.is_file():
            self.report({"WARNING"}, "Documentation not built yet "
                        "(run tools/build_docs.py)")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=str(docs))
        return {"FINISHED"}


# --- parameter mapping ------------------------------

def _pick(context):
    """The parameter a map is about to be added to: (owner, uid, param)."""
    raw = context.window_manager.get("plantgen_map_pick", "")
    parts = raw.split(":", 2)
    if len(parts) != 3:
        return None
    return parts[0], int(parts[1]), parts[2]


class PLANTGEN_OT_map_param(Operator):
    """Map this parameter"""

    bl_idname = "plantgen.map_param"
    bl_label = "Map Parameter"
    bl_options = {"INTERNAL"}

    owner: StringProperty(options={"HIDDEN"})
    owner_uid: IntProperty(options={"HIDDEN"})
    param: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, props):
        plant = bridge.resolve_plant(context)
        if plant is None:
            return "Map this parameter"
        return mapping.describe(plant.plantgen, props.owner, props.owner_uid, props.param)

    def invoke(self, context, event):
        plant = bridge.resolve_plant(context)
        if plant is None:
            return {"CANCELLED"}
        s = plant.plantgen
        if mapping.entries(s, self.owner, self.owner_uid, self.param):
            mapping.select(s, self.owner, self.owner_uid, self.param)
        else:
            context.window_manager["plantgen_map_pick"] = (
                f"{self.owner}:{self.owner_uid}:{self.param}")
            bpy.ops.wm.call_menu(name="PLANTGEN_MT_map_presets")
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_MT_map_presets(Menu):
    """Start a map on the parameter just picked from one of its presets."""

    bl_idname = "PLANTGEN_MT_map_presets"
    bl_label = "Add Map"

    def draw(self, context):
        layout = self.layout
        plant = bridge.resolve_plant(context)
        pick = _pick(context)
        if plant is None or pick is None:
            layout.label(text="Pick a parameter first")
            return
        s = plant.plantgen
        owner, uid, param = pick
        pg = mapping.owner_pg(s, owner, uid)
        layout.label(text=f"{mapping.owner_title(s, owner, uid)} \u203a "
                          f"{mapping.param_label(pg, param)}",
                     icon_value=mapping.icon(s, owner, uid, "swatch"))
        layout.separator()
        # The parameter's own profile curve first, where it has one.
        curve = mapping.curve_entry(s, owner, uid, param)
        if curve is not None and not any(k == "CURVE" for k, _i in
                                         mapping.entries(s, owner, uid, param)):
            op = layout.operator("plantgen.param_map_add", text=curve[1])
            op.owner, op.owner_uid, op.param, op.preset = owner, uid, param, 'PROFILE'
        for preset in mapping.presets_for(owner, param):
            op = layout.operator("plantgen.param_map_add",
                                 text=mapping.preset_label(owner, param, preset))
            op.owner, op.owner_uid, op.param, op.preset = owner, uid, param, preset


def _row_mode(s, row):
    """The selected map's mode: Random Range for a +/- value's row, else its
    stored preset; None for a row with no modes (a profile curve)."""
    if row is None or row.kind in ("OWNER", "CURVE"):
        return None
    if row.kind == "VAR":
        return 'RANGE'
    m = mapping.stored(s, row.kind, row.owner, row.owner_uid, row.index)
    return getattr(m, "preset", None)


class PLANTGEN_MT_map_mode(Menu):
    """The selected map's mode -- only the modes its parameter supports (a
    parameter read once per plant has no attribute to vary along)."""

    bl_idname = "PLANTGEN_MT_map_mode"
    bl_label = "Mode"

    def draw(self, context):
        layout = self.layout
        plant = bridge.resolve_plant(context)
        s = plant.plantgen if plant is not None else None
        row = mapping.active_row(s) if s is not None else None
        current = _row_mode(s, row)
        if current is None:
            return
        for preset in mapping.presets_for(row.owner, row.param):
            op = layout.operator("plantgen.map_mode",
                                 text=mapping.preset_label(row.owner, row.param, preset),
                                 icon="CHECKMARK" if preset == current else "BLANK1")
            op.preset = preset


class PLANTGEN_OT_map_mode(Operator):
    """Switch the selected map to another mode (a Random Range on a +/- value
    becomes that map, its +/- going to 0)"""

    bl_idname = "plantgen.map_mode"
    bl_label = "Map Mode"
    bl_options = {"REGISTER", "INTERNAL"}

    preset: StringProperty(default="RANGE", options={"HIDDEN"})

    def execute(self, context):
        plant = bridge.resolve_plant(context)
        s = plant.plantgen if plant is not None else None
        row = mapping.active_row(s) if s is not None else None
        current = _row_mode(s, row)
        if current is None or self.preset == current:
            return {"CANCELLED"}
        with _undo_step("Map mode"):
            if row.kind == "VAR":
                if mapping.convert_var(plant, row.owner, row.owner_uid, row.param,
                                       self.preset) is None:
                    return {"CANCELLED"}
            else:
                m = mapping.stored(s, row.kind, row.owner, row.owner_uid, row.index)
                m.preset = self.preset           # its update re-shapes driver + curve
                mapping.rebuild_rows(s)
            bridge.regrow(plant)
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_param_map_add(Operator):
    """Add a map to a parameter"""

    bl_idname = "plantgen.param_map_add"
    bl_label = "Add Map"
    bl_options = {"REGISTER", "INTERNAL"}

    owner: StringProperty(options={"HIDDEN"})
    owner_uid: IntProperty(options={"HIDDEN"})
    param: StringProperty(options={"HIDDEN"})
    preset: StringProperty(default="RANGE", options={"HIDDEN"})

    def execute(self, context):
        plant = bridge.resolve_plant(context)
        if plant is None:
            return {"CANCELLED"}
        with _undo_step("Add map"):
            if mapping.add_map(plant, self.owner, self.owner_uid, self.param,
                               self.preset) is None:
                return {"CANCELLED"}
            bridge.regrow(plant)
        _redraw(context)
        return {"FINISHED"}


_SEARCH_ITEMS: list = []


def _search_items(self, context):
    """Every mappable parameter on the plant, as 'Owner > Parameter'. Kept in
    a module list: Blender drops a callback's items that nothing references."""
    _SEARCH_ITEMS.clear()
    plant = bridge.resolve_plant(context)
    if plant is not None:
        s = plant.plantgen
        for owner, uid in mapping.owners(s):
            pg = mapping.owner_pg(s, owner, uid)
            title = mapping.owner_title(s, owner, uid)
            for param in mapping.searchable_params(pg, owner):
                key = f"{owner}:{uid}:{param}"
                _SEARCH_ITEMS.append((key, f"{title} \u203a {mapping.param_label(pg, param)}",
                                      ""))
    return _SEARCH_ITEMS or [("", "(nothing to map)", "")]


class PLANTGEN_OT_map_search_add(Operator):
    """Map any parameter of the plant: search it by name"""

    bl_idname = "plantgen.map_search_add"
    bl_label = "Map Parameter"
    bl_property = "target"
    bl_options = {"INTERNAL"}

    target: EnumProperty(name="Parameter", items=_search_items)

    @classmethod
    def poll(cls, context):
        return bridge.resolve_plant(context) is not None

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.target:
            return {"CANCELLED"}
        context.window_manager["plantgen_map_pick"] = self.target
        bpy.ops.wm.call_menu(name="PLANTGEN_MT_map_presets")
        return {"FINISHED"}


class PLANTGEN_OT_map_row_remove(Operator):
    """Remove the selected map (a Random Range from before goes to 0)"""

    bl_idname = "plantgen.map_row_remove"
    bl_label = "Remove Map"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        plant = bridge.resolve_plant(context)
        if plant is None:
            return False
        row = mapping.active_row(plant.plantgen)
        return row is not None and row.kind != "OWNER"

    def execute(self, context):
        plant = bridge.resolve_plant(context)
        row = mapping.active_row(plant.plantgen)
        if row is None or row.kind == "OWNER":
            return {"CANCELLED"}
        with _undo_step("Remove map"):
            mapping.remove(plant, row)
            bridge.regrow(plant)
        _redraw(context)
        return {"FINISHED"}


# --- variations (variations.py) ---------------------------------------------------

def _variation_source(context):
    from . import variations
    return variations.source_of(bridge.resolve_plant(context))


def _with_progress(context, run):
    wm = context.window_manager
    wm.progress_begin(0, 100)
    try:
        return run(lambda done, total: wm.progress_update(int(100 * done / max(total, 1))))
    finally:
        wm.progress_end()


class PLANTGEN_OT_variations_build(Operator):
    """Build (or rebuild) this plant's variations from its current settings: a row
    or grid of copies, each its own seed, the chosen parameters stepped across
    them. The old set is replaced"""

    bl_idname = "plantgen.variations_build"
    bl_label = "Build Variations"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _variation_source(context) is not None

    def execute(self, context):
        from . import variations
        source = _variation_source(context)
        count, note = _with_progress(context, lambda p: variations.build(source, p))
        if note:
            self.report({'WARNING'}, note)
        self.report({'INFO'}, f"Built {count} variation{'s' if count != 1 else ''}")
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_variations_clear(Operator):
    """Delete this plant's variations (their plants and labels). Baked meshes stay"""

    bl_idname = "plantgen.variations_clear"
    bl_label = "Clear Variations"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from . import variations
        source = _variation_source(context)
        return source is not None and bool(variations.generated(source))

    def execute(self, context):
        from . import variations
        source = _variation_source(context)
        if context.active_object is not None and context.active_object.get(variations.MARKER):
            context.view_layer.objects.active = source      # never hold a deleted one
        count = variations.clear(source)
        self.report({'INFO'}, f"Removed {count} variation{'s' if count != 1 else ''}")
        _redraw(context)
        return {"FINISHED"}


class PLANTGEN_OT_variations_export(Operator):
    """Export this plant and its variations as one USD file of Nanite Assemblies
    for Unreal 5.7+. Imported from one file, they share their leaf and flower
    meshes, materials and textures. Beside it, one wind JSON per plant. A plant
    with no current rig gets one for the export"""

    bl_idname = "plantgen.variations_export"
    bl_label = "Export Variations"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.usdc", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        from . import variations
        source = _variation_source(context)
        if source is None or not variations.generated(source):
            cls.poll_message_set("Build the variations first")
            return False
        return True

    def invoke(self, context, event):
        if not self.filepath:
            name = bpy.path.clean_name(_variation_source(context).name)
            self.filepath = name + "_variations.usdc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from . import assembly
        source = _variation_source(context)
        out = _with_progress(
            context, lambda p: assembly.export_variations(source, self.filepath, p))
        if "error" in out:
            self.report({'ERROR'}, f"Export failed: {out['error']}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported {out['plants']} plants, {out['instances']} instances "
                    f"to {out['filepath']}")
        return {'FINISHED'}


class PLANTGEN_OT_variations_bake(Operator):
    """Bake every variation to one plain mesh object (welded, leaves and flowers
    realized, no modifiers), in its own collection. The live variations are hidden"""

    bl_idname = "plantgen.variations_bake"
    bl_label = "Bake Variations"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from . import variations
        source = _variation_source(context)
        return source is not None and bool(variations.generated(source))

    def execute(self, context):
        from . import variations
        source = _variation_source(context)
        count = _with_progress(context, lambda p: variations.bake(source, p))
        self.report({'INFO'}, f"Baked {count} plant{'s' if count != 1 else ''} to meshes")
        _redraw(context)
        return {"FINISHED"}


_VARIATION_ITEMS: list = []


def _variation_param_items(self, context):
    """Every plant-wide number an axis can step (kept in a module list:
    Blender drops a callback's items nothing references)."""
    from . import variations
    _VARIATION_ITEMS.clear()
    _VARIATION_ITEMS.append(("NONE", "None", "Nothing stepped: the plants differ by seed only"))
    for ident, name in sorted(variations.parameter_items(), key=lambda e: e[1]):
        _VARIATION_ITEMS.append((ident, name, ident))
    return _VARIATION_ITEMS


class PLANTGEN_OT_variations_pick(Operator):
    """Pick the parameter to step: search it by name"""

    bl_idname = "plantgen.variations_pick"
    bl_label = "Pick Parameter"
    bl_property = "param"
    bl_options = {"INTERNAL", "UNDO"}

    axis: StringProperty(default="x", options={"HIDDEN"})
    param: EnumProperty(name="Parameter", items=_variation_param_items)

    @classmethod
    def poll(cls, context):
        return _variation_source(context) is not None

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        v = _variation_source(context).plantgen.variations
        setattr(v, f"{self.axis}_param", "" if self.param == "NONE" else self.param)
        _redraw(context)
        return {"FINISHED"}


_classes = (
    PLANTGEN_OT_variations_build, PLANTGEN_OT_variations_clear,
    PLANTGEN_OT_variations_bake, PLANTGEN_OT_variations_export, PLANTGEN_OT_variations_pick,
    PLANTGEN_OT_map_param, PLANTGEN_MT_map_presets, PLANTGEN_OT_param_map_add,
    PLANTGEN_MT_map_mode, PLANTGEN_OT_map_mode,
    PLANTGEN_OT_map_search_add, PLANTGEN_OT_map_row_remove,
    PLANTGEN_OT_level_mask_debug,
    PLANTGEN_OT_component_copy,
    PLANTGEN_OT_tree_add_level, PLANTGEN_OT_tree_remove, PLANTGEN_OT_tree_move,
    PLANTGEN_OT_tree_indent, PLANTGEN_OT_tree_move_to, PLANTGEN_OT_tree_place,
    PLANTGEN_OT_tree_expand, PLANTGEN_OT_rename,
    PLANTGEN_AddonPreferences,
    PLANTGEN_OT_save_undo_log,
    PLANTGEN_OT_clear_undo_log,
    PLANTGEN_OT_generate_docs,
    PLANTGEN_OT_open_docs,
    PLANTGEN_OT_add_plant,
    PLANTGEN_OT_regrow,
    PLANTGEN_OT_weld,
    PLANTGEN_OT_bake_timelapse,
    PLANTGEN_OT_build_rig,
    PLANTGEN_OT_cache_timelapse,
    PLANTGEN_OT_clear_timelapse,
    PLANTGEN_OT_export_rig,
    PLANTGEN_OT_export_nanite_assembly,
    PLANTGEN_OT_apply_preset,
    PLANTGEN_OT_reload_preset,
    PLANTGEN_OT_save_preset,
    PLANTGEN_OT_load_preset,
    PLANTGEN_OT_add_effector,
    PLANTGEN_OT_showcase,
    PLANTGEN_OT_toggle_debug_view,
    PLANTGEN_OT_toggle_skin_debug,
    PLANTGEN_OT_toggle_inspect,
    PLANTGEN_OT_calendar_year,
    PLANTGEN_OT_calendar_color,
    PLANTGEN_OT_calendar_color_reset,
    PLANTGEN_OT_toggle_calendar,
    PLANTGEN_OT_toggle_pin,
    PLANTGEN_OT_unpin,
    PLANTGEN_OT_prune,
    PLANTGEN_OT_clear_pruning,
    PLANTGEN_OT_component_add,
    PLANTGEN_OT_component_remove,
    PLANTGEN_OT_component_move,
    PLANTGEN_OT_mask_add,
    PLANTGEN_OT_mask_remove,
    PLANTGEN_OT_mask_move,
    PLANTGEN_OT_map_add,
    PLANTGEN_OT_map_remove,
    PLANTGEN_OT_plant_list_add,
    PLANTGEN_OT_plant_list_remove,
    PLANTGEN_OT_mask_debug,
    PLANTGEN_OT_run_checks,
    PLANTGEN_OT_edit_variance,
    PLANTGEN_OT_curve_shape,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is not None:
        # "User Interface" runs for the sidebar; the operator's poll limits it
        # to the PlantGen tab, so F2 anywhere else still renames the object.
        km = kc.keymaps.new(name="User Interface")
        _KEYMAPS.append((km, km.keymap_items.new("plantgen.rename", 'F2', 'PRESS')))


_KEYMAPS: list = []


def unregister():
    for km, kmi in _KEYMAPS:
        try:
            km.keymap_items.remove(kmi)
        except (ReferenceError, RuntimeError):
            pass
    _KEYMAPS.clear()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
