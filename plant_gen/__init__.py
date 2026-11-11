bl_info = {
    "name": "PlantGen",
    "author": "Daan",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > N-Panel > PlantGen",
    "description": "Procedural perennial plants grown by simulation, with a leaf and a "
                   "flower generator for their assets.",
    "category": "Add Mesh",
}

# bpy is only present inside Blender. Import the bpy-dependent modules lazily so
# `plant_gen.growth` (pure logic) can still be imported for standalone testing.
#
# `effectors` registers before `properties`: it owns
# `Object.plantgen_effector`, which the Environment panel and the add
# operator both reach through, and a PointerProperty's type has to exist before
# anything draws it.
try:
    import bpy  # noqa: F401

    from .blender import (
        calendar_overlay,
        classify_overlay,
        effectors,
        flower_live,
        flower_material,
        flower_ops,
        flower_props,
        icons,
        live,
        migrate,
        operators,
        overlay,
        properties,
        regrow_report,
        stem_material,
        ui,
        ui_flower,
        ui_materials,
        ui_traits,
    )

    # `overlay`, `calendar_overlay` and `classify_overlay` each own a viewport draw
    # handler, so they must be torn down on unregister like any other
    # registration surface -- a handler left behind survives an addon reload and
    # draws forever with nothing left to reach it.
    # FlowerGen (the flower generator, moved in from the parametric add-on 2026-10-13):
    # its own icons, settings, operators, panels and curve watcher. `migrate`
    # first: a file saved as PlantGen Sim is brought up to date before anything
    # reads it.
    _modules = (migrate, icons, live, effectors, properties, stem_material, operators, ui,
                ui_traits, ui_materials, overlay, calendar_overlay, classify_overlay,
                regrow_report, flower_props, flower_material, flower_ops, ui_flower,
                flower_live)

    # The leaf tool is a self-contained work-in-progress module. It is imported
    # AND registered in guarded steps so an error in it -- a NameError while it
    # is mid-edit, a half-written engine change -- degrades to "leaf tool absent"
    # rather than taking the whole add-on down with it (which unregisters masks,
    # growth and everything else). Self-contained (its own scene PropertyGroup),
    # so registering it after the core modules is fine.
    try:
        from .blender import ui_leaf
        _optional_modules = (ui_leaf,)
    except Exception:
        import traceback
        print("plant_gen: leaf tool failed to import, continuing without it:")
        traceback.print_exc()
        _optional_modules = ()
except ModuleNotFoundError:
    _modules = ()
    _optional_modules = ()


def register():
    for mod in _modules:
        mod.register()
    for mod in _optional_modules:
        try:
            mod.register()
        except Exception:
            import traceback
            print("plant_gen: leaf tool failed to register, continuing without it:")
            traceback.print_exc()


def unregister():
    for mod in reversed(_optional_modules):
        try:
            mod.unregister()
        except Exception:
            import traceback
            traceback.print_exc()
    for mod in reversed(_modules):
        mod.unregister()


if __name__ == "__main__":
    register()
