"""Files saved before the rename (2026-10-13: the add-on was `plant_gen_sim`,
its state `plantgensim`) brought up to date on load.

- Every key that starts `plantgensim` moves to the same key with `plantgen`,
  on every datablock type and in BOTH of its stores: the custom properties
  (`plantgensim_organs_name`...) and, since Blender 5.0 a separate store, the
  system properties that hold the registered settings (`Object.plantgensim`,
  `Material.plantgensim_stem`, `Scene.plantgensim_leaf`... under their
  attribute names; `bl_system_properties_get`, invisible to `keys()`).
- Every datablock and modifier named `PlantGenSim...` takes `PlantGen...`;
  the main node group and the curve groups take `PlantGen · Plant` /
  `PlantGen · Curves`, since the retired parametric add-on used
  `PlantGen_Mesh` / `PlantGen_Curves` and an old file can still hold those.
- The retired parametric add-on's own settings (`plantgen` holding a stem's
  leaves / flowers / branches / traits, `plantgen_clump`,
  `plantgen_traits`) are cleared: nothing reads them, and `plantgen` is now
  this add-on's key. FlowerGen's `plantgen_flower` is untouched -- it never
  changed name.

Idempotent: a file with nothing named the old way is left alone."""

import bpy
from bpy.app.handlers import persistent

OLD_KEY, NEW_KEY = "plantgensim", "plantgen"
OLD_NAME, NEW_NAME = "PlantGenSim", "PlantGen"
_SPECIAL_NAMES = {"PlantGenSim_Mesh": "PlantGen · Plant",
                  "PlantGenSim_Curves": "PlantGen · Curves"}
_PARAMETRIC_KEYS = ("plantgen_clump", "plantgen_traits")
_PARAMETRIC_FIELDS = {"leaves", "flowers", "branches", "traits"}
_PLANT_FIELDS = {"branch_levels", "components", "curve_group", "day"}


def _id_collections():
    data = bpy.data
    return (data.objects, data.meshes, data.curves, data.materials, data.node_groups,
            data.collections, data.scenes, data.armatures, data.images, data.actions,
            data.worlds, data.textures, data.window_managers)


def _stores(idb):
    """The datablock's custom properties, then its system properties (if any)."""
    yield idb
    system = idb.bl_system_properties_get(do_create=False)
    if system is not None:
        yield system


def _renamed(name: str) -> str | None:
    for old, new in _SPECIAL_NAMES.items():
        if name == old or name.startswith(old + "."):
            return new + name[len(old):]
    if name.startswith(OLD_NAME):
        return NEW_NAME + name[len(OLD_NAME):]
    return None


def _parametric(group) -> bool:
    """The retired parametric add-on's `plantgen` settings, not a plant's."""
    try:
        keys = set(group.keys())
    except AttributeError:
        return False
    return bool(keys & _PARAMETRIC_FIELDS) and not keys & _PLANT_FIELDS


def needs_migration() -> bool:
    for coll in _id_collections():
        for idb in coll:
            if idb.library is not None:
                continue
            if idb.name.startswith(OLD_NAME):
                return True
            for store in _stores(idb):
                if any(k.startswith(OLD_KEY) for k in store.keys()):
                    return True
    return False


def _migrate_store(store, stats: dict) -> None:
    keys = list(store.keys())
    for k in _PARAMETRIC_KEYS:
        if k in keys:
            del store[k]
            stats["parametric"] += 1
    if NEW_KEY in keys and OLD_KEY not in keys and _parametric(store[NEW_KEY]):
        del store[NEW_KEY]
        stats["parametric"] += 1
    for k in keys:
        if not k.startswith(OLD_KEY):
            continue
        new = NEW_KEY + k[len(OLD_KEY):]
        if new in store.keys():
            del store[new]            # the parametric add-on's, or an empty default
            stats["parametric"] += 1
        store[new] = store[k]         # a deep copy, ID references kept
        del store[k]
        stats["keys"] += 1


def migrate() -> dict:
    """Bring the open file up to date; returns what changed, by kind."""
    stats = {"keys": 0, "names": 0, "modifiers": 0, "parametric": 0}
    for coll in _id_collections():
        for idb in coll:
            if idb.library is not None:
                continue
            for store in _stores(idb):
                _migrate_store(store, stats)
    for coll in _id_collections():
        for idb in coll:
            if idb.library is None and (new := _renamed(idb.name)) is not None:
                idb.name = new
                stats["names"] += 1
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if (new := _renamed(mod.name)) is not None:
                mod.name = new
                stats["modifiers"] += 1
    return stats


def _report(stats: dict) -> None:
    if any(stats.values()):
        print(f"PlantGen: brought the file up to date from PlantGen Sim -- {stats['keys']} "
              f"settings, {stats['names']} datablocks, {stats['modifiers']} modifiers renamed; "
              f"{stats['parametric']} of the retired parametric add-on's settings cleared")


@persistent
def _on_load(_dummy=None):
    try:
        if needs_migration():
            _report(migrate())
    except Exception:  # noqa: BLE001 -- never block a file from loading
        import traceback
        traceback.print_exc()


def _migrate_open_file():
    """The file already open when the add-on is enabled: migrated once, as its
    own undo step so Ctrl+Z cannot bring the old names back."""
    try:
        if needs_migration():
            _report(migrate())
            bpy.ops.ed.undo_push(message="PlantGen: update file from PlantGen Sim")
    except Exception:  # noqa: BLE001 -- a timer that raises is unregistered silently
        import traceback
        traceback.print_exc()
    return None


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    bpy.app.timers.register(_migrate_open_file, first_interval=0.0)


def unregister():
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_on_load" and getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.load_post.remove(h)
    if bpy.app.timers.is_registered(_migrate_open_file):
        bpy.app.timers.unregister(_migrate_open_file)
