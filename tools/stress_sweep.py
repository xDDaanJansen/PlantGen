"""Stress sweep before a release: plants pushed
through combinations of level, growth and flower settings by the real add-on, each
grown, welded, rigged and exported in a background Blender.

Usage:  python tools/stress_sweep.py [--workers 6] [--presets a,b] [--random 15]
                                     [--limit 300] [--out <folder>] [--no-rig]
                                     [--only preset:case,preset:case]

Each case changes settings on a fresh plant from a bundled preset: one enum value
at a time over the first levels and the flower component, then random mixes. It
flags a raised error, a crashed or hung Blender, a stem mesh with open or
non-manifold edges or bad numbers, a rig or assembly that fails, and a case over 5x
its preset's median time. A crash costs its case and nothing else: the worker
restarts after it. Results: <out>/results.jsonl, a summary on stdout; exit 1 if
anything was flagged.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLENDER = os.environ.get(
    "BLENDER_EXE", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
LIGHT = ("alopecurus_pratensis", "anemone_hupehensis", "digitalis_purpurea",
         "rudbeckia_fulgida", "lycopodium_clavatum", "araucaria_araucana")
LEVELS = 3
"""Levels per preset the enums are swept over."""
LEVEL_ENUMS = ("growth_pattern", "phyllotaxis")
FLOWER_ENUMS = ("inflorescence_kind", "inflorescence_attachments")
TOGGLES = (("light_stalks", True), ("stem_smoothness", 1), ("stem_smoothness", 8),
           ("stem_fillet", 0.0), ("stem_fillet", 2.0))
STOPPED = (0x40010004, 0xC000026B)
"""Exit codes of a process Windows ended (shutdown, logoff): the machine stopped, not
the case, so the case is unfinished, not a crash."""
PASSES = 2
"""How many enum changes a random mix takes per plant part."""


def enum_ids(prop, *classes):
    """The enum's values from the first of `classes` that has it."""
    from plant_gen.blender import properties
    for name in classes:
        rna = getattr(properties, name).bl_rna.properties.get(prop)
        if rna is not None:
            return [e.identifier for e in rna.enum_items]
    return []


def cases_for(preset, count_random):
    out = [{"preset": preset, "name": "default", "set": []}]
    level_class = "PLANTGEN_PG_branch_level"
    flower = ("PLANTGEN_PG_component", "PLANTGEN_PG_growth")
    for level in range(LEVELS):
        for prop in LEVEL_ENUMS:
            for value in enum_ids(prop, level_class):
                out.append({"preset": preset, "name": f"L{level}.{prop}={value}",
                            "set": [["level", level, prop, value]]})
    for prop in FLOWER_ENUMS:
        for value in enum_ids(prop, *flower):
            out.append({"preset": preset, "name": f"flower.{prop}={value}",
                        "set": [["flower", 0, prop, value]]})
    for prop, value in TOGGLES:
        out.append({"preset": preset, "name": f"{prop}={value}",
                    "set": [["plant", 0, prop, value]]})
    rng = random.Random(f"{preset}-sweep")
    for n in range(count_random):
        steps = []
        for _ in range(PASSES):
            level = rng.randrange(LEVELS)
            prop = rng.choice(LEVEL_ENUMS)
            steps.append(["level", level, prop, rng.choice(enum_ids(prop, level_class))])
        prop = rng.choice(FLOWER_ENUMS)
        ids = enum_ids(prop, *flower)
        if ids:
            steps.append(["flower", 0, prop, rng.choice(ids)])
        steps.append(["plant", 0, "seed", rng.randrange(1, 100000)])
        out.append({"preset": preset, "name": f"mix{n}", "set": steps})
    return out


def apply_step(obj, step):
    """False when the plant has no such part or setting (nothing to test)."""
    where, index, prop, value = step
    s = obj.plantgen
    if where == "level":
        if index >= len(s.branch_levels):
            return False
        target = s.branch_levels[index]
    elif where == "flower":
        flowers = [c for c in s.components if c.type == 'flower']
        if not flowers:
            return False
        target = flowers[0] if hasattr(flowers[0], prop) else s
    else:
        target = s
    if not hasattr(target, prop):
        return False
    rna = target.bl_rna.properties[prop]
    setattr(target, prop, {value} if getattr(rna, "is_enum_flag", False) else value)
    return True


def mesh_report(me):
    import numpy as np
    count = len(me.vertices)
    co = np.empty(count * 3)
    me.vertices.foreach_get("co", co)
    loops = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("edge_index", loops)
    uses = np.bincount(loops, minlength=len(me.edges))
    return {"verts": count, "faces": len(me.polygons), "open": int((uses == 1).sum()),
            "nonmanifold": int((uses > 2).sum()), "bad_numbers": int((~np.isfinite(co)).sum())}


def run_case(case, rig_too):
    import bpy

    from plant_gen.blender import assembly, bridge, rig

    row = {"preset": case["preset"], "name": case["name"]}
    t = time.perf_counter()
    obj = bridge.create_plant(bpy.context, name="Sweep", preset=case["preset"])
    obj.plantgen.live_update = False
    row["applied"] = [apply_step(obj, step) for step in case["set"]]
    bpy.context.view_layer.objects.active = obj
    bridge.regrow(obj, weld=bridge.WELD_NOW)
    row["grow_weld_s"] = round(time.perf_counter() - t, 2)
    row["weld"] = list(obj.get("plantgen_weld", []))
    row.update(mesh_report(obj.data))
    if rig_too:
        t = time.perf_counter()
        try:
            rig.build_rig(obj, preview=False)
        except rig.RigError as exc:
            row["rig_refused"] = str(exc)[:120]
        else:
            found = rig.find_rig(obj)
            with tempfile.TemporaryDirectory(prefix="pg_sweep_") as folder:
                path = os.path.join(folder, "Sweep_assembly.usdc")
                out = assembly.export_assembly(obj, path)
                row["instances"] = out.get("instances")
                if "error" in out:
                    row["export_error"] = out["error"]
                elif out.get("instances"):
                    checks = assembly.verify(path, out["instances"])
                    row["verify_failed"] = {k: v[1] for k, v in checks.items() if not v[0]}
            row["bones"] = len(found[0].data.bones) if found else 0
        row["rig_s"] = round(time.perf_counter() - t, 2)
    row["seconds"] = round(row["grow_weld_s"] + row.get("rig_s", 0.0), 2)
    for child in list(obj.children_recursive):
        bpy.data.objects.remove(child, do_unlink=True)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    return row


def worker(args):
    """Inside Blender: this worker's share of the cases from --start on, one line
    `start` before and one `done` after each, so a crash names its case."""
    import bpy
    sys.path.insert(0, str(REPO))
    bpy.context.preferences.addons.new().module = "plant_gen"
    import plant_gen
    plant_gen.register()
    cases = [c for preset in args.presets.split(",") for c in cases_for(preset, args.random)]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if f"{c['preset']}:{c['name']}" in wanted]
    mine = list(enumerate(cases))[args.index::args.workers]
    log = open(Path(args.out) / f"worker{args.index}.jsonl", "a", encoding="utf-8")
    clock = {"case": None, "since": time.perf_counter()}

    def watchdog():
        while True:
            time.sleep(5)
            if clock["case"] is not None and time.perf_counter() - clock["since"] > args.limit:
                log.write(json.dumps({"timeout": clock["case"]}) + "\n")
                log.flush()
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    for number, case in mine:
        if number < args.start:
            continue
        clock.update(case=number, since=time.perf_counter())
        started = {"start": number, "preset": case["preset"], "name": case["name"]}
        log.write(json.dumps(started) + "\n")
        log.flush()
        try:
            row = run_case(case, not args.no_rig)
        except Exception:  # noqa: BLE001 -- the sweep's whole point is to record these
            row = {"preset": case["preset"], "name": case["name"],
                   "error": traceback.format_exc()[-700:]}
        log.write(json.dumps({"done": number, "row": row}) + "\n")
        log.flush()
    clock["case"] = None
    log.write(json.dumps({"end": True}) + "\n")
    log.close()
    os._exit(0)


def drive_worker(index, args, results, lock):
    start = 0
    log_path = Path(args.out) / f"worker{index}.jsonl"
    log_path.unlink(missing_ok=True)
    seen = 0
    while True:
        command = [BLENDER, "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
                   "--", "--worker", "--index", str(index), "--workers", str(args.workers),
                   "--presets", args.presets, "--random", str(args.random),
                   "--limit", str(args.limit), "--start", str(start), "--out", args.out]
        command += ["--no-rig"] if args.no_rig else []
        command += ["--only", args.only] if args.only else []
        code = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              check=False).returncode
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
        last_start, finished = None, False
        for line in lines[seen:]:
            entry = json.loads(line)
            if "start" in entry:
                last_start = entry
            elif "done" in entry:
                last_start = None
                with lock:
                    results.append(entry["row"])
            elif "end" in entry:
                finished = True
        seen = len(lines)
        if finished:
            return
        if last_start is None:
            with lock:
                results.append({"preset": "?", "name": f"worker {index} outside a case",
                                "crash": code})
            return
        stopped = code in STOPPED
        verdict = "interrupted" if stopped else "timeout" if code == 3 else "crash"
        with lock:
            results.append({"preset": last_start["preset"], "name": last_start["name"],
                            verdict: code})
        if stopped:
            return
        start = last_start["start"] + 1


def flags(row, medians):
    out = []
    for key in ("error", "crash", "timeout", "interrupted", "export_error"):
        if key in row:
            out.append(key)
    if row.get("verify_failed"):
        out.append("verify")
    if row.get("verts") == 0:
        out.append("empty")
    for key in ("open", "nonmanifold", "bad_numbers"):
        if row.get(key):
            out.append(f"{key}={row[key]}")
    median = medians.get(row.get("preset"))
    if median and row.get("seconds", 0.0) > 5.0 * median and row["seconds"] > 60.0:
        out.append(f"slow={row['seconds']}s")
    return out


def main_driver(args):
    args.out = args.out or tempfile.mkdtemp(prefix="pg_sweep_")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results, lock = [], threading.Lock()
    t = time.perf_counter()
    threads = [threading.Thread(target=drive_worker, args=(i, args, results, lock))
               for i in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with open(Path(args.out) / "results.jsonl", "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")
    times = {}
    for row in results:
        if "seconds" in row:
            times.setdefault(row["preset"], []).append(row["seconds"])
    medians = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    flagged = [(row, flags(row, medians)) for row in results]
    flagged = [(row, f) for row, f in flagged if f]
    took = time.perf_counter() - t
    print(f"{len(results)} cases in {took:.0f} s, {len(flagged)} flagged. Results in {args.out}")
    for row, why in flagged:
        print(f"  {row.get('preset')} {row.get('name')}: {', '.join(why)}")
    return 1 if flagged else 0


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--presets", default=",".join(LIGHT))
    parser.add_argument("--random", type=int, default=15)
    parser.add_argument("--limit", type=float, default=300.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-rig", action="store_true")
    parser.add_argument("--only", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    tail = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    options = parse(tail)
    if options.worker:
        worker(options)
    else:
        raise SystemExit(main_driver(options))
