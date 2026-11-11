"""Growth and leaf collision in a process of their own, so a live edit never
freezes Blender.

Blender runs the add-on's Python on its main thread -- the one that draws the
interface -- so a regrow there freezes Blender for as long as it takes (a
growth edit on a collision plant was seconds). A thread would not help: growth
and collision are Python and would hold the GIL. So `bridge.regrow_live` hands
the two heavy, bpy-free pieces -- `development.build` and
`leaf_collision.resolve` -- to this one long-lived worker and lands their
results in the caches a regrow reads (`bridge._GROW_CACHE`,
`petioles._SETTLE_CACHE`); the regrow that follows on Blender's thread then only
writes the plant. Same code, same input: the same result, byte for byte.

Run as

    python compute_worker.py <jobs folder> <code digest> <parent pid>

It reads job ids from stdin, one per line: `<id>.job` in the folder is a pickle
of (kind, payload), kind "grow" (profile, seed, day, environment, signals) or
"collide" (leaves, shapes, stems). It writes `<id>.out` (the result, pickled,
atomically) or `<id>.err` (the traceback). It imports the package from beside
this file -- `plant_gen/__init__` skips Blender's side without bpy -- and
refuses every job keyed on code other than what it loaded (`code_digest`): a
result must never come from code the add-on is not running. It exits on its
own if Blender does.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import threading
import time
import traceback

_SOURCES = ("growth", os.path.join("geometry", "leaf_collision.py"))
"""What a job's result depends on: the growth package and the collision solver."""


def code_digest(package_dir: str) -> str:
    """A digest of every source a job's result depends on (`_SOURCES`)."""
    h = hashlib.blake2b(digest_size=16)
    for rel in _SOURCES:
        path = os.path.join(package_dir, rel)
        files = ([os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".py")]
                 if os.path.isdir(path) else [path])
        for f in files:
            h.update(os.path.relpath(f, package_dir).replace("\\", "/").encode("utf-8"))
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _exit_with_parent(pid: int) -> None:
    """Exit as soon as process `pid` (Blender) has."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)     # SYNCHRONIZE
        if not handle:
            os._exit(2)

        def watch() -> None:
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)       # INFINITE
            os._exit(2)
    else:
        def watch() -> None:
            while os.getppid() == pid:
                time.sleep(1.0)
            os._exit(2)
    threading.Thread(target=watch, daemon=True).start()


def _write(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def main(argv) -> int:
    folder, digest, parent = argv[1], argv[2], int(argv[3])
    _exit_with_parent(parent)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(package_dir))
    package = os.path.basename(package_dir)
    loaded = code_digest(package_dir)
    development = __import__(f"{package}.growth.development", fromlist=["build"])
    leaf_collision = __import__(f"{package}.geometry.leaf_collision", fromlist=["resolve"])
    for line in sys.stdin:
        job_id = line.strip()
        if not job_id:
            continue
        base = os.path.join(folder, job_id)
        try:
            if loaded != digest:
                raise RuntimeError("stale worker: the add-on's code changed")
            with open(base + ".job", "rb") as fh:
                kind, payload = pickle.load(fh)
            if kind == "grow":
                profile, seed, day, environment, signals = payload
                out = development.build(profile, seed=seed, day=day,
                                        environment=environment, signals=signals)
            elif kind == "collide":
                out = leaf_collision.resolve(*payload)
            else:
                raise ValueError(f"unknown job kind {kind!r}")
            _write(base + ".out", pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:  # noqa: BLE001 -- reported to the add-on, which grows in Blender
            _write(base + ".err", traceback.format_exc().encode("utf-8"))
        finally:
            try:
                os.remove(base + ".job")
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
