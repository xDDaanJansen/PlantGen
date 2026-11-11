"""The stem mesher in a process of its own, so welding never freezes Blender.

The welding is nearly all of a regrow on a flowering plant (17 s of 18 s on a
460-tube plant). When the Weld button asks for it, `bridge` runs the mesher
here: the unwelded preview stays up -- the node graph sweeping the bare
skeleton -- and Blender stays usable until the weld lands. A process, not a
thread: the mesher is mostly Python, so a thread would hold the GIL and slow
Blender's own Python (panels, the next regrow) for as long as it ran. A process
costs ~0.5 s to start, and an edit that makes its work stale simply kills it.

Run as

    python stem_worker.py <job .pkl> <out .npz> <mesher digest> <parent pid>

It loads `stem_mesher.py` from beside this file by path -- not through the
package, whose `__init__` brings in Blender's side -- and refuses to run when
that file is not the one the job was keyed on (the digest: a mesh must never be
cached under code that did not make it). It applies the parent's module
switches, builds, and writes the result where `bridge`'s on-disk stem cache
reads it; `save` / `load` are that cache's format, shared by both sides. It
exits on its own if Blender does. bpy-free.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import pickle
import sys
import threading
import time

import numpy as np

EXIT_STALE_CODE = 3
"""Exit status when `stem_mesher.py` is not the code the job was keyed on."""


def mesher_digest(path: str) -> str:
    """The digest a stem mesh is keyed on for its code (`bridge._source_digest`)."""
    with open(path, "rb") as fh:
        return hashlib.blake2b(fh.read(), digest_size=16).hexdigest()


def pack(verts, edges, attrs: dict, section, kw: dict, switches: dict) -> dict:
    """A job as plain data, so it unpickles in a process that has never
    imported this package: the skeleton exactly as the mesher would be handed
    it in Blender (the same lists, so the same mesh), `SectionSpec`s as dicts."""
    if isinstance(section, dict):
        sec = ("by_level", {int(k): dataclasses.asdict(v) for k, v in section.items()})
    else:
        sec = ("one", dataclasses.asdict(section))
    return {"verts": verts, "edges": edges, "attrs": dict(attrs),
            "section": sec, "kw": dict(kw), "switches": dict(switches)}


def _unpack_section(stem_mesher, sec):
    kind, data = sec
    if kind == "by_level":
        return {k: stem_mesher.SectionSpec(**v) for k, v in data.items()}
    return stem_mesher.SectionSpec(**data)


def save(path: str, out) -> None:
    """Write a stem mesh, atomically (a reader never sees half a file)."""
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, positions=out.positions, face_sizes=out.face_sizes,
             face_loops=out.face_loops, corner_uv=out.corner_uv,
             counts=np.array([out.junctions, out.fallbacks]),
             **{"attr_" + k: np.asarray(v) for k, v in out.point_attrs.items()})
    os.replace(tmp, path)


def load(stem_mesher, path: str):
    """The stem mesh `save` wrote, as the parent's `stem_mesher.MeshOut`."""
    with np.load(path, allow_pickle=False) as z:
        return stem_mesher.MeshOut(
            positions=z["positions"], face_sizes=z["face_sizes"],
            face_loops=z["face_loops"], corner_uv=z["corner_uv"],
            point_attrs={k[5:]: z[k] for k in z.files if k.startswith("attr_")},
            junctions=int(z["counts"][0]), fallbacks=int(z["counts"][1]))


def _exit_with_parent(pid: int) -> None:
    """Exit as soon as process `pid` (Blender) has: a mesh nobody will read
    is not worth finishing."""
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


def main(argv) -> int:
    job_path, out_path, digest, parent = argv[1], argv[2], argv[3], int(argv[4])
    _exit_with_parent(parent)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stem_mesher.py")
    if mesher_digest(src) != digest:
        return EXIT_STALE_CODE
    spec = importlib.util.spec_from_file_location("_plantgen_stem_mesher", src)
    stem_mesher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stem_mesher        # (dataclasses look their module up)
    spec.loader.exec_module(stem_mesher)
    with open(job_path, "rb") as fh:
        job = pickle.load(fh)
    for name, value in job["switches"].items():
        setattr(stem_mesher, name, value)
    out = stem_mesher.build_stems(job["verts"], job["edges"], job["attrs"],
                                  section=_unpack_section(stem_mesher, job["section"]),
                                  **job["kw"])
    save(out_path, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
