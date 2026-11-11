"""The UI source digest the manual's reference is stamped with -- bpy-free, so
the generator (`blender/ui_spec.py`, in Blender) and the docs build
(`tools/build_docs.py`, plain Python) compute the same value. A reference whose
stamp no longer matches the files was generated from older panels: stale."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCE_FILES = ("blender/ui.py", "blender/ui_traits.py", "blender/ui_materials.py",
                "blender/ui_flower.py", "blender/ui_leaf.py", "blender/stem_material.py",
                "blender/flower_material.py", "blender/properties.py",
                "blender/flower_props.py", "blender/operators.py", "blender/flower_ops.py",
                "blender/effectors.py", "blender/mapping.py", "blender/ui_spec.py")
"""What the reference is made from: the draw code, the properties (labels,
tooltips, ranges), the operators (button labels) and the generator itself."""


def source_digest(root: Path = ROOT) -> str:
    """Same files, same order, newline-normalised (a checkout's line endings
    must not make the reference look stale)."""
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        p = root / rel
        h.update(rel.encode())
        if p.is_file():
            h.update(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return h.hexdigest()[:16]
