"""Build the installable add-on: plant_gen.zip at the repo root.

Usage:  python tools/build_zip.py

Install it with Edit > Preferences > Add-ons > Install from Disk. What a user's
Blender needs and nothing else: the code, presets and icons, the built manual
(`docs/plantgen-docs.html`) and the spec the panels read (`docs/llm/`). Left out:
the tests, except `tests/structural.py`, which the model and Run Checks import;
the plans and manual sources; caches. Same files in the same order give the same
zip, byte for byte.
"""

import ast
import fnmatch
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "plant_gen"
OUT = REPO / "plant_gen.zip"
STAMP = (2026, 1, 1, 0, 0, 0)
"""Every entry's date: the zip then depends on the files alone."""
KEEP = ("tests/__init__.py", "tests/structural.py",
        "docs/plantgen-docs.html", "docs/llm/*")
"""Kept from the folders SKIP drops."""
SKIP_DIRS = ("tests", "docs")
CACHES = ("__pycache__",)
SKIP_FILES = ("*.pyc", "*.pyo", "*.blend1", "*.blend2", "*.blend@")
REQUIRED = ("plant_gen/__init__.py", "plant_gen/LICENSE", "plant_gen/compute_worker.py",
            "plant_gen/geometry/stem_worker.py", "plant_gen/tests/structural.py",
            "plant_gen/docs/plantgen-docs.html", "plant_gen/docs/llm/plantgen-spec.json")
"""Reached by path at run time, so a missing one fails on the user's machine, late."""


def version() -> str:
    tree = ast.parse((ADDON / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "bl_info":
            return ".".join(str(v) for v in ast.literal_eval(node.value)["version"])
    raise SystemExit("no bl_info")


def wanted(rel: Path) -> bool:
    if any(part in CACHES for part in rel.parts):
        return False
    if any(fnmatch.fnmatch(rel.name, pattern) for pattern in SKIP_FILES):
        return False
    if rel.parts[0] in SKIP_DIRS:
        return any(fnmatch.fnmatch(rel.as_posix(), pattern) for pattern in KEEP)
    return True


def add(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def main() -> int:
    files = sorted(p for p in ADDON.rglob("*") if p.is_file() and wanted(p.relative_to(ADDON)))
    with zipfile.ZipFile(OUT, "w") as archive:
        for path in files:
            add(archive, f"plant_gen/{path.relative_to(ADDON).as_posix()}", path.read_bytes())
        add(archive, "plant_gen/LICENSE", (REPO / "LICENSE").read_bytes())
    with zipfile.ZipFile(OUT) as archive:
        names = set(archive.namelist())
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        OUT.unlink()
        print(f"missing from the zip: {missing}", file=sys.stderr)
        return 1
    size = OUT.stat().st_size / 2 ** 20
    print(f"{OUT.name}: PlantGen {version()}, {len(names)} files, {size:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
