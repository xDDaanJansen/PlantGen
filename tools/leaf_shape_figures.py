"""Draw the manual's leaf-shape figures: plant_gen/docs/figures/leafshape-<key>.svg,
one per built-in Leaf Generator shape (`growth.leaf.LEAF_SHAPES`).

    python tools/leaf_shape_figures.py

Each outline is read off the blade `generate_leaf` builds for that shape -- flat,
smooth margin, the shape's own proportion (width = length * aspect) -- so a
figure is the generator's output, not a redrawing of it. Every leaf is drawn at
the same length, base at the bottom, so the shapes compare at a glance. The
reference (`tools/ui_reference.py`) lays them out under Leaf Generator > Leaf Shape,
labelled by each file's <title>. Plain Python + numpy, no Blender.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "plant_gen" / "docs" / "figures"
sys.path.insert(0, str(REPO))

from plant_gen.growth import leaf as leaf_engine  # noqa: E402

BOX = 100.0          # viewBox side; the blade runs PAD..BOX-PAD vertically
PAD = 6.0
STATIONS = 160       # rows along the blade: fine enough for a cordate notch


def outline(key: str) -> list[tuple[float, float]]:
    """The flat blade's outline in leaf units (x across, y base -> tip, length
    1): up the right edge, back down the left."""
    shape = leaf_engine.LEAF_SHAPES[key]
    blade = leaf_engine.generate_leaf(length=1.0, width=shape.aspect, shape=key,
                                      stations=STATIONS, res_h=1)
    rows: dict[float, list[float]] = {}
    for x, y, _z in blade.verts:
        rows.setdefault(round(float(y), 9), []).append(float(x))
    right = [(max(xs), y) for y, xs in sorted(rows.items())]
    left = [(min(xs), y) for y, xs in sorted(rows.items(), reverse=True)]
    return right + left


def svg(key: str) -> str:
    """One shape as a standalone SVG: the blade as a plain silhouette, no
    outline or midrib (user decision 2026-10-18), filled in currentColor (the
    manual's CSS fills it in the theme's green)."""
    label = leaf_engine.LEAF_SHAPES[key].label
    note = leaf_engine.LEAF_SHAPE_NOTES.get(key, "")
    scale = BOX - 2 * PAD
    pts = outline(key)
    blade = "M" + " L".join(f"{BOX / 2 + x * scale:.1f} {BOX - PAD - y * scale:.1f}"
                            for x, y in pts) + " Z"
    return (f'<svg viewBox="0 0 {BOX:g} {BOX:g}" xmlns="http://www.w3.org/2000/svg">'
            f"<title>{label}</title>"
            + (f"<desc>{note}</desc>" if note else "") +
            f'<path class="blade" d="{blade}" fill="currentColor"/>'
            "</svg>\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("leafshape-*.svg"):      # a shape since removed
        old.unlink()
    for key in sorted(leaf_engine.LEAF_SHAPES):
        (OUT_DIR / f"leafshape-{key}.svg").write_text(svg(key), encoding="utf-8")
    print(f"wrote {len(leaf_engine.LEAF_SHAPES)} leaf shapes to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
