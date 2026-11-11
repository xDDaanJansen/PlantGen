"""Draw the manual's leaf margin and compound figures:
plant_gen/docs/figures/leafmargin-<key>.svg, one per Leaf Generator margin
(`growth.leaf.LEAF_MARGINS`), and leafcompound-<key>.svg, one per compound type
(`growth.leaf.COMPOUND_TYPES`).

    python tools/leaf_margin_compound_figures.py

The sibling of `leaf_shape_figures.py`, drawn the same way: each figure is the
outline of the flat mesh the generator builds (no deformers), read off the
mesh's boundary edges, so it is the generator's output and not a redrawing.
Margins share one ovate blade at the default Teeth and Depth. Compound types
keep the panel's default layout (leaflet count, pairs, Spread, Rachis Length)
with that blade as the leaflet, drawn shorter than the default Leaflet Length
(`LEAFLET`) so neighbouring leaflets don't pile up and the layout reads. The reference
(`tools/ui_reference.py`) puts each figure beside its value under Leaf
Generator > Margin and > Compound. Plain Python + numpy, no Blender.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "plant_gen" / "docs" / "figures"
sys.path.insert(0, str(REPO))

from plant_gen.growth import leaf as leaf_engine  # noqa: E402

BOX = 100.0          # viewBox side; the drawing is fitted inside PAD..BOX-PAD
PAD = 6.0
SHAPE = "ovate"      # the one blade every margin is drawn on
STATIONS = 240       # rows along the blade: resolves the finest (serrulate) teeth
# Leaflet length per compound type (the panel default is 0.12, which on the
# default rachis overlaps its neighbours into a blot).
LEAFLET = {"simple": leaf_engine.DEFAULT_LEAF_LENGTH,
           "trifoliate": leaf_engine.DEFAULT_LEAF_LENGTH,
           "palmate": 0.06, "pinnate": 0.045, "bipinnate": 0.016}


def _blade(stations: int = STATIONS,
           length: float = leaf_engine.DEFAULT_LEAF_LENGTH) -> dict:
    """`generate_leaf` keywords for the figures' blade: the shape's own
    proportion, flat."""
    return {"length": length, "width": length * leaf_engine.LEAF_SHAPES[SHAPE].aspect,
            "shape": SHAPE, "stations": stations, "res_h": 2}


def boundary_loops(leaf) -> list[list[tuple[float, float]]]:
    """The mesh's outline as closed XY loops, each counter-clockwise: the
    edges only one face uses, chained end to end."""
    count: dict[tuple[int, int], int] = {}
    for face in leaf.faces:
        face = tuple(face)
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            k = (min(a, b), max(a, b))
            count[k] = count.get(k, 0) + 1
    nbrs: dict[int, list[int]] = {}
    for (a, b), c in count.items():
        if c == 1:
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)
    loops, seen = [], set()
    for start in nbrs:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = next((n for n in nbrs[cur] if n != prev and n not in seen), None)
            if nxt is None:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        pts = [(float(leaf.verts[i][0]), float(leaf.verts[i][1])) for i in loop]
        area = sum(x0 * y1 - x1 * y0
                   for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True))
        loops.append(pts if area > 0 else pts[::-1])
    return loops


def svg(label: str, leaf) -> str:
    """The leaf as a standalone SVG fitted to the box, base at the bottom: a
    plain silhouette, every outline loop in one filled path (overlaps fill
    once), no outline or midrib (user decision 2026-10-18). Filled in
    currentColor; the manual's CSS fills it in the theme's green."""
    loops = boundary_loops(leaf)
    xs = [x for lp in loops for x, _y in lp]
    ys = [y for lp in loops for _x, y in lp]
    cx, y0 = (min(xs) + max(xs)) / 2, min(ys)
    span = max(max(xs) - min(xs), max(ys) - y0)
    scale = (BOX - 2 * PAD) / span
    lift = (BOX - 2 * PAD - (max(ys) - y0) * scale) / 2    # centre a wide leaf

    def pt(x, y):
        return f"{BOX / 2 + (x - cx) * scale:.1f} {BOX - PAD - lift - (y - y0) * scale:.1f}"

    d = " ".join("M" + " L".join(pt(x, y) for x, y in lp) + " Z" for lp in loops)
    return (f'<svg viewBox="0 0 {BOX:g} {BOX:g}" xmlns="http://www.w3.org/2000/svg">'
            f"<title>{label}</title>"
            f'<path class="blade" d="{d}" fill="currentColor"/>'
            "</svg>\n")


def margins() -> dict[str, str]:
    return {key: svg(key.capitalize(), leaf_engine.generate_leaf(margin=key, **_blade()))
            for key in leaf_engine.LEAF_MARGINS}


def compounds() -> dict[str, str]:
    return {key: svg(key.capitalize(),
                     leaf_engine.generate_compound_leaf(
                         compound=key, rachis_length=leaf_engine.default_rachis_length(key),
                         **_blade(60, LEAFLET[key])))
            for key in leaf_engine.COMPOUND_TYPES}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, figs in (("leafmargin", margins()), ("leafcompound", compounds())):
        for old in OUT_DIR.glob(f"{prefix}-*.svg"):     # a value since removed
            old.unlink()
        for key, text in figs.items():
            (OUT_DIR / f"{prefix}-{key}.svg").write_text(text, encoding="utf-8")
        print(f"wrote {len(figs)} {prefix} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
