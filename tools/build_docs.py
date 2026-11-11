"""Build the self-contained artist manual HTML.

Usage:  python tools/build_docs.py

Reads:
  * plant_gen/docs/ui-reference.json       -- every panel's controls, generated in
                                              Blender from the panels themselves
                                              (`blender/ui_spec.py`); refused when
                                              stale (`tools/ui_reference.py`)
  * plant_gen/docs/ui-notes.json           -- hand-written notes per control
  * plant_gen/docs/llm/plantgen-spec.json  -- the inflorescence kinds and the
                                              curve shapes (the model, not a panel)
  * plant_gen/docs/pages/*.md              -- hand-written Diataxis prose
                                              (tutorial / how-to / explanation)
Writes:
  * plant_gen/docs/plantgen-docs.html      -- one offline file: ToC sidebar,
                                              client-side search, theme toggle.

The reference is generated from the panels so it cannot drift from the UI; only
the prose and the notes are hand-written. No Blender, no external toolchain, no
network -- everything is inlined, so the file works opened straight off disk
(Plant Generator's help button does exactly that via file://).
"""

import base64
import hashlib
import html
import json
import re
from pathlib import Path

import ui_reference
from _docs_assets import CSS, JS, TEMPLATE

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "plant_gen" / "docs"
SPEC_PATH = DOCS / "llm" / "plantgen-spec.json"
PAGES_DIR = DOCS / "pages"
OUT_PATH = DOCS / "plantgen-docs.html"
IMAGES_DIR = DOCS / "images"
SHOTS_PATH = DOCS / "shots.md"
SHOTS_SEEN = IMAGES_DIR / "shots-seen.json"
FIGURES_DIR = DOCS / "figures"


def _figure_svg(name: str) -> str:
    """The inline <svg> for docs/figures/<name>.svg, or '' if absent -- a missing
    figure never breaks the build. Inlined (not <img src>) so the one HTML file
    stays self-contained and the figure adapts to the light/dark theme via
    currentColor."""
    f = FIGURES_DIR / (name + ".svg")
    if not f.is_file():
        return ""
    return re.sub(r"<\?xml.*?\?>", "", f.read_text(encoding="utf-8"), flags=re.S).strip()


def _icon(name: str) -> str:
    """A small inline UI-icon glyph (docs/figures/icon-<name>.svg)."""
    svg = _figure_svg("icon-" + name)
    return f'<span class="ui-icon">{svg}</span>' if svg else ""


# --- minimal markdown (subset) for the prose pages --------------------------

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![*\w])\*([^*\s][^*]*?)\*(?![*\w])"), r"<em>\1</em>"),
    (re.compile(r"`([^`]+?)`"), r"<code>\1</code>"),
    (re.compile(r"\[([^\]]+?)\]\(([^)]+?)\)"), r'<a href="\2">\1</a>'),
]


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    for pat, repl in _INLINE:
        text = pat.sub(repl, text)
    text = re.sub(r"\{icon:([a-z0-9_-]+)\}", lambda m: _icon(m.group(1)), text)
    return text


def _figure(name: str, caption: str) -> str:
    """`![caption](name)`: docs/images/<name>.png embedded (a render, r-*, or
    the user's screenshot), else a drawn SVG figure, else a labelled
    placeholder -- a missing image never breaks the build."""
    cap = f"<figcaption>{_inline(caption)}</figcaption>" if caption else ""
    path = next((p for p in (IMAGES_DIR / f"{name}.png", IMAGES_DIR / f"{name}.jpg")
                 if p.is_file()), None)
    if path is not None:
        cls = "render" if name.startswith("r-") else "shot"
        alt = html.escape(caption)
        if name in EMBEDDED_IMAGES:          # embedded once; later uses copy it (JS)
            return (f'<figure class="fig fig-img"><img class="{cls}" alt="{alt}" '
                    f'data-same="img-{name}">{cap}</figure>')
        EMBEDDED_IMAGES.add(name)
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return (f'<figure class="fig fig-img"><img id="img-{name}" class="{cls}" alt="{alt}" '
                f'src="data:{mime};base64,{data}">{cap}</figure>')
    svg = _figure_svg(name)
    if svg:
        return f'<figure class="fig">{svg}{cap}</figure>'
    MISSING_IMAGES.append(name)
    what = "Render" if name.startswith("r-") else "Screenshot"
    return (f'<figure class="shot-missing"><strong>{what} to come:</strong> '
            f'<code>{name}.png</code>{cap}</figure>')


MISSING_IMAGES: list[str] = []
EMBEDDED_IMAGES: set[str] = set()


_TABLE_RULE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_ITEM = re.compile(r"^(?:[-*]|\d+\.) ")


def _md_list(lines: list[str], i: int) -> tuple[str, int]:
    """One - or 1. list from line `i`: an item runs on over the indented lines
    after it, and an indented list inside an item is rendered nested."""
    ordered = bool(re.match(r"\d+\. ", lines[i]))
    items = []
    while i < len(lines) and _ITEM.match(lines[i]) and \
            bool(re.match(r"\d+\. ", lines[i])) == ordered:
        body = [_ITEM.sub("", lines[i], count=1)]
        i += 1
        while i < len(lines) and lines[i].startswith(" ") and lines[i].strip():
            body.append(lines[i])
            i += 1
        text, rest = [body[0]], body[1:]
        while rest and not _ITEM.match(rest[0].strip()):
            text.append(rest.pop(0).strip())
        nested = ""
        if rest:
            indent = len(rest[0]) - len(rest[0].lstrip())
            nested = md_to_html("\n".join(r[indent:] if r[:indent].isspace() else r.strip()
                                          for r in rest))
        items.append(f"<li>{_inline(' '.join(text))}{nested}</li>")
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(items) + f"</{tag}>", i


def md_to_html(text: str) -> str:
    """A deliberately small Markdown subset: ## / ### headings, paragraphs,
    - and 1. lists (items run on over indented lines; an indented list nests),
    | pipe | tables |, > blockquotes, ``` code fences, and inline **bold**,
    `code`, [links](url). Enough for tutorial / how-to / explanation prose."""
    out, i, lines = [], 0, text.splitlines()
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue
        img = re.match(r"!\[(.*?)\]\(([a-z0-9_-]+)\)\s*$", line)
        if img:
            out.append(_figure(img.group(2), img.group(1)))
            i += 1
            continue
        if line.startswith("### "):
            out.append(f"<h4 class='sub'>{_inline(line[4:])}</h4>")
        elif line.startswith("## "):
            out.append(f"<h3 class='sub'>{_inline(line[3:])}</h3>")
        elif line.startswith("|") and i + 1 < len(lines) and _TABLE_RULE.match(lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            out.append("<div class='table-scroll md'><table class='md'><thead><tr>"
                       + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                                 for r in body)
                       + "</tbody></table></div>")
            continue
        elif line.startswith(("- ", "* ")) or re.match(r"\d+\. ", line):
            html_list, i = _md_list(lines, i)
            out.append(html_list)
            continue
        elif line.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].startswith("> "):
                quote.append(_inline(lines[i][2:]))
                i += 1
            out.append("<blockquote>" + "<br>".join(quote) + "</blockquote>")
            continue
        elif line.strip():
            para = [line]
            i += 1
            # a list item ends a paragraph ("- " / "* "); a line opening on
            # bold ("**Replace Me**, ...") or a minus sign does not
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(
                    ("#", "- ", "* ", ">", "```", "|", "![")) and not re.match(r"\d+\. ", lines[i]):
                para.append(lines[i])
                i += 1
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            continue
        i += 1
    return "\n".join(out)




def _catmull_rom(pts) -> str:
    """A smooth SVG path (Catmull-Rom -> cubic bezier) through pts, so a curve
    the widget draws smoothly is not shown as faceted segments."""
    d = f"M{pts[0][0]},{pts[0][1]}"
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
        c1 = (round(p1[0] + (p2[0] - p0[0]) / 6, 2), round(p1[1] + (p2[1] - p0[1]) / 6, 2))
        c2 = (round(p2[0] - (p3[0] - p1[0]) / 6, 2), round(p2[1] - (p3[1] - p1[1]) / 6, 2))
        d += f" C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {p2[0]},{p2[1]}"
    return d


def _curve_svg(points, smooth=False) -> str:
    """A mini plot of a curve shape, from its own points (x,y in 0..1) -- so it
    always matches the real shape. Smooth shapes get a curved path, the rest a
    polyline."""
    pts = [(round(5 + x * 60, 2), round(37 - y * 34, 2)) for x, y in points]
    if smooth and len(pts) >= 3:
        shape = f'<path d="{_catmull_rom(pts)}"/>'
    else:
        shape = '<polyline points="' + " ".join(f"{x},{y}" for x, y in pts) + '"/>'
    return ('<svg viewBox="0 0 70 42" xmlns="http://www.w3.org/2000/svg" '
            f'class="curve-svg">{shape}</svg>')


def render_curve_shapes(spec: dict, toc: list) -> str:
    toc.append((2, "curves", "Curve shapes", "ref"))
    cells = "".join(
        f'<div class="shape-cell">{_curve_svg(s["points"], s.get("smooth"))}'
        f'<div class="shape-key"><code>{html.escape(s["key"])}</code></div>'
        f'<div class="shape-label">{html.escape(s["label"])}</div></div>'
        for s in spec["curve_shapes"])
    note = html.escape(spec.get("curves", ""))
    return (f'<section id="curves"><div class="sec-head"><h2>Curve shapes</h2>'
            f'<span class="sec-kicker">named shapes for every curve field</span>'
            f'</div><p class="sec-intro">{note}</p>'
            f'<div class="shape-grid">{cells}</div></section>')


GLOSSARY = {
    "Acropetal": "Developing or opening from the base upward.",
    "Adventitious": "Arising in an unusual place, from tissue that carried no bud.",
    "Apex": "The growing tip of a stem or axis.",
    "Axil": "The angle between a leaf and the stem above it.",
    "Axillary": "Arising in a leaf axil.",
    "Basipetal": "Developing or opening from the tip downward.",
    "Bolting": "The rapid elongation of a stem into flower, especially from a rosette.",
    "Bract": "A reduced or modified leaf, often at the base of a flower or inflorescence.",
    "Cauliflorous": "Flowering directly from the trunk or main stem, as cacao does.",
    "Corymb": "A flat-topped inflorescence whose lower flowers have longer stalks.",
    "Cyme": "An inflorescence whose axis ends in a flower, further flowers arising below it.",
    "Decussate": "Leaves in opposite pairs, each pair at right angles to the next.",
    "Determinate": "An axis that ends in a flower and stops growing.",
    "Dichotomous": "Branching by the growing tip forking into two equal branches.",
    "Distichous": "Leaves in two opposite ranks, all in one plane.",
    "Indeterminate": "An axis that keeps growing, flowering along its length.",
    "Inflorescence": "The flower-bearing part of a plant, and the arrangement of its flowers.",
    "Internode": "The length of stem between two nodes.",
    "Lignified": "Hardened with woody tissue.",
    "Meristem": "A region of dividing cells from which new growth arises.",
    "Monopodial": "Growth from a single persistent main axis.",
    "Node": "A point on a stem where leaves, buds or branches arise.",
    "Panicle": "A branched inflorescence, a raceme of racemes.",
    "Pedicel": "The stalk of a single flower.",
    "Peduncle": "The stalk of an inflorescence, or of a solitary flower.",
    "Pendulous": "Hanging downward.",
    "Phyllotaxis": "The arrangement of leaves around a stem.",
    "Pseudomonopodial": "A false single axis, built from a series of dominant forks.",
    "Raceme": "An unbranched inflorescence of stalked flowers along a continuing axis.",
    "Rachis": "The main axis of an inflorescence or a compound leaf.",
    "Ramiflorous": "Flowering from old wood along the branches rather than the trunk.",
    "Rosette": "A circular cluster of leaves at the base of a plant.",
    "Scape": "A leafless flowering stalk rising from the ground.",
    "Secund": "With flowers or leaves all turned to one side.",
    "Sessile": "Without a stalk, attached directly.",
    "Spike": "An unbranched inflorescence of sessile flowers.",
    "Sympodial": "Growth continued by successive lateral branches, the tip stopping each time.",
    "Terminal": "At the tip of a stem or axis.",
    "Tiller": "A shoot arising from the base of a grass or similar plant.",
    "Umbel": "An inflorescence whose flower stalks all arise from one point.",
    "Vegetativeness": "How strongly a meristem keeps making stem rather than flowers.",
    "Whorl": "Three or more leaves or flowers arising at one node.",
}


def _gloss_id(term: str) -> str:
    return "gloss-" + term.lower().replace(" ", "-")


def render_glossary(toc: list) -> str:
    toc.append((1, "glossary", "Glossary", "ref"))
    items = "".join(
        f'<div class="gloss-item" id="{_gloss_id(t)}"><dt>{html.escape(t)}</dt>'
        f'<dd>{_inline(d)}</dd></div>'
        for t, d in sorted(GLOSSARY.items()))
    return ('<section id="glossary"><div class="sec-head"><h2>Glossary</h2>'
            '<span class="sec-kicker">botanical terms used here</span></div>'
            f'<dl class="gloss">{items}</dl></section>')


# --- prose pages ------------------------------------------------------------

def load_pages():
    """Read docs/pages/*.md in filename order. Each file: an optional
    `<!-- mode: How-to -->` first line (the kicker), then a `# Title` H1, then
    the body. Returns [(mode_id, order_slug, title, mode_label, body_html)]."""
    pages = []
    if not PAGES_DIR.is_dir():
        return pages
    for path in sorted(PAGES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        mode = "Guide"
        m = re.match(r"<!--\s*mode:\s*(.+?)\s*-->", text)
        if m:
            mode = m.group(1)
            text = text[m.end():].lstrip("\n")
        title_m = re.match(r"#\s+(.+)", text)
        title = title_m.group(1).strip() if title_m else path.stem
        body = text[title_m.end():] if title_m else text
        pages.append((path.stem, title, mode, md_to_html(body.strip())))
    return _link_titles(pages)


def _link_titles(pages):
    """An italic page title in prose (*Make a hanging raceme*) becomes a link
    to that page. Any other italic is listed, so a renamed or misspelt title
    shows up at build time instead of as dead text."""
    by_title = {title.lower(): stem for stem, title, _mode, _body in pages}
    loose = []

    def link(m):
        stem = by_title.get(re.sub(r"\s+", " ", m.group(1)).lower())
        if stem is None:
            loose.append(m.group(1))
            return m.group(0)
        return f'<a href="#page-{stem}"><em>{m.group(1)}</em></a>'

    out = [(stem, title, mode, re.sub(r"<em>(.+?)</em>", link, body))
           for stem, title, mode, body in pages]
    for text in loose:
        print(f"  italic, not a page title: {text!r}")
    return out


MODE_ORDER = {"Tutorial": 0, "How-to": 1, "Explanation": 2, "Guide": 3}


def render_pages(pages, toc):
    if not pages:
        return ""
    # group by mode, in Diataxis order
    groups = {}
    for stem, title, mode, body in pages:
        groups.setdefault(mode, []).append((stem, title, body))
    out = []
    for mode in sorted(groups, key=lambda m: MODE_ORDER.get(m, 9)):
        gid = "guide-" + re.sub(r"[^a-z]+", "-", mode.lower())
        heading = {"Tutorial": "Tutorial", "How-to": "How-to guides",
                   "Explanation": "Explanation"}.get(mode, mode)
        # Point the group heading at its first page (there is no element with
        # id=gid), so clicking it in the ToC scrolls somewhere.
        toc.append((1, "page-" + groups[mode][0][0], heading, "guide"))
        for stem, title, body in groups[mode]:
            pid = "page-" + stem
            toc.append((2, pid, title, "guide"))
            out.append(f'<section id="{pid}" data-firstmode="'
                       f'{gid if groups[mode][0][0] == stem else ""}">'
                       f'<div class="sec-head"><h2>{html.escape(title)}</h2>'
                       f'<span class="sec-kicker">{html.escape(mode)}</span></div>'
                       f'{body}</section>')
    return "\n".join(out)


# --- search index -----------------------------------------------------------

def search_index(spec: dict, pages, ui_index) -> list:
    idx = []
    for stem, title, mode, _body in pages:
        idx.append({"id": "page-" + stem, "label": title,
                    "sub": mode, "kind": "guide"})
    # every control, under the panel path it sits in (its choices' names too, so
    # searching a visible option word surfaces the control that has it)
    idx.extend(ui_index)
    for term in GLOSSARY:
        idx.append({"id": _gloss_id(term), "label": term,
                    "sub": "glossary term", "kind": "term"})
    return idx


# --- the shell --------------------------------------------------------------

def render_toc(toc):
    out = ['<nav class="toc"><div class="toc-inner">']
    for level, tid, title, group in toc:
        out.append(f'<a class="toc-{level} toc-{group}" href="#{tid}" '
                   f'data-target="{tid}">{html.escape(title)}</a>')
    out.append("</div></nav>")
    return "\n".join(out)


def shot_list() -> list[tuple[str, str]]:
    """(shot, panel) for every row of docs/shots.md's table."""
    rows = []
    if SHOTS_PATH.is_file():
        for line in SHOTS_PATH.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if (line.startswith("|") and len(cells) >= 2
                    and re.fullmatch(r"[a-z][a-z0-9-]*", cells[0])):
                rows.append((cells[0], cells[1]))
    return rows


def check_shots(ui: dict) -> list[str]:
    """Which screenshots show a panel whose controls changed since the shot
    was taken. A shot's panel is fingerprinted the first time the build sees
    the image (or a newer copy of it); a later build that finds the panel's
    fingerprint moved reports the shot as stale until it is retaken."""
    panels = {p["title"]: hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
              for p in ui["panels"]}
    seen = json.loads(SHOTS_SEEN.read_text(encoding="utf-8")) if SHOTS_SEEN.is_file() else {}
    stale, changed = [], False
    for shot, panel in shot_list():
        png = IMAGES_DIR / f"{shot}.png"
        if panel not in panels:
            print(f"  shots.md: {shot} names no panel called {panel!r}")
            continue
        if not png.is_file():
            continue
        mtime = int(png.stat().st_mtime)
        rec = seen.get(shot)
        if rec is None or rec.get("mtime") != mtime:
            seen[shot] = {"mtime": mtime, "panel": panel, "hash": panels[panel]}
            changed = True
        elif rec.get("hash") != panels[panel]:
            stale.append(shot)
    if changed:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        SHOTS_SEEN.write_text(json.dumps(seen, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
    return stale


def build():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    ui = ui_reference.load()          # raises StaleReference on old panels
    pages = load_pages()
    toc = []
    pages_html = render_pages(pages, toc)
    ref_html, ui_index, n_controls = ui_reference.render(ui, toc, figure=_figure)
    ref_html += render_curve_shapes(spec, toc)
    ref_html += render_glossary(toc)
    toc_html = render_toc(toc)
    index = json.dumps(search_index(spec, pages, ui_index))

    doc = TEMPLATE.format(
        css=CSS, js=JS.replace("__INDEX__", index),
        toc=toc_html, pages=pages_html, reference=ref_html,
        n_fields=n_controls, n_kinds=len(spec["inflorescence_kinds"]))
    OUT_PATH.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")
    print(f"  {len(pages)} prose pages, {n_controls} controls in "
          f"{len(ui['panels'])} panels")
    shots = {s for s, _p in shot_list()}
    missing = sorted(set(MISSING_IMAGES))
    if missing:
        print(f"  {len(missing)} images still to come: " + ", ".join(missing))
    unknown = [m for m in missing if not m.startswith("r-") and m not in shots]
    if unknown:
        print("  not in shots.md: " + ", ".join(unknown))
    for shot in check_shots(ui):
        print(f"  stale screenshot: {shot} (its panel changed since it was taken; "
              f"python tools/panel_shots.py {shot})")
    return OUT_PATH


if __name__ == "__main__":
    build()
