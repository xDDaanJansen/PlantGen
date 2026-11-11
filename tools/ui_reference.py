"""Render the manual's panel reference from `plant_gen/docs/ui-reference.json`
(written in Blender by `plant_gen/blender/ui_spec.py`). Plain Python, no bpy.

Each panel is a section; each fold and sub-panel a heading under it, in the
order the N-panel draws them; each control an entry in the Blender manual's
term -> description form: its label, what kind of control it
is, default / range / unit, its values, its tooltip, and the hand-written note
for it in `ui-notes.json` when there is one. A control a state reveals carries
"Shown for ...". The reference is refused when its source digest no longer
matches the UI source files (`plant_gen/docs_digest.py`): regenerate it in
Blender, never edit it."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "plant_gen" / "docs"
UI_PATH = DOCS / "ui-reference.json"
NOTES_PATH = DOCS / "ui-notes.json"
FIGURES_DIR = DOCS / "figures"

# A choice control whose values each have a drawn figure, docs/figures/<prefix>-<value>.svg
# (property id -> prefix). Leaf Generator's Shape lists its values at draw time -- the
# built-ins and the user's own folder -- so the reference records none: its built-in
# shapes are listed from their figures (`tools/leaf_shape_figures.py`).
_VALUE_FIGURES = {"gen_shape": "leafshape", "gen_margin": "leafmargin",
                  "gen_compound": "leafcompound", "inflorescence_kind": "kind"}

# A flower component's Type: what each kind makes inert, from the generated
# spec (`docs/llm/plantgen-spec.json`); a COMPOUND_<kind> value is that kind
# compounded, drawn as docs/figures/compound-<kind>.svg.
SPEC_PATH = DOCS / "llm" / "plantgen-spec.json"
_KIND_NOTE = ("A control listed as inert on a kind does nothing there. A kind with no "
              "pedicels ignores the pedicel controls; a terminal cluster ignores the "
              "flowering-gradient curves; only flat-topped kinds use the head controls; "
              "only relayed kinds use relay rotation. Why it works this way: "
              '<a href="#page-45-inflorescence-types">Inflorescence types</a>.')

# A choice whose values each have a small diagram drawn beside the value's own
# row (docs/figures/<prefix>-<value>.svg, the value's identifier in lower case;
# a value with no file simply has none).
_ITEM_FIGURES = {"growth_pattern": "arch", "inflorescence_attachments": "attach",
                 "phyllotaxis": "phyllo", "gen_phyllotaxis": "phyllo",
                 "cross_section": "xsec", "inflorescence_orientation": "orient",
                 "bloom_order": "bloom", "kind": "effector"}

# Panel screenshots (docs/shots.md) shown at the top of what they show: the
# reference path of a panel, fold or sub-panel -> (shot, caption).
REFERENCE_SHOTS = {
    "Flower Generator": [("flower-generator", "A flower selected, every sub-panel closed")],
    "Flower Generator › Bloom": [("flower-bloom", "Bloom open")],
    "Flower Generator › Petals": [
        ("flower-petals", "Petals open, Outline and Placement unfolded")],
    "Flower Generator › Petals › Layers": [
        ("flower-petal-layers", "Petals' Mesh fold and Layers open")],
    "Flower Generator › Petals › Deform": [
        ("flower-petal-deform", "Deform open, with Curl and Rotation")],
    "Flower Generator › Petals › Bud Shape": [("flower-petal-bud", "Bud Shape open")],
    "Flower Generator › Petals › Corolla Tube": [
        ("flower-corolla-tube", "Corolla Tube open and on")],
    "Flower Generator › Sepals": [
        ("flower-sepals", "Sepals open and on; its Layers, Deform, Bud Shape and Calyx "
                          "Tube sub-panels match the petals'")],
    "Flower Generator › Disc": [
        ("flower-disc-receptacle", "Disc and Receptacle open, Receptacle on")],
    "Flower Generator › Material": [
        ("flower-material", "Material open with a flower material, Textures & Settings "
                            "unfolded")],
    "Leaf Generator": [("leaf-generator", "Generate mode, every fold closed"),
                       ("leaf-cutout", "Cutout mode with an atlas image picked")],
    "Leaf Generator › Profile Curve": [
        ("leaf-profile-margin", "Profile Curve and Margin open, Margin set to Serrate")],
    "Leaf Generator › Deform": [("leaf-deform", "Deform open")],
    "Leaf Generator › Compound": [("leaf-compound", "Compound open, set to Pinnate")],
    "Leaf Generator › Material": [
        ("leaf-edit", "A generated leaf selected: the panel edits it, Material open")],
}

_TAG_NAMES = {
    "component": ("components", {"leaf": "Leaf", "bract": "Bract", "flower": "Flower",
                                 "bud": "Bud", "fruit": "Fruit"}),
    "level": ("levels", {"stems": "the stems level", "branches": "branch levels"}),
    "map": ("maps", {"var": "Random Range", "curve": "profile curves",
                     "imap": "instance maps", "pmap": "per-plant maps",
                     "lmap": "level maps", "gmap": "growth maps", "fmap": "inflorescence maps"}),
    "texture": ("rows", {"level": "level rows", "material": "leaf / flower material rows"}),
    "leaf": ("mode", {"generate": "Generate mode", "cutout": "Cutout mode"}),
    "effector": ("kinds", {"attract": "Attract", "repel": "Repel", "forbid": "Forbid",
                           "target": "Target", "flatten": "Flatten", "orbit": "Orbit",
                           "align": "Align Up", "level": "Level Up"}),
}
_EXTRA_NAMES = {"map": "mappable", "curve": "curve", "variance": "± variance",
                "debug view": "debug view"}
# What each badge says, by family (its colour). A control's own type comes
# first, filled; what it carries beside it, outlined; where it shows, plain.
_BADGES = {
    "number": ("value", "A field or slider holding a decimal value"),
    "whole number": ("value", "A field holding a whole number"),
    "text": ("value", "A text field"),
    "checkbox": ("switch", "On or off"),
    "choice": ("switch", "A dropdown: pick one value"),
    "choices": ("switch", "A row of toggles: pick any number of values"),
    "picker": ("data", "Points at a Blender datablock: an object, collection, material or image"),
    "list": ("data", "Rows you add, remove and select"),
    "curve": ("data", "A curve you draw"),
    "button": ("action", "Runs an operation when clicked"),
    "menu": ("action", "Opens a menu of operations"),
}
_EXTRA_BADGES = {
    "mappable": ("extra", "A map icon beside it: vary it per branch, node or organ "
                          "in Mapping & Classification"),
    "± variance": ("extra", "A ± button beside it: how much it varies from one part to the next"),
    "curve": ("extra", "A curve button beside it: shape the value along the part"),
    "Debug Tools": ("debug", "Shown only with the Debug Tools add-on preference on"),
    "closed by default": ("where", "The fold starts closed in the panel"),
}
_LEGEND = (("value", "Value", ("number", "whole number", "text")),
           ("switch", "Switch or choice", ("checkbox", "choice", "choices")),
           ("data", "Data", ("picker", "list", "curve")),
           ("action", "Action", ("button", "menu")))
_LEGEND_EXTRAS = (("Beside it", ("mappable", "± variance", "curve")),
                  ("Where it shows", ("Debug Tools", "closed by default", "shown for …")))


def _badge(text: str, extra: bool = False) -> str:
    """One badge, coloured by its family, its meaning on hover."""
    table = _EXTRA_BADGES if extra else _BADGES
    if text.startswith("shown for "):
        family, tip = "where", "Shown only for these components, levels or modes"
    else:
        family, tip = table.get(text, ("value", ""))
    title = f' title="{html.escape(tip)}"' if tip else ""
    return f'<span class="badge b-{family}"{title}>{html.escape(text)}</span>'


def legend() -> str:
    """The key to the badges, shown once above the reference."""
    rows = "".join(
        f'<div class="legend-row"><span class="legend-name">{name}</span>'
        + "".join(_badge(b) for b in items) + "</div>"
        for _, name, items in _LEGEND)
    rows += "".join(
        f'<div class="legend-row"><span class="legend-name">{name}</span>'
        + "".join(_badge(b, extra=True) for b in items) + "</div>"
        for name, items in _LEGEND_EXTRAS)
    return f'<div class="badge-legend">{rows}<p>Hover a badge for what it means.</p></div>'


_UNITS = {"length": "m", "rotation": "°", "time": "s", "mass": "kg", "area": "m²",
          "volume": "m³", "velocity": "m/s"}


class StaleReference(RuntimeError):
    pass


def load(check: bool = True) -> dict:
    """The reference, checked against the UI source it was generated from."""
    doc = json.loads(UI_PATH.read_text(encoding="utf-8"))
    if check:
        sys.path.insert(0, str(REPO))
        try:
            from plant_gen import docs_digest
        finally:
            sys.path.pop(0)
        now = docs_digest.source_digest()
        if doc.get("source_digest") != now:
            raise StaleReference(
                f"ui-reference.json was generated from older panels ({doc.get('source_digest')}"
                f" != {now}): regenerate it in Blender (Debug Tools > Documentation > "
                "Regenerate Docs).")
    return doc


def notes() -> dict:
    if NOTES_PATH.is_file():
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    return {}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _num(v) -> str:
    if isinstance(v, float):
        v = round(v, 4)
        if v.is_integer():
            return str(int(v))
    return str(v)


def _unit(e) -> str:
    if e.get("unit") in _UNITS:
        return " " + _UNITS[e["unit"]]
    if e.get("subtype") == "angle":
        return "°"
    if e.get("subtype") == "percentage":
        return "%"
    if e.get("subtype") == "factor":
        return ""
    return ""


def _default(e) -> str | None:
    d = e.get("default")
    if d is None or d == "" or d == []:
        return None
    if e["type"] == "boolean":
        return "on" if d else "off"
    if e["type"] == "enum":
        names = {i[0]: i[1] for i in e.get("items", [])}
        if isinstance(d, list):
            return ", ".join(names.get(x, x) for x in d) or None
        return names.get(d, d)
    if isinstance(d, list):
        return ", ".join(_num(x) for x in d)
    if e.get("subtype") == "angle" and isinstance(d, float):
        return f"{_num(d * 57.29577951308232)}°"
    return _num(d) + _unit(e)


def _range(e) -> str | None:
    if e["type"] not in ("float", "int"):
        return None
    angle = e.get("subtype") == "angle"

    def fmt(v):
        return _num(v * 57.29577951308232 if angle else v) + ("°" if angle else _unit(e))

    lo, hi = e.get("min"), e.get("max")
    # Blender's integer limits are no limit at all: read them as unbounded.
    if lo is not None and lo <= -2147483647:
        lo = None
    if hi is not None and hi >= 2147483647:
        hi = None
    slo, shi = e.get("soft_min"), e.get("soft_max")
    if lo is None and hi is None and slo is None and shi is None:
        return None
    hard = f"{fmt(lo) if lo is not None else '−∞'} to {fmt(hi) if hi is not None else '∞'}"
    if (slo, shi) != (lo, hi) and slo is not None and shi is not None:
        return f"{fmt(slo)} to {fmt(shi)} on the slider ({hard} typed)"
    return hard


def _shown_for(tags, scope) -> str | None:
    """'Shown for Leaf, Bract' when the control appears in some, not all, of
    the states its panel was drawn in."""
    if not tags:
        return None
    parts = []
    for cat, (_word, names) in _TAG_NAMES.items():
        mine = sorted(t.split(":", 1)[1] for t in tags if t.startswith(cat + ":"))
        if not mine:
            continue
        seen = scope.get(cat, set())
        if set(mine) >= seen:
            continue
        order = list(names)
        mine.sort(key=lambda m: order.index(m) if m in order else 99)
        parts.append(", ".join(names.get(m, m) for m in mine))
    return "; ".join(parts) or None


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text


class _Ctx:
    def __init__(self):
        self.ids: dict = {}
        self.index: list = []
        self.notes = notes()
        self.n_controls = 0
        self.figure = None          # (shot, caption) -> HTML, given by the build

    def shots(self, path: str) -> str:
        if self.figure is None:
            return ""
        return "".join(self.figure(shot, cap) for shot, cap in REFERENCE_SHOTS.get(path, ()))

    def uid(self, base: str) -> str:
        base = base or "x"
        n = self.ids.get(base, 0)
        self.ids[base] = n + 1
        return base if n == 0 else f"{base}-{n + 1}"


def _entry(e, ctx: _Ctx, where: str, scope) -> str:
    kind = e["kind"]
    if kind == "heading":
        return f'<h5 class="ref-subhead">{html.escape(e["text"])}</h5>'
    label = e.get("label") or {"list": "List", "curve": "Curve"}.get(kind, kind.title())
    key = f'{e.get("owner", "")}.{e.get("id", "")}'
    rid = ctx.uid("ui-" + slug(where + "-" + label))
    badges = []
    if kind == "prop":
        ctx.n_controls += 1
        t = {"float": "number", "int": "whole number", "boolean": "checkbox",
             "enum": "choice" if not e.get("flag") else "choices", "string": "text",
             "pointer": "picker"}.get(e["type"], e["type"])
        badges.append(t)
    else:
        badges.append({"op": "button", "menu": "menu", "list": "list", "picker": "picker",
                       "curve": "curve"}.get(kind, kind))
    extras = []
    for ex in e.get("extras", ()):
        if ex in _EXTRA_NAMES and ex != "debug view":
            extras.append(_EXTRA_NAMES[ex])
    if e.get("mappable") and "mappable" not in extras:
        extras.append("mappable")
    if e.get("debug"):
        extras.append("Debug Tools")
    head = (f'<div class="ref-head"><span class="ref-label">{html.escape(label)}</span>'
            + "".join(_badge(b) for b in badges)
            + "".join(_badge(b, extra=True) for b in extras)
            + "</div>")
    meta = []
    if kind == "prop":
        d = _default(e)
        if d is not None:
            meta.append(f"default {html.escape(d)}")
        r = _range(e)
        if r:
            meta.append(html.escape(r))
    shown = _shown_for(e.get("tags"), scope)
    if shown:
        meta.append("shown for " + html.escape(shown))
    body = [f'<div class="ref-item" id="{rid}">', head]
    if meta:
        body.append('<div class="ref-meta">' + " &middot; ".join(meta) + "</div>")
    desc = (e.get("desc") or "").strip()
    if desc:
        body.append(f'<p class="ref-desc">{_inline(desc)}</p>')
    note = ctx.notes.get(key) or ctx.notes.get(f"{where}/{label}")
    if note:
        body.append(f'<p class="ref-note">{_inline(note)}</p>')
    figs, fig_names = _value_figures(e, ctx, rid) if kind == "prop" else ("", [])
    if figs:
        body.append(figs)                       # the values as figure cards instead
    elif kind == "prop" and e["type"] == "enum" and e.get("items"):
        rows = "".join(
            f'<div class="value-item">{_item_figure(e.get("id", ""), ident)}'
            f'<div><span class="value-name">{html.escape(n)}</span>'
            + (f'<p class="value-desc">{_inline(dsc)}</p>' if dsc else "")
            + "</div></div>"
            for ident, n, dsc in e["items"])
        body.append(f'<div class="value-list">{rows}</div>')
    body.append("</div>")
    if kind in ("prop", "op", "menu"):
        ctx.index.append({"id": rid, "label": label, "sub": where, "kind": "field",
                          "desc": desc[:160],
                          "terms": " ".join([i[1] for i in e.get("items", [])] + fig_names)})
    return "\n".join(body)


def _item_figure(prop_id: str, ident: str) -> str:
    """The small diagram beside one value of a choice
    (docs/figures/<prefix>-<value>.svg), or '' when that value has none."""
    prefix = _ITEM_FIGURES.get(prop_id)
    if prefix is None:
        return ""
    path = FIGURES_DIR / f"{prefix}-{ident.lower()}.svg"
    if not path.is_file():
        return ""
    svg = re.sub(r"<\?xml.*?\?>", "", path.read_text(encoding="utf-8"), flags=re.S).strip()
    return f'<div class="value-fig">{svg}</div>'


def _value_figures(e, ctx: _Ctx | None = None, rid: str = "") -> tuple[str, list[str]]:
    """A choice control's values as figure cards, laid out like the
    inflorescence kinds (the figure on the left; the name, the value's
    identifier and its description beside it), and the names; ('', []) when
    the control has no figures. A control with recorded values lists them in
    their own order, each with its figure; one whose values are listed at draw
    time (Leaf Generator's Shape) lists its figures, named by their <title>."""
    prefix = _VALUE_FIGURES.get(e.get("id", ""))
    if prefix is None:
        return "", []
    figures = {f.stem[len(prefix) + 1:]: f for f in FIGURES_DIR.glob(f"{prefix}-*.svg")}
    if not figures:
        return "", []

    def svg_of(path):
        return re.sub(r"<\?xml.*?\?>", "", path.read_text(encoding="utf-8"), flags=re.S).strip()

    kinds = prefix == "kind"
    caps_of = _kind_caps() if kinds else {}
    values = []                                 # (identifier, name, description, path)
    if e.get("items"):
        for ident, name, desc in e["items"]:
            key = ident.lower()
            path = figures.get(key)
            if path is None and kinds and key.startswith("compound_"):
                cand = FIGURES_DIR / f"compound-{key[len('compound_'):]}.svg"
                path = cand if cand.is_file() else None
            values.append((ident, name, desc or "", path))
    else:
        for key, path in figures.items():
            text = path.read_text(encoding="utf-8")
            m = re.search(r"<title>(.*?)</title>", text)
            d = re.search(r"<desc>(.*?)</desc>", text)
            values.append((key, m.group(1) if m else key, html.unescape(d.group(1)) if d else "",
                           path))
        values.sort(key=lambda v: v[1])
    cards = []
    for ident, name, desc, path in values:
        fig = f'<div class="kind-fig">{svg_of(path)}</div>' if path is not None else ""
        cid = f' id="{rid}-{slug(ident)}"' if rid else ""
        caps = ""
        if kinds:
            key = ident.lower()
            compound = key.startswith("compound_")
            chips = _caps_chips(caps_of.get(key[len("compound_"):] if compound else key))
            if compound:
                chips.insert(0, "compound")
            if chips:
                caps = ('<div class="kind-caps">'
                        + " &middot; ".join(html.escape(c) for c in chips) + "</div>")
            if ctx is not None and rid:
                ctx.index.append({"id": f"{rid}-{slug(ident)}", "label": f"{name} (kind)",
                                  "sub": "inflorescence kind", "kind": "kind"})
        cards.append(
            f'<div class="kind-card value-card"{cid}>{fig}<div class="kind-body">'
            f'<div class="kind-name">{html.escape(name)}<code>{html.escape(ident)}</code></div>'
            + caps
            + (f'<p class="kind-desc">{_inline(desc)}</p>' if desc else "")
            + "</div></div>")
    intro = f'<p class="ref-note">{_KIND_NOTE}</p>' if kinds else ""
    return (f'{intro}<div class="value-cards">{"".join(cards)}</div>',
            [v[1] for v in values])


def _kind_caps() -> dict:
    """Each inflorescence kind's capabilities, from the generated spec."""
    try:
        return json.loads(SPEC_PATH.read_text(encoding="utf-8"))["inflorescence_kinds"]
    except (OSError, KeyError, ValueError):
        return {}


def _caps_chips(caps) -> list:
    """A kind's shape and what it makes inert, as short words."""
    if not caps:
        return []
    chips = [caps["topology"].replace("_", " "),
             "pedicels" if caps["has_pedicels"] else "sessile"]
    if caps["terminal_cluster"]:
        chips.append("terminal cluster")
    if caps["flat_topped"]:
        chips.append("flat-topped")
    if caps["relayed"]:
        chips.append("relayed")
    if not caps["flowering_gradients_apply"]:
        chips.append("no flowering gradients")
    return chips


def _children(items) -> list:
    """A node's children, minus a heading that only repeats the label of the
    control right below it (a box titled "Placement" around one "Placement")."""
    out = []
    for i, item in enumerate(items):
        nxt = items[i + 1] if i + 1 < len(items) else None
        if (item.get("kind") == "heading" and nxt is not None and "node" not in nxt
                and (nxt.get("label") or "").lower() == item["text"].lower()):
            continue
        out.append(item)
    return out


def _scope(node) -> dict:
    """Per tag category, every value the node's states showed."""
    out: dict = {}
    for t in node.get("tags", ()):
        cat, val = t.split(":", 1)
        out.setdefault(cat, set()).add(val)
    return out


def _node(node, ctx: _Ctx, where: str, depth: int, toc: list, scope) -> str:
    title = node.get("title") or ""
    kind = {"fold": "fold", "subpanel": "sub-panel"}.get(node["node"], "")
    nid = ctx.uid("ui-" + slug(where + "-" + title))
    path = f"{where} › {title}" if where else title
    tag_scope = scope or _scope(node)
    out = [f'<div class="ref-node depth-{depth}" id="{nid}">']
    hlevel = min(3 + depth, 6)
    notes_badges = []
    if node.get("closed"):
        notes_badges.append("closed by default")
    if node.get("debug"):
        notes_badges.append("Debug Tools")
    shown = _shown_for(node.get("tags"), scope) if scope else None
    if shown:
        notes_badges.append("shown for " + shown)
    out.append(f'<h{hlevel} class="ref-group">{html.escape(title)}'
               f'<span class="ref-kind">{kind}</span>'
               + "".join(_badge(b, extra=True) for b in notes_badges) + f"</h{hlevel}>")
    if depth <= 1:
        toc.append((3, nid, title, "ref"))
    out.append(ctx.shots(path))
    for child in _children(node.get("children", [])):
        if "node" in child:
            out.append(_node(child, ctx, path, depth + 1, toc, tag_scope))
        else:
            out.append(_entry(child, ctx, path, tag_scope))
    out.append("</div>")
    return "\n".join(out)


def render(doc: dict, toc: list, figure=None) -> tuple[str, list, int]:
    """(HTML, search entries, number of controls). `figure(shot, caption)`
    draws a REFERENCE_SHOTS screenshot; without it none are shown."""
    ctx = _Ctx()
    ctx.figure = figure
    out = []
    toc.append((1, "reference-top", "Reference", "ref"))
    out.append(legend())
    for panel in doc["panels"]:
        title = panel["title"]
        pid = ctx.uid("panel-" + slug(title))
        toc.append((2, pid, title, "ref"))
        kicker = "Debug Tools — shown with the preference on" if panel.get("debug") else "panel"
        out.append(f'<section id="{pid}"><div class="sec-head"><h2>{html.escape(title)}</h2>'
                   f'<span class="sec-kicker">{html.escape(kicker)}</span></div>')
        note = ctx.notes.get("panel/" + title)
        if note:
            out.append(f'<p class="sec-intro">{_inline(note)}</p>')
        out.append(ctx.shots(title))
        scope = _scope(panel)
        for child in _children(panel.get("children", [])):
            if "node" in child:
                out.append(_node(child, ctx, title, 1, toc, scope))
            else:
                out.append(_entry(child, ctx, title, scope))
        out.append("</section>")
        ctx.index.append({"id": pid, "label": title, "sub": "panel", "kind": "guide"})
    return "\n".join(out), ctx.index, ctx.n_controls
