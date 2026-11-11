"""Curve evaluation for the age-driven growth model -- pure, bpy-free.

Curves are the mechanism the architecture doc names as the point of the whole
rebuild: one Age dial cannot produce species-specific developmental
trajectories unless each species gets to say *how* it interprets that dial.
The doc's own illustration is two plants with the same Age slider and
different branching curves -- one branching early and levelling off, one
holding back and then branching hard.

Everything here is a `Curve`: a fixed-resolution lookup table over [0, 1] with
linear interpolation between samples and clamping outside. That shape is
chosen so a Blender `CurveMapping` (which the panel will expose in Phase D)
can be sampled into one at build time and the growth core never has to import
bpy. The parametric add-on already proved `CurveMapping.evaluate()` works from Python and
that `template_curve_mapping` renders in a custom panel, so the widget side is
known-good; this is only the bpy-free half of that seam.

Inputs are always normalised to [0, 1] before evaluation. For curves driven by
an element's own age in whole steps -- internode elongation, leaf expansion --
use `Curve.over(span)`, which returns a plain callable that divides by the
species' maturation span first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

RESOLUTION = 64


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


class Curve:
    """A 1-D curve on [0, 1], sampled to a lookup table.

    Deliberately immutable-in-practice and cheap to call: the growth loop
    evaluates several curves per element per step, so this is on the hot path
    and a table lookup keeps it to two multiplies regardless of how the curve
    was authored.
    """

    __slots__ = ("samples", "points", "handles")

    def __init__(self, samples: Sequence[float],
                 points: Sequence[tuple[float, float]] | None = None,
                 handles: Sequence[str] = ()):
        if len(samples) < 2:
            raise ValueError("a Curve needs at least two samples")
        self.samples = tuple(float(s) for s in samples)
        self.points = tuple((float(x), float(y)) for x, y in points) if points else ()
        """The control points this curve was authored from, when it had any.

        The lookup table is what the growth loop evaluates, but it cannot be
        handed back to a Blender curve widget -- 64 samples would arrive as 64
        control points and be uneditable. Keeping the authoring points lets a
        species' curve seed the widget as the handful of points a person would
        actually draw. Empty for a curve that came *from* a widget, which has
        no shape to preserve beyond its samples."""
        self.handles = tuple(str(h) for h in handles) if handles and points else ()
        """A widget curve's own handle types, one per point (`from_blender`):
        with them the points re-seed the widget to exactly the shape the
        samples were taken from. Empty for a hand-authored (linear) curve."""

    def __call__(self, x: float) -> float:
        n = len(self.samples)
        t = min(1.0, max(0.0, float(x))) * (n - 1)
        i = int(t)
        if i >= n - 1:
            return self.samples[-1]
        frac = t - i
        return self.samples[i] * (1.0 - frac) + self.samples[i + 1] * frac

    def over(self, span: float) -> Callable[[float], float]:
        """Adapt this curve to an input measured in whole steps rather than a
        0-1 fraction. `span` is how many steps map to the curve's full domain
        -- e.g. `elongation.over(6)` means an internode reaches its final
        length six steps after it appears. A span of zero means "instant",
        which reads the curve's endpoint.
        """
        if span <= 0:
            end = self.samples[-1]
            return lambda _steps: end
        return lambda steps: self(float(steps) / span)

    def scaled(self, factor: float) -> Curve:
        return Curve([s * factor for s in self.samples],
                     [(x, y * factor) for x, y in self.points], self.handles)

    # -- constructors -----------------------------------------------------
    @classmethod
    def constant(cls, value: float) -> Curve:
        return cls([value, value], [(0.0, value), (1.0, value)])

    @classmethod
    def linear(cls, start: float = 0.0, end: float = 1.0) -> Curve:
        return cls([start, end], [(0.0, start), (1.0, end)])

    @classmethod
    def ramp(cls, onset: float, full: float, low: float = 0.0, high: float = 1.0) -> Curve:
        """Flat at `low` until `onset`, smoothly rising to `high` by `full`,
        flat after -- the exact shape of the architecture doc's branching
        diagrams, and the workhorse for every onset curve.

        Reverses cleanly: `ramp(0.2, 0.7, low=1.0, high=0.0)` is a decay,
        which is what apical dominance wants (strong in a juvenile plant,
        releasing as it matures).
        """
        if full <= onset:
            full = onset + 1e-6
        # Six segments across the transition, sampled off the smoothstep, and
        # then the curve is built *from those points* rather than from the
        # smoothstep itself.
        #
        # That last part is the important one, and it changed when these curves
        # became editable. A curve widget can only hold control points, so
        # whatever a ramp's authoring points do not capture is lost the moment
        # the curve round-trips through the panel -- and the old single
        # midpoint missed the smoothstep by up to 0.095. That is not cosmetic:
        # branching onset is compared against per-bud thresholds spread
        # uniformly over 0..1, so a 0.095 error moves a tenth of the buds
        # across the release line and compounds into a visibly different plant.
        # Six segments cut it to 0.016, but any residual is still a curve that
        # reads one way in the panel and evaluates another. Defining the ramp
        # as its own points removes the discrepancy by construction: what the
        # widget shows is what the growth loop evaluates, exactly.
        segments = 6
        points = [(0.0, low), (onset, low)]
        for i in range(1, segments):
            t = i / segments
            points.append((onset + (full - onset) * t,
                           low + (high - low) * _smoothstep(t)))
        points += [(full, high), (1.0, high)]
        return cls.from_points(points)

    @classmethod
    def sigmoid(cls, center: float = 0.5, width: float = 0.3, low: float = 0.0,
                high: float = 1.0) -> Curve:
        """A symmetric S centred on `center`. `ramp` is usually the better
        choice for onsets because its flat regions are exactly flat; this is
        for the cases that want no hard start or end point at all."""
        half = max(width, 1e-6) * 0.5
        return cls.ramp(center - half, center + half, low=low, high=high)

    @classmethod
    def from_points(cls, points: Sequence[tuple[float, float]]) -> Curve:
        """Piecewise-linear through explicit (x, y) control points, x in
        [0, 1] and sorted. This is the interchange format for hand-authored
        species curves and for anything round-tripping through a preset."""
        pts = sorted((min(1.0, max(0.0, float(x))), float(y)) for x, y in points)
        if not pts:
            raise ValueError("from_points needs at least one point")
        if len(pts) == 1:
            return cls.constant(pts[0][1])

        out = []
        seg = 0
        for i in range(RESOLUTION):
            x = i / (RESOLUTION - 1)
            while seg < len(pts) - 2 and x > pts[seg + 1][0]:
                seg += 1
            x0, y0 = pts[seg]
            x1, y1 = pts[seg + 1]
            if x <= x0:
                out.append(y0)
            elif x >= x1:
                out.append(y1)
            else:
                frac = (x - x0) / (x1 - x0)
                out.append(y0 + (y1 - y0) * frac)
        return cls(out, pts)

    @classmethod
    def from_blender(cls, mapping, index: int = 0) -> Curve:
        """Sample a Blender `CurveMapping` into a bpy-free Curve.

        The only function in this module that touches Blender types, and it
        takes the mapping as an argument rather than importing bpy -- so this
        file stays importable headless and the growth core stays testable
        without Blender, per the repository's standing convention for
        `model/` and `skeleton/`.
        """
        mapping.initialize()
        curve = mapping.curves[index]
        pts = list(curve.points)
        return cls([mapping.evaluate(curve, i / (RESOLUTION - 1)) for i in range(RESOLUTION)],
                   [(p.location[0], p.location[1]) for p in pts],
                   [p.handle_type for p in pts])


# -- Defaults ------------------------------------------------------------
# Every curve a species profile needs, with a value that already produces a
# sensible plant. A species overrides what it cares about and inherits the
# rest; nothing in the growth loop may require a curve to be supplied.



def default_flowering_onset() -> Curve:
    """Age -> whether reproductive fate is available at all.

    Open early, and deliberately so. This used to be `ramp(0.55, 0.8)` on the
    reasoning that a plant builds structure before it blooms -- true of a
    plant, but the gate it was written for was never actually applied, so
    every species was tuned against a model where an axis flowered the moment
    it completed its node count. Switching the gate on with the old default
    took a generic perennial from 143 nodes to 265 and a Rudbeckia from 13 to
    234, because a determinate axis forbidden to flower does not wait, it
    keeps growing.

    So the default states only the part that is unarguable -- a two-node
    seedling does not bloom -- and is fully open long before any axis
    completes its node count, which reproduces the tuned behaviour exactly. A
    species that genuinely blooms late says so by moving this curve, which is
    now a real control rather than a dead one.
    """
    return Curve.ramp(0.05, 0.2)




def default_senescence() -> Curve:
    """WS3 age death-front: plant Age (X) -> the fraction of lateral branches,
    oldest first, that have died by then (Y).

    Flat 0 is the default and means nothing dies, which keeps every plant
    byte-identical until senescence is asked for. A species (or the panel curve)
    raises the right-hand end to shed its oldest/lowest branches as it matures --
    crown-lifting -- since the earliest laterals are the lowest ones."""
    return Curve.from_points([(0.0, 0.0), (1.0, 0.0)])



def default_elongation() -> Curve:
    """Element age -> fraction of final internode length. Fast then easing,
    so a newly created phytomer visibly extends over the following steps
    instead of springing into existence at full size. Use via
    `.over(elongation_steps)`."""
    return Curve.from_points(ELONGATION_MODES["OVERLAPPING"][0])


ELONGATION_MODES: dict[str, tuple[tuple[tuple[float, float], ...], int]] = {
    # (curve points, duration in steps). Overlapping is `default_elongation`
    # over the profile's default `elongation_steps` -- what every plant grew
    # with before the modes existed.
    "SEQUENTIAL": (((0.0, 0.05), (0.5, 0.6), (1.0, 1.0)), 1),
    "OVERLAPPING": (((0.0, 0.06), (0.35, 0.62), (0.7, 0.93), (1.0, 1.0)), 5),
    "DELAYED": (((0.0, 0.04), (0.45, 0.1), (0.8, 0.85), (1.0, 1.0)), 6),
}
"""Internode Elongation modes -- one mechanism (each internode reaches its
full length along `elongation` over `elongation_steps` after it forms; one
node forms per step), three shapes of it: **Sequential** -- an internode is
full by the time the next forms, one stretching at a time; **Overlapping** --
several stretch at once, the lower ones still growing while new ones form
(most herbs); **Delayed** -- a new internode stays short while the next nodes
form, then stretches (compressed young growth that spreads out later)."""


def elongation_mode(mode: str) -> tuple[Curve, int]:
    """A mode's (curve, duration) -- exact, piecewise-linear."""
    points, steps = ELONGATION_MODES[mode]
    return Curve.from_points(points), steps


STEM_THICKENING_POINTS = ((0.0, 0.3), (0.5, 0.72), (1.0, 1.0))


def default_stem_thickening() -> Curve:
    """An axis's growth so far (0 just started -> 1 grown) -> share of its
    stated widths: a young stem is a slimmer version of its final taper and
    thickens as it lengthens, reaching its level's widths when complete
    (`development._thicken`). Flat at 1 is the old behaviour (full width from
    the first node)."""
    return Curve.from_points(STEM_THICKENING_POINTS)


def default_leaf_expansion() -> Curve:
    """Element age -> fraction of final leaf scale. Slower than internode
    elongation, so young leaves near a growing tip read as smaller than
    established ones further down."""
    return Curve.from_points([(0.0, 0.04), (0.3, 0.35), (0.65, 0.8), (1.0, 1.0)])


def default_leaf_size_profile() -> Curve:
    """Position along an axis -> leaf size multiplier. Flat, so a species that
    says nothing gets exactly the old behaviour. Real cauline leaves shrink
    hard toward the tip; Rudbeckia sets a real gradient."""
    return Curve.constant(1.0)


DOME_POINTS = ((0.0, 0.0), (0.25, 0.0625), (0.5, 0.25), (0.75, 0.5625), (1.0, 1.0))
"""A parabola, `t**2`. Shared with `blender.curves.SHAPES`' own "Dome" entry so
the widget names the default rather than showing it as Custom."""


def default_head_dome_profile() -> Curve:
    """Distance from a flat-topped head's centre -> how far that flower sits
    below the head's own plane, as a fraction of `Inflorescence.head_dome`.

    **Not flat at 1.0, unlike every other profile here, and it cannot be.** A
    constant would drop the centre by exactly as much as the rim, which lowers
    the plane and leaves it just as flat -- the control would do nothing at any
    setting. So this curve *is* the head's cross-section rather than a
    multiplier on a shape hidden in the solver, which is also what makes it
    readable: draw a circular arc and the head is spherical. Same call, and for
    the same reason, as `pedicel_width_profile`'s real taper.

    A parabola by default rather than a true sphere cap (`1 - sqrt(1 - t**2)`,
    which the existing Exponential Growth shape already approximates to within
    0.015). A sphere cap is nearly flat across the middle and then turns hard
    at the rim; the parabola curves gently the whole way, which is what "a bit
    rounder rather than completely flat" asks for. Either is one click away in
    the shape dropdown.
    """
    return Curve.from_points(DOME_POINTS)


def default_leaf_angle_profile() -> Curve:
    """Position along an axis -> leaf angle multiplier. Flat by default. Lower
    leaves on many perennials are held far more horizontally than the ones
    crowded under the inflorescence."""
    return Curve.constant(1.0)
