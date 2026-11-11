"""The core parametric primitive: a `Parameter` whose value depends on context.

A Parameter is `base` optionally modulated by:
  * a ramp over `factor` (0 = base of parent, 1 = tip)   ← SpeedTree-style curve
  * a ramp over `age`    (0 = seedling, 1 = mature)        ← Natsura-style growth
  * per-element `variance` (deterministic, seeded)

This mirrors the Float Curves found inside the blend file's `Trunk`/`Branches`
groups (bend profile, taper profile, length/radius % of parent), but as plain
serializable data so it drives the Python skeleton AND round-trips to JSON presets.

Pure module: no `bpy`. Inside Blender, ramps are additionally backed by a
`CurveMapping` widget for authoring; the control points live here as the source of
truth and are pushed to/read from the widget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- ramp (a 1D curve authored as control points) ----------------------------

@dataclass
class Ramp:
    """A 0..1-domain response curve stored as sorted (x, y) control points.

    Evaluation here is piecewise-linear — deterministic and dependency-free, good
    for headless generation and preset round-tripping. Inside Blender the same
    points feed a `CurveMapping` widget, which is the authoring surface; keep the
    two in sync when a widget is present.
    """

    points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)]
    )

    def evaluate(self, x: float) -> float:
        pts = self.points
        if not pts:
            return x
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                return y0 + t * (y1 - y0)
        return pts[-1][1]

    # --- serialization ---
    def to_dict(self) -> dict:
        return {"points": [[round(x, 6), round(y, 6)] for x, y in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> Ramp:
        return cls(points=[(float(x), float(y)) for x, y in d["points"]])

    @staticmethod
    def constant(y: float = 1.0) -> Ramp:
        return Ramp(points=[(0.0, y), (1.0, y)])


def varied(name: str, base: float, variance: float, seed: int, floor: float | None = None) -> float:
    """Deterministically jitter `base` by up to `variance` (absolute, see
    `Parameter.variance`), reproducible per (seed, name) — re-rolling the owning
    object's Seed re-rolls every varied field at once, like a new random instance of
    the same authored species. `floor`, when given, keeps the result from crossing a
    hard domain boundary (e.g. Length/Base Radius can never be <= 0) regardless of
    how large `variance` is set — variance alone can't be relied on to stay inside
    the valid range since it's an unbounded +/- window around `base`.

    For a *global* (once-per-generator) re-roll, not a per-element one — see
    `Parameter.evaluate`'s own `element_id` for per-element variance instead.
    """
    if not variance:
        return base
    v = Parameter(name=name, base=base, variance=variance).evaluate(seed=seed)
    return v if floor is None else max(floor, v)


# --- parameter ---------------------------------------------------------------

@dataclass
class Parameter:
    name: str = ""
    base: float = 0.0
    # absolute, same units as `base`: value += uniform(-variance, +variance). e.g.
    # base=3, variance=1 -> anywhere from 2 to 4, not a fraction of base.
    variance: float = 0.0
    factor_ramp: Ramp | None = None  # multiplier vs. position-along-parent
    age_ramp: Ramp | None = None     # multiplier vs. global age

    def evaluate(
        self,
        factor: float = 0.0,
        age: float = 1.0,
        element_id: int = 0,
        seed: int = 0,
    ) -> float:
        """Resolve the parameter for one element in context.

        `factor`   : 0..1 position along the parent for this element.
        `age`      : 0..1 global developmental time.
        `element_id`+`seed` : make per-element variance deterministic.
        """
        v = self.base
        if self.variance:
            # a str seed (not a tuple) so this is stable regardless of Python's
            # hash randomization AND across Python versions — newer stdlib `random`
            # rejects arbitrary tuple seeds outright (found live: Blender 5.1's
            # bundled Python raises TypeError here where an older interpreter
            # only warned and fell back to hashing).
            rng = random.Random(f"{seed}:{element_id}:{self.name}")
            v += rng.uniform(-self.variance, self.variance)
        # variance jitters the base value itself, *then* the ramps shape whatever
        # that jittered base turns out to be — not the other way around, so e.g. a
        # tapered radius's variance shifts the whole taper up/down by a consistent
        # amount rather than scaling it.
        if self.factor_ramp is not None:
            v *= self.factor_ramp.evaluate(factor)
        if self.age_ramp is not None:
            v *= self.age_ramp.evaluate(age)
        return v

    # --- serialization ---
    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "base": self.base}
        if self.variance:
            d["variance"] = self.variance
        if self.factor_ramp is not None:
            d["factor_ramp"] = self.factor_ramp.to_dict()
        if self.age_ramp is not None:
            d["age_ramp"] = self.age_ramp.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Parameter:
        return cls(
            name=d.get("name", ""),
            base=float(d.get("base", 0.0)),
            variance=float(d.get("variance", 0.0)),
            factor_ramp=Ramp.from_dict(d["factor_ramp"]) if "factor_ramp" in d else None,
            age_ramp=Ramp.from_dict(d["age_ramp"]) if "age_ramp" in d else None,
        )
