"""The phytomer data model (rebuild plan, Phase A) -- pure, bpy-free.

This replaces `sim/attributes.py`'s columnar `NodeArrays`. That layout existed
to feed vectorised space-colonization distance matrices; with space
colonization demoted out of v1 there is nothing
left that wants a column store, and plain objects express the botany far more
directly.

The unit is the **phytomer**: one internode, plus the node at its distal end
carrying leaves and their axillary buds. Three consequences of that choice are
worth stating up front, because each one closes a specific failure of the
previous build:

1. **Leaves are structural, not a dice roll.** A phytomer *has* leaves the
   moment it exists, placed by the phyllotaxis pattern. The old model made a
   leaf the outcome of a `leaf_probability` roll on a released bud, which is
   why a 598-node plant shipped with two leaves on it. Nothing here can
   produce a leafless phytomer by accident.

2. **The axillary bud sits in the leaf axil**, one per leaf, exactly as in
   real morphology. So "how many buds per node" is not a separate parameter --
   it falls out of the phyllotaxis pattern that already decided how many
   leaves the node carries.

3. **Branch order is declared, not accumulated.** An axis knows its own
   `order` because its parent bud's axis had one; there is no path by which a
   fork silently deepens it. The previous build reached order 17 because
   `order` really meant "forks behind me", and diameter was derived from it.

Geometry is deliberately *not* stored as absolute positions. Each phytomer
keeps its own direction, length and a parallel-transported reference vector;
`resolve_geometry()` walks the axis tree and derives world positions from
that chain. This matters for two later phases: internode elongation changes
lengths every step (so positions must be re-derived anyway), and Stage 4's
gravity pass bends the structure by rotating directions at a joint and letting
the change propagate down the chain -- which is only cheap if positions are
derived rather than stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-9
_UP = np.array([0.0, 0.0, 1.0])
_DOWN = -_UP


class BudPosition:
    """Where on the plant a meristem sits (architecture doc's "Bud / Meristem
    Positions").

    INTERNODAL and CAULINE were added 2026-08-20 with the attachment layer,
    cashing in the note this docstring used to carry: they were "reachable
    additions to this model, not rewrites of it", and they were. LEAF_OPPOSED
    followed, along with BASAL and ADVENTITIOUS. EXTRA_AXILLARY is the last one
    the architecture doc names that is still unbuilt."""

    TERMINAL = "terminal"
    AXILLARY = "axillary"
    INTERNODAL = "internodal"
    """Between two nodes rather than at one -- carries `Bud.along_internode`."""
    LEAF_OPPOSED = "leaf_opposed"
    """At a node but on the far side of the stem from the leaf, rather than in
    its axil -- half a turn round from where an axillary bud would sit."""
    CAULINE = "cauline"
    """On older wood, well below the leaves: the meristem cauliflory arises
    from. Released by **age** rather than by position, because it breaks from
    wood laid down long before, so it cannot be released at the moment its own
    node is made.

    Shared by `Attachment.CAULIFLOROUS` and `Attachment.RAMIFLOROUS`: the two
    differ in which axes they may break from, not in what kind of meristem it
    is, so one position kind covers both."""
    BASAL = "basal"
    """At the root crown, below everything else the plant made."""
    ADVENTITIOUS = "adventitious"
    """From tissue that never carried a meristem -- the only bud here that
    development SYNTHESISES rather than finds, since having no pre-existing
    bud site is the definition of adventitious."""


class BudState:
    ACTIVE = "active"      # a terminal bud currently producing phytomers
    DORMANT = "dormant"    # an axillary bud not yet released; re-evaluated each step
    RELEASED = "released"  # committed to a fate, never evaluated again
    ABORTED = "aborted"    # will never do anything (e.g. an indeterminate axis just stopping)


class BudFate:
    NONE = "none"      # not yet decided, or decided to be nothing
    SHOOT = "shoot"    # became a new axis
    FLOWER = "flower"  # became a flower organ


class LifecycleState:
    """What has become of an organ or an axis by the end of the plant's life,
    written by the post-growth `lifecycle` pass (WS3) -- never during growth.

    The sim grows the whole plant first; this pass then marks what dies, which
    is how removal stays compatible with "the environment modifies, never
    drives": nothing is stopped from growing, structure is edited after the
    fact, and environment-driven death only *reads* the already-built light
    field. Default ALIVE, so the pass is inert until a trigger fires and every
    plant is byte-identical until senescence is asked for."""

    ALIVE = "alive"
    DEAD = "dead"    # stops developing, stays attached, recoloured
    SHED = "shed"    # removed from the structure entirely


class BudBlock:
    """Which gate in `development._release_buds` last held a bud dormant.

    Diagnostic only -- nothing in the model reads it back, and writing it
    changes no decision. It exists because a bud that never released is the
    plant's most common invisible failure, and `release_threshold` alone
    cannot explain one: every gate is a comparison against that same number,
    so the threshold says *how reluctant* the bud is and this says *what it
    actually ran into*.

    Overwritten on every evaluation, so at the end of growth it holds the
    reason the bud is still dormant **now** rather than the first thing that
    ever stopped it.

    Two of these lift on their own and three do not, which is the distinction
    worth knowing before tuning anything: DOMINANCE lifts when the parent axis
    stops growing, and ONSET lifts as Age rises. DENSITY, ZONE and ORDER are
    fixed for a given set of settings -- a bud held by one of those will never
    release however long the plant grows.
    """

    NONE = "none"
    """Not held: the bud released, or has not been evaluated yet."""
    DENSITY = "density"
    ZONE = "zone"
    DOMINANCE = "dominance"
    ONSET = "onset"
    """Retired with the branching overhaul:
    every bud with a next level grows out, so density, zone, dominance and onset
    no longer hold any. Kept so an old saved reason still reads."""
    ORDER = "order"
    """Its axis is on the deepest branching level -- there is no next level for
    it to become -- and it is not floral enough to become a flower instead.
    Permanent."""
    MASK = "mask"
    """Its branch's level Masks rejected it: the
    survival draw fell above the folded mask. Decided once; permanent."""


class OrganKind:
    LEAF = "leaf"
    FLOWER = "flower"
    CATKIN = "catkin"
    """A catkin, and the one organ kind that carries no instanced asset.

    Every other organ is a point the node group instances a mesh on. A catkin
    is a *hanging curve* instead -- a stalk that keeps going and swells into a
    tube -- so its geometry is written into the plant's own skeleton mesh
    alongside the pedicels and swept by the same `Curve to Mesh`. Nothing is
    placed for it in the organs object.

    That is the whole reason it is its own kind rather than a leaf form. It was
    built as `LeafForm.CATKIN` first, riding the leaf instancing path, and a
    supplied mesh cannot do what a catkin has to do: hang under its own weight
    and bend as the branch carrying it moves. See `Organ.catkin_length`.
    """
    FRUIT = "fruit"
    """A flower that has set fruit (calendar phase 4). Botanically a fruit
    develops from a pollinated flower in place, so it is the *same* organ with
    its kind flipped once the calendar's fruit-set date passes, and removed once
    fruit-drop does. Written only by the day-path `apply_fruit` pass; off the day
    path (goldens, every existing caller) no organ ever becomes a fruit.

    **Data only in milestone 1**: it carries no instanced asset, so a fruit is
    not drawn (a flower simply disappears from the render at fruit-set), but it
    is counted in Diagnostics and shown in the WS1 inspector. The fruit asset /
    GN branch that renders it is milestone-2 frontend."""


class LeafForm:
    """Which mesh a leaf-kind organ draws from -- see `Organ.form`.

    Not an `OrganKind`: all three are leaves, and everything downstream of the
    leaf/flower split is right for a bract unchanged. Only the asset differs.
    A catkin is **not** one of these -- it briefly was, and an instanced mesh
    cannot hang. See `OrganKind.CATKIN`.

    """

    JUVENILE = 0
    # 1 was ADULT (heteroblasty); removed -- one leaf collection now, and a
    # species wanting two leaf shapes uses two leaf components with a mask. The
    # value is left as a gap so BRACT stays 2 (baked as an int attr, read by GN).
    BRACT = 2


def orthonormal_reference(direction: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to `direction` -- the seed of an axis's
    phyllotaxis frame. Which one it is doesn't matter (it only sets where
    divergence angle zero points); that it is stable and perpendicular does.
    """
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, axis))) > 0.99:
        axis = np.array([1.0, 0.0, 0.0])
    ref = _cross(direction, axis)
    return ref / max(float(np.linalg.norm(ref)), _EPS)


def parallel_transport(reference: np.ndarray, from_dir: np.ndarray,
                       to_dir: np.ndarray) -> np.ndarray:
    """Carry `reference` from one internode's frame to the next by the minimal
    rotation taking `from_dir` to `to_dir` (Rodrigues about their cross
    product).

    This is the piece the previous build did not have. `sim/loop.py`'s
    `_lateral_direction` rebuilt its perpendicular from a fixed world axis on
    every call, and its own docstring flagged the consequence: successive
    divergence angles could not accumulate consistently around the stem, so a
    137.5-degree spiral was 137.5 degrees from an arbitrary reference rather
    than from the previous leaf. Transporting the frame is what makes a real
    spiral, and what makes Decussate's 90-degree alternation mean anything.
    """
    axis = _cross(from_dir, to_dir)
    sin_theta = float(np.linalg.norm(axis))
    cos_theta = float(np.dot(from_dir, to_dir))
    if sin_theta < _EPS:
        # Parallel (no rotation needed) or antiparallel (degenerate; the
        # reference is still perpendicular to both, so keeping it is correct).
        return reference
    axis = axis / sin_theta
    transported = (
        reference * cos_theta
        + _cross(axis, reference) * sin_theta
        + axis * float(np.dot(axis, reference)) * (1.0 - cos_theta)
    )
    return transported / max(float(np.linalg.norm(transported)), _EPS)


_CATKIN_FAN_DEG = 14.0
"""How far each catkin leans off the stalk where it attaches.

Not a control. What it has to do is stop a catkin lying against the stalk it
hangs from; beyond that nobody reads the angle."""

_CATKIN_SPIRAL_DEG = 137.5
"""The golden angle, so successive catkins on one stalk sit round it rather
than in a row. The same number `Phyllotaxis` uses, and for the same reason."""

_CATKIN_FIRST_ATTACH = 0.5
"""Where along the stalk the first catkin hangs. They occupy its second half,
so the stalk reads as a real stalk with catkins on it rather than as a stem
that happens to end in a bunch."""

_CATKIN_CAP_SEGMENTS = 6
_CATKIN_MID_SEGMENTS = 18
_CATKIN_STALK_SEGMENTS = 8

_CATKIN_DROP = 1.1
"""How far below its attachment a catkin's crown sits, in multiples of its own
width. Attaching exactly at the pole leaves the stalk meeting the catkin
perpendicular to its very top, which reads as mechanical; a real one hangs from
a point a little way down its shoulder."""

_CATKIN_LIMP_DEG = 200.0
"""How far a catkin *body* is allowed to sag, regardless of what its stalk did.

**A catkin has almost no structural stability of its own** -- what holds it out
from the twig is the pedicel, and the catkin simply hangs off the end of it. So
the body is not given a share of the stalk's droop, which is how this was built
first: it is given more turn than it can possibly use, and `_sag` stops it at
straight down. The result is that a catkin hangs vertically from wherever its
stalk happens to end, which is what they do.

Built the other way -- stalk and body splitting one droop budget between them --
catkins on upward-angled stalks came out held up rigidly, because the body's
share was not enough to swing it down from where the stalk left it. Reported as
"they appear much too rigid".

The shared Orientation angle therefore controls the **stalk**, which is the part
that actually has stiffness to vary -- 180 degrees hangs it straight down. It is
the same control every other kind aims its axis with; here the implementation is
a polyline sag rather than a gravitropic setpoint. See
`morphology.OrientationMode`."""


def _catkin_profile_ts(width: float, length: float) -> list[float]:
    """Where to put a catkin body's vertices along its own length.

    **Crowded at both ends and sparse through the middle**, because that is
    where the shape is: the caps are hemispheres about as long as the tube is
    wide -- six percent of the body on the default -- and evenly spaced
    vertices put *less than one* segment inside them. The cap then renders as a
    single straight taper, which is a pencil point rather than a round end, and
    was reported as exactly that.

    **Inside a cap the vertices are spaced by even ANGLE, not even length**,
    which is the difference between a clean hemisphere and a mangled one.
    A cap's radius is `width * cos(theta)`, so stepping evenly along the axis
    crowds the rings hard against the pole -- they end up separated by microns
    in radius while the surface is still turning, and the shading breaks up.
    Stepping evenly in theta spreads them over the dome instead. Reported as
    "the bottom spherical part is a bit mangled".

    The middle is a straight tube and needs almost nothing.
    """
    cap = min(0.45, width / max(length, _EPS))
    half = math.pi * 0.5
    # theta 0 at the equator (full width) to 90 degrees at the pole; the
    # along-axis offset of a ring is `cap * sin(theta)`, its radius
    # `width * cos(theta)`. See `catkin_radius`, which is the same capsule.
    dome = [math.sin(half * i / _CATKIN_CAP_SEGMENTS)
            for i in range(_CATKIN_CAP_SEGMENTS + 1)]
    ts = [cap * (1.0 - s) for s in reversed(dome)]           # top cap, pole first
    span = 1.0 - 2.0 * cap
    ts += [cap + span * i / _CATKIN_MID_SEGMENTS
           for i in range(1, _CATKIN_MID_SEGMENTS + 1)]
    ts += [(1.0 - cap) + cap * s for s in dome[1:]]          # bottom cap
    return ts


def _catkin_window(t: float, cap: float) -> float:
    """Position along a catkin, flattened inside both of its domes.

    Zero through the crown, one through the tip, smooth in between with zero
    derivative where it meets each -- so anything driven by it leaves the domes
    **rigid** and puts all of its variation in the straight middle.

    Both the sag and the noise read it, and both need it for the same reason:
    a dome's vertices sit a fraction of a millimetre apart along the axis,
    while the tube's rings are built square to the *local* tangent. Turn or
    displace those vertices relative to each other by even a few microns and
    the tangent swings right round, so the end of the catkin corkscrews.
    Reported as "what is up with the weird rotation at the bottom".

    Clamping position instead was tried and leaves a corner at each boundary,
    which across a sub-millimetre segment is 50 degrees of tangent swing.
    """
    inner = max(cap, _EPS)
    x = min(1.0, max(0.0, (t - inner) / max(1.0 - 2.0 * inner, _EPS)))
    return x * x * (3.0 - 2.0 * x)


def _catkin_noise(points: list[np.ndarray], ts: list[float], cap: float,
                  amount: float, width: float, seed_key: int) -> list[np.ndarray]:
    """Bend a catkin slightly out of true, so a bunch of them does not read as
    several copies of one tube.

    Two low harmonics with a per-catkin phase rather than per-vertex jitter:
    what wants breaking up is the *shape*, and a catkin that wobbles vertex to
    vertex reads as a bad mesh rather than as a real one. Amplitude is a
    fraction of the catkin's own width, so the same setting is equally subtle
    on a thick catkin and a thin one, and it grows from nothing at the
    attachment -- a catkin is held at the top and free at the bottom.

    **Read at each vertex's real position along the catkin, not its index.**
    The vertices are crowded at the ends, so index fractions spread the whole
    envelope across the cap's 0.15 mm and tear the dome apart -- the pole ends
    up displaced by more than its own ring radius, which shows as a pinch. The
    third place in this feature where an index was mistaken for a position.

    **Frozen through both caps**, so a dome moves rigidly rather than being
    stirred from inside. Cap vertices sit a fraction of a millimetre apart
    along the axis, and the tube's rings are built perpendicular to the *local*
    tangent -- so a lateral nudge of even a few microns between two of them
    swings that tangent right round, and the end of the catkin visibly
    corkscrews. Reported as "what is up with the weird rotation at the bottom".
    Holding the offset constant across a cap keeps the centreline straight
    there, which keeps the tangent stable and the rings square to it.
    """
    if amount <= 0.0 or len(points) < 3:
        return points
    rng = np.random.default_rng(seed_key)
    phase = rng.uniform(0.0, 2.0 * math.pi, 4)
    heading = points[-1] - points[0]
    norm = float(np.linalg.norm(heading))
    if norm < _EPS:
        return points
    u_axis = orthonormal_reference(heading / norm)
    v_axis = _cross(heading / norm, u_axis)
    scale = amount * width
    out = []
    for point, t in zip(points, ts, strict=True):
        u = _catkin_window(t, cap)
        swing = (math.sin(phase[0] + 2.3 * math.pi * u) * u_axis
                 + math.sin(phase[1] + 3.7 * math.pi * u) * v_axis)
        out.append(point + swing * (scale * u))
    return out


def catkin_bunch(
    node: np.ndarray, launch: np.ndarray, stalk: float, body: float, count: int,
    droop: float, width: float = 0.004, noise: float = 0.0,
    seed_key: int = 0, length_variation: float = 0.0,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[int], list[list[float]]]:
    """One stalk leaving `node`, with `count` catkins hung along its second half.

    Returns `(stalk_points, [body_points, ...], [attach_index, ...],
    [body_radii, ...])`.

    **The radii are returned rather than recomputed by the caller, and that is
    the point.** They were recomputed, from the vertex's *index* fraction --
    but the vertices are deliberately unevenly spaced (see
    `_catkin_profile_ts`), so an index fraction is not a position, and the
    whole rounded end collapsed into the final segment. Reported as the cap
    looking like "a flat disc that is turned". Computing both here, from the
    one `ts` that placed the points, makes them unable to disagree.

    The
    attach indices say **which stalk vertex each body forks from**, and the
    caller has to honour them: chaining every body from the stalk's last vertex
    instead put the bodies in the right places but ran a visible strand back to
    the tip from each -- reported as "the attachment point for each catkin is
    still at the tip of the pedicel, but their height/distribution seems to
    appear correct", which is exactly that failure.

    **A bunch is one pedicel carrying several catkins, not several catkins each
    with a pedicel.** That is what hazel, birch and willow do, and building it
    the other way fuses stalk and body into one tapering shape with nothing to
    read as the join.

    They hang **along** the stalk rather than all from its tip, spaced across
    its second half and turned by the golden angle so they sit round it rather
    than in a row. All-from-the-tip was the first version and reads as a tassel.

    How many hang here is `count`, deliberately **not** the node's phyllotaxis:
    an earlier version took it from a whorled arrangement, which does put three
    organs on a node -- and whorls the *leaves* with them, and every
    catkin-bearing species here has spiral foliage.
    """
    direction = np.asarray(launch, dtype=float)
    norm = float(np.linalg.norm(direction))
    total = stalk + body
    if norm < _EPS or total <= _EPS or count < 1:
        return [], [], [], []
    direction = direction / norm
    # A catkin hangs -- it never emerges upward. The launch direction is radial
    # (perpendicular to the twig), so on an angled twig it points up for bunches
    # on the twig's upper side, which then read as "growing up before drooping".
    # Reflect any upward launch to point downward-out instead, keeping its
    # azimuth, so every bunch emerges below horizontal and the sag only deepens
    # an already-downward hang. (A limp peduncle would not hold an upward base
    # under a heavy bunch either -- this is what the stiff-tube stalk misses.)
    if direction[2] > 0.0:
        direction = direction.copy()
        direction[2] = -direction[2]
    origin = np.asarray(node, dtype=float)

    # **`droop` is the stalk's sag in degrees**, computed physically by
    # `physics._set_catkin_droop` from the bunch's own weight on this fine
    # stalk -- a heavy bunch on a thin stalk hangs, a light one barely does. It
    # replaces the old 0..1 fraction that came from an Orientation dial. `_sag`
    # stops every segment at vertical, so a value past what it takes to reach
    # straight down simply hangs the stalk and no further.
    if stalk > _EPS:
        stalk_points = _sag(
            [origin + direction * (stalk * i / _CATKIN_STALK_SEGMENTS)
             for i in range(_CATKIN_STALK_SEGMENTS + 1)],
            -min(180.0, max(0.0, droop)))
    else:
        stalk_points = [origin]

    # The body hangs limp whatever the stalk did -- see `_CATKIN_LIMP_DEG`.
    # Not a share of the stalk's droop: that left catkins on upward-angled
    # stalks held rigidly out, having run out of turn before reaching vertical.
    limp = -_CATKIN_LIMP_DEG

    rng = np.random.default_rng(seed_key * 31 + 7)
    bodies, attach, radii = [], [], []
    last_stalk = len(stalk_points) - 1
    for k in range(count):
        # Drawn per catkin rather than per bunch, so the catkins on one stalk
        # differ from each other and not merely from the next stalk's.
        this_body = max(1e-4, body * (1.0 + length_variation
                                      * float(rng.uniform(-1.0, 1.0))))
        ts = _catkin_profile_ts(width, this_body)
        # Spread across the stalk's second half, the last one at its tip.
        where = (1.0 if count == 1
                 else _CATKIN_FIRST_ATTACH
                 + (1.0 - _CATKIN_FIRST_ATTACH) * k / (count - 1))
        index = min(last_stalk, max(1, int(round(where * last_stalk))))
        attach_point = stalk_points[index]
        heading = stalk_points[index] - stalk_points[index - 1]
        heading_norm = float(np.linalg.norm(heading))
        heading = heading / heading_norm if heading_norm > _EPS else direction

        radial = rotate_about(orthonormal_reference(heading), heading,
                              _CATKIN_SPIRAL_DEG * k)
        start = rotate_about(heading, _cross(heading, radial), _CATKIN_FAN_DEG)
        # Offset off the stalk so the catkin clears it rather than growing out
        # of its side, and dropped a little way down `start` so the stalk meets
        # it just below its crown. Attaching exactly at the pole puts the join
        # perpendicular to the catkin's own top, which reads as mechanical --
        # a real one hangs from a point slightly down its shoulder.
        base = attach_point + radial * (width * 1.2) + start * (width * _CATKIN_DROP)
        # The sag is read through the same window, so the crown and the tip
        # each stay rigid and every degree of the turn happens in the straight
        # part between them. Read at raw position instead, the crown bent
        # 19 degrees inside itself -- most of a limp catkin's turn is
        # front-loaded, and the crown is exactly where it lands.
        cap_frac = min(0.45, width / max(this_body, _EPS))
        points = _sag([base + start * (this_body * t) for t in ts], limp,
                      fractions=[_catkin_window((ts[i] + ts[i + 1]) * 0.5, cap_frac)
                                 for i in range(len(ts) - 1)])
        bodies.append(_catkin_noise(points, ts, cap_frac, noise, width,
                                    seed_key * 97 + k))
        attach.append(index)
        radii.append([catkin_radius(x, width, this_body) for x in ts])
    return stalk_points, bodies, attach, radii


def catkin_radius(along: float, width: float, length: float) -> float:
    """Tube radius at `along` (0 at the catkin's top, 1 at its tip).

    A **capsule**: constant through the middle with a hemisphere at each end,
    the caps being exactly as long as the tube is wide. That is what makes the
    ends read as rounded.

    An ellipse across the whole body was tried first and is wrong for the same
    reason a rugby ball is not a capsule: with a body twenty times longer than
    it is wide, an ellipse is a spindle, and its ends come to a visible point.
    Reported as exactly that -- "the bottom of the catkins are too pointed".

    The profile alone is not enough; the vertices have to be there to carry it.
    See `_catkin_profile_ts`.

    Never returns zero -- a ring of zero radius is a degenerate face rather than
    a closed end, and at this scale a hair's width reads as closed.
    """
    cap = min(0.45, width / max(length, _EPS))
    if cap <= _EPS:
        return max(_MIN_TUBE_RADIUS, width)
    if along < cap:
        t = (cap - along) / cap
    elif along > 1.0 - cap:
        t = (along - (1.0 - cap)) / cap
    else:
        return max(_MIN_TUBE_RADIUS, width)
    return max(_MIN_TUBE_RADIUS, width * math.sqrt(max(0.0, 1.0 - t * t)))


def _cross(a, b):
    """np.cross for two float64 3-vectors without its setup (~50 us a call,
    ~30k calls a grow): the same products and differences in the same order,
    so the same bits. Anything else goes to np.cross."""
    if (type(a) is np.ndarray and type(b) is np.ndarray and a.shape == (3,) and b.shape == (3,)
            and a.dtype == np.float64 and b.dtype == np.float64):
        a0, a1, a2 = a
        b0, b1, b2 = b
        return np.array((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0))
    return np.cross(a, b)


def rotate_about(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation of `vector` about unit `axis` by `degrees`."""
    theta = np.radians(degrees)
    return (
        vector * np.cos(theta)
        + _cross(axis, vector) * np.sin(theta)
        + axis * float(np.dot(axis, vector)) * (1.0 - np.cos(theta))
    )


@dataclass
class Organ:
    """A leaf or a flower. Carries its own birth step so it can mature on its
    own clock -- the mechanism that lets one Age dial show mature organs low
    on the plant and juvenile ones at the tips."""

    id: int
    kind: str
    birth_step: int
    final_scale: float = 1.0
    divergence_deg: float = 0.0
    elevation_deg: float = 45.0
    along_internode: float = 1.0
    """Where along its node's internode the organ sits, 0 at its base and 1 at
    the node. Only an Internodal solitary flower uses anything but 1."""
    inflorescence: int = 0
    """Which inflorescence a flower belongs to (`MorphologyProfile.
    inflorescence_at`): 0 the primary, k the k-th further one -- so the bridge
    can hand it to the flower component that grows it."""
    head_azimuth_deg: float = -1.0
    """Where this flower sits **around its own flat-topped head**, in degrees,
    or -1 while unset. Only `inflorescence.solve_flat_head` reads or writes it.

    Its own field because `divergence_deg` cannot serve both ends of that
    solve: the head layout reads an azimuth, and the answer the solve produces
    is a *pedicel launch* azimuth in the phytomer's own frame, which it writes
    back to the same place. That is fine while the solve runs once and silently
    wrong the moment it runs twice -- which it now does, because a head has to
    be re-flattened after gravity bends the stem under it. Measured over three
    passes, flowers rotated up to 28.7 degrees each time and one azimuth octant
    went 20 -> 11 -> 0 while another went 22 -> 27 -> 31: the head piling up on
    one side, reported exactly as lopsided.

    Captured once from the phyllotactic divergence, so the head keeps the
    plant's own arrangement and every later solve reproduces the same layout."""
    pedicel_length: float = 0.0
    """Stalk length between the node and the flower. Zero means sessile --
    the organ sits directly on the axis, which is what separates a spike from
    a raceme and a head from an umbel. The stalk becomes real geometry in the
    bridge, not just an offset."""
    pedicel_bend: float = 0.0
    """How strongly this pedicel curves rather than running straight. See
    Inflorescence.pedicel_bend."""
    pedicel_width: float = 0.3
    """This stalk's own radius where it leaves the axis, as a fraction of that
    axis's radius. Stored per organ rather than read from the profile in the
    bridge, for the same reason `pedicel_length` is: it carries this flower's
    own place in the gradient and its own variance draw, and only growth knows
    those."""
    pedicel_droop_deg: float = 0.0
    """How far this pedicel has turned out of the way it was aimed by the time
    it reaches its flower, in degrees: **negative sags toward world down,
    positive lifts toward world up**, matching `elevation_deg`'s own sign. The
    flower's own weight is already folded in, so this is a resolved angle and
    not a factor. Set by `physics.apply_pedicel_droop` from the flower's weight;
    `resolve_geometry` only applies it, which keeps the profile out of
    `model.py`."""
    catkin_length: float = 0.0
    """Length of the catkin body hanging past the end of its stalk, in metres.
    **0 means this organ is not a catkin.**

    A catkin is modelled rather than instanced, and the shape it needs is
    exactly the one `pedicel_polyline` already produces: a curve that sags
    under weight, keeps the length it was given, and stops at vertical. So a
    catkin *is* its own pedicel, continued -- one hanging polyline, thin over
    `pedicel_length` and swollen over this, written into the skeleton mesh and
    swept by the same `Curve to Mesh` that makes the stems.

    This is what a supplied mesh could not do. An instanced catkin is rigid: it
    can be rotated to point downward but it cannot hang, cannot curve, and
    cannot respond to the branch above it moving. Built as a curve it gets all
    three from machinery that already exists."""
    catkin_bodies: list = field(default_factory=list)
    """The catkin bodies hanging off this organ's shared stalk, each a polyline
    in world space. `pedicel_points` holds the stalk they all fork from.

    A list because **a bunch is one pedicel carrying several catkins**, which is
    what hazel, birch and willow all do -- see `catkin_bunch`."""
    catkin_count: int = 3
    """How many catkins hang from this one stalk."""
    catkin_droop: float = 0.0
    """How far this bunch's stalk swings toward straight down, 0 to 1 -- a
    fraction of the turn *this* stalk needs, so 1 is vertical for every stalk
    whatever angle its twig left at. Its own field rather than the flower
    pedicel's `pedicel_droop_deg`, which is a resolved angle in degrees and
    means something different."""
    catkin_radii: list = field(default_factory=list)
    """Tube radius at every vertex of every body in `catkin_bodies`.

    Carried rather than recomputed downstream: the vertices are unevenly
    spaced, so anything deriving a radius from a vertex *index* gets the cap
    boundary wrong and flattens the rounded end. See `catkin_bunch`."""
    catkin_attach: list = field(default_factory=list)
    """Which vertex of the stalk each body in `catkin_bodies` forks from.
    The mesh has to chain from these, or every catkin trails a strand back to
    the stalk's tip -- see `catkin_bunch`."""
    catkin_length_variation: float = 0.0
    """Length scatter between the catkins on one stalk, as a fraction."""
    catkin_noise: float = 0.0
    """How far each catkin bends out of true, as a fraction of its own width.
    Subtle by design: what wants breaking up is the shape, so a bunch does not
    read as several copies of one tube."""
    catkin_width: float = 0.004
    """Radius of the catkin body, in metres. The body is a capsule -- constant
    through the middle with a hemisphere at each end -- which is what makes the
    ends read as rounded. See `catkin_radius`."""
    yaw_deg: float = 0.0
    """Swing left or right within the organ's own plane, about its face
    normal. The blade still lies flat; it just no longer points straight out
    from the node."""
    roll_deg: float = 0.0
    """Rotation about the organ's own length axis -- a blade turning its face
    away from flat, pivoting on its midrib."""
    pitch_deg: float = 0.0
    """Which way a flower faces, stated exactly like `elevation_deg` states
    where its stalk goes: 0 straight out sideways, +90 flat up the axis, -90
    flat down.

    Nothing to do with `elevation_deg` itself, which aims the *stalk*: this
    turns the flower on the end of it, so a hanging foxglove bell and an
    outward-facing delphinium can share one stalk geometry. A flower whose
    pitch equals its stalk angle points straight along its stalk.

    **Flowers only** -- see `resolve_geometry`. It was a relative tilt applied
    on top of wherever the stalk pointed until 2026-08-15, which meant 0 was
    "along the stalk" rather than "sideways", and the same number produced a
    different facing depending on the stalk angle beside it."""
    bloom_start: float = 0.0
    """Where this flower sits on the inflorescence's own bloom timeline, 0 to
    1. Baked at creation from bloom order and position so the Blender side can
    sweep a travelling open-band across the whole plant by moving one number,
    without regrowing anything -- the parametric add-on proved that a per-frame rebuild
    is far too slow to scrub or play back."""
    form: int = 0
    """Which leaf mesh this organ draws from: **0 foliage leaf, 2 bract** (1 was
    the adult heteroblasty form, removed). Each has its own asset collection.

    A bract is a bract because of *where* it sits, not when it was made, so
    `development._add_bracts` writes form 2 outright at a flowering node.

    A bract belongs on this axis rather than on `kind` because botanically it
    **is** a modified leaf, and because everything downstream of the leaf/flower
    split is right for it unchanged. Only the mesh differs, which is exactly what
    this field selects."""
    map_factors: dict | None = field(default=None, repr=False, compare=False)
    """A flower's inflorescence-map factors (`growth.maps.FLOWER_TARGETS`),
    by target, read once at creation -- so re-reading its gradients for a
    truncated region (`development._renormalise_truncated_regions`) scales
    them by the same factors. None without such a map."""

    # Blade shape (curl/cup/gravity/wave/twist) was dropped from the sim add-on:
    # leaves arrive pre-shaped from the leaf tool. A leaf is an opaque
    # user-supplied mesh; this module authors its placement and size, never its
    # vertices.
    pedicel_points: list = field(default_factory=list)
    """The stalk as a resolved polyline from node to flower, derived by
    `resolve_geometry`. Empty for a straight or absent pedicel, in which case
    the bridge draws a single edge."""
    tip_joined: bool = False
    """This stalk ends its axis and continues it (`_tip_join`), derived by
    `resolve_geometry`: the bridge tapers its width down from the stem's rather
    than starting at the pedicel's own, so the stem runs on into the flower."""

    # Derived by resolve_geometry() / the development pass; never authored.
    scale: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    """Face normal of the organ's own plane -- perpendicular to both the organ
    and the stem carrying it, so the stem's direction lies *in* that plane.

    A direction alone cannot orient a leaf. It fixes which way the blade
    points and leaves the roll about that axis free, and the instancing node
    then picks a roll of its own, which comes out near-identical for every
    near-vertical leaf on the plant -- every blade facing the same way in world
    space regardless of which stem it grew on. This is the second axis that
    pins the frame down."""
    lifecycle: str = LifecycleState.ALIVE
    """Alive / dead-retained / shed, set by the post-growth lifecycle pass. A
    flower's *natural* death is the bloom "spent" render instead; this only goes
    non-alive on an organ when its axis dies and the cascade takes it."""

    def age_steps(self, current_step: int) -> int:
        return max(0, current_step - self.birth_step)


@dataclass
class Bud:
    """A meristem. First-class and stateful, unlike the previous build's
    `bud_fate` flag on a bare position -- a bud needs to remember where it is
    on its parent (`divergence_deg`), when it appeared, and what it became,
    and a parallel array of strings could not carry that."""

    id: int
    position_kind: str
    birth_step: int
    axis_id: int                 # the axis this bud belongs to
    phytomer_index: int          # -1 for a terminal bud (it caps the axis, not a node)

    def age_steps(self, now: int) -> int:
        """How long this meristem has existed, in developmental steps. What
        vegetativeness declines against -- a lateral that starts vegetative and
        flowers later is a panicle, and "later" has to be measured against the
        bud's own clock rather than the plant's."""
        return max(0, now - self.birth_step)

    state: str = BudState.DORMANT
    fate: str = BudFate.NONE
    divergence_deg: float = 0.0
    elevation_deg: float = 45.0
    child_axis_id: int = -1      # set when fate becomes SHOOT
    released_step: int = -1      # the step it became a shoot / flower
    along_internode: float = 1.0
    """Where along its own internode this bud sits, 0 at the base and 1 at the
    node above it. Only `BudPosition.INTERNODAL` uses anything but 1 -- every
    other meristem sits at a node by definition."""
    release_threshold: float = 0.0
    """The bud's own intrinsic reluctance to break dormancy, in [0, 1], drawn
    once at creation from a stream seeded on this bud's id. Development
    releases the bud when the branching-onset curve rises past it, rather than
    re-rolling a die every step. Two reasons: a bud can then never un-release
    (so phytomer count stays monotonic in Age, assertion 4), and the result is
    independent of the order buds happen to be visited in, which is what makes
    determinism hold without threading a shared RNG through the loop."""

    level: int = -1
    """The branching level its branch would be (an index into
    `profile.levels`), for a branch bud; -1 for any other bud, whose branch
    level is its axis's first child level (`development._bud_level`)."""
    masked: bool = False
    """Its branch's level Masks rejected it: it stays dormant for good and is
    never evaluated again."""
    mask_windows: tuple = ()
    """Each of its branch level's mask entries' own window at this bud when it
    was judged (-1 for a muted entry) -- what the mask preview shows. Empty
    when its level has no masks. Diagnostic: nothing reads it back."""

    blocked_by: str = BudBlock.NONE
    """Which gate last held this bud dormant -- see `BudBlock`."""
    blocked_value: float = 0.0
    blocked_limit: float = 0.0
    """The two sides of that gate's own comparison, so the readout can say how
    far off the bud was rather than only which rule stopped it. What they mean
    depends on the gate (a threshold against a density, a position against a
    zone, a node count against a reach), which is why they are two bare
    numbers here and are phrased per gate where they are shown."""

    quota_selected: bool = False
    """Retired with WS4's branch quota (every node branches now); never set."""

    # Derived.
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class Phytomer:
    """One internode plus the node at its **distal** end.

    The distal convention is worth stating because it is arbitrary but load-
    bearing: leaves and axillary buds attach at `tip_position`, so the first
    phytomer of an axis puts its node one internode above the axis base, and
    an axis of N phytomers has N nodes. Anything reading this model
    (diameter, meshing, physics) assumes it.
    """

    index: int                   # position along its own axis, 0 at the base
    birth_step: int
    final_length: float
    direction: np.ndarray
    reference: np.ndarray        # parallel-transported frame vector, perpendicular to direction
    factor: float = 0.0
    """`index` normalised against the axis's expected final node count -- 0 at
    the base, 1 at the tip. The axis-relative coordinate every profile is
    stated in: internode length already varies along it, organ placement is a
    window in it, and the size-along-parent curves will read it too.

    Stored rather than recomputed by each consumer, because the expected node
    count is a development-time quantity -- an axis that has stopped growing no
    longer knows what it once expected. Two consumers deriving it separately
    had already drifted: internode length measured against a tonic-scaled count
    and branch placement against a raw one."""
    organs: list[Organ] = field(default_factory=list)
    """Leaves placed structurally at this node, plus any flower a released
    axillary bud turned into. One list rather than two because they share
    every downstream behaviour -- placement off the same frame, maturation on
    their own clock, instancing in the GN bridge -- and `kind` already
    distinguishes them."""
    buds: list[Bud] = field(default_factory=list)

    # Derived by resolve_geometry(); never authored directly.
    length: float = 0.0          # final_length scaled by the elongation curve
    radius: float = 0.0          # filled by the pipe-model pass (Phase C)
    debug_curvature: float = 0.0
    debug_load: float = 0.0
    debug_stiffness: float = 0.0
    debug_moment: float = 0.0
    """The beam solver's own per-node readouts, written by `physics.apply_gravity`
    purely so the four gravity debug views can shade what one beam did to every
    organ -- nothing reads them back. `debug_curvature` is the *realized* turn
    per unit length (deg/m), so a joint held by the never-kink cap reads lower
    here than its moment alone would imply; `debug_load` is the mass carried
    beyond this node (kg); `debug_stiffness` is the local Young's modulus after
    the stiffness profile and per-axis variance (Pa); `debug_moment` is the
    bending moment `g * load * arm` (N*m). A split arc point inherits its
    parent's four, so a subdivided bend shades as one continuous arc."""
    is_split: bool = False
    """A geometric arc point inserted by `physics.smooth_bent_arcs`, not a
    botanical internode. It carries a position, a radius and a share of a bend,
    so it meshes and shades like any node -- but it bears no organ and holds
    none of its parent's identity, so anything counting real phytomers (a leaf
    on every leaf-bearing node, an organ per flowering node) must skip it. The
    last part of a split keeps the original node's identity and is not marked;
    only the intermediate arc points are."""
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tip_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lying: bool = False
    """Set by `physics.apply_gravity`: this internode lies on the ground, rooted
    there, so `physics.settle_on_ground` lays it back along the ground after
    the arc smoothing -- down a falling slope too. A split part takes its
    parent's (the parts of a steep climb rise, but the stem still lies)."""

    def age_steps(self, current_step: int) -> int:
        return max(0, current_step - self.birth_step)


@dataclass
class Axis:
    """A continuous chain of phytomers sharing one apical meristem.

    `order` is real botanical branch order: 0 for a basal axis (the plant may
    have several -- see Plant.basal_axis_ids for the multi-stem case), and
    parent order + 1 for an axis grown from an axillary bud. A sympodial
    re-leader deliberately inherits its parent's order instead, because it
    *is* the axis continuing.
    """

    id: int
    order: int
    birth_step: int
    parent_axis_id: int = -1
    parent_phytomer_index: int = -1  # where on the parent axis this one attached
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    base_reference: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    parent_along_internode: float = 1.0
    """Where on its parent's internode this axis is attached, 0 at the node
    below and 1 at the node above. Only an `INTERNODAL` inflorescence uses
    anything but 1.

    Stored on the axis rather than only on the bud because **positions are
    derived, not kept**: `resolve_geometry` rebuilds every axis base from its
    parent's tip on each pass, so an offset applied once at release would be
    thrown away the first time the plant resolved."""
    orientation_target_deg: float = -1.0
    """Where this axis's gravitropic setpoint is migrating to as it fills with
    flowers, or **-1 for not migrating at all**, which is every axis unless its
    inflorescence asks for `OrientationMode.DYNAMIC`. Paired with
    `orientation_from_deg`; `development` interpolates between them."""
    orientation_from_deg: float = 0.0
    gravitropism: float = -1.0
    """How hard this axis corrects back toward `setpoint_tilt_deg`, or **-1 to
    use the species' own `gravitropism`** -- which every axis does unless its
    inflorescence states an `OrientationMode` that says otherwise. A per-axis
    value rather than a profile one because orientation is a property of the
    inflorescence, and one plant can carry more than one."""
    setpoint_tilt_deg: float = 0.0
    """The angle from vertical this axis *maintains* as it grows -- its
    gravitropic setpoint angle. A main shoot's is near zero (orthotropic, it
    grows up); a lateral's is the angle it launched at, so it holds that angle
    rather than straightening.

    This is the difference between a plant and a bundle of parallel sticks.
    Steering every axis toward world up, which is the naive reading of
    "gravitropism", collapsed a 132-axis plant to 0.99 m tall and 0.11 m wide
    -- every branch curved back to vertical within two phytomers of launching.
    """
    is_fork: bool = False
    """Whether this axis is a vegetative **fork** child (dichotomy) rather than
    an inflorescence branch. Both share `structural_depth` -- WS2's Q8 chose one
    depth counter for both self-similar structures -- so this discriminates them
    for the three readers that are genuinely inflorescence-specific
    (`_flowering_target`, and the two branches of `diameter.py`): a fork is an
    ordinary vegetative stem, sized and lengthened like one, and must not take
    an inflorescence branch's node count or width."""
    level: int = -1
    """The branching level this axis grows by (an index into `profile.levels`):
    set on stems, branches and forks; -1 on an inflorescence branch, and on an
    axis made outside the level machinery, which then falls back to its order
    (`development._level`)."""
    node_budget: int = -1
    """The node count this axis builds when it is a **fork** child (its level's
    Fork): its parent's remaining Steps, so a fork splits a stem's growth
    instead of adding a fresh full-length stem. -1 -- every other axis -- builds
    its level's Steps."""
    emergence_shift: float = 0.0
    """Steps this axis is SHOWN later than it grows (Emergence Delay): its own
    level's delay plus its parent's shift, a fork / relay / inflorescence
    branch its parent's. Growth never reads it; `development._reveal` does."""
    node_reach: float = -1.0
    """A fork / relay child's share of a chain cut short by Growth %: how many
    of its `node_budget` nodes it builds, fractional (`development._vegetative_reach`).
    -1 -- an uncut chain, or an axis that is no child -- builds them all."""
    hidden_nodes: int = 0
    """Nodes the display left off this axis's tip: grown already, not yet
    shown at this Day (`development._trim_unborn` / `_reveal`). While any
    are, the last node shown is not the axis's tip, so no flower on it is the
    tip flower (`resolve_geometry`) -- else each new flower of a preformed
    raceme stood straight up for its first frame and then dropped."""
    fork_checked: int = 0
    """How many of this axis's nodes the chance-fork test has already seen, so
    each node is rolled exactly once however many steps it lives through."""
    vigour: float = 1.0
    """This axis's base width relative to what its level gives it: below 1 on
    an Unequal Fork's weak forks. Not inherited
    -- a child's width already starts from this axis's thinner radius."""
    unit_ended: bool = False
    """Whether this sympodial unit has ended (relayed and flowered / aborted),
    so it ends exactly once."""
    is_continuation: bool = False
    """Whether this axis is its parent's **continuation**: grown on from the
    parent's tip, in line, as the next level down
    -- so it starts at exactly the parent's tip radius."""
    displace_deg: float = 0.0
    displace_normal: np.ndarray | None = None
    """The Aligned Relay's spent apex: its first flowering node turns this far
    about this normal, pushed aside for the relay that continues in line."""
    structural_depth: int = -1
    """How deep inside a self-similar structure this axis sits -- an
    **inflorescence** (`is_fork` False) or a vegetative **fork** (`is_fork`
    True) -- counted from that structure's base: `-1` while still an ordinary
    vegetative axis, `0` for the axis that began the structure, and parent + 1
    for each branch/fork it puts out. An inflorescence is bounded by its own
    `branching_depth`, a fork by the Steps its chain has left.

    **Separate from `order` on purpose, and the tree case is why.** `order` is
    the plant's architecture: trunk, scaffold, twig. An inflorescence is a
    *bounded, modified shoot system* that begins where vegetative growth ends,
    and its internal branching has nothing to do with how far from the root it
    happens to sit. A perennial hides the difference -- both counters are small
    and start together -- but a tree carrying panicles has five orders of wood
    under two orders of inflorescence, and one number cannot say both: keyed on
    `order`, a panicle on a twig could only ever be a single order deep, which
    is a raceme.

    So an inflorescence branch **inherits its parent's `order`** rather than
    deepening it (the same call a sympodial re-leader already makes, and for
    the same reason -- it is not a subordinate lateral). That is what keeps
    the branching levels purely vegetative, and what makes a panicle behave
    identically whether it sits on a perennial at order 0 or a tree twig at
    order 5."""
    is_cyme_relay: bool = False
    """Whether this axis is a **cyme unit**, relayed from its parent's spent tip.
    The units of a cyme read as one stem, so the inflorescence's taper runs
    along the whole chain rather than restarting in each unit
    (`diameter.compute_radii`)."""
    relay_azimuth_deg: float = 0.0
    """The azimuth this axis was launched at as a **cyme relay**, in its
    parent's transported frame.

    Carried rather than recomputed so that `relay_rotation_deg` accumulates
    down the chain: a helicoid cyme is "the same azimuth every generation" and
    a scorpioid is "180 more than last time", and neither can be stated without
    remembering what last time was. Because the frame is parallel-transported,
    holding the number constant genuinely means the same side of the stem
    rather than drifting -- which is what makes a helicoid coil.

    Meaningless on any axis that is not a relay."""
    inflorescence_scale: float = 1.0
    """`branch_length_profile` read once, where this inflorescence branch
    attaches to its parent. Cached at birth rather than looked up, for the same
    reason `_expected_phytomers` is fixed for an axis's whole life: a branch
    whose target length changed as its parent grew would re-scale internodes
    that are already drawn."""
    inflorescence_width: float = 1.0
    inflorescence_droop: float = 0.0
    inflorescence_length: float = 1.0
    inflorescence_bend: float = 0.0
    """`branch_width` and `branch_droop` with this branch's own variance already
    drawn in. Cached at birth because the stages that read them --
    `diameter.compute_radii` and `physics.apply_gravity` -- take no seed, and
    threading one through two signatures to re-draw a value that cannot change
    would be worse than storing it."""
    growth: float = 1.0
    """This axis's Growth %: its level's x the level's `growth` map, 0.01-1,
    fixed at birth (`development._new_axis`) -- the part of its Steps it builds."""
    map_gravitropism: float = 1.0
    map_width: float = 1.0
    """The growth-map multipliers (`growth/maps.py`) on this axis's gravitropic
    strength (`gravitropism`) and base width (`start_width`), evaluated once
    at the axis's birth and held for its whole life. Fixed rather than re-read
    for the same reason `_expected_phytomers` is: a lateral whose target length
    changed as the plant grew would re-scale internodes already drawn and could
    flip between flowering and not. 1.0 -- every axis on a plant with no maps --
    is an exact no-op (x * 1.0 == x), so the build stays byte-identical."""
    flowering_from: int = -1
    """Index of the first phytomer belonging to this axis's inflorescence, or
    -1 while the axis is still vegetative. Nodes at or past it bear flowers
    instead of leaves and vegetative buds -- which is what an inflorescence
    botanically is, a shoot whose phytomers changed what they carry."""
    internode_base: float = -1.0
    """This axis's nominal first internode, in metres, from its branching level
   : the parent's nominal internode where it
    attaches x the level's Start Length (on a stem, Internode Length x Start
    Length). Node i is this x Length / Step ** i. Fixed at birth; -1 on an axis
    no level drives (an inflorescence branch)."""
    phyllotaxis_offset_deg: float = 0.0
    """Where this axis's phyllotaxis spiral starts, measured in its own frame.

    Without it every axis begins at index 0 and therefore at divergence 0, so
    every lateral's first leaf and bud point the same way relative to its
    parent. Across the hundreds of axes in one plant that biases the whole
    bud distribution measurably -- one 45-degree octant carried 1.5x the buds
    of another, chi-square 81.8 against uniform.

    Botanically this is the prophyll position: a lateral shoot's first leaf is
    oriented relative to the leaf that subtends it, not to some global zero."""
    growth_credit: float = 0.0
    """Fractional progress toward the next phytomer. Each step an axis earns
    `1 / plastochron` credit and spends 1.0 to build a node (it earned
    `apical_control ** order / plastochron` until the branching overhaul, which
    is what the note below is about).

    This replaces a `step % plastochron == 0` test, which could only express
    integer ratios: apical control of 0.62 became a plastochron of
    round(1/0.62) = 2, i.e. half rate rather than 62%, and every value between
    0.4 and 0.67 collapsed to the same behaviour."""
    phytomers: list[Phytomer] = field(default_factory=list)
    apex: Bud | None = None
    terminal_organ: Organ | None = None  # a determinate axis's terminal flower
    lifecycle: str = LifecycleState.ALIVE
    """Alive / dead-retained / shed for the whole axis, set by the post-growth
    lifecycle pass. Death cascades: a dead or shed axis takes its child axes and
    all their organs with it (you prune a limb, not one internode)."""

    inflorescence: int = -1
    """Which inflorescence this axis is (`MorphologyProfile.inflorescence_at`),
    -1 while undecided: a vegetative axis then reads its level's tip
    claimant (`development._inf_index`). Stored when the axis begins an
    inflorescence or is released as a stalk at a node; an inflorescence
    branch reads its parent's."""
    stalk_base_radius: float = -1.0
    """The radius a stalk in this axis's flowering region takes its Pedicel
    Width of: the rachis's where the region starts. Set where a level axis's
    rachis tapers down to its top stalk's width (`development._level_radii`),
    so a stalk read against the local radius would thin twice; -1 reads the
    node's own radius (`stalk_radius`)."""

    def stalk_radius(self, index: int) -> float:
        """The radius a stalk (pedicel, catkin stalk) at node `index` is a
        Pedicel Width fraction of."""
        if self.stalk_base_radius > 0.0 and 0 <= self.flowering_from <= index:
            return self.stalk_base_radius
        return self.phytomers[index].radius

    @property
    def is_growing(self) -> bool:
        return self.apex is not None and self.apex.state == BudState.ACTIVE

    @property
    def is_terminated(self) -> bool:
        """Every axis must end up here by Age = 1.0. The previous build had no
        such guarantee -- 93 buds finished a run still 'active' but unable to
        grow, so they were neither drawn nor capped with a flower. Assertion 3
        in tests/structural.py exists to keep that from recurring."""
        return not self.is_growing

    @property
    def length(self) -> float:
        return sum(p.length for p in self.phytomers)

    def tip_position(self) -> np.ndarray:
        if not self.phytomers:
            return self.base_position
        return self.phytomers[-1].tip_position


@dataclass
class Plant:
    """The whole structure at one developmental step.

    `step` is the discrete developmental clock the growth loop and every bud's
    age are measured against -- integer, which keeps determinism trivial. Topology
    (which nodes exist, which buds are active) is decided in integer steps.

    `continuous_step` is the *fractional* position within that clock, used only by
    `resolve_geometry` to elongate the growing tip in smoothly instead of a node
    popping at each integer boundary (calendar plan phase 2). It defaults to -1
    meaning "unset -- use the integer `step`", so any caller that does not set it
    (and every pre-phase-2 path) is byte-identical. `grow` sets it to
    `age * total_steps`; at an integer value it equals `step`, so nothing changes
    at whole steps -- only the sub-step elongation of the newest node.
    """

    step: int = 0
    continuous_step: float = -1.0
    axes: dict[int, Axis] = field(default_factory=dict)
    basal_axis_ids: list[int] = field(default_factory=list)

    _next_axis_id: int = 0
    _next_bud_id: int = 0
    _next_organ_id: int = 0

    # -- id allocation ----------------------------------------------------
    def new_axis_id(self) -> int:
        axis_id = self._next_axis_id
        self._next_axis_id += 1
        return axis_id

    def new_bud_id(self) -> int:
        bud_id = self._next_bud_id
        self._next_bud_id += 1
        return bud_id

    def new_organ_id(self) -> int:
        organ_id = self._next_organ_id
        self._next_organ_id += 1
        return organ_id

    # -- structure --------------------------------------------------------
    def add_axis(self, axis: Axis) -> Axis:
        self.axes[axis.id] = axis
        if axis.parent_axis_id < 0:
            self.basal_axis_ids.append(axis.id)
        return axis

    def children_of(self, axis_id: int) -> list[Axis]:
        return [a for a in self.axes.values() if a.parent_axis_id == axis_id]

    def axes_in_topological_order(self) -> list[Axis]:
        """Parents before children. `resolve_geometry` needs this because a
        child axis's base position is read off its parent's already-resolved
        phytomer."""
        ordered: list[Axis] = []
        stack = [self.axes[i] for i in reversed(self.basal_axis_ids)]
        while stack:
            axis = stack.pop()
            ordered.append(axis)
            stack.extend(reversed(self.children_of(axis.id)))
        return ordered

    def growing_axes(self) -> list[Axis]:
        return [a for a in self.axes.values() if a.is_growing]

    def dormant_buds(self) -> list[Bud]:
        return [
            bud
            for axis in self.axes.values()
            for phytomer in axis.phytomers
            for bud in phytomer.buds
            if bud.state == BudState.DORMANT
        ]

    def all_phytomers(self) -> list[Phytomer]:
        return [p for axis in self.axes.values() for p in axis.phytomers]

    def all_organs(self) -> list[Organ]:
        organs = [
            organ
            for axis in self.axes.values()
            for phytomer in axis.phytomers
            for organ in phytomer.organs
        ]
        organs.extend(a.terminal_organ for a in self.axes.values() if a.terminal_organ is not None)
        return organs

    def organs_of_kind(self, kind: str) -> list[Organ]:
        return [o for o in self.all_organs() if o.kind == kind]

    @property
    def phytomer_count(self) -> int:
        return sum(len(a.phytomers) for a in self.axes.values())

    @property
    def max_order(self) -> int:
        return max((a.order for a in self.axes.values()), default=0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        points = [p.tip_position for p in self.all_phytomers()]
        points.extend(self.axes[i].base_position for i in self.basal_axis_ids)
        stacked = np.array(points)
        return stacked.min(axis=0), stacked.max(axis=0)


def _organ_expansion(organ, expansion, step: int) -> float:
    """How far this organ has inflated, 0 to 1. Always 1 for a flower -- a
    flower is sized by its node's elongation instead (`resolve_geometry`). A
    fruit takes its own `fruit_size` outright (no swell), so it is kept full.

    **Why a flower follows its node now, reversing "a flower's immaturity is
    its bloom stage, never its size".** That rule was made when a raceme laid
    its flowers down one per step, so at Age 1.0 66% of them were still
    swelling (mean 0.61, minimum 0.04) and size stopped meaning what Flower
    Size said. A raceme is now laid down whole at onset
    (`development._preform_region`) and its nodes elongate together, so a
    raceme started in good time is full-size throughout -- and one started at
    the very end of the season is genuinely young: compact, and, read off the
    same clock, with short thin stalks and small buds rather than full-size
    flowers crammed together."""
    if expansion is None or organ.kind in (OrganKind.FLOWER, OrganKind.FRUIT):
        return 1.0
    return float(expansion(organ.age_steps(step)))


def _catkin_geometry(organ, node: np.ndarray) -> tuple:
    """`catkin_bunch` for one organ at its node, remembered on the organ by its
    exact inputs: a resolve that moved nothing under it (the second flat-head
    pass) takes copies of the last build instead of re-sweeping every catkin
    point by point -- the bulk of a catkin-laden plant's build."""
    launch = organ.direction
    key = (tuple(float(v) for v in node), tuple(float(v) for v in launch),
           organ.pedicel_length, organ.catkin_length, max(1, organ.catkin_count),
           organ.catkin_droop, organ.catkin_width, organ.catkin_noise, organ.id,
           organ.catkin_length_variation)
    memo = getattr(organ, "_catkin_memo", None)
    if memo is None or memo[0] != key:
        memo = (key, catkin_bunch(node, launch, organ.pedicel_length, organ.catkin_length,
                                  max(1, organ.catkin_count), organ.catkin_droop,
                                  organ.catkin_width, organ.catkin_noise, organ.id,
                                  organ.catkin_length_variation))
        organ._catkin_memo = memo
    stalk, bodies, attach, radii = memo[1]
    return ([p.copy() for p in stalk], [[p.copy() for p in body] for body in bodies],
            list(attach), [list(r) for r in radii])


def resolve_geometry(plant: Plant, elongation=None, expansion=None,
                     development=None, catkins: bool = True) -> None:
    """Derive every world position from the stored direction/length chain.

    Called once at the end of each growth step (and again after Stage 4's
    physics pass, which only rewrites directions and lets this recompute the
    consequences). `elongation` and `expansion` are optional callables of
    element age in steps returning a 0-1 fraction -- the growth loop passes
    real curves; callers that just want positions for a hand-built structure
    can leave them None, which treats everything as fully grown.
    `development(axis, fraction)` maps a flowering node's elongation fraction
    to how developed its inflorescence reads (`Inflorescence.development_size`);
    None uses the fraction as it is. `catkins` False leaves every catkin bunch
    unswept, sitting at its node (`build` before the catkin droop is known --
    nothing reads a catkin's shape until then).

    **`expansion` applies to leaves and not to flowers**, which is a real
    distinction rather than an omission. It exists to show an organ inflating
    as the plant matures, and for a leaf that is the only cue there is. A
    flower has a better one: the bloom stage system renders an immature flower
    as an actual *bud mesh*, so shrinking it as well double-counts -- and
    shrinking is the worse of the two, because a bud is not a small open
    flower. Measured before this split: at Age 1.0, 66% of a raceme's flowers
    were still below full size (mean 0.61, minimum 0.04), which silently
    overrode whatever Flower Size and Pedicel Length said. Those two controls now
    own final flower size outright, once the flower's node has elongated -- see
    `_organ_expansion` for the node clock a flower does follow.
    """
    # The clock elongation/expansion is measured against: the fractional
    # `continuous_step` when set (phase 2 -- the growing tip elongates in smoothly
    # between whole steps), else the integer `step` (byte-identical to before).
    ref_step = plant.step if plant.continuous_step < 0.0 else plant.continuous_step
    for axis in plant.axes_in_topological_order():
        # A child axis hangs off a parent phytomer that is already resolved,
        # because axes_in_topological_order guarantees parents come first.
        if axis.parent_axis_id >= 0:
            parent = plant.axes[axis.parent_axis_id]
            if 0 <= axis.parent_phytomer_index < len(parent.phytomers):
                attach = parent.phytomers[axis.parent_phytomer_index]
                axis.base_position = attach.tip_position
                if axis.parent_along_internode < 1.0:
                    # An internodal attachment sits partway up its parent's
                    # internode rather than on the node above it.
                    axis.base_position = (
                        attach.base_position
                        + (attach.tip_position - attach.base_position)
                        * axis.parent_along_internode)
                elif axis.parent_along_internode > 1.0:
                    # Slid up past its node: partway up the internode above
                    # (on along this one's heading at the axis's last node).
                    k = axis.parent_phytomer_index + 1
                    nxt = parent.phytomers[k] if k < len(parent.phytomers) else attach
                    axis.base_position = (
                        attach.tip_position
                        + (nxt.tip_position - nxt.base_position)
                        * (axis.parent_along_internode - 1.0))

        cursor = np.array(axis.base_position, dtype=float)
        # The node whose stalks get `_tip_join`: the axis's last, when it ends
        # in one or two stalks -- an elbow off the end of the stem. Three or more
        # radiating from the tip are a hub (an umbel, a tip whorl) whose spread
        # IS its shape, so those keep their ribs.
        # The tip node only when it is the axis's own last, not the last one
        # a Day shows (`Axis.hidden_nodes`).
        last_phytomer = (axis.phytomers[-1] if axis.phytomers and not axis.hidden_nodes
                         else None)
        tip_stalks = 0 if last_phytomer is None else sum(
            1 for o in last_phytomer.organs
            if o.pedicel_length > 0.0 and o.catkin_length <= 0.0)
        if tip_stalks > 2:
            last_phytomer = None
        for phytomer in axis.phytomers:
            fraction = (1.0 if elongation is None
                        else float(elongation(phytomer.age_steps(ref_step))))
            # A flowering node, and any flower, develops along the
            # inflorescence's own Young Size + growth curve rather than the raw
            # elongation fraction -- see `Inflorescence.young_size`.
            bloom = (fraction if development is None or fraction >= 1.0
                     else float(development(axis, fraction)))
            if 0 <= axis.flowering_from <= phytomer.index:
                fraction = bloom
            phytomer.length = phytomer.final_length * fraction
            phytomer.base_position = cursor.copy()
            cursor = cursor + phytomer.direction * phytomer.length
            phytomer.tip_position = cursor.copy()

            for organ in phytomer.organs:
                # A flower develops with the node carrying it: the same
                # elongation fraction that opens a young raceme's spacing sizes
                # its buds, and (through `scale / final_scale`) its stalk's
                # length here and its width in the bridge. See `_organ_expansion`
                # for why this is the node's clock and not a flower-own one.
                organ.scale = organ.final_scale * (
                    bloom if organ.kind == OrganKind.FLOWER
                    else _organ_expansion(organ, expansion, ref_step))
                if (tip_stalks == 1 and phytomer is last_phytomer
                        and organ.kind == OrganKind.FLOWER and organ.pedicel_length > 0.0):
                    # **The tip-most flower runs straight on from its stem.** The
                    # one flower ending an axis is its continuation, so it
                    # takes the axis heading rather than Pedicel Angle, which
                    # aims the side flowers. Curving the join in (`_tip_join`)
                    # removed the elbow but kept the side flowers' bend at the
                    # tip -- reported as still bending off. Droop still sags
                    # it. Two tip stalks keep their angles (and the curved
                    # join), or they would run up the same line.
                    organ.direction = np.array(phytomer.direction, dtype=float)
                else:
                    organ.direction = radial_direction(
                        phytomer.direction, phytomer.reference,
                        organ.divergence_deg, organ.elevation_deg)
                # A pedicel extends the organ away from its node along the
                # same direction, and grows with the organ rather than
                # springing out at full length.
                if organ.catkin_length > 0.0:
                    # A bunch of catkins on one shared stalk, grown as hanging
                    # curves rather than placed and oriented. Its geometry goes
                    # into the skeleton mesh, not the organs object. See
                    # `catkin_bunch` and `Organ.catkin_length`.
                    if not catkins:
                        # A pass that reads no catkin shape (`build` before the
                        # catkin droop is known): it sits at its node for now.
                        organ.position = phytomer.tip_position.copy()
                        continue
                    (organ.pedicel_points, organ.catkin_bodies,
                     organ.catkin_attach, organ.catkin_radii) = _catkin_geometry(
                        organ, phytomer.tip_position)
                    organ.position = (organ.pedicel_points[-1].copy()
                                      if organ.pedicel_points
                                      else phytomer.tip_position.copy())
                    continue
                stalk = organ.pedicel_length * (
                    organ.scale / organ.final_scale if organ.final_scale > 0 else 1.0
                )
                anchor = phytomer.tip_position
                if organ.along_internode < 1.0:
                    # partway down its internode (an Internodal solitary flower)
                    anchor = (phytomer.base_position + (phytomer.tip_position
                              - phytomer.base_position) * organ.along_internode)
                organ.position = anchor + organ.direction * stalk
                organ.tip_joined = stalk > 0.0 and phytomer is last_phytomer
                organ.pedicel_points = pedicel_polyline(
                    anchor, phytomer.direction, organ.position,
                    organ.pedicel_bend, organ.pedicel_droop_deg,
                    at_tip=organ.tip_joined,
                ) if stalk > 0.0 else []
                # **A pedicel decides where its flower goes and which way it
                # faces, and nothing else does.** Both were separately settable
                # until 2026-08-16 -- Pedicel Angle aimed the stalk and Flower
                # Pitch aimed the flower, in the same frame with the same sign
                # convention -- and any disagreement between them read as a
                # flower detached from the stalk holding it. A real pedicel is
                # rigid at the join; the bell of a foxglove hangs because its
                # *stalk* hangs, which is what `pedicel_droop_deg` now says.
                #
                # So the flower takes the tangent where the pedicel arrives,
                # not the chord from the node, which on a deep arc points
                # somewhere the pedicel never goes. Recomputed from
                # divergence/elevation at the top of the next resolve, so this
                # cannot accumulate.
                if stalk > 0.0:
                    # A straight pedicel needs nothing done: `organ.direction`
                    # is already the direction it runs in, set above. Only a
                    # curved or sagging one arrives pointing somewhere else.
                    if len(organ.pedicel_points) >= 2:
                        organ.position = organ.pedicel_points[-1].copy()
                        tangent = organ.pedicel_points[-1] - organ.pedicel_points[-2]
                        norm = float(np.linalg.norm(tangent))
                        if norm > _EPS:
                            organ.direction = tangent / norm
                elif organ.kind in (OrganKind.FLOWER, OrganKind.FRUIT):
                    # Sessile: no stalk to take a facing from, so Flower Pitch
                    # is the flower's own, stated exactly the way Pedicel Angle
                    # states a stalk's -- 0 straight out sideways, +90 flat up
                    # the axis, -90 flat down. This is the spike, the catkin
                    # and the head, and it is the only place Pitch now applies.
                    #
                    # Flowers only. A leaf has no separate facing: it is aimed
                    # by Leaf Angle through `elevation_deg` above, and reading a
                    # defaulted 0 here as "face sideways" would silently re-aim
                    # every leaf on the plant.
                    organ.direction = radial_direction(
                        phytomer.direction, phytomer.reference,
                        organ.divergence_deg, organ.pitch_deg
                    )
                organ.normal = organ_normal(phytomer.direction, organ.direction, phytomer.reference)
                organ.direction, organ.normal = orient_organ(
                    organ.direction, organ.normal, organ.yaw_deg, organ.roll_deg)
            for bud in phytomer.buds:
                bud.position = phytomer.tip_position.copy()
                bud.direction = radial_direction(
                    phytomer.direction, phytomer.reference, bud.divergence_deg, bud.elevation_deg
                )

        if axis.apex is not None:
            axis.apex.position = axis.tip_position()
            axis.apex.direction = (
                axis.phytomers[-1].direction if axis.phytomers else axis.base_direction
            ).copy()
        if axis.terminal_organ is not None:
            organ = axis.terminal_organ
            organ.direction = (
                axis.phytomers[-1].direction if axis.phytomers else axis.base_direction
            ).copy()
            if organ.elevation_deg != 90.0 and axis.phytomers:
                # Leaning off the tip (an Aligned Relay's spent apex): its
                # divergence / elevation in the tip's own frame.
                organ.direction = radial_direction(
                    organ.direction, axis.phytomers[-1].reference,
                    organ.divergence_deg, organ.elevation_deg)
            organ.scale = organ.final_scale * _organ_expansion(
                organ, expansion, plant.step)
            stalk = organ.pedicel_length * (
                organ.scale / organ.final_scale if organ.final_scale > 0 else 1.0
            )
            organ.position = axis.tip_position() + organ.direction * stalk
            reference = axis.phytomers[-1].reference if axis.phytomers else axis.base_reference
            organ.normal = organ_normal(organ.direction, organ.direction, reference)
            organ.direction, organ.normal = orient_organ(
                organ.direction, organ.normal, organ.yaw_deg, organ.roll_deg)


def _scaled(value, factor: float):
    """Scale a scalar, a vector, or a (possibly nested) list of either, by
    `factor` -- so one call handles a pedicel polyline (a list of points), a
    catkin bunch (a list of polylines) and its per-vertex radii (a list of
    lists) alike."""
    if isinstance(value, list):
        return [_scaled(v, factor) for v in value]
    return value * factor


def scale_plant(plant: Plant, factor: float) -> None:
    """Uniformly scale a fully-resolved plant's geometry about the origin
    (calendar phase 5 per-year vigour): every position, length, radius and organ
    size the bridge reads, so the shape, topology and phenology dates are
    untouched and only the size changes. Directions and reference frames are unit
    vectors and are left alone. A no-op at 1.0, so a year-0 or non-perennial
    plant is byte-identical.

    Runs last, on the finished geometry, which is the 'same shape, bigger' that
    the grill settled on (QI-a): the gravity bend keeps the shape it solved at
    the base size rather than being re-solved larger -- negligible at the subtle
    multipliers a perennial uses, and what 'same shape' means."""
    if factor == 1.0:
        return

    def scale_organ(o) -> None:
        o.position = o.position * factor
        o.scale *= factor
        o.final_scale *= factor
        o.pedicel_length *= factor
        o.catkin_length *= factor
        o.catkin_width *= factor
        o.pedicel_points = _scaled(o.pedicel_points, factor)
        o.catkin_bodies = _scaled(o.catkin_bodies, factor)
        o.catkin_radii = _scaled(o.catkin_radii, factor)

    for axis in plant.axes.values():
        axis.base_position = axis.base_position * factor
        for phy in axis.phytomers:
            phy.base_position = phy.base_position * factor
            phy.tip_position = phy.tip_position * factor
            phy.length *= factor
            phy.final_length *= factor
            phy.radius *= factor
            for organ in phy.organs:
                scale_organ(organ)
        if axis.terminal_organ is not None:
            scale_organ(axis.terminal_organ)


def organ_normal(
    stem_direction: np.ndarray, organ_direction: np.ndarray, fallback_reference: np.ndarray
) -> np.ndarray:
    """The organ plane's face normal: the stem's own direction, squared up
    against the organ so it is a true perpendicular. A leaf blade therefore
    faces *along* its stem and lies across it -- held out flat, the way a leaf
    presents itself to the light.

    This was `cross(stem, organ)` at first, which is the same frame turned 90
    degrees about the leaf's own length: it puts the blade's *plane* through
    the stem, standing every leaf on edge like a fin. Both are well-defined
    frames that follow the stem; only one of them is how a leaf sits.

    Degenerate when the organ points straight along its stem -- a terminal
    flower always does -- and then every plane through the axis is equally
    valid, so the phytomer's own transported reference is used. That keeps the
    frame continuous with the phyllotaxis rather than snapping to a world axis.
    """
    along = float(np.dot(stem_direction, organ_direction))
    normal = stem_direction - organ_direction * along
    norm = float(np.linalg.norm(normal))
    if norm > 1e-6:
        return normal / norm
    normal = np.cross(organ_direction, fallback_reference)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-6:
        return normal / norm
    return orthonormal_reference(organ_direction)


def orient_organ(
    direction: np.ndarray, normal: np.ndarray, yaw_deg: float, roll_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Turn an organ on its own two axes, after the stem has decided where it
    sits and which way its face points.

    Pitch is already spoken for -- it is the angle from the stem, authored as
    `elevation_deg` and shown in the panel as Leaf Angle -- so these are the
    other two of the three. Yaw swings the blade left or right *within* its own
    plane, turning about the face normal, and leaves it lying just as flat.
    Roll turns the blade about its own length, pivoting on the midrib, so the
    face stops being flat to the sky without the blade pointing anywhere new.

    Each rotation is about one of the frame's own axes and carries the other
    axis with it, so the pair comes back orthonormal by construction rather
    than by a re-normalisation afterwards. Applied here rather than in the
    Geometry Nodes group because the frame these turn is already the thing
    Python hands over: doing it in GN would mean rebuilding the same frame on
    the other side of the seam just to rotate it.
    """
    if yaw_deg:
        direction = rotate_about(direction, normal, yaw_deg)
    if roll_deg:
        normal = rotate_about(normal, direction, roll_deg)
    return direction, normal


_MIN_TUBE_RADIUS = 3e-5
_PEDICEL_SEGMENTS = 10

_PEDICEL_MAX_EXPONENT = 4.0
"""How steeply the deepest bend rises. The stalk climbs as `t ** exponent`
against a radius that runs out linearly, so this is the power in `height goes
as radius ** n` -- 5 at full bend, a stalk that runs out flat and turns up
hard."""


def pedicel_polyline(
    node: np.ndarray, axis_tangent: np.ndarray, tip: np.ndarray, bend: float,
    droop_deg: float = 0.0, at_tip: bool = False,
) -> list[np.ndarray]:
    """A curved stalk from `node` to `tip`, running out sideways and turning up
    into the flower -- the rib of an umbrella.

    Built in the stalk's own two axes rather than as a Bezier: radius from the
    stem runs out linearly with the parameter while height climbs as
    `t ** exponent`, so the whole path is `height ~ radius ** n`. `bend` sets
    that exponent, from 1 (the straight chord, exactly) up to
    `_PEDICEL_MAX_EXPONENT + 1`.

    Two properties this shape has that a Bezier did not, both of them things
    the Bezier version got wrong:

    * **It never bulges past the flower.** Radius is `reach * t`, so the widest
      the stalk ever gets is exactly where its own flower sits -- and since the
      flat-top solve caps that at `head_radius`, a head can never be wider than
      its stated radius. The Bezier put its outward handle a fraction of the
      *total* length out, which on a long corymb pedicel (tall climb, small
      radius) threw the arc several times the head radius sideways before
      bringing it back.
    * **It curves the whole way instead of bending once.** A cubic's second
      handle leaves a straight run into the endpoint as long as the handle
      itself -- nearly half the stalk. Here curvature falls off smoothly and
      only vanishes at the flower, which is also why uniform steps in `t` land
      most of the vertices around the bend and few on the near-vertical
      approach.

    Built from the node's own tangent rather than from world up, so a pedicel
    on a leaning branch curves relative to that branch -- the same reason the
    flat-top solve measures against the axis rather than world Z.

    `droop_deg` then turns the finished curve out of that plane: negative sags
    it toward world down, positive lifts it toward world up. It is applied
    *after* the shape rather than folded into it because the two are stated in
    different frames on purpose: bend lives in the pedicel's own plane and
    turns with the axis, droop is measured against gravity and does not.

    `at_tip` marks a stalk on its axis's last node, which gets `_tip_join`.
    """
    if bend <= 0.0 and droop_deg == 0.0:
        # A straight stalk is just its chord -- unless it ends its axis at a
        # steep angle, where the chord itself is the kink. See `_tip_join`.
        return _tip_join([node, tip], axis_tangent) if at_tip else []

    delta = tip - node
    along = float(np.dot(delta, axis_tangent))
    outward_vec = delta - axis_tangent * along
    reach = float(np.linalg.norm(outward_vec))
    if reach <= _EPS:
        if droop_deg == 0.0:
            # The flower sits on the axis itself: there is no plane for the
            # stalk to arc in, so a straight line is the only honest answer.
            return []
        # A drooping pedicel still needs vertices to turn, even with no arc.
        outward = np.zeros(3)
        exponent = 1.0
    else:
        outward = outward_vec / reach
        exponent = 1.0 + bend * _PEDICEL_MAX_EXPONENT

    points = []
    for i in range(_PEDICEL_SEGMENTS + 1):
        t = i / _PEDICEL_SEGMENTS
        points.append(node + outward * (reach * t) + axis_tangent * (along * t ** exponent))
    points = _sag(points, droop_deg)
    return _tip_join(points, axis_tangent) if at_tip else points


_TIP_JOIN_MIN_DEG = 1.0
"""Below this a tip stalk already leaves along its axis and is left alone."""


def _tip_join(points: list[np.ndarray], axis_tangent: np.ndarray) -> list[np.ndarray]:
    """Re-shape a stalk that ends its axis so it leaves **along** the axis
    heading and curves into its flower, keeping where the flower sits and which
    way it faces.

    **A stalk at the tip is a continuation, not a branch.** Mid-axis a stalk
    leaving square to the stem is fine -- the stem carries on past it, so the
    eye reads a side branch. At the tip the stem stops at that node, and every
    curved pedicel leaves *exactly* perpendicular (its curve starts flat out,
    `t ** exponent` has zero slope at 0), so the last flower on every raceme
    branch hung off a 90-degree elbow. Capping that at 60 degrees was tried
    first and still read as a stalk stuck onto a blunt stem end; what looks
    natural is the stem running on into the flower. So the stalk is rebuilt
    as a cubic from the node whose first tangent IS the axis heading (tangent
    continuous with the stem), and whose last tangent and end point are left
    as they were -- Pedicel Angle, Bend and Droop still decide where the
    flower goes and how it faces; only the join changes. The bridge tapers its
    width down from the stem's to match (`Organ.tip_joined`).
    """
    if len(points) < 2:
        return points
    node, end = points[0], points[-1]
    first = points[1] - node
    norm = float(np.linalg.norm(first))
    if norm <= _EPS:
        return points
    first = first / norm
    tangent = np.asarray(axis_tangent, dtype=float)
    angle = float(np.degrees(np.arccos(np.clip(float(np.dot(first, tangent)), -1.0, 1.0))))
    if angle <= _TIP_JOIN_MIN_DEG:
        return points
    start_dir = tangent / max(float(np.linalg.norm(tangent)), _EPS)
    last = end - points[-2]
    last_norm = float(np.linalg.norm(last))
    end_dir = last / last_norm if last_norm > _EPS else start_dir
    span = float(np.linalg.norm(end - node))
    if span <= _EPS:
        return points
    handle = span / 3.0
    p1 = node + start_dir * handle
    p2 = end - end_dir * handle
    count = max(len(points) - 1, _PEDICEL_SEGMENTS)
    out = []
    for i in range(count + 1):
        # Samples bunched toward both ends: the tube is swept through these
        # points, so the join the eye sees is the first *chord*, not the
        # tangent -- evenly spaced, that chord already leans ~10 degrees into
        # the turn. The far end is bunched for the same reason, since the
        # last chord is what the flower takes its facing from.
        t = 0.5 - 0.5 * np.cos(np.pi * i / count)
        u = 1.0 - t
        out.append(u ** 3 * node + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t ** 3 * end)
    return out


def _sag(points: list[np.ndarray], droop_deg: float,
         fractions: list[float] | None = None) -> list[np.ndarray]:
    """Turn a pedicel out of the plane it was built in: **negative sags it
    toward world down under its flower's weight, positive lifts it toward world
    up.**

    Sign follows `Inflorescence.pedicel_angle_deg`, where negative already hangs
    a stalk below horizontal and positive swings it up toward the tip. The two
    controls sit next to each other on the same organ and now agree, which is
    worth more than "droop" reading as a positive quantity.

    Done by rotating each segment and re-walking, not by displacing points:
    rotation preserves every segment's length, so a sagging pedicel keeps the
    length it was given instead of being quietly shortened by its own droop.
    The same reason `physics.apply_gravity` rotates the stem rather than
    offsetting it.

    Curvature falls off with the remaining moment arm, which is the textbook
    tip-loaded cantilever: the load is the flower at the far end, so the
    section nearest the node carries the whole moment and the section at the
    flower carries none. Integrating that gives a cumulative angle of
    `droop * t * (2 - t)`, reaching exactly `droop` at the tip -- so the number
    on the panel is the angle the flower ends up at, which is the one thing
    about it that has to be predictable.

    **No segment turns past vertical**, in either direction, which is a real
    stop rather than a safety clamp: gravity can hang a stalk straight down and
    no further, and past that a large enough flower folded its own pedicel back
    under itself and up the other side. It needs no separate tangent
    continuation the way the leaf Gravity deformer does -- once a segment lands
    on vertical the following ones clamp to vertical too, so the pedicel simply
    hangs straight from wherever it gave way, which is what a limp stalk does.
    The lifting direction stops the same way, at straight up.

    `fractions` overrides the per-segment position the droop profile is read at.
    Without it each segment is read at its own **index** fraction, which is
    correct only while the points are evenly spaced -- and every pedicel's are.
    A catkin's are not: its ends carry many more vertices than its middle so the
    rounded caps resolve, and read by index those crowded segments would take a
    quarter of the whole turn across six percent of the length, kinking the top.
    Passing arc-length fractions makes the bend depend on where a segment
    actually is rather than on how finely it happens to be divided. Defaulted to
    None so no existing stalk moves.
    """
    if droop_deg == 0.0 or len(points) < 2:
        return points
    node = points[0]
    # The bend axis comes from the chord, so the whole pedicel turns in one
    # plane rather than each segment picking its own and corkscrewing. Taken
    # against _UP rather than _DOWN, which is the whole of the sign flip: a
    # positive rotation about `chord x up` carries the chord toward up.
    axis = np.cross(points[-1] - node, _UP)
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        # Already pointing straight up (or straight down, where any plane is as
        # good as another and leaving it alone is the stable answer).
        return points
    axis = axis / norm

    out = [node]
    cursor = node.copy()
    count = len(points) - 1
    for i in range(count):
        segment = points[i + 1] - points[i]
        length = float(np.linalg.norm(segment))
        # Sampled at the segment's midpoint so the sum is the midpoint rule
        # rather than a half-segment biased to one end.
        t = (i + 0.5) / count if fractions is None else fractions[i]
        angle = droop_deg * t * (2.0 - t)
        if length > _EPS:
            # How far this segment can still turn before it points straight up
            # (positive) or straight down (negative). Measured per segment
            # rather than once off the chord, so a pedicel that has already
            # turned partway is clamped against where it actually is.
            to_up = float(np.degrees(np.arccos(
                np.clip(float(np.dot(segment / length, _UP)), -1.0, 1.0))))
            angle = min(angle, to_up) if angle > 0.0 else max(angle, to_up - 180.0)
        cursor = cursor + rotate_about(segment, axis, angle)
        out.append(cursor.copy())
    return out


def radial_direction(
    tangent: np.ndarray, reference: np.ndarray, divergence_deg: float, elevation_deg: float
) -> np.ndarray:
    """A unit direction `divergence_deg` around the stem from `reference`, then
    tilted `elevation_deg` up toward the stem tangent. Elevation 0 is straight
    out sideways; 90 is straight up the stem.

    Unlike the previous build's equivalent, `reference` is the phytomer's own
    transported frame vector, so consecutive nodes' divergence angles compose
    into a real spiral instead of each being measured from a fresh world axis.
    """
    radial = rotate_about(reference, tangent, divergence_deg)
    phi = np.radians(elevation_deg)
    direction = radial * np.cos(phi) + tangent * np.sin(phi)
    return direction / max(float(np.linalg.norm(direction)), _EPS)
