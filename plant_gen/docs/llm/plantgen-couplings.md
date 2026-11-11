# PlantGen — parameter couplings & rules

The rules a schema can't hold: how the parameters interact, what each subsystem
may and may not do, and how the important quantities are calculated. Read this
alongside `plantgen-spec.json` (the fields, ranges, and per-field endpoint
visuals) before authoring a preset. Every rule here is a property of the model,
not a guideline.

---

## 0. Scope — what you set vs. what the user provides

**You set the whole `MorphologyProfile`**: growth form, architecture,
phyllotaxis, placement, inflorescence kind and structure, all angles, densities,
curves, **sizes as multipliers**, blade deformers, cross-section, and the physics
set.

**The user provides the asset meshes** — the actual leaf-blade outline, the
flower / petal form, and bud / bract / catkin geometry — and does the final
tweaking. These meshes are not in the profile at all; they are scene-collection
references the add-on instances. So this boundary is the add-on's own seam.

- **Never try to author a blade shape or a petal form through parameters.** If a
  look needs a lobed leaf or a trumpet flower, say so plainly: *"model a lobed
  leaf ~6 cm and drop it in the Leaves collection; I set its placement, size
  multiplier, angle and arrangement."*
- Because size is a **multiplier on the mesh** (`1.0` = the mesh exactly as
  modelled), you set **relative proportions**, not absolute size. You may advise
  a rough real size to model to; you do not control it.

---

## 1. Size and mass — the square / cube rule

Size is *visual*; weight is *mass*. They are independent controls and you
normally set both.

- **Flower / catkin mass** ∝ (`flower_size` × `flower_size_profile` × bloom
  opening)³ × `flower_weight`.
- **Leaf mass** ∝ (`leaf_size` × `leaf_size_profile` × expansion)² ×
  `leaf_weight`.

**Consequence — a big flower nods hard.** Doubling `flower_size` makes a head
~8× heavier, so its stem bends much further. To make a large flower that does
*not* collapse its stem, lower `flower_weight`, or stiffen with
`stem_stiffness` / `rigid_base`. Leaf mass is small against stem stiffness on
most plants, so `leaf_weight` reads far more subtly than the flower's.

---

## 2. Thickness — the pipe model

Stem thickness is **derived**, not painted on. A cross-section is as thick as the
load it must carry (foliage + structure below it).

- **`tip_radius` sets the whole plant's proportional thickness.** Every thicker
  part scales from it, so raising it thickens everything together — it does not
  merely fatten the tips.
- `pipe_per_leaf` is how much each leaf adds to the stem it hangs on. A bare
  flowering stalk still tapers, as a load-bearing column.
- `pedicel_width` and `branch_width` (and inflorescence `branch_width`) are
  **fractions of the parent's thickness at the attachment point** (≤ 1). **A
  branch is never thicker than what carries it.**
- `radius_profile` shapes the taper from base (1.0) to tip (0.0).

---

## 3. The bend group — only meaningful together

Pipeline order: **grow topology → pipe-model radius → physics bend →
senescence.** The bend is a **post-growth deformation**: it changes the plant's
*shape*, never what is attached to what.

These all multiply, so any one can hide the others:

| Control | What it does |
|---|---|
| `droop` | overall self-weight bend of the whole plant |
| `stem_stiffness` | resistance (elastic modulus); thick parts resist as r⁴, so it mostly straightens tips and outer branches |
| `rigid_base` | *where* the bend happens — the fraction held straight from the ground up. Moves the bend, does not reduce it |
| `flowering_droop` | extra turn-over of the flowering stretch, as an **angle** (0.5 ≈ horizontal, 1 ≈ straight down) |
| `peduncle_reach` | how far below the lowest flower that turn is spread (0 = a one-node hook) |
| `peduncle_resolution` | smoothness of that bend (subdivides only the bending stretch) |
| `flower_weight` / `leaf_weight` | the mass that drives all of the above |

Pedicel- and branch-level pairs — one is **shape**, one is **weight**:

- `pedicel_bend` (shape; held in the pedicel's plane, turns with the stem; does
  **not** move the flower) vs. `pedicel_droop` (weight; rotates the whole
  pedicel against gravity and **carries the flower with it**; scaled by flower
  weight).
- inflorescence `branch_bend` (shape, own plane) vs. `branch_droop` (weight,
  against gravity).

**Angle-based vs. weight-scaled — the distinction that decides which lever.**
Some bends are a **pure angle**, applied regardless of mass, so they work on any
plant: `flowering_droop` (turns the flowering stretch of stem), and
`pedicel_angle_deg` / `flower_pitch_deg` (aim a pedicel / flower). Others are
**self-weight**, so they need real mass to show and are near-inert on a small or
light plant: `droop` (whole-plant, ∝ mass × length) and `pedicel_droop` /
`branch_droop` (scaled by `flower_weight`). *Measured:* on a 0.19 m raceme with
`flower_weight` 0.00015, `pedicel_droop` −0.7 turns the flowers the same −5° as 0
does; at `flower_weight` 0.004 the same factor gives −66°.

**How to make something hang** (pick the lever that matches the botany):

- a **nodding head or spike** → `flowering_droop` (a pure angle — 0.5 ≈
  horizontal, 1 ≈ down — reliable at any weight); `peduncle_reach` spreads the
  arc, and on a `solitary` flower it is what gives the flower a stalk to bend.
- a **light flower pointing down** (lily-of-the-valley, snowdrop) → negative
  `pedicel_angle_deg`, the geometric aim — weight-independent.
- a **heavy flower sagging on its stalk** (foxglove, fuchsia) → negative
  `pedicel_droop` **and** a real `flower_weight`; without the weight it does
  nothing.
- a **weeping panicle** → negative inflorescence `branch_droop`.
- a **one-sided raceme** → `flower_spread_deg` below 360, plus one of the flower
  aims above.
- a **whole weeping plant** → high `droop` + lower `stem_stiffness` — but only if
  it has the mass and length to bend (a tall, heavy plant, not a tiny herb).

**Trap — "nothing happens" has three causes; check in this order:** (1) the stem
is too stiff — lower `stem_stiffness` or `rigid_base`; (2) the flower is too light
for a weight-scaled bend — raise `flower_weight`, or switch to the angle-based
lever (`pedicel_angle_deg`, `flowering_droop`); (3) the plant is too small for
self-weight to matter at all — use the angle-based levers, not `droop`.

---

## 4. The growth-direction group — steering the growing tip

Four controls decide where a growing apex points each step, applied in this
order: **wander → circumnutation → gravitropism → environment.** They shape the
stem *during* growth (unlike the bend group, which deforms the finished stem
afterward).

| Control | What it does |
|---|---|
| `growth_irregularity_deg` | random wander — an incoherent nod, a fresh direction each step (natural imperfection) |
| `circumnutation_deg` | the coherent version — the nod's azimuth precesses, so it winds a helix instead of sprawling (the spiralling of a growth timelapse) |
| `circumnutation_rate_deg` | how far that azimuth turns per step: the helix pitch; its sign is the handedness |
| `gravitropism` | correction back toward the axis's setpoint angle each step (0 = floppy, 1 = ramrod) |
| `autotropism` | age-stiffening: straightens *older* stem toward its setpoint after growth, concentrating the nod at the tip |

**Coherent vs. incoherent is a real distinction, not a relabel.** *Measured:* at
the same per-step bend size, `growth_irregularity_deg` drifts ~68 mm off a 1.2 m
stem and barely winds, while `circumnutation_deg` stays a ~13 mm helix winding
~14 turns. Set them independently — irregularity for imperfection, circumnutation
for spiralling.

**Circumnutation's visibility trades against gravitropism, and that is the whole
of why it can look absent.** Gravitropism corrects each node once, at birth, so a
strong upright correction straightens the nod almost as fast as it is made — on
the default plant the helix is ~2 mm and effectively invisible. To see it, lower
`gravitropism` and/or raise `autotropism`.

**Autotropism is what makes it read on an upright plant.** Gravitropism is
memoryless (per node, at birth), so on its own circumnutation is a *uniform*
helix all the way to the tip. Autotropism revisits older nodes and straightens
them, leaving a tidy stem with the nod confined to the growing tip — the
canonical timelapse look. It only ever undoes wander, and is a separate pass from
`droop` (opposite sign: droop sags the finished stem *down* under load,
autotropism straightens it *up* toward its setpoint). With no circumnutation it
simply tidies the random `growth_irregularity_deg` of older stem; higher = a
shorter, tidier live zone at the very tip.

**Trap — "circumnutation does nothing":** it is being straightened faster than it
is made. Lower `gravitropism` or raise `autotropism`; raising `circumnutation_deg`
alone on a stiff upright stem mostly wins back a little waviness, not a spiral.

**Upturned branch tips are a Gravitropic Setpoint map, not Spiral Pitch.** A
branch's setpoint is its launch angle from vertical, so `gravitropism` holds it
there along its whole length. A level's Gravitropic Setpoint map multiplies that
angle per node, in the world's frame: below 1 a node aims more upright,
whichever way its branch faces (Digby & Firn's set-point angle, more upright at
a young tip). Spiral Pitch bends in each branch's own frame, which every member
of a whorl has turned differently. *Measured:* with a 180-degree roll, one
branch of five curled up and the others swung up to 78 degrees sideways or
down. A setpoint map has no sideways swing (0.1 degrees). The map reads along
the branch's *full* Steps, so a branch cut short by Growth % never reaches the
part of the curve near the tip. Lift the short ones with a Pitch map instead.
Leaf weight and `gravitropism` below 1 pull a tip back short of its setpoint.

---

## 5. Age drives development — the master dial

One **Age 0→1** is the plant's whole life; `max_steps` is the length of that
timeline. Two families of curve read different x-axes — **do not confuse them**:

- **Development curves** read `0 (seedling) → 1 (mature)`:
  `branching_onset` (when buds may break), `apical_dominance` (how hard the
  leader suppresses laterals, declining with age), `flowering_onset` (when
  flowering is permitted), `leaf_form_profile` (juvenile → adult blade),
  `senescence_onset`.
- **Profile curves** read `0 (base) → 1 (tip)` along an axis:
  `internode_profile`, `leaf_size_profile`, `leaf_angle_profile`,
  `radius_profile`, `stiffness_profile`, `lateral_vigour`,
  `branch_angle_profile`.

**Worked consequence — a rosette that bolts is not a special mechanism.** It is
`internode_profile` held near zero at the base and rising sharply: short crowded
basal internodes, long bare ones above. Same node count, different shape.

---

## 6. Environment modifies, never drives

Environmental response changes the **density** and **direction** of growth. It
**never creates, places, or terminates** an organ — morphology decides those.

- `shade_tolerance` scales each bud's branch density by the local light, so
  branch/leaf/flower density becomes a *result of the surroundings* rather than a
  number you set. `light_seeking` leans tips out of shade. `obstacle_clearance`
  curves tips away from surfaces.
- **Heliotropism** re-aims *organ facing* toward a tracked object (the plant's
  after growth: each flower component's own `flower_heliotropism` toward its
  own Sun Object (per inflorescence; a variant component's flowers take their
  inflorescence's), and the plant's `leaf_heliotropism` toward the plant's
  **Sun Object** (Light). A flower's own stalk bends with it as one even arc
  by the flower's whole turn (cut into pieces so no joint turns more than
  7 degrees; its end moves as the arc's chord, half the turn): a capped
  flower's top internode from the node below, but only where no branch
  leaves that internode, and a pedicel from its node; no other node moves. It needs
  **both** a Sun Object
  picked **and** a non-zero strength — either alone does nothing. Organs face
  wherever the object sits *relative to the plant*, so it is placed by moving it,
  not rotating it. Keep the three sun/light responses straight: `light_seeking`
  steers the growing *stem* toward a light gradient; an `ALIGN` effector twists
  the growth *frame* (rotating the whole leaf set, and the branches with it);
  heliotropism turns each *organ's own face* to the sun, after everything else,
  so a drooped head still faces it and no stem moves.
- **At the neutral defaults** (`shade_tolerance` 1, `light_seeking` 0,
  `obstacle_clearance` 0) the plant responds to nothing and grows identically to
  a no-environment run — the light field is not even built. Leave them at the
  defaults unless the target explicitly involves shade, phototropism, or an
  obstacle.
- `senescence_shade_threshold` is the one that removes structure, and it runs
  *after* growth (a post-growth judgement), so it does not violate the rule.

---

## 7. Inertness by inflorescence kind

The `inflorescence_kinds` table in the spec is authoritative — it lists, per
kind: `topology`, `has_pedicels`, `flowering_gradients_apply`,
`terminal_cluster`, `flat_topped`, `relayed`. **Setting an inert control does
nothing; do not spend a value on it.** The rules that follow from the table:

- **No pedicels** (so `pedicel_*` is inert): `spike`, `catkin` (sessile);
  `solitary`, `cyme`, `scorpioid`, `helicoid`, `dichasium`, `pleiochasium`
  (capped / relayed).
- **Flowering gradients inert** (the along-the-axis profiles): the terminal
  clusters `umbel`, `flat_umbel`. Their variances still apply.
- **`head_radius`, `head_dome`** only shape `corymb` and `flat_umbel`.
- **`cluster_spread_deg`** only opens `umbel` (and its compound forms).
- **`relay_rotation_deg`** only used by the cymose kinds.
- **The inflorescence `branch_*` group** only applies when `branching_depth > 0`
  (a compound / panicle form).
- **`flower_pitch_deg`** is live only on `spike` and `catkin` — the sessile kinds
  where a flower has no pedicel to take its facing from and so states its own.
  Everywhere else it is inert: a pedicelled flower faces where its pedicel
  arrives (aim it with `pedicel_droop` / `pedicel_angle_deg`), and a capped or
  clustered flower is aimed by its dome.

---

## 8. Counts, span, and compounding

- `flower_count` is capped by the steps an axis has to lay flowers in; the model
  **widens the timeline (`total_steps`) to fit**, so ask for the count you want
  rather than working around a cap.
- **Cymose kinds ignore `flower_count`**: their flower count is
  `branching_depth × laterals_per_node` (a dichasium at depth 4 is 2⁴ = 16 tips).
- **A cyme's bloom order is its relay generation**: `bloom_start` = depth /
  `branching_depth` (tip flower 0) for `centrifugal` / `basipetal`, reversed for
  `centripetal` / `acropetal`; picking a cymose kind stamps `centrifugal`.
- **A flower component's Bloom Timing** instance map shifts each flower's
  `bloom_start` by (curve − 1) × Amount, clamped to 0..1 (Random Range ±Range:
  a mix of buds and open flowers); `bloom_spread` scales how much any of it shows.
  A delay also delays `bloom_birth` (From Age opens a flower at the later of
  the two; an early shift never opens a flower before it exists). Under From
  Age the mix needs a Day inside the calendar's bloom window: past its end
  Progress is 1 and every flower is out.
- **`branching_depth` makes an inflorescence compound** (raceme→panicle,
  umbel→compound umbel, …). The count multiplies by the rays each level *and
  again* by however many branch tips the plant itself has — **depth 2 on a bushy
  plant is thousands of flowers and a slow grow.** Use `max_branch_order` 0 while
  tuning a compound form.
- `branching_depth` is independent of `max_branch_order`: the first is orders of
  *inflorescence*, the second is the plant's own *woody* architecture.

---

## 9. Determinism and variance

Everything is seeded: the same profile builds the same plant, byte for byte.
Every `*_variation` field scatters its quantity deterministically — a fraction
of the value (sizes, lengths, widths) or a spread in degrees (angles) — drawn per
organ, per stem, or per gap as the field notes say. Use variance freely; a plant
with zero variance everywhere reads as instanced, not grown.

---

## 10. What to reach for first

Author in this order, and lean on the shipped presets as worked examples
(`presets/*.json`, all valid and round-trip-guarded):

1. **Habit & size**: `axis_architecture`, `architecture`, `max_branch_order`,
   `nodes_before_flowering` + `internode_length` (these two set height),
   `tip_radius`.
2. **Leaves**: `trunk_phyllotaxis` / `lateral_phyllotaxis`, `leaf_placement`,
   `leaf_angle_deg`, `leaf_size` (a multiplier), the size/angle profiles.
3. **Inflorescence**: `inflorescence.kind` first (it decides which of the rest
   apply — see §7), then its structure, then the flowering gradients.
4. **Physical form**: the bend group (§3) and the organ weights.
5. **Environment**: only if the target calls for it (§6).

See `plantgen-recipes.md` for archetype → knob starting points.
