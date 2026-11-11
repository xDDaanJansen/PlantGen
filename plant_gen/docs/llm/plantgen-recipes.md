# PlantGen — archetype recipes

Starting points, not finished plants. Each names the few knobs that *make* the
archetype; everything else stays at its default and the shipped presets
(`presets/*.json`) are fuller worked examples. Read with `plantgen-spec.json`
(fields + endpoint visuals) and `plantgen-couplings.md` (why these knobs
interact). Where a recipe needs a particular mesh, it says so — that mesh is the
**user's** to model, not yours to fake with parameters.

---

### Upright leafy perennial (the baseline)
`axis_architecture: single_stem`, `architecture: monopodial`,
`max_branch_order: 2`, `branch_density: ~0.5`. Height from
`nodes_before_flowering` × `internode_length`. This is the generic default;
start here and change only what the target needs.

### Nodding single-flower head (poppy, sunflower, wind-anemone bud)
`inflorescence.kind: solitary`. Make it hang with the bend group, not by
rotating a mesh: raise `flowering_droop` (0.5 ≈ horizontal, 1 ≈ straight down),
give it a stalk to bend over with `peduncle_reach` (a solitary flower sits on
the very tip, so with reach 0 there is nothing to bend), and set `flower_weight`
for how heavy the head reads. `rigid_base` high keeps the stem straight with the
bend only at the neck. *User provides:* the flower/disc mesh.

### Flat-topped cluster (yarrow, elder, cow-parsley head)
`inflorescence.kind: corymb` (branched) or `flat_umbel` (from one point).
`head_radius` sets how wide, `head_dome` how rounded (0 dead flat, +1 a mound,
−1 cupped). For a real cow-parsley, make it compound: `branching_depth: 1`
(= compound umbel) — but watch the count (couplings §8). *User provides:* the
small individual floret mesh.

### Grass / sedge tuft
`axis_architecture: multi_stem`, `stem_count` 5–20, `tiller_interval` a few
steps; `trunk_phyllotaxis.pattern: distichous` (two-ranked) or `alternate`;
`inflorescence.kind: spike` (sessile — no pedicels); `cross_section: triangular`
for a sedge. Long blade leaves via a high-aspect *user* leaf mesh + `leaf_size`.

### One-sided hanging bells (foxglove)
`inflorescence.kind: raceme`; crowd the flowers onto one side with
`flower_spread_deg` ≈ 90. Hang the bells with the lever that matches their
weight: foxglove's bells are heavy, so negative `pedicel_droop` **with a real
`flower_weight`** (it is weight-gated — inert on light flowers); a light-flowered
relative (lily-of-the-valley) wants negative `pedicel_angle_deg` instead, the
weight-independent aim. `flower_pitch_deg` is inert on a pedicelled kind — the
flower faces where its pedicel arrives. Grade size down the spire with
`flower_size_profile: fade_out`. A basal rosette that bolts: `internode_profile:
rosette`. *User provides:* the tubular bell mesh.

### Globe / sphere umbel (allium, ivy)
`inflorescence.kind: umbel`, `cluster_spread_deg: 360` (a complete sphere;
180 is a hemisphere), `flower_count` high. Flowers fill the whole dome including
its centre. *User provides:* the tiny star-floret mesh.

### Weeping / drooping panicle (buddleia, some grasses in flower)
`inflorescence.kind: panicle`, `branching_depth: 1`; sweep the branches down
with negative inflorescence `branch_droop` (−1 hangs them straight down);
`branch_angle_deg` near 0 or negative for a wide open form. Shape a pyramid with
`branch_length_profile` / `branch_angle` gradients (longer, wider at the base).

### Hanging catkins (hazel, birch, willow)
`inflorescence.kind: catkin` (this stamps the axillary + pendulous set), then
the `catkin_*` block: `catkin_length` (0 = none), `catkin_count` a small bunch
(2–4) with `catkin_count_variation`, `catkin_stalk` so they hang off the twig
rather than sprout from it. A catkin is a built hanging curve — it sags on its
own; you do not model it. Usually branches-only placement, upper stem. Set
`max_steps` generously (~80): a lateral grows slowly, and a short span gives
stubby catkins.

### Pendant flower chains (laburnum, wisteria-like)
`inflorescence.kind: raceme` + `inflorescence.attachments: [axillary]` (borne
along the stems, not at the tips) + `orientation_mode: pendulous`
(`orientation_strength` toward 1). The chain hangs from a node and the stem
carries on past it. *User provides:* the pea-flower mesh.

### Forking foliage plant (clubmoss, Selaginella)
`architecture: dichotomous` (the tip forks in two — no lateral branching
imitates this), `inflorescence.kind: none` (never flowers), `branch_density: 0`.
`fork_angle_deg` for the splay, `fork_twist_deg` 0 for a flat fan or 90 for a
3-D bush. For a trunk-and-branches look instead, `architecture:
pseudomonopodial` with `fork_balance` ~0.6.

### Mint-family whorls (dead-nettle, salvia)
`inflorescence.kind: verticillaster`; `whorl_count` ~6 (flowers ringed at each
node); `bract_size` for the leafy bract under each whorl;
`trunk_phyllotaxis.pattern: decussate` foliage; `cross_section: square`.

---

**If the target doesn't match an archetype**, compose from the parts: pick the
`kind` first (it gates the rest — couplings §7), set habit and size, then the
inflorescence structure, then the bend group for how it carries itself. Verify
by loading the preset and growing it.
