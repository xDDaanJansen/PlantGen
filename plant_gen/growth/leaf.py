"""Leaf-mesh authoring -- the PURE engine (numpy, no bpy).

The bpy-free geometry logic the leaf tool's Blender bridge (`blender/leaf.py`)
drives. Kept here under `growth/`, like every module in this package, so it must
import and run without Blender and can be unit-tested headless. Its *internals*
are image processing rather than the developmental/spline math the other growth
modules do, but its *role* is identical: turn parameters into plain mesh data
the bridge turns into a real object.

This file covers BOTH halves of the tool, which share one output contract and
one downstream (the bridge's triangulation + object build):

    * CUTOUT (`cut_leaf`): an existing leaf texture atlas -> one clean low-poly
      mesh per leaf, via silhouette tracing (below).
    * GENERATE (`generate_leaf`): a procedural blade silhouette from a
      width-profile (a named botanical shape, or explicit samples the bridge can
      feed from a curve widget) mirrored about the midrib -- no image needed.

Both return a `CutLeaf` (ordered boundary loop + per-vertex UVs + warnings) in
the same leaf local-space contract, so the bridge meshes them identically.

Pipeline (per the atlas-cutout research, `docs`/scratch `03-atlas-cutout-uv.md`):

    alpha (H,W) -> threshold -> scanline-rail boundary trace (sub-pixel)
                -> Visvalingam-Whyatt simplify to an exact vertex budget
                -> reorient into the leaf local-space contract + atlas UVs

Interior TRIANGULATION is deliberately NOT done here -- it is a bmesh concern
(`bmesh.ops.triangle_fill`) handled in the bridge, since Blender ships no
Delaunay in pure Python and the native op is the right tool. This engine returns
the ordered boundary loop (verts + UVs); the bridge fills it.

THE LEAF LOCAL-SPACE CONTRACT (non-negotiable -- both the parametric add-on and plant_gen
measure it off every instanced leaf; get it wrong and the leaf renders sideways/
backwards/mis-scaled with NO error, only a screenshot catches it):

    * flat in local XY (every vertex Z == 0), so the face normal is +Z
    * length runs along local +Y, from the origin (petiole/base) toward the tip
    * origin sits at the attachment point (base/petiole), not the bbox centre
    * real-world metric scale (NOT a normalized 0..1 box)

UV is a SEPARATE coordinate space from position: UV stays atlas-relative (the
traced pixel coordinate over the whole atlas, so every cut leaf shares one
material and batches), while position is reorimented into the contract above.
The two are derived from a common pixel origin and must never be conflated.

Two tracers, dispatched by `trace` (default "auto"):
    * scanline-rail -- fast, SUB-PIXEL edges, but assumes ONE opaque interval per
      row (single-lobe leaves: ovate, lanceolate, elliptic, cordate, obovate --
      the large majority). Used for clean single-lobe silhouettes.
    * Moore boundary-follow -- handles ANY silhouette (deep lobes: maple, oak;
      and holes), tracing the largest opaque component at PIXEL resolution.
      "auto" falls back to this the moment the rail sees a row with >1 interval,
      so lobed leaves get correct (concavity-preserving) geometry rather than
      the notch-filled outline the rail would produce.

MVP scope / known limits (flagged, not hidden):
    * the boundary-follow path traces the OUTER silhouette only -- interior
      holes/fenestration (e.g. Monstera) are recorded in `CutLeaf.warnings`, not
      cut out. Cutting real holes needs multi-contour meshing (a later pass).
    * long axis is assumed vertical in the image (tip up / base down, the common
      atlas convention). `base` picks which vertical end is the petiole.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Default metric scale: pixels of atlas per Blender unit (metre). A ~256 px leaf
# at 2000 ppu is ~0.128 m tall, a plausible real leaf. Exposed as a control in
# the bridge/UI; only ever a uniform scale on the output, never shape-affecting.
DEFAULT_PIXELS_PER_UNIT = 2000.0

# Default boundary vertex budget. 24-64 covers simple leaf outlines without
# visible faceting at foliage distance (research: `03-atlas-cutout-uv.md` sec 1.3).
DEFAULT_BOUNDARY_VERTS = 48

# Batch atlas cut: opaque components smaller than this fraction of the whole
# atlas are treated as specks/dust and skipped, so `cut_atlas` doesn't emit a
# mesh per stray pixel. Exposed as a control in the bridge/UI.
DEFAULT_ATLAS_MIN_AREA_FRAC = 0.0005

# Hard ceiling on the vertex count of ONE leaf mesh, whatever the settings.
# A boundary-only cut has verts == boundary count, so this clamps the budget;
# the interior-grid and compound-leaf paths (later) must count their total
# against this same number. A real-time leaf is tens-to-low-hundreds of verts,
# so this is a safety rail against the poly-explosion the research flagged for
# fine margins and bipinnate leaves, not a normal operating point.
MAX_LEAF_VERTS = 5000

# --- GENERATE-half defaults (procedural blade) ---
# Metric size of a default blade: ~12 cm long, ~5 cm across -- a plausible
# mid-size leaf. `width` is the FULL blade width (tip-to-tip across the midrib),
# so the half-width at each station is 0.5 * width * profile(t).
DEFAULT_LEAF_LENGTH = 0.12
DEFAULT_LEAF_WIDTH = 0.05
# Resolution knobs (independent): rows along the LENGTH (vertical) and the number
# of half-columns each side across the WIDTH (horizontal). Total verts ~
# (rows+1)*(2*cols+1), clamped against MAX_LEAF_VERTS. Vertical and horizontal are
# separate so a leaf that needs fine teeth (rows) need not also be dense across.
DEFAULT_LEAF_STATIONS = 24      # vertical rows (a.k.a. stations)
DEFAULT_RES_H = 6               # horizontal half-columns per side

# Margin (edge) profiles -- a periodic perturbation of the OUTLINE (X, in-plane),
# applied only to the blade's boundary column. Each
# type has its OWN tooth-shape function (`_margin_unit_profile`) so they render
# genuinely differently; `margin_teeth` (frequency) and `margin_depth` (amplitude
# vs local half-width) tune them. Every margin is in-plane -- the only out-of-
# plane (Z) edge motion is the WAVE deformer, kept separate on purpose.
# Grouped by character:
#   GROUP 1 -- pointed teeth with sharp V-valleys:
#     dentate       -- sharp, symmetric triangle teeth
#     crenate       -- "rounded dentate": rounded peak, still-sharp valley
#     denticulate   -- dentate but finer (more, smaller teeth)
#   GROUP 2 -- forward-raked teeth, sharp tip, slightly curved edges:
#     serrate       -- forward-pointing sharp teeth
#     serrulate     -- serrate but finer (more, smaller teeth)
#     doubleserrate -- big serrate teeth that carry smaller serrations
#   GROUP 3 -- smooth signed waves (rounded peaks AND valleys, no straight bits):
#     sinuate       -- rounded wave (medium)
#     undulate      -- large rounded wave (fewer)
#     lobate        -- largest rounded wave, deepest valleys
#   UNIQUE:
#     entire        -- smooth, no teeth
#     spiny         -- sharp, narrow spine points
LEAF_MARGINS = (
    "entire", "dentate", "crenate", "denticulate",
    "serrate", "serrulate", "doubleserrate",
    "sinuate", "undulate", "lobate", "spiny",
)
DEFAULT_LEAF_MARGIN = "entire"
DEFAULT_MARGIN_TEETH = 10
DEFAULT_MARGIN_DEPTH = 0.15
# Per-margin tooth-spacing (wavelength) multiplier: a "-ulate" variant is the base
# type at more, smaller teeth; the large rounded waves are the base at fewer teeth.
# doubleserrate keeps big-tooth spacing (mult 1) and adds its sub-teeth inside its
# own profile, so it is NOT listed here.
_MARGIN_FREQ_MULT = {
    "denticulate": 2.0, "serrulate": 2.0, "undulate": 0.6, "lobate": 0.45,
}
# The signed margins cut INWARD (valleys go below the outline); they need a wider
# margin band so the valley never reaches the interior mesh.
_MARGIN_SIGNED = ("sinuate", "undulate", "lobate")
# Extra clearance (fraction of half-width) between the deepest valley and the
# interior mesh, so teeth never touch the inner geometry.
MARGIN_CLEARANCE = 0.12
# Hard ceiling on tooth depth (fraction of the local half-width). Teeth protrude
# outward (the UV fit includes them, so they never clip the unit square); this
# cap also keeps teeth from self-intersecting / reading as a
# broken spiky rim at extreme depth. Clamped in the engine AND the UI max.
MARGIN_DEPTH_MAX = 0.4

# Named leaf shapes -- each a HAND-AUTHORED half-width profile, dialled by the
# artist in the Custom Profile curve, saved as a leaf preset JSON (one per shape,
# the project's "Leaf preset" folder) and baked here. `profile` is the half-width
# sampled uniformly base (t=0) -> tip (t=1), normalised so its peak is exactly 1,
# so `width` stays the blade's true widest width. `aspect` is widest-width/length
# as authored; the panel applies it (width = length * aspect) when a shape is
# picked, so the silhouette matches the authored one at any size.
# `_resample_profile` maps the samples onto the grid rows (linear interpolation --
# the same way the authored curve was rendered, so the bake is exact). Sharp
# features (a cordate notch, pinnatisect lobes) want Resolution V ~60+.
@dataclass(frozen=True)
class LeafShape:
    label: str                   # display name
    aspect: float                # widest width / length, as authored
    profile: tuple[float, ...]   # half-width base -> tip, peak 1


LEAF_SHAPES: dict[str, LeafShape] = {
    "acicular": LeafShape("Acicular", 0.095, (
        0.1215, 0.0845, 0.1009, 0.1341, 0.1827, 0.2465, 0.3253, 0.4163, 0.5153,
        0.6167, 0.7139, 0.8012, 0.8739, 0.9302, 0.9695, 0.9923, 1.0000, 0.9966,
        0.9870, 0.9701, 0.9451, 0.9116, 0.8690, 0.8169, 0.7550, 0.6837, 0.6033,
        0.5150, 0.4198, 0.3192, 0.2148, 0.1079, 0.0000,
    )),
    "acuminate": LeafShape("Acuminate", 0.343, (
        0.0336, 0.0247, 0.0232, 0.0303, 0.0818, 0.3189, 0.5007, 0.6511, 0.7740,
        0.8701, 0.9404, 0.9841, 1.0000, 0.9963, 0.9811, 0.9515, 0.9039, 0.8380,
        0.7595, 0.6785, 0.6022, 0.5339, 0.4732, 0.4167, 0.3630, 0.3125, 0.2650,
        0.2202, 0.1772, 0.1346, 0.0913, 0.0466, 0.0000,
    )),
    "aristate": LeafShape("Aristate", 0.378, (
        0.0382, 0.0169, 0.0198, 0.0335, 0.0818, 0.3929, 0.6435, 0.7911, 0.8825,
        0.9405, 0.9751, 0.9940, 1.0000, 0.9939, 0.9758, 0.9459, 0.9047, 0.8526,
        0.7900, 0.7168, 0.6358, 0.5478, 0.4552, 0.3612, 0.2696, 0.1867, 0.1200,
        0.0796, 0.0607, 0.0436, 0.0280, 0.0147, 0.0000,
    )),
    "cordate": LeafShape("Cordate", 0.664, (
        0.0459, 0.0282, 0.0274, 0.0264, 0.0253, 0.6709, 0.8025, 0.8752, 0.9219,
        0.9529, 0.9731, 0.9863, 0.9948, 0.9984, 0.9996, 1.0000, 0.9963, 0.9830,
        0.9588, 0.9269, 0.8898, 0.8489, 0.7991, 0.7285, 0.6126, 0.4777, 0.3796,
        0.3047, 0.2373, 0.1723, 0.1126, 0.0584, 0.0000,
    )),
    "cuneate": LeafShape("Cuneate", 0.265, (
        0.1152, 0.0709, 0.0665, 0.0626, 0.1037, 0.1572, 0.2120, 0.2680, 0.3251,
        0.3833, 0.4423, 0.5019, 0.5617, 0.6212, 0.6796, 0.7364, 0.7899, 0.8398,
        0.8840, 0.9226, 0.9532, 0.9768, 0.9919, 1.0000, 0.9998, 0.9918, 0.9709,
        0.9338, 0.8725, 0.7728, 0.6065, 0.3321, 0.0000,
    )),
    "deltoid": LeafShape("Deltoid", 0.531, (
        0.0765, 0.0471, 0.0482, 0.0494, 0.0505, 0.0516, 0.0525, 0.0532, 0.0529,
        0.4629, 0.8019, 0.8976, 0.9415, 0.9669, 0.9816, 0.9910, 0.9967, 1.0000,
        0.9596, 0.9131, 0.8595, 0.7980, 0.7285, 0.6531, 0.5760, 0.5017, 0.4332,
        0.3713, 0.3138, 0.2588, 0.2052, 0.1482, 0.0000,
    )),
    "digitate": LeafShape("Digitate", 0.150, (
        0.1000, 0.1402, 0.1827, 0.2264, 0.2708, 0.3156, 0.3606, 0.4055, 0.4502,
        0.4946, 0.5386, 0.5821, 0.6249, 0.6670, 0.7082, 0.7483, 0.7872, 0.8247,
        0.8604, 0.8939, 0.9249, 0.9525, 0.9757, 0.9926, 1.0000, 0.9948, 0.9771,
        0.9441, 0.8925, 0.8156, 0.7016, 0.5208, 0.0000,
    )),
    "elliptic": LeafShape("Elliptic", 0.448, (
        0.0335, 0.0136, 0.0142, 0.0461, 0.1474, 0.2854, 0.4513, 0.6083, 0.7296,
        0.8176, 0.8798, 0.9244, 0.9558, 0.9769, 0.9905, 0.9980, 1.0000, 0.9969,
        0.9882, 0.9741, 0.9542, 0.9286, 0.8958, 0.8569, 0.8111, 0.7577, 0.6962,
        0.6261, 0.5458, 0.4534, 0.3461, 0.2156, 0.0000,
    )),
    "falcate": LeafShape("Falcate", 0.265, (
        0.0850, 0.0408, 0.0306, 0.0506, 0.1477, 0.2836, 0.4493, 0.6201, 0.7694,
        0.8811, 0.9532, 0.9911, 1.0000, 0.9850, 0.9511, 0.9010, 0.8384, 0.7709,
        0.7052, 0.6418, 0.5806, 0.5220, 0.4664, 0.4137, 0.3640, 0.3173, 0.2729,
        0.2300, 0.1875, 0.1441, 0.0988, 0.0509, 0.0000,
    )),
    "flabellate": LeafShape("Flabellate", 0.562, (
        0.0501, 0.0250, 0.0253, 0.0256, 0.0259, 0.0262, 0.0265, 0.0268, 0.0271,
        0.0274, 0.0277, 0.0280, 0.0283, 0.0286, 0.0289, 0.0292, 0.9451, 0.9978,
        1.0000, 0.9932, 0.9819, 0.9665, 0.9473, 0.9240, 0.8962, 0.8631, 0.8244,
        0.7779, 0.7219, 0.6530, 0.5638, 0.4372, 0.0000,
    )),
    "hastate": LeafShape("Hastate", 0.562, (
        0.0501, 0.0251, 0.0276, 0.0301, 0.0326, 0.0352, 0.0377, 0.0402, 0.3099,
        0.9256, 1.0000, 0.9972, 0.9367, 0.7804, 0.5758, 0.4543, 0.4097, 0.3932,
        0.3820, 0.3783, 0.3751, 0.3703, 0.3642, 0.3573, 0.3479, 0.3195, 0.2900,
        0.2578, 0.2181, 0.1628, 0.1044, 0.0506, 0.0000,
    )),
    "lanceolate": LeafShape("Lanceolate", 0.191, (
        0.1436, 0.0558, 0.0480, 0.0690, 0.1121, 0.1893, 0.3209, 0.5190, 0.7158,
        0.8461, 0.9229, 0.9666, 0.9881, 0.9976, 1.0000, 0.9958, 0.9840, 0.9634,
        0.9342, 0.8955, 0.8469, 0.7890, 0.7226, 0.6489, 0.5699, 0.4882, 0.4060,
        0.3258, 0.2494, 0.1781, 0.1123, 0.0533, 0.0000,
    )),
    "linear": LeafShape("Linear", 0.169, (
        0.1622, 0.0762, 0.0479, 0.0548, 0.1234, 0.5913, 0.8587, 0.9502, 0.9846,
        0.9948, 0.9991, 1.0000, 0.9984, 0.9949, 0.9903, 0.9854, 0.9808, 0.9771,
        0.9748, 0.9744, 0.9758, 0.9796, 0.9823, 0.9721, 0.9468, 0.9018, 0.8367,
        0.7489, 0.6372, 0.5013, 0.3448, 0.1744, 0.0000,
    )),
    "obcordate": LeafShape("Obcordate", 0.348, (
        0.0787, 0.0300, 0.0260, 0.0296, 0.1211, 0.2302, 0.3479, 0.4501, 0.5177,
        0.5712, 0.6200, 0.6752, 0.7358, 0.7834, 0.8167, 0.8424, 0.8635, 0.8820,
        0.8987, 0.9141, 0.9285, 0.9423, 0.9552, 0.9678, 0.9800, 0.9896, 0.9976,
        1.0000, 0.9962, 0.9845, 0.9604, 0.9159, 0.0000,
    )),
    "obovate": LeafShape("Obovate", 0.508, (
        0.0701, 0.0268, 0.0233, 0.0306, 0.0539, 0.1428, 0.2220, 0.2907, 0.3617,
        0.4414, 0.5236, 0.5995, 0.6651, 0.7211, 0.7698, 0.8125, 0.8513, 0.8865,
        0.9190, 0.9494, 0.9780, 1.0000, 0.9976, 0.9903, 0.9781, 0.9600, 0.9342,
        0.8984, 0.8508, 0.7839, 0.6886, 0.5311, 0.0000,
    )),
    "obtuse": LeafShape("Obtuse", 0.626, (
        0.0599, 0.0342, 0.0315, 0.0349, 0.0384, 0.3086, 0.5559, 0.6599, 0.7272,
        0.7772, 0.8208, 0.8631, 0.9022, 0.9363, 0.9635, 0.9834, 0.9955, 1.0000,
        0.9974, 0.9882, 0.9731, 0.9528, 0.9276, 0.8963, 0.8576, 0.8096, 0.7499,
        0.6758, 0.5830, 0.4685, 0.3306, 0.1718, 0.0000,
    )),
    "orbicular": LeafShape("Orbicular", 0.762, (
        0.0574, 0.0327, 0.0301, 0.0334, 0.0348, 0.1198, 0.2329, 0.3742, 0.5247,
        0.6576, 0.7609, 0.8374, 0.8935, 0.9337, 0.9621, 0.9815, 0.9931, 0.9987,
        1.0000, 0.9984, 0.9939, 0.9862, 0.9750, 0.9596, 0.9392, 0.9131, 0.8800,
        0.8375, 0.7831, 0.7119, 0.6142, 0.4647, 0.0000,
    )),
    "ovate": LeafShape("Ovate", 0.494, (
        0.0886, 0.0519, 0.0495, 0.0563, 0.0619, 0.2783, 0.5198, 0.7068, 0.8288,
        0.9063, 0.9544, 0.9827, 0.9967, 1.0000, 0.9970, 0.9895, 0.9784, 0.9639,
        0.9465, 0.9263, 0.9029, 0.8767, 0.8475, 0.8147, 0.7782, 0.7377, 0.6920,
        0.6403, 0.5808, 0.5112, 0.4249, 0.3068, 0.0000,
    )),
    "pinnatisect": LeafShape("Pinnatisect", 0.295, (
        0.0743, 0.0376, 0.0200, 0.0138, 0.0148, 0.0208, 0.0301, 0.0411, 0.0510,
        0.5718, 0.5884, 0.2919, 0.2148, 0.7522, 0.5504, 0.3259, 0.2487, 0.9528,
        0.7858, 0.5155, 0.3165, 0.5487, 1.0000, 0.8730, 0.3615, 0.2961, 0.4668,
        0.9229, 0.8694, 0.7829, 0.6626, 0.4849, 0.0000,
    )),
    "reniform": LeafShape("Reniform", 0.996, (
        0.0439, 0.0215, 0.0113, 0.0078, 0.0086, 0.0122, 0.0178, 0.0243, 0.0303,
        0.5461, 0.7654, 0.8498, 0.8996, 0.9332, 0.9566, 0.9735, 0.9853, 0.9931,
        0.9978, 1.0000, 0.9999, 0.9974, 0.9906, 0.9797, 0.9640, 0.9426, 0.9143,
        0.8768, 0.8271, 0.7615, 0.6697, 0.5281, 0.0000,
    )),
    "rhomboid": LeafShape("Rhomboid", 0.597, (
        0.0524, 0.0260, 0.0137, 0.0095, 0.0102, 0.0147, 0.0213, 0.0290, 0.0360,
        0.0949, 0.2591, 0.4266, 0.5962, 0.7661, 0.9333, 1.0000, 0.8676, 0.8069,
        0.7984, 0.8427, 0.7458, 0.6891, 0.6504, 0.6256, 0.6787, 0.5412, 0.4748,
        0.4516, 0.4669, 0.2870, 0.2210, 0.1284, 0.0000,
    )),
    "spatulate": LeafShape("Spatulate", 0.250, (
        0.1250, 0.0793, 0.0798, 0.0837, 0.0902, 0.1003, 0.1143, 0.1328, 0.1563,
        0.1852, 0.2202, 0.2616, 0.3114, 0.3689, 0.4358, 0.5114, 0.5949, 0.6833,
        0.7712, 0.8510, 0.9164, 0.9634, 0.9907, 1.0000, 0.9963, 0.9823, 0.9562,
        0.9160, 0.8573, 0.7756, 0.6604, 0.4854, 0.0000,
    )),
    "spear_shaped": LeafShape("Spear-Shaped", 0.326, (
        0.1164, 0.0585, 0.0478, 0.0575, 0.0669, 0.0728, 0.0840, 1.0000, 0.9991,
        0.9967, 0.9929, 0.9879, 0.9817, 0.9744, 0.9661, 0.9571, 0.9475, 0.9374,
        0.9271, 0.9167, 0.9064, 0.8965, 0.8869, 0.8780, 0.8686, 0.7930, 0.7142,
        0.6300, 0.5372, 0.4304, 0.3055, 0.1608, 0.0000,
    )),
    "subulate": LeafShape("Subulate", 0.129, (
        0.2950, 0.1642, 0.1806, 0.1887, 0.2684, 0.8967, 0.9795, 1.0000, 0.9953,
        0.9756, 0.9465, 0.9113, 0.8707, 0.8267, 0.7799, 0.7299, 0.6771, 0.6210,
        0.5621, 0.5008, 0.4380, 0.3755, 0.3151, 0.2587, 0.2080, 0.1636, 0.1255,
        0.0933, 0.0666, 0.0445, 0.0265, 0.0118, 0.0000,
    )),
    "truncate": LeafShape("Truncate", 0.594, (
        0.0639, 0.0356, 0.0391, 0.0409, 0.0414, 0.1524, 0.2523, 0.3378, 0.4152,
        0.4868, 0.5530, 0.6145, 0.6716, 0.7246, 0.7734, 0.8180, 0.8582, 0.8939,
        0.9249, 0.9510, 0.9719, 0.9873, 0.9968, 1.0000, 0.9980, 0.9918, 0.9804,
        0.9626, 0.9349, 0.8934, 0.8271, 0.7055, 0.3985,
    )),
}
DEFAULT_LEAF_SHAPE = "ovate"

# One line per built-in shape describing the outline it draws (the Shape
# dropdown's tooltips and the manual's shape cards). Described from the baked
# profile, not the botanical term: the blade is symmetric, so Falcate is not
# curved.
LEAF_SHAPE_NOTES: dict[str, str] = {
    "acicular": "Needle-like: very narrow, tapering to a sharp tip",
    "acuminate": "A rounded lower half drawn out into a long, tapering point",
    "aristate": "A rounded blade ending in a fine, bristle-like point",
    "cordate": "Broad and round-bottomed, with a short pointed tip",
    "cuneate": "Wedge-shaped: widest near the tip, narrowing straight down to the base",
    "deltoid": "Broad shoulders below a pointed tip, narrowing to the base",
    "digitate": "One long, narrow finger with a rounded tip",
    "elliptic": "Oval: widest at the middle, both ends tapering evenly",
    "falcate": "Narrow, widest low down and tapering to a long point (drawn straight, "
               "not sickle-curved)",
    "flabellate": "Fan-shaped: a broad, rounded top on a long, narrow base",
    "hastate": "Spear-headed: a narrow point above two flared lobes at the base",
    "lanceolate": "Lance-shaped: narrow, widest below the middle, tapering to a point",
    "linear": "Long and narrow, the sides nearly parallel",
    "obcordate": "Widest at its broad, flat tip, tapering to the base",
    "obovate": "Egg-shaped with the broad end at the tip",
    "obtuse": "Broad and rounded, with a blunt tip",
    "orbicular": "Nearly round",
    "ovate": "Egg-shaped with the broad end at the base",
    "pinnatisect": "Cut deeply into side lobes along a narrow centre",
    "reniform": "Kidney-shaped: much wider than long, rounded all round",
    "rhomboid": "Diamond-shaped, the edges cut into shallow lobes",
    "spatulate": "Spoon-shaped: a rounded tip on a long, narrow base",
    "spear_shaped": "Straight sides under a pointed top, cut square at the base",
    "subulate": "Awl-shaped: narrow, tapering from the base to a fine point",
    "truncate": "A broad, almost flat-topped tip narrowing to the base",
}


# Venation type -- HIDDEN, derived from the shape (each real leaf shape conforms
# to one venation type). The generate path builds a structured quad grid whose
# edge loops follow it: for `pinnate` the midrib is the central column and the
# constant-y rows are the secondary veins; for `parallel` (grass-like narrow
# blades) the along-length columns are the parallel veins. `palmate` and
# `dichotomous` need radial/forking flow that is a later stage, so a shape mapped
# to them still builds on the pinnate grid for now (label is truthful; geometry
# is deferred). The rule: very-narrow ~parallel
# blades -> parallel; flabellate -> dichotomous; orbicular -> palmate; else
# pinnate.
VENATIONS = ("pinnate", "palmate", "parallel", "dichotomous")
DEFAULT_VENATION = "pinnate"
VENATION_BY_SHAPE: dict[str, str] = {
    "linear": "parallel",
    "acicular": "parallel",
    "subulate": "parallel",
    "flabellate": "dichotomous",
    "orbicular": "palmate",
    # Palmate-veined forms: radial lobing /
    # basal lobes fan their major veins from one basal point.
    "cordate": "palmate",
    "reniform": "palmate",
    "hastate": "palmate",
    "digitate": "palmate",
    # everything else falls through venation_for() to pinnate.
}


def venation_for(shape: str) -> str:
    """The venation type a shape conforms to (default pinnate)."""
    return VENATION_BY_SHAPE.get((shape or "").lower(), DEFAULT_VENATION)


@dataclass(frozen=True)
class CutLeaf:
    """Plain mesh data for one leaf, ready for the bridge to build.

    Two shapes, distinguished by `faces`:
      * `faces` EMPTY -- `verts` is an ordered boundary LOOP in leaf-local space
        (CCW viewed down +Z, open: verts[0] != verts[-1]); the bridge triangulates
        it. The CUTOUT path (traced atlas silhouettes) returns this.
      * `faces` NON-EMPTY -- `verts` is a full MESH and `faces` are its polygons
        (index tuples, quads or tris). The GENERATE path returns this: a
        venation-following quad grid the bridge builds directly (no fill).

    `verts` is (N, 3) float, Z == 0, +Y from the base at the origin (the
    local-space contract). `uvs` is the matching (N, 2) UV per vertex. `warnings`
    names anything the trace/build could not handle cleanly."""

    verts: np.ndarray
    uvs: np.ndarray
    warnings: tuple[str, ...] = field(default_factory=tuple)
    faces: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    # Optional per-vertex scalar in [0, 1] the bridge writes to a vertex-colour
    # RED channel ("LeafMask"): the wave-influence / edge mask -- 0 along the
    # rigid midrib, 1 out at the margins. Only the generate path fills it.
    mask: np.ndarray | None = None

    @property
    def boundary_count(self) -> int:
        return int(self.verts.shape[0])


def alpha_to_mask(alpha: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boolean opaque mask from an alpha plane. `alpha` is (H, W) float in
    [0, 1], row 0 = TOP (image orientation -- the bridge flips Blender's
    bottom-up pixels before calling in)."""
    return alpha >= threshold


def _subpixel_left(alpha_row: np.ndarray, col: int, threshold: float) -> float:
    """Sub-pixel X of the outside->inside alpha crossing at the left edge of an
    opaque run whose first inside pixel is `col`. Linear-interpolated between the
    straddling pixels so an anti-aliased edge lands where it visually is, not on
    a pixel centre."""
    if col <= 0:
        return float(col)
    a0, a1 = float(alpha_row[col - 1]), float(alpha_row[col])
    if a1 == a0:
        return float(col)
    return (col - 1) + (threshold - a0) / (a1 - a0)


def _subpixel_right(alpha_row: np.ndarray, col: int, threshold: float) -> float:
    """Sub-pixel X of the inside->outside crossing at the right edge of an opaque
    run whose last inside pixel is `col`."""
    width = alpha_row.shape[0]
    if col >= width - 1:
        return float(col)
    a0, a1 = float(alpha_row[col]), float(alpha_row[col + 1])
    if a0 == a1:
        return float(col)
    return col + (a0 - threshold) / (a0 - a1)


def _count_runs(mask_row: np.ndarray) -> int:
    """How many separate opaque intervals this row has (0 for empty). >1 is the
    lobed/holed case the scanline-rail trace cannot resolve on its own."""
    if not mask_row.any():
        return 0
    # A run starts at each True whose left neighbour is False.
    padded = np.concatenate(([False], mask_row))
    return int(np.count_nonzero(padded[1:] & ~padded[:-1]))


def _row_runs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    """The opaque intervals in a row as (first_col, last_col) inclusive pairs."""
    cols = np.nonzero(mask_row)[0]
    if cols.size == 0:
        return []
    breaks = np.nonzero(np.diff(cols) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [cols.size - 1]))
    return [(int(cols[s]), int(cols[e])) for s, e in zip(starts, ends, strict=True)]


def _label_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int, list]]:
    """Label every 8-connected opaque component. Returns one tuple per
    component: (y0, y1, x0, x1, area, runs) -- inclusive bbox, pixel area, and
    the component's rows as (row, start_col, end_col) runs. Run-based union-find:
    cheap for atlas-sized images, no scipy. Robust against specks (they come
    back as tiny-area components the caller can filter)."""
    H = mask.shape[0]
    runs_by_row = [_row_runs(mask[r]) for r in range(H)]

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r, runs in enumerate(runs_by_row):
        for j in range(len(runs)):
            parent[(r, j)] = (r, j)
    # 8-connectivity: run [s1,e1] and the row below's [s2,e2] touch if their
    # column spans overlap or are diagonally adjacent.
    for r in range(H - 1):
        for j, (s1, e1) in enumerate(runs_by_row[r]):
            for k, (s2, e2) in enumerate(runs_by_row[r + 1]):
                if e1 >= s2 - 1 and e2 >= s1 - 1:
                    union((r, j), (r + 1, k))

    groups: dict[tuple[int, int], list] = defaultdict(list)
    for r, runs in enumerate(runs_by_row):
        for j, (s, e) in enumerate(runs):
            groups[find((r, j))].append((r, s, e))

    comps = []
    for runs in groups.values():
        ys = [r for r, _, _ in runs]
        ss = [s for _, s, _ in runs]
        es = [e for _, _, e in runs]
        area = sum(e - s + 1 for _, s, e in runs)
        comps.append((min(ys), max(ys), min(ss), max(es), area, runs))
    return comps


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the single largest 8-connected opaque component -- the
    speck-robust silhouette selector the boundary tracer uses on one leaf."""
    comps = _label_components(mask)
    if not comps:
        return mask
    _y0, _y1, _x0, _x1, _area, runs = max(comps, key=lambda c: c[4])
    out = np.zeros_like(mask)
    for (r, s, e) in runs:
        out[r, s:e + 1] = True
    return out


# 8-neighbourhood offsets (drow, dcol) in clockwise order starting at North.
_MOORE_CW = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _moore_trace(mask: np.ndarray) -> list[tuple[int, int]]:
    """Moore-neighbour boundary trace of one simply-connected blob: an ordered
    loop of (row, col) boundary pixels, clockwise in image coords. Starts at the
    topmost-then-leftmost opaque pixel (whose west/north neighbours are empty,
    so it is genuinely on the boundary). Preserves concavities the scanline-rail
    fills -- the whole point of this path for lobed leaves. Interior holes are
    not followed (outer silhouette only)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    start = (int(ys[0]), int(xs[0]))

    def opaque(p) -> bool:
        r, c = p
        return 0 <= r < H and 0 <= c < W and bool(mask[r, c])

    loop: list[tuple[int, int]] = [start]
    b = start
    prev_dir = 6  # entered the start from the west
    max_iter = 4 * H * W + 16
    for _ in range(max_iter):
        found = None
        for k in range(1, 9):
            idx = (prev_dir + k) % 8
            dr, dc = _MOORE_CW[idx]
            n = (b[0] + dr, b[1] + dc)
            if opaque(n):
                found = (idx, n)
                break
        if found is None:
            break  # isolated pixel
        idx, n = found
        prev_dir = (idx + 4) % 8  # backtrack: direction from n back to b
        b = n
        if b == start and len(loop) >= 3:
            break  # closed the loop
        loop.append(n)
    return loop


def _trace_boundary(mask: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Contour of the largest opaque component via boundary following, as
    pixel-space (x, y-from-top) points -- the lobed/holed fallback for silhouettes
    the single-interval scanline-rail cannot resolve."""
    comp = _largest_component_mask(mask)
    loop = _moore_trace(comp)
    if len(loop) < 3:
        return (np.empty((0, 2), dtype=np.float64),
                ("boundary trace: degenerate silhouette",))
    contour = np.array([(c, r) for (r, c) in loop], dtype=np.float64)
    contour = _dedupe_consecutive(contour)
    warn = ("boundary trace: lobed/holed leaf traced at pixel resolution "
            "(largest opaque component; interior holes/fenestration ignored -- "
            "outer silhouette only)",)
    return contour, warn


def _trace_scanline_rail(
    alpha: np.ndarray, threshold: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Trace the opaque silhouette into an ordered pixel-space polyline via the
    scanline-rail method: per row, the leftmost and rightmost sub-pixel alpha
    crossings become two rails that concatenate into one closed loop with no
    stitching ambiguity. Returns (contour (K,2) float in pixel space as
    (x, y) with y measured from the top, warnings)."""
    mask = alpha >= threshold
    warnings: list[str] = []

    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64), ("empty: no opaque pixels",)

    multi = 0
    left_rail: list[tuple[float, float]] = []
    right_rail: list[tuple[float, float]] = []
    for r in rows:
        cols = np.nonzero(mask[r])[0]
        first, last = int(cols[0]), int(cols[-1])
        if _count_runs(mask[r]) > 1:
            multi += 1  # gap in the row -- outer rail still uses first/last
        lx = _subpixel_left(alpha[r], first, threshold)
        rx = _subpixel_right(alpha[r], last, threshold)
        # Centre of a 1-px-wide row would collapse the two rails onto each other;
        # nudge apart by half a pixel so the loop keeps a (thin) interior.
        if rx - lx < 0.5:
            mid = 0.5 * (lx + rx)
            lx, rx = mid - 0.25, mid + 0.25
        left_rail.append((lx, float(r)))
        right_rail.append((rx, float(r)))

    if multi:
        warnings.append(
            f"scanline-rail: {multi} row(s) had multiple opaque intervals "
            "(lobed/holed leaf) -- outer silhouette traced, interior gaps ignored; "
            "marching-squares fallback is a later pass")

    # Down the left edge (top->bottom), then up the right edge (bottom->top).
    contour = np.array(left_rail + right_rail[::-1], dtype=np.float64)
    contour = _dedupe_consecutive(contour)
    return contour, tuple(warnings)


def _dedupe_consecutive(pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Drop points equal to their predecessor (and wrap: last vs first). Keeps
    the polygon from carrying zero-length edges at the tip/base where the two
    rails meet."""
    if pts.shape[0] < 2:
        return pts
    keep = [0]
    for i in range(1, pts.shape[0]):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > eps:
            keep.append(i)
    out = pts[keep]
    if out.shape[0] >= 2 and np.hypot(*(out[0] - out[-1])) <= eps:
        out = out[:-1]
    return out


def _signed_area(pts: np.ndarray) -> float:
    """Shoelace signed area of a closed polygon given as open vertex list
    (verts[0] != verts[-1]). Positive == CCW in a y-up frame."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _vw_simplify(pts: np.ndarray, n_target: int) -> np.ndarray:
    """Visvalingam-Whyatt simplification of a closed polygon (cyclic) down to
    exactly `n_target` vertices: repeatedly remove the vertex forming the
    smallest-area triangle with its two neighbours. High-curvature points
    (corners, the tip) form large triangles and survive; smooth arcs are thinned
    -- corner-preserving simplification with an exact final count.

    Naive O((M-n)*M); fine for atlas-cell contours (M in the hundreds)."""
    n = pts.shape[0]
    if n_target >= n:
        return pts
    keep = pts.tolist()

    def tri_area(a, b, c) -> float:
        return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

    while len(keep) > n_target:
        m = len(keep)
        best_i, best_a = 0, float("inf")
        for i in range(m):
            a = keep[(i - 1) % m]
            c = keep[(i + 1) % m]
            area = tri_area(a, keep[i], c)
            if area < best_a:
                best_a, best_i = area, i
        keep.pop(best_i)
    return np.array(keep, dtype=np.float64)


def _resolve_base(kind: str) -> str:
    kind = (kind or "auto").lower()
    if kind == "auto":
        return "bottom"
    if kind in ("bottom", "top"):
        return kind
    raise ValueError(f"unknown base end {kind!r} (expected auto/bottom/top)")


def _to_leaf_local(
    contour_px: np.ndarray,
    *,
    cell_origin: tuple[int, int],
    atlas_size: tuple[int, int],
    pixels_per_unit: float,
    base: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a pixel-space contour into (leaf-local verts (N,3), atlas UVs (N,2)).

    `contour_px` is in CELL-local pixel coords with y from the cell's top;
    `cell_origin` is the cell's (x, y) offset in the FULL atlas (y from the
    atlas top) so UVs stay atlas-relative. Applies the module's local-space
    contract: flat XY, +Y from the base at the origin, midrib-centred X, metric
    scale."""
    px, py = contour_px[:, 0], contour_px[:, 1]
    ppu = float(pixels_per_unit)

    # --- UV: atlas-relative, Blender V is bottom-up ---
    # The tracers work in pixel-CENTRE coordinates (pixel i's centre is i), and a
    # pixel's centre sits at (i + 0.5) / size in UV -- without the half texel the
    # texture lands half a pixel off the geometry and a strip of alpha shows.
    atlas_w, atlas_h = atlas_size
    ox, oy = cell_origin
    u = (px + ox + 0.5) / float(atlas_w)
    v = 1.0 - (py + oy + 0.5) / float(atlas_h)
    uvs = np.column_stack([u, v])

    # --- Position: reorient to the contract ---
    base = _resolve_base(base)
    min_py, max_py = float(py.min()), float(py.max())
    # +Y from the base toward the tip. Base at image-bottom (max py) is the
    # common tip-up convention; 'top' flips it.
    if base == "bottom":
        y_local = (max_py - py) / ppu
    else:  # top
        y_local = (py - min_py) / ppu
    # Midrib-centred X: centre on the blade's own mid-x so the origin's X is the
    # midrib, matching how a petiole attaches on the centreline.
    mid_px = 0.5 * (float(px.min()) + float(px.max()))
    x_local = (px - mid_px) / ppu

    verts = np.column_stack([x_local, y_local, np.zeros_like(x_local)])

    # Ensure CCW viewed down +Z (consistent winding for the bridge's face).
    if _signed_area(verts[:, :2]) < 0.0:
        verts = verts[::-1].copy()
        uvs = uvs[::-1].copy()
    return verts, uvs


def _trace_contour(
    alpha: np.ndarray, threshold: float, mode: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Trace the silhouette to a pixel-space contour, dispatching by `mode`:

        "rail"     : scanline-rail only (fast, sub-pixel, single-lobe assumption).
        "boundary" : Moore boundary follow only (any shape, pixel resolution).
        "auto"     : rail, but fall back to boundary follow when the rail detects
                     a row with multiple opaque intervals (a lobe/hole it cannot
                     resolve). The common, clean single-lobe leaf keeps the
                     rail's sub-pixel edge; only genuinely lobed leaves pay for
                     the fallback."""
    mode = (mode or "auto").lower()
    if mode not in ("auto", "rail", "boundary"):
        raise ValueError(f"unknown trace mode {mode!r} (expected auto/rail/boundary)")
    if mode == "boundary":
        return _trace_boundary(alpha >= threshold)
    contour, warnings = _trace_scanline_rail(alpha, threshold)
    if mode == "auto" and any("multiple opaque intervals" in w for w in warnings):
        return _trace_boundary(alpha >= threshold)
    return contour, warnings


def _cut_region(
    alpha: np.ndarray,
    *,
    cell_origin: tuple[int, int],
    atlas_size: tuple[int, int],
    threshold: float,
    boundary_verts: int,
    pixels_per_unit: float,
    base: str,
    trace: str,
) -> CutLeaf:
    """Trace one already-isolated region's alpha into a `CutLeaf`: contour trace
    -> vertex-budget clamp -> leaf local space + atlas UVs. Shared by `cut_leaf`
    (whole image / one cell) and `cut_atlas` (one labelled component).

    The vertex cap is enforced HERE. A boundary-only cut has verts == boundary
    count, so the budget IS the vertex count and MAX_LEAF_VERTS clamps it; the
    floor of 3 keeps a valid polygon, the ceiling is the rail the raw traced
    contour (2 points per opaque row) would otherwise blow past. `cell_origin`
    is the region's (x, y) offset in the full atlas (y from the top) so UVs stay
    atlas-relative."""
    contour, warnings = _trace_contour(alpha, threshold, trace)
    if contour.shape[0] < 3:
        raise ValueError("degenerate silhouette: fewer than 3 boundary points")

    requested = max(3, int(boundary_verts))
    target = min(requested, MAX_LEAF_VERTS)
    if requested > MAX_LEAF_VERTS:
        warnings = warnings + (
            f"boundary_verts {requested} exceeds the {MAX_LEAF_VERTS}-vertex "
            f"hard cap; clamped to {MAX_LEAF_VERTS}",)
    contour = _vw_simplify(contour, target)
    verts, uvs = _to_leaf_local(
        contour, cell_origin=cell_origin, atlas_size=atlas_size,
        pixels_per_unit=pixels_per_unit, base=base)
    leaf = CutLeaf(verts=verts, uvs=uvs, warnings=warnings)
    if leaf.boundary_count > MAX_LEAF_VERTS:
        raise ValueError(
            f"leaf mesh has {leaf.boundary_count} verts, over the "
            f"{MAX_LEAF_VERTS} hard cap")
    return leaf


def cut_leaf(
    alpha: np.ndarray,
    *,
    atlas_size: tuple[int, int],
    cell: tuple[int, int, int, int] | None = None,
    threshold: float = 0.5,
    boundary_verts: int = DEFAULT_BOUNDARY_VERTS,
    pixels_per_unit: float = DEFAULT_PIXELS_PER_UNIT,
    base: str = "auto",
    trace: str = "auto",
) -> CutLeaf:
    """Cut ONE leaf mesh from an atlas alpha plane -- the whole image, or a
    single `cell`. For a multi-leaf atlas, `cut_atlas` cuts every part at once.

    `alpha`     : (H, W) float in [0, 1], row 0 = top (the bridge pre-flips
                  Blender's bottom-up pixels).
    `atlas_size`: (W, H) of the WHOLE atlas, for atlas-relative UV.
    `cell`      : (x0, y0, x1, y1) sub-rectangle in atlas pixels (y from top);
                  None = the whole `alpha`. `alpha` is expected already cropped
                  to the cell; the offset only keeps UVs atlas-relative.
    `threshold`, `boundary_verts`, `pixels_per_unit`, `base`, `trace` : see the
    module doc and `_trace_contour`.

    Returns a `CutLeaf`. Raises ValueError on an empty/degenerate silhouette."""
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be (H, W); got shape {alpha.shape}")
    cell_origin = (int(cell[0]), int(cell[1])) if cell is not None else (0, 0)
    return _cut_region(
        alpha, cell_origin=cell_origin, atlas_size=atlas_size, threshold=threshold,
        boundary_verts=boundary_verts, pixels_per_unit=pixels_per_unit,
        base=base, trace=trace)


def cut_atlas(
    alpha: np.ndarray,
    *,
    atlas_size: tuple[int, int],
    threshold: float = 0.5,
    boundary_verts: int = DEFAULT_BOUNDARY_VERTS,
    pixels_per_unit: float = DEFAULT_PIXELS_PER_UNIT,
    base: str = "auto",
    trace: str = "auto",
    min_area_frac: float = DEFAULT_ATLAS_MIN_AREA_FRAC,
) -> list[CutLeaf]:
    """Cut EVERY separate opaque part of an atlas into its own leaf mesh.

    Labels all 8-connected opaque components, drops specks below `min_area_frac`
    of the image, and cuts each survivor independently -- masking it to its own
    bounding box first, so a neighbour whose bbox overlaps cannot bleed into the
    trace. Each leaf comes back in its OWN leaf-local space (base at the origin)
    with atlas-relative UVs keyed to its own atlas region, so all share one
    material. Ordered top-to-bottom then left-to-right (reading order) for
    stable output. A degenerate component above the size floor is skipped rather
    than aborting the batch. Raises ValueError only if nothing survives."""
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be (H, W); got shape {alpha.shape}")
    mask = alpha >= threshold
    total = alpha.shape[0] * alpha.shape[1]
    min_area = max(1, int(min_area_frac * total))
    comps = [c for c in _label_components(mask) if c[4] >= min_area]
    comps.sort(key=lambda c: (c[0], c[2]))  # top->bottom, then left->right

    leaves: list[CutLeaf] = []
    for (y0, y1, x0, x1, _area, runs) in comps:
        # Alpha of THIS component only, cropped to its bbox (neighbours zeroed).
        sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=alpha.dtype)
        for (r, s, e) in runs:
            sub[r - y0, s - x0:e - x0 + 1] = alpha[r, s:e + 1]
        try:
            leaves.append(_cut_region(
                sub, cell_origin=(x0, y0), atlas_size=atlas_size, threshold=threshold,
                boundary_verts=boundary_verts, pixels_per_unit=pixels_per_unit,
                base=base, trace=trace))
        except ValueError:
            continue  # untraceable sliver that still cleared the area floor
    if not leaves:
        raise ValueError(
            "no opaque regions above the minimum area; lower Min Region, or "
            "check the atlas alpha channel")
    return leaves


# ---------------------------------------------------------------------------
# GENERATE half: procedural blade silhouette from a single width profile.
# ---------------------------------------------------------------------------


def _resolve_shape(shape: str) -> LeafShape:
    key = (shape or DEFAULT_LEAF_SHAPE).lower()
    try:
        return LEAF_SHAPES[key]
    except KeyError:
        raise ValueError(
            f"unknown leaf shape {shape!r} (expected one of "
            f"{', '.join(sorted(LEAF_SHAPES))})") from None


def _resolve_margin(margin: str) -> str:
    m = (margin or DEFAULT_LEAF_MARGIN).lower()
    if m not in LEAF_MARGINS:
        raise ValueError(f"unknown leaf margin {margin!r} (expected one of "
                         f"{', '.join(LEAF_MARGINS)})")
    return m


def _margin_unit_profile(margin: str, f: np.ndarray) -> np.ndarray:
    """Per-tooth displacement shape, `f` the fractional position within one cycle
    in [0, 1). Group 1/2 tooth types return [0, 1] (0 at the sinus, 1 at the tip);
    group 3 wavy types return [-1, 1] (edge waves out AND in)."""
    two_pi = 2.0 * np.pi

    def serrate_shape(g):              # forward-raked, sharp tip, slightly bowed edges
        rise = np.clip(g / 0.7, 0.0, 1.0) ** 0.85
        fall = np.clip((1.0 - g) / 0.3, 0.0, 1.0) ** 0.85
        return np.where(g < 0.7, rise, fall)

    # GROUP 2 -- forward-raked teeth (sharp tip). doubleserrate overlays a run of
    # smaller serrations (3x) on each big tooth so it reads as genuinely double.
    if margin in ("serrate", "serrulate"):
        return serrate_shape(f)
    if margin == "doubleserrate":
        return np.clip(0.65 * serrate_shape(f)
                       + 0.45 * serrate_shape(np.mod(f * 3.0, 1.0)), 0.0, 1.0)
    # GROUP 1 -- pointed teeth, sharp V-valleys.
    if margin in ("dentate", "denticulate"):   # sharp symmetric triangle
        return 1.0 - 2.0 * np.abs(f - 0.5)
    if margin == "crenate":                     # rounded dentate: round peak, sharp valley
        return np.sin(np.pi * f)
    # UNIQUE -- sharp narrow spine.
    if margin == "spiny":
        return np.clip(1.0 - np.abs(f - 0.5) / 0.16, 0.0, 1.0)
    # GROUP 3 -- smooth signed waves (rounded peaks AND valleys), no clipping so no
    # straight segments. lobate deepens its valleys (peaks scaled to 0.8, valleys full).
    if margin in ("sinuate", "undulate"):
        return np.sin(two_pi * f)
    if margin == "lobate":
        s = np.sin(two_pi * f)
        return np.where(s < 0.0, s, 0.8 * s)
    return np.zeros_like(f)            # entire / doubleserrate offset handled in caller


def _margin_band(margin: str, depth: float) -> float:
    """Width (fraction of the half-width) of the smooth margin band reserved
    between the toothed outline and the interior mesh. Signed margins cut inward
    by `depth`, so their band is `depth` + clearance; outward-only margins need
    only the clearance strip."""
    inward = depth if margin in _MARGIN_SIGNED else 0.0
    return min(inward + MARGIN_CLEARANCE, 0.6)


def _resample_profile(samples: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Resample an arbitrary half-width profile (given as uniform samples over
    [0, 1], e.g. from a curve widget) onto the station coordinates `t`. Clamped
    to [0, 1]; the literal profile is used as-is (NOT peak-normalised, so a curve
    that never reaches 1 gives a genuinely narrower blade)."""
    samples = np.asarray(samples, dtype=np.float64).ravel()
    if samples.size < 2:
        raise ValueError("width_samples needs at least 2 points")
    xs = np.linspace(0.0, 1.0, samples.size)
    return np.clip(np.interp(t, xs, samples), 0.0, 1.0)


def _blade_grid(
    half: np.ndarray, y: np.ndarray, *, length: float, width: float, cols: int,
    margin: str = "entire", teeth: int = 0, depth: float = 0.0, band: float = 0.0,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], np.ndarray]:
    """Structured venation-following quad grid from per-row half-widths.

    `half` (R+1,) is the half-width at each row (0 at a pointed pole); `y` (R+1,)
    the row's Y (base->tip); `cols` = N half-columns each side, so a full row has
    2N+1 verts spanning -half..+half. Column i=N is the midrib (x=0); constant-y
    rows are the secondary veins -- the pinnate/parallel flow the quads follow.
    A row whose half-width is ~0 (a pointed tip/base) collapses to ONE midrib
    vertex and fans to its full neighbour with triangles, like a UV-sphere pole.

    The margin (if any) is applied AFTER the flat grid is built, as teeth around
    the whole outline boundary by arc length (`_apply_boundary_teeth`) -- so teeth
    run continuously up the sides AND around the apex (a terminal tooth), not just
    on the two side columns. Faces are wound CCW (normal +Z). Returns
    (verts (K,3), faces, uvs (K,2), across (K,)) where `across` is the relative
    distance from the midrib (0 midrib -> 1 edge) at every row -- the wave/edge
    mask, so edges stay masked even where the blade is narrow near the tip."""
    n = cols
    ncol = 2 * n + 1
    eps = 1e-9 * width if width else 1e-12

    nrow = len(half)
    verts: list[tuple[float, float, float]] = []
    across: list[float] = []          # |column|/n: 0 at midrib -> 1 at the edge
    row_start: list[int] = []
    row_full: list[bool] = []
    for k in range(nrow):
        row_start.append(len(verts))
        yk = float(y[k])
        if half[k] <= eps:
            verts.append((0.0, yk, 0.0))
            across.append(0.0)
            row_full.append(False)
        else:
            hk = float(half[k])
            inner = hk * (1.0 - band)     # interior stops here; band = tooth strip
            for i in range(ncol):
                if band <= 0.0:
                    x = (i - n) / n * hk
                elif i == 0:
                    x = -hk                # boundary sits on the outline
                elif i == ncol - 1:
                    x = hk
                elif n > 1:                # interior columns compressed to +-inner
                    x = inner * ((i - 1) / (n - 1) - 1.0)
                else:
                    x = 0.0                # single interior column = midrib
                verts.append((x, yk, 0.0))
                across.append(abs(i - n) / n)
            row_full.append(True)
    varr = np.asarray(verts, dtype=np.float64)
    across_arr = np.asarray(across, dtype=np.float64)

    faces = _grid_faces(row_start, row_full, ncol)

    if margin != "entire" and teeth > 0 and depth > 0.0:
        _apply_boundary_teeth(
            varr, row_start, row_full, ncol, half, margin, int(teeth),
            float(depth), float(length))

    uvs = _fit_uvs(varr)
    return varr, tuple(faces), uvs, across_arr


def _grid_faces(row_start: list[int], row_full: list[bool], ncol: int
                ) -> list[tuple[int, ...]]:
    """Faces of a row grid (each row either `ncol` verts or one pole vertex):
    quads between full rows, a triangle fan where a row collapses to a pole.
    Wound CCW viewed down +Z. Shared by the generated blade and the cutout grid."""
    faces: list[tuple[int, ...]] = []
    for k in range(len(row_start) - 1):
        a, b = row_start[k], row_start[k + 1]
        fa, fb = row_full[k], row_full[k + 1]
        if fa and fb:
            for i in range(ncol - 1):
                faces.append((a + i, a + i + 1, b + i + 1, b + i))
        elif fb and not fa:          # pole below, full row above
            for i in range(ncol - 1):
                faces.append((a, b + i + 1, b + i))
        elif fa and not fb:          # full row below, pole above
            for i in range(ncol - 1):
                faces.append((a + i, a + i + 1, b))
        # pole-to-pole (both collapsed) contributes nothing
    return faces


def _fit_uvs(verts: np.ndarray) -> np.ndarray:
    """UVs of the FLAT blade with ONE scale for U and V, so a texel is square
    on the leaf whatever its proportions (a separate U and V fit squashed a
    narrow leaf by 1/aspect -- 10x for acicular). The tooth-inclusive bounding
    box is fitted into the unit square by its longer side and centred; the
    midrib stays at U = 0.5 (fitted by the larger half so both sides fit)."""
    x, y = verts[:, 0], verts[:, 1]
    half = float(np.abs(x).max()) if x.size else 0.0
    y0, y1 = (float(y.min()), float(y.max())) if y.size else (0.0, 0.0)
    size = max(2.0 * half, y1 - y0)
    if size <= 0.0:
        return np.full((len(verts), 2), 0.5)
    return np.column_stack([0.5 + x / size, 0.5 + (y - 0.5 * (y0 + y1)) / size])


def _apply_boundary_teeth(
    verts: np.ndarray, row_start: list[int], row_full: list[bool], ncol: int,
    half: np.ndarray, margin: str, teeth: int, depth: float, length: float,
) -> None:
    """Displace the outline BOUNDARY vertices in place to form margin teeth,
    parameterised by arc length so teeth are even up the sides AND around the apex
    (the tip pole becomes a terminal tooth). The base region is faded smooth. Each
    boundary vertex moves along the local outward normal by depth * amplitude *
    tooth-profile(arc); inward (signed) motion is capped so it never crosses the
    midrib."""
    nrow = len(half)
    depth = min(max(depth, 0.0), MARGIN_DEPTH_MAX)

    # Boundary loop, base -> up the right side -> tip -> down the left side.
    right: list[int] = []
    left: list[int] = []
    tip = base = None
    for k in range(nrow):
        if row_full[k]:
            left.append(row_start[k])
            right.append(row_start[k] + ncol - 1)
        elif k == 0:
            base = row_start[k]
        else:
            tip = row_start[k]
    loop: list[int] = []
    if base is not None:
        loop.append(base)
    loop.extend(right)
    if tip is not None:
        loop.append(tip)
    loop.extend(reversed(left))
    if len(loop) < 4:
        return

    idx = np.asarray(loop)
    P = verts[idx, :2].copy()
    m = len(loop)

    # Arc length around the closed loop, and one-sided wavelength.
    seg = np.hypot(*(np.roll(P, -1, axis=0) - P).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))[:m]
    total = float(s[-1] + seg[-1]) or 1.0
    freq = teeth * _MARGIN_FREQ_MULT.get(margin, 1)
    lam = (0.5 * total) / max(freq, 1e-6)
    prof = _margin_unit_profile(margin, np.mod(s / lam, 1.0))

    # Outward normals from the loop tangent, oriented away from the midline.
    T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    tl = np.hypot(T[:, 0], T[:, 1])
    tl[tl == 0.0] = 1.0
    nx, ny = T[:, 1] / tl, -T[:, 0] / tl
    cy = float(P[:, 1].mean())
    flip = (nx * P[:, 0] + ny * (P[:, 1] - cy)) < 0.0
    nx[flip] *= -1.0
    ny[flip] *= -1.0

    # Amplitude: local half-width, floored so teeth stay visible up to the apex,
    # faded to zero over the base so the bottom stays smooth ("as is").
    maxhalf = float(np.max(half)) if half.size else 0.0
    local_half = np.abs(P[:, 0])
    ny_pos = P[:, 1] / length if length else np.zeros(m)
    bs = np.clip((ny_pos - 0.05) / 0.10, 0.0, 1.0)
    base_fade = bs * bs * (3.0 - 2.0 * bs)
    amp = depth * np.maximum(local_half, 0.30 * maxhalf) * base_fade

    disp = amp * prof
    inward_cap = 0.85 * local_half           # never cross the midrib on a valley
    disp = np.maximum(disp, -inward_cap)

    P[:, 0] += disp * nx
    P[:, 1] += disp * ny
    verts[idx, 0] = P[:, 0]
    verts[idx, 1] = P[:, 1]


# --- Blade deformation (per-leaf 3D shaping) -------------------------------
# Five deformers that lift the flat blade out of local XY into a real 3D leaf.
# ALL default to 0.0 => the blade stays flat (Z == 0) and the local-space
# contract is preserved untouched; only a non-zero control ever moves Z. They
# compose in a fixed order so the local cross-section detail (cup/wave/twist)
# rides along the length-wise bend (curl/gravity):
#
#   cup     cross-section curvature: the two edges rise (or fall) relative to
#           the midrib, so a section across the blade is a shallow arc (a gutter
#           / taco). Signed: + lifts the edges, - drops them.
#   wave    a rippling of the blade along its length (a sine in Z vs. length) --
#           the undulate/rumpled look. `wave_count` sets how many ripples.
#   twist   a progressive rotation of the cross-section about the midrib, base to
#           tip, so the tip is rolled relative to the base. Signed (radians*pi).
#   curl    a constant-curvature bend of the whole blade along its length in the
#           Y-Z plane (the midrib becomes a circular arc). Signed: + curls the
#           tip up (+Z), - down. Length-preserving (a true bend, not a lift).
#   gravity a downward droop with a cantilever profile (grows ~ as the square of
#           the distance from the base, so the tip sags most). Unsigned (>= 0),
#           always pulls -Z; distinct from curl in BOTH sign and profile.
#
# curl and gravity both act in the Y-Z plane but are genuinely different shapes:
# curl is a uniform-curvature arc (symmetric, signed styling), gravity is a
# tip-weighted sag (asymmetric, one-way load). They are meant to be combined.
LEAF_DEFORM_DEFAULTS: dict[str, float] = {
    "curl": 0.0, "cup": 0.0, "twist": 0.0, "gravity": 0.0, "wave": 0.0,
    "noise": 0.0,
}
DEFAULT_WAVE_COUNT = 3
# Perlin-like noise deform defaults: `noise` is the amplitude (fraction of the
# half-width), `noise_scale` the number of noise cells across the blade, `seed`
# the GLOBAL random seed. noise 0 -> no-op; a fixed seed is byte-deterministic.
DEFAULT_NOISE_SCALE = 4.0
DEFAULT_LEAF_SEED = 0
# Crumple: the WHOLE blade scrunched (a dried leaf), where noise only roughens
# its surface. The midrib bends at many small joints by smooth seeded noise --
# up/down about the blade's side axis and sideways in its plane -- and each bend
# carries everything beyond it, so the leaf deforms as one body and its midrib
# keeps its length. `crumple` 1 lets the heading wander up to half a turn either
# way; `crumple_scale` is how many bends run along the blade. 0 = no-op.
DEFAULT_CRUMPLE_SCALE = 5.0
_CRUMPLE_JOINTS = 96
# Seed offsets for the two bend streams (independent of the surface noise).
_CRUMPLE_SALT = (7919, 104729)


def _value_noise(nx: np.ndarray, ny: np.ndarray, seed: int, freq: float) -> np.ndarray:
    """Smooth, deterministic Perlin-like value noise in [-1, 1] over normalised
    coords `nx`,`ny` in [0, 1]. A seeded random lattice sampled with smoothstep
    bilinear interpolation -- pure numpy, no scipy, and byte-identical for a given
    seed (the project's determinism contract)."""
    cells = max(1, int(round(freq)))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    grid = rng.random((cells + 2, cells + 2))
    fx = np.clip(nx, 0.0, 1.0) * cells
    fy = np.clip(ny, 0.0, 1.0) * cells
    ix = np.clip(np.floor(fx).astype(int), 0, cells)
    iy = np.clip(np.floor(fy).astype(int), 0, cells)
    tx, ty = fx - ix, fy - iy
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)
    v00, v10 = grid[iy, ix], grid[iy, ix + 1]
    v01, v11 = grid[iy + 1, ix], grid[iy + 1, ix + 1]
    a = v00 * (1.0 - sx) + v10 * sx
    b = v01 * (1.0 - sx) + v11 * sx
    return (a * (1.0 - sy) + b * sy) * 2.0 - 1.0


def _deform_blade(
    verts: np.ndarray, *, length: float, width: float,
    curl: float = 0.0, cup: float = 0.0, twist: float = 0.0,
    gravity: float = 0.0, wave: float = 0.0, wave_count: int = DEFAULT_WAVE_COUNT,
    noise: float = 0.0, noise_scale: float = DEFAULT_NOISE_SCALE,
    seed: int = DEFAULT_LEAF_SEED, wave_mask: np.ndarray | None = None,
    midline: np.ndarray | None = None,
    crumple: float = 0.0, crumple_scale: float = DEFAULT_CRUMPLE_SCALE,
) -> np.ndarray:
    """Apply the blade deformers to a flat (Z==0) blade `verts` (N,3), returning
    new (N,3). Order: cup -> wave -> noise -> twist -> curl -> gravity ->
    crumple. All controls 0 -> `verts` returned unchanged (still flat), so the
    leaf-local contract survives a no-op call byte-for-byte.

    `wave_mask` (N,) in [0, 1], if given, scales the WAVE term per vertex -- the
    edge mask that keeps the rigid midrib flat and confines rippling to the
    margins (0 at the midrib, 1 at the edges). It never touches the others.
    `noise`/`noise_scale`/`seed` add a small Perlin-like organic Z displacement.
    `midline` (N,), if given, is the x of each vertex's own row centre line: cup,
    noise and twist then work about it (an asymmetric cutout leaf), not about
    x = 0. None (the symmetric generated blade) leaves the maths untouched."""
    if crumple and length:
        return _crumple(verts, length=length, width=width, curl=curl, cup=cup,
                        twist=twist, gravity=gravity, wave=wave, wave_count=wave_count,
                        noise=noise, noise_scale=noise_scale, seed=seed,
                        wave_mask=wave_mask, midline=midline, crumple=crumple,
                        crumple_scale=crumple_scale)
    x = verts[:, 0].astype(np.float64).copy()
    mid = None
    if midline is not None:
        mid = np.asarray(midline, dtype=np.float64)
        x = x - mid                           # across-blade offset from the row centre
    y = verts[:, 1].astype(np.float64).copy()
    z = np.zeros_like(x)                      # local normal offset, built up below
    t = y / length if length else np.zeros_like(y)
    hw = 0.5 * float(width)                   # nominal max half-width (peak of prof)

    # cup: edges rise as x^2, reaching cup*hw at the widest point (x == +-hw).
    if cup and hw > 0.0:
        z = z + cup * (x * x) / hw
    # wave: a sine ripple in Z along the length, amplitude wave*hw, confined to
    # the edges by the mask so the midrib stays flat and rigid.
    if wave and hw > 0.0:
        ripple = wave * hw * np.sin(2.0 * np.pi * float(wave_count) * t)
        if wave_mask is not None:
            ripple = ripple * wave_mask
        z = z + ripple
    # noise: small organic Z displacement from Perlin-like value noise, seeded.
    if noise and hw > 0.0:
        nx = x / width + 0.5 if width else np.zeros_like(x)
        z = z + noise * hw * _value_noise(nx, t, seed, noise_scale)
    # twist: rotate the (x, z) cross-section about the midrib by t*twist*pi.
    if twist:
        ang = twist * np.pi * t
        c, s = np.cos(ang), np.sin(ang)
        x, z = x * c - z * s, x * s + z * c
    # curl: bend the (y, z) plane into a constant-curvature arc of total angle
    # curl*pi. The midrib maps to the arc; the local z-offset rides its normal.
    theta = curl * np.pi
    if abs(theta) > 1e-9 and length:
        r = length / theta
        phi = theta * t
        sp, cp = np.sin(phi), np.cos(phi)
        y = r * sp - z * sp
        z = r * (1.0 - cp) + z * cp
    # gravity: cantilever sag, tip-weighted (~ t^2), one-way (-Z).
    if gravity:
        z = z - gravity * length * (t * t)

    if mid is not None:
        x = x + mid
    return np.column_stack([x, y, z])


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: rows of `v` turned by `angle` about the unit `axis`."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1.0 - c)


def _crumple(verts: np.ndarray, *, length: float, crumple: float,
             crumple_scale: float, seed: int, midline: np.ndarray | None,
             **deform) -> np.ndarray:
    """The crumple deformer (see `DEFAULT_CRUMPLE_SCALE`): the blade is shaped by
    every other deformer first, with probe points on its midrib riding along;
    then, joint by joint from the tip down, everything past a joint turns about
    that joint's midrib point. Joints run tip-first so each turns about a point
    no later bend has moved yet -- the result is a chain of bends, base fixed,
    midrib length kept."""
    v0 = np.asarray(verts, dtype=np.float64)
    t_v = v0[:, 1] / length
    tj = np.linspace(0.0, 1.0, _CRUMPLE_JOINTS + 1)
    # Probes on the midrib (the row centre line, for an asymmetric cutout).
    if midline is not None:
        order = np.argsort(t_v)
        pmid = np.interp(tj, t_v[order], np.asarray(midline, dtype=np.float64)[order])
        mids = np.concatenate([np.asarray(midline, dtype=np.float64), pmid])
    else:
        pmid, mids = np.zeros_like(tj), None
    probes = np.column_stack([pmid, tj * length, np.zeros_like(tj)])
    mask = deform.pop("wave_mask", None)
    if mask is not None:
        mask = np.concatenate([np.asarray(mask, dtype=np.float64), np.zeros_like(tj)])
    shaped = _deform_blade(np.vstack([v0, probes]), length=length, seed=seed,
                           wave_mask=mask, midline=mids, **deform)
    out, spine = shaped[:len(v0)].copy(), shaped[len(v0):]
    # Bend angles: smooth seeded noise along the length, relative to the base.
    amp = float(crumple) * np.pi
    nx = np.full_like(tj, 0.37)
    up = _value_noise(nx, tj, int(seed) + _CRUMPLE_SALT[0], crumple_scale)
    side = _value_noise(nx, tj, int(seed) + _CRUMPLE_SALT[1], crumple_scale)
    d_up = np.diff(amp * (up - up[0]))
    d_side = np.diff(amp * (side - side[0]))
    across = np.array([1.0, 0.0, 0.0])
    for j in range(_CRUMPLE_JOINTS - 1, 0, -1):
        sel = t_v > tj[j]
        if not np.any(sel):
            continue
        tangent = spine[min(j + 1, _CRUMPLE_JOINTS)] - spine[j - 1]
        n = np.linalg.norm(tangent)
        if n <= 1e-12:
            continue
        tangent /= n
        normal = np.cross(tangent, across)
        nn = np.linalg.norm(normal)
        normal = normal / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
        rel = out[sel] - spine[j]
        rel = _rotate(rel, across, d_up[j - 1])
        rel = _rotate(rel, normal, d_side[j - 1])
        out[sel] = spine[j] + rel
    return out


# The generate parameters that carry a per-parameter VARIANCE (like the parametric add-on's
# Parameter.variance). Order is FIXED so the seeded jitter is stable regardless of
# which parameters actually have variance set. Keyed by engine-kwarg name.
VARIABLE_PARAMS = (
    "length", "width", "stations", "res_h", "margin_teeth", "margin_depth",
    "curl", "cup", "twist", "gravity", "wave", "wave_count", "noise", "noise_scale",
    # compound leaves -- appended, so every earlier roll stays put
    "rachis_length", "spread", "leaflet_taper",
    # crumple -- appended likewise
    "crumple", "crumple_scale",
)


def randomize(values: dict, variances: dict, seed: int) -> dict:
    """Apply per-parameter variance to `values`, returning a new dict. Each
    parameter is jittered proportionally: effective = base * (1 + r*var), where
    r is uniform in [-1, 1] drawn from `seed`. One draw per VARIABLE_PARAMS entry
    IN ORDER (whether or not it has variance) so a given seed is byte-stable and
    adding variance to one parameter never shifts another's roll. Variance 0 (or a
    key absent from `variances`) leaves the value untouched; non-VARIABLE_PARAMS
    keys pass through. The caller rounds integers and clamps to valid ranges."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    out = dict(values)
    for key in VARIABLE_PARAMS:
        r = float(rng.uniform(-1.0, 1.0))
        var = float(variances.get(key, 0.0))
        if key in values and var > 0.0:
            out[key] = values[key] * (1.0 + r * var)
    return out


def shape_from_preset(data: dict, label: str) -> LeafShape:
    """A `LeafShape` from a saved leaf preset dict (the panel's Save Preset JSON):
    its `profile_samples` peak-normalised, and `aspect` = widest width / length
    (`gen_width * peak / gen_length`) -- the same rule the built-in LEAF_SHAPES
    were baked with, so a custom shape behaves exactly like a default one. A
    preset without a size falls back to the default leaf's proportion. Raises
    ValueError when the preset carries no usable profile."""
    if not isinstance(data, dict):
        raise ValueError(f"{label}: not a leaf preset")
    raw = data.get("profile_samples") or ()
    samples = np.asarray(raw, dtype=np.float64).ravel()
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError(f"{label}: preset has no profile_samples")
    peak = float(samples.max())
    if peak <= 0.0:
        raise ValueError(f"{label}: profile is zero everywhere")
    params = data.get("params") or {}
    length = float(params.get("gen_length", 0.0) or 0.0)
    width = float(params.get("gen_width", 0.0) or 0.0)
    if length > 0.0 and width > 0.0:
        aspect = width * peak / length
    else:
        aspect = DEFAULT_LEAF_WIDTH / DEFAULT_LEAF_LENGTH
    profile = tuple(float(v) for v in np.clip(samples / peak, 0.0, 1.0))
    return LeafShape(label, aspect, profile)


def shape_profile_samples(shape: str, count: int = 17) -> np.ndarray:
    """The baked half-width profile of a named shape resampled to `count` points
    over [0, 1] (base -> tip). Used to seed the panel's Custom Profile curve from
    the current shape so the artist tweaks from what is there rather than a blank
    curve."""
    t = np.linspace(0.0, 1.0, max(2, int(count)))
    return _resample_profile(_resolve_shape(shape).profile, t)


def generate_leaf(
    *,
    length: float = DEFAULT_LEAF_LENGTH,
    width: float = DEFAULT_LEAF_WIDTH,
    shape: str = DEFAULT_LEAF_SHAPE,
    stations: int = DEFAULT_LEAF_STATIONS,
    res_h: int | None = None,
    width_samples: np.ndarray | None = None,
    margin: str = DEFAULT_LEAF_MARGIN,
    margin_teeth: int = DEFAULT_MARGIN_TEETH,
    margin_depth: float = DEFAULT_MARGIN_DEPTH,
    curl: float = 0.0,
    cup: float = 0.0,
    twist: float = 0.0,
    gravity: float = 0.0,
    wave: float = 0.0,
    wave_count: int = DEFAULT_WAVE_COUNT,
    noise: float = 0.0,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    seed: int = DEFAULT_LEAF_SEED,
    crumple: float = 0.0,
    crumple_scale: float = DEFAULT_CRUMPLE_SCALE,
) -> CutLeaf:
    """Build a procedural leaf blade as a venation-following QUAD MESH.

    `length`/`width` : metric size (`width` is the FULL tip-to-tip blade width).
    `shape`          : a `LEAF_SHAPES` key -- its baked half-width profile is the
                       whole outline (peak = `width`); also fixes the venation
                       (`venation_for`), which the grid's edge loops follow. Used
                       when `width_samples` is None. The shape's authored
                       proportion (`LeafShape.aspect`) is NOT applied here -- the
                       caller passes the width it wants.
    `width_samples`  : optional explicit half-width fractions over [0, 1] (base
                       -> tip) -- the curve-widget hook, taken VERBATIM (overrides
                       `shape`).
    `stations`       : VERTICAL resolution -- grid rows along the length. Drives
                       how finely teeth are resolved (teeth sit on the row
                       boundary), so raise it for crisper teeth.
    `res_h`          : HORIZONTAL resolution -- half-columns across the width each
                       side (None = auto from the blade aspect). Independent of
                       `stations`. Both clamped so the mesh stays under
                       MAX_LEAF_VERTS.
    `margin`/`margin_teeth`/`margin_depth` : the edge profile (see `LEAF_MARGINS`);
                       default `entire` (smooth). A toothed margin reserves a smooth
                       margin BAND so teeth never overlap the interior mesh. Applies
                       to a `width_samples` outline too.
    `curl`/`cup`/`twist`/`gravity`/`wave`/`wave_count` : blade deformers (see
                       `_deform_blade`); all default to a flat no-op. `wave` is
                       edge-masked (rigid midrib).
    `noise`/`noise_scale`/`seed` : Perlin-like organic Z deform + the GLOBAL seed
                       (byte-deterministic; `noise` 0 = no-op).
    `crumple`/`crumple_scale` : the whole blade scrunched along its midrib (a
                       dried leaf; see `DEFAULT_CRUMPLE_SCALE`); 0 = no-op.

    Returns a `CutLeaf` carrying a full quad mesh (`faces` non-empty): base at
    the origin, +Y to the tip, midrib the central column, evenly-spaced quads for
    good deformation. `mask` carries the edge/wave weight (0 midrib -> 1 edge).
    Flat in local XY when no deformer is set; a deformer lifts it into 3D. UVs map
    the flat blade into the unit square at ONE scale for U and V (true
    proportions, no stretch; midrib at U 0.5, V base -> tip, centred).
    Raises ValueError on non-positive size or a degenerate outline."""
    if length <= 0.0 or width <= 0.0:
        raise ValueError(f"length and width must be positive; got {length}, {width}")

    margin = _resolve_margin(margin)
    # Vertical (rows) and horizontal (cols) resolution are independent and set
    # directly by the caller -- no teeth-driven override, so the resolution knobs
    # always take effect. cols default follows the blade aspect for ~square quads.
    aspect = float(width) / float(length)
    rows = max(1, int(stations))
    cols = max(1, int(res_h)) if res_h is not None else max(1, round(rows * aspect / 2.0))
    warnings: list[str] = []
    if (rows + 1) * (2 * cols + 1) > MAX_LEAF_VERTS:
        warnings.append(
            f"resolution clamped to stay under the {MAX_LEAF_VERTS}-vertex hard cap")
        while (rows + 1) * (2 * cols + 1) > MAX_LEAF_VERTS and cols > 1:
            cols -= 1
        while (rows + 1) * (2 * cols + 1) > MAX_LEAF_VERTS and rows > 1:
            rows -= 1
    v = np.linspace(0.0, 1.0, rows + 1)
    if width_samples is not None:
        prof = _resample_profile(width_samples, v)          # explicit curve wins
    else:
        prof = _resample_profile(_resolve_shape(shape).profile, v)   # baked shape

    half = 0.5 * float(width) * prof
    y = v * float(length)
    depth_used = min(max(float(margin_depth), 0.0), MARGIN_DEPTH_MAX)
    # Teeth are applied around the whole outline boundary inside _blade_grid (so
    # they wrap the apex), with a smooth margin BAND reserving clearance from the
    # interior. The UV fit includes the teeth, so they never clip the unit square.
    toothed = margin != "entire" and margin_teeth > 0 and depth_used > 0.0
    band = _margin_band(margin, depth_used) if toothed else 0.0
    verts, faces, uvs, across = _blade_grid(
        half, y, length=float(length), width=float(width), cols=cols,
        margin=margin, teeth=int(margin_teeth), depth=depth_used, band=band)

    if not faces:
        raise ValueError("degenerate blade: no faces (all-zero width profile?)")

    # Edge mask (wave influence): smoothstep of the RELATIVE across-blade position
    # (0 on the midrib -> 1 at the edge at EVERY row), so the wavy edge does not
    # fall off where the blade narrows near the tip. Drives the wave and is
    # exported as the red vertex colour for reuse downstream.
    mask = across * across * (3.0 - 2.0 * across)

    verts = _deform_blade(
        verts, length=float(length), width=float(width),
        curl=curl, cup=cup, twist=twist, gravity=gravity,
        wave=wave, wave_count=wave_count, noise=noise, noise_scale=noise_scale,
        seed=seed, wave_mask=mask, crumple=crumple, crumple_scale=crumple_scale)
    leaf = CutLeaf(verts=verts, uvs=uvs, warnings=tuple(warnings), faces=faces, mask=mask)
    if leaf.boundary_count > MAX_LEAF_VERTS:
        raise ValueError(
            f"leaf mesh has {leaf.boundary_count} verts, over the "
            f"{MAX_LEAF_VERTS} hard cap")
    return leaf


# ---------------------------------------------------------------------------
# COMPOUND leaves: leaflets on a rachis
# ---------------------------------------------------------------------------
# A compound leaf is several generated blades -- the LEAFLETS, each exactly the
# blade `generate_leaf` builds (shape / profile, margin, deform) -- set on a thin
# RACHIS strip, all in one mesh (each leaflet its own island). The types:
#   trifoliate  three leaflets: a terminal one at the rachis tip (Rachis Length 0
#               = all three from one point, clover-like) and a lateral pair at
#               the base.
#   palmate     N leaflets fanned from one point (horse chestnut, lupin).
#   pinnate     leaflet pairs along the rachis, with or without a terminal
#               leaflet (odd- / even-pinnate: ash, rose).
#   bipinnate   pinnate of pinnates: each side branch (pinna) is a pinnate
#               rachis of its own (mimosa, acacia).
# One meaning per control across the types: Spread = the lateral (or outermost)
# leaflets' angle from the axis they leave; Taper = the size of the base pair
# (pinnate, trifoliate) / outermost leaflets (palmate) / lowest pinnae
# (bipinnate) relative to the tip or centre one. Leaflet Length/Width are the
# size at scale 1. Every type stands on a PETIOLE: a strip from the base up to
# where the leaflets start (the fan point of a palmate leaf). All leaflets share
# one blade (scaled), so they share its deform; Rachis Curl then bends the whole
# leaf along its length.
# The leaf-local contract holds: base at the origin, +Y along the rachis, flat
# XY until a deformer is set. Leaflets keep the blade's unit-square UVs; the
# rachis and stalks map onto a sliver of the texture's midrib (U 0.5) so the
# leaflet's own material colours them.
COMPOUND_TYPES = ("simple", "trifoliate", "palmate", "pinnate", "bipinnate")
DEFAULT_COMPOUND = "simple"
DEFAULT_LEAFLETS = 5            # palmate: leaflets; pinnate / bipinnate: pairs
DEFAULT_SUB_PAIRS = 6           # bipinnate: leaflet pairs on each pinna
DEFAULT_RACHIS_LENGTH = 0.15
# Trifoliate's "rachis" is only the terminal leaflet's stalk, above the two
# side leaflets: short in real trifoliate leaves (about a quarter leaflet in a
# bean, shorter in clover), so a pinnate-length rachis read as a bare pole
# (user decision 2026-10-18). Picking the type sets Rachis Length to this.
DEFAULT_TRIFOLIATE_RACHIS = 0.03
DEFAULT_PETIOLE = 0.03          # main leaf stalk, base -> first leaflets
DEFAULT_RACHIS_WIDTH = 0.002
DEFAULT_PETIOLULE = 0.004       # leaflet stalk length at scale 1
DEFAULT_SPREAD = 60.0           # degrees
DEFAULT_LEAFLET_TAPER = 0.7
DEFAULT_PINNA_LENGTH = 0.5      # bipinnate: pinna length / rachis length
# The texture sliver the rachis and stalks sample (the leaflet's midrib).
_RACHIS_U = (0.495, 0.505)
_RACHIS_V = (0.02, 0.3)


def _heading(angle: float) -> np.ndarray:
    """Unit direction of `angle` (radians, CCW from +Y)."""
    return np.array([-math.sin(angle), math.cos(angle)])


def default_rachis_length(compound: str) -> float:
    """The Rachis Length a compound type starts at: the trifoliate one's short
    terminal stalk, every other type's shared default."""
    return DEFAULT_TRIFOLIATE_RACHIS if compound == "trifoliate" else DEFAULT_RACHIS_LENGTH


def _compound_layout(compound: str, *, leaflets: int, terminal: bool,
                     sub_pairs: int, rachis_length: float, rachis_width: float,
                     petiolule: float, spread: float, taper: float,
                     pinna_length: float, petiole: float = 0.0):
    """Where each leaflet goes and the stalk strips that carry them:
    ([(base (2,), angle, scale)], [(points (M, 2), widths (M,))])."""
    spread_r = math.radians(spread)
    places: list = []
    strips: list = []

    def stalk(start, angle, scale):
        # a leaflet on a petiolule from `start`, heading `angle`
        end = start + _heading(angle) * (petiolule * scale)
        if petiolule * scale > 0.0:
            strips.append((np.array([start, end]),
                           np.array([0.5, 0.4]) * rachis_width * scale))
        places.append((end, angle, scale))

    def pinnate(origin, angle, length, pairs, scale, inner_taper, sub):
        d = _heading(angle)
        left = _heading(angle + 0.5 * math.pi)
        width = rachis_width * scale
        segs = max(4, 2 * pairs + 2)
        f = np.linspace(0.0, 1.0, segs + 1)
        strips.append((origin + np.outer(f * length, d), width * (1.0 - 0.5 * f)))
        for k in range(pairs):
            at = (k + 0.5) / pairs if terminal else (k + 1) / pairs
            s = scale * (inner_taper + (1.0 - inner_taper) * k / (pairs - 1)
                         if pairs > 1 else 1.0)
            base = origin + d * (length * at)
            for side in (-1.0, 1.0):        # -1: right of the rachis, +1: left
                a = angle + side * spread_r
                start = base + left * (side * 0.5 * width * (1.0 - 0.5 * at))
                if sub:
                    pinnate(start, a, pinna_length * rachis_length * s, sub, s, 1.0, 0)
                else:
                    stalk(start, a, s)
        if terminal:
            tip = origin + d * length
            if sub:
                pinnate(tip, angle, pinna_length * rachis_length * scale, sub, scale, 1.0, 0)
            else:
                stalk(tip, angle, scale)

    origin = np.array([0.0, petiole])
    if petiole > 0.0:                       # the main stalk, a touch wider at its foot
        f = np.linspace(0.0, 1.0, 5)
        strips.append((np.outer(f * petiole, _heading(0.0)),
                       rachis_width * (1.2 - 0.2 * f)))
    if compound == "trifoliate":
        if rachis_length > 0.0:
            f = np.linspace(0.0, 1.0, 5)
            strips.append((origin + np.outer(f * rachis_length, _heading(0.0)),
                           rachis_width * (1.0 - 0.5 * f)))
        for side in (-1.0, 1.0):
            stalk(origin, side * spread_r, taper)
        stalk(origin + np.array([0.0, rachis_length]), 0.0, 1.0)
    elif compound == "palmate":
        n = max(2, int(leaflets))
        for i in range(n):
            u = -1.0 + 2.0 * i / (n - 1)    # -1 .. 1 across the fan
            stalk(origin, u * spread_r, taper + (1.0 - taper) * (1.0 - abs(u)))
    elif compound == "pinnate":
        pinnate(origin, 0.0, rachis_length, max(1, int(leaflets)), 1.0, taper, 0)
    elif compound == "bipinnate":
        pinnate(origin, 0.0, rachis_length, max(1, int(leaflets)), 1.0, taper,
                max(1, int(sub_pairs)))
    else:
        raise ValueError(f"unknown compound leaf type {compound!r}")
    return places, strips


def generate_compound_leaf(
    *, compound: str = DEFAULT_COMPOUND, leaflets: int = DEFAULT_LEAFLETS,
    terminal: bool = True, sub_pairs: int = DEFAULT_SUB_PAIRS,
    rachis_length: float = DEFAULT_RACHIS_LENGTH,
    rachis_width: float = DEFAULT_RACHIS_WIDTH, petiolule: float = DEFAULT_PETIOLULE,
    spread: float = DEFAULT_SPREAD, leaflet_taper: float = DEFAULT_LEAFLET_TAPER,
    pinna_length: float = DEFAULT_PINNA_LENGTH, rachis_curl: float = 0.0,
    petiole: float = DEFAULT_PETIOLE, **blade,
) -> CutLeaf:
    """A leaf of `compound` type (see the section note); `simple` is exactly
    `generate_leaf(**blade)`. `blade` takes `generate_leaf`'s keywords and makes
    the leaflet. The leaflet resolution is lowered (with a warning) so the whole
    leaf stays under MAX_LEAF_VERTS; ValueError if even the coarsest leaflets
    would not fit, or on a degenerate blade."""
    if compound == "simple":
        return generate_leaf(**blade)
    places, strips = _compound_layout(
        compound, leaflets=leaflets, terminal=terminal, sub_pairs=sub_pairs,
        rachis_length=max(0.0, float(rachis_length)),
        rachis_width=max(0.0, float(rachis_width)),
        petiolule=max(0.0, float(petiolule)), spread=float(spread),
        taper=min(max(float(leaflet_taper), 0.05), 1.0),
        pinna_length=max(0.0, float(pinna_length)),
        petiole=max(0.0, float(petiole)))
    n = len(places)
    strip_verts = sum(2 * len(pts) for pts, _w in strips)

    # Leaflet resolution: the blade's own, lowered until every leaflet fits.
    length = float(blade.get("length", DEFAULT_LEAF_LENGTH))
    width = float(blade.get("width", DEFAULT_LEAF_WIDTH))
    rows = max(1, int(blade.get("stations", DEFAULT_LEAF_STATIONS)))
    res_h = blade.get("res_h")
    cols = (max(1, int(res_h)) if res_h is not None
            else max(1, round(rows * (width / length if length > 0 else 1.0) / 2.0)))
    budget = MAX_LEAF_VERTS - strip_verts
    warnings: list[str] = []

    def total(r, c):
        return n * (r + 1) * (2 * c + 1)

    if total(rows, cols) > budget:
        warnings.append(f"leaflet resolution lowered to keep the leaf under the "
                        f"{MAX_LEAF_VERTS}-vertex hard cap")
        while total(rows, cols) > budget and cols > 1:
            cols -= 1
        while total(rows, cols) > budget and rows > 2:
            rows -= 1
        if total(rows, cols) > budget:
            raise ValueError(f"{n} leaflets do not fit the {MAX_LEAF_VERTS}-vertex "
                             f"hard cap -- use fewer leaflets")
    leaflet = generate_leaf(**{**blade, "stations": rows, "res_h": cols})
    warnings += list(leaflet.warnings)

    verts, uvs, faces, masks = [], [], [], []
    count = 0
    lv = np.asarray(leaflet.verts, dtype=np.float64)
    lmask = (np.asarray(leaflet.mask, dtype=np.float64) if leaflet.mask is not None
             else np.zeros(len(lv)))
    for base, angle, scale in places:
        c, s = math.cos(angle), math.sin(angle)
        v = lv * scale
        x, y = v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c
        verts.append(np.column_stack([x + base[0], y + base[1], v[:, 2]]))
        uvs.append(leaflet.uvs)
        masks.append(lmask)
        faces += [tuple(i + count for i in f) for f in leaflet.faces]
        count += len(v)
    for pts, w in strips:
        d = np.gradient(pts, axis=0)
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-30)[:, None]
        left = np.column_stack([-d[:, 1], d[:, 0]]) * (0.5 * w)[:, None]
        m = len(pts)
        right_pts, left_pts = pts - left, pts + left
        verts.append(np.column_stack([np.vstack([right_pts, left_pts]), np.zeros(2 * m)]))
        t = np.linspace(_RACHIS_V[0], _RACHIS_V[1], m)
        uvs.append(np.column_stack([np.r_[np.full(m, _RACHIS_U[1]), np.full(m, _RACHIS_U[0])],
                                    np.r_[t, t]]))
        masks.append(np.zeros(2 * m))
        # right_i, right_i+1, left_i+1, left_i: CCW seen from +Z
        faces += [(count + i, count + i + 1, count + m + i + 1, count + m + i)
                  for i in range(m - 1)]
        count += 2 * m
    allv = np.vstack(verts)
    if rachis_curl:
        reach = float(allv[:, 1].max())
        if reach > 0.0:
            allv = _deform_blade(allv, length=reach, width=1.0, curl=rachis_curl)
    out = CutLeaf(verts=allv, uvs=np.vstack(uvs), warnings=tuple(dict.fromkeys(warnings)),
                  faces=tuple(faces), mask=np.concatenate(masks))
    if out.boundary_count > MAX_LEAF_VERTS:
        raise ValueError(f"compound leaf has {out.boundary_count} verts, over the "
                         f"{MAX_LEAF_VERTS} hard cap")
    return out


# ---------------------------------------------------------------------------
# CUTOUT -> deformable triangle mesh
# ---------------------------------------------------------------------------
# A traced cutout is an exact outline loop with atlas UVs. To deform it like a
# generated blade it is rebuilt as an even, near-equilateral triangle mesh --
# isotropic, so it bends alike in every direction (no rows or columns a curl or
# a twist has to line up with):
#   * The SILHOUETTE comes first: the traced outline, simplified one-sided (it
#     may lose a sliver of leaf, never grow over alpha) and freed of pixel hooks,
#     is the mesh's boundary. Every kept point stays; sides longer than the
#     target edge are split evenly.
#   * The inside is a hexagonal lattice at the target edge length (the leaf's
#     size / Density), kept more than half an edge clear of the boundary.
#   * A constrained Delaunay triangulation joins them (the outline's edges are
#     never flipped). A triangle that is both badly shaped and large gets a
#     vertex at its circumcentre, so fine edge detail grades smoothly into the
#     lattice; then the inner vertices are relaxed.
# Deform centre line and edge mask are smooth functions of POSITION (the
# outline's outer extents at each height, smoothed along the length), so the
# sheet bends without a step where a lobe begins.
DEFAULT_CUT_DENSITY = 20
# Smoothing floor as a fraction of the leaf's size.
_CUT_TOL = 0.0025
# Inner lattice points closer than this (in target edges) to the outline are
# dropped: past half an edge none can sit in an outline edge's diametral circle,
# so the outline's edges are always Delaunay edges.
_CUT_CLEAR = 0.6
# Refinement: a triangle whose smallest angle is under this AND whose
# circumradius is over _CUT_MIN_R target edges gets its circumcentre.
_CUT_MIN_ANGLE = 22.0
_CUT_MIN_R = 0.3


def uv_affine(verts: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """The (3, 2) matrix A with [u, v] = [x, y, 1] @ A, fitted to a cut leaf's
    boundary -- exact, since `_to_leaf_local` makes UV an affine map of XY."""
    v = np.asarray(verts, dtype=np.float64)
    pts = np.column_stack([v[:, 0], v[:, 1], np.ones(len(v))])
    a, *_ = np.linalg.lstsq(pts, np.asarray(uvs, dtype=np.float64), rcond=None)
    return a


def _row_crossings(poly: np.ndarray, y: float) -> np.ndarray:
    """x of every place the closed polygon `poly` (K, 2) crosses height `y`,
    sorted. Half-open (an edge counts when it spans y with one end at or below
    it), so a row exactly at a vertex samples the outline just ABOVE that height."""
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    hit = (y0 <= y) != (y1 <= y)
    t = (y - y0[hit]) / (y1[hit] - y0[hit])
    return np.sort(x0[hit] + t * (x1[hit] - x0[hit]))


def _simplify_closed(poly: np.ndarray, eps: float, inner: bool = False,
                     outward: float = 0.0) -> np.ndarray:
    """Douglas-Peucker on a closed loop: keeps every point that sits more than
    `eps` off the simplified edge, so the result stays within `eps` of `poly`.
    `inner` (for a CCW loop): a simplified edge may pass OUTSIDE a dropped point
    by at most `outward` -- the loop shrinks (a sliver of leaf) but never grows
    over alpha by more than that (a sub-pixel `outward` lets trace noise go)."""
    k = len(poly)
    if k <= 4:
        return poly
    a = int(np.argmin(poly[:, 0]))
    b = int(np.argmax(np.hypot(*(poly - poly[a]).T)))
    keep = np.zeros(k, dtype=bool)
    keep[a] = keep[b] = True
    stack = [(a, b), (b, a)]
    while stack:
        i, j = stack.pop()
        idx = np.arange(i + 1, j if j > i else j + k) % k
        if idx.size == 0:
            continue
        p, q = poly[i], poly[j]
        dx, dy = q - p
        norm = np.hypot(dx, dy)
        pts = poly[idx]
        if norm == 0.0:
            dist = np.hypot(*(pts - p).T)
            side = np.zeros_like(dist)
        else:
            side = (dx * (pts[:, 1] - p[1]) - dy * (pts[:, 0] - p[0])) / norm
            dist = np.abs(side)
        if inner:
            # A point on the interior (left) side would leave the edge outside it:
            # split at the DEEPEST such point (splitting at the first would crawl
            # along a concave run keeping every point, even on straight sides).
            viol = side > max(outward, 1e-3 * eps)
            if viol.any():
                m = int(np.argmax(np.where(viol, side, -np.inf)))
                keep[idx[m]] = True
                stack.append((i, int(idx[m])))
                stack.append((int(idx[m]), j))
                continue
        m = int(np.argmax(dist))
        if dist[m] > eps:
            keep[idx[m]] = True
            stack.append((i, int(idx[m])))
            stack.append((int(idx[m]), j))
    out = poly[keep]
    return out if len(out) >= 3 else poly


def _extrema(poly: np.ndarray) -> list[tuple[int, int, bool, bool]]:
    """Local y-extrema of a CCW loop in traversal order, as (first, last, is_max,
    convex): a flat run is one extremum spanning first..last. Convex = the top or
    bottom of a piece (`end` / `start`); reflex = a notch apex or sinus."""
    k = len(poly)
    ys = poly[:, 1]
    out = []
    start = next((i for i in range(k) if ys[i] != ys[i - 1]), None)
    if start is None:
        return out
    i = start
    for _ in range(k):
        j = i
        while ys[(j + 1) % k] == ys[i % k] and (j + 1 - i) < k:
            j += 1
        a, b = i % k, j % k
        prev_y, next_y = ys[(a - 1) % k], ys[(b + 1) % k]
        y = ys[a]
        if (prev_y - y) * (next_y - y) > 0:
            direction = poly[(b + 1) % k, 0] - poly[(a - 1) % k, 0]
            is_max = prev_y < y
            convex = (direction < 0) if is_max else (direction > 0)
            out.append((a, b, bool(is_max), bool(convex)))
        i = j + 1
        if (i - start) >= k:
            break
    return out


def _flatten_hooks(poly: np.ndarray, h_min: float) -> np.ndarray:
    """Remove turning-point pairs less than `h_min` apart in height: the convex
    one (the tip) is trimmed to the reflex one's level (the sinus), together with
    every vertex on the way that pokes past that level -- so the outline only
    ever loses a sliver of leaf, never gains alpha."""
    poly = poly.copy()
    k = len(poly)
    for _ in range(4 * k):
        ext = _extrema(poly)
        m = len(ext)
        if m < 4:
            break
        done = True
        for t in range(m):
            c, r = ext[t], ext[(t + 1) % m]
            for cvx, rfx, other in ((c, r, ext[(t - 1) % m]), (r, c, ext[(t + 2) % m])):
                if not cvx[3] or rfx[3] or cvx[2] == rfx[2]:
                    continue
                y_r = poly[rfx[0], 1]
                if abs(poly[cvx[0], 1] - y_r) >= h_min:
                    continue
                # clamp the path from the extremum beyond the tip to the sinus
                lo, hi = (other[1], rfx[0]) if (other is ext[(t - 1) % m]) else (rfx[1], other[0])
                idx = np.arange(lo, hi + (0 if hi >= lo else k) + 1) % k
                if cvx[2]:
                    poly[idx, 1] = np.minimum(poly[idx, 1], y_r)
                else:
                    poly[idx, 1] = np.maximum(poly[idx, 1], y_r)
                done = False
                break
            if not done:
                break
        if done:
            break
    return poly


def cutout_grid(
    outline: np.ndarray, uv_map: np.ndarray, *,
    density: int = DEFAULT_CUT_DENSITY,
    curl: float = 0.0, cup: float = 0.0, twist: float = 0.0, gravity: float = 0.0,
    wave: float = 0.0, wave_count: int = DEFAULT_WAVE_COUNT,
    noise: float = 0.0, noise_scale: float = DEFAULT_NOISE_SCALE,
    seed: int = DEFAULT_LEAF_SEED, pixel: float = 0.0,
    crumple: float = 0.0, crumple_scale: float = DEFAULT_CRUMPLE_SCALE,
) -> CutLeaf:
    """Rebuild a traced cutout as a deformable triangle mesh (see the section note).

    `outline` : the cut leaf's boundary loop in leaf-local space ((K, 2) or (K, 3)).
    `uv_map`  : its (3, 2) affine XY -> atlas-UV map (`uv_affine`).
    `density` : target edges across the leaf's larger dimension (its length or
    its width); the target edge length is that dimension / density.
    `pixel`   : the atlas pixel size in leaf units (1 / pixels-per-unit), if known:
    the traced edge's 1-px jaggies are smoothed away (within 1.5 px).
    The deformers are the generated blade's (`_deform_blade`), all 0 = flat.
    Returns a `CutLeaf` with triangle `faces`, atlas `uvs` and the edge `mask`.
    Raises ValueError on a degenerate outline."""
    poly = np.asarray(outline, dtype=np.float64)[:, :2].copy()
    if poly.shape[0] < 3:
        raise ValueError("cutout outline needs at least 3 points")
    if _signed_area(poly) < 0.0:
        poly = poly[::-1].copy()                       # CCW (the contract's winding)
    span = float(np.ptp(poly[:, 1]))
    width = float(np.ptp(poly[:, 0]))
    if span <= 0.0 or width <= 0.0:
        raise ValueError("degenerate cutout outline (no area)")
    tol = max(_CUT_TOL * max(span, width), float(pixel))
    smooth = max(0.5 * tol, 1.5 * float(pixel))
    # One-sided smoothing (outward by at most a quarter pixel -- invisible, but
    # it lets the trace's sub-pixel noise go), then pixel hooks trimmed.
    outward = 0.25 * float(pixel) if pixel else 0.1 * smooth
    poly = _flatten_hooks(_simplify_closed(poly, smooth, inner=True, outward=outward),
                          2.0 * smooth)
    # Pixel-stair corners closer than a fifth of an edge: trimmed within a pixel.
    if pixel:
        h0 = max(np.ptp(poly[:, 1]), np.ptp(poly[:, 0])) / max(1, int(density))
        poly = _thin_outline(poly, 0.2 * h0, float(pixel), outward)
    # The leaf actually meshed (smoothing may pull an end inward).
    y_lo, y_hi = float(poly[:, 1].min()), float(poly[:, 1].max())
    span = y_hi - y_lo
    width = float(np.ptp(poly[:, 0]))
    warnings: list[str] = []

    h = max(span, width) / max(1, int(density))
    area = abs(_signed_area(poly))
    sides = np.hypot(*(np.roll(poly, -1, axis=0) - poly).T)

    def estimate(edge: float) -> float:
        return area / (0.866 * edge * edge) + float(np.ceil(sides / edge).sum())

    budget = 0.8 * MAX_LEAF_VERTS                      # room for the refinement
    if estimate(h) > budget:
        warnings.append(
            f"density clamped to stay under the {MAX_LEAF_VERTS}-vertex hard cap")
        while estimate(h) > budget:
            h *= 1.05
    built = _tri_mesh(poly, h, MAX_LEAF_VERTS)
    if built is None:
        raise ValueError("degenerate cutout outline (could not be triangulated)")
    pts, faces = built

    varr = np.column_stack([pts, np.zeros(len(pts))])
    mids, across = _centre_line(poly, pts, y_lo, y_hi)
    mask = across * across * (3.0 - 2.0 * across)
    uvs = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))]) @ np.asarray(
        uv_map, dtype=np.float64)
    flat = varr.copy()
    flat[:, 1] -= y_lo                                 # the deformers expect base at 0
    bent = _deform_blade(
        flat, length=span, width=width,
        curl=curl, cup=cup, twist=twist, gravity=gravity,
        wave=wave, wave_count=wave_count, noise=noise, noise_scale=noise_scale,
        seed=seed, wave_mask=mask, midline=mids,
        crumple=crumple, crumple_scale=crumple_scale)
    bent[:, 1] += y_lo
    return CutLeaf(verts=bent, uvs=uvs, warnings=tuple(dict.fromkeys(warnings)),
                   faces=tuple(faces), mask=mask)


def _centre_line(poly: np.ndarray, xy: np.ndarray, y_lo: float, y_hi: float,
                 samples: int = 129) -> tuple[np.ndarray, np.ndarray]:
    """Per point of `xy`: the x of the leaf's centre line at its height, and how
    far across the half-width it sits (0 on the centre line .. 1 at the edge).
    The centre line is a quadratic in height fitted to each slice's centroid,
    weighted by the slice's width -- it follows a leaning or curved blade but
    never steps where a lobe begins or ends (a step would crease every bend).
    The half-width is the outline's outer reach from it, smoothed along the
    length."""
    span = y_hi - y_lo
    ys = np.linspace(y_lo, y_hi, samples)
    lo = np.full(samples, np.nan)
    hi = np.full(samples, np.nan)
    cen = np.full(samples, np.nan)
    wid = np.zeros(samples)
    for i, y in enumerate(ys):
        xs = _row_crossings(poly, min(max(y, y_lo + 1e-6 * span), y_hi - 1e-6 * span))
        if len(xs) < 2:
            continue
        a, b = xs[0:len(xs) - 1:2], xs[1::2]
        w = float((b - a).sum())
        if w <= 0.0:
            continue
        lo[i], hi[i] = xs[0], xs[-1]
        cen[i] = float((0.5 * (a + b) * (b - a)).sum()) / w
        wid[i] = w
    ok = ~np.isnan(cen)
    t = (ys - y_lo) / span
    deg = min(2, int(ok.sum()) - 1)
    coef = np.polyfit(t[ok], cen[ok], deg, w=np.sqrt(wid[ok])) if deg >= 0 else [0.0]
    mid = np.polyval(coef, t)
    reach = np.interp(ys, ys[ok], np.maximum(mid[ok] - lo[ok], hi[ok] - mid[ok]))
    sigma = 0.04 * (samples - 1)
    r = int(3 * sigma)
    kern = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    kern /= kern.sum()
    half = np.convolve(np.pad(reach, r, mode="edge"), kern, mode="valid")
    m = np.polyval(coef, (xy[:, 1] - y_lo) / span)
    hw = np.interp(xy[:, 1], ys, half)
    across = np.where(hw > 0.0, np.minimum(1.0, np.abs(xy[:, 0] - m) / np.maximum(hw, 1e-30)),
                      0.0)
    return m, across


def _thin_outline(poly: np.ndarray, min_len: float, cut: float,
                  outward: float) -> np.ndarray:
    """Drop outline points that end a side shorter than `min_len` (pixel-stair
    corners the one-sided smoothing had to keep), where the chord that replaces
    them only trims the leaf -- by at most `cut` -- or grows it by at most
    `outward`, measured against every point of `poly` it replaces."""
    k = len(poly)
    keep = list(range(k))
    for _sweep in range(8):
        changed = False
        i = 0
        while i < len(keep) and len(keep) > 3:
            m = len(keep)
            pi, ci, ni = keep[i - 1], keep[i], keep[(i + 1) % m]
            p, c, q = poly[pi], poly[ci], poly[ni]
            if min(math.hypot(*(c - p)), math.hypot(*(q - c))) >= min_len:
                i += 1
                continue
            idx = np.arange(pi + 1, ni if ni > pi else ni + k) % k
            d = q - p
            n = math.hypot(*d)
            if n == 0.0:
                i += 1
                continue
            side = (d[0] * (poly[idx, 1] - p[1]) - d[1] * (poly[idx, 0] - p[0])) / n
            # right of the chord (outside it): trimmed off; left: alpha gained
            if (-side).max() <= cut and side.max() <= outward:
                del keep[i]
                changed = True
            else:
                i += 1
        if not changed:
            break
    return poly[keep]


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _incircle(a, b, c, d) -> float:
    """> 0 when `d` lies inside the circumcircle of the CCW triangle a, b, c."""
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    return ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
            + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
            + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))


def _tri_mesh(poly: np.ndarray, h: float, max_verts: int):
    """(points (N, 2), CCW triangles) meshing the CCW loop `poly` at target edge
    length `h` (see the section note), or None if it cannot be triangulated."""
    k = len(poly)
    nxt = np.roll(poly, -1, axis=0)
    sides = np.hypot(*(nxt - poly).T)
    bnd = []
    for i in range(k):                     # every outline point; long sides split
        n = max(1, int(math.ceil(sides[i] / h - 1e-9)))
        for j in range(n):
            bnd.append(poly[i] + (nxt[i] - poly[i]) * (j / n))
    bnd = np.asarray(bnd)
    bnd = bnd[np.hypot(*(bnd - np.roll(bnd, 1, axis=0)).T) > 1e-9 * h]
    nb = len(bnd)
    if nb < 3:
        return None
    centre = 0.5 * (bnd.min(axis=0) + bnd.max(axis=0))
    unit = (bnd - centre) / h                      # work in target-edge units
    mesh = _TriMesh([(float(x), float(y)) for x, y in unit])
    tris = _ear_fill(mesh.p, nb, closed=True) or _ear_fill(mesh.p, nb, closed=False)
    if not tris:
        return None
    for tri in tris:
        mesh.add(*tri)
    for i in range(nb):
        j = (i + 1) % nb
        mesh.fixed.update(((i, j), (j, i)))
    mesh.legalize(list(mesh.edge))                 # the outline's CDT
    seg_a, seg_b = unit, np.roll(unit, -1, axis=0)
    t = 0
    for q in _lattice(seg_a, seg_b):
        if len(mesh.p) >= max_verts:
            break
        got = mesh.insert((float(q[0]), float(q[1])), t)
        if got is not None:
            t = got
    mesh.refine(seg_a, seg_b, max_verts)
    mesh.relax(nb)
    pts = np.asarray(mesh.p, dtype=np.float64) * h + centre
    return pts, list(mesh.tris)


def _inside(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Even-odd: which points of `q` (N, 2) lie inside the loop a -> b."""
    px, py = q[:, 0:1], q[:, 1:2]
    hit = (a[:, 1] > py) != (b[:, 1] > py)
    dy = b[:, 1] - a[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = a[:, 0] + (py - a[:, 1]) * (b[:, 0] - a[:, 0]) / dy
    return ((hit & (px < xint)).sum(axis=1) % 2) == 1


def _seg_dist(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of `q` (N, 2) to the nearest segment a -> b."""
    px, py = q[:, 0:1], q[:, 1:2]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(axis=1), 1e-30)
    t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / l2, 0.0, 1.0)
    return np.hypot(a[:, 0] + t * ab[:, 0] - px, a[:, 1] + t * ab[:, 1] - py).min(axis=1)


def _lattice(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The unit hexagonal lattice inside the loop a -> b (target-edge units,
    centred on the origin), clear of the loop by `_CUT_CLEAR`, row by row."""
    lo, hi = a.min(axis=0), a.max(axis=0)
    dy = math.sqrt(3.0) / 2.0
    rows = []
    for r in range(math.floor(lo[1] / dy), math.ceil(hi[1] / dy) + 1):
        xs = np.arange(math.floor(lo[0]) - 1, math.ceil(hi[0]) + 2) + 0.5 * (r % 2)
        rows.append(np.column_stack([xs, np.full(len(xs), r * dy)]))
    pts = np.concatenate(rows)
    keep = np.zeros(len(pts), dtype=bool)
    for s in range(0, len(pts), 256):
        q = pts[s:s + 256]
        keep[s:s + 256] = _inside(q, a, b) & (_seg_dist(q, a, b) >= _CUT_CLEAR)
    return pts[keep]


def _ear_fill(p: list, n: int, closed: bool) -> list | None:
    """Triangulate the simple CCW polygon p[0..n) by ear clipping (any valid
    triangulation -- the Delaunay flips make it good). Points that are not
    strictly convex are bucketed on a grid for the ear tests. `closed`: a point
    on an ear's edge blocks it (no T-junctions); False is the fallback for an
    outline that touches itself. None if it gets stuck."""
    prev = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    xs = [p[i][0] for i in range(n)]
    ys = [p[i][1] for i in range(n)]
    x0, y0 = min(xs), min(ys)
    cs = max(2.0, max(max(xs) - x0, max(ys) - y0) / 32.0)
    eps = 1e-12

    def key(i):
        return int((p[i][0] - x0) // cs), int((p[i][1] - y0) // cs)

    def convex(i):
        return _cross(p[prev[i]], p[i], p[nxt[i]]) > eps

    grid: dict = defaultdict(set)
    concave = set()
    for i in range(n):
        if not convex(i):
            concave.add(i)
            grid[key(i)].add(i)

    def blocked(a, b, c):
        pa, pb, pc = p[a], p[b], p[c]
        gx0 = int((min(pa[0], pb[0], pc[0]) - x0) // cs)
        gx1 = int((max(pa[0], pb[0], pc[0]) - x0) // cs)
        gy0 = int((min(pa[1], pb[1], pc[1]) - y0) // cs)
        gy1 = int((max(pa[1], pb[1], pc[1]) - y0) // cs)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                for j in grid.get((gx, gy), ()):
                    if j == a or j == b or j == c:
                        continue
                    q = p[j]
                    if q == pa or q == pb or q == pc:
                        continue                   # a coincident duplicate point
                    c1, c2, c3 = _cross(pa, pb, q), _cross(pb, pc, q), _cross(pc, pa, q)
                    if closed:
                        if c1 >= -eps and c2 >= -eps and c3 >= -eps:
                            return True
                    elif c1 > eps and c2 > eps and c3 > eps:
                        return True
        return False

    tris = []
    remaining, i, misses = n, 0, 0
    while remaining > 3:
        a, c = prev[i], nxt[i]
        if i not in concave and not blocked(a, i, c):
            tris.append((a, i, c))
            nxt[a], prev[c] = c, a
            remaining -= 1
            misses = 0
            for v in (a, c):                       # a clipped ear only sharpens them
                if v in concave and convex(v):
                    concave.discard(v)
                    grid[key(v)].discard(v)
            i = c
        else:
            i = nxt[i]
            misses += 1
            if misses > remaining:
                return None
    a, c = prev[i], nxt[i]
    if _cross(p[a], p[i], p[c]) > eps:
        tris.append((a, i, c))
    return tris


class _TriMesh:
    """A 2-D triangulation under edits: CCW triangles, a directed edge ->
    triangle map, and the fixed (outline) edges that are never flipped."""

    def __init__(self, pts: list):
        self.p = pts
        self.tris: list[tuple[int, int, int]] = []
        self.edge: dict[tuple[int, int], int] = {}
        self.fixed: set[tuple[int, int]] = set()

    def add(self, a, b, c) -> int:
        t = len(self.tris)
        self.tris.append((a, b, c))
        self.edge[(a, b)] = self.edge[(b, c)] = self.edge[(c, a)] = t
        return t

    def _set(self, t, a, b, c):
        o = self.tris[t]
        for e in ((o[0], o[1]), (o[1], o[2]), (o[2], o[0])):
            if self.edge.get(e) == t:
                del self.edge[e]
        self.tris[t] = (a, b, c)
        self.edge[(a, b)] = self.edge[(b, c)] = self.edge[(c, a)] = t

    def _apex(self, t, a, b) -> int:
        for v in self.tris[t]:
            if v != a and v != b:
                return v
        raise ValueError("degenerate triangle")

    def legalize(self, edges: list):
        """Lawson flips until every non-fixed edge in (and reached from) `edges`
        is locally Delaunay."""
        p, edge, fixed = self.p, self.edge, self.fixed
        guard = 40 * (len(self.tris) + len(edges)) + 100
        while edges and guard > 0:
            guard -= 1
            a, b = edges.pop()
            if (a, b) in fixed:
                continue
            t1, t2 = edge.get((a, b)), edge.get((b, a))
            if t1 is None or t2 is None:
                continue
            c, d = self._apex(t1, a, b), self._apex(t2, b, a)
            # d inside abc's circle (and across ab) => the quad is convex: flip.
            if _incircle(p[a], p[b], p[c], p[d]) <= 1e-10:
                continue
            self._set(t1, a, d, c)
            self._set(t2, d, b, c)
            edges += [(a, d), (d, b), (b, c), (c, a)]

    def _contains(self, t, q) -> bool:
        a, b, c = self.tris[t]
        p = self.p
        return (_cross(p[a], p[b], q) >= -1e-12 and _cross(p[b], p[c], q) >= -1e-12
                and _cross(p[c], p[a], q) >= -1e-12)

    def locate(self, q, t: int) -> int | None:
        """The triangle holding `q`: a walk from triangle `t`, else a scan (the
        walk can be blocked by the outline in a concave leaf). None = outside."""
        p, tris, edge = self.p, self.tris, self.edge
        t = min(t, len(tris) - 1)
        for _ in range(len(tris) + 8):
            a, b, c = tris[t]
            for u, v in ((a, b), (b, c), (c, a)):
                if _cross(p[u], p[v], q) < -1e-12:
                    t2 = edge.get((v, u))
                    if t2 is None:
                        return self._scan(q)
                    t = t2
                    break
            else:
                return t
        return self._scan(q)

    def _scan(self, q) -> int | None:
        return next((t for t in range(len(self.tris)) if self._contains(t, q)), None)

    def insert(self, q, t0: int) -> int | None:
        """Add point `q` (kept Delaunay). Returns a triangle at it, or None if
        it is outside or on the outline (not inserted)."""
        t = self.locate(q, t0)
        if t is None:
            return None
        p = self.p
        a, b, c = self.tris[t]
        i = len(p)
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            luv = math.hypot(p[v][0] - p[u][0], p[v][1] - p[u][1])
            if abs(_cross(p[u], p[v], q)) > 1e-9 * luv:
                continue
            # On edge u-v: split it and the triangle across it.
            if (u, v) in self.fixed:
                return None
            t2 = self.edge.get((v, u))
            if t2 is None:
                return None
            x = self._apex(t2, v, u)
            p.append(q)
            self._set(t, i, v, w)
            self.add(u, i, w)
            self._set(t2, v, i, x)
            self.add(i, u, x)
            self.legalize([(v, w), (w, u), (u, x), (x, v)])
            return t
        p.append(q)
        self._set(t, a, b, i)
        self.add(b, c, i)
        self.add(c, a, i)
        self.legalize([(a, b), (b, c), (c, a)])
        return t

    def refine(self, seg_a: np.ndarray, seg_b: np.ndarray, max_verts: int):
        """Circumcentres into triangles both badly shaped and large (see the
        section constants), skipping any that would fall outside the leaf or
        inside an outline edge's diametral circle (the silhouette is fixed)."""
        p = self.p
        sin_min = math.sin(math.radians(_CUT_MIN_ANGLE))
        mid = 0.5 * (seg_a + seg_b)
        rad2 = 0.25 * ((seg_b - seg_a) ** 2).sum(axis=1)
        for _pass in range(4):
            bad = []
            for t, (a, b, c) in enumerate(self.tris):
                pa, pb, pc = p[a], p[b], p[c]
                area2 = _cross(pa, pb, pc)
                if area2 <= 0.0:
                    continue
                s = sorted((math.hypot(pb[0] - pa[0], pb[1] - pa[1]),
                            math.hypot(pc[0] - pb[0], pc[1] - pb[1]),
                            math.hypot(pa[0] - pc[0], pa[1] - pc[1])))
                quality = area2 / (s[1] * s[2])            # sin of the smallest angle
                if quality < sin_min and s[0] * s[1] * s[2] / (2.0 * area2) > _CUT_MIN_R:
                    bad.append((quality, t, (a, b, c)))
            if not bad:
                return
            bad.sort()
            cands, keys = [], []
            for _q, t, tri in bad:
                pa, pb, pc = (p[v] for v in tri)
                bx, by = pb[0] - pa[0], pb[1] - pa[1]
                cx, cy = pc[0] - pa[0], pc[1] - pa[1]
                d = 2.0 * (bx * cy - by * cx)
                ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d
                uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d
                cands.append((pa[0] + ux, pa[1] + uy))
                keys.append((t, tri))
            cc = np.asarray(cands)
            ok = _inside(cc, seg_a, seg_b)
            for s in range(0, len(cc), 256):
                q = cc[s:s + 256]
                enc = (((q[:, None, :] - mid[None, :, :]) ** 2).sum(axis=2)
                       < rad2[None, :]).any(axis=1)
                ok[s:s + 256] &= ~enc
            added = 0
            for (t, tri), q, good in zip(keys, cands, ok, strict=True):
                if not good or self.tris[t] != tri:
                    continue                       # outside / encroaching / already split
                if len(p) >= max_verts:
                    return
                if self.insert(q, t) is not None:
                    added += 1
            if not added:
                return

    def relax(self, first: int, iters: int = 3):
        """Move each inner vertex (index >= `first`) halfway to the mean of its
        neighbours, unless that would fold a triangle; re-Delaunay after each
        sweep."""
        p = self.p
        n = len(p)
        for _ in range(iters):
            ring: list[list[int]] = [[] for _ in range(n)]
            fan: list[list[int]] = [[] for _ in range(n)]
            for (a, b) in self.edge:
                ring[a].append(b)
            for t, tri in enumerate(self.tris):
                for v in tri:
                    fan[v].append(t)
            for v in range(first, n):
                nb = ring[v]
                if not nb:
                    continue
                mx = sum(p[u][0] for u in nb) / len(nb)
                my = sum(p[u][1] for u in nb) / len(nb)
                old = p[v]
                p[v] = (0.5 * (old[0] + mx), 0.5 * (old[1] + my))
                for t in fan[v]:
                    a, b, c = self.tris[t]
                    if _cross(p[a], p[b], p[c]) <= 1e-6:
                        p[v] = old
                        break
            self.legalize(list(self.edge))
