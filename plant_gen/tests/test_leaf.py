"""Tests for the leaf cutout/generate engine (`growth/leaf.py`).

Locks the three things a screenshot cannot cheaply prove and a later refactor
could silently break: the leaf LOCAL-SPACE CONTRACT, the MAX_LEAF_VERTS hard
cap, and that the lobed-leaf boundary tracer preserves concavities the fast
scanline-rail fills. All numeric -- this project's standard for exactly these
claims.
"""

from __future__ import annotations

import numpy as np

from ..growth.leaf import (
    DEFAULT_BOUNDARY_VERTS,
    LEAF_SHAPES,
    MAX_LEAF_VERTS,
    VARIABLE_PARAMS,
    cut_atlas,
    cut_leaf,
    generate_leaf,
    randomize,
)

# --- fixtures (synthetic alpha planes) -------------------------------------

def _diamond(h: int = 400, w: int = 200) -> np.ndarray:
    """A single-lobe (convex-ish) diamond, tip up -- the scanline-rail case."""
    yy, xx = np.mgrid[0:h, 0:w]
    halfw = (1.0 - np.abs((yy - h / 2) / (h / 2))) * (w * 0.4)
    a = np.zeros((h, w), np.float64)
    a[np.abs(xx - w / 2) <= halfw] = 1.0
    return a


def _bilobed(h: int = 300, w: int = 200) -> np.ndarray:
    """Two lobes over a common base, with a deep notch between them -- rows in
    the lobe band have TWO opaque intervals, which the rail cannot resolve."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w), np.float64)
    a[(yy >= 150) & (np.abs(xx - 100) <= 70)] = 1.0            # base band
    a[(yy >= 60) & (yy < 150) & (np.abs(xx - 60) <= 30)] = 1.0  # left lobe
    a[(yy >= 60) & (yy < 150) & (np.abs(xx - 140) <= 30)] = 1.0  # right lobe
    return a


def _point_in_polygon(poly: np.ndarray, p: tuple[float, float]) -> bool:
    """Even-odd ray cast; `poly` is (N, 2) open loop, `p` is (x, y)."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


# --- the local-space contract ----------------------------------------------

def test_contract_flat_base_at_origin_tip_up():
    leaf = cut_leaf(_diamond(), atlas_size=(200, 400))
    assert np.all(leaf.verts[:, 2] == 0.0), "not flat in local XY (Z != 0)"
    # base 'bottom' (tip up): base sits at the origin, blade extends to +Y.
    assert abs(float(leaf.verts[:, 1].min())) < 1e-9, "base not at the origin"
    assert float(leaf.verts[:, 1].max()) > 0.0, "blade does not run along +Y"
    # midrib-centred X: the blade straddles X == 0.
    assert leaf.verts[:, 0].min() < 0.0 < leaf.verts[:, 0].max()


def test_base_top_flips_the_axis():
    a = _diamond()
    bottom = cut_leaf(a, atlas_size=(200, 400), base="bottom")
    top = cut_leaf(a, atlas_size=(200, 400), base="top")
    # Both keep the base at the origin; the length just measures from the other
    # image end, so both still start at Y == 0 and grow +Y.
    assert abs(float(bottom.verts[:, 1].min())) < 1e-9
    assert abs(float(top.verts[:, 1].min())) < 1e-9
    # The two are genuinely different reorientations, not the same array.
    assert not np.allclose(bottom.verts, top.verts)


def test_metric_scale_is_uniform_and_ppu_driven():
    a = _diamond()
    coarse = cut_leaf(a, atlas_size=(200, 400), pixels_per_unit=2000.0)
    fine = cut_leaf(a, atlas_size=(200, 400), pixels_per_unit=4000.0)
    # Doubling pixels-per-unit halves the metric size, uniformly.
    ratio = float(coarse.verts[:, 1].max()) / float(fine.verts[:, 1].max())
    assert abs(ratio - 2.0) < 1e-6


def test_uv_stays_atlas_relative_in_unit_square():
    leaf = cut_leaf(_diamond(), atlas_size=(200, 400))
    assert leaf.uvs.shape == (leaf.boundary_count, 2)
    assert leaf.uvs.min() >= 0.0 and leaf.uvs.max() <= 1.0


# --- the MAX_LEAF_VERTS hard cap --------------------------------------------

def test_default_budget_is_respected():
    leaf = cut_leaf(_diamond(), atlas_size=(200, 400))
    assert leaf.boundary_count == DEFAULT_BOUNDARY_VERTS


def test_over_budget_request_clamps_to_cap_with_warning():
    leaf = cut_leaf(_diamond(), atlas_size=(200, 400), boundary_verts=10 * MAX_LEAF_VERTS)
    assert leaf.boundary_count <= MAX_LEAF_VERTS
    assert any("hard cap" in w for w in leaf.warnings)


def test_at_cap_request_has_no_cap_warning():
    leaf = cut_leaf(_diamond(), atlas_size=(200, 400), boundary_verts=MAX_LEAF_VERTS)
    assert leaf.boundary_count <= MAX_LEAF_VERTS
    assert not any("hard cap" in w for w in leaf.warnings)


def test_cap_holds_on_boundary_path_too():
    leaf = cut_leaf(_bilobed(), atlas_size=(200, 300),
                    boundary_verts=10 * MAX_LEAF_VERTS, trace="boundary")
    assert leaf.boundary_count <= MAX_LEAF_VERTS


# --- lobed leaves: boundary tracer preserves the notch ----------------------

def test_auto_falls_back_to_boundary_on_lobed_leaf():
    leaf = cut_leaf(_bilobed(), atlas_size=(200, 300), boundary_verts=64)
    assert any("boundary trace" in w for w in leaf.warnings)


def test_notch_is_outside_the_boundary_but_inside_the_rail():
    a = _bilobed()
    # Notch centre in pixel space is ~(x=100, y=100); the blade's mid-x is 100,
    # so local X == 0, and base='bottom' puts local Y = (H-1 - py)/ppu.
    x_notch = 0.0
    y_notch = (a.shape[0] - 1 - 100) / 2000.0

    auto = cut_leaf(a, atlas_size=(200, 300), boundary_verts=64)  # -> boundary
    rail = cut_leaf(a, atlas_size=(200, 300), boundary_verts=64, trace="rail")

    assert not _point_in_polygon(auto.verts[:, :2], (x_notch, y_notch)), \
        "boundary tracer wrongly filled the notch"
    assert _point_in_polygon(rail.verts[:, :2], (x_notch, y_notch)), \
        "rail unexpectedly did NOT fill the notch (fixture no longer exercises it)"


def test_boundary_tracer_ignores_stray_specks():
    clean = cut_leaf(_bilobed(), atlas_size=(200, 300), trace="boundary")
    speckled = _bilobed()
    speckled[2, 2] = 1.0  # a lone opaque pixel far from the leaf
    got = cut_leaf(speckled, atlas_size=(200, 300), trace="boundary")
    # Largest-component selection means the speck never joins the traced loop,
    # so the blade tip (max +Y) is unchanged.
    assert abs(float(got.verts[:, 1].max()) - float(clean.verts[:, 1].max())) < 1e-9


def test_lobed_leaf_still_obeys_the_contract():
    leaf = cut_leaf(_bilobed(), atlas_size=(200, 300), boundary_verts=64)
    assert np.all(leaf.verts[:, 2] == 0.0)
    assert abs(float(leaf.verts[:, 1].min())) < 1e-9


# --- degenerate inputs ------------------------------------------------------

def test_empty_alpha_raises():
    try:
        cut_leaf(np.zeros((32, 32), np.float64), atlas_size=(32, 32))
    except ValueError:
        return
    raise AssertionError("expected ValueError on an all-transparent alpha")


def test_non_2d_alpha_raises():
    try:
        cut_leaf(np.zeros((8, 8, 4), np.float64), atlas_size=(8, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError on a non-(H,W) alpha")


def test_unknown_trace_mode_raises():
    try:
        cut_leaf(_diamond(), atlas_size=(200, 400), trace="marching")
    except ValueError:
        return
    raise AssertionError("expected ValueError on an unknown trace mode")


# --- CUTOUT: whole-atlas batch (cut every part) -----------------------------

def _atlas_with_diamonds(centres, hx=30, hy=40, h=200, w=300):
    """Alpha with one diamond per (cx, cy) centre, each 2*hx wide x 2*hy tall."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w), np.float64)
    for cx, cy in centres:
        a[(np.abs(xx - cx) / hx + np.abs(yy - cy) / hy) <= 1.0] = 1.0
    return a


def test_atlas_cuts_every_separate_part():
    atlas = _atlas_with_diamonds([(50, 100), (150, 100), (250, 100)])
    leaves = cut_atlas(atlas, atlas_size=(300, 200))
    assert len(leaves) == 3


def test_atlas_skips_specks():
    atlas = _atlas_with_diamonds([(50, 100), (150, 100), (250, 100)])
    atlas[3:5, 3:5] = 1.0  # a 2x2 speck, far below the area floor
    leaves = cut_atlas(atlas, atlas_size=(300, 200))
    assert len(leaves) == 3


def test_atlas_leaves_are_each_contract_and_capped():
    atlas = _atlas_with_diamonds([(50, 100), (150, 100), (250, 100)])
    for leaf in cut_atlas(atlas, atlas_size=(300, 200)):
        assert np.all(leaf.verts[:, 2] == 0.0)
        assert abs(float(leaf.verts[:, 1].min())) < 1e-9  # base at origin
        assert float(leaf.verts[:, 1].max()) > 0.0
        assert 3 <= leaf.boundary_count <= MAX_LEAF_VERTS


def test_atlas_uvs_are_atlas_relative_per_part():
    # Reading order (left->right here) => each part's UV column increases, and
    # each part's mean U tracks its own atlas position, proving UVs are keyed to
    # the source region rather than all mapping to one spot.
    atlas = _atlas_with_diamonds([(50, 100), (150, 100), (250, 100)])
    leaves = cut_atlas(atlas, atlas_size=(300, 200))
    means = [float(leaf.uvs[:, 0].mean()) for leaf in leaves]
    assert means[0] < means[1] < means[2]
    for mean, cx in zip(means, (50, 150, 250), strict=True):
        assert abs(mean - cx / 300.0) < 0.05


def test_atlas_single_part_returns_one():
    atlas = _atlas_with_diamonds([(150, 100)])
    assert len(cut_atlas(atlas, atlas_size=(300, 200))) == 1


def test_atlas_empty_raises():
    try:
        cut_atlas(np.zeros((64, 64), np.float64), atlas_size=(64, 64))
    except ValueError:
        return
    raise AssertionError("expected ValueError on an all-transparent atlas")


# --- GENERATE half: procedural blade ----------------------------------------

def test_generate_obeys_the_contract():
    leaf = generate_leaf(length=0.12, width=0.05, shape="ovate")
    assert np.all(leaf.verts[:, 2] == 0.0), "not flat in local XY"
    assert abs(float(leaf.verts[:, 1].min())) < 1e-12, "base not at the origin"
    assert float(leaf.verts[:, 1].max()) > 0.0, "blade does not run along +Y"
    # midrib-centred: bbox symmetric about X == 0.
    xmin, xmax = float(leaf.verts[:, 0].min()), float(leaf.verts[:, 0].max())
    assert abs(xmin + xmax) < 1e-12, "blade not symmetric about the midrib"


def test_generate_honours_metric_size():
    leaf = generate_leaf(length=0.2, width=0.06, shape="elliptic")
    assert abs(float(leaf.verts[:, 1].max()) - 0.2) < 1e-9, "length not honoured"
    span = float(leaf.verts[:, 0].max() - leaf.verts[:, 0].min())
    assert span <= 0.06 + 1e-9, "blade wider than requested width"
    assert span > 0.5 * 0.06, "blade implausibly narrow for its shape"


def test_generate_shapes_differ_in_widest_position():
    def widest_fraction(shape):
        leaf = generate_leaf(length=0.1, width=0.05, shape=shape, stations=64)
        i = int(np.argmax(np.abs(leaf.verts[:, 0])))
        return float(leaf.verts[i, 1]) / 0.1

    ovate = widest_fraction("ovate")
    obovate = widest_fraction("obovate")
    elliptic = widest_fraction("elliptic")
    assert ovate < 0.5 < obovate, "ovate/obovate not on opposite sides of mid-blade"
    assert abs(elliptic - 0.5) < 0.1, "elliptic not widest near the middle"


def test_generate_width_samples_override_shape():
    # A constant 1.0 profile is a full-width strap: the span reaches the whole
    # width at every station, so max span == width exactly.
    leaf = generate_leaf(length=0.1, width=0.04,
                         width_samples=np.ones(8), stations=32)
    span = float(leaf.verts[:, 0].max() - leaf.verts[:, 0].min())
    assert abs(span - 0.04) < 1e-9


def test_generate_all_named_shapes_build():
    for shape in LEAF_SHAPES:
        leaf = generate_leaf(shape=shape)
        assert leaf.boundary_count >= 3
        assert np.all(leaf.verts[:, 2] == 0.0)


def test_generate_uv_in_unit_square_midrib_centred():
    leaf = generate_leaf(width=0.05, length=0.1)
    assert leaf.uvs.min() >= -1e-9 and leaf.uvs.max() <= 1.0 + 1e-9
    # midrib vertices (x==0) map to U==0.5.
    on_midrib = np.abs(leaf.verts[:, 0]) < 1e-12
    assert np.allclose(leaf.uvs[on_midrib, 0], 0.5)


def test_generate_uvs_are_unstretched():
    # One scale for U and V: on a flat leaf every edge's UV length / real length
    # is the same number, whatever the leaf's proportions (a per-axis fit used to
    # squash acicular 10x), teeth included; the bbox fills the square's long side.
    from ..growth.leaf import LEAF_SHAPES
    cases = [dict(shape=k, width=0.1 * s.aspect) for k, s in LEAF_SHAPES.items()]
    cases.append(dict(shape="ovate", width=0.05, margin="serrate",
                      margin_teeth=10, margin_depth=0.3))
    for kw in cases:
        leaf = generate_leaf(length=0.1, stations=40, **kw)
        ratios = []
        for f in leaf.faces:
            for a, b in zip(f, f[1:] + f[:1], strict=True):
                d3 = np.linalg.norm(leaf.verts[a] - leaf.verts[b])
                if d3 > 1e-12:
                    ratios.append(np.linalg.norm(leaf.uvs[a] - leaf.uvs[b]) / d3)
        ratios = np.array(ratios)
        assert ratios.max() - ratios.min() < 1e-9 * ratios.max(), f"{kw}: UV stretched"
        lo, hi = leaf.uvs.min(axis=0), leaf.uvs.max(axis=0)
        assert lo.min() >= -1e-12 and hi.max() <= 1.0 + 1e-12, f"{kw}: UV outside square"
        assert abs(max(hi - lo) - 1.0) < 1e-9, f"{kw}: long side does not fill the square"


def test_generate_stations_clamped_to_cap():
    leaf = generate_leaf(stations=10 * MAX_LEAF_VERTS)
    assert leaf.boundary_count <= MAX_LEAF_VERTS
    assert any("hard cap" in w for w in leaf.warnings)


def test_generate_unknown_shape_raises():
    try:
        generate_leaf(shape="fractal")
    except ValueError:
        return
    raise AssertionError("expected ValueError on an unknown shape")


def test_generate_nonpositive_size_raises():
    for kwargs in ({"length": 0.0}, {"width": -0.1}):
        try:
            generate_leaf(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_generate_all_zero_profile_is_degenerate():
    try:
        generate_leaf(width_samples=np.zeros(6))
    except ValueError:
        return
    raise AssertionError("expected ValueError on an all-zero width profile")


# --- GENERATE half: single-outline shape (no caps) --------------------------

def test_named_shapes_reproduce_their_baked_profile():
    # Each named shape is an artist-authored half-width profile; the generated
    # outline must follow it exactly. With stations == samples-1 every grid row
    # lands on a baked sample, so the row half-width is profile[k] * width/2.
    from ..growth.leaf import LEAF_SHAPES
    length, width = 0.1, 0.05
    for name, shape in LEAF_SHAPES.items():
        n = len(shape.profile) - 1
        leaf = generate_leaf(length=length, width=width, shape=name, stations=n)
        v = leaf.verts
        for k in range(n + 1):
            row = v[np.abs(v[:, 1] - k / n * length) < 1e-9]
            got = float(np.max(np.abs(row[:, 0])))
            want = shape.profile[k] * 0.5 * width
            assert abs(got - want) < 1e-9, f"{name}: row {k} off its baked profile"


def test_shape_table_is_normalised_with_an_aspect():
    # Profiles are peak-normalised (so `width` is the true widest width) and each
    # carries its authored widest-width/length ratio for the panel to apply.
    from ..growth.leaf import LEAF_SHAPES
    for name, shape in LEAF_SHAPES.items():
        prof = np.asarray(shape.profile)
        assert abs(prof.max() - 1.0) < 1e-9, f"{name}: profile peak is not 1"
        assert prof.min() >= 0.0, f"{name}: negative half-width"
        assert 0.0 < shape.aspect <= 1.5, f"{name}: implausible aspect"
        assert shape.label, f"{name}: missing display label"
    # a round leaf is far stubbier than a needle.
    assert LEAF_SHAPES["orbicular"].aspect > 4 * LEAF_SHAPES["acicular"].aspect


def test_shape_outline_is_a_single_smooth_curve():
    # No cap kink: the half-width profile has no sudden slope reversal in its
    # interior (a cap join used to introduce one). Check the sampled profile is
    # unimodal -- rises then falls, no second bump.
    length, width = 0.1, 0.05
    leaf = generate_leaf(length=length, width=width, shape="ovate", stations=80)
    v = leaf.verts
    ts = np.linspace(0.05, 0.95, 30)
    hw = []
    for t in ts:
        band = v[np.abs(v[:, 1] - t * length) < 0.02 * length]
        hw.append(float(np.max(np.abs(band[:, 0]))) if band.size else 0.0)
    hw = np.array(hw)
    peak = int(np.argmax(hw))
    # rising up to the peak, falling after -- allow tiny sampling noise.
    assert np.all(np.diff(hw[:peak + 1]) > -1e-4), "profile not monotone up to peak"
    assert np.all(np.diff(hw[peak:]) < 1e-4), "profile not monotone after peak"


def test_all_named_shapes_build_and_are_flat_by_default():
    # The 25 artist-authored shapes baked from the leaf preset JSONs.
    assert len(LEAF_SHAPES) == 25
    for shape in LEAF_SHAPES:
        leaf = generate_leaf(shape=shape)
        assert leaf.faces, shape
        assert leaf.boundary_count <= MAX_LEAF_VERTS, shape
        # no deformer set => still flat.
        assert np.allclose(leaf.verts[:, 2], 0.0), shape


# --- GENERATE half: blade deformers -----------------------------------------

def test_no_deformer_leaves_the_blade_flat():
    flat = generate_leaf(shape="elliptic", stations=20)
    assert np.all(flat.verts[:, 2] == 0.0), "default generate is not flat"


def test_curl_bends_the_tip_out_of_plane_and_is_signed():
    up = generate_leaf(shape="elliptic", length=0.12, width=0.05, curl=0.5, stations=24)
    down = generate_leaf(shape="elliptic", length=0.12, width=0.05, curl=-0.5, stations=24)
    tip_up = up.verts[np.argmax(up.verts[:, 1])]
    tip_down = down.verts[np.argmax(down.verts[:, 1])]
    assert tip_up[2] > 1e-3, "curl + did not lift the tip"
    assert tip_down[2] < -1e-3, "curl - did not drop the tip"


def test_cup_lifts_the_edges_relative_to_the_midrib():
    leaf = generate_leaf(shape="elliptic", length=0.1, width=0.06, cup=0.6, stations=24)
    v = leaf.verts
    band = v[np.abs(v[:, 1] - 0.5 * 0.1) < 0.02 * 0.1]  # a mid cross-section
    midrib_z = float(band[np.argmin(np.abs(band[:, 0])), 2])
    edge_z = float(band[np.argmax(np.abs(band[:, 0])), 2])
    assert edge_z - midrib_z > 1e-3, "cup did not raise the edges above the midrib"


def test_gravity_droops_the_tip_downward_only():
    leaf = generate_leaf(shape="elliptic", length=0.12, width=0.05, gravity=0.5, stations=24)
    tip = leaf.verts[np.argmax(leaf.verts[:, 1])]
    base_band = leaf.verts[leaf.verts[:, 1] < 0.05 * 0.12]
    assert tip[2] < -1e-3, "gravity did not droop the tip down"
    assert np.all(base_band[:, 2] > -1e-3), "gravity moved the base (should be tip-weighted)"


def test_twist_rolls_the_tip_cross_section():
    flat = generate_leaf(shape="elliptic", length=0.1, width=0.06, stations=24)
    twisted = generate_leaf(shape="elliptic", length=0.1, width=0.06, twist=0.5, stations=24)
    # An edge vertex near the tip gains |Z| once the section is rolled.
    def tip_edge_z(leaf):
        v = leaf.verts
        near = v[v[:, 1] > 0.8 * 0.1]
        return float(np.max(np.abs(near[:, 2])))
    assert tip_edge_z(flat) < 1e-9
    assert tip_edge_z(twisted) > 1e-3, "twist did not roll the tip section"


def test_wave_ripples_the_blade_both_ways():
    leaf = generate_leaf(shape="linear", length=0.14, width=0.04,
                         wave=0.5, wave_count=3, stations=40)
    z = leaf.verts[:, 2]
    assert z.max() > 1e-3 and z.min() < -1e-3, "wave did not ripple up AND down"


def test_deformers_do_not_change_vertex_count():
    flat = generate_leaf(shape="ovate", stations=24)
    deformed = generate_leaf(shape="ovate", stations=24, curl=0.4, cup=0.3,
                             twist=0.2, gravity=0.3, wave=0.2)
    assert flat.boundary_count == deformed.boundary_count
    assert deformed.boundary_count <= MAX_LEAF_VERTS


def test_deformed_leaf_keeps_base_at_origin():
    leaf = generate_leaf(shape="ovate", curl=0.4, cup=0.3, gravity=0.2, stations=24)
    # The base may be a full row (a blunt base), whose edge verts cup legitimately
    # lifts -- the attachment point is the MIDRIB vertex of that row.
    base_row = leaf.verts[np.abs(leaf.verts[:, 1]) < 1e-9]
    base = base_row[np.argmin(np.abs(base_row[:, 0]))]
    assert np.allclose(base, (0.0, 0.0, 0.0), atol=1e-6), "base drifted off the origin"


# --- GENERATE half: edge mask (wave confined to the margins) ----------------

def test_mask_is_zero_at_midrib_and_one_at_edge():
    leaf = generate_leaf(shape="elliptic", length=0.1, width=0.06, stations=24)
    assert leaf.mask is not None, "generate must author the edge mask"
    v = leaf.verts
    midrib = leaf.mask[np.abs(v[:, 0]) < 1e-9]
    assert np.allclose(midrib, 0.0), "mask nonzero on the midrib"
    assert float(leaf.mask.max()) > 0.99, "mask never reaches ~1 at the edge"


def test_wave_leaves_the_midrib_flat():
    # The whole point of the mask: a waved leaf keeps a flat, rigid midrib while
    # the edges ripple.
    leaf = generate_leaf(shape="linear", length=0.14, width=0.05,
                         wave=0.6, wave_count=3, stations=40)
    v = leaf.verts
    midrib_z = np.abs(v[np.abs(v[:, 0]) < 1e-9][:, 2])
    edge_z = np.abs(v[:, 2])
    assert float(midrib_z.max()) < 1e-6, "midrib is not flat under wave"
    assert float(edge_z.max()) > 1e-3, "edges did not ripple"


# --- GENERATE half: margin (edge profile) -----------------------------------

def _edge_halfwidths(leaf, length, n=12):
    """Max |x| in each of n bands up the blade -- the boundary silhouette."""
    v = leaf.verts
    out = []
    for k in range(n):
        lo, hi = k / n * length, (k + 1) / n * length
        band = v[(v[:, 1] >= lo) & (v[:, 1] < hi)]
        out.append(float(np.max(np.abs(band[:, 0]))) if band.size else 0.0)
    return np.array(out)


def test_entire_margin_is_the_default_and_smooth():
    leaf = generate_leaf(shape="elliptic", length=0.1, width=0.05, stations=32)
    # An entire elliptic edge is monotone up to the middle then down: no teeth.
    hw = _edge_halfwidths(leaf, 0.1, n=10)
    peak = int(np.argmax(hw))
    assert np.all(np.diff(hw[:peak + 1]) > -1e-4)
    assert np.all(np.diff(hw[peak:]) < 1e-4)


def test_serrate_margin_adds_teeth_to_the_edge():
    entire = generate_leaf(shape="elliptic", length=0.12, width=0.05,
                           margin="entire", stations=48)
    serr = generate_leaf(shape="elliptic", length=0.12, width=0.05,
                         margin="serrate", margin_teeth=10, margin_depth=0.25,
                         stations=48)
    # Teeth push some boundary verts beyond the smooth outline's local width.
    ent_hw = _edge_halfwidths(entire, 0.12, n=24)
    ser_hw = _edge_halfwidths(serr, 0.12, n=24)
    assert float(ser_hw.max()) > float(ent_hw.max()) + 1e-4, "serrate added no teeth"
    # And the toothed edge oscillates (many local reversals) where entire is smooth.
    def reversals(a):
        d = np.diff(a)
        return int(np.sum(np.diff(np.sign(d)) != 0))
    assert reversals(ser_hw) > reversals(ent_hw), "serrate edge not oscillating"


def test_all_margins_build_and_stay_under_cap():
    from ..growth.leaf import LEAF_MARGINS
    for m in LEAF_MARGINS:
        leaf = generate_leaf(shape="ovate", margin=m, margin_teeth=8,
                             margin_depth=0.2)
        assert leaf.faces, m
        assert leaf.boundary_count <= MAX_LEAF_VERTS, m


def test_margins_produce_distinct_silhouettes():
    # The complaint was "all margins look the same". Each named margin must give a
    # measurably different boundary silhouette from the others.
    from ..growth.leaf import LEAF_MARGINS
    sigs = {}
    for m in LEAF_MARGINS:
        if m == "entire":
            continue
        leaf = generate_leaf(shape="elliptic", length=0.12, width=0.06,
                             margin=m, margin_teeth=9, margin_depth=0.3, stations=48)
        sigs[m] = np.round(_edge_halfwidths(leaf, 0.12, n=24), 5)
    keys = list(sigs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.array_equal(sigs[keys[i]], sigs[keys[j]]), \
                f"{keys[i]} and {keys[j]} have identical silhouettes"


def test_margin_depth_is_clamped_to_prevent_clipping():
    from ..growth.leaf import MARGIN_DEPTH_MAX
    # A huge requested depth is clamped: the result matches the clamp value, and
    # UVs never leave the unit square (no clipping).
    huge = generate_leaf(shape="elliptic", length=0.1, width=0.06,
                         margin="serrate", margin_teeth=8, margin_depth=5.0,
                         stations=40)
    clamped = generate_leaf(shape="elliptic", length=0.1, width=0.06,
                            margin="serrate", margin_teeth=8,
                            margin_depth=MARGIN_DEPTH_MAX, stations=40)
    assert np.array_equal(huge.verts, clamped.verts), "depth not clamped"
    assert huge.uvs.min() >= -1e-9 and huge.uvs.max() <= 1.0 + 1e-9, "UV clipped"


def test_unknown_margin_raises():
    try:
        generate_leaf(margin="frilly")
    except ValueError:
        return
    raise AssertionError("expected ValueError on an unknown margin")


def test_teeth_wrap_the_tip():
    # Teeth must reach the apex, not vanish as the blade tapers. In the top region
    # the toothed outline oscillates (teeth), while an entire outline is smooth.
    L, W = 0.12, 0.06
    ent = generate_leaf(shape="ovate", length=L, width=W, margin="entire", stations=60)
    tooth = generate_leaf(shape="ovate", length=L, width=W, margin="dentate",
                          margin_teeth=12, margin_depth=0.35, stations=60)

    def tip_reversals(leaf, lo=0.8, hi=0.99, n=16):
        v = leaf.verts
        hw = []
        for k in range(n):
            a, b = lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n
            band = v[(v[:, 1] >= a * L) & (v[:, 1] < b * L)]
            hw.append(float(np.max(np.abs(band[:, 0]))) if band.size else 0.0)
        d = np.diff(np.array(hw))
        return int(np.sum(np.diff(np.sign(d)) != 0))

    assert tip_reversals(tooth) > tip_reversals(ent), \
        "no teeth near the apex (tip outline is smooth)"


def test_serrulate_is_finer_than_serrate():
    L, W = 0.12, 0.05
    # Sample well above the tooth frequency: serrulate's 16 teeth/side need >32
    # bands to be counted at all (40 bands aliased it below serrate).
    def edge_reversals(margin):
        leaf = generate_leaf(shape="elliptic", length=L, width=W, margin=margin,
                             margin_teeth=8, margin_depth=0.3, stations=128)
        hw = _edge_halfwidths(leaf, L, n=80)
        d = np.diff(hw)
        return int(np.sum(np.diff(np.sign(d)) != 0))
    assert edge_reversals("serrulate") > edge_reversals("serrate"), \
        "serrulate is not finer than serrate"


def _row_count(leaf):
    return len(np.unique(np.round(leaf.verts[:, 1], 6)))


def _col_count(leaf):
    return len(np.unique(np.round(leaf.verts[:, 0], 6)))


def test_resolution_v_and_h_are_independent():
    base = generate_leaf(shape="elliptic", stations=20, res_h=4)
    more_v = generate_leaf(shape="elliptic", stations=40, res_h=4)
    more_h = generate_leaf(shape="elliptic", stations=20, res_h=10)
    assert _row_count(more_v) > _row_count(base), "vertical res did not add rows"
    assert _col_count(more_h) > _col_count(base), "horizontal res did not add columns"
    assert _row_count(more_h) == _row_count(base), "horizontal res changed the rows"


def test_resolution_v_controls_rows_even_with_a_margin():
    # Regression: a toothed margin must NOT override the resolution slider.
    lo = generate_leaf(shape="ovate", margin="dentate", margin_teeth=8,
                       stations=20, res_h=6)
    hi = generate_leaf(shape="ovate", margin="dentate", margin_teeth=8,
                       stations=60, res_h=6)
    assert _row_count(hi) > _row_count(lo), "resolution slider dead with a margin"


def test_margin_band_gives_interior_clearance():
    from ..growth.leaf import MARGIN_CLEARANCE, _margin_band
    # Signed (inward) margins reserve a deeper band than outward-only ones.
    assert _margin_band("dentate", 0.3) == MARGIN_CLEARANCE
    assert _margin_band("sinuate", 0.3) > _margin_band("dentate", 0.3)
    # The interior mesh of a signed margin sits well inside the plain outline, so
    # the teeth never overlap the inner geometry.
    L, W = 0.13, 0.06
    ent = generate_leaf(shape="elliptic", length=L, width=W, margin="entire",
                        stations=40, res_h=8)
    sig = generate_leaf(shape="elliptic", length=L, width=W, margin="sinuate",
                        margin_teeth=6, margin_depth=0.3, stations=40, res_h=8)

    def interior_max(leaf):            # |x| of the outermost INTERIOR column at one row
        v = leaf.verts
        ys = np.unique(np.round(v[:, 1], 6))
        yc = ys[np.argmin(np.abs(ys - 0.5 * L))]
        row = v[np.abs(v[:, 1] - yc) < 1e-6]
        ax = np.sort(np.abs(row[:, 0]))
        return ax[-3] if ax.size >= 3 else ax[0]   # drop the 2 boundary (edge) verts

    assert interior_max(sig) < interior_max(ent) - 1e-4, \
        "no margin band -- interior not pulled in from the toothed edge"


# --- GENERATE half: custom width-profile presets ----------------------------

def test_shape_profile_samples_follows_the_shape():
    from ..growth.leaf import shape_profile_samples
    s = shape_profile_samples("elliptic", count=9)
    assert s.shape == (9,)
    assert s[0] < 0.2 and s[-1] < 0.2 and s[4] > 0.9   # ellipse: 0 ends, 1 middle


def test_shape_from_preset_matches_the_bake_rule():
    # A custom shape is built exactly like the built-ins: profile peak-normalised,
    # aspect = width * peak / length.
    from ..growth.leaf import shape_from_preset
    data = {"params": {"gen_length": 0.1, "gen_width": 0.06},
            "profile_samples": [0.0, 0.25, 0.5, 0.25, 0.0]}
    shape = shape_from_preset(data, "Mine")
    assert shape.label == "Mine"
    assert np.allclose(shape.profile, (0.0, 0.5, 1.0, 0.5, 0.0))
    assert abs(shape.aspect - 0.06 * 0.5 / 0.1) < 1e-12
    for bad in ({}, {"profile_samples": [0.0, 0.0]}, "nope"):
        try:
            shape_from_preset(bad, "bad")
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_margin_applies_to_a_drawn_profile():
    # Every leaf is profile-driven now, so a margin must cut a width_samples
    # outline too (it used to be forced to entire).
    prof = np.sin(np.linspace(0.0, np.pi, 33))
    ent = generate_leaf(width_samples=prof, margin="entire", stations=60)
    tooth = generate_leaf(width_samples=prof, margin="dentate", margin_teeth=8,
                          margin_depth=0.3, stations=60)
    assert not np.allclose(ent.verts, tooth.verts), "margin ignored on a drawn profile"


def test_shape_profile_samples_match_the_baked_table():
    # Seeding the Custom Profile at the baked sample count must give the baked
    # samples back verbatim (Load Shape reproduces the shape exactly).
    from ..growth.leaf import LEAF_SHAPES, shape_profile_samples
    for name, shape in LEAF_SHAPES.items():
        s = shape_profile_samples(name, count=len(shape.profile))
        assert np.allclose(s, shape.profile), f"{name}: resample drifted"


# --- GENERATE half: per-parameter variance (randomize) ----------------------

def test_randomize_zero_variance_is_identity():
    base = {k: 1.0 for k in VARIABLE_PARAMS}
    assert randomize(base, {}, seed=5) == base
    assert randomize(base, {k: 0.0 for k in VARIABLE_PARAMS}, seed=5) == base


def test_randomize_is_deterministic_and_bounded():
    base = {k: 2.0 for k in VARIABLE_PARAMS}
    var = {k: 0.5 for k in VARIABLE_PARAMS}
    a = randomize(base, var, seed=7)
    b = randomize(base, var, seed=7)
    c = randomize(base, var, seed=8)
    assert a == b, "same seed must be byte-identical"
    assert a != c, "different seed must give a different roll"
    for k in VARIABLE_PARAMS:                       # within base*(1 +- var)
        assert 2.0 * 0.5 - 1e-9 <= a[k] <= 2.0 * 1.5 + 1e-9


def test_randomize_one_param_does_not_shift_another():
    # Variance on a later param must not change an earlier param's roll (fixed
    # draw order), so enabling one control never disturbs the others.
    base = {k: 1.0 for k in VARIABLE_PARAMS}
    only_first = randomize(base, {VARIABLE_PARAMS[0]: 0.5}, seed=3)
    plus_last = randomize(base, {VARIABLE_PARAMS[0]: 0.5, VARIABLE_PARAMS[-1]: 0.5}, seed=3)
    assert only_first[VARIABLE_PARAMS[0]] == plus_last[VARIABLE_PARAMS[0]]


def test_randomize_passes_through_unlisted_keys():
    out = randomize({"shape": "ovate", "length": 1.0}, {"length": 0.5}, seed=1)
    assert out["shape"] == "ovate"


# --- GENERATE half: noise deform + global seed ------------------------------

def test_noise_zero_is_a_flat_no_op():
    flat = generate_leaf(shape="elliptic", stations=24, noise=0.0)
    assert np.all(flat.verts[:, 2] == 0.0), "noise=0 is not flat"


def test_noise_displaces_the_blade():
    leaf = generate_leaf(shape="elliptic", length=0.1, width=0.06,
                         noise=0.5, noise_scale=5, seed=3, stations=32)
    assert float(np.max(np.abs(leaf.verts[:, 2]))) > 1e-3, "noise did not displace"


def test_seed_is_deterministic_and_changes_the_result():
    a1 = generate_leaf(shape="elliptic", noise=0.5, noise_scale=5, seed=7, stations=24)
    a2 = generate_leaf(shape="elliptic", noise=0.5, noise_scale=5, seed=7, stations=24)
    b = generate_leaf(shape="elliptic", noise=0.5, noise_scale=5, seed=8, stations=24)
    assert np.array_equal(a1.verts, a2.verts), "same seed not byte-identical"
    assert not np.array_equal(a1.verts, b.verts), "different seed gave same leaf"


# --- GENERATE: crumple (the whole blade scrunched) ---------------------------

def _midrib(leaf, flat):
    """The deformed midrib polyline: the flat blade's x == 0 column, base -> tip."""
    idx = np.nonzero(np.abs(flat.verts[:, 0]) < 1e-12)[0]
    idx = idx[np.argsort(flat.verts[idx, 1])]
    return leaf.verts[idx]


def test_crumple_zero_is_byte_identical():
    kw = dict(shape="linear", length=0.18, width=0.004, stations=24, curl=-0.3,
              gravity=0.2, twist=0.15, noise=0.05, seed=4)
    a = generate_leaf(**kw)
    b = generate_leaf(**kw, crumple=0.0, crumple_scale=9.0)
    assert np.array_equal(a.verts, b.verts), "crumple 0 moved the blade"


def test_crumple_bends_the_whole_blade_keeping_its_base_and_length():
    kw = dict(shape="linear", length=0.18, width=0.004, stations=48, curl=-0.2,
              gravity=0.15, seed=4)
    flat = generate_leaf(shape="linear", length=0.18, width=0.004, stations=48)
    plain = generate_leaf(**kw)
    crumpled = generate_leaf(**kw, crumple=0.3, crumple_scale=6)
    a, b = _midrib(plain, flat), _midrib(crumpled, flat)
    assert np.allclose(a[0], b[0]), "the base moved"
    la = np.linalg.norm(np.diff(a, axis=0), axis=1).sum()
    lb = np.linalg.norm(np.diff(b, axis=0), axis=1).sum()
    # Within 0.5%: a bend falling between two mesh rows shortens that one chord.
    assert abs(la - lb) < 0.005 * la, f"midrib length changed {la} -> {lb}"
    # A whole-blade bend: the tip moves by centimetres, not the noise's fraction
    # of the half-width -- and sideways too (x), which no other deformer does.
    assert np.linalg.norm(a[-1] - b[-1]) > 0.02, "crumple barely moved the tip"
    assert abs(b[-1][0]) > 1e-3, "crumple did not bend the blade sideways"
    # Each cross-section rides along rigidly: the width at the base stays the width.
    w_plain = np.ptp(plain.verts[:, 0][np.abs(flat.verts[:, 1]) < 1e-12])
    w_crum = np.linalg.norm(np.ptp(crumpled.verts[np.abs(flat.verts[:, 1]) < 1e-12], axis=0))
    assert abs(w_plain - w_crum) < 1e-9, "the base row was distorted"


def test_crumple_is_seeded():
    kw = dict(shape="linear", length=0.18, width=0.004, stations=24, crumple=0.3)
    a1, a2 = generate_leaf(**kw, seed=2), generate_leaf(**kw, seed=2)
    b = generate_leaf(**kw, seed=3)
    assert np.array_equal(a1.verts, a2.verts), "same seed not byte-identical"
    assert not np.array_equal(a1.verts, b.verts), "different seed gave same crumple"


def test_crumple_reaches_a_cutout():
    from ..growth.leaf import cutout_grid
    outline = np.array([[0.0, 0.0], [0.02, 0.05], [0.0, 0.12], [-0.02, 0.05]])
    uv = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    a = cutout_grid(outline, uv, density=12)
    b = cutout_grid(outline, uv, density=12, crumple=0.4, seed=1)
    assert a.verts.shape == b.verts.shape
    assert float(np.max(np.abs(b.verts - a.verts))) > 0.01, "cutout did not crumple"


# --- GENERATE: venation-following quad mesh ---------------------------------

def test_generate_emits_a_quad_dominant_mesh():
    leaf = generate_leaf(shape="elliptic", length=0.12, width=0.06, stations=20)
    assert leaf.faces, "generate must emit faces (a mesh, not a bare loop)"
    quads = [f for f in leaf.faces if len(f) == 4]
    assert len(quads) >= len(leaf.faces) // 2, "mesh should be quad-dominant"


def test_generate_grid_has_midrib_and_interior_columns():
    leaf = generate_leaf(shape="elliptic", length=0.12, width=0.06, stations=20)
    v = leaf.verts
    # A structured grid, not a boundary loop: several distinct X columns.
    assert np.unique(np.round(v[:, 0], 6)).size > 3
    # Midrib column at X==0 spanning base -> tip.
    mid = v[np.abs(v[:, 0]) < 1e-9]
    assert mid.shape[0] >= 3
    assert mid[:, 1].min() < 1e-9 and mid[:, 1].max() > 0.11


def test_generate_face_normals_point_up():
    # CCW winding => face normal +Z, the leaf contract's front face.
    leaf = generate_leaf(shape="ovate", stations=16)
    v = leaf.verts
    for f in leaf.faces:
        p = v[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if abs(n[2]) > 1e-15:
            assert n[2] > 0, "a face is wound the wrong way (normal not +Z)"


def test_venation_derived_from_shape():
    from ..growth.leaf import venation_for
    for shape in ("elliptic", "ovate", "obovate", "lanceolate", "rhomboid",
                  "spatulate", "pinnatisect"):
        assert venation_for(shape) == "pinnate"
    for shape in ("linear", "acicular", "subulate"):
        assert venation_for(shape) == "parallel"
    assert venation_for("flabellate") == "dichotomous"
    for shape in ("orbicular", "cordate", "reniform", "hastate", "digitate"):
        assert venation_for(shape) == "palmate"
    assert venation_for("unknown-shape") == "pinnate"  # sensible default


# --- CUTOUT -> deformable triangle mesh ---------------------------------------

def _skewed_leaf(h: int = 400, w: int = 300) -> np.ndarray:
    """An asymmetric single-lobe leaf: an ellipse whose centre line leans right
    toward the tip (a falcate-ish cutout -- not symmetric about any x)."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = 1.0 - yy / h                                   # 0 base (bottom) -> 1 tip
    centre = w * 0.35 + w * 0.3 * t ** 2
    half = w * 0.22 * np.sqrt(np.clip(t * (1.0 - t), 0.0, None)) * 2.0
    return (np.abs(xx - centre) <= half).astype(np.float64)


def _notched_leaf(h: int = 400, w: int = 300) -> np.ndarray:
    """A cordate-ish leaf: an ellipse with a V notch cut into the base."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = (((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h / 2) / (h * 0.48)) ** 2 <= 1.0)
    notch = (yy > h * 0.7) & (np.abs(xx - w / 2) < (yy - h * 0.7) * 0.8)
    return (a & ~notch).astype(np.float64)


def _forked_leaf(h: int = 300, w: int = 300) -> np.ndarray:
    """Two lobes with a WIDE sinus between them over a common base (a deep maple-
    like cut): spanning it adds a large area to the quad mesh."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w), np.float64)
    a[(yy >= 200) & (xx >= 20) & (xx < 280)] = 1.0                  # base
    a[(yy < 200) & (((xx >= 20) & (xx < 80)) | ((xx >= 220) & (xx < 280)))] = 1.0
    return a


def _uneven_fork(h: int = 300, w: int = 300) -> np.ndarray:
    """Two lobes of DIFFERENT lengths over a common base: the short lobe's piece
    ends mid-height (an `end` below the leaf's top)."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w), np.float64)
    a[(yy >= 200) & (xx >= 20) & (xx < 280)] = 1.0
    a[(yy < 200) & (xx >= 220) & (xx < 280)] = 1.0                   # long lobe
    a[(yy >= 90) & (yy < 200) & (xx >= 20) & (xx < 80)] = 1.0        # short lobe
    return a


def _cut_grid(alpha, **kw):
    from ..growth.leaf import cutout_grid, uv_affine
    h, w = alpha.shape
    # Like the bridge in mesh mode: the RAW trace (no two-sided boundary budget,
    # which can step outward over alpha); cutout_grid simplifies one-sided.
    leaf = cut_leaf(alpha, atlas_size=(w, h), boundary_verts=MAX_LEAF_VERTS)
    kw.setdefault("pixel", 1.0 / 2000.0)               # cut_leaf's default ppu
    return leaf, cutout_grid(leaf.verts, uv_affine(leaf.verts, leaf.uvs), **kw)


def test_cutout_mesh_is_even_triangles_with_exact_atlas_uvs():
    from ..growth.leaf import uv_affine
    leaf, grid = _cut_grid(_skewed_leaf(), density=20)
    assert all(len(f) == 3 for f in grid.faces), "not a triangle mesh"
    assert np.allclose(grid.verts[:, 2], 0.0), "flat by default"
    assert grid.verts[:, 1].min() >= -1e-12, "base below the origin"
    # UVs: exactly the cutout's own affine atlas map, applied at each vertex.
    a = uv_affine(leaf.verts, leaf.uvs)
    resid = np.column_stack([leaf.verts[:, :2], np.ones(len(leaf.verts))]) @ a - leaf.uvs
    assert np.abs(resid).max() < 1e-9, "cutout UV is not affine in XY"
    want = np.column_stack([grid.verts[:, :2], np.ones(len(grid.verts))]) @ a
    assert np.abs(grid.uvs - want).max() < 1e-12
    # Even and near-equilateral: inner edges sit near the target length and
    # most triangles are well shaped.
    angles = _min_angles(grid)
    assert np.mean(angles > 30.0) > 0.85, f"only {np.mean(angles > 30):.0%} well shaped"
    v = grid.verts[:, :2]
    target = max(np.ptp(v[:, 1]), np.ptp(v[:, 0])) / 20
    inner = [np.hypot(*(v[i] - v[j])) for (i, j), n in _edge_counts(grid).items() if n == 2]
    assert 0.75 < np.median(inner) / target < 1.25, "edges not at the target length"


def _min_angles(grid):
    v = grid.verts[:, :2]
    out = []
    for a, b, c in grid.faces:
        pts = v[[a, b, c]]
        ang = []
        for k in range(3):
            d1, d2 = pts[(k + 1) % 3] - pts[k], pts[(k + 2) % 3] - pts[k]
            ang.append(np.degrees(np.arccos(np.clip(
                d1 @ d2 / (np.hypot(*d1) * np.hypot(*d2)), -1.0, 1.0))))
        out.append(min(ang))
    return np.asarray(out)


def _edge_counts(grid):
    from collections import Counter
    edges = Counter()
    for f in grid.faces:
        for a, b in zip(f, f[1:] + f[:1], strict=True):
            edges[tuple(sorted((a, b)))] += 1
    return edges


def test_cutout_mesh_density_and_cap():
    _leaf, lo = _cut_grid(_diamond(), density=10)
    _leaf, hi = _cut_grid(_diamond(), density=20)
    # Twice the density: about four times the inner vertices (an area fill).
    assert 2.5 < len(hi.verts) / len(lo.verts) < 4.5
    _leaf, huge = _cut_grid(_diamond(), density=5000)
    assert huge.boundary_count <= MAX_LEAF_VERTS
    assert any("hard cap" in w for w in huge.warnings)


def _petiole_leaf(h: int = 500, w: int = 300) -> np.ndarray:
    """Like the user's atlas leaves: a blade with a cordate notch at its base and
    a thin petiole running down out of the notch."""
    yy, xx = np.mgrid[0:h, 0:w]
    blade = ((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h * 0.4) / (h * 0.36)) ** 2 <= 1.0
    notch = (yy > h * 0.6) & (np.abs(xx - w / 2) < (yy - h * 0.6) * 1.2)
    stalk = (yy >= h * 0.55) & (yy < h * 0.98) & (np.abs(xx - w / 2) <= 4)
    return ((blade & ~notch) | stalk).astype(np.float64)


def _coverage(leaf, grid, n=200):
    """(alpha covered, leaf missed) as fractions of the outline's area, by
    rasterising the traced outline and the flat mesh on an n x n grid."""
    poly = leaf.verts[:, :2]
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    # Cell CENTRES: a sample exactly on a straight outline edge would count as
    # outside the outline but inside the mesh (a phantom 'alpha' strip).
    cx = lo[0] + (np.arange(n) + 0.5) * (hi[0] - lo[0]) / n
    cy = lo[1] + (np.arange(n) + 0.5) * (hi[1] - lo[1]) / n
    gx, gy = np.meshgrid(cx, cy)
    px, py = gx.ravel(), gy.ravel()
    inside = np.zeros(px.shape, bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if y1 == y2:
            continue
        hit = (y1 > py) != (y2 > py)
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= hit & (px < xint)
    v = grid.verts[:, :2]
    in_mesh = np.zeros(px.shape, bool)
    for a, b, c in grid.faces:
        (ax, ay), (bx, by), (cx, cy) = v[a], v[b], v[c]
        d1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        d2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
        d3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
        in_mesh |= (d1 >= 0) & (d2 >= 0) & (d3 >= 0)
    area = inside.sum()
    return (in_mesh & ~inside).sum() / area, (inside & ~in_mesh).sum() / area


def test_cutout_mesh_covers_no_alpha():
    # The mesh follows the outline: a notch, lobes or a stalk beside the blade
    # stay open (no alpha covered) and the leaf is covered to within the
    # one-sided smoothing -- for plain, skewed, notched, lobed and petioled leaves.
    for name, alpha in (("diamond", _diamond()), ("skewed", _skewed_leaf()),
                        ("notched", _notched_leaf()), ("forked", _forked_leaf()),
                        ("bilobed", _bilobed()), ("petiole", _petiole_leaf()),
                        ("uneven", _uneven_fork())):
        leaf, grid = _cut_grid(alpha, density=20)
        extra, missed = _coverage(leaf, grid)
        assert not grid.warnings, f"{name}: {grid.warnings}"
        assert extra < 0.005, f"{name}: mesh covers {extra:.2%} alpha"
        assert missed < 0.03, f"{name}: mesh misses {missed:.1%} of the leaf"
        for f in grid.faces:                            # every face CCW (normal +Z)
            pts = grid.verts[list(f), :2]
            x, y = pts[:, 0], pts[:, 1]
            assert np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) > 0, name


def _bilinear(alpha, u, v):
    """Alpha at UV (u, v) with bilinear filtering (pixel centres at (i+0.5)/size),
    as a render's linear texture filter reads it. Row 0 of `alpha` is the TOP."""
    h, w = alpha.shape
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    def px(c, r):
        return alpha[min(h - 1, max(0, r)), min(w - 1, max(0, c))]
    top = px(x0, y0) * (1 - fx) + px(x0 + 1, y0) * fx
    bot = px(x0, y0 + 1) * (1 - fx) + px(x0 + 1, y0 + 1) * fx
    return top * (1 - fy) + bot * fy


def test_cutout_mesh_shows_no_alpha_through_its_uvs():
    # What an artist sees: the atlas ALPHA inside every face, read through the
    # face's UVs with linear filtering. Guards the geometry and the UV alignment
    # together (a half-texel UV offset alone used to expose a strip of alpha
    # along two sides). Samples are strictly inside each face. The mesh edge sits
    # ON the traced 50 % alpha contour, so samples right at the edge read just
    # under 0.5 (the sub-pixel anti-alias ramp); visible alpha = clearly
    # transparent texels (< 0.3) under the mesh.
    bary = [(a / 6, b / 6) for a in range(1, 6) for b in range(1, 6 - a)]
    for name, alpha in (("skewed", _skewed_leaf()), ("petiole", _petiole_leaf()),
                        ("notched", _notched_leaf())):
        _leaf, grid = _cut_grid(alpha, density=20)
        uv = grid.uvs
        hits = total = 0
        for a, b, c in grid.faces:
            for s_, t_ in bary:
                u, v = uv[a] + s_ * (uv[b] - uv[a]) + t_ * (uv[c] - uv[a])
                total += 1
                hits += _bilinear(alpha, u, v) < 0.3
        assert hits / total < 0.001, f"{name}: {hits / total:.2%} of samples show alpha"


def test_cutout_mesh_is_watertight_with_one_outline_loop():
    # No T-junctions (a vertex lying on another face's edge opens a crack once
    # the leaf is curled): every edge is shared by at most two faces, no vertex
    # sits inside any edge, and the open edges form ONE loop -- the silhouette.
    for name, alpha in (("skewed", _skewed_leaf()), ("notched", _notched_leaf()),
                        ("petiole", _petiole_leaf()), ("forked", _forked_leaf()),
                        ("uneven", _uneven_fork())):
        _leaf, grid = _cut_grid(alpha, density=20)
        v = grid.verts[:, :2]
        edges = _edge_counts(grid)
        assert max(edges.values()) <= 2, f"{name}: an edge shared by 3+ faces"
        scale = np.ptp(v, axis=0).max()
        for a, b in edges:
            p, q = v[a], v[b]
            d = q - p
            L2 = float(d @ d)
            t = ((v - p) @ d) / L2
            off = np.abs(d[0] * (v[:, 1] - p[1]) - d[1] * (v[:, 0] - p[0])) / np.sqrt(L2)
            on = (t > 1e-6) & (t < 1 - 1e-6) & (off < 1e-9 * scale)
            on[[a, b]] = False
            assert not on.any(), f"{name}: T-junction on edge {a}-{b}"
        open_edges = [e for e, n in edges.items() if n == 1]
        nbrs = {}
        for a, b in open_edges:
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)
        assert all(len(n) == 2 for n in nbrs.values()), f"{name}: open edges branch"
        start = open_edges[0][0]
        prev, cur, steps = None, start, 0
        while True:
            nxt = next(x for x in nbrs[cur] if x != prev)
            prev, cur, steps = cur, nxt, steps + 1
            if cur == start:
                break
        assert steps == len(open_edges), f"{name}: the outline is not one loop"


def test_cutout_mesh_stalk_stays_light_and_square():
    # The petiole is a strip between its two sides -- no inner vertices -- and
    # its free end keeps the traced square end (not run to a point).
    _leaf, grid = _cut_grid(_petiole_leaf(), density=20)
    v = grid.verts[:, :2]
    on_outline = {i for e, n in _edge_counts(grid).items() if n == 1 for i in e}
    px = 1.0 / 2000.0
    stalk_top = 0.9 * (0.98 - 0.76) * 500 * px          # below the blade's lowest point
    stalk = np.nonzero(v[:, 1] < stalk_top)[0]
    assert len(stalk) >= 4
    assert all(int(i) in on_outline for i in stalk), "inner vertices in the stalk"
    end = v[v[:, 1] < v[:, 1].min() + 1.0 * px]
    assert np.ptp(end[:, 0]) > 6 * px, "stalk end is pointed, not square"


def test_cutout_mesh_deforms_like_a_generated_blade():
    _leaf, flat = _cut_grid(_skewed_leaf(), density=20)
    _leaf, curled = _cut_grid(_skewed_leaf(), density=20, curl=0.4)
    assert flat.faces == curled.faces, "deform changed topology"
    tip = np.argmax(flat.verts[:, 1])
    assert curled.verts[tip, 2] > 1e-3, "curl did not lift the tip"
    # Cup works about the leaf's own centre line (the skewed leaf's is not x = 0):
    # the edges rise, the vertex nearest the centre line mid-length stays low.
    _leaf, cupped = _cut_grid(_skewed_leaf(), density=20, cup=0.5)
    z = cupped.verts[:, 2]
    assert z.min() > -1e-12 and z.max() > 1e-3
    v = flat.verts
    y_mid = 0.5 * (v[:, 1].min() + v[:, 1].max())
    band = np.nonzero(np.abs(v[:, 1] - y_mid) < 0.05 * np.ptp(v[:, 1]))[0]
    lo_x, hi_x = v[band, 0].min(), v[band, 0].max()
    centre = band[np.argmin(np.abs(v[band, 0] - 0.5 * (lo_x + hi_x)))]
    assert z[centre] < 0.05 * z[band].max(), "cup not about the centre line"


def test_cutout_mesh_bends_without_steps_over_lobes():
    # The centre line follows the outer extents, smoothed along the length: a
    # lobe starting part-way up must not put a step into a cup or twist (an
    # edge whose rise is far steeper than the cup's own slope).
    for name, alpha in (("forked", _forked_leaf()), ("uneven", _uneven_fork())):
        _leaf, flat = _cut_grid(alpha, density=20)
        for kw in ({"cup": 0.5}, {"twist": 0.3, "cup": 0.3}):
            _leaf, bent = _cut_grid(alpha, density=20, **kw)
            worst = 0.0
            for a, b in _edge_counts(flat):
                run = np.hypot(*(flat.verts[a, :2] - flat.verts[b, :2]))
                worst = max(worst, abs(bent.verts[a, 2] - bent.verts[b, 2]) / run)
            assert worst < 1.5, f"{name} {kw}: an edge rises {worst:.2f} per unit"


# --- COMPOUND leaves ----------------------------------------------------------

def _islands(leaf):
    """Vertex counts of the mesh's connected pieces (union-find over faces)."""
    from collections import Counter
    par = list(range(len(leaf.verts)))

    def root(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for f in leaf.faces:
        for v in f[1:]:
            par[root(v)] = root(f[0])
    return Counter(root(i) for i in range(len(leaf.verts)))


def _leaflet_count(leaf, per_leaflet):
    """Pieces with exactly a leaflet's vertex count (strips have fewer / other)."""
    return sum(1 for n in _islands(leaf).values() if n == per_leaflet)


def test_compound_simple_is_the_plain_blade():
    from ..growth.leaf import generate_compound_leaf
    a = generate_leaf(shape="ovate", curl=0.2, stations=20)
    b = generate_compound_leaf(compound="simple", shape="ovate", curl=0.2, stations=20)
    assert np.array_equal(a.verts, b.verts) and a.faces == b.faces


def test_compound_types_have_their_leaflet_counts():
    from ..growth.leaf import generate_compound_leaf
    blade = dict(shape="elliptic", length=0.04, width=0.015, stations=8, res_h=2)
    one = len(generate_leaf(**blade).verts)
    cases = (
        (dict(compound="trifoliate"), 3),
        (dict(compound="trifoliate", rachis_length=0.0), 3),
        (dict(compound="palmate", leaflets=7), 7),
        (dict(compound="pinnate", leaflets=4), 9),                 # odd-pinnate
        (dict(compound="pinnate", leaflets=4, terminal=False), 8),  # even-pinnate
        (dict(compound="bipinnate", leaflets=2, sub_pairs=3), 5 * 7),
        (dict(compound="bipinnate", leaflets=2, sub_pairs=3, terminal=False), 4 * 6),
    )
    for kw, want in cases:
        leaf = generate_compound_leaf(**kw, **blade)
        assert _leaflet_count(leaf, one) == want, (kw, _leaflet_count(leaf, one))
        assert np.allclose(leaf.verts[:, 2], 0.0), f"{kw}: not flat by default"
        assert len(leaf.uvs) == len(leaf.verts) == len(leaf.mask)
        for f in leaf.faces:                            # every face CCW (normal +Z)
            pts = leaf.verts[list(f), :2]
            x, y = pts[:, 0], pts[:, 1]
            assert np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) > -1e-15, kw


def test_compound_spread_and_taper():
    # Palmate: the outermost leaflets leave the centre at +-Spread and are Taper
    # times the centre leaflet's size; pinnate: the base pair is Taper times the
    # terminal leaflet.
    from ..growth.leaf import _compound_layout
    kw = dict(terminal=True, sub_pairs=1, rachis_length=0.1, rachis_width=0.002,
              petiolule=0.0, spread=50.0, taper=0.6, pinna_length=0.5)
    places, _strips = _compound_layout("palmate", leaflets=5, **kw)
    angles = sorted(np.degrees(a) for _b, a, _s in places)
    assert np.allclose(angles, [-50, -25, 0, 25, 50])
    scale = {round(float(np.degrees(a))): s for _b, a, s in places}
    assert abs(scale[50] - 0.6) < 1e-12 and abs(scale[0] - 1.0) < 1e-12
    places, _strips = _compound_layout("pinnate", leaflets=4, **kw)
    lowest = min(places, key=lambda p: p[0][1])
    tip = max(places, key=lambda p: p[0][1])
    assert abs(lowest[2] - 0.6) < 1e-12 and abs(tip[2] - 1.0) < 1e-12
    assert abs(abs(np.degrees(lowest[1])) - 50.0) < 1e-9 and tip[1] == 0.0


def test_compound_stays_under_the_cap():
    from ..growth.leaf import generate_compound_leaf
    leaf = generate_compound_leaf(compound="bipinnate", leaflets=4, sub_pairs=6,
                                  length=0.01, width=0.004, stations=40, res_h=8)
    assert leaf.boundary_count <= MAX_LEAF_VERTS
    assert any("hard cap" in w for w in leaf.warnings)
    try:
        generate_compound_leaf(compound="bipinnate", leaflets=20, sub_pairs=20)
    except ValueError as exc:
        assert "fewer leaflets" in str(exc)
    else:
        raise AssertionError("an impossible leaflet count did not raise")


def test_compound_stands_on_its_petiole():
    # Every type: nothing but the petiole strip below Petiole; palmate and
    # trifoliate leaflets start at its top.
    from ..growth.leaf import _compound_layout
    kw = dict(leaflets=5, terminal=True, sub_pairs=2, rachis_length=0.1,
              rachis_width=0.002, petiolule=0.0, spread=60.0, taper=0.7,
              pinna_length=0.5, petiole=0.04)
    for kind in ("trifoliate", "palmate", "pinnate", "bipinnate"):
        places, strips = _compound_layout(kind, **kw)
        assert min(b[1] for b, _a, _s in places) >= 0.04 - 1e-12, kind
        foot = [pts for pts, _w in strips if pts[:, 1].min() < 0.04 - 1e-9]
        assert len(foot) == 1 and np.allclose(foot[0][:, 0], 0.0), kind
        assert abs(foot[0][:, 1].max() - 0.04) < 1e-12, kind
    for kind in ("palmate", "trifoliate"):
        places, _strips = _compound_layout(kind, **kw)
        lowest = min(b[1] for b, _a, _s in places)
        assert abs(lowest - 0.04) < 1e-12, f"{kind}: leaflets not at the petiole top"


def test_compound_rachis_curl_bends_the_whole_leaf():
    from ..growth.leaf import generate_compound_leaf
    flat = generate_compound_leaf(compound="pinnate", leaflets=3)
    up = generate_compound_leaf(compound="pinnate", leaflets=3, rachis_curl=0.3)
    assert flat.faces == up.faces
    tip = np.argmax(flat.verts[:, 1])
    assert up.verts[tip, 2] > 0.01, "rachis curl did not lift the tip"
    base = np.argmin(np.hypot(flat.verts[:, 0], flat.verts[:, 1]))
    assert abs(up.verts[base, 2]) < 1e-9, "the base moved"


# --- plain runner (no pytest needed) ----------------------------------------

def _run() -> int:
    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 -- test runner reports, not raises
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
