"""Tests for ground support in the gravity pass, headless.

The ground (the plant's own base level) carries whatever reaches it: a
segment that would bend below it lands on it instead, keeping its compass
heading, so a stem that reaches the ground lies along it. A plant that never
gets there bends exactly as before.

1. A long, soft, level stem -- a cantilever that would hang far below its
   base -- lies on the ground: no node below it, the stem's reach kept.
2. Without the ground it really would hang (the test's own control).
3. A plant clear of the ground is byte-identical with and without it.

    python -m plant_gen.tests.ground_support
"""

from __future__ import annotations

import numpy as np

from ..growth import development as dev
from ..growth import physics
from ..growth.morphology import BranchLevel, Inflorescence, InflorescenceKind, MorphologyProfile

SEED = 3
NO_FLOWERS = Inflorescence(kind=InflorescenceKind.NONE)
# a level runner, 1.2 m of thin, soft stem: a cantilever that cannot hold itself out
RUNNER = BranchLevel(start_pitch_deg=90.0, steps=40, start_length_scale=0.03,
                     start_width_scale=0.0015, stiffness=5e7)
UPRIGHT = BranchLevel(start_pitch_deg=10.0, steps=10, start_length_scale=0.03,
                      start_width_scale=0.004)


def _profile(level: BranchLevel) -> MorphologyProfile:
    return MorphologyProfile(levels=[level], gravitropism=0.0, circumnutation_deg=0.0,
                             inflorescence=NO_FLOWERS)


def _points(plant):
    return np.array([p.tip_position for a in plant.axes.values() for p in a.phytomers])


def _without_ground(profile):
    orig = physics._ground_lift
    physics._ground_lift = lambda *a, **k: None
    try:
        return dev.build(profile, seed=SEED)
    finally:
        physics._ground_lift = orig


def test_a_stem_reaching_the_ground_lies_on_it():
    plant = dev.build(_profile(RUNNER), seed=SEED)
    phs = [p for a in plant.axes.values() for p in a.phytomers]
    # it rests on the ground: each node's underside (centre - radius) at the
    # base level -- never below it, and no higher once lying
    under = np.array([p.tip_position[2] - p.radius for p in phs])
    assert under.min() >= -1e-9, under.min()
    lying = under[len(under) // 4:]
    assert np.abs(lying).max() < 1e-6, np.abs(lying).max()
    # the reach is kept: the stem's length runs out along the ground
    pts = _points(plant)
    reach = float(np.linalg.norm(pts[-1] - np.array(phs[0].base_position)))
    length = sum(p.length for p in phs)
    assert abs(reach - length) < 1e-3, (reach, length)


def test_without_the_ground_it_would_hang():
    pts = _points(_without_ground(_profile(RUNNER)))
    assert pts[:, 2].min() < -0.5, pts[:, 2].min()


def _cone(slope: float):
    """Terrain rising (or falling) `slope` per metre out from the plant's base."""
    from ..growth.environment import Environment, Ground
    xs = np.linspace(-2.0, 2.0, 81)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    heights = slope * np.hypot(gx, gy)
    return Environment(ground=Ground(origin=np.array([-2.0, -2.0]), cell=0.05,
                                     heights=heights))


def test_a_stem_follows_uneven_ground():
    for slope in (0.15, -0.3):
        env = _cone(slope)
        plant = dev.build(_profile(RUNNER), seed=SEED, environment=env)
        phs = [p for a in plant.axes.values() for p in a.phytomers]
        gaps = []
        for p in phs:
            x, y, z = p.tip_position
            surface = env.ground.height(float(x), float(y))
            gaps.append(z - p.radius - surface)
        gaps = np.array(gaps)
        # never into the terrain, and lying on it (within a mm) up a hill and
        # down one alike: a lying stem roots, so it follows a falling slope
        # instead of standing off it as a stiff cantilever
        assert gaps.min() > -1e-3, (slope, gaps.min())
        assert np.abs(gaps).max() < 1e-3, (slope, np.abs(gaps).max())


def _ridge():
    """A ring-shaped ridge 6 cm high, 40 cm out from the plant's base."""
    from ..growth.environment import Environment, Ground
    xs = np.linspace(-2.0, 2.0, 81)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    heights = 0.06 * np.exp(-((np.hypot(gx, gy) - 0.4) / 0.15) ** 2)
    return Environment(ground=Ground(origin=np.array([-2.0, -2.0]), cell=0.05,
                                     heights=heights))


def test_a_stem_comes_back_down_over_a_crest():
    # climbing a ridge tilts the stem up; past the crest it must not carry
    # that tilt on into the air -- it lies, so it follows the far side down
    env = _ridge()
    plant = dev.build(_profile(RUNNER), seed=SEED, environment=env)
    gaps = np.array([p.tip_position[2] - p.radius
                     - env.ground.height(float(p.tip_position[0]), float(p.tip_position[1]))
                     for a in plant.axes.values() for p in a.phytomers])
    assert gaps.min() > -1e-3, gaps.min()
    # within a stem radius: on the crest itself the arc the smoothing draws
    # between two 3 cm nodes is rounder than this sharp ridge (1.2 mm off it)
    assert np.abs(gaps).max() < RUNNER.start_width_scale, np.abs(gaps).max()


def test_a_stem_starts_on_the_terrain_under_it():
    # stems grow from the plant's own level; on a tilted terrain 2 cm above
    # it (rising 0.15 a metre along +x) each stem's foot sits on the surface,
    # and a runner lies along it from there
    from ..growth.environment import Environment, Ground
    xs = np.linspace(-2.0, 2.0, 81)
    gx, _gy = np.meshgrid(xs, xs, indexing="ij")
    env = Environment(ground=Ground(origin=np.array([-2.0, -2.0]), cell=0.05,
                                    heights=0.02 + 0.15 * gx))
    plant = dev.build(_profile(RUNNER), seed=SEED, environment=env)
    for axis_id in plant.basal_axis_ids:
        base = plant.axes[axis_id].base_position
        assert abs(base[2] - env.ground.height(float(base[0]), float(base[1]))) < 1e-9, base
    gaps = np.array([p.tip_position[2] - p.radius
                     - env.ground.height(float(p.tip_position[0]), float(p.tip_position[1]))
                     for a in plant.axes.values() for p in a.phytomers])
    assert gaps.min() > -1e-3 and np.abs(gaps).max() < 1e-3, (gaps.min(), gaps.max())


def test_an_upright_on_a_slope_is_left_to_rise():
    # following the ground is for a stem lying on it: one heading up off it
    # bends exactly as on flat ground
    a = dev.build(_profile(UPRIGHT), seed=SEED)
    b = dev.build(_profile(UPRIGHT), seed=SEED, environment=_cone(-0.3))
    assert np.array_equal(_points(a), _points(b))


def test_a_ground_only_environment_grows_identically():
    # the terrain is read by the gravity pass alone: growth itself is untouched
    a = dev.grow(_profile(RUNNER), seed=SEED)
    b = dev.grow(_profile(RUNNER), seed=SEED, environment=_cone(0.15))
    assert np.array_equal(_points(a), _points(b))


def test_a_plant_clear_of_the_ground_is_unchanged():
    a = dev.build(_profile(UPRIGHT), seed=SEED)
    b = _without_ground(_profile(UPRIGHT))
    assert np.array_equal(_points(a), _points(b))


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
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
