"""Tests for the bpy-free Traits pick (`growth/traits.py`), against closed forms.

The contract: a neutral sheet is detected (the caller keeps the uniform pick);
with equal chances the walk lands where the uniform pick does; a +1 preference
makes a mesh's chance proportional to the channel, -1 to its complement, 0 to
neither; Weights scale frequency; an absent channel has no effect; a point
every mesh scores 0 falls back to the Weights.

    python -m plant_gen.tests.traits
"""

from __future__ import annotations

import numpy as np

from ..growth.traits import chances, is_neutral, pick


def _draws(n: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def test_neutral_detection():
    assert is_neutral([1, 1, 1], np.zeros((3, 2)))
    assert is_neutral([2.5, 2.5], np.zeros((2, 0)))
    assert not is_neutral([1, 2], np.zeros((2, 1))), "unequal Weights are not neutral"
    assert not is_neutral([1, 1], [[0.0], [0.3]]), "a preference is not neutral"
    assert not is_neutral([0, 0], np.zeros((2, 1))), "all-zero Weights are not neutral"


def test_equal_chances_match_uniform_pick():
    k, n = 5, 20000
    d = _draws(n)
    uniform = np.clip((d * k).astype(np.int64), 0, k - 1)
    got = pick(np.ones(k), np.zeros((k, 1)), np.full((n, 1), 0.3), d)
    assert np.array_equal(got, uniform), "equal chances must land where floor(draw*k) does"


def test_plus_minus_one_is_proportional():
    # Two meshes: young (-1 on Age), old (+1). P(old) at age a = a exactly.
    a = np.array([0.0, 0.1, 0.25, 0.5, 0.8, 1.0])
    ch = chances([1, 1], [[-1.0], [1.0]], a[:, None])
    p_old = ch[:, 1] / ch.sum(axis=1)
    assert np.allclose(p_old, a), p_old
    # And empirically through the seeded walk, at a = 0.25.
    n = 200000
    got = pick([1, 1], [[-1.0], [1.0]], np.full((n, 1), 0.25), _draws(n))
    frac = got.mean()
    assert abs(frac - 0.25) < 0.005, frac


def test_half_preference_blends():
    # One mesh at +0.5, one neutral: chances (1 + 0.5(2c-1)) : 1.
    c = np.array([0.0, 0.5, 1.0])
    ch = chances([1, 1], [[0.5], [0.0]], c[:, None])
    assert np.allclose(ch[:, 0], [0.5, 1.0, 1.5]) and np.allclose(ch[:, 1], 1.0)


def test_weights_scale_frequency():
    n = 200000
    got = pick([3, 1], np.zeros((2, 0)), np.zeros((n, 0)), _draws(n, 3))
    assert abs((got == 0).mean() - 0.75) < 0.005


def test_columns_multiply():
    # Mesh 0 +1 on both columns: chance c1 * c2 * ... relative to a neutral mesh.
    v = np.array([[0.5, 0.5], [1.0, 0.25], [0.2, 1.0]])
    ch = chances([1, 1], [[1.0, 1.0], [0.0, 0.0]], v)
    assert np.allclose(ch[:, 0], (2 * v[:, 0]) * (2 * v[:, 1]))


def test_absent_channel_has_no_effect():
    ch = chances([1, 2], [[1.0], [-1.0]], np.full((4, 1), np.nan))
    assert np.allclose(ch, [[1, 2]] * 4)


def test_all_zero_falls_back_to_weights():
    # Both meshes want Age high; at Age 0 both score 0 -> Weights 1:3.
    n = 100000
    got = pick([1, 3], [[1.0], [1.0]], np.zeros((n, 1)), _draws(n, 5))
    assert abs((got == 1).mean() - 0.75) < 0.006
    # ...and with every Weight 0, uniform.
    got = pick([0, 0], [[1.0], [1.0]], np.zeros((n, 1)), _draws(n, 5))
    assert abs((got == 1).mean() - 0.5) < 0.006


def test_deterministic():
    d = _draws(1000)
    v = np.random.default_rng(1).random((1000, 2))
    a = pick([1, 2, 1], [[0.5, -1], [0, 1], [-0.3, 0]], v, d)
    b = pick([1, 2, 1], [[0.5, -1], [0, 1], [-0.3, 0]], v, d)
    assert np.array_equal(a, b)


# --- plain runner (no pytest needed) -----------------------------------------

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
