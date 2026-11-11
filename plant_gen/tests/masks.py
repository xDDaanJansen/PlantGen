"""Tests for the bpy-free mask/map evaluator (`growth/masks.py`).

Locks the survival contract a later refactor could silently break: empty = no-op,
hard windows are deterministic keep/drop, soft windows/prune are seeded thinning,
combine ops compose, curves remap, a missing channel removes nothing, and the
survival draw is order-independent. All numeric -- the project standard.

    python -m plant_gen.tests.masks
"""

from __future__ import annotations

import numpy as np

from ..growth.masks import MaskEntry, MaskStack, evaluate


def _ids(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


# --- no-op --------------------------------------------------------------------

def test_empty_stack_is_noop():
    ids = _ids(500)
    mask, survive = evaluate(MaskStack(), {}, ids)
    assert np.all(mask == 1.0), "empty stack should give mask 1 everywhere"
    assert np.all(survive), "empty stack should keep every point"


def test_none_stack_is_noop():
    ids = _ids(64)
    mask, survive = evaluate(None, {}, ids)
    assert np.all(survive) and np.all(mask == 1.0)


def test_empty_ids_is_safe():
    empty = np.array([], dtype=np.int64)
    mask, survive = evaluate(MaskStack([MaskEntry("height")]), {"height": np.array([])}, empty)
    assert mask.shape == (0,) and survive.shape == (0,)


# --- keep-band: deterministic keep/drop --------------------------------------

def test_band_keeps_inside_min_max():
    # min AND max both bound the kept range -- only 0.4..0.6 survives.
    ids = _ids(11)
    factor = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ... 1.0
    stack = MaskStack([MaskEntry("factor", lo=0.4, hi=0.6)])
    mask, survive = evaluate(stack, {"factor": factor}, ids)
    assert set(np.unique(mask)).issubset({0.0, 1.0})  # a band is a hard keep/drop
    assert np.array_equal(survive, (factor >= 0.4) & (factor <= 0.6))


def test_band_is_order_independent_in_lo_hi():
    ids = _ids(11)
    factor = np.linspace(0.0, 1.0, 11)
    lohi = evaluate(MaskStack([MaskEntry("factor", lo=0.3, hi=0.7)]), {"factor": factor}, ids)[1]
    hilo = evaluate(MaskStack([MaskEntry("factor", lo=0.7, hi=0.3)]), {"factor": factor}, ids)[1]
    assert np.array_equal(lohi, hilo)


def test_binary_channel_selects_on_side_ignoring_lo_hi():
    # on_branch is a 0/1 flag: it keeps the "1" side whatever lo/hi are set to
    # (min 0 must not read as "keep both"), and invert flips to the "0" side.
    ids = _ids(6)
    flag = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    stack = MaskStack([MaskEntry("on_branch", lo=0.0, hi=1.0)])
    keep = evaluate(stack, {"on_branch": flag}, ids)[1]
    assert np.array_equal(keep, flag > 0.5)     # only the 1s, even at lo=0
    inv = evaluate(MaskStack([MaskEntry("on_branch", lo=0.0, hi=1.0, invert=True)]),
                   {"on_branch": flag}, ids)[1]
    assert np.array_equal(inv, flag < 0.5)      # invert -> only the 0s


def test_full_band_keeps_everything():
    # The default [0, 1] band is a no-op -- a fresh filter changes nothing.
    ids = _ids(50)
    vals = np.linspace(0, 1, 50)
    mask, survive = evaluate(MaskStack([MaskEntry("age", lo=0.0, hi=1.0)]), {"age": vals}, ids)
    assert np.all(mask == 1.0) and survive.all()


# --- modulo: every Nth by raw ordinal ---------------------------------------

def test_modulo_keeps_every_nth():
    ids = _ids(6)
    ordinal = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    stack = MaskStack([MaskEntry("node_index", modulo=2)])
    ch = {"node_index": np.ones(6), "node_ordinal": ordinal}
    keep = evaluate(stack, ch, ids)[1]
    assert np.array_equal(keep, ordinal % 2 == 0)          # 0, 2, 4


def test_modulo_combines_with_band():
    # every other node AND only ordinals whose normalised value is in the upper band
    ids = _ids(6)
    ordinal = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    norm = ordinal / 5.0
    stack = MaskStack([MaskEntry("node_index", lo=0.5, hi=1.0, modulo=2)])
    keep = evaluate(stack, {"node_index": norm, "node_ordinal": ordinal}, ids)[1]
    assert np.array_equal(keep, (ordinal % 2 == 0) & (norm >= 0.5))  # 4 only


def test_modulo_off_is_noop():
    ids = _ids(6)
    ordinal = np.arange(6, dtype=float)
    stack = MaskStack([MaskEntry("node_index", modulo=0)])
    keep = evaluate(stack, {"node_index": np.ones(6), "node_ordinal": ordinal}, ids)[1]
    assert keep.all()


def test_modulo_invert_flips():
    ids = _ids(6)
    ordinal = np.arange(6, dtype=float)
    plain = MaskStack([MaskEntry("node_index", modulo=2)])
    inv = MaskStack([MaskEntry("node_index", modulo=2, invert=True)])
    ch = {"node_index": np.ones(6), "node_ordinal": ordinal}
    assert np.array_equal(evaluate(inv, ch, ids)[1], ~evaluate(plain, ch, ids)[1])


# --- prune (keep fraction) ---------------------------------------------------

def test_prune_only_keeps_the_fraction():
    ids = _ids(20000)
    stack = MaskStack(keep=0.25)  # no entries: pure prune
    mask, survive = evaluate(stack, {}, ids)
    assert np.allclose(mask, 0.25)
    assert abs(survive.mean() - 0.25) < 0.02


def test_prune_is_seeded_and_reproducible():
    ids = _ids(1000)
    a = evaluate(MaskStack(keep=0.5, seed=7), {}, ids)[1]
    b = evaluate(MaskStack(keep=0.5, seed=7), {}, ids)[1]
    assert np.array_equal(a, b)
    c = evaluate(MaskStack(keep=0.5, seed=8), {}, ids)[1]
    assert not np.array_equal(a, c), "different seed should thin differently"


# --- invert ------------------------------------------------------------------

def test_invert_flips_a_band():
    ids = _ids(11)
    factor = np.linspace(0.0, 1.0, 11)
    plain = evaluate(MaskStack([MaskEntry("factor", lo=0.3, hi=0.7)]), {"factor": factor}, ids)[1]
    inv_stack = MaskStack([MaskEntry("factor", lo=0.3, hi=0.7, invert=True)])
    inv = evaluate(inv_stack, {"factor": factor}, ids)[1]
    assert np.array_equal(inv, ~plain)


# --- combine ops: set operations on bands ------------------------------------

def test_multiply_is_and():
    # keep factor in [0.5, 1] AND age in [0, 0.5) -- multiply is the default AND.
    ids = _ids(100)
    factor = np.linspace(0, 1, 100)
    age = np.linspace(0, 1, 100)
    stack = MaskStack([
        MaskEntry("factor", lo=0.5, hi=1.0),
        MaskEntry("age", lo=0.5, hi=1.0, invert=True, op="multiply"),
    ])
    _, survive = evaluate(stack, {"factor": factor, "age": age}, ids)
    assert np.array_equal(survive, (factor >= 0.5) & (age < 0.5))


def test_add_is_union():
    ids = _ids(100)
    factor = np.linspace(0, 1, 100)
    age = np.linspace(0, 1, 100)
    stack = MaskStack([
        MaskEntry("factor", lo=0.5, hi=1.0),
        MaskEntry("age", lo=0.5, hi=1.0, invert=True, op="add"),
    ])
    _, survive = evaluate(stack, {"factor": factor, "age": age}, ids)
    assert np.array_equal(survive, (factor >= 0.5) | (age < 0.5))


def test_subtract_removes_a_band():
    # keep factor in [0.2, 0.8], then subtract the middle [0.4, 0.6].
    ids = _ids(100)
    factor = np.linspace(0, 1, 100)
    stack = MaskStack([
        MaskEntry("factor", lo=0.2, hi=0.8),
        MaskEntry("factor", lo=0.4, hi=0.6, op="subtract"),
    ])
    _, survive = evaluate(stack, {"factor": factor}, ids)
    want = ((factor >= 0.2) & (factor <= 0.8)) & ~((factor >= 0.4) & (factor <= 0.6))
    assert np.array_equal(survive, want)


def test_first_entry_op_is_ignored_base():
    # A single entry with op='add' must use its own band as the base, not
    # max(1.0, w) -- otherwise everything would survive.
    ids = _ids(11)
    factor = np.linspace(0, 1, 11)
    stack = MaskStack([MaskEntry("factor", lo=0.5, hi=1.0, op="add")])
    mask, _ = evaluate(stack, {"factor": factor}, ids)
    assert np.array_equal(mask, (factor >= 0.5).astype(float))


# --- robustness --------------------------------------------------------------

def test_missing_channel_is_identity():
    ids = _ids(50)
    # 'nonesuch' is not supplied and not a known channel -> removes nothing.
    mask, survive = evaluate(MaskStack([MaskEntry("nonesuch")]), {}, ids)
    assert np.all(mask == 1.0) and np.all(survive)


def test_survival_is_order_independent():
    # Prune's seeded draw must give each id the same fate regardless of position.
    ids = _ids(500)
    stack = MaskStack(keep=0.5, seed=5)
    _, base = evaluate(stack, {}, ids)
    perm = np.random.default_rng(9).permutation(500)
    _, shuffled = evaluate(stack, {}, ids[perm])
    assert np.array_equal(base[perm], shuffled)


def test_random_channel_is_noise_not_all_pass():
    ids = _ids(4000)
    # A random-channel band keeping the lower half thins ~half, deterministically.
    stack = MaskStack([MaskEntry("random", lo=0.0, hi=0.5)], seed=2)
    _, survive = evaluate(stack, {}, ids)
    assert 0.4 < survive.mean() < 0.6
    _, again = evaluate(MaskStack([MaskEntry("random", lo=0.0, hi=0.5)], seed=2), {}, ids)
    assert np.array_equal(survive, again)


def test_muted_entry_skipped():
    # Muting the second map reproduces the stack without it -- and does not shift
    # the first map's own seeded survival (its salt index is unchanged).
    ids = _ids(50)
    vals = np.linspace(0, 1, 50)
    only_first = MaskStack([MaskEntry("age", lo=0.3, hi=0.7)], seed=5)
    muted_second = MaskStack([MaskEntry("age", lo=0.3, hi=0.7),
                              MaskEntry("age", lo=0.5, hi=0.9, mute=True)], seed=5)
    m0, s0 = evaluate(only_first, {"age": vals}, ids)
    m1, s1 = evaluate(muted_second, {"age": vals}, ids)
    assert np.allclose(m0, m1) and np.array_equal(s0, s1)


def test_all_muted_is_no_mask():
    ids = _ids(20)
    vals = np.linspace(0, 1, 20)
    stack = MaskStack([MaskEntry("age", lo=0.4, hi=0.6, mute=True)])
    mask, survive = evaluate(stack, {"age": vals}, ids)
    assert np.allclose(mask, 1.0) and survive.all()


def test_unknown_op_raises():
    ids = _ids(10)
    vals = np.linspace(0, 1, 10)
    try:
        evaluate(MaskStack([MaskEntry("age"), MaskEntry("age", op="nonsense")]), {"age": vals}, ids)
    except ValueError:
        return
    raise AssertionError("expected ValueError on an unknown combine op")


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
