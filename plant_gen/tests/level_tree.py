"""Tests for branching levels as a tree.

A level's `parent` says which level it grows on. Levels sharing a parent each
branch at every node of it, with their own settings; masks split the nodes
between them. A chain -- every species -- is the one-child case, byte-identical
to before (the goldens guard that; `test_explicit_chain_is_the_default_chain`
here guards the explicit spelling of it).

    python -m plant_gen.tests.level_tree
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..growth.development import build, grow
from ..growth.masks import MaskEntry
from ..growth.morphology import PARENT_PLANT, BranchLevel, generic_perennial
from .maps import _fingerprint

SEED = 0


def _profile(levels):
    return replace(generic_perennial(), levels=levels)


def _stems(**kw):
    base = generic_perennial().levels[0]
    return replace(base, parent=PARENT_PLANT, **kw)


def _branch(parent, **kw):
    base = generic_perennial().levels[1]
    return replace(base, parent=parent, **kw)


def _axes_on(plant, level):
    return [a for a in plant.axes.values() if a.level == level and not a.is_fork]


def _launch_angle(plant, axis):
    parent = plant.axes[axis.parent_axis_id].phytomers[axis.parent_phytomer_index]
    c = float(np.dot(parent.direction, axis.base_direction))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def test_explicit_chain_is_the_default_chain():
    base = generic_perennial()
    explicit = replace(base, levels=[replace(lv, parent=i - 1)
                                     for i, lv in enumerate(base.levels)])
    assert [explicit.level_parent(i) for i in range(2)] == [-1, 0]
    assert _fingerprint(build(explicit, seed=SEED)) == _fingerprint(build(base, seed=SEED))


def test_sibling_levels_each_branch_every_node():
    # Stems with two branch levels on them: A at 30 degrees, B at 70.
    p = _profile([_stems(), _branch(0, start_pitch_deg=30.0, steps=3),
                  _branch(0, start_pitch_deg=70.0, steps=5)])
    plant = grow(p, seed=SEED)
    stem = plant.axes[0]
    # every vegetative node but the tip (a monopodial stem's last node is bare)
    stem_nodes = (stem.flowering_from if stem.flowering_from >= 0 else len(stem.phytomers)) - 1
    a, b = _axes_on(plant, 1), _axes_on(plant, 2)
    angles_a = sorted({round(_launch_angle(plant, x), 6) for x in a})
    angles_b = sorted({round(_launch_angle(plant, x), 6) for x in b})
    veg_a = {x.flowering_from if x.flowering_from >= 0 else len(x.phytomers) for x in a}
    veg_b = {x.flowering_from if x.flowering_from >= 0 else len(x.phytomers) for x in b}
    print(f"         {stem_nodes} stem nodes: {len(a)} A branches at {angles_a} deg "
          f"({veg_a} nodes), {len(b)} B at {angles_b} deg ({veg_b} nodes)")
    assert len(a) == len(b) == stem_nodes
    assert angles_a == [30.0] and angles_b == [70.0]
    assert veg_a == {3} and veg_b == {5}
    assert all(x.order == 1 for x in a + b)


def test_masks_split_a_parent_between_siblings():
    low = MaskEntry("factor", 0.0, 0.5)
    high = MaskEntry("factor", 0.0, 0.5, invert=True)
    p = _profile([_stems(), _branch(0, masks=[low]), _branch(0, masks=[high])])
    plant = grow(p, seed=SEED)
    stem = plant.axes[0]
    at = {lvl: sorted(stem.phytomers[x.parent_phytomer_index].factor for x in _axes_on(plant, lvl))
          for lvl in (1, 2)}
    print(f"         lower level on factors {[round(v, 2) for v in at[1]]}, "
          f"upper on {[round(v, 2) for v in at[2]]}")
    assert at[1] and at[2]
    assert max(at[1]) <= 0.5 < min(at[2])
    veg = stem.flowering_from if stem.flowering_from >= 0 else len(stem.phytomers)
    assert len(at[1]) + len(at[2]) == veg - 1     # every node but the bare tip


def test_grandchildren_grow_only_on_their_parent():
    # Stems -> A and B; C grows on B only.
    p = _profile([_stems(), _branch(0, steps=3), _branch(0, steps=4),
                  _branch(2, start_pitch_deg=40.0, steps=2)])
    plant = grow(p, seed=SEED)
    c = _axes_on(plant, 3)
    parents = {plant.axes[x.parent_axis_id].level for x in c}
    print(f"         {len(c)} level-C branches, all on level {parents}")
    assert c and parents == {2}
    assert all(x.order == 2 for x in c)
    assert not any(plant.axes[x.parent_axis_id].level == 1 for x in c)


def test_two_stems_levels_both_emerge():
    p = _profile([_stems(whorl=3), _branch(0), _stems(whorl=2, start_pitch_deg=25.0)])
    plant = grow(p, seed=SEED)
    basal = [plant.axes[i] for i in plant.basal_axis_ids]
    by_level = sorted(a.level for a in basal)
    print(f"         basal stems by level {by_level}")
    assert by_level == [0, 0, 0, 2, 2]
    # the second stems level bears no branches (nothing grows on it)
    assert not any(plant.axes[a.parent_axis_id].level == 2
                   for a in plant.axes.values() if a.parent_axis_id >= 0 and not a.is_fork
                   and a.structural_depth < 1)


def test_span_is_the_longest_path():
    # stems 12 -> A 3, and -> B 4 -> C 6: the longest is stems + B + C
    p = _profile([_stems(steps=12), _branch(0, steps=3), _branch(0, steps=4),
                  _branch(2, steps=6)])
    chain = (12 + 1) + (4 + 1) + (6 + 1)
    extra = p.total_steps - chain
    q = _profile([_stems(steps=12), _branch(0, steps=4), _branch(1, steps=6)])
    assert q.total_steps - ((12 + 1) + (4 + 1) + (6 + 1)) == extra
    assert p.total_steps == q.total_steps
    assert p.max_level_depth == 2 and p.level_children(0) == [1, 2]


def test_a_loop_reads_as_a_stems_level():
    lv = BranchLevel
    p = _profile([lv(parent=PARENT_PLANT), lv(parent=2), lv(parent=1)])
    assert p.level_parent(1) == -1 and p.level_parent(2) == -1
    assert p.max_level_depth == 0
    grow(p, seed=SEED)          # must not hang


# --- plain runner (no pytest needed) -----------------------------------------------

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
