"""Tests for floral heliotropism per inflorescence and its stalk bend, against
closed forms.

The contract: a flower turns its face the fraction `heliotropism` of the way to
its sun (the angle to it shrinks by exactly that fraction); each inflorescence
turns at its own strength toward its own sun, so one at 0 is untouched while
another tracks; the flower's own stalk curves with it and nothing else
moves -- a capped flower's top internode bends from the node below it unless a
branch leaves it, a pedicel from where it leaves its node, each as one even arc
by the flower's turn (unbent, it lies on the stalk it was; no joint turns more
than a facet); every other node stays where it was; and an older preset's plant-wide
`flower_heliotropism` migrates onto every inflorescence.

    python -m plant_gen.tests.heliotropism
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from ..growth import development as dev
from ..growth import serialize, species
from ..growth.environment import Environment
from ..growth.model import OrganKind, rotate_about
from ..growth.morphology import Attachment, InflorescenceKind
from ..growth.physics import _MAX_ARC_FACET_DEG
from .growth_patterns import RACEME, STEM, _angle, _flowers, _node_borne, _on, _profile

SUN = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
OTHER_SUN = np.array([0.0, -1.0, 0.2]) / np.linalg.norm([0.0, -1.0, 0.2])


def _two(primary_k: float, extra_k: float):
    """A terminal raceme (inflorescence 0) and axillary solitaries (1) on one stem."""
    extra = replace(_on(_node_borne(InflorescenceKind.SOLITARY, Attachment.AXILLARY), 0),
                    heliotropism=extra_k)
    return _profile(STEM, inflorescence=replace(_on(RACEME, 0), heliotropism=primary_k),
                    extra_inflorescences=[extra])


def _faces(profile, env=None) -> dict:
    plant = dev.build(profile, seed=3, environment=env)
    return {o.id: (o.inflorescence, np.array(o.direction), np.array(o.position))
            for _a, o in _flowers(plant)}


def _sun(*flower_suns):
    return Environment(sun_target=SUN.copy(), flower_sun_targets=tuple(flower_suns))


def test_each_inflorescence_its_own_strength():
    still = _faces(_two(0.0, 0.0))
    turned = _faces(_two(1.0, 0.0), _sun())
    assert {i for i, _d, _p in still.values()} == {0, 1}
    for oid, (inf, d, p) in turned.items():
        if inf == 0:
            assert _angle(d, SUN) < 1e-6, f"flower {oid} at strength 1 faces the sun"
        else:
            assert np.allclose(d, still[oid][1]) and np.allclose(p, still[oid][2]), \
                f"flower {oid} at strength 0 is untouched"


def test_each_inflorescence_its_own_sun():
    # Inflorescence 0 tracks nothing (None), 1 its own sun -- never the plant's.
    still = _faces(_two(1.0, 1.0))
    turned = _faces(_two(1.0, 1.0), _sun(None, OTHER_SUN))
    for oid, (inf, d, _p) in turned.items():
        if inf == 0:
            assert np.allclose(d, still[oid][1]), f"flower {oid} has no sun of its own"
        else:
            assert _angle(d, OTHER_SUN) < 1e-6, f"flower {oid} faces its own sun"


def test_turns_the_stated_fraction():
    still = _faces(_two(0.0, 0.0))
    half = _faces(_two(0.0, 0.5), _sun())
    checked = 0
    for oid, (inf, d, _p) in half.items():
        before = _angle(still[oid][1], SUN)
        if inf != 1 or before < 1.0 or before > 179.0:
            continue
        assert abs(_angle(d, SUN) - 0.5 * before) < 1e-6, (oid, before, _angle(d, SUN))
        checked += 1
    assert checked > 0


def test_off_is_byte_identical():
    on, off = _faces(_two(0.0, 0.0), _sun()), _faces(_two(0.0, 0.0))
    assert on.keys() == off.keys()
    assert all(np.array_equal(on[k][1], off[k][1]) and np.array_equal(on[k][2], off[k][2])
               for k in on)


def _top_group(axis) -> int:
    start = len(axis.phytomers) - 1
    while start > 0 and axis.phytomers[start - 1].is_split:
        start -= 1
    return start


def _turn(direction, k):
    """The (axis, angle) a flower facing `direction` turns by at strength `k`."""
    d = np.asarray(direction, float)
    axis = np.cross(d, SUN)
    axis /= np.linalg.norm(axis)
    return axis, k * _angle(d, SUN)


def _unbent(points, axis, angle):
    """`points` with an even arc of `angle` taken back out -- each piece turned
    back by `angle` x its midpoint's share of the length -- and the biggest
    turn the arc put between two pieces."""
    pts = [np.asarray(p, float) for p in points]
    pieces = [b - a for a, b in zip(pts, pts[1:], strict=False)]
    lengths = [float(np.linalg.norm(v)) for v in pieces]
    total = sum(lengths)
    out, run = [pts[0]], 0.0
    for v, n in zip(pieces, lengths, strict=True):
        out.append(out[-1] + rotate_about(v, axis, -angle * (run + 0.5 * n) / total))
        run += n
    step = max([angle * 0.5 * (m + n) / total
                for m, n in zip(lengths, lengths[1:], strict=False)] or [0.0])
    return out, step


def _off_line(point, line) -> float:
    """How far `point` is from the polyline `line`."""
    best = np.inf
    for a, b in zip(line, line[1:], strict=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        t = np.clip(float(np.dot(point - a, ab)) / max(float(np.dot(ab, ab)), 1e-30), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point - (a + ab * t))))
    return best


def _check_arc(old, new, axis, angle, label):
    """`new` is `old` bent by `angle` about `axis` as one even arc: unbent it
    lies on `old`, end to end, and no joint turns more than a facet."""
    unbent, step = _unbent(new, axis, angle)
    assert np.allclose(unbent[0], old[0]), label
    assert np.allclose(unbent[-1], old[-1], atol=1e-9), (label, "same length")
    assert max(_off_line(q, old) for q in unbent) < 1e-9, (label, "on the old stalk")
    assert step <= _MAX_ARC_FACET_DEG + 1e-9, (label, step)


def test_capped_flower_bends_its_free_top_internode_only():
    # A branched dichasium (the cyme stamp): its central flowers sit where
    # their cyme forks, the outermost on free tips.
    k = 0.8
    base = species.profile("ANEMONE")
    base = replace(base, inflorescence=replace(
        base.inflorescence, kind=InflorescenceKind.DICHASIUM, laterals_per_node=2,
        branching_depth=3, heliotropism=k))
    still = dev.build(base, seed=3)
    bent = dev.build(base, seed=3, environment=Environment(sun_target=SUN.copy()))
    branched = {(a.parent_axis_id, a.parent_phytomer_index)
                for a in still.axes.values() if a.parent_axis_id >= 0}
    free = held = smoothed = 0
    for aid, a in still.axes.items():
        b = bent.axes[aid]
        start = _top_group(a)
        for i in range(start):        # every node below the top internode stays put
            assert np.allclose(a.phytomers[i].tip_position, b.phytomers[i].tip_position)
        f = a.terminal_organ
        if f is None or f.kind != OrganKind.FLOWER or not a.phytomers:
            continue
        top_free = not any((aid, i) in branched for i in range(start, len(a.phytomers)))
        old = [a.phytomers[start].base_position] + [p.tip_position for p in a.phytomers[start:]]
        new = [b.phytomers[start].base_position] + [p.tip_position for p in b.phytomers[start:]]
        if not top_free:
            assert len(old) == len(new) and np.allclose(old, new), \
                f"axis {aid}: a branch leaves its top node"
            held += 1
            continue
        axis, angle = _turn(f.direction, k)
        _check_arc(old, new, axis, angle, f"axis {aid}")
        # The node stays on the last piece, with its organs.
        assert not b.phytomers[-1].is_split
        assert len(b.phytomers[-1].organs) == len(a.phytomers[-1].organs)
        assert np.allclose(b.terminal_organ.direction,
                           rotate_about(np.asarray(f.direction, float), axis, angle))
        smoothed += len(new) > len(old)
        free += 1
    assert free > 0 and held > 0 and smoothed > 0, (free, held, smoothed)


def test_pedicel_bends_about_its_node():
    k = 0.6
    base = species.profile("FOXGLOVE")
    base = replace(base, inflorescence=replace(base.inflorescence, heliotropism=k))
    still = dev.build(base, seed=3)
    bent = dev.build(base, seed=3, environment=Environment(sun_target=SUN.copy()))
    checked = 0
    for aid, a in still.axes.items():
        b = bent.axes[aid]
        # A capped flower's top internode may bend (checked above).
        limit = _top_group(a) if a.terminal_organ is not None else len(a.phytomers)
        for i in range(limit):
            pa, pb = a.phytomers[i], b.phytomers[i]
            assert np.allclose(pa.tip_position, pb.tip_position), (aid, i)
            for oa, ob in zip(pa.organs, pb.organs, strict=True):
                if oa.kind != OrganKind.FLOWER or oa.pedicel_length <= 0.0 or oa.scale <= 0.0:
                    continue
                anchor = pa.base_position + (pa.tip_position - pa.base_position) * min(
                    oa.along_internode, 1.0)
                old = oa.pedicel_points or [anchor, oa.position]
                new = ob.pedicel_points or [anchor, ob.position]
                axis, angle = _turn(oa.direction, k)
                _check_arc(old, new, axis, angle, f"flower {oa.id}")
                assert np.allclose(ob.position, new[-1])
                assert np.allclose(ob.direction,
                                   rotate_about(np.asarray(oa.direction, float), axis, angle))
                checked += 1
    assert checked > 0


def test_plant_wide_value_migrates_onto_every_inflorescence():
    profile = serialize.profile_to_dict(_two(0.0, 0.0))
    for inf in [profile["inflorescence"], *profile["extra_inflorescences"]]:
        inf.pop("heliotropism")
    profile["flower_heliotropism"] = 0.4
    old = {"schema_version": 10, "kind": serialize.KIND, "name": "old", "profile": profile}
    migrated = serialize.from_json(json.dumps(old)).profile
    assert migrated.inflorescence.heliotropism == 0.4
    assert [i.heliotropism for i in migrated.extra_inflorescences] == [0.4]
    assert migrated.tracks_sun


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
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
