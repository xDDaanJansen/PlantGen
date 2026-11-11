"""Tests for FlowerGen's Around the Whorl curves, headless, against closed forms.

Each petal's Length, Width and Pitch are multiplied by a curve over its place
on the ring, measured both ways from the first petal (0) to straight opposite
it (1) -- so a whorl comes out mirror-symmetric about the first petal: a
bilateral corolla (a pea flower's standard, wings and keel). Flat at 1 (the
default) the whorl is the radial one, unchanged.

    python -m plant_gen.tests.flower_whorl
"""

from __future__ import annotations

import math
from dataclasses import replace

from ..growth.flower import FlowerMeshParams, build_flower_mesh
from ..growth.flower_params import Ramp

COUNT = 5
LENGTH = 0.02
PITCH = math.radians(60.0)


def _open(**kw) -> FlowerMeshParams:
    """An open five-petal flower whose petals run straight from their base."""
    return replace(FlowerMeshParams(), bloom=1.0, petal_count=COUNT, petal_length=LENGTH,
                   petal_curl_updown=0.0, petal_curl_leftright=0.0, petal_curl_outin=0.0,
                   petal_gravity=0.0, petal_twist=0.0, petal_rotation_pitch=PITCH,
                   petal_clamp_overlap=0.0, **kw)


def _length(spine) -> float:
    return sum(math.dist(a, b) for a, b in zip(spine.points, spine.points[1:], strict=False))


def _pitch(spine) -> float:
    (x0, y0, z0), (x1, y1, z1) = spine.points[0], spine.points[1]
    return math.atan2(z1 - z0, math.hypot(x1 - x0, y1 - y0))


def _place(k: int) -> float:
    return 2.0 * min(k, COUNT - k) / COUNT


def test_flat_curves_leave_the_radial_whorl():
    flat = Ramp.constant(1.0)
    a = build_flower_mesh(_open())
    b = build_flower_mesh(_open(petal_whorl_length_profile=flat,
                                petal_whorl_width_profile=flat,
                                petal_whorl_pitch_profile=flat))
    assert [p.points for p in a.petals] == [p.points for p in b.petals]
    lengths = [_length(p) for p in a.petals]
    assert max(lengths) - min(lengths) < 1e-12, lengths


def test_each_petal_takes_the_curves_at_its_place():
    length = Ramp([(0.0, 1.0), (1.0, 0.6)])
    width = Ramp([(0.0, 1.0), (0.4, 0.5), (1.0, 0.5)])
    pitch = Ramp([(0.0, 0.25), (0.3, 1.0), (1.0, 1.0)])
    flat = build_flower_mesh(_open())
    flower = build_flower_mesh(_open(petal_whorl_length_profile=length,
                                     petal_whorl_width_profile=width,
                                     petal_whorl_pitch_profile=pitch))
    for k, (petal, plain) in enumerate(zip(flower.petals, flat.petals, strict=True)):
        x = _place(k)
        assert math.isclose(_length(petal), _length(plain) * length.evaluate(x), rel_tol=1e-9)
        for w, w0 in zip(petal.widths, plain.widths, strict=True):
            assert math.isclose(w, w0 * width.evaluate(x), rel_tol=1e-9, abs_tol=1e-15)
        assert math.isclose(_pitch(petal), PITCH * pitch.evaluate(x), abs_tol=1e-9), k


def test_the_whorl_is_mirror_symmetric_about_the_first_petal():
    flower = build_flower_mesh(_open(
        petal_whorl_length_profile=Ramp([(0.0, 1.0), (1.0, 0.5)]),
        petal_whorl_pitch_profile=Ramp([(0.0, 0.2), (1.0, 1.0)])))
    for k in range(1, COUNT):
        a, b = flower.petals[k], flower.petals[COUNT - k]
        assert math.isclose(_length(a), _length(b), rel_tol=1e-12)
        assert math.isclose(_pitch(a), _pitch(b), abs_tol=1e-12)


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
