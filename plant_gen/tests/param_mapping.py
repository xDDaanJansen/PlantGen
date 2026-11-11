"""Parameter mapping (growth/plant_maps.py): the per-plant map against closed
forms, and the registry against the real property names and growth targets.

    python -m plant_gen.tests.param_mapping
"""

from __future__ import annotations

import re
from pathlib import Path

from ..growth import maps as growth_maps
from ..growth import plant_maps as pm
from ..growth.curves import Curve

_PROPS = (Path(__file__).resolve().parents[1] / "blender" / "properties.py").read_text(
    encoding="utf-8")


def _declared(name: str) -> bool:
    """Whether properties.py declares a property of that name."""
    return re.search(rf"^\s+{re.escape(name)}\s*[:=]", _PROPS, re.M) is not None or \
        f'"{name}"' in _PROPS


def test_no_map_is_identity():
    # No enabled map: the value itself, untouched (the byte-identical path).
    for value in (0.3, 7, 0.0):
        assert pm.apply(value, [], 1, 2) is value
        off = pm.PlantMap(param="x", preset="RANGE", range=0.5, enabled=False)
        assert pm.apply(value, [off], 1, 2) is value


def test_random_range_multiplies_within_range():
    # value * (1 + range * (2u - 1)), u the per-plant draw.
    m = pm.PlantMap(param="steps", preset="RANGE", range=0.25)
    u = pm.plant_draw(3, 4, "steps", 0)
    got = pm.apply(10.0, [m], 3, 4)
    assert abs(got - 10.0 * (1.0 + 0.25 * (2.0 * u - 1.0))) < 1e-12
    for seed in range(50):
        v = pm.apply(10.0, [pm.PlantMap(param="steps", preset="RANGE", range=0.25,
                                        seed=seed)], 3, 4)
        assert 7.5 - 1e-9 <= v <= 12.5 + 1e-9


def test_angle_range_adds():
    # An angle adds: value + angle * (2u - 1) -- so a 0 base still varies.
    m = pm.PlantMap(param="gen_pitch", preset="RANGE", angle=15.0, additive=True)
    u = pm.plant_draw(0, 0, "gen_pitch", 0)
    assert abs(pm.apply(0.0, [m], 0, 0) - 15.0 * (2.0 * u - 1.0)) < 1e-12


def test_curve_map_and_amount():
    # A curve map: value * (1 + amount * (curve(u) - 1)); flat at 1 is a no-op.
    flat = pm.PlantMap(param="p", curve=Curve.constant(1.0))
    assert pm.apply(2.0, [flat], 5, 6) == 2.0
    double = pm.PlantMap(param="p", curve=Curve.constant(2.0), amount=0.5)
    assert abs(pm.apply(2.0, [double], 5, 6) - 3.0) < 1e-12
    # an angle's curve: offset = amount * angle * (2 * curve(u) - 1); middle adds none
    mid = pm.PlantMap(param="yaw", curve=Curve.constant(0.5), angle=30.0, additive=True)
    assert abs(pm.apply(4.0, [mid], 5, 6) - 4.0) < 1e-12


def test_integer_rounds_and_draws_are_deterministic():
    m = pm.PlantMap(param="whorl", preset="RANGE", range=0.2)
    a = pm.apply(5, [m], 9, 1, integer=True)
    assert isinstance(a, int) and 4 <= a <= 6
    assert a == pm.apply(5, [m], 9, 1, integer=True)
    # the parameter, the owner seed and the map seed all reshuffle the draw
    base = pm.plant_draw(9, 1, "whorl", 0)
    assert base != pm.plant_draw(9, 1, "steps", 0)
    assert base != pm.plant_draw(9, 2, "whorl", 0)
    assert base != pm.plant_draw(9, 1, "whorl", 1)
    assert 0.0 <= base < 1.0


def test_hooks_point_at_real_targets():
    # Every per-element hook is a target its evaluator knows.
    for target in pm.LEVEL_HOOKS.values():
        assert target in growth_maps.LEVEL_TARGETS, target
    for target in pm.PLANT_HOOKS.values():
        assert target in growth_maps.MAP_TARGETS, target
    assert pm.ORIENTATION_TARGETS <= set(pm.INSTANCE_HOOKS.values())
    assert pm.ROTATION_TARGETS <= set(pm.INSTANCE_HOOKS.values())
    assert set(pm.FLOWER_HOOKS.values()) == set(growth_maps.FLOWER_TARGETS)
    # and every hooked parameter is a real property (or a row of its own)
    for param in list(pm.LEVEL_HOOKS) + list(pm.PLANT_HOOKS) + list(pm.FLOWER_HOOKS):
        assert param in pm.PSEUDO_PARAMS or _declared(param), param


def test_every_variance_pairing_is_real():
    # Every +/- the registry knows is a declared property, and so is the
    # parameter it varies (or it is a row of its own).
    tables = (pm.VARIANCE_LEVEL, pm.VARIANCE_PLANT, pm.VARIANCE_FLOWER, pm.VARIANCE_LEAF)
    for table in tables:
        for param, var in table.items():
            assert _declared(var), var
            assert param in pm.PSEUDO_PARAMS or _declared(param), param


def test_every_declared_variance_is_known():
    # No +/- property is left behind as a stray slider: each one is in a
    # registry table (or retired with the plant-level leaf fields).
    declared = set(re.findall(r"^\s+([a-z_]+_(?:variation|variance))\s*[:=]", _PROPS, re.M))
    missing = sorted(declared - set(pm.VARIANCE_PROPS))
    assert not missing, missing


_CURVES = (Path(__file__).resolve().parents[1] / "blender" / "curves.py").read_text(
    encoding="utf-8")
_SPEC_KEYS = set(re.findall(r'CurveSpec\("([a-z_]+)"', _CURVES))


def test_curve_rows_point_at_real_curves():
    # Every profile curve the Mapping list shows is a real curve on a real
    # parameter: a CurveSpec (the plant's / a flower component's copy) or a
    # component's own size / petiole taper curve.
    tables = (pm.CURVE_PLANT, pm.CURVE_FLOWER, pm.CURVE_ORGAN, pm.CURVE_LEAF)
    for table in tables:
        for param, (key, (short, caption)) in table.items():
            assert key in _SPEC_KEYS or key in ("size", "petiole_width"), key
            assert _declared(param), param
            assert short and caption
    # the shapes with no neutral are never a map row
    shapes = {"head_dome", "flowering_onset", "senescence"}
    used = {key for table in tables for key, _axis in table.values()}
    assert not used & shapes, used & shapes
    # a non-multiplier row has its neutral: Young Growth's straight default line,
    # the profile's own default (so a default plant shows it unmapped)
    from ..growth.morphology import Inflorescence
    default = Inflorescence().young_growth_profile
    for x in (0.0, 0.25, 0.5, 1.0):
        assert abs(pm.curve_neutral_y("young_growth", x) - float(default(x))) < 1e-9
        assert pm.curve_neutral_y("pedicel_length", x) == 1.0
    assert pm.curve_table(pm.OWNER_LEVEL) == {}
    assert pm.curve_table(pm.OWNER_COMPONENT, "flower")["size"][0] == "flower_size"
    assert "pedicel_width" not in pm.FLOWERING_CURVES     # along one stalk, every kind


def test_evaluated_per():
    assert pm.evaluated_per(pm.OWNER_LEVEL, "start_pitch", stems_level=True) == "stem"
    assert pm.evaluated_per(pm.OWNER_LEVEL, "start_pitch") == "branch"
    assert pm.evaluated_per(pm.OWNER_LEVEL, "length_scale_per_step") == "node"
    assert pm.evaluated_per(pm.OWNER_LEVEL, "whorl") == "plant"
    assert pm.evaluated_per(pm.OWNER_COMPONENT, "pedicel_length",
                            component_type="flower") == "flower"
    assert pm.evaluated_per(pm.OWNER_COMPONENT, "size", component_type="leaf") == "instance"
    assert pm.evaluated_per(pm.OWNER_PLANT, "gravitropism") == "axis"


# --- plain runner (no pytest needed) -----------------------------------------------

def _run() -> int:
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
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
