"""Tests for the rig's Unreal Dynamic Wind classification, headless.

Measured in UE 5.7 on Rudbeckia: the branch path
shakes a stem of a few bones, so stems take the smooth trunk path -- but its
turn adds up along a line of bones, so only to a depth.

1. A perennial's stems are all Trunk, side stems included.
2. A long line is Trunk only for its first TRUNK_LINE_BONES bones; the rest ride
   their parent at Influence 0, as the branch path jitters at any other value.
   A run past that depth shorter than BRANCH_MIN_BONES stays Trunk.
3. An organ's Influence is a single value (a chain of one takes a dual
   influence's Min), scaled, and comes from the stem bone it hangs from.
4. GustAttenuation reads the level 0 stems alone.
5. Every bone is in a group, and the import file names every joint.

    python -m plant_gen.tests.wind_groups
"""

from __future__ import annotations

import numpy as np

from ..geometry import rig_layout as rl


def _bone(name, parent, group="Level 0", flex=1.0, organ=False):
    return rl.Bone(name, np.zeros(3), np.array([0.0, 0.0, 0.1]), parent, False, group,
                   flex=flex, organ=organ)


def _line(prefix, parent, count, group, flex=1.0):
    bones = []
    for k in range(count):
        bones.append(_bone(f"{prefix}{k}", parent, group, flex))
        parent = bones[-1].name
    return bones


def _rudbeckia_like():
    """A stem of 5 bones with a side stem of 2 and a flower on each."""
    bones = [_bone(rl.ROOT, None, "Root")]
    bones += _line("s", rl.ROOT, 5, "Level 0", flex=2.0)
    bones += _line("b", "s3", 2, "Level 1", flex=4.0)
    bones += [_bone("flower_1", "s4", "Flowers", organ=True),
              _bone("flower_2", "b1", "Flowers", organ=True)]
    return bones


def test_a_perennials_stems_are_all_trunk():
    bones = _rudbeckia_like()
    wind = rl.wind_groups(bones)
    names = [g["name"] for g in wind["groups"]]
    assert not any(n.startswith("Level") for n in names), names
    trunk = names.index(rl.TRUNK_GROUP)
    assert wind["groups"][trunk]["settings"]["bIsTrunkGroup"] == 1
    for b in bones:
        if not b.organ:
            assert wind["bones"][b.name] == trunk, b.name


def test_a_long_line_is_trunk_only_to_a_depth():
    count = rl.TRUNK_LINE_BONES + 10
    bones = [_bone(rl.ROOT, None, "Root")] + _line("s", rl.ROOT, 4, "Level 0") \
        + _line("t", "s3", count, "Level 1")
    wind = rl.wind_groups(bones)
    names = [g["name"] for g in wind["groups"]]
    trunk = names.index(rl.TRUNK_GROUP)
    line = [b.name for b in bones if b.name != rl.ROOT]
    on_trunk = [n for n in line if wind["bones"][n] == trunk]
    # the line's bones past the cap ride, whatever their level
    assert len(on_trunk) == rl.TRUNK_LINE_BONES, len(on_trunk)
    assert on_trunk == line[:rl.TRUNK_LINE_BONES]


def test_a_stem_past_the_depth_has_no_wind_of_its_own():
    tip = f"s{rl.TRUNK_LINE_BONES - 1}"
    bones = [_bone(rl.ROOT, None, "Root")] + _line("s", rl.ROOT, rl.TRUNK_LINE_BONES, "Level 0") \
        + _line("t", tip, rl.BRANCH_MIN_BONES, "Level 1") \
        + _line("u", tip, rl.BRANCH_MIN_BONES, "Level 2")
    wind = rl.wind_groups(bones)
    names = [g["name"] for g in wind["groups"]]
    assert not any(n.startswith("Level") for n in names), names
    rider = wind["groups"][names.index(rl.RIDER_GROUP)]["settings"]
    assert rider["MinInfluence"] == 0.0 and rider["MaxInfluence"] == 0.0, rider
    assert rider["bIsTrunkGroup"] == 0


def test_a_short_chain_past_the_depth_stays_on_the_trunk_path():
    tip = f"s{rl.TRUNK_LINE_BONES - 1}"
    bones = [_bone(rl.ROOT, None, "Root")] + _line("s", rl.ROOT, rl.TRUNK_LINE_BONES, "Level 0") \
        + _line("long", tip, rl.BRANCH_MIN_BONES, "Level 1") \
        + _line("twig", tip, rl.BRANCH_MIN_BONES - 1, "Level 2")
    wind = rl.wind_groups(bones)
    names = [g["name"] for g in wind["groups"]]
    trunk, rider = names.index(rl.TRUNK_GROUP), names.index(rl.RIDER_GROUP)
    for b in bones:
        if b.name.startswith("twig"):
            assert wind["bones"][b.name] == trunk, b.name
        if b.name.startswith("long"):
            assert wind["bones"][b.name] == rider, b.name


def test_a_side_twig_is_a_chain_of_its_own():
    parent = {f"a{k}": f"a{k - 1}" if k else None for k in range(9)}
    parent |= {f"b{k}": f"b{k - 1}" if k else "a2" for k in range(3)}
    group = dict.fromkeys(parent, "Level 1")
    # a0..a8 run on; the twig is 3 bones, however long the line it leaves
    chains = rl.branch_chains(parent, group)
    assert sorted(len(c) for c in chains) == [3, 9], chains
    assert {"b0", "b1", "b2"} in [set(c) for c in chains]


def test_an_organ_takes_one_scaled_influence_from_its_stem():
    wind = rl.wind_groups(_rudbeckia_like())
    groups = {g["name"]: g["settings"] for g in wind["groups"]}
    # flower_1 hangs from a bone half as free as the limpest, flower_2 from the limpest
    low, high = groups[f"{rl.ORGAN_GROUP} 3"], groups[f"{rl.ORGAN_GROUP} 4"]
    for g in (low, high):
        assert g["bUseDualInfluence"] == 0 and g["bIsTrunkGroup"] == 0
    assert low["Influence"] == round(rl.BRANCH_WIND_SCALE * 0.5, 3), low
    assert high["Influence"] == round(rl.BRANCH_WIND_SCALE * 1.0, 3), high


def test_gust_attenuation_reads_the_level_0_stems():
    wind = rl.wind_groups(_rudbeckia_like())
    # stems flex 2 of the limpest bone's 4 (a side stem, in Trunk but not level 0)
    assert wind["gust_attenuation"] == 0.5, wind["gust_attenuation"]


def test_a_long_trunk_line_shares_its_gust():
    # the root + 15 stem bones = 16 trunk bones, stems at half the limpest bone's flex
    bones = [_bone(rl.ROOT, None, "Root")] + _line("s", rl.ROOT, 15, "Level 0", flex=2.0) \
        + [_bone("b0", "s14", "Level 1", flex=4.0)]
    wind = rl.wind_groups(bones)
    want = 1.0 - 0.5 * rl.TRUNK_SWAY_BONES / 17
    assert wind["gust_attenuation"] == round(want, 3), wind["gust_attenuation"]


def test_every_joint_is_named_in_the_import_file():
    bones = _rudbeckia_like()
    wind = rl.wind_groups(bones)
    data = rl.wind_json(wind, [b.name for b in bones])
    assert len(data["Joints"]) == len(bones)
    assert {j["JointName"] for j in data["Joints"]} == {b.name for b in bones}
    assert max(j["SimulationGroupIndex"] for j in data["Joints"]) < len(data["SimulationGroups"])


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
