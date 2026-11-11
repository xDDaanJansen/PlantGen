"""Calendar-authoritative phenology guard.

The goldens prove the *age* path is byte-identical with calendars authored onto
the species; this proves the *day* path actually date-gates flowering and
senescence, and that the whole thing stays inert off the day path.

bpy-free and headless, like the other tests. Run:
    python -m plant_gen.tests.calendar
"""

from __future__ import annotations

import dataclasses as dc
import sys

from ..growth.development import build
from ..growth.model import LifecycleState, OrganKind
from ..growth.morphology import (
    Lifespan,
    PhenologyCalendar,
    PhenoWindow,
    generic_perennial,
)
from ..growth.species import meadow_foxtail, rudbeckia_fulgida
from .inflorescence_golden import digest

EMERGE = 90.0  # generic_perennial default; season is day 90..270.


def _cal(bloom=None, sen=None, emergence=EMERGE):
    return dc.replace(generic_perennial(), emergence_day=emergence,
                      calendar=PhenologyCalendar(bloom=bloom, senescence=sen))


def _flowers(plant):
    return len(plant.organs_of_kind(OrganKind.FLOWER))


def _dead(plant):
    return sum(1 for a in plant.axes.values() if a.lifecycle != LifecycleState.ALIVE)


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {detail}")
    return bool(ok)


def run() -> int:
    results = []

    # 1. The age path ignores the calendar -- the golden-freeze guarantee. A
    #    species carries a real calendar and its age-1.0 fixture does not move.
    plain = dc.replace(generic_perennial(), calendar=None)
    withcal = generic_perennial()  # carries a calendar
    same = digest(build(plain, age=1.0)) == digest(build(withcal, age=1.0))
    results.append(_check(
        "1. Age path ignores the calendar (goldens frozen)",
        same, f"generic with vs without calendar at age 1.0: "
              f"{'identical' if same else 'DIFFERENT'}"))

    # 2. Flowering date-gate, isolated: same Day (same growth), bloom open vs shut.
    D = 200.0
    f_open = _flowers(build(_cal(bloom=PhenoWindow(40.0, 60.0)), day=D))   # day 130..190
    f_shut = _flowers(build(_cal(bloom=PhenoWindow(150.0, 60.0)), day=D))  # day 240..
    results.append(_check(
        "2. Flowering opens only once the bloom date is reached",
        f_open > 0 and f_shut == 0,
        f"day {D:.0f}, identical growth: bloom open -> {f_open} flowers, "
        f"bloom future -> {f_shut} flowers"))

    # 3. Senescence date-gate, isolated: same Day, window active vs future.
    Ds = 250.0
    d_act = _dead(build(_cal(sen=PhenoWindow(120.0, 50.0)), day=Ds))   # day 210..260
    d_fut = _dead(build(_cal(sen=PhenoWindow(180.0, 50.0)), day=Ds))   # day 270..
    results.append(_check(
        "3. Senescence advances only within its window",
        d_act > 0 and d_fut == 0,
        f"day {Ds:.0f}, identical growth: window active -> {d_act} dead axes, "
        f"window future -> {d_fut} dead axes"))

    # 4. Emergence Day is the master offset: shift it, the bloom date shifts with
    #    it, so at a fixed Day a later-emerging plant has not opened yet.
    bloom = PhenoWindow(40.0, 60.0)
    early = _flowers(build(_cal(bloom=bloom, emergence=90.0), day=150.0))   # opens day 130
    late = _flowers(build(_cal(bloom=bloom, emergence=150.0), day=150.0))   # opens day 190
    results.append(_check(
        "4. Emergence Day shifts the whole calendar (master offset)",
        early > 0 and late == 0,
        f"day 150: emergence 90 (opens 130) -> {early} flowers, "
        f"emergence 150 (opens 190) -> {late} flowers"))

    # 5. bloom_window_progress: 0 before, rising through, clamped at 1 after.
    p = _cal(bloom=PhenoWindow(40.0, 60.0))   # window day 130..190
    before = p.bloom_window_progress(120.0)
    mid = p.bloom_window_progress(160.0)
    after = p.bloom_window_progress(230.0)
    mono = before == 0.0 and abs(mid - 0.5) < 1e-9 and after == 1.0
    results.append(_check(
        "5. Bloom window progress runs 0 -> 1 across the window",
        mono, f"day 120 -> {before:.2f}, day 160 -> {mid:.2f}, day 230 -> {after:.2f} "
              f"(window day 130..190)"))

    # 6. Off the calendar: a calendar-less species still flowers emergently on
    #    the day path (unchanged from phase 2), and a shipped species with a
    #    calendar is in full flower with no die-back at the default mature Day 270.
    nocal = _flowers(build(plain, day=270.0))
    rud = rudbeckia_fulgida()
    r270_f = _flowers(build(rud, day=270.0))
    r270_d = _dead(build(rud, day=270.0))
    results.append(_check(
        "6. No-calendar day path unchanged; shipped Day 270 pristine",
        nocal > 0 and r270_f > 0 and r270_d == 0,
        f"calendar-less day 270 -> {nocal} flowers; Rudbeckia day 270 -> "
        f"{r270_f} flowers, {r270_d} dead axes (mature, no die-back)"))

    # 7. Fruit: flowers set fruit then drop on the fruit window, isolated at a
    #    fixed Day (same growth). Fruit is data-only, counted from the model.
    fr = dc.replace(generic_perennial(), calendar=dc.replace(
        generic_perennial().calendar, fruit=PhenoWindow(80.0, 60.0)))  # day 170..230

    def fruit_counts(day):
        pl = build(fr, day=day)
        return (len(pl.organs_of_kind(OrganKind.FLOWER)),
                len(pl.organs_of_kind(OrganKind.FRUIT)))

    f_pre, r_pre = fruit_counts(160.0)   # before set
    f_in, r_in = fruit_counts(200.0)     # in window
    f_post, r_post = fruit_counts(240.0)  # after drop
    results.append(_check(
        "7. Flowers set fruit, then drop, on the fruit window",
        f_pre > 0 and r_pre == 0 and f_in == 0 and r_in > 0 and r_post == 0,
        f"day 160 -> {f_pre} flowers/{r_pre} fruit; day 200 -> {f_in}/{r_in}; "
        f"day 240 -> {f_post}/{r_post} (window day 170..230)"))

    # 8. Autumn stage rises 0 -> 1 through the senescence window, and is 0 off
    #    the day path (foliage stays green until a species asks for autumn).
    a = generic_perennial()   # senescence window day 270..330
    a_before = a.autumn_progress(260.0)
    a_mid = a.autumn_progress(300.0)
    a_after = a.autumn_progress(340.0)
    a_age = a.autumn_progress(None)
    results.append(_check(
        "8. Autumn stage rises through the senescence window",
        a_before == 0.0 and abs(a_mid - 0.5) < 1e-9 and a_after == 1.0 and a_age == 0.0,
        f"day 260 -> {a_before:.2f}, day 300 -> {a_mid:.2f}, day 340 -> {a_after:.2f}; "
        f"age path -> {a_age:.2f}"))

    # 9. Multi-season perennial (phase 5): year 0 == single-season, later years
    #    subtly bigger, dormant off-season bare.
    single = dc.replace(generic_perennial(), lifespan=None)
    per = dc.replace(generic_perennial(), lifespan=Lifespan.HERBACEOUS_PERENNIAL,
                     years_to_mature=4, mature_vigour=1.3)

    def height(pl):
        if not pl.phytomer_count:
            return 0.0
        lo, hi = pl.bounds()
        return hi[2] - lo[2]

    yr0_same = digest(build(single, day=270.0)) == digest(build(per, day=270.0))
    h0 = height(build(per, day=270.0))
    h4 = height(build(per, day=1730.0))          # year 4, day-of-year 270
    dormant = build(per, day=340.0).phytomer_count   # past senescence end (330)
    results.append(_check(
        "9. Perennial: year 0 == single season, later years bigger, dormancy bare",
        yr0_same and abs(h4 / h0 - 1.3) < 1e-6 and dormant == 0,
        f"year-0 identical: {yr0_same}; year-4/year-0 size {h4 / h0:.3f} (want 1.3); "
        f"dormant day 340 -> {dormant} phytomers"))

    # 10. Annual: dies whole past its one season, and never regrows.
    ann = dc.replace(generic_perennial(), lifespan=Lifespan.ANNUAL)
    alive0 = all(a.lifecycle == LifecycleState.ALIVE for a in build(ann, day=270.0).axes.values())
    p_dead = build(ann, day=400.0)               # year 1 -- past its one life
    all_dead = p_dead.axes and all(
        a.lifecycle != LifecycleState.ALIVE for a in p_dead.axes.values())
    results.append(_check(
        "10. Annual dies whole past its season, standing (not regrown)",
        alive0 and all_dead and p_dead.phytomer_count > 0,
        f"day 270 all alive: {alive0}; day 400 all dead: {bool(all_dead)}, "
        f"{p_dead.phytomer_count} phytomers standing"))

    # 11. Clump accumulation: a MULTI_STEM perennial gains stems year over year.
    fox = dc.replace(meadow_foxtail(), lifespan=Lifespan.HERBACEOUS_PERENNIAL,
                     years_to_mature=4, mature_stem_factor=1.6)
    s0 = len(build(fox, day=270.0).basal_axis_ids)
    s4 = len(build(fox, day=1730.0).basal_axis_ids)
    results.append(_check(
        "11. Clump accumulates stems year over year (MULTI_STEM)",
        s4 > s0 and s0 == meadow_foxtail().levels[0].whorl,
        f"year 0 {s0} stems (== the stems level's Whorl), year 4 {s4} stems"))

    passed = sum(results)
    print(f"\n  {passed} passed, {len(results) - passed} failed")
    return 1 if passed != len(results) else 0


def main() -> int:
    print("\ncalendar phenology checks (day path)")
    print("-" * 64)
    return run()


if __name__ == "__main__":
    sys.exit(main())
