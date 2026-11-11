<!-- mode: Explanation -->
# Seeds and the same plant twice

A PlantGen plant is random in its details and exact as a whole. The same
settings and the same seed grow the same plant, bit for bit, every time. A
different seed grows a sibling: the same species, a different individual.

## What a seed changes

Every random choice in the plant, such as a branch's exact angle, how long one
stem grows, or which mesh a leaf takes, is drawn from the seeds. The settings
set the ranges; the seed picks inside them.

On the Generic perennial at Day 270, five master seeds give five plants with
the same 12 axes and 89 nodes, standing between 0.47 m and 0.61 m tall. The
structure is the species. The lean, the lengths and the exact angles are the
individual.

![The Generic perennial on seeds 0 to 4](r-seeds)

## Master Seed and the part seeds

The **Master Seed** in Plant Generator reshuffles everything. Every level and
every component also has its own **Seed**, mixed with the master:

- A level's Seed reshuffles that level's draws: its branches' angles, lengths
  and which nodes fork.
- A component's Seed reshuffles which mesh each point takes and its jitter in
  size and rotation.

At 0, a level or component follows the master alone, and that is the default.
Changing one part's seed leaves the others' draws alone. Setting the Generic
plant's branch level to Seed 7 changes every branch, and its stem grows exactly
as before. The stem still moves, by 0.4 mm at most, because the reshuffled
branches load it slightly differently.

## Why draws are tied to parts, not order

Each draw is keyed on the part it belongs to, never on the order the code
happens to reach it. Two consequences follow.

First, adding a component doesn't reshuffle what's already there. A leaf placed
on a stem keeps its look because its draw comes from where it sits on the
plant, not from its number in a list. That's what lets a timelapse grow a leaf
in without the leaves around it flickering.

Second, settings that change when parts are made would deal every later part a
different hand. So timing controls like **Emergence Delay** don't change when
parts are made; they change when they're shown. The full-grown plant stays the
same whatever the delay.

## Practical use

- To pick between individuals, use **Variations**: a row of the same plant on
  seeds 100, 200, 300 and so on, side by side.
- To keep a plant you like while trying other settings, leave its seeds alone.
  Seeds are saved with the plant and in its preset.
- To vary only one part, change that level's or component's Seed instead of the
  Master Seed.
