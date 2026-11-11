<!-- mode: Explanation -->
# How a plant grows

PlantGen doesn't draw a plant shape and decorate it. It grows the plant, one
node at a time, the way a shoot builds itself, and then lets the grown plant
bend under its own weight. Knowing the two halves explains most of what the
controls do and why some of them only show at certain Days.

## The phytomer

A stem is a stack of repeated units botanists call phytomers: a node, the
internode below it, the leaf at the node, and a bud in that leaf's axil. Every
node PlantGen grows is one of these. There is no way to grow a bare node, so
leaves and buds come with the structure rather than being scattered over it
afterwards.

Each growth step does the same few things in order. Every growing tip adds a
node. Buds that are due break into branches, laid out by the next level's
phyllotaxis. Axes that have reached their length end, and an ending tip
flowers when a flower component is placed there with Terminal attachment;
otherwise it stops or hands on to a relay or a fork, as its level's Growth
Pattern says. A branch is an axis like the stem, with its own tip and buds, so
the same step builds every level.

The stems level's **Steps** sets the step count. The Generic perennial grows in
26 steps.

## The second half: physical form

Once growth is done, a fixed pass shapes the grown plant without adding or
removing anything. It thickens every stem from what it carries, bends the
stems and branches under their own weight and the weight of their leaves,
flowers and fruit, droops the flower stalks, turns flowers and leaves toward
the sun, and marks what is dying back in autumn. The stiffness and weight
controls act here (*Bend and droop under load*). Because this pass never
changes what grows where, a bend can't make a branch appear or vanish.

## The calendar

**Day** is a real day of the year, and the plant's development runs on it.
The plant comes up on **Emergence Day** and is full grown **Season Length**
days later. Between the two, development is spread evenly over the days, so
the Generic perennial (Emergence Day 90, Season Length 180) grows a step about
every 6.9 days:

| Day | Progress | Nodes | Height |
|-----|----------|-------|--------|
| 90  | 0        | 0     | 0      |
| 120 | 0.17     | 10    | 0.08 m |
| 150 | 0.33     | 36    | 0.26 m |
| 200 | 0.61     | 83    | 0.54 m |
| 270 | 1        | 89    | 0.61 m |

![The Generic perennial on Day 120, Day 200 and Day 270](r-growth-days)

Between two steps the newest node lengthens smoothly, so scrubbing Day shows a
growing tip rather than nodes popping in.

Every step reads the calendar at its own Day, never at the Day the plant is
grown to. So a plant grown to Day 200 is the Day 150 plant grown further. At
Day 150 the Generic plant has 9 axes, and every one of them is still there at
Day 200 with at least as many nodes. A timelapse depends on this.

The calendar's windows are counted in days after emergence. The Phenology
Calendar edits them as bands:

![The Phenology Calendar](calendar)

- **Bloom** opens 55 days after emergence and lasts 75 by default. Flowering
  starts on the bloom date, or later if the plant is not yet big enough to
  flower. The Generic perennial's bloom window opens on Day 145; its first
  flower forms on Day 150 and its last tip flowers by Day 220.
- **Fruit** opens at 185 days and lasts 95, then the fruit drops.
- **Autumn** opens at 180 days and lasts 60: leaves colour and the plant dies
  back.

Days past the first season are where **Lifespan** matters. A Single season
plant stays mature. An Annual dies standing at season end. A Herbaceous
perennial dies back to the ground in winter and comes up again next spring a
little bigger. The Generic perennial is one: on Day 400 it shows nothing, and
on Day 635, a year after its first full growth, it stands 0.64 m, 5% taller
than in its first year.

## Why timing controls shift display, not growth

Every organ's random variation (a leaf's size, its yaw, a flower's pitch) is
dealt from the order parts are made in. A control that changed when parts are
made would re-deal that variation over the whole plant. So **Emergence Delay**,
which makes branches come up later, grows the plant exactly as before and holds
the late parts back from view until their day. The full-grown plant is the same
either way, and each earlier Day is a part of it.
