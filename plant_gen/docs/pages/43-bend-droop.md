<!-- mode: Explanation -->
# Bend and droop under load

A PlantGen stem bends because it carries weight, not because a slider tells it
to. Once the plant has grown, a fixed pass treats every stem and branch as a
beam fixed at its base and bends it under the weight it holds. That pass never
adds or removes a part. It only changes which way each part points.

## What a stem carries

At each node the load is everything beyond it: the stem's own wood above, and
every leaf, flower and fruit that stem and its branches carry. A load far out to
the side turns the node harder than the same load close in.

The organs' weights come from their components:

- A leaf weighs its component's **Weight** times its size squared, since a leaf
  is a plate.
- A flower weighs its component's **Weight** times its size cubed, since a head
  is a lump. A head twice as big is eight times as heavy.
- A fruit weighs the fruit component's **Weight** times its size cubed. A
  fruiting branch bows more than the same branch in flower.

Weight scales with the organ's own growth, so a bud that has not opened yet
hardly loads its stem.

## What resists it

A stem's resistance is its **Stiffness** times its cross-section's resistance
to bending, which grows with the fourth power of its radius. A stem twice as
thick resists 16 times the bend. That is why a stem stays upright at its thick
base and nods at its thin tip under the same head, and why the stem's taper
under **Width** shapes the bend as much as Stiffness does.

**Stiffness** is set per level and runs on a log scale: each 0.1 doubles it,
so 0 to 1 spans a thousand-fold range. **Physiology › Stiffness › Preset**
fills every level from a stem type, from Herbaceous to Woody. A bend can't go
tighter than about 15 times the stem's radius, the point where a real tube
kinks, so even a very soft stem curves rather than folds.

![Physiology › Stiffness](stiffness-fold)

## The two numbers trade off

Weight and Stiffness only mean something together. Measured on a new plant at
Day 270 with a single flower at the stem's tip, the tip's angle from vertical:

| Stiffness | Weight 0.05 | Weight 0.1 | Weight 0.2 |
|---|---|---|---|
| 0.5 | 1° | 3° | 5° |
| 0.2 | 12° | 22° | 42° |
| 0.1 | 23° | 44° | 85° |

![The head at Stiffness 0.5, 0.2 and 0.1, Weight 0.2](r-nod-stiffness)

In this range the angle about doubles when Stiffness drops by 0.1, and about
doubles when Weight doubles. Every flower a stem's branches carry loads the
stem too: with a flower at all 12 tips of the same plant, Weight 0.01 at
Stiffness 0.1 already turns the stem's tip 119°. On a small, light plant there
is little mass, so the bend stays small until Stiffness is low.

## Flower stalks

A flower's own stalk, the pedicel, sags by the same beam rule: the flower's
weight at its end, against the stalk's length and thickness. A long thin stalk
under a heavy flower hangs; a short thick one holds the flower out.

The pedicel's **Angle** and **Bend** are not weights. They set the angle the
stalk leaves at and the curve it grows with, whatever the flower weighs, and
the sag adds to them. Use them when flowers should point a certain way on any
plant, like the bells of a hanging raceme. A flower component's
**Orientation** (Erect, Horizontal, Pendulous and others), with its
**Strength**, aims the inflorescence's flowers the same way.

## Gravitropism is growth, not bend

**Gravitropism** in Physiology › Tropisms acts while the plant grows: each new
node steers back toward upright. It changes where the stem goes, and the bend
then acts on that. **Autotropism** straightens older stem tissue after growth,
so a mature stem is tidy while its youngest nodes keep their nod.

## When a bend seems dead

It is almost always one of three things. The stem is too stiff for the load.
The organ is too light, because its component's Weight is low or its Size is
small and Weight goes with the square or cube of Size. Or the load sits near
the stem's thick base, where the fourth-power radius term wins. The bend
debug views in Debug Tools show the load, stiffness and moment at every node.
