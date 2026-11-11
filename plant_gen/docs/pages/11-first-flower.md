<!-- mode: Tutorial -->
# Your first flower

This tutorial builds a flower with the Flower Generator, bakes it into bud to
open stages, and puts those stages on the plant from *Your first plant*. Each
flower on the plant then shows the stage its own openness calls for, so buds,
half-open and open flowers grow side by side.

## 1. Add a flower

1. In the PlantGen tab open **Flower Generator** and click **Add Flower**.
2. Leave the new flower selected. The panel shows **Editing:** and its name.

The flower sits in a collection of its own. The panel edits whichever flower is
selected.

![The Flower Generator panel with a flower selected](flower-generator)

## 2. Shape the petals

1. Open **Petals**.
2. In **Corolla Type** pick **Rotate / Wheel**, a flat open flower. The other
   presets range from **Tubular** and **Bell** to **Cruciform**; each sets how
   the petals fuse and spread.
3. In **Placement** set **Count** to 5.
4. In **Outline** raise **Length** until the petals reach past the disc.

![The Petals sub-panel, with Outline and Placement open](flower-petals)

## 3. Close and open it

1. Open **Bloom** and drag **Bloom** from 1 to 0. The flower folds into a bud.
2. Drag it back to 1.

**Maturity**, under it, scales the whole flower over its life and is separate
from how open it is.

![The Bloom sub-panel](flower-bloom)

## 4. Bake the bloom stages

1. In **Bloom** set **Stages** to 5.
2. Click **Bake Bloom Stages**.

PlantGen writes five copies of the flower, from fully closed to fully open,
into the shared `PlantGen_FlowerAssets` collection, named in that order.

## 5. Put the flowers on the plant

1. Select the plant from *Your first plant*.
2. In **Components**, select its flower component, **Replace Me** unless you
   renamed it.
3. In the component's settings set **Collection** to `PlantGen_FlowerAssets`.

The plant redraws its flowers from the five stages. The Generic perennial
forms its first flower on Day 150, at the tip of the stem, and by Day 220 all
12 tips carry one. From Day 275 they turn to fruit. Each flower shows the
stage that matches how far it has opened.

A flower component's collection is one bloom sequence, so keep one flower's
stages in it. Bake a second flower into a collection of its own.

You have made a flower and grown it on a plant. *Make a nodding flower head* and
*Make a hanging raceme* arrange flowers on the stem, and the reference lists
every Flower Generator control.
