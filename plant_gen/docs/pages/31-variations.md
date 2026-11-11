<!-- mode: How-to -->
# Make a grid of variations and bake it

For a row or grid of plants of the same species, each one a different
individual, to pick a favourite from or to scatter in a scene.

1. Select the plant and open **Plant Generator › Variations**.
2. Pick a **Layout**. **Row** is one row of **Plants** (10 by default). **Grid**
   adds **Rows** (3 by default).
3. Set **Seed Step**, the gap between the plants' seeds. On Seed 0 with Step
   100 the plants grow from seeds 100, 200, 300 and so on. Change it for a fresh
   set.
4. Pick the setting stepped along the row with the button beside **Along the
   Row**. It starts at the plant's own value; set **Step**, the amount added at
   each next plant. A new plant steps **Day** back by 18 from its own Day, so
   the row runs from the full-grown plant at Day 270 back to a young shoot at
   Day 108. In a Grid, **Across Rows** steps a second setting from row to row;
   leave it empty and the rows differ only by seed.
5. Click **Build 10 Plants**. The copies appear in a row to the right of the
   plant, in a collection of their own.

**Auto Spacing** spaces the copies by the widest one's footprint plus a
quarter. Turn it off to set **Spacing X** and **Spacing Y** yourself.
**Labels** writes each copy's stepped values and seed under it. **Weld** welds
every copy as it is built, which is slow; off, the copies show the quick
unwelded preview.

![The Variations fold](variations-fold)

The copies are rebuilt from the source plant each time, never edited. Edit the
source and click **Rebuild**. Selecting a copy shows the source's settings.
**Clear** removes the copies.

## Bake

Click **Bake to Meshes**. Every copy becomes one plain mesh object in its own
collection: welded, with its leaves, flowers and fruit made real geometry, and
no modifiers. Bake welds even when **Weld** is off. The live copies are hidden,
not deleted, so Rebuild and Clear still work on them.

## Export to Unreal

Click **Export Nanite Assembly** in the same fold to write the plant and all its
copies as one USD file, so they share their leaf and flower meshes in Unreal.
*Export a Nanite Assembly to Unreal* has the steps.
