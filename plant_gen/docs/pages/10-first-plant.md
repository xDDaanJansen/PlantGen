<!-- mode: Tutorial -->
# Your first plant

This tutorial takes you from an empty scene to a grown, leafy, flowering plant
with welded stems. It is one path with no choices. You need one leaf mesh and
one flower mesh. The Leaf Generator and Flower Generator tutorials make them if
you have none.

## 1. Open the panel

In the 3D viewport press `N` to open the sidebar and click the **PlantGen** tab.
The first panel is **Plant Generator**.

## 2. Add a plant

1. Click **Add Plant**.
2. Type a name, `Poppy`, and click **OK**.

PlantGen grows a plant at the 3D cursor from the Generic perennial preset. It
puts everything in a `Poppy_Plant` collection, with two empty folders inside
it, `Poppy_Leafassets` and `Poppy_Flowerassets`. The plant's flower component,
**Replace Me**, draws from the second; it shows nothing until a flower is in
the folder.

![The Plant Generator panel with a plant selected](plant-generator)

## 3. Give it leaves

PlantGen places your leaf mesh. It does not invent the blade.

1. Make or import one leaf, flat in the XY plane with its face up (+Z), its base
   at the origin and its tip along +Y.
2. Drag it into the `Poppy_Leafassets` folder.

The plant redraws with that leaf at every leaf position.

## 4. Give it flowers

1. Make or import one flower, facing +Z with its base at the origin.
2. Drag it into `Poppy_Flowerassets`.

A flower now sits at the tip of the stem and of every branch, 12 in all. They
come from the plant's flower component, **Replace Me** in the **Components**
tree, which Add Plant made to draw from `Poppy_Flowerassets` on every level.
Rename it to what it holds (double-click it in the tree).

## 5. Move through the year

**Day** is the day of the plant's year it is grown to. A new plant starts at
Day 270, fully grown. The Generic perennial comes up on Day 90.

1. Set **Day** to 120. The plant is a 0.08 m shoot with 10 nodes.
2. Set it to 200. It stands 0.54 m with 83 nodes.
3. Set it back to 270, the full-grown 0.61 m plant.

**Live Update** is on, so the plant regrows a moment after each change.

![The Generic perennial on Day 120, Day 200 and Day 270](r-growth-days)

## 6. Weld the stems

1. Open the **Physiology** panel and its **Surface** fold.
2. Click **Weld**.

The weld runs in the background and you can keep working. When it lands, every
stem and branch is one watertight mesh with filleted junctions and UVs.

## 7. Render

Frame the plant with your camera and render. Stems, leaves and flowers are
geometry like any other object.

You have grown a plant from a preset. The how-to guides go further: give it
branches, place leaves with components, vary a setting along the plant with a
map, rig it for a game engine. The explanations cover how the growth works.
