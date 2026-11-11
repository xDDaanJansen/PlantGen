<!-- mode: How-to -->
# Choose between mesh variants with Traits

For a leaf, bract, bud or fruit component with several meshes in its
collection, when the choice should follow the plant: young leaves take the small
pale mesh, shaded leaves the dark one, the tip the narrow one. Without Traits
each point picks a mesh at random.

Flower components have no Traits. Their meshes are bloom stages, not variants.

## Start a sheet

1. Select the component in **Components** and open its **Traits** fold.
2. Click **Add Property**. The sheet appears: a row per mesh in the collection,
   a column for the property.
3. Pick the column's property in the dropdown above the sheet: **Age**,
   **Height**, **Shade**, or any other attribute, custom attribute or class.

**Weights Only** starts a sheet with no columns, to set how common each mesh is
and nothing else.

![A Traits sheet with an Age column](traits-sheet)

## Fill it in

Each row has a **Weight** and one cell per column.

- **Weight** is how common the mesh is. A mesh at 2 is picked twice as often as
  one at 1; at 0 it is never picked, unless no mesh suits a point.
- A **cell** runs from -1 to 1 and scales the mesh's weight at each point by the
  property's value there, from 0 to 1. At +1 the weight is multiplied by twice
  the value: 0 where the property is 0, 2 where it is 1. At -1 it is multiplied
  by twice the remainder, 2 where the property is 0. At 0 the property makes no
  difference. Values between blend.

A worked case: two leaf meshes, the young one at -1 on Age and the old one at
+1. At a point of Age 0.25 the old mesh is picked with chance 0.25 and the young
one with 0.75. With several columns, a mesh's chances multiply.

The pick stays random, so neighbouring points still mix. A sheet with every
weight equal and every cell 0 picks exactly as no sheet does.

## Keep it in step with the collection

When you add or remove meshes in the collection, the sheet shows **The
collection changed**. Click **Update Rows**: new meshes join with weight 1, and
the rows of removed ones go. The bin button clears the sheet.

The sheet belongs to the component, so the same mesh can lean differently in
another component or plant. Presets save it.
