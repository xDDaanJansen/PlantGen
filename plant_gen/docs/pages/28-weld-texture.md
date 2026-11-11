<!-- mode: How-to -->
# Weld and texture the plant

For turning the stems into one clean mesh with UVs, giving each level its own
bark, and setting up the leaf and flower materials.

## Weld the stems

1. Open **Physiology › Surface**. **Weld Junctions** is on by default.
2. Click **Weld**.

The weld runs in the background while you keep working, and the button reads
**Welding...** until it lands. Then every stem and branch is one watertight
mesh, each junction filleted, with UVs. A junction that cannot be built, such
as a branch too short to clear its fillet, stays a capped tube inside its
parent. The fold counts the built and the capped junctions. **Weld
Range** sets how round the fillet is, in branch radii, 0.5 by default; 0 leaves
a hard crease. **Smoothness** sets how densely rings sit along the stems; at
the default 4 the faces are close to square.

**Light Stalks** is for a plant with thousands of flowers. It builds pedicels
with four sides and gives stems too thin to show bark fewer rings on their
straight runs, while branch welds and bends keep their density. Laburnum's stems
came to 663,705 vertices with it on, its pedicels 8 vertices each.

![Physiology › Surface after a weld](surface-fold)

A change that regrows the plant drops back to the unwelded preview. Weld again
when the plant is settled. A plant you welded before at the same settings shows
its weld again without rebuilding it.

## Give each level its own stem material

1. Open **Texturing › Materials**. The list holds a row per branching level,
   then every material the leaves and flowers wear.
2. Select a level's row and pick a material, or click **New** for a PlantGen
   stem material (the Green Stem preset, untextured).
3. Pick a **Preset** for its look, and open **Textures & Settings** for its
   image slots and shader values.

A level without a material of its own wears its parent level's, shown greyed in
the list.

![Texturing › Materials with a level row selected](texturing-materials)

## Edit the leaf and flower materials

Select a leaf or flower material in the same list. Its preset and **Textures &
Settings** appear below. A material is often shared by several meshes, and an
edit changes every mesh that wears it.

## Set the texture scale

**Texturing › UV › Size** is the world size of one texture tile, 0.025 m by
default, the same on every stem, branch and stalk, so a thin stalk shows the
same size squares as the thickest stem. The checker button beside it shows the
stems with a checker texture instead of their material.

![A new plant's welded stems under the UV Checker](r-uv-stems)

The cells stay square along the stem and the branches, and the checker runs on
across each joint without a seam in the middle of the fillet.

![A branch joint, side-on and three-quarter](r-uv-weld)

## How the stems unwrap

Each stem and each branch is one UV island, unwrapped round its circumference
and along its length, tiling the texture. The weld cuts a hole in the stem's
island where the branch leaves; it doesn't move the stem's UVs. The branch's
island starts at the joint: its fillet rings flare out to meet the hole's edge,
then the plain branch runs on.

![The joint's UV layout over the tiled checker: the stem's island with its hole (left), the branch's island with its fillet (right)](r-uv-layout)

Because the islands tile one texture, they overlap in UV space; in Blender's UV
editor every stem and branch lies across the same tiles. Use a tiling bark
texture, not a baked atlas, on stem materials.

## Read the vertex colours

**Texturing › Vertex Colors** shows the four channels every exported mesh
carries, for a game engine's foliage shader:

- **Red** rises from 0 at each leaf's base to 1 at its tip, flat on stems and
  flowers.
- **Green** is one random value per leaf or flower.
- **Blue** is one random value for the whole plant.
- **Alpha** is height up the plant, 0 at the root to 1 at the top.

Click a channel's button to show the plant coloured by it in the viewport;
click it again to go back. **Vertex Colors** shows all four at once. The view
is viewport-only; the colours themselves are always on the mesh and export
with it.
