<!-- mode: Tutorial -->
# Your first leaf

This tutorial makes a toothed, drooping leaf with the Leaf Generator and puts
it on the plant from *Your first plant*. The generator builds the blade from an
outline curve, so you need no texture.

## 1. Switch to Generate

1. In the PlantGen tab open **Leaf Generator**.
2. Set **Mode** to **Generate**.

![The Leaf Generator panel in Generate mode](leaf-generator)

**Cutout** is the other mode. It cuts leaves from a texture atlas, one leaf per
opaque region of the image.

![The Leaf Generator in Cutout mode, with an atlas picked](leaf-cutout)

## 2. Set the outline

1. Pick a **Leaf Shape**. It loads its outline into the **Profile Curve** and
   sets **Width** from that shape's proportion. The reference draws all 25 under
   [Leaf Generator › Leaf Shape](#ui-leaf-generator-leaf-shape), and every margin
   under [Margin](#ui-leaf-generator-margin-margin).
2. Set **Length** to 0.12 m.

The profile curve is the half outline, base on the left, tip on the right. Drag
its points to change the blade's silhouette.

## 3. Give it teeth and droop

1. Open **Margin** and pick **Serrate**. Set **Teeth** to 14.
2. Open **Deform** and set **Gravity** to 0.4. The tip sags under the blade's own
   weight.
3. Set **Cup** to 0.2. The two edges lift into a shallow gutter.

![Profile Curve and Margin open, Margin set to Serrate with 14 teeth](leaf-profile-margin)

## 4. Make the leaf

Click **Generate Leaf**. A new leaf object appears, flat along +Y from the
origin, which is the layout the plant expects. While it is selected the panel
edits it live.

![The panel editing a generated leaf, its Material fold open](leaf-edit)

## 5. Put it on the plant

1. Drag the new leaf into the plant's `Poppy_Leafassets` folder.
2. Remove the old leaf from that folder.

Every leaf position picks one mesh from the folder at random, so with both in
it the plant would mix them.

You have made a leaf and grown it on a plant. To mix several leaf meshes and
choose between them by age or shade, see *Choose between mesh variants with
Traits*. The leaf's material is in the generator's **Material** fold and in
**Texturing**.
