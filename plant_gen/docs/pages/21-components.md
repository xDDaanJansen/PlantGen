<!-- mode: How-to -->
# Place organs with components

For putting leaves, bracts, flowers, buds and fruit on a plant, choosing which
parts of the plant each kind goes on, and mixing several meshes. A component is
one collection of meshes of one organ type, placed on one or more levels.

You supply the meshes. A component places them; it does not model them.

## Add a component

1. In **Components**, select where it goes in the tree: the **Plant** row for
   every level, or one level.
2. Click **+**, then **New Component**, and pick the type: Leaf, Bract, Flower,
   Bud or Fruit.
3. In its settings set **Collection** to the collection holding its meshes.

Each point the component takes gets one mesh from the collection, picked at
random. **Seed** reshuffles that pick for this component alone.

![A leaf component selected in the Components tree](components-tree)

## Put it on other levels

Select a level and pick the component from **+**, or select the component and
use **Move To** in the down-arrow menu beside the tree. A component on several levels is one
component: its settings are shared. Components on no level are listed under
**Not Placed** and draw nothing.

## Choose where its instances go

**Placement** lists where instances go, and several can be on at once:

- **On organs** uses the organ points growth makes, leaves at nodes and flowers
  on the inflorescence.
- **Nodes**, **Internodes**, **Terminal** and **Intersections** place along the
  stems at those positions.
- **Random** scatters a set count over the stems.
- **Basal** makes a rosette of **Basal Count** leaves (8 by default) at the foot
  of every stem.

## Keep it off part of the plant

Masks decide which points a component takes.

1. In the component's **Masks**, click **+** and pick an attribute, for example
   **Height**.
2. Set **Min** and **Max**. A point outside the band goes to another component
   of the same type that takes it. A point no component takes shows nothing.

**Classify View** plots the component's points on two attributes, **X** and
**Y**, and lets you drag its mask window around the ones it should take.

## Mix two components on the same points

Give both the same type and placement. Where both may take a point, one is
picked at random by **Density**: two at 1 split the points evenly, 2 against 1
splits them 2:1. **Prune** deletes a share of a component's own instances at
random: 0.5 deletes about half, 1 deletes all.

## Leaf and bract extras

Leaf and bract components also have **Sky** (turn each face toward the sky),
**Emerging** (how a young leaf is curled before it opens), a **Petiole** fold
for a stalk, and **Collision**, which turns leaves out of each other's way.

To choose between meshes inside one collection by age, height or shade instead
of at random, see *Choose between mesh variants with Traits*.
