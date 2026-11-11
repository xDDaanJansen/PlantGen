<!-- mode: How-to -->
# Shape the branching with levels

For changing how many stems a plant has, which branches it grows, at what angle
and how long. The numbers below are measured on the Generic perennial at Day
270, which has one stem and 11 branches leaving it at 48 degrees.

The branching is a stack of levels in the **Components** tree. The **Stems**
level is the plant's stems; a **Branches** level under it is the branches on
those stems, and a level under that is the branches on the branches. Select a
level in the tree to see its settings below it.

![The Branches level selected in the Components tree](components-level)

## More stems

Select the **Stems** level and raise **Stems**. At 3 the plant has three stems,
and each carries its own branches: 33 in all. **Spread** sets how far apart the
stems stand at the ground, and **Splay** how far they lean out from vertical.

## A wider or narrower branch angle

Select the **Branches** level and set **Splay**, the angle each branch leaves
its stem at, measured from vertical. At 75 the branches leave at 75 degrees.
**Phyllotaxis** sets how the branches turn round the stem from node to node.

## Shorter branches

On the **Branches** level lower **Growth**. At 50% each branch is the lower half
of its full length, 0.08 m instead of 0.16 m, and still flowers at its cut tip.
**Steps** sets how many nodes a full branch has.

![Branches at Growth 100% and 50%](r-branch-growth)

## Branches only on the lower part of the stem

1. Select the **Branches** level and find **Masks** at the bottom of its
   settings.
2. Click **+** and pick **Height**.
3. Set **Min** to 0 and **Max** to 0.5.

The plant keeps the branches whose bud sits in the lower half of its height and
grows 7 instead of 11. A rejected bud never grows, so the plant does not carry
or bend under the missing branches.

## Branches on the branches

Select the **Branches** level, click **+** and pick **Branches**. A new level
appears under it with its own settings.

To vary a level setting from branch to branch, map it: click the map icon beside
the setting. *Vary a parameter across the plant with a map* covers it. The
explanation *Levels and components* covers how the stack and the components fit
together.
