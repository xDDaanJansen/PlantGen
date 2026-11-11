<!-- mode: Explanation -->
# Levels and components

The Components tree holds two kinds of thing. **Levels** are the plant's
woody structure: the stems, the branches on them, the branches on those.
**Components** are what that structure carries: leaves, bracts, flowers, buds
and fruit. Each is a separate stack because they answer separate questions.
A level says what grows. A component says what it looks like and where it goes.

## Levels

Level 0 is the stems. Each level below it is the branches growing on the level
above. The Generic perennial has two: one stem, and 11 branches on it.

A level holds everything about its axes: how many and how they turn round
their parent (Phyllotaxis), the angles they leave at, how many nodes a full one
has (Steps) and how much of that it builds (Growth), how long and thick it is,
how stiff, its material, and its own Masks and Maps.

Every node of an axis bears branches of the next level, except the tip node.
What continues the tip is the axis itself, a relay or a fork, as the level's
**Growth Pattern** says:

- **Monopodial**: one tip builds the whole axis, and branches leave its sides.
- **Sympodial**, **Dichasial**, **Pleiochasial**: the axis grows in units. Each
  unit ends, and one, two or a ring of buds below it carry on.
- **Pseudomonopodial (Aligned Relay)**: a sympodium whose relay continues in
  line, so it reads as one straight stem.
- **Dichotomous** and **Pseudomonopodial (Unequal Fork)**: the tip splits, into
  equal forks or into a leader and weaker forks.

The deepest level has no level below it. Its leaf-axil buds can only become
flowers.

A level has no setting for whether it ends in a flower. That is the flower
component's job.

## Components

A component is one organ type and one collection of meshes, placed on any set
of levels. Placed on the Plant row, it goes on every level.

**Where flowers grow is the flower component's Attachment Location.**
**Terminal** flowers the tips, including each unit's tip in a sympodium.
**Axillary**, **Internodal** and **Leaf-opposed** flower the nodes. Growth only
flowers on levels a flower component is placed on. A new plant comes with one,
**Replace Me**, on the Plant row with Terminal attachment, so every tip flowers.

## The first component of a type owns growth

Growth needs a single answer to questions like "how far apart are the flowers"
and "how heavy is a leaf". It takes them from the first component of each type
placed **On organs**, in list order:

- The first flower component grows the main inflorescence. A later flower
  component with **Own Inflorescence** on grows another one on its own levels;
  where two want the same tip, the one higher in the list wins. Other flower
  components draw on the flowers already there.
- The first leaf component's **Phyllotaxis** sets how the grown leaves are
  arranged, and its **Weight** loads the stems.
- The first bract component's **Size** and the first fruit component's **Size**,
  **Set** and **Weight** are the inflorescence's bracts and fruit.

So reordering components can change the plant, not only its look. Muting a
component never does. **Mute** hides what the component draws and leaves growth
alone, including when the muted one is the owner.

## Every placed leaf weighs

A leaf is never decoration only. Leaves placed by a component (on Nodes,
Random, Basal and so on) add their weight to the stem nodes they sit on, the
same as grown leaves. A component's Masks are not counted when working out
that weight, and its Prune is counted as an average share.

## Seeds

Every level and every component has its own **Seed**, mixed with the plant's
**Master Seed**. At 0 a level or component follows the master alone. Change
one to reshuffle that part only: a new branch layout on one level, or a new
mesh pick for one component, with everything else unchanged. *Seeds and the
same plant twice* covers why this works.
