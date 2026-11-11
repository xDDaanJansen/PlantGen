<!-- mode: Explanation -->
# The environment nudges, it doesn't decide

A PlantGen plant can respond to light, obstacles and effectors. None of them
decide its shape. The species' own growth decides how many stems, branches,
leaves and flowers there are and where they sit. The surroundings only change
a number that growth was already using, such as which way a tip is heading.
Nothing in the environment creates a part, places one, or ends an axis outright.

## Why it works this way

An early PlantGen grew branches toward open space, the way tree generators
fill a crown. The surroundings then decided where branches went, when they
forked and when they stopped, and the species settings shrank to decoration.
Every plant came out as the same shape at a different size. A perennial's form
is mostly its own program: a foxglove is a spike whether it stands in a field
or by a wall. So the environment was made to bend that program, not to write it.

## What each one changes

All of these are in **Physiology**:

- **Light Seeking** (Light) turns growing tips toward the light each step, after
  gravity has had its say. The light comes from a coarse grid of the plant's
  surroundings, where the plant's own leaves and stems and the obstacles cast
  shade.
- **Heliotropism** with its own **Sun Object** (each flower component's
  Inflorescence box) turns that component's flowers toward the object after
  growth, each flower's own stalk curving with it as one even arc, no joint
  turning more than 7 degrees. **Leaf Heliotropism**
  turns leaf blades toward the plant's **Sun Object** (Light).
- **Clearance** (Obstacles) curves tips away from the meshes in the obstacle
  **Collection** as they come within that distance. A tip that runs out of room
  stops rather than grows through; that is the one case where the surroundings
  end an axis, and only because the space is solid.
- **Effectors** pull, push, flatten or twist growth near an empty you place
  (*Grow around obstacles and steer with effectors*).
- **Shade Death** (Senescence) judges the finished plant: a branch whose base
  ends up darker than the threshold dies, with everything it carries. Stems are
  never killed this way. It runs after growth, on the finished plant's own
  shade, so it doesn't change how the rest of the plant grew.

## Off means exactly off

Every one of these starts at 0, or empty. At 0 the plant is byte-for-byte the
same as one grown with no surroundings at all, and the light grid is not even
built. PlantGen builds it only when something reads it: Light Seeking, Shade
Death, or a mask or map on the Shade channel.

Obstacles and effectors are read in the plant's own space, so moving the plant
next to a rock is the same as moving the rock. Neither move regrows the plant on
its own. Click **Regrow** after moving one.
