<!-- mode: Explanation -->
# The rig and Unreal's wind

PlantGen exports a plant to a game engine as a skeletal mesh, and lets the
engine do the swaying. This page explains why the rig is built the way it is
and what the wind file actually tells Unreal.

## Why the engine animates, not PlantGen

A baked wind animation loops the same few seconds, ignores the scene's wind
direction, and costs memory per plant. Unreal's Dynamic Wind instead simulates
the sway live from the skeleton, for one global wind. What it needs from us is
not motion but a description: which bones form chains, and how freely each
group moves. So Export Rig writes the mesh in its rest pose and a wind file
that classifies the bones. Nothing moving is exported.

## How the bones are placed

A bone per node would be thousands of bones on a leafy perennial. A bone per
stem would sway as a stiff stick. The rig sits between the two by asking two
questions of each stretch of stem:

- **Shape.** A bone runs on while every node it covers stays within the
  level's **Bone Angle** of the bone's line. A straight run shares one bone; a
  curve gets more.
- **Flex.** It also stops where the stem would turn more than the Bone Angle
  under its own weight pushed sideways, the same beam rule the bend pass uses.
  A thin, flexible tip gets more bones than a thick base of the same shape,
  because that's where the sway is.

A bone chain also breaks wherever a branch leaves, so each branch hangs off a
bone that ends at its base. Every drawn leaf, bract, flower, bud and fruit gets
one bone from its stalk's root to its far end, under the stem bone carrying it.

The exception is flowers that share an axis. They have no bone of their own and
ride the stem bone at their node, so a raceme sways as one piece. A flowering
tree needs this: Laburnum's 29,595 flowers were 62% of a 47,400-bone rig. A lone
flower head, such as Rudbeckia's, keeps its bone and nods on its own stalk.

## How the mesh follows the bones

Each vertex follows at most four bones. Along a stem, two neighbouring bones
blend across each joint so it bends smoothly. A branch is handed over from its
parent's bones to its own over a short distance from the parent's surface, so
the junction doesn't tear. A stalk's foot is handed over to its organ's bone the
same way. Which bone a vertex belongs to is worked out from the geometry and
the grown plant, not from the welded mesh's labels, which are unreliable at
junctions.

## What the wind file says

The wind file puts every bone in one simulation group:

- **Trunk**: the root and every stem bone within 16 bones of it, at any level,
  plus any run of fewer than 8 bones deeper than that.
- **Branches**: the other stem bones deeper than that, at influence 0.
- **Organs 1 to 4**: every leaf, flower and fruit bone, sorted by how free the
  stem bone it hangs from is.

The trunk is where the depth rule matters. Dynamic Wind bends trunk bones along
one smooth, slow line, and that bend adds up from bone to bone, so the limit is a
line's depth and never its level. Other groups give each bone its own noise and a
fast oscillation. Stems jittered there at every influence tried: a two-bone side
stem at any value, and Laburnum and Corylus branches down to a maximum of 0.01.
So the deep stems get no wind of their own and ride their parent, which moves a
tree's outer wood as a whole. A perennial's stems are all on the trunk. On
Corylus, 1,779 bones are and 2,213 outer branch bones ride.

An organ's influence is the flexibility of the bone it hangs from, relative to
the most flexible stem bone in the plant, scaled by 0.35. The file's gust
attenuation starts from one minus the mean flexibility of the level 0 stems, so a
stiff plant shrugs off gusts that move a lax one. Past 10 bones on the longest
trunk line, the gust that gets through is shared over the line's bones, because
more bones would otherwise mean more sway: Araucaria's 17 bones went from 0.6 to
0.765.

Because the groups come from the same beam model as the bend, a plant that
droops easily in PlantGen sways easily in Unreal.

## The previews

**Bone Wind** in Blender approximates what Unreal will do: each stem bone bends
by its own flexibility times a gust. Since the bend comes from the stem's own
flexibility, the sway depends on how long and stiff the stems are, not on how
many bones they have. Its Strength, Stiffness and Gust settings exist only for
the preview; Unreal has its own.

**Vertex Wind** previews the leaf flutter a foliage material makes from the
vertex colours. Engines do that in the material, so it is not exported either.

**Fast Preview** keeps playback quick. A deforming mesh costs Blender most in
redrawing it, about 130 ms a frame for a 250,000-vertex plant. So Rig View
deforms only the stems and moves the leaves and flowers as rigid copies with
their bones. On the plants it was measured on, the shape matches the full mesh
to 0.001 mm and playback runs 2 to 2.7 times faster. It falls back to the full
mesh while Vertex Wind is on, and the export always uses the full mesh.
