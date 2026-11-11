<!-- mode: How-to -->
# Rig the plant and export it to Unreal with wind

For taking a plant into Unreal Engine as a skeletal mesh that sways in the
wind. Unreal's Dynamic Wind simulates the sway itself from bone groups that
PlantGen writes. No motion is baked or exported. For large plantings, build the
rig here and then follow *Export a Nanite Assembly to Unreal*.

## Build the rig

1. Open **Physiology › Rig**.
2. Set **Bone Angle** for each level: how far a stem may bend along one bone
   before the next bone starts, 15° by default. Straight runs share a bone and
   bends get more. At 0 every node gets a bone.
3. Click **Build Rig**.

Build Rig welds the stems first, since the rig skins the welded mesh. It then
builds a bone chain down every stem and branch and a bone for every leaf,
bract, flower, bud and fruit, skins the mesh with smooth weights, and turns on
**Rig View**. The plant you see is now the rigged mesh, and the fold counts the
stem and organ bones.

![Physiology › Rig after Build Rig](rig-fold)

**Show Bones** draws the skeleton over the plant, and **Display** sets how it
draws. **Fast Preview**, on by default, moves the leaves and flowers rigidly
with their bones so playback stays quick; the export always uses the full mesh.

An edit that regrows the plant drops back to the live plant. The fold then warns
in red that the plant changed since the rig was built, and Rig View and Export
still show the old plant. Click **Build Rig** again after the edit.

## Preview the wind

1. Open **Physiology › Wind** and turn on **Bone Wind**.
2. Play the timeline.

Each stem bone bends by its own flexibility, so long thin parts sway most.
**Strength**, **Stiffness** and the **Gusts** settings shape this preview only;
they never reach Unreal. **Vertex Wind** adds the leaf-tip flutter that an
engine's foliage material makes from the vertex colours, also preview only.

## Export

1. Click **Export Rig** and give a file name ending in `.fbx` or `.usdc`.
2. PlantGen writes the skeletal mesh in its bind pose with weights, UVs,
   materials and vertex colours, the plant's root at the origin. Beside it goes
   `<name>_wind.json`.

## In Unreal

1. Import the FBX or USD as a skeletal mesh.
2. Right-click the mesh, choose **Scripted Asset Actions > Import Dynamic Wind
   Data** and pick `<name>_wind.json`.
3. Create a **Dynamic Wind Data** asset and tick **Enabled**.
4. Place the mesh through an **Instanced Skinned Mesh** component with that asset
   as its **Transform Provider**.
5. Drag Epic's `BP_GlobalFoliageActor_UE5` into the level to drive the wind. Its
   location is in *Export a Nanite Assembly to Unreal*.

The wind file puts every bone in a simulation group: the trunk, one group per
branching level, the inflorescences, and the organs. The stems of a perennial all
go on the trunk. Each other group's influence comes from how flexible its bones
are. The **Unreal wind groups** list at
the bottom of the Wind fold shows the groups, their bone counts and their
influence. Dynamic Wind is experimental in Unreal 5.7 and blows from one global
direction.

![Physiology › Wind with the Unreal wind groups](wind-fold)
