<!-- mode: How-to -->
# Export a Nanite Assembly to Unreal

For taking a plant into Unreal Engine 5.7 as a Nanite Assembly with wind, the
export to use for large plantings. The stems become one skinned mesh. Each leaf
and flower becomes an instance of its component's mesh, tied to its own bone.
Rudbeckia fulgida exports as a 37,266-vertex base mesh and 267 instances of 13
meshes on 364 bones. These steps were tested in Unreal 5.7.4.

You need a built rig first: *Rig the plant and export it to Unreal with wind*
covers Bone Angle and the Blender wind previews.

## Choose the export

Each button in **Physiology › Rig** writes one model and one wind file.

| Button | Writes | Use it for |
|---|---|---|
| **Export Nanite Assembly** | `<name>_assembly.usdc`, `<name>_assembly_wind.json`, `textures/` | Many plants in Unreal 5.7 or later |
| **Export Rig** | `<name>.fbx` or a USD file, `<name>_wind.json` | One ordinary skeletal mesh, with every leaf as unique geometry |

**Plant Generator › Variations** has a third button, also named **Export Nanite
Assembly**. It writes `<name>_variations.usdc` for a plant and all its built
variations, one wind file per plant and `textures/`. Use it whenever you want
more than one variation in Unreal. Unreal shares assets only within one import:
ten Rudbeckia fulgida variations made 124 part meshes, 42 materials and 10
textures as ten files, and 13, 4 and 1 as one file. A variation with no current
rig gets one for the export and loses it after, so your scene stays as it was.
The ten took 5.5 minutes.

The USD file finds its images through a relative path, so keep `textures/`
beside it. The wind file says how much each bone sways. Exports of the same rig
write identical wind files, so one copy is enough. Each plant in a variations
file needs its own wind file, because each has its own skeleton.

## Set up the Unreal project

Do this once per project.

1. In **Edit > Plugins**, enable **USD Importer** and **Procedural Vegetation
   Editor**. The second brings in Dynamic Wind and holds the wind actor.
2. Leave **Interchange OpenUSD** off.
3. In **Project Settings > Engine > Rendering > Nanite**, turn on **Nanite
   Foliage (Experimental)** and restart the editor when asked. The first restart
   compiles shaders.

## Export from Blender

1. Select the plant and set its **Day**.
2. Open **Physiology › Rig** and click **Build Rig**.
3. Click **Export Nanite Assembly** and save with the extension `.usdc`.

The export changes four things:

- A flower takes its nearest bloom stage, up to 2.9 mm away on Rudbeckia fulgida.
- An organ that matches none of its component's meshes, such as a leaf still
  unfolding, stays in the base mesh on its own bone.
- Materials carry the colour texture, colour and roughness. Opacity is left out,
  because Unreal's USD import makes an opacity material translucent and a
  translucent material on a Nanite mesh crashes the editor. Leaves and petals
  must be cut-out meshes, not cards with alpha.
- The instanced leaves and flowers lose their per-organ vertex colours.

## Import into Unreal

1. In the Content Drawer, click **Import** and pick the `.usdc`.
2. Keep **Geometry** and **Materials & Textures** on. Turn off **Actors**,
   **Skeletal Animations** and **LevelSequences**.
3. Click **Save All**. The new assets exist only in memory until you save.

The import creates `SK_root`, the assembly, and one `SK_PGS_Proto_*` mesh per
part, in a `SkeletalMeshes` folder. A variations file names each assembly after
its plant instead, such as `SK_Rudbeckia_fulgida_var_001`, and the parts are
shared between them. The `SK_*_Rig` meshes are a step of Unreal's import that
nothing uses afterwards. On the ten variations they are 102 of 154 MB.

## Keep the flowers at a distance

Nanite removes thin parts as it simplifies a distant mesh, so set this on every
mesh or the flowers fade out. USD cannot store the setting.

1. Open each `SK_PGS_Proto_*` mesh and set **Nanite Settings > Shape
   Preservation** to **Voxelize**. A part builds from its own settings.
2. Do the same on `SK_root` last, then save. In a variations import, do every
   assembly last.

On a field of 10,000 Rudbeckia fulgida plants, yellow fell from 40% of the plant's
pixels at 10 m to 0.8% at 200 m without Voxelize. With Voxelize it held 46-56% at
every distance, and frame time stayed within about 1 ms on an RTX 2080.

## Add the wind

1. Right-click `SK_root` and choose **Scripted Asset Actions > Import Dynamic
   Wind Data**. Pick `<name>_assembly_wind.json`, then save. In a variations
   import, do this once per assembly, each with its own plant's file.
2. Create a **Dynamic Wind Data** asset in the Content Browser, open it and tick
   **Enabled**.

## Place the plant

1. Place an empty Actor and add an **Instanced Skinned Mesh** component.
2. Set the component's mesh to `SK_root` and its **Transform Provider** to the
   Dynamic Wind Data asset.
3. Add one entry to **Instances** per plant. For a field, fill the list from a
   Blueprint or Python with **Add Instances**.
4. Give the plants different Z rotations. Dynamic Wind simulates 8 poses per
   skeleton, one per slice of rotation, so plants that share a rotation sway
   alike. The same limit keeps wind cost flat: it added 0.06 ms at most, with
   100,000 plants 40 cm apart running at 64 FPS on an RTX 2080.

## Start the wind in the level

Dynamic Wind has no gusts of its own, and without a driver the sway stops and
starts.

1. In the Content Browser settings, turn on **Show Plugin Content**.
2. Open `ProceduralVegetationEditor/SampleAssets/Materials/GlobalFoliageActor`.
3. Drag `BP_GlobalFoliageActor_UE5` into the level. It runs in the editor as well
   as in play.
4. Set **Wind Speed** on the placed actor.

## Replace a plant with a new export

Voxelize and the wind data live on the imported meshes. Delete the old import,
import the new file, set Voxelize and import the wind data again, and set the
mesh on the component again.
