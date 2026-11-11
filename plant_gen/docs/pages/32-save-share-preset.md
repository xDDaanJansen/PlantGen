<!-- mode: How-to -->
# Save and share a preset

For keeping a plant you like, reusing it in other files, and sending it to
someone else.

A preset is the whole plant. It holds every setting, every level, component,
mask, map and curve, and its assets: the component collections with their leaf
and flower meshes, the level materials, and their textures. Loading it gives a
1-to-1 copy of the plant you saved.

## Save

1. Select the plant. In **Plant Generator**, click the save button beside the
   preset menu.
2. Give it a **Name** (shown in the menu) and, if you like, a one-line
   **Description**.
3. Keep the suggested folder and click **Save Preset**.

PlantGen writes two files of the same name: `<name>.json` with the settings,
and `<name>.blend` with the assets, textures packed. The suggested folder is
your user presets folder, so the preset appears under **My presets** in the
menu. To keep presets somewhere else, such as a shared drive, set **User
Presets Folder** in the add-on's preferences.

## Load

![The preset menu](preset-menu)

1. Select a plant. Loading a preset replaces all of its settings, so use a new
   plant (**Add Plant**) for a preset you only want to look at.
2. Open the preset menu in **Plant Generator** and pick a preset under
   **Bundled species** or **My presets**. For a file anywhere else, pick
   **Load from File...**.

The plant regrows as the preset. The assets come in from the `.blend` beside
the `.json`; a collection or material the file already has, by name, is used as
it is. The menu now shows the preset's name, and the reload button beside it
re-applies the preset file, picking up changes made to it since.

## Share

Send both files, `<name>.json` and `<name>.blend`. The recipient picks
**Load from File...** and opens the `.json`; the `.blend` must sit beside it.

To check a preset's growth settings without Blender, run this from the PlantGen
folder:

```bash
python tools/validate_preset.py <name>.json
```

It checks that the settings load and save unchanged, prints the plant's size at
three ages, and runs the 14 structural checks on the grown plant. It does not
look at the `.blend`.
