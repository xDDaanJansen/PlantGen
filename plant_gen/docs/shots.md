# Panel screenshots for the manual

Each row is one screenshot, saved as `plant_gen/docs/images/<Shot>.png`. The
pages show it where they say `![...](<Shot>)`, or a labelled placeholder until
it exists.

`tools/panel_shots.py` takes them: it opens a factory-settings Blender window
per shot (never your session or files), stages the setup below, and crops the
screenshot to the panel. Every shot, or the named ones:

```bash
python tools/panel_shots.py
```

```bash
python tools/panel_shots.py rig-fold wind-fold
```

The staging for each shot lives in `SHOTS` in that script; this table says what
each one shows. **Panel** is the panel's sidebar name. The build watches that
panel and reports a screenshot as stale once the panel's controls change: rerun
the script for it. A shot you would rather take by hand is just a PNG of that
name: crop to the panel, sidebar at its default width, Blender's default theme.

Figures named `r-...` are renders. `tools/render_figures.py` makes those; they
are not in this list. Neither are the leaf-shape drawings in `docs/figures/`
(`leafshape-<shape>.svg`): `tools/leaf_shape_figures.py` draws them from the
Leaf Generator's own blades.

A shot marked **Reference** is also shown in the reference, at the top of the
panel, fold or sub-panel it shows (`REFERENCE_SHOTS` in `tools/ui_reference.py`).

| Shot | Panel | Setup | Used in |
|---|---|---|---|
| plant-generator | Plant Generator | A new plant, Growth Timelapse and Variations folded | 10 Your first plant |
| preset-menu | Plant Generator | The preset menu open (cropped from the viewport) | 32 Save and share a preset |
| calendar | Plant Generator | The Phenology Calendar open (cropped from the viewport) | 40 How a plant grows |
| timelapse-fold | Plant Generator | Growth Timelapse open, Grow Over Timeline on | 30 Render a growth timelapse |
| variations-fold | Plant Generator | Variations open, Layout Row | 31 Make a grid of variations |
| components-tree | Components | The tree with Replace Me, a leaf component (selected), Stems and Branches | 21 Place organs with components |
| components-level | Components | The Branches level selected, its settings below | 20 Shape the branching with levels |
| components-flower | Components | Replace Me as a Raceme, selected: Inflorescence and Flowers | 25 Make a hanging raceme |
| map-menu | Components | The map icon's menu for Branches › Start Length (cropped from the viewport) | 22 Vary a parameter with a map |
| mapping-panel | Mapping & Classification | An Along Height map on Branches › Start Length, selected | 22 Vary a parameter with a map |
| traits-sheet | Components | A leaf component with three meshes, Traits open, one Age column | 23 Choose between mesh variants |
| stiffness-fold | Physiology | Stiffness open | 43 Bend and droop under load |
| pruning-fold | Physiology | Pruning open | 27 Prune limbs by hand |
| surface-fold | Physiology | Surface open after a weld | 28 Weld and texture the plant |
| rig-fold | Physiology | Rig open after Build Rig | 29 Rig the plant and export it |
| wind-fold | Physiology | Wind open after Build Rig, Bone Wind on, the Unreal wind groups listed | 29 Rig the plant and export it |
| effector-panel | Plant Effector | An effector selected, Kind Flatten | 26 Grow around obstacles |
| texturing-materials | Texturing | Materials open, the Stems row selected with a stem material | 28 Weld and texture the plant |
| flower-generator | Flower Generator | A new FlowerGen flower selected | 11 Your first flower; Reference |
| flower-bloom | Flower Generator | A new flower, Bloom open | 11 Your first flower; Reference |
| flower-petals | Flower Generator | Petals open, Outline and Placement unfolded | 11 Your first flower; Reference |
| flower-petal-layers | Flower Generator | Petals open, its Mesh fold and Layers open | Reference |
| flower-petal-deform | Flower Generator | Petals › Deform open, with Curl and Rotation | Reference |
| flower-petal-bud | Flower Generator | Petals › Bud Shape open | Reference |
| flower-corolla-tube | Flower Generator | Petals › Corolla Tube open, Corolla Tube on | Reference |
| flower-sepals | Flower Generator | Sepals open and on, Outline and Placement unfolded | Reference |
| flower-disc-receptacle | Flower Generator | Disc and Receptacle open, Receptacle on | Reference |
| flower-material | Flower Generator | Material open after New Flower Material, Textures & Settings unfolded | Reference |
| leaf-generator | Leaf Generator | Generate mode | 12 Your first leaf; Reference |
| leaf-profile-margin | Leaf Generator | Generate mode, Profile Curve and Margin open, Margin Serrate with 14 teeth | 12 Your first leaf; Reference |
| leaf-deform | Leaf Generator | Generate mode, Deform open: Cup 0.2, Gravity 0.4, Wave 0.1, Noise 0.1 | Reference |
| leaf-compound | Leaf Generator | Generate mode, Compound open, set to Pinnate | Reference |
| leaf-cutout | Leaf Generator | Cutout mode with a 512 px atlas image picked | 12 Your first leaf; Reference |
| leaf-edit | Leaf Generator | A generated leaf selected, its Material fold open | 12 Your first leaf; Reference |
