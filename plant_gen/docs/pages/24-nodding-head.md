<!-- mode: How-to -->
# Make a nodding flower head

For a single flower or a dense head that bows over on its stem, like a poppy
bud or a young sunflower. The numbers below are measured on a new plant (the
Generic perennial) at Day 270, with one flower component on the **Stems**
level, so the only flower is at the stem's tip. Each is the angle of the stem's
tip from vertical.

A head at the tip of a stem nods because it is heavy and the stem gives. Two
settings decide it together: the flower component's **Weight** and the stems
level's **Stiffness**.

## Put one head on the stem

A new plant's flower component, **Replace Me**, sits on the **Plant** row and
flowers every tip. For one head on each stem:

1. In **Components**, select the flower component.
2. In the down-arrow menu beside the tree, pick **Move To** and the **Stems**
   level.
3. Set its **Collection** to the head's mesh.

On the Stems level it flowers the stem's tip only. On the Plant row the branch
tips flower too, and those flowers load the stem as well.

## Make the head heavy

1. Select the flower component.
2. Raise **Weight**. Its default, 0.0004, barely loads the stem.

Doubling Weight about doubles the nod. Flower mass also grows with the cube of
the component's **Size**: at Stiffness 0.3 and Weight 0.05, Size 1 nods 6°,
Size 1.5 nods 18° and Size 2 nods 42°.

## Let the stem give

1. Select the **Stems** level.
2. Lower **Stiffness**. Each 0.1 lower about doubles the nod.

The tip's angle from vertical, by Stiffness and Weight:

| Stiffness | Weight 0.01 | Weight 0.05 | Weight 0.1 | Weight 0.2 |
|---|---|---|---|---|
| 0.5 | 1° | 1° | 3° | 5° |
| 0.3 | 3° | 6° | 11° | 21° |
| 0.2 | 5° | 12° | 22° | 42° |
| 0.1 | 11° | 23° | 44° | 85° |

At Stiffness 0.1 and Weight 0.2 the head is level. A softer stem or a heavier
head hangs it past level.

![The head at Stiffness 0.5, 0.2 and 0.1, Weight 0.2](r-nod-stiffness)

A thick stem resists bending far more than a thin one, so a stiff base with a
nodding tip comes from the stem's taper, set on the **Stems** level under
**Width**.

## If it will not bend

Check the plant is not tiny. The bend comes from mass against stiffness, and on
a small, light plant there is little mass to bend with. For flowers along an
inflorescence that should point down whatever they weigh, use the pedicel's
**Angle** instead, as *Make a hanging raceme* shows.

Model the flower or head mesh yourself, or build it with the Flower Generator;
PlantGen decides where the head sits and how far it nods, not its shape.
