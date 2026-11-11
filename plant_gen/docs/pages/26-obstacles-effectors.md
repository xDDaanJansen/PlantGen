<!-- mode: How-to -->
# Grow around obstacles and steer with effectors

For a plant that leans out from under a table, grows around a rock, or bends
toward a point you choose. All of it is in **Physiology**.

## Grow around an obstacle

1. Put the obstacle meshes in their own collection, for example `Obstacles`.
   Closed shapes work best.
2. Open **Physiology › Obstacles** and set **Collection** to it.
3. Raise **Clearance**, the distance in metres the growing tips keep from a
   surface, up to 0.5 m.

Tips curve away as they approach a surface, and a tip that runs out of room
stops instead of growing through. At Clearance 0 the plant ignores obstacles.
The obstacles also shade the plant.

## Lean out of the shade

In **Physiology › Light** raise **Light Seeking**. It turns the growing tips
toward the light each step, after gravity has had its say, so a shaded shoot
leans out from under whatever covers it. At 0 the plant ignores where the light
comes from.

## Steer by hand with an effector

1. In **Physiology › Effectors** click **Add Effector**. An empty appears at the
   3D cursor, linked to the plant.
2. Select the empty. The **Plant Effector** panel shows its settings.
3. Pick a **Kind**:
   - **Attract** pulls nearby growth toward it.
   - **Repel** pushes nearby growth away.
   - **Forbid** keeps growth out of its sphere; tips that reach the edge stop.
   - **Target** aims the whole plant at it from any distance; Radius is ignored.
   - **Flatten** grows the plant in the plane the empty faces, for an espalier
     or a fan. Rotate the empty to turn the plane.
   - **Orbit** spirals growth around the empty's axis, for a stem twining up a
     pole. Pair it with an Attract on the same spot to keep the stem on the pole.
   - **Align Up** turns each node's leaves and flowers to face the empty. It
     turns where branches leave too, so a bushy plant shifts as a whole.
   - **Level Up** turns each node to lie flat in the empty's plane. Rotate the
     empty to turn the plane.
4. Set **Radius**, how far its influence reaches, and **Strength**, how far a
   tip turns each step. At Strength 1 a tip aims straight at it every step.

An effector in the plant's effector collection acts at once; there is nothing
to switch on first.

![The Plant Effector panel](effector-panel)

## After moving things

The plant regrows when you change a setting. When you move the plant, an
obstacle or an effector, click **Regrow** in Plant Generator so the plant grows
against where things stand now.

The explanation *The environment nudges, it doesn't decide* covers why the
surroundings only ever adjust growth.
