<!-- mode: How-to -->
# Render a growth timelapse

For an animation of the plant growing from seedling to flowering, where every
frame is the real plant on that day.

1. In **Plant Generator** open **Growth Timelapse** and turn on **Grow Over
   Timeline**. **Day** now follows the timeline and shows greyed.
2. Set **Start Day** and **End Day**. The scene's first frame shows Start Day
   and its last frame shows End Day; the defaults run Day 90 to 300. The fold
   prints the mapping, for example *Frames 1-250 play Day 90 -> 300*.
3. Click **Bake Timelapse**. PlantGen grows every frame once and records it, so
   the timeline plays back smoothly. Press `Esc` to stop; the frames done so far
   are kept, and the next bake skips them. The fold counts the recorded frames.
4. Render the animation as usual. A render always regrows each frame exactly.

Each frame is a full regrow, which is why the bake exists. Without it, the first
pass over the timeline regrows the plant on every frame. Any edit to the plant
clears the recorded frames; the bin button beside Bake clears them by hand.

![Growth Timelapse with Grow Over Timeline on](timelapse-fold)

The scene's frame range sets the pace: 250 frames over Day 90 to 300 is 0.84
days a frame. An End Day past the first season runs into winter, and
**Lifespan** decides what happens there. A **Single season** plant stays mature,
an **Annual** dies standing at season end, and a **Herbaceous perennial** dies
back each winter and regrows a slightly bigger top each spring.
