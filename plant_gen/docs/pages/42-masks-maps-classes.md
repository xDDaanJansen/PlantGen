<!-- mode: Explanation -->
# Masks, maps and classes

PlantGen computes a plant from botany and physics: how fast a shoot grows, how
much a stem bends under its load. Masks and maps let you steer that result part
by part without replacing it. The split is simple. The simulation computes a
value; a map multiplies it; a mask decides whether a part is there at all.

## Channels: what every part knows about itself

Every branch, node, leaf and flower carries a set of values from 0 to 1 called
channels: how high up the plant it sits (Height), how early it was made (Age),
how shaded it is (Shade), its index along its stem, its branch depth, and a
random value of its own. Masks and maps read these. A channel is a measurement,
not a setting: Height is 0.5 halfway up whatever the plant ends up being.

You can add channels. A **custom attribute** is a new one: closeness to an
object, a 3D noise, or two channels combined. A **class** is a named set of
masks, read as a channel that is 1 on the parts the masks keep and 0 elsewhere.
Use a class when several components or maps should agree on the same group of
parts, such as "the sunny upper side", defined once.

## Masks decide which

A mask keeps the parts whose channel falls in a band, **Min** to **Max**, and
can invert to keep the outside instead. Several masks combine into one keep
value.

Where a mask sits decides what it does to the plant:

- **A level's masks** run during growth and decide which branches grow. A
  rejected bud never grows, so the plant does not carry or bend under it.
  Keeping Height 0 to 0.5 on the Generic perennial's branches leaves 7 of 11.
- **A component's masks** run after growth and decide which points the
  component draws. The plant is already grown and bent by then, so a masked
  leaf still counted toward the stem's load; it isn't drawn by this
  component. Another component of the same type may take the point.

## Maps decide how much

A map turns a channel into a multiplier through a curve, and multiplies a
setting by it:

> value = setting × (1 + Amount × (curve(channel) − 1))

At Amount 0, or with the curve flat at 1, the map does nothing. That is how
every map starts.

A map never creates, places or removes anything. It scales a number the
simulation was already using. A map on a branch level's **Length › Start**
makes each branch longer or shorter, and growth then carries on as usual with
that branch: it bears as many nodes, and its weight bends the stem below it
accordingly. An Along Height map there makes the Generic perennial's lowest
branch 1.22 times as long and its highest 0.86 times.

## When a map is read

A map is read where its setting is used, and that decides what it can see:

- **Growth maps**, on a level's settings, the plant's Internode Length and
  Gravitropism, and a flower component's inflorescence settings, are read per
  branch, node or flower while the plant grows. Their effects pass through the
  physics.
- **Instance maps**, on a component's Size, Pitch, Yaw, Roll, Fold and petiole
  settings, are read per drawn leaf or flower after the plant is grown.
- **Per-plant maps** cover every other setting, which is one value for the
  whole plant. They draw one random value per plant through the curve. In a
  row of Variations each plant gets its own value.

A **Random Range** on a setting is a per-part random spread around it. On a
setting that used to have a +/- variance, the Random Range is that variance,
so older plants are unchanged.

## Why maps and not more sliders

Each extra slider is one more number that has to agree with all the others. A
map instead says how an existing setting varies over the plant, in terms of
something the plant already measures. The setting keeps one meaning, and one
map icon beside it shows whether anything varies it.
