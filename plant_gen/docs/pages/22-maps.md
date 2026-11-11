<!-- mode: How-to -->
# Vary a parameter across the plant with a map

For making a setting change from branch to branch, leaf to leaf or flower to
flower: longer branches low down, smaller leaves near the tip, a random spread
of flower sizes.

Almost every numeric setting of the plant, a level or a component has a map icon
beside it. A filled icon means the setting is mapped.

## Add a map

1. Click the map icon beside the setting.
2. Pick a mode from the menu:
   - **Along Height** multiplies the setting by 1.25 at the base of the plant,
     1 halfway up and 0.75 at the top.
   - **By Age** runs from 0.75 on the first-made parts to 1.15 on the last.
   - **Random Range** gives each element its own value between 0.8 and 1.2
     times the setting.
   - **Custom Curve** starts flat and does nothing until you shape the curve.

![The map menu beside Length › Start](map-menu)

The map opens in **Mapping & Classification**, under the part it belongs to,
with its settings below the list.

![An Along Height map selected in Mapping & Classification](mapping-panel)

As a worked case, an Along Height map on the **Branches** level's **Length ›
Start** makes the Generic perennial's lowest branch 1.22 times as long, 0.177 m
instead of 0.145 m, and its highest 0.86 times, 0.185 m instead of 0.214 m.

## Change what drives it

In the map's settings, **Driver** is the value the curve reads, from 0 to 1:
Height, Age, Shade, Node Index, Branch Depth, a custom attribute, a class, or
Random. The curve turns that value into a multiplier. **Amount** blends the map
in: 0 is no effect, 1 is the curve in full.

**Mode** switches the map to another of the four modes.

## A setting the plant reads once

Some settings are one value for the whole plant, so there is nothing to vary
along. Their map offers **Random Range** and **Per Plant (random)**, a curve
over one random value per plant. The map's settings say how it is evaluated:
per plant, per branch, per node, per instance, per leaf or per flower.

## Switch a map off or remove it

The checkbox on the map's row in **Mapping & Classification** switches it off
and keeps it. The **-** button beside the list removes it.

To build your own driver, a distance to an object, a noise or two channels
combined, add a **Custom Attribute** in Mapping & Classification. To reuse a
set of masks as one driver, add a **Class**. The explanation *Masks, maps and
classes* covers how a map changes a value the simulation computes.
