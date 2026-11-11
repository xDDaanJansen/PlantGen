"""Export PlantGen's own UI icons: plant_gen/icons/plant/<name>.svg -> <name>.png.

Each SVG (16 px grid, white, dimmed parts as opacity) is rendered at 32 x 32 by
headless Microsoft Edge on a transparent page, and saved as a white PNG whose
alpha is the anti-aliased coverage -- the mask `blender/plant_icons.py` tints to
the theme at load time. A dev tool: needs Edge (or Chrome, `--browser`) and
Pillow; the add-on itself only reads the PNGs.

    python tools/export_icons.py [--browser PATH]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ICONS = os.path.normpath(os.path.join(HERE, "..", "plant_gen", "icons", "plant"))
SIZE = 32
GAP = 8
BROWSERS = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--browser", help="path to msedge.exe / chrome.exe")
    args = ap.parse_args()
    browser = args.browser or next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        print("No Edge/Chrome found; pass --browser", file=sys.stderr)
        return 1

    names = sorted(f[:-4] for f in os.listdir(ICONS) if f.endswith(".svg"))
    width = len(names) * (SIZE + GAP)
    # <img> keeps each SVG a document of its own (no clash between mask ids)
    cells = "".join(
        f'<img src="{os.path.join(ICONS, n + ".svg").replace(os.sep, "/")}" '
        f'style="position:absolute;left:{i * (SIZE + GAP)}px;top:0;'
        f'width:{SIZE}px;height:{SIZE}px">' for i, n in enumerate(names))
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "icons.html")
        shot = os.path.join(tmp, "icons.png")
        with open(page, "w", encoding="utf-8") as f:
            f.write('<!doctype html><html><body style="margin:0;background:transparent">'
                    f"{cells}</body></html>")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", "--default-background-color=00000000",
                        f"--window-size={width},{SIZE + GAP}", f"--screenshot={shot}",
                        "file:///" + page.replace(os.sep, "/")],
                       check=True, capture_output=True, timeout=60)
        sheet = Image.open(shot).convert("RGBA")
        for i, n in enumerate(names):
            x = i * (SIZE + GAP)
            alpha = sheet.crop((x, 0, x + SIZE, SIZE)).getchannel("A")
            if not alpha.getbbox():
                print(f"{n}: rendered empty", file=sys.stderr)
                return 1
            out = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 0))
            out.putalpha(alpha)
            out.save(os.path.join(ICONS, n + ".png"), optimize=True)
            print(f"{n}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
