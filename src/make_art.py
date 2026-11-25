"""Turn the reference animal photos into two-ink risograph halftones (dot data for the dashboard).

Each dot: [x, y, r, ink] where ink 0 = black plate, 1 = tiger-orange plate.
Run:  python -m src.make_art  (expects the reference images in web/img/ref_*.png)
"""
import json
import numpy as np
from PIL import Image
from .config import ROOT

IMG = ROOT / "web" / "img"


def plates(rgb):
    a = rgb.astype(float) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    lo, hi = np.percentile(lum, [3, 97])
    lum = np.clip((lum - lo) / (hi - lo + 1e-6), 0, 1)  # auto-contrast
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    warm = np.clip((r - b - 0.06) * 3.0, 0, 1) * np.clip(sat * 1.6, 0, 1) * np.clip(lum * 2.2, 0, 1)
    black = np.clip((0.42 - lum) / 0.42, 0, 1) ** 0.8
    orange = np.clip(warm * 1.4 - black * 0.5, 0, 1)
    return black, orange, lum


def halftone(rgb, step, mask=None, scale=1.0, orange_gain=1.0):
    black, orange, _ = plates(rgb)
    h, w = black.shape
    dots = []
    for yi, y in enumerate(np.arange(step / 2, h, step)):
        off = step / 2 if yi % 2 else 0  # hex packing
        for x in np.arange(step / 2 + off, w, step):
            iy, ix = int(y), int(min(x, w - 1))
            if mask is not None and not mask[iy, ix]:
                continue
            kb = black[iy, ix]; ko = min(1, orange[iy, ix] * orange_gain)
            if ko > 0.18:
                dots.append([round(x * scale, 1), round(y * scale, 1), round(step * 0.58 * np.sqrt(ko) * scale, 2), 1])
            if kb > 0.10:
                # black plate slightly mis-registered, the riso look
                dots.append([round((x + step * 0.18) * scale, 1), round((y + step * 0.12) * scale, 1),
                             round(step * 0.52 * np.sqrt(kb) * scale, 2), 0])
    return dots


def animal_mask(rgb):
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    sky = (b > r + 8) | ((r + g + b) > 600)
    grass = (g > r + 12) & (g > b + 12)
    return ~(sky | grass)


def main():
    herd = np.array(Image.open(IMG / "ref_herd.png").convert("RGB"))
    collage = np.array(Image.open(IMG / "ref_collage.png").convert("RGB"))
    H, W = collage.shape[:2]
    tiger = collage[H // 3 + 2:2 * H // 3 - 1, W // 3 + 2:2 * W // 3 - 1]
    tiger = np.array(Image.fromarray(tiger).resize((150, 150), Image.LANCZOS))
    art = {}
    art["tiger"] = {"w": 150, "h": 150, "dots": halftone(tiger, 2.6, orange_gain=1.25)}
    el = herd[52:120, 118:218]
    el = np.array(Image.fromarray(el).resize((200, 136), Image.LANCZOS))
    art["elephant"] = {"w": 200, "h": 136, "dots": halftone(el, 2.4, mask=animal_mask(el))}
    hd = herd[36:146, 40:220]
    hd = np.array(Image.fromarray(hd).resize((360, 220), Image.LANCZOS))
    art["herd"] = {"w": 360, "h": 220, "dots": halftone(hd, 3.2, mask=animal_mask(hd))}
    bd = herd[6:36, 78:126]
    bd = np.array(Image.fromarray(bd).resize((96, 60), Image.LANCZOS))
    bl, _, lum = plates(bd)
    art["bird"] = {"w": 96, "h": 60, "dots": [[round(x, 1), round(y, 1), 1.5, 0] for y in range(1, 60, 3)
                                               for x in range(1, 96, 3) if lum[y, x] < 0.32]}
    for k, v in art.items():
        print(k, len(v["dots"]))
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)
    json.dump(art, open(ROOT / "web" / "data" / "art.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
