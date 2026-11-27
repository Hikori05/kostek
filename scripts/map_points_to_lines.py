#!/usr/bin/env python3
"""Przenosi punkty P1..P4 z czystych obrazów (.png) na wykadrowane obrazy z liniami (.jpg)
i tworzy gotowy dataset YOLO-pose (bez bocznych czarnych pasów ze zrzutu 4254x1518).

Działa na samym Pillow + PyYAML (opcjonalnie doprecyzowuje dopasowanie przez OpenCV, jeśli jest zainstalowane).

Użycie:
    python3 scripts/map_points_to_lines.py \
        --src data/dataset \
        --csv data/dataset/Cases.csv \
        --out data/datasets/tpa_lines \
        --save-vis data/vis_lines
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageDraw

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

KPT_NAMES = ["talus", "eminence", "cranial_plateau", "caudal_plateau"]


def xray_bbox(im_rgb: Image.Image, thr: int = 10, color_thr: int = 18):
    """Zwraca (x0, y0, x1, y1) nieczarnego prostokąta RTG z pominięciem kolorowych linii lekarza."""
    r, g, b = im_rgb.split()
    gray = im_rgb.convert("L")
    d1 = ImageChops.difference(r, g)
    d2 = ImageChops.difference(g, b)
    d3 = ImageChops.difference(r, b)
    color = ImageChops.lighter(ImageChops.lighter(d1, d2), d3)
    bright_m = gray.point(lambda p: 255 if p > thr else 0)
    gray_m = color.point(lambda p: 255 if p <= color_thr else 0)
    mask = ImageChops.darker(bright_m, gray_m)
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, im_rgb.width, im_rgb.height)
    return bbox


def map_png_to_jpg(png_rgb: Image.Image, jpg_rgb: Image.Image):
    """Zwraca (px0, py0, jx0, jy0, jx1, jy1, sx, sy) do wykadrowania .jpg i przeniesienia punktów."""
    px0, py0, px1, py1 = xray_bbox(png_rgb)
    jx0, jy0, jx1, jy1 = xray_bbox(jpg_rgb)

    pw, ph = px1 - px0, py1 - py0
    jw, jh = jx1 - jx0, jy1 - jy0
    sx, sy = jw / pw, jh / ph

    if HAS_CV2:
        scale = (sx + sy) / 2.0
        png_gray = np.asarray(png_rgb.crop((px0, py0, px1, py1)).convert("L"))
        jpg_gray = np.asarray(jpg_rgb.convert("L"))
        tmpl = cv2.resize(
            png_gray,
            (max(1, round(pw * scale)), max(1, round(ph * scale))),
            interpolation=cv2.INTER_AREA,
        )
        pad = 25
        sy0 = max(0, jy0 - pad)
        sx0 = max(0, jx0 - pad)
        sy1 = min(jpg_gray.shape[0], jy0 + tmpl.shape[0] + pad)
        sx1 = min(jpg_gray.shape[1], jx0 + tmpl.shape[1] + pad)
        search = jpg_gray[sy0:sy1, sx0:sx1]
        if search.shape[0] >= tmpl.shape[0] and search.shape[1] >= tmpl.shape[1]:
            res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            ox = sx0 + max_loc[0]
            oy = sy0 + max_loc[1]
            return px0, py0, ox, oy, ox + tmpl.shape[1], oy + tmpl.shape[0], scale, scale

    return px0, py0, jx0, jy0, jx1, jy1, sx, sy


def make_label(pts, w, h, pad=0.08):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0 = max(0.0, min(xs) - pad * w)
    x1 = min(float(w), max(xs) + pad * w)
    y0 = max(0.0, min(ys) - pad * h)
    y1 = min(float(h), max(ys) + pad * h)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for x, y in pts:
        parts.append(f"{x / w:.6f} {y / h:.6f} 2")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dataset", help="folder z podfolderami case_XXXX")
    ap.add_argument("--csv", default="data/dataset/Cases.csv", help="CSV ze współrzędnymi punktów")
    ap.add_argument("--out", default="data/datasets/tpa_lines", help="folder wyjściowy YOLO-pose")
    ap.add_argument("--val-cases", type=int, default=2, help="ile przypadków (pre+post) do walidacji")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.08, help="margines bbox jako ułamek wymiaru obrazu")
    ap.add_argument("--save-vis", default=None, help="folder na obrazy kontrolne z naniesionymi punktami")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    vis_dir = Path(a.save_vis) if a.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if not r or not r[0].strip():
                continue
            case, stage = r[0].strip(), r[2].strip()
            num = int(case.split("_")[-1])
            png_p = src / case / f"{num}_{stage}.png"
            jpg_p = src / case / f"{num}_{stage}.jpg"
            if not (png_p.exists() and jpg_p.exists()):
                print(f"[SKIP] brak pary dla {case} {stage}")
                continue
            pts = [
                (float(r[3]), float(r[4])),
                (float(r[5]), float(r[6])),
                (float(r[7]), float(r[8])),
                (float(r[9]), float(r[10])),
            ]
            rows.append((case, f"{num}_{stage}", png_p, jpg_p, pts))

    cases = sorted({r[0] for r in rows})
    random.Random(a.seed).shuffle(cases)
    val_set = set(cases[:a.val_cases])

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    counts = {"train": 0, "val": 0}
    for case, stem, png_p, jpg_p, pts in rows:
        png_rgb = Image.open(png_p).convert("RGB")
        jpg_rgb = Image.open(jpg_p).convert("RGB")

        px0, py0, jx0, jy0, jx1, jy1, sx, sy = map_png_to_jpg(png_rgb, jpg_rgb)
        crop = jpg_rgb.crop((jx0, jy0, jx1, jy1))
        cw, ch = crop.size

        mapped = [((x - px0) * sx, (y - py0) * sy) for x, y in pts]

        split = "val" if case in val_set else "train"
        crop.save(out / "images" / split / f"{stem}.jpg", quality=95)
        (out / "labels" / split / f"{stem}.txt").write_text(make_label(mapped, cw, ch, a.pad) + "\n")
        counts[split] += 1

        if vis_dir:
            vis = crop.copy()
            d = ImageDraw.Draw(vis)
            rad = max(5, cw // 120)
            d.line([mapped[0], mapped[1]], fill=(255, 200, 0), width=max(2, rad // 3))
            d.line([mapped[2], mapped[3]], fill=(0, 255, 255), width=max(2, rad // 3))
            for idx, (mx, my) in enumerate(mapped, start=1):
                d.ellipse([mx - rad, my - rad, mx + rad, my + rad], fill=(255, 0, 0), outline=(255, 255, 255))
                d.text((mx + rad + 4, my - rad), f"P{idx}", fill=(255, 255, 0))
            vis.save(vis_dir / f"{stem}_check.jpg", quality=92)

    cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [4, 3],
        "flip_idx": [0, 1, 2, 3],
        "names": {0: "tibia"},
        "kpt_names": {0: KPT_NAMES},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    mode = "OpenCV matchTemplate" if HAS_CV2 else "Pillow xray_bbox"
    print(f"Gotowe ({mode}): {out}/dataset.yaml")
    print(f"train: {counts['train']} obrazów, val: {counts['val']} obrazów, val cases: {sorted(val_set)}")
    if vis_dir:
        print(f"Wizualizacje kontrolne: {vis_dir}/")


if __name__ == "__main__":
    main()
