#!/usr/bin/env python3
"""Cases.csv + obrazy -> dataset YOLO-pose (1 klasa 'tibia', 4 keypointy P1..P4).

Użycie:
    python prepare_yolo_pose.py --src /sciezka/do/folderu --out datasets/tpa --val-cases 2
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

KPT_NAMES = ["talus", "eminence", "cranial_plateau", "caudal_plateau"]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_image(src: Path, case: str, stage: str):
    num = int(case.split("_")[-1])
    for stem in (f"{num}_{stage}", f"{case}_{stage}"):
        for ext in IMG_EXTS:
            p = src / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def read_rows(csv_path: Path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # nagłówek (kolumny mają końcowe spacje, więc czytamy pozycyjnie)
        for r in reader:
            if not r or not r[0].strip():
                continue
            yield {
                "case": r[0].strip(),
                "side": r[1].strip(),
                "stage": r[2].strip(),
                "pts": [(float(r[3]), float(r[4])), (float(r[5]), float(r[6])),
                        (float(r[7]), float(r[8])), (float(r[9]), float(r[10]))],
                "csv_wh": (int(r[11]), int(r[12])),
                "tpa": r[13].strip().replace(",", "."),
            }


def make_label(pts, w, h, pad):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0 = max(0.0, min(xs) - pad * w)
    x1 = min(float(w), max(xs) + pad * w)
    y0 = max(0.0, min(ys) - pad * h)
    y1 = min(float(h), max(ys) + pad * h)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for x, y in pts:
        parts.append(f"{x / w:.6f} {y / h:.6f} 2")  # 2 = widoczny
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".", help="folder z obrazami i Cases.csv")
    ap.add_argument("--csv", default=None, help="ścieżka do CSV (domyślnie SRC/Cases.csv)")
    ap.add_argument("--out", default="datasets/tpa")
    ap.add_argument("--val-cases", type=int, default=2,
                    help="ile przypadków (pre+post) do walidacji; 0 = val == train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.08,
                    help="margines bbox jako ułamek wymiaru obrazu")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    csv_path = Path(a.csv) if a.csv else src / "Cases.csv"

    items, skipped = [], 0
    for r in read_rows(csv_path):
        img = find_image(src, r["case"], r["stage"])
        if img is None:
            print(f"[SKIP] brak obrazu dla {r['case']} {r['stage']}")
            skipped += 1
            continue
        with Image.open(img) as im:
            w, h = im.size
        if (w, h) != r["csv_wh"]:
            print(f"[WARN] {img.name}: rozmiar pliku {w}x{h} != CSV {r['csv_wh'][0]}x{r['csv_wh'][1]} "
                  f"(używam rozmiaru pliku)")
        bad = [p for p in r["pts"] if not (0 <= p[0] <= w and 0 <= p[1] <= h)]
        if bad:
            print(f"[SKIP] {img.name}: punkty poza obrazem {bad} (rozmiar {w}x{h})")
            skipped += 1
            continue
        items.append((r, img, w, h))

    if not items:
        sys.exit("Brak danych do przetworzenia.")

    # podział PO PRZYPADKACH (pre i post tego samego psa nie mogą trafić do różnych zbiorów)
    cases = sorted({r["case"] for r, *_ in items})
    random.Random(a.seed).shuffle(cases)
    val_set = set(cases[:a.val_cases])

    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    n = {"train": 0, "val": 0}
    for r, img, w, h in items:
        splits = ["val"] if r["case"] in val_set else ["train"]
        if a.val_cases == 0:
            splits = ["train", "val"]
        label = make_label(r["pts"], w, h, a.pad)
        for s in splits:
            shutil.copy2(img, out / "images" / s / img.name)
            (out / "labels" / s / f"{img.stem}.txt").write_text(label + "\n")
            n[s] += 1

    cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [4, 3],
        "flip_idx": [0, 1, 2, 3],  # odbicie poziome nie zmienia znaczenia punktów
        "names": {0: "tibia"},
        "kpt_names": {0: KPT_NAMES},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    print(f"\nGotowe: {out}/dataset.yaml")
    print(f"train: {n['train']} obrazów, val: {n['val']} obrazów, pominięto: {skipped}")
    print(f"val cases: {sorted(val_set) if val_set else '(val == train)'}")


if __name__ == "__main__":
    main()
