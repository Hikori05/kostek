#!/usr/bin/env python3
"""Benchmark modelu YOLO-pose: porównanie przewidzianych punktów z ręcznie przepisanymi.

CSV (separator , lub ; — liczby z kropką lub przecinkiem), np.:
    image,P1-x,P1-y,P2-x,P2-y,P3-x,P3-y,P4-x,P4-y
    10_preop.png,344,983,382,300,332,289,430,352

Kolumna z nazwą pliku może nazywać się image / file / filename / name / plik;
nazwa może być z rozszerzeniem lub bez. Nagłówki punktów: P1-x, P1_x, P1 x, p1x...

Użycie:
    python benchmark.py --weights models/kostek-v2n.pt --images zdjecia/ \
        --csv wspolrzedne.csv --imgsz 800 --save-vis vis/

Uwaga: wynik ma sens tylko na zdjęciach, które NIE były w zbiorze treningowym.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

NAMES = ["P1", "P2", "P3", "P4"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NAME_KEYS = {"image", "file", "filename", "name", "plik", "zdjecie", "obraz"}


def num(s):
    return float(s.strip().replace(",", "."))


def load_gt(path):
    """-> {nazwa_pliku: ndarray (4, 2)}"""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    first = lines[0]
    if first.count(";") > first.count(","):
        delim = ";"
    elif first.count("\t") > first.count(","):
        delim = "\t"
    else:
        delim = ","
    rows = list(csv.reader(lines, delimiter=delim))

    cols, name_col = {}, 0
    for i, h in enumerate(rows[0]):
        m = re.match(r"^p([1-4])[\s_\-]*([xy])(?![a-z])", h.strip().lower())
        if m:
            cols[(int(m.group(1)) - 1, m.group(2))] = i
        elif h.strip().lower() in NAME_KEYS:
            name_col = i
    missing = [f"P{k + 1}-{a}" for k in range(4) for a in "xy" if (k, a) not in cols]
    if missing:
        sys.exit(f"W CSV brakuje kolumn: {', '.join(missing)}")

    gt = {}
    for r in rows[1:]:
        if not r or not r[name_col].strip():
            continue
        gt[r[name_col].strip()] = np.array(
            [[num(r[cols[(k, "x")]]), num(r[cols[(k, "y")]])] for k in range(4)]
        )
    return gt


def build_index(folder):
    idx = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() in IMG_EXTS:
            idx[p.name.lower()] = p
            idx.setdefault(p.stem.lower(), p)
    return idx


def load_image(path):
    # to samo przygotowanie co w aplikacji webowej
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def predict_points(model, img, imgsz, device):
    r = model.predict(img, imgsz=imgsz, device=device, verbose=False)[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return None
    return r.keypoints.xy[0].cpu().numpy()  # (4, 2), najpewniejsza detekcja


def save_vis(img, gt, pred, out_path):
    d = ImageDraw.Draw(img)
    rad = max(3, img.width // 250)
    for k in range(4):
        gx, gy = gt[k]
        d.ellipse([gx - rad, gy - rad, gx + rad, gy + rad], outline=(0, 255, 0), width=max(2, rad // 2))
        if pred is not None:
            px, py = pred[k]
            d.line([(gx, gy), (px, py)], fill=(255, 255, 0), width=max(1, rad // 3))
            d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(255, 0, 0))
        d.text((gx + rad + 4, gy - rad), NAMES[k], fill=(255, 255, 255))
    img.save(out_path, quality=90)


def summary_table(errs, thr):
    """errs: (N, 4) błędy w px -> tekst tabeli."""
    head = f"{'Punkt':<6}{'N':>4}{'średnia':>9}{'mediana':>9}{'p90':>8}{'max':>8}"
    head += "".join(f"{'≤' + format(t, 'g') + 'px':>8}" for t in thr)
    out = [head, "-" * len(head)]

    def row(label, e):
        s = f"{label:<6}{len(e):>4}{e.mean():>9.1f}{np.median(e):>9.1f}{np.percentile(e, 90):>8.1f}{e.max():>8.1f}"
        s += "".join(f"{(e <= t).mean() * 100:>7.0f}%" for t in thr)
        return s

    for k in range(4):
        out.append(row(NAMES[k], errs[:, k]))
    out.append("-" * len(head))
    out.append(row("ALL", errs.ravel()))
    return "\n".join(out)


def run(model, args):
    gt = load_gt(args.csv)
    index = build_index(args.images)
    thr = [float(t) for t in args.thr.split(",")]
    vis_dir = Path(args.save_vis) if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    records, errs, rel_errs, times = [], [], [], []
    not_found, not_detected, warmed = [], [], False

    for name, g in gt.items():
        path = index.get(name.lower()) or index.get(Path(name).stem.lower())
        if path is None:
            not_found.append(name)
            continue
        img = load_image(path)
        w, h = img.size
        if not warmed:  # rozgrzewka (pierwsze wywołanie jest wolniejsze), nie liczona w czasie
            predict_points(model, img, args.imgsz, args.device)
            warmed = True

        t0 = time.perf_counter()
        pred = predict_points(model, img, args.imgsz, args.device)
        dt = (time.perf_counter() - t0) * 1000

        rec = {"image": path.name, "width": w, "height": h, "detected": int(pred is not None),
               "time_ms": round(dt, 1)}
        if pred is None:
            not_detected.append(path.name)
        else:
            e = np.linalg.norm(pred - g, axis=1)
            errs.append(e)
            rel_errs.append(e / h * 100)
            times.append(dt)
            for k in range(4):
                rec[f"{NAMES[k]}_pred_x"] = round(float(pred[k, 0]), 1)
                rec[f"{NAMES[k]}_pred_y"] = round(float(pred[k, 1]), 1)
                rec[f"{NAMES[k]}_err_px"] = round(float(e[k]), 1)
            rec["mean_err_px"] = round(float(e.mean()), 1)
        records.append(rec)
        if vis_dir:
            save_vis(img, g, pred, vis_dir / f"{Path(path.name).stem}.jpg")

    print(f"\nZdjęć w CSV: {len(gt)}, ocenionych: {len(errs)}, "
          f"bez detekcji: {len(not_detected)}, nie znaleziono pliku: {len(not_found)}")
    for n in not_found:
        print(f"  [brak pliku] {n}")
    for n in not_detected:
        print(f"  [brak detekcji] {n}")
    if not errs:
        sys.exit("Brak wyników do podsumowania.")

    errs = np.array(errs)
    print("\nBłąd odległości predykcja–GT w pikselach:\n")
    print(summary_table(errs, thr))
    print(f"\nŚredni błąd względem wysokości obrazu: {np.mean(rel_errs):.2f}%")
    print(f"Czas inferencji: średnio {np.mean(times):.0f} ms/obraz "
          f"(mediana {np.median(times):.0f} ms, imgsz={args.imgsz})")

    fields = ["image", "width", "height", "detected", "time_ms"]
    fields += [f"{n}_{s}" for n in NAMES for s in ("pred_x", "pred_y", "err_px")] + ["mean_err_px"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(records)
    print(f"\nWyniki per zdjęcie: {args.out}" + (f", wizualizacje: {vis_dir}/" if vis_dir else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="folder z oryginalnymi zdjęciami")
    ap.add_argument("--csv", required=True, help="CSV z ręcznymi współrzędnymi P1-x ... P4-y")
    ap.add_argument("--imgsz", type=int, default=800, help="ten sam co przy treningu")
    ap.add_argument("--device", default=None, help="np. 0 albo cpu (domyślnie auto)")
    ap.add_argument("--thr", default="5,10,20", help="progi w px do % trafień, np. 5,10,20")
    ap.add_argument("--out", default="benchmark_results.csv")
    ap.add_argument("--save-vis", default=None,
                    help="folder na obrazy: zielone = GT, czerwone = predykcja")
    args = ap.parse_args()

    from ultralytics import YOLO
    run(YOLO(args.weights), args)


if __name__ == "__main__":
    main()