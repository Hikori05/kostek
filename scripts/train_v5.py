#!/usr/bin/env python3
"""Ulepszony trening YOLO-pose pod pomiar TPA (naprawa ucinania punktu P1).

Działa zarówno na czystym zbiorze (data/datasets/dataset.yaml), jak i na wykadrowanych
obrazach z liniami (data/datasets/tpa_lines/dataset.yaml).

Użycie:
    python scripts/train_v5.py --data data/datasets/dataset.yaml --model models/yolo26n-pose.pt --name kostek-v5
    python scripts/train_v5.py --data data/datasets/tpa_lines/dataset.yaml --model models/yolo26n-pose.pt --name kostek-lines
"""
import argparse
from pathlib import Path

import albumentations as A
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ścieżka do dataset.yaml")
    ap.add_argument("--model", default="models/yolo26s-pose.pt", help="bazowy model YOLO-pose")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="data/runs/pose")
    ap.add_argument("--name", default="kostek-v5s")
    a = ap.parse_args()

    xray_aug = [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(75, 135), p=0.5),
        A.CLAHE(clip_limit=(1, 3), p=0.3),
        A.GaussNoise(p=0.2),
    ]

    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project=str(Path(a.project).resolve()),
        name=a.name,
        workers=2,
        patience=100,
        cache=True,
        # Geometria bezpieczna dla całej piszczeli (bez ucinania stawu skokowego P1)
        degrees=15.0,
        translate=0.10,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,          # wyłączone: przy małym zbiorze odbicie ściąga P3 i P4 do środka kolana
        flipud=0.0,
        # Wyłączamy mozaikę, random erasing i randaugment (psują precyzję keypointów)
        mosaic=0.0,
        mixup=0.0,
        erasing=0.0,
        auto_augment=None,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        augmentations=xray_aug,
        # Zbalansowane wagi strat: silny box obejmuje całą kość od kolana po kostkę P1
        box=7.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
    )

    metrics = model.val()
    print("Pose mAP50-95:", metrics.pose.map)
    print("Najlepsze wagi:", model.trainer.best)


if __name__ == "__main__":
    main()
