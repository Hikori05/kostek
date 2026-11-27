#!/usr/bin/env python3
"""Trening YOLO-pose na danych z prepare_yolo_pose.py (pod GTX 1650 4GB).

    pip install ultralytics
    python train_yolo_pose.py --data datasets/tpa/dataset.yaml
"""
import argparse

from ultralytics import YOLO
import albumentations as A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/tpa/dataset.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="np. yolo11n-pose.pt (4GB VRAM), yolo11s-pose.pt (zmniejsz batch)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=800,
                    help="większe = dokładniejsze punkty, ale więcej VRAM (obrazy mają ~2000+ px)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="tpa_pose")
    a = ap.parse_args()

    xray_aug = [
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.RandomGamma(gamma_limit=(70, 150), p=0.5),
    A.CLAHE(clip_limit=(1, 4), p=0.3),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Sharpen(p=0.2),
]

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, name=a.name, patience=30, cache=True,
        # geometria (bezpieczna dla TPA)
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,        # patrz uwaga wyżej
        flipud=0.0,
        # jasność
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
        # łączenie obrazów
        mosaic=1.0, close_mosaic=30, mixup=0.0,
        # własne augmentacje (wymaga nowszego ultralytics: pip install -U ultralytics)
        augmentations=xray_aug,
        box=1.5,    # domyślnie 7.5 — box przestaje dominować
        dfl=0.5,    # domyślnie 1.5 — druga strata boxa
        pose=40.0,  # domyślnie 12 — nacisk na położenie punktów
        kobj=2.0,   # domyślnie 1.0 — pewność/widoczność punktów
    )

    metrics = model.val()
    print("Pose mAP50-95:", metrics.pose.map)
    print("Najlepsze wagi:", model.trainer.best)


if __name__ == "__main__":
    main()
