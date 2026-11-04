#!/usr/bin/env python3
"""Eksport wag YOLO-pose (.pt) do ONNX dla aplikacji.

Uruchamiaj w środowisku do treningu (z ultralytics), NIE w venv aplikacji:
    python export_onnx.py --weights runs/pose/tpa_pose/weights/best.pt --imgsz 800
Wynik: models/kostek-v2n.onnx
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=800, help="ten sam co przy treningu")
ap.add_argument("--out", default="models/kostek-v2n.onnx")
a = ap.parse_args()

exported = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(exported, out)
print(f"Zapisano: {out}")
