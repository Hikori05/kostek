import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

PAD = 10          # margines ramki wokół punktów (px)
VAL_RATIO = 0.2   # część zdjęć na walidację


def first_existing(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    raise SystemExit(f"Nie znaleziono żadnej z ścieżek: {candidates}")


CSV = first_existing("dataset/Cases.csv", "Cases.csv")
SRC = {
    "clean": first_existing("img_rtg", "dat/img_rtg"),
    "lines": first_existing("img_lines", "dat/img_lines"),
}

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()


def col(prefix):
    return next(c for c in df.columns if c.startswith(prefix))


PC = [(col(f"P{i} X"), col(f"P{i} Y")) for i in range(1, 5)]
need = [c for xy in PC for c in xy] + ["case", "image", "Image Width", "Image Height"]
df = df.dropna(subset=need)

# case_0010 + preop -> 10_preop
num = df["case"].str.extract(r"(\d+)")[0].astype(int).astype(str)
df["stem"] = num + "_" + df["image"].str.strip()

# --- etykiety YOLO-pose (jedna linia na obiekt) ---
labels = {}
for stem, group in df.groupby("stem"):
    lines = []
    for _, r in group.iterrows():
        w, h = r["Image Width"], r["Image Height"]
        xs = [r[x] for x, _ in PC]
        ys = [r[y] for _, y in PC]
        x0, x1 = max(0, min(xs) - PAD), min(w, max(xs) + PAD)
        y0, y1 = max(0, min(ys) - PAD), min(h, max(ys) + PAD)
        box = (f"{(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
               f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        kps = " ".join(f"{x / w:.6f} {y / h:.6f} 2" for x, y in zip(xs, ys))
        lines.append(f"0 {box} {kps}")
    labels[stem] = ("\n".join(lines), (int(w), int(h)))

# --- tylko zdjęcia, które istnieją w obu folderach ---
stems = sorted(
    s for s in labels
    if all(next(src.glob(s + ".*"), None) for src in SRC.values())
)
print(f"zdjęć do użycia: {len(stems)} z {len(labels)} wierszy w CSV")

if not stems:
    for v, s in SRC.items():
        print(v, s.resolve(), "plików:", len(list(s.glob("*"))))
    print("przykładowe nazwy z CSV:", sorted(labels)[:5])
    raise SystemExit("Brak pasujących zdjęć")

random.seed(0)
random.shuffle(stems)
split = max(1, int(len(stems) * (1 - VAL_RATIO)))

# --- zapis dwóch zbiorów: dataset_clean i dataset_lines ---
for variant, src in SRC.items():
    out = Path(f"dataset_{variant}")
    out.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        img = next(src.glob(stem + ".*"))
        if Image.open(img).size != labels[stem][1]:
            print(f"[{variant}] UWAGA: rozmiar {img.name} różni się od CSV")
        part = "train" if i < split else "val"
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, out / "images" / part / img.name)
        (out / "labels" / part / f"{stem}.txt").write_text(labels[stem][0] + "\n")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [0, 1, 2, 3]\n"
        "names:\n  0: bone\n"
    )
    print(f"[{variant}] gotowe -> {out}")
