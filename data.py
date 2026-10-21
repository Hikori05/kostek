import json, random, shutil
from pathlib import Path

EXPORT = "export.json"
IMAGES = Path.home() / ".local/share/label-studio/media/upload/8"   # numer projektu zmień
OUT = Path("dataset")
ORDER = ["p1", "p2", "p3", "p4"]
KP_FIELD = "kp"               # name z <KeyPointLabels name="kp">
PAD = 0.05

tasks = json.load(open(EXPORT, encoding="utf-8"))
random.seed(0)
random.shuffle(tasks)
split = int(len(tasks) * 0.8)

for i, t in enumerate(tasks):
    pts = {k["keypointlabels"][0]: (k["x"] / 100, k["y"] / 100)
           for k in t.get(KP_FIELD, [])}
    if any(l not in pts for l in ORDER):
        print("pomijam (brakuje punktów):", t["image"])
        continue

    part = "train" if i < split else "val"
    (OUT / "images" / part).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / part).mkdir(parents=True, exist_ok=True)

    p = [pts[l] for l in ORDER]
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    x0, x1 = max(min(xs) - PAD, 0), min(max(xs) + PAD, 1)
    y0, y1 = max(min(ys) - PAD, 0), min(max(ys) + PAD, 1)
    box = f"{(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}"
    kps = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in p)

    name = Path(t["image"]).name
    stem = Path(name).stem
    (OUT / "labels" / part / f"{stem}.txt").write_text(f"0 {box} {kps}\n")
    shutil.copy(IMAGES / name, OUT / "images" / part / name)
