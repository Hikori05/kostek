import csv
import io
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs" / "pose"
MODELS_DIR = ROOT_DIR / "models"
DEMO_MODELS_DIR = ROOT_DIR / "demo" / "models"
SCRIPTS_DIR = ROOT_DIR / "scripts"
BENCH_DIR = ROOT_DIR / "bench"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NAMES = ["P1", "P2", "P3", "P4"]

app = FastAPI(title="Kostek ML Dashboard")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ------------------------------------------------------------- stan treningu
_train_lock = threading.Lock()
_train_state = {
    "running": False,
    "proc": None,
    "name": None,
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "log_path": None,
    "cmd": None,
}


def safe_name(name: str) -> str:
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowa nazwa: {name}")
    return name


def get_font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def calc_tpa(pts) -> Optional[float]:
    if len(pts) < 4:
        return None
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[:4]
    vx1, vy1 = x2 - x1, y2 - y1
    vx2, vy2 = x4 - x3, y4 - y3
    n1 = math.hypot(vx1, vy1)
    n2 = math.hypot(vx2, vy2)
    if n1 * n2 < 1e-9:
        return None
    cos_a = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (n1 * n2)))
    return round(math.degrees(math.acos(cos_a)) - 90.0, 1)


def parse_yolo_pose_label(lbl_path: Path, w: int, h: int):
    if not lbl_path.exists():
        return None, []
    text = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None, []
    parts = text.splitlines()[0].split()
    if len(parts) < 5:
        return None, []
    cx, cy, bw, bh = (float(parts[1]) * w, float(parts[2]) * h,
                      float(parts[3]) * w, float(parts[4]) * h)
    bbox = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    pts = []
    kparts = parts[5:]
    for k in range(min(4, len(kparts) // 3)):
        pts.append((float(kparts[k * 3]) * w, float(kparts[k * 3 + 1]) * h))
    return bbox, pts


def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out or "Brak GPU (CPU)"
    except Exception:
        return "CPU (nvidia-smi niedostępne)"


def read_run_metrics(run_dir: Path) -> dict:
    res_csv = run_dir / "results.csv"
    info = {"epochs": 0, "best_map50_95": None, "last_map50_95": None}
    if not res_csv.exists():
        return info
    try:
        with open(res_csv, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return info
        info["epochs"] = len(rows)
        key = next((k for k in rows[0].keys() if "mAP50-95(P)" in k), None)
        if key:
            vals = [float(r[key].strip()) for r in rows if r.get(key, "").strip()]
            if vals:
                info["best_map50_95"] = round(max(vals), 4)
                info["last_map50_95"] = round(vals[-1], 4)
    except Exception:
        pass
    return info


# ------------------------------------------------------------------ strony UI
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"gpu": gpu_info()},
    )


# ------------------------------------------------------------ API: DANE
@app.get("/api/datasets")
async def api_list_datasets():
    datasets = []
    if DATASETS_DIR.exists():
        for d in sorted(DATASETS_DIR.iterdir()):
            if not d.is_dir() or not (d / "images").exists():
                continue
            item = {"name": d.name, "splits": {}}
            for split in ("train", "val"):
                img_dir = d / "images" / split
                lbl_dir = d / "labels" / split
                imgs = []
                if img_dir.exists():
                    for p in sorted(img_dir.iterdir()):
                        if p.suffix.lower() in IMG_EXTS:
                            lbl = lbl_dir / f"{p.stem}.txt"
                            tpa = None
                            wh = None
                            try:
                                with Image.open(p) as im:
                                    w, h = im.size
                                wh = f"{w}×{h}"
                                _, pts = parse_yolo_pose_label(lbl, w, h)
                                tpa = calc_tpa(pts)
                            except Exception:
                                pass
                            imgs.append({
                                "file": p.name,
                                "size": wh,
                                "has_label": lbl.exists(),
                                "tpa": tpa,
                            })
                item["splits"][split] = imgs
            datasets.append(item)

    raw_clean = len([p for p in (DATA_DIR / "dataset_clean").iterdir() if p.suffix.lower() in IMG_EXTS]) if (DATA_DIR / "dataset_clean").exists() else 0
    raw_cases = len(list((DATA_DIR / "dataset").glob("case_*"))) if (DATA_DIR / "dataset").exists() else 0
    return {
        "datasets": datasets,
        "raw_clean_images": raw_clean,
        "raw_cases": raw_cases,
    }


@app.get("/api/datasets/{dataset}/{split}/{filename}/preview")
async def api_dataset_preview(dataset: str, split: str, filename: str):
    dataset = safe_name(dataset)
    if split not in {"train", "val"}:
        raise HTTPException(status_code=400, detail="Nieprawidłowy split")
    filename = safe_name(filename)

    img_path = DATASETS_DIR / dataset / "images" / split / filename
    lbl_path = DATASETS_DIR / dataset / "labels" / split / f"{Path(filename).stem}.txt"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Brak obrazu")

    im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    w, h = im.size
    bbox, pts = parse_yolo_pose_label(lbl_path, w, h)

    draw = ImageDraw.Draw(im)
    s = max(2, w // 260)
    font = get_font(max(18, w // 36))

    if bbox:
        draw.rectangle(bbox, outline=(56, 189, 248), width=s)
    if len(pts) == 4:
        draw.line([pts[0], pts[1]], fill=(255, 200, 0), width=s)
        draw.line([pts[2], pts[3]], fill=(0, 255, 0), width=s)
        for name, (x, y) in zip(NAMES, pts):
            rad = s * 3
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 0, 0), outline=(255, 255, 255))
            draw.text((x + rad + 6, y - rad), name, font=font, fill=(255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))
        tpa = calc_tpa(pts)
        if tpa is not None:
            draw.text((16, 16), f"GT TPA: {tpa:.1f}°", font=get_font(max(22, w // 28)),
                      fill=(56, 189, 248), stroke_width=3, stroke_fill=(0, 0, 0))

    im.thumbnail((1200, 1200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.post("/api/datasets/upload")
async def api_upload_data(
    target: str = Form(default="dataset_clean"),
    files: list[UploadFile] = File(...),
):
    if target not in {"dataset_clean", "dataset"}:
        raise HTTPException(status_code=400, detail="Dozwolony cel: dataset_clean lub dataset")
    dest_dir = DATA_DIR / target
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        fname = safe_name(f.filename or "")
        raw = await f.read()
        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for member in zf.infolist():
                    if member.is_dir() or "__MACOSX" in member.filename:
                        continue
                    rel = Path(member.filename)
                    target_path = (dest_dir / rel).resolve()
                    if not str(target_path).startswith(str(dest_dir.resolve())):
                        continue
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(zf.read(member.filename))
                    saved.append(str(rel))
        else:
            (dest_dir / fname).write_bytes(raw)
            saved.append(fname)

    return {"ok": True, "target": target, "saved_count": len(saved), "files": saved[:50]}


@app.post("/api/datasets/rebuild")
async def api_rebuild_dataset(
    kind: str = Form(default="tpa"),
    val_cases: int = Form(default=2),
    pad: float = Form(default=0.08),
):
    if kind == "tpa":
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "prep_dat.py"),
            "--src", str(DATA_DIR / "dataset_clean"),
            "--out", str(DATASETS_DIR / "tpa"),
            "--val-cases", str(val_cases),
            "--pad", str(pad),
        ]
    elif kind == "tpa_lines":
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "map_points_to_lines.py"),
            "--src", str(DATA_DIR / "dataset"),
            "--csv", str(DATA_DIR / "dataset" / "Cases.csv"),
            "--out", str(DATASETS_DIR / "tpa_lines"),
            "--val-cases", str(val_cases),
            "--pad", str(pad),
        ]
    else:
        raise HTTPException(status_code=400, detail="Nieznany typ zbioru (tpa lub tpa_lines)")

    proc = await run_in_threadpool(
        lambda: subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": (proc.stdout or "") + (proc.stderr or ""),
    }


# ---------------------------------------------------------- API: TRENING
def _watch_training(proc: subprocess.Popen, log_file):
    rc = proc.wait()
    try:
        log_file.close()
    except Exception:
        pass
    with _train_lock:
        _train_state["running"] = False
        _train_state["proc"] = None
        _train_state["exit_code"] = rc
        _train_state["finished_at"] = time.time()


@app.post("/api/train/start")
async def api_train_start(
    name: str = Form(...),
    dataset: str = Form(default="tpa"),
    base_model: str = Form(default="yolo26s-pose.pt"),
    epochs: int = Form(default=200),
    imgsz: int = Form(default=640),
    batch: int = Form(default=4),
    patience: int = Form(default=100),
    fliplr: float = Form(default=0.0),
):
    name = safe_name(re.sub(r"[^a-zA-Z0-9._-]", "_", name.strip()))
    dataset = safe_name(dataset)
    base_model = safe_name(base_model)

    data_yaml = DATASETS_DIR / dataset / "dataset.yaml"
    if not data_yaml.exists():
        raise HTTPException(status_code=400, detail=f"Brak pliku {data_yaml} – najpierw przebuduj dataset.")
    model_path = MODELS_DIR / base_model
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Brak modelu bazowego: {model_path}")

    with _train_lock:
        if _train_state["running"]:
            raise HTTPException(status_code=409, detail="Trening już trwa – zatrzymaj go lub poczekaj aż się zakończy.")

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUNS_DIR / f"{name}_train.log"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_v5.py"),
            "--data", str(data_yaml),
            "--model", str(model_path),
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--patience", str(patience),
            "--fliplr", str(fliplr),
            "--project", str(RUNS_DIR),
            "--name", name,
        ]
        log_f = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _train_state.update({
            "running": True,
            "proc": proc,
            "name": name,
            "started_at": time.time(),
            "finished_at": None,
            "exit_code": None,
            "log_path": str(log_path),
            "cmd": " ".join(cmd),
        })
        threading.Thread(target=_watch_training, args=(proc, log_f), daemon=True).start()

    return {"ok": True, "name": name, "cmd": _train_state["cmd"]}


@app.post("/api/train/stop")
async def api_train_stop():
    with _train_lock:
        proc = _train_state.get("proc")
        if not _train_state["running"] or proc is None:
            return {"ok": False, "message": "Żaden trening nie jest uruchomiony."}
        proc.terminate()
    return {"ok": True, "message": "Wysłano sygnał zatrzymania treningu."}


@app.get("/api/train/status")
async def api_train_status():
    with _train_lock:
        state = {k: v for k, v in _train_state.items() if k != "proc"}

    log_tail = ""
    if state.get("log_path") and Path(state["log_path"]).exists():
        raw = Path(state["log_path"]).read_text(encoding="utf-8", errors="ignore")
        # Zamiana \r z pasków postępu na \n i pobranie ostatnich 60 linii
        lines = [ln for ln in raw.replace("\r", "\n").splitlines() if ln.strip()]
        log_tail = "\n".join(lines[-60:])

    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            best_pt = d / "weights" / "best.pt"
            metrics = read_run_metrics(d)
            runs.append({
                "name": d.name,
                "has_best": best_pt.exists(),
                "best_path": str(best_pt.relative_to(ROOT_DIR)) if best_pt.exists() else None,
                "epochs": metrics["epochs"],
                "best_map50_95": metrics["best_map50_95"],
                "last_map50_95": metrics["last_map50_95"],
            })

    return {"state": state, "log_tail": log_tail, "runs": runs}


# --------------------------------------------------------- API: BENCHMARK
@app.post("/api/benchmark")
async def api_benchmark(
    weights_rel: str = Form(...),
    suite: str = Form(default="clean"),
    imgsz: int = Form(default=640),
):
    weights_path = (ROOT_DIR / weights_rel).resolve()
    if not str(weights_path).startswith(str(ROOT_DIR.resolve())) or not weights_path.exists():
        raise HTTPException(status_code=400, detail=f"Nie znaleziono wag: {weights_rel}")

    if suite == "clean":
        images_dir = BENCH_DIR / "img"
        csv_path = BENCH_DIR / "dat.csv"
    elif suite == "lines":
        images_dir = DATASETS_DIR / "tpa_lines" / "images" / "val"
        csv_path = DATA_DIR / "benchmarks" / "bench_lines_gt.csv"
    else:
        raise HTTPException(status_code=400, detail="Dozwolony zbiór: clean lub lines")

    if not images_dir.exists() or not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"Brak danych testowych: {images_dir} lub {csv_path}")

    out_csv = DATA_DIR / "benchmarks" / "last_dashboard_bench.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "bench.py"),
        "--weights", str(weights_path),
        "--images", str(images_dir),
        "--csv", str(csv_path),
        "--imgsz", str(imgsz),
        "--out", str(out_csv),
    ]
    proc = await run_in_threadpool(
        lambda: subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    )
    return {
        "ok": proc.returncode == 0,
        "output": (proc.stdout or "") + (proc.stderr or ""),
    }


# ---------------------------------------------------- API: MODELE I WDROŻENIE
@app.get("/api/models")
async def api_list_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    repo_models = []
    for p in sorted(MODELS_DIR.iterdir()):
        if p.suffix.lower() in {".pt", ".onnx"}:
            repo_models.append({
                "name": p.name,
                "ext": p.suffix.lower(),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                "in_demo": (DEMO_MODELS_DIR / p.name).exists(),
                "rel_path": f"models/{p.name}",
            })

    demo_models = []
    for p in sorted(DEMO_MODELS_DIR.iterdir()):
        if p.suffix.lower() == ".onnx":
            demo_models.append({
                "name": p.name,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
            })

    return {"repo_models": repo_models, "demo_models": demo_models}


@app.post("/api/models/upload")
async def api_upload_model(
    file: UploadFile = File(...),
    deploy_to_demo: bool = Form(default=False),
):
    fname = safe_name(file.filename or "")
    ext = Path(fname).suffix.lower()
    if ext not in {".pt", ".onnx"}:
        raise HTTPException(status_code=400, detail="Dozwolone rozszerzenia modelu: .pt lub .onnx")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw = await file.read()
    (MODELS_DIR / fname).write_bytes(raw)
    deployed = False
    if deploy_to_demo and ext == ".onnx":
        DEMO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (DEMO_MODELS_DIR / fname).write_bytes(raw)
        deployed = True
    return {"ok": True, "name": fname, "deployed_to_demo": deployed}


@app.post("/api/models/export")
async def api_export_and_deploy(
    source_rel: str = Form(...),
    out_name: str = Form(...),
    imgsz: int = Form(default=640),
    deploy_to_demo: bool = Form(default=True),
):
    src_path = (ROOT_DIR / source_rel).resolve()
    if not str(src_path).startswith(str(ROOT_DIR.resolve())) or not src_path.exists():
        raise HTTPException(status_code=400, detail=f"Brak pliku źródłowego: {source_rel}")

    stem = safe_name(re.sub(r"[^a-zA-Z0-9._-]", "_", Path(out_name.strip()).stem))
    pt_dest = MODELS_DIR / f"{stem}.pt"
    onnx_dest = MODELS_DIR / f"{stem}.onnx"
    demo_dest = DEMO_MODELS_DIR / f"{stem}.onnx"

    def _do_export():
        from ultralytics import YOLO

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if src_path != pt_dest.resolve():
            shutil.copy2(src_path, pt_dest)
        exported = YOLO(str(pt_dest)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False)
        shutil.copy2(exported, onnx_dest)
        if deploy_to_demo:
            DEMO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exported, demo_dest)
        return str(onnx_dest)

    exported_path = await run_in_threadpool(_do_export)
    return {
        "ok": True,
        "pt": f"models/{stem}.pt",
        "onnx": f"models/{stem}.onnx",
        "deployed_to_demo": deploy_to_demo,
        "path": exported_path,
    }


@app.post("/api/models/deploy/{filename}")
async def api_deploy_existing_onnx(filename: str):
    filename = safe_name(filename)
    src = MODELS_DIR / filename
    if not src.exists() or src.suffix.lower() != ".onnx":
        raise HTTPException(status_code=404, detail="Brak pliku .onnx w models/")
    DEMO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, DEMO_MODELS_DIR / filename)
    return {"ok": True, "deployed": filename}


@app.delete("/api/models/demo/{filename}")
async def api_remove_from_demo(filename: str):
    filename = safe_name(filename)
    target = DEMO_MODELS_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Brak pliku w demo/models/")
    target.unlink()
    return {"ok": True, "removed": filename}


@app.get("/api/models/download/{scope}/{filename}")
async def api_download_model(scope: str, filename: str):
    filename = safe_name(filename)
    folder = DEMO_MODELS_DIR if scope == "demo" else MODELS_DIR
    path = folder / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Brak pliku modelu")
    return FileResponse(path, filename=filename)
