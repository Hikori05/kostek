import base64
import io
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL = "kostek-v5s.onnx"

IMGSZ = 640         # używane tylko, gdy model ma dynamiczny rozmiar wejścia
CONF_THRES = 0.25   # minimalna pewność detekcji

NAMES = ["P1", "P2", "P3", "P4"]
NK = len(NAMES)

MODEL_LABELS = {
    "kostek-v5s.onnx": "Kostek v5s (nowy – czyste RTG, domyślny)",
    "kostek-lines.onnx": "Kostek lines (nowy – RTG z liniami)",
    "kostek-v4s.onnx": "Kostek v4s (poprzednia wersja – 1024 px)",
}

PROVIDERS = [
    p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
    if p in ort.get_available_providers()
]
_SESSION_CACHE: dict[str, tuple[ort.InferenceSession, str, int, int]] = {}


def list_models() -> list[dict[str, str]]:
    files = sorted(p.name for p in MODELS_DIR.glob("*.onnx"))
    if DEFAULT_MODEL in files:
        files.remove(DEFAULT_MODEL)
        files.insert(0, DEFAULT_MODEL)
    return [
        {"file": name, "label": MODEL_LABELS.get(name, name)}
        for name in files
    ]


def get_model(model_file: str) -> tuple[ort.InferenceSession, str, int, int]:
    available = {m["file"] for m in list_models()}
    if not available:
        raise FileNotFoundError(
            f"Brak modeli .onnx w {MODELS_DIR}\nWyeksportuj wagi: python export_onnx.py --weights best.pt"
        )
    if model_file not in available:
        raise HTTPException(status_code=400, detail=f"Nieznany model: {model_file}")

    if model_file not in _SESSION_CACHE:
        model_path = MODELS_DIR / model_file
        sess = ort.InferenceSession(str(model_path), providers=PROVIDERS)
        inp_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        if all(isinstance(v, int) for v in shape[2:]):
            mh, mw = shape[2], shape[3]
        else:
            mh = mw = IMGSZ
        _SESSION_CACHE[model_file] = (sess, inp_name, mh, mw)
    return _SESSION_CACHE[model_file]


def preprocess(img: Image.Image, model_h: int, model_w: int):
    """Letterbox jak w Ultralytics: zachowanie proporcji + szare pola (114)."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    scale = min(model_h / h, model_w / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw, dh = (model_w - nw) / 2, (model_h - nh) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)

    canvas = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), scale, left, top


def detect(img: Image.Image, model_file: str):
    """Zwraca None albo (xs, ys, conf) dla punktów P1..P4 w pikselach oryginału."""
    sess, inp_name, model_h, model_w = get_model(model_file)
    w, h = img.size
    blob, scale, left, top = preprocess(img, model_h, model_w)
    out = sess.run(None, {inp_name: blob})[0][0]

    if out.shape[0] < out.shape[1]:
        out = out.T
        conf, kpts = out[:, 4], out[:, 5:]
    else:
        conf, kpts = out[:, 4], out[:, 6:]

    i = int(np.argmax(conf))
    if conf[i] < CONF_THRES:
        return None

    k = kpts[i][: NK * 3].reshape(NK, 3)
    xs = np.clip((k[:, 0] - left) / scale, 0, w)
    ys = np.clip((k[:, 1] - top) / scale, 0, h)
    return xs, ys, k[:, 2]


def calc_tpa(pts):
    """Wylicza kąt TPA oraz kąt między wektorami P1->P2 i P3->P4 (wzór z Cases.xlsx)."""
    if len(pts) < 4:
        return None, None
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[:4]
    v_tibia = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    v_plateau = np.array([x4 - x3, y4 - y3], dtype=np.float64)
    denom = float(np.linalg.norm(v_tibia) * np.linalg.norm(v_plateau))
    if denom < 1e-9:
        return None, None
    cos_a = float(np.clip(np.dot(v_tibia, v_plateau) / denom, -1.0, 1.0))
    theta = float(np.degrees(np.arccos(cos_a)))
    tpa = theta - 90.0
    return round(tpa, 1), round(theta, 1)


def build_excel_base64(filename: str, model_label: str, width: int, height: int,
                       tpa_angle, raw_angle, points: list[dict]) -> str:
    wb = Workbook()
    ws = wb.active
    ws.title = "Pomiar TPA"

    ws.append(["Parametr", "Wartość"])
    ws.append(["Plik", filename])
    ws.append(["Model", model_label])
    ws.append(["Szerokość [px]", width])
    ws.append(["Wysokość [px]", height])
    ws.append(["Kąt TPA [°]", tpa_angle if tpa_angle is not None else "brak detekcji"])
    ws.append(["Kąt między osiami [°]", raw_angle if raw_angle is not None else "brak detekcji"])
    ws.append([])
    ws.append(["Punkt", "X [px]", "Y [px]", "Pewność [%]"])
    for p in points:
        ws.append([p["name"], p["x"], p["y"], p["conf"]])

    buf = io.BytesIO()
    wb.save(buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------------------------------------------------------------ aplikacja
app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Kostek – pomiar TPA",
            "models": list_models(),
        },
    )


@app.post("/api/submit")
async def handle_form(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default=""),
):
    models = list_models()
    if not models:
        raise HTTPException(status_code=500, detail="Brak plików .onnx w folderze demo/models/")
    selected_model = model_name if model_name else models[0]["file"]

    contents = await file.read()
    input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")

    det = await run_in_threadpool(detect, input_image, selected_model)

    annotated_image = input_image.copy()
    points = []
    tpa_angle = None
    raw_angle = None

    if det is not None:
        xs, ys, confs = det
        pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
        tpa_angle, raw_angle = calc_tpa(pts)

        points = [
            {
                "name": name,
                "x": round(x, 1),
                "y": round(y, 1),
                "conf": int(round(min(max(float(c), 0.0), 1.0) * 100)),
            }
            for name, (x, y), c in zip(NAMES, pts, confs)
        ]

        draw = ImageDraw.Draw(annotated_image)
        s = max(2, annotated_image.width // 300)
        font = get_font(max(18, annotated_image.width // 40))

        draw.line([pts[0], pts[1]], fill=(255, 200, 0), width=s)
        draw.line([pts[2], pts[3]], fill=(0, 255, 0), width=s)

        for name, (x, y) in zip(NAMES, pts):
            rad = s * 3
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 0, 0))
            draw.text((x + rad + 6, y - rad), name, font=font, fill=(255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))

        if tpa_angle is not None:
            label = f"TPA: {tpa_angle:.1f}°"
            draw.text((20, 20), label, font=get_font(max(24, annotated_image.width // 28)),
                      fill=(56, 189, 248), stroke_width=3, stroke_fill=(0, 0, 0))

    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")
    encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    orig_buf = io.BytesIO()
    input_image.save(orig_buf, format="JPEG")
    orig_base64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")

    model_label = MODEL_LABELS.get(selected_model, selected_model)
    excel_base64 = build_excel_base64(
        filename=file.filename or "obraz.jpg",
        model_label=model_label,
        width=input_image.width,
        height=input_image.height,
        tpa_angle=tpa_angle,
        raw_angle=raw_angle,
        points=points,
    )
    stem = Path(file.filename or "wynik").stem

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "image_base64": encoded_string,
            "orig_image_base64": orig_base64,
            "excel_base64": excel_base64,
            "download_stem": stem,
            "mime_type": "image/jpeg",
            "points": points,
            "has_conf": bool(points),
            "width": input_image.width,
            "height": input_image.height,
            "tpa_angle": tpa_angle,
            "raw_angle": raw_angle,
            "model_used": model_label,
        },
    )
