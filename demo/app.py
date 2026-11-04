import base64
import io
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "kostek-v4s.onnx"

IMGSZ = 1024        # używane tylko, gdy model ma dynamiczny rozmiar wejścia
CONF_THRES = 0.25  # minimalna pewność detekcji

NAMES = [
    "P1",
    "P2",
    "P3",
    "P4",
]
NK = len(NAMES)

# ---------------------------------------------------------------- model ONNX
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Brak modelu: {MODEL_PATH}\nWyeksportuj wagi: python export_onnx.py --weights best.pt"
    )

providers = [
    p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
    if p in ort.get_available_providers()
]
session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
input_name = session.get_inputs()[0].name
_shape = session.get_inputs()[0].shape  # [1, 3, H, W] albo nazwy symboliczne przy dynamicznym
if all(isinstance(v, int) for v in _shape[2:]):
    MODEL_H, MODEL_W = _shape[2], _shape[3]
else:
    MODEL_H = MODEL_W = IMGSZ


def preprocess(img: Image.Image):
    """Letterbox jak w Ultralytics: zachowanie proporcji + szare pola (114)."""
    arr = np.asarray(img)  # RGB, HxWx3
    h, w = arr.shape[:2]
    scale = min(MODEL_H / h, MODEL_W / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw, dh = (MODEL_W - nw) / 2, (MODEL_H - nh) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)

    canvas = np.full((MODEL_H, MODEL_W, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), scale, left, top


def detect(img: Image.Image):
    """Zwraca None albo (xs, ys, conf) dla punktów P1..P4 w pikselach oryginału."""
    w, h = img.size
    blob, scale, left, top = preprocess(img)
    out = session.run(None, {input_name: blob})[0][0]

    if out.shape[0] < out.shape[1]:
        # klasyczny eksport: [4 + 1 + NK*3, kotwice] -> [cx, cy, w, h, conf, kpts...]
        out = out.T
        conf, kpts = out[:, 4], out[:, 5:]
    else:
        # eksport end-to-end (bez NMS): [x1, y1, x2, y2, conf, cls, kpts...]
        conf, kpts = out[:, 4], out[:, 6:]

    i = int(np.argmax(conf))  # jedna kość na zdjęciu = najpewniejsza detekcja
    if conf[i] < CONF_THRES:
        return None

    k = kpts[i][: NK * 3].reshape(NK, 3)
    xs = np.clip((k[:, 0] - left) / scale, 0, w)
    ys = np.clip((k[:, 1] - top) / scale, 0, h)
    return xs, ys, k[:, 2]


# ------------------------------------------------------------------ aplikacja
app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Wskazanie folderu z szablonami HTML
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()  # awaryjnie: polskie znaki mogą się nie wyświetlić


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Moja strona FastAPI", "greeting": "Witaj na stronie!"},
    )


@app.get("/waiting", response_class=HTMLResponse)
async def waiting(request: Request):
    return templates.TemplateResponse(request=request, name="waiting.html")


@app.post("/api/submit")
async def handle_form(request: Request, file: UploadFile = File(...)):

    # 1. Odczytanie przesłanych bajtów z pamięci
    contents = await file.read()

    # 2. Załadowanie bajtów jako obraz Pillow (z poprawką orientacji EXIF, w RGB)
    input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")

    # 3. Inferencja ONNX (w wątku, żeby nie blokować serwera)
    det = await run_in_threadpool(detect, input_image)

    # 4. Rysowanie punktów, linii i nazw na kopii obrazu + zebranie współrzędnych do tabeli
    annotated_image = input_image.copy()
    points = []

    if det is not None:
        xs, ys, confs = det
        pts = [(float(x), float(y)) for x, y in zip(xs, ys)]

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
        s = max(2, annotated_image.width // 300)               # grubość linii
        font = get_font(max(18, annotated_image.width // 40))

        draw.line([pts[0], pts[1]], fill=(255, 200, 0), width=s)   # oś piszczeli P1-P2
        draw.line([pts[2], pts[3]], fill=(0, 255, 0), width=s)     # plateau P3-P4

        for name, (x, y) in zip(NAMES, pts):
            rad = s * 3
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 0, 0))
            draw.text((x + rad + 6, y - rad), name, font=font, fill=(255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))

    # 5. Zapisanie obrazu do bufora w pamięci RAM
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")
    output_bytes = buffer.getvalue()

    # 6. Kodowanie do Base64
    encoded_string = base64.b64encode(output_bytes).decode("utf-8")

    # 7. Wstrzyknięcie danych do szablonu HTML
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "image_base64": encoded_string,
            "mime_type": "image/jpeg",
            "points": points,
            "has_conf": bool(points),
            "width": input_image.width,
            "height": input_image.height,
        },
    )
