import base64
import io

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics.models.yolo.model import YOLO

# Ustaw tak samo jak imgsz użyty przy treningu (domyślne 640 psuje dokładność punktów)
IMGSZ = 800

model = YOLO("./models/kostek-v3s.pt")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Wskazanie folderu z szablonami HTML
templates = Jinja2Templates(directory="templates")

NAMES = [
    "P1",
    "P2",
    "P3",
    "P4",
]


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

    # 3. Przekazanie obrazu do YOLO
    results = model(input_image, imgsz=IMGSZ, verbose=False)
    r = results[0]

    # 4. Rysowanie punktów, linii i nazw na kopii obrazu + zebranie współrzędnych do tabeli
    annotated_image = input_image.copy()
    points = []

    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        xy = r.keypoints.xy[0].tolist()
        confs = (
            r.keypoints.conf[0].tolist()
            if r.keypoints.conf is not None
            else [None] * len(xy)
        )
        pts = [(x, y) for x, y in xy]

        points = [
            {
                "name": name,
                "x": round(x, 1),
                "y": round(y, 1),
                "conf": None if c is None else round(c * 100),
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
            "has_conf": any(p["conf"] is not None for p in points),
            "width": input_image.width,
            "height": input_image.height,
        },
    )