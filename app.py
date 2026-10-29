import base64
import io
import math

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics.models.yolo.model import YOLO

model = YOLO("./runs/pose/lines/weights/best.pt")

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


def tpa(p):
    a_axis = math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0])
    a_plat = math.atan2(p[3][1] - p[2][1], p[3][0] - p[2][0])
    return math.degrees(a_plat - a_axis) % 180 - 90


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
    results = model(input_image)
    r = results[0]

    # 4. Rysowanie punktów, linii, nazw i kąta TPA na kopii obrazu
    annotated_image = input_image.copy()
    tpa_value = None

    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        p = r.keypoints.xy[0].tolist()
        pts = [(x, y) for x, y in p]
        tpa_value = tpa(p)

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

        draw.text((20, 20), f"TPA: {tpa_value:.1f}°", font=font, fill=(255, 255, 255),
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
            "tpa": f"{tpa_value:.1f}" if tpa_value is not None else None,
        },
    )
