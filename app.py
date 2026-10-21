import base64
import io

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from ultralytics.models.yolo.model import YOLO

model = YOLO("./models/yolo26n-pose.pt")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Wskazanie folderu z szablonami HTML
templates = Jinja2Templates(directory="templates")


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

    # 2. Załadowanie bajtów jako obraz Pillow
    input_image = Image.open(io.BytesIO(contents))

    # 3. Przekazanie obrazu do YOLO (YOLO obsługuje bezpośrednio obiekty PIL.Image)
    results = model(input_image)

    # 4. Wyrenderowanie obrazu z naniesionymi punktami (keypoints)
    # r.plot() zwraca tablicę numpy (BGR) z naniesionymi detekcjami
    annotated_frame = results[0].plot()

    # 5. Konwersja tablicy OpenCV/Numpy (BGR -> RGB) do obiektu PIL Image
    annotated_image = Image.fromarray(annotated_frame[..., ::-1])

    # 6. Zapisanie przetworzonego obrazu do bufora bajtów w pamięci RAM (JPEG/PNG)
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")
    output_bytes = buffer.getvalue()

    # 7. Kodowanie do Base64 i wysłanie do szablonu
    encoded_string = base64.b64encode(output_bytes).decode("utf-8")

    # 3. Wstrzyknięcie danych bezpośrednio do szablonu HTML
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"image_base64": encoded_string},
    )
