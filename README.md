# Kostek

Aplikacja webowa (FastAPI) oraz zestaw narzędzi ML do automatycznego wykrywania punktów anatomicznych kości piszczelowej na zdjęciach RTG przy użyciu modeli **YOLO-pose**. Projekt wspomaga wyznaczanie osi anatomicznych wykorzystywanych m.in. przy pomiarze kąta nachylenia plateau kości piszczelowej (**TPA** – *Tibial Plateau Angle*) w ortopedii weterynaryjnej.

---

## Funkcjonalności

- **Detekcja 4 punktów kluczowych (`P1`–`P4`)** na zdjęciach rentgenowskich:
  - **Oś długa piszczeli (`P1`–`P2`)** – od środka kości skokowej (*talus*, `P1`) do wyniosłości międzykłykciowej (*eminence*, `P2`), oznaczana na obrazie kolorem żółtym.
  - **Linia plateau piszczeli (`P3`–`P4`)** – od krawędzi doczaszkowej (*cranial plateau*, `P3`) do krawędzi doogonowej (*caudal plateau*, `P4`), oznaczana kolorem zielonym.
- **Przetwarzanie w pamięci RAM** – przesłane zdjęcia nie są zapisywane na dysku serwera; aplikacja automatycznie koryguje orientację EXIF i zwraca wynik zakodowany w Base64 wraz z tabelą współrzędnych (`X`, `Y` w pikselach oryginału) oraz pewnością detekcji (`%`).
- **Dwa tryby uruchomieniowe**:
  - **Lekka paczka wdrożeniowa (`demo/`)** – oparta o **ONNX Runtime** (bez konieczności instalowania PyTorcha i `ultralytics`), z gotowymi skryptami instalacyjnymi dla systemów Linux, macOS i Windows.
  - **Środowisko deweloperskie (katalog główny)** – oparte bezpośrednio o **Ultralytics YOLO** i wagi PyTorch (`.pt`), wraz ze skryptami do przygotowania danych, treningu, eksportu i benchmarkingu.

---

## Struktura projektu

```text
kostek/
├── demo/                   # Samodzielna wersja demonstracyjna/wdrożeniowa (ONNX Runtime)
│   ├── app.py              # Serwer FastAPI z inferencją ONNX (kostek-v4s.onnx)
│   ├── models/             # Wagi modelu dla wersji demo (.onnx, .pt)
│   ├── setup.sh / .bat     # Skrypty tworzące .venv i instalujące zależności
│   ├── run.sh / .bat       # Skrypty uruchamiające serwer Uvicorn
│   ├── requirements.txt    # Lekkie zależności (bez PyTorcha)
│   ├── static/             # Arkusze stylów CSS
│   └── templates/          # Szablony HTML (Jinja2)
├── models/                 # Wytrenowane wagi (.pt, .onnx) oraz bazowy model YOLO-pose
├── scripts/                # Skrypty pomocnicze ML
│   ├── data.py             # Konwersja adnotacji z CSV do formatu YOLO-pose (train/val)
│   ├── main.py             # Skrypt treningowy YOLO-pose
│   ├── export.py           # Pomocniczy eksport wag do ONNX
│   └── bench.py            # Benchmark dokładności predykcji względem Ground Truth (CSV)
├── static/                 # Arkusze stylów dla głównej aplikacji
├── templates/              # Szablony HTML dla głównej aplikacji
├── app.py                  # Aplikacja FastAPI korzystająca z wag PyTorch (.pt)
├── export_onnx.py          # Narzędzie CLI do eksportu wag .pt -> .onnx
└── requirements.txt        # Zależności środowiska deweloperskiego
```

---

## Szybki start (wersja `demo` – zalecana)

Wersja w katalogu `demo/` nie wymaga kart graficznych ani ciężkich bibliotek treningowych (korzysta z `onnxruntime` na CPU lub GPU CUDA, jeśli jest dostępne).

### Wymagania
- **Python 3.10** lub nowszy (na systemach Debian/Ubuntu również pakiet `python3-venv`).

### Linux / macOS

```bash
cd demo
bash setup.sh
bash run.sh
```

### Windows (CMD / PowerShell)

```bat
cd demo
setup.bat
run.bat
```

Po uruchomieniu aplikacja jest dostępna w przeglądarce pod adresem:
**http://127.0.0.1:8000**

> **Konfiguracja hosta i portu:**
> Możesz zmienić domyślny adres nasłuchiwania za pomocą zmiennych środowiskowych `APP_HOST` oraz `APP_PORT`:
> - Linux / macOS: `APP_HOST=0.0.0.0 APP_PORT=9000 bash run.sh`
> - Windows (CMD): `set APP_HOST=0.0.0.0 & set APP_PORT=9000 & run.bat`

---

## Środowisko deweloperskie (PyTorch / Ultralytics)

Jeśli chcesz rozwijać model, trenować nowe wagi lub uruchomić prototyp bezpośrednio na pliku `.pt` z katalogu głównego:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Uruchomienie serwera deweloperskiego:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

---

## Trening, eksport i ewaluacja modelu

### 1. Przygotowanie zbioru danych
Skrypt `scripts/data.py` konwertuje współrzędne punktów `P1`–`P4` z pliku `Cases.csv` oraz obrazy źródłowe do struktury wymaganej przez Ultralytics YOLO-pose (`images/`, `labels/`, `data.yaml`) z podziałem na zbiór treningowy i walidacyjny:

```bash
python scripts/data.py
```

### 2. Trening modelu
Uruchomienie treningu bazowego modelu `yolo26n-pose.pt` (domyślnie `imgsz=1024`, 500 epok):

```bash
python scripts/main.py
```

### 3. Eksport wag `.pt` do formatu `.onnx`
Po zakończeniu treningu wyeksportuj najlepsze wagi do formatu ONNX (pamiętaj, aby podać ten sam parametr `--imgsz`, który był użyty podczas treningu):

```bash
python export_onnx.py \
  --weights runs/pose/lines/weights/best.pt \
  --imgsz 1024 \
  --out demo/models/kostek-v4s.onnx
```

### 4. Benchmark dokładności (`scripts/bench.py`)
Skrypt porównuje punkty przewidziane przez model z ręcznymi oznaczeniami z pliku CSV (na zdjęciach spoza zbioru treningowego), wylicza błędy odległości w pikselach (średnia, mediana, 90. percentyl, max, odsetek trafień poniżej progów) oraz opcjonalnie generuje wizualizacje porównawcze (zielony = Ground Truth, czerwony = predykcja):

```bash
python scripts/bench.py \
  --weights models/kostek-v4s.pt \
  --images bench/img \
  --csv bench/dat.csv \
  --imgsz 1024 \
  --save-vis bench/vis \
  --out benchmark_results.csv
```

---

## Technologie

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), Jinja2
- **Detekcja / ML:** [Ultralytics YOLO-pose](https://docs.ultralytics.com/tasks/pose/), [ONNX Runtime](https://onnxruntime.ai/)
- **Przetwarzanie obrazu:** Pillow (PIL), OpenCV, NumPy
