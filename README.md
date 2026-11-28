# Kostek

Aplikacja webowa (FastAPI) oraz zestaw narzędzi ML do automatycznego wykrywania punktów anatomicznych kości piszczelowej na zdjęciach RTG i wyliczania kąta nachylenia plateau kości piszczelowej (**TPA** – *Tibial Plateau Angle*) przy użyciu modeli **YOLO-pose**.

---

## Funkcjonalności

- **Detekcja 4 punktów kluczowych (`P1`–`P4`) i automatyczny pomiar kąta TPA:**
  - **Oś długa piszczeli (`P1`–`P2`)** – środek kości skokowej (*talus*, `P1`) → wyniosłość międzykłykciowa (*eminence*, `P2`), rysowana na żółto.
  - **Linia plateau piszczeli (`P3`–`P4`)** – krawędź doczaszkowa (*cranial plateau*, `P3`) → krawędź doogonowa (*caudal plateau*, `P4`), rysowana na zielono.
  - **Kąt TPA (`°`)** – wyliczany automatycznie z wektorów `P1→P2` i `P3→P4` jako odchylenie od `90°`, nanoszony na obraz wynikowy i wyświetlany w karcie podsumowania.
- **Wybór modelu ONNX z listy:** możliwość przełączania w formularzu między `kostek-v5s.onnx` (domyślny, czyste RTG), `kostek-lines.onnx` (RTG z liniami pomiarowymi, z automatycznym przycinaniem czarnych pasów bocznych) oraz `kostek-v4s.onnx`.
- **Podgląd porównawczy i eksport w pamięci RAM:**
  - Porównanie **Oryginał** vs **Wynik detekcji** obok siebie (bez zapisywania zdjęć pacjentów na dysku serwera).
  - Pobieranie przetworzonego zdjęcia (`.jpg`) oraz raportu **Excel (`.xlsx`)** z kątem TPA, współrzędnymi `P1–P4`, pewnością detekcji i nazwą modelu.
- **Lokalny ML Dashboard (`dashboard/`):** panel webowy na porcie `8001` do przesyłania nowych danych (zdjęcia / `Cases.csv` / ZIP), podglądu adnotacji, uruchamiania treningów na GPU, benchmarków i eksportu wag `.pt` → `.onnx` wprost do `demo/models/`.

---

## Jak uruchomić demo

W katalogu `demo/` znajduje się samodzielna wersja demonstracyjna oparta o **ONNX Runtime** – nie wymaga karty graficznej ani instalowania ciężkich bibliotek treningowych (PyTorch / Ultralytics).

### Wymagania wstępne
- **Python 3.10** lub nowszy zainstalowany w systemie.
  - *Linux (Debian / Ubuntu / Mint / Pop!_OS):* upewnij się, że masz pakiet `venv`: `sudo apt install python3-venv`
  - *Windows:* podczas instalacji Pythona z [python.org](https://www.python.org/downloads/) zaznacz opcję **„Add python.exe to PATH”**.

### Krok 1: Pobranie projektu i wejście do katalogu `demo`

```bash
git clone https://github.com/Hikori05/kostek.git
cd kostek/demo
```

### Krok 2: Jednorazowa instalacja środowiska (`setup`)

Skrypt automatycznie utworzy lokalne środowisko wirtualne `.venv` i zainstaluje lekkie zależności z `demo/requirements.txt`.

- **Linux / macOS:**
  ```bash
  bash setup.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  .\setup.bat
  ```
- **Windows (CMD lub Eksplorator plików):**
  ```bat
  setup.bat
  ```

### Krok 3: Uruchomienie serwera aplikacji (`run`)

- **Linux / macOS:**
  ```bash
  bash run.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  .\run.bat
  ```
- **Windows (CMD / Eksplorator plików):**
  ```bat
  run.bat
  ```

### Krok 4: Korzystanie z aplikacji w przeglądarce

1. Otwórz w przeglądarce adres **http://127.0.0.1:8000**.
2. Wybierz model z listy (domyślnie `kostek-v5s.onnx`) i wskaż zdjęcie RTG kości piszczelowej.
3. Kliknij **„Przetwórz w pamięci”** – zobaczysz oryginał obok wyniku, wyliczony **kąt TPA**, tabelę współrzędnych oraz przyciski **Pobierz obraz (JPG)** i **Pobierz raport (Excel)**.
4. Aby zatrzymać serwer, naciśnij `Ctrl + C` w terminalu.

> **Zmiana hosta/portu:**
> - Linux / macOS: `APP_HOST=0.0.0.0 APP_PORT=9000 bash run.sh`
> - Windows (PowerShell): `$env:APP_HOST="0.0.0.0"; $env:APP_PORT="9000"; .\run.bat`

---

## Wyniki modeli (zbiór walidacyjny)

| Model | Zbiór testowy | Błąd `P1` (kostka) | Średni błąd `ALL` | Średni błąd kąta TPA | Czas inferencji |
|---|---|---:|---:|---:|---:|
| `kostek-v4s` (`1024 px`) | czyste RTG | `671.1 px` | `196.3 px` (`7.11%`) | `17.70°` (med. `18.23°`) | `73 ms` |
| **`kostek-v5s`** (`640 px`) | czyste RTG | **`191.2 px`** | **`102.1 px`** (`3.72%`) | **`10.60°`** (med. **`9.96°`**) | **`44 ms`** |
| **`kostek-lines`** (`640 px`) | wykadrowane RTG z liniami | `217.9 px` | **`74.6 px`** (med. `36.4 px`) | `23.02°` (med. `25.02°`) | **`31 ms`** |

---

## Struktura projektu

```text
kostek/
├── demo/                         # Samodzielna wersja demonstracyjna (ONNX Runtime)
│   ├── app.py                    # Serwer FastAPI: wybór modelu, kąt TPA, eksport .xlsx
│   ├── models/                   # Aktywne modele ONNX (kostek-v5s, kostek-lines, kostek-v4s)
│   ├── setup.sh / setup.bat      # Tworzenie .venv i instalacja zależności
│   ├── run.sh / run.bat          # Uruchamianie serwera Uvicorn (port 8000)
│   ├── requirements.txt          # Lekkie zależności (bez PyTorcha)
│   ├── static/ & templates/      # Style CSS i szablony HTML (Jinja2)
├── dashboard/                    # Lokalny panel ML (dane, podgląd, trening GPU, eksport ONNX)
│   ├── app.py                    # Backend panelu (FastAPI)
│   ├── run.sh                    # Uruchamianie panelu na porcie 8001
│   └── templates/index.html      # Interfejs 4-zakładkowy
├── models/                       # Wagi wytrenowane (.pt, .onnx) oraz bazowe (yolo26*-pose.pt)
├── scripts/                      # Skrypty ML (wszystkie w jednym miejscu)
│   ├── prep_dat.py               # Budowa czystego zbioru YOLO-pose (podział po case_XXXX)
│   ├── map_points_to_lines.py    # Przenoszenie punktów .png -> .jpg + kadrowanie czarnych pasów
│   ├── train_v5.py               # Ulepszony skrypt treningowy YOLO-pose (mosaic=0, box=7.5)
│   ├── bench.py                  # Benchmark błędów w px oraz błędu kąta TPA w stopniach
│   └── data.py / main.py / export.py / trai.py
├── KartaAudytu/                  # Dokumenty audytu projektu
├── Dokumentacja_projektu_AI.pdf  # Pełna dokumentacja projektowa (PDF)
├── app.py                        # Prototypowa aplikacja FastAPI (PyTorch / Ultralytics)
├── export_onnx.py                # Narzędzie CLI do eksportu wag .pt -> .onnx
└── requirements.txt              # Zależności środowiska deweloperskiego
```

---

## Panel ML (`dashboard/`) i trening z linii poleceń

Do treningu i zarządzania danymi użyj głównego środowiska `.venv` (z PyTorchem, CUDA i Ultralytics):

```bash
# Uruchomienie lokalnego ML Dashboardu (http://127.0.0.1:8001)
bash dashboard/run.sh
```

Lub bezpośrednio z linii poleceń:

```bash
# 1. Przygotowanie zbiorów danych
python scripts/prep_dat.py --src data/dataset_clean --out data/datasets/tpa --val-cases 2 --pad 0.08
python scripts/map_points_to_lines.py --src data/dataset --csv data/dataset/Cases.csv --out data/datasets/tpa_lines

# 2. Trening modeli (czyste RTG / RTG z liniami)
python scripts/train_v5.py --data data/datasets/tpa/dataset.yaml --model models/yolo26s-pose.pt --name kostek-v5s
python scripts/train_v5.py --data data/datasets/tpa_lines/dataset.yaml --model models/yolo26s-pose.pt --name kostek-lines

# 3. Benchmark (błąd w px + błąd kąta TPA w stopniach)
python scripts/bench.py --weights models/kostek-v5s.pt --images bench/img --csv bench/dat.csv --imgsz 640

# 4. Eksport wag .pt -> .onnx
python export_onnx.py --weights models/kostek-v5s.pt --imgsz 640 --out demo/models/kostek-v5s.onnx
```

---

## Technologie

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), Jinja2, OpenPyXL
- **Detekcja / ML:** [Ultralytics YOLO-pose](https://docs.ultralytics.com/tasks/pose/), [ONNX Runtime](https://onnxruntime.ai/), Albumentations
- **Przetwarzanie obrazu:** Pillow (PIL), OpenCV, NumPy
