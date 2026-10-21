from ultralytics import YOLO

model = YOLO("models/yolo26n-pose.pt")   # mały model, mieści się w 4 GB VRAM
model.train(
    data="data.yaml",
    epochs=150,
    imgsz=640,
    batch=8,          # jeśli braknie pamięci GPU: 4
    patience=30,      # zatrzymaj, gdy przestaje się poprawiać
    fliplr=0.0,       # bez odbić lewo-prawo (kolejność punktów)
    device=0,
)
