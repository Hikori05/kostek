from ultralytics import YOLO

model = YOLO("./models/yolo26n-pose.pt")
model.train(
    data="dataset_lines/data.yaml",
    epochs=500,
    imgsz=1024,
    batch=2,
    patience=100,
    degrees=10,
    scale=0.3,
    translate=0.1,
    hsv_v=0.4,
    workers=2,
   device=0,
    name="lines",
)
