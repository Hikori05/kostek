from ultralytics import YOLO

YOLO('./runs/pose/clean/weights/best.pt').export(format="onnx", imgsz=1024, nms=False)
