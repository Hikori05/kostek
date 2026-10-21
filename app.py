from ultralytics import YOLO

model = YOLO("runs/pose/train-2/weights/best.pt")

results = model("./test.jpg")

for r in results:
    r.save(filename="result.jpg")

# import cv2
# from pathlib import Path
#
# img_dir = Path("dataset/images/train")
# lab_dir = Path("dataset/labels/train")
#
# missing = []
# for img_path in list(img_dir.glob("*"))[:20]:
#     lab_path = lab_dir / (img_path.stem + ".txt")
#     if not lab_path.exists():
#         missing.append(img_path.name)
#         continue
#     img = cv2.imread(str(img_path))
#     h, w = img.shape[:2]
#     kp = [float(v) for v in lab_path.read_text().split()[5:]]
#     pts = [(int(kp[i]*w), int(kp[i+1]*h)) for i in range(0, 12, 3)]
#     for n, p in enumerate(pts, 1):
#         cv2.circle(img, p, 8, (0, 255, 0), -1)
#         cv2.putText(img, str(n), (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imwrite(f"check_{img_path.stem}.jpg", img)
#
# print("bez etykiety:", missing)
