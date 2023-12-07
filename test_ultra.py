from ultralytics import YOLO

model = YOLO("yolov8m.pt")
link = "D:/BaseProject/test_video6.mp4"
results = model(show=True, source=link, device = '0')
