from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(show=True, source="test_video6.mp4")
