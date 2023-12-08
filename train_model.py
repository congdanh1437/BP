from ultralytics import YOLO

model = YOLO("yolov8n.pt")
if __name__ == '__main__':
    result = model.train(data="data/mydataset.yaml", epochs=1, device="0", batch=6, workers=4)
# result = model.predict(source=0, show=True)
