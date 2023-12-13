from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")
# if __name__ == '__main__':
#     result = model.train(data="data/mydataset.yaml", epochs=100, device="0", batch=2, workers=2)
result = model.predict(source="fire000010.jpg", show=True, save=True)
