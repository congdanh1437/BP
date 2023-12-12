import cv2
import numpy as np
import os

# Đường dẫn thư mục chứa hình ảnh
image_directory = "data2/images"

# Đường dẫn thư mục chứa nhãn
label_directory = "data2/labels"

# Tạo thư mục nhãn nếu chưa tồn tại
if not os.path.exists(label_directory):
    os.makedirs(label_directory)

# Load mô hình YOLOv3 và các trọng số
net = cv2.dnn.readNet("model/yolov4-tiny.weights", "model/yolov4-tiny.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Lặp qua từng tệp hình ảnh trong thư mục
for image_file in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_file)

    # Đọc hình ảnh
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Chuẩn bị hình ảnh để đưa vào mô hình
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Chạy mô hình để nhận diện đối tượng
    detections = net.forward(layer_names)

    # Xác định đối tượng là người (có thể cần điều chỉnh theo class của YOLO)
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Xác định đối tượng là người
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Tạo nhãn từ kết quả nhận diện
                label_content = f"person {center_x} {center_y} {w} {h}\n"

                # Lưu nhãn vào file
                label_file = os.path.join(label_directory, f"{image_file.split('.')[0]}.txt")
                with open(label_file, 'w') as label_writer:
                    label_writer.write(label_content)