from yolodetect import YoloDetect
import cv2
import numpy as np
import json


link = "D:/BaseProject/test_video6.mp4"
cap = cv2.VideoCapture(link)

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

width = 1080
height = 720
cap.set(3, width)  # ID 3 là chiều rộng của video
cap.set(4, height)  # ID 4 là chiều cao của video
points = []

# Check if there are saved points in the file
points_file_path = "D:/BaseProject/BP/saved_points.json"
try:
    with open(points_file_path, 'r') as file:
        points = json.load(file)
except FileNotFoundError:
    pass

model_detect = YoloDetect()

def handle_left_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([x, y])

def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)
    return frame

detect = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc frame từ camera.")
        break

    resized_frame = cv2.resize(frame, (width, height))
    resized_frame = draw_polygon(resized_frame, points)

    if detect:
        resized_frame = model_detect.detect(frame=resized_frame, points=points)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])
        detect = True
    elif key == ord('s'):
        # Save the points to a JSON file
        with open(points_file_path, 'w') as file:
            json.dump(points, file)

    cv2.imshow("YOLOv8", resized_frame)

    cv2.setMouseCallback('YOLOv8', handle_left_click, points)

cap.release()
cv2.destroyAllWindows()
