import time
import cv2
import numpy as np
from yolodetect import YoloDetect

cap = cv2.VideoCapture("test_video6.mp4")

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

width = 1080
height = 720
cap.set(3, width)  # ID 3 là chiều rộng của video
cap.set(4, height)  # ID 4 là chiều cao của video
points = []


model = YoloDetect()


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:

        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame

detect = False

while True:

    ret,frame = cap.read()

    # frame = cv2.flip(frame, 1)
    if not ret:
        print("Không thể đọc frame từ camera.")
        break
    resized_frame = cv2.resize(frame, (width, height))
    # Ve ploygon
    resized_frame = draw_polygon(resized_frame, points)

    if detect:
        resized_frame = model.detect(frame= resized_frame, points= points)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])
        detect = True

    # Hien anh ra man hinh
    cv2.imshow("Intrusion Warning", resized_frame)

    cv2.setMouseCallback('Intrusion Warning', handle_left_click, points)
cap.release()
# video.stop()
cv2.destroyAllWindows()
