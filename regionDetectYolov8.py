import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from telegram_utils import send_telegram
import datetime
import threading
from sendEmail import send_email
import multiprocessing
from PIL import Image
from datetime import datetime
import json

password = "wnmd msav gljs qofz"
from_email = "congdanhzer0x2002@gmail.com"
to_email = "ddanh14372@gmail.com"

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)

points = []  # List to store polygon points


class ObjectDetection(threading.Thread):
    def __init__(self, capture_index, alert_queue):
        threading.Thread.__init__(self)
        self.capture_index = capture_index
        self.model = YOLO("runs/detect/train8/weights/best.pt")
        self.alert_queue = alert_queue
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_alert = None
        self.alert_each = 15  # seconds
        self.detected_centroids = []
        self.detect = None

    def predict(self, im0):
        results = self.model(im0, device='0')
        return results

    def alert(self, img, object_detected):
        if (self.last_alert is None) or (
                (datetime.utcnow() - self.last_alert).total_seconds() > self.alert_each):
            self.last_alert = datetime.utcnow()
            alert_image = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
            self.alert_queue.put((alert_image, object_detected))

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255),
                      -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.detected_centroids = []  # Reset detected centroids
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            centroid_x = int((box[0] + box[2]) / 2)
            centroid_y = int((box[1] + box[3]) / 2)
            self.detected_centroids.append((centroid_x, centroid_y))
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def draw_polygon(self, frame, points):
        for point in points:
            frame = cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
        if len(points) > 1:
            frame = cv2.polylines(frame, [np.array(points)], isClosed=False, color=(255, 0, 0), thickness=2)
        return frame

    def draw_centroids(self, frame, centroids):
        for centroid in centroids:
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    def connect_last_point(self):
        if len(points) > 2:
            points.append(points[0])

    def is_centroid_inside_polygon(self, centroid):
        if len(points) < 5 or self.detect is not True:
            return False
        return cv2.pointPolygonTest(np.array(points), centroid, False) >= 0

    def save_polygon_to_json(self):
        if len(points) > 2:
            with open('polygon_points.json', 'w') as json_file:
                json.dump(points, json_file)

    def load_polygon_from_json(self):
        try:
            with open('polygon_points.json', 'r') as json_file:
                loaded_points = json.load(json_file)
                points.clear()
                points.extend(loaded_points)
                print("Polygon loaded successfully.")
        except FileNotFoundError:
            print("Polygon file not found.")

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        width = 1080
        height = 720

        assert cap.isOpened()

        frame_count = 0

        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            im1 = cv2.resize(im0, (width, height))
            # Draw the polygon
            im0 = self.draw_polygon(im1, points)

            # Predict and plot bounding boxes
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:
                for centroid in self.detected_centroids:
                    if self.is_centroid_inside_polygon(centroid):
                        self.alert(img=im0, object_detected=len(class_ids))
                        break  # Only alert once for any detected object inside the polygon

            self.display_fps(im0)

            # Draw centroids
            self.draw_centroids(im0, self.detected_centroids)

            cv2.imshow('YOLOv8 Detection', im1)

            frame_count += 1

            key = cv2.waitKey(5)
            if key == ord("q"):
                break
            elif key == ord("d"):
                self.connect_last_point()
                self.detect = True
            elif key == ord("s"):
                self.save_polygon_to_json()
            elif key == ord("l"):
                self.load_polygon_from_json()
                self.detect = True

            cv2.namedWindow('YOLOv8 Detection')
            cv2.setMouseCallback('YOLOv8 Detection', self.mouse_callback)
        cap.release()
        cv2.destroyAllWindows()
        server.quit()


def alerting_process(alert_queue):
    while True:
        alert_data = alert_queue.get()
        if alert_data:
            alert_image, object_detected = alert_data
            cv2.imwrite("alert1.png", alert_image)
            send_telegram(photo_path="alert1.png")
            send_email(to_email, from_email, object_detected=object_detected, image_path="alert1.png")
            img = Image.open('D:/BaseProject/BP/alert1.png')
            current_date = datetime.now()
            timestamp = current_date.strftime("%d%b%Y_%Hh%Mm%Ss")
            filename = f"alert1_{timestamp}.png"
            img.save('detected/' + filename)


if __name__ == "__main__":
    alert_queue = multiprocessing.Queue()

    # Start the alerting process
    alert_process = multiprocessing.Process(target=alerting_process, args=(alert_queue,))
    alert_process.start()

    # Start the image processing thread
    thread_vid = ObjectDetection(capture_index="D:/BaseProject/test_video6.mp4", alert_queue=alert_queue)
    thread_vid.start()

    # Wait for threads/processes to finish
    thread_vid.join()
    alert_process.join()
