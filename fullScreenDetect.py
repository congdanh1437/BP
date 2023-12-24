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
import os
password = "wnmd msav gljs qofz"
from_email = "congdanhzer0x2002@gmail.com"
to_email = "ddanh14372@gmail.com"

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)

class ObjectDetection(threading.Thread):
    def __init__(self, capture_index, alert_queue, output_video_path):
        threading.Thread.__init__(self)
        self.capture_index = capture_index
        self.model = YOLO("runs/detect/train8/weights/best.pt")
        self.alert_queue = alert_queue
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.output_video_path = output_video_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_alert = None
        self.alert_each = 15  # seconds

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
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        width = 1080
        height = 720

        assert cap.isOpened()

        frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, 25, (width, height))
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:
                self.alert(img=im0, object_detected=len(class_ids))

            self.display_fps(im0)
            im1 = cv2.resize(im0, (width, height))
            cv2.imshow('YOLOv8 Detection', im1)
            self.video_writer.write(im1)
            frame_count += 1

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
        self.video_writer.release()
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
    current_date = datetime.now()
    timestamp = current_date.strftime("%d%b%Y_%Hh%Mm%Ss")
    output_folder = "recorded_fullScreen_video"
    os.makedirs(output_folder, exist_ok=True)
    output_video_path = os.path.join(output_folder, f"fullScreen_video_{timestamp}.avi")
    alert_queue = multiprocessing.Queue()

    # Start the alerting process
    alert_process = multiprocessing.Process(target=alerting_process, args=(alert_queue,))
    alert_process.start()

    # Start the image processing thread
    thread_vid = ObjectDetection(capture_index="D:/BaseProject/test_video6.mp4",
                                 alert_queue=alert_queue,
                                 output_video_path=output_video_path)
    thread_vid.start()

    # Wait for threads/processes to finish
    thread_vid.join()
    alert_process.join()
