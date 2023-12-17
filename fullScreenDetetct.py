import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from yolodetect import YoloDetect
from telegram_utils import send_telegram
import datetime
import threading
from sendEmail import send_email

password = "wnmd msav gljs qofz"
from_email = "congdanhzer0x2002@gmail.com"  # must match the email used to generate the password
to_email = "ddanh14372@gmail.com"  # receiver email

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)


# def send_email(to_email, from_email, object_detected=1):
#     message = MIMEMultipart()
#     message['From'] = from_email
#     message['To'] = to_email
#     message['Subject'] = "Security Alert"
#     # Add in the message body
#     message_body = f'ALERT - {object_detected} objects has been detected!!'
#
#     message.attach(MIMEText(message_body, 'plain'))
#     server.sendmail(from_email, to_email, message.as_string())


class ObjectDetection:
    def __init__(self, capture_index):
        # default parameters
        self.capture_index = capture_index
        self.email_sent = False

        # model information
        self.model = YOLO("runs/detect/train8/weights/best.pt")

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_alert = None
        self.alert_each = 15  # seconds

    def predict(self, im0):
        results = self.model(im0, device='0')
        return results

    def alert(self, img, object_detected):
        # New thread to send telegram after 15 seconds
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_each):
            self.last_alert = datetime.datetime.utcnow()
            cv2.imwrite("alert1.png", cv2.resize(img, dsize=None, fx=0.5, fy=0.5))
            thread = threading.Thread(target=send_telegram(photo_path="alert1.png", object_detected= object_detected))
            thread1 = threading.Thread(
                target=send_email(to_email, from_email, object_detected=object_detected, image_path="alert1.png"))
            thread.start()
        return img

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

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        width = 1080
        height = 720

        assert cap.isOpened()

        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:  # Only send email If not sent before
                self.alert(img=im0, object_detected=len(class_ids))
            self.display_fps(im0)
            im1 = cv2.resize(im0, (width, height))
            cv2.imshow('YOLOv8 Detection', im1)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()


detector = ObjectDetection(capture_index="D:/BaseProject/test_video6.mp4")
detector()
