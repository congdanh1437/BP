from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
def send_email(to_email, from_email, object_detected=1, image_path=None):
    password = "wnmd msav gljs qofz"
    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
    server.login(from_email, password)
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"

    message_body = f'ALERT - {object_detected} objects have been detected!!'
    message.attach(MIMEText(message_body, 'plain'))

    # Attach image if provided
    if image_path:
        with open(image_path, 'rb') as image_file:
            image_attachment = MIMEImage(image_file.read(), name='alert1.png')
            message.attach(image_attachment)
    server.sendmail(from_email, to_email, message.as_string())
