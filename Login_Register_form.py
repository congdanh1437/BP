import sys
import re
import pyodbc
import smtplib
from email.mime.text import MIMEText
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox, QDialog, QRadioButton
import subprocess

class RegistrationForm(QDialog):
    def __init__(self, username, password, verification_code):
        super().__init__()

        self.setWindowTitle("Verification Code")
        self.setGeometry(200, 200, 400, 150)

        self.username = username
        self.password = password
        self.verification_code = verification_code

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.code_label = QLabel("Enter Verification Code:")
        self.code_input = QLineEdit()

        self.verify_button = QPushButton("Verify")
        layout.addWidget(self.code_label)
        layout.addWidget(self.code_input)
        layout.addWidget(self.verify_button)

        self.verify_button.clicked.connect(self.verify_code)

        self.setLayout(layout)

    def verify_code(self):
        entered_code = self.code_input.text()

        if entered_code == self.verification_code:
            if self.insert_user_into_database():
                QMessageBox.information(self, "Registration Successful", "User registered successfully.")
                self.done(QDialog.Accepted)
            else:
                QMessageBox.warning(self, "Registration Failed", "Failed to register user.")
        else:
            QMessageBox.warning(self, "Invalid Code", "The entered verification code is invalid.")

    def insert_user_into_database(self):
        try:
            connection_string = "DRIVER={SQL Server};SERVER=LAPTOP-MPPBE2T0;DATABASE=BaseProject;UID=danh;PWD=123"
            connection = pyodbc.connect(connection_string)
            cursor = connection.cursor()

            query = "INSERT INTO users (username, password, verification_code) VALUES (?, ?, ?)"
            cursor.execute(query, (self.username, self.password, self.verification_code))

            connection.commit()
            connection.close()

            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

class OptionsForm(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Choose Detection Option")
        self.setGeometry(300, 300, 300, 150)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.full_screen_radio = QRadioButton("Detect Full Screen")
        self.region_detect_radio = QRadioButton("Region Detect")

        self.confirm_button = QPushButton("Confirm")
        layout.addWidget(self.full_screen_radio)
        layout.addWidget(self.region_detect_radio)
        layout.addWidget(self.confirm_button)

        self.confirm_button.clicked.connect(self.confirm_option)

        self.setLayout(layout)

    def confirm_option(self):
        self.close()
        if self.full_screen_radio.isChecked():
            self.start_full_screen_detection()
        elif self.region_detect_radio.isChecked():
            self.start_region_detection()

    def start_full_screen_detection(self):
        try:
            subprocess.run(["D:/BaseProject/BP/venv/Scripts/python.exe", "fullScreenDetect.py", to_email])
        except Exception as e:
            print(f"Error starting full_screen_detection_script.py: {e}")

    def start_region_detection(self):
        try:
            subprocess.run(["D:/BaseProject/BP/venv/Scripts/python.exe", "regionDetectYolov8.py", to_email])
        except Exception as e:
            print(f"Error starting regionDetectYolov8.py: {e}")

class LoginRegisterForm(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Login/Register Form")
        self.setGeometry(100, 100, 400, 200)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.username_label = QLabel("Username:")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Login")
        self.register_button = QPushButton("Register")

        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.register_button)

        layout.addLayout(button_layout)

        self.login_button.clicked.connect(self.login)
        self.register_button.clicked.connect(self.start_registration)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not self.is_gmail(username):
            QMessageBox.warning(self, "Invalid Username", "Please enter a valid Gmail address.")
            self.username_input.clear()
            return

        if self.verify_user_credentials(username, password):
            global to_email  # Declare from_email as a global variable
            to_email = username  # Set from_email to the logged-in username
            QMessageBox.information(self, "Login Successful", "User logged in successfully.")
            self.close()
            self.show_options_form()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def show_options_form(self):
        options_form = OptionsForm()
        options_form.exec_()

    def username_exists(self, username):
        try:
            connection_string = "DRIVER={SQL Server};SERVER=LAPTOP-MPPBE2T0;DATABASE=BaseProject;UID=danh;PWD=123"
            connection = pyodbc.connect(connection_string)
            cursor = connection.cursor()

            query = "SELECT COUNT(*) FROM users WHERE username = ?"
            cursor.execute(query, (username,))
            result = cursor.fetchone()

            connection.close()

            return result[0] > 0
        except Exception as e:
            print(f"Error: {e}")
            return False

    def start_registration(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not self.is_gmail(username):
            QMessageBox.warning(self, "Invalid Username", "Please enter a valid Gmail address.")
            self.username_input.clear()
            return

        if self.username_exists(username):
            QMessageBox.warning(self, "Username Exists", "The entered username already exists. Please choose a different one.")
            self.username_input.clear()
            return

        verification_code = self.generate_verification_code()

        if self.send_verification_email(username, verification_code):
            QMessageBox.information(self, "Verification Email Sent", "A verification email has been sent to your Gmail address.")

            verification_form = RegistrationForm(username, password, verification_code)
            if verification_form.exec_() == QDialog.Accepted:
                pass
            else:
                pass
        else:
            QMessageBox.warning(self, "Email Sending Failed", "Failed to send a verification email. Please try again.")

    def is_gmail(self, username):
        gmail_pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@gmail\.com$')
        return bool(gmail_pattern.match(username))

    def generate_verification_code(self):
        import random
        return f"{random.randint(100000, 999999)}"

    def send_verification_email(self, username, verification_code):
        try:
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
            smtp_username = 'ddanh14372@gmail.com'
            smtp_password = 'xbzn fjic qois oflr'

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)

            subject = 'Verification Code for Registration'
            body = f'Hello {username},\n\nYour verification code is: {verification_code}.\n\nThank you!'
            message = MIMEText(body)
            message['Subject'] = subject
            message['From'] = smtp_username
            message['To'] = username

            server.sendmail(smtp_username, username, message.as_string())

            server.quit()

            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    def verify_user_credentials(self, username, password):
        try:
            connection_string = "DRIVER={SQL Server};SERVER=LAPTOP-MPPBE2T0;DATABASE=BaseProject;UID=danh;PWD=123"
            connection = pyodbc.connect(connection_string)
            cursor = connection.cursor()

            query = "SELECT password, verification_code FROM users WHERE username = ?"
            cursor.execute(query, (username,))
            result = cursor.fetchone()

            if result:
                stored_password = result[0]
                verification_code = result[1]

                if password == stored_password:
                    if verification_code:
                        return True
                    else:
                        QMessageBox.warning(self, "Account Not Verified", "Please verify your account before logging in.")
                else:
                    QMessageBox.warning(self, "Login Failed", "Invalid password.")
            else:
                QMessageBox.warning(self, "Login Failed", "Invalid username.")

            connection.close()

        except Exception as e:
            print(f"Error: {e}")

        return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginRegisterForm()
    window.show()
    sys.exit(app.exec_())
