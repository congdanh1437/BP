import telepot
def send_telegram(photo_path="alert1.png"):
    my_token = "6843881066:AAGOQOPHXbfhPGYnXKc7eZ3kBVq9YD4uebY"
    bot = telepot.Bot(token=my_token)
    bot.sendPhoto(chat_id="5440234297", photo=open(photo_path, "rb"), caption="Security Alert")
    print("Send sucess")