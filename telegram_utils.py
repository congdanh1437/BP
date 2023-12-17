import telepot
def send_telegram(photo_path="alert1.png", object_detected = 1):
    try:
        my_token = "6843881066:AAGOQOPHXbfhPGYnXKc7eZ3kBVq9YD4uebY"
        bot = telepot.Bot(token=my_token)
        bot.sendPhoto(chat_id="5440234297", photo=open(photo_path, "rb"), caption=f'ALERT - {object_detected} objects have been detected!!')
    except Exception as ex:
        print("Can not send message telegram ", ex)

    print("Send sucess")