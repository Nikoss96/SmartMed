import numpy
import requests
import telebot

API_link = "https://api.telegram.org/bot6586604489:AAGbck2PsvKCm3SeQ7Y5Xf6-agRtgbxf1xk"

updates = requests.get(API_link + "/getUpdates").json()
bot_token = '6586604489:AAGbck2PsvKCm3SeQ7Y5Xf6-agRtgbxf1xk'
bot = telebot.TeleBot('6586604489:AAGbck2PsvKCm3SeQ7Y5Xf6-agRtgbxf1xk')

keyboard1 = telebot.types.ReplyKeyboardMarkup()
keyboard1.row('хочу начать', 'поздравляйте')

trigger = False
#keyboard2 = telebot.types.ReplyKeyboardMarkup()
#keyboard2.row('smth like poetry','smth like films')

    
@bot.message_handler(commands=['start'])
def start_message(message):
    global user_id
    user_id = message.from_user.username
    bot.send_message(message.chat.id, 'Привет, юзер!')
    bot.send_message(message.chat.id, 'В этот раз мы не стоим на месте и день рождения перемещаается в телеграм!')
    bot.send_message(message.chat.id, 'Выбирай)', reply_markup=keyboard1)

@bot.message_handler(content_types=['document'])
def handle_document(message,trigger):
    # Получаем информацию о файле
    file_info = bot.get_file(message.document.file_id)
    file_url = f'https://api.telegram.org/file/bot{bot_token}/{file_info.file_path}'
    
    # Скачиваем файл
    response = requests.get(file_url)
    if response.status_code == 200:
        # Указываем путь для сохранения файла
        file_name = message.document.file_name
        if trigger == True:
            with open(file_name, 'wb') as file:
                file.write(response.content)
            bot.reply_to(message, f'Файл {file_name} успешно загружен')
            trigger = False
    else:
        bot.reply_to(message, 'Произошла ошибка при загрузке файла')
@bot.message_handler(content_types=['text'])
def default_test(message):
    if message.text.lower() == 'поздравляйте':
        keyboard00 = telebot.types.InlineKeyboardMarkup()
        keyboard00.add(telebot.types.InlineKeyboardButton(text="видик", url="https://www.tiktok.com/@solanodasilva/video/7060826336591416578?is_copy_url=1&is_from_webapp=v1&q=%D1%81%20%D0%B4%D0%BD%D0%B5%D0%BC%20%D1%80%D0%BE%D0%B6%D0%B4%D0%B5%D0%BD%D0%B8%D1%8F&t=1649688128360") )
        keyboard00.add(telebot.types.InlineKeyboardButton(text="какие-то буквы", url="https://pastebin.com/XzLwR36x") )
        bot.send_message(message.chat.id, "Choose", reply_markup=keyboard00)  
    elif message.text.lower() == 'хочу начать':
        bot.send_message(message.chat.id, "Ага, да")
        keyboard01 = telebot.types.InlineKeyboardMarkup()
        keyboard01.add(telebot.types.InlineKeyboardButton(text="объясните мне, что вообще происходит и какие нафиг части", url="https://pastebin.com/eb9q0ahj") ) 
        keyboard01.add(telebot.types.InlineKeyboardButton(text="часть 1", url="http://puzzlecup.com/crossword-ru/?guess=33C8B19C87837763") )
        keyboard01.add(telebot.types.InlineKeyboardButton(text="часть 2 (сыграть со мной в)", url="https://www.iphones.ru/wp-content/uploads/2020/03/970146_53d9c33901.jpg") )
        keyboard01.add(telebot.types.InlineKeyboardButton(text="часть 3", url="https://pastebin.com/p9c14fuF") )
        bot.send_message(message.chat.id, "А вот и квест-меню", reply_markup=keyboard01)
    elif message.text.lower() == '000':
        bot.send_message(message.chat.id, "Загляни-ка в гараж")
    
bot.polling()

