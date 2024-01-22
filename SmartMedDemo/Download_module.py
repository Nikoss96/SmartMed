import requests
import telebot
from telegram import Update
from telegram.ext import CallbackContext, CommandHandler, Updater
# Токен бота
#Main_test_bot
bot_token = '6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU'
#First_test_bot
#bot_token = '6586604489:AAGbck2PsvKCm3SeQ7Y5Xf6-agRtgbxf1xk'

bot = telebot.TeleBot(bot_token)

#Словарь терминов
dict = {'Распределение':'Закон, описывающий область значений случайной величины и соответствующие вероятности появления этих значений.',
        'Среднее':'Информативная мера "центрального положения" наблюдаемой переменной. Исследователю нужны такие статистики, которые позволяют сделать вывод относительно популяции в целом. Одной из таких статистик является среднее.',
        'Кластер':'Объединение нескольких однородных элементов, которое может рассматриваться как самостоятельная единица, обладающая определёнными свойствами.',
        'Дендрограмма':'Это способ визуализации, используемый для представления результатов иерархической кластеризации.',
        'Кривая выживаемости':'Графическое отображение зависимости доли выживших исследуемых объектов от их возраста.'}
#Подкачиваем файл из диалога
def get_anyfile(bot,call):
    bot.answer_callback_query(call.id, "Можете прислать свой файл прямо сюда.")
    @bot.message_handler(content_types=['document'])
    def handle_document(message):
    # Получаем информацию о файле
        file_info = bot.get_file(message.document.file_id)
        file_url = f'https://api.telegram.org/file/bot{bot_token}/{file_info.file_path}'
        
        # Скачиваем файл
        response = requests.get(file_url)
        if response.status_code == 200:
            # Указываем путь для сохранения файла
            file_name = message.document.file_name
            with open(file_name, 'wb') as file:
                file.write(response.content)
            bot.reply_to(message, f'Файл {file_name} успешно загружен')
        else:
            bot.reply_to(message, 'Произошла ошибка при загрузке файла')
#Клавиатуры внутри бота
keyboard_main_menu = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_main_menu.row('Модули','Словарь','Chat-GPT')

keyboard_modules = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_modules.row('кластер', 'lifeline','предикшн','компаратив','bioequal')

keyboard_dict = telebot.types.InlineKeyboardMarkup()
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Распределение",callback_data="distribution") )
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Среднее",callback_data="mediana") )
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Назад",callback_data="back") )


#Инит для обработки колбека
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data == "example_bioequal":
        bot.answer_callback_query(call.id, "Прислали вам пример файла. Оформляйте в точности так.")
        file = open("Пример_биоэквивал.xls","rb")
        bot.send_document(call.from_user.id,file)
    elif call.data == "download_bioequal":
        bot.answer_callback_query(call.id, "Можете прислать свой файл прямо сюда.")
        @bot.message_handler(content_types=['document'])
        def handle_document(message):
        # Получаем информацию о файле
            file_info = bot.get_file(message.document.file_id)
            file_url = f'https://api.telegram.org/file/bot{bot_token}/{file_info.file_path}'
            
            # Скачиваем файл
            response = requests.get(file_url)
            if response.status_code == 200:
                # Указываем путь для сохранения файла
                file_name = message.document.file_name
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                bot.reply_to(message, f'Файл {file_name} успешно загружен')
            else:
                bot.reply_to(message, 'Произошла ошибка при загрузке файла')
    elif call.data == "example_lifeline":
        pass
    elif call.data == "download_lifeline":
        get_anyfile(bot,call)
    elif call.data == "distribution":
        bot.send_message(call.from_user.id, f"Распределение \n {dict['Распределение']}")
    elif call.data == "mediana":
        bot.send_message(call.from_user.id, f"Среднее \n {dict['Среднее']}")
    elif call.data == "back":
        bot.send_message(call.from_user.id, 'Вы снова можете выбрать модуль.', reply_markup=keyboard_main_menu)
        #Здесь файл уходит в функцию и происходит следующее: !!!!
        # 1. Забираем файл из текуцщей директории
        # 2. Предобрабатываем файл эксель
        # 3. Отдаем файл в модуль
        # 4. Забираем из модуля результат в виде графика \ веб-морды

        
#Инициализация бота + описание содержания клавиатуры

@bot.message_handler(commands=['start'])
def start_message(message):
    global user_id
    user_id = message.from_user.username
    bot.send_message(message.chat.id, 'Доброго дня!')
    bot.send_message(message.chat.id, 'Рады приветствовать вас в SmartMedicine!')
    bot.send_message(message.chat.id, 'Вам доступен следующий функционал: \n - Вызов медицинских модулей; \n - Вызов словаря; \n - Общение с виртуальным ассистентом.', reply_markup=keyboard_main_menu)
    
@bot.message_handler(content_types=['text'])
def default_test(message):
    if message.text.lower() == 'bioequal':
        keyboard00 = telebot.types.InlineKeyboardMarkup()
        keyboard00.add(telebot.types.InlineKeyboardButton(text="Пример файла",callback_data="example_bioequal") )
        keyboard00.add(telebot.types.InlineKeyboardButton(text="Загружу свой",callback_data="download_bioequal") )
        keyboard00.add(telebot.types.InlineKeyboardButton(text="Назад",callback_data="back") )
        bot.send_message(message.chat.id, "Готовы загрузить сразу или требуется пояснение?", reply_markup = keyboard00)
    elif message.text.lower() == 'lifeline':
       keyboard01 = telebot.types.InlineKeyboardMarkup()
       keyboard01.add(telebot.types.InlineKeyboardButton(text="Пример файла",callback_data="example_lifeline") )
       keyboard01.add(telebot.types.InlineKeyboardButton(text="Загружу свой",callback_data="download_lifeline") )
       bot.send_message(message.chat.id, "Готовы загрузить сразу или требуется пояснение?", reply_markup = keyboard01)
    elif message.text.lower() == 'предикшн':
        keyboard02 = telebot.types.InlineKeyboardMarkup()
        keyboard02.add(telebot.types.InlineKeyboardButton(text="Пример файла",callback_data="example_predict") )
        keyboard02.add(telebot.types.InlineKeyboardButton(text="Загружу свой",callback_data="download_predict") )
        bot.send_message(message.chat.id, "Готовы загрузить сразу или требуется пояснение?", reply_markup = keyboard02)
    elif message.text.lower() == 'модули':
        bot.send_message(message.chat.id, 'Выберите модуль из предложенных ниже.', reply_markup=keyboard_modules)
    elif message.text.lower() == 'назад':
        bot.send_message(message.chat.id, 'Вы снова можете выбрать модуль.', reply_markup=keyboard_main_menu)
    elif message.text.lower() == 'словарь':
        bot.send_message(message.chat.id, 'Выберите интересующий вас термин:', reply_markup=keyboard_dict)
    elif message.text.lower() == 'chat-gpt':
        bot.send_message(message.chat.id, 'Coming soon')
        
    
        """
    @bot.message_handler(content_types=['document'])
        def handle_document(message):
    # Получаем информацию о файле
    file_info = bot.get_file(message.document.file_id)
    file_url = f'https://api.telegram.org/file/bot{bot_token}/{file_info.file_path}'
    
    # Скачиваем файл
    response = requests.get(file_url)
    if response.status_code == 200:
        # Указываем путь для сохранения файла
        file_name = message.document.file_name
        with open(file_name, 'wb') as file:
            file.write(response.content)
        bot.reply_to(message, f'Файл {file_name} успешно загружен')
    else:
        bot.reply_to(message, 'Произошла ошибка при загрузке файла')

    """
    """
    if message.text.lower() == 'параллельный_реф':
        keyboard00 = telebot.types.InlineKeyboardMarkup()
        keyboard00.add(telebot.types.InlineKeyboardButton(text="Пример файла",callback_data="example_bioequal") )
        keyboard00.add(telebot.types.InlineKeyboardButton(text="Загружу свой",callback_data="download_bioequal") )
        bot.send_message(message.chat.id, "Готовы загрузить сразу или требуется пояснение?", reply_markup = keyboard00)
    if message.text.lower() == "тестовый_референсный"
    """
bot.polling()