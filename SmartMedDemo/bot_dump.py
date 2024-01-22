import requests
import telebot
#from telegram import Update
#from telegram.ext import CallbackContext, CommandHandler, Updater
# Токен бота
#Main_test_bot
bot_token = '6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU'
#First_test_bot
#bot_token = '6586604489:AAGbck2PsvKCm3SeQ7Y5Xf6-agRtgbxf1xk'

bot = telebot.TeleBot(bot_token)

#Словарь терминов
dict = {'T-критерий Стьюдента для независимых переменных':'статистический критерий, применяемый для сравнения средних значений двух независимых выборок. Он основан на t-распределении и используется для определения, есть ли статистически значимые различия между средними значениями двух групп.',
        'Коэффициент корреляции Спирмена':'это непараметрическая мера ранговой корреляции, показывающая, насколько изменения в рангах одной переменной связаны с изменениями в рангах другой переменной, используется, если данные имеют порядковый характер, а не абсолютные числовые значения, метод устойчив к выбросам в данных. Формула расчета коэффициента включает в себя разность между рангами переменных и вычисление суммы квадратов этих разностей.',
        'Кривая выживаемости':'графическое представление вероятности выжить после некоторого начального события в зависимости от времени. Используется для изучения продолжительности жизни и других временных параметров.',
        'Диаграмма <ящик с усами>':'это визуальное представление о распределении данных, позволяющее оценить медиану, квартили, разброс и наличие выбросов в наборе данных, где “ящик” представляет собой межквартильный размах, в котором содержится ровно 50% всех наблюдений, по его размеру можно судить о величине диапазона разброса значений, если размер ящика значительный, то значения имеют большое различие. Ящик с усами может восприниматься как вид сверху на всем известную гистограмму'
        }
"""
dict = {'Распределение':'Закон, описывающий область значений случайной величины и соответствующие вероятности появления этих значений.',
        'Среднее':'Информативная мера "центрального положения" наблюдаемой переменной. Исследователю нужны такие статистики, которые позволяют сделать вывод относительно популяции в целом. Одной из таких статистик является среднее.',
        'Кластер':'Объединение нескольких однородных элементов, которое может рассматриваться как самостоятельная единица, обладающая определёнными свойствами.',
        'Дендрограмма':'Это способ визуализации, используемый для представления результатов иерархической кластеризации.',
        'Кривая выживаемости':'Графическое отображение зависимости доли выживших исследуемых объектов от их возраста.'}
"""
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
keyboard_modules.row('bioequal','cluster', 'describe','predict')

keyboard_dict = telebot.types.InlineKeyboardMarkup()
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="T-критерий Стьюдента для независимых переменных",callback_data="t-crit") )
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Коэффициент корреляции Спирмена",callback_data="spearman-corr") )
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Кривая выживаемости",callback_data="curve") )
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Диаграмма Ящик с усами",callback_data="box") )
keyboard_dict.add(telebot.types.InlineKeyboardButton(text="Назад",callback_data="back") )

#Инит для обработки колбека
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data == "example_bioequal":
        bot.answer_callback_query(call.id, "Прислали вам пример файла. Оформляйте в точности так.")
        file = open("параллельный тестовый.xlsx","rb")
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
                #file = open("picdist.png","rb")
                bot.send_document(call.from_user.id,file)
            else:
                bot.reply_to(message, 'Произошла ошибка при загрузке файла')
    elif call.data == "example_describe":
        bot.answer_callback_query(call.id, "Прислали вам пример файла. Оформляйте в точности так.")
        file = open("Описательный_анализ_пример.xls","rb")
        bot.send_document(call.from_user.id,file)
    elif call.data == "download_describe":
        #get_anyfile(bot,call)
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
                bot.send_message(call.from_user.id,"Результат анализа:")
                file = open("picdist.png","rb")
                bot.send_document(call.from_user.id,file)
                #bot.send_message(call.from_user.id,"Результат анализа:")
                #file = open("picdist.png","rb")
                #bot.send_document(call.from_user.id,file)
            else:
                bot.reply_to(message, 'Произошла ошибка при загрузке файла')
        file = open("picdist.png","rb")
    elif call.data == "t-crit":
        bot.send_message(call.from_user.id, f"T-критерий Стьюдента для независимых переменных - \n {dict['T-критерий Стьюдента для независимых переменных']}")
    elif call.data == "spearman-corr":
        bot.send_message(call.from_user.id, f"Коэффициент корреляции Спирмена - \n {dict['Коэффициент корреляции Спирмена']}")
        file_cur = open("unnamed.png","rb")
        bot.send_document(call.from_user.id,file_cur)
    elif call.data == "curve":
        bot.send_message(call.from_user.id, f"Кривая выживаемости - \n {dict['Кривая выживаемости']}")
    elif call.data == "box":
        bot.send_message(call.from_user.id, f"Диаграмма Ящик с усами - \n {dict['Диаграмма <ящик с усами>']}")
        file_cur = open("box.jpg","rb")
        bot.send_document(call.from_user.id,file_cur)
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
    elif message.text.lower() == 'describe':
       keyboard01 = telebot.types.InlineKeyboardMarkup()
       keyboard01.add(telebot.types.InlineKeyboardButton(text="Пример файла",callback_data="example_describe") )
       keyboard01.add(telebot.types.InlineKeyboardButton(text="Загружу свой",callback_data="download_describe") )
       bot.send_message(message.chat.id, "Готовы загрузить сразу или требуется пояснение?", reply_markup = keyboard01)
    elif message.text.lower() == 'predict':
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
    elif message.text.lower() == 'cluster':
        bot.send_message(message.chat.id, 'Coming soon', reply_markup=keyboard_modules)
    elif message.text.lower() == 'cluster':
        bot.send_message(message.chat.id, 'Coming soon', reply_markup=keyboard_modules)
        
    
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