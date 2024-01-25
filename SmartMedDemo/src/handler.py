import requests

from functions import get_anyfile, get_file_for_descriptive_analysis
from tokens import main_bot_token
from keyboard import (
    keyboard_main_menu,
    keyboard00,
    keyboard01,
    keyboard02,
    keyboard_modules,
    keyboard_dict,
)


def callback_query_handler(bot, call):
    """
    Обработка нажатия кнопок.
    """
    if call.data == "example_bioequal":
        bot.answer_callback_query(
            callback_query_id=call.id,
            text="Прислали вам пример файла. Оформляйте в точности так.",
        )

        file = open("media/data/параллельный тестовый.xlsx", "rb")

        bot.send_document(chat_id=call.from_user.id, document=file)

    elif call.data == "download_bioequal":
        bot.answer_callback_query(
            callback_query_id=call.id, text="Можете прислать свой файл прямо сюда."
        )

        get_anyfile(bot, call)

    elif call.data == "example_describe":
        bot.answer_callback_query(
            callback_query_id=call.id,
            text="Прислали вам пример файла. Оформляйте в точности так.",
        )

        file = open("media/data/Описательный_анализ_пример.xls", "rb")

        bot.send_document(chat_id=call.from_user.id, document=file)

    elif call.data == "download_describe":
        get_file_for_descriptive_analysis(bot, call)

    elif call.data == "t-crit":
        bot.send_message(
            chat_id=call.from_user.id,
            text=f"T-критерий Стьюдента для независимых переменных - \n {dict['T-критерий Стьюдента для независимых переменных']}",
        )

    elif call.data == "spearman-corr":
        bot.send_message(
            chat_id=call.from_user.id,
            text=f"Коэффициент корреляции Спирмена - \n {dict['Коэффициент корреляции Спирмена']}",
        )

        file_cur = open("media/images/unnamed.png", "rb")

        bot.send_document(chat_id=call.from_user.id, document=file_cur)

    elif call.data == "curve":
        bot.send_message(
            chat_id=call.from_user.id,
            text=f"Кривая выживаемости - \n {dict['Кривая выживаемости']}",
        )

    elif call.data == "box":
        bot.send_message(
            chat_id=call.from_user.id,
            text=f"Диаграмма Ящик с усами - \n {dict['Диаграмма <ящик с усами>']}",
        )

        file_cur = open("media/images/box.jpg", "rb")

        bot.send_document(chat_id=call.from_user.id, document=file_cur)

    elif call.data == "back":
        bot.send_message(
            chat_id=call.from_user.id,
            text="Вы снова можете выбрать модуль.",
            reply_markup=keyboard_main_menu,
        )


def start_message_handler(bot, message):
    """
    Обработка кнопки Start. Запускается при запуске бота пользователем.
    """
    user_id = message.from_user.username
    chat_id = message.chat.id

    bot.send_message(chat_id=chat_id, text="Доброго дня!")
    bot.send_message(
        chat_id=chat_id, text="Рады приветствовать вас в " "SmartMedicine!"
    )
    bot.send_message(
        chat_id=chat_id,
        text="Вам доступен следующий функционал: \n - Вызов медицинских "
        "модулей; \n - Вызов словаря; \n - Общение с виртуальным "
        "ассистентом.",
        reply_markup=keyboard_main_menu,
    )


def text_handler(bot, message):
    """
    Обработка текста, присылаемого пользователем.
    """
    command = message.text.lower()

    switch = {
        "bioequal": keyboard00,
        "describe": keyboard01,
        "predict": keyboard02,
        "модули": keyboard_modules,
        "назад": keyboard_main_menu,
        "словарь": keyboard_dict,
        "chat-gpt": None,
        "cluster": keyboard_modules,
    }

    reply_markup = switch.get(command, None)

    if reply_markup is not None:
        if command == "модули":
            bot.send_message(
                chat_id=message.chat.id,
                text="Выберите модуль из предложенных ниже.",
                reply_markup=reply_markup,
            )
        elif command == "назад":
            bot.send_message(
                chat_id=message.chat.id,
                text="Вы снова можете выбрать модуль.",
                reply_markup=reply_markup,
            )
        elif command == "словарь":
            bot.send_message(
                chat_id=message.chat.id,
                text="Выберите интересующий вас термин:",
                reply_markup=reply_markup,
            )
        else:
            bot.send_message(
                chat_id=message.chat.id,
                text="Готовы загрузить сразу или требуется пояснение?",
                reply_markup=reply_markup,
            )
    else:
        bot.send_message(chat_id=message.chat.id, text="Coming soon")
