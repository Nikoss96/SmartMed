from statistical_terms import statistical_terms
from functions import get_anyfile, get_file_for_descriptive_analysis, \
    open_and_send_file, \
    generate_dictionary_keyboard
from keyboard import (
    keyboard_main_menu,
    keyboard00,
    keyboard01,
    keyboard_modules,
)


def callback_query_handler(bot, call):
    """
    Обработка нажатия кнопок.
    """
    try:
        command: str = call.data

        print(
            f"User {call.from_user.username} in {call.from_user.id} chat asked for {command}")

        if command.startswith("prev_") or command.startswith("next_"):

            action, page = command.split('_') if '_' in command else (
                command, 0)

            page = int(page)
            if action == "prev":
                page -= 1

            bot.edit_message_text(chat_id=call.message.chat.id,
                                  message_id=call.message.message_id,
                                  text="Выберите интересующий вас термин:",
                                  reply_markup=generate_dictionary_keyboard(
                                      page)
                                  )

        if command.startswith('statistical_term'):
            term = command.replace("statistical_term_", "")

            bot.send_message(
                chat_id=call.from_user.id,
                text="".join(statistical_terms[f"term_{term}"]),
            )

            open_and_send_file(bot, call.from_user.id, command)

        if command == "example_bioequal":
            bot.answer_callback_query(
                callback_query_id=call.id,
                text="Прислали вам пример файла. Оформляйте в точности так.",
            )

            file = open("media/data/параллельный тестовый.xlsx", "rb")

            bot.send_document(chat_id=call.from_user.id, document=file)

        elif command == "download_bioequal":
            bot.answer_callback_query(
                callback_query_id=call.id,
                text="Можете прислать свой файл прямо сюда."
            )

            get_anyfile(bot, call)

        elif command == "example_describe":
            bot.answer_callback_query(
                callback_query_id=call.id,
                text="Прислали вам пример файла. Оформляйте в точности так.",
            )

            file = open("media/data/Описательный_анализ_пример.xls", "rb")

            bot.send_document(chat_id=call.from_user.id, document=file)

        elif command == "download_describe":
            get_file_for_descriptive_analysis(bot, call)

        elif command == "back":

            bot.send_message(
                chat_id=call.from_user.id,
                text="Выберите модуль.",
                reply_markup=keyboard_main_menu,
            )

    except Exception as e:
        print(f"Ошибка: \n{e}")


def start_message_handler(bot, message):
    """
    Обработка кнопки Start. Запускается при запуске бота пользователем.
    """
    try:
        user_id = message.from_user.username
        chat_id = message.chat.id

        bot.send_message(chat_id=chat_id, text="Доброго дня!")
        bot.send_message(chat_id=chat_id,
                         text="Рады приветствовать вас в Smart-Медицине!")
        bot.send_message(
            chat_id=chat_id,
            text="Вам доступен следующий функционал: \n - Вызов медицинских "
                 "модулей; \n - Вызов словаря; \n - Общение с виртуальным "
                 "ассистентом.",
            reply_markup=keyboard_main_menu,
        )
    except Exception as e:
        print(f"Ошибка: \n{e}")


def text_handler(bot, message):
    """
    Обработка текста, присылаемого пользователем.
    """
    try:
        command = message.text.lower()

        switch = {
            "bioequal": keyboard00,
            "describe": keyboard01,
            "predict": None,
            "модули": keyboard_modules,
            "назад": keyboard_main_menu,
            "словарь": generate_dictionary_keyboard(0),
            "chat-gpt": None,
            "cluster": None,
        }

        reply_markup = switch.get(command, None)

        if reply_markup is not None:
            if command == "модули":
                bot.send_message(
                    chat_id=message.chat.id,
                    text="Выберите модуль из предложенных ниже.",
                    reply_markup=reply_markup,
                )
            # elif command == "словарь":
            #     bot.send_message(
            #         chat_id=message.chat.id,
            #         text="Выберите интересующий вас термин:",
            #         reply_markup=reply_markup,
            #     )
            else:
                bot.send_message(
                    chat_id=message.chat.id,
                    text="Готовы загрузить сразу или требуется пояснение?",
                    reply_markup=reply_markup,
                )
        else:
            bot.send_message(chat_id=message.chat.id, text="В разработке")
    except Exception as e:
        print(f"Ошибка: \n{e}")
