from functions import (get_reply_markup, handle_back, handle_download_bioequal,
                       handle_download_describe, handle_example_bioequal,
                       handle_example_describe, handle_pagination,
                       handle_statistical_term, send_text_message)
from keyboard import keyboard_main_menu


def callback_query_handler(bot, call):
    """
    Обработка нажатия кнопок.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    try:
        command: str = call.data
        user_id = call.from_user.id
        username = call.from_user.username

        print(f"User {username} in {user_id} chat asked for {command}")

        if command.startswith("prev_") or command.startswith("next_"):
            handle_pagination(bot, call)

        elif command.startswith("statistical_term"):
            handle_statistical_term(bot, call)

        elif command == "example_bioequal":
            handle_example_bioequal(bot, call)

        elif command == "download_bioequal":
            handle_download_bioequal(bot, call)

        elif command == "example_describe":
            handle_example_describe(bot, call)

        elif command == "download_describe":
            handle_download_describe(bot, call)

        elif command == "back":
            handle_back(bot, user_id)

    except Exception as e:
        print(f"Ошибка: \n{e}")


def start_message_handler(bot, message):
    """
    Обработка кнопки Start. Запускается при запуске бота пользователем.
    """
    try:
        user = message.from_user.username
        chat_id = message.chat.id

        print(f"User {user} in {chat_id} chat started the bot!")

        greeting_text = "Доброго дня!"
        welcome_text = "Рады приветствовать вас в Smart-Медицине!"
        functionality_text = (
            "Вам доступен следующий функционал: \n"
            " - Вызов медицинских модулей; \n"
            " - Вызов словаря; \n"
            " - Общение с виртуальным ассистентом."
        )

        send_text_message(bot, chat_id, greeting_text)
        send_text_message(bot, chat_id, welcome_text)
        send_text_message(
            bot, chat_id, functionality_text, reply_markup=keyboard_main_menu
        )

    except Exception as e:
        print(f"Ошибка: \n{e}")


def text_handler(bot, message):
    """
    Обработка текста, присылаемого пользователем.
    """
    try:
        command = message.text.lower()
        reply_markup = get_reply_markup(command)
        chat_id = message.chat.id
        username = message.from_user.username

        if reply_markup is None:
            send_text_message(bot, chat_id=message.chat.id, text="В разработке")
            return

        print(f"User {username} in {chat_id} chat wrote {command}")

        if command == "модули":
            send_text_message(
                bot,
                chat_id=message.chat.id,
                text="Выберите модуль из предложенных ниже.",
                reply_markup=reply_markup,
            )
        else:
            send_text_message(
                bot,
                chat_id=message.chat.id,
                text="Готовы загрузить сразу или требуется пояснение?",
                reply_markup=reply_markup,
            )
    except Exception as e:
        print(f"Ошибка: \n{e}")
