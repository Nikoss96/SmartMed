from describe_analysis.functions_descriptive import (
    handle_example_describe,
    handle_describe_build_graphs,
    handle_describe_correlation_analysis,
    handle_download_describe,
    handle_downloaded_describe_file,
    handle_describe_table,
)
from dictionary.functions_dictionary import (
    handle_pagination_dictionary,
    handle_statistical_term,
)
from keyboard import keyboard_main_menu, keyboard_in_development
from functions import (
    get_reply_markup,
    handle_back,
    send_text_message,
)


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
            handle_pagination_dictionary(bot, call)

        elif command.startswith("statistical_term"):
            handle_statistical_term(bot, call)

        elif command == "example_describe":
            handle_example_describe(bot, call)

        elif command == "download_describe":
            handle_download_describe(bot, call)

        elif command == "back":
            handle_back(bot, user_id)

        elif command == "describe_build_graphs":
            handle_describe_build_graphs(bot, call)

        elif command == "describe_correlation_analysis":
            handle_describe_correlation_analysis(bot, call)

        elif command == "describe_table":
            handle_describe_table(bot, call)

        elif command in [
            "replace_null_with_mean",
            "delete_null_rows_dropna",
            "replace_null_with_median",
        ]:
            handle_downloaded_describe_file(bot, call, command)

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

        greeting_text = (
            "Доброго дня!\n\nРады приветствовать вас "
            "в Smart-Медицине!\n\nВам доступен следующий "
            "функционал: \n- Модули анализа данных\n"
            "- Словарь терминов"
        )

        send_text_message(bot, chat_id, greeting_text, reply_markup=keyboard_main_menu)

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

        if reply_markup is keyboard_in_development:
            send_text_message(
                bot,
                chat_id=message.chat.id,
                text="Данный модуль пока находится в разработке",
                reply_markup=reply_markup,
            )
            return

        print(f"User {username} in {chat_id} chat wrote {command}")

        if command in ["Описательный анализ"]:
            send_text_message(
                bot,
                chat_id=message.chat.id,
                text="Выберите опцию при работе с модулем:",
                reply_markup=reply_markup,
            )

        elif command == "модули":
            send_text_message(
                bot,
                chat_id=message.chat.id,
                text="Выберите интересующий вас модуль:",
                reply_markup=reply_markup,
            )
        else:
            send_text_message(
                bot,
                chat_id=message.chat.id,
                text="Выберите интересующий вас термин:",
                reply_markup=reply_markup,
            )
    except Exception as e:
        print(f"Ошибка: \n{e}")
