import os

import pandas as pd
import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException

from data.paths import MEDIA_PATH, DATA_PATH, CLUSTER_ANALYSIS, USER_DATA_PATH
from functions import send_document_from_file, test_bot_token, save_file, \
    check_input_file, clear_user_files


def handle_example_cluster_analysis(bot, call):
    """
    Обработка запроса на пример файла для cluster analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id,
        text="Прислали пример файла. Вы можете использовать этот файл для проведения анализа",
    )
    send_document_from_file(
        bot,
        call.from_user.id,
        f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/Кластерный_анализ_пример.xlsx",
    )


def handle_download_cluster(bot, call):
    """
    Обработка запроса на загрузку файла для cluster analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.send_message(
        chat_id=call.from_user.id,
        text="Загрузите Ваш файл.\n\n"
             "Файл должен иметь следующие характеристики:\n"
             "\n1. Формат файла: .csv, .xlsx или .xls"
             "\n2. Размер файла: до 20 Мб"
             "\n3. Рекомендуемое количество столбцов для более"
             " наглядной визуализации — до 25."
             "\n4. Названия столбцов в файле не должны состоять только из"
             " цифр и содержать специальные символы",
    )
    clear_user_files(call.from_user.id, CLUSTER_ANALYSIS)
    get_file_for_cluster_analysis(bot)


def get_file_for_cluster_analysis(bot):
    """
    Обработка загрузки файла для кластерного анализа.
    """

    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        try:
            file_info = bot.get_file(message.document.file_id)

            file_url = f"https://api.telegram.org/file/bot{test_bot_token}/{file_info.file_path}"

            response = requests.get(file_url)

            if response.status_code == 200:
                file_name = save_file(
                    response.content, message.document.file_name,
                    message.chat.id,
                    CLUSTER_ANALYSIS
                )
                check_input_file(bot, message, file_name)

            else:
                bot.reply_to(message, "Произошла ошибка при загрузке файла")

        except ApiTelegramException as e:
            if e.description == "Bad Request: file is too big":
                bot.reply_to(
                    message=message,
                    text="Ваш файл превышает допустимый лимит 20 Мегабайт.",
                )

            else:
                bot.reply_to(message, "Произошла ошибка при загрузке файла")

            print(f"Error: {e}")

        except RequestException as e:
            print(f"Error while downloading file: {e}")
            bot.reply_to(message, "Произошла ошибка при загрузке файла")
        except Exception as e:
            print(f"Unexpected error: {e}")
            bot.reply_to(message, "Произошла ошибка при загрузке файла")
