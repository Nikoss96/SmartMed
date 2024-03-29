import os

import pandas as pd
import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException

from cluster_analysis.ClusterModule import ClusterModule
from cluster_analysis.keyboard_cluster import keyboard_choice_cluster, \
    keyboard_choice_number_of_clusters
from data.paths import MEDIA_PATH, DATA_PATH, CLUSTER_ANALYSIS, USER_DATA_PATH, \
    ELBOW_METHOD, EXAMPLES
from describe_analysis.functions_descriptive import get_user_file_df, \
    get_user_file
from functions import send_document_from_file, test_bot_token, save_file, \
    check_input_file, clear_user_files, create_dataframe_and_save_file


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
        f"{MEDIA_PATH}/{DATA_PATH}/{EXAMPLES}/Кластерный_анализ_пример.xlsx",
    )


def handle_download_cluster(bot, call, command):
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
    clear_user_files(call.from_user.id)
    get_user_file(bot, command)


# def get_file_for_cluster_analysis(bot):
#     """
#     Обработка загрузки файла для кластерного анализа.
#     """
#
#     @bot.message_handler(content_types=["document"])
#     def handle_document(message):
#         try:
#             file_info = bot.get_file(message.document.file_id)
#
#             file_url = f"https://api.telegram.org/file/bot{test_bot_token}/{file_info.file_path}"
#
#             response = requests.get(file_url)
#
#             if response.status_code == 200:
#                 file_name = save_file(
#                     response.content, message.document.file_name,
#                     message.chat.id,
#                     CLUSTER_ANALYSIS
#                 )
#                 print(file_name)
#                 check_input_file(bot, message, file_name)
#
#             else:
#                 bot.reply_to(message, "Произошла ошибка при загрузке файла")
#
#         except ApiTelegramException as e:
#             if e.description == "Bad Request: file is too big":
#                 bot.reply_to(
#                     message=message,
#                     text="Ваш файл превышает допустимый лимит 20 Мегабайт.",
#                 )
#
#             else:
#                 bot.reply_to(message, "Произошла ошибка при загрузке файла")
#
#             print(f"Error: {e}")
#
#         except RequestException as e:
#             print(f"Error while downloading file: {e}")
#             bot.reply_to(message, "Произошла ошибка при загрузке файла")
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             bot.reply_to(message, "Произошла ошибка при загрузке файла")


def handle_downloaded_cluster_file(bot, call, command):
    """
    Обработка файла, присланного пользователем для дальнейших расчетов.
    """
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите интересующий Вас метод кластеризации:",
        reply_markup=keyboard_choice_cluster,
    )


def handle_cluster_k_means(bot, call):
    """
    Обработка при нажатии на "Метод k-средних"
    после прочтения файла кластерного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ClusterModule(df, call.from_user.id)

    optimal_clusters = module.elbow_method_and_optimal_clusters(max_clusters=15)

    chat_id = call.from_user.id

    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{chat_id}.png"

    if os.path.isfile(file_path):
        file = open(file_path, "rb")

        bot.send_photo(chat_id=chat_id, photo=file)

        bot.send_message(
            chat_id=chat_id,
            text=f"На основе Ваших данных был построен Метод Локтя для определения "
                 f"оптимального количества кластеров по данным. "
                 f"Рекомендованное количество кластеров – {optimal_clusters}. "
                 "Вы можете оставить рекомендованное количество кластеров, либо выбрать количество кластеров самостоятельно.",
            reply_markup=keyboard_choice_number_of_clusters
        )
