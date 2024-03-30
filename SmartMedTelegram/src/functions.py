import os

import pandas as pd
import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException

from cluster_analysis.keyboard_cluster import keyboard_cluster_analysis, \
    keyboard_replace_null_values_cluster
from describe_analysis.keyboard_descriptive import (
    keyboard_describe_analysis, keyboard_replace_null_values_describe,
)
from dictionary.functions_dictionary import generate_dictionary_keyboard
from keyboard import keyboard_in_development, keyboard_modules, \
    keyboard_main_menu
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    USER_DATA_PATH,
    DESCRIBE_ANALYSIS,
    DESCRIBE_TABLES,
)
from preprocessing.preprocessing import PandasPreprocessor

"6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"
test_bot_token = "6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"
user_commands = {}


def get_reply_markup(command):
    """
    Вспомогательная функция для получения клавиатур.
    """
    switch = {
        "bioequal": keyboard_in_development,
        "описательный анализ": keyboard_describe_analysis,
        "кластерный анализ": keyboard_cluster_analysis,
        "predict": keyboard_in_development,
        "модули": keyboard_modules,
        "назад": keyboard_main_menu,
        "словарь": generate_dictionary_keyboard(0),
        "cluster": keyboard_in_development,
    }
    return switch.get(command, None)


def send_text_message(bot, chat_id, text, reply_markup=None):
    """
    Вспомогательная функция для отправки текстовых сообщений.
    """
    bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


def save_file(file_content, file_name, chat_id):
    directory = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}"
    pattern = f"{chat_id}"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                return

    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/{chat_id}_{file_name}"
    with open(file_path, "wb") as file:
        file.write(file_content)
    return file_path


def get_user_file(bot):
    """
    Обработка загрузки файла для описательного анализа.
    """

    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        try:
            command = user_commands.pop(message.chat.id)
            if not command:
                raise ApiTelegramException
            file_info = bot.get_file(message.document.file_id)

            file_url = f"https://api.telegram.org/file/bot{test_bot_token}/{file_info.file_path}"

            response = requests.get(file_url)

            if response.status_code == 200:
                file_name = save_file(
                    response.content, message.document.file_name,
                    message.chat.id,
                )
                check_input_file(bot, message, file_name, command)

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


def find_user_file(chat_id):
    directory = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/"
    pattern = f"{chat_id}"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                return os.path.join(root, file)


def handle_back(bot, user_id):
    """
    Обработка запроса на возвращение к выбору модуля.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        user_id (int): Идентификатор пользователя.
    """
    bot.send_message(
        chat_id=user_id,
        text="Выберите интересующий Вас раздел:",
        reply_markup=keyboard_main_menu,
    )


def send_document_from_file(bot, chat_id, file_path):
    """
    Отправка документа из файла.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        chat_id (int): Идентификатор чата.
        file_path (str): Путь к файлу.
    """
    if os.path.isfile(file_path):
        file = open(file_path, "rb")
        bot.send_document(chat_id=chat_id, document=file)


def clear_user_files(chat_id):
    """
    Очистка старых файлов пользователя при загрузке нового.
    """
    directory = f"{MEDIA_PATH}/{DATA_PATH}"
    pattern = f"{chat_id}"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)


def handle_download(bot, call, command):
    """
    Обработка запроса на загрузку файла для descriptive analysis.

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
    if call.from_user.id in user_commands:
        user_commands.pop(call.from_user.id)

    user_commands[call.from_user.id] = command
    clear_user_files(call.from_user.id)
    get_user_file(bot)


def check_input_file(bot, message, file_path, command):
    """
    Начальная проверка файла, загруженного
    пользователем на расширение, размер и данные.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        supported_formats = [".csv", ".xlsx", ".xls"]

        if file_extension not in supported_formats:
            bot.reply_to(
                message,
                "Ваш файл не подходит. "
                "Файл должен иметь формат .csv, "
                ".xlsx или .xls",
            )
            if os.path.exists(file_path):
                os.remove(file_path)
            return

        file_size = os.path.getsize(file_path)
        max_file_size = 20 * 1024 * 1024  # 20 Мегабайт

        if file_size > max_file_size:
            bot.reply_to(
                message,
                f"Ваш файл превышает допустимый лимит "
                f"{max_file_size // (1024 * 1024)}.",
            )
            if os.path.exists(file_path):
                os.remove(file_path)
            return

        df = None

        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        if df is not None:
            if command == "download_describe":
                bot.reply_to(
                    message,
                    f"Файл {message.document.file_name} успешно прочитан."
                    f" Выберите метод обработки пустых значений в Вашем файле:",
                    reply_markup=keyboard_replace_null_values_describe,
                )
            elif command == "download_cluster":
                bot.reply_to(
                    message,
                    f"Файл {message.document.file_name} успешно прочитан."
                    f" Выберите метод обработки пустых значений в Вашем файле:",
                    reply_markup=keyboard_replace_null_values_cluster,
                )

            return True

        else:
            bot.reply_to(
                message,
                "Ваш файл не подходит. Прочитайте требования к столбцам, "
                "измените данные и попробуйте еще раз.",
            )
            if os.path.exists(file_path):
                os.remove(file_path)
            return

    except Exception as e:
        print(f"Error preprocessing file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        bot.reply_to(
            message,
            "Ошибка в чтении Вашего файла. "
            "Попробуйте еще раз или загрузите новый файл",
        )


def create_dataframe_and_save_file(chat_id, command):
    # Найти загруженный файл пользователя
    directory = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}"
    files_in_directory = os.listdir(directory)

    file_name = [file for file in files_in_directory if
                 file.startswith(f"{chat_id}")]

    # Формируем настройки для корректной предобработки данных
    path = f"{directory}/{file_name[0]}"
    command = command.split("_")
    settings = {}
    settings["path"] = path
    settings["fillna"] = command[3]
    settings["encoding"] = "label_encoding"

    PandasPreprocessor(settings, chat_id)
