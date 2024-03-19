import os

import pandas as pd
import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException
from describe_analysis.keyboard_descriptive import (
    keyboard_choice,
)


from functions import save_file, send_document_from_file
from describe_analysis.describe_mid import display_correlation_matrix, make_plots
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    USER_DATA_PATH,
    SENDING_FILES_PATH,
    DESCRIBE_ANALYSIS,
)

test_bot_token = "6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"


def get_file_for_descriptive_analysis(bot, call):
    """
    Обработка загрузки файла для описательного анализа.
    """

    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        try:
            file_info = bot.get_file(message.document.file_id)

            file_url = f"https://api.telegram.org/file/bot{test_bot_token}/{file_info.file_path}"

            response = requests.get(file_url)

            if response.status_code == 200:
                file_name = save_file(response.content, message.document.file_name)
                preprocess_input_file(bot, message, file_name)

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


def preprocess_input_file(bot, message, file_path):
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
            return

        file_size = os.path.getsize(file_path)
        max_file_size = 20 * 1024 * 1024  # 20 Мегабайт

        if file_size > max_file_size:
            bot.reply_to(
                message,
                f"Ваш файл превышает допустимый лимит "
                f"{max_file_size // (1024 * 1024)}.",
            )
            os.remove(file_path)
            return

        df = None

        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        if df is not None:
            # df = check_dataframe(df)

            # IF BAD FILE -> DELETE

            # display_correlation_matrix(dataframe=df, chat_id=message.chat.id)
            # make_plots(df=df, chat_id=message.chat.id)

            bot.reply_to(
                message,
                f"Файл {message.document.file_name} успешно прочитан.",
                reply_markup=keyboard_choice,
            )

    except Exception as e:
        print(f"Error preprocessing file: {e}")
        bot.reply_to(
            message,
            "Ошибка в чтении вашего файла. "
            "Попробуйте еще раз или пришлите новый файл",
        )


def check_dataframe(df):
    df.fillna("", inplace=True)
    max_row_length = max(df.apply(lambda x: x.astype(str).map(len)).max())
    return df.apply(lambda x: x.astype(str).map(lambda x: x.ljust(max_row_length)))


def send_correlation_file(bot, chat_id):
    """
    Открытие и отправка картинки рассчитанного корреляционного анализа.
    """
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/{SENDING_FILES_PATH}/describe_corr_{chat_id}.png"

    if os.path.isfile(file_path):
        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)


def send_describe_plots_file(bot, chat_id):
    """
    Открытие и отправка гистограмм из датафрейма описательного анализа.
    """
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/{SENDING_FILES_PATH}/describe_plots_{chat_id}.png"

    if os.path.isfile(file_path):
        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)


def handle_example_describe(bot, call):
    """
    Обработка запроса на пример файла для descriptive analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id,
        text="Прислали вам пример файла. Оформляйте в точности так.",
    )
    send_document_from_file(
        bot,
        call.from_user.id,
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/Описательный_анализ_пример.xlsx",
    )


def handle_download_describe(bot, call):
    """
    Обработка запроса на загрузку файла для descriptive analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.send_message(
        chat_id=call.from_user.id,
        text="Пришлите ваш файл.\n\n"
        "Файл должен иметь следующие характеристики:\n"
        "\n1.  Формат файла: .csv, .xlsx или .xls"
        "\n2.  Размер файла: до 20 Мегабайт"
        "\n3.  Файл должен иметь не более 25 параметров (столбцов)"
        "\n4.  Содержимое файла: Название каждого столбца "
        "должно быть читаемым.",
    )
    get_file_for_descriptive_analysis(bot, call)


def handle_describe_build_graphs(bot, call):
    """
    Обработка при нажатии на "Построение графиков"
    после прочтения файла описательного анализа.
    """
    bot.send_message(
        chat_id=call.from_user.id,
        text="По каждому параметру приложенных "
        "Вами данных была построена гистограмма. "
        "Результаты представлены на дашборде.",
    )
    send_describe_plots_file(bot, call.from_user.id)


def handle_describe_correlation_analysis(bot, call):
    """
    Обработка при нажатии на "Корреляционный анализ"
    после прочтения файла описательного анализа.
    """

    bot.send_message(
        chat_id=call.from_user.id,
        text="На основе Ваших данных были построены матрицы корреляций"
        " с помощью коэффициентов корреляции Пирсона и Спирмена. "
        "Результаты представлены на дашборде.",
    )
    send_correlation_file(bot, call.from_user.id)
