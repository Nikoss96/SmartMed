import os

import pandas as pd
import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

from keyboard import (
    keyboard00,
    keyboard01,
    keyboard_in_development,
    keyboard_main_menu,
    keyboard_modules,
    keyboard_choose_describe,
)
from statistical_terms import statistical_terms
from describe_mid import display_correlation_matrix, make_df_plot, make_plots
from paths import (
    MEDIA_PATH,
    DATA_PATH,
    IMAGES_PATH,
    TERMS_PATH,
    USER_DATA_PATH,
    SENDING_FILES_PATH,
)

"6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"
test_bot_token = "6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"


# Constants


def get_reply_markup(command):
    """
    Вспомогательная функция для получения клавиатур.
    """
    switch = {
        "bioequal": keyboard_in_development,
        "describe": keyboard01,
        "predict": keyboard_in_development,
        "модули": keyboard_modules,
        "назад": keyboard_main_menu,
        "словарь": generate_dictionary_keyboard(0),
        "chat-gpt": keyboard_in_development,
        "cluster": keyboard_in_development,
    }
    return switch.get(command, None)


def send_text_message(bot, chat_id, text, reply_markup=None):
    """
    Вспомогательная функция для отправки текстовых сообщений.
    """
    bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


def get_anyfile(bot, call):
    """
    Обработка загрузки любого файла.
    """

    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        try:
            file_info = bot.get_file(message.document.file_id)
            file_url = f"https://api.telegram.org/file/bot{test_bot_token}/{file_info.file_path}"

            response = requests.get(file_url)

            if response.status_code == 200:
                file_name = f"{MEDIA_PATH}/{IMAGES_PATH}/{message.document.file_name}"

                with open(file_name, "wb") as file:
                    file.write(response.content)

                bot.reply_to(
                    message=message,
                    text=f"Файл {message.document.file_name} успешно загружен",
                )
                send_document_from_file(bot, call.from_user.id, file_name)

            else:
                bot.reply_to(message, "Произошла ошибка при загрузке файла")

        except (ApiTelegramException, RequestException) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


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
                file_name = download_file(response.content,
                                          message.document.file_name)
                preprocess_input_file(bot, message, file_name)

            else:
                bot.reply_to(message, "Произошла ошибка при загрузке файла")
                # Prompt user to upload another file
                bot.reply_to(message, "Пожалуйста, отправьте файл для анализа.")

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


def download_file(file_content, file_name):
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/{file_name}"
    with open(file_path, "wb") as file:
        file.write(file_content)
    return file_path


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
            df = preprocess_dataframe(df)

            display_correlation_matrix(dataframe=df, chat_id=message.chat.id)
            make_plots(df=df, chat_id=message.chat.id)

            bot.reply_to(
                message,
                f"Файл {message.document.file_name} успешно прочитан.",
                reply_markup=keyboard_choose_describe,
            )

    except Exception as e:
        print(f"Error preprocessing file: {e}")
        bot.reply_to(
            message,
            "Ошибка в чтении вашего файла. "
            "Попробуйте еще раз или пришлите новый файл",
        )


def preprocess_dataframe(df):
    df.fillna("", inplace=True)
    max_row_length = max(df.apply(lambda x: x.astype(str).map(len)).max())
    return df.apply(
        lambda x: x.astype(str).map(lambda x: x.ljust(max_row_length)))


def generate_dictionary_keyboard(page):
    """
    Генерация клавиатуры с терминами для словаря.
    """
    keyboard_terms = InlineKeyboardMarkup()
    words_per_page = 4

    for term_key in list(statistical_terms.keys())[
                    page * words_per_page: (page + 1) * words_per_page
                    ]:
        term_description = statistical_terms[term_key][0]
        button = InlineKeyboardButton(
            term_description, callback_data=f"statistical_{term_key}"
        )
        keyboard_terms.add(button)

    prev_button = (
        InlineKeyboardButton("Назад", callback_data=f"prev_{page}")
        if page > 0
        else None
    )
    next_button = (
        InlineKeyboardButton("Далее", callback_data=f"next_{page + 1}")
        if (page + 1) * words_per_page < len(statistical_terms)
        else None
    )
    home_button = InlineKeyboardButton("Главное меню", callback_data="back")

    if prev_button and next_button:
        keyboard_terms.add(prev_button, home_button, next_button)
    elif prev_button:
        keyboard_terms.add(prev_button, home_button)
    elif next_button:
        keyboard_terms.add(home_button, next_button)
    else:
        keyboard_terms.add(home_button)

    return keyboard_terms


def open_and_send_file(bot, chat_id, image):
    """
    Открытие и отправка изображения по его имени.
    """
    file_path = f"{MEDIA_PATH}/{IMAGES_PATH}/{TERMS_PATH}/{image}.png"

    if os.path.isfile(file_path):
        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)


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


def handle_pagination(bot, call):
    """
    Обработка нажатия кнопок "Prev" и "Next" для пагинации терминов.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    action, page = call.data.split("_") if "_" in call.data else (call.data, 0)
    page = int(page)

    if action == "prev":
        page -= 1

    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="Выберите интересующий вас термин:",
        reply_markup=generate_dictionary_keyboard(page),
    )


def handle_statistical_term(bot, call):
    """
    Обработка выбора статистического термина.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    term = call.data.replace("statistical_term_", "")
    bot.send_message(
        chat_id=call.from_user.id,
        text=" – это ".join(statistical_terms[f"term_{term}"]),
    )
    open_and_send_file(bot, call.from_user.id, call.data)


def handle_example_bioequal(bot, call):
    """
    Обработка запроса на пример файла для bioequal.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id,
        text="Прислали вам пример файла. Оформляйте в точности так.",
    )
    send_document_from_file(
        bot, call.from_user.id,
        f"{MEDIA_PATH}/{DATA_PATH}/параллельный тестовый.xlsx"
    )


def handle_download_bioequal(bot, call):
    """
    Обработка запроса на загрузку файла для bioequal.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id, text="Можете прислать свой файл прямо сюда."
    )
    get_anyfile(bot, call)


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
        f"{MEDIA_PATH}/{DATA_PATH}/Описательный_анализ_пример.xlsx",
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


def handle_back(bot, user_id):
    """
    Обработка запроса на возвращение к выбору модуля.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        user_id (int): Идентификатор пользователя.
    """
    bot.send_message(
        chat_id=user_id,
        text="Выберите интересующий вас раздел:",
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
    file = open(file_path, "rb")
    bot.send_document(chat_id=chat_id, document=file)


def handle_describe_build_graphs(bot, call):
    """
    Обработка при нажатии на "Построение графиков"
    после прочтения файла описательного анализа.
    """
    bot.send_message(
        chat_id=call.from_user.id,
        text="По каждому параметру приложенных "
             "Вами данных была построена гистограмма. "
             "Результаты представлены на дашборде."
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
             "Результаты представлены на дашборде."
    )
    send_correlation_file(bot, call.from_user.id)
