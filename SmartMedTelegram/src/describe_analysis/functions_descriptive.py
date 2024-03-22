import os
import time

import pandas as pd
import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException
from telebot.types import (
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from describe_analysis.DescribeModule import (
    DescribeModule,
    filter_columns_with_more_than_2_unique_values,
)
from describe_analysis.keyboard_descriptive import (
    keyboard_replace_null_values,
    keyboard_choice,
)
from describe_analysis.utils.preprocessing import PandasPreprocessor

from functions import save_file, send_document_from_file
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    USER_DATA_PATH,
    DESCRIBE_ANALYSIS,
    DESCRIBE_TABLES,
    CORRELATION_MATRICES,
    PLOTS,
    BOXPLOTS,
)

test_bot_token = "6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"


def handle_example_describe(bot, call):
    """
    Обработка запроса на пример файла для descriptive analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id,
        text="Прислали пример файла. Оформляйте в точности так.",
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
        text="Пришлите Ваш файл.\n\n"
        "Файл должен иметь следующие характеристики:\n"
        "\n1.  Формат файла: .csv, .xlsx или .xls"
        "\n2.  Размер файла: до 20 Мегабайт"
        "\n3.  Файл должен иметь не более 25 параметров (столбцов)"
        "\n4.  Содержимое файла: Название каждого столбца "
        "должно быть читаемым.",
    )
    clear_user_files_descriptive_analysis(call.from_user.id)
    get_file_for_descriptive_analysis(bot)


def clear_user_files_descriptive_analysis(chat_id):
    """
    Очистка старых файлов пользователя при загрузке нового.
    """
    directory = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}"
    pattern = f"{chat_id}"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)


def get_file_for_descriptive_analysis(bot):
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
                file_name = save_file(
                    response.content, message.document.file_name, message.chat.id
                )
                check_input_file_descriptive(bot, message, file_name)

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


def check_input_file_descriptive(bot, message, file_path):
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
            bot.reply_to(
                message,
                f"Файл {message.document.file_name} успешно прочитан."
                f" Выберите метод обработки пустых значений в вашем файле:",
                reply_markup=keyboard_replace_null_values,
            )

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
            "Попробуйте еще раз или пришлите новый файл",
        )


def handle_downloaded_describe_file(bot, call, command):
    """
    Обработка файла, присланного пользователем для дальнейших расчетов.
    """
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите элемент описательного анализа,"
        " который хотите рассчитать по своим данным:",
        reply_markup=keyboard_choice,
    )


def create_dataframe_and_save_file(chat_id, command):
    # Найти загруженный файл пользователя
    directory = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}"
    files_in_directory = os.listdir(directory)

    file_name = [file for file in files_in_directory if file.startswith(f"{chat_id}")]

    # Формируем настройки для корректной предобработки данных
    path = f"{directory}/{file_name[0]}"
    command = command.split("_")
    settings = {}
    settings["path"] = path
    settings["fillna"] = command[3]
    settings["encoding"] = "label_encoding"

    PandasPreprocessor(settings, chat_id)


def get_user_file_df(directory, chat_id):
    files = os.listdir(directory)
    pattern = f"{chat_id}"

    matching_files = [file for file in files if pattern in file]

    if matching_files:
        file_path = os.path.join(directory, matching_files[0])
        df = pd.read_excel(file_path)
        return df


def handle_describe_build_graphs(bot, call):
    """
    Обработка при нажатии на "Построение графиков"
    после прочтения файла описательного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = DescribeModule(df, call.from_user.id)

    module.make_plots()

    chat_id = call.from_user.id
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{PLOTS}/describe_plots_{chat_id}.png"

    if os.path.isfile(file_path):
        bot.send_message(
            chat_id=call.from_user.id,
            text="По каждому параметру приложенных "
            "Вами данных была построена гистограмма. "
            "Результаты представлены на дашборде.",
        )

        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)


def handle_describe_correlation_analysis(bot, call):
    """
    Обработка при нажатии на "Корреляционный анализ"
    после прочтения файла описательного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = DescribeModule(df, call.from_user.id)

    module.create_correlation_matrices()

    chat_id = call.from_user.id
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{CORRELATION_MATRICES}/describe_corr_{chat_id}.png"

    if os.path.isfile(file_path):
        file_cur = open(file_path, "rb")

        bot.send_message(
            chat_id=chat_id,
            text="На основе Ваших данных были построены матрицы корреляций"
            " с помощью коэффициентов корреляции Пирсона и Спирмена. "
            "Результаты представлены на дашборде.",
        )

        bot.send_photo(chat_id=chat_id, photo=file_cur)


def handle_describe_table(bot, call):
    """
    Обработка при нажатии на "Описательная таблица"
    после прочтения файла описательного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = DescribeModule(df, call.from_user.id)

    module.generate_table()

    chat_id = call.from_user.id

    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{DESCRIBE_TABLES}/{chat_id}_describe_table.xlsx"

    if os.path.isfile(file_path):
        file = open(file_path, "rb")

        bot.send_message(
            chat_id=chat_id,
            text="На основе Ваших данных была составлена описательная таблица"
            " с вычисленными основными описательными характеристиками. "
            "Результаты отправлены в качестве Excel файла.",
        )

        bot.send_document(
            chat_id=chat_id,
            document=file,
            visible_file_name="Описательная_таблица.xlsx",
        )


def handle_describe_box_plot(bot, call):
    """
    Обработка при нажатии на "Ящик с усами"
    после прочтения файла описательного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    send_column_selection_message(bot, call.from_user.id, df)


def send_column_selection_message(bot, user_id, df):
    """
    Отправляет сообщение для выбора столбца для построения ящика с усами.

    Parameters:
        bot (telegram.Bot): Объект бота.
        user_id (int): ID пользователя.
        df (pandas.DataFrame): DataFrame с данными.

    Returns:
        None
    """

    df = filter_columns_with_more_than_2_unique_values(df)

    columns = df.columns.tolist()
    keyboard = generate_column_keyboard(columns, 0)

    bot.send_message(
        chat_id=user_id,
        text="Выберите столбец для построения ящика с усами:",
        reply_markup=keyboard,
    )


def handle_pagination_columns(bot, call) -> None:
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = filter_columns_with_more_than_2_unique_values(df)

    columns = df.columns.tolist()

    data = call.data.split("_") if "_" in call.data else (call.data, 0)
    _, action, page = data[0], data[1], int(data[2])

    if action == "prev":
        page -= 1

    edit_column_selection_message(
        bot, call.message.chat.id, call.message.message_id, columns, page
    )


def edit_column_selection_message(bot, chat_id, message_id, columns, page):
    """
    Редактирует сообщение для выбора столбца для построения ящика с усами.

    Parameters:
        bot (telegram.Bot): Объект бота.
        chat_id (int): ID чата.
        message_id (int): ID сообщения.
        columns (list): Список названий колонок.
        page (int): Номер страницы.

    Returns:
        None
    """
    keyboard = generate_column_keyboard(columns, page)

    bot.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text="Выберите столбец для построения ящика с усами:",
        reply_markup=keyboard,
    )


def generate_column_keyboard(columns: list, page: int) -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с названиями колонок для пагинации.

    Parameters:
        columns (list): Список названий колонок.
        page (int): Номер страницы.

    Returns:
        InlineKeyboardMarkup: Созданная встроенная клавиатура.
    """
    keyboard = InlineKeyboardMarkup()
    columns_per_page = 4
    start_index = page * columns_per_page
    end_index = min((page + 1) * columns_per_page, len(columns))
    current_columns = columns[start_index:end_index]

    for index, column in enumerate(current_columns):
        button = InlineKeyboardButton(
            column, callback_data=f"column_{start_index + index}"
        )
        keyboard.add(button)

    add_pagination_buttons(keyboard, columns, page)

    return keyboard


def add_pagination_buttons(
    keyboard: InlineKeyboardMarkup, columns: list, page: int
) -> None:
    """
    Добавляет кнопки пагинации на клавиатуру.

    Parameters:
        keyboard (InlineKeyboardMarkup): Объект клавиатуры.
        columns (list): Список названий колонок.
        page (int): Номер страницы.

    Returns:
        None
    """
    prev_button = (
        InlineKeyboardButton("Назад", callback_data=f"boxplot_prev_{page}")
        if page > 0
        else None
    )
    next_button = (
        InlineKeyboardButton("Далее", callback_data=f"boxplot_next_{page + 1}")
        if (page + 1) * 4 < len(columns)
        else None
    )
    home_button = InlineKeyboardButton("Главное меню", callback_data="back")

    if prev_button and next_button:
        keyboard.row(prev_button, home_button, next_button)
    elif prev_button:
        keyboard.row(prev_button, home_button)
    elif next_button:
        keyboard.row(home_button, next_button)
    else:
        keyboard.row(home_button)


def handle_box_plot(bot, call):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = filter_columns_with_more_than_2_unique_values(df)

    columns = df.columns.tolist()

    column = int(call.data.replace("column_", ""))

    module = DescribeModule(df, call.from_user.id)
    module.generate_box_hist(columns[column])

    chat_id = call.from_user.id
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{USER_DATA_PATH}/{BOXPLOTS}/describe_boxplot_{chat_id}.png"

    if os.path.isfile(file_path):
        bot.send_message(
            chat_id=call.from_user.id,
            text="По данному столбцу был построен Ящик с Усами:",
        )

        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)
