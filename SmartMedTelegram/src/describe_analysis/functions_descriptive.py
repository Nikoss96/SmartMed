import os

import pandas as pd
from telebot.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from describe_analysis.DescribeModule import (
    DescribeModule,
    filter_columns_with_more_than_2_unique_values,
)
from describe_analysis.keyboard_descriptive import (
    keyboard_choice_describe,
)

from functions import send_document_from_file, create_dataframe_and_save_file, \
    get_user_file_df
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    USER_DATA_PATH,
    DESCRIBE_ANALYSIS,
    DESCRIBE_TABLES,
    CORRELATION_MATRICES,
    PLOTS,
    BOXPLOTS,
    EXAMPLES,
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
        text="Прислали пример файла. Вы можете использовать этот файл для проведения анализа",
    )
    send_document_from_file(
        bot,
        call.from_user.id,
        f"{MEDIA_PATH}/{DATA_PATH}/{EXAMPLES}/Описательный_анализ_пример.xlsx",
    )


def handle_downloaded_describe_file(bot, call, command):
    """
    Обработка файла, присланного пользователем для дальнейших расчетов.
    """
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите интересующий Вас раздел:",
        reply_markup=keyboard_choice_describe,
    )


def handle_describe_build_graphs(bot, call):
    """
    Обработка при нажатии на "Построение графиков"
    после прочтения файла описательного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = DescribeModule(df, call.from_user.id)

    module.make_plots()

    chat_id = call.from_user.id
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{PLOTS}/describe_plots_{chat_id}.png"

    if os.path.isfile(file_path):
        bot.send_message(
            chat_id=call.from_user.id,
            text="По каждому параметру Ваших данных построена гистограмма."
                 " Результаты представлены на дашборде.",
        )

        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)


def handle_describe_correlation_analysis(bot, call):
    """
    Обработка при нажатии на "Корреляционный анализ"
    после прочтения файла описательного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = DescribeModule(df, call.from_user.id)

    module.create_correlation_matrices()

    chat_id = call.from_user.id
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{CORRELATION_MATRICES}/describe_corr_{chat_id}.png"

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
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = DescribeModule(df, call.from_user.id)

    module.generate_table()

    chat_id = call.from_user.id

    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{DESCRIBE_TABLES}/{chat_id}_describe_table.xlsx"

    if os.path.isfile(file_path):
        file = open(file_path, "rb")

        bot.send_message(
            chat_id=chat_id,
            text="На основе Ваших данных подготовлена описательная "
                 "таблица с основными статистиками. "
                 "Результаты представлены в прилагаемом Excel файле.",
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
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
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
        text="Выберите столбец для построения графика Ящик с усами:",
        reply_markup=keyboard,
    )


def handle_pagination_columns(bot, call) -> None:
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
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
        text="Выберите столбец для построения Ящика с усами:",
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
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = filter_columns_with_more_than_2_unique_values(df)

    columns = df.columns.tolist()

    column = int(call.data.replace("column_", ""))

    module = DescribeModule(df, call.from_user.id)
    module.generate_box_hist(columns[column])

    chat_id = call.from_user.id
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{DESCRIBE_ANALYSIS}/{BOXPLOTS}/describe_boxplot_{chat_id}.png"

    if os.path.isfile(file_path):
        bot.send_message(
            chat_id=call.from_user.id,
            text="Для данного параметра был построен график Ящик с усами:",
        )

        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)
