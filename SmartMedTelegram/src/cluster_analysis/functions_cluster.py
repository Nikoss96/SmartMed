import os

from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from cluster_analysis.ClusterModule import ClusterModule
from cluster_analysis.keyboard_cluster import (
    keyboard_choice_cluster,
    keyboard_choice_number_of_clusters,
)
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    CLUSTER_ANALYSIS,
    USER_DATA_PATH,
    ELBOW_METHOD,
    EXAMPLES,
    K_MEANS,
)
from describe_analysis.DescribeModule import (
    filter_columns_with_more_than_2_unique_values,
)
from describe_analysis.functions_descriptive import get_user_file_df
from functions import send_document_from_file, create_dataframe_and_save_file

number_of_clusters = {}


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

    df = filter_columns_with_more_than_2_unique_values(df)

    module = ClusterModule(df, call.from_user.id)

    optimal_clusters = module.elbow_method_and_optimal_clusters(max_clusters=10)

    number_of_clusters[call.from_user.id] = optimal_clusters
    chat_id = call.from_user.id

    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{chat_id}.png"

    if os.path.isfile(file_path):
        file = open(file_path, "rb")

        bot.send_photo(chat_id=chat_id, photo=file)

        bot.send_message(
            chat_id=chat_id,
            text=f"На основе Ваших данных был построен Метод Локтя для определения "
            f"оптимального количества кластеров по данным.\n\n"
            f"Рекомендованное количество кластеров – {optimal_clusters}.\n\n"
            "Вы можете оставить рекомендованное количество кластеров, либо выбрать количество кластеров самостоятельно.",
            reply_markup=keyboard_choice_number_of_clusters,
        )


def handle_choose_number_of_clusters(bot, call):
    """
    Отправляет сообщение для выбора количества кластеров.

    Parameters:
        bot (telegram.Bot): Объект бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    Returns:
        None
    """
    columns = [i + 1 for i in range(20)]

    keyboard = generate_column_keyboard(columns, 0)

    if call.from_user.id in number_of_clusters:
        number_of_clusters.pop(call.from_user.id)

    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите количество кластеров:",
        reply_markup=keyboard,
    )


def handle_pagination_columns_cluster(bot, call) -> None:
    columns = [i + 1 for i in range(20)]

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
        text="Выберите количество кластеров:",
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
            column, callback_data=f"cluster_{start_index + index + 1}"
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
        InlineKeyboardButton("Назад", callback_data=f"cluster_prev_{page}")
        if page > 0
        else None
    )
    next_button = (
        InlineKeyboardButton("Далее", callback_data=f"cluster_next_{page + 1}")
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


def handle_cluster_numbers(bot, call, command):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = filter_columns_with_more_than_2_unique_values(df)

    if command.startswith("cluster_"):
        n_clusters = int(call.data.replace("cluster_", ""))
    else:
        n_clusters = number_of_clusters.pop(call.from_user.id)

    module = ClusterModule(df, call.from_user.id)
    module.generate_k_means(n_clusters)

    chat_id = call.from_user.id

    png_file_path = (
        f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{chat_id}.png"
    )
    excel_file_path = (
        f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{chat_id}.xlsx"
    )

    if os.path.isfile(png_file_path) and os.path.isfile(excel_file_path):
        bot.send_message(
            chat_id=call.from_user.id,
            text="По заданному количеству кластеров с помощью метода k-средних"
            " была построен точечный график распределения элементов,"
            " а также создана таблица распределения элементов.",
        )

        file_cur = open(png_file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)

        file_cur = open(excel_file_path, "rb")
        bot.send_document(
            chat_id=chat_id,
            document=file_cur,
            visible_file_name="Принадлежность_элементов_к_кластерам.xlsx",
        )
