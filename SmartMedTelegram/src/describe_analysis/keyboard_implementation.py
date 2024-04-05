from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from data.paths import MEDIA_PATH, DATA_PATH, USER_DATA_PATH
from functions import get_user_file_df
from preprocessing.preprocessing import get_numeric_df


def handle_pagination_columns(bot, call) -> None:
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = get_numeric_df(df)

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
            column, callback_data=f"boxplot_column_{start_index + index}"
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
