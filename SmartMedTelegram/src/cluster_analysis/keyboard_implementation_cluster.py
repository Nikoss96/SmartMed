from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton


def handle_pagination_columns_cluster(bot, call, command) -> None:
    columns = [i + 1 for i in range(10)]

    if command.startswith("cluster_"):
        data = call.data.split("_") if "_" in call.data else (call.data, 0)
        _, action, page = data[0], data[1], int(data[2])

        if action == "prev":
            page -= 1

        edit_column_selection_message(
            bot, call.message.chat.id, call.message.message_id, columns, page,
            command
        )

    elif command.startswith("hierarchical"):
        data = call.data.split("_") if "_" in call.data else (call.data, 0)
        _, prefix, action, page = data[0], data[1], data[2], int(data[3])

        if action == "prev":
            page -= 1

        edit_column_selection_message(
            bot, call.message.chat.id, call.message.message_id, columns, page,
            command
        )


def edit_column_selection_message(bot, chat_id, message_id, columns, page,
                                  command):
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
    keyboard = generate_column_keyboard(columns, page, command)

    bot.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text="Выберите количество кластеров:",
        reply_markup=keyboard,
    )


def generate_column_keyboard(columns: list, page: int,
                             command) -> InlineKeyboardMarkup:
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

    prefix = ""
    if "hierarchical" in command:
        prefix = "hierarchical_"

    for index, column in enumerate(current_columns):
        button = InlineKeyboardButton(
            column, callback_data=f"{prefix}cluster_{start_index + index + 1}"
        )
        keyboard.add(button)

    add_pagination_buttons(keyboard, columns, page, prefix)

    return keyboard


def add_pagination_buttons(
        keyboard: InlineKeyboardMarkup, columns: list, page: int, prefix
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
        InlineKeyboardButton("Назад",
                             callback_data=f"{prefix}cluster_prev_{page}")
        if page > 0
        else None
    )
    next_button = (
        InlineKeyboardButton("Далее",
                             callback_data=f"{prefix}cluster_next_{page + 1}")
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
