from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup


def handle_pagination_columns_variance(bot, call, command, columns) -> None:
    data = call.data.split("_") if "_" in call.data else (call.data, 0)
    if command.startswith("test_kruskal_wallis"):
        l, _, prefix, action, page = data[0], data[1], data[2], data[3], int(
            data[4])
    else:
        _, prefix, action, page = data[0], data[1], data[2], int(
            data[3])

    if action == "prev":
        page -= 1

    edit_column_selection_message(
        bot, call.message.chat.id, call.message.message_id, columns, page,
        command
    )


def edit_column_selection_message(bot, chat_id, message_id, columns, page,
                                  command):
    """
    Редактирует сообщение для выбора столбца.

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
        text="Выберите следующую переменную:",
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

    if command.startswith("test_kruskal_wallis"):
        for index, column in enumerate(current_columns):
            button = InlineKeyboardButton(
                column,
                callback_data=f"test_kruskal_wallis_{start_index + index}"
            )
            keyboard.add(button)

        add_pagination_buttons(keyboard, columns, page, command)

    elif command.startswith("test_friedman"):
        for index, column in enumerate(current_columns):
            button = InlineKeyboardButton(
                column,
                callback_data=f"test_friedman_{start_index + index}"
            )
            keyboard.add(button)

        add_pagination_buttons(keyboard, columns, page, command)

    return keyboard


def add_pagination_buttons(
        keyboard: InlineKeyboardMarkup, columns: list, page: int, command: str
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
    prev_button, next_button = None, None

    if command.startswith("test_kruskal_wallis"):
        prev_button = (
            InlineKeyboardButton(
                "Назад",
                callback_data=f"test_kruskal_wallis_prev_{page}",
            )
            if page > 0
            else None
        )
        next_button = (
            InlineKeyboardButton(
                "Далее",
                callback_data=f"test_kruskal_wallis_next_{page + 1}",
            )
            if (page + 1) * 4 < len(columns)
            else None
        )

    elif command.startswith("test_friedman"):
        prev_button = (
            InlineKeyboardButton(
                "Назад",
                callback_data=f"test_friedman_prev_{page}",
            )
            if page > 0
            else None
        )
        next_button = (
            InlineKeyboardButton(
                "Далее",
                callback_data=f"test_friedman_next_{page + 1}",
            )
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
