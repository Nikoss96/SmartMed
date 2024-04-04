from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup


def handle_choose_column_comparative(bot, call, columns, command):
    """
    Отправляет сообщение для выбора независимой переменной.

    Parameters:
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    Returns:
        None
    """
    if command == "categorical_columns":
        keyboard = generate_column_keyboard(columns, 0, command)

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите группирующую переменную:",
            reply_markup=keyboard,
        )
    else:
        keyboard = generate_column_keyboard(columns, 0, command)

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите независимую переменную:",
            reply_markup=keyboard,
        )


def handle_pagination_columns_comparative(bot, call, command, columns) -> None:
    # if command.startswith("continuous_columns_"):
    data = call.data.split("_") if "_" in call.data else (call.data, 0)
    _, prefix, action, page = data[0], data[1], data[2], int(data[3])

    if action == "prev":
        page -= 1

    edit_column_selection_message(
        bot, call.message.chat.id, call.message.message_id, columns, page, command
    )


def edit_column_selection_message(bot, chat_id, message_id, columns, page, command):
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

    if command.startswith("continuous_columns_"):
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text="Выберите независимую переменную:",
            reply_markup=keyboard,
        )

    else:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text="Выберите группирующую переменную:",
            reply_markup=keyboard,
        )


def generate_column_keyboard(columns: list, page: int, command) -> InlineKeyboardMarkup:
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

    if command.startswith("categorical_columns"):
        for index, column in enumerate(current_columns):
            button = InlineKeyboardButton(
                column, callback_data=f"categorical_column_{start_index + index}"
            )
            keyboard.add(button)

        add_pagination_buttons(keyboard, columns, page, command)

    else:
        for index, column in enumerate(current_columns):
            button = InlineKeyboardButton(
                column, callback_data=f"continuous_column_{start_index + index}"
            )
            keyboard.add(button)

        add_pagination_buttons(keyboard, columns, page, command)

    return keyboard


def generate_categorical_value_column_keyboard(columns: dict) -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с названиями колонок для пагинации.

    Parameters:
        columns (dict): Список названий колонок.

    Returns:
        InlineKeyboardMarkup: Созданная встроенная клавиатура.
    """
    keyboard = InlineKeyboardMarkup()

    for key, value in columns.items():
        button = InlineKeyboardButton(
            str(value), callback_data=f"t_criteria_categorical_value_{key}"
        )
        keyboard.add(button)

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

    if command.startswith("categorical_columns"):
        prev_button = (
            InlineKeyboardButton(
                "Назад", callback_data=f"categorical_columns_prev_{page}"
            )
            if page > 0
            else None
        )
        next_button = (
            InlineKeyboardButton(
                "Далее", callback_data=f"categorical_columns_next_{page + 1}"
            )
            if (page + 1) * 4 < len(columns)
            else None
        )

    else:
        prev_button = (
            InlineKeyboardButton(
                "Назад", callback_data=f"continuous_columns_prev_{page}"
            )
            if page > 0
            else None
        )
        next_button = (
            InlineKeyboardButton(
                "Далее", callback_data=f"continuous_columns_next_{page + 1}"
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
