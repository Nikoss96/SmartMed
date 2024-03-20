from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

"""
keyboard_describe_analysis: Выбор работы с описательным анализом.
keyboard_choice: Выбор опции описательного анализа после загрузки файла.
"""

keyboard_describe_analysis = InlineKeyboardMarkup()
keyboard_describe_analysis.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_describe")
)
keyboard_describe_analysis.add(
    InlineKeyboardButton(text="Загрузить свой файл", callback_data="download_describe")
)
keyboard_describe_analysis.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)

keyboard_choice = InlineKeyboardMarkup()
keyboard_choice.add(
    InlineKeyboardButton(
        text="Построение графиков", callback_data="describe_build_graphs"
    )
)
keyboard_choice.add(
    InlineKeyboardButton(
        text="Корреляционный анализ", callback_data="describe_correlation_analysis"
    )
)
keyboard_choice.add(InlineKeyboardButton(text="Главное меню", callback_data="back"))

keyboard_replace_null_values = InlineKeyboardMarkup()

keyboard_replace_null_values.add(
    InlineKeyboardButton(
        text="Средним/модой", callback_data="replace_null_with_mean"
    )
)

keyboard_replace_null_values.add(
    InlineKeyboardButton(
        text="Удалять строки с пропущенными значениями", callback_data="delete_null_rows_dropna"
    )
)

keyboard_replace_null_values.add(
    InlineKeyboardButton(
        text="Медианой/модой", callback_data="replace_null_with_median"
    )
)