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
