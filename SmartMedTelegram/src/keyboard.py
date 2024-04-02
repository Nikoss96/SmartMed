from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)

"""
keyboard_main_menu: Первый ответ бота после нажатия пользователем кнопки Start.
keyboard_modules: Ответ бота после нажатия на "Модули" на стартовой клавиатуре.
keyboard_in_development: При отсутствии дальнейшего алгоритма у чат-бота, 
возвращается клавиатура с возможность вернуться домой.
"""

keyboard_main_menu = ReplyKeyboardMarkup(
    resize_keyboard=True,
    one_time_keyboard=True
)
keyboard_main_menu.row("Модули", "Словарь", "GPT")

keyboard_modules = ReplyKeyboardMarkup(
    resize_keyboard=True,
    one_time_keyboard=True
)
keyboard_modules.row("Описательный анализ", "Кластерный анализ",
                     "Сравнительный анализ")

keyboard_in_development = InlineKeyboardMarkup()
keyboard_in_development.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
