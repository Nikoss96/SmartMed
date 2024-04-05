from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)

"""
keyboard_start: Клавиатура, содержащая разделы чат-бота.
keyboard_modules: Клавиатура, содержащая модули чат-бота.
keyboard_main_menu: Клавиатура возвращения в главное меню.
"""

keyboard_start = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
keyboard_start.row("Модули", "Словарь", "Искусственный интеллект")

keyboard_modules = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
keyboard_modules.row("Описательный анализ", "Кластерный анализ", "Сравнительный анализ")

keyboard_main_menu = InlineKeyboardMarkup()
keyboard_main_menu.add(InlineKeyboardButton(text="Главное меню", callback_data="back"))
