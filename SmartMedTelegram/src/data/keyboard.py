from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)

"""
keyboard_main_menu: Первый ответ бота после нажатия пользователем кнопки Start
keyboard_modules: Ответ бота после нажатия на "Модули" на стартовой клавиатуре
keyboard00: Ответ бота после нажатия на "bioequal" на вкладке "Модули"
keyboard01: Ответ бота после нажатия на "Описательный анализ" на вкладке "Модули"
keyboard02: Ответ бота после нажатия на "predict" на вкладке "Модули"
"""

keyboard_main_menu = ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_main_menu.row("Модули", "Словарь", "Chat-GPT")

keyboard_modules = ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_modules.row("Описательный анализ")

keyboard00 = InlineKeyboardMarkup()
keyboard00.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_bioequal")
)
keyboard00.add(
    InlineKeyboardButton(text="Загрузить свой файл",
                         callback_data="download_bioequal")
)
keyboard00.add(InlineKeyboardButton(text="Главное меню", callback_data="back"))

keyboard01 = InlineKeyboardMarkup()
keyboard01.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_describe")
)
keyboard01.add(
    InlineKeyboardButton(text="Загрузить свой файл",
                         callback_data="download_describe")
)
keyboard01.add(InlineKeyboardButton(text="Главное меню", callback_data="back"))

keyboard02 = InlineKeyboardMarkup()
keyboard02.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_predict")
)
keyboard02.add(
    InlineKeyboardButton(text="Загрузить свой файл",
                         callback_data="download_predict")
)

keyboard_in_development = InlineKeyboardMarkup()
keyboard_in_development.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)

keyboard_choose_describe = InlineKeyboardMarkup()
keyboard_choose_describe.add(
    InlineKeyboardButton(
        text="Построение графиков", callback_data="describe_build_graphs"
    )
)
keyboard_choose_describe.add(
    InlineKeyboardButton(
        text="Корреляционный анализ",
        callback_data="describe_correlation_analysis"
    )
)
keyboard_choose_describe.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
