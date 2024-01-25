from telebot.types import (
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
)

"""
keyboard_main_menu: Первый ответ бота после нажатия пользователем кнопки Start
keyboard_modules: Ответ бота после нажатия на "Модули" на стартовой клавиатуре
keyboard_dict: Ответ бота после нажатия на "Словарь" на стартовой клавиатуре
keyboard00: Ответ бота после нажатия на "bioequal" на вкладке "Модули" 
keyboard01: Ответ бота после нажатия на "describe" на вкладке "Модули" 
keyboard02: Ответ бота после нажатия на "predict" на вкладке "Модули" 
"""
keyboard_main_menu = ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_main_menu.row("Модули", "Словарь", "Chat-GPT")

keyboard_modules = ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_modules.row("bioequal", "cluster", "describe", "predict")

keyboard_dict = InlineKeyboardMarkup()
keyboard_dict.add(
    InlineKeyboardButton(
        text="T-критерий Стьюдента для независимых переменных", callback_data="t-crit"
    )
)
keyboard_dict.add(
    InlineKeyboardButton(
        text="Коэффициент корреляции Спирмена", callback_data="spearman-corr"
    )
)
keyboard_dict.add(
    InlineKeyboardButton(text="Кривая выживаемости", callback_data="curve")
)
keyboard_dict.add(
    InlineKeyboardButton(text="Диаграмма Ящик с усами", callback_data="box")
)
keyboard_dict.add(InlineKeyboardButton(text="Назад", callback_data="back"))

keyboard00 = InlineKeyboardMarkup()
keyboard00.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_bioequal")
)
keyboard00.add(
    InlineKeyboardButton(text="Загрузить свой файл", callback_data="download_bioequal")
)
keyboard00.add(InlineKeyboardButton(text="Назад", callback_data="back"))

keyboard01 = InlineKeyboardMarkup()
keyboard01.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_describe")
)
keyboard01.add(
    InlineKeyboardButton(text="Загрузить свой файл", callback_data="download_describe")
)

keyboard02 = InlineKeyboardMarkup()
keyboard02.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_predict")
)
keyboard02.add(
    InlineKeyboardButton(text="Загрузить свой файл", callback_data="download_predict")
)
