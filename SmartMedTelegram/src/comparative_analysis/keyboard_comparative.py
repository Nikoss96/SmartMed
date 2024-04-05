from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

"""
keyboard_comparative_analysis: Выбор работы со сравнительным анализом.
keyboard_replace_null_values_comparative: Выбор опции замены пустых ячеек.
keyboard_choice_comparative: Выбор опции сравнительного анализа после загрузки 
файла.
"""

keyboard_comparative_analysis = InlineKeyboardMarkup()
keyboard_comparative_analysis.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_comparative")
)
keyboard_comparative_analysis.add(
    InlineKeyboardButton(
        text="Загрузить свой файл", callback_data="download_comparative"
    )
)
keyboard_comparative_analysis.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)

keyboard_replace_null_values_comparative = InlineKeyboardMarkup()

keyboard_replace_null_values_comparative.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек средним значением",
        callback_data="replace_null_with_mean_comparative",
    )
)

keyboard_replace_null_values_comparative.add(
    InlineKeyboardButton(
        text="Удаление строк с пропущенными значениями",
        callback_data="delete_null_rows_dropna_comparative",
    )
)

keyboard_replace_null_values_comparative.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек медианой",
        callback_data="replace_null_with_median_comparative",
    )
)

keyboard_choice_comparative = InlineKeyboardMarkup()
keyboard_choice_comparative.add(
    InlineKeyboardButton(
        text="Критерий Колмогорова-Смирнова",
        callback_data="kolmogorov_smirnov_test_comparative",
    )
)
keyboard_choice_comparative.add(
    InlineKeyboardButton(
        text="Т-критерий Стьюдента для независимых переменных",
        callback_data="t_criterion_student_independent_comparative",
    )
)
keyboard_choice_comparative.add(
    InlineKeyboardButton(
        text="T-критерий Стьюдента для зависимых переменных",
        callback_data="t_criterion_student_dependent_comparative",
    )
)
keyboard_choice_comparative.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
