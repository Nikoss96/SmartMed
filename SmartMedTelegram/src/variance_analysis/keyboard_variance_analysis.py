from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

keyboard_variance_analysis = InlineKeyboardMarkup()
keyboard_variance_analysis.add(
    InlineKeyboardButton(text="Пример файла", callback_data="example_variance")
)
keyboard_variance_analysis.add(
    InlineKeyboardButton(text="Загрузить свой файл",
                         callback_data="download_variance")
)
keyboard_variance_analysis.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
keyboard_replace_null_values_variance = InlineKeyboardMarkup()

keyboard_replace_null_values_variance.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек средним значением",
        callback_data="replace_null_with_mean_variance",
    )
)

keyboard_replace_null_values_variance.add(
    InlineKeyboardButton(
        text="Удаление строк с пропущенными значениями",
        callback_data="delete_null_rows_dropna_variance",
    )
)

keyboard_replace_null_values_variance.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек медианой",
        callback_data="replace_null_with_median_variance",
    )
)
keyboard_choice_variance = InlineKeyboardMarkup()
keyboard_choice_variance.add(
    InlineKeyboardButton(text="Критерий Краскела-Уоллиса",
                         callback_data="test_kruskal_wallis")
)
keyboard_choice_variance.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
