from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

"""
keyboard_describe_analysis: Выбор работы с описательным анализом.
keyboard_replace_null_values_cluster: Выбор опции замены пустых ячеек.
keyboard_choice: Выбор опции описательного анализа после загрузки файла.
"""

keybaord_cluster_analysis = InlineKeyboardMarkup()
keybaord_cluster_analysis.add(
    InlineKeyboardButton(text="Пример файла",
                         callback_data="example_cluster")
)
keybaord_cluster_analysis.add(
    InlineKeyboardButton(text="Загрузить свой файл",
                         callback_data="download_cluster")
)
keybaord_cluster_analysis.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
keyboard_replace_null_values_cluster = InlineKeyboardMarkup()

keyboard_replace_null_values_cluster.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек средним значением",
        callback_data="replace_null_with_mean",
    )
)

keyboard_replace_null_values_cluster.add(
    InlineKeyboardButton(
        text="Удаление строк с пропущенными значениями",
        callback_data="delete_null_rows_dropna",
    )
)

keyboard_replace_null_values_cluster.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек медианой",
        callback_data="replace_null_with_median"
    )
)
# keyboard_choice = InlineKeyboardMarkup()
# keyboard_choice.add(
#     InlineKeyboardButton(
#         text="Гистограммы данных",
#         callback_data="describe_build_graphs"
#     )
# )
# keyboard_choice.add(
#     InlineKeyboardButton(
#         text="Матрица корреляции",
#         callback_data="describe_correlation_analysis"
#     )
# )
# keyboard_choice.add(
#     InlineKeyboardButton(text="Описательная таблица",
#                          callback_data="describe_table")
# )
# keyboard_choice.add(
#     InlineKeyboardButton(
#         text="График Ящик с усами",
#         callback_data="describe_box_plot"
#     )
# )
# keyboard_choice.add(
#     InlineKeyboardButton(text="Главное меню", callback_data="back"))
