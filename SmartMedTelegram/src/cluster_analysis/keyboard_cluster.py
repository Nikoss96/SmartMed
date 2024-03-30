from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

"""
keyboard_describe_analysis: Выбор работы с описательным анализом.
keyboard_replace_null_values_cluster: Выбор опции замены пустых ячеек.
keyboard_choice_cluster: Выбор опции кластерного анализа после загрузки файла.
"""

keyboard_cluster_analysis = InlineKeyboardMarkup()
keyboard_cluster_analysis.add(
    InlineKeyboardButton(text="Пример файла",
                         callback_data="example_cluster")
)
keyboard_cluster_analysis.add(
    InlineKeyboardButton(text="Загрузить свой файл",
                         callback_data="download_cluster")
)
keyboard_cluster_analysis.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
keyboard_replace_null_values_cluster = InlineKeyboardMarkup()

keyboard_replace_null_values_cluster.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек средним значением",
        callback_data="replace_null_with_mean_cluster",
    )
)

keyboard_replace_null_values_cluster.add(
    InlineKeyboardButton(
        text="Удаление строк с пропущенными значениями",
        callback_data="delete_null_rows_dropna_cluster",
    )
)

keyboard_replace_null_values_cluster.add(
    InlineKeyboardButton(
        text="Замена пустых ячеек медианой",
        callback_data="replace_null_with_median_cluster"
    )
)
keyboard_choice_cluster = InlineKeyboardMarkup()
keyboard_choice_cluster.add(
    InlineKeyboardButton(
        text="Метод k-средних",
        callback_data="k_means_cluster"
    )
)
keyboard_choice_cluster.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)

keyboard_choice_number_of_clusters = InlineKeyboardMarkup()
keyboard_choice_number_of_clusters.add(
    InlineKeyboardButton(
        text="Оставить рекомендованное количество кластеров",
        callback_data="recommended_number_of_clusters"
    )
)
keyboard_choice_number_of_clusters.add(
    InlineKeyboardButton(
        text="Выбрать количество кластеров самостоятельно",
        callback_data="choose_number_of_clusters"
    )
)
keyboard_choice_number_of_clusters.add(
    InlineKeyboardButton(text="Главное меню", callback_data="back")
)
