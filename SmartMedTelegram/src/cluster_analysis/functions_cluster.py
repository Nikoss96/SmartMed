import os
from cluster_analysis.ClusterModule import ClusterModule
from cluster_analysis.keyboard_cluster import (
    keyboard_choice_cluster,
    keyboard_choice_number_of_clusters,
    keyboard_choice_number_of_clusters_hierarchical,
)
from cluster_analysis.keyboard_implementation_cluster import \
    generate_column_keyboard
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    CLUSTER_ANALYSIS,
    USER_DATA_PATH,
    ELBOW_METHOD,
    EXAMPLES,
    K_MEANS,
    HIERARCHICAL,
)
from functions import (
    send_document_from_file,
    create_dataframe_and_save_file, get_user_file_df,
)
from preprocessing.preprocessing import get_numeric_df

number_of_clusters = {}


def handle_example_cluster_analysis(bot, call):
    """
    Обработка запроса на пример файла для cluster analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id,
        text="Прислали пример файла. Вы можете использовать этот файл для проведения анализа",
    )
    send_document_from_file(
        bot,
        call.from_user.id,
        f"{MEDIA_PATH}/{DATA_PATH}/{EXAMPLES}/Кластерный_анализ_пример.xlsx",
    )


def handle_downloaded_cluster_file(bot, call, command):
    """
    Обработка файла, присланного пользователем для дальнейших расчетов.
    """
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите интересующий Вас метод кластеризации:",
        reply_markup=keyboard_choice_cluster,
    )


def handle_cluster_method(bot, call, command):
    """
    Обработка при выборе метода осле прочтения файла кластерного анализа.
    """
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = get_numeric_df(df)

    module = ClusterModule(df, call.from_user.id)

    optimal_clusters = module.elbow_method_and_optimal_clusters(max_clusters=10)

    if call.from_user.id in number_of_clusters:
        number_of_clusters.pop(call.from_user.id)

    number_of_clusters[call.from_user.id] = optimal_clusters
    chat_id = call.from_user.id

    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{chat_id}.png"

    keyboard = None

    if command == "hierarchical_cluster":
        keyboard = keyboard_choice_number_of_clusters_hierarchical

    elif command == "k_means_cluster":
        keyboard = keyboard_choice_number_of_clusters

    if os.path.isfile(file_path):
        file = open(file_path, "rb")

        bot.send_photo(chat_id=chat_id, photo=file)

        bot.send_message(
            chat_id=chat_id,
            text=f"На основе Ваших данных был построен график Метод локтя для определения "
                 f"оптимального количества кластеров по данным.\n\n"
                 f"Рекомендованное количество кластеров – {optimal_clusters}.\n\n"
                 "Вы можете выбрать рекомендованное количество кластеров, либо выбрать количество кластеров самостоятельно.",
            reply_markup=keyboard,
        )


def handle_choose_number_of_clusters(bot, call, command):
    """
    Отправляет сообщение для выбора количества кластеров.

    Parameters:
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    Returns:
        None
    """
    columns = [i + 1 for i in range(10)]

    keyboard = generate_column_keyboard(columns, 0, command)

    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите количество кластеров:",
        reply_markup=keyboard,
    )


def handle_cluster_numbers(bot, call, command):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = get_numeric_df(df)

    if command.startswith("cluster_"):
        n_clusters = int(call.data.replace("cluster_", ""))
    else:
        n_clusters = number_of_clusters.pop(call.from_user.id)

    module = ClusterModule(df, call.from_user.id)
    module.generate_k_means(n_clusters)

    chat_id = call.from_user.id

    png_file_path = (
        f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{chat_id}.png"
    )
    excel_file_path = (
        f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{chat_id}.xlsx"
    )

    if os.path.isfile(png_file_path) and os.path.isfile(excel_file_path):
        bot.send_message(
            chat_id=call.from_user.id,
            text="По заданному количеству кластеров с помощью Метода k-средних"
                 " был построен точечный график,"
                 " а также создана таблица распределения элементов по кластерам.",
        )

        file_cur = open(png_file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)

        file_cur = open(excel_file_path, "rb")
        bot.send_document(
            chat_id=chat_id,
            document=file_cur,
            visible_file_name="Принадлежность_элементов_к_кластерам.xlsx",
        )


def handle_hierarchical(bot, call):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = get_numeric_df(df)

    module = ClusterModule(df, call.from_user.id)
    module.plot_dendrogram()

    chat_id = call.from_user.id

    png_file_path = f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{HIERARCHICAL}/hierarchical_{chat_id}.png"

    columns = [i + 1 for i in range(10)]

    keyboard = generate_column_keyboard(columns, 0, "hierarchical")

    if os.path.isfile(png_file_path):
        file_cur = open(png_file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)

        bot.send_message(
            chat_id=call.from_user.id,
            text="По Вашим данным с помощью метода Иерархической кластеризации"
                 " была построена дендрограмма. Вы можете поменять количество кластеров:",
            reply_markup=keyboard,
        )


def handle_hierarchical_cluster_numbers(bot, call, command):
    n_clusters = int(command.replace("hierarchical_cluster_", ""))
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    df = get_numeric_df(df)

    module = ClusterModule(df, call.from_user.id)

    module.plot_dendrogram(n_clusters)

    chat_id = call.from_user.id

    png_file_path = f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{HIERARCHICAL}/hierarchical_{chat_id}.png"

    if os.path.isfile(png_file_path):
        file_cur = open(png_file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)
