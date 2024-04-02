from comparative_analysis.keyboard_comparative import \
    keyboard_choice_comparative
from data.paths import MEDIA_PATH, DATA_PATH, EXAMPLES
from functions import send_document_from_file, create_dataframe_and_save_file
from keyboard import keyboard_in_development


def handle_example_comparative_analysis(bot, call):
    """
    Обработка запроса на пример файла для comparative analysis.

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
        f"{MEDIA_PATH}/{DATA_PATH}/{EXAMPLES}/Сравнительный_анализ_пример.xlsx",
    )


def handle_downloaded_comparative_file(bot, call, command):
    """
    Обработка файла, присланного пользователем для дальнейших расчетов.
    """
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите интересующий Вас метод сравнительного анализа:",
        reply_markup=keyboard_choice_comparative,
    )


def handle_kolmogorov_smirnov_test_comparative(bot, call, command):
    """
    Обработка при выборе метода после прочтения файла сравнительного анализа.
    """
    bot.send_message(
        chat_id=call.from_user.id,
        text=f"Критерий согласия Колмогорова-Смирнова предназначен для "
             f"проверки гипотезы о принадлежности выборки нормальному "
             f"закону распределения.\n\nВам необходимо указать независимую и "
             f"группирующую переменные.",
    )

    bot.send_message(
        chat_id=call.from_user.id,
        text=f"Группирующая переменная - переменная, используемая для разбиения "
             f"независимой переменной на группы, для данного критерия является "
             f"бинарной переменной. Например, пол, группа и т.д.\n\nНезависимая"
             f" переменная представляет набор количественных, непрерывных "
             f"значений. Например, возраст пациента, уровень лейкоцитов и т.д.",
    )

    # Критерий согласия Колмогорова-Смирнова предназначен для проверки гипотезы о принадлежности выборки нормальному закону распределения.
    # df = get_user_file_df(
    #     f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
    #     call.from_user.id,
    # )
    #
    # df = filter_columns_with_more_than_2_unique_values(df)
    #
    # module = ClusterModule(df, call.from_user.id)
    #
    # optimal_clusters = module.elbow_method_and_optimal_clusters(max_clusters=10)
    #
    # if call.from_user.id in number_of_clusters:
    #     number_of_clusters.pop(call.from_user.id)
    #
    # number_of_clusters[call.from_user.id] = optimal_clusters
    # chat_id = call.from_user.id
    #
    # file_path = f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{chat_id}.png"
    #
    # keyboard = None
    #
    # if command == "hierarchical_cluster":
    #     keyboard = keyboard_choice_number_of_clusters_hierarchical
    #
    # elif command == "k_means_cluster":
    #     keyboard = keyboard_choice_number_of_clusters
    #
    # if os.path.isfile(file_path):
    #     file = open(file_path, "rb")
    #
    #     bot.send_photo(chat_id=chat_id, photo=file)
    #
    #     bot.send_message(
    #         chat_id=chat_id,
    #         text=f"На основе Ваших данных был построен график Метод локтя для определения "
    #              f"оптимального количества кластеров по данным.\n\n"
    #              f"Рекомендованное количество кластеров – {optimal_clusters}.\n\n"
    #              "Вы можете оставить рекомендованное количество кластеров, либо выбрать количество кластеров самостоятельно.",
    #         reply_markup=keyboard,
    #     )