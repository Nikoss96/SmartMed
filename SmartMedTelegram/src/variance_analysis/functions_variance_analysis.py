import os

from comparative_analysis.functions_comparative import user_columns
from data.paths import MEDIA_PATH, DATA_PATH, EXAMPLES, USER_DATA_PATH, \
    VARIANCE_ANALYSIS, KRUSKAL_TEST
from functions import send_document_from_file, create_dataframe_and_save_file, \
    get_user_file_df
from variance_analysis.VarianceModule import VarianceModule
from variance_analysis.keyboard_implementation_variance import \
    generate_column_keyboard
from variance_analysis.keyboard_variance_analysis import \
    keyboard_choice_variance, keyboard_variance_analysis


def handle_example_variance(bot, call):
    """
    Обработка запроса на пример файла для variance analysis.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    bot.answer_callback_query(
        callback_query_id=call.id,
        text="Прислали пример файла. "
             "Вы можете использовать этот файл для проведения анализа",
    )
    send_document_from_file(
        bot,
        call.from_user.id,
        f"{MEDIA_PATH}/{DATA_PATH}/{EXAMPLES}/Дисперсионный_анализ_пример.xlsx",
    )


def handle_downloaded_variance_file(bot, call, command):
    """
    Обработка файла, присланного пользователем для дальнейших расчетов.
    """
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите интересующий Вас метод дисперсионного анализа:",
        reply_markup=keyboard_choice_variance,
    )


def handle_variance_module(bot, call, command):
    """
    Обработка при выборе метода после прочтения файла дисперсионного анализа.
    """

    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = VarianceModule(df, call.from_user.id)

    columns = module.get_all_columns()

    if len(columns) < 1:
        bot.send_message(
            chat_id=call.from_user.id,
            text="В Вашем файле отсутствуют переменные. "
                 "Загрузите файл, который содержит переменные.",
            reply_markup=keyboard_variance_analysis,
        )

    else:
        user_columns[call.from_user.id] = {}
        user_columns[call.from_user.id]["columns"] = columns
        user_columns[call.from_user.id]["command"] = command

        if command == "test_kruskal_wallis":
            if len(columns) < 3:
                bot.send_message(
                    chat_id=call.from_user.id,
                    text="В Вашем файле отсутствует 3 группы переменных. "
                         "Загрузите файл, который содержит 3 группы переменных.",
                    reply_markup=keyboard_variance_analysis,
                )
            else:
                bot.send_message(
                    chat_id=call.from_user.id,
                    text=f"Для применения критерия Краскела-Уоллиса необходимо, "
                         f"чтобы количество выборок в данных было не менее трех."
                         f"\n\nЭтот метод математической статистики, при котором "
                         f"сравнивают средние значения в трех и более выборках. "
                         f" Инструмент помогает узнать и оценить различия между "
                         f"ними."
                         f"\n\nЕсли в ваших данных только две выборки, "
                         f"воспользуйтесь "
                         f"критерием Манна-Уитни из сравнительного анализа."
                         f"\n\nВам необходимо указать три переменные, "
                         f"представляющие независимые выборки данных.",
                )

                keyboard = generate_column_keyboard(columns, 0, command)

                bot.send_message(
                    chat_id=call.from_user.id,
                    text="Выберите первую переменную:",
                    reply_markup=keyboard,
                )


def handle_test_kruskal_wallis_variance(bot, call, command):
    if not "test_kruskal_wallis" in user_columns[call.from_user.id]:
        user_columns[call.from_user.id]["test_kruskal_wallis"] = [
            int(command.replace("test_kruskal_wallis_", ""))
        ]

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите вторую переменную:",
        )


    elif len(user_columns[call.from_user.id][
                 "test_kruskal_wallis"]) == 1:
        if (
                int(command.replace("test_kruskal_wallis_", ""))
                not in user_columns[call.from_user.id][
            "test_kruskal_wallis"]
        ):
            user_columns[call.from_user.id][
                "test_kruskal_wallis"].append(
                int(command.replace("test_kruskal_wallis_", ""))
            )

            bot.send_message(
                chat_id=call.from_user.id,
                text="Выберите третью переменную:",
            )

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую переменную",
            )

    elif len(user_columns[call.from_user.id][
                 "test_kruskal_wallis"]) == 2:
        if (
                int(command.replace("test_kruskal_wallis_", ""))
                not in user_columns[call.from_user.id][
            "test_kruskal_wallis"]
        ):
            user_columns[call.from_user.id][
                "test_kruskal_wallis"].append(
                int(command.replace("test_kruskal_wallis_", ""))
            )
            build_test_kruskal_wallis_variance(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую переменную",
            )

    else:
        user_columns[call.from_user.id]["test_kruskal_wallis"].pop(0)

        if (
                int(command.replace("test_kruskal_wallis_", ""))
                not in user_columns[call.from_user.id][
            "test_kruskal_wallis"]
        ):
            user_columns[call.from_user.id][
                "test_kruskal_wallis"].append(
                int(command.replace("test_kruskal_wallis_", ""))
            )
            build_test_kruskal_wallis_variance(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую переменную",
            )


def build_test_kruskal_wallis_variance(bot, call, command):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = VarianceModule(df, call.from_user.id)
    columns = user_columns[call.from_user.id]["test_kruskal_wallis"]

    if not columns:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_variance_analysis,
        )

    command = user_columns[call.from_user.id]["command"]

    if not command:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_variance_analysis,
        )

    else:
        if command == "test_kruskal_wallis":
            names_of_columns = user_columns[call.from_user.id]["columns"]

            module.generate_test_kruskal_wallis(
                names_of_columns[columns[0]], names_of_columns[columns[1]],
                names_of_columns[columns[2]]
            )
            table_file = f"{MEDIA_PATH}/{DATA_PATH}/{VARIANCE_ANALYSIS}/{KRUSKAL_TEST}/test_kruskal_wallis_{call.from_user.id}.xlsx"

            if os.path.isfile(table_file):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text=f"На основе Ваших данных была построена таблица "
                         f"критерия Краскела-Уоллиса для переменных: '{names_of_columns[columns[0]]}', "
                         f"'{names_of_columns[columns[1]]}' и '{names_of_columns[columns[2]]}'. "
                         f"\n\nЕсли p <= 0.05, нулевая гипотеза отвергается. "
                         f"Нулевая гипотеза в данном случае утверждает, что "
                         f"распределения во всех группах одинаковы. Поскольку"
                         f" нулевая гипотеза отвергается, можно сделать вывод о"
                         f" наличии статистически значимых различий между "
                         f"группами. \n\nВ случае, если p-value больше 0.05,"
                         f" нулевая гипотеза не отвергается. "
                         f"Это означает, что нет статистически значимых"
                         f" различий между группами. В таком случае различия "
                         f"между группами могут быть случайными и не "
                         f"свидетельствовать о наличии значимых различий "
                         f"в их распределениях."
                )

                file_cur = open(table_file, "rb")
                bot.send_document(
                    chat_id=call.from_user.id,
                    document=file_cur,
                    visible_file_name=f"Критерий_Краскела_Уоллиса_{names_of_columns[columns[0]]}_{names_of_columns[columns[1]]}_{names_of_columns[columns[2]]}.xlsx",
                )
