import os

from comparative_analysis.ComparativeModule import ComparativeModule
from comparative_analysis.keyboard_comparative import (
    keyboard_choice_comparative,
    keyboard_comparative_analysis,
)
from comparative_analysis.keyboard_implementation_comparative import (
    handle_choose_column_comparative,
    generate_categorical_value_column_keyboard,
    generate_column_keyboard,
)
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    EXAMPLES,
    USER_DATA_PATH,
    COMPARATIVE_ANALYSIS,
    KOLMOGOROVA_SMIRNOVA,
    T_CRITERIA_INDEPENDENT,
    T_CRITERIA_DEPENDENT, MANN_WHITNEY_TEST, WILCOXON_TEST,
)
from functions import send_document_from_file, create_dataframe_and_save_file, \
    get_user_file_df

user_columns = {}


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
    if call.from_user.id in user_columns:
        user_columns.pop(call.from_user.id)
    create_dataframe_and_save_file(call.from_user.id, command)
    bot.send_message(
        chat_id=call.from_user.id,
        text="Выберите интересующий Вас метод сравнительного анализа:",
        reply_markup=keyboard_choice_comparative,
    )


def handle_comparative_module(bot, call, command):
    """
    Обработка при выборе метода после прочтения файла сравнительного анализа.
    """

    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)

    (
        categorical_columns,
        continuous_columns,
    ) = module.get_categorical_and_continuous_columns()

    if len(categorical_columns) < 1:
        bot.send_message(
            chat_id=call.from_user.id,
            text="В Вашем файле отсутствуют категориальные переменные. "
                 "Загрузите файл, который содержит категориальные переменные.",
            reply_markup=keyboard_comparative_analysis,
        )
    elif len(continuous_columns) < 1:
        bot.send_message(
            chat_id=call.from_user.id,
            text="В Вашем файле отсутствуют независимые переменные. "
                 "Загрузите файл, который содержит независимые переменные.",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        user_columns[call.from_user.id] = {}
        user_columns[call.from_user.id][
            "categorical_columns"] = categorical_columns
        user_columns[call.from_user.id][
            "continuous_columns"] = continuous_columns
        user_columns[call.from_user.id]["command"] = command

        if command.startswith("kolmogorov"):
            bot.send_message(
                chat_id=call.from_user.id,
                text=f"Критерий согласия Колмогорова-Смирнова предназначен для "
                     f"проверки гипотезы о принадлежности выборки нормальному "
                     f"закону распределения.\n\nВам необходимо указать независимую и "
                     f"группирующую переменные.\n\n"
                     f"Группирующая переменная - переменная, используемая для разбиения "
                     f"независимой переменной на группы, для данного критерия является "
                     f"бинарной переменной. Например, пол, группа и т.д.\n\nНезависимая"
                     f" переменная представляет набор количественных, непрерывных "
                     f"значений. Например, возраст пациента, уровень лейкоцитов и т.д.",
            )
        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text=f"Для применения t-критерия Стьюдента необходимо, чтобы "
                     f"исходные данные имели нормальное распределение.\n\n"
                     f"Данный статистический метод служит для сравнения двух "
                     f"независимых между собой групп. Примеры сравниваемых "
                     f"величин: возраст в основной и контрольной группе, "
                     f"содержание глюкозы в крови пациентов, принимавших "
                     f"препарат или плацебо.\n\nВам необходимо указать независимую и "
                     f"группирующую переменные.\n\n"
                     f"Группирующая переменная - переменная, используемая для разбиения "
                     f"независимой переменной на группы, для данного критерия является "
                     f"бинарной переменной. Например, пол, группа и т.д.\n\nНезависимая"
                     f" переменная представляет набор количественных, непрерывных "
                     f"значений. Например, возраст пациента, уровень лейкоцитов и т.д.",
            )
        handle_continuous_columns_comparative(bot, call)


def handle_continuous_columns_comparative(bot, call):
    continuous_columns = user_columns[call.from_user.id]["continuous_columns"]

    handle_choose_column_comparative(
        bot, call, continuous_columns, "continuous_columns"
    )


def handle_categorical_columns_comparative(bot, call, command):
    categorical_columns = user_columns[call.from_user.id]["categorical_columns"]

    user_columns[call.from_user.id]["continuous_column"] = int(
        command.replace("continuous_column_", "")
    )

    if "categorical_column" in user_columns[call.from_user.id]:
        handle_create_table_for_module_comparative(bot, call)

    else:
        handle_choose_column_comparative(
            bot, call, categorical_columns, "categorical_columns"
        )


def handle_categorical_column_comparative(bot, call, command):
    user_columns[call.from_user.id]["categorical_column"] = int(
        command.replace("categorical_column_", "")
    )
    handle_create_table_for_module_comparative(bot, call)


def handle_create_table_for_module_comparative(bot, call):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)
    categorical_column_index = user_columns[call.from_user.id][
        "categorical_column"]

    categorical_column = user_columns[call.from_user.id]["categorical_columns"][
        categorical_column_index
    ]

    continuous_column_index = user_columns[call.from_user.id][
        "continuous_column"]

    continuous_column = user_columns[call.from_user.id]["continuous_columns"][
        continuous_column_index
    ]

    if not categorical_column or not continuous_column:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    command = user_columns[call.from_user.id]["command"]

    if not command:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        if command == "kolmogorov_smirnov_test_comparative":
            module.generate_test_kolmagorova_smirnova(
                categorical_column, continuous_column
            )

            table_file = f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{KOLMOGOROVA_SMIRNOVA}/kolmogorova_smirnova_{call.from_user.id}.xlsx"

            if os.path.isfile(table_file):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text=f"На основе Ваших данных была построена таблица "
                         f"распределения переменной '{continuous_column}' "
                         f"по группирующей переменной '{categorical_column}'. ",
                )

                file_cur = open(table_file, "rb")
                bot.send_document(
                    chat_id=call.from_user.id,
                    document=file_cur,
                    visible_file_name=f"Колмогорова_Смирнова_{continuous_column}_{categorical_column}.xlsx",
                )

        else:
            class_names = module.get_class_names(categorical_column, module.df)

            if len(class_names) < 2:
                bot.send_message(
                    chat_id=call.from_user.id,
                    text="Выбранная группирующая переменная должна иметь "
                         "хотя бы два уникальных значения."
                         " Загрузите файл, который содержит хотя бы два"
                         " уникальных значения в группирующей переменной.",
                    reply_markup=keyboard_comparative_analysis,
                )
            elif len(class_names) == 2:
                module.generate_t_criterion_student_independent(
                    categorical_column, continuous_column, class_names
                )

                table_file = f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{T_CRITERIA_INDEPENDENT}/t_criteria_independent_{call.from_user.id}.xlsx"

                if os.path.isfile(table_file):
                    values_as_strings = [str(value) for value in
                                         class_names.values()]

                    bot.send_message(
                        chat_id=call.from_user.id,
                        text=f"На основе Ваших данных была построена таблица "
                             f"распределения переменной '{continuous_column}' "
                             f"по группирующей переменной '{categorical_column}'."
                             f" Группы, выбранные в группирующей переменной: {', '.join(values_as_strings)}."
                             f"\n\nЕсли p < 0.05, нулевая гипотеза отвергается,"
                             f" принимается альтернативная, различия обладают "
                             f"статистической значимостью и носят системный "
                             f"характер.\n\nЕсли p ≥ 0.05, не отвергается нулевая "
                             f"гипотеза, различия не являются статистически "
                             f"значимыми и носят случайный характер.",
                    )

                    file_cur = open(table_file, "rb")
                    bot.send_document(
                        chat_id=call.from_user.id,
                        document=file_cur,
                        visible_file_name=f"T_критерий_Стьюдента_независимых_{continuous_column}_{categorical_column}_{'_'.join(values_as_strings)}.xlsx",
                    )
            else:
                keyboard = generate_categorical_value_column_keyboard(
                    class_names)
                bot.send_message(
                    chat_id=call.from_user.id,
                    text="Выберите два значения группирующей переменной, по "
                         "которым рассчитать T-критерий Стьюдента",
                    reply_markup=keyboard,
                )
                user_columns[call.from_user.id]["class_names"] = class_names


def build_t_criteria_independent(bot, call):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)
    categorical_column_index = user_columns[call.from_user.id][
        "categorical_column"]

    categorical_column = user_columns[call.from_user.id]["categorical_columns"][
        categorical_column_index
    ]

    continuous_column_index = user_columns[call.from_user.id][
        "continuous_column"]

    continuous_column = user_columns[call.from_user.id]["continuous_columns"][
        continuous_column_index
    ]

    categorical_values = user_columns[call.from_user.id][
        "categorical_column_values"]

    class_names = user_columns[call.from_user.id]["class_names"]

    merged_dict = {
        key: class_names[key] for key in categorical_values if
        key in class_names
    }

    module.generate_t_criterion_student_independent(
        categorical_column, continuous_column, merged_dict
    )

    values_as_strings = [str(value) for value in merged_dict.values()]

    table_file = f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{T_CRITERIA_INDEPENDENT}/t_criteria_independent_{call.from_user.id}.xlsx"

    if os.path.isfile(table_file):
        bot.send_message(
            chat_id=call.from_user.id,
            text=f"На основе Ваших данных была построена таблица "
                 f"распределения переменной '{continuous_column}' "
                 f"по группирующей переменной '{categorical_column}'. "
                 f" Группы, выбранные в группирующей переменной: {', '.join(values_as_strings)}."
                 f"\n\nЕсли p < 0.05, нулевая гипотеза отвергается, "
                 f"принимается альтернативная, различия обладают "
                 f"статистической значимостью и носят системный характер."
                 f"\n\nЕсли p ≥ 0.05, не отвергается нулевая гипотеза, различия "
                 f"не являются статистически значимыми и носят случайный "
                 f"характер.",
        )

        file_cur = open(table_file, "rb")
        bot.send_document(
            chat_id=call.from_user.id,
            document=file_cur,
            visible_file_name=f"T_критерий_Стьюдента_независимых_{continuous_column}_{categorical_column}_{'_'.join(values_as_strings)}.xlsx",
        )


def handle_t_criteria_categorical_value(bot, call, command):
    current_value = command.replace("t_criteria_categorical_value_", "")
    if current_value.isdigit():
        current_value = int(current_value)

    if "categorical_column_values" in user_columns[call.from_user.id]:
        current_length = len(
            user_columns[call.from_user.id]["categorical_column_values"]
        )

        if current_length == 1:
            if (
                    current_value
                    in user_columns[call.from_user.id][
                "categorical_column_values"]
            ):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
                )

            else:
                user_columns[call.from_user.id][
                    "categorical_column_values"].append(
                    current_value
                )
                build_t_criteria_independent(bot, call)

        elif current_length == 2:
            user_columns[call.from_user.id]["categorical_column_values"].pop(0)
            if (
                    current_value
                    in user_columns[call.from_user.id][
                "categorical_column_values"]
            ):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
                )
            else:
                user_columns[call.from_user.id][
                    "categorical_column_values"].append(
                    current_value
                )
                build_t_criteria_independent(bot, call)
    else:
        user_columns[call.from_user.id]["categorical_column_values"] = [
            current_value]
        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите вторую переменную на клавиатуре:"
        )


def handle_t_criterion_student_dependent(bot, call, command):
    """
    Обработка при выборе метода после прочтения файла сравнительного анализа.
    """

    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)

    columns = module.get_all_columns()

    if len(columns) < 1:
        bot.send_message(
            chat_id=call.from_user.id,
            text="В Вашем файле отсутствуют переменные. "
                 "Загрузите файл, который содержит переменные.",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        user_columns[call.from_user.id] = {}
        user_columns[call.from_user.id]["columns"] = columns
        user_columns[call.from_user.id]["command"] = command

        if command == "t_criterion_student_dependent_comparative":
            bot.send_message(
                chat_id=call.from_user.id,
                text=f"Для применения t-критерия Стьюдента необходимо, чтобы"
                     f" исходные данные имели нормальное распределение."
                     f" \n\nДанный метод используется для сравнения двух "
                     f"зависимых групп пациентов. Примеры сравниваемых "
                     f"величин: частота сердечных сокращений до и после "
                     f"приема.\n\nВам необходимо указать две переменные.",
            )

        keyboard = generate_column_keyboard(columns, 0, command)

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите первую переменную:",
            reply_markup=keyboard,
        )


def handle_t_criteria_for_dependent(bot, call, command):
    if not "dependent_column" in user_columns[call.from_user.id]:
        user_columns[call.from_user.id]["dependent_column"] = [
            int(command.replace("dependent_column_", ""))
        ]

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите вторую переменную:",
        )

    elif len(user_columns[call.from_user.id]["dependent_column"]) == 1:
        if (
                int(command.replace("dependent_column_", ""))
                not in user_columns[call.from_user.id]["dependent_column"]
        ):
            user_columns[call.from_user.id]["dependent_column"].append(
                int(command.replace("dependent_column_", ""))
            )
            build_t_criteria_table_dependent(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
            )

    else:
        user_columns[call.from_user.id]["dependent_column"].pop(0)

        if (
                int(command.replace("dependent_column_", ""))
                not in user_columns[call.from_user.id]["dependent_column"]
        ):
            user_columns[call.from_user.id]["dependent_column"].append(
                int(command.replace("dependent_column_", ""))
            )
            build_t_criteria_table_dependent(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
            )


def build_t_criteria_table_dependent(bot, call, command):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)
    columns = user_columns[call.from_user.id]["dependent_column"]

    if not columns:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    command = user_columns[call.from_user.id]["command"]

    if not command:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        if command == "t_criterion_student_dependent_comparative":
            names_of_columns = user_columns[call.from_user.id]["columns"]

            module.generate_t_criteria_student_dependent(
                names_of_columns[columns[0]], names_of_columns[columns[1]]
            )

            table_file = f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{T_CRITERIA_DEPENDENT}/t_criteria_dependent_{call.from_user.id}.xlsx"

            if os.path.isfile(table_file):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text=f"На основе Ваших данных была построена таблица "
                         f"T-критерия Стьюдента для переменной '{names_of_columns[columns[0]]}' "
                         f"и переменной '{names_of_columns[columns[1]]}'. "
                         f"\n\nЕсли p < 0.05, нулевая гипотеза отвергается, "
                         f"принимается альтернативная, различия обладают "
                         f"статистической значимостью и носят системный "
                         f"характер.\n\nЕсли p ≥ 0.05, принимается нулевая "
                         f"гипотеза, различия не являются статистически "
                         f"значимыми и носят случайный характер.",
                )

                file_cur = open(table_file, "rb")
                bot.send_document(
                    chat_id=call.from_user.id,
                    document=file_cur,
                    visible_file_name=f"T_критерий_Стьюдента_зависимых_{names_of_columns[columns[0]]}_{names_of_columns[columns[1]]}.xlsx",
                )


def handle_nonparametric_tests_comparative(bot, call, command):
    """
    Обработка при выборе метода после прочтения файла сравнительного анализа.
    """

    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)

    columns = module.get_all_columns()

    if len(columns) < 1:
        bot.send_message(
            chat_id=call.from_user.id,
            text="В Вашем файле отсутствуют переменные. "
                 "Загрузите файл, который содержит переменные.",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        user_columns[call.from_user.id] = {}
        user_columns[call.from_user.id]["columns"] = columns
        user_columns[call.from_user.id]["command"] = command

        if command == "mann_whitney_test_comparative":
            bot.send_message(
                chat_id=call.from_user.id,
                text=f"Для применения критерия Манна-Уитни необходимо, "
                     f"чтобы исходные данные соответствовали определённым "
                     f"требованиям.\n\nЭтот метод используется для сравнения"
                     f" двух независимых групп, когда данные не обязательно "
                     f"должны иметь нормальное распределение. Это делает"
                     f" критерий Манна-Уитни особенно полезным для анализа"
                     f" данных, которые не соответствуют предположению о "
                     f"нормальности.\n\nПримеры сравниваемых величин: уровни "
                     f"стресса у двух различных групп людей.\n\nВам необходимо"
                     f" указать две переменные, представляющие независимые"
                     f" выборки данных.",
            )

        elif command == "wilcoxon_test_comparative":
            bot.send_message(
                chat_id=call.from_user.id,
                text=f"Для применения критерия Уилкоксона необходимо, "
                     f"чтобы исходные данные соответствовали определённым "
                     f"требованиям.\n\nОн используется для проверки наличия"
                     f" значительной разницы между двумя средними значениями "
                     f"генеральной совокупности, когда распределение различий "
                     f"между двумя выборками нельзя считать нормальным."
                     f"\n\nПримеры сравниваемых величин: уровни "
                     f"стресса у двух различных групп людей.\n\nВам необходимо"
                     f" указать две переменные, представляющие независимые"
                     f" выборки данных.",
            )

        keyboard = generate_column_keyboard(columns, 0, command)

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите первую переменную:",
            reply_markup=keyboard,
        )


def handle_mann_whitney_test_comparative(bot, call, command):
    if not "mann_whitney_test_comparative" in user_columns[call.from_user.id]:
        user_columns[call.from_user.id]["mann_whitney_test_comparative"] = [
            int(command.replace("mann_whitney_test_comparative_", ""))
        ]

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите вторую переменную:",
        )


    elif len(user_columns[call.from_user.id][
                 "mann_whitney_test_comparative"]) == 1:
        if (
                int(command.replace("mann_whitney_test_comparative_", ""))
                not in user_columns[call.from_user.id][
            "mann_whitney_test_comparative"]
        ):
            user_columns[call.from_user.id][
                "mann_whitney_test_comparative"].append(
                int(command.replace("mann_whitney_test_comparative_", ""))
            )
            build_mann_whitney_test_comparative(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
            )

    else:
        user_columns[call.from_user.id]["mann_whitney_test_comparative"].pop(0)

        if (
                int(command.replace("mann_whitney_test_comparative_", ""))
                not in user_columns[call.from_user.id][
            "mann_whitney_test_comparative"]
        ):
            user_columns[call.from_user.id][
                "mann_whitney_test_comparative"].append(
                int(command.replace("mann_whitney_test_comparative_", ""))
            )
            build_mann_whitney_test_comparative(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
            )


def build_mann_whitney_test_comparative(bot, call, command):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)
    columns = user_columns[call.from_user.id]["mann_whitney_test_comparative"]

    if not columns:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    command = user_columns[call.from_user.id]["command"]

    if not command:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        if command == "mann_whitney_test_comparative":
            names_of_columns = user_columns[call.from_user.id]["columns"]

            module.generate_mann_whitney_test_comparative(
                names_of_columns[columns[0]], names_of_columns[columns[1]]
            )
            table_file = f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{MANN_WHITNEY_TEST}/mann_whitney_test_comparative_{call.from_user.id}.xlsx"

            if os.path.isfile(table_file):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text=f"На основе Ваших данных была построена таблица "
                         f"критерия Манна-Уитни для переменной '{names_of_columns[columns[0]]}' "
                         f"и переменной '{names_of_columns[columns[1]]}'. "
                         f"\n\nЕсли p <= 0.05, нулевая гипотеза отвергается. "
                         f"Есть статистически значимые различия между двумя "
                         f"независимыми выборками. Это означает, что "
                         f"распределения двух групп различаются.\n\nЕсли "
                         f"p > 0.05, то нет статистически значимых различий "
                         f"между двумя независимыми выборками. Это означает, "
                         f"что различия между группами могут быть случайными "
                         f"и не свидетельствуют о значимых различиях в их "
                         f"распределениях.",
                )

                file_cur = open(table_file, "rb")
                bot.send_document(
                    chat_id=call.from_user.id,
                    document=file_cur,
                    visible_file_name=f"Критерий_Манна_Уитни_{names_of_columns[columns[0]]}_{names_of_columns[columns[1]]}.xlsx",
                )


def handle_wilcoxon_test_comparative(bot, call, command):
    if not "wilcoxon_test_comparative" in user_columns[call.from_user.id]:
        user_columns[call.from_user.id]["wilcoxon_test_comparative"] = [
            int(command.replace("wilcoxon_test_comparative_", ""))
        ]

        bot.send_message(
            chat_id=call.from_user.id,
            text="Выберите вторую переменную:",
        )


    elif len(user_columns[call.from_user.id][
                 "wilcoxon_test_comparative"]) == 1:
        if (
                int(command.replace("wilcoxon_test_comparative_", ""))
                not in user_columns[call.from_user.id][
            "wilcoxon_test_comparative"]
        ):
            user_columns[call.from_user.id][
                "wilcoxon_test_comparative"].append(
                int(command.replace("wilcoxon_test_comparative_", ""))
            )
            build_wilcoxon_test_comparative(bot, call, command)

        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
            )

    else:
        user_columns[call.from_user.id]["wilcoxon_test_comparative"].pop(0)

        if (
                int(command.replace("wilcoxon_test_comparative_", ""))
                not in user_columns[call.from_user.id][
            "wilcoxon_test_comparative"]
        ):
            user_columns[call.from_user.id][
                "wilcoxon_test_comparative"].append(
                int(command.replace("wilcoxon_test_comparative_", ""))
            )
            build_wilcoxon_test_comparative(bot, call, command)


        else:
            bot.send_message(
                chat_id=call.from_user.id,
                text="Вы уже выбрали эту переменную. Выберите другую вторую переменную",
            )


def build_wilcoxon_test_comparative(bot, call, command):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)
    columns = user_columns[call.from_user.id]["wilcoxon_test_comparative"]

    if not columns:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    command = user_columns[call.from_user.id]["command"]

    if not command:
        bot.send_message(
            chat_id=call.from_user.id,
            text="Ошибка при обработке файла, попробуйте еще раз",
            reply_markup=keyboard_comparative_analysis,
        )

    else:
        if command == "wilcoxon_test_comparative":
            names_of_columns = user_columns[call.from_user.id]["columns"]

            module.generate_wilcoxon_test_comparative(
                names_of_columns[columns[0]], names_of_columns[columns[1]]
            )
            table_file = f"{MEDIA_PATH}/{DATA_PATH}/{COMPARATIVE_ANALYSIS}/{WILCOXON_TEST}/wilcoxon_test_comparative_{call.from_user.id}.xlsx"

            if os.path.isfile(table_file):
                bot.send_message(
                    chat_id=call.from_user.id,
                    text=f"На основе Ваших данных была построена таблица "
                         f"критерия Уилкоксона для переменной '{names_of_columns[columns[0]]}' "
                         f"и переменной '{names_of_columns[columns[1]]}'. "
                         f"\n\nЕсли p <= 0.05, можно сделать вывод о наличии "
                         f"статистически значимых различий между двумя "
                         f"связанными выборками.\n\nЕсли p > 0.05, это не"
                         f" позволяет отвергнуть нулевую гипотезу, и мы "
                         f"заключаем, что нет статистически значимых различий "
                         f"между двумя независимыми выборками."
                )

                file_cur = open(table_file, "rb")
                bot.send_document(
                    chat_id=call.from_user.id,
                    document=file_cur,
                    visible_file_name=f"Критерий_Уилкоксона_{names_of_columns[columns[0]]}_{names_of_columns[columns[1]]}.xlsx",
                )
#
