import os

from comparative_analysis.ComparativeModule import ComparativeModule
from comparative_analysis.keyboard_comparative import (
    keyboard_choice_comparative,
    keyboard_comparative_analysis,
)
from comparative_analysis.keyboard_implementation import (
    handle_choose_column_comparative,
    generate_categorical_value_column_keyboard,
)
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    EXAMPLES,
    USER_DATA_PATH,
    COMPARATIVE_ANALYSIS,
    KOLMOGOROVA_SMIRNOVA,
    T_CRITERIA_INDEPENDENT,
)
from describe_analysis.functions_descriptive import get_user_file_df
from functions import send_document_from_file, create_dataframe_and_save_file
from keyboard import keyboard_in_development

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
                             f"характер.\n\nЕсли p ≥ 0.05, принимается нулевая "
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
                 f"\n\nЕсли p ≥ 0.05, принимается нулевая гипотеза, различия "
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
