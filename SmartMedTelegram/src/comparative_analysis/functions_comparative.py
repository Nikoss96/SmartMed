import os

from comparative_analysis.ComparativeModule import ComparativeModule
from comparative_analysis.keyboard_comparative import \
    keyboard_choice_comparative, keyboard_comparative_analysis
from comparative_analysis.keyboard_implementation import (
    handle_choose_column_comparative,
)
from data.paths import MEDIA_PATH, DATA_PATH, EXAMPLES, USER_DATA_PATH, \
    COMPARATIVE_ANALYSIS
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


def handle_kolmogorov_smirnov_test_comparative(bot, call, command):
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
            text="В Вашем файле отсутствуют непрерывные переменные. "
                 "Загрузите файл, который содержит непрерывные переменные.",
            reply_markup=keyboard_comparative_analysis,
        )

    else:

        user_columns[call.from_user.id] = {}
        user_columns[call.from_user.id][
            "categorical_columns"] = categorical_columns
        user_columns[call.from_user.id][
            "continuous_columns"] = continuous_columns

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

        handle_continuous_columns_comparative(bot, call)


def handle_continuous_columns_comparative(bot, call):
    continuous_columns = user_columns[call.from_user.id]["continuous_columns"]

    handle_choose_column_comparative(
        bot, call, continuous_columns, "continuous_columns"
    )


def handle_categorical_columns_comparative(bot, call, command):
    categorical_columns = user_columns[call.from_user.id]["categorical_columns"]

    user_columns[call.from_user.id]["continuous_column"] = int(
        command.replace("continuous_column_", ""))

    if "categorical_column" in user_columns[call.from_user.id]:
        build_kolmogorova_smirnova(bot, call)

    else:
        handle_choose_column_comparative(
            bot, call, categorical_columns, "categorical_columns"
        )


def handle_categorical_column_comparative(bot, call, command):
    user_columns[call.from_user.id]["categorical_column"] = int(
        command.replace("categorical_column_", ""))

    build_kolmogorova_smirnova(bot, call)


def build_kolmogorova_smirnova(bot, call):
    df = get_user_file_df(
        f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}",
        call.from_user.id,
    )

    module = ComparativeModule(df, call.from_user.id)
    categorical_column_index = user_columns[call.from_user.id][
        "categorical_column"]

    categorical_column = user_columns[call.from_user.id]["categorical_columns"][
        categorical_column_index]

    continuous_column_index = user_columns[call.from_user.id][
        "continuous_column"]

    continuous_column = user_columns[call.from_user.id]["continuous_columns"][
        continuous_column_index]

    if not categorical_column or not continuous_column:
        bot.send_message(chat_id=call.from_user.id,
                         text="Ошибка при обработке файла, попробуйте еще раз",
                         reply_markup=keyboard_comparative_analysis)
    result1, resul2 = module.generate_test_kolmagorova_smirnova(
        categorical_column,
        continuous_column)

    print(result1, resul2)
