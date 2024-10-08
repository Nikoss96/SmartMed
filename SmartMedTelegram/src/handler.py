from cluster_analysis.functions_cluster import (
    handle_example_cluster_analysis,
    handle_downloaded_cluster_file,
    handle_cluster_method,
    handle_choose_number_of_clusters,
    handle_cluster_numbers,
    handle_hierarchical,
    handle_hierarchical_cluster_numbers,
)
from cluster_analysis.keyboard_implementation_cluster import \
    handle_pagination_columns_cluster
from comparative_analysis.functions_comparative import (
    handle_example_comparative_analysis,
    handle_downloaded_comparative_file,
    handle_comparative_module,
    user_columns,
    handle_categorical_columns_comparative,
    handle_categorical_column_comparative,
    handle_t_criteria_categorical_value,
    handle_t_criterion_student_dependent,
    handle_t_criteria_for_dependent,
    handle_nonparametric_tests_comparative,
    handle_mann_whitney_test_comparative, handle_wilcoxon_test_comparative,
)
from comparative_analysis.keyboard_implementation_comparative import (
    handle_pagination_columns_comparative,
    handle_pagination_columns_t_criteria_dependent_comparative,
)
from describe_analysis.functions_descriptive import (
    handle_example_describe,
    handle_describe_build_graphs,
    handle_describe_correlation_analysis,
    handle_downloaded_describe_file,
    handle_describe_table,
    handle_describe_box_plot,
    handle_box_plot,
)
from describe_analysis.keyboard_implementation_describe import \
    handle_pagination_columns
from dictionary.functions_dictionary import (
    handle_pagination_dictionary,
    handle_statistical_term,
)
from gpt.functions_gpt import handle_gpt_message
from keyboard import keyboard_start
from functions import (
    get_reply_markup,
    handle_back,
    send_text_message,
    handle_download,
)
from variance_analysis.functions_variance_analysis import \
    handle_example_variance, handle_downloaded_variance_file, \
    handle_variance_module, handle_test_kruskal_wallis_variance, \
    handle_test_friedman_variance
from variance_analysis.keyboard_implementation_variance import \
    handle_pagination_columns_variance


def callback_query_handler(bot, call):
    """
    Обработка нажатия кнопок.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    try:
        command: str = call.data
        user_id = call.from_user.id
        username = call.from_user.username

        print(f"User {username} in {user_id} chat asked for {command}")

        # Обработка кнопки "Главное меню"
        if command == "back":
            handle_back(bot, user_id)

        # Обработка кнопок словаря
        elif command.startswith("dictionary_prev_") or command.startswith(
                "dictionary_next_"
        ):
            handle_pagination_dictionary(bot, call)

        elif command.startswith("statistical_term"):
            handle_statistical_term(bot, call)

        # Общие команды при обработке модулей
        elif command in [
            "download_describe",
            "download_cluster",
            "download_comparative",
            "download_variance",
        ]:
            handle_download(bot, call, command)

        # Обработка кнопок Описательного анализа
        elif command == "example_describe":
            handle_example_describe(bot, call)

        elif command in [
            "replace_null_with_mean_describe",
            "delete_null_rows_dropna_describe",
            "replace_null_with_median_describe",
        ]:
            handle_downloaded_describe_file(bot, call, command)

        elif command == "describe_build_graphs":
            handle_describe_build_graphs(bot, call)

        elif command == "describe_correlation_analysis":
            handle_describe_correlation_analysis(bot, call)

        elif command == "describe_table":
            handle_describe_table(bot, call)

        elif command == "describe_box_plot":
            handle_describe_box_plot(bot, call)

        elif command.startswith("boxplot_prev_") or command.startswith(
                "boxplot_next_"):
            handle_pagination_columns(bot, call)

        elif command.startswith("boxplot_column_"):
            handle_box_plot(bot, call)

        # Обработка кнопок Кластерного анализа
        elif command == "example_cluster":
            handle_example_cluster_analysis(bot, call)

        elif command in [
            "replace_null_with_mean_cluster",
            "delete_null_rows_dropna_cluster",
            "replace_null_with_median_cluster",
        ]:
            handle_downloaded_cluster_file(bot, call, command)

        elif command == "k_means_cluster":
            handle_cluster_method(bot, call, command)

        elif command == "choose_number_of_clusters":
            handle_choose_number_of_clusters(bot, call, command)

        elif command == "recommended_number_of_clusters":
            handle_cluster_numbers(bot, call, command)

        elif command == "hierarchical_cluster":
            handle_hierarchical(bot, call)

        elif command == "choose_number_of_clusters_hierarchical":
            handle_choose_number_of_clusters(bot, call, command)

        elif command.startswith("cluster_prev_") or command.startswith(
                "cluster_next_"):
            handle_pagination_columns_cluster(bot, call, command)

        elif command.startswith(
                "hierarchical_cluster_prev_") or command.startswith(
            "hierarchical_cluster_next_"
        ):
            handle_pagination_columns_cluster(bot, call, command)

        elif command.startswith("cluster_"):
            handle_cluster_numbers(bot, call, command)

        elif command.startswith("hierarchical_cluster_"):
            handle_hierarchical_cluster_numbers(bot, call, command)

        # Обработка кнопок Сравнительного анализа

        elif command == "example_comparative":
            handle_example_comparative_analysis(bot, call)

        elif command in [
            "replace_null_with_mean_comparative",
            "delete_null_rows_dropna_comparative",
            "replace_null_with_median_comparative",
        ]:
            handle_downloaded_comparative_file(bot, call, command)

        elif command in [
            "kolmogorov_smirnov_test_comparative",
            "t_criterion_student_independent_comparative",
        ]:
            handle_comparative_module(bot, call, command)

        elif command == "t_criterion_student_dependent_comparative":
            handle_t_criterion_student_dependent(bot, call, command)

        elif command in ["mann_whitney_test_comparative",
                         "wilcoxon_test_comparative"]:
            handle_nonparametric_tests_comparative(bot, call,
                                                   command)

        elif command.startswith(
                "continuous_columns_prev_") or command.startswith(
            "continuous_columns_next_"
        ):
            columns = user_columns[call.from_user.id]["continuous_columns"]
            handle_pagination_columns_comparative(bot, call, command, columns)

        elif command.startswith(
                "t_criterion_student_dependent_comparative_next_"
        ) or command.startswith(
            "t_criterion_student_dependent_comparative_prev_") or command.startswith(
            "mann_whitney_test_comparative_next_"
        ) or command.startswith(
            "mann_whitney_test_comparative_prev_") or command.startswith(
            "wilcoxon_test_comparative_next_"
        ) or command.startswith(
            "wilcoxon_test_comparative_prev_"):
            columns = user_columns[call.from_user.id]["columns"]
            handle_pagination_columns_t_criteria_dependent_comparative(
                bot, call, command, columns
            )

        elif command.startswith(
                "categorical_columns_prev_") or command.startswith(
            "categorical_columns_next_"
        ):
            columns = user_columns[call.from_user.id]["categorical_columns"]
            handle_pagination_columns_comparative(bot, call, command, columns)

        elif command.startswith("continuous_column_"):
            handle_categorical_columns_comparative(bot, call, command)

        elif command.startswith("categorical_column_"):
            handle_categorical_column_comparative(bot, call, command)

        elif command.startswith("dependent_column_"):
            handle_t_criteria_for_dependent(bot, call, command)

        elif command.startswith("mann_whitney_test_comparative_"):
            handle_mann_whitney_test_comparative(bot, call, command)

        elif command.startswith("wilcoxon_test_comparative_"):
            handle_wilcoxon_test_comparative(bot, call, command)

        elif command.startswith("t_criteria_categorical_value_"):
            handle_t_criteria_categorical_value(bot, call, command)

        # Обработка кнопок Дисперсионного анализа
        elif command == "example_variance":
            handle_example_variance(bot, call)

        elif command in [
            "replace_null_with_mean_variance",
            "delete_null_rows_dropna_variance",
            "replace_null_with_median_variance",
        ]:
            handle_downloaded_variance_file(bot, call, command)

        elif command in [
            "test_kruskal_wallis",
            "test_friedman"
        ]:
            handle_variance_module(bot, call, command)

        elif command.startswith(
                "test_kruskal_wallis_prev_") or command.startswith(
            "test_kruskal_wallis_next_"
        ):
            columns = user_columns[call.from_user.id]["columns"]
            handle_pagination_columns_variance(bot, call, command, columns)

        elif command.startswith(
                "test_friedman_prev_") or command.startswith(
            "test_friedman_next_"
        ):
            columns = user_columns[call.from_user.id]["columns"]
            handle_pagination_columns_variance(bot, call, command, columns)

        elif command.startswith("test_kruskal_wallis_"):
            handle_test_kruskal_wallis_variance(bot, call, command)

        elif command.startswith("test_friedman_"):
            handle_test_friedman_variance(bot, call, command)

    except Exception as e:
        print(f"Ошибка: \n{e}")


def start_message_handler(bot, message):
    """
    Обработка кнопки Start. Запускается при запуске бота пользователем.
    """
    try:
        user = message.from_user.username
        chat_id = message.chat.id

        print(f"User {user} in {chat_id} chat started the bot!")

        greeting_text = (
            "Доброго дня!\n\nРады приветствовать Вас "
            "в приложении Smart-Медицина!\n\nВам доступен следующий "
            "функционал: \n- Модули анализа данных\n"
            "- Словарь терминов\n- Искусственный интеллект"
        )

        send_text_message(
            bot,
            chat_id,
            greeting_text,
            reply_markup=keyboard_start,
            parse_mode="Markdown",
        )

    except Exception as e:
        print(f"Ошибка: \n{e}")


def text_handler(bot, message):
    """
    Обработка текста, присылаемого пользователем.
    """
    try:
        command = message.text.lower()
        reply_markup = get_reply_markup(command)
        chat_id = message.chat.id
        username = message.from_user.username

        if command in [
            "описательный анализ",
            "кластерный анализ",
            "сравнительный анализ",
            "дисперсионный анализ",
        ]:
            send_text_message(
                bot,
                chat_id=chat_id,
                text="Выберите опцию при работе с модулем:",
                reply_markup=reply_markup,
            )

        elif command == "модули":
            send_text_message(
                bot,
                chat_id=chat_id,
                text="Выберите интересующий Вас модуль:",
                reply_markup=reply_markup,
            )

        elif command == "искусственный интеллект":
            send_text_message(
                bot,
                chat_id=chat_id,
                text="Напишите Ваш запрос к искусственному интеллекту:",
            )

        elif command == "словарь":
            send_text_message(
                bot,
                chat_id=chat_id,
                text="Выберите интересующий Вас раздел:",
                reply_markup=reply_markup,
            )

        else:
            handle_gpt_message(bot, message)

        print(f"User {username} in {chat_id} chat wrote {command}")
    except Exception as e:
        print(f"Ошибка: \n{e}")
