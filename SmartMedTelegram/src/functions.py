import os

from describe_analysis.keyboard_descriptive import (
    keyboard_describe_analysis,
)
from dictionary.functions_dictionary import generate_dictionary_keyboard
from keyboard import keyboard_in_development, keyboard_modules, keyboard_main_menu
from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    USER_DATA_PATH,
)

"6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"
test_bot_token = "6727256721:AAEtOViOFY46Vk-cvEyLPRntAkwKPH_KVkU"


def get_reply_markup(command):
    """
    Вспомогательная функция для получения клавиатур.
    """
    switch = {
        "bioequal": keyboard_in_development,
        "описательный анализ": keyboard_describe_analysis,
        "predict": keyboard_in_development,
        "модули": keyboard_modules,
        "назад": keyboard_main_menu,
        "словарь": generate_dictionary_keyboard(0),
        "chat-gpt": keyboard_in_development,
        "cluster": keyboard_in_development,
    }
    return switch.get(command, None)


def send_text_message(bot, chat_id, text, reply_markup=None):
    """
    Вспомогательная функция для отправки текстовых сообщений.
    """
    bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


def save_file(file_content, file_name):
    file_path = f"{MEDIA_PATH}/{DATA_PATH}/{USER_DATA_PATH}/{file_name}"
    with open(file_path, "wb") as file:
        file.write(file_content)
    return file_path


def handle_back(bot, user_id):
    """
    Обработка запроса на возвращение к выбору модуля.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        user_id (int): Идентификатор пользователя.
    """
    bot.send_message(
        chat_id=user_id,
        text="Выберите интересующий вас раздел:",
        reply_markup=keyboard_main_menu,
    )


def send_document_from_file(bot, chat_id, file_path):
    """
    Отправка документа из файла.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        chat_id (int): Идентификатор чата.
        file_path (str): Путь к файлу.
    """
    if os.path.isfile(file_path):
        file = open(file_path, "rb")
        bot.send_document(chat_id=chat_id, document=file)
