# Подкачиваем файл из диалога
import requests

from tokens import main_bot_token


def get_anyfile(bot, call):
    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        file_info = bot.get_file(message.document.file_id)
        file_url = (
            f"https://api.telegram.org/file/bot{main_bot_token}/{file_info.file_path}"
        )

        response = requests.get(file_url)

        # Получение файла пользователя
        if response.status_code == 200:
            file_name = message.document.file_name

            with open(file_name, "wb") as file:
                file.write(response.content)

            bot.reply_to(message=message, text=f"Файл {file_name} успешно загружен")

            file = open(file_name, "rb")

            bot.send_document(chat_id=call.from_user.id, document=file)

        else:
            bot.reply_to(message, "Произошла ошибка при загрузке файла")


def get_file_for_descriptive_analysis(bot, call):
    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        file_info = bot.get_file(message.document.file_id)
        file_url = (
            f"https://api.telegram.org/file/bot{main_bot_token}/{file_info.file_path}"
        )

        response = requests.get(file_url)

        # Получение файла пользователя
        if response.status_code == 200:
            file_name = message.document.file_name

            with open(file_name, "wb") as file:
                file.write(response.content)

            bot.reply_to(message=message, text=f"Файл {file_name} успешно загружен")

            # Временная обертка
            file = open("media/images/picdist.png", "rb")

            bot.send_document(chat_id=call.from_user.id, document=file)

        else:
            bot.reply_to(message, "Произошла ошибка при загрузке файла")
