import requests
from requests import RequestException
from telebot.apihelper import ApiTelegramException
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

from statistical_terms import statistical_terms
from tokens import main_bot_token


def get_anyfile(bot, call):
    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        try:
            file_info = bot.get_file(message.document.file_id)
            file_url = f"https://api.telegram.org/file/bot{main_bot_token}/{file_info.file_path}"

            response = requests.get(file_url)

            # Получение файла пользователя
            if response.status_code == 200:
                file_name = f"media/images/{message.document.file_name}"

                with open(file_name, "wb") as file:
                    file.write(response.content)

                bot.reply_to(message=message,
                             text=f"Файл {message.document.file_name} успешно загружен")

                file = open(file_name, "rb")

                bot.send_document(chat_id=call.from_user.id, document=file)

            else:
                bot.reply_to(message, "Произошла ошибка при загрузке файла")

        except ApiTelegramException as e:
            print(f"API Error: {e}")
        except RequestException as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


def get_file_for_descriptive_analysis(bot, call):
    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        try:
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

                bot.reply_to(message=message,
                             text=f"Файл {file_name} успешно загружен")

                # Временная обертка
                file = open("media/images/statistical_term_3.png", "rb")

                bot.send_document(chat_id=call.from_user.id, document=file)

            else:
                bot.reply_to(message, "Произошла ошибка при загрузке файла")

        except ApiTelegramException as e:
            print(f"API Error: {e}")

        except RequestException as e:
            print(f"Request error: {e}")

        except Exception as e:
            print(f"Unexpected error: {e}")


def generate_inline_keyboard_rows(start_key=1, limit=3):
    keyboard_terms = InlineKeyboardMarkup()
    for key_word, value in statistical_terms.items():
        key = int(key_word.replace("term_", ""))
        if start_key + limit >= key >= start_key:
            keyboard_terms.add(
                InlineKeyboardButton(text=value[0],
                                     callback_data=f'statistical_term_{key}')
            )
    keyboard_terms.add(
        InlineKeyboardButton(text="Назад", callback_data="back"))

    return keyboard_terms


def open_and_send_file(bot, chat_id, image):
    file_cur = open(f"media/images/{image}.png", "rb")

    bot.send_photo(chat_id=chat_id, photo=file_cur)
