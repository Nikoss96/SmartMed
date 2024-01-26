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


def generate_dictionary_keyboard(page):
    keyboard_terms = InlineKeyboardMarkup()

    words_per_page = 2

    for term_key in list(statistical_terms.keys())[
                    page * words_per_page: (page + 1) * words_per_page]:
        term_description = statistical_terms[term_key][0]
        button = InlineKeyboardButton(term_description,
                                      callback_data=f"statistical_{term_key}")
        keyboard_terms.add(button)

    if page > 0:
        prev_button = InlineKeyboardButton("Назад",
                                           callback_data=f'prev_{page}')
    else:
        prev_button = None

    if (page + 1) * words_per_page < len(statistical_terms):
        next_button = InlineKeyboardButton("Далее",
                                           callback_data=f'next_{page + 1}')
    else:
        next_button = None

    home_button = InlineKeyboardButton(text="В главное меню",
                                       callback_data="back")

    if prev_button and next_button:
        keyboard_terms.add(prev_button, next_button)
        keyboard_terms.add(home_button)

    elif prev_button:
        keyboard_terms.add(prev_button, home_button)

    elif next_button:
        keyboard_terms.add(home_button, next_button)

    return keyboard_terms


def open_and_send_file(bot, chat_id, image):
    file_cur = open(f"media/images/{image}.png", "rb")

    bot.send_photo(chat_id=chat_id, photo=file_cur)
