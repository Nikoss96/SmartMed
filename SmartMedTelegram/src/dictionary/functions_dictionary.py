import os

from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from data.paths import MEDIA_PATH, IMAGES_PATH, TERMS_PATH
from dictionary.statistical_terms import statistical_terms


def generate_dictionary_keyboard(page):
    """
    Генерация клавиатуры с терминами для словаря.
    """
    keyboard_terms = InlineKeyboardMarkup()
    words_per_page = 4

    for term_key in list(statistical_terms.keys())[
        page * words_per_page : (page + 1) * words_per_page
    ]:
        term_description = statistical_terms[term_key][0]
        button = InlineKeyboardButton(
            term_description, callback_data=f"statistical_{term_key}"
        )
        keyboard_terms.add(button)

    prev_button = (
        InlineKeyboardButton("Назад", callback_data=f"prev_{page}")
        if page > 0
        else None
    )
    next_button = (
        InlineKeyboardButton("Далее", callback_data=f"next_{page + 1}")
        if (page + 1) * words_per_page < len(statistical_terms)
        else None
    )
    home_button = InlineKeyboardButton("Главное меню", callback_data="back")

    if prev_button and next_button:
        keyboard_terms.add(prev_button, home_button, next_button)
    elif prev_button:
        keyboard_terms.add(prev_button, home_button)
    elif next_button:
        keyboard_terms.add(home_button, next_button)
    else:
        keyboard_terms.add(home_button)

    return keyboard_terms


def handle_pagination(bot, call):
    """
    Обработка нажатия кнопок "Prev" и "Next" для пагинации терминов.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    action, page = call.data.split("_") if "_" in call.data else (call.data, 0)
    page = int(page)

    if action == "prev":
        page -= 1

    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="Выберите интересующий вас термин:",
        reply_markup=generate_dictionary_keyboard(page),
    )


def handle_statistical_term(bot, call):
    """
    Обработка выбора статистического термина.

    Parameters:
        bot (telebot.TeleBot): Экземпляр бота.
        call (telebot.types.CallbackQuery): Callback-запрос от пользователя.
    """
    term = call.data.replace("statistical_term_", "")
    bot.send_message(
        chat_id=call.from_user.id,
        text=" – это ".join(statistical_terms[f"term_{term}"]),
    )
    send_term_image(bot, call.from_user.id, call.data)


def send_term_image(bot, chat_id, image):
    """
    Открытие и отправка изображения по его имени.
    """
    file_path = f"{MEDIA_PATH}/{IMAGES_PATH}/{TERMS_PATH}/{image}.png"

    if os.path.isfile(file_path):
        file_cur = open(file_path, "rb")
        bot.send_photo(chat_id=chat_id, photo=file_cur)
