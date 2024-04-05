from telebot.apihelper import ApiTelegramException
from yandexgptlite import YandexGPTLite

from keyboard import keyboard_main_menu
from settings import yandex_gpt_folder, yandex_gpt_token

account = YandexGPTLite(yandex_gpt_folder, yandex_gpt_token)


def handle_gpt_message(bot, message):
    chat_id = message.from_user.id

    bot_message = bot.send_message(
        chat_id=chat_id,
        text="Ваш запрос обрабатывается...",
    )

    try:
        text = account.create_completion(
            message.text,
            "0.6",
        )

        if text:
            bot.edit_message_text(
                chat_id=bot_message.chat.id,
                message_id=bot_message.message_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=keyboard_main_menu,
            )

    except ApiTelegramException as e:
        bot.send_message(
            chat_id=chat_id,
            text="Извините, не удалось сгенерировать ответ. Попробуйте еще раз.",
            reply_markup=keyboard_main_menu,
        )
