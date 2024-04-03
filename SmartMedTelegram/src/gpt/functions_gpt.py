import os

from dotenv import load_dotenv
from telebot.apihelper import ApiTelegramException
from yandexgptlite import YandexGPTLite

from keyboard import keyboard_in_development

load_dotenv()
yandex_gpt_folder = os.getenv("YANDEX_GPT_FOLDER")
yandex_gpt_token = os.getenv("YANDEX_GPT_TOKEN")
account = YandexGPTLite(yandex_gpt_folder, yandex_gpt_token)


def handle_gpt_message(bot, message):
    bot_message = bot.send_message(
        chat_id=message.from_user.id,
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
                reply_markup=keyboard_in_development,
            )

    except ApiTelegramException as e:
        bot.send_message(
            chat_id=message.from_user.id,
            text="Извините, не удалось сгенерировать ответ. Попробуйте еще раз.",
            reply_markup=keyboard_in_development,
        )
