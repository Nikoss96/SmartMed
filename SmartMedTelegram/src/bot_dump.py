import os

from telebot import TeleBot

from handler import callback_query_handler, start_message_handler, text_handler

bot_token = os.getenv("BOT_TOKEN")

bot = TeleBot(bot_token)


@bot.message_handler(commands=["start"])
def handle_start_message(message):
    start_message_handler(bot, message)


@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(message):
    callback_query_handler(bot, message)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    text_handler(bot, message)


bot.infinity_polling(timeout=10, long_polling_timeout=5)
