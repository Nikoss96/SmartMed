from telebot import TeleBot
from handler import start_message_handler, callback_query_handler, text_handler
from tokens import main_bot_token

bot = TeleBot(main_bot_token)


@bot.message_handler(commands=["start"])
def handle_start_message(message):
    start_message_handler(bot, message)


@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(message):
    callback_query_handler(bot, message)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    text_handler(bot, message)


bot.polling()
