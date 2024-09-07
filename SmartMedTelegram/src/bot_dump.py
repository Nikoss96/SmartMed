from telebot import TeleBot

from handler import callback_query_handler, start_message_handler, text_handler
from settings import bot_token

bot = TeleBot(bot_token)


# Обработка стартового сообщения пользователя "/start"
@bot.message_handler(commands=["start"])
def handle_start_message(message):
    start_message_handler(bot, message)


# Обработка нажатий на кнопки пользователем
@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(message):
    callback_query_handler(bot, message)


# Обработка текстовых сообщений пользователя
@bot.message_handler(content_types=["text"])
def handle_text(message):
    text_handler(bot, message)


# Запуск бесконечной работы чат-бота
bot.infinity_polling(timeout=10, long_polling_timeout=5)
