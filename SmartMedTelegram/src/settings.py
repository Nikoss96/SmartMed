import os

from dotenv import load_dotenv

load_dotenv()
yandex_gpt_folder = os.getenv("YANDEX_GPT_FOLDER")
yandex_gpt_token = os.getenv("YANDEX_GPT_TOKEN")
bot_token = os.getenv("BOT_TOKEN")
