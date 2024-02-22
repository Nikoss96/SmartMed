import os

from dotenv import load_dotenv


env_path = os.path.join(os.path.dirname(__file__), "..", ".env")

load_dotenv(env_path)

main_bot_token = os.getenv("MAIN_BOT_TOKEN")
test_bot_token = os.getenv("TEST_BOT_TOKEN")
