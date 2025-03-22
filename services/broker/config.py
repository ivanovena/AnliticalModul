import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
AGENT_TOPIC = os.getenv("AGENT_TOPIC", "agent_decisions")
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
USE_DATALAKE = os.getenv("USE_DATALAKE", "false").lower() == "true"
