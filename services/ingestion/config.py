import os
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api/v3")
DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@postgres:5432/trading")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
INGESTION_TOPIC = os.getenv("INGESTION_TOPIC", "ingestion_events")
