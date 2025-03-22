import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "batch_events")
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH", "/models")
