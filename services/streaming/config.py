import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
INGESTION_TOPIC = os.getenv("INGESTION_TOPIC", "ingestion_events")
STREAMING_TOPIC = os.getenv("STREAMING_TOPIC", "streaming_events")
