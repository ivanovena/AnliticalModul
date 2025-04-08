import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Kafka configuration
KAFKA_HOST = os.getenv('KAFKA_HOST', 'kafka')
KAFKA_PORT = os.getenv('KAFKA_PORT', '9092')
KAFKA_BROKER = f"{KAFKA_HOST}:{KAFKA_PORT}"
INGESTION_TOPIC = os.getenv('INGESTION_TOPIC', 'ingestion_events')
STREAMING_TOPIC = os.getenv('STREAMING_TOPIC', 'streaming_events')

# API configuration
FMP_API_KEY = os.getenv('FMP_API_KEY', 'h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx')
FMP_BASE_URL = os.getenv('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Model configuration
MODEL_OUTPUT_PATH = os.getenv('MODEL_OUTPUT_PATH', '/models')
