import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Kafka configuration
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka:9092')
INGESTION_TOPIC = os.environ.get('INGESTION_TOPIC', 'ingestion_events')
REALTIME_TOPIC = os.environ.get('REALTIME_TOPIC', 'realtime_events') # Nuevo topic para datos en tiempo real
STREAMING_TOPIC = os.environ.get('STREAMING_TOPIC', 'streaming_events')

# API Configuration
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx')
FMP_BASE_URL = os.environ.get('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Model configuration
MODEL_OUTPUT_PATH = os.getenv('MODEL_OUTPUT_PATH', '/models')
