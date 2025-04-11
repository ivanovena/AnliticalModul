import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuración de servicios
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
AGENT_TOPIC = os.getenv("AGENT_TOPIC", "agent_decisions")
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
USE_DATALAKE = os.getenv("USE_DATALAKE", "false").lower() == "true"

# URLs de servicios con valores por defecto para entorno Docker
INGESTION_SERVICE_URL = os.getenv("INGESTION_SERVICE_URL", "http://ingestion:8080")
INGESTION_WS_URL = os.getenv("INGESTION_WS_URL", "ws://ingestion:8080")
STREAMING_SERVICE_URL = os.getenv("STREAMING_SERVICE_URL", "http://streaming:8090")

# Configuración de Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # Vacío por defecto
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Configuración del modelo Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "phi4:latest")
CODER_MODEL = os.getenv("CODER_MODEL", "deepseek-coder:33b")

# Versión API para verificación de compatibilidad
API_VERSION = "1.0"

logger.info(f"Configuración cargada. URL Ingestion: {INGESTION_SERVICE_URL}")
logger.info(f"URL WebSocket Ingestion: {INGESTION_WS_URL}")
logger.info(f"Redis configurado en: {REDIS_HOST}:{REDIS_PORT}")
logger.info(f"Versión API: {API_VERSION}")
