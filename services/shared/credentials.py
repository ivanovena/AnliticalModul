"""
Módulo compartido para gestionar credenciales y configuraciones sensibles.
Centraliza el acceso a las API keys y credenciales.
"""

import os
import logging
from dotenv import load_dotenv

# Configurar logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env si existe
load_dotenv()

class Credentials:
    """Clase para gestionar credenciales y configuraciones sensibles."""
    
    @staticmethod
    def get_fmp_api_key():
        """Obtiene la clave API de Financial Modeling Prep"""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.critical("FMP_API_KEY no está configurada en las variables de entorno")
            raise EnvironmentError("FMP_API_KEY is missing")
        return api_key
    
    @staticmethod
    def get_telegram_token():
        """Obtiene el token del bot de Telegram, devuelve None si no está configurado"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.warning("TELEGRAM_BOT_TOKEN no está configurada. La funcionalidad de Telegram estará deshabilitada.")
        return token
    
    @staticmethod
    def get_db_uri():
        """Obtiene la URI de conexión a la base de datos"""
        db_uri = os.getenv("DB_URI")
        if not db_uri:
            logger.critical("DB_URI no está configurada en las variables de entorno")
            raise EnvironmentError("DB_URI is missing")
        return db_uri
    
    @staticmethod
    def get_kafka_broker():
        """Obtiene la dirección del broker de Kafka"""
        broker = os.getenv("KAFKA_BROKER", "kafka:9092")
        return broker
    
    @staticmethod
    def validate_all_credentials():
        """Valida que todas las credenciales necesarias estén configuradas"""
        required_vars = [
            "FMP_API_KEY",
            "DB_URI",
            "KAFKA_BROKER"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            missing_vars_str = ", ".join(missing_vars)
            logger.critical(f"Faltan las siguientes variables de entorno: {missing_vars_str}")
            raise EnvironmentError(f"Missing required environment variables: {missing_vars_str}")
        
        optional_vars = ["TELEGRAM_BOT_TOKEN"]
        for var in optional_vars:
            if not os.getenv(var):
                logger.warning(f"Variable opcional {var} no configurada")
        
        return True


# Ejecutar validación al importar este módulo
if __name__ != "__main__":  # No ejecutar automáticamente si se ejecuta como script
    try:
        Credentials.validate_all_credentials()
        logger.info("Todas las credenciales obligatorias están configuradas correctamente")
    except Exception as e:
        logger.error(f"Error validando credenciales: {e}")
