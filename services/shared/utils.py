"""
Utilidades comunes para todos los servicios
"""

import logging
import os
import time
from functools import wraps

def setup_logging(service_name, log_level=None):
    """
    Configura el sistema de logging para un servicio.
    
    Args:
        service_name: Nombre del servicio
        log_level: Nivel de logging (si es None, se toma de la variable de entorno LOG_LEVEL)
    
    Returns:
        Logger configurado
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Configurar formato y handlers
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Crear directorio de logs si no existe
    log_dir = os.path.join(os.getcwd(), "logs", service_name.lower())
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{service_name.lower()}.log")),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(service_name)
    logger.info(f"Logging configurado para {service_name} con nivel {log_level}")
    
    return logger

def retry(max_tries=3, delay_seconds=1, backoff_factor=2, exceptions=(Exception,)):
    """
    Decorador para reintentar funciones con backoff exponencial
    
    Args:
        max_tries: Número máximo de intentos
        delay_seconds: Retardo inicial en segundos
        backoff_factor: Factor de backoff
        exceptions: Excepciones que deben capturarse para reintentar
        
    Returns:
        Decorador
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_tries, delay_seconds
            last_exception = None
            
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    mtries -= 1
                    if mtries == 0:
                        raise
                        
                    logging.warning(f"Reintentando {func.__name__} en {mdelay} segundos... ({max_tries - mtries}/{max_tries})")
                    time.sleep(mdelay)
                    mdelay *= backoff_factor
                    
            return last_exception
        return wrapper
    return decorator
