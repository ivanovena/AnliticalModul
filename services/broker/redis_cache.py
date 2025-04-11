# redis_cache.py
import json
import os
import logging
import redis
from typing import Any, Dict, List, Optional, Union
from datetime import timedelta
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("redis_cache.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RedisCache")

class RedisCache:
    """Cliente para interactuar con Redis como caché para el broker"""
    
    def __init__(self, host: str = None, port: int = None, password: str = None, 
                db: int = 0, prefix: str = "broker:", ttl: int = 300):
        """
        Inicializa el cliente de Redis
        
        Args:
            host: Host de Redis (por defecto usa la variable de entorno REDIS_HOST)
            port: Puerto de Redis (por defecto usa la variable de entorno REDIS_PORT)
            password: Contraseña de Redis (por defecto usa la variable de entorno REDIS_PASSWORD)
            db: Base de datos de Redis
            prefix: Prefijo para las claves de Redis
            ttl: Tiempo de vida por defecto de las claves en segundos
        """
        self.host = host or os.getenv("REDIS_HOST", "redis")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.password = password or os.getenv("REDIS_PASSWORD", "redis123")
        self.db = db
        self.prefix = prefix
        self.default_ttl = ttl
        self.client = None
        self.connected = False
        
        # Intentar conectar
        self._connect()
        
    def _connect(self) -> None:
        """Establece conexión con Redis"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Verificar conexión
            self.client.ping()
            self.connected = True
            logger.info(f"Conexión establecida con Redis en {self.host}:{self.port}")
            
        except Exception as e:
            self.connected = False
            logger.error(f"Error al conectar con Redis: {e}")
    
    def _build_key(self, key: str) -> str:
        """
        Construye una clave con el prefijo
        
        Args:
            key: Clave base
            
        Returns:
            Clave con prefijo
        """
        return f"{self.prefix}{key}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Guarda un valor en Redis
        
        Args:
            key: Clave del valor
            value: Valor a guardar (se convierte a JSON)
            ttl: Tiempo de vida en segundos (opcional)
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        if not self.connected or not self.client:
            self._connect()
            if not self.connected:
                return False
        
        try:
            redis_key = self._build_key(key)
            json_value = json.dumps(value)
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            
            self.client.setex(redis_key, ttl_seconds, json_value)
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar en Redis: {e}")
            self.connected = False
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de Redis
        
        Args:
            key: Clave del valor
            default: Valor por defecto si no existe la clave
            
        Returns:
            Valor almacenado o valor por defecto
        """
        if not self.connected or not self.client:
            self._connect()
            if not self.connected:
                return default
        
        try:
            redis_key = self._build_key(key)
            value = self.client.get(redis_key)
            
            if value is None:
                return default
                
            return json.loads(value)
            
        except Exception as e:
            logger.error(f"Error al obtener de Redis: {e}")
            self.connected = False
            return default
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def delete(self, key: str) -> bool:
        """
        Elimina una clave de Redis
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if not self.connected or not self.client:
            self._connect()
            if not self.connected:
                return False
        
        try:
            redis_key = self._build_key(key)
            self.client.delete(redis_key)
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar de Redis: {e}")
            self.connected = False
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def exists(self, key: str) -> bool:
        """
        Verifica si existe una clave en Redis
        
        Args:
            key: Clave a verificar
            
        Returns:
            True si existe, False en caso contrario
        """
        if not self.connected or not self.client:
            self._connect()
            if not self.connected:
                return False
        
        try:
            redis_key = self._build_key(key)
            return bool(self.client.exists(redis_key))
            
        except Exception as e:
            logger.error(f"Error al verificar existencia en Redis: {e}")
            self.connected = False
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Obtiene claves que coinciden con un patrón
        
        Args:
            pattern: Patrón de búsqueda
            
        Returns:
            Lista de claves que coinciden con el patrón
        """
        if not self.connected or not self.client:
            self._connect()
            if not self.connected:
                return []
        
        try:
            redis_pattern = self._build_key(pattern)
            keys = self.client.keys(redis_pattern)
            
            # Eliminar el prefijo de las claves
            prefix_len = len(self.prefix)
            return [key[prefix_len:] for key in keys]
            
        except Exception as e:
            logger.error(f"Error al obtener claves de Redis: {e}")
            self.connected = False
            return []
    
    def close(self) -> None:
        """Cierra la conexión con Redis"""
        if self.client:
            try:
                self.client.close()
                logger.info("Conexión con Redis cerrada")
            except Exception as e:
                logger.error(f"Error al cerrar conexión con Redis: {e}")
        
        self.connected = False
        self.client = None

# Crear instancia única para uso en la aplicación
_redis_cache_instance = None

def get_redis_cache() -> RedisCache:
    """
    Obtiene una instancia única de RedisCache
    
    Returns:
        Instancia de RedisCache
    """
    global _redis_cache_instance
    
    if _redis_cache_instance is None:
        _redis_cache_instance = RedisCache()
    
    return _redis_cache_instance

# Función para limpiar recursos al salir
def close_redis_connection():
    """Cierra la conexión con Redis al salir"""
    global _redis_cache_instance
    
    if _redis_cache_instance:
        _redis_cache_instance.close()
        _redis_cache_instance = None