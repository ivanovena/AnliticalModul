import os
import json
import requests
import logging
import time
from typing import Dict, List, Any, Optional
from functools import lru_cache
import threading

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ollama_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OllamaClient")

class OllamaClient:
    """Cliente para interactuar con la API de Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", max_retries: int = 3, timeout: int = 60):
        """
        Inicializa el cliente de Ollama
        
        Args:
            base_url: URL base de la API de Ollama
            max_retries: Número máximo de reintentos
            timeout: Tiempo de espera en segundos
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.lock = threading.RLock()  # Para operaciones seguras en hilos
        
        # Verificar conexión
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """
        Verifica la conexión con Ollama
        
        Returns:
            True si la conexión es exitosa, False en caso contrario
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("Conexión con Ollama establecida correctamente")
                return True
            else:
                logger.warning(f"Error conectando con Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error verificando conexión con Ollama: {e}")
            return False
    
    def _call_api(self, endpoint: str, method: str = "post", data: Dict[str, Any] = None, 
                 retry_count: int = 0) -> Dict[str, Any]:
        """
        Llama a un endpoint de la API de Ollama con manejo de errores y reintentos
        
        Args:
            endpoint: Endpoint de la API (sin la URL base)
            method: Método HTTP ("get" o "post")
            data: Datos para enviar en solicitudes POST
            retry_count: Contador de reintentos (uso interno)
            
        Returns:
            Respuesta de la API como diccionario
        """
        url = f"{self.base_url}/api/{endpoint}"
        
        try:
            with self.lock:  # Proteger llamadas a la API en un entorno multi-hilo
                if method.lower() == "get":
                    response = self.session.get(url, timeout=self.timeout)
                else:
                    response = self.session.post(url, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Error en llamada a API {endpoint}: {response.status_code}, {response.text}")
                    
                    # Reintentar si no hemos alcanzado el máximo
                    if retry_count < self.max_retries:
                        time.sleep(1 * (retry_count + 1))  # Backoff exponencial simple
                        return self._call_api(endpoint, method, data, retry_count + 1)
                    
                    return {"error": f"Error en llamada a API: {response.status_code}", "status_code": response.status_code}
                    
        except requests.exceptions.Timeout:
            logger.error(f"Timeout en llamada a API {endpoint}")
            
            # Reintentar si no hemos alcanzado el máximo
            if retry_count < self.max_retries:
                time.sleep(1 * (retry_count + 1))
                return self._call_api(endpoint, method, data, retry_count + 1)
            
            return {"error": "Timeout en llamada a API"}
            
        except Exception as e:
            logger.error(f"Error en llamada a API {endpoint}: {e}")
            
            # Reintentar si no hemos alcanzado el máximo
            if retry_count < self.max_retries:
                time.sleep(1 * (retry_count + 1))
                return self._call_api(endpoint, method, data, retry_count + 1)
            
            return {"error": f"Error en llamada a API: {str(e)}"}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista los modelos disponibles en Ollama
        
        Returns:
            Lista de modelos
        """
        response = self._call_api("tags", method="get")
        return response.get("models", [])
    
    def generate(self, model: str, prompt: str, system: Optional[str] = None, 
                options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera texto con un modelo de Ollama
        
        Args:
            model: Nombre del modelo
            prompt: Prompt para la generación
            system: Prompt de sistema (opcional)
            options: Opciones de generación (opcional)
            
        Returns:
            Respuesta del modelo
        """
        if options is None:
            options = {}
        
        data = {
            "model": model,
            "prompt": prompt,
            "options": options
        }
        
        if system:
            data["system"] = system
        
        return self._call_api("generate", data=data)
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]], 
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera una respuesta de chat con un modelo de Ollama
        
        Args:
            model: Nombre del modelo
            messages: Lista de mensajes (cada uno con "role" y "content")
            options: Opciones de generación (opcional)
            
        Returns:
            Respuesta del modelo
        """
        if options is None:
            options = {}
        
        data = {
            "model": model,
            "messages": messages,
            "options": options
        }
        
        return self._call_api("chat", data=data)
    
    def pull_model(self, model: str) -> Dict[str, Any]:
        """
        Descarga un modelo de Ollama
        
        Args:
            model: Nombre del modelo
            
        Returns:
            Estado de la descarga
        """
        data = {"name": model}
        return self._call_api("pull", data=data)
    
    def model_info(self, model: str) -> Dict[str, Any]:
        """
        Obtiene información sobre un modelo
        
        Args:
            model: Nombre del modelo
            
        Returns:
            Información del modelo
        """
        data = {"name": model}
        return self._call_api("show", data=data)

# Crear y cachear cliente para singleton
@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    """
    Obtiene una instancia única del cliente de Ollama
    
    Returns:
        Cliente de Ollama
    """
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    return OllamaClient(base_url=ollama_url)

# Si se ejecuta como script principal, mostrar modelos disponibles
if __name__ == "__main__":
    client = get_ollama_client()
    models = client.list_models()
    print(f"Modelos disponibles: {json.dumps(models, indent=2)}")