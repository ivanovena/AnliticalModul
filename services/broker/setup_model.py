#!/usr/bin/env python3
"""
Script para descargar y configurar el modelo LLM para el broker IA
"""
import os
import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelSetup")

# Directorio de modelos
MODEL_DIR = Path("models")
MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def setup_model():
    """Descargar y configurar el modelo LLM"""
    try:
        logger.info("Iniciando configuración del modelo...")
        
        # Crear directorio de modelos si no existe
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / MODEL_NAME
        
        # Verificar si el modelo ya existe
        if model_path.exists():
            logger.info(f"El modelo ya existe en: {model_path}")
            return True
        
        # Intentar descargar desde Hugging Face
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Descargando modelo {MODEL_FILE} desde {MODEL_REPO}...")
            downloaded_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            
            # Renombrar para consistencia
            target_path = MODEL_DIR / MODEL_NAME
            if Path(downloaded_path) != target_path:
                logger.info(f"Renombrando modelo a {MODEL_NAME}")
                os.rename(downloaded_path, target_path)
            
            logger.info(f"Modelo descargado correctamente en: {target_path}")
            return True
            
        except ImportError:
            logger.error("No se pudo importar huggingface_hub. Instalando...")
            os.system(f"{sys.executable} -m pip install huggingface-hub")
            
            # Intentar de nuevo
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Descargando modelo {MODEL_FILE} desde {MODEL_REPO}...")
            downloaded_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            
            # Renombrar para consistencia
            target_path = MODEL_DIR / MODEL_NAME
            if Path(downloaded_path) != target_path:
                logger.info(f"Renombrando modelo a {MODEL_NAME}")
                os.rename(downloaded_path, target_path)
            
            logger.info(f"Modelo descargado correctamente en: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error descargando modelo: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error configurando modelo: {e}")
        return False

def check_dependencies():
    """Verificar y instalar dependencias"""
    try:
        # Verificar llama-cpp-python
        try:
            import llama_cpp
            logger.info("llama-cpp-python ya está instalado")
        except ImportError:
            logger.info("Instalando llama-cpp-python...")
            os.system(f"{sys.executable} -m pip install llama-cpp-python")
        
        # Verificar huggingface_hub
        try:
            import huggingface_hub
            logger.info("huggingface-hub ya está instalado")
        except ImportError:
            logger.info("Instalando huggingface-hub...")
            os.system(f"{sys.executable} -m pip install huggingface-hub")
        
        return True
    except Exception as e:
        logger.error(f"Error verificando dependencias: {e}")
        return False
    
if __name__ == "__main__":
    logger.info("Verificando dependencias...")
    if check_dependencies():
        logger.info("Todas las dependencias están instaladas")
        
        logger.info("Configurando modelo...")
        if setup_model():
            logger.info("Modelo configurado correctamente")
            sys.exit(0)
        else:
            logger.error("Error configurando modelo")
            sys.exit(1)
    else:
        logger.error("Error instalando dependencias")
        sys.exit(1)
