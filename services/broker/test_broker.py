#!/usr/bin/env python3
"""
Test script para verificar el funcionamiento del broker
"""
import requests
import json
import time
import logging
import sys
from typing import Dict, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("broker_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrokerTest")

# URL base del servicio
BASE_URL = "http://localhost:8001"

def test_health():
    """Test del endpoint de salud"""
    try:
        url = f"{BASE_URL}/health"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Health check exitoso: {data}")
            return True
        else:
            logger.error(f"Health check fallido: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return False

def test_chat(message: str) -> Dict[str, Any]:
    """
    Test del endpoint de chat
    
    Args:
        message: Mensaje para enviar al broker
        
    Returns:
        Respuesta del chat o diccionario vacío en caso de error
    """
    try:
        url = f"{BASE_URL}/chat"
        data = {"message": message}
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Chat exitoso: {message}")
            logger.info(f"Respuesta: {result['response'][:100]}...")
            return result
        else:
            logger.error(f"Chat fallido: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error en test de chat: {e}")
        return {}

def test_portfolio():
    """Test del endpoint de portafolio"""
    try:
        url = f"{BASE_URL}/portfolio"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Consulta de portafolio exitosa: {data}")
            return True
        else:
            logger.error(f"Consulta de portafolio fallida: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error en consulta de portafolio: {e}")
        return False

def test_market_data(symbol: str):
    """
    Test del endpoint de datos de mercado
    
    Args:
        symbol: Símbolo a consultar
    """
    try:
        url = f"{BASE_URL}/market-data/{symbol}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Consulta de datos de mercado exitosa para {symbol}")
            return True
        else:
            logger.error(f"Consulta de datos de mercado fallida: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error en consulta de datos de mercado: {e}")
        return False

def test_strategy(symbol: str):
    """
    Test del endpoint de estrategia
    
    Args:
        symbol: Símbolo a consultar
    """
    try:
        url = f"{BASE_URL}/strategy/{symbol}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Consulta de estrategia exitosa para {symbol}")
            return True
        else:
            logger.error(f"Consulta de estrategia fallida: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error en consulta de estrategia: {e}")
        return False

def main():
    """Ejecuta todos los tests"""
    logger.info("Iniciando tests del broker IA")
    
    # Esperar a que el servicio esté disponible
    max_retries = 5
    for i in range(max_retries):
        if test_health():
            break
        logger.info(f"Esperando a que el servicio esté disponible... ({i+1}/{max_retries})")
        time.sleep(5)
    else:
        logger.error("No se pudo conectar al servicio después de varios intentos")
        sys.exit(1)
    
    # Test de chat
    chat_messages = [
        "Hola, ¿qué puedes hacer?",
        "¿Qué estrategia recomiendas para AAPL?",
        "Muéstrame mi portafolio",
        "¿Cuáles son mis métricas de riesgo?",
        "Dame información de mercado para MSFT"
    ]
    
    conversation_id = None
    for message in chat_messages:
        result = test_chat(message)
        if result and 'conversation_id' in result:
            conversation_id = result['conversation_id']
            # Usar el mismo ID de conversación para mensajes sucesivos
            time.sleep(1)
    
    # Test de portafolio
    test_portfolio()
    
    # Test de datos de mercado
    for symbol in ["AAPL", "MSFT", "GOOG"]:
        test_market_data(symbol)
        test_strategy(symbol)
        time.sleep(1)
    
    logger.info("Tests completados")

if __name__ == "__main__":
    main()
