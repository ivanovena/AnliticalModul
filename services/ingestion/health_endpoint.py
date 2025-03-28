# Simplificado: Eliminar dependencia Flask
import logging
import threading
import time
import socket
import psycopg2
from kafka import KafkaProducer
from config import KAFKA_BROKER, DB_URI
from flask import Flask, jsonify

logger = logging.getLogger(__name__)

# Create a separate Flask app for health check
health_app = Flask(__name__)

@health_app.route('/health', methods=['GET'])
def health():
    status = check_health()
    if status["status"] == "healthy":
        return jsonify(status), 200
    else:
        return jsonify(status), 500

def check_health():
    """Función simple para verificar la salud de los servicios"""
    status = {"status": "healthy", "components": {}}
    
    # Verificar conexión a la base de datos
    try:
        db_params = DB_URI.replace('postgresql://', '').split('@')
        user_part = db_params[0].split(':')
        server_part = db_params[1].split('/')
        
        conn = psycopg2.connect(
            dbname=server_part[1],
            user=user_part[0],
            password=user_part[1],
            host=server_part[0].split(':')[0],
            port=server_part[0].split(':')[1] if ':' in server_part[0] else '5432'
        )
        conn.close()
        status["components"]["database"] = "healthy"
        logger.info("Database health check: healthy")
    except Exception as e:
        logger.error(f"Error en la conexión a la base de datos: {e}")
        status["components"]["database"] = "unhealthy"
        status["status"] = "unhealthy"
    
    # Verificar conexión a Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: str(v).encode('utf-8'),
            acks='all',
            retries=1,
            request_timeout_ms=1000
        )
        producer.send("test_health_topic", "test").get(timeout=1)
        producer.close()
        status["components"]["kafka"] = "healthy"
        logger.info("Kafka health check: healthy")
    except Exception as e:
        logger.error(f"Error en la conexión a Kafka: {e}")
        status["components"]["kafka"] = "unhealthy"
        status["status"] = "unhealthy"
    
    return status

def health_check_thread():
    """Thread para ejecutar chequeos periódicos"""
    while True:
        try:
            status = check_health()
            if status["status"] != "healthy":
                logger.warning(f"Health check failed: {status}")
            else:
                logger.info("Health check passed")
        except Exception as e:
            logger.error(f"Error en el health check: {e}")
        time.sleep(30)  # Chequeo cada 30 segundos

def start_health_server():
    """Iniciar el thread de health check y el servidor Flask en segundo plano"""
    # Start health check thread
    check_thread = threading.Thread(target=health_check_thread, daemon=True)
    check_thread.start()
    logger.info("Health check thread iniciado")
    
    # Start health server in background
    server_thread = threading.Thread(target=lambda: health_app.run(host='0.0.0.0', port=8080), daemon=True)
    server_thread.start()
    logger.info("Health server started on port 8080")
