#!/bin/bash
set -e

# Configuración de variables de entorno con valores por defecto
KAFKA_HOST=${KAFKA_HOST:-kafka}
KAFKA_PORT=${KAFKA_PORT:-9092}
MAX_RETRIES=${MAX_RETRIES:-30}
INITIAL_BACKOFF=${INITIAL_BACKOFF:-2}

echo "Iniciando servicio de streaming..."
echo "KAFKA_HOST=$KAFKA_HOST"
echo "KAFKA_PORT=$KAFKA_PORT"

# Verificar e instalar dependencias necesarias
echo "Verificando dependencias..."
pip install --no-cache-dir -r requirements.txt

# Función para verificar si Kafka está disponible
check_kafka() {
    # Intentar conectarse a Kafka usando el cliente kafka-python
    python -c "
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import time
import socket

def is_port_open(host, port, timeout=5):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

try:
    # Primero comprobar si el puerto está abierto
    if not is_port_open('$KAFKA_HOST', $KAFKA_PORT):
        print('Puerto de Kafka no accesible')
        exit(1)
    
    # Luego intentar crear un productor
    producer = KafkaProducer(bootstrap_servers=['$KAFKA_HOST:$KAFKA_PORT'], 
                            api_version=(0, 10, 1),
                            request_timeout_ms=5000,
                            security_protocol='PLAINTEXT')
    producer.close()
    print('Kafka está disponible')
    exit(0)
except NoBrokersAvailable:
    print('No se puede conectar a Kafka: NoBrokersAvailable')
    exit(1)
except Exception as e:
    print(f'Error conectando a Kafka: {e}')
    exit(1)
"
    return $?
}

# Esperar a que Kafka esté disponible usando backoff exponencial
echo "Esperando a que Kafka esté disponible en $KAFKA_HOST:$KAFKA_PORT"
backoff=$INITIAL_BACKOFF
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    if check_kafka; then
        echo "Kafka está disponible después de $retry_count intentos"
        break
    fi
    
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $MAX_RETRIES ]; then
        echo "Error: No se pudo conectar a Kafka después de $MAX_RETRIES intentos"
        echo "Procediendo de todos modos para permitir que el servicio funcione en modo de respaldo"
    else
        echo "Reintentando en $backoff segundos (intento $retry_count de $MAX_RETRIES)..."
        sleep $backoff
        backoff=$((backoff * 2))
    fi
done

# Crear archivos necesarios si no existen
if [ ! -f "config.py" ]; then
    echo "Creando archivo config.py..."
    cat > config.py << EOL
# Configuración para el servicio de streaming
KAFKA_CONFIG = {
    'bootstrap_servers': ['$KAFKA_HOST:$KAFKA_PORT'],
    'api_version': (0, 10, 1),
    'client_id': 'streaming-service',
    'auto_offset_reset': 'latest',
    'security_protocol': 'PLAINTEXT'
}

# Configuración API
API_HOST = '0.0.0.0'
API_PORT = 8002
API_DEBUG = False
API_WORKERS = 4

# Configuración de tópicos Kafka
TOPICS = {
    'input': 'ingestion_events',
    'output': 'streaming_events',
    'batch_updates': 'batch_events'
}

# Configuración de modelos
MODEL_STORAGE_PATH = './models'
DEFAULT_PREDICTION_HORIZON = '1d'

# Configuración de logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configuración de caché
CACHE_EXPIRY = 300  # segundos

# Configuración de fallback cuando los servicios están caídos
FALLBACK_ENABLED = True
EOL
fi

# Crear modelo personalizado si no existe
if [ ! -f "custom_models.py" ]; then
    echo "Creando archivo custom_models.py..."
    cat > custom_models.py << EOL
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np

class ALMARegressor(BaseEstimator, RegressorMixin):
    """
    Arnaud Legoux Moving Average Regressor
    
    Un regresor personalizado que combina ALMA (Arnaud Legoux Moving Average)
    con una regresión lineal para predecir valores futuros.
    """
    def __init__(self, window=10, offset=0.85, sigma=6, trend_factor=0.3):
        self.window = window
        self.offset = offset
        self.sigma = sigma
        self.trend_factor = trend_factor
        self.linear_model = LinearRegression()
        
    def alma_filter(self, X):
        """Aplicar el filtro ALMA a los datos"""
        m = np.arange(self.window)
        s = self.window / self.sigma
        w = np.exp(-((m - self.offset * (self.window - 1)) ** 2) / (2 * s * s))
        w = w / np.sum(w)
        
        # Aplicar el filtro a cada característica
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[1]):
            # Aplicar convolución con los pesos
            for j in range(X.shape[0]):
                if j >= self.window - 1:
                    X_filtered[j, i] = np.sum(X[j - self.window + 1:j + 1, i] * w)
                else:
                    # Para las primeras filas donde no tenemos suficiente historia
                    X_filtered[j, i] = X[j, i]
        
        return X_filtered
        
    def fit(self, X, y):
        """Entrenar el modelo con datos filtrados y tendencia"""
        X_filtered = self.alma_filter(X)
        
        # Añadir componente de tendencia
        if X.shape[0] > 1:
            trend = np.arange(X.shape[0]).reshape(-1, 1) * self.trend_factor
            X_with_trend = np.hstack([X_filtered, trend])
        else:
            X_with_trend = X_filtered
        
        self.linear_model.fit(X_with_trend, y)
        return self
        
    def predict(self, X):
        """Predecir usando datos filtrados y tendencia"""
        X_filtered = self.alma_filter(X)
        
        # Añadir componente de tendencia
        if X.shape[0] > 1:
            trend = np.arange(X.shape[0]).reshape(-1, 1) * self.trend_factor
            X_with_trend = np.hstack([X_filtered, trend])
        else:
            X_with_trend = X_filtered
        
        return self.linear_model.predict(X_with_trend)
EOL
fi

# Crear archivo para la API de FastAPI si no existe
if [ ! -f "api.py" ]; then
    echo "Creando archivo api.py..."
    cat > api.py << EOL
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
import json
import asyncio
from datetime import datetime

from kafka_client import KafkaClient
from streaming_processor import StreamingProcessor
from feature_engineering import FeatureEngineer
from model_manager import ModelManager
from historical_data import HistoricalDataManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("streaming-api")

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Streaming API",
    description="API para obtener predicciones y datos en tiempo real",
    version="1.0.0",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos, inicializar procesador, etc.
kafka_client = KafkaClient()
feature_engineer = FeatureEngineer()
model_manager = ModelManager(feature_engineer)
streaming_processor = StreamingProcessor(kafka_client, model_manager, feature_engineer)
historical_data = HistoricalDataManager()

# Variables para mantener estado
is_processing = False
last_processed = {}
cached_predictions = {}

# Modelos de datos
class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predictions: Dict[str, float]
    features: Optional[Dict[str, float]] = None
    last_updated: str
    confidence: Optional[float] = None

class StatusResponse(BaseModel):
    status: str
    active_symbols: List[str]
    processing: bool
    last_processed: Dict[str, str]

# Endpoints de la API
@app.get("/")
async def root():
    return {"message": "Streaming API funcionando correctamente"}

@app.get("/health")
async def health():
    # Comprobar conexión a Kafka
    kafka_status = "healthy" if kafka_client.is_connected() else "degraded"
    # Comprobar modelos cargados
    models_status = "healthy" if model_manager.models_loaded() else "degraded"
    
    overall_status = "healthy" if kafka_status == "healthy" and models_status == "healthy" else "degraded"
    
    return {
        "status": overall_status,
        "kafka": kafka_status,
        "models": models_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/prediction/{symbol}", response_model=PredictionResponse)
async def get_prediction(symbol: str, include_features: bool = False):
    symbol = symbol.upper()
    try:
        # Intentar obtener predicción en tiempo real
        prediction = streaming_processor.get_latest_prediction(symbol)
        if not prediction:
            # Si no hay predicción en tiempo real, intentar con datos históricos
            prediction = model_manager.predict_from_historical(symbol)
            if not prediction:
                raise HTTPException(status_code=404, detail=f"No hay datos disponibles para {symbol}")
        
        # Formatear respuesta
        response = {
            "symbol": symbol,
            "current_price": prediction["current_price"],
            "predictions": prediction["predictions"],
            "last_updated": prediction["timestamp"],
            "confidence": prediction.get("confidence", 0.8)
        }
        
        # Incluir características si se solicita
        if include_features and "features" in prediction:
            response["features"] = prediction["features"]
            
        return response
    except Exception as e:
        logger.error(f"Error al generar predicción para {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la predicción: {str(e)}")

@app.get("/prediction/all", response_model=Dict[str, PredictionResponse])
async def get_all_predictions(include_features: bool = False):
    try:
        predictions = streaming_processor.get_all_predictions()
        
        formatted_predictions = {}
        for symbol, pred in predictions.items():
            formatted_pred = {
                "symbol": symbol,
                "current_price": pred["current_price"],
                "predictions": pred["predictions"],
                "last_updated": pred["timestamp"],
                "confidence": pred.get("confidence", 0.8)
            }
            
            if include_features and "features" in pred:
                formatted_pred["features"] = pred["features"]
                
            formatted_predictions[symbol] = formatted_pred
            
        return formatted_predictions
    except Exception as e:
        logger.error(f"Error al obtener todas las predicciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener predicciones: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    try:
        active_symbols = streaming_processor.get_active_symbols()
        
        formatted_last_processed = {}
        for symbol, timestamp in last_processed.items():
            formatted_last_processed[symbol] = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
            
        return {
            "status": "online",
            "active_symbols": active_symbols,
            "processing": is_processing,
            "last_processed": formatted_last_processed
        }
    except Exception as e:
        logger.error(f"Error al obtener estado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estado: {str(e)}")

@app.post("/process", status_code=202)
async def trigger_processing(background_tasks: BackgroundTasks):
    global is_processing
    
    if is_processing:
        return {"message": "Ya hay un proceso en ejecución"}
    
    async def process_data():
        global is_processing, last_processed
        is_processing = True
        try:
            # Procesar datos y actualizar predicciones
            await streaming_processor.process_batch()
            # Actualizar timestamp de último procesamiento
            for symbol in streaming_processor.get_active_symbols():
                last_processed[symbol] = datetime.now()
        except Exception as e:
            logger.error(f"Error en procesamiento: {str(e)}")
        finally:
            is_processing = False
    
    background_tasks.add_task(process_data)
    return {"message": "Procesamiento iniciado en segundo plano"}

@app.get("/historical/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1d", limit: int = 30):
    try:
        symbol = symbol.upper()
        data = historical_data.get_data(symbol, timeframe, limit)
        if not data:
            raise HTTPException(status_code=404, detail=f"No hay datos históricos para {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error al obtener datos históricos para {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener datos históricos: {str(e)}")

# Iniciar el procesador de streaming en segundo plano al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    # Cargar modelos
    await model_manager.load_models()
    
    # Iniciar consumidor de Kafka en segundo plano
    asyncio.create_task(streaming_processor.start_consuming())
    
    logger.info("Streaming API iniciada correctamente")

@app.on_event("shutdown")
async def shutdown_event():
    # Detener consumidor de Kafka
    await streaming_processor.stop_consuming()
    
    logger.info("Streaming API detenida correctamente")
EOL
fi

# Crear streaming_processor.py con gestión de errores mejorada
if [ ! -f "streaming_processor.py" ]; then
    echo "Creando archivo streaming_processor.py con gestión de errores mejorada..."
    cat > streaming_processor.py << EOL
import asyncio
import logging
import json
from datetime import datetime
import time
from typing import Dict, Any, List, Optional
import traceback

from kafka_client import KafkaClient
from model_manager import ModelManager
from feature_engineering import FeatureEngineer

logger = logging.getLogger("streaming-processor")

class StreamingProcessor:
    def __init__(self, kafka_client: KafkaClient, model_manager: ModelManager, feature_engineer: FeatureEngineer):
        self.kafka_client = kafka_client
        self.model_manager = model_manager
        self.feature_engineer = feature_engineer
        self.running = False
        self.consumer_task = None
        self.predictions = {}
        self.active_symbols = set()
        self.last_update = {}
        self.max_error_count = 5
        self.error_counts = {}
        self.retry_delays = {}

    async def start_consuming(self):
        """Iniciar el consumo de mensajes de Kafka en segundo plano"""
        if self.running:
            logger.warning("El procesador ya está en ejecución")
            return

        self.running = True
        self.consumer_task = asyncio.create_task(self._consume_loop())
        logger.info("Iniciado consumo de Kafka en segundo plano")

    async def stop_consuming(self):
        """Detener el consumo de mensajes de Kafka"""
        if not self.running:
            return

        self.running = False
        if self.consumer_task:
            try:
                self.consumer_task.cancel()
                await self.consumer_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error al detener consumidor: {str(e)}")
        
        logger.info("Detenido consumo de Kafka")

    async def _consume_loop(self):
        """Bucle principal para consumir mensajes de Kafka con manejo de errores mejorado"""
        while self.running:
            try:
                if not self.kafka_client.is_connected():
                    await self._reconnect_kafka()
                    continue

                # Consumir mensajes
                messages = await self.kafka_client.consume(timeout_ms=1000)
                
                if messages:
                    for message in messages:
                        try:
                            await self._process_message(message)
                        except Exception as e:
                            symbol = self._extract_symbol_from_message(message)
                            logger.error(f"Error procesando mensaje para {symbol}: {str(e)}")
                            traceback.print_exc()
                            self._increment_error_count(symbol)
                
                # Actualizar modelos con datos del batch si hay disponibles
                await self._check_model_updates()
                
                # Breve pausa para no sobrecargar el bucle
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error en bucle de consumo: {str(e)}")
                traceback.print_exc()
                await asyncio.sleep(5)  # Esperar antes de reintentar

    async def _reconnect_kafka(self):
        """Intentar reconectar a Kafka con backoff exponencial"""
        try:
            logger.info("Intentando reconectar a Kafka...")
            connected = await self.kafka_client.connect()
            
            if connected:
                logger.info("Reconectado a Kafka exitosamente")
                # Resetear contadores de error relacionados con Kafka
                for symbol in list(self.error_counts.keys()):
                    if "kafka" in self.error_counts.get(symbol, {}).get("reason", ""):
                        self.error_counts.pop(symbol, None)
                        self.retry_delays.pop(symbol, None)
            else:
                logger.warning("No se pudo reconectar a Kafka")
                await asyncio.sleep(5)  # Esperar antes de reintentar
        except Exception as e:
            logger.error(f"Error al reconectar a Kafka: {str(e)}")
            await asyncio.sleep(5)  # Esperar antes de reintentar

    def _extract_symbol_from_message(self, message: dict) -> str:
        """Extraer el símbolo de un mensaje"""
        try:
            if isinstance(message, dict) and "symbol" in message:
                return message["symbol"]
            elif isinstance(message, str):
                data = json.loads(message)
                if "symbol" in data:
                    return data["symbol"]
        except:
            pass
        return "unknown"

    def _increment_error_count(self, symbol: str, reason: str = "general"):
        """Incrementar contador de errores para un símbolo"""
        if symbol not in self.error_counts:
            self.error_counts[symbol] = {"count": 0, "reason": reason}
            self.retry_delays[symbol] = 1
        
        self.error_counts[symbol]["count"] += 1
        self.error_counts[symbol]["reason"] = reason
        
        # Si supera el máximo de errores, usar datos de respaldo
        if self.error_counts[symbol]["count"] >= self.max_error_count:
            logger.warning(f"Demasiados errores para {symbol}, usando datos de respaldo")
            # Implementar lógica de respaldo
            self._use_fallback_data(symbol)
            
            # Resetear contador después de aplicar respaldo, pero aumentar delay
            self.error_counts[symbol]["count"] = 0
            self.retry_delays[symbol] = min(self.retry_delays[symbol] * 2, 600)  # Max 10 minutos

    def _use_fallback_data(self, symbol: str):
        """Usar datos de respaldo cuando hay demasiados errores"""
        try:
            # Obtener predicción de respaldo del modelo
            fallback_prediction = self.model_manager.get_fallback_prediction(symbol)
            
            if fallback_prediction:
                fallback_prediction["is_fallback"] = True
                fallback_prediction["timestamp"] = datetime.now().isoformat()
                
                self.predictions[symbol] = fallback_prediction
                self.active_symbols.add(symbol)
                self.last_update[symbol] = time.time()
                
                logger.info(f"Aplicados datos de respaldo para {symbol}")
        except Exception as e:
            logger.error(f"Error al aplicar datos de respaldo para {symbol}: {str(e)}")

    async def _process_message(self, message: Dict[str, Any]):
        """Procesar un mensaje de Kafka"""
        if not message:
            return
            
        # Convertir mensaje a objeto si viene como string
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"Error al decodificar mensaje: {message[:100]}...")
                return
        
        # Extraer datos del mensaje
        symbol = message.get("symbol")
        if not symbol:
            logger.warning("Mensaje sin símbolo, ignorando")
            return
            
        # Registrar símbolo como activo
        self.active_symbols.add(symbol)
        
        # Extraer datos de precio y generar características
        price_data = message.get("data", {})
        if not price_data:
            logger.warning(f"Mensaje sin datos de precio para {symbol}, ignorando")
            return
            
        try:
            # Generar características para el modelo
            features = self.feature_engineer.generate_features(price_data)
            
            # Obtener predicción del modelo
            prediction = self.model_manager.predict(symbol, features)
            
            # Actualizar predicciones
            self.predictions[symbol] = {
                "symbol": symbol,
                "current_price": price_data.get("close", price_data.get("price", 0)),
                "predictions": prediction,
                "features": features,
                "timestamp": datetime.now().isoformat(),
                "is_fallback": False
            }
            
            # Actualizar timestamp
            self.last_update[symbol] = time.time()
            
            # Publicar predicción en Kafka
            await self.kafka_client.produce("streaming_events", {
                "symbol": symbol,
                "predictions": prediction,
                "timestamp": datetime.now().isoformat()
            })
            
            # Resetear contador de errores para este símbolo
            if symbol in self.error_counts:
                self.error_counts.pop(symbol)
                self.retry_delays.pop(symbol, None)
                
            logger.debug(f"Procesado mensaje para {symbol} y generada predicción")
            
        except Exception as e:
            logger.error(f"Error al procesar datos para {symbol}: {str(e)}")
            self._increment_error_count(symbol, reason=str(e))
            raise

    async def _check_model_updates(self):
        """Comprobar si hay actualizaciones de modelos desde batch"""
        try:
            model_messages = await self.kafka_client.consume("batch_events", timeout_ms=100)
            
            if model_messages:
                for message in model_messages:
                    try:
                        if isinstance(message, str):
                            message = json.loads(message)
                            
                        symbol = message.get("symbol")
                        model_type = message.get("model_type")
                        model_data = message.get("model_data")
                        
                        if symbol and model_type and model_data:
                            # Actualizar modelo en el gestor de modelos
                            await self.model_manager.update_model(symbol, model_type, model_data)
                            logger.info(f"Actualizado modelo {model_type} para {symbol}")
                    except Exception as e:
                        logger.error(f"Error al procesar actualización de modelo: {str(e)}")
        except Exception as e:
            logger.error(f"Error al comprobar actualizaciones de modelos: {str(e)}")

    async def process_batch(self):
        """Procesar un lote de datos para todos los símbolos activos"""
        for symbol in self.active_symbols:
            try:
                # Obtener datos históricos recientes
                historical_data = await self.model_manager.get_historical_data(symbol)
                
                if historical_data:
                    # Generar características
                    features = self.feature_engineer.generate_features_from_historical(historical_data)
                    
                    # Obtener predicción
                    prediction = self.model_manager.predict(symbol, features)
                    
                    # Actualizar predicciones
                    self.predictions[symbol] = {
                        "symbol": symbol,
                        "current_price": historical_data[-1].get("close", 0),
                        "predictions": prediction,
                        "features": features,
                        "timestamp": datetime.now().isoformat(),
                        "is_fallback": False
                    }
                    
                    # Actualizar timestamp
                    self.last_update[symbol] = time.time()
                    
                    logger.info(f"Procesado lote para {symbol}")
            except Exception as e:
                logger.error(f"Error al procesar lote para {symbol}: {str(e)}")
                self._increment_error_count(symbol, reason=f"batch: {str(e)}")

    def get_latest_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtener la última predicción para un símbolo"""
        return self.predictions.get(symbol)

    def get_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las predicciones"""
        return self.predictions

    def get_active_symbols(self) -> List[str]:
        """Obtener lista de símbolos activos"""
        return list(self.active_symbols)
EOL
fi

# Iniciar aplicación principal
echo "Iniciando aplicación principal..."
if [ -f "main.py" ]; then
    # Intentar ejecutar la aplicación con Uvicorn
    exec uvicorn main:app --host 0.0.0.0 --port 8002 --workers 4 --reload
else
    echo "Error: No se encontró el archivo main.py"
    # Crear un archivo main.py básico
    cat > main.py << EOL
from api import app

# Este archivo es el punto de entrada para uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
EOL
    
    # Ejecutar la aplicación recién creada
    exec uvicorn main:app --host 0.0.0.0 --port 8002 --workers 4 --reload
fi
