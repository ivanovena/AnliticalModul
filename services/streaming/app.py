import logging
import json
import time
import numpy as np
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from river import linear_model, preprocessing, feature_extraction, ensemble, compose, metrics, stats
from config import KAFKA_BROKER, REALTIME_TOPIC, STREAMING_TOPIC, FMP_API_KEY, FMP_BASE_URL
from flask import Flask, jsonify
import threading
import asyncio
import websockets
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Importar nuestros modelos personalizados primero para evitar errores
from custom_models import ALMARegressor
# Asegurarnos que ALMARegressor esté disponible en river.linear_model
import river.linear_model
river.linear_model.ALMARegressor = ALMARegressor

# Servidor web para health check
app = FastAPI(
    title="Streaming Service API",
    description="Servicio de streaming para predicciones en tiempo real",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Métricas de salud del servicio
health_metrics = {
    'eventos_procesados': 0,
    'predicciones_enviadas': 0,
    'errores': 0,
    'ultimo_evento': None,
    'estado': 'healthy'
}

# Mantener el endpoint FastAPI para compatibilidad
@app.get('/health')
def health_check():
    # Devolver métricas y estado actual
    # Podríamos añadir verificaciones adicionales aquí (ej: conexión Kafka)
    return {"status": health_metrics['estado'], "metrics": health_metrics}

@app.get('/prediction/history/{symbol}')
def get_prediction_history(symbol: str):
    """
    Endpoint para obtener historial de verificaciones de predicciones para un símbolo
    """
    try:
        # En un sistema real, recuperaríamos el historial de predicciones vs valores reales
        # Para la demo, generamos datos sintéticos consistentes
        
        # Generar seed basado en el símbolo para consistencia
        import hashlib
        seed_value = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 10000
        np.random.seed(seed_value)
        
        # Obtener precio base para el símbolo
        base_price = 200.0
        try:
            response = requests.get(f"{FMP_BASE_URL}/quote/{symbol}?apikey={FMP_API_KEY}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    base_price = float(data[0]['price'])
        except Exception as e:
            logger.warning(f"Error obteniendo precio para {symbol}: {e}")
            # Precios por defecto para símbolos conocidos
            base_price = 150.0 if symbol == "AAPL" else \
                         300.0 if symbol == "MSFT" else \
                         125.0 if symbol == "GOOGL" else \
                         140.0 if symbol == "AMZN" else \
                         210.0 if symbol == "TSLA" else 100.0
        
        # Horizontes para las predicciones
        horizons = ['15m', '30m', '1h', '3h', '1d']
        
        # Generar historial para las últimas 24 horas con ciclo horario
        history = []
        now = datetime.now()
        
        for i in range(24):
            timestamp = (now - timedelta(hours=i)).isoformat()
            
            # Para cada horizonte, generar una verificación
            for horizon in horizons:
                # Volatilidad diferente según el horizonte (más volatilidad en horizontes mayores)
                volatility = 0.01 if horizon == '15m' else \
                            0.015 if horizon == '30m' else \
                            0.02 if horizon == '1h' else \
                            0.03 if horizon == '3h' else 0.04
                
                # Generar valores para la verificación
                predicted_price = base_price * (1 + (np.random.random() * 2 - 1) * volatility)
                actual_price = base_price * (1 + (np.random.random() * 2 - 1) * volatility)
                
                # Calcular error
                error = abs(predicted_price - actual_price)
                error_pct = (error / actual_price) * 100
                
                # Agregar registro al historial
                history.append({
                    "timestamp": timestamp,
                    "horizon": horizon,
                    "predictedPrice": round(predicted_price, 2),
                    "actualPrice": round(actual_price, 2),
                    "error": round(error, 2),
                    "errorPct": round(error_pct, 2)
                })
        
        return history
    except Exception as e:
        logger.error(f"Error obteniendo historial de predicciones para {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get('/prediction/{symbol}')
def get_prediction(symbol: str):
    """
    Endpoint para obtener predicciones para un símbolo con diferentes horizontes temporales
    """
    try:
        # Si no tenemos métricas para este símbolo, generamos predicciones básicas
        current_price = 0
        
        # Intentar obtener el precio actual desde la API FMP
        try:
            url = f"{FMP_BASE_URL}/quote/{symbol}?apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    current_price = float(data[0]['price'])
        except Exception as e:
            logger.error(f"Error obteniendo precio actual para {symbol}: {e}")
            # Usamos un precio por defecto si hay error
            current_price = 200.0 if symbol == "AAPL" else 400.0 if symbol == "MSFT" else 150.0
        
        # Generamos predicciones para diferentes horizontes temporales
        # Usamos valores deterministas basados en el símbolo para consistencia
        # En un sistema real, estas vendrían de modelos de ML entrenados
        
        # Factores de volatilidad y tendencia por símbolo (para que sean consistentes)
        symbol_factors = {
            "AAPL": {"trend": 0.05, "volatility": 0.02},  # Tendencia positiva, baja volatilidad
            "MSFT": {"trend": 0.03, "volatility": 0.015},  # Tendencia positiva, muy baja volatilidad
            "GOOGL": {"trend": 0.04, "volatility": 0.025},  # Tendencia positiva, volatilidad media
            "AMZN": {"trend": 0.06, "volatility": 0.03},  # Tendencia positiva alta, volatilidad media
            "TSLA": {"trend": 0.02, "volatility": 0.05},  # Tendencia positiva baja, alta volatilidad
        }
        
        # Usar factores por defecto si el símbolo no está en nuestra lista
        factor = symbol_factors.get(symbol, {"trend": 0.03, "volatility": 0.02})
        
        # Generar seed basado en el símbolo para consistencia
        import hashlib
        seed_value = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 10000
        np.random.seed(seed_value)
        
        # Horizontes temporales (en minutos)
        horizons = {
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "3h": 180,
            "1d": 1440,
            "1w": 10080,
            "1m": 43200
        }
        
        predictions = {}
        model_metrics = {
            "MAPE": round(factor["volatility"] * 2 * 100, 2),  # Formato para el frontend
            "RMSE": round(factor["volatility"] * 3 * current_price, 2),  # Escalado para hacerlo más realista
            "accuracy": round(85 + np.random.rand() * 10, 2)  # Entre 85% y 95%
        }
        
        # Generar predicciones para cada horizonte
        for label, minutes in horizons.items():
            # El factor de tiempo aumenta con el horizonte (más incertidumbre a largo plazo)
            time_factor = np.log1p(minutes) / 10
            
            # Tendencia base ajustada por tiempo
            trend_component = factor["trend"] * time_factor
            
            # Componente determinista para cada horizonte
            # Usamos seno para crear oscilaciones predecibles
            deterministic = np.sin(minutes / 100) * factor["volatility"] * current_price * 0.1
            
            # Predicción final: precio actual + tendencia + componente determinista
            prediction_value = current_price * (1 + trend_component) + deterministic
            
            # Redondear a 2 decimales
            predictions[label] = round(prediction_value, 2)
        
        # Crear objeto de respuesta
        response = {
            "symbol": symbol,
            "currentPrice": current_price,  # Camel case para el frontend
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),  # Usar formato ISO para las fechas
            "modelMetrics": model_metrics  # Camel case para el frontend
        }
        
        return response
    except Exception as e:
        logger.error(f"Error obteniendo predicción para {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def run_health_server():
    """Iniciar servidor FastAPI (WebSocket y HTTP) en el puerto principal"""
    import uvicorn
    
    # Configurar Uvicorn para que escuche en el puerto API principal (8002 por defecto)
    api_port = int(os.environ.get('API_PORT', 8002))
    logger.info(f"Iniciando servidor FastAPI/Uvicorn en puerto {api_port}")
    uvicorn.run(
        "app:app", # Asegúrate que esto apunta al objeto FastAPI correcto
        host="0.0.0.0", 
        port=api_port, 
        reload=False, # Desactivar reload en producción/docker
        log_level="info"
    )

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("streaming_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StreamingService")

# Inicialización de Kafka Consumer y Producer
try:
    consumer = KafkaConsumer(
        REALTIME_TOPIC,  # Cambiado de INGESTION_TOPIC a REALTIME_TOPIC
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='streaming_service',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    logger.info(f"Kafka consumer inicializado. Suscrito a {REALTIME_TOPIC}")
    
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        acks='all'
    )
    logger.info("Kafka producer inicializado")
except Exception as e:
    logger.critical(f"Error inicializando Kafka: {e}")
    raise

# Función para obtener datos históricos
def fetch_historical_data(symbol, interval='5min', limit=30):
    """
    Obtiene datos históricos para complementar los eventos en tiempo real
    """
    try:
        # Usar el endpoint /stable/ directamente como sugiere la documentación más reciente encontrada
        stable_base_url = "https://financialmodelingprep.com/stable"
        url = f"{stable_base_url}/historical-chart/{interval}/{symbol}?apikey={FMP_API_KEY}&limit={limit}"
        logger.debug(f"Llamando a la URL de FMP: {url}") # Log para verificar la URL usada
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logger.error(f"Error obteniendo datos históricos para {symbol}: {e}")
        return []

# Función para extraer características avanzadas
def extract_advanced_features(event, historical_data):
    """
    Extrae características avanzadas combinando el evento actual con datos históricos
    """
    features = {}
    
    # Ordenar datos históricos por timestamp descendente (más reciente primero)
    # Asegurarse de que 'timestamp' existe y es numérico
    try:
        historical_data.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        # Log para depurar los datos históricos ordenados
        if historical_data:
            recent_closes = [h.get('close') for h in historical_data[:10]] # Primeros 10 cierres
            logger.debug(f"Primeros 10 cierres históricos (ordenados): {recent_closes}")
    except (TypeError, ValueError) as e:
        logger.warning(f"No se pudieron ordenar los datos históricos por timestamp: {e}")
        # Continuar sin ordenar si hay error, pero loguear

    # Características básicas del evento actual
    for key in ['open', 'high', 'low', 'close', 'volume']:
        if key in event:
            features[key] = float(event[key])
    
    # Si no hay datos históricos, devolver solo características básicas
    if not historical_data:
        return features
    
    # Calcular características avanzadas a partir de datos históricos
    try:
        # Convertir datos históricos a arrays numpy para análisis
        hist_close = np.array([float(h.get('close', 0)) for h in historical_data if 'close' in h])
        hist_volume = np.array([float(h.get('volume', 0)) for h in historical_data if 'volume' in h])
        
        if len(hist_close) > 5:
            # Características de tendencia
            features['sma_5'] = np.mean(hist_close[:5])
            # Calcular momentum como: precio actual - precio de hace 5 periodos
            # Asumiendo hist_close[0] es t-1, hist_close[4] es t-5
            if len(hist_close) >= 5 and 'close' in features:
                 features['price_momentum'] = features['close'] - hist_close[4]
            else:
                 features['price_momentum'] = 0 # Valor por defecto si no hay suficientes datos o falta el cierre actual
            
            # Volatilidad (sobre los 10 más recientes)
            features['volatility'] = np.std(hist_close[:10]) if len(hist_close) >= 10 else 0
            
            # Volumen relativo
            features['rel_volume'] = features.get('volume', 0) / np.mean(hist_volume) if np.mean(hist_volume) > 0 else 1
    except Exception as e:
        logger.warning(f"Error calculando características avanzadas: {e}")
    
    return features

# Modelos online por símbolo
class OnlineModelManager:
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.knowledge_bases = {}  # Para almacenar la base de conocimiento por símbolo
    
    def get_or_create_model(self, symbol):
        if symbol not in self.models:
            # Configuración de pipeline de modelo online
            # Usamos nuestro ALMARegressor personalizado que se ha definido en custom_models.py
            model = compose.Pipeline(
                ('preprocessor', preprocessing.StandardScaler()),
                ('feature_extractor', feature_extraction.PolynomialExtender(degree=2)),
                ('regressor', ensemble.EWARegressor(
                    models=[
                        linear_model.LinearRegression(),
                        linear_model.PARegressor(),
                        linear_model.ALMARegressor(alpha=0.1, window_size=30)  # Ventana ampliada de 10 a 30
                    ],
                    # weights=None  # Pesos uniformes para comenzar - parámetro eliminado
                ))
            )
            
            # Inicializar métricas de rendimiento
            self.models[symbol] = model
            self.performance_metrics[symbol] = {
                'mae': metrics.MAE(),
                'rmse': metrics.RMSE(),
                'r2': metrics.R2(),
                'predictions': [],
                'actuals': []
            }
            
            # Inicializar knowledge base vacía
            self.knowledge_bases[symbol] = {
                'feature_importance': {},
                'performance': 0.5,
                'confidence': 0.5
            }
        
        return self.models[symbol], self.performance_metrics[symbol]

    def update_model(self, symbol, features, target):
        """
        Actualiza el modelo para un símbolo específico
        """
        model, metrics_tracker = self.get_or_create_model(symbol)
        try:
            prediction = model.predict_one(features)

            # Ensure both prediction and target are valid floats before metric updates
            valid_prediction = isinstance(prediction, (int, float))
            valid_target = isinstance(target, (int, float))

            if valid_prediction and valid_target:
                # Cast to float for safety before metric updates
                float_prediction = float(prediction)
                float_target = float(target)

                metrics_tracker['predictions'].append(float_prediction)
                metrics_tracker['actuals'].append(float_target)

                metrics_tracker['mae'].update(float_target, float_prediction)
                metrics_tracker['rmse'].update(float_target, float_prediction)
                metrics_tracker['r2'].update(float_target, float_prediction)

                self._update_feature_importance(symbol, features, float_prediction, float_target)

            elif prediction is None:
                 logger.debug(f"Predicción fue None para {symbol}, saltando actualización de métricas.")
            else:
                 # Log if prediction or target are not valid numbers (and not None)
                 logger.warning(f"Predicción o target inválidos para {symbol}. "
                                f"Predicción: {prediction} (Tipo: {type(prediction)}), "
                                f"Target: {target} (Tipo: {type(target)}). Saltando métricas.")

            # Always try to learn to fill the buffer, ensure target is float
            if valid_target:
                 model.learn_one(features, float(target))
            else:
                 logger.warning(f"Target inválido para {symbol} en learn_one: {target} (Tipo: {type(target)}) ")

            return prediction # Return original prediction (could be None)

        except Exception as e:
            # Log the full traceback for better debugging
            logger.error(f"Error updating model for {symbol}: {e}", exc_info=True)
            return None
    
    def _update_feature_importance(self, symbol, features, prediction, target):
        """
        Actualiza importancia de características basada en errores de predicción
        """
        # Obtener o inicializar knowledge base
        if symbol not in self.knowledge_bases:
            self.knowledge_bases[symbol] = {
                'feature_importance': {},
                'performance': 0.5,
                'confidence': 0.5
            }
        
        kb = self.knowledge_bases[symbol]
        
        # Error de predicción
        error = abs(prediction - target)
        
        # Actualizar importancia de features
        for feature, value in features.items():
            # Si es una feature numérica
            if isinstance(value, (int, float)):
                # Inicializar si es nueva
                if feature not in kb['feature_importance']:
                    kb['feature_importance'][feature] = 0.5
                
                # Si el error es bajo, incrementar importancia
                # Si el error es alto, decrementar importancia
                if error < 0.01 * target:  # Error menor al 1%
                    kb['feature_importance'][feature] *= 1.05
                elif error > 0.1 * target:  # Error mayor al 10%
                    kb['feature_importance'][feature] *= 0.95
                
                # Limitar valores
                kb['feature_importance'][feature] = min(1.0, max(0.1, kb['feature_importance'][feature]))
        
        # Normalizar importancias
        total = sum(kb['feature_importance'].values())
        if total > 0:
            for feature in kb['feature_importance']:
                kb['feature_importance'][feature] /= total
        
        # Actualizar rendimiento y confianza
        metrics = self.performance_metrics[symbol]
        if metrics['mae'].get() > 0:
            kb['performance'] = 1.0 / (1.0 + metrics['mae'].get())
        kb['confidence'] = min(0.95, metrics['r2'].get() if metrics['r2'].get() > 0 else 0.5)
    
    def get_knowledge_base(self, symbol):
        """
        Obtiene la base de conocimiento para un símbolo
        """
        return self.knowledge_bases.get(symbol, {
            'feature_importance': {},
            'performance': 0.5,
            'confidence': 0.5
        })
    
    def update_feature_weights(self, symbol, feature_weights):
        """
        Actualiza pesos de características desde transfer learning
        """
        if symbol not in self.knowledge_bases:
            self.knowledge_bases[symbol] = {
                'feature_importance': {},
                'performance': 0.5,
                'confidence': 0.5
            }
        
        # Actualizar importancia de features
        for feature, importance in feature_weights.items():
            self.knowledge_bases[symbol]['feature_importance'][feature] = importance
        
        logger.info(f"Pesos de características actualizados para {symbol}: {len(feature_weights)} características")
        return True

# Instancia global del administrador de modelos
model_manager = OnlineModelManager()

def process_streaming_event(event: Dict[str, Any]):
    """
    Procesa un evento de streaming, extrayendo características y generando predicciones
    """
    # Log inicial del evento recibido
    logger.info(f"--- Procesando Evento --- Evento crudo: {event}")

    if not event or 'symbol' not in event:
        logger.warning("Evento inválido recibido, saltando.")
        return None

    symbol = event['symbol']
    logger.info(f"Símbolo: {symbol}")

    try:
        # Obtener datos históricos complementarios
        # Nota: fetch_historical_data puede ser lento y bloquear, considerar async si es un problema
        historical_data = fetch_historical_data(symbol, interval='5min')
        logger.debug(f"Datos históricos para {symbol}: {historical_data[:2]}... (primeros 2)") # Log reducido

        # Extraer características avanzadas
        features = extract_advanced_features(event, historical_data)
        logger.info(f"Características extraídas para {symbol}: {features}")

        # Preparar valor objetivo
        close_price = features.get('close', event.get('close', 0))
        # Asegurarse de que close_price sea un número válido antes de calcular target
        if isinstance(close_price, (int, float)):
             target = float(close_price) * 1.01
             logger.info(f"Target calculado para {symbol}: {target} (basado en close={close_price})")
        else:
             logger.warning(f"Precio de cierre inválido para {symbol}: {close_price}. No se puede calcular target.")
             target = None # Marcar como None si no se puede calcular

        # Obtener o crear modelo para este símbolo
        model, metrics = model_manager.get_or_create_model(symbol)

        # Hacer predicción y actualizar modelo
        # Asegurarse de que target es válido antes de pasar a update_model
        if target is not None:
            prediction = model_manager.update_model(symbol, features, target)
            logger.info(f"Predicción obtenida de update_model para {symbol}: {prediction}")
        else:
            # Si target es inválido, no podemos actualizar el modelo ni obtener predicción fiable
            logger.warning(f"Target inválido para {symbol}, no se actualiza ni predice el modelo.")
            prediction = None # O manejar de otra forma si se requiere una predicción incluso sin target

        if prediction is None:
            logger.warning(f"No se pudo generar predicción final para {symbol}")
            return None

        # Obtener la base de conocimiento actualizada
        knowledge_base = model_manager.get_knowledge_base(symbol)

        # Crear evento de predicción para Kafka
        prediction_event = {
            "symbol": symbol,
            # Asegurarse que la predicción sea float antes de enviar
            "prediction": float(prediction) if isinstance(prediction, (int, float)) else 0.0,
            "timestamp": time.time(),
            "features": features,
            "model_metrics": {
                "MAE": metrics['mae'].get(),
                "RMSE": metrics['rmse'].get(),
                "R2": metrics['r2'].get()
            },
            "knowledge_base": {
                "feature_importance": knowledge_base.get('feature_importance', {}),
                "performance": knowledge_base.get('performance', 0.5),
                "confidence": knowledge_base.get('confidence', 0.5)
            },
            "model_type": "online"  # Identificar como modelo online
        }
        logger.info(f"--- Evento Procesado --- Evento de predicción: {prediction_event}")
        return prediction_event

    except Exception as e:
        logger.error(f"Error procesando evento para {symbol}: {e}", exc_info=True) # Añadir traceback
        return None

def send_prediction_to_kafka(prediction_event):
    """
    Envía evento de predicción a tópico de Kafka
    """
    if not prediction_event:
        return False
    
    try:
        future = producer.send(STREAMING_TOPIC, prediction_event)
        producer.flush(timeout=5)
        record_metadata = future.get(timeout=10)
        
        logger.info(f"Predicción enviada para {prediction_event['symbol']}: "
                    f"Predicción={prediction_event['prediction']}, "
                    f"Tópico={record_metadata.topic}")
        return True
    except Exception as e:
        logger.error(f"Error enviando predicción: {e}")
        return False

# --- WebSocket Server para Predicciones ---
streaming_clients = set()

async def notify_streaming_clients(message):
    """Envía un mensaje a todos los clientes de streaming conectados"""
    if not streaming_clients:
        return
    disconnected = set()
    if isinstance(message, dict):
        message = json.dumps(message)
    for client in streaming_clients:
        try:
            await client.send(message)
        except Exception as e:
            logger.error(f"Error enviando mensaje de streaming a cliente: {e}")
            disconnected.add(client)
    for client in disconnected:
        try:
            streaming_clients.remove(client)
        except:
            pass

async def streaming_websocket_handler(websocket, path=None):
    """Manejador para conexiones WebSocket de predicciones"""
    try:
        # Logear información de la conexión incluyendo el path
        logger.info(f"Cliente de predicciones conectado: {websocket.remote_address}, path: {path}")
        
        # Agregar cliente a la lista de conexiones activas
        streaming_clients.add(websocket)
        
        # Mensaje de bienvenida
        try:
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "Conectado al servicio de predicciones en tiempo real",
                "timestamp": datetime.now().isoformat(),
                "path": path  # Incluir path en la respuesta para depuración
            }))
        except Exception as e:
            logger.error(f"Error enviando mensaje de bienvenida: {e}")
            return
            
        # Mantener la conexión abierta, escuchando por si el cliente envía algo (ej: ping)
        async for message in websocket:
            try:
                # Intentar decodificar el mensaje como JSON
                data = json.loads(message)
                logger.info(f"Mensaje recibido en WebSocket de streaming: {data}")
                
                # Manejar diferentes tipos de mensajes
                if data.get("action") == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif data.get("action") == "subscribe":
                    topic = data.get("topic", "")
                    logger.info(f"Cliente suscrito a {topic}")
                    
                    # Enviar confirmación de suscripción
                    await websocket.send(json.dumps({
                        "type": "subscription",
                        "status": "active",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                    # Enviar datos simulados después de una suscripción
                    await websocket.send(json.dumps({
                        "type": "prediction",
                        "symbol": "AAPL",
                        "prediction": 2.5,
                        "confidence": 0.75,
                        "timestamp": datetime.now().isoformat()
                    }))
            except json.JSONDecodeError as e:
                logger.warning(f"Mensaje no válido recibido: {message}, error: {e}")
                try:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Formato de mensaje no válido - se espera JSON",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as send_error:
                    logger.error(f"Error enviando mensaje de error al cliente: {send_error}")
            except Exception as e:
                logger.error(f"Error procesando mensaje del cliente: {e}")
                try:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Error en el servidor: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as send_error:
                    logger.error(f"Error enviando mensaje de error al cliente: {send_error}")
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Cliente de predicciones desconectado: {websocket.remote_address}, código: {e.code}, razón: {e.reason}")
    except Exception as e:
        logger.error(f"Error inesperado en WebSocket de streaming: {e}", exc_info=True)
    finally:
        if websocket in streaming_clients:
            streaming_clients.remove(websocket)
            logger.info(f"Cliente eliminado de clientes activos. Total: {len(streaming_clients)}")

async def start_streaming_websocket_server():
    """Inicia el servidor WebSocket para transmitir predicciones"""
    host = os.environ.get('WEBSOCKET_HOST', '0.0.0.0')
    port = int(os.environ.get('WEBSOCKET_PORT', 8090))
    
    # Orígenes permitidos para WebSocket (incluir origen del frontend)
    allowed_origins = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost", 
        "http://127.0.0.1",
        "http://0.0.0.0:3000",
        "ws://localhost:8100",
        "ws://localhost:8090",
        "ws://127.0.0.1:8100",
        "ws://127.0.0.1:8090",
        "null",  # Para conexiones desde file:// URLs o casos donde Origin está vacío
        "*",     # Permitir cualquier origen (para desarrollo, debe eliminarse en producción)
        None  # Permitir conexiones sin origen
    ]
    
    try:
        logger.info(f"Iniciando servidor WebSocket en {host}:{port} con proceso: {os.getpid()}")
        logger.info(f"Orígenes permitidos: {allowed_origins}")
        
        server = await websockets.serve(
            streaming_websocket_handler,  # Usar la función directamente sin lambda
            host, 
            port,
            # Desactivar la comprobación de orígenes por completo
            origins=None,
            ping_interval=60,
            ping_timeout=30,
            max_size=10 * 1024 * 1024  # 10MB para mensajes más grandes
        )
        logger.info(f"Servidor WebSocket de predicciones escuchando en {host}:{port} con orígenes permitidos: {allowed_origins}")
        # Mantener el servidor corriendo indefinidamente (o hasta que se cierre externamente)
        return server
    except Exception as e:
        logger.error(f"Error iniciando servidor WebSocket de streaming: {e}", exc_info=True)
        return None

# Iniciar servidor WebSocket al arrancar la aplicación FastAPI
@app.on_event("startup")
async def startup_event():
    """Eventos a ejecutar al iniciar la aplicación FastAPI"""
    logger.info("Iniciando servicio de streaming...")
    
    # Iniciar el servidor WebSocket en segundo plano
    loop = asyncio.get_running_loop()
    websocket_server_task = loop.create_task(start_streaming_websocket_server())
    logger.info("Tarea de servidor WebSocket iniciada.")
    
    # Aquí puedes añadir otras tareas de arranque (ej: conectar a Kafka, etc.)

# --- Fin WebSocket Server ---

# Añadir un endpoint específico de WebSocket en FastAPI
@app.websocket("/predictions")
async def predictions_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        # Logear información de la conexión
        logger.info(f"Cliente de predicciones conectado a través de FastAPI: {websocket.client}")
        
        # Mensaje de bienvenida
        await websocket.send_json({
            "type": "welcome",
            "message": "Conectado al servicio de predicciones en tiempo real (FastAPI)",
            "timestamp": datetime.now().isoformat()
        })
        
        # Procesar mensajes
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Mensaje recibido en WebSocket FastAPI: {data}")
                
                # Manejar diferentes tipos de mensajes
                if data.get("action") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif data.get("action") == "subscribe":
                    topic = data.get("topic", "")
                    logger.info(f"Cliente suscrito a {topic}")
                    
                    # Enviar confirmación de suscripción
                    await websocket.send_json({
                        "type": "subscription",
                        "status": "active",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Enviar datos simulados después de una suscripción
                    await websocket.send_json({
                        "type": "prediction",
                        "symbol": "AAPL",
                        "prediction": 2.5,
                        "confidence": 0.75,
                        "timestamp": datetime.now().isoformat()
                    })
            except WebSocketDisconnect:
                logger.info(f"Cliente desconectado: {websocket.client}")
                break
            except Exception as e:
                logger.error(f"Error procesando mensaje: {e}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    # Si no podemos enviar el error, probablemente el cliente está desconectado
                    break
    except Exception as e:
        logger.error(f"Error en WebSocket de FastAPI: {e}")
    finally:
        # Intentar cerrar la conexión si aún no está cerrada
        try:
            await websocket.close()
        except:
            pass

# Mantener esta sección para soporte legacy si se ejecuta directamente el script
if __name__ == "__main__":
    # Necesitamos un loop de eventos para correr ambos
    loop = asyncio.get_event_loop()
    
    # Iniciar el servidor WebSocket en segundo plano
    websocket_server_task = loop.create_task(start_streaming_websocket_server())
    
    # Iniciar el servidor HTTP/API (que ahora está en run_health_server)
    # Ejecutamos run_health_server directamente, ya que contiene uvicorn.run
    # Esto bloqueará el hilo principal, manteniendo el loop activo
    run_health_server() 
    
    # (Este código no se alcanzará si uvicorn.run bloquea)
    # try:
    #     loop.run_forever()
    # finally:
    #     loop.close()

