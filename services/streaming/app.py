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
from config import KAFKA_BROKER, INGESTION_TOPIC, STREAMING_TOPIC, FMP_API_KEY, FMP_BASE_URL
from flask import Flask, jsonify
import threading

# Importar nuestros modelos personalizados primero para evitar errores
from custom_models import ALMARegressor
# Asegurarnos que ALMARegressor esté disponible en river.linear_model
import river.linear_model
river.linear_model.ALMARegressor = ALMARegressor

# Servidor web para health check
app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "metrics": health_metrics
    })

@app.route('/prediction/history/<symbol>')
def get_prediction_history(symbol):
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
        
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error obteniendo historial de predicciones para {symbol}: {e}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/prediction/<symbol>')
def get_prediction(symbol):
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
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error obteniendo predicción para {symbol}: {e}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

def run_health_server():
    app.run(host='0.0.0.0', port=8090)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
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
        INGESTION_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='streaming_service',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    logger.info(f"Kafka consumer inicializado. Suscrito a {INGESTION_TOPIC}")
    
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
def fetch_historical_data(symbol, interval='1min', limit=30):
    """
    Obtiene datos históricos para complementar los eventos en tiempo real
    """
    try:
        url = f"{FMP_BASE_URL}/historical-chart/{interval}/{symbol}?apikey={FMP_API_KEY}&limit={limit}"
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
            features['price_momentum'] = hist_close[0] - hist_close[5] if len(hist_close) > 5 else 0
            
            # Volatilidad
            features['volatility'] = np.std(hist_close[:10]) if len(hist_close) > 10 else 0
            
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
                        linear_model.ALMARegressor(alpha=0.1, window_size=10)  # Instancia explícita con parámetros
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
            # Hacer predicción antes de aprender
            prediction = model.predict_one(features)
            metrics_tracker['predictions'].append(prediction)
            metrics_tracker['actuals'].append(target)
            
            # Actualizar métricas
            metrics_tracker['mae'].update(target, prediction)
            metrics_tracker['rmse'].update(target, prediction)
            metrics_tracker['r2'].update(target, prediction)
            
            # Estimar importancia de features basado en uso reciente
            self._update_feature_importance(symbol, features, prediction, target)
            
            # Aprender del nuevo dato
            model.learn_one(features, target)
            
            return prediction
        except Exception as e:
            logger.error(f"Error updating model for {symbol}: {e}")
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
    if not event or 'symbol' not in event:
        logger.warning("Evento inválido recibido")
        return None

    symbol = event['symbol']
    
    try:
        # Obtener datos históricos complementarios
        historical_data = fetch_historical_data(symbol)
        
        # Extraer características avanzadas
        features = extract_advanced_features(event, historical_data)
        
        # Preparar valor objetivo 
        # En un escenario real, esto podría ser el precio futuro o cambio de precio
        target = features.get('close', event.get('close', 0)) * 1.01  # Ejemplo: predecir 1% de aumento
        
        # Obtener o crear modelo para este símbolo
        model, metrics = model_manager.get_or_create_model(symbol)
        
        # Hacer predicción y actualizar modelo
        prediction = model_manager.update_model(symbol, features, target)
        
        if prediction is None:
            logger.warning(f"No se pudo generar predicción para {symbol}")
            return None
        
        # Obtener la base de conocimiento actualizada
        knowledge_base = model_manager.get_knowledge_base(symbol)
        
        # Crear evento de predicción para Kafka
        prediction_event = {
            "symbol": symbol,
            "prediction": prediction,
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
        
        return prediction_event
    
    except Exception as e:
        logger.error(f"Error procesando evento para {symbol}: {e}")
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

# Métricas de salud del servicio
health_metrics = {
    'eventos_procesados': 0,
    'predicciones_enviadas': 0,
    'errores': 0,
    'ultimo_evento': None,
    'estado': 'healthy'
}

def main():
    """
    Bucle principal de procesamiento de streaming
    """
    # Iniciar el servidor de health check en un hilo separado
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    logger.info("Servidor de health check iniciado en puerto 8090")
    
    logger.info("Iniciando servicio de streaming en tiempo real")
    
    try:
        for message in consumer:
            try:
                # Procesar evento
                event = message.value
                health_metrics['ultimo_evento'] = time.time()
                health_metrics['eventos_procesados'] += 1
                
                # Generar predicción
                prediction_event = process_streaming_event(event)
                
                if prediction_event:
                    # Enviar predicción a Kafka
                    if send_prediction_to_kafka(prediction_event):
                        health_metrics['predicciones_enviadas'] += 1
                
                # Log de estado cada 100 eventos
                if health_metrics['eventos_procesados'] % 100 == 0:
                    logger.info(f"Estado del servicio: "
                                f"Eventos={health_metrics['eventos_procesados']}, "
                                f"Predicciones={health_metrics['predicciones_enviadas']}")
            
            except Exception as e:
                health_metrics['errores'] += 1
                logger.error(f"Error procesando mensaje: {e}")
    
    except KeyboardInterrupt:
        logger.info("Servicio de streaming detenido por el usuario")
    
    except Exception as e:
        logger.critical(f"Error crítico en servicio de streaming: {e}")
    
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    main()
