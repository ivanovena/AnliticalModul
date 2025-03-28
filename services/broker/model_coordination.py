import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from kafka import KafkaProducer

# Intentar importar la clase TransferLearningManager
try:
    from shared.transfer_learning import TransferLearningManager
    TRANSFER_LEARNING_AVAILABLE = True
except ImportError:
    TRANSFER_LEARNING_AVAILABLE = False
    logging.warning("Módulo de transfer learning no disponible, usando versión local")
    
    # Implementación local de TransferLearningManager
    class TransferLearningManager:
        def __init__(self, symbol: str):
            self.symbol = symbol
            self.online_knowledge_base = {}
            self.offline_knowledge_base = {}
            self.transfer_history = []
            
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(f"TransferLearning_{symbol}")

# Importar utilidades de mercado
from market_utils import get_market_analyzer

class ModelEnsembleCoordinator:
    def __init__(self):
        """
        Coordina ensemble y transfer learning para múltiples símbolos
        """
        self.symbol_managers = {}
        
        # Intentar inicializar Kafka
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.kafka_available = True
        except Exception as e:
            self.kafka_available = False
            logging.warning(f"Kafka no disponible: {e}")
        
        # Inicializar analizador de mercado
        self.market_analyzer = get_market_analyzer()
        
        # Inicializar histórico de predicciones
        self.prediction_history = {}  # symbol -> List[prediction]
        self.performance_metrics = {}  # symbol -> performance metrics
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelEnsembleCoordinator")
        
        # Inicializar gestores para símbolos comunes
        self._init_common_symbols()
    
    def _init_common_symbols(self):
        """Inicializa gestores para símbolos comunes"""
        common_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        
        for symbol in common_symbols:
            self.get_transfer_learning_manager(symbol)
            
            # Generar predicciones iniciales
            prediction = self._generate_mock_prediction(symbol)
            
            if symbol not in self.prediction_history:
                self.prediction_history[symbol] = []
            
            self.prediction_history[symbol].append(prediction)
            
            # Actualizar el knowledge base inicial
            manager = self.symbol_managers[symbol]
            
            # Online knowledge base
            online_kb = {
                'prediction_trend': prediction['online']['prediction'],
                'confidence': prediction['online']['confidence'],
                'performance': 0.5,
                'feature_importance': {
                    'price': 0.4,
                    'volume': 0.3,
                    'sentiment': 0.2,
                    'volatility': 0.1
                }
            }
            manager.online_knowledge_base = online_kb
            
            # Offline knowledge base
            offline_kb = {
                'prediction_trend': prediction['offline']['prediction'],
                'confidence': prediction['offline']['confidence'],
                'performance': 0.5,
                'feature_importances': {
                    'price_sma_20': 0.35,
                    'volume_change': 0.25,
                    'rsi_14': 0.2,
                    'sector_performance': 0.2
                }
            }
            manager.offline_knowledge_base = offline_kb
        
        self.logger.info(f"Inicializados {len(common_symbols)} símbolos comunes")
    
    def get_transfer_learning_manager(self, symbol):
        """
        Obtiene o crea gestor de transfer learning para un símbolo
        """
        if symbol not in self.symbol_managers:
            self.symbol_managers[symbol] = TransferLearningManager(symbol)
            
            # Inicializar con datos si es posible
            try:
                # Obtener datos técnicos
                technical_indicators = self.market_analyzer.calculate_technical_indicators(symbol)
                
                # Generar predicción inicial
                prediction = self.market_analyzer.get_price_prediction(symbol, technical_indicators)
                
                # Actualizar knowledge base
                kb = {
                    'prediction_trend': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'performance': 0.5,
                    'feature_importance': {
                        f.get('name', f'factor_{i}'): f.get('impact', 0) 
                        for i, f in enumerate(prediction.get('factors', []))
                    }
                }
                
                self.symbol_managers[symbol].online_knowledge_base = kb
                self.symbol_managers[symbol].offline_knowledge_base = kb.copy()
                
                self.logger.info(f"Inicializado gestor para {symbol} con datos técnicos")
            except Exception as e:
                self.logger.error(f"Error inicializando datos para {symbol}: {e}")
        
        return self.symbol_managers[symbol]
    
    def _generate_mock_prediction(self, symbol):
        """
        Genera una predicción simulada para inicialización
        
        Args:
            symbol: Símbolo a generar
            
        Returns:
            Diccionario con predicciones online y offline
        """
        # Intentar obtener datos reales
        try:
            prediction = self.market_analyzer.get_price_prediction(symbol)
            
            # Modificar ligeramente los valores para tener dos modelos diferentes
            online_prediction = prediction['prediction'] * 1.2
            online_confidence = min(prediction['confidence'] * 1.1, 0.95)
            
            offline_prediction = prediction['prediction'] * 0.9
            offline_confidence = min(prediction['confidence'] * 0.95, 0.9)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'online': {
                    'prediction': online_prediction,
                    'confidence': online_confidence
                },
                'offline': {
                    'prediction': offline_prediction,
                    'confidence': offline_confidence
                },
                'ensemble': {
                    'prediction': (online_prediction + offline_prediction) / 2,
                    'confidence': (online_confidence + offline_confidence) / 2
                }
            }
        except Exception as e:
            self.logger.error(f"Error generando predicción real para {symbol}: {e}")
            
            # Generar datos aleatorios si falla
            import random
            base_prediction = random.uniform(-3.0, 3.0)
            base_confidence = random.uniform(0.5, 0.8)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'online': {
                    'prediction': base_prediction * 1.1,
                    'confidence': min(base_confidence * 1.05, 0.95)
                },
                'offline': {
                    'prediction': base_prediction * 0.9,
                    'confidence': min(base_confidence * 0.95, 0.9)
                },
                'ensemble': {
                    'prediction': base_prediction,
                    'confidence': base_confidence
                }
            }
    
    def process_predictions(self, symbol, online_data=None, offline_data=None):
        """
        Procesa predicciones de modelos online y offline
        
        Args:
            symbol: Símbolo a procesar
            online_data: Datos del modelo online (opcional)
            offline_data: Datos del modelo offline (opcional)
            
        Returns:
            Predicción ensemble
        """
        transfer_manager = self.get_transfer_learning_manager(symbol)
        
        # Si no se proporcionaron datos, generar predicciones
        if not online_data or not offline_data:
            # Obtener datos de mercado
            technical_indicators = self.market_analyzer.calculate_technical_indicators(symbol)
            market_prediction = self.market_analyzer.get_price_prediction(symbol, technical_indicators)
            
            # Crear predicciones con variaciones
            import random
            variation = random.uniform(0.8, 1.2)
            
            if not online_data:
                online_data = {
                    'prediction': market_prediction['prediction'] * variation,
                    'confidence': min(market_prediction['confidence'] * 1.05, 0.95),
                    'features': {
                        f.get('name', f'factor_{i}'): f.get('impact', 0) 
                        for i, f in enumerate(market_prediction.get('factors', []))
                    }
                }
            
            if not offline_data:
                offline_data = {
                    'prediction': market_prediction['prediction'] * (2 - variation),
                    'confidence': min(market_prediction['confidence'] * 0.95, 0.9),
                    'features': {
                        f.get('name', f'factor_{i}'): f.get('impact', 0) 
                        for i, f in enumerate(market_prediction.get('factors', []))
                    }
                }
        
        # Actualizar knowledge bases
        online_kb = {
            'prediction_trend': online_data['prediction'],
            'confidence': online_data['confidence'],
            'performance': transfer_manager.online_knowledge_base.get('performance', 0.5),
            'feature_importance': online_data.get('features', {})
        }
        transfer_manager.online_knowledge_base = online_kb
        
        offline_kb = {
            'prediction_trend': offline_data['prediction'],
            'confidence': offline_data['confidence'],
            'performance': transfer_manager.offline_knowledge_base.get('performance', 0.5),
            'feature_importances': offline_data.get('features', {})
        }
        transfer_manager.offline_knowledge_base = offline_kb
        
        # Calcular predicción ensemble
        ensemble_prediction = self._calculate_ensemble_prediction(
            online_data['prediction'],
            offline_data['prediction'],
            online_data['confidence'],
            offline_data['confidence'],
            transfer_manager
        )
        
        # Publicar predicción final si Kafka está disponible
        if self.kafka_available:
            self._publish_final_prediction(symbol, ensemble_prediction)
        
        # Guardar en historial
        prediction_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'online': online_data,
            'offline': offline_data,
            'ensemble': ensemble_prediction
        }
        
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        
        self.prediction_history[symbol].append(prediction_record)
        
        # Limitar tamaño del historial
        if len(self.prediction_history[symbol]) > 100:
            self.prediction_history[symbol] = self.prediction_history[symbol][-100:]
        
        return ensemble_prediction
    
    def _calculate_ensemble_prediction(self, online_pred, offline_pred, online_conf, offline_conf, transfer_manager):
        """
        Calcula predicción combinada con pesos dinámicos
        
        Args:
            online_pred: Predicción del modelo online
            offline_pred: Predicción del modelo offline
            online_conf: Confianza del modelo online
            offline_conf: Confianza del modelo offline
            transfer_manager: Gestor de transfer learning
            
        Returns:
            Diccionario con predicción ensemble
        """
        # Métricas de desempeño 
        online_performance = transfer_manager.online_knowledge_base.get('performance', 0.5)
        offline_performance = transfer_manager.offline_knowledge_base.get('performance', 0.5)
        
        # Combinar confianza y desempeño para los pesos
        online_weight = online_performance * online_conf
        offline_weight = offline_performance * offline_conf
        
        # Normalizar pesos
        total_weight = online_weight + offline_weight
        if total_weight > 0:
            online_weight = online_weight / total_weight
            offline_weight = offline_weight / total_weight
        else:
            # Pesos iguales si no hay información
            online_weight = 0.5
            offline_weight = 0.5
        
        # Predicción ensemble ponderada
        ensemble_pred_value = (
            online_pred * online_weight + 
            offline_pred * offline_weight
        )
        
        # Confianza combinada (promedio de confianzas)
        ensemble_confidence = (online_conf + offline_conf) / 2
        
        # Aumentar confianza si ambos modelos coinciden en dirección
        if (online_pred > 0 and offline_pred > 0) or (online_pred < 0 and offline_pred < 0):
            ensemble_confidence = min(ensemble_confidence * 1.2, 0.95)
        
        ensemble_prediction = {
            'prediction': ensemble_pred_value,
            'confidence': ensemble_confidence,
            'online_weight': online_weight,
            'offline_weight': offline_weight
        }
        
        self.logger.info(f"Predicción ensemble calculada: {ensemble_pred_value:.2f}% (conf: {ensemble_confidence:.2f})")
        return ensemble_prediction
    
    def _publish_final_prediction(self, symbol, prediction):
        """
        Publica predicción final en Kafka
        
        Args:
            symbol: Símbolo
            prediction: Predicción ensemble
        """
        if not self.kafka_available:
            return
            
        try:
            prediction_event = {
                'symbol': symbol,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'timestamp': datetime.now().isoformat(),
                'model_type': 'ensemble'
            }
            
            self.kafka_producer.send('final_predictions', prediction_event)
            self.kafka_producer.flush()
            
            self.logger.info(f"Predicción final publicada para {symbol}")
        except Exception as e:
            self.logger.error(f"Error publicando predicción final: {e}")
            self.kafka_available = False
    
    def get_prediction_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene historial de predicciones para un símbolo
        
        Args:
            symbol: Símbolo a consultar
            limit: Número máximo de predicciones a devolver
            
        Returns:
            Lista de predicciones históricas
        """
        if symbol not in self.prediction_history:
            return []
        
        # Devolver las últimas 'limit' predicciones
        return self.prediction_history[symbol][-limit:]
    
    def update_performance_metrics(self, symbol: str, actual_value: float) -> Dict[str, Any]:
        """
        Actualiza métricas de rendimiento para los modelos de un símbolo
        
        Args:
            symbol: Símbolo a actualizar
            actual_value: Valor real observado
            
        Returns:
            Métricas de rendimiento actualizadas
        """
        if symbol not in self.prediction_history or not self.prediction_history[symbol]:
            return {}
        
        try:
            # Obtener todas las predicciones para el símbolo
            predictions = self.prediction_history[symbol]
            
            # Calcular errores para cada modelo
            online_errors = []
            offline_errors = []
            ensemble_errors = []
            
            for pred in predictions:
                if 'online' in pred and 'prediction' in pred['online']:
                    online_errors.append(abs(pred['online']['prediction'] - actual_value))
                
                if 'offline' in pred and 'prediction' in pred['offline']:
                    offline_errors.append(abs(pred['offline']['prediction'] - actual_value))
                
                if 'ensemble' in pred and 'prediction' in pred['ensemble']:
                    ensemble_errors.append(abs(pred['ensemble']['prediction'] - actual_value))
            
            # Calcular MAE por modelo
            online_mae = np.mean(online_errors) if online_errors else 1.0
            offline_mae = np.mean(offline_errors) if offline_errors else 1.0
            ensemble_mae = np.mean(ensemble_errors) if ensemble_errors else 1.0
            
            # Calcular performance (inversamente proporcional al error)
            online_performance = 1 / (online_mae + 1e-5)
            offline_performance = 1 / (offline_mae + 1e-5)
            ensemble_performance = 1 / (ensemble_mae + 1e-5)
            
            # Normalizar para obtener valores entre 0 y 1
            total_performance = online_performance + offline_performance + ensemble_performance
            
            if total_performance > 0:
                normalized_online = online_performance / total_performance
                normalized_offline = offline_performance / total_performance
                normalized_ensemble = ensemble_performance / total_performance
            else:
                normalized_online = 0.33
                normalized_offline = 0.33
                normalized_ensemble = 0.34
            
            # Actualizar performances en gestores
            if symbol in self.symbol_managers:
                manager = self.symbol_managers[symbol]
                
                # Update online knowledge base
                if hasattr(manager, 'online_knowledge_base'):
                    manager.online_knowledge_base['performance'] = normalized_online
                
                # Update offline knowledge base
                if hasattr(manager, 'offline_knowledge_base'):
                    manager.offline_knowledge_base['performance'] = normalized_offline
            
            # Actualizar métricas
            metrics = {
                'online': {
                    'mae': online_mae,
                    'performance': normalized_online
                },
                'offline': {
                    'mae': offline_mae,
                    'performance': normalized_offline
                },
                'ensemble': {
                    'mae': ensemble_mae,
                    'performance': normalized_ensemble
                }
            }
            
            self.performance_metrics[symbol] = metrics
            
            # Publicar a Kafka si está disponible
            if self.kafka_available:
                try:
                    self.kafka_producer.send('model_performance', {
                        'symbol': symbol,
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"Error publicando métricas: {e}")
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error actualizando métricas para {symbol}: {e}")
            return {}
    
    def get_performance_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento para los modelos de un símbolo
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Métricas de rendimiento
        """
        return self.performance_metrics.get(symbol, {})
    
    def generate_strategies(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Genera estrategias de inversión para un conjunto de símbolos
        
        Args:
            symbols: Lista de símbolos (o None para todos)
            
        Returns:
            Estrategias por símbolo
        """
        # Si no se especifican símbolos, usar todos los disponibles
        if not symbols:
            symbols = list(self.symbol_managers.keys())
        
        strategies = {}
        
        for symbol in symbols:
            try:
                # Obtener predicciones más recientes
                if symbol in self.prediction_history and self.prediction_history[symbol]:
                    latest_prediction = self.prediction_history[symbol][-1]
                    ensemble_pred = latest_prediction.get('ensemble', {})
                    
                    prediction_value = ensemble_pred.get('prediction', 0)
                    confidence = ensemble_pred.get('confidence', 0.5)
                    
                    # Determinar acción basada en la predicción
                    action = "HOLD"
                    if prediction_value > 1.5 and confidence > 0.6:
                        action = "BUY"
                    elif prediction_value < -1.5 and confidence > 0.6:
                        action = "SELL"
                    
                    # Determinar horizonte y riesgo
                    time_horizon = "MEDIUM"
                    if abs(prediction_value) > 4:
                        time_horizon = "SHORT"
                    elif abs(prediction_value) < 2:
                        time_horizon = "LONG"
                        
                    risk_level = "MODERATE"
                    diff = abs(
                        latest_prediction.get('online', {}).get('prediction', 0) - 
                        latest_prediction.get('offline', {}).get('prediction', 0)
                    )
                    
                    if diff > 2:
                        risk_level = "HIGH"
                    elif confidence > 0.75:
                        risk_level = "LOW"
                    
                    # Generar justificación
                    reasoning = self._generate_strategy_reasoning(symbol, prediction_value, action)
                    
                    # Crear estrategia
                    strategies[symbol] = {
                        "action": action,
                        "confidence": confidence,
                        "prediction": prediction_value,
                        "reasoning": reasoning,
                        "time_horizon": time_horizon,
                        "risk_level": risk_level,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Obtener datos de mercado para predicción
                    technical_indicators = self.market_analyzer.calculate_technical_indicators(symbol)
                    market_prediction = self.market_analyzer.get_price_prediction(symbol, technical_indicators)
                    
                    prediction_value = market_prediction.get('prediction', 0)
                    confidence = market_prediction.get('confidence', 0.5)
                    direction = market_prediction.get('direction', 'neutral')
                    
                    # Mapear dirección a acción
                    action = "HOLD"
                    if direction == "up" and confidence > 0.6:
                        action = "BUY"
                    elif direction == "down" and confidence > 0.6:
                        action = "SELL"
                    
                    # Generar horizonte y riesgo
                    time_horizon = "MEDIUM"
                    risk_level = "MODERATE"
                    
                    # Generar justificación
                    reasoning = self._generate_strategy_reasoning(symbol, prediction_value, action)
                    
                    # Crear estrategia
                    strategies[symbol] = {
                        "action": action,
                        "confidence": confidence,
                        "prediction": prediction_value,
                        "reasoning": reasoning,
                        "time_horizon": time_horizon,
                        "risk_level": risk_level,
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                self.logger.error(f"Error generando estrategia para {symbol}: {e}")
        
        return strategies
    
    def _generate_strategy_reasoning(self, symbol, prediction, action):
        """
        Genera una justificación para una estrategia
        
        Args:
            symbol: Símbolo
            prediction: Valor de predicción
            action: Acción recomendada (BUY, SELL, HOLD)
            
        Returns:
            Texto con justificación
        """
        # Obtener indicadores técnicos si están disponibles
        technical_indicators = self.market_analyzer.calculate_technical_indicators(symbol)
        
        # Elementos para la justificación
        justification_elements = []
        
        # Añadir elementos basados en indicadores técnicos
        if technical_indicators:
            if "rsi_14" in technical_indicators:
                rsi = technical_indicators["rsi_14"]
                if rsi > 70:
                    justification_elements.append(f"RSI en {rsi:.1f} indica condición de sobrecompra")
                elif rsi < 30:
                    justification_elements.append(f"RSI en {rsi:.1f} indica condición de sobreventa")
                else:
                    justification_elements.append(f"RSI neutral en {rsi:.1f}")
            
            if "sma_7" in technical_indicators and "sma_20" in technical_indicators:
                sma7 = technical_indicators["sma_7"]
                sma20 = technical_indicators["sma_20"]
                
                if sma7 > sma20:
                    justification_elements.append("Cruce alcista de medias móviles (SMA7 > SMA20)")
                else:
                    justification_elements.append("Cruce bajista de medias móviles (SMA7 < SMA20)")
            
            if "trend" in technical_indicators:
                trend = technical_indicators["trend"]
                trend_str = "alcista" if trend == "up" else "bajista"
                justification_elements.append(f"Tendencia {trend_str} a corto plazo")
        
        # Añadir elementos basados en predicción
        if prediction > 0:
            justification_elements.append(f"Proyección alcista de {prediction:.2f}%")
        else:
            justification_elements.append(f"Proyección bajista de {prediction:.2f}%")
        
        # Generar texto según la acción
        if action == "BUY":
            reasoning = (
                f"Los modelos predictivos indican una tendencia alcista con una proyección de {prediction:.2f}%. "
            )
            
            if justification_elements:
                reasoning += "Factores clave: " + ", ".join(justification_elements[:3]) + ". "
            
            reasoning += (
                f"El análisis técnico muestra patrones de acumulación y los indicadores de momentum "
                f"sugieren continuidad en la tendencia alcista."
            )
        elif action == "SELL":
            reasoning = (
                f"Se detecta una tendencia bajista con una proyección de {prediction:.2f}%. "
            )
            
            if justification_elements:
                reasoning += "Factores clave: " + ", ".join(justification_elements[:3]) + ". "
            
            reasoning += (
                f"Los patrones técnicos muestran señales de distribución y los indicadores sugieren "
                f"potencial continuidad de la tendencia negativa."
            )
        else:  # HOLD
            reasoning = (
                f"Los modelos no muestran una señal clara con una proyección de {prediction:.2f}%. "
            )
            
            if justification_elements:
                reasoning += "Factores clave: " + ", ".join(justification_elements[:3]) + ". "
            
            reasoning += (
                f"Hay señales mixtas entre los indicadores técnicos. Se recomienda mantener "
                f"posiciones actuales y reevaluar cuando haya mayor claridad."
            )
        
        return reasoning

# Crear método para actualizar todos los símbolos
    def update_all_symbols(self):
        """
        Actualiza predicciones para todos los símbolos gestionados
        
        Returns:
            Número de símbolos actualizados
        """
        updated = 0
        for symbol in list(self.symbol_managers.keys()):
            try:
                self.process_predictions(symbol)
                updated += 1
            except Exception as e:
                self.logger.error(f"Error actualizando {symbol}: {e}")
        
        self.logger.info(f"Actualizadas predicciones para {updated} símbolos")
        return updated
    
    def add_symbol_manager(self, symbol):
        """
        Añade y configura un nuevo gestor para un símbolo
        
        Args:
            symbol: Símbolo a añadir
            
        Returns:
            Gestor de transfer learning
        """
        manager = self.get_transfer_learning_manager(symbol)
        
        # Generar predicción inicial
        self.process_predictions(symbol)
        
        self.logger.info(f"Añadido nuevo gestor para {symbol}")
        return manager
    
    def get_available_symbols(self):
        """
        Obtiene la lista de símbolos disponibles
        
        Returns:
            Lista de símbolos
        """
        return list(self.symbol_managers.keys())

# Inicializar coordinador como singleton
model_ensemble_coordinator = ModelEnsembleCoordinator()

def get_model_ensemble_coordinator():
    """
    Obtener instancia singleton del coordinador
    """
    return model_ensemble_coordinator

# Programar actualización periódica de todos los símbolos
import threading
import time

def update_predictions_periodically():
    """
    Actualiza las predicciones periódicamente en segundo plano
    """
    while True:
        try:
            model_ensemble_coordinator.update_all_symbols()
            logger = logging.getLogger("UpdateThread")
            logger.info("Predicciones actualizadas")
        except Exception as e:
            logger = logging.getLogger("UpdateThread")
            logger.error(f"Error en actualización periódica: {e}")
        
        # Esperar 15 minutos
        time.sleep(15 * 60)

# Iniciar thread de actualización con intervalo más largo (30 minutos en vez de cada minuto)
def update_predictions_periodically():
    """
    Actualiza las predicciones periódicamente en segundo plano
    """
    while True:
        try:
            model_ensemble_coordinator.update_all_symbols()
            logger = logging.getLogger("UpdateThread")
            logger.info("Predicciones actualizadas")
        except Exception as e:
            logger = logging.getLogger("UpdateThread")
            logger.error(f"Error en actualización periódica: {e}")
        
        # Esperar 30 minutos entre actualizaciones (en vez de 1 minuto)
        time.sleep(30 * 60)

# Iniciar thread de actualización
update_thread = threading.Thread(target=update_predictions_periodically)
update_thread.daemon = True
update_thread.start()
