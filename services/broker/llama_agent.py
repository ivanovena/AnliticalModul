# llama_agent.py
import os
import json
import logging
import time
from datetime import datetime
import uuid
import re
import numpy as np
import pickle
from pydantic import BaseModel
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import requests
from fastapi import HTTPException
import concurrent.futures

# Configuración para usar menos memoria con Llama
os.environ["LLAMA_CPP_ENABLE_MLOCK"] = "0"  # Desactivar bloqueo de memoria
os.environ["LLAMA_CPP_MAX_BATCH_SIZE"] = "8"  # Limitar tamaño de batch
os.environ["LLAMA_CPP_SEED"] = "42"  # Seed para reproducibilidad

# Importación condicional de Llama
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("llama-cpp-python no está disponible. Usando respuestas predefinidas.")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("broker_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LlamaAgent")

# Modelo de datos para el chat
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

class ConversationHistory(BaseModel):
    messages: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class BrokerStrategy(BaseModel):
    """Modelo para representar estrategias de trading"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    prediction: float
    reasoning: str
    time_horizon: str  # SHORT, MEDIUM, LONG
    risk_level: str    # LOW, MODERATE, HIGH

class MarketData(BaseModel):
    """Modelo para datos de mercado"""
    symbol: str
    price: float
    change: float
    volume: int
    timestamp: str
    
class ModelPrediction(BaseModel):
    """Modelo para predicciones de modelos de ML"""
    symbol: str
    prediction: float
    confidence: float
    timestamp: str
    model_type: str
    features: Dict[str, float] = {}

class OrderRequest(BaseModel):
    """Modelo para solicitudes de órdenes"""
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    
# Caché para datos de mercado y predicciones
class ModelCache:
    def __init__(self):
        self.market_data = {}  # symbol -> MarketData
        self.predictions = {}  # symbol -> List[ModelPrediction]
        self.strategies = {}   # symbol -> BrokerStrategy
        self.last_update = {}  # symbol -> timestamp
        self.lock = threading.RLock()
        
    def update_market_data(self, symbol: str, data: MarketData):
        """Actualiza datos de mercado para un símbolo"""
        with self.lock:
            self.market_data[symbol] = data
            self.last_update[symbol] = datetime.now()
            
    def update_prediction(self, prediction: ModelPrediction):
        """Actualiza predicciones para un símbolo"""
        with self.lock:
            symbol = prediction.symbol
            if symbol not in self.predictions:
                self.predictions[symbol] = []
            
            # Agregar nueva predicción and mantener solo las 5 más recientes
            self.predictions[symbol].append(prediction)
            self.predictions[symbol] = self.predictions[symbol][-5:]
            self.last_update[symbol] = datetime.now()
    
    def update_strategy(self, symbol: str, strategy: BrokerStrategy):
        """Actualiza estrategia para un símbolo"""
        with self.lock:
            self.strategies[symbol] = strategy
            self.last_update[symbol] = datetime.now()
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Obtiene datos de mercado para un símbolo"""
        with self.lock:
            return self.market_data.get(symbol)
    
    def get_predictions(self, symbol: str) -> List[ModelPrediction]:
        """Obtiene predicciones para un símbolo"""
        with self.lock:
            return self.predictions.get(symbol, [])
    
    def get_strategy(self, symbol: str) -> Optional[BrokerStrategy]:
        """Obtiene estrategia para un símbolo"""
        with self.lock:
            return self.strategies.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Obtiene todos los símbolos con datos"""
        with self.lock:
            # Unión de todos los símbolos en caché
            all_symbols = set(self.market_data.keys()) | set(self.predictions.keys()) | set(self.strategies.keys())
            return list(all_symbols)

class LlamaAgent:
    """Agente de chat basado en Llama para el broker"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el agente Llama
        
        Args:
            model_path: Ruta al modelo Llama (GGUF) or nombre del modelo en HF
        """
        self.conversations = {}  # Almacena el historial de conversaciones
        self.model = None
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        self.cache = ModelCache()  # Caché para datos and predicciones
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Para operaciones asíncronas
        self.fmp_api_key = os.getenv("FMP_API_KEY", "h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx")
        
        # Nuevo: Sistema de aprendizaje y retroalimentación
        self.feedback_store = self._load_feedback_store()
        self.exemplary_conversations = self._load_exemplary_conversations()
        self.learning_metrics = {
            "total_conversations": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "avg_conversation_length": 0,
            "most_common_intents": {},
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Inicializando LlamaAgent con modelo {self.model_path}")
        
        try:
            # Importar coordinador de modelos ensemble
            from model_coordination import get_model_ensemble_coordinator
            self.model_coordinator = get_model_ensemble_coordinator()
            logger.info("Coordinador de modelos ensemble inicializado")
        except Exception as e:
            logger.error(f"Error importando coordinador de modelos: {e}")
            self.model_coordinator = None
        
        try:
            # Inicializar modelo usando prioritariamente Ollama
            if LLAMA_AVAILABLE:
                # Priorizar uso de modelos disponibles en Ollama
                try:
                    # Importar nuestro cliente de Ollama personalizado
                    from ollama_client import get_ollama_client
                    
                    # Obtener instancia del cliente
                    ollama_client = get_ollama_client()
                    
                    # Lista de modelos a intentar usar en orden de preferencia
                    ollama_models = ["deepseek-coder:33b", "phi4:latest", "gemma3:27b", "codellama:13b", "deepseek-coder:6.7b"]
                    
                    # Función para enviar consultas a Ollama
                    def generate_with_ollama(prompt, max_tokens=512, model="deepseek-coder:33b"):
                        try:
                            options = {
                                "num_predict": max_tokens,
                                "temperature": 0.7
                            }
                            
                            # Sistema prompt mejorado para un análisis financiero más completo
                            system_prompt = (
                                "Eres un asistente financiero especializado en trading algorítmico con las siguientes responsabilidades:\n\n"
                                "1. ANÁLISIS DE MERCADO: Proporciona análisis técnico y fundamental detallado basado en datos actuales.\n"
                                "2. ESTRATEGIAS DE INVERSIÓN: Ofrece recomendaciones concretas (COMPRAR/VENDER/MANTENER) con niveles de confianza.\n"
                                "3. INTERPRETACIÓN DE MODELOS: Explica predicciones de modelos de ML en términos comprensibles.\n"
                                "4. GESTIÓN DE RIESGO: Incluye siempre evaluaciones de riesgo y horizontes temporales en tus recomendaciones.\n"
                                "5. EDUCACIÓN FINANCIERA: Explica conceptos complejos cuando sea relevante.\n\n"
                                "LIMITACIONES IMPORTANTES:\n"
                                "- Indica siempre el nivel de incertidumbre en tus análisis. Nunca garantices resultados específicos.\n"
                                "- Cuando falten datos, sé transparente sobre las limitaciones de tu análisis.\n"
                                "- Aclara que tus recomendaciones son educativas, no asesoramiento financiero regulado.\n"
                                "- Si una pregunta está fuera del ámbito financiero o de trading, redirige amablemente al usuario.\n\n"
                                "FORMATO DE RESPUESTAS:\n"
                                "- Para análisis de símbolos: Estructura con Precio Actual, Predicción, Confianza, Recomendación y Razonamiento.\n"
                                "- Para estrategias: Incluye Acción, Horizonte Temporal, Nivel de Riesgo y Factores Clave.\n"
                                "- Para métricas: Presenta datos con formato claro y explica su significado.\n\n"
                                "Tu objetivo es ayudar a los usuarios a tomar decisiones financieras informadas basadas en datos y análisis algorítmico."
                            )
                            
                            # Usar chat completion para modelos nuevos
                            if model in ["phi4:latest", "gemma3:27b"]:
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt}
                                ]
                                response = ollama_client.chat_completion(
                                    model=model, 
                                    messages=messages,
                                    options=options
                                )
                                return {"response": response.get("message", {}).get("content", "")}
                            else:
                                # Usar generate para otros modelos
                                response = ollama_client.generate(
                                    model=model,
                                    prompt=prompt,
                                    system=system_prompt,
                                    options=options
                                )
                                return response
                        except Exception as e:
                            logger.error(f"Error con Ollama: {e}")
                            return {"response": "Error generando respuesta con el modelo."}
                    
                    # Probar cada modelo en orden de preferencia
                    model_found = False
                    for model_name in ollama_models:
                        try:
                            logger.info(f"Intentando cargar modelo {model_name} desde Ollama...")
                            test_response = generate_with_ollama("Hola, ¿cómo estás?", max_tokens=10, model=model_name)
                            
                            if test_response and "response" in test_response:
                                # Modelo encontrado y funcional
                                logger.info(f"Modelo {model_name} cargado correctamente desde Ollama")
                                
                                # Crear un wrapper compatible con la interfaz esperada
                                class OllamaWrapper:
                                    def __init__(self, model_name):
                                        self.model_name = model_name
                                    
                                    def __call__(self, prompt, max_tokens=512, stop=None, temperature=0.7, repeat_penalty=1.1):
                                        response = generate_with_ollama(
                                            prompt=prompt, 
                                            max_tokens=max_tokens,
                                            model=self.model_name
                                        )
                                        # Formatear respuesta para compatibilidad
                                        return {
                                            "choices": [
                                                {
                                                    "text": response.get("response", "")
                                                }
                                            ]
                                        }
                                
                                # Usar el wrapper como modelo
                                self.model = OllamaWrapper(model_name)
                                model_found = True
                                break
                        except Exception as e:
                            logger.warning(f"Error al intentar usar modelo {model_name}: {e}")
                    
                    if not model_found:
                        logger.warning("No se pudo cargar ningún modelo desde Ollama")
                except ImportError as e:
                    logger.warning(f"No se pudo importar la librería de cliente de Ollama: {e}. Intentando cargar modelo directo.")
                
                # Si no se pudo usar Ollama, intentar cargar el modelo directamente
                if self.model is None and os.path.exists(self.model_path):
                    # Si es un archivo local GGUF
                    logger.info(f"Cargando modelo local: {self.model_path}")
                    self.model = Llama(
                        model_path=self.model_path,
                        n_ctx=4096,        # Contexto mayor
                        n_batch=64,        # Batch moderado
                        n_gpu_layers=-1,   # Usar todas las capas en GPU
                        use_mlock=True,    # Bloquear en memoria
                        seed=42            # Reproducibilidad
                    )
                    logger.info(f"Modelo Llama cargado desde {self.model_path}")
            else:
                logger.warning("llama-cpp-python no está disponible. Usando respuestas predefinidas.")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error inicializando el modelo Llama: {traceback.format_exc()}")
            self.model = None
            logger.info("Usando respuestas pre-programadas como fallback")

        # Iniciar thread de actualización de datos de mercado
        self.should_run = True
        self.update_thread = threading.Thread(target=self._update_market_data_periodically)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Cargar estrategias iniciales
        self._load_initial_strategies()

    def _load_feedback_store(self) -> Dict[str, Any]:
        """Carga almacén de retroalimentación desde un archivo"""
        try:
            if os.path.exists("feedback_store.pkl"):
                with open("feedback_store.pkl", "rb") as f:
                    return pickle.load(f)
            return {"responses": {}, "metrics": {}}
        except Exception as e:
            logger.error(f"Error cargando almacén de retroalimentación: {e}")
            return {"responses": {}, "metrics": {}}
    
    def _load_exemplary_conversations(self) -> List[Dict[str, Any]]:
        """Carga conversaciones ejemplares para entrenamiento futuro"""
        try:
            if os.path.exists("exemplary_conversations.pkl"):
                with open("exemplary_conversations.pkl", "rb") as f:
                    return pickle.load(f)
            return []
        except Exception as e:
            logger.error(f"Error cargando conversaciones ejemplares: {e}")
            return []
    
    def _save_feedback_store(self):
        """Guarda almacén de retroalimentación en un archivo"""
        try:
            with open("feedback_store.pkl", "wb") as f:
                pickle.dump(self.feedback_store, f)
            logger.info("Almacén de retroalimentación guardado")
        except Exception as e:
            logger.error(f"Error guardando almacén de retroalimentación: {e}")
    
    def _save_exemplary_conversations(self):
        """Guarda conversaciones ejemplares en un archivo"""
        try:
            with open("exemplary_conversations.pkl", "wb") as f:
                pickle.dump(self.exemplary_conversations, f)
            logger.info("Conversaciones ejemplares guardadas")
        except Exception as e:
            logger.error(f"Error guardando conversaciones ejemplares: {e}")
        
    def add_feedback(self, conversation_id: str, response_id: str, feedback: Dict[str, Any]):
        """
        Añade retroalimentación para una respuesta específica
        
        Args:
            conversation_id: ID de la conversación
            response_id: ID de la respuesta (timestamp)
            feedback: Diccionario con retroalimentación (rating, comments, etc.)
        """
        if conversation_id not in self.feedback_store["responses"]:
            self.feedback_store["responses"][conversation_id] = {}
        
        self.feedback_store["responses"][conversation_id][response_id] = feedback
        
        # Actualizar métricas
        rating = feedback.get("rating", 0)
        if rating > 3:  # Escala 1-5
            self.learning_metrics["positive_feedback"] += 1
        else:
            self.learning_metrics["negative_feedback"] += 1
        
        # Si es muy positivo, considerar como conversación ejemplar
        if rating >= 4.5 and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            if conversation not in self.exemplary_conversations:
                self.exemplary_conversations.append(conversation)
                self._save_exemplary_conversations()
        
        self._save_feedback_store()

    def _load_initial_strategies(self):
        """Carga estrategias iniciales para símbolos comunes"""
        common_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", 
                         "IAG.MC", "PHM.MC", "BKY.MC", "AENA.MC", "BA", 
                         "NLGO", "CAR", "DLTR", "CANTE.IS", "SASA.IS"]
        
        for symbol in common_symbols:
            # Crear estrategia simulada para cada símbolo
            strategy = self._generate_symbol_strategy(symbol)
            self.cache.update_strategy(symbol, strategy)
        
        logger.info(f"Estrategias iniciales cargadas para {len(common_symbols)} símbolos")

    def _update_market_data_periodically(self):
        """Actualiza datos de mercado periódicamente"""
        while self.should_run:
            try:
                # Obtener símbolos actuales
                symbols = self.cache.get_all_symbols()
                
                if symbols:
                    # Añadir símbolos comunes si no hay ninguno en caché
                    if not symbols:
                        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
                    
                    # Consultar datos de mercado
                    symbols_str = ",".join(symbols)
                    url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}?apikey={self.fmp_api_key}"
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        quotes = response.json()
                        
                        # Actualizar caché con datos de mercado
                        for quote in quotes:
                            market_data = MarketData(
                                symbol=quote["symbol"],
                                price=quote["price"],
                                change=quote["change"],
                                volume=quote["volume"],
                                timestamp=datetime.now().isoformat()
                            )
                            self.cache.update_market_data(quote["symbol"], market_data)
                        
                        logger.info(f"Datos de mercado actualizados para {len(quotes)} símbolos")
                    else:
                        logger.warning(f"Error obteniendo datos de mercado: {response.status_code}")
                
                # Actualizar predicciones y estrategias
                self._update_predictions_and_strategies()
                
            except Exception as e:
                logger.error(f"Error en actualización periódica: {e}")
            
            # Dormir antes de la próxima actualización (5 minutos)
            time.sleep(300)

    def _update_predictions_and_strategies(self):
        """Actualiza predicciones y estrategias basadas en el modelo ensemble"""
        try:
            if self.model_coordinator:
                # Obtener símbolos con gestores
                symbol_managers = self.model_coordinator.symbol_managers
                
                for symbol, manager in symbol_managers.items():
                    # Extraer predicciones
                    online_kb = manager.online_knowledge_base
                    offline_kb = manager.offline_knowledge_base
                    
                    if online_kb:
                        # Crear predicción del modelo online
                        online_prediction = ModelPrediction(
                            symbol=symbol,
                            prediction=online_kb.get('prediction_trend', 0.0),
                            confidence=online_kb.get('confidence', 0.5),
                            timestamp=datetime.now().isoformat(),
                            model_type='online',
                            features=online_kb.get('feature_importance', {})
                        )
                        self.cache.update_prediction(online_prediction)
                    
                    if offline_kb:
                        # Crear predicción del modelo offline
                        offline_prediction = ModelPrediction(
                            symbol=symbol,
                            prediction=offline_kb.get('prediction_trend', 0.0),
                            confidence=offline_kb.get('confidence', 0.5),
                            timestamp=datetime.now().isoformat(),
                            model_type='offline',
                            features=offline_kb.get('feature_importances', {})
                        )
                        self.cache.update_prediction(offline_prediction)
                    
                    # Intentar generar estrategia enriquecida
                    try:
                        # Crear un loop asíncrono
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Generar estrategia enriquecida con datos fundamentales y noticias
                        enriched_strategy = loop.run_until_complete(self._generate_enriched_symbol_strategy(symbol))
                        loop.close()
                        
                        # Actualizar caché con la estrategia enriquecida
                        self.cache.update_strategy(symbol, enriched_strategy)
                    except Exception as e:
                        logger.warning(f"Error generando estrategia enriquecida para {symbol}, usando método tradicional: {e}")
                        # En caso de error, usar el método tradicional
                        strategy = self._generate_symbol_strategy(symbol)
                        self.cache.update_strategy(symbol, strategy)
                
                logger.info(f"Predicciones y estrategias actualizadas para {len(symbol_managers)} símbolos")
            else:
                # Si no hay coordinador, generar predicciones simuladas
                symbols = self.cache.get_all_symbols()
                
                for symbol in symbols:
                    # Intentar generar estrategia enriquecida
                    try:
                        # Crear un loop asíncrono
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Generar estrategia enriquecida
                        enriched_strategy = loop.run_until_complete(self._generate_enriched_symbol_strategy(symbol))
                        loop.close()
                        
                        # Actualizar caché
                        self.cache.update_strategy(symbol, enriched_strategy)
                    except Exception as e:
                        logger.warning(f"Error generando estrategia enriquecida simulada para {symbol}: {e}")
                        # En caso de error, usar el método tradicional
                        strategy = self._generate_symbol_strategy(symbol)
                        self.cache.update_strategy(symbol, strategy)
                
                logger.info(f"Estrategias simuladas actualizadas para {len(symbols)} símbolos")
        except Exception as e:
            logger.error(f"Error actualizando predicciones y estrategias: {e}")

    def process_message(self, chat_message: ChatMessage) -> ChatResponse:
        """
        Procesa un mensaje de chat y genera una respuesta
        
        Args:
            chat_message: Mensaje del usuario
            
        Returns:
            Respuesta generada y ID de conversación
        """
        # Obtener or crear ID de conversación
        conversation_id = chat_message.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = ConversationHistory(
                messages=[],
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            )
        elif conversation_id not in self.conversations:
            # Si el ID no existe, crear nueva conversación
            self.conversations[conversation_id] = ConversationHistory(
                messages=[],
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            )
        
        # Agregar mensaje a la conversación
        conversation = self.conversations[conversation_id]
        conversation.messages.append({
            "role": "user",
            "content": chat_message.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generar respuesta
        response = self._generate_response(chat_message.message, conversation_id)
        
        # Agregar respuesta a la conversación
        conversation.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Actualizar timestamp
        conversation.metadata["last_updated"] = datetime.now().isoformat()
        
        # Limitar tamaño del historial de conversaciones
        if len(conversation.messages) > 50:
            conversation.messages = conversation.messages[-50:]
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id
        )
        
    def _generate_response(self, message: str, conversation_id: str) -> str:
        """
        Genera una respuesta utilizando el modelo Llama or reglas predefinidas
        
        Args:
            message: Mensaje del usuario
            conversation_id: ID de la conversación
            
        Returns:
            Respuesta generada
        """
        # Si el mensaje es vacío or muy corto
        if not message or len(message) < 2:
            return "Por favor, envía un mensaje más detallado para poder ayudarte."
        
        # Detectar intenciones específicas
        intent, params = self._detect_intent(message)
        
        # Procesar según la intención detectada
        if intent == "get_strategy":
            return self._get_investment_strategy(params.get("symbol"))
        elif intent == "get_portfolio":
            return self._get_portfolio_info()
        elif intent == "get_cash":
            return self._get_cash_info()
        elif intent == "place_order":
            return self._handle_order_intent(params)
        elif intent == "get_risk":
            return self._get_risk_metrics()
        elif intent == "get_market_data":
            return self._get_market_data(params.get("symbol"))
        elif intent == "get_predictions":
            return self._get_predictions(params.get("symbol"))
        elif intent == "get_help":
            return self._get_help_info()
        
        # Si llegamos aquí, intentar generar respuesta con Llama
        if self.model:
            try:
                # Obtener contexto de la conversación
                conversation = self.conversations[conversation_id]
                prompt = self._build_llama_prompt(conversation, message)
                
                # Generar respuesta con el modelo
                response = self.model(
                    prompt,
                    max_tokens=512,
                    stop=["<|user|>", "<|assistant|>", "\n\n"],
                    temperature=0.7,
                    repeat_penalty=1.1
                )
                
                # Extraer texto de respuesta
                if isinstance(response, dict) and "choices" in response:
                    text = response["choices"][0]["text"]
                else:
                    text = str(response)
                
                # Limpiar respuesta
                text = text.strip()
                
                # Si la respuesta está vacía, usar respuesta predeterminada
                if not text:
                    text = "No pude generar una respuesta. ¿Puedes reformular tu pregunta?"
                
                return text
            except Exception as e:
                logger.error(f"Error generando respuesta con Llama: {e}")
                return "Lo siento, tuve un problema generando una respuesta. Por favor, intenta de nuevo."
        else:
            # Respuesta por defecto si no hay modelo
            return self._get_fallback_response(message)
    
    def _build_llama_prompt(self, conversation: ConversationHistory, message: str) -> str:
        """
        Construye un prompt para el modelo Llama basado en el historial de conversación
        
        Args:
            conversation: Historial de conversación
            message: Mensaje actual del usuario
            
        Returns:
            Prompt formateado
        """
        # Obtener contexto adicional (datos de mercado, predicciones y datos enriquecidos)
        market_context = self._get_market_context()
        enriched_context = self._get_enriched_context(message)
        
        # Sistema prompt mejorado para análisis financiero más completo
        system_prompt = (
            "Eres un asistente financiero especializado en trading algorítmico con las siguientes responsabilidades:\n\n"
            "1. ANÁLISIS DE MERCADO: Proporciona análisis técnico y fundamental detallado basado en datos actuales.\n"
            "2. ESTRATEGIAS DE INVERSIÓN: Ofrece recomendaciones concretas (COMPRAR/VENDER/MANTENER) con niveles de confianza.\n"
            "3. INTERPRETACIÓN DE MODELOS: Explica predicciones de modelos de ML en términos comprensibles.\n"
            "4. GESTIÓN DE RIESGO: Incluye siempre evaluaciones de riesgo y horizontes temporales en tus recomendaciones.\n"
            "5. EDUCACIÓN FINANCIERA: Explica conceptos complejos cuando sea relevante.\n\n"
            "LIMITACIONES IMPORTANTES:\n"
            "- Indica siempre el nivel de incertidumbre en tus análisis. Nunca garantices resultados específicos.\n"
            "- Cuando falten datos, sé transparente sobre las limitaciones de tu análisis.\n"
            "- Aclara que tus recomendaciones son educativas, no asesoramiento financiero regulado.\n"
            "- Si una pregunta está fuera del ámbito financiero o de trading, redirige amablemente al usuario.\n\n"
            "FORMATO DE RESPUESTAS:\n"
            "- Para análisis de símbolos: Estructura con Precio Actual, Predicción, Confianza, Recomendación y Razonamiento.\n"
            "- Para estrategias: Incluye Acción, Horizonte Temporal, Nivel de Riesgo y Factores Clave.\n"
            "- Para métricas: Presenta datos con formato claro y explica su significado.\n\n"
            "DATOS HISTÓRICOS DE ÉXITO:\n"
            "- Con tus análisis previos has ayudado a usuarios a entender tendencias de mercado y tomar decisiones más informadas.\n"
            "- Las estrategias que mejor han funcionado combinan análisis fundamental y técnico, con especial atención a la gestión del riesgo.\n\n"
            "Tu objetivo es ayudar a los usuarios a tomar decisiones financieras informadas basadas en datos y análisis algorítmico."
        )
        
        # Construir historial de conversación
        conversation_history = ""
        for msg in conversation.messages[-6:-1]:  # últimos 5 mensajes (excluyendo el actual)
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                conversation_history += f"<|user|>\n{content}\n"
            else:
                conversation_history += f"<|assistant|>\n{content}\n"
        
        # Incluir ejemplos de conversaciones exitosas previas si hay disponibles (few-shot learning)
        exemplary_context = ""
        if hasattr(self, 'exemplary_conversations') and self.exemplary_conversations:
            # Seleccionar una conversación ejemplar relevante (basada en similaridad con el mensaje actual)
            # En una implementación completa, aquí se añadiría un algoritmo de similitud semántica
            exemplary = self.exemplary_conversations[0] if self.exemplary_conversations else None
            
            if exemplary and exemplary.messages and len(exemplary.messages) >= 2:
                # Tomar solo un par pregunta-respuesta de la conversación ejemplar
                user_msg = next((msg for msg in exemplary.messages if msg["role"] == "user"), None)
                assistant_msg = next((msg for msg in exemplary.messages if msg["role"] == "assistant"), None)
                
                if user_msg and assistant_msg:
                    exemplary_context = (
                        "Ejemplo de consulta similar:\n"
                        f"Usuario: {user_msg['content']}\n"
                        f"Respuesta: {assistant_msg['content']}\n\n"
                    )
        
        # Formato final del prompt
        prompt = f"{system_prompt}\n\n"
        
        if market_context:
            prompt += f"Información de mercado actual:\n{market_context}\n\n"
            
        if enriched_context:
            prompt += f"{enriched_context}\n\n"
        
        if exemplary_context:
            prompt += f"{exemplary_context}\n"
            
        if conversation_history:
            prompt += f"{conversation_history}\n"
            
        prompt += f"<|user|>\n{message}\n<|assistant|>\n"
        
        return prompt
    
    def _get_market_context(self) -> str:
        """
        Obtiene un resumen del contexto de mercado actual
        
        Returns:
            Texto con información de mercado
        """
        symbols = self.cache.get_all_symbols()
        if not symbols:
            return ""
        
        context_lines = []
        
        # Seleccionar hasta 5 símbolos para incluir en el contexto
        selected_symbols = symbols[:5]
        
        for symbol in selected_symbols:
            market_data = self.cache.get_market_data(symbol)
            predictions = self.cache.get_predictions(symbol)
            strategy = self.cache.get_strategy(symbol)
            
            if market_data:
                line = f"{symbol}: ${market_data.price:.2f} ({market_data.change:+.2f}%)"
                
                if predictions and len(predictions) > 0:
                    latest_pred = predictions[-1]
                    line += f", Predicción: {latest_pred.prediction:.2f} (Confianza: {latest_pred.confidence:.2f})"
                
                if strategy:
                    line += f", Recomendación: {strategy.action}"
                
                context_lines.append(line)
        
        return "\n".join(context_lines)
    
    def _get_enriched_context(self, message: str) -> str:
        """
        Obtiene un contexto enriquecido basado en el mensaje del usuario
        usando datos adicionales de FMP como noticias y datos fundamentales
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            Texto con información enriquecida relevante para el contexto
        """
        try:
            # Detectar símbolos en el mensaje
            symbol_pattern = r'\b([A-Z]{1,5}(?:\.MC|\.IS)?)\b'
            symbols = re.findall(symbol_pattern, message.upper())
            
            # Si no hay símbolos específicos, usar algunas acciones populares
            if not symbols:
                symbols = ["AAPL", "MSFT", "GOOG"]
            
            # Limitar a máximo 2 símbolos para el contexto
            symbols = symbols[:2]
            
            # Importar cliente de datos
            from data_client import get_data_client
            data_client = get_data_client()
            
            # Crear un loop para operaciones asíncronas
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Función para obtener datos de un símbolo de forma asíncrona
            async def get_symbol_data(symbol):
                try:
                    # Obtener noticias
                    news = await data_client.get_company_news(symbol, limit=3)
                    
                    # Obtener perfil de la empresa
                    profile = await data_client.get_company_profile(symbol)
                    
                    # Obtener métricas financieras
                    metrics = await data_client.get_key_metrics(symbol)
                    
                    # Obtener análisis de sentimiento
                    sentiment = await data_client.get_market_sentiment(symbol)
                    
                    return {
                        "symbol": symbol,
                        "news": news,
                        "profile": profile,
                        "metrics": metrics,
                        "sentiment": sentiment
                    }
                except Exception as e:
                    logger.error(f"Error obteniendo datos para {symbol}: {e}")
                    return {"symbol": symbol, "error": str(e)}
            
            # Crear y ejecutar tareas para cada símbolo
            tasks = [get_symbol_data(symbol) for symbol in symbols]
            symbol_data_list = loop.run_until_complete(asyncio.gather(*tasks))
            
            # Cerrar cliente de datos y loop
            loop.run_until_complete(data_client.close())
            loop.close()
            
            # Formatear la información relevante para el contexto
            context_parts = []
            
            for data in symbol_data_list:
                symbol = data["symbol"]
                
                # Verificar si hubo error
                if "error" in data:
                    context_parts.append(f"No se pudieron obtener datos completos para {symbol}.")
                    continue
                
                symbol_context = [f"Información enriquecida para {symbol}:"]
                
                # Añadir información del perfil si está disponible
                profile = data.get("profile", {})
                if profile:
                    symbol_context.append(
                        f"Perfil: {profile.get('name', symbol)} - "
                        f"Sector: {profile.get('sector', 'N/A')}, "
                        f"Industria: {profile.get('industry', 'N/A')}, "
                        f"Cap. de Mercado: ${profile.get('marketCap', 0)/1000000000:.2f}B"
                    )
                
                # Añadir métricas financieras clave si están disponibles
                metrics = data.get("metrics", {})
                if metrics:
                    symbol_context.append(
                        f"Métricas: P/E: {metrics.get('pe_ratio', 0):.2f}, "
                        f"P/B: {metrics.get('price_to_book', 0):.2f}, "
                        f"ROE: {metrics.get('roe', 0)*100:.2f}%, "
                        f"Margen operativo: {metrics.get('operating_margin', 0)*100:.2f}%"
                    )
                
                # Añadir análisis de sentimiento si está disponible
                sentiment = data.get("sentiment", {})
                if sentiment and "overall_score" in sentiment:
                    symbol_context.append(
                        f"Sentimiento: {sentiment.get('overall_score', 0):.2f}, "
                        f"Recomendación: {sentiment.get('recommendation', 'NEUTRAL')}"
                    )
                
                # Añadir titulares de noticias recientes
                news = data.get("news", [])
                if news:
                    symbol_context.append("Noticias recientes:")
                    for i, item in enumerate(news[:3], 1):
                        # Añadir titular con sentimiento
                        news_sentiment = item.get('sentiment', {}).get('score', 0)
                        sentiment_icon = "🟢" if news_sentiment > 30 else "🔴" if news_sentiment < -30 else "🟡"
                        
                        symbol_context.append(
                            f"{i}. {sentiment_icon} {item.get('title', 'Sin titular')} "
                            f"({item.get('source', 'N/A')})"
                        )
                
                # Agregar toda la información del símbolo al contexto
                context_parts.append("\n".join(symbol_context))
            
            # Añadir próximos eventos si se detecta una intención relacionada con planificación
            if re.search(r'\bcalendario|eventos|próximo|futuro|ganancias|dividendos\b', message.lower()):
                try:
                    # Obtener próximos eventos de ganancias
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    data_client = get_data_client()
                    
                    earnings_events = loop.run_until_complete(data_client.get_earnings_calendar())
                    loop.run_until_complete(data_client.close())
                    loop.close()
                    
                    if earnings_events and len(earnings_events) > 0:
                        events_context = ["Próximos eventos de ganancias:"]
                        for event in earnings_events[:5]:  # Limitar a 5 eventos
                            events_context.append(
                                f"{event.get('symbol', 'N/A')}: {event.get('date', 'N/A')}, "
                                f"EPS est.: ${event.get('eps', 0):.2f}"
                            )
                        context_parts.append("\n".join(events_context))
                except Exception as e:
                    logger.error(f"Error obteniendo calendario de ganancias: {e}")
            
            # Unir todas las partes del contexto
            full_context = "\n\n".join(context_parts)
            
            return full_context
            
        except Exception as e:
            logger.error(f"Error generando contexto enriquecido: {e}")
            # Devolver un contexto mínimo en caso de error
            return "No se pudieron obtener datos adicionales en este momento."
    
    def _detect_intent(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detecta la intención del usuario a partir del mensaje
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            Tuple con intención detectada y parámetros
        """
        message = message.lower()
        params = {}
        
        # Detectar símbolos mencionados
        symbol_pattern = r'\b([A-Z]{1,5}(?:\.MC|\.IS)?)\b'
        symbols = re.findall(symbol_pattern, message.upper())
        if symbols:
            params["symbol"] = symbols[0]  # Usar el primer símbolo encontrado
        
        # Detectar intenciones específicas
        if re.search(r'\bestrategia|recomienda|qué (debo|debería) hacer|compro|vendo\b', message):
            return "get_strategy", params
        elif re.search(r'\bportafolio|cartera|posiciones|inversiones|tenencias\b', message):
            return "get_portfolio", params
        elif re.search(r'\befectivo|cash|saldo|dinero|capital\b', message):
            return "get_cash", params
        elif re.search(r'\bcompra[r]?|vende[r]?|orden\b', message) and "symbol" in params:
            # Detectar cantidad
            qty_pattern = r'(\d+)\s+(acciones|títulos|unidades)'
            qty_match = re.search(qty_pattern, message)
            if qty_match:
                params["quantity"] = int(qty_match.group(1))
            
            # Detectar acción
            if re.search(r'\bcompra[r]?\b', message):
                params["action"] = "BUY"
            elif re.search(r'\bvende[r]?\b', message):
                params["action"] = "SELL"
                
            return "place_order", params
        elif re.search(r'\briesgo|volatilidad|exposición\b', message):
            return "get_risk", params
        elif re.search(r'\bprecio|cotización|datos|valor\b', message) and "symbol" in params:
            return "get_market_data", params
        elif re.search(r'\bpredicci[óo]n|previ[óo]n|estimaci[óo]n|forecast\b', message) and "symbol" in params:
            return "get_predictions", params
        elif re.search(r'\bayuda|help|comandos|cómo\b', message):
            return "get_help", params
        
        # Si no se detecta ninguna intención específica
        return "general_chat", params
    
    def _generate_symbol_strategy(self, symbol: str) -> BrokerStrategy:
        """
        Genera una estrategia de inversión para un símbolo
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Estrategia de inversión
        """
        # Intentar obtener predicciones reales
        predictions = self.cache.get_predictions(symbol)
        market_data = self.cache.get_market_data(symbol)
        
        # Si tenemos datos reales, usarlos para la estrategia
        if predictions and len(predictions) > 0 and market_data:
            # Usar la predicción más reciente
            latest_pred = predictions[-1]
            prediction = latest_pred.prediction
            confidence = latest_pred.confidence
            
            # Determinar acción basada en la predicción
            if prediction > 0.15 and confidence > 0.6:
                action = "BUY"
                reasoning = f"Fuerte señal alcista con {confidence:.2f} de confianza."
                risk_level = "MODERATE"
            elif prediction > 0.05 and confidence > 0.5:
                action = "BUY"
                reasoning = f"Señal alcista con {confidence:.2f} de confianza."
                risk_level = "LOW"
            elif prediction < -0.15 and confidence > 0.6:
                action = "SELL"
                reasoning = f"Fuerte señal bajista con {confidence:.2f} de confianza."
                risk_level = "HIGH"
            elif prediction < -0.05 and confidence > 0.5:
                action = "SELL"
                reasoning = f"Señal bajista con {confidence:.2f} de confianza."
                risk_level = "MODERATE"
            else:
                action = "HOLD"
                reasoning = f"No hay señal clara. Confianza: {confidence:.2f}"
                risk_level = "LOW"
                
            # Determinar horizonte temporal
            time_horizon = "MEDIUM"  # Por defecto
            
            # Crear y devolver estrategia
            return BrokerStrategy(
                symbol=symbol,
                action=action,
                confidence=confidence,
                prediction=prediction,
                reasoning=reasoning,
                time_horizon=time_horizon,
                risk_level=risk_level
            )
        else:
            # Generar estrategia simulada si no hay datos reales
            # Usar números aleatorios deterministas
            np.random.seed(hash(symbol) % 10000)
            
            # Generar predicción aleatoria entre -0.2 y 0.2
            prediction = np.random.uniform(-0.2, 0.2)
            confidence = np.random.uniform(0.5, 0.9)
            
            # Determinar acción basada en la predicción simulada
            if prediction > 0.1:
                action = "BUY"
                reasoning = "Tendencia alcista detectada."
                risk_level = "MODERATE"
            elif prediction < -0.1:
                action = "SELL"
                reasoning = "Tendencia bajista detectada."
                risk_level = "MODERATE"
            else:
                action = "HOLD"
                reasoning = "Mercado sin tendencia clara."
                risk_level = "LOW"
            
            # Posibles horizontes temporales
            time_horizons = ["SHORT", "MEDIUM", "LONG"]
            time_horizon = np.random.choice(time_horizons)
            
            # Crear y devolver estrategia simulada
            return BrokerStrategy(
                symbol=symbol,
                action=action,
                confidence=confidence,
                prediction=prediction,
                reasoning=reasoning,
                time_horizon=time_horizon,
                risk_level=risk_level
            )
    
    async def _generate_enriched_symbol_strategy(self, symbol: str) -> BrokerStrategy:
        """
        Genera una estrategia de inversión enriquecida para un símbolo
        combinando predicciones de modelos técnicos con datos fundamentales
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Estrategia de inversión enriquecida
        """
        try:
            # Importar cliente de datos
            from data_client import get_data_client
            data_client = get_data_client()
            
            # Crear loop asíncrono
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Obtener datos enriquecidos
            profile = await data_client.get_company_profile(symbol)
            metrics = await data_client.get_key_metrics(symbol)
            news = await data_client.get_company_news(symbol, limit=5)
            sentiment = await data_client.get_market_sentiment(symbol)
            
            # Obtener predicciones técnicas
            predictions = self.cache.get_predictions(symbol)
            market_data = self.cache.get_market_data(symbol)
            
            # Cerrar cliente
            await data_client.close()
            loop.close()
            
            # Variables para la estrategia final
            action = "HOLD"
            confidence = 0.5
            prediction = 0.0
            reasoning = []
            risk_level = "MODERATE"
            time_horizon = "MEDIUM"
            
            # 1. Considerar predicciones técnicas de nuestros modelos
            if predictions and len(predictions) > 0:
                latest_pred = predictions[-1]
                technical_prediction = latest_pred.prediction
                technical_confidence = latest_pred.confidence
                
                # Agregar esta información a la estrategia
                prediction = technical_prediction
                confidence = technical_confidence
                
                if technical_prediction > 0.15 and technical_confidence > 0.6:
                    reasoning.append(f"Análisis técnico: Fuerte señal alcista con {technical_confidence:.2f} de confianza.")
                    action = "BUY"
                elif technical_prediction > 0.05 and technical_confidence > 0.5:
                    reasoning.append(f"Análisis técnico: Señal alcista moderada con {technical_confidence:.2f} de confianza.")
                    action = "BUY"
                elif technical_prediction < -0.15 and technical_confidence > 0.6:
                    reasoning.append(f"Análisis técnico: Fuerte señal bajista con {technical_confidence:.2f} de confianza.")
                    action = "SELL"
                elif technical_prediction < -0.05 and technical_confidence > 0.5:
                    reasoning.append(f"Análisis técnico: Señal bajista moderada con {technical_confidence:.2f} de confianza.")
                    action = "SELL"
                else:
                    reasoning.append(f"Análisis técnico: Sin tendencia clara. Confianza: {technical_confidence:.2f}")
            
            # 2. Considerar análisis de sentimiento de mercado
            if sentiment and "overall_score" in sentiment:
                sentiment_score = sentiment.get("overall_score", 0)
                sentiment_recommendation = sentiment.get("recommendation", "NEUTRAL")
                
                reasoning.append(f"Sentimiento de mercado: {sentiment_recommendation} (score: {sentiment_score:.2f})")
                
                # Ajustar confianza y predicción según el sentimiento
                if abs(sentiment_score) > 30:
                    # Si el sentimiento es fuerte, ajustar la confianza y la predicción
                    confidence = (confidence + 0.8) / 2  # Promedio con alta confianza
                    prediction = (prediction + sentiment_score/100) / 2  # Ajustar predicción
                    
                    # Si el sentimiento contradice fuertemente el análisis técnico, ajustar acción
                    if sentiment_score > 30 and action == "SELL":
                        action = "HOLD"
                        reasoning.append("Señal mixta: Sentimiento positivo contradice análisis técnico bajista.")
                    elif sentiment_score < -30 and action == "BUY":
                        action = "HOLD"
                        reasoning.append("Señal mixta: Sentimiento negativo contradice análisis técnico alcista.")
            
            # 3. Considerar datos fundamentales
            if metrics:
                pe_ratio = metrics.get("pe_ratio", 0)
                price_to_book = metrics.get("price_to_book", 0)
                debt_to_equity = metrics.get("debt_to_equity", 0)
                current_ratio = metrics.get("current_ratio", 0)
                
                # Evaluar métricas de valoración
                if pe_ratio > 0:  # Asegurarse que el PE no sea negativo
                    pe_interpretation = ""
                    
                    # Interpretación del P/E (estos son valores generales, pueden ajustarse por sector)
                    if pe_ratio < 10:
                        pe_interpretation = "potencialmente infravalorada"
                    elif pe_ratio > 30:
                        pe_interpretation = "potencialmente sobrevalorada"
                    
                    if pe_interpretation:
                        reasoning.append(f"Valoración: P/E ratio de {pe_ratio:.2f} indica empresa {pe_interpretation}.")
                
                # Evaluar salud financiera
                if current_ratio < 1:
                    reasoning.append(f"Riesgo de liquidez: Ratio de liquidez bajo ({current_ratio:.2f}).")
                    risk_level = "HIGH"
                
                if debt_to_equity > 2:
                    reasoning.append(f"Riesgo de deuda: Alto nivel de apalancamiento (D/E: {debt_to_equity:.2f}).")
                    risk_level = "HIGH"
                elif debt_to_equity < 0.5:
                    reasoning.append(f"Fortaleza financiera: Bajo nivel de deuda (D/E: {debt_to_equity:.2f}).")
                    if risk_level != "HIGH":  # No bajar el nivel si ya es alto
                        risk_level = "LOW"
            
            # 4. Considerar noticias recientes
            if news and len(news) > 0:
                # Calcular sentimiento promedio de las noticias
                news_scores = [item.get("sentiment", {}).get("score", 0) for item in news if "sentiment" in item]
                if news_scores:
                    avg_news_sentiment = sum(news_scores) / len(news_scores)
                    
                    # Encontrar la noticia más importante (con mayor sentimiento absoluto)
                    important_news = max(news, key=lambda x: abs(x.get("sentiment", {}).get("score", 0)) if "sentiment" in x else 0)
                    
                    if abs(avg_news_sentiment) > 30:
                        news_direction = "positivas" if avg_news_sentiment > 0 else "negativas"
                        reasoning.append(f"Noticias recientes: Predominantemente {news_direction}.")
                        
                        # Destacar noticia importante
                        important_title = important_news.get("title", "")
                        important_score = important_news.get("sentiment", {}).get("score", 0)
                        if important_title and abs(important_score) > 40:
                            reasoning.append(f"Noticia destacada: '{important_title[:50]}...'")
                            
                            # Si hay una noticia muy negativa y la acción es comprar, aumentar el riesgo
                            if important_score < -50 and action == "BUY":
                                risk_level = "HIGH"
                                reasoning.append("Precaución: Noticias muy negativas aumentan el riesgo.")
                            
                            # Si hay una noticia muy positiva y la acción es vender, reconsiderar
                            if important_score > 50 and action == "SELL":
                                reasoning.append("Nota: A pesar de los indicadores técnicos negativos, las noticias son muy positivas.")
            
            # 5. Determinar horizonte temporal basado en la combinación de factores
            if "Fortaleza financiera" in ' '.join(reasoning) and action == "BUY":
                time_horizon = "LONG"
                reasoning.append("Recomendación para largo plazo basada en fortaleza financiera.")
            elif "Riesgo" in ' '.join(reasoning) and action != "HOLD":
                time_horizon = "SHORT"
                reasoning.append("Horizonte temporal corto debido a factores de riesgo identificados.")
            
            # Crear y devolver estrategia enriquecida
            return BrokerStrategy(
                symbol=symbol,
                action=action,
                confidence=round(confidence, 2),
                prediction=round(prediction, 2),
                reasoning=". ".join(reasoning),
                time_horizon=time_horizon,
                risk_level=risk_level
            )
        
        except Exception as e:
            logger.error(f"Error generando estrategia enriquecida para {symbol}: {e}")
            # En caso de error, volver al método tradicional
            return self._generate_symbol_strategy(symbol)
    
    def _get_investment_strategy(self, symbol: str) -> str:
        """
        Obtiene la estrategia de inversión para un símbolo
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Texto con la estrategia
        """
        if not symbol:
            return "Por favor, especifica un símbolo de activo (por ejemplo, AAPL, MSFT, etc.)"
        
        try:
            # Intentar generar una estrategia enriquecida con datos fundamentales
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Generar estrategia enriquecida
            enriched_strategy = loop.run_until_complete(self._generate_enriched_symbol_strategy(symbol))
            loop.close()
            
            # Actualizar la caché con la estrategia enriquecida
            self.cache.update_strategy(symbol, enriched_strategy)
            
            # Obtener datos de mercado
            market_data = self.cache.get_market_data(symbol)
            
            # Construir respuesta
            response = [f"**Estrategia para {symbol}**"]
            
            if market_data:
                response.append(f"Precio actual: ${market_data.price:.2f} ({market_data.change:+.2f}%)")
            
            response.append(f"Recomendación: {enriched_strategy.action}")
            response.append(f"Confianza: {enriched_strategy.confidence:.2f}")
            response.append(f"Predicción: {enriched_strategy.prediction:.2f}%")
            response.append(f"Horizonte temporal: {enriched_strategy.time_horizon}")
            response.append(f"Nivel de riesgo: {enriched_strategy.risk_level}")
            response.append("\n**Análisis Completo:**")
            response.append(f"{enriched_strategy.reasoning}")
            
            return "\n".join(response)
            
        except Exception as e:
            logger.error(f"Error generando estrategia enriquecida: {e}")
            
            # Si hubo un error, usar el método tradicional como fallback
            strategy = self.cache.get_strategy(symbol)
            market_data = self.cache.get_market_data(symbol)
            
            if not strategy:
                # Generar estrategia si no existe
                strategy = self._generate_symbol_strategy(symbol)
                self.cache.update_strategy(symbol, strategy)
            
            # Construir respuesta
            response = [f"**Estrategia para {symbol}**"]
            
            if market_data:
                response.append(f"Precio actual: ${market_data.price:.2f} ({market_data.change:+.2f}%)")
            
            response.append(f"Recomendación: {strategy.action}")
            response.append(f"Confianza: {strategy.confidence:.2f}")
            response.append(f"Predicción: {strategy.prediction:.2f}")
            response.append(f"Razonamiento: {strategy.reasoning}")
            response.append(f"Horizonte temporal: {strategy.time_horizon}")
            response.append(f"Nivel de riesgo: {strategy.risk_level}")
            
            return "\n".join(response)
    
    def _get_portfolio_info(self) -> str:
        """
        Obtiene información del portafolio
        
        Returns:
            Texto con la información del portafolio
        """
        # Esta es una versión simulada - en una implementación real conectaría con una BD
        portfolio = [
            {"symbol": "AAPL", "quantity": 10, "avg_price": 175.23, "current_price": 178.45},
            {"symbol": "MSFT", "quantity": 5, "avg_price": 330.12, "current_price": 338.22},
            {"symbol": "GOOG", "quantity": 8, "avg_price": 138.72, "current_price": 141.98},
        ]
        
        total_value = sum(item["quantity"] * item["current_price"] for item in portfolio)
        total_cost = sum(item["quantity"] * item["avg_price"] for item in portfolio)
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
        
        # Construir respuesta
        response = ["**Tu Portafolio**"]
        response.append(f"Valor total: ${total_value:.2f}")
        response.append(f"Ganancia/Pérdida: ${total_gain_loss:.2f} ({total_gain_loss_pct:+.2f}%)")
        response.append("\n**Posiciones:**")
        
        for item in portfolio:
            symbol = item["symbol"]
            qty = item["quantity"]
            avg_price = item["avg_price"]
            current_price = item["current_price"]
            position_value = qty * current_price
            gain_loss = (current_price - avg_price) * qty
            gain_loss_pct = ((current_price / avg_price) - 1) * 100
            
            position_str = (
                f"{symbol}: {qty} acciones, "
                f"Valor: ${position_value:.2f}, "
                f"G/P: ${gain_loss:.2f} ({gain_loss_pct:+.2f}%)"
            )
            response.append(position_str)
        
        return "\n".join(response)
    
    def _get_cash_info(self) -> str:
        """
        Obtiene información del efectivo disponible
        
        Returns:
            Texto con la información del efectivo
        """
        # Versión simulada
        cash_balance = 15432.78
        buying_power = cash_balance * 2  # Con margen
        
        response = [
            "**Información de Efectivo**",
            f"Saldo disponible: ${cash_balance:.2f}",
            f"Poder de compra: ${buying_power:.2f}",
            "\nPuedes usar este efectivo para realizar nuevas inversiones."
        ]
        
        return "\n".join(response)
    
    def _handle_order_intent(self, params: Dict[str, Any]) -> str:
        """
        Procesa una intención de orden
        
        Args:
            params: Parámetros de la orden
            
        Returns:
            Texto con la respuesta
        """
        symbol = params.get("symbol")
        action = params.get("action")
        quantity = params.get("quantity", 1)
        
        if not symbol:
            return "Por favor, especifica un símbolo para la orden (por ejemplo, AAPL, MSFT)."
        
        if not action:
            return f"¿Quieres comprar o vender {symbol}? Por favor, especifica la acción."
        
        # Obtener datos de mercado
        market_data = self.cache.get_market_data(symbol)
        price = market_data.price if market_data else 100.0  # Precio predeterminado
        
        # Simular ejecución de orden
        order_id = str(uuid.uuid4())[:8]
        total_value = price * quantity
        
        # Construir respuesta
        if action == "BUY":
            response = [
                f"✅ Orden de compra enviada para {quantity} acciones de {symbol}",
                f"Precio estimado: ${price:.2f}",
                f"Valor total: ${total_value:.2f}",
                f"ID de orden: {order_id}",
                "\nLa orden se procesará en breve. Puedes consultar su estado más adelante."
            ]
        else:  # SELL
            response = [
                f"✅ Orden de venta enviada para {quantity} acciones de {symbol}",
                f"Precio estimado: ${price:.2f}",
                f"Valor total: ${total_value:.2f}",
                f"ID de orden: {order_id}",
                "\nLa orden se procesará en breve. Puedes consultar su estado más adelante."
            ]
        
        return "\n".join(response)
    
    def _get_risk_metrics(self) -> str:
        """
        Obtiene métricas de riesgo del portafolio
        
        Returns:
            Texto con las métricas de riesgo
        """
        # Valores simulados
        risk_metrics = {
            "volatility": 12.4,  # Volatilidad anualizada (%)
            "var_95": 2.3,       # Value at Risk 95% (%)
            "beta": 1.05,        # Beta contra el mercado
            "sharpe": 1.85,      # Ratio de Sharpe
            "max_drawdown": 8.2  # Máximo drawdown (%)
        }
        
        response = [
            "**Métricas de Riesgo**",
            f"Volatilidad anualizada: {risk_metrics['volatility']:.2f}%",
            f"Value at Risk (95%): {risk_metrics['var_95']:.2f}%",
            f"Beta: {risk_metrics['beta']:.2f}",
            f"Ratio de Sharpe: {risk_metrics['sharpe']:.2f}",
            f"Máximo drawdown: {risk_metrics['max_drawdown']:.2f}%",
            "\nEstas métricas te ayudan a entender el nivel de riesgo de tu portafolio."
        ]
        
        return "\n".join(response)
    
    def _get_market_data(self, symbol: str) -> str:
        """
        Obtiene datos de mercado para un símbolo
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Texto con datos de mercado
        """
        if not symbol:
            return "Por favor, especifica un símbolo (por ejemplo, AAPL, MSFT)."
        
        # Obtener datos de mercado
        market_data = self.cache.get_market_data(symbol)
        
        if not market_data:
            # Intentar obtener datos en tiempo real
            try:
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={self.fmp_api_key}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    quotes = response.json()
                    if quotes and len(quotes) > 0:
                        quote = quotes[0]
                        market_data = MarketData(
                            symbol=quote["symbol"],
                            price=quote["price"],
                            change=quote["change"],
                            volume=quote["volume"],
                            timestamp=datetime.now().isoformat()
                        )
                        self.cache.update_market_data(symbol, market_data)
            except Exception as e:
                logger.error(f"Error obteniendo datos de mercado: {e}")
        
        # Construir respuesta
        if market_data:
            response = [
                f"**Datos de Mercado: {symbol}**",
                f"Precio: ${market_data.price:.2f}",
                f"Cambio diario: {market_data.change:+.2f}%",
                f"Volumen: {market_data.volume:,}",
                f"Última actualización: {market_data.timestamp}"
            ]
            return "\n".join(response)
        else:
            return f"No se pudieron obtener datos para {symbol}. Verifica que el símbolo sea correcto."
    
    def _get_predictions(self, symbol: str) -> str:
        """
        Obtiene predicciones para un símbolo
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Texto con predicciones
        """
        if not symbol:
            return "Por favor, especifica un símbolo (por ejemplo, AAPL, MSFT)."
        
        # Obtener predicciones
        predictions = self.cache.get_predictions(symbol)
        
        if not predictions or len(predictions) == 0:
            # Generar predicción simulada
            prediction = ModelPrediction(
                symbol=symbol,
                prediction=np.random.uniform(-0.2, 0.2),
                confidence=np.random.uniform(0.5, 0.9),
                timestamp=datetime.now().isoformat(),
                model_type="simulado",
                features={}
            )
            self.cache.update_prediction(prediction)
            predictions = [prediction]
        
        # Obtener estrategia
        strategy = self.cache.get_strategy(symbol)
        if not strategy:
            strategy = self._generate_symbol_strategy(symbol)
            self.cache.update_strategy(symbol, strategy)
        
        # Construir respuesta
        response = [f"**Predicciones para {symbol}**"]
        
        for idx, pred in enumerate(predictions):
            model_type = pred.model_type.capitalize()
            direction = "alcista" if pred.prediction > 0 else "bajista"
            response.append(
                f"Modelo {model_type}: Tendencia {direction} ({pred.prediction:+.2f}), "
                f"Confianza: {pred.confidence:.2f}"
            )
            
            # Añadir features importantes (hasta 3)
            if pred.features:
                features_sorted = sorted(
                    pred.features.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:3]
                
                features_str = ", ".join(
                    f"{feature}: {importance:+.2f}" 
                    for feature, importance in features_sorted
                )
                
                response.append(f"Factores importantes: {features_str}")
        
        # Añadir recomendación basada en estrategia
        response.append(f"\n**Recomendación: {strategy.action}**")
        response.append(f"Razonamiento: {strategy.reasoning}")
        response.append(f"Horizonte temporal: {strategy.time_horizon}")
        response.append(f"Nivel de riesgo: {strategy.risk_level}")
        
        return "\n".join(response)
    
    def _get_help_info(self) -> str:
        """
        Obtiene información de ayuda
        
        Returns:
            Texto con información de ayuda
        """
        commands = [
            ("Estrategia para [símbolo]", "Obtener estrategia de inversión para un activo"),
            ("Mi portafolio", "Ver tus posiciones actuales"),
            ("Mi efectivo", "Ver saldo disponible"),
            ("Comprar [cantidad] acciones de [símbolo]", "Colocar orden de compra"),
            ("Vender [cantidad] acciones de [símbolo]", "Colocar orden de venta"),
            ("Riesgo de mi portafolio", "Ver métricas de riesgo"),
            ("Datos de [símbolo]", "Ver datos de mercado actuales"),
            ("Predicción para [símbolo]", "Ver predicciones de modelos")
        ]
        
        response = ["**Comandos Disponibles**"]
        
        for cmd, desc in commands:
            response.append(f"• **{cmd}**: {desc}")
        
        response.append("\nTambién puedes hacer preguntas generales sobre inversiones y mercados.")
        
        return "\n".join(response)
    
    def _get_fallback_response(self, message: str) -> str:
        """
        Genera una respuesta predeterminada cuando no hay modelo disponible
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            Respuesta predeterminada
        """
        # Respuestas generales pre-programadas
        general_responses = [
            "Entiendo tu consulta. Para obtener información específica, pregunta por un símbolo como AAPL o MSFT.",
            "Los mercados financieros son complejos. ¿Hay algún activo específico que te interese?",
            "Para darte la mejor asesoría, necesito más detalles sobre qué activos te interesan.",
            "Puedo ayudarte con estrategias de inversión. Prueba a preguntar por un símbolo específico.",
            "Si buscas recomendaciones, puedes preguntar por 'Estrategia para AAPL' o similar."
        ]
        
        # Seleccionar una respuesta basada en una función hash determinista
        response_idx = hash(message) % len(general_responses)
        return general_responses[response_idx]
    
    def shutdown(self):
        """Limpia recursos al apagar el servicio"""
        self.should_run = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        if self.executor:
            self.executor.shutdown(wait=False)
        logger.info("LlamaAgent apagado correctamente")
