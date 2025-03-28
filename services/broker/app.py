from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import logging
import os
import atexit
import random
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaProducer
import requests
from llama_agent import ChatMessage, ChatResponse, get_llama_agent
import threading

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("broker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrokerService")

# Variables de entorno
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
AGENT_TOPIC = os.getenv("AGENT_TOPIC", "agent_decisions")
BATCH_TOPIC = os.getenv("BATCH_TOPIC", "batch_events")  # Tópico para actualizaciones de modelos batch
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000"))
FMP_API_KEY = os.getenv("FMP_API_KEY", "h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx")

# Crear aplicación FastAPI
app = FastAPI(
    title="Broker Service API",
    description="API para el servicio de broker inteligente con IA",
    version="1.0.0"
)

# Configuración del servidor para usar el puerto 8001 como se espera en Nginx
import uvicorn

def start_app():
    """Inicia la aplicación en el puerto 8001"""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    start_app()

# Inicializar agente Llama como parte del broker
llama_agent = get_llama_agent()

# Obtener el coordinador de modelos del agente
model_coordinator = llama_agent.model_coordinator if hasattr(llama_agent, 'model_coordinator') else None

# Función para procesar actualizaciones de modelos batch
def process_batch_model_update(event):
    """
    Procesa una actualización de modelo batch
    
    Args:
        event: Evento de actualización de modelo
    """
    if not event or 'symbol' not in event:
        logger.warning("Evento de actualización de modelo inválido")
        return
    
    symbol = event['symbol']
    logger.info(f"Recibida actualización de modelo batch para {symbol}")
    
    try:
        # Extraer información del modelo
        model_data = event.get('model_data', {})
        model_metrics = event.get('metrics', {})
        feature_importance = event.get('feature_importance', {})
        
        # Si no tenemos coordinador de modelos, no podemos procesar la actualización
        if not model_coordinator:
            logger.warning(f"No hay coordinador de modelos disponible para actualizar {symbol}")
            return
        
        # Obtener el gestor de transfer learning para este símbolo
        transfer_manager = model_coordinator.get_transfer_learning_manager(symbol)
        
        # Actualizar knowledge base offline
        offline_kb = {
            'prediction_trend': model_data.get('prediction', 0),
            'confidence': model_data.get('confidence', 0.5),
            'performance': model_metrics.get('accuracy', 0.5),
            'feature_importances': feature_importance
        }
        
        # Guardar knowledge base
        transfer_manager.offline_knowledge_base = offline_kb
        
        # Transferir conocimiento importantes al modelo online
        if hasattr(transfer_manager, 'transfer_knowledge'):
            transfer_manager.transfer_knowledge('offline', 'online')
            logger.info(f"Conocimiento transferido de offline a online para {symbol}")
        
        # Procesar predicciones ensemble con los nuevos datos
        offline_data = {
            'prediction': model_data.get('prediction', 0),
            'confidence': model_data.get('confidence', 0.5),
            'features': feature_importance
        }
        
        # Actualizar predicción ensemble
        model_coordinator.process_predictions(symbol, offline_data=offline_data)
        logger.info(f"Predicción ensemble actualizada para {symbol}")
        
    except Exception as e:
        logger.error(f"Error procesando actualización de modelo para {symbol}: {e}")

# Función para consumir eventos de actualización de modelo batch
def consume_batch_model_updates():
    """
    Consume eventos de actualización de modelo batch de Kafka
    """
    try:
        # Inicializar consumidor de Kafka
        consumer = KafkaConsumer(
            BATCH_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='broker_service',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        logger.info(f"Consumidor Kafka inicializado para {BATCH_TOPIC}")
        
        # Procesar mensajes
        for message in consumer:
            try:
                event = message.value
                process_batch_model_update(event)
            except Exception as e:
                logger.error(f"Error procesando mensaje de batch_events: {e}")
                
    except Exception as e:
        logger.error(f"Error inicializando consumidor Kafka para batch_events: {e}")
        
    finally:
        logger.info("Consumidor de batch_events finalizado")

# Iniciar consumidor de batch_events en un hilo separado
batch_consumer_thread = threading.Thread(target=consume_batch_model_updates, daemon=True)
batch_consumer_thread.start()
logger.info("Consumidor de actualizaciones de modelo batch iniciado")

# Estado del broker - Portfolio vacío inicial con mejoras de seguimiento
broker_state = {
    "portfolio": {
        "cash": INITIAL_CASH,
        "initial_cash": INITIAL_CASH,  # Agregado para seguimiento del efectivo inicial
        "positions": {},  # Sin posiciones iniciales
        "total_value": INITIAL_CASH,
        "last_update": time.time()
    },
    "orders": [],  # Sin órdenes históricas
    "operations_by_stock": {},  # Nuevo: registro de operaciones por acción
    "metrics": {
        "performance": {
            "total_return": 0,
            "cash_ratio": 1.0,
            "positions_count": 0,
            "trading_frequency": 0
        },
        "stock_performance": {},  # Nuevo: rendimiento por acción
        "risk_metrics": {
            "portfolio": {
                "diversification_score": 0,
                "cash_ratio": 1.0,
                "total_value": INITIAL_CASH
            }
        }
    }
}

# Función para reiniciar el estado del broker a valores iniciales
def reset_broker_state():
    """Reinicia el estado del broker a valores iniciales"""
    global broker_state
    broker_state = {
        "portfolio": {
            "cash": INITIAL_CASH,
            "initial_cash": INITIAL_CASH,  # Agregado para seguimiento del efectivo inicial
            "positions": {},  # Sin posiciones iniciales
            "total_value": INITIAL_CASH,
            "last_update": time.time()
        },
        "orders": [],  # Sin órdenes históricas
        "operations_by_stock": {},  # Nuevo: registro de operaciones por acción
        "metrics": {
            "performance": {
                "total_return": 0,
                "cash_ratio": 1.0,
                "positions_count": 0,
                "trading_frequency": 0
            },
            "stock_performance": {},  # Nuevo: rendimiento por acción
            "risk_metrics": {
                "portfolio": {
                    "diversification_score": 0,
                    "cash_ratio": 1.0,
                    "total_value": INITIAL_CASH
                }
            }
        }
    }

# Inicializar Kafka
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        acks='all'
    )
    logger.info(f"Kafka producer inicializado con broker {KAFKA_BROKER}")
except Exception as e:
    logger.error(f"Error inicializando Kafka producer: {e}")
    producer = None

# Modelos de datos
class OrderRequest(BaseModel):
    symbol: str
    action: str
    quantity: int
    price: float

class Order(BaseModel):
    symbol: str
    action: str
    quantity: int
    price: float
    timestamp: str
    
class ChatRequest(BaseModel):
    """Modelo para solicitudes de chat al broker IA"""
    message: str
    conversation_id: Optional[str] = None

# Endpoints
@app.get("/health")
async def health_check():
    """Endpoint para verificar la salud del servicio"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "kafka_connected": producer is not None
    }

@app.get("/portfolio")
async def get_portfolio():
    """Obtener el estado actual del portafolio"""
    # Actualizar precios actuales en posiciones
    await update_portfolio_prices()
    
    # Calcular y agregar métricas adicionales al portfolio
    portfolio = broker_state["portfolio"].copy()
    
    # Agregar información de money_spent, money_left y predicted_profit
    total_money_spent = 0
    total_predicted_profit = 0
    
    for symbol, position in portfolio["positions"].items():
        # Dinero gastado en esta acción (costo promedio * cantidad)
        position["money_spent"] = position["avg_cost"] * position["quantity"]
        total_money_spent += position["money_spent"]
        
        # Valor actual de la posición
        current_value = position["current_price"] * position["quantity"]
        
        # Ganancia/pérdida actual (valor actual - dinero gastado)
        position["current_profit"] = current_value - position["money_spent"]
        
        # Obtener predicción para esta acción
        prediction = await get_stock_prediction(symbol)
        prediction_pct = prediction.get("prediction", 0)
        
        # Calcular ganancia predicha
        predicted_price = position["current_price"] * (1 + prediction_pct/100)
        predicted_value = predicted_price * position["quantity"]
        position["predicted_profit"] = predicted_value - position["money_spent"]
        
        total_predicted_profit += position["predicted_profit"]
    
    # Información global del portfolio
    portfolio["total_money_spent"] = total_money_spent
    portfolio["money_left"] = portfolio["cash"]  # Efectivo disponible
    portfolio["total_predicted_profit"] = total_predicted_profit
    
    return portfolio

@app.get("/orders")
async def get_orders():
    """Obtener historial de órdenes"""
    return broker_state["orders"]

@app.get("/operations-by-stock")
async def get_operations_by_stock():
    """Obtener operaciones agrupadas por acción"""
    return broker_state["operations_by_stock"]

@app.get("/metrics")
async def get_metrics():
    """Obtener métricas del portafolio"""
    await update_portfolio_metrics()
    return broker_state["metrics"]

@app.post("/orders")
async def place_order(order: OrderRequest):
    """Colocar una nueva orden"""
    # Validar la orden
    if order.quantity <= 0:
        raise HTTPException(status_code=400, detail="La cantidad debe ser positiva")
    
    # Validar fondos disponibles para compras
    if order.action == "BUY":
        total_cost = order.quantity * order.price
        if total_cost > broker_state["portfolio"]["cash"]:
            raise HTTPException(status_code=400, detail="Fondos insuficientes para esta compra")
    
    # Validar posiciones disponibles para ventas
    if order.action == "SELL":
        positions = broker_state["portfolio"]["positions"]
        if order.symbol not in positions or positions[order.symbol]["quantity"] < order.quantity:
            raise HTTPException(status_code=400, detail="No hay suficientes acciones para vender")
    
    # Crear la orden
    new_order = {
        "symbol": order.symbol,
        "action": order.action,
        "quantity": order.quantity,
        "price": order.price,
        "timestamp": datetime.now().isoformat(),
        "total_value": order.quantity * order.price
    }
    
    # Actualizar el portafolio
    if order.action == "BUY":
        # Reducir el efectivo
        total_cost = order.quantity * order.price
        broker_state["portfolio"]["cash"] -= total_cost
        
        # Actualizar o crear posición
        if order.symbol not in broker_state["portfolio"]["positions"]:
            broker_state["portfolio"]["positions"][order.symbol] = {
                "quantity": order.quantity,
                "current_price": order.price,
                "market_value": order.quantity * order.price,
                "avg_cost": order.price,
                "total_cost": total_cost,
                "operations": []
            }
        else:
            position = broker_state["portfolio"]["positions"][order.symbol]
            total_shares = position["quantity"] + order.quantity
            position_cost = position.get("total_cost", position["quantity"] * position["avg_cost"])
            total_cost = position_cost + (order.quantity * order.price)
            position["avg_cost"] = total_cost / total_shares
            position["quantity"] = total_shares
            position["current_price"] = order.price
            position["market_value"] = total_shares * order.price
            position["total_cost"] = total_cost
    
    elif order.action == "SELL":
        # Aumentar el efectivo
        sale_value = order.quantity * order.price
        broker_state["portfolio"]["cash"] += sale_value
        
        # Actualizar posición
        position = broker_state["portfolio"]["positions"][order.symbol]
        position["quantity"] -= order.quantity
        position["current_price"] = order.price
        position["market_value"] = position["quantity"] * order.price
        
        # Calcular ganancia de esta venta
        avg_cost_per_share = position["avg_cost"]
        profit_on_sale = (order.price - avg_cost_per_share) * order.quantity
        new_order["profit"] = profit_on_sale
        
        # Eliminar posición si ya no hay acciones
        if position["quantity"] == 0:
            del broker_state["portfolio"]["positions"][order.symbol]
    
    # Registrar operación en el historial por acción
    if order.symbol not in broker_state["operations_by_stock"]:
        broker_state["operations_by_stock"][order.symbol] = []
    
    operation_record = {
        "action": order.action,
        "quantity": order.quantity,
        "price": order.price,
        "timestamp": new_order["timestamp"],
        "total_value": new_order["total_value"]
    }
    
    if order.action == "SELL" and "profit" in new_order:
        operation_record["profit"] = new_order["profit"]
    
    broker_state["operations_by_stock"][order.symbol].append(operation_record)
    
    # Actualizar valor total
    total_value = broker_state["portfolio"]["cash"]
    for symbol, position in broker_state["portfolio"]["positions"].items():
        total_value += position["market_value"]
    broker_state["portfolio"]["total_value"] = total_value
    broker_state["portfolio"]["last_update"] = time.time()
    
    # Agregar al historial
    broker_state["orders"].append(new_order)
    
    # Publicar decisión a Kafka si está disponible
    if producer:
        try:
            producer.send(AGENT_TOPIC, {
                "action": "ORDER_PLACED",
                "data": new_order
            })
            producer.flush()
            logger.info(f"Orden enviada a Kafka: {new_order}")
        except Exception as e:
            logger.error(f"Error publicando orden a Kafka: {e}")
    
    # Actualizar métricas
    await update_portfolio_metrics()
    
    return new_order

async def update_portfolio_prices():
    """Actualizar precios actuales de las posiciones en el portafolio"""
    positions = broker_state["portfolio"]["positions"]
    if not positions:
        return
    
    # Obtener precios actuales
    symbols = list(positions.keys())
    try:
        symbols_str = ",".join(symbols)
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            quotes = response.json()
            # Crear diccionario para búsqueda rápida
            quotes_dict = {quote["symbol"]: quote for quote in quotes}
            
            # Actualizar posiciones
            total_market_value = 0
            for symbol, position in positions.items():
                if symbol in quotes_dict:
                    quote = quotes_dict[symbol]
                    position["current_price"] = quote["price"]
                    position["market_value"] = position["quantity"] * quote["price"]
                    
                    # Calcular ganancia/pérdida actual
                    position["current_profit"] = position["market_value"] - (position["avg_cost"] * position["quantity"])
                    position["profit_percent"] = (position["current_profit"] / (position["avg_cost"] * position["quantity"])) * 100 if position["avg_cost"] > 0 else 0
                
                total_market_value += position["market_value"]
            
            # Actualizar valor total del portafolio
            broker_state["portfolio"]["total_value"] = broker_state["portfolio"]["cash"] + total_market_value
            broker_state["portfolio"]["last_update"] = time.time()
            
            logger.info(f"Precios del portafolio actualizados. Valor total: {broker_state['portfolio']['total_value']}")
        else:
            logger.warning(f"No se pudieron actualizar los precios. Código de estado: {response.status_code}")
    except Exception as e:
        logger.error(f"Error actualizando precios: {e}")

async def get_stock_prediction(symbol):
    """Obtener predicción para una acción específica"""
    try:
        # Obtener predicciones desde el agente o usar el módulo market_utils
        # Aquí podemos usar la función _get_predictions del agente llama o implementar algo básico
        # Por simplicidad, devolvemos una predicción básica
        from market_utils import get_market_analyzer
        analyzer = get_market_analyzer()
        prediction = analyzer.get_price_prediction(symbol)
        return prediction
    except Exception as e:
        logger.error(f"Error obteniendo predicción para {symbol}: {e}")
        return {"prediction": 0, "confidence": 0, "direction": "neutral", "factors": []}

async def update_portfolio_metrics():
    """Actualizar métricas del portafolio"""
    portfolio = broker_state["portfolio"]
    metrics = broker_state["metrics"]
    
    # Calcular diversificación
    positions = portfolio["positions"]
    position_count = len(positions)
    
    # Rendimiento total (comparado con el efectivo inicial)
    initial_cash = portfolio["initial_cash"]
    current_value = portfolio["total_value"]
    total_return = ((current_value - initial_cash) / initial_cash) * 100 if initial_cash > 0 else 0
    
    # Ratio de efectivo
    cash_ratio = portfolio["cash"] / portfolio["total_value"] if portfolio["total_value"] > 0 else 1.0
    
    # Diversificación (indicador simple basado en número de posiciones)
    diversification = min(position_count / 10, 1.0) if position_count > 0 else 0
    
    # Actualizar métricas
    metrics["performance"]["total_return"] = total_return
    metrics["performance"]["cash_ratio"] = cash_ratio
    metrics["performance"]["positions_count"] = position_count
    
    # Actualizar métricas por acción
    stock_performance = {}
    for symbol, position in positions.items():
        # Calcular rendimiento por acción
        performance = {
            "symbol": symbol,
            "quantity": position["quantity"],
            "current_price": position["current_price"],
            "avg_cost": position["avg_cost"],
            "market_value": position["market_value"],
            "profit": position["market_value"] - (position["avg_cost"] * position["quantity"]),
            "profit_percent": ((position["current_price"] / position["avg_cost"]) - 1) * 100 if position["avg_cost"] > 0 else 0
        }
        
        # Agregar predicción
        prediction = await get_stock_prediction(symbol)
        performance["prediction"] = prediction["prediction"]
        performance["prediction_direction"] = prediction["direction"]
        
        stock_performance[symbol] = performance
    
    metrics["stock_performance"] = stock_performance
    
    metrics["risk_metrics"]["portfolio"]["diversification_score"] = diversification
    metrics["risk_metrics"]["portfolio"]["cash_ratio"] = cash_ratio
    metrics["risk_metrics"]["portfolio"]["total_value"] = portfolio["total_value"]
    
    logger.info(f"Métricas actualizadas. Rendimiento total: {total_return:.2f}%")
    return metrics

# Inicialización del servicio de broker sin posiciones previas
@app.on_event("startup")
async def startup_event():
    """Inicializar el servicio con datos iniciales vacíos"""
    logger.info("Iniciando servicio de broker...")
    
    # Reinicia completamente el estado del broker
    reset_broker_state()
    
    # Actualizar métricas iniciales
    await update_portfolio_metrics()
    
    logger.info(f"Portafolio inicializado con efectivo: {INITIAL_CASH}")
    logger.info(f"Valor total del portafolio: {broker_state['portfolio']['total_value']}")

# Endpoint para resetear el estado del broker (útil para pruebas)
@app.post("/reset")
async def reset_broker():
    """Resetea el estado del broker a valores iniciales"""
    reset_broker_state()
    await update_portfolio_metrics()
    return {"status": "success", "message": "Broker reiniciado correctamente"}

# Endpoint adicional para obtener resumen de rendimiento por acción
@app.get("/stock-performance")
async def get_stock_performance():
    """Obtener métricas de rendimiento detalladas por acción"""
    await update_portfolio_metrics()
    return broker_state["metrics"]["stock_performance"]

# Endpoint para chat con el broker IA
@app.post("/chat", response_model=ChatResponse)
async def chat_with_broker(chat_request: ChatRequest):
    """
    Endpoint para interactuar con el asistente IA del broker
    
    El broker IA analiza el mensaje y proporciona asistencia sobre
    estrategias de inversión, análisis de predicciones, y recomendaciones
    basadas en el portafolio y datos de mercado.
    """
    logger.info(f"Mensaje recibido: {chat_request.message}")
    
    # Convertir a ChatMessage para el agente
    chat_message = ChatMessage(
        message=chat_request.message,
        conversation_id=chat_request.conversation_id
    )
    
    # Procesar mensaje con el agente
    response = llama_agent.process_message(chat_message)
    
    logger.info(f"Respuesta generada para conversación {response.conversation_id}")
    
    # Publicar en Kafka si está disponible
    if producer:
        try:
            producer.send("broker_chat", {
                "message": chat_request.message,
                "response": response.response,
                "conversation_id": response.conversation_id,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error publicando chat a Kafka: {e}")
    
    return response

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """
    Obtener datos de mercado actualizados para un símbolo
    """
    try:
        # Primero intentamos obtener datos de mercado desde el agente
        market_data_response = llama_agent._get_market_data(symbol)
        
        if not market_data_response or "No pude obtener datos" in market_data_response:
            # Si el agente no tiene datos, intentamos obtenerlos desde la API
            try:
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        quote = data[0]
                        return {
                            "symbol": symbol,
                            "price": quote.get('price', 0),
                            "change": quote.get('change', 0),
                            "percentChange": quote.get('changesPercentage', 0),
                            "volume": quote.get('volume', 0),
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                logger.error(f"Error obteniendo datos de mercado para {symbol} desde API: {e}")
            
            # Si aún no hay datos, lanzamos la excepción
            raise HTTPException(status_code=404, detail=f"No hay datos disponibles para {symbol}")
        
        # Reformatear respuesta para que coincida con lo que espera el frontend
        return {
            "symbol": symbol,
            "price": float(market_data_response.get('price', 0)),
            "change": float(market_data_response.get('change', 0)),
            "percentChange": float(market_data_response.get('percentChange', 0)),
            "volume": int(market_data_response.get('volume', 0)),
            "timestamp": datetime.now().isoformat(),
            "analysis": market_data_response
        }
    except Exception as e:
        logger.error(f"Error en get_market_data para {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/predictions/{symbol}")
async def get_predictions(symbol: str):
    """
    Obtener predicciones y estrategias para un símbolo
    """
    # Obtener predicciones desde el agente
    predictions_response = llama_agent._get_predictions(symbol)
    
    if not predictions_response or "No pude obtener" in predictions_response:
        raise HTTPException(status_code=404, detail=f"No hay predicciones disponibles para {symbol}")
    
    return {"symbol": symbol, "predictions": predictions_response}

@app.get("/strategy/{symbol}")
async def get_strategy(symbol: str):
    """
    Obtener estrategia de inversión para un símbolo
    """
    # Generar estrategia desde el agente
    strategy_response = llama_agent._get_investment_strategy(symbol)
    
    if not strategy_response or "No pude generar" in strategy_response:
        raise HTTPException(status_code=404, detail=f"No se pudo generar estrategia para {symbol}")
    
    # Reformatear respuesta para que coincida con lo que espera el frontend
    basePrice = 0
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                basePrice = float(data[0]['price'])
    except Exception as e:
        logger.error(f"Error obteniendo precio para estrategia {symbol}: {e}")
        basePrice = 150  # Precio por defecto
    
    return {
        "symbol": symbol,
        "summary": strategy_response.get('summary', f"Análisis estratégico para {symbol}"),
        "recommendation": {
            "action": strategy_response.get('action', 'mantener'),
            "price": float(strategy_response.get('price', basePrice)),
            "quantity": int(strategy_response.get('quantity', 1)),
            "stopLoss": float(strategy_response.get('stopLoss', basePrice * 0.95)),
            "takeProfit": float(strategy_response.get('takeProfit', basePrice * 1.05)),
            "confidence": int(strategy_response.get('confidence', 75)),
            "timeframe": strategy_response.get('timeframe', "corto plazo")
        },
        "factors": strategy_response.get('factors', []),
        "technicalMetrics": strategy_response.get('technicalMetrics', {})
    }

@app.get("/historical/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1d"):
    """
    Obtener datos históricos para un símbolo
    """
    try:
        # Mapear timeframe a formato FMP API
        fmp_timeframe = "1min" if timeframe == "1m" else \
                       "5min" if timeframe == "5m" else \
                       "15min" if timeframe == "15m" else \
                       "30min" if timeframe == "30m" else \
                       "1hour" if timeframe == "1h" else \
                       "4hour" if timeframe == "4h" else "1day"
        
        # Determinar límite según timeframe
        limit = 60 if timeframe in ["1m", "5m", "15m"] else \
               48 if timeframe == "30m" else \
               24 if timeframe == "1h" else \
               30 if timeframe == "4h" else 30
        
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_timeframe}/{symbol}?apikey={FMP_API_KEY}&limit={limit}"
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"Error API FMP: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Error obteniendo datos históricos")
        
        data = response.json()
        
        # Formatear datos para que coincidan con lo que espera el frontend
        formatted_data = []
        for item in data:
            formatted_data.append({
                "date": item.get("date"),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "close": float(item.get("close", 0)),
                "volume": int(item.get("volume", 0))
            })
        
        return formatted_data
    except Exception as e:
        logger.error(f"Error obteniendo datos históricos para {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/model-status")
async def get_model_status():
    """
    Obtener estado de los modelos de ML
    """
    try:
        # Intentamos obtener estado real desde el servicio de streaming
        streaming_status = None
        try:
            response = requests.get("http://streaming:8090/health", timeout=5)
            if response.status_code == 200:
                streaming_status = response.json()
        except Exception as e:
            logger.warning(f"Error obteniendo estado desde streaming: {e}")
        
        # Generar estado de modelos con métricas detalladas
        now = datetime.now().isoformat()
        
        # Usar datos reales de streaming si disponibles
        online_status = "healthy"
        online_metrics = {}
        
        if streaming_status and 'metrics' in streaming_status:
            # Extraer métricas de salud desde el servicio de streaming
            metrics = streaming_status['metrics']
            errors = metrics.get('errores', 0)
            total = metrics.get('eventos_procesados', 100)
            
            # Determinar estado basado en tasa de errores
            if errors / total > 0.1:
                online_status = "critical"
            elif errors / total > 0.05:
                online_status = "degraded"
            
            online_metrics = {
                "MAPE": round(1.0 + (errors / total) * 10, 2),
                "RMSE": round(1.5 + (errors / total) * 15, 2),
                "accuracy": round(95 - (errors / total) * 100, 2)
            }
        else:
            # Métricas simuladas
            online_metrics = {
                "MAPE": round(1.0 + 0.5 * (0.5 - random.random()), 2),
                "RMSE": round(1.5 + 0.5 * (0.5 - random.random()), 2),
                "accuracy": round(90 + 5 * random.random(), 2)
            }
        
        return {
            "online": {
                "status": online_status,
                "accuracy": online_metrics.get("accuracy", 90),
                "metrics": online_metrics,
                "lastUpdated": now
            },
            "batch": {
                "status": ["healthy", "degraded", "critical"][random.randint(0, 2)],
                "accuracy": round(88 + 7 * random.random(), 2),
                "metrics": {
                    "MAPE": round(0.8 + 2 * random.random(), 2),
                    "RMSE": round(0.8 + 1.5 * random.random(), 2),
                    "accuracy": round(88 + 7 * random.random(), 2)
                },
                "lastUpdated": now
            },
            "ensemble": {
                "status": "healthy",
                "accuracy": round(90 + 5 * random.random(), 2),
                "metrics": {
                    "MAPE": round(0.5 + 1.5 * random.random(), 2),
                    "RMSE": round(0.5 + 1 * random.random(), 2),
                    "accuracy": round(90 + 5 * random.random(), 2)
                },
                "lastUpdated": now
            },
            "lastUpdated": now
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado de modelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    logger.info("Deteniendo servicio de broker...")
    
    # Cerrar agente Llama
    if llama_agent:
        try:
            llama_agent.shutdown()
            logger.info("Agente IA cerrado correctamente")
        except Exception as e:
            logger.error(f"Error cerrando agente IA: {e}")
    
    # Cerrar Kafka
    if producer:
        producer.close()
        logger.info("Kafka producer cerrado")

# Registrar función de limpieza para SIGTERM
atexit.register(lambda: llama_agent.shutdown() if llama_agent else None)