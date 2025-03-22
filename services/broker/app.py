import logging
import json
import time
import threading
import os
from fastapi import FastAPI, HTTPException
import uvicorn
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel
from threading import Lock
from config import KAFKA_BROKER, AGENT_TOPIC, INITIAL_CASH, TELEGRAM_BOT_TOKEN, FMP_API_KEY, USE_DATALAKE
import requests

# Importaciones para gestión de memoria y autoaprendizaje
import numpy as np
from sklearn.linear_model import SGDRegressor
import joblib
import spacy
import faiss

# Importaciones para motor conversacional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Importaciones para Telegram Bot
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Importaciones para análisis de sentimiento
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BrokerService")

app = FastAPI(
    title="Broker Trading Simulator Avanzado",
    description=("Simulador de trading avanzado con integración real de datos FMP, "
                 "gestión avanzada de riesgos, memoria vectorial con FAISS, motor conversacional Transformer, "
                 "autoentrenamiento continuo y comunicación multicanal (REST + Telegram)."),
    version="3.0"
)

portfolio = {
    "cash": INITIAL_CASH,
    "positions": {}
}
orders = []

portfolio_lock = Lock()

conversation_history_file = "conversation_history.json"
if os.path.exists(conversation_history_file):
    with open(conversation_history_file, "r") as f:
        conversation_history = json.load(f)
else:
    conversation_history = []

embeddings_list = []
feedback_list = []

learned_params = {
    "buy_threshold": 100.0,
    "sell_threshold": 100.0,
    "strategy_weight": 0.5
}

# --- Integración con FMP API o Datalake ---
def get_stock_quote(symbol: str):
    if USE_DATALAKE:
         url = f"http://datalake-service/quote/{symbol}"
    else:
         url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data:
        return data[0]
    else:
        raise Exception(f"No se encontraron datos para {symbol}")

def get_historical_prices(symbol: str):
    if USE_DATALAKE:
         url = f"http://datalake-service/historical/{symbol}"
    else:
         url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if "historical" in data:
        return [entry["close"] for entry in data["historical"]]
    else:
        raise Exception(f"No se encontraron datos históricos para {symbol}")

def get_stock_sentiment(symbol: str):
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit=5&apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    news_data = response.json()
    if not news_data:
        return 0.0
    sentiments = [sia.polarity_scores(news.get("title", ""))["compound"] for news in news_data]
    return np.mean(sentiments)

# --- Gestión Avanzada de Riesgos ---
def advanced_risk_management(num_simulations=5000, confidence_level=0.95):
    risk_results = {}
    symbols = list(portfolio["positions"].keys())
    for symbol in symbols:
        try:
            prices = get_historical_prices(symbol)
            prices = np.array(prices)
            returns = np.diff(prices) / prices[:-1]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            cvar = -np.mean(simulated_returns[simulated_returns <= -var])
            sentiment = get_stock_sentiment(symbol)
            risk_results[symbol] = {"VaR": var, "CVaR": cvar, "sentiment": sentiment}
        except Exception as e:
            logger.error("Error calculando riesgo para %s: %s", symbol, e)
            risk_results[symbol] = {"VaR": None, "CVaR": None, "sentiment": None}
    return risk_results

# --- Módulo de Memoria y Aprendizaje con FAISS ---
class MemoryManager:
    def __init__(self, model_path="memory_model.pkl"):
        self.nlp = spacy.load("en_core_web_md")
        self.regressor = SGDRegressor(max_iter=1000, tol=1e-3)
        self.model_path = model_path
        self.dim = self.nlp.vocab.vectors_length
        self.index = faiss.IndexFlatL2(self.dim)
        if os.path.exists(self.model_path):
            self.regressor = joblib.load(self.model_path)
    
    def add_conversation(self, message, response, feedback):
        combined_text = message + " " + response
        doc = self.nlp(combined_text)
        vector = doc.vector.astype(np.float32)
        embeddings_list.append(vector)
        feedback_list.append(feedback)
        self.index.add(np.expand_dims(vector, axis=0))
    
    def update_model(self):
        if len(embeddings_list) >= 5:
            X = np.array(embeddings_list)
            y = np.array(feedback_list)
            self.regressor.partial_fit(X, y)
            joblib.dump(self.regressor, self.model_path)
    
    def predict_adjustment(self, text):
        doc = self.nlp(text)
        vector = doc.vector.astype(np.float32).reshape(1, -1)
        adjustment = self.regressor.predict(vector)[0]
        return adjustment

memory_manager = MemoryManager()

# --- Motor Conversacional basado en Transformers ---
class ConversationalEngine:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate_response(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    
    def fine_tune(self, training_data, num_train_epochs=1, per_device_train_batch_size=1):
        from datasets import Dataset
        dataset = Dataset.from_list(training_data)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir="./finetuned_model",
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=10,
            logging_steps=5,
            learning_rate=5e-5,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
        self.model.save_pretrained("./finetuned_model")
        self.tokenizer.save_pretrained("./finetuned_model")
        return "Fine-tuning completado"

conversational_engine = ConversationalEngine()

# --- Autoentrenamiento Continuo ---
def auto_train_parameters():
    with portfolio_lock:
        total_value = portfolio["cash"]
        for order in orders:
            total_value += order.get("price", 0) * order.get("quantity", 0)
    performance_factor = (total_value - INITIAL_CASH) / INITIAL_CASH
    learned_params["buy_threshold"] *= (1 - 0.05 * performance_factor)
    learned_params["sell_threshold"] *= (1 + 0.05 * performance_factor)
    logger.info("Autoentrenamiento realizado. Nuevos parámetros: %s", learned_params)

# --- Definición de Modelos de Datos para API ---
class OrderRequest(BaseModel):
    symbol: str
    action: str  # "BUY" o "SELL"
    quantity: int
    price: float

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: float  # Rango: -1.0 a 1.0
    comment: str = None

class PlanRequest(BaseModel):
    predictions: dict  # Ej: {"AAPL": 5.0, "GOOGL": 3.2}

class TrainConversationRequest(BaseModel):
    prompt: str
    target: str

# --- Kafka Producer ---
kafka_producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- Funciones de Ejecución de Órdenes ---
def execute_order(symbol: str, action: str, quantity: int, price: float, source="manual"):
    with portfolio_lock:
        if action.upper() == "BUY":
            cost = quantity * price
            if portfolio["cash"] >= cost:
                portfolio["cash"] -= cost
                portfolio["positions"][symbol] = portfolio["positions"].get(symbol, 0) + quantity
                order = {
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "price": price,
                    "timestamp": time.time(),
                    "source": source
                }
                orders.append(order)
                logger.info("Orden de COMPRA ejecutada: %s", order)
                return order
            else:
                logger.warning("Efectivo insuficiente para comprar %s", symbol)
                raise Exception("Efectivo insuficiente")
        elif action.upper() == "SELL":
            if portfolio["positions"].get(symbol, 0) >= quantity:
                portfolio["positions"][symbol] -= quantity
                proceeds = quantity * price
                portfolio["cash"] += proceeds
                order = {
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": quantity,
                    "price": price,
                    "timestamp": time.time(),
                    "source": source
                }
                orders.append(order)
                logger.info("Orden de VENTA ejecutada: %s", order)
                return order
            else:
                logger.warning("Acciones insuficientes para vender %s", symbol)
                raise Exception("Acciones insuficientes")
        else:
            raise Exception("Acción inválida")

# --- Endpoints de la API REST ---
@app.get("/portfolio")
def get_portfolio():
    with portfolio_lock:
        return portfolio

@app.get("/orders")
def get_orders():
    return orders

@app.post("/order")
def place_order(order_req: OrderRequest):
    try:
        order = execute_order(order_req.symbol, order_req.action, order_req.quantity, order_req.price)
        return {"status": "success", "order": order}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def get_metrics():
    with portfolio_lock:
        total_positions_value = 0
        for symbol, qty in portfolio["positions"].items():
            if qty > 0:
                try:
                    quote = get_stock_quote(symbol)
                    current_price = quote.get("price", 0)
                except Exception as e:
                    logger.error("Error obteniendo cotización para %s: %s", symbol, e)
                    current_price = 0
                total_positions_value += qty * current_price
        total_value = portfolio["cash"] + total_positions_value
        risk_metrics = advanced_risk_management()
        return {"total_value": total_value, "cash": portfolio["cash"], "positions_value": total_positions_value, "risk_metrics": risk_metrics}

@app.post("/chat")
def chat_with_broker(chat_req: ChatRequest):
    message = chat_req.message.strip().lower()
    response = ""
    
    if "plan" in message or "inversión" in message:
        response = (
            "Recomiendo diversificar en AAPL y GOOGL. "
            f"Compra si la ganancia esperada supera {learned_params['buy_threshold']}% y vende si cae por debajo de {learned_params['sell_threshold']}%."
        )
    elif "estrategia" in message:
        response = (
            f"La estrategia usa umbrales dinámicos: comprar si supera {learned_params['buy_threshold']} y vender si baja de {learned_params['sell_threshold']}. "
            "¿Deseas ajustar estos parámetros?"
        )
    elif "capital" in message or "efectivo" in message:
        with portfolio_lock:
            response = f"El efectivo disponible es ${portfolio['cash']:.2f}."
    elif "posición" in message:
        with portfolio_lock:
            response = f"Las posiciones actuales son: {portfolio['positions']}."
    elif "recomendación" in message:
        response = "Recomiendo mantener una estrategia equilibrada y revisar periódicamente métricas y riesgos."
    else:
        response = "No tengo respuesta específica. Pregunta sobre estrategias, capital o posiciones."
    
    try:
        adjustment = memory_manager.predict_adjustment(message)
        learned_params["buy_threshold"] += 0.1 * adjustment
        learned_params["sell_threshold"] -= 0.1 * adjustment
        response += f" (Ajustado: buy_threshold={learned_params['buy_threshold']:.2f}, sell_threshold={learned_params['sell_threshold']:.2f})"
    except Exception as e:
        logger.error("Error en ajuste de memoria: %s", e)
    
    try:
        prompt = f"Usuario: {message}\nBroker:"
        generated = conversational_engine.generate_response(prompt)
        response = response + " " + generated
    except Exception as e:
        logger.error("Error en generación Transformer: %s", e)
    
    conversation_entry = {"message": message, "response": response, "timestamp": time.time()}
    conversation_history.append(conversation_entry)
    with open(conversation_history_file, "w") as f:
        json.dump(conversation_history, f)
    memory_manager.add_conversation(message, response, 0.0)
    
    logger.info("Chat: '%s' -> Respuesta: '%s'", message, response)
    return {"response": response, "conversation_id": len(conversation_history) - 1}

@app.post("/feedback")
def feedback(feedback_req: FeedbackRequest):
    conv_id = feedback_req.conversation_id
    rating = feedback_req.rating
    comment = feedback_req.comment or ""
    
    try:
        conversation = conversation_history[conv_id]
        memory_manager.add_conversation(conversation["message"], conversation["response"], rating)
        memory_manager.update_model()
        logger.info("Feedback recibido para conv %s: rating=%s, comentario=%s", conv_id, rating, comment)
        auto_train_parameters()
    except Exception as e:
        logger.error("Error en feedback: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    
    return {"status": "success", "learned_params": learned_params}

@app.get("/learned_params")
def get_learned_params():
    return learned_params

@app.post("/plan")
def generate_plan(plan_req: PlanRequest):
    plan = {}
    for symbol, gain in plan_req.predictions.items():
        if gain > learned_params["buy_threshold"]:
            plan[symbol] = "BUY"
        elif gain < learned_params["sell_threshold"]:
            plan[symbol] = "SELL"
        else:
            plan[symbol] = "HOLD"
    logger.info("Plan generado: %s", plan)
    return {"investment_plan": plan}

@app.post("/train_conversation")
def train_conversation(train_req: TrainConversationRequest):
    training_data = [{"input_text": train_req.prompt, "target_text": train_req.target}]
    try:
        result = conversational_engine.fine_tune(training_data, num_train_epochs=1)
        logger.info("Fine-tuning completado: %s", training_data)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error("Error en fine-tuning: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

def kafka_consumer_thread():
    consumer = KafkaConsumer(
        AGENT_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='earliest',
        group_id="broker-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    logger.info("Kafka consumer iniciado.")
    for message in consumer:
        decision = message.value
        symbol = decision.get("symbol")
        action = decision.get("action")
        price = decision.get("prediction")
        quantity = 10
        logger.info("Decisión Kafka: %s", decision)
        try:
            order = execute_order(symbol, action, quantity, price, source="agent")
            kafka_producer.send("broker_orders", order)
            kafka_producer.flush()
        except Exception as e:
            logger.error("Error en Kafka para %s: %s", symbol, e)

threading.Thread(target=kafka_consumer_thread, daemon=True).start()

def start_telegram_bot():
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
    
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    
    def start(update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text="Hola, soy el Broker Inteligente. ¿Cómo puedo ayudarte?")
    
    def help_command(update, context):
        help_text = ("Comandos disponibles:\n"
                     "/portfolio - Ver portafolio\n"
                     "/orders - Ver órdenes\n"
                     "/chat <mensaje> - Conversar con el broker\n"
                     "/plan - Generar plan de inversión (requiere JSON de predicciones)\n"
                     "/feedback <id> <rating> <comentario> - Enviar feedback\n"
                     "/train_conversation - Entrenar el motor conversacional")
        context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)
    
    def portfolio_command(update, context):
        with portfolio_lock:
            text = f"Portafolio:\nEfectivo: ${portfolio['cash']:.2f}\nPosiciones: {portfolio['positions']}"
        context.bot.send_message(chat_id=update.effective_chat.id, text=text)
    
    def chat_command(update, context):
        message = " ".join(context.args)
        if not message:
            context.bot.send_message(chat_id=update.effective_chat.id, text="Envía un mensaje después de /chat.")
            return
        from fastapi.encoders import jsonable_encoder
        chat_req = ChatRequest(message=message)
        result = chat_with_broker(chat_req)
        context.bot.send_message(chat_id=update.effective_chat.id, text=result["response"])
    
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("portfolio", portfolio_command))
    dispatcher.add_handler(CommandHandler("chat", chat_command, pass_args=True))
    
    updater.start_polling()
    logger.info("Telegram Bot iniciado.")
    updater.idle()

if TELEGRAM_BOT_TOKEN:
    threading.Thread(target=start_telegram_bot, daemon=True).start()
else:
    logger.warning("TELEGRAM_BOT_TOKEN no definido. Telegram Bot no se iniciará.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
