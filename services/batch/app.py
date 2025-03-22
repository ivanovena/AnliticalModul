import logging
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from kafka import KafkaProducer
import json
import time
import joblib
import os
from config import DB_URI, KAFKA_BROKER, KAFKA_TOPIC, MODEL_OUTPUT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchService")

engine = create_engine(DB_URI, pool_pre_ping=True)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def load_data(symbol):
    query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' ORDER BY datetime ASC"
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Datos cargados para {symbol}: {len(df)} registros.")
        return df
    except Exception as e:
        logger.error(f"Error cargando datos para {symbol}: {e}")
        return pd.DataFrame()

def train_model(df):
    if df.empty:
        logger.error("El DataFrame está vacío, no se puede entrenar el modelo.")
        return None
    df['target'] = df['close'].shift(-1)
    df = df.dropna()
    features = df[['open', 'high', 'low', 'close', 'volume']]
    target = df['target']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    logger.info("Modelo entrenado correctamente.")
    return model

def save_model(model, symbol):
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_OUTPUT_PATH, f"{symbol}_model.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Modelo guardado en {model_file}")
    return model_file

def publish_offline_update(symbol, model_file):
    event = {"service": "batch", "symbol": symbol, "model_file": model_file, "timestamp": time.time()}
    producer.send("offline_model_updates", event)
    producer.flush()
    logger.info(f"Evento offline publicado para {symbol}")

def main():
    symbols = ["AAPL", "GOOGL", "MSFT"]
    for symbol in symbols:
        logger.info(f"Iniciando entrenamiento para {symbol}")
        df = load_data(symbol)
        if df.empty:
            logger.error(f"No se pudieron cargar datos para {symbol}")
            continue
        model = train_model(df)
        if model:
            model_file = save_model(model, symbol)
            publish_offline_update(symbol, model_file)
        time.sleep(1)
    logger.info("Proceso batch finalizado.")

if __name__ == "__main__":
    main()
