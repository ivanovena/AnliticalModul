import logging
from kafka import KafkaConsumer, KafkaProducer
import json
import time
from river import linear_model, preprocessing
from config import KAFKA_BROKER, INGESTION_TOPIC, STREAMING_TOPIC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("StreamingService")

consumer = KafkaConsumer(
    INGESTION_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset='earliest',
    group_id="streaming-group",
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

online_model = preprocessing.StandardScaler() | linear_model.LinearRegression()

def process_event(event):
    symbol = event.get("symbol")
    # Se deben usar datos reales; aquí se usa un ejemplo de features
    sample_features = {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000000}
    target = sample_features["close"] * 1.01
    online_model.learn_one(sample_features, target)
    prediction = online_model.predict_one(sample_features)
    return symbol, prediction

def main():
    logger.info("Servicio streaming iniciado.")
    for message in consumer:
        event = message.value
        logger.info(f"Evento recibido: {event}")
        try:
            symbol, prediction = process_event(event)
            streaming_event = {"service": "streaming", "symbol": symbol, "prediction": prediction, "timestamp": time.time()}
            producer.send(STREAMING_TOPIC, streaming_event)
            producer.flush()
            logger.info(f"Evento streaming publicado para {symbol}")
        except Exception as e:
            logger.error(f"Error procesando el evento: {e}")

if __name__ == "__main__":
    main()
