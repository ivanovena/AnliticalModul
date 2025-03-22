import logging
import requests
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import time
from config import FMP_API_KEY, FMP_BASE_URL, DB_URI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestionService")

# Conexión al datalake (base de datos)
engine = create_engine(DB_URI, pool_pre_ping=True)

def fetch_data(symbol, interval="1min"):
    url = f"{FMP_BASE_URL}/historical-chart/{interval}/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def store_data(symbol, data):
    sql = text("""
        INSERT INTO market_data (symbol, datetime, open, high, low, close, volume)
        VALUES (:symbol, :datetime, :open, :high, :low, :close, :volume)
        ON CONFLICT (symbol, datetime) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
    """)
    with engine.begin() as conn:
        for entry in data:
            conn.execute(sql, {
                "symbol": symbol,
                "datetime": entry.get("date"),
                "open": entry.get("open"),
                "high": entry.get("high"),
                "low": entry.get("low"),
                "close": entry.get("close"),
                "volume": entry.get("volume")
            })

def main():
    symbols = ["AAPL", "GOOGL", "MSFT"]
    interval = "1min"
    for symbol in symbols:
        logger.info(f"Procesando ingesta para {symbol}")
        data = fetch_data(symbol, interval)
        if data:
            store_data(symbol, data)
            logger.info(f"Datos almacenados para {symbol}")
        else:
            logger.warning(f"No se recibieron datos para {symbol}")
        time.sleep(1)

if __name__ == "__main__":
    main()
