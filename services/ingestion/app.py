import time
import logging
import os
import json
import requests
import threading
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from kafka import KafkaProducer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/app/ingestion_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Variables de entorno
FMP_API_KEY = os.environ.get('FMP_API_KEY')
FMP_BASE_URL = os.environ.get('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka:9092') 
INGESTION_TOPIC = os.environ.get('INGESTION_TOPIC', 'ingestion_events')
DB_URI = os.environ.get('DB_URI', 'postgresql://market_admin:password@postgres:5432/market_data')

# Crear conexión a PostgreSQL
engine = create_engine(DB_URI)

# Crear productor Kafka
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    logger.info(f"Kafka producer initialized with broker {KAFKA_BROKER}")
except Exception as e:
    logger.error(f"Error initializing Kafka producer: {e}")
    producer = None

class IngestionManager:
    """Gestor de ingesta de datos financieros"""
    
    def __init__(self, api_key, base_url, db_engine, kafka_producer=None):
        self.api_key = api_key
        self.base_url = base_url
        self.db_engine = db_engine
        self.kafka_producer = kafka_producer
        # Caché para las últimas fechas de datos (evita consultas repetidas a la BD)
        self.last_dates_cache = {}
        # Tiempo de expiración de la caché en segundos (30 minutos)
        self.cache_expiry = 30 * 60
        # Última actualización de la caché
        self.cache_last_update = {}
        logger.info("IngestionManager initialized")
    
    def get_last_date_for_symbol_and_type(self, symbol, data_type):
        """Obtener la fecha más reciente para un símbolo y tipo de datos específico"""
        cache_key = f"{symbol}_{data_type}"
        
        # Verificar si tenemos un valor en caché que no haya expirado
        current_time = time.time()
        if (cache_key in self.last_dates_cache and 
            cache_key in self.cache_last_update and 
            current_time - self.cache_last_update[cache_key] < self.cache_expiry):
            logger.debug(f"Using cached last date for {symbol} {data_type}")
            return self.last_dates_cache[cache_key]
        
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT MAX(datetime) as last_date 
                    FROM market_data 
                    WHERE symbol = :symbol AND data_type = :data_type
                """)
                
                result = conn.execute(query, {"symbol": symbol, "data_type": data_type}).fetchone()
                
                if result and result[0]:
                    last_date = result[0]
                    if isinstance(last_date, str):
                        last_date = datetime.fromisoformat(last_date.replace('Z', '+00:00'))
                    
                    # Actualizar caché
                    self.last_dates_cache[cache_key] = last_date
                    self.cache_last_update[cache_key] = current_time
                    
                    logger.info(f"Last date for {symbol} {data_type}: {last_date}")
                    return last_date
                else:
                    logger.info(f"No previous data found for {symbol} {data_type}")
                    # Actualizar caché con None para evitar consultas repetidas
                    self.last_dates_cache[cache_key] = None
                    self.cache_last_update[cache_key] = current_time
                    return None
                
        except Exception as e:
            logger.error(f"Error getting last date for {symbol} {data_type}: {e}")
            return None
    
    def check_record_exists(self, symbol, data_type, datetime_str):
        """Verificar si un registro específico ya existe en la base de datos"""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT COUNT(*) 
                    FROM market_data 
                    WHERE symbol = :symbol AND data_type = :data_type AND datetime = :datetime
                """)
                
                result = conn.execute(query, {
                    "symbol": symbol, 
                    "data_type": data_type,
                    "datetime": datetime_str
                }).fetchone()
                
                return result[0] > 0
                
        except Exception as e:
            logger.error(f"Error checking record existence for {symbol} {data_type} {datetime_str}: {e}")
            return False
    
    def fetch_real_time_quote(self, symbol):
        """Obtener cotización en tiempo real"""
        # Para evitar superar límites de la API, usamos datos simulados
        # en caso de entorno de desarrollo
        if os.environ.get('USE_DATALAKE') == 'true':
            # Aquí se implementaría la lógica para usar datos locales
            pass
        
        try:
            url = f"{self.base_url}/quote/{symbol}?apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    quote_data = data[0]
                    
                    # Convertir a formato estándar
                    quote = {
                        'symbol': symbol,
                        'price': quote_data.get('price'),
                        'change': quote_data.get('change'),
                        'change_percent': quote_data.get('changesPercentage'),
                        'volume': quote_data.get('volume'),
                        'timestamp': int(time.time()),
                        'datetime': datetime.now().isoformat(),
                        'data_type': 'real_time',
                        'open': quote_data.get('open'),
                        'high': quote_data.get('dayHigh'),
                        'low': quote_data.get('dayLow'),
                        'close': quote_data.get('previousClose'),
                        'market_cap': quote_data.get('marketCap')
                    }
                    return quote
                else:
                    logger.warning(f"Empty data returned for symbol {symbol}")
            else:
                logger.error(f"Error fetching quote for {symbol}: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Exception fetching quote for {symbol}: {e}")
        
        return None
    
    def fetch_intraday_data(self, symbol, interval, start_date, end_date):
        """Obtener datos históricos intradía"""
        try:
            # Convertir fechas a formato de string yyyy-MM-dd
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/historical-chart/{interval}/{symbol}?from={start_str}&to={end_str}&apikey={self.api_key}"
            logger.info(f"Fetching {interval} data for {symbol} from {start_str} to {end_str} with URL: {url}")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    # Procesar datos
                    processed_data = []
                    data_type = f"intraday_{interval}"
                    
                    # Obtener la última fecha en la BD para este símbolo y tipo
                    last_date = self.get_last_date_for_symbol_and_type(symbol, data_type)
                    
                    for item in data:
                        # Determinar el data_type correcto basado en el intervalo
                        datetime_str = item.get('date')
                        record_timestamp = int(datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').timestamp())
                        record_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                        
                        # Si tenemos una fecha más reciente en la BD, saltamos este registro
                        if last_date and record_datetime <= last_date:
                            logger.debug(f"Skipping {interval} data point for {symbol} at {datetime_str} (already in DB)")
                            continue
                            
                        # Verificar si este registro específico ya existe
                        if self.check_record_exists(symbol, data_type, datetime_str):
                            logger.debug(f"Skipping duplicate {interval} data point for {symbol} at {datetime_str}")
                            continue
                        
                        record = {
                            'symbol': symbol,
                            'price': item.get('close'),  # Usamos close como precio actual
                            'change': None,  # No disponible en datos históricos
                            'change_percent': None,  # No disponible en datos históricos
                            'volume': item.get('volume'),
                            'timestamp': record_timestamp,
                            'datetime': datetime_str,
                            'data_type': data_type,
                            'open': item.get('open'),
                            'high': item.get('high'),
                            'low': item.get('low'),
                            'close': item.get('close'),
                            'market_cap': None  # No disponible en datos históricos
                        }
                        processed_data.append(record)
                    
                    logger.info(f"Successfully processed {len(processed_data)} NEW {interval} data points for {symbol}")
                    return processed_data
                else:
                    logger.warning(f"Empty data returned for {symbol} at {interval} interval")
            else:
                logger.error(f"Error fetching {interval} data for {symbol}: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Exception fetching {interval} data for {symbol}: {e}")
        
        return None
    
    def store_quote(self, quote):
        """Almacenar cotización en tiempo real en la BD y publicar en Kafka"""
        try:
            # Verificar si ya existe una cotización con la misma marca de tiempo (aproximadamente)
            with self.db_engine.connect() as conn:
                # Buscamos cotizaciones en un rango de 1 segundo alrededor de la marca de tiempo actual
                query = text("""
                    SELECT COUNT(*) FROM market_data
                    WHERE symbol = :symbol 
                    AND data_type = 'real_time'
                    AND timestamp BETWEEN :timestamp - 1 AND :timestamp + 1
                """)
                
                result = conn.execute(query, {
                    "symbol": quote['symbol'],
                    "timestamp": quote['timestamp']
                }).fetchone()
                
                if result[0] > 0:
                    logger.debug(f"Skipping duplicate real-time quote for {quote['symbol']}")
                    return True
            
            # Insertar en PostgreSQL
            with self.db_engine.begin() as conn:  # begin() crea una transacción y hace commit al salir del bloque
                query = text("""
                    INSERT INTO market_data 
                    (symbol, price, change, change_percent, volume, timestamp, datetime, data_type, open, high, low, close, market_cap)
                    VALUES 
                    (:symbol, :price, :change, :change_percent, :volume, :timestamp, :datetime, :data_type, :open, :high, :low, :close, :market_cap)
                """)
                
                conn.execute(query, quote)
                # No necesitamos hacer commit aquí, se hace automáticamente al salir del bloque with
            
            # Publicar en Kafka si está disponible
            if self.kafka_producer:
                self.kafka_producer.send(INGESTION_TOPIC, quote)
            
            logger.info(f"Stored and published quote for {quote['symbol']}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing quote for {quote['symbol']}: {e}")
            return False
    
    def store_intraday_data(self, data):
        """Almacenar datos históricos intradía en lote"""
        if not data or len(data) == 0:
            return False
        
        try:
            # Insertar en PostgreSQL en lote
            with self.db_engine.begin() as conn:  # begin() crea una transacción y hace commit al salir del bloque
                query = text("""
                    INSERT INTO market_data 
                    (symbol, price, change, change_percent, volume, timestamp, datetime, data_type, open, high, low, close, market_cap)
                    VALUES 
                    (:symbol, :price, :change, :change_percent, :volume, :timestamp, :datetime, :data_type, :open, :high, :low, :close, :market_cap)
                    ON CONFLICT DO NOTHING
                """)
                
                # Ejecutar en lote
                errors = 0
                for record in data:
                    try:
                        conn.execute(query, record)
                    except Exception as e:
                        errors += 1
                        logger.error(f"Error inserting record: {e}")
                
                # No necesitamos hacer commit aquí, se hace automáticamente al salir del bloque with
            
            # Publicar en Kafka en lote si está disponible
            if self.kafka_producer:
                for record in data:
                    self.kafka_producer.send(INGESTION_TOPIC, record)
            
            logger.info(f"Stored {len(data) - errors} data points in database, {errors} errors")
            return True
        
        except Exception as e:
            logger.error(f"Error storing batch data: {e}")
            return False

# Crear una instancia del gestor de ingesta
ingestion_manager = IngestionManager(
    api_key=FMP_API_KEY,
    base_url=FMP_BASE_URL,
    db_engine=engine,
    kafka_producer=producer
)

def ingestion_thread():
    """Background thread to run the ingestion process"""
    symbols = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
        "META", "NVDA", "NFLX", "INTC", "AMD",
        "IAG.MC", "PHM.MC", "AENA.MC", "BA", 
        "CAR", "DLTR", "SASA.IS"
    ]
    
    # Configuración de intervalos respetando límites de FMP Starter Plan
    intraday_config = {
        '1min': {'max_days': 3, 'target_days': 30, 'interval_seconds': 15*60},
        '5min': {'max_days': 10, 'target_days': 30, 'interval_seconds': 30*60},
        '15min': {'max_days': 45, 'target_days': 30, 'interval_seconds': 60*60},
        '30min': {'max_days': 30, 'target_days': 30, 'interval_seconds': 2*60*60},
        '45min': {'max_days': 45, 'target_days': 30, 'interval_seconds': 3*60*60},
        '1hour': {'max_days': 90, 'target_days': 30, 'interval_seconds': 4*60*60}
    }
    
    while True:
        current_time = time.time()
        
        for symbol in symbols:
            try:
                # Fetch real-time quote
                quote = ingestion_manager.fetch_real_time_quote(symbol)
                if quote:
                    ingestion_manager.store_quote(quote)
                
                # Procesamiento de intervalos históricos
                for interval, config in intraday_config.items():
                    max_days = config['max_days']
                    target_days = config['target_days']
                    data_type = f"intraday_{interval}"
                    
                    # Obtener la última fecha para este símbolo y tipo de datos
                    last_date = ingestion_manager.get_last_date_for_symbol_and_type(symbol, data_type)
                    
                    # Calcular fecha de inicio según si tenemos datos previos o no
                    end_date = datetime.now() - timedelta(days=1)  # Ayer
                    
                    if last_date:
                        # Si tenemos datos, comenzar desde el último dato + 1 minuto
                        # (para evitar solapamientos y duplicados)
                        start_date = last_date + timedelta(minutes=1)
                        logger.info(f"Incremental fetch for {symbol} {data_type} from {start_date}")
                        
                        # Si la fecha de inicio es posterior a la fecha de fin, no hay nada que hacer
                        if start_date > end_date:
                            logger.info(f"No new data needed for {symbol} {data_type}, already up to date")
                            continue
                    else:
                        # Si no tenemos datos, usar la configuración predeterminada
                        start_date = end_date - timedelta(days=max_days)
                        logger.info(f"Initial fetch for {symbol} {data_type} from {start_date}")
                    
                    data = ingestion_manager.fetch_intraday_data(
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if data and len(data) > 0:
                        ingestion_manager.store_intraday_data(data)
                        logger.info(f"Stored {len(data)} {interval} data points for {symbol}")
                    
                time.sleep(1)  # Pequeña pausa entre símbolos
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
        
        time.sleep(30)  # Pausa entre ciclos de ingesta

# Crear servidor HTTP para endpoints de salud
from fastapi import FastAPI, HTTPException
import uvicorn
from threading import Thread

app = FastAPI()

@app.get("/health")
async def health_check():
    """Verificar el estado del servicio"""
    db_healthy = True
    kafka_healthy = True
    
    # Verificar conexión a BD
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database health check: healthy")
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False
    
    # Verificar conexión a Kafka
    try:
        if producer:
            producer.send('test_health_topic', {'test': 'data'})
            logger.info("Kafka health check: healthy")
        else:
            logger.warning("Kafka producer not available")
            kafka_healthy = False
    except Exception as e:
        logger.error(f"Kafka health check failed: {e}")
        kafka_healthy = False
    
    if db_healthy and kafka_healthy:
        logger.info("Health check passed")
        return {"status": "healthy"}
    else:
        logger.error("Health check failed")
        raise HTTPException(status_code=500, detail="Service unhealthy")

# Iniciar el thread de ingestion y el servidor HTTP
if __name__ == "__main__":
    # Iniciar thread de ingesta
    thread = Thread(target=ingestion_thread, daemon=True)
    thread.start()
    logger.info("Ingestion thread started")
    
    # Iniciar servidor HTTP
    uvicorn.run(app, host="0.0.0.0", port=8080)