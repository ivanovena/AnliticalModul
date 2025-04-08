import time
import logging
import os
import json
import requests
import threading
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from kafka import KafkaProducer
import math
from threading import Thread
import asyncio
import websockets

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
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx')
FMP_BASE_URL = os.environ.get('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')
FMP_STABLE_URL = os.environ.get('FMP_STABLE_URL', 'https://financialmodelingprep.com/stable')
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka:9092') 
INGESTION_TOPIC = os.environ.get('INGESTION_TOPIC', 'ingestion_events')
REALTIME_TOPIC = os.environ.get('REALTIME_TOPIC', 'realtime_events')  # Nuevo topic para datos en tiempo real
DB_URI = os.environ.get('DB_URI', 'postgresql://postgres:postgres@postgres:5432/trading')
REALTIME_INTERVAL_SECONDS = int(os.environ.get('REALTIME_INTERVAL_SECONDS', 300))  # 5 minutos por defecto

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

# --- WebSocket Server --- 
connected_clients = set()

async def notify_clients(message):
    if connected_clients:
        # Ensure the loop is running when sending messages
        loop = asyncio.get_running_loop()
        await asyncio.wait([loop.create_task(client.send(message)) for client in connected_clients])

async def websocket_handler(websocket):
    connected_clients.add(websocket)
    logger.info(f"Cliente WebSocket conectado: {websocket.remote_address}")
    try:
        async for message in websocket:
            # No se espera que los clientes envíen mensajes en este caso
            logger.info(f"Mensaje recibido de WS (ignorado): {message}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Cliente WebSocket desconectado: {websocket.remote_address}")
    finally:
        connected_clients.remove(websocket)

async def start_websocket_server():
    host = os.environ.get('WEBSOCKET_HOST', '0.0.0.0')
    port = int(os.environ.get('WEBSOCKET_PORT', 8080))
    server = await websockets.serve(websocket_handler, host, port)
    logger.info(f"Servidor WebSocket escuchando en {host}:{port}")
    await server.wait_closed()
# --- Fin WebSocket Server ---

class IngestionManager:
    """Gestor de ingesta de datos financieros"""
    
    def __init__(self, api_key, base_url, db_engine):
        self.api_key = api_key if api_key else 'h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx'
        self.base_url = base_url if base_url else 'https://financialmodelingprep.com/api/v3'
        self.stable_url = FMP_STABLE_URL
        self.db_engine = db_engine
        self.kafka_producer = producer
        # Caché para las últimas fechas de datos (evita consultas repetidas a la BD)
        self.last_dates_cache = {}
        # Tiempo de expiración de la caché en segundos (30 minutos)
        self.cache_expiry = 30 * 60
        # Última actualización de la caché
        self.cache_last_update = {}
        # Para evitar procesar duplicados en tiempo real
        self.last_processed_timestamps = {}
        logger.info(f"IngestionManager initialized with API_KEY: {'*****' if self.api_key else 'None'}, BASE_URL: {self.base_url}")
    
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
    
    # NUEVA FUNCIÓN: Obtener el último punto de datos de 5 minutos (tiempo casi real)
    def fetch_latest_5min_data(self, symbol):
        """
        Obtiene sólo el último punto de datos de 5 minutos para un símbolo desde la API FMP.
        Esta función se utiliza para alimentar el stream de eventos "en tiempo real".
        """
        try:
            # Usar el endpoint /stable/ que sabemos que funciona para intervalo de 5min
            url = f"{self.stable_url}/historical-chart/5min?symbol={symbol}&apikey={self.api_key}&limit=1"
            logger.debug(f"Fetching latest 5min data for {symbol} from URL: {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Lanzar excepción para códigos de error HTTP
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No 5min data returned for {symbol} or empty response")
                return None
            
            # Tomar sólo el primer punto (el más reciente)
            latest_point = data[0]
            
            # Verificar que estén todos los campos necesarios
            required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in latest_point for field in required_fields):
                logger.warning(f"Missing required fields in 5min data for {symbol}: {latest_point}")
                return None
            
            # Convertir la fecha a timestamp
            datetime_str = latest_point['date']
            try:
                record_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                record_timestamp = int(record_datetime.timestamp())
            except ValueError as e:
                logger.error(f"Error parsing datetime '{datetime_str}' for {symbol}: {e}")
                return None
            
            # Verificar si este punto ya ha sido procesado anteriormente
            if symbol in self.last_processed_timestamps and self.last_processed_timestamps[symbol] >= record_timestamp:
                logger.debug(f"Skipping already processed 5min data point for {symbol} at {datetime_str}")
                return None
            
            # Crear objeto con formato estándar
            data_point = {
                'symbol': symbol,
                'price': float(latest_point['close']),
                'change': None,  # No disponible en datos históricos
                'change_percent': None,  # No disponible en datos históricos
                'volume': int(latest_point['volume']),
                'timestamp': record_timestamp,
                'datetime': datetime_str,
                'data_type': 'realtime_5min',  # Tipo especial para diferenciar del histórico
                'open': float(latest_point['open']),
                'high': float(latest_point['high']),
                'low': float(latest_point['low']),
                'close': float(latest_point['close']),
                'market_cap': None  # No disponible en datos históricos
            }
            
            # Actualizar el último timestamp procesado
            self.last_processed_timestamps[symbol] = record_timestamp
            
            return data_point
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching latest 5min data for {symbol}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in 5min data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching latest 5min data for {symbol}: {e}", exc_info=True)
        
        return None
    
    def fetch_intraday_data(self, symbol, interval, start_date, end_date):
        """Obtener datos históricos intradía"""
        try:
            # Convertir fechas a formato de string yyyy-MM-dd
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Construir URL con formato correcto para la API de FMP
            url = f"{self.base_url}/historical-chart/{interval}/{symbol}?from={start_str}&to={end_str}&apikey={self.api_key}"
            logger.info(f"Fetching {interval} data for {symbol} from {start_str} to {end_str}")
            
            # Realizar la petición con timeout adecuado
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    # Procesar datos
                    processed_data = []
                    data_type = f"intraday_{interval}"
                    
                    # No necesitamos verificar la última fecha global aquí, ya que ahora estamos 
                    # iterando correctamente en la función principal y filtramos duplicados
                    
                    for item in data:
                        try:
                            # Procesar fecha y timestamp
                            datetime_str = item.get('date')
                            if not datetime_str:
                                logger.warning(f"Missing date in data point for {symbol} {interval}")
                                continue
                                
                            record_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                            record_timestamp = int(record_datetime.timestamp())
                            
                            # Verificar si este registro específico ya existe (verificación rápida)
                            if self.check_record_exists(symbol, data_type, datetime_str):
                                logger.debug(f"Skipping duplicate {interval} data point for {symbol} at {datetime_str}")
                                continue
                            
                            # Crear registro con todos los campos necesarios
                            record = {
                                'symbol': symbol,
                                'price': item.get('close', 0),  # Usamos close como precio actual
                                'change': None,  # No disponible en datos históricos
                                'change_percent': None,  # No disponible en datos históricos
                                'volume': item.get('volume', 0),
                                'timestamp': record_timestamp,
                                'datetime': datetime_str,
                                'data_type': data_type,
                                'open': item.get('open', 0),
                                'high': item.get('high', 0),
                                'low': item.get('low', 0),
                                'close': item.get('close', 0),
                                'market_cap': None  # No disponible en datos históricos
                            }
                            processed_data.append(record)
                        except Exception as e:
                            logger.error(f"Error processing data point for {symbol} {interval}: {e}")
                    
                    # Organizar datos por fecha ascendente (más antiguos primero)
                    processed_data.sort(key=lambda x: x['timestamp'])
                    
                    logger.info(f"Successfully processed {len(processed_data)} NEW {interval} data points for {symbol}")
                    return processed_data
                else:
                    logger.warning(f"Empty data returned for {symbol} at {interval} interval")
            else:
                logger.error(f"Error fetching {interval} data for {symbol}: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Exception fetching {interval} data for {symbol}: {e}", exc_info=True)
        
        return []
    
    def fetch_daily_data(self, symbol, start_date, end_date):
        """Obtener datos históricos diarios"""
        try:
            # Convertir fechas a formato de string yyyy-MM-dd
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Construir URL con formato correcto para la API de FMP
            url = f"{self.base_url}/historical-price-full/{symbol}?from={start_str}&to={end_str}&apikey={self.api_key}"
            logger.info(f"Fetching daily data for {symbol} from {start_str} to {end_str}")
            
            # Realizar la petición con timeout adecuado
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and 'historical' in data:
                    # Procesar datos
                    processed_data = []
                    data_type = "daily"
                    
                    for item in data['historical']:
                        try:
                            # Procesar fecha y timestamp
                            datetime_str = item.get('date')
                            if not datetime_str:
                                logger.warning(f"Missing date in data point for {symbol} daily")
                                continue
                                
                            record_datetime = datetime.strptime(datetime_str, '%Y-%m-%d')
                            record_timestamp = int(record_datetime.timestamp())
                            
                            # Verificar si este registro específico ya existe
                            if self.check_record_exists(symbol, data_type, datetime_str):
                                logger.debug(f"Skipping duplicate daily data point for {symbol} at {datetime_str}")
                                continue
                            
                            # Crear registro con todos los campos necesarios
                            record = {
                                'symbol': symbol,
                                'price': item.get('close', 0),  # Usamos close como precio actual
                                'change': item.get('change', None),
                                'change_percent': item.get('changePercent', None),
                                'volume': item.get('volume', 0),
                                'timestamp': record_timestamp,
                                'datetime': datetime_str,
                                'data_type': data_type,
                                'open': item.get('open', 0),
                                'high': item.get('high', 0),
                                'low': item.get('low', 0),
                                'close': item.get('close', 0),
                                'market_cap': None  # No disponible en datos históricos diarios
                            }
                            processed_data.append(record)
                        except Exception as e:
                            logger.error(f"Error processing daily data point for {symbol}: {e}")
                    
                    # Organizar datos por fecha ascendente (más antiguos primero)
                    processed_data.sort(key=lambda x: x['timestamp'])
                    
                    logger.info(f"Successfully processed {len(processed_data)} NEW daily data points for {symbol}")
                    return processed_data
                else:
                    logger.warning(f"Empty data or missing 'historical' key returned for {symbol} daily data")
            else:
                logger.error(f"Error fetching daily data for {symbol}: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Exception fetching daily data for {symbol}: {e}", exc_info=True)
        
        return []
    
    def store_quote(self, quote):
        """Almacenar cotización en tiempo real en la BD y publicar en Kafka"""
        try:
            # Verificar si ya existe una cotización con la misma marca de tiempo (aproximadamente)
            with self.db_engine.connect() as conn:
                # Buscamos cotizaciones en un rango de 1 segundo alrededor de la marca de tiempo actual
                query = text("""
                    SELECT COUNT(*) 
                    FROM market_data 
                    WHERE symbol = :symbol AND data_type = :data_type 
                    AND ABS(EXTRACT(EPOCH FROM to_timestamp(:timestamp) - to_timestamp(market_data.timestamp))) <= 1
                """)
                
                result = conn.execute(query, {
                    "symbol": quote['symbol'],
                    "data_type": quote['data_type'],
                    "timestamp": quote['timestamp']
                }).fetchone()
                
                # Si ya existe una cotización similar, no hacemos nada
                if result and result[0] > 0:
                    logger.debug(f"Duplicate quote for {quote['symbol']} at {quote['datetime']}")
                    return False
                
                # Insertar la cotización
                insert_query = text("""
                    INSERT INTO market_data (
                        symbol, price, change, change_percent, volume, timestamp, datetime, 
                        data_type, open, high, low, close, market_cap
                    ) VALUES (
                        :symbol, :price, :change, :change_percent, :volume, :timestamp, :datetime, 
                        :data_type, :open, :high, :low, :close, :market_cap
                    )
                """)
                
                conn.execute(insert_query, quote)
            
            # Publicar en Kafka
            if self.kafka_producer:
                self.kafka_producer.send(INGESTION_TOPIC, quote)
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing quote: {e}")
            return False
    
    def store_intraday_data(self, data):
        """Almacenar datos históricos intradía en lote"""
        if not data or len(data) == 0:
            logger.warning("No data provided for storage.")
            return False

        try:
            with self.db_engine.connect() as conn:
                for record in data:
                    logger.debug(f"Attempting to insert record: {record}")
                    insert_query = text("""
                        INSERT INTO market_data (
                            symbol, price, change, change_percent, volume, timestamp, datetime, 
                            data_type, open, high, low, close, market_cap
                        ) VALUES (
                            :symbol, :price, :change, :change_percent, :volume, :timestamp, :datetime, 
                            :data_type, :open, :high, :low, :close, :market_cap
                        )
                        ON CONFLICT ON CONSTRAINT market_data_symbol_datetime_key DO NOTHING
                    """)
                    conn.execute(insert_query, record)
                    logger.info(f"Successfully inserted record for symbol: {record['symbol']} at {record['datetime']}")

            return True

        except Exception as e:
            logger.error(f"Error storing intraday data: {e}", exc_info=True)
            return False

    def store_and_publish_realtime_data(self, data_point):
        """
        Almacena un punto de datos en tiempo real en la BD y lo publica en Kafka.
        Esta función se usa específicamente para los datos de 5min en tiempo "casi real".
        """
        if not data_point:
            logger.warning("No data point provided for storage.")
            return False

        try:
            with self.db_engine.connect() as conn:
                logger.debug(f"Attempting to insert data point: {data_point}")
                insert_query = text("""
                    INSERT INTO market_data (
                        symbol, price, change, change_percent, volume, timestamp, datetime, 
                        data_type, open, high, low, close, market_cap
                    ) VALUES (
                        :symbol, :price, :change, :change_percent, :volume, :timestamp, :datetime, 
                        :data_type, :open, :high, :low, :close, :market_cap
                    )
                    ON CONFLICT ON CONSTRAINT market_data_symbol_datetime_key DO NOTHING
                """)
                conn.execute(insert_query, data_point)
                logger.info(f"Successfully inserted realtime data point for symbol: {data_point['symbol']} at {data_point['datetime']}")

            if self.kafka_producer:
                self.kafka_producer.send(REALTIME_TOPIC, data_point)
                logger.info(f"Realtime data point for {data_point['symbol']} at {data_point['datetime']} published to Kafka")
                return True
            else:
                logger.error("Cannot publish to Kafka: producer not available")
                return False

        except Exception as e:
            logger.error(f"Error storing and publishing realtime data: {e}", exc_info=True)
            return False

# Inicializar el gestor de ingesta
ingestion_manager = IngestionManager(FMP_API_KEY, FMP_BASE_URL, engine)

def realtime_ingestion_thread():
    """
    Hilo que obtiene datos "en tiempo real" (5min) para alimentar el streaming.
    Se ejecuta cada REALTIME_INTERVAL_SECONDS (por defecto 5 minutos).
    """
    symbols = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
        "META", "NVDA", "NFLX", "INTC", "AMD"
    ]
    
    logger.info(f"Starting realtime ingestion thread with interval of {REALTIME_INTERVAL_SECONDS} seconds")
    
    while True:
        start_time = time.time()
        
        # Para cada símbolo, obtener y procesar el último punto de datos de 5min
        for symbol in symbols:
            try:
                logger.debug(f"Fetching latest 5min data for {symbol}")
                data_point = ingestion_manager.fetch_latest_5min_data(symbol)
                
                if data_point:
                    # Almacenar en BD y publicar en Kafka
                    success = ingestion_manager.store_and_publish_realtime_data(data_point)
                    logger.info(f"Processed realtime 5min data for {symbol} at {data_point['datetime']} - Success: {success}")
                else:
                    logger.debug(f"No new 5min data available for {symbol}")
                
                # Pequeña pausa entre símbolos para no saturar la API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing realtime data for {symbol}: {e}", exc_info=True)
        
        # Calcular tiempo a esperar hasta la próxima ejecución
        elapsed = time.time() - start_time
        wait_time = max(0, REALTIME_INTERVAL_SECONDS - elapsed)
        
        logger.info(f"Realtime ingestion cycle completed in {elapsed:.2f} seconds. Waiting {wait_time:.2f} seconds until next cycle.")
        time.sleep(wait_time)

def historical_ingestion_thread():
    """Background thread to run the historical ingestion process"""
    symbols = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
        "META", "NVDA", "NFLX", "INTC", "AMD",
        "IAG.MC", "PHM.MC", "AENA.MC", "BA", 
        "CAR", "DLTR", "SASA.IS"
    ]
    
    # Configuración de intervalos respetando límites de FMP Starter Plan
    intraday_config = {
        '1min': {'max_days': 3, 'target_days': 90, 'interval_seconds': 15*60},
        '5min': {'max_days': 10, 'target_days': 90, 'interval_seconds': 30*60},
        '15min': {'max_days': 45, 'target_days': 90, 'interval_seconds': 60*60},
        '30min': {'max_days': 30, 'target_days': 90, 'interval_seconds': 2*60*60},
        '45min': {'max_days': 45, 'target_days': 90, 'interval_seconds': 3*60*60},
        '1hour': {'max_days': 90, 'target_days': 90, 'interval_seconds': 4*60*60}
    }
    
    while True:
        current_time = time.time()
        
        for symbol in symbols:
            try:
                # Procesamiento de intervalos históricos intradía
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
                
                # Procesamiento de datos diarios con historial de 4 años
                data_type_daily = "daily"
                years_of_history = 4
                max_days_daily = 365 * years_of_history
                
                # Obtener la última fecha para datos diarios
                last_date_daily = ingestion_manager.get_last_date_for_symbol_and_type(symbol, data_type_daily)
                end_date_daily = datetime.now() - timedelta(days=1)  # Hasta ayer
                
                if last_date_daily:
                    # Si tenemos datos diarios, continuar desde el último día + 1
                    start_date_daily = last_date_daily + timedelta(days=1)
                    logger.info(f"Incremental fetch for {symbol} daily data from {start_date_daily}")
                    
                    # Si ya estamos actualizados, no hacer nada
                    if start_date_daily > end_date_daily:
                        logger.info(f"No new daily data needed for {symbol}, already up to date")
                        continue
                else:
                    # Si no tenemos datos diarios, obtener 4 años de historial
                    start_date_daily = end_date_daily - timedelta(days=max_days_daily)
                    logger.info(f"Initial fetch for {symbol} daily data with {years_of_history} years of history from {start_date_daily}")
                
                # Obtener datos diarios
                daily_data = ingestion_manager.fetch_daily_data(
                    symbol=symbol,
                    start_date=start_date_daily,
                    end_date=end_date_daily
                )
                
                # Almacenar datos diarios
                if daily_data and len(daily_data) > 0:
                    ingestion_manager.store_intraday_data(daily_data)  # Reutilizamos la misma función para guardar
                    logger.info(f"Stored {len(daily_data)} daily data points for {symbol}")
                else:
                    logger.warning(f"No daily data points retrieved for {symbol}")
                
                # Pequeña pausa entre símbolos para no saturar la API
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing historical data for {symbol}: {e}", exc_info=True)
        
        # Calcular el tiempo transcurrido y esperar hasta el próximo ciclo
        elapsed = time.time() - current_time
        logger.info(f"Historical ingestion cycle completed in {elapsed:.2f} seconds")
        
        # Esperar 6 horas para el próximo ciclo completo de ingestión histórica
        wait_time = 6 * 60 * 60  # 6 horas en segundos
        logger.info(f"Waiting {wait_time} seconds until next historical ingestion cycle")
        time.sleep(wait_time)

if __name__ == "__main__":
    logger.info("Iniciando servicio de ingestión...")
    
    # --- NUEVO: Iniciar servidor WebSocket en un hilo --- 
    loop = asyncio.get_event_loop()
    websocket_thread = threading.Thread(target=lambda: loop.run_until_complete(start_websocket_server()), daemon=True)
    websocket_thread.start()
    # --- FIN NUEVO ---

    # Iniciar hilos de ingesta
    historical_thread = Thread(target=historical_ingestion_thread, daemon=True)
    realtime_thread = Thread(target=realtime_ingestion_thread, daemon=True)
    
    logger.info("Starting ingestion threads")
    
    # Iniciar el hilo de ingesta histórica
    historical_thread.start()
    logger.info("Historical ingestion thread started")
    
    # Iniciar el hilo de ingesta en tiempo real
    realtime_thread.start()
    logger.info("Realtime ingestion thread started")
    
    # Mantener el proceso principal vivo
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Stopping ingestion service...")