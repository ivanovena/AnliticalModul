import logging
from sqlalchemy import create_engine, text
from config import DB_URI
import time

# Configura el logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DBInit")

def init_database():
    """Inicializa la base de datos con las tablas necesarias"""
    # Esperar a que PostgreSQL esté listo
    max_retries = 10
    retry_interval = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Intento {attempt+1}/{max_retries} de conectar a la base de datos")
            engine = create_engine(DB_URI)
            with engine.connect() as conn:
                logger.info("Conexión a la base de datos establecida. Creando tablas...")
                
                # First drop existing market_data if it exists
                try:
                    conn.execute(text("DROP TABLE IF EXISTS market_data"))
                    logger.info("Dropped existing market_data table to update schema")
                except Exception as e:
                    logger.warning(f"Error dropping market_data table: {e}")
                
                # Crear tabla market_data con el nuevo esquema
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        price DECIMAL(18, 4),
                        change DECIMAL(18, 4),
                        change_percent DECIMAL(8, 4),
                        volume BIGINT,
                        timestamp BIGINT,
                        datetime VARCHAR(50),
                        data_type VARCHAR(20),
                        open DECIMAL(18, 4),
                        high DECIMAL(18, 4),
                        low DECIMAL(18, 4),
                        close DECIMAL(18, 4),
                        market_cap DECIMAL(18, 2)
                    )
                """))
                
                # Crear índices para optimizar consultas
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
                    CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_market_data_data_type ON market_data(data_type);
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_data_type ON market_data(symbol, data_type);
                """))
                
                # Crear tabla positions
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price DECIMAL(18, 4) NOT NULL,
                        entry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        exit_price DECIMAL(18, 4),
                        exit_date TIMESTAMP,
                        pnl DECIMAL(18, 4),
                        status VARCHAR(20) DEFAULT 'OPEN'
                    )
                """))
                
                # Crear tabla orders
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY,
                        order_id VARCHAR(100) UNIQUE NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        quantity INTEGER NOT NULL,
                        price DECIMAL(18, 4) NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        source VARCHAR(20) DEFAULT 'manual'
                    )
                """))
                
                # Crear tabla performance_metrics
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        portfolio_value DECIMAL(18, 4) NOT NULL,
                        cash DECIMAL(18, 4) NOT NULL,
                        positions_value DECIMAL(18, 4) NOT NULL,
                        daily_pnl DECIMAL(18, 4),
                        total_pnl DECIMAL(18, 4),
                        sharpe_ratio DECIMAL(8, 4),
                        max_drawdown DECIMAL(8, 4)
                    )
                """))
                
                # Crear tabla model_training
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_training (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL,
                        model_type VARCHAR(50) NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'training',
                        progress INTEGER DEFAULT 0,
                        metrics JSONB,
                        params JSONB
                    )
                """))
                
                # Crear tabla chat_history
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        feedback INTEGER
                    )
                """))
                
                logger.info("Tablas creadas correctamente")
                return True
                
        except Exception as e:
            logger.error(f"Error inicializando la base de datos: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Reintentando en {retry_interval} segundos...")
                time.sleep(retry_interval)
            else:
                logger.error("Número máximo de intentos alcanzado. No se pudo inicializar la base de datos.")
                return False

if __name__ == "__main__":
    init_database()
