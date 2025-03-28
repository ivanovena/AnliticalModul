-- Creación de tabla para datos del mercado
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    datetime TIMESTAMP NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    change_percent DECIMAL(8, 4),
    market_cap DECIMAL(20, 2),
    UNIQUE(symbol, datetime)
);

-- Índices para mejorar el rendimiento de consultas
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_datetime ON market_data (datetime);

-- Creación de tabla de órdenes
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(4) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(12, 4) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    source VARCHAR(50),
    order_id VARCHAR(100) UNIQUE
);

-- Tabla para historial de portfolio
CREATE TABLE IF NOT EXISTS portfolio_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    cash DECIMAL(14, 2) NOT NULL,
    total_value DECIMAL(14, 2) NOT NULL
);

-- Aseguramos que el usuario tenga permisos correctos
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO usuario;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO usuario;
