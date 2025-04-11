-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_data (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  datetime TIMESTAMP NOT NULL,
  price NUMERIC(15,4),
  change NUMERIC(15,4),
  change_percent NUMERIC(15,4),
  volume BIGINT,
  timestamp BIGINT,
  data_type VARCHAR(20) NOT NULL,
  open NUMERIC(15,4),
  high NUMERIC(15,4),
  low NUMERIC(15,4),
  close NUMERIC(15,4),
  market_cap NUMERIC(18,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT market_data_symbol_datetime_key UNIQUE(symbol, datetime)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_datetime ON market_data(datetime);
CREATE INDEX IF NOT EXISTS idx_market_data_data_type ON market_data(data_type);

-- Create table for predictions
CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  prediction_date TIMESTAMP NOT NULL,
  predicted_price NUMERIC(15,4) NOT NULL,
  confidence NUMERIC(5,4) NOT NULL,
  model_version VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(symbol, prediction_date)
);

-- Create table for trading activity
CREATE TABLE IF NOT EXISTS trades (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  action VARCHAR(10) NOT NULL,
  quantity INTEGER NOT NULL,
  price NUMERIC(15,4) NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  source VARCHAR(20) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for model performance
CREATE TABLE IF NOT EXISTS model_performance (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  model_version VARCHAR(50) NOT NULL,
  mae NUMERIC(15,4) NOT NULL,
  rmse NUMERIC(15,4) NOT NULL,
  r2 NUMERIC(5,4) NOT NULL,
  feature_importance JSONB,
  training_date TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to get historical data with period aggregation
CREATE OR REPLACE FUNCTION get_historical_data(
  p_symbol VARCHAR,
  p_period VARCHAR,
  p_start_date TIMESTAMP,
  p_end_date TIMESTAMP
)
RETURNS TABLE (
  period_start TIMESTAMP,
  open NUMERIC,
  high NUMERIC,
  low NUMERIC,
  close NUMERIC,
  volume BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    date_trunc(p_period, datetime) AS period_start,
    first_value(open) OVER w AS open,
    max(high) OVER w AS high,
    min(low) OVER w AS low,
    last_value(close) OVER w AS close,
    sum(volume) OVER w AS volume
  FROM market_data
  WHERE symbol = p_symbol
    AND datetime >= p_start_date
    AND datetime <= p_end_date
  WINDOW w AS (PARTITION BY date_trunc(p_period, datetime) ORDER BY datetime)
  ORDER BY period_start;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO current_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO current_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO current_user;
