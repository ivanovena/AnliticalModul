-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_data (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  datetime TIMESTAMP NOT NULL,
  open NUMERIC(15,4) NOT NULL,
  high NUMERIC(15,4) NOT NULL,
  low NUMERIC(15,4) NOT NULL,
  close NUMERIC(15,4) NOT NULL,
  volume BIGINT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(symbol, datetime)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_datetime ON market_data(datetime);

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

-- Insert some sample data for testing
-- INSERT INTO market_data (symbol, datetime, open, high, low, close, volume)
-- VALUES 
--   ('AAPL', '2023-05-01 09:30:00', 150.0, 151.2, 149.8, 150.5, 1000000),
--   ('AAPL', '2023-05-01 09:31:00', 150.5, 152.0, 150.2, 151.8, 1200000),
--   ('AAPL', '2023-05-01 09:32:00', 151.8, 153.0, 151.5, 152.2, 1100000),
--   ('GOOGL', '2023-05-01 09:30:00', 120.0, 121.5, 119.8, 121.0, 800000),
--   ('GOOGL', '2023-05-01 09:31:00', 121.0, 122.0, 120.5, 121.5, 750000),
--   ('GOOGL', '2023-05-01 09:32:00', 121.5, 122.5, 121.2, 122.0, 820000),
--   ('MSFT', '2023-05-01 09:30:00', 280.0, 282.0, 279.5, 281.5, 500000),
--   ('MSFT', '2023-05-01 09:31:00', 281.5, 283.0, 281.0, 282.5, 480000),
--   ('MSFT', '2023-05-01 09:32:00', 282.5, 284.0, 282.0, 283.8, 520000)
-- ON CONFLICT (symbol, datetime) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO current_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO current_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO current_user;
