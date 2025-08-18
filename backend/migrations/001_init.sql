-- LUX DEX Database Schema
-- Initial migration for order book and trading

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    filled DECIMAL(20, 8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open',
    trader_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_orders_symbol_status (symbol, status),
    INDEX idx_orders_trader (trader_id),
    INDEX idx_orders_created (created_at DESC)
);

-- Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('orders', 'created_at', if_not_exists => TRUE);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    buyer_order_id UUID REFERENCES orders(id),
    seller_order_id UUID REFERENCES orders(id),
    buyer_id VARCHAR(100) NOT NULL,
    seller_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_trades_symbol (symbol),
    INDEX idx_trades_created (created_at DESC)
);

-- Convert to hypertable
SELECT create_hypertable('trades', 'created_at', if_not_exists => TRUE);

-- Market data table for OHLCV
CREATE TABLE IF NOT EXISTS market_data (
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (symbol, interval, timestamp),
    INDEX idx_market_data_symbol (symbol),
    INDEX idx_market_data_timestamp (timestamp DESC)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    unit VARCHAR(20),
    metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_metrics_name (metric_name),
    INDEX idx_metrics_timestamp (timestamp DESC)
);

-- Convert to hypertable
SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);

-- Create continuous aggregates for 1m, 5m, 1h candles
CREATE MATERIALIZED VIEW market_data_1m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 minute', timestamp) AS bucket,
    first(open, timestamp) as open,
    max(high) as high,
    min(low) as low,
    last(close, timestamp) as close,
    sum(volume) as volume
FROM market_data
GROUP BY symbol, bucket
WITH NO DATA;

-- Add retention policy (keep 7 days of raw data)
SELECT add_retention_policy('orders', INTERVAL '7 days');
SELECT add_retention_policy('trades', INTERVAL '7 days');
SELECT add_retention_policy('market_data', INTERVAL '30 days');
SELECT add_retention_policy('performance_metrics', INTERVAL '1 day');