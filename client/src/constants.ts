/**
 * Lux Exchange Constants
 */

// API Endpoints
export const ENDPOINTS = {
  PRODUCTION: 'https://api.luxexchange.io',
  TESTNET: 'https://testnet-api.luxexchange.io',
  WS_PRODUCTION: 'wss://stream.luxexchange.io',
  WS_TESTNET: 'wss://testnet-stream.luxexchange.io'
};

// WebSocket Channels
export const WS_CHANNELS = {
  ORDERBOOK: 'orderbook',
  TRADES: 'trades',
  TICKER: 'ticker',
  KLINE: 'kline',
  USER_ORDERS: 'user_orders',
  USER_TRADES: 'user_trades',
  USER_POSITIONS: 'user_positions',
  USER_BALANCE: 'user_balance'
};

// Time intervals for klines
export const INTERVALS = {
  '1m': '1m',
  '3m': '3m',
  '5m': '5m',
  '15m': '15m',
  '30m': '30m',
  '1h': '1h',
  '2h': '2h',
  '4h': '4h',
  '6h': '6h',
  '8h': '8h',
  '12h': '12h',
  '1d': '1d',
  '3d': '3d',
  '1w': '1w',
  '1M': '1M'
};

// Popular trading pairs
export const POPULAR_SYMBOLS = {
  // Crypto
  CRYPTO: [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'BNB-USD',
    'XRP-USD',
    'ADA-USD',
    'AVAX-USD',
    'DOGE-USD',
    'DOT-USD',
    'MATIC-USD'
  ],
  // Stocks
  STOCKS: [
    'AAPL',
    'GOOGL',
    'MSFT',
    'AMZN',
    'TSLA',
    'META',
    'NVDA',
    'AMD',
    'NFLX',
    'SPY'
  ],
  // Forex
  FOREX: [
    'EUR-USD',
    'GBP-USD',
    'USD-JPY',
    'USD-CHF',
    'AUD-USD',
    'USD-CAD',
    'NZD-USD',
    'EUR-GBP',
    'EUR-JPY',
    'GBP-JPY'
  ],
  // Commodities
  COMMODITIES: [
    'GOLD',
    'SILVER',
    'OIL',
    'NATGAS',
    'COPPER',
    'WHEAT',
    'CORN',
    'SOYBEANS',
    'COFFEE',
    'SUGAR'
  ]
};

// Fee tiers
export const FEE_TIERS = {
  TIER_1: { makerFee: 0.0002, takerFee: 0.0005, volume: 0 },
  TIER_2: { makerFee: 0.00018, takerFee: 0.00045, volume: 100000 },
  TIER_3: { makerFee: 0.00016, takerFee: 0.0004, volume: 1000000 },
  TIER_4: { makerFee: 0.00014, takerFee: 0.00035, volume: 10000000 },
  TIER_5: { makerFee: 0.00012, takerFee: 0.0003, volume: 50000000 },
  VIP: { makerFee: 0.0001, takerFee: 0.00025, volume: 100000000 }
};

// Rate limits
export const RATE_LIMITS = {
  DEFAULT: {
    maxRequestsPerSecond: 20,
    maxOrdersPerSecond: 10,
    burstCapacity: 100
  },
  VIP: {
    maxRequestsPerSecond: 100,
    maxOrdersPerSecond: 50,
    burstCapacity: 500
  }
};

// Order size limits
export const ORDER_LIMITS = {
  MIN_NOTIONAL: 10, // USD
  MAX_ORDERS_PER_SYMBOL: 200,
  MAX_OPEN_ORDERS: 1000
};

// System status
export const SYSTEM_STATUS = {
  OPERATIONAL: 'operational',
  DEGRADED: 'degraded',
  MAINTENANCE: 'maintenance',
  OFFLINE: 'offline'
};