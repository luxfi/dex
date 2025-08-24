/**
 * Utility functions for Lux Exchange client
 */

import Decimal from 'decimal.js';
import { createHmac } from 'crypto';

/**
 * Format number with specified decimal places
 */
export function formatNumber(value: number | string | Decimal, decimals: number = 2): string {
  const decimal = new Decimal(value);
  return decimal.toFixed(decimals);
}

/**
 * Format currency value
 */
export function formatCurrency(value: number | string | Decimal, currency: string = 'USD'): string {
  const decimal = new Decimal(value);
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: currency === 'BTC' ? 8 : 2
  });
  return formatter.format(decimal.toNumber());
}

/**
 * Format percentage
 */
export function formatPercent(value: number | string | Decimal, decimals: number = 2): string {
  const decimal = new Decimal(value);
  return `${decimal.toFixed(decimals)}%`;
}

/**
 * Round to tick size
 */
export function roundToTickSize(value: number | string | Decimal, tickSize: number | string | Decimal): Decimal {
  const decimal = new Decimal(value);
  const tick = new Decimal(tickSize);
  return decimal.dividedBy(tick).round().times(tick);
}

/**
 * Calculate position size based on risk
 */
export function calculatePositionSize(
  accountBalance: Decimal,
  riskPercent: Decimal,
  entryPrice: Decimal,
  stopPrice: Decimal
): Decimal {
  const riskAmount = accountBalance.times(riskPercent.dividedBy(100));
  const priceRisk = entryPrice.minus(stopPrice).abs();
  
  if (priceRisk.isZero()) {
    return new Decimal(0);
  }
  
  return riskAmount.dividedBy(priceRisk);
}

/**
 * Generate request signature
 */
export function generateSignature(
  secret: string,
  method: string,
  path: string,
  timestamp: number,
  body?: string
): string {
  const message = `${timestamp}${method}${path}${body || ''}`;
  return createHmac('sha256', secret).update(message).digest('hex');
}

/**
 * Parse API error
 */
export function parseApiError(error: any): {
  code: string;
  message: string;
  details?: any;
} {
  if (typeof error === 'string') {
    return { code: 'UNKNOWN', message: error };
  }
  
  if (error.response) {
    return {
      code: error.response.code || 'API_ERROR',
      message: error.response.message || 'API request failed',
      details: error.response.data
    };
  }
  
  if (error.code) {
    return {
      code: error.code,
      message: error.message || 'Unknown error',
      details: error.details
    };
  }
  
  return {
    code: 'UNKNOWN',
    message: error.message || 'An unknown error occurred'
  };
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry function with exponential backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i < maxAttempts; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (i < maxAttempts - 1) {
        await sleep(delay * Math.pow(2, i));
      }
    }
  }
  
  throw lastError;
}

/**
 * Validate symbol format
 */
export function isValidSymbol(symbol: string): boolean {
  // Crypto pairs: BTC-USD, ETH-BTC
  // Stocks: AAPL, GOOGL
  // Forex: EUR-USD
  const cryptoPattern = /^[A-Z]{2,10}-[A-Z]{2,10}$/;
  const stockPattern = /^[A-Z]{1,5}$/;
  
  return cryptoPattern.test(symbol) || stockPattern.test(symbol);
}

/**
 * Parse symbol components
 */
export function parseSymbol(symbol: string): {
  base?: string;
  quote?: string;
  ticker?: string;
} {
  if (symbol.includes('-')) {
    const [base, quote] = symbol.split('-');
    return { base, quote };
  }
  return { ticker: symbol };
}

/**
 * Format timestamp
 */
export function formatTimestamp(timestamp: Date | number | string): string {
  const date = new Date(timestamp);
  return date.toISOString();
}

/**
 * Calculate price change percentage
 */
export function calculateChangePercent(current: Decimal, previous: Decimal): Decimal {
  if (previous.isZero()) {
    return new Decimal(0);
  }
  return current.minus(previous).dividedBy(previous).times(100);
}

/**
 * Validate order parameters
 */
export function validateOrderParams(params: {
  symbol: string;
  side: string;
  type: string;
  quantity: number | string;
  price?: number | string;
}): string[] {
  const errors: string[] = [];
  
  if (!params.symbol || !isValidSymbol(params.symbol)) {
    errors.push('Invalid symbol');
  }
  
  if (!['BUY', 'SELL'].includes(params.side)) {
    errors.push('Invalid side');
  }
  
  const quantity = new Decimal(params.quantity);
  if (quantity.lessThanOrEqualTo(0)) {
    errors.push('Quantity must be positive');
  }
  
  if (params.type === 'LIMIT' && params.price) {
    const price = new Decimal(params.price);
    if (price.lessThanOrEqualTo(0)) {
      errors.push('Price must be positive');
    }
  }
  
  return errors;
}