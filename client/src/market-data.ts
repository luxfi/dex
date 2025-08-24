/**
 * Lux Exchange Market Data
 * Real-time market data management
 */

import Decimal from 'decimal.js';
import { EventEmitter } from 'eventemitter3';
import { MarketTick, Kline } from './types';

export class MarketData extends EventEmitter {
  private tickers: Map<string, MarketTick> = new Map();
  private klines: Map<string, Map<string, Kline[]>> = new Map();
  
  /**
   * Update ticker data
   */
  updateTicker(tick: MarketTick): void {
    this.tickers.set(tick.symbol, tick);
    this.emit('ticker', tick);
  }

  /**
   * Get ticker for symbol
   */
  getTicker(symbol: string): MarketTick | null {
    return this.tickers.get(symbol) || null;
  }

  /**
   * Get all tickers
   */
  getAllTickers(): MarketTick[] {
    return Array.from(this.tickers.values());
  }

  /**
   * Update kline/candle data
   */
  updateKline(kline: Kline): void {
    const symbolKlines = this.klines.get(kline.symbol) || new Map();
    const intervalKlines = symbolKlines.get(kline.interval) || [];
    
    // Find and update or append
    const index = intervalKlines.findIndex((k: Kline) => 
      k.openTime.getTime() === kline.openTime.getTime()
    );
    
    if (index >= 0) {
      intervalKlines[index] = kline;
    } else {
      intervalKlines.push(kline);
      // Keep sorted by time
      intervalKlines.sort((a: Kline, b: Kline) => a.openTime.getTime() - b.openTime.getTime());
      // Limit stored candles
      if (intervalKlines.length > 1000) {
        intervalKlines.shift();
      }
    }
    
    symbolKlines.set(kline.interval, intervalKlines);
    this.klines.set(kline.symbol, symbolKlines);
    
    this.emit('kline', kline);
  }

  /**
   * Get klines for symbol and interval
   */
  getKlines(symbol: string, interval: string, limit?: number): Kline[] {
    const symbolKlines = this.klines.get(symbol);
    if (!symbolKlines) return [];
    
    const intervalKlines = symbolKlines.get(interval) || [];
    
    if (limit) {
      return intervalKlines.slice(-limit);
    }
    return intervalKlines;
  }

  /**
   * Calculate VWAP (Volume Weighted Average Price)
   */
  getVWAP(symbol: string, period: number = 20): Decimal | null {
    const ticker = this.getTicker(symbol);
    if (!ticker) return null;
    
    // Simplified VWAP using ticker data
    // In production, this would use actual trade data
    return ticker.last;
  }

  /**
   * Get 24h statistics
   */
  get24hStats(symbol: string): {
    high: Decimal;
    low: Decimal;
    volume: Decimal;
    changePercent: Decimal;
  } | null {
    const ticker = this.getTicker(symbol);
    if (!ticker) return null;
    
    return {
      high: ticker.high24h,
      low: ticker.low24h,
      volume: ticker.volume24h,
      changePercent: ticker.changePercent24h
    };
  }

  /**
   * Get top movers
   */
  getTopMovers(limit: number = 10): {
    gainers: MarketTick[];
    losers: MarketTick[];
    volume: MarketTick[];
  } {
    const tickers = this.getAllTickers();
    
    const gainers = [...tickers]
      .sort((a, b) => b.changePercent24h.comparedTo(a.changePercent24h))
      .slice(0, limit);
    
    const losers = [...tickers]
      .sort((a, b) => a.changePercent24h.comparedTo(b.changePercent24h))
      .slice(0, limit);
    
    const volume = [...tickers]
      .sort((a, b) => b.volume24h.comparedTo(a.volume24h))
      .slice(0, limit);
    
    return { gainers, losers, volume };
  }

  /**
   * Clear market data for symbol
   */
  clear(symbol: string): void {
    this.tickers.delete(symbol);
    this.klines.delete(symbol);
    this.emit('clear', symbol);
  }

  /**
   * Clear all market data
   */
  clearAll(): void {
    this.tickers.clear();
    this.klines.clear();
    this.emit('clear-all');
  }
}