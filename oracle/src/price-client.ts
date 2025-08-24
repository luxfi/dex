/**
 * Pyth Network Price Client for Lux Exchange
 * Connects to Hermes node for real-time price feeds
 */

import { EventEmitter } from 'events';
import WebSocket from 'ws';
import axios from 'axios';

export interface PythPrice {
  id: string;
  price: string;
  conf: string;
  expo: number;
  publishTime: number;
  emaPrice: string;
  emaConf: string;
  status: 'trading' | 'halted' | 'auction' | 'unknown';
  numPublishers: number;
  maxNumPublishers: number;
  attestationTime?: number;
  prevPublishTime?: number;
  prevPrice?: string;
  prevConf?: string;
}

export interface PriceUpdate {
  symbol: string;
  feedId: string;
  price: number;
  confidence: number;
  timestamp: Date;
  emaPrice: number;
  status: string;
}

export class PythPriceClient extends EventEmitter {
  private hermesUrl: string;
  private wsUrl: string;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private subscriptions: Set<string> = new Set();
  private priceFeeds: Map<string, string> = new Map();
  private symbolMap: Map<string, string> = new Map();

  constructor(
    hermesUrl: string = 'http://localhost:2000',
    wsUrl: string = 'ws://localhost:2002'
  ) {
    super();
    this.hermesUrl = hermesUrl;
    this.wsUrl = wsUrl;
    this.loadPriceFeedIds();
  }

  private loadPriceFeedIds(): void {
    // Load price feed IDs from config
    const config = require('../config/price-feeds.json');
    
    // Map all asset classes
    for (const [assetClass, feeds] of Object.entries(config.priceFeedIds)) {
      for (const [symbol, feedId] of Object.entries(feeds as any)) {
        this.priceFeeds.set(symbol, feedId as string);
        this.symbolMap.set(feedId as string, symbol);
      }
    }
  }

  /**
   * Connect to Hermes WebSocket
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`${this.wsUrl}/ws`);

        this.ws.on('open', () => {
          console.log('Connected to Pyth Hermes');
          this.reconnectAttempts = 0;
          this.emit('connected');
          this.resubscribe();
          resolve();
        });

        this.ws.on('message', (data: Buffer) => {
          this.handleMessage(data.toString());
        });

        this.ws.on('error', (error) => {
          console.error('WebSocket error:', error);
          this.emit('error', error);
        });

        this.ws.on('close', () => {
          console.log('Disconnected from Pyth Hermes');
          this.emit('disconnected');
          this.handleReconnect();
        });

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle incoming price messages
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      if (message.type === 'price_update') {
        const pythPrice = message.price_feed as PythPrice;
        const symbol = this.symbolMap.get(pythPrice.id);
        
        if (symbol) {
          const update: PriceUpdate = {
            symbol,
            feedId: pythPrice.id,
            price: this.parsePrice(pythPrice.price, pythPrice.expo),
            confidence: this.parsePrice(pythPrice.conf, pythPrice.expo),
            timestamp: new Date(pythPrice.publishTime * 1000),
            emaPrice: this.parsePrice(pythPrice.emaPrice, pythPrice.expo),
            status: pythPrice.status
          };
          
          this.emit('price', update);
          this.emit(`price:${symbol}`, update);
        }
      }
    } catch (error) {
      console.error('Failed to parse message:', error);
    }
  }

  /**
   * Parse Pyth price with exponent
   */
  private parsePrice(price: string, expo: number): number {
    const value = parseFloat(price);
    return value * Math.pow(10, expo);
  }

  /**
   * Subscribe to price feeds
   */
  subscribe(symbols: string[]): void {
    const feedIds: string[] = [];
    
    for (const symbol of symbols) {
      const feedId = this.priceFeeds.get(symbol);
      if (feedId) {
        feedIds.push(feedId);
        this.subscriptions.add(feedId);
      } else {
        console.warn(`Unknown symbol: ${symbol}`);
      }
    }

    if (feedIds.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        ids: feedIds
      }));
    }
  }

  /**
   * Unsubscribe from price feeds
   */
  unsubscribe(symbols: string[]): void {
    const feedIds: string[] = [];
    
    for (const symbol of symbols) {
      const feedId = this.priceFeeds.get(symbol);
      if (feedId) {
        feedIds.push(feedId);
        this.subscriptions.delete(feedId);
      }
    }

    if (feedIds.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        ids: feedIds
      }));
    }
  }

  /**
   * Get latest price via REST API
   */
  async getLatestPrice(symbol: string): Promise<PriceUpdate | null> {
    const feedId = this.priceFeeds.get(symbol);
    if (!feedId) {
      throw new Error(`Unknown symbol: ${symbol}`);
    }

    try {
      const response = await axios.get(
        `${this.hermesUrl}/api/latest_price_feeds`,
        {
          params: { ids: [feedId] }
        }
      );

      const priceData = response.data[0];
      if (!priceData) return null;

      const pythPrice = priceData.price;
      return {
        symbol,
        feedId: pythPrice.id,
        price: this.parsePrice(pythPrice.price, pythPrice.expo),
        confidence: this.parsePrice(pythPrice.conf, pythPrice.expo),
        timestamp: new Date(pythPrice.publish_time * 1000),
        emaPrice: this.parsePrice(pythPrice.ema_price, pythPrice.expo),
        status: pythPrice.status
      };
    } catch (error) {
      console.error(`Failed to get price for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get multiple prices
   */
  async getLatestPrices(symbols: string[]): Promise<Map<string, PriceUpdate>> {
    const feedIds: string[] = [];
    const symbolToFeedId = new Map<string, string>();
    
    for (const symbol of symbols) {
      const feedId = this.priceFeeds.get(symbol);
      if (feedId) {
        feedIds.push(feedId);
        symbolToFeedId.set(feedId, symbol);
      }
    }

    const prices = new Map<string, PriceUpdate>();
    
    try {
      const response = await axios.get(
        `${this.hermesUrl}/api/latest_price_feeds`,
        {
          params: { ids: feedIds }
        }
      );

      for (const priceData of response.data) {
        const pythPrice = priceData.price;
        const symbol = symbolToFeedId.get(pythPrice.id);
        
        if (symbol) {
          prices.set(symbol, {
            symbol,
            feedId: pythPrice.id,
            price: this.parsePrice(pythPrice.price, pythPrice.expo),
            confidence: this.parsePrice(pythPrice.conf, pythPrice.expo),
            timestamp: new Date(pythPrice.publish_time * 1000),
            emaPrice: this.parsePrice(pythPrice.ema_price, pythPrice.expo),
            status: pythPrice.status
          });
        }
      }
    } catch (error) {
      console.error('Failed to get prices:', error);
    }
    
    return prices;
  }

  /**
   * Get all available symbols
   */
  getAvailableSymbols(): string[] {
    return Array.from(this.priceFeeds.keys());
  }

  /**
   * Get symbols by asset class
   */
  getSymbolsByAssetClass(assetClass: 'crypto' | 'stocks' | 'forex' | 'commodities' | 'indices'): string[] {
    const config = require('../config/price-feeds.json');
    const feeds = config.priceFeedIds[assetClass] || {};
    return Object.keys(feeds);
  }

  /**
   * Handle reconnection
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.emit('error', new Error('Max reconnection attempts reached'));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    setTimeout(() => {
      console.log(`Reconnecting to Pyth Hermes (attempt ${this.reconnectAttempts})...`);
      this.connect().catch(console.error);
    }, delay);
  }

  /**
   * Resubscribe after reconnection
   */
  private resubscribe(): void {
    if (this.subscriptions.size > 0 && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        ids: Array.from(this.subscriptions)
      }));
    }
  }

  /**
   * Disconnect from Hermes
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Check connection status
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Export singleton instance
export const pythClient = new PythPriceClient();