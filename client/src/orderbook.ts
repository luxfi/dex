/**
 * Lux Exchange Order Book
 * Real-time order book management
 */

import Decimal from 'decimal.js';
import { EventEmitter } from 'eventemitter3';
import { OrderBookLevel, OrderBookSnapshot } from './types';

export class OrderBook extends EventEmitter {
  private books: Map<string, OrderBookData> = new Map();

  /**
   * Update order book with new snapshot
   */
  update(snapshot: OrderBookSnapshot): void {
    const book = this.getOrCreateBook(snapshot.symbol);
    
    book.bids = new Map(
      snapshot.bids.map(level => [level.price.toString(), level])
    );
    book.asks = new Map(
      snapshot.asks.map(level => [level.price.toString(), level])
    );
    book.lastUpdate = snapshot.timestamp;
    book.sequenceNumber = snapshot.sequenceNumber;

    this.emit('update', snapshot.symbol, this.getSnapshot(snapshot.symbol));
  }

  /**
   * Apply incremental updates
   */
  applyUpdate(symbol: string, updates: {
    bids?: OrderBookLevel[];
    asks?: OrderBookLevel[];
    sequenceNumber?: number;
  }): void {
    const book = this.getOrCreateBook(symbol);

    // Apply bid updates
    if (updates.bids) {
      for (const level of updates.bids) {
        if (level.quantity.isZero()) {
          book.bids.delete(level.price.toString());
        } else {
          book.bids.set(level.price.toString(), level);
        }
      }
    }

    // Apply ask updates
    if (updates.asks) {
      for (const level of updates.asks) {
        if (level.quantity.isZero()) {
          book.asks.delete(level.price.toString());
        } else {
          book.asks.set(level.price.toString(), level);
        }
      }
    }

    book.lastUpdate = new Date();
    if (updates.sequenceNumber) {
      book.sequenceNumber = updates.sequenceNumber;
    }

    this.emit('update', symbol, this.getSnapshot(symbol));
  }

  /**
   * Get order book snapshot
   */
  getSnapshot(symbol: string, depth?: number): OrderBookSnapshot | null {
    const book = this.books.get(symbol);
    if (!book) return null;

    const bids = Array.from(book.bids.values())
      .sort((a, b) => b.price.comparedTo(a.price))
      .slice(0, depth);

    const asks = Array.from(book.asks.values())
      .sort((a, b) => a.price.comparedTo(b.price))
      .slice(0, depth);

    return {
      symbol,
      bids,
      asks,
      timestamp: book.lastUpdate,
      sequenceNumber: book.sequenceNumber
    };
  }

  /**
   * Get best bid price
   */
  getBestBid(symbol: string): Decimal | null {
    const book = this.books.get(symbol);
    if (!book || book.bids.size === 0) return null;

    let best: Decimal | null = null;
    for (const level of book.bids.values()) {
      if (!best || level.price.greaterThan(best)) {
        best = level.price;
      }
    }
    return best;
  }

  /**
   * Get best ask price
   */
  getBestAsk(symbol: string): Decimal | null {
    const book = this.books.get(symbol);
    if (!book || book.asks.size === 0) return null;

    let best: Decimal | null = null;
    for (const level of book.asks.values()) {
      if (!best || level.price.lessThan(best)) {
        best = level.price;
      }
    }
    return best;
  }

  /**
   * Get spread
   */
  getSpread(symbol: string): Decimal | null {
    const bid = this.getBestBid(symbol);
    const ask = this.getBestAsk(symbol);
    
    if (!bid || !ask) return null;
    return ask.minus(bid);
  }

  /**
   * Get mid price
   */
  getMidPrice(symbol: string): Decimal | null {
    const bid = this.getBestBid(symbol);
    const ask = this.getBestAsk(symbol);
    
    if (!bid || !ask) return null;
    return bid.plus(ask).dividedBy(2);
  }

  /**
   * Get total volume at price levels
   */
  getDepth(symbol: string, levels: number = 10): {
    bidVolume: Decimal;
    askVolume: Decimal;
    bidLevels: number;
    askLevels: number;
  } | null {
    const snapshot = this.getSnapshot(symbol, levels);
    if (!snapshot) return null;

    const bidVolume = snapshot.bids.reduce(
      (sum, level) => sum.plus(level.quantity),
      new Decimal(0)
    );

    const askVolume = snapshot.asks.reduce(
      (sum, level) => sum.plus(level.quantity),
      new Decimal(0)
    );

    return {
      bidVolume,
      askVolume,
      bidLevels: snapshot.bids.length,
      askLevels: snapshot.asks.length
    };
  }

  /**
   * Check if order book is crossed
   */
  isCrossed(symbol: string): boolean {
    const bid = this.getBestBid(symbol);
    const ask = this.getBestAsk(symbol);
    
    if (!bid || !ask) return false;
    return bid.greaterThanOrEqualTo(ask);
  }

  /**
   * Clear order book for symbol
   */
  clear(symbol: string): void {
    this.books.delete(symbol);
    this.emit('clear', symbol);
  }

  /**
   * Clear all order books
   */
  clearAll(): void {
    this.books.clear();
    this.emit('clear-all');
  }

  /**
   * Get all tracked symbols
   */
  getSymbols(): string[] {
    return Array.from(this.books.keys());
  }

  private getOrCreateBook(symbol: string): OrderBookData {
    let book = this.books.get(symbol);
    if (!book) {
      book = {
        symbol,
        bids: new Map(),
        asks: new Map(),
        lastUpdate: new Date()
      };
      this.books.set(symbol, book);
    }
    return book;
  }
}

interface OrderBookData {
  symbol: string;
  bids: Map<string, OrderBookLevel>;
  asks: Map<string, OrderBookLevel>;
  lastUpdate: Date;
  sequenceNumber?: number;
}