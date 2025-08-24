/**
 * Portfolio Management for Lux Exchange
 */

import Decimal from 'decimal.js';
import { EventEmitter } from 'eventemitter3';
import { Position, Balance, Order, Trade, AssetClass } from './types';

export class Portfolio extends EventEmitter {
  private positions: Map<string, Position> = new Map();
  private balances: Map<string, Balance> = new Map();
  private orders: Map<string, Order> = new Map();
  private trades: Trade[] = [];

  /**
   * Update position
   */
  updatePosition(position: Position): void {
    this.positions.set(position.symbol, position);
    this.emit('position', position);
  }

  /**
   * Update balance
   */
  updateBalance(balance: Balance): void {
    this.balances.set(balance.currency, balance);
    this.emit('balance', balance);
  }

  /**
   * Add or update order
   */
  updateOrder(order: Order): void {
    this.orders.set(order.orderId, order);
    this.emit('order', order);
  }

  /**
   * Add trade
   */
  addTrade(trade: Trade): void {
    this.trades.push(trade);
    // Keep only recent trades (last 1000)
    if (this.trades.length > 1000) {
      this.trades = this.trades.slice(-1000);
    }
    this.emit('trade', trade);
  }

  /**
   * Get all positions
   */
  getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get position for symbol
   */
  getPosition(symbol: string): Position | null {
    return this.positions.get(symbol) || null;
  }

  /**
   * Get all balances
   */
  getBalances(): Balance[] {
    return Array.from(this.balances.values());
  }

  /**
   * Get balance for currency
   */
  getBalance(currency: string): Balance | null {
    return this.balances.get(currency) || null;
  }

  /**
   * Get open orders
   */
  getOpenOrders(): Order[] {
    return Array.from(this.orders.values()).filter(
      order => order.status === 'NEW' || order.status === 'PARTIALLY_FILLED'
    );
  }

  /**
   * Get recent trades
   */
  getRecentTrades(limit: number = 100): Trade[] {
    return this.trades.slice(-limit);
  }

  /**
   * Calculate total portfolio value in USD
   */
  getTotalValueUsd(prices: Map<string, Decimal>): Decimal {
    let total = new Decimal(0);

    // Add balance values
    for (const balance of this.balances.values()) {
      const price = prices.get(balance.currency) || new Decimal(1);
      total = total.plus(balance.total.times(price));
    }

    // Add position values
    for (const position of this.positions.values()) {
      const value = position.quantity.times(position.markPrice);
      total = total.plus(value);
    }

    return total;
  }

  /**
   * Calculate total PnL
   */
  getTotalPnl(): {
    realized: Decimal;
    unrealized: Decimal;
    total: Decimal;
  } {
    let realizedPnl = new Decimal(0);
    let unrealizedPnl = new Decimal(0);

    for (const position of this.positions.values()) {
      realizedPnl = realizedPnl.plus(position.realizedPnl);
      unrealizedPnl = unrealizedPnl.plus(position.unrealizedPnl);
    }

    return {
      realized: realizedPnl,
      unrealized: unrealizedPnl,
      total: realizedPnl.plus(unrealizedPnl)
    };
  }

  /**
   * Get positions by asset class
   */
  getPositionsByAssetClass(assetClass: AssetClass): Position[] {
    return Array.from(this.positions.values()).filter(
      position => position.assetClass === assetClass
    );
  }

  /**
   * Calculate exposure by asset class
   */
  getExposureByAssetClass(): Map<AssetClass, Decimal> {
    const exposure = new Map<AssetClass, Decimal>();

    for (const position of this.positions.values()) {
      const current = exposure.get(position.assetClass) || new Decimal(0);
      const value = position.quantity.times(position.markPrice);
      exposure.set(position.assetClass, current.plus(value));
    }

    return exposure;
  }

  /**
   * Get margin utilization
   */
  getMarginUtilization(): {
    used: Decimal;
    free: Decimal;
    level: Decimal;
  } | null {
    let totalMargin = new Decimal(0);
    let usedMargin = new Decimal(0);

    for (const position of this.positions.values()) {
      usedMargin = usedMargin.plus(position.margin);
    }

    // Calculate from balances (simplified)
    for (const balance of this.balances.values()) {
      if (balance.currency === 'USD' || balance.currency === 'USDT') {
        totalMargin = totalMargin.plus(balance.total);
      }
    }

    if (totalMargin.isZero()) return null;

    return {
      used: usedMargin,
      free: totalMargin.minus(usedMargin),
      level: usedMargin.dividedBy(totalMargin).times(100)
    };
  }

  /**
   * Clear all portfolio data
   */
  clear(): void {
    this.positions.clear();
    this.balances.clear();
    this.orders.clear();
    this.trades = [];
    this.emit('clear');
  }
}