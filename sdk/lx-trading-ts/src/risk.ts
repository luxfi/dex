/**
 * Risk management.
 */

import { Decimal } from 'decimal.js';
import type { RiskConfig } from './config.js';
import { Side, TradingPair, type OrderRequest } from './types.js';

/**
 * Risk limit exceeded error.
 */
export class RiskError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RiskError';
  }
}

/**
 * Risk manager for enforcing trading limits.
 */
export class RiskManager {
  private positions: Map<string, Decimal> = new Map();
  private dailyPnl = new Decimal(0);
  private openOrdersCount: Map<string, number> = new Map();
  private killSwitch = false;

  constructor(public readonly config: RiskConfig) {}

  /**
   * Check if risk management is enabled.
   */
  get isEnabled(): boolean {
    return this.config.enabled;
  }

  /**
   * Check if kill switch is active.
   */
  get isKilled(): boolean {
    return this.killSwitch;
  }

  /**
   * Activate kill switch - stops all trading.
   */
  kill(): void {
    this.killSwitch = true;
  }

  /**
   * Deactivate kill switch.
   */
  reset(): void {
    this.killSwitch = false;
  }

  /**
   * Validate order against risk limits. Throws RiskError if invalid.
   */
  validateOrder(request: OrderRequest): void {
    if (!this.config.enabled) {
      return;
    }

    if (this.killSwitch) {
      throw new RiskError('Kill switch is active');
    }

    // Check order size
    if (
      this.config.maxOrderSize.gt(0) &&
      request.quantity.gt(this.config.maxOrderSize)
    ) {
      throw new RiskError(
        `Order size ${request.quantity.toString()} exceeds max ${this.config.maxOrderSize.toString()}`,
      );
    }

    // Check position limit
    const pair = TradingPair.fromSymbol(request.symbol);
    if (pair) {
      const current = this.positions.get(pair.base) ?? new Decimal(0);

      const newPosition =
        request.side === Side.BUY
          ? current.plus(request.quantity)
          : current.minus(request.quantity);

      // Asset-specific limit
      const assetLimit = this.config.positionLimits.get(pair.base);
      if (assetLimit && newPosition.abs().gt(assetLimit)) {
        throw new RiskError(
          `Position limit exceeded for ${pair.base}: ${current.toString()} + ${request.quantity.toString()} > ${assetLimit.toString()}`,
        );
      }

      // Global position limit
      if (
        this.config.maxPositionSize.gt(0) &&
        newPosition.abs().gt(this.config.maxPositionSize)
      ) {
        throw new RiskError(
          `Max position size exceeded: ${newPosition.abs().toString()} > ${this.config.maxPositionSize.toString()}`,
        );
      }
    }

    // Check open orders count
    const count = this.openOrdersCount.get(request.symbol) ?? 0;
    if (count >= this.config.maxOpenOrders) {
      throw new RiskError(
        `Max open orders (${this.config.maxOpenOrders}) reached for ${request.symbol}`,
      );
    }

    // Check daily loss
    if (this.config.maxDailyLoss.gt(0)) {
      if (this.dailyPnl.lt(this.config.maxDailyLoss.neg())) {
        throw new RiskError(
          `Daily loss limit exceeded: ${this.dailyPnl.abs().toString()} > ${this.config.maxDailyLoss.toString()}`,
        );
      }
    }
  }

  /**
   * Update position after a trade.
   */
  updatePosition(asset: string, quantity: Decimal, side: Side): void {
    const current = this.positions.get(asset) ?? new Decimal(0);
    const newPosition = side === Side.BUY ? current.plus(quantity) : current.minus(quantity);
    this.positions.set(asset, newPosition);
  }

  /**
   * Update daily PnL.
   */
  updatePnl(pnl: Decimal): void {
    this.dailyPnl = this.dailyPnl.plus(pnl);

    // Auto kill switch
    if (
      this.config.killSwitchEnabled &&
      this.config.maxDailyLoss.gt(0) &&
      this.dailyPnl.lt(this.config.maxDailyLoss.neg())
    ) {
      this.killSwitch = true;
    }
  }

  /**
   * Increment open orders count.
   */
  orderOpened(symbol: string): void {
    const count = this.openOrdersCount.get(symbol) ?? 0;
    this.openOrdersCount.set(symbol, count + 1);
  }

  /**
   * Decrement open orders count.
   */
  orderClosed(symbol: string): void {
    const count = this.openOrdersCount.get(symbol) ?? 0;
    this.openOrdersCount.set(symbol, Math.max(0, count - 1));
  }

  /**
   * Get current position for an asset.
   */
  position(asset: string): Decimal {
    return this.positions.get(asset) ?? new Decimal(0);
  }

  /**
   * Get all positions.
   */
  allPositions(): Map<string, Decimal> {
    return new Map(this.positions);
  }

  /**
   * Get daily PnL.
   */
  getDailyPnl(): Decimal {
    return this.dailyPnl;
  }

  /**
   * Reset daily PnL (call at start of trading day).
   */
  resetDailyPnl(): void {
    this.dailyPnl = new Decimal(0);
  }

  /**
   * Get open orders count for a symbol.
   */
  openOrders(symbol: string): number {
    return this.openOrdersCount.get(symbol) ?? 0;
  }
}
