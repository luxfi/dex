/**
 * Tests for LX Trading SDK risk module
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Decimal } from 'decimal.js';
import { RiskManager, RiskError } from './risk.js';
import { Side, OrderType, TimeInForce, type OrderRequest } from './types.js';
import type { RiskConfig } from './config.js';

function createRiskConfig(overrides?: Partial<RiskConfig>): RiskConfig {
  return {
    enabled: true,
    maxPositionSize: new Decimal(100),
    maxOrderSize: new Decimal(10),
    maxDailyLoss: new Decimal(1000),
    maxOpenOrders: 5,
    killSwitchEnabled: false,
    positionLimits: new Map(),
    ...overrides,
  };
}

function createOrderRequest(overrides?: Partial<OrderRequest>): OrderRequest {
  return {
    symbol: 'BTC-USDC',
    side: Side.BUY,
    orderType: OrderType.LIMIT,
    quantity: new Decimal(1),
    price: new Decimal(50000),
    timeInForce: TimeInForce.GTC,
    reduceOnly: false,
    postOnly: false,
    clientOrderId: `test-${Date.now()}`,
    ...overrides,
  };
}

describe('RiskManager', () => {
  describe('Configuration', () => {
    it('should respect enabled flag', () => {
      const manager = new RiskManager(createRiskConfig({ enabled: true }));
      assert.ok(manager.isEnabled);

      const disabledManager = new RiskManager(createRiskConfig({ enabled: false }));
      assert.ok(!disabledManager.isEnabled);
    });

    it('should skip validation when disabled', () => {
      const manager = new RiskManager(createRiskConfig({ enabled: false }));

      // Oversized order should pass when disabled
      const order = createOrderRequest({ quantity: new Decimal(1000) });
      assert.doesNotThrow(() => manager.validateOrder(order));
    });
  });

  describe('Kill switch', () => {
    it('should start inactive', () => {
      const manager = new RiskManager(createRiskConfig());
      assert.ok(!manager.isKilled);
    });

    it('should activate on kill()', () => {
      const manager = new RiskManager(createRiskConfig());
      manager.kill();
      assert.ok(manager.isKilled);
    });

    it('should deactivate on reset()', () => {
      const manager = new RiskManager(createRiskConfig());
      manager.kill();
      manager.reset();
      assert.ok(!manager.isKilled);
    });

    it('should block all orders when active', () => {
      const manager = new RiskManager(createRiskConfig());
      manager.kill();

      const order = createOrderRequest();
      assert.throws(() => manager.validateOrder(order), RiskError);
    });
  });

  describe('Order size limits', () => {
    it('should reject orders exceeding max size', () => {
      const manager = new RiskManager(createRiskConfig({ maxOrderSize: new Decimal(5) }));

      const order = createOrderRequest({ quantity: new Decimal(10) });
      assert.throws(() => manager.validateOrder(order), RiskError);
    });

    it('should allow orders within limit', () => {
      const manager = new RiskManager(createRiskConfig({ maxOrderSize: new Decimal(10) }));

      const order = createOrderRequest({ quantity: new Decimal(5) });
      assert.doesNotThrow(() => manager.validateOrder(order));
    });

    it('should allow any size when max is zero (no limit)', () => {
      const manager = new RiskManager(createRiskConfig({
        maxOrderSize: new Decimal(0),
        maxPositionSize: new Decimal(0), // Also disable position limit
      }));

      const order = createOrderRequest({ quantity: new Decimal(1000) });
      assert.doesNotThrow(() => manager.validateOrder(order));
    });
  });

  describe('Position limits', () => {
    it('should reject orders exceeding max position', () => {
      const manager = new RiskManager(createRiskConfig({ maxPositionSize: new Decimal(10) }));

      // Simulate existing position
      manager.updatePosition('BTC', new Decimal(8), Side.BUY);

      // Trying to buy 5 more would exceed limit (8 + 5 = 13 > 10)
      const order = createOrderRequest({ quantity: new Decimal(5) });
      assert.throws(() => manager.validateOrder(order), RiskError);
    });

    it('should allow orders within position limit', () => {
      const manager = new RiskManager(createRiskConfig({ maxPositionSize: new Decimal(10) }));

      manager.updatePosition('BTC', new Decimal(5), Side.BUY);

      const order = createOrderRequest({ quantity: new Decimal(3) });
      assert.doesNotThrow(() => manager.validateOrder(order));
    });

    it('should respect asset-specific limits', () => {
      const positionLimits = new Map([['BTC', new Decimal(5)]]);
      const manager = new RiskManager(createRiskConfig({ positionLimits }));

      // Existing position of 4
      manager.updatePosition('BTC', new Decimal(4), Side.BUY);

      // Trying to buy 2 more exceeds BTC-specific limit
      const order = createOrderRequest({ quantity: new Decimal(2) });
      assert.throws(() => manager.validateOrder(order), RiskError);
    });
  });

  describe('Open orders limit', () => {
    it('should reject orders when max open orders reached', () => {
      const manager = new RiskManager(createRiskConfig({ maxOpenOrders: 3 }));

      // Open 3 orders
      manager.orderOpened('BTC-USDC');
      manager.orderOpened('BTC-USDC');
      manager.orderOpened('BTC-USDC');

      const order = createOrderRequest();
      assert.throws(() => manager.validateOrder(order), RiskError);
    });

    it('should allow orders when under limit', () => {
      const manager = new RiskManager(createRiskConfig({ maxOpenOrders: 5 }));

      manager.orderOpened('BTC-USDC');
      manager.orderOpened('BTC-USDC');

      const order = createOrderRequest();
      assert.doesNotThrow(() => manager.validateOrder(order));
    });

    it('should track orders per symbol', () => {
      const manager = new RiskManager(createRiskConfig({ maxOpenOrders: 2 }));

      manager.orderOpened('BTC-USDC');
      manager.orderOpened('BTC-USDC');

      // Different symbol should still be allowed
      const order = createOrderRequest({ symbol: 'ETH-USDC' });
      assert.doesNotThrow(() => manager.validateOrder(order));
    });

    it('should decrement on order close', () => {
      const manager = new RiskManager(createRiskConfig({ maxOpenOrders: 2 }));

      manager.orderOpened('BTC-USDC');
      manager.orderOpened('BTC-USDC');
      manager.orderClosed('BTC-USDC');

      const order = createOrderRequest();
      assert.doesNotThrow(() => manager.validateOrder(order));
    });
  });

  describe('Daily loss limit', () => {
    it('should reject orders when daily loss exceeded', () => {
      const manager = new RiskManager(createRiskConfig({ maxDailyLoss: new Decimal(100) }));

      // Record loss exceeding daily limit
      manager.updatePnl(new Decimal(-150));

      const order = createOrderRequest();
      assert.throws(() => manager.validateOrder(order), RiskError);
    });

    it('should allow orders when within daily loss limit', () => {
      const manager = new RiskManager(createRiskConfig({ maxDailyLoss: new Decimal(100) }));

      manager.updatePnl(new Decimal(-50));

      const order = createOrderRequest();
      assert.doesNotThrow(() => manager.validateOrder(order));
    });

    it('should auto-trigger kill switch when enabled', () => {
      const manager = new RiskManager(
        createRiskConfig({
          maxDailyLoss: new Decimal(100),
          killSwitchEnabled: true,
        }),
      );

      manager.updatePnl(new Decimal(-150));

      assert.ok(manager.isKilled);
    });

    it('should not trigger kill switch when disabled', () => {
      const manager = new RiskManager(
        createRiskConfig({
          maxDailyLoss: new Decimal(100),
          killSwitchEnabled: false,
        }),
      );

      manager.updatePnl(new Decimal(-150));

      assert.ok(!manager.isKilled);
    });
  });

  describe('Position tracking', () => {
    it('should track position after buys', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.updatePosition('BTC', new Decimal(5), Side.BUY);
      assert.ok(manager.position('BTC').eq(5));

      manager.updatePosition('BTC', new Decimal(3), Side.BUY);
      assert.ok(manager.position('BTC').eq(8));
    });

    it('should track position after sells', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.updatePosition('BTC', new Decimal(10), Side.BUY);
      manager.updatePosition('BTC', new Decimal(3), Side.SELL);
      assert.ok(manager.position('BTC').eq(7));
    });

    it('should allow negative positions (shorts)', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.updatePosition('BTC', new Decimal(5), Side.SELL);
      assert.ok(manager.position('BTC').eq(-5));
    });

    it('should return zero for unknown assets', () => {
      const manager = new RiskManager(createRiskConfig());
      assert.ok(manager.position('UNKNOWN').eq(0));
    });

    it('should return all positions', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.updatePosition('BTC', new Decimal(5), Side.BUY);
      manager.updatePosition('ETH', new Decimal(10), Side.BUY);

      const positions = manager.allPositions();
      assert.strictEqual(positions.size, 2);
      assert.ok(positions.get('BTC')?.eq(5));
      assert.ok(positions.get('ETH')?.eq(10));
    });
  });

  describe('PnL tracking', () => {
    it('should track daily PnL', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.updatePnl(new Decimal(50));
      assert.ok(manager.getDailyPnl().eq(50));

      manager.updatePnl(new Decimal(-20));
      assert.ok(manager.getDailyPnl().eq(30));
    });

    it('should reset daily PnL', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.updatePnl(new Decimal(100));
      manager.resetDailyPnl();

      assert.ok(manager.getDailyPnl().eq(0));
    });
  });

  describe('Open orders tracking', () => {
    it('should track open orders count', () => {
      const manager = new RiskManager(createRiskConfig());

      assert.strictEqual(manager.openOrders('BTC-USDC'), 0);

      manager.orderOpened('BTC-USDC');
      manager.orderOpened('BTC-USDC');
      assert.strictEqual(manager.openOrders('BTC-USDC'), 2);

      manager.orderClosed('BTC-USDC');
      assert.strictEqual(manager.openOrders('BTC-USDC'), 1);
    });

    it('should not go below zero', () => {
      const manager = new RiskManager(createRiskConfig());

      manager.orderClosed('BTC-USDC');
      manager.orderClosed('BTC-USDC');

      assert.strictEqual(manager.openOrders('BTC-USDC'), 0);
    });
  });

  describe('RiskError', () => {
    it('should have correct name', () => {
      const error = new RiskError('test message');
      assert.strictEqual(error.name, 'RiskError');
      assert.strictEqual(error.message, 'test message');
    });
  });
});
