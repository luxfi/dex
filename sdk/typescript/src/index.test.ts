/**
 * Tests for LX TypeScript SDK
 */

import {
  OrderType,
  OrderSide,
  OrderStatus,
  TimeInForce,
  LXDexClient,
  Order,
  OrderBookLevel,
} from './index';

describe('OrderType enum', () => {
  it('should have correct values', () => {
    expect(OrderType.LIMIT).toBe(0);
    expect(OrderType.MARKET).toBe(1);
    expect(OrderType.STOP).toBe(2);
    expect(OrderType.STOP_LIMIT).toBe(3);
    expect(OrderType.ICEBERG).toBe(4);
    expect(OrderType.PEG).toBe(5);
  });
});

describe('OrderSide enum', () => {
  it('should have correct values', () => {
    expect(OrderSide.BUY).toBe(0);
    expect(OrderSide.SELL).toBe(1);
  });
});

describe('OrderStatus enum', () => {
  it('should have correct string values', () => {
    expect(OrderStatus.OPEN).toBe('open');
    expect(OrderStatus.PARTIAL).toBe('partial');
    expect(OrderStatus.FILLED).toBe('filled');
    expect(OrderStatus.CANCELLED).toBe('cancelled');
    expect(OrderStatus.REJECTED).toBe('rejected');
  });
});

describe('TimeInForce enum', () => {
  it('should have correct values', () => {
    expect(TimeInForce.GTC).toBe('GTC');
    expect(TimeInForce.IOC).toBe('IOC');
    expect(TimeInForce.FOK).toBe('FOK');
    expect(TimeInForce.DAY).toBe('DAY');
  });
});

describe('LXDexClient', () => {
  describe('constructor', () => {
    it('should create client with default config', () => {
      const client = new LXDexClient();
      expect(client).toBeDefined();
      expect(client.marketData).toBeDefined();
      expect(client.liquidationMonitor).toBeDefined();
    });

    it('should create client with custom config', () => {
      const client = new LXDexClient({
        jsonRpcUrl: 'http://custom:8080',
        wsUrl: 'ws://custom:8081',
        apiKey: 'test-key'
      });
      expect(client).toBeDefined();
    });
  });

  describe('static utility methods', () => {
    it('formatPrice should format correctly', () => {
      expect(LXDexClient.formatPrice(50000.12345)).toBe('50000.12');
      expect(LXDexClient.formatPrice(50000.12346, 4)).toBe('50000.1235');
      expect(LXDexClient.formatPrice(100, 0)).toBe('100');
    });

    it('formatSize should format correctly', () => {
      expect(LXDexClient.formatSize(1.123456789)).toBe('1.12345679');
      expect(LXDexClient.formatSize(1.5, 2)).toBe('1.50');
    });

    it('calculateTotal should compute price * size', () => {
      expect(LXDexClient.calculateTotal(50000, 1)).toBe(50000);
      expect(LXDexClient.calculateTotal(50000, 0.5)).toBe(25000);
      expect(LXDexClient.calculateTotal(100.5, 2)).toBe(201);
    });
  });

  describe('disconnect', () => {
    it('should handle disconnect when not connected', () => {
      const client = new LXDexClient();
      expect(() => client.disconnect()).not.toThrow();
    });
  });

  describe('subscribe', () => {
    it('should add callback to callbacks map', () => {
      const client = new LXDexClient();
      const callback = jest.fn();
      client.subscribe('test-channel', callback);
      // The callback should be registered (internal state)
      expect(() => client.unsubscribe('test-channel')).not.toThrow();
    });

    it('should handle multiple callbacks for same channel', () => {
      const client = new LXDexClient();
      const callback1 = jest.fn();
      const callback2 = jest.fn();
      client.subscribe('test-channel', callback1);
      client.subscribe('test-channel', callback2);
      expect(() => client.unsubscribe('test-channel')).not.toThrow();
    });
  });

  describe('unsubscribe', () => {
    it('should remove specific callback', () => {
      const client = new LXDexClient();
      const callback = jest.fn();
      client.subscribe('test-channel', callback);
      expect(() => client.unsubscribe('test-channel', callback)).not.toThrow();
    });

    it('should remove all callbacks for channel', () => {
      const client = new LXDexClient();
      const callback1 = jest.fn();
      const callback2 = jest.fn();
      client.subscribe('test-channel', callback1);
      client.subscribe('test-channel', callback2);
      expect(() => client.unsubscribe('test-channel')).not.toThrow();
    });

    it('should handle unsubscribe for non-existent channel', () => {
      const client = new LXDexClient();
      expect(() => client.unsubscribe('non-existent')).not.toThrow();
    });
  });
});

describe('Order interface', () => {
  it('should allow creating a valid order', () => {
    const order: Order = {
      symbol: 'BTC-USD',
      type: OrderType.LIMIT,
      side: OrderSide.BUY,
      price: 50000,
      size: 1.0
    };

    expect(order.symbol).toBe('BTC-USD');
    expect(order.type).toBe(OrderType.LIMIT);
    expect(order.side).toBe(OrderSide.BUY);
    expect(order.price).toBe(50000);
    expect(order.size).toBe(1.0);
  });

  it('should allow optional fields', () => {
    const order: Order = {
      symbol: 'ETH-USD',
      type: OrderType.MARKET,
      side: OrderSide.SELL,
      price: 0,
      size: 2.5,
      orderId: 12345,
      timeInForce: TimeInForce.IOC,
      postOnly: true,
      reduceOnly: false,
      status: OrderStatus.OPEN,
      filled: 0,
      remaining: 2.5
    };

    expect(order.orderId).toBe(12345);
    expect(order.timeInForce).toBe(TimeInForce.IOC);
    expect(order.postOnly).toBe(true);
  });
});

describe('OrderBookLevel interface', () => {
  it('should represent price level correctly', () => {
    const level: OrderBookLevel = {
      price: 50000,
      size: 10.5,
      count: 5
    };

    expect(level.price).toBe(50000);
    expect(level.size).toBe(10.5);
    expect(level.count).toBe(5);
  });

  it('should allow optional count', () => {
    const level: OrderBookLevel = {
      price: 49999,
      size: 5.0
    };

    expect(level.price).toBe(49999);
    expect(level.size).toBe(5.0);
    expect(level.count).toBeUndefined();
  });
});
