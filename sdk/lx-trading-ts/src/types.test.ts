/**
 * Tests for LX Trading SDK types
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Decimal } from 'decimal.js';
import {
  TradingPair,
  Side,
  OrderType,
  OrderStatus,
  TimeInForce,
  balanceTotal,
  aggregatedBalanceTotal,
  createMarketOrder,
  createLimitOrder,
  orderIsOpen,
  orderIsDone,
  orderFillPercent,
  tradeValue,
  tickerMidPrice,
  tickerSpread,
  tickerSpreadPercent,
  priceLevelValue,
  type Balance,
  type AggregatedBalance,
  type Order,
  type Trade,
  type Ticker,
  type PriceLevel,
} from './types.js';

describe('TradingPair', () => {
  it('should parse symbol with dash separator', () => {
    const pair = TradingPair.fromSymbol('BTC-USDC');
    assert.ok(pair);
    assert.strictEqual(pair.base, 'BTC');
    assert.strictEqual(pair.quote, 'USDC');
  });

  it('should parse symbol with slash separator', () => {
    const pair = TradingPair.fromSymbol('ETH/USD');
    assert.ok(pair);
    assert.strictEqual(pair.base, 'ETH');
    assert.strictEqual(pair.quote, 'USD');
  });

  it('should parse symbol with underscore separator', () => {
    const pair = TradingPair.fromSymbol('SOL_USDT');
    assert.ok(pair);
    assert.strictEqual(pair.base, 'SOL');
    assert.strictEqual(pair.quote, 'USDT');
  });

  it('should return null for invalid symbol', () => {
    const pair = TradingPair.fromSymbol('BTCUSDC');
    assert.strictEqual(pair, null);
  });

  it('should format to Hummingbot style', () => {
    const pair = new TradingPair('BTC', 'USDC');
    assert.strictEqual(pair.toHummingbot(), 'BTC-USDC');
  });

  it('should format to CCXT style', () => {
    const pair = new TradingPair('ETH', 'USD');
    assert.strictEqual(pair.toCcxt(), 'ETH/USD');
  });
});

describe('Balance functions', () => {
  it('should calculate total balance', () => {
    const balance: Balance = {
      asset: 'BTC',
      venue: 'binance',
      free: new Decimal('1.5'),
      locked: new Decimal('0.5'),
    };
    assert.strictEqual(balanceTotal(balance).toString(), '2');
  });

  it('should calculate aggregated balance total', () => {
    const aggBalance: AggregatedBalance = {
      asset: 'ETH',
      totalFree: new Decimal('10'),
      totalLocked: new Decimal('5'),
      byVenue: [],
    };
    assert.strictEqual(aggregatedBalanceTotal(aggBalance).toString(), '15');
  });
});

describe('Order functions', () => {
  it('should create market order', () => {
    const order = createMarketOrder('BTC-USDC', Side.BUY, new Decimal('0.5'));
    assert.strictEqual(order.symbol, 'BTC-USDC');
    assert.strictEqual(order.side, Side.BUY);
    assert.strictEqual(order.orderType, OrderType.MARKET);
    assert.strictEqual(order.quantity.toString(), '0.5');
    assert.strictEqual(order.timeInForce, TimeInForce.IOC);
    assert.ok(order.clientOrderId);
  });

  it('should create limit order', () => {
    const order = createLimitOrder('ETH-USD', Side.SELL, new Decimal('1'), new Decimal('2000'));
    assert.strictEqual(order.symbol, 'ETH-USD');
    assert.strictEqual(order.side, Side.SELL);
    assert.strictEqual(order.orderType, OrderType.LIMIT);
    assert.strictEqual(order.price?.toString(), '2000');
    assert.strictEqual(order.timeInForce, TimeInForce.GTC);
  });

  it('should detect open orders', () => {
    const openOrder: Order = {
      orderId: '1',
      clientOrderId: 'c1',
      symbol: 'BTC-USDC',
      venue: 'lx',
      side: Side.BUY,
      orderType: OrderType.LIMIT,
      status: OrderStatus.OPEN,
      quantity: new Decimal('1'),
      filledQuantity: new Decimal('0'),
      remainingQuantity: new Decimal('1'),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      fees: [],
    };
    assert.ok(orderIsOpen(openOrder));
    assert.ok(!orderIsDone(openOrder));
  });

  it('should detect done orders', () => {
    const filledOrder: Order = {
      orderId: '2',
      clientOrderId: 'c2',
      symbol: 'ETH-USD',
      venue: 'lx',
      side: Side.SELL,
      orderType: OrderType.MARKET,
      status: OrderStatus.FILLED,
      quantity: new Decimal('1'),
      filledQuantity: new Decimal('1'),
      remainingQuantity: new Decimal('0'),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      fees: [],
    };
    assert.ok(!orderIsOpen(filledOrder));
    assert.ok(orderIsDone(filledOrder));
  });

  it('should calculate fill percent', () => {
    const partialOrder: Order = {
      orderId: '3',
      clientOrderId: 'c3',
      symbol: 'BTC-USDC',
      venue: 'lx',
      side: Side.BUY,
      orderType: OrderType.LIMIT,
      status: OrderStatus.PARTIALLY_FILLED,
      quantity: new Decimal('10'),
      filledQuantity: new Decimal('3'),
      remainingQuantity: new Decimal('7'),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      fees: [],
    };
    assert.strictEqual(orderFillPercent(partialOrder).toString(), '30');
  });
});

describe('Trade functions', () => {
  it('should calculate trade value', () => {
    const trade: Trade = {
      tradeId: 't1',
      orderId: 'o1',
      symbol: 'BTC-USDC',
      venue: 'lx',
      side: Side.BUY,
      price: new Decimal('50000'),
      quantity: new Decimal('0.1'),
      fee: { asset: 'USDC', amount: new Decimal('5') },
      timestamp: Date.now(),
      isMaker: false,
    };
    assert.strictEqual(tradeValue(trade).toString(), '5000');
  });
});

describe('Ticker functions', () => {
  const ticker: Ticker = {
    symbol: 'BTC-USDC',
    venue: 'lx',
    bid: new Decimal('49900'),
    ask: new Decimal('50100'),
    last: new Decimal('50000'),
    timestamp: Date.now(),
  };

  it('should calculate mid price', () => {
    const mid = tickerMidPrice(ticker);
    assert.strictEqual(mid?.toString(), '50000');
  });

  it('should calculate spread', () => {
    const spread = tickerSpread(ticker);
    assert.strictEqual(spread?.toString(), '200');
  });

  it('should calculate spread percent', () => {
    const spreadPct = tickerSpreadPercent(ticker);
    // (200 / 49900) * 100 ≈ 0.401
    assert.ok(spreadPct && spreadPct.gt(0.4) && spreadPct.lt(0.41));
  });

  it('should return last as mid when bid/ask missing', () => {
    const tickerNoSpread: Ticker = {
      symbol: 'ETH-USD',
      venue: 'lx',
      last: new Decimal('2000'),
      timestamp: Date.now(),
    };
    assert.strictEqual(tickerMidPrice(tickerNoSpread)?.toString(), '2000');
  });
});

describe('PriceLevel functions', () => {
  it('should calculate price level value', () => {
    const level: PriceLevel = {
      price: new Decimal('100'),
      quantity: new Decimal('5'),
    };
    assert.strictEqual(priceLevelValue(level).toString(), '500');
  });
});
