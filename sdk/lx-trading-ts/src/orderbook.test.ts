/**
 * Tests for LX Trading SDK orderbook module
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Decimal } from 'decimal.js';
import { Orderbook, AggregatedOrderbook } from './orderbook.js';
import { Side } from './types.js';

// Helper for floating point comparison
function approxEqual(actual: number, expected: number, epsilon = 0.01): boolean {
  return Math.abs(actual - expected) < epsilon;
}

describe('Orderbook', () => {
  describe('Basic operations', () => {
    it('should create with symbol and venue', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      assert.strictEqual(book.symbol, 'BTC-USDC');
      assert.strictEqual(book.venue, 'lx');
      assert.ok(book.timestamp > 0);
    });

    it('should add and retrieve bids', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addBid(new Decimal('99'), new Decimal('2'));
      book.sort();

      assert.strictEqual(book.bids.length, 2);
      assert.ok(book.bids[0]?.price.eq(100)); // Highest bid first
      assert.ok(book.bids[1]?.price.eq(99));
    });

    it('should add and retrieve asks', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('102'), new Decimal('1.5'));
      book.addAsk(new Decimal('101'), new Decimal('2.5'));
      book.sort();

      assert.strictEqual(book.asks.length, 2);
      assert.ok(book.asks[0]?.price.eq(101)); // Lowest ask first
      assert.ok(book.asks[1]?.price.eq(102));
    });

    it('should sort bids descending and asks ascending', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('98'), new Decimal('1'));
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addBid(new Decimal('99'), new Decimal('1'));
      book.addAsk(new Decimal('103'), new Decimal('1'));
      book.addAsk(new Decimal('101'), new Decimal('1'));
      book.addAsk(new Decimal('102'), new Decimal('1'));
      book.sort();

      assert.ok(book.bids[0]?.price.eq(100));
      assert.ok(book.bids[1]?.price.eq(99));
      assert.ok(book.bids[2]?.price.eq(98));

      assert.ok(book.asks[0]?.price.eq(101));
      assert.ok(book.asks[1]?.price.eq(102));
      assert.ok(book.asks[2]?.price.eq(103));
    });
  });

  describe('Price properties', () => {
    it('should return best bid', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addBid(new Decimal('99'), new Decimal('2'));
      book.sort();

      assert.ok(book.bestBid?.eq(100));
    });

    it('should return best ask', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('101'), new Decimal('1'));
      book.addAsk(new Decimal('102'), new Decimal('2'));
      book.sort();

      assert.ok(book.bestAsk?.eq(101));
    });

    it('should return undefined when empty', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      assert.strictEqual(book.bestBid, undefined);
      assert.strictEqual(book.bestAsk, undefined);
    });

    it('should calculate mid price', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addAsk(new Decimal('102'), new Decimal('1'));
      book.sort();

      assert.ok(book.midPrice?.eq(101));
    });

    it('should calculate spread', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addAsk(new Decimal('102'), new Decimal('1'));
      book.sort();

      assert.ok(book.spread?.eq(2));
    });

    it('should calculate spread percent', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addAsk(new Decimal('102'), new Decimal('1'));
      book.sort();

      // spread = 2, mid = 101, spread% = 2/101*100 ≈ 1.98
      const spreadPct = book.spreadPercent;
      assert.ok(spreadPct);
      assert.ok(approxEqual(spreadPct.toNumber(), 1.98, 0.01));
    });
  });

  describe('Liquidity', () => {
    it('should calculate bid liquidity', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1')); // 100 value
      book.addBid(new Decimal('99'), new Decimal('2')); // 198 value
      book.sort();

      // Total = 100 + 198 = 298
      assert.ok(book.bidLiquidity.eq(298));
    });

    it('should calculate ask liquidity', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('101'), new Decimal('1.5')); // 151.5 value
      book.addAsk(new Decimal('102'), new Decimal('2.5')); // 255 value
      book.sort();

      // Total = 151.5 + 255 = 406.5
      assert.ok(book.askLiquidity.eq(406.5));
    });

    it('should calculate bid depth for specific levels', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1')); // 100 value
      book.addBid(new Decimal('99'), new Decimal('2')); // 198 value
      book.addBid(new Decimal('98'), new Decimal('3')); // 294 value
      book.sort();

      assert.ok(book.bidDepth(1).eq(100)); // First level only
      assert.ok(book.bidDepth(2).eq(298)); // First two levels
    });

    it('should calculate ask depth for specific levels', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('101'), new Decimal('1.5')); // 151.5
      book.addAsk(new Decimal('102'), new Decimal('2.5')); // 255
      book.sort();

      assert.ok(book.askDepth(1).eq(151.5));
    });
  });

  describe('VWAP', () => {
    it('should calculate VWAP for small buy', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('100'), new Decimal('1'));
      book.addAsk(new Decimal('101'), new Decimal('2'));
      book.addAsk(new Decimal('102'), new Decimal('3'));
      book.sort();

      // Buying 0.5 at 100 = VWAP 100
      const vwap = book.vwapBuy(new Decimal('0.5'));
      assert.ok(vwap?.eq(100));
    });

    it('should calculate VWAP across multiple levels', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('100'), new Decimal('1'));
      book.addAsk(new Decimal('101'), new Decimal('2'));
      book.addAsk(new Decimal('102'), new Decimal('3'));
      book.sort();

      // Buying 2.5: 1@100 + 1.5@101 = 100 + 151.5 = 251.5 / 2.5 = 100.6
      const vwap = book.vwapBuy(new Decimal('2.5'));
      assert.ok(vwap);
      assert.ok(approxEqual(vwap.toNumber(), 100.6, 0.01));
    });

    it('should calculate VWAP for sell', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('1'));
      book.addBid(new Decimal('99'), new Decimal('2'));
      book.sort();

      // Selling 0.5 at 100 = VWAP 100
      const vwap = book.vwapSell(new Decimal('0.5'));
      assert.ok(vwap?.eq(100));
    });

    it('should return undefined for empty book', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      assert.strictEqual(book.vwapBuy(new Decimal('1')), undefined);
      assert.strictEqual(book.vwapSell(new Decimal('1')), undefined);
    });
  });

  describe('Liquidity check', () => {
    it('should detect sufficient liquidity for buy', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addAsk(new Decimal('100'), new Decimal('5'));
      book.sort();

      assert.ok(book.hasLiquidity(Side.BUY, new Decimal('3')));
      assert.ok(!book.hasLiquidity(Side.BUY, new Decimal('10')));
    });

    it('should detect sufficient liquidity for sell', () => {
      const book = new Orderbook('BTC-USDC', 'lx');
      book.addBid(new Decimal('100'), new Decimal('5'));
      book.sort();

      assert.ok(book.hasLiquidity(Side.SELL, new Decimal('3')));
      assert.ok(!book.hasLiquidity(Side.SELL, new Decimal('10')));
    });
  });
});

describe('AggregatedOrderbook', () => {
  function createTestBooks(): [Orderbook, Orderbook] {
    const book1 = new Orderbook('BTC-USDC', 'venue1');
    book1.addBid(new Decimal('100'), new Decimal('1'));
    book1.addAsk(new Decimal('102'), new Decimal('1'));
    book1.sort();

    const book2 = new Orderbook('BTC-USDC', 'venue2');
    book2.addBid(new Decimal('99'), new Decimal('2'));
    book2.addAsk(new Decimal('101'), new Decimal('1.5'));
    book2.sort();

    return [book1, book2];
  }

  it('should aggregate orderbooks from multiple venues', () => {
    const agg = new AggregatedOrderbook('BTC-USDC');
    const [book1, book2] = createTestBooks();

    agg.addOrderbook(book1);
    agg.addOrderbook(book2);

    assert.strictEqual(agg.bids.size, 2);
    assert.strictEqual(agg.asks.size, 2);
  });

  describe('Best bid/ask', () => {
    it('should find best bid across venues', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      const [book1, book2] = createTestBooks();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const best = agg.bestBid();
      assert.ok(best);
      assert.ok(best.price.eq(100));
      assert.strictEqual(best.venue, 'venue1');
    });

    it('should find best ask across venues', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      const [book1, book2] = createTestBooks();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const best = agg.bestAsk();
      assert.ok(best);
      assert.ok(best.price.eq(101));
      assert.strictEqual(best.venue, 'venue2');
    });

    it('should return undefined for empty book', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      assert.strictEqual(agg.bestBid(), undefined);
      assert.strictEqual(agg.bestAsk(), undefined);
    });
  });

  describe('Aggregated levels', () => {
    it('should aggregate bids sorted descending', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      const [book1, book2] = createTestBooks();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const bids = agg.aggregatedBids();
      assert.strictEqual(bids.length, 2);
      assert.ok(bids[0]?.price.eq(100)); // Highest first
      assert.ok(bids[1]?.price.eq(99));
    });

    it('should aggregate asks sorted ascending', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      const [book1, book2] = createTestBooks();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const asks = agg.aggregatedAsks();
      assert.strictEqual(asks.length, 2);
      assert.ok(asks[0]?.price.eq(101)); // Lowest first
      assert.ok(asks[1]?.price.eq(102));
    });

    it('should sum quantities at same price from different venues', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');

      const book1 = new Orderbook('BTC-USDC', 'venue1');
      book1.addBid(new Decimal('100'), new Decimal('1'));
      book1.sort();

      const book2 = new Orderbook('BTC-USDC', 'venue2');
      book2.addBid(new Decimal('100'), new Decimal('2'));
      book2.sort();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const bids = agg.aggregatedBids();
      assert.strictEqual(bids.length, 1);
      assert.ok(bids[0]?.quantity.eq(3)); // 1 + 2
    });
  });

  describe('Best venue routing', () => {
    it('should find best venue for buying', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      const [book1, book2] = createTestBooks();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const best = agg.bestVenueBuy(new Decimal('1'));
      assert.ok(best);
      assert.strictEqual(best.venue, 'venue2'); // Lower ask at 101
      assert.ok(best.price.eq(101));
    });

    it('should find best venue for selling', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      const [book1, book2] = createTestBooks();

      agg.addOrderbook(book1);
      agg.addOrderbook(book2);

      const best = agg.bestVenueSell(new Decimal('0.5'));
      assert.ok(best);
      assert.strictEqual(best.venue, 'venue1'); // Higher bid at 100
      assert.ok(best.price.eq(100));
    });

    it('should return undefined for empty book', () => {
      const agg = new AggregatedOrderbook('BTC-USDC');
      assert.strictEqual(agg.bestVenueBuy(new Decimal('1')), undefined);
      assert.strictEqual(agg.bestVenueSell(new Decimal('1')), undefined);
    });
  });
});
