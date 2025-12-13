/**
 * Tests for Market Data module
 */

import {
  MarketDataProviders,
  MarketDataSource,
  LiquidationInfo,
  SettlementBatch,
  MarginInfo,
  InsuranceFundStatus,
  MarketStats,
  LiquidationRisk,
} from './marketData';

describe('MarketDataProviders', () => {
  it('should have correct provider values', () => {
    expect(MarketDataProviders.ALPACA).toBe('alpaca');
    expect(MarketDataProviders.NYSE_ARCA).toBe('nyse_arca');
    expect(MarketDataProviders.IEX_CLOUD).toBe('iex');
    expect(MarketDataProviders.POLYGON).toBe('polygon');
    expect(MarketDataProviders.CME_GROUP).toBe('cme');
    expect(MarketDataProviders.REFINITIV).toBe('refinitiv');
    expect(MarketDataProviders.ICE_DATA).toBe('ice');
    expect(MarketDataProviders.BLOOMBERG).toBe('bloomberg');
    expect(MarketDataProviders.NASDAQ_TOTALVIEW).toBe('nasdaq');
    expect(MarketDataProviders.COINBASE_PRO).toBe('coinbase');
  });

  it('should have all providers as const', () => {
    const providers = Object.values(MarketDataProviders);
    expect(providers).toHaveLength(10);
    expect(providers).toContain('alpaca');
    expect(providers).toContain('bloomberg');
  });
});

describe('MarketDataSource interface', () => {
  it('should represent market data correctly', () => {
    const source: MarketDataSource = {
      name: 'BTC-USD',
      symbol: 'BTC-USD',
      price: 50000.00,
      bid: 49999.50,
      ask: 50000.50,
      volume: 1000000,
      latencyNs: 500000,
      provider: MarketDataProviders.COINBASE_PRO
    };

    expect(source.name).toBe('BTC-USD');
    expect(source.price).toBe(50000.00);
    expect(source.bid).toBeLessThan(source.ask);
    expect(source.latencyNs).toBe(500000);
    expect(source.provider).toBe('coinbase');
  });
});

describe('LiquidationInfo interface', () => {
  it('should represent liquidation correctly', () => {
    const liq: LiquidationInfo = {
      userId: 'user123',
      positionId: 'pos456',
      symbol: 'BTC-USD',
      size: 1.5,
      liquidationPrice: 45000,
      markPrice: 44500,
      status: 'pending',
      timestamp: new Date('2025-01-01T00:00:00Z')
    };

    expect(liq.userId).toBe('user123');
    expect(liq.positionId).toBe('pos456');
    expect(liq.size).toBe(1.5);
    expect(liq.markPrice).toBeLessThan(liq.liquidationPrice);
    expect(liq.timestamp).toBeInstanceOf(Date);
  });
});

describe('SettlementBatch interface', () => {
  it('should represent settlement batch correctly', () => {
    const batch: SettlementBatch = {
      batchId: 1001,
      orderIds: [1, 2, 3, 4, 5],
      status: 'completed',
      txHash: '0xabc123',
      gasUsed: 150000,
      timestamp: new Date()
    };

    expect(batch.batchId).toBe(1001);
    expect(batch.orderIds).toHaveLength(5);
    expect(batch.status).toBe('completed');
    expect(batch.txHash).toBe('0xabc123');
    expect(batch.gasUsed).toBe(150000);
  });

  it('should allow optional txHash and gasUsed', () => {
    const batch: SettlementBatch = {
      batchId: 1002,
      orderIds: [10],
      status: 'pending',
      timestamp: new Date()
    };

    expect(batch.txHash).toBeUndefined();
    expect(batch.gasUsed).toBeUndefined();
  });
});

describe('MarginInfo interface', () => {
  it('should represent margin info correctly', () => {
    const margin: MarginInfo = {
      userId: 'trader1',
      initialMargin: 10000,
      maintenanceMargin: 5000,
      marginRatio: 0.5,
      freeMargin: 5000,
      marginLevel: 2.0
    };

    expect(margin.userId).toBe('trader1');
    expect(margin.marginRatio).toBe(0.5);
    expect(margin.marginLevel).toBe(2.0);
    expect(margin.freeMargin).toBe(margin.initialMargin - margin.maintenanceMargin);
  });
});

describe('InsuranceFundStatus interface', () => {
  it('should represent insurance fund correctly', () => {
    const fund: InsuranceFundStatus = {
      totalFund: 1000000,
      availableFund: 900000,
      usedFund: 100000,
      pendingClaims: 5,
      lastUpdate: new Date()
    };

    expect(fund.totalFund).toBe(1000000);
    expect(fund.availableFund + fund.usedFund).toBe(fund.totalFund);
    expect(fund.pendingClaims).toBe(5);
    expect(fund.lastUpdate).toBeInstanceOf(Date);
  });
});

describe('MarketStats interface', () => {
  it('should represent market stats correctly', () => {
    const stats: MarketStats = {
      symbol: 'ETH-USD',
      volume24h: 5000000000,
      high24h: 3500,
      low24h: 3200,
      priceChange24h: 100,
      priceChangePercent24h: 3.03,
      openInterest: 1000000,
      fundingRate: 0.0001,
      nextFundingTime: new Date()
    };

    expect(stats.symbol).toBe('ETH-USD');
    expect(stats.high24h).toBeGreaterThan(stats.low24h);
    expect(stats.priceChangePercent24h).toBeCloseTo(100 / 3300 * 100, 1);
    expect(stats.fundingRate).toBe(0.0001);
  });
});

describe('LiquidationRisk interface', () => {
  it('should represent low risk correctly', () => {
    const risk: LiquidationRisk = {
      userId: 'safe_trader',
      riskLevel: 'low',
      marginLevel: 5.0,
      liquidationPrice: 30000,
      timeToLiquidation: null,
      recommendations: []
    };

    expect(risk.riskLevel).toBe('low');
    expect(risk.marginLevel).toBeGreaterThan(2);
    expect(risk.timeToLiquidation).toBeNull();
    expect(risk.recommendations).toHaveLength(0);
  });

  it('should represent critical risk correctly', () => {
    const risk: LiquidationRisk = {
      userId: 'risky_trader',
      riskLevel: 'critical',
      marginLevel: 1.1,
      liquidationPrice: 49000,
      timeToLiquidation: 3600,
      recommendations: [
        'Add more margin immediately',
        'Consider closing position'
      ]
    };

    expect(risk.riskLevel).toBe('critical');
    expect(risk.marginLevel).toBeLessThan(1.5);
    expect(risk.timeToLiquidation).toBe(3600);
    expect(risk.recommendations).toHaveLength(2);
  });

  it('should support all risk levels', () => {
    const riskLevels: LiquidationRisk['riskLevel'][] = ['low', 'medium', 'high', 'critical'];

    riskLevels.forEach(level => {
      const risk: LiquidationRisk = {
        userId: 'test',
        riskLevel: level,
        marginLevel: 2.0,
        liquidationPrice: 40000,
        timeToLiquidation: null,
        recommendations: []
      };
      expect(risk.riskLevel).toBe(level);
    });
  });
});
