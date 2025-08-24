/**
 * Market data, liquidation, and settlement features for LX DEX TypeScript SDK
 */

export interface MarketDataSource {
  name: string;
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  latencyNs: number;
  provider: string;
}

export interface LiquidationInfo {
  userId: string;
  positionId: string;
  symbol: string;
  size: number;
  liquidationPrice: number;
  markPrice: number;
  status: string;
  timestamp: Date;
}

export interface SettlementBatch {
  batchId: number;
  orderIds: number[];
  status: string;
  txHash?: string;
  gasUsed?: number;
  timestamp: Date;
}

export interface MarginInfo {
  userId: string;
  initialMargin: number;
  maintenanceMargin: number;
  marginRatio: number;
  freeMargin: number;
  marginLevel: number;
}

export interface InsuranceFundStatus {
  totalFund: number;
  availableFund: number;
  usedFund: number;
  pendingClaims: number;
  lastUpdate: Date;
}

export interface MarketStats {
  symbol: string;
  volume24h: number;
  high24h: number;
  low24h: number;
  priceChange24h: number;
  priceChangePercent24h: number;
  openInterest: number;
  fundingRate: number;
  nextFundingTime: Date;
}

export interface LiquidationRisk {
  userId: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  marginLevel: number;
  liquidationPrice: number;
  timeToLiquidation: number | null;
  recommendations: string[];
}

export class MarketDataClient {
  private jsonRpc: any;

  constructor(jsonRpcClient: any) {
    this.jsonRpc = jsonRpcClient;
  }

  /**
   * Get market data from a specific source
   */
  async getMarketData(symbol: string, source: string): Promise<MarketDataSource> {
    const result = await this.jsonRpc.call('market_data.get', {
      symbol,
      source
    });

    return {
      name: result.name,
      symbol: result.symbol,
      price: result.price,
      bid: result.bid,
      ask: result.ask,
      volume: result.volume,
      latencyNs: result.latency_ns,
      provider: result.provider
    };
  }

  /**
   * Get aggregated market data from all sources
   */
  async getAggregatedMarketData(symbol: string): Promise<MarketDataSource[]> {
    const result = await this.jsonRpc.call('market_data.aggregate', {
      symbol
    });

    return result.map((data: any) => ({
      name: data.name,
      symbol: data.symbol,
      price: data.price,
      bid: data.bid,
      ask: data.ask,
      volume: data.volume,
      latencyNs: data.latency_ns,
      provider: data.provider
    }));
  }

  /**
   * Get recent liquidations
   */
  async getLiquidations(symbol: string, limit: number = 100): Promise<LiquidationInfo[]> {
    const result = await this.jsonRpc.call('liquidations.get', {
      symbol,
      limit
    });

    return result.map((liq: any) => ({
      userId: liq.user_id,
      positionId: liq.position_id,
      symbol: liq.symbol,
      size: liq.size,
      liquidationPrice: liq.liquidation_price,
      markPrice: liq.mark_price,
      status: liq.status,
      timestamp: new Date(liq.timestamp)
    }));
  }

  /**
   * Get settlement batch information
   */
  async getSettlementBatch(batchId: number): Promise<SettlementBatch> {
    const result = await this.jsonRpc.call('settlement.batch', {
      batch_id: batchId
    });

    return {
      batchId: result.batch_id,
      orderIds: result.order_ids,
      status: result.status,
      txHash: result.tx_hash,
      gasUsed: result.gas_used,
      timestamp: new Date(result.timestamp)
    };
  }

  /**
   * Get margin information for a user
   */
  async getMarginInfo(userId: string): Promise<MarginInfo> {
    const result = await this.jsonRpc.call('margin.info', {
      user_id: userId
    });

    return {
      userId: result.user_id,
      initialMargin: result.initial_margin,
      maintenanceMargin: result.maintenance_margin,
      marginRatio: result.margin_ratio,
      freeMargin: result.free_margin,
      marginLevel: result.margin_level
    };
  }

  /**
   * Check liquidation risk for a user
   */
  async checkLiquidationRisk(userId: string): Promise<LiquidationRisk> {
    const result = await this.jsonRpc.call('margin.liquidation_risk', {
      user_id: userId
    });

    return {
      userId: result.user_id,
      riskLevel: result.risk_level,
      marginLevel: result.margin_level,
      liquidationPrice: result.liquidation_price,
      timeToLiquidation: result.time_to_liquidation,
      recommendations: result.recommendations || []
    };
  }

  /**
   * Get insurance fund status
   */
  async getInsuranceFundStatus(): Promise<InsuranceFundStatus> {
    const result = await this.jsonRpc.call('insurance_fund.status');

    return {
      totalFund: result.total_fund,
      availableFund: result.available_fund,
      usedFund: result.used_fund,
      pendingClaims: result.pending_claims,
      lastUpdate: new Date(result.last_update)
    };
  }

  /**
   * Get list of available market data sources
   */
  async getMarketDataSources(): Promise<string[]> {
    return await this.jsonRpc.call('market_data.sources');
  }

  /**
   * Get comprehensive market statistics
   */
  async getMarketStats(symbol: string): Promise<MarketStats> {
    const result = await this.jsonRpc.call('market.stats', {
      symbol
    });

    return {
      symbol: result.symbol,
      volume24h: result.volume_24h,
      high24h: result.high_24h,
      low24h: result.low_24h,
      priceChange24h: result.price_change_24h,
      priceChangePercent24h: result.price_change_percent_24h,
      openInterest: result.open_interest,
      fundingRate: result.funding_rate,
      nextFundingTime: new Date(result.next_funding_time)
    };
  }
}

export class LiquidationMonitor {
  private ws: WebSocket | null;
  private callbacks: Map<string, Function[]> = new Map();

  constructor(wsConnection: WebSocket | null) {
    this.ws = wsConnection;
  }

  /**
   * Set WebSocket connection
   */
  setWebSocket(ws: WebSocket): void {
    this.ws = ws;
  }

  /**
   * Subscribe to liquidation events
   */
  subscribeLiquidations(callback: (liquidation: LiquidationInfo) => void): void {
    if (!this.callbacks.has('liquidations')) {
      this.callbacks.set('liquidations', []);
    }
    this.callbacks.get('liquidations')!.push(callback);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'liquidations'
      }));
    }
  }

  /**
   * Subscribe to settlement events
   */
  subscribeSettlements(callback: (settlement: SettlementBatch) => void): void {
    if (!this.callbacks.has('settlements')) {
      this.callbacks.set('settlements', []);
    }
    this.callbacks.get('settlements')!.push(callback);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'settlements'
      }));
    }
  }

  /**
   * Subscribe to margin call events for a user
   */
  subscribeMarginCalls(userId: string, callback: (marginCall: any) => void): void {
    const channel = `margin_calls:${userId}`;
    if (!this.callbacks.has(channel)) {
      this.callbacks.set(channel, []);
    }
    this.callbacks.get(channel)!.push(callback);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channel
      }));
    }
  }

  /**
   * Unsubscribe from a channel
   */
  unsubscribe(channel: string): void {
    this.callbacks.delete(channel);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        channel
      }));
    }
  }

  /**
   * Handle incoming message
   */
  handleMessage(channel: string, data: any): void {
    const callbacks = this.callbacks.get(channel);
    if (callbacks) {
      callbacks.forEach(cb => cb(data));
    }
  }
}

/**
 * Market data sources supported by LX DEX
 */
export const MarketDataProviders = {
  ALPACA: 'alpaca',
  NYSE_ARCA: 'nyse_arca',
  IEX_CLOUD: 'iex',
  POLYGON: 'polygon',
  CME_GROUP: 'cme',
  REFINITIV: 'refinitiv',
  ICE_DATA: 'ice',
  BLOOMBERG: 'bloomberg',
  NASDAQ_TOTALVIEW: 'nasdaq',
  COINBASE_PRO: 'coinbase'
} as const;

export type MarketDataProvider = typeof MarketDataProviders[keyof typeof MarketDataProviders];