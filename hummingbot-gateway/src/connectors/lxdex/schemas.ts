/**
 * LX DEX Gateway Connector Schemas
 *
 * Request and response schemas for the LX DEX connector.
 * Implements Router, AMM, and CLMM schema types.
 */

// ============================================================================
// Common Types
// ============================================================================

export interface TokenInfo {
  symbol: string;
  address: string;
  decimals: number;
  name?: string;
}

export interface TradingPair {
  baseToken: string;
  quoteToken: string;
  symbol: string;
}

export type OrderSide = 'BUY' | 'SELL';
export type OrderType = 'LIMIT' | 'MARKET' | 'STOP' | 'STOP_LIMIT';
export type TimeInForce = 'GTC' | 'IOC' | 'FOK' | 'GTT';

// ============================================================================
// Router Schema - Swap/Quote Endpoints
// ============================================================================

export interface LXDexQuoteSwapRequest {
  network?: string;
  baseToken: string;
  quoteToken: string;
  amount: string;
  side: OrderSide;
  slippagePct?: number;
  maxHops?: number;
}

export interface SwapRoute {
  path: string[];
  pools: string[];
  expectedOutput: string;
  priceImpact: string;
  fee: string;
}

export interface LXDexQuoteSwapResponse {
  quoteId: string;
  tokenIn: TokenInfo;
  tokenOut: TokenInfo;
  amountIn: string;
  amountOut: string;
  price: string;
  priceImpactPct: string;
  minAmountOut: string;
  maxAmountIn: string;
  route: SwapRoute;
  estimatedGas: string;
  expiresAt: number;
}

export interface LXDexExecuteSwapRequest {
  network?: string;
  walletAddress: string;
  baseToken: string;
  quoteToken: string;
  amount: string;
  side: OrderSide;
  slippagePct?: number;
  maxHops?: number;
  gasPrice?: string;
}

export interface LXDexExecuteQuoteRequest {
  network?: string;
  walletAddress: string;
  quoteId: string;
  gasPrice?: string;
}

export interface LXDexSwapResponse {
  txHash: string;
  status: 'pending' | 'confirmed' | 'failed';
  tokenIn: TokenInfo;
  tokenOut: TokenInfo;
  amountIn: string;
  amountOut: string;
  price: string;
  fee: string;
  gasUsed?: string;
  blockNumber?: number;
}

// ============================================================================
// AMM Schema - Liquidity Pool Endpoints
// ============================================================================

export interface LXDexPoolInfoRequest {
  network?: string;
  tokenA: string;
  tokenB: string;
}

export interface PoolInfo {
  address: string;
  tokenA: TokenInfo;
  tokenB: TokenInfo;
  reserveA: string;
  reserveB: string;
  totalLiquidity: string;
  fee: string;
  apy?: string;
  volume24h?: string;
}

export interface LXDexPoolInfoResponse {
  pools: PoolInfo[];
}

export interface LXDexPositionInfoRequest {
  network?: string;
  walletAddress: string;
  poolAddress?: string;
}

export interface LPPosition {
  poolAddress: string;
  tokenA: TokenInfo;
  tokenB: TokenInfo;
  liquidity: string;
  sharePercent: string;
  valueUSD: string;
  unclaimedFees: {
    tokenA: string;
    tokenB: string;
  };
}

export interface LXDexPositionInfoResponse {
  positions: LPPosition[];
  totalValueUSD: string;
}

export interface LXDexAddLiquidityRequest {
  network?: string;
  walletAddress: string;
  tokenA: string;
  tokenB: string;
  amountA: string;
  amountB: string;
  slippagePct?: number;
}

export interface LXDexAddLiquidityResponse {
  txHash: string;
  status: 'pending' | 'confirmed' | 'failed';
  poolAddress: string;
  liquidityMinted: string;
  amountA: string;
  amountB: string;
}

export interface LXDexRemoveLiquidityRequest {
  network?: string;
  walletAddress: string;
  poolAddress: string;
  liquidity: string;
  slippagePct?: number;
}

export interface LXDexRemoveLiquidityResponse {
  txHash: string;
  status: 'pending' | 'confirmed' | 'failed';
  amountA: string;
  amountB: string;
  liquidityBurned: string;
}

// ============================================================================
// CLMM Schema - Concentrated Liquidity Endpoints
// ============================================================================

export interface LXDexCLMMPoolInfoRequest {
  network?: string;
  tokenA: string;
  tokenB: string;
  fee?: number; // Fee tier in basis points
}

export interface CLMMPoolInfo {
  address: string;
  tokenA: TokenInfo;
  tokenB: TokenInfo;
  fee: number;
  tickSpacing: number;
  currentTick: number;
  currentPrice: string;
  liquidity: string;
  volume24h: string;
  tvlUSD: string;
}

export interface LXDexCLMMPoolInfoResponse {
  pools: CLMMPoolInfo[];
}

export interface LXDexPositionsOwnedRequest {
  network?: string;
  walletAddress: string;
}

export interface CLMMPosition {
  tokenId: string;
  poolAddress: string;
  tokenA: TokenInfo;
  tokenB: TokenInfo;
  tickLower: number;
  tickUpper: number;
  liquidity: string;
  amountA: string;
  amountB: string;
  unclaimedFeesA: string;
  unclaimedFeesB: string;
  inRange: boolean;
  valueUSD: string;
}

export interface LXDexPositionsOwnedResponse {
  positions: CLMMPosition[];
  totalValueUSD: string;
}

export interface LXDexQuotePositionRequest {
  network?: string;
  tokenA: string;
  tokenB: string;
  fee: number;
  tickLower: number;
  tickUpper: number;
  amountA?: string;
  amountB?: string;
}

export interface LXDexQuotePositionResponse {
  estimatedAmountA: string;
  estimatedAmountB: string;
  estimatedLiquidity: string;
  priceRange: {
    lower: string;
    upper: string;
    current: string;
  };
  inRange: boolean;
}

export interface LXDexOpenPositionRequest {
  network?: string;
  walletAddress: string;
  tokenA: string;
  tokenB: string;
  fee: number;
  tickLower: number;
  tickUpper: number;
  amountA: string;
  amountB: string;
  slippagePct?: number;
}

export interface LXDexOpenPositionResponse {
  txHash: string;
  status: 'pending' | 'confirmed' | 'failed';
  tokenId: string;
  liquidity: string;
  amountA: string;
  amountB: string;
}

export interface LXDexClosePositionRequest {
  network?: string;
  walletAddress: string;
  tokenId: string;
  slippagePct?: number;
}

export interface LXDexClosePositionResponse {
  txHash: string;
  status: 'pending' | 'confirmed' | 'failed';
  amountA: string;
  amountB: string;
  feesCollectedA: string;
  feesCollectedB: string;
}

export interface LXDexCollectFeesRequest {
  network?: string;
  walletAddress: string;
  tokenId: string;
}

export interface LXDexCollectFeesResponse {
  txHash: string;
  status: 'pending' | 'confirmed' | 'failed';
  feesCollectedA: string;
  feesCollectedB: string;
}

// ============================================================================
// Order Book Schema - Central Limit Order Book
// ============================================================================

export interface LXDexPlaceOrderRequest {
  network?: string;
  walletAddress: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  price?: string;
  size: string;
  timeInForce?: TimeInForce;
  clientOrderId?: string;
}

export interface Order {
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  status: 'open' | 'partial' | 'filled' | 'cancelled' | 'rejected';
  price: string;
  size: string;
  filledSize: string;
  remainingSize: string;
  avgFillPrice?: string;
  fee: string;
  createdAt: number;
  updatedAt: number;
}

export interface LXDexPlaceOrderResponse {
  txHash?: string;
  order: Order;
}

export interface LXDexCancelOrderRequest {
  network?: string;
  walletAddress: string;
  orderId: string;
}

export interface LXDexCancelOrderResponse {
  txHash?: string;
  orderId: string;
  status: 'cancelled' | 'failed';
}

export interface LXDexGetOrdersRequest {
  network?: string;
  walletAddress: string;
  symbol?: string;
  status?: string[];
  limit?: number;
}

export interface LXDexGetOrdersResponse {
  orders: Order[];
  total: number;
}

export interface LXDexOrderBookRequest {
  network?: string;
  symbol: string;
  depth?: number;
}

export interface OrderBookLevel {
  price: string;
  size: string;
  orders: number;
}

export interface LXDexOrderBookResponse {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
  sequenceNumber: number;
}

// ============================================================================
// Validation Schemas (JSON Schema format for Fastify)
// ============================================================================

export const quoteSwapRequestSchema = {
  type: 'object',
  required: ['baseToken', 'quoteToken', 'amount', 'side'],
  properties: {
    network: { type: 'string' },
    baseToken: { type: 'string' },
    quoteToken: { type: 'string' },
    amount: { type: 'string' },
    side: { type: 'string', enum: ['BUY', 'SELL'] },
    slippagePct: { type: 'number', minimum: 0, maximum: 100 },
    maxHops: { type: 'number', minimum: 1, maximum: 10 },
  },
};

export const executeSwapRequestSchema = {
  type: 'object',
  required: ['walletAddress', 'baseToken', 'quoteToken', 'amount', 'side'],
  properties: {
    network: { type: 'string' },
    walletAddress: { type: 'string' },
    baseToken: { type: 'string' },
    quoteToken: { type: 'string' },
    amount: { type: 'string' },
    side: { type: 'string', enum: ['BUY', 'SELL'] },
    slippagePct: { type: 'number', minimum: 0, maximum: 100 },
    maxHops: { type: 'number', minimum: 1, maximum: 10 },
    gasPrice: { type: 'string' },
  },
};

export const placeOrderRequestSchema = {
  type: 'object',
  required: ['walletAddress', 'symbol', 'side', 'type', 'size'],
  properties: {
    network: { type: 'string' },
    walletAddress: { type: 'string' },
    symbol: { type: 'string' },
    side: { type: 'string', enum: ['BUY', 'SELL'] },
    type: { type: 'string', enum: ['LIMIT', 'MARKET', 'STOP', 'STOP_LIMIT'] },
    price: { type: 'string' },
    size: { type: 'string' },
    timeInForce: { type: 'string', enum: ['GTC', 'IOC', 'FOK', 'GTT'] },
    clientOrderId: { type: 'string' },
  },
};

export const orderBookRequestSchema = {
  type: 'object',
  required: ['symbol'],
  properties: {
    network: { type: 'string' },
    symbol: { type: 'string' },
    depth: { type: 'number', minimum: 1, maximum: 500 },
  },
};
