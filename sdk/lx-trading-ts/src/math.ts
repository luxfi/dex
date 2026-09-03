/**
 * Financial mathematics for accurate market pricing.
 *
 * Includes:
 * - Options pricing (Black-Scholes, Greeks)
 * - AMM pricing (Constant Product, Concentrated Liquidity)
 * - Risk metrics (VaR, CVaR, Sharpe, Sortino)
 * - Statistical measures (volatility, drawdown)
 */

// =============================================================================
// Options Pricing
// =============================================================================

/**
 * Black-Scholes option pricing.
 *
 * @param S - Current spot price
 * @param K - Strike price
 * @param T - Time to expiration in years
 * @param r - Risk-free interest rate (annualized)
 * @param sigma - Volatility (annualized)
 * @param optionType - "call" or "put"
 * @returns Option price
 *
 * @example
 * ```typescript
 * const price = blackScholes(100, 100, 1, 0.05, 0.2, 'call');
 * console.log(`Call price: $${price.toFixed(2)}`);
 * ```
 */
export function blackScholes(
  S: number,
  K: number,
  T: number,
  r: number,
  sigma: number,
  optionType: 'call' | 'put' = 'call',
): number {
  if (T <= 0) {
    // At expiry
    if (optionType === 'call') {
      return Math.max(S - K, 0);
    }
    return Math.max(K - S, 0);
  }

  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);

  if (optionType === 'call') {
    return S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
  }
  return K * Math.exp(-r * T) * normCdf(-d2) - S * normCdf(-d1);
}

/**
 * Calculate implied volatility from option price using Newton-Raphson.
 *
 * @param price - Observed option price
 * @param S - Spot price
 * @param K - Strike price
 * @param T - Time to expiry
 * @param r - Risk-free rate
 * @param optionType - "call" or "put"
 * @param tol - Tolerance for convergence
 * @param maxIter - Maximum iterations
 * @returns Implied volatility
 */
export function impliedVolatility(
  price: number,
  S: number,
  K: number,
  T: number,
  r: number,
  optionType: 'call' | 'put' = 'call',
  tol = 1e-6,
  maxIter = 100,
): number {
  let sigma = 0.2; // Initial guess

  for (let i = 0; i < maxIter; i++) {
    const bsPrice = blackScholes(S, K, T, r, sigma, optionType);
    const vega = calcVega(S, K, T, r, sigma);

    if (Math.abs(vega) < 1e-10) {
      break;
    }

    const diff = bsPrice - price;
    if (Math.abs(diff) < tol) {
      return sigma;
    }

    sigma -= diff / vega;
    sigma = Math.max(0.001, Math.min(sigma, 5.0));
  }

  return sigma;
}

/**
 * Option Greeks.
 */
export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

/**
 * Calculate option Greeks.
 *
 * @param S - Spot price
 * @param K - Strike price
 * @param T - Time to expiry
 * @param r - Risk-free rate
 * @param sigma - Volatility
 * @param optionType - "call" or "put"
 * @returns Object with delta, gamma, theta, vega, rho
 */
export function greeks(
  S: number,
  K: number,
  T: number,
  r: number,
  sigma: number,
  optionType: 'call' | 'put' = 'call',
): Greeks {
  if (T <= 0) {
    return { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
  }

  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
  const d2 = d1 - sigma * sqrtT;

  const pdfD1 = normPdf(d1);
  const cdfD1 = normCdf(d1);
  const cdfD2 = normCdf(d2);
  const cdfNegD2 = normCdf(-d2);

  let delta: number;
  let theta: number;
  let rho: number;

  if (optionType === 'call') {
    delta = cdfD1;
    theta = -S * pdfD1 * sigma / (2 * sqrtT) - r * K * Math.exp(-r * T) * cdfD2;
    rho = K * T * Math.exp(-r * T) * cdfD2;
  } else {
    delta = cdfD1 - 1;
    theta = -S * pdfD1 * sigma / (2 * sqrtT) + r * K * Math.exp(-r * T) * cdfNegD2;
    rho = -K * T * Math.exp(-r * T) * cdfNegD2;
  }

  const gamma = pdfD1 / (S * sigma * sqrtT);
  const vega = S * pdfD1 * sqrtT / 100; // Per 1% change in vol
  theta = theta / 365; // Daily theta

  return { delta, gamma, theta, vega, rho };
}

function calcVega(S: number, K: number, T: number, r: number, sigma: number): number {
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
  return S * normPdf(d1) * sqrtT;
}

// =============================================================================
// AMM Pricing
// =============================================================================

/**
 * Result of constant product AMM calculation.
 */
export interface AmmPriceResult {
  outputAmount: number;
  effectivePrice: number;
}

/**
 * Calculate output amount for Uniswap V2 style constant product AMM.
 *
 * Formula: x * y = k (invariant)
 *
 * @param reserveX - Reserve of token X
 * @param reserveY - Reserve of token Y
 * @param amountIn - Amount of input token
 * @param feeRate - Trading fee (e.g., 0.003 = 0.3%)
 * @param isXtoY - True if swapping X for Y
 * @returns Object with outputAmount and effectivePrice
 *
 * @example
 * ```typescript
 * const result = constantProductPrice(1000, 1000, 10, 0.003);
 * console.log(`Output: ${result.outputAmount.toFixed(4)}`);
 * ```
 */
export function constantProductPrice(
  reserveX: number,
  reserveY: number,
  amountIn: number,
  feeRate = 0.003,
  isXtoY = true,
): AmmPriceResult {
  const amountInWithFee = amountIn * (1 - feeRate);

  const amountOut = isXtoY
    ? (reserveY * amountInWithFee) / (reserveX + amountInWithFee)
    : (reserveX * amountInWithFee) / (reserveY + amountInWithFee);

  const effectivePrice = amountIn > 0 ? amountOut / amountIn : 0;

  return { outputAmount: amountOut, effectivePrice };
}

/**
 * Result of concentrated liquidity calculation.
 */
export interface ConcentratedLiquidityResult {
  outputAmount: number;
  newSqrtPrice: number;
  priceImpact: number;
}

/**
 * Calculate output for Uniswap V3 style concentrated liquidity.
 *
 * @param liquidity - L value (sqrt(x * y))
 * @param sqrtPriceCurrent - Current sqrt(P) = sqrt(y/x)
 * @param sqrtPriceLower - Lower tick sqrt price
 * @param sqrtPriceUpper - Upper tick sqrt price
 * @param amountIn - Input amount
 * @param feeRate - Trading fee
 * @param isToken0In - True if swapping token0 for token1
 * @returns Object with outputAmount, newSqrtPrice, priceImpact
 */
export function concentratedLiquidityPrice(
  liquidity: number,
  sqrtPriceCurrent: number,
  sqrtPriceLower: number,
  sqrtPriceUpper: number,
  amountIn: number,
  feeRate = 0.003,
  isToken0In = true,
): ConcentratedLiquidityResult {
  const amountInWithFee = amountIn * (1 - feeRate);
  let newSqrtP: number;
  let amountOut: number;

  if (isToken0In) {
    // Swapping X for Y (price goes up)
    const deltaInvSqrtP = amountInWithFee / liquidity;
    const newInvSqrtP = 1 / sqrtPriceCurrent - deltaInvSqrtP;

    if (newInvSqrtP <= 0) {
      newSqrtP = sqrtPriceUpper;
    } else {
      newSqrtP = 1 / newInvSqrtP;
    }

    newSqrtP = Math.min(newSqrtP, sqrtPriceUpper);
    amountOut = liquidity * (newSqrtP - sqrtPriceCurrent);
  } else {
    // Swapping Y for X (price goes down)
    const deltaSqrtP = amountInWithFee / liquidity;
    newSqrtP = sqrtPriceCurrent - deltaSqrtP;

    newSqrtP = Math.max(newSqrtP, sqrtPriceLower);
    amountOut = liquidity * (1 / newSqrtP - 1 / sqrtPriceCurrent);
  }

  // Price impact
  const oldPrice = sqrtPriceCurrent * sqrtPriceCurrent;
  const newPrice = newSqrtP * newSqrtP;
  const priceImpact = Math.abs(newPrice - oldPrice) / oldPrice;

  return {
    outputAmount: Math.max(amountOut, 0),
    newSqrtPrice: newSqrtP,
    priceImpact,
  };
}

/**
 * Calculate liquidity (L) for a concentrated liquidity position.
 *
 * @param amountX - Amount of token X to provide
 * @param amountY - Amount of token Y to provide
 * @param sqrtPriceCurrent - Current sqrt price
 * @param sqrtPriceLower - Lower tick sqrt price
 * @param sqrtPriceUpper - Upper tick sqrt price
 * @returns Liquidity value L
 */
export function calculateLiquidity(
  amountX: number,
  amountY: number,
  sqrtPriceCurrent: number,
  sqrtPriceLower: number,
  sqrtPriceUpper: number,
): number {
  if (sqrtPriceCurrent <= sqrtPriceLower) {
    // Only token X
    return (amountX * sqrtPriceLower * sqrtPriceUpper) / (sqrtPriceUpper - sqrtPriceLower);
  } else if (sqrtPriceCurrent >= sqrtPriceUpper) {
    // Only token Y
    return amountY / (sqrtPriceUpper - sqrtPriceLower);
  } else {
    // Both tokens
    const lX =
      (amountX * sqrtPriceCurrent * sqrtPriceUpper) / (sqrtPriceUpper - sqrtPriceCurrent);
    const lY = amountY / (sqrtPriceCurrent - sqrtPriceLower);
    return Math.min(lX, lY);
  }
}

// =============================================================================
// Risk Metrics
// =============================================================================

/**
 * Calculate historical volatility.
 *
 * @param returns - Array of period returns
 * @param annualize - Whether to annualize the result
 * @param periodsPerYear - Trading periods per year (252 for daily)
 * @returns Volatility (standard deviation of returns)
 */
export function volatility(
  returns: number[],
  annualize = true,
  periodsPerYear = 252,
): number {
  if (returns.length < 2) {
    return 0;
  }

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
  let std = Math.sqrt(variance);

  if (annualize) {
    std *= Math.sqrt(periodsPerYear);
  }

  return std;
}

/**
 * Calculate Sharpe ratio.
 *
 * @param returns - Array of period returns
 * @param riskFreeRate - Annual risk-free rate
 * @param periodsPerYear - Trading periods per year
 * @returns Sharpe ratio
 */
export function sharpeRatio(
  returns: number[],
  riskFreeRate = 0,
  periodsPerYear = 252,
): number {
  if (returns.length < 2) {
    return 0;
  }

  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + (r - meanReturn) ** 2, 0) / (returns.length - 1);
  const stdReturn = Math.sqrt(variance);

  if (stdReturn === 0) {
    return 0;
  }

  const periodRf = riskFreeRate / periodsPerYear;
  const excessReturn = meanReturn - periodRf;

  return (excessReturn * periodsPerYear) / (stdReturn * Math.sqrt(periodsPerYear));
}

/**
 * Calculate Sortino ratio (uses downside deviation).
 *
 * @param returns - Array of period returns
 * @param riskFreeRate - Annual risk-free rate
 * @param targetReturn - Minimum acceptable return
 * @param periodsPerYear - Trading periods per year
 * @returns Sortino ratio
 */
export function sortinoRatio(
  returns: number[],
  riskFreeRate = 0,
  targetReturn = 0,
  periodsPerYear = 252,
): number {
  if (returns.length < 2) {
    return 0;
  }

  // Calculate downside deviation
  const downsideReturns = returns.map((r) => Math.min(r - targetReturn, 0) ** 2);
  const downsideStd = Math.sqrt(downsideReturns.reduce((a, b) => a + b, 0) / downsideReturns.length);
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;

  if (downsideStd === 0) {
    return meanReturn > riskFreeRate / periodsPerYear ? Infinity : 0;
  }

  const periodRf = riskFreeRate / periodsPerYear;
  const excessReturn = meanReturn - periodRf;

  return (excessReturn * periodsPerYear) / (downsideStd * Math.sqrt(periodsPerYear));
}

/**
 * Maximum drawdown result.
 */
export interface MaxDrawdownResult {
  maxDrawdown: number;
  peakIndex: number;
  troughIndex: number;
}

/**
 * Calculate maximum drawdown.
 *
 * @param prices - Array of prices/equity values
 * @returns Object with maxDrawdown, peakIndex, troughIndex
 */
export function maxDrawdown(prices: number[]): MaxDrawdownResult {
  if (prices.length < 2) {
    return { maxDrawdown: 0, peakIndex: 0, troughIndex: 0 };
  }

  let peak = prices[0] ?? 0;
  let peakIdx = 0;
  let maxDd = 0;
  let maxDdPeak = 0;
  let maxDdTrough = 0;

  for (let i = 0; i < prices.length; i++) {
    const price = prices[i] ?? 0;
    if (price > peak) {
      peak = price;
      peakIdx = i;
    }

    const dd = peak > 0 ? (peak - price) / peak : 0;

    if (dd > maxDd) {
      maxDd = dd;
      maxDdPeak = peakIdx;
      maxDdTrough = i;
    }
  }

  return {
    maxDrawdown: maxDd,
    peakIndex: maxDdPeak,
    troughIndex: maxDdTrough,
  };
}

/**
 * Calculate Value at Risk.
 *
 * @param returns - Array of returns
 * @param confidence - Confidence level (e.g., 0.95 for 95%)
 * @param method - "historical" or "parametric"
 * @returns VaR as a positive number (potential loss)
 */
export function valueAtRisk(
  returns: number[],
  confidence = 0.95,
  method: 'historical' | 'parametric' = 'historical',
): number {
  if (returns.length < 10) {
    return 0;
  }

  if (method === 'historical') {
    const sorted = [...returns].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * (1 - confidence));
    return -(sorted[idx] ?? 0);
  }

  // Parametric (assumes normal distribution)
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
  const std = Math.sqrt(variance);

  // Approximate z-scores
  const z = confidence === 0.95 ? -1.645 : confidence === 0.99 ? -2.326 : -1.645;

  return -(mean + z * std);
}

/**
 * Calculate Conditional Value at Risk (Expected Shortfall).
 *
 * @param returns - Array of returns
 * @param confidence - Confidence level
 * @returns CVaR as a positive number (expected loss beyond VaR)
 */
export function conditionalVaR(returns: number[], confidence = 0.95): number {
  if (returns.length < 10) {
    return 0;
  }

  const varValue = valueAtRisk(returns, confidence, 'historical');

  // Average of returns worse than VaR
  const tailReturns = returns.filter((r) => r <= -varValue);

  if (tailReturns.length === 0) {
    return varValue;
  }

  return -tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Standard normal CDF.
 */
export function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/**
 * Standard normal PDF.
 */
export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Error function approximation.
 */
function erf(x: number): number {
  // Approximation using Horner's method
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return sign * y;
}

/**
 * Convert price to sqrt price for concentrated liquidity.
 */
export function priceToSqrtPrice(price: number): number {
  return Math.sqrt(price);
}

/**
 * Convert sqrt price to regular price.
 */
export function sqrtPriceToPrice(sqrtPrice: number): number {
  return sqrtPrice * sqrtPrice;
}

/**
 * Convert tick to sqrt price (Uniswap V3 style).
 */
export function tickToSqrtPrice(tick: number): number {
  return Math.pow(1.0001, tick / 2);
}

/**
 * Convert sqrt price to nearest tick.
 */
export function sqrtPriceToTick(sqrtPrice: number, tickSpacing = 60): number {
  const tick = Math.round((2 * Math.log(sqrtPrice)) / Math.log(1.0001));
  return Math.floor(tick / tickSpacing) * tickSpacing;
}

/**
 * Calculate returns from price series.
 */
export function calculateReturns(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const prev = prices[i - 1];
    const curr = prices[i];
    if (prev && curr && prev !== 0) {
      returns.push((curr - prev) / prev);
    }
  }
  return returns;
}

/**
 * Calculate log returns from price series.
 */
export function calculateLogReturns(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const prev = prices[i - 1];
    const curr = prices[i];
    if (prev && curr && prev > 0 && curr > 0) {
      returns.push(Math.log(curr / prev));
    }
  }
  return returns;
}
