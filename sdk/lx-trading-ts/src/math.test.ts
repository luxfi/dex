/**
 * Tests for LX Trading SDK math module
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  blackScholes,
  impliedVolatility,
  greeks,
  constantProductPrice,
  concentratedLiquidityPrice,
  calculateLiquidity,
  volatility,
  sharpeRatio,
  sortinoRatio,
  maxDrawdown,
  valueAtRisk,
  conditionalVaR,
  normCdf,
  normPdf,
  priceToSqrtPrice,
  sqrtPriceToPrice,
  tickToSqrtPrice,
  sqrtPriceToTick,
  calculateReturns,
  calculateLogReturns,
} from './math.js';

// Helper for floating point comparison
function approxEqual(actual: number, expected: number, epsilon = 0.01): boolean {
  return Math.abs(actual - expected) < epsilon;
}

describe('Black-Scholes Options Pricing', () => {
  it('should price ATM call option correctly', () => {
    // At-the-money call: S=100, K=100, T=1yr, r=5%, vol=20%
    const price = blackScholes(100, 100, 1, 0.05, 0.2, 'call');
    // Expected ~10.45 based on standard B-S
    assert.ok(approxEqual(price, 10.45, 0.5));
  });

  it('should price ATM put option correctly', () => {
    const price = blackScholes(100, 100, 1, 0.05, 0.2, 'put');
    // Put-call parity: Put = Call - S + K*exp(-rT)
    // Put ≈ 10.45 - 100 + 100*0.951 = 5.58
    assert.ok(approxEqual(price, 5.57, 0.5));
  });

  it('should return intrinsic value at expiry', () => {
    // At expiry (T=0), call = max(S-K, 0)
    const callITM = blackScholes(110, 100, 0, 0.05, 0.2, 'call');
    assert.strictEqual(callITM, 10);

    const callOTM = blackScholes(90, 100, 0, 0.05, 0.2, 'call');
    assert.strictEqual(callOTM, 0);

    const putITM = blackScholes(90, 100, 0, 0.05, 0.2, 'put');
    assert.strictEqual(putITM, 10);
  });

  it('should price deep ITM call close to intrinsic', () => {
    // Deep in-the-money call
    const price = blackScholes(150, 100, 0.1, 0.05, 0.2, 'call');
    assert.ok(price > 49); // At least intrinsic value
    assert.ok(price < 52); // Not too much time value left
  });

  it('should price deep OTM call close to zero', () => {
    const price = blackScholes(50, 100, 0.1, 0.05, 0.2, 'call');
    assert.ok(price < 0.01);
  });
});

describe('Implied Volatility', () => {
  it('should recover volatility from option price', () => {
    const trueVol = 0.25;
    const price = blackScholes(100, 100, 1, 0.05, trueVol, 'call');
    const iv = impliedVolatility(price, 100, 100, 1, 0.05, 'call');
    assert.ok(approxEqual(iv, trueVol, 0.01));
  });

  it('should work for puts', () => {
    const trueVol = 0.30;
    const price = blackScholes(100, 105, 0.5, 0.03, trueVol, 'put');
    const iv = impliedVolatility(price, 100, 105, 0.5, 0.03, 'put');
    assert.ok(approxEqual(iv, trueVol, 0.01));
  });
});

describe('Greeks', () => {
  it('should calculate delta for ATM call', () => {
    const g = greeks(100, 100, 1, 0.05, 0.2, 'call');
    // ATM call delta should be around 0.5-0.6
    assert.ok(g.delta > 0.5 && g.delta < 0.7);
  });

  it('should calculate delta for ATM put', () => {
    const g = greeks(100, 100, 1, 0.05, 0.2, 'put');
    // Put delta is negative
    assert.ok(g.delta > -0.6 && g.delta < -0.3);
  });

  it('should have positive gamma', () => {
    const g = greeks(100, 100, 1, 0.05, 0.2, 'call');
    assert.ok(g.gamma > 0);
  });

  it('should have negative theta for long call', () => {
    const g = greeks(100, 100, 1, 0.05, 0.2, 'call');
    assert.ok(g.theta < 0); // Time decay hurts long positions
  });

  it('should have positive vega', () => {
    const g = greeks(100, 100, 1, 0.05, 0.2, 'call');
    assert.ok(g.vega > 0); // Higher vol helps long options
  });

  it('should return zeros at expiry', () => {
    const g = greeks(100, 100, 0, 0.05, 0.2, 'call');
    assert.strictEqual(g.delta, 0);
    assert.strictEqual(g.gamma, 0);
    assert.strictEqual(g.vega, 0);
  });
});

describe('Constant Product AMM', () => {
  it('should calculate output for balanced pool', () => {
    // Equal reserves, swap 10 X for Y with 0.3% fee
    const result = constantProductPrice(1000, 1000, 10, 0.003, true);
    // Without fee: dy = 1000 * 10 / (1000 + 10) ≈ 9.9
    // With fee: dy = 1000 * 9.97 / (1000 + 9.97) ≈ 9.87
    assert.ok(approxEqual(result.outputAmount, 9.87, 0.1));
  });

  it('should calculate effective price', () => {
    const result = constantProductPrice(1000, 2000, 10, 0.003, true);
    // Price = output/input
    assert.ok(result.effectivePrice > 0);
    assert.ok(result.effectivePrice < 2); // Less than spot rate due to slippage
  });

  it('should work in both directions', () => {
    const xToY = constantProductPrice(1000, 1000, 10, 0.003, true);
    const yToX = constantProductPrice(1000, 1000, 10, 0.003, false);
    // Symmetric pool, same input, same output
    assert.ok(approxEqual(xToY.outputAmount, yToX.outputAmount, 0.1));
  });

  it('should handle zero input', () => {
    const result = constantProductPrice(1000, 1000, 0, 0.003, true);
    assert.strictEqual(result.outputAmount, 0);
    assert.strictEqual(result.effectivePrice, 0);
  });

  it('should account for fees', () => {
    const noFee = constantProductPrice(1000, 1000, 100, 0, true);
    const withFee = constantProductPrice(1000, 1000, 100, 0.01, true);
    assert.ok(withFee.outputAmount < noFee.outputAmount);
  });
});

describe('Concentrated Liquidity', () => {
  it('should calculate output within range', () => {
    // L=1000, sqrt price = 10 (price=100), range [9, 11]
    const result = concentratedLiquidityPrice(1000, 10, 9, 11, 10, 0.003, true);
    assert.ok(result.outputAmount > 0);
    assert.ok(result.newSqrtPrice > 10); // Price goes up when buying Y
    assert.ok(result.newSqrtPrice <= 11); // Capped at upper bound
  });

  it('should calculate price impact', () => {
    const result = concentratedLiquidityPrice(1000, 10, 9, 11, 100, 0.003, true);
    assert.ok(result.priceImpact > 0);
    assert.ok(result.priceImpact < 1); // Less than 100%
  });

  it('should handle swapping Y for X', () => {
    const result = concentratedLiquidityPrice(1000, 10, 9, 11, 10, 0.003, false);
    assert.ok(result.outputAmount > 0);
    assert.ok(result.newSqrtPrice < 10); // Price goes down when selling Y
    assert.ok(result.newSqrtPrice >= 9); // Capped at lower bound
  });
});

describe('Calculate Liquidity', () => {
  it('should calculate liquidity for in-range position', () => {
    const L = calculateLiquidity(100, 1000, 10, 9, 11);
    assert.ok(L > 0);
  });

  it('should calculate liquidity below range (only X)', () => {
    const L = calculateLiquidity(100, 0, 8, 9, 11);
    assert.ok(L > 0);
  });

  it('should calculate liquidity above range (only Y)', () => {
    const L = calculateLiquidity(0, 100, 12, 9, 11);
    assert.ok(L > 0);
  });
});

describe('Volatility', () => {
  it('should calculate volatility of returns', () => {
    const returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005];
    const vol = volatility(returns, false);
    assert.ok(vol > 0);
    assert.ok(vol < 1); // Should be reasonable for small returns
  });

  it('should annualize volatility', () => {
    const returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005];
    const dailyVol = volatility(returns, false);
    const annualVol = volatility(returns, true, 252);
    // Annual vol ≈ daily vol * sqrt(252)
    assert.ok(approxEqual(annualVol, dailyVol * Math.sqrt(252), 0.01));
  });

  it('should return 0 for insufficient data', () => {
    assert.strictEqual(volatility([0.01]), 0);
    assert.strictEqual(volatility([]), 0);
  });
});

describe('Sharpe Ratio', () => {
  it('should calculate positive Sharpe for good returns', () => {
    // Positive average returns with moderate vol
    const returns = [0.01, 0.02, 0.015, 0.005, 0.02, 0.01, 0.015];
    const sharpe = sharpeRatio(returns, 0, 252);
    assert.ok(sharpe > 0);
  });

  it('should calculate negative Sharpe for poor returns', () => {
    const returns = [-0.01, -0.02, -0.015, -0.005, -0.02, -0.01, -0.015];
    const sharpe = sharpeRatio(returns, 0, 252);
    assert.ok(sharpe < 0);
  });

  it('should return 0 for insufficient data', () => {
    assert.strictEqual(sharpeRatio([0.01]), 0);
  });

  it('should account for risk-free rate', () => {
    const returns = [0.0001, 0.0002, 0.0001, 0.0002, 0.0001]; // ~2.5% annual
    const sharpeNoRf = sharpeRatio(returns, 0, 252);
    const sharpeWithRf = sharpeRatio(returns, 0.05, 252); // 5% risk-free
    assert.ok(sharpeWithRf < sharpeNoRf);
  });
});

describe('Sortino Ratio', () => {
  it('should calculate Sortino ratio', () => {
    const returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.015];
    const sortino = sortinoRatio(returns, 0, 0, 252);
    // Sortino should be defined for this data
    assert.ok(!isNaN(sortino));
  });

  it('should be higher than Sharpe when downside limited', () => {
    // Returns with small downside
    const returns = [0.02, 0.01, 0.015, -0.002, 0.02, 0.01, 0.015];
    const sharpe = sharpeRatio(returns, 0, 252);
    const sortino = sortinoRatio(returns, 0, 0, 252);
    // With limited downside, Sortino typically > Sharpe
    assert.ok(sortino >= sharpe);
  });
});

describe('Maximum Drawdown', () => {
  it('should find maximum drawdown', () => {
    const prices = [100, 110, 105, 120, 90, 95, 100];
    const result = maxDrawdown(prices);
    // Peak at 120, trough at 90: drawdown = (120-90)/120 = 0.25
    assert.ok(approxEqual(result.maxDrawdown, 0.25, 0.01));
  });

  it('should track peak and trough indices', () => {
    const prices = [100, 110, 105, 120, 90, 95, 100];
    const result = maxDrawdown(prices);
    assert.strictEqual(result.peakIndex, 3); // Index of 120
    assert.strictEqual(result.troughIndex, 4); // Index of 90
  });

  it('should return 0 for monotonic increase', () => {
    const prices = [100, 110, 120, 130, 140];
    const result = maxDrawdown(prices);
    assert.strictEqual(result.maxDrawdown, 0);
  });

  it('should handle insufficient data', () => {
    const result = maxDrawdown([100]);
    assert.strictEqual(result.maxDrawdown, 0);
  });
});

describe('Value at Risk', () => {
  it('should calculate historical VaR', () => {
    // Generate some returns with known distribution
    const returns = Array.from({ length: 100 }, (_, i) => (i - 50) / 1000);
    const var95 = valueAtRisk(returns, 0.95, 'historical');
    assert.ok(var95 > 0);
  });

  it('should calculate parametric VaR', () => {
    const returns = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, -0.015, 0.015, -0.025];
    const var95 = valueAtRisk(returns, 0.95, 'parametric');
    assert.ok(var95 > 0);
  });

  it('should return 0 for insufficient data', () => {
    const var95 = valueAtRisk([0.01, 0.02], 0.95);
    assert.strictEqual(var95, 0);
  });

  it('should be higher for 99% confidence than 95%', () => {
    const returns = Array.from({ length: 100 }, () => (Math.random() - 0.5) * 0.1);
    const var95 = valueAtRisk(returns, 0.95);
    const var99 = valueAtRisk(returns, 0.99);
    assert.ok(var99 >= var95);
  });
});

describe('Conditional VaR (CVaR)', () => {
  it('should be greater than or equal to VaR', () => {
    const returns = Array.from({ length: 100 }, () => (Math.random() - 0.5) * 0.1);
    const var95 = valueAtRisk(returns, 0.95);
    const cvar95 = conditionalVaR(returns, 0.95);
    assert.ok(cvar95 >= var95 * 0.99); // Allow small numerical tolerance
  });

  it('should return 0 for insufficient data', () => {
    assert.strictEqual(conditionalVaR([0.01, 0.02], 0.95), 0);
  });
});

describe('Normal Distribution Helpers', () => {
  it('should calculate CDF correctly', () => {
    assert.ok(approxEqual(normCdf(0), 0.5, 0.001));
    assert.ok(approxEqual(normCdf(-1.96), 0.025, 0.01));
    assert.ok(approxEqual(normCdf(1.96), 0.975, 0.01));
  });

  it('should calculate PDF correctly', () => {
    // PDF at 0 = 1/sqrt(2*pi) ≈ 0.3989
    assert.ok(approxEqual(normPdf(0), 0.3989, 0.001));
    // PDF is symmetric
    assert.ok(approxEqual(normPdf(-1), normPdf(1), 0.0001));
  });
});

describe('Price/Tick Conversions', () => {
  it('should convert price to sqrt price', () => {
    assert.strictEqual(priceToSqrtPrice(100), 10);
    assert.strictEqual(priceToSqrtPrice(1), 1);
  });

  it('should convert sqrt price to price', () => {
    assert.strictEqual(sqrtPriceToPrice(10), 100);
    assert.strictEqual(sqrtPriceToPrice(1), 1);
  });

  it('should be inverse operations', () => {
    const price = 150;
    const recovered = sqrtPriceToPrice(priceToSqrtPrice(price));
    assert.ok(approxEqual(recovered, price, 0.0001));
  });

  it('should convert tick to sqrt price', () => {
    // tick = 0 => sqrt(1.0001^0) = 1
    assert.ok(approxEqual(tickToSqrtPrice(0), 1, 0.0001));
  });

  it('should convert sqrt price to tick', () => {
    const sqrtP = tickToSqrtPrice(1000);
    const tick = sqrtPriceToTick(sqrtP, 1);
    assert.ok(approxEqual(tick, 1000, 1));
  });
});

describe('Returns Calculation', () => {
  it('should calculate simple returns', () => {
    const prices = [100, 110, 105, 115];
    const returns = calculateReturns(prices);
    assert.strictEqual(returns.length, 3);
    assert.ok(approxEqual(returns[0]!, 0.1, 0.0001)); // (110-100)/100
    assert.ok(approxEqual(returns[1]!, -0.0455, 0.001)); // (105-110)/110
    assert.ok(approxEqual(returns[2]!, 0.0952, 0.001)); // (115-105)/105
  });

  it('should calculate log returns', () => {
    const prices = [100, 110, 105, 115];
    const returns = calculateLogReturns(prices);
    assert.strictEqual(returns.length, 3);
    assert.ok(approxEqual(returns[0]!, Math.log(110 / 100), 0.0001));
  });

  it('should handle empty array', () => {
    assert.strictEqual(calculateReturns([]).length, 0);
    assert.strictEqual(calculateLogReturns([]).length, 0);
  });

  it('should handle single price', () => {
    assert.strictEqual(calculateReturns([100]).length, 0);
    assert.strictEqual(calculateLogReturns([100]).length, 0);
  });
});
