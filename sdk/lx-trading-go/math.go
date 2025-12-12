// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"math"
	"sort"

	"github.com/shopspring/decimal"
)

// =============================================================================
// Options Pricing - Black-Scholes
// =============================================================================

// BlackScholes calculates option price using Black-Scholes model.
//
// Parameters:
//   - S: Current spot price
//   - K: Strike price
//   - T: Time to expiration in years
//   - r: Risk-free interest rate (annualized)
//   - sigma: Volatility (annualized)
//   - isCall: true for call option, false for put
//
// Returns the option price.
func BlackScholes(S, K, T, r, sigma float64, isCall bool) float64 {
	if T <= 0 {
		if isCall {
			return math.Max(S-K, 0)
		}
		return math.Max(K-S, 0)
	}

	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * math.Sqrt(T))
	d2 := d1 - sigma*math.Sqrt(T)

	if isCall {
		return S*normCDF(d1) - K*math.Exp(-r*T)*normCDF(d2)
	}
	return K*math.Exp(-r*T)*normCDF(-d2) - S*normCDF(-d1)
}

// ImpliedVolatility calculates implied volatility from option price.
// Uses Newton-Raphson iteration.
func ImpliedVolatility(price, S, K, T, r float64, isCall bool) float64 {
	if T <= 0 {
		return 0
	}

	sigma := 0.2 // Initial guess
	const tol = 1e-6
	const maxIter = 100

	for i := 0; i < maxIter; i++ {
		bsPrice := BlackScholes(S, K, T, r, sigma, isCall)
		vega := calcVega(S, K, T, r, sigma)

		if math.Abs(vega) < 1e-10 {
			break
		}

		diff := bsPrice - price
		if math.Abs(diff) < tol {
			return sigma
		}

		sigma -= diff / vega
		sigma = math.Max(0.001, math.Min(sigma, 5.0))
	}

	return sigma
}

// Greeks holds option Greeks.
type Greeks struct {
	Delta float64
	Gamma float64
	Theta float64
	Vega  float64
	Rho   float64
}

// CalculateGreeks calculates option Greeks.
func CalculateGreeks(S, K, T, r, sigma float64, isCall bool) Greeks {
	if T <= 0 {
		return Greeks{}
	}

	sqrtT := math.Sqrt(T)
	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * sqrtT)
	d2 := d1 - sigma*sqrtT

	pdfD1 := normPDF(d1)
	cdfD1 := normCDF(d1)
	cdfD2 := normCDF(d2)
	cdfNegD2 := normCDF(-d2)

	var delta, theta, rho float64

	if isCall {
		delta = cdfD1
		theta = (-S*pdfD1*sigma/(2*sqrtT) - r*K*math.Exp(-r*T)*cdfD2)
		rho = K * T * math.Exp(-r*T) * cdfD2
	} else {
		delta = cdfD1 - 1
		theta = (-S*pdfD1*sigma/(2*sqrtT) + r*K*math.Exp(-r*T)*cdfNegD2)
		rho = -K * T * math.Exp(-r*T) * cdfNegD2
	}

	gamma := pdfD1 / (S * sigma * sqrtT)
	vega := S * pdfD1 * sqrtT / 100 // Per 1% change in vol

	return Greeks{
		Delta: delta,
		Gamma: gamma,
		Theta: theta / 365, // Daily theta
		Vega:  vega,
		Rho:   rho,
	}
}

func calcVega(S, K, T, r, sigma float64) float64 {
	sqrtT := math.Sqrt(T)
	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * sqrtT)
	return S * normPDF(d1) * sqrtT
}

// =============================================================================
// AMM Pricing
// =============================================================================

// ConstantProductPrice calculates output for constant product AMM (xy=k).
//
// Returns (outputAmount, effectivePrice).
func ConstantProductPrice(reserveX, reserveY, amountIn decimal.Decimal, feeRate decimal.Decimal, isXToY bool) (decimal.Decimal, decimal.Decimal) {
	amountInWithFee := amountIn.Mul(decimal.NewFromInt(1).Sub(feeRate))

	var amountOut decimal.Decimal
	if isXToY {
		// x -> y: dy = y * dx / (x + dx)
		amountOut = reserveY.Mul(amountInWithFee).Div(reserveX.Add(amountInWithFee))
	} else {
		// y -> x: dx = x * dy / (y + dy)
		amountOut = reserveX.Mul(amountInWithFee).Div(reserveY.Add(amountInWithFee))
	}

	effectivePrice := decimal.Zero
	if !amountIn.IsZero() {
		effectivePrice = amountOut.Div(amountIn)
	}

	return amountOut, effectivePrice
}

// ConcentratedLiquidityPrice calculates output for concentrated liquidity AMM.
//
// Returns (outputAmount, newSqrtPrice, priceImpact).
func ConcentratedLiquidityPrice(
	liquidity, sqrtPriceCurrent, sqrtPriceLower, sqrtPriceUpper, amountIn decimal.Decimal,
	feeRate decimal.Decimal,
	isToken0In bool,
) (decimal.Decimal, decimal.Decimal, decimal.Decimal) {
	amountInWithFee := amountIn.Mul(decimal.NewFromInt(1).Sub(feeRate))

	var newSqrtP, amountOut decimal.Decimal

	if isToken0In {
		// Swapping X for Y (price goes up)
		// dY = L * (sqrt_P_new - sqrt_P_current)
		// dX = L * (1/sqrt_P_current - 1/sqrt_P_new)
		deltaInvSqrtP := amountInWithFee.Div(liquidity)
		newInvSqrtP := decimal.NewFromInt(1).Div(sqrtPriceCurrent).Sub(deltaInvSqrtP)

		if newInvSqrtP.LessThanOrEqual(decimal.Zero) {
			newSqrtP = sqrtPriceUpper
		} else {
			newSqrtP = decimal.NewFromInt(1).Div(newInvSqrtP)
		}

		newSqrtP = decimal.Min(newSqrtP, sqrtPriceUpper)
		amountOut = liquidity.Mul(newSqrtP.Sub(sqrtPriceCurrent))
	} else {
		// Swapping Y for X (price goes down)
		deltaSqrtP := amountInWithFee.Div(liquidity)
		newSqrtP = sqrtPriceCurrent.Sub(deltaSqrtP)

		newSqrtP = decimal.Max(newSqrtP, sqrtPriceLower)
		amountOut = liquidity.Mul(decimal.NewFromInt(1).Div(newSqrtP).Sub(decimal.NewFromInt(1).Div(sqrtPriceCurrent)))
	}

	// Price impact
	oldPrice := sqrtPriceCurrent.Mul(sqrtPriceCurrent)
	newPrice := newSqrtP.Mul(newSqrtP)
	priceImpact := newPrice.Sub(oldPrice).Abs().Div(oldPrice)

	if amountOut.IsNegative() {
		amountOut = decimal.Zero
	}

	return amountOut, newSqrtP, priceImpact
}

// CalculateLiquidity calculates L for a concentrated liquidity position.
func CalculateLiquidity(amountX, amountY, sqrtPriceCurrent, sqrtPriceLower, sqrtPriceUpper decimal.Decimal) decimal.Decimal {
	if sqrtPriceCurrent.LessThanOrEqual(sqrtPriceLower) {
		// Only token X
		return amountX.Mul(sqrtPriceLower).Mul(sqrtPriceUpper).Div(sqrtPriceUpper.Sub(sqrtPriceLower))
	} else if sqrtPriceCurrent.GreaterThanOrEqual(sqrtPriceUpper) {
		// Only token Y
		return amountY.Div(sqrtPriceUpper.Sub(sqrtPriceLower))
	}

	// Both tokens
	lX := amountX.Mul(sqrtPriceCurrent).Mul(sqrtPriceUpper).Div(sqrtPriceUpper.Sub(sqrtPriceCurrent))
	lY := amountY.Div(sqrtPriceCurrent.Sub(sqrtPriceLower))
	return decimal.Min(lX, lY)
}

// =============================================================================
// Risk Metrics
// =============================================================================

// Volatility calculates historical volatility from returns.
func Volatility(returns []float64, annualize bool, periodsPerYear int) float64 {
	if len(returns) < 2 {
		return 0
	}

	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, r := range returns {
		variance += (r - mean) * (r - mean)
	}
	variance /= float64(len(returns) - 1)

	std := math.Sqrt(variance)

	if annualize {
		std *= math.Sqrt(float64(periodsPerYear))
	}

	return std
}

// SharpeRatio calculates the Sharpe ratio.
func SharpeRatio(returns []float64, riskFreeRate float64, periodsPerYear int) float64 {
	if len(returns) < 2 {
		return 0
	}

	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, r := range returns {
		variance += (r - mean) * (r - mean)
	}
	variance /= float64(len(returns) - 1)
	std := math.Sqrt(variance)

	if std == 0 {
		return 0
	}

	periodRf := riskFreeRate / float64(periodsPerYear)
	excessReturn := mean - periodRf

	return (excessReturn * float64(periodsPerYear)) / (std * math.Sqrt(float64(periodsPerYear)))
}

// SortinoRatio calculates the Sortino ratio (uses downside deviation).
func SortinoRatio(returns []float64, riskFreeRate, targetReturn float64, periodsPerYear int) float64 {
	if len(returns) < 2 {
		return 0
	}

	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	// Calculate downside deviation
	downsideSum := 0.0
	for _, r := range returns {
		diff := r - targetReturn
		if diff < 0 {
			downsideSum += diff * diff
		}
	}
	downsideStd := math.Sqrt(downsideSum / float64(len(returns)))

	if downsideStd == 0 {
		if mean > riskFreeRate/float64(periodsPerYear) {
			return math.Inf(1)
		}
		return 0
	}

	periodRf := riskFreeRate / float64(periodsPerYear)
	excessReturn := mean - periodRf

	return (excessReturn * float64(periodsPerYear)) / (downsideStd * math.Sqrt(float64(periodsPerYear)))
}

// MaxDrawdown calculates maximum drawdown.
// Returns (maxDrawdown, peakIndex, troughIndex).
func MaxDrawdown(prices []float64) (float64, int, int) {
	if len(prices) < 2 {
		return 0, 0, 0
	}

	peak := prices[0]
	peakIdx := 0
	maxDD := 0.0
	maxDDPeak := 0
	maxDDTrough := 0

	for i, price := range prices {
		if price > peak {
			peak = price
			peakIdx = i
		}

		dd := 0.0
		if peak > 0 {
			dd = (peak - price) / peak
		}

		if dd > maxDD {
			maxDD = dd
			maxDDPeak = peakIdx
			maxDDTrough = i
		}
	}

	return maxDD, maxDDPeak, maxDDTrough
}

// VaR calculates Value at Risk.
// Method can be "historical" or "parametric".
func VaR(returns []float64, confidence float64, method string) float64 {
	if len(returns) < 10 {
		return 0
	}

	if method == "historical" {
		sorted := make([]float64, len(returns))
		copy(sorted, returns)
		sort.Float64s(sorted)

		idx := int(float64(len(sorted)) * (1 - confidence))
		return -sorted[idx]
	}

	// Parametric (assumes normal distribution)
	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, r := range returns {
		variance += (r - mean) * (r - mean)
	}
	variance /= float64(len(returns) - 1)
	std := math.Sqrt(variance)

	// Z-score for confidence level
	var z float64
	switch {
	case confidence >= 0.99:
		z = -2.326
	case confidence >= 0.95:
		z = -1.645
	case confidence >= 0.90:
		z = -1.282
	default:
		z = -1.645
	}

	return -(mean + z*std)
}

// CVaR calculates Conditional Value at Risk (Expected Shortfall).
func CVaR(returns []float64, confidence float64) float64 {
	if len(returns) < 10 {
		return 0
	}

	varValue := VaR(returns, confidence, "historical")

	// Average of returns worse than VaR
	var tailSum float64
	var tailCount int
	for _, r := range returns {
		if r <= -varValue {
			tailSum += r
			tailCount++
		}
	}

	if tailCount == 0 {
		return varValue
	}

	return -tailSum / float64(tailCount)
}

// =============================================================================
// Helper Functions
// =============================================================================

// normCDF calculates standard normal CDF.
func normCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

// normPDF calculates standard normal PDF.
func normPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

// PriceToSqrtPrice converts price to sqrt price for concentrated liquidity.
func PriceToSqrtPrice(price float64) float64 {
	return math.Sqrt(price)
}

// SqrtPriceToPrice converts sqrt price to regular price.
func SqrtPriceToPrice(sqrtPrice float64) float64 {
	return sqrtPrice * sqrtPrice
}

// TickToSqrtPrice converts tick to sqrt price (Uniswap V3 style).
func TickToSqrtPrice(tick int) float64 {
	return math.Pow(1.0001, float64(tick)/2)
}

// SqrtPriceToTick converts sqrt price to nearest tick.
func SqrtPriceToTick(sqrtPrice float64, tickSpacing int) int {
	tick := int(2 * math.Log(sqrtPrice) / math.Log(1.0001))
	return (tick / tickSpacing) * tickSpacing
}

// CalculateReturns calculates returns from prices.
func CalculateReturns(prices []float64) []float64 {
	if len(prices) < 2 {
		return nil
	}

	returns := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] != 0 {
			returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
		}
	}
	return returns
}

// CalculateLogReturns calculates log returns from prices.
func CalculateLogReturns(prices []float64) []float64 {
	if len(prices) < 2 {
		return nil
	}

	returns := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] > 0 && prices[i] > 0 {
			returns[i-1] = math.Log(prices[i] / prices[i-1])
		}
	}
	return returns
}
