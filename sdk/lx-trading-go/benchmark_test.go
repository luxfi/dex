// Copyright 2024 Lux Industries Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trading

import (
	"testing"

	"github.com/shopspring/decimal"
)

// BenchmarkParseTradingPair measures trading pair parsing performance.
// This is a critical path for every market data update.
func BenchmarkParseTradingPair(b *testing.B) {
	symbols := []string{
		"BTC-USDC",
		"ETHUSDT",
		"LUX/USD",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ParseTradingPair(symbols[i%len(symbols)])
	}
}

// BenchmarkOrderbookAddBid measures orderbook bid insertion performance.
func BenchmarkOrderbookAddBid(b *testing.B) {
	book := NewOrderbook("BTC-USDC", "test")
	price := decimal.NewFromFloat(50000)
	qty := decimal.NewFromFloat(1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		book.AddBid(price, qty)
	}
}

// BenchmarkOrderbookSort measures orderbook sorting performance.
// Sorting is required after orderbook updates.
func BenchmarkOrderbookSort(b *testing.B) {
	for _, depth := range []int{10, 100, 1000} {
		b.Run(itoa(depth)+"_levels", func(b *testing.B) {
			book := NewOrderbook("BTC-USDC", "test")
			basePrice := 50000.0

			for i := 0; i < depth; i++ {
				book.AddBid(decimal.NewFromFloat(basePrice-float64(i)), decimal.NewFromFloat(1.0))
				book.AddAsk(decimal.NewFromFloat(basePrice+float64(i)+1), decimal.NewFromFloat(1.0))
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				book.Sort()
			}
		})
	}
}

// BenchmarkOrderbookBestBid measures best bid retrieval performance.
// This is called frequently for spread and arbitrage calculations.
func BenchmarkOrderbookBestBid(b *testing.B) {
	book := NewOrderbook("BTC-USDC", "test")
	for i := 0; i < 100; i++ {
		book.AddBid(decimal.NewFromFloat(50000-float64(i)), decimal.NewFromFloat(1.0))
	}
	book.Sort()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		book.BestBid()
	}
}

// BenchmarkOrderbookVwapBuy measures VWAP calculation performance.
// VWAP is used for large order execution planning.
func BenchmarkOrderbookVwapBuy(b *testing.B) {
	book := NewOrderbook("BTC-USDC", "test")
	for i := 0; i < 100; i++ {
		book.AddAsk(decimal.NewFromFloat(50000+float64(i)*10), decimal.NewFromFloat(1.0))
	}
	book.Sort()

	qty := decimal.NewFromFloat(50)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		book.VwapBuy(qty)
	}
}

// BenchmarkBlackScholes measures Black-Scholes pricing performance.
// Options pricing is computationally intensive.
func BenchmarkBlackScholes(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BlackScholes(100, 100, 1, 0.05, 0.2, true)
	}
}

// BenchmarkCalculateGreeks measures Greek calculation performance.
func BenchmarkCalculateGreeks(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CalculateGreeks(100, 100, 1, 0.05, 0.2, true)
	}
}

// BenchmarkImpliedVolatility measures IV calculation performance.
// IV calculation uses Newton-Raphson iteration.
func BenchmarkImpliedVolatility(b *testing.B) {
	price := BlackScholes(100, 100, 1, 0.05, 0.2, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ImpliedVolatility(price, 100, 100, 1, 0.05, true)
	}
}

// BenchmarkConstantProductPrice measures AMM pricing performance.
func BenchmarkConstantProductPrice(b *testing.B) {
	reserveX := decimal.NewFromFloat(1000)
	reserveY := decimal.NewFromFloat(2000000)
	amountIn := decimal.NewFromFloat(10)
	feeRate := decimal.NewFromFloat(0.003)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ConstantProductPrice(reserveX, reserveY, amountIn, feeRate, true)
	}
}

// BenchmarkRiskManagerValidateOrder measures order validation performance.
// Every order goes through risk validation.
func BenchmarkRiskManagerValidateOrder(b *testing.B) {
	config := RiskConfig{
		Enabled:         true,
		MaxOrderSize:    decimal.NewFromFloat(1000),
		MaxPositionSize: decimal.NewFromFloat(10000),
		MaxOpenOrders:   100,
	}

	rm := NewRiskManager(config)
	order := OrderRequest{
		Symbol:   "BTC-USDC",
		Side:     SideBuy,
		Quantity: decimal.NewFromFloat(5),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rm.ValidateOrder(order)
	}
}

// BenchmarkRiskManagerUpdatePosition measures position update performance.
func BenchmarkRiskManagerUpdatePosition(b *testing.B) {
	config := RiskConfig{Enabled: true}
	rm := NewRiskManager(config)
	qty := decimal.NewFromFloat(1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			rm.UpdatePosition("BTC", qty, SideBuy)
		} else {
			rm.UpdatePosition("BTC", qty, SideSell)
		}
	}
}

// BenchmarkVolatility measures volatility calculation performance.
func BenchmarkVolatility(b *testing.B) {
	returns := make([]float64, 252)
	for i := range returns {
		returns[i] = 0.01 * float64(i%10-5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Volatility(returns, true, 252)
	}
}

// BenchmarkSharpeRatio measures Sharpe ratio calculation performance.
func BenchmarkSharpeRatio(b *testing.B) {
	returns := make([]float64, 252)
	for i := range returns {
		returns[i] = 0.01 * float64(i%10-5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SharpeRatio(returns, 0.02, 252)
	}
}

// BenchmarkMaxDrawdown measures max drawdown calculation performance.
func BenchmarkMaxDrawdown(b *testing.B) {
	prices := make([]float64, 1000)
	for i := range prices {
		prices[i] = 100 + float64(i%50-25)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MaxDrawdown(prices)
	}
}

// BenchmarkVaR measures Value at Risk calculation performance.
func BenchmarkVaR(b *testing.B) {
	returns := make([]float64, 1000)
	for i := range returns {
		returns[i] = 0.01 * float64(i%20-10)
	}

	b.Run("historical", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			VaR(returns, 0.95, "historical")
		}
	})

	b.Run("parametric", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			VaR(returns, 0.95, "parametric")
		}
	})
}

// BenchmarkAggregatedOrderbook measures orderbook aggregation performance.
// Aggregation is done on every market data update in multi-venue mode.
func BenchmarkAggregatedOrderbook(b *testing.B) {
	// Create orderbooks for 3 venues
	books := make([]*Orderbook, 3)
	venues := []string{"lx_dex", "binance", "mexc"}

	for i := 0; i < 3; i++ {
		books[i] = NewOrderbook("BTC-USDC", venues[i])
		for j := 0; j < 10; j++ {
			books[i].AddBid(decimal.NewFromFloat(50000-float64(j*10+i)), decimal.NewFromFloat(1.0))
			books[i].AddAsk(decimal.NewFromFloat(50100+float64(j*10+i)), decimal.NewFromFloat(1.0))
		}
		books[i].Sort()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg := NewAggregatedOrderbook("BTC-USDC")
		for _, book := range books {
			agg.AddOrderbook(book)
		}
		agg.BestBid()
		agg.BestAsk()
	}
}

// BenchmarkDecimalOperations measures decimal arithmetic performance.
// Decimal operations are ubiquitous in trading calculations.
func BenchmarkDecimalOperations(b *testing.B) {
	a := decimal.NewFromFloat(50000.12345678)
	c := decimal.NewFromFloat(1.23456789)

	b.Run("add", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			a.Add(c)
		}
	})

	b.Run("mul", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			a.Mul(c)
		}
	})

	b.Run("div", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			a.Div(c)
		}
	})

	b.Run("cmp", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			a.GreaterThan(c)
		}
	})
}
