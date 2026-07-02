package dex

import (
	"math"
	"math/big"
	"testing"
)

// floatEquals compares two float64 values with tolerance for floating point precision
func floatEquals(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestGetMaintenanceMargin(t *testing.T) {
	re := NewRiskEngine()

	tests := []struct {
		name     string
		symbol   string
		expected float64
	}{
		{
			name:     "BTC-USDT known symbol",
			symbol:   "BTC-USDT",
			expected: 0.005,
		},
		{
			name:     "ETH-USDT known symbol",
			symbol:   "ETH-USDT",
			expected: 0.005,
		},
		{
			name:     "BNB-USDT known symbol",
			symbol:   "BNB-USDT",
			expected: 0.01,
		},
		{
			name:     "unknown symbol falls back to default",
			symbol:   "UNKNOWN-USDT",
			expected: 0.025,
		},
		{
			name:     "empty symbol falls back to default",
			symbol:   "",
			expected: 0.025,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := re.GetMaintenanceMargin(tt.symbol)
			if result != tt.expected {
				t.Errorf("GetMaintenanceMargin(%q) = %v, want %v", tt.symbol, result, tt.expected)
			}
		})
	}
}

func TestGetInitialMargin(t *testing.T) {
	re := NewRiskEngine()

	tests := []struct {
		name     string
		symbol   string
		expected float64
	}{
		{
			name:     "BTC-USDT known symbol",
			symbol:   "BTC-USDT",
			expected: 0.01,
		},
		{
			name:     "ETH-USDT known symbol",
			symbol:   "ETH-USDT",
			expected: 0.01,
		},
		{
			name:     "BNB-USDT known symbol",
			symbol:   "BNB-USDT",
			expected: 0.02,
		},
		{
			name:     "unknown symbol falls back to default",
			symbol:   "UNKNOWN-USDT",
			expected: 0.05,
		},
		{
			name:     "empty symbol falls back to default",
			symbol:   "",
			expected: 0.05,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := re.GetInitialMargin(tt.symbol)
			if result != tt.expected {
				t.Errorf("GetInitialMargin(%q) = %v, want %v", tt.symbol, result, tt.expected)
			}
		})
	}
}

func TestCalculateLiquidationPrice(t *testing.T) {
	re := NewRiskEngine()

	tests := []struct {
		name     string
		position *MarginPosition
		expected float64
	}{
		{
			name: "long BTC position liquidation",
			position: &MarginPosition{
				Symbol:     "BTC-USDT",
				Side:       Buy,
				EntryPrice: 50000.0,
			},
			// Maintenance margin for BTC-USDT is 0.005
			// Long liquidation: EntryPrice * (1 - maintenanceMargin) = 50000 * 0.995 = 49750
			expected: 49750.0,
		},
		{
			name: "short BTC position liquidation",
			position: &MarginPosition{
				Symbol:     "BTC-USDT",
				Side:       Sell,
				EntryPrice: 50000.0,
			},
			// Short liquidation: EntryPrice * (1 + maintenanceMargin) = 50000 * 1.005 = 50250
			expected: 50250.0,
		},
		{
			name: "long ETH position liquidation",
			position: &MarginPosition{
				Symbol:     "ETH-USDT",
				Side:       Buy,
				EntryPrice: 3000.0,
			},
			// Maintenance margin for ETH-USDT is 0.005
			// Long liquidation: 3000 * 0.995 = 2985
			expected: 2985.0,
		},
		{
			name: "short ETH position liquidation",
			position: &MarginPosition{
				Symbol:     "ETH-USDT",
				Side:       Sell,
				EntryPrice: 3000.0,
			},
			// Short liquidation: 3000 * 1.005 = 3015
			expected: 3015.0,
		},
		{
			name: "long BNB position with higher margin",
			position: &MarginPosition{
				Symbol:     "BNB-USDT",
				Side:       Buy,
				EntryPrice: 400.0,
			},
			// Maintenance margin for BNB-USDT is 0.01
			// Long liquidation: 400 * 0.99 = 396
			expected: 396.0,
		},
		{
			name: "short BNB position with higher margin",
			position: &MarginPosition{
				Symbol:     "BNB-USDT",
				Side:       Sell,
				EntryPrice: 400.0,
			},
			// Short liquidation: 400 * 1.01 = 404
			expected: 404.0,
		},
		{
			name: "unknown symbol uses default margin",
			position: &MarginPosition{
				Symbol:     "UNKNOWN-USDT",
				Side:       Buy,
				EntryPrice: 100.0,
			},
			// Default maintenance margin is 0.025
			// Long liquidation: 100 * 0.975 = 97.5
			expected: 97.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := re.CalculateLiquidationPrice(tt.position)
			// Use epsilon comparison for floating point
			if !floatEquals(result, tt.expected, 0.0001) {
				t.Errorf("CalculateLiquidationPrice() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCheckMarginCall(t *testing.T) {
	re := NewRiskEngine()

	tests := []struct {
		name       string
		account    *MarginAccount
		markPrices map[string]float64
		expected   bool
	}{
		{
			name: "no margin call with sufficient collateral",
			account: &MarginAccount{
				Positions: map[string]*MarginPosition{
					"BTC-USDT": {
						Symbol: "BTC-USDT",
						Size:   1.0,
					},
				},
				CollateralAssets: map[string]*CollateralAsset{
					"USDT": {
						Asset:   "USDT",
						Amount:  big.NewInt(100000 * 1e8), // 100,000 USDT
						Haircut: 1.0,                      // 100% value
					},
				},
			},
			markPrices: map[string]float64{
				"BTC-USDT": 50000.0,
				"USDT":     1.0,
			},
			// Position value: 1 * 50000 = 50000
			// Maintenance margin for BTC-USDT: 0.005
			// Required margin: 50000 * 0.005 * 1.2 = 300
			// Available margin: 100000 * 1.0 * 1.0 = 100000
			// No margin call: 100000 >= 300
			expected: false,
		},
		{
			name: "margin call with insufficient collateral",
			account: &MarginAccount{
				Positions: map[string]*MarginPosition{
					"BTC-USDT": {
						Symbol: "BTC-USDT",
						Size:   100.0, // Large position
					},
				},
				CollateralAssets: map[string]*CollateralAsset{
					"USDT": {
						Asset:   "USDT",
						Amount:  big.NewInt(100 * 1e8), // Only 100 USDT
						Haircut: 1.0,
					},
				},
			},
			markPrices: map[string]float64{
				"BTC-USDT": 50000.0,
				"USDT":     1.0,
			},
			// Position value: 100 * 50000 = 5,000,000
			// Maintenance margin for BTC-USDT: 0.005
			// Required margin: 5,000,000 * 0.005 * 1.2 = 30,000
			// Available margin: 100 * 1.0 * 1.0 = 100
			// Margin call: 100 < 30,000
			expected: true,
		},
		{
			name: "margin call with haircut applied",
			account: &MarginAccount{
				Positions: map[string]*MarginPosition{
					"ETH-USDT": {
						Symbol: "ETH-USDT",
						Size:   10.0,
					},
				},
				CollateralAssets: map[string]*CollateralAsset{
					"ETH": {
						Asset:   "ETH",
						Amount:  big.NewInt(1 * 1e8), // 1 ETH
						Haircut: 0.5,                 // 50% haircut
					},
				},
			},
			markPrices: map[string]float64{
				"ETH-USDT": 3000.0,
				"ETH":      3000.0,
			},
			// Position value: 10 * 3000 = 30,000
			// Maintenance margin for ETH-USDT: 0.005
			// Required margin: 30,000 * 0.005 * 1.2 = 180
			// Available margin: 1 * 3000 / 1e8 * 1e8 * 0.5 = 1500
			// No margin call: 1500 >= 180
			expected: false,
		},
		{
			name: "empty positions no margin call",
			account: &MarginAccount{
				Positions:        map[string]*MarginPosition{},
				CollateralAssets: map[string]*CollateralAsset{},
			},
			markPrices: map[string]float64{},
			// No positions, no margin required
			// Available margin: 0
			// 0 < 0 is false
			expected: false,
		},
		{
			name: "multiple positions combined margin check",
			account: &MarginAccount{
				Positions: map[string]*MarginPosition{
					"BTC-USDT": {
						Symbol: "BTC-USDT",
						Size:   1.0,
					},
					"ETH-USDT": {
						Symbol: "ETH-USDT",
						Size:   10.0,
					},
				},
				CollateralAssets: map[string]*CollateralAsset{
					"USDT": {
						Asset:   "USDT",
						Amount:  big.NewInt(500 * 1e8), // 500 USDT
						Haircut: 1.0,
					},
				},
			},
			markPrices: map[string]float64{
				"BTC-USDT": 50000.0,
				"ETH-USDT": 3000.0,
				"USDT":     1.0,
			},
			// BTC position value: 1 * 50000 = 50000
			// ETH position value: 10 * 3000 = 30000
			// Total position value: 80000
			// BTC margin: 50000 * 0.005 = 250
			// ETH margin: 30000 * 0.005 = 150
			// Total required: (250 + 150) * 1.2 = 480
			// Available margin: 500 * 1.0 * 1.0 = 500
			// No margin call: 500 >= 480
			expected: false,
		},
		{
			name: "margin call boundary case",
			account: &MarginAccount{
				Positions: map[string]*MarginPosition{
					"BTC-USDT": {
						Symbol: "BTC-USDT",
						Size:   1.0,
					},
				},
				CollateralAssets: map[string]*CollateralAsset{
					"USDT": {
						Asset:   "USDT",
						Amount:  big.NewInt(299 * 1e8), // 299 USDT
						Haircut: 1.0,
					},
				},
			},
			markPrices: map[string]float64{
				"BTC-USDT": 50000.0,
				"USDT":     1.0,
			},
			// Position value: 1 * 50000 = 50000
			// Required margin: 50000 * 0.005 * 1.2 = 300
			// Available margin: 299
			// Margin call: 299 < 300
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := re.CheckMarginCall(tt.account, tt.markPrices)
			if result != tt.expected {
				t.Errorf("CheckMarginCall() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestValidateLeverage(t *testing.T) {
	re := NewRiskEngine()

	tests := []struct {
		name        string
		accountType MarginAccountType
		symbol      string
		leverage    float64
		expected    bool
	}{
		// CrossMargin tests (max 10x regardless of symbol limit)
		{
			name:        "cross margin valid leverage under limit",
			accountType: CrossMargin,
			symbol:      "BTC-USDT",
			leverage:    5.0,
			expected:    true,
		},
		{
			name:        "cross margin valid leverage at limit",
			accountType: CrossMargin,
			symbol:      "BTC-USDT",
			leverage:    10.0,
			expected:    true,
		},
		{
			name:        "cross margin invalid leverage over limit",
			accountType: CrossMargin,
			symbol:      "BTC-USDT",
			leverage:    15.0,
			expected:    false,
		},
		{
			name:        "cross margin capped at 10x even if symbol allows 100x",
			accountType: CrossMargin,
			symbol:      "BTC-USDT", // BTC-USDT allows 100x
			leverage:    11.0,
			expected:    false,
		},

		// IsolatedMargin tests (max 20x regardless of symbol limit)
		{
			name:        "isolated margin valid leverage under limit",
			accountType: IsolatedMargin,
			symbol:      "BTC-USDT",
			leverage:    15.0,
			expected:    true,
		},
		{
			name:        "isolated margin valid leverage at limit",
			accountType: IsolatedMargin,
			symbol:      "BTC-USDT",
			leverage:    20.0,
			expected:    true,
		},
		{
			name:        "isolated margin invalid leverage over limit",
			accountType: IsolatedMargin,
			symbol:      "BTC-USDT",
			leverage:    25.0,
			expected:    false,
		},

		// PortfolioMargin tests (uses full symbol limit)
		{
			name:        "portfolio margin valid BTC leverage",
			accountType: PortfolioMargin,
			symbol:      "BTC-USDT",
			leverage:    50.0,
			expected:    true,
		},
		{
			name:        "portfolio margin valid at max BTC leverage",
			accountType: PortfolioMargin,
			symbol:      "BTC-USDT",
			leverage:    100.0,
			expected:    true,
		},
		{
			name:        "portfolio margin invalid over max BTC leverage",
			accountType: PortfolioMargin,
			symbol:      "BTC-USDT",
			leverage:    101.0,
			expected:    false,
		},
		{
			name:        "portfolio margin respects lower BNB limit",
			accountType: PortfolioMargin,
			symbol:      "BNB-USDT", // BNB-USDT max is 50x
			leverage:    50.0,
			expected:    true,
		},
		{
			name:        "portfolio margin over BNB limit",
			accountType: PortfolioMargin,
			symbol:      "BNB-USDT",
			leverage:    51.0,
			expected:    false,
		},

		// Unknown symbol tests (uses default 20x)
		{
			name:        "cross margin unknown symbol uses default",
			accountType: CrossMargin,
			symbol:      "UNKNOWN-USDT",
			leverage:    10.0,
			expected:    true,
		},
		{
			name:        "isolated margin unknown symbol uses default",
			accountType: IsolatedMargin,
			symbol:      "UNKNOWN-USDT",
			leverage:    20.0,
			expected:    true,
		},
		{
			name:        "portfolio margin unknown symbol uses default 20x",
			accountType: PortfolioMargin,
			symbol:      "UNKNOWN-USDT",
			leverage:    20.0,
			expected:    true,
		},
		{
			name:        "portfolio margin unknown symbol over default",
			accountType: PortfolioMargin,
			symbol:      "UNKNOWN-USDT",
			leverage:    21.0,
			expected:    false,
		},

		// Invalid account type
		{
			name:        "invalid account type returns false",
			accountType: MarginAccountType(999),
			symbol:      "BTC-USDT",
			leverage:    5.0,
			expected:    false,
		},

		// Edge cases
		{
			name:        "zero leverage valid",
			accountType: CrossMargin,
			symbol:      "BTC-USDT",
			leverage:    0.0,
			expected:    true,
		},
		{
			name:        "negative leverage valid (treated as <= limit)",
			accountType: CrossMargin,
			symbol:      "BTC-USDT",
			leverage:    -1.0,
			expected:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := re.ValidateLeverage(tt.accountType, tt.symbol, tt.leverage)
			if result != tt.expected {
				t.Errorf("ValidateLeverage(%v, %q, %v) = %v, want %v",
					tt.accountType, tt.symbol, tt.leverage, result, tt.expected)
			}
		})
	}
}

func TestRiskEngineConcurrency(t *testing.T) {
	re := NewRiskEngine()

	// Test concurrent access to margin functions
	done := make(chan bool)

	// Spawn multiple goroutines accessing different functions
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				re.GetMaintenanceMargin("BTC-USDT")
				re.GetInitialMargin("ETH-USDT")
				re.CalculateLiquidationPrice(&MarginPosition{
					Symbol:     "BTC-USDT",
					Side:       Buy,
					EntryPrice: 50000.0,
				})
				re.ValidateLeverage(CrossMargin, "BTC-USDT", 5.0)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// If we get here without race conditions, the test passes
}
