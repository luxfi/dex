package lx

import (
	"testing"
	"time"
)

// TestSimpleOrderBook tests basic order book functionality
func TestSimpleOrderBook(t *testing.T) {
	ob := NewOrderBook("BTC-USD")
	if ob == nil {
		t.Fatal("Failed to create order book")
	}

	if ob.Symbol != "BTC-USD" {
		t.Errorf("Expected symbol BTC-USD, got %s", ob.Symbol)
	}

	t.Log("✅ Order book created successfully")
}

// TestSimpleTradingEngine tests trading engine creation
func TestSimpleTradingEngine(t *testing.T) {
	config := EngineConfig{
		EnablePerps:   true,
		EnableVaults:  true,
		EnableLending: true,
	}

	engine := NewTradingEngine(config)
	if engine == nil {
		t.Fatal("Failed to create trading engine")
	}

	t.Log("✅ Trading engine created successfully")
}

// TestSimpleFundingConfig tests funding configuration
func TestSimpleFundingConfig(t *testing.T) {
	config := DefaultFundingConfig()

	if config.Interval != 8*time.Hour {
		t.Errorf("Expected 8 hour interval, got %v", config.Interval)
	}

	if len(config.FundingHours) != 3 {
		t.Errorf("Expected 3 funding hours, got %d", len(config.FundingHours))
	}

	// Check funding hours are 00:00, 08:00, 16:00
	expectedHours := []int{0, 8, 16}
	for i, hour := range config.FundingHours {
		if hour != expectedHours[i] {
			t.Errorf("Expected hour %d, got %d", expectedHours[i], hour)
		}
	}

	if config.MaxFundingRate != 0.0075 {
		t.Errorf("Expected max funding rate 0.0075, got %f", config.MaxFundingRate)
	}

	t.Log("✅ Funding configuration correct - 8 hour intervals at 00:00, 08:00, 16:00 UTC")
}
