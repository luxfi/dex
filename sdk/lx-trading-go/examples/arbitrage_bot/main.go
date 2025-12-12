// Example: Omnichain Arbitrage Bot using LX Trading SDK
//
// This bot detects and executes arbitrage opportunities across:
// - LX DEX (native RPC)
// - LX AMM (native RPC)
// - Binance, MEXC, OKX (via CCXT)
// - Uniswap, PancakeSwap (via Hummingbot Gateway)
//
// NO SMART CONTRACTS - just coordinated trades through unified API
//
// Cross-chain transport:
// - Warp: For Lux subnet communication (instant)
// - Teleport: For EVM chain bridging (~30s)
// - CEX API: Direct trading (instant)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	trading "github.com/luxfi/trading"
	"github.com/luxfi/trading/arbitrage"
	"github.com/shopspring/decimal"
)

func main() {
	// Configuration from environment
	config := trading.NewConfig()

	// Native LX DEX (fastest, lowest latency)
	config.WithNative("lx_dex", &trading.NativeVenueConfig{
		VenueType: "dex",
		APIURL:    getEnv("LX_DEX_URL", "https://api.dex.lux.network"),
		APIKey:    os.Getenv("LX_DEX_KEY"),
		APISecret: os.Getenv("LX_DEX_SECRET"),
	})

	// Native LX AMM
	config.WithNative("lx_amm", &trading.NativeVenueConfig{
		VenueType: "amm",
		APIURL:    getEnv("LX_AMM_URL", "https://api.amm.lux.network"),
	})

	// CCXT exchanges
	if key := os.Getenv("BINANCE_KEY"); key != "" {
		config.WithCcxt("binance", &trading.CcxtConfig{
			ExchangeID: "binance",
			APIKey:     key,
			APISecret:  os.Getenv("BINANCE_SECRET"),
		})
	}

	if key := os.Getenv("MEXC_KEY"); key != "" {
		config.WithCcxt("mexc", &trading.CcxtConfig{
			ExchangeID: "mexc",
			APIKey:     key,
			APISecret:  os.Getenv("MEXC_SECRET"),
		})
	}

	// Hummingbot Gateway for external DEXs
	if host := os.Getenv("GATEWAY_HOST"); host != "" {
		config.WithHummingbot("gateway", &trading.HummingbotConfig{
			Host:      host,
			Port:      15888,
			Connector: "uniswap",
			Chain:     "ethereum",
			Network:   "mainnet",
		})
	}

	// Risk management
	config.Risk = trading.RiskConfig{
		Enabled:          true,
		MaxPositionSize:  decimal.NewFromInt(10000),  // $10k max per trade
		MaxOrderSize:     decimal.NewFromInt(5000),   // $5k max order
		MaxDailyLoss:     decimal.NewFromInt(500),    // $500 daily loss limit
		KillSwitchEnabled: true,
	}

	// Create unified client
	client := trading.NewClient(config)

	// Connect to all venues
	ctx := context.Background()
	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Disconnect(ctx)

	log.Println("Connected to all venues")

	// Create client adapter for arbitrage
	arbClient := &clientAdapter{client: client}

	// Create arbitrage system
	arbConfig := arbitrage.UnifiedArbConfig{
		MinSpreadBps:     decimal.NewFromInt(15),   // 0.15% minimum spread
		MinProfit:        decimal.NewFromInt(10),   // $10 minimum profit
		MaxPositionSize:  decimal.NewFromInt(5000), // $5k max per arb
		MaxTotalExposure: decimal.NewFromInt(50000),// $50k max total
		Symbols: []string{
			"BTC-USDC",
			"ETH-USDC",
			"LUX-USDC",
			"SOL-USDC",
		},
		VenuePriority: []string{
			"lx_dex",   // Fastest (native)
			"binance",  // High liquidity
			"mexc",
			"lx_amm",
		},
		ScanInterval:    50 * time.Millisecond, // 20 scans/second
		ExecuteTimeout:  3 * time.Second,
		MaxDailyLoss:    decimal.NewFromInt(500),
		MaxTradesPerDay: 200,
	}

	arb := arbitrage.NewUnifiedArbitrage(arbClient, arbConfig)

	// Start arbitrage system
	if err := arb.Start(); err != nil {
		log.Fatalf("Failed to start arbitrage: %v", err)
	}

	log.Println("Arbitrage bot started")
	log.Printf("Monitoring: %v", arbConfig.Symbols)
	log.Printf("Min spread: %s bps, Min profit: $%s", arbConfig.MinSpreadBps, arbConfig.MinProfit)

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Stats reporting
	statsTicker := time.NewTicker(30 * time.Second)
	defer statsTicker.Stop()

	// Main loop
	for {
		select {
		case <-sigChan:
			log.Println("Shutting down...")
			arb.Stop()
			printFinalStats(arb)
			return

		case <-statsTicker.C:
			printStats(arb)
		}
	}
}

// clientAdapter adapts trading.Client to arbitrage.TradingClient
type clientAdapter struct {
	client *trading.Client
}

func (a *clientAdapter) AggregatedOrderbook(ctx context.Context, symbol string) (*arbitrage.AggregatedBook, error) {
	book, err := a.client.AggregatedOrderbook(ctx, symbol)
	if err != nil {
		return nil, err
	}

	result := &arbitrage.AggregatedBook{
		Symbol: book.Symbol,
		Bids:   make([]arbitrage.AggregatedLevel, 0),
		Asks:   make([]arbitrage.AggregatedLevel, 0),
	}

	// Convert bids
	for price, venues := range book.Bids {
		for _, vq := range venues {
			p, _ := decimal.NewFromString(price)
			result.Bids = append(result.Bids, arbitrage.AggregatedLevel{
				Price:     p,
				Quantity:  vq.Quantity,
				Venue:     vq.Venue,
				Timestamp: time.Now(),
			})
		}
	}

	// Convert asks
	for price, venues := range book.Asks {
		for _, vq := range venues {
			p, _ := decimal.NewFromString(price)
			result.Asks = append(result.Asks, arbitrage.AggregatedLevel{
				Price:     p,
				Quantity:  vq.Quantity,
				Venue:     vq.Venue,
				Timestamp: time.Now(),
			})
		}
	}

	return result, nil
}

func (a *clientAdapter) PlaceOrder(ctx context.Context, req arbitrage.OrderRequest) (*arbitrage.Order, error) {
	orderReq := trading.OrderRequest{
		Symbol:    req.Symbol,
		Side:      trading.Side(req.Side),
		OrderType: trading.OrderType(req.OrderType),
		Quantity:  req.Quantity,
		Price:     req.Price,
		Venue:     req.Venue,
	}

	order, err := a.client.PlaceOrder(ctx, orderReq)
	if err != nil {
		return nil, err
	}

	avgPrice := decimal.Zero
	if order.AveragePrice != nil {
		avgPrice = *order.AveragePrice
	}

	return &arbitrage.Order{
		OrderID:        order.OrderID,
		Symbol:         order.Symbol,
		Venue:          order.Venue,
		Side:           arbitrage.Side(order.Side),
		Quantity:       order.Quantity,
		FilledQuantity: order.FilledQuantity,
		AveragePrice:   avgPrice,
		Status:         string(order.Status),
	}, nil
}

func (a *clientAdapter) GetConnectedVenues() []arbitrage.VenueInfo {
	// Return configured venues (simplified - in real implementation would check connectivity)
	return []arbitrage.VenueInfo{
		{Name: "lx_dex", VenueType: "dex", Connected: true},
		{Name: "lx_amm", VenueType: "amm", Connected: true},
		{Name: "binance", VenueType: "cex", Connected: true},
		{Name: "mexc", VenueType: "cex", Connected: true},
	}
}

func printStats(arb *arbitrage.UnifiedArbitrage) {
	stats := arb.GetStats()
	log.Printf("Stats: Executions=%d, Successful=%d, WinRate=%.1f%%, PnL=$%s",
		stats.TotalExecutions,
		stats.SuccessfulExecutions,
		stats.WinRate*100,
		stats.TotalPnL.StringFixed(2),
	)
}

func printFinalStats(arb *arbitrage.UnifiedArbitrage) {
	stats := arb.GetStats()
	fmt.Println("\n=== FINAL STATS ===")
	fmt.Printf("Total Executions:  %d\n", stats.TotalExecutions)
	fmt.Printf("Successful:        %d\n", stats.SuccessfulExecutions)
	fmt.Printf("Win Rate:          %.1f%%\n", stats.WinRate*100)
	fmt.Printf("Total PnL:         $%s\n", stats.TotalPnL.StringFixed(2))
	fmt.Println("==================")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
