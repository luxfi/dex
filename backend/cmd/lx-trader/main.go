package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"time"
)

// Order represents a trading order
type Order struct {
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"`
	Price    float64 `json:"price"`
	Quantity float64 `json:"quantity"`
	Type     string  `json:"type"`
	TraderID string  `json:"trader_id"`
}

// Asset represents a tradeable asset
type Asset struct {
	Symbol   string  `json:"symbol"`
	Name     string  `json:"name"`
	Decimals int     `json:"decimals"`
	Active   bool    `json:"active"`
}

// Initialize assets on the DEX
func initializeAssets(endpoint string) error {
	assets := []Asset{
		{Symbol: "BTC-USD", Name: "Bitcoin/USD", Decimals: 8, Active: true},
		{Symbol: "ETH-USD", Name: "Ethereum/USD", Decimals: 8, Active: true},
		{Symbol: "LUX-USD", Name: "Lux/USD", Decimals: 8, Active: true},
		{Symbol: "SOL-USD", Name: "Solana/USD", Decimals: 8, Active: true},
	}
	
	for _, asset := range assets {
		data, _ := json.Marshal(asset)
		resp, err := http.Post(endpoint+"/asset", "application/json", bytes.NewReader(data))
		if err != nil {
			return fmt.Errorf("failed to create asset %s: %v", asset.Symbol, err)
		}
		resp.Body.Close()
		fmt.Printf("✅ Initialized asset: %s\n", asset.Symbol)
	}
	
	return nil
}

// Submit an order to the DEX
func submitOrder(endpoint string, order Order) error {
	data, err := json.Marshal(order)
	if err != nil {
		return err
	}
	
	resp, err := http.Post(endpoint+"/order", "application/json", bytes.NewReader(data))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	body, _ := io.ReadAll(resp.Body)
	
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("order failed: %s", string(body))
	}
	
	fmt.Printf("📈 Order submitted: %s %s %.2f @ %.2f (Trader: %s)\n", 
		order.Side, order.Symbol, order.Quantity, order.Price, order.TraderID)
	
	return nil
}

// Create market makers that provide liquidity
func runMarketMaker(endpoint string, symbol string, makerID string) {
	// Initial spread around mid price
	midPrices := map[string]float64{
		"BTC-USD": 50000,
		"ETH-USD": 3000,
		"LUX-USD": 100,
		"SOL-USD": 150,
	}
	
	midPrice := midPrices[symbol]
	spread := midPrice * 0.001 // 0.1% spread
	
	// Place initial orders
	for i := 0; i < 5; i++ {
		// Buy orders (bids)
		buyPrice := midPrice - spread*float64(i+1)
		buyOrder := Order{
			Symbol:   symbol,
			Side:     "buy",
			Price:    buyPrice,
			Quantity: rand.Float64() * 2,
			Type:     "limit",
			TraderID: makerID,
		}
		submitOrder(endpoint, buyOrder)
		
		// Sell orders (asks)
		sellPrice := midPrice + spread*float64(i+1)
		sellOrder := Order{
			Symbol:   symbol,
			Side:     "sell",
			Price:    sellPrice,
			Quantity: rand.Float64() * 2,
			Type:     "limit",
			TraderID: makerID,
		}
		submitOrder(endpoint, sellOrder)
		
		time.Sleep(100 * time.Millisecond)
	}
	
	// Continue making markets
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// Random walk the mid price
		midPrice += (rand.Float64() - 0.5) * spread * 10
		
		// Cancel and replace orders (simplified - just add new ones)
		side := []string{"buy", "sell"}[rand.Intn(2)]
		price := midPrice
		if side == "buy" {
			price -= spread * (1 + rand.Float64()*2)
		} else {
			price += spread * (1 + rand.Float64()*2)
		}
		
		order := Order{
			Symbol:   symbol,
			Side:     side,
			Price:    price,
			Quantity: rand.Float64() * 3,
			Type:     "limit",
			TraderID: makerID,
		}
		
		submitOrder(endpoint, order)
	}
}

// Simulate retail traders taking liquidity
func runRetailTrader(endpoint string, traderID string) {
	symbols := []string{"BTC-USD", "ETH-USD", "LUX-USD", "SOL-USD"}
	
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		symbol := symbols[rand.Intn(len(symbols))]
		side := []string{"buy", "sell"}[rand.Intn(2)]
		
		// Sometimes use market orders, sometimes limit
		orderType := "limit"
		if rand.Float64() > 0.3 {
			orderType = "market"
		}
		
		order := Order{
			Symbol:   symbol,
			Side:     side,
			Quantity: rand.Float64() * 0.5,
			Type:     orderType,
			TraderID: traderID,
		}
		
		// For limit orders, set a price
		if orderType == "limit" {
			// Get rough price based on symbol
			basePrices := map[string]float64{
				"BTC-USD": 50000,
				"ETH-USD": 3000,
				"LUX-USD": 100,
				"SOL-USD": 150,
			}
			order.Price = basePrices[symbol] * (0.95 + rand.Float64()*0.1)
		}
		
		submitOrder(endpoint, order)
	}
}

func main() {
	var (
		endpoint = flag.String("endpoint", "http://localhost:8080", "DEX endpoint")
		mode     = flag.String("mode", "all", "Mode: init, maker, trader, all")
		symbol   = flag.String("symbol", "BTC-USD", "Symbol to trade")
		traderID = flag.String("trader", "", "Trader ID")
	)
	flag.Parse()
	
	if *traderID == "" {
		*traderID = fmt.Sprintf("trader-%d", rand.Intn(1000))
	}
	
	fmt.Println("=================================================")
	fmt.Println("   LX DEX Trading Client")
	fmt.Println("=================================================")
	fmt.Printf("Endpoint: %s\n", *endpoint)
	fmt.Printf("Mode: %s\n", *mode)
	fmt.Printf("Trader ID: %s\n", *traderID)
	fmt.Println()
	
	switch *mode {
	case "init":
		fmt.Println("🚀 Initializing assets...")
		if err := initializeAssets(*endpoint); err != nil {
			log.Fatal(err)
		}
		fmt.Println("✅ Assets initialized successfully!")
		
	case "maker":
		fmt.Printf("🤖 Starting market maker for %s...\n", *symbol)
		runMarketMaker(*endpoint, *symbol, *traderID+"-maker")
		
	case "trader":
		fmt.Println("👤 Starting retail trader...")
		runRetailTrader(*endpoint, *traderID+"-retail")
		
	case "all":
		// Initialize assets first
		fmt.Println("🚀 Initializing assets...")
		if err := initializeAssets(*endpoint); err != nil {
			log.Printf("Asset init error (may already exist): %v", err)
		}
		
		// Start market makers for each symbol
		symbols := []string{"BTC-USD", "ETH-USD", "LUX-USD", "SOL-USD"}
		for i, sym := range symbols {
			go runMarketMaker(*endpoint, sym, fmt.Sprintf("maker-%d", i))
			time.Sleep(500 * time.Millisecond)
		}
		
		// Start a few retail traders
		for i := 0; i < 3; i++ {
			go runRetailTrader(*endpoint, fmt.Sprintf("retail-%d", i))
			time.Sleep(1 * time.Second)
		}
		
		// Keep running
		fmt.Println("\n📊 Trading simulation running... Press Ctrl+C to stop")
		select {}
		
	default:
		fmt.Printf("Unknown mode: %s\n", *mode)
		flag.Usage()
	}
}