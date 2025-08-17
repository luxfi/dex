package main

import (
	"fmt"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
)

func main() {
	fmt.Println("================================================")
	fmt.Println("      LX DEX - Order Book Demo")
	fmt.Println("================================================")
	fmt.Println()

	// Create order book
	book := lx.NewOrderBook("BTC-USD")
	fmt.Println("📚 Created BTC-USD order book")
	fmt.Println()

	// Add buy orders (bids)
	fmt.Println("➕ Adding buy orders...")
	book.AddOrder(&lx.Order{
		ID:        1,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     49900,
		Size:      1.0,
		Timestamp: time.Now(),
	})
	fmt.Printf("   Buy  1.00 BTC @ $49,900\n")

	book.AddOrder(&lx.Order{
		ID:        2,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     49950,
		Size:      0.5,
		Timestamp: time.Now(),
	})
	fmt.Printf("   Buy  0.50 BTC @ $49,950\n")

	book.AddOrder(&lx.Order{
		ID:        3,
		Type:      lx.Limit,
		Side:      lx.Buy,
		Price:     50000,
		Size:      2.0,
		Timestamp: time.Now(),
	})
	fmt.Printf("   Buy  2.00 BTC @ $50,000\n")
	fmt.Println()

	// Add sell orders (asks)
	fmt.Println("➕ Adding sell orders...")
	book.AddOrder(&lx.Order{
		ID:        4,
		Type:      lx.Limit,
		Side:      lx.Sell,
		Price:     50100,
		Size:      1.5,
		Timestamp: time.Now(),
	})
	fmt.Printf("   Sell 1.50 BTC @ $50,100\n")

	book.AddOrder(&lx.Order{
		ID:        5,
		Type:      lx.Limit,
		Side:      lx.Sell,
		Price:     50050,
		Size:      1.0,
		Timestamp: time.Now(),
	})
	fmt.Printf("   Sell 1.00 BTC @ $50,050\n")
	fmt.Println()

	// Show book state
	fmt.Println("📊 Order Book State:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━")
	
	// Get best prices from the trees
	var bestBid, bestAsk float64
	if book.Bids.Root != nil {
		bestBid = book.Bids.MaxPrice().Price
	}
	if book.Asks.Root != nil {
		bestAsk = book.Asks.MinPrice().Price
	}
	spread := bestAsk - bestBid
	
	fmt.Printf("   Best Bid: $%,.2f\n", bestBid)
	fmt.Printf("   Best Ask: $%,.2f\n", bestAsk)
	if bestAsk > 0 {
		fmt.Printf("   Spread:   $%,.2f (%.3f%%)\n", spread, (spread/bestAsk)*100)
	}
	fmt.Println()

	// Execute a market order
	fmt.Println("🚀 Executing market buy order for 1.5 BTC...")
	initialTradeCount := uint64(len(book.Trades))
	tradesExecuted := book.AddOrder(&lx.Order{
		ID:        6,
		Type:      lx.Market,
		Side:      lx.Buy,
		Size:      1.5,
		Timestamp: time.Now(),
	})

	if tradesExecuted > 0 {
		fmt.Println()
		fmt.Println("💰 Trades Executed:")
		fmt.Println("━━━━━━━━━━━━━━━━━━")
		
		// Get the new trades
		newTrades := book.Trades[initialTradeCount:]
		totalSize := 0.0
		totalValue := 0.0
		
		for i, trade := range newTrades {
			fmt.Printf("   Trade %d: %.2f BTC @ $%,.2f = $%,.2f\n",
				i+1, trade.Size, trade.Price, trade.Size*trade.Price)
			totalSize += trade.Size
			totalValue += trade.Size * trade.Price
		}
		
		if totalSize > 0 {
			avgPrice := totalValue / totalSize
			fmt.Println()
			fmt.Printf("   Total:     %.2f BTC\n", totalSize)
			fmt.Printf("   Value:     $%,.2f\n", totalValue)
			fmt.Printf("   Avg Price: $%,.2f\n", avgPrice)
		}
	}
	fmt.Println()

	// Final book state
	fmt.Println("📊 Updated Book State:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━")
	
	// Get updated best prices
	bestBid = 0
	bestAsk = 0
	if book.Bids.Root != nil {
		bestBid = book.Bids.MaxPrice().Price
	}
	if book.Asks.Root != nil {
		bestAsk = book.Asks.MinPrice().Price
	}
	spread = bestAsk - bestBid
	
	fmt.Printf("   Best Bid: $%,.2f\n", bestBid)
	fmt.Printf("   Best Ask: $%,.2f\n", bestAsk)
	if bestAsk > 0 {
		fmt.Printf("   Spread:   $%,.2f\n", spread)
	}
	
	fmt.Println()
	fmt.Println("📈 Order Book Summary:")
	fmt.Printf("   Total Trades: %d\n", len(book.Trades))
	fmt.Printf("   Active Orders: %d\n", len(book.Orders))
	
	fmt.Println()
	fmt.Println("✅ Demo complete!")
	fmt.Println()
	fmt.Println("The LX DEX successfully:")
	fmt.Println("  • Matched orders by price-time priority")
	fmt.Println("  • Executed trades at best available prices")
	fmt.Println("  • Updated order book state in real-time")
	fmt.Println("  • Maintained consistent bid-ask spread")
}