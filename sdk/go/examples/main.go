package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/luxfi/dex/sdk/go/client"
)

func main() {
	// Create client with custom options
	c, err := client.NewClient(
		client.WithJSONRPCURL("http://localhost:8080"),
		client.WithWebSocketURL("ws://localhost:8081"),
		client.WithGRPCURL("localhost:50051"),
	)
	if err != nil {
		log.Fatal("Failed to create client:", err)
	}
	defer c.Disconnect()

	ctx := context.Background()

	// Example 1: Connect to gRPC for high-performance operations
	fmt.Println("Connecting to gRPC...")
	if err := c.ConnectGRPC(ctx); err != nil {
		log.Printf("Failed to connect to gRPC (will use JSON-RPC): %v", err)
	}

	// Example 2: Place a limit order
	fmt.Println("\n--- Placing Limit Order ---")
	order := &client.Order{
		Symbol:      "BTC-USD",
		Type:        client.OrderTypeLimit,
		Side:        client.OrderSideBuy,
		Price:       50000.00,
		Size:        0.1,
		UserID:      "user123",
		ClientID:    "my-order-001",
		TimeInForce: client.TimeInForceGTC,
	}

	orderResp, err := c.PlaceOrder(ctx, order)
	if err != nil {
		log.Printf("Failed to place order: %v", err)
	} else {
		fmt.Printf("Order placed successfully: ID=%d, Status=%s\n", orderResp.OrderID, orderResp.Status)
	}

	// Example 3: Get order book
	fmt.Println("\n--- Getting Order Book ---")
	orderBook, err := c.GetOrderBook(ctx, "BTC-USD", 10)
	if err != nil {
		log.Printf("Failed to get order book: %v", err)
	} else {
		fmt.Printf("Order Book for %s:\n", orderBook.Symbol)
		fmt.Printf("Best Bid: %.2f\n", orderBook.BestBid())
		fmt.Printf("Best Ask: %.2f\n", orderBook.BestAsk())
		fmt.Printf("Spread: %.2f (%.3f%%)\n", orderBook.Spread(), orderBook.SpreadPercentage())

		fmt.Println("\nTop 5 Bids:")
		for i := 0; i < min(5, len(orderBook.Bids)); i++ {
			fmt.Printf("  %.2f @ %.8f\n", orderBook.Bids[i].Price, orderBook.Bids[i].Size)
		}

		fmt.Println("\nTop 5 Asks:")
		for i := 0; i < min(5, len(orderBook.Asks)); i++ {
			fmt.Printf("  %.2f @ %.8f\n", orderBook.Asks[i].Price, orderBook.Asks[i].Size)
		}
	}

	// Example 4: Get recent trades
	fmt.Println("\n--- Getting Recent Trades ---")
	trades, err := c.GetTrades(ctx, "BTC-USD", 10)
	if err != nil {
		log.Printf("Failed to get trades: %v", err)
	} else {
		fmt.Printf("Recent trades for BTC-USD:\n")
		for _, trade := range trades {
			side := "BUY"
			if trade.Side == client.OrderSideSell {
				side = "SELL"
			}
			fmt.Printf("  %s %.2f @ %.8f (Total: %.2f) at %s\n",
				side, trade.Price, trade.Size, trade.TotalValue(),
				trade.TimestampTime().Format("15:04:05"))
		}
	}

	// Example 5: Connect WebSocket for real-time data
	fmt.Println("\n--- Connecting WebSocket ---")
	if err := c.ConnectWebSocket(ctx); err != nil {
		log.Printf("Failed to connect WebSocket: %v", err)
	} else {
		// Subscribe to order book updates
		err := c.SubscribeOrderBook("BTC-USD", func(ob *client.OrderBook) {
			fmt.Printf("[OrderBook Update] Best Bid: %.2f, Best Ask: %.2f, Spread: %.2f\n",
				ob.BestBid(), ob.BestAsk(), ob.Spread())
		})
		if err != nil {
			log.Printf("Failed to subscribe to order book: %v", err)
		}

		// Subscribe to trade updates
		err = c.SubscribeTrades("BTC-USD", func(trade *client.Trade) {
			side := "BUY"
			if trade.Side == client.OrderSideSell {
				side = "SELL"
			}
			fmt.Printf("[Trade] %s %.2f @ %.8f\n", side, trade.Price, trade.Size)
		})
		if err != nil {
			log.Printf("Failed to subscribe to trades: %v", err)
		}

		// Let it run for a bit to receive updates
		fmt.Println("Listening for WebSocket updates for 10 seconds...")
		time.Sleep(10 * time.Second)
	}

	// Example 6: Stream order book via gRPC
	if c.ConnectGRPC(ctx) == nil {
		fmt.Println("\n--- Streaming Order Book via gRPC ---")
		streamCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		obChan, err := c.StreamOrderBook(streamCtx, "BTC-USD")
		if err != nil {
			log.Printf("Failed to stream order book: %v", err)
		} else {
			fmt.Println("Streaming order book updates for 10 seconds...")
			for ob := range obChan {
				fmt.Printf("[gRPC Stream] Best Bid: %.2f, Best Ask: %.2f\n",
					ob.BestBid(), ob.BestAsk())
			}
		}
	}

	// Example 7: Cancel an order
	if orderResp != nil && orderResp.OrderID > 0 {
		fmt.Println("\n--- Cancelling Order ---")
		if err := c.CancelOrder(ctx, orderResp.OrderID); err != nil {
			log.Printf("Failed to cancel order: %v", err)
		} else {
			fmt.Printf("Order %d cancelled successfully\n", orderResp.OrderID)
		}
	}

	// Example 8: Get node info
	fmt.Println("\n--- Getting Node Info ---")
	info, err := c.GetInfo(ctx)
	if err != nil {
		log.Printf("Failed to get node info: %v", err)
	} else {
		fmt.Printf("Node Info:\n")
		fmt.Printf("  Version: %s\n", info.Version)
		fmt.Printf("  Network: %s\n", info.Network)
		fmt.Printf("  Orders: %d\n", info.OrderCount)
		fmt.Printf("  Trades: %d\n", info.TradeCount)
		fmt.Printf("  Syncing: %v\n", info.Syncing)
	}

	// Example 9: Ping server
	fmt.Println("\n--- Pinging Server ---")
	if err := c.Ping(ctx); err != nil {
		log.Printf("Failed to ping server: %v", err)
	} else {
		fmt.Println("Server is responsive!")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
