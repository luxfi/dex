package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/luxfi/dex/backend/pkg/client"
	"github.com/luxfi/dex/backend/pkg/engine"
	"github.com/luxfi/dex/backend/pkg/orderbook"
	pb "github.com/luxfi/dex/backend/pkg/proto/engine"
)

var (
	mode   = flag.String("mode", "both", "Client mode: local, remote, or both")
	server = flag.String("server", "localhost:50051", "Remote server address")
)

func testClient(c *client.LXClient, modeName string) error {
	ctx := context.Background()

	fmt.Printf("\n=== Testing %s Mode ===\n", modeName)

	// Test 1: Submit Buy Order
	fmt.Println("1. Submitting buy order...")
	buyResp, err := c.SubmitOrder(ctx, &pb.SubmitOrderRequest{
		Symbol:        "ETH-USD",
		Side:          pb.OrderSide_ORDER_SIDE_BUY,
		Type:          pb.OrderType_ORDER_TYPE_LIMIT,
		Quantity:      1.0,
		Price:         2500.0,
		ClientOrderId: fmt.Sprintf("%s-buy-001", modeName),
	})
	if err != nil {
		return fmt.Errorf("submit buy order: %w", err)
	}
	fmt.Printf("   ✅ Order ID: %s\n", buyResp.OrderId)

	// Test 2: Submit Sell Order
	fmt.Println("2. Submitting sell order...")
	sellResp, err := c.SubmitOrder(ctx, &pb.SubmitOrderRequest{
		Symbol:        "ETH-USD",
		Side:          pb.OrderSide_ORDER_SIDE_SELL,
		Type:          pb.OrderType_ORDER_TYPE_LIMIT,
		Quantity:      1.0,
		Price:         2600.0,
		ClientOrderId: fmt.Sprintf("%s-sell-001", modeName),
	})
	if err != nil {
		return fmt.Errorf("submit sell order: %w", err)
	}
	fmt.Printf("   ✅ Order ID: %s\n", sellResp.OrderId)

	// Test 3: Get Order Book
	fmt.Println("3. Getting order book...")
	book, err := c.GetOrderBook(ctx, &pb.GetOrderBookRequest{
		Symbol: "ETH-USD",
		Depth:  5,
	})
	if err != nil {
		return fmt.Errorf("get order book: %w", err)
	}
	fmt.Printf("   ✅ Bids: %d, Asks: %d\n", len(book.Bids), len(book.Asks))

	// Test 4: Cancel Order
	fmt.Println("4. Cancelling order...")
	cancelResp, err := c.CancelOrder(ctx, &pb.CancelOrderRequest{
		OrderId: buyResp.OrderId,
	})
	if err != nil {
		return fmt.Errorf("cancel order: %w", err)
	}
	fmt.Printf("   ✅ Cancelled: %v\n", cancelResp.Success)

	// Test 5: Performance Test
	fmt.Println("5. Performance test (100 orders)...")
	start := time.Now()
	for i := 0; i < 100; i++ {
		_, err := c.SubmitOrder(ctx, &pb.SubmitOrderRequest{
			Symbol:        "ETH-USD",
			Side:          pb.OrderSide(i%2 + 1),
			Type:          pb.OrderType_ORDER_TYPE_LIMIT,
			Quantity:      0.1,
			Price:         2500.0 + float64(i),
			ClientOrderId: fmt.Sprintf("%s-perf-%03d", modeName, i),
		})
		if err != nil {
			return fmt.Errorf("performance test order %d: %w", i, err)
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("   ✅ 100 orders in %v (%.0f/sec)\n", elapsed, 100.0/elapsed.Seconds())

	return nil
}

func main() {
	flag.Parse()

	fmt.Println("=== LX CLIENT LIBRARY TEST ===")

	switch *mode {
	case "local":
		// Test local embedded engine
		cfg := client.ClientConfig{
			Mode: client.ModeLocal,
			EngineConfig: engine.Config{
				Mode: "development",
				OrderBook: orderbook.Config{
					Implementation:    orderbook.ImplGo,
					MaxOrdersPerLevel: 1000,
					PricePrecision:    7,
				},
			},
		}

		c, err := client.NewLXClient(cfg)
		if err != nil {
			log.Fatalf("Failed to create local client: %v", err)
		}
		defer c.Close()

		if err := testClient(c, "Local"); err != nil {
			log.Printf("❌ Local test failed: %v", err)
		}

	case "remote":
		// Test remote gRPC client
		cfg := client.ClientConfig{
			Mode:          client.ModeRemote,
			ServerAddress: *server,
		}

		c, err := client.NewLXClient(cfg)
		if err != nil {
			log.Fatalf("Failed to create remote client: %v", err)
		}
		defer c.Close()

		if err := testClient(c, "Remote"); err != nil {
			log.Printf("❌ Remote test failed: %v", err)
		}

	case "both":
		// Test both modes
		// First: Local
		localCfg := client.ClientConfig{
			Mode: client.ModeLocal,
			EngineConfig: engine.Config{
				Mode: "development",
				OrderBook: orderbook.Config{
					Implementation:    orderbook.ImplGo,
					MaxOrdersPerLevel: 1000,
					PricePrecision:    7,
				},
			},
		}

		localClient, err := client.NewLXClient(localCfg)
		if err != nil {
			log.Fatalf("Failed to create local client: %v", err)
		}

		if err := testClient(localClient, "Local"); err != nil {
			log.Printf("❌ Local test failed: %v", err)
		}
		localClient.Close()

		// Second: Remote
		remoteCfg := client.ClientConfig{
			Mode:          client.ModeRemote,
			ServerAddress: *server,
		}

		remoteClient, err := client.NewLXClient(remoteCfg)
		if err != nil {
			log.Fatalf("Failed to create remote client: %v", err)
		}

		if err := testClient(remoteClient, "Remote"); err != nil {
			log.Printf("❌ Remote test failed: %v", err)
		}
		remoteClient.Close()

	default:
		log.Fatalf("Invalid mode: %s (use local, remote, or both)", *mode)
	}

	fmt.Println("\n=== ALL CLIENT TESTS COMPLETE ===")
}
