package main

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "github.com/luxfi/dex/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// Connect to server
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewEngineServiceClient(conn)
	ctx := context.Background()

	fmt.Println("=== TESTING LX ENGINE gRPC APIs ===")
	fmt.Println()

	// Test 1: Submit Buy Order
	fmt.Println("TEST 1: Submit Buy Order")
	buyOrder, err := client.SubmitOrder(ctx, &pb.SubmitOrderRequest{
		Symbol:   "BTC-USD",
		Side:     pb.OrderSide_ORDER_SIDE_BUY,
		Type:     pb.OrderType_ORDER_TYPE_LIMIT,
		Quantity: 1.0,
		Price:    42000.0,
		ClientOrderId: "test-buy-001",
	})
	if err != nil {
		log.Printf("❌ Submit buy order failed: %v", err)
	} else {
		fmt.Printf("✅ Buy order submitted: ID=%s, Status=%s\n", buyOrder.OrderId, buyOrder.Status)
	}

	// Test 2: Submit Sell Order
	fmt.Println("\nTEST 2: Submit Sell Order")
	sellOrder, err := client.SubmitOrder(ctx, &pb.SubmitOrderRequest{
		Symbol:   "BTC-USD",
		Side:     pb.OrderSide_ORDER_SIDE_SELL,
		Type:     pb.OrderType_ORDER_TYPE_LIMIT,
		Quantity: 1.0,
		Price:    43000.0,
		ClientOrderId: "test-sell-001",
	})
	if err != nil {
		log.Printf("❌ Submit sell order failed: %v", err)
	} else {
		fmt.Printf("✅ Sell order submitted: ID=%s, Status=%s\n", sellOrder.OrderId, sellOrder.Status)
	}

	// Test 3: Get Order Book
	fmt.Println("\nTEST 3: Get Order Book")
	orderBook, err := client.GetOrderBook(ctx, &pb.GetOrderBookRequest{
		Symbol: "BTC-USD",
		Depth:  10,
	})
	if err != nil {
		log.Printf("❌ Get order book failed: %v", err)
	} else {
		fmt.Printf("✅ Order book retrieved for %s\n", orderBook.Symbol)
		fmt.Printf("   Bids: %d levels\n", len(orderBook.Bids))
		fmt.Printf("   Asks: %d levels\n", len(orderBook.Asks))
		for i, bid := range orderBook.Bids {
			if i < 3 {
				fmt.Printf("     Bid %d: %.2f @ %.2f\n", i+1, bid.Quantity, bid.Price)
			}
		}
		for i, ask := range orderBook.Asks {
			if i < 3 {
				fmt.Printf("     Ask %d: %.2f @ %.2f\n", i+1, ask.Quantity, ask.Price)
			}
		}
	}

	// Test 4: Cancel Order
	fmt.Println("\nTEST 4: Cancel Order")
	if buyOrder != nil {
		cancelResp, err := client.CancelOrder(ctx, &pb.CancelOrderRequest{
			OrderId: buyOrder.OrderId,
		})
		if err != nil {
			log.Printf("❌ Cancel order failed: %v", err)
		} else {
			fmt.Printf("✅ Order cancelled: Success=%v\n", cancelResp.Success)
		}
	}

	// Test 5: Stream Order Book
	fmt.Println("\nTEST 5: Stream Order Book (3 seconds)")
	stream, err := client.StreamOrderBook(ctx, &pb.StreamOrderBookRequest{
		Symbol: "BTC-USD",
		Depth:  5,
	})
	if err != nil {
		log.Printf("❌ Stream order book failed: %v", err)
	} else {
		go func() {
			updates := 0
			for {
				update, err := stream.Recv()
				if err != nil {
					return
				}
				updates++
				fmt.Printf("✅ Order book update %d: %s, Bids=%d, Asks=%d\n", 
					updates, update.Symbol, len(update.BidUpdates), len(update.AskUpdates))
				if updates >= 3 {
					return
				}
			}
		}()
		time.Sleep(3 * time.Second)
	}

	// Test 6: Multiple Orders (Load Test)
	fmt.Println("\nTEST 6: Submit 100 Orders (Load Test)")
	start := time.Now()
	successCount := 0
	for i := 0; i < 100; i++ {
		_, err := client.SubmitOrder(ctx, &pb.SubmitOrderRequest{
			Symbol:   "ETH-USD",
			Side:     pb.OrderSide(i%2 + 1), // Alternate buy/sell
			Type:     pb.OrderType_ORDER_TYPE_LIMIT,
			Quantity: 0.1,
			Price:    2000.0 + float64(i),
			ClientOrderId: fmt.Sprintf("load-test-%03d", i),
		})
		if err == nil {
			successCount++
		}
	}
	duration := time.Since(start)
	fmt.Printf("✅ Load test complete: %d/100 orders submitted in %v\n", successCount, duration)
	fmt.Printf("   Rate: %.0f orders/second\n", float64(successCount)/duration.Seconds())

	fmt.Println("\n=== ALL TESTS COMPLETE ===")
}