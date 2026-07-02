package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/luxfi/dex/pkg/dex"
)

// ---------------------------------------------------------------------------
// 1. BenchmarkOrderInsert — Single order insertion latency
// ---------------------------------------------------------------------------

func BenchmarkOrderInsert(b *testing.B) {
	ob := dex.NewOrderBook("BENCH-INSERT")

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		order := &dex.Order{
			Symbol: "BENCH-INSERT",
			Type:   dex.Limit,
			Side:   dex.Side(i % 2),
			Price:  100.0 + float64(i%100),
			Size:   1.0,
			UserID: "bench",
		}
		ob.AddOrder(order)
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "orders/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N), "ns/order")
}

// ---------------------------------------------------------------------------
// 2. BenchmarkOrderMatch — Match latency for crossing orders
// ---------------------------------------------------------------------------

func BenchmarkOrderMatch(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ob := dex.NewOrderBook("BENCH-MATCH")

		// Insert one buy and one sell that cross
		ob.AddOrder(&dex.Order{
			Symbol: "BENCH-MATCH",
			Type:   dex.Limit,
			Side:   dex.Buy,
			Price:  100.0,
			Size:   1.0,
			UserID: "buyer",
		})
		ob.AddOrder(&dex.Order{
			Symbol: "BENCH-MATCH",
			Type:   dex.Limit,
			Side:   dex.Sell,
			Price:  100.0,
			Size:   1.0,
			UserID: "seller",
		})

		ob.MatchOrders()
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "matches/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N), "ns/match")
}

// ---------------------------------------------------------------------------
// 3. BenchmarkCancelOrder — Cancel latency
// ---------------------------------------------------------------------------

func BenchmarkCancelOrder(b *testing.B) {
	// Pre-fill the book so cancels have work to do
	ob := dex.NewOrderBook("BENCH-CANCEL")
	ids := make([]uint64, b.N)

	for i := 0; i < b.N; i++ {
		order := &dex.Order{
			Symbol: "BENCH-CANCEL",
			Type:   dex.Limit,
			Side:   dex.Side(i % 2),
			Price:  100.0 + float64(i%200),
			Size:   1.0,
			UserID: "bench",
		}
		ids[i] = ob.AddOrder(order)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ob.CancelOrder(ids[i])
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "cancels/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N), "ns/cancel")
}

// ---------------------------------------------------------------------------
// 4. BenchmarkBatchOrders — 1000 orders in batch
// ---------------------------------------------------------------------------

func BenchmarkBatchOrders(b *testing.B) {
	const batchSize = 1000

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ob := dex.NewOrderBook("BENCH-BATCH")

		// Submit a batch of 1000 orders (500 buy, 500 sell)
		for j := 0; j < batchSize; j++ {
			order := &dex.Order{
				Symbol: "BENCH-BATCH",
				Type:   dex.Limit,
				Side:   dex.Side(j % 2),
				Price:  90.0 + float64(j%20),
				Size:   1.0,
				UserID: fmt.Sprintf("user-%d", j%10),
			}
			ob.AddOrder(order)
		}

		// Match all crossing orders
		ob.MatchOrders()
	}

	totalOrders := float64(b.N) * batchSize
	b.ReportMetric(totalOrders/b.Elapsed().Seconds(), "orders/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/totalOrders, "ns/order")
}

// ---------------------------------------------------------------------------
// 5. BenchmarkConcurrentMatching — Parallel matching across 100 markets
// ---------------------------------------------------------------------------

func BenchmarkConcurrentMatching(b *testing.B) {
	const numMarkets = 100

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		books := make([]*dex.OrderBook, numMarkets)
		for m := 0; m < numMarkets; m++ {
			books[m] = dex.NewOrderBook(fmt.Sprintf("MKT-%d", m))

			// Seed each book with crossing orders
			for j := 0; j < 10; j++ {
				books[m].AddOrder(&dex.Order{
					Symbol: fmt.Sprintf("MKT-%d", m),
					Type:   dex.Limit,
					Side:   dex.Buy,
					Price:  100.0,
					Size:   1.0,
					UserID: fmt.Sprintf("buyer-%d", j),
				})
				books[m].AddOrder(&dex.Order{
					Symbol: fmt.Sprintf("MKT-%d", m),
					Type:   dex.Limit,
					Side:   dex.Sell,
					Price:  100.0,
					Size:   1.0,
					UserID: fmt.Sprintf("seller-%d", j),
				})
			}
		}

		// Match all markets concurrently
		var wg sync.WaitGroup
		wg.Add(numMarkets)
		for m := 0; m < numMarkets; m++ {
			go func(book *dex.OrderBook) {
				defer wg.Done()
				book.MatchOrders()
			}(books[m])
		}
		wg.Wait()
	}

	totalMatches := float64(b.N) * numMarkets
	b.ReportMetric(totalMatches/b.Elapsed().Seconds(), "markets/sec")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/totalMatches, "ns/market")
}
