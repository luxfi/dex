package lx

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkMultiNodeScaling simulates multiple nodes processing orders
func BenchmarkMultiNodeScaling(b *testing.B) {
	configs := []struct {
		name  string
		nodes int
	}{
		{"1_Node", 1},
		{"2_Nodes", 2},
		{"4_Nodes", 4},
		{"8_Nodes", 8},
		{"10_Nodes", 10},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			// Create separate orderbooks for each "node"
			orderbooks := make([]*OrderBook, cfg.nodes)
			for i := 0; i < cfg.nodes; i++ {
				orderbooks[i] = NewOrderBook(fmt.Sprintf("BTC-USD-NODE%d", i))
			}

			// Pre-populate each orderbook
			for _, ob := range orderbooks {
				for i := 0; i < 100; i++ {
					ob.AddOrder(&Order{
						Symbol: ob.Symbol,
						Side:   Buy,
						Type:   Limit,
						Price:  49000 + float64(i),
						Size:   1.0,
						User:   "maker",
					})
					ob.AddOrder(&Order{
						Symbol: ob.Symbol,
						Side:   Sell,
						Type:   Limit,
						Price:  51000 + float64(i),
						Size:   1.0,
						User:   "maker",
					})
				}
			}

			ordersPerNode := 10000
			totalOrders := ordersPerNode * cfg.nodes

			b.ResetTimer()
			b.ReportAllocs()

			start := time.Now()
			var wg sync.WaitGroup
			var processedOrders int64

			// Launch concurrent "nodes"
			for nodeID := 0; nodeID < cfg.nodes; nodeID++ {
				wg.Add(1)
				go func(id int, ob *OrderBook) {
					defer wg.Done()
					for i := 0; i < ordersPerNode; i++ {
						order := &Order{
							Symbol: ob.Symbol,
							Side:   Side(i % 2),
							Type:   Limit,
							Price:  50000 + float64(i%1000),
							Size:   1.0,
							User:   fmt.Sprintf("node%d_user%d", id, i),
						}
						ob.AddOrder(order)
						atomic.AddInt64(&processedOrders, 1)
					}
				}(nodeID, orderbooks[nodeID])
			}

			wg.Wait()
			elapsed := time.Since(start)

			throughput := float64(totalOrders) / elapsed.Seconds()
			latencyUs := elapsed.Microseconds() / int64(totalOrders)

			b.ReportMetric(throughput, "orders/sec")
			b.ReportMetric(float64(latencyUs), "μs/order")
			b.ReportMetric(float64(cfg.nodes), "nodes")

			b.Logf("  %d nodes: %.0f orders/sec (%.1fx scaling)",
				cfg.nodes, throughput, throughput/(throughput/float64(cfg.nodes)))
		})
	}
}

// BenchmarkScalingEfficiency measures scaling efficiency
func BenchmarkScalingEfficiency(b *testing.B) {
	maxCores := runtime.NumCPU()
	if maxCores > 10 {
		maxCores = 10
	}

	singleCoreThroughput := 0.0

	for cores := 1; cores <= maxCores; cores *= 2 {
		if cores > maxCores {
			cores = maxCores
		}

		b.Run(fmt.Sprintf("%d_Cores", cores), func(b *testing.B) {
			runtime.GOMAXPROCS(cores)
			defer runtime.GOMAXPROCS(runtime.NumCPU())

			ob := NewOrderBook("BTC-USD")
			ordersPerCore := 10000
			totalOrders := ordersPerCore * cores

			// Pre-populate
			for i := 0; i < 1000; i++ {
				ob.AddOrder(&Order{
					Symbol: "BTC-USD",
					Side:   Side(i % 2),
					Type:   Limit,
					Price:  50000 + float64(i),
					Size:   1.0,
					User:   "maker",
				})
			}

			b.ResetTimer()

			start := time.Now()
			var wg sync.WaitGroup
			var processedOrders int64

			for c := 0; c < cores; c++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					for i := 0; i < ordersPerCore; i++ {
						order := &Order{
							Symbol: "BTC-USD",
							Side:   Side(i % 2),
							Type:   Limit,
							Price:  50000 + float64(i%1000),
							Size:   1.0,
							User:   fmt.Sprintf("core%d", id),
						}
						ob.AddOrder(order)
						atomic.AddInt64(&processedOrders, 1)
					}
				}(c)
			}

			wg.Wait()
			elapsed := time.Since(start)

			throughput := float64(totalOrders) / elapsed.Seconds()

			if cores == 1 {
				singleCoreThroughput = throughput
			}

			efficiency := 100.0
			if singleCoreThroughput > 0 {
				idealThroughput := singleCoreThroughput * float64(cores)
				efficiency = (throughput / idealThroughput) * 100
			}

			b.ReportMetric(throughput, "orders/sec")
			b.ReportMetric(efficiency, "% efficiency")

			b.Logf("  %d cores: %.0f orders/sec (%.1f%% efficiency)",
				cores, throughput, efficiency)
		})

		if cores == maxCores {
			break
		}
	}
}