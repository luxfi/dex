package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type DexTrader struct {
	nc          *nats.Conn
	id          string
	serverFound bool
	orders      int64
	errors      int64
}

func main() {
	natsURL := flag.String("nats", "", "NATS URL (empty = auto-discover)")
	traders := flag.Int("traders", 10, "Number of traders")
	rate := flag.Int("rate", 1000, "Orders/sec per trader")
	duration := flag.Duration("duration", 30*time.Second, "Duration")
	autoScale := flag.Bool("auto", false, "Auto-scale to find max throughput")
	flag.Parse()

	hostname, _ := os.Hostname()
	id := fmt.Sprintf("%s-%d", hostname, os.Getpid())

	// Check if auto-scale mode
	if *autoScale {
		log.Printf("🚀 DEX Auto-Scaling Trader")
		log.Printf("📍 ID: %s", id)
		log.Printf("🎯 Finding maximum throughput...")
		runAutoScale(*natsURL, *traders, *rate, *duration)
		return
	}

	log.Printf("💹 DEX Trader starting")
	log.Printf("📍 ID: %s", id)
	log.Printf("📊 Config: %d traders @ %d orders/sec", *traders, *rate)

	// Auto-discover NATS
	if *natsURL == "" {
		*natsURL = discoverNATS()
	}

	// Connect to NATS
	nc, err := nats.Connect(*natsURL)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	trader := &DexTrader{
		nc: nc,
		id: id,
	}

	// Listen for server announcements
	nc.Subscribe("dex.announce", func(m *nats.Msg) {
		var ann struct {
			Type string `json:"type"`
			ID   string `json:"id"`
		}
		if json.Unmarshal(m.Data, &ann) == nil && ann.Type == "dex-server" {
			if !trader.serverFound {
				trader.serverFound = true
				log.Printf("🔍 Found DEX server: %s", ann.ID)
			}
		}
	})

	// Wait for server discovery
	log.Println("⏳ Waiting for DEX server...")
	for i := 0; i < 10; i++ {
		time.Sleep(time.Second)
		if trader.serverFound {
			break
		}
	}

	if !trader.serverFound {
		log.Fatal("❌ No DEX server found")
	}

	// Start trading
	log.Printf("📈 Starting %d traders", *traders)
	var wg sync.WaitGroup
	startTime := time.Now()

	for i := 0; i < *traders; i++ {
		wg.Add(1)
		go func(tid int) {
			defer wg.Done()
			trader.runTrader(tid, *rate, *duration)
		}(i)
	}

	// Stats printer
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		lastOrders := int64(0)
		
		for range ticker.C {
			orders := atomic.LoadInt64(&trader.orders)
			errors := atomic.LoadInt64(&trader.errors)
			delta := orders - lastOrders
			log.Printf("📊 Orders: %d | Rate: %d/sec | Errors: %d", 
				orders, delta, errors)
			lastOrders = orders
		}
	}()

	wg.Wait()
	
	// Final stats
	elapsed := time.Since(startTime).Seconds()
	finalOrders := atomic.LoadInt64(&trader.orders)
	log.Printf("✅ Complete: %d orders in %.1fs = %.0f orders/sec", 
		finalOrders, elapsed, float64(finalOrders)/elapsed)
}

func (t *DexTrader) runTrader(tid int, rate int, duration time.Duration) {
	orderID := uint64(tid * 1000000)
	endTime := time.Now().Add(duration)
	sleepNs := time.Duration(1000000000 / rate)
	
	for time.Now().Before(endTime) {
		orderID++
		
		order := fmt.Sprintf(`{"id":%d,"symbol":"BTC/USD","side":"buy","price":50000,"qty":1}`, orderID)
		
		msg, err := t.nc.Request("dex.orders", []byte(order), 100*time.Millisecond)
		if err != nil {
			atomic.AddInt64(&t.errors, 1)
		} else if msg != nil {
			atomic.AddInt64(&t.orders, 1)
		}
		
		if rate < 10000 {
			time.Sleep(sleepNs)
		}
	}
}

func discoverNATS() string {
	locations := []string{
		"nats://localhost:4222",
		"nats://127.0.0.1:4222",
		"nats://nats:4222",
	}
	
	for _, loc := range locations {
		nc, err := nats.Connect(loc, nats.Timeout(1*time.Second))
		if err == nil {
			nc.Close()
			log.Printf("✅ Found NATS at %s", loc)
			return loc
		}
	}
	
	return nats.DefaultURL
}

func runAutoScale(natsURL string, maxTraders int, targetRate int, maxDuration time.Duration) {
	// Auto-discover NATS
	if natsURL == "" {
		natsURL = discoverNATS()
	}

	// Connect to NATS
	nc, err := nats.Connect(natsURL)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	// Wait for server
	waitForServer(nc)

	// Find optimal configuration
	currentTraders := 1
	bestTraders := 1
	bestRate := float64(0)
	
	for currentTraders <= maxTraders {
		log.Printf("\n🧪 Testing with %d traders...", currentTraders)
		
		rate := testConfiguration(nc, currentTraders, targetRate, 10*time.Second)
		log.Printf("📈 Rate: %.0f orders/sec", rate)
		
		if rate > bestRate {
			bestRate = rate
			bestTraders = currentTraders
			log.Printf("✅ New best: %d traders = %.0f orders/sec", bestTraders, rate)
		} else if rate < bestRate*0.95 {
			log.Printf("📉 Performance degraded, stopping")
			break
		}
		
		// Scale up
		if currentTraders < 10 {
			currentTraders *= 2
		} else {
			currentTraders += 10
		}
		
		if currentTraders > maxTraders {
			break
		}
	}
	
	log.Printf("\n🏆 OPTIMAL: %d traders = %.0f orders/sec", bestTraders, bestRate)
	
	// Run optimal configuration
	log.Printf("🚀 Running optimal configuration for %v...", maxDuration)
	testConfiguration(nc, bestTraders, targetRate, maxDuration)
}

func waitForServer(nc *nats.Conn) {
	serverFound := false
	
	nc.Subscribe("dex.announce", func(m *nats.Msg) {
		var ann struct {
			Type string `json:"type"`
			ID   string `json:"id"`
		}
		if json.Unmarshal(m.Data, &ann) == nil && ann.Type == "dex-server" {
			if !serverFound {
				serverFound = true
				log.Printf("🔍 Found DEX server: %s", ann.ID)
			}
		}
	})

	log.Println("⏳ Waiting for DEX server...")
	for i := 0; i < 10; i++ {
		time.Sleep(time.Second)
		if serverFound {
			return
		}
	}
	
	log.Println("⚠️ No server found, continuing...")
}

func testConfiguration(nc *nats.Conn, numTraders int, targetRate int, duration time.Duration) float64 {
	var orders int64
	var wg sync.WaitGroup
	
	start := time.Now()
	ctx := make(chan bool)
	
	for i := 0; i < numTraders; i++ {
		wg.Add(1)
		go func(tid int) {
			defer wg.Done()
			
			sleepDuration := time.Second / time.Duration(targetRate)
			orderID := uint64(tid * 1000000)
			
			for {
				select {
				case <-ctx:
					return
				default:
					orderID++
					order := fmt.Sprintf(`{"id":%d,"symbol":"BTC/USD","side":"buy","price":50000,"qty":1}`, orderID)
					
					if _, err := nc.Request("dex.orders", []byte(order), 10*time.Millisecond); err == nil {
						atomic.AddInt64(&orders, 1)
					}
					
					if targetRate > 0 && targetRate < 100000 {
						time.Sleep(sleepDuration)
					}
				}
			}
		}(i)
	}
	
	time.Sleep(duration)
	close(ctx)
	wg.Wait()
	
	elapsed := time.Since(start).Seconds()
	finalOrders := atomic.LoadInt64(&orders)
	
	return float64(finalOrders) / elapsed
}