package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/nats-io/nats.go"
)

type PersistentServer struct {
	ob           *lx.OrderBook
	nc           *nats.Conn
	js           nats.JetStreamContext
	ordersCount  int64
	tradesCount  int64
	snapshotFile string
	mu           sync.RWMutex
	lastSnapshot time.Time
}

type Snapshot struct {
	Timestamp   time.Time `json:"timestamp"`
	OrdersCount int64     `json:"orders_count"`
	TradesCount int64     `json:"trades_count"`
	OrderBook   string    `json:"orderbook_state"`
}

func main() {
	var (
		natsURL      = flag.String("nats", nats.DefaultURL, "NATS server URL")
		snapshotFile = flag.String("snapshot", "dex-snapshot.json", "Snapshot file for persistence")
		restore      = flag.Bool("restore", true, "Restore from snapshot on startup")
	)
	flag.Parse()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	server := &PersistentServer{
		ob:           lx.NewOrderBook("BTC-USD"),
		snapshotFile: *snapshotFile,
	}

	// Restore from snapshot if exists
	if *restore {
		if err := server.restoreSnapshot(); err != nil {
			log.Printf("⚠️  No snapshot to restore: %v", err)
		} else {
			log.Printf("✅ Restored from snapshot: %d orders, %d trades",
				server.ordersCount, server.tradesCount)
		}
	}

	// Connect to NATS
	nc, err := nats.Connect(*natsURL,
		nats.MaxReconnects(-1),
		nats.ReconnectWait(1*time.Second),
	)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()
	server.nc = nc

	js, err := nc.JetStream()
	if err != nil {
		log.Fatalf("Failed to get JetStream: %v", err)
	}
	server.js = js

	fmt.Println("🏦 Persistent DEX Server")
	fmt.Println("========================")
	fmt.Printf("📍 ID: persistent-%d\n", os.Getpid())
	fmt.Printf("💾 Snapshot: %s\n", *snapshotFile)
	fmt.Printf("📊 Orders: %d, Trades: %d\n", server.ordersCount, server.tradesCount)
	fmt.Println("✅ Server ready!")

	// Start workers
	go server.orderProcessor()
	go server.snapshotWorker()
	go server.statsReporter()
	go server.heartbeat()

	// Wait for shutdown signal
	<-sigChan
	fmt.Println("\n⚠️  Shutting down...")

	// Save final snapshot
	if err := server.saveSnapshot(); err != nil {
		log.Printf("Failed to save snapshot: %v", err)
	} else {
		fmt.Printf("✅ Snapshot saved: %d orders, %d trades\n",
			server.ordersCount, server.tradesCount)
	}
}

func (s *PersistentServer) orderProcessor() {
	// Subscribe to orders
	sub, err := s.nc.SubscribeSync("orders.>")
	if err != nil {
		log.Fatal(err)
	}

	for {
		msg, err := sub.NextMsg(5 * time.Second)
		if err != nil {
			continue
		}

		// Process order
		s.mu.Lock()
		// Simulate order processing
		atomic.AddInt64(&s.ordersCount, 1)
		if atomic.LoadInt64(&s.ordersCount)%2 == 0 {
			atomic.AddInt64(&s.tradesCount, 1)
		}
		s.mu.Unlock()

		// Acknowledge
		msg.Ack()
	}
}

func (s *PersistentServer) snapshotWorker() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if err := s.saveSnapshot(); err != nil {
			log.Printf("Failed to save snapshot: %v", err)
		}
	}
}

func (s *PersistentServer) saveSnapshot() error {
	s.mu.RLock()
	snapshot := Snapshot{
		Timestamp:   time.Now(),
		OrdersCount: atomic.LoadInt64(&s.ordersCount),
		TradesCount: atomic.LoadInt64(&s.tradesCount),
		OrderBook:   fmt.Sprintf("%+v", s.ob.GetSnapshot()),
	}
	s.mu.RUnlock()

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}

	// Write to temp file first
	tempFile := s.snapshotFile + ".tmp"
	if err := os.WriteFile(tempFile, data, 0644); err != nil {
		return err
	}

	// Atomic rename
	return os.Rename(tempFile, s.snapshotFile)
}

func (s *PersistentServer) restoreSnapshot() error {
	data, err := os.ReadFile(s.snapshotFile)
	if err != nil {
		return err
	}

	var snapshot Snapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return err
	}

	s.ordersCount = snapshot.OrdersCount
	s.tradesCount = snapshot.TradesCount
	s.lastSnapshot = snapshot.Timestamp

	return nil
}

func (s *PersistentServer) statsReporter() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		orders := atomic.LoadInt64(&s.ordersCount)
		trades := atomic.LoadInt64(&s.tradesCount)
		fmt.Printf("📊 Stats: Orders=%d, Trades=%d\n", orders, trades)
	}
}

func (s *PersistentServer) heartbeat() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		data := fmt.Sprintf("persistent-%d", os.Getpid())
		s.nc.Publish("dex.heartbeat", []byte(data))
	}
}
