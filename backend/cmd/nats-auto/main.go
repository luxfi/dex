package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type Order struct {
	ID        uint64    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      string    `json:"side"`
	Price     float64   `json:"price"`
	Quantity  float64   `json:"quantity"`
	Timestamp time.Time `json:"timestamp"`
	NodeID    string    `json:"node_id"`
}

type NodeInfo struct {
	ID        string    `json:"id"`
	IP        string    `json:"ip"`
	Hostname  string    `json:"hostname"`
	Type      string    `json:"type"`
	Status    string    `json:"status"`
	Orders    int64     `json:"orders"`
	Trades    int64     `json:"trades"`
	Timestamp time.Time `json:"timestamp"`
}

type AutoNode struct {
	nc          *nats.Conn
	nodeID      string
	nodeIP      string
	hostname    string
	ordersCount int64
	tradesCount int64
	isServer    bool
	isTrader    bool
	peers       map[string]*NodeInfo
	mu          sync.RWMutex
}

func main() {
	mode := flag.String("mode", "auto", "Mode: auto, server, trader, or all")
	natsURL := flag.String("nats", "", "NATS URL (empty = auto-discover)")
	workers := flag.Int("workers", 0, "Workers (0 = auto)")
	traders := flag.Int("traders", 0, "Traders (0 = auto)")
	rate := flag.Int("rate", 1000, "Orders/sec per trader")
	duration := flag.Duration("duration", 0, "Duration (0 = run forever)")
	flag.Parse()

	if *workers == 0 {
		*workers = runtime.NumCPU() * 2
	}
	if *traders == 0 {
		*traders = runtime.NumCPU()
	}

	// Get node info
	hostname, _ := os.Hostname()
	nodeIP := getLocalIP()
	nodeID := fmt.Sprintf("%s-%d", hostname, os.Getpid())

	log.Printf("🚀 NATS Auto-Discovery Node")
	log.Printf("📍 Node ID: %s", nodeID)
	log.Printf("🌐 IP: %s", nodeIP)
	log.Printf("🖥️  Hostname: %s", hostname)
	log.Printf("⚙️  Mode: %s", *mode)

	// Auto-discover NATS if not specified
	natsConn := *natsURL
	if natsConn == "" {
		natsConn = discoverNATS()
	}
	log.Printf("📡 NATS: %s", natsConn)

	// Connect to NATS
	nc, err := nats.Connect(natsConn)
	if err != nil {
		log.Printf("⚠️  No NATS found, starting local NATS server...")
		startLocalNATS()
		time.Sleep(2 * time.Second)
		nc, err = nats.Connect(nats.DefaultURL)
		if err != nil {
			log.Fatalf("Failed to connect to NATS: %v", err)
		}
	}
	defer nc.Close()

	node := &AutoNode{
		nc:       nc,
		nodeID:   nodeID,
		nodeIP:   nodeIP,
		hostname: hostname,
		peers:    make(map[string]*NodeInfo),
	}

	// Announce ourselves
	go node.announcer()

	// Listen for other nodes
	go node.discoverer()

	// Wait a bit for discovery
	time.Sleep(2 * time.Second)

	// Decide what to run based on mode
	switch *mode {
	case "auto":
		node.autoMode(*workers, *traders, *rate, *duration)
	case "server":
		node.runServer(*workers)
	case "trader":
		node.runTrader(*traders, *rate, *duration)
	case "all":
		go node.runServer(*workers)
		node.runTrader(*traders, *rate, *duration)
	}

	// Keep running if duration is 0
	if *duration == 0 {
		select {}
	}
}

func (n *AutoNode) autoMode(workers, traders int, rate int, duration time.Duration) {
	log.Println("🤖 AUTO MODE - Detecting network topology...")
	
	// Wait for discovery
	time.Sleep(3 * time.Second)
	
	n.mu.RLock()
	serverCount := 0
	traderCount := 0
	var earliestServer *NodeInfo
	
	for _, peer := range n.peers {
		if peer.Type == "server" || peer.Type == "all" {
			serverCount++
			// Track the earliest server (by timestamp)
			if earliestServer == nil || peer.Timestamp.Before(earliestServer.Timestamp) {
				earliestServer = peer
			}
		}
		if peer.Type == "trader" || peer.Type == "all" {
			traderCount++
		}
	}
	n.mu.RUnlock()
	
	log.Printf("📊 Found %d servers, %d traders", serverCount, traderCount)
	
	// SINGLE SERVER LOGIC - Only ONE server allowed!
	if serverCount == 0 {
		// No server exists - I become THE server
		log.Println("🏆 No server found - becoming THE SERVER (and trader)")
		go n.runServer(workers)
		n.runTrader(traders, rate, duration)
	} else {
		// Server already exists - I'm just a trader
		log.Printf("📍 Server already exists (%s) - becoming TRADER only", earliestServer.ID)
		n.runTrader(traders, rate, duration)
	}
}

func (n *AutoNode) runServer(workers int) {
	n.isServer = true
	log.Printf("🏦 Starting DEX Server with %d workers", workers)
	
	// Subscribe to orders
	n.nc.QueueSubscribe("dex.orders", "servers", func(m *nats.Msg) {
		var order Order
		if json.Unmarshal(m.Data, &order) == nil {
			atomic.AddInt64(&n.ordersCount, 1)
			
			// Simple matching
			if order.ID%2 == 0 {
				atomic.AddInt64(&n.tradesCount, 1)
			}
			
			// Respond
			resp := map[string]interface{}{
				"status": "accepted",
				"id":     order.ID,
				"server": n.nodeID,
			}
			data, _ := json.Marshal(resp)
			m.Respond(data)
		}
	})
	
	// Stats printer
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			orders := atomic.LoadInt64(&n.ordersCount)
			trades := atomic.LoadInt64(&n.tradesCount)
			log.Printf("📊 Server Stats: Orders=%d, Trades=%d", orders, trades)
		}
	}()
	
	log.Println("✅ DEX Server ready!")
}

func (n *AutoNode) runTrader(traders int, rate int, duration time.Duration) {
	n.isTrader = true
	log.Printf("💹 Starting %d traders at %d orders/sec each", traders, rate)
	
	var wg sync.WaitGroup
	wg.Add(traders)
	
	startTime := time.Now()
	endTime := startTime.Add(duration)
	if duration == 0 {
		endTime = startTime.Add(365 * 24 * time.Hour) // Run for a year
	}
	
	var totalOrders int64
	var totalAccepted int64
	
	for i := 0; i < traders; i++ {
		go func(id int) {
			defer wg.Done()
			
			sleepNs := time.Duration(1000000000 / rate)
			orderID := uint64(0)
			
			for time.Now().Before(endTime) {
				order := Order{
					ID:        atomic.AddUint64(&orderID, 1),
					Symbol:    "BTC/USD",
					Side:      []string{"buy", "sell"}[rand.Intn(2)],
					Price:     50000 + rand.Float64()*10000,
					Quantity:  rand.Float64() * 10,
					Timestamp: time.Now(),
					NodeID:    n.nodeID,
				}
				
				data, _ := json.Marshal(order)
				
				// Send order
				msg, err := n.nc.Request("dex.orders", data, 100*time.Millisecond)
				if err == nil {
					atomic.AddInt64(&totalOrders, 1)
					var resp map[string]interface{}
					if json.Unmarshal(msg.Data, &resp) == nil {
						if resp["status"] == "accepted" {
							atomic.AddInt64(&totalAccepted, 1)
						}
					}
				}
				
				if rate < 10000 {
					time.Sleep(sleepNs)
				}
			}
		}(i)
	}
	
	// Stats printer
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		lastOrders := int64(0)
		
		for range ticker.C {
			orders := atomic.LoadInt64(&totalOrders)
			accepted := atomic.LoadInt64(&totalAccepted)
			delta := orders - lastOrders
			
			log.Printf("📈 Orders: %d | Rate: %d/sec | Accepted: %d", 
				orders, delta, accepted)
			lastOrders = orders
		}
	}()
	
	if duration > 0 {
		wg.Wait()
		log.Printf("✅ Trading complete: %d orders sent", totalOrders)
	}
}

func (n *AutoNode) announcer() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		nodeType := "none"
		if n.isServer && n.isTrader {
			nodeType = "all"
		} else if n.isServer {
			nodeType = "server"
		} else if n.isTrader {
			nodeType = "trader"
		}
		
		info := NodeInfo{
			ID:        n.nodeID,
			IP:        n.nodeIP,
			Hostname:  n.hostname,
			Type:      nodeType,
			Status:    "active",
			Orders:    atomic.LoadInt64(&n.ordersCount),
			Trades:    atomic.LoadInt64(&n.tradesCount),
			Timestamp: time.Now(),
		}
		
		data, _ := json.Marshal(info)
		n.nc.Publish("nodes.announce", data)
	}
}

func (n *AutoNode) discoverer() {
	n.nc.Subscribe("nodes.announce", func(m *nats.Msg) {
		var info NodeInfo
		if json.Unmarshal(m.Data, &info) == nil {
			if info.ID != n.nodeID {
				n.mu.Lock()
				if _, exists := n.peers[info.ID]; !exists {
					log.Printf("🔍 Discovered node: %s (%s) - %s at %s", 
						info.ID, info.Type, info.Hostname, info.IP)
				}
				n.peers[info.ID] = &info
				n.mu.Unlock()
			}
		}
	})
}

func getLocalIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "127.0.0.1"
	}
	
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				return ipnet.IP.String()
			}
		}
	}
	return "127.0.0.1"
}

func discoverNATS() string {
	// Try common NATS locations
	locations := []string{
		"nats://localhost:4222",
		"nats://nats:4222",
		"nats://nats-server:4222",
		"nats://127.0.0.1:4222",
	}
	
	for _, loc := range locations {
		nc, err := nats.Connect(loc, nats.Timeout(1*time.Second))
		if err == nil {
			nc.Close()
			log.Printf("✅ Found NATS at %s", loc)
			return loc
		}
	}
	
	// Try mDNS/Bonjour discovery
	// For now, just use default
	return nats.DefaultURL
}

func startLocalNATS() {
	cmd := exec.Command("nats-server", "-p", "4222")
	err := cmd.Start()
	if err != nil {
		log.Printf("Could not start NATS server: %v", err)
	} else {
		log.Println("✅ Started local NATS server on port 4222")
	}
}