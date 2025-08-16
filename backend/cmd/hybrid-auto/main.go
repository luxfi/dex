package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	zmq "github.com/pebbe/zmq4"
)

type ServerInfo struct {
	ID        string    `json:"id"`
	ZMQAddr   string    `json:"zmq_addr"`  // LX address for high-perf trading
	IP        string    `json:"ip"`
	Hostname  string    `json:"hostname"`
	Timestamp time.Time `json:"timestamp"`
}

type HybridNode struct {
	nc          *nats.Conn
	nodeID      string
	hostname    string
	isServer    bool
	zmqAddr     string
	serverMu    sync.RWMutex
	activeServer *ServerInfo
}

func main() {
	mode := flag.String("mode", "auto", "Mode: auto, server, trader")
	natsURL := flag.String("nats", "", "NATS URL (empty = auto-discover)")
	zmqPort := flag.Int("zmq-port", 5555, "LX port for server")
	workers := flag.Int("workers", 0, "Workers (0 = auto)")
	traders := flag.Int("traders", 0, "Traders (0 = auto)")
	rate := flag.Int("rate", 1000, "Orders/sec per trader")
	duration := flag.Duration("duration", 30*time.Second, "Duration")
	flag.Parse()

	if *workers == 0 {
		*workers = runtime.NumCPU() * 2
	}
	if *traders == 0 {
		*traders = runtime.NumCPU()
	}

	hostname, _ := os.Hostname()
	nodeID := fmt.Sprintf("%s-%d", hostname, os.Getpid())
	nodeIP := getLocalIP()

	log.Printf("🚀 HYBRID Auto-Discovery (NATS + LX)")
	log.Printf("📍 Node ID: %s", nodeID)
	log.Printf("🌐 IP: %s", nodeIP)
	log.Printf("⚙️  Mode: %s", *mode)

	// Auto-discover NATS
	natsConn := *natsURL
	if natsConn == "" {
		natsConn = discoverNATS()
	}

	// Connect to NATS for discovery only
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

	node := &HybridNode{
		nc:       nc,
		nodeID:   nodeID,
		hostname: hostname,
		zmqAddr:  fmt.Sprintf("tcp://%s:%d", nodeIP, *zmqPort),
	}

	// Listen for server announcements
	go node.discoverer()

	// Wait for discovery
	time.Sleep(2 * time.Second)

	switch *mode {
	case "auto":
		node.autoMode(*workers, *traders, *rate, *duration)
	case "server":
		node.runZMQServer(*workers)
	case "trader":
		node.runZMQTrader(*traders, *rate, *duration)
	}
}

func (n *HybridNode) autoMode(workers, traders int, rate int, duration time.Duration) {
	log.Println("🤖 AUTO MODE - Using NATS for discovery, LX for trading")
	
	// Wait for discovery
	time.Sleep(2 * time.Second)
	
	n.serverMu.RLock()
	hasServer := n.activeServer != nil
	n.serverMu.RUnlock()
	
	if !hasServer {
		// No server exists - I become THE server
		log.Println("🏆 No server found - becoming THE LX SERVER")
		go n.runZMQServer(workers)
		time.Sleep(2 * time.Second) // Let server start
		// Also run traders
		n.runZMQTrader(traders, rate, duration)
	} else {
		// Server exists - connect via LX
		log.Printf("📍 Server found at %s - connecting via LX", n.activeServer.ZMQAddr)
		n.runZMQTrader(traders, rate, duration)
	}
}

func (n *HybridNode) runZMQServer(workers int) {
	n.isServer = true
	log.Printf("🏦 Starting LX Exchange Server on %s", n.zmqAddr)
	
	// Announce via NATS
	go n.announcer()
	
	// Start actual LX server
	context, _ := zmq.NewContext()
	defer context.Term()
	
	// Router socket for clients
	router, _ := context.NewSocket(zmq.ROUTER)
	defer router.Close()
	router.Bind(n.zmqAddr)
	
	// Dealer socket for workers
	dealer, _ := context.NewSocket(zmq.DEALER)
	defer dealer.Close()
	dealer.Bind("inproc://workers")
	
	// Start worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			n.zmqWorker(id)
		}(i)
	}
	
	// Proxy between router and dealer
	log.Printf("✅ LX server ready on %s", n.zmqAddr)
	zmq.Proxy(router, dealer, nil)
}

func (n *HybridNode) zmqWorker(id int) {
	context, _ := zmq.NewContext()
	defer context.Term()
	
	worker, _ := context.NewSocket(zmq.REP)
	defer worker.Close()
	worker.Connect("inproc://workers")
	
	var ordersProcessed int64
	var tradesMatched int64
	
	for {
		_, err := worker.RecvBytes(0)
		if err != nil {
			break
		}
		
		ordersProcessed++
		
		// Simple matching logic
		if ordersProcessed%2 == 0 {
			tradesMatched++
		}
		
		// Send response
		response := fmt.Sprintf(`{"id":%d,"status":"accepted","matched":%v}`, 
			ordersProcessed, ordersProcessed%2 == 0)
		worker.SendBytes([]byte(response), 0)
		
		if ordersProcessed%100000 == 0 {
			log.Printf("Worker %d: Orders=%d, Trades=%d", id, ordersProcessed, tradesMatched)
		}
	}
}

func (n *HybridNode) runZMQTrader(traders int, rate int, duration time.Duration) {
	// Wait for server discovery
	for i := 0; i < 10; i++ {
		n.serverMu.RLock()
		server := n.activeServer
		n.serverMu.RUnlock()
		
		if server != nil || n.isServer {
			break
		}
		log.Println("⏳ Waiting for server discovery...")
		time.Sleep(time.Second)
	}
	
	// Get server address
	serverAddr := n.zmqAddr // If we're the server
	n.serverMu.RLock()
	if n.activeServer != nil {
		serverAddr = n.activeServer.ZMQAddr
	}
	n.serverMu.RUnlock()
	
	log.Printf("💹 Starting %d LX traders connecting to %s", traders, serverAddr)
	
	var wg sync.WaitGroup
	var totalOrders int64
	var totalErrors int64
	startTime := time.Now()
	
	// Start traders
	for i := 0; i < traders; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			context, _ := zmq.NewContext()
			defer context.Term()
			
			socket, _ := context.NewSocket(zmq.REQ)
			defer socket.Close()
			
			socket.Connect(serverAddr)
			
			orderID := uint64(id * 1000000)
			endTime := startTime.Add(duration)
			sleepNs := time.Duration(1000000000 / rate)
			
			for time.Now().Before(endTime) {
				orderID++
				
				// Send order (binary format for speed)
				order := fmt.Sprintf(`{"id":%d,"symbol":"BTC/USD","side":"buy","price":50000,"qty":1}`, orderID)
				_, err := socket.SendBytes([]byte(order), 0)
				if err != nil {
					atomic.AddInt64(&totalErrors, 1)
					continue
				}
				
				// Receive response
				_, err = socket.RecvBytes(0)
				if err != nil {
					atomic.AddInt64(&totalErrors, 1)
					continue
				}
				
				atomic.AddInt64(&totalOrders, 1)
				
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
			errors := atomic.LoadInt64(&totalErrors)
			delta := orders - lastOrders
			
			log.Printf("📈 ZMQ Orders: %d | Rate: %d/sec | Errors: %d", 
				orders, delta, errors)
			lastOrders = orders
		}
	}()
	
	wg.Wait()
	finalOrders := atomic.LoadInt64(&totalOrders)
	elapsed := time.Since(startTime).Seconds()
	log.Printf("✅ Trading complete: %d orders in %.1fs = %.0f orders/sec", 
		finalOrders, elapsed, float64(finalOrders)/elapsed)
}

func (n *HybridNode) announcer() {
	// Announce server info on NATS
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		info := ServerInfo{
			ID:        n.nodeID,
			ZMQAddr:   n.zmqAddr,
			IP:        getLocalIP(),
			Hostname:  n.hostname,
			Timestamp: time.Now(),
		}
		
		data, _ := json.Marshal(info)
		n.nc.Publish("hybrid.server", data)
		log.Printf("📡 Announcing ZMQ server at %s", n.zmqAddr)
	}
}

func (n *HybridNode) discoverer() {
	// Listen for server announcements
	n.nc.Subscribe("hybrid.server", func(m *nats.Msg) {
		var info ServerInfo
		if json.Unmarshal(m.Data, &info) == nil {
			if info.ID != n.nodeID {
				n.serverMu.Lock()
				if n.activeServer == nil || info.Timestamp.Before(n.activeServer.Timestamp) {
					n.activeServer = &info
					log.Printf("🔍 Discovered ZMQ server: %s at %s", info.ID, info.ZMQAddr)
				}
				n.serverMu.Unlock()
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
	locations := []string{
		"nats://localhost:4222",
		"nats://nats:4222",
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