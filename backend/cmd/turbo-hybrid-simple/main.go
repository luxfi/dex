package main

// #cgo CFLAGS: -I/usr/local/include -I/opt/homebrew/include
// #cgo LDFLAGS: -L/usr/local/lib -L/opt/homebrew/lib -lzmq
// #include <zmq.h>
// #include <stdlib.h>
// #include <string.h>
import "C"
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
	"unsafe"

	"github.com/nats-io/nats.go"
)

type ServerInfo struct {
	ID        string    `json:"id"`
	ZMQAddr   string    `json:"zmq_addr"`
	IP        string    `json:"ip"`
	Hostname  string    `json:"hostname"`
	Timestamp time.Time `json:"timestamp"`
	Orders    int64     `json:"orders"`
	Trades    int64     `json:"trades"`
	Rate      int64     `json:"rate"`
}

type TurboHybrid struct {
	nc           *nats.Conn
	nodeID       string
	hostname     string
	isServer     bool
	zmqAddr      string
	zmqContext   unsafe.Pointer
	serverMu     sync.RWMutex
	activeServer *ServerInfo
	ordersCount  int64
	tradesCount  int64
	currentRate  int64
}

func main() {
	mode := flag.String("mode", "auto", "Mode: auto, server, trader")
	natsURL := flag.String("nats", "", "NATS URL (empty = auto-discover)")
	zmqPort := flag.Int("zmq-port", 5555, "ZeroMQ port for server")
	workers := flag.Int("workers", 0, "Workers (0 = auto)")
	traders := flag.Int("traders", 0, "Traders (0 = auto)")
	rate := flag.Int("rate", 10000, "Orders/sec per trader")
	duration := flag.Duration("duration", 30*time.Second, "Duration")
	flag.Parse()

	if *workers == 0 {
		*workers = runtime.NumCPU() * 4 // More workers for C ZMQ
	}
	if *traders == 0 {
		*traders = runtime.NumCPU() * 2 // More traders with C performance
	}

	hostname, _ := os.Hostname()
	nodeID := fmt.Sprintf("%s-%d", hostname, os.Getpid())
	nodeIP := getLocalIP()

	log.Printf("⚡ TURBO HYBRID - NATS Discovery + C ZeroMQ Trading")
	log.Printf("📍 Node ID: %s", nodeID)
	log.Printf("🌐 IP: %s", nodeIP)
	log.Printf("⚙️  Mode: %s", *mode)
	log.Printf("🔧 Using CGO=1 for maximum ZeroMQ performance")

	// Auto-discover NATS
	natsConn := *natsURL
	if natsConn == "" {
		natsConn = discoverNATS()
	}

	// Connect to NATS for discovery
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

	// Initialize C ZeroMQ context
	zmqContext := C.zmq_ctx_new()
	if zmqContext == nil {
		log.Fatal("Failed to create ZMQ context")
	}
	defer C.zmq_ctx_destroy(zmqContext)

	// Set high I/O threads for better performance
	C.zmq_ctx_set(zmqContext, C.ZMQ_IO_THREADS, 4)

	node := &TurboHybrid{
		nc:         nc,
		nodeID:     nodeID,
		hostname:   hostname,
		zmqAddr:    fmt.Sprintf("tcp://%s:%d", nodeIP, *zmqPort),
		zmqContext: zmqContext,
	}

	// Listen for server announcements
	go node.discoverer()

	// Wait for discovery
	time.Sleep(2 * time.Second)

	// Start metrics reporter
	go node.metricsReporter()

	switch *mode {
	case "auto":
		node.autoMode(*workers, *traders, *rate, *duration)
	case "server":
		node.runCZMQServer(*workers)
	case "trader":
		node.runCZMQTrader(*traders, *rate, *duration)
	}
}

func (n *TurboHybrid) autoMode(workers, traders int, rate int, duration time.Duration) {
	log.Println("🤖 AUTO MODE - Using NATS for discovery, C ZeroMQ for trading")
	
	// Wait for discovery
	time.Sleep(2 * time.Second)
	
	n.serverMu.RLock()
	hasServer := n.activeServer != nil
	n.serverMu.RUnlock()
	
	if !hasServer {
		// No server exists - I become THE server
		log.Println("🏆 No server found - becoming THE C ZeroMQ SERVER")
		go n.runCZMQServer(workers)
		time.Sleep(2 * time.Second) // Let server start
		// Also run traders
		n.runCZMQTrader(traders, rate, duration)
	} else {
		// Server exists - connect via ZeroMQ
		log.Printf("📍 Server found at %s - connecting via C ZeroMQ", n.activeServer.ZMQAddr)
		n.runCZMQTrader(traders, rate, duration)
	}
}

func (n *TurboHybrid) runCZMQServer(workers int) {
	n.isServer = true
	log.Printf("🏦 Starting C ZeroMQ Exchange Server on %s with %d workers", n.zmqAddr, workers)
	
	// Announce via NATS
	go n.announcer()
	
	// Create router socket using C API
	router := C.zmq_socket(n.zmqContext, C.ZMQ_ROUTER)
	if router == nil {
		log.Fatal("Failed to create router socket")
	}
	defer C.zmq_close(router)
	
	// Set socket options for performance
	hwm := C.int(1000000)
	C.zmq_setsockopt(router, C.ZMQ_RCVHWM, unsafe.Pointer(&hwm), C.size_t(unsafe.Sizeof(hwm)))
	C.zmq_setsockopt(router, C.ZMQ_SNDHWM, unsafe.Pointer(&hwm), C.size_t(unsafe.Sizeof(hwm)))
	
	// Bind
	addr := C.CString(n.zmqAddr)
	defer C.free(unsafe.Pointer(addr))
	if C.zmq_bind(router, addr) != 0 {
		log.Fatalf("Failed to bind to %s", n.zmqAddr)
	}
	
	// Create dealer socket for workers
	dealer := C.zmq_socket(n.zmqContext, C.ZMQ_DEALER)
	if dealer == nil {
		log.Fatal("Failed to create dealer socket")
	}
	defer C.zmq_close(dealer)
	
	workerAddr := C.CString("inproc://workers")
	defer C.free(unsafe.Pointer(workerAddr))
	if C.zmq_bind(dealer, workerAddr) != 0 {
		log.Fatal("Failed to bind dealer")
	}
	
	// Start worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			n.czmqWorker(id)
		}(i)
	}
	
	// Run proxy
	log.Printf("✅ C ZeroMQ server ready on %s", n.zmqAddr)
	C.zmq_proxy(router, dealer, nil)
}

func (n *TurboHybrid) czmqWorker(id int) {
	// Create worker socket using C API
	worker := C.zmq_socket(n.zmqContext, C.ZMQ_REP)
	if worker == nil {
		log.Printf("Failed to create worker socket %d", id)
		return
	}
	defer C.zmq_close(worker)
	
	workerAddr := C.CString("inproc://workers")
	defer C.free(unsafe.Pointer(workerAddr))
	if C.zmq_connect(worker, workerAddr) != 0 {
		log.Printf("Failed to connect worker %d", id)
		return
	}
	
	var ordersProcessed int64
	var tradesMatched int64
	buffer := make([]byte, 1024)
	
	for {
		// Receive message
		size := C.zmq_recv(worker, unsafe.Pointer(&buffer[0]), C.size_t(len(buffer)), 0)
		if size < 0 {
			break
		}
		
		ordersProcessed++
		atomic.AddInt64(&n.ordersCount, 1)
		
		// Simple matching logic (50% match rate)
		if ordersProcessed%2 == 0 {
			tradesMatched++
			atomic.AddInt64(&n.tradesCount, 1)
		}
		
		// Send response
		response := fmt.Sprintf(`{"id":%d,"status":"accepted","matched":%v}`, 
			ordersProcessed, ordersProcessed%2 == 0)
		resp := []byte(response)
		C.zmq_send(worker, unsafe.Pointer(&resp[0]), C.size_t(len(response)), 0)
		
		if ordersProcessed%100000 == 0 {
			log.Printf("Worker %d: Orders=%d, Trades=%d", id, ordersProcessed, tradesMatched)
		}
	}
}

func (n *TurboHybrid) runCZMQTrader(traders int, rate int, duration time.Duration) {
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
	
	log.Printf("💹 Starting %d C ZeroMQ traders connecting to %s", traders, serverAddr)
	
	var wg sync.WaitGroup
	var totalOrders int64
	var totalErrors int64
	startTime := time.Now()
	
	// Start traders
	for i := 0; i < traders; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			// Create socket using C API
			socket := C.zmq_socket(n.zmqContext, C.ZMQ_REQ)
			if socket == nil {
				log.Printf("Failed to create trader socket %d", id)
				return
			}
			defer C.zmq_close(socket)
			
			// Connect
			addr := C.CString(serverAddr)
			defer C.free(unsafe.Pointer(addr))
			if C.zmq_connect(socket, addr) != 0 {
				log.Printf("Failed to connect trader %d", id)
				return
			}
			
			orderID := uint64(id * 1000000)
			endTime := startTime.Add(duration)
			sleepNs := time.Duration(1000000000 / rate)
			buffer := make([]byte, 1024)
			
			for time.Now().Before(endTime) {
				orderID++
				
				// Send order
				order := fmt.Sprintf(`{"id":%d,"symbol":"BTC/USD","side":"buy","price":50000,"qty":1}`, orderID)
				orderData := []byte(order)
				
				if C.zmq_send(socket, unsafe.Pointer(&orderData[0]), C.size_t(len(order)), 0) < 0 {
					atomic.AddInt64(&totalErrors, 1)
					continue
				}
				
				// Receive response
				if C.zmq_recv(socket, unsafe.Pointer(&buffer[0]), C.size_t(len(buffer)), 0) < 0 {
					atomic.AddInt64(&totalErrors, 1)
					continue
				}
				
				atomic.AddInt64(&totalOrders, 1)
				
				if rate < 100000 {
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
			
			atomic.StoreInt64(&n.currentRate, delta)
			
			log.Printf("📈 C ZMQ Orders: %d | Rate: %d/sec | Errors: %d", 
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

func (n *TurboHybrid) announcer() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		info := ServerInfo{
			ID:        n.nodeID,
			ZMQAddr:   n.zmqAddr,
			IP:        getLocalIP(),
			Hostname:  n.hostname,
			Timestamp: time.Now(),
			Orders:    atomic.LoadInt64(&n.ordersCount),
			Trades:    atomic.LoadInt64(&n.tradesCount),
			Rate:      atomic.LoadInt64(&n.currentRate),
		}
		
		data, _ := json.Marshal(info)
		n.nc.Publish("turbo.server", data)
		log.Printf("📡 Announcing C ZMQ server at %s (Orders: %d, Rate: %d/sec)", 
			n.zmqAddr, info.Orders, info.Rate)
	}
}

func (n *TurboHybrid) discoverer() {
	n.nc.Subscribe("turbo.server", func(m *nats.Msg) {
		var info ServerInfo
		if json.Unmarshal(m.Data, &info) == nil {
			if info.ID != n.nodeID {
				n.serverMu.Lock()
				if n.activeServer == nil || info.Timestamp.Before(n.activeServer.Timestamp) {
					n.activeServer = &info
					log.Printf("🔍 Discovered C ZMQ server: %s at %s (Rate: %d/sec)", 
						info.ID, info.ZMQAddr, info.Rate)
				}
				n.serverMu.Unlock()
			}
		}
	})
}

func (n *TurboHybrid) metricsReporter() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		orders := atomic.LoadInt64(&n.ordersCount)
		trades := atomic.LoadInt64(&n.tradesCount)
		rate := atomic.LoadInt64(&n.currentRate)
		
		if orders > 0 {
			log.Printf("📊 Metrics: Orders=%d, Trades=%d, Rate=%d/sec, Match=%.1f%%",
				orders, trades, rate, float64(trades)/float64(orders)*100)
		}
	}
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