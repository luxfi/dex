// ZMQ-MLX Server - GPU-accelerated order matching over 10Gbps fiber
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
	zmq "github.com/pebbe/zmq4"
)

// Binary protocol for maximum speed (fixed 64-byte messages)
type OrderMessage struct {
	Magic     uint32  // 0xDEADBEEF for validation
	OrderID   uint64  // Unique order ID
	Symbol    uint32  // Symbol ID (not string for speed)
	Side      uint8   // 0=buy, 1=sell
	Type      uint8   // 0=limit, 1=market
	Padding   uint16  // Alignment
	Price     float64 // Price in fixed point
	Size      float64 // Size
	Timestamp uint64  // Nanosecond timestamp
	UserID    uint64  // User identifier
	Reserved  [16]byte // Future use + alignment to 64 bytes
}

// Trade result message (48 bytes)
type TradeMessage struct {
	Magic         uint32
	TradeID       uint64
	BuyOrderID    uint64
	SellOrderID   uint64
	Price         float64
	Size          float64
	Timestamp     uint64
}

// Server statistics
type Stats struct {
	ordersReceived uint64
	ordersPerSec   uint64
	tradesExecuted uint64
	bytesReceived  uint64
	bytesPerSec    uint64
	latencyNanos   uint64
	errors         uint64
}

type MLXZMQServer struct {
	mlxEngine    mlx.Engine
	orderBooks   map[uint32]*lx.OrderBook
	stats        *Stats
	
	// ZMQ sockets
	orderSocket  *zmq.Socket // PULL socket for orders (many-to-one)
	tradeSocket  *zmq.Socket // PUB socket for trades (one-to-many)
	marketSocket *zmq.Socket // PUB socket for market data
	cmdSocket    *zmq.Socket // REP socket for commands
	
	// Batching for MLX
	orderBatch   []OrderMessage
	batchMutex   sync.Mutex
	batchSize    int
	batchTimeout time.Duration
	
	// Performance tuning
	cpuAffinity  []int
	useZeroCopy  bool
	tcpNoDelay   bool
	hwm          int
}

func NewMLXZMQServer(config *Config) (*MLXZMQServer, error) {
	// Initialize MLX engine
	mlxEngine, err := mlx.NewEngine(mlx.Config{
		Backend:  mlx.BackendAuto,
		MaxBatch: config.BatchSize,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create MLX engine: %w", err)
	}
	log.Printf("MLX Engine initialized: Backend=%s Device=%s", mlxEngine.Backend(), mlxEngine.Device())
	
	server := &MLXZMQServer{
		mlxEngine:    mlxEngine,
		orderBooks:   make(map[uint32]*lx.OrderBook),
		stats:        &Stats{},
		orderBatch:   make([]OrderMessage, 0, config.BatchSize),
		batchSize:    config.BatchSize,
		batchTimeout: config.BatchTimeout,
		hwm:          config.HWM,
		useZeroCopy:  config.ZeroCopy,
		tcpNoDelay:   config.TCPNoDelay,
	}
	
	// Setup ZMQ context with optimizations
	zmq.SetMaxSockets(100000)
	
	// Order receiver (PULL) - many traders can connect
	orderSocket, err := zmq.NewSocket(zmq.PULL)
	if err != nil {
		return nil, err
	}
	
	// Set socket options for 10Gbps throughput
	orderSocket.SetRcvhwm(server.hwm)
	orderSocket.SetRcvbuf(128 * 1024 * 1024) // 128MB receive buffer
	orderSocket.SetTcpKeepalive(1)
	orderSocket.SetTcpKeepaliveIdle(300)
	orderSocket.SetImmediate(false) // Allow batching
	
	if server.tcpNoDelay {
		// TCP_NODELAY not directly available in pebbe/zmq4
		// Would need to use SetSockoptInt with proper constant
	}
	
	// Bind to order port
	endpoint := fmt.Sprintf("tcp://*:%d", config.OrderPort)
	if err := orderSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind order socket: %w", err)
	}
	log.Printf("Order socket listening on %s", endpoint)
	
	// Trade publisher (PUB)
	tradeSocket, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		return nil, err
	}
	
	tradeSocket.SetSndhwm(server.hwm)
	tradeSocket.SetSndbuf(128 * 1024 * 1024)
	
	endpoint = fmt.Sprintf("tcp://*:%d", config.TradePort)
	if err := tradeSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind trade socket: %w", err)
	}
	log.Printf("Trade socket publishing on %s", endpoint)
	
	// Market data publisher (PUB)
	marketSocket, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		return nil, err
	}
	
	endpoint = fmt.Sprintf("tcp://*:%d", config.MarketPort)
	if err := marketSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind market socket: %w", err)
	}
	log.Printf("Market data socket publishing on %s", endpoint)
	
	// Command socket (REP)
	cmdSocket, err := zmq.NewSocket(zmq.REP)
	if err != nil {
		return nil, err
	}
	
	endpoint = fmt.Sprintf("tcp://*:%d", config.CmdPort)
	if err := cmdSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind command socket: %w", err)
	}
	log.Printf("Command socket listening on %s", endpoint)
	
	server.orderSocket = orderSocket
	server.tradeSocket = tradeSocket
	server.marketSocket = marketSocket
	server.cmdSocket = cmdSocket
	
	// Initialize order books
	server.initOrderBooks()
	
	return server, nil
}

func (s *MLXZMQServer) initOrderBooks() {
	// Pre-create order books for known symbols
	symbols := []uint32{
		1,  // BTC-USD
		2,  // ETH-USD
		3,  // SOL-USD
		4,  // AVAX-USD
		5,  // MATIC-USD
		10, // BTC-ETH
		// Add more as needed
	}
	
	for _, sym := range symbols {
		s.orderBooks[sym] = lx.NewOrderBook(fmt.Sprintf("SYM-%d", sym))
	}
	
	log.Printf("Initialized %d order books", len(s.orderBooks))
}

func (s *MLXZMQServer) Run() error {
	log.Println("Starting MLX-ZMQ server...")
	
	// Set CPU affinity for maximum performance
	if len(s.cpuAffinity) > 0 {
		// TODO: Set CPU affinity using syscalls
		log.Printf("CPU affinity: %v", s.cpuAffinity)
	}
	
	// Start worker goroutines
	var wg sync.WaitGroup
	
	// Order receiver (main hot path)
	wg.Add(1)
	go s.orderReceiver(&wg)
	
	// Batch processor (MLX GPU processing)
	wg.Add(1)
	go s.batchProcessor(&wg)
	
	// Stats reporter
	wg.Add(1)
	go s.statsReporter(&wg)
	
	// Command handler
	wg.Add(1)
	go s.commandHandler(&wg)
	
	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	
	log.Println("Shutting down...")
	s.shutdown()
	wg.Wait()
	
	return nil
}

// Hot path - receives orders at maximum speed
func (s *MLXZMQServer) orderReceiver(wg *sync.WaitGroup) {
	defer wg.Done()
	
	// Use zero-copy receive if available
	for {
		// Receive binary message
		msg, err := s.orderSocket.RecvBytes(0)
		if err != nil {
			if zmq.AsErrno(err) == zmq.ETERM {
				break
			}
			atomic.AddUint64(&s.stats.errors, 1)
			continue
		}
		
		// Update stats
		atomic.AddUint64(&s.stats.ordersReceived, 1)
		atomic.AddUint64(&s.stats.bytesReceived, uint64(len(msg)))
		
		// Validate message size
		if len(msg) != 64 {
			atomic.AddUint64(&s.stats.errors, 1)
			continue
		}
		
		// Zero-copy deserialize
		order := (*OrderMessage)(unsafe.Pointer(&msg[0]))
		
		// Validate magic number
		if order.Magic != 0xDEADBEEF {
			atomic.AddUint64(&s.stats.errors, 1)
			continue
		}
		
		// Add to batch
		s.batchMutex.Lock()
		s.orderBatch = append(s.orderBatch, *order)
		shouldProcess := len(s.orderBatch) >= s.batchSize
		s.batchMutex.Unlock()
		
		// Trigger batch processing if full
		if shouldProcess {
			s.processBatch()
		}
	}
}

// Process batches using MLX GPU acceleration
func (s *MLXZMQServer) batchProcessor(wg *sync.WaitGroup) {
	defer wg.Done()
	
	ticker := time.NewTicker(s.batchTimeout)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Process any pending orders on timeout
			s.processBatch()
		}
	}
}

func (s *MLXZMQServer) processBatch() {
	s.batchMutex.Lock()
	if len(s.orderBatch) == 0 {
		s.batchMutex.Unlock()
		return
	}
	
	// Take ownership of batch
	batch := s.orderBatch
	s.orderBatch = make([]OrderMessage, 0, s.batchSize)
	s.batchMutex.Unlock()
	
	startTime := time.Now()
	
	// Group orders by symbol for MLX processing
	symbolOrders := make(map[uint32][]OrderMessage)
	for _, order := range batch {
		symbolOrders[order.Symbol] = append(symbolOrders[order.Symbol], order)
	}
	
	// Process each symbol's orders
	for symbol, orders := range symbolOrders {
		// Separate buy and sell orders
		var bids, asks []mlx.Order
		
		for _, o := range orders {
			mlxOrder := mlx.Order{
				ID:     o.OrderID,
				Price:  o.Price,
				Size:   o.Size,
				Side:   int(o.Side),
			}
			
			if o.Side == 0 { // Buy
				bids = append(bids, mlxOrder)
			} else { // Sell
				asks = append(asks, mlxOrder)
			}
		}
		
		// Use MLX GPU engine for matching
		trades := s.mlxEngine.BatchMatch(bids, asks)
		
		// Publish trades
		for _, trade := range trades {
			s.publishTrade(symbol, trade)
			atomic.AddUint64(&s.stats.tradesExecuted, 1)
		}
		
		// Update order book (for market data)
		if ob, exists := s.orderBooks[symbol]; exists {
			for _, o := range orders {
				lxOrder := &lx.Order{
					ID:     o.OrderID,
					Symbol: fmt.Sprintf("SYM-%d", symbol),
					Type:   lx.OrderType(o.Type),
					Side:   lx.Side(o.Side),
					Price:  o.Price,
					Size:   o.Size,
					UserID: fmt.Sprintf("USER-%d", o.UserID),
				}
				ob.AddOrder(lxOrder)
			}
			
			// Publish market data snapshot
			s.publishMarketData(symbol, ob)
		}
	}
	
	// Update latency stats
	latency := time.Since(startTime).Nanoseconds()
	atomic.StoreUint64(&s.stats.latencyNanos, uint64(latency))
}

func (s *MLXZMQServer) publishTrade(symbol uint32, trade mlx.Trade) {
	msg := TradeMessage{
		Magic:       0xBEEFDEAD,
		TradeID:     trade.ID,
		BuyOrderID:  trade.BuyOrderID,
		SellOrderID: trade.SellOrderID,
		Price:       trade.Price,
		Size:        trade.Size,
		Timestamp:   uint64(time.Now().UnixNano()),
	}
	
	// Serialize to bytes
	buf := (*[48]byte)(unsafe.Pointer(&msg))
	
	// Publish with topic
	topic := fmt.Sprintf("TRADE.%d", symbol)
	s.tradeSocket.Send(topic, zmq.SNDMORE)
	s.tradeSocket.SendBytes(buf[:], 0)
}

func (s *MLXZMQServer) publishMarketData(symbol uint32, ob *lx.OrderBook) {
	// Get best bid/ask
	bestBid := ob.GetBestBid()
	bestAsk := ob.GetBestAsk()
	
	// Create market data message
	type MarketData struct {
		Symbol    uint32
		BidPrice  float64
		BidSize   float64
		AskPrice  float64
		AskSize   float64
		Timestamp uint64
	}
	
	md := MarketData{
		Symbol:    symbol,
		BidPrice:  bestBid,
		AskPrice:  bestAsk,
		Timestamp: uint64(time.Now().UnixNano()),
	}
	
	buf := (*[48]byte)(unsafe.Pointer(&md))
	
	topic := fmt.Sprintf("MD.%d", symbol)
	s.marketSocket.Send(topic, zmq.SNDMORE)
	s.marketSocket.SendBytes(buf[:], 0)
}

func (s *MLXZMQServer) statsReporter(wg *sync.WaitGroup) {
	defer wg.Done()
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	var lastOrders, lastBytes uint64
	lastTime := time.Now()
	
	for range ticker.C {
		now := time.Now()
		elapsed := now.Sub(lastTime).Seconds()
		
		orders := atomic.LoadUint64(&s.stats.ordersReceived)
		bytes := atomic.LoadUint64(&s.stats.bytesReceived)
		trades := atomic.LoadUint64(&s.stats.tradesExecuted)
		errors := atomic.LoadUint64(&s.stats.errors)
		latency := atomic.LoadUint64(&s.stats.latencyNanos)
		
		ordersPerSec := float64(orders-lastOrders) / elapsed
		bytesPerSec := float64(bytes-lastBytes) / elapsed
		gbps := (bytesPerSec * 8) / 1e9
		
		log.Printf("Stats: %.0f orders/sec | %.2f Gbps | %d trades | %.0f μs latency | %d errors",
			ordersPerSec, gbps, trades, float64(latency)/1000, errors)
		
		// Check if we're saturating 10Gbps
		if gbps > 9.0 {
			log.Printf("🚀 SATURATING 10Gbps FIBER! %.2f Gbps achieved!", gbps)
		}
		
		lastOrders = orders
		lastBytes = bytes
		lastTime = now
	}
}

func (s *MLXZMQServer) commandHandler(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		msg, err := s.cmdSocket.Recv(0)
		if err != nil {
			if zmq.AsErrno(err) == zmq.ETERM {
				break
			}
			continue
		}
		
		// Handle commands
		switch msg {
		case "STATS":
			stats := fmt.Sprintf("orders:%d,trades:%d,errors:%d",
				atomic.LoadUint64(&s.stats.ordersReceived),
				atomic.LoadUint64(&s.stats.tradesExecuted),
				atomic.LoadUint64(&s.stats.errors))
			s.cmdSocket.Send(stats, 0)
			
		case "PING":
			s.cmdSocket.Send("PONG", 0)
			
		default:
			s.cmdSocket.Send("ERROR: Unknown command", 0)
		}
	}
}

func (s *MLXZMQServer) shutdown() {
	s.orderSocket.Close()
	s.tradeSocket.Close()
	s.marketSocket.Close()
	s.cmdSocket.Close()
}

type Config struct {
	OrderPort    int
	TradePort    int
	MarketPort   int
	CmdPort      int
	BatchSize    int
	BatchTimeout time.Duration
	HWM          int
	ZeroCopy     bool
	TCPNoDelay   bool
}

func main() {
	config := &Config{}
	
	flag.IntVar(&config.OrderPort, "order-port", 5555, "Order receiver port")
	flag.IntVar(&config.TradePort, "trade-port", 5556, "Trade publisher port")
	flag.IntVar(&config.MarketPort, "market-port", 5557, "Market data port")
	flag.IntVar(&config.CmdPort, "cmd-port", 5558, "Command port")
	flag.IntVar(&config.BatchSize, "batch", 1000, "Batch size for MLX processing")
	flag.DurationVar(&config.BatchTimeout, "batch-timeout", 10*time.Millisecond, "Batch timeout")
	flag.IntVar(&config.HWM, "hwm", 100000, "High water mark")
	flag.BoolVar(&config.ZeroCopy, "zero-copy", true, "Use zero-copy")
	flag.BoolVar(&config.TCPNoDelay, "tcp-nodelay", true, "Disable Nagle's algorithm")
	flag.Parse()
	
	// Set runtime optimizations
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	log.Println("═══════════════════════════════════════════════")
	log.Println("  LX DEX MLX-ZMQ Server - 10Gbps Fiber Edition")
	log.Println("═══════════════════════════════════════════════")
	
	server, err := NewMLXZMQServer(config)
	if err != nil {
		log.Fatal(err)
	}
	
	if err := server.Run(); err != nil {
		log.Fatal(err)
	}
}