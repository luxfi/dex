//go:build zmqtest

// ZMQ-Accel Server - GPU-accelerated order matching over 10Gbps fiber
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

	"github.com/luxfi/accel/ops/dex"
	"github.com/luxfi/czmq/v4"
	"github.com/luxfi/dex/pkg/lx"
)

// Binary protocol for maximum speed (fixed 64-byte messages)
type OrderMessage struct {
	Magic     uint32   // 0xDEADBEEF for validation
	OrderID   uint64   // Unique order ID
	Symbol    uint32   // Symbol ID (not string for speed)
	Side      uint8    // 0=buy, 1=sell
	Type      uint8    // 0=limit, 1=market
	Padding   uint16   // Alignment
	Price     float64  // Price in fixed point
	Size      float64  // Size
	Timestamp uint64   // Nanosecond timestamp
	UserID    uint64   // User identifier
	Reserved  [16]byte // Future use + alignment to 64 bytes
}

// Trade result message (48 bytes)
type TradeMessage struct {
	Magic       uint32
	TradeID     uint64
	BuyOrderID  uint64
	SellOrderID uint64
	Price       float64
	Size        float64
	Timestamp   uint64
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

type AccelZMQServer struct {
	orderBooks map[uint32]*lx.OrderBook
	stats      *Stats

	// ZMQ sockets
	orderSocket  *czmq.Sock // PULL socket for orders (many-to-one)
	tradeSocket  *czmq.Sock // PUB socket for trades (one-to-many)
	marketSocket *czmq.Sock // PUB socket for market data
	cmdSocket    *czmq.Sock // REP socket for commands

	// Batching for accel
	orderBatch   []OrderMessage
	batchMutex   sync.Mutex
	batchSize    int
	batchTimeout time.Duration

	// Performance tuning
	cpuAffinity []int
	useZeroCopy bool
	tcpNoDelay  bool
	hwm         int
}

func NewAccelZMQServer(config *Config) (*AccelZMQServer, error) {
	log.Printf("Accel Engine initialized with GPU acceleration via luxfi/accel")

	server := &AccelZMQServer{
		orderBooks:   make(map[uint32]*lx.OrderBook),
		stats:        &Stats{},
		orderBatch:   make([]OrderMessage, 0, config.BatchSize),
		batchSize:    config.BatchSize,
		batchTimeout: config.BatchTimeout,
		hwm:          config.HWM,
		useZeroCopy:  config.ZeroCopy,
		tcpNoDelay:   config.TCPNoDelay,
	}

	// Order receiver (PULL) - many traders can connect
	orderSocket := czmq.NewSock(czmq.Pull)

	// Set socket options for 10Gbps throughput
	orderSocket.SetRcvhwm(server.hwm)

	// Bind to order port
	endpoint := fmt.Sprintf("tcp://*:%d", config.OrderPort)
	if _, err := orderSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind order socket: %w", err)
	}
	log.Printf("Order socket listening on %s", endpoint)

	// Trade publisher (PUB)
	tradeSocket := czmq.NewSock(czmq.Pub)

	tradeSocket.SetSndhwm(server.hwm)

	endpoint = fmt.Sprintf("tcp://*:%d", config.TradePort)
	if _, err := tradeSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind trade socket: %w", err)
	}
	log.Printf("Trade socket publishing on %s", endpoint)

	// Market data publisher (PUB)
	marketSocket := czmq.NewSock(czmq.Pub)

	endpoint = fmt.Sprintf("tcp://*:%d", config.MarketPort)
	if _, err := marketSocket.Bind(endpoint); err != nil {
		return nil, fmt.Errorf("failed to bind market socket: %w", err)
	}
	log.Printf("Market data socket publishing on %s", endpoint)

	// Command socket (REP)
	cmdSocket := czmq.NewSock(czmq.Rep)

	endpoint = fmt.Sprintf("tcp://*:%d", config.CmdPort)
	if _, err := cmdSocket.Bind(endpoint); err != nil {
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

func (s *AccelZMQServer) initOrderBooks() {
	// Pre-create order books for known symbols
	symbols := []uint32{
		1,  // BTC-USD
		2,  // ETH-USD
		3,  // SOL-USD
		4,  // AVAX-USD
		5,  // MATIC-USD
		10, // BTC-ETH
	}

	for _, sym := range symbols {
		s.orderBooks[sym] = lx.NewOrderBook(fmt.Sprintf("SYM-%d", sym))
	}

	log.Printf("Initialized %d order books", len(s.orderBooks))
}

func (s *AccelZMQServer) Run() error {
	log.Println("Starting Accel-ZMQ server...")

	// Set CPU affinity for maximum performance
	if len(s.cpuAffinity) > 0 {
		log.Printf("CPU affinity: %v", s.cpuAffinity)
	}

	// Start worker goroutines
	var wg sync.WaitGroup

	// Order receiver (main hot path)
	wg.Add(1)
	go s.orderReceiver(&wg)

	// Batch processor (GPU processing)
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
func (s *AccelZMQServer) orderReceiver(wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		// Receive binary message
		msg, _, err := s.orderSocket.RecvFrame()
		if err != nil {
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

// Process batches using Accel GPU acceleration
func (s *AccelZMQServer) batchProcessor(wg *sync.WaitGroup) {
	defer wg.Done()

	ticker := time.NewTicker(s.batchTimeout)
	defer ticker.Stop()

	for range ticker.C {
		// Process any pending orders on timeout
		s.processBatch()
	}
}

func (s *AccelZMQServer) processBatch() {
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

	// Group orders by symbol for processing
	symbolOrders := make(map[uint32][]OrderMessage)
	for _, order := range batch {
		symbolOrders[order.Symbol] = append(symbolOrders[order.Symbol], order)
	}

	// Process each symbol's orders using accel/ops/dex
	for symbol, orders := range symbolOrders {
		// Separate buy and sell orders
		var bids, asks []dex.Order

		for _, o := range orders {
			accelOrder := dex.Order{
				ID:        o.OrderID,
				UserID:    o.UserID,
				Price:     uint64(o.Price * 1e8), // Convert to fixed-point
				Quantity:  uint64(o.Size * 1e8),
				Remaining: uint64(o.Size * 1e8),
				Side:      dex.Side(o.Side),
				Type:      dex.OrderType(o.Type),
			}

			if o.Side == 0 { // Buy
				bids = append(bids, accelOrder)
			} else { // Sell
				asks = append(asks, accelOrder)
			}
		}

		// Use accel for GPU-accelerated matching
		trades, _, err := dex.MatchOrders(bids, asks, nil)
		if err != nil {
			continue
		}

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

func (s *AccelZMQServer) publishTrade(symbol uint32, trade dex.Trade) {
	// Determine buy/sell order IDs based on taker side
	buyID, sellID := trade.MakerID, trade.TakerID
	if trade.TakerSide == dex.Ask {
		buyID, sellID = trade.TakerID, trade.MakerID
	}

	msg := TradeMessage{
		Magic:       0xBEEFDEAD,
		TradeID:     trade.ID,
		BuyOrderID:  buyID,
		SellOrderID: sellID,
		Price:       float64(trade.Price) / 1e8,
		Size:        float64(trade.Quantity) / 1e8,
		Timestamp:   uint64(time.Now().UnixNano()),
	}

	// Serialize to bytes
	buf := (*[48]byte)(unsafe.Pointer(&msg))

	// Publish with topic
	topic := fmt.Sprintf("TRADE.%d", symbol)
	s.tradeSocket.SendFrame([]byte(topic), czmq.FlagMore)
	s.tradeSocket.SendFrame(buf[:], 0)
}

func (s *AccelZMQServer) publishMarketData(symbol uint32, ob *lx.OrderBook) {
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
	s.marketSocket.SendFrame([]byte(topic), czmq.FlagMore)
	s.marketSocket.SendFrame(buf[:], 0)
}

func (s *AccelZMQServer) statsReporter(wg *sync.WaitGroup) {
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

		log.Printf("Stats: %.0f orders/sec | %.2f Gbps | %d trades | %.0f us latency | %d errors",
			ordersPerSec, gbps, trades, float64(latency)/1000, errors)

		// Check if we're saturating 10Gbps
		if gbps > 9.0 {
			log.Printf("[PERF] SATURATING 10Gbps FIBER! %.2f Gbps achieved!", gbps)
		}

		lastOrders = orders
		lastBytes = bytes
		lastTime = now
	}
}

func (s *AccelZMQServer) commandHandler(wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		msg, _, err := s.cmdSocket.RecvFrame()
		if err != nil {
			continue
		}

		// Handle commands
		switch string(msg) {
		case "STATS":
			stats := fmt.Sprintf("orders:%d,trades:%d,errors:%d",
				atomic.LoadUint64(&s.stats.ordersReceived),
				atomic.LoadUint64(&s.stats.tradesExecuted),
				atomic.LoadUint64(&s.stats.errors))
			s.cmdSocket.SendFrame([]byte(stats), 0)

		case "PING":
			s.cmdSocket.SendFrame([]byte("PONG"), 0)

		default:
			s.cmdSocket.SendFrame([]byte("ERROR: Unknown command"), 0)
		}
	}
}

func (s *AccelZMQServer) shutdown() {
	s.orderSocket.Destroy()
	s.tradeSocket.Destroy()
	s.marketSocket.Destroy()
	s.cmdSocket.Destroy()
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
	flag.IntVar(&config.BatchSize, "batch", 1000, "Batch size for processing")
	flag.DurationVar(&config.BatchTimeout, "batch-timeout", 10*time.Millisecond, "Batch timeout")
	flag.IntVar(&config.HWM, "hwm", 100000, "High water mark")
	flag.BoolVar(&config.ZeroCopy, "zero-copy", true, "Use zero-copy")
	flag.BoolVar(&config.TCPNoDelay, "tcp-nodelay", true, "Disable Nagle's algorithm")
	flag.Parse()

	// Set runtime optimizations
	runtime.GOMAXPROCS(runtime.NumCPU())

	log.Println("===============================================")
	log.Println("  LX Accel-ZMQ Server - 10Gbps Fiber Edition")
	log.Println("===============================================")

	server, err := NewAccelZMQServer(config)
	if err != nil {
		log.Fatal(err)
	}

	if err := server.Run(); err != nil {
		log.Fatal(err)
	}
}
