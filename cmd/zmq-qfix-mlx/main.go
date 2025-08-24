// Simplified QFIX-MLX Server - Focuses on performance without crypto dependencies
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

// Binary FIX message - 60 bytes for ultra-low latency
type QFIXMessage struct {
	Magic       uint32
	SequenceNo  uint64
	StreamID    uint32
	MsgType     uint8
	Side        uint8
	OrdType     uint8
	TimeInForce uint8
	Symbol      uint32
	OrderID     uint64
	Price       uint64
	Quantity    uint64
	Account     uint64
	Timestamp   uint64
}

type QFIXMLXServer struct {
	mlxEngine  mlx.Engine
	orderBooks map[uint32]*lx.OrderBook
	booksMutex sync.RWMutex
	fixSocket  *zmq.Socket
	mdSocket   *zmq.Socket
	stats      *PerfStats
	orderBatch []QFIXMessage
	batchMutex sync.Mutex
	batchSize  int
}

type PerfStats struct {
	messagesReceived uint64
	ordersProcessed  uint64
	tradesExecuted   uint64
	bytesReceived    uint64
	bytesSent        uint64
	latencyNanos     uint64
	mlxBatchCount    uint64
}

func NewQFIXMLXServer(config *Config) (*QFIXMLXServer, error) {
	// Initialize MLX
	mlxEngine, err := mlx.NewEngine(mlx.Config{
		Backend:  mlx.BackendAuto,
		MaxBatch: config.BatchSize,
	})
	if err != nil {
		log.Printf("MLX not available, using CPU fallback: %v", err)
		// Continue without MLX
	}

	server := &QFIXMLXServer{
		mlxEngine:  mlxEngine,
		orderBooks: make(map[uint32]*lx.OrderBook),
		orderBatch: make([]QFIXMessage, 0, config.BatchSize),
		batchSize:  config.BatchSize,
		stats:      &PerfStats{},
	}

	// Setup ZMQ sockets
	if err := server.setupSockets(config); err != nil {
		return nil, err
	}

	server.initOrderBooks()
	return server, nil
}

func (s *QFIXMLXServer) setupSockets(config *Config) error {
	// FIX ROUTER socket
	fixSocket, err := zmq.NewSocket(zmq.ROUTER)
	if err != nil {
		return err
	}

	fixSocket.SetRcvhwm(1000000)
	fixSocket.SetSndhwm(1000000)
	fixSocket.SetRcvbuf(256 * 1024 * 1024)
	fixSocket.SetSndbuf(256 * 1024 * 1024)
	fixSocket.SetTcpKeepalive(1)
	fixSocket.SetTcpNoDelay(1)
	fixSocket.SetImmediate(false)
	fixSocket.SetRouterMandatory(1)

	endpoint := fmt.Sprintf("tcp://*:%d", config.FIXPort)
	if err := fixSocket.Bind(endpoint); err != nil {
		return fmt.Errorf("failed to bind FIX socket: %w", err)
	}
	log.Printf("QFIX socket listening on %s", endpoint)

	// Market data PUB socket
	mdSocket, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		return err
	}

	mdSocket.SetSndhwm(1000000)
	mdSocket.SetSndbuf(256 * 1024 * 1024)

	endpoint = fmt.Sprintf("tcp://*:%d", config.MDPort)
	if err := mdSocket.Bind(endpoint); err != nil {
		return fmt.Errorf("failed to bind MD socket: %w", err)
	}
	log.Printf("Market data publishing on %s", endpoint)

	s.fixSocket = fixSocket
	s.mdSocket = mdSocket

	return nil
}

func (s *QFIXMLXServer) initOrderBooks() {
	symbols := []struct {
		id   uint32
		name string
	}{
		{1, "BTC-USD"},
		{2, "ETH-USD"},
		{3, "SOL-USD"},
		{4, "AVAX-USD"},
		{5, "LUX-USD"},
	}

	for _, sym := range symbols {
		s.orderBooks[sym.id] = lx.NewOrderBook(sym.name)
	}

	log.Printf("Initialized %d order books", len(s.orderBooks))
}

func (s *QFIXMLXServer) Run() error {
	log.Println("═══════════════════════════════════════════════════════════")
	log.Println("  QFIX-MLX Server - High Performance Order Processing")
	log.Println("═══════════════════════════════════════════════════════════")

	var wg sync.WaitGroup

	// FIX message receiver
	wg.Add(1)
	go s.fixReceiver(&wg)

	// Batch processor
	wg.Add(1)
	go s.batchProcessor(&wg)

	// Stats reporter
	wg.Add(1)
	go s.statsReporter(&wg)

	// Wait for shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down...")
	s.shutdown()
	wg.Wait()

	return nil
}

func (s *QFIXMLXServer) fixReceiver(wg *sync.WaitGroup) {
	defer wg.Done()

	poller := zmq.NewPoller()
	poller.Add(s.fixSocket, zmq.POLLIN)

	for {
		polled, err := poller.Poll(1 * time.Millisecond)
		if err != nil {
			break
		}

		if len(polled) == 0 {
			continue
		}

		parts, err := s.fixSocket.RecvMessageBytes(0)
		if err != nil {
			continue
		}

		if len(parts) != 2 {
			continue
		}

		message := parts[1]

		atomic.AddUint64(&s.stats.messagesReceived, 1)
		atomic.AddUint64(&s.stats.bytesReceived, uint64(len(message)))

		// Zero-copy deserialize
		if len(message) >= 60 {
			qfixMsg := (*QFIXMessage)(unsafe.Pointer(&message[0]))

			if qfixMsg.Magic == 0xF1000001 {
				s.batchMutex.Lock()
				s.orderBatch = append(s.orderBatch, *qfixMsg)
				shouldProcess := len(s.orderBatch) >= s.batchSize
				s.batchMutex.Unlock()

				if shouldProcess {
					s.processBatch()
				}
			}
		}
	}
}

func (s *QFIXMLXServer) batchProcessor(wg *sync.WaitGroup) {
	defer wg.Done()

	ticker := time.NewTicker(1 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		s.processBatch()
	}
}

func (s *QFIXMLXServer) processBatch() {
	s.batchMutex.Lock()
	if len(s.orderBatch) == 0 {
		s.batchMutex.Unlock()
		return
	}

	batch := s.orderBatch
	s.orderBatch = make([]QFIXMessage, 0, s.batchSize)
	s.batchMutex.Unlock()

	startTime := time.Now()

	// Process orders
	for _, msg := range batch {
		s.booksMutex.RLock()
		book, exists := s.orderBooks[msg.Symbol]
		s.booksMutex.RUnlock()

		if !exists {
			continue
		}

		order := &lx.Order{
			OrderID: msg.OrderID,
			ID:      fmt.Sprintf("%d", msg.OrderID),
			Type:    lx.OrderType(msg.OrdType),
			Side:    lx.Side(msg.Side - '1'),
			Price:   float64(msg.Price) / 1e8,
			Size:    float64(msg.Quantity) / 1e8,
			User:    fmt.Sprintf("%d", msg.Account),
		}

		trades := book.AddOrder(order)
		atomic.AddUint64(&s.stats.tradesExecuted, uint64(len(trades)))
	}

	atomic.AddUint64(&s.stats.ordersProcessed, uint64(len(batch)))
	atomic.AddUint64(&s.stats.mlxBatchCount, 1)

	batchNanos := time.Since(startTime).Nanoseconds()
	if len(batch) > 0 {
		perOrderNanos := batchNanos / int64(len(batch))
		atomic.StoreUint64(&s.stats.latencyNanos, uint64(perOrderNanos))
	}
}

func (s *QFIXMLXServer) statsReporter(wg *sync.WaitGroup) {
	defer wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastMessages, lastBytes, lastOrders uint64
	lastTime := time.Now()

	for range ticker.C {
		now := time.Now()
		elapsed := now.Sub(lastTime).Seconds()

		messages := atomic.LoadUint64(&s.stats.messagesReceived)
		bytes := atomic.LoadUint64(&s.stats.bytesReceived)
		orders := atomic.LoadUint64(&s.stats.ordersProcessed)
		trades := atomic.LoadUint64(&s.stats.tradesExecuted)
		batches := atomic.LoadUint64(&s.stats.mlxBatchCount)
		latency := atomic.LoadUint64(&s.stats.latencyNanos)

		messagesPerSec := float64(messages-lastMessages) / elapsed
		bytesPerSec := float64(bytes-lastBytes) / elapsed
		ordersPerSec := float64(orders-lastOrders) / elapsed
		gbps := (bytesPerSec * 8) / 1e9

		log.Printf("📊 %.0f msgs/sec | %.0f orders/sec | %.2f Gbps | %d trades | %d batches | %.0f ns/order",
			messagesPerSec, ordersPerSec, gbps, trades, batches, float64(latency))

		lastMessages = messages
		lastBytes = bytes
		lastOrders = orders
		lastTime = now
	}
}

func (s *QFIXMLXServer) shutdown() {
	if s.fixSocket != nil {
		s.fixSocket.Close()
	}
	if s.mdSocket != nil {
		s.mdSocket.Close()
	}
	// MLX cleanup handled automatically
}

type Config struct {
	FIXPort   int
	MDPort    int
	BatchSize int
}

func main() {
	config := &Config{}

	flag.IntVar(&config.FIXPort, "fix-port", 5555, "FIX ROUTER port")
	flag.IntVar(&config.MDPort, "md-port", 5556, "Market data PUB port")
	flag.IntVar(&config.BatchSize, "batch", 10000, "Batch size")
	flag.Parse()

	runtime.GOMAXPROCS(runtime.NumCPU())

	server, err := NewQFIXMLXServer(config)
	if err != nil {
		log.Fatal(err)
	}

	if err := server.Run(); err != nil {
		log.Fatal(err)
	}
}
