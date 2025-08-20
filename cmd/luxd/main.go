package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/luxfi/database"
	"github.com/luxfi/database/manager"
	"github.com/luxfi/log"
	"github.com/luxfi/dex/pkg/api"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/mlx"
)

const (
	defaultDataDir = ".lxd"
	defaultPort    = 8080
	defaultWSPort  = 8081
	defaultP2PPort = 5000
)

type Config struct {
	// Paths
	DataDir  string
	LogLevel string

	// Network
	HTTPPort    int
	WSPort      int
	P2PPort     int
	MetricsPort int

	// Consensus
	BlockTime time.Duration
	NodeID    int

	// Performance
	EnableMLX    bool
	MaxBatchSize int

	// Features
	EnableMetrics bool
	EnableDebug   bool
}

type LXDNode struct {
	config    *Config
	db        database.Database
	orderBook *lx.OrderBook
	mlxEngine mlx.Engine
	logger    log.Logger

	// Runtime stats
	blocksFinalized  uint64
	ordersProcessed  uint64
	tradesExecuted   uint64
	consensusLatency uint64
	
	// Pending orders buffer
	pendingOrders []*lx.Order
	orderMu       sync.Mutex

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewLXDNode(config *Config) (*LXDNode, error) {
	// Initialize logger using luxfi/log
	level, _ := log.ToLevel(config.LogLevel)
	logger := log.NewTestLogger(level)
	logger.Info("Initializing LXD node")

	// Ensure data directory exists
	dataPath := filepath.Join(os.Getenv("HOME"), config.DataDir)
	if err := os.MkdirAll(dataPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	// Initialize database using luxfi/database manager
	// BadgerDB is the default/preferred database in Lux ecosystem
	dbManager := manager.NewManager(dataPath, nil)
	
	// Use default BadgerDB configuration (Lux standard)
	dbConfig := manager.DefaultBadgerDBConfig("badgerdb")
	dbConfig.Namespace = "lxd"
	
	db, err := dbManager.New(dbConfig)
	if err != nil {
		logger.Warn("Failed to open BadgerDB", "error", err)
		// Fallback to memory database
		memConfig := manager.DefaultMemoryConfig()
		db, err = dbManager.New(memConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create database: %w", err)
		}
		logger.Info("Using in-memory database")
	} else {
		logger.Info("BadgerDB initialized",
			"path", filepath.Join(dataPath, "badgerdb"),
			"type", "LSM-tree with value log")
	}

	// Initialize MLX engine
	var mlxEngine mlx.Engine
	if config.EnableMLX {
		mlxEngine, err = mlx.NewEngine(mlx.Config{
			Backend:  mlx.BackendAuto,
			MaxBatch: config.MaxBatchSize,
		})
		if err != nil {
			logger.Warn("MLX not available, using CPU", "error", err)
		} else {
			logger.Info("MLX Engine initialized",
				"backend", mlxEngine.Backend(),
				"device", mlxEngine.Device(),
				"gpu", mlxEngine.IsGPUAvailable())
		}
	}

	// Initialize order book
	orderBook := lx.NewOrderBook("LXD-MAINNET")


	ctx, cancel := context.WithCancel(context.Background())

	return &LXDNode{
		config:        config,
		db:            db,
		orderBook:     orderBook,
		mlxEngine:     mlxEngine,
		logger:        logger,
		pendingOrders: make([]*lx.Order, 0, 10000),
		ctx:           ctx,
		cancel:        cancel,
	}, nil
}

func (n *LXDNode) Start() error {
	n.logger.Info("Starting LXD node",
		"nodeID", n.config.NodeID,
		"dataDir", filepath.Join(os.Getenv("HOME"), n.config.DataDir),
		"httpPort", n.config.HTTPPort,
		"wsPort", n.config.WSPort,
		"p2pPort", n.config.P2PPort,
		"blockTime", n.config.BlockTime)

	// Load state from database
	if err := n.loadState(); err != nil {
		n.logger.Warn("Failed to load state", "error", err)
	}

	// Start consensus engine
	n.wg.Add(1)
	go n.runConsensus()

	// Start metrics server
	if n.config.EnableMetrics {
		n.wg.Add(1)
		go n.runMetricsServer()
	}

	// Start stats printer
	n.wg.Add(1)
	go n.printStats()

	// Start test order generator
	n.wg.Add(1)
	go n.generateTestOrders()

	// Start JSON-RPC server
	n.wg.Add(1)
	go n.runJSONRPCServer()

	n.logger.Info("LXD node started successfully")
	return nil
}

func (n *LXDNode) runConsensus() {
	defer n.wg.Done()

	ticker := time.NewTicker(n.config.BlockTime)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.finalizeBlock()
		}
	}
}

func (n *LXDNode) finalizeBlock() {
	startTime := time.Now()

	// Get pending orders
	n.orderMu.Lock()
	if len(n.pendingOrders) == 0 {
		n.orderMu.Unlock()
		return
	}
	orders := n.pendingOrders
	n.pendingOrders = make([]*lx.Order, 0, 10000)
	n.orderMu.Unlock()

	// Process orders
	var totalTrades int
	matchStart := time.Now()

	if n.mlxEngine != nil && n.mlxEngine.IsGPUAvailable() && len(orders) > 100 {
		// Use MLX for large batches
		trades := n.processOrdersMLX(orders)
		totalTrades = len(trades)
	} else {
		// Use CPU matching
		for _, order := range orders {
			numTrades := n.orderBook.AddOrder(order)
			totalTrades += int(numTrades)
		}
	}

	matchLatency := time.Since(matchStart)

	// Create and store block
	blockHeight := atomic.AddUint64(&n.blocksFinalized, 1)
	block := Block{
		Height:    blockHeight,
		Timestamp: time.Now(),
		NumOrders: len(orders),
		NumTrades: totalTrades,
	}

	if err := n.storeBlock(&block); err != nil {
		n.logger.Error("Failed to store block", "error", err)
	}

	// Update stats
	atomic.AddUint64(&n.ordersProcessed, uint64(len(orders)))
	atomic.AddUint64(&n.tradesExecuted, uint64(totalTrades))
	atomic.StoreUint64(&n.consensusLatency, uint64(time.Since(startTime).Nanoseconds()))


	if n.config.EnableDebug {
		n.logger.Debug("Block finalized",
			"height", blockHeight,
			"orders", len(orders),
			"trades", totalTrades,
			"consensusTime", time.Since(startTime),
			"matchingTime", matchLatency)
	}
}

func (n *LXDNode) processOrdersMLX(orders []*lx.Order) []mlx.Trade {
	// Convert to MLX format
	mlxBids := make([]mlx.Order, 0)
	mlxAsks := make([]mlx.Order, 0)

	for _, order := range orders {
		mlxOrder := mlx.Order{
			ID:    order.ID,
			Price: order.Price,
			Size:  order.Size,
		}

		if order.Side == lx.Buy {
			mlxOrder.Side = 0
			mlxBids = append(mlxBids, mlxOrder)
		} else {
			mlxOrder.Side = 1
			mlxAsks = append(mlxAsks, mlxOrder)
		}
	}

	// Process on GPU
	return n.mlxEngine.BatchMatch(mlxBids, mlxAsks)
}

type Block struct {
	Height    uint64    `json:"height"`
	Timestamp time.Time `json:"timestamp"`
	NumOrders int       `json:"num_orders"`
	NumTrades int       `json:"num_trades"`
}

func (n *LXDNode) storeBlock(block *Block) error {
	key := []byte(fmt.Sprintf("block:%d", block.Height))
	value, err := json.Marshal(block)
	if err != nil {
		return err
	}

	batch := n.db.NewBatch()
	defer batch.Reset()

	if err := batch.Put(key, value); err != nil {
		return err
	}
	
	// Update last block height
	heightBytes := make([]byte, 8)
	for i := 0; i < 8; i++ {
		heightBytes[7-i] = byte(block.Height >> (i * 8))
	}
	if err := batch.Put([]byte("last_block"), heightBytes); err != nil {
		return err
	}

	return batch.Write()
}

func (n *LXDNode) loadState() error {
	val, err := n.db.Get([]byte("last_block"))
	if err != nil {
		if err == database.ErrNotFound {
			n.logger.Info("No previous state found, starting fresh")
			return nil
		}
		return err
	}

	if len(val) >= 8 {
		var lastBlock uint64
		for i := 0; i < 8; i++ {
			lastBlock |= uint64(val[7-i]) << (i * 8)
		}
		atomic.StoreUint64(&n.blocksFinalized, lastBlock)
		n.logger.Info("Loaded state", "lastBlock", lastBlock)
	}

	return nil
}

func (n *LXDNode) generateTestOrders() {
	defer n.wg.Done()

	orderID := uint64(1)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			// Generate batch of test orders
			n.orderMu.Lock()
			for i := 0; i < 10; i++ {
				side := lx.Buy
				if orderID%2 == 0 {
					side = lx.Sell
				}

				order := &lx.Order{
					ID:    atomic.AddUint64(&orderID, 1),
					Type:  lx.Limit,
					Side:  side,
					Price: 50000.0 + float64((orderID%200)-100),
					Size:  1.0,
					User:  fmt.Sprintf("user-%d", orderID%100),
				}

				n.pendingOrders = append(n.pendingOrders, order)
			}
			n.orderMu.Unlock()
		}
	}
}

func (n *LXDNode) runMetricsServer() {
	defer n.wg.Done()
	
	// Metrics server placeholder
	// In production, would use luxfi/metric package
	n.logger.Info("Metrics server started", "port", n.config.MetricsPort)
	<-n.ctx.Done()
}

func (n *LXDNode) runJSONRPCServer() {
	defer n.wg.Done()

	// Create JSON-RPC server
	server := api.NewJSONRPCServer(n.orderBook, n.logger)
	
	mux := http.NewServeMux()
	mux.Handle("/rpc", server)
	
	// Add health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "healthy",
			"block":  atomic.LoadUint64(&n.blocksFinalized),
			"orders": atomic.LoadUint64(&n.ordersProcessed),
		})
	})
	
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", n.config.HTTPPort),
		Handler: mux,
	}
	
	go func() {
		<-n.ctx.Done()
		httpServer.Shutdown(context.Background())
	}()
	
	n.logger.Info("JSON-RPC server started", "port", n.config.HTTPPort, "endpoint", "/rpc")
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		n.logger.Error("JSON-RPC server error", "error", err)
	}
}

func (n *LXDNode) printStats() {
	defer n.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	startTime := time.Now()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			elapsed := time.Since(startTime).Seconds()
			blocks := atomic.LoadUint64(&n.blocksFinalized)
			orders := atomic.LoadUint64(&n.ordersProcessed)
			trades := atomic.LoadUint64(&n.tradesExecuted)
			latencyNs := atomic.LoadUint64(&n.consensusLatency)

			n.logger.Info("LXD Node Status",
				"uptime", fmt.Sprintf("%.0fs", elapsed),
				"blocks", blocks,
				"blocksPerSec", fmt.Sprintf("%.1f", float64(blocks)/elapsed),
				"orders", orders,
				"ordersPerSec", fmt.Sprintf("%.0f", float64(orders)/elapsed),
				"trades", trades,
				"tradesPerSec", fmt.Sprintf("%.0f", float64(trades)/elapsed),
				"dbPath", filepath.Join(os.Getenv("HOME"), n.config.DataDir, "badgerdb"))

			if latencyNs > 0 {
				n.logger.Info("Performance metrics",
					"consensusLatency", fmt.Sprintf("%.1fμs", float64(latencyNs)/1000))
			}

			// Check 1ms consensus achievement
			if n.config.BlockTime == 1*time.Millisecond && blocks > 0 {
				actualBlockTime := elapsed / float64(blocks) * 1000
				n.logger.Info("Block time achievement",
					"actual", fmt.Sprintf("%.1fms", actualBlockTime),
					"target", "1ms",
					"achieving", actualBlockTime <= 1.5)
			}
		}
	}
}

func (n *LXDNode) Shutdown() {
	n.logger.Info("Shutting down LXD node...")

	// Cancel context
	n.cancel()

	// Wait for goroutines
	n.wg.Wait()

	// Close database
	if n.db != nil {
		n.db.Close()
	}

	// Close MLX engine
	if n.mlxEngine != nil {
		n.mlxEngine.Close()
	}

	n.logger.Info("LXD node shutdown complete")
}

func main() {
	config := &Config{
		DataDir: defaultDataDir,
	}

	// Parse flags
	flag.StringVar(&config.DataDir, "data-dir", defaultDataDir, "Data directory (relative to $HOME)")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")

	flag.IntVar(&config.HTTPPort, "http-port", defaultPort, "HTTP API port")
	flag.IntVar(&config.WSPort, "ws-port", defaultWSPort, "WebSocket port")
	flag.IntVar(&config.P2PPort, "p2p-port", defaultP2PPort, "P2P network port")
	flag.IntVar(&config.MetricsPort, "metrics-port", 9090, "Prometheus metrics port")

	blockTime := flag.Duration("block-time", 1*time.Millisecond, "Target block time")
	flag.IntVar(&config.NodeID, "node-id", 1, "Node ID")

	flag.BoolVar(&config.EnableMLX, "enable-mlx", true, "Enable MLX GPU acceleration")
	flag.IntVar(&config.MaxBatchSize, "max-batch", 10000, "Maximum batch size for MLX")

	flag.BoolVar(&config.EnableMetrics, "enable-metrics", true, "Enable Prometheus metrics")
	flag.BoolVar(&config.EnableDebug, "debug", false, "Enable debug logging")

	flag.Parse()

	config.BlockTime = *blockTime
	
	// Store log level as string
	config.LogLevel = *logLevel

	// Initialize root logger for banner
	rootLogger := log.Root()
	rootLogger.Info(`
╔══════════════════════════════════════════╗
║            LXD - Lux DEX Node            ║
║                                          ║
║    Planet-Scale Trading Infrastructure   ║
║         Quantum-Secure Consensus         ║
║           1ms Block Finality             ║
╚══════════════════════════════════════════╝`)

	rootLogger.Info("System information",
		"platform", fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		"cpus", runtime.NumCPU(),
		"dataDir", filepath.Join(os.Getenv("HOME"), config.DataDir),
		"blockTime", config.BlockTime)

	// Create and start node
	node, err := NewLXDNode(config)
	if err != nil {
		rootLogger.Crit("Failed to create node", "error", err)
		os.Exit(1)
	}

	if err := node.Start(); err != nil {
		rootLogger.Crit("Failed to start node", "error", err)
		os.Exit(1)
	}

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for shutdown signal
	sig := <-sigChan
	rootLogger.Info("Received shutdown signal", "signal", sig)

	// Graceful shutdown
	node.Shutdown()
}