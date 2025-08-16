package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/luxfi/dex/backend/pkg/consensus"
	"github.com/luxfi/dex/backend/pkg/lx"
	zmq "github.com/pebbe/zmq4"
)

// NodeConfig represents configuration for a DAG node
type NodeConfig struct {
	ID          string
	HTTPPort    int
	ZMQPubPort  int
	ZMQSubPort  int
	ZMQRepPort  int
	Peers       []string
	IsLeader    bool
}

// DAGNode represents a node in the DAG network
type DAGNode struct {
	config    NodeConfig
	dagBook   *consensus.FPCDAGOrderBook
	zmqCtx    *zmq.Context
	pubSocket *zmq.Socket
	subSocket *zmq.Socket
	repSocket *zmq.Socket
	httpServer *http.Server
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
}

// NewDAGNode creates a new DAG node
func NewDAGNode(config NodeConfig) (*DAGNode, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	// Create FPC DAG order book with quantum finality
	dagBook, err := consensus.NewFPCDAGOrderBook(config.ID, "BTC-USD")
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create FPC DAG order book: %w", err)
	}
	
	// Create ZMQ context
	zmqCtx, err := zmq.NewContext()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create ZMQ context: %w", err)
	}
	
	node := &DAGNode{
		config:  config,
		dagBook: dagBook,
		zmqCtx:  zmqCtx,
		ctx:     ctx,
		cancel:  cancel,
	}
	
	// Initialize ZMQ sockets
	if err := node.initZMQSockets(); err != nil {
		cancel()
		return nil, err
	}
	
	// Initialize HTTP server
	node.initHTTPServer()
	
	return node, nil
}

// initZMQSockets initializes ZeroMQ sockets
func (n *DAGNode) initZMQSockets() error {
	var err error
	
	// Publisher socket for broadcasting vertices
	n.pubSocket, err = n.zmqCtx.NewSocket(zmq.PUB)
	if err != nil {
		return fmt.Errorf("failed to create PUB socket: %w", err)
	}
	if err := n.pubSocket.Bind(fmt.Sprintf("tcp://*:%d", n.config.ZMQPubPort)); err != nil {
		return fmt.Errorf("failed to bind PUB socket: %w", err)
	}
	
	// Subscriber socket for receiving vertices
	n.subSocket, err = n.zmqCtx.NewSocket(zmq.SUB)
	if err != nil {
		return fmt.Errorf("failed to create SUB socket: %w", err)
	}
	n.subSocket.SetSubscribe("")
	
	// Connect to peer publishers
	for _, peer := range n.config.Peers {
		if err := n.subSocket.Connect(peer); err != nil {
			log.Printf("Failed to connect to peer %s: %v", peer, err)
		}
	}
	
	// Reply socket for handling requests
	n.repSocket, err = n.zmqCtx.NewSocket(zmq.REP)
	if err != nil {
		return fmt.Errorf("failed to create REP socket: %w", err)
	}
	if err := n.repSocket.Bind(fmt.Sprintf("tcp://*:%d", n.config.ZMQRepPort)); err != nil {
		return fmt.Errorf("failed to bind REP socket: %w", err)
	}
	
	return nil
}

// initHTTPServer initializes the HTTP API server
func (n *DAGNode) initHTTPServer() {
	mux := http.NewServeMux()
	
	// Health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status": "healthy",
			"node":   n.config.ID,
		})
	})
	
	// Submit order
	mux.HandleFunc("/order", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		
		var order lx.Order
		if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		// Add order to DAG
		vertex, err := n.dagBook.AddOrder(&order)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		// Broadcast vertex to network
		n.broadcastVertex(vertex)
		
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"vertex_id": vertex.ID.String(),
			"order_id":  order.ID,
			"trades":    vertex.Trades,
		})
	})
	
	// Get DAG stats
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		stats := n.dagBook.GetStats()
		json.NewEncoder(w).Encode(stats)
	})
	
	// Get DAG visualization data
	mux.HandleFunc("/dag", func(w http.ResponseWriter, r *http.Request) {
		// Return DAG structure for visualization
		dagData := n.getDagVisualizationData()
		json.NewEncoder(w).Encode(dagData)
	})
	
	n.httpServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", n.config.HTTPPort),
		Handler: mux,
	}
}

// Start starts the DAG node
func (n *DAGNode) Start() error {
	log.Printf("Starting DAG node %s", n.config.ID)
	log.Printf("  HTTP API: http://localhost:%d", n.config.HTTPPort)
	log.Printf("  ZMQ PUB: tcp://localhost:%d", n.config.ZMQPubPort)
	log.Printf("  ZMQ REP: tcp://localhost:%d", n.config.ZMQRepPort)
	
	// Start FPC consensus with quantum finality
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		if err := n.dagBook.RunFPCConsensus(); err != nil {
			log.Printf("FPC Consensus error: %v", err)
		}
	}()
	
	// Start vertex receiver
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		n.receiveVertices()
	}()
	
	// Start request handler
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		n.handleRequests()
	}()
	
	// Start HTTP server
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		if err := n.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()
	
	// If leader, start generating test orders
	if n.config.IsLeader {
		n.wg.Add(1)
		go func() {
			defer n.wg.Done()
			n.generateTestOrders()
		}()
	}
	
	return nil
}

// receiveVertices receives vertices from peers
func (n *DAGNode) receiveVertices() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			// Receive vertex from peers
			msg, err := n.subSocket.RecvBytes(zmq.DONTWAIT)
			if err != nil {
				time.Sleep(10 * time.Millisecond)
				continue
			}
			
			// Unmarshal message with quantum certificate
			var msgData struct {
				Vertex      *consensus.OrderVertex
				Certificate *consensus.QuantumCertificate
			}
			if err := json.Unmarshal(msg, &msgData); err != nil {
				// Try legacy format without certificate
				var vertex consensus.OrderVertex
				if err := json.Unmarshal(msg, &vertex); err != nil {
					log.Printf("Failed to unmarshal vertex: %v", err)
					continue
				}
				// Process without quantum certificate (legacy)
				if err := n.dagBook.ProcessRemoteVertex(&vertex, nil); err != nil {
					log.Printf("Failed to process vertex: %v", err)
				}
			} else {
				// Process with quantum certificate
				if err := n.dagBook.ProcessRemoteVertex(msgData.Vertex, msgData.Certificate); err != nil {
					log.Printf("Failed to process vertex with quantum cert: %v", err)
				}
			}
		}
	}
}

// handleRequests handles ZMQ requests
func (n *DAGNode) handleRequests() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			// Handle request
			msg, err := n.repSocket.RecvBytes(0)
			if err != nil {
				continue
			}
			
			// Simple echo for now
			n.repSocket.SendBytes(msg, 0)
		}
	}
}

// broadcastVertex broadcasts a vertex to all peers
func (n *DAGNode) broadcastVertex(vertex *consensus.OrderVertex) {
	data, err := json.Marshal(vertex)
	if err != nil {
		log.Printf("Failed to marshal vertex: %v", err)
		return
	}
	
	// Broadcast via ZMQ PUB socket
	if _, err := n.pubSocket.SendBytes(data, zmq.DONTWAIT); err != nil {
		log.Printf("Failed to broadcast vertex: %v", err)
	}
}

// generateTestOrders generates test orders (leader only)
func (n *DAGNode) generateTestOrders() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	orderID := uint64(0)
	
	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			// Generate a few test orders
			for i := 0; i < 5; i++ {
				orderID++
				
				side := lx.Buy
				if r.Float32() > 0.5 {
					side = lx.Sell
				}
				
				order := &lx.Order{
					ID:     orderID,
					Symbol: "BTC-USD",
					Side:   side,
					Type:   lx.Limit,
					Price:  50000 + r.Float64()*1000 - 500,
					Size:   r.Float64() * 2,
					User:   fmt.Sprintf("user-%d", r.Intn(10)),
				}
				
				vertex, err := n.dagBook.AddOrder(order)
				if err != nil {
					log.Printf("Failed to add order: %v", err)
					continue
				}
				
				n.broadcastVertex(vertex)
				
				if len(vertex.Trades) > 0 {
					log.Printf("Node %s: Order %d matched, %d trades executed",
						n.config.ID, order.ID, len(vertex.Trades))
				}
			}
		}
	}
}

// getDagVisualizationData returns DAG structure for visualization
func (n *DAGNode) getDagVisualizationData() map[string]interface{} {
	stats := n.dagBook.GetStats()
	
	// Add visualization-specific data
	return map[string]interface{}{
		"node_id":    n.config.ID,
		"stats":      stats,
		"timestamp":  time.Now().Unix(),
		// In production, would include vertex graph structure
	}
}

// Stop stops the DAG node
func (n *DAGNode) Stop() {
	log.Printf("Stopping DAG node %s", n.config.ID)
	
	n.cancel()
	n.httpServer.Close()
	
	if n.pubSocket != nil {
		n.pubSocket.Close()
	}
	if n.subSocket != nil {
		n.subSocket.Close()
	}
	if n.repSocket != nil {
		n.repSocket.Close()
	}
	
	n.wg.Wait()
	n.zmqCtx.Term()
	
	log.Printf("DAG node %s stopped", n.config.ID)
}

func main() {
	var (
		nodeID     = flag.String("node", "node0", "Node ID")
		httpPort   = flag.Int("http", 8080, "HTTP API port")
		zmqPubPort = flag.Int("pub", 5000, "ZMQ PUB port")
		zmqSubPort = flag.Int("sub", 5001, "ZMQ SUB port")
		zmqRepPort = flag.Int("rep", 5002, "ZMQ REP port")
		peers      = flag.String("peers", "", "Comma-separated peer PUB addresses")
		isLeader   = flag.Bool("leader", false, "Is this the leader node")
	)
	flag.Parse()
	
	// Parse peers
	peerList := []string{}
	if *peers != "" {
		for _, peer := range []string{*peers} {
			if peer != "" {
				peerList = append(peerList, peer)
			}
		}
	}
	
	// Create node configuration
	config := NodeConfig{
		ID:         *nodeID,
		HTTPPort:   *httpPort,
		ZMQPubPort: *zmqPubPort,
		ZMQSubPort: *zmqSubPort,
		ZMQRepPort: *zmqRepPort,
		Peers:      peerList,
		IsLeader:   *isLeader,
	}
	
	// Create and start node
	node, err := NewDAGNode(config)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}
	
	if err := node.Start(); err != nil {
		log.Fatalf("Failed to start node: %v", err)
	}
	
	// Wait for interrupt
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan
	
	// Shutdown
	node.Stop()
}