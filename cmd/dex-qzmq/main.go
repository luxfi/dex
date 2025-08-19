// DEX node with QZMQ post-quantum security
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/qzmq"
)

type NodeConfig struct {
	ID       int
	Port     int
	IsLeader bool
	Mode     string // "hybrid", "pq-only", "classical"
}

type DEXNode struct {
	config    NodeConfig
	transport qzmq.Transport
	orderBook *lx.OrderBook
	pubSocket qzmq.Socket
	subSocket qzmq.Socket
}

func NewDEXNode(config NodeConfig) (*DEXNode, error) {
	// Configure QZMQ options based on mode
	var opts qzmq.Options
	switch config.Mode {
	case "pq-only":
		opts = qzmq.ConservativeOptions()
	case "classical":
		opts = qzmq.PerformanceOptions()
	default:
		opts = qzmq.DefaultOptions()
	}
	
	// Create QZMQ transport
	transport, err := qzmq.New(opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create QZMQ transport: %w", err)
	}
	
	node := &DEXNode{
		config:    config,
		transport: transport,
		orderBook: lx.NewOrderBook("BTC-USD"),
	}
	
	node.orderBook.EnableImmediateMatching = true
	
	return node, nil
}

func (n *DEXNode) Start() error {
	// Create publisher socket
	pub, err := n.transport.NewSocket(qzmq.PUB)
	if err != nil {
		return err
	}
	n.pubSocket = pub
	
	// Bind publisher
	pubAddr := fmt.Sprintf("tcp://*:%d", n.config.Port)
	if err := pub.Bind(pubAddr); err != nil {
		return err
	}
	
	log.Printf("Node %d: Publisher bound to %s (QZMQ %s mode)", 
		n.config.ID, pubAddr, n.config.Mode)
	
	// Create subscriber socket
	sub, err := n.transport.NewSocket(qzmq.SUB)
	if err != nil {
		return err
	}
	n.subSocket = sub
	
	// Subscribe to all messages
	sub.Subscribe("")
	
	// Connect to other nodes
	if !n.config.IsLeader {
		leaderPort := 5000
		leaderAddr := fmt.Sprintf("tcp://localhost:%d", leaderPort)
		if err := sub.Connect(leaderAddr); err != nil {
			return err
		}
		log.Printf("Node %d: Connected to leader at %s", n.config.ID, leaderAddr)
	}
	
	// Start processing
	go n.processMessages()
	
	// If leader, generate orders
	if n.config.IsLeader {
		go n.generateOrders()
	}
	
	// Start metrics reporter
	go n.reportMetrics()
	
	return nil
}

func (n *DEXNode) processMessages() {
	for {
		// Receive encrypted message
		data, err := n.subSocket.Recv()
		if err != nil {
			log.Printf("Node %d: Receive error: %v", n.config.ID, err)
			continue
		}
		
		// Parse order
		var order lx.Order
		if err := json.Unmarshal(data, &order); err != nil {
			continue
		}
		
		// Process order
		trades := n.orderBook.AddOrder(&order)
		if trades > 0 {
			log.Printf("Node %d: Matched %d trades (QZMQ encrypted)", 
				n.config.ID, trades)
		}
	}
}

func (n *DEXNode) generateOrders() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	orderID := uint64(n.config.ID * 1000000)
	
	for range ticker.C {
		orderID++
		
		// Create order
		order := &lx.Order{
			ID:        orderID,
			Symbol:    "BTC-USD",
			Type:      lx.Limit,
			Side:      lx.Side(orderID % 2),
			Price:     50000 + float64((orderID%100)-50)*10,
			Size:      0.1 + float64(orderID%10)*0.01,
			Timestamp: time.Now(),
			UserID:    fmt.Sprintf("user-%d", n.config.ID),
		}
		
		// Add locally
		n.orderBook.AddOrder(order)
		
		// Broadcast encrypted
		data, _ := json.Marshal(order)
		if err := n.pubSocket.Send(data); err != nil {
			log.Printf("Node %d: Send error: %v", n.config.ID, err)
		}
	}
}

func (n *DEXNode) reportMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// Get socket metrics
		pubMetrics := n.pubSocket.GetMetrics()
		subMetrics := n.subSocket.GetMetrics()
		
		log.Printf("Node %d Metrics:", n.config.ID)
		log.Printf("  Publisher: Sent %d msgs (%d bytes)", 
			pubMetrics.MessagesSent, pubMetrics.BytesSent)
		log.Printf("  Subscriber: Recv %d msgs (%d bytes)", 
			subMetrics.MessagesReceived, subMetrics.BytesReceived)
		
		// Get transport stats
		stats := n.transport.Stats()
		log.Printf("  Handshakes: %d, Encrypted: %d, Decrypted: %d, Key Rotations: %d",
			stats.HandshakesCompleted, stats.MessagesEncrypted, 
			stats.MessagesDecrypted, stats.KeyRotations)
		
		// Report order book state
		bestBid := n.orderBook.GetBestBid()
		bestAsk := n.orderBook.GetBestAsk()
		spread := bestAsk - bestBid
		
		log.Printf("  OrderBook: Bid %.2f, Ask %.2f, Spread %.2f",
			bestBid, bestAsk, spread)
	}
}

func (n *DEXNode) Close() error {
	if n.pubSocket != nil {
		n.pubSocket.Close()
	}
	if n.subSocket != nil {
		n.subSocket.Close()
	}
	if n.transport != nil {
		n.transport.Close()
	}
	return nil
}

func main() {
	var (
		nodeID   = flag.Int("id", 0, "Node ID")
		port     = flag.Int("port", 5000, "Base port")
		leader   = flag.Bool("leader", false, "Is leader node")
		mode     = flag.String("mode", "hybrid", "Security mode: hybrid, pq-only, classical")
	)
	flag.Parse()
	
	config := NodeConfig{
		ID:       *nodeID,
		Port:     *port + *nodeID,
		IsLeader: *leader,
		Mode:     *mode,
	}
	
	// Create and start node
	node, err := NewDEXNode(config)
	if err != nil {
		log.Fatal(err)
	}
	defer node.Close()
	
	log.Printf("Starting DEX Node %d with QZMQ %s mode (Leader: %v)",
		config.ID, config.Mode, config.IsLeader)
	
	if err := node.Start(); err != nil {
		log.Fatal(err)
	}
	
	// Display QZMQ features
	log.Println("QZMQ Security Features Active:")
	switch config.Mode {
	case "pq-only":
		log.Println("  ✓ ML-KEM-1024 (post-quantum KEM)")
		log.Println("  ✓ ML-DSA-3 (post-quantum signatures)")
		log.Println("  ✓ Maximum quantum resistance")
	case "hybrid":
		log.Println("  ✓ X25519 + ML-KEM-768 (hybrid KEM)")
		log.Println("  ✓ ML-DSA-2 (post-quantum signatures)")
		log.Println("  ✓ Transitional security")
	case "classical":
		log.Println("  ✓ X25519 (classical ECDH)")
		log.Println("  ✓ Ed25519 (classical signatures)")
		log.Println("  ✓ High performance mode")
	}
	log.Println("  ✓ AES-256-GCM authenticated encryption")
	log.Println("  ✓ Automatic key rotation")
	log.Println("  ✓ Perfect forward secrecy")
	
	// Keep running
	select {}
}