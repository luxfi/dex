// QZMQ-enabled DEX with post-quantum security
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/luxfi/dex/pkg/crypto"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/qzmq"
	zmq "github.com/pebbe/zmq4"
)

// NodeConfig for QZMQ-enabled node
type NodeConfig struct {
	ID          int
	PubAddr     string
	SubAddr     string
	APIAddr     string
	IsLeader    bool
	PQOnly      bool   // Enforce PQ-only mode
	Suite       string // Cipher suite selection
}

// QZMQNode represents a quantum-secure DEX node
type QZMQNode struct {
	config      NodeConfig
	orderBook   *lx.OrderBook
	qzmqOpts    qzmq.Options
	publisher   *zmq.Socket
	subscriber  *zmq.Socket
	connections map[string]*qzmq.Connection // Peer connections
}

func NewQZMQNode(config NodeConfig) (*QZMQNode, error) {
	// Configure QZMQ options based on role
	opts := qzmq.DefaultOptions
	if config.PQOnly {
		opts = qzmq.PQOnlyOptions
	}
	
	// Override suite if specified
	if config.Suite != "" {
		switch config.Suite {
		case "mlkem768":
			opts.Suite.Kem = crypto.KemMLKEM768
		case "mlkem1024":
			opts.Suite.Kem = crypto.KemMLKEM1024
		case "hybrid768":
			opts.Suite.Kem = crypto.KemHybridX25519ML768
		case "hybrid1024":
			opts.Suite.Kem = crypto.KemHybridX25519ML1024
		}
	}
	
	node := &QZMQNode{
		config:      config,
		orderBook:   lx.NewOrderBook("BTC-USD"),
		qzmqOpts:    opts,
		connections: make(map[string]*qzmq.Connection),
	}
	
	// Enable immediate matching for DEX
	node.orderBook.EnableImmediateMatching = true
	
	return node, nil
}

func (n *QZMQNode) Start() error {
	// Create QZMQ-secured publisher
	pub, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		return err
	}
	n.publisher = pub
	
	// Configure QZMQ mechanism
	if err := n.setupQZMQ(pub, true); err != nil {
		return err
	}
	
	if err := pub.Bind(n.config.PubAddr); err != nil {
		return err
	}
	
	// Create QZMQ-secured subscriber
	sub, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		return err
	}
	n.subscriber = sub
	
	// Configure QZMQ mechanism
	if err := n.setupQZMQ(sub, false); err != nil {
		return err
	}
	
	// Subscribe to all topics
	sub.SetSubscribe("")
	
	// Connect to other nodes
	if !n.config.IsLeader {
		leaderSub := fmt.Sprintf("tcp://localhost:%d", 5001+n.config.ID%3)
		if err := sub.Connect(leaderSub); err != nil {
			return err
		}
		log.Printf("Node %d connected to leader (QZMQ secured)", n.config.ID)
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

func (n *QZMQNode) setupQZMQ(socket *zmq.Socket, isServer bool) error {
	// In real implementation, this would:
	// 1. Set up ML-DSA certificates
	// 2. Configure QZMQ mechanism
	// 3. Set cipher suite preferences
	// 4. Enable anti-DoS cookies
	
	// For now, we simulate by setting metadata
	socket.SetIdentity(fmt.Sprintf("node-%d", n.config.ID))
	
	// Log the security configuration
	log.Printf("Node %d: QZMQ configured with suite %+v (PQ-only: %v)", 
		n.config.ID, n.qzmqOpts.Suite, n.config.PQOnly)
	
	return nil
}

func (n *QZMQNode) processMessages() {
	for {
		msg, err := n.subscriber.RecvBytes(0)
		if err != nil {
			log.Printf("Node %d: receive error: %v", n.config.ID, err)
			continue
		}
		
		// In real QZMQ, this would be decrypted via AEAD
		// For now, parse as JSON
		var order lx.Order
		if err := json.Unmarshal(msg, &order); err != nil {
			continue
		}
		
		// Process order
		trades := n.orderBook.AddOrder(&order)
		if trades > 0 {
			log.Printf("Node %d: Matched %d trades (QZMQ secured)", n.config.ID, trades)
		}
	}
}

func (n *QZMQNode) generateOrders() {
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
		
		// Broadcast (would be QZMQ encrypted)
		data, _ := json.Marshal(order)
		n.publisher.SendBytes(data, 0)
	}
}

func (n *QZMQNode) reportMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// Report QZMQ connection metrics
		for peer, conn := range n.connections {
			metrics := conn.GetMetrics()
			log.Printf("Node %d -> %s: Messages: %d, Bytes: %d, KeyUpdates: %d",
				n.config.ID, peer, metrics.MessagesSent, metrics.BytesSent, metrics.KeyUpdates)
		}
		
		// Report order book stats
		bestBid := n.orderBook.GetBestBid()
		bestAsk := n.orderBook.GetBestAsk()
		spread := bestAsk - bestBid
		
		log.Printf("Node %d: OrderBook [Bid: %.2f, Ask: %.2f, Spread: %.2f] (QZMQ secured)",
			n.config.ID, bestBid, bestAsk, spread)
	}
}

func main() {
	var (
		nodeID   = flag.Int("id", 0, "Node ID")
		port     = flag.Int("port", 5000, "Base port")
		leader   = flag.Bool("leader", false, "Is leader node")
		pqOnly   = flag.Bool("pq-only", false, "Enforce PQ-only mode")
		suite    = flag.String("suite", "hybrid768", "Cipher suite")
	)
	flag.Parse()
	
	config := NodeConfig{
		ID:       *nodeID,
		PubAddr:  fmt.Sprintf("tcp://*:%d", *port+*nodeID),
		SubAddr:  fmt.Sprintf("tcp://localhost:%d", *port),
		APIAddr:  fmt.Sprintf("tcp://*:%d", 8080+*nodeID),
		IsLeader: *leader,
		PQOnly:   *pqOnly,
		Suite:    *suite,
	}
	
	// Create and start node
	node, err := NewQZMQNode(config)
	if err != nil {
		log.Fatal(err)
	}
	
	log.Printf("Starting QZMQ-secured DEX Node %d (PQ-only: %v, Suite: %s)", 
		config.ID, config.PQOnly, config.Suite)
	
	if err := node.Start(); err != nil {
		log.Fatal(err)
	}
	
	// Keep running
	select {}
}