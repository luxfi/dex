//go:build zmqtest

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/luxfi/czmq/v4"
	"github.com/luxfi/dex/pkg/lx"
)

// NodeConfig represents configuration for a single node
type NodeConfig struct {
	NodeID     string
	PubPort    int      // Port for publishing events
	SubPort    int      // Port for subscribing to events
	RouterPort int      // Port for ROUTER socket (receiving orders)
	DealerPort int      // Port for DEALER socket (sending orders)
	IsLeader   bool     // Whether this node is the leader
	Peers      []string // List of peer addresses
}

// MultiNode represents a single node in the multi-node setup
type MultiNode struct {
	config       *NodeConfig
	orderBook    *lx.OrderBook
	pubSocket    *czmq.Sock
	subSocket    *czmq.Sock
	routerSocket *czmq.Sock
	dealerSocket *czmq.Sock

	// Metrics
	ordersProcessed  uint64
	tradesExecuted   uint64
	messagesReceived uint64
	messagesSent     uint64

	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// Message types for inter-node communication
type MessageType string

const (
	OrderSubmit   MessageType = "ORDER_SUBMIT"
	OrderCancel   MessageType = "ORDER_CANCEL"
	TradeExecuted MessageType = "TRADE_EXECUTED"
	BookUpdate    MessageType = "BOOK_UPDATE"
	Heartbeat     MessageType = "HEARTBEAT"
	StateSync     MessageType = "STATE_SYNC"
)

// NodeMessage represents a message between nodes
type NodeMessage struct {
	Type      MessageType `json:"type"`
	NodeID    string      `json:"node_id"`
	Timestamp time.Time   `json:"timestamp"`
	Payload   interface{} `json:"payload"`
	Sequence  uint64      `json:"sequence"`
}

// NewMultiNode creates a new multi-node instance
func NewMultiNode(config *NodeConfig) (*MultiNode, error) {
	ctx, cancel := context.WithCancel(context.Background())

	node := &MultiNode{
		config:    config,
		orderBook: lx.NewOrderBook("BTC-USD"),
		ctx:       ctx,
		cancel:    cancel,
	}

	// Initialize ZMQ sockets

	// Publisher socket for broadcasting events
	node.pubSocket = czmq.NewSock(czmq.Pub)
	if _, err := node.pubSocket.Bind(fmt.Sprintf("tcp://*:%d", config.PubPort)); err != nil {
		return nil, fmt.Errorf("failed to bind PUB socket: %w", err)
	}

	// Subscriber socket for receiving events from peers
	node.subSocket = czmq.NewSock(czmq.Sub)
	node.subSocket.SetSubscribe("")

	// Connect to peer publishers
	for _, peer := range config.Peers {
		if err := node.subSocket.Connect(peer); err != nil {
			log.Printf("Failed to connect to peer %s: %v", peer, err)
		}
	}

	// Router socket for receiving orders (server mode)
	node.routerSocket = czmq.NewSock(czmq.Router)
	if _, err := node.routerSocket.Bind(fmt.Sprintf("tcp://*:%d", config.RouterPort)); err != nil {
		return nil, fmt.Errorf("failed to bind ROUTER socket: %w", err)
	}

	// Dealer socket for load balancing (client mode)
	node.dealerSocket = czmq.NewSock(czmq.Dealer)
	node.dealerSocket.SetIdentity(config.NodeID)

	return node, nil
}

// Start begins the node's operation
func (n *MultiNode) Start() {
	log.Printf("Node %s starting (Leader: %v)...", n.config.NodeID, n.config.IsLeader)

	// Start goroutines for different tasks
	go n.handleSubscriptions()
	go n.handleRouterRequests()
	go n.heartbeatLoop()
	go n.metricsLoop()

	if n.config.IsLeader {
		go n.leaderLoop()
	}

	// Wait for context cancellation
	<-n.ctx.Done()
	n.cleanup()
}

// handleSubscriptions processes messages from peer nodes
func (n *MultiNode) handleSubscriptions() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			msg, _, err := n.subSocket.RecvFrame()
			if err != nil {
				time.Sleep(10 * time.Millisecond)
				continue
			}

			atomic.AddUint64(&n.messagesReceived, 1)
			n.processMessage(msg)
		}
	}
}

// handleRouterRequests handles incoming order requests
func (n *MultiNode) handleRouterRequests() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			// Receive multi-part message [identity, empty, data]
			identity, _, err := n.routerSocket.RecvFrame()
			if err != nil {
				log.Printf("Router receive error: %v", err)
				continue
			}

			// Empty delimiter
			_, _, err = n.routerSocket.RecvFrame()
			if err != nil {
				continue
			}

			// Actual message
			data, _, err := n.routerSocket.RecvFrame()
			if err != nil {
				continue
			}

			// Process order and send response
			response := n.processOrderRequest(data)

			// Send response back [identity, empty, response]
			n.routerSocket.SendFrame(identity, czmq.FlagMore)
			n.routerSocket.SendFrame([]byte(""), czmq.FlagMore)
			n.routerSocket.SendFrame(response, 0)

			atomic.AddUint64(&n.ordersProcessed, 1)
		}
	}
}

// processMessage handles messages from peer nodes
func (n *MultiNode) processMessage(data []byte) {
	var msg NodeMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		log.Printf("Failed to unmarshal message: %v", err)
		return
	}

	// Skip messages from self
	if msg.NodeID == n.config.NodeID {
		return
	}

	switch msg.Type {
	case TradeExecuted:
		// Update local order book based on executed trade
		n.handleTradeUpdate(msg)
	case BookUpdate:
		// Sync order book state
		n.handleBookUpdate(msg)
	case StateSync:
		// Full state synchronization
		n.handleStateSync(msg)
	case Heartbeat:
		// Track peer liveness
		n.handleHeartbeat(msg)
	}
}

// processOrderRequest processes incoming order requests
func (n *MultiNode) processOrderRequest(data []byte) []byte {
	var order lx.Order
	if err := json.Unmarshal(data, &order); err != nil {
		return []byte(`{"error": "invalid order format"}`)
	}

	// Add order to local book
	n.mu.Lock()
	orderID := n.orderBook.AddOrder(&order)
	trades := n.orderBook.MatchOrders()
	n.mu.Unlock()

	// Broadcast trades to peers
	if len(trades) > 0 {
		atomic.AddUint64(&n.tradesExecuted, uint64(len(trades)))
		n.broadcastTrades(trades)
	}

	// Return response
	response := map[string]interface{}{
		"order_id": orderID,
		"status":   "accepted",
		"trades":   trades,
	}

	respData, _ := json.Marshal(response)
	return respData
}

// broadcastTrades broadcasts executed trades to all peers
func (n *MultiNode) broadcastTrades(trades []lx.Trade) {
	msg := NodeMessage{
		Type:      TradeExecuted,
		NodeID:    n.config.NodeID,
		Timestamp: time.Now(),
		Payload:   trades,
		Sequence:  atomic.LoadUint64(&n.tradesExecuted),
	}

	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("Failed to marshal trades: %v", err)
		return
	}

	n.pubSocket.SendFrame(data, 0)
	atomic.AddUint64(&n.messagesSent, 1)
}

// heartbeatLoop sends periodic heartbeats to peers
func (n *MultiNode) heartbeatLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			msg := NodeMessage{
				Type:      Heartbeat,
				NodeID:    n.config.NodeID,
				Timestamp: time.Now(),
				Payload: map[string]interface{}{
					"orders_processed": atomic.LoadUint64(&n.ordersProcessed),
					"trades_executed":  atomic.LoadUint64(&n.tradesExecuted),
				},
			}

			data, _ := json.Marshal(msg)
			n.pubSocket.SendFrame(data, 0)
		}
	}
}

// leaderLoop performs leader-specific tasks
func (n *MultiNode) leaderLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			// Broadcast full state for synchronization
			n.broadcastStateSync()
		}
	}
}

// broadcastStateSync broadcasts full order book state
func (n *MultiNode) broadcastStateSync() {
	n.mu.RLock()
	snapshot := n.orderBook.GetSnapshot()
	n.mu.RUnlock()

	msg := NodeMessage{
		Type:      StateSync,
		NodeID:    n.config.NodeID,
		Timestamp: time.Now(),
		Payload:   snapshot,
	}

	data, _ := json.Marshal(msg)
	n.pubSocket.SendFrame(data, 0)
	atomic.AddUint64(&n.messagesSent, 1)
}

// metricsLoop reports metrics periodically
func (n *MultiNode) metricsLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			log.Printf("Node %s metrics - Orders: %d, Trades: %d, Messages In: %d, Out: %d",
				n.config.NodeID,
				atomic.LoadUint64(&n.ordersProcessed),
				atomic.LoadUint64(&n.tradesExecuted),
				atomic.LoadUint64(&n.messagesReceived),
				atomic.LoadUint64(&n.messagesSent))
		}
	}
}

// Stub handlers for message types
func (n *MultiNode) handleTradeUpdate(msg NodeMessage) {
	// Update local order book based on trades from other nodes
	log.Printf("Node %s received trade update from %s", n.config.NodeID, msg.NodeID)
}

func (n *MultiNode) handleBookUpdate(msg NodeMessage) {
	// Update order book based on incremental updates
	log.Printf("Node %s received book update from %s", n.config.NodeID, msg.NodeID)
}

func (n *MultiNode) handleStateSync(msg NodeMessage) {
	// Synchronize full state from leader
	if !n.config.IsLeader {
		log.Printf("Node %s syncing state from leader %s", n.config.NodeID, msg.NodeID)
	}
}

func (n *MultiNode) handleHeartbeat(msg NodeMessage) {
	// Track peer health
	log.Printf("Node %s received heartbeat from %s", n.config.NodeID, msg.NodeID)
}

// cleanup closes all sockets
func (n *MultiNode) cleanup() {
	log.Printf("Node %s shutting down...", n.config.NodeID)

	if n.pubSocket != nil {
		n.pubSocket.Destroy()
	}
	if n.subSocket != nil {
		n.subSocket.Destroy()
	}
	if n.routerSocket != nil {
		n.routerSocket.Destroy()
	}
	if n.dealerSocket != nil {
		n.dealerSocket.Destroy()
	}
}

func main() {
	var (
		nodeID   = flag.String("node", "node1", "Node identifier")
		basePort = flag.Int("port", 5000, "Base port number")
		leader   = flag.Bool("leader", false, "Whether this node is the leader")
		peers    = flag.String("peers", "", "Comma-separated list of peer addresses")
	)
	flag.Parse()

	// Calculate ports based on base port
	config := &NodeConfig{
		NodeID:     *nodeID,
		PubPort:    *basePort,
		SubPort:    *basePort + 1,
		RouterPort: *basePort + 2,
		DealerPort: *basePort + 3,
		IsLeader:   *leader,
		Peers:      []string{},
	}

	// Parse peer addresses
	if *peers != "" {
		for _, peer := range splitAndTrim(*peers, ",") {
			config.Peers = append(config.Peers, peer)
		}
	}

	// Create and start node
	node, err := NewMultiNode(config)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		node.cancel()
	}()

	node.Start()
}

func splitAndTrim(s string, sep string) []string {
	parts := []string{}
	for _, p := range strings.Split(s, sep) {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	return parts
}
