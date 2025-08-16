package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/backend/pkg/lx"
	zmq "github.com/pebbe/zmq4"
)

// TestClient represents a client that sends orders to the multi-node cluster
type TestClient struct {
	dealers      []*zmq.Socket
	nodeAddrs    []string
	ordersent    uint64
	responsesRcv uint64
	mu           sync.RWMutex
}

// NewTestClient creates a new test client
func NewTestClient(nodeAddrs []string) (*TestClient, error) {
	client := &TestClient{
		nodeAddrs: nodeAddrs,
		dealers:   make([]*zmq.Socket, len(nodeAddrs)),
	}
	
	// Create DEALER socket for each node
	for i, addr := range nodeAddrs {
		dealer, err := zmq.NewSocket(zmq.DEALER)
		if err != nil {
			return nil, fmt.Errorf("failed to create dealer socket: %w", err)
		}
		
		// Set identity for the dealer
		dealer.SetIdentity(fmt.Sprintf("client-%d-%d", time.Now().Unix(), i))
		
		// Connect to node's ROUTER socket
		if err := dealer.Connect(addr); err != nil {
			return nil, fmt.Errorf("failed to connect to %s: %w", addr, err)
		}
		
		client.dealers[i] = dealer
	}
	
	return client, nil
}

// SendOrder sends an order to a random node
func (c *TestClient) SendOrder(order *lx.Order) error {
	// Select random node for load balancing
	nodeIdx := rand.Intn(len(c.dealers))
	dealer := c.dealers[nodeIdx]
	
	// Marshal order to JSON
	data, err := json.Marshal(order)
	if err != nil {
		return fmt.Errorf("failed to marshal order: %w", err)
	}
	
	// Send to dealer (no identity needed, DEALER handles it)
	dealer.Send("", zmq.SNDMORE)
	if _, err := dealer.SendBytes(data, 0); err != nil {
		return fmt.Errorf("failed to send order: %w", err)
	}
	
	atomic.AddUint64(&c.ordersent, 1)
	
	// Receive response asynchronously
	go c.receiveResponse(dealer)
	
	return nil
}

// receiveResponse receives response from node
func (c *TestClient) receiveResponse(dealer *zmq.Socket) {
	// Empty delimiter
	if _, err := dealer.Recv(zmq.DONTWAIT); err != nil {
		return
	}
	
	// Response data
	data, err := dealer.RecvBytes(zmq.DONTWAIT)
	if err != nil {
		return
	}
	
	atomic.AddUint64(&c.responsesRcv, 1)
	
	var response map[string]interface{}
	if err := json.Unmarshal(data, &response); err != nil {
		log.Printf("Failed to unmarshal response: %v", err)
		return
	}
	
	if trades, ok := response["trades"].([]interface{}); ok && len(trades) > 0 {
		log.Printf("Order matched! Trades: %v", trades)
	}
}

// RunLoadTest runs a load test sending many orders
func (c *TestClient) RunLoadTest(duration time.Duration, ordersPerSec int) {
	log.Printf("Starting load test: %d orders/sec for %v", ordersPerSec, duration)
	
	ticker := time.NewTicker(time.Second / time.Duration(ordersPerSec))
	defer ticker.Stop()
	
	timeout := time.After(duration)
	
	for {
		select {
		case <-timeout:
			log.Printf("Load test complete. Sent: %d, Received: %d", 
				atomic.LoadUint64(&c.ordersent),
				atomic.LoadUint64(&c.responsesRcv))
			return
		case <-ticker.C:
			// Generate random order
			order := &lx.Order{
				Symbol: "BTC-USD",
				Side:   lx.Side(rand.Intn(2)),
				Type:   lx.Limit,
				Price:  45000 + float64(rand.Intn(10000)),
				Size:   float64(rand.Intn(10)+1) * 0.1,
				User:   fmt.Sprintf("user-%d", rand.Intn(100)),
			}
			
			if err := c.SendOrder(order); err != nil {
				log.Printf("Failed to send order: %v", err)
			}
		}
	}
}

// Close closes all dealer sockets
func (c *TestClient) Close() {
	for _, dealer := range c.dealers {
		if dealer != nil {
			dealer.Close()
		}
	}
}

// RunTestClient is the main function for the test client
func RunTestClient() {
	var (
		nodes    = flag.String("nodes", "tcp://localhost:5002,tcp://localhost:5102,tcp://localhost:5202", "Comma-separated node addresses")
		duration = flag.Duration("duration", 30*time.Second, "Test duration")
		rate     = flag.Int("rate", 100, "Orders per second")
	)
	flag.Parse()
	
	// Parse node addresses
	nodeAddrs := splitAndTrim(*nodes, ",")
	if len(nodeAddrs) == 0 {
		log.Fatal("No node addresses provided")
	}
	
	// Create client
	client, err := NewTestClient(nodeAddrs)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()
	
	// Run load test
	client.RunLoadTest(*duration, *rate)
	
	// Wait a bit for final responses
	time.Sleep(2 * time.Second)
	
	log.Printf("Final stats - Sent: %d, Received: %d",
		atomic.LoadUint64(&client.ordersent),
		atomic.LoadUint64(&client.responsesRcv))
}