// Multi-node DEX deployment test with 3+ nodes
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	zmq "github.com/pebbe/zmq4"
)

type Node struct {
	ID           string
	Type         NodeType
	NatsConn     *nats.Conn
	ZmqPub       *zmq.Socket
	ZmqSub       *zmq.Socket
	Orders       int64
	Trades       int64
	IsLeader     bool
	LastHeartbeat time.Time
}

type NodeType int

const (
	MasterNode NodeType = iota
	ReplicaNode
	GatewayNode
)

type ClusterManager struct {
	nodes       map[string]*Node
	leader      *Node
	mu          sync.RWMutex
	natsURL     string
	zmqPubPort  int
	zmqSubPort  int
}

func main() {
	numNodes := flag.Int("nodes", 3, "Number of nodes to deploy")
	natsURL := flag.String("nats", "nats://localhost:4222", "NATS URL")
	zmqBase := flag.Int("zmq", 5555, "ZeroMQ base port")
	duration := flag.Duration("duration", 30*time.Second, "Test duration")
	flag.Parse()

	log.Printf("🚀 Multi-Node DEX Test")
	log.Printf("📊 Nodes: %d | Duration: %v", *numNodes, *duration)

	cm := &ClusterManager{
		nodes:      make(map[string]*Node),
		natsURL:    *natsURL,
		zmqPubPort: *zmqBase,
		zmqSubPort: *zmqBase + 1000,
	}

	// Start nodes
	var wg sync.WaitGroup
	
	// Start master node
	wg.Add(1)
	go func() {
		defer wg.Done()
		cm.startNode("node-0", MasterNode, 0)
	}()
	
	time.Sleep(2 * time.Second) // Let master start
	
	// Start replica nodes
	for i := 1; i < *numNodes; i++ {
		wg.Add(1)
		nodeID := fmt.Sprintf("node-%d", i)
		nodeNum := i
		go func() {
			defer wg.Done()
			cm.startNode(nodeID, ReplicaNode, nodeNum)
		}()
		time.Sleep(500 * time.Millisecond)
	}
	
	// Monitor cluster
	go cm.monitorCluster()
	
	// Run test workload
	go cm.runWorkload(*duration)
	
	// Wait for duration
	time.Sleep(*duration)
	
	// Print results
	cm.printResults()
	
	// Cleanup
	cm.shutdown()
	wg.Wait()
}

func (cm *ClusterManager) startNode(nodeID string, nodeType NodeType, nodeNum int) {
	node := &Node{
		ID:   nodeID,
		Type: nodeType,
	}
	
	// Connect to NATS
	nc, err := nats.Connect(cm.natsURL)
	if err != nil {
		log.Printf("❌ Node %s: Failed to connect to NATS: %v", nodeID, err)
		return
	}
	node.NatsConn = nc
	
	// Setup ZeroMQ
	pubPort := cm.zmqPubPort + nodeNum
	subPort := cm.zmqSubPort + nodeNum
	
	// Publisher socket for market data
	pub, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		log.Printf("❌ Node %s: Failed to create ZMQ PUB: %v", nodeID, err)
		return
	}
	pub.Bind(fmt.Sprintf("tcp://*:%d", pubPort))
	node.ZmqPub = pub
	
	// Subscriber socket for receiving data
	sub, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		log.Printf("❌ Node %s: Failed to create ZMQ SUB: %v", nodeID, err)
		return
	}
	
	// Subscribe to other nodes
	for i := 0; i < 10; i++ {
		if i != nodeNum {
			subAddr := fmt.Sprintf("tcp://localhost:%d", cm.zmqPubPort+i)
			sub.Connect(subAddr)
		}
	}
	sub.SetSubscribe("")
	node.ZmqSub = sub
	
	// Leader election for master
	if nodeType == MasterNode {
		node.IsLeader = true
		cm.mu.Lock()
		cm.leader = node
		cm.mu.Unlock()
	}
	
	// Store node
	cm.mu.Lock()
	cm.nodes[nodeID] = node
	cm.mu.Unlock()
	
	log.Printf("✅ Node %s started (Type: %v, ZMQ Pub: %d, Sub: %d)", 
		nodeID, nodeType, pubPort, subPort)
	
	// Start node operations
	go node.processOrders()
	go node.publishMarketData()
	go node.subscribeToMarketData()
	go node.heartbeat()
	
	// Keep node running
	select {}
}

func (node *Node) processOrders() {
	// Subscribe to orders via NATS
	node.NatsConn.Subscribe("dex.orders", func(m *nats.Msg) {
		atomic.AddInt64(&node.Orders, 1)
		
		// Process order
		var order map[string]interface{}
		if err := json.Unmarshal(m.Data, &order); err == nil {
			// Simulate matching
			if node.IsLeader {
				// Leader does matching
				if atomic.LoadInt64(&node.Orders)%2 == 0 {
					atomic.AddInt64(&node.Trades, 1)
					
					// Publish trade via ZeroMQ
					trade := fmt.Sprintf(`{"type":"trade","id":%d,"price":50000,"qty":1}`, 
						atomic.LoadInt64(&node.Trades))
					node.ZmqPub.Send(trade, 0)
				}
			}
			
			// Send response
			response := fmt.Sprintf(`{"status":"accepted","node":"%s"}`, node.ID)
			m.Respond([]byte(response))
		}
	})
}

func (node *Node) publishMarketData() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for range ticker.C {
		if node.IsLeader {
			// Publish orderbook snapshot
			snapshot := fmt.Sprintf(`{"type":"snapshot","node":"%s","orders":%d,"trades":%d,"timestamp":%d}`,
				node.ID, 
				atomic.LoadInt64(&node.Orders),
				atomic.LoadInt64(&node.Trades),
				time.Now().UnixNano())
			
			node.ZmqPub.Send(snapshot, zmq.DONTWAIT)
		}
	}
}

func (node *Node) subscribeToMarketData() {
	for {
		msg, err := node.ZmqSub.Recv(0)
		if err != nil {
			continue
		}
		
		// Process market data from other nodes
		var data map[string]interface{}
		if err := json.Unmarshal([]byte(msg), &data); err == nil {
			if data["type"] == "trade" && !node.IsLeader {
				// Replicas track trades from leader
				atomic.AddInt64(&node.Trades, 1)
			}
		}
	}
}

func (node *Node) heartbeat() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		heartbeat := fmt.Sprintf(`{"node":"%s","type":"%v","leader":%v,"orders":%d,"trades":%d}`,
			node.ID, node.Type, node.IsLeader,
			atomic.LoadInt64(&node.Orders),
			atomic.LoadInt64(&node.Trades))
		
		node.NatsConn.Publish("cluster.heartbeat", []byte(heartbeat))
		node.LastHeartbeat = time.Now()
	}
}

func (cm *ClusterManager) monitorCluster() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		cm.mu.RLock()
		
		log.Println("\n📊 Cluster Status:")
		log.Println("================")
		
		totalOrders := int64(0)
		totalTrades := int64(0)
		activeNodes := 0
		
		for _, node := range cm.nodes {
			orders := atomic.LoadInt64(&node.Orders)
			trades := atomic.LoadInt64(&node.Trades)
			totalOrders += orders
			totalTrades += trades
			
			status := "✅"
			if time.Since(node.LastHeartbeat) > 5*time.Second {
				status = "⚠️"
			} else {
				activeNodes++
			}
			
			role := "Replica"
			if node.IsLeader {
				role = "Leader"
			}
			
			log.Printf("%s %s (%s): Orders=%d, Trades=%d", 
				status, node.ID, role, orders, trades)
		}
		
		log.Printf("\n📈 Total: %d orders, %d trades across %d active nodes",
			totalOrders, totalTrades, activeNodes)
		
		if totalOrders > 0 {
			log.Printf("💹 Trade Rate: %.2f%%", 
				float64(totalTrades)/float64(totalOrders)*100)
		}
		
		cm.mu.RUnlock()
	}
}

func (cm *ClusterManager) runWorkload(duration time.Duration) {
	// Connect as client
	nc, err := nats.Connect(cm.natsURL)
	if err != nil {
		log.Printf("❌ Failed to connect workload generator: %v", err)
		return
	}
	defer nc.Close()
	
	// Generate orders
	orderID := uint64(0)
	ticker := time.NewTicker(10 * time.Millisecond) // 100 orders/sec
	defer ticker.Stop()
	
	timeout := time.After(duration)
	
	for {
		select {
		case <-timeout:
			return
		case <-ticker.C:
			orderID++
			order := fmt.Sprintf(`{"id":%d,"symbol":"BTC/USD","side":"buy","price":50000,"qty":1}`, orderID)
			
			// Send via NATS (will be load balanced across nodes)
			msg, err := nc.Request("dex.orders", []byte(order), 100*time.Millisecond)
			if err == nil && msg != nil {
				// Order accepted
			}
		}
	}
}

func (cm *ClusterManager) printResults() {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("📊 MULTI-NODE TEST RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	
	totalOrders := int64(0)
	totalTrades := int64(0)
	
	for _, node := range cm.nodes {
		orders := atomic.LoadInt64(&node.Orders)
		trades := atomic.LoadInt64(&node.Trades)
		totalOrders += orders
		totalTrades += trades
		
		role := "Replica"
		if node.IsLeader {
			role = "Leader "
		}
		
		fmt.Printf("%-10s [%s]: %6d orders, %6d trades\n", 
			node.ID, role, orders, trades)
	}
	
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Total Orders:  %d\n", totalOrders)
	fmt.Printf("Total Trades:  %d\n", totalTrades)
	fmt.Printf("Trade Rate:    %.2f%%\n", float64(totalTrades)/float64(totalOrders)*100)
	fmt.Printf("Nodes Active:  %d\n", len(cm.nodes))
	
	if cm.leader != nil {
		fmt.Printf("Leader Node:   %s\n", cm.leader.ID)
	}
	
	fmt.Println(strings.Repeat("=", 60))
}

func (cm *ClusterManager) shutdown() {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	for _, node := range cm.nodes {
		if node.NatsConn != nil {
			node.NatsConn.Close()
		}
		if node.ZmqPub != nil {
			node.ZmqPub.Close()
		}
		if node.ZmqSub != nil {
			node.ZmqSub.Close()
		}
	}
}

var strings = struct {
	Repeat func(string, int) string
}{
	Repeat: func(s string, n int) string {
		result := ""
		for i := 0; i < n; i++ {
			result += s
		}
		return result
	},
}