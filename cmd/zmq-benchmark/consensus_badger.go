package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"sync/atomic"
	"time"

	"github.com/luxfi/database"
	zmq "github.com/pebbe/zmq4"
)

// ConsensusBlock represents a finalized block of orders
type ConsensusBlock struct {
	BlockNumber uint64
	ParentHash  [32]byte
	Timestamp   uint64
	ProposerID  uint32
	NumOrders   uint32
	OrdersHash  [32]byte
	Signature   [64]byte
	Orders      []BinaryFIXOrder
}

// BadgerConsensusNode extends the basic consensus node with database storage
type BadgerConsensusNode struct {
	db              database.Database
	nodeID          int
	blockHeight     uint64
	pendingOrders   []BinaryFIXOrder
	finalizedBlocks uint64
	metrics         *Metrics
}

// InitBadgerConsensus initializes database for consensus storage
func InitBadgerConsensus(nodeID int, dataDir string) (*BadgerConsensusNode, error) {
	// Create data directory if not exists
	err := os.MkdirAll(dataDir, 0755)
	if err != nil {
		return nil, err
	}

	// Use luxfi/database (or our memDB for testing)
	// In production would use persistent database
	db := newSimpleMemDB()

	node := &BadgerConsensusNode{
		db:            db,
		nodeID:        nodeID,
		blockHeight:   0,
		pendingOrders: make([]BinaryFIXOrder, 0, 10000),
		metrics:       &Metrics{StartTime: time.Now()},
	}

	// Load last block height
	val, err := db.Get([]byte("last_block_height"))
	if err == nil && len(val) >= 8 {
		node.blockHeight = binary.BigEndian.Uint64(val)
	}

	fmt.Printf("Database initialized at height: %d\n", node.blockHeight)
	return node, nil
}

// StoreBlock persists a finalized block to database
func (n *BadgerConsensusNode) StoreBlock(block *ConsensusBlock) error {
	batch := n.db.NewBatch()

	// Store block header
	blockKey := fmt.Sprintf("block:%d", block.BlockNumber)
	blockData := serializeBlock(block)
	err := batch.Put([]byte(blockKey), blockData)
	if err != nil {
		return err
	}

	// Store individual orders with block reference
	for i, order := range block.Orders {
		orderKey := fmt.Sprintf("order:%d:%d", block.BlockNumber, i)
		orderData := make([]byte, 60)
		serializeOrder(&order, orderData)
		err = batch.Put([]byte(orderKey), orderData)
		if err != nil {
			return err
		}
	}

	// Update block height
	heightData := make([]byte, 8)
	binary.BigEndian.PutUint64(heightData, block.BlockNumber)
	err = batch.Put([]byte("last_block_height"), heightData)
	if err != nil {
		return err
	}

	// Store block hash for chain verification
	hashKey := fmt.Sprintf("hash:%d", block.BlockNumber)
	err = batch.Put([]byte(hashKey), block.OrdersHash[:])
	if err != nil {
		return err
	}

	// Write the batch
	err = batch.Write()
	if err != nil {
		return err
	}

	// Update metrics
	atomic.AddUint64(&n.finalizedBlocks, 1)
	atomic.AddUint64(&n.metrics.TradesExecuted, uint64(len(block.Orders)))

	return nil
}

// GetBlock retrieves a block from database
func (n *BadgerConsensusNode) GetBlock(blockNumber uint64) (*ConsensusBlock, error) {
	var block ConsensusBlock

	blockKey := fmt.Sprintf("block:%d", blockNumber)
	val, err := n.db.Get([]byte(blockKey))
	if err != nil {
		return nil, err
	}

	deserializeBlock(val, &block)
	return &block, nil
}

// GetOrdersInBlock retrieves all orders in a specific block
func (n *BadgerConsensusNode) GetOrdersInBlock(blockNumber uint64) ([]BinaryFIXOrder, error) {
	// Since we don't have iterator support in simpleMemDB,
	// we'll retrieve the block and get orders from it
	block, err := n.GetBlock(blockNumber)
	if err != nil {
		return nil, err
	}

	return block.Orders, nil
}

// RunConsensusWithBadger runs a consensus node with BadgerDB persistence
func RunConsensusWithBadger(config BenchmarkConfig) {
	// Initialize BadgerDB
	dataDir := fmt.Sprintf("./badger-node-%d", config.NodeID)
	node, err := InitBadgerConsensus(config.NodeID, dataDir)
	if err != nil {
		log.Fatalf("Failed to initialize BadgerDB: %v", err)
	}
	defer node.db.Close()

	fmt.Printf("Starting Consensus Node %d with BadgerDB\n", config.NodeID)
	fmt.Printf("Data directory: %s\n", dataDir)
	fmt.Printf("Current height: %d\n", node.blockHeight)

	// Create ZMQ sockets
	router, err := zmq.NewSocket(zmq.ROUTER)
	if err != nil {
		log.Fatalf("Failed to create router socket: %v", err)
	}
	defer router.Close()

	router.SetIdentity(fmt.Sprintf("node-%d", config.NodeID))
	router.Bind(config.Endpoint)

	// Create PUB socket for consensus messages
	pub, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		log.Fatalf("Failed to create pub socket: %v", err)
	}
	defer pub.Close()

	pubEndpoint := fmt.Sprintf("tcp://*:%d", 6000+config.NodeID)
	pub.Bind(pubEndpoint)

	// Create SUB socket for receiving consensus messages
	sub, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		log.Fatalf("Failed to create sub socket: %v", err)
	}
	defer sub.Close()

	sub.SetSubscribe("")
	for _, nodeAddr := range config.ConsensusNodes {
		sub.Connect(nodeAddr)
	}

	// Start consensus worker with BadgerDB
	go node.consensusWorkerWithStorage(router, pub, sub, config)

	// Start metrics printer
	go printBadgerMetrics(node)

	// Start periodic compaction
	go node.periodicCompaction()

	// Run forever
	select {}
}

func (n *BadgerConsensusNode) consensusWorkerWithStorage(router, pub, sub *zmq.Socket, config BenchmarkConfig) {
	poller := zmq.NewPoller()
	poller.Add(router, zmq.POLLIN)
	poller.Add(sub, zmq.POLLIN)

	consensusTicker := time.NewTicker(100 * time.Millisecond) // 10 blocks/sec
	defer consensusTicker.Stop()

	for {
		sockets, err := poller.Poll(10 * time.Millisecond)
		if err != nil {
			continue
		}

		for _, socket := range sockets {
			switch s := socket.Socket; s {
			case router:
				// Receive client orders
				msg, _ := s.RecvMessageBytes(0)
				if len(msg) >= 2 {
					data := msg[len(msg)-1]
					numOrders := len(data) / 60

					// Parse orders and add to pending
					for i := 0; i < numOrders; i++ {
						var order BinaryFIXOrder
						deserializeOrder(data[i*60:(i+1)*60], &order)
						n.pendingOrders = append(n.pendingOrders, order)
					}

					atomic.AddUint64(&n.metrics.MessagesIn, uint64(numOrders))
					atomic.AddUint64(&n.metrics.BytesIn, uint64(len(data)))
				}

			case sub:
				// Receive consensus block from other node
				msg, _ := s.RecvBytes(0)
				if len(msg) > 0 {
					// Process consensus block
					var block ConsensusBlock
					deserializeBlock(msg, &block)

					// Verify and store block
					if block.BlockNumber > n.blockHeight {
						err := n.StoreBlock(&block)
						if err == nil {
							n.blockHeight = block.BlockNumber
							atomic.AddUint64(&n.metrics.ConsensusRounds, 1)
						}
					}
				}
			}
		}

		// Check if it's time to propose a new block
		select {
		case <-consensusTicker.C:
			if len(n.pendingOrders) > 0 {
				// Create new block
				block := ConsensusBlock{
					BlockNumber: n.blockHeight + 1,
					Timestamp:   uint64(time.Now().UnixNano()),
					ProposerID:  uint32(n.nodeID),
					NumOrders:   uint32(len(n.pendingOrders)),
					Orders:      n.pendingOrders,
				}

				// Calculate orders hash
				block.OrdersHash = calculateBlockHash(&block)

				// Store block locally
				err := n.StoreBlock(&block)
				if err == nil {
					// Broadcast block to other nodes
					blockData := serializeBlock(&block)
					pub.SendBytes(blockData, zmq.DONTWAIT)

					n.blockHeight++
					atomic.AddUint64(&n.metrics.MessagesOut, uint64(len(n.pendingOrders)))

					// Clear pending orders
					n.pendingOrders = n.pendingOrders[:0]
				}
			}
		default:
		}
	}
}

func (n *BadgerConsensusNode) periodicCompaction() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		// Run compaction on database
		// For luxfi/database, this is a no-op for memDB but would work for persistent storage
		err := n.db.Compact(nil, nil)
		if err != nil {
			log.Printf("Compaction error: %v", err)
		}
	}
}

func serializeBlock(block *ConsensusBlock) []byte {
	// Calculate size
	size := 8 + 32 + 8 + 4 + 4 + 32 + 64 + (60 * len(block.Orders))
	buf := make([]byte, size)

	offset := 0
	binary.BigEndian.PutUint64(buf[offset:], block.BlockNumber)
	offset += 8
	copy(buf[offset:], block.ParentHash[:])
	offset += 32
	binary.BigEndian.PutUint64(buf[offset:], block.Timestamp)
	offset += 8
	binary.BigEndian.PutUint32(buf[offset:], block.ProposerID)
	offset += 4
	binary.BigEndian.PutUint32(buf[offset:], block.NumOrders)
	offset += 4
	copy(buf[offset:], block.OrdersHash[:])
	offset += 32
	copy(buf[offset:], block.Signature[:])
	offset += 64

	// Serialize orders
	for _, order := range block.Orders {
		serializeOrder(&order, buf[offset:offset+60])
		offset += 60
	}

	return buf
}

func deserializeBlock(buf []byte, block *ConsensusBlock) {
	offset := 0
	block.BlockNumber = binary.BigEndian.Uint64(buf[offset:])
	offset += 8
	copy(block.ParentHash[:], buf[offset:offset+32])
	offset += 32
	block.Timestamp = binary.BigEndian.Uint64(buf[offset:])
	offset += 8
	block.ProposerID = binary.BigEndian.Uint32(buf[offset:])
	offset += 4
	block.NumOrders = binary.BigEndian.Uint32(buf[offset:])
	offset += 4
	copy(block.OrdersHash[:], buf[offset:offset+32])
	offset += 32
	copy(block.Signature[:], buf[offset:offset+64])
	offset += 64

	// Deserialize orders
	block.Orders = make([]BinaryFIXOrder, block.NumOrders)
	for i := uint32(0); i < block.NumOrders; i++ {
		deserializeOrder(buf[offset:offset+60], &block.Orders[i])
		offset += 60
	}
}

func calculateBlockHash(block *ConsensusBlock) [32]byte {
	// Simple hash calculation for demo
	var hash [32]byte
	data := make([]byte, 16)
	binary.BigEndian.PutUint64(data[0:8], block.BlockNumber)
	binary.BigEndian.PutUint64(data[8:16], block.Timestamp)
	copy(hash[:16], data)
	return hash
}

func printBadgerMetrics(node *BadgerConsensusNode) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		elapsed := time.Since(node.metrics.StartTime).Seconds()

		// Get database stats
		ctx := context.Background()
		dbStats, _ := node.db.HealthCheck(ctx)

		blocks := atomic.LoadUint64(&node.finalizedBlocks)
		trades := atomic.LoadUint64(&node.metrics.TradesExecuted)

		fmt.Printf("[%.0fs] Blocks: %d (%.1f/s) | Orders: %d (%.0f/s) | DB: %v | Height: %d\n",
			elapsed,
			blocks,
			float64(blocks)/elapsed,
			trades,
			float64(trades)/elapsed,
			dbStats,
			node.blockHeight,
		)
	}
}
