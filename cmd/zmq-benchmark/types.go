package main

import (
	"time"
)

// BinaryFIXOrder represents a compact 60-byte FIX order message
type BinaryFIXOrder struct {
	MsgType      uint8   // 1 byte: 'D'=NewOrder, 'F'=Cancel
	Side         uint8   // 1 byte: 1=Buy, 2=Sell
	OrdType      uint8   // 1 byte: 1=Market, 2=Limit
	TimeInForce  uint8   // 1 byte: 0=Day, 1=IOC, 2=FOK, 3=GTC
	Symbol       [8]byte // 8 bytes: "BTC-USD\0"
	OrderID      uint64  // 8 bytes
	ClOrdID      uint64  // 8 bytes
	Price        uint64  // 8 bytes: Fixed point 8 decimals
	OrderQty     uint64  // 8 bytes: Fixed point 8 decimals
	TransactTime uint64  // 8 bytes: Unix nanos
	Account      uint32  // 4 bytes
	ExecInst     uint32  // 4 bytes
	Checksum     uint32  // 4 bytes: CRC32
}

// Metrics tracks benchmark performance
type Metrics struct {
	StartTime       time.Time
	MessagesOut     uint64
	MessagesIn      uint64
	BytesOut        uint64
	BytesIn         uint64
	TradesExecuted  uint64
	ConsensusRounds uint64
	MinLatency      uint64
	MaxLatency      uint64
	TotalLatency    uint64
	LatencyCount    uint64
}

// BenchmarkConfig holds configuration for benchmark runs
type BenchmarkConfig struct {
	Mode           string        // producer, consumer, consensus, relay
	NodeID         int           // Node identifier for consensus
	Endpoint       string        // ZMQ endpoint
	ConsensusNodes []string      // Peer consensus nodes
	Rate           int           // Messages per second
	BatchSize      int           // Messages per batch
	Duration       time.Duration // Test duration
	Producers      int           // Number of producer threads
	Consumers      int           // Number of consumer threads
	TrackLatency   bool          // Enable latency tracking
	DataDir        string        // BadgerDB data directory
}

// Helper functions for serialization
func serializeOrder(order *BinaryFIXOrder, buf []byte) {
	// Pack order into 60-byte buffer
	buf[0] = order.MsgType
	buf[1] = order.Side
	buf[2] = order.OrdType
	buf[3] = order.TimeInForce
	copy(buf[4:12], order.Symbol[:])
	// ... rest of serialization
}

func deserializeOrder(buf []byte, order *BinaryFIXOrder) {
	// Unpack order from 60-byte buffer
	order.MsgType = buf[0]
	order.Side = buf[1]
	order.OrdType = buf[2]
	order.TimeInForce = buf[3]
	copy(order.Symbol[:], buf[4:12])
	// ... rest of deserialization
}
