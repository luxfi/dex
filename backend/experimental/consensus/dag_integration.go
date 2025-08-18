package consensus

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/consensus/engine/dag"
	"github.com/luxfi/consensus/protocol/nebula"
	"github.com/luxfi/dex/backend/pkg/lx"
	"github.com/luxfi/ids"
)

// OrderVertex represents an order in the DAG
type OrderVertex struct {
	ID        ids.ID
	Order     *lx.Order
	Trades    []lx.Trade
	Parents   []ids.ID
	Timestamp time.Time
	NodeID    string
	Height    uint64
	mu        sync.RWMutex
}

// GetID returns the vertex ID
func (v *OrderVertex) GetID() ids.ID {
	return v.ID
}

// GetParents returns parent vertex IDs
func (v *OrderVertex) GetParents() []ids.ID {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.Parents
}

// DAGOrderBook implements a DAG-based distributed order book
type DAGOrderBook struct {
	// Node identity
	nodeID string

	// Order book state
	orderBook *lx.OrderBook

	// DAG structure
	vertices  map[ids.ID]*OrderVertex
	frontier  []ids.ID
	finalized map[ids.ID]bool

	// Consensus engine
	consensus nebula.Protocol[ids.ID]
	dagEngine *dag.Engine

	// Network peers
	peers map[string]*RemoteNode

	// Metrics
	vertexCount atomic.Uint64
	tradeCount  atomic.Uint64

	// Synchronization
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// RemoteNode represents a peer in the network
type RemoteNode struct {
	ID       string
	Endpoint string
	Client   interface{} // Will be ZMQ or gRPC client
	LastSeen time.Time
}

// NewDAGOrderBook creates a new DAG-based order book
func NewDAGOrderBook(nodeID string, symbol string) (*DAGOrderBook, error) {
	ctx, cancel := context.WithCancel(context.Background())

	dob := &DAGOrderBook{
		nodeID:    nodeID,
		orderBook: lx.NewOrderBook(symbol),
		vertices:  make(map[ids.ID]*OrderVertex),
		finalized: make(map[ids.ID]bool),
		peers:     make(map[string]*RemoteNode),
		ctx:       ctx,
		cancel:    cancel,
	}

	// Initialize DAG consensus engine
	params := dag.Parameters{
		K:                   20, // Sample size
		AlphaPreference:     15, // Quorum size for preference
		AlphaConfidence:     15, // Quorum size for confidence
		Beta:                20, // Confidence threshold
		MaxParents:          5,  // Max parent vertices
		MaxVerticesPerRound: 100,
		ConflictSetSize:     10,
		VertexTimeout:       time.Second,
	}

	// Create Nebula consensus configuration
	cfg := nebula.Config[ids.ID]{
		Graph:      dob,
		Tips:       dob.GetFrontier,
		Thresholds: &thresholds{},
		Confidence: &confidence{},
		Orderer:    &orderer{dob: dob},
		Propose:    dob.ProposeVertex,
		Apply:      dob.ApplyVertices,
		Send:       dob.BroadcastVertex,
	}

	consensus, err := nebula.New[ids.ID](cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create consensus: %w", err)
	}

	dob.consensus = consensus

	return dob, nil
}

// AddOrder adds an order to the DAG
func (dob *DAGOrderBook) AddOrder(order *lx.Order) (*OrderVertex, error) {
	dob.mu.Lock()
	defer dob.mu.Unlock()

	// Create vertex for this order
	vertex := &OrderVertex{
		Order:     order,
		Timestamp: time.Now(),
		NodeID:    dob.nodeID,
		Height:    dob.vertexCount.Add(1),
	}

	// Set parents to current frontier
	vertex.Parents = append([]ids.ID{}, dob.frontier...)
	if len(vertex.Parents) > 5 {
		vertex.Parents = vertex.Parents[:5] // Limit parents
	}

	// Generate vertex ID
	vertex.ID = dob.generateVertexID(vertex)

	// Try to match order locally first
	trades := dob.orderBook.MatchOrders()
	vertex.Trades = trades

	// Add to DAG
	dob.vertices[vertex.ID] = vertex
	dob.updateFrontier(vertex.ID)

	// Broadcast to peers
	go dob.broadcastVertexToPeers(vertex)

	return vertex, nil
}

// ProcessRemoteVertex processes a vertex received from a peer
func (dob *DAGOrderBook) ProcessRemoteVertex(vertex *OrderVertex) error {
	dob.mu.Lock()
	defer dob.mu.Unlock()

	// Validate vertex
	if err := dob.validateVertex(vertex); err != nil {
		return fmt.Errorf("invalid vertex: %w", err)
	}

	// Check if we already have it
	if _, exists := dob.vertices[vertex.ID]; exists {
		return nil // Already processed
	}

	// Add to DAG
	dob.vertices[vertex.ID] = vertex

	// Process order if not from our node
	if vertex.NodeID != dob.nodeID {
		// Apply order to local book
		dob.orderBook.AddOrder(vertex.Order)

		// Apply trades
		for _, trade := range vertex.Trades {
			dob.tradeCount.Add(1)
		}
	}

	// Update frontier
	dob.updateFrontier(vertex.ID)

	return nil
}

// RunConsensus runs the consensus protocol
func (dob *DAGOrderBook) RunConsensus() error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-dob.ctx.Done():
			return dob.ctx.Err()
		case <-ticker.C:
			// Run consensus step
			if err := dob.consensus.Step(dob.ctx); err != nil {
				return err
			}

			// Process finalized vertices
			finalized := dob.consensus.Finalized()
			for _, id := range finalized {
				dob.finalizeVertex(id)
			}
		}
	}
}

// Graph interface implementation for Nebula

// Parents returns the parents of a vertex
func (dob *DAGOrderBook) Parents(id ids.ID) []ids.ID {
	dob.mu.RLock()
	defer dob.mu.RUnlock()

	if vertex, exists := dob.vertices[id]; exists {
		return vertex.GetParents()
	}
	return nil
}

// GetFrontier returns the current frontier vertices
func (dob *DAGOrderBook) GetFrontier() []ids.ID {
	dob.mu.RLock()
	defer dob.mu.RUnlock()
	return append([]ids.ID{}, dob.frontier...)
}

// ProposeVertex proposes a new vertex for consensus
func (dob *DAGOrderBook) ProposeVertex(ctx context.Context) (ids.ID, error) {
	// In production, this would create a new vertex with pending orders
	// For now, return empty ID
	return ids.Empty, nil
}

// ApplyVertices applies finalized vertices
func (dob *DAGOrderBook) ApplyVertices(ctx context.Context, vertices []ids.ID) error {
	for _, id := range vertices {
		dob.finalizeVertex(id)
	}
	return nil
}

// BroadcastVertex broadcasts a vertex to peers
func (dob *DAGOrderBook) BroadcastVertex(ctx context.Context, id ids.ID, peers []ids.ID) error {
	vertex, exists := dob.vertices[id]
	if !exists {
		return fmt.Errorf("vertex not found: %s", id)
	}

	go dob.broadcastVertexToPeers(vertex)
	return nil
}

// Helper methods

func (dob *DAGOrderBook) generateVertexID(vertex *OrderVertex) ids.ID {
	h := sha256.New()
	h.Write([]byte(vertex.NodeID))
	binary.Write(h, binary.BigEndian, vertex.Height)
	binary.Write(h, binary.BigEndian, vertex.Timestamp.UnixNano())
	if vertex.Order != nil {
		binary.Write(h, binary.BigEndian, vertex.Order.ID)
	}
	return ids.ID(h.Sum(nil)[:32])
}

func (dob *DAGOrderBook) validateVertex(vertex *OrderVertex) error {
	// Validate parents exist
	for _, parentID := range vertex.Parents {
		if _, exists := dob.vertices[parentID]; !exists {
			return fmt.Errorf("parent vertex not found: %s", parentID)
		}
	}

	// Validate order
	if vertex.Order == nil {
		return fmt.Errorf("vertex missing order")
	}

	return nil
}

func (dob *DAGOrderBook) updateFrontier(newVertex ids.ID) {
	// Remove parents from frontier
	vertex := dob.vertices[newVertex]
	parentSet := make(map[ids.ID]bool)
	for _, p := range vertex.Parents {
		parentSet[p] = true
	}

	newFrontier := []ids.ID{}
	for _, id := range dob.frontier {
		if !parentSet[id] {
			newFrontier = append(newFrontier, id)
		}
	}

	// Add new vertex to frontier
	newFrontier = append(newFrontier, newVertex)
	dob.frontier = newFrontier
}

func (dob *DAGOrderBook) finalizeVertex(id ids.ID) {
	dob.mu.Lock()
	defer dob.mu.Unlock()

	if dob.finalized[id] {
		return
	}

	dob.finalized[id] = true

	// Apply trades permanently
	if vertex, exists := dob.vertices[id]; exists {
		for _, trade := range vertex.Trades {
			// Store trade in persistent storage
			_ = trade // TODO: Store in database
		}
	}
}

func (dob *DAGOrderBook) broadcastVertexToPeers(vertex *OrderVertex) {
	data, err := json.Marshal(vertex)
	if err != nil {
		return
	}

	for _, peer := range dob.peers {
		// Send to peer (would use ZMQ or gRPC)
		_ = peer
		_ = data
	}
}

// Consensus helper types

type thresholds struct{}

func (t *thresholds) Alpha(k int, phase uint64) (int, int) {
	// Return (AlphaPreference, AlphaConfidence)
	return k/2 + 1, k/2 + 1
}

type confidence struct {
	count int
	mu    sync.Mutex
}

func (c *confidence) Record(success bool) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if success {
		c.count++
		return c.count >= 20 // Beta threshold
	}
	c.count = 0
	return false
}

func (c *confidence) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.count = 0
}

type orderer struct {
	dob *DAGOrderBook
}

func (o *orderer) Schedule(ctx context.Context, vertices []ids.ID) ([]ids.ID, error) {
	// Simple FIFO ordering for now
	// In production, would use more sophisticated ordering
	return vertices, nil
}

// GetStats returns DAG statistics
func (dob *DAGOrderBook) GetStats() map[string]interface{} {
	dob.mu.RLock()
	defer dob.mu.RUnlock()

	return map[string]interface{}{
		"node_id":       dob.nodeID,
		"vertices":      len(dob.vertices),
		"finalized":     len(dob.finalized),
		"frontier_size": len(dob.frontier),
		"total_trades":  dob.tradeCount.Load(),
		"peers":         len(dob.peers),
	}
}
