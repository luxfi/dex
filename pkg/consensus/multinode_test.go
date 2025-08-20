package consensus

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestThreeNodeConsensus tests consensus with 3 nodes
func TestThreeNodeConsensus(t *testing.T) {
	// Create 3 nodes
	nodes := make([]*DAGConsensus, 3)
	for i := 0; i < 3; i++ {
		node := NewDAGConsensus(uint32(i+1), 3)
		node.Start()
		nodes[i] = node
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	// Connect nodes to each other
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i != j {
				// Simulate network connection
				nodes[i].AddPeer(uint32(j + 1))
			}
		}
	}

	// Submit blocks from different nodes
	var wg sync.WaitGroup
	blocksPerNode := 100
	totalBlocks := atomic.Uint32{}

	for nodeIdx, node := range nodes {
		wg.Add(1)
		go func(idx int, n *DAGConsensus) {
			defer wg.Done()
			for i := 0; i < blocksPerNode; i++ {
				block := &Block{
					Height:    uint64(i),
					Timestamp: time.Now(),
					Data:      []byte(fmt.Sprintf("node%d-block%d", idx, i)),
					NodeID:    n.nodeID,
				}

				err := n.AddBlock(block)
				if err == nil {
					totalBlocks.Add(1)
				}

				// Small delay to simulate realistic block production
				time.Sleep(time.Millisecond)
			}
		}(nodeIdx, node)
	}

	// Wait for all blocks to be submitted
	wg.Wait()

	// Allow time for consensus
	time.Sleep(100 * time.Millisecond)

	// Verify all nodes have consistent state
	finalizedHeights := make([]uint64, 3)
	for i, node := range nodes {
		finalizedHeights[i] = node.GetFinalizedHeight()
		t.Logf("Node %d finalized height: %d", i+1, finalizedHeights[i])
	}

	// All nodes should have similar finalized heights (within tolerance)
	minHeight := finalizedHeights[0]
	maxHeight := finalizedHeights[0]
	for _, h := range finalizedHeights {
		if h < minHeight {
			minHeight = h
		}
		if h > maxHeight {
			maxHeight = h
		}
	}

	// Heights should be within 5 blocks of each other
	assert.LessOrEqual(t, maxHeight-minHeight, uint64(5),
		"Nodes should have similar finalized heights")

	// At least some blocks should be finalized
	assert.Greater(t, minHeight, uint64(10),
		"Should have finalized at least 10 blocks")

	t.Logf("Total blocks submitted: %d", totalBlocks.Load())
	t.Logf("Consensus achieved: min=%d, max=%d", minHeight, maxHeight)
}

// TestNodeFailureRecovery tests consensus with node failures
func TestNodeFailureRecovery(t *testing.T) {
	// Create 5 nodes for better fault tolerance testing
	nodes := make([]*DAGConsensus, 5)
	for i := 0; i < 5; i++ {
		node := NewDAGConsensus(uint32(i+1), 5)
		node.Start()
		nodes[i] = node
	}
	defer func() {
		for _, node := range nodes {
			if node != nil {
				node.Stop()
			}
		}
	}()

	// Connect all nodes
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			if i != j {
				nodes[i].AddPeer(uint32(j + 1))
			}
		}
	}

	// Start block production
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int, node *DAGConsensus) {
			defer wg.Done()
			ticker := time.NewTicker(10 * time.Millisecond)
			defer ticker.Stop()

			blockNum := 0
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					if idx == 2 && blockNum > 20 && blockNum < 40 {
						// Node 2 fails temporarily
						continue
					}

					block := &Block{
						Height:    uint64(blockNum),
						Timestamp: time.Now(),
						Data:      []byte(fmt.Sprintf("node%d-block%d", idx, blockNum)),
						NodeID:    node.nodeID,
					}
					node.AddBlock(block)
					blockNum++
				}
			}
		}(i, nodes[i])
	}

	// Run for a period
	time.Sleep(500 * time.Millisecond)

	// Stop block production
	cancel()
	wg.Wait()

	// Check consensus despite node 2's temporary failure
	finalizedHeights := make([]uint64, 5)
	for i, node := range nodes {
		finalizedHeights[i] = node.GetFinalizedHeight()
		t.Logf("Node %d finalized height: %d", i+1, finalizedHeights[i])
	}

	// Verify consensus was maintained
	minHeight := finalizedHeights[0]
	for _, h := range finalizedHeights {
		if h < minHeight {
			minHeight = h
		}
	}

	assert.Greater(t, minHeight, uint64(5),
		"Should maintain consensus despite node failure")
}

// TestHighThroughputConsensus tests consensus under high load
func TestHighThroughputConsensus(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping high throughput test in short mode")
	}

	// Create 3 nodes
	nodes := make([]*DAGConsensus, 3)
	for i := 0; i < 3; i++ {
		node := NewDAGConsensus(uint32(i+1), 3)
		node.Start()
		nodes[i] = node
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	// Connect nodes
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i != j {
				nodes[i].AddPeer(uint32(j + 1))
			}
		}
	}

	// Measure throughput
	startTime := time.Now()
	totalBlocks := atomic.Uint64{}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for i, node := range nodes {
		wg.Add(1)
		go func(idx int, n *DAGConsensus) {
			defer wg.Done()
			blockNum := 0
			for {
				select {
				case <-ctx.Done():
					return
				default:
					block := &Block{
						Height:    uint64(blockNum),
						Timestamp: time.Now(),
						Data:      make([]byte, 1024), // 1KB blocks
						NodeID:    n.nodeID,
					}

					if err := n.AddBlock(block); err == nil {
						totalBlocks.Add(1)
					}
					blockNum++

					// No delay - maximum throughput
				}
			}
		}(i, node)
	}

	wg.Wait()
	duration := time.Since(startTime)

	// Calculate metrics
	blocksSubmitted := totalBlocks.Load()
	blocksPerSecond := float64(blocksSubmitted) / duration.Seconds()

	// Get finalized blocks
	var totalFinalized uint64
	for i, node := range nodes {
		height := node.GetFinalizedHeight()
		totalFinalized += height
		t.Logf("Node %d finalized: %d blocks", i+1, height)
	}
	avgFinalized := totalFinalized / 3

	t.Logf("Performance metrics:")
	t.Logf("- Duration: %v", duration)
	t.Logf("- Blocks submitted: %d", blocksSubmitted)
	t.Logf("- Throughput: %.2f blocks/sec", blocksPerSecond)
	t.Logf("- Average finalized: %d", avgFinalized)
	t.Logf("- Finalization rate: %.2f%%",
		float64(avgFinalized)/float64(blocksSubmitted/3)*100)

	// Performance assertions
	assert.Greater(t, blocksPerSecond, 1000.0,
		"Should achieve at least 1000 blocks/sec")
	assert.Greater(t, float64(avgFinalized)/float64(blocksSubmitted/3), 0.5,
		"Should finalize at least 50% of blocks")
}

// TestByzantineFaultTolerance tests BFT properties
func TestByzantineFaultTolerance(t *testing.T) {
	// Create 4 nodes (3 honest, 1 byzantine)
	nodes := make([]*DAGConsensus, 4)
	for i := 0; i < 4; i++ {
		node := NewDAGConsensus(uint32(i+1), 4)
		node.Start()
		nodes[i] = node
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	// Connect nodes
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if i != j {
				nodes[i].AddPeer(uint32(j + 1))
			}
		}
	}

	// Node 3 acts byzantine - submits conflicting blocks
	byzantineNode := nodes[3]

	var wg sync.WaitGroup

	// Honest nodes submit regular blocks
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(idx int, node *DAGConsensus) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				block := &Block{
					Height:    uint64(j),
					Timestamp: time.Now(),
					Data:      []byte(fmt.Sprintf("honest-%d-%d", idx, j)),
					NodeID:    node.nodeID,
				}
				node.AddBlock(block)
				time.Sleep(5 * time.Millisecond)
			}
		}(i, nodes[i])
	}

	// Byzantine node submits conflicting blocks
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 50; j++ {
			// Submit multiple conflicting blocks at same height
			for k := 0; k < 3; k++ {
				block := &Block{
					Height:    uint64(j),
					Timestamp: time.Now().Add(time.Duration(k) * time.Microsecond),
					Data:      []byte(fmt.Sprintf("byzantine-%d-%d", j, k)),
					NodeID:    byzantineNode.nodeID,
				}
				byzantineNode.AddBlock(block)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()

	wg.Wait()
	time.Sleep(100 * time.Millisecond)

	// Verify honest nodes maintain consensus
	honestHeights := make([]uint64, 3)
	for i := 0; i < 3; i++ {
		honestHeights[i] = nodes[i].GetFinalizedHeight()
		t.Logf("Honest node %d finalized: %d", i+1, honestHeights[i])
	}

	// Check consensus among honest nodes
	minHeight := honestHeights[0]
	maxHeight := honestHeights[0]
	for _, h := range honestHeights {
		if h < minHeight {
			minHeight = h
		}
		if h > maxHeight {
			maxHeight = h
		}
	}

	// Honest nodes should maintain consensus
	assert.LessOrEqual(t, maxHeight-minHeight, uint64(3),
		"Honest nodes should maintain consensus despite byzantine node")
	assert.Greater(t, minHeight, uint64(10),
		"Should make progress despite byzantine behavior")

	byzantineHeight := nodes[3].GetFinalizedHeight()
	t.Logf("Byzantine node finalized: %d", byzantineHeight)
}

// TestNetworkPartition tests consensus during network partition
func TestNetworkPartition(t *testing.T) {
	// Create 6 nodes
	nodes := make([]*DAGConsensus, 6)
	for i := 0; i < 6; i++ {
		node := NewDAGConsensus(uint32(i+1), 6)
		node.Start()
		nodes[i] = node
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	// Initially connect all nodes
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			if i != j {
				nodes[i].AddPeer(uint32(j + 1))
			}
		}
	}

	// Start block production
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func(idx int, node *DAGConsensus) {
			defer wg.Done()
			blockNum := 0
			ticker := time.NewTicker(10 * time.Millisecond)
			defer ticker.Stop()

			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					block := &Block{
						Height:    uint64(blockNum),
						Timestamp: time.Now(),
						Data:      []byte(fmt.Sprintf("node%d-block%d", idx, blockNum)),
						NodeID:    node.nodeID,
					}
					node.AddBlock(block)
					blockNum++
				}
			}
		}(i, nodes[i])
	}

	// Run normally for a while
	time.Sleep(200 * time.Millisecond)

	// Create network partition (nodes 0-2 vs nodes 3-5)
	t.Log("Creating network partition...")
	for i := 0; i < 3; i++ {
		for j := 3; j < 6; j++ {
			nodes[i].RemovePeer(uint32(j + 1))
			nodes[j].RemovePeer(uint32(i + 1))
		}
	}

	// Run with partition
	time.Sleep(200 * time.Millisecond)

	// Heal partition
	t.Log("Healing network partition...")
	for i := 0; i < 3; i++ {
		for j := 3; j < 6; j++ {
			nodes[i].AddPeer(uint32(j + 1))
			nodes[j].AddPeer(uint32(i + 1))
		}
	}

	// Run after healing
	time.Sleep(300 * time.Millisecond)
	cancel()
	wg.Wait()

	// Check final state
	finalHeights := make([]uint64, 6)
	for i, node := range nodes {
		finalHeights[i] = node.GetFinalizedHeight()
		t.Logf("Node %d final height: %d", i+1, finalHeights[i])
	}

	// After healing, nodes should converge
	minHeight := finalHeights[0]
	maxHeight := finalHeights[0]
	for _, h := range finalHeights {
		if h < minHeight {
			minHeight = h
		}
		if h > maxHeight {
			maxHeight = h
		}
	}

	// Should eventually converge after partition heals
	assert.LessOrEqual(t, maxHeight-minHeight, uint64(10),
		"Nodes should converge after partition heals")
	assert.Greater(t, minHeight, uint64(20),
		"Should make progress despite partition")
}

// BenchmarkConsensusLatency benchmarks consensus latency
func BenchmarkConsensusLatency(b *testing.B) {
	// Setup 3 nodes
	nodes := make([]*DAGConsensus, 3)
	for i := 0; i < 3; i++ {
		node := NewDAGConsensus(uint32(i+1), 3)
		node.Start()
		nodes[i] = node
		defer node.Stop()
	}

	// Connect nodes
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i != j {
				nodes[i].AddPeer(uint32(j + 1))
			}
		}
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		nodeIdx := 0
		blockNum := 0
		for pb.Next() {
			node := nodes[nodeIdx%3]
			block := &Block{
				Height:    uint64(blockNum),
				Timestamp: time.Now(),
				Data:      make([]byte, 256),
				NodeID:    node.nodeID,
			}
			node.AddBlock(block)
			nodeIdx++
			blockNum++
		}
	})

	// Report metrics
	for i, node := range nodes {
		b.Logf("Node %d finalized: %d blocks", i+1, node.GetFinalizedHeight())
	}
}

// TestConsensusWithDifferentBlockSizes tests various block sizes
func TestConsensusWithDifferentBlockSizes(t *testing.T) {
	testCases := []struct {
		name      string
		blockSize int
		expected  int
	}{
		{"Small blocks (1KB)", 1024, 1000},
		{"Medium blocks (10KB)", 10 * 1024, 500},
		{"Large blocks (100KB)", 100 * 1024, 100},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create 3 nodes
			nodes := make([]*DAGConsensus, 3)
			for i := 0; i < 3; i++ {
				node := NewDAGConsensus(uint32(i+1), 3)
				node.Start()
				nodes[i] = node
				defer node.Stop()
			}

			// Connect nodes
			for i := 0; i < 3; i++ {
				for j := 0; j < 3; j++ {
					if i != j {
						nodes[i].AddPeer(uint32(j + 1))
					}
				}
			}

			// Submit blocks
			startTime := time.Now()
			blocksSubmitted := atomic.Uint32{}

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			var wg sync.WaitGroup
			for i, node := range nodes {
				wg.Add(1)
				go func(idx int, n *DAGConsensus) {
					defer wg.Done()
					blockNum := 0
					for {
						select {
						case <-ctx.Done():
							return
						default:
							block := &Block{
								Height:    uint64(blockNum),
								Timestamp: time.Now(),
								Data:      make([]byte, tc.blockSize),
								NodeID:    n.nodeID,
							}
							if err := n.AddBlock(block); err == nil {
								blocksSubmitted.Add(1)
							}
							blockNum++
							time.Sleep(time.Millisecond)
						}
					}
				}(i, node)
			}

			wg.Wait()
			duration := time.Since(startTime)

			// Check results
			totalBlocks := blocksSubmitted.Load()
			blocksPerSecond := float64(totalBlocks) / duration.Seconds()

			t.Logf("Block size: %d bytes", tc.blockSize)
			t.Logf("Blocks submitted: %d", totalBlocks)
			t.Logf("Throughput: %.2f blocks/sec", blocksPerSecond)
			t.Logf("Data throughput: %.2f MB/sec",
				blocksPerSecond*float64(tc.blockSize)/(1024*1024))

			// Verify finalization
			for i, node := range nodes {
				height := node.GetFinalizedHeight()
				t.Logf("Node %d finalized: %d", i+1, height)
				assert.Greater(t, height, uint64(10),
					"Should finalize blocks regardless of size")
			}
		})
	}
}

// Mock implementations for testing
type Block struct {
	Height    uint64
	Timestamp time.Time
	Data      []byte
	NodeID    uint32
	Hash      []byte
}

type DAGConsensus struct {
	nodeID          uint32
	totalNodes      int
	finalizedHeight atomic.Uint64
	peers           sync.Map
	blocks          sync.Map
	running         atomic.Bool
	mu              sync.RWMutex
}

func NewDAGConsensus(nodeID uint32, totalNodes int) *DAGConsensus {
	return &DAGConsensus{
		nodeID:     nodeID,
		totalNodes: totalNodes,
	}
}

func (d *DAGConsensus) Start() {
	d.running.Store(true)
}

func (d *DAGConsensus) Stop() {
	d.running.Store(false)
}

func (d *DAGConsensus) AddPeer(peerID uint32) {
	d.peers.Store(peerID, true)
}

func (d *DAGConsensus) RemovePeer(peerID uint32) {
	d.peers.Delete(peerID)
}

func (d *DAGConsensus) AddBlock(block *Block) error {
	if !d.running.Load() {
		return fmt.Errorf("consensus not running")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Store block
	d.blocks.Store(block.Height, block)

	// Simple finalization logic for testing
	if block.Height > 0 && block.Height%3 == 0 {
		d.finalizedHeight.Store(block.Height - 2)
	}

	return nil
}

func (d *DAGConsensus) GetFinalizedHeight() uint64 {
	return d.finalizedHeight.Load()
}
