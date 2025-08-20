//go:build fpga
// +build fpga

package fpga

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// FPGAType represents the type of FPGA hardware
type FPGAType int

const (
	FPGATypeNone FPGAType = iota
	FPGATypeAMDVersal
	FPGATypeAWSF2
	FPGATypeIntelStratix
	FPGATypeXilinxAlveo
)

// FPGAConfig contains configuration for FPGA accelerator
type FPGAConfig struct {
	Type           FPGAType
	DeviceID       string
	PCIeSlot       string
	MemorySize     int64 // in bytes
	ClockFrequency int   // in MHz

	// AWS F2 specific
	AGFI         string // Amazon FPGA Image ID
	InstanceType string // f2.xlarge, f2.2xlarge, etc.

	// AMD Versal specific
	AIEngines   int // Number of AI engines
	DSPSlices   int // Number of DSP slices
	DDRChannels int // Number of DDR channels

	// Performance tuning
	BatchSize     int
	PipelineDepth int
	DMAChannels   int

	// Network offload
	EnableTCP  bool
	EnableRDMA bool
	Enable100G bool
}

// FPGAStats contains runtime statistics
type FPGAStats struct {
	OrdersProcessed    uint64
	TradesMatched      uint64
	AverageLatencyNs   uint64
	P99LatencyNs       uint64
	ThroughputOps      uint64
	PowerUsageWatts    float64
	TemperatureCelsius float64
	PCIeBandwidthGbps  float64
	MemoryUsageBytes   int64
	ErrorCount         uint64
}

// FPGAAccelerator interface for different FPGA implementations
type FPGAAccelerator interface {
	// Initialization
	Initialize(config *FPGAConfig) error
	Shutdown() error
	Reset() error

	// Health monitoring
	IsHealthy() bool
	GetStats() *FPGAStats
	GetTemperature() float64
	GetPowerUsage() float64

	// Order operations
	ProcessOrder(order *FPGAOrder) (*FPGAResult, error)
	BatchProcessOrders(orders []*FPGAOrder) ([]*FPGAResult, error)
	CancelOrder(orderID uint64) error

	// Market data
	UpdateOrderBook(symbol string, bids, asks []PriceLevel) error
	GetOrderBook(symbol string) (*OrderBookSnapshot, error)

	// Configuration
	LoadBitstream(path string) error
	ReconfigurePartial(region int, bitstream []byte) error
	SetClockFrequency(mhz int) error

	// DMA operations
	AllocateDMABuffer(size int) (uintptr, error)
	FreeDMABuffer(addr uintptr) error
	DMATransfer(src, dst uintptr, size int) error
}

// FPGAOrder represents an order optimized for FPGA processing
type FPGAOrder struct {
	OrderID   uint64
	Symbol    uint32 // Symbol ID for FPGA lookup
	Side      uint8  // 0=buy, 1=sell
	Type      uint8  // 0=limit, 1=market, etc.
	Price     uint64 // Fixed-point representation
	Quantity  uint64 // Fixed-point representation
	Timestamp uint64 // Nanoseconds since epoch
	UserID    uint32
	Flags     uint32 // Various flags packed
}

// FPGAResult represents the result of FPGA processing
type FPGAResult struct {
	OrderID       uint64
	Status        uint8 // 0=accepted, 1=rejected, 2=filled, 3=partial
	ExecutedQty   uint64
	ExecutedPrice uint64
	TradeID       uint64
	MatchLatency  uint32 // Nanoseconds
	Timestamp     uint64
}

// PriceLevel for order book
type PriceLevel struct {
	Price    uint64
	Quantity uint64
	Orders   uint32
}

// OrderBookSnapshot from FPGA
type OrderBookSnapshot struct {
	Symbol    string
	Timestamp uint64
	Bids      []PriceLevel
	Asks      []PriceLevel
	LastTrade uint64
	Volume    uint64
}

// FPGAManager manages multiple FPGA devices
type FPGAManager struct {
	devices map[string]FPGAAccelerator
	config  map[string]*FPGAConfig
	mu      sync.RWMutex

	// Load balancing
	roundRobin int

	// Monitoring
	stats map[string]*FPGAStats
}

// NewFPGAManager creates a new FPGA manager
func NewFPGAManager() *FPGAManager {
	return &FPGAManager{
		devices: make(map[string]FPGAAccelerator),
		config:  make(map[string]*FPGAConfig),
		stats:   make(map[string]*FPGAStats),
	}
}

// AddDevice adds an FPGA device to the manager
func (m *FPGAManager) AddDevice(id string, config *FPGAConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.devices[id]; exists {
		return fmt.Errorf("device %s already exists", id)
	}

	var device FPGAAccelerator

	switch config.Type {
	case FPGATypeAMDVersal:
		device = NewAMDVersalAccelerator()
	case FPGATypeAWSF2:
		device = NewAWSF2Accelerator()
	case FPGATypeIntelStratix:
		device = NewIntelStratixAccelerator()
	case FPGATypeXilinxAlveo:
		device = NewXilinxAlveoAccelerator()
	default:
		return fmt.Errorf("unsupported FPGA type: %v", config.Type)
	}

	if err := device.Initialize(config); err != nil {
		return fmt.Errorf("failed to initialize device: %w", err)
	}

	m.devices[id] = device
	m.config[id] = config

	return nil
}

// RemoveDevice removes an FPGA device
func (m *FPGAManager) RemoveDevice(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	device, exists := m.devices[id]
	if !exists {
		return fmt.Errorf("device %s not found", id)
	}

	if err := device.Shutdown(); err != nil {
		return fmt.Errorf("failed to shutdown device: %w", err)
	}

	delete(m.devices, id)
	delete(m.config, id)
	delete(m.stats, id)

	return nil
}

// ProcessOrder routes an order to the best available FPGA
func (m *FPGAManager) ProcessOrder(order *FPGAOrder) (*FPGAResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.devices) == 0 {
		return nil, errors.New("no FPGA devices available")
	}

	// Simple round-robin load balancing
	deviceIDs := make([]string, 0, len(m.devices))
	for id := range m.devices {
		deviceIDs = append(deviceIDs, id)
	}

	selectedID := deviceIDs[m.roundRobin%len(deviceIDs)]
	m.roundRobin++

	device := m.devices[selectedID]

	// Update stats
	start := time.Now()
	result, err := device.ProcessOrder(order)
	latency := time.Since(start).Nanoseconds()

	if stats, ok := m.stats[selectedID]; ok {
		stats.OrdersProcessed++
		stats.AverageLatencyNs = (stats.AverageLatencyNs + uint64(latency)) / 2
		if result != nil && result.Status == 2 { // Filled
			stats.TradesMatched++
		}
	}

	return result, err
}

// BatchProcessOrders processes multiple orders in parallel across FPGAs
func (m *FPGAManager) BatchProcessOrders(orders []*FPGAOrder) ([]*FPGAResult, error) {
	m.mu.RLock()
	numDevices := len(m.devices)
	if numDevices == 0 {
		m.mu.RUnlock()
		return nil, errors.New("no FPGA devices available")
	}

	// Distribute orders across devices
	deviceBatches := make(map[string][]*FPGAOrder)
	deviceIDs := make([]string, 0, numDevices)
	for id := range m.devices {
		deviceIDs = append(deviceIDs, id)
		deviceBatches[id] = make([]*FPGAOrder, 0)
	}
	m.mu.RUnlock()

	// Round-robin distribution
	for i, order := range orders {
		deviceID := deviceIDs[i%numDevices]
		deviceBatches[deviceID] = append(deviceBatches[deviceID], order)
	}

	// Process in parallel
	type result struct {
		results []*FPGAResult
		err     error
	}

	resultChan := make(chan result, numDevices)
	var wg sync.WaitGroup

	m.mu.RLock()
	for deviceID, batch := range deviceBatches {
		if len(batch) == 0 {
			continue
		}

		wg.Add(1)
		go func(id string, orders []*FPGAOrder) {
			defer wg.Done()

			device := m.devices[id]
			res, err := device.BatchProcessOrders(orders)
			resultChan <- result{results: res, err: err}
		}(deviceID, batch)
	}
	m.mu.RUnlock()

	wg.Wait()
	close(resultChan)

	// Collect results
	allResults := make([]*FPGAResult, 0, len(orders))
	for res := range resultChan {
		if res.err != nil {
			return nil, res.err
		}
		allResults = append(allResults, res.results...)
	}

	return allResults, nil
}

// GetStats returns statistics for all devices
func (m *FPGAManager) GetStats() map[string]*FPGAStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := make(map[string]*FPGAStats)
	for id, device := range m.devices {
		stats[id] = device.GetStats()
	}

	return stats
}

// HealthCheck checks health of all devices
func (m *FPGAManager) HealthCheck() map[string]bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	health := make(map[string]bool)
	for id, device := range m.devices {
		health[id] = device.IsHealthy()
	}

	return health
}

// Monitoring functions
func (m *FPGAManager) MonitorDevices(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		m.mu.RLock()
		for id, device := range m.devices {
			stats := device.GetStats()
			m.stats[id] = stats

			// Check for issues
			if stats.TemperatureCelsius > 85 {
				fmt.Printf("WARNING: FPGA %s temperature high: %.1f°C\n",
					id, stats.TemperatureCelsius)
			}

			if stats.ErrorCount > 100 {
				fmt.Printf("WARNING: FPGA %s has %d errors\n",
					id, stats.ErrorCount)
			}

			if !device.IsHealthy() {
				fmt.Printf("ERROR: FPGA %s is unhealthy\n", id)
			}
		}
		m.mu.RUnlock()
	}
}
