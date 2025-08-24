//go:build fpga
// +build fpga

package fpga

import (
	"time"
	"unsafe"
)

// FPGAOrderBook is hardware-accelerated order book
type FPGAOrderBook struct {
	symbol      string
	deviceMem   unsafe.Pointer // Direct FPGA memory mapping
	bidTree     uint64         // FPGA address of bid tree
	askTree     uint64         // FPGA address of ask tree
	matchEngine uint64         // FPGA matching engine instance
}

// FPGAStats tracks nanosecond-precision performance
type FPGAStats struct {
	// Core metrics
	OrdersProcessed uint64
	TradesExecuted  uint64
	TradesMatched   uint64
	ProcessedOrders uint64
	MatchedOrders   uint64
	RejectedOrders  uint64

	// Latency metrics
	LatencyNanos     uint64 // Running average in nanoseconds
	AverageLatencyNs uint64 // Average latency in nanoseconds
	AvgLatencyNs     uint64 // Alias for average latency
	MinLatencyNs     uint64
	MaxLatencyNs     uint64

	// Throughput metrics
	ThroughputMOps    float64 // Millions of operations/second
	NetworkGbps       float64 // Network throughput
	PCIeBandwidthGbps float64 // PCIe bandwidth utilization

	// Time tracking
	LastUpdate time.Time
}

// DMA Direction for FPGA transfers
type DMADirection int

const (
	DMAToDevice DMADirection = iota
	DMAFromDevice
	DMABidirectional
)

// Wire protocol sizes (fixed-size binary messages)
const (
	OrderWireSize  = 48 // 48 bytes per order
	AckWireSize    = 32 // 32 bytes per acknowledgment
	CancelWireSize = 24 // 24 bytes per cancel
	TradeWireSize  = 64 // 64 bytes per trade
)

// FPGAOrder represents an order in FPGA wire format
type FPGAOrder struct {
	Symbol    [8]byte // 8 bytes - fixed symbol
	OrderID   uint64  // 8 bytes
	Price     uint64  // 8 bytes - fixed point with 8 decimals
	Quantity  uint64  // 8 bytes
	Side      uint8   // 1 byte - 0=buy, 1=sell
	OrderType uint8   // 1 byte - 0=limit, 1=market, 2=stop
	Flags     uint16  // 2 bytes - various flags
	Timestamp uint64  // 8 bytes - unix nano
	UserID    uint32  // 4 bytes
	// Total: 48 bytes
}

// FPGAAck represents an acknowledgment in wire format
type FPGAAck struct {
	OrderID    uint64  // 8 bytes
	Status     uint8   // 1 byte - 0=accepted, 1=rejected, 2=cancelled
	RejectCode uint8   // 1 byte - reason if rejected
	Reserved   [6]byte // 6 bytes - padding
	Timestamp  uint64  // 8 bytes
	Sequence   uint64  // 8 bytes - sequence number
	// Total: 32 bytes
}

// FPGACancel represents a cancel request in wire format
type FPGACancel struct {
	OrderID   uint64 // 8 bytes
	UserID    uint32 // 4 bytes
	Reserved  uint32 // 4 bytes - padding
	Timestamp uint64 // 8 bytes
	// Total: 24 bytes
}

// FPGATrade represents a trade in wire format
type FPGATrade struct {
	TradeID     uint64 // 8 bytes
	BuyOrderID  uint64 // 8 bytes
	SellOrderID uint64 // 8 bytes
	Price       uint64 // 8 bytes - execution price
	Quantity    uint64 // 8 bytes
	BuyUserID   uint32 // 4 bytes
	SellUserID  uint32 // 4 bytes
	Timestamp   uint64 // 8 bytes
	Flags       uint32 // 4 bytes - maker/taker flags
	Reserved    uint32 // 4 bytes - padding
	// Total: 64 bytes
}

// FPGADevice represents the FPGA hardware accelerator
type FPGADevice struct {
	Type        string // "versal", "awsf2", "stratix10"
	PCIeAddress string
	Memory      uint64 // Device memory in bytes
	Frequency   uint32 // Clock frequency in MHz
	Lanes       uint32 // PCIe lanes
}

// fpgaOrder is internal representation for DMA
type fpgaOrder FPGAOrder

// FPGAAccelerator interface for different FPGA implementations
type FPGAAccelerator interface {
	// Initialization
	Initialize() error
	Shutdown() error
	Reset() error

	// Health monitoring
	IsHealthy() bool
	GetStats() *FPGAStats
	GetTemperature() float64
	GetPowerUsage() float64

	// Order operations
	ProcessOrder(order *FPGAOrder) *FPGAAck
	BatchProcessOrders(orders []*FPGAOrder) ([]*FPGAResult, error)
	CancelOrder(cancel *FPGACancel) *FPGAAck

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
	DMATransfer(src, dst uintptr, size int, direction DMADirection) error
}

// FPGAManager manages multiple FPGA accelerators
type FPGAManager struct {
	accelerators map[string]FPGAAccelerator
	stats        *FPGAStats
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

// OrderBookSnapshot represents a point-in-time order book state
type OrderBookSnapshot struct {
	Symbol    string
	Bids      []PriceLevel
	Asks      []PriceLevel
	Timestamp uint64
	Sequence  uint64
}

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

// FPGAType represents the type of FPGA device
type FPGAType int

const (
	FPGATypeNone FPGAType = iota
	FPGATypeAMDVersal
	FPGATypeAWSF2
	FPGATypeIntelStratix
	FPGATypeXilinxAlveo
)
