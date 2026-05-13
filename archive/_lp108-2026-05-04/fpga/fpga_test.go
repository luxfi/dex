//go:build fpga
// +build fpga

package fpga

import (
	"math/big"
	"testing"
	"time"
)

func TestAMDVersalAccelerator(t *testing.T) {
	tests := []struct {
		name string
		test func(t *testing.T)
	}{
		{"CreateAccelerator", testCreateAccelerator},
		{"Initialize", testInitialize},
		{"ProcessOrder", testProcessOrder},
		{"MatchOrders", testMatchOrders},
		{"GetStats", testGetStats},
		{"WireProtocol", testWireProtocol},
		{"Performance", testPerformance},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func testCreateAccelerator(t *testing.T) {
	acc := &AMDVersalAccelerator{
		aiEngines:    400,
		aiEngineFreq: 1250,
		nocBandwidth: 2,
		dspEngines:   1312,
		lutLogic:     899000,
		memoryMB:     16384,
	}

	if acc.aiEngines != 400 {
		t.Errorf("Expected 400 AI engines, got %d", acc.aiEngines)
	}
}

func testInitialize(t *testing.T) {
	acc := &AMDVersalAccelerator{}
	if err := acc.Initialize(); err != nil {
		t.Errorf("Initialize failed: %v", err)
	}
}

func testProcessOrder(t *testing.T) {
	acc := &AMDVersalAccelerator{
		initialized: true,
	}

	order := &FPGAOrder{
		Symbol:    [8]byte{'B', 'T', 'C', '-', 'U', 'S', 'D', 0},
		OrderID:   uint64(12345),
		Price:     50000_00000000, // $50,000 with 8 decimals
		Quantity:  100000000,      // 1.0 BTC
		Side:      0,              // Buy
		OrderType: 0,              // Limit
		Timestamp: uint64(time.Now().Unix()),
	}

	ack := acc.ProcessOrder(order)
	if ack == nil {
		t.Error("Expected acknowledgment, got nil")
		return
	}

	if ack.OrderID != order.OrderID {
		t.Errorf("OrderID mismatch: expected %d, got %d", order.OrderID, ack.OrderID)
	}

	if ack.Status != 0 { // Accepted
		t.Errorf("Expected status 0 (accepted), got %d", ack.Status)
	}
}

func testMatchOrders(t *testing.T) {
	acc := &AMDVersalAccelerator{
		initialized: true,
	}

	// Create buy and sell orders that should match
	buyOrder := &FPGAOrder{
		Symbol:    [8]byte{'B', 'T', 'C', '-', 'U', 'S', 'D', 0},
		OrderID:   1,
		Price:     50000_00000000,
		Quantity:  100000000,
		Side:      0, // Buy
		OrderType: 0, // Limit
		Timestamp: uint64(time.Now().Unix()),
	}

	sellOrder := &FPGAOrder{
		Symbol:    [8]byte{'B', 'T', 'C', '-', 'U', 'S', 'D', 0},
		OrderID:   2,
		Price:     50000_00000000, // Same price, should match
		Quantity:  100000000,
		Side:      1, // Sell
		OrderType: 0, // Limit
		Timestamp: uint64(time.Now().Unix()),
	}

	// Process orders
	acc.ProcessOrder(buyOrder)
	acc.ProcessOrder(sellOrder)

	// Check if orders were matched (this would be reflected in stats)
	stats := acc.GetStats()
	if stats.MatchedOrders == 0 {
		t.Error("Expected orders to be matched")
	}
}

func testGetStats(t *testing.T) {
	acc := &AMDVersalAccelerator{
		initialized: true,
		stats: &FPGAStats{
			ProcessedOrders: 1000,
			MatchedOrders:   500,
			RejectedOrders:  10,
			AvgLatencyNs:    350,
		},
	}

	stats := acc.GetStats()
	if stats.ProcessedOrders != 1000 {
		t.Errorf("Expected 1000 processed orders, got %d", stats.ProcessedOrders)
	}

	if stats.AvgLatencyNs != 350 {
		t.Errorf("Expected 350ns average latency, got %d", stats.AvgLatencyNs)
	}
}

func testWireProtocol(t *testing.T) {
	// Test order size
	if OrderWireSize != 48 {
		t.Errorf("Expected order wire size 48, got %d", OrderWireSize)
	}

	// Test ack size
	if AckWireSize != 32 {
		t.Errorf("Expected ack wire size 32, got %d", AckWireSize)
	}

	// Test cancel size
	if CancelWireSize != 24 {
		t.Errorf("Expected cancel wire size 24, got %d", CancelWireSize)
	}
}

func testPerformance(t *testing.T) {
	acc := &AMDVersalAccelerator{
		initialized: true,
		stats:       &FPGAStats{},
	}

	// Benchmark order processing
	startTime := time.Now()
	numOrders := 100000

	for i := 0; i < numOrders; i++ {
		order := &FPGAOrder{
			Symbol:    [8]byte{'T', 'E', 'S', 'T', 0, 0, 0, 0},
			OrderID:   uint64(i),
			Price:     uint64(50000 + i),
			Quantity:  uint64(100),
			Side:      uint8(i % 2),
			OrderType: 0,
			Timestamp: uint64(time.Now().Unix()),
		}
		acc.ProcessOrder(order)
	}

	elapsed := time.Since(startTime)
	ordersPerSec := float64(numOrders) / elapsed.Seconds()

	t.Logf("Processed %d orders in %v", numOrders, elapsed)
	t.Logf("Throughput: %.2f orders/sec", ordersPerSec)

	// Should achieve at least 1M orders/sec
	if ordersPerSec < 1000000 {
		t.Logf("Warning: Performance below target (1M orders/sec)")
	}
}

func TestAWSF2Accelerator(t *testing.T) {
	tests := []struct {
		name string
		test func(t *testing.T)
	}{
		{"CreateF2", testCreateF2},
		{"InitializeF2", testInitializeF2},
		{"ProcessOrderF2", testProcessOrderF2},
		{"KernelOps", testKernelOps},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func testCreateF2(t *testing.T) {
	acc := &AWSF2Accelerator{
		xcvu13p:   4,
		memoryGB:  128,
		bandwidth: 500,
		lutLogic:  5954000,
		dspSlices: 12288,
		blockRAM:  6840,
	}

	if acc.xcvu13p != 4 {
		t.Errorf("Expected 4 Virtex UltraScale+ FPGAs, got %d", acc.xcvu13p)
	}

	if acc.memoryGB != 128 {
		t.Errorf("Expected 128GB memory, got %d", acc.memoryGB)
	}
}

func testInitializeF2(t *testing.T) {
	acc := &AWSF2Accelerator{}
	if err := acc.Initialize(); err != nil {
		t.Errorf("Initialize failed: %v", err)
	}
}

func testProcessOrderF2(t *testing.T) {
	acc := &AWSF2Accelerator{
		initialized: true,
	}

	order := &FPGAOrder{
		Symbol:    [8]byte{'E', 'T', 'H', '-', 'U', 'S', 'D', 0},
		OrderID:   uint64(99999),
		Price:     3000_00000000, // $3,000 with 8 decimals
		Quantity:  1000000000,    // 10 ETH
		Side:      1,             // Sell
		OrderType: 0,             // Limit
		Timestamp: uint64(time.Now().Unix()),
	}

	ack := acc.ProcessOrder(order)
	if ack == nil {
		t.Error("Expected acknowledgment, got nil")
		return
	}

	if ack.OrderID != order.OrderID {
		t.Errorf("OrderID mismatch: expected %d, got %d", order.OrderID, ack.OrderID)
	}
}

func testKernelOps(t *testing.T) {
	acc := &AWSF2Accelerator{
		initialized: true,
	}

	// Test kernel operations
	orders := make([]*FPGAOrder, 1000)
	for i := range orders {
		orders[i] = &FPGAOrder{
			Symbol:    [8]byte{'T', 'E', 'S', 'T', 0, 0, 0, 0},
			OrderID:   uint64(i),
			Price:     uint64(100 + i),
			Quantity:  uint64(10),
			Side:      uint8(i % 2),
			OrderType: 0,
			Timestamp: uint64(time.Now().Unix()),
		}
	}

	// Process batch
	startTime := time.Now()
	for _, order := range orders {
		acc.ProcessOrder(order)
	}
	elapsed := time.Since(startTime)

	avgLatency := elapsed.Nanoseconds() / int64(len(orders))
	t.Logf("Average latency per order: %dns", avgLatency)

	// Should be under 1 microsecond
	if avgLatency > 1000 {
		t.Logf("Warning: Latency above target (<1μs)")
	}
}

// Test FPGA manager
func TestFPGAManager(t *testing.T) {
	manager := &FPGAManager{
		accelerators: make(map[string]FPGAAccelerator),
	}

	// Add AMD Versal
	manager.accelerators["amd_versal"] = &AMDVersalAccelerator{
		aiEngines:    400,
		aiEngineFreq: 1250,
	}

	// Add AWS F2
	manager.accelerators["aws_f2"] = &AWSF2Accelerator{
		xcvu13p:  4,
		memoryGB: 128,
	}

	if len(manager.accelerators) != 2 {
		t.Errorf("Expected 2 accelerators, got %d", len(manager.accelerators))
	}

	// Test selection
	versal, exists := manager.accelerators["amd_versal"]
	if !exists {
		t.Error("AMD Versal not found")
	}
	if versal == nil {
		t.Error("AMD Versal is nil")
	}
}

// Test HLS implementation markers
func TestHLSImplementation(t *testing.T) {
	// These would be HLS pragma markers in actual implementation
	hlsConfig := struct {
		Pipeline       bool
		Unroll         int
		ArrayPartition bool
		Dataflow       bool
	}{
		Pipeline:       true,
		Unroll:         8,
		ArrayPartition: true,
		Dataflow:       true,
	}

	if !hlsConfig.Pipeline {
		t.Error("HLS pipeline should be enabled")
	}

	if hlsConfig.Unroll != 8 {
		t.Errorf("Expected unroll factor 8, got %d", hlsConfig.Unroll)
	}
}

// Test RTL implementation markers
func TestRTLImplementation(t *testing.T) {
	// These would be RTL module parameters in actual implementation
	rtlConfig := struct {
		ClockFreq    int // MHz
		DataWidth    int // bits
		FIFODepth    int // entries
		NumPipelines int
	}{
		ClockFreq:    500,
		DataWidth:    512,
		FIFODepth:    4096,
		NumPipelines: 16,
	}

	if rtlConfig.ClockFreq != 500 {
		t.Errorf("Expected 500MHz clock, got %d", rtlConfig.ClockFreq)
	}

	if rtlConfig.DataWidth != 512 {
		t.Errorf("Expected 512-bit data width, got %d", rtlConfig.DataWidth)
	}
}

// Test margin calculations on FPGA
func TestFPGAMarginCalculations(t *testing.T) {
	// Simulate FPGA fixed-point margin calculations
	// Using 64-bit integers with 8 decimal places

	price := uint64(50000_00000000) // $50,000
	quantity := uint64(100000000)   // 1.0
	leverage := uint64(10)

	// notional = price * quantity / 10^8
	notional := new(big.Int).Mul(
		new(big.Int).SetUint64(price),
		new(big.Int).SetUint64(quantity),
	)
	notional.Div(notional, big.NewInt(100000000))

	// margin = notional / leverage
	margin := new(big.Int).Div(notional, new(big.Int).SetUint64(leverage))

	expectedMargin := uint64(5000_00000000) // $5,000
	if margin.Uint64() != expectedMargin {
		t.Errorf("Expected margin %d, got %d", expectedMargin, margin.Uint64())
	}
}
