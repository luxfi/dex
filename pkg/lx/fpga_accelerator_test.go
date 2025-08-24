package lx

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Test FPGA accelerator initialization and basic functionality
func TestFPGAAccelerator(t *testing.T) {
	t.Run("NewFPGAAccelerator", func(t *testing.T) {
		accelerator := NewFPGAAccelerator()
		assert.NotNil(t, accelerator)
	})

	accelerator := NewFPGAAccelerator()

	t.Run("detectHardware", func(t *testing.T) {
		// Should not panic - mock environment won't have FPGA
		accelerator.detectHardware()
		assert.False(t, accelerator.enabled)
	})

	t.Run("ProcessOrder", func(t *testing.T) {
		order := &Order{
			ID:     1,
			Symbol: "BTC-USD",
			Side:   Buy,
			Price:  50000.0,
			Size:   1.0,
		}

		// Should return error when FPGA not available
		result, err := accelerator.ProcessOrder(order)
		assert.Nil(t, result)
		assert.Error(t, err)
	})

	t.Run("ProcessOrderBatch", func(t *testing.T) {
		orders := []*Order{
			{ID: 1, Symbol: "BTC-USD", Side: Buy, Price: 50000, Size: 1.0},
			{ID: 2, Symbol: "ETH-USD", Side: Sell, Price: 3000, Size: 2.0},
		}

		results, err := accelerator.ProcessOrderBatch(orders)
		assert.Nil(t, results)
		assert.Error(t, err)
	})

	t.Run("WireToWireLatency", func(t *testing.T) {
		packet := []byte("test packet")
		_, latencyNs, err := accelerator.WireToWireLatency(packet)
		assert.Equal(t, uint64(0), latencyNs)
		assert.Error(t, err)
	})

	t.Run("initializePipelines", func(t *testing.T) {
		// Should not panic
		accelerator.initializePipelines()
	})

	t.Run("initializeMemoryPools", func(t *testing.T) {
		// Should not panic
		accelerator.initializeMemoryPools()
	})

	t.Run("startDMAChannels", func(t *testing.T) {
		// Should handle gracefully when no FPGA available
		accelerator.startDMAChannels()
	})

	t.Run("selectDMAChannel", func(t *testing.T) {
		// Function returns *DMAChannel, will create default channel
		channel := accelerator.selectDMAChannel()
		assert.NotNil(t, channel) // Returns first channel
	})

	t.Run("sendDMA", func(t *testing.T) {
		data := []byte("test data")
		// Should not panic - nil channel will be handled gracefully
		err := accelerator.sendDMA(nil, data)
		assert.NoError(t, err) // Mock implementation returns nil
	})

	t.Run("receiveDMA", func(t *testing.T) {
		// Should handle gracefully
		data, err := accelerator.receiveDMA(nil)
		assert.Nil(t, data)
		assert.NoError(t, err) // Mock implementation
	})

	t.Run("encodeOrder", func(t *testing.T) {
		order := &Order{
			ID:     123,
			Symbol: "BTC-USD",
			Side:   Buy,
			Price:  50000,
			Size:   1.5,
		}
		
		encoded := accelerator.encodeOrder(order)
		assert.NotNil(t, encoded)
		assert.Greater(t, len(encoded), 0)
	})

	t.Run("decodeResult", func(t *testing.T) {
		// Create some mock result data
		data := make([]byte, 32)
		data[0] = 1 // success flag
		
		result, err := accelerator.decodeResult(data)
		assert.NotNil(t, result)
		assert.NoError(t, err)
	})

	t.Run("encodeReject", func(t *testing.T) {
		encoded := accelerator.encodeReject("Invalid order")
		assert.NotNil(t, encoded)
		assert.Greater(t, len(encoded), 0)
	})

	t.Run("getPTPTimestamp", func(t *testing.T) {
		timestamp := accelerator.getPTPTimestamp()
		assert.Greater(t, timestamp, uint64(0))
	})

	t.Run("GetMetrics", func(t *testing.T) {
		metrics := accelerator.GetMetrics()
		assert.NotNil(t, metrics)
	})
}


// Benchmark FPGA accelerator functions
func BenchmarkFPGAAccelerator(b *testing.B) {
	accelerator := NewFPGAAccelerator()
	
	order := &Order{
		ID:     1,
		Symbol: "BTC-USD", 
		Side:   Buy,
		Price:  50000,
		Size:   1.0,
	}

	b.Run("ProcessOrder", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			accelerator.ProcessOrder(order)
		}
	})

	b.Run("encodeOrder", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			accelerator.encodeOrder(order)
		}
	})

	b.Run("WireToWireLatency", func(b *testing.B) {
		packet := []byte("test packet")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			accelerator.WireToWireLatency(packet)
		}
	})
}