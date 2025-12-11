package metric

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewRegistry(t *testing.T) {
	reg := NewRegistry()
	assert.NotNil(t, reg)
	assert.NotNil(t, reg.counters)
	assert.NotNil(t, reg.histograms)
}

func TestCounterInc(t *testing.T) {
	counter := &Counter{}
	assert.Equal(t, int64(0), counter.Count())

	counter.Inc(1)
	assert.Equal(t, int64(1), counter.Count())

	counter.Inc(5)
	assert.Equal(t, int64(6), counter.Count())

	counter.Inc(10)
	assert.Equal(t, int64(16), counter.Count())
}

func TestCounterConcurrency(t *testing.T) {
	counter := &Counter{}
	var wg sync.WaitGroup

	// 100 goroutines each incrementing 1000 times
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				counter.Inc(1)
			}
		}()
	}

	wg.Wait()
	assert.Equal(t, int64(100000), counter.Count())
}

func TestRegistryCounter(t *testing.T) {
	reg := NewRegistry()

	// Get counter creates new one
	c1 := reg.Counter("requests_total")
	assert.NotNil(t, c1)
	assert.Equal(t, int64(0), c1.Count())

	// Increment it
	c1.Inc(5)
	assert.Equal(t, int64(5), c1.Count())

	// Get same counter returns same instance
	c2 := reg.Counter("requests_total")
	assert.Equal(t, int64(5), c2.Count())

	// Different name creates different counter
	c3 := reg.Counter("errors_total")
	assert.Equal(t, int64(0), c3.Count())
}

func TestRegistryCounterConcurrency(t *testing.T) {
	reg := NewRegistry()
	var wg sync.WaitGroup

	// Multiple goroutines getting and incrementing same counter
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counter := reg.Counter("concurrent_counter")
			for j := 0; j < 100; j++ {
				counter.Inc(1)
			}
		}()
	}

	wg.Wait()
	assert.Equal(t, int64(5000), reg.Counter("concurrent_counter").Count())
}

func TestHistogramObserve(t *testing.T) {
	h := &Histogram{values: make([]float64, 0)}

	h.Observe(1.0)
	h.Observe(2.0)
	h.Observe(3.0)

	assert.Len(t, h.values, 3)
}

func TestHistogramPercentile(t *testing.T) {
	h := &Histogram{values: make([]float64, 0)}

	// Empty histogram
	assert.Equal(t, 0.0, h.Percentile(0.5))

	// Add values
	h.Observe(10.0)
	h.Observe(20.0)
	h.Observe(30.0)

	// Current implementation returns average
	p50 := h.Percentile(0.5)
	assert.Equal(t, 20.0, p50) // Average of 10, 20, 30
}

func TestHistogramConcurrency(t *testing.T) {
	h := &Histogram{values: make([]float64, 0, 10000)}
	var wg sync.WaitGroup

	// Multiple goroutines observing values
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(val float64) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				h.Observe(val)
			}
		}(float64(i))
	}

	wg.Wait()
	assert.Len(t, h.values, 10000)
}

func TestRegistryHistogram(t *testing.T) {
	reg := NewRegistry()

	// Get histogram creates new one
	h1 := reg.Histogram("latency_ms")
	assert.NotNil(t, h1)

	// Observe values
	h1.Observe(10.0)
	h1.Observe(20.0)

	// Get same histogram returns same instance
	h2 := reg.Histogram("latency_ms")
	h2.Observe(30.0)

	// Should have all 3 values
	assert.Equal(t, 20.0, h1.Percentile(0.5)) // Average: (10+20+30)/3

	// Different name creates different histogram
	h3 := reg.Histogram("other_latency")
	assert.Equal(t, 0.0, h3.Percentile(0.5))
}

func TestRegistryMultipleMetrics(t *testing.T) {
	reg := NewRegistry()

	// Create various metrics
	requestCounter := reg.Counter("http_requests_total")
	errorCounter := reg.Counter("http_errors_total")
	latencyHist := reg.Histogram("http_request_duration_ms")
	sizeHist := reg.Histogram("http_response_size_bytes")

	// Use them
	requestCounter.Inc(100)
	errorCounter.Inc(5)
	latencyHist.Observe(50.0)
	latencyHist.Observe(100.0)
	sizeHist.Observe(1024.0)

	// Verify
	assert.Equal(t, int64(100), requestCounter.Count())
	assert.Equal(t, int64(5), errorCounter.Count())
	assert.Equal(t, 75.0, latencyHist.Percentile(0.5))
	assert.Equal(t, 1024.0, sizeHist.Percentile(0.5))
}

func TestCounterNegativeIncrement(t *testing.T) {
	counter := &Counter{}
	counter.Inc(10)
	counter.Inc(-3) // Decrement
	assert.Equal(t, int64(7), counter.Count())
}

func TestRegistryGetOrCreate(t *testing.T) {
	reg := NewRegistry()

	// First call creates
	c1 := reg.Counter("test")
	c1.Inc(5)

	// Second call returns existing
	c2 := reg.Counter("test")
	assert.Equal(t, int64(5), c2.Count())

	// Same for histogram
	h1 := reg.Histogram("test_hist")
	h1.Observe(100.0)

	h2 := reg.Histogram("test_hist")
	assert.Equal(t, 100.0, h2.Percentile(0.5))
}

func TestHistogramWithManyValues(t *testing.T) {
	h := &Histogram{values: make([]float64, 0, 1000)}

	// Add 1000 values from 1 to 1000
	for i := 1; i <= 1000; i++ {
		h.Observe(float64(i))
	}

	// Average should be 500.5
	avg := h.Percentile(0.5)
	assert.InDelta(t, 500.5, avg, 0.001)
}

func BenchmarkCounterInc(b *testing.B) {
	counter := &Counter{}
	for i := 0; i < b.N; i++ {
		counter.Inc(1)
	}
}

func BenchmarkHistogramObserve(b *testing.B) {
	h := &Histogram{values: make([]float64, 0, b.N)}
	for i := 0; i < b.N; i++ {
		h.Observe(float64(i))
	}
}

func BenchmarkRegistryCounter(b *testing.B) {
	reg := NewRegistry()
	for i := 0; i < b.N; i++ {
		reg.Counter("test").Inc(1)
	}
}
