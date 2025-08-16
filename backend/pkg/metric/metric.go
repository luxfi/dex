package metric

import (
	"sync"
	"sync/atomic"
)

// Registry holds all metrics
type Registry struct {
	counters   map[string]*Counter
	histograms map[string]*Histogram
	mu         sync.RWMutex
}

// NewRegistry creates a new metrics registry
func NewRegistry() *Registry {
	return &Registry{
		counters:   make(map[string]*Counter),
		histograms: make(map[string]*Histogram),
	}
}

// Counter is a simple counter metric
type Counter struct {
	value int64
}

// Inc increments the counter
func (c *Counter) Inc(delta int64) {
	atomic.AddInt64(&c.value, delta)
}

// Count returns the current count
func (c *Counter) Count() int64 {
	return atomic.LoadInt64(&c.value)
}

// Counter gets or creates a counter
func (r *Registry) Counter(name string) *Counter {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if c, exists := r.counters[name]; exists {
		return c
	}
	
	c := &Counter{}
	r.counters[name] = c
	return c
}

// Histogram tracks distribution of values
type Histogram struct {
	values []float64
	mu     sync.Mutex
}

// Observe adds a value to the histogram
func (h *Histogram) Observe(value float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.values = append(h.values, value)
}

// Percentile calculates the percentile value
func (h *Histogram) Percentile(p float64) float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	if len(h.values) == 0 {
		return 0
	}
	
	// Simple approximation - just return average for now
	var sum float64
	for _, v := range h.values {
		sum += v
	}
	return sum / float64(len(h.values))
}

// Histogram gets or creates a histogram
func (r *Registry) Histogram(name string) *Histogram {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if h, exists := r.histograms[name]; exists {
		return h
	}
	
	h := &Histogram{
		values: make([]float64, 0, 1000),
	}
	r.histograms[name] = h
	return h
}