package metrics

import (
	"context"
	"net/http"
	"runtime"
	"time"

	"github.com/luxfi/log"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// LXMetrics uses native luxfi/metric for Prometheus integration
type LXMetrics struct {
	namespace string
	registry  *prometheus.Registry
	gatherer  prometheus.Gatherer
	logger    log.Logger

	// Order book metrics
	ordersProcessed prometheus.Counter
	tradesExecuted  prometheus.Counter
	orderBookDepth  prometheus.GaugeVec
	matchingLatency prometheus.Histogram
	consensusRounds prometheus.Counter
	blockHeight     prometheus.Gauge

	// Network metrics
	zmqMessagesIn  prometheus.Counter
	zmqMessagesOut prometheus.Counter
	natsPublished  prometheus.Counter
	natsReceived   prometheus.Counter

	// System metrics
	memoryUsage    prometheus.Gauge
	goroutines     prometheus.Gauge
	consensusNodes prometheus.Gauge
}

// NewLXMetrics creates metrics using luxfi packages
func NewLXMetrics(namespace string) (*LXMetrics, error) {
	// Use luxfi/log for logging
	logger := log.Root().New("module", "metrics")
	logger.Info("Initializing LX metrics")

	// Create Prometheus registry
	registry := prometheus.NewRegistry()

	m := &LXMetrics{
		namespace: namespace,
		registry:  registry,
		gatherer:  registry,
		logger:    logger,

		// Initialize order metrics
		ordersProcessed: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "orders_processed_total",
			Help:      "Total number of orders processed",
		}),

		tradesExecuted: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "trades_executed_total",
			Help:      "Total number of trades executed",
		}),

		orderBookDepth: *prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "orderbook_depth",
			Help:      "Current order book depth by side",
		}, []string{"symbol", "side"}),

		matchingLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "matching_latency_nanoseconds",
			Help:      "Order matching latency in nanoseconds",
			Buckets:   []float64{10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000},
		}),

		consensusRounds: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "consensus_rounds_total",
			Help:      "Total consensus rounds completed",
		}),

		blockHeight: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "block_height",
			Help:      "Current block height",
		}),

		// Network metrics
		zmqMessagesIn: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "zmq_messages_received_total",
			Help:      "Total ZeroMQ messages received",
		}),

		zmqMessagesOut: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "zmq_messages_sent_total",
			Help:      "Total ZeroMQ messages sent",
		}),

		natsPublished: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "nats_messages_published_total",
			Help:      "Total NATS messages published",
		}),

		natsReceived: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "nats_messages_received_total",
			Help:      "Total NATS messages received",
		}),

		// System metrics
		memoryUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "memory_usage_bytes",
			Help:      "Current memory usage in bytes",
		}),

		goroutines: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "goroutines_count",
			Help:      "Current number of goroutines",
		}),

		consensusNodes: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "consensus_nodes_active",
			Help:      "Number of active consensus nodes",
		}),
	}

	// Register all metrics
	registry.MustRegister(
		m.ordersProcessed,
		m.tradesExecuted,
		m.orderBookDepth,
		m.matchingLatency,
		m.consensusRounds,
		m.blockHeight,
		m.zmqMessagesIn,
		m.zmqMessagesOut,
		m.natsPublished,
		m.natsReceived,
		m.memoryUsage,
		m.goroutines,
		m.consensusNodes,
	)

	// No external gatherer registration needed

	logger.Info("LX metrics initialized successfully")
	return m, nil
}

// StartServer starts Prometheus metrics server
func (m *LXMetrics) StartServer(port string) error {
	m.logger.Info("Starting Prometheus metrics server", "port", port)

	// Expose standard Prometheus endpoint
	http.Handle("/metrics", promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{}))

	go func() {
		if err := http.ListenAndServe(":"+port, nil); err != nil {
			m.logger.Error("Metrics server failed", "error", err)
		}
	}()

	m.logger.Info("Prometheus metrics available",
		"endpoint", "http://localhost:"+port+"/metrics")

	return nil
}

// RecordOrder records an order processed
func (m *LXMetrics) RecordOrder() {
	m.ordersProcessed.Inc()
}

// RecordTrade records a trade executed
func (m *LXMetrics) RecordTrade() {
	m.tradesExecuted.Inc()
}

// RecordMatchingLatency records order matching latency
func (m *LXMetrics) RecordMatchingLatency(nanoseconds float64) {
	m.matchingLatency.Observe(nanoseconds)
}

// UpdateOrderBookDepth updates order book depth metrics
func (m *LXMetrics) UpdateOrderBookDepth(symbol, side string, depth float64) {
	m.orderBookDepth.WithLabelValues(symbol, side).Set(depth)
}

// RecordZMQMessage records ZeroMQ message metrics
func (m *LXMetrics) RecordZMQMessage(direction string) {
	switch direction {
	case "in":
		m.zmqMessagesIn.Inc()
	case "out":
		m.zmqMessagesOut.Inc()
	}
}

// RecordNATSMessage records NATS message metrics
func (m *LXMetrics) RecordNATSMessage(direction string) {
	switch direction {
	case "published":
		m.natsPublished.Inc()
	case "received":
		m.natsReceived.Inc()
	}
}

// UpdateBlockHeight updates current block height
func (m *LXMetrics) UpdateBlockHeight(height float64) {
	m.blockHeight.Set(height)
}

// UpdateConsensusNodes updates active consensus node count
func (m *LXMetrics) UpdateConsensusNodes(count float64) {
	m.consensusNodes.Set(count)
}

// CollectSystemMetrics collects system-level metrics
func (m *LXMetrics) CollectSystemMetrics(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Collect runtime stats
			var memStats runtime.MemStats
			runtime.ReadMemStats(&memStats)
			m.memoryUsage.Set(float64(memStats.Alloc))
			m.goroutines.Set(float64(runtime.NumGoroutine()))
		}
	}
}

// LogMetrics logs current metrics using luxfi/log
func (m *LXMetrics) LogMetrics() {
	// Get current metric values as a formatted string
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	m.logger.Info("Current metrics snapshot",
		"memory_mb", memStats.Alloc/1024/1024,
		"goroutines", runtime.NumGoroutine(),
	)
}
