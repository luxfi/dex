# LX DEX Monitoring & Observability Guide

## Overview

Comprehensive monitoring strategy for the LX DEX platform covering metrics, logging, tracing, and alerting across all engine implementations.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Grafana                          │
│              (Dashboards & Alerts)                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│                  Prometheus                         │
│            (Metrics Collection)                     │
└──────┬───────────────┬───────────────┬─────────────┘
       │               │               │
┌──────▼────┐  ┌───────▼────┐  ┌──────▼─────┐
│ Go Engine │  │ C++ Engine │  │TS Engine   │
│  Metrics  │  │  Metrics   │  │  Metrics   │
└───────────┘  └────────────┘  └────────────┘
       │               │               │
┌──────┴───────────────┴───────────────┴─────────────┐
│                    Jaeger                          │
│              (Distributed Tracing)                 │
└─────────────────────────────────────────────────────┘
       │
┌─────────────────────────────────────────────────────┐
│                  Elasticsearch                      │
│              (Log Aggregation)                      │
└─────────────────────────────────────────────────────┘
```

## Metrics Collection

### Core Metrics

#### Order Processing Metrics
```go
// pkg/metrics/metrics.go
var (
    OrdersSubmitted = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "lx_orders_submitted_total",
            Help: "Total number of orders submitted",
        },
        []string{"engine", "order_type", "symbol"},
    )
    
    OrderLatency = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "lx_order_latency_seconds",
            Help:    "Order processing latency",
            Buckets: prometheus.ExponentialBuckets(0.0001, 2, 15), // 100μs to 1.6s
        },
        []string{"engine", "operation"},
    )
    
    OrderBookDepth = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "lx_orderbook_depth",
            Help: "Current order book depth",
        },
        []string{"symbol", "side"},
    )
    
    MatchingRate = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "lx_matching_rate",
            Help: "Orders matched per second",
        },
        []string{"engine", "symbol"},
    )
)
```

#### System Metrics
```yaml
# Collected automatically
system_metrics:
  - go_memstats_*        # Memory usage
  - go_gc_*              # Garbage collection
  - process_cpu_*        # CPU usage
  - process_open_fds     # File descriptors
  - process_resident_*   # Memory
```

### Engine-Specific Metrics

#### Go Engine
```go
// Goroutine metrics
runtime.NumGoroutine()

// GC metrics
var stats runtime.MemStats
runtime.ReadMemStats(&stats)
```

#### C++ Engine
```cpp
// Custom C++ metrics
class MetricsCollector {
    std::atomic<uint64_t> orders_processed{0};
    std::atomic<uint64_t> bytes_allocated{0};
    std::chrono::steady_clock::time_point start_time;
    
    void recordOrder() {
        orders_processed.fetch_add(1, std::memory_order_relaxed);
    }
};
```

#### TypeScript Engine
```typescript
// Node.js metrics
import { register, collectDefaultMetrics } from 'prom-client';

collectDefaultMetrics({ register });

// Custom metrics
const orderCounter = new Counter({
  name: 'lx_ts_orders_total',
  help: 'Total orders processed by TypeScript engine',
  labelNames: ['type', 'symbol']
});
```

## Logging Strategy

### Structured Logging

```go
// pkg/logger/logger.go
import "log/slog"

type Logger struct {
    *slog.Logger
}

func NewLogger() *Logger {
    opts := &slog.HandlerOptions{
        Level: slog.LevelDebug,
        AddSource: true,
    }
    
    handler := slog.NewJSONHandler(os.Stdout, opts)
    return &Logger{slog.New(handler)}
}

// Usage
logger.Info("order submitted",
    slog.String("order_id", order.ID),
    slog.Float64("price", order.Price),
    slog.Int("quantity", order.Quantity),
    slog.String("trace_id", traceID),
)
```

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| DEBUG | Detailed debugging | Order book state changes |
| INFO | Normal operations | Order submitted/filled |
| WARN | Potential issues | High latency detected |
| ERROR | Errors requiring attention | Order validation failed |
| FATAL | System failures | Cannot connect to database |

### Log Aggregation

```yaml
# docker-compose.monitoring.yml
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - es_data:/usr/share/elasticsearch/data
      
  kibana:
    image: kibana:8.11.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
      
  filebeat:
    image: elastic/filebeat:8.11.0
    volumes:
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
```

## Distributed Tracing

### OpenTelemetry Integration

```go
// pkg/tracing/tracing.go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/trace"
)

func InitTracing() (*trace.TracerProvider, error) {
    exporter, err := jaeger.New(
        jaeger.WithCollectorEndpoint(
            jaeger.WithEndpoint("http://jaeger:14268/api/traces"),
        ),
    )
    
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exporter),
        trace.WithSampler(trace.AlwaysSample()),
    )
    
    otel.SetTracerProvider(tp)
    return tp, nil
}
```

### Trace Context Propagation

```go
// Order processing with tracing
func (e *Engine) SubmitOrder(ctx context.Context, order *Order) error {
    ctx, span := tracer.Start(ctx, "submit_order")
    defer span.End()
    
    span.SetAttributes(
        attribute.String("order.id", order.ID),
        attribute.Float64("order.price", order.Price),
    )
    
    // Validate order
    ctx, validateSpan := tracer.Start(ctx, "validate_order")
    if err := e.validateOrder(ctx, order); err != nil {
        validateSpan.RecordError(err)
        validateSpan.End()
        return err
    }
    validateSpan.End()
    
    // Match order
    ctx, matchSpan := tracer.Start(ctx, "match_order")
    matches := e.orderBook.Match(ctx, order)
    matchSpan.SetAttributes(
        attribute.Int("matches.count", len(matches)),
    )
    matchSpan.End()
    
    return nil
}
```

## Dashboards

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "LX DEX Performance",
    "panels": [
      {
        "title": "Orders Per Second",
        "targets": [
          {
            "expr": "rate(lx_orders_submitted_total[1m])",
            "legendFormat": "{{engine}} - {{order_type}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Order Latency P99",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(lx_order_latency_seconds_bucket[5m]))",
            "legendFormat": "{{engine}} - {{operation}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Order Book Depth",
        "targets": [
          {
            "expr": "lx_orderbook_depth",
            "legendFormat": "{{symbol}} - {{side}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(lx_errors_total[5m])",
            "legendFormat": "{{engine}} - {{error_type}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### Key Performance Indicators (KPIs)

1. **Throughput**: Orders processed per second
2. **Latency**: P50, P95, P99 order processing time
3. **Availability**: Uptime percentage
4. **Error Rate**: Failed orders percentage
5. **Resource Usage**: CPU, memory, network

## Alerting Rules

### Prometheus Alert Configuration

```yaml
# prometheus/alerts.yml
groups:
  - name: performance
    rules:
      - alert: HighOrderLatency
        expr: histogram_quantile(0.99, rate(lx_order_latency_seconds_bucket[5m])) > 0.001
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High order latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 1ms)"
          
      - alert: LowThroughput
        expr: rate(lx_orders_submitted_total[5m]) < 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low order throughput"
          description: "Only {{ $value }} orders/sec (expected: >1000)"
          
      - alert: HighErrorRate
        expr: rate(lx_errors_total[5m]) / rate(lx_orders_submitted_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
          
      - alert: OrderBookImbalance
        expr: abs(lx_orderbook_depth{side="bid"} - lx_orderbook_depth{side="ask"}) / (lx_orderbook_depth{side="bid"} + lx_orderbook_depth{side="ask"}) > 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Order book imbalance detected"
          description: "{{ $labels.symbol }} has significant bid/ask imbalance"
```

### Alert Routing

```yaml
# alertmanager/config.yml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  
  routes:
    - match:
        severity: critical
      receiver: pagerduty
      
    - match:
        severity: warning
      receiver: slack

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:9093/webhook'
      
  - name: 'slack'
    slack_configs:
      - channel: '#lx-alerts'
        title: 'LX DEX Alert'
        
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PD_SERVICE_KEY'
```

## Health Checks

### Application Health Endpoints

```go
// pkg/health/health.go
type HealthChecker struct {
    checks map[string]func() error
}

func (h *HealthChecker) RegisterCheck(name string, check func() error) {
    h.checks[name] = check
}

func (h *HealthChecker) Handler() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        status := "healthy"
        details := make(map[string]string)
        
        for name, check := range h.checks {
            if err := check(); err != nil {
                status = "unhealthy"
                details[name] = err.Error()
            } else {
                details[name] = "ok"
            }
        }
        
        response := map[string]interface{}{
            "status": status,
            "checks": details,
            "timestamp": time.Now().Unix(),
        }
        
        json.NewEncoder(w).Encode(response)
    }
}

// Register checks
health.RegisterCheck("orderbook", func() error {
    if orderBook.GetDepth() == 0 {
        return errors.New("orderbook empty")
    }
    return nil
})

health.RegisterCheck("database", func() error {
    return db.Ping()
})

health.RegisterCheck("nats", func() error {
    return natsConn.Status() == nats.CONNECTED
})
```

## Performance Monitoring

### Continuous Profiling

```go
// Enable continuous profiling
import _ "net/http/pprof"

go func() {
    log.Println(http.ListenAndServe("localhost:6060", nil))
}()

// Access profiles at:
// http://localhost:6060/debug/pprof/
```

### Benchmark Monitoring

```bash
#!/bin/bash
# scripts/monitor-benchmarks.sh

# Run benchmarks and export to Prometheus
while true; do
    result=$(make bench-quick | grep "Orders/sec" | awk '{print $2}')
    
    echo "lx_benchmark_throughput{engine=\"go\"} $result" | \
        curl --data-binary @- http://prometheus:9090/metrics/job/benchmark
    
    sleep 300  # Every 5 minutes
done
```

## Monitoring Stack Deployment

### Docker Compose

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    ports:
      - "3000:3000"
      
  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
      
  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager/config.yml:/etc/alertmanager/config.yml
    ports:
      - "9093:9093"

volumes:
  prometheus_data:
  grafana_data:
  es_data:
```

### Kubernetes Deployment

```yaml
# k8s/monitoring.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: lx-engine
spec:
  selector:
    matchLabels:
      app: lx-engine
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: lx-engine-alerts
spec:
  groups:
  - name: lx-engine
    interval: 30s
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.99, lx_order_latency_seconds_bucket) > 0.001
      for: 5m
      annotations:
        summary: "High order processing latency"
```

## Best Practices

### 1. Metric Naming
- Use consistent prefixes: `lx_`
- Include unit in name: `_seconds`, `_bytes`
- Use labels for dimensions

### 2. Log Correlation
- Include trace_id in all logs
- Use consistent timestamp format
- Structure logs as JSON

### 3. Alert Fatigue Prevention
- Set appropriate thresholds
- Use alert grouping
- Implement alert suppression

### 4. Dashboard Design
- Single pane of glass for critical metrics
- Drill-down capability
- Mobile-responsive layouts

### 5. Data Retention
- Metrics: 30 days high-res, 1 year downsampled
- Logs: 7 days hot, 90 days warm
- Traces: 7 days

## Troubleshooting Guide

### High Latency
1. Check GC metrics: `go_gc_duration_seconds`
2. Review trace spans for bottlenecks
3. Analyze CPU profiles
4. Check network latency

### Low Throughput
1. Monitor goroutine count
2. Check connection pool usage
3. Review batch sizes
4. Analyze lock contention

### Memory Issues
1. Review heap profiles
2. Check for goroutine leaks
3. Monitor GC frequency
4. Analyze object allocation

## Monitoring Checklist

### Daily
- [ ] Review error rates
- [ ] Check latency trends
- [ ] Monitor resource usage
- [ ] Verify backup completion

### Weekly
- [ ] Analyze performance trends
- [ ] Review alert effectiveness
- [ ] Update dashboards
- [ ] Check log retention

### Monthly
- [ ] Performance baseline update
- [ ] Alert threshold review
- [ ] Capacity planning
- [ ] Cost optimization

## Conclusion

Effective monitoring is crucial for maintaining the LX DEX platform's performance and reliability. This comprehensive monitoring strategy provides visibility into all aspects of the system, enabling proactive issue detection and rapid resolution.