//! Metrics collection for trading operations.
//!
//! Provides:
//! - Latency tracking (p50, p95, p99)
//! - Throughput measurement
//! - Order/trade statistics
//! - Venue performance comparison

use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Metrics collector for the trading system
pub struct MetricsCollector {
    /// Latency metrics by operation type
    latencies: RwLock<HashMap<String, LatencyTracker>>,
    /// Order metrics
    orders: OrderMetrics,
    /// Trade metrics
    trades: TradeMetrics,
    /// Venue-specific metrics
    venues: RwLock<HashMap<String, Arc<VenueMetrics>>>,
    /// System start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            latencies: RwLock::new(HashMap::new()),
            orders: OrderMetrics::default(),
            trades: TradeMetrics::default(),
            venues: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
        }
    }

    /// Record latency for an operation
    pub fn record_latency(&self, operation: &str, latency: Duration) {
        let mut latencies = self.latencies.write();
        latencies
            .entry(operation.to_string())
            .or_default()
            .record(latency);
    }

    /// Record latency using a timer
    pub fn start_timer(&self, operation: &str) -> Timer {
        Timer {
            collector: self,
            operation: operation.to_string(),
            start: Instant::now(),
        }
    }

    /// Get latency statistics for an operation
    pub fn latency_stats(&self, operation: &str) -> Option<LatencyStats> {
        self.latencies.read().get(operation).map(|t| t.stats())
    }

    /// Get all latency statistics
    pub fn all_latency_stats(&self) -> HashMap<String, LatencyStats> {
        self.latencies
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.stats()))
            .collect()
    }

    /// Record an order submission
    pub fn record_order_submitted(&self, venue: &str) {
        self.orders.submitted.fetch_add(1, Ordering::Relaxed);
        self.venue_metrics(venue)
            .orders_submitted
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order fill
    pub fn record_order_filled(&self, venue: &str, quantity: Decimal, price: Decimal) {
        self.orders.filled.fetch_add(1, Ordering::Relaxed);
        self.venue_metrics(venue)
            .orders_filled
            .fetch_add(1, Ordering::Relaxed);

        // Track volume
        let volume = quantity * price;
        let volume_u64 = (volume * Decimal::from(1_000_000))
            .to_string()
            .parse::<u64>()
            .unwrap_or(0);
        self.orders
            .total_volume_micro
            .fetch_add(volume_u64, Ordering::Relaxed);
        self.venue_metrics(venue)
            .volume_micro
            .fetch_add(volume_u64, Ordering::Relaxed);
    }

    /// Record an order cancellation
    pub fn record_order_cancelled(&self, venue: &str) {
        self.orders.cancelled.fetch_add(1, Ordering::Relaxed);
        self.venue_metrics(venue)
            .orders_cancelled
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order rejection
    pub fn record_order_rejected(&self, venue: &str) {
        self.orders.rejected.fetch_add(1, Ordering::Relaxed);
        self.venue_metrics(venue)
            .orders_rejected
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a trade
    pub fn record_trade(
        &self,
        venue: &str,
        side: crate::types::Side,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal,
        is_maker: bool,
    ) {
        self.trades.total.fetch_add(1, Ordering::Relaxed);

        let volume = quantity * price;
        let volume_u64 = (volume * Decimal::from(1_000_000))
            .to_string()
            .parse::<u64>()
            .unwrap_or(0);
        let fee_u64 = (fee * Decimal::from(1_000_000))
            .to_string()
            .parse::<u64>()
            .unwrap_or(0);

        self.trades
            .total_volume_micro
            .fetch_add(volume_u64, Ordering::Relaxed);
        self.trades
            .total_fees_micro
            .fetch_add(fee_u64, Ordering::Relaxed);

        match side {
            crate::types::Side::Buy => self.trades.buys.fetch_add(1, Ordering::Relaxed),
            crate::types::Side::Sell => self.trades.sells.fetch_add(1, Ordering::Relaxed),
        };

        if is_maker {
            self.trades.maker.fetch_add(1, Ordering::Relaxed);
        } else {
            self.trades.taker.fetch_add(1, Ordering::Relaxed);
        }

        // Update venue metrics
        let venue_metrics = self.venue_metrics(venue);
        venue_metrics.trades.fetch_add(1, Ordering::Relaxed);
        venue_metrics
            .volume_micro
            .fetch_add(volume_u64, Ordering::Relaxed);
        venue_metrics
            .fees_micro
            .fetch_add(fee_u64, Ordering::Relaxed);
    }

    /// Record venue latency
    pub fn record_venue_latency(&self, venue: &str, latency: Duration) {
        self.venue_metrics(venue)
            .latency_tracker
            .write()
            .record(latency);
    }

    /// Get order metrics
    pub fn order_metrics(&self) -> &OrderMetrics {
        &self.orders
    }

    /// Get trade metrics
    pub fn trade_metrics(&self) -> &TradeMetrics {
        &self.trades
    }

    /// Get venue metrics
    pub fn venue_metrics(&self, venue: &str) -> Arc<VenueMetrics> {
        {
            let venues = self.venues.read();
            if let Some(metrics) = venues.get(venue) {
                return Arc::clone(metrics);
            }
        }

        let mut venues = self.venues.write();
        Arc::clone(
            venues
                .entry(venue.to_string())
                .or_insert_with(|| Arc::new(VenueMetrics::new(venue))),
        )
    }

    /// Get all venue metrics
    pub fn all_venue_metrics(&self) -> HashMap<String, VenueMetricsSnapshot> {
        self.venues
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.snapshot()))
            .collect()
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get throughput (orders per second)
    pub fn order_throughput(&self) -> f64 {
        let total = self.orders.submitted.load(Ordering::Relaxed) as f64;
        let elapsed = self.uptime().as_secs_f64();
        if elapsed > 0.0 {
            total / elapsed
        } else {
            0.0
        }
    }

    /// Get trade throughput (trades per second)
    pub fn trade_throughput(&self) -> f64 {
        let total = self.trades.total.load(Ordering::Relaxed) as f64;
        let elapsed = self.uptime().as_secs_f64();
        if elapsed > 0.0 {
            total / elapsed
        } else {
            0.0
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.latencies.write().clear();
        self.venues.write().clear();
        // Note: Atomic counters would need individual reset
    }

    /// Export metrics as JSON
    pub fn export_json(&self) -> serde_json::Value {
        serde_json::json!({
            "uptime_secs": self.uptime().as_secs(),
            "orders": {
                "submitted": self.orders.submitted.load(Ordering::Relaxed),
                "filled": self.orders.filled.load(Ordering::Relaxed),
                "cancelled": self.orders.cancelled.load(Ordering::Relaxed),
                "rejected": self.orders.rejected.load(Ordering::Relaxed),
                "throughput_per_sec": self.order_throughput(),
            },
            "trades": {
                "total": self.trades.total.load(Ordering::Relaxed),
                "buys": self.trades.buys.load(Ordering::Relaxed),
                "sells": self.trades.sells.load(Ordering::Relaxed),
                "maker": self.trades.maker.load(Ordering::Relaxed),
                "taker": self.trades.taker.load(Ordering::Relaxed),
                "volume": self.trades.total_volume_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                "fees": self.trades.total_fees_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                "throughput_per_sec": self.trade_throughput(),
            },
            "latencies": self.all_latency_stats().iter().map(|(k, v)| {
                (k.clone(), serde_json::json!({
                    "count": v.count,
                    "min_ms": v.min.as_secs_f64() * 1000.0,
                    "max_ms": v.max.as_secs_f64() * 1000.0,
                    "avg_ms": v.avg.as_secs_f64() * 1000.0,
                    "p50_ms": v.p50.as_secs_f64() * 1000.0,
                    "p95_ms": v.p95.as_secs_f64() * 1000.0,
                    "p99_ms": v.p99.as_secs_f64() * 1000.0,
                }))
            }).collect::<HashMap<_, _>>(),
            "venues": self.all_venue_metrics().iter().map(|(k, v)| {
                (k.clone(), serde_json::json!({
                    "orders_submitted": v.orders_submitted,
                    "orders_filled": v.orders_filled,
                    "orders_cancelled": v.orders_cancelled,
                    "trades": v.trades,
                    "volume": v.volume,
                    "fees": v.fees,
                    "latency": v.latency,
                }))
            }).collect::<HashMap<_, _>>(),
        })
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring operation latency
pub struct Timer<'a> {
    collector: &'a MetricsCollector,
    operation: String,
    start: Instant,
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        self.collector
            .record_latency(&self.operation, self.start.elapsed());
    }
}

/// Latency tracker with percentile support
pub struct LatencyTracker {
    /// Recent latencies (for percentile calculation)
    samples: VecDeque<Duration>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Total count
    count: u64,
    /// Sum for average
    sum_us: u64,
    /// Minimum latency
    min: Duration,
    /// Maximum latency
    max: Duration,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self::with_max_samples(10_000)
    }

    pub fn with_max_samples(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            count: 0,
            sum_us: 0,
            min: Duration::MAX,
            max: Duration::ZERO,
        }
    }

    pub fn record(&mut self, latency: Duration) {
        self.count += 1;
        self.sum_us += latency.as_micros() as u64;

        if latency < self.min {
            self.min = latency;
        }
        if latency > self.max {
            self.max = latency;
        }

        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(latency);
    }

    pub fn stats(&self) -> LatencyStats {
        if self.count == 0 {
            return LatencyStats::default();
        }

        let avg = Duration::from_micros(self.sum_us / self.count);

        // Calculate percentiles
        let mut sorted: Vec<_> = self.samples.iter().copied().collect();
        sorted.sort();

        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);

        LatencyStats {
            count: self.count,
            min: self.min,
            max: self.max,
            avg,
            p50,
            p95,
            p99,
        }
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Latency statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LatencyStats {
    pub count: u64,
    #[serde(serialize_with = "serialize_duration")]
    pub min: Duration,
    #[serde(serialize_with = "serialize_duration")]
    pub max: Duration,
    #[serde(serialize_with = "serialize_duration")]
    pub avg: Duration,
    #[serde(serialize_with = "serialize_duration")]
    pub p50: Duration,
    #[serde(serialize_with = "serialize_duration")]
    pub p95: Duration,
    #[serde(serialize_with = "serialize_duration")]
    pub p99: Duration,
}

fn serialize_duration<S>(duration: &Duration, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_f64(duration.as_secs_f64() * 1000.0) // Serialize as milliseconds
}

/// Order metrics
#[derive(Default)]
pub struct OrderMetrics {
    pub submitted: AtomicU64,
    pub filled: AtomicU64,
    pub partially_filled: AtomicU64,
    pub cancelled: AtomicU64,
    pub rejected: AtomicU64,
    pub expired: AtomicU64,
    pub total_volume_micro: AtomicU64, // Volume * 1_000_000 for precision
}

impl OrderMetrics {
    pub fn fill_rate(&self) -> f64 {
        let submitted = self.submitted.load(Ordering::Relaxed) as f64;
        if submitted == 0.0 {
            return 0.0;
        }
        let filled = self.filled.load(Ordering::Relaxed) as f64;
        filled / submitted
    }

    pub fn reject_rate(&self) -> f64 {
        let submitted = self.submitted.load(Ordering::Relaxed) as f64;
        if submitted == 0.0 {
            return 0.0;
        }
        let rejected = self.rejected.load(Ordering::Relaxed) as f64;
        rejected / submitted
    }

    pub fn total_volume(&self) -> Decimal {
        let micro = self.total_volume_micro.load(Ordering::Relaxed);
        Decimal::from(micro) / Decimal::from(1_000_000)
    }
}

/// Trade metrics
#[derive(Default)]
pub struct TradeMetrics {
    pub total: AtomicU64,
    pub buys: AtomicU64,
    pub sells: AtomicU64,
    pub maker: AtomicU64,
    pub taker: AtomicU64,
    pub total_volume_micro: AtomicU64,
    pub total_fees_micro: AtomicU64,
}

impl TradeMetrics {
    pub fn maker_ratio(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed) as f64;
        if total == 0.0 {
            return 0.0;
        }
        let maker = self.maker.load(Ordering::Relaxed) as f64;
        maker / total
    }

    pub fn total_volume(&self) -> Decimal {
        let micro = self.total_volume_micro.load(Ordering::Relaxed);
        Decimal::from(micro) / Decimal::from(1_000_000)
    }

    pub fn total_fees(&self) -> Decimal {
        let micro = self.total_fees_micro.load(Ordering::Relaxed);
        Decimal::from(micro) / Decimal::from(1_000_000)
    }
}

/// Per-venue metrics
pub struct VenueMetrics {
    pub name: String,
    pub orders_submitted: AtomicU64,
    pub orders_filled: AtomicU64,
    pub orders_cancelled: AtomicU64,
    pub orders_rejected: AtomicU64,
    pub trades: AtomicU64,
    pub volume_micro: AtomicU64,
    pub fees_micro: AtomicU64,
    pub latency_tracker: RwLock<LatencyTracker>,
}

impl VenueMetrics {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            orders_submitted: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            trades: AtomicU64::new(0),
            volume_micro: AtomicU64::new(0),
            fees_micro: AtomicU64::new(0),
            latency_tracker: RwLock::new(LatencyTracker::new()),
        }
    }

    pub fn snapshot(&self) -> VenueMetricsSnapshot {
        VenueMetricsSnapshot {
            name: self.name.clone(),
            orders_submitted: self.orders_submitted.load(Ordering::Relaxed),
            orders_filled: self.orders_filled.load(Ordering::Relaxed),
            orders_cancelled: self.orders_cancelled.load(Ordering::Relaxed),
            orders_rejected: self.orders_rejected.load(Ordering::Relaxed),
            trades: self.trades.load(Ordering::Relaxed),
            volume: Decimal::from(self.volume_micro.load(Ordering::Relaxed))
                / Decimal::from(1_000_000),
            fees: Decimal::from(self.fees_micro.load(Ordering::Relaxed)) / Decimal::from(1_000_000),
            latency: self.latency_tracker.read().stats(),
        }
    }
}

/// Snapshot of venue metrics
#[derive(Debug, Clone)]
pub struct VenueMetricsSnapshot {
    pub name: String,
    pub orders_submitted: u64,
    pub orders_filled: u64,
    pub orders_cancelled: u64,
    pub orders_rejected: u64,
    pub trades: u64,
    pub volume: Decimal,
    pub fees: Decimal,
    pub latency: LatencyStats,
}

impl VenueMetricsSnapshot {
    pub fn fill_rate(&self) -> f64 {
        if self.orders_submitted == 0 {
            return 0.0;
        }
        self.orders_filled as f64 / self.orders_submitted as f64
    }
}

/// Global metrics instance (optional singleton pattern)
static GLOBAL_METRICS: std::sync::OnceLock<Arc<MetricsCollector>> = std::sync::OnceLock::new();

/// Get or initialize global metrics
pub fn global_metrics() -> Arc<MetricsCollector> {
    GLOBAL_METRICS
        .get_or_init(|| Arc::new(MetricsCollector::new()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new();

        tracker.record(Duration::from_millis(10));
        tracker.record(Duration::from_millis(20));
        tracker.record(Duration::from_millis(30));
        tracker.record(Duration::from_millis(40));
        tracker.record(Duration::from_millis(50));

        let stats = tracker.stats();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.avg, Duration::from_millis(30)); // (10+20+30+40+50)/5 = 30
    }

    #[test]
    fn test_percentile() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();

        // Percentile calculation uses floor, so p50 of 100 items = index 50 = 51ms
        let p50 = percentile(&samples, 0.50);
        let p95 = percentile(&samples, 0.95);
        let p99 = percentile(&samples, 0.99);

        // Values should be close to expected (allowing for index calculation differences)
        assert!(p50.as_millis() >= 50 && p50.as_millis() <= 51);
        assert!(p95.as_millis() >= 95 && p95.as_millis() <= 96);
        assert!(p99.as_millis() >= 99 && p99.as_millis() <= 100);
    }

    #[test]
    fn test_metrics_collector() {
        let metrics = MetricsCollector::new();

        metrics.record_order_submitted("venue1");
        metrics.record_order_submitted("venue1");
        metrics.record_order_filled("venue1", Decimal::from(1), Decimal::from(50000));

        assert_eq!(metrics.orders.submitted.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.orders.filled.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_order_metrics() {
        let metrics = OrderMetrics::default();

        metrics.submitted.store(100, Ordering::Relaxed);
        metrics.filled.store(90, Ordering::Relaxed);
        metrics.rejected.store(5, Ordering::Relaxed);

        assert!((metrics.fill_rate() - 0.9).abs() < 0.001);
        assert!((metrics.reject_rate() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_trade_metrics() {
        let metrics = TradeMetrics::default();

        metrics.total.store(100, Ordering::Relaxed);
        metrics.maker.store(60, Ordering::Relaxed);
        metrics.taker.store(40, Ordering::Relaxed);

        assert!((metrics.maker_ratio() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_timer() {
        let metrics = MetricsCollector::new();

        {
            let _timer = metrics.start_timer("test_operation");
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = metrics.latency_stats("test_operation");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().count, 1);
    }

    #[test]
    fn test_export_json() {
        let metrics = MetricsCollector::new();
        metrics.record_order_submitted("test");
        metrics.record_latency("get_orderbook", Duration::from_millis(5));

        let json = metrics.export_json();
        assert!(json["orders"]["submitted"].as_u64().unwrap() >= 1);
    }
}
