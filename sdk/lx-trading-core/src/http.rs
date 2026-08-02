//! HTTP client with connection pooling and retry logic.
//!
//! Provides:
//! - Connection pooling for efficient HTTP requests
//! - Retry logic with exponential backoff
//! - Rate limiting support
//! - Request/response interceptors

use parking_lot::RwLock;
use reqwest::{Client, Method, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

/// HTTP client configuration
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Base URL for requests
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Pool idle timeout
    pub pool_idle_timeout: Duration,
    /// Max idle connections per host
    pub pool_max_idle_per_host: usize,
    /// Enable HTTP/2
    pub http2_enabled: bool,
    /// User agent
    pub user_agent: String,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            pool_idle_timeout: Duration::from_secs(90),
            pool_max_idle_per_host: 32,
            http2_enabled: true,
            user_agent: format!("lx-trading/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

impl HttpConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..Default::default()
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn with_pool_size(mut self, max_idle: usize) -> Self {
        self.pool_max_idle_per_host = max_idle;
        self
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial backoff delay
    pub initial_delay: Duration,
    /// Maximum backoff delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub multiplier: f64,
    /// Jitter factor (0.0 - 1.0)
    pub jitter: f64,
    /// Retryable status codes
    pub retryable_statuses: Vec<StatusCode>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: 0.1,
            retryable_statuses: vec![
                StatusCode::TOO_MANY_REQUESTS,
                StatusCode::SERVICE_UNAVAILABLE,
                StatusCode::GATEWAY_TIMEOUT,
                StatusCode::BAD_GATEWAY,
                StatusCode::REQUEST_TIMEOUT,
            ],
        }
    }
}

impl RetryConfig {
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Calculate delay for retry attempt
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay =
            self.initial_delay.as_millis() as f64 * self.multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(self.max_delay.as_millis() as f64);

        // Add jitter
        let jitter_range = capped_delay * self.jitter;
        let jitter = (rand_simple() * 2.0 - 1.0) * jitter_range;
        let final_delay = (capped_delay + jitter).max(0.0);

        Duration::from_millis(final_delay as u64)
    }
}

/// Simple pseudo-random number for jitter (no external dependency)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos as f64 % 1000.0) / 1000.0
}

/// Rate limiter for API requests
pub struct RateLimiter {
    /// Requests per second limit
    requests_per_second: f64,
    /// Token bucket
    tokens: RwLock<f64>,
    /// Last refill time
    last_refill: RwLock<Instant>,
    /// Maximum tokens (burst size)
    max_tokens: f64,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(requests_per_second: f64) -> Self {
        Self {
            requests_per_second,
            tokens: RwLock::new(requests_per_second),
            last_refill: RwLock::new(Instant::now()),
            max_tokens: requests_per_second * 2.0, // Allow 2x burst
        }
    }

    /// Acquire a token (wait if necessary)
    #[allow(clippy::await_holding_lock)]
    pub async fn acquire(&self) {
        loop {
            self.refill();

            {
                let mut tokens = self.tokens.write();
                if *tokens >= 1.0 {
                    *tokens -= 1.0;
                    return;
                }
            } // Guard dropped here before await

            // Wait for token
            let wait_time = Duration::from_secs_f64(1.0 / self.requests_per_second);
            tokio::time::sleep(wait_time).await;
        }
    }

    /// Try to acquire a token without waiting
    pub fn try_acquire(&self) -> bool {
        self.refill();

        let mut tokens = self.tokens.write();
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&self) {
        let now = Instant::now();
        let mut last_refill = self.last_refill.write();
        let elapsed = now.duration_since(*last_refill);
        *last_refill = now;

        let mut tokens = self.tokens.write();
        let new_tokens = elapsed.as_secs_f64() * self.requests_per_second;
        *tokens = (*tokens + new_tokens).min(self.max_tokens);
    }

    /// Get current available tokens
    pub fn available(&self) -> f64 {
        self.refill();
        *self.tokens.read()
    }
}

/// HTTP client with connection pooling and retry support
pub struct HttpClient {
    client: Client,
    config: HttpConfig,
    retry_config: RetryConfig,
    rate_limiter: Option<RateLimiter>,
    metrics: Arc<HttpMetrics>,
    default_headers: RwLock<HashMap<String, String>>,
}

/// HTTP request metrics
pub struct HttpMetrics {
    /// Total requests made
    pub total_requests: AtomicU64,
    /// Successful requests
    pub successful_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Retried requests
    pub retried_requests: AtomicU64,
    /// Rate limited requests
    pub rate_limited_requests: AtomicU64,
    /// Total latency in microseconds
    pub total_latency_us: AtomicU64,
}

impl Default for HttpMetrics {
    fn default() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            retried_requests: AtomicU64::new(0),
            rate_limited_requests: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
        }
    }
}

impl HttpMetrics {
    /// Get average latency in milliseconds
    pub fn average_latency_ms(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let total_us = self.total_latency_us.load(Ordering::Relaxed);
        (total_us as f64 / total as f64) / 1000.0
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let successful = self.successful_requests.load(Ordering::Relaxed);
        successful as f64 / total as f64
    }
}

impl HttpClient {
    /// Create a new HTTP client
    pub fn new(config: HttpConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .pool_idle_timeout(config.pool_idle_timeout)
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            config,
            retry_config: RetryConfig::default(),
            rate_limiter: None,
            metrics: Arc::new(HttpMetrics::default()),
            default_headers: RwLock::new(HashMap::new()),
        })
    }

    /// Set retry configuration
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Set rate limiter
    pub fn with_rate_limit(mut self, requests_per_second: f64) -> Self {
        self.rate_limiter = Some(RateLimiter::new(requests_per_second));
        self
    }

    /// Set a default header
    pub fn set_default_header(&self, key: impl Into<String>, value: impl Into<String>) {
        self.default_headers
            .write()
            .insert(key.into(), value.into());
    }

    /// Get metrics
    pub fn metrics(&self) -> Arc<HttpMetrics> {
        self.metrics.clone()
    }

    /// Make a GET request
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.request(Method::GET, path, Option::<()>::None).await
    }

    /// Make a POST request
    pub async fn post<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: B) -> Result<T> {
        self.request(Method::POST, path, Some(body)).await
    }

    /// Make a PUT request
    pub async fn put<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: B) -> Result<T> {
        self.request(Method::PUT, path, Some(body)).await
    }

    /// Make a DELETE request
    pub async fn delete<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.request(Method::DELETE, path, Option::<()>::None).await
    }

    /// Make a DELETE request with body
    pub async fn delete_with_body<T: DeserializeOwned, B: Serialize>(
        &self,
        path: &str,
        body: B,
    ) -> Result<T> {
        self.request(Method::DELETE, path, Some(body)).await
    }

    /// Make a request with full control
    pub async fn request<T: DeserializeOwned, B: Serialize>(
        &self,
        method: Method,
        path: &str,
        body: Option<B>,
    ) -> Result<T> {
        // Rate limiting
        if let Some(limiter) = &self.rate_limiter {
            limiter.acquire().await;
        }

        let url = if path.starts_with("http") {
            path.to_string()
        } else {
            format!("{}{}", self.config.base_url.trim_end_matches('/'), path)
        };

        let mut attempt = 0;
        let mut last_error: Option<Error> = None;

        while attempt <= self.retry_config.max_retries {
            if attempt > 0 {
                self.metrics
                    .retried_requests
                    .fetch_add(1, Ordering::Relaxed);
                let delay = self.retry_config.delay_for_attempt(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
            let start = Instant::now();

            let result = self.execute_request(&method, &url, &body).await;
            let latency = start.elapsed();

            self.metrics
                .total_latency_us
                .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);

            match result {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        self.metrics
                            .successful_requests
                            .fetch_add(1, Ordering::Relaxed);
                        let text = response.text().await.map_err(|e| {
                            Error::NetworkError(format!("Failed to read response: {e}"))
                        })?;
                        return serde_json::from_str(&text).map_err(|e| {
                            Error::DeserializationError(format!(
                                "Failed to parse response: {} - body: {}",
                                e,
                                &text[..text.len().min(500)]
                            ))
                        });
                    }

                    // Check for rate limiting
                    if status == StatusCode::TOO_MANY_REQUESTS {
                        self.metrics
                            .rate_limited_requests
                            .fetch_add(1, Ordering::Relaxed);

                        // Extract retry-after header if present
                        if let Some(retry_after) = response.headers().get("retry-after") {
                            if let Ok(secs) = retry_after.to_str().unwrap_or("1").parse::<u64>() {
                                tokio::time::sleep(Duration::from_secs(secs)).await;
                            }
                        }
                    }

                    // Check if retryable
                    if self.retry_config.retryable_statuses.contains(&status) {
                        attempt += 1;
                        last_error = Some(Error::ApiError {
                            venue: self.config.base_url.clone(),
                            code: status.as_str().to_string(),
                            message: response.text().await.unwrap_or_default(),
                        });
                        continue;
                    }

                    // Non-retryable error
                    self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                    let text = response.text().await.unwrap_or_default();
                    return Err(Error::ApiError {
                        venue: self.config.base_url.clone(),
                        code: status.as_str().to_string(),
                        message: text,
                    });
                }
                Err(e) => {
                    // Network errors are retryable
                    if e.is_retryable() {
                        attempt += 1;
                        last_error = Some(e);
                        continue;
                    }

                    self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                    return Err(e);
                }
            }
        }

        self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
        Err(last_error.unwrap_or_else(|| Error::NetworkError("Max retries exceeded".into())))
    }

    async fn execute_request<B: Serialize>(
        &self,
        method: &Method,
        url: &str,
        body: &Option<B>,
    ) -> Result<Response> {
        let mut request = self.client.request(method.clone(), url);

        // Add default headers
        for (key, value) in self.default_headers.read().iter() {
            request = request.header(key, value);
        }

        // Add body
        if let Some(body) = body {
            request = request.json(body);
        }

        request.send().await.map_err(Error::from)
    }

    /// Make a raw request and return the response
    pub async fn raw_request(
        &self,
        method: Method,
        url: &str,
        headers: HashMap<String, String>,
        body: Option<String>,
    ) -> Result<(StatusCode, String)> {
        // Rate limiting
        if let Some(limiter) = &self.rate_limiter {
            limiter.acquire().await;
        }

        let full_url = if url.starts_with("http") {
            url.to_string()
        } else {
            format!("{}{}", self.config.base_url.trim_end_matches('/'), url)
        };

        let mut request = self.client.request(method, &full_url);

        // Add headers
        for (key, value) in &headers {
            request = request.header(key, value);
        }

        // Add default headers
        for (key, value) in self.default_headers.read().iter() {
            request = request.header(key, value);
        }

        // Add body
        if let Some(body) = body {
            request = request.body(body);
        }

        let start = Instant::now();
        let response = request.send().await?;
        let latency = start.elapsed();

        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_latency_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| Error::NetworkError(format!("Failed to read response: {e}")))?;

        if status.is_success() {
            self.metrics
                .successful_requests
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
        }

        Ok((status, text))
    }
}

/// Connection pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub idle_connections: usize,
    pub active_connections: usize,
    pub total_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_config() {
        let config = HttpConfig::new("https://api.example.com")
            .with_timeout(Duration::from_secs(60))
            .with_pool_size(64);

        assert_eq!(config.base_url, "https://api.example.com");
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.pool_max_idle_per_host, 64);
    }

    #[test]
    fn test_retry_config() {
        let config = RetryConfig::default()
            .with_max_retries(5)
            .with_initial_delay(Duration::from_millis(200));

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay, Duration::from_millis(200));
    }

    #[test]
    fn test_retry_delay_calculation() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: 0.0, // No jitter for deterministic test
            retryable_statuses: vec![],
        };

        // First attempt: 100ms
        let delay0 = config.delay_for_attempt(0);
        assert!(delay0.as_millis() >= 95 && delay0.as_millis() <= 105);

        // Second attempt: 200ms
        let delay1 = config.delay_for_attempt(1);
        assert!(delay1.as_millis() >= 195 && delay1.as_millis() <= 205);

        // Third attempt: 400ms
        let delay2 = config.delay_for_attempt(2);
        assert!(delay2.as_millis() >= 395 && delay2.as_millis() <= 405);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10.0); // 10 requests per second

        // Should be able to acquire first token immediately
        assert!(limiter.try_acquire());

        // Available should decrease
        let available = limiter.available();
        assert!(available < 10.0);
    }

    #[test]
    fn test_http_metrics() {
        let metrics = HttpMetrics::default();

        metrics.total_requests.store(100, Ordering::Relaxed);
        metrics.successful_requests.store(95, Ordering::Relaxed);
        metrics.total_latency_us.store(1_000_000, Ordering::Relaxed); // 1 second total

        assert!((metrics.success_rate() - 0.95).abs() < 0.001);
        assert!((metrics.average_latency_ms() - 10.0).abs() < 0.1); // 10ms average
    }

    #[tokio::test]
    async fn test_http_client_creation() {
        let config = HttpConfig::new("https://api.example.com");
        let client = HttpClient::new(config).unwrap();

        assert_eq!(client.metrics.total_requests.load(Ordering::Relaxed), 0);
    }
}
