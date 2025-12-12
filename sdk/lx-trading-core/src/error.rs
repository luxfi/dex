//! Error types for the LX Trading SDK.

use thiserror::Error;

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

/// LX Trading SDK errors
#[derive(Error, Debug)]
pub enum Error {
    // Connection errors
    #[error("Connection failed to {venue}: {message}")]
    ConnectionFailed { venue: String, message: String },

    #[error("Venue not connected: {0}")]
    VenueNotConnected(String),

    #[error("Venue not found: {0}")]
    VenueNotFound(String),

    #[error("Authentication failed for {venue}: {message}")]
    AuthenticationFailed { venue: String, message: String },

    // Order errors
    #[error("Order rejected: {0}")]
    OrderRejected(String),

    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Insufficient balance: need {required} {asset}, have {available}")]
    InsufficientBalance {
        asset: String,
        required: String,
        available: String,
    },

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Order quantity below minimum: {quantity} < {minimum}")]
    QuantityBelowMinimum { quantity: String, minimum: String },

    #[error("Order notional below minimum: {notional} < {minimum}")]
    NotionalBelowMinimum { notional: String, minimum: String },

    // Market errors
    #[error("Market not found: {0}")]
    MarketNotFound(String),

    #[error("Trading pair not supported: {pair} on {venue}")]
    PairNotSupported { pair: String, venue: String },

    #[error("No liquidity available for {0}")]
    NoLiquidity(String),

    // Risk errors
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),

    #[error("Position limit exceeded for {asset}: {current} + {requested} > {limit}")]
    PositionLimitExceeded {
        asset: String,
        current: String,
        requested: String,
        limit: String,
    },

    #[error("Daily loss limit exceeded: {loss} > {limit}")]
    DailyLossLimitExceeded { loss: String, limit: String },

    // Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Missing configuration: {0}")]
    MissingConfig(String),

    #[error("Invalid configuration: {field} - {message}")]
    InvalidConfig { field: String, message: String },

    // Network errors
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Request timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Rate limited by {venue}, retry after {retry_after_ms}ms")]
    RateLimited { venue: String, retry_after_ms: u64 },

    // API errors
    #[error("API error from {venue}: [{code}] {message}")]
    ApiError {
        venue: String,
        code: String,
        message: String,
    },

    #[error("Websocket error: {0}")]
    WebsocketError(String),

    // Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    // Internal errors
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    // Adapter errors
    #[error("CCXT error: {0}")]
    CcxtError(String),

    #[error("Hummingbot error: {0}")]
    HummingbotError(String),

    #[error("Adapter error for {adapter}: {message}")]
    AdapterError { adapter: String, message: String },
}

impl Error {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::NetworkError(_)
                | Error::Timeout { .. }
                | Error::RateLimited { .. }
                | Error::WebsocketError(_)
        )
    }

    /// Check if error is a temporary condition
    pub fn is_temporary(&self) -> bool {
        matches!(
            self,
            Error::RateLimited { .. }
                | Error::NoLiquidity(_)
                | Error::Timeout { .. }
        )
    }

    /// Get suggested retry delay in milliseconds
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            Error::RateLimited { retry_after_ms, .. } => Some(*retry_after_ms),
            Error::Timeout { .. } => Some(1000),
            Error::NetworkError(_) => Some(500),
            _ => None,
        }
    }
}

// Conversions from common error types
impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Error::Timeout { timeout_ms: 30000 }
        } else if err.is_connect() {
            Error::NetworkError(format!("Connection failed: {err}"))
        } else {
            Error::NetworkError(err.to_string())
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::DeserializationError(err.to_string())
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(err: tokio_tungstenite::tungstenite::Error) -> Self {
        Error::WebsocketError(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Internal(format!("IO error: {err}"))
    }
}

impl From<url::ParseError> for Error {
    fn from(err: url::ParseError) -> Self {
        Error::ConfigError(format!("Invalid URL: {err}"))
    }
}

impl From<toml::de::Error> for Error {
    fn from(err: toml::de::Error) -> Self {
        Error::ConfigError(format!("TOML parse error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retryable_errors() {
        assert!(Error::RateLimited {
            venue: "test".into(),
            retry_after_ms: 1000
        }
        .is_retryable());

        assert!(Error::Timeout { timeout_ms: 5000 }.is_retryable());

        assert!(!Error::OrderRejected("test".into()).is_retryable());
    }

    #[test]
    fn test_retry_delay() {
        let err = Error::RateLimited {
            venue: "test".into(),
            retry_after_ms: 5000,
        };
        assert_eq!(err.retry_delay_ms(), Some(5000));
    }
}
