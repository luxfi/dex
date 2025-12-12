//! Error types for the LX SDK.
//!
//! Provides a unified error type with precise failure modes.

use std::fmt;

/// Result type alias for SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for all SDK operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// WebSocket connection failed.
    #[error("websocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    /// HTTP request failed.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// URL parsing failed.
    #[error("url error: {0}")]
    Url(#[from] url::ParseError),

    /// Server returned an error response.
    #[error("server error {code}: {message}")]
    Server { code: i32, message: String },

    /// Authentication failed.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// Client not connected.
    #[error("not connected")]
    NotConnected,

    /// Connection already established.
    #[error("already connected")]
    AlreadyConnected,

    /// Request timed out.
    #[error("request timed out")]
    Timeout,

    /// Invalid order parameters.
    #[error("invalid order: {0}")]
    InvalidOrder(String),

    /// Order not found.
    #[error("order not found: {0}")]
    OrderNotFound(u64),

    /// Insufficient funds.
    #[error("insufficient funds: {0}")]
    InsufficientFunds(String),

    /// Rate limit exceeded.
    #[error("rate limit exceeded")]
    RateLimited,

    /// Channel send failed.
    #[error("channel closed")]
    ChannelClosed,

    /// Invalid response format.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

impl Error {
    /// Create a server error from code and message.
    pub fn server(code: i32, message: impl Into<String>) -> Self {
        Self::Server {
            code,
            message: message.into(),
        }
    }

    /// Create an authentication error.
    pub fn auth(message: impl Into<String>) -> Self {
        Self::Auth(message.into())
    }

    /// Create an invalid order error.
    pub fn invalid_order(reason: impl Into<String>) -> Self {
        Self::InvalidOrder(reason.into())
    }

    /// Create an invalid response error.
    pub fn invalid_response(reason: impl Into<String>) -> Self {
        Self::InvalidResponse(reason.into())
    }

    /// Returns true if error is recoverable (retry may succeed).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Error::Timeout | Error::RateLimited | Error::WebSocket(_)
        )
    }

    /// Returns true if reconnection is needed.
    pub fn needs_reconnect(&self) -> bool {
        matches!(
            self,
            Error::NotConnected | Error::WebSocket(_) | Error::ChannelClosed
        )
    }
}

/// RPC error response from server.
#[derive(Debug, serde::Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RPC error {}: {}", self.code, self.message)
    }
}

impl From<RpcError> for Error {
    fn from(e: RpcError) -> Self {
        Error::Server {
            code: e.code,
            message: e.message,
        }
    }
}
