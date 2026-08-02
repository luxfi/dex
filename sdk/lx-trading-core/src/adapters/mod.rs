//! Venue adapters for the LX Trading SDK.
//!
//! This module provides a unified interface to different trading venues:
//! - Native LX DEX and LX AMM
//! - CCXT-compatible exchanges
//! - Hummingbot Gateway connectors
//!
//! All adapters implement the `VenueAdapter` trait for consistent API.

pub mod adapter;
pub mod ccxt;
pub mod hummingbot;
pub mod native;

pub use adapter::{VenueAdapter, VenueCapabilities};
pub use ccxt::CcxtAdapter;
pub use hummingbot::HummingbotAdapter;
pub use native::{LxAmmAdapter, LxDexAdapter};

use crate::config::{CcxtConfig, HummingbotConfig, NativeVenueConfig};
use crate::error::Result;
use std::sync::Arc;

/// Create adapter from native venue config
pub async fn create_native_adapter(
    name: &str,
    config: &NativeVenueConfig,
) -> Result<Arc<dyn VenueAdapter>> {
    match config.venue_type.as_str() {
        "dex" | "orderBook" => {
            let adapter = LxDexAdapter::new(name, config.clone()).await?;
            Ok(Arc::new(adapter))
        }
        "amm" | "pool" => {
            let adapter = LxAmmAdapter::new(name, config.clone()).await?;
            Ok(Arc::new(adapter))
        }
        _ => Err(crate::error::Error::ConfigError(format!(
            "Unknown native venue type: {}",
            config.venue_type
        ))),
    }
}

/// Create adapter from CCXT config
pub async fn create_ccxt_adapter(name: &str, config: &CcxtConfig) -> Result<Arc<dyn VenueAdapter>> {
    let adapter = CcxtAdapter::new(name, config.clone()).await?;
    Ok(Arc::new(adapter))
}

/// Create adapter from Hummingbot config
pub async fn create_hummingbot_adapter(
    name: &str,
    config: &HummingbotConfig,
) -> Result<Arc<dyn VenueAdapter>> {
    let adapter = HummingbotAdapter::new(name, config.clone()).await?;
    Ok(Arc::new(adapter))
}
