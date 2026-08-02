//! Cross-Chain Arbitrage Transports.
//!
//! 1. WARP (Lux Native)
//!    - Only works WITHIN Lux ecosystem (between subnets)
//!    - Sub-second message delivery (<500ms)
//!    - Use for: LX DEX <-> LX AMM <-> Other Lux subnets
//!    - Cannot reach external chains
//!
//! 2. TELEPORT (EVM Bridge)
//!    - Works with ANY EVM-compatible chain
//!    - Lux <-> Ethereum, BSC, Arbitrum, Polygon, etc.
//!    - ~30 second finality (depends on source chain)
//!    - Uses validator attestations
//!
//! 3. CEX API
//!    - No bridging needed - just API calls
//!    - Sub-second execution
//!    - Settlement via withdraw/deposit (slow but doesn't block arb)
//!
//! 4. FOR OMNICHAIN ARBITRAGE:
//!    - Lux internal: Warp (instant)
//!    - External EVM: Teleport (~30s)
//!    - CEX: Direct API (instant trade, later settle)

use crate::arbitrage::types::*;
use async_trait::async_trait;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Warp client interface for Lux-native messaging.
#[async_trait]
pub trait WarpClient: Send + Sync {
    /// Send a Warp message to another Lux subnet.
    async fn send_message(&self, dest_subnet: &str, payload: &[u8]) -> Result<String, String>;

    /// Receive a Warp message.
    async fn receive_message(&self, message_id: &str) -> Result<Vec<u8>, String>;

    /// Get this subnet's ID.
    fn get_blockchain_id(&self) -> String;
}

/// Teleport client interface for EVM bridging.
#[async_trait]
pub trait TeleportClient: Send + Sync {
    /// Bridge assets to another EVM chain.
    async fn bridge(
        &self,
        dest_chain: &str,
        token: &str,
        amount: Decimal,
    ) -> Result<String, String>;

    /// Get bridge transaction status.
    async fn get_bridge_status(&self, tx_id: &str) -> Result<BridgeStatus, String>;

    /// Estimate bridge fee.
    async fn estimate_bridge_fee(
        &self,
        dest_chain: &str,
        token: &str,
        amount: Decimal,
    ) -> Result<Decimal, String>;
}

/// Cross-chain router for determining optimal transport.
pub struct CrossChainRouter {
    config: CrossChainConfig,
    warp: Arc<RwLock<Option<Arc<dyn WarpClient>>>>,
    teleport: Arc<RwLock<Option<Arc<dyn TeleportClient>>>>,
}

impl CrossChainRouter {
    /// Create a new cross-chain router.
    pub fn new(config: CrossChainConfig) -> Self {
        Self {
            config,
            warp: Arc::new(RwLock::new(None)),
            teleport: Arc::new(RwLock::new(None)),
        }
    }

    /// Set the Warp client.
    pub async fn set_warp_client(&self, client: Arc<dyn WarpClient>) {
        let mut warp = self.warp.write().await;
        *warp = Some(client);
    }

    /// Set the Teleport client.
    pub async fn set_teleport_client(&self, client: Arc<dyn TeleportClient>) {
        let mut teleport = self.teleport.write().await;
        *teleport = Some(client);
    }

    /// Get the Warp client.
    pub async fn warp(&self) -> Option<Arc<dyn WarpClient>> {
        let warp = self.warp.read().await;
        warp.clone()
    }

    /// Get the Teleport client.
    pub async fn teleport(&self) -> Option<Arc<dyn TeleportClient>> {
        let teleport = self.teleport.read().await;
        teleport.clone()
    }

    /// Determine the best transport between two chains.
    pub fn determine_transport(&self, source_chain: &str, dest_chain: &str) -> CrossChainTransport {
        let src = self.config.chains.get(source_chain);
        let dst = self.config.chains.get(dest_chain);

        // Same chain = direct
        if source_chain == dest_chain {
            return CrossChainTransport::Direct;
        }

        // CEX = API
        if let Some(s) = src {
            if s.chain_type == ChainType::Cex {
                return CrossChainTransport::CexApi;
            }
        }
        if let Some(d) = dst {
            if d.chain_type == ChainType::Cex {
                return CrossChainTransport::CexApi;
            }
        }

        // Both Lux subnets = Warp (fastest)
        if let (Some(s), Some(d)) = (src, dst) {
            if s.chain_type == ChainType::LuxSubnet
                && d.chain_type == ChainType::LuxSubnet
                && s.warp_supported
                && d.warp_supported
                && self.config.warp_enabled
            {
                return CrossChainTransport::Warp;
            }
        }

        // Both EVM or mixed = Teleport
        if let (Some(s), Some(d)) = (src, dst) {
            if s.teleport_supported && d.teleport_supported && self.config.teleport_enabled {
                return CrossChainTransport::Teleport;
            }
        }

        // No viable transport - return Direct as fallback
        CrossChainTransport::Direct
    }

    /// Estimate latency for cross-chain message (ms).
    pub fn estimate_latency(&self, source_chain: &str, dest_chain: &str) -> i64 {
        let transport = self.determine_transport(source_chain, dest_chain);

        match transport {
            CrossChainTransport::Direct => 0,
            CrossChainTransport::Warp => 500,   // Sub-second
            CrossChainTransport::CexApi => 100, // API call
            CrossChainTransport::Teleport => {
                let src = self.config.chains.get(source_chain);
                src.map(|s| s.finality_ms as i64 + 10000).unwrap_or(3600000)
                // Finality + processing
            }
        }
    }

    /// Estimate cost for cross-chain transfer.
    pub async fn estimate_cost(
        &self,
        source_chain: &str,
        dest_chain: &str,
        token: &str,
        amount: Decimal,
    ) -> Decimal {
        let transport = self.determine_transport(source_chain, dest_chain);

        match transport {
            CrossChainTransport::Direct => Decimal::ZERO,
            CrossChainTransport::Warp => Decimal::new(1, 3), // 0.001 - Nearly free
            CrossChainTransport::CexApi => Decimal::ZERO,    // No bridge cost
            CrossChainTransport::Teleport => {
                let teleport = self.teleport.read().await;
                if let Some(ref client) = *teleport {
                    client
                        .estimate_bridge_fee(dest_chain, token, amount)
                        .await
                        .unwrap_or_else(|_| Decimal::from(1))
                } else {
                    Decimal::from(1) // Estimate $1
                }
            }
        }
    }

    /// Get chain ID from venue name.
    pub fn venue_to_chain(&self, venue: &str) -> String {
        for (chain_id, info) in &self.config.chains {
            if info.venues.contains(&venue.to_string()) {
                return chain_id.clone();
            }
        }
        venue.to_string() // Fallback to venue name
    }

    /// Enhance an opportunity with routing information.
    pub async fn enhance_opportunity(&self, opp: UnifiedOpportunity) -> EnhancedOpportunity {
        let buy_chain = self.venue_to_chain(&opp.buy_venue);
        let sell_chain = self.venue_to_chain(&opp.sell_venue);

        let transport = self.determine_transport(&buy_chain, &sell_chain);
        let estimated_latency = self.estimate_latency(&buy_chain, &sell_chain);
        let bridge_cost = self
            .estimate_cost(&buy_chain, &sell_chain, &opp.symbol, opp.max_size)
            .await;

        EnhancedOpportunity {
            base: opp.clone(),
            transport,
            estimated_latency,
            bridge_cost,
            adjusted_net_profit: opp.net_profit - bridge_cost,
        }
    }

    /// Get current config.
    pub fn config(&self) -> &CrossChainConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_transport_same_chain() {
        let config = CrossChainConfig::default();
        let router = CrossChainRouter::new(config);

        let transport = router.determine_transport("lux_mainnet", "lux_mainnet");
        assert_eq!(transport, CrossChainTransport::Direct);
    }

    #[test]
    fn test_determine_transport_lux_subnets() {
        let config = CrossChainConfig::default();
        let router = CrossChainRouter::new(config);

        let transport = router.determine_transport("lux_mainnet", "lx_dex_subnet");
        assert_eq!(transport, CrossChainTransport::Warp);
    }

    #[test]
    fn test_determine_transport_cex() {
        let config = CrossChainConfig::default();
        let router = CrossChainRouter::new(config);

        let transport = router.determine_transport("binance", "lux_mainnet");
        assert_eq!(transport, CrossChainTransport::CexApi);
    }

    #[test]
    fn test_determine_transport_evm() {
        let config = CrossChainConfig::default();
        let router = CrossChainRouter::new(config);

        let transport = router.determine_transport("ethereum", "arbitrum");
        assert_eq!(transport, CrossChainTransport::Teleport);
    }

    #[test]
    fn test_estimate_latency() {
        let config = CrossChainConfig::default();
        let router = CrossChainRouter::new(config);

        // Direct = 0
        let latency = router.estimate_latency("lux_mainnet", "lux_mainnet");
        assert_eq!(latency, 0);

        // Warp = 500ms
        let latency = router.estimate_latency("lux_mainnet", "lx_dex_subnet");
        assert_eq!(latency, 500);

        // CEX = 100ms
        let latency = router.estimate_latency("binance", "lux_mainnet");
        assert_eq!(latency, 100);
    }

    #[test]
    fn test_venue_to_chain() {
        let config = CrossChainConfig::default();
        let router = CrossChainRouter::new(config);

        let chain = router.venue_to_chain("lx_dex");
        assert!(chain == "lux_mainnet" || chain == "lx_dex_subnet");

        let chain = router.venue_to_chain("binance");
        assert_eq!(chain, "binance");

        let chain = router.venue_to_chain("uniswap");
        assert!(chain == "ethereum" || chain == "arbitrum");
    }
}
