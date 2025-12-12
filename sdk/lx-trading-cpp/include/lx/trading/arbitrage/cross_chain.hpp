// LX Trading SDK - Cross-Chain Arbitrage Transports
//
// 1. WARP (Lux Native)
//    - Only works WITHIN Lux ecosystem (between subnets)
//    - Sub-second message delivery (<500ms)
//    - Use for: LX DEX <-> LX AMM <-> Other Lux subnets
//    - Cannot reach external chains
//
// 2. TELEPORT (EVM Bridge)
//    - Works with ANY EVM-compatible chain
//    - Lux <-> Ethereum, BSC, Arbitrum, Polygon, etc.
//    - ~30 second finality (depends on source chain)
//    - Uses validator attestations
//
// 3. CEX API
//    - No bridging needed - just API calls
//    - Sub-second execution
//    - Settlement via withdraw/deposit (slow but doesn't block arb)
//
// 4. FOR OMNICHAIN ARBITRAGE:
//    - Lux internal: Warp (instant)
//    - External EVM: Teleport (~30s)
//    - CEX: Direct API (instant trade, later settle)

#pragma once

#include <lx/trading/arbitrage/types.hpp>
#include <memory>
#include <mutex>

namespace lx::trading::arbitrage {

/// Warp client interface for Lux-native messaging
class WarpClient {
public:
    virtual ~WarpClient() = default;

    /// Send a Warp message to another Lux subnet
    virtual std::string send_message(const std::string& dest_subnet,
                                     const std::vector<uint8_t>& payload) = 0;

    /// Receive a Warp message
    virtual std::vector<uint8_t> receive_message(const std::string& message_id) = 0;

    /// Get this subnet's ID
    virtual std::string get_blockchain_id() const = 0;
};

/// Teleport client interface for EVM bridging
class TeleportClient {
public:
    virtual ~TeleportClient() = default;

    /// Bridge assets to another EVM chain
    virtual std::string bridge(const std::string& dest_chain,
                               const std::string& token,
                               Decimal amount) = 0;

    /// Get bridge transaction status
    virtual BridgeStatus get_bridge_status(const std::string& tx_id) = 0;

    /// Estimate bridge fee
    virtual Decimal estimate_bridge_fee(const std::string& dest_chain,
                                        const std::string& token,
                                        Decimal amount) = 0;
};

/// Cross-chain router for determining optimal transport
class CrossChainRouter {
public:
    explicit CrossChainRouter(CrossChainConfig config);
    ~CrossChainRouter() = default;

    // Non-copyable
    CrossChainRouter(const CrossChainRouter&) = delete;
    CrossChainRouter& operator=(const CrossChainRouter&) = delete;

    /// Set the Warp client
    void set_warp_client(std::shared_ptr<WarpClient> client);

    /// Set the Teleport client
    void set_teleport_client(std::shared_ptr<TeleportClient> client);

    /// Get the Warp client
    [[nodiscard]] std::shared_ptr<WarpClient> warp() const;

    /// Get the Teleport client
    [[nodiscard]] std::shared_ptr<TeleportClient> teleport() const;

    /// Determine the best transport between two chains
    [[nodiscard]] CrossChainTransport determine_transport(const std::string& source_chain,
                                                          const std::string& dest_chain) const;

    /// Estimate latency for cross-chain message (ms)
    [[nodiscard]] int64_t estimate_latency(const std::string& source_chain,
                                           const std::string& dest_chain) const;

    /// Estimate cost for cross-chain transfer
    [[nodiscard]] Decimal estimate_cost(const std::string& source_chain,
                                        const std::string& dest_chain,
                                        const std::string& token,
                                        Decimal amount) const;

    /// Get chain ID from venue name
    [[nodiscard]] std::string venue_to_chain(const std::string& venue) const;

    /// Enhance an opportunity with routing information
    [[nodiscard]] EnhancedOpportunity enhance_opportunity(const UnifiedOpportunity& opp) const;

    /// Get current config
    [[nodiscard]] const CrossChainConfig& config() const noexcept { return config_; }

private:
    CrossChainConfig config_;
    std::shared_ptr<WarpClient> warp_client_;
    std::shared_ptr<TeleportClient> teleport_client_;
    mutable std::mutex warp_mutex_;
    mutable std::mutex teleport_mutex_;
};

}  // namespace lx::trading::arbitrage
