// LX Trading SDK - Arbitrage Types Implementation

#include <lx/trading/arbitrage/types.hpp>

namespace lx::trading::arbitrage {

CrossChainConfig CrossChainConfig::defaults() {
    CrossChainConfig config;

    // Lux ecosystem (Warp enabled)
    config.chains["lux_mainnet"] = CrossChainInfo{
        .chain_id = "lux_mainnet",
        .name = "Lux Mainnet",
        .chain_type = ChainType::LuxSubnet,
        .block_time_ms = 400,
        .finality_ms = 400,
        .warp_supported = true,
        .teleport_supported = true,
        .venues = {"lx_dex", "lx_amm"}
    };

    config.chains["lx_dex_subnet"] = CrossChainInfo{
        .chain_id = "lx_dex_subnet",
        .name = "LX DEX Subnet",
        .chain_type = ChainType::LuxSubnet,
        .block_time_ms = 200,
        .finality_ms = 200,
        .warp_supported = true,
        .teleport_supported = false,
        .venues = {"lx_dex"}
    };

    // EVM chains (Teleport enabled)
    config.chains["ethereum"] = CrossChainInfo{
        .chain_id = "1",
        .name = "Ethereum",
        .chain_type = ChainType::Evm,
        .block_time_ms = 12000,
        .finality_ms = 15 * 60 * 1000,  // 15 minutes
        .warp_supported = false,
        .teleport_supported = true,
        .venues = {"uniswap", "sushiswap"}
    };

    config.chains["bsc"] = CrossChainInfo{
        .chain_id = "56",
        .name = "BNB Smart Chain",
        .chain_type = ChainType::Evm,
        .block_time_ms = 3000,
        .finality_ms = 45000,
        .warp_supported = false,
        .teleport_supported = true,
        .venues = {"pancakeswap"}
    };

    config.chains["arbitrum"] = CrossChainInfo{
        .chain_id = "42161",
        .name = "Arbitrum One",
        .chain_type = ChainType::Evm,
        .block_time_ms = 250,
        .finality_ms = 15 * 60 * 1000,
        .warp_supported = false,
        .teleport_supported = true,
        .venues = {"uniswap", "camelot"}
    };

    // CEX (API only)
    config.chains["binance"] = CrossChainInfo{
        .chain_id = "binance",
        .name = "Binance",
        .chain_type = ChainType::Cex,
        .block_time_ms = 0,
        .finality_ms = 0,
        .warp_supported = false,
        .teleport_supported = false,
        .venues = {"binance"}
    };

    config.chains["mexc"] = CrossChainInfo{
        .chain_id = "mexc",
        .name = "MEXC",
        .chain_type = ChainType::Cex,
        .block_time_ms = 0,
        .finality_ms = 0,
        .warp_supported = false,
        .teleport_supported = false,
        .venues = {"mexc"}
    };

    return config;
}

}  // namespace lx::trading::arbitrage
