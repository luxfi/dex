// Package arbitrage provides cross-chain arbitrage via Warp and Teleport
package arbitrage

import (
	"context"
	"fmt"
	"time"

	"github.com/shopspring/decimal"
)

/*
CROSS-CHAIN ARBITRAGE TRANSPORTS

1. WARP (Lux Native)
   - Only works WITHIN Lux ecosystem (between subnets)
   - Sub-second message delivery
   - Use for: LX DEX <-> LX AMM <-> Other Lux subnets
   - Cannot reach external chains

2. TELEPORT (EVM Bridge)
   - Works with ANY EVM-compatible chain
   - Lux <-> Ethereum, BSC, Arbitrum, Polygon, etc.
   - ~30 second finality (depends on source chain)
   - Uses validator attestations

3. CEX API
   - No bridging needed - just API calls
   - Sub-second execution
   - Settlement via withdraw/deposit (slow but doesn't block arb)

4. FOR OMNICHAIN ARBITRAGE:
   - Lux internal: Warp (instant)
   - External EVM: Teleport (~30s)
   - CEX: Direct API (instant trade, later settle)
*/

// CrossChainTransport represents a cross-chain messaging protocol
type CrossChainTransport string

const (
	TransportWarp     CrossChainTransport = "warp"     // Lux native, subnets only
	TransportTeleport CrossChainTransport = "teleport" // EVM bridge
	TransportDirect   CrossChainTransport = "direct"   // Same chain, no bridge
	TransportCEXAPI   CrossChainTransport = "cex_api"  // CEX API calls
)

// ChainType represents the type of blockchain
type ChainType string

const (
	ChainTypeLuxSubnet ChainType = "lux_subnet"
	ChainTypeEVM       ChainType = "evm"
	ChainTypeCEX       ChainType = "cex" // Centralized exchange (not a chain)
)

// CrossChainConfig holds cross-chain transport configuration
type CrossChainConfig struct {
	// Warp configuration (Lux internal)
	WarpEnabled  bool
	WarpEndpoint string
	WarpTimeout  time.Duration

	// Teleport configuration (EVM bridging)
	TeleportEnabled    bool
	TeleportEndpoint   string
	TeleportTimeout    time.Duration
	TeleportValidators []string

	// Chain registry
	Chains map[string]CrossChainInfo
}

// CrossChainInfo holds information about a chain
type CrossChainInfo struct {
	ChainID           string
	Name              string
	ChainType         ChainType
	BlockTime         time.Duration
	Finality          time.Duration
	WarpSupported     bool     // Can use Warp (Lux subnets only)
	TeleportSupported bool     // Can use Teleport (EVM chains)
	Venues            []string // Trading venues on this chain
}

// CrossChainRouter routes messages between chains
type CrossChainRouter struct {
	config   CrossChainConfig
	warp     WarpClient
	teleport TeleportClient
}

// WarpClient interface for Lux Warp messaging
type WarpClient interface {
	// SendMessage sends a Warp message to another Lux subnet
	SendMessage(ctx context.Context, destSubnet string, payload []byte) (string, error)

	// ReceiveMessage waits for a Warp message
	ReceiveMessage(ctx context.Context, messageID string) ([]byte, error)

	// GetBlockchainID returns this subnet's ID
	GetBlockchainID() string
}

// TeleportClient interface for EVM bridging
type TeleportClient interface {
	// Bridge bridges assets to another EVM chain
	Bridge(ctx context.Context, destChain string, token string, amount decimal.Decimal) (string, error)

	// GetBridgeStatus checks bridge transaction status
	GetBridgeStatus(ctx context.Context, txID string) (BridgeStatus, error)

	// EstimateBridgeFee estimates the bridge fee
	EstimateBridgeFee(ctx context.Context, destChain string, token string, amount decimal.Decimal) (decimal.Decimal, error)
}

// BridgeStatus represents bridge transaction status
type BridgeStatus struct {
	TxID        string
	Status      string // pending, confirming, completed, failed
	SourceChain string
	DestChain   string
	Amount      decimal.Decimal
	Fee         decimal.Decimal
	SourceTx    string
	DestTx      string
	Timestamp   time.Time
}

// NewCrossChainRouter creates a new cross-chain router
func NewCrossChainRouter(config CrossChainConfig) *CrossChainRouter {
	return &CrossChainRouter{
		config: config,
	}
}

// SetWarpClient sets the Warp client
func (r *CrossChainRouter) SetWarpClient(client WarpClient) {
	r.warp = client
}

// SetTeleportClient sets the Teleport client
func (r *CrossChainRouter) SetTeleportClient(client TeleportClient) {
	r.teleport = client
}

// DetermineTransport determines the best transport between two chains
func (r *CrossChainRouter) DetermineTransport(sourceChain, destChain string) CrossChainTransport {
	src := r.config.Chains[sourceChain]
	dst := r.config.Chains[destChain]

	// Same chain = direct
	if sourceChain == destChain {
		return TransportDirect
	}

	// CEX = API
	if src.ChainType == ChainTypeCEX || dst.ChainType == ChainTypeCEX {
		return TransportCEXAPI
	}

	// Both Lux subnets = Warp (fastest)
	if src.ChainType == ChainTypeLuxSubnet && dst.ChainType == ChainTypeLuxSubnet {
		if src.WarpSupported && dst.WarpSupported && r.config.WarpEnabled {
			return TransportWarp
		}
	}

	// Both EVM or mixed = Teleport
	if src.TeleportSupported && dst.TeleportSupported && r.config.TeleportEnabled {
		return TransportTeleport
	}

	// No viable transport
	return ""
}

// EstimateLatency estimates the latency for cross-chain message
func (r *CrossChainRouter) EstimateLatency(sourceChain, destChain string) time.Duration {
	transport := r.DetermineTransport(sourceChain, destChain)

	switch transport {
	case TransportDirect:
		return 0
	case TransportWarp:
		return 500 * time.Millisecond // Sub-second
	case TransportCEXAPI:
		return 100 * time.Millisecond // API call
	case TransportTeleport:
		// Depends on source chain finality
		src := r.config.Chains[sourceChain]
		return src.Finality + 10*time.Second // Finality + processing
	default:
		return time.Hour // Unknown/unsupported
	}
}

// EstimateCost estimates the cost for cross-chain transfer
func (r *CrossChainRouter) EstimateCost(ctx context.Context, sourceChain, destChain, token string, amount decimal.Decimal) (decimal.Decimal, error) {
	transport := r.DetermineTransport(sourceChain, destChain)

	switch transport {
	case TransportDirect:
		return decimal.Zero, nil
	case TransportWarp:
		return decimal.NewFromFloat(0.001), nil // Nearly free
	case TransportCEXAPI:
		return decimal.Zero, nil // No bridge cost (but withdrawal fees apply)
	case TransportTeleport:
		if r.teleport != nil {
			return r.teleport.EstimateBridgeFee(ctx, destChain, token, amount)
		}
		return decimal.NewFromFloat(1.0), nil // Estimate $1
	default:
		return decimal.Zero, fmt.Errorf("no viable transport")
	}
}

// DefaultCrossChainConfig returns default configuration with common chains
func DefaultCrossChainConfig() CrossChainConfig {
	return CrossChainConfig{
		WarpEnabled:     true,
		WarpTimeout:     5 * time.Second,
		TeleportEnabled: true,
		TeleportTimeout: 60 * time.Second,
		Chains: map[string]CrossChainInfo{
			// Lux ecosystem (Warp enabled)
			"lux_mainnet": {
				ChainID:           "lux_mainnet",
				Name:              "Lux Mainnet",
				ChainType:         ChainTypeLuxSubnet,
				BlockTime:         400 * time.Millisecond,
				Finality:          400 * time.Millisecond,
				WarpSupported:     true,
				TeleportSupported: true,
				Venues:            []string{"lx_dex", "lx_amm"},
			},
			"lx_dex_subnet": {
				ChainID:           "lx_dex_subnet",
				Name:              "LX DEX Subnet",
				ChainType:         ChainTypeLuxSubnet,
				BlockTime:         200 * time.Millisecond,
				Finality:          200 * time.Millisecond,
				WarpSupported:     true,
				TeleportSupported: false,
				Venues:            []string{"lx_dex"},
			},

			// EVM chains (Teleport enabled)
			"ethereum": {
				ChainID:           "1",
				Name:              "Ethereum",
				ChainType:         ChainTypeEVM,
				BlockTime:         12 * time.Second,
				Finality:          15 * time.Minute, // Conservative
				WarpSupported:     false,
				TeleportSupported: true,
				Venues:            []string{"uniswap", "sushiswap"},
			},
			"bsc": {
				ChainID:           "56",
				Name:              "BNB Smart Chain",
				ChainType:         ChainTypeEVM,
				BlockTime:         3 * time.Second,
				Finality:          45 * time.Second,
				WarpSupported:     false,
				TeleportSupported: true,
				Venues:            []string{"pancakeswap"},
			},
			"arbitrum": {
				ChainID:           "42161",
				Name:              "Arbitrum One",
				ChainType:         ChainTypeEVM,
				BlockTime:         250 * time.Millisecond,
				Finality:          15 * time.Minute, // ETH finality
				WarpSupported:     false,
				TeleportSupported: true,
				Venues:            []string{"uniswap", "camelot"},
			},
			"polygon": {
				ChainID:           "137",
				Name:              "Polygon",
				ChainType:         ChainTypeEVM,
				BlockTime:         2 * time.Second,
				Finality:          30 * time.Second,
				WarpSupported:     false,
				TeleportSupported: true,
				Venues:            []string{"quickswap"},
			},

			// CEX (API only)
			"binance": {
				ChainID:           "binance",
				Name:              "Binance",
				ChainType:         ChainTypeCEX,
				BlockTime:         0,
				Finality:          0, // Instant
				WarpSupported:     false,
				TeleportSupported: false,
				Venues:            []string{"binance"},
			},
			"mexc": {
				ChainID:           "mexc",
				Name:              "MEXC",
				ChainType:         ChainTypeCEX,
				BlockTime:         0,
				Finality:          0,
				WarpSupported:     false,
				TeleportSupported: false,
				Venues:            []string{"mexc"},
			},
			"okx": {
				ChainID:           "okx",
				Name:              "OKX",
				ChainType:         ChainTypeCEX,
				BlockTime:         0,
				Finality:          0,
				WarpSupported:     false,
				TeleportSupported: false,
				Venues:            []string{"okx"},
			},
		},
	}
}

// ArbitrageOpportunityWithRouting extends opportunity with routing info
type ArbitrageOpportunityWithRouting struct {
	UnifiedOpportunity

	// Routing information
	Transport        CrossChainTransport
	EstimatedLatency time.Duration
	BridgeCost       decimal.Decimal

	// Adjusted profitability
	AdjustedNetProfit decimal.Decimal
}

// EnhanceOpportunity adds routing information to an opportunity
func (r *CrossChainRouter) EnhanceOpportunity(ctx context.Context, opp *UnifiedOpportunity) *ArbitrageOpportunityWithRouting {
	enhanced := &ArbitrageOpportunityWithRouting{
		UnifiedOpportunity: *opp,
	}

	// Determine transport
	buyChain := r.venueToChain(opp.BuyVenue)
	sellChain := r.venueToChain(opp.SellVenue)

	enhanced.Transport = r.DetermineTransport(buyChain, sellChain)
	enhanced.EstimatedLatency = r.EstimateLatency(buyChain, sellChain)

	// Estimate bridge cost
	bridgeCost, _ := r.EstimateCost(ctx, buyChain, sellChain, opp.Symbol, opp.MaxSize)
	enhanced.BridgeCost = bridgeCost

	// Adjust profit
	enhanced.AdjustedNetProfit = opp.NetProfit.Sub(bridgeCost)

	return enhanced
}

// venueToChain maps venue to chain
func (r *CrossChainRouter) venueToChain(venue string) string {
	for chainID, info := range r.config.Chains {
		for _, v := range info.Venues {
			if v == venue {
				return chainID
			}
		}
	}
	return venue // Fallback to venue name
}
