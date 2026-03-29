package trading

import (
	"context"
	"fmt"
	"math/big"
	"time"
)

// --- Warp Bridge (Lux AWM) ---
// Native cross-subnet messaging via precompile 0x0200000000000000000000000000000000000005.
// Near-instant finality between Lux subnets (e.g. Lux↔Zoo).

const WarpPrecompile = "0x0200000000000000000000000000000000000005"

type WarpBridge struct {
	routes [][2]int
	tokens map[int][]string
	feeBPS int
}

type WarpBridgeConfig struct {
	Routes [][2]int
	Tokens map[int][]string
	FeeBPS int
}

func NewWarpBridge(cfg WarpBridgeConfig) *WarpBridge {
	feeBPS := cfg.FeeBPS
	if feeBPS == 0 {
		feeBPS = 5
	}
	return &WarpBridge{routes: cfg.Routes, tokens: cfg.Tokens, feeBPS: feeBPS}
}

func (b *WarpBridge) BridgeType() BridgeType   { return BridgeWarp }
func (b *WarpBridge) SupportedRoutes() [][2]int { return b.routes }
func (b *WarpBridge) SupportedTokens(src, _ int) []string {
	return b.tokens[src]
}

func (b *WarpBridge) QuoteBridge(_ context.Context, req BridgeQuoteRequest) (*BridgeQuote, error) {
	if !bridgeSupportsRoute(b.routes, req.SourceChain, req.DestChain) {
		return nil, nil
	}
	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}
	fee := new(big.Int).Mul(amountIn, big.NewInt(int64(b.feeBPS)))
	fee.Div(fee, big.NewInt(10000))
	amountOut := new(big.Int).Sub(amountIn, fee)
	return &BridgeQuote{
		Bridge: BridgeWarp, AmountOut: amountOut.String(),
		Fee: fee.String(), GasEstimate: "100000",
		Duration: 2 * time.Second,
	}, nil
}

// --- Teleport Bridge (EIP-712 + MPC Oracles) ---
// Cross-L1 bridge for Ethereum↔Lux, Base↔Lux via precompile 0x6010.

const TeleportPrecompile = "0x0000000000000000000000000000000000006010"

type TeleportBridge struct {
	routes [][2]int
	tokens map[int][]string
	feeBPS int
}

type TeleportBridgeConfig struct {
	Routes [][2]int
	Tokens map[int][]string
	FeeBPS int
}

func NewTeleportBridge(cfg TeleportBridgeConfig) *TeleportBridge {
	feeBPS := cfg.FeeBPS
	if feeBPS == 0 {
		feeBPS = 15
	}
	return &TeleportBridge{routes: cfg.Routes, tokens: cfg.Tokens, feeBPS: feeBPS}
}

func (b *TeleportBridge) BridgeType() BridgeType   { return BridgeTeleport }
func (b *TeleportBridge) SupportedRoutes() [][2]int { return b.routes }
func (b *TeleportBridge) SupportedTokens(src, _ int) []string {
	return b.tokens[src]
}

func (b *TeleportBridge) QuoteBridge(_ context.Context, req BridgeQuoteRequest) (*BridgeQuote, error) {
	if !bridgeSupportsRoute(b.routes, req.SourceChain, req.DestChain) {
		return nil, nil
	}
	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}
	fee := new(big.Int).Mul(amountIn, big.NewInt(int64(b.feeBPS)))
	fee.Div(fee, big.NewInt(10000))
	amountOut := new(big.Int).Sub(amountIn, fee)
	return &BridgeQuote{
		Bridge: BridgeTeleport, AmountOut: amountOut.String(),
		Fee: fee.String(), GasEstimate: "200000",
		Duration: 15 * time.Minute,
	}, nil
}

// --- CCTP Bridge (Circle USDC) ---
// Domains: Ethereum=0, Base=6, Arbitrum=3, Optimism=2, Polygon=7.
// USDC only, no bridge fee (gas only).

type CCTPBridge struct {
	routes [][2]int
	tokens map[int][]string
	feeBPS int
}

type CCTPBridgeConfig struct {
	Routes [][2]int
	Tokens map[int][]string
	FeeBPS int
}

func NewCCTPBridge(cfg CCTPBridgeConfig) *CCTPBridge {
	return &CCTPBridge{routes: cfg.Routes, tokens: cfg.Tokens, feeBPS: cfg.FeeBPS}
}

func (b *CCTPBridge) BridgeType() BridgeType   { return BridgeCCTP }
func (b *CCTPBridge) SupportedRoutes() [][2]int { return b.routes }
func (b *CCTPBridge) SupportedTokens(src, _ int) []string {
	return b.tokens[src]
}

func (b *CCTPBridge) QuoteBridge(_ context.Context, req BridgeQuoteRequest) (*BridgeQuote, error) {
	if !bridgeSupportsRoute(b.routes, req.SourceChain, req.DestChain) {
		return nil, nil
	}
	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}
	fee := new(big.Int).Mul(amountIn, big.NewInt(int64(b.feeBPS)))
	fee.Div(fee, big.NewInt(10000))
	amountOut := new(big.Int).Sub(amountIn, fee)
	return &BridgeQuote{
		Bridge: BridgeCCTP, AmountOut: amountOut.String(),
		Fee: fee.String(), GasEstimate: "65000",
		Duration: 20 * time.Minute,
	}, nil
}

// --- Lux Bridge (Precompiles 0x0440-0x0445) ---
// Gateway (0x0440), Router (0x0441), Verifier (0x0442),
// Pool (0x0443), FeeManager (0x0444), Signer (0x0445).

const (
	LuxBridgeGateway  = "0x0000000000000000000000000000000000000440"
	LuxBridgeRouter   = "0x0000000000000000000000000000000000000441"
	LuxBridgeVerifier = "0x0000000000000000000000000000000000000442"
)

type LuxBridgeVenue struct {
	routes [][2]int
	tokens map[int][]string
	feeBPS int
}

type LuxBridgeConfig struct {
	Routes [][2]int
	Tokens map[int][]string
	FeeBPS int
}

func NewLuxBridge(cfg LuxBridgeConfig) *LuxBridgeVenue {
	feeBPS := cfg.FeeBPS
	if feeBPS == 0 {
		feeBPS = 10
	}
	return &LuxBridgeVenue{routes: cfg.Routes, tokens: cfg.Tokens, feeBPS: feeBPS}
}

func (b *LuxBridgeVenue) BridgeType() BridgeType   { return BridgeLux }
func (b *LuxBridgeVenue) SupportedRoutes() [][2]int { return b.routes }
func (b *LuxBridgeVenue) SupportedTokens(src, _ int) []string {
	return b.tokens[src]
}

func (b *LuxBridgeVenue) QuoteBridge(_ context.Context, req BridgeQuoteRequest) (*BridgeQuote, error) {
	if !bridgeSupportsRoute(b.routes, req.SourceChain, req.DestChain) {
		return nil, nil
	}
	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}
	fee := new(big.Int).Mul(amountIn, big.NewInt(int64(b.feeBPS)))
	fee.Div(fee, big.NewInt(10000))
	amountOut := new(big.Int).Sub(amountIn, fee)
	return &BridgeQuote{
		Bridge: BridgeLux, AmountOut: amountOut.String(),
		Fee: fee.String(), GasEstimate: "150000",
		Duration: 5 * time.Minute,
	}, nil
}

// --- Shared ---

func bridgeSupportsRoute(routes [][2]int, src, dst int) bool {
	for _, r := range routes {
		if (r[0] == src && r[1] == dst) || (r[0] == dst && r[1] == src) {
			return true
		}
	}
	return false
}
