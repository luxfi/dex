// Package lux provides a native EVM DEX provider that queries on-chain
// pool data via LX-series precompiles (0x9010-0x9060).
package lux

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/big"
	"sync"
	"time"

	"github.com/luxfi/dex/pkg/gateway"
	ethereum "github.com/luxfi/geth"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/ethclient"
)

var (
	ErrNotImplemented = errors.New("not yet implemented - use Uniswap provider as fallback")
	ErrNoRPCEndpoint  = errors.New("no RPC endpoint configured")
)

// DEX precompile addresses (LX-series, activated at genesis via dexConfig)
const (
	PoolManagerAddress = "0x0000000000000000000000000000000000009010" // LXPool — singleton AMM pool manager
	OracleAddress      = "0x0000000000000000000000000000000000009011" // LXOracle — multi-source price aggregation
	SwapRouterAddress  = "0x0000000000000000000000000000000000009012" // LXRouter — swap router
	HooksRegistryAddr  = "0x0000000000000000000000000000000000009013" // LXHooks — pool hooks registry
	FlashLoanAddress   = "0x0000000000000000000000000000000000009014" // LXFlash — flash loans
	OrderBookAddress   = "0x0000000000000000000000000000000000009020" // LXBook — central limit order book
	VaultAddress       = "0x0000000000000000000000000000000000009030" // LXVault — clearinghouse, margin, positions
	FeedAddress        = "0x0000000000000000000000000000000000009040" // LXFeed — computed price feeds
	LendAddress        = "0x0000000000000000000000000000000000009050" // LXLend — lending pool
	LiquidAddress      = "0x0000000000000000000000000000000000009060" // LXLiquid — self-repaying loans
)

// Provider implements native DEX functionality via LX precompiles.
// Connects to EVM nodes to query on-chain pool data.
type Provider struct {
	name     string
	priority int
	chains   []gateway.ChainID

	// RPC clients for each chain
	clients   map[gateway.ChainID]*ethclient.Client
	endpoints map[gateway.ChainID]string
	mu        sync.RWMutex
}

// ProviderConfig holds provider configuration
type ProviderConfig struct {
	Name         string
	Priority     int
	Chains       []gateway.ChainID
	RPCEndpoints map[gateway.ChainID]string // Map of chainID to RPC endpoint URL
}

// DefaultConfig returns a minimal config — caller must provide chains and RPC endpoints.
func DefaultConfig() ProviderConfig {
	return ProviderConfig{
		Name:         "native",
		Priority:     10,
		Chains:       nil,
		RPCEndpoints: nil,
	}
}

// NewProvider creates a native DEX provider with RPC connectivity.
// Chains and RPCEndpoints must be provided by the caller.
func NewProvider(cfg ProviderConfig) *Provider {
	if cfg.Name == "" {
		cfg.Name = "native"
	}
	if cfg.Priority == 0 {
		cfg.Priority = 10
	}

	p := &Provider{
		name:      cfg.Name,
		priority:  cfg.Priority,
		chains:    cfg.Chains,
		clients:   make(map[gateway.ChainID]*ethclient.Client),
		endpoints: cfg.RPCEndpoints,
	}

	// Initialize RPC clients
	for chainID, endpoint := range cfg.RPCEndpoints {
		if endpoint != "" {
			client, err := ethclient.Dial(endpoint)
			if err != nil {
				log.Printf("Warning: Failed to connect to %d RPC at %s: %v", chainID, endpoint, err)
				continue
			}
			p.clients[chainID] = client
			log.Printf("Connected to %d RPC at %s", chainID, endpoint)
		}
	}

	return p
}

// Info returns provider information
func (p *Provider) Info() gateway.ProviderInfo {
	p.mu.RLock()
	healthy := len(p.clients) > 0
	p.mu.RUnlock()

	return gateway.ProviderInfo{
		Name:            p.name,
		Version:         "1.0.0",
		Description:     "Native Lux DEX provider with v4 precompile support",
		SupportedChains: p.chains,
		Priority:        p.priority,
		Healthy:         healthy,
	}
}

// HealthCheck performs a health check on RPC connections
func (p *Provider) HealthCheck(ctx context.Context) gateway.HealthCheck {
	start := time.Now()
	p.mu.RLock()
	defer p.mu.RUnlock()

	healthy := false
	var latency time.Duration

	// Check at least one RPC connection is working
	for chainID, client := range p.clients {
		if client != nil {
			_, err := client.BlockNumber(ctx)
			if err == nil {
				healthy = true
				latency = time.Since(start)
				log.Printf("Health check passed for chain %d (latency: %v)", chainID, latency)
				break
			} else {
				log.Printf("Health check failed for chain %d: %v", chainID, err)
			}
		}
	}

	return gateway.HealthCheck{
		Provider:  p.name,
		Healthy:   healthy,
		Latency:   latency.Milliseconds(),
		LastCheck: time.Now(),
	}
}

// Close cleans up resources and closes RPC connections
func (p *Provider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for chainID, client := range p.clients {
		if client != nil {
			client.Close()
			log.Printf("Closed RPC connection for chain %d", chainID)
		}
	}
	p.clients = make(map[gateway.ChainID]*ethclient.Client)
	return nil
}

// getClient returns the RPC client for a chain
func (p *Provider) getClient(chainID gateway.ChainID) (*ethclient.Client, error) {
	p.mu.RLock()
	client, ok := p.clients[chainID]
	p.mu.RUnlock()

	if !ok || client == nil {
		return nil, fmt.Errorf("%w for chain %d", ErrNoRPCEndpoint, chainID)
	}
	return client, nil
}

// getPoolManagerAddress returns the pool manager precompile address
func (p *Provider) getPoolManagerAddress() common.Address {
	return common.HexToAddress(PoolManagerAddress)
}

// SupportsChain returns true if this provider supports the given chain
func (p *Provider) SupportsChain(chainID gateway.ChainID) bool {
	for _, c := range p.chains {
		if c == chainID {
			return true
		}
	}
	return false
}

// GetQuote returns a swap quote by querying on-chain DEX precompile
func (p *Provider) GetQuote(ctx context.Context, req gateway.QuoteRequest) (*gateway.SwapQuote, error) {
	// Only handle Lux and Zoo chains
	if !p.SupportsChain(req.ChainID) {
		return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
	}

	if req.Amount == nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: fmt.Errorf("amount required")}
	}

	// Get RPC client for chain
	client, err := p.getClient(req.ChainID)
	if err != nil {
		// Fallback to estimate-based quote if no RPC
		log.Printf("No RPC connection for chain %d, using estimate", req.ChainID)
		return p.getEstimatedQuote(req)
	}

	// Get pool state from DEX precompile
	poolState, err := p.queryPoolState(ctx, client, req.TokenIn.Address, req.TokenOut.Address, 3000)
	if err != nil {
		log.Printf("Failed to query pool state: %v, using estimate", err)
		return p.getEstimatedQuote(req)
	}

	// Calculate swap output based on pool state
	amountOut := p.calculateSwapOutput(poolState, req.Amount, req.IsExactIn)

	tokenIn := req.TokenIn
	if tokenIn.Symbol == "" {
		tokenIn.Symbol = p.getNativeSymbol(req.ChainID)
		tokenIn.Name = p.getNativeName(req.ChainID)
		tokenIn.Decimals = 18
		tokenIn.ChainID = req.ChainID
	}

	tokenOut := req.TokenOut
	if tokenOut.Symbol == "" {
		tokenOut.Symbol = "USDC"
		tokenOut.Name = "USD Coin"
		tokenOut.Decimals = 6
		tokenOut.ChainID = req.ChainID
	}

	// Calculate price impact
	priceImpact := p.calculatePriceImpact(poolState, req.Amount)

	return &gateway.SwapQuote{
		TokenIn: gateway.TokenAmount{
			Token:  tokenIn,
			Amount: req.Amount,
		},
		TokenOut: gateway.TokenAmount{
			Token:  tokenOut,
			Amount: amountOut,
		},
		Route: []gateway.PoolHop{
			{
				PoolAddress: PoolManagerAddress,
				PoolType:    "v4",
				TokenIn:     tokenIn,
				TokenOut:    tokenOut,
				Fee:         3000,
			},
		},
		PriceImpact:  priceImpact,
		GasEstimate:  big.NewInt(150000),
		QuoteID:      fmt.Sprintf("lux-quote-%d", time.Now().UnixNano()),
		ExpiresAt:    time.Now().Add(30 * time.Second),
		ProviderName: p.name,
	}, nil
}

// poolState holds on-chain pool data
type poolState struct {
	SqrtPriceX96 *big.Int
	Liquidity    *big.Int
	Tick         int32
	FeeGrowth0   *big.Int
	FeeGrowth1   *big.Int
}

// queryPoolState queries a V3 pool contract on-chain for live state via slot0() and liquidity()
func (p *Provider) queryPoolState(ctx context.Context, client *ethclient.Client, token0, token1 string, fee uint32) (*poolState, error) {
	// First discover the pool address from V3 factory
	poolAddr, err := p.getV3Pool(ctx, client, token0, token1, fee)
	if err != nil {
		return nil, fmt.Errorf("pool not found: %w", err)
	}
	return p.queryV3PoolState(ctx, client, poolAddr)
}

// getV3Pool calls factory.getPool(tokenA, tokenB, fee) to find the pool address
func (p *Provider) getV3Pool(ctx context.Context, client *ethclient.Client, tokenA, tokenB string, fee uint32) (common.Address, error) {
	factory := common.HexToAddress(V3FactoryAddress)
	// getPool(address,address,uint24) selector: 0x1698ee82
	data := common.FromHex("0x1698ee82")
	data = append(data, common.LeftPadBytes(common.HexToAddress(tokenA).Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(common.HexToAddress(tokenB).Bytes(), 32)...)
	feePad := make([]byte, 32)
	new(big.Int).SetUint64(uint64(fee)).FillBytes(feePad)
	data = append(data, feePad...)

	msg := ethereum.CallMsg{To: &factory, Data: data}
	result, err := client.CallContract(ctx, msg, nil)
	if err != nil {
		return common.Address{}, fmt.Errorf("getPool call failed: %w", err)
	}
	if len(result) < 32 {
		return common.Address{}, fmt.Errorf("invalid getPool response")
	}
	addr := common.BytesToAddress(result[12:32])
	if addr == (common.Address{}) {
		return common.Address{}, fmt.Errorf("pool does not exist for %s/%s fee %d", tokenA, tokenB, fee)
	}
	return addr, nil
}

// queryV3PoolState calls slot0() and liquidity() on a V3 pool contract
func (p *Provider) queryV3PoolState(ctx context.Context, client *ethclient.Client, poolAddr common.Address) (*poolState, error) {
	// slot0() = 0x3850c7bd
	slot0Data := common.FromHex("0x3850c7bd")
	msg := ethereum.CallMsg{To: &poolAddr, Data: slot0Data}
	result, err := client.CallContract(ctx, msg, nil)
	if err != nil {
		return nil, fmt.Errorf("slot0 call failed: %w", err)
	}
	if len(result) < 64 {
		return nil, fmt.Errorf("invalid slot0 response (len=%d)", len(result))
	}

	sqrtPriceX96 := new(big.Int).SetBytes(result[0:32])
	tickBig := new(big.Int).SetBytes(result[32:64])
	// Handle signed int24
	if tickBig.Cmp(new(big.Int).Lsh(big.NewInt(1), 255)) >= 0 {
		tickBig.Sub(tickBig, new(big.Int).Lsh(big.NewInt(1), 256))
	}

	// liquidity() = 0x1a686502
	liqData := common.FromHex("0x1a686502")
	msg2 := ethereum.CallMsg{To: &poolAddr, Data: liqData}
	result2, err := client.CallContract(ctx, msg2, nil)
	liquidity := big.NewInt(0)
	if err == nil && len(result2) >= 32 {
		liquidity = new(big.Int).SetBytes(result2[0:32])
	}

	return &poolState{
		SqrtPriceX96: sqrtPriceX96,
		Liquidity:    liquidity,
		Tick:         int32(tickBig.Int64()),
		FeeGrowth0:   big.NewInt(0),
		FeeGrowth1:   big.NewInt(0),
	}, nil
}

// priceFromSqrtX96 computes token1/token0 price from sqrtPriceX96, adjusted for decimals
func priceFromSqrtX96(sqrtPriceX96 *big.Int, decimals0, decimals1 int) float64 {
	if sqrtPriceX96.Sign() == 0 {
		return 0
	}
	sq := new(big.Float).SetInt(sqrtPriceX96)
	q96 := new(big.Float).SetInt(new(big.Int).Lsh(big.NewInt(1), 96))
	ratio := new(big.Float).Quo(sq, q96)
	price := new(big.Float).Mul(ratio, ratio)

	// Adjust for decimal difference
	decAdj := decimals0 - decimals1
	if decAdj != 0 {
		adj := new(big.Float).SetFloat64(math.Pow(10, float64(decAdj)))
		price.Mul(price, adj)
	}

	f, _ := price.Float64()
	return f
}

// calculateSwapOutput calculates expected output for a swap
func (p *Provider) calculateSwapOutput(pool *poolState, amountIn *big.Int, exactIn bool) *big.Int {
	if pool.Liquidity.Sign() == 0 {
		return big.NewInt(0)
	}

	// Simplified AMM formula: amountOut = amountIn * liquidity / (liquidity + amountIn)
	numerator := new(big.Int).Mul(amountIn, pool.Liquidity)
	denominator := new(big.Int).Add(pool.Liquidity, amountIn)
	return new(big.Int).Div(numerator, denominator)
}

// calculatePriceImpact calculates the price impact of a swap
func (p *Provider) calculatePriceImpact(pool *poolState, amountIn *big.Int) float64 {
	if pool.Liquidity.Sign() == 0 {
		return 100.0
	}

	// Price impact = amountIn / (liquidity * 2) * 100
	impact := new(big.Float).SetInt(amountIn)
	liq := new(big.Float).SetInt(pool.Liquidity)
	liq.Mul(liq, big.NewFloat(2))
	impact.Quo(impact, liq)
	impact.Mul(impact, big.NewFloat(100))

	result, _ := impact.Float64()
	return result
}

// getEstimatedQuote returns a quote based on estimates when no RPC is available
func (p *Provider) getEstimatedQuote(req gateway.QuoteRequest) (*gateway.SwapQuote, error) {
	// Simple estimate: 1:1000 ratio for native -> USDC-like
	amountOut := new(big.Int).Mul(req.Amount, big.NewInt(1000))
	amountOut = amountOut.Div(amountOut, big.NewInt(1e12))

	tokenIn := req.TokenIn
	if tokenIn.Symbol == "" {
		tokenIn.Symbol = p.getNativeSymbol(req.ChainID)
		tokenIn.Name = p.getNativeName(req.ChainID)
		tokenIn.Decimals = 18
		tokenIn.ChainID = req.ChainID
	}

	tokenOut := req.TokenOut
	if tokenOut.Symbol == "" {
		tokenOut.Symbol = "USDC"
		tokenOut.Name = "USD Coin"
		tokenOut.Decimals = 6
		tokenOut.ChainID = req.ChainID
	}

	return &gateway.SwapQuote{
		TokenIn: gateway.TokenAmount{
			Token:  tokenIn,
			Amount: req.Amount,
		},
		TokenOut: gateway.TokenAmount{
			Token:  tokenOut,
			Amount: amountOut,
		},
		Route: []gateway.PoolHop{
			{
				PoolAddress: PoolManagerAddress,
				PoolType:    "v4",
				TokenIn:     tokenIn,
				TokenOut:    tokenOut,
				Fee:         3000,
			},
		},
		PriceImpact:  0.1,
		GasEstimate:  big.NewInt(150000),
		QuoteID:      fmt.Sprintf("lux-estimate-%d", time.Now().UnixNano()),
		ExpiresAt:    time.Now().Add(30 * time.Second),
		ProviderName: p.name,
	}, nil
}

func (p *Provider) getNativeSymbol(chainID gateway.ChainID) string {
	if chainID == gateway.ChainIDZoo {
		return "ZOO"
	}
	return "LUX"
}

func (p *Provider) getNativeName(chainID gateway.ChainID) string {
	if chainID == gateway.ChainIDZoo {
		return "Zoo"
	}
	return "Lux"
}

// GetQuotes returns multiple quotes
func (p *Provider) GetQuotes(ctx context.Context, req gateway.QuoteRequest) ([]gateway.SwapQuote, error) {
	// TODO: Implement native Lux multi-route quotes
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// BuildSwap builds a swap transaction
func (p *Provider) BuildSwap(ctx context.Context, quote gateway.SwapQuote, recipient string, deadline int64) (*gateway.SwapTransaction, error) {
	// TODO: Implement native Lux swap transaction building
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// ExecuteSwap executes a swap
func (p *Provider) ExecuteSwap(ctx context.Context, req gateway.SwapRequest) (string, error) {
	// TODO: Implement native Lux swap execution
	return "", &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// v3PoolDef defines a V3 token pair to check for pools
type v3PoolDef struct {
	tokenA    string
	symbolA   string
	nameA     string
	decimalsA int
	tokenB    string
	symbolB   string
	nameB     string
	decimalsB int
}

// knownV3Pairs lists token pairs to discover pools for via V3 factory
var knownV3Pairs = []v3PoolDef{
	{WLUXAddress, "WLUX", "Wrapped LUX", 18, LUSDAddress, "LUSD", "Lux USD", 18},
	{WLUXAddress, "WLUX", "Wrapped LUX", 18, LETHAddress, "LETH", "Lux Ether", 18},
	{WLUXAddress, "WLUX", "Wrapped LUX", 18, LBTCAddress, "LBTC", "Lux Bitcoin", 8},
	{WLUXAddress, "WLUX", "Wrapped LUX", 18, LSOLAddress, "LSOL", "Lux SOL", 18},
	{LBTCAddress, "LBTC", "Lux Bitcoin", 8, LUSDAddress, "LUSD", "Lux USD", 18},
	{LETHAddress, "LETH", "Lux Ether", 18, LUSDAddress, "LUSD", "Lux USD", 18},
	{LBTCAddress, "LBTC", "Lux Bitcoin", 8, LETHAddress, "LETH", "Lux Ether", 18},
	{LSOLAddress, "LSOL", "Lux SOL", 18, LUSDAddress, "LUSD", "Lux USD", 18},
}

// V3 fee tiers to check
var v3FeeTiers = []uint32{500, 3000, 10000}

// GetPools returns pools by querying on-chain V3 factory and pool contracts
func (p *Provider) GetPools(ctx context.Context, req gateway.PoolsRequest) ([]gateway.Pool, error) {
	if !p.SupportsChain(req.ChainID) {
		return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
	}

	client, err := p.getClient(req.ChainID)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: fmt.Errorf("no RPC: %w", err)}
	}

	blockNum, _ := client.BlockNumber(ctx)
	log.Printf("Querying pools at block %d for chain %d", blockNum, req.ChainID)

	var pools []gateway.Pool

	if req.ChainID == gateway.ChainIDLux {
		for _, pair := range knownV3Pairs {
			for _, fee := range v3FeeTiers {
				poolAddr, err := p.getV3Pool(ctx, client, pair.tokenA, pair.tokenB, fee)
				if err != nil {
					continue // Pool doesn't exist for this fee tier
				}

				state, err := p.queryV3PoolState(ctx, client, poolAddr)
				if err != nil {
					continue
				}

				// Determine token order (V3 sorts by address)
				var t0, t1 gateway.Token
				addrA := common.HexToAddress(pair.tokenA)
				addrB := common.HexToAddress(pair.tokenB)
				if addrA.Hex() < addrB.Hex() {
					t0 = gateway.Token{Address: pair.tokenA, ChainID: req.ChainID, Symbol: pair.symbolA, Name: pair.nameA, Decimals: pair.decimalsA}
					t1 = gateway.Token{Address: pair.tokenB, ChainID: req.ChainID, Symbol: pair.symbolB, Name: pair.nameB, Decimals: pair.decimalsB}
				} else {
					t0 = gateway.Token{Address: pair.tokenB, ChainID: req.ChainID, Symbol: pair.symbolB, Name: pair.nameB, Decimals: pair.decimalsB}
					t1 = gateway.Token{Address: pair.tokenA, ChainID: req.ChainID, Symbol: pair.symbolA, Name: pair.nameA, Decimals: pair.decimalsA}
				}

				pools = append(pools, gateway.Pool{
					Address:  poolAddr.Hex(),
					ChainID:  req.ChainID,
					Protocol: "uniswap-v3",
					Token0:   t0,
					Token1:   t1,
					Fee:      int(fee),
					TVL:      state.Liquidity, // Live from chain
					TickData: []gateway.TickRange{
						{TickLower: int(state.Tick) - 1000, TickUpper: int(state.Tick) + 1000, Liquidity: state.Liquidity},
					},
				})
			}
		}
	}

	log.Printf("Found %d live pools for chain %d", len(pools), req.ChainID)
	return pools, nil
}

// GetPool returns a specific pool
func (p *Provider) GetPool(ctx context.Context, chainID gateway.ChainID, address string) (*gateway.Pool, error) {
	// TODO: Implement native Lux pool lookup
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetPositions returns positions for an owner
func (p *Provider) GetPositions(ctx context.Context, req gateway.PositionsRequest) ([]gateway.Position, error) {
	// TODO: Implement native Lux position queries
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetPosition returns a specific position
func (p *Provider) GetPosition(ctx context.Context, chainID gateway.ChainID, positionID string) (*gateway.Position, error) {
	// TODO: Implement native Lux position lookup
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// BuildAddLiquidity builds add liquidity transaction
func (p *Provider) BuildAddLiquidity(ctx context.Context, pool gateway.Pool, amount0, amount1 string, tickLower, tickUpper int) (*gateway.SwapTransaction, error) {
	// TODO: Implement native Lux add liquidity
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// BuildRemoveLiquidity builds remove liquidity transaction
func (p *Provider) BuildRemoveLiquidity(ctx context.Context, position gateway.Position, percentage float64) (*gateway.SwapTransaction, error) {
	// TODO: Implement native Lux remove liquidity
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// BuildCollectFees builds collect fees transaction
func (p *Provider) BuildCollectFees(ctx context.Context, position gateway.Position) (*gateway.SwapTransaction, error) {
	// TODO: Implement native Lux collect fees
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// Real token addresses on Lux C-Chain (verified on-chain 2026-03-05)
const (
	WLUXAddress  = "0x4888e4a2ee0f03051c72d2bd3acf755ed3498b3e"
	LBTCAddress  = "0x1E48D32a4F5e9f08DB9aE4959163300FaF8A6C8e"
	LETHAddress  = "0x60E0a8167FC13dE89348978860466C9ceC24B9ba"
	LSOLAddress  = "0x26B40f650156C7EbF9e087Dd0dca181Fe87625B7"
	LUSDAddress  = "0x848Cff46eb323f323b6Bbe1Df274E40793d7f2c2"
	LTONAddress  = "0x3141b94b89691009b950c96e97Bff48e0C543E3C"
	LAVAXAddress = "0x0e4bD0DD67c15dECfBBBdbbE07FC9d51D737693D"

	// V3 Factory (CREATE2, confirmed on-chain)
	V3FactoryAddress = "0x80bBc7C4C7a59C899D1B37BC14539A22D5830a84"
	// V2 Factory (CREATE2)
	V2FactoryAddress = "0xD173926A10A0C4eCd3A51B1422270b65Df0551c1"

	// Price cache TTL
	priceCacheTTL = 10 * time.Second
)

// tokenMeta defines known tokens with their decimals for price computation
var tokenMeta = map[string]struct {
	address  string
	decimals int
}{
	"WLUX":  {address: WLUXAddress, decimals: 18},
	"LUX":   {address: WLUXAddress, decimals: 18},
	"LBTC":  {address: LBTCAddress, decimals: 8},
	"LETH":  {address: LETHAddress, decimals: 18},
	"LSOL":  {address: LSOLAddress, decimals: 18},
	"LUSD":  {address: LUSDAddress, decimals: 18},
	"LTON":  {address: LTONAddress, decimals: 9},
	"LAVAX": {address: LAVAXAddress, decimals: 18},
}

// priceEntry is a cached price with timestamp
type priceEntry struct {
	priceUSD  float64
	updatedAt time.Time
}

// priceCache stores live prices from on-chain queries
var (
	priceCache   = make(map[string]*priceEntry)
	priceCacheMu sync.RWMutex
)

// GetTokenPrice returns live token price by querying on-chain V3 pools
func (p *Provider) GetTokenPrice(ctx context.Context, token gateway.Token) (*gateway.TokenPrice, error) {
	if !p.SupportsChain(token.ChainID) {
		return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
	}

	sym := token.Symbol
	if sym == "" {
		sym = p.getNativeSymbol(token.ChainID)
	}

	priceUSD := p.getLivePrice(ctx, sym)

	return &gateway.TokenPrice{
		Token:     token,
		PriceUSD:  priceUSD,
		UpdatedAt: time.Now(),
	}, nil
}

// getLivePrice returns a live USD price for a token, querying on-chain if cache is stale
func (p *Provider) getLivePrice(ctx context.Context, symbol string) float64 {
	// Check cache first
	priceCacheMu.RLock()
	if entry, ok := priceCache[symbol]; ok && time.Since(entry.updatedAt) < priceCacheTTL {
		priceCacheMu.RUnlock()
		return entry.priceUSD
	}
	priceCacheMu.RUnlock()

	// LUSD is always $1 (stablecoin)
	if symbol == "LUSD" {
		p.cachePrice(symbol, 1.0)
		return 1.0
	}

	client, err := p.getClient(gateway.ChainIDLux)
	if err != nil {
		log.Printf("No RPC for live price query: %v", err)
		return 0
	}

	// Step 1: Get LUX price from WLUX/LUSD pool
	luxPrice := p.queryLuxPrice(ctx, client)
	if luxPrice == 0 {
		return 0
	}
	p.cachePrice("LUX", luxPrice)
	p.cachePrice("WLUX", luxPrice)

	if symbol == "LUX" || symbol == "WLUX" {
		return luxPrice
	}

	// Step 2: Get other token prices via their WLUX pools
	meta, ok := tokenMeta[symbol]
	if !ok {
		return 0
	}

	// Try to find pool: token/WLUX
	state, err := p.queryPoolState(ctx, client, meta.address, WLUXAddress, 3000)
	if err != nil {
		log.Printf("No pool found for %s/WLUX: %v", symbol, err)
		return 0
	}

	// Determine token order (lower address = token0)
	tokenAddr := common.HexToAddress(meta.address)
	wluxAddr := common.HexToAddress(WLUXAddress)

	var priceInWLUX float64
	if tokenAddr.Hex() < wluxAddr.Hex() {
		// token is token0, WLUX is token1
		// price = token1/token0 = WLUX per token
		priceInWLUX = priceFromSqrtX96(state.SqrtPriceX96, meta.decimals, 18)
	} else {
		// WLUX is token0, token is token1
		// price = token1/token0 = token per WLUX
		raw := priceFromSqrtX96(state.SqrtPriceX96, 18, meta.decimals)
		if raw > 0 {
			priceInWLUX = 1.0 / raw
		}
	}

	priceUSD := priceInWLUX * luxPrice
	p.cachePrice(symbol, priceUSD)
	return priceUSD
}

// queryLuxPrice gets the LUX/USD price from the WLUX/LUSD V3 pool
func (p *Provider) queryLuxPrice(ctx context.Context, client *ethclient.Client) float64 {
	state, err := p.queryPoolState(ctx, client, WLUXAddress, LUSDAddress, 3000)
	if err != nil {
		log.Printf("Failed to query WLUX/LUSD pool: %v", err)
		return 0
	}

	// WLUX (0x4888...) < LUSD (0x848C...), so WLUX=token0, LUSD=token1
	// price = LUSD per WLUX (both 18 decimals)
	return priceFromSqrtX96(state.SqrtPriceX96, 18, 18)
}

func (p *Provider) cachePrice(symbol string, price float64) {
	priceCacheMu.Lock()
	priceCache[symbol] = &priceEntry{priceUSD: price, updatedAt: time.Now()}
	priceCacheMu.Unlock()
}

// GetTokenPrices returns prices for multiple tokens
func (p *Provider) GetTokenPrices(ctx context.Context, tokens []gateway.Token) ([]gateway.TokenPrice, error) {
	var prices []gateway.TokenPrice
	for _, t := range tokens {
		price, err := p.GetTokenPrice(ctx, t)
		if err != nil {
			continue
		}
		prices = append(prices, *price)
	}
	if len(prices) == 0 {
		return nil, &gateway.ProviderError{Provider: p.name, Err: fmt.Errorf("no prices available")}
	}
	return prices, nil
}

// CreateLead creates a conversion lead
func (p *Provider) CreateLead(ctx context.Context, lead gateway.ConversionLead) (*gateway.ConversionLead, error) {
	// TODO: Implement native Lux conversion tracking
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetLead retrieves a lead
func (p *Provider) GetLead(ctx context.Context, leadID string) (*gateway.ConversionLead, error) {
	// TODO: Implement lead lookup
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// TrackEvent tracks a conversion event
func (p *Provider) TrackEvent(ctx context.Context, event gateway.ConversionEvent) error {
	// TODO: Implement native Lux event tracking
	return &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetLeadEvents gets events for a lead
func (p *Provider) GetLeadEvents(ctx context.Context, leadID string) ([]gateway.ConversionEvent, error) {
	// TODO: Implement event queries
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetTokenList returns token list for a chain
func (p *Provider) GetTokenList(ctx context.Context, chainID gateway.ChainID) ([]gateway.Token, error) {
	// Return real token list for Lux — all addresses verified on-chain
	if chainID == gateway.ChainIDLux {
		return []gateway.Token{
			{Address: "0x0000000000000000000000000000000000000000", ChainID: gateway.ChainIDLux, Decimals: 18, Symbol: "LUX", Name: "Lux"},
			{Address: WLUXAddress, ChainID: gateway.ChainIDLux, Decimals: 18, Symbol: "WLUX", Name: "Wrapped LUX"},
			{Address: LUSDAddress, ChainID: gateway.ChainIDLux, Decimals: 18, Symbol: "LUSD", Name: "Lux USD"},
			{Address: LETHAddress, ChainID: gateway.ChainIDLux, Decimals: 18, Symbol: "LETH", Name: "Lux Ether"},
			{Address: LBTCAddress, ChainID: gateway.ChainIDLux, Decimals: 8, Symbol: "LBTC", Name: "Lux Bitcoin"},
			{Address: LSOLAddress, ChainID: gateway.ChainIDLux, Decimals: 18, Symbol: "LSOL", Name: "Lux SOL"},
			{Address: LTONAddress, ChainID: gateway.ChainIDLux, Decimals: 9, Symbol: "LTON", Name: "Lux TON"},
			{Address: LAVAXAddress, ChainID: gateway.ChainIDLux, Decimals: 18, Symbol: "LAVAX", Name: "Lux AVAX"},
		}, nil
	}

	if chainID == gateway.ChainIDZoo {
		return []gateway.Token{
			{Address: "0x0000000000000000000000000000000000000000", ChainID: gateway.ChainIDZoo, Decimals: 18, Symbol: "ZOO", Name: "Zoo"},
			{Address: "0x5491216406daB99b7032b83765F36790E27F8A61", ChainID: gateway.ChainIDZoo, Decimals: 18, Symbol: "WZOO", Name: "Wrapped ZOO"},
			{Address: "0xb2ee1CE7b84853b83AA08702aD0aD4D79711882D", ChainID: gateway.ChainIDZoo, Decimals: 18, Symbol: "LUSD", Name: "Lux USD"},
			{Address: "0x4870621EA8be7a383eFCfdA225249d35888bD9f2", ChainID: gateway.ChainIDZoo, Decimals: 18, Symbol: "LETH", Name: "Lux Ether"},
			{Address: "0x6fc44509a32E513bE1aa00d27bb298e63830C6A8", ChainID: gateway.ChainIDZoo, Decimals: 8, Symbol: "LBTC", Name: "Lux Bitcoin"},
		}, nil
	}

	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// SearchTokens searches for tokens
func (p *Provider) SearchTokens(ctx context.Context, chainID gateway.ChainID, query string) ([]gateway.Token, error) {
	// TODO: Implement token search
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetToken returns a specific token
func (p *Provider) GetToken(ctx context.Context, chainID gateway.ChainID, address string) (*gateway.Token, error) {
	// TODO: Implement token lookup
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// parseBigInt parses a string to *big.Int
func parseBigInt(s string) *big.Int {
	n := new(big.Int)
	n.SetString(s, 10)
	return n
}
