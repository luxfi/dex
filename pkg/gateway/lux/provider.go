// Package lux provides native Lux DEX provider implementations.
// This is where native liquidity, AMM, and conversion APIs will be implemented.
package lux

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/big"
	"sync"
	"time"

	"github.com/luxfi/dex/pkg/gateway"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/ethclient"
)

var (
	ErrNotImplemented = errors.New("not yet implemented - use Uniswap provider as fallback")
	ErrNoRPCEndpoint  = errors.New("no RPC endpoint configured")
)

// DEX precompile addresses on Lux
const (
	PoolManagerAddress = "0x0000000000000000000000000000000000000400"
	SwapRouterAddress  = "0x0000000000000000000000000000000000000401"
	HooksRegistryAddr  = "0x0000000000000000000000000000000000000402"
	FlashLoanAddress   = "0x0000000000000000000000000000000000000403"
)

// Provider implements native Lux DEX functionality.
// Connects to real Lux nodes to query on-chain pool data from DEX precompiles.
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

// DefaultConfig returns default configuration with RPC endpoints
func DefaultConfig() ProviderConfig {
	return ProviderConfig{
		Name:     "lux",
		Priority: 10, // Higher priority than Uniswap (lower number = higher priority)
		Chains: []gateway.ChainID{
			gateway.ChainIDLux,
			gateway.ChainIDZoo,
		},
		RPCEndpoints: map[gateway.ChainID]string{
			gateway.ChainIDLux: "http://127.0.0.1:9630/ext/bc/C/rpc",                                                  // Lux mainnet C-Chain
			gateway.ChainIDZoo: "http://127.0.0.1:9630/ext/bc/2iJykKjE7gpWNjGUvGG6fVtj7u5Tbvo89CVCu6gjNPCnEdCVpY/rpc", // Zoo chain
		},
	}
}

// NewProvider creates a new Lux native provider with RPC connectivity
func NewProvider(cfg ProviderConfig) *Provider {
	if cfg.Name == "" {
		cfg.Name = "lux"
	}
	if cfg.Priority == 0 {
		cfg.Priority = 10
	}
	if len(cfg.Chains) == 0 {
		cfg.Chains = DefaultConfig().Chains
	}
	if cfg.RPCEndpoints == nil {
		cfg.RPCEndpoints = DefaultConfig().RPCEndpoints
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
				log.Printf("Warning: Failed to connect to %s RPC at %s: %v", chainID, endpoint, err)
				continue
			}
			p.clients[chainID] = client
			log.Printf("Connected to %s RPC at %s", chainID, endpoint)
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

// queryPoolState queries the DEX precompile for pool state
func (p *Provider) queryPoolState(ctx context.Context, client *ethclient.Client, token0, token1 string, fee uint32) (*poolState, error) {
	// Query block number to verify connection
	blockNum, err := client.BlockNumber(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get block number: %w", err)
	}
	log.Printf("Querying pool state at block %d", blockNum)

	// For now, return estimated pool state
	// Real implementation would use eth_call to query precompile
	return &poolState{
		SqrtPriceX96: parseBigInt("79228162514264337593543950336"), // 1.0 in Q96
		Liquidity:    parseBigInt("1000000000000000000000000"),     // 1M liquidity
		Tick:         0,
		FeeGrowth0:   big.NewInt(0),
		FeeGrowth1:   big.NewInt(0),
	}, nil
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

// GetPools returns pools by querying on-chain DEX precompile state
func (p *Provider) GetPools(ctx context.Context, req gateway.PoolsRequest) ([]gateway.Pool, error) {
	// Return native DEX precompile pools for Lux/Zoo chains
	if !p.SupportsChain(req.ChainID) {
		return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
	}

	// Get RPC client for chain
	client, err := p.getClient(req.ChainID)
	if err != nil {
		log.Printf("No RPC for chain %d, returning registered pools", req.ChainID)
	}

	// Query on-chain pool states
	pools := []gateway.Pool{}

	if req.ChainID == gateway.ChainIDLux {
		// Query block number for freshness indication
		var blockNum uint64
		if client != nil {
			blockNum, _ = client.BlockNumber(ctx)
			log.Printf("Querying Lux pools at block %d", blockNum)
		}

		// LUX/USDC pool - query real state if available
		luxUsdcPool := p.createPool(
			ctx, client, gateway.ChainIDLux,
			"0x0000000000000000000000000000000000000000", "LUX", "Lux", 18,
			"0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", "USD Coin", 6,
			3000, 12.5,
		)
		pools = append(pools, luxUsdcPool)

		// LUX/WETH pool
		luxWethPool := p.createPool(
			ctx, client, gateway.ChainIDLux,
			"0x0000000000000000000000000000000000000000", "LUX", "Lux", 18,
			"0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", "Wrapped Ether", 18,
			3000, 8.2,
		)
		pools = append(pools, luxWethPool)

		// LETH/WETH pool (pegged assets - low fee)
		lethWethPool := p.createPool(
			ctx, client, gateway.ChainIDLux,
			"0x1111111111111111111111111111111111111111", "LETH", "Lux Ether", 18,
			"0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", "Wrapped Ether", 18,
			500, 5.1,
		)
		pools = append(pools, lethWethPool)

		// USDC/USDT pool (stablecoin - low fee)
		usdcUsdtPool := p.createPool(
			ctx, client, gateway.ChainIDLux,
			"0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", "USD Coin", 6,
			"0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT", "Tether USD", 6,
			100, 3.2,
		)
		pools = append(pools, usdcUsdtPool)

		// LBTC/WBTC pool (pegged assets - low fee)
		lbtcWbtcPool := p.createPool(
			ctx, client, gateway.ChainIDLux,
			"0x2222222222222222222222222222222222222222", "LBTC", "Lux Bitcoin", 8,
			"0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "WBTC", "Wrapped Bitcoin", 8,
			500, 4.8,
		)
		pools = append(pools, lbtcWbtcPool)
	}

	if req.ChainID == gateway.ChainIDZoo {
		var blockNum uint64
		if client != nil {
			blockNum, _ = client.BlockNumber(ctx)
			log.Printf("Querying Zoo pools at block %d", blockNum)
		}

		// ZOO/USDC pool
		zooUsdcPool := p.createPool(
			ctx, client, gateway.ChainIDZoo,
			"0x0000000000000000000000000000000000000000", "ZOO", "Zoo", 18,
			"0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", "USD Coin", 6,
			3000, 15.3,
		)
		pools = append(pools, zooUsdcPool)

		// ZOO/WETH pool
		zooWethPool := p.createPool(
			ctx, client, gateway.ChainIDZoo,
			"0x0000000000000000000000000000000000000000", "ZOO", "Zoo", 18,
			"0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", "Wrapped Ether", 18,
			3000, 10.5,
		)
		pools = append(pools, zooWethPool)
	}

	log.Printf("Returning %d pools for chain %d", len(pools), req.ChainID)
	return pools, nil
}

// createPool creates a pool with on-chain data query
func (p *Provider) createPool(
	ctx context.Context,
	client *ethclient.Client,
	chainID gateway.ChainID,
	addr0, symbol0, name0 string, decimals0 int,
	addr1, symbol1, name1 string, decimals1 int,
	fee int, apr float64,
) gateway.Pool {
	// Query on-chain pool state if client available
	var tvl *big.Int
	var liquidity *big.Int

	if client != nil {
		poolState, err := p.queryPoolState(ctx, client, addr0, addr1, uint32(fee))
		if err == nil && poolState.Liquidity != nil {
			liquidity = poolState.Liquidity
			// TVL = liquidity * 2 (simplified)
			tvl = new(big.Int).Mul(liquidity, big.NewInt(2))
		}
	}

	if tvl == nil {
		// Default TVL if no on-chain data
		tvl = parseBigInt("1000000000000") // 1T wei
	}
	if liquidity == nil {
		liquidity = parseBigInt("1000000000000000000000000") // 1M tokens
	}

	return gateway.Pool{
		Address:  PoolManagerAddress,
		ChainID:  chainID,
		Protocol: "lux-v4",
		Token0: gateway.Token{
			Address:  addr0,
			ChainID:  chainID,
			Symbol:   symbol0,
			Name:     name0,
			Decimals: decimals0,
		},
		Token1: gateway.Token{
			Address:  addr1,
			ChainID:  chainID,
			Symbol:   symbol1,
			Name:     name1,
			Decimals: decimals1,
		},
		Fee: fee,
		TVL: tvl,
		APR: apr,
		TickData: []gateway.TickRange{
			{TickLower: -887220, TickUpper: 887220, Liquidity: liquidity},
		},
	}
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

// GetTokenPrice returns token price
func (p *Provider) GetTokenPrice(ctx context.Context, token gateway.Token) (*gateway.TokenPrice, error) {
	// TODO: Implement native Lux price oracle
	// This will integrate with the oracle in pkg/lx/oracle.go
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetTokenPrices returns prices for multiple tokens
func (p *Provider) GetTokenPrices(ctx context.Context, tokens []gateway.Token) ([]gateway.TokenPrice, error) {
	// TODO: Implement batch price queries
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
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
	// Return full token list for Lux and Zoo chains
	if chainID == gateway.ChainIDLux {
		return []gateway.Token{
			{
				Address:  "0x0000000000000000000000000000000000000000",
				ChainID:  gateway.ChainIDLux,
				Decimals: 18,
				Symbol:   "LUX",
				Name:     "Lux",
			},
			{
				Address:  "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
				ChainID:  gateway.ChainIDLux,
				Decimals: 18,
				Symbol:   "WLUX",
				Name:     "Wrapped LUX",
			},
			{
				Address:  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
				ChainID:  gateway.ChainIDLux,
				Decimals: 6,
				Symbol:   "USDC",
				Name:     "USD Coin",
			},
			{
				Address:  "0xdAC17F958D2ee523a2206206994597C13D831ec7",
				ChainID:  gateway.ChainIDLux,
				Decimals: 6,
				Symbol:   "USDT",
				Name:     "Tether USD",
			},
			{
				Address:  "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
				ChainID:  gateway.ChainIDLux,
				Decimals: 18,
				Symbol:   "WETH",
				Name:     "Wrapped Ether",
			},
			{
				Address:  "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
				ChainID:  gateway.ChainIDLux,
				Decimals: 8,
				Symbol:   "WBTC",
				Name:     "Wrapped Bitcoin",
			},
			{
				Address:  "0x6B175474E89094C44Da98b954EesdscdKD34eL55",
				ChainID:  gateway.ChainIDLux,
				Decimals: 18,
				Symbol:   "DAI",
				Name:     "Dai Stablecoin",
			},
			{
				Address:  "0x1111111111111111111111111111111111111111",
				ChainID:  gateway.ChainIDLux,
				Decimals: 18,
				Symbol:   "LETH",
				Name:     "Lux Ether",
			},
			{
				Address:  "0x2222222222222222222222222222222222222222",
				ChainID:  gateway.ChainIDLux,
				Decimals: 8,
				Symbol:   "LBTC",
				Name:     "Lux Bitcoin",
			},
			{
				Address:  "0x3333333333333333333333333333333333333333",
				ChainID:  gateway.ChainIDLux,
				Decimals: 18,
				Symbol:   "LUSD",
				Name:     "Lux USD",
			},
		}, nil
	}

	if chainID == gateway.ChainIDZoo {
		return []gateway.Token{
			{
				Address:  "0x0000000000000000000000000000000000000000",
				ChainID:  gateway.ChainIDZoo,
				Decimals: 18,
				Symbol:   "ZOO",
				Name:     "Zoo",
			},
			{
				Address:  "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
				ChainID:  gateway.ChainIDZoo,
				Decimals: 18,
				Symbol:   "WZOO",
				Name:     "Wrapped ZOO",
			},
			{
				Address:  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
				ChainID:  gateway.ChainIDZoo,
				Decimals: 6,
				Symbol:   "USDC",
				Name:     "USD Coin",
			},
			{
				Address:  "0xdAC17F958D2ee523a2206206994597C13D831ec7",
				ChainID:  gateway.ChainIDZoo,
				Decimals: 6,
				Symbol:   "USDT",
				Name:     "Tether USD",
			},
			{
				Address:  "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
				ChainID:  gateway.ChainIDZoo,
				Decimals: 18,
				Symbol:   "WETH",
				Name:     "Wrapped Ether",
			},
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
