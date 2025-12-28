package uniswap

import (
	"context"
	"math/big"
	"strconv"
	"time"

	"github.com/luxfi/dex/pkg/gateway"
)

// Provider implements the gateway provider interfaces using Uniswap APIs
type Provider struct {
	client   *Client
	name     string
	priority int
	chains   []gateway.ChainID
}

// ProviderConfig holds provider configuration
type ProviderConfig struct {
	Name      string
	Priority  int
	APIKey    string
	Timeout   time.Duration
	Endpoints APIEndpoints
	Chains    []gateway.ChainID
}

// DefaultConfig returns the default provider configuration
func DefaultConfig() ProviderConfig {
	return ProviderConfig{
		Name:     "uniswap",
		Priority: 100, // Default priority (lower = higher priority)
		Timeout:  30 * time.Second,
		Chains: []gateway.ChainID{
			gateway.ChainIDEthereum,
			gateway.ChainIDArbitrum,
			gateway.ChainIDOptimism,
			gateway.ChainIDPolygon,
			gateway.ChainIDBase,
			gateway.ChainIDBNB,
		},
	}
}

// NewProvider creates a new Uniswap provider
func NewProvider(cfg ProviderConfig) *Provider {
	if cfg.Name == "" {
		cfg.Name = "uniswap"
	}
	if cfg.Priority == 0 {
		cfg.Priority = 100
	}
	if len(cfg.Chains) == 0 {
		cfg.Chains = DefaultConfig().Chains
	}
	
	client := NewClient(ClientConfig{
		Endpoints: cfg.Endpoints,
		APIKey:    cfg.APIKey,
		Timeout:   cfg.Timeout,
	})
	
	return &Provider{
		client:   client,
		name:     cfg.Name,
		priority: cfg.Priority,
		chains:   cfg.Chains,
	}
}

// Info returns provider information
func (p *Provider) Info() gateway.ProviderInfo {
	return gateway.ProviderInfo{
		Name:            p.name,
		Version:         "1.0.0",
		Description:     "Uniswap API provider for quotes, liquidity, and conversion tracking",
		SupportedChains: p.chains,
		Priority:        p.priority,
		Healthy:         true, // Will be updated by health checks
	}
}

// HealthCheck performs a health check
func (p *Provider) HealthCheck(ctx context.Context) gateway.HealthCheck {
	start := time.Now()
	err := p.client.HealthCheck(ctx)
	latency := time.Since(start).Milliseconds()
	
	result := gateway.HealthCheck{
		Provider:  p.name,
		Healthy:   err == nil,
		Latency:   latency,
		LastCheck: time.Now(),
	}
	
	if err != nil {
		result.Error = err.Error()
	}
	
	return result
}

// Close cleans up resources
func (p *Provider) Close() error {
	return nil
}

// GetQuote returns a swap quote
func (p *Provider) GetQuote(ctx context.Context, req gateway.QuoteRequest) (*gateway.SwapQuote, error) {
	quoteType := "EXACT_INPUT"
	if !req.IsExactIn {
		quoteType = "EXACT_OUTPUT"
	}
	
	apiReq := QuoteAPIRequest{
		TokenIn:         req.TokenIn.Address,
		TokenInChainID:  uint64(req.TokenIn.ChainID),
		TokenOut:        req.TokenOut.Address,
		TokenOutChainID: uint64(req.TokenOut.ChainID),
		Amount:          req.Amount.String(),
		Type:            quoteType,
	}
	
	if req.Slippage > 0 {
		apiReq.Slippage = int(req.Slippage * 10000) // Convert to bps
	}
	
	apiResp, err := p.client.GetQuote(ctx, apiReq)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	return p.convertQuoteResponse(req, apiResp), nil
}

// GetQuotes returns multiple quotes
func (p *Provider) GetQuotes(ctx context.Context, req gateway.QuoteRequest) ([]gateway.SwapQuote, error) {
	quote, err := p.GetQuote(ctx, req)
	if err != nil {
		return nil, err
	}
	return []gateway.SwapQuote{*quote}, nil
}

// BuildSwap builds a swap transaction
func (p *Provider) BuildSwap(ctx context.Context, quote gateway.SwapQuote, recipient string, deadline int64) (*gateway.SwapTransaction, error) {
	quoteType := "EXACT_INPUT"
	
	apiReq := QuoteAPIRequest{
		TokenIn:         quote.TokenIn.Token.Address,
		TokenInChainID:  uint64(quote.TokenIn.Token.ChainID),
		TokenOut:        quote.TokenOut.Token.Address,
		TokenOutChainID: uint64(quote.TokenOut.Token.ChainID),
		Amount:          quote.TokenIn.Amount.String(),
		Type:            quoteType,
		Recipient:       recipient,
	}
	
	apiResp, err := p.client.GetQuote(ctx, apiReq)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	if apiResp.MethodParameters == nil {
		return nil, &gateway.ProviderError{
			Provider: p.name,
			Err:      ErrNoMethodParameters,
		}
	}
	
	value := new(big.Int)
	if apiResp.MethodParameters.Value != "" {
		value.SetString(apiResp.MethodParameters.Value, 10)
	}
	
	gasLimit := new(big.Int)
	if apiResp.GasUseEstimate != "" {
		gasLimit.SetString(apiResp.GasUseEstimate, 10)
		// Add 20% buffer
		gasLimit.Mul(gasLimit, big.NewInt(120))
		gasLimit.Div(gasLimit, big.NewInt(100))
	}
	
	return &gateway.SwapTransaction{
		To:       apiResp.MethodParameters.To,
		Data:     apiResp.MethodParameters.Calldata,
		Value:    value,
		GasLimit: gasLimit,
		ChainID:  quote.TokenIn.Token.ChainID,
	}, nil
}

// ExecuteSwap executes a swap (not supported - returns tx data only)
func (p *Provider) ExecuteSwap(ctx context.Context, req gateway.SwapRequest) (string, error) {
	return "", &gateway.ProviderError{
		Provider: p.name,
		Err:      ErrExecuteNotSupported,
	}
}

// GetPools returns pools matching criteria
func (p *Provider) GetPools(ctx context.Context, req gateway.PoolsRequest) ([]gateway.Pool, error) {
	apiReq := PoolsAPIRequest{
		ChainID: uint64(req.ChainID),
		TokenA:  req.Token0,
		TokenB:  req.Token1,
		Limit:   req.Limit,
		Offset:  req.Offset,
	}
	
	if req.MinTVL != nil {
		tvl, _ := req.MinTVL.Float64()
		apiReq.MinTVL = tvl
	}
	
	apiPools, err := p.client.GetPools(ctx, apiReq)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	pools := make([]gateway.Pool, len(apiPools))
	for i, ap := range apiPools {
		pools[i] = p.convertPoolData(ap)
	}
	
	return pools, nil
}

// GetPool returns a specific pool
func (p *Provider) GetPool(ctx context.Context, chainID gateway.ChainID, address string) (*gateway.Pool, error) {
	pools, err := p.GetPools(ctx, gateway.PoolsRequest{
		ChainID: chainID,
		Token0:  address,
		Limit:   1,
	})
	if err != nil {
		return nil, err
	}
	
	if len(pools) == 0 {
		return nil, &gateway.ProviderError{Provider: p.name, Err: ErrPoolNotFound}
	}
	
	return &pools[0], nil
}

// GetPositions returns positions for an owner
func (p *Provider) GetPositions(ctx context.Context, req gateway.PositionsRequest) ([]gateway.Position, error) {
	apiReq := PositionsAPIRequest{
		ChainID: uint64(req.ChainID),
		Owner:   req.Owner,
		PoolID:  req.PoolID,
	}
	
	apiPositions, err := p.client.GetPositions(ctx, apiReq)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	positions := make([]gateway.Position, len(apiPositions))
	for i, ap := range apiPositions {
		positions[i] = p.convertPositionData(ap)
	}
	
	return positions, nil
}

// GetPosition returns a specific position
func (p *Provider) GetPosition(ctx context.Context, chainID gateway.ChainID, positionID string) (*gateway.Position, error) {
	// Query all positions and find the matching one
	positions, err := p.GetPositions(ctx, gateway.PositionsRequest{
		ChainID: chainID,
	})
	if err != nil {
		return nil, err
	}
	
	for _, pos := range positions {
		if pos.ID == positionID {
			return &pos, nil
		}
	}
	
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrPositionNotFound}
}

// BuildAddLiquidity builds add liquidity transaction (not implemented)
func (p *Provider) BuildAddLiquidity(ctx context.Context, pool gateway.Pool, amount0, amount1 string, tickLower, tickUpper int) (*gateway.SwapTransaction, error) {
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// BuildRemoveLiquidity builds remove liquidity transaction (not implemented)
func (p *Provider) BuildRemoveLiquidity(ctx context.Context, position gateway.Position, percentage float64) (*gateway.SwapTransaction, error) {
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// BuildCollectFees builds collect fees transaction (not implemented)
func (p *Provider) BuildCollectFees(ctx context.Context, position gateway.Position) (*gateway.SwapTransaction, error) {
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// GetTokenPrice returns token price
func (p *Provider) GetTokenPrice(ctx context.Context, token gateway.Token) (*gateway.TokenPrice, error) {
	apiPrice, err := p.client.GetTokenPrice(ctx, uint64(token.ChainID), token.Address)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	return &gateway.TokenPrice{
		Token:       token,
		PriceUSD:    apiPrice.PriceUSD,
		PriceChange: apiPrice.PriceChange24h,
		Volume24h:   parseBigInt(apiPrice.Volume24h),
		MarketCap:   parseBigInt(apiPrice.MarketCap),
		UpdatedAt:   time.Now(),
	}, nil
}

// GetTokenPrices returns prices for multiple tokens
func (p *Provider) GetTokenPrices(ctx context.Context, tokens []gateway.Token) ([]gateway.TokenPrice, error) {
	prices := make([]gateway.TokenPrice, 0, len(tokens))
	for _, token := range tokens {
		price, err := p.GetTokenPrice(ctx, token)
		if err != nil {
			continue // Skip failed price lookups
		}
		prices = append(prices, *price)
	}
	return prices, nil
}

// CreateLead creates a conversion lead
func (p *Provider) CreateLead(ctx context.Context, lead gateway.ConversionLead) (*gateway.ConversionLead, error) {
	apiLead := LeadAPIData{
		Source:     lead.Source,
		Medium:     lead.Medium,
		Campaign:   lead.Campaign,
		WalletAddr: lead.WalletAddr,
		Metadata:   lead.Metadata,
	}
	
	created, err := p.client.CreateLead(ctx, apiLead)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	return p.convertLeadData(*created), nil
}

// GetLead retrieves a lead
func (p *Provider) GetLead(ctx context.Context, leadID string) (*gateway.ConversionLead, error) {
	// Not directly supported - would need lead storage
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrNotImplemented}
}

// TrackEvent tracks a conversion event
func (p *Provider) TrackEvent(ctx context.Context, event gateway.ConversionEvent) error {
	apiEvent := EventAPIData{
		LeadID:    event.LeadID,
		EventType: event.EventType,
		ChainID:   uint64(event.ChainID),
		TxHash:    event.TxHash,
		Timestamp: event.Timestamp.Format(time.RFC3339),
		Metadata:  event.Metadata,
	}
	
	if event.Value != nil {
		apiEvent.Value = event.Value.String()
	}
	
	if err := p.client.TrackEvent(ctx, apiEvent); err != nil {
		return &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	return nil
}

// GetLeadEvents gets events for a lead
func (p *Provider) GetLeadEvents(ctx context.Context, leadID string) ([]gateway.ConversionEvent, error) {
	apiEvents, err := p.client.GetLeadEvents(ctx, leadID)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	events := make([]gateway.ConversionEvent, len(apiEvents))
	for i, ae := range apiEvents {
		events[i] = p.convertEventData(ae)
	}
	
	return events, nil
}

// GetTokenList returns token list for a chain
func (p *Provider) GetTokenList(ctx context.Context, chainID gateway.ChainID) ([]gateway.Token, error) {
	apiTokens, err := p.client.GetTokenList(ctx, uint64(chainID))
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	tokens := make([]gateway.Token, len(apiTokens))
	for i, at := range apiTokens {
		tokens[i] = p.convertTokenData(at)
	}
	
	return tokens, nil
}

// SearchTokens searches for tokens
func (p *Provider) SearchTokens(ctx context.Context, chainID gateway.ChainID, query string) ([]gateway.Token, error) {
	apiTokens, err := p.client.SearchTokens(ctx, uint64(chainID), query)
	if err != nil {
		return nil, &gateway.ProviderError{Provider: p.name, Err: err}
	}
	
	tokens := make([]gateway.Token, len(apiTokens))
	for i, at := range apiTokens {
		tokens[i] = p.convertTokenData(at)
	}
	
	return tokens, nil
}

// GetToken returns a specific token
func (p *Provider) GetToken(ctx context.Context, chainID gateway.ChainID, address string) (*gateway.Token, error) {
	tokens, err := p.SearchTokens(ctx, chainID, address)
	if err != nil {
		return nil, err
	}
	
	for _, token := range tokens {
		if token.Address == address {
			return &token, nil
		}
	}
	
	return nil, &gateway.ProviderError{Provider: p.name, Err: ErrTokenNotFound}
}

// Helper conversion functions

func (p *Provider) convertQuoteResponse(req gateway.QuoteRequest, resp *QuoteAPIResponse) *gateway.SwapQuote {
	amountOut, _ := new(big.Int).SetString(resp.Quote, 10)
	gasEstimate, _ := new(big.Int).SetString(resp.GasUseEstimate, 10)
	
	route := make([]gateway.PoolHop, len(resp.Route))
	for i, r := range resp.Route {
		fee := 0
		if r.Fee != "" {
			fee, _ = strconv.Atoi(r.Fee)
		}
		route[i] = gateway.PoolHop{
			PoolAddress: r.Address,
			PoolType:    r.Type,
			TokenIn:     p.convertTokenData(r.TokenIn),
			TokenOut:    p.convertTokenData(r.TokenOut),
			Fee:         fee,
		}
	}
	
	priceImpact := 0.0
	if resp.PriceImpact != "" {
		priceImpact, _ = strconv.ParseFloat(resp.PriceImpact, 64)
	}
	
	return &gateway.SwapQuote{
		TokenIn: gateway.TokenAmount{
			Token:  req.TokenIn,
			Amount: req.Amount,
		},
		TokenOut: gateway.TokenAmount{
			Token:  req.TokenOut,
			Amount: amountOut,
		},
		Route:        route,
		PriceImpact:  priceImpact,
		GasEstimate:  gasEstimate,
		ProviderName: p.name,
	}
}

func (p *Provider) convertPoolData(ap PoolAPIData) gateway.Pool {
	tvl, _ := new(big.Int).SetString(ap.TVL, 10)
	volume, _ := new(big.Int).SetString(ap.Volume24h, 10)
	
	return gateway.Pool{
		Address:   ap.Address,
		ChainID:   gateway.ChainID(ap.ChainID),
		Protocol:  ap.Protocol,
		Token0:    p.convertTokenData(ap.Token0),
		Token1:    p.convertTokenData(ap.Token1),
		Fee:       ap.FeeTier,
		TVL:       tvl,
		Volume24h: volume,
		APR:       ap.APR,
	}
}

func (p *Provider) convertPositionData(ap PositionAPIData) gateway.Position {
	liquidity, _ := new(big.Int).SetString(ap.Liquidity, 10)
	token0Owed, _ := new(big.Int).SetString(ap.Token0Owed, 10)
	token1Owed, _ := new(big.Int).SetString(ap.Token1Owed, 10)
	
	var feesEarned *gateway.FeesEarned
	if ap.Fees0 != "" || ap.Fees1 != "" {
		fees0, _ := new(big.Int).SetString(ap.Fees0, 10)
		fees1, _ := new(big.Int).SetString(ap.Fees1, 10)
		feesEarned = &gateway.FeesEarned{
			Token0Fees: fees0,
			Token1Fees: fees1,
		}
	}
	
	return gateway.Position{
		ID:         ap.ID,
		Owner:      ap.Owner,
		Pool:       p.convertPoolData(ap.Pool),
		Liquidity:  liquidity,
		Token0Owed: token0Owed,
		Token1Owed: token1Owed,
		TickLower:  ap.TickLower,
		TickUpper:  ap.TickUpper,
		FeesEarned: feesEarned,
	}
}

func (p *Provider) convertTokenData(at TokenAPIData) gateway.Token {
	return gateway.Token{
		Address:  at.Address,
		ChainID:  gateway.ChainID(at.ChainID),
		Decimals: at.Decimals,
		Symbol:   at.Symbol,
		Name:     at.Name,
	}
}

func (p *Provider) convertLeadData(al LeadAPIData) *gateway.ConversionLead {
	ts, _ := time.Parse(time.RFC3339, al.CreatedAt)
	return &gateway.ConversionLead{
		ID:         al.ID,
		Source:     al.Source,
		Medium:     al.Medium,
		Campaign:   al.Campaign,
		WalletAddr: al.WalletAddr,
		Timestamp:  ts,
		Metadata:   al.Metadata,
	}
}

func (p *Provider) convertEventData(ae EventAPIData) gateway.ConversionEvent {
	ts, _ := time.Parse(time.RFC3339, ae.Timestamp)
	value, _ := new(big.Int).SetString(ae.Value, 10)
	
	return gateway.ConversionEvent{
		LeadID:    ae.LeadID,
		EventType: ae.EventType,
		ChainID:   gateway.ChainID(ae.ChainID),
		TxHash:    ae.TxHash,
		Value:     value,
		Timestamp: ts,
		Metadata:  ae.Metadata,
	}
}

func parseBigInt(s string) *big.Int {
	if s == "" {
		return nil
	}
	n, _ := new(big.Int).SetString(s, 10)
	return n
}
