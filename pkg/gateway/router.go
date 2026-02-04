package gateway

import (
	"context"
	"sort"
	"sync"
)

// Router aggregates multiple providers and routes requests appropriately
type Router struct {
	registry       *Registry
	enableFallback bool
}

// NewRouter creates a new router
func NewRouter(registry *Registry, enableFallback bool) *Router {
	return &Router{
		registry:       registry,
		enableFallback: enableFallback,
	}
}

// GetBestQuote gets the best quote from all providers
func (r *Router) GetBestQuote(ctx context.Context, req QuoteRequest) (*SwapQuote, error) {
	providers := r.registry.GetQuoteProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	// Get quotes from all providers in parallel
	type result struct {
		quote *SwapQuote
		err   error
	}

	results := make(chan result, len(providers))
	var wg sync.WaitGroup

	for _, provider := range providers {
		wg.Add(1)
		go func(p QuoteProvider) {
			defer wg.Done()
			quote, err := p.GetQuote(ctx, req)
			results <- result{quote: quote, err: err}
		}(provider)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect quotes
	var quotes []SwapQuote
	var lastErr error

	for res := range results {
		if res.err != nil {
			lastErr = res.err
			continue
		}
		quotes = append(quotes, *res.quote)
	}

	if len(quotes) == 0 {
		if lastErr != nil {
			return nil, lastErr
		}
		return nil, ErrAllProvidersFailed
	}

	// Sort by output amount (descending) for EXACT_INPUT
	// or input amount (ascending) for EXACT_OUTPUT
	sort.Slice(quotes, func(i, j int) bool {
		if req.IsExactIn {
			return quotes[i].TokenOut.Amount.Cmp(quotes[j].TokenOut.Amount) > 0
		}
		return quotes[i].TokenIn.Amount.Cmp(quotes[j].TokenIn.Amount) < 0
	})

	return &quotes[0], nil
}

// GetAllQuotes gets quotes from all providers
func (r *Router) GetAllQuotes(ctx context.Context, req QuoteRequest) ([]SwapQuote, error) {
	providers := r.registry.GetQuoteProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	type result struct {
		quotes []SwapQuote
		err    error
	}

	results := make(chan result, len(providers))
	var wg sync.WaitGroup

	for _, provider := range providers {
		wg.Add(1)
		go func(p QuoteProvider) {
			defer wg.Done()
			quotes, err := p.GetQuotes(ctx, req)
			results <- result{quotes: quotes, err: err}
		}(provider)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var allQuotes []SwapQuote
	for res := range results {
		if res.err != nil {
			continue
		}
		allQuotes = append(allQuotes, res.quotes...)
	}

	if len(allQuotes) == 0 {
		return nil, ErrAllProvidersFailed
	}

	// Sort by output amount
	sort.Slice(allQuotes, func(i, j int) bool {
		if req.IsExactIn {
			return allQuotes[i].TokenOut.Amount.Cmp(allQuotes[j].TokenOut.Amount) > 0
		}
		return allQuotes[i].TokenIn.Amount.Cmp(allQuotes[j].TokenIn.Amount) < 0
	})

	return allQuotes, nil
}

// GetPools gets pools from providers with fallback
func (r *Router) GetPools(ctx context.Context, req PoolsRequest) ([]Pool, error) {
	providers := r.registry.GetLiquidityProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	for _, provider := range providers {
		pools, err := provider.GetPools(ctx, req)
		if err == nil && len(pools) > 0 {
			return pools, nil
		}

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, ErrAllProvidersFailed
}

// GetPool gets a specific pool with fallback
func (r *Router) GetPool(ctx context.Context, chainID ChainID, address string) (*Pool, error) {
	providers := r.registry.GetLiquidityProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	var lastErr error
	for _, provider := range providers {
		pool, err := provider.GetPool(ctx, chainID, address)
		if err == nil {
			return pool, nil
		}
		lastErr = err

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, lastErr
}

// GetPositions gets positions with fallback
func (r *Router) GetPositions(ctx context.Context, req PositionsRequest) ([]Position, error) {
	providers := r.registry.GetLiquidityProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	var lastErr error
	for _, provider := range providers {
		positions, err := provider.GetPositions(ctx, req)
		if err == nil {
			return positions, nil
		}
		lastErr = err

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, lastErr
}

// GetTokenPrice gets token price from the first available provider
func (r *Router) GetTokenPrice(ctx context.Context, token Token) (*TokenPrice, error) {
	providers := r.registry.GetPriceProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	var lastErr error
	for _, provider := range providers {
		price, err := provider.GetTokenPrice(ctx, token)
		if err == nil {
			return price, nil
		}
		lastErr = err

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, lastErr
}

// GetTokenPrices gets token prices from all providers, aggregating results
func (r *Router) GetTokenPrices(ctx context.Context, tokens []Token) ([]TokenPrice, error) {
	providers := r.registry.GetPriceProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	// Try first provider
	for _, provider := range providers {
		prices, err := provider.GetTokenPrices(ctx, tokens)
		if err == nil && len(prices) > 0 {
			return prices, nil
		}

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, ErrAllProvidersFailed
}

// CreateLead creates a conversion lead
func (r *Router) CreateLead(ctx context.Context, lead ConversionLead) (*ConversionLead, error) {
	providers := r.registry.GetConversionProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	var lastErr error
	for _, provider := range providers {
		created, err := provider.CreateLead(ctx, lead)
		if err == nil {
			return created, nil
		}
		lastErr = err

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, lastErr
}

// TrackEvent tracks a conversion event to all providers
func (r *Router) TrackEvent(ctx context.Context, event ConversionEvent) error {
	providers := r.registry.GetConversionProviders()
	if len(providers) == 0 {
		return ErrNoProvidersAvailable
	}

	// Track to all providers (don't fail if one fails)
	var lastErr error
	for _, provider := range providers {
		if err := provider.TrackEvent(ctx, event); err != nil {
			lastErr = err
		}
	}

	return lastErr
}

// GetTokenList gets the token list for a chain from the first available provider
func (r *Router) GetTokenList(ctx context.Context, chainID ChainID) ([]Token, error) {
	providers := r.registry.GetTokenListProviders()
	if len(providers) == 0 {
		return nil, ErrNoProvidersAvailable
	}

	var lastErr error
	for _, provider := range providers {
		tokens, err := provider.GetTokenList(ctx, chainID)
		if err == nil && len(tokens) > 0 {
			return tokens, nil
		}
		lastErr = err

		if !r.enableFallback {
			return nil, err
		}
	}

	return nil, lastErr
}

// HealthCheck performs health checks on all providers
func (r *Router) HealthCheck(ctx context.Context) []HealthCheck {
	return r.registry.HealthCheckAll(ctx)
}

// ListProviders lists all registered providers
func (r *Router) ListProviders() []ProviderInfo {
	return r.registry.ListProviders()
}
