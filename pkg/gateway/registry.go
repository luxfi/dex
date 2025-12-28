package gateway

import (
	"context"
	"errors"
	"sort"
	"sync"
)

var (
	ErrProviderNotFound     = errors.New("provider not found")
	ErrProviderExists       = errors.New("provider already exists")
	ErrNoProvidersAvailable = errors.New("no providers available")
	ErrAllProvidersFailed   = errors.New("all providers failed")
)

// Registry implements ProviderRegistry with thread-safe provider management
type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
	order     []string // Sorted by priority
}

// NewRegistry creates a new provider registry
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
		order:     make([]string, 0),
	}
}

// RegisterProvider registers a provider
func (r *Registry) RegisterProvider(provider Provider) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	info := provider.Info()
	if _, exists := r.providers[info.Name]; exists {
		return ErrProviderExists
	}
	
	r.providers[info.Name] = provider
	r.order = append(r.order, info.Name)
	r.sortByPriority()
	
	return nil
}

// UnregisterProvider removes a provider
func (r *Registry) UnregisterProvider(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.providers[name]; !exists {
		return ErrProviderNotFound
	}
	
	delete(r.providers, name)
	
	// Remove from order slice
	newOrder := make([]string, 0, len(r.order)-1)
	for _, n := range r.order {
		if n != name {
			newOrder = append(newOrder, n)
		}
	}
	r.order = newOrder
	
	return nil
}

// GetProvider returns a provider by name
func (r *Registry) GetProvider(name string) (Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	provider, exists := r.providers[name]
	if !exists {
		return nil, ErrProviderNotFound
	}
	
	return provider, nil
}

// ListProviders returns all registered providers
func (r *Registry) ListProviders() []ProviderInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	infos := make([]ProviderInfo, 0, len(r.providers))
	for _, name := range r.order {
		if provider, exists := r.providers[name]; exists {
			infos = append(infos, provider.Info())
		}
	}
	
	return infos
}

// GetQuoteProviders returns all quote providers sorted by priority
func (r *Registry) GetQuoteProviders() []QuoteProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	providers := make([]QuoteProvider, 0)
	for _, name := range r.order {
		if provider, exists := r.providers[name]; exists {
			if qp, ok := provider.(QuoteProvider); ok {
				providers = append(providers, qp)
			}
		}
	}
	
	return providers
}

// GetLiquidityProviders returns all liquidity providers
func (r *Registry) GetLiquidityProviders() []LiquidityProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	providers := make([]LiquidityProvider, 0)
	for _, name := range r.order {
		if provider, exists := r.providers[name]; exists {
			if lp, ok := provider.(LiquidityProvider); ok {
				providers = append(providers, lp)
			}
		}
	}
	
	return providers
}

// GetPriceProviders returns all price providers
func (r *Registry) GetPriceProviders() []PriceProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	providers := make([]PriceProvider, 0)
	for _, name := range r.order {
		if provider, exists := r.providers[name]; exists {
			if pp, ok := provider.(PriceProvider); ok {
				providers = append(providers, pp)
			}
		}
	}
	
	return providers
}

// GetConversionProviders returns all conversion providers
func (r *Registry) GetConversionProviders() []ConversionProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	providers := make([]ConversionProvider, 0)
	for _, name := range r.order {
		if provider, exists := r.providers[name]; exists {
			if cp, ok := provider.(ConversionProvider); ok {
				providers = append(providers, cp)
			}
		}
	}
	
	return providers
}

// GetTokenListProviders returns all token list providers
func (r *Registry) GetTokenListProviders() []TokenListProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	providers := make([]TokenListProvider, 0)
	for _, name := range r.order {
		if provider, exists := r.providers[name]; exists {
			if tp, ok := provider.(TokenListProvider); ok {
				providers = append(providers, tp)
			}
		}
	}

	return providers
}

// sortByPriority sorts providers by priority (lower = higher priority)
func (r *Registry) sortByPriority() {
	sort.Slice(r.order, func(i, j int) bool {
		pi := r.providers[r.order[i]].Info().Priority
		pj := r.providers[r.order[j]].Info().Priority
		return pi < pj
	})
}

// HealthCheckAll performs health checks on all providers
func (r *Registry) HealthCheckAll(ctx context.Context) []HealthCheck {
	r.mu.RLock()
	providers := make([]Provider, 0, len(r.providers))
	for _, name := range r.order {
		providers = append(providers, r.providers[name])
	}
	r.mu.RUnlock()
	
	results := make([]HealthCheck, len(providers))
	var wg sync.WaitGroup
	
	for i, provider := range providers {
		wg.Add(1)
		go func(idx int, p Provider) {
			defer wg.Done()
			results[idx] = p.HealthCheck(ctx)
		}(i, provider)
	}
	
	wg.Wait()
	return results
}

// Close closes all providers
func (r *Registry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	var lastErr error
	for _, provider := range r.providers {
		if err := provider.Close(); err != nil {
			lastErr = err
		}
	}
	
	r.providers = make(map[string]Provider)
	r.order = make([]string, 0)
	
	return lastErr
}
