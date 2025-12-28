package gateway

import (
	"context"
	"math/big"
	"testing"
	"time"
)

// MockProvider is a mock provider for testing
type MockProvider struct {
	name     string
	priority int
	chains   []ChainID
	healthy  bool
}

func NewMockProvider(name string, priority int, chains []ChainID) *MockProvider {
	return &MockProvider{
		name:     name,
		priority: priority,
		chains:   chains,
		healthy:  true,
	}
}

func (m *MockProvider) Info() ProviderInfo {
	return ProviderInfo{
		Name:            m.name,
		Version:         "1.0.0",
		Description:     "Mock provider for testing",
		SupportedChains: m.chains,
		Priority:        m.priority,
		Healthy:         m.healthy,
	}
}

func (m *MockProvider) HealthCheck(ctx context.Context) HealthCheck {
	return HealthCheck{
		Provider:  m.name,
		Healthy:   m.healthy,
		Latency:   10,
		LastCheck: time.Now(),
	}
}

func (m *MockProvider) Close() error {
	return nil
}

func (m *MockProvider) GetQuote(ctx context.Context, req QuoteRequest) (*SwapQuote, error) {
	return &SwapQuote{
		TokenIn: TokenAmount{
			Token:  req.TokenIn,
			Amount: req.Amount,
		},
		TokenOut: TokenAmount{
			Token:  req.TokenOut,
			Amount: new(big.Int).Mul(req.Amount, big.NewInt(2)), // 2x for testing
		},
		PriceImpact:  0.01,
		GasEstimate:  big.NewInt(100000),
		ProviderName: m.name,
	}, nil
}

func (m *MockProvider) GetQuotes(ctx context.Context, req QuoteRequest) ([]SwapQuote, error) {
	quote, err := m.GetQuote(ctx, req)
	if err != nil {
		return nil, err
	}
	return []SwapQuote{*quote}, nil
}

func TestRegistryRegisterProvider(t *testing.T) {
	registry := NewRegistry()
	
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	
	// Register should succeed
	if err := registry.RegisterProvider(provider); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	
	// Registering same provider should fail
	if err := registry.RegisterProvider(provider); err == nil {
		t.Fatal("Expected error when registering duplicate provider")
	}
}

func TestRegistryGetProvider(t *testing.T) {
	registry := NewRegistry()
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	
	if err := registry.RegisterProvider(provider); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	
	// Get should succeed
	got, err := registry.GetProvider("test")
	if err != nil {
		t.Fatalf("Failed to get provider: %v", err)
	}
	
	if got.Info().Name != "test" {
		t.Errorf("Expected provider name 'test', got %s", got.Info().Name)
	}
	
	// Get non-existent should fail
	_, err = registry.GetProvider("nonexistent")
	if err == nil {
		t.Fatal("Expected error when getting non-existent provider")
	}
}

func TestRegistryListProviders(t *testing.T) {
	registry := NewRegistry()
	
	// Register providers with different priorities
	p1 := NewMockProvider("high", 10, []ChainID{ChainIDLux})
	p2 := NewMockProvider("low", 100, []ChainID{ChainIDLux})
	p3 := NewMockProvider("medium", 50, []ChainID{ChainIDLux})
	
	registry.RegisterProvider(p1)
	registry.RegisterProvider(p2)
	registry.RegisterProvider(p3)
	
	providers := registry.ListProviders()
	
	if len(providers) != 3 {
		t.Fatalf("Expected 3 providers, got %d", len(providers))
	}
	
	// Should be sorted by priority
	if providers[0].Name != "high" {
		t.Errorf("Expected first provider to be 'high', got %s", providers[0].Name)
	}
	if providers[1].Name != "medium" {
		t.Errorf("Expected second provider to be 'medium', got %s", providers[1].Name)
	}
	if providers[2].Name != "low" {
		t.Errorf("Expected third provider to be 'low', got %s", providers[2].Name)
	}
}

func TestRegistryUnregisterProvider(t *testing.T) {
	registry := NewRegistry()
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	
	if err := registry.RegisterProvider(provider); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	
	// Unregister should succeed
	if err := registry.UnregisterProvider("test"); err != nil {
		t.Fatalf("Failed to unregister provider: %v", err)
	}
	
	// Provider should no longer exist
	_, err := registry.GetProvider("test")
	if err == nil {
		t.Fatal("Expected error when getting unregistered provider")
	}
	
	// Unregister non-existent should fail
	if err := registry.UnregisterProvider("nonexistent"); err == nil {
		t.Fatal("Expected error when unregistering non-existent provider")
	}
}

func TestRouterGetBestQuote(t *testing.T) {
	registry := NewRegistry()
	
	// Create two mock providers that implement QuoteProvider
	p1 := NewMockProvider("fast", 10, []ChainID{ChainIDLux})
	p2 := NewMockProvider("slow", 100, []ChainID{ChainIDLux})
	
	registry.RegisterProvider(p1)
	registry.RegisterProvider(p2)
	
	router := NewRouter(registry, true)
	
	req := QuoteRequest{
		TokenIn: Token{
			Address: "0x1111111111111111111111111111111111111111",
			ChainID: ChainIDLux,
		},
		TokenOut: Token{
			Address: "0x2222222222222222222222222222222222222222",
			ChainID: ChainIDLux,
		},
		Amount:    big.NewInt(1000000),
		IsExactIn: true,
		ChainID:   ChainIDLux,
	}
	
	quote, err := router.GetBestQuote(context.Background(), req)
	if err != nil {
		t.Fatalf("Failed to get quote: %v", err)
	}
	
	if quote == nil {
		t.Fatal("Expected quote, got nil")
	}
	
	// Should return the quote with highest output amount
	expectedOutput := new(big.Int).Mul(req.Amount, big.NewInt(2))
	if quote.TokenOut.Amount.Cmp(expectedOutput) != 0 {
		t.Errorf("Expected output amount %s, got %s", expectedOutput, quote.TokenOut.Amount)
	}
}

func TestRouterNoProviders(t *testing.T) {
	registry := NewRegistry()
	router := NewRouter(registry, true)
	
	req := QuoteRequest{
		TokenIn: Token{
			Address: "0x1111111111111111111111111111111111111111",
			ChainID: ChainIDLux,
		},
		TokenOut: Token{
			Address: "0x2222222222222222222222222222222222222222",
			ChainID: ChainIDLux,
		},
		Amount:    big.NewInt(1000000),
		IsExactIn: true,
		ChainID:   ChainIDLux,
	}
	
	_, err := router.GetBestQuote(context.Background(), req)
	if err == nil {
		t.Fatal("Expected error when no providers available")
	}
	
	if err != ErrNoProvidersAvailable {
		t.Errorf("Expected ErrNoProvidersAvailable, got %v", err)
	}
}

func TestGatewayNew(t *testing.T) {
	config := DefaultGatewayConfig()
	gw := New(config)
	
	if gw == nil {
		t.Fatal("Expected gateway, got nil")
	}
	
	if gw.router == nil {
		t.Fatal("Expected router, got nil")
	}
	
	if gw.registry == nil {
		t.Fatal("Expected registry, got nil")
	}
}

func TestGatewayRegisterProvider(t *testing.T) {
	config := DefaultGatewayConfig()
	gw := New(config)
	
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	
	if err := gw.RegisterProvider(provider); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	
	providers := gw.ListProviders()
	if len(providers) != 1 {
		t.Fatalf("Expected 1 provider, got %d", len(providers))
	}
	
	if providers[0].Name != "test" {
		t.Errorf("Expected provider name 'test', got %s", providers[0].Name)
	}
}

func TestGatewayHealthCheck(t *testing.T) {
	config := DefaultGatewayConfig()
	gw := New(config)
	
	provider := NewMockProvider("test", 10, []ChainID{ChainIDLux})
	gw.RegisterProvider(provider)
	
	checks := gw.HealthCheck(context.Background())
	
	if len(checks) != 1 {
		t.Fatalf("Expected 1 health check, got %d", len(checks))
	}
	
	if !checks[0].Healthy {
		t.Error("Expected healthy provider")
	}
	
	if checks[0].Provider != "test" {
		t.Errorf("Expected provider 'test', got %s", checks[0].Provider)
	}
}

func TestChainIDConstants(t *testing.T) {
	// Verify chain ID constants are correct
	if ChainIDLux != 96369 {
		t.Errorf("Expected ChainIDLux to be 96369, got %d", ChainIDLux)
	}
	
	if ChainIDZoo != 200200 {
		t.Errorf("Expected ChainIDZoo to be 200200, got %d", ChainIDZoo)
	}
	
	if ChainIDEthereum != 1 {
		t.Errorf("Expected ChainIDEthereum to be 1, got %d", ChainIDEthereum)
	}
}

func TestProviderError(t *testing.T) {
	err := &ProviderError{
		Provider: "test",
		Err:      ErrNoProvidersAvailable,
	}
	
	expected := "test: no providers available"
	if err.Error() != expected {
		t.Errorf("Expected error message %q, got %q", expected, err.Error())
	}
	
	// Test Unwrap
	if err.Unwrap() != ErrNoProvidersAvailable {
		t.Error("Expected Unwrap to return original error")
	}
}

func TestWithRequestID(t *testing.T) {
	ctx := context.Background()
	requestID := "test-request-123"
	
	ctxWithID := WithRequestID(ctx, requestID)
	
	got := GetRequestID(ctxWithID)
	if got != requestID {
		t.Errorf("Expected request ID %q, got %q", requestID, got)
	}
	
	// Test getting request ID from context without one
	got = GetRequestID(ctx)
	if got != "" {
		t.Errorf("Expected empty request ID, got %q", got)
	}
}
