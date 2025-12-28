package gateway

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// Gateway is the main API gateway that aggregates multiple providers
type Gateway struct {
	registry *Registry
	router   *Router
	server   *Server
	config   GatewayConfig
}

// New creates a new Gateway with the given configuration
func New(config GatewayConfig) *Gateway {
	registry := NewRegistry()
	router := NewRouter(registry, config.EnableFallback)
	
	return &Gateway{
		registry: registry,
		router:   router,
		config:   config,
	}
}

// RegisterProvider registers a provider with the gateway
func (g *Gateway) RegisterProvider(provider Provider) error {
	return g.registry.RegisterProvider(provider)
}

// UnregisterProvider removes a provider
func (g *Gateway) UnregisterProvider(name string) error {
	return g.registry.UnregisterProvider(name)
}

// GetRouter returns the router for direct access
func (g *Gateway) GetRouter() *Router {
	return g.router
}

// GetRegistry returns the registry for direct access
func (g *Gateway) GetRegistry() *Registry {
	return g.registry
}

// Start starts the gateway HTTP server
func (g *Gateway) Start(addr string) error {
	cfg := DefaultServerConfig()
	cfg.Addr = addr
	
	g.server = NewServer(g.router, cfg)
	
	log.Printf("Starting gateway server on %s", addr)
	log.Printf("Registered providers: %d", len(g.registry.ListProviders()))
	for _, info := range g.registry.ListProviders() {
		log.Printf("  - %s (priority: %d, chains: %v)", info.Name, info.Priority, info.SupportedChains)
	}
	
	return g.server.Start()
}

// StartWithGracefulShutdown starts the server with graceful shutdown handling
func (g *Gateway) StartWithGracefulShutdown(addr string) error {
	cfg := DefaultServerConfig()
	cfg.Addr = addr
	
	g.server = NewServer(g.router, cfg)
	
	// Channel to receive OS signals
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	
	// Channel to receive server errors
	errChan := make(chan error, 1)
	
	go func() {
		log.Printf("Starting gateway server on %s", addr)
		log.Printf("Registered providers: %d", len(g.registry.ListProviders()))
		for _, info := range g.registry.ListProviders() {
			log.Printf("  - %s (priority: %d, chains: %v)", info.Name, info.Priority, info.SupportedChains)
		}
		
		if err := g.server.Start(); err != nil {
			errChan <- err
		}
	}()
	
	select {
	case err := <-errChan:
		return err
	case sig := <-quit:
		log.Printf("Received signal %v, shutting down...", sig)
		
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		
		if err := g.Shutdown(ctx); err != nil {
			return fmt.Errorf("shutdown error: %w", err)
		}
		
		log.Println("Server gracefully stopped")
		return nil
	}
}

// Shutdown gracefully shuts down the gateway
func (g *Gateway) Shutdown(ctx context.Context) error {
	var errs []error
	
	// Shutdown HTTP server
	if g.server != nil {
		if err := g.server.Shutdown(ctx); err != nil {
			errs = append(errs, fmt.Errorf("server shutdown: %w", err))
		}
	}
	
	// Close all providers
	if err := g.registry.Close(); err != nil {
		errs = append(errs, fmt.Errorf("registry close: %w", err))
	}
	
	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}
	
	return nil
}

// HealthCheck performs health checks on all providers
func (g *Gateway) HealthCheck(ctx context.Context) []HealthCheck {
	return g.router.HealthCheck(ctx)
}

// ListProviders returns information about all registered providers
func (g *Gateway) ListProviders() []ProviderInfo {
	return g.registry.ListProviders()
}

// DefaultGatewayConfig returns the default gateway configuration
func DefaultGatewayConfig() GatewayConfig {
	return GatewayConfig{
		Providers:       []ProviderConfig{},
		DefaultChainID:  ChainIDLux,
		EnableFallback:  true,
		CacheEnabled:    true,
		CacheTTLSeconds: 60,
	}
}
