// Gateway API server that aggregates multiple DEX providers.
// Connects to real Lux nodes for on-chain pool data with Uniswap fallback.
package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/luxfi/dex/pkg/gateway"
	"github.com/luxfi/dex/pkg/gateway/lux"
	"github.com/luxfi/dex/pkg/gateway/uniswap"
)

func main() {
	// Command line flags
	addr := flag.String("addr", ":8080", "Server address")
	uniswapAPIKey := flag.String("uniswap-api-key", "", "Uniswap API key (optional)")
	enableFallback := flag.Bool("fallback", true, "Enable fallback to next provider on failure")

	// RPC endpoints for Lux chains
	luxRPC := flag.String("lux-rpc", "", "Lux mainnet C-Chain RPC endpoint")
	zooRPC := flag.String("zoo-rpc", "", "Zoo chain RPC endpoint")
	flag.Parse()

	// Check environment variables for RPC endpoints
	if *luxRPC == "" {
		*luxRPC = os.Getenv("LUX_RPC_URL")
	}
	if *zooRPC == "" {
		*zooRPC = os.Getenv("ZOO_RPC_URL")
	}

	// Default endpoints if not provided
	if *luxRPC == "" {
		*luxRPC = "http://127.0.0.1:9630/ext/bc/C/rpc"
	}
	if *zooRPC == "" {
		*zooRPC = "http://127.0.0.1:9630/ext/bc/2iJykKjE7gpWNjGUvGG6fVtj7u5Tbvo89CVCu6gjNPCnEdCVpY/rpc"
	}

	// Create gateway with configuration
	config := gateway.GatewayConfig{
		DefaultChainID:  gateway.ChainIDLux,
		EnableFallback:  *enableFallback,
		CacheEnabled:    true,
		CacheTTLSeconds: 60,
	}
	
	gw := gateway.New(config)

	// Register Lux native provider with RPC connectivity (highest priority)
	rpcEndpoints := map[gateway.ChainID]string{
		gateway.ChainIDLux: *luxRPC,
		gateway.ChainIDZoo: *zooRPC,
	}

	luxProvider := lux.NewProvider(lux.ProviderConfig{
		Name:         "lux",
		Priority:     10, // Lower number = higher priority
		Chains:       []gateway.ChainID{gateway.ChainIDLux, gateway.ChainIDZoo},
		RPCEndpoints: rpcEndpoints,
	})

	if err := gw.RegisterProvider(luxProvider); err != nil {
		log.Fatalf("Failed to register Lux provider: %v", err)
	}
	log.Printf("Registered Lux native provider (priority: 10)")
	log.Printf("  LUX RPC: %s", *luxRPC)
	log.Printf("  ZOO RPC: %s", *zooRPC)

	// Register Uniswap provider (fallback for all chains)
	uniswapProvider := uniswap.NewProvider(uniswap.ProviderConfig{
		Name:     "uniswap",
		Priority: 100, // Lower priority - used as fallback
		APIKey:   *uniswapAPIKey,
		Timeout:  30 * time.Second,
		Chains: []gateway.ChainID{
			gateway.ChainIDEthereum,
			gateway.ChainIDArbitrum,
			gateway.ChainIDOptimism,
			gateway.ChainIDPolygon,
			gateway.ChainIDBase,
			gateway.ChainIDBNB,
			// Also supports Lux chains as fallback
			gateway.ChainIDLux,
			gateway.ChainIDZoo,
		},
	})
	
	if err := gw.RegisterProvider(uniswapProvider); err != nil {
		log.Fatalf("Failed to register Uniswap provider: %v", err)
	}
	log.Println("Registered Uniswap provider (priority: 100, fallback)")

	// Start gateway with graceful shutdown
	log.Printf("Gateway server starting on %s", *addr)
	log.Println("Provider priority: lux (10) > uniswap (100)")
	log.Println("Supported chains: Lux (96369), Zoo (200200), Ethereum (1), Arbitrum (42161), Optimism (10), Polygon (137), Base (8453), BNB (56)")
	
	if err := gw.StartWithGracefulShutdown(*addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
