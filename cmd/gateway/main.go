// Gateway API server that aggregates multiple DEX providers.
// Connects to EVM nodes for on-chain pool data via DEX precompiles.
//
// Configuration via flags or environment variables:
//
//	--rpc / RPC_URL        EVM JSON-RPC endpoint (required)
//	--chain-id / CHAIN_ID  EVM chain ID (required)
//	--addr                 Listen address (default :8080)
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/luxfi/dex/pkg/gateway"
	"github.com/luxfi/dex/pkg/gateway/lux"
)

func main() {
	addr := flag.String("addr", ":8080", "Server listen address")
	rpcURL := flag.String("rpc", "", "EVM JSON-RPC endpoint (env: RPC_URL)")
	chainIDFlag := flag.String("chain-id", "", "EVM chain ID (env: CHAIN_ID)")
	flag.Parse()

	// Resolve from env if not set via flags
	if *rpcURL == "" {
		*rpcURL = os.Getenv("RPC_URL")
	}
	if *chainIDFlag == "" {
		*chainIDFlag = os.Getenv("CHAIN_ID")
	}

	if *rpcURL == "" {
		log.Fatal("--rpc or RPC_URL is required")
	}
	if *chainIDFlag == "" {
		log.Fatal("--chain-id or CHAIN_ID is required")
	}

	chainID, err := strconv.ParseUint(*chainIDFlag, 10, 64)
	if err != nil {
		log.Fatalf("invalid chain ID %q: %v", *chainIDFlag, err)
	}

	gwChainID := gateway.ChainID(chainID)

	config := gateway.GatewayConfig{
		DefaultChainID:  gwChainID,
		EnableFallback:  false,
		CacheEnabled:    true,
		CacheTTLSeconds: 60,
	}

	gw := gateway.New(config)

	provider := lux.NewProvider(lux.ProviderConfig{
		Name:     "native",
		Priority: 10,
		Chains:   []gateway.ChainID{gwChainID},
		RPCEndpoints: map[gateway.ChainID]string{
			gwChainID: *rpcURL,
		},
	})

	if err := gw.RegisterProvider(provider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	log.Printf("DEX gateway starting on %s", *addr)
	log.Printf("  Chain ID: %d", chainID)
	log.Printf("  RPC: %s", *rpcURL)
	fmt.Println()

	if err := gw.StartWithGracefulShutdown(*addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
