// The trading surface, served.
//
// `pkg/gateway` has carried the whole thing for a while — the quote, swap,
// approval and order paths, a router over venues, and a Uniswap provider that
// speaks the shapes at trade-api.gateway.uniswap.org — and nothing started it.
// `dexd run -http` mounts the D-Chain venue's own handlers under /v1/dex, which
// is a different surface for a different reader. So an interface asking for a
// quote reached nothing, on any chain.
//
// One binary, and one rule about what it answers: a market on a Lux chain is
// quoted from our own contracts, and a market anywhere else is quoted upstream.
// The response is the same shape either way, because a client should not have
// to know which arm served it — that is the whole reason a gateway exists
// rather than two clients in the interface.
//
// Everything it needs comes from the environment, because the addresses differ
// per chain and per deployment and a binary that carries them is a binary that
// is wrong somewhere.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/luxfi/dex/pkg/gateway"
	"github.com/luxfi/dex/pkg/gateway/uniswap"
)

func env(name, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return fallback
}

func main() {
	addr := env("GATEWAY_ADDR", ":8080")
	rpc := env("LUX_RPC", "https://api.lux.cloud/v1/chain/C/rpc")

	chain, err := strconv.ParseUint(env("LUX_CHAIN_ID", "96369"), 10, 64)
	if err != nil {
		log.Fatalf("LUX_CHAIN_ID: %v", err)
	}

	// Our own chain, quoted from our own contracts. The native venue reads the
	// DEX precompiles; a V2 router is added beside it when one is deployed,
	// which is a fact about the chain rather than about this binary — an
	// unset address means the venue is not offered, not that it is broken.
	venues := []gateway.Venue{
		gateway.NewNativeDEXVenue(gateway.NativeDEXConfig{RPCURL: rpc, UseDEX: true}),
	}
	if router := strings.TrimSpace(os.Getenv("LUX_V2_ROUTER")); router != "" {
		venues = append(venues, gateway.NewUniswapV2Venue(gateway.UniswapV2Config{
			RPCURL:        rpc,
			RouterAddress: router,
			Name:          "lux_v2",
		}))
	}

	gw := gateway.New(gateway.GatewayConfig{
		DefaultChainID: gateway.ChainID(chain),
		EnableFallback: true,
		CacheEnabled:   true,
		CacheTTLSeconds: func() int {
			n, err := strconv.Atoi(env("GATEWAY_CACHE_TTL", "10"))
			if err != nil {
				return 10
			}
			return n
		}(),
	})

	// Every other chain, quoted upstream. The provider already names the six
	// Uniswap serves — Ethereum, Arbitrum, Optimism, Polygon, Base, BNB — and
	// an API key is optional: without one the public endpoints answer, with one
	// they answer under a quota that is ours.
	up := uniswap.DefaultConfig()
	up.APIKey = os.Getenv("UNISWAP_API_KEY")
	up.Timeout = 30 * time.Second
	if err := gw.RegisterProvider(uniswap.NewProvider(up)); err != nil {
		log.Fatalf("uniswap provider: %v", err)
	}

	cfg := gateway.DefaultServerConfig()
	cfg.Addr = addr
	server := gateway.NewServer(gw.GetRouter(), cfg, gateway.WithVenues(gateway.NewVenueRouter(venues...)))

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()

	log.Printf("trading gateway on %s — chain %d at %s, %d native venue(s), upstream for %v",
		addr, chain, rpc, len(venues), up.Chains)
	if err := server.Start(); err != nil {
		log.Fatalf("gateway: %v", err)
	}
}
