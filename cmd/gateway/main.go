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

	// Every chain is read from its own pools. Ours through the V4 precompiles
	// with V2 beside them when a router is deployed; every other chain through
	// the V2 router and the V3 quoter Uniswap published there. No API key, no
	// upstream account: the hosted API answers ACCESS_DENIED without a
	// contract, and a pool answers anyone.
	chains := gateway.DefaultChainVenues(rpc)
	mine := chains[gateway.ChainID(chain)]
	mine.RPC = rpc
	mine.Native = true
	if router := strings.TrimSpace(os.Getenv("LUX_V2_ROUTER")); router != "" {
		mine.V2Router = router
	}
	if quoter := strings.TrimSpace(os.Getenv("LUX_V3_QUOTER")); quoter != "" {
		mine.V3Quoter = quoter
	}
	chains[gateway.ChainID(chain)] = mine

	// A deployment overrides any chain's endpoint by name, because a public RPC
	// is a courtesy and an interface with traffic wants its own.
	for id := range chains {
		if url := strings.TrimSpace(os.Getenv("RPC_" + strconv.FormatUint(uint64(id), 10))); url != "" {
			c := chains[id]
			c.RPC = url
			chains[id] = c
		}
	}

	gw := gateway.New(gateway.GatewayConfig{
		DefaultChainID: gateway.ChainID(chain),
		EnableFallback: true,
	})

	// The hosted provider is registered only when a key is given. Without one
	// it refuses every request, and a provider that always refuses is worse
	// than none: it turns a pair with no pool into an error about somebody
	// else's quota.
	if key := strings.TrimSpace(os.Getenv("UNISWAP_API_KEY")); key != "" {
		up := uniswap.DefaultConfig()
		up.APIKey = key
		up.Timeout = 30 * time.Second
		if err := gw.RegisterProvider(uniswap.NewProvider(up)); err != nil {
			log.Fatalf("uniswap provider: %v", err)
		}
		log.Printf("hosted upstream registered for %v", up.Chains)
	}

	ttl := 10 * time.Second
	if n, err := strconv.Atoi(env("GATEWAY_CACHE_TTL", "10")); err == nil && n >= 0 {
		ttl = time.Duration(n) * time.Second
	}

	cfg := gateway.DefaultServerConfig()
	cfg.Addr = addr
	routers := gateway.NewChainRouters(chains)
	server := gateway.NewServer(gw.GetRouter(), cfg,
		gateway.WithChainVenues(routers),
		gateway.WithQuoteCache(ttl))

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()

	log.Printf("trading gateway on %s — chain %d at %s, venues on %v, quotes cached %s",
		addr, chain, rpc, routers.Chains(), ttl)
	if err := server.Start(); err != nil {
		log.Fatalf("gateway: %v", err)
	}
}
