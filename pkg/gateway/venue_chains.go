package gateway

import (
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Where each chain's pools are read from.
//
// Every one of these is a contract on a public chain, read over JSON-RPC with
// no key and no upstream account. That is the point: an interface that needs a
// vendor's API to price a Uniswap pool is an interface that stops working when
// the vendor says no, and it said no — the hosted API answers ACCESS_DENIED
// without a contract. The pool is the source.
type ChainVenues struct {
	// RPC is the endpoint the venues read.
	RPC string
	// V2Router is a Router02 (getAmountsOut). Empty means no V2 arm.
	V2Router string
	// V3Quoter is a QuoterV2. Empty means no V3 arm.
	V3Quoter string
	// Native turns on the Lux precompile venue — the V4 arm, PoolManager at
	// 0x9010 with the order book at 0x9020 beside it. Only our chains have it.
	Native bool
}

// The addresses Uniswap publishes for its own deployments. V2's Router02 is at
// one address on the chains it was deployed to later, and at its original one
// on Ethereum; QuoterV2 is at one address nearly everywhere.
const (
	v2RouterEthereum = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
	v2RouterCommon   = "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24"
	v3QuoterCommon   = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
	v3QuoterBase     = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"
	v3QuoterBNB      = "0x78D78E420Da98ad378D7799bE8f4AF69033EB077"
)

// DefaultChainVenues is what a deployment reads when it is told nothing else:
// our own chain, and the chains Uniswap deployed to. The RPCs are public and
// are meant to be overridden per deployment — a busy interface wants its own.
func DefaultChainVenues(luxRPC string) map[ChainID]ChainVenues {
	return map[ChainID]ChainVenues{
		96369:           {RPC: luxRPC, Native: true},
		ChainIDEthereum: {RPC: "https://ethereum-rpc.publicnode.com", V2Router: v2RouterEthereum, V3Quoter: v3QuoterCommon},
		ChainIDArbitrum: {RPC: "https://arbitrum-one-rpc.publicnode.com", V2Router: v2RouterCommon, V3Quoter: v3QuoterCommon},
		ChainIDOptimism: {RPC: "https://optimism-rpc.publicnode.com", V2Router: v2RouterCommon, V3Quoter: v3QuoterCommon},
		ChainIDPolygon:  {RPC: "https://polygon-bor-rpc.publicnode.com", V2Router: v2RouterCommon, V3Quoter: v3QuoterCommon},
		ChainIDBase:     {RPC: "https://base-rpc.publicnode.com", V2Router: v2RouterCommon, V3Quoter: v3QuoterBase},
		ChainIDBNB:      {RPC: "https://bsc-rpc.publicnode.com", V2Router: v2RouterCommon, V3Quoter: v3QuoterBNB},
	}
}

// Venues builds the venue arms one chain offers.
func (c ChainVenues) Venues() []Venue {
	out := make([]Venue, 0, 3)
	if c.Native {
		out = append(out, NewNativeDEXVenue(NativeDEXConfig{RPCURL: c.RPC, UseDEX: true}))
	}
	if c.V2Router != "" {
		out = append(out, NewUniswapV2Venue(UniswapV2Config{RPCURL: c.RPC, RouterAddress: c.V2Router, Name: "uniswap_v2"}))
	}
	if c.V3Quoter != "" {
		out = append(out, NewUniswapV3Venue(UniswapV3Config{RPCURL: c.RPC, QuoterAddress: c.V3Quoter, Name: "uniswap_v3"}))
	}
	return out
}

// ChainRouters is the venues per chain, and the endpoint each was read from.
// The endpoint is kept because quoting is not the only thing that asks a chain
// a question — an allowance is read from the same place the pool is.
type ChainRouters struct {
	byChain map[ChainID]*VenueRouter
	rpc     map[ChainID]string
}

// NewChainRouters builds one venue router per chain.
func NewChainRouters(chains map[ChainID]ChainVenues) *ChainRouters {
	byChain := make(map[ChainID]*VenueRouter, len(chains))
	rpc := make(map[ChainID]string, len(chains))
	for id, c := range chains {
		if v := c.Venues(); len(v) > 0 {
			byChain[id] = NewVenueRouter(v...)
			rpc[id] = c.RPC
		}
	}
	return &ChainRouters{byChain: byChain, rpc: rpc}
}

// RPC is the endpoint this deployment reads one chain from, or "" for a chain
// it does not read.
func (c *ChainRouters) RPC(chain ChainID) string {
	if c == nil {
		return ""
	}
	return c.rpc[chain]
}

// For returns the venues that read one chain, or nil.
func (c *ChainRouters) For(chain ChainID) *VenueRouter {
	if c == nil {
		return nil
	}
	return c.byChain[chain]
}

// Chains returns every chain with venues, in order, for the health and venue
// listings. Ordered because a listing that permutes itself between two reads
// of the same deployment reads as a deployment that changed.
func (c *ChainRouters) Chains() []ChainID {
	if c == nil {
		return nil
	}
	out := make([]ChainID, 0, len(c.byChain))
	for id := range c.byChain {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// A quote is a reading of a pool at a moment, and a pool moves when somebody
// trades in it. Cached, briefly: the same pair asked twice in the same second
// is one reading, and every extra call is an eth_call a public endpoint is
// entitled to refuse. The key is the whole question, so two questions never
// share an answer.
type quoteCache struct {
	mu   sync.Mutex
	ttl  time.Duration
	kept map[string]cached
}

type cached struct {
	quote *SwapQuote
	until time.Time
}

func newQuoteCache(ttl time.Duration) *quoteCache {
	return &quoteCache{ttl: ttl, kept: map[string]cached{}}
}

func quoteKey(req QuoteRequest) string {
	var b strings.Builder
	b.WriteString(strings.ToLower(req.TokenIn.Address))
	b.WriteByte('>')
	b.WriteString(strings.ToLower(req.TokenOut.Address))
	b.WriteByte('@')
	b.WriteString(strconv.FormatUint(uint64(req.ChainID), 10))
	b.WriteByte('#')
	if req.Amount != nil {
		b.WriteString(req.Amount.String())
	}
	if req.IsExactIn {
		b.WriteString("|in")
	} else {
		b.WriteString("|out")
	}
	return b.String()
}

func (c *quoteCache) get(req QuoteRequest) *SwapQuote {
	if c == nil || c.ttl <= 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	got, ok := c.kept[quoteKey(req)]
	if !ok || time.Now().After(got.until) {
		return nil
	}
	return got.quote
}

func (c *quoteCache) put(req QuoteRequest, quote *SwapQuote) {
	if c == nil || c.ttl <= 0 || quote == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// A cache nobody sweeps is a leak with a good reputation. Every write
	// drops what has expired, which on this traffic is cheaper than a timer.
	now := time.Now()
	for k, v := range c.kept {
		if now.After(v.until) {
			delete(c.kept, k)
		}
	}
	c.kept[quoteKey(req)] = cached{quote: quote, until: now.Add(c.ttl)}
}
