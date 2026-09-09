package gateway

import (
	_ "embed"
	"encoding/json"
	"sort"
	"strings"
)

// The tokens this gateway can trade, and nothing else about them.
//
// A screen drawing a market needs three things before it can draw anything: a
// name, a symbol, and the decimals an amount is read with. Our own chain has an
// indexer that answers that; the twenty others do not, and the hosted API that
// would answer costs a contract — it returns ACCESS_DENIED without one. So the
// list travels with the binary. Nine hundred and thirty-five tokens over the
// seven chains this gateway prices, each entry as that token's own contract
// states it, and no request to anybody to get them.
//
// It carries NO logo, and has nowhere to put one. Every public token list gives
// logos as URLs on assets.coingecko.com, coin-images.coingecko.com,
// raw.githubusercontent.com and s2.coinmarketcap.com — three companies fetched
// by every viewer's browser on every market screen, none of them ours. Marks
// come from cdn.lux.cloud, and a token the cdn has no mark for draws its own
// letters. A field that does not exist here cannot be filled in by accident.
//
//go:embed tokens.json
var vendored []byte

// listed is one entry of the embedded list. It is the standard token-list
// shape minus the logo and the bridge extensions, which is the whole of what a
// screen needs and none of what it should be fetching from elsewhere.
type listed struct {
	ChainID  uint64 `json:"chainId"`
	Address  string `json:"address"`
	Symbol   string `json:"symbol"`
	Name     string `json:"name"`
	Decimals int    `json:"decimals"`
}

// Catalog answers the two questions a market screen asks about a token: what
// does this chain list, and what is the token at this address.
type Catalog struct {
	// byChain keeps the file's order, which is the order a listing is drawn
	// in: the chain's own token, then the dollar it is priced in, then the
	// rest by symbol. There is no volume on any chain but ours, so there is
	// nothing to rank by, and an order that permutes itself between two reads
	// reads as a venue that changed.
	byChain map[ChainID][]Token
	byAddr  map[ChainID]map[string]Token
}

// NewCatalog reads the embedded list, keeping the chains named.
//
// A chain this deployment cannot quote has no tokens here. /v1/trade/venues and
// /v1/trade/tokens then answer about one set rather than two, and a screen
// cannot list a market this gateway has no way to price.
func NewCatalog(chains []ChainID) *Catalog {
	var doc struct {
		Tokens []listed `json:"tokens"`
	}
	// The file ships in the binary and a test reads every field of it, so a
	// malformed one fails the build rather than a request.
	if err := json.Unmarshal(vendored, &doc); err != nil {
		panic("gateway: embedded token list: " + err.Error())
	}

	keep := make(map[ChainID]bool, len(chains))
	for _, id := range chains {
		keep[id] = true
	}

	c := &Catalog{
		byChain: map[ChainID][]Token{},
		byAddr:  map[ChainID]map[string]Token{},
	}
	for _, l := range doc.Tokens {
		id := ChainID(l.ChainID)
		if !keep[id] {
			continue
		}
		t := Token{
			Address:  l.Address,
			ChainID:  id,
			Decimals: l.Decimals,
			Symbol:   l.Symbol,
			Name:     l.Name,
		}
		c.byChain[id] = append(c.byChain[id], t)
		if c.byAddr[id] == nil {
			c.byAddr[id] = map[string]Token{}
		}
		c.byAddr[id][strings.ToLower(t.Address)] = t
	}
	return c
}

// Token is the token at one address, as its contract states it.
//
// Addresses are compared in lower case and returned in the case the list gives
// them, which is the checksummed form a wallet displays. A caller that types an
// address in either case is asking about the same token, and answering "no such
// token" because of a capital letter is a fact about string comparison dressed
// as a fact about the chain.
func (c *Catalog) Token(chain ChainID, address string) (Token, bool) {
	t, ok := c.byAddr[chain][strings.ToLower(address)]
	return t, ok
}

// Chains is every chain with tokens here, in ascending order.
func (c *Catalog) Chains() []ChainID {
	out := make([]ChainID, 0, len(c.byChain))
	for id := range c.byChain {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// List is a chain's tokens, or every chain's when none is named, narrowed to
// what a caller typed and capped at limit.
func (c *Catalog) List(chain ChainID, query string, limit int) []Token {
	var from []Token
	if chain != 0 {
		from = c.byChain[chain]
	} else {
		for _, id := range c.Chains() {
			from = append(from, c.byChain[id]...)
		}
	}

	query = strings.TrimSpace(query)
	if query == "" {
		if limit < len(from) {
			from = from[:limit]
		}
		out := make([]Token, len(from))
		copy(out, from)
		return out
	}

	// Ranked, because a search that returns matches in storage order is a
	// search nobody can use: "us" on Ethereum contains eleven tokens before
	// USDC, and a person who typed the symbol of the thing they wanted has to
	// scroll past all of them.
	type hit struct {
		token Token
		rank  int
		at    int
	}
	q := strings.ToLower(query)
	hits := make([]hit, 0, 64)
	for i, t := range from {
		if r := rank(t, q); r >= 0 {
			hits = append(hits, hit{token: t, rank: r, at: i})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].rank != hits[j].rank {
			return hits[i].rank < hits[j].rank
		}
		return hits[i].at < hits[j].at
	})
	if limit < len(hits) {
		hits = hits[:limit]
	}
	out := make([]Token, len(hits))
	for i, h := range hits {
		out[i] = h.token
	}
	return out
}

// rank orders one token against what was typed, lower being nearer. It returns
// -1 for a token the query does not touch at all.
//
// The rule is one sentence: the symbol beats the name, and within each, an
// exact match beats a start beats a mention. People type tickers at a market
// screen, so "eth" has to reach WETH before it reaches Ethena and Ethereum
// Name Service, both of whose NAMES start with it.
//
// An address is matched whole and never by fragment. A partial address is a
// typo, and offering the token that happens to share a prefix with a mistyped
// one is how funds reach the wrong contract.
func rank(t Token, q string) int {
	symbol, name := strings.ToLower(t.Symbol), strings.ToLower(t.Name)
	switch {
	case strings.ToLower(t.Address) == q:
		return 0
	case symbol == q:
		return 1
	case strings.HasPrefix(symbol, q):
		return 2
	case strings.Contains(symbol, q):
		return 3
	case strings.HasPrefix(name, q):
		return 4
	case strings.Contains(name, q):
		return 5
	}
	return -1
}
