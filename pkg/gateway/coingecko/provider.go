// Where the tokens come from.
//
// A chain's pools tell you what a pair is worth and nothing about what tokens
// exist: an AMM has no list. Our own chains have an indexer that keeps one, and
// every other chain had nothing — which is why an interface switched to
// Ethereum, asked for nothing, and drew an empty table. The hosted Uniswap API
// would answer, and refuses without a contract.
//
// CoinGecko publishes both halves openly: a per-chain token list with names,
// decimals and logos, and a price by contract address with the day's change and
// volume. No key, no account. Measured: 5,821 tokens on Ethereum, 4,214 on BNB,
// 2,679 on Base, every one with a logo.
//
// The lists are large and change slowly, so they are held for hours; a price is
// a reading of a market and is held for a minute. Both are held HERE rather
// than in a browser, so a thousand readers are one request to a service that
// asks nothing of us.
package coingecko

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/luxfi/dex/pkg/gateway"
)

// The name CoinGecko gives each chain. A chain absent here is a chain this
// provider does not claim — it is not an error, it is a gap, and the registry
// asks somebody else.
var platforms = map[gateway.ChainID]string{
	gateway.ChainIDEthereum: "ethereum",
	gateway.ChainIDBase:     "base",
	gateway.ChainIDArbitrum: "arbitrum-one",
	gateway.ChainIDPolygon:  "polygon-pos",
	gateway.ChainIDOptimism: "optimistic-ethereum",
	gateway.ChainIDBNB:      "binance-smart-chain",
}

const (
	listHost  = "https://tokens.coingecko.com"
	priceHost = "https://api.coingecko.com/api/v3"
	listTTL   = 6 * time.Hour
	priceTTL  = time.Minute
	// A price call takes a batch of addresses. CoinGecko's URL is the limit
	// rather than a documented count, so the batch is small enough that the
	// request stays well inside any gateway's line length.
	batch = 40
)

type Provider struct {
	http  *http.Client
	key   string // optional; a free account raises the rate limit and nothing else
	mu    sync.Mutex
	lists map[gateway.ChainID]listing
	price map[string]quoted
}

type listing struct {
	tokens []gateway.Token
	until  time.Time
}

type quoted struct {
	price gateway.TokenPrice
	until time.Time
}

type Config struct {
	APIKey  string
	Timeout time.Duration
}

func New(cfg Config) *Provider {
	t := cfg.Timeout
	if t == 0 {
		t = 20 * time.Second
	}
	return &Provider{
		http:  &http.Client{Timeout: t},
		key:   strings.TrimSpace(cfg.APIKey),
		lists: map[gateway.ChainID]listing{},
		price: map[string]quoted{},
	}
}

func (p *Provider) Info() gateway.ProviderInfo {
	chains := make([]gateway.ChainID, 0, len(platforms))
	for id := range platforms {
		chains = append(chains, id)
	}
	return gateway.ProviderInfo{
		Name:            "coingecko",
		Version:         "1.0.0",
		Description:     "Token lists and prices for public chains, read from CoinGecko's open endpoints",
		SupportedChains: chains,
		Priority:        200,
	}
}

func (p *Provider) HealthCheck(ctx context.Context) gateway.HealthCheck {
	start := time.Now()
	err := p.get(ctx, priceHost+"/ping", new(map[string]any))
	return gateway.HealthCheck{
		Provider:  "coingecko",
		Healthy:   err == nil,
		Latency:   time.Since(start).Milliseconds(),
		LastCheck: time.Now(),
		Error:     errText(err),
	}
}

func (p *Provider) Close() error { return nil }

func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func (p *Provider) get(ctx context.Context, url string, into any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("accept", "application/json")
	if p.key != "" {
		req.Header.Set("x-cg-demo-api-key", p.key)
	}
	res, err := p.http.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("coingecko: %s answered %d", url, res.StatusCode)
	}
	return json.NewDecoder(res.Body).Decode(into)
}

// GetTokenList returns every token CoinGecko lists for a chain.
func (p *Provider) GetTokenList(ctx context.Context, chain gateway.ChainID) ([]gateway.Token, error) {
	platform, ok := platforms[chain]
	if !ok {
		return nil, fmt.Errorf("coingecko: no list for chain %d", chain)
	}

	p.mu.Lock()
	if got, ok := p.lists[chain]; ok && time.Now().Before(got.until) {
		p.mu.Unlock()
		return got.tokens, nil
	}
	p.mu.Unlock()

	var body struct {
		Tokens []struct {
			Address  string `json:"address"`
			ChainID  int64  `json:"chainId"`
			Decimals int    `json:"decimals"`
			Symbol   string `json:"symbol"`
			Name     string `json:"name"`
			LogoURI  string `json:"logoURI"`
		} `json:"tokens"`
	}
	if err := p.get(ctx, fmt.Sprintf("%s/%s/all.json", listHost, platform), &body); err != nil {
		return nil, err
	}

	tokens := make([]gateway.Token, 0, len(body.Tokens))
	for _, t := range body.Tokens {
		// The list carries every chain it knows in one file on some platforms;
		// a row for another chain under this chain's name is the one mistake
		// that would put a price on an address that does not exist here.
		if t.ChainID != 0 && gateway.ChainID(t.ChainID) != chain {
			continue
		}
		tokens = append(tokens, gateway.Token{
			Address:  strings.ToLower(t.Address),
			ChainID:  chain,
			Decimals: t.Decimals,
			Symbol:   t.Symbol,
			Name:     t.Name,
			LogoURI:  t.LogoURI,
		})
	}

	p.mu.Lock()
	p.lists[chain] = listing{tokens: tokens, until: time.Now().Add(listTTL)}
	p.mu.Unlock()
	return tokens, nil
}

// SearchTokens answers by symbol, name or address, over the same list.
func (p *Provider) SearchTokens(ctx context.Context, chain gateway.ChainID, query string) ([]gateway.Token, error) {
	all, err := p.GetTokenList(ctx, chain)
	if err != nil {
		return nil, err
	}
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return all, nil
	}

	// An exact symbol first, then an address, then anything containing it —
	// so searching USDC does not bury USD Coin under a hundred tokens whose
	// names merely mention it.
	var exact, addr, rest []gateway.Token
	for _, t := range all {
		switch {
		case strings.ToLower(t.Symbol) == q:
			exact = append(exact, t)
		case strings.ToLower(t.Address) == q:
			addr = append(addr, t)
		case strings.Contains(strings.ToLower(t.Symbol), q) || strings.Contains(strings.ToLower(t.Name), q):
			rest = append(rest, t)
		}
	}
	return append(append(exact, addr...), rest...), nil
}

// GetToken returns one token by address.
func (p *Provider) GetToken(ctx context.Context, chain gateway.ChainID, address string) (*gateway.Token, error) {
	all, err := p.GetTokenList(ctx, chain)
	if err != nil {
		return nil, err
	}
	want := strings.ToLower(address)
	for i := range all {
		if all[i].Address == want {
			return &all[i], nil
		}
	}
	return nil, fmt.Errorf("coingecko: %s is not listed on chain %d", address, chain)
}

// GetTokenPrice returns one token's price.
func (p *Provider) GetTokenPrice(ctx context.Context, token gateway.Token) (*gateway.TokenPrice, error) {
	prices, err := p.GetTokenPrices(ctx, []gateway.Token{token})
	if err != nil {
		return nil, err
	}
	if len(prices) == 0 {
		return nil, fmt.Errorf("coingecko: no price for %s", token.Address)
	}
	return &prices[0], nil
}

// GetTokenPrices prices a batch by contract address.
//
// A token CoinGecko does not price is absent from the answer rather than
// present at zero. A screen can draw a dash; it cannot un-draw a wrong number.
func (p *Provider) GetTokenPrices(ctx context.Context, tokens []gateway.Token) ([]gateway.TokenPrice, error) {
	if len(tokens) == 0 {
		return nil, nil
	}
	chain := tokens[0].ChainID
	platform, ok := platforms[chain]
	if !ok {
		return nil, fmt.Errorf("coingecko: no prices for chain %d", chain)
	}

	out := make([]gateway.TokenPrice, 0, len(tokens))
	var ask []gateway.Token

	p.mu.Lock()
	for _, t := range tokens {
		if got, ok := p.price[key(chain, t.Address)]; ok && time.Now().Before(got.until) {
			out = append(out, got.price)
			continue
		}
		ask = append(ask, t)
	}
	p.mu.Unlock()

	for start := 0; start < len(ask); start += batch {
		end := start + batch
		if end > len(ask) {
			end = len(ask)
		}
		part := ask[start:end]

		addresses := make([]string, len(part))
		for i, t := range part {
			addresses[i] = strings.ToLower(t.Address)
		}

		var body map[string]struct {
			USD       float64 `json:"usd"`
			Change24h float64 `json:"usd_24h_change"`
			Volume24h float64 `json:"usd_24h_vol"`
			MarketCap float64 `json:"usd_market_cap"`
		}
		url := fmt.Sprintf(
			"%s/simple/token_price/%s?contract_addresses=%s&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true",
			priceHost, platform, strings.Join(addresses, ","))
		if err := p.get(ctx, url, &body); err != nil {
			// One batch refused is not every price unknown: keep what the
			// others gave rather than failing a screen over a rate limit.
			continue
		}

		now := time.Now()
		p.mu.Lock()
		for _, t := range part {
			got, ok := body[strings.ToLower(t.Address)]
			if !ok || got.USD <= 0 {
				continue
			}
			price := gateway.TokenPrice{
				Token:       t,
				PriceUSD:    got.USD,
				PriceChange: got.Change24h,
				Volume24h:   whole(got.Volume24h),
				MarketCap:   whole(got.MarketCap),
				UpdatedAt:   now,
			}
			p.price[key(chain, t.Address)] = quoted{price: price, until: now.Add(priceTTL)}
			out = append(out, price)
		}
		p.mu.Unlock()
	}

	return out, nil
}

func key(chain gateway.ChainID, address string) string {
	return fmt.Sprintf("%d:%s", chain, strings.ToLower(address))
}

// whole turns a figure CoinGecko states as a float into the integer the type
// carries, and returns nil for nothing — an absent volume is absent, not zero.
func whole(f float64) *big.Int {
	if f <= 0 {
		return nil
	}
	n, _ := big.NewFloat(f).Int(nil)
	return n
}
