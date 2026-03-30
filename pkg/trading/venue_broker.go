package trading

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"
)

// SymbolRegistry maps token-pair addresses to broker trading symbols.
// It is safe for concurrent use.
type SymbolRegistry struct {
	mu sync.RWMutex
	m  map[string]string
}

// NewSymbolRegistry creates an empty symbol registry.
func NewSymbolRegistry() *SymbolRegistry {
	return &SymbolRegistry{m: make(map[string]string)}
}

// Register maps a token pair to a broker symbol (e.g. "BTC-USD").
// Addresses are lowercased for case-insensitive lookup.
func (r *SymbolRegistry) Register(tokenIn, tokenOut, symbol string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[strings.ToLower(tokenIn)+":"+strings.ToLower(tokenOut)] = symbol
}

// Lookup returns the broker symbol for a token pair, checking both directions.
// Address comparison is case-insensitive (EVM addresses are hex, case does not matter).
func (r *SymbolRegistry) Lookup(tokenIn, tokenOut string) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	inLower, outLower := strings.ToLower(tokenIn), strings.ToLower(tokenOut)
	if sym, ok := r.m[inLower+":"+outLower]; ok {
		return sym, true
	}
	if sym, ok := r.m[outLower+":"+inLower]; ok {
		return sym, true
	}
	return "", false
}

// Delete removes a token pair mapping. Used in tests for cleanup.
func (r *SymbolRegistry) Delete(tokenIn, tokenOut string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.m, strings.ToLower(tokenIn)+":"+strings.ToLower(tokenOut))
}

// DefaultRegistry is the package-level registry used when venues are
// constructed without an explicit registry. Retained for backward compat.
var DefaultRegistry = NewSymbolRegistry()

// BrokerHTTPVenue queries the Lux Broker REST API for off-chain CEX quotes.
// The broker federates across 16 providers: Alpaca, IBKR, Binance, Kraken,
// Coinbase, Gemini, SFOX, FalconX, Fireblocks, BitGo, Circle, CurrencyCloud,
// LMAX, Tradier, Polygon, Finix.
type BrokerHTTPVenue struct {
	brokerURL  string
	provider   string
	apiKey     string
	httpClient *http.Client
	registry   *SymbolRegistry
}

type BrokerHTTPConfig struct {
	BrokerURL string
	Provider  string
	APIKey    string
	Registry  *SymbolRegistry
}

func NewBrokerHTTPVenue(cfg BrokerHTTPConfig) *BrokerHTTPVenue {
	name := cfg.Provider
	if name == "" {
		name = "broker"
	}
	reg := cfg.Registry
	if reg == nil {
		reg = DefaultRegistry
	}
	return &BrokerHTTPVenue{
		brokerURL: cfg.BrokerURL,
		provider:  name,
		apiKey:    cfg.APIKey,
		registry:  reg,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

func (v *BrokerHTTPVenue) Name() string      { return v.provider }
func (v *BrokerHTTPVenue) IsExecutable() bool { return false }

func (v *BrokerHTTPVenue) Quote(ctx context.Context, req QuoteRequest) (*VenueQuote, error) {
	symbol, ok := v.registry.Lookup(req.TokenIn, req.TokenOut)
	if !ok {
		return nil, nil
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	url := fmt.Sprintf("%s/v1/market/%s/snapshot/%s", v.brokerURL, v.provider, symbol)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	if v.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+v.apiKey)
	}

	resp, err := v.httpClient.Do(httpReq)
	if err != nil {
		return nil, nil // broker unreachable
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, nil
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, nil
	}

	var snapshot brokerSnapshot
	if err := json.Unmarshal(body, &snapshot); err != nil {
		return nil, nil
	}

	if snapshot.AskPrice <= 0 {
		return nil, nil
	}

	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	priceInt := floatToBigInt(snapshot.AskPrice, 18)
	amountOut := new(big.Int).Mul(amountIn, priceInt)
	amountOut.Div(amountOut, e18)

	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	feeBPS := "0"
	if snapshot.AskPrice > 0 && snapshot.BidPrice > 0 {
		spread := (snapshot.AskPrice - snapshot.BidPrice) / snapshot.AskPrice * 10000
		feeBPS = fmt.Sprintf("%.0f", spread)
	}

	return &VenueQuote{
		Venue:       v.provider,
		AmountOut:   amountOut.String(),
		Fee:         feeBPS,
		GasEstimate: "0",
		Executable:  false,
	}, nil
}

type brokerSnapshot struct {
	Symbol   string  `json:"symbol"`
	BidPrice float64 `json:"bid_price"`
	AskPrice float64 `json:"ask_price"`
}

// BrokerSORVenue uses the broker's Smart Order Router across all 16 providers.
type BrokerSORVenue struct {
	brokerURL  string
	apiKey     string
	httpClient *http.Client
	registry   *SymbolRegistry
}

// NewBrokerSORVenue creates a SOR venue. If registry is nil, DefaultRegistry is used.
func NewBrokerSORVenue(brokerURL, apiKey string, registry *SymbolRegistry) *BrokerSORVenue {
	if registry == nil {
		registry = DefaultRegistry
	}
	return &BrokerSORVenue{
		brokerURL: brokerURL,
		apiKey:    apiKey,
		registry:  registry,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (v *BrokerSORVenue) Name() string      { return "broker_sor" }
func (v *BrokerSORVenue) IsExecutable() bool { return false }

func (v *BrokerSORVenue) Quote(ctx context.Context, req QuoteRequest) (*VenueQuote, error) {
	symbol, ok := v.registry.Lookup(req.TokenIn, req.TokenOut)
	if !ok {
		return nil, nil
	}

	url := fmt.Sprintf("%s/v1/route/%s", v.brokerURL, symbol)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	if v.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+v.apiKey)
	}

	resp, err := v.httpClient.Do(httpReq)
	if err != nil {
		return nil, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, nil
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, nil
	}

	var route brokerRouteResponse
	if err := json.Unmarshal(body, &route); err != nil {
		return nil, nil
	}

	if route.BestAsk.Price <= 0 {
		return nil, nil
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, nil
	}

	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	priceInt := floatToBigInt(route.BestAsk.Price, 18)
	amountOut := new(big.Int).Mul(amountIn, priceInt)
	amountOut.Div(amountOut, e18)

	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	feeBPS := "0"
	if route.SpreadBPS > 0 {
		feeBPS = fmt.Sprintf("%.0f", route.SpreadBPS)
	}

	return &VenueQuote{
		Venue:       "broker_sor:" + route.BestAsk.Provider,
		AmountOut:   amountOut.String(),
		Fee:         feeBPS,
		GasEstimate: "0",
		Executable:  false,
	}, nil
}

type brokerRouteResponse struct {
	Symbol  string `json:"symbol"`
	BestBid struct {
		Price    float64 `json:"price"`
		Provider string  `json:"provider"`
	} `json:"best_bid"`
	BestAsk struct {
		Price    float64 `json:"price"`
		Provider string  `json:"provider"`
	} `json:"best_ask"`
	SpreadBPS float64 `json:"spread_bps"`
}

// RegisterSymbol maps a token pair to a broker symbol on the DefaultRegistry.
// Convenience wrapper for backward compatibility.
func RegisterSymbol(tokenIn, tokenOut, symbol string) {
	DefaultRegistry.Register(tokenIn, tokenOut, symbol)
}

func floatToBigInt(f float64, decimals int) *big.Int {
	bf := new(big.Float).SetFloat64(f)
	scale := new(big.Float).SetInt(new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(decimals)), nil))
	bf.Mul(bf, scale)
	result, _ := bf.Int(nil)
	return result
}
