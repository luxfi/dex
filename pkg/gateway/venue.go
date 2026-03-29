package gateway

import (
	"context"
	"fmt"
	"math/big"
)

// Venue represents a liquidity source — on-chain pool or off-chain exchange.
type Venue interface {
	// Name returns the unique identifier for this venue (e.g. "v4_native", "alpaca").
	Name() string

	// Quote returns a price quote for the given request from this venue.
	// Returns nil VenueQuote with no error if the venue has no liquidity for the pair.
	Quote(ctx context.Context, req VenueQuoteRequest) (*VenueQuote, error)

	// IsExecutable returns true if this venue can produce on-chain transactions,
	// false if it only provides quote-only off-chain prices.
	IsExecutable() bool
}

// VenueQuoteRequest is the input for a single venue quote.
// Uses string-based fields to decouple from the gateway's typed QuoteRequest.
type VenueQuoteRequest struct {
	TokenIn        string
	TokenOut       string
	Amount         string // wei decimal string
	Type           string // "EXACT_INPUT" or "EXACT_OUTPUT"
	PreferredVenue string // optional: restrict to this venue name
}

// VenueQuote is a quote from a single venue.
type VenueQuote struct {
	Venue       string `json:"venue"`
	AmountOut   string `json:"amountOut"`
	Fee         string `json:"fee"`
	GasEstimate string `json:"gasEstimate"`
	Executable  bool   `json:"executable"`
}

// VenueInfo describes an available liquidity venue.
type VenueInfo struct {
	Name       string `json:"name"`
	Status     string `json:"status"`
	Executable bool   `json:"executable"`
	Type       string `json:"type"`
}

// VenueQuoteType constants.
const (
	VenueQuoteTypeExactInput  = "EXACT_INPUT"
	VenueQuoteTypeExactOutput = "EXACT_OUTPUT"
)

// V4Venue quotes from V4 pools via the DEX precompile (LXRouter at 0x9012).
// In production this would call the Liquidity EVM RPC. The current implementation
// uses a deterministic pricing model for the mock backend.
type V4Venue struct {
	rpcURL string
	// pools maps "tokenIn:tokenOut" to a pool config for deterministic quoting.
	pools map[string]*v4Pool
}

// v4Pool holds mock pool state for deterministic quoting.
type v4Pool struct {
	PoolID   string
	Fee      int // basis points
	Reserve0 *big.Int
	Reserve1 *big.Int
}

// NewV4Venue creates a V4Venue with the given RPC endpoint.
// If rpcURL is empty, the venue operates in mock mode with synthetic pools.
func NewV4Venue(rpcURL string) *V4Venue {
	v := &V4Venue{
		rpcURL: rpcURL,
		pools:  make(map[string]*v4Pool),
	}
	return v
}

// AddPool registers a pool for quoting. tokenIn and tokenOut are 0x-prefixed addresses.
func (v *V4Venue) AddPool(tokenIn, tokenOut string, poolID string, feeBPS int, reserve0, reserve1 *big.Int) {
	key := tokenIn + ":" + tokenOut
	reverseKey := tokenOut + ":" + tokenIn
	pool := &v4Pool{
		PoolID:   poolID,
		Fee:      feeBPS,
		Reserve0: new(big.Int).Set(reserve0),
		Reserve1: new(big.Int).Set(reserve1),
	}
	v.pools[key] = pool
	// Reverse direction uses the same pool with swapped reserves.
	v.pools[reverseKey] = &v4Pool{
		PoolID:   poolID,
		Fee:      feeBPS,
		Reserve0: new(big.Int).Set(reserve1),
		Reserve1: new(big.Int).Set(reserve0),
	}
}

func (v *V4Venue) Name() string { return "v4_native" }

func (v *V4Venue) IsExecutable() bool { return true }

func (v *V4Venue) Quote(_ context.Context, req VenueQuoteRequest) (*VenueQuote, error) {
	key := req.TokenIn + ":" + req.TokenOut
	pool, ok := v.pools[key]
	if !ok {
		return nil, nil // no liquidity
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	var amountOut *big.Int
	switch req.Type {
	case VenueQuoteTypeExactInput:
		amountOut = constantProductQuote(amountIn, pool.Reserve0, pool.Reserve1, pool.Fee)
	case VenueQuoteTypeExactOutput:
		// For exact output, the "amount" is the desired output.
		// Compute how much input is needed.
		amountOut = new(big.Int).Set(amountIn) // amountOut IS the requested amount
		amountIn = constantProductInverse(amountOut, pool.Reserve0, pool.Reserve1, pool.Fee)
	default:
		return nil, fmt.Errorf("unsupported quote type: %s", req.Type)
	}

	if amountOut.Sign() <= 0 || amountIn.Sign() <= 0 {
		return nil, nil // insufficient liquidity
	}

	gasEstimate := "150000" // typical V4 single-hop swap gas

	return &VenueQuote{
		Venue:       v.Name(),
		AmountOut:   amountOut.String(),
		Fee:         fmt.Sprintf("%d", pool.Fee),
		GasEstimate: gasEstimate,
		Executable:  true,
	}, nil
}

// constantProductQuote computes output from a constant-product AMM.
// amountOut = (amountIn * (10000 - feeBPS) * reserveOut) / (reserveIn * 10000 + amountIn * (10000 - feeBPS))
func constantProductQuote(amountIn, reserveIn, reserveOut *big.Int, feeBPS int) *big.Int {
	feeMultiplier := big.NewInt(int64(10000 - feeBPS))
	tenK := big.NewInt(10000)

	// numerator = amountIn * feeMultiplier * reserveOut
	num := new(big.Int).Mul(amountIn, feeMultiplier)
	num.Mul(num, reserveOut)

	// denominator = reserveIn * 10000 + amountIn * feeMultiplier
	denom := new(big.Int).Mul(reserveIn, tenK)
	inputAfterFee := new(big.Int).Mul(amountIn, feeMultiplier)
	denom.Add(denom, inputAfterFee)

	if denom.Sign() == 0 {
		return big.NewInt(0)
	}

	return num.Div(num, denom)
}

// constantProductInverse computes the input amount needed for a desired output.
// amountIn = (reserveIn * amountOut * 10000) / ((reserveOut - amountOut) * (10000 - feeBPS))
func constantProductInverse(amountOut, reserveIn, reserveOut *big.Int, feeBPS int) *big.Int {
	feeMultiplier := big.NewInt(int64(10000 - feeBPS))
	tenK := big.NewInt(10000)

	// If requested output exceeds reserves, return 0 (insufficient liquidity).
	if amountOut.Cmp(reserveOut) >= 0 {
		return big.NewInt(0)
	}

	// numerator = reserveIn * amountOut * 10000
	num := new(big.Int).Mul(reserveIn, amountOut)
	num.Mul(num, tenK)

	// denominator = (reserveOut - amountOut) * (10000 - feeBPS)
	denom := new(big.Int).Sub(reserveOut, amountOut)
	denom.Mul(denom, feeMultiplier)

	if denom.Sign() == 0 {
		return big.NewInt(0)
	}

	// Round up: (num + denom - 1) / denom
	result := new(big.Int).Add(num, new(big.Int).Sub(denom, big.NewInt(1)))
	return result.Div(result, denom)
}

// BrokerVenue quotes from external exchanges via Lux Broker.
// In production this would call the Broker HTTP API. The current implementation
// uses a configurable spread model for the mock backend.
type BrokerVenue struct {
	brokerURL string
	provider  string
	// mockPrice is the simulated mid-market price (tokenOut per tokenIn unit).
	// Set to 0 to simulate no liquidity.
	mockPrice *big.Int
	// spreadBPS is the venue's spread in basis points (applied to mock price).
	spreadBPS int
	// feeBPS is the venue's trading fee in basis points.
	feeBPS int
}

// NewBrokerVenue creates a BrokerVenue for the given provider.
func NewBrokerVenue(brokerURL, provider string) *BrokerVenue {
	return &BrokerVenue{
		brokerURL: brokerURL,
		provider:  provider,
		spreadBPS: 10, // 0.1% default spread
		feeBPS:    25, // 0.25% default fee
	}
}

// SetMockPrice sets a deterministic price for this venue (tokenOut per 1e18 tokenIn).
func (b *BrokerVenue) SetMockPrice(price *big.Int) {
	b.mockPrice = price
}

// SetFees configures spread and fee in basis points.
func (b *BrokerVenue) SetFees(spreadBPS, feeBPS int) {
	b.spreadBPS = spreadBPS
	b.feeBPS = feeBPS
}

func (b *BrokerVenue) Name() string { return b.provider }

func (b *BrokerVenue) IsExecutable() bool { return false }

func (b *BrokerVenue) Quote(_ context.Context, req VenueQuoteRequest) (*VenueQuote, error) {
	if b.mockPrice == nil || b.mockPrice.Sign() == 0 {
		return nil, nil // no liquidity
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	// amountOut = amountIn * mockPrice / 1e18, then apply spread + fee
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	amountOut := new(big.Int).Mul(amountIn, b.mockPrice)
	amountOut.Div(amountOut, e18)

	// Apply spread: reduce output by spreadBPS
	spreadDeduction := new(big.Int).Mul(amountOut, big.NewInt(int64(b.spreadBPS)))
	spreadDeduction.Div(spreadDeduction, big.NewInt(10000))
	amountOut.Sub(amountOut, spreadDeduction)

	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	return &VenueQuote{
		Venue:       b.Name(),
		AmountOut:   amountOut.String(),
		Fee:         fmt.Sprintf("%d", b.feeBPS),
		GasEstimate: "0",
		Executable:  false,
	}, nil
}
