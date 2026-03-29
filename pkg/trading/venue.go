package trading

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
	Quote(ctx context.Context, req QuoteRequest) (*VenueQuote, error)

	// IsExecutable returns true if this venue can produce on-chain transactions,
	// false if it only provides quote-only off-chain prices.
	IsExecutable() bool
}

// --- Mock venues for testing ---

// V4Venue is a mock V4 AMM venue using constant-product math.
// For testing and development. Production uses NativeDEXVenue.
type V4Venue struct {
	rpcURL string
	pools  map[string]*v4Pool
}

type v4Pool struct {
	PoolID   string
	Fee      int
	Reserve0 *big.Int
	Reserve1 *big.Int
}

func NewV4Venue(rpcURL string) *V4Venue {
	return &V4Venue{
		rpcURL: rpcURL,
		pools:  make(map[string]*v4Pool),
	}
}

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
	v.pools[reverseKey] = &v4Pool{
		PoolID:   poolID,
		Fee:      feeBPS,
		Reserve0: new(big.Int).Set(reserve1),
		Reserve1: new(big.Int).Set(reserve0),
	}
}

func (v *V4Venue) Name() string        { return "v4_native" }
func (v *V4Venue) IsExecutable() bool   { return true }

func (v *V4Venue) Quote(_ context.Context, req QuoteRequest) (*VenueQuote, error) {
	key := req.TokenIn + ":" + req.TokenOut
	pool, ok := v.pools[key]
	if !ok {
		return nil, nil
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	var amountOut *big.Int
	switch req.Type {
	case QuoteTypeExactInput:
		amountOut = constantProductQuote(amountIn, pool.Reserve0, pool.Reserve1, pool.Fee)
	case QuoteTypeExactOutput:
		amountOut = new(big.Int).Set(amountIn)
		amountIn = constantProductInverse(amountOut, pool.Reserve0, pool.Reserve1, pool.Fee)
	default:
		return nil, fmt.Errorf("unsupported quote type: %s", req.Type)
	}

	if amountOut.Sign() <= 0 || amountIn.Sign() <= 0 {
		return nil, nil
	}

	return &VenueQuote{
		Venue:       v.Name(),
		AmountOut:   amountOut.String(),
		Fee:         fmt.Sprintf("%d", pool.Fee),
		GasEstimate: "150000",
		Executable:  true,
	}, nil
}

func constantProductQuote(amountIn, reserveIn, reserveOut *big.Int, feeBPS int) *big.Int {
	feeMultiplier := big.NewInt(int64(10000 - feeBPS))
	tenK := big.NewInt(10000)

	num := new(big.Int).Mul(amountIn, feeMultiplier)
	num.Mul(num, reserveOut)

	denom := new(big.Int).Mul(reserveIn, tenK)
	inputAfterFee := new(big.Int).Mul(amountIn, feeMultiplier)
	denom.Add(denom, inputAfterFee)

	if denom.Sign() == 0 {
		return big.NewInt(0)
	}

	return num.Div(num, denom)
}

func constantProductInverse(amountOut, reserveIn, reserveOut *big.Int, feeBPS int) *big.Int {
	feeMultiplier := big.NewInt(int64(10000 - feeBPS))
	tenK := big.NewInt(10000)

	if amountOut.Cmp(reserveOut) >= 0 {
		return big.NewInt(0)
	}

	num := new(big.Int).Mul(reserveIn, amountOut)
	num.Mul(num, tenK)

	denom := new(big.Int).Sub(reserveOut, amountOut)
	denom.Mul(denom, feeMultiplier)

	if denom.Sign() == 0 {
		return big.NewInt(0)
	}

	result := new(big.Int).Add(num, new(big.Int).Sub(denom, big.NewInt(1)))
	return result.Div(result, denom)
}

// BrokerVenue is a mock off-chain broker venue for testing.
// Production uses BrokerHTTPVenue.
type BrokerVenue struct {
	brokerURL string
	provider  string
	mockPrice *big.Int
	spreadBPS int
	feeBPS    int
}

func NewBrokerVenue(brokerURL, provider string) *BrokerVenue {
	return &BrokerVenue{
		brokerURL: brokerURL,
		provider:  provider,
		spreadBPS: 10,
		feeBPS:    25,
	}
}

func (b *BrokerVenue) SetMockPrice(price *big.Int) { b.mockPrice = price }

func (b *BrokerVenue) SetFees(spreadBPS, feeBPS int) {
	b.spreadBPS = spreadBPS
	b.feeBPS = feeBPS
}

func (b *BrokerVenue) Name() string        { return b.provider }
func (b *BrokerVenue) IsExecutable() bool   { return false }

func (b *BrokerVenue) Quote(_ context.Context, req QuoteRequest) (*VenueQuote, error) {
	if b.mockPrice == nil || b.mockPrice.Sign() == 0 {
		return nil, nil
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	amountOut := new(big.Int).Mul(amountIn, b.mockPrice)
	amountOut.Div(amountOut, e18)

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
