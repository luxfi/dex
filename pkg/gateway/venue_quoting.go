package gateway

import (
	"context"
	"fmt"
	"math/big"
	"sort"
	"sync"
)

// VenueRouter fans out quote requests to multiple venues in parallel
// and returns the best result.
type VenueRouter struct {
	venues []Venue
}

// NewVenueRouter creates a VenueRouter with the given venues.
func NewVenueRouter(venues ...Venue) *VenueRouter {
	return &VenueRouter{venues: venues}
}

// Venues returns the registered venues.
func (vr *VenueRouter) Venues() []Venue {
	return vr.venues
}

// QueryAllVenues asks every venue at once and returns what they hold, best
// first, and whether the chain could be read at all.
//
// A venue distinguishes the two answers already: no liquidity is (nil, nil),
// and a chain it could not reach is (nil, err). Both were discarded here and
// came out as an empty list, so a dead endpoint and an empty pool were the same
// answer — and the surface reported one of the deepest pools in existence as a
// pair nobody holds, which sends a reader looking for liquidity instead of at
// their endpoint.
//
// The error is returned only when EVERY venue asked failed. One arm down while
// another answers is a quote, not an outage.
func (vr *VenueRouter) QueryAllVenues(ctx context.Context, req VenueQuoteRequest) ([]VenueQuote, error) {
	type result struct {
		asked bool
		quote *VenueQuote
		err   error
	}

	results := make([]result, len(vr.venues))
	var wg sync.WaitGroup

	for i, v := range vr.venues {
		wg.Add(1)
		go func(idx int, venue Venue) {
			defer wg.Done()
			if req.PreferredVenue != "" && venue.Name() != req.PreferredVenue {
				return
			}
			q, err := venue.Quote(ctx, req)
			results[idx] = result{asked: true, quote: q, err: err}
		}(i, v)
	}
	wg.Wait()

	var quotes []VenueQuote
	var asked, failed int
	var why error

	for _, res := range results {
		if !res.asked {
			continue // a preferred venue was named and this is not it
		}
		asked++
		switch {
		case res.err != nil:
			failed++
			why = res.err
		case res.quote != nil:
			quotes = append(quotes, *res.quote)
		}
	}

	// Sort by amountOut descending.
	sort.Slice(quotes, func(i, j int) bool {
		ai, _ := new(big.Int).SetString(quotes[i].AmountOut, 10)
		aj, _ := new(big.Int).SetString(quotes[j].AmountOut, 10)
		if ai == nil {
			return false
		}
		if aj == nil {
			return true
		}
		return ai.Cmp(aj) > 0
	})

	// Every arm that was asked failed, and none of them held anything: the
	// chain could not be read. One arm down while another answers is a quote.
	if failed > 0 && failed == asked {
		return nil, why
	}
	return quotes, nil
}

// ListVenueInfo returns info about all registered venues.
func (vr *VenueRouter) ListVenueInfo() []VenueInfo {
	infos := make([]VenueInfo, len(vr.venues))
	for i, v := range vr.venues {
		vType := "off_chain"
		if v.IsExecutable() {
			vType = "on_chain"
		}
		infos[i] = VenueInfo{
			Name:       v.Name(),
			Status:     "active",
			Executable: v.IsExecutable(),
			Type:       vType,
		}
	}
	return infos
}

// ComputePriceImpact estimates price impact from the spread between
// the two best venue quotes.
func ComputePriceImpact(quotes []VenueQuote) string {
	if len(quotes) < 2 {
		return "0.00"
	}
	best, _ := new(big.Int).SetString(quotes[0].AmountOut, 10)
	second, _ := new(big.Int).SetString(quotes[1].AmountOut, 10)
	if best == nil || second == nil || best.Sign() == 0 {
		return "0.00"
	}
	diff := new(big.Int).Sub(best, second)
	diff.Mul(diff, big.NewInt(10000))
	diff.Div(diff, best)
	whole := diff.Int64() / 100
	frac := diff.Int64() % 100
	return fmt.Sprintf("%d.%02d", whole, frac)
}

// ComputeExecutionPrice returns the execution price as amountOut*1e18/amountIn.
func ComputeExecutionPrice(amountInStr, amountOutStr string) string {
	amountIn, ok := new(big.Int).SetString(amountInStr, 10)
	if !ok || amountIn.Sign() == 0 {
		return "0"
	}
	amountOut, ok := new(big.Int).SetString(amountOutStr, 10)
	if !ok {
		return "0"
	}
	e18 := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	price := new(big.Int).Mul(amountOut, e18)
	price.Div(price, amountIn)
	return price.String()
}

// ParseBPS parses a basis-point string (digits only) into an int.
func ParseBPS(s string) int {
	n := 0
	for _, c := range s {
		if c >= '0' && c <= '9' {
			n = n*10 + int(c-'0')
		} else {
			return 0
		}
	}
	return n
}

// ValidateVenueQuoteRequest validates the fields of a VenueQuoteRequest.
func ValidateVenueQuoteRequest(req VenueQuoteRequest) error {
	if !isValidHexAddr(req.TokenIn) {
		return fmt.Errorf("invalid tokenIn address")
	}
	if !isValidHexAddr(req.TokenOut) {
		return fmt.Errorf("invalid tokenOut address")
	}
	if req.TokenIn == req.TokenOut {
		return fmt.Errorf("tokenIn and tokenOut must be different")
	}
	if !isPositiveDecStr(req.Amount) {
		return fmt.Errorf("amount must be a positive integer")
	}
	if req.Type != VenueQuoteTypeExactInput && req.Type != VenueQuoteTypeExactOutput {
		return fmt.Errorf("type must be EXACT_INPUT or EXACT_OUTPUT")
	}
	return nil
}

// isValidHexAddr checks that s is a 0x-prefixed 40-char hex address.
func isValidHexAddr(s string) bool {
	s = trimHexPrefix(s)
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// isPositiveDecStr returns true if s is a non-zero decimal integer string.
func isPositiveDecStr(s string) bool {
	if s == "" || s == "0" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// trimHexPrefix removes a 0x prefix from a hex string.
func trimHexPrefix(s string) string {
	if len(s) >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		return s[2:]
	}
	return s
}
