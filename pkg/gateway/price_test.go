package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// A chain with two pools on it, at prices that are round numbers: one WLUX is
// one LUSD, and one LETH is three thousand WLUX. Everything below reads those
// two pools, so a price that comes back wrong is wrong by an amount that names
// its own cause — 0.3% high is the fee left in, 0.6% high is two of them.
func newMarketServer(extra ...Venue) *Server {
	unit := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
	million := new(big.Int).Mul(big.NewInt(1_000_000), unit)

	pools := NewV4Venue("")
	pools.AddPool(usdLux.Stable, usdLux.Hub,
		"0x0000000000000000000000000000000000000000000000000000000000000001",
		30, million, million)
	pools.AddPool(usdLux.Hub, luxToken("LETH"),
		"0x0000000000000000000000000000000000000000000000000000000000000002",
		30, new(big.Int).Mul(big.NewInt(3_000_000), unit), new(big.Int).Mul(big.NewInt(1_000), unit))

	return NewServer(
		NewRouter(NewRegistry(), true),
		DefaultServerConfig(),
		WithChainVenues(&ChainRouters{
			byChain: map[ChainID]*VenueRouter{ChainIDLux: NewVenueRouter(append([]Venue{pools}, extra...)...)},
			usd:     map[ChainID]Numeraires{ChainIDLux: usdLux},
		}),
	)
}

// luxToken is one of 96369's tokens by symbol, from the list the binary
// carries.
func luxToken(symbol string) string {
	for _, t := range NewCatalog([]ChainID{ChainIDLux}).List(ChainIDLux, "", 100) {
		if t.Symbol == symbol {
			return t.Address
		}
	}
	panic("96369 lists no " + symbol)
}

func markOfSymbol(t *testing.T, s *Server, symbol string) mark {
	t.Helper()
	tok, ok := s.catalog.Token(ChainIDLux, luxToken(symbol))
	if !ok {
		t.Fatalf("the catalog lists no %s", symbol)
	}
	return s.markOf(context.Background(), ChainIDLux, tok)
}

func near(t *testing.T, what string, got *big.Rat, want, tolerance float64) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s has no price at all", what)
	}
	f, _ := got.Float64()
	if off := math.Abs(f-want) / want; off > tolerance {
		t.Errorf("%s is %.8f, want %.8f — off by %.4f%%, tolerance %.4f%%", what, f, want, 100*off, 100*tolerance)
	}
}

// The dollar this chain prices in is worth a dollar, by construction, and
// saying so costs nothing. Read through a pool instead it came back $1.0061 —
// two fees deep — which is a wrong number on the one row nobody would check.
func TestTheDollarIsTheDollar(t *testing.T) {
	got := markOfSymbol(t, newMarketServer(), "LUSD")
	if got.usd == nil || got.usd.Cmp(big.NewRat(1, 1)) != 0 {
		t.Fatalf("LUSD is %v", got.usd)
	}
	if got.via != "LUSD" {
		t.Errorf("LUSD is priced via %q", got.via)
	}
}

// A pool hands back the mid price LESS its tier, so a reading with the fee
// still in it is high by exactly that tier. On a 0.3% pool that is $1.0031 for
// a token worth $1.00 — visible on the screen, and wrong.
//
// This is also the check that every arm states a fee in one unit. The V2 arm
// used to answer 30 where the V3 arm answered 3000 for the same three tenths
// of a percent; read in the wrong unit the correction becomes 0.0003% and this
// comes back at $1.0031.
func TestTheFeeComesBackOutOfThePrice(t *testing.T) {
	near(t, "WLUX", markOfSymbol(t, newMarketServer(), "WLUX").usd, 1.0, 0.001)
}

// Most of a chain's liquidity is against its own token, not against the dollar
// — on Ethereum, WETH. So a token with no dollar pool is read against the hub,
// and the hub against the dollar. LETH has no LUSD pool at all here.
func TestATokenWithNoDollarPoolIsReadThroughTheHub(t *testing.T) {
	got := markOfSymbol(t, newMarketServer(), "LETH")
	near(t, "LETH", got.usd, 3000, 0.005)
	if got.via != "WLUX" {
		t.Errorf("LETH is priced via %q, want WLUX", got.via)
	}
	if got.venue != VenueNameNative {
		t.Errorf("LETH was priced by %q", got.venue)
	}
}

// A token no pool here holds has no price. Not zero — absent. gLUX is in no
// pool on 96369, and a market screen drawing $0.00 beside it states something
// false about a token that simply has not traded.
func TestATokenInNoPoolHasNoPrice(t *testing.T) {
	if got := markOfSymbol(t, newMarketServer(), "gLUX"); got.usd != nil {
		t.Fatalf("gLUX priced at %v", got.usd)
	}
}

// countingVenue is an arm that says how often it was asked.
type countingVenue struct {
	inner Venue
	asked atomic.Int64
}

func (c *countingVenue) Name() string       { return c.inner.Name() }
func (c *countingVenue) IsExecutable() bool { return c.inner.IsExecutable() }
func (c *countingVenue) Quote(ctx context.Context, req VenueQuoteRequest) (*VenueQuote, error) {
	c.asked.Add(1)
	return c.inner.Quote(ctx, req)
}
func (c *countingVenue) Swap(o SwapOrder) (*UnsignedTxResponse, error) { return c.inner.Swap(o) }

// Two hundred rows opening at once must be one reading per token, not one per
// token per reader. Every price is a fan-out of eth_calls against an endpoint
// we hold no account with, and a page reloaded three times while the first
// read is still in flight is how that endpoint stops answering.
func TestOneColdPriceIsReadOnce(t *testing.T) {
	counter := &countingVenue{inner: NewV4Venue("")}
	s := newMarketServer(counter)
	tok, _ := s.catalog.Token(ChainIDLux, luxToken("WLUX"))

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.markOf(context.Background(), ChainIDLux, tok)
		}()
	}
	wg.Wait()

	// One reading of WLUX against the dollar. Thirty-two readers, one fan-out.
	if n := counter.asked.Load(); n != 1 {
		t.Errorf("32 readers asked the chain %d times, want 1", n)
	}
	// And the next reader gets the kept answer without asking again.
	s.markOf(context.Background(), ChainIDLux, tok)
	if n := counter.asked.Load(); n != 1 {
		t.Errorf("a kept price was read again: %d", n)
	}
}

// deadVenue is an arm on a chain that cannot be reached.
type deadVenue struct{}

func (deadVenue) Name() string       { return "dead" }
func (deadVenue) IsExecutable() bool { return true }
func (deadVenue) Quote(context.Context, VenueQuoteRequest) (*VenueQuote, error) {
	return nil, fmt.Errorf("%w: dial tcp: connection refused", ErrUnreachable)
}
func (deadVenue) Swap(SwapOrder) (*UnsignedTxResponse, error) { return nil, nil }

// A chain nobody can reach is not a chain with no pools. Answered as a page of
// dashes it sends a reader looking for liquidity instead of at their endpoint.
func TestAnUnreadableChainSaysSo(t *testing.T) {
	s := NewServer(
		NewRouter(NewRegistry(), true),
		DefaultServerConfig(),
		WithChainVenues(&ChainRouters{
			byChain: map[ChainID]*VenueRouter{ChainIDLux: NewVenueRouter(deadVenue{})},
			usd:     map[ChainID]Numeraires{ChainIDLux: usdLux},
		}),
	)

	w := ask(s, http.MethodGet, "/v1/trade/price?chainId=96369&token="+luxToken("LETH"), nil)
	if w.Code != http.StatusBadGateway {
		t.Fatalf("an unreachable chain answered %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "could not be read") {
		t.Errorf("it says: %s", w.Body.String())
	}
}

// /v1/trade/price, as a screen asks it: several tokens at once, answered in
// the order asked, with the ones nothing holds coming back without a price.
func TestPriceAnswersInTheOrderAsked(t *testing.T) {
	s := newMarketServer()
	asked := []string{luxToken("LETH"), luxToken("gLUX"), luxToken("LUSD"), luxToken("WLUX")}
	w := ask(s, http.MethodGet, "/v1/trade/price?chainId=96369&token="+strings.Join(asked, ","), nil)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}

	var got struct{ Data []Price }
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != len(asked) {
		t.Fatalf("asked for %d, got %d", len(asked), len(got.Data))
	}
	for i, address := range asked {
		if !strings.EqualFold(got.Data[i].Address, address) {
			t.Fatalf("row %d is %s, asked %s", i, got.Data[i].Address, address)
		}
	}
	if got.Data[1].USD != nil {
		t.Errorf("gLUX, in no pool, came back at %v", *got.Data[1].USD)
	}
	// Absent means absent: no venue and no numéraire either, because a number
	// nobody can trace is a number nobody can check.
	if got.Data[1].Venue != "" || got.Data[1].Via != "" {
		t.Errorf("gLUX has no price and names %q via %q", got.Data[1].Venue, got.Data[1].Via)
	}
	if got.Data[1].Symbol != "gLUX" || got.Data[1].Decimals != 18 {
		t.Errorf("an unpriced token lost its identity: %+v", got.Data[1])
	}
	if got.Data[3].USD == nil || math.Abs(*got.Data[3].USD-1) > 0.001 {
		t.Errorf("WLUX came back %v", got.Data[3].USD)
	}

	// And the JSON says nothing at all about a price that is not there, rather
	// than saying zero.
	if strings.Contains(w.Body.String(), `"priceUSD":0`) {
		t.Errorf("an absent price was written as zero: %s", w.Body.String())
	}
}

func TestPriceRefusesWhatItCannotAnswer(t *testing.T) {
	s := newMarketServer()
	many := make([]string, pricesAtOnce+1)
	for i := range many {
		many[i] = luxToken("WLUX")
	}

	for _, c := range []struct {
		why  string
		path string
		want int
	}{
		{"no chain", "/v1/trade/price?token=" + luxToken("WLUX"), http.StatusBadRequest},
		{"no token", "/v1/trade/price?chainId=96369", http.StatusBadRequest},
		{"not an address", "/v1/trade/price?chainId=96369&token=WLUX", http.StatusBadRequest},
		{"more than it will read at once", "/v1/trade/price?chainId=96369&token=" + strings.Join(many, ","), http.StatusBadRequest},
		{"a chain with no venues", "/v1/trade/price?chainId=1&token=" + usdEthereum.Hub, http.StatusNotFound},
		{"a token this chain does not list", "/v1/trade/price?chainId=96369&token=0x0000000000000000000000000000000000000001", http.StatusNotFound},
	} {
		if w := ask(s, http.MethodGet, c.path, nil); w.Code != c.want {
			t.Errorf("%s: answered %d, want %d: %s", c.why, w.Code, c.want, w.Body.String())
		}
	}
}

func TestTokensAnswersTheListItCarries(t *testing.T) {
	s := newMarketServer()

	var rows struct{ Data []Token }
	w := ask(s, http.MethodGet, "/v1/trade/tokens?chainId=96369", nil)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	if err := json.Unmarshal(w.Body.Bytes(), &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows.Data) != 14 {
		t.Errorf("96369 lists %d tokens, want 14", len(rows.Data))
	}
	if rows.Data[0].Symbol != "WLUX" {
		t.Errorf("96369 leads with %s", rows.Data[0].Symbol)
	}
	// Nothing here points a browser at a third party.
	if strings.Contains(w.Body.String(), "logo") || strings.Contains(w.Body.String(), "http") {
		t.Errorf("a token row carries a URL: %s", w.Body.String())
	}

	// A chain this deployment cannot quote lists nothing, so a screen cannot
	// draw a market it has no way to price.
	w = ask(s, http.MethodGet, "/v1/trade/tokens?chainId=1", nil)
	if err := json.Unmarshal(w.Body.Bytes(), &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows.Data) != 0 {
		t.Errorf("Ethereum is not quoted here and listed %d tokens", len(rows.Data))
	}

	if w := ask(s, http.MethodGet, "/v1/trade/tokens?chainId=96369&limit=nope", nil); w.Code != http.StatusBadRequest {
		t.Errorf("a limit that is not a count answered %d", w.Code)
	}
}

// The arithmetic, on its own, at numbers whose answer can be read off the page.
func TestTheArithmeticOfAReading(t *testing.T) {
	e18 := pow10(18)
	e6 := pow10(6)

	// A hundred dollars of a six-decimal dollar is 100000000 of its units;
	// of a token worth $2500, it is 0.04 of one.
	if got := probeAmount(big.NewRat(1, 1), 6); got.Cmp(big.NewInt(100_000_000)) != 0 {
		t.Errorf("a hundred dollars of USDC is %s units", got)
	}
	if got := probeAmount(big.NewRat(2500, 1), 18); got.Cmp(new(big.Int).Div(e18, big.NewInt(25))) != 0 {
		t.Errorf("a hundred dollars of a $2500 token is %s units", got)
	}

	// A hundred USDC bought 0.04 of an 18-decimal token through a 0.3% pool.
	// The pool kept 0.3%, so the token is worth 2500 × 0.997 = 2492.50.
	price := unitPrice(new(big.Int).Mul(big.NewInt(100), e6), 6, big.NewRat(1, 1),
		new(big.Int).Div(e18, big.NewInt(25)), 18, 3000)
	near(t, "the reading", price, 2492.50, 1e-12)

	// Through a pool that charges nothing, it is worth exactly 2500.
	price = unitPrice(new(big.Int).Mul(big.NewInt(100), e6), 6, big.NewRat(1, 1),
		new(big.Int).Div(e18, big.NewInt(25)), 18, 0)
	near(t, "a free reading", price, 2500, 1e-12)
}

// A chain that cannot be reached says so, including through the arm that reads
// our own precompiles.
//
// Every arm here distinguishes a reverting contract from an endpoint that never
// answered, and the router reports a chain it could not read only when EVERY
// arm on that chain failed. So a single arm answering "no liquidity" to an
// endpoint that never replied is enough on its own to hide it, and the one
// reading our own chain is the arm that would do the hiding.
func TestOurOwnChainDoesNotHideItsOutage(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad gateway", http.StatusBadGateway)
	}))
	defer dead.Close()

	native := NewNativeDEXVenue(NativeDEXConfig{RPCURL: dead.URL})
	_, err := native.Quote(context.Background(), VenueQuoteRequest{
		TokenIn: usdLux.Hub, TokenOut: usdLux.Stable, Amount: "1", Type: VenueQuoteTypeExactInput,
	})
	if !errors.Is(err, ErrUnreachable) {
		t.Fatalf("a dead endpoint answered %v, want an unreachable chain", err)
	}

	// And a whole chain built on that arm reports the outage rather than an
	// empty market.
	s := NewServer(
		NewRouter(NewRegistry(), true),
		DefaultServerConfig(),
		WithChainVenues(&ChainRouters{
			byChain: map[ChainID]*VenueRouter{ChainIDLux: NewVenueRouter(native)},
			usd:     map[ChainID]Numeraires{ChainIDLux: usdLux},
		}),
	)
	if w := ask(s, http.MethodGet, "/v1/trade/price?chainId=96369&token="+luxToken("LETH"), nil); w.Code != http.StatusBadGateway {
		t.Errorf("a price on a dead chain answered %d: %s", w.Code, w.Body.String())
	}
	w := ask(s, http.MethodPost, "/v1/trade/quote", map[string]any{
		"tokenIn": usdLux.Hub, "tokenOut": usdLux.Stable, "chainId": 96369,
		"amount": "1000000000000000000", "isExactIn": true,
	})
	if w.Code != http.StatusBadGateway {
		t.Errorf("a quote on a dead chain answered %d: %s", w.Code, w.Body.String())
	}
}
