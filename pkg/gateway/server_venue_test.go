package gateway

import (
	"bytes"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The deployment this exercises is the ordinary one: venues over pools, and no
// hosted provider at all. The hosted trade API answers ACCESS_DENIED without a
// contract, so a provider that always refuses is worse than none — which means
// every path a client uses has to be correct with an EMPTY registry, and three
// of them were not.
func newPoolOnlyServer() *Server {
	unit := new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

	pools := NewV4Venue("")
	pools.AddPool(testLUSD, testWETH,
		"0x0000000000000000000000000000000000000000000000000000000000000002",
		30,
		new(big.Int).Mul(big.NewInt(5_000_000), unit),
		new(big.Int).Mul(big.NewInt(1666), unit))

	return NewServer(
		NewRouter(NewRegistry(), true),
		DefaultServerConfig(),
		WithChainVenues(&ChainRouters{byChain: map[ChainID]*VenueRouter{
			ChainIDLux: NewVenueRouter(pools),
		}}),
	)
}

func ask(s *Server, method, path string, body interface{}) *httptest.ResponseRecorder {
	var r *http.Request
	if body == nil {
		r = httptest.NewRequest(method, path, nil)
	} else {
		b, _ := json.Marshal(body)
		r = httptest.NewRequest(method, path, bytes.NewReader(b))
		r.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	s.mux.ServeHTTP(w, r)
	return w
}

// /healthz reported the hosted provider registry and nothing else, so the
// ordinary deployment answered 503 forever. Anything probing it to decide
// whether a replica can serve then holds every replica out of its own Service
// while every quote it would have served works.
func TestHealthIsTheVenues(t *testing.T) {
	w := ask(newPoolOnlyServer(), http.MethodGet, "/healthz", nil)
	if w.Code != http.StatusOK {
		t.Fatalf("healthz with venues and no provider: got %d, want 200: %s", w.Code, w.Body.String())
	}

	var got struct {
		Data struct {
			Status string    `json:"status"`
			Chains []ChainID `json:"chains"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("body: %v", err)
	}
	if got.Data.Status != "healthy" {
		t.Errorf("status = %q, want healthy", got.Data.Status)
	}
	if len(got.Data.Chains) != 1 || got.Data.Chains[0] != ChainIDLux {
		t.Errorf("chains = %v, want [%d]", got.Data.Chains, ChainIDLux)
	}
}

// /v1/quotes read the provider registry alone, so a chain we settle ourselves
// answered "no providers available" — an error about an upstream account —
// while /v1/quote priced the very same pair from the very same pools.
func TestQuotesReadThePools(t *testing.T) {
	s := newPoolOnlyServer()
	asked := quoteRequest{
		ChainID:   uint64(ChainIDLux),
		TokenIn:   testWETH,
		TokenOut:  testLUSD,
		Amount:    "1000000000000000000",
		IsExactIn: true,
	}

	one := ask(s, http.MethodPost, "/v1/trade/quote", asked)
	if one.Code != http.StatusOK {
		t.Fatalf("/v1/trade/quote: got %d, want 200: %s", one.Code, one.Body.String())
	}
	all := ask(s, http.MethodPost, "/v1/trade/quotes", asked)
	if all.Code != http.StatusOK {
		t.Fatalf("/v1/trade/quotes: got %d, want 200: %s", all.Code, all.Body.String())
	}

	var best struct {
		Data SwapQuote `json:"data"`
	}
	var every struct {
		Data []SwapQuote `json:"data"`
	}
	if err := json.Unmarshal(one.Body.Bytes(), &best); err != nil {
		t.Fatalf("quote body: %v", err)
	}
	if err := json.Unmarshal(all.Body.Bytes(), &every); err != nil {
		t.Fatalf("quotes body: %v", err)
	}
	if len(every.Data) == 0 {
		t.Fatal("/v1/trade/quotes returned nothing where /v1/quote returned a price")
	}
	// The best of every is the one, or the two disagree about one pool.
	if every.Data[0].TokenOut.Amount.Cmp(best.Data.TokenOut.Amount) != 0 {
		t.Errorf("head of /v1/quotes = %s, /v1/quote = %s",
			every.Data[0].TokenOut.Amount, best.Data.TokenOut.Amount)
	}
	if every.Data[0].ProviderName == "" {
		t.Error("a quote that does not name the venue that answered it")
	}
}

// A pair no pool holds is 404 and says so about the venues, never 500 about a
// registry the caller never asked for.
func TestUnheldPairIsNotAnUpstreamError(t *testing.T) {
	w := ask(newPoolOnlyServer(), http.MethodPost, "/v1/trade/quotes", quoteRequest{
		ChainID:   uint64(ChainIDLux),
		TokenIn:   testWETH,
		TokenOut:  testOther,
		Amount:    "1000000000000000000",
		IsExactIn: true,
	})
	if w.Code != http.StatusNotFound {
		t.Fatalf("got %d, want 404: %s", w.Code, w.Body.String())
	}
	if bytes.Contains(w.Body.Bytes(), []byte("no providers available")) {
		t.Errorf("a chain we settle ourselves reported an upstream registry: %s", w.Body.String())
	}
}

// The pause/freeze surface authorised on `X-User-Role: admin` — a header the
// caller writes — and drove state nothing read. It is gone; nothing may bring
// back a route that answers a header.
func TestNoSurfaceAnswersAHeaderForARole(t *testing.T) {
	s := newPoolOnlyServer()
	for _, path := range []string{
		"/v1/admin/status",
		"/v1/admin/pause",
		"/v1/admin/resume",
		"/v1/admin/pool/pause",
		"/v1/admin/pool/resume",
		"/v1/admin/pool/freeze",
		"/v1/admin/pool/status",
	} {
		for _, method := range []string{http.MethodGet, http.MethodPost} {
			r := httptest.NewRequest(method, path, bytes.NewReader([]byte(`{"poolId":"0x1","reason":"x"}`)))
			r.Header.Set("X-User-Role", "admin")
			r.Header.Set("X-User-ID", "anyone")
			w := httptest.NewRecorder()
			s.mux.ServeHTTP(w, r)
			if w.Code != http.StatusNotFound {
				t.Errorf("%s %s answered %d: %s", method, path, w.Code, w.Body.String())
			}
		}
	}
}

// /v1/trade/venues is the question a screen asks before it asks anything else:
// what can you price. Without it a client learns a chain is unquotable by
// quoting on it and reading an error, which is a worse way to find out.
func TestVenuesNameTheChainsAndTheirArms(t *testing.T) {
	s := NewServer(
		NewRouter(NewRegistry(), true),
		DefaultServerConfig(),
		WithChainVenues(NewChainRouters(map[ChainID]ChainVenues{
			ChainIDLux:      {RPC: "http://chain.invalid", Native: true, V3Quoter: testOther},
			ChainIDEthereum: {RPC: "http://eth.invalid", V2Router: testOther, V3Quoter: testOther},
		})))

	w := ask(s, http.MethodGet, "/v1/trade/venues", nil)
	if w.Code != http.StatusOK {
		t.Fatalf("got %d: %s", w.Code, w.Body.String())
	}

	var got struct {
		Data []ChainVenueInfo `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("body: %v", err)
	}
	if len(got.Data) != 2 {
		t.Fatalf("listed %d chains, want 2: %s", len(got.Data), w.Body.String())
	}
	// Ordered, because a listing that permutes itself between two reads of one
	// deployment reads as a deployment that changed.
	if got.Data[0].ChainID != ChainIDEthereum || got.Data[1].ChainID != ChainIDLux {
		t.Errorf("chains = %d, %d — want them in order", got.Data[0].ChainID, got.Data[1].ChainID)
	}
	// Only our own chain has the precompile arm, and the listing says which.
	if got.Data[0].Native {
		t.Error("chain 1 reported as a chain we settle")
	}
	if !got.Data[1].Native {
		t.Error("our own chain reported as one we only read")
	}
	if len(got.Data[0].Venues) != 2 || len(got.Data[1].Venues) != 2 {
		t.Errorf("arms = %d and %d, want 2 each: %s",
			len(got.Data[0].Venues), len(got.Data[1].Venues), w.Body.String())
	}

	one := ask(s, http.MethodGet, "/v1/trade/venues?chainId=1", nil)
	if err := json.Unmarshal(one.Body.Bytes(), &got); err != nil {
		t.Fatalf("body: %v", err)
	}
	if len(got.Data) != 1 || got.Data[0].ChainID != ChainIDEthereum {
		t.Errorf("?chainId=1 answered %s", one.Body.String())
	}

	none := ask(s, http.MethodGet, "/v1/trade/venues?chainId=999", nil)
	if none.Code != http.StatusNotFound {
		t.Errorf("a chain nothing here reads answered %d, want 404", none.Code)
	}
}

// One surface, one name. The same operations were answering at /v1/* and again
// at /trading/* in a second shape, while a published openapi.yaml described a
// third set no binary served. A client that can reach an operation by two
// names will reach it by both, and the two will drift.
func TestTheSurfaceHasOneName(t *testing.T) {
	s := newPoolOnlyServer()
	for _, gone := range []string{
		"/v1/quote", "/v1/quotes", "/v1/swap", "/v1/route", "/v1/venues",
		"/v1/pools", "/v1/positions", "/v1/price", "/v1/prices", "/v1/tokens",
		"/v1/stats", "/v1/order", "/v1/position", "/v1/approval/check",
		"/v1/permit2/check",
		"/trading/quote", "/trading/swap", "/trading/check_approval",
		"/trading/order", "/trading/orders", "/trading/swaps", "/trading/send",
		"/trading/swappable_tokens", "/trading/lp/create",
	} {
		for _, method := range []string{http.MethodGet, http.MethodPost} {
			w := ask(s, method, gone, map[string]any{})
			if w.Code != http.StatusNotFound {
				t.Errorf("%s %s still answers %d — the surface is at %s", method, gone, w.Code, tradePrefix)
			}
		}
	}
}

// Nothing here invents a number.
//
// /v1/history/prices and /v1/history/tvl returned a random walk seeded from
// hardcoded base prices — LUX at 2.47, WBTC at 96420, 12.5M of TVL on 96369 —
// and both were routed to the public internet. A screen drawing that chart
// shows a market that does not exist, and it is indistinguishable on the wire
// from one that does. /v1/tokens/search answered 501 on a TODO, which is a
// route that lies about the surface in the other direction.
func TestNothingHereInventsANumber(t *testing.T) {
	s := newPoolOnlyServer()
	for _, gone := range []string{
		"/v1/trade/history/prices?symbol=LUX&period=30d",
		"/v1/trade/history/tvl?chainId=96369&period=30d",
		"/v1/history/prices",
		"/v1/history/tvl",
		"/v1/trade/tokens/search?q=lux",
	} {
		w := ask(s, http.MethodGet, gone, nil)
		if w.Code != http.StatusNotFound {
			t.Errorf("GET %s answered %d: %s", gone, w.Code, w.Body.String())
		}
	}
}
