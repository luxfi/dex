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

	one := ask(s, http.MethodPost, "/v1/quote", asked)
	if one.Code != http.StatusOK {
		t.Fatalf("/v1/quote: got %d, want 200: %s", one.Code, one.Body.String())
	}
	all := ask(s, http.MethodPost, "/v1/quotes", asked)
	if all.Code != http.StatusOK {
		t.Fatalf("/v1/quotes: got %d, want 200: %s", all.Code, all.Body.String())
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
		t.Fatal("/v1/quotes returned nothing where /v1/quote returned a price")
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
	w := ask(newPoolOnlyServer(), http.MethodPost, "/v1/quotes", quoteRequest{
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
