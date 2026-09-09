package gateway

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// The most tokens one listing will draw. The largest chain here lists 403
	// and every chain together lists 935, so this is not a page size — it is a
	// ceiling on a request, and the default is everything. A limit that
	// silently cut Ethereum off at two hundred would be a screen missing two
	// hundred markets with nothing to say it.
	tokensAtMost = 1000
	// A price is a fan-out of eth_calls against a public endpoint we hold no
	// account with, so the number of them one request can start is stated
	// rather than left to the caller. Fifty rows is a page of a market screen;
	// a screen wanting two hundred asks four times, and by the second one most
	// of them are already read.
	pricesAtOnce = 50
	// And this many at a time, so a cold page is a steady stream of calls
	// rather than five hundred at once.
	pricesInFlight = 8
	// The whole fan-out is bounded well inside the server's own write timeout,
	// so a chain that has gone slow answers rather than holding the connection
	// until it is cut. A token that ran out of time counts as a chain that
	// could not be read and not as a token with no pool, which is the
	// difference between a 502 and a page of dashes.
	pricesWithin = 45 * time.Second
)

// handleTokens handles GET /v1/trade/tokens.
//
// The list travels with the binary, so this answers with no upstream, no key
// and no round trip. That is the whole point of it: every chain but ours has no
// indexer behind it, and lux.exchange drew nothing at all on Ethereum — not an
// empty table, no request made — because there was nothing to ask.
//
// Rows come back in the order they are listed: the chain's own token, then the
// dollar it is priced in, then the rest by symbol. There is no volume on any
// chain but ours, so there is nothing to rank by, and inventing a rank is
// inventing a number.
func (s *Server) handleTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	q := r.URL.Query()
	chain, err := askedChain(q)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}
	if chain != 0 && s.arms(chain) == nil {
		s.writeError(w, http.StatusNotFound, fmt.Errorf("no venue here reads chain %d", chain))
		return
	}

	limit := tokensAtMost
	if v := q.Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			s.writeError(w, http.StatusBadRequest, fmt.Errorf("limit: %q is not a count", v))
			return
		}
		limit = min(n, tokensAtMost)
	}

	s.writeJSON(w, http.StatusOK, s.catalog.List(chain, q.Get("q"), limit))
}

// handlePrice handles GET /v1/trade/price.
//
// One token or fifty, at ?token= repeated or comma-separated, and one entry
// back for each in the order asked. A token no pool here holds comes back
// without a price rather than with a zero: a screen can draw a dash, and it
// cannot un-draw a wrong number.
func (s *Server) handlePrice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	q := r.URL.Query()
	chain, err := askedChain(q)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}
	if chain == 0 {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("chainId is required: a price is a reading of one chain's pools"))
		return
	}
	if s.arms(chain) == nil {
		s.writeError(w, http.StatusNotFound, fmt.Errorf("no venue here reads chain %d", chain))
		return
	}

	asked, err := askedTokens(q)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	tokens := make([]Token, len(asked))
	for i, address := range asked {
		t, ok := s.catalog.Token(chain, address)
		if !ok {
			s.writeError(w, http.StatusNotFound, fmt.Errorf("chain %d lists no token at %s", chain, address))
			return
		}
		tokens[i] = t
	}

	out := make([]Price, len(tokens))
	unreachable := 0
	var mu sync.Mutex
	var wg sync.WaitGroup
	inFlight := make(chan struct{}, pricesInFlight)

	ctx, done := context.WithTimeout(s.requestContext(r), pricesWithin)
	defer done()
	for i, t := range tokens {
		wg.Add(1)
		go func(i int, t Token) {
			defer wg.Done()
			inFlight <- struct{}{}
			defer func() { <-inFlight }()

			mk := s.markOf(ctx, chain, t)
			mu.Lock()
			out[i] = mk.price(chain, t)
			if mk.unreachable {
				unreachable++
			}
			mu.Unlock()
		}(i, t)
	}
	wg.Wait()

	// Every one of them failed to reach the chain. That is a fact about an
	// endpoint, not about the pools, and answering it with a page of dashes
	// sends a reader looking for liquidity instead of at their RPC.
	if unreachable == len(tokens) && len(tokens) > 0 {
		s.writeError(w, http.StatusBadGateway, fmt.Errorf("chain %d could not be read", chain))
		return
	}

	s.writeJSON(w, http.StatusOK, out)
}

// askedChain reads ?chainId=, zero meaning every chain.
func askedChain(q url.Values) (ChainID, error) {
	v := q.Get("chainId")
	if v == "" {
		return 0, nil
	}
	id, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("chainId: %w", err)
	}
	return ChainID(id), nil
}

// askedTokens reads ?token=, repeated or comma-separated, in the order given.
func askedTokens(q url.Values) ([]string, error) {
	var out []string
	for _, group := range q["token"] {
		for _, one := range strings.Split(group, ",") {
			one = strings.TrimSpace(one)
			if one == "" {
				continue
			}
			if !isValidHexAddr(one) {
				return nil, fmt.Errorf("token: %q is not an address", one)
			}
			out = append(out, one)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one ?token= is required")
	}
	if len(out) > pricesAtOnce {
		return nil, fmt.Errorf("%d tokens asked for at once; %d is the most, and a screen wanting more asks again", len(out), pricesAtOnce)
	}
	return out, nil
}
