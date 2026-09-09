package gateway

import (
	"context"
	"math/big"
	"strconv"
	"strings"
	"sync"
	"time"
)

// What a token is worth here.
//
// There is no oracle behind this and no vendor. The number is a reading of the
// same pools /v1/trade/quote reads, and it is the same number: spend some
// dollars of a numéraire, see how much of the token comes back, and put the
// pool's fee back into the answer. The fee matters — a pool charges its tier on
// the way through, so what you receive is the mid price less that tier, and the
// correction is the difference between $1.0031 and $1.0000 for WLUX, and
// between $3018.51 and $3000.44 for LETH two hops deep.
//
// Because it is the same number, the row on a market screen and the amount in
// the swap panel agree. Measured on Ethereum: /v1/trade/price says ASM is
// $0.036834 and /v1/trade/quote for the same size implies $0.036806. A screen
// that says one thing and a panel that pays another is worse than either.

// markSmall and markLarge are the two sizes a reading is taken at, in dollars.
//
// Not one whole token: one whole token is a fifty-cent trade in one market and
// a hundred-thousand-dollar trade in the next, and the second one moves the
// pool it is measuring — priced that way SHIB came out forty percent high.
//
// And not one size either. A pool with almost nothing left in it still answers,
// and the implied price of a rounding error comes out in the millions. Over the
// fifty tokens Ethereum lists first, a single reading put half of them more
// than one percent off an independent source and read AUCTION — a token worth
// $3.44 — at $889,434,176.
//
// So the pools are asked twice, ten times apart, and a price that does not
// survive the difference is not a price. What comes back is the smaller
// reading, which is the least moved by having been asked.
const (
	markSmall = 10
	markLarge = 100
)

// markDrift is how far apart those two readings may land before this stops
// calling the smaller one a price. Two percent for ten times the size.
//
// Chosen by measuring, over the fifty tokens Ethereum lists first: every
// reading that survives it agrees with an independent source, and every one
// that does not is a pool with nothing left in it. Loosen it to a quarter and
// six more tokens come back, four of them wrong.
var markDrift = big.NewRat(1, 50)

// markTTL is how long a reading stands.
//
// A quote is the price of a trade somebody is about to make, and is kept for
// ten seconds. A mark is what a list of two hundred rows draws beside each
// name, and asking a public endpoint for two hundred readings on every page
// view is how a public endpoint stops answering.
const markTTL = time.Minute

// markRetry is how long an unreadable chain is remembered. Much less, because
// the pools have not changed — our access to them has, and a minute of dashes
// outlives the thirty seconds of outage that caused them.
const markRetry = 10 * time.Second

// Price is a token and what it is worth, as /v1/trade/price answers.
type Price struct {
	ChainID  ChainID `json:"chainId"`
	Address  string  `json:"address"`
	Symbol   string  `json:"symbol"`
	Name     string  `json:"name"`
	Decimals int     `json:"decimals"`
	// USD is absent — not zero — when no pool this gateway reads holds the
	// token against either numéraire. A screen draws a dash for an absent
	// price, and it cannot un-draw a wrong number.
	USD *float64 `json:"priceUSD,omitempty"`
	// Venue is the arm that answered and Via the numéraire the reading was
	// taken against. They go missing with the price, because a number nobody
	// can trace is a number nobody can check.
	Venue string `json:"venue,omitempty"`
	Via   string `json:"via,omitempty"`
	// AsOf is when the pools were read, so a row drawn from a mark says how
	// old it is instead of implying it is now.
	AsOf time.Time `json:"asOf"`
}

// mark is one reading, kept in exact arithmetic until it reaches the wire.
type mark struct {
	usd   *big.Rat
	venue string
	via   string
	at    time.Time
	// unreachable records that the chain could not be read at all, which is a
	// different fact from a token no pool holds and is kept for less time.
	unreachable bool
}

func (m mark) price(chain ChainID, t Token) Price {
	p := Price{
		ChainID:  chain,
		Address:  t.Address,
		Symbol:   t.Symbol,
		Name:     t.Name,
		Decimals: t.Decimals,
		AsOf:     m.at,
	}
	if m.usd != nil {
		usd, _ := m.usd.Float64()
		p.USD, p.Venue, p.Via = &usd, m.venue, m.via
	}
	return p
}

// marks keeps each token's reading for a while, and computes a cold one once
// however many readers arrive for it at the same moment. Two hundred rows
// opening at once is one fan-out per token, not one per token per reader.
type marks struct {
	mu   sync.Mutex
	kept map[string]*held
}

type held struct {
	done  chan struct{}
	mark  mark
	until time.Time
}

func newMarks() *marks { return &marks{kept: map[string]*held{}} }

// take hands back either a reading to wait on or the job of making one.
func (m *marks) take(key string) (*held, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	if h, ok := m.kept[key]; ok {
		select {
		case <-h.done:
			if now.Before(h.until) {
				return h, false
			}
			delete(m.kept, key)
		default:
			// Still being read. Waiting for it is the whole point.
			return h, false
		}
	}

	// A cache nobody sweeps is a leak with a good reputation. Every miss drops
	// what has expired, which on this traffic is cheaper than a timer.
	for k, h := range m.kept {
		select {
		case <-h.done:
			if now.After(h.until) {
				delete(m.kept, k)
			}
		default:
		}
	}

	h := &held{done: make(chan struct{})}
	m.kept[key] = h
	return h, true
}

// fill publishes a reading to whoever is waiting for it, to stand for ttl. A
// ttl of zero publishes the reading and keeps nothing: the next caller reads
// the chain again.
func (m *marks) fill(h *held, mk mark, ttl time.Duration) {
	m.mu.Lock()
	h.mark, h.until = mk, time.Now().Add(ttl)
	m.mu.Unlock()
	close(h.done)
}

// markOf reads what one token is worth, or waits for the reader who got there
// first.
//
// The dollar this chain prices in is worth a dollar by construction, so it
// costs nothing to say so. Everything else is read against that dollar and
// against the chain's own token, and the cheaper of the two answers wins: a
// price is what somebody spending a hundred dollars actually receives, and the
// route that hands over the most tokens is the one nearest the middle of the
// market.
func (s *Server) markOf(ctx context.Context, chain ChainID, token Token) mark {
	num := s.chains.Numeraires(chain)
	if strings.EqualFold(token.Address, num.Stable) {
		return mark{usd: big.NewRat(1, 1), via: token.Symbol, at: time.Now()}
	}

	key := strconv.FormatUint(uint64(chain), 10) + ":" + strings.ToLower(token.Address)
	h, mine := s.marks.take(key)
	if !mine {
		select {
		case <-h.done:
			return h.mark
		case <-ctx.Done():
			// A reading we ran out of time for is not a token with no pool.
			return mark{at: time.Now(), unreachable: true}
		}
	}

	// Whatever happens below, including a panic, whoever is waiting on this
	// entry is handed the same answer this call returns.
	mk, kept := mark{at: time.Now()}, markTTL
	defer func() { s.marks.fill(h, mk, kept) }()

	// The best price at each of the two sizes, over every numéraire and every
	// arm. Cheapest wins at each size and the two are compared once, at the
	// end: gating each numéraire on its own would refuse the cheapest route
	// for being thin and then quote a dearer one that was not, which is a
	// price nobody would have paid.
	var small, large *mark
	keep := func(into **mark, from []mark) {
		for i := range from {
			if *into == nil || from[i].usd.Cmp((*into).usd) < 0 {
				*into = &from[i]
			}
		}
	}
	read, unreachable := 0, 0
	against := func(num Token, numUSD *big.Rat) {
		read++
		cheap, dear, err := s.readSizes(ctx, chain, num, numUSD, token)
		if err != nil {
			unreachable++
			return
		}
		keep(&small, cheap)
		keep(&large, dear)
	}

	if stable, ok := s.catalog.Token(chain, num.Stable); ok {
		against(stable, big.NewRat(1, 1))
	}
	// The chain's own token is where most of its pools are, and its own mark
	// is read against the dollar first. A token with no dollar pool — which on
	// Ethereum is most of them — is reachable only this way.
	if hub, ok := s.catalog.Token(chain, num.Hub); ok &&
		!strings.EqualFold(hub.Address, token.Address) &&
		!strings.EqualFold(hub.Address, num.Stable) {
		if hm := s.markOf(ctx, chain, hub); hm.usd != nil {
			against(hub, hm.usd)
		}
	}

	if small != nil && large != nil && steady(small.usd, large.usd) {
		mk.usd, mk.venue, mk.via = small.usd, small.venue, small.via
	}

	switch {
	case ctx.Err() != nil:
		// The caller went away, or the request ran out of time, while this was
		// still reading. Nothing here is an answer about the token: not the
		// absence, and not a partial reading that happened to land first.
		// Kept, it would hand the next caller a dash it did not earn.
		mk, kept = mark{at: time.Now(), unreachable: true}, 0
	case read > 0 && unreachable == read:
		// Every numéraire failed to reach the chain. The pools have not
		// changed — our access to them has — so this is remembered for
		// seconds rather than for a minute.
		mk.unreachable, kept = true, markRetry
	}
	return mk
}

// steady reports whether two readings of one market, ten times apart in size,
// agree closely enough to call the smaller one a price.
//
// A pool with nothing left in it still answers. Ten dollars into one buys a
// rounding error and a hundred buys a tenth of the same rounding error, so the
// two readings come back an order of magnitude apart and neither is a market.
func steady(small, large *big.Rat) bool {
	drift := new(big.Rat).Sub(large, small)
	drift.Quo(drift, small)
	return drift.Abs(drift).Cmp(markDrift) <= 0
}

// readSizes asks the same pools both sizes at once, because they are one
// question about one market and asking them in turn would double how long a
// cold page waits.
func (s *Server) readSizes(ctx context.Context, chain ChainID, num Token, numUSD *big.Rat, token Token) (cheap, dear []mark, fatal error) {
	var failed [2]error
	got := [2][]mark{}
	var wg sync.WaitGroup
	for i, dollars := range [2]int64{markSmall, markLarge} {
		wg.Add(1)
		go func(i int, dollars int64) {
			defer wg.Done()
			got[i], failed[i] = s.readAgainst(ctx, chain, num, numUSD, token, dollars)
		}(i, dollars)
	}
	wg.Wait()

	// The chain could not be read. One size failing while the other answers is
	// not that, and neither is a pair no pool holds.
	if failed[0] != nil && failed[1] != nil {
		return nil, nil, failed[0]
	}
	return got[0], got[1], nil
}

// readAgainst asks every arm on the chain what some dollars of one numéraire
// buy of a token, and turns each answer into a price.
//
// The error means the CHAIN could not be read: every arm on it failed. That is
// a fact about an endpoint and not about a pool, and reporting the two as one
// thing sends a reader looking for liquidity that is sitting right there.
func (s *Server) readAgainst(ctx context.Context, chain ChainID, num Token, numUSD *big.Rat, token Token, dollars int64) ([]mark, error) {
	venues := s.arms(chain)
	if venues == nil || numUSD.Sign() <= 0 {
		return nil, nil
	}
	probe := probeAmount(numUSD, num.Decimals, dollars)
	if probe.Sign() <= 0 {
		return nil, nil
	}

	held, err := venues.QueryAllVenues(ctx, VenueQuoteRequest{
		TokenIn:  num.Address,
		TokenOut: token.Address,
		Amount:   probe.String(),
		Type:     VenueQuoteTypeExactInput,
	})
	if err != nil {
		return nil, err
	}

	out := make([]mark, 0, len(held))
	for _, q := range held {
		got, ok := new(big.Int).SetString(q.AmountOut, 10)
		if !ok || got.Sign() <= 0 {
			continue
		}
		fee, err := strconv.ParseUint(q.Fee, 10, 32)
		if err != nil || fee >= feeScale {
			// A tier nobody can read is a tier nobody can undo. Reporting the
			// price with the fee still in it is high by that fee; guessing at
			// the fee is wrong by whatever the guess was.
			fee = 0
		}
		out = append(out, mark{
			usd:   unitPrice(probe, num.Decimals, numUSD, got, token.Decimals, fee),
			venue: q.Venue,
			via:   num.Symbol,
		})
	}
	return out, nil
}

// probeAmount is some dollars of a token, in its smallest units.
func probeAmount(numUSD *big.Rat, decimals int, dollars int64) *big.Int {
	n := new(big.Rat).SetInt(new(big.Int).Mul(big.NewInt(dollars), pow10(decimals)))
	n.Quo(n, numUSD)
	return new(big.Int).Quo(n.Num(), n.Denom())
}

// unitPrice is what one whole token is worth, given what a probe of the
// numéraire bought and the tier it went through.
func unitPrice(probe *big.Int, numDecimals int, numUSD *big.Rat, got *big.Int, tokenDecimals int, fee uint64) *big.Rat {
	spent := new(big.Rat).SetFrac(probe, pow10(numDecimals))
	spent.Mul(spent, numUSD)
	received := new(big.Rat).SetFrac(got, pow10(tokenDecimals))

	ask := new(big.Rat).Quo(spent, received)
	// The pool kept its tier on the way through, so what came back is the mid
	// price less that tier. Putting it back is what turns a 0.3% pool's $1.0031
	// into $1.0000, and two hops of it turn LETH's $3018.51 into $3000.44.
	return ask.Mul(ask, big.NewRat(int64(feeScale-fee), feeScale))
}

// pow10 is ten to the nth, for converting a token's smallest unit to a whole
// one.
func pow10(n int) *big.Int {
	return new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(n)), nil)
}
