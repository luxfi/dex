package trading

import (
	"context"
	"fmt"
	"math/big"
	"sort"
	"strconv"
	"sync"
)

// ArbType identifies the arbitrage strategy.
type ArbType string

const (
	ArbTriangular ArbType = "TRIANGULAR"  // A→B→C→A on same chain
	ArbCrossChain ArbType = "CROSS_CHAIN" // Same pair priced differently across chains
)

// ArbOpportunity describes a detected arbitrage opportunity.
type ArbOpportunity struct {
	Type      ArbType   `json:"type"`
	Path      []ArbStep `json:"path"`
	ProfitBPS int       `json:"profitBps"`
	// ProfitScore: normalized profit indicator. 0.0 = break-even, 0.5 = 50 bps, 1.0 = 100+ bps.
	// NOT a probability — purely a function of spread.
	ProfitScore float64 `json:"profitScore"`
	GasCost     string  `json:"gasCost"`
	NetProfit   string  `json:"netProfit"`
	InputAmount string  `json:"inputAmount"`
}

// ArbStep is one leg of an arbitrage path.
type ArbStep struct {
	ChainID   int    `json:"chainId"`
	TokenIn   string `json:"tokenIn"`
	TokenOut  string `json:"tokenOut"`
	Venue     string `json:"venue"`
	AmountIn  string `json:"amountIn"`
	AmountOut string `json:"amountOut"`
}

// ArbScanRequest is the input for POST /v1/trade/arbitrage/scan.
type ArbScanRequest struct {
	Tokens       []string `json:"tokens"`
	ChainIDs     []int    `json:"chainIds"`
	MinProfitBPS int      `json:"minProfitBps"`
	InputAmount  string   `json:"inputAmount"`
}

// ArbScanResponse is returned by POST /v1/trade/arbitrage/scan.
type ArbScanResponse struct {
	Opportunities []ArbOpportunity `json:"opportunities"`
	ScannedPairs  int              `json:"scannedPairs"`
	ElapsedMs     int64            `json:"elapsedMs"`
}

// ArbScanner detects arbitrage opportunities across venues and chains.
type ArbScanner struct {
	router       *CrossChainRouter
	minProfitBPS int
}

func NewArbScanner(router *CrossChainRouter, minProfitBPS int) *ArbScanner {
	if minProfitBPS <= 0 {
		minProfitBPS = 10
	}
	return &ArbScanner{router: router, minProfitBPS: minProfitBPS}
}

// ScanTriangular finds A→B→C→A arbitrage on a single chain.
func (s *ArbScanner) ScanTriangular(ctx context.Context, chainID int, tokens []string, inputAmount string) ([]ArbOpportunity, error) {
	s.router.mu.RLock()
	chain := s.router.chains[chainID]
	s.router.mu.RUnlock()

	if chain == nil {
		return nil, fmt.Errorf("chain %d not registered", chainID)
	}
	if len(tokens) < 3 {
		return nil, fmt.Errorf("need at least 3 tokens for triangular arb")
	}

	amount, ok := new(big.Int).SetString(inputAmount, 10)
	if !ok || amount.Sign() <= 0 {
		return nil, fmt.Errorf("invalid input amount: %s", inputAmount)
	}

	var (
		opps []ArbOpportunity
		mu   sync.Mutex
		wg   sync.WaitGroup
	)

	// Semaphore limits concurrent goroutines to prevent resource exhaustion.
	// With 20 tokens max the permutation count is 6,840 — sem of 64 keeps
	// it bounded while still parallelizing across CPU cores.
	sem := make(chan struct{}, 64)

	for i := 0; i < len(tokens); i++ {
		for j := 0; j < len(tokens); j++ {
			if j == i {
				continue
			}
			for k := 0; k < len(tokens); k++ {
				if k == i || k == j {
					continue
				}
				wg.Add(1)
				a, b, c := tokens[i], tokens[j], tokens[k]
				sem <- struct{}{} // acquire
				go func() {
					defer wg.Done()
					defer func() { <-sem }() // release
					opp := s.evaluateTriangle(ctx, chain, a, b, c, inputAmount, amount)
					if opp != nil && opp.ProfitBPS >= s.minProfitBPS {
						mu.Lock()
						opps = append(opps, *opp)
						mu.Unlock()
					}
				}()
			}
		}
	}
	wg.Wait()

	sort.Slice(opps, func(i, j int) bool {
		return opps[i].ProfitBPS > opps[j].ProfitBPS
	})
	return opps, nil
}

func (s *ArbScanner) evaluateTriangle(ctx context.Context, chain *Chain, tokenA, tokenB, tokenC, inputAmount string, amount *big.Int) *ArbOpportunity {
	leg1 := bestQuoteOnChain(ctx, chain, tokenA, tokenB, inputAmount)
	if leg1 == nil {
		return nil
	}
	leg2 := bestQuoteOnChain(ctx, chain, tokenB, tokenC, leg1.AmountOut)
	if leg2 == nil {
		return nil
	}
	leg3 := bestQuoteOnChain(ctx, chain, tokenC, tokenA, leg2.AmountOut)
	if leg3 == nil {
		return nil
	}

	finalAmount, ok := new(big.Int).SetString(leg3.AmountOut, 10)
	if !ok || finalAmount.Sign() <= 0 {
		return nil
	}

	profit := new(big.Int).Sub(finalAmount, amount)
	if profit.Sign() <= 0 {
		return nil
	}

	profitBPS := new(big.Int).Mul(profit, big.NewInt(10000))
	profitBPS.Div(profitBPS, amount)

	profitScore := float64(profitBPS.Int64()) / 100.0
	if profitScore > 1.0 {
		profitScore = 1.0
	}

	gasCost := sumVenueGas([]*VenueQuote{leg1, leg2, leg3})

	return &ArbOpportunity{
		Type: ArbTriangular,
		Path: []ArbStep{
			{ChainID: chain.ChainID, TokenIn: tokenA, TokenOut: tokenB, Venue: leg1.Venue, AmountIn: inputAmount, AmountOut: leg1.AmountOut},
			{ChainID: chain.ChainID, TokenIn: tokenB, TokenOut: tokenC, Venue: leg2.Venue, AmountIn: leg1.AmountOut, AmountOut: leg2.AmountOut},
			{ChainID: chain.ChainID, TokenIn: tokenC, TokenOut: tokenA, Venue: leg3.Venue, AmountIn: leg2.AmountOut, AmountOut: leg3.AmountOut},
		},
		ProfitBPS:   int(profitBPS.Int64()),
		ProfitScore: profitScore,
		GasCost:     strconv.Itoa(gasCost),
		NetProfit:   profit.String(),
		InputAmount: inputAmount,
	}
}

// ScanCrossChain finds price differences for the same pair across chains.
func (s *ArbScanner) ScanCrossChain(ctx context.Context, baseToken, quoteToken string, chainIDs []int, inputAmount string) ([]ArbOpportunity, error) {
	if len(chainIDs) < 2 {
		return nil, fmt.Errorf("need at least 2 chains for cross-chain arb")
	}

	amount, ok := new(big.Int).SetString(inputAmount, 10)
	if !ok || amount.Sign() <= 0 {
		return nil, fmt.Errorf("invalid input amount: %s", inputAmount)
	}

	type chainQuote struct {
		chainID int
		quote   *VenueQuote
	}

	var quotes []chainQuote
	for _, chainID := range chainIDs {
		s.router.mu.RLock()
		chain := s.router.chains[chainID]
		s.router.mu.RUnlock()
		if chain == nil {
			continue
		}
		q := bestQuoteOnChain(ctx, chain, baseToken, quoteToken, inputAmount)
		if q != nil {
			quotes = append(quotes, chainQuote{chainID: chainID, quote: q})
		}
	}

	if len(quotes) < 2 {
		return nil, nil
	}

	var opps []ArbOpportunity
	for i := 0; i < len(quotes); i++ {
		for j := i + 1; j < len(quotes); j++ {
			outi, _ := new(big.Int).SetString(quotes[i].quote.AmountOut, 10)
			outj, _ := new(big.Int).SetString(quotes[j].quote.AmountOut, 10)
			if outi == nil || outj == nil {
				continue
			}

			var buyChain, sellChain chainQuote
			var buyOut, sellOut *big.Int
			if outi.Cmp(outj) > 0 {
				buyChain, sellChain = quotes[j], quotes[i]
				buyOut, sellOut = outj, outi
			} else {
				buyChain, sellChain = quotes[i], quotes[j]
				buyOut, sellOut = outi, outj
			}

			spread := new(big.Int).Sub(sellOut, buyOut)
			spreadBPS := new(big.Int).Mul(spread, big.NewInt(10000))
			spreadBPS.Div(spreadBPS, buyOut)

			if int(spreadBPS.Int64()) < s.minProfitBPS {
				continue
			}

			profitScore := float64(spreadBPS.Int64()) / 100.0
			if profitScore > 1.0 {
				profitScore = 1.0
			}

			gasCost := sumVenueGas([]*VenueQuote{buyChain.quote, sellChain.quote})

			opps = append(opps, ArbOpportunity{
				Type: ArbCrossChain,
				Path: []ArbStep{
					{ChainID: buyChain.chainID, TokenIn: baseToken, TokenOut: quoteToken, Venue: buyChain.quote.Venue, AmountIn: inputAmount, AmountOut: buyOut.String()},
					{ChainID: sellChain.chainID, TokenIn: baseToken, TokenOut: quoteToken, Venue: sellChain.quote.Venue, AmountIn: inputAmount, AmountOut: sellOut.String()},
				},
				ProfitBPS:   int(spreadBPS.Int64()),
				ProfitScore: profitScore,
				NetProfit:   spread.String(),
				InputAmount: inputAmount,
				GasCost:     strconv.Itoa(gasCost),
			})
		}
	}

	sort.Slice(opps, func(i, j int) bool {
		return opps[i].ProfitBPS > opps[j].ProfitBPS
	})
	return opps, nil
}

// sumVenueGas sums the GasEstimate fields from venue quotes.
// Falls back to 450000 if the total is zero (e.g. off-chain venues report "0").
func sumVenueGas(quotes []*VenueQuote) int {
	total := 0
	for _, q := range quotes {
		g, _ := strconv.Atoi(q.GasEstimate)
		total += g
	}
	if total == 0 {
		total = 450000
	}
	return total
}

func bestQuoteOnChain(ctx context.Context, chain *Chain, tokenIn, tokenOut, amount string) *VenueQuote {
	req := QuoteRequest{
		TokenIn: tokenIn, TokenOut: tokenOut,
		Amount: amount, Type: QuoteTypeExactInput, ChainID: chain.ChainID,
	}
	var best *VenueQuote
	for _, venue := range chain.Venues {
		q, err := venue.Quote(ctx, req)
		if err != nil || q == nil {
			continue
		}
		if best == nil {
			best = q
			continue
		}
		curr, _ := new(big.Int).SetString(q.AmountOut, 10)
		prev, _ := new(big.Int).SetString(best.AmountOut, 10)
		if curr != nil && prev != nil && curr.Cmp(prev) > 0 {
			best = q
		}
	}
	return best
}
