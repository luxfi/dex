package trading

import (
	"context"
	"fmt"
	"math/big"
	"sort"
	"sync"
)

// CrossChainRouter finds optimal paths across chains using BFS.
type CrossChainRouter struct {
	chains  map[int]*Chain
	bridges []BridgeVenue
	mu      sync.RWMutex
}

func NewCrossChainRouter() *CrossChainRouter {
	return &CrossChainRouter{chains: make(map[int]*Chain)}
}

func (r *CrossChainRouter) RegisterChain(chain *Chain) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.chains[chain.ChainID] = chain
}

func (r *CrossChainRouter) RegisterBridge(bridge BridgeVenue) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.bridges = append(r.bridges, bridge)
}

// FindRoutes discovers all valid routes between source and dest chains.
func (r *CrossChainRouter) FindRoutes(ctx context.Context, req CrossChainQuoteRequest) ([]CrossChainRoute, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if _, ok := r.chains[req.SourceChain]; !ok {
		return nil, fmt.Errorf("unsupported source chain: %d", req.SourceChain)
	}
	if _, ok := r.chains[req.DestChain]; !ok {
		return nil, fmt.Errorf("unsupported dest chain: %d", req.DestChain)
	}

	maxHops := req.MaxHops
	if maxHops <= 0 {
		maxHops = 4
	}

	if req.SourceChain == req.DestChain {
		return r.findLocalRoutes(ctx, req)
	}

	return r.bfsRoutes(ctx, req, maxHops)
}

func (r *CrossChainRouter) findLocalRoutes(ctx context.Context, req CrossChainQuoteRequest) ([]CrossChainRoute, error) {
	chain := r.chains[req.SourceChain]
	if chain == nil {
		return nil, nil
	}

	quoteReq := QuoteRequest{
		TokenIn: req.TokenIn, TokenOut: req.TokenOut,
		Amount: req.Amount, Type: req.Type, ChainID: req.SourceChain,
		Swapper: req.Swapper, Slippage: req.Slippage,
	}

	var routes []CrossChainRoute
	for _, venue := range chain.Venues {
		q, err := venue.Quote(ctx, quoteReq)
		if err != nil || q == nil {
			continue
		}
		routes = append(routes, CrossChainRoute{
			Hops: []CrossChainHop{{
				Type: HopTypeSwap, ChainID: req.SourceChain,
				TokenIn: req.TokenIn, TokenOut: req.TokenOut,
				Venue: q.Venue, AmountIn: req.Amount, AmountOut: q.AmountOut,
				Fee: q.Fee, GasEstimate: q.GasEstimate,
			}},
			TotalAmountIn: req.Amount, TotalAmountOut: q.AmountOut,
			TotalGasEstimate: q.GasEstimate, TotalBridgeFee: "0",
			Chains: []int{req.SourceChain},
		})
	}
	return routes, nil
}

// bfsNode tracks state during BFS path exploration.
type bfsNode struct {
	chainID       int
	token         string
	amount        string
	originalInput string
	hops          []CrossChainHop
	chains        []int
	totalGas      *big.Int
	totalFee      *big.Int
	depth         int
}

func (r *CrossChainRouter) bfsRoutes(ctx context.Context, req CrossChainQuoteRequest, maxHops int) ([]CrossChainRoute, error) {
	var results []CrossChainRoute

	start := &bfsNode{
		chainID: req.SourceChain, token: req.TokenIn,
		amount: req.Amount, originalInput: req.Amount,
		chains: []int{req.SourceChain},
		totalGas: big.NewInt(0), totalFee: big.NewInt(0),
	}

	queue := []*bfsNode{start}

	for len(queue) > 0 && len(results) < 10 {
		node := queue[0]
		queue = queue[1:]

		if node.depth > maxHops {
			continue
		}

		// Reached destination chain.
		if node.chainID == req.DestChain {
			if node.token == req.TokenOut {
				results = append(results, r.nodeToRoute(node)...)
			} else {
				results = append(results, r.tryLocalSwap(ctx, node, req.TokenOut)...)
			}
			continue
		}

		// Try bridging to other chains.
		for _, bridge := range r.bridges {
			if req.PreferBridge != "" && bridge.BridgeType() != req.PreferBridge {
				continue
			}

			for _, route := range bridge.SupportedRoutes() {
				destChain := bridgeDest(route, node.chainID)
				if destChain < 0 || chainVisited(node.chains, destChain) {
					continue
				}

				supported := bridge.SupportedTokens(node.chainID, destChain)
				if len(supported) > 0 && !tokenInList(node.token, supported) {
					// Swap to a bridgeable token first.
					for _, tok := range supported {
						queue = append(queue, r.trySwapAndBridge(ctx, node, tok, bridge, destChain, maxHops)...)
					}
					continue
				}

				// Bridge directly.
				bridged := r.tryDirectBridge(ctx, node, bridge, destChain, req)
				if bridged != nil {
					queue = append(queue, bridged)
				}
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		ai, _ := new(big.Int).SetString(results[i].TotalAmountOut, 10)
		aj, _ := new(big.Int).SetString(results[j].TotalAmountOut, 10)
		if ai == nil {
			return false
		}
		if aj == nil {
			return true
		}
		return ai.Cmp(aj) > 0
	})

	return results, nil
}

func (r *CrossChainRouter) tryDirectBridge(ctx context.Context, node *bfsNode, bridge BridgeVenue, destChain int, req CrossChainQuoteRequest) *bfsNode {
	bq, err := bridge.QuoteBridge(ctx, BridgeQuoteRequest{
		Token: node.token, Amount: node.amount,
		SourceChain: node.chainID, DestChain: destChain,
		Recipient: req.Swapper,
	})
	if err != nil || bq == nil {
		return nil
	}

	bridgeFee := parseBigInt(bq.Fee)
	bridgeGas := parseBigInt(bq.GasEstimate)

	hops := appendHop(node.hops, CrossChainHop{
		Type: HopTypeBridge, ChainID: node.chainID,
		TokenIn: node.token, TokenOut: node.token,
		Bridge: bridge.BridgeType(), DestChainID: destChain,
		AmountIn: node.amount, AmountOut: bq.AmountOut,
		Fee: bq.Fee, GasEstimate: bq.GasEstimate,
	})

	return &bfsNode{
		chainID: destChain, token: node.token,
		amount: bq.AmountOut, originalInput: node.originalInput,
		hops: hops, chains: append(append([]int{}, node.chains...), destChain),
		totalGas: new(big.Int).Add(node.totalGas, bridgeGas),
		totalFee: new(big.Int).Add(node.totalFee, bridgeFee),
		depth:    node.depth + 1,
	}
}

func (r *CrossChainRouter) tryLocalSwap(ctx context.Context, node *bfsNode, targetToken string) []CrossChainRoute {
	chain := r.chains[node.chainID]
	if chain == nil {
		return nil
	}

	quoteReq := QuoteRequest{
		TokenIn: node.token, TokenOut: targetToken,
		Amount: node.amount, Type: QuoteTypeExactInput,
		ChainID: node.chainID,
	}

	var routes []CrossChainRoute
	for _, venue := range chain.Venues {
		q, err := venue.Quote(ctx, quoteReq)
		if err != nil || q == nil {
			continue
		}

		hops := appendHop(node.hops, CrossChainHop{
			Type: HopTypeSwap, ChainID: node.chainID,
			TokenIn: node.token, TokenOut: targetToken,
			Venue: q.Venue, AmountIn: node.amount, AmountOut: q.AmountOut,
			Fee: q.Fee, GasEstimate: q.GasEstimate,
		})

		swapGas := parseBigInt(q.GasEstimate)
		routes = append(routes, CrossChainRoute{
			Hops: hops, TotalAmountIn: node.originalInput, TotalAmountOut: q.AmountOut,
			TotalGasEstimate: new(big.Int).Add(node.totalGas, swapGas).String(),
			TotalBridgeFee:   node.totalFee.String(),
			Chains:           node.chains,
		})
	}
	return routes
}

func (r *CrossChainRouter) trySwapAndBridge(ctx context.Context, node *bfsNode, targetToken string, bridge BridgeVenue, destChain, maxHops int) []*bfsNode {
	if node.depth+2 > maxHops {
		return nil
	}

	chain := r.chains[node.chainID]
	if chain == nil {
		return nil
	}

	quoteReq := QuoteRequest{
		TokenIn: node.token, TokenOut: targetToken,
		Amount: node.amount, Type: QuoteTypeExactInput,
		ChainID: node.chainID,
	}

	var results []*bfsNode
	for _, venue := range chain.Venues {
		q, err := venue.Quote(ctx, quoteReq)
		if err != nil || q == nil {
			continue
		}

		bq, err := bridge.QuoteBridge(ctx, BridgeQuoteRequest{
			Token: targetToken, Amount: q.AmountOut,
			SourceChain: node.chainID, DestChain: destChain,
		})
		if err != nil || bq == nil {
			continue
		}

		swapGas := parseBigInt(q.GasEstimate)
		bridgeGas := parseBigInt(bq.GasEstimate)
		bridgeFee := parseBigInt(bq.Fee)

		hops := appendHop(node.hops,
			CrossChainHop{
				Type: HopTypeSwap, ChainID: node.chainID,
				TokenIn: node.token, TokenOut: targetToken,
				Venue: q.Venue, AmountIn: node.amount, AmountOut: q.AmountOut,
				Fee: q.Fee, GasEstimate: q.GasEstimate,
			},
			CrossChainHop{
				Type: HopTypeBridge, ChainID: node.chainID,
				TokenIn: targetToken, TokenOut: targetToken,
				Bridge: bridge.BridgeType(), DestChainID: destChain,
				AmountIn: q.AmountOut, AmountOut: bq.AmountOut,
				Fee: bq.Fee, GasEstimate: bq.GasEstimate,
			},
		)

		results = append(results, &bfsNode{
			chainID: destChain, token: targetToken,
			amount: bq.AmountOut, originalInput: node.originalInput,
			hops: hops, chains: append(append([]int{}, node.chains...), destChain),
			totalGas: new(big.Int).Add(new(big.Int).Add(node.totalGas, swapGas), bridgeGas),
			totalFee: new(big.Int).Add(node.totalFee, bridgeFee),
			depth:    node.depth + 2,
		})
	}
	return results
}

func (r *CrossChainRouter) nodeToRoute(node *bfsNode) []CrossChainRoute {
	if len(node.hops) == 0 {
		return nil
	}
	return []CrossChainRoute{{
		Hops: node.hops, TotalAmountIn: node.originalInput, TotalAmountOut: node.amount,
		TotalGasEstimate: node.totalGas.String(), TotalBridgeFee: node.totalFee.String(),
		Chains: node.chains,
	}}
}

// --- helpers ---

func bridgeDest(route [2]int, src int) int {
	if route[0] == src {
		return route[1]
	}
	if route[1] == src {
		return route[0]
	}
	return -1
}

func chainVisited(chains []int, id int) bool {
	for _, c := range chains {
		if c == id {
			return true
		}
	}
	return false
}

func tokenInList(token string, list []string) bool {
	for _, t := range list {
		if t == token {
			return true
		}
	}
	return false
}

func appendHop(existing []CrossChainHop, hops ...CrossChainHop) []CrossChainHop {
	result := make([]CrossChainHop, len(existing)+len(hops))
	copy(result, existing)
	copy(result[len(existing):], hops)
	return result
}

func parseBigInt(s string) *big.Int {
	n, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return big.NewInt(0)
	}
	return n
}
