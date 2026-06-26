// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"math/big"

	"github.com/luxfi/dex/pkg/lx"
)

// execute.go is the top-level swap orchestrator — the entry point the 0x9999
// precompile (and the dchain VM) call to route ONE swap across the on-chain
// sources and journal the joint C+D effect. The sequence is:
//
//  1. LOCK the taker's full input ONCE (available -> locked, and record the C-side
//     vault debit in the journal). A split across CLOB+AMM draws from this single
//     lock, so the taker can never spend more than they locked.
//  2. ROUTE: best-execution across the sources (router.Route), which mutates the
//     book/ledger/reserves in the Store and records the proceeds in the journal.
//  3. REFUND the unspent input remainder (locked -> available, and record the
//     C-side vault refund in the journal).
//
// Steps 1-3 run inside ONE EVM call covered by ONE EVM snapshot in the precompile,
// so a revert anywhere rolls back the lock, the routing, and the journal together —
// both-or-neither in one consensus-finalized block. The function returns the
// SwapResult: the aggregate fills, the BalanceDelta (the taker's net C-side move),
// and the journal the caller drains onto EVM balances.

// SwapResult is the outcome of one routed swap.
type SwapResult struct {
	// Fills are the per-source fills (CLOB trades + AMM deltas) in execution order,
	// for events/receipts/the indexer.
	Fills []Fill
	// AmountIn / AmountOut are the input consumed and the output produced, in the
	// spend/receive asset's integer units.
	AmountIn  uint64
	AmountOut uint64
	// AmountInAsset / AmountOutAsset are the full injective ids of the spent and
	// received assets (so the caller maps the BalanceDelta back to EVM tokens).
	AmountInAsset  AssetID
	AmountOutAsset AssetID
	// Journal is the joint journal: the C-surface deltas the caller applies to the
	// EVM StateDB in the same Run.
	Journal *Journal
}

// ExecuteSwap routes one swap across the given router's sources, writing the D-side
// effects to db and the C-side deltas to a fresh journal, and returns the result.
//
// The taker's spend asset is quote for a BUY (taker buys base) and base for a SELL.
// The lock is the FULL req.AmountIn of that asset; whatever the router does not
// consume is refunded. The router enforces the taker price limit and MinOut; on a
// breach ExecuteSwap returns the error and the caller reverts (nothing committed).
func ExecuteSwap(db Store, router *Router, req SwapRequest) (*SwapResult, error) {
	if req.Class != ClassPublicCLOB {
		// Only the public route is built in the first cut. An FHE-private swap is
		// refused here (not silently downgraded) until the private-pool source lands.
		return nil, ErrNoLiquidity
	}
	if req.AmountIn == 0 {
		return nil, ErrNoLiquidity
	}
	// An exact-input BUY off a quote budget MUST carry a price ceiling (unbounded-
	// slippage market buys are refused). A SELL needs no limit.
	if req.Side == lx.Buy && req.LimitPrice <= 0 {
		return nil, ErrBuyRequiresLimit
	}
	// Resolve the spend/receive assets from the side.
	inAsset, outAsset := req.Quote, req.Base // BUY: spend quote, receive base
	if req.Side == lx.Sell {
		inAsset, outAsset = req.Base, req.Quote // SELL: spend base, receive quote
	}

	j := NewJournal()

	// (1) Lock the taker's full input once. The C-side debit (taker EVM balance ->
	// 0x9999 vault) is recorded; the D-side lock (available -> locked) is written.
	if err := LockFromAvailable(db, req.TakerUser, inAsset, req.AmountIn); err != nil {
		return nil, err
	}
	j.DebitTakerInput(inAsset, req.AmountIn)

	// (2) Route best-execution across the sources. The router writes the book/ledger/
	// reserve effects and records the proceeds credits in the journal.
	totalOut, totalInUsed, fills, err := router.Route(db, req, j)
	if err != nil {
		return nil, err
	}

	// (3) Refund the unspent input remainder.
	if totalInUsed < req.AmountIn {
		refund := req.AmountIn - totalInUsed
		if err := UnlockToAvailable(db, req.TakerUser, inAsset, refund); err != nil {
			return nil, err
		}
		j.RefundTakerInput(inAsset, refund)
	}

	return &SwapResult{
		Fills:          fills,
		AmountIn:       totalInUsed,
		AmountOut:      totalOut,
		AmountInAsset:  inAsset,
		AmountOutAsset: outAsset,
		Journal:        j,
	}, nil
}

// SignedLegs returns the swap's signed (inAsset, -in) and (outAsset, +out) so the
// caller maps them to the V4 BalanceDelta legs by matching the asset id to
// currency0 (base) / currency1 (quote). This keeps dex free of the V4 currency-
// ordering convention while giving the caller exactly the signed magnitudes.
func (r *SwapResult) SignedLegs() (inAsset AssetID, inNeg *big.Int, outAsset AssetID, outPos *big.Int) {
	return r.AmountInAsset, new(big.Int).Neg(new(big.Int).SetUint64(r.AmountIn)),
		r.AmountOutAsset, new(big.Int).SetUint64(r.AmountOut)
}
