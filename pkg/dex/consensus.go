// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"sync/atomic"
	"time"
)

// consensus.go holds the determinism guards that let an OrderBook be driven from
// a consensus VM (the d-chain) without any non-deterministic input leaking into
// the matched state. There is exactly one rule: in the consensus path the VM owns
// the order's ID and timestamp; the book MUST NOT mint a process-local
// LastOrderID nor stamp wall-clock time.Now(), because both fork across
// validators replaying the same ordered block.
//
// This does NOT fork the matcher. ConsensusAddOrder reuses the identical resting
// insertion that AddOrder uses (addRestingLocked); it only removes the two
// non-deterministic side effects. The taker path (SubmitMarketable) was already
// made deterministic upstream (it never mints/stamps), so the two consensus
// entrypoints are: ConsensusAddOrder (rest a limit order) and SubmitMarketable
// (cross a marketable order). The d-chain never calls MatchOrders — that
// resting-cross sweep stamps time.Now() (orderbook.go) and is for the
// non-consensus, single-process server only.

// nanosToTime converts unix-nanoseconds (as carried by a DEXOrder row and by the
// block timestamp) back to a time.Time. A zero value maps to the zero time so a
// never-stamped order stays unstamped.
func nanosToTime(nanos int64) time.Time {
	if nanos == 0 {
		return time.Time{}
	}
	return time.Unix(0, nanos).UTC()
}

// ConsensusAddOrder rests a limit order using a caller-supplied, deterministic
// ID and Timestamp. It is the d-chain's single entry for placing resting
// liquidity. Unlike AddOrder it never mints order.ID from the LastOrderID counter
// and never stamps order.Timestamp from time.Now(): the VM derives both from
// block context (height/txIndex for the ID, block timestamp for the stamp) before
// calling this, so every validator replaying the same block produces the
// byte-identical resting order.
//
// Preconditions (the VM guarantees them; we do not silently repair a violation
// because a zero ID/timestamp in consensus is a determinism bug, not a default):
//   - order.ID != 0
//   - !order.Timestamp.IsZero()
//
// If either preconditions is violated the order is rejected (returns 0) rather
// than minting a non-deterministic value. Returns the order ID on success.
func (ob *OrderBook) ConsensusAddOrder(order *Order) uint64 {
	if order == nil || order.ID == 0 || order.Timestamp.IsZero() {
		return 0
	}
	if order.Status == "" {
		order.Status = Open
	}
	if order.RemainingSize == 0 && order.Filled == 0 {
		order.RemainingSize = order.Size
	}
	return ob.addRestingLocked(order)
}

// addRestingLocked is the resting-insertion core shared by the consensus path. It
// validates the order, runs self-trade / post-only guards, inserts into the
// correct price tree under ob.mu, and tracks it in the order maps. It performs NO
// matching: a resting limit order placed via the OrderBook never crosses on placement
// (takers cross via SubmitMarketable). It assumes order.ID and order.RemainingSize
// are already set.
//
// This is the same insertion AddOrder performs for a plain GTC limit order; it is
// factored out so the consensus entry and the legacy entry share one body (no
// second copy of the tree-insert/tracking logic to drift).
func (ob *OrderBook) addRestingLocked(order *Order) uint64 {
	if err := ob.validateOrder(order); err != nil {
		order.Status = Rejected
		return 0
	}

	priceInt := PriceInt(order.Price * PriceMultiplier)

	ob.mu.Lock()
	defer ob.mu.Unlock()

	// Self-trade prevention against the resting book.
	userIdentifier := order.User
	if userIdentifier == "" {
		userIdentifier = order.UserID
	}
	if userIdentifier != "" && ob.checkSelfTrade(order) {
		order.Status = Rejected
		return 0
	}

	// Post-only: a resting order that would take is rejected.
	if order.PostOnly || order.Flags&OrderFlagPostOnly != 0 {
		if ob.wouldTakeLiquidity(order) {
			order.Status = Rejected
			return 0
		}
	}

	var tree *OrderTree
	if order.Side == Buy {
		tree = (*OrderTree)(atomic.LoadPointer(&ob.bids))
	} else {
		tree = (*OrderTree)(atomic.LoadPointer(&ob.asks))
	}
	ob.addToTreeOptimized(tree, order, priceInt)

	ob.Orders[order.ID] = order
	ob.ordersMap.Store(order.ID, order)
	if ob.UserOrders[order.User] == nil {
		ob.UserOrders[order.User] = make([]uint64, 0)
	}
	ob.UserOrders[order.User] = append(ob.UserOrders[order.User], order.ID)

	return order.ID
}
