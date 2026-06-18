// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"math/big"
	"testing"

	"github.com/luxfi/dex/pkg/lx"
)

// TestUserIDFoldMatchesLx pins that the d-chain ledger's user->8-byte fold
// (userID8) is byte-identical to the fold lx uses for DEXTrade/DEXOrder UserIDs
// (via OrderToRow). If these diverged, the ledger would credit/debit a different
// account than the trade names — a value-conservation break.
func TestUserIDFoldMatchesLx(t *testing.T) {
	for _, u := range []string{"", "a", "maker-a", "taker-x-very-long-name-overflow"} {
		// lx's fold is exercised through OrderToRow's UserID.
		row := lx.OrderToRow(&lx.Order{ID: 1, User: u, UserID: u, Size: 1, Price: 1, Side: lx.Buy})
		if got, want := userID8(u), row.UserID; got != want {
			t.Errorf("userID8(%q) = %#x, lx OrderToRow.UserID = %#x (folds diverged)", u, got, want)
		}
	}
}

// TestQuoteUnitsCeilCoversFloor proves the lock ceil is always >= the matcher's
// floor quote for the same (base, price): lock >= spend, so a settle never
// overspends the lock (the affordability invariant that makes the custody ledger
// conserving without matcher surgery).
func TestQuoteUnitsCeilCoversFloor(t *testing.T) {
	for _, base := range []uint64{1, 3, 7, 100, 999} {
		for _, price := range []float64{0.5, 1, 1.5, 5, 101.25, 3.333333} {
			pi := priceToInt(price)
			lock := quoteUnitsCeil(new(big.Int).SetUint64(base), pi)
			// matcher floor: floor(base*priceInt/PriceMultiplier)
			floor := base * uint64(pi) / uint64(lx.PriceMultiplier)
			if lock < floor {
				t.Errorf("ceil lock %d < floor spend %d for base=%d price=%v", lock, floor, base, price)
			}
		}
	}
}
