// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"bytes"
	"encoding/binary"
	"math/big"
	"testing"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// TestUserIDFoldMatchesLx pins the DECOMPLECTED identity split that the custody
// model rests on:
//
//   - The matcher's order-book ROW (dex.DEXOrder.UserID) stays the 8-byte STP
//     handle (the frozen GPU ABI). dex.OrderToRow folds the user to the leading 8
//     bytes; that is UNCHANGED and is sufficient for self-trade prevention and
//     price-time rebuild.
//   - The d-chain SETTLEMENT identity (userKey16) is the FULL 16-byte wire user
//     (zapwire.UserSize) — the value balance:/locked: key by. It is byte-identical
//     to the wire's null-padded user[16] field, so the ledger credits the SAME
//     account the orders/deposits name.
//
// The two are deliberately DIFFERENT widths: the row models matching, d-chain
// state models settlement identity. This test pins both so neither silently
// regresses (a widened row would break the GPU ABI; a narrowed ledger identity
// would reopen the cross-user drain).
func TestUserIDFoldMatchesLx(t *testing.T) {
	for _, u := range []string{"", "a", "maker-a", "taker-x-very-long-name-overflow"} {
		// The matcher ROW handle is the 8-byte fold (frozen GPU ABI / STP).
		row := dex.OrderToRow(&dex.Order{ID: 1, User: u, UserID: u, Size: 1, Price: 1, Side: dex.Buy})
		var want8 [8]byte
		copy(want8[:], u)
		if got, want := row.UserID, binary.BigEndian.Uint64(want8[:]); got != want {
			t.Errorf("ROW handle: lx OrderToRow.UserID(%q) = %#x, want 8-byte fold %#x", u, got, want)
		}

		// The d-chain SETTLEMENT identity is the FULL 16-byte wire user.
		var wantWire [zapwire.UserSize]byte
		copy(wantWire[:], u)
		if got := userKey16(u); got != wantWire {
			t.Errorf("LEDGER identity: userKey16(%q) = %x, want full wire user[16] %x", u, got, wantWire)
		}
		if len(userKey16(u)) != zapwire.UserSize {
			t.Errorf("LEDGER identity width = %d, want %d (zapwire.UserSize)", len(userKey16(u)), zapwire.UserSize)
		}
	}
}

// TestLedgerUserKeyNoCollision is the PERMANENT REGRESSION for CRITICAL-1
// (cross-user native drain). It tests the REAL production derivation — userKey16
// folded into balanceLedgerKey — not a test double. Two addresses that SHARE their
// leading 8 bytes but differ in bytes 8..15 MUST map to DISTINCT ledger accounts,
// so the second can never read/withdraw the first's balance.
//
// FAIL-on-old -> PASS-on-fix: the deleted 8-byte fold (userID8) produced an
// IDENTICAL uint64 for both addresses, so their balance keys collided — the drain.
// The 16-byte userKey16 keeps every byte, so the keys differ. The test asserts the
// keys differ AND demonstrates the old 8-byte fold would have collided.
func TestLedgerUserKeyNoCollision(t *testing.T) {
	// Two 16-byte users sharing the leading 8 bytes, differing in bytes 8..15 —
	// exactly the red PoC's "colliding address" shape.
	victim := string([]byte{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08})
	attacker := string([]byte{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8})

	var asset [32]byte // native (ids.Empty) — the asset the PoC drains
	for i := range asset {
		asset[i] = 0
	}

	vKey := userKey16(victim)
	aKey := userKey16(attacker)
	if vKey == aKey {
		t.Fatalf("userKey16 collided for leading-8-shared users: %x == %x (CRITICAL-1 regressed)", vKey, aKey)
	}

	vLedger := balanceLedgerKey(prefixBalance, vKey, asset)
	aLedger := balanceLedgerKey(prefixBalance, aKey, asset)
	if bytes.Equal(vLedger, aLedger) {
		t.Fatalf("balance ledger keys collided for distinct users:\n  victim   = %x\n  attacker = %x\n(CRITICAL-1: attacker would read victim's balance)", vLedger, aLedger)
	}

	// Demonstrate the DELETED 8-byte fold WOULD have collided (the bug we removed):
	// folding either user to its leading 8 bytes yields the SAME uint64, so the
	// pre-fix balance:<user:8> keys were identical.
	old8 := func(u string) uint64 {
		var b [8]byte
		copy(b[:], u)
		return binary.BigEndian.Uint64(b[:])
	}
	if old8(victim) != old8(attacker) {
		t.Fatalf("test setup invalid: the two users do not share leading 8 bytes (old fold %#x != %#x)", old8(victim), old8(attacker))
	}
}

// TestQuoteUnitsCeilCoversFloor proves the lock ceil is always >= the matcher's
// floor quote for the same (base, price): lock >= spend, so a settle never
// overspends the lock (the affordability invariant that makes the custody ledger
// conserving without matcher surgery).
func TestQuoteUnitsCeilCoversFloor(t *testing.T) {
	// Exact PriceInt grid values (the wire's price ×1e8): 0.5, 1, 1.5, 5, 101.25, 3.333333.
	for _, base := range []uint64{1, 3, 7, 100, 999} {
		for _, pi := range []dex.PriceInt{50_000_000, 100_000_000, 150_000_000, 500_000_000, 10_125_000_000, 333_333_300} {
			lock := quoteUnitsCeil(new(big.Int).SetUint64(base), pi)
			// matcher floor: floor(base*priceInt/PriceMultiplier)
			floor := base * uint64(pi) / uint64(dex.PriceMultiplier)
			if lock < floor {
				t.Errorf("ceil lock %d < floor spend %d for base=%d priceInt=%d", lock, floor, base, pi)
			}
		}
	}
}
