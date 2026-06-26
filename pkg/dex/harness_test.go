// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"testing"
	"time"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/lx"
)

// harness_test.go is the dex proof harness. It uses REAL asset ids — the
// left-pad of a real 20-byte EVM token address (the production assetID), NEVER a
// synthetic ASCII-ticker handle. The Store is an in-memory database that satisfies
// dex.Store exactly as the precompile's 0x9999-storage adapter and dchain's
// versiondb overlay do, so the logic proven here is the identical logic both homes
// run.

// realAsset returns the canonical 32-byte asset id for a real 20-byte EVM token
// address (left-padded) — the SAME map the production precompile uses. This is the
// real-asset model: identity IS the token's on-chain address, never a ticker.
func realAsset(addr20 [20]byte) AssetID {
	var id AssetID
	copy(id[12:], addr20[:])
	return id
}

// addr builds a deterministic 20-byte token address from a single byte tag, so a
// test names real tokens (e.g. LUSD at 0x..01, LETH at 0x..02) by their address —
// the production identity — without any ticker packing.
func addr(tag byte) [20]byte {
	var a [20]byte
	a[19] = tag
	return a
}

// Real test tokens, identified by their on-chain ADDRESS (the production identity).
var (
	tokLUX  = realAsset(addr(0x00)) // native LUX is the zero address -> all-zero id; use a distinct base here
	tokLUSD = realAsset(addr(0x01)) // a real ERC-20 at 0x..01 (the quote)
	tokLETH = realAsset(addr(0x02)) // a real ERC-20 at 0x..02 (a base)
	tokWBTC = realAsset(addr(0x03)) // a real ERC-20 at 0x..03 (another base)
)

// account builds a deterministic 16-byte account id from a tag byte (the wallet
// identity the ledger keys by). Distinct tags are distinct accounts.
func account(tag byte) AccountID {
	var k AccountID
	k[0] = tag
	return k
}

// market builds a deterministic 32-byte pool id from a tag.
func market(tag byte) [32]byte {
	var p [32]byte
	p[0] = 0x9d
	p[31] = tag
	return p
}

// newStore returns a fresh in-memory dex.Store.
func newStore() Store { return memdb.New() }

// seedMarket opens a market binding (base, quote) and returns its pool id.
func seedMarket(t *testing.T, db Store, tag byte, base, quote AssetID) [32]byte {
	t.Helper()
	pid := market(tag)
	if err := OpenMarket(db, pid, base, quote); err != nil {
		t.Fatalf("OpenMarket: %v", err)
	}
	return pid
}

// seedDeposit credits an account's available balance for an asset.
func seedDeposit(t *testing.T, db Store, acct AccountID, asset AssetID, amount uint64) {
	t.Helper()
	if err := Deposit(db, acct, asset, amount); err != nil {
		t.Fatalf("Deposit: %v", err)
	}
}

// restOrder places a resting maker LIMIT order and asserts it rested.
func restOrder(t *testing.T, db Store, pid [32]byte, maker AccountID, side lx.Side, price, size float64, orderID uint64) {
	t.Helper()
	ok, err := PlaceOrder(db, pid, maker, side, price, size, orderID, time.Unix(0, 1_000))
	if err != nil {
		t.Fatalf("PlaceOrder: %v", err)
	}
	if !ok {
		t.Fatalf("PlaceOrder: order %d did not rest (insufficient funds or dust)", orderID)
	}
}

// totalLedger returns sum(available)+sum(locked) over a set of (account,asset)
// pairs — the conservation quantity. We sum the explicit set the test cares about
// (the harness knows every participant), which is exact for a closed test economy.
func totalLedger(t *testing.T, db Store, accts []AccountID, assets []AssetID) map[AssetID]uint64 {
	t.Helper()
	out := map[AssetID]uint64{}
	for _, a := range accts {
		for _, asset := range assets {
			av, err := GetAvailable(db, a, asset)
			if err != nil {
				t.Fatalf("GetAvailable: %v", err)
			}
			lk, err := GetLocked(db, a, asset)
			if err != nil {
				t.Fatalf("GetLocked: %v", err)
			}
			out[asset] += av + lk
		}
	}
	return out
}

// orderBookRouter returns a router with only the OrderBook source (the common case).
func orderBookRouter() *Router { return NewRouter(NewOrderBookSource()) }

// price builds a PriceInt from a human price (quote per base).
func price(p float64) lx.PriceInt { return PriceToInt(p) }
