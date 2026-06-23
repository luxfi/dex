// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import (
	"math/big"
)

// journal.go is the JOINT JOURNAL — the single record of BOTH execution surfaces'
// value moves for one swap, so the swap commits both-or-neither in ONE consensus-
// finalized block.
//
// THE MODEL (one consensus domain, two execution surfaces over one state machine):
//   - The D surface (the native DEX core): book rows, fills, and the custody ledger
//     (available/locked). These are written DIRECTLY to the Store during routing —
//     in dchain the Store is a versiondb overlay, in the precompile it is the 0x9999
//     EVM-storage namespace. Either way the D writes are part of the SAME state the
//     block commits.
//   - The C surface (the EVM-adapter): the taker's EVM token/native balance and the
//     0x9999 vault pots. These cannot be written by dexcore directly (dexcore is
//     store-agnostic and knows nothing of EVM balances), so the router RECORDS the
//     required C-side deltas here, and the CALLER (the precompile's 0x9999 swap Run)
//     applies them to the EVM StateDB in the SAME Run.
//
// Atomicity is then total and free: the D writes and the recorded C deltas are
// applied within one EVM call covered by ONE EVM snapshot. A revert anywhere — a
// price-limit breach, a min-out breach, a forced error — rolls back BOTH the D
// writes (the snapshot reverts the StateDB the Store is backed by) AND the C
// deltas (never applied, or reverted by the same snapshot). There is no second
// store and no async shared-memory leg to keep in sync. The "two domains, two
// commit semantics" hazard the cross-process async path had to engineer around
// simply does not exist here.

// CDeltaKind tags a recorded C-surface value move so the caller applies it to the
// correct EVM account/pot. dexcore stays store-agnostic by naming WHAT must move,
// not HOW (which is EVM-specific and lives in the precompile).
type CDeltaKind uint8

const (
	// CTakerDebitInput: debit the taker's input asset from their spendable EVM
	// balance INTO the 0x9999 vault (the lock leg). Applied before/at routing.
	CTakerDebitInput CDeltaKind = iota
	// CTakerCreditOutput: credit the taker's output asset from the 0x9999 vault to
	// their spendable EVM balance (the proceeds leg).
	CTakerCreditOutput
	// CTakerRefundInput: refund unspent input from the 0x9999 vault back to the
	// taker's spendable EVM balance (the IOC remainder).
	CTakerRefundInput
)

// CDelta is one recorded C-surface value move: which kind, which asset, how much.
// The amount is a non-negative magnitude; the kind fixes the direction. The asset
// is the FULL 32-byte injective id (the caller maps it back to the EVM token
// address / native).
type CDelta struct {
	Kind   CDeltaKind
	Asset  AssetID
	Amount *big.Int
}

// Journal accumulates the C-surface deltas for one swap. The D-surface writes go
// straight to the Store during routing (they ARE the block state), so the journal
// only carries the C side the caller must mirror onto EVM balances.
//
// The journal is APPEND-ONLY during routing and CONSUMED once at the end of the
// swap Run. It holds no Store reference and performs no I/O — it is a plain value
// the router fills and the caller drains, which keeps dexcore free of any EVM
// dependency and makes the C/D split auditable in one place.
type Journal struct {
	cDeltas []CDelta
}

// NewJournal returns an empty joint journal for one swap.
func NewJournal() *Journal { return &Journal{} }

// recordC appends a C-surface delta. amount must be non-negative; a zero amount is
// dropped (no move). Internal to dexcore — sources call the typed helpers below.
func (j *Journal) recordC(kind CDeltaKind, asset AssetID, amount *big.Int) {
	if amount == nil || amount.Sign() <= 0 {
		return
	}
	j.cDeltas = append(j.cDeltas, CDelta{Kind: kind, Asset: asset, Amount: new(big.Int).Set(amount)})
}

// DebitTakerInput records that `amount` of `asset` must be debited from the
// taker's EVM balance into the 0x9999 vault (the lock leg).
func (j *Journal) DebitTakerInput(asset AssetID, amount uint64) {
	j.recordC(CTakerDebitInput, asset, new(big.Int).SetUint64(amount))
}

// CreditTakerOutput records that `amount` of `asset` must be credited from the
// 0x9999 vault to the taker's EVM balance (the proceeds leg).
func (j *Journal) CreditTakerOutput(asset AssetID, amount uint64) {
	j.recordC(CTakerCreditOutput, asset, new(big.Int).SetUint64(amount))
}

// RefundTakerInput records that `amount` of `asset` must be refunded from the
// 0x9999 vault to the taker's EVM balance (the unspent IOC remainder).
func (j *Journal) RefundTakerInput(asset AssetID, amount uint64) {
	j.recordC(CTakerRefundInput, asset, new(big.Int).SetUint64(amount))
}

// CDeltas returns the recorded C-surface deltas in append order for the caller to
// apply to the EVM StateDB. The caller applies them within the same EVM snapshot
// as the D writes, so the whole swap is one atomic both-or-neither transition.
func (j *Journal) CDeltas() []CDelta { return j.cDeltas }

// NetTakerDelta collapses the recorded deltas into the taker's net (input debited,
// output credited, input refunded) per asset, as signed big.Ints keyed by asset.
// A positive value means the taker's spendable balance of that asset INCREASED.
// Used by the conservation assertions and the BalanceDelta return.
func (j *Journal) NetTakerDelta() map[AssetID]*big.Int {
	out := map[AssetID]*big.Int{}
	add := func(asset AssetID, v *big.Int) {
		cur, ok := out[asset]
		if !ok {
			cur = new(big.Int)
			out[asset] = cur
		}
		cur.Add(cur, v)
	}
	for _, d := range j.cDeltas {
		switch d.Kind {
		case CTakerDebitInput:
			add(d.Asset, new(big.Int).Neg(d.Amount))
		case CTakerCreditOutput, CTakerRefundInput:
			add(d.Asset, d.Amount)
		}
	}
	return out
}
