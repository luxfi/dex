// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"fmt"
	"math/big"
	"time"

	"github.com/luxfi/dex/pkg/lx"
)

// place.go is the resting-liquidity path: a maker places a LIMIT order that rests
// on the book (the counterparty a taker's swap crosses). It is the custody-gated
// place from the proven dchain path, now over the shared Store: lock the spend
// asset, rest the order deterministically, persist the row + the per-order reserve
// (orderasset:) + the FULL settlement identity (orderuser:) so a later fill/cancel
// settles to the exact account even after a restart.
//
// Cancel is the inverse: unlock the order's remaining reserve to its recorded full
// identity (fail-closed if the orderuser: row is missing) and delete the rows.

// PlaceOrder rests a maker LIMIT order on the market. maker is the FULL 16-byte
// settlement identity; orderID is the block-deterministic id; ts is the block time.
// The spend asset is locked (buy -> ceil(size*price) quote, sell -> size base) and
// the per-order reserve + identity rows are recorded. Returns the resting order id.
//
// Rejects (deterministic, no error): an unaffordable lock (insufficient available)
// or a zero-notional executable order (dust price / sub-unit size) — the order does
// NOT rest and nothing is locked, signaled by ok=false. A post-only-would-take
// rejection also yields ok=false with the lock refunded.
func PlaceOrder(db Store, poolID [32]byte, maker AccountID, side lx.Side, price, size float64, orderID uint64, ts time.Time) (ok bool, err error) {
	base, quote, bound, err := ReadMarketAssets(db, poolID)
	if err != nil {
		return false, err
	}
	if !bound {
		return false, ErrUnknownMarket
	}
	if !(price > 0) || !(size > 0) {
		return false, nil // malformed: not an error, just no rest
	}
	// Custody gate: an executable order whose notional floors to zero lock is the
	// "free executable order" hazard — reject it.
	if FloorsToZeroLock(side, price, size, base, quote) {
		return false, ErrDustPriceOrZeroLock
	}
	lockAsset, lockAmt := OrderLock(side, price, size, base, quote)
	if lockAmt == 0 {
		return false, ErrDustPriceOrZeroLock
	}
	// Lock the spend. Insufficient available => deterministic reject (no rest).
	if err := LockFromAvailable(db, maker, lockAsset, lockAmt); err != nil {
		if err == ErrInsufficientAvailable {
			return false, nil
		}
		return false, err
	}

	// Rest the order via the deterministic consensus entrypoint, over a rebuilt book.
	symbol, exists, err := ReadMarketSymbol(db, poolID)
	if err != nil {
		return false, err
	}
	if !exists {
		return false, ErrUnknownMarket
	}
	book, err := RebuildBook(db, poolID, symbol)
	if err != nil {
		return false, err
	}
	order := &lx.Order{
		ID:        orderID,
		Type:      lx.Limit,
		Side:      side,
		Price:     price,
		Size:      size,
		SizeUnits: new(big.Int).SetUint64(SizeToUnits(size)),
		User:      matcherHint(maker),
		UserID:    matcherHint(maker),
		Symbol:    symbol,
		Timestamp: ts,
	}
	if book.ConsensusAddOrder(order) == 0 {
		// Rejected (e.g. post-only would take): refund the lock, rest nothing.
		if uerr := UnlockToAvailable(db, maker, lockAsset, lockAmt); uerr != nil {
			return false, uerr
		}
		return false, nil
	}

	// Persist the resting row + the per-order reserve + the full settlement identity.
	row := lx.OrderToRow(order)
	row.OrderID = order.ID
	if err := PutOrderRow(db, poolID, row); err != nil {
		return false, err
	}
	if err := PutOrderLock(db, poolID, orderID, lockAsset, lockAmt); err != nil {
		return false, err
	}
	if err := PutOrderUser(db, poolID, orderID, maker); err != nil {
		return false, err
	}
	if err := WriteBookWatermark(db, poolID, book.LastOrderID); err != nil {
		return false, err
	}
	return true, nil
}

// CancelOrder removes a resting order and unlocks its remaining reserve to the
// order's recorded FULL settlement identity (orderuser:) — the ONLY source. A
// cancel of an order that holds a non-zero reserve but has NO orderuser: row is
// state corruption: FAIL CLOSED (ErrMissingSettlementUser), never unlock to a
// compact-handle-derived account. A cancel of an unknown order is a deterministic
// no-op (ok=false).
func CancelOrder(db Store, poolID [32]byte, orderID uint64) (ok bool, err error) {
	symbol, exists, err := ReadMarketSymbol(db, poolID)
	if err != nil {
		return false, err
	}
	if !exists {
		return false, nil
	}
	book, err := RebuildBook(db, poolID, symbol)
	if err != nil {
		return false, err
	}
	if book.GetOrder(orderID) == nil {
		return false, nil // unknown order: no-op
	}
	if cerr := book.CancelOrder(orderID); cerr != nil {
		return false, nil // matcher reject: no-op
	}

	// Unlock the remaining reserve to the recorded full identity.
	asset, amt, hasLock, err := GetOrderLock(db, poolID, orderID)
	if err != nil {
		return false, err
	}
	if hasLock && amt > 0 {
		owner, hasUser, uerr := GetOrderUser(db, poolID, orderID)
		if uerr != nil {
			return false, uerr
		}
		if !hasUser {
			return false, fmt.Errorf("%w: cancel of order %d in market %x", ErrMissingSettlementUser, orderID, poolID[:8])
		}
		if err := UnlockToAvailable(db, owner, asset, amt); err != nil {
			return false, err
		}
	}
	// Delete the persisted row + the per-order reserve + the identity row.
	if err := DeleteOrderRow(db, poolID, orderID); err != nil {
		return false, err
	}
	if err := PutOrderLock(db, poolID, orderID, asset, 0); err != nil {
		return false, err
	}
	if err := DeleteOrderUser(db, poolID, orderID); err != nil {
		return false, err
	}
	if err := WriteBookWatermark(db, poolID, book.LastOrderID); err != nil {
		return false, err
	}
	return true, nil
}

// Deposit credits a maker/taker's available balance for an asset — the on-ramp the
// custody ledger draws from. In the precompile this mirrors a real ERC-20/native
// lock in the 0x9999 vault; here it is the ledger-side credit. The asset is the
// FULL injective id.
func Deposit(db Store, account AccountID, asset AssetID, amount uint64) error {
	return CreditAvailable(db, account, asset, amount)
}

// OpenMarket binds a market's (base,quote) assets and records its existence, so the
// custody gate value-checks orders. Idempotent.
//
// NOTE: OpenMarket binds whatever AssetIDs it is given — it does NOT verify the assets
// are registered real on-chain assets. It is the low-level binding primitive used by
// tests and by callers that have ALREADY admitted the assets. The value path (the
// 0x9999 cEVM router) MUST use OpenMarketChecked, which gates admission through the
// AssetResolver so a left-padded address is not sufficient to create a market.
func OpenMarket(db Store, poolID [32]byte, base, quote AssetID) error {
	if exists, err := MarketExists(db, poolID); err != nil {
		return err
	} else if !exists {
		if err := WriteMarket(db, poolID, PoolSymbol(poolID)); err != nil {
			return err
		}
	}
	return WriteMarketAssets(db, poolID, base, quote)
}

// AssetSide names one side of a market by its real on-chain coordinates: the asset
// kind (EVM_NATIVE | ERC20 | UTXO) and the canonical reference (the native marker, the
// 20-byte token address, or the 32-byte UTXO assetID). It is what the value path knows
// at swap time (the V4 currency address -> kind+ref) and what the resolver admits over.
type AssetSide struct {
	Kind AssetKind
	Ref  []byte
}

// valueKey is the side's CUSTODY/LEDGER value key — the established left-padded asset id
// the vault, the ledger (LockFromAvailable/Deposit), and the journal key value by. For an
// ERC-20 (20-byte ref) it is the address left-padded into 32 bytes (== the precompile's
// assetID); for the native marker (20 zero bytes) it is the all-zero native key; for a
// UTXO (32-byte ref) it is the ref itself. This is DELIBERATELY distinct from the
// canonical DeriveAssetID used for ADMISSION: identity-for-admission and
// identity-for-value-keying are orthogonal. Admission decides "is this a real, enabled
// asset?" (DeriveAssetID, registry-shared, domain-separated); value keying is the proven
// conservation key the whole custody path already uses, and C1 must not disturb it.
func (s AssetSide) valueKey() AssetID {
	var k AssetID
	if len(s.Ref) >= 32 {
		copy(k[:], s.Ref[len(s.Ref)-32:])
		return k
	}
	copy(k[32-len(s.Ref):], s.Ref) // left-pad: 20-byte address -> [12:], native marker -> all-zero
	return k
}

// OpenMarketChecked is the PERMISSIONLESS value-path market-open: it ADMITS both sides
// through TWO orthogonal steps before binding, so any market over two REAL assets is
// created permissionlessly, while a synthetic/fabricated asset is refused. There is NO
// allowlist and NO "approved-by-admin": OpenMarket(base, quote) resolves each side's
// canonical identity, proves each side's live on-chain reality, derives the market, and
// creates it if absent. Any real (base, quote) pair yields a market.
//
// THE EXACT STEP ORDER (the value-path call chain, run BEFORE any C-balance debit and
// BEFORE any DEX state write):
//
//	Resolve(base) -> Resolve(quote) -> VerifyOnChain(base) -> VerifyOnChain(quote) -> EnsureMarket
//
//   - Resolve (RequireResolvedAsset): prove each side's CANONICAL IDENTITY on the node's
//     bound network — PERMISSIONLESS. A well-formed real EVM-native/ERC-20 reference always
//     resolves; only a malformed/wrong-network/explicitly-disabled/synthetic reference is
//     refused (ErrAssetNotResolved). No resolver => ErrNoAssetResolver (fail-closed).
//   - VerifyOnChain (RequireRealOnChain): the AUTHORITATIVE reality check — is each side
//     backed by LIVE on-chain code in the state this block mutates? A fabricated/synthetic
//     ERC-20 (an ASCII-ticker address, a self-destructed or never-deployed contract) has no
//     code => ErrAssetNotOnChain; no verifier => ErrNoOnChainVerifier. Native (the
//     zero-address marker) is the chain's own coin and is always real.
//
// Both steps run for BOTH sides before ANY write, so a refused open leaves the store
// untouched (fail-closed): the market existence row is created only after all four checks
// pass. On success it binds the market to the established left-padded VALUE KEYS (the same
// ids the vault/ledger/journal key value by), NOT the canonical admission id — so admission
// proves reality WITHOUT disturbing the proven conservation keying. Idempotent (create-if-
// absent).
func OpenMarketChecked(db Store, resolver AssetResolver, verifier OnChainAssetVerifier, poolID [32]byte, base, quote AssetSide) error {
	// (1) Resolve both sides' canonical identity (permissionless: well-formed real ref on
	//     the bound network resolves; malformed/wrong-network/disabled/synthetic is refused).
	if _, _, err := RequireResolvedAsset(resolver, base.Kind, base.Ref); err != nil {
		return err
	}
	if _, _, err := RequireResolvedAsset(resolver, quote.Kind, quote.Ref); err != nil {
		return err
	}
	// (2) Verify both sides against live on-chain reality (the AUTHORITATIVE check: a
	//     fabricated/synthetic asset has no contract code and is refused here).
	if err := RequireRealOnChain(verifier, base.Kind, base.Ref); err != nil {
		return err
	}
	if err := RequireRealOnChain(verifier, quote.Kind, quote.Ref); err != nil {
		return err
	}
	// (3) Only now bind the market (the FIRST and ONLY state write of this admission).
	return OpenMarket(db, poolID, base.valueKey(), quote.valueKey())
}
