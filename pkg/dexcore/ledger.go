// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package dexcore is the SHARED, caller-agnostic deterministic core of the Lux
// DEX: the CLOB custody ledger, the integer-exact settlement of matcher fills,
// the fail-closed maker-identity discipline, the smart-order-router across
// on-chain liquidity sources, and the execution root. It is a pure function of
// (ordered txs, prior state) over a generic key/value Store, so two homes —
//
//   - the standalone D-Chain VM (dex/pkg/dchain), over a versiondb overlay, and
//   - the in-process 0x9999 cEVM precompile (precompile/dex), over the 0x9999
//     EVM-storage namespace —
//
// run the IDENTICAL settlement and routing logic and derive byte-identical
// state. There is exactly one settlement algorithm and one router, and it lives
// here. No clock, no process-local counter, no float touches a value: every
// number that participates in a value move is derived from block context and the
// matcher's authoritative integer lane.
//
// ledger.go is the custody schema + the conserving balance primitives. It is the
// part that makes "the money lives in the order book": a deposit credits
// available; a place/submit moves available -> locked for the asset the order
// spends; a fill moves locked between maker and taker; a cancel/refund returns
// locked -> available; a withdraw debits available. The global invariant is
// sum(available)+sum(locked) over an (account,asset) is constant across the trade
// lifecycle, because every unit a fill removes from one account's locked is added
// to exactly one other account's available — checked integer arithmetic, no
// float, no silent overflow.
package dexcore

import (
	"encoding/binary"
	"fmt"

	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/zapwire"
)

// Store is the key/value surface dexcore reads and writes its state over. It is
// exactly database's read/write/delete + prefix-iteration contract, which both
// homes already satisfy: dchain hands it a versiondb overlay, the precompile
// hands it a 0x9999-EVM-storage adapter. dexcore NEVER assumes a concrete store;
// the only requirements are the ones named here.
type Store interface {
	database.KeyValueReaderWriterDeleter
	database.Iteratee
}

// AssetID is the FULL 32-byte injective asset identity the ledger keys value by
// — NEVER a truncated handle. For an on-chain ERC-20 it is the left-pad of the
// 20-byte token address (precompile/dex/asset.go assetID); native LUX (the zero
// address) is the all-zero id. Keeping the asset at full width makes two distinct
// tokens — or a token vs native — ALWAYS distinct ledger rows (the native-vault-
// drain fix on the asset axis).
type AssetID = [32]byte

// AccountID is the FULL 16-byte injective wire identity the ledger keys an
// account by — NEVER a folded 8-byte handle. Keeping the account at full width
// makes two addresses sharing a leading 8-byte prefix DISTINCT accounts so
// neither can withdraw the other's balance (the cross-user native-drain fix on
// the account axis). It is byte-identical to the wire user[16] field.
type AccountID = [zapwire.UserSize]byte

// AccountFromString folds a user-identity string to the canonical 16-byte ledger
// identity: the leading 16 bytes, zero-padded — byte-identical to the wire's
// user[16] null-padding so the account the ledger credits is the SAME account the
// orders/deposits name. This is the ONE fold from a user string to the settlement
// identity; there is no 8-byte fold anywhere on the value path.
func AccountFromString(user string) AccountID {
	var k AccountID
	copy(k[:], user)
	return k
}

// KEY SCHEMA — the durable ledger rows. Identical in both homes; the precompile
// maps each []byte key to a keccak-derived 0x9999 storage slot, dchain writes it
// to versiondb directly. BOTH halves of a balance key are full injective width.
//
//	balance:<user:16><asset:32> -> uint64  AVAILABLE (deposited, un-escrowed)
//	locked:<user:16><asset:32>  -> uint64  LOCKED    (reserved by live orders)
//	asset:<poolID:32>           -> base[32]quote[32] market's (base,quote) ids
//	orderasset:<poolID:32><orderID:8> -> assetID[32]|amount[8] per-order reserve
//	orderuser:<poolID:32><orderID:8>  -> user[16] resting order's FULL identity
const (
	prefixBalance   = "balance:"
	prefixLocked    = "locked:"
	prefixAsset     = "asset:"      // market (base,quote) asset binding
	prefixOrderLock = "orderasset:" // per-resting-order locked (asset,amount)
	prefixOrderUser = "orderuser:"  // per-resting-order FULL 16-byte identity
)

// balanceLedgerKey builds <prefix><user:16><asset:32>. Both the user (full
// 16-byte injective wire identity) and the asset (full 32-byte injective id) are
// kept at full width, so no two distinct accounts and no two distinct assets ever
// collide on a balance key.
func balanceLedgerKey(prefix string, user AccountID, asset AssetID) []byte {
	k := make([]byte, len(prefix)+len(user)+len(asset))
	copy(k, prefix)
	copy(k[len(prefix):], user[:])
	copy(k[len(prefix)+len(user):], asset[:])
	return k
}

// readU64 reads an 8-byte big-endian value, returning 0 when absent (an account
// with no row has zero balance — the natural ledger default).
func readU64(db database.KeyValueReader, key []byte) (uint64, error) {
	v, err := db.Get(key)
	if err == database.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if len(v) != 8 {
		return 0, fmt.Errorf("dexcore: corrupt ledger value len=%d", len(v))
	}
	return binary.BigEndian.Uint64(v), nil
}

// writeU64 persists an 8-byte big-endian value, deleting the row when it reaches
// zero (a zero balance is the absence of a row — keeps the leaf set minimal).
func writeU64(db database.KeyValueWriterDeleter, key []byte, v uint64) error {
	if v == 0 {
		return db.Delete(key)
	}
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	return db.Put(key, b[:])
}

// GetAvailable / GetLocked return an account's available / locked balance for an
// asset (keyed by the full 16-byte user and the full 32-byte asset id).
func GetAvailable(db database.KeyValueReader, user AccountID, asset AssetID) (uint64, error) {
	return readU64(db, balanceLedgerKey(prefixBalance, user, asset))
}

// GetLocked returns an account's locked balance for an asset.
func GetLocked(db database.KeyValueReader, user AccountID, asset AssetID) (uint64, error) {
	return readU64(db, balanceLedgerKey(prefixLocked, user, asset))
}

// CreditAvailable adds amount to an account's available balance (the deposit
// primitive and the fill-proceeds primitive). Refuses overflow — a credit that
// would wrap uint64 is a corrupt input, never silently truncated.
func CreditAvailable(db Store, user AccountID, asset AssetID, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := GetAvailable(db, user, asset)
	if err != nil {
		return err
	}
	if cur+amount < cur {
		return fmt.Errorf("dexcore: available credit overflow user=%x asset=%x", user, asset)
	}
	return writeU64(db, balanceLedgerKey(prefixBalance, user, asset), cur+amount)
}

// DebitAvailable removes amount from an account's available balance. Returns
// ErrInsufficientAvailable when the account lacks the balance — a debit draws
// ONLY available (deposited, un-escrowed) funds, so an over-debit is a hard
// refusal, never an underflow.
func DebitAvailable(db Store, user AccountID, asset AssetID, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := GetAvailable(db, user, asset)
	if err != nil {
		return err
	}
	if cur < amount {
		return fmt.Errorf("%w: user=%x asset=%x have=%d need=%d", ErrInsufficientAvailable, user, asset, cur, amount)
	}
	return writeU64(db, balanceLedgerKey(prefixBalance, user, asset), cur-amount)
}

// LockFromAvailable moves amount from available to locked (placing/submitting an
// order reserves the asset it will spend). All-or-nothing: the debit and the lock
// are written together, so a partial lock can never strand value.
func LockFromAvailable(db Store, user AccountID, asset AssetID, amount uint64) error {
	if amount == 0 {
		return nil
	}
	if err := DebitAvailable(db, user, asset, amount); err != nil {
		return err
	}
	cur, err := GetLocked(db, user, asset)
	if err != nil {
		return err
	}
	if cur+amount < cur {
		return fmt.Errorf("dexcore: locked credit overflow user=%x asset=%x", user, asset)
	}
	return writeU64(db, balanceLedgerKey(prefixLocked, user, asset), cur+amount)
}

// UnlockToAvailable moves amount from locked back to available (cancel, reject,
// or the unspent remainder of a partially-filled order). Refuses to unlock more
// than is locked — that would mint against the ledger.
func UnlockToAvailable(db Store, user AccountID, asset AssetID, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := GetLocked(db, user, asset)
	if err != nil {
		return err
	}
	if cur < amount {
		return fmt.Errorf("%w: unlock user=%x asset=%x locked=%d need=%d", ErrInsufficientLocked, user, asset, cur, amount)
	}
	if err := writeU64(db, balanceLedgerKey(prefixLocked, user, asset), cur-amount); err != nil {
		return err
	}
	return CreditAvailable(db, user, asset, amount)
}

// SpendLocked removes amount from an account's locked balance WITHOUT crediting
// it back — used on a fill, where the spender's locked asset LEAVES the spender
// and is credited (by CreditAvailable) to the COUNTERPARTY's available. The pair
// (SpendLocked from spender + CreditAvailable to counterparty) is what conserves
// value across a fill: every unit removed here is added to exactly one other
// account, so sum(available)+sum(locked) is invariant. Refuses spend > locked.
func SpendLocked(db Store, user AccountID, asset AssetID, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := GetLocked(db, user, asset)
	if err != nil {
		return err
	}
	if cur < amount {
		return fmt.Errorf("%w: spend user=%x asset=%x locked=%d need=%d", ErrInsufficientLocked, user, asset, cur, amount)
	}
	return writeU64(db, balanceLedgerKey(prefixLocked, user, asset), cur-amount)
}

// ---- market (base,quote) asset binding ----

// assetKey builds asset:<poolID:32>.
func assetKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixAsset)+32)
	copy(k, prefixAsset)
	copy(k[len(prefixAsset):], poolID[:])
	return k
}

// WriteMarketAssets binds a market's (base,quote) asset ids (idempotent across
// re-opens with the same pair). The value is base[32]|quote[32] — the FULL
// injective ids the ledger keys balance:/locked: by, so an order's locked spend
// names the SAME asset key a deposit credited.
func WriteMarketAssets(db database.KeyValueWriter, poolID [32]byte, base, quote AssetID) error {
	var v [64]byte
	copy(v[0:32], base[:])
	copy(v[32:64], quote[:])
	return db.Put(assetKey(poolID), v[:])
}

// ReadMarketAssets returns a market's (base,quote) ids, ok=false if the market's
// assets were never bound.
func ReadMarketAssets(db database.KeyValueReader, poolID [32]byte) (base, quote AssetID, ok bool, err error) {
	v, gerr := db.Get(assetKey(poolID))
	if gerr == database.ErrNotFound {
		return base, quote, false, nil
	}
	if gerr != nil {
		return base, quote, false, gerr
	}
	if len(v) != 64 {
		return base, quote, false, fmt.Errorf("dexcore: corrupt market assets len=%d", len(v))
	}
	copy(base[:], v[0:32])
	copy(quote[:], v[32:64])
	return base, quote, true, nil
}

// ---- per-order reserve (so cancel/fill can unlock the exact remaining lock) ----

// orderLockKey builds orderasset:<poolID:32><orderID:8>.
func orderLockKey(poolID [32]byte, orderID uint64) []byte {
	k := make([]byte, len(prefixOrderLock)+32+8)
	copy(k, prefixOrderLock)
	copy(k[len(prefixOrderLock):], poolID[:])
	binary.BigEndian.PutUint64(k[len(prefixOrderLock)+32:], orderID)
	return k
}

// PutOrderLock records the (asset, remaining-locked-amount) a resting order
// holds, so a later cancel or a maker fill knows exactly how much to unlock /
// spend. The asset is the FULL 32-byte id; the value is asset[32]|amount[8].
// Deleted when the remaining lock reaches zero.
func PutOrderLock(db database.KeyValueWriterDeleter, poolID [32]byte, orderID uint64, asset AssetID, amount uint64) error {
	if amount == 0 {
		return db.Delete(orderLockKey(poolID, orderID))
	}
	var v [40]byte
	copy(v[0:32], asset[:])
	binary.BigEndian.PutUint64(v[32:40], amount)
	return db.Put(orderLockKey(poolID, orderID), v[:])
}

// GetOrderLock returns the (asset, remaining-locked-amount) for a resting order,
// ok=false if none recorded.
func GetOrderLock(db database.KeyValueReader, poolID [32]byte, orderID uint64) (asset AssetID, amount uint64, ok bool, err error) {
	v, gerr := db.Get(orderLockKey(poolID, orderID))
	if gerr == database.ErrNotFound {
		return asset, 0, false, nil
	}
	if gerr != nil {
		return asset, 0, false, gerr
	}
	if len(v) != 40 {
		return asset, 0, false, fmt.Errorf("dexcore: corrupt order lock len=%d", len(v))
	}
	copy(asset[:], v[0:32])
	return asset, binary.BigEndian.Uint64(v[32:40]), true, nil
}

// ---- per-order FULL settlement identity ----
//
// The matcher's persisted DEXOrder row carries only an 8-byte UserID (the frozen
// GPU ABI / STP handle), so a resting order rebuilt from a row (RowToOrder) loses
// bytes 8..15 of its user. Settlement, however, MUST credit the maker's FULL
// 16-byte account (the one its deposit/lock live under). This row carries that
// full identity, keyed by orderID — orthogonal to the GPU row — so a maker fill
// (live OR post-restart) and a cancel settle to the exact account.

// orderUserKey builds orderuser:<poolID:32><orderID:8>.
func orderUserKey(poolID [32]byte, orderID uint64) []byte {
	k := make([]byte, len(prefixOrderUser)+32+8)
	copy(k, prefixOrderUser)
	copy(k[len(prefixOrderUser):], poolID[:])
	binary.BigEndian.PutUint64(k[len(prefixOrderUser)+32:], orderID)
	return k
}

// PutOrderUser records a resting order's FULL 16-byte settlement identity.
func PutOrderUser(db database.KeyValueWriter, poolID [32]byte, orderID uint64, user AccountID) error {
	return db.Put(orderUserKey(poolID, orderID), user[:])
}

// DeleteOrderUser removes a resting order's settlement-identity row.
func DeleteOrderUser(db database.KeyValueDeleter, poolID [32]byte, orderID uint64) error {
	return db.Delete(orderUserKey(poolID, orderID))
}

// GetOrderUser returns a resting order's FULL 16-byte settlement identity,
// ok=false if none recorded. On a CUSTODY (assets-bound) market the settlement
// callers (the maker settle key, the cancel unlock) FAIL CLOSED with
// ErrMissingSettlementUser when ok=false — there is NO fallback to the matcher's
// 8-byte handle, because a compact-handle fallback would make handle collisions
// value-bearing (a theft bug).
func GetOrderUser(db database.KeyValueReader, poolID [32]byte, orderID uint64) (user AccountID, ok bool, err error) {
	v, gerr := db.Get(orderUserKey(poolID, orderID))
	if gerr == database.ErrNotFound {
		return user, false, nil
	}
	if gerr != nil {
		return user, false, gerr
	}
	if len(v) != len(user) {
		return user, false, fmt.Errorf("dexcore: corrupt order user len=%d", len(v))
	}
	copy(user[:], v)
	return user, true, nil
}
