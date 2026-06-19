// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/binary"
	"fmt"
	"sort"

	"github.com/luxfi/crypto/hash"
	"github.com/luxfi/crypto/merkle"
	"github.com/luxfi/database"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// state.go is the d-chain key/value schema and the state root. The schema lives
// in the VM's init.DB namespace; the proxy (chains/dexvm) holds NONE of these
// keys — proxy is transport, d-chain is state.
//
// KEY SCHEMA (all values are fixed-width images from pkg/lx/persist.go):
//
//	order:<poolID:32><orderID:8>  -> DEXOrder (64B)   resting liquidity
//	book:<poolID:32>              -> uint64 LastOrderID watermark for the market
//	market:<poolID:32>           -> symbol bytes      market existence + symbol
//	trade:<height:8><seq:8>       -> DEXTrade (80B)    append-only fill log
//	meta:lastAccepted            -> ids.ID (32B)      last accepted block id
//	meta:height                  -> uint64 (8B)       last accepted height
//	meta:root                    -> [32]byte          last execution root
//
// The order:* rows are sufficient to rebuild every book on restart (the in-RAM
// lx.OrderBook is a fold of these). trade:* is the durable event log. The roots
// are derived, not stored as truth — meta:root caches the parent root for the
// next block's chained composition.
const (
	prefixOrder  = "order:"
	prefixBook   = "book:"
	prefixMarket = "market:"
	prefixTrade  = "trade:"
	prefixSeen   = "seen:" // processed tx ids (idempotency / replay dedup)
	prefixMeta   = "meta:"

	// The CLOB custody ledger: where the money LIVES inside the d-chain. A
	// deposit credits balance:; placing/submitting an order moves available ->
	// locked: for the asset the order spends; a fill moves locked: between maker
	// and taker; a cancel/reject returns locked: -> balance:; a withdraw debits
	// balance:. The conservation invariant is global: sum(balance:) + sum(locked:)
	// over an (account,asset) is constant across deposit -> trade -> withdraw
	// because every unit a fill removes from one account's locked is added to
	// another account's available (no float, exact integer units).
	//
	//	balance:<user:16><asset:32> -> uint64  AVAILABLE (deposited, un-escrowed)
	//	locked:<user:16><asset:32>  -> uint64  LOCKED    (reserved by live orders)
	//	asset:<poolID:32>           -> base[32]quote[32]   market's (base,quote) ids
	//	orderasset:<poolID:32><orderID:8> -> assetID[32]|amount[8]  per-order reserve
	//	orderuser:<poolID:32><orderID:8>  -> user[16]  resting order's FULL identity
	//
	// BOTH halves of a ledger key are kept at FULL injective width:
	//   - The asset is the FULL 32-byte injective id (zapwire.AssetIDSize), NOT a
	//     truncated handle: keying a value-bearing ledger by a truncated asset is
	//     unsound (a token whose id folds to the native key could mint a native
	//     claim and drain the native vault; two tokens sharing the truncated prefix
	//     would share balances).
	//   - The user is the FULL 16-byte injective wire identity (zapwire.UserSize),
	//     NOT a folded 8-byte handle: keying the ledger by a truncated user is the
	//     SAME unsoundness on the account axis — two addresses sharing a leading
	//     8-byte prefix would share balances, letting one withdraw the other's
	//     deposit (the cross-user native-drain fix). The matcher's DEXOrder.UserID
	//     stays an 8-byte STP handle (the frozen GPU ABI); the SETTLEMENT identity
	//     that the ledger keys by is this full 16-byte user, decomplected from the
	//     matching handle exactly as pkg/lx/persist.go declares.
	prefixBalance   = "balance:"
	prefixLocked    = "locked:"
	prefixAsset     = "asset:"      // market (base,quote) asset binding
	prefixOrderLock = "orderasset:" // per-resting-order locked (asset,amount) for unlock on cancel/fill
	prefixOrderUser = "orderuser:"  // per-resting-order FULL 16-byte settlement identity

	metaLastAccepted = prefixMeta + "lastAccepted"
	metaHeight       = prefixMeta + "height"
	metaRoot         = prefixMeta + "root"
)

// seenKey builds seen:<txID:32>. A tx id committed here has been applied by an
// accepted block; re-applying it is a no-op. This is the d-chain STATE
// idempotency index: the frozen zapwire frame carries no nonce, so dedup is
// content-addressed on the tx id (which subsumes the wire user[16] and the full
// payload). A relay that retries a dropped clob_submit re-sends the identical
// bytes -> identical tx id -> deduped, exactly once.
func seenKey(txID ids.ID) []byte {
	k := make([]byte, len(prefixSeen)+32)
	copy(k, prefixSeen)
	copy(k[len(prefixSeen):], txID[:])
	return k
}

// isSeen reports whether a tx id has already been applied by an accepted block.
func isSeen(db database.KeyValueReader, txID ids.ID) (bool, error) {
	_, err := db.Get(seenKey(txID))
	if err == database.ErrNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// markSeen records a tx id as applied (written into the block's overlay so it
// commits atomically with the tx's effects). The stored value is the tx's
// ORIGINAL outcome (status + orderID + fills) so a later replay of the same
// content-addressed tx returns the FIRST result verbatim rather than a blanket
// reject — the idempotency contract a caller (precompile / proxy) needs: a
// retried clob_place returns its original Placed+orderID, a retried clob_submit
// returns its original fills, never a spurious "rejected". (A bare presence
// byte made a deduped place indistinguishable from a genuine rejection, which
// reverted the EVM tx whose precompile re-ran the place across the host's
// multiple executions of one transaction.)
func markSeen(db database.KeyValueWriter, txID ids.ID, o txOutcome) error {
	return db.Put(seenKey(txID), encodeSeenOutcome(o))
}

// getSeenOutcome returns the recorded outcome for an already-applied tx id, with
// ok=false if the tx has not been seen. The returned outcome carries the tx id
// and type from the replayed tx (the stored value holds only status/orderID/
// fills, which is all a replay needs).
func getSeenOutcome(db database.KeyValueReader, txID ids.ID, typ TxType) (txOutcome, bool, error) {
	v, err := db.Get(seenKey(txID))
	if err == database.ErrNotFound {
		return txOutcome{}, false, nil
	}
	if err != nil {
		return txOutcome{}, false, err
	}
	o := decodeSeenOutcome(v)
	o.txID = txID
	o.typ = typ
	return o, true, nil
}

// encodeSeenOutcome serializes the replay-relevant fields: status[1] |
// orderID[8] | EncodeFills(fills). EncodeFills self-delimits with a leading
// count, so the value parses unambiguously.
func encodeSeenOutcome(o txOutcome) []byte {
	out := make([]byte, 9)
	out[0] = o.status
	binary.BigEndian.PutUint64(out[1:9], o.orderID)
	if len(o.fills) > 0 {
		out = append(out, zapwire.EncodeFills(o.fills)...)
	}
	return out
}

// decodeSeenOutcome inverts encodeSeenOutcome. A legacy 1-byte presence value
// (or any value shorter than the 9-byte header) decodes to a Placed status with
// no orderID/fills — backward-safe for any pre-existing seen marks.
func decodeSeenOutcome(v []byte) txOutcome {
	if len(v) < 9 {
		return txOutcome{status: zapwire.StatusPlaced}
	}
	o := txOutcome{status: v[0], orderID: binary.BigEndian.Uint64(v[1:9])}
	if len(v) > 9 {
		if fills, err := zapwire.DecodeFills(v[9:]); err == nil {
			o.fills = fills
		}
	}
	return o
}

// orderKey builds order:<poolID:32><orderID:8>.
func orderKey(poolID [32]byte, orderID uint64) []byte {
	k := make([]byte, len(prefixOrder)+32+8)
	copy(k, prefixOrder)
	copy(k[len(prefixOrder):], poolID[:])
	binary.BigEndian.PutUint64(k[len(prefixOrder)+32:], orderID)
	return k
}

// orderPrefixFor builds the order:<poolID:32> iteration prefix for one market.
func orderPrefixFor(poolID [32]byte) []byte {
	k := make([]byte, len(prefixOrder)+32)
	copy(k, prefixOrder)
	copy(k[len(prefixOrder):], poolID[:])
	return k
}

// marketKey builds market:<poolID:32>.
func marketKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixMarket)+32)
	copy(k, prefixMarket)
	copy(k[len(prefixMarket):], poolID[:])
	return k
}

// bookKey builds book:<poolID:32> (the per-market LastOrderID watermark).
func bookKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixBook)+32)
	copy(k, prefixBook)
	copy(k[len(prefixBook):], poolID[:])
	return k
}

// tradeKey builds trade:<height:8><seq:8>.
func tradeKey(height, seq uint64) []byte {
	k := make([]byte, len(prefixTrade)+16)
	copy(k, prefixTrade)
	binary.BigEndian.PutUint64(k[len(prefixTrade):], height)
	binary.BigEndian.PutUint64(k[len(prefixTrade)+8:], seq)
	return k
}

// putOrderRow writes a resting order row, or deletes it when the order is no
// longer live (filled or cancelled or zero remaining). The book is the live
// resting set; spent rows are removed so iteration over order:* yields exactly
// the rebuildable liquidity.
func putOrderRow(db database.KeyValueWriterDeleter, poolID [32]byte, row lx.DEXOrder) error {
	key := orderKey(poolID, row.OrderID)
	if row.Status == lx.DEXStatusFilled || row.Status == lx.DEXStatusCancelled || row.Remaining == 0 {
		return db.Delete(key)
	}
	return db.Put(key, lx.EncodeRow(row))
}

// deleteOrderRow removes a resting order row (used on cancel).
func deleteOrderRow(db database.KeyValueWriterDeleter, poolID [32]byte, orderID uint64) error {
	return db.Delete(orderKey(poolID, orderID))
}

// readMarketSymbol returns the market's symbol, or ok=false if the market does
// not exist.
func readMarketSymbol(db database.KeyValueReader, poolID [32]byte) (string, bool, error) {
	v, err := db.Get(marketKey(poolID))
	if err == database.ErrNotFound {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return string(v), true, nil
}

// writeMarket records a market's existence + symbol (idempotent).
func writeMarket(db database.KeyValueWriter, poolID [32]byte, symbol string) error {
	return db.Put(marketKey(poolID), []byte(symbol))
}

// readBookWatermark returns the market's LastOrderID watermark (0 if absent).
func readBookWatermark(db database.KeyValueReader, poolID [32]byte) (uint64, error) {
	v, err := db.Get(bookKey(poolID))
	if err == database.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if len(v) != 8 {
		return 0, fmt.Errorf("dchain: corrupt book watermark len=%d", len(v))
	}
	return binary.BigEndian.Uint64(v), nil
}

// writeBookWatermark persists the market's LastOrderID watermark.
func writeBookWatermark(db database.KeyValueWriter, poolID [32]byte, watermark uint64) error {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], watermark)
	return db.Put(bookKey(poolID), b[:])
}

// =========================================================================
// Custody ledger — the per-account (available, locked) balances the order book
// draws from, plus the market (base,quote) asset binding and per-order reserve.
// All amounts are uint64 atomic asset units; all arithmetic is checked integer
// (no float, no silent overflow), so the conservation invariant holds to the
// unit.
// =========================================================================

// userKey is the d-chain's SETTLEMENT identity: the FULL 16-byte wire user
// (zapwire.UserSize), the value the ledger keys an account by. It is DISTINCT from
// the matcher's 8-byte DEXOrder.UserID STP handle — the book row models matching,
// d-chain state models settlement identity (pkg/lx/persist.go). 16 bytes makes the
// account axis birthday-safe (2^64) and, crucially, makes two addresses sharing a
// leading 8-byte prefix DISTINCT accounts so neither can withdraw the other's
// balance.
//
// The settlement identity is the WALLET (AccountID); UserID8 is only a matcher
// hint. The full account identity resolved via orderuser: IS the wallet that bears
// value; the matcher's 8-byte UserID is purely a matching/STP collision hint and
// is NEVER authoritative for settlement (see the AccountID32 migration target on
// userKey16 below).
type userKey = [zapwire.UserSize]byte

// userKey16 renders a user-identity string to the canonical 16-byte ledger
// identity: the leading 16 bytes, zero-padded — byte-identical to the wire's
// user[16] field (zapwire null-padding) so the account the ledger credits is the
// SAME account the orders/deposits name. This is the ONE fold from a user string
// to the settlement identity; there is no 8-byte fold anywhere on the value path.
//
// TODO(accountid32): the settlement identity stays the full 16-byte wire user
// NOW. The target derivation is a domain-separated 32-byte account id:
//
//	AccountID32 = H("lux.account.v1", networkID, sourceChainID, addressKind, fullAddressBytes)
//	UserID8     = first8(H("lux.matcher.user.v1", AccountID))   // matcher/STP handle
//
// UserID8 may collide (birthday over 2^64) — acceptable ONLY because it is NEVER
// authoritative for value (it is a compact STP handle; settlement resolves through
// orderuser: to the full AccountID). The 16-byte userKey is birthday-safe at 2^64
// for the account axis; AccountID32 widens that to 2^128 and binds the network +
// source chain + address kind so identities are globally unique across chains. Do
// NOT implement the 32-byte form here yet — this records the migration target.
//
// NAMING: the settlement identity IS the wallet (AccountID); UserID8 is only a
// matcher hint. This pass keeps the field name `UserID` (the frozen GPU ABI handle)
// and clarifies the wallet/AccountID framing here in the comment + the AccountID32
// TODO above; the rename to an explicit AccountID field is deferred to the
// AccountID32 migration (a separate roadmap).
func userKey16(user string) userKey {
	var k userKey
	copy(k[:], user)
	return k
}

// balanceLedgerKey builds <prefix><user:16><asset:32>. BOTH the user (the full
// 16-byte injective wire identity, zapwire.UserSize) and the asset (the full
// 32-byte injective asset id, zapwire.AssetIDSize) are kept at full width, so no
// two distinct accounts and no two distinct assets ever collide on a balance key
// (the cross-user native-drain fix on the account axis; the native-vault-drain fix
// on the asset axis).
func balanceLedgerKey(prefix string, user userKey, asset [32]byte) []byte {
	k := make([]byte, len(prefix)+len(user)+len(asset))
	copy(k, prefix)
	copy(k[len(prefix):], user[:])
	copy(k[len(prefix)+len(user):], asset[:])
	return k
}

// readUint64 reads an 8-byte big-endian value, returning 0 when absent (an
// account with no row has zero balance — the natural ledger default).
func readUint64(db database.KeyValueReader, key []byte) (uint64, error) {
	v, err := db.Get(key)
	if err == database.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if len(v) != 8 {
		return 0, fmt.Errorf("dchain: corrupt ledger value len=%d", len(v))
	}
	return binary.BigEndian.Uint64(v), nil
}

// writeUint64 persists an 8-byte big-endian value, deleting the row when it
// reaches zero (a zero balance is the absence of a row — keeps the merkle leaf
// set minimal and iteration clean).
func writeUint64(db database.KeyValueWriterDeleter, key []byte, v uint64) error {
	if v == 0 {
		return db.Delete(key)
	}
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	return db.Put(key, b[:])
}

// getAvailable / getLocked return an account's available / locked balance for an
// asset (keyed by the full 16-byte user and the full 32-byte asset id).
func getAvailable(db database.KeyValueReader, user userKey, asset [32]byte) (uint64, error) {
	return readUint64(db, balanceLedgerKey(prefixBalance, user, asset))
}
func getLocked(db database.KeyValueReader, user userKey, asset [32]byte) (uint64, error) {
	return readUint64(db, balanceLedgerKey(prefixLocked, user, asset))
}

// creditAvailable adds amount to an account's available balance (the deposit
// primitive and the fill-proceeds primitive). Refuses overflow — a credit that
// would wrap uint64 is a corrupt input, never silently truncated.
func creditAvailable(db database.KeyValueReaderWriterDeleter, user userKey, asset [32]byte, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := getAvailable(db, user, asset)
	if err != nil {
		return err
	}
	if cur+amount < cur {
		return fmt.Errorf("dchain: available credit overflow user=%x asset=%x", user, asset)
	}
	return writeUint64(db, balanceLedgerKey(prefixBalance, user, asset), cur+amount)
}

// debitAvailable removes amount from an account's available balance. Returns
// ErrInsufficientAvailable when the account lacks the balance — the matcher MUST
// debit only AVAILABLE (deposited, un-escrowed) funds, so an over-debit is a
// hard refusal, never an underflow.
func debitAvailable(db database.KeyValueReaderWriterDeleter, user userKey, asset [32]byte, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := getAvailable(db, user, asset)
	if err != nil {
		return err
	}
	if cur < amount {
		return fmt.Errorf("%w: user=%x asset=%x have=%d need=%d", ErrInsufficientAvailable, user, asset, cur, amount)
	}
	return writeUint64(db, balanceLedgerKey(prefixBalance, user, asset), cur-amount)
}

// lockFromAvailable moves amount from available to locked (placing/submitting an
// order reserves the asset it will spend). All-or-nothing: the debit and the
// lock are written together, so a partial lock can never strand value.
func lockFromAvailable(db database.KeyValueReaderWriterDeleter, user userKey, asset [32]byte, amount uint64) error {
	if amount == 0 {
		return nil
	}
	if err := debitAvailable(db, user, asset, amount); err != nil {
		return err
	}
	cur, err := getLocked(db, user, asset)
	if err != nil {
		return err
	}
	if cur+amount < cur {
		return fmt.Errorf("dchain: locked credit overflow user=%x asset=%x", user, asset)
	}
	return writeUint64(db, balanceLedgerKey(prefixLocked, user, asset), cur+amount)
}

// unlockToAvailable moves amount from locked back to available (cancel, reject,
// or the unspent remainder of a partially-filled order). Refuses to unlock more
// than is locked — that would mint against the ledger.
func unlockToAvailable(db database.KeyValueReaderWriterDeleter, user userKey, asset [32]byte, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := getLocked(db, user, asset)
	if err != nil {
		return err
	}
	if cur < amount {
		return fmt.Errorf("%w: unlock user=%x asset=%x locked=%d need=%d", ErrInsufficientLocked, user, asset, cur, amount)
	}
	if err := writeUint64(db, balanceLedgerKey(prefixLocked, user, asset), cur-amount); err != nil {
		return err
	}
	return creditAvailable(db, user, asset, amount)
}

// spendLocked removes amount from an account's locked balance WITHOUT crediting
// it back — used on a fill, where the spender's locked asset LEAVES the spender
// and is credited (by creditAvailable) to the COUNTERPARTY's available. The pair
// (spendLocked from spender + creditAvailable to counterparty) is what conserves
// value across a fill: every unit removed here is added to exactly one other
// account, so sum(available)+sum(locked) is invariant. Refuses spend > locked.
func spendLocked(db database.KeyValueReaderWriterDeleter, user userKey, asset [32]byte, amount uint64) error {
	if amount == 0 {
		return nil
	}
	cur, err := getLocked(db, user, asset)
	if err != nil {
		return err
	}
	if cur < amount {
		return fmt.Errorf("%w: spend user=%x asset=%x locked=%d need=%d", ErrInsufficientLocked, user, asset, cur, amount)
	}
	return writeUint64(db, balanceLedgerKey(prefixLocked, user, asset), cur-amount)
}

// ---- market (base,quote) asset binding ----

// assetKey builds asset:<poolID:32>.
func assetKey(poolID [32]byte) []byte {
	k := make([]byte, len(prefixAsset)+32)
	copy(k, prefixAsset)
	copy(k[len(prefixAsset):], poolID[:])
	return k
}

// writeMarketAssets binds a market's (base,quote) asset ids (idempotent across
// re-opens with the same pair). The value is base[32]|quote[32] — the FULL
// injective ids the ledger keys balance:/locked: by, so an order's locked spend
// names the SAME asset key a deposit credited.
func writeMarketAssets(db database.KeyValueWriter, poolID [32]byte, base, quote [32]byte) error {
	var v [64]byte
	copy(v[0:32], base[:])
	copy(v[32:64], quote[:])
	return db.Put(assetKey(poolID), v[:])
}

// readMarketAssets returns a market's (base,quote) ids, ok=false if the market's
// assets were never bound (a market created via the asset-less EnsureMarket path
// — orders on it cannot be balance-checked).
func readMarketAssets(db database.KeyValueReader, poolID [32]byte) (base, quote [32]byte, ok bool, err error) {
	v, gerr := db.Get(assetKey(poolID))
	if gerr == database.ErrNotFound {
		return base, quote, false, nil
	}
	if gerr != nil {
		return base, quote, false, gerr
	}
	if len(v) != 64 {
		return base, quote, false, fmt.Errorf("dchain: corrupt market assets len=%d", len(v))
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

// putOrderLock records the (asset, remaining-locked-amount) a resting order
// holds, so a later cancel or a maker fill knows exactly how much to unlock /
// spend. The asset is the FULL 32-byte id; the value is asset[32]|amount[8].
// Deleted when the remaining lock reaches zero.
func putOrderLock(db database.KeyValueWriterDeleter, poolID [32]byte, orderID uint64, asset [32]byte, amount uint64) error {
	if amount == 0 {
		return db.Delete(orderLockKey(poolID, orderID))
	}
	var v [40]byte
	copy(v[0:32], asset[:])
	binary.BigEndian.PutUint64(v[32:40], amount)
	return db.Put(orderLockKey(poolID, orderID), v[:])
}

// getOrderLock returns the (asset, remaining-locked-amount) for a resting order,
// ok=false if none recorded (e.g. an order placed before the balance model, or a
// fully-consumed lock).
func getOrderLock(db database.KeyValueReader, poolID [32]byte, orderID uint64) (asset [32]byte, amount uint64, ok bool, err error) {
	v, gerr := db.Get(orderLockKey(poolID, orderID))
	if gerr == database.ErrNotFound {
		return asset, 0, false, nil
	}
	if gerr != nil {
		return asset, 0, false, gerr
	}
	if len(v) != 40 {
		return asset, 0, false, fmt.Errorf("dchain: corrupt order lock len=%d", len(v))
	}
	copy(asset[:], v[0:32])
	return asset, binary.BigEndian.Uint64(v[32:40]), true, nil
}

// ---- per-order FULL settlement identity (so a fill/cancel credits the resting
// maker's full 16-byte account even after a restart) ----
//
// The matcher's persisted DEXOrder row carries only an 8-byte UserID (the frozen
// GPU ABI / STP handle), so a resting order rebuilt from a row (RowToOrder) loses
// bytes 8..15 of its user. Settlement, however, MUST credit the maker's FULL
// 16-byte account (the one its deposit/lock live under). This row carries that
// full identity in d-chain state, keyed by orderID — orthogonal to the GPU row —
// so a maker fill (live OR post-restart) and a cancel settle to the exact account.
// Written when an order rests (alongside orderasset:), deleted when the reserve is.

// orderUserKey builds orderuser:<poolID:32><orderID:8>.
func orderUserKey(poolID [32]byte, orderID uint64) []byte {
	k := make([]byte, len(prefixOrderUser)+32+8)
	copy(k, prefixOrderUser)
	copy(k[len(prefixOrderUser):], poolID[:])
	binary.BigEndian.PutUint64(k[len(prefixOrderUser)+32:], orderID)
	return k
}

// putOrderUser records a resting order's FULL 16-byte settlement identity, or
// deletes the row when amount-less (called with the order's user on rest, and
// deleted in lockstep with the per-order reserve on cancel/reject/full-fill).
func putOrderUser(db database.KeyValueWriterDeleter, poolID [32]byte, orderID uint64, user userKey) error {
	return db.Put(orderUserKey(poolID, orderID), user[:])
}

// deleteOrderUser removes a resting order's settlement-identity row.
func deleteOrderUser(db database.KeyValueWriterDeleter, poolID [32]byte, orderID uint64) error {
	return db.Delete(orderUserKey(poolID, orderID))
}

// getOrderUser returns a resting order's FULL 16-byte settlement identity, ok=false
// if none recorded (an asset-less / pre-custody market, or an order placed before
// this row existed). On a CUSTODY (assets-bound) market the settlement callers
// (makerSettleKey, the cancel unlock) FAIL CLOSED with ErrMissingSettlementUser when
// ok=false — there is NO fallback to the matcher's 8-byte handle, because a compact-
// handle fallback would make handle collisions value-bearing (a theft bug). The
// boot audit (assertOrderUserCoverage) guarantees ok=true for every resting custody
// order, so ok=false at settle time is state corruption, not a normal path.
func getOrderUser(db database.KeyValueReader, poolID [32]byte, orderID uint64) (user userKey, ok bool, err error) {
	v, gerr := db.Get(orderUserKey(poolID, orderID))
	if gerr == database.ErrNotFound {
		return user, false, nil
	}
	if gerr != nil {
		return user, false, gerr
	}
	if len(v) != len(user) {
		return user, false, fmt.Errorf("dchain: corrupt order user len=%d", len(v))
	}
	copy(user[:], v)
	return user, true, nil
}

// readMeta helpers.

func readLastAccepted(db database.KeyValueReader) (ids.ID, error) {
	v, err := db.Get([]byte(metaLastAccepted))
	if err != nil {
		return ids.Empty, err
	}
	return ids.ToID(v)
}

func writeLastAccepted(db database.KeyValueWriter, id ids.ID) error {
	return db.Put([]byte(metaLastAccepted), id[:])
}

func readHeight(db database.KeyValueReader) (uint64, error) {
	v, err := db.Get([]byte(metaHeight))
	if err == database.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if len(v) != 8 {
		return 0, fmt.Errorf("dchain: corrupt height len=%d", len(v))
	}
	return binary.BigEndian.Uint64(v), nil
}

func writeHeight(db database.KeyValueWriter, height uint64) error {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], height)
	return db.Put([]byte(metaHeight), b[:])
}

func readRoot(db database.KeyValueReader) ([32]byte, error) {
	var r [32]byte
	v, err := db.Get([]byte(metaRoot))
	if err == database.ErrNotFound {
		return r, nil // genesis parent root is zero
	}
	if err != nil {
		return r, err
	}
	if len(v) != 32 {
		return r, fmt.Errorf("dchain: corrupt root len=%d", len(v))
	}
	copy(r[:], v)
	return r, nil
}

func writeRoot(db database.KeyValueWriter, root [32]byte) error {
	return db.Put([]byte(metaRoot), root[:])
}

// =========================================================================
// State root — REUSES the audited xvm execution_root construction:
// keccak256 tagged binary Merkle (RFC-6962, github.com/luxfi/crypto/merkle) for
// each variable-length sub-root, and a fixed-shape keccak256 composition for the
// final root. Identical primitives to vms/xvm/state/xvmroot; the leaves here are
// the d-chain's fixed-point DEXOrder / DEXTrade rows.
// =========================================================================

// Size is the byte length of a root (a keccak256 digest).
const Size = hash.KeccakSize

// bookRoot is the RFC-6962 tagged binary Merkle root over the canonical resting
// book rows. The leaf digest is keccak256 over the order's deterministic
// (Q64.64) field image; rows MUST already be in canonical order (BookToRows /
// sortRows) so the root is insertion-order-independent. An empty book yields
// merkle.EmptyRoot.
func bookRoot(rows []lx.DEXOrder) [Size]byte {
	leaves := make([][Size]byte, len(rows))
	for i := range rows {
		leaves[i] = orderLeafDigest(rows[i], uint32(i))
	}
	return merkle.Root(leaves)
}

// tradeRoot is the RFC-6962 tagged Merkle root over the block's fill rows (the
// event root). Trades are folded in production order (matcher output order),
// each bound to its index.
func tradeRoot(trades []lx.DEXTrade) [Size]byte {
	leaves := make([][Size]byte, len(trades))
	for i := range trades {
		leaves[i] = tradeLeafDigest(trades[i], uint32(i))
	}
	return merkle.Root(leaves)
}

// txRoot is the RFC-6962 tagged Merkle root over the block's transactions (the
// log root). The leaf is keccak256 over the canonical [type][body] tx bytes.
func txRoot(txs []*Tx) [Size]byte {
	leaves := make([][Size]byte, len(txs))
	for i := range txs {
		d := hash.ComputeKeccak256Array(txs[i].Bytes())
		var idx [4]byte
		binary.LittleEndian.PutUint32(idx[:], uint32(i))
		leaves[i] = hash.ComputeKeccak256Array(d[:], idx[:])
	}
	return merkle.Root(leaves)
}

// orderLeafDigest = keccak256(canonical 64B row image ‖ index_le). Binding the
// index makes the leaf position-aware, matching the xvmroot leaf discipline. The
// row image is already the deterministic Q64.64 form, so the digest is
// cross-architecture stable.
func orderLeafDigest(row lx.DEXOrder, i uint32) [Size]byte {
	img := lx.EncodeRow(row)
	var idx [4]byte
	binary.LittleEndian.PutUint32(idx[:], i)
	return hash.ComputeKeccak256Array(img, idx[:])
}

// tradeLeafDigest = keccak256(canonical 80B trade image ‖ index_le).
func tradeLeafDigest(t lx.DEXTrade, i uint32) [Size]byte {
	img := lx.EncodeTrade(t)
	var idx [4]byte
	binary.LittleEndian.PutUint32(idx[:], i)
	return hash.ComputeKeccak256Array(img, idx[:])
}

// ComposeRoot returns the d-chain execution root from the parent root, the three
// sub-roots, and the height — the SAME fixed-shape un-tagged keccak256
// composition xvmroot.Compose uses (this is NOT a Merkle node):
//
//	dex_execution_root = keccak256(
//	    parent ‖ bookRoot ‖ tradeRoot ‖ txRoot ‖ height_u64_le )
//
// Chaining the parent root makes the root a commitment to the whole accepted
// history, not just the current snapshot — a validator that forked at any past
// block produces a different root here.
func ComposeRoot(parent, book, trade, txr [Size]byte, height uint64) [Size]byte {
	var h [8]byte
	binary.LittleEndian.PutUint64(h[:], height)
	return hash.ComputeKeccak256Array(parent[:], book[:], trade[:], txr[:], h[:])
}

// sortRowsCanonical imposes a total, deterministic order on resting rows that may
// span multiple markets (the per-block book root is over every touched market's
// liquidity). Ordering: by side (bids before asks), then — to disambiguate rows
// from different markets at the same price — by orderID, which is globally unique
// across the chain (blockDeterministicID). Within a single market this matches
// lx.sortRows; across markets the unique orderID makes it total.
func sortRowsCanonical(rows []lx.DEXOrder) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Side != b.Side {
			return a.Side == lx.DEXSideBid
		}
		if a.Side == lx.DEXSideBid {
			if a.Price.Integer != b.Price.Integer {
				return a.Price.Integer > b.Price.Integer
			}
			if a.Price.Fraction != b.Price.Fraction {
				return a.Price.Fraction > b.Price.Fraction
			}
		} else {
			if a.Price.Integer != b.Price.Integer {
				return a.Price.Integer < b.Price.Integer
			}
			if a.Price.Fraction != b.Price.Fraction {
				return a.Price.Fraction < b.Price.Fraction
			}
		}
		return a.OrderID < b.OrderID
	})
}

// ExecutionRoot computes the full d-chain execution root over a block's resulting
// state: the canonical resting rows (book), the block's fills (trades), and the
// block's transactions (txs), composed with the parent root and height. Returns
// the execution root and the three sub-roots (for receipts / debugging).
func ExecutionRoot(parent [Size]byte, rows []lx.DEXOrder, trades []lx.DEXTrade, txs []*Tx, height uint64) (root, book, trade, txr [Size]byte) {
	book = bookRoot(rows)
	trade = tradeRoot(trades)
	txr = txRoot(txs)
	root = ComposeRoot(parent, book, trade, txr, height)
	return root, book, trade, txr
}
