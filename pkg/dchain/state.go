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
