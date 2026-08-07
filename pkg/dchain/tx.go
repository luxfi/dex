// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package dchain is the standalone D-Chain DEX virtual machine: a
// block.ChainVM (github.com/luxfi/consensus/engine/chain/block) that runs the
// dex.OrderBook matcher inside consensus. Orders arrive as ZAP frames, are
// queued in a mempool (never executed synchronously), drained into a block in
// sequence order, matched at Block.Verify against a versiondb overlay so every
// validator re-derives the fills, and committed to durable zapdb at
// Block.Accept. The persisted chainstate is authoritative; the in-RAM book is a
// rebuildable accelerator folded from the order:* rows on restart.
//
// This file defines the transaction layer: the four DEX operations, their
// canonical [type][body] wire form, and the deterministic TxID.
package dchain

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// TxType is the one-byte discriminant prefixing every transaction body. The body
// for the order operations is the FROZEN zapwire payload verbatim (see
// pkg/zapwire), so the chains/dexvm proxy can forward a client frame into a
// d-chain tx without re-encoding, and the precompile/maker/server all speak the
// same bytes.
type TxType uint8

const (
	// TxEnsureMarket idempotently creates a market (order book) for a poolId.
	// Body = zapwire EnsureMarket payload: poolId[32].
	TxEnsureMarket TxType = iota + 1
	// TxPlace rests a limit order. Body = zapwire Place payload (65B):
	// poolId[32] + side[1] + price[8] + size[8] + user[16].
	TxPlace
	// TxCancel cancels a resting order. Body = zapwire Cancel payload (40B):
	// poolId[32] + orderId[8].
	TxCancel
	// TxSubmit crosses the book with a marketable order. Body = zapwire Submit
	// payload (66B): poolId[32] + side[1] + isMarket[1] + price[8] + size[8] +
	// user[16].
	TxSubmit
	// TxDeposit credits an account's available balance from value the proxy
	// atomically imported. Body = zapwire Deposit payload (88B): user[16] +
	// asset[32] + amount[8] + ref[32].
	TxDeposit
	// TxWithdraw debits an account's realized available balance for the proxy to
	// atomically export. Body = zapwire Withdraw payload (88B): user[16] +
	// asset[32] + amount[8] + ref[32] (clamped to available).
	TxWithdraw
	// TxOpenMarket binds a market's (base,quote) asset ids so orders on it can be
	// value-checked. Body = zapwire OpenMarket payload (96B): poolId[32] +
	// baseAsset[32] + quoteAsset[32].
	TxOpenMarket
	// TxImport CONSUMES a C->D atomic intent object and funds the taker's native D
	// account with the recorded asset (atomic.go executeImport). Body = intentID[32]:
	// the shared-memory key the 0x9999 precompile PUT the C->D object under. It is NOT
	// signature-authorized — its authority is the consumed cross-chain object (the
	// taker already authenticated + locked on C), exactly as the proxy's ImportTx is
	// authorized by the UTXO it consumes, not a per-account signature.
	TxImport
	// TxExport DEBITS the taker's settled native D balance and writes a D->C atomic
	// settlement object the precompile's ImportSettlement consumes (atomic.go
	// executeExport). Body = intentID[32] | asset[32] | amount[8] | spent[8]. Also
	// object-authorized (the D->C object it produces is owned by the intent's recorded
	// owner, so it can only settle value back to the rightful taker), so no signature.
	TxExport
	// TxIntentSubmit CROSSES THE BOOK for an imported cross-chain intent. Body =
	// intentID[32]. It is the OBJECT-AUTHORIZED sibling of TxSubmit, exactly as
	// TxImport is the object-authorized sibling of TxDeposit and TxExport of
	// TxWithdraw — this chain has two ways an operation is authorized, a signature
	// over the tx or a cross-chain object the taker already authorized on C, and the
	// tx TYPE is where that distinction lives.
	//
	// It cannot be a plain TxSubmit. TxSubmit.requiresAuth() is true, so a signature-
	// free one is refused at parse (parseTxFrame), at the RPC edge, and again in
	// authorizeTx; the only ways to emit one from the seam would be to weaken that
	// gate — which would let ANY unsigned submit move ANY account's funds — or to
	// forge a signature nobody holds. So the seam gets its own type and keeps
	// TxSubmit's gate exactly as strong as it was.
	//
	// It carries the intent id ALONE. The operation was declared once, in the C->D
	// object, and recorded in the escrow at import; execute derives the zapwire Submit
	// frame from that escrow (seamSubmitBody) and runs it through the IDENTICAL
	// custody -> match -> settle path a signed dex_submit takes. Restating market/side/
	// limit/size on this wire would create a second copy of the taker's operation for
	// the two to be cross-checked against each other, which is exactly the restatement
	// the C-side seam removed.
	//
	// Injecting one is harmless: the escrow must exist and be `open`, which only this
	// block's own TxImport can make true, and the order it runs is the taker's OWN
	// recorded operation. A replay finds the escrow already `submitted` and rejects.
	TxIntentSubmit
)

// requiresAuth reports whether a tx type moves USER funds and therefore must
// carry a signature authorizing the asserted account. The order operations and
// the two custody legs do; market admin (ensure/open) moves no user value and is
// unauthenticated (it has no asserted account to bind).
func (t TxType) requiresAuth() bool {
	switch t {
	case TxPlace, TxCancel, TxSubmit, TxDeposit, TxWithdraw:
		return true
	default:
		return false
	}
}

// isSeamDriven reports whether a tx is produced by the PROPOSER-SIDE DRIVE rather
// than submitted by a client. Drive txs are a pure function of committed state
// (drive.go), so the next BuildBlock regenerates exactly the ones that are still
// warranted. Requeuing them into the mempool on a rejected block therefore cannot
// help and can only duplicate: the block would carry each leg twice, once from the
// mempool and once from the drive.
func (t TxType) isSeamDriven() bool {
	switch t {
	case TxImport, TxIntentSubmit, TxExport:
		return true
	default:
		return false
	}
}

func (t TxType) String() string {
	switch t {
	case TxEnsureMarket:
		return "ensure_market"
	case TxPlace:
		return "place"
	case TxCancel:
		return "cancel"
	case TxSubmit:
		return "submit"
	case TxDeposit:
		return "deposit"
	case TxWithdraw:
		return "withdraw"
	case TxOpenMarket:
		return "open_market"
	case TxImport:
		return "seam_import"
	case TxExport:
		return "seam_export"
	case TxIntentSubmit:
		return "seam_submit"
	default:
		return "unknown"
	}
}

// seamImportBodySize is the TxImport body width: intentID[32] | object[118]. The
// object's own bytes ride in the transaction so execution is a pure function of the
// block and replays byte-identically forever (see EncodeSeamImportBody).
// seamExportBodySize is the TxExport body width: intentID[32] | asset[32] |
// amount[8] | spent[8].
// seamSubmitBodySize is the TxIntentSubmit body width: intentID[32].
const (
	seamImportBodySize = 32 + seamIntentObjectSize
	seamExportBodySize = 32 + 32 + 8 + 8
	seamSubmitBodySize = 32
)

// Errors returned by the tx layer.
var (
	ErrEmptyTx      = errors.New("dchain: empty transaction")
	ErrUnknownTx    = errors.New("dchain: unknown transaction type")
	ErrShortTxBody  = errors.New("dchain: transaction body too short for its type")
	ErrEmptyMempool = errors.New("dchain: no pending transactions")

	// Custody-ledger errors (the "money lives in the order book" invariants).
	// ErrInsufficientAvailable: an order/withdraw would debit more available
	// balance than an account holds — the matcher debits only deposited funds.
	ErrInsufficientAvailable = errors.New("dchain: insufficient available balance")
	// ErrInsufficientLocked: an unlock/spend would move more than is locked —
	// would mint against the ledger.
	ErrInsufficientLocked = errors.New("dchain: insufficient locked balance")
	// ErrMarketAssetsUnbound: a balance-bearing order on a market whose (base,
	// quote) assets were never bound (dex_open_market) — cannot value-check it.
	ErrMarketAssetsUnbound = errors.New("dchain: market assets not bound")
	// ErrFillMissingUnits: a consensus fill reached settlement without the exact
	// integer lane (BaseUnits/QuoteUnits) — the VM must set SizeUnits on every
	// order so settlement is value-exact; a missing lane is refused, never guessed.
	ErrFillMissingUnits = errors.New("dchain: fill missing integer units")
	// ErrFillUnitsOverflow: a fill's integer units exceed uint64 — refused rather
	// than truncated into a mint/burn.
	ErrFillUnitsOverflow = errors.New("dchain: fill units exceed uint64")
	// ErrMissingSettlementUser: a fill/cancel reached settlement for a resting order
	// that has NO orderuser: row — its full 16-byte settlement identity is
	// unavailable. This is a STATE-CORRUPTION error, NOT a deterministic per-tx
	// reject: it aborts block execution (every validator reads the identical overlay
	// state, so the row is missing on all of them — the abort is consensus-neutral),
	// the block/tx is REJECTED with NO partial settlement and NO balance mutation.
	//
	// Settlement fails closed if full account identity is unavailable. Falling back
	// to matcher UserID would make compact-handle collisions value-bearing — a critical theft bug.
	ErrMissingSettlementUser = errors.New("dchain: resting order missing full settlement identity (orderuser)")

	// ErrDustPriceOrZeroLock: an EXECUTABLE order (qty>0, and for a limit a price>0)
	// whose REQUIRED LOCK floors to zero under the market's integer notional —
	// a BUY whose ceil(size*price) quote rounds to 0 (a sub-tick price), or a SELL
	// whose base size truncates to 0. Such an order is REJECTED at placement: a
	// zero-lock executable order is the "free executable order" hazard — it could
	// rest or cross while reserving NOTHING, letting arithmetic flooring acquire the
	// counterparty's asset for no spend. This is a DETERMINISTIC per-tx reject (the
	// notional floor is a pure function of the frozen wire fields, so every validator
	// rejects identically) — NOT a block abort.
	ErrDustPriceOrZeroLock = errors.New("dchain: executable order floors to zero lock (dust price / zero notional)")
)

// Tx is a parsed DEX transaction: the type discriminant plus the raw zapwire body
// for that type. The raw bytes are retained so the tx is re-serialized
// byte-identically (Bytes) and the TxID is stable.
type Tx struct {
	Type TxType
	Body []byte // the zapwire payload for Type (verbatim)
	// Auth is the authorization envelope (signature + per-account nonce) that
	// proves the asserted account authorized this exact operation. It is non-nil
	// for every money-moving tx (Type.requiresAuth()) and nil for market admin.
	// It is part of the canonical Bytes (hence the TxID), so a tx is bound to its
	// signature and a replay carries the identical, single-use nonce.
	Auth *TxAuth
}

// isMarketScoped reports whether a tx body begins with poolId[32] (the order
// operations + open-market). Deposit/withdraw are ACCOUNT-scoped (their body
// begins with user[16]), not market-scoped, so they have no poolId.
func (t TxType) isMarketScoped() bool {
	switch t {
	case TxEnsureMarket, TxPlace, TxCancel, TxSubmit, TxOpenMarket:
		return true
	default:
		return false
	}
}

// poolID returns the 32-byte market id a MARKET-SCOPED tx body begins with.
// Account-scoped txs (deposit/withdraw) return ok=false — their body has no
// poolId, so callers must dispatch on the type first.
func (tx *Tx) poolID() ([32]byte, bool) {
	var id [32]byte
	if !tx.Type.isMarketScoped() {
		return id, false
	}
	if len(tx.Body) < zapwire.PoolIDSize {
		return id, false
	}
	copy(id[:], tx.Body[:zapwire.PoolIDSize])
	return id, true
}

// NewTx builds an UNAUTHENTICATED Tx (market admin: ensure/open) from a type and
// its zapwire body, validating the body length matches the type's fixed frame
// size. A money-moving type is refused here — it MUST carry a signature, so use
// NewSignedTx. A malformed length is rejected so a short/garbage body can never
// reach the matcher.
func NewTx(t TxType, body []byte) (*Tx, error) {
	if t.requiresAuth() {
		return nil, fmt.Errorf("dchain: %s is money-moving and requires a signature; use NewSignedTx", t)
	}
	want, ok := bodySize(t)
	if !ok {
		return nil, ErrUnknownTx
	}
	if len(body) < want {
		return nil, fmt.Errorf("%w: type=%s have=%d want=%d", ErrShortTxBody, t, len(body), want)
	}
	// Copy so the tx owns its bytes independent of the caller's buffer.
	b := make([]byte, want)
	copy(b, body[:want])
	return &Tx{Type: t, Body: b}, nil
}

// NewSignedTx builds a money-moving Tx from a type, its zapwire body, and the
// authorization envelope. The envelope must be non-nil; the body length must
// match the type's fixed frame size. The signature is NOT verified here (that is
// the gate's job — verifyTxSignature at the RPC boundary and authoritatively in
// block.execute); this constructor only assembles a well-formed frame.
func NewSignedTx(t TxType, body []byte, auth *TxAuth) (*Tx, error) {
	if !t.requiresAuth() {
		return nil, fmt.Errorf("dchain: %s takes no signature; use NewTx", t)
	}
	if auth == nil {
		return nil, fmt.Errorf("%w: %s", ErrTxUnsigned, t)
	}
	want, ok := bodySize(t)
	if !ok {
		return nil, ErrUnknownTx
	}
	if len(body) < want {
		return nil, fmt.Errorf("%w: type=%s have=%d want=%d", ErrShortTxBody, t, len(body), want)
	}
	b := make([]byte, want)
	copy(b, body[:want])
	return &Tx{Type: t, Body: b, Auth: auth}, nil
}

// bodySize returns the fixed zapwire body length for a tx type.
func bodySize(t TxType) (int, bool) {
	switch t {
	case TxEnsureMarket:
		return zapwire.EnsureMarketReqSize, true
	case TxPlace:
		return zapwire.PlaceReqSize, true
	case TxCancel:
		return zapwire.CancelReqSize, true
	case TxSubmit:
		return zapwire.SubmitReqSize, true
	case TxDeposit:
		return zapwire.DepositReqSize, true
	case TxWithdraw:
		return zapwire.WithdrawReqSize, true
	case TxOpenMarket:
		return zapwire.OpenMarketReqSize, true
	case TxImport:
		return seamImportBodySize, true
	case TxExport:
		return seamExportBodySize, true
	case TxIntentSubmit:
		return seamSubmitBodySize, true
	default:
		return 0, false
	}
}

// Bytes returns the canonical wire form: [type:1][body][auth?]. The auth
// envelope is appended for a money-moving tx (nil for admin), so the TxID hashes
// over the signature + nonce too — a forged or replayed authorization yields a
// different (or already-seen) id. The fixed per-type body size delimits body
// from the auth trailer on parse. Deterministic and self-describing.
func (tx *Tx) Bytes() []byte {
	var auth []byte
	if tx.Auth != nil {
		auth = tx.Auth.encode()
	}
	out := make([]byte, 1+len(tx.Body)+len(auth))
	out[0] = byte(tx.Type)
	n := copy(out[1:], tx.Body)
	copy(out[1+n:], auth)
	return out
}

// ID is the deterministic transaction id: Checksum256 over the canonical bytes.
// ids.Checksum256 is sha256(sha256(b))-style content addressing used across the
// Lux stack, so the same tx bytes always yield the same id on every node.
func (tx *Tx) ID() ids.ID {
	return ids.Checksum256(tx.Bytes())
}

// ParseTx decodes a [type:1][body] wire transaction, validating the type and the
// body length. Used by ParseBlock and by the mempool when accepting a relayed
// frame.
func ParseTx(b []byte) (*Tx, error) {
	if len(b) == 0 {
		return nil, ErrEmptyTx
	}
	return parseTxFrame(TxType(b[0]), b[1:])
}

// parseTxFrame decodes a type's [body][auth?] tail into a Tx — the shared body of
// ParseTx (block decode) and the handler's RPC-payload decode, so the wire split
// (fixed body size delimits the auth trailer) is defined exactly once.
func parseTxFrame(t TxType, rest []byte) (*Tx, error) {
	want, ok := bodySize(t)
	if !ok {
		return nil, ErrUnknownTx
	}
	if len(rest) < want {
		return nil, fmt.Errorf("%w: type=%s have=%d want=%d", ErrShortTxBody, t, len(rest), want)
	}
	body := rest[:want]
	if !t.requiresAuth() {
		// Admin tx: body must be exactly the frame, no trailer.
		if len(rest) != want {
			return nil, fmt.Errorf("%w: %s carries unexpected %d trailing bytes", ErrTxMalformedAuth, t, len(rest)-want)
		}
		return NewTx(t, body)
	}
	auth, err := decodeTxAuth(rest[want:])
	if err != nil {
		return nil, err
	}
	return NewSignedTx(t, body, auth)
}

// encodeTxList serializes a list of txs as [count:4][len:4 body]*, the block body
// format. Each tx is length-prefixed so ParseBlock can split them back exactly.
func encodeTxList(txs []*Tx) []byte {
	// Materialize each tx's canonical bytes first (they include the variable-length
	// auth trailer), then size the buffer from them — never from len(Body) alone.
	raws := make([][]byte, len(txs))
	size := 4
	for i, tx := range txs {
		raws[i] = tx.Bytes()
		size += 4 + len(raws[i])
	}
	out := make([]byte, size)
	binary.BigEndian.PutUint32(out[0:4], uint32(len(txs)))
	off := 4
	for _, raw := range raws {
		binary.BigEndian.PutUint32(out[off:off+4], uint32(len(raw)))
		off += 4
		off += copy(out[off:], raw)
	}
	return out
}

// decodeTxList parses a [count:4][len:4 body]* tx list. Every length is bounds
// checked so a truncated block body is rejected rather than panicking.
func decodeTxList(b []byte) ([]*Tx, error) {
	if len(b) < 4 {
		return nil, fmt.Errorf("dchain: tx list too short: %d", len(b))
	}
	n := int(binary.BigEndian.Uint32(b[0:4]))
	off := 4
	txs := make([]*Tx, 0, n)
	for i := 0; i < n; i++ {
		if off+4 > len(b) {
			return nil, fmt.Errorf("dchain: tx list truncated at tx %d (len header)", i)
		}
		l := int(binary.BigEndian.Uint32(b[off : off+4]))
		off += 4
		if off+l > len(b) {
			return nil, fmt.Errorf("dchain: tx list truncated at tx %d (body): need %d have %d", i, l, len(b)-off)
		}
		tx, err := ParseTx(b[off : off+l])
		if err != nil {
			return nil, fmt.Errorf("dchain: tx %d: %w", i, err)
		}
		off += l
		txs = append(txs, tx)
	}
	return txs, nil
}

// EncodeSeamSubmitBody builds a TxIntentSubmit body: intentID[32] — the imported
// intent whose recorded operation this tx runs. Exported for the keeper / SDK / tests.
func EncodeSeamSubmitBody(intentID ids.ID) []byte {
	b := make([]byte, seamSubmitBodySize)
	copy(b, intentID[:])
	return b
}
