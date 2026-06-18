// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package dchain is the standalone D-Chain DEX virtual machine: a
// block.ChainVM (github.com/luxfi/consensus/engine/chain/block) that runs the
// lx.OrderBook matcher inside consensus. Orders arrive as ZAP frames, are
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
)

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
	default:
		return "unknown"
	}
}

// Errors returned by the tx layer.
var (
	ErrEmptyTx      = errors.New("dchain: empty transaction")
	ErrUnknownTx    = errors.New("dchain: unknown transaction type")
	ErrShortTxBody  = errors.New("dchain: transaction body too short for its type")
	ErrEmptyMempool = errors.New("dchain: no pending transactions")
)

// Tx is a parsed DEX transaction: the type discriminant plus the raw zapwire body
// for that type. The raw bytes are retained so the tx is re-serialized
// byte-identically (Bytes) and the TxID is stable.
type Tx struct {
	Type TxType
	Body []byte // the zapwire payload for Type (verbatim)
}

// poolID returns the 32-byte market id every tx body begins with. All four tx
// bodies start with poolId[32]; the type-specific decode is done in execute.go.
func (tx *Tx) poolID() ([32]byte, bool) {
	var id [32]byte
	if len(tx.Body) < zapwire.PoolIDSize {
		return id, false
	}
	copy(id[:], tx.Body[:zapwire.PoolIDSize])
	return id, true
}

// NewTx builds a Tx from a type and its zapwire body, validating the body length
// matches the type's fixed frame size. A malformed length is rejected here so a
// short/garbage body can never reach the matcher.
func NewTx(t TxType, body []byte) (*Tx, error) {
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
	default:
		return 0, false
	}
}

// Bytes returns the canonical wire form: [type:1][body]. This is what a block
// stores and what the TxID hashes — deterministic and self-describing.
func (tx *Tx) Bytes() []byte {
	out := make([]byte, 1+len(tx.Body))
	out[0] = byte(tx.Type)
	copy(out[1:], tx.Body)
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
	t := TxType(b[0])
	return NewTx(t, b[1:])
}

// encodeTxList serializes a list of txs as [count:4][len:4 body]*, the block body
// format. Each tx is length-prefixed so ParseBlock can split them back exactly.
func encodeTxList(txs []*Tx) []byte {
	size := 4
	for _, tx := range txs {
		size += 4 + 1 + len(tx.Body)
	}
	out := make([]byte, size)
	binary.BigEndian.PutUint32(out[0:4], uint32(len(txs)))
	off := 4
	for _, tx := range txs {
		raw := tx.Bytes()
		binary.BigEndian.PutUint32(out[off:off+4], uint32(len(raw)))
		off += 4
		copy(out[off:off+len(raw)], raw)
		off += len(raw)
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
