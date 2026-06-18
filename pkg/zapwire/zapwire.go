// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package zapwire is the single, frozen source of truth for the CLOB ZAP wire
// frame the D-Chain gateway (dex/pkg/api ZAPServer) speaks and that every
// client — the chains/dexvm atomic proxy and the LP-9010 EVM precompile —
// encodes against.
//
// It is a PURE-GO LEAF: it imports only the standard library, so a public,
// CGO-disabled build can import these constants and codecs WITHOUT dragging in
// the cgo/GPU matching engine that lives in dex/pkg/lx. That separation is what
// lets the public precompile stay a pure-Go module while still framing requests
// byte-identically to the venue's d-chain.
//
// The frame is FROZEN. The same bytes must round-trip across the precompile
// (engine_zap.go), the proxy relay client, and the server (zap_server.go).
// Changing any size or method name here is a wire-breaking change and must be
// versioned at the transport layer, never silently.
package zapwire

import (
	"encoding/binary"
	"fmt"
	"math"
)

// CLOB market-routed ZAP method names. These are the poolId-keyed framing of
// the order-book operations the V4 PoolManager facade (a central-limit-order
// book, NOT an AMM) exposes:
//
//   - EnsureMarket ("initialize"): idempotently create the market/book.
//   - Place       ("modifyLiquidity" +delta): rest a limit order.
//   - Cancel      ("modifyLiquidity" -delta): cancel a resting order.
//   - Submit      ("swap"): submit a MARKETABLE order, get its fills back.
const (
	MethodEnsureMarket = "clob_ensure_market"
	MethodPlace        = "clob_place"
	MethodCancel       = "clob_cancel"
	MethodSubmit       = "clob_submit" // marketable order, returns fills
)

// CLOB ack status bytes.
const (
	StatusPlaced   uint8 = 0
	StatusCanceled uint8 = 1
	StatusRejected uint8 = 2
)

// Wire frame sizes. All multi-byte integers are big-endian; all floats are
// IEEE-754 float64 in big-endian bit form.
const (
	// PoolIDSize is the V4 poolId width: keccak256(PoolKey).
	PoolIDSize = 32
	// UserSize is the on-wire user identity width (null-padded).
	UserSize = 16

	// EnsureMarketReqSize: poolId[32].
	EnsureMarketReqSize = PoolIDSize

	// PlaceReqSize: poolId[32] + side[1] + price[8] + size[8] + user[16].
	PlaceReqSize = PoolIDSize + 1 + 8 + 8 + UserSize // 65

	// CancelReqSize: poolId[32] + orderId[8].
	CancelReqSize = PoolIDSize + 8 // 40

	// SubmitReqSize: poolId[32] + side[1] + isMarket[1] + limitPrice[8] +
	// size[8] + user[16].
	SubmitReqSize = PoolIDSize + 1 + 1 + 8 + 8 + UserSize // 66

	// AckSize: orderId[8] + status[1] + seq[8] + pad[7].
	AckSize = 24

	// FillWireSize is one fill in a Submit response: price[8] + size[8] +
	// takerSide[1]. Base traded = sum(size); quote traded = sum(price*size).
	// The adapter derives the V4 BalanceDelta from these server-produced
	// numbers ONLY — it never trusts a client-supplied amount.
	FillWireSize = 17
)

// Side encodes the order side on the wire: 0=buy/bid, 1=sell/ask.
const (
	SideBuy  uint8 = 0
	SideSell uint8 = 1
)

// Fill is one decoded fill from a Submit response.
type Fill struct {
	Price     float64
	Size      float64
	TakerSide uint8
}

// PutFloat64 writes f as a big-endian IEEE-754 float64 into b[0:8].
func PutFloat64(b []byte, f float64) {
	binary.BigEndian.PutUint64(b, math.Float64bits(f))
}

// Float64 reads a big-endian IEEE-754 float64 from b[0:8].
func Float64(b []byte) float64 {
	return math.Float64frombits(binary.BigEndian.Uint64(b))
}

// padNull copies s into a fresh size-byte buffer, null-padded.
func padNull(s string, size int) []byte {
	out := make([]byte, size)
	copy(out, s)
	return out
}

// trimNull returns b with trailing NUL bytes removed.
func trimNull(b []byte) []byte {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] != 0 {
			return b[:i+1]
		}
	}
	return b[:0]
}

// EncodeEnsureMarket builds the EnsureMarket request payload.
func EncodeEnsureMarket(poolID [32]byte) []byte {
	out := make([]byte, EnsureMarketReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	return out
}

// EncodePlace builds a Place (rest limit order) request payload.
func EncodePlace(poolID [32]byte, side uint8, price, size float64, user string) []byte {
	out := make([]byte, PlaceReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	out[PoolIDSize] = side
	PutFloat64(out[PoolIDSize+1:PoolIDSize+9], price)
	PutFloat64(out[PoolIDSize+9:PoolIDSize+17], size)
	copy(out[PoolIDSize+17:], padNull(user, UserSize))
	return out
}

// EncodeCancel builds a Cancel request payload.
func EncodeCancel(poolID [32]byte, orderID uint64) []byte {
	out := make([]byte, CancelReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	binary.BigEndian.PutUint64(out[PoolIDSize:PoolIDSize+8], orderID)
	return out
}

// EncodeSubmit builds a Submit (marketable order) request payload. When
// isMarket is true the order sweeps unconditionally and limitPrice is ignored;
// otherwise it is an IOC limit bounded by limitPrice.
func EncodeSubmit(poolID [32]byte, side uint8, isMarket bool, limitPrice, size float64, user string) []byte {
	out := make([]byte, SubmitReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	out[PoolIDSize] = side
	if isMarket {
		out[PoolIDSize+1] = 1
	}
	PutFloat64(out[PoolIDSize+2:PoolIDSize+10], limitPrice)
	PutFloat64(out[PoolIDSize+10:PoolIDSize+18], size)
	copy(out[PoolIDSize+18:], padNull(user, UserSize))
	return out
}

// DecodeFills parses a Submit response: fillCount[4] then fillCount ×
// (price[8] + size[8] + takerSide[1]). Returns an error on a malformed frame.
func DecodeFills(resp []byte) ([]Fill, error) {
	if len(resp) < 4 {
		return nil, fmt.Errorf("zapwire: submit response too short: %d", len(resp))
	}
	n := int(binary.BigEndian.Uint32(resp[0:4]))
	if len(resp) < 4+n*FillWireSize {
		return nil, fmt.Errorf("zapwire: submit response truncated: have %d, need %d for %d fills",
			len(resp), 4+n*FillWireSize, n)
	}
	fills := make([]Fill, n)
	off := 4
	for i := 0; i < n; i++ {
		fills[i] = Fill{
			Price:     Float64(resp[off : off+8]),
			Size:      Float64(resp[off+8 : off+16]),
			TakerSide: resp[off+16],
		}
		off += FillWireSize
	}
	return fills, nil
}

// EncodeFills builds a Submit response from fills (server side / test fakes).
func EncodeFills(fills []Fill) []byte {
	out := make([]byte, 4+len(fills)*FillWireSize)
	binary.BigEndian.PutUint32(out[0:4], uint32(len(fills)))
	off := 4
	for _, f := range fills {
		PutFloat64(out[off:off+8], f.Price)
		PutFloat64(out[off+8:off+16], f.Size)
		out[off+16] = f.TakerSide
		off += FillWireSize
	}
	return out
}

// DecodeSubmit parses a Submit request payload into its fields. Clients build
// these; the server and test harnesses decode them.
func DecodeSubmit(payload []byte) (poolID [32]byte, side uint8, isMarket bool, limitPrice, size float64, user string, err error) {
	if len(payload) < SubmitReqSize {
		err = fmt.Errorf("zapwire: submit request too short: %d", len(payload))
		return
	}
	copy(poolID[:], payload[0:PoolIDSize])
	side = payload[PoolIDSize]
	isMarket = payload[PoolIDSize+1] == 1
	limitPrice = Float64(payload[PoolIDSize+2 : PoolIDSize+10])
	size = Float64(payload[PoolIDSize+10 : PoolIDSize+18])
	user = string(trimNull(payload[PoolIDSize+18 : PoolIDSize+18+UserSize]))
	return
}

// EncodeAck builds an ack response (orderId, status, seq).
func EncodeAck(orderID uint64, status uint8, seq uint64) []byte {
	out := make([]byte, AckSize)
	binary.BigEndian.PutUint64(out[0:8], orderID)
	out[8] = status
	binary.BigEndian.PutUint64(out[9:17], seq)
	return out
}

// DecodeAck reads orderId and status from an ack response. A reject frame
// (orderId[8] + status[1] + reasonLen[2] + reason) is also accepted: status is
// read from byte 8 and the rest ignored.
func DecodeAck(resp []byte) (orderID uint64, status uint8, err error) {
	if len(resp) < 9 {
		err = fmt.Errorf("zapwire: ack response too short: %d", len(resp))
		return
	}
	orderID = binary.BigEndian.Uint64(resp[0:8])
	status = resp[8]
	return
}
