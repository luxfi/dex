// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package venue is the protocol-neutral seam between a wire front-end and THE
// matcher. It defines ONE interface — the four operations of the FROZEN dex_*
// ZAP surface (pkg/zapwire) — and ONE production implementation that dials the
// venue's dex_* endpoint. Every protocol front-end (FIX in pkg/fix, the WS
// venue handler in pkg/api, the 0x9010 precompile in lux/precompile/dex)
// translates its dialect into these calls and renders the consensus-computed
// result back; none of them owns a matcher. This is the decomplection: the
// matcher and book live in ONE place (pkg/dchain), reached over ONE surface
// (dex_*), and a "path" is purely a wire translation onto this Venue.
package venue

import (
	"context"
	"fmt"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// Venue is the protocol-neutral seam between a wire front-end (FIX in pkg/fix, WS
// in pkg/api) and THE matcher. It is the SAME four operations the FROZEN dex_*
// ZAP surface exposes (pkg/zapwire) — ensure-market, place a resting limit order,
// cancel a resting order, submit a marketable order and get its fills. A front-end
// translates its protocol into these calls; it does NOT match. There is exactly
// ONE matcher (the D-Chain DEX) and exactly ONE network surface onto it (dex_*);
// Venue is that surface expressed as a Go interface so a front-end can be unit
// tested against a fake and run in production against the real ZAP client.
//
// Side/poolID semantics are the zapwire ones: side 0=buy, 1=sell; poolID is the
// 32-byte market identity. price/size are float64 on the price grid (the same
// units dex_place/dex_submit carry); user is the <=16-byte account handle whose
// ledger the order locks against and settles into.
type Venue interface {
	// EnsureMarket idempotently creates the book for poolID.
	EnsureMarket(ctx context.Context, poolID [32]byte) error
	// Place rests a limit order and returns the consensus-assigned order id.
	Place(ctx context.Context, poolID [32]byte, side uint8, price, size float64, user string) (orderID uint64, err error)
	// Cancel cancels a resting order by id. Returns ErrRejected if not cancelable.
	Cancel(ctx context.Context, poolID [32]byte, orderID uint64) error
	// Submit crosses a marketable order against the book and returns the
	// consensus-computed fills (possibly empty if nothing crossed).
	Submit(ctx context.Context, poolID [32]byte, side uint8, isMarket bool, limit, size float64, user string) ([]zapwire.Fill, error)
}

// ErrRejected is returned by a Venue operation the matcher refused (e.g. a place
// the matcher would not rest, or a cancel of an unknown order).
var ErrRejected = fmt.Errorf("venue: rejected")

// zapVenue is the production Venue: a dex_* ZAP client over github.com/luxfi/rpc,
// dialing the SAME rpc.Listen socket dchain.VM.RegisterDEX serves and the SAME
// socket the 0x9010 precompile (engine_zap) and the chains/dexvm proxy dial. It
// holds no matcher state — every call is a frozen zapwire frame to the venue.
type zapVenue struct {
	conn *rpc.ZAPConn
}

// DialVenue opens a ZAP client to the venue's dex_* endpoint (e.g. the addr
// returned by rpc.Listen / the node runtime). The returned Venue speaks the
// frozen dex_* frames; Close releases the connection.
func DialVenue(ctx context.Context, addr string) (Venue, error) {
	conn, err := rpc.ZAPDial(ctx, addr)
	if err != nil {
		return nil, fmt.Errorf("fix: dial venue %s: %w", addr, err)
	}
	return &zapVenue{conn: conn}, nil
}

// NewZAPVenue wraps an already-dialed ZAP connection as a Venue (used when the
// caller manages the connection lifecycle, e.g. the gateway daemon).
func NewZAPVenue(conn *rpc.ZAPConn) Venue { return &zapVenue{conn: conn} }

// Close releases the underlying ZAP connection.
func (v *zapVenue) Close() error { return v.conn.Close() }

func (v *zapVenue) EnsureMarket(ctx context.Context, poolID [32]byte) error {
	resp, err := v.conn.Call(ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(poolID))
	if err != nil {
		return fmt.Errorf("fix: dex_ensure_market: %w", err)
	}
	if _, status, derr := zapwire.DecodeAck(resp); derr != nil {
		return fmt.Errorf("fix: dex_ensure_market ack: %w", derr)
	} else if status == zapwire.StatusRejected {
		return fmt.Errorf("%w: ensure_market", ErrRejected)
	}
	return nil
}

// wirePrice scales a human price to the exact ×1e8 fixed-point wire integer; wireSize
// truncates a human size to whole atomic base units. The ZAP wire carries exact
// integers only (no float64), so a value-bearing quantity is byte-identical on
// every validator.
func wirePrice(price float64) uint64 {
	if !(price > 0) {
		return 0
	}
	return uint64(price * float64(zapwire.PriceScale))
}
func wireSize(size float64) uint64 {
	if !(size > 0) {
		return 0
	}
	return uint64(size)
}

func (v *zapVenue) Place(ctx context.Context, poolID [32]byte, side uint8, price, size float64, user string) (uint64, error) {
	resp, err := v.conn.Call(ctx, zapwire.MethodPlace, zapwire.EncodePlace(poolID, side, wirePrice(price), wireSize(size), user))
	if err != nil {
		return 0, fmt.Errorf("fix: dex_place: %w", err)
	}
	orderID, status, derr := zapwire.DecodeAck(resp)
	if derr != nil {
		return 0, fmt.Errorf("fix: dex_place ack: %w", derr)
	}
	if status == zapwire.StatusRejected {
		return 0, fmt.Errorf("%w: place", ErrRejected)
	}
	return orderID, nil
}

func (v *zapVenue) Cancel(ctx context.Context, poolID [32]byte, orderID uint64) error {
	resp, err := v.conn.Call(ctx, zapwire.MethodCancel, zapwire.EncodeCancel(poolID, orderID))
	if err != nil {
		return fmt.Errorf("fix: dex_cancel: %w", err)
	}
	_, status, derr := zapwire.DecodeAck(resp)
	if derr != nil {
		return fmt.Errorf("fix: dex_cancel ack: %w", derr)
	}
	if status != zapwire.StatusCanceled {
		return fmt.Errorf("%w: cancel order %d", ErrRejected, orderID)
	}
	return nil
}

func (v *zapVenue) Submit(ctx context.Context, poolID [32]byte, side uint8, isMarket bool, limit, size float64, user string) ([]zapwire.Fill, error) {
	resp, err := v.conn.Call(ctx, zapwire.MethodSubmit, zapwire.EncodeSubmit(poolID, side, isMarket, wirePrice(limit), wireSize(size), user))
	if err != nil {
		return nil, fmt.Errorf("fix: dex_submit: %w", err)
	}
	fills, err := zapwire.DecodeFills(resp)
	if err != nil {
		return nil, fmt.Errorf("fix: dex_submit fills decode: %w", err)
	}
	return fills, nil
}

// SymbolToPoolID derives the venue poolID for a FIX Symbol (tag 55). The venue
// keys markets by a raw 32-byte id; a FIX client names the market by its human
// symbol (e.g. "LUX/LUSD"). This is the SAME stable mapping the venue localnet
// harness uses (poolIDForSymbol): embed the ASCII symbol and tag the high byte so
// the all-symbol-prefix pool can never collide with a zero id. A front-end that
// already has the 32-byte poolID (e.g. the precompile) bypasses this and keys
// directly; FIX clients address by symbol, so this is FIX's resolver.
func SymbolToPoolID(symbol string) [32]byte {
	var p [32]byte
	copy(p[:], symbol)
	p[31] = 0xD0
	return p
}

// UserHandle truncates a sender/account identity to the <=16-byte on-wire user
// handle the venue ledger keys balances by. Two distinct senders that share a
// 16-byte prefix would collide; production binds a checked account id, but for the
// DEX user field the 16-byte handle is the canonical width (zapwire.UserSize).
func UserHandle(s string) string {
	if len(s) > zapwire.UserSize {
		return s[:zapwire.UserSize]
	}
	return s
}
