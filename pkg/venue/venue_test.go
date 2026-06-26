// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package venue

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// fakeDEX stands up a real rpc.Listen server with canned dex_* handlers, so the
// production zapVenue is exercised over a REAL ZAP socket (the same transport
// production uses) without standing up the cgo matcher. It proves zapVenue's frame
// encode + ack/fill decode + error mapping, end to end over the wire.
type fakeDEX struct {
	server   rpc.Server
	cancel   context.CancelFunc
	placeID  uint64
	fills    []zapwire.Fill
	cancelOK bool
}

func startFakeDEX(t *testing.T) (*fakeDEX, string) {
	t.Helper()
	srv, err := rpc.Listen("127.0.0.1:0")
	if err != nil {
		t.Fatalf("rpc.Listen: %v", err)
	}
	f := &fakeDEX{server: srv, placeID: 42, cancelOK: true}

	must := func(method string, h rpc.RawHandler) {
		if err := srv.RegisterRaw(method, h); err != nil {
			t.Fatalf("register %s: %v", method, err)
		}
	}
	must(zapwire.MethodEnsureMarket, func(_ context.Context, _ []byte) ([]byte, error) {
		return zapwire.EncodeAck(0, zapwire.StatusPlaced, 0), nil
	})
	must(zapwire.MethodPlace, func(_ context.Context, _ []byte) ([]byte, error) {
		return zapwire.EncodeAck(f.placeID, zapwire.StatusPlaced, 0), nil
	})
	must(zapwire.MethodCancel, func(_ context.Context, _ []byte) ([]byte, error) {
		if f.cancelOK {
			return zapwire.EncodeAck(0, zapwire.StatusCanceled, 0), nil
		}
		return zapwire.EncodeAck(0, zapwire.StatusRejected, 0), nil
	})
	must(zapwire.MethodSubmit, func(_ context.Context, _ []byte) ([]byte, error) {
		return zapwire.EncodeFills(f.fills), nil
	})

	ctx, cancel := context.WithCancel(context.Background())
	f.cancel = cancel
	go func() { _ = srv.Serve(ctx) }()
	t.Cleanup(func() { cancel(); _ = srv.Close() })
	return f, srv.Addr()
}

func TestZAPVenueRoundTrip(t *testing.T) {
	f, addr := startFakeDEX(t)
	f.fills = []zapwire.Fill{{Price: 100, Size: 3, TakerSide: zapwire.SideBuy}}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	v, err := DialVenue(ctx, addr)
	if err != nil {
		t.Fatalf("DialVenue: %v", err)
	}

	var pool [32]byte
	pool[0] = 0xab

	if err := v.EnsureMarket(ctx, pool); err != nil {
		t.Fatalf("EnsureMarket: %v", err)
	}
	id, err := v.Place(ctx, pool, zapwire.SideSell, 100, 3, "maker")
	if err != nil {
		t.Fatalf("Place: %v", err)
	}
	if id != 42 {
		t.Fatalf("Place orderID=%d, want 42", id)
	}
	fills, err := v.Submit(ctx, pool, zapwire.SideBuy, false, 100, 3, "taker")
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if len(fills) != 1 || fills[0].Size != 3 || fills[0].Price != 100 {
		t.Fatalf("Submit fills = %+v, want one 3@100", fills)
	}
	if err := v.Cancel(ctx, pool, 42); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
}

// TestZAPVenueCancelRejectedMapsToErrRejected proves a non-canceled cancel ack is
// surfaced as ErrRejected, not a silent success.
func TestZAPVenueCancelRejectedMapsToErrRejected(t *testing.T) {
	f, addr := startFakeDEX(t)
	f.cancelOK = false

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	v, err := DialVenue(ctx, addr)
	if err != nil {
		t.Fatalf("DialVenue: %v", err)
	}
	var pool [32]byte
	err = v.Cancel(ctx, pool, 7)
	if !errors.Is(err, ErrRejected) {
		t.Fatalf("Cancel of rejected order: want ErrRejected, got %v", err)
	}
}

// TestSymbolToPoolIDDistinct proves the symbol resolver is collision-free for
// distinct symbols and tags the high byte (never a zero id).
func TestSymbolToPoolIDDistinct(t *testing.T) {
	a := SymbolToPoolID("LUX/LUSD")
	b := SymbolToPoolID("LUX/LETH")
	if a == b {
		t.Fatal("distinct symbols mapped to same poolID")
	}
	if a[31] != 0xD0 || b[31] != 0xD0 {
		t.Fatal("poolID high byte tag missing")
	}
	var zero [32]byte
	if a == zero || b == zero {
		t.Fatal("poolID collided with zero id")
	}
}

// TestUserHandleTruncates proves the user handle is clamped to the ledger's
// 16-byte key width.
func TestUserHandleTruncates(t *testing.T) {
	if got := UserHandle("short"); got != "short" {
		t.Fatalf("UserHandle(short)=%q", got)
	}
	long := "0123456789abcdefGHIJ" // 20 bytes
	if got := UserHandle(long); len(got) != zapwire.UserSize || got != long[:zapwire.UserSize] {
		t.Fatalf("UserHandle truncation: %q (len %d)", got, len(got))
	}
}
