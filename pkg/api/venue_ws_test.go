// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/dex/pkg/venue"
	"github.com/luxfi/dex/pkg/zapwire"
)

// fakeVenue is a canned-fills Venue for unit-testing VenueWS in isolation from the
// matcher — the WS handler's job is JSON<->Venue translation, proven here without
// consensus (the unified pkg/dchain e2e proves the REAL matcher over WS).
type fakeVenue struct {
	lastSubmit submitArgs
	fills      []zapwire.Fill
}

type submitArgs struct {
	pool        [32]byte
	side        uint8
	isMarket    bool
	limit, size float64
	user        string
}

func (f *fakeVenue) EnsureMarket(context.Context, [32]byte) error { return nil }
func (f *fakeVenue) Place(context.Context, [32]byte, uint8, float64, float64, string) (uint64, error) {
	return 1, nil
}
func (f *fakeVenue) Cancel(context.Context, [32]byte, uint64) error { return nil }
func (f *fakeVenue) Submit(_ context.Context, pool [32]byte, side uint8, isMarket bool, limit, size float64, user string) ([]zapwire.Fill, error) {
	f.lastSubmit = submitArgs{pool, side, isMarket, limit, size, user}
	return f.fills, nil
}

func startVenueWS(t *testing.T, v venue.Venue) string {
	t.Helper()
	h := NewVenueWS(v)
	srv := httptest.NewServer(http.HandlerFunc(h.HandleConnection))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

func dialWS(t *testing.T, url string) *websocket.Conn {
	t.Helper()
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatalf("WS dial: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

type wsResp struct {
	Type  string `json:"type"`
	Error string `json:"error"`
	Fills []struct {
		Price     float64 `json:"price"`
		Size      float64 `json:"size"`
		TakerSide string  `json:"takerSide"`
	} `json:"fills"`
	CumQty float64 `json:"cumQty"`
}

func readWS(t *testing.T, conn *websocket.Conn) wsResp {
	t.Helper()
	_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	var r wsResp
	if err := conn.ReadJSON(&r); err != nil {
		t.Fatalf("WS read: %v", err)
	}
	return r
}

// TestVenueWSFills proves a JSON marketable order routes to Venue.Submit with the
// right market/side/limit and the fills come back as JSON.
func TestVenueWSFills(t *testing.T) {
	v := &fakeVenue{fills: []zapwire.Fill{
		{Price: 100, Size: 10, TakerSide: zapwire.SideBuy},
		{Price: 101, Size: 2, TakerSide: zapwire.SideBuy},
	}}
	conn := dialWS(t, startVenueWS(t, v))

	if err := conn.WriteJSON(map[string]interface{}{
		"type": "submit", "symbol": "LUX/LUSD", "side": "buy",
		"orderQty": 12.0, "price": 101.0, "user": "alice", "clOrdId": "W1",
	}); err != nil {
		t.Fatalf("WS write: %v", err)
	}

	r := readWS(t, conn)
	if r.Type != "fills" {
		t.Fatalf("resp type=%q want fills (err=%q)", r.Type, r.Error)
	}
	if len(r.Fills) != 2 || r.CumQty != 12 {
		t.Fatalf("fills=%+v cumQty=%v, want 2 fills / 12", r.Fills, r.CumQty)
	}
	if v.lastSubmit.pool != venue.SymbolToPoolID("LUX/LUSD") || v.lastSubmit.side != zapwire.SideBuy ||
		v.lastSubmit.limit != 101 || v.lastSubmit.size != 12 || v.lastSubmit.user != "alice" {
		t.Fatalf("submit args wrong: %+v", v.lastSubmit)
	}
}

// TestVenueWSBadOrder proves a malformed order is reported as a typed error, never
// dropped or routed.
func TestVenueWSBadOrder(t *testing.T) {
	v := &fakeVenue{}
	conn := dialWS(t, startVenueWS(t, v))

	// Missing side.
	if err := conn.WriteJSON(map[string]interface{}{
		"type": "submit", "symbol": "LUX/LUSD", "orderQty": 1.0, "price": 1.0,
	}); err != nil {
		t.Fatalf("WS write: %v", err)
	}
	r := readWS(t, conn)
	if r.Type != "error" || r.Error == "" {
		t.Fatalf("bad order must error, got %+v", r)
	}
}
