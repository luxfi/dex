// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/dex/pkg/venue"
	"github.com/luxfi/dex/pkg/zapwire"
)

// venue_ws.go is the WebSocket trading FRONT-END onto THE matcher — the WS
// counterpart of the FIX acceptor (pkg/fix) and the 0x9010 precompile. It does
// ONE thing: accept a JSON order over /ws, translate it into a venue.Venue call
// (the SAME dex_* surface ZAP/FIX/precompile use), and stream the
// consensus-computed fills back as JSON. It owns NO matcher and NO book — that is
// the whole point of the decomplection: every access path is a thin wire dialect
// onto the single D-Chain DEX.
//
// This is deliberately separate from WebSocketServer (this file), which drives the
// in-process dex.TradingEngine for the rich margin/lending/vault product surface.
// The two are orthogonal: WebSocketServer is the product API; VenueWS is the
// on-chain DEX trading path. Mixing the two matchers into one handler would braid
// "what the product does" with "where the order book lives"; keeping VenueWS a
// focused Venue translator keeps the on-chain path identical across all four
// access methods.

// VenueWS is a minimal WebSocket handler that routes orders to a venue.Venue. One
// VenueWS serves many connections; each connection is independent.
type VenueWS struct {
	venue    venue.Venue
	upgrader websocket.Upgrader
}

// NewVenueWS builds a WS venue front-end over the given matcher seam.
func NewVenueWS(v venue.Venue) *VenueWS {
	return &VenueWS{
		venue: v,
		upgrader: websocket.Upgrader{
			// The DEX trading path serves both browsers (allow-listed origins, the
			// CSRF surface) and non-browser trading clients (server-to-server, CLI,
			// the SDK) that send NO Origin header. An absent Origin is therefore
			// allowed (there is no browser to forge a cross-site request); a present
			// Origin must be on the allow-list. This mirrors how exchange order
			// gateways treat machine clients vs. browser sessions.
			CheckOrigin:     venueCheckOrigin,
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
		},
	}
}

// venueOrder is the inbound JSON order over the WS venue path. It mirrors the FIX
// NewOrderSingle fields and the dex_submit frame: a marketable order bounded by
// an optional limit price that crosses the resting book.
type venueOrder struct {
	Type     string  `json:"type"`     // "submit" (marketable) — the cross primitive
	Symbol   string  `json:"symbol"`   // market, e.g. "LUX/LUSD"
	Side     string  `json:"side"`     // "buy" | "sell"
	OrderQty float64 `json:"orderQty"` // base quantity
	Price    float64 `json:"price"`    // limit; 0 or omitted with isMarket => market
	IsMarket bool    `json:"isMarket"` // true => sweep unconditionally
	User     string  `json:"user"`     // account whose ledger settles
	ClOrdID  string  `json:"clOrdId"`  // client handle echoed back
}

// venueFill is one fill rendered back to the WS client.
type venueFill struct {
	Price     float64 `json:"price"`
	Size      float64 `json:"size"`
	TakerSide string  `json:"takerSide"`
}

// venueResponse is the WS reply: the order ack plus the consensus-computed fills.
type venueResponse struct {
	Type      string      `json:"type"` // "fills" | "error"
	ClOrdID   string      `json:"clOrdId,omitempty"`
	Symbol    string      `json:"symbol,omitempty"`
	Fills     []venueFill `json:"fills,omitempty"`
	CumQty    float64     `json:"cumQty,omitempty"`
	Error     string      `json:"error,omitempty"`
	Timestamp int64       `json:"timestamp"`
}

// HandleConnection upgrades an HTTP request to a WebSocket and serves the venue
// order loop until the peer disconnects. Each inbound JSON order is translated to
// venue.Venue.Submit and answered with its fills.
func (h *VenueWS) HandleConnection(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	for {
		var ord venueOrder
		if err := conn.ReadJSON(&ord); err != nil {
			return // peer closed or bad frame
		}
		h.handleOrder(r.Context(), conn, ord)
	}
}

// handleOrder translates one JSON order into a Venue submit and writes the fills
// back. A bad order or a venue error is reported as a typed error response, never
// a silent drop.
func (h *VenueWS) handleOrder(ctx context.Context, conn *websocket.Conn, ord venueOrder) {
	if ord.Symbol == "" {
		h.writeErr(conn, ord.ClOrdID, "missing symbol")
		return
	}
	var side uint8
	switch ord.Side {
	case "buy", "BUY":
		side = zapwire.SideBuy
	case "sell", "SELL":
		side = zapwire.SideSell
	default:
		h.writeErr(conn, ord.ClOrdID, fmt.Sprintf("bad side %q", ord.Side))
		return
	}
	if ord.OrderQty <= 0 {
		h.writeErr(conn, ord.ClOrdID, "non-positive orderQty")
		return
	}
	if !ord.IsMarket && ord.Price <= 0 {
		h.writeErr(conn, ord.ClOrdID, "limit order needs a positive price")
		return
	}

	poolID := venue.SymbolToPoolID(ord.Symbol)
	user := venue.UserHandle(ord.User)
	fills, err := h.venue.Submit(ctx, poolID, side, ord.IsMarket, ord.Price, ord.OrderQty, user)
	if err != nil {
		h.writeErr(conn, ord.ClOrdID, fmt.Sprintf("venue submit: %v", err))
		return
	}

	resp := venueResponse{
		Type:      "fills",
		ClOrdID:   ord.ClOrdID,
		Symbol:    ord.Symbol,
		Fills:     make([]venueFill, 0, len(fills)),
		Timestamp: time.Now().UnixMilli(),
	}
	for _, f := range fills {
		// Wire fills are EXACT integers (price = PriceInt ×1e8, size = base units);
		// render human floats for the JSON venue client.
		px := float64(f.Price) / float64(zapwire.PriceScale)
		sz := float64(f.Size)
		resp.CumQty += sz
		resp.Fills = append(resp.Fills, venueFill{Price: px, Size: sz, TakerSide: sideStr(f.TakerSide)})
	}
	_ = conn.WriteJSON(resp)
}

func (h *VenueWS) writeErr(conn *websocket.Conn, clOrdID, msg string) {
	_ = conn.WriteJSON(venueResponse{Type: "error", ClOrdID: clOrdID, Error: msg, Timestamp: time.Now().UnixMilli()})
}

func sideStr(s uint8) string {
	if s == zapwire.SideBuy {
		return "buy"
	}
	return "sell"
}

// venueCheckOrigin allows non-browser clients (no Origin header) and browser
// clients from an allow-listed origin. A present-but-disallowed Origin is
// refused.
func venueCheckOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true // non-browser client (SDK / server-to-server)
	}
	return isOriginAllowed(origin)
}
