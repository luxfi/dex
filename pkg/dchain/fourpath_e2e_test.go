// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo
// +build !cgo

// fourpath_e2e_test.go is the UNIFIED four-path trading proof: it stands up ONE
// D-Chain venue with ONE seeded LUX/LUSD market and drives a real limit order ->
// match -> FILL over EACH of the four DEX access paths against the SAME book,
// asserting each path produces a real CONSENSUS-COMPUTED fill and the correct
// custody balance change, with GLOBAL value conservation (I = available + locked)
// intact across all four crosses.
//
// THE FOUR PATHS — each is a thin wire dialect onto the SAME dex_* matcher
// surface (pkg/zapwire), never its own matcher:
//
//  1. V4 precompile (0x9010): the engine_zap boundary — the EXACT 66-byte
//     dex_submit frame lux/precompile/dex/engine_zap.go Swap() builds (the V4
//     zeroForOne/AmountSpecified/SqrtPriceLimit -> DEX side/size/limit mapping),
//     submitted to the venue, with the V4 BalanceDelta derived from the
//     consensus fills exactly as fillsToDelta does. See v4SubmitFrame /
//     v4BalanceDelta below and the NOTE on what a live luxd EVM adds.
//  2. ZAP native: pkg/zapwire dex_* frames over the venue's rpc.Listen socket —
//     the existing e2eClient.submit path.
//  3. FIX: the new pkg/fix acceptor — a real TCP FIX 4.4 session (Logon,
//     NewOrderSingle, ExecutionReport) whose order routes to the SAME venue via
//     the venue.Venue (dex_*) seam.
//  4. WS: the new pkg/api VenueWS handler — a JSON order over a real
//     gorilla/websocket /ws connection routed to the SAME venue via venue.Venue.
//
// All four BUY orders cross the SAME resting asks the maker seeded over ZAP; the
// fills every path receives are what the consensus auto-sealer re-derived at
// Verify (match -> Verify -> Accept), not a synchronous single-process mutation.

package dchain

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	dexapi "github.com/luxfi/dex/pkg/api"
	"github.com/luxfi/dex/pkg/fix"
	dexvenue "github.com/luxfi/dex/pkg/venue"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// signingVenue is a test gateway that implements dexvenue.Venue by emitting
// SIGNED dex_* frames over its own ZAP connection. The real zapVenue builds
// UNSIGNED frames; the d-chain authorization gate now refuses those, so a
// protocol front-end (FIX/WS) must present a frame signed by the authenticated
// session's account. This models the production reality where the gateway holds a
// per-session signing key: the FIX SenderCompID / WS user names the account, and
// the gateway signs on its behalf at the account's next nonce. (It is NOT a
// weakening of the gate — every frame still carries a valid signature binding the
// asserted account; the test simply moves the signer to the gateway, which is
// where a real multi-protocol venue would hold delegated session keys.)
type signingVenue struct {
	t    *testing.T
	conn *rpc.ZAPConn
}

func newSigningVenue(t *testing.T, addr string) *signingVenue {
	t.Helper()
	conn, err := rpc.ZAPDial(context.Background(), addr)
	if err != nil {
		t.Fatalf("signingVenue dial %s: %v", addr, err)
	}
	return &signingVenue{t: t, conn: conn}
}

func (s *signingVenue) close() { _ = s.conn.Close() }

func (s *signingVenue) EnsureMarket(ctx context.Context, poolID [32]byte) error {
	_, err := s.conn.Call(ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(poolID))
	return err
}

func (s *signingVenue) Place(ctx context.Context, poolID [32]byte, side uint8, price, size float64, user string) (uint64, error) {
	body := zapwire.EncodePlace(poolID, side, price, size, wireUser(s.t, user))
	resp, err := s.conn.Call(ctx, zapwire.MethodPlace, signedPayload(s.t, user, TxPlace, body))
	if err != nil {
		return 0, err
	}
	orderID, status, derr := zapwire.DecodeAck(resp)
	if derr != nil {
		return 0, derr
	}
	if status == zapwire.StatusRejected {
		return 0, dexvenue.ErrRejected
	}
	return orderID, nil
}

func (s *signingVenue) Cancel(ctx context.Context, poolID [32]byte, orderID uint64) error {
	// Not exercised by the four-path e2e (it never cancels over FIX/WS). A real
	// gateway would resolve the resting order's owner session and sign as that
	// account; we fail loudly if a future test path reaches here unimplemented.
	s.t.Fatalf("signingVenue.Cancel called but unimplemented (no FIX/WS cancel in this e2e)")
	return nil
}

func (s *signingVenue) Submit(ctx context.Context, poolID [32]byte, side uint8, isMarket bool, limit, size float64, user string) ([]zapwire.Fill, error) {
	body := zapwire.EncodeSubmit(poolID, side, isMarket, limit, size, wireUser(s.t, user))
	resp, err := s.conn.Call(ctx, zapwire.MethodSubmit, signedPayload(s.t, user, TxSubmit, body))
	if err != nil {
		return nil, err
	}
	return zapwire.DecodeFills(resp)
}

// fixClient is a minimal FIX 4.4 client over TCP for the e2e — it speaks the SAME
// pkg/fix codec the acceptor uses (so this also proves client<->server interop),
// with a monotonic outbound seq and length-prefixed framed reads.
type fixClient struct {
	conn   net.Conn
	r      *bufio.Reader
	sender string
	target string
	seq    int
}

func dialFIX(t *testing.T, addr, sender, target string) *fixClient {
	t.Helper()
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial FIX acceptor %s: %v", addr, err)
	}
	return &fixClient{conn: conn, r: bufio.NewReader(conn), sender: sender, target: target, seq: 1}
}

func (c *fixClient) close() { _ = c.conn.Close() }

func (c *fixClient) send(t *testing.T, m *fix.Message) {
	t.Helper()
	m.Set(fix.TagSenderCompID, c.sender).
		Set(fix.TagTargetCompID, c.target).
		SetInt(fix.TagMsgSeqNum, c.seq)
	c.seq++
	if _, err := c.conn.Write(m.Serialize()); err != nil {
		t.Fatalf("FIX client write: %v", err)
	}
}

// recv reads one framed FIX message: 8=..SOH, 9=<n>SOH, n body bytes, 7-byte 10=
// trailer — then re-validates via the codec.
func (c *fixClient) recv(t *testing.T) *fix.Message {
	t.Helper()
	_ = c.conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	begin, err := c.r.ReadBytes(fix.SOH)
	if err != nil {
		t.Fatalf("FIX recv begin: %v", err)
	}
	blField, err := c.r.ReadBytes(fix.SOH)
	if err != nil {
		t.Fatalf("FIX recv bodylen: %v", err)
	}
	n, err := strconv.Atoi(string(blField[2 : len(blField)-1]))
	if err != nil {
		t.Fatalf("FIX recv bad bodylen %q: %v", blField, err)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(c.r, body); err != nil {
		t.Fatalf("FIX recv body: %v", err)
	}
	trailer := make([]byte, 7)
	if _, err := io.ReadFull(c.r, trailer); err != nil {
		t.Fatalf("FIX recv trailer: %v", err)
	}
	frame := make([]byte, 0, len(begin)+len(blField)+len(body)+len(trailer))
	frame = append(frame, begin...)
	frame = append(frame, blField...)
	frame = append(frame, body...)
	frame = append(frame, trailer...)
	m, err := fix.Parse(frame)
	if err != nil {
		t.Fatalf("FIX recv parse: %v (frame=%q)", err, frame)
	}
	return m
}

func (c *fixClient) logon(t *testing.T) {
	t.Helper()
	c.send(t, fix.NewMessage(fix.MsgTypeLogon).SetInt(fix.TagEncryptMethod, 0).SetInt(fix.TagHeartBtInt, 5))
	reply := c.recv(t)
	if reply.MsgType() != fix.MsgTypeLogon {
		t.Fatalf("FIX: expected Logon reply, got %q", reply.MsgType())
	}
}

// custody seeds a deposit over the venue's ZAP socket (the SAME dex_deposit wire
// path the chains/dexvm proxy uses). Uses contentRef so a retry would dedup.
func (c *e2eClient) deposit(t *testing.T, user string, asset [32]byte, amount uint64) {
	t.Helper()
	ref := contentRef(byte(TxDeposit), user, asset, amount)
	body := zapwire.EncodeDeposit(wireUser(t, user), asset, amount, ref)
	resp, err := c.conn.Call(c.ctx, zapwire.MethodDeposit, signedPayload(t, user, TxDeposit, body))
	if err != nil {
		t.Fatalf("dex_deposit %s %d: %v", user, amount, err)
	}
	status, credited, derr := zapwire.DecodeBalanceResp(resp)
	if derr != nil || status != zapwire.StatusPlaced || credited != amount {
		t.Fatalf("dex_deposit %s %d: status=%d credited=%d err=%v", user, amount, status, credited, derr)
	}
}

// openMarket binds a market's (base, quote) asset ids over ZAP (dex_open_market)
// so the venue knows which asset each order locks/settles.
func (c *e2eClient) openMarket(t *testing.T, pool, base, quote [32]byte) {
	t.Helper()
	resp, err := c.conn.Call(c.ctx, zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(pool, base, quote))
	if err != nil {
		t.Fatalf("dex_open_market: %v", err)
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		t.Fatalf("dex_open_market not placed: resp=%x", resp)
	}
}

// v4SubmitFrame builds the EXACT dex_submit payload lux/precompile/dex
// engine_zap.go Swap() emits for a marketable order, so the V4 leg exercises the
// real engine_zap WIRE CONTRACT (the frozen 66-byte frame), not a re-derivation:
//
//	[0:32]  poolID
//	[32]    side          (V4 zeroForOne -> 1=sell, else 0=buy)
//	[33]    isMarket      (SqrtPriceLimit at the extreme => market)
//	[34:42] limitPrice    big-endian float64
//	[42:50] size          big-endian float64 = |AmountSpecified|
//	[50:66] caller[:16]   the EVM caller's 16-byte handle (taker account)
//
// This is byte-identical to zapwire.EncodeSubmit(pool, side, isMarket, limit,
// size, caller) — the frame is frozen and re-declared in three homes precisely so
// the precompile, the proxy, and the d-chain match. We assert that equality too.
func v4SubmitFrame(poolID [32]byte, side uint8, isMarket bool, limit, size float64, caller string) []byte {
	payload := make([]byte, 66)
	copy(payload[0:32], poolID[:])
	payload[32] = side
	if isMarket {
		payload[33] = 1
	}
	binary.BigEndian.PutUint64(payload[34:42], math.Float64bits(limit))
	binary.BigEndian.PutUint64(payload[42:50], math.Float64bits(size))
	copy(payload[50:66], caller)
	return payload
}

// v4BalanceDelta computes the V4 (amount0, amount1) BalanceDelta from the
// consensus fills exactly as lux/precompile/dex engine_zap.go fillsToDelta does:
// V4 taker-perspective sign convention — positive = user owes the pool, negative =
// pool owes the user. For a BUY (!zeroForOne): receives base (amount0<0), owes
// quote (amount1>0). This is the value the precompile would hand the EVM
// PoolManager; deriving it here proves the V4 leg's settlement math end to end.
func v4BalanceDelta(zeroForOne bool, fills []zapwire.Fill) (amount0, amount1 *big.Int) {
	base, quote := 0.0, 0.0
	for _, f := range fills {
		base += f.Size
		quote += f.Price * f.Size
	}
	ceil := func(v float64) *big.Int { return big.NewInt(int64(math.Ceil(v))) }
	floor := func(v float64) *big.Int { return big.NewInt(int64(math.Floor(v))) }
	if zeroForOne {
		return ceil(base), new(big.Int).Neg(floor(quote))
	}
	return new(big.Int).Neg(floor(base)), ceil(quote)
}

// fixCross drives the FIX path: connect to the acceptor, Logon as `user`,
// NewOrderSingle(BUY size@limit), and collect the fills from the ExecutionReports
// until a terminal status (Fill/Canceled). Returns the fills as zapwire.Fill so
// the e2e asserts them uniformly with the other paths.
func fixCross(t *testing.T, addr, user, symbol string, limit, size float64) []zapwire.Fill {
	t.Helper()
	cli := dialFIX(t, addr, user, "LXDEX")
	defer cli.close()
	cli.logon(t)

	cli.send(t, fix.NewMessage(fix.MsgTypeNewOrderSingle).
		Set(fix.TagClOrdID, "FIX-"+user).
		Set(fix.TagSymbol, symbol).
		Set(fix.TagSide, fix.SideBuy).
		Set(fix.TagOrderQty, formatFloat(size)).
		Set(fix.TagOrdType, fix.OrdTypeLimit).
		Set(fix.TagPrice, formatFloat(limit)))

	var fills []zapwire.Fill
	for {
		m := cli.recv(t)
		if m.MsgType() != fix.MsgTypeExecutionReport {
			t.Fatalf("FIX: expected ExecutionReport, got %q", m.MsgType())
		}
		execType, _ := m.Get(fix.TagExecType)
		switch execType {
		case fix.ExecTypePartialFill, fix.ExecTypeFill:
			lastQty, _ := m.GetFloat(fix.TagLastQty)
			lastPx, _ := m.GetFloat(fix.TagLastPx)
			fills = append(fills, zapwire.Fill{Price: lastPx, Size: lastQty, TakerSide: zapwire.SideBuy})
			if execType == fix.ExecTypeFill {
				return fills
			}
		case fix.ExecTypeNew:
			// ack before fills (or before an IOC cancel) — keep reading.
		case fix.ExecTypeCanceled:
			return fills // IOC remainder / no-liquidity terminal
		case fix.ExecTypeRejected:
			text, _ := m.Get(fix.TagText)
			t.Fatalf("FIX order rejected: %s", text)
		default:
			t.Fatalf("FIX: unexpected ExecType %q", execType)
		}
	}
}

// wsCross drives the WS path: open a /ws connection to the VenueWS test server,
// send a JSON marketable BUY, and read back the fills.
func wsCross(t *testing.T, wsURL, user, symbol string, limit, size float64) []zapwire.Fill {
	t.Helper()
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("WS dial: %v", err)
	}
	defer conn.Close()

	if err := conn.WriteJSON(map[string]interface{}{
		"type":     "submit",
		"symbol":   symbol,
		"side":     "buy",
		"orderQty": size,
		"price":    limit,
		"isMarket": false,
		"user":     user,
		"clOrdId":  "WS-" + user,
	}); err != nil {
		t.Fatalf("WS write: %v", err)
	}

	_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	var resp struct {
		Type  string `json:"type"`
		Error string `json:"error"`
		Fills []struct {
			Price     float64 `json:"price"`
			Size      float64 `json:"size"`
			TakerSide string  `json:"takerSide"`
		} `json:"fills"`
	}
	if err := conn.ReadJSON(&resp); err != nil {
		t.Fatalf("WS read: %v", err)
	}
	if resp.Type == "error" {
		t.Fatalf("WS order error: %s", resp.Error)
	}
	fills := make([]zapwire.Fill, 0, len(resp.Fills))
	for _, f := range resp.Fills {
		side := zapwire.SideBuy
		if f.TakerSide == "sell" {
			side = zapwire.SideSell
		}
		fills = append(fills, zapwire.Fill{Price: f.Price, Size: f.Size, TakerSide: side})
	}
	return fills
}

func formatFloat(v float64) string { return fmt.Sprintf("%g", v) }

// TestFourPathTradingE2E is the unified four-path proof.
func TestFourPathTradingE2E(t *testing.T) {
	dir := t.TempDir()
	v := bootVenue(t, dir)
	defer v.stop()
	t.Logf("venue d-chain up: ZAP endpoint=%s", v.addr)

	// One ZAP control client for seeding (deposits, open-market, resting asks) +
	// the native ZAP cross.
	cli, closeCli := dialVenue(t, v.addr)
	defer closeCli()

	const symbol = "LUX/LUSD"
	pool := poolIDForSymbol(symbol)
	const askSize = 1.0 // each resting ask is 1 LUX; each path buys exactly 1

	// One leg PER path, in execution order. Each path crosses a DISTINCT resting
	// ask at its own price, lowest-first (price-time priority): the maker rests
	// 1 LUX @ {10,20,30,40}; V4 buys @10 (the then-lowest), ZAP @20, FIX @30, WS
	// @40. Distinct prices ALSO make the four place/submit frames distinct, so the
	// d-chain's content-addressed dedup (seen: keyed on the whole frame) treats
	// them as four genuinely different orders rather than one idempotent retry.
	const maker = "maker"
	type leg struct {
		path  string
		user  string
		price float64
	}
	legs := []leg{
		{"V4", "v4taker", 10.0},
		{"ZAP", "zaptaker", 20.0},
		{"FIX", "fixtaker", 30.0},
		{"WS", "wstaker", 40.0},
	}

	// --- seed custody + market (over ZAP, the canonical custody wire path) ---
	cli.ensureMarket(t, pool)
	cli.openMarket(t, pool, assetLUX, assetLUSD)
	// Maker funds len(legs) LUX (to rest one 1-LUX ask per path).
	cli.deposit(t, maker, assetLUX, uint64(len(legs)))
	// Each taker funds exactly its leg price in LUSD (to buy 1 LUX at that price).
	var totalQuoteIn uint64
	for _, l := range legs {
		cli.deposit(t, dexvenue.UserHandle(l.user), assetLUSD, uint64(l.price))
		totalQuoteIn += uint64(l.price)
	}
	assertBalance(t, v.vm, maker, assetLUX, uint64(len(legs)), 0)
	t.Logf("seeded: maker=%d LUX; takers fund %d LUSD total (10+20+30+40)", len(legs), totalQuoteIn)

	// --- maker rests one SELL ask of 1 LUX per leg price (locks len(legs) LUX). ---
	for _, l := range legs {
		cli.place(t, pool, zapwire.SideSell, l.price, askSize, maker)
	}
	assertBalance(t, v.vm, maker, assetLUX, 0, uint64(len(legs))) // all LUX locked in resting asks
	wantRem := float64(len(legs)) * askSize
	if n, rem, _, _, ok := v.vm.BookDepth(pool); !ok || n != len(legs) || rem != wantRem {
		t.Fatalf("seeded asks: orders=%d remaining=%v ok=%v, want %d / %v", n, rem, ok, len(legs), wantRem)
	}
	t.Logf("maker rested %d asks of %g LUX @ {10,20,30,40} LUSD (%d LUX locked)", len(legs), askSize, len(legs))

	// --- stand up the FIX acceptor and the WS venue front-end, BOTH onto the SAME
	// venue via the venue.Venue (dex_*) seam — one matcher, many protocols. The
	// gateway holds the per-session signer: a signingVenue translates a protocol
	// order into a SIGNED dex_* frame on the authenticated session's behalf (the
	// FIX SenderCompID / WS user is the account), so the d-chain gate authorizes it.
	fixVenue := newSigningVenue(t, v.addr)
	defer fixVenue.close()
	acc, err := fix.NewAcceptor(fix.Config{Addr: "127.0.0.1:0", Venue: fixVenue, TargetCompID: "LXDEX", HeartBtInt: 5})
	if err != nil {
		t.Fatalf("NewAcceptor: %v", err)
	}
	fixCtx, fixCancel := context.WithCancel(context.Background())
	defer fixCancel()
	go func() { _ = acc.Serve(fixCtx) }()
	defer acc.Close()

	wsVenueConn := newSigningVenue(t, v.addr)
	defer wsVenueConn.close()
	wsHandler := dexapi.NewVenueWS(wsVenueConn)
	httpSrv := httptest.NewServer(http.HandlerFunc(wsHandler.HandleConnection))
	defer httpSrv.Close()
	wsURL := "ws" + strings.TrimPrefix(httpSrv.URL, "http")

	t.Logf("FIX acceptor=%s  WS endpoint=%s/ws  (both -> same venue %s)", acc.Addr(), wsURL, v.addr)

	// expectedRemaining tracks the book as each path consumes one ask.
	expectedRemaining := float64(len(legs)) * askSize
	crossOne := func(l leg, fills []zapwire.Fill) {
		t.Helper()
		if len(fills) != 1 {
			t.Fatalf("%s: got %d fills, want exactly 1", l.path, len(fills))
		}
		f := fills[0]
		if f.Size != askSize || f.Price != l.price {
			t.Fatalf("%s: fill %g@%g, want %g@%g", l.path, f.Size, f.Price, askSize, l.price)
		}
		expectedRemaining -= askSize
		if _, rem, _, _, ok := v.vm.BookDepth(pool); !ok || rem != expectedRemaining {
			t.Fatalf("%s: book remaining=%v after cross, want %v", l.path, rem, expectedRemaining)
		}
		// Per-path custody change: the taker received 1 LUX, spent its price in LUSD
		// (limit*size == lock exactly, so 0 LUSD refunded, 0 left).
		u := dexvenue.UserHandle(l.user)
		assertBalance(t, v.vm, u, assetLUX, 1, 0)
		assertBalance(t, v.vm, u, assetLUSD, 0, 0)
		t.Logf("  [%s] FILL %g LUX @ %g LUSD (consensus-settled); book now %g; taker %s = 1 LUX / 0 LUSD",
			l.path, f.Size, f.Price, expectedRemaining, u)
	}

	// legByPath looks a leg up by its path label (the four blocks below address
	// their own leg by name for readability).
	legByPath := func(name string) leg {
		for _, l := range legs {
			if l.path == name {
				return l
			}
		}
		t.Fatalf("no leg %q", name)
		return leg{}
	}

	// ===== PATH 1: V4 precompile (engine_zap boundary) =====
	// Build the EXACT engine_zap dex_submit frame and submit it to the venue. A
	// BUY of 1 LUX is !zeroForOne, AmountSpecified magnitude 1, a bounded
	// (non-market) IOC limited at the leg price. Assert the frame is byte-identical
	// to zapwire.EncodeSubmit (the frozen-frame three-homes guarantee), then derive
	// the V4 BalanceDelta from the consensus fills.
	{
		l := legByPath("V4")
		// The V4 caller is the taker's key-derived account (the 16-byte wire user).
		caller := wireUser(t, dexvenue.UserHandle(l.user))
		frame := v4SubmitFrame(pool, zapwire.SideBuy, false, l.price, askSize, caller)
		want := zapwire.EncodeSubmit(pool, zapwire.SideBuy, false, l.price, askSize, caller)
		if string(frame) != string(want) {
			t.Fatalf("V4 engine_zap frame != zapwire.EncodeSubmit (frozen-frame parity broken)")
		}
		// The frozen 66-byte engine_zap frame is the BODY; the gate requires it be
		// signed, so the precompile/proxy appends the auth envelope (here the test
		// signs as the V4 taker's account at its next nonce). Body parity above is
		// unchanged — the signature rides as a trailer the d-chain splits off.
		resp, err := cli.conn.Call(cli.ctx, zapwire.MethodSubmit,
			signedPayload(t, dexvenue.UserHandle(l.user), TxSubmit, frame))
		if err != nil {
			t.Fatalf("V4 dex_submit: %v", err)
		}
		fills, err := zapwire.DecodeFills(resp)
		if err != nil {
			t.Fatalf("V4 fills decode: %v", err)
		}
		crossOne(l, fills)
		// V4 BalanceDelta (taker perspective): buy => receives base (amount0<0),
		// owes quote (amount1>0). With 1 LUX @ price: amount0=-1, amount1=+price.
		a0, a1 := v4BalanceDelta(false, fills)
		if a0.Int64() != -1 || a1.Int64() != int64(l.price) {
			t.Fatalf("V4 BalanceDelta = (%d,%d), want (-1,+%d)", a0.Int64(), a1.Int64(), int64(l.price))
		}
		t.Logf("  [V4] BalanceDelta amount0=%d (base in) amount1=%d (quote owed) — V4 sign convention OK", a0.Int64(), a1.Int64())
	}

	// ===== PATH 2: ZAP native =====
	{
		l := legByPath("ZAP")
		fills := cli.submit(t, pool, zapwire.SideBuy, l.price, askSize, dexvenue.UserHandle(l.user))
		crossOne(l, fills)
	}

	// ===== PATH 3: FIX =====
	{
		l := legByPath("FIX")
		fills := fixCross(t, acc.Addr(), l.user, symbol, l.price, askSize)
		crossOne(l, fills)
	}

	// ===== PATH 4: WS =====
	{
		l := legByPath("WS")
		fills := wsCross(t, wsURL+"/ws", l.user, symbol, l.price, askSize)
		crossOne(l, fills)
	}

	// --- all asks consumed: book empty, maker fully settled ---
	if _, rem, _, _, ok := v.vm.BookDepth(pool); !ok || rem != 0.0 {
		t.Fatalf("after %d crosses, book remaining=%v, want 0", len(legs), rem)
	}
	// Maker sold all LUX, received the sum of the leg prices in LUSD.
	assertBalance(t, v.vm, maker, assetLUX, 0, 0)
	assertBalance(t, v.vm, maker, assetLUSD, totalQuoteIn, 0)

	// --- GLOBAL CONSERVATION (I = available + locked), per asset, across every
	// account that touched the book. Nothing minted, nothing burned. ---
	accounts := []string{maker}
	for _, l := range legs {
		accounts = append(accounts, dexvenue.UserHandle(l.user))
	}
	totalLUX := conserved(t, v.vm, accounts, assetLUX)
	totalLUSD := conserved(t, v.vm, accounts, assetLUSD)
	if totalLUX != uint64(len(legs)) {
		t.Fatalf("LUX conservation: total available+locked=%d, want %d (maker deposited %d)", totalLUX, len(legs), len(legs))
	}
	if totalLUSD != totalQuoteIn {
		t.Fatalf("LUSD conservation: total available+locked=%d, want %d (sum of leg prices)", totalLUSD, totalQuoteIn)
	}

	fmt.Printf("\n=== FOUR-PATH TRADING E2E: PASS ===\n"+
		"  one venue, one LUX/LUSD book; a real consensus FILL over EACH path:\n"+
		"    V4 precompile (engine_zap frame) | ZAP native | FIX acceptor | WS venue\n"+
		"  %d asks consumed; maker 0 LUX / %d LUSD; each taker 1 LUX / 0 LUSD\n"+
		"  CONSERVATION: LUX=%d (=%d in), LUSD=%d (=%d in) — available+locked, nothing minted\n",
		len(legs), totalQuoteIn, totalLUX, len(legs), totalLUSD, totalQuoteIn)
}

// conserved sums (available + locked) of asset across accounts — the
// conservation quantity I. Every unit a fill removes from one account's locked is
// added to a counterparty's available, so this total is invariant under matching.
func conserved(t *testing.T, vm *VM, accounts []string, asset [32]byte) uint64 {
	t.Helper()
	var total uint64
	for _, u := range accounts {
		avail, locked, err := vm.Balance(wireUser(t, u), asset)
		if err != nil {
			t.Fatalf("Balance(%s): %v", u, err)
		}
		total += avail + locked
	}
	return total
}
