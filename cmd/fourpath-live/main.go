// Command fourpath-live drives the unified four-path DEX proof against a LIVE,
// already-running D-Chain venue (cmd/dvenue) + an optional C-Chain EVM RPC,
// instead of an in-process test venue. It is the live-network analogue of
// pkg/dchain TestFourPathTradingE2E: ONE LUX/LUSD book, a REAL consensus FILL
// over EACH of the four access paths, then GLOBAL conservation.
//
// The four paths, all converging on the same venue.Venue (clob_*) seam:
//  1. V4 precompile (0x9010): the EXACT 66-byte engine_zap clob_submit frame,
//     asserted byte-identical to zapwire.EncodeSubmit, sent over the venue's ZAP
//     socket (the same wire the on-chain precompile relays).
//  2. ZAP native: zapwire clob_submit over rpc.ZAPDial.
//  3. FIX: a real pkg/fix 4.4 acceptor in front of the venue; a FIX client logs
//     on and sends NewOrderSingle, reads ExecutionReport fills.
//  4. WS: the pkg/api VenueWS JSON front-end in front of the venue; a websocket
//     client submits a JSON order and reads fills.
//
// Conservation + book depth are read over the venue's read-only clob_balance /
// clob_depth observation methods (served by cmd/dvenue), so this is a pure
// client — it needs no in-process VM handle.
//
// Every run uses a UNIQUE market symbol + unique user handles (seeded from -run
// or the wall clock) so repeated runs against the same shared live venue never
// collide on content-addressed dedup, resting book, or custody rows.
//
// Usage:
//
//	go run ./cmd/fourpath-live \
//	  -venue 127.0.0.1:9099 \
//	  -cchain-rpc http://127.0.0.1:9650/ext/bc/C/rpc \
//	  -run devnet-$(date +%s)
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"time"

	dexapi "github.com/luxfi/dex/pkg/api"
	"github.com/luxfi/dex/pkg/fix"
	dexvenue "github.com/luxfi/dex/pkg/venue"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"

	"github.com/gorilla/websocket"
)

// read-only observation methods served by cmd/dvenue.
const (
	clobDepthMethod   = "clob_depth"
	clobBalanceMethod = "clob_balance"
)

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FAIL: "+format+"\n", args...)
	os.Exit(1)
}

// poolIDForSymbol mirrors pkg/dchain.poolIDForSymbol: the symbol bytes, high
// byte tagged 0xD0 so an all-symbol-prefix pool can't collide with a zero id.
func poolIDForSymbol(sym string) [32]byte {
	var p [32]byte
	copy(p[:], sym)
	p[31] = 0xD0
	return p
}

// assetID mirrors pkg/dchain.a32: a label packed big-endian into the low 8 bytes.
func assetID(label uint64) [32]byte {
	var a [32]byte
	binary.BigEndian.PutUint64(a[24:32], label)
	return a
}

// contentRef mirrors pkg/dchain.contentRef so deposits are exactly-once on retry
// and genuinely distinct otherwise.
func contentRef(typ byte, user string, asset [32]byte, amount uint64) [32]byte {
	var ref [32]byte
	copy(ref[:], asset[:])
	ref[0] ^= typ
	for i := 0; i < len(user) && i+1 < 24; i++ {
		ref[1+i] ^= user[i]
	}
	binary.BigEndian.PutUint64(ref[24:32], binary.BigEndian.Uint64(asset[24:32])^amount)
	return ref
}

// txDeposit is dchain.TxDeposit's wire tag (the type byte mixed into the ref).
// Kept in sync with pkg/dchain tx kinds; deposits use the deposit tag.
const txDeposit = 1

// v4SubmitFrame builds the EXACT 66-byte engine_zap clob_submit frame (mirrors
// pkg/dchain.v4SubmitFrame and lux/precompile/dex engine_zap.go Swap()).
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

// zapConn is a thin live ZAP client over rpc.ZAPDial used for seeding +
// observation + the V4/ZAP crosses.
type zapConn struct {
	conn *rpc.ZAPConn
	ctx  context.Context
}

func dialVenue(addr string) (*zapConn, func()) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	conn, err := rpc.ZAPDial(ctx, addr)
	if err != nil {
		cancel()
		die("ZAPDial(%s): %v", addr, err)
	}
	return &zapConn{conn: conn, ctx: ctx}, func() { _ = conn.Close(); cancel() }
}

func (c *zapConn) ensureMarket(pool [32]byte) {
	if _, err := c.conn.Call(c.ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pool)); err != nil {
		die("clob_ensure_market: %v", err)
	}
}

func (c *zapConn) openMarket(pool, base, quote [32]byte) {
	resp, err := c.conn.Call(c.ctx, zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(pool, base, quote))
	if err != nil {
		die("clob_open_market: %v", err)
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		die("clob_open_market not placed: resp=%x", resp)
	}
}

func (c *zapConn) deposit(user string, asset [32]byte, amount uint64) {
	ref := contentRef(byte(txDeposit), user, asset, amount)
	resp, err := c.conn.Call(c.ctx, zapwire.MethodDeposit, zapwire.EncodeDeposit(user, asset, amount, ref))
	if err != nil {
		die("clob_deposit %s %d: %v", user, amount, err)
	}
	// The live venue ACKs StatusPlaced (accepted into the sealer) but credits the
	// custody ledger when the block seals — so confirm via the read method rather
	// than trusting a synchronous credited echo (which this build returns as 0).
	if status, _, derr := zapwire.DecodeBalanceResp(resp); derr == nil && status == zapwire.StatusRejected {
		die("clob_deposit %s %d: REJECTED", user, amount)
	}
	c.waitAvailable(user, asset, amount)
}

// waitAvailable polls clob_balance until the account's AVAILABLE balance for the
// asset is at least want (the deposit/credit has sealed), or times out.
func (c *zapConn) waitAvailable(user string, asset [32]byte, want uint64) {
	deadline := time.Now().Add(20 * time.Second)
	for {
		if a, _ := c.balance(user, asset); a >= want {
			return
		}
		if time.Now().After(deadline) {
			a, lck := c.balance(user, asset)
			die("clob_deposit/credit %s: available=%d locked=%d, want available>=%d (sealer not crediting)", user, a, lck, want)
		}
		time.Sleep(150 * time.Millisecond)
	}
}

// waitBalance polls until (available, locked) for user/asset match exactly.
func (c *zapConn) waitBalance(tag, user string, asset [32]byte, wantA, wantL uint64) {
	deadline := time.Now().Add(20 * time.Second)
	for {
		a, lck := c.balance(user, asset)
		if a == wantA && lck == wantL {
			return
		}
		if time.Now().After(deadline) {
			die("%s: %s balance=(%d,%d), want (%d,%d)", tag, user, a, lck, wantA, wantL)
		}
		time.Sleep(150 * time.Millisecond)
	}
}

// waitDepth polls until the book's order count and remaining match (the resting
// asks / consumed crosses have sealed).
func (c *zapConn) waitDepth(tag string, pool [32]byte, wantOrders int, wantRem float64) {
	deadline := time.Now().Add(20 * time.Second)
	for {
		n, rem, ok := c.depth(pool)
		if ok && n == wantOrders && rem == wantRem {
			return
		}
		if time.Now().After(deadline) {
			die("%s: book orders=%d remaining=%v ok=%v, want %d / %v", tag, n, rem, ok, wantOrders, wantRem)
		}
		time.Sleep(150 * time.Millisecond)
	}
}

func (c *zapConn) place(pool [32]byte, side uint8, price, size float64, user string) {
	resp, err := c.conn.Call(c.ctx, zapwire.MethodPlace, zapwire.EncodePlace(pool, side, price, size, user))
	if err != nil {
		die("clob_place %s: %v", user, err)
	}
	if _, status, derr := zapwire.DecodeAck(resp); derr != nil || status != zapwire.StatusPlaced {
		die("clob_place %s: status=%d err=%v", user, status, derr)
	}
}

func (c *zapConn) submit(pool [32]byte, side uint8, limit, size float64, user string) []zapwire.Fill {
	resp, err := c.conn.Call(c.ctx, zapwire.MethodSubmit, zapwire.EncodeSubmit(pool, side, false, limit, size, user))
	if err != nil {
		die("clob_submit %s: %v", user, err)
	}
	fills, err := zapwire.DecodeFills(resp)
	if err != nil {
		die("clob_submit %s fills decode: %v", user, err)
	}
	return fills
}

// submitFrame sends a pre-built frame (the V4 path) and decodes fills.
func (c *zapConn) submitFrame(frame []byte) []zapwire.Fill {
	resp, err := c.conn.Call(c.ctx, zapwire.MethodSubmit, frame)
	if err != nil {
		die("V4 clob_submit: %v", err)
	}
	fills, err := zapwire.DecodeFills(resp)
	if err != nil {
		die("V4 fills decode: %v", err)
	}
	return fills
}

// depth reads clob_depth: orders, remaining, ok.
func (c *zapConn) depth(pool [32]byte) (orders int, remaining float64, ok bool) {
	resp, err := c.conn.Call(c.ctx, clobDepthMethod, pool[:])
	if err != nil {
		die("clob_depth: %v", err)
	}
	if len(resp) < 29 {
		die("clob_depth short resp: %x", resp)
	}
	orders = int(binary.BigEndian.Uint32(resp[0:4]))
	remaining = math.Float64frombits(binary.BigEndian.Uint64(resp[4:12]))
	ok = resp[28] == 1
	return
}

// balance reads clob_balance: available, locked.
func (c *zapConn) balance(user string, asset [32]byte) (available, locked uint64) {
	req := make([]byte, zapwire.UserSize+zapwire.AssetIDSize)
	copy(req[0:zapwire.UserSize], user)
	copy(req[zapwire.UserSize:], asset[:])
	resp, err := c.conn.Call(c.ctx, clobBalanceMethod, req)
	if err != nil {
		die("clob_balance %s: %v", user, err)
	}
	if len(resp) < 16 {
		die("clob_balance short resp: %x", resp)
	}
	available = binary.BigEndian.Uint64(resp[0:8])
	locked = binary.BigEndian.Uint64(resp[8:16])
	return
}

// --- FIX path client (mirrors pkg/dchain fixClient: a minimal FIX 4.4 client
// over raw TCP that speaks the SAME pkg/fix codec the acceptor uses, with a
// monotonic outbound seq and length-prefixed framed reads). ---

type fixClient struct {
	conn   net.Conn
	r      *bufio.Reader
	sender string
	target string
	seq    int
}

func dialFIX(addr, sender, target string) *fixClient {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		die("dial FIX acceptor %s: %v", addr, err)
	}
	return &fixClient{conn: conn, r: bufio.NewReader(conn), sender: sender, target: target, seq: 1}
}

func (c *fixClient) send(m *fix.Message) {
	m.Set(fix.TagSenderCompID, c.sender).
		Set(fix.TagTargetCompID, c.target).
		SetInt(fix.TagMsgSeqNum, c.seq)
	c.seq++
	if _, err := c.conn.Write(m.Serialize()); err != nil {
		die("FIX client write: %v", err)
	}
}

func (c *fixClient) recv() *fix.Message {
	_ = c.conn.SetReadDeadline(time.Now().Add(15 * time.Second))
	begin, err := c.r.ReadBytes(fix.SOH)
	if err != nil {
		die("FIX recv begin: %v", err)
	}
	blField, err := c.r.ReadBytes(fix.SOH)
	if err != nil {
		die("FIX recv bodylen: %v", err)
	}
	n, err := strconv.Atoi(string(blField[2 : len(blField)-1]))
	if err != nil {
		die("FIX recv bad bodylen %q: %v", blField, err)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(c.r, body); err != nil {
		die("FIX recv body: %v", err)
	}
	trailer := make([]byte, 7)
	if _, err := io.ReadFull(c.r, trailer); err != nil {
		die("FIX recv trailer: %v", err)
	}
	frame := make([]byte, 0, len(begin)+len(blField)+len(body)+len(trailer))
	frame = append(frame, begin...)
	frame = append(frame, blField...)
	frame = append(frame, body...)
	frame = append(frame, trailer...)
	m, err := fix.Parse(frame)
	if err != nil {
		die("FIX recv parse: %v (frame=%q)", err, frame)
	}
	return m
}

func (c *fixClient) logon() {
	c.send(fix.NewMessage(fix.MsgTypeLogon).SetInt(fix.TagEncryptMethod, 0).SetInt(fix.TagHeartBtInt, 5))
	reply := c.recv()
	if reply.MsgType() != fix.MsgTypeLogon {
		die("FIX logon reply not Logon: %q", reply.MsgType())
	}
}

func fixCross(addr, user, symbol string, limit, size float64) []zapwire.Fill {
	cli := dialFIX(addr, user, "LXDEX")
	defer cli.conn.Close()
	cli.logon()
	cli.send(fix.NewMessage(fix.MsgTypeNewOrderSingle).
		Set(fix.TagClOrdID, "FIX-"+user).
		Set(fix.TagSymbol, symbol).
		Set(fix.TagSide, fix.SideBuy).
		Set(fix.TagOrderQty, fmt.Sprintf("%g", size)).
		Set(fix.TagOrdType, fix.OrdTypeLimit).
		Set(fix.TagPrice, fmt.Sprintf("%g", limit)))
	var fills []zapwire.Fill
	for {
		m := cli.recv()
		if m.MsgType() != fix.MsgTypeExecutionReport {
			die("FIX: expected ExecutionReport, got %q", m.MsgType())
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
		case fix.ExecTypeCanceled:
			return fills
		case fix.ExecTypeRejected:
			text, _ := m.Get(fix.TagText)
			die("FIX order rejected: %s", text)
		default:
			die("FIX: unexpected ExecType %q", execType)
		}
	}
}

// --- WS path client (mirrors pkg/dchain.wsCross). JSON order over gorilla/ws. ---

func wsCross(wsURL, user, symbol string, limit, size float64) []zapwire.Fill {
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		die("WS dial: %v", err)
	}
	defer conn.Close()
	if err := conn.WriteJSON(map[string]any{
		"type": "submit", "symbol": symbol, "side": "buy",
		"orderQty": size, "price": limit, "isMarket": false,
		"user": user, "clOrdId": "WS-" + user,
	}); err != nil {
		die("WS write: %v", err)
	}
	_ = conn.SetReadDeadline(time.Now().Add(15 * time.Second))
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
		die("WS read: %v", err)
	}
	if resp.Type == "error" {
		die("WS order error: %s", resp.Error)
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

func main() {
	var (
		venueAddr = flag.String("venue", "127.0.0.1:9099", "live D-Chain venue ZAP address (host:port)")
		cchainRPC = flag.String("cchain-rpc", "", "optional C-Chain EVM JSON-RPC; if set, assert code at 0x9010")
		runTag    = flag.String("run", "", "unique run tag (default: wall-clock ns) to namespace symbol+users")
	)
	flag.Parse()

	tag := *runTag
	if tag == "" {
		tag = fmt.Sprintf("%d", time.Now().UnixNano())
	}
	symbol := "LUX/LUSD#" + tag
	pool := poolIDForSymbol(symbol)
	// Per-run asset ids: a fixed "LUX"/"LUSD" tag in the high bytes plus a
	// run-derived discriminant in the low bytes, so conservation is measured on
	// THIS run's ledger rows only (the shared live venue may carry rows from
	// prior runs under different ids).
	disc := hashTag(tag) // 24-bit
	assetLUX := assetID(0x4c5558<<32 | disc)          // "LUX"  | disc
	assetLUSD := assetID(0x4c55534400<<24 | disc)      // "LUSD" | disc

	const askSize = 1.0
	maker := dexvenue.UserHandle("mk-" + tag)
	type leg struct {
		path  string
		user  string
		price float64
	}
	legs := []leg{
		{"V4", dexvenue.UserHandle("v4-" + tag), 10.0},
		{"ZAP", dexvenue.UserHandle("zp-" + tag), 20.0},
		{"FIX", dexvenue.UserHandle("fx-" + tag), 30.0},
		{"WS", dexvenue.UserHandle("ws-" + tag), 40.0},
	}

	fmt.Printf("=== FOUR-PATH LIVE PROOF (devnet) ===\n")
	fmt.Printf("  venue=%s  symbol=%s\n", *venueAddr, symbol)

	// Optional: assert the on-chain V4 precompile is present at 0x9010.
	if *cchainRPC != "" {
		assertPrecompile(*cchainRPC)
	}

	cli, closeCli := dialVenue(*venueAddr)
	defer closeCli()

	// --- seed custody + market over ZAP (the canonical custody wire path) ---
	cli.ensureMarket(pool)
	cli.openMarket(pool, assetLUX, assetLUSD)
	cli.deposit(maker, assetLUX, uint64(len(legs)))
	var totalQuoteIn uint64
	for _, l := range legs {
		cli.deposit(l.user, assetLUSD, uint64(l.price))
		totalQuoteIn += uint64(l.price)
	}
	cli.waitBalance("seed-maker-LUX", maker, assetLUX, uint64(len(legs)), 0)
	fmt.Printf("  seeded: maker=%d LUX; takers fund %d LUSD total (10+20+30+40)\n", len(legs), totalQuoteIn)

	// --- maker rests one SELL ask of 1 LUX per leg price ---
	for _, l := range legs {
		cli.place(pool, zapwire.SideSell, l.price, askSize, maker)
	}
	wantRem := float64(len(legs)) * askSize
	cli.waitBalance("rest-maker-LUX", maker, assetLUX, 0, uint64(len(legs))) // all LUX locked in asks
	cli.waitDepth("seeded-asks", pool, len(legs), wantRem)
	fmt.Printf("  maker rested %d asks of %g LUX @ {10,20,30,40} LUSD (%d LUX locked)\n", len(legs), askSize, len(legs))

	// --- stand up FIX acceptor + WS front-end, BOTH onto the SAME live venue ---
	fixVenue, err := dexvenue.DialVenue(context.Background(), *venueAddr)
	if err != nil {
		die("FIX venue dial: %v", err)
	}
	acc, err := fix.NewAcceptor(fix.Config{Addr: "127.0.0.1:0", Venue: fixVenue, TargetCompID: "LXDEX", HeartBtInt: 5})
	if err != nil {
		die("NewAcceptor: %v", err)
	}
	fixCtx, fixCancel := context.WithCancel(context.Background())
	defer fixCancel()
	go func() { _ = acc.Serve(fixCtx) }()
	defer acc.Close()

	wsVenueConn, err := dexvenue.DialVenue(context.Background(), *venueAddr)
	if err != nil {
		die("WS venue dial: %v", err)
	}
	wsHandler := dexapi.NewVenueWS(wsVenueConn)
	httpSrv := httptest.NewServer(http.HandlerFunc(wsHandler.HandleConnection))
	defer httpSrv.Close()
	wsURL := "ws" + strings.TrimPrefix(httpSrv.URL, "http")
	fmt.Printf("  FIX acceptor=%s  WS endpoint=%s/ws  (both -> live venue %s)\n", acc.Addr(), wsURL, *venueAddr)

	expectedRemaining := wantRem
	crossOne := func(l leg, fills []zapwire.Fill) {
		if len(fills) != 1 {
			die("%s: got %d fills, want exactly 1", l.path, len(fills))
		}
		f := fills[0]
		if f.Size != askSize || f.Price != l.price {
			die("%s: fill %g@%g, want %g@%g", l.path, f.Size, f.Price, askSize, l.price)
		}
		expectedRemaining -= askSize
		cli.waitDepth(l.path+"-book", pool, len(legs)-int(wantRem-expectedRemaining), expectedRemaining)
		// Per-path custody change: the taker received 1 LUX, spent its price in
		// LUSD (limit*size == lock exactly, so 0 refunded, 0 left).
		cli.waitBalance(l.path+"-taker-LUX", l.user, assetLUX, 1, 0)
		cli.waitBalance(l.path+"-taker-LUSD", l.user, assetLUSD, 0, 0)
		fmt.Printf("  [%s] FILL %g LUX @ %g LUSD (consensus-settled); book now %g; taker = 1 LUX / 0 LUSD\n",
			l.path, f.Size, f.Price, expectedRemaining)
	}
	legByPath := func(name string) leg {
		for _, l := range legs {
			if l.path == name {
				return l
			}
		}
		die("no leg %q", name)
		return leg{}
	}

	// ===== PATH 1: V4 precompile (engine_zap boundary) =====
	{
		l := legByPath("V4")
		frame := v4SubmitFrame(pool, zapwire.SideBuy, false, l.price, askSize, l.user)
		want := zapwire.EncodeSubmit(pool, zapwire.SideBuy, false, l.price, askSize, l.user)
		if string(frame) != string(want) {
			die("V4 engine_zap frame != zapwire.EncodeSubmit (frozen-frame parity broken)")
		}
		fmt.Printf("  [V4] engine_zap frame == zapwire.EncodeSubmit (66-byte frozen-frame parity OK)\n")
		crossOne(l, cli.submitFrame(frame))
	}
	// ===== PATH 2: ZAP native =====
	{
		l := legByPath("ZAP")
		crossOne(l, cli.submit(pool, zapwire.SideBuy, l.price, askSize, l.user))
	}
	// ===== PATH 3: FIX =====
	{
		l := legByPath("FIX")
		crossOne(l, fixCross(acc.Addr(), l.user, symbol, l.price, askSize))
	}
	// ===== PATH 4: WS =====
	{
		l := legByPath("WS")
		crossOne(l, wsCross(wsURL+"/ws", l.user, symbol, l.price, askSize))
	}

	// --- all asks consumed: book empty, maker fully settled ---
	cli.waitDepth("final-book", pool, 0, 0.0)
	cli.waitBalance("final-maker-LUX", maker, assetLUX, 0, 0)
	cli.waitBalance("final-maker-LUSD", maker, assetLUSD, totalQuoteIn, 0)

	// --- GLOBAL CONSERVATION (I = available + locked) per asset, every account ---
	accounts := []string{maker}
	for _, l := range legs {
		accounts = append(accounts, l.user)
	}
	conserved := func(asset [32]byte) uint64 {
		var total uint64
		for _, u := range accounts {
			a, lck := cli.balance(u, asset)
			total += a + lck
		}
		return total
	}
	totalLUX := conserved(assetLUX)
	totalLUSD := conserved(assetLUSD)
	if totalLUX != uint64(len(legs)) {
		die("LUX conservation: total available+locked=%d, want %d", totalLUX, len(legs))
	}
	if totalLUSD != totalQuoteIn {
		die("LUSD conservation: total available+locked=%d, want %d", totalLUSD, totalQuoteIn)
	}

	fmt.Printf("\n=== FOUR-PATH LIVE PROOF: PASS ===\n")
	fmt.Printf("  one LIVE venue, one LUX/LUSD book; a REAL consensus FILL over EACH path:\n")
	fmt.Printf("    V4 precompile (engine_zap frame) | ZAP native | FIX acceptor | WS venue\n")
	fmt.Printf("  %d asks consumed; maker 0 LUX / %d LUSD; each taker 1 LUX / 0 LUSD\n", len(legs), totalQuoteIn)
	fmt.Printf("  CONSERVATION: LUX=%d (=%d in), LUSD=%d (=%d in) — available+locked, nothing minted\n",
		totalLUX, len(legs), totalLUSD, totalQuoteIn)
}

// hashTag folds a run tag into a 24-bit value used to namespace per-run asset ids.
func hashTag(s string) uint64 {
	var h uint64 = 1469598103934665603
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h & 0xffffff
}

// assertPrecompile confirms the V4 precompile is deployed at 0x9010 on the
// C-Chain EVM (eth_getCode returns non-empty), proving the on-chain leg of the
// V4 path exists where the engine_zap frame is relayed from.
func assertPrecompile(rpcURL string) {
	body := `{"jsonrpc":"2.0","id":1,"method":"eth_getCode","params":["0x0000000000000000000000000000000000009010","latest"]}`
	req, _ := http.NewRequest("POST", rpcURL, strings.NewReader(body))
	req.Header.Set("content-type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		die("eth_getCode 0x9010: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Result string `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		die("eth_getCode decode: %v", err)
	}
	if out.Result == "" || out.Result == "0x" {
		die("V4 precompile 0x9010 has no code on C-Chain (%q)", out.Result)
	}
	fmt.Printf("  [V4] on-chain precompile 0x9010 present on C-Chain (eth_getCode=%s)\n", out.Result)
}
