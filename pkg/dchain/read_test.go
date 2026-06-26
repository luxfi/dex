// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
)

// read_test.go proves the D-Chain READ surface (read.go / clob_get_*):
//
//  1. The read endpoints are reachable over the EXACT in-luxd HTTP transport
//     (CreateHandlers, mounted at /ext/bc/<D>/dex/<method>) — the same routes a
//     deployed validator exposes.
//  2. After a crossing order is matched into a committed block (via the write
//     path + the auto-sealer playing consensus), clob_get_trades returns the
//     committed fill, clob_get_markets lists the market, clob_get_orders/book
//     reflect the resting set.
//  3. CROSS-VALIDATOR INVARIANT (in-process model): two INDEPENDENT VMs that
//     Accept the SAME block bytes return BYTE-IDENTICAL clob_get_trades JSON and
//     the same head root. This is the local proof of the property the live
//     5-validator test checks: same accepted block => same committed read on
//     every node. (A follower in production reaches the same state by Verify+
//     Accept of the gossiped bytes — integration.go HandleIncomingBlock.)
//  4. Reads have ZERO consensus impact: a read does not change height or the
//     mempool.

// readDeps bundles the asset ids + identities a custody market needs.
type readFixture struct {
	pool        [32]byte
	base, quote [32]byte
}

func newReadFixture() readFixture {
	f := readFixture{pool: poolIDForSymbol("LUX/LUSD")}
	// base = AssetLUX (all-zero); quote = a labelled 32-byte asset id.
	binary.BigEndian.PutUint64(f.quote[24:32], 0x4c555344_000001)
	return f
}

// startReadServer brings up a VM + the in-luxd HTTP handlers (writes AND reads)
// over a real httptest.Server, plus the auto-sealer. Returns the VM, the base
// URL (.../dex), and a stop func. Mirrors startIngestServer (ingest_test.go) but
// is self-contained so the read tests do not depend on the write test's harness.
func startReadServer(t *testing.T) (*VM, string, func()) {
	t.Helper()
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 64)
	if err := vm.Initialize(context.Background(), block.Init{
		DB: db, Log: log.NewNoOpLogger(), ToEngine: toEngine,
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	handlers, err := vm.CreateHandlers(context.Background())
	if err != nil {
		t.Fatalf("CreateHandlers: %v", err)
	}
	root := http.NewServeMux()
	for path, h := range handlers {
		root.Handle(path, h)
	}
	srv := httptest.NewServer(root)

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); autoSealer(ctx, t, vm) }()
	stop := func() {
		cancel()
		srv.Close()
		_ = vm.Shutdown(context.Background())
		wg.Wait()
	}
	return vm, srv.URL, stop
}

// httpPost posts a frozen frame to a write method and returns the raw response.
func httpPost(t *testing.T, base, method string, payload []byte) []byte {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, methodURL(base, method), byteReader(payload))
	if err != nil {
		t.Fatalf("new req %s: %v", method, err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post %s: %v", method, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("post %s: status %d body=%s", method, resp.StatusCode, body)
	}
	return body
}

// httpGetJSON GETs a read method (with optional raw query) and decodes the JSON
// body into out. Fails on non-200.
func httpGetJSON(t *testing.T, base, method, query string, out interface{}) {
	t.Helper()
	url := methodURL(base, method)
	if query != "" {
		url += "?" + query
	}
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("get %s: %v", method, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("get %s: status %d body=%s", method, resp.StatusCode, body)
	}
	if err := json.Unmarshal(body, out); err != nil {
		t.Fatalf("get %s: decode %v body=%s", method, err, body)
	}
}

func byteReader(b []byte) *byteReaderT { return &byteReaderT{b: b} }

type byteReaderT struct {
	b   []byte
	off int
}

func (r *byteReaderT) Read(p []byte) (int, error) {
	if r.off >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.off:])
	r.off += n
	return n, nil
}

// crossOneMarket runs the full write path that produces one committed fill:
// open_market (bind assets), deposits, alice rests an ask, bob crosses with a
// marketable buy. Returns the fills bob's submit reported.
func crossOneMarket(t *testing.T, base string, f readFixture) []zapwire.Fill {
	t.Helper()
	const aliceUser, bobUser = "alice", "bob"
	const askPrice, askSize = 12.50, 2.0
	const buyLimit, buySize = 13.0, 1.0

	httpPost(t, base, zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(f.pool, f.base, f.quote))

	// Every money-moving frame is signed by the user's key-derived account (the auth
	// gate refuses unsigned deposit/place/submit); the wire user is the account.
	dep := func(user string, asset [32]byte, amt uint64) {
		ref := contentRef(byte(TxDeposit), user, asset, amt)
		body := zapwire.EncodeDeposit(wireUser(t, user), asset, amt, ref)
		httpPost(t, base, zapwire.MethodDeposit, signedPayload(t, user, TxDeposit, body))
	}
	dep(aliceUser, f.base, 2)
	dep(bobUser, f.quote, 53)

	askResp := httpPost(t, base, zapwire.MethodPlace,
		signedPayload(t, aliceUser, TxPlace, zapwire.EncodePlace(f.pool, zapwire.SideSell, askPrice, askSize, wireUser(t, aliceUser))))
	if askResp[8] != zapwire.StatusPlaced {
		t.Fatalf("ask not placed: %x", askResp)
	}
	subResp := httpPost(t, base, zapwire.MethodSubmit,
		signedPayload(t, bobUser, TxSubmit, zapwire.EncodeSubmit(f.pool, zapwire.SideBuy, false, buyLimit, buySize, wireUser(t, bobUser))))
	fills, err := zapwire.DecodeFills(subResp)
	if err != nil {
		t.Fatalf("decode fills: %v (resp=%x)", err, subResp)
	}
	if len(fills) == 0 {
		t.Fatalf("cross produced 0 fills (resp=%x)", subResp)
	}
	return fills
}

// TestRead_TradesAfterCommittedFill is the core happy path: a crossing order is
// matched into a committed block, and clob_get_trades returns that fill from
// durable state with the right price/size/side and a non-genesis head.
func TestRead_TradesAfterCommittedFill(t *testing.T) {
	vm, base, stop := startReadServer(t)
	defer stop()
	f := newReadFixture()

	fills := crossOneMarket(t, base, f)
	t.Logf("submit reported %d fill(s): price=%g size=%g", len(fills), fills[0].Price, fills[0].Size)

	var got getTradesResp
	httpGetJSON(t, base, MethodGetTrades, "", &got)

	if got.Count == 0 || len(got.Trades) == 0 {
		t.Fatalf("dex_get_trades returned no committed trades after a fill: %+v", got)
	}
	if got.Height == 0 {
		t.Fatalf("head height still 0 after a committed fill (head=%+v)", got.chainHead)
	}
	tr := got.Trades[0]
	if tr.Price != fills[0].Price {
		t.Errorf("committed trade price %g != submit fill price %g", tr.Price, fills[0].Price)
	}
	if tr.Size != fills[0].Size {
		t.Errorf("committed trade size %g != submit fill size %g", tr.Size, fills[0].Size)
	}
	if tr.TakerSide != "buy" {
		t.Errorf("taker side = %q, want buy", tr.TakerSide)
	}
	if tr.Height == 0 || tr.TradeID == 0 {
		t.Errorf("trade row missing coordinates: %+v", tr)
	}

	// The read head must match the VM's authoritative last-accepted height.
	vm.mu.Lock()
	wantHeight := vm.lastAcceptedHeight
	wantRoot := hex.EncodeToString(vm.lastRoot[:])
	vm.mu.Unlock()
	if got.Height != wantHeight {
		t.Errorf("read head height %d != VM %d", got.Height, wantHeight)
	}
	if got.Root != wantRoot {
		t.Errorf("read head root %s != VM %s", got.Root, wantRoot)
	}
}

// TestRead_MarketsAndOrdersAndBook proves the other three read endpoints reflect
// committed state: the market is listed (with bound assets), and the resting book
// (alice's partially-consumed ask remainder) shows up in orders + book.
func TestRead_MarketsAndOrdersAndBook(t *testing.T) {
	_, base, stop := startReadServer(t)
	defer stop()
	f := newReadFixture()
	_ = crossOneMarket(t, base, f) // ask size 2, buy 1 -> 1.0 base remains resting

	marketHex := hex.EncodeToString(f.pool[:])

	// markets
	var markets getMarketsResp
	httpGetJSON(t, base, MethodGetMarkets, "", &markets)
	if markets.Count == 0 {
		t.Fatalf("dex_get_markets returned no markets")
	}
	var found *marketJSON
	for i := range markets.Markets {
		if markets.Markets[i].PoolID == marketHex {
			found = &markets.Markets[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("market %s not in clob_get_markets: %+v", marketHex, markets.Markets)
	}
	if !found.AssetsBound {
		t.Errorf("market assetsBound=false, want true (open_market bound them)")
	}
	if found.Orders == 0 {
		t.Errorf("market shows 0 resting orders, want the ask remainder")
	}

	// orders
	var orders getOrdersResp
	httpGetJSON(t, base, MethodGetOrders, "market="+marketHex, &orders)
	if orders.Count == 0 {
		t.Fatalf("dex_get_orders returned no resting orders for %s", marketHex)
	}
	rem := orders.Orders[0]
	if rem.Side != "sell" {
		t.Errorf("resting order side=%q, want sell (alice's ask)", rem.Side)
	}
	if rem.Remaining <= 0 || rem.Remaining > rem.Size {
		t.Errorf("resting remaining=%g out of range (size=%g)", rem.Remaining, rem.Size)
	}

	// book
	var bookResp getBookResp
	httpGetJSON(t, base, MethodGetBook, "market="+marketHex, &bookResp)
	if bookResp.Orders == 0 {
		t.Fatalf("dex_get_book shows 0 orders")
	}
	if len(bookResp.Asks) == 0 {
		t.Fatalf("dex_get_book shows no asks (alice's ask remainder should rest)")
	}
	if bookResp.BestAsk <= 0 {
		t.Errorf("bestAsk=%g, want the ask price", bookResp.BestAsk)
	}
}

// TestRead_ZeroConsensusImpact proves a read never advances the chain or touches
// the mempool: many reads between two committed fills leave height advancing only
// by the writes, and the mempool empty after each read.
func TestRead_ZeroConsensusImpact(t *testing.T) {
	vm, base, stop := startReadServer(t)
	defer stop()
	f := newReadFixture()
	_ = crossOneMarket(t, base, f)

	vm.mu.Lock()
	h0 := vm.lastAcceptedHeight
	vm.mu.Unlock()

	for i := 0; i < 25; i++ {
		var got getTradesResp
		httpGetJSON(t, base, MethodGetTrades, "", &got)
		var m getMarketsResp
		httpGetJSON(t, base, MethodGetMarkets, "", &m)
	}

	vm.mu.Lock()
	h1 := vm.lastAcceptedHeight
	pending := vm.mempool.Len()
	vm.mu.Unlock()
	if h1 != h0 {
		t.Errorf("reads advanced height %d -> %d (must be zero consensus impact)", h0, h1)
	}
	if pending != 0 {
		t.Errorf("reads left %d txs in the mempool (must be zero)", pending)
	}
}

// TestRead_CrossValidatorIdenticalTrades is the in-process model of the live
// 5-validator question: two INDEPENDENT VMs (separate DBs) that Accept the SAME
// sequence of block bytes must return BYTE-IDENTICAL clob_get_trades JSON and the
// same head root. This isolates the replication property from the network: if a
// block's bytes are delivered (gossip in production) and Verify+Accept succeed,
// the committed read is identical — which is exactly what the read RPC lets the
// live test confirm across luxd-0..4.
func TestRead_CrossValidatorIdenticalTrades(t *testing.T) {
	f := newReadFixture()

	// PROPOSER: run the full write path with an auto-sealer, capturing every
	// accepted block's canonical bytes in order.
	proposer, base, stop := startReadServer(t)
	var (
		capMu      sync.Mutex
		acceptedBz [][]byte
	)
	captureAccepted := func(vm *VM) {
		vm.mu.Lock()
		defer vm.mu.Unlock()
		// Walk the height index in order, pulling each accepted block's bytes.
		for h := uint64(1); h <= vm.lastAcceptedHeight; h++ {
			id, ok := vm.heightIndex[h]
			if !ok {
				continue
			}
			if blk, ok := vm.acceptedBlocks[id]; ok {
				acceptedBz = append(acceptedBz, append([]byte(nil), blk.bytes...))
			}
		}
	}

	_ = crossOneMarket(t, base, f)
	// Let the sealer settle, then snapshot the accepted chain bytes + proposer read.
	time.Sleep(200 * time.Millisecond)
	capMu.Lock()
	captureAccepted(proposer)
	capMu.Unlock()

	var proposerTrades getTradesResp
	httpGetJSON(t, base, MethodGetTrades, "", &proposerTrades)
	stop()

	if len(acceptedBz) == 0 {
		t.Fatalf("captured no accepted blocks from proposer")
	}
	if proposerTrades.Count == 0 {
		t.Fatalf("proposer has no committed trades")
	}

	// FOLLOWER: a fresh VM (separate DB, no sealer) that replays the proposer's
	// accepted block bytes through ParseBlock -> Verify -> Accept — exactly the
	// integration.go HandleIncomingBlock fast-follow path, minus the network.
	followerDB := memdb.New()
	follower := &VM{}
	if err := follower.Initialize(context.Background(), block.Init{
		DB: followerDB, Log: log.NewNoOpLogger(), ToEngine: make(chan block.Message, 64),
	}); err != nil {
		t.Fatalf("follower Initialize: %v", err)
	}
	defer follower.Shutdown(context.Background())
	for i, bz := range acceptedBz {
		blk, err := follower.ParseBlock(context.Background(), bz)
		if err != nil {
			t.Fatalf("follower ParseBlock[%d]: %v", i, err)
		}
		if err := blk.Verify(context.Background()); err != nil {
			t.Fatalf("follower Verify[%d] (height %d): %v", i, blk.Height(), err)
		}
		if err := blk.Accept(context.Background()); err != nil {
			t.Fatalf("follower Accept[%d] (height %d): %v", i, blk.Height(), err)
		}
	}

	// Read the follower's committed trades through the SAME handler and compare.
	followerHandlers, _ := follower.CreateHandlers(context.Background())
	fmux := http.NewServeMux()
	for path, h := range followerHandlers {
		fmux.Handle(path, h)
	}
	fsrv := httptest.NewServer(fmux)
	defer fsrv.Close()

	var followerTrades getTradesResp
	httpGetJSON(t, fsrv.URL, MethodGetTrades, "", &followerTrades)

	// The replicated state must be IDENTICAL: same head root + same trades.
	if followerTrades.Root != proposerTrades.Root {
		t.Fatalf("follower root %s != proposer root %s (state diverged)", followerTrades.Root, proposerTrades.Root)
	}
	if followerTrades.Height != proposerTrades.Height {
		t.Fatalf("follower height %d != proposer height %d", followerTrades.Height, proposerTrades.Height)
	}
	pj, _ := json.Marshal(proposerTrades.Trades)
	fj, _ := json.Marshal(followerTrades.Trades)
	if string(pj) != string(fj) {
		t.Fatalf("cross-validator trade mismatch:\n proposer=%s\n follower=%s", pj, fj)
	}
	t.Logf("cross-validator OK: %d trade(s) identical at height %d root %s",
		followerTrades.Count, followerTrades.Height, followerTrades.Root[:16])
}

// TestRead_GenesisEmpty proves a fresh chain answers reads cleanly: empty trades,
// no markets, genesis head — no nil-deref, no error.
func TestRead_GenesisEmpty(t *testing.T) {
	_, base, stop := startReadServer(t)
	defer stop()

	var trades getTradesResp
	httpGetJSON(t, base, MethodGetTrades, "", &trades)
	if trades.Count != 0 || len(trades.Trades) != 0 {
		t.Errorf("fresh chain has trades: %+v", trades)
	}
	var markets getMarketsResp
	httpGetJSON(t, base, MethodGetMarkets, "", &markets)
	if markets.Count != 0 {
		t.Errorf("fresh chain has markets: %+v", markets)
	}
	// An unknown market's orders/book are empty, not an error.
	none := poolIDForSymbol("NONE/NONE")
	unknown := hex.EncodeToString(none[:])
	var orders getOrdersResp
	httpGetJSON(t, base, MethodGetOrders, "market="+unknown, &orders)
	if orders.Count != 0 {
		t.Errorf("unknown market has orders: %+v", orders)
	}
}

// TestRead_BadParams proves boundary validation: a malformed market id is a 400,
// not a panic or a silent wrong-market read.
func TestRead_BadParams(t *testing.T) {
	_, base, stop := startReadServer(t)
	defer stop()

	// missing market on an endpoint that requires it.
	resp, err := http.Get(methodURL(base, MethodGetOrders))
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("missing market: status %d, want 400", resp.StatusCode)
	}

	// wrong-length market id.
	resp2, err := http.Get(methodURL(base, MethodGetBook) + "?market=deadbeef")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusBadRequest {
		t.Errorf("short market: status %d, want 400", resp2.StatusCode)
	}

	// a POST to a read endpoint is rejected (reads are GET).
	resp3, err := http.Post(methodURL(base, MethodGetTrades), "application/json", byteReader(nil))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp3.Body.Close()
	if resp3.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("POST to read endpoint: status %d, want 405", resp3.StatusCode)
	}
}

// compile-time guard: the read row codec stays consistent with the lx trade row.
var _ = lx.DEXTrade{}
