// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
)

// ingest_test.go proves the IN-LUXD ingestion seam (ingest.go / CreateHandlers):
// an order submitted over the VM's HTTP handler — the EXACT handler luxd mounts at
// /v1/bc/<D>/dex/<method> — lands in the mempool and is matched into a block by
// the VM's own consensus lifecycle (BuildBlock -> Verify -> Accept), with the fill
// committed as D-Chain state (a trade:<height><seq> row). The matcher is the
// chain; there is no external venue and no relay.
//
// The transport under test is vm.dexHTTPHandler() driven through a real
// httptest.Server (which is exactly what the plugin transport stands up inside the
// plugin process). An auto-sealer goroutine plays the consensus engine, identical
// to handler_test.go's socket harness — proving the SAME submitTx->mempool->
// Verify-match path is reached whether the order arrives over HTTP or ZAP.

// startIngestServer brings up a VM + the in-luxd HTTP DEX handler over a real
// httptest.Server, plus the auto-sealer. Returns the VM, the base URL
// (.../dex), and a stop func.
func startIngestServer(t *testing.T) (*VM, string, func()) {
	t.Helper()
	db := memdb.New()
	vm := &VM{}
	toEngine := make(chan block.Message, 64)
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: toEngine,
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	// Model the node EXACTLY: CreateHandlers returns one handler per method keyed
	// by its full sub-path ("/dex/<method>"); luxd registers each as an exact route
	// under /v1/bc/<chainID>. We register each key as an exact route here, so the
	// request paths and matching semantics match production (exact path, not a
	// subtree — node/server/http/router.go uses gorilla Handle, not PathPrefix).
	handlers, err := vm.CreateHandlers(context.Background())
	if err != nil {
		t.Fatalf("CreateHandlers: %v", err)
	}
	if len(handlers) == 0 {
		t.Fatal("CreateHandlers returned no handlers")
	}
	root := http.NewServeMux()
	for path, h := range handlers {
		// gorilla matches exact paths; net/http ServeMux matches exact when the
		// pattern has no trailing slash, which mirrors the node's exact-route
		// behavior for these method endpoints.
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

// methodURL builds the full ingestion URL for a method: <base>/dex/<method>.
func methodURL(base, method string) string {
	return base + "/" + ingestNamespace + "/" + method
}

// postFrame POSTs a frozen frame to <base>/dex/<method> and returns the raw
// response body. It fails the test on any transport error or non-200 status.
func postFrame(t *testing.T, base, method string, payload []byte) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, methodURL(base, method), bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("new request %s: %v", method, err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", method, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read %s response: %v", method, err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s: status %d body=%q", method, resp.StatusCode, string(body))
	}
	return body
}

// TestIngestCrossingOrdersMatchInBlock is the deliverable proof of the ingestion
// seam: two crossing orders submitted through the in-luxd HTTP handler produce a
// fill, and that fill is recorded as D-Chain state (a trade: row), matched by the
// VM's BuildBlock->Verify->Accept — not by any external matcher.
func TestIngestCrossingOrdersMatchInBlock(t *testing.T) {
	vm, base, stop := startIngestServer(t)
	defer stop()

	var pool [32]byte
	pool[0], pool[1] = 0xab, 0xcd

	// 1) ensure_market over HTTP -> ack(placed).
	ack := postFrame(t, base, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	if _, status, err := zapwire.DecodeAck(ack); err != nil || status != zapwire.StatusPlaced {
		t.Fatalf("ensure_market ack: status=%d err=%v", status, err)
	}

	// 2) place a resting ask @101 size 5 over HTTP -> ack(placed) with order id. The
	// money-moving frame is signed by the maker's key-derived account (the auth gate
	// refuses an unsigned place); the auth envelope rides as a trailer on the frame.
	ack = postFrame(t, base, zapwire.MethodPlace,
		signedPayload(t, "maker", TxPlace, encPlace(pool, zapwire.SideSell, 101.0, 5.0, wireUser(t, "maker"))))
	makerID, status, err := zapwire.DecodeAck(ack)
	if err != nil || status != zapwire.StatusPlaced {
		t.Fatalf("place ack: status=%d err=%v", status, err)
	}
	if makerID == 0 {
		t.Fatal("place returned zero order id")
	}

	// 3) submit a crossing buy @101 size 3 over HTTP -> CONSENSUS-computed fills.
	fillsResp := postFrame(t, base, zapwire.MethodSubmit,
		signedPayload(t, "taker", TxSubmit, encSubmit(pool, zapwire.SideBuy, false, 101.0, 3.0, wireUser(t, "taker"))))
	fills, err := zapwire.DecodeFills(fillsResp)
	if err != nil {
		t.Fatalf("DecodeFills: %v", err)
	}
	if len(fills) != 1 {
		t.Fatalf("expected 1 fill, got %d: %+v", len(fills), fills)
	}
	if fills[0].TakerSide != zapwire.SideBuy || fills[0].Size != 3 || fills[0].Price != 101*uint64(zapwire.PriceScale) {
		t.Fatalf("fill = %+v, want buy 3 @ 101e8", fills[0])
	}

	// 4) STATE PROOF: a trade:<height><seq> row exists — the fill is committed
	// D-Chain state, written by Block.Accept (settle.go), not a synchronous book
	// mutation. Scan the durable DB the VM committed into.
	if n := countTradeRows(t, vm); n != 1 {
		t.Fatalf("expected exactly 1 committed trade: row, got %d", n)
	}

	// 5) BLOCK PROOF: the chain advanced past genesis — the order was sequenced
	// into a real block by consensus. (genesis is height 0; ensure_market, place,
	// and submit each seal a block, so height >= 3.)
	vm.mu.Lock()
	h := vm.lastAcceptedHeight
	vm.mu.Unlock()
	if h < 3 {
		t.Fatalf("lastAcceptedHeight = %d, want >= 3 (orders must seal blocks)", h)
	}
	t.Logf("native fill committed: trade rows=1, lastAcceptedHeight=%d", h)
}

// countTradeRows counts trade:* rows in the VM's durable DB — the authoritative
// committed fill log. Read under the VM lock (the sealer also touches the DB).
func countTradeRows(t *testing.T, vm *VM) int {
	t.Helper()
	vm.mu.Lock()
	defer vm.mu.Unlock()
	it := vm.db.NewIteratorWithPrefix([]byte(prefixTrade))
	defer it.Release()
	n := 0
	for it.Next() {
		// trade:<height:8><seq:8>
		if len(it.Key()) == len(prefixTrade)+16 {
			n++
		}
	}
	if err := it.Error(); err != nil {
		t.Fatalf("trade iterator: %v", err)
	}
	return n
}

// TestIngestRejectsNonPost proves the boundary contract: the ingestion handler is
// POST-only (an order is a write); a GET is 405 and never reaches the mempool.
func TestIngestRejectsNonPost(t *testing.T) {
	_, base, stop := startIngestServer(t)
	defer stop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, methodURL(base, zapwire.MethodSubmit), nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want 405", resp.StatusCode)
	}
}

// TestIngestUnknownMethod404 proves an unrouted method path is a 404 (the mux only
// mounts the frozen dex_* methods); it must not panic or hang.
func TestIngestUnknownMethod404(t *testing.T) {
	_, base, stop := startIngestServer(t)
	defer stop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, methodURL(base, "dex_bogus"), bytes.NewReader([]byte{0x00}))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("unknown method status = %d, want 404", resp.StatusCode)
	}
}

// sanity: the ingestion response for a place ack carries the order id in the first
// 8 bytes big-endian (frozen ack frame), independent of transport.
func TestIngestAckFrameShape(t *testing.T) {
	_, base, stop := startIngestServer(t)
	defer stop()

	var pool [32]byte
	pool[0] = 0x07
	postFrame(t, base, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	ack := postFrame(t, base, zapwire.MethodPlace,
		signedPayload(t, "u", TxPlace, encPlace(pool, zapwire.SideBuy, 5.0, 2.0, wireUser(t, "u"))))
	if len(ack) < 9 {
		t.Fatalf("ack frame too short: %d", len(ack))
	}
	if id := binary.BigEndian.Uint64(ack[0:8]); id == 0 {
		t.Fatal("ack order id is zero")
	}
}
