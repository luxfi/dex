// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/luxfi/dex/pkg/dex"
	"github.com/luxfi/dex/pkg/zapwire"
)

// read.go is the D-Chain's READ surface: dex_get_* query endpoints that return
// COMMITTED chain state (resting orders, the trade log, markets, the book) as
// JSON. It is the orthogonal counterpart to handler.go's write surface:
//
//	handler.go  dex_place/submit/cancel/...  -> mempool -> consensus -> Verify-match.
//	            FROZEN binary zapwire frames. Mutates state through a consensus round.
//	read.go     dex_get_trades/orders/markets/book
//	            JSON. Reads committed state directly under vm.mu. ZERO consensus
//	            impact — no mempool, no block, no waiter; a read can never change a
//	            fill or a balance.
//
// Why JSON, not the frozen binary frames: the read surface is NOT on the
// consensus-critical wire (no byte-parity requirement with the proxy/precompile/
// maker — those only WRITE). It feeds two consumers that want self-describing
// records: (1) the markets-display pipeline (the indexer/exchange-api reads the
// committed trade log to render markets — native fills are trade: rows, so this
// is the source the graph/exchange-api consumes), and (2) cross-validator
// verification — querying the SAME endpoint on every validator and diffing the
// JSON proves whether the accepted fill replicated identically across the set.
//
// AUTHORITATIVE SOURCE: trades and markets are read from durable state (vm.db) —
// the committed truth, identical on every node for the same accepted height.
// Resting orders + book depth are read from the in-RAM book (vm.books), which is
// a deterministic fold of the committed order:* rows (rebuilt at Initialize,
// advanced at Accept), so it equals the durable resting set. Every response
// carries the chain head (height, lastAccepted, root) so a cross-validator diff
// is anchored to an exact accepted block — two nodes at the same height with the
// same root MUST return identical trades.

// read method names. They live in the dex_ namespace beside the write methods
// but are pure reads. The dex_ prefix means the node mounts them under the same
// /v1/chain/<D>/dex/ route group as the writes (one chain surface).
const (
	MethodGetTrades  = "dex_get_trades"
	MethodGetOrders  = "dex_get_orders"
	MethodGetMarkets = "dex_get_markets"
	MethodGetBook    = "dex_get_book"
	MethodGetNonce   = "dex_get_nonce"
)

// defaultTradeLimit bounds an unparameterized dex_get_trades response so a chain
// with a long fill history cannot return an unbounded body by default. A caller
// paging the full log passes an explicit limit + since cursor.
const defaultTradeLimit = 1000

// readMethod pairs a read method name with its http.Handler. Reads are GET (a
// query has no body and is safely cacheable/idempotent), distinct from the writes'
// POST. The table is the single source of the read surface, folded into
// httpHandlers alongside the write table.
type readMethod struct {
	method  string
	handler http.HandlerFunc
}

// readMethods is the authoritative read method->handler table. A method added
// here is mounted under /v1/chain/<D>/dex/<method> automatically (httpHandlers).
func (vm *VM) readMethods() []readMethod {
	return []readMethod{
		{MethodGetTrades, vm.handleGetTrades},
		{MethodGetOrders, vm.handleGetOrders},
		{MethodGetMarkets, vm.handleGetMarkets},
		{MethodGetBook, vm.handleGetBook},
		{MethodGetNonce, vm.handleGetNonce},
	}
}

// chainHead is the accepted-block anchor every read response carries. Two nodes
// reporting the same Height with the same Root have committed the identical
// history — so their Trades for that range MUST match byte-for-byte. This is the
// hook a cross-validator verifier compares first.
type chainHead struct {
	Height       uint64 `json:"height"`
	LastAccepted string `json:"lastAccepted"`
	Root         string `json:"root"`
}

// headLocked reads the accepted-block anchor. The caller MUST hold vm.mu (every
// read handler takes it to snapshot state + head together, so the reported head
// is exactly the block the returned rows were read at). One place builds the head
// — the four handlers do not each re-implement it.
func (vm *VM) headLocked() chainHead {
	return chainHead{
		Height:       vm.lastAcceptedHeight,
		LastAccepted: vm.lastAcceptedID.String(),
		Root:         hex.EncodeToString(vm.lastRoot[:]),
	}
}

// tradeJSON is one committed fill rendered for display/verification. Prices/sizes
// are decoded from the row's deterministic fixed-point form to human floats for
// the display feed; the integer truth stays on-chain. Coordinates (Height, Seq)
// pin the fill to its trade:<height><seq> row so paging and diffing are exact.
type tradeJSON struct {
	Height       uint64  `json:"height"`
	Seq          uint64  `json:"seq"`
	TradeID      uint64  `json:"tradeId"`
	Price        float64 `json:"price"`
	Size         float64 `json:"size"`
	TakerSide    string  `json:"takerSide"` // "buy" | "sell"
	MakerOrderID uint64  `json:"makerOrderId"`
	TakerOrderID uint64  `json:"takerOrderId"`
	Timestamp    int64   `json:"timestamp"` // unix nanos
}

// getTradesResp is the dex_get_trades body: the chain head plus the committed
// fills for the requested range, in matcher production order (height then seq).
type getTradesResp struct {
	chainHead
	Count  int         `json:"count"`
	Trades []tradeJSON `json:"trades"`
}

// handleGetTrades serves the committed trade log:
//
//	GET .../dex/dex_get_trades?since=<height>&limit=<n>
//
// since (optional): first accepted height to include (inclusive); default 0.
// limit (optional): max rows; default defaultTradeLimit.
//
// SCOPE: the trade: log is GLOBAL and height-ordered — the 80-byte on-chain trade
// row carries no poolID (it is the frozen GPU ABI), so this returns every market's
// fills in commit order, not a per-market slice. That is exactly right for the two
// consumers: a single-market devnet (the cross-validator verification case) has one
// market so the global log IS that market's log; the markets-display indexer keys
// fills to a market by the maker/taker order ids it already tracks per market. A
// true server-side per-market filter would require a separate per-market trade
// index in consensus state — a deliberate state change, not a read concern.
//
// The fills are read from durable trade:* rows — the authoritative committed log,
// identical on every validator at the same height — so this is BOTH the markets
// feed and the cross-validator replication probe.
func (vm *VM) handleGetTrades(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeReadErr(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	q := r.URL.Query()
	since, err := parseUintParam(q.Get("since"), 0)
	if err != nil {
		writeReadErr(w, http.StatusBadRequest, "bad since: "+err.Error())
		return
	}
	limit, err := parseIntParam(q.Get("limit"), defaultTradeLimit)
	if err != nil {
		writeReadErr(w, http.StatusBadRequest, "bad limit: "+err.Error())
		return
	}

	// Read committed trade rows under the VM lock so the snapshot is consistent
	// with the head we report.
	vm.mu.Lock()
	rows, rerr := readTrades(vm.db, since, limit)
	head := vm.headLocked()
	vm.mu.Unlock()
	if rerr != nil {
		writeReadErr(w, http.StatusInternalServerError, rerr.Error())
		return
	}

	resp := getTradesResp{chainHead: head, Trades: make([]tradeJSON, 0, len(rows))}
	for _, ct := range rows {
		resp.Trades = append(resp.Trades, toTradeJSON(ct))
	}
	resp.Count = len(resp.Trades)
	writeJSON(w, resp)
}

// orderJSON is one resting order rendered for display. Price/size are the float
// projection of the order's committed value (the integer lane stays on-chain).
type orderJSON struct {
	OrderID   uint64  `json:"orderId"`
	Side      string  `json:"side"` // "buy" | "sell"
	Price     float64 `json:"price"`
	Size      float64 `json:"size"`
	Remaining float64 `json:"remaining"`
	User      string  `json:"user"`
}

// getOrdersResp is the dex_get_orders body: the chain head plus the market's
// resting book rows.
type getOrdersResp struct {
	chainHead
	Market string      `json:"market"`
	Count  int         `json:"count"`
	Orders []orderJSON `json:"orders"`
}

// handleGetOrders serves a market's resting orders:
//
//	GET .../dex/dex_get_orders?market=<hex>
//
// The orders are read from the in-RAM book (a deterministic fold of the committed
// order:* rows), so a follower and the proposer at the same height return the same
// resting set. market is required (resting orders are per-market).
func (vm *VM) handleGetOrders(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeReadErr(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	pid, err := parsePoolID(r.URL.Query().Get("market"))
	if err != nil {
		writeReadErr(w, http.StatusBadRequest, "bad market: "+err.Error())
		return
	}

	vm.mu.Lock()
	ob := vm.books[pid]
	head := vm.headLocked()
	var orders []orderJSON
	if ob != nil {
		orders = make([]orderJSON, 0, len(ob.Orders))
		for _, o := range ob.Orders {
			if o == nil {
				continue
			}
			orders = append(orders, orderJSON{
				OrderID:   o.ID,
				Side:      sideString(o.Side),
				Price:     o.Price,
				Size:      o.Size,
				Remaining: o.RemainingSize,
				// Hex-encode the 16-byte account: the raw settlement id is not valid
				// UTF-8, so emitting it as a Go string is LOSSY (json replaces the
				// invalid bytes with U+FFFD) — an owner read back from this field
				// could never round-trip. Hex round-trips, so a client can match its
				// own account (the maker's startup reconcile) and it stays the same
				// encoding as dex_get_nonce's account field. One encoding for an
				// account across the read surface.
				User: hex.EncodeToString([]byte(o.User)),
			})
		}
	}
	vm.mu.Unlock()

	sortOrdersJSON(orders)
	writeJSON(w, getOrdersResp{
		chainHead: head,
		Market:    hex.EncodeToString(pid[:]),
		Count:     len(orders),
		Orders:    orders,
	})
}

// marketJSON is one market's display record: its poolID + symbol, whether its
// (base,quote) assets are bound (a custody market), and a one-glance book summary.
type marketJSON struct {
	PoolID      string  `json:"poolId"`
	Symbol      string  `json:"symbol"`
	Base        string  `json:"base,omitempty"`
	Quote       string  `json:"quote,omitempty"`
	AssetsBound bool    `json:"assetsBound"`
	Orders      int     `json:"orders"`
	Remaining   float64 `json:"remaining"`
	BestBid     float64 `json:"bestBid"`
	BestAsk     float64 `json:"bestAsk"`
}

// getMarketsResp is the dex_get_markets body: the chain head plus every market.
type getMarketsResp struct {
	chainHead
	Count   int          `json:"count"`
	Markets []marketJSON `json:"markets"`
}

// handleGetMarkets serves every market on the chain:
//
//	GET .../dex/dex_get_markets
//
// Each record carries the market's durable identity (poolID, symbol, bound
// assets) and an in-RAM book summary (resting orders, best bid/ask). This is what
// the markets-display lists; combined with dex_get_trades it renders a venue.
func (vm *VM) handleGetMarkets(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeReadErr(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	vm.mu.Lock()
	markets, lerr := listMarkets(vm.db)
	if lerr != nil {
		vm.mu.Unlock()
		writeReadErr(w, http.StatusInternalServerError, lerr.Error())
		return
	}
	head := vm.headLocked()
	// Build display records under the lock so the book summary is consistent with
	// the head. assetsBound/base/quote come from durable state; orders/bid/ask from
	// the in-RAM book.
	out := make([]marketJSON, 0, len(markets))
	for _, m := range markets {
		base, quote, bound, aerr := readMarketAssets(vm.db, m.PoolID)
		if aerr != nil {
			vm.mu.Unlock()
			writeReadErr(w, http.StatusInternalServerError, aerr.Error())
			return
		}
		rec := marketJSON{
			PoolID:      hex.EncodeToString(m.PoolID[:]),
			Symbol:      m.Symbol,
			AssetsBound: bound,
		}
		if bound {
			rec.Base = hex.EncodeToString(base[:])
			rec.Quote = hex.EncodeToString(quote[:])
		}
		if ob := vm.books[m.PoolID]; ob != nil {
			for _, o := range ob.Orders {
				if o == nil {
					continue
				}
				rec.Orders++
				rec.Remaining += o.RemainingSize
			}
			if d := ob.GetDepth(1); d != nil {
				if len(d.Bids) > 0 {
					rec.BestBid = d.Bids[0].Price
				}
				if len(d.Asks) > 0 {
					rec.BestAsk = d.Asks[0].Price
				}
			}
		}
		out = append(out, rec)
	}
	vm.mu.Unlock()

	writeJSON(w, getMarketsResp{chainHead: head, Count: len(out), Markets: out})
}

// getNonceResp is the dex_get_nonce body: the chain head plus an account's
// EXPECTED-NEXT nonce. A signed client reads this at startup to resync its nonce
// stream to the venue (every money-moving tx must carry exactly this nonce), so it
// can restart/redeploy without replaying — the orthogonal counterpart to the
// monotonic gate in authorizeTx.
type getNonceResp struct {
	chainHead
	Account string `json:"account"`
	Nonce   uint64 `json:"nonce"`
}

// handleGetNonce serves an account's expected-next nonce:
//
//	GET .../dex/dex_get_nonce?account=<hex of the 16-byte settlement account>
//
// Pure read of the durable nonce: row (state.go getNonce); absent == 0 (the
// account's first op). The account is the same 16-byte id the body's user[16]
// field carries (dchain Account16).
func (vm *VM) handleGetNonce(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeReadErr(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	b, err := hex.DecodeString(strings.TrimPrefix(r.URL.Query().Get("account"), "0x"))
	if err != nil || len(b) != zapwire.UserSize {
		writeReadErr(w, http.StatusBadRequest, fmt.Sprintf("account must be %d-byte hex", zapwire.UserSize))
		return
	}
	var account userKey
	copy(account[:], b)

	vm.mu.Lock()
	n, nerr := getNonce(vm.db, account)
	head := vm.headLocked()
	vm.mu.Unlock()
	if nerr != nil {
		writeReadErr(w, http.StatusInternalServerError, nerr.Error())
		return
	}
	writeJSON(w, getNonceResp{chainHead: head, Account: hex.EncodeToString(account[:]), Nonce: n})
}

// levelJSON is one aggregated price level in the book.
type levelJSON struct {
	Price float64 `json:"price"`
	Size  float64 `json:"size"`
	Count int     `json:"count"`
}

// getBookResp is the dex_get_book body: the chain head plus the market's
// aggregated bid/ask ladder and totals.
type getBookResp struct {
	chainHead
	Market    string      `json:"market"`
	Orders    int         `json:"orders"`
	Remaining float64     `json:"remaining"`
	BestBid   float64     `json:"bestBid"`
	BestAsk   float64     `json:"bestAsk"`
	Bids      []levelJSON `json:"bids"`
	Asks      []levelJSON `json:"asks"`
}

// handleGetBook serves a market's aggregated order book:
//
//	GET .../dex/dex_get_book?market=<hex>&depth=<n>
//
// depth (optional): number of price levels per side; default 20. Levels and
// totals are read from the in-RAM book (the committed-row fold).
func (vm *VM) handleGetBook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeReadErr(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	q := r.URL.Query()
	pid, err := parsePoolID(q.Get("market"))
	if err != nil {
		writeReadErr(w, http.StatusBadRequest, "bad market: "+err.Error())
		return
	}
	depth, err := parseIntParam(q.Get("depth"), 20)
	if err != nil {
		writeReadErr(w, http.StatusBadRequest, "bad depth: "+err.Error())
		return
	}
	if depth <= 0 {
		depth = 20
	}

	vm.mu.Lock()
	ob := vm.books[pid]
	head := vm.headLocked()
	resp := getBookResp{
		chainHead: head,
		Market:    hex.EncodeToString(pid[:]),
		Bids:      []levelJSON{},
		Asks:      []levelJSON{},
	}
	if ob != nil {
		for _, o := range ob.Orders {
			if o == nil {
				continue
			}
			resp.Orders++
			resp.Remaining += o.RemainingSize
		}
		if d := ob.GetDepth(depth); d != nil {
			resp.Bids = levelsJSON(d.Bids)
			resp.Asks = levelsJSON(d.Asks)
			if len(d.Bids) > 0 {
				resp.BestBid = d.Bids[0].Price
			}
			if len(d.Asks) > 0 {
				resp.BestAsk = d.Asks[0].Price
			}
		}
	}
	vm.mu.Unlock()

	writeJSON(w, resp)
}

// ---- small, pure helpers (no consensus, no state mutation) ----

// toTradeJSON projects a committed trade row to its display/verification form,
// decoding the deterministic fixed-point price/qty to floats and the side byte to
// a string. The on-chain row keeps the integer truth.
func toTradeJSON(ct committedTrade) tradeJSON {
	t := ct.Trade
	side := "buy"
	if t.TakerSide == dex.DEXSideAsk {
		side = "sell"
	}
	return tradeJSON{
		Height:       ct.Height,
		Seq:          ct.Seq,
		TradeID:      t.TradeID,
		Price:        t.Price.Float(),
		Size:         dex.FixedQtyToFloat(t.Quantity),
		TakerSide:    side,
		MakerOrderID: t.MakerOrderID,
		TakerOrderID: t.TakerOrderID,
		Timestamp:    int64(t.Timestamp),
	}
}

// sortOrdersJSON imposes a stable display order on resting orders: bids by price
// descending, asks by price ascending, ties broken by orderID. The in-RAM book's
// Orders map has no inherent order, so without this two nodes could return the
// same set in different array positions — sorting makes a cross-node JSON diff
// position-stable as well as set-stable.
func sortOrdersJSON(orders []orderJSON) {
	sort.Slice(orders, func(i, j int) bool {
		a, b := orders[i], orders[j]
		if a.Side != b.Side {
			return a.Side == "buy" // buys first
		}
		if a.Price != b.Price {
			if a.Side == "buy" {
				return a.Price > b.Price
			}
			return a.Price < b.Price
		}
		return a.OrderID < b.OrderID
	})
}

// sideString maps a book Side to "buy"/"sell".
func sideString(s dex.Side) string {
	if s == dex.Sell {
		return "sell"
	}
	return "buy"
}

// levelsJSON projects aggregated price levels to their display form.
func levelsJSON(levels []dex.PriceLevel) []levelJSON {
	out := make([]levelJSON, 0, len(levels))
	for _, l := range levels {
		size := l.TotalSize
		if size == 0 {
			size = l.Size
		}
		count := l.OrderCount
		if count == 0 {
			count = l.Count
		}
		out = append(out, levelJSON{Price: l.Price, Size: size, Count: count})
	}
	return out
}

// parsePoolID decodes a 32-byte poolID from hex (with or without 0x). It errors
// on a wrong length so a malformed market id can never alias a different market's
// keyspace.
func parsePoolID(s string) ([32]byte, error) {
	var pid [32]byte
	if s == "" {
		return pid, fmt.Errorf("market required")
	}
	if len(s) >= 2 && (s[0:2] == "0x" || s[0:2] == "0X") {
		s = s[2:]
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return pid, err
	}
	if len(b) != 32 {
		return pid, fmt.Errorf("poolID must be 32 bytes, got %d", len(b))
	}
	copy(pid[:], b)
	return pid, nil
}

// parseUintParam parses an optional uint64 query param, returning def when empty.
func parseUintParam(s string, def uint64) (uint64, error) {
	if s == "" {
		return def, nil
	}
	return strconv.ParseUint(s, 10, 64)
}

// parseIntParam parses an optional int query param, returning def when empty.
func parseIntParam(s string, def int) (int, error) {
	if s == "" {
		return def, nil
	}
	return strconv.Atoi(s)
}

// writeJSON writes v as a 200 JSON response. A marshal failure (should be
// impossible for these fixed-shape structs) is surfaced as a 500.
func writeJSON(w http.ResponseWriter, v interface{}) {
	body, err := json.Marshal(v)
	if err != nil {
		writeReadErr(w, http.StatusInternalServerError, "marshal: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// writeReadErr writes a JSON error body with the given status.
func writeReadErr(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(`{"error":` + strconv.Quote(msg) + `}`))
}
