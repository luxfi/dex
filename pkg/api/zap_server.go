// Copyright (C) 2019-2025, Lux Industries, Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package api

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/luxfi/dex/pkg/log"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// ZAP wire message types for DEX (zero-copy binary protocol)
const (
	MsgPlaceOrder uint8 = iota + 1
	MsgCancelOrder
	MsgModifyOrder
	MsgGetBestBid
	MsgGetBestAsk
	MsgGetOrderBook
	MsgGetOrder
	MsgOrderAck
	MsgOrderReject
	MsgTrade
	MsgMarketData
)

// Wire sizes for fixed-format messages (cache-aligned)
const (
	OrderWireSize  = 64 // Order: symbol(8) + id(8) + price(8) + size(8) + side(1) + type(1) + flags(2) + ts(8) + user(16) + pad(4)
	CancelWireSize = 32 // Cancel: order_id(8) + user(16) + pad(8)
	AckWireSize    = 24 // Ack: order_id(8) + status(1) + seq(8) + pad(7)
	QuoteWireSize  = 24 // Quote: price(8) + size(8) + count(4) + pad(4)
	TradeWireSize  = 48 // Trade: id(8) + price(8) + size(8) + buyer(8) + seller(8) + ts(8)
)

// DEX market-routed method names, frame sizes, status bytes, and FillWireSize
// are the FROZEN wire frame defined ONCE in github.com/luxfi/dex/pkg/zapwire —
// a pure-Go leaf so the public EVM precompile can import them without dragging
// this cgo/GPU engine. They are re-exported here as package-local aliases so
// the handlers below read naturally; the bytes are byte-identical to what the
// precompile (engine_zap.go) and the chains/dexvm proxy relay client encode.
//
// The V4 PoolManager facade is a central-limit-order-book, NOT an AMM, so
// "initialize" ensures a market, "modifyLiquidity" places/cancels a resting
// limit order, and "swap" submits a marketable order and gets its fills back.
// Market identity on the wire is the 32-byte V4 poolId (keccak256 of the
// PoolKey). There is exactly one matcher (*lx.OrderBook); two wire framings for
// two callers (maker = 8-byte symbol, precompile/proxy = 32-byte poolId).
const (
	DEXMethodEnsureMarket = zapwire.MethodEnsureMarket
	DEXMethodPlace        = zapwire.MethodPlace
	DEXMethodCancel       = zapwire.MethodCancel
	DEXMethodSubmit       = zapwire.MethodSubmit
)

// DEX ack status bytes (shared with the legacy ack/reject codec).
const (
	dexStatusPlaced   = zapwire.StatusPlaced
	dexStatusCanceled = zapwire.StatusCanceled
	dexStatusRejected = zapwire.StatusRejected
)

// FillWireSize is one fill in a dex_submit response. See zapwire.
const FillWireSize = zapwire.FillWireSize

// ZAPServer provides ultra-low-latency order handling for HFT.
//
// It is multi-market: legacy symbol-keyed methods (place_order/…) operate on a
// default book (or a per-symbol book when a symbol is supplied), and the
// poolId-keyed DEX methods (dex_*) operate on per-market books keyed by the
// 32-byte V4 poolId. Every method routes to the same *lx.OrderBook matcher.
type ZAPServer struct {
	// orderBook is the default (single) book the legacy symbol-keyed handlers
	// use, preserving the original single-book behavior for existing callers.
	orderBook *lx.OrderBook
	server    rpc.Server
	logger    log.Logger
	addr      string

	// markets holds per-poolId books for the DEX (dex_*) surface. Keyed by the
	// raw 32-byte poolId.
	//
	// This is the GATEWAY to the d-chain: the in-memory book here is the
	// deterministic REFERENCE matcher that the d-chain consensus executes; it is
	// NOT a standalone authority. The authoritative resting-order/market state
	// is the book persisted under the d-chain's consensus, built from the
	// private C++/GPU engine. A standalone dex-server/lxd booting an ephemeral
	// ZAPServer is a dev/test gateway only, non-authoritative. The precompile
	// adapter and the chains/dexvm proxy hold NONE of this state — they relay to
	// it over the frozen zapwire frame.
	marketsMu sync.Mutex
	markets   map[[32]byte]*lx.OrderBook

	// Statistics
	ordersProcessed atomic.Uint64
	tradesExecuted  atomic.Uint64
	cancelProcessed atomic.Uint64

	// Pre-allocated response buffers (per-connection pools)
	bufferPool sync.Pool

	// Sequence counter for order acks
	sequence atomic.Uint64
}

// NewZAPServer creates a new ZAP server for ultra-low-latency trading
func NewZAPServer(orderBook *lx.OrderBook, addr string, logger log.Logger) *ZAPServer {
	return &ZAPServer{
		orderBook: orderBook,
		addr:      addr,
		logger:    logger,
		markets:   make(map[[32]byte]*lx.OrderBook),
		bufferPool: sync.Pool{
			New: func() interface{} {
				// Pre-allocate 4KB buffer for responses
				return make([]byte, 4096)
			},
		},
	}
}

// market returns the per-poolId book, creating it on first use. The poolId is
// rendered hex for the book symbol so logs are human-readable. createIfAbsent
// false returns nil when the market was never ensured.
func (s *ZAPServer) market(id [32]byte, createIfAbsent bool) *lx.OrderBook {
	s.marketsMu.Lock()
	defer s.marketsMu.Unlock()
	if ob, ok := s.markets[id]; ok {
		return ob
	}
	if !createIfAbsent {
		return nil
	}
	ob := lx.NewOrderBook(fmt.Sprintf("%x", id[:6]))
	s.markets[id] = ob
	return ob
}

// MarketCount returns the number of canonical DEX markets held server-side.
// Diagnostic: proves resting-order state lives here, not in the adapter.
func (s *ZAPServer) MarketCount() int {
	s.marketsMu.Lock()
	defer s.marketsMu.Unlock()
	return len(s.markets)
}

// BestBidAsk returns the canonical best bid/ask for a poolId market. ok is false
// when the market does not exist. Diagnostic for tests.
func (s *ZAPServer) BestBidAsk(id [32]byte) (bid, ask float64, ok bool) {
	ob := s.market(id, false)
	if ob == nil {
		return 0, 0, false
	}
	return ob.GetBestBid(), ob.GetBestAsk(), true
}

// Start starts the ZAP server
func (s *ZAPServer) Start(ctx context.Context) error {
	server, err := rpc.Listen(s.addr)
	if err != nil {
		return fmt.Errorf("failed to start ZAP server: %w", err)
	}
	s.server = server

	if err := s.register(server); err != nil {
		return err
	}

	s.logger.Info("ZAP server started", "addr", s.server.Addr())

	go s.server.Serve(ctx)
	return nil
}

// register wires every handler onto an rpc.Server. Split out so tests can drive
// the same handler set against a server they own (mirrors maker/seeder_test.go).
func (s *ZAPServer) register(server rpc.Server) error {
	raw := []struct {
		method  string
		handler rpc.RawHandler
	}{
		// Legacy symbol-keyed surface (single default book).
		{"place_order", s.handlePlaceOrder},
		{"cancel_order", s.handleCancelOrder},
		{"modify_order", s.handleModifyOrder},
		{"best_bid", s.handleGetBestBid},
		{"best_ask", s.handleGetBestAsk},
		{"orderbook", s.handleGetOrderBook},
		{"order", s.handleGetOrder},
		// DEX poolId-keyed surface (per-market books) — the V4 precompile path.
		{DEXMethodEnsureMarket, s.handleEnsureMarket},
		{DEXMethodPlace, s.handleDEXPlace},
		{DEXMethodCancel, s.handleDEXCancel},
		{DEXMethodSubmit, s.handleDEXSubmit},
	}
	for _, r := range raw {
		if err := server.RegisterRaw(r.method, r.handler); err != nil {
			return fmt.Errorf("register %s: %w", r.method, err)
		}
	}
	return nil
}

// RegisterDEX registers ONLY the poolId-keyed DEX handlers on an rpc.Server.
// It is the integration seam the LP-9010 precompile tests use to stand up the
// real matching engine without the legacy single-book surface. Additive: the
// caller may register other handlers on the same server.
func RegisterDEX(server rpc.Server, s *ZAPServer) error {
	regs := []struct {
		method  string
		handler rpc.RawHandler
	}{
		{DEXMethodEnsureMarket, s.handleEnsureMarket},
		{DEXMethodPlace, s.handleDEXPlace},
		{DEXMethodCancel, s.handleDEXCancel},
		{DEXMethodSubmit, s.handleDEXSubmit},
	}
	for _, r := range regs {
		if err := server.RegisterRaw(r.method, r.handler); err != nil {
			return fmt.Errorf("register %s: %w", r.method, err)
		}
	}
	return nil
}

// Stop stops the ZAP server
func (s *ZAPServer) Stop() error {
	if s.server != nil {
		return s.server.Close()
	}
	return nil
}

// Addr returns the server address
func (s *ZAPServer) Addr() string {
	if s.server != nil {
		return s.server.Addr()
	}
	return s.addr
}

// Stats returns server statistics
func (s *ZAPServer) Stats() (orders, trades, cancels uint64) {
	return s.ordersProcessed.Load(), s.tradesExecuted.Load(), s.cancelProcessed.Load()
}

// =========================================================================
// DEX (poolId-keyed) handlers — the V4 PoolManager facade.
// =========================================================================

// handleEnsureMarket: payload = poolId[32]. Idempotently creates the market
// (order book) for the V4 poolId. Returns ack(0, placed).
func (s *ZAPServer) handleEnsureMarket(_ context.Context, payload []byte) ([]byte, error) {
	if len(payload) < 32 {
		return s.encodeReject(0, "ensure_market: short payload"), nil
	}
	var id [32]byte
	copy(id[:], payload[0:32])
	s.market(id, true) // create if absent
	return s.encodeAck(0, dexStatusPlaced, s.sequence.Add(1)), nil
}

// handleDEXPlace places a RESTING limit order on a poolId market — the DEX
// meaning of V4 modifyLiquidity(+delta). Payload (65 bytes):
//
//	[0:32]  poolId
//	[32]    side (0=buy/bid, 1=sell/ask)
//	[33:41] price (float64, IEEE 754)
//	[41:49] size  (float64, IEEE 754)
//	[49:65] user (16 bytes, null-padded)
//
// Returns ack(orderId, placed). A rest order never crosses here: the precompile
// places liquidity, takers cross it via dex_submit.
func (s *ZAPServer) handleDEXPlace(_ context.Context, payload []byte) ([]byte, error) {
	if len(payload) < 65 {
		return s.encodeReject(0, "dex_place: short payload"), nil
	}
	var id [32]byte
	copy(id[:], payload[0:32])
	side := lx.Side(payload[32])
	price := decodeFloat64(payload[33:41])
	size := decodeFloat64(payload[41:49])
	user := string(trimNull(payload[49:65]))

	if price <= 0 || size <= 0 {
		return s.encodeReject(0, "dex_place: invalid price or size"), nil
	}

	ob := s.market(id, true)
	order := &lx.Order{
		Type:   lx.Limit,
		Side:   side,
		Price:  price,
		Size:   size,
		User:   user,
		UserID: user,
		Symbol: ob.Symbol,
	}
	orderID := ob.AddOrder(order)
	if orderID == 0 {
		return s.encodeReject(0, "dex_place: order rejected"), nil
	}
	s.ordersProcessed.Add(1)
	return s.encodeAck(orderID, dexStatusPlaced, s.sequence.Add(1)), nil
}

// handleDEXCancel cancels a resting order on a poolId market — the DEX meaning
// of V4 modifyLiquidity(-delta). Payload (40 bytes): poolId[32] + orderId[8].
func (s *ZAPServer) handleDEXCancel(_ context.Context, payload []byte) ([]byte, error) {
	if len(payload) < 40 {
		return s.encodeReject(0, "dex_cancel: short payload"), nil
	}
	var id [32]byte
	copy(id[:], payload[0:32])
	orderID := binary.BigEndian.Uint64(payload[32:40])

	ob := s.market(id, false)
	if ob == nil {
		return s.encodeReject(orderID, "dex_cancel: unknown market"), nil
	}
	if err := ob.CancelOrder(orderID); err != nil {
		return s.encodeReject(orderID, err.Error()), nil
	}
	s.cancelProcessed.Add(1)
	return s.encodeAck(orderID, dexStatusCanceled, s.sequence.Add(1)), nil
}

// handleDEXSubmit submits a MARKETABLE order against a poolId market — the DEX
// meaning of V4 swap. It crosses the resting book and returns the resulting
// fills; the adapter derives the BalanceDelta from those fills alone. Payload
// (66 bytes):
//
//	[0:32]  poolId
//	[32]    side (0=buy, 1=sell)
//	[33]    isMarket (0 = IOC limit bounded by price, 1 = pure market)
//	[34:42] limitPrice (float64; ignored when isMarket=1)
//	[42:50] size (float64, base units)
//	[50:66] user (16 bytes, null-padded)
//
// Response: fillCount[4] then fillCount × (price[8] + size[8] + takerSide[1]).
func (s *ZAPServer) handleDEXSubmit(_ context.Context, payload []byte) ([]byte, error) {
	if len(payload) < 66 {
		return nil, fmt.Errorf("dex_submit: short payload: %d", len(payload))
	}
	var id [32]byte
	copy(id[:], payload[0:32])
	side := lx.Side(payload[32])
	isMarket := payload[33] == 1
	limitPrice := decodeFloat64(payload[34:42])
	size := decodeFloat64(payload[42:50])
	user := string(trimNull(payload[50:66]))

	if size <= 0 {
		return nil, fmt.Errorf("dex_submit: non-positive size")
	}

	ob := s.market(id, false)
	if ob == nil {
		return nil, fmt.Errorf("dex_submit: unknown market")
	}

	order := &lx.Order{
		Side:   side,
		Size:   size,
		User:   user,
		UserID: user,
		Symbol: ob.Symbol,
	}
	if isMarket {
		order.Type = lx.Market
	} else {
		if limitPrice <= 0 {
			return nil, fmt.Errorf("dex_submit: IOC limit needs positive price")
		}
		order.Type = lx.Limit
		order.Price = limitPrice
	}

	fills, err := ob.SubmitMarketable(order)
	if err != nil {
		return nil, fmt.Errorf("dex_submit: %w", err)
	}
	s.tradesExecuted.Add(uint64(len(fills)))

	resp := make([]byte, 4+len(fills)*FillWireSize)
	binary.BigEndian.PutUint32(resp[0:4], uint32(len(fills)))
	off := 4
	for _, tr := range fills {
		encodeFloat64(resp[off:off+8], tr.Price)
		encodeFloat64(resp[off+8:off+16], tr.Size)
		resp[off+16] = byte(side) // taker side this submit took with
		off += FillWireSize
	}
	return resp, nil
}

// =========================================================================
// Legacy symbol-keyed handlers (single default book) — unchanged behavior.
// =========================================================================

// handlePlaceOrder handles order placement with zero-copy
// Wire format (64 bytes):
//
//	[0:8]   symbol (8 bytes, null-padded)
//	[8:16]  order_id (uint64, network byte order)
//	[16:24] price (float64, IEEE 754)
//	[24:32] size (float64, IEEE 754)
//	[32]    side (uint8: 0=buy, 1=sell)
//	[33]    order_type (uint8: 0=limit, 1=market, 2=stop)
//	[34:36] flags (uint16: post-only, reduce-only, STP)
//	[36:44] timestamp (uint64, unix nanos)
//	[44:60] user_id (16 bytes, null-padded)
//	[60:64] padding
func (s *ZAPServer) handlePlaceOrder(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < OrderWireSize {
		return s.encodeReject(0, "invalid message size"), nil
	}

	// Zero-copy decode order from wire format
	order := s.decodeOrder(payload)
	if order == nil {
		return s.encodeReject(0, "invalid order data"), nil
	}

	// Validate order
	if order.Price <= 0 || order.Size <= 0 {
		return s.encodeReject(order.ID, "invalid price or size"), nil
	}

	// Process order through matching engine
	// AddOrder returns the order ID (0 if rejected)
	orderID := s.orderBook.AddOrder(order)
	if orderID == 0 {
		return s.encodeReject(order.ID, "order rejected"), nil
	}

	s.ordersProcessed.Add(1)

	// Return ack with sequence number
	return s.encodeAck(orderID, 0, s.sequence.Add(1)), nil
}

// handleCancelOrder handles order cancellation
// Wire format (32 bytes):
//
//	[0:8]   order_id (uint64)
//	[8:24]  user_id (16 bytes)
//	[24:32] padding
func (s *ZAPServer) handleCancelOrder(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < CancelWireSize {
		return s.encodeReject(0, "invalid message size"), nil
	}

	orderID := binary.BigEndian.Uint64(payload[0:8])

	// Cancel the order
	err := s.orderBook.CancelOrder(orderID)
	if err != nil {
		return s.encodeReject(orderID, err.Error()), nil
	}

	s.cancelProcessed.Add(1)
	return s.encodeAck(orderID, 1, s.sequence.Add(1)), nil
}

// handleModifyOrder handles order modification
func (s *ZAPServer) handleModifyOrder(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < OrderWireSize {
		return s.encodeReject(0, "invalid message size"), nil
	}

	order := s.decodeOrder(payload)
	if order == nil {
		return s.encodeReject(0, "invalid order data"), nil
	}

	// Modify is cancel + add
	_ = s.orderBook.CancelOrder(order.ID)
	orderID := s.orderBook.AddOrder(order)
	if orderID == 0 {
		return s.encodeReject(order.ID, "order rejected"), nil
	}

	return s.encodeAck(orderID, 0, s.sequence.Add(1)), nil
}

// handleGetBestBid returns the best bid price/size
func (s *ZAPServer) handleGetBestBid(ctx context.Context, payload []byte) ([]byte, error) {
	price := s.orderBook.GetBestBid()
	if price == 0 {
		return s.encodeQuote(0, 0, 0), nil
	}
	// Get depth for size/count info
	depth := s.orderBook.GetDepth(1)
	if depth != nil && len(depth.Bids) > 0 {
		return s.encodeQuote(depth.Bids[0].Price, depth.Bids[0].Size, depth.Bids[0].Count), nil
	}
	return s.encodeQuote(price, 0, 0), nil
}

// handleGetBestAsk returns the best ask price/size
func (s *ZAPServer) handleGetBestAsk(ctx context.Context, payload []byte) ([]byte, error) {
	price := s.orderBook.GetBestAsk()
	if price == 0 {
		return s.encodeQuote(0, 0, 0), nil
	}
	// Get depth for size/count info
	depth := s.orderBook.GetDepth(1)
	if depth != nil && len(depth.Asks) > 0 {
		return s.encodeQuote(depth.Asks[0].Price, depth.Asks[0].Size, depth.Asks[0].Count), nil
	}
	return s.encodeQuote(price, 0, 0), nil
}

// handleGetOrderBook returns top N levels
func (s *ZAPServer) handleGetOrderBook(ctx context.Context, payload []byte) ([]byte, error) {
	levels := 10
	if len(payload) >= 4 {
		levels = int(binary.BigEndian.Uint32(payload[0:4]))
		if levels > 100 {
			levels = 100
		}
	}

	// Get orderbook depth
	depth := s.orderBook.GetDepth(levels)
	if depth == nil {
		return s.encodeQuote(0, 0, 0), nil
	}

	// Encode response: [4 bytes: bid count][4 bytes: ask count][bids...][asks...]
	bidCount := len(depth.Bids)
	askCount := len(depth.Asks)
	size := 8 + bidCount*QuoteWireSize + askCount*QuoteWireSize
	resp := s.getBuffer()[:size]

	binary.BigEndian.PutUint32(resp[0:4], uint32(bidCount))
	binary.BigEndian.PutUint32(resp[4:8], uint32(askCount))

	offset := 8

	// Encode bids
	for _, level := range depth.Bids {
		s.encodeQuoteAt(resp[offset:], level.Price, level.Size, level.Count)
		offset += QuoteWireSize
	}

	// Encode asks
	for _, level := range depth.Asks {
		s.encodeQuoteAt(resp[offset:], level.Price, level.Size, level.Count)
		offset += QuoteWireSize
	}

	return resp[:offset], nil
}

// handleGetOrder returns a specific order
func (s *ZAPServer) handleGetOrder(ctx context.Context, payload []byte) ([]byte, error) {
	if len(payload) < 8 {
		return nil, fmt.Errorf("invalid order id")
	}

	orderID := binary.BigEndian.Uint64(payload[0:8])
	order := s.orderBook.GetOrder(orderID)
	if order == nil {
		return s.encodeReject(orderID, "order not found"), nil
	}

	return s.encodeOrderResponse(order), nil
}

// decodeOrder decodes order from wire format (zero-copy)
func (s *ZAPServer) decodeOrder(data []byte) *lx.Order {
	if len(data) < OrderWireSize {
		return nil
	}

	// Extract symbol (8 bytes, trim null padding)
	symbol := trimNull(data[0:8])

	order := &lx.Order{
		ID:        binary.BigEndian.Uint64(data[8:16]),
		Price:     decodeFloat64(data[16:24]),
		Size:      decodeFloat64(data[24:32]),
		Side:      lx.Side(data[32]),
		Type:      lx.OrderType(data[33]),
		Flags:     lx.OrderFlags(binary.BigEndian.Uint16(data[34:36])),
		Timestamp: time.Unix(0, int64(binary.BigEndian.Uint64(data[36:44]))),
		UserID:    string(trimNull(data[44:60])),
		Symbol:    string(symbol),
	}

	// Validate basic fields
	if order.Price < 0 || order.Size < 0 {
		return nil
	}

	return order
}

// encodeAck encodes an order acknowledgment
func (s *ZAPServer) encodeAck(orderID uint64, status uint8, seq uint64) []byte {
	resp := s.getBuffer()[:AckWireSize]
	binary.BigEndian.PutUint64(resp[0:8], orderID)
	resp[8] = status
	binary.BigEndian.PutUint64(resp[9:17], seq)
	return resp[:AckWireSize]
}

// encodeReject encodes an order rejection
func (s *ZAPServer) encodeReject(orderID uint64, reason string) []byte {
	// Rejection: order_id(8) + status(1=rejected) + reason_len(2) + reason(variable)
	reasonBytes := []byte(reason)
	if len(reasonBytes) > 256 {
		reasonBytes = reasonBytes[:256]
	}

	size := 11 + len(reasonBytes)
	resp := s.getBuffer()[:size]
	binary.BigEndian.PutUint64(resp[0:8], orderID)
	resp[8] = 2 // Rejected status
	binary.BigEndian.PutUint16(resp[9:11], uint16(len(reasonBytes)))
	copy(resp[11:], reasonBytes)
	return resp[:size]
}

// encodeQuote encodes a price level quote
func (s *ZAPServer) encodeQuote(price, size float64, count int) []byte {
	resp := s.getBuffer()[:QuoteWireSize]
	s.encodeQuoteAt(resp, price, size, count)
	return resp[:QuoteWireSize]
}

// encodeQuoteAt encodes a quote at a specific buffer position
func (s *ZAPServer) encodeQuoteAt(buf []byte, price, size float64, count int) {
	encodeFloat64(buf[0:8], price)
	encodeFloat64(buf[8:16], size)
	binary.BigEndian.PutUint32(buf[16:20], uint32(count))
}

// encodeOrderResponse encodes a full order response
func (s *ZAPServer) encodeOrderResponse(order *lx.Order) []byte {
	resp := s.getBuffer()[:OrderWireSize]

	// Symbol (8 bytes, null-padded)
	copy(resp[0:8], padNull([]byte(order.Symbol), 8))
	binary.BigEndian.PutUint64(resp[8:16], order.ID)
	encodeFloat64(resp[16:24], order.Price)
	encodeFloat64(resp[24:32], order.Size)
	resp[32] = byte(order.Side)
	resp[33] = byte(order.Type)
	binary.BigEndian.PutUint16(resp[34:36], uint16(order.Flags))
	binary.BigEndian.PutUint64(resp[36:44], uint64(order.Timestamp.UnixNano()))
	copy(resp[44:60], padNull([]byte(order.UserID), 16))

	return resp[:OrderWireSize]
}

// getBuffer gets a buffer from the pool
func (s *ZAPServer) getBuffer() []byte {
	return s.bufferPool.Get().([]byte)
}

// Helper functions for binary encoding

func decodeFloat64(b []byte) float64 {
	bits := binary.BigEndian.Uint64(b)
	return *(*float64)(unsafe.Pointer(&bits))
}

func encodeFloat64(b []byte, f float64) {
	bits := *(*uint64)(unsafe.Pointer(&f))
	binary.BigEndian.PutUint64(b, bits)
}

func trimNull(b []byte) []byte {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] != 0 {
			return b[:i+1]
		}
	}
	return b[:0]
}

func padNull(b []byte, size int) []byte {
	if len(b) >= size {
		return b[:size]
	}
	result := make([]byte, size)
	copy(result, b)
	return result
}
