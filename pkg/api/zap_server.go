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

// ZAPServer provides ultra-low-latency order handling for HFT
type ZAPServer struct {
	orderBook *lx.OrderBook
	server    rpc.Server
	logger    log.Logger
	addr      string

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
		bufferPool: sync.Pool{
			New: func() interface{} {
				// Pre-allocate 4KB buffer for responses
				return make([]byte, 4096)
			},
		},
	}
}

// Start starts the ZAP server
func (s *ZAPServer) Start(ctx context.Context) error {
	server, err := rpc.Listen(s.addr)
	if err != nil {
		return fmt.Errorf("failed to start ZAP server: %w", err)
	}
	s.server = server

	// Register raw handlers for zero-copy processing
	s.server.RegisterRaw("place_order", s.handlePlaceOrder)
	s.server.RegisterRaw("cancel_order", s.handleCancelOrder)
	s.server.RegisterRaw("modify_order", s.handleModifyOrder)
	s.server.RegisterRaw("best_bid", s.handleGetBestBid)
	s.server.RegisterRaw("best_ask", s.handleGetBestAsk)
	s.server.RegisterRaw("orderbook", s.handleGetOrderBook)
	s.server.RegisterRaw("order", s.handleGetOrder)

	s.logger.Info("ZAP server started", "addr", s.server.Addr())

	go s.server.Serve(ctx)
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
