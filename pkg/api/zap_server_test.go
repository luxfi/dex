// Copyright (C) 2019-2025, Lux Industries, Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package api

import (
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/log"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/rpc"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// nopLogger is a no-op logger for testing
type nopLogger struct{}

func (nopLogger) Info(msg string, args ...interface{})               {}
func (nopLogger) Error(msg string, args ...interface{})              {}
func (nopLogger) Warn(msg string, args ...interface{})               {}
func (nopLogger) Debug(msg string, args ...interface{})              {}
func (nopLogger) Fatal(msg string, args ...interface{})              {}
func (nopLogger) WithField(key string, value interface{}) log.Logger { return nopLogger{} }

func TestZAPServerBasic(t *testing.T) {
	// Create orderbook
	ob := lx.NewOrderBook("BTC-USD")

	// Create and start ZAP server
	logger := nopLogger{}
	server := NewZAPServer(ob, ":0", logger)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := server.Start(ctx)
	require.NoError(t, err)
	defer server.Stop()

	// Give server time to start
	time.Sleep(10 * time.Millisecond)

	// Connect client
	client, err := rpc.Dial(ctx, server.Addr())
	require.NoError(t, err)
	defer client.Close()

	// Test place order
	orderPayload := encodeTestOrder(1, 50000.0, 1.0, lx.Buy, lx.Limit, "user1")
	resp, err := client.CallRaw(ctx, "place_order", orderPayload)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Verify ack
	assert.GreaterOrEqual(t, len(resp), 8)
	orderID := binary.BigEndian.Uint64(resp[0:8])
	assert.Greater(t, orderID, uint64(0))

	// Check stats
	orders, trades, cancels := server.Stats()
	assert.Equal(t, uint64(1), orders)
	assert.Equal(t, uint64(0), trades)
	assert.Equal(t, uint64(0), cancels)
}

func TestZAPServerOrderBook(t *testing.T) {
	ob := lx.NewOrderBook("BTC-USD")
	logger := nopLogger{}
	server := NewZAPServer(ob, ":0", logger)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := server.Start(ctx)
	require.NoError(t, err)
	defer server.Stop()

	time.Sleep(10 * time.Millisecond)

	client, err := rpc.Dial(ctx, server.Addr())
	require.NoError(t, err)
	defer client.Close()

	// Place some orders
	for i := 0; i < 5; i++ {
		price := 50000.0 - float64(i)*100
		orderPayload := encodeTestOrder(uint64(i+1), price, 1.0, lx.Buy, lx.Limit, "user1")
		_, err := client.CallRaw(ctx, "place_order", orderPayload)
		require.NoError(t, err)
	}

	for i := 0; i < 5; i++ {
		price := 50100.0 + float64(i)*100
		orderPayload := encodeTestOrder(uint64(i+100), price, 1.0, lx.Sell, lx.Limit, "user2")
		_, err := client.CallRaw(ctx, "place_order", orderPayload)
		require.NoError(t, err)
	}

	// Get best bid
	resp, err := client.CallRaw(ctx, "best_bid", nil)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(resp), QuoteWireSize)

	price := decodeFloat64(resp[0:8])
	assert.Equal(t, 50000.0, price)

	// Get best ask
	resp, err = client.CallRaw(ctx, "best_ask", nil)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(resp), QuoteWireSize)

	price = decodeFloat64(resp[0:8])
	assert.Equal(t, 50100.0, price)

	// Get orderbook
	depthReq := make([]byte, 4)
	binary.BigEndian.PutUint32(depthReq, 10)
	resp, err = client.CallRaw(ctx, "orderbook", depthReq)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(resp), 8)

	bidCount := binary.BigEndian.Uint32(resp[0:4])
	askCount := binary.BigEndian.Uint32(resp[4:8])
	assert.Equal(t, uint32(5), bidCount)
	assert.Equal(t, uint32(5), askCount)
}

func TestZAPServerCancelOrder(t *testing.T) {
	ob := lx.NewOrderBook("BTC-USD")
	logger := nopLogger{}
	server := NewZAPServer(ob, ":0", logger)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := server.Start(ctx)
	require.NoError(t, err)
	defer server.Stop()

	time.Sleep(10 * time.Millisecond)

	client, err := rpc.Dial(ctx, server.Addr())
	require.NoError(t, err)
	defer client.Close()

	// Place order
	orderPayload := encodeTestOrder(1, 50000.0, 1.0, lx.Buy, lx.Limit, "user1")
	resp, err := client.CallRaw(ctx, "place_order", orderPayload)
	require.NoError(t, err)

	orderID := binary.BigEndian.Uint64(resp[0:8])

	// Cancel order
	cancelPayload := make([]byte, CancelWireSize)
	binary.BigEndian.PutUint64(cancelPayload[0:8], orderID)
	copy(cancelPayload[8:24], padNull([]byte("user1"), 16))

	resp, err = client.CallRaw(ctx, "cancel_order", cancelPayload)
	require.NoError(t, err)

	// Verify ack with cancel status
	assert.GreaterOrEqual(t, len(resp), 9)
	respOrderID := binary.BigEndian.Uint64(resp[0:8])
	assert.Equal(t, orderID, respOrderID)
	assert.Equal(t, uint8(1), resp[8]) // Cancel status

	// Check stats
	_, _, cancels := server.Stats()
	assert.Equal(t, uint64(1), cancels)
}

func TestZAPServerGetOrder(t *testing.T) {
	ob := lx.NewOrderBook("BTC-USD")
	logger := nopLogger{}
	server := NewZAPServer(ob, ":0", logger)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := server.Start(ctx)
	require.NoError(t, err)
	defer server.Stop()

	time.Sleep(10 * time.Millisecond)

	client, err := rpc.Dial(ctx, server.Addr())
	require.NoError(t, err)
	defer client.Close()

	// Place order
	orderPayload := encodeTestOrder(1, 50000.0, 2.5, lx.Buy, lx.Limit, "user1")
	resp, err := client.CallRaw(ctx, "place_order", orderPayload)
	require.NoError(t, err)

	orderID := binary.BigEndian.Uint64(resp[0:8])

	// Get order
	getReq := make([]byte, 8)
	binary.BigEndian.PutUint64(getReq, orderID)

	resp, err = client.CallRaw(ctx, "order", getReq)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(resp), OrderWireSize)

	// Decode order response
	respOrderID := binary.BigEndian.Uint64(resp[8:16])
	respPrice := decodeFloat64(resp[16:24])
	respSize := decodeFloat64(resp[24:32])
	respSide := lx.Side(resp[32])

	assert.Equal(t, orderID, respOrderID)
	assert.Equal(t, 50000.0, respPrice)
	assert.Equal(t, 2.5, respSize)
	assert.Equal(t, lx.Buy, respSide)
}

func BenchmarkZAPOrderPlacement(b *testing.B) {
	ob := lx.NewOrderBook("BTC-USD")
	logger := nopLogger{}
	server := NewZAPServer(ob, ":0", logger)

	ctx := context.Background()
	err := server.Start(ctx)
	if err != nil {
		b.Fatal(err)
	}
	defer server.Stop()

	time.Sleep(10 * time.Millisecond)

	client, err := rpc.Dial(ctx, server.Addr())
	if err != nil {
		b.Fatal(err)
	}
	defer client.Close()

	orderPayload := encodeTestOrder(0, 50000.0, 1.0, lx.Buy, lx.Limit, "user1")

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Update order ID
		binary.BigEndian.PutUint64(orderPayload[8:16], uint64(i+1))

		_, err := client.CallRaw(ctx, "place_order", orderPayload)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZAPBestBid(b *testing.B) {
	ob := lx.NewOrderBook("BTC-USD")

	// Add some orders
	for i := 0; i < 100; i++ {
		order := &lx.Order{
			ID:     uint64(i + 1),
			Price:  50000.0 - float64(i),
			Size:   1.0,
			Side:   lx.Buy,
			Type:   lx.Limit,
			UserID: "user1",
		}
		ob.AddOrder(order)
	}

	logger := nopLogger{}
	server := NewZAPServer(ob, ":0", logger)

	ctx := context.Background()
	err := server.Start(ctx)
	if err != nil {
		b.Fatal(err)
	}
	defer server.Stop()

	time.Sleep(10 * time.Millisecond)

	client, err := rpc.Dial(ctx, server.Addr())
	if err != nil {
		b.Fatal(err)
	}
	defer client.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := client.CallRaw(ctx, "best_bid", nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Helper to encode a test order
func encodeTestOrder(id uint64, price, size float64, side lx.Side, orderType lx.OrderType, userID string) []byte {
	buf := make([]byte, OrderWireSize)

	// Symbol (8 bytes)
	copy(buf[0:8], padNull([]byte("BTC-USD"), 8))

	// Order ID
	binary.BigEndian.PutUint64(buf[8:16], id)

	// Price
	encodeFloat64(buf[16:24], price)

	// Size
	encodeFloat64(buf[24:32], size)

	// Side
	buf[32] = byte(side)

	// Type
	buf[33] = byte(orderType)

	// Flags (none)
	binary.BigEndian.PutUint16(buf[34:36], 0)

	// Timestamp
	binary.BigEndian.PutUint64(buf[36:44], uint64(time.Now().UnixNano()))

	// User ID
	copy(buf[44:60], padNull([]byte(userID), 16))

	return buf
}
