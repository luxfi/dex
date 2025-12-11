package client

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/lx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewTraderClient(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.lux.exchange",
		WSEndpoint:  "wss://ws.lux.exchange",
		APIKey:      "test_key",
		APISecret:   "test_secret",
		UserID:      "user123",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)
	assert.NotNil(t, client)
	assert.Equal(t, "https://api.lux.exchange", client.apiEndpoint)
	assert.Equal(t, "wss://ws.lux.exchange", client.wsEndpoint)
	assert.Equal(t, "test_key", client.apiKey)
	assert.Equal(t, "test_secret", client.apiSecret)
	assert.Equal(t, "user123", client.userID)
	assert.False(t, client.connected)
	assert.False(t, client.authenticated)
}

func TestNewTraderClientMissingEndpoints(t *testing.T) {
	tests := []struct {
		name   string
		config ClientConfig
	}{
		{
			name:   "MissingAPIEndpoint",
			config: ClientConfig{WSEndpoint: "wss://ws.test.com"},
		},
		{
			name:   "MissingWSEndpoint",
			config: ClientConfig{APIEndpoint: "https://api.test.com"},
		},
		{
			name:   "BothMissing",
			config: ClientConfig{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewTraderClient(tt.config)
			assert.Error(t, err)
			assert.Nil(t, client)
			assert.Contains(t, err.Error(), "endpoints required")
		})
	}
}

func TestTraderClientInitialState(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Check maps are initialized
	assert.NotNil(t, client.positions)
	assert.NotNil(t, client.orders)
	assert.NotNil(t, client.balances)
	assert.NotNil(t, client.orderBooks)
	assert.NotNil(t, client.prices)
	assert.NotNil(t, client.trades)
	assert.NotNil(t, client.subscriptions)
	assert.NotNil(t, client.callbacks)

	// Check channels are initialized
	assert.NotNil(t, client.orderUpdates)
	assert.NotNil(t, client.positionUpdates)
	assert.NotNil(t, client.priceUpdates)
	assert.NotNil(t, client.tradeUpdates)
	assert.NotNil(t, client.errorChan)

	// Check context
	assert.NotNil(t, client.ctx)
	assert.NotNil(t, client.cancel)
}

func TestOrderUpdateStruct(t *testing.T) {
	update := OrderUpdate{
		Order:     nil, // Would be *lx.Order in real use
		Status:    0,   // Would be lx.OrderStatus
		Timestamp: time.Now(),
		Message:   "Order filled",
	}

	assert.Equal(t, "Order filled", update.Message)
	assert.False(t, update.Timestamp.IsZero())
}

func TestPositionUpdateStruct(t *testing.T) {
	update := PositionUpdate{
		Position:  nil,
		Action:    "opened",
		Timestamp: time.Now(),
	}

	assert.Equal(t, "opened", update.Action)
	assert.False(t, update.Timestamp.IsZero())
}

func TestPriceUpdateStruct(t *testing.T) {
	update := PriceUpdate{
		Symbol:    "BTC/USDC",
		Price:     50000.0,
		Bid:       49990.0,
		Ask:       50010.0,
		Volume:    1000.0,
		Timestamp: time.Now(),
	}

	assert.Equal(t, "BTC/USDC", update.Symbol)
	assert.Equal(t, 50000.0, update.Price)
	assert.Equal(t, 49990.0, update.Bid)
	assert.Equal(t, 50010.0, update.Ask)
	assert.Equal(t, 1000.0, update.Volume)
}

func TestTradeUpdateStruct(t *testing.T) {
	update := TradeUpdate{
		Trade:     nil,
		Symbol:    "ETH/USDC",
		Side:      "buy",
		Timestamp: time.Now(),
	}

	assert.Equal(t, "ETH/USDC", update.Symbol)
	assert.Equal(t, "buy", update.Side)
}

func TestClientConfigStruct(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.example.com",
		WSEndpoint:  "wss://ws.example.com",
		APIKey:      "key123",
		APISecret:   "secret456",
		UserID:      "user789",
	}

	assert.Equal(t, "https://api.example.com", config.APIEndpoint)
	assert.Equal(t, "wss://ws.example.com", config.WSEndpoint)
	assert.Equal(t, "key123", config.APIKey)
	assert.Equal(t, "secret456", config.APISecret)
	assert.Equal(t, "user789", config.UserID)
}

func TestClientChannelBufferSizes(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Check channel capacities
	assert.Equal(t, 100, cap(client.orderUpdates))
	assert.Equal(t, 100, cap(client.positionUpdates))
	assert.Equal(t, 1000, cap(client.priceUpdates))
	assert.Equal(t, 1000, cap(client.tradeUpdates))
	assert.Equal(t, 10, cap(client.errorChan))
}

func TestDisconnectNotConnected(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Disconnect when not connected should be no-op
	err = client.Disconnect()
	assert.NoError(t, err)
}

func TestConnectAlreadyConnected(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Manually set connected state
	client.mu.Lock()
	client.connected = true
	client.mu.Unlock()

	// Connect when already connected should return error
	err = client.Connect()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already connected")
}

func TestPriceUpdateActions(t *testing.T) {
	actions := []string{"opened", "modified", "closed", "liquidated"}

	for _, action := range actions {
		update := PositionUpdate{Action: action}
		assert.Equal(t, action, update.Action)
	}
}

func TestTradeSides(t *testing.T) {
	sides := []string{"buy", "sell"}

	for _, side := range sides {
		update := TradeUpdate{Side: side}
		assert.Equal(t, side, update.Side)
	}
}

func TestGetBalanceEmpty(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Get balance for non-existent asset returns 0
	balance, err := client.GetBalance("BTC")
	require.NoError(t, err)
	assert.Equal(t, int64(0), balance.Int64())
}

func TestGetBalanceExists(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Set balance directly
	client.mu.Lock()
	client.balances["ETH"] = big.NewInt(1000000000)
	client.mu.Unlock()

	balance, err := client.GetBalance("ETH")
	require.NoError(t, err)
	assert.Equal(t, int64(1000000000), balance.Int64())
}

func TestGetPositionNotFound(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	pos, err := client.GetPosition("nonexistent")
	assert.Error(t, err)
	assert.Nil(t, pos)
	assert.Contains(t, err.Error(), "not found")
}

func TestGetPositionExists(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Add position directly
	testPos := &lx.MarginPosition{
		ID:       "pos123",
		Symbol:   "BTC/USDC",
		Leverage: 10,
	}
	client.mu.Lock()
	client.positions["pos123"] = testPos
	client.mu.Unlock()

	pos, err := client.GetPosition("pos123")
	require.NoError(t, err)
	assert.Equal(t, "pos123", pos.ID)
	assert.Equal(t, "BTC/USDC", pos.Symbol)
	assert.Equal(t, float64(10), pos.Leverage)
}

func TestGetPositionsEmpty(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	positions := client.GetPositions()
	assert.Empty(t, positions)
}

func TestGetPositionsMultiple(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Add positions
	client.mu.Lock()
	client.positions["pos1"] = &lx.MarginPosition{ID: "pos1", Symbol: "BTC/USDC"}
	client.positions["pos2"] = &lx.MarginPosition{ID: "pos2", Symbol: "ETH/USDC"}
	client.mu.Unlock()

	positions := client.GetPositions()
	assert.Len(t, positions, 2)
	assert.NotNil(t, positions["pos1"])
	assert.NotNil(t, positions["pos2"])
}

func TestGetOrdersEmpty(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	orders := client.GetOrders()
	assert.Empty(t, orders)
}

func TestGetOrdersMultiple(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Add orders
	client.mu.Lock()
	client.orders[1] = &lx.Order{ID: 1, Symbol: "BTC/USDC"}
	client.orders[2] = &lx.Order{ID: 2, Symbol: "ETH/USDC"}
	client.orders[3] = &lx.Order{ID: 3, Symbol: "SOL/USDC"}
	client.mu.Unlock()

	orders := client.GetOrders()
	assert.Len(t, orders, 3)
	assert.NotNil(t, orders[1])
	assert.NotNil(t, orders[2])
	assert.NotNil(t, orders[3])
}

func TestGetPriceNotFound(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	price, err := client.GetPrice("BTC/USDC")
	assert.Error(t, err)
	assert.Equal(t, float64(0), price)
	assert.Contains(t, err.Error(), "no price")
}

func TestGetPriceExists(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Set price directly
	client.mu.Lock()
	client.prices["BTC/USDC"] = 50000.0
	client.mu.Unlock()

	price, err := client.GetPrice("BTC/USDC")
	require.NoError(t, err)
	assert.Equal(t, 50000.0, price)
}

func TestGetOrderBookNotFound(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	ob, err := client.GetOrderBook("BTC/USDC")
	assert.Error(t, err)
	assert.Nil(t, ob)
	assert.Contains(t, err.Error(), "no order book")
}

func TestGetOrderBookExists(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Set order book directly
	testOB := &lx.OrderBookSnapshot{
		Symbol: "ETH/USDC",
		Bids:   []lx.OrderLevel{{Price: 3000, Size: 10}},
		Asks:   []lx.OrderLevel{{Price: 3010, Size: 5}},
	}
	client.mu.Lock()
	client.orderBooks["ETH/USDC"] = testOB
	client.mu.Unlock()

	ob, err := client.GetOrderBook("ETH/USDC")
	require.NoError(t, err)
	assert.Equal(t, "ETH/USDC", ob.Symbol)
	assert.Len(t, ob.Bids, 1)
	assert.Len(t, ob.Asks, 1)
}

func TestPlaceOrderNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	order := &lx.Order{Symbol: "BTC/USDC", Price: 50000}
	result, err := client.PlaceOrder(order)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestCancelOrderNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.CancelOrder(12345)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestModifyOrderNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.ModifyOrder(12345, 51000, 2.0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestOpenMarginPositionNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	pos, err := client.OpenMarginPosition("BTC/USDC", lx.Buy, 1.0, 10.0)
	assert.Error(t, err)
	assert.Nil(t, pos)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestClosePositionNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.ClosePosition("pos123", 1.0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestModifyLeverageNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.ModifyLeverage("pos123", 20.0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestDepositToVaultNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.DepositToVault("vault1", big.NewInt(1000))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestWithdrawFromVaultNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.WithdrawFromVault("vault1", big.NewInt(500))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestSupplyNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.Supply("USDC", big.NewInt(10000))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestBorrowNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.Borrow("USDC", big.NewInt(5000))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestRepayNotAuthenticated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	err = client.Repay("USDC", big.NewInt(2000))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not authenticated")
}

func TestOnOrderUpdateCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *OrderUpdate
	client.OnOrderUpdate(func(update *OrderUpdate) {
		received = update
	})

	assert.NotNil(t, client.callbacks["order_update"])

	// Test callback execution
	testUpdate := &OrderUpdate{Message: "test order"}
	client.callbacks["order_update"](testUpdate)
	assert.Equal(t, "test order", received.Message)
}

func TestOnPositionUpdateCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *PositionUpdate
	client.OnPositionUpdate(func(update *PositionUpdate) {
		received = update
	})

	assert.NotNil(t, client.callbacks["position_update"])

	// Test callback execution
	testUpdate := &PositionUpdate{Action: "opened"}
	client.callbacks["position_update"](testUpdate)
	assert.Equal(t, "opened", received.Action)
}

func TestOnPriceUpdateCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *PriceUpdate
	client.OnPriceUpdate(func(update *PriceUpdate) {
		received = update
	})

	assert.NotNil(t, client.callbacks["price_update"])

	// Test callback execution
	testUpdate := &PriceUpdate{Symbol: "BTC/USDC", Price: 50000}
	client.callbacks["price_update"](testUpdate)
	assert.Equal(t, "BTC/USDC", received.Symbol)
	assert.Equal(t, float64(50000), received.Price)
}

func TestOnTradeUpdateCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *TradeUpdate
	client.OnTradeUpdate(func(update *TradeUpdate) {
		received = update
	})

	assert.NotNil(t, client.callbacks["trade_update"])

	// Test callback execution
	testUpdate := &TradeUpdate{Symbol: "ETH/USDC", Side: "buy"}
	client.callbacks["trade_update"](testUpdate)
	assert.Equal(t, "ETH/USDC", received.Symbol)
	assert.Equal(t, "buy", received.Side)
}

func TestProcessMessageOrderUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type": "order_update",
		"order": map[string]interface{}{
			"id":     float64(123),
			"symbol": "BTC/USDC",
		},
	}

	client.processMessage(msg)

	// Check if order was stored
	client.mu.RLock()
	_, exists := client.orders[123]
	client.mu.RUnlock()
	assert.True(t, exists)
}

func TestProcessMessagePositionUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "position_update",
		"action": "opened",
		"position": map[string]interface{}{
			"id":     "pos456",
			"symbol": "ETH/USDC",
		},
	}

	client.processMessage(msg)

	// Check if position was stored
	client.mu.RLock()
	_, exists := client.positions["pos456"]
	client.mu.RUnlock()
	assert.True(t, exists)
}

func TestProcessMessagePositionClosed(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// First add a position
	client.mu.Lock()
	client.positions["pos789"] = &lx.MarginPosition{ID: "pos789"}
	client.mu.Unlock()

	msg := map[string]interface{}{
		"type":   "position_update",
		"action": "closed",
		"position": map[string]interface{}{
			"id": "pos789",
		},
	}

	client.processMessage(msg)

	// Check if position was removed
	client.mu.RLock()
	_, exists := client.positions["pos789"]
	client.mu.RUnlock()
	assert.False(t, exists)
}

func TestProcessMessagePriceUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "price_update",
		"symbol": "SOL/USDC",
		"price":  150.5,
	}

	client.processMessage(msg)

	// Check if price was stored
	client.mu.RLock()
	price := client.prices["SOL/USDC"]
	client.mu.RUnlock()
	assert.Equal(t, 150.5, price)
}

func TestProcessMessageBalanceUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// handleBalanceUpdate expects "balances" map with string values
	msg := map[string]interface{}{
		"type": "balance_update",
		"balances": map[string]interface{}{
			"USDC": "1000000000000",
			"ETH":  "5000000000000000000",
		},
	}

	client.processMessage(msg)

	// Check if balances were stored
	client.mu.RLock()
	usdcBalance := client.balances["USDC"]
	ethBalance := client.balances["ETH"]
	client.mu.RUnlock()
	assert.NotNil(t, usdcBalance)
	assert.NotNil(t, ethBalance)
	assert.Equal(t, "1000000000000", usdcBalance.String())
	assert.Equal(t, "5000000000000000000", ethBalance.String())
}

func TestProcessMessageInvalidType(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"invalid": "no type field",
	}

	// Should not panic
	client.processMessage(msg)
}

func TestProcessMessageUnknownType(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type": "unknown_type",
	}

	// Should not panic
	client.processMessage(msg)
}

func TestConcurrentMapAccess(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	done := make(chan bool, 100)

	// Concurrent writes
	for i := 0; i < 50; i++ {
		go func(idx int) {
			client.mu.Lock()
			client.prices[fmt.Sprintf("PAIR%d/USDC", idx)] = float64(idx * 100)
			client.mu.Unlock()
			done <- true
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 50; i++ {
		go func() {
			_, _ = client.GetPrice("BTC/USDC")
			done <- true
		}()
	}

	// Wait for all
	for i := 0; i < 100; i++ {
		<-done
	}

	// Should have 50 prices
	client.mu.RLock()
	count := len(client.prices)
	client.mu.RUnlock()
	assert.Equal(t, 50, count)
}

func BenchmarkGetBalance(b *testing.B) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}
	client, _ := NewTraderClient(config)
	client.balances["BTC"] = big.NewInt(1000000000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.GetBalance("BTC")
	}
}

func BenchmarkGetPrice(b *testing.B) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}
	client, _ := NewTraderClient(config)
	client.prices["BTC/USDC"] = 50000.0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.GetPrice("BTC/USDC")
	}
}

func BenchmarkProcessMessage(b *testing.B) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}
	client, _ := NewTraderClient(config)

	msg := map[string]interface{}{
		"type":   "price_update",
		"symbol": "BTC/USDC",
		"price":  50000.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.processMessage(msg)
	}
}

// Test handleTradeUpdate
func TestHandleTradeUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "trade_update",
		"symbol": "BTC/USDC",
		"trade": map[string]interface{}{
			"id":       "trade123",
			"price":    50000.0,
			"quantity": 1.5,
		},
	}

	client.handleTradeUpdate(msg)

	// Check if trade was stored
	client.mu.RLock()
	trades := client.trades["BTC/USDC"]
	client.mu.RUnlock()
	assert.Len(t, trades, 1)
}

// Test handleTradeUpdate without symbol
func TestHandleTradeUpdateNoSymbol(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type": "trade_update",
		"trade": map[string]interface{}{
			"id":       "trade123",
			"price":    50000.0,
			"quantity": 1.5,
		},
	}

	// Should not panic
	client.handleTradeUpdate(msg)
}

// Test handleTradeUpdate with trade history limit
func TestHandleTradeUpdateHistoryLimit(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Add more than 100 trades
	for i := 0; i < 105; i++ {
		msg := map[string]interface{}{
			"type":   "trade_update",
			"symbol": "BTC/USDC",
			"trade": map[string]interface{}{
				"id":    fmt.Sprintf("trade%d", i),
				"price": float64(50000 + i),
			},
		}
		client.handleTradeUpdate(msg)
	}

	// Check trade history is limited to 100
	client.mu.RLock()
	trades := client.trades["BTC/USDC"]
	client.mu.RUnlock()
	assert.LessOrEqual(t, len(trades), 100)
}

// Test handleOrderBookUpdate
func TestHandleOrderBookUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "orderbook_update",
		"symbol": "ETH/USDC",
		"snapshot": map[string]interface{}{
			"symbol": "ETH/USDC",
			"bids":   []interface{}{},
			"asks":   []interface{}{},
		},
	}

	client.handleOrderBookUpdate(msg)

	// Check if order book was stored
	client.mu.RLock()
	ob := client.orderBooks["ETH/USDC"]
	client.mu.RUnlock()
	assert.NotNil(t, ob)
}

// Test handleOrderBookUpdate missing symbol
func TestHandleOrderBookUpdateNoSymbol(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type": "orderbook_update",
		"snapshot": map[string]interface{}{
			"bids": []interface{}{},
		},
	}

	// Should not panic
	client.handleOrderBookUpdate(msg)
}

// Test handleError
func TestHandleError(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":  "error",
		"error": "test error message",
	}

	client.handleError(msg)

	// Check error was sent to channel
	select {
	case err := <-client.errorChan:
		assert.Contains(t, err.Error(), "test error message")
	default:
		t.Fatal("expected error in channel")
	}
}

// Test handleError no error field
func TestHandleErrorNoField(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type": "error",
	}

	// Should not panic
	client.handleError(msg)
}

// Test GetOrderUpdates channel access
func TestGetOrderUpdates(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	ch := client.GetOrderUpdates()
	assert.NotNil(t, ch)

	// Send an update and verify
	go func() {
		client.orderUpdates <- &OrderUpdate{Message: "test"}
	}()

	select {
	case update := <-ch:
		assert.Equal(t, "test", update.Message)
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for order update")
	}
}

// Test GetPositionUpdates channel access
func TestGetPositionUpdates(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	ch := client.GetPositionUpdates()
	assert.NotNil(t, ch)

	// Send an update and verify
	go func() {
		client.positionUpdates <- &PositionUpdate{Action: "opened"}
	}()

	select {
	case update := <-ch:
		assert.Equal(t, "opened", update.Action)
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for position update")
	}
}

// Test GetPriceUpdates channel access
func TestGetPriceUpdates(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	ch := client.GetPriceUpdates()
	assert.NotNil(t, ch)

	// Send an update and verify
	go func() {
		client.priceUpdates <- &PriceUpdate{Symbol: "BTC/USDC", Price: 50000}
	}()

	select {
	case update := <-ch:
		assert.Equal(t, "BTC/USDC", update.Symbol)
		assert.Equal(t, float64(50000), update.Price)
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for price update")
	}
}

// Test GetTradeUpdates channel access
func TestGetTradeUpdates(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	ch := client.GetTradeUpdates()
	assert.NotNil(t, ch)

	// Send an update and verify
	go func() {
		client.tradeUpdates <- &TradeUpdate{Symbol: "ETH/USDC", Side: "buy"}
	}()

	select {
	case update := <-ch:
		assert.Equal(t, "ETH/USDC", update.Symbol)
		assert.Equal(t, "buy", update.Side)
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for trade update")
	}
}

// Test GetErrors channel access
func TestGetErrors(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	ch := client.GetErrors()
	assert.NotNil(t, ch)

	// Send an error and verify
	go func() {
		client.errorChan <- fmt.Errorf("test error")
	}()

	select {
	case err := <-ch:
		assert.Contains(t, err.Error(), "test error")
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for error")
	}
}

// Test processMessage with trade_update type
func TestProcessMessageTradeUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "trade_update",
		"symbol": "BTC/USDC",
		"trade": map[string]interface{}{
			"id":    "trade1",
			"price": 51000.0,
		},
	}

	client.processMessage(msg)

	// Check if trade was stored
	client.mu.RLock()
	trades := client.trades["BTC/USDC"]
	client.mu.RUnlock()
	assert.Len(t, trades, 1)
}

// Test processMessage with orderbook_update type
func TestProcessMessageOrderbookUpdate(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "orderbook_update",
		"symbol": "SOL/USDC",
		"snapshot": map[string]interface{}{
			"symbol": "SOL/USDC",
			"bids":   []interface{}{},
			"asks":   []interface{}{},
		},
	}

	client.processMessage(msg)

	// Check if orderbook was stored
	client.mu.RLock()
	ob := client.orderBooks["SOL/USDC"]
	client.mu.RUnlock()
	assert.NotNil(t, ob)
}

// Test processMessage with error type
func TestProcessMessageError(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":  "error",
		"error": "some server error",
	}

	client.processMessage(msg)

	// Check error was sent to channel
	select {
	case err := <-client.errorChan:
		assert.Contains(t, err.Error(), "some server error")
	default:
		t.Fatal("expected error in channel")
	}
}

// Test handlePriceUpdate with bid/ask
func TestHandlePriceUpdateWithBidAsk(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	msg := map[string]interface{}{
		"type":   "price_update",
		"symbol": "BTC/USDC",
		"price":  50000.0,
		"bid":    49990.0,
		"ask":    50010.0,
		"volume": 1000.0,
	}

	client.handlePriceUpdate(msg)

	// Check if price was stored
	client.mu.RLock()
	price := client.prices["BTC/USDC"]
	client.mu.RUnlock()
	assert.Equal(t, 50000.0, price)
}

// Test handlePriceUpdate callback
func TestHandlePriceUpdateWithCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *PriceUpdate
	client.OnPriceUpdate(func(update *PriceUpdate) {
		received = update
	})

	msg := map[string]interface{}{
		"type":   "price_update",
		"symbol": "ETH/USDC",
		"price":  3000.0,
		"bid":    2999.0,
		"ask":    3001.0,
	}

	client.handlePriceUpdate(msg)

	// Check callback was called
	require.NotNil(t, received)
	assert.Equal(t, "ETH/USDC", received.Symbol)
	assert.Equal(t, 3000.0, received.Price)
	assert.Equal(t, 2999.0, received.Bid)
	assert.Equal(t, 3001.0, received.Ask)
}

// Test handleOrderUpdate with callback
func TestHandleOrderUpdateWithCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *OrderUpdate
	client.OnOrderUpdate(func(update *OrderUpdate) {
		received = update
	})

	msg := map[string]interface{}{
		"type":   "order_update",
		"status": float64(1),
		"order": map[string]interface{}{
			"id":       float64(456),
			"symbol":   "BTC/USDC",
			"price":    51000.0,
			"quantity": 1.0,
		},
	}

	client.handleOrderUpdate(msg)

	// Check callback was called
	require.NotNil(t, received)
	assert.NotNil(t, received.Order)
}

// Test handlePositionUpdate with callback
func TestHandlePositionUpdateWithCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *PositionUpdate
	client.OnPositionUpdate(func(update *PositionUpdate) {
		received = update
	})

	msg := map[string]interface{}{
		"type":   "position_update",
		"action": "modified",
		"position": map[string]interface{}{
			"id":       "pos999",
			"symbol":   "BTC/USDC",
			"leverage": 5.0,
		},
	}

	client.handlePositionUpdate(msg)

	// Check callback was called
	require.NotNil(t, received)
	assert.Equal(t, "modified", received.Action)
}

// Test handleTradeUpdate with callback
func TestHandleTradeUpdateWithCallback(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	var received *TradeUpdate
	client.OnTradeUpdate(func(update *TradeUpdate) {
		received = update
	})

	msg := map[string]interface{}{
		"type":   "trade_update",
		"symbol": "SOL/USDC",
		"trade": map[string]interface{}{
			"id":       "tradeXYZ",
			"price":    150.0,
			"quantity": 10.0,
		},
	}

	client.handleTradeUpdate(msg)

	// Check callback was called
	require.NotNil(t, received)
	assert.Equal(t, "SOL/USDC", received.Symbol)
}

// Test position update with liquidated action
func TestHandlePositionUpdateLiquidated(t *testing.T) {
	config := ClientConfig{
		APIEndpoint: "https://api.test.com",
		WSEndpoint:  "wss://ws.test.com",
	}

	client, err := NewTraderClient(config)
	require.NoError(t, err)

	// Add a position first
	client.mu.Lock()
	client.positions["liqPos"] = &lx.MarginPosition{ID: "liqPos"}
	client.mu.Unlock()

	msg := map[string]interface{}{
		"type":   "position_update",
		"action": "liquidated",
		"position": map[string]interface{}{
			"id": "liqPos",
		},
	}

	client.handlePositionUpdate(msg)

	// Check if position was removed (liquidated = removed)
	client.mu.RLock()
	_, exists := client.positions["liqPos"]
	client.mu.RUnlock()
	assert.False(t, exists)
}
