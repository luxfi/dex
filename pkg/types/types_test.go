package types

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOrderStruct(t *testing.T) {
	order := Order{
		ID:        1,
		Symbol:    "BTC/USDC",
		Side:      "buy",
		Price:     50000.0,
		Quantity:  1.5,
		Type:      "limit",
		Status:    "pending",
		Timestamp: time.Now(),
		User:      "user123",
		Filled:    0.5,
	}

	assert.Equal(t, uint64(1), order.ID)
	assert.Equal(t, "BTC/USDC", order.Symbol)
	assert.Equal(t, "buy", order.Side)
	assert.Equal(t, 50000.0, order.Price)
	assert.Equal(t, 1.5, order.Quantity)
	assert.Equal(t, "limit", order.Type)
	assert.Equal(t, "pending", order.Status)
	assert.Equal(t, "user123", order.User)
	assert.Equal(t, 0.5, order.Filled)
}

func TestOrderJSONSerialization(t *testing.T) {
	order := Order{
		ID:        123,
		Symbol:    "ETH/USDC",
		Side:      "sell",
		Price:     3000.0,
		Quantity:  10.0,
		Type:      "market",
		Status:    "filled",
		Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
		User:      "trader1",
		Filled:    10.0,
	}

	// Serialize
	data, err := json.Marshal(order)
	require.NoError(t, err)

	// Deserialize
	var decoded Order
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, order.ID, decoded.ID)
	assert.Equal(t, order.Symbol, decoded.Symbol)
	assert.Equal(t, order.Side, decoded.Side)
	assert.Equal(t, order.Price, decoded.Price)
	assert.Equal(t, order.Quantity, decoded.Quantity)
	assert.Equal(t, order.Type, decoded.Type)
	assert.Equal(t, order.Status, decoded.Status)
	assert.Equal(t, order.User, decoded.User)
	assert.Equal(t, order.Filled, decoded.Filled)
}

func TestTradeStruct(t *testing.T) {
	trade := Trade{
		ID:        1,
		Symbol:    "BTC/USDC",
		Price:     50000.0,
		Size:      0.5,
		BuyerID:   "buyer123",
		SellerID:  "seller456",
		Timestamp: time.Now(),
	}

	assert.Equal(t, uint64(1), trade.ID)
	assert.Equal(t, "BTC/USDC", trade.Symbol)
	assert.Equal(t, 50000.0, trade.Price)
	assert.Equal(t, 0.5, trade.Size)
	assert.Equal(t, "buyer123", trade.BuyerID)
	assert.Equal(t, "seller456", trade.SellerID)
}

func TestTradeJSONSerialization(t *testing.T) {
	trade := Trade{
		ID:        456,
		Symbol:    "SOL/USDC",
		Price:     100.0,
		Size:      50.0,
		BuyerID:   "alice",
		SellerID:  "bob",
		Timestamp: time.Date(2025, 6, 15, 10, 30, 0, 0, time.UTC),
	}

	data, err := json.Marshal(trade)
	require.NoError(t, err)

	var decoded Trade
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, trade.ID, decoded.ID)
	assert.Equal(t, trade.Symbol, decoded.Symbol)
	assert.Equal(t, trade.Price, decoded.Price)
	assert.Equal(t, trade.Size, decoded.Size)
	assert.Equal(t, trade.BuyerID, decoded.BuyerID)
	assert.Equal(t, trade.SellerID, decoded.SellerID)
}

func TestOrderBookStruct(t *testing.T) {
	ob := OrderBook{
		Symbol: "BTC/USDC",
		Bids: [][]float64{
			{49900.0, 1.0},
			{49800.0, 2.0},
			{49700.0, 3.0},
		},
		Asks: [][]float64{
			{50100.0, 1.5},
			{50200.0, 2.5},
			{50300.0, 3.5},
		},
		Time: time.Now(),
	}

	assert.Equal(t, "BTC/USDC", ob.Symbol)
	assert.Len(t, ob.Bids, 3)
	assert.Len(t, ob.Asks, 3)
	assert.Equal(t, 49900.0, ob.Bids[0][0])
	assert.Equal(t, 50100.0, ob.Asks[0][0])
}

func TestOrderBookJSONSerialization(t *testing.T) {
	ob := OrderBook{
		Symbol: "ETH/BTC",
		Bids:   [][]float64{{0.05, 10.0}},
		Asks:   [][]float64{{0.051, 5.0}},
		Time:   time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
	}

	data, err := json.Marshal(ob)
	require.NoError(t, err)

	var decoded OrderBook
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, ob.Symbol, decoded.Symbol)
	assert.Equal(t, ob.Bids, decoded.Bids)
	assert.Equal(t, ob.Asks, decoded.Asks)
}

func TestResponseStruct(t *testing.T) {
	t.Run("SuccessResponse", func(t *testing.T) {
		resp := Response{
			Success: true,
			Message: "Order placed successfully",
			Data:    map[string]interface{}{"order_id": 123},
		}

		assert.True(t, resp.Success)
		assert.Equal(t, "Order placed successfully", resp.Message)
		assert.NotNil(t, resp.Data)
		assert.Empty(t, resp.Error)
	})

	t.Run("ErrorResponse", func(t *testing.T) {
		resp := Response{
			Success: false,
			Error:   "Insufficient balance",
		}

		assert.False(t, resp.Success)
		assert.Equal(t, "Insufficient balance", resp.Error)
		assert.Nil(t, resp.Data)
	})
}

func TestResponseJSONSerialization(t *testing.T) {
	resp := Response{
		Success: true,
		Message: "Trade executed",
		Data: map[string]interface{}{
			"trade_id": 789,
			"price":    50000.0,
		},
	}

	data, err := json.Marshal(resp)
	require.NoError(t, err)

	var decoded Response
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, resp.Success, decoded.Success)
	assert.Equal(t, resp.Message, decoded.Message)
	assert.NotNil(t, decoded.Data)
}

func TestOrderSides(t *testing.T) {
	buyOrder := Order{Side: "buy"}
	sellOrder := Order{Side: "sell"}

	assert.Equal(t, "buy", buyOrder.Side)
	assert.Equal(t, "sell", sellOrder.Side)
}

func TestOrderTypes(t *testing.T) {
	tests := []struct {
		name     string
		orderType string
	}{
		{"Market", "market"},
		{"Limit", "limit"},
		{"StopLoss", "stop_loss"},
		{"StopLimit", "stop_limit"},
		{"TakeProfit", "take_profit"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			order := Order{Type: tt.orderType}
			assert.Equal(t, tt.orderType, order.Type)
		})
	}
}

func TestOrderStatuses(t *testing.T) {
	statuses := []string{"pending", "filled", "cancelled", "partial", "expired"}

	for _, status := range statuses {
		order := Order{Status: status}
		assert.Equal(t, status, order.Status)
	}
}
