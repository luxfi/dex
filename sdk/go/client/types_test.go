package client

import (
	"testing"
)

func TestOrderTypes(t *testing.T) {
	tests := []struct {
		name  string
		ot    OrderType
		value int32
	}{
		{"Limit", OrderTypeLimit, 0},
		{"Market", OrderTypeMarket, 1},
		{"Stop", OrderTypeStop, 2},
		{"StopLimit", OrderTypeStopLimit, 3},
		{"Iceberg", OrderTypeIceberg, 4},
		{"Peg", OrderTypePeg, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.ot) != tt.value {
				t.Errorf("OrderType %s = %d, want %d", tt.name, tt.ot, tt.value)
			}
		})
	}
}

func TestOrderSides(t *testing.T) {
	if int32(OrderSideBuy) != 0 {
		t.Errorf("OrderSideBuy = %d, want 0", OrderSideBuy)
	}
	if int32(OrderSideSell) != 1 {
		t.Errorf("OrderSideSell = %d, want 1", OrderSideSell)
	}
}

func TestOrderStatus(t *testing.T) {
	tests := []struct {
		name   string
		status OrderStatus
		value  string
	}{
		{"Open", OrderStatusOpen, "open"},
		{"Partial", OrderStatusPartial, "partial"},
		{"Filled", OrderStatusFilled, "filled"},
		{"Cancelled", OrderStatusCancelled, "cancelled"},
		{"Rejected", OrderStatusRejected, "rejected"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.status) != tt.value {
				t.Errorf("OrderStatus %s = %s, want %s", tt.name, tt.status, tt.value)
			}
		})
	}
}

func TestTimeInForce(t *testing.T) {
	tests := []struct {
		name  string
		tif   TimeInForce
		value string
	}{
		{"GTC", TimeInForceGTC, "GTC"},
		{"IOC", TimeInForceIOC, "IOC"},
		{"FOK", TimeInForceFOK, "FOK"},
		{"DAY", TimeInForceDAY, "DAY"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.tif) != tt.value {
				t.Errorf("TimeInForce %s = %s, want %s", tt.name, tt.tif, tt.value)
			}
		})
	}
}

func TestOrderIsOpen(t *testing.T) {
	order := &Order{Status: OrderStatusOpen}
	if !order.IsOpen() {
		t.Error("Order with status OPEN should be open")
	}

	order.Status = OrderStatusPartial
	if !order.IsOpen() {
		t.Error("Order with status PARTIAL should be open")
	}

	order.Status = OrderStatusFilled
	if order.IsOpen() {
		t.Error("Order with status FILLED should not be open")
	}
}

func TestOrderIsClosed(t *testing.T) {
	closedStatuses := []OrderStatus{OrderStatusFilled, OrderStatusCancelled, OrderStatusRejected}
	for _, status := range closedStatuses {
		order := &Order{Status: status}
		if !order.IsClosed() {
			t.Errorf("Order with status %s should be closed", status)
		}
	}

	openStatuses := []OrderStatus{OrderStatusOpen, OrderStatusPartial}
	for _, status := range openStatuses {
		order := &Order{Status: status}
		if order.IsClosed() {
			t.Errorf("Order with status %s should not be closed", status)
		}
	}
}

func TestOrderFillRate(t *testing.T) {
	order := &Order{Size: 10.0, Filled: 5.0}
	if order.FillRate() != 0.5 {
		t.Errorf("FillRate = %f, want 0.5", order.FillRate())
	}

	order.Filled = 10.0
	if order.FillRate() != 1.0 {
		t.Errorf("FillRate = %f, want 1.0", order.FillRate())
	}

	order.Size = 0
	if order.FillRate() != 0 {
		t.Errorf("FillRate with zero size = %f, want 0", order.FillRate())
	}
}

func TestTradeTotalValue(t *testing.T) {
	trade := &Trade{Price: 50000.0, Size: 2.0}
	expected := 100000.0
	if trade.TotalValue() != expected {
		t.Errorf("TotalValue = %f, want %f", trade.TotalValue(), expected)
	}
}

func TestTradeTimestampTime(t *testing.T) {
	trade := &Trade{Timestamp: 1704067200} // 2024-01-01
	tm := trade.TimestampTime()
	if tm.Unix() != 1704067200 {
		t.Errorf("TimestampTime = %d, want 1704067200", tm.Unix())
	}
}

func TestOrderBookBestBid(t *testing.T) {
	book := &OrderBook{
		Bids: []PriceLevel{{Price: 50000}, {Price: 49999}},
	}
	if book.BestBid() != 50000 {
		t.Errorf("BestBid = %f, want 50000", book.BestBid())
	}

	book.Bids = nil
	if book.BestBid() != 0 {
		t.Errorf("BestBid with no bids = %f, want 0", book.BestBid())
	}
}

func TestOrderBookBestAsk(t *testing.T) {
	book := &OrderBook{
		Asks: []PriceLevel{{Price: 50001}, {Price: 50002}},
	}
	if book.BestAsk() != 50001 {
		t.Errorf("BestAsk = %f, want 50001", book.BestAsk())
	}

	book.Asks = nil
	if book.BestAsk() != 0 {
		t.Errorf("BestAsk with no asks = %f, want 0", book.BestAsk())
	}
}

func TestOrderBookSpread(t *testing.T) {
	book := &OrderBook{
		Bids: []PriceLevel{{Price: 50000}},
		Asks: []PriceLevel{{Price: 50001}},
	}
	if book.Spread() != 1.0 {
		t.Errorf("Spread = %f, want 1.0", book.Spread())
	}

	book.Bids = nil
	if book.Spread() != 0 {
		t.Errorf("Spread with no bids = %f, want 0", book.Spread())
	}
}

func TestOrderBookMidPrice(t *testing.T) {
	book := &OrderBook{
		Bids: []PriceLevel{{Price: 50000}},
		Asks: []PriceLevel{{Price: 50002}},
	}
	if book.MidPrice() != 50001 {
		t.Errorf("MidPrice = %f, want 50001", book.MidPrice())
	}

	book.Bids = nil
	if book.MidPrice() != 0 {
		t.Errorf("MidPrice with no bids = %f, want 0", book.MidPrice())
	}
}

func TestOrderBookSpreadPercentage(t *testing.T) {
	book := &OrderBook{
		Bids: []PriceLevel{{Price: 50000}},
		Asks: []PriceLevel{{Price: 50100}},
	}
	// spread = 100, mid = 50050, percentage = 100/50050 * 100 = ~0.1998
	expected := (100.0 / 50050.0) * 100
	result := book.SpreadPercentage()
	if result < expected-0.01 || result > expected+0.01 {
		t.Errorf("SpreadPercentage = %f, want ~%f", result, expected)
	}
}

func TestBalanceUtilization(t *testing.T) {
	balance := &Balance{Locked: 5000, Total: 10000}
	if balance.Utilization() != 0.5 {
		t.Errorf("Utilization = %f, want 0.5", balance.Utilization())
	}

	balance.Total = 0
	if balance.Utilization() != 0 {
		t.Errorf("Utilization with zero total = %f, want 0", balance.Utilization())
	}
}

func TestPositionUnrealizedPnL(t *testing.T) {
	pos := &Position{
		Size:       1.0,
		EntryPrice: 50000,
		MarkPrice:  51000,
	}
	expected := 1000.0
	if pos.UnrealizedPnL() != expected {
		t.Errorf("UnrealizedPnL = %f, want %f", pos.UnrealizedPnL(), expected)
	}
}

func TestPositionPnLPercentage(t *testing.T) {
	pos := &Position{
		Size:       1.0,
		EntryPrice: 50000,
		MarkPrice:  51000,
	}
	expected := 2.0 // 1000/50000 * 100 = 2%
	if pos.PnLPercentage() != expected {
		t.Errorf("PnLPercentage = %f, want %f", pos.PnLPercentage(), expected)
	}

	pos.EntryPrice = 0
	if pos.PnLPercentage() != 0 {
		t.Errorf("PnLPercentage with zero entry = %f, want 0", pos.PnLPercentage())
	}
}
