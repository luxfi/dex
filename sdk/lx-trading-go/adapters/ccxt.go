// Package adapters provides venue adapter implementations.
package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/luxfi/trading"
	"github.com/shopspring/decimal"
)

// CcxtAdapter implements VenueAdapter for CCXT exchanges via HTTP gateway.
// CCXT runs as a separate service and this adapter communicates via REST.
type CcxtAdapter struct {
	trading.BaseAdapter
	config *trading.CcxtConfig
	client *http.Client
	name   string
}

// NewCcxtAdapter creates a new CCXT adapter.
func NewCcxtAdapter(name string, config *trading.CcxtConfig) *CcxtAdapter {
	return &CcxtAdapter{
		name:   name,
		config: config,
	}
}

// Name returns the venue name.
func (a *CcxtAdapter) Name() string {
	return a.name
}

// VenueType returns the venue type.
func (a *CcxtAdapter) VenueType() trading.VenueType {
	return trading.VenueTypeCCXT
}

// Capabilities returns venue capabilities.
func (a *CcxtAdapter) Capabilities() trading.VenueCapabilities {
	caps := trading.OrderBookCapabilities()
	caps.BatchOrders = false // CCXT doesn't have unified batch
	return caps
}

// Connect establishes connection to the CCXT gateway.
func (a *CcxtAdapter) Connect(ctx context.Context) error {
	a.client = &http.Client{Timeout: 30 * time.Second}

	// Load markets to verify connection
	start := time.Now()
	_, err := a.doRequest(ctx, http.MethodPost, "/load_markets", map[string]interface{}{
		"exchange": a.config.ExchangeID,
		"apiKey":   a.config.APIKey,
		"secret":   a.config.APISecret,
		"password": a.config.Password,
		"sandbox":  a.config.Sandbox,
	})
	if err != nil {
		return fmt.Errorf("failed to connect to CCXT gateway: %w", err)
	}

	a.SetLatency(time.Since(start).Milliseconds())
	a.SetConnected(true)
	return nil
}

// Disconnect closes the connection.
func (a *CcxtAdapter) Disconnect(ctx context.Context) error {
	a.client = nil
	a.SetConnected(false)
	return nil
}

func (a *CcxtAdapter) doRequest(ctx context.Context, method, path string, body interface{}) (map[string]interface{}, error) {
	if a.client == nil {
		return nil, trading.ErrNotConnected
	}

	url := a.config.BaseURL + path

	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	a.SetLatency(time.Since(start).Milliseconds())

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, trading.NewTradingError("HTTP_ERROR", fmt.Sprintf("status %d: %s", resp.StatusCode, string(body)))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

func (a *CcxtAdapter) doRequestArray(ctx context.Context, method, path string, body interface{}) ([]map[string]interface{}, error) {
	if a.client == nil {
		return nil, trading.ErrNotConnected
	}

	url := a.config.BaseURL + path

	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	a.SetLatency(time.Since(start).Milliseconds())

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, trading.NewTradingError("HTTP_ERROR", fmt.Sprintf("status %d: %s", resp.StatusCode, string(body)))
	}

	var result []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

func (a *CcxtAdapter) exchangeParams() map[string]interface{} {
	return map[string]interface{}{
		"exchange": a.config.ExchangeID,
		"apiKey":   a.config.APIKey,
		"secret":   a.config.APISecret,
		"password": a.config.Password,
		"sandbox":  a.config.Sandbox,
	}
}

// GetMarkets returns available markets.
func (a *CcxtAdapter) GetMarkets(ctx context.Context) ([]trading.MarketInfo, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/markets", a.exchangeParams())
	if err != nil {
		return nil, err
	}

	var markets []trading.MarketInfo
	for symbol, mkt := range data {
		m, ok := mkt.(map[string]interface{})
		if !ok {
			continue
		}

		precision := getMap(m, "precision")
		limits := getMap(m, "limits")
		amountLimits := getMap(limits, "amount")
		costLimits := getMap(limits, "cost")

		minQty := decimal.NewFromFloat(getFloat(amountLimits, "min"))
		maxQty := decimal.NewFromFloat(getFloat(amountLimits, "max"))
		minNotional := decimal.NewFromFloat(getFloat(costLimits, "min"))

		var maxQtyPtr, minNotPtr *decimal.Decimal
		if !maxQty.IsZero() {
			maxQtyPtr = &maxQty
		}
		if !minNotional.IsZero() {
			minNotPtr = &minNotional
		}

		markets = append(markets, trading.MarketInfo{
			Symbol:            symbol,
			Base:              getString(m, "base"),
			Quote:             getString(m, "quote"),
			PricePrecision:    getInt(precision, "price"),
			QuantityPrecision: getInt(precision, "amount"),
			MinQuantity:       minQty,
			MaxQuantity:       maxQtyPtr,
			MinNotional:       minNotPtr,
			TickSize:          decimal.NewFromFloat(0.00000001),
			LotSize:           decimal.NewFromFloat(0.00000001),
		})
	}

	return markets, nil
}

// GetTicker returns ticker for a symbol.
func (a *CcxtAdapter) GetTicker(ctx context.Context, symbol string) (trading.Ticker, error) {
	params := a.exchangeParams()
	params["symbol"] = symbol

	data, err := a.doRequest(ctx, http.MethodPost, "/fetch_ticker", params)
	if err != nil {
		return trading.Ticker{}, err
	}

	ticker := trading.Ticker{
		Symbol:    symbol,
		Venue:     a.name,
		Timestamp: time.UnixMilli(int64(getFloat(data, "timestamp"))),
	}

	if bid := getFloat(data, "bid"); bid > 0 {
		d := decimal.NewFromFloat(bid)
		ticker.Bid = &d
	}
	if ask := getFloat(data, "ask"); ask > 0 {
		d := decimal.NewFromFloat(ask)
		ticker.Ask = &d
	}
	if last := getFloat(data, "last"); last > 0 {
		d := decimal.NewFromFloat(last)
		ticker.Last = &d
	}
	if vol := getFloat(data, "baseVolume"); vol > 0 {
		d := decimal.NewFromFloat(vol)
		ticker.Volume24H = &d
	}
	if high := getFloat(data, "high"); high > 0 {
		d := decimal.NewFromFloat(high)
		ticker.High24H = &d
	}
	if low := getFloat(data, "low"); low > 0 {
		d := decimal.NewFromFloat(low)
		ticker.Low24H = &d
	}
	if pct := getFloat(data, "percentage"); pct != 0 {
		d := decimal.NewFromFloat(pct)
		ticker.Change24H = &d
	}

	return ticker, nil
}

// GetOrderbook returns the orderbook for a symbol.
func (a *CcxtAdapter) GetOrderbook(ctx context.Context, symbol string, depth int) (*trading.Orderbook, error) {
	params := a.exchangeParams()
	params["symbol"] = symbol
	if depth > 0 {
		params["limit"] = depth
	}

	data, err := a.doRequest(ctx, http.MethodPost, "/fetch_order_book", params)
	if err != nil {
		return nil, err
	}

	book := trading.NewOrderbook(symbol, a.name)

	if bids, ok := data["bids"].([]interface{}); ok {
		for _, b := range bids {
			if bid, ok := b.([]interface{}); ok && len(bid) >= 2 {
				price := decimal.NewFromFloat(toFloat(bid[0]))
				qty := decimal.NewFromFloat(toFloat(bid[1]))
				book.AddBid(price, qty)
			}
		}
	}

	if asks, ok := data["asks"].([]interface{}); ok {
		for _, a := range asks {
			if ask, ok := a.([]interface{}); ok && len(ask) >= 2 {
				price := decimal.NewFromFloat(toFloat(ask[0]))
				qty := decimal.NewFromFloat(toFloat(ask[1]))
				book.AddAsk(price, qty)
			}
		}
	}

	book.Sort()
	return book, nil
}

// GetTrades returns recent trades.
func (a *CcxtAdapter) GetTrades(ctx context.Context, symbol string, limit int) ([]trading.Trade, error) {
	params := a.exchangeParams()
	params["symbol"] = symbol
	if limit > 0 {
		params["limit"] = limit
	}

	data, err := a.doRequestArray(ctx, http.MethodPost, "/fetch_trades", params)
	if err != nil {
		return nil, err
	}

	trades := make([]trading.Trade, len(data))
	for i, t := range data {
		side := trading.SideBuy
		if getString(t, "side") == "sell" {
			side = trading.SideSell
		}

		fee := trading.Fee{}
		if feeData := getMap(t, "fee"); feeData != nil {
			fee.Asset = getString(feeData, "currency")
			fee.Amount = decimal.NewFromFloat(getFloat(feeData, "cost"))
		}

		trades[i] = trading.Trade{
			TradeID:   getString(t, "id"),
			OrderID:   getString(t, "order"),
			Symbol:    symbol,
			Venue:     a.name,
			Side:      side,
			Price:     decimal.NewFromFloat(getFloat(t, "price")),
			Quantity:  decimal.NewFromFloat(getFloat(t, "amount")),
			Fee:       fee,
			Timestamp: time.UnixMilli(int64(getFloat(t, "timestamp"))),
			IsMaker:   getString(t, "takerOrMaker") == "maker",
		}
	}

	return trades, nil
}

// GetBalances returns all balances.
func (a *CcxtAdapter) GetBalances(ctx context.Context) ([]trading.Balance, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/fetch_balance", a.exchangeParams())
	if err != nil {
		return nil, err
	}

	var balances []trading.Balance

	total := getMap(data, "total")
	free := getMap(data, "free")
	used := getMap(data, "used")

	for asset, amount := range total {
		amt := toFloat(amount)
		if amt > 0 {
			balances = append(balances, trading.Balance{
				Asset:  asset,
				Venue:  a.name,
				Free:   decimal.NewFromFloat(getFloat(free, asset)),
				Locked: decimal.NewFromFloat(getFloat(used, asset)),
			})
		}
	}

	return balances, nil
}

// GetBalance returns balance for a specific asset.
func (a *CcxtAdapter) GetBalance(ctx context.Context, asset string) (trading.Balance, error) {
	balances, err := a.GetBalances(ctx)
	if err != nil {
		return trading.Balance{}, err
	}

	for _, b := range balances {
		if b.Asset == asset {
			return b, nil
		}
	}

	return trading.Balance{
		Asset:  asset,
		Venue:  a.name,
		Free:   decimal.Zero,
		Locked: decimal.Zero,
	}, nil
}

// GetOpenOrders returns open orders.
func (a *CcxtAdapter) GetOpenOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	params := a.exchangeParams()
	if symbol != "" {
		params["symbol"] = symbol
	}

	data, err := a.doRequestArray(ctx, http.MethodPost, "/fetch_open_orders", params)
	if err != nil {
		return nil, err
	}

	orders := make([]trading.Order, len(data))
	for i, o := range data {
		orders[i] = a.convertOrder(o)
	}

	return orders, nil
}

// PlaceOrder places a new order.
func (a *CcxtAdapter) PlaceOrder(ctx context.Context, request trading.OrderRequest) (trading.Order, error) {
	params := a.exchangeParams()
	params["symbol"] = request.Symbol
	params["type"] = "limit"
	if request.OrderType == trading.OrderTypeMarket {
		params["type"] = "market"
	}
	params["side"] = "buy"
	if request.Side == trading.SideSell {
		params["side"] = "sell"
	}
	params["amount"] = request.Quantity.InexactFloat64()
	if request.Price != nil {
		params["price"] = request.Price.InexactFloat64()
	}
	params["params"] = map[string]interface{}{
		"clientOrderId": request.ClientOrderID,
	}

	data, err := a.doRequest(ctx, http.MethodPost, "/create_order", params)
	if err != nil {
		return trading.Order{}, err
	}

	return a.convertOrder(data), nil
}

// CancelOrder cancels an order.
func (a *CcxtAdapter) CancelOrder(ctx context.Context, orderID, symbol string) (trading.Order, error) {
	params := a.exchangeParams()
	params["id"] = orderID
	params["symbol"] = symbol

	data, err := a.doRequest(ctx, http.MethodPost, "/cancel_order", params)
	if err != nil {
		return trading.Order{}, err
	}

	return a.convertOrder(data), nil
}

// CancelAllOrders cancels all orders.
func (a *CcxtAdapter) CancelAllOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	params := a.exchangeParams()
	if symbol != "" {
		params["symbol"] = symbol
	}

	data, err := a.doRequestArray(ctx, http.MethodPost, "/cancel_all_orders", params)
	if err != nil {
		// Fallback: cancel one by one
		openOrders, err := a.GetOpenOrders(ctx, symbol)
		if err != nil {
			return nil, err
		}

		var cancelled []trading.Order
		for _, order := range openOrders {
			if o, err := a.CancelOrder(ctx, order.OrderID, order.Symbol); err == nil {
				cancelled = append(cancelled, o)
			}
		}
		return cancelled, nil
	}

	orders := make([]trading.Order, len(data))
	for i, o := range data {
		orders[i] = a.convertOrder(o)
	}

	return orders, nil
}

func (a *CcxtAdapter) convertOrder(o map[string]interface{}) trading.Order {
	quantity := decimal.NewFromFloat(getFloat(o, "amount"))
	filled := decimal.NewFromFloat(getFloat(o, "filled"))

	side := trading.SideBuy
	if getString(o, "side") == "sell" {
		side = trading.SideSell
	}

	orderType := trading.OrderTypeLimit
	switch getString(o, "type") {
	case "market":
		orderType = trading.OrderTypeMarket
	case "stop":
		orderType = trading.OrderTypeStopLoss
	case "stop_limit":
		orderType = trading.OrderTypeStopLossLimit
	}

	status := trading.OrderStatusOpen
	switch getString(o, "status") {
	case "closed":
		status = trading.OrderStatusFilled
	case "canceled", "cancelled":
		status = trading.OrderStatusCancelled
	case "expired":
		status = trading.OrderStatusExpired
	case "rejected":
		status = trading.OrderStatusRejected
	}

	order := trading.Order{
		OrderID:           getString(o, "id"),
		ClientOrderID:     getString(o, "clientOrderId"),
		Symbol:            getString(o, "symbol"),
		Venue:             a.name,
		Side:              side,
		OrderType:         orderType,
		Status:            status,
		Quantity:          quantity,
		FilledQuantity:    filled,
		RemainingQuantity: quantity.Sub(filled),
		CreatedAt:         time.UnixMilli(int64(getFloat(o, "timestamp"))),
		UpdatedAt:         time.UnixMilli(int64(getFloat(o, "lastTradeTimestamp"))),
	}

	if price := getFloat(o, "price"); price > 0 {
		d := decimal.NewFromFloat(price)
		order.Price = &d
	}
	if avg := getFloat(o, "average"); avg > 0 {
		d := decimal.NewFromFloat(avg)
		order.AveragePrice = &d
	}

	return order
}

// Helper functions for type-safe map access
func getMap(m map[string]interface{}, key string) map[string]interface{} {
	if m == nil {
		return nil
	}
	if v, ok := m[key].(map[string]interface{}); ok {
		return v
	}
	return nil
}

func getString(m map[string]interface{}, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getFloat(m map[string]interface{}, key string) float64 {
	if m == nil {
		return 0
	}
	return toFloat(m[key])
}

func getInt(m map[string]interface{}, key string) int {
	if m == nil {
		return 0
	}
	return int(toFloat(m[key]))
}

func toFloat(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case string:
		d, _ := decimal.NewFromString(n)
		f, _ := d.Float64()
		return f
	}
	return 0
}
