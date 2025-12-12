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

// LxDexAdapter implements VenueAdapter for LX DEX (CLOB).
type LxDexAdapter struct {
	trading.BaseAdapter
	name   string
	config *trading.NativeVenueConfig
	client *http.Client
}

// NewLxDexAdapter creates a new LX DEX adapter.
func NewLxDexAdapter(name string, config *trading.NativeVenueConfig) *LxDexAdapter {
	return &LxDexAdapter{
		name:   name,
		config: config,
	}
}

// Name returns the venue name.
func (a *LxDexAdapter) Name() string {
	return a.name
}

// VenueType returns the venue type.
func (a *LxDexAdapter) VenueType() trading.VenueType {
	return trading.VenueTypeNative
}

// Capabilities returns venue capabilities.
func (a *LxDexAdapter) Capabilities() trading.VenueCapabilities {
	return trading.CLOBCapabilities()
}

// Connect establishes connection to the DEX.
func (a *LxDexAdapter) Connect(ctx context.Context) error {
	a.client = &http.Client{Timeout: 30 * time.Second}

	// Test connection
	start := time.Now()
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/health", nil)
	if err != nil {
		return fmt.Errorf("connection failed: %w", err)
	}
	resp.Body.Close()

	latency := time.Since(start).Milliseconds()
	a.SetLatency(latency)
	a.SetConnected(true)
	return nil
}

// Disconnect closes the connection.
func (a *LxDexAdapter) Disconnect(ctx context.Context) error {
	a.client = nil
	a.SetConnected(false)
	return nil
}

func (a *LxDexAdapter) doRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	if a.client == nil {
		return nil, trading.ErrNotConnected
	}

	url := a.config.APIURL + path

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
	if a.config.APIKey != "" {
		req.Header.Set("X-API-KEY", a.config.APIKey)
		req.Header.Set("X-TIMESTAMP", fmt.Sprintf("%d", time.Now().UnixMilli()))
	}

	start := time.Now()
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	a.SetLatency(time.Since(start).Milliseconds())

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, trading.NewTradingError("HTTP_ERROR", fmt.Sprintf("status %d: %s", resp.StatusCode, string(body)))
	}

	return resp, nil
}

func (a *LxDexAdapter) decodeJSON(resp *http.Response, v interface{}) error {
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

// GetMarkets returns available markets.
func (a *LxDexAdapter) GetMarkets(ctx context.Context) ([]trading.MarketInfo, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/markets", nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		Symbol            string `json:"symbol"`
		Base              string `json:"base"`
		Quote             string `json:"quote"`
		PricePrecision    int    `json:"pricePrecision"`
		QuantityPrecision int    `json:"quantityPrecision"`
		MinQuantity       string `json:"minQuantity"`
		MaxQuantity       string `json:"maxQuantity"`
		MinNotional       string `json:"minNotional"`
		TickSize          string `json:"tickSize"`
		LotSize           string `json:"lotSize"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	markets := make([]trading.MarketInfo, len(data))
	for i, m := range data {
		minQty, _ := decimal.NewFromString(m.MinQuantity)
		maxQty, _ := decimal.NewFromString(m.MaxQuantity)
		minNot, _ := decimal.NewFromString(m.MinNotional)
		tickSize, _ := decimal.NewFromString(m.TickSize)
		lotSize, _ := decimal.NewFromString(m.LotSize)

		var maxQtyPtr, minNotPtr *decimal.Decimal
		if !maxQty.IsZero() {
			maxQtyPtr = &maxQty
		}
		if !minNot.IsZero() {
			minNotPtr = &minNot
		}

		markets[i] = trading.MarketInfo{
			Symbol:            m.Symbol,
			Base:              m.Base,
			Quote:             m.Quote,
			PricePrecision:    m.PricePrecision,
			QuantityPrecision: m.QuantityPrecision,
			MinQuantity:       minQty,
			MaxQuantity:       maxQtyPtr,
			MinNotional:       minNotPtr,
			TickSize:          tickSize,
			LotSize:           lotSize,
		}
	}

	return markets, nil
}

// GetTicker returns ticker for a symbol.
func (a *LxDexAdapter) GetTicker(ctx context.Context, symbol string) (trading.Ticker, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/ticker/"+symbol, nil)
	if err != nil {
		return trading.Ticker{}, err
	}

	var data struct {
		Symbol    string `json:"symbol"`
		Bid       string `json:"bid"`
		Ask       string `json:"ask"`
		Last      string `json:"last"`
		Volume24H string `json:"volume24h"`
		High24H   string `json:"high24h"`
		Low24H    string `json:"low24h"`
		Change24H string `json:"change24h"`
		Timestamp int64  `json:"timestamp"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Ticker{}, err
	}

	ticker := trading.Ticker{
		Symbol:    data.Symbol,
		Venue:     a.Name(),
		Timestamp: time.UnixMilli(data.Timestamp),
	}

	if data.Bid != "" {
		bid, _ := decimal.NewFromString(data.Bid)
		ticker.Bid = &bid
	}
	if data.Ask != "" {
		ask, _ := decimal.NewFromString(data.Ask)
		ticker.Ask = &ask
	}
	if data.Last != "" {
		last, _ := decimal.NewFromString(data.Last)
		ticker.Last = &last
	}
	if data.Volume24H != "" {
		vol, _ := decimal.NewFromString(data.Volume24H)
		ticker.Volume24H = &vol
	}
	if data.High24H != "" {
		high, _ := decimal.NewFromString(data.High24H)
		ticker.High24H = &high
	}
	if data.Low24H != "" {
		low, _ := decimal.NewFromString(data.Low24H)
		ticker.Low24H = &low
	}
	if data.Change24H != "" {
		change, _ := decimal.NewFromString(data.Change24H)
		ticker.Change24H = &change
	}

	return ticker, nil
}

// GetOrderbook returns the orderbook for a symbol.
func (a *LxDexAdapter) GetOrderbook(ctx context.Context, symbol string, depth int) (*trading.Orderbook, error) {
	path := "/api/v1/orderbook/" + symbol
	if depth > 0 {
		path += fmt.Sprintf("?depth=%d", depth)
	}

	resp, err := a.doRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	var data struct {
		Bids [][2]string `json:"bids"`
		Asks [][2]string `json:"asks"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	book := trading.NewOrderbook(symbol, a.Name())

	for _, bid := range data.Bids {
		price, _ := decimal.NewFromString(bid[0])
		qty, _ := decimal.NewFromString(bid[1])
		book.AddBid(price, qty)
	}

	for _, ask := range data.Asks {
		price, _ := decimal.NewFromString(ask[0])
		qty, _ := decimal.NewFromString(ask[1])
		book.AddAsk(price, qty)
	}

	book.Sort()
	return book, nil
}

// GetTrades returns recent trades.
func (a *LxDexAdapter) GetTrades(ctx context.Context, symbol string, limit int) ([]trading.Trade, error) {
	path := "/api/v1/trades/" + symbol
	if limit > 0 {
		path += fmt.Sprintf("?limit=%d", limit)
	}

	resp, err := a.doRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		ID        string `json:"id"`
		OrderID   string `json:"orderId"`
		Side      string `json:"side"`
		Price     string `json:"price"`
		Quantity  string `json:"quantity"`
		FeeAsset  string `json:"feeAsset"`
		FeeAmount string `json:"feeAmount"`
		Timestamp int64  `json:"timestamp"`
		IsMaker   bool   `json:"isMaker"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	trades := make([]trading.Trade, len(data))
	for i, t := range data {
		price, _ := decimal.NewFromString(t.Price)
		qty, _ := decimal.NewFromString(t.Quantity)
		feeAmt, _ := decimal.NewFromString(t.FeeAmount)

		side := trading.SideBuy
		if t.Side == "sell" {
			side = trading.SideSell
		}

		trades[i] = trading.Trade{
			TradeID:   t.ID,
			OrderID:   t.OrderID,
			Symbol:    symbol,
			Venue:     a.Name(),
			Side:      side,
			Price:     price,
			Quantity:  qty,
			Fee:       trading.Fee{Asset: t.FeeAsset, Amount: feeAmt},
			Timestamp: time.UnixMilli(t.Timestamp),
			IsMaker:   t.IsMaker,
		}
	}

	return trades, nil
}

// GetBalances returns all balances.
func (a *LxDexAdapter) GetBalances(ctx context.Context) ([]trading.Balance, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/account/balances", nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		Asset  string `json:"asset"`
		Free   string `json:"free"`
		Locked string `json:"locked"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	balances := make([]trading.Balance, len(data))
	for i, b := range data {
		free, _ := decimal.NewFromString(b.Free)
		locked, _ := decimal.NewFromString(b.Locked)
		balances[i] = trading.Balance{
			Asset:  b.Asset,
			Venue:  a.Name(),
			Free:   free,
			Locked: locked,
		}
	}

	return balances, nil
}

// GetBalance returns balance for a specific asset.
func (a *LxDexAdapter) GetBalance(ctx context.Context, asset string) (trading.Balance, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/account/balance/"+asset, nil)
	if err != nil {
		return trading.Balance{}, err
	}

	var data struct {
		Asset  string `json:"asset"`
		Free   string `json:"free"`
		Locked string `json:"locked"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Balance{}, err
	}

	free, _ := decimal.NewFromString(data.Free)
	locked, _ := decimal.NewFromString(data.Locked)

	return trading.Balance{
		Asset:  data.Asset,
		Venue:  a.Name(),
		Free:   free,
		Locked: locked,
	}, nil
}

// GetOpenOrders returns open orders.
func (a *LxDexAdapter) GetOpenOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	path := "/api/v1/orders?status=open"
	if symbol != "" {
		path += "&symbol=" + symbol
	}

	resp, err := a.doRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	var data []orderResponse
	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	orders := make([]trading.Order, len(data))
	for i, o := range data {
		orders[i] = a.convertOrder(o)
	}

	return orders, nil
}

// PlaceOrder places a new order.
func (a *LxDexAdapter) PlaceOrder(ctx context.Context, request trading.OrderRequest) (trading.Order, error) {
	body := map[string]interface{}{
		"clientOrderId": request.ClientOrderID,
		"symbol":        request.Symbol,
		"side":          string(request.Side),
		"type":          string(request.OrderType),
		"quantity":      request.Quantity.String(),
		"timeInForce":   string(request.TimeInForce),
	}

	if request.Price != nil {
		body["price"] = request.Price.String()
	}

	resp, err := a.doRequest(ctx, http.MethodPost, "/api/v1/orders", body)
	if err != nil {
		return trading.Order{}, err
	}

	var data orderResponse
	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Order{}, err
	}

	return a.convertOrder(data), nil
}

// CancelOrder cancels an order.
func (a *LxDexAdapter) CancelOrder(ctx context.Context, orderID, symbol string) (trading.Order, error) {
	body := map[string]interface{}{"symbol": symbol}

	resp, err := a.doRequest(ctx, http.MethodDelete, "/api/v1/orders/"+orderID, body)
	if err != nil {
		return trading.Order{}, err
	}

	var data orderResponse
	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Order{}, err
	}

	return a.convertOrder(data), nil
}

// CancelAllOrders cancels all orders.
func (a *LxDexAdapter) CancelAllOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	body := map[string]interface{}{}
	if symbol != "" {
		body["symbol"] = symbol
	}

	resp, err := a.doRequest(ctx, http.MethodDelete, "/api/v1/orders/all", body)
	if err != nil {
		return nil, err
	}

	var data []orderResponse
	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	orders := make([]trading.Order, len(data))
	for i, o := range data {
		orders[i] = a.convertOrder(o)
	}

	return orders, nil
}

type orderResponse struct {
	OrderID        string `json:"orderId"`
	ClientOrderID  string `json:"clientOrderId"`
	Symbol         string `json:"symbol"`
	Side           string `json:"side"`
	Type           string `json:"type"`
	Status         string `json:"status"`
	Quantity       string `json:"quantity"`
	FilledQuantity string `json:"filledQuantity"`
	Price          string `json:"price"`
	AveragePrice   string `json:"averagePrice"`
	CreatedAt      int64  `json:"createdAt"`
	UpdatedAt      int64  `json:"updatedAt"`
}

func (a *LxDexAdapter) convertOrder(o orderResponse) trading.Order {
	quantity, _ := decimal.NewFromString(o.Quantity)
	filled, _ := decimal.NewFromString(o.FilledQuantity)

	side := trading.SideBuy
	if o.Side == "sell" {
		side = trading.SideSell
	}

	orderType := trading.OrderTypeLimit
	switch o.Type {
	case "market":
		orderType = trading.OrderTypeMarket
	case "stop_loss":
		orderType = trading.OrderTypeStopLoss
	case "stop_loss_limit":
		orderType = trading.OrderTypeStopLossLimit
	}

	status := trading.OrderStatusOpen
	switch o.Status {
	case "pending":
		status = trading.OrderStatusPending
	case "partially_filled":
		status = trading.OrderStatusPartiallyFilled
	case "filled":
		status = trading.OrderStatusFilled
	case "cancelled":
		status = trading.OrderStatusCancelled
	case "rejected":
		status = trading.OrderStatusRejected
	case "expired":
		status = trading.OrderStatusExpired
	}

	order := trading.Order{
		OrderID:           o.OrderID,
		ClientOrderID:     o.ClientOrderID,
		Symbol:            o.Symbol,
		Venue:             a.Name(),
		Side:              side,
		OrderType:         orderType,
		Status:            status,
		Quantity:          quantity,
		FilledQuantity:    filled,
		RemainingQuantity: quantity.Sub(filled),
		CreatedAt:         time.UnixMilli(o.CreatedAt),
		UpdatedAt:         time.UnixMilli(o.UpdatedAt),
	}

	if o.Price != "" {
		price, _ := decimal.NewFromString(o.Price)
		order.Price = &price
	}
	if o.AveragePrice != "" {
		avgPrice, _ := decimal.NewFromString(o.AveragePrice)
		order.AveragePrice = &avgPrice
	}

	return order
}

// =============================================================================
// LxAmmAdapter - AMM adapter
// =============================================================================

// LxAmmAdapter implements VenueAdapter for LX AMM.
type LxAmmAdapter struct {
	trading.BaseAdapter
	config *trading.NativeVenueConfig
	client *http.Client
	name   string
}

// NewLxAmmAdapter creates a new LX AMM adapter.
func NewLxAmmAdapter(name string, config *trading.NativeVenueConfig) *LxAmmAdapter {
	return &LxAmmAdapter{
		name:   name,
		config: config,
	}
}

// Name returns the venue name.
func (a *LxAmmAdapter) Name() string {
	return a.name
}

// VenueType returns the venue type.
func (a *LxAmmAdapter) VenueType() trading.VenueType {
	return trading.VenueTypeNative
}

// Capabilities returns venue capabilities.
func (a *LxAmmAdapter) Capabilities() trading.VenueCapabilities {
	return trading.AMMCapabilities()
}

// Connect establishes connection to the AMM.
func (a *LxAmmAdapter) Connect(ctx context.Context) error {
	a.client = &http.Client{Timeout: 30 * time.Second}
	a.SetConnected(true)
	return nil
}

// Disconnect closes the connection.
func (a *LxAmmAdapter) Disconnect(ctx context.Context) error {
	a.client = nil
	a.SetConnected(false)
	return nil
}

func (a *LxAmmAdapter) doRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	if a.client == nil {
		return nil, trading.ErrNotConnected
	}

	url := a.config.APIURL + path

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
	a.SetLatency(time.Since(start).Milliseconds())

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, trading.NewTradingError("HTTP_ERROR", fmt.Sprintf("status %d: %s", resp.StatusCode, string(body)))
	}

	return resp, nil
}

func (a *LxAmmAdapter) decodeJSON(resp *http.Response, v interface{}) error {
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

// GetMarkets returns available pools as markets.
func (a *LxAmmAdapter) GetMarkets(ctx context.Context) ([]trading.MarketInfo, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/amm/pools", nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		BaseToken  string `json:"baseToken"`
		QuoteToken string `json:"quoteToken"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	markets := make([]trading.MarketInfo, len(data))
	for i, p := range data {
		markets[i] = trading.MarketInfo{
			Symbol:            p.BaseToken + "-" + p.QuoteToken,
			Base:              p.BaseToken,
			Quote:             p.QuoteToken,
			PricePrecision:    8,
			QuantityPrecision: 8,
			MinQuantity:       decimal.Zero,
			TickSize:          decimal.NewFromFloat(0.00000001),
			LotSize:           decimal.NewFromFloat(0.00000001),
		}
	}

	return markets, nil
}

// GetTicker returns price ticker from AMM.
func (a *LxAmmAdapter) GetTicker(ctx context.Context, symbol string) (trading.Ticker, error) {
	pair := trading.ParseTradingPair(symbol)

	resp, err := a.doRequest(ctx, http.MethodGet, fmt.Sprintf("/api/v1/amm/price/%s/%s", pair.Base, pair.Quote), nil)
	if err != nil {
		return trading.Ticker{}, err
	}

	var data struct {
		Price     string `json:"price"`
		Volume24H string `json:"volume24h"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Ticker{}, err
	}

	price, _ := decimal.NewFromString(data.Price)
	ticker := trading.Ticker{
		Symbol:    symbol,
		Venue:     a.name,
		Bid:       &price,
		Ask:       &price,
		Last:      &price,
		Timestamp: time.Now(),
	}

	if data.Volume24H != "" {
		vol, _ := decimal.NewFromString(data.Volume24H)
		ticker.Volume24H = &vol
	}

	return ticker, nil
}

// GetOrderbook is not supported for AMM.
func (a *LxAmmAdapter) GetOrderbook(ctx context.Context, symbol string, depth int) (*trading.Orderbook, error) {
	return nil, trading.ErrNotSupported
}

// GetTrades returns recent swaps.
func (a *LxAmmAdapter) GetTrades(ctx context.Context, symbol string, limit int) ([]trading.Trade, error) {
	pair := trading.ParseTradingPair(symbol)

	path := fmt.Sprintf("/api/v1/amm/swaps/%s/%s", pair.Base, pair.Quote)
	if limit > 0 {
		path += fmt.Sprintf("?limit=%d", limit)
	}

	resp, err := a.doRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		TxHash    string `json:"txHash"`
		Side      string `json:"side"`
		Price     string `json:"price"`
		Amount    string `json:"amount"`
		Fee       string `json:"fee"`
		Timestamp int64  `json:"timestamp"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	trades := make([]trading.Trade, len(data))
	for i, t := range data {
		price, _ := decimal.NewFromString(t.Price)
		amount, _ := decimal.NewFromString(t.Amount)
		fee, _ := decimal.NewFromString(t.Fee)

		side := trading.SideBuy
		if t.Side == "sell" {
			side = trading.SideSell
		}

		trades[i] = trading.Trade{
			TradeID:   t.TxHash,
			OrderID:   t.TxHash,
			Symbol:    symbol,
			Venue:     a.name,
			Side:      side,
			Price:     price,
			Quantity:  amount,
			Fee:       trading.Fee{Amount: fee},
			Timestamp: time.UnixMilli(t.Timestamp),
			IsMaker:   false,
		}
	}

	return trades, nil
}

// GetBalances returns all balances.
func (a *LxAmmAdapter) GetBalances(ctx context.Context) ([]trading.Balance, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/account/balances", nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		Asset  string `json:"asset"`
		Free   string `json:"free"`
		Locked string `json:"locked"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	balances := make([]trading.Balance, len(data))
	for i, b := range data {
		free, _ := decimal.NewFromString(b.Free)
		locked, _ := decimal.NewFromString(b.Locked)
		balances[i] = trading.Balance{
			Asset:  b.Asset,
			Venue:  a.name,
			Free:   free,
			Locked: locked,
		}
	}

	return balances, nil
}

// GetBalance returns balance for a specific asset.
func (a *LxAmmAdapter) GetBalance(ctx context.Context, asset string) (trading.Balance, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/account/balance/"+asset, nil)
	if err != nil {
		return trading.Balance{}, err
	}

	var data struct {
		Asset  string `json:"asset"`
		Free   string `json:"free"`
		Locked string `json:"locked"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Balance{}, err
	}

	free, _ := decimal.NewFromString(data.Free)
	locked, _ := decimal.NewFromString(data.Locked)

	return trading.Balance{
		Asset:  data.Asset,
		Venue:  a.name,
		Free:   free,
		Locked: locked,
	}, nil
}

// GetOpenOrders returns empty list for AMM.
func (a *LxAmmAdapter) GetOpenOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	return []trading.Order{}, nil
}

// PlaceOrder executes a swap as an order.
func (a *LxAmmAdapter) PlaceOrder(ctx context.Context, request trading.OrderRequest) (trading.Order, error) {
	pair := trading.ParseTradingPair(request.Symbol)

	trade, err := a.ExecuteSwap(ctx, pair.Base, pair.Quote, request.Quantity, request.Side == trading.SideBuy, decimal.NewFromFloat(0.01))
	if err != nil {
		return trading.Order{}, err
	}

	return trading.Order{
		OrderID:           trade.TradeID,
		ClientOrderID:     request.ClientOrderID,
		Symbol:            request.Symbol,
		Venue:             a.name,
		Side:              request.Side,
		OrderType:         trading.OrderTypeMarket,
		Status:            trading.OrderStatusFilled,
		Quantity:          request.Quantity,
		FilledQuantity:    trade.Quantity,
		RemainingQuantity: decimal.Zero,
		Price:             &trade.Price,
		AveragePrice:      &trade.Price,
		CreatedAt:         trade.Timestamp,
		UpdatedAt:         trade.Timestamp,
		Fees:              []trading.Fee{trade.Fee},
	}, nil
}

// CancelOrder is not supported for AMM.
func (a *LxAmmAdapter) CancelOrder(ctx context.Context, orderID, symbol string) (trading.Order, error) {
	return trading.Order{}, trading.ErrNotSupported
}

// CancelAllOrders is not supported for AMM.
func (a *LxAmmAdapter) CancelAllOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	return []trading.Order{}, nil
}

// GetSwapQuote returns a swap quote.
func (a *LxAmmAdapter) GetSwapQuote(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool) (trading.SwapQuote, error) {
	side := "sell"
	if isBuy {
		side = "buy"
	}

	resp, err := a.doRequest(ctx, http.MethodPost, "/api/v1/amm/quote", map[string]interface{}{
		"baseToken":  baseToken,
		"quoteToken": quoteToken,
		"amount":     amount.String(),
		"side":       side,
	})
	if err != nil {
		return trading.SwapQuote{}, err
	}

	var data struct {
		OutputAmount string   `json:"outputAmount"`
		Price        string   `json:"price"`
		PriceImpact  string   `json:"priceImpact"`
		Fee          string   `json:"fee"`
		Route        []string `json:"route"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.SwapQuote{}, err
	}

	output, _ := decimal.NewFromString(data.OutputAmount)
	price, _ := decimal.NewFromString(data.Price)
	impact, _ := decimal.NewFromString(data.PriceImpact)
	fee, _ := decimal.NewFromString(data.Fee)

	return trading.SwapQuote{
		BaseToken:    baseToken,
		QuoteToken:   quoteToken,
		InputAmount:  amount,
		OutputAmount: output,
		Price:        price,
		PriceImpact:  impact,
		Fee:          fee,
		Route:        data.Route,
		ExpiresAt:    time.Now().Add(60 * time.Second),
	}, nil
}

// ExecuteSwap executes a swap.
func (a *LxAmmAdapter) ExecuteSwap(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool, slippage decimal.Decimal) (trading.Trade, error) {
	side := "sell"
	if isBuy {
		side = "buy"
	}

	resp, err := a.doRequest(ctx, http.MethodPost, "/api/v1/amm/swap", map[string]interface{}{
		"baseToken":  baseToken,
		"quoteToken": quoteToken,
		"amount":     amount.String(),
		"side":       side,
		"slippage":   slippage.String(),
	})
	if err != nil {
		return trading.Trade{}, err
	}

	var data struct {
		TxHash string `json:"txHash"`
		Price  string `json:"price"`
		Fee    string `json:"fee"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.Trade{}, err
	}

	price, _ := decimal.NewFromString(data.Price)
	fee, _ := decimal.NewFromString(data.Fee)

	tradeSide := trading.SideSell
	if isBuy {
		tradeSide = trading.SideBuy
	}

	return trading.Trade{
		TradeID:   data.TxHash,
		OrderID:   data.TxHash,
		Symbol:    baseToken + "-" + quoteToken,
		Venue:     a.name,
		Side:      tradeSide,
		Price:     price,
		Quantity:  amount,
		Fee:       trading.Fee{Amount: fee},
		Timestamp: time.Now(),
		IsMaker:   false,
	}, nil
}

// GetPoolInfo returns pool information.
func (a *LxAmmAdapter) GetPoolInfo(ctx context.Context, baseToken, quoteToken string) (trading.PoolInfo, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, fmt.Sprintf("/api/v1/amm/pool/%s/%s", baseToken, quoteToken), nil)
	if err != nil {
		return trading.PoolInfo{}, err
	}

	var data struct {
		Address        string `json:"address"`
		BaseReserve    string `json:"baseReserve"`
		QuoteReserve   string `json:"quoteReserve"`
		TotalLiquidity string `json:"totalLiquidity"`
		FeeRate        string `json:"feeRate"`
		APY            string `json:"apy"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.PoolInfo{}, err
	}

	baseRes, _ := decimal.NewFromString(data.BaseReserve)
	quoteRes, _ := decimal.NewFromString(data.QuoteReserve)
	totalLiq, _ := decimal.NewFromString(data.TotalLiquidity)
	feeRate, _ := decimal.NewFromString(data.FeeRate)

	pool := trading.PoolInfo{
		Address:        data.Address,
		BaseToken:      baseToken,
		QuoteToken:     quoteToken,
		BaseReserve:    baseRes,
		QuoteReserve:   quoteRes,
		TotalLiquidity: totalLiq,
		FeeRate:        feeRate,
	}

	if data.APY != "" {
		apy, _ := decimal.NewFromString(data.APY)
		pool.APY = &apy
	}

	return pool, nil
}

// AddLiquidity adds liquidity to a pool.
func (a *LxAmmAdapter) AddLiquidity(ctx context.Context, baseToken, quoteToken string, baseAmount, quoteAmount, slippage decimal.Decimal) (trading.LiquidityResult, error) {
	resp, err := a.doRequest(ctx, http.MethodPost, "/api/v1/amm/liquidity/add", map[string]interface{}{
		"baseToken":   baseToken,
		"quoteToken":  quoteToken,
		"baseAmount":  baseAmount.String(),
		"quoteAmount": quoteAmount.String(),
		"slippage":    slippage.String(),
	})
	if err != nil {
		return trading.LiquidityResult{}, err
	}

	var data struct {
		TxHash       string `json:"txHash"`
		PoolAddress  string `json:"poolAddress"`
		LpTokens     string `json:"lpTokens"`
		SharePercent string `json:"sharePercent"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.LiquidityResult{}, err
	}

	lpTokens, _ := decimal.NewFromString(data.LpTokens)
	share, _ := decimal.NewFromString(data.SharePercent)

	return trading.LiquidityResult{
		TxHash:       data.TxHash,
		PoolAddress:  data.PoolAddress,
		BaseAmount:   baseAmount,
		QuoteAmount:  quoteAmount,
		LpTokens:     lpTokens,
		SharePercent: share,
	}, nil
}

// RemoveLiquidity removes liquidity from a pool.
func (a *LxAmmAdapter) RemoveLiquidity(ctx context.Context, poolAddress string, liquidityAmount, slippage decimal.Decimal) (trading.LiquidityResult, error) {
	resp, err := a.doRequest(ctx, http.MethodPost, "/api/v1/amm/liquidity/remove", map[string]interface{}{
		"poolAddress": poolAddress,
		"liquidity":   liquidityAmount.String(),
		"slippage":    slippage.String(),
	})
	if err != nil {
		return trading.LiquidityResult{}, err
	}

	var data struct {
		TxHash      string `json:"txHash"`
		BaseAmount  string `json:"baseAmount"`
		QuoteAmount string `json:"quoteAmount"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return trading.LiquidityResult{}, err
	}

	baseAmt, _ := decimal.NewFromString(data.BaseAmount)
	quoteAmt, _ := decimal.NewFromString(data.QuoteAmount)

	return trading.LiquidityResult{
		TxHash:       data.TxHash,
		PoolAddress:  poolAddress,
		BaseAmount:   baseAmt,
		QuoteAmount:  quoteAmt,
		LpTokens:     liquidityAmount,
		SharePercent: decimal.Zero,
	}, nil
}

// GetLpPositions returns LP positions.
func (a *LxAmmAdapter) GetLpPositions(ctx context.Context) ([]trading.LpPosition, error) {
	resp, err := a.doRequest(ctx, http.MethodGet, "/api/v1/amm/positions", nil)
	if err != nil {
		return nil, err
	}

	var data []struct {
		PoolAddress   string `json:"poolAddress"`
		BaseToken     string `json:"baseToken"`
		QuoteToken    string `json:"quoteToken"`
		LpTokens      string `json:"lpTokens"`
		BaseAmount    string `json:"baseAmount"`
		QuoteAmount   string `json:"quoteAmount"`
		SharePercent  string `json:"sharePercent"`
		UnrealizedPnL string `json:"unrealizedPnl"`
	}

	if err := a.decodeJSON(resp, &data); err != nil {
		return nil, err
	}

	positions := make([]trading.LpPosition, len(data))
	for i, p := range data {
		lpTokens, _ := decimal.NewFromString(p.LpTokens)
		baseAmt, _ := decimal.NewFromString(p.BaseAmount)
		quoteAmt, _ := decimal.NewFromString(p.QuoteAmount)
		share, _ := decimal.NewFromString(p.SharePercent)

		positions[i] = trading.LpPosition{
			PoolAddress:  p.PoolAddress,
			BaseToken:    p.BaseToken,
			QuoteToken:   p.QuoteToken,
			LpTokens:     lpTokens,
			BaseAmount:   baseAmt,
			QuoteAmount:  quoteAmt,
			SharePercent: share,
		}

		if p.UnrealizedPnL != "" {
			pnl, _ := decimal.NewFromString(p.UnrealizedPnL)
			positions[i].UnrealizedPnL = &pnl
		}
	}

	return positions, nil
}
