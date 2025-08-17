package client

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/dex/backend/pkg/lx"
)

// TraderClient is the main client for traders to interact with LX DEX
type TraderClient struct {
	// Connection
	wsConn          *websocket.Conn
	apiEndpoint     string
	wsEndpoint      string
	apiKey          string
	apiSecret       string
	
	// State
	connected       bool
	authenticated   bool
	userID          string
	
	// Account information
	marginAccount   *lx.MarginAccount
	positions       map[string]*lx.MarginPosition
	orders          map[uint64]*lx.Order
	balances        map[string]*big.Int
	
	// Market data
	orderBooks      map[string]*lx.OrderBookSnapshot
	prices          map[string]float64
	trades          map[string][]*lx.Trade
	
	// Subscriptions
	subscriptions   map[string]bool
	callbacks       map[string]func(interface{})
	
	// Channels
	orderUpdates    chan *OrderUpdate
	positionUpdates chan *PositionUpdate
	priceUpdates    chan *PriceUpdate
	tradeUpdates    chan *TradeUpdate
	errorChan       chan error
	
	// Control
	ctx             context.Context
	cancel          context.CancelFunc
	mu              sync.RWMutex
}

// OrderUpdate represents an order update event
type OrderUpdate struct {
	Order     *lx.Order
	Status    lx.OrderStatus
	Timestamp time.Time
	Message   string
}

// PositionUpdate represents a position update event
type PositionUpdate struct {
	Position  *lx.MarginPosition
	Action    string // "opened", "modified", "closed", "liquidated"
	Timestamp time.Time
}

// PriceUpdate represents a price update event
type PriceUpdate struct {
	Symbol    string
	Price     float64
	Bid       float64
	Ask       float64
	Volume    float64
	Timestamp time.Time
}

// TradeUpdate represents a trade execution event
type TradeUpdate struct {
	Trade     *lx.Trade
	Symbol    string
	Side      string
	Timestamp time.Time
}

// ClientConfig contains configuration for the trader client
type ClientConfig struct {
	APIEndpoint string
	WSEndpoint  string
	APIKey      string
	APISecret   string
	UserID      string
}

// NewTraderClient creates a new trader client
func NewTraderClient(config ClientConfig) (*TraderClient, error) {
	if config.APIEndpoint == "" || config.WSEndpoint == "" {
		return nil, errors.New("API and WebSocket endpoints required")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	client := &TraderClient{
		apiEndpoint:     config.APIEndpoint,
		wsEndpoint:      config.WSEndpoint,
		apiKey:          config.APIKey,
		apiSecret:       config.APISecret,
		userID:          config.UserID,
		positions:       make(map[string]*lx.MarginPosition),
		orders:          make(map[uint64]*lx.Order),
		balances:        make(map[string]*big.Int),
		orderBooks:      make(map[string]*lx.OrderBookSnapshot),
		prices:          make(map[string]float64),
		trades:          make(map[string][]*lx.Trade),
		subscriptions:   make(map[string]bool),
		callbacks:       make(map[string]func(interface{})),
		orderUpdates:    make(chan *OrderUpdate, 100),
		positionUpdates: make(chan *PositionUpdate, 100),
		priceUpdates:    make(chan *PriceUpdate, 1000),
		tradeUpdates:    make(chan *TradeUpdate, 1000),
		errorChan:       make(chan error, 10),
		ctx:             ctx,
		cancel:          cancel,
	}
	
	return client, nil
}

// Connect establishes connection to the DEX
func (c *TraderClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.connected {
		return errors.New("already connected")
	}
	
	// Connect to WebSocket
	conn, _, err := websocket.DefaultDialer.Dial(c.wsEndpoint, nil)
	if err != nil {
		return fmt.Errorf("websocket connection failed: %w", err)
	}
	
	c.wsConn = conn
	c.connected = true
	
	// Start message handlers
	go c.handleMessages()
	go c.heartbeat()
	
	// Authenticate
	if err := c.authenticate(); err != nil {
		c.Disconnect()
		return fmt.Errorf("authentication failed: %w", err)
	}
	
	return nil
}

// Disconnect closes the connection
func (c *TraderClient) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if !c.connected {
		return nil
	}
	
	c.cancel()
	
	if c.wsConn != nil {
		c.wsConn.Close()
	}
	
	c.connected = false
	c.authenticated = false
	
	// Close channels
	close(c.orderUpdates)
	close(c.positionUpdates)
	close(c.priceUpdates)
	close(c.tradeUpdates)
	close(c.errorChan)
	
	return nil
}

// authenticate authenticates with the DEX
func (c *TraderClient) authenticate() error {
	auth := map[string]interface{}{
		"type":      "auth",
		"apiKey":    c.apiKey,
		"apiSecret": c.apiSecret,
		"timestamp": time.Now().Unix(),
	}
	
	if err := c.wsConn.WriteJSON(auth); err != nil {
		return err
	}
	
	// Wait for auth response
	var response map[string]interface{}
	if err := c.wsConn.ReadJSON(&response); err != nil {
		return err
	}
	
	if response["type"] == "auth_success" {
		c.authenticated = true
		return nil
	}
	
	return fmt.Errorf("authentication failed: %v", response["error"])
}

// Trading Methods

// PlaceOrder places a new order
func (c *TraderClient) PlaceOrder(order *lx.Order) (*lx.Order, error) {
	if !c.authenticated {
		return nil, errors.New("not authenticated")
	}
	
	order.User = c.userID
	order.Timestamp = time.Now()
	
	// Send order via WebSocket
	msg := map[string]interface{}{
		"type":  "place_order",
		"order": order,
	}
	
	if err := c.wsConn.WriteJSON(msg); err != nil {
		return nil, err
	}
	
	// Store order locally
	c.mu.Lock()
	c.orders[order.ID] = order
	c.mu.Unlock()
	
	return order, nil
}

// CancelOrder cancels an existing order
func (c *TraderClient) CancelOrder(orderID uint64) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":    "cancel_order",
		"orderID": orderID,
	}
	
	return c.wsConn.WriteJSON(msg)
}

// ModifyOrder modifies an existing order
func (c *TraderClient) ModifyOrder(orderID uint64, newPrice, newSize float64) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":     "modify_order",
		"orderID":  orderID,
		"newPrice": newPrice,
		"newSize":  newSize,
	}
	
	return c.wsConn.WriteJSON(msg)
}

// Margin Trading Methods

// OpenMarginPosition opens a new margin position
func (c *TraderClient) OpenMarginPosition(symbol string, side lx.Side, size, leverage float64) (*lx.MarginPosition, error) {
	if !c.authenticated {
		return nil, errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":     "open_position",
		"symbol":   symbol,
		"side":     side,
		"size":     size,
		"leverage": leverage,
	}
	
	if err := c.wsConn.WriteJSON(msg); err != nil {
		return nil, err
	}
	
	// Wait for position update
	select {
	case update := <-c.positionUpdates:
		if update.Action == "opened" {
			return update.Position, nil
		}
	case <-time.After(5 * time.Second):
		return nil, errors.New("timeout waiting for position")
	}
	
	return nil, errors.New("failed to open position")
}

// ClosePosition closes an existing position
func (c *TraderClient) ClosePosition(positionID string, size float64) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":       "close_position",
		"positionID": positionID,
		"size":       size,
	}
	
	return c.wsConn.WriteJSON(msg)
}

// ModifyLeverage modifies leverage for a position
func (c *TraderClient) ModifyLeverage(positionID string, newLeverage float64) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":        "modify_leverage",
		"positionID":  positionID,
		"newLeverage": newLeverage,
	}
	
	return c.wsConn.WriteJSON(msg)
}

// Vault Operations

// DepositToVault deposits funds to a vault
func (c *TraderClient) DepositToVault(vaultID string, amount *big.Int) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":    "vault_deposit",
		"vaultID": vaultID,
		"amount":  amount.String(),
	}
	
	return c.wsConn.WriteJSON(msg)
}

// WithdrawFromVault withdraws funds from a vault
func (c *TraderClient) WithdrawFromVault(vaultID string, shares *big.Int) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":    "vault_withdraw",
		"vaultID": vaultID,
		"shares":  shares.String(),
	}
	
	return c.wsConn.WriteJSON(msg)
}

// Lending Operations

// Supply supplies assets to lending pool
func (c *TraderClient) Supply(asset string, amount *big.Int) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":   "lending_supply",
		"asset":  asset,
		"amount": amount.String(),
	}
	
	return c.wsConn.WriteJSON(msg)
}

// Borrow borrows from lending pool
func (c *TraderClient) Borrow(asset string, amount *big.Int) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":   "lending_borrow",
		"asset":  asset,
		"amount": amount.String(),
	}
	
	return c.wsConn.WriteJSON(msg)
}

// Repay repays borrowed amount
func (c *TraderClient) Repay(asset string, amount *big.Int) error {
	if !c.authenticated {
		return errors.New("not authenticated")
	}
	
	msg := map[string]interface{}{
		"type":   "lending_repay",
		"asset":  asset,
		"amount": amount.String(),
	}
	
	return c.wsConn.WriteJSON(msg)
}

// Market Data Methods

// Subscribe subscribes to market data
func (c *TraderClient) Subscribe(channel string, symbols []string) error {
	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": channel,
		"symbols": symbols,
	}
	
	if err := c.wsConn.WriteJSON(msg); err != nil {
		return err
	}
	
	// Mark as subscribed
	c.mu.Lock()
	for _, symbol := range symbols {
		key := fmt.Sprintf("%s:%s", channel, symbol)
		c.subscriptions[key] = true
	}
	c.mu.Unlock()
	
	return nil
}

// Unsubscribe unsubscribes from market data
func (c *TraderClient) Unsubscribe(channel string, symbols []string) error {
	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"channel": channel,
		"symbols": symbols,
	}
	
	if err := c.wsConn.WriteJSON(msg); err != nil {
		return err
	}
	
	// Remove subscriptions
	c.mu.Lock()
	for _, symbol := range symbols {
		key := fmt.Sprintf("%s:%s", channel, symbol)
		delete(c.subscriptions, key)
	}
	c.mu.Unlock()
	
	return nil
}

// GetOrderBook returns current order book for a symbol
func (c *TraderClient) GetOrderBook(symbol string) (*lx.OrderBookSnapshot, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if ob, exists := c.orderBooks[symbol]; exists {
		return ob, nil
	}
	
	return nil, fmt.Errorf("no order book for %s", symbol)
}

// GetPrice returns current price for a symbol
func (c *TraderClient) GetPrice(symbol string) (float64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if price, exists := c.prices[symbol]; exists {
		return price, nil
	}
	
	return 0, fmt.Errorf("no price for %s", symbol)
}

// Account Methods

// GetBalance returns balance for an asset
func (c *TraderClient) GetBalance(asset string) (*big.Int, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if balance, exists := c.balances[asset]; exists {
		return balance, nil
	}
	
	return big.NewInt(0), nil
}

// GetPosition returns a specific position
func (c *TraderClient) GetPosition(positionID string) (*lx.MarginPosition, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if position, exists := c.positions[positionID]; exists {
		return position, nil
	}
	
	return nil, fmt.Errorf("position %s not found", positionID)
}

// GetPositions returns all positions
func (c *TraderClient) GetPositions() map[string]*lx.MarginPosition {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	positions := make(map[string]*lx.MarginPosition)
	for k, v := range c.positions {
		positions[k] = v
	}
	
	return positions
}

// GetOrders returns all orders
func (c *TraderClient) GetOrders() map[uint64]*lx.Order {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	orders := make(map[uint64]*lx.Order)
	for k, v := range c.orders {
		orders[k] = v
	}
	
	return orders
}

// Callback Registration

// OnOrderUpdate registers callback for order updates
func (c *TraderClient) OnOrderUpdate(callback func(*OrderUpdate)) {
	c.callbacks["order_update"] = func(data interface{}) {
		if update, ok := data.(*OrderUpdate); ok {
			callback(update)
		}
	}
}

// OnPositionUpdate registers callback for position updates
func (c *TraderClient) OnPositionUpdate(callback func(*PositionUpdate)) {
	c.callbacks["position_update"] = func(data interface{}) {
		if update, ok := data.(*PositionUpdate); ok {
			callback(update)
		}
	}
}

// OnPriceUpdate registers callback for price updates
func (c *TraderClient) OnPriceUpdate(callback func(*PriceUpdate)) {
	c.callbacks["price_update"] = func(data interface{}) {
		if update, ok := data.(*PriceUpdate); ok {
			callback(update)
		}
	}
}

// OnTradeUpdate registers callback for trade updates
func (c *TraderClient) OnTradeUpdate(callback func(*TradeUpdate)) {
	c.callbacks["trade_update"] = func(data interface{}) {
		if update, ok := data.(*TradeUpdate); ok {
			callback(update)
		}
	}
}

// Internal Methods

// handleMessages handles incoming WebSocket messages
func (c *TraderClient) handleMessages() {
	for c.connected {
		var msg map[string]interface{}
		if err := c.wsConn.ReadJSON(&msg); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.errorChan <- err
			}
			break
		}
		
		c.processMessage(msg)
	}
}

// processMessage processes a single message
func (c *TraderClient) processMessage(msg map[string]interface{}) {
	msgType, ok := msg["type"].(string)
	if !ok {
		return
	}
	
	switch msgType {
	case "order_update":
		c.handleOrderUpdate(msg)
	case "position_update":
		c.handlePositionUpdate(msg)
	case "price_update":
		c.handlePriceUpdate(msg)
	case "trade_update":
		c.handleTradeUpdate(msg)
	case "orderbook_update":
		c.handleOrderBookUpdate(msg)
	case "balance_update":
		c.handleBalanceUpdate(msg)
	case "error":
		c.handleError(msg)
	}
}

// handleOrderUpdate handles order update messages
func (c *TraderClient) handleOrderUpdate(msg map[string]interface{}) {
	// Parse order update
	update := &OrderUpdate{
		Timestamp: time.Now(),
	}
	
	// Extract order data
	if orderData, ok := msg["order"].(map[string]interface{}); ok {
		// Convert to Order struct
		orderJSON, _ := json.Marshal(orderData)
		var order lx.Order
		json.Unmarshal(orderJSON, &order)
		update.Order = &order
		
		// Update local state
		c.mu.Lock()
		c.orders[order.ID] = &order
		c.mu.Unlock()
	}
	
	// Send to channel
	select {
	case c.orderUpdates <- update:
	default:
	}
	
	// Call callback
	if callback, exists := c.callbacks["order_update"]; exists {
		callback(update)
	}
}

// handlePositionUpdate handles position update messages
func (c *TraderClient) handlePositionUpdate(msg map[string]interface{}) {
	update := &PositionUpdate{
		Timestamp: time.Now(),
	}
	
	if action, ok := msg["action"].(string); ok {
		update.Action = action
	}
	
	// Extract position data
	if posData, ok := msg["position"].(map[string]interface{}); ok {
		// Convert to Position struct
		posJSON, _ := json.Marshal(posData)
		var position lx.MarginPosition
		json.Unmarshal(posJSON, &position)
		update.Position = &position
		
		// Update local state
		c.mu.Lock()
		if update.Action == "closed" || update.Action == "liquidated" {
			delete(c.positions, position.ID)
		} else {
			c.positions[position.ID] = &position
		}
		c.mu.Unlock()
	}
	
	// Send to channel
	select {
	case c.positionUpdates <- update:
	default:
	}
	
	// Call callback
	if callback, exists := c.callbacks["position_update"]; exists {
		callback(update)
	}
}

// handlePriceUpdate handles price update messages
func (c *TraderClient) handlePriceUpdate(msg map[string]interface{}) {
	update := &PriceUpdate{
		Timestamp: time.Now(),
	}
	
	if symbol, ok := msg["symbol"].(string); ok {
		update.Symbol = symbol
	}
	if price, ok := msg["price"].(float64); ok {
		update.Price = price
		
		// Update local state
		c.mu.Lock()
		c.prices[update.Symbol] = price
		c.mu.Unlock()
	}
	if bid, ok := msg["bid"].(float64); ok {
		update.Bid = bid
	}
	if ask, ok := msg["ask"].(float64); ok {
		update.Ask = ask
	}
	if volume, ok := msg["volume"].(float64); ok {
		update.Volume = volume
	}
	
	// Send to channel
	select {
	case c.priceUpdates <- update:
	default:
	}
	
	// Call callback
	if callback, exists := c.callbacks["price_update"]; exists {
		callback(update)
	}
}

// handleTradeUpdate handles trade update messages
func (c *TraderClient) handleTradeUpdate(msg map[string]interface{}) {
	update := &TradeUpdate{
		Timestamp: time.Now(),
	}
	
	// Extract trade data
	if tradeData, ok := msg["trade"].(map[string]interface{}); ok {
		// Convert to Trade struct
		tradeJSON, _ := json.Marshal(tradeData)
		var trade lx.Trade
		json.Unmarshal(tradeJSON, &trade)
		update.Trade = &trade
		
		// Update local state
		c.mu.Lock()
		if c.trades[trade.Symbol] == nil {
			c.trades[trade.Symbol] = make([]*lx.Trade, 0)
		}
		c.trades[trade.Symbol] = append(c.trades[trade.Symbol], &trade)
		
		// Limit trade history
		if len(c.trades[trade.Symbol]) > 100 {
			c.trades[trade.Symbol] = c.trades[trade.Symbol][1:]
		}
		c.mu.Unlock()
	}
	
	// Send to channel
	select {
	case c.tradeUpdates <- update:
	default:
	}
	
	// Call callback
	if callback, exists := c.callbacks["trade_update"]; exists {
		callback(update)
	}
}

// handleOrderBookUpdate handles order book update messages
func (c *TraderClient) handleOrderBookUpdate(msg map[string]interface{}) {
	if symbol, ok := msg["symbol"].(string); ok {
		if snapshot, ok := msg["snapshot"].(map[string]interface{}); ok {
			// Convert to OrderBookSnapshot
			snapshotJSON, _ := json.Marshal(snapshot)
			var ob lx.OrderBookSnapshot
			json.Unmarshal(snapshotJSON, &ob)
			
			// Update local state
			c.mu.Lock()
			c.orderBooks[symbol] = &ob
			c.mu.Unlock()
		}
	}
}

// handleBalanceUpdate handles balance update messages
func (c *TraderClient) handleBalanceUpdate(msg map[string]interface{}) {
	if balances, ok := msg["balances"].(map[string]interface{}); ok {
		c.mu.Lock()
		for asset, balance := range balances {
			if balStr, ok := balance.(string); ok {
				bal, _ := new(big.Int).SetString(balStr, 10)
				c.balances[asset] = bal
			}
		}
		c.mu.Unlock()
	}
}

// handleError handles error messages
func (c *TraderClient) handleError(msg map[string]interface{}) {
	if errMsg, ok := msg["error"].(string); ok {
		err := fmt.Errorf("server error: %s", errMsg)
		select {
		case c.errorChan <- err:
		default:
		}
	}
}

// heartbeat sends periodic heartbeat messages
func (c *TraderClient) heartbeat() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			if c.connected {
				msg := map[string]interface{}{
					"type": "ping",
				}
				c.wsConn.WriteJSON(msg)
			}
		case <-c.ctx.Done():
			return
		}
	}
}

// GetOrderUpdates returns the order updates channel
func (c *TraderClient) GetOrderUpdates() <-chan *OrderUpdate {
	return c.orderUpdates
}

// GetPositionUpdates returns the position updates channel
func (c *TraderClient) GetPositionUpdates() <-chan *PositionUpdate {
	return c.positionUpdates
}

// GetPriceUpdates returns the price updates channel
func (c *TraderClient) GetPriceUpdates() <-chan *PriceUpdate {
	return c.priceUpdates
}

// GetTradeUpdates returns the trade updates channel
func (c *TraderClient) GetTradeUpdates() <-chan *TradeUpdate {
	return c.tradeUpdates
}

// GetErrors returns the error channel
func (c *TraderClient) GetErrors() <-chan error {
	return c.errorChan
}