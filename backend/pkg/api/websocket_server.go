package api

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/dex/backend/pkg/lx"
)

// WebSocketServer handles real-time trading connections
type WebSocketServer struct {
	// Core components
	engine            *lx.TradingEngine
	marginEngine      *lx.MarginEngine
	lendingPool       *lx.LendingPool
	unifiedPool       *lx.UnifiedLiquidityPool
	oracle            *lx.PriceOracle
	vaultManager      *lx.VaultManager
	xchain            *lx.XChainIntegration
	liquidationEngine *lx.LiquidationEngine
	
	// WebSocket
	upgrader          websocket.Upgrader
	clients           map[string]*Client
	broadcast         chan []byte
	
	// Subscriptions
	subscriptions     map[string]map[string]bool // client -> symbols
	
	// Auth
	authService       AuthService
	
	// Metrics
	metrics           *ServerMetrics
	
	// Control
	ctx               context.Context
	cancel            context.CancelFunc
	mu                sync.RWMutex
}

// Client represents a connected trader
type Client struct {
	ID                string
	UserID            string
	conn              *websocket.Conn
	send              chan []byte
	subscriptions     map[string]bool
	authenticated     bool
	lastActivity      time.Time
	rateLimiter       *RateLimiter
	mu                sync.RWMutex
}

// Message represents a WebSocket message
type Message struct {
	Type              string                 `json:"type"`
	Data              map[string]interface{} `json:"data,omitempty"`
	Error             string                 `json:"error,omitempty"`
	RequestID         string                 `json:"request_id,omitempty"`
	Timestamp         int64                  `json:"timestamp"`
}

// AuthService handles authentication
type AuthService interface {
	Authenticate(apiKey, apiSecret string) (string, error)
	ValidateSession(sessionID string) (string, error)
	GetUserID(sessionID string) (string, error)
}

// ServerMetrics tracks server performance
type ServerMetrics struct {
	ConnectionsTotal     uint64
	ConnectionsActive    uint64
	MessagesReceived     uint64
	MessagesSent         uint64
	SubscriptionsActive  uint64
	AuthFailures         uint64
	OrdersProcessed      uint64
	PositionsOpened      uint64
	LiquidationsExecuted uint64
	ErrorCount           uint64
	mu                   sync.RWMutex
}

// RateLimiter implements rate limiting
type RateLimiter struct {
	requests          int
	maxRequests       int
	window            time.Duration
	lastReset         time.Time
	mu                sync.Mutex
}

// NewWebSocketServer creates a new WebSocket server
func NewWebSocketServer(config ServerConfig) *WebSocketServer {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &WebSocketServer{
		engine:            config.Engine,
		marginEngine:      config.MarginEngine,
		lendingPool:       config.LendingPool,
		unifiedPool:       config.UnifiedPool,
		oracle:            config.Oracle,
		vaultManager:      config.VaultManager,
		xchain:            config.XChain,
		liquidationEngine: config.LiquidationEngine,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// Configure CORS as needed
				return true
			},
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
		},
		clients:       make(map[string]*Client),
		broadcast:     make(chan []byte, 256),
		subscriptions: make(map[string]map[string]bool),
		authService:   config.AuthService,
		metrics:       NewServerMetrics(),
		ctx:           ctx,
		cancel:        cancel,
	}
}

// ServerConfig contains server configuration
type ServerConfig struct {
	Engine            *lx.TradingEngine
	MarginEngine      *lx.MarginEngine
	LendingPool       *lx.LendingPool
	UnifiedPool       *lx.UnifiedLiquidityPool
	Oracle            *lx.PriceOracle
	VaultManager      *lx.VaultManager
	XChain            *lx.XChainIntegration
	LiquidationEngine *lx.LiquidationEngine
	AuthService       AuthService
}

// HandleConnection handles new WebSocket connections
func (ws *WebSocketServer) HandleConnection(w http.ResponseWriter, r *http.Request) {
	conn, err := ws.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	
	clientID := generateClientID()
	client := &Client{
		ID:            clientID,
		conn:          conn,
		send:          make(chan []byte, 256),
		subscriptions: make(map[string]bool),
		lastActivity:  time.Now(),
		rateLimiter:   NewRateLimiter(100, time.Minute),
	}
	
	ws.mu.Lock()
	ws.clients[clientID] = client
	ws.metrics.ConnectionsActive++
	ws.metrics.ConnectionsTotal++
	ws.mu.Unlock()
	
	// Start client handlers
	go client.writePump()
	go client.readPump(ws)
	
	// Send welcome message
	welcome := Message{
		Type:      "connected",
		Data:      map[string]interface{}{"client_id": clientID},
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(welcome)
}

// Client message handlers

func (c *Client) readPump(ws *WebSocketServer) {
	defer func() {
		ws.removeClient(c)
		c.conn.Close()
	}()
	
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})
	
	for {
		var msg map[string]interface{}
		err := c.conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				// Log error
			}
			break
		}
		
		// Rate limiting
		if !c.rateLimiter.Allow() {
			c.sendError("Rate limit exceeded", "")
			continue
		}
		
		// Update activity
		c.lastActivity = time.Now()
		ws.metrics.MessagesReceived++
		
		// Process message
		ws.processMessage(c, msg)
	}
}

func (c *Client) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()
	
	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			
			c.conn.WriteMessage(websocket.TextMessage, message)
			
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// Message processing

func (ws *WebSocketServer) processMessage(client *Client, msg map[string]interface{}) {
	msgType, ok := msg["type"].(string)
	if !ok {
		client.sendError("Invalid message type", "")
		return
	}
	
	// Get request ID if present
	requestID := ""
	if rid, ok := msg["request_id"].(string); ok {
		requestID = rid
	}
	
	switch msgType {
	case "auth":
		ws.handleAuth(client, msg, requestID)
	case "ping":
		ws.handlePing(client, requestID)
	case "subscribe":
		ws.handleSubscribe(client, msg, requestID)
	case "unsubscribe":
		ws.handleUnsubscribe(client, msg, requestID)
	case "place_order":
		ws.handlePlaceOrder(client, msg, requestID)
	case "cancel_order":
		ws.handleCancelOrder(client, msg, requestID)
	case "modify_order":
		ws.handleModifyOrder(client, msg, requestID)
	case "open_position":
		ws.handleOpenPosition(client, msg, requestID)
	case "close_position":
		ws.handleClosePosition(client, msg, requestID)
	case "modify_leverage":
		ws.handleModifyLeverage(client, msg, requestID)
	case "vault_deposit":
		ws.handleVaultDeposit(client, msg, requestID)
	case "vault_withdraw":
		ws.handleVaultWithdraw(client, msg, requestID)
	case "lending_supply":
		ws.handleLendingSupply(client, msg, requestID)
	case "lending_borrow":
		ws.handleLendingBorrow(client, msg, requestID)
	case "lending_repay":
		ws.handleLendingRepay(client, msg, requestID)
	case "get_balances":
		ws.handleGetBalances(client, requestID)
	case "get_positions":
		ws.handleGetPositions(client, requestID)
	case "get_orders":
		ws.handleGetOrders(client, requestID)
	default:
		client.sendError(fmt.Sprintf("Unknown message type: %s", msgType), requestID)
	}
}

// Authentication handler
func (ws *WebSocketServer) handleAuth(client *Client, msg map[string]interface{}, requestID string) {
	apiKey, _ := msg["apiKey"].(string)
	apiSecret, _ := msg["apiSecret"].(string)
	
	if apiKey == "" || apiSecret == "" {
		client.sendError("Missing credentials", requestID)
		ws.metrics.AuthFailures++
		return
	}
	
	userID, err := ws.authService.Authenticate(apiKey, apiSecret)
	if err != nil {
		client.sendError("Authentication failed", requestID)
		ws.metrics.AuthFailures++
		return
	}
	
	client.mu.Lock()
	client.authenticated = true
	client.UserID = userID
	client.mu.Unlock()
	
	response := Message{
		Type:      "auth_success",
		Data:      map[string]interface{}{"user_id": userID},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
	
	// Send initial data
	ws.sendInitialData(client)
}

// Trading handlers

// Helper function to parse Side from string
func parseSide(s string) lx.Side {
	switch s {
	case "buy", "BUY":
		return lx.Buy
	case "sell", "SELL":
		return lx.Sell
	default:
		return lx.Buy
	}
}

// Helper function to parse OrderType from string
func parseOrderType(s string) lx.OrderType {
	switch s {
	case "market", "MARKET":
		return lx.Market
	case "limit", "LIMIT":
		return lx.Limit
	case "stop", "STOP":
		return lx.Stop
	case "stop_limit", "STOP_LIMIT":
		return lx.StopLimit
	default:
		return lx.Limit
	}
}

// Helper function to get opposite side
func oppositeSide(side lx.Side) lx.Side {
	if side == lx.Buy {
		return lx.Sell
	}
	return lx.Buy
}

func (ws *WebSocketServer) handlePlaceOrder(client *Client, msg map[string]interface{}, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Parse order from message with proper type checking
	orderDataRaw, ok := msg["order"]
	if !ok {
		client.sendError("Missing order data", requestID)
		return
	}
	
	orderData, ok := orderDataRaw.(map[string]interface{})
	if !ok {
		client.sendError("Invalid order data format", requestID)
		return
	}
	
	// Extract fields with safe type assertions
	symbol, ok := orderData["symbol"].(string)
	if !ok {
		client.sendError("Missing or invalid symbol", requestID)
		return
	}
	
	side, ok := orderData["side"].(string)
	if !ok {
		client.sendError("Missing or invalid side", requestID)
		return
	}
	
	orderType, ok := orderData["type"].(string)
	if !ok {
		client.sendError("Missing or invalid order type", requestID)
		return
	}
	
	price, ok := orderData["price"].(float64)
	if !ok {
		client.sendError("Missing or invalid price", requestID)
		return
	}
	
	size, ok := orderData["size"].(float64)
	if !ok {
		client.sendError("Missing or invalid size", requestID)
		return
	}
	
	order := &lx.Order{
		Symbol:    symbol,
		Side:      parseSide(side),
		Type:      parseOrderType(orderType),
		Price:     price,
		Size:      size,
		User:      client.UserID,
		Timestamp: time.Now(),
	}
	
	// Submit order - get the order book and add the order
	book := ws.engine.GetOrderBook(order.Symbol)
	if book == nil {
		ws.engine.CreateOrderBook(order.Symbol)
		book = ws.engine.GetOrderBook(order.Symbol)
	}
	
	orderID := book.AddOrder(order)
	if orderID == 0 {
		client.sendError("Order failed", requestID)
		return
	}
	order.ID = orderID
	
	ws.metrics.OrdersProcessed++
	
	// Get any trades that may have occurred (would need to be tracked separately)
	// For now, we'll skip broadcasting trades as AddOrder doesn't return them
	
	response := Message{
		Type: "order_update",
		Data: map[string]interface{}{
			"order":  order,
			"status": "submitted",
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

func (ws *WebSocketServer) handleCancelOrder(client *Client, msg map[string]interface{}, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Safe type assertion for orderID
	orderIDRaw, ok := msg["orderID"]
	if !ok {
		client.sendError("Missing orderID", requestID)
		return
	}
	
	orderIDFloat, ok := orderIDRaw.(float64)
	if !ok {
		client.sendError("Invalid orderID format", requestID)
		return
	}
	
	orderID := uint64(orderIDFloat)
	
	// Find and cancel the order across all order books
	var err error
	var orderFound bool
	for _, book := range ws.engine.OrderBooks {
		if book.CancelOrder(orderID) == nil {
			orderFound = true
			break
		}
	}
	
	if !orderFound {
		err = fmt.Errorf("order not found")
	}
	if err != nil {
		client.sendError(fmt.Sprintf("Cancel failed: %v", err), requestID)
		return
	}
	
	response := Message{
		Type: "order_update",
		Data: map[string]interface{}{
			"order_id": orderID,
			"status":   "cancelled",
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

// Margin trading handlers

func (ws *WebSocketServer) handleOpenPosition(client *Client, msg map[string]interface{}, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Safe type assertions for all fields
	symbol, ok := msg["symbol"].(string)
	if !ok {
		client.sendError("Missing or invalid symbol", requestID)
		return
	}
	
	sideStr, ok := msg["side"].(string)
	if !ok {
		client.sendError("Missing or invalid side", requestID)
		return
	}
	side := parseSide(sideStr)
	
	size, ok := msg["size"].(float64)
	if !ok {
		client.sendError("Missing or invalid size", requestID)
		return
	}
	
	leverage, ok := msg["leverage"].(float64)
	if !ok {
		client.sendError("Missing or invalid leverage", requestID)
		return
	}
	
	// Create order for position
	order := &lx.Order{
		Symbol: symbol,
		Side:   side,
		Type:   lx.Market,
		Size:   size,
		User:   client.UserID,
	}
	
	// Open position
	position, err := ws.marginEngine.OpenPosition(client.UserID, order, leverage)
	if err != nil {
		client.sendError(fmt.Sprintf("Position failed: %v", err), requestID)
		return
	}
	
	ws.metrics.PositionsOpened++
	
	response := Message{
		Type: "position_update",
		Data: map[string]interface{}{
			"position": position,
			"action":   "opened",
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

func (ws *WebSocketServer) handleClosePosition(client *Client, msg map[string]interface{}, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Safe type assertions
	positionID, ok := msg["positionID"].(string)
	if !ok {
		client.sendError("Missing or invalid positionID", requestID)
		return
	}
	
	size, ok := msg["size"].(float64)
	if !ok {
		client.sendError("Missing or invalid size", requestID)
		return
	}
	
	err := ws.marginEngine.ClosePosition(client.UserID, positionID, size)
	if err != nil {
		client.sendError(fmt.Sprintf("Close failed: %v", err), requestID)
		return
	}
	
	response := Message{
		Type: "position_update",
		Data: map[string]interface{}{
			"position_id": positionID,
			"action":      "closed",
			"size":        size,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

// Vault operations

func (ws *WebSocketServer) handleVaultDeposit(client *Client, msg map[string]interface{}, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Safe type assertions
	vaultID, ok := msg["vaultID"].(string)
	if !ok {
		client.sendError("Missing or invalid vaultID", requestID)
		return
	}
	
	amountStr, ok := msg["amount"].(string)
	if !ok {
		client.sendError("Missing or invalid amount", requestID)
		return
	}
	
	amount, success := new(big.Int).SetString(amountStr, 10)
	if !success {
		client.sendError("Invalid amount format", requestID)
		return
	}
	
	vault, err := ws.vaultManager.GetVault(vaultID)
	if err != nil {
		client.sendError(fmt.Sprintf("Vault not found: %v", err), requestID)
		return
	}
	
	position, err := vault.Deposit(client.UserID, amount)
	if err != nil {
		client.sendError(fmt.Sprintf("Deposit failed: %v", err), requestID)
		return
	}
	
	// Convert position to a proper map for JSON serialization
	positionData := map[string]interface{}{
		"User":          position.User,
		"Shares":        position.Shares.String(),
		"DepositValue":  position.DepositValue.String(),
		"CurrentValue":  position.CurrentValue.String(),
		"LockedUntil":   position.LockedUntil.Format(time.RFC3339),
		"LastUpdate":    position.LastUpdate.Format(time.RFC3339),
		"RealizedPnL":   position.RealizedPnL.String(),
		"UnrealizedPnL": position.UnrealizedPnL.String(),
	}
	
	response := Message{
		Type: "vault_update",
		Data: map[string]interface{}{
			"vault_id": vaultID,
			"action":   "deposited",
			"position": positionData,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

// Lending operations

func (ws *WebSocketServer) handleLendingSupply(client *Client, msg map[string]interface{}, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Safe type assertions
	asset, ok := msg["asset"].(string)
	if !ok {
		client.sendError("Missing or invalid asset", requestID)
		return
	}
	
	amountStr, ok := msg["amount"].(string)
	if !ok {
		client.sendError("Missing or invalid amount", requestID)
		return
	}
	
	amount, success := new(big.Int).SetString(amountStr, 10)
	if !success {
		client.sendError("Invalid amount format", requestID)
		return
	}
	
	err := ws.lendingPool.Supply(client.UserID, asset, amount)
	if err != nil {
		client.sendError(fmt.Sprintf("Supply failed: %v", err), requestID)
		return
	}
	
	response := Message{
		Type: "lending_update",
		Data: map[string]interface{}{
			"action": "supplied",
			"asset":  asset,
			"amount": amount.String(),
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

// Market data subscriptions

func (ws *WebSocketServer) handleSubscribe(client *Client, msg map[string]interface{}, requestID string) {
	// Safe type assertions
	channel, ok := msg["channel"].(string)
	if !ok {
		client.sendError("Missing or invalid channel", requestID)
		return
	}
	
	symbolsRaw, ok := msg["symbols"]
	if !ok {
		client.sendError("Missing symbols", requestID)
		return
	}
	
	symbols, ok := symbolsRaw.([]interface{})
	if !ok {
		client.sendError("Invalid symbols format", requestID)
		return
	}
	
	client.mu.Lock()
	for _, symbol := range symbols {
		key := fmt.Sprintf("%s:%s", channel, symbol)
		client.subscriptions[key] = true
	}
	client.mu.Unlock()
	
	ws.mu.Lock()
	if ws.subscriptions[client.ID] == nil {
		ws.subscriptions[client.ID] = make(map[string]bool)
	}
	for _, symbol := range symbols {
		ws.subscriptions[client.ID][symbol.(string)] = true
	}
	ws.metrics.SubscriptionsActive++
	ws.mu.Unlock()
	
	response := Message{
		Type: "subscribed",
		Data: map[string]interface{}{
			"channel": channel,
			"symbols": symbols,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

func (ws *WebSocketServer) handleUnsubscribe(client *Client, msg map[string]interface{}, requestID string) {
	// Safe type assertions
	channel, ok := msg["channel"].(string)
	if !ok {
		client.sendError("Missing or invalid channel", requestID)
		return
	}
	
	symbolsRaw, ok := msg["symbols"]
	if !ok {
		client.sendError("Missing symbols", requestID)
		return
	}
	
	symbols, ok := symbolsRaw.([]interface{})
	if !ok {
		client.sendError("Invalid symbols format", requestID)
		return
	}
	
	client.mu.Lock()
	for _, symbol := range symbols {
		key := fmt.Sprintf("%s:%s", channel, symbol)
		delete(client.subscriptions, key)
	}
	client.mu.Unlock()
	
	ws.mu.Lock()
	for _, symbol := range symbols {
		delete(ws.subscriptions[client.ID], symbol.(string))
	}
	ws.metrics.SubscriptionsActive--
	ws.mu.Unlock()
	
	response := Message{
		Type: "unsubscribed",
		Data: map[string]interface{}{
			"channel": channel,
			"symbols": symbols,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

// Account data handlers

func (ws *WebSocketServer) handleGetBalances(client *Client, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Get balances from margin account
	account := ws.marginEngine.GetAccount(client.UserID)
	if account == nil {
		client.sendError("Account not found", requestID)
		return
	}
	
	balances := make(map[string]string)
	for asset, collateral := range account.CollateralAssets {
		balances[asset] = collateral.Amount.String()
	}
	
	response := Message{
		Type: "balance_update",
		Data: map[string]interface{}{
			"balances": balances,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

func (ws *WebSocketServer) handleGetPositions(client *Client, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	account := ws.marginEngine.GetAccount(client.UserID)
	if account == nil {
		client.sendError("Account not found", requestID)
		return
	}
	
	response := Message{
		Type: "positions_update",
		Data: map[string]interface{}{
			"positions": account.Positions,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

func (ws *WebSocketServer) handleGetOrders(client *Client, requestID string) {
	if !client.authenticated {
		client.sendError("Not authenticated", requestID)
		return
	}
	
	// Get orders from engine
	orders := ws.engine.GetUserOrders(client.UserID)
	
	response := Message{
		Type: "orders_update",
		Data: map[string]interface{}{
			"orders": orders,
		},
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

// Broadcast methods

func (ws *WebSocketServer) BroadcastOrderBook(symbol string, snapshot *lx.OrderBookSnapshot) {
	msg := Message{
		Type: "orderbook_update",
		Data: map[string]interface{}{
			"symbol":   symbol,
			"snapshot": snapshot,
		},
		Timestamp: time.Now().Unix(),
	}
	
	ws.broadcastToSubscribers(fmt.Sprintf("orderbook:%s", symbol), msg)
}

func (ws *WebSocketServer) BroadcastTrade(trade *lx.Trade) {
	msg := Message{
		Type: "trade_update",
		Data: map[string]interface{}{
			"trade": trade,
		},
		Timestamp: time.Now().Unix(),
	}
	
	ws.broadcastToSubscribers(fmt.Sprintf("trades:%s", trade.Symbol), msg)
}

func (ws *WebSocketServer) BroadcastPrice(symbol string, price float64) {
	msg := Message{
		Type: "price_update",
		Data: map[string]interface{}{
			"symbol": symbol,
			"price":  price,
		},
		Timestamp: time.Now().Unix(),
	}
	
	ws.broadcastToSubscribers(fmt.Sprintf("prices:%s", symbol), msg)
}

func (ws *WebSocketServer) broadcastToSubscribers(channel string, msg Message) {
	ws.mu.RLock()
	defer ws.mu.RUnlock()
	
	data, _ := json.Marshal(msg)
	
	for _, client := range ws.clients {
		client.mu.RLock()
		if client.subscriptions[channel] {
			select {
			case client.send <- data:
				ws.metrics.MessagesSent++
			default:
				// Client send channel full
			}
		}
		client.mu.RUnlock()
	}
}

// Helper methods

func (ws *WebSocketServer) handlePing(client *Client, requestID string) {
	response := Message{
		Type:      "pong",
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(response)
}

func (ws *WebSocketServer) sendInitialData(client *Client) {
	// Send current balances
	ws.handleGetBalances(client, "")
	
	// Send current positions
	ws.handleGetPositions(client, "")
	
	// Send open orders
	ws.handleGetOrders(client, "")
}

func (ws *WebSocketServer) removeClient(client *Client) {
	ws.mu.Lock()
	delete(ws.clients, client.ID)
	delete(ws.subscriptions, client.ID)
	ws.metrics.ConnectionsActive--
	ws.mu.Unlock()
	
	close(client.send)
}

func (c *Client) sendMessage(msg Message) {
	data, _ := json.Marshal(msg)
	select {
	case c.send <- data:
	default:
		// Send channel full
	}
}

func (c *Client) sendError(errMsg string, requestID string) {
	msg := Message{
		Type:      "error",
		Error:     errMsg,
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}
	c.sendMessage(msg)
}

// RateLimiter implementation

func NewRateLimiter(maxRequests int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		maxRequests: maxRequests,
		window:      window,
		lastReset:   time.Now(),
	}
}

func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	
	now := time.Now()
	if now.Sub(rl.lastReset) > rl.window {
		rl.requests = 0
		rl.lastReset = now
	}
	
	if rl.requests >= rl.maxRequests {
		return false
	}
	
	rl.requests++
	return true
}

// Metrics

func NewServerMetrics() *ServerMetrics {
	return &ServerMetrics{}
}

func (sm *ServerMetrics) GetSnapshot() map[string]interface{} {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	return map[string]interface{}{
		"connections_total":      sm.ConnectionsTotal,
		"connections_active":     sm.ConnectionsActive,
		"messages_received":      sm.MessagesReceived,
		"messages_sent":          sm.MessagesSent,
		"subscriptions_active":   sm.SubscriptionsActive,
		"auth_failures":          sm.AuthFailures,
		"orders_processed":       sm.OrdersProcessed,
		"positions_opened":       sm.PositionsOpened,
		"liquidations_executed":  sm.LiquidationsExecuted,
		"error_count":            sm.ErrorCount,
	}
}

// Utility functions

func generateClientID() string {
	return fmt.Sprintf("client_%d", time.Now().UnixNano())
}

// Start starts the WebSocket server
func (ws *WebSocketServer) Start() {
	// Start market data broadcaster
	go ws.marketDataBroadcaster()
	
	// Start position monitor
	go ws.positionMonitor()
	
	// Start metrics reporter
	go ws.metricsReporter()
}

// Market data broadcaster
func (ws *WebSocketServer) marketDataBroadcaster() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Broadcast order book updates
			symbols := []string{"BTC-USDT", "ETH-USDT", "BNB-USDT"}
			for _, symbol := range symbols {
				ob := ws.engine.GetOrderBook(symbol)
				if ob != nil {
					snapshot := ob.GetSnapshot()
					ws.BroadcastOrderBook(symbol, snapshot)
				}
				
				// Broadcast price updates
				price := ws.oracle.GetPrice(symbol)
				if price > 0 {
					ws.BroadcastPrice(symbol, price)
				}
			}
		case <-ws.ctx.Done():
			return
		}
	}
}

// Position monitor checks for liquidations
func (ws *WebSocketServer) positionMonitor() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ws.checkLiquidations()
		case <-ws.ctx.Done():
			return
		}
	}
}

func (ws *WebSocketServer) checkLiquidations() {
	// TODO: Implement GetAllAccounts and liquidation monitoring
	// accounts := ws.marginEngine.GetAllAccounts()
	
	// for _, account := range accounts {
	// 	for _, position := range account.Positions {
	// 		// Check if position should be liquidated
	// 		markPrice := ws.oracle.GetPrice(position.Symbol)
	// 		if ws.marginEngine.ShouldLiquidate(position, markPrice) {
	// 			// Process liquidation
	// 			liquidationOrder := &lx.Order{
	// 				Symbol: position.Symbol,
	// 				Side:   oppositeSide(position.Side),
	// 				Type:   lx.Market,
	// 				Size:   position.Size,
	// 				User:   "liquidation_engine",
	// 			}
			// 	
			// 	err := ws.liquidationEngine.ProcessLiquidation(account.UserID, position, liquidationOrder)
			// 	if err == nil {
			// 		ws.metrics.LiquidationsExecuted++
			// 		
			// 		// Notify client
			// 		ws.notifyLiquidation(account.UserID, position)
			// 	}
			// }
		// }
	// }
}

func (ws *WebSocketServer) notifyLiquidation(userID string, position *lx.MarginPosition) {
	// Find client for user
	ws.mu.RLock()
	var targetClient *Client
	for _, client := range ws.clients {
		if client.UserID == userID {
			targetClient = client
			break
		}
	}
	ws.mu.RUnlock()
	
	if targetClient != nil {
		msg := Message{
			Type: "position_update",
			Data: map[string]interface{}{
				"position": position,
				"action":   "liquidated",
				"message":  fmt.Sprintf("Position %s liquidated at %.2f", position.ID, position.MarkPrice),
			},
			Timestamp: time.Now().Unix(),
		}
		targetClient.sendMessage(msg)
	}
}

// handleModifyOrder handles order modification requests
func (ws *WebSocketServer) handleModifyOrder(client *Client, msg map[string]interface{}, requestID string) {
	// Extract order data - handle both formats
	var orderID uint64
	if id, ok := msg["orderID"].(float64); ok {
		orderID = uint64(id)
	} else if id, ok := msg["order_id"].(float64); ok {
		orderID = uint64(id)
	}
	
	var newPrice, newSize float64
	if price, ok := msg["newPrice"].(float64); ok {
		newPrice = price
	} else if price, ok := msg["price"].(float64); ok {
		newPrice = price
	}
	
	if size, ok := msg["newSize"].(float64); ok {
		newSize = size
	} else if size, ok := msg["size"].(float64); ok {
		newSize = size
	}
	
	// Modify order logic would go here
	// For now, send acknowledgment
	client.sendMessage(Message{
		Type:      "order_modified",
		Data:      map[string]interface{}{"order_id": orderID, "price": newPrice, "size": newSize},
		Timestamp: time.Now().Unix(),
	})
}

// handleModifyLeverage handles leverage modification requests
func (ws *WebSocketServer) handleModifyLeverage(client *Client, msg map[string]interface{}, requestID string) {
	userID := client.UserID
	
	// Safe type assertions
	positionID, ok := msg["position_id"].(string)
	if !ok {
		client.sendError("Missing or invalid position_id", requestID)
		return
	}
	
	newLeverage, ok := msg["leverage"].(float64)
	if !ok {
		client.sendError("Missing or invalid leverage", requestID)
		return
	}
	
	err := ws.marginEngine.ModifyLeverage(userID, positionID, newLeverage)
	if err != nil {
		client.sendError(err.Error(), requestID)
		return
	}
	
	client.sendMessage(Message{
		Type:      "leverage_modified",
		Data:      map[string]interface{}{"position_id": positionID, "leverage": newLeverage},
		Timestamp: time.Now().Unix(),
	})
}

// handleVaultWithdraw handles vault withdrawal requests
func (ws *WebSocketServer) handleVaultWithdraw(client *Client, msg map[string]interface{}, requestID string) {
	// Safe type assertions
	vaultID, ok := msg["vault_id"].(string)
	if !ok {
		client.sendError("Missing or invalid vault_id", requestID)
		return
	}
	
	amountStr, ok := msg["amount"].(string)
	if !ok {
		client.sendError("Missing or invalid amount", requestID)
		return
	}
	
	amount := new(big.Int)
	if _, success := amount.SetString(amountStr, 10); !success {
		client.sendError("Invalid amount format", requestID)
		return
	}
	
	// Vault withdrawal logic would go here
	client.sendMessage(Message{
		Type:      "vault_withdrawal",
		Data:      map[string]interface{}{"vault_id": vaultID, "amount": amount.String()},
		Timestamp: time.Now().Unix(),
	})
}

// handleLendingBorrow handles lending borrow requests
func (ws *WebSocketServer) handleLendingBorrow(client *Client, msg map[string]interface{}, requestID string) {
	// Safe type assertions
	asset, ok := msg["asset"].(string)
	if !ok {
		client.sendError("Missing or invalid asset", requestID)
		return
	}
	
	amountStr, ok := msg["amount"].(string)
	if !ok {
		client.sendError("Missing or invalid amount", requestID)
		return
	}
	
	amount := new(big.Int)
	if _, success := amount.SetString(amountStr, 10); !success {
		client.sendError("Invalid amount format", requestID)
		return
	}
	
	err := ws.lendingPool.Borrow(asset, amount)
	if err != nil {
		client.sendError(err.Error(), requestID)
		return
	}
	
	client.sendMessage(Message{
		Type:      "borrow_success",
		Data:      map[string]interface{}{"asset": asset, "amount": amount.String()},
		Timestamp: time.Now().Unix(),
	})
}

// handleLendingRepay handles lending repayment requests
func (ws *WebSocketServer) handleLendingRepay(client *Client, msg map[string]interface{}, requestID string) {
	// Safe type assertions
	asset, ok := msg["asset"].(string)
	if !ok {
		client.sendError("Missing or invalid asset", requestID)
		return
	}
	
	amountStr, ok := msg["amount"].(string)
	if !ok {
		client.sendError("Missing or invalid amount", requestID)
		return
	}
	
	interestStr, ok := msg["interest"].(string)
	if !ok {
		client.sendError("Missing or invalid interest", requestID)
		return
	}
	
	amount := new(big.Int)
	if _, success := amount.SetString(amountStr, 10); !success {
		client.sendError("Invalid amount format", requestID)
		return
	}
	
	interest := new(big.Int)
	if _, success := interest.SetString(interestStr, 10); !success {
		client.sendError("Invalid interest format", requestID)
		return
	}
	
	ws.lendingPool.Repay(asset, amount, interest)
	
	client.sendMessage(Message{
		Type:      "repay_success",
		Data:      map[string]interface{}{"asset": asset, "amount": amount.String()},
		Timestamp: time.Now().Unix(),
	})
}

// Metrics reporter
func (ws *WebSocketServer) metricsReporter() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			metrics := ws.metrics.GetSnapshot()
			// Log or send metrics to monitoring system
			_ = metrics
		case <-ws.ctx.Done():
			return
		}
	}
}

// Shutdown gracefully shuts down the server
func (ws *WebSocketServer) Shutdown() {
	ws.cancel()
	
	ws.mu.Lock()
	for _, client := range ws.clients {
		client.conn.Close()
	}
	ws.mu.Unlock()
	
	close(ws.broadcast)
}