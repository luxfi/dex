// Package websocket provides WebSocket server for real-time market data
package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/log"
)

// Server represents a WebSocket server for real-time data
type Server struct {
	orderBook *lx.OrderBook
	logger    log.Logger

	// Client management
	clients    map[*Client]bool
	clientsMu  sync.RWMutex
	register   chan *Client
	unregister chan *Client
	broadcast  chan Message

	// Subscription management
	subscriptions map[string]map[*Client]bool // channel -> clients
	subMu         sync.RWMutex

	// Stats
	messagesOut uint64
	clientCount int32

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Client represents a WebSocket client connection
type Client struct {
	id       string
	conn     *websocket.Conn
	server   *Server
	send     chan []byte
	channels map[string]bool
	mu       sync.RWMutex
}

// Message represents a WebSocket message
type Message struct {
	Type      string      `json:"type"`
	Channel   string      `json:"channel,omitempty"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp int64       `json:"timestamp"`
	Sequence  uint64      `json:"sequence,omitempty"`
}

// SubscribeRequest represents a subscription request
type SubscribeRequest struct {
	Type     string   `json:"type"`
	Channels []string `json:"channels"`
}

// OrderBookUpdate represents an order book update
type OrderBookUpdate struct {
	Type      string          `json:"type"` // "snapshot" or "update"
	Symbol    string          `json:"symbol"`
	Bids      []lx.PriceLevel `json:"bids"`
	Asks      []lx.PriceLevel `json:"asks"`
	Timestamp int64           `json:"timestamp"`
	Sequence  uint64          `json:"sequence"`
}

// TradeUpdate represents a trade update
type TradeUpdate struct {
	TradeID     uint64  `json:"tradeId"`
	Symbol      string  `json:"symbol"`
	Price       float64 `json:"price"`
	Size        float64 `json:"size"`
	Side        string  `json:"side"`
	BuyOrderID  uint64  `json:"buyOrderId"`
	SellOrderID uint64  `json:"sellOrderId"`
	Timestamp   int64   `json:"timestamp"`
}

// Config holds WebSocket server configuration
type Config struct {
	Port            int
	ReadBufferSize  int
	WriteBufferSize int
	MaxMessageSize  int64
	WriteTimeout    time.Duration
	PongTimeout     time.Duration
	PingPeriod      time.Duration
}

// DefaultConfig returns default WebSocket configuration
func DefaultConfig() Config {
	return Config{
		Port:            8081,
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		MaxMessageSize:  512 * 1024, // 512KB
		WriteTimeout:    10 * time.Second,
		PongTimeout:     60 * time.Second,
		PingPeriod:      54 * time.Second, // Must be less than PongTimeout
	}
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		// Allow all origins in development
		// In production, implement proper CORS
		return true
	},
}

// NewServer creates a new WebSocket server
func NewServer(orderBook *lx.OrderBook, logger log.Logger, config Config) *Server {
	ctx, cancel := context.WithCancel(context.Background())

	return &Server{
		orderBook:     orderBook,
		logger:        logger,
		clients:       make(map[*Client]bool),
		register:      make(chan *Client, 100),
		unregister:    make(chan *Client, 100),
		broadcast:     make(chan Message, 1000),
		subscriptions: make(map[string]map[*Client]bool),
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start begins the WebSocket server
func (s *Server) Start(port int) error {
	// Start hub goroutine
	s.wg.Add(1)
	go s.runHub()

	// Start HTTP server
	http.HandleFunc("/ws", s.handleWebSocket)
	http.HandleFunc("/health", s.handleHealth)

	addr := fmt.Sprintf(":%d", port)
	s.logger.Info("WebSocket server starting", "port", port)

	server := &http.Server{
		Addr:         addr,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	go func() {
		<-s.ctx.Done()
		server.Shutdown(context.Background())
	}()

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("WebSocket server error: %w", err)
	}

	return nil
}

// Stop shuts down the WebSocket server
func (s *Server) Stop() {
	s.logger.Info("Stopping WebSocket server")
	s.cancel()
	s.wg.Wait()
}

// runHub manages client connections and message routing
func (s *Server) runHub() {
	defer s.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			// Close all client connections
			s.clientsMu.Lock()
			for client := range s.clients {
				close(client.send)
			}
			s.clientsMu.Unlock()
			return

		case client := <-s.register:
			s.clientsMu.Lock()
			s.clients[client] = true
			atomic.AddInt32(&s.clientCount, 1)
			s.clientsMu.Unlock()
			s.logger.Debug("Client connected", "id", client.id, "total", atomic.LoadInt32(&s.clientCount))

		case client := <-s.unregister:
			s.clientsMu.Lock()
			if _, ok := s.clients[client]; ok {
				delete(s.clients, client)
				close(client.send)
				atomic.AddInt32(&s.clientCount, -1)

				// Remove from all subscriptions
				s.unsubscribeAll(client)
			}
			s.clientsMu.Unlock()
			s.logger.Debug("Client disconnected", "id", client.id, "total", atomic.LoadInt32(&s.clientCount))

		case message := <-s.broadcast:
			s.broadcastMessage(message)

		case <-ticker.C:
			// Log stats periodically
			s.logger.Debug("WebSocket stats",
				"clients", atomic.LoadInt32(&s.clientCount),
				"messages", atomic.LoadUint64(&s.messagesOut))
		}
	}
}

// handleWebSocket handles WebSocket upgrade and client connection
func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Error("WebSocket upgrade failed", "error", err)
		return
	}

	client := &Client{
		id:       generateClientID(),
		conn:     conn,
		server:   s,
		send:     make(chan []byte, 256),
		channels: make(map[string]bool),
	}

	s.register <- client

	// Start client goroutines
	go client.writePump()
	go client.readPump()

	// Send welcome message
	welcome := Message{
		Type:      "welcome",
		Data:      map[string]interface{}{"id": client.id},
		Timestamp: time.Now().Unix(),
	}
	client.sendMessage(welcome)
}

// handleHealth provides health check endpoint
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "healthy",
		"clients":  atomic.LoadInt32(&s.clientCount),
		"messages": atomic.LoadUint64(&s.messagesOut),
	})
}

// readPump handles incoming messages from client
func (c *Client) readPump() {
	defer func() {
		c.server.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(512 * 1024)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		var msg json.RawMessage
		err := c.conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.server.logger.Error("WebSocket read error", "error", err)
			}
			break
		}

		// Process message
		c.handleMessage(msg)
	}
}

// writePump handles outgoing messages to client
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
			atomic.AddUint64(&c.server.messagesOut, 1)

			// Drain queued messages
			n := len(c.send)
			for i := 0; i < n; i++ {
				c.conn.WriteMessage(websocket.TextMessage, <-c.send)
				atomic.AddUint64(&c.server.messagesOut, 1)
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// handleMessage processes incoming client messages
func (c *Client) handleMessage(raw json.RawMessage) {
	var msg map[string]interface{}
	if err := json.Unmarshal(raw, &msg); err != nil {
		c.sendError("Invalid message format")
		return
	}

	msgType, ok := msg["type"].(string)
	if !ok {
		c.sendError("Missing message type")
		return
	}

	switch msgType {
	case "subscribe":
		c.handleSubscribe(msg)
	case "unsubscribe":
		c.handleUnsubscribe(msg)
	case "ping":
		c.sendMessage(Message{Type: "pong", Timestamp: time.Now().Unix()})
	default:
		c.sendError(fmt.Sprintf("Unknown message type: %s", msgType))
	}
}

// handleSubscribe handles subscription requests
func (c *Client) handleSubscribe(msg map[string]interface{}) {
	channels, ok := msg["channels"].([]interface{})
	if !ok {
		c.sendError("Invalid channels format")
		return
	}

	for _, ch := range channels {
		channel, ok := ch.(string)
		if !ok {
			continue
		}

		c.mu.Lock()
		c.channels[channel] = true
		c.mu.Unlock()

		c.server.subscribe(channel, c)

		// Send initial snapshot for order book channels
		if len(channel) > 10 && channel[:10] == "orderbook:" {
			symbol := channel[10:]
			c.sendOrderBookSnapshot(symbol)
		}
	}

	c.sendMessage(Message{
		Type:      "subscribed",
		Data:      map[string]interface{}{"channels": channels},
		Timestamp: time.Now().Unix(),
	})
}

// handleUnsubscribe handles unsubscription requests
func (c *Client) handleUnsubscribe(msg map[string]interface{}) {
	channels, ok := msg["channels"].([]interface{})
	if !ok {
		c.sendError("Invalid channels format")
		return
	}

	for _, ch := range channels {
		channel, ok := ch.(string)
		if !ok {
			continue
		}

		c.mu.Lock()
		delete(c.channels, channel)
		c.mu.Unlock()

		c.server.unsubscribe(channel, c)
	}

	c.sendMessage(Message{
		Type:      "unsubscribed",
		Data:      map[string]interface{}{"channels": channels},
		Timestamp: time.Now().Unix(),
	})
}

// sendOrderBookSnapshot sends current order book state
func (c *Client) sendOrderBookSnapshot(symbol string) {
	book := c.server.orderBook.GetOrderBookSnapshot(symbol, 20)

	update := OrderBookUpdate{
		Type:      "snapshot",
		Symbol:    symbol,
		Bids:      book.Bids,
		Asks:      book.Asks,
		Timestamp: time.Now().Unix(),
		Sequence:  0, // TODO: Add sequence tracking
	}

	c.sendMessage(Message{
		Type:      "orderbook",
		Channel:   fmt.Sprintf("orderbook:%s", symbol),
		Data:      update,
		Timestamp: time.Now().Unix(),
	})
}

// sendMessage sends a message to the client
func (c *Client) sendMessage(msg Message) {
	data, err := json.Marshal(msg)
	if err != nil {
		c.server.logger.Error("Failed to marshal message", "error", err)
		return
	}

	select {
	case c.send <- data:
	default:
		// Client send channel is full, close connection
		c.server.unregister <- c
		close(c.send)
	}
}

// sendError sends an error message to the client
func (c *Client) sendError(message string) {
	c.sendMessage(Message{
		Type:      "error",
		Data:      map[string]interface{}{"message": message},
		Timestamp: time.Now().Unix(),
	})
}

// subscribe adds a client to a channel
func (s *Server) subscribe(channel string, client *Client) {
	s.subMu.Lock()
	defer s.subMu.Unlock()

	if s.subscriptions[channel] == nil {
		s.subscriptions[channel] = make(map[*Client]bool)
	}
	s.subscriptions[channel][client] = true
}

// unsubscribe removes a client from a channel
func (s *Server) unsubscribe(channel string, client *Client) {
	s.subMu.Lock()
	defer s.subMu.Unlock()

	if clients, ok := s.subscriptions[channel]; ok {
		delete(clients, client)
		if len(clients) == 0 {
			delete(s.subscriptions, channel)
		}
	}
}

// unsubscribeAll removes a client from all channels
func (s *Server) unsubscribeAll(client *Client) {
	s.subMu.Lock()
	defer s.subMu.Unlock()

	for channel, clients := range s.subscriptions {
		delete(clients, client)
		if len(clients) == 0 {
			delete(s.subscriptions, channel)
		}
	}
}

// broadcastMessage sends a message to all subscribed clients
func (s *Server) broadcastMessage(msg Message) {
	s.subMu.RLock()
	clients := s.subscriptions[msg.Channel]
	s.subMu.RUnlock()

	if len(clients) == 0 {
		return
	}

	data, err := json.Marshal(msg)
	if err != nil {
		s.logger.Error("Failed to marshal broadcast message", "error", err)
		return
	}

	for client := range clients {
		select {
		case client.send <- data:
			atomic.AddUint64(&s.messagesOut, 1)
		default:
			// Client's send channel is full, close it
			s.unregister <- client
			close(client.send)
		}
	}
}

// BroadcastOrderBookUpdate broadcasts an order book update
func (s *Server) BroadcastOrderBookUpdate(symbol string, bids, asks []lx.PriceLevel) {
	update := OrderBookUpdate{
		Type:      "update",
		Symbol:    symbol,
		Bids:      bids,
		Asks:      asks,
		Timestamp: time.Now().Unix(),
		Sequence:  atomic.AddUint64(&s.messagesOut, 1),
	}

	s.broadcast <- Message{
		Type:      "orderbook",
		Channel:   fmt.Sprintf("orderbook:%s", symbol),
		Data:      update,
		Timestamp: time.Now().Unix(),
		Sequence:  update.Sequence,
	}
}

// BroadcastTrade broadcasts a trade
func (s *Server) BroadcastTrade(trade *lx.Trade) {
	tradeUpdate := TradeUpdate{
		TradeID:     trade.ID,
		Symbol:      "BTC-USD", // TODO: Get from trade
		Price:       trade.Price,
		Size:        trade.Size,
		Side:        "buy", // TODO: Determine from trade
		BuyOrderID:  trade.BuyOrderID,
		SellOrderID: trade.SellOrderID,
		Timestamp:   trade.Timestamp,
	}

	s.broadcast <- Message{
		Type:      "trade",
		Channel:   fmt.Sprintf("trades:%s", "BTC-USD"),
		Data:      tradeUpdate,
		Timestamp: time.Now().Unix(),
	}
}

// GetStats returns server statistics
func (s *Server) GetStats() map[string]interface{} {
	s.subMu.RLock()
	numChannels := len(s.subscriptions)
	s.subMu.RUnlock()

	return map[string]interface{}{
		"clients":       atomic.LoadInt32(&s.clientCount),
		"messages_sent": atomic.LoadUint64(&s.messagesOut),
		"channels":      numChannels,
	}
}

// generateClientID generates a unique client ID
func generateClientID() string {
	return fmt.Sprintf("client-%d-%d", time.Now().Unix(), time.Now().Nanosecond())
}
