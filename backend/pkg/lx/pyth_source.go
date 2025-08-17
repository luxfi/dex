package lx

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// PythPriceSource connects to Pyth Network for real-time price feeds
type PythPriceSource struct {
	// Connection
	wsURL           string
	httpURL         string
	conn            *websocket.Conn
	httpClient      *http.Client
	
	// Price IDs for different assets
	priceIDs        map[string]string
	
	// Cached prices
	prices          map[string]*PriceData
	lastUpdate      map[string]time.Time
	
	// Subscriptions
	subscriptions   map[string]bool
	
	// Health monitoring
	healthy         bool
	lastHeartbeat   time.Time
	reconnectDelay  time.Duration
	maxReconnect    int
	
	// Control
	mu              sync.RWMutex
	done            chan struct{}
}

// PythPriceFeed represents a price update from Pyth
type PythPriceFeed struct {
	ID              string  `json:"id"`
	Price           float64 `json:"price"`
	Confidence      float64 `json:"conf"`
	ExponentPrice   int32   `json:"expo"`
	PublishTime     int64   `json:"publish_time"`
	EMAPrice        float64 `json:"ema_price"`
	EMAConfidence   float64 `json:"ema_conf"`
	Status          string  `json:"status"`
}

// NewPythPriceSource creates a new Pyth price source
func NewPythPriceSource(wsURL, httpURL string) *PythPriceSource {
	return &PythPriceSource{
		wsURL:          wsURL,
		httpURL:        httpURL,
		httpClient:     &http.Client{Timeout: 5 * time.Second},
		priceIDs:       initPythPriceIDs(),
		prices:         make(map[string]*PriceData),
		lastUpdate:     make(map[string]time.Time),
		subscriptions:  make(map[string]bool),
		healthy:        false,
		reconnectDelay: 1 * time.Second,
		maxReconnect:   10,
		done:          make(chan struct{}),
	}
}

// initPythPriceIDs initializes Pyth price feed IDs for major assets
func initPythPriceIDs() map[string]string {
	return map[string]string{
		"BTC-USDT":  "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43", // BTC/USD
		"ETH-USDT":  "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace", // ETH/USD
		"BNB-USDT":  "0x2f95862b045670cd22bee3114c39763a4a08beeb663b145d283c31d7d1101c4f", // BNB/USD
		"SOL-USDT":  "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d", // SOL/USD
		"AVAX-USDT": "0x93da3352f9f1d105fdfe4971cfa80e9dd777bfc5d0f683ebb6e1294b92137bb7", // AVAX/USD
		"MATIC-USDT": "0x5de33a9112c2b700b8d30b8a3402c103578ccfa2765696471cc672bd5cf6ac52", // MATIC/USD
		"ARB-USDT":  "0x3fa4252848f9f0a1480be62745a4629d9eb1322aebab8a791e344b3b9c1adcf5", // ARB/USD
		"OP-USDT":   "0x385f64d993f7b77d8182ed5003d97c60aa3361f3cecfe711544d2d59165e9bdf", // OP/USD
	}
}

// Connect establishes WebSocket connection to Pyth
func (ps *PythPriceSource) Connect() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	
	// Close existing connection
	if ps.conn != nil {
		ps.conn.Close()
	}
	
	// Establish new connection
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}
	
	conn, _, err := dialer.Dial(ps.wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to Pyth: %w", err)
	}
	
	ps.conn = conn
	ps.healthy = true
	ps.lastHeartbeat = time.Now()
	
	// Start reading messages
	go ps.readMessages()
	
	// Start heartbeat
	go ps.heartbeat()
	
	// Subscribe to all configured price feeds
	for symbol := range ps.priceIDs {
		ps.subscribeToPriceFeed(symbol)
	}
	
	return nil
}

// readMessages reads messages from WebSocket
func (ps *PythPriceSource) readMessages() {
	defer func() {
		ps.mu.Lock()
		ps.healthy = false
		ps.mu.Unlock()
		ps.reconnect()
	}()
	
	for {
		select {
		case <-ps.done:
			return
		default:
			var msg map[string]interface{}
			err := ps.conn.ReadJSON(&msg)
			if err != nil {
				return
			}
			
			ps.handleMessage(msg)
		}
	}
}

// handleMessage processes incoming Pyth messages
func (ps *PythPriceSource) handleMessage(msg map[string]interface{}) {
	msgType, ok := msg["type"].(string)
	if !ok {
		return
	}
	
	switch msgType {
	case "price_update":
		ps.handlePriceUpdate(msg)
	case "heartbeat":
		ps.mu.Lock()
		ps.lastHeartbeat = time.Now()
		ps.mu.Unlock()
	}
}

// handlePriceUpdate processes price updates
func (ps *PythPriceSource) handlePriceUpdate(msg map[string]interface{}) {
	data, ok := msg["data"].(map[string]interface{})
	if !ok {
		return
	}
	
	priceID, ok := data["price_id"].(string)
	if !ok {
		return
	}
	
	// Find symbol for this price ID
	var symbol string
	for sym, id := range ps.priceIDs {
		if id == priceID {
			symbol = sym
			break
		}
	}
	
	if symbol == "" {
		return
	}
	
	// Parse price data
	price, _ := data["price"].(float64)
	confidence, _ := data["confidence"].(float64)
	publishTime, _ := data["publish_time"].(float64)
	
	// Adjust for exponent if present
	if expo, ok := data["expo"].(float64); ok {
		price = price * math.Pow(10, expo)
	}
	
	// Update cached price
	ps.mu.Lock()
	ps.prices[symbol] = &PriceData{
		Symbol:     symbol,
		Price:      price,
		Confidence: 1.0 - (confidence / price), // Convert confidence interval to confidence score
		Timestamp:  time.Unix(int64(publishTime), 0),
		Source:     "pyth",
	}
	ps.lastUpdate[symbol] = time.Now()
	ps.mu.Unlock()
}

// GetPrice returns the latest price for a symbol
func (ps *PythPriceSource) GetPrice(symbol string) (*PriceData, error) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	
	price, exists := ps.prices[symbol]
	if !exists {
		// Try to fetch via HTTP API
		return ps.fetchPriceHTTP(symbol)
	}
	
	// Check staleness
	if time.Since(ps.lastUpdate[symbol]) > 30*time.Second {
		price.IsStale = true
	}
	
	return price, nil
}

// fetchPriceHTTP fetches price via HTTP API fallback
func (ps *PythPriceSource) fetchPriceHTTP(symbol string) (*PriceData, error) {
	priceID, exists := ps.priceIDs[symbol]
	if !exists {
		return nil, fmt.Errorf("no price ID for symbol %s", symbol)
	}
	
	url := fmt.Sprintf("%s/api/latest_price_feeds?ids[]=%s", ps.httpURL, priceID)
	resp, err := ps.httpClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var result []PythPriceFeed
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	if len(result) == 0 {
		return nil, errors.New("no price data returned")
	}
	
	feed := result[0]
	price := feed.Price * math.Pow(10, float64(feed.ExponentPrice))
	
	return &PriceData{
		Symbol:     symbol,
		Price:      price,
		Confidence: 1.0 - (feed.Confidence / price),
		Timestamp:  time.Unix(feed.PublishTime, 0),
		Source:     "pyth",
	}, nil
}

// GetPrices returns prices for multiple symbols
func (ps *PythPriceSource) GetPrices(symbols []string) (map[string]*PriceData, error) {
	prices := make(map[string]*PriceData)
	
	for _, symbol := range symbols {
		price, err := ps.GetPrice(symbol)
		if err == nil {
			prices[symbol] = price
		}
	}
	
	return prices, nil
}

// Subscribe subscribes to price updates for a symbol
func (ps *PythPriceSource) Subscribe(symbol string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	
	if ps.subscriptions[symbol] {
		return nil // Already subscribed
	}
	
	ps.subscriptions[symbol] = true
	return ps.subscribeToPriceFeed(symbol)
}

// subscribeToPriceFeed sends subscription message
func (ps *PythPriceSource) subscribeToPriceFeed(symbol string) error {
	priceID, exists := ps.priceIDs[symbol]
	if !exists {
		return fmt.Errorf("no price ID for symbol %s", symbol)
	}
	
	if ps.conn == nil {
		return errors.New("not connected")
	}
	
	msg := map[string]interface{}{
		"type": "subscribe",
		"ids":  []string{priceID},
	}
	
	return ps.conn.WriteJSON(msg)
}

// Unsubscribe unsubscribes from price updates
func (ps *PythPriceSource) Unsubscribe(symbol string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	
	delete(ps.subscriptions, symbol)
	
	if ps.conn == nil {
		return nil
	}
	
	priceID, exists := ps.priceIDs[symbol]
	if !exists {
		return nil
	}
	
	msg := map[string]interface{}{
		"type": "unsubscribe",
		"ids":  []string{priceID},
	}
	
	return ps.conn.WriteJSON(msg)
}

// IsHealthy returns the health status
func (ps *PythPriceSource) IsHealthy() bool {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	
	if !ps.healthy {
		return false
	}
	
	// Check heartbeat
	if time.Since(ps.lastHeartbeat) > 30*time.Second {
		return false
	}
	
	return true
}

// GetName returns the source name
func (ps *PythPriceSource) GetName() string {
	return "pyth"
}

// GetWeight returns the source weight for aggregation
func (ps *PythPriceSource) GetWeight() float64 {
	return 1.5 // Higher weight for Pyth due to high frequency updates
}

// heartbeat sends periodic heartbeats
func (ps *PythPriceSource) heartbeat() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ps.done:
			return
		case <-ticker.C:
			ps.mu.RLock()
			conn := ps.conn
			ps.mu.RUnlock()
			
			if conn != nil {
				msg := map[string]string{"type": "heartbeat"}
				conn.WriteJSON(msg)
			}
		}
	}
}

// reconnect attempts to reconnect on failure
func (ps *PythPriceSource) reconnect() {
	attempts := 0
	for attempts < ps.maxReconnect {
		select {
		case <-ps.done:
			return
		case <-time.After(ps.reconnectDelay):
			attempts++
			if err := ps.Connect(); err == nil {
				return
			}
			ps.reconnectDelay = time.Duration(math.Min(float64(ps.reconnectDelay*2), float64(30*time.Second)))
		}
	}
}

// Close closes the connection
func (ps *PythPriceSource) Close() error {
	close(ps.done)
	
	ps.mu.Lock()
	defer ps.mu.Unlock()
	
	if ps.conn != nil {
		return ps.conn.Close()
	}
	return nil
}