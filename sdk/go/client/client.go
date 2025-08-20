// Package client provides a Go SDK for interacting with LX DEX
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	pb "github.com/luxfi/dex/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is the main client for interacting with LX DEX
type Client struct {
	// Configuration
	jsonRPCURL string
	wsURL      string
	grpcURL    string
	apiKey     string

	// JSON-RPC
	httpClient *http.Client
	idCounter  uint64

	// WebSocket
	wsConn      *websocket.Conn
	wsCallbacks map[string]func(interface{})
	wsMu        sync.RWMutex
	wsRunning   bool
	wsStop      chan struct{}

	// gRPC
	grpcConn   *grpc.ClientConn
	grpcClient pb.LXDEXServiceClient

	mu sync.RWMutex
}

// NewClient creates a new LX DEX client
func NewClient(opts ...Option) (*Client, error) {
	c := &Client{
		jsonRPCURL:  "http://localhost:8080",
		wsURL:       "ws://localhost:8081",
		grpcURL:     "localhost:50051",
		httpClient:  &http.Client{Timeout: 30 * time.Second},
		wsCallbacks: make(map[string]func(interface{})),
		wsStop:      make(chan struct{}),
	}

	// Apply options
	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// Option is a client configuration option
type Option func(*Client)

// WithJSONRPCURL sets the JSON-RPC URL
func WithJSONRPCURL(url string) Option {
	return func(c *Client) {
		c.jsonRPCURL = url
	}
}

// WithWebSocketURL sets the WebSocket URL
func WithWebSocketURL(url string) Option {
	return func(c *Client) {
		c.wsURL = url
	}
}

// WithGRPCURL sets the gRPC URL
func WithGRPCURL(url string) Option {
	return func(c *Client) {
		c.grpcURL = url
	}
}

// WithAPIKey sets the API key for authentication
func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.apiKey = key
	}
}

// ConnectGRPC establishes a gRPC connection
func (c *Client) ConnectGRPC(ctx context.Context) error {
	conn, err := grpc.DialContext(ctx, c.grpcURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to gRPC: %w", err)
	}

	c.grpcConn = conn
	c.grpcClient = pb.NewLXDEXServiceClient(conn)
	return nil
}

// ConnectWebSocket establishes a WebSocket connection
func (c *Client) ConnectWebSocket(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.wsConn != nil {
		return nil // Already connected
	}

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.DialContext(ctx, c.wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to WebSocket: %w", err)
	}

	c.wsConn = conn
	c.wsRunning = true

	// Start message handler
	go c.handleWebSocketMessages()

	return nil
}

// handleWebSocketMessages processes incoming WebSocket messages
func (c *Client) handleWebSocketMessages() {
	defer func() {
		c.mu.Lock()
		c.wsRunning = false
		c.wsConn.Close()
		c.wsConn = nil
		c.mu.Unlock()
	}()

	for {
		select {
		case <-c.wsStop:
			return
		default:
			c.wsConn.SetReadDeadline(time.Now().Add(60 * time.Second))

			var msg map[string]interface{}
			err := c.wsConn.ReadJSON(&msg)
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					fmt.Printf("WebSocket error: %v\n", err)
				}
				return
			}

			// Handle message
			if channel, ok := msg["channel"].(string); ok {
				c.wsMu.RLock()
				if callback, exists := c.wsCallbacks[channel]; exists {
					c.wsMu.RUnlock()
					go callback(msg["data"])
				} else {
					c.wsMu.RUnlock()
				}
			}
		}
	}
}

// Disconnect closes all connections
func (c *Client) Disconnect() error {
	// Close WebSocket
	if c.wsRunning {
		close(c.wsStop)
		c.mu.Lock()
		if c.wsConn != nil {
			c.wsConn.Close()
		}
		c.mu.Unlock()
	}

	// Close gRPC
	if c.grpcConn != nil {
		c.grpcConn.Close()
	}

	return nil
}

// PlaceOrder places a new order
func (c *Client) PlaceOrder(ctx context.Context, order *Order) (*OrderResponse, error) {
	// Use gRPC if connected
	if c.grpcClient != nil {
		req := &pb.PlaceOrderRequest{
			Symbol:      order.Symbol,
			Type:        pb.OrderType(order.Type),
			Side:        pb.OrderSide(order.Side),
			Price:       order.Price,
			Size:        order.Size,
			UserId:      order.UserID,
			ClientId:    order.ClientID,
			TimeInForce: order.TimeInForce,
			PostOnly:    order.PostOnly,
			ReduceOnly:  order.ReduceOnly,
		}

		resp, err := c.grpcClient.PlaceOrder(ctx, req)
		if err != nil {
			return nil, err
		}

		return &OrderResponse{
			OrderID: resp.OrderId,
			Status:  resp.Status,
			Message: resp.Message,
		}, nil
	}

	// Fallback to JSON-RPC
	params := map[string]interface{}{
		"symbol":      order.Symbol,
		"type":        order.Type,
		"side":        order.Side,
		"price":       order.Price,
		"size":        order.Size,
		"userID":      order.UserID,
		"clientID":    order.ClientID,
		"timeInForce": order.TimeInForce,
		"postOnly":    order.PostOnly,
		"reduceOnly":  order.ReduceOnly,
	}

	var result map[string]interface{}
	if err := c.callJSONRPC(ctx, "lx_placeOrder", params, &result); err != nil {
		return nil, err
	}

	return &OrderResponse{
		OrderID: uint64(result["orderId"].(float64)),
		Status:  result["status"].(string),
	}, nil
}

// CancelOrder cancels an existing order
func (c *Client) CancelOrder(ctx context.Context, orderID uint64) error {
	// Use gRPC if connected
	if c.grpcClient != nil {
		_, err := c.grpcClient.CancelOrder(ctx, &pb.CancelOrderRequest{
			OrderId: orderID,
		})
		return err
	}

	// Fallback to JSON-RPC
	params := map[string]interface{}{
		"orderId": orderID,
	}

	var result map[string]interface{}
	return c.callJSONRPC(ctx, "lx_cancelOrder", params, &result)
}

// GetOrderBook retrieves the order book for a symbol
func (c *Client) GetOrderBook(ctx context.Context, symbol string, depth int32) (*OrderBook, error) {
	// Use gRPC if connected
	if c.grpcClient != nil {
		resp, err := c.grpcClient.GetOrderBook(ctx, &pb.GetOrderBookRequest{
			Symbol: symbol,
			Depth:  depth,
		})
		if err != nil {
			return nil, err
		}

		ob := &OrderBook{
			Symbol:    resp.Symbol,
			Timestamp: resp.Timestamp,
			Bids:      make([]PriceLevel, len(resp.Bids)),
			Asks:      make([]PriceLevel, len(resp.Asks)),
		}

		for i, bid := range resp.Bids {
			ob.Bids[i] = PriceLevel{
				Price: bid.Price,
				Size:  bid.Size,
			}
		}

		for i, ask := range resp.Asks {
			ob.Asks[i] = PriceLevel{
				Price: ask.Price,
				Size:  ask.Size,
			}
		}

		return ob, nil
	}

	// Fallback to JSON-RPC
	params := map[string]interface{}{
		"symbol": symbol,
		"depth":  depth,
	}

	var result map[string]interface{}
	if err := c.callJSONRPC(ctx, "lx_getOrderBook", params, &result); err != nil {
		return nil, err
	}

	ob := &OrderBook{
		Symbol:    result["Symbol"].(string),
		Timestamp: int64(result["Timestamp"].(float64)),
	}

	// Parse bids
	bids := result["Bids"].([]interface{})
	ob.Bids = make([]PriceLevel, len(bids))
	for i, bid := range bids {
		b := bid.(map[string]interface{})
		ob.Bids[i] = PriceLevel{
			Price: b["Price"].(float64),
			Size:  b["Size"].(float64),
		}
	}

	// Parse asks
	asks := result["Asks"].([]interface{})
	ob.Asks = make([]PriceLevel, len(asks))
	for i, ask := range asks {
		a := ask.(map[string]interface{})
		ob.Asks[i] = PriceLevel{
			Price: a["Price"].(float64),
			Size:  a["Size"].(float64),
		}
	}

	return ob, nil
}

// GetTrades retrieves recent trades
func (c *Client) GetTrades(ctx context.Context, symbol string, limit int32) ([]*Trade, error) {
	// Use gRPC if connected
	if c.grpcClient != nil {
		resp, err := c.grpcClient.GetTrades(ctx, &pb.GetTradesRequest{
			Symbol: symbol,
			Limit:  limit,
		})
		if err != nil {
			return nil, err
		}

		trades := make([]*Trade, len(resp.Trades))
		for i, t := range resp.Trades {
			trades[i] = &Trade{
				TradeID:     t.TradeId,
				Symbol:      t.Symbol,
				Price:       t.Price,
				Size:        t.Size,
				Side:        OrderSide(t.Side),
				BuyOrderID:  t.BuyOrderId,
				SellOrderID: t.SellOrderId,
				BuyerID:     t.BuyerId,
				SellerID:    t.SellerId,
				Timestamp:   t.Timestamp,
			}
		}

		return trades, nil
	}

	// Fallback to JSON-RPC
	params := map[string]interface{}{
		"symbol": symbol,
		"limit":  limit,
	}

	var result []interface{}
	if err := c.callJSONRPC(ctx, "lx_getTrades", params, &result); err != nil {
		return nil, err
	}

	trades := make([]*Trade, len(result))
	for i, t := range result {
		trade := t.(map[string]interface{})
		trades[i] = &Trade{
			TradeID:     uint64(trade["tradeId"].(float64)),
			Symbol:      trade["symbol"].(string),
			Price:       trade["price"].(float64),
			Size:        trade["size"].(float64),
			Side:        OrderSide(trade["side"].(float64)),
			BuyOrderID:  uint64(trade["buyOrderId"].(float64)),
			SellOrderID: uint64(trade["sellOrderId"].(float64)),
			BuyerID:     trade["buyerId"].(string),
			SellerID:    trade["sellerId"].(string),
			Timestamp:   int64(trade["timestamp"].(float64)),
		}
	}

	return trades, nil
}

// Subscribe subscribes to a WebSocket channel
func (c *Client) Subscribe(channel string, callback func(interface{})) error {
	c.wsMu.Lock()
	c.wsCallbacks[channel] = callback
	c.wsMu.Unlock()

	// Send subscription message
	msg := map[string]interface{}{
		"type":    "subscribe",
		"channel": channel,
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.wsConn == nil {
		return fmt.Errorf("WebSocket not connected")
	}

	return c.wsConn.WriteJSON(msg)
}

// Unsubscribe unsubscribes from a WebSocket channel
func (c *Client) Unsubscribe(channel string) error {
	c.wsMu.Lock()
	delete(c.wsCallbacks, channel)
	c.wsMu.Unlock()

	// Send unsubscription message
	msg := map[string]interface{}{
		"type":    "unsubscribe",
		"channel": channel,
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.wsConn == nil {
		return nil // Not connected, nothing to do
	}

	return c.wsConn.WriteJSON(msg)
}

// SubscribeOrderBook subscribes to order book updates
func (c *Client) SubscribeOrderBook(symbol string, callback func(*OrderBook)) error {
	return c.Subscribe(fmt.Sprintf("orderbook:%s", symbol), func(data interface{}) {
		// Convert data to OrderBook
		if m, ok := data.(map[string]interface{}); ok {
			ob := &OrderBook{
				Symbol:    m["symbol"].(string),
				Timestamp: int64(m["timestamp"].(float64)),
			}
			// Parse bids and asks
			// ... conversion logic ...
			callback(ob)
		}
	})
}

// SubscribeTrades subscribes to trade updates
func (c *Client) SubscribeTrades(symbol string, callback func(*Trade)) error {
	return c.Subscribe(fmt.Sprintf("trades:%s", symbol), func(data interface{}) {
		// Convert data to Trade
		if m, ok := data.(map[string]interface{}); ok {
			trade := &Trade{
				TradeID:   uint64(m["tradeId"].(float64)),
				Symbol:    m["symbol"].(string),
				Price:     m["price"].(float64),
				Size:      m["size"].(float64),
				Timestamp: int64(m["timestamp"].(float64)),
			}
			callback(trade)
		}
	})
}

// StreamOrderBook streams order book updates via gRPC
func (c *Client) StreamOrderBook(ctx context.Context, symbol string) (<-chan *OrderBook, error) {
	if c.grpcClient == nil {
		return nil, fmt.Errorf("gRPC not connected")
	}

	stream, err := c.grpcClient.StreamOrderBook(ctx, &pb.StreamOrderBookRequest{
		Symbol: symbol,
	})
	if err != nil {
		return nil, err
	}

	ch := make(chan *OrderBook, 100)

	go func() {
		defer close(ch)
		for {
			update, err := stream.Recv()
			if err != nil {
				return
			}

			ob := &OrderBook{
				Symbol:    update.Symbol,
				Timestamp: update.Timestamp,
				Bids:      make([]PriceLevel, len(update.Bids)),
				Asks:      make([]PriceLevel, len(update.Asks)),
			}

			for i, bid := range update.Bids {
				ob.Bids[i] = PriceLevel{
					Price: bid.Price,
					Size:  bid.Size,
				}
			}

			for i, ask := range update.Asks {
				ob.Asks[i] = PriceLevel{
					Price: ask.Price,
					Size:  ask.Size,
				}
			}

			select {
			case ch <- ob:
			case <-ctx.Done():
				return
			}
		}
	}()

	return ch, nil
}

// callJSONRPC makes a JSON-RPC call
func (c *Client) callJSONRPC(ctx context.Context, method string, params interface{}, result interface{}) error {
	id := atomic.AddUint64(&c.idCounter, 1)

	request := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  method,
		"params":  params,
		"id":      id,
	}

	reqBody, err := json.Marshal(request)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.jsonRPCURL+"/rpc", bytes.NewBuffer(reqBody))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var response struct {
		Result interface{} `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		ID uint64 `json:"id"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return err
	}

	if response.Error != nil {
		return fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	// Convert response.Result to the expected type
	if result != nil {
		resultBytes, _ := json.Marshal(response.Result)
		return json.Unmarshal(resultBytes, result)
	}

	return nil
}

// Ping checks if the server is responsive
func (c *Client) Ping(ctx context.Context) error {
	var result string
	return c.callJSONRPC(ctx, "lx_ping", nil, &result)
}

// GetInfo retrieves node information
func (c *Client) GetInfo(ctx context.Context) (*NodeInfo, error) {
	var result NodeInfo
	if err := c.callJSONRPC(ctx, "lx_getInfo", nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
