// LX DEX Trading Client
//
// Multi-protocol programmatic trading client for LX DEX.
// Supports WebSocket and gRPC protocols with unified interface.
//
// Usage:
//   - As package: import "github.com/luxfi/dex/client/go/lxclient"
//   - As CLI: ./lx-client [flags] [command]
//   - Interactive: ./lx-client -i
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/luxfi/dex/pkg/grpc/pb"
)

// Protocol identifies the transport protocol
type Protocol string

const (
	ProtocolWS   Protocol = "ws"
	ProtocolGRPC Protocol = "grpc"
)

// Order represents a trading order
type Order struct {
	Symbol string  `json:"symbol"`
	Side   string  `json:"side"`
	Type   string  `json:"type"`
	Price  float64 `json:"price"`
	Size   float64 `json:"size"`
}

// OrderResponse represents order operation response
type OrderResponse struct {
	OrderID uint64 `json:"order_id"`
	Status  string `json:"status"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}

// Position represents an open position
type Position struct {
	Symbol     string  `json:"symbol"`
	Size       float64 `json:"size"`
	EntryPrice float64 `json:"entry_price"`
	MarkPrice  float64 `json:"mark_price"`
	PnL        float64 `json:"pnl"`
}

// Client defines the trading client interface
type Client interface {
	// Trading operations
	PlaceOrder(ctx context.Context, order *Order) (*OrderResponse, error)
	CancelOrder(ctx context.Context, orderID uint64) error
	GetOrders(ctx context.Context) ([]Order, error)
	GetPositions(ctx context.Context) ([]Position, error)

	// Market data
	Subscribe(ctx context.Context, symbol string) error

	// Connection management
	Protocol() Protocol
	Close() error
}

// ----- WebSocket Client Implementation -----

// WsMessage represents a WebSocket message
type WsMessage struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
	RequestID string                 `json:"request_id,omitempty"`
	Timestamp int64                  `json:"timestamp,omitempty"`
}

// WsClient implements Client over WebSocket
type WsClient struct {
	conn       *websocket.Conn
	mu         sync.Mutex
	reqCounter int
	verbose    bool
	responses  chan WsMessage
	url        string
}

// NewWsClient creates a WebSocket client
func NewWsClient(url string, verbose bool) (*WsClient, error) {
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(url, nil)
	if err != nil {
		return nil, fmt.Errorf("dial failed: %w", err)
	}

	c := &WsClient{
		conn:      conn,
		verbose:   verbose,
		responses: make(chan WsMessage, 100),
		url:       url,
	}

	go c.readLoop()
	return c, nil
}

func (c *WsClient) readLoop() {
	for {
		var msg WsMessage
		err := c.conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				fmt.Fprintf(os.Stderr, "ws read error: %v\n", err)
			}
			close(c.responses)
			return
		}

		if c.verbose {
			data, _ := json.MarshalIndent(msg, "", "  ")
			fmt.Printf("<< %s\n", data)
		}

		c.responses <- msg
	}
}

func (c *WsClient) send(msgType string, data map[string]interface{}) (string, error) {
	c.mu.Lock()
	c.reqCounter++
	reqID := fmt.Sprintf("req-%d", c.reqCounter)
	c.mu.Unlock()

	msg := map[string]interface{}{
		"type":       msgType,
		"request_id": reqID,
	}
	for k, v := range data {
		msg[k] = v
	}

	if c.verbose {
		jsonData, _ := json.MarshalIndent(msg, "", "  ")
		fmt.Printf(">> %s\n", jsonData)
	}

	c.mu.Lock()
	err := c.conn.WriteJSON(msg)
	c.mu.Unlock()
	return reqID, err
}

func (c *WsClient) waitResponse(reqID string, timeout time.Duration) (*WsMessage, error) {
	deadline := time.After(timeout)
	for {
		select {
		case msg, ok := <-c.responses:
			if !ok {
				return nil, fmt.Errorf("connection closed")
			}
			if msg.RequestID == reqID {
				return &msg, nil
			}
		case <-deadline:
			return nil, fmt.Errorf("timeout waiting for response")
		}
	}
}

// Protocol returns the protocol type
func (c *WsClient) Protocol() Protocol {
	return ProtocolWS
}

// PlaceOrder places an order via WebSocket
func (c *WsClient) PlaceOrder(ctx context.Context, order *Order) (*OrderResponse, error) {
	reqID, err := c.send("place_order", map[string]interface{}{
		"order": order,
	})
	if err != nil {
		return nil, err
	}

	msg, err := c.waitResponse(reqID, 5*time.Second)
	if err != nil {
		return nil, err
	}

	if msg.Error != "" {
		return nil, fmt.Errorf("place order failed: %s", msg.Error)
	}

	resp := &OrderResponse{
		Status:  "submitted",
		Message: "Order placed successfully",
	}
	if oid, ok := msg.Data["order_id"].(float64); ok {
		resp.OrderID = uint64(oid)
	}
	return resp, nil
}

// CancelOrder cancels an order via WebSocket
func (c *WsClient) CancelOrder(ctx context.Context, orderID uint64) error {
	reqID, err := c.send("cancel_order", map[string]interface{}{
		"orderID": orderID,
	})
	if err != nil {
		return err
	}

	msg, err := c.waitResponse(reqID, 5*time.Second)
	if err != nil {
		return err
	}

	if msg.Error != "" {
		return fmt.Errorf("cancel order failed: %s", msg.Error)
	}
	return nil
}

// GetOrders retrieves open orders via WebSocket
func (c *WsClient) GetOrders(ctx context.Context) ([]Order, error) {
	reqID, err := c.send("get_orders", nil)
	if err != nil {
		return nil, err
	}

	msg, err := c.waitResponse(reqID, 5*time.Second)
	if err != nil {
		return nil, err
	}

	if msg.Error != "" {
		return nil, fmt.Errorf("get orders failed: %s", msg.Error)
	}

	// Parse orders from response
	var orders []Order
	if ordersData, ok := msg.Data["orders"].([]interface{}); ok {
		for _, o := range ordersData {
			if om, ok := o.(map[string]interface{}); ok {
				orders = append(orders, Order{
					Symbol: getString(om, "symbol"),
					Side:   getString(om, "side"),
					Type:   getString(om, "type"),
					Price:  getFloat(om, "price"),
					Size:   getFloat(om, "size"),
				})
			}
		}
	}
	return orders, nil
}

// GetPositions retrieves positions via WebSocket
func (c *WsClient) GetPositions(ctx context.Context) ([]Position, error) {
	reqID, err := c.send("get_positions", nil)
	if err != nil {
		return nil, err
	}

	msg, err := c.waitResponse(reqID, 5*time.Second)
	if err != nil {
		return nil, err
	}

	if msg.Error != "" {
		return nil, fmt.Errorf("get positions failed: %s", msg.Error)
	}

	var positions []Position
	if posData, ok := msg.Data["positions"].([]interface{}); ok {
		for _, p := range posData {
			if pm, ok := p.(map[string]interface{}); ok {
				positions = append(positions, Position{
					Symbol:     getString(pm, "symbol"),
					Size:       getFloat(pm, "size"),
					EntryPrice: getFloat(pm, "entry_price"),
					MarkPrice:  getFloat(pm, "mark_price"),
					PnL:        getFloat(pm, "pnl"),
				})
			}
		}
	}
	return positions, nil
}

// Subscribe subscribes to market data via WebSocket
func (c *WsClient) Subscribe(ctx context.Context, symbol string) error {
	_, err := c.send("subscribe", map[string]interface{}{
		"symbols": []string{symbol},
	})
	return err
}

// Auth authenticates via WebSocket
func (c *WsClient) Auth(apiKey, apiSecret string) error {
	reqID, err := c.send("auth", map[string]interface{}{
		"apiKey":    apiKey,
		"apiSecret": apiSecret,
	})
	if err != nil {
		return err
	}

	msg, err := c.waitResponse(reqID, 5*time.Second)
	if err != nil {
		return err
	}

	if msg.Error != "" {
		return fmt.Errorf("auth failed: %s", msg.Error)
	}
	return nil
}

// WaitConnected waits for WebSocket connection confirmation
func (c *WsClient) WaitConnected(timeout time.Duration) error {
	select {
	case msg := <-c.responses:
		if msg.Type == "connected" {
			return nil
		}
		return fmt.Errorf("unexpected message type: %s", msg.Type)
	case <-time.After(timeout):
		return fmt.Errorf("connection timeout")
	}
}

// Responses returns the response channel for streaming
func (c *WsClient) Responses() <-chan WsMessage {
	return c.responses
}

// Close closes the WebSocket connection
func (c *WsClient) Close() error {
	return c.conn.Close()
}

// ----- gRPC Client Implementation -----

// GrpcClient implements Client over gRPC
type GrpcClient struct {
	conn    *grpc.ClientConn
	client  pb.LXDEXServiceClient
	address string
	userID  string
	verbose bool
}

// NewGrpcClient creates a gRPC client
func NewGrpcClient(address string, verbose bool) (*GrpcClient, error) {
	conn, err := grpc.NewClient(address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(100*1024*1024),
			grpc.MaxCallSendMsgSize(100*1024*1024),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("grpc dial failed: %w", err)
	}

	return &GrpcClient{
		conn:    conn,
		client:  pb.NewLXDEXServiceClient(conn),
		address: address,
		userID:  "default",
		verbose: verbose,
	}, nil
}

// Protocol returns the protocol type
func (c *GrpcClient) Protocol() Protocol {
	return ProtocolGRPC
}

// SetUserID sets the user ID for requests
func (c *GrpcClient) SetUserID(userID string) {
	c.userID = userID
}

// PlaceOrder places an order via gRPC
func (c *GrpcClient) PlaceOrder(ctx context.Context, order *Order) (*OrderResponse, error) {
	req := &pb.PlaceOrderRequest{
		Symbol: order.Symbol,
		Type:   parseOrderType(order.Type),
		Side:   parseOrderSide(order.Side),
		Price:  order.Price,
		Size:   order.Size,
		UserId: c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC PlaceOrder: %+v\n", req)
	}

	resp, err := c.client.PlaceOrder(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc PlaceOrder failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %+v\n", resp)
	}

	return &OrderResponse{
		OrderID: resp.OrderId,
		Status:  resp.Status.String(),
		Message: resp.Message,
	}, nil
}

// CancelOrder cancels an order via gRPC
func (c *GrpcClient) CancelOrder(ctx context.Context, orderID uint64) error {
	req := &pb.CancelOrderRequest{
		OrderId: orderID,
		UserId:  c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC CancelOrder: %+v\n", req)
	}

	resp, err := c.client.CancelOrder(ctx, req)
	if err != nil {
		return fmt.Errorf("grpc CancelOrder failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %+v\n", resp)
	}

	if !resp.Success {
		return fmt.Errorf("cancel failed: %s", resp.Message)
	}
	return nil
}

// GetOrders retrieves open orders via gRPC
func (c *GrpcClient) GetOrders(ctx context.Context) ([]Order, error) {
	req := &pb.GetOrdersRequest{
		UserId: c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC GetOrders: %+v\n", req)
	}

	resp, err := c.client.GetOrders(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc GetOrders failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %d orders\n", len(resp.Orders))
	}

	var orders []Order
	for _, o := range resp.Orders {
		orders = append(orders, Order{
			Symbol: o.Symbol,
			Side:   o.Side.String(),
			Type:   o.Type.String(),
			Price:  o.Price,
			Size:   o.Size,
		})
	}
	return orders, nil
}

// GetPositions retrieves positions via gRPC
func (c *GrpcClient) GetPositions(ctx context.Context) ([]Position, error) {
	req := &pb.GetPositionsRequest{
		UserId: c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC GetPositions: %+v\n", req)
	}

	resp, err := c.client.GetPositions(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc GetPositions failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %d positions\n", len(resp.Positions))
	}

	var positions []Position
	for _, p := range resp.Positions {
		positions = append(positions, Position{
			Symbol:     p.Symbol,
			Size:       p.Size,
			EntryPrice: p.EntryPrice,
			MarkPrice:  p.MarkPrice,
			PnL:        p.Pnl,
		})
	}
	return positions, nil
}

// Subscribe subscribes to market data via gRPC streaming
func (c *GrpcClient) Subscribe(ctx context.Context, symbol string) error {
	req := &pb.StreamOrderBookRequest{
		Symbol: symbol,
		Depth:  20,
	}

	if c.verbose {
		fmt.Printf(">> gRPC StreamOrderBook: %+v\n", req)
	}

	stream, err := c.client.StreamOrderBook(ctx, req)
	if err != nil {
		return fmt.Errorf("grpc StreamOrderBook failed: %w", err)
	}

	// Start goroutine to read stream
	go func() {
		for {
			update, err := stream.Recv()
			if err != nil {
				if c.verbose {
					fmt.Printf("stream ended: %v\n", err)
				}
				return
			}
			fmt.Printf("OrderBook [%s]: bids=%d asks=%d\n",
				update.Symbol, len(update.BidUpdates), len(update.AskUpdates))
		}
	}()

	return nil
}

// Ping tests connectivity via gRPC
func (c *GrpcClient) Ping(ctx context.Context) (time.Duration, error) {
	start := time.Now()
	req := &pb.PingRequest{
		Timestamp: start.UnixNano(),
	}

	resp, err := c.client.Ping(ctx, req)
	if err != nil {
		return 0, fmt.Errorf("grpc Ping failed: %w", err)
	}

	latency := time.Since(start)
	if c.verbose {
		fmt.Printf("Ping response: %s (latency: %v)\n", resp.Message, latency)
	}
	return latency, nil
}

// GetNodeInfo retrieves node information via gRPC
func (c *GrpcClient) GetNodeInfo(ctx context.Context) (*pb.NodeInfo, error) {
	resp, err := c.client.GetNodeInfo(ctx, &pb.GetNodeInfoRequest{})
	if err != nil {
		return nil, fmt.Errorf("grpc GetNodeInfo failed: %w", err)
	}
	return resp, nil
}

// Close closes the gRPC connection
func (c *GrpcClient) Close() error {
	return c.conn.Close()
}

// ----- Multi-Protocol Manager -----

// ClientManager manages multiple protocol clients
type ClientManager struct {
	wsClient   *WsClient
	grpcClient *GrpcClient
	active     Client
	mu         sync.RWMutex
}

// NewClientManager creates a client manager
func NewClientManager() *ClientManager {
	return &ClientManager{}
}

// ConnectWs connects via WebSocket
func (m *ClientManager) ConnectWs(url string, verbose bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	client, err := NewWsClient(url, verbose)
	if err != nil {
		return err
	}

	m.wsClient = client
	if m.active == nil {
		m.active = client
	}
	return nil
}

// ConnectGrpc connects via gRPC
func (m *ClientManager) ConnectGrpc(address string, verbose bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	client, err := NewGrpcClient(address, verbose)
	if err != nil {
		return err
	}

	m.grpcClient = client
	if m.active == nil {
		m.active = client
	}
	return nil
}

// SwitchProtocol switches the active protocol
func (m *ClientManager) SwitchProtocol(proto Protocol) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	switch proto {
	case ProtocolWS:
		if m.wsClient == nil {
			return fmt.Errorf("websocket client not connected")
		}
		m.active = m.wsClient
	case ProtocolGRPC:
		if m.grpcClient == nil {
			return fmt.Errorf("grpc client not connected")
		}
		m.active = m.grpcClient
	default:
		return fmt.Errorf("unknown protocol: %s", proto)
	}
	return nil
}

// Active returns the active client
func (m *ClientManager) Active() Client {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.active
}

// WsClient returns the WebSocket client
func (m *ClientManager) WsClient() *WsClient {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.wsClient
}

// GrpcClient returns the gRPC client
func (m *ClientManager) GrpcClient() *GrpcClient {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.grpcClient
}

// Close closes all connections
func (m *ClientManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errs []error
	if m.wsClient != nil {
		if err := m.wsClient.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if m.grpcClient != nil {
		if err := m.grpcClient.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}
	return nil
}

// ----- Helper Functions -----

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getFloat(m map[string]interface{}, key string) float64 {
	if v, ok := m[key].(float64); ok {
		return v
	}
	return 0
}

func parseOrderType(t string) pb.OrderType {
	switch strings.ToLower(t) {
	case "market":
		return pb.OrderType_MARKET
	case "stop":
		return pb.OrderType_STOP
	case "stop_limit":
		return pb.OrderType_STOP_LIMIT
	default:
		return pb.OrderType_LIMIT
	}
}

func parseOrderSide(s string) pb.OrderSide {
	switch strings.ToLower(s) {
	case "sell":
		return pb.OrderSide_SELL
	default:
		return pb.OrderSide_BUY
	}
}

// ----- CLI Implementation -----

func printHelp() {
	fmt.Println(`
LX DEX Trading Client Commands:

  place_order <symbol> <side> <type> <price> <size>
    Example: place_order BTC-USD buy limit 50000 0.1

  cancel_order <order_id>
    Example: cancel_order 12345

  get_orderbook <symbol>
    Example: get_orderbook BTC-USD

  get_positions
    Show all open positions

  get_orders
    Show all open orders

  subscribe <symbol>
    Subscribe to orderbook updates

  auth <api_key> <api_secret>
    Authenticate with credentials (WebSocket only)

  switch <ws|grpc>
    Switch active protocol

  ping
    Test connectivity (gRPC only)

  info
    Show node info (gRPC only)

  protocol
    Show current protocol

  help
    Show this help message

  quit / exit
    Exit the client
`)
}

func printMessage(data interface{}) {
	out, _ := json.MarshalIndent(data, "", "  ")
	fmt.Println(string(out))
}

func runInteractive(mgr *ClientManager) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Printf("LX DEX Trading Client [%s] - Type 'help' for commands\n", mgr.Active().Protocol())
	fmt.Print("> ")

	ctx := context.Background()

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			fmt.Print("> ")
			continue
		}

		parts := strings.Fields(line)
		cmd := strings.ToLower(parts[0])

		switch cmd {
		case "help":
			printHelp()

		case "quit", "exit":
			fmt.Println("Goodbye")
			return

		case "protocol":
			fmt.Printf("Active protocol: %s\n", mgr.Active().Protocol())

		case "switch":
			if len(parts) < 2 {
				fmt.Println("Usage: switch <ws|grpc>")
			} else {
				proto := Protocol(strings.ToLower(parts[1]))
				if err := mgr.SwitchProtocol(proto); err != nil {
					fmt.Printf("Switch failed: %v\n", err)
				} else {
					fmt.Printf("Switched to %s\n", proto)
				}
			}

		case "auth":
			if len(parts) < 3 {
				fmt.Println("Usage: auth <api_key> <api_secret>")
			} else if ws := mgr.WsClient(); ws != nil {
				if err := ws.Auth(parts[1], parts[2]); err != nil {
					fmt.Printf("Auth failed: %v\n", err)
				} else {
					fmt.Println("Authenticated successfully")
				}
			} else {
				fmt.Println("Auth requires WebSocket connection")
			}

		case "ping":
			if grpc := mgr.GrpcClient(); grpc != nil {
				latency, err := grpc.Ping(ctx)
				if err != nil {
					fmt.Printf("Ping failed: %v\n", err)
				} else {
					fmt.Printf("Pong! Latency: %v\n", latency)
				}
			} else {
				fmt.Println("Ping requires gRPC connection")
			}

		case "info":
			if grpc := mgr.GrpcClient(); grpc != nil {
				info, err := grpc.GetNodeInfo(ctx)
				if err != nil {
					fmt.Printf("GetNodeInfo failed: %v\n", err)
				} else {
					printMessage(info)
				}
			} else {
				fmt.Println("Info requires gRPC connection")
			}

		case "place_order":
			if len(parts) < 6 {
				fmt.Println("Usage: place_order <symbol> <side> <type> <price> <size>")
			} else {
				price, err1 := strconv.ParseFloat(parts[4], 64)
				size, err2 := strconv.ParseFloat(parts[5], 64)
				if err1 != nil || err2 != nil {
					fmt.Println("Invalid price or size")
				} else {
					order := &Order{
						Symbol: parts[1],
						Side:   parts[2],
						Type:   parts[3],
						Price:  price,
						Size:   size,
					}
					resp, err := mgr.Active().PlaceOrder(ctx, order)
					if err != nil {
						fmt.Printf("Failed: %v\n", err)
					} else {
						printMessage(resp)
					}
				}
			}

		case "cancel_order":
			if len(parts) < 2 {
				fmt.Println("Usage: cancel_order <order_id>")
			} else {
				orderID, err := strconv.ParseUint(parts[1], 10, 64)
				if err != nil {
					fmt.Println("Invalid order ID")
				} else if err := mgr.Active().CancelOrder(ctx, orderID); err != nil {
					fmt.Printf("Failed: %v\n", err)
				} else {
					fmt.Println("Order cancelled")
				}
			}

		case "get_orderbook", "subscribe":
			if len(parts) < 2 {
				fmt.Println("Usage: subscribe <symbol>")
			} else if err := mgr.Active().Subscribe(ctx, parts[1]); err != nil {
				fmt.Printf("Failed: %v\n", err)
			} else {
				fmt.Printf("Subscribed to %s\n", parts[1])
			}

		case "get_positions":
			positions, err := mgr.Active().GetPositions(ctx)
			if err != nil {
				fmt.Printf("Failed: %v\n", err)
			} else {
				printMessage(positions)
			}

		case "get_orders":
			orders, err := mgr.Active().GetOrders(ctx)
			if err != nil {
				fmt.Printf("Failed: %v\n", err)
			} else {
				printMessage(orders)
			}

		default:
			fmt.Printf("Unknown command: %s. Type 'help' for commands.\n", cmd)
		}

		fmt.Printf("[%s]> ", mgr.Active().Protocol())
	}
}

func runCommand(mgr *ClientManager, args []string) {
	if len(args) == 0 {
		fmt.Println("No command specified. Use -h for help.")
		return
	}

	ctx := context.Background()
	cmd := args[0]

	switch cmd {
	case "place_order":
		if len(args) < 6 {
			fmt.Println("Usage: lx-client place_order <symbol> <side> <type> <price> <size>")
			os.Exit(1)
		}
		price, err1 := strconv.ParseFloat(args[4], 64)
		size, err2 := strconv.ParseFloat(args[5], 64)
		if err1 != nil || err2 != nil {
			fmt.Println("Invalid price or size")
			os.Exit(1)
		}
		order := &Order{
			Symbol: args[1],
			Side:   args[2],
			Type:   args[3],
			Price:  price,
			Size:   size,
		}
		resp, err := mgr.Active().PlaceOrder(ctx, order)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		printMessage(resp)

	case "cancel_order":
		if len(args) < 2 {
			fmt.Println("Usage: lx-client cancel_order <order_id>")
			os.Exit(1)
		}
		orderID, err := strconv.ParseUint(args[1], 10, 64)
		if err != nil {
			fmt.Println("Invalid order ID")
			os.Exit(1)
		}
		if err := mgr.Active().CancelOrder(ctx, orderID); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(`{"status":"cancelled"}`)

	case "get_orderbook":
		if len(args) < 2 {
			fmt.Println("Usage: lx-client get_orderbook <symbol>")
			os.Exit(1)
		}
		if err := mgr.Active().Subscribe(ctx, args[1]); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		// For WS, wait for message
		if ws := mgr.WsClient(); ws != nil && mgr.Active().Protocol() == ProtocolWS {
			select {
			case msg := <-ws.Responses():
				printMessage(msg)
			case <-time.After(5 * time.Second):
				fmt.Println("Timeout waiting for orderbook")
			}
		}

	case "get_positions":
		positions, err := mgr.Active().GetPositions(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		printMessage(positions)

	case "get_orders":
		orders, err := mgr.Active().GetOrders(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		printMessage(orders)

	case "ping":
		if grpc := mgr.GrpcClient(); grpc != nil {
			latency, err := grpc.Ping(ctx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf(`{"latency_ns":%d,"latency_ms":%.3f}`+"\n", latency.Nanoseconds(), float64(latency.Nanoseconds())/1e6)
		} else {
			fmt.Fprintln(os.Stderr, "Ping requires gRPC connection")
			os.Exit(1)
		}

	case "info":
		if grpc := mgr.GrpcClient(); grpc != nil {
			info, err := grpc.GetNodeInfo(ctx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			printMessage(info)
		} else {
			fmt.Fprintln(os.Stderr, "Info requires gRPC connection")
			os.Exit(1)
		}

	default:
		fmt.Printf("Unknown command: %s\n", cmd)
		os.Exit(1)
	}
}

func main() {
	// Flags
	protocol := flag.String("protocol", "ws", "Protocol: ws or grpc")
	wsURL := flag.String("ws-url", "ws://localhost:8081", "WebSocket server URL")
	grpcAddr := flag.String("grpc-addr", "localhost:9090", "gRPC server address")
	apiKey := flag.String("key", "", "API key for authentication")
	apiSecret := flag.String("secret", "", "API secret for authentication")
	interactive := flag.Bool("i", false, "Interactive mode")
	verbose := flag.Bool("v", false, "Verbose output")
	flag.Parse()

	mgr := NewClientManager()

	// Connect based on protocol
	switch Protocol(*protocol) {
	case ProtocolWS:
		if err := mgr.ConnectWs(*wsURL, *verbose); err != nil {
			fmt.Fprintf(os.Stderr, "WebSocket connection failed: %v\n", err)
			os.Exit(1)
		}
		// Wait for connected message
		if ws := mgr.WsClient(); ws != nil {
			if err := ws.WaitConnected(5 * time.Second); err != nil {
				fmt.Fprintf(os.Stderr, "WebSocket handshake failed: %v\n", err)
				os.Exit(1)
			}
			if *verbose {
				fmt.Println("Connected to LX DEX via WebSocket")
			}
			// Authenticate if credentials provided
			if *apiKey != "" && *apiSecret != "" {
				if err := ws.Auth(*apiKey, *apiSecret); err != nil {
					fmt.Fprintf(os.Stderr, "Authentication failed: %v\n", err)
					os.Exit(1)
				}
				if *verbose {
					fmt.Println("Authenticated")
				}
			}
		}

	case ProtocolGRPC:
		if err := mgr.ConnectGrpc(*grpcAddr, *verbose); err != nil {
			fmt.Fprintf(os.Stderr, "gRPC connection failed: %v\n", err)
			os.Exit(1)
		}
		if *verbose {
			fmt.Println("Connected to LX DEX via gRPC")
		}

	default:
		fmt.Fprintf(os.Stderr, "Unknown protocol: %s (use ws or grpc)\n", *protocol)
		os.Exit(1)
	}

	defer mgr.Close()

	// Handle signals for graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		mgr.Close()
		os.Exit(0)
	}()

	// Run in interactive or command mode
	if *interactive || flag.NArg() == 0 {
		runInteractive(mgr)
	} else {
		runCommand(mgr, flag.Args())
	}
}
