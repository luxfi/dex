// LX DEX CLI Client
//
// Command-line trading interface for LX DEX WebSocket API.
// Connect to ws://localhost:8081 for real-time trading.
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
)

// Message represents a WebSocket message
type Message struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
	RequestID string                 `json:"request_id,omitempty"`
	Timestamp int64                  `json:"timestamp,omitempty"`
}

// Order represents an order for placement
type Order struct {
	Symbol string  `json:"symbol"`
	Side   string  `json:"side"`
	Type   string  `json:"type"`
	Price  float64 `json:"price"`
	Size   float64 `json:"size"`
}

// Client wraps WebSocket connection with helpers
type Client struct {
	conn       *websocket.Conn
	mu         sync.Mutex
	reqCounter int
	verbose    bool
	responses  chan Message
}

// NewClient creates a new WebSocket client
func NewClient(url string, verbose bool) (*Client, error) {
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(url, nil)
	if err != nil {
		return nil, fmt.Errorf("dial failed: %w", err)
	}

	c := &Client{
		conn:      conn,
		verbose:   verbose,
		responses: make(chan Message, 100),
	}

	go c.readLoop()
	return c, nil
}

// readLoop handles incoming messages
func (c *Client) readLoop() {
	for {
		var msg Message
		err := c.conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				fmt.Fprintf(os.Stderr, "read error: %v\n", err)
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

// Send sends a message and returns the request ID
func (c *Client) Send(msgType string, data map[string]interface{}) (string, error) {
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

// WaitResponse waits for a response with matching request ID
func (c *Client) WaitResponse(reqID string, timeout time.Duration) (*Message, error) {
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
			// Print non-matching messages (e.g., subscriptions)
			if msg.Type != "connected" {
				printMessage(&msg)
			}
		case <-deadline:
			return nil, fmt.Errorf("timeout waiting for response")
		}
	}
}

// Close closes the connection
func (c *Client) Close() error {
	return c.conn.Close()
}

// Auth authenticates with API credentials
func (c *Client) Auth(apiKey, apiSecret string) error {
	reqID, err := c.Send("auth", map[string]interface{}{
		"apiKey":    apiKey,
		"apiSecret": apiSecret,
	})
	if err != nil {
		return err
	}

	resp, err := c.WaitResponse(reqID, 5*time.Second)
	if err != nil {
		return err
	}

	if resp.Error != "" {
		return fmt.Errorf("auth failed: %s", resp.Error)
	}
	return nil
}

// PlaceOrder places an order
func (c *Client) PlaceOrder(order *Order) (*Message, error) {
	reqID, err := c.Send("place_order", map[string]interface{}{
		"order": order,
	})
	if err != nil {
		return nil, err
	}
	return c.WaitResponse(reqID, 5*time.Second)
}

// CancelOrder cancels an order
func (c *Client) CancelOrder(orderID uint64) (*Message, error) {
	reqID, err := c.Send("cancel_order", map[string]interface{}{
		"orderID": orderID,
	})
	if err != nil {
		return nil, err
	}
	return c.WaitResponse(reqID, 5*time.Second)
}

// GetPositions retrieves positions
func (c *Client) GetPositions() (*Message, error) {
	reqID, err := c.Send("get_positions", nil)
	if err != nil {
		return nil, err
	}
	return c.WaitResponse(reqID, 5*time.Second)
}

// GetOrders retrieves open orders
func (c *Client) GetOrders() (*Message, error) {
	reqID, err := c.Send("get_orders", nil)
	if err != nil {
		return nil, err
	}
	return c.WaitResponse(reqID, 5*time.Second)
}

// Subscribe to a channel
func (c *Client) Subscribe(symbol string) error {
	_, err := c.Send("subscribe", map[string]interface{}{
		"symbols": []string{symbol},
	})
	return err
}

func printMessage(msg *Message) {
	if msg.Error != "" {
		fmt.Printf("Error: %s\n", msg.Error)
		return
	}

	switch msg.Type {
	case "order_update":
		fmt.Printf("Order Update: %v\n", msg.Data)
	case "position_update":
		fmt.Printf("Position Update: %v\n", msg.Data)
	case "orderbook":
		if symbol, ok := msg.Data["symbol"].(string); ok {
			fmt.Printf("OrderBook [%s]:\n", symbol)
			if bids, ok := msg.Data["bids"].([]interface{}); ok {
				fmt.Printf("  Bids: %d levels\n", len(bids))
				for i, b := range bids {
					if i >= 5 {
						break
					}
					if bid, ok := b.(map[string]interface{}); ok {
						fmt.Printf("    %.2f @ %.4f\n", bid["price"], bid["size"])
					}
				}
			}
			if asks, ok := msg.Data["asks"].([]interface{}); ok {
				fmt.Printf("  Asks: %d levels\n", len(asks))
				for i, a := range asks {
					if i >= 5 {
						break
					}
					if ask, ok := a.(map[string]interface{}); ok {
						fmt.Printf("    %.2f @ %.4f\n", ask["price"], ask["size"])
					}
				}
			}
		}
	default:
		data, _ := json.MarshalIndent(msg, "", "  ")
		fmt.Printf("%s\n", data)
	}
}

func printHelp() {
	fmt.Println(`
LX DEX CLI Commands:

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
    Authenticate with credentials

  help
    Show this help message

  quit / exit
    Exit the CLI
`)
}

func runInteractive(client *Client) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("LX DEX CLI - Type 'help' for commands")
	fmt.Print("> ")

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

		case "auth":
			if len(parts) < 3 {
				fmt.Println("Usage: auth <api_key> <api_secret>")
			} else {
				err := client.Auth(parts[1], parts[2])
				if err != nil {
					fmt.Printf("Auth failed: %v\n", err)
				} else {
					fmt.Println("Authenticated successfully")
				}
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
					resp, err := client.PlaceOrder(order)
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
				} else {
					resp, err := client.CancelOrder(orderID)
					if err != nil {
						fmt.Printf("Failed: %v\n", err)
					} else {
						printMessage(resp)
					}
				}
			}

		case "get_orderbook":
			if len(parts) < 2 {
				fmt.Println("Usage: get_orderbook <symbol>")
			} else {
				err := client.Subscribe(parts[1])
				if err != nil {
					fmt.Printf("Failed: %v\n", err)
				} else {
					fmt.Printf("Subscribed to %s orderbook\n", parts[1])
				}
			}

		case "get_positions":
			resp, err := client.GetPositions()
			if err != nil {
				fmt.Printf("Failed: %v\n", err)
			} else {
				printMessage(resp)
			}

		case "get_orders":
			resp, err := client.GetOrders()
			if err != nil {
				fmt.Printf("Failed: %v\n", err)
			} else {
				printMessage(resp)
			}

		case "subscribe":
			if len(parts) < 2 {
				fmt.Println("Usage: subscribe <symbol>")
			} else {
				err := client.Subscribe(parts[1])
				if err != nil {
					fmt.Printf("Failed: %v\n", err)
				} else {
					fmt.Printf("Subscribed to %s\n", parts[1])
				}
			}

		default:
			fmt.Printf("Unknown command: %s. Type 'help' for commands.\n", cmd)
		}

		fmt.Print("> ")
	}
}

func runCommand(client *Client, args []string) {
	if len(args) == 0 {
		fmt.Println("No command specified. Use -h for help.")
		return
	}

	cmd := args[0]
	switch cmd {
	case "place_order":
		if len(args) < 6 {
			fmt.Println("Usage: lx-cli place_order <symbol> <side> <type> <price> <size>")
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
		resp, err := client.PlaceOrder(order)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		data, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(data))

	case "cancel_order":
		if len(args) < 2 {
			fmt.Println("Usage: lx-cli cancel_order <order_id>")
			os.Exit(1)
		}
		orderID, err := strconv.ParseUint(args[1], 10, 64)
		if err != nil {
			fmt.Println("Invalid order ID")
			os.Exit(1)
		}
		resp, err := client.CancelOrder(orderID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		data, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(data))

	case "get_orderbook":
		if len(args) < 2 {
			fmt.Println("Usage: lx-cli get_orderbook <symbol>")
			os.Exit(1)
		}
		err := client.Subscribe(args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		// Wait for response
		select {
		case msg := <-client.responses:
			printMessage(&msg)
		case <-time.After(5 * time.Second):
			fmt.Println("Timeout waiting for orderbook")
		}

	case "get_positions":
		resp, err := client.GetPositions()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		data, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(data))

	case "get_orders":
		resp, err := client.GetOrders()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		data, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(data))

	default:
		fmt.Printf("Unknown command: %s\n", cmd)
		os.Exit(1)
	}
}

func main() {
	wsURL := flag.String("url", "ws://localhost:8081", "WebSocket server URL")
	apiKey := flag.String("key", "", "API key for authentication")
	apiSecret := flag.String("secret", "", "API secret for authentication")
	interactive := flag.Bool("i", false, "Interactive mode")
	verbose := flag.Bool("v", false, "Verbose output")
	flag.Parse()

	// Connect
	client, err := NewClient(*wsURL, *verbose)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Connection failed: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// Wait for connected message
	select {
	case msg := <-client.responses:
		if msg.Type == "connected" {
			if *verbose {
				fmt.Println("Connected to LX DEX")
			}
		}
	case <-time.After(5 * time.Second):
		fmt.Fprintln(os.Stderr, "Timeout waiting for connection")
		os.Exit(1)
	}

	// Authenticate if credentials provided
	if *apiKey != "" && *apiSecret != "" {
		if err := client.Auth(*apiKey, *apiSecret); err != nil {
			fmt.Fprintf(os.Stderr, "Authentication failed: %v\n", err)
			os.Exit(1)
		}
		if *verbose {
			fmt.Println("Authenticated")
		}
	}

	// Handle signals for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
		client.Close()
	}()

	// Run in interactive or command mode
	if *interactive || flag.NArg() == 0 {
		runInteractive(client)
	} else {
		runCommand(client, flag.Args())
	}

	_ = ctx // silence unused variable warning
}
