package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/luxfi/log"
)

type Order struct {
	Symbol      string  `json:"symbol"`
	Type        string  `json:"type"`
	Side        string  `json:"side"`
	Price       float64 `json:"price"`
	Size        float64 `json:"size"`
	User        string  `json:"user,omitempty"`
	TimeInForce string  `json:"timeInForce,omitempty"`
}

type OrderResponse struct {
	Success bool   `json:"success"`
	OrderID string `json:"orderId,omitempty"`
	Message string `json:"message,omitempty"`
}

type MarketDataResponse struct {
	Symbol string      `json:"symbol"`
	Bids   [][]float64 `json:"bids"`
	Asks   [][]float64 `json:"asks"`
	Last   float64     `json:"last"`
	Volume float64     `json:"volume"`
}

type LuxClient struct {
	baseURL string
	logger  log.Logger
	client  *http.Client
}

func NewLuxClient(baseURL string) *LuxClient {
	level, _ := log.ToLevel("info")
	logger := log.NewTestLogger(level)
	return &LuxClient{
		baseURL: baseURL,
		logger:  logger,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (c *LuxClient) PlaceOrder(order Order) (*OrderResponse, error) {
	data, err := json.Marshal(order)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Post(c.baseURL+"/order", "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var orderResp OrderResponse
	if err := json.Unmarshal(body, &orderResp); err != nil {
		// If we can't parse the response, return raw message
		return &OrderResponse{
			Success: resp.StatusCode == 200,
			Message: string(body),
		}, nil
	}

	return &orderResp, nil
}

func (c *LuxClient) GetOrderBook(symbol string) (*MarketDataResponse, error) {
	resp, err := c.client.Get(c.baseURL + "/orderbook/" + symbol)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var marketData MarketDataResponse
	if err := json.Unmarshal(body, &marketData); err != nil {
		c.logger.Warn("Failed to parse orderbook response", "error", err, "body", string(body))
		return nil, fmt.Errorf("failed to parse response: %s", string(body))
	}

	return &marketData, nil
}

func (c *LuxClient) GetMetrics() (string, error) {
	resp, err := c.client.Get("http://localhost:9090/metrics")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

func main() {
	var (
		serverURL = flag.String("server", "http://localhost:8080", "LXD server URL")
		action    = flag.String("action", "test", "Action: test, buy, sell, book, metrics")
		symbol    = flag.String("symbol", "BTC-USD", "Trading symbol")
		price     = flag.Float64("price", 50000, "Order price")
		size      = flag.Float64("size", 1.0, "Order size")
		user      = flag.String("user", "client1", "User ID")
	)
	flag.Parse()

	logger := log.Root()
	logger.Info("LUX DEX Client", "server", *serverURL, "action", *action)

	client := NewLuxClient(*serverURL)

	switch *action {
	case "test":
		// Place a few test orders
		logger.Info("Placing test orders...")

		// Buy order
		buyOrder := Order{
			Symbol: *symbol,
			Type:   "limit",
			Side:   "buy",
			Price:  49000,
			Size:   1.0,
			User:   *user,
		}
		resp, err := client.PlaceOrder(buyOrder)
		if err != nil {
			logger.Info("Note: HTTP API not yet implemented in luxd", "error", err)
		} else {
			logger.Info("Buy order placed", "response", resp)
		}

		// Sell order
		sellOrder := Order{
			Symbol: *symbol,
			Type:   "limit",
			Side:   "sell",
			Price:  51000,
			Size:   1.0,
			User:   *user,
		}
		resp, err = client.PlaceOrder(sellOrder)
		if err != nil {
			logger.Info("Note: HTTP API not yet implemented in luxd", "error", err)
		} else {
			logger.Info("Sell order placed", "response", resp)
		}

		// Try to get orderbook
		book, err := client.GetOrderBook(*symbol)
		if err != nil {
			logger.Info("Note: HTTP API not yet implemented in luxd", "error", err)
		} else {
			logger.Info("Order book", "symbol", book.Symbol, "bids", len(book.Bids), "asks", len(book.Asks))
		}

	case "buy":
		order := Order{
			Symbol: *symbol,
			Type:   "limit",
			Side:   "buy",
			Price:  *price,
			Size:   *size,
			User:   *user,
		}
		resp, err := client.PlaceOrder(order)
		if err != nil {
			logger.Error("Failed to place buy order", "error", err)
			os.Exit(1)
		}
		logger.Info("Buy order placed", "response", resp)

	case "sell":
		order := Order{
			Symbol: *symbol,
			Type:   "limit",
			Side:   "sell",
			Price:  *price,
			Size:   *size,
			User:   *user,
		}
		resp, err := client.PlaceOrder(order)
		if err != nil {
			logger.Error("Failed to place sell order", "error", err)
			os.Exit(1)
		}
		logger.Info("Sell order placed", "response", resp)

	case "book":
		book, err := client.GetOrderBook(*symbol)
		if err != nil {
			logger.Error("Failed to get order book", "error", err)
			os.Exit(1)
		}
		logger.Info("Order book retrieved",
			"symbol", book.Symbol,
			"bids", len(book.Bids),
			"asks", len(book.Asks),
			"last", book.Last,
			"volume", book.Volume)

	case "metrics":
		metrics, err := client.GetMetrics()
		if err != nil {
			logger.Info("Note: Metrics endpoint may not be implemented yet", "error", err)
		} else {
			fmt.Println("Metrics from luxd:")
			fmt.Println(metrics[:500]) // First 500 chars
		}

	default:
		logger.Error("Unknown action", "action", *action)
		os.Exit(1)
	}

	logger.Info("Client operation complete")
}
