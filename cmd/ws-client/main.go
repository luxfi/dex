package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/url"
	"os"
	"os/signal"
	"time"

	"github.com/gorilla/websocket"
	"github.com/luxfi/log"
)

type Message struct {
	Type   string      `json:"type"`
	Action string      `json:"action,omitempty"`
	Data   interface{} `json:"data,omitempty"`
}

type SubscribeRequest struct {
	Type    string   `json:"type"`
	Symbols []string `json:"symbols"`
	Depth   int      `json:"depth,omitempty"`
}

func main() {
	var (
		wsURL   = flag.String("url", "ws://localhost:8081", "WebSocket URL")
		symbol  = flag.String("symbol", "LXD-MAINNET", "Symbol to subscribe")
		timeout = flag.Duration("timeout", 10*time.Second, "Connection timeout")
	)
	flag.Parse()

	level, _ := log.ToLevel("info")
	logger := log.NewTestLogger(level)
	
	logger.Info("Connecting to LXD WebSocket", "url", *wsURL)

	// Parse URL
	u, err := url.Parse(*wsURL)
	if err != nil {
		logger.Error("Invalid URL", "error", err)
		os.Exit(1)
	}

	// Connect to WebSocket
	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		logger.Info("Note: WebSocket API may not be implemented in luxd yet", "error", err)
		logger.Info("This is expected - luxd currently focuses on core consensus")
		return
	}
	defer conn.Close()

	logger.Info("Connected to WebSocket")

	// Send subscription message
	sub := SubscribeRequest{
		Type:    "subscribe",
		Symbols: []string{*symbol},
		Depth:   10,
	}

	if err := conn.WriteJSON(sub); err != nil {
		logger.Error("Failed to send subscription", "error", err)
		return
	}

	logger.Info("Subscription sent", "symbol", *symbol)

	// Setup signal handler
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)

	// Read messages
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			messageType, message, err := conn.ReadMessage()
			if err != nil {
				logger.Warn("Read error", "error", err)
				return
			}

			if messageType == websocket.TextMessage {
				var msg Message
				if err := json.Unmarshal(message, &msg); err != nil {
					logger.Info("Raw message", "data", string(message))
				} else {
					logger.Info("Message received", "type", msg.Type, "action", msg.Action)
					if msg.Data != nil {
						logger.Info("Message data", "data", fmt.Sprintf("%+v", msg.Data))
					}
				}
			}
		}
	}()

	// Wait for interrupt or timeout
	select {
	case <-done:
		logger.Info("Connection closed")
	case <-interrupt:
		logger.Info("Interrupt received, closing connection")
		err := conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		if err != nil {
			logger.Warn("Failed to send close message", "error", err)
		}
		select {
		case <-done:
		case <-time.After(time.Second):
		}
	case <-time.After(*timeout):
		logger.Info("Timeout reached")
	}

	logger.Info("WebSocket client terminated")
}