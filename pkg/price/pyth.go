package price

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// PythSource provides real-time prices from Pyth Network.
// Sub-second updates via WebSocket for low-latency trading.
type PythSource struct {
	wsURL  string
	apiURL string
	conn   *websocket.Conn
	client *http.Client

	priceIDs map[string]string
	prices   map[string]*Data
	subs     map[string]bool

	healthy   bool
	heartbeat time.Time

	mu   sync.RWMutex
	done chan struct{}
}

// NewPythSource creates a Pyth Network price source.
func NewPythSource(wsURL, apiURL string) *PythSource {
	return &PythSource{
		wsURL:    wsURL,
		apiURL:   apiURL,
		client:   &http.Client{Timeout: 5 * time.Second},
		priceIDs: pythPriceIDs(),
		prices:   make(map[string]*Data),
		subs:     make(map[string]bool),
		done:     make(chan struct{}),
	}
}

func pythPriceIDs() map[string]string {
	return map[string]string{
		"BTC-USD":  "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
		"ETH-USD":  "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
		"SOL-USD":  "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
		"AVAX-USD": "0x93da3352f9f1d105fdfe4971cfa80e9dd777bfc5d0f683ebb6e1294b92137bb7",
		"LUX-USD":  "0x0000000000000000000000000000000000000000000000000000000000000001", // Placeholder
	}
}

// Connect establishes WebSocket connection.
func (s *PythSource) Connect() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn != nil {
		s.conn.Close()
	}

	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, _, err := dialer.Dial(s.wsURL, nil)
	if err != nil {
		return err
	}

	s.conn = conn
	s.healthy = true
	s.heartbeat = time.Now()

	go s.read()
	go s.ping()

	// Subscribe to all price feeds
	for symbol := range s.priceIDs {
		s.subscribe(symbol)
	}

	return nil
}

func (s *PythSource) read() {
	defer func() {
		s.mu.Lock()
		s.healthy = false
		s.mu.Unlock()
	}()

	for {
		select {
		case <-s.done:
			return
		default:
			var msg map[string]interface{}
			if err := s.conn.ReadJSON(&msg); err != nil {
				return
			}
			s.handle(msg)
		}
	}
}

func (s *PythSource) handle(msg map[string]interface{}) {
	msgType, _ := msg["type"].(string)

	switch msgType {
	case "price_update":
		s.handlePrice(msg)
	case "heartbeat":
		s.mu.Lock()
		s.heartbeat = time.Now()
		s.mu.Unlock()
	}
}

func (s *PythSource) handlePrice(msg map[string]interface{}) {
	data, ok := msg["data"].(map[string]interface{})
	if !ok {
		return
	}

	priceID, _ := data["price_id"].(string)

	// Find symbol
	var symbol string
	for sym, id := range s.priceIDs {
		if id == priceID {
			symbol = sym
			break
		}
	}
	if symbol == "" {
		return
	}

	price, _ := data["price"].(float64)
	conf, _ := data["confidence"].(float64)
	ts, _ := data["publish_time"].(float64)

	if expo, ok := data["expo"].(float64); ok {
		price = price * math.Pow(10, expo)
	}

	s.mu.Lock()
	s.prices[symbol] = &Data{
		Symbol:     symbol,
		Price:      price,
		Confidence: 1.0 - (conf / price),
		Timestamp:  time.Unix(int64(ts), 0),
		Source:     "pyth",
	}
	s.mu.Unlock()
}

func (s *PythSource) subscribe(symbol string) error {
	id, ok := s.priceIDs[symbol]
	if !ok {
		return fmt.Errorf("unknown symbol: %s", symbol)
	}

	if s.conn == nil {
		return nil
	}

	return s.conn.WriteJSON(map[string]interface{}{
		"type": "subscribe",
		"ids":  []string{id},
	})
}

func (s *PythSource) ping() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			s.mu.RLock()
			conn := s.conn
			s.mu.RUnlock()
			if conn != nil {
				conn.WriteJSON(map[string]string{"type": "heartbeat"})
			}
		}
	}
}

// Price returns the latest price for a symbol.
func (s *PythSource) Price(symbol string) (*Data, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p, ok := s.prices[symbol]
	if !ok {
		return s.fetchHTTP(symbol)
	}

	copy := *p
	return &copy, nil
}

func (s *PythSource) fetchHTTP(symbol string) (*Data, error) {
	id, ok := s.priceIDs[symbol]
	if !ok {
		return nil, ErrNotFound
	}

	url := fmt.Sprintf("%s/api/latest_price_feeds?ids[]=%s", s.apiURL, id)
	resp, err := s.client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result []struct {
		Price      float64 `json:"price"`
		Confidence float64 `json:"conf"`
		Expo       int32   `json:"expo"`
		Time       int64   `json:"publish_time"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result) == 0 {
		return nil, ErrNotFound
	}

	r := result[0]
	price := r.Price * math.Pow(10, float64(r.Expo))

	return &Data{
		Symbol:     symbol,
		Price:      price,
		Confidence: 1.0 - (r.Confidence / price),
		Timestamp:  time.Unix(r.Time, 0),
		Source:     "pyth",
	}, nil
}

// Prices returns prices for multiple symbols.
func (s *PythSource) Prices(symbols []string) (map[string]*Data, error) {
	result := make(map[string]*Data)
	for _, sym := range symbols {
		if p, err := s.Price(sym); err == nil {
			result[sym] = p
		}
	}
	return result, nil
}

// Subscribe registers for updates.
func (s *PythSource) Subscribe(symbol string) error {
	s.mu.Lock()
	s.subs[symbol] = true
	s.mu.Unlock()
	return s.subscribe(symbol)
}

// Unsubscribe removes subscription.
func (s *PythSource) Unsubscribe(symbol string) error {
	s.mu.Lock()
	delete(s.subs, symbol)
	s.mu.Unlock()
	return nil
}

// Healthy returns true if connected.
func (s *PythSource) Healthy() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.healthy && time.Since(s.heartbeat) < 30*time.Second
}

// Name returns "pyth".
func (s *PythSource) Name() string { return "pyth" }

// Weight returns 1.2 (external oracle).
func (s *PythSource) Weight() float64 { return 1.2 }

// Close shuts down the source.
func (s *PythSource) Close() error {
	close(s.done)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn != nil {
		return s.conn.Close()
	}
	return nil
}
