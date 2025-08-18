// Go SDK for LX Trading Platform
package lxsdk

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
    
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

// Side represents order side
type Side int

const (
    Buy Side = iota
    Sell
)

// OrderType represents order type
type OrderType int

const (
    Market OrderType = iota
    Limit
    Stop
    StopLimit
)

// Order represents a trading order
type Order struct {
    ID        string
    Symbol    string
    Price     float64
    Quantity  float64
    Side      Side
    Type      OrderType
    Timestamp time.Time
}

// Trade represents an executed trade
type Trade struct {
    ID        string
    OrderID   string
    Symbol    string
    Price     float64
    Quantity  float64
    Side      Side
    Timestamp time.Time
}

// Client is the main trading client
type Client struct {
    conn      *grpc.ClientConn
    address   string
    
    // FIX support
    fixEnabled bool
    senderID   string
    targetID   string
    seqNum     atomic.Int32
    
    // Performance metrics
    ordersSent   atomic.Int64
    tradesRecv   atomic.Int64
    latencyNanos atomic.Int64
    
    // Callbacks
    onTrade    func(*Trade)
    onOrder    func(*Order)
    onError    func(error)
    
    mu sync.RWMutex
}

// NewClient creates a new trading client
func NewClient(address string) (*Client, error) {
    conn, err := grpc.Dial(address,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(100*1024*1024), // 100MB
            grpc.MaxCallSendMsgSize(100*1024*1024),
        ),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to connect: %w", err)
    }
    
    client := &Client{
        conn:     conn,
        address:  address,
        senderID: "GO_CLIENT",
        targetID: "EXCHANGE",
    }
    
    client.seqNum.Store(1)
    
    return client, nil
}

// EnableFIX enables FIX protocol mode
func (c *Client) EnableFIX(senderID, targetID string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.fixEnabled = true
    c.senderID = senderID
    c.targetID = targetID
}

// SendOrder sends a new order
func (c *Client) SendOrder(symbol string, price, quantity float64, side Side) (*Order, error) {
    start := time.Now()
    defer func() {
        c.latencyNanos.Store(time.Since(start).Nanoseconds())
    }()
    
    order := &Order{
        ID:        fmt.Sprintf("ORD%d", time.Now().UnixNano()),
        Symbol:    symbol,
        Price:     price,
        Quantity:  quantity,
        Side:      side,
        Type:      Limit,
        Timestamp: time.Now(),
    }
    
    if c.fixEnabled {
        // Send as FIX message
        fixMsg := c.buildFIXNewOrder(order)
        if err := c.sendFIXMessage(fixMsg); err != nil {
            return nil, err
        }
    } else {
        // Send via gRPC
        // Implementation would call the actual gRPC service
    }
    
    c.ordersSent.Add(1)
    
    if c.onOrder != nil {
        c.onOrder(order)
    }
    
    return order, nil
}

// SendBulkOrders sends multiple orders efficiently
func (c *Client) SendBulkOrders(orders []*Order) error {
    // Batch processing for high throughput
    batch := make([]*Order, 0, 1000)
    
    for _, order := range orders {
        batch = append(batch, order)
        
        if len(batch) >= 1000 {
            if err := c.processBatch(batch); err != nil {
                return err
            }
            batch = batch[:0]
        }
    }
    
    if len(batch) > 0 {
        return c.processBatch(batch)
    }
    
    return nil
}

// CancelOrder cancels an existing order
func (c *Client) CancelOrder(orderID string) error {
    // Implementation
    return nil
}

// Subscribe subscribes to market data
func (c *Client) Subscribe(symbols []string, callback func(*Trade)) error {
    c.onTrade = callback
    // Implementation would set up streaming
    return nil
}

// GetOrderBook gets current order book
func (c *Client) GetOrderBook(symbol string) (bids, asks []Order, err error) {
    // Implementation
    return nil, nil, nil
}

// Performance metrics methods

// GetMetrics returns performance metrics
func (c *Client) GetMetrics() map[string]interface{} {
    return map[string]interface{}{
        "orders_sent":    c.ordersSent.Load(),
        "trades_received": c.tradesRecv.Load(),
        "latency_ns":     c.latencyNanos.Load(),
        "latency_ms":     float64(c.latencyNanos.Load()) / 1e6,
    }
}

// GetThroughput returns current throughput
func (c *Client) GetThroughput() float64 {
    // Implementation would track time-based throughput
    return float64(c.ordersSent.Load())
}

// Close closes the client connection
func (c *Client) Close() error {
    if c.conn != nil {
        return c.conn.Close()
    }
    return nil
}

// Private methods

func (c *Client) buildFIXNewOrder(order *Order) string {
    // Build FIX message
    seqNum := c.seqNum.Add(1)
    
    side := "1" // Buy
    if order.Side == Sell {
        side = "2"
    }
    
    msg := fmt.Sprintf("8=FIX.4.4\x0135=D\x0149=%s\x0156=%s\x0134=%d\x0152=%s\x01"+
        "11=%s\x0155=%s\x0154=%s\x0138=%.2f\x0140=2\x0144=%.2f\x0159=0\x0160=%s\x01",
        c.senderID, c.targetID, seqNum, 
        time.Now().Format("20060102-15:04:05.000"),
        order.ID, order.Symbol, side, order.Quantity, order.Price,
        time.Now().Format("20060102-15:04:05.000"))
    
    // Add checksum
    checksum := c.calculateChecksum(msg)
    msg += fmt.Sprintf("10=%03d\x01", checksum)
    
    return msg
}

func (c *Client) calculateChecksum(msg string) int {
    sum := 0
    for _, b := range []byte(msg) {
        sum += int(b)
    }
    return sum % 256
}

func (c *Client) sendFIXMessage(msg string) error {
    // Implementation would send via TCP
    return nil
}

func (c *Client) processBatch(orders []*Order) error {
    // Batch processing implementation
    for range orders {
        c.ordersSent.Add(1)
    }
    return nil
}

// HighFrequencyClient is optimized for HFT
type HighFrequencyClient struct {
    *Client
    
    // Pre-allocated order pool
    orderPool sync.Pool
    
    // Lock-free queues
    sendQueue chan *Order
    recvQueue chan *Trade
}

// NewHighFrequencyClient creates HFT-optimized client
func NewHighFrequencyClient(address string) (*HighFrequencyClient, error) {
    client, err := NewClient(address)
    if err != nil {
        return nil, err
    }
    
    hft := &HighFrequencyClient{
        Client:    client,
        sendQueue: make(chan *Order, 100000),
        recvQueue: make(chan *Trade, 100000),
    }
    
    hft.orderPool.New = func() interface{} {
        return &Order{}
    }
    
    // Start worker goroutines
    for i := 0; i < 10; i++ {
        go hft.sendWorker()
    }
    
    return hft, nil
}

func (hft *HighFrequencyClient) sendWorker() {
    for order := range hft.sendQueue {
        hft.SendOrder(order.Symbol, order.Price, order.Quantity, order.Side)
        hft.orderPool.Put(order)
    }
}

// SendOrderAsync sends order without waiting
func (hft *HighFrequencyClient) SendOrderAsync(symbol string, price, quantity float64, side Side) {
    order := hft.orderPool.Get().(*Order)
    order.Symbol = symbol
    order.Price = price
    order.Quantity = quantity
    order.Side = side
    
    select {
    case hft.sendQueue <- order:
    default:
        // Queue full, drop order
        hft.orderPool.Put(order)
    }
}
