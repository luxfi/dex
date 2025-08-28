//go:build !cgo
// +build !cgo

package orderbook

import (
	"github.com/luxfi/dex/pkg/lx"
)

// NewOrderBookImpl creates a Go implementation of the order book
func NewOrderBookImpl(cfg Config) OrderBook {
	// Create a wrapper that implements the OrderBook interface
	return &goOrderBook{
		ob: lx.NewOrderBook("default"),
	}
}

// goOrderBook wraps the lx.OrderBook to implement the orderbook.OrderBook interface
type goOrderBook struct {
	ob *lx.OrderBook
}

func (g *goOrderBook) AddOrder(order *Order) uint64 {
	lxOrder := &lx.Order{
		ID:        order.ID,
		Type:      lx.OrderType(order.Type),
		Side:      lx.Side(order.Side),
		Price:     order.Price,
		Size:      order.Quantity,
		User:      "user",
		Timestamp: order.Timestamp,
	}
	g.ob.AddOrder(lxOrder)
	return order.ID
}

func (g *goOrderBook) CancelOrder(orderID uint64) bool {
	err := g.ob.CancelOrder(orderID)
	return err == nil
}

func (g *goOrderBook) ModifyOrder(orderID uint64, newPrice, newQuantity float64) bool {
	// Not implemented in lx.OrderBook
	return false
}

func (g *goOrderBook) MatchOrders() []Trade {
	// Matching happens automatically in lx.OrderBook
	return []Trade{}
}

func (g *goOrderBook) GetBestBid() float64 {
	bid, _ := g.ob.GetBestBidAsk()
	if bid != nil {
		return bid.Price
	}
	return 0
}

func (g *goOrderBook) GetBestAsk() float64 {
	_, ask := g.ob.GetBestBidAsk()
	if ask != nil {
		return ask.Price
	}
	return 0
}

func (g *goOrderBook) GetDepth(level int) *Depth {
	bids, asks := g.ob.GetOrderBookDepth(level)
	
	depth := &Depth{
		Bids: make([]PriceLevel, len(bids)),
		Asks: make([]PriceLevel, len(asks)),
	}
	
	for i, bid := range bids {
		depth.Bids[i] = PriceLevel{
			Price: bid.Price,
			Size:  bid.Size,
		}
	}
	
	for i, ask := range asks {
		depth.Asks[i] = PriceLevel{
			Price: ask.Price,
			Size:  ask.Size,
		}
	}
	
	return depth
}

func (g *goOrderBook) GetVolume() uint64 {
	// Not directly available in lx.OrderBook
	return 0
}