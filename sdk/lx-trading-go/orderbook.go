// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"sort"
	"sync"
	"time"

	"github.com/shopspring/decimal"
)

// Orderbook represents a venue's orderbook.
type Orderbook struct {
	Symbol    string
	Venue     string
	Bids      []PriceLevel
	Asks      []PriceLevel
	Timestamp time.Time
	Sequence  int64

	mu sync.RWMutex
}

// NewOrderbook creates a new orderbook.
func NewOrderbook(symbol, venue string) *Orderbook {
	return &Orderbook{
		Symbol:    symbol,
		Venue:     venue,
		Bids:      make([]PriceLevel, 0),
		Asks:      make([]PriceLevel, 0),
		Timestamp: time.Now(),
	}
}

// AddBid adds a bid level.
func (b *Orderbook) AddBid(price, quantity decimal.Decimal) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Bids = append(b.Bids, PriceLevel{Price: price, Quantity: quantity})
}

// AddAsk adds an ask level.
func (b *Orderbook) AddAsk(price, quantity decimal.Decimal) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Asks = append(b.Asks, PriceLevel{Price: price, Quantity: quantity})
}

// Sort sorts bids descending and asks ascending.
func (b *Orderbook) Sort() {
	b.mu.Lock()
	defer b.mu.Unlock()

	sort.Slice(b.Bids, func(i, j int) bool {
		return b.Bids[i].Price.GreaterThan(b.Bids[j].Price)
	})
	sort.Slice(b.Asks, func(i, j int) bool {
		return b.Asks[i].Price.LessThan(b.Asks[j].Price)
	})
}

// BestBid returns the best bid price.
func (b *Orderbook) BestBid() *decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if len(b.Bids) == 0 {
		return nil
	}
	return &b.Bids[0].Price
}

// BestAsk returns the best ask price.
func (b *Orderbook) BestAsk() *decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if len(b.Asks) == 0 {
		return nil
	}
	return &b.Asks[0].Price
}

// MidPrice returns (bid + ask) / 2.
func (b *Orderbook) MidPrice() *decimal.Decimal {
	bid := b.BestBid()
	ask := b.BestAsk()
	if bid != nil && ask != nil {
		mid := bid.Add(*ask).Div(decimal.NewFromInt(2))
		return &mid
	}
	return nil
}

// Spread returns ask - bid.
func (b *Orderbook) Spread() *decimal.Decimal {
	bid := b.BestBid()
	ask := b.BestAsk()
	if bid != nil && ask != nil {
		spread := ask.Sub(*bid)
		return &spread
	}
	return nil
}

// SpreadPercent returns spread as percentage of mid price.
func (b *Orderbook) SpreadPercent() *decimal.Decimal {
	spread := b.Spread()
	mid := b.MidPrice()
	if spread != nil && mid != nil && !mid.IsZero() {
		pct := spread.Div(*mid).Mul(decimal.NewFromInt(100))
		return &pct
	}
	return nil
}

// BidLiquidity returns total bid-side liquidity in quote terms.
func (b *Orderbook) BidLiquidity() decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	total := decimal.Zero
	for _, level := range b.Bids {
		total = total.Add(level.Value())
	}
	return total
}

// AskLiquidity returns total ask-side liquidity in quote terms.
func (b *Orderbook) AskLiquidity() decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	total := decimal.Zero
	for _, level := range b.Asks {
		total = total.Add(level.Value())
	}
	return total
}

// BidDepth returns liquidity for top N levels.
func (b *Orderbook) BidDepth(levels int) decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	total := decimal.Zero
	for i := 0; i < levels && i < len(b.Bids); i++ {
		total = total.Add(b.Bids[i].Value())
	}
	return total
}

// AskDepth returns liquidity for top N levels.
func (b *Orderbook) AskDepth(levels int) decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	total := decimal.Zero
	for i := 0; i < levels && i < len(b.Asks); i++ {
		total = total.Add(b.Asks[i].Value())
	}
	return total
}

// VwapBuy calculates VWAP for buying `amount` quantity.
func (b *Orderbook) VwapBuy(amount decimal.Decimal) *decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return calculateVwap(b.Asks, amount)
}

// VwapSell calculates VWAP for selling `amount` quantity.
func (b *Orderbook) VwapSell(amount decimal.Decimal) *decimal.Decimal {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return calculateVwap(b.Bids, amount)
}

func calculateVwap(levels []PriceLevel, amount decimal.Decimal) *decimal.Decimal {
	remaining := amount
	totalValue := decimal.Zero
	totalQty := decimal.Zero

	for _, level := range levels {
		if remaining.LessThanOrEqual(decimal.Zero) {
			break
		}

		fillQty := decimal.Min(remaining, level.Quantity)
		totalValue = totalValue.Add(fillQty.Mul(level.Price))
		totalQty = totalQty.Add(fillQty)
		remaining = remaining.Sub(fillQty)
	}

	if totalQty.IsZero() {
		return nil
	}

	vwap := totalValue.Div(totalQty)
	return &vwap
}

// HasLiquidity checks if there's enough liquidity for the given side and amount.
func (b *Orderbook) HasLiquidity(side Side, amount decimal.Decimal) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()

	levels := b.Asks
	if side == SideSell {
		levels = b.Bids
	}

	total := decimal.Zero
	for _, level := range levels {
		total = total.Add(level.Quantity)
	}

	return total.GreaterThanOrEqual(amount)
}

// Imbalance returns (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity).
// Positive means more bids, negative means more asks.
func (b *Orderbook) Imbalance() *decimal.Decimal {
	bidLiq := b.BidLiquidity()
	askLiq := b.AskLiquidity()
	total := bidLiq.Add(askLiq)
	if total.IsZero() {
		return nil
	}
	imb := bidLiq.Sub(askLiq).Div(total)
	return &imb
}

// =============================================================================
// AggregatedOrderbook - Multi-venue aggregation
// =============================================================================

// AggregatedLevel represents a price level with venue attribution.
type AggregatedLevel struct {
	Price    decimal.Decimal
	Quantity decimal.Decimal
	Venue    string
}

// AggregatedOrderbook aggregates orderbooks from multiple venues.
type AggregatedOrderbook struct {
	Symbol    string
	Bids      map[string][]AggregatedLevel // price -> []levels (key is price string for map)
	Asks      map[string][]AggregatedLevel
	Timestamp time.Time

	bidPrices []decimal.Decimal // sorted descending
	askPrices []decimal.Decimal // sorted ascending
	mu        sync.RWMutex
}

// NewAggregatedOrderbook creates a new aggregated orderbook.
func NewAggregatedOrderbook(symbol string) *AggregatedOrderbook {
	return &AggregatedOrderbook{
		Symbol:    symbol,
		Bids:      make(map[string][]AggregatedLevel),
		Asks:      make(map[string][]AggregatedLevel),
		Timestamp: time.Now(),
	}
}

// AddOrderbook adds an orderbook from a venue.
func (a *AggregatedOrderbook) AddOrderbook(book *Orderbook) {
	a.mu.Lock()
	defer a.mu.Unlock()

	book.mu.RLock()
	defer book.mu.RUnlock()

	for _, level := range book.Bids {
		key := level.Price.String()
		a.Bids[key] = append(a.Bids[key], AggregatedLevel{
			Price:    level.Price,
			Quantity: level.Quantity,
			Venue:    book.Venue,
		})
	}

	for _, level := range book.Asks {
		key := level.Price.String()
		a.Asks[key] = append(a.Asks[key], AggregatedLevel{
			Price:    level.Price,
			Quantity: level.Quantity,
			Venue:    book.Venue,
		})
	}

	if book.Timestamp.After(a.Timestamp) {
		a.Timestamp = book.Timestamp
	}

	// Rebuild sorted price lists
	a.rebuildPrices()
}

func (a *AggregatedOrderbook) rebuildPrices() {
	// Collect unique bid prices
	bidPriceSet := make(map[string]decimal.Decimal)
	for _, levels := range a.Bids {
		if len(levels) > 0 {
			bidPriceSet[levels[0].Price.String()] = levels[0].Price
		}
	}
	a.bidPrices = make([]decimal.Decimal, 0, len(bidPriceSet))
	for _, p := range bidPriceSet {
		a.bidPrices = append(a.bidPrices, p)
	}
	sort.Slice(a.bidPrices, func(i, j int) bool {
		return a.bidPrices[i].GreaterThan(a.bidPrices[j])
	})

	// Collect unique ask prices
	askPriceSet := make(map[string]decimal.Decimal)
	for _, levels := range a.Asks {
		if len(levels) > 0 {
			askPriceSet[levels[0].Price.String()] = levels[0].Price
		}
	}
	a.askPrices = make([]decimal.Decimal, 0, len(askPriceSet))
	for _, p := range askPriceSet {
		a.askPrices = append(a.askPrices, p)
	}
	sort.Slice(a.askPrices, func(i, j int) bool {
		return a.askPrices[i].LessThan(a.askPrices[j])
	})
}

// BestBid returns the best bid: (price, venue, quantity).
func (a *AggregatedOrderbook) BestBid() *AggregatedLevel {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(a.bidPrices) == 0 {
		return nil
	}

	bestPrice := a.bidPrices[0]
	levels := a.Bids[bestPrice.String()]
	if len(levels) == 0 {
		return nil
	}

	return &levels[0]
}

// BestAsk returns the best ask: (price, venue, quantity).
func (a *AggregatedOrderbook) BestAsk() *AggregatedLevel {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(a.askPrices) == 0 {
		return nil
	}

	bestPrice := a.askPrices[0]
	levels := a.Asks[bestPrice.String()]
	if len(levels) == 0 {
		return nil
	}

	return &levels[0]
}

// AggregatedBids returns bid levels with quantities summed across venues.
func (a *AggregatedOrderbook) AggregatedBids() []PriceLevel {
	a.mu.RLock()
	defer a.mu.RUnlock()

	result := make([]PriceLevel, 0, len(a.bidPrices))
	for _, price := range a.bidPrices {
		levels := a.Bids[price.String()]
		totalQty := decimal.Zero
		for _, l := range levels {
			totalQty = totalQty.Add(l.Quantity)
		}
		result = append(result, PriceLevel{Price: price, Quantity: totalQty})
	}
	return result
}

// AggregatedAsks returns ask levels with quantities summed across venues.
func (a *AggregatedOrderbook) AggregatedAsks() []PriceLevel {
	a.mu.RLock()
	defer a.mu.RUnlock()

	result := make([]PriceLevel, 0, len(a.askPrices))
	for _, price := range a.askPrices {
		levels := a.Asks[price.String()]
		totalQty := decimal.Zero
		for _, l := range levels {
			totalQty = totalQty.Add(l.Quantity)
		}
		result = append(result, PriceLevel{Price: price, Quantity: totalQty})
	}
	return result
}

// BestVenueBuy finds the best venue for buying `amount`: (venue, price).
func (a *AggregatedOrderbook) BestVenueBuy(amount decimal.Decimal) (string, *decimal.Decimal) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	var bestVenue string
	var bestPrice *decimal.Decimal
	remaining := amount

	for _, price := range a.askPrices {
		if remaining.LessThanOrEqual(decimal.Zero) {
			break
		}

		levels := a.Asks[price.String()]
		for _, level := range levels {
			fill := decimal.Min(remaining, level.Quantity)
			if bestPrice == nil || price.LessThan(*bestPrice) {
				bestPrice = &price
				bestVenue = level.Venue
			}
			remaining = remaining.Sub(fill)
			if remaining.LessThanOrEqual(decimal.Zero) {
				break
			}
		}
	}

	return bestVenue, bestPrice
}

// BestVenueSell finds the best venue for selling `amount`: (venue, price).
func (a *AggregatedOrderbook) BestVenueSell(amount decimal.Decimal) (string, *decimal.Decimal) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	var bestVenue string
	var bestPrice *decimal.Decimal
	remaining := amount

	for _, price := range a.bidPrices {
		if remaining.LessThanOrEqual(decimal.Zero) {
			break
		}

		levels := a.Bids[price.String()]
		for _, level := range levels {
			fill := decimal.Min(remaining, level.Quantity)
			if bestPrice == nil || price.GreaterThan(*bestPrice) {
				bestPrice = &price
				bestVenue = level.Venue
			}
			remaining = remaining.Sub(fill)
			if remaining.LessThanOrEqual(decimal.Zero) {
				break
			}
		}
	}

	return bestVenue, bestPrice
}

// VwapBuy calculates VWAP across all venues for buying `amount`.
func (a *AggregatedOrderbook) VwapBuy(amount decimal.Decimal) *decimal.Decimal {
	levels := a.AggregatedAsks()
	return calculateVwap(levels, amount)
}

// VwapSell calculates VWAP across all venues for selling `amount`.
func (a *AggregatedOrderbook) VwapSell(amount decimal.Decimal) *decimal.Decimal {
	levels := a.AggregatedBids()
	return calculateVwap(levels, amount)
}
