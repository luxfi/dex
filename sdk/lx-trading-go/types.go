// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"time"

	"github.com/shopspring/decimal"
)

// Side represents buy or sell.
type Side string

const (
	SideBuy  Side = "buy"
	SideSell Side = "sell"
)

// OrderType represents the type of order.
type OrderType string

const (
	OrderTypeMarket        OrderType = "market"
	OrderTypeLimit         OrderType = "limit"
	OrderTypeLimitMaker    OrderType = "limit_maker"
	OrderTypeStopLoss      OrderType = "stop_loss"
	OrderTypeStopLossLimit OrderType = "stop_loss_limit"
	OrderTypeTakeProfit    OrderType = "take_profit"
)

// TimeInForce specifies how long an order remains active.
type TimeInForce string

const (
	TimeInForceGTC      TimeInForce = "GTC"
	TimeInForceIOC      TimeInForce = "IOC"
	TimeInForceFOK      TimeInForce = "FOK"
	TimeInForcePostOnly TimeInForce = "POST_ONLY"
)

// OrderStatus represents the status of an order.
type OrderStatus string

const (
	OrderStatusPending         OrderStatus = "pending"
	OrderStatusOpen            OrderStatus = "open"
	OrderStatusPartiallyFilled OrderStatus = "partially_filled"
	OrderStatusFilled          OrderStatus = "filled"
	OrderStatusCancelled       OrderStatus = "cancelled"
	OrderStatusRejected        OrderStatus = "rejected"
	OrderStatusExpired         OrderStatus = "expired"
)

// VenueType represents the type of trading venue.
type VenueType string

const (
	VenueTypeNative     VenueType = "native"
	VenueTypeCCXT       VenueType = "ccxt"
	VenueTypeHummingbot VenueType = "hummingbot"
)

// TradingPair represents a trading pair.
type TradingPair struct {
	Base  string
	Quote string
}

// String returns the pair as BASE-QUOTE.
func (p TradingPair) String() string {
	return p.Base + "-" + p.Quote
}

// ToCCXT returns the pair as BASE/QUOTE (CCXT format).
func (p TradingPair) ToCCXT() string {
	return p.Base + "/" + p.Quote
}

// ParseTradingPair parses a symbol into base and quote.
// Supports formats: BASE-QUOTE, BASE/QUOTE, BASEQUOTE (with common quote currencies).
func ParseTradingPair(symbol string) TradingPair {
	// Try dash separator
	for i := 0; i < len(symbol); i++ {
		if symbol[i] == '-' {
			return TradingPair{Base: symbol[:i], Quote: symbol[i+1:]}
		}
	}

	// Try slash separator (CCXT format)
	for i := 0; i < len(symbol); i++ {
		if symbol[i] == '/' {
			return TradingPair{Base: symbol[:i], Quote: symbol[i+1:]}
		}
	}

	// Try underscore separator
	for i := 0; i < len(symbol); i++ {
		if symbol[i] == '_' {
			return TradingPair{Base: symbol[:i], Quote: symbol[i+1:]}
		}
	}

	// Try common quote currencies (USDT, USDC, USD, BTC, ETH)
	quotes := []string{"USDT", "USDC", "USD", "BTC", "ETH", "LUX"}
	for _, q := range quotes {
		if len(symbol) > len(q) && symbol[len(symbol)-len(q):] == q {
			return TradingPair{Base: symbol[:len(symbol)-len(q)], Quote: q}
		}
	}

	return TradingPair{Base: symbol, Quote: ""}
}

// Fee represents a trading fee.
type Fee struct {
	Asset  string
	Amount decimal.Decimal
	Rate   *decimal.Decimal
}

// Balance represents an asset balance.
type Balance struct {
	Asset  string
	Venue  string
	Free   decimal.Decimal
	Locked decimal.Decimal
}

// Total returns free + locked.
func (b Balance) Total() decimal.Decimal {
	return b.Free.Add(b.Locked)
}

// AggregatedBalance represents balance across venues.
type AggregatedBalance struct {
	Asset       string
	TotalFree   decimal.Decimal
	TotalLocked decimal.Decimal
	ByVenue     []Balance
}

// Total returns total balance.
func (b AggregatedBalance) Total() decimal.Decimal {
	return b.TotalFree.Add(b.TotalLocked)
}

// OrderRequest represents an order to be placed.
type OrderRequest struct {
	ClientOrderID string
	Symbol        string
	Side          Side
	OrderType     OrderType
	Quantity      decimal.Decimal
	Price         *decimal.Decimal
	StopPrice     *decimal.Decimal
	TimeInForce   TimeInForce
	ReduceOnly    bool
	PostOnly      bool
	Venue         string
}

// NewMarketOrder creates a market order request.
func NewMarketOrder(symbol string, side Side, quantity decimal.Decimal) OrderRequest {
	return OrderRequest{
		ClientOrderID: generateOrderID(),
		Symbol:        symbol,
		Side:          side,
		OrderType:     OrderTypeMarket,
		Quantity:      quantity,
		TimeInForce:   TimeInForceIOC,
	}
}

// NewLimitOrder creates a limit order request.
func NewLimitOrder(symbol string, side Side, quantity, price decimal.Decimal) OrderRequest {
	return OrderRequest{
		ClientOrderID: generateOrderID(),
		Symbol:        symbol,
		Side:          side,
		OrderType:     OrderTypeLimit,
		Quantity:      quantity,
		Price:         &price,
		TimeInForce:   TimeInForceGTC,
	}
}

// WithVenue sets the target venue.
func (r OrderRequest) WithVenue(venue string) OrderRequest {
	r.Venue = venue
	return r
}

// Order represents an order.
type Order struct {
	OrderID           string
	ClientOrderID     string
	Symbol            string
	Venue             string
	Side              Side
	OrderType         OrderType
	Status            OrderStatus
	Quantity          decimal.Decimal
	FilledQuantity    decimal.Decimal
	RemainingQuantity decimal.Decimal
	Price             *decimal.Decimal
	AveragePrice      *decimal.Decimal
	CreatedAt         time.Time
	UpdatedAt         time.Time
	Fees              []Fee
}

// IsOpen returns true if order is still active.
func (o Order) IsOpen() bool {
	return o.Status == OrderStatusOpen ||
		o.Status == OrderStatusPartiallyFilled ||
		o.Status == OrderStatusPending
}

// IsDone returns true if order is complete.
func (o Order) IsDone() bool {
	return o.Status == OrderStatusFilled ||
		o.Status == OrderStatusCancelled ||
		o.Status == OrderStatusRejected ||
		o.Status == OrderStatusExpired
}

// FillPercent returns the fill percentage.
func (o Order) FillPercent() decimal.Decimal {
	if o.Quantity.IsZero() {
		return decimal.Zero
	}
	return o.FilledQuantity.Div(o.Quantity).Mul(decimal.NewFromInt(100))
}

// Trade represents a trade/fill.
type Trade struct {
	TradeID   string
	OrderID   string
	Symbol    string
	Venue     string
	Side      Side
	Price     decimal.Decimal
	Quantity  decimal.Decimal
	Fee       Fee
	Timestamp time.Time
	IsMaker   bool
}

// Value returns price * quantity.
func (t Trade) Value() decimal.Decimal {
	return t.Price.Mul(t.Quantity)
}

// Ticker represents market ticker data.
type Ticker struct {
	Symbol    string
	Venue     string
	Bid       *decimal.Decimal
	Ask       *decimal.Decimal
	Last      *decimal.Decimal
	Volume24H *decimal.Decimal
	High24H   *decimal.Decimal
	Low24H    *decimal.Decimal
	Change24H *decimal.Decimal
	Timestamp time.Time
}

// MidPrice returns (bid + ask) / 2.
func (t Ticker) MidPrice() *decimal.Decimal {
	if t.Bid != nil && t.Ask != nil {
		mid := t.Bid.Add(*t.Ask).Div(decimal.NewFromInt(2))
		return &mid
	}
	return t.Last
}

// Spread returns ask - bid.
func (t Ticker) Spread() *decimal.Decimal {
	if t.Bid != nil && t.Ask != nil {
		spread := t.Ask.Sub(*t.Bid)
		return &spread
	}
	return nil
}

// SwapQuote represents an AMM swap quote.
type SwapQuote struct {
	BaseToken    string
	QuoteToken   string
	InputAmount  decimal.Decimal
	OutputAmount decimal.Decimal
	Price        decimal.Decimal
	PriceImpact  decimal.Decimal
	Fee          decimal.Decimal
	Route        []string
	ExpiresAt    time.Time
}

// PoolInfo represents AMM pool information.
type PoolInfo struct {
	Address        string
	BaseToken      string
	QuoteToken     string
	BaseReserve    decimal.Decimal
	QuoteReserve   decimal.Decimal
	TotalLiquidity decimal.Decimal
	FeeRate        decimal.Decimal
	APY            *decimal.Decimal
}

// LpPosition represents a liquidity provider position.
type LpPosition struct {
	PoolAddress   string
	BaseToken     string
	QuoteToken    string
	LpTokens      decimal.Decimal
	BaseAmount    decimal.Decimal
	QuoteAmount   decimal.Decimal
	SharePercent  decimal.Decimal
	UnrealizedPnL *decimal.Decimal
}

// LiquidityResult represents the result of add/remove liquidity.
type LiquidityResult struct {
	TxHash       string
	PoolAddress  string
	BaseAmount   decimal.Decimal
	QuoteAmount  decimal.Decimal
	LpTokens     decimal.Decimal
	SharePercent decimal.Decimal
}

// VenueInfo represents venue information.
type VenueInfo struct {
	Name           string
	VenueType      VenueType
	Connected      bool
	LatencyMs      *int64
	SupportedPairs []string
	MakerFee       decimal.Decimal
	TakerFee       decimal.Decimal
}

// MarketInfo represents market/pair information.
type MarketInfo struct {
	Symbol            string
	Base              string
	Quote             string
	PricePrecision    int
	QuantityPrecision int
	MinQuantity       decimal.Decimal
	MaxQuantity       *decimal.Decimal
	MinNotional       *decimal.Decimal
	TickSize          decimal.Decimal
	LotSize           decimal.Decimal
}

// PriceLevel represents a price level in the orderbook.
type PriceLevel struct {
	Price    decimal.Decimal
	Quantity decimal.Decimal
}

// Value returns price * quantity.
func (p PriceLevel) Value() decimal.Decimal {
	return p.Price.Mul(p.Quantity)
}

// Helper functions
func generateOrderID() string {
	return time.Now().Format("20060102150405.000000")
}
