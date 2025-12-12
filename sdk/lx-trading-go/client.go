// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"context"
	"fmt"
	"sync"

	"github.com/shopspring/decimal"
)

// Client is the unified trading client that abstracts multiple venues.
//
// It provides a single interface for:
//   - Native LX DEX and LX AMM
//   - CCXT exchanges (Binance, MEXC, OKX, etc.)
//   - Hummingbot Gateway connectors
//
// Example:
//
//	config := NewConfig().
//	    WithNative("lx_dex", NewLxDexConfig("https://api.lx.exchange")).
//	    WithCcxt("binance", NewCcxtConfig("binance").WithCredentials(key, secret))
//
//	client := NewClient(config)
//	ctx := context.Background()
//
//	if err := client.Connect(ctx); err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Disconnect(ctx)
//
//	// Get aggregated orderbook
//	book, _ := client.AggregatedOrderbook(ctx, "BTC-USDC")
//
//	// Smart order routing
//	order, _ := client.Buy(ctx, "BTC-USDC", decimal.NewFromFloat(0.1))
//
//	// Target specific venue
//	order, _ = client.Buy(ctx, "BTC-USDC", decimal.NewFromFloat(0.1), WithVenue("binance"))
type Client struct {
	config       *Config
	venues       map[string]VenueAdapter
	defaultVenue string
	riskManager  *RiskManager

	mu sync.RWMutex
}

// NewClient creates a new unified trading client.
func NewClient(config *Config) *Client {
	return &Client{
		config:      config,
		venues:      make(map[string]VenueAdapter),
		riskManager: NewRiskManager(config.Risk),
	}
}

// Connect connects to all configured venues.
func (c *Client) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Connect native venues
	for name, cfg := range c.config.Native {
		var adapter VenueAdapter
		if cfg.VenueType == "amm" {
			adapter = createAmmAdapter(name, cfg)
		} else {
			adapter = createDexAdapter(name, cfg)
		}

		if err := adapter.Connect(ctx); err != nil {
			return fmt.Errorf("failed to connect to %s: %w", name, err)
		}
		c.venues[name] = adapter
	}

	// Connect CCXT exchanges
	for name, cfg := range c.config.Ccxt {
		adapter := createCcxtAdapter(name, cfg)
		if err := adapter.Connect(ctx); err != nil {
			return fmt.Errorf("failed to connect to %s: %w", name, err)
		}
		c.venues[name] = adapter
	}

	// Connect Hummingbot gateways
	for name, cfg := range c.config.Hummingbot {
		adapter := createHummingbotAdapter(name, cfg)
		if err := adapter.Connect(ctx); err != nil {
			return fmt.Errorf("failed to connect to %s: %w", name, err)
		}
		c.venues[name] = adapter
	}

	// Set default venue
	if len(c.config.General.VenuePriority) > 0 {
		c.defaultVenue = c.config.General.VenuePriority[0]
	} else if len(c.venues) > 0 {
		for name := range c.venues {
			c.defaultVenue = name
			break
		}
	}

	return nil
}

// Disconnect disconnects from all venues.
func (c *Client) Disconnect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, adapter := range c.venues {
		adapter.Disconnect(ctx)
	}
	c.venues = make(map[string]VenueAdapter)
	return nil
}

// Venue returns a specific venue adapter.
func (c *Client) Venue(name string) VenueAdapter {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.venues[name]
}

// Venues returns info about all connected venues.
func (c *Client) Venues() []VenueInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	infos := make([]VenueInfo, 0, len(c.venues))
	for _, adapter := range c.venues {
		infos = append(infos, adapter.Info())
	}
	return infos
}

// RiskManager returns the risk manager.
func (c *Client) RiskManager() *RiskManager {
	return c.riskManager
}

// =============================================================================
// Market Data
// =============================================================================

// Orderbook returns orderbook from a specific venue.
func (c *Client) Orderbook(ctx context.Context, symbol string, opts ...OrderOption) (*Orderbook, error) {
	o := applyOrderOptions(opts)
	venue := c.resolveVenue(o.Venue)
	if venue == "" {
		return nil, ErrVenueNotFound
	}

	adapter := c.venues[venue]
	if adapter == nil {
		return nil, ErrVenueNotFound
	}

	return adapter.GetOrderbook(ctx, symbol, 0)
}

// AggregatedOrderbook returns orderbook aggregated from all venues.
func (c *Client) AggregatedOrderbook(ctx context.Context, symbol string) (*AggregatedOrderbook, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	agg := NewAggregatedOrderbook(symbol)

	for _, adapter := range c.venues {
		if !adapter.Capabilities().Orderbook {
			continue
		}

		book, err := adapter.GetOrderbook(ctx, symbol, 20)
		if err != nil {
			continue // Skip venues that don't support this pair
		}

		agg.AddOrderbook(book)
	}

	return agg, nil
}

// Ticker returns ticker from a specific venue.
func (c *Client) Ticker(ctx context.Context, symbol string, opts ...OrderOption) (Ticker, error) {
	o := applyOrderOptions(opts)
	venue := c.resolveVenue(o.Venue)
	if venue == "" {
		return Ticker{}, ErrVenueNotFound
	}

	adapter := c.venues[venue]
	if adapter == nil {
		return Ticker{}, ErrVenueNotFound
	}

	return adapter.GetTicker(ctx, symbol)
}

// Tickers returns tickers from all venues.
func (c *Client) Tickers(ctx context.Context, symbol string) ([]Ticker, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var tickers []Ticker
	for _, adapter := range c.venues {
		ticker, err := adapter.GetTicker(ctx, symbol)
		if err != nil {
			continue
		}
		tickers = append(tickers, ticker)
	}

	return tickers, nil
}

// =============================================================================
// Account
// =============================================================================

// Balances returns balances aggregated across all venues.
func (c *Client) Balances(ctx context.Context) ([]AggregatedBalance, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	byAsset := make(map[string][]Balance)

	for _, adapter := range c.venues {
		balances, err := adapter.GetBalances(ctx)
		if err != nil {
			continue
		}

		for _, b := range balances {
			byAsset[b.Asset] = append(byAsset[b.Asset], b)
		}
	}

	result := make([]AggregatedBalance, 0, len(byAsset))
	for asset, bals := range byAsset {
		totalFree := decimal.Zero
		totalLocked := decimal.Zero
		for _, b := range bals {
			totalFree = totalFree.Add(b.Free)
			totalLocked = totalLocked.Add(b.Locked)
		}
		result = append(result, AggregatedBalance{
			Asset:       asset,
			TotalFree:   totalFree,
			TotalLocked: totalLocked,
			ByVenue:     bals,
		})
	}

	return result, nil
}

// Balance returns balance for an asset from a specific venue.
func (c *Client) Balance(ctx context.Context, asset string, opts ...OrderOption) (Balance, error) {
	o := applyOrderOptions(opts)
	venue := c.resolveVenue(o.Venue)
	if venue == "" {
		return Balance{}, ErrVenueNotFound
	}

	adapter := c.venues[venue]
	if adapter == nil {
		return Balance{}, ErrVenueNotFound
	}

	return adapter.GetBalance(ctx, asset)
}

// =============================================================================
// Order Management
// =============================================================================

// OrderOption configures order execution.
type OrderOption func(*orderOptions)

type orderOptions struct {
	Venue string
}

func applyOrderOptions(opts []OrderOption) orderOptions {
	o := orderOptions{}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// WithVenue specifies the target venue.
func WithVenue(venue string) OrderOption {
	return func(o *orderOptions) {
		o.Venue = venue
	}
}

// Buy places a market buy order.
func (c *Client) Buy(ctx context.Context, symbol string, quantity decimal.Decimal, opts ...OrderOption) (Order, error) {
	o := applyOrderOptions(opts)
	request := NewMarketOrder(symbol, SideBuy, quantity)
	request.Venue = o.Venue

	return c.executeOrder(ctx, request)
}

// Sell places a market sell order.
func (c *Client) Sell(ctx context.Context, symbol string, quantity decimal.Decimal, opts ...OrderOption) (Order, error) {
	o := applyOrderOptions(opts)
	request := NewMarketOrder(symbol, SideSell, quantity)
	request.Venue = o.Venue

	return c.executeOrder(ctx, request)
}

// LimitBuy places a limit buy order.
func (c *Client) LimitBuy(ctx context.Context, symbol string, quantity, price decimal.Decimal, opts ...OrderOption) (Order, error) {
	o := applyOrderOptions(opts)
	request := NewLimitOrder(symbol, SideBuy, quantity, price)
	request.Venue = o.Venue

	return c.executeOrder(ctx, request)
}

// LimitSell places a limit sell order.
func (c *Client) LimitSell(ctx context.Context, symbol string, quantity, price decimal.Decimal, opts ...OrderOption) (Order, error) {
	o := applyOrderOptions(opts)
	request := NewLimitOrder(symbol, SideSell, quantity, price)
	request.Venue = o.Venue

	return c.executeOrder(ctx, request)
}

// PlaceOrder places an order.
func (c *Client) PlaceOrder(ctx context.Context, request OrderRequest) (Order, error) {
	return c.executeOrder(ctx, request)
}

// CancelOrder cancels an order.
func (c *Client) CancelOrder(ctx context.Context, orderID, symbol, venue string) (Order, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return Order{}, ErrVenueNotFound
	}

	order, err := adapter.CancelOrder(ctx, orderID, symbol)
	if err == nil {
		c.riskManager.OrderClosed(symbol)
	}
	return order, err
}

// CancelAllOrders cancels all orders.
func (c *Client) CancelAllOrders(ctx context.Context, symbol string, opts ...OrderOption) ([]Order, error) {
	o := applyOrderOptions(opts)

	if o.Venue != "" {
		c.mu.RLock()
		adapter := c.venues[o.Venue]
		c.mu.RUnlock()

		if adapter == nil {
			return nil, ErrVenueNotFound
		}

		orders, err := adapter.CancelAllOrders(ctx, symbol)
		for _, order := range orders {
			c.riskManager.OrderClosed(order.Symbol)
		}
		return orders, err
	}

	// Cancel on all venues
	c.mu.RLock()
	defer c.mu.RUnlock()

	var allOrders []Order
	for _, adapter := range c.venues {
		orders, err := adapter.CancelAllOrders(ctx, symbol)
		if err != nil {
			continue
		}
		for _, order := range orders {
			c.riskManager.OrderClosed(order.Symbol)
		}
		allOrders = append(allOrders, orders...)
	}

	return allOrders, nil
}

// OpenOrders returns all open orders across venues.
func (c *Client) OpenOrders(ctx context.Context, symbol string) ([]Order, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var allOrders []Order
	for _, adapter := range c.venues {
		orders, err := adapter.GetOpenOrders(ctx, symbol)
		if err != nil {
			continue
		}
		allOrders = append(allOrders, orders...)
	}

	return allOrders, nil
}

// =============================================================================
// AMM Operations
// =============================================================================

// Quote returns a swap quote.
func (c *Client) Quote(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool, venue string) (SwapQuote, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return SwapQuote{}, ErrVenueNotFound
	}

	return adapter.GetSwapQuote(ctx, baseToken, quoteToken, amount, isBuy)
}

// Swap executes a swap.
func (c *Client) Swap(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool, slippage decimal.Decimal, venue string) (Trade, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return Trade{}, ErrVenueNotFound
	}

	return adapter.ExecuteSwap(ctx, baseToken, quoteToken, amount, isBuy, slippage)
}

// PoolInfo returns pool information.
func (c *Client) PoolInfo(ctx context.Context, baseToken, quoteToken, venue string) (PoolInfo, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return PoolInfo{}, ErrVenueNotFound
	}

	return adapter.GetPoolInfo(ctx, baseToken, quoteToken)
}

// AddLiquidity adds liquidity to a pool.
func (c *Client) AddLiquidity(ctx context.Context, baseToken, quoteToken string, baseAmount, quoteAmount, slippage decimal.Decimal, venue string) (LiquidityResult, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return LiquidityResult{}, ErrVenueNotFound
	}

	return adapter.AddLiquidity(ctx, baseToken, quoteToken, baseAmount, quoteAmount, slippage)
}

// RemoveLiquidity removes liquidity from a pool.
func (c *Client) RemoveLiquidity(ctx context.Context, poolAddress string, liquidityAmount, slippage decimal.Decimal, venue string) (LiquidityResult, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return LiquidityResult{}, ErrVenueNotFound
	}

	return adapter.RemoveLiquidity(ctx, poolAddress, liquidityAmount, slippage)
}

// LpPositions returns LP positions.
func (c *Client) LpPositions(ctx context.Context, venue string) ([]LpPosition, error) {
	c.mu.RLock()
	adapter := c.venues[venue]
	c.mu.RUnlock()

	if adapter == nil {
		return nil, ErrVenueNotFound
	}

	return adapter.GetLpPositions(ctx)
}

// =============================================================================
// Internal
// =============================================================================

func (c *Client) resolveVenue(venue string) string {
	if venue != "" {
		return venue
	}
	return c.defaultVenue
}

func (c *Client) executeOrder(ctx context.Context, request OrderRequest) (Order, error) {
	// Risk check
	if err := c.riskManager.ValidateOrder(request); err != nil {
		return Order{}, err
	}

	var venue string
	var adapter VenueAdapter

	if request.Venue != "" {
		venue = request.Venue
		c.mu.RLock()
		adapter = c.venues[venue]
		c.mu.RUnlock()
	} else if c.config.General.SmartRouting {
		// Smart routing
		venue, adapter = c.smartRoute(ctx, request)
	} else {
		venue = c.defaultVenue
		c.mu.RLock()
		adapter = c.venues[venue]
		c.mu.RUnlock()
	}

	if adapter == nil {
		return Order{}, ErrVenueNotFound
	}

	order, err := adapter.PlaceOrder(ctx, request)
	if err != nil {
		return Order{}, err
	}

	// Update risk state
	c.riskManager.OrderOpened(request.Symbol)
	if order.Status == OrderStatusFilled {
		pair := ParseTradingPair(request.Symbol)
		c.riskManager.UpdatePosition(pair.Base, order.FilledQuantity, order.Side)
		c.riskManager.OrderClosed(request.Symbol)
	}

	return order, nil
}

func (c *Client) smartRoute(ctx context.Context, request OrderRequest) (string, VenueAdapter) {
	aggBook, err := c.AggregatedOrderbook(ctx, request.Symbol)
	if err != nil {
		c.mu.RLock()
		adapter := c.venues[c.defaultVenue]
		c.mu.RUnlock()
		return c.defaultVenue, adapter
	}

	var bestVenue string
	if request.Side == SideBuy {
		bestVenue, _ = aggBook.BestVenueBuy(request.Quantity)
	} else {
		bestVenue, _ = aggBook.BestVenueSell(request.Quantity)
	}

	if bestVenue == "" {
		bestVenue = c.defaultVenue
	}

	c.mu.RLock()
	adapter := c.venues[bestVenue]
	c.mu.RUnlock()

	return bestVenue, adapter
}

// Adapter factory functions - these will be implemented by the adapters package
// but we need placeholders here to avoid import cycles

type adapterFactory interface {
	CreateDexAdapter(name string, cfg *NativeVenueConfig) VenueAdapter
	CreateAmmAdapter(name string, cfg *NativeVenueConfig) VenueAdapter
	CreateCcxtAdapter(name string, cfg *CcxtConfig) VenueAdapter
	CreateHummingbotAdapter(name string, cfg *HummingbotConfig) VenueAdapter
}

var adapterFactoryInstance adapterFactory

// RegisterAdapterFactory registers the adapter factory (called by adapters package).
func RegisterAdapterFactory(f adapterFactory) {
	adapterFactoryInstance = f
}

func createDexAdapter(name string, cfg *NativeVenueConfig) VenueAdapter {
	if adapterFactoryInstance != nil {
		return adapterFactoryInstance.CreateDexAdapter(name, cfg)
	}
	return nil
}

func createAmmAdapter(name string, cfg *NativeVenueConfig) VenueAdapter {
	if adapterFactoryInstance != nil {
		return adapterFactoryInstance.CreateAmmAdapter(name, cfg)
	}
	return nil
}

func createCcxtAdapter(name string, cfg *CcxtConfig) VenueAdapter {
	if adapterFactoryInstance != nil {
		return adapterFactoryInstance.CreateCcxtAdapter(name, cfg)
	}
	return nil
}

func createHummingbotAdapter(name string, cfg *HummingbotConfig) VenueAdapter {
	if adapterFactoryInstance != nil {
		return adapterFactoryInstance.CreateHummingbotAdapter(name, cfg)
	}
	return nil
}
