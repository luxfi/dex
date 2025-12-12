// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"context"
	"time"

	"github.com/shopspring/decimal"
)

// Executor is the interface for execution algorithms.
type Executor interface {
	Execute(ctx context.Context) ([]Order, error)
}

// =============================================================================
// TWAP Executor
// =============================================================================

// TwapExecutor implements Time-Weighted Average Price execution.
type TwapExecutor struct {
	client        *Client
	symbol        string
	side          Side
	totalQuantity decimal.Decimal
	numSlices     int
	interval      time.Duration
}

// TwapConfig configures TWAP execution.
type TwapConfig struct {
	Symbol          string
	Side            Side
	TotalQuantity   decimal.Decimal
	DurationSeconds int
	NumSlices       int
}

// NewTwapExecutor creates a TWAP executor.
func NewTwapExecutor(client *Client, cfg TwapConfig) *TwapExecutor {
	interval := time.Duration(cfg.DurationSeconds/cfg.NumSlices) * time.Second
	return &TwapExecutor{
		client:        client,
		symbol:        cfg.Symbol,
		side:          cfg.Side,
		totalQuantity: cfg.TotalQuantity,
		numSlices:     cfg.NumSlices,
		interval:      interval,
	}
}

// Execute runs the TWAP strategy.
func (e *TwapExecutor) Execute(ctx context.Context) ([]Order, error) {
	sliceQty := e.totalQuantity.Div(decimal.NewFromInt(int64(e.numSlices)))
	var orders []Order

	for i := 0; i < e.numSlices; i++ {
		select {
		case <-ctx.Done():
			return orders, ctx.Err()
		default:
		}

		remaining := e.totalQuantity.Sub(sliceQty.Mul(decimal.NewFromInt(int64(i))))
		qty := decimal.Min(sliceQty, remaining)

		if qty.LessThanOrEqual(decimal.Zero) {
			break
		}

		var order Order
		var err error

		if e.side == SideBuy {
			order, err = e.client.Buy(ctx, e.symbol, qty)
		} else {
			order, err = e.client.Sell(ctx, e.symbol, qty)
		}

		if err != nil {
			return orders, err
		}
		orders = append(orders, order)

		// Wait for next slice (except for last)
		if i < e.numSlices-1 {
			select {
			case <-ctx.Done():
				return orders, ctx.Err()
			case <-time.After(e.interval):
			}
		}
	}

	return orders, nil
}

// =============================================================================
// VWAP Executor
// =============================================================================

// VwapExecutor implements Volume-Weighted Average Price execution.
type VwapExecutor struct {
	client            *Client
	symbol            string
	side              Side
	totalQuantity     decimal.Decimal
	participationRate decimal.Decimal
	maxDuration       time.Duration
	checkInterval     time.Duration
}

// VwapConfig configures VWAP execution.
type VwapConfig struct {
	Symbol            string
	Side              Side
	TotalQuantity     decimal.Decimal
	ParticipationRate decimal.Decimal // e.g., 0.1 = 10% of volume
	MaxDurationSeconds int
}

// NewVwapExecutor creates a VWAP executor.
func NewVwapExecutor(client *Client, cfg VwapConfig) *VwapExecutor {
	return &VwapExecutor{
		client:            client,
		symbol:            cfg.Symbol,
		side:              cfg.Side,
		totalQuantity:     cfg.TotalQuantity,
		participationRate: cfg.ParticipationRate,
		maxDuration:       time.Duration(cfg.MaxDurationSeconds) * time.Second,
		checkInterval:     5 * time.Second,
	}
}

// Execute runs the VWAP strategy.
func (e *VwapExecutor) Execute(ctx context.Context) ([]Order, error) {
	var orders []Order
	remaining := e.totalQuantity
	start := time.Now()

	for remaining.GreaterThan(decimal.Zero) {
		// Check timeout
		if time.Since(start) > e.maxDuration {
			break
		}

		select {
		case <-ctx.Done():
			return orders, ctx.Err()
		default:
		}

		// Get current volume
		ticker, err := e.client.Ticker(ctx, e.symbol)
		if err != nil {
			return orders, err
		}

		volume := decimal.NewFromInt(1000) // Default
		if ticker.Volume24H != nil && !ticker.Volume24H.IsZero() {
			volume = *ticker.Volume24H
		}

		// Calculate slice based on participation rate
		// hourlyVolume = volume / 24
		// sliceVolume = hourlyVolume * participationRate / (3600 / checkInterval)
		hourlyVolume := volume.Div(decimal.NewFromInt(24))
		checksPerHour := decimal.NewFromFloat(float64(time.Hour) / float64(e.checkInterval))
		sliceVolume := hourlyVolume.Mul(e.participationRate).Div(checksPerHour)
		qty := decimal.Min(sliceVolume, remaining)

		if qty.GreaterThan(decimal.Zero) {
			var order Order

			if e.side == SideBuy {
				order, err = e.client.Buy(ctx, e.symbol, qty)
			} else {
				order, err = e.client.Sell(ctx, e.symbol, qty)
			}

			if err != nil {
				return orders, err
			}
			orders = append(orders, order)
			remaining = remaining.Sub(order.FilledQuantity)
		}

		// Wait for next check
		select {
		case <-ctx.Done():
			return orders, ctx.Err()
		case <-time.After(e.checkInterval):
		}
	}

	return orders, nil
}

// =============================================================================
// Iceberg Executor
// =============================================================================

// IcebergExecutor implements iceberg order execution.
type IcebergExecutor struct {
	client          *Client
	symbol          string
	side            Side
	totalQuantity   decimal.Decimal
	visibleQuantity decimal.Decimal
	price           decimal.Decimal
	venue           string
	pollInterval    time.Duration
}

// IcebergConfig configures iceberg execution.
type IcebergConfig struct {
	Symbol          string
	Side            Side
	TotalQuantity   decimal.Decimal
	VisibleQuantity decimal.Decimal
	Price           decimal.Decimal
	Venue           string
}

// NewIcebergExecutor creates an iceberg executor.
func NewIcebergExecutor(client *Client, cfg IcebergConfig) *IcebergExecutor {
	return &IcebergExecutor{
		client:          client,
		symbol:          cfg.Symbol,
		side:            cfg.Side,
		totalQuantity:   cfg.TotalQuantity,
		visibleQuantity: cfg.VisibleQuantity,
		price:           cfg.Price,
		venue:           cfg.Venue,
		pollInterval:    500 * time.Millisecond,
	}
}

// Execute runs the iceberg strategy.
func (e *IcebergExecutor) Execute(ctx context.Context) ([]Order, error) {
	var orders []Order
	remaining := e.totalQuantity
	opts := []OrderOption{}
	if e.venue != "" {
		opts = append(opts, WithVenue(e.venue))
	}

	for remaining.GreaterThan(decimal.Zero) {
		select {
		case <-ctx.Done():
			return orders, ctx.Err()
		default:
		}

		qty := decimal.Min(e.visibleQuantity, remaining)

		var order Order
		var err error

		if e.side == SideBuy {
			order, err = e.client.LimitBuy(ctx, e.symbol, qty, e.price, opts...)
		} else {
			order, err = e.client.LimitSell(ctx, e.symbol, qty, e.price, opts...)
		}

		if err != nil {
			return orders, err
		}

		// Wait for fill (simplified - would need order status polling)
		for order.IsOpen() {
			select {
			case <-ctx.Done():
				// Cancel remaining order
				e.client.CancelOrder(ctx, order.OrderID, order.Symbol, order.Venue)
				return orders, ctx.Err()
			case <-time.After(e.pollInterval):
				// In real implementation, refresh order status here
				break
			}
			break // Simplified: assume filled after one poll
		}

		remaining = remaining.Sub(order.FilledQuantity)
		orders = append(orders, order)
	}

	return orders, nil
}

// =============================================================================
// Sniper Executor
// =============================================================================

// SniperExecutor waits for a price target before executing.
type SniperExecutor struct {
	client       *Client
	symbol       string
	side         Side
	quantity     decimal.Decimal
	targetPrice  decimal.Decimal
	timeout      time.Duration
	pollInterval time.Duration
}

// SniperConfig configures sniper execution.
type SniperConfig struct {
	Symbol         string
	Side           Side
	Quantity       decimal.Decimal
	TargetPrice    decimal.Decimal
	TimeoutSeconds int
}

// NewSniperExecutor creates a sniper executor.
func NewSniperExecutor(client *Client, cfg SniperConfig) *SniperExecutor {
	return &SniperExecutor{
		client:       client,
		symbol:       cfg.Symbol,
		side:         cfg.Side,
		quantity:     cfg.Quantity,
		targetPrice:  cfg.TargetPrice,
		timeout:      time.Duration(cfg.TimeoutSeconds) * time.Second,
		pollInterval: 100 * time.Millisecond,
	}
}

// Execute runs the sniper strategy.
func (e *SniperExecutor) Execute(ctx context.Context) ([]Order, error) {
	deadline := time.Now().Add(e.timeout)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		ticker, err := e.client.Ticker(ctx, e.symbol)
		if err != nil {
			return nil, err
		}

		shouldExecute := false

		if e.side == SideBuy {
			if ticker.Ask != nil && ticker.Ask.LessThanOrEqual(e.targetPrice) {
				shouldExecute = true
			}
		} else {
			if ticker.Bid != nil && ticker.Bid.GreaterThanOrEqual(e.targetPrice) {
				shouldExecute = true
			}
		}

		if shouldExecute {
			var order Order

			if e.side == SideBuy {
				order, err = e.client.Buy(ctx, e.symbol, e.quantity)
			} else {
				order, err = e.client.Sell(ctx, e.symbol, e.quantity)
			}

			if err != nil {
				return nil, err
			}
			return []Order{order}, nil
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(e.pollInterval):
		}
	}

	// Timeout - no execution
	return nil, ErrTimeout
}

// =============================================================================
// POV (Percentage of Volume) Executor
// =============================================================================

// PovExecutor implements Percentage of Volume execution.
type PovExecutor struct {
	client            *Client
	symbol            string
	side              Side
	totalQuantity     decimal.Decimal
	participationRate decimal.Decimal
	maxDuration       time.Duration
	minOrderSize      decimal.Decimal
	checkInterval     time.Duration
}

// PovConfig configures POV execution.
type PovConfig struct {
	Symbol            string
	Side              Side
	TotalQuantity     decimal.Decimal
	ParticipationRate decimal.Decimal // e.g., 0.05 = 5% of volume
	MaxDurationSeconds int
	MinOrderSize      decimal.Decimal
}

// NewPovExecutor creates a POV executor.
func NewPovExecutor(client *Client, cfg PovConfig) *PovExecutor {
	minOrder := cfg.MinOrderSize
	if minOrder.IsZero() {
		minOrder = decimal.NewFromFloat(0.001)
	}

	return &PovExecutor{
		client:            client,
		symbol:            cfg.Symbol,
		side:              cfg.Side,
		totalQuantity:     cfg.TotalQuantity,
		participationRate: cfg.ParticipationRate,
		maxDuration:       time.Duration(cfg.MaxDurationSeconds) * time.Second,
		minOrderSize:      minOrder,
		checkInterval:     10 * time.Second,
	}
}

// Execute runs the POV strategy.
func (e *PovExecutor) Execute(ctx context.Context) ([]Order, error) {
	var orders []Order
	remaining := e.totalQuantity
	start := time.Now()
	var lastVolume *decimal.Decimal

	for remaining.GreaterThan(decimal.Zero) {
		if time.Since(start) > e.maxDuration {
			break
		}

		select {
		case <-ctx.Done():
			return orders, ctx.Err()
		default:
		}

		ticker, err := e.client.Ticker(ctx, e.symbol)
		if err != nil {
			return orders, err
		}

		if ticker.Volume24H == nil || ticker.Volume24H.IsZero() {
			select {
			case <-ctx.Done():
				return orders, ctx.Err()
			case <-time.After(e.checkInterval):
			}
			continue
		}

		// Calculate volume delta
		var volumeDelta decimal.Decimal
		if lastVolume != nil {
			volumeDelta = ticker.Volume24H.Sub(*lastVolume)
			if volumeDelta.IsNegative() {
				volumeDelta = decimal.Zero // Volume reset (24h window)
			}
		} else {
			// First iteration: estimate from check interval
			volumeDelta = ticker.Volume24H.Mul(decimal.NewFromFloat(float64(e.checkInterval) / float64(24*time.Hour)))
		}

		lastVolume = ticker.Volume24H

		// Calculate order size as participation rate of volume
		qty := volumeDelta.Mul(e.participationRate)
		qty = decimal.Min(qty, remaining)
		qty = decimal.Max(qty, decimal.Zero)

		if qty.GreaterThanOrEqual(e.minOrderSize) {
			var order Order

			if e.side == SideBuy {
				order, err = e.client.Buy(ctx, e.symbol, qty)
			} else {
				order, err = e.client.Sell(ctx, e.symbol, qty)
			}

			if err != nil {
				return orders, err
			}
			orders = append(orders, order)
			remaining = remaining.Sub(order.FilledQuantity)
		}

		select {
		case <-ctx.Done():
			return orders, ctx.Err()
		case <-time.After(e.checkInterval):
		}
	}

	return orders, nil
}
