package client

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"time"
)

// MarketDataSource represents a market data provider
type MarketDataSource struct {
	Name     string  `json:"name"`
	Symbol   string  `json:"symbol"`
	Price    float64 `json:"price"`
	Bid      float64 `json:"bid"`
	Ask      float64 `json:"ask"`
	Volume   float64 `json:"volume"`
	Latency  int64   `json:"latency_ns"`
	Provider string  `json:"provider"`
}

// LiquidationInfo represents liquidation information
type LiquidationInfo struct {
	UserID           string    `json:"user_id"`
	PositionID       string    `json:"position_id"`
	Symbol           string    `json:"symbol"`
	Size             float64   `json:"size"`
	LiquidationPrice float64   `json:"liquidation_price"`
	MarkPrice        float64   `json:"mark_price"`
	Status           string    `json:"status"`
	Timestamp        time.Time `json:"timestamp"`
}

// SettlementBatch represents a batch of settlements
type SettlementBatch struct {
	BatchID   uint64    `json:"batch_id"`
	Orders    []uint64  `json:"order_ids"`
	Status    string    `json:"status"`
	TxHash    string    `json:"tx_hash,omitempty"`
	GasUsed   uint64    `json:"gas_used,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// MarginInfo represents margin information
type MarginInfo struct {
	UserID            string  `json:"user_id"`
	InitialMargin     float64 `json:"initial_margin"`
	MaintenanceMargin float64 `json:"maintenance_margin"`
	MarginRatio       float64 `json:"margin_ratio"`
	FreeMargin        float64 `json:"free_margin"`
	MarginLevel       float64 `json:"margin_level"`
}

// GetMarketData retrieves market data from a specific source
func (c *Client) GetMarketData(ctx context.Context, symbol, source string) (*MarketDataSource, error) {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "market_data.get",
		"params": map[string]string{
			"symbol": symbol,
			"source": source,
		},
		"id": atomic.AddUint64(&c.idCounter, 1),
	}

	resp, err := c.sendJSONRPCRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var data MarketDataSource
	if err := json.Unmarshal(resp.Result, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal market data: %w", err)
	}

	return &data, nil
}

// GetAggregatedMarketData retrieves aggregated market data from all sources
func (c *Client) GetAggregatedMarketData(ctx context.Context, symbol string) ([]MarketDataSource, error) {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "market_data.aggregate",
		"params": map[string]string{
			"symbol": symbol,
		},
		"id": atomic.AddUint64(&c.idCounter, 1),
	}

	resp, err := c.sendJSONRPCRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var data []MarketDataSource
	if err := json.Unmarshal(resp.Result, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal aggregated market data: %w", err)
	}

	return data, nil
}

// GetLiquidations retrieves recent liquidations
func (c *Client) GetLiquidations(ctx context.Context, symbol string, limit int) ([]LiquidationInfo, error) {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "liquidations.get",
		"params": map[string]interface{}{
			"symbol": symbol,
			"limit":  limit,
		},
		"id": atomic.AddUint64(&c.idCounter, 1),
	}

	resp, err := c.sendJSONRPCRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var data []LiquidationInfo
	if err := json.Unmarshal(resp.Result, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal liquidations: %w", err)
	}

	return data, nil
}

// GetSettlementBatch retrieves settlement batch information
func (c *Client) GetSettlementBatch(ctx context.Context, batchID uint64) (*SettlementBatch, error) {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "settlement.batch",
		"params": map[string]interface{}{
			"batch_id": batchID,
		},
		"id": atomic.AddUint64(&c.idCounter, 1),
	}

	resp, err := c.sendJSONRPCRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var data SettlementBatch
	if err := json.Unmarshal(resp.Result, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal settlement batch: %w", err)
	}

	return &data, nil
}

// GetMarginInfo retrieves margin information for a user
func (c *Client) GetMarginInfo(ctx context.Context, userID string) (*MarginInfo, error) {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "margin.info",
		"params": map[string]string{
			"user_id": userID,
		},
		"id": atomic.AddUint64(&c.idCounter, 1),
	}

	resp, err := c.sendJSONRPCRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var data MarginInfo
	if err := json.Unmarshal(resp.Result, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal margin info: %w", err)
	}

	return &data, nil
}

// SubscribeToLiquidations subscribes to liquidation events
func (c *Client) SubscribeToLiquidations(callback func(*LiquidationInfo)) error {
	c.wsMu.Lock()
	c.wsCallbacks["liquidations"] = func(data interface{}) {
		if liq, ok := data.(*LiquidationInfo); ok {
			callback(liq)
		}
	}
	c.wsMu.Unlock()

	return c.sendWSMessage(map[string]interface{}{
		"type":    "subscribe",
		"channel": "liquidations",
	})
}

// SubscribeToSettlements subscribes to settlement events
func (c *Client) SubscribeToSettlements(callback func(*SettlementBatch)) error {
	c.wsMu.Lock()
	c.wsCallbacks["settlements"] = func(data interface{}) {
		if batch, ok := data.(*SettlementBatch); ok {
			callback(batch)
		}
	}
	c.wsMu.Unlock()

	return c.sendWSMessage(map[string]interface{}{
		"type":    "subscribe",
		"channel": "settlements",
	})
}
