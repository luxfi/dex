// Package adapters provides venue adapter implementations.
package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/luxfi/trading"
	"github.com/shopspring/decimal"
)

// HummingbotAdapter implements VenueAdapter for Hummingbot Gateway.
type HummingbotAdapter struct {
	trading.BaseAdapter
	config *trading.HummingbotConfig
	client *http.Client
	name   string
}

// NewHummingbotAdapter creates a new Hummingbot Gateway adapter.
func NewHummingbotAdapter(name string, config *trading.HummingbotConfig) *HummingbotAdapter {
	return &HummingbotAdapter{
		name:   name,
		config: config,
	}
}

// Name returns the venue name.
func (a *HummingbotAdapter) Name() string {
	return a.name
}

// VenueType returns the venue type.
func (a *HummingbotAdapter) VenueType() trading.VenueType {
	return trading.VenueTypeHummingbot
}

// Capabilities returns venue capabilities.
func (a *HummingbotAdapter) Capabilities() trading.VenueCapabilities {
	return trading.AMMCapabilities()
}

// Connect establishes connection to the Gateway.
func (a *HummingbotAdapter) Connect(ctx context.Context) error {
	a.client = &http.Client{Timeout: 30 * time.Second}

	// Test connection
	start := time.Now()
	resp, err := a.client.Get(a.config.BaseURL())
	if err != nil {
		return fmt.Errorf("failed to connect to Hummingbot Gateway: %w", err)
	}
	defer resp.Body.Close()

	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return err
	}

	if getString(data, "status") != "ok" {
		return fmt.Errorf("gateway not ready")
	}

	a.SetLatency(time.Since(start).Milliseconds())
	a.SetConnected(true)
	return nil
}

// Disconnect closes the connection.
func (a *HummingbotAdapter) Disconnect(ctx context.Context) error {
	a.client = nil
	a.SetConnected(false)
	return nil
}

func (a *HummingbotAdapter) doRequest(ctx context.Context, method, path string, body map[string]interface{}) (map[string]interface{}, error) {
	if a.client == nil {
		return nil, trading.ErrNotConnected
	}

	// Add gateway parameters
	if body == nil {
		body = make(map[string]interface{})
	}
	body["chain"] = a.config.Chain
	body["network"] = a.config.Network
	body["connector"] = a.config.Connector
	if a.config.WalletAddress != "" {
		body["address"] = a.config.WalletAddress
	}

	url := a.config.BaseURL() + path

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	a.SetLatency(time.Since(start).Milliseconds())

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, trading.NewTradingError("HTTP_ERROR", fmt.Sprintf("status %d: %s", resp.StatusCode, string(body)))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

func (a *HummingbotAdapter) doRequestArray(ctx context.Context, method, path string, body map[string]interface{}) ([]map[string]interface{}, error) {
	if a.client == nil {
		return nil, trading.ErrNotConnected
	}

	// Add gateway parameters
	if body == nil {
		body = make(map[string]interface{})
	}
	body["chain"] = a.config.Chain
	body["network"] = a.config.Network
	body["connector"] = a.config.Connector
	if a.config.WalletAddress != "" {
		body["address"] = a.config.WalletAddress
	}

	url := a.config.BaseURL() + path

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	a.SetLatency(time.Since(start).Milliseconds())

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, trading.NewTradingError("HTTP_ERROR", fmt.Sprintf("status %d: %s", resp.StatusCode, string(body)))
	}

	var result []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		// Try single object response
		return nil, err
	}

	return result, nil
}

// GetMarkets returns available trading pairs from gateway tokens.
func (a *HummingbotAdapter) GetMarkets(ctx context.Context) ([]trading.MarketInfo, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/amm/tokens", nil)
	if err != nil {
		return nil, err
	}

	tokens, ok := data["tokens"].([]interface{})
	if !ok {
		return nil, nil
	}

	var markets []trading.MarketInfo

	// Create pairs from all token combinations
	for i, t1 := range tokens {
		tok1, ok := t1.(map[string]interface{})
		if !ok {
			continue
		}
		symbol1 := getString(tok1, "symbol")

		for _, t2 := range tokens[i+1:] {
			tok2, ok := t2.(map[string]interface{})
			if !ok {
				continue
			}
			symbol2 := getString(tok2, "symbol")

			if symbol1 != "" && symbol2 != "" {
				markets = append(markets, trading.MarketInfo{
					Symbol:            symbol1 + "-" + symbol2,
					Base:              symbol1,
					Quote:             symbol2,
					PricePrecision:    8,
					QuantityPrecision: 8,
					MinQuantity:       decimal.Zero,
					TickSize:          decimal.NewFromFloat(0.00000001),
					LotSize:           decimal.NewFromFloat(0.00000001),
				})
			}
		}
	}

	return markets, nil
}

// GetTicker returns ticker for a symbol.
func (a *HummingbotAdapter) GetTicker(ctx context.Context, symbol string) (trading.Ticker, error) {
	pair := trading.ParseTradingPair(symbol)

	data, err := a.doRequest(ctx, http.MethodPost, "/amm/price", map[string]interface{}{
		"base":   pair.Base,
		"quote":  pair.Quote,
		"amount": "1",
		"side":   "BUY",
	})
	if err != nil {
		return trading.Ticker{}, err
	}

	ticker := trading.Ticker{
		Symbol:    symbol,
		Venue:     a.name,
		Timestamp: time.Now(),
	}

	if price := getFloat(data, "price"); price > 0 {
		d := decimal.NewFromFloat(price)
		ticker.Bid = &d
		ticker.Ask = &d
		ticker.Last = &d
	}

	return ticker, nil
}

// GetOrderbook is not supported for Gateway AMM.
func (a *HummingbotAdapter) GetOrderbook(ctx context.Context, symbol string, depth int) (*trading.Orderbook, error) {
	return nil, trading.ErrNotSupported
}

// GetTrades returns empty list (Gateway doesn't provide trade history).
func (a *HummingbotAdapter) GetTrades(ctx context.Context, symbol string, limit int) ([]trading.Trade, error) {
	return []trading.Trade{}, nil
}

// GetBalances returns all balances.
func (a *HummingbotAdapter) GetBalances(ctx context.Context) ([]trading.Balance, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/chain/balances", nil)
	if err != nil {
		return nil, err
	}

	balances, ok := data["balances"].(map[string]interface{})
	if !ok {
		return nil, nil
	}

	var result []trading.Balance
	for asset, amount := range balances {
		result = append(result, trading.Balance{
			Asset:  asset,
			Venue:  a.name,
			Free:   decimal.NewFromFloat(toFloat(amount)),
			Locked: decimal.Zero,
		})
	}

	return result, nil
}

// GetBalance returns balance for a specific asset.
func (a *HummingbotAdapter) GetBalance(ctx context.Context, asset string) (trading.Balance, error) {
	balances, err := a.GetBalances(ctx)
	if err != nil {
		return trading.Balance{}, err
	}

	for _, b := range balances {
		if b.Asset == asset {
			return b, nil
		}
	}

	return trading.Balance{
		Asset:  asset,
		Venue:  a.name,
		Free:   decimal.Zero,
		Locked: decimal.Zero,
	}, nil
}

// GetOpenOrders returns empty list (AMM doesn't have orders).
func (a *HummingbotAdapter) GetOpenOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	return []trading.Order{}, nil
}

// PlaceOrder executes a swap as an order.
func (a *HummingbotAdapter) PlaceOrder(ctx context.Context, request trading.OrderRequest) (trading.Order, error) {
	pair := trading.ParseTradingPair(request.Symbol)

	trade, err := a.ExecuteSwap(ctx, pair.Base, pair.Quote, request.Quantity, request.Side == trading.SideBuy, decimal.NewFromFloat(0.01))
	if err != nil {
		return trading.Order{}, err
	}

	return trading.Order{
		OrderID:           trade.TradeID,
		ClientOrderID:     request.ClientOrderID,
		Symbol:            request.Symbol,
		Venue:             a.name,
		Side:              request.Side,
		OrderType:         trading.OrderTypeMarket,
		Status:            trading.OrderStatusFilled,
		Quantity:          request.Quantity,
		FilledQuantity:    trade.Quantity,
		RemainingQuantity: decimal.Zero,
		Price:             &trade.Price,
		AveragePrice:      &trade.Price,
		CreatedAt:         trade.Timestamp,
		UpdatedAt:         trade.Timestamp,
		Fees:              []trading.Fee{trade.Fee},
	}, nil
}

// CancelOrder is not supported for Gateway AMM.
func (a *HummingbotAdapter) CancelOrder(ctx context.Context, orderID, symbol string) (trading.Order, error) {
	return trading.Order{}, trading.ErrNotSupported
}

// CancelAllOrders returns empty list.
func (a *HummingbotAdapter) CancelAllOrders(ctx context.Context, symbol string) ([]trading.Order, error) {
	return []trading.Order{}, nil
}

// GetSwapQuote returns a swap quote.
func (a *HummingbotAdapter) GetSwapQuote(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool) (trading.SwapQuote, error) {
	side := "SELL"
	if isBuy {
		side = "BUY"
	}

	data, err := a.doRequest(ctx, http.MethodPost, "/amm/price", map[string]interface{}{
		"base":   baseToken,
		"quote":  quoteToken,
		"amount": amount.String(),
		"side":   side,
	})
	if err != nil {
		return trading.SwapQuote{}, err
	}

	return trading.SwapQuote{
		BaseToken:    baseToken,
		QuoteToken:   quoteToken,
		InputAmount:  amount,
		OutputAmount: decimal.NewFromFloat(getFloat(data, "expectedAmount")),
		Price:        decimal.NewFromFloat(getFloat(data, "price")),
		PriceImpact:  decimal.Zero,
		Fee:          decimal.Zero,
		Route:        nil,
		ExpiresAt:    time.Now().Add(60 * time.Second),
	}, nil
}

// ExecuteSwap executes a swap.
func (a *HummingbotAdapter) ExecuteSwap(ctx context.Context, baseToken, quoteToken string, amount decimal.Decimal, isBuy bool, slippage decimal.Decimal) (trading.Trade, error) {
	side := "SELL"
	if isBuy {
		side = "BUY"
	}

	data, err := a.doRequest(ctx, http.MethodPost, "/amm/trade", map[string]interface{}{
		"base":           baseToken,
		"quote":          quoteToken,
		"amount":         amount.String(),
		"side":           side,
		"limitPrice":     "",
		"allowedSlippage": fmt.Sprintf("%s/100", slippage.String()),
	})
	if err != nil {
		return trading.Trade{}, err
	}

	tradeSide := trading.SideSell
	if isBuy {
		tradeSide = trading.SideBuy
	}

	return trading.Trade{
		TradeID:   getString(data, "txHash"),
		OrderID:   getString(data, "txHash"),
		Symbol:    baseToken + "-" + quoteToken,
		Venue:     a.name,
		Side:      tradeSide,
		Price:     decimal.NewFromFloat(getFloat(data, "price")),
		Quantity:  amount,
		Fee:       trading.Fee{Asset: "GAS", Amount: decimal.NewFromFloat(getFloat(data, "gasPrice"))},
		Timestamp: time.Now(),
		IsMaker:   false,
	}, nil
}

// GetPoolInfo returns pool information.
func (a *HummingbotAdapter) GetPoolInfo(ctx context.Context, baseToken, quoteToken string) (trading.PoolInfo, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/amm/poolPrice", map[string]interface{}{
		"token0": baseToken,
		"token1": quoteToken,
	})
	if err != nil {
		return trading.PoolInfo{}, err
	}

	return trading.PoolInfo{
		Address:        getString(data, "token0Address"),
		BaseToken:      baseToken,
		QuoteToken:     quoteToken,
		BaseReserve:    decimal.NewFromFloat(getFloat(data, "token0Balance")),
		QuoteReserve:   decimal.NewFromFloat(getFloat(data, "token1Balance")),
		TotalLiquidity: decimal.Zero,
		FeeRate:        decimal.NewFromFloat(0.003),
	}, nil
}

// AddLiquidity adds liquidity to a pool.
func (a *HummingbotAdapter) AddLiquidity(ctx context.Context, baseToken, quoteToken string, baseAmount, quoteAmount, slippage decimal.Decimal) (trading.LiquidityResult, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/amm/liquidity/add", map[string]interface{}{
		"token0":          baseToken,
		"token1":          quoteToken,
		"amount0":         baseAmount.String(),
		"amount1":         quoteAmount.String(),
		"allowedSlippage": fmt.Sprintf("%s/100", slippage.String()),
	})
	if err != nil {
		return trading.LiquidityResult{}, err
	}

	return trading.LiquidityResult{
		TxHash:       getString(data, "txHash"),
		PoolAddress:  getString(data, "poolAddress"),
		BaseAmount:   baseAmount,
		QuoteAmount:  quoteAmount,
		LpTokens:     decimal.Zero,
		SharePercent: decimal.Zero,
	}, nil
}

// RemoveLiquidity removes liquidity from a pool.
func (a *HummingbotAdapter) RemoveLiquidity(ctx context.Context, poolAddress string, liquidityAmount, slippage decimal.Decimal) (trading.LiquidityResult, error) {
	data, err := a.doRequest(ctx, http.MethodPost, "/amm/liquidity/remove", map[string]interface{}{
		"tokenId":         poolAddress,
		"decreasePercent": "100",
		"allowedSlippage": fmt.Sprintf("%s/100", slippage.String()),
	})
	if err != nil {
		return trading.LiquidityResult{}, err
	}

	return trading.LiquidityResult{
		TxHash:       getString(data, "txHash"),
		PoolAddress:  poolAddress,
		BaseAmount:   decimal.Zero,
		QuoteAmount:  decimal.Zero,
		LpTokens:     liquidityAmount,
		SharePercent: decimal.Zero,
	}, nil
}

// GetLpPositions returns LP positions.
func (a *HummingbotAdapter) GetLpPositions(ctx context.Context) ([]trading.LpPosition, error) {
	data, err := a.doRequestArray(ctx, http.MethodPost, "/amm/position", nil)
	if err != nil {
		return nil, err
	}

	positions := make([]trading.LpPosition, len(data))
	for i, p := range data {
		positions[i] = trading.LpPosition{
			PoolAddress:  getString(p, "tokenId"),
			BaseToken:    getString(p, "token0"),
			QuoteToken:   getString(p, "token1"),
			LpTokens:     decimal.Zero,
			BaseAmount:   decimal.NewFromFloat(getFloat(p, "amount0")),
			QuoteAmount:  decimal.NewFromFloat(getFloat(p, "amount1")),
			SharePercent: decimal.Zero,
		}

		if unclaimed := getFloat(p, "unclaimedToken0"); unclaimed > 0 {
			d := decimal.NewFromFloat(unclaimed)
			positions[i].UnrealizedPnL = &d
		}
	}

	return positions, nil
}
