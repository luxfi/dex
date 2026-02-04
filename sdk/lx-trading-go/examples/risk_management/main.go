// Copyright 2024 Lux Partners Limited. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example: Position and Risk Tracking
//
// This example demonstrates comprehensive risk management:
// - Position tracking across venues
// - PnL monitoring
// - Risk limit enforcement
// - Kill switch functionality
// - Portfolio-level metrics
//
// Run with: go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	trading "github.com/luxfi/trading"
	_ "github.com/luxfi/trading/adapters"
	"github.com/shopspring/decimal"
)

// RiskMonitor provides real-time risk monitoring and enforcement.
type RiskMonitor struct {
	client      *trading.Client
	riskManager *trading.RiskManager

	// Configuration
	config RiskConfig

	// State
	mu          sync.RWMutex
	positions   map[string]Position // asset -> position
	pnlHistory  []PnLSnapshot
	alerts      []Alert
	lastBalance map[string]decimal.Decimal

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// RiskConfig configures risk monitoring.
type RiskConfig struct {
	// Position limits
	MaxPositionUSD   decimal.Decimal
	MaxPositionAsset map[string]decimal.Decimal

	// PnL limits
	MaxDailyLoss decimal.Decimal
	MaxDrawdown  decimal.Decimal
	ProfitTarget decimal.Decimal

	// Alert thresholds
	AlertPositionPct decimal.Decimal // Alert when position > X% of limit
	AlertLossPct     decimal.Decimal // Alert when loss > X% of limit

	// Kill switch
	EnableKillSwitch bool
	KillOnMaxLoss    bool

	// Monitoring
	CheckInterval time.Duration
}

// Position represents a position in an asset.
type Position struct {
	Asset         string
	Quantity      decimal.Decimal
	AvgEntryPrice decimal.Decimal
	MarketPrice   decimal.Decimal
	UnrealizedPnL decimal.Decimal
	RealizedPnL   decimal.Decimal
	ByVenue       map[string]decimal.Decimal
	UpdatedAt     time.Time
}

// PnLSnapshot is a point-in-time PnL record.
type PnLSnapshot struct {
	Timestamp     time.Time
	UnrealizedPnL decimal.Decimal
	RealizedPnL   decimal.Decimal
	TotalPnL      decimal.Decimal
}

// Alert represents a risk alert.
type Alert struct {
	Timestamp time.Time
	Level     string // "WARNING", "CRITICAL"
	Type      string
	Message   string
}

// DefaultRiskConfig returns sensible defaults.
func DefaultRiskConfig() RiskConfig {
	return RiskConfig{
		MaxPositionUSD:   decimal.NewFromFloat(100000),
		MaxPositionAsset: make(map[string]decimal.Decimal),
		MaxDailyLoss:     decimal.NewFromFloat(5000),
		MaxDrawdown:      decimal.NewFromFloat(10000),
		ProfitTarget:     decimal.NewFromFloat(10000),
		AlertPositionPct: decimal.NewFromFloat(80),
		AlertLossPct:     decimal.NewFromFloat(50),
		EnableKillSwitch: true,
		KillOnMaxLoss:    true,
		CheckInterval:    5 * time.Second,
	}
}

// NewRiskMonitor creates a new risk monitor.
func NewRiskMonitor(client *trading.Client, config RiskConfig) *RiskMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	return &RiskMonitor{
		client:      client,
		riskManager: client.RiskManager(),
		config:      config,
		positions:   make(map[string]Position),
		lastBalance: make(map[string]decimal.Decimal),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start begins risk monitoring.
func (rm *RiskMonitor) Start() {
	log.Println("Starting risk monitor")
	log.Printf("Max position: $%s, Max daily loss: $%s",
		rm.config.MaxPositionUSD, rm.config.MaxDailyLoss)

	rm.wg.Add(1)
	go rm.monitorLoop()
}

// Stop stops risk monitoring.
func (rm *RiskMonitor) Stop() {
	rm.cancel()
	rm.wg.Wait()
	rm.printFinalReport()
}

// monitorLoop continuously monitors risk metrics.
func (rm *RiskMonitor) monitorLoop() {
	defer rm.wg.Done()

	ticker := time.NewTicker(rm.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.checkRisk()
		}
	}
}

// checkRisk performs risk checks.
func (rm *RiskMonitor) checkRisk() {
	// Update positions from balances
	rm.updatePositions()

	// Check PnL
	rm.checkPnL()

	// Check position limits
	rm.checkPositionLimits()

	// Check kill switch
	rm.checkKillSwitch()
}

// updatePositions updates position tracking from venue balances.
func (rm *RiskMonitor) updatePositions() {
	balances, err := rm.client.Balances(rm.ctx)
	if err != nil {
		return
	}

	rm.mu.Lock()
	defer rm.mu.Unlock()

	for _, bal := range balances {
		pos, ok := rm.positions[bal.Asset]
		if !ok {
			pos = Position{
				Asset:   bal.Asset,
				ByVenue: make(map[string]decimal.Decimal),
			}
		}

		// Track quantity by venue
		for _, vb := range bal.ByVenue {
			pos.ByVenue[vb.Venue] = vb.Total()
		}

		// Total quantity
		pos.Quantity = bal.Total()
		pos.UpdatedAt = time.Now()

		// Calculate PnL change
		lastTotal, hasLast := rm.lastBalance[bal.Asset]
		if hasLast {
			change := bal.Total().Sub(lastTotal)
			if !change.IsZero() {
				// Simplified: treat balance changes as realized PnL
				// In production, track actual trade fills
				pos.RealizedPnL = pos.RealizedPnL.Add(change)
			}
		}
		rm.lastBalance[bal.Asset] = bal.Total()

		rm.positions[bal.Asset] = pos
	}
}

// checkPnL checks PnL limits.
func (rm *RiskMonitor) checkPnL() {
	// Get risk manager state
	pnl := rm.riskManager.DailyPnL()

	rm.mu.Lock()

	// Record snapshot
	snapshot := PnLSnapshot{
		Timestamp: time.Now(),
		TotalPnL:  pnl,
	}
	rm.pnlHistory = append(rm.pnlHistory, snapshot)

	// Keep last 24 hours
	cutoff := time.Now().Add(-24 * time.Hour)
	filtered := rm.pnlHistory[:0]
	for _, s := range rm.pnlHistory {
		if s.Timestamp.After(cutoff) {
			filtered = append(filtered, s)
		}
	}
	rm.pnlHistory = filtered

	rm.mu.Unlock()

	// Check loss limit
	if pnl.IsNegative() {
		lossAbs := pnl.Abs()
		lossPct := lossAbs.Div(rm.config.MaxDailyLoss).Mul(decimal.NewFromInt(100))

		if lossPct.GreaterThan(rm.config.AlertLossPct) {
			rm.addAlert("WARNING", "LOSS",
				fmt.Sprintf("Daily loss at %s%% of limit ($%s)",
					lossPct.StringFixed(1), lossAbs.StringFixed(2)))
		}

		if lossAbs.GreaterThanOrEqual(rm.config.MaxDailyLoss) {
			rm.addAlert("CRITICAL", "LOSS",
				fmt.Sprintf("Daily loss limit reached: $%s", lossAbs.StringFixed(2)))

			if rm.config.KillOnMaxLoss {
				rm.triggerKillSwitch("Max daily loss reached")
			}
		}
	}

	// Check profit target
	if pnl.GreaterThanOrEqual(rm.config.ProfitTarget) {
		rm.addAlert("WARNING", "PROFIT",
			fmt.Sprintf("Profit target reached: $%s", pnl.StringFixed(2)))
	}
}

// checkPositionLimits checks position limits.
func (rm *RiskMonitor) checkPositionLimits() {
	rm.mu.RLock()
	positions := rm.positions
	rm.mu.RUnlock()

	for _, pos := range positions {
		if pos.Quantity.IsZero() {
			continue
		}

		// Check asset-specific limit
		if limit, ok := rm.config.MaxPositionAsset[pos.Asset]; ok {
			usagePct := pos.Quantity.Abs().Div(limit).Mul(decimal.NewFromInt(100))

			if usagePct.GreaterThan(rm.config.AlertPositionPct) {
				rm.addAlert("WARNING", "POSITION",
					fmt.Sprintf("%s position at %s%% of limit (%s / %s)",
						pos.Asset, usagePct.StringFixed(1),
						pos.Quantity.StringFixed(4), limit.StringFixed(4)))
			}
		}
	}
}

// checkKillSwitch checks if kill switch should be triggered.
func (rm *RiskMonitor) checkKillSwitch() {
	if rm.riskManager.IsKilled() {
		rm.addAlert("CRITICAL", "KILL_SWITCH", "Kill switch is active - trading halted")
	}
}

// triggerKillSwitch activates the kill switch.
func (rm *RiskMonitor) triggerKillSwitch(reason string) {
	if !rm.config.EnableKillSwitch {
		return
	}

	log.Printf("KILL SWITCH ACTIVATED: %s", reason)
	rm.riskManager.Kill()
	rm.addAlert("CRITICAL", "KILL_SWITCH", reason)

	// Cancel all orders
	for _, symbol := range []string{"BTC-USDC", "ETH-USDC", "LUX-USDC"} {
		rm.client.CancelAllOrders(rm.ctx, symbol)
	}
}

// addAlert adds a risk alert.
func (rm *RiskMonitor) addAlert(level, alertType, message string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	alert := Alert{
		Timestamp: time.Now(),
		Level:     level,
		Type:      alertType,
		Message:   message,
	}

	rm.alerts = append(rm.alerts, alert)

	// Log the alert
	log.Printf("[%s] %s: %s", level, alertType, message)

	// Keep last 100 alerts
	if len(rm.alerts) > 100 {
		rm.alerts = rm.alerts[len(rm.alerts)-100:]
	}
}

// GetSummary returns a risk summary.
func (rm *RiskMonitor) GetSummary() RiskSummary {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	summary := RiskSummary{
		Timestamp:  time.Now(),
		DailyPnL:   rm.riskManager.DailyPnL(),
		IsKilled:   rm.riskManager.IsKilled(),
		Positions:  make(map[string]decimal.Decimal),
		AlertCount: len(rm.alerts),
	}

	for asset, pos := range rm.positions {
		if !pos.Quantity.IsZero() {
			summary.Positions[asset] = pos.Quantity
		}
	}

	return summary
}

// RiskSummary provides a snapshot of risk state.
type RiskSummary struct {
	Timestamp  time.Time
	DailyPnL   decimal.Decimal
	IsKilled   bool
	Positions  map[string]decimal.Decimal
	AlertCount int
}

// PrintStatus prints current risk status.
func (rm *RiskMonitor) PrintStatus() {
	summary := rm.GetSummary()

	fmt.Printf("\n=== Risk Status (%s) ===\n", summary.Timestamp.Format("15:04:05"))
	fmt.Printf("Daily PnL:    $%s\n", summary.DailyPnL.StringFixed(2))
	fmt.Printf("Kill Switch:  %v\n", summary.IsKilled)
	fmt.Printf("Alerts:       %d\n", summary.AlertCount)

	if len(summary.Positions) > 0 {
		fmt.Println("\nPositions:")
		for asset, qty := range summary.Positions {
			fmt.Printf("  %s: %s\n", asset, qty.StringFixed(8))
		}
	}

	// Recent alerts
	rm.mu.RLock()
	if len(rm.alerts) > 0 {
		fmt.Println("\nRecent Alerts:")
		start := len(rm.alerts) - 5
		if start < 0 {
			start = 0
		}
		for _, a := range rm.alerts[start:] {
			fmt.Printf("  [%s] %s: %s (%s)\n",
				a.Level, a.Type, a.Message,
				a.Timestamp.Format("15:04:05"))
		}
	}
	rm.mu.RUnlock()

	fmt.Println("=============================")
}

// printFinalReport prints final risk report.
func (rm *RiskMonitor) printFinalReport() {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	fmt.Println("\n=== Final Risk Report ===")
	fmt.Printf("Session End: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Printf("Daily PnL:   $%s\n", rm.riskManager.DailyPnL().StringFixed(2))
	fmt.Printf("Kill Switch: %v\n", rm.riskManager.IsKilled())
	fmt.Printf("Total Alerts: %d\n", len(rm.alerts))

	// Count alerts by type
	alertCounts := make(map[string]int)
	for _, a := range rm.alerts {
		alertCounts[a.Type]++
	}
	if len(alertCounts) > 0 {
		fmt.Println("\nAlerts by Type:")
		for t, c := range alertCounts {
			fmt.Printf("  %s: %d\n", t, c)
		}
	}

	// Final positions
	fmt.Println("\nFinal Positions:")
	for asset, pos := range rm.positions {
		if !pos.Quantity.IsZero() {
			fmt.Printf("  %s: %s\n", asset, pos.Quantity.StringFixed(8))
		}
	}

	fmt.Println("=========================")
}

func main() {
	// Create configuration
	config := trading.NewConfig()

	// Add venues
	config.WithNative("lx_dex", trading.NewLxDexConfig(
		getEnv("LX_DEX_URL", "https://api.dex.lux.network"),
	).WithCredentials(
		os.Getenv("LX_DEX_KEY"),
		os.Getenv("LX_DEX_SECRET"),
	))

	if key := os.Getenv("BINANCE_KEY"); key != "" {
		config.WithCcxt("binance", trading.NewCcxtConfig("binance").
			WithCredentials(key, os.Getenv("BINANCE_SECRET")))
	}

	// Risk config for the client
	config.Risk = trading.RiskConfig{
		Enabled:           true,
		MaxPositionSize:   decimal.NewFromFloat(100000),
		MaxOrderSize:      decimal.NewFromFloat(10000),
		MaxDailyLoss:      decimal.NewFromFloat(5000),
		MaxOpenOrders:     100,
		KillSwitchEnabled: true,
	}

	// Create client
	client := trading.NewClient(config)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Disconnect(context.Background())

	// Create risk monitor
	riskConfig := DefaultRiskConfig()
	riskConfig.MaxPositionAsset["BTC"] = decimal.NewFromFloat(1)
	riskConfig.MaxPositionAsset["ETH"] = decimal.NewFromFloat(10)

	monitor := NewRiskMonitor(client, riskConfig)
	monitor.Start()
	defer monitor.Stop()

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Status ticker
	statusTicker := time.NewTicker(10 * time.Second)
	defer statusTicker.Stop()

	log.Println("Risk monitor started. Press Ctrl+C to exit.")

	for {
		select {
		case <-sigCh:
			return
		case <-statusTicker.C:
			monitor.PrintStatus()
		}
	}
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
