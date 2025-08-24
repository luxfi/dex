package lx

import (
	"math/big"
	"time"
)

// PerpPosition represents a perpetual position
type PerpPosition struct {
	Symbol           string
	User             string
	Size             float64
	EntryPrice       float64
	MarkPrice        float64
	UnrealizedPnL    float64
	RealizedPnL      float64
	LiquidationPrice float64
	Margin           *big.Int
	FundingPaid      float64
	OpenTime         time.Time
	UpdateTime       time.Time
}

// PerpetualManager manages perpetual contracts
type PerpetualManager struct {
	engine *TradingEngine
}

// NewPerpetualManager creates a new perpetual manager
func NewPerpetualManager(engine *TradingEngine) *PerpetualManager {
	return &PerpetualManager{
		engine: engine,
	}
}

// VaultManager and LendingPool are defined in their respective files
