// Package trading provides a unified HFT trading SDK with multi-venue support.
package trading

import (
	"errors"
	"fmt"
)

// Standard errors.
var (
	ErrNotConnected    = errors.New("not connected to venue")
	ErrVenueNotFound   = errors.New("venue not found")
	ErrOrderNotFound   = errors.New("order not found")
	ErrInvalidSymbol   = errors.New("invalid trading symbol")
	ErrInvalidQuantity = errors.New("invalid quantity")
	ErrInvalidPrice    = errors.New("invalid price")
	ErrInsufficientBalance = errors.New("insufficient balance")
	ErrRateLimited     = errors.New("rate limited")
	ErrTimeout         = errors.New("request timeout")
	ErrCancelled       = errors.New("operation cancelled")
)

// TradingError represents a trading-specific error.
type TradingError struct {
	Code      string
	Message   string
	Venue     string
	Retryable bool
	Cause     error
}

func (e *TradingError) Error() string {
	if e.Venue != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Venue, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *TradingError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns true if the error is retryable.
func (e *TradingError) IsRetryable() bool {
	return e.Retryable
}

// NewTradingError creates a new trading error.
func NewTradingError(code, message string) *TradingError {
	return &TradingError{
		Code:    code,
		Message: message,
	}
}

// WithVenue adds venue information to the error.
func (e *TradingError) WithVenue(venue string) *TradingError {
	e.Venue = venue
	return e
}

// WithRetryable marks the error as retryable.
func (e *TradingError) WithRetryable() *TradingError {
	e.Retryable = true
	return e
}

// WithCause adds a cause to the error.
func (e *TradingError) WithCause(cause error) *TradingError {
	e.Cause = cause
	return e
}

// RiskError represents a risk limit exceeded error.
type RiskError struct {
	Limit   string
	Current string
	Max     string
}

func (e *RiskError) Error() string {
	return fmt.Sprintf("risk limit exceeded: %s (current: %s, max: %s)", e.Limit, e.Current, e.Max)
}

// NewRiskError creates a new risk error.
func NewRiskError(limit, current, max string) *RiskError {
	return &RiskError{
		Limit:   limit,
		Current: current,
		Max:     max,
	}
}
