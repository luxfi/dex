// Copyright 2024 Lux Partners Limited. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trading

import (
	"errors"
	"fmt"
)

// Standard sentinel errors for the trading SDK.
// Use errors.Is to check for these errors in error chains.
var (
	// ErrNotConnected indicates the venue adapter is not connected.
	ErrNotConnected = errors.New("trading: not connected to venue")

	// ErrVenueNotFound indicates the requested venue does not exist.
	ErrVenueNotFound = errors.New("trading: venue not found")

	// ErrOrderNotFound indicates the order does not exist.
	ErrOrderNotFound = errors.New("trading: order not found")

	// ErrInvalidSymbol indicates an invalid trading symbol format.
	ErrInvalidSymbol = errors.New("trading: invalid trading symbol")

	// ErrInvalidQuantity indicates an invalid order quantity.
	ErrInvalidQuantity = errors.New("trading: invalid quantity")

	// ErrInvalidPrice indicates an invalid order price.
	ErrInvalidPrice = errors.New("trading: invalid price")

	// ErrInsufficientBalance indicates insufficient balance for the operation.
	ErrInsufficientBalance = errors.New("trading: insufficient balance")

	// ErrRateLimited indicates the request was rate limited by the venue.
	ErrRateLimited = errors.New("trading: rate limited")

	// ErrTimeout indicates the operation timed out.
	ErrTimeout = errors.New("trading: request timeout")

	// ErrCancelled indicates the operation was cancelled.
	ErrCancelled = errors.New("trading: operation cancelled")
)

// TradingError represents a trading-specific error with rich context.
// It implements the error interface and supports error wrapping via Unwrap.
//
// Example usage:
//
//	err := NewTradingError("INVALID_ORDER", "order quantity too small").
//		WithVenue("binance").
//		WithCause(originalErr).
//		WithRetryable()
//
//	if err.IsRetryable() {
//		// retry logic
//	}
type TradingError struct {
	// Code is a machine-readable error code (e.g., "RATE_LIMITED", "INVALID_ORDER").
	Code string

	// Message is a human-readable error description.
	Message string

	// Venue is the trading venue where the error occurred.
	Venue string

	// Retryable indicates if the operation can be retried.
	Retryable bool

	// Cause is the underlying error that caused this error.
	Cause error
}

// Error implements the error interface.
func (e *TradingError) Error() string {
	if e.Venue != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Venue, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying cause for use with errors.Is and errors.As.
func (e *TradingError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns true if the error indicates a transient failure
// that may succeed on retry.
func (e *TradingError) IsRetryable() bool {
	return e.Retryable
}

// NewTradingError creates a new TradingError with the given code and message.
func NewTradingError(code, message string) *TradingError {
	return &TradingError{
		Code:    code,
		Message: message,
	}
}

// WithVenue returns a copy of the error with venue information added.
func (e *TradingError) WithVenue(venue string) *TradingError {
	e.Venue = venue
	return e
}

// WithRetryable returns a copy of the error marked as retryable.
func (e *TradingError) WithRetryable() *TradingError {
	e.Retryable = true
	return e
}

// WithCause returns a copy of the error with the underlying cause set.
// This enables error chain inspection via errors.Is and errors.As.
func (e *TradingError) WithCause(cause error) *TradingError {
	e.Cause = cause
	return e
}

// Wrap creates a new TradingError that wraps the given error.
// This is a convenience function for wrapping errors with context.
func Wrap(err error, code, message string) *TradingError {
	return &TradingError{
		Code:    code,
		Message: message,
		Cause:   err,
	}
}

// RiskError represents a risk limit exceeded error.
// It provides details about which limit was exceeded and the current/max values.
type RiskError struct {
	// Limit is the name of the exceeded limit (e.g., "position_size", "daily_loss").
	Limit string

	// Current is the current value that triggered the limit.
	Current string

	// Max is the maximum allowed value.
	Max string
}

// Error implements the error interface.
func (e *RiskError) Error() string {
	return fmt.Sprintf("trading: risk limit exceeded: %s (current: %s, max: %s)", e.Limit, e.Current, e.Max)
}

// NewRiskError creates a new RiskError for the given limit.
func NewRiskError(limit, current, max string) *RiskError {
	return &RiskError{
		Limit:   limit,
		Current: current,
		Max:     max,
	}
}

// IsRiskError returns true if err is or wraps a RiskError.
func IsRiskError(err error) bool {
	var riskErr *RiskError
	return errors.As(err, &riskErr)
}
