package price

import "errors"

var (
	// ErrNotFound indicates a price is not available.
	ErrNotFound = errors.New("price not found")

	// ErrStale indicates a price is too old.
	ErrStale = errors.New("price is stale")

	// ErrInsufficientSources indicates not enough price sources.
	ErrInsufficientSources = errors.New("insufficient price sources")

	// ErrCircuitBreaker indicates circuit breaker tripped.
	ErrCircuitBreaker = errors.New("circuit breaker tripped")
)
