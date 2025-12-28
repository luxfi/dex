package uniswap

import "errors"

var (
	// ErrNoMethodParameters indicates quote response missing method parameters
	ErrNoMethodParameters = errors.New("quote response missing method parameters")
	
	// ErrExecuteNotSupported indicates server-side swap execution is not supported
	ErrExecuteNotSupported = errors.New("server-side swap execution not supported")
	
	// ErrPoolNotFound indicates the requested pool was not found
	ErrPoolNotFound = errors.New("pool not found")
	
	// ErrPositionNotFound indicates the requested position was not found
	ErrPositionNotFound = errors.New("position not found")
	
	// ErrTokenNotFound indicates the requested token was not found
	ErrTokenNotFound = errors.New("token not found")
	
	// ErrNotImplemented indicates the feature is not implemented
	ErrNotImplemented = errors.New("feature not implemented in Uniswap provider")
)
