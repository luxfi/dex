// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"errors"
	"math/big"
)

// MaxUint256 is 2^256 - 1, the maximum value of a Solidity uint256.
var MaxUint256 = new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))

// Math errors matching Solidity revert semantics.
var (
	ErrDivisionByZero = errors.New("division by zero")
	ErrUint256Overflow = errors.New("uint256 overflow")
)

// MulDiv returns floor(a * b / denominator) with full precision.
// Returns ErrDivisionByZero if denominator is zero.
// Returns ErrUint256Overflow if the result exceeds 2^256-1.
func MulDiv(a, b, denominator *big.Int) (*big.Int, error) {
	if denominator.Sign() == 0 {
		return nil, ErrDivisionByZero
	}
	product := new(big.Int).Mul(a, b)
	result := new(big.Int).Div(product, denominator)
	if result.BitLen() > 256 {
		return nil, ErrUint256Overflow
	}
	return result, nil
}

// MulDivRoundingUp returns ceil(a * b / denominator) with full precision.
// Returns ErrDivisionByZero if denominator is zero.
// Returns ErrUint256Overflow if the result exceeds 2^256-1.
func MulDivRoundingUp(a, b, denominator *big.Int) (*big.Int, error) {
	if denominator.Sign() == 0 {
		return nil, ErrDivisionByZero
	}
	product := new(big.Int).Mul(a, b)
	result := new(big.Int).Div(product, denominator)
	remainder := new(big.Int).Mod(product, denominator)
	if remainder.Sign() != 0 {
		result.Add(result, big.NewInt(1))
	}
	if result.BitLen() > 256 {
		return nil, ErrUint256Overflow
	}
	return result, nil
}

// DivRoundingUp returns ceil(a / b) for non-negative values.
// Returns ErrDivisionByZero if b is zero.
// Returns ErrUint256Overflow if the result exceeds 2^256-1.
func DivRoundingUp(a, b *big.Int) (*big.Int, error) {
	if b.Sign() == 0 {
		return nil, ErrDivisionByZero
	}
	result := new(big.Int).Div(a, b)
	remainder := new(big.Int).Mod(a, b)
	if remainder.Sign() != 0 {
		result.Add(result, big.NewInt(1))
	}
	if result.BitLen() > 256 {
		return nil, ErrUint256Overflow
	}
	return result, nil
}

// SimpleMulDiv returns a * b / denominator without overflow protection.
// Functionally identical to MulDiv when using math/big (arbitrary precision).
// Returns ErrDivisionByZero if denominator is zero.
// Returns ErrUint256Overflow if the result exceeds 2^256-1.
func SimpleMulDiv(a, b, denominator *big.Int) (*big.Int, error) {
	if denominator.Sign() == 0 {
		return nil, ErrDivisionByZero
	}
	result := new(big.Int).Div(new(big.Int).Mul(a, b), denominator)
	if result.BitLen() > 256 {
		return nil, ErrUint256Overflow
	}
	return result, nil
}
