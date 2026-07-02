// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"errors"
	"math/big"
)

// uint128Max is 2^128 - 1.
var uint128Max = new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 128), big.NewInt(1))

// ErrLiquidityUnderflow is returned when removing more liquidity than available.
var ErrLiquidityUnderflow = errors.New("liquidity underflow")

// ErrLiquidityOverflow is returned when the result exceeds uint128.
var ErrLiquidityOverflow = errors.New("liquidity overflow")

// AddDelta adds a signed liquidity delta to unsigned liquidity.
// x is uint128 (current liquidity, non-negative).
// y is int128 (delta, can be negative for removal).
//
// Matches Uniswap V4 LiquidityMath.addDelta:
//   - if y < 0: result = x - |y|, must not underflow (result < 0)
//   - if y >= 0: result = x + y, must not overflow uint128
func AddDelta(x, y *big.Int) (*big.Int, error) {
	result := new(big.Int).Add(x, y)
	if y.Sign() < 0 {
		// Removal: result must be non-negative.
		if result.Sign() < 0 {
			return nil, ErrLiquidityUnderflow
		}
	} else {
		// Addition: result must fit in uint128.
		if result.Cmp(uint128Max) > 0 {
			return nil, ErrLiquidityOverflow
		}
	}
	return result, nil
}
