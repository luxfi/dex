// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"errors"
	"math/big"
)

// SqrtPriceMath errors matching Uniswap V4 SqrtPriceMath.sol
var (
	ErrInvalidPriceOrLiquidity = errors.New("invalid price or liquidity")
	ErrZeroSqrtPrice           = errors.New("invalid sqrt price (zero)")
	ErrInvalidPriceSqrt        = errors.New("invalid price")
	ErrNotEnoughLiquidity      = errors.New("not enough liquidity")
	ErrPriceOverflow           = errors.New("price overflow")
)

// MaxUint160 is 2^160 - 1, the maximum value of a Solidity uint160.
var MaxUint160 = new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 160), big.NewInt(1))

// checkUint160 validates that v fits in uint160. Returns ErrPriceOverflow if not.
// V4 Solidity reverts on uint160 overflow; silent clamping would allow fund extraction.
func checkUint160(v *big.Int) error {
	if v.BitLen() > 160 {
		return ErrPriceOverflow
	}
	return nil
}

// GetNextSqrtPriceFromAmount0RoundingUp computes the next sqrt price given a
// delta of currency0. Always rounds up per V4 SqrtPriceMath.sol.
//
// Formula: liquidity * sqrtPX96 / (liquidity +/- amount * sqrtPX96)
// With math/big we never overflow, so the primary formula always works.
func GetNextSqrtPriceFromAmount0RoundingUp(sqrtPX96, liquidity, amount *big.Int, add bool) (*big.Int, error) {
	if sqrtPX96.Sign() <= 0 {
		return nil, ErrZeroSqrtPrice
	}
	if amount.Sign() == 0 {
		return new(big.Int).Set(sqrtPX96), nil
	}

	// numerator1 = liquidity << 96
	numerator1 := new(big.Int).Lsh(liquidity, 96)

	if add {
		// product = amount * sqrtPX96
		product := new(big.Int).Mul(amount, sqrtPX96)
		// denominator = numerator1 + product
		denominator := new(big.Int).Add(numerator1, product)
		// result = ceil(numerator1 * sqrtPX96 / denominator)
		result, err := MulDivRoundingUp(numerator1, sqrtPX96, denominator)
		if err != nil {
			return nil, err
		}
		if err := checkUint160(result); err != nil {
			return nil, err
		}
		return result, nil
	}

	// !add path
	product := new(big.Int).Mul(amount, sqrtPX96)
	// Require numerator1 > product (otherwise price overflows)
	if numerator1.Cmp(product) <= 0 {
		return nil, ErrPriceOverflow
	}
	denominator := new(big.Int).Sub(numerator1, product)
	result, err := MulDivRoundingUp(numerator1, sqrtPX96, denominator)
	if err != nil {
		return nil, err
	}
	if err := checkUint160(result); err != nil {
		return nil, err
	}
	return result, nil
}

// GetNextSqrtPriceFromAmount1RoundingDown computes the next sqrt price given a
// delta of currency1. Always rounds down per V4 SqrtPriceMath.sol.
func GetNextSqrtPriceFromAmount1RoundingDown(sqrtPX96, liquidity, amount *big.Int, add bool) (*big.Int, error) {
	if sqrtPX96.Sign() <= 0 {
		return nil, ErrZeroSqrtPrice
	}
	if add {
		var quotient *big.Int
		if amount.Cmp(MaxUint160) <= 0 {
			// (amount << 96) / liquidity -- floor division
			shifted := new(big.Int).Lsh(amount, 96)
			quotient = new(big.Int).Div(shifted, liquidity)
		} else {
			var err error
			quotient, err = MulDiv(amount, Q96, liquidity)
			if err != nil {
				return nil, err
			}
		}
		result := new(big.Int).Add(sqrtPX96, quotient)
		if err := checkUint160(result); err != nil {
			return nil, err
		}
		return result, nil
	}

	// !add path
	var quotient *big.Int
	if amount.Cmp(MaxUint160) <= 0 {
		shifted := new(big.Int).Lsh(amount, 96)
		var err error
		quotient, err = DivRoundingUp(shifted, liquidity)
		if err != nil {
			return nil, err
		}
	} else {
		var err error
		quotient, err = MulDivRoundingUp(amount, Q96, liquidity)
		if err != nil {
			return nil, err
		}
	}
	if sqrtPX96.Cmp(quotient) <= 0 {
		return nil, ErrNotEnoughLiquidity
	}
	return new(big.Int).Sub(sqrtPX96, quotient), nil
}

// GetNextSqrtPriceFromInput returns the next sqrt price given an input amount.
// zeroForOne == true means currency0 is being swapped in.
func GetNextSqrtPriceFromInput(sqrtPX96, liquidity, amountIn *big.Int, zeroForOne bool) (*big.Int, error) {
	if sqrtPX96.Sign() <= 0 || liquidity.Sign() <= 0 {
		return nil, ErrInvalidPriceOrLiquidity
	}
	if zeroForOne {
		return GetNextSqrtPriceFromAmount0RoundingUp(sqrtPX96, liquidity, amountIn, true)
	}
	return GetNextSqrtPriceFromAmount1RoundingDown(sqrtPX96, liquidity, amountIn, true)
}

// GetNextSqrtPriceFromOutput returns the next sqrt price given an output amount.
// zeroForOne == true means currency1 is being output.
func GetNextSqrtPriceFromOutput(sqrtPX96, liquidity, amountOut *big.Int, zeroForOne bool) (*big.Int, error) {
	if sqrtPX96.Sign() <= 0 || liquidity.Sign() <= 0 {
		return nil, ErrInvalidPriceOrLiquidity
	}
	if zeroForOne {
		return GetNextSqrtPriceFromAmount1RoundingDown(sqrtPX96, liquidity, amountOut, false)
	}
	return GetNextSqrtPriceFromAmount0RoundingUp(sqrtPX96, liquidity, amountOut, false)
}

// GetAmount0Delta computes the amount of currency0 between two sqrt prices.
// Matches V4 SqrtPriceMath.getAmount0Delta(uint160,uint160,uint128,bool).
// Returns ErrZeroSqrtPrice if the lower sqrt price is zero.
func GetAmount0Delta(sqrtPriceAX96, sqrtPriceBX96, liquidity *big.Int, roundUp bool) (*big.Int, error) {
	a, b := sqrtPriceAX96, sqrtPriceBX96
	if a.Cmp(b) > 0 {
		a, b = b, a
	}
	if a.Sign() <= 0 {
		return nil, ErrZeroSqrtPrice
	}

	numerator1 := new(big.Int).Lsh(liquidity, 96)
	numerator2 := new(big.Int).Sub(b, a)

	if roundUp {
		inner, err := MulDivRoundingUp(numerator1, numerator2, b)
		if err != nil {
			return nil, err
		}
		result, err := DivRoundingUp(inner, a)
		if err != nil {
			return nil, err
		}
		return result, nil
	}
	inner, err := MulDiv(numerator1, numerator2, b)
	if err != nil {
		return nil, err
	}
	return new(big.Int).Div(inner, a), nil
}

// GetAmount1Delta computes the amount of currency1 between two sqrt prices.
// Matches V4 SqrtPriceMath.getAmount1Delta(uint160,uint160,uint128,bool).
func GetAmount1Delta(sqrtPriceAX96, sqrtPriceBX96, liquidity *big.Int, roundUp bool) (*big.Int, error) {
	numerator := absDiff(sqrtPriceAX96, sqrtPriceBX96)
	if roundUp {
		return MulDivRoundingUp(liquidity, numerator, Q96)
	}
	return MulDiv(liquidity, numerator, Q96)
}

// absDiff returns |a - b| for non-negative big.Ints.
func absDiff(a, b *big.Int) *big.Int {
	if a.Cmp(b) >= 0 {
		return new(big.Int).Sub(a, b)
	}
	return new(big.Int).Sub(b, a)
}

// GetAmount0DeltaSigned returns the signed currency0 delta for a liquidity change.
// Matches V4 SqrtPriceMath.getAmount0Delta(uint160,uint160,int128).
//
// liquidity > 0 (adding): return -(rounded up amount)  -- user pays
// liquidity < 0 (removing): return +(rounded down amount) -- user receives
func GetAmount0DeltaSigned(sqrtPriceAX96, sqrtPriceBX96, liquidity *big.Int) (*big.Int, error) {
	if liquidity.Sign() < 0 {
		absLiq := new(big.Int).Neg(liquidity)
		return GetAmount0Delta(sqrtPriceAX96, sqrtPriceBX96, absLiq, false)
	}
	result, err := GetAmount0Delta(sqrtPriceAX96, sqrtPriceBX96, liquidity, true)
	if err != nil {
		return nil, err
	}
	return new(big.Int).Neg(result), nil
}

// GetAmount1DeltaSigned returns the signed currency1 delta for a liquidity change.
// Matches V4 SqrtPriceMath.getAmount1Delta(uint160,uint160,int128).
//
// liquidity > 0 (adding): return -(rounded up amount)  -- user pays
// liquidity < 0 (removing): return +(rounded down amount) -- user receives
func GetAmount1DeltaSigned(sqrtPriceAX96, sqrtPriceBX96, liquidity *big.Int) (*big.Int, error) {
	if liquidity.Sign() < 0 {
		absLiq := new(big.Int).Neg(liquidity)
		return GetAmount1Delta(sqrtPriceAX96, sqrtPriceBX96, absLiq, false)
	}
	result, err := GetAmount1Delta(sqrtPriceAX96, sqrtPriceBX96, liquidity, true)
	if err != nil {
		return nil, err
	}
	return new(big.Int).Neg(result), nil
}
