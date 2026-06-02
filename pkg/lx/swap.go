// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"fmt"
	"math/big"
)

var bigMaxSwapFee = big.NewInt(MaxSwapFee)

// Swap-level errors.
var (
	ErrMaxFeeExactOut = fmt.Errorf("cannot use max swap fee with exact output")
)

// GetSqrtPriceTarget returns the price target for the next swap step.
// Matches V4 SwapMath.getSqrtPriceTarget.
//
// zeroForOne == true:  max(sqrtPriceNextX96, sqrtPriceLimitX96)
// zeroForOne == false: min(sqrtPriceNextX96, sqrtPriceLimitX96)
func GetSqrtPriceTarget(zeroForOne bool, sqrtPriceNextX96, sqrtPriceLimitX96 *big.Int) *big.Int {
	if zeroForOne {
		if sqrtPriceNextX96.Cmp(sqrtPriceLimitX96) >= 0 {
			return new(big.Int).Set(sqrtPriceNextX96)
		}
		return new(big.Int).Set(sqrtPriceLimitX96)
	}
	if sqrtPriceNextX96.Cmp(sqrtPriceLimitX96) <= 0 {
		return new(big.Int).Set(sqrtPriceNextX96)
	}
	return new(big.Int).Set(sqrtPriceLimitX96)
}

// ComputeSwapStep computes the result of swapping within a single tick range.
// Matches V4 SwapMath.computeSwapStep exactly.
//
// amountRemaining is SIGNED per V4 convention:
//
//	negative = exact input  (user specifies how much to spend)
//	positive = exact output (user specifies how much to receive)
//
// Returns sqrtPriceNextX96, amountIn, amountOut, feeAmount, err.
// On error, all numeric returns are nil.
func ComputeSwapStep(
	sqrtPriceCurrentX96, sqrtPriceTargetX96, liquidity, amountRemaining *big.Int,
	feePips uint32,
) (sqrtPriceNextX96, amountIn, amountOut, feeAmount *big.Int, err error) {

	zeroForOne := sqrtPriceCurrentX96.Cmp(sqrtPriceTargetX96) >= 0
	exactIn := amountRemaining.Sign() < 0

	feePipsBig := big.NewInt(int64(feePips))
	feeComplement := new(big.Int).Sub(bigMaxSwapFee, feePipsBig) // MAX_SWAP_FEE - feePips

	// Guard: exact output with 100% fee is a division by zero.
	if !exactIn && feePips >= MaxSwapFee {
		return nil, nil, nil, nil, ErrMaxFeeExactOut
	}

	if exactIn {
		// |amountRemaining|
		absRemaining := new(big.Int).Neg(amountRemaining)

		// amountRemainingLessFee = MulDiv(|amountRemaining|, MAX_SWAP_FEE - feePips, MAX_SWAP_FEE)
		amountRemainingLessFee, mErr := MulDiv(absRemaining, feeComplement, bigMaxSwapFee)
		if mErr != nil {
			return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep fee calc: %w", mErr)
		}

		if zeroForOne {
			amountIn, err = GetAmount0Delta(sqrtPriceTargetX96, sqrtPriceCurrentX96, liquidity, true)
		} else {
			amountIn, err = GetAmount1Delta(sqrtPriceCurrentX96, sqrtPriceTargetX96, liquidity, true)
		}
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep amountIn delta: %w", err)
		}

		if amountRemainingLessFee.Cmp(amountIn) >= 0 {
			// We can reach the target price.
			sqrtPriceNextX96 = new(big.Int).Set(sqrtPriceTargetX96)
			if feePips == MaxSwapFee {
				feeAmount = new(big.Int).Set(amountIn)
			} else {
				feeAmount, err = MulDivRoundingUp(amountIn, feePipsBig, feeComplement)
				if err != nil {
					return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep fee rounding: %w", err)
				}
			}
		} else {
			// Exhaust the remaining amount before reaching target.
			amountIn = amountRemainingLessFee
			sqrtPriceNextX96, err = GetNextSqrtPriceFromInput(
				sqrtPriceCurrentX96, liquidity, amountRemainingLessFee, zeroForOne,
			)
			if err != nil {
				return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep next price from input: %w", err)
			}
			// Fee = remainder of the max input not used for the swap.
			feeAmount = new(big.Int).Sub(absRemaining, amountIn)
		}

		if zeroForOne {
			amountOut, err = GetAmount1Delta(sqrtPriceNextX96, sqrtPriceCurrentX96, liquidity, false)
		} else {
			amountOut, err = GetAmount0Delta(sqrtPriceCurrentX96, sqrtPriceNextX96, liquidity, false)
		}
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep amountOut delta: %w", err)
		}
	} else {
		// exactOut: amountRemaining >= 0

		if zeroForOne {
			amountOut, err = GetAmount1Delta(sqrtPriceTargetX96, sqrtPriceCurrentX96, liquidity, false)
		} else {
			amountOut, err = GetAmount0Delta(sqrtPriceCurrentX96, sqrtPriceTargetX96, liquidity, false)
		}
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep amountOut target delta: %w", err)
		}

		if amountRemaining.Cmp(amountOut) >= 0 {
			// We can reach the target price.
			sqrtPriceNextX96 = new(big.Int).Set(sqrtPriceTargetX96)
		} else {
			// Cap the output and compute the resulting price.
			amountOut = new(big.Int).Set(amountRemaining)
			sqrtPriceNextX96, err = GetNextSqrtPriceFromOutput(
				sqrtPriceCurrentX96, liquidity, amountOut, zeroForOne,
			)
			if err != nil {
				return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep next price from output: %w", err)
			}
		}

		if zeroForOne {
			amountIn, err = GetAmount0Delta(sqrtPriceNextX96, sqrtPriceCurrentX96, liquidity, true)
		} else {
			amountIn, err = GetAmount1Delta(sqrtPriceCurrentX96, sqrtPriceNextX96, liquidity, true)
		}
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep amountIn exact out delta: %w", err)
		}

		// feeComplement is guaranteed non-zero here (MaxSwapFee check above).
		feeAmount, err = MulDivRoundingUp(amountIn, feePipsBig, feeComplement)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("ComputeSwapStep exact out fee: %w", err)
		}
	}

	return
}

// EngineCL implements Engine using in-process Go math for
// Uniswap V4-style concentrated liquidity. This IS the engine --
// all V4 math (swap loop, tick crossing, fee growth, position
// tracking) lives here.
type EngineCL struct{}

// NewEngine creates a new concentrated liquidity engine.
func NewEngine() *EngineCL {
	return &EngineCL{}
}

// Verify interface compliance at compile time.
var _ Engine = (*EngineCL)(nil)

// Initialize computes the initial tick from sqrtPriceX96.
func (e *EngineCL) Initialize(sqrtPriceX96 *big.Int) (int24, error) {
	tick, err := GetTickAtSqrtPrice(sqrtPriceX96)
	if err != nil {
		return 0, err
	}
	return tick, nil
}

// Swap executes the V4 tick-crossing swap loop.
// Mutates pool state in place (sqrtPriceX96, tick, liquidity, feeGrowth).
func (e *EngineCL) Swap(pool *PoolState, params SwapParams) (BalanceDelta, error) {
	zeroForOne := params.ZeroForOne

	if params.AmountSpecified.Sign() == 0 {
		return ZeroBalanceDelta(), nil
	}

	// Validate price limit
	if zeroForOne {
		if params.SqrtPriceLimitX96.Cmp(pool.SqrtPriceX96) >= 0 {
			return ZeroBalanceDelta(), fmt.Errorf("%w: zeroForOne limit %s >= current %s",
				ErrPriceLimitReached, params.SqrtPriceLimitX96, pool.SqrtPriceX96)
		}
		if params.SqrtPriceLimitX96.Cmp(MinSqrtRatio) <= 0 {
			return ZeroBalanceDelta(), fmt.Errorf("%w: limit below MIN_SQRT_RATIO", ErrInvalidSqrtPrice)
		}
	} else {
		if params.SqrtPriceLimitX96.Cmp(pool.SqrtPriceX96) <= 0 {
			return ZeroBalanceDelta(), fmt.Errorf("%w: oneForZero limit %s <= current %s",
				ErrPriceLimitReached, params.SqrtPriceLimitX96, pool.SqrtPriceX96)
		}
		if params.SqrtPriceLimitX96.Cmp(MaxSqrtRatio) >= 0 {
			return ZeroBalanceDelta(), fmt.Errorf("%w: limit above MAX_SQRT_RATIO", ErrInvalidSqrtPrice)
		}
	}

	// 100% fee makes exact output impossible
	swapFee := pool.LPFee
	if swapFee >= MaxSwapFee && params.AmountSpecified.Sign() > 0 {
		return ZeroBalanceDelta(), fmt.Errorf("invalid fee for exact output")
	}

	amountSpecifiedRemaining := new(big.Int).Set(params.AmountSpecified)
	amountCalculated := big.NewInt(0)

	sqrtPriceX96 := new(big.Int).Set(pool.SqrtPriceX96)
	tick := pool.Tick
	liquidity := new(big.Int).Set(pool.Liquidity)

	var feeGrowthGlobalX128 *big.Int
	if zeroForOne {
		feeGrowthGlobalX128 = new(big.Int).Set(pool.FeeGrowth0X128)
	} else {
		feeGrowthGlobalX128 = new(big.Int).Set(pool.FeeGrowth1X128)
	}

	// MAIN SWAP LOOP
	for amountSpecifiedRemaining.Sign() != 0 && sqrtPriceX96.Cmp(params.SqrtPriceLimitX96) != 0 {
		sqrtPriceStartX96 := new(big.Int).Set(sqrtPriceX96)

		tickNext, initialized := pool.TickBitmap.NextInitializedTickWithinOneWord(
			tick, pool.TickSpacing, zeroForOne,
		)

		if tickNext < MinTick {
			tickNext = MinTick
		}
		if tickNext > MaxTick {
			tickNext = MaxTick
		}

		sqrtPriceNextX96, err := GetSqrtPriceAtTick(tickNext)
		if err != nil {
			return ZeroBalanceDelta(), fmt.Errorf("GetSqrtPriceAtTick(%d): %w", tickNext, err)
		}

		sqrtPriceTarget := GetSqrtPriceTarget(zeroForOne, sqrtPriceNextX96, params.SqrtPriceLimitX96)

		var stepAmountIn, stepAmountOut, stepFeeAmount *big.Int
		sqrtPriceX96, stepAmountIn, stepAmountOut, stepFeeAmount, err = ComputeSwapStep(
			sqrtPriceX96, sqrtPriceTarget, liquidity, amountSpecifiedRemaining, swapFee,
		)
		if err != nil {
			return ZeroBalanceDelta(), fmt.Errorf("ComputeSwapStep at tick %d: %w", tick, err)
		}

		if params.AmountSpecified.Sign() > 0 {
			// exactOutput
			amountSpecifiedRemaining.Sub(amountSpecifiedRemaining, stepAmountOut)
			consumed := new(big.Int).Add(stepAmountIn, stepFeeAmount)
			amountCalculated.Sub(amountCalculated, consumed)
		} else {
			// exactInput
			consumed := new(big.Int).Add(stepAmountIn, stepFeeAmount)
			amountSpecifiedRemaining.Add(amountSpecifiedRemaining, consumed)
			amountCalculated.Add(amountCalculated, stepAmountOut)
		}

		if liquidity.Sign() > 0 {
			feeGrowthDelta, fgErr := SimpleMulDiv(stepFeeAmount, Q128, liquidity)
			if fgErr != nil {
				return ZeroBalanceDelta(), fmt.Errorf("fee growth calc: %w", fgErr)
			}
			feeGrowthGlobalX128.Add(feeGrowthGlobalX128, feeGrowthDelta)
		}

		if sqrtPriceX96.Cmp(sqrtPriceNextX96) == 0 {
			if initialized {
				var fg0, fg1 *big.Int
				if zeroForOne {
					fg0 = feeGrowthGlobalX128
					fg1 = pool.FeeGrowth1X128
				} else {
					fg0 = pool.FeeGrowth0X128
					fg1 = feeGrowthGlobalX128
				}

				liquidityNet := crossTick(pool, tickNext, fg0, fg1)
				if zeroForOne {
					liquidityNet.Neg(liquidityNet)
				}

				var addErr error
				liquidity, addErr = AddDelta(liquidity, liquidityNet)
				if addErr != nil {
					return ZeroBalanceDelta(), fmt.Errorf("liquidity overflow crossing tick %d: %w", tickNext, addErr)
				}
			}

			if zeroForOne {
				tick = tickNext - 1
			} else {
				tick = tickNext
			}
		} else if sqrtPriceX96.Cmp(sqrtPriceStartX96) != 0 {
			var tErr error
			tick, tErr = GetTickAtSqrtPrice(sqrtPriceX96)
			if tErr != nil {
				return ZeroBalanceDelta(), fmt.Errorf("swap tick lookup: %w", tErr)
			}
		}
	}

	pool.SqrtPriceX96 = sqrtPriceX96
	pool.Tick = tick
	pool.Liquidity = liquidity
	if zeroForOne {
		pool.FeeGrowth0X128 = feeGrowthGlobalX128
	} else {
		pool.FeeGrowth1X128 = feeGrowthGlobalX128
	}

	amountUsed := new(big.Int).Sub(params.AmountSpecified, amountSpecifiedRemaining)

	if zeroForOne != (params.AmountSpecified.Sign() < 0) {
		return NewBalanceDelta(amountCalculated, amountUsed), nil
	}
	return NewBalanceDelta(amountUsed, amountCalculated), nil
}

// ModifyLiquidity implements V4 concentrated liquidity add/remove.
func (e *EngineCL) ModifyLiquidity(
	pool *PoolState, owner Address, params ModifyLiquidityParams,
) (BalanceDelta, BalanceDelta, error) {
	liquidityDelta := params.LiquidityDelta
	tickLower := params.TickLower
	tickUpper := params.TickUpper

	if tickLower >= tickUpper {
		return ZeroBalanceDelta(), ZeroBalanceDelta(), ErrInvalidTickRange
	}
	if tickLower < MinTick || tickUpper > MaxTick {
		return ZeroBalanceDelta(), ZeroBalanceDelta(), ErrTickOutOfRange
	}
	if tickLower%pool.TickSpacing != 0 || tickUpper%pool.TickSpacing != 0 {
		return ZeroBalanceDelta(), ZeroBalanceDelta(), ErrTickMisaligned
	}

	var flippedLower, flippedUpper bool
	var grossAfterLower, grossAfterUpper *big.Int

	if liquidityDelta.Sign() != 0 {
		var uErr error
		flippedLower, grossAfterLower, uErr = updateTick(pool, tickLower, liquidityDelta, false)
		if uErr != nil {
			return ZeroBalanceDelta(), ZeroBalanceDelta(), uErr
		}
		flippedUpper, grossAfterUpper, uErr = updateTick(pool, tickUpper, liquidityDelta, true)
		if uErr != nil {
			return ZeroBalanceDelta(), ZeroBalanceDelta(), uErr
		}

		if liquidityDelta.Sign() >= 0 {
			maxLiqPerTick := tickSpacingToMaxLiquidityPerTick(pool.TickSpacing)
			if grossAfterLower.Cmp(maxLiqPerTick) > 0 {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("%w: tick %d liquidity overflow", ErrInsufficientLiquidity, tickLower)
			}
			if grossAfterUpper.Cmp(maxLiqPerTick) > 0 {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("%w: tick %d liquidity overflow", ErrInsufficientLiquidity, tickUpper)
			}
		}

		if flippedLower {
			if fErr := pool.TickBitmap.FlipTick(tickLower, pool.TickSpacing); fErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("flip lower tick: %w", fErr)
			}
		}
		if flippedUpper {
			if fErr := pool.TickBitmap.FlipTick(tickUpper, pool.TickSpacing); fErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("flip upper tick: %w", fErr)
			}
		}
	}

	feeGrowthInside0, feeGrowthInside1 := getFeeGrowthInside(pool, tickLower, tickUpper)
	feeDelta, fErr := updatePosition(pool, owner, params, feeGrowthInside0, feeGrowthInside1)
	if fErr != nil {
		return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("update position: %w", fErr)
	}

	if liquidityDelta.Sign() < 0 {
		if flippedLower {
			clearTick(pool, tickLower)
		}
		if flippedUpper {
			clearTick(pool, tickUpper)
		}
	}

	var delta BalanceDelta
	if liquidityDelta.Sign() != 0 {
		currentTick := pool.Tick

		if currentTick < tickLower {
			sqrtLower, sErr := GetSqrtPriceAtTick(tickLower)
			if sErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("sqrt price lower: %w", sErr)
			}
			sqrtUpper, sErr := GetSqrtPriceAtTick(tickUpper)
			if sErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("sqrt price upper: %w", sErr)
			}
			amount0, err := GetAmount0DeltaSigned(sqrtLower, sqrtUpper, liquidityDelta)
			if err != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("amount0 delta (below range): %w", err)
			}
			delta = NewBalanceDelta(amount0, big.NewInt(0))
		} else if currentTick < tickUpper {
			sqrtUpper, sErr := GetSqrtPriceAtTick(tickUpper)
			if sErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("sqrt price upper: %w", sErr)
			}
			sqrtLower, sErr := GetSqrtPriceAtTick(tickLower)
			if sErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("sqrt price lower: %w", sErr)
			}
			amount0, err := GetAmount0DeltaSigned(pool.SqrtPriceX96, sqrtUpper, liquidityDelta)
			if err != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("amount0 delta (in range): %w", err)
			}
			amount1, err := GetAmount1DeltaSigned(sqrtLower, pool.SqrtPriceX96, liquidityDelta)
			if err != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("amount1 delta (in range): %w", err)
			}
			delta = NewBalanceDelta(amount0, amount1)

			newLiq, lErr := AddDelta(pool.Liquidity, liquidityDelta)
			if lErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("%w: %v", ErrInsufficientLiquidity, lErr)
			}
			pool.Liquidity = newLiq
		} else {
			sqrtLower, sErr := GetSqrtPriceAtTick(tickLower)
			if sErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("sqrt price lower: %w", sErr)
			}
			sqrtUpper, sErr := GetSqrtPriceAtTick(tickUpper)
			if sErr != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(), fmt.Errorf("sqrt price upper: %w", sErr)
			}
			amount1, err := GetAmount1DeltaSigned(sqrtLower, sqrtUpper, liquidityDelta)
			if err != nil {
				return ZeroBalanceDelta(), ZeroBalanceDelta(),
					fmt.Errorf("amount1 delta (above range): %w", err)
			}
			delta = NewBalanceDelta(big.NewInt(0), amount1)
		}
	}

	return delta, feeDelta, nil
}

// Donate distributes tokens to LPs via fee growth updates.
func (e *EngineCL) Donate(pool *PoolState, amount0, amount1 *big.Int) (BalanceDelta, error) {
	if pool.Liquidity == nil || pool.Liquidity.Sign() <= 0 {
		return ZeroBalanceDelta(), ErrNoLiquidity
	}

	if amount0 != nil && amount0.Sign() > 0 {
		fg, err := SimpleMulDiv(amount0, Q128, pool.Liquidity)
		if err != nil {
			return ZeroBalanceDelta(), fmt.Errorf("donate fee growth0: %w", err)
		}
		pool.FeeGrowth0X128 = new(big.Int).Add(pool.FeeGrowth0X128, fg)
	}
	if amount1 != nil && amount1.Sign() > 0 {
		fg, err := SimpleMulDiv(amount1, Q128, pool.Liquidity)
		if err != nil {
			return ZeroBalanceDelta(), fmt.Errorf("donate fee growth1: %w", err)
		}
		pool.FeeGrowth1X128 = new(big.Int).Add(pool.FeeGrowth1X128, fg)
	}

	return NewBalanceDelta(amount0, amount1), nil
}

// Quote estimates swap output without mutating state.
func (e *EngineCL) Quote(pool *Pool, amountIn *big.Int, zeroForOne bool) *big.Int {
	if pool.Liquidity.Sign() == 0 || amountIn.Sign() <= 0 {
		return big.NewInt(0)
	}
	// Use ComputeSwapStep for a single step (no tick crossing).
	// V4 exactInput convention: amountRemaining is negative.
	negAmountIn := new(big.Int).Neg(amountIn)

	// Use the boundary sqrt price as target (MIN for zeroForOne, MAX for oneForZero).
	var target *big.Int
	if zeroForOne {
		target = new(big.Int).Add(MinSqrtRatio, big.NewInt(1))
	} else {
		target = new(big.Int).Sub(MaxSqrtRatio, big.NewInt(1))
	}

	_, _, amountOut, _, err := ComputeSwapStep(
		pool.SqrtPriceX96, target, pool.Liquidity, negAmountIn, 0, // fee=0 for estimation
	)
	if err != nil {
		return big.NewInt(0)
	}
	return amountOut
}

// =========================================================================
// V4 Tick Helpers (engine-internal)
// =========================================================================

func updateTick(
	pool *PoolState, tick int32, liquidityDelta *big.Int, upper bool,
) (flipped bool, liquidityGrossAfter *big.Int, err error) {
	info := pool.getOrCreateTick(tick)
	liquidityGrossBefore := new(big.Int).Set(info.LiquidityGross)

	liquidityGrossAfter, err = AddDelta(liquidityGrossBefore, liquidityDelta)
	if err != nil {
		return false, nil, fmt.Errorf("tick %d liquidity: %w", tick, err)
	}

	flipped = (liquidityGrossAfter.Sign() == 0) != (liquidityGrossBefore.Sign() == 0)

	if liquidityGrossBefore.Sign() == 0 {
		if tick <= pool.Tick {
			info.FeeGrowthOutside0X128 = new(big.Int).Set(pool.FeeGrowth0X128)
			info.FeeGrowthOutside1X128 = new(big.Int).Set(pool.FeeGrowth1X128)
		}
	}

	info.LiquidityGross = liquidityGrossAfter
	if upper {
		info.LiquidityNet = new(big.Int).Sub(info.LiquidityNet, liquidityDelta)
	} else {
		info.LiquidityNet = new(big.Int).Add(info.LiquidityNet, liquidityDelta)
	}
	return
}

func crossTick(
	pool *PoolState, tick int32, feeGrowthGlobal0X128, feeGrowthGlobal1X128 *big.Int,
) *big.Int {
	info := pool.getOrCreateTick(tick)
	info.FeeGrowthOutside0X128 = new(big.Int).Sub(feeGrowthGlobal0X128, info.FeeGrowthOutside0X128)
	info.FeeGrowthOutside1X128 = new(big.Int).Sub(feeGrowthGlobal1X128, info.FeeGrowthOutside1X128)
	return new(big.Int).Set(info.LiquidityNet)
}

func clearTick(pool *PoolState, tick int32) {
	delete(pool.Ticks, tick)
}

// wrapUint256 replicates Solidity uint256 wrapping: result mod 2^256.
var uint256Modulus = new(big.Int).Lsh(big.NewInt(1), 256) // 2^256

func wrapUint256(v *big.Int) *big.Int {
	v.Mod(v, uint256Modulus)
	return v
}

func getFeeGrowthInside(
	pool *PoolState, tickLower, tickUpper int32,
) (feeGrowthInside0X128, feeGrowthInside1X128 *big.Int) {
	lower := pool.getOrCreateTick(tickLower)
	upper := pool.getOrCreateTick(tickUpper)
	tickCurrent := pool.Tick

	if tickCurrent < tickLower {
		feeGrowthInside0X128 = wrapUint256(new(big.Int).Sub(lower.FeeGrowthOutside0X128, upper.FeeGrowthOutside0X128))
		feeGrowthInside1X128 = wrapUint256(new(big.Int).Sub(lower.FeeGrowthOutside1X128, upper.FeeGrowthOutside1X128))
	} else if tickCurrent >= tickUpper {
		feeGrowthInside0X128 = wrapUint256(new(big.Int).Sub(upper.FeeGrowthOutside0X128, lower.FeeGrowthOutside0X128))
		feeGrowthInside1X128 = wrapUint256(new(big.Int).Sub(upper.FeeGrowthOutside1X128, lower.FeeGrowthOutside1X128))
	} else {
		feeGrowthInside0X128 = wrapUint256(new(big.Int).Sub(
			wrapUint256(new(big.Int).Sub(pool.FeeGrowth0X128, lower.FeeGrowthOutside0X128)),
			upper.FeeGrowthOutside0X128,
		))
		feeGrowthInside1X128 = wrapUint256(new(big.Int).Sub(
			wrapUint256(new(big.Int).Sub(pool.FeeGrowth1X128, lower.FeeGrowthOutside1X128)),
			upper.FeeGrowthOutside1X128,
		))
	}
	return
}

func updatePosition(
	pool *PoolState, owner Address, params ModifyLiquidityParams,
	feeGrowthInside0X128, feeGrowthInside1X128 *big.Int,
) (BalanceDelta, error) {
	posKey := LPPositionKey(owner, params.TickLower, params.TickUpper, params.Salt)

	pos, ok := pool.Positions[posKey]
	if !ok {
		pos = &LPPosition{
			Owner: owner, TickLower: params.TickLower, TickUpper: params.TickUpper,
			Liquidity:                big.NewInt(0),
			FeeGrowthInside0LastX128: big.NewInt(0), FeeGrowthInside1LastX128: big.NewInt(0),
			TokensOwed0: big.NewInt(0), TokensOwed1: big.NewInt(0),
		}
		pool.Positions[posKey] = pos
	}

	var feesOwed0, feesOwed1 *big.Int
	if pos.Liquidity.Sign() > 0 {
		var err error
		feesOwed0, err = SimpleMulDiv(wrapUint256(new(big.Int).Sub(feeGrowthInside0X128, pos.FeeGrowthInside0LastX128)), pos.Liquidity, Q128)
		if err != nil {
			return ZeroBalanceDelta(), fmt.Errorf("fee0 calc: %w", err)
		}
		feesOwed1, err = SimpleMulDiv(wrapUint256(new(big.Int).Sub(feeGrowthInside1X128, pos.FeeGrowthInside1LastX128)), pos.Liquidity, Q128)
		if err != nil {
			return ZeroBalanceDelta(), fmt.Errorf("fee1 calc: %w", err)
		}
	} else {
		feesOwed0 = big.NewInt(0)
		feesOwed1 = big.NewInt(0)
	}

	pos.FeeGrowthInside0LastX128 = new(big.Int).Set(feeGrowthInside0X128)
	pos.FeeGrowthInside1LastX128 = new(big.Int).Set(feeGrowthInside1X128)

	if params.LiquidityDelta.Sign() != 0 {
		newLiq := new(big.Int).Add(pos.Liquidity, params.LiquidityDelta)
		if newLiq.Sign() < 0 {
			newLiq = big.NewInt(0)
		}
		pos.Liquidity = newLiq
	}

	pos.Owner = owner
	pos.TickLower = params.TickLower
	pos.TickUpper = params.TickUpper

	return NewBalanceDelta(feesOwed0, feesOwed1), nil
}

func tickSpacingToMaxLiquidityPerTick(tickSpacing int32) *big.Int {
	minTick := MinTick / tickSpacing
	if MinTick%tickSpacing != 0 {
		minTick--
	}
	maxTick := MaxTick / tickSpacing
	numTicks := maxTick - minTick + 1
	return new(big.Int).Div(uint128Max, big.NewInt(int64(numTicks)))
}
