// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/big"
	"testing"
)

func TestSwapMath_GetSqrtPriceTarget(t *testing.T) {
	low := bi(100)
	high := bi(200)

	t.Run("zeroForOne returns max", func(t *testing.T) {
		got := GetSqrtPriceTarget(true, low, high)
		if got.Cmp(high) != 0 {
			t.Fatalf("expected %s (max), got %s", high, got)
		}
		got = GetSqrtPriceTarget(true, high, low)
		if got.Cmp(high) != 0 {
			t.Fatalf("expected %s (max), got %s", high, got)
		}
	})

	t.Run("oneForZero returns min", func(t *testing.T) {
		got := GetSqrtPriceTarget(false, low, high)
		if got.Cmp(low) != 0 {
			t.Fatalf("expected %s (min), got %s", low, got)
		}
		got = GetSqrtPriceTarget(false, high, low)
		if got.Cmp(low) != 0 {
			t.Fatalf("expected %s (min), got %s", low, got)
		}
	})

	t.Run("equal values", func(t *testing.T) {
		got := GetSqrtPriceTarget(true, low, low)
		if got.Cmp(low) != 0 {
			t.Fatalf("expected %s, got %s", low, got)
		}
		got = GetSqrtPriceTarget(false, low, low)
		if got.Cmp(low) != 0 {
			t.Fatalf("expected %s, got %s", low, got)
		}
	})
}

func TestSwapMath_ComputeSwapStep_ExactIn_ZeroForOne(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Div(Q96, bi(2))
	liquidity := bis("1000000000000000000")
	feePips := uint32(3000)
	amountRemaining := bi(-500_000_000_000_000_000)

	sqrtPriceNext, amountIn, amountOut, feeAmount, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	t.Run("price moves down", func(t *testing.T) {
		if sqrtPriceNext.Cmp(sqrtPriceCurrent) > 0 {
			t.Fatalf("price should not increase for zeroForOne, next=%s current=%s", sqrtPriceNext, sqrtPriceCurrent)
		}
	})

	t.Run("price stays at or above target", func(t *testing.T) {
		if sqrtPriceNext.Cmp(sqrtPriceTarget) < 0 {
			t.Fatalf("price should not go below target, next=%s target=%s", sqrtPriceNext, sqrtPriceTarget)
		}
	})

	t.Run("amounts are positive", func(t *testing.T) {
		if amountIn.Sign() < 0 {
			t.Fatalf("amountIn should be non-negative, got %s", amountIn)
		}
		if amountOut.Sign() < 0 {
			t.Fatalf("amountOut should be non-negative, got %s", amountOut)
		}
		if feeAmount.Sign() < 0 {
			t.Fatalf("feeAmount should be non-negative, got %s", feeAmount)
		}
	})

	t.Run("amountIn + feeAmount <= |amountRemaining|", func(t *testing.T) {
		total := new(big.Int).Add(amountIn, feeAmount)
		absRemaining := new(big.Int).Neg(amountRemaining)
		if total.Cmp(absRemaining) > 0 {
			t.Fatalf("amountIn(%s) + fee(%s) = %s > |remaining| %s", amountIn, feeAmount, total, absRemaining)
		}
	})

	t.Run("nonzero output", func(t *testing.T) {
		if amountOut.Sign() == 0 {
			t.Fatal("expected nonzero output for valid swap")
		}
	})
}

func TestSwapMath_ComputeSwapStep_ExactIn_OneForZero(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Mul(Q96, bi(2))
	liquidity := bis("1000000000000000000")
	feePips := uint32(3000)
	amountRemaining := bi(-500_000_000_000_000_000)

	sqrtPriceNext, amountIn, amountOut, feeAmount, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	t.Run("price moves up", func(t *testing.T) {
		if sqrtPriceNext.Cmp(sqrtPriceCurrent) < 0 {
			t.Fatalf("price should not decrease for oneForZero, next=%s current=%s", sqrtPriceNext, sqrtPriceCurrent)
		}
	})

	t.Run("price stays at or below target", func(t *testing.T) {
		if sqrtPriceNext.Cmp(sqrtPriceTarget) > 0 {
			t.Fatalf("price should not exceed target, next=%s target=%s", sqrtPriceNext, sqrtPriceTarget)
		}
	})

	t.Run("amounts non-negative", func(t *testing.T) {
		if amountIn.Sign() < 0 || amountOut.Sign() < 0 || feeAmount.Sign() < 0 {
			t.Fatalf("negative amounts: in=%s out=%s fee=%s", amountIn, amountOut, feeAmount)
		}
	})
}

func TestSwapMath_ComputeSwapStep_ExactOut_ZeroForOne(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Div(Q96, bi(2))
	liquidity := bis("1000000000000000000")
	feePips := uint32(3000)
	amountRemaining := bi(100_000_000_000_000_000)

	sqrtPriceNext, amountIn, amountOut, feeAmount, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	t.Run("price moves down for zeroForOne", func(t *testing.T) {
		if sqrtPriceNext.Cmp(sqrtPriceCurrent) > 0 {
			t.Fatalf("price should decrease, next=%s current=%s", sqrtPriceNext, sqrtPriceCurrent)
		}
	})

	t.Run("output capped by amountRemaining", func(t *testing.T) {
		if amountOut.Cmp(amountRemaining) > 0 {
			t.Fatalf("amountOut %s > amountRemaining %s", amountOut, amountRemaining)
		}
	})

	t.Run("fee computed on amountIn", func(t *testing.T) {
		if feeAmount.Sign() <= 0 {
			t.Fatalf("expected positive fee, got %s", feeAmount)
		}
		expected, fErr := MulDivRoundingUp(amountIn, bi(int64(feePips)), bi(int64(MaxSwapFee-feePips)))
		if fErr != nil {
			t.Fatalf("MulDivRoundingUp unexpected error: %v", fErr)
		}
		if feeAmount.Cmp(expected) != 0 {
			t.Fatalf("fee mismatch: got %s, expected %s", feeAmount, expected)
		}
	})
}

func TestSwapMath_ComputeSwapStep_ExactOut_OneForZero(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Mul(Q96, bi(2))
	liquidity := bis("1000000000000000000")
	feePips := uint32(500)
	amountRemaining := bi(100_000_000_000_000_000)

	sqrtPriceNext, amountIn, amountOut, feeAmount, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	t.Run("price moves up for oneForZero", func(t *testing.T) {
		if sqrtPriceNext.Cmp(sqrtPriceCurrent) < 0 {
			t.Fatalf("price should increase, next=%s current=%s", sqrtPriceNext, sqrtPriceCurrent)
		}
	})

	t.Run("output capped", func(t *testing.T) {
		if amountOut.Cmp(amountRemaining) > 0 {
			t.Fatalf("amountOut %s > amountRemaining %s", amountOut, amountRemaining)
		}
	})

	t.Run("fee positive", func(t *testing.T) {
		if feeAmount.Sign() <= 0 {
			t.Fatalf("expected positive fee, got %s", feeAmount)
		}
	})

	t.Run("all amounts non-negative", func(t *testing.T) {
		if amountIn.Sign() < 0 || amountOut.Sign() < 0 || feeAmount.Sign() < 0 {
			t.Fatalf("negative: in=%s out=%s fee=%s", amountIn, amountOut, feeAmount)
		}
	})
}

func TestSwapMath_ComputeSwapStep_ReachesTarget(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Sub(Q96, bi(1))
	liquidity := bi(1_000_000)
	feePips := uint32(3000)
	amountRemaining := new(big.Int).Neg(bis("1000000000000000000000"))

	sqrtPriceNext, _, _, _, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	if sqrtPriceNext.Cmp(sqrtPriceTarget) != 0 {
		t.Fatalf("expected to reach target %s, got %s", sqrtPriceTarget, sqrtPriceNext)
	}
}

func TestSwapMath_ComputeSwapStep_ZeroFee(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Div(Q96, bi(2))
	liquidity := bis("1000000000000000000")
	feePips := uint32(0)
	amountRemaining := bi(-500_000_000_000_000_000)

	_, amountIn, _, feeAmount, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	t.Run("zero fee means no fee taken", func(t *testing.T) {
		if feeAmount.Sign() != 0 {
			t.Fatalf("expected zero fee, got %s", feeAmount)
		}
	})

	t.Run("all input goes to amountIn", func(t *testing.T) {
		absRemaining := new(big.Int).Neg(amountRemaining)
		total := new(big.Int).Add(amountIn, feeAmount)
		if total.Cmp(absRemaining) > 0 {
			t.Fatalf("total %s > |remaining| %s with zero fee", total, absRemaining)
		}
	})
}

func TestSwapMath_ComputeSwapStep_MaxFee_ExactIn(t *testing.T) {
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Div(Q96, bi(2))
	liquidity := bis("1000000000000000000")
	feePips := uint32(MaxSwapFee)
	amountRemaining := bi(-500_000_000_000_000_000)

	sqrtPriceNext, amountIn, amountOut, feeAmount, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, amountRemaining, feePips,
	)
	if err != nil {
		t.Fatalf("ComputeSwapStep unexpected error: %v", err)
	}

	t.Run("100% fee means no swap happens", func(t *testing.T) {
		if amountIn.Sign() != 0 {
			t.Fatalf("expected zero amountIn with 100%% fee, got %s", amountIn)
		}
		if amountOut.Sign() != 0 {
			t.Fatalf("expected zero amountOut with 100%% fee, got %s", amountOut)
		}
		absRemaining := new(big.Int).Neg(amountRemaining)
		if feeAmount.Cmp(absRemaining) != 0 {
			t.Fatalf("expected fee = |remaining| = %s, got %s", absRemaining, feeAmount)
		}
		if sqrtPriceNext.Cmp(sqrtPriceCurrent) != 0 {
			t.Fatalf("price should not move with 100%% fee, next=%s current=%s", sqrtPriceNext, sqrtPriceCurrent)
		}
	})
}

func TestSwapMath_ComputeSwapStep_MaxFee_ExactOut_Errors(t *testing.T) {
	// Exact output with 100% fee must error (division by zero in fee calculation).
	sqrtPriceCurrent := new(big.Int).Set(Q96)
	sqrtPriceTarget := new(big.Int).Div(Q96, bi(2))
	liquidity := bis("1000000000000000000")

	_, _, _, _, err := ComputeSwapStep(
		sqrtPriceCurrent, sqrtPriceTarget, liquidity, bi(100), uint32(MaxSwapFee),
	)
	if err == nil {
		t.Fatal("expected error for exact output with max fee")
	}
}
