// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"math/big"
	"testing"
)

// helper to build big.Ints concisely
func bi(v int64) *big.Int   { return big.NewInt(v) }
func bis(s string) *big.Int { v, _ := new(big.Int).SetString(s, 10); return v }

func TestSqrtPriceMath_GetNextSqrtPriceFromAmount0RoundingUp(t *testing.T) {
	// Use a realistic sqrtPrice ~= sqrt(1) * 2^96 = 2^96
	sqrtPrice := new(big.Int).Set(Q96) // 1:1 price
	liquidity := bi(1_000_000_000)     // 1e9

	t.Run("amount zero returns input", func(t *testing.T) {
		got, err := GetNextSqrtPriceFromAmount0RoundingUp(sqrtPrice, liquidity, bi(0), true)
		if err != nil {
			t.Fatal(err)
		}
		if got.Cmp(sqrtPrice) != 0 {
			t.Fatalf("expected %s, got %s", sqrtPrice, got)
		}
	})

	t.Run("add amount decreases price", func(t *testing.T) {
		// Adding token0 should decrease price (more token0 means it's cheaper)
		got, err := GetNextSqrtPriceFromAmount0RoundingUp(sqrtPrice, liquidity, bi(500_000_000), true)
		if err != nil {
			t.Fatal(err)
		}
		if got.Cmp(sqrtPrice) >= 0 {
			t.Fatalf("price should decrease when adding token0, got %s >= %s", got, sqrtPrice)
		}
		if got.Sign() <= 0 {
			t.Fatal("price should be positive")
		}
	})

	t.Run("remove amount increases price", func(t *testing.T) {
		got, err := GetNextSqrtPriceFromAmount0RoundingUp(sqrtPrice, liquidity, bi(100), false)
		if err != nil {
			t.Fatal(err)
		}
		if got.Cmp(sqrtPrice) <= 0 {
			t.Fatalf("price should increase when removing token0, got %s <= %s", got, sqrtPrice)
		}
	})

	t.Run("remove too much reverts PriceOverflow", func(t *testing.T) {
		// Removing more than liquidity/sqrtPrice should fail
		hugeAmount := new(big.Int).Add(liquidity, bi(1)) // more than liquidity at 1:1
		_, err := GetNextSqrtPriceFromAmount0RoundingUp(sqrtPrice, liquidity, hugeAmount, false)
		if err != ErrPriceOverflow {
			t.Fatalf("expected ErrPriceOverflow, got %v", err)
		}
	})
}

func TestSqrtPriceMath_GetNextSqrtPriceFromAmount1RoundingDown(t *testing.T) {
	sqrtPrice := new(big.Int).Set(Q96)
	liquidity := bi(1_000_000_000)

	t.Run("add amount increases price", func(t *testing.T) {
		got, err := GetNextSqrtPriceFromAmount1RoundingDown(sqrtPrice, liquidity, bi(100_000), true)
		if err != nil {
			t.Fatal(err)
		}
		if got.Cmp(sqrtPrice) <= 0 {
			t.Fatalf("price should increase when adding token1, got %s <= %s", got, sqrtPrice)
		}
	})

	t.Run("remove amount decreases price", func(t *testing.T) {
		got, err := GetNextSqrtPriceFromAmount1RoundingDown(sqrtPrice, liquidity, bi(100), false)
		if err != nil {
			t.Fatal(err)
		}
		if got.Cmp(sqrtPrice) >= 0 {
			t.Fatalf("price should decrease when removing token1, got %s >= %s", got, sqrtPrice)
		}
	})

	t.Run("remove too much reverts NotEnoughLiquidity", func(t *testing.T) {
		// At 1:1, removing liquidity * 2 worth of token1 should fail
		huge := new(big.Int).Mul(liquidity, bi(2))
		_, err := GetNextSqrtPriceFromAmount1RoundingDown(sqrtPrice, liquidity, huge, false)
		if err != ErrNotEnoughLiquidity {
			t.Fatalf("expected ErrNotEnoughLiquidity, got %v", err)
		}
	})
}

func TestSqrtPriceMath_GetNextSqrtPriceFromInput(t *testing.T) {
	t.Run("rejects zero price", func(t *testing.T) {
		_, err := GetNextSqrtPriceFromInput(bi(0), bi(1000), bi(100), true)
		if err != ErrInvalidPriceOrLiquidity {
			t.Fatalf("expected ErrInvalidPriceOrLiquidity, got %v", err)
		}
	})

	t.Run("rejects zero liquidity", func(t *testing.T) {
		_, err := GetNextSqrtPriceFromInput(Q96, bi(0), bi(100), true)
		if err != ErrInvalidPriceOrLiquidity {
			t.Fatalf("expected ErrInvalidPriceOrLiquidity, got %v", err)
		}
	})

	t.Run("zeroForOne delegates to Amount0", func(t *testing.T) {
		sqrtPrice := new(big.Int).Set(Q96)
		liq := bi(1_000_000)
		got, err := GetNextSqrtPriceFromInput(sqrtPrice, liq, bi(1000), true)
		if err != nil {
			t.Fatal(err)
		}
		// Adding token0 (zeroForOne input) decreases price
		if got.Cmp(sqrtPrice) >= 0 {
			t.Fatal("zeroForOne input should decrease price")
		}
	})

	t.Run("oneForZero delegates to Amount1", func(t *testing.T) {
		sqrtPrice := new(big.Int).Set(Q96)
		liq := bi(1_000_000)
		got, err := GetNextSqrtPriceFromInput(sqrtPrice, liq, bi(1000), false)
		if err != nil {
			t.Fatal(err)
		}
		// Adding token1 (oneForZero input) increases price
		if got.Cmp(sqrtPrice) <= 0 {
			t.Fatal("oneForZero input should increase price")
		}
	})
}

func TestSqrtPriceMath_GetNextSqrtPriceFromOutput(t *testing.T) {
	t.Run("rejects zero price", func(t *testing.T) {
		_, err := GetNextSqrtPriceFromOutput(bi(0), bi(1000), bi(100), true)
		if err != ErrInvalidPriceOrLiquidity {
			t.Fatalf("expected ErrInvalidPriceOrLiquidity, got %v", err)
		}
	})

	t.Run("zeroForOne output delegates to Amount1 remove", func(t *testing.T) {
		sqrtPrice := new(big.Int).Set(Q96)
		liq := bi(1_000_000_000)
		got, err := GetNextSqrtPriceFromOutput(sqrtPrice, liq, bi(100), true)
		if err != nil {
			t.Fatal(err)
		}
		// zeroForOne output removes token1, decreasing price
		if got.Cmp(sqrtPrice) >= 0 {
			t.Fatal("zeroForOne output should decrease price")
		}
	})
}

func TestSqrtPriceMath_GetAmount0Delta(t *testing.T) {
	// Two prices: sqrtPrice at tick 0 (1:1) and at a higher price
	priceLow := new(big.Int).Set(Q96)         // sqrt(1) * 2^96
	priceHigh := new(big.Int).Mul(Q96, bi(2)) // sqrt(4) * 2^96 = 2 * Q96
	liquidity := bis("1000000000000000000")   // 1e18

	t.Run("roundUp >= roundDown", func(t *testing.T) {
		up, err := GetAmount0Delta(priceLow, priceHigh, liquidity, true)
		if err != nil {
			t.Fatal(err)
		}
		down, err := GetAmount0Delta(priceLow, priceHigh, liquidity, false)
		if err != nil {
			t.Fatal(err)
		}
		if up.Cmp(down) < 0 {
			t.Fatalf("roundUp %s < roundDown %s", up, down)
		}
	})

	t.Run("symmetric in price order", func(t *testing.T) {
		a, err := GetAmount0Delta(priceLow, priceHigh, liquidity, true)
		if err != nil {
			t.Fatal(err)
		}
		b, err := GetAmount0Delta(priceHigh, priceLow, liquidity, true)
		if err != nil {
			t.Fatal(err)
		}
		if a.Cmp(b) != 0 {
			t.Fatalf("not symmetric: %s != %s", a, b)
		}
	})

	t.Run("zero liquidity gives zero", func(t *testing.T) {
		got, err := GetAmount0Delta(priceLow, priceHigh, bi(0), true)
		if err != nil {
			t.Fatal(err)
		}
		if got.Sign() != 0 {
			t.Fatalf("expected 0, got %s", got)
		}
	})

	t.Run("same price gives zero", func(t *testing.T) {
		got, err := GetAmount0Delta(priceLow, priceLow, liquidity, true)
		if err != nil {
			t.Fatal(err)
		}
		if got.Sign() != 0 {
			t.Fatalf("expected 0, got %s", got)
		}
	})

	// Known value: amount0 = L * (1/sqrt(pA) - 1/sqrt(pB))
	// With pA=1, pB=4, L=1e18: amount0 = 1e18 * (1 - 0.5) = 5e17
	t.Run("known value check", func(t *testing.T) {
		got, err := GetAmount0Delta(priceLow, priceHigh, liquidity, false)
		if err != nil {
			t.Fatal(err)
		}
		expected := bis("500000000000000000") // 5e17
		if got.Cmp(expected) != 0 {
			t.Fatalf("expected %s, got %s", expected, got)
		}
	})

	t.Run("zero sqrt price returns error", func(t *testing.T) {
		_, err := GetAmount0Delta(bi(0), priceHigh, liquidity, true)
		if err != ErrZeroSqrtPrice {
			t.Fatalf("expected ErrZeroSqrtPrice, got %v", err)
		}
	})
}

func TestSqrtPriceMath_GetAmount1Delta(t *testing.T) {
	priceLow := new(big.Int).Set(Q96)
	priceHigh := new(big.Int).Mul(Q96, bi(2))
	liquidity := bis("1000000000000000000") // 1e18

	t.Run("roundUp >= roundDown", func(t *testing.T) {
		up, err := GetAmount1Delta(priceLow, priceHigh, liquidity, true)
		if err != nil {
			t.Fatal(err)
		}
		down, err := GetAmount1Delta(priceLow, priceHigh, liquidity, false)
		if err != nil {
			t.Fatal(err)
		}
		if up.Cmp(down) < 0 {
			t.Fatalf("roundUp %s < roundDown %s", up, down)
		}
	})

	t.Run("symmetric in price order", func(t *testing.T) {
		a, err := GetAmount1Delta(priceLow, priceHigh, liquidity, false)
		if err != nil {
			t.Fatal(err)
		}
		b, err := GetAmount1Delta(priceHigh, priceLow, liquidity, false)
		if err != nil {
			t.Fatal(err)
		}
		if a.Cmp(b) != 0 {
			t.Fatalf("not symmetric: %s != %s", a, b)
		}
	})

	// Known value: amount1 = L * (sqrt(pB) - sqrt(pA))
	// With sqrtPA = Q96 (sqrt(1)), sqrtPB = 2*Q96 (sqrt(4)):
	// amount1 = L * (2*Q96 - Q96) / Q96 = L * 1 = 1e18
	t.Run("known value check", func(t *testing.T) {
		got, err := GetAmount1Delta(priceLow, priceHigh, liquidity, false)
		if err != nil {
			t.Fatal(err)
		}
		expected := bis("1000000000000000000") // 1e18
		if got.Cmp(expected) != 0 {
			t.Fatalf("expected %s, got %s", expected, got)
		}
	})
}

func TestSqrtPriceMath_GetAmount0DeltaSigned(t *testing.T) {
	priceLow := new(big.Int).Set(Q96)
	priceHigh := new(big.Int).Mul(Q96, bi(2))
	liquidity := bis("1000000000000000000")

	t.Run("positive liquidity returns negative (user pays)", func(t *testing.T) {
		got, err := GetAmount0DeltaSigned(priceLow, priceHigh, liquidity)
		if err != nil {
			t.Fatal(err)
		}
		if got.Sign() >= 0 {
			t.Fatalf("expected negative (user pays), got %s", got)
		}
	})

	t.Run("negative liquidity returns positive (user receives)", func(t *testing.T) {
		negLiq := new(big.Int).Neg(liquidity)
		got, err := GetAmount0DeltaSigned(priceLow, priceHigh, negLiq)
		if err != nil {
			t.Fatal(err)
		}
		if got.Sign() < 0 {
			t.Fatalf("expected non-negative (user receives), got %s", got)
		}
	})

	t.Run("adding rounds against user", func(t *testing.T) {
		add, err := GetAmount0DeltaSigned(priceLow, priceHigh, liquidity)
		if err != nil {
			t.Fatal(err)
		}
		remove, err := GetAmount0DeltaSigned(priceLow, priceHigh, new(big.Int).Neg(liquidity))
		if err != nil {
			t.Fatal(err)
		}
		// |add| >= remove (user pays at least as much as they'd receive)
		absAdd := new(big.Int).Neg(add)
		if absAdd.Cmp(remove) < 0 {
			t.Fatalf("|add| %s < remove %s -- rounding should favor protocol", absAdd, remove)
		}
	})
}

func TestSqrtPriceMath_GetAmount1DeltaSigned(t *testing.T) {
	priceLow := new(big.Int).Set(Q96)
	priceHigh := new(big.Int).Mul(Q96, bi(2))
	liquidity := bis("1000000000000000000")

	t.Run("positive liquidity returns negative (user pays)", func(t *testing.T) {
		got, err := GetAmount1DeltaSigned(priceLow, priceHigh, liquidity)
		if err != nil {
			t.Fatal(err)
		}
		if got.Sign() >= 0 {
			t.Fatalf("expected negative (user pays), got %s", got)
		}
	})

	t.Run("negative liquidity returns positive (user receives)", func(t *testing.T) {
		negLiq := new(big.Int).Neg(liquidity)
		got, err := GetAmount1DeltaSigned(priceLow, priceHigh, negLiq)
		if err != nil {
			t.Fatal(err)
		}
		if got.Sign() < 0 {
			t.Fatalf("expected non-negative (user receives), got %s", got)
		}
	})
}
