// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/big"
	"testing"
)

func TestTickMath_GetSqrtPriceAtTick(t *testing.T) {
	tests := []struct {
		name   string
		tick   int32
		expect string
	}{
		{"tick 0 = 1*2^96", 0, "79228162514264337593543950336"},
		{"min tick", MinTick, "4295128739"},
		{"max tick", MaxTick, "1461446703485210103287273052203988822378723970342"},
		{"tick 1", 1, "79232123823359799118286999568"},
		{"tick -1", -1, "79224201403219477170569942574"},
		{"tick 100", 100, "79625275426524748796330556128"},
		{"tick -100", -100, "78833030112140176575862854579"},
		{"tick 887271", 887271, "1461373636630004318706518188784493106690254656249"},
		{"tick -887271", -887271, "4295343490"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSqrtPriceAtTick(tt.tick)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			expect, _ := new(big.Int).SetString(tt.expect, 10)
			if got.Cmp(expect) != 0 {
				t.Errorf("GetSqrtPriceAtTick(%d) = %s, want %s", tt.tick, got, expect)
			}
		})
	}
}

func TestTickMath_GetSqrtPriceAtTick_OutOfRange(t *testing.T) {
	_, err := GetSqrtPriceAtTick(887273)
	if err == nil {
		t.Error("expected error for tick > max, got nil")
	}
	_, err = GetSqrtPriceAtTick(-887273)
	if err == nil {
		t.Error("expected error for tick < min, got nil")
	}
}

func TestTickMath_GetTickAtSqrtPrice(t *testing.T) {
	tests := []struct {
		name      string
		sqrtPrice string
		expect    int32
	}{
		{"min sqrt ratio", "4295128739", MinTick},
		{"1*2^96 = tick 0", "79228162514264337593543950336", 0},
		{"tick 1 price", "79232123823359799118286999568", 1},
		{"tick -1 price", "79224201403219477170569942574", -1},
		{"tick 100 price", "79625275426524748796330556128", 100},
		{"tick -100 price", "78833030112140176575862854579", -100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sqrtPrice, _ := new(big.Int).SetString(tt.sqrtPrice, 10)
			got, err := GetTickAtSqrtPrice(sqrtPrice)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.expect {
				t.Errorf("GetTickAtSqrtPrice(%s) = %d, want %d", tt.sqrtPrice, got, tt.expect)
			}
		})
	}
}

func TestTickMath_GetTickAtSqrtPrice_OutOfRange(t *testing.T) {
	// Below min
	_, err := GetTickAtSqrtPrice(big.NewInt(4295128738))
	if err == nil {
		t.Error("expected error for sqrtPrice < min, got nil")
	}

	// At max (>= is invalid)
	_, err = GetTickAtSqrtPrice(MaxSqrtRatio)
	if err == nil {
		t.Error("expected error for sqrtPrice >= max, got nil")
	}
}

func TestTickMath_RoundTrip(t *testing.T) {
	// For various ticks: GetTickAtSqrtPrice(GetSqrtPriceAtTick(tick)) == tick
	ticks := []int32{0, 1, -1, 100, -100, 500, -500, 10000, -10000, 887271, -887271, MinTick, MaxTick}

	for _, tick := range ticks {
		sqrtPrice, err := GetSqrtPriceAtTick(tick)
		if err != nil {
			t.Fatalf("GetSqrtPriceAtTick(%d): %v", tick, err)
		}

		// Skip max tick: its sqrtPrice == MaxSqrtRatio which is out of range for GetTickAtSqrtPrice
		if tick == MaxTick {
			continue
		}

		recovered, err := GetTickAtSqrtPrice(sqrtPrice)
		if err != nil {
			t.Fatalf("GetTickAtSqrtPrice(%s) for tick %d: %v", sqrtPrice, tick, err)
		}
		if recovered != tick {
			t.Errorf("round-trip failed: tick %d -> sqrtPrice %s -> tick %d", tick, sqrtPrice, recovered)
		}
	}
}

func TestTickMath_MostSignificantBit(t *testing.T) {
	tests := []struct {
		name   string
		x      *big.Int
		expect uint8
	}{
		{"one", big.NewInt(1), 0},
		{"two", big.NewInt(2), 1},
		{"255", big.NewInt(255), 7},
		{"2^128", new(big.Int).Lsh(big.NewInt(1), 128), 128},
		{"2^255", new(big.Int).Lsh(big.NewInt(1), 255), 255},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MostSignificantBit(tt.x)
			if got != tt.expect {
				t.Errorf("MostSignificantBit(%s) = %d, want %d", tt.x, got, tt.expect)
			}
		})
	}

	// Zero panics
	t.Run("zero panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for zero, got none")
			}
		}()
		MostSignificantBit(big.NewInt(0))
	})
}

func TestTickBitmapCompress(t *testing.T) {
	tb := NewTickBitmap()

	tests := []struct {
		tick, spacing, want int32
	}{
		{0, 1, 0},
		{1, 1, 1},
		{-1, 1, -1},
		{60, 60, 1},
		{-60, 60, -1},
		{59, 60, 0},
		{-59, 60, -1}, // rounds toward negative infinity
		{120, 60, 2},
		{-120, 60, -2},
		{-121, 60, -3},
	}

	for _, tt := range tests {
		got := tb.Compress(tt.tick, tt.spacing)
		if got != tt.want {
			t.Errorf("Compress(%d, %d) = %d, want %d", tt.tick, tt.spacing, got, tt.want)
		}
	}
}

func TestTickBitmapPosition(t *testing.T) {
	tb := NewTickBitmap()

	tests := []struct {
		tick    int32
		wordPos int16
		bitPos  uint8
	}{
		{0, 0, 0},
		{1, 0, 1},
		{255, 0, 255},
		{256, 1, 0},
		{-1, -1, 255},
		{-256, -1, 0},
		{-257, -2, 255},
	}

	for _, tt := range tests {
		wp, bp := tb.Position(tt.tick)
		if wp != tt.wordPos || bp != tt.bitPos {
			t.Errorf("Position(%d) = (%d, %d), want (%d, %d)", tt.tick, wp, bp, tt.wordPos, tt.bitPos)
		}
	}
}

func TestTickBitmapFlipTick(t *testing.T) {
	tb := NewTickBitmap()

	// Flip tick 0 on
	if err := tb.FlipTick(0, 1); err != nil {
		t.Fatal(err)
	}
	word := tb.getWord(0)
	if word.Bit(0) != 1 {
		t.Fatal("expected bit 0 set after flip")
	}

	// Flip tick 0 off
	if err := tb.FlipTick(0, 1); err != nil {
		t.Fatal(err)
	}
	word = tb.getWord(0)
	if word.Sign() != 0 {
		t.Fatal("expected word 0 to be zero after double flip")
	}

	// Flip tick 60 with spacing 60
	if err := tb.FlipTick(60, 60); err != nil {
		t.Fatal(err)
	}
	// compressed = 60/60 = 1, wordPos = 0, bitPos = 1
	word = tb.getWord(0)
	if word.Bit(1) != 1 {
		t.Fatal("expected bit 1 set after flipping tick 60 with spacing 60")
	}

	// Misaligned tick should error
	if err := tb.FlipTick(1, 60); err == nil {
		t.Fatal("expected error for misaligned tick")
	}
}

func TestTickBitmapNextInitialized_LTE(t *testing.T) {
	tb := NewTickBitmap()

	// Flip tick 0 (spacing 1)
	if err := tb.FlipTick(0, 1); err != nil {
		t.Fatal(err)
	}

	// Searching from tick 0, lte=true: should find tick 0
	next, init := tb.NextInitializedTickWithinOneWord(0, 1, true)
	if !init || next != 0 {
		t.Errorf("from tick 0 lte: got next=%d init=%v, want next=0 init=true", next, init)
	}

	// Searching from tick -1, lte=true: should NOT find tick 0 (it's to the right)
	next, init = tb.NextInitializedTickWithinOneWord(-1, 1, true)
	if init {
		t.Errorf("from tick -1 lte: got init=true (next=%d), want init=false", next)
	}

	// Searching from tick 255, lte=true: should find tick 0
	next, init = tb.NextInitializedTickWithinOneWord(255, 1, true)
	if !init || next != 0 {
		t.Errorf("from tick 255 lte: got next=%d init=%v, want next=0 init=true", next, init)
	}
}

func TestTickBitmapNextInitialized_GT(t *testing.T) {
	tb := NewTickBitmap()

	// Flip tick 0 (spacing 1)
	if err := tb.FlipTick(0, 1); err != nil {
		t.Fatal(err)
	}

	// Searching from tick -1, lte=false: should find tick 0
	next, init := tb.NextInitializedTickWithinOneWord(-1, 1, false)
	if !init || next != 0 {
		t.Errorf("from tick -1 gt: got next=%d init=%v, want next=0 init=true", next, init)
	}

	// Searching from tick 0, lte=false: should NOT find tick 0 (searching > 0)
	next, init = tb.NextInitializedTickWithinOneWord(0, 1, false)
	if init {
		t.Errorf("from tick 0 gt: got init=true (next=%d), want init=false", next)
	}
}

func TestTickBitmapNextInitialized_WithSpacing(t *testing.T) {
	tb := NewTickBitmap()

	// Flip tick 60 with spacing 60
	if err := tb.FlipTick(60, 60); err != nil {
		t.Fatal(err)
	}

	// From tick 60, lte=true: should find 60
	next, init := tb.NextInitializedTickWithinOneWord(60, 60, true)
	if !init || next != 60 {
		t.Errorf("from tick 60 lte: got next=%d init=%v, want next=60 init=true", next, init)
	}

	// From tick 0, lte=false: should find 60
	next, init = tb.NextInitializedTickWithinOneWord(0, 60, false)
	if !init || next != 60 {
		t.Errorf("from tick 0 gt: got next=%d init=%v, want next=60 init=true", next, init)
	}

	// From tick 119, lte=true: should find 60
	next, init = tb.NextInitializedTickWithinOneWord(119, 60, true)
	if !init || next != 60 {
		t.Errorf("from tick 119 lte: got next=%d init=%v, want next=60 init=true", next, init)
	}
}

func TestTickBitmapNextInitialized_Empty(t *testing.T) {
	tb := NewTickBitmap()

	next, init := tb.NextInitializedTickWithinOneWord(100, 1, true)
	if init {
		t.Errorf("empty lte: got init=true (next=%d), want init=false", next)
	}
	if next != 0 {
		t.Errorf("empty lte: got next=%d, want 0", next)
	}

	next, init = tb.NextInitializedTickWithinOneWord(100, 1, false)
	if init {
		t.Errorf("empty gt: got init=true (next=%d), want init=false", next)
	}
	if next != 255 {
		t.Errorf("empty gt: got next=%d, want 255", next)
	}
}

func TestTickBitmapNegativeTicks(t *testing.T) {
	tb := NewTickBitmap()

	if err := tb.FlipTick(-60, 60); err != nil {
		t.Fatal(err)
	}

	// From tick -60, lte=true: should find -60
	next, init := tb.NextInitializedTickWithinOneWord(-60, 60, true)
	if !init || next != -60 {
		t.Errorf("from tick -60 lte: got next=%d init=%v, want next=-60 init=true", next, init)
	}

	// From tick 0, lte=true: tick -60 is in a different word
	next, init = tb.NextInitializedTickWithinOneWord(0, 60, true)
	if init {
		t.Errorf("from tick 0 lte: got init=true (next=%d), tick -60 is in different word", next)
	}

	// From tick -120, lte=false should find -60
	next, init = tb.NextInitializedTickWithinOneWord(-120, 60, false)
	if !init || next != -60 {
		t.Errorf("from tick -120 gt: got next=%d init=%v, want next=-60 init=true", next, init)
	}
}

func TestTickBitmapMultipleTicks(t *testing.T) {
	tb := NewTickBitmap()

	for _, tick := range []int32{0, 60, 120, 180} {
		if err := tb.FlipTick(tick, 60); err != nil {
			t.Fatal(err)
		}
	}

	next, init := tb.NextInitializedTickWithinOneWord(120, 60, true)
	if !init || next != 120 {
		t.Errorf("got next=%d init=%v, want next=120 init=true", next, init)
	}

	next, init = tb.NextInitializedTickWithinOneWord(119, 60, true)
	if !init || next != 60 {
		t.Errorf("got next=%d init=%v, want next=60 init=true", next, init)
	}

	next, init = tb.NextInitializedTickWithinOneWord(60, 60, false)
	if !init || next != 120 {
		t.Errorf("got next=%d init=%v, want next=120 init=true", next, init)
	}
}

func TestMostSignificantBitBitmap(t *testing.T) {
	tests := []struct {
		x    int64
		want uint8
	}{
		{1, 0},
		{2, 1},
		{3, 1},
		{4, 2},
		{128, 7},
		{255, 7},
		{256, 8},
	}
	for _, tt := range tests {
		got := MostSignificantBit(big.NewInt(tt.x))
		if got != tt.want {
			t.Errorf("MSB(%d) = %d, want %d", tt.x, got, tt.want)
		}
	}
}

func TestLeastSignificantBitBitmap(t *testing.T) {
	tests := []struct {
		x    int64
		want uint8
	}{
		{1, 0},
		{2, 1},
		{3, 0},
		{4, 2},
		{128, 7},
		{256, 8},
		{6, 1}, // 110 in binary
	}
	for _, tt := range tests {
		got := LeastSignificantBit(big.NewInt(tt.x))
		if got != tt.want {
			t.Errorf("LSB(%d) = %d, want %d", tt.x, got, tt.want)
		}
	}
}
