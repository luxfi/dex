// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"errors"
	"math/big"
)

func mustParseBig(s string) *big.Int {
	n, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("mustParseBig: invalid number: " + s)
	}
	return n
}

func mustParseHex(s string) *big.Int {
	n, ok := new(big.Int).SetString(s, 16)
	if !ok {
		panic("mustParseHex: invalid hex: " + s)
	}
	return n
}

// Magic constants from Uniswap V4 TickMath.sol getSqrtPriceAtTick.
// Each constant corresponds to a bit position in absTick.
var tickMagicConstants = [20]*big.Int{
	mustParseHex("fffcb933bd6fad37aa2d162d1a594001"),
	mustParseHex("fff97272373d413259a46990580e213a"),
	mustParseHex("fff2e50f5f656932ef12357cf3c7fdcc"),
	mustParseHex("ffe5caca7e10e4e61c3624eaa0941cd0"),
	mustParseHex("ffcb9843d60f6159c9db58835c926644"),
	mustParseHex("ff973b41fa98c081472e6896dfb254c0"),
	mustParseHex("ff2ea16466c96a3843ec78b326b52861"),
	mustParseHex("fe5dee046a99a2a811c461f1969c3053"),
	mustParseHex("fcbe86c7900a88aedcffc83b479aa3a4"),
	mustParseHex("f987a7253ac413176f2b074cf7815e54"),
	mustParseHex("f3392b0822b70005940c7a398e4b70f3"),
	mustParseHex("e7159475a2c29b7443b29c7fa6e889d9"),
	mustParseHex("d097f3bdfd2022b8845ad8f792aa5825"),
	mustParseHex("a9f746462d870fdf8a65dc1f90e061e5"),
	mustParseHex("70d869a156d2a1b890bb3df62baf32f7"),
	mustParseHex("31be135f97d08fd981231505542fcfa6"),
	mustParseHex("9aa508b5b7a84e1c677de54f3e99bc9"),
	mustParseHex("5d6af8dedb81196699c329225ee604"),
	mustParseHex("2216e584f5fa1ea926041bedfe98"),
	mustParseHex("48a170391f7dc42444e8fa2"),
}

var (
	errTickOutOfRange      = errors.New("tick out of range")
	errSqrtPriceOutOfRange = errors.New("sqrtPriceX96 out of range")
)

// maxUint256 aliases MaxUint256 from math.go for internal use.
var maxUint256 = MaxUint256

// ErrTickMisaligned is returned when a tick is not a multiple of tickSpacing.
var ErrTickMisaligned = errors.New("tick not aligned to tick spacing")

// ErrBitMathZero is returned when MostSignificantBit or LeastSignificantBit is called with zero.
var ErrBitMathZero = errors.New("bit math: input must be > 0")

// GetSqrtPriceAtTick returns the sqrt price as a Q64.96 value for a given tick.
// Equivalent to Uniswap V4 TickMath.sol getSqrtPriceAtTick.
func GetSqrtPriceAtTick(tick int32) (*big.Int, error) {
	absTick := int32(tick)
	if absTick < 0 {
		absTick = -absTick
	}
	if absTick > MaxTick {
		return nil, errTickOutOfRange
	}

	// Start: if bit 0 is set, use magic constant 0; otherwise 1<<128
	var price *big.Int
	if absTick&1 != 0 {
		price = new(big.Int).Set(tickMagicConstants[0])
	} else {
		price = new(big.Int).Lsh(big.NewInt(1), 128)
	}

	// Multiply by each magic constant where the corresponding bit is set
	for i := 1; i < 20; i++ {
		if absTick&(1<<uint(i)) != 0 {
			price.Mul(price, tickMagicConstants[i])
			price.Rsh(price, 128)
		}
	}

	// If tick is positive, invert: price = MaxUint256 / price
	if tick > 0 {
		price.Div(maxUint256, price)
	}

	// Shift from Q128.128 to Q64.96: add (1<<32 - 1) then >> 32
	// This rounds up in the shift
	roundUp := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 32), big.NewInt(1))
	price.Add(price, roundUp)
	price.Rsh(price, 32)

	return price, nil
}

// GetTickAtSqrtPrice returns the greatest tick whose sqrt price is <= sqrtPriceX96.
// Equivalent to Uniswap V4 TickMath.sol getTickAtSqrtPrice.
func GetTickAtSqrtPrice(sqrtPriceX96 *big.Int) (int32, error) {
	if sqrtPriceX96.Cmp(MinSqrtRatio) < 0 || sqrtPriceX96.Cmp(MaxSqrtRatio) >= 0 {
		return 0, errSqrtPriceOutOfRange
	}

	// price = sqrtPriceX96 << 32
	price := new(big.Int).Lsh(sqrtPriceX96, 32)

	// msb = highest set bit (as int for safe arithmetic)
	msb := int(MostSignificantBit(price))

	// Normalize r to [2^127, 2^128)
	var r *big.Int
	if msb >= 128 {
		r = new(big.Int).Rsh(price, uint(msb-127))
	} else {
		r = new(big.Int).Lsh(price, uint(127-msb))
	}

	// log_2 = (msb - 128) << 64, stored as a signed big.Int
	log2 := new(big.Int).Lsh(big.NewInt(int64(msb-128)), 64)

	// 14 iterations: refine log_2 with fractional bits at positions 63 down to 50
	for i := 0; i < 14; i++ {
		// r = (r * r) >> 127
		r.Mul(r, r)
		r.Rsh(r, 127)

		// f = r >> 128 (0 or 1)
		f := new(big.Int).Rsh(r, 128)

		// log_2 |= f << (63 - i)
		shifted := new(big.Int).Lsh(f, uint(63-i))
		log2.Or(log2, shifted)

		// r >>= f
		r.Rsh(r, uint(f.Int64()))
	}

	// log_sqrt10001 = log_2 * 255738958999603826347141
	logSqrt10001 := new(big.Int).Mul(log2, mustParseBig("255738958999603826347141"))

	// tickLow = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
	tickLowBig := new(big.Int).Sub(logSqrt10001, mustParseBig("3402992956809132418596140100660247210"))
	tickLowBig.Rsh(tickLowBig, 128)

	// tickHi = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128
	tickHiBig := new(big.Int).Add(logSqrt10001, mustParseBig("291339464771989622907027621153398088495"))
	tickHiBig.Rsh(tickHiBig, 128)

	tickLow := int32(tickLowBig.Int64())
	tickHi := int32(tickHiBig.Int64())

	if tickLow == tickHi {
		return tickLow, nil
	}

	// Verify which tick is correct by computing sqrt price at tickHi
	sqrtPriceHi, err := GetSqrtPriceAtTick(tickHi)
	if err != nil {
		return 0, err
	}

	if sqrtPriceHi.Cmp(sqrtPriceX96) <= 0 {
		return tickHi, nil
	}
	return tickLow, nil
}

// MostSignificantBit returns the index of the highest set bit (0-255).
// Panics if x is zero.
func MostSignificantBit(x *big.Int) uint8 {
	if x.Sign() == 0 {
		panic("MostSignificantBit: zero")
	}
	return uint8(x.BitLen() - 1)
}

// LeastSignificantBit returns the index of the lowest set bit (0-255).
// Panics if x is zero.
func LeastSignificantBit(x *big.Int) uint8 {
	if x.Sign() == 0 {
		panic("LeastSignificantBit: zero")
	}
	// The lowest set bit position is the number of trailing zeros.
	for i := 0; i < 256; i++ {
		if x.Bit(i) != 0 {
			return uint8(i)
		}
	}
	// Unreachable for non-zero values within 256 bits.
	panic("LeastSignificantBit: no set bit in 256-bit range")
}

// getWord returns the 256-bit word at wordPos, defaulting to zero.
func (tb *TickBitmap) getWord(wordPos int16) *big.Int {
	if w, ok := tb.bitmap[wordPos]; ok {
		return new(big.Int).Set(w)
	}
	return big.NewInt(0)
}

// Compress rounds a tick toward negative infinity by tickSpacing.
// Returns compressed = tick / tickSpacing, adjusted down for negative remainders.
func (tb *TickBitmap) Compress(tick, tickSpacing int32) int32 {
	compressed := tick / tickSpacing
	if tick < 0 && tick%tickSpacing != 0 {
		compressed--
	}
	return compressed
}

// Position returns the word position and bit position for a compressed tick.
// wordPos = tick >> 8 (arithmetic shift right)
// bitPos  = tick & 0xFF
func (tb *TickBitmap) Position(tick int32) (wordPos int16, bitPos uint8) {
	wordPos = int16(tick >> 8)
	bitPos = uint8(tick & 0xFF)
	return
}

// FlipTick toggles the initialized state for a tick.
// The tick must be aligned to tickSpacing.
func (tb *TickBitmap) FlipTick(tick, tickSpacing int32) error {
	if tick%tickSpacing != 0 {
		return ErrTickMisaligned
	}
	compressed := tick / tickSpacing
	wordPos, bitPos := tb.Position(compressed)

	mask := new(big.Int).Lsh(big.NewInt(1), uint(bitPos))
	word := tb.getWord(wordPos)
	word.Xor(word, mask)
	tb.bitmap[wordPos] = word
	return nil
}

// NextInitializedTickWithinOneWord finds the next initialized tick within
// the same word (or adjacent word) as the given tick.
//
// If lte is true, searches left/down (less than or equal to tick).
// If lte is false, searches right/up (greater than tick).
//
// Returns the next tick and whether it is initialized. The search covers
// at most 256 compressed ticks (one word).
func (tb *TickBitmap) NextInitializedTickWithinOneWord(tick, tickSpacing int32, lte bool) (next int32, initialized bool) {
	compressed := tb.Compress(tick, tickSpacing)

	if lte {
		wordPos, bitPos := tb.Position(compressed)
		// All 1s at or to the right of bitPos.
		mask := new(big.Int).Rsh(maxUint256, uint(255-bitPos))
		masked := new(big.Int).And(tb.getWord(wordPos), mask)

		initialized = masked.Sign() != 0
		if initialized {
			msb := MostSignificantBit(masked)
			next = (compressed - int32(bitPos) + int32(msb)) * tickSpacing
		} else {
			next = (compressed - int32(bitPos)) * tickSpacing
		}
	} else {
		// Start from the next tick.
		compressed++
		wordPos, bitPos := tb.Position(compressed)
		// All 1s at or to the left of bitPos.
		mask := new(big.Int).Not(new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(bitPos)), big.NewInt(1)))
		// Clamp to 256 bits -- Not() on a positive number produces a negative big.Int.
		mask.And(mask, maxUint256)
		masked := new(big.Int).And(tb.getWord(wordPos), mask)

		initialized = masked.Sign() != 0
		if initialized {
			lsb := LeastSignificantBit(masked)
			next = (compressed + int32(lsb) - int32(bitPos)) * tickSpacing
		} else {
			next = (compressed + int32(255) - int32(bitPos)) * tickSpacing
		}
	}
	return
}
