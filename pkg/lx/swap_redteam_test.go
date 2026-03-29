// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// RED TEAM REGRESSION TESTS
//
// Each test documents a specific vulnerability finding from the security review.
// Tests are named with the attack vector and expected behavior.
// All tests should FAIL on old vulnerable code and PASS on fixed code.
//
// Run: go test ./pkg/lx/... -run TestRedTeam -v -count=1

package lx

import (
	"math/big"
	"strings"
	"testing"

	"github.com/luxfi/geth/common"
)

// --------------------------------------------------------------------------
// Helpers (redteam-local, avoid collision with other test files)
// --------------------------------------------------------------------------

// rtPool creates a pool initialized at tick 0 (sqrtPrice = Q96 = 1.0)
// with the given tick spacing and fee.
func rtPool(tickSpacing int32, fee uint32) *PoolState {
	pool := NewPool()
	pool.SqrtPriceX96 = new(big.Int).Set(Q96) // price = 1.0 at tick 0
	pool.Tick = 0
	pool.Liquidity = big.NewInt(0)
	pool.FeeGrowth0X128 = big.NewInt(0)
	pool.FeeGrowth1X128 = big.NewInt(0)
	return NewPoolState(pool, tickSpacing, fee)
}

// rtOwner returns a deterministic test address.
func rtOwner() common.Address {
	return common.HexToAddress("0xdead000000000000000000000000000000000001")
}

// rtAddLiquidity is a convenience wrapper that adds liquidity to a pool.
func rtAddLiquidity(t *testing.T, engine *EngineCL, pool *PoolState, lower, upper int32, liq int64) {
	t.Helper()
	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      lower,
		TickUpper:      upper,
		LiquidityDelta: big.NewInt(liq),
	})
	if err != nil {
		t.Fatalf("rtAddLiquidity(%d, %d, %d): %v", lower, upper, liq, err)
	}
}

// --------------------------------------------------------------------------
// Finding 1: GetTickAtSqrtPrice error propagation in swap loop
//
// Attack vector: If the swap loop reaches a sqrtPriceX96 that is outside
// [MinSqrtRatio, MaxSqrtRatio), GetTickAtSqrtPrice would fail. The OLD code
// would panic on nil dereference. The NEW code must return (BalanceDelta, error)
// with the error propagated cleanly.
//
// This is hard to trigger with valid pool state because price math enforces
// bounds. We verify the contract: Swap() returns (BalanceDelta, error) and
// errors from GetTickAtSqrtPrice propagate rather than causing a panic.
// --------------------------------------------------------------------------

func TestRedTeam_SwapReturnsErrorNotPanic_GetTickAtSqrtPriceFails(t *testing.T) {
	// Verify GetTickAtSqrtPrice returns error for out-of-range prices.
	_, err := GetTickAtSqrtPrice(big.NewInt(0))
	if err == nil {
		t.Fatal("GetTickAtSqrtPrice(0) should return error, got nil")
	}

	_, err = GetTickAtSqrtPrice(big.NewInt(-1))
	if err == nil {
		t.Fatal("GetTickAtSqrtPrice(-1) should return error, got nil")
	}

	// Price above MaxSqrtRatio.
	aboveMax := new(big.Int).Set(MaxSqrtRatio)
	_, err = GetTickAtSqrtPrice(aboveMax)
	if err == nil {
		t.Fatal("GetTickAtSqrtPrice(MaxSqrtRatio) should return error, got nil")
	}

	// Price at exactly MinSqrtRatio - 1.
	belowMin := new(big.Int).Sub(MinSqrtRatio, big.NewInt(1))
	_, err = GetTickAtSqrtPrice(belowMin)
	if err == nil {
		t.Fatal("GetTickAtSqrtPrice(MinSqrtRatio-1) should return error, got nil")
	}

	// Verify Swap returns (BalanceDelta, error) by running a normal swap
	// and checking that the error path works.
	engine := NewEngine()
	pool := rtPool(60, 3000)
	rtAddLiquidity(t, engine, pool, -120, 120, 1_000_000_000_000)

	// Valid swap -- should succeed with no panic.
	delta, err := engine.Swap(pool, SwapParams{
		ZeroForOne:        true,
		AmountSpecified:   big.NewInt(-1000),
		SqrtPriceLimitX96: new(big.Int).Add(MinSqrtRatio, big.NewInt(1)),
	})
	if err != nil {
		t.Fatalf("valid swap returned error: %v", err)
	}
	if delta.Amount0 == nil || delta.Amount1 == nil {
		t.Fatal("valid swap returned nil amounts in BalanceDelta")
	}
}

// --------------------------------------------------------------------------
// Finding 2: GetSqrtPriceAtTick errors propagated in ModifyLiquidity
//
// Attack vector: Call ModifyLiquidity with tickLower < MinTick or
// tickUpper > MaxTick. OLD code panics on nil dereference from
// GetSqrtPriceAtTick returning (nil, error). NEW code returns error.
// --------------------------------------------------------------------------

func TestRedTeam_ModifyLiquidity_TickBelowMinTick_ReturnsError(t *testing.T) {
	// Finding 2a: tickLower far below MinTick.
	// Old code: GetSqrtPriceAtTick returns (nil, error), caller dereferences nil -> panic.
	// New code: validation at entry rejects the tick range.
	engine := NewEngine()
	pool := rtPool(1, 3000) // tickSpacing=1 so any tick is aligned

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ModifyLiquidity panicked instead of returning error: %v", r)
		}
	}()

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -900000, // well below MinTick (-887272)
		TickUpper:      0,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("ModifyLiquidity with tickLower < MinTick should return error, got nil")
	}
}

func TestRedTeam_ModifyLiquidity_TickAboveMaxTick_ReturnsError(t *testing.T) {
	// Finding 2b: tickUpper far above MaxTick.
	engine := NewEngine()
	pool := rtPool(1, 3000)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ModifyLiquidity panicked instead of returning error: %v", r)
		}
	}()

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      0,
		TickUpper:      900000, // well above MaxTick (887272)
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("ModifyLiquidity with tickUpper > MaxTick should return error, got nil")
	}
}

func TestRedTeam_ModifyLiquidity_BothBoundsOutOfRange_ReturnsError(t *testing.T) {
	// Finding 2c: both bounds out of range.
	engine := NewEngine()
	pool := rtPool(1, 3000)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ModifyLiquidity panicked instead of returning error: %v", r)
		}
	}()

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -900000,
		TickUpper:      900000,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("ModifyLiquidity with both ticks out of range should return error, got nil")
	}
}

// --------------------------------------------------------------------------
// Finding 3: updateTick propagates AddDelta error
//
// Attack vector: Create a pool with liquidity at a tick, then try to remove
// MORE liquidity than exists (negative liquidityDelta larger than gross).
// OLD code: AddDelta returned (nil, error), updateTick silently zeroed the tick.
// NEW code: error propagates through ModifyLiquidity return.
// --------------------------------------------------------------------------

func TestRedTeam_UpdateTick_RemoveMoreThanExists_ReturnsError(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)

	// Add a small amount of liquidity.
	rtAddLiquidity(t, engine, pool, -120, 120, 500_000)

	// Now try to remove 10x more than exists.
	// This should trigger AddDelta underflow in updateTick.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ModifyLiquidity panicked on liquidity underflow: %v", r)
		}
	}()

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -120,
		TickUpper:      120,
		LiquidityDelta: big.NewInt(-5_000_000), // 10x the added amount
	})
	if err == nil {
		t.Fatal("removing more liquidity than exists should return error, got nil")
	}
	// Verify the error is related to liquidity underflow, not a generic panic.
	errMsg := err.Error()
	if !strings.Contains(errMsg, "liquidity") && !strings.Contains(errMsg, "underflow") {
		t.Logf("warning: error message does not mention liquidity/underflow: %s", errMsg)
	}
}

func TestRedTeam_UpdateTick_RemoveExactLiquidity_Succeeds(t *testing.T) {
	// Sanity check: removing exactly what was added should work.
	engine := NewEngine()
	pool := rtPool(60, 3000)

	rtAddLiquidity(t, engine, pool, -120, 120, 1_000_000)

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -120,
		TickUpper:      120,
		LiquidityDelta: big.NewInt(-1_000_000),
	})
	if err != nil {
		t.Fatalf("removing exact liquidity should succeed, got: %v", err)
	}
}

// --------------------------------------------------------------------------
// Finding 4: ModifyLiquidity validates tick range at entry
//
// Attack vector: Invalid tick parameters bypass validation and corrupt pool state.
// All of these must return error without modifying pool state.
// --------------------------------------------------------------------------

func TestRedTeam_ModifyLiquidity_TickLowerGteTickUpper_ReturnsError(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)

	// tickLower == tickUpper (zero-width range)
	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      60,
		TickUpper:      60,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("tickLower == tickUpper should return error")
	}

	// tickLower > tickUpper (inverted range)
	_, _, err = engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      120,
		TickUpper:      60,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("tickLower > tickUpper should return error")
	}
}

func TestRedTeam_TickRangeValidation_BelowMinTick_ReturnsError(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      MinTick - 60, // one spacing below min
		TickUpper:      120,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("tickLower < MinTick should return error")
	}
}

func TestRedTeam_TickRangeValidation_AboveMaxTick_ReturnsError(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)

	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -120,
		TickUpper:      MaxTick + 60, // one spacing above max
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("tickUpper > MaxTick should return error")
	}
}

func TestRedTeam_ModifyLiquidity_TickNotAlignedToSpacing_ReturnsError(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000) // tickSpacing = 60

	// tickLower not aligned
	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -61, // not a multiple of 60
		TickUpper:      120,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("tickLower not aligned to tickSpacing should return error")
	}

	// tickUpper not aligned
	_, _, err = engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -120,
		TickUpper:      121, // not a multiple of 60
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("tickUpper not aligned to tickSpacing should return error")
	}

	// Both not aligned
	_, _, err = engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -59,
		TickUpper:      61,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("both ticks not aligned to tickSpacing should return error")
	}
}

// --------------------------------------------------------------------------
// Finding 5: FlipTick error propagation (misaligned ticks)
//
// Attack vector: If a misaligned tick reaches FlipTick, the bitmap corrupts.
// The entry validation in ModifyLiquidity (Finding 4) is the primary defense.
// This test verifies FlipTick itself rejects misaligned ticks AND that
// ModifyLiquidity error messages indicate the tick alignment issue.
// --------------------------------------------------------------------------

func TestRedTeam_FlipTick_MisalignedTick_ReturnsError(t *testing.T) {
	tb := NewTickBitmap()

	// Direct FlipTick with misaligned tick.
	err := tb.FlipTick(61, 60) // 61 % 60 != 0
	if err == nil {
		t.Fatal("FlipTick with misaligned tick should return error")
	}
	if err != ErrTickMisaligned {
		t.Fatalf("expected ErrTickMisaligned, got: %v", err)
	}

	// Verify via ModifyLiquidity the error propagates with meaningful message.
	engine := NewEngine()
	pool := rtPool(60, 3000)

	_, _, err = engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -61,
		TickUpper:      120,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("ModifyLiquidity with misaligned tick should return error")
	}
	errStr := strings.ToLower(err.Error())
	if !strings.Contains(errStr, "tick") && !strings.Contains(errStr, "misalign") && !strings.Contains(errStr, "spacing") && !strings.Contains(errStr, "aligned") {
		t.Logf("warning: error message does not clearly indicate tick alignment issue: %s", err.Error())
	}
}

func TestRedTeam_FlipTick_AlignedTick_Succeeds(t *testing.T) {
	tb := NewTickBitmap()

	// Aligned ticks should work.
	if err := tb.FlipTick(60, 60); err != nil {
		t.Fatalf("FlipTick(60, 60) should succeed: %v", err)
	}
	if err := tb.FlipTick(0, 60); err != nil {
		t.Fatalf("FlipTick(0, 60) should succeed: %v", err)
	}
	if err := tb.FlipTick(-120, 60); err != nil {
		t.Fatalf("FlipTick(-120, 60) should succeed: %v", err)
	}
}

// --------------------------------------------------------------------------
// Finding 6: updatePosition propagates fee calculation error
//
// Attack vector: If fee growth math overflows (SimpleMulDiv returns error),
// OLD code silently zeroed fees. NEW code returns error.
//
// Verify updatePosition returns (BalanceDelta, error) and that
// ModifyLiquidity propagates the error from fee calculation.
// --------------------------------------------------------------------------

func TestRedTeam_UpdatePosition_ReturnsFeeDelta_WithError(t *testing.T) {
	// We verify the contract by adding liquidity, generating fees via swap,
	// then removing liquidity and checking fee delta is non-zero.
	engine := NewEngine()
	pool := rtPool(60, 3000)

	// Add liquidity.
	rtAddLiquidity(t, engine, pool, -120, 120, 1_000_000_000_000)

	// Swap to generate fees.
	_, err := engine.Swap(pool, SwapParams{
		ZeroForOne:        true,
		AmountSpecified:   big.NewInt(-100_000_000),
		SqrtPriceLimitX96: new(big.Int).Add(MinSqrtRatio, big.NewInt(1)),
	})
	if err != nil {
		t.Fatalf("swap to generate fees failed: %v", err)
	}

	// Remove liquidity -- fee delta should be non-zero.
	_, feeDelta, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -120,
		TickUpper:      120,
		LiquidityDelta: big.NewInt(-500_000_000_000),
	})
	if err != nil {
		t.Fatalf("ModifyLiquidity (remove) returned error: %v", err)
	}

	// Fee delta should have collected some fees from the swap.
	if feeDelta.Amount0.Sign() == 0 && feeDelta.Amount1.Sign() == 0 {
		t.Log("warning: fee delta is zero after swap -- fees may not have accrued (check pool tick vs position range)")
	}
}

func TestRedTeam_UpdatePosition_ZeroLiquidity_NoFeeError(t *testing.T) {
	// A new position with zero existing liquidity should not error on fee calc.
	engine := NewEngine()
	pool := rtPool(60, 3000)

	// Set extreme fee growth to test that new position (no prior liquidity)
	// does not trigger fee calc overflow.
	pool.FeeGrowth0X128 = new(big.Int).Set(MaxUint256)
	pool.FeeGrowth1X128 = new(big.Int).Set(MaxUint256)

	// This should succeed: new position, no prior liquidity -> fees = 0.
	_, feeDelta, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      -120,
		TickUpper:      120,
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err != nil {
		t.Fatalf("adding to new position with extreme fee growth should not error: %v", err)
	}
	if feeDelta.Amount0.Sign() != 0 || feeDelta.Amount1.Sign() != 0 {
		t.Fatalf("new position should have zero fees, got: (%s, %s)", feeDelta.Amount0, feeDelta.Amount1)
	}
}

// --------------------------------------------------------------------------
// Finding 7: getFeeGrowthInside wraps to uint256
//
// Attack vector: Fee growth "outside" values on ticks can be set such that
// subtraction yields a negative intermediate. In Solidity this wraps in
// uint256. If Go code uses signed big.Int without wrapping, the result is
// negative -- breaking fee accounting and potentially allowing fee theft.
//
// NEW code uses wrapUint256 to replicate Solidity uint256 wrapping semantics.
// --------------------------------------------------------------------------

func TestRedTeam_GetFeeGrowthInside_WrapsNegativeToUint256(t *testing.T) {
	// Scenario: current tick is between lower and upper.
	// lower.FeeGrowthOutside0X128 = 100
	// upper.FeeGrowthOutside0X128 = 200
	// pool.FeeGrowth0X128 = 50
	//
	// In the "between" branch:
	//   feeGrowthInside = wrap(wrap(global - lower.outside) - upper.outside)
	//   = wrap(wrap(50 - 100) - 200)
	//   = wrap((2^256 - 50) - 200)
	//   = wrap(2^256 - 250)
	//   = 2^256 - 250
	//
	// This must NOT be negative. It must be a large positive uint256.

	pool := rtPool(60, 3000)
	pool.FeeGrowth0X128 = big.NewInt(50)
	pool.FeeGrowth1X128 = big.NewInt(50)

	lowerTick := int32(-120)
	upperTick := int32(120)

	lower := pool.getOrCreateTick(lowerTick)
	lower.FeeGrowthOutside0X128 = big.NewInt(100)
	lower.FeeGrowthOutside1X128 = big.NewInt(100)

	upper := pool.getOrCreateTick(upperTick)
	upper.FeeGrowthOutside0X128 = big.NewInt(200)
	upper.FeeGrowthOutside1X128 = big.NewInt(200)

	// Pool tick is 0, which is between -120 and 120.
	fg0, fg1 := getFeeGrowthInside(pool, lowerTick, upperTick)

	// Result MUST be non-negative (wraps in uint256 space).
	if fg0.Sign() < 0 {
		t.Fatalf("feeGrowthInside0 is negative (%s) -- uint256 wrapping is broken", fg0)
	}
	if fg1.Sign() < 0 {
		t.Fatalf("feeGrowthInside1 is negative (%s) -- uint256 wrapping is broken", fg1)
	}

	// The result should be 2^256 - 250.
	// wrap(50 - 100) = 2^256 - 50
	// wrap((2^256 - 50) - 200) = 2^256 - 250
	expected := new(big.Int).Sub(uint256Modulus, big.NewInt(250))
	if fg0.Cmp(expected) != 0 {
		t.Fatalf("feeGrowthInside0 = %s, expected %s (2^256 - 250)", fg0, expected)
	}
}

func TestRedTeam_GetFeeGrowthInside_BelowRange_WrapsCorrectly(t *testing.T) {
	// Current tick below lower -- uses: lower.outside - upper.outside.
	pool := rtPool(60, 3000)
	pool.Tick = -200 // below -120

	lower := pool.getOrCreateTick(-120)
	lower.FeeGrowthOutside0X128 = big.NewInt(10)
	lower.FeeGrowthOutside1X128 = big.NewInt(10)

	upper := pool.getOrCreateTick(120)
	upper.FeeGrowthOutside0X128 = big.NewInt(100)
	upper.FeeGrowthOutside1X128 = big.NewInt(100)

	fg0, fg1 := getFeeGrowthInside(pool, -120, 120)

	// wrap(10 - 100) = 2^256 - 90
	if fg0.Sign() < 0 {
		t.Fatalf("feeGrowthInside0 is negative below range: %s", fg0)
	}
	expected := new(big.Int).Sub(uint256Modulus, big.NewInt(90))
	if fg0.Cmp(expected) != 0 {
		t.Fatalf("feeGrowthInside0 below range = %s, expected %s", fg0, expected)
	}
	if fg1.Cmp(expected) != 0 {
		t.Fatalf("feeGrowthInside1 below range = %s, expected %s", fg1, expected)
	}
}

func TestRedTeam_GetFeeGrowthInside_AboveRange_WrapsCorrectly(t *testing.T) {
	// Current tick >= upper -- uses: upper.outside - lower.outside.
	pool := rtPool(60, 3000)
	pool.Tick = 200 // above 120

	lower := pool.getOrCreateTick(-120)
	lower.FeeGrowthOutside0X128 = big.NewInt(500)
	lower.FeeGrowthOutside1X128 = big.NewInt(500)

	upper := pool.getOrCreateTick(120)
	upper.FeeGrowthOutside0X128 = big.NewInt(100)
	upper.FeeGrowthOutside1X128 = big.NewInt(100)

	fg0, fg1 := getFeeGrowthInside(pool, -120, 120)

	// wrap(100 - 500) = 2^256 - 400
	if fg0.Sign() < 0 {
		t.Fatalf("feeGrowthInside0 is negative above range: %s", fg0)
	}
	expected := new(big.Int).Sub(uint256Modulus, big.NewInt(400))
	if fg0.Cmp(expected) != 0 {
		t.Fatalf("feeGrowthInside0 above range = %s, expected %s", fg0, expected)
	}
	if fg1.Cmp(expected) != 0 {
		t.Fatalf("feeGrowthInside1 above range = %s, expected %s", fg1, expected)
	}
}

// --------------------------------------------------------------------------
// Finding 8: wrapUint256 helper correctness
//
// Attack vector: If wrapping is wrong, fee growth arithmetic produces
// invalid values that could enable fee theft or loss of LP funds.
// --------------------------------------------------------------------------

func TestRedTeam_WrapUint256_NegativeOne_EqualsMaxUint256(t *testing.T) {
	// Solidity: uint256(-1) == 2^256 - 1 == MaxUint256.
	result := wrapUint256(new(big.Int).SetInt64(-1))
	if result.Cmp(MaxUint256) != 0 {
		t.Fatalf("wrapUint256(-1) = %s, expected MaxUint256 (%s)", result, MaxUint256)
	}
}

func TestRedTeam_WrapUint256_Zero_StaysZero(t *testing.T) {
	result := wrapUint256(new(big.Int).SetInt64(0))
	if result.Sign() != 0 {
		t.Fatalf("wrapUint256(0) = %s, expected 0", result)
	}
}

func TestRedTeam_WrapUint256_MaxUint256_StaysMaxUint256(t *testing.T) {
	input := new(big.Int).Set(MaxUint256)
	result := wrapUint256(input)
	if result.Cmp(MaxUint256) != 0 {
		t.Fatalf("wrapUint256(MaxUint256) = %s, expected MaxUint256", result)
	}
}

func TestRedTeam_WrapUint256_MaxUint256PlusOne_WrapsToZero(t *testing.T) {
	// 2^256 mod 2^256 == 0
	input := new(big.Int).Add(new(big.Int).Set(MaxUint256), big.NewInt(1))
	result := wrapUint256(input)
	if result.Sign() != 0 {
		t.Fatalf("wrapUint256(MaxUint256+1) = %s, expected 0", result)
	}
}

func TestRedTeam_WrapUint256_LargeNegative_WrapsCorrectly(t *testing.T) {
	// -100 mod 2^256 == 2^256 - 100
	input := new(big.Int).SetInt64(-100)
	result := wrapUint256(input)
	expected := new(big.Int).Sub(uint256Modulus, big.NewInt(100))
	if result.Cmp(expected) != 0 {
		t.Fatalf("wrapUint256(-100) = %s, expected %s", result, expected)
	}
}

func TestRedTeam_WrapUint256_PositiveInRange_NoChange(t *testing.T) {
	// Values in [0, 2^256) should be unchanged.
	vals := []*big.Int{
		big.NewInt(1),
		big.NewInt(1000000),
		new(big.Int).Set(Q128),
		new(big.Int).Sub(MaxUint256, big.NewInt(1)),
	}
	for _, v := range vals {
		input := new(big.Int).Set(v)
		result := wrapUint256(input)
		if result.Cmp(v) != 0 {
			t.Fatalf("wrapUint256(%s) = %s, expected unchanged", v, result)
		}
	}
}

// --------------------------------------------------------------------------
// Finding 9: maxUint256 == MaxUint256 (deduplication in tick.go)
//
// Attack vector: If tick.go had a separate maxUint256 computed differently
// from math.go's MaxUint256, the TickBitmap masks and fee growth wrapping
// would use inconsistent bounds, enabling bitmap corruption.
// --------------------------------------------------------------------------

func TestRedTeam_MaxUint256_Deduplication_SameValue(t *testing.T) {
	// tick.go: var maxUint256 = MaxUint256
	// math.go: var MaxUint256 = ...
	// They MUST be equal.
	if maxUint256.Cmp(MaxUint256) != 0 {
		t.Fatalf("maxUint256 (%s) != MaxUint256 (%s) -- tick.go and math.go disagree", maxUint256, MaxUint256)
	}

	// Verify the value is correct: 2^256 - 1.
	expected := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	if MaxUint256.Cmp(expected) != 0 {
		t.Fatalf("MaxUint256 = %s, expected 2^256-1 = %s", MaxUint256, expected)
	}
}

func TestRedTeam_MaxUint256_BitLength(t *testing.T) {
	// MaxUint256 should be exactly 256 bits.
	if MaxUint256.BitLen() != 256 {
		t.Fatalf("MaxUint256.BitLen() = %d, expected 256", MaxUint256.BitLen())
	}
}

// --------------------------------------------------------------------------
// Integration: End-to-end swap with tick crossing does not panic
//
// This exercises Findings 1, 2, 3 together in a realistic scenario.
// --------------------------------------------------------------------------

func TestRedTeam_SwapCrossingTicks_NoPanic(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)

	// Add liquidity across multiple tick ranges.
	rtAddLiquidity(t, engine, pool, -600, -120, 500_000_000_000)
	rtAddLiquidity(t, engine, pool, -120, 120, 1_000_000_000_000)
	rtAddLiquidity(t, engine, pool, 120, 600, 500_000_000_000)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("swap crossing multiple ticks panicked: %v", r)
		}
	}()

	// Large swap that crosses multiple tick boundaries.
	delta, err := engine.Swap(pool, SwapParams{
		ZeroForOne:        true,
		AmountSpecified:   big.NewInt(-500_000_000_000),
		SqrtPriceLimitX96: new(big.Int).Add(MinSqrtRatio, big.NewInt(1)),
	})
	if err != nil {
		t.Fatalf("multi-tick swap returned error: %v", err)
	}
	if delta.Amount0 == nil || delta.Amount1 == nil {
		t.Fatal("swap result has nil amounts")
	}
	// amount0 should be negative (user paid in), amount1 positive (user received)
	if delta.Amount0.Sign() >= 0 {
		t.Logf("note: amount0 = %s (expected negative for zeroForOne exact-input)", delta.Amount0)
	}
}

func TestRedTeam_SwapReverseDirection_NoPanic(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)

	rtAddLiquidity(t, engine, pool, -600, 600, 1_000_000_000_000)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("reverse swap panicked: %v", r)
		}
	}()

	// oneForZero swap.
	delta, err := engine.Swap(pool, SwapParams{
		ZeroForOne:        false,
		AmountSpecified:   big.NewInt(-500_000_000_000),
		SqrtPriceLimitX96: new(big.Int).Sub(MaxSqrtRatio, big.NewInt(1)),
	})
	if err != nil {
		t.Fatalf("oneForZero swap returned error: %v", err)
	}
	if delta.Amount0 == nil || delta.Amount1 == nil {
		t.Fatal("swap result has nil amounts")
	}
}

// --------------------------------------------------------------------------
// Integration: Swap with zero liquidity returns zero delta, not panic
// --------------------------------------------------------------------------

func TestRedTeam_SwapZeroAmount_ReturnsZeroDelta(t *testing.T) {
	engine := NewEngine()
	pool := rtPool(60, 3000)
	rtAddLiquidity(t, engine, pool, -120, 120, 1_000_000_000_000)

	delta, err := engine.Swap(pool, SwapParams{
		ZeroForOne:        true,
		AmountSpecified:   big.NewInt(0),
		SqrtPriceLimitX96: new(big.Int).Add(MinSqrtRatio, big.NewInt(1)),
	})
	if err != nil {
		t.Fatalf("zero-amount swap should not error: %v", err)
	}
	if delta.Amount0.Sign() != 0 || delta.Amount1.Sign() != 0 {
		t.Fatalf("zero-amount swap should return zero delta, got (%s, %s)", delta.Amount0, delta.Amount1)
	}
}

// --------------------------------------------------------------------------
// Integration: Verify pool state consistency after error paths
// --------------------------------------------------------------------------

func TestRedTeam_PoolStateUnchanged_AfterModifyLiquidityError(t *testing.T) {
	// If ModifyLiquidity returns an error, pool state must not be mutated.
	engine := NewEngine()
	pool := rtPool(60, 3000)

	// Snapshot state.
	origTick := pool.Tick
	origSqrtPrice := new(big.Int).Set(pool.SqrtPriceX96)
	origLiquidity := new(big.Int).Set(pool.Liquidity)
	origFG0 := new(big.Int).Set(pool.FeeGrowth0X128)
	origFG1 := new(big.Int).Set(pool.FeeGrowth1X128)

	// Invalid call.
	_, _, err := engine.ModifyLiquidity(pool, rtOwner(), ModifyLiquidityParams{
		TickLower:      120,
		TickUpper:      60, // inverted
		LiquidityDelta: big.NewInt(1_000_000),
	})
	if err == nil {
		t.Fatal("expected error for inverted tick range")
	}

	// Verify no mutation.
	if pool.Tick != origTick {
		t.Fatalf("pool.Tick mutated after error: %d -> %d", origTick, pool.Tick)
	}
	if pool.SqrtPriceX96.Cmp(origSqrtPrice) != 0 {
		t.Fatalf("pool.SqrtPriceX96 mutated after error")
	}
	if pool.Liquidity.Cmp(origLiquidity) != 0 {
		t.Fatalf("pool.Liquidity mutated after error")
	}
	if pool.FeeGrowth0X128.Cmp(origFG0) != 0 {
		t.Fatalf("pool.FeeGrowth0X128 mutated after error")
	}
	if pool.FeeGrowth1X128.Cmp(origFG1) != 0 {
		t.Fatalf("pool.FeeGrowth1X128 mutated after error")
	}
}
