// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"math/big"

	luxcrypto "github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// Address is an alias for common.Address used in pool types.
type Address = common.Address

// Tick/fee type aliases matching Uniswap V4 conventions.
type (
	uint24 = uint32 // Fee in basis points
	int24  = int32  // Tick index
)

// Pool fee tiers (basis points)
const (
	Fee001 uint24 = 100    // 0.01% - stablecoins
	Fee005 uint24 = 500    // 0.05% - stable pairs
	Fee030 uint24 = 3000   // 0.30% - standard
	Fee100 uint24 = 10000  // 1.00% - exotic pairs
	FeeMax uint24 = 100000 // 10% max fee
)

// Tick spacing for different fee tiers
const (
	TickSpacing001 int24 = 1
	TickSpacing005 int24 = 10
	TickSpacing030 int24 = 60
	TickSpacing100 int24 = 200
)

// Math constants
var (
	Q96  = new(big.Int).Lsh(big.NewInt(1), 96)
	Q128 = new(big.Int).Lsh(big.NewInt(1), 128)

	MinTick int24 = -887272
	MaxTick int24 = 887272

	MinSqrtRatio    = new(big.Int).SetUint64(4295128739)
	MaxSqrtRatio, _ = new(big.Int).SetString("1461446703485210103287273052203988822378723970342", 10)
)

// MaxSwapFee is 1_000_000 (100% in hundredths of a bip).
const MaxSwapFee = 1_000_000

// Currency represents a token (native or ERC20).
// Address(0) represents native LUX.
type Currency struct {
	Address common.Address
}

// NativeCurrency represents native LUX (no wrapping needed).
var NativeCurrency = Currency{Address: common.Address{}}

// IsNative returns true if this currency is native LUX.
func (c Currency) IsNative() bool {
	return c.Address == common.Address{}
}

// PoolKey uniquely identifies a pool.
type PoolKey struct {
	Currency0   Currency       // Lower address token
	Currency1   Currency       // Higher address token
	Fee         uint24         // Fee in basis points
	TickSpacing int24          // Tick spacing for concentrated liquidity
	Hooks       common.Address // Hook contract address (zero = no hooks)
}

// BalanceDelta represents the net token changes during a transaction.
// Positive = owed to the pool, Negative = owed to the user.
type BalanceDelta struct {
	Amount0 *big.Int
	Amount1 *big.Int
}

// NewBalanceDelta creates a new balance delta.
func NewBalanceDelta(amount0, amount1 *big.Int) BalanceDelta {
	return BalanceDelta{
		Amount0: new(big.Int).Set(amount0),
		Amount1: new(big.Int).Set(amount1),
	}
}

// ZeroBalanceDelta returns a zero balance delta.
func ZeroBalanceDelta() BalanceDelta {
	return BalanceDelta{
		Amount0: big.NewInt(0),
		Amount1: big.NewInt(0),
	}
}

// Add combines two balance deltas.
func (bd BalanceDelta) Add(other BalanceDelta) BalanceDelta {
	return BalanceDelta{
		Amount0: new(big.Int).Add(bd.Amount0, other.Amount0),
		Amount1: new(big.Int).Add(bd.Amount1, other.Amount1),
	}
}

// Pool represents the state of a liquidity pool.
type Pool struct {
	SqrtPriceX96   *big.Int // sqrt(price) * 2^96 (Q64.96)
	Tick           int24    // Current tick
	Liquidity      *big.Int // Total liquidity (L)
	FeeGrowth0X128 *big.Int // Fee growth for currency0 (Q128.128)
	FeeGrowth1X128 *big.Int // Fee growth for currency1 (Q128.128)
	ProtocolFees0  *big.Int // Accumulated protocol fees currency0
	ProtocolFees1  *big.Int // Accumulated protocol fees currency1
}

// IsInitialized returns true if the pool has been initialized.
func (p *Pool) IsInitialized() bool {
	return p.SqrtPriceX96 != nil && p.SqrtPriceX96.Sign() > 0
}

// NewPool creates a new uninitialized pool.
func NewPool() *Pool {
	return &Pool{
		SqrtPriceX96:   big.NewInt(0),
		Tick:           0,
		Liquidity:      big.NewInt(0),
		FeeGrowth0X128: big.NewInt(0),
		FeeGrowth1X128: big.NewInt(0),
		ProtocolFees0:  big.NewInt(0),
		ProtocolFees1:  big.NewInt(0),
	}
}

// LPPosition represents a liquidity position.
type LPPosition struct {
	Owner                    common.Address
	TickLower                int24
	TickUpper                int24
	Liquidity                *big.Int
	FeeGrowthInside0LastX128 *big.Int
	FeeGrowthInside1LastX128 *big.Int
	TokensOwed0              *big.Int
	TokensOwed1              *big.Int
}

// LPPositionKey computes the unique position identifier.
// keccak256(abi.encodePacked(owner, tickLower, tickUpper, salt))
func LPPositionKey(owner common.Address, tickLower, tickUpper int24, salt [32]byte) [32]byte {
	data := make([]byte, 58) // 20 + 3 + 3 + 32
	copy(data[0:20], owner.Bytes())
	data[20] = byte(tickLower >> 16)
	data[21] = byte(tickLower >> 8)
	data[22] = byte(tickLower)
	data[23] = byte(tickUpper >> 16)
	data[24] = byte(tickUpper >> 8)
	data[25] = byte(tickUpper)
	copy(data[26:58], salt[:])
	h := luxcrypto.Keccak256(data)
	var key [32]byte
	copy(key[:], h)
	return key
}

// SwapParams contains parameters for a swap.
// V4 convention: amountSpecified < 0 means exact input, > 0 means exact output.
type SwapParams struct {
	ZeroForOne        bool     // true = swap currency0 for currency1
	AmountSpecified   *big.Int // Negative = exact input, Positive = exact output
	SqrtPriceLimitX96 *big.Int // Price limit (sqrt(price) * 2^96)
}

// ModifyLiquidityParams contains parameters for adding/removing liquidity.
type ModifyLiquidityParams struct {
	TickLower      int24
	TickUpper      int24
	LiquidityDelta *big.Int // Positive = add, Negative = remove
	Salt           [32]byte // Position salt for uniqueness
}

// TickInfo stores per-tick state matching Uniswap V4 Pool.TickInfo.
type TickInfo struct {
	LiquidityGross        *big.Int // Total position liquidity referencing this tick (uint128)
	LiquidityNet          *big.Int // Net liquidity change when crossing left-to-right (int128)
	FeeGrowthOutside0X128 *big.Int // Fee growth outside for token0 (uint256)
	FeeGrowthOutside1X128 *big.Int // Fee growth outside for token1 (uint256)
}

// NewTickInfo creates a zero-initialized TickInfo.
func NewTickInfo() *TickInfo {
	return &TickInfo{
		LiquidityGross:        big.NewInt(0),
		LiquidityNet:          big.NewInt(0),
		FeeGrowthOutside0X128: big.NewInt(0),
		FeeGrowthOutside1X128: big.NewInt(0),
	}
}

// IsTickInitialized returns true if any liquidity references this tick.
func (ti *TickInfo) IsTickInitialized() bool {
	return ti.LiquidityGross.Sign() > 0
}

// TickBitmap tracks initialized tick state in a packed bitmap.
type TickBitmap struct {
	bitmap map[int16]*big.Int // wordPos -> 256-bit word
}

// NewTickBitmap creates an empty tick bitmap.
func NewTickBitmap() *TickBitmap {
	return &TickBitmap{
		bitmap: make(map[int16]*big.Int),
	}
}

// PoolState extends Pool with V4 tick-level state for concentrated liquidity.
type PoolState struct {
	*Pool
	Ticks       map[int32]*TickInfo      // Per-tick state
	TickBitmap  *TickBitmap              // Initialized tick tracking
	Positions   map[[32]byte]*LPPosition // Position states
	TickSpacing int32                    // From pool key
	LPFee       uint32                   // LP fee in pips
	ProtocolFee uint32                   // Protocol fee in pips
}

// NewPoolState wraps a Pool with V4 tick-level state.
func NewPoolState(pool *Pool, tickSpacing int32, lpFee uint32) *PoolState {
	return &PoolState{
		Pool:        pool,
		Ticks:       make(map[int32]*TickInfo),
		TickBitmap:  NewTickBitmap(),
		Positions:   make(map[[32]byte]*LPPosition),
		TickSpacing: tickSpacing,
		LPFee:       lpFee,
	}
}

// getOrCreateTick returns the TickInfo for a tick, creating it if absent.
func (ps *PoolState) getOrCreateTick(tick int32) *TickInfo {
	ti, ok := ps.Ticks[tick]
	if !ok {
		ti = NewTickInfo()
		ps.Ticks[tick] = ti
	}
	return ti
}

// Engine is the interface for DEX computation.
type Engine interface {
	// Initialize computes the initial tick from sqrtPriceX96.
	Initialize(sqrtPriceX96 *big.Int) (int24, error)

	// Swap executes the full V4 tick-crossing swap loop.
	Swap(pool *PoolState, params SwapParams) (BalanceDelta, error)

	// ModifyLiquidity adds/removes concentrated liquidity.
	ModifyLiquidity(pool *PoolState, owner Address, params ModifyLiquidityParams) (callerDelta BalanceDelta, feesAccrued BalanceDelta, err error)

	// Donate distributes tokens to LPs via fee growth updates.
	Donate(pool *PoolState, amount0, amount1 *big.Int) (BalanceDelta, error)

	// Quote estimates swap output without mutating state.
	Quote(pool *Pool, amountIn *big.Int, zeroForOne bool) *big.Int
}

// Errors
var (
	ErrPoolNotInitialized     = dexError("pool not initialized")
	ErrPoolAlreadyInitialized = dexError("pool already initialized")
	ErrInvalidTickRange       = dexError("invalid tick range")
	ErrInsufficientLiquidity  = dexError("insufficient liquidity")
	ErrPriceLimitReached      = dexError("price limit reached")
	ErrInvalidFee             = dexError("invalid fee")
	ErrInvalidSqrtPrice       = dexError("invalid sqrt price")
	ErrTickOutOfRange         = dexError("tick out of range")
	ErrPoolNoLiquidity        = dexError("no liquidity in pool")
)

type dexError string

func (e dexError) Error() string { return string(e) }
