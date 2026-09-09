package gateway

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"time"
)

// A quote and the transaction that takes it are the same question asked of the
// same venue, so the venue that priced it is the one that builds it.
//
// /v1/trade/swap used to ask the hosted provider registry for a quote — which
// this deployment has none of, so it answered 500 while /v1/trade/quote priced
// the pair from the pools a line away — and then encode the answer for the
// PoolManager precompile at 0x9010 regardless of where the price came from.
// That address answers 0x on 96369, so the calldata named a contract that is
// not there. A router on Ethereum was quoted and a precompile on our chain was
// addressed, in one response.

// SwapOrder is a quote a caller has decided to take.
type SwapOrder struct {
	TokenIn   string
	TokenOut  string
	AmountIn  *big.Int
	MinOut    *big.Int
	Recipient string
	// Deadline is unix seconds. Zero means twenty minutes from now, which is
	// what a router with no deadline field ignores anyway.
	Deadline int64
	// Fee is the tier the quote came from, in hundredths of a basis point. A
	// V3 swap has to name the pool the quote was read out of, or it executes
	// against a different one at a different price.
	Fee uint32
}

func (o SwapOrder) deadline() *big.Int {
	if o.Deadline > 0 {
		return big.NewInt(o.Deadline)
	}
	return big.NewInt(time.Now().Add(20 * time.Minute).Unix())
}

// SwapRouter02.exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))
//
// Seven fields in a static tuple, so they encode inline after the selector —
// no offset word. The repository carried this same selector against a SIX-field
// signature it named ExactInputSingle(address,address,uint256,uint256,uint160,
// bytes32); that signature hashes to f48f227e. A router handed the short form
// reads amountIn where it expects the fee tier and amountOutMinimum where it
// expects the recipient.
var selectorV3ExactInputSingle = lxrMustDecodeHex("04e45aaf")

// SwapRouter02.multicall(uint256 deadline, bytes[] data).
//
// The 02 router dropped the deadline from exactInputSingle's tuple and takes it
// here instead, which is how Uniswap's own interface sends one. Without this
// wrap a deadline handed to /v1/trade/swap was accepted and silently dropped on
// every V3 route while the V2 route honoured it — one field meaning two things
// depending on which pool happened to win the quote.
//
// amountOutMinimum bounds the PRICE; the deadline bounds the TIME. A signed
// swap that sits in a mempool and lands an hour later still passes its floor,
// at a price nobody would choose now.
var selectorV3Multicall = lxrMustDecodeHex("5ae401dc")

// Swap builds a SwapRouter02 call for a quote this venue gave.
func (v *UniswapV3Venue) Swap(o SwapOrder) (*UnsignedTxResponse, error) {
	if v.router == "" {
		return nil, nil
	}
	in, err := lxrDecodeAddress(o.TokenIn)
	if err != nil {
		return nil, fmt.Errorf("tokenIn: %w", err)
	}
	out, err := lxrDecodeAddress(o.TokenOut)
	if err != nil {
		return nil, fmt.Errorf("tokenOut: %w", err)
	}
	to, err := lxrDecodeAddress(o.Recipient)
	if err != nil {
		return nil, fmt.Errorf("recipient: %w", err)
	}
	if o.Fee == 0 {
		return nil, fmt.Errorf("a v3 swap needs the tier its quote was read from")
	}

	data := make([]byte, 0, 4+32*7)
	data = append(data, selectorV3ExactInputSingle...)
	data = append(data, lxrPadAddress(in)...)
	data = append(data, lxrPadAddress(out)...)
	data = append(data, lxrPadUint256(new(big.Int).SetUint64(uint64(o.Fee)))...)
	data = append(data, lxrPadAddress(to)...)
	data = append(data, lxrPadUint256(o.AmountIn)...)
	data = append(data, lxrPadUint256(o.MinOut)...)
	// sqrtPriceLimitX96 zero: the price bound is amountOutMinimum, and a
	// second bound here would refuse trades the caller already accepted.
	data = append(data, lxrPadUint256(big.NewInt(0))...)

	return &UnsignedTxResponse{
		To:       v.router,
		Data:     "0x" + hex.EncodeToString(withDeadline(data, o.deadline())),
		Value:    "0",
		GasLimit: 220_000,
	}, nil
}

// withDeadline wraps one call in SwapRouter02's multicall(deadline, bytes[]).
//
// Head: selector, deadline, offset to the array. Then the array: its length,
// one offset per element, and each element as a length-prefixed blob padded to
// a word — the offsets are counted from the start of the array, not the start
// of the calldata, which is the detail that makes a hand-written dynamic
// encoding revert.
func withDeadline(call []byte, deadline *big.Int) []byte {
	pad := (32 - len(call)%32) % 32

	out := make([]byte, 0, 4+32*4+len(call)+pad)
	out = append(out, selectorV3Multicall...)
	out = append(out, lxrPadUint256(deadline)...)
	out = append(out, lxrPadUint256(big.NewInt(64))...) // the array starts after two head words

	out = append(out, lxrPadUint256(big.NewInt(1))...)  // one call
	out = append(out, lxrPadUint256(big.NewInt(32))...) // which starts one word into the array body
	out = append(out, lxrPadUint256(big.NewInt(int64(len(call))))...)
	out = append(out, call...)
	return append(out, make([]byte, pad)...)
}

// Swap builds a Router02 call for a quote this venue gave.
//
// The same contract that answered getAmountsOut executes the swap, so there is
// no second address to configure and no way for the two to disagree.
func (v *UniswapV2Venue) Swap(o SwapOrder) (*UnsignedTxResponse, error) {
	if v.routerAddress == "" {
		return nil, nil
	}
	in, err := lxrDecodeAddress(o.TokenIn)
	if err != nil {
		return nil, fmt.Errorf("tokenIn: %w", err)
	}
	out, err := lxrDecodeAddress(o.TokenOut)
	if err != nil {
		return nil, fmt.Errorf("tokenOut: %w", err)
	}
	to, err := lxrDecodeAddress(o.Recipient)
	if err != nil {
		return nil, fmt.Errorf("recipient: %w", err)
	}

	// swapExactTokensForTokens(uint256,uint256,address[],address,uint256).
	// path is dynamic, so word three is an offset to it and the array itself
	// follows the five head words: 5*32 = 160.
	data := make([]byte, 0, 4+32*8)
	data = append(data, selectorV2SwapExact...)
	data = append(data, lxrPadUint256(o.AmountIn)...)
	data = append(data, lxrPadUint256(o.MinOut)...)
	data = append(data, lxrPadUint256(big.NewInt(160))...)
	data = append(data, lxrPadAddress(to)...)
	data = append(data, lxrPadUint256(o.deadline())...)
	data = append(data, lxrPadUint256(big.NewInt(2))...)
	data = append(data, lxrPadAddress(in)...)
	data = append(data, lxrPadAddress(out)...)

	return &UnsignedTxResponse{
		To:       v.routerAddress,
		Data:     "0x" + hex.EncodeToString(data),
		Value:    "0",
		GasLimit: 200_000,
	}, nil
}

// Swap returns nothing: this venue prices a market it cannot yet settle.
//
// The PoolManager at 0x9010 answers 0x on 96369 — measured, with the quoter
// selector the venue itself sends — so the arm quotes nothing there today, and
// the router beside it at 0x9012 has no ABI that can be checked against a live
// chain. Emitting calldata for either would be a transaction shaped like a
// guess. It gets a builder when the precompile answers.
func (v *NativeDEXVenue) Swap(SwapOrder) (*UnsignedTxResponse, error) { return nil, nil }

// Swap returns nothing: a price with no chain behind it.
func (v *V4Venue) Swap(SwapOrder) (*UnsignedTxResponse, error) { return nil, nil }

// Swap returns nothing: an off-chain venue quotes, it does not produce calldata.
func (b *BrokerVenue) Swap(SwapOrder) (*UnsignedTxResponse, error) { return nil, nil }
