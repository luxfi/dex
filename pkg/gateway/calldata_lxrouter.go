package gateway

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
)

// EVM ABI encoding for LXRouter (0x9012) precompile calls.
//
// Function selectors (first 4 bytes of keccak256 of the signature):
//
//	ExactInputSingle(address,address,uint256,uint256,uint160,bytes32)
//	ExactInput(bytes,uint256,uint256,uint160)
//	ExactOutputSingle(address,address,uint256,uint256,uint160,bytes32)
//	ExactOutput(bytes,uint256,uint256,uint160)

// LXRouterAddress is the V4 router precompile.
const LXRouterAddress = "0x0000000000000000000000000000000000009012"

// Precomputed function selectors (keccak256 prefix).
// These match the V4 router's Solidity interface.
var (
	// exactInputSingle(address tokenIn, address tokenOut, uint256 amountIn,
	//   uint256 amountOutMinimum, uint160 sqrtPriceLimitX96, bytes32 poolId)
	SelectorExactInputSingle = lxrMustDecodeHex("04e45aaf")

	// exactInput(bytes path, uint256 amountIn, uint256 amountOutMinimum,
	//   uint160 sqrtPriceLimitX96)
	SelectorExactInput = lxrMustDecodeHex("b858183f")

	// exactOutputSingle(address tokenIn, address tokenOut, uint256 amountOut,
	//   uint256 amountInMaximum, uint160 sqrtPriceLimitX96, bytes32 poolId)
	SelectorExactOutputSingle = lxrMustDecodeHex("5023b4df")

	// exactOutput(bytes path, uint256 amountOut, uint256 amountInMaximum,
	//   uint160 sqrtPriceLimitX96)
	SelectorExactOutput = lxrMustDecodeHex("09b81346")
)

// BuildExactInputSingleCalldata encodes an ExactInputSingle call to LXRouter.
//
// Parameters:
//   - tokenIn, tokenOut: 20-byte addresses (0x-prefixed hex)
//   - amountIn: input amount in wei
//   - amountOutMin: minimum output (slippage protection)
//   - sqrtPriceLimitX96: price limit (0 for no limit)
//   - poolID: 32-byte pool identifier (0x-prefixed hex, or empty for default)
//
// Returns 0x-prefixed hex-encoded calldata.
func BuildExactInputSingleCalldata(tokenIn, tokenOut string, amountIn, amountOutMin, sqrtPriceLimitX96 *big.Int, poolID string) (string, error) {
	inAddr, err := lxrDecodeAddress(tokenIn)
	if err != nil {
		return "", fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := lxrDecodeAddress(tokenOut)
	if err != nil {
		return "", fmt.Errorf("tokenOut: %w", err)
	}

	var poolBytes [32]byte
	if poolID != "" {
		pid, err := lxrDecodeBytes32(poolID)
		if err != nil {
			return "", fmt.Errorf("poolId: %w", err)
		}
		poolBytes = pid
	}

	// ABI layout: selector(4) + tokenIn(32) + tokenOut(32) + amountIn(32) +
	//             amountOutMin(32) + sqrtPriceLimitX96(32) + poolId(32) = 196 bytes
	data := make([]byte, 0, 196)
	data = append(data, SelectorExactInputSingle...)
	data = append(data, lxrPadAddress(inAddr)...)
	data = append(data, lxrPadAddress(outAddr)...)
	data = append(data, lxrPadUint256(amountIn)...)
	data = append(data, lxrPadUint256(amountOutMin)...)
	data = append(data, lxrPadUint256(sqrtPriceLimitX96)...)
	data = append(data, poolBytes[:]...)

	return "0x" + hex.EncodeToString(data), nil
}

// BuildExactInputCalldata encodes a multi-hop ExactInput call to LXRouter.
//
// Parameters:
//   - path: sequence of token addresses for the multi-hop route (at least 2)
//   - amountIn: input amount in wei
//   - amountOutMin: minimum output (slippage protection)
//   - sqrtPriceLimitX96: price limit (0 for no limit)
//
// The path is ABI-encoded as dynamic bytes: packed 20-byte addresses.
// Returns 0x-prefixed hex-encoded calldata.
func BuildExactInputCalldata(path []string, amountIn, amountOutMin, sqrtPriceLimitX96 *big.Int) (string, error) {
	if len(path) < 2 {
		return "", fmt.Errorf("path must have at least 2 tokens, got %d", len(path))
	}

	// Encode packed path: each address is 20 bytes, no padding.
	packedPath := make([]byte, 0, len(path)*20)
	for i, addr := range path {
		a, err := lxrDecodeAddress(addr)
		if err != nil {
			return "", fmt.Errorf("path[%d]: %w", i, err)
		}
		packedPath = append(packedPath, a[:]...)
	}

	// ABI layout for dynamic bytes:
	// selector(4) + offset_to_path(32) + amountIn(32) + amountOutMin(32) +
	// sqrtPriceLimitX96(32) + path_length(32) + path_data(padded to 32)
	pathPadded := lxrPadBytes(packedPath)

	data := make([]byte, 0, 4+32*5+len(pathPadded))
	data = append(data, SelectorExactInput...)
	data = append(data, lxrPadUint256(big.NewInt(128))...)              // offset to path data (4 * 32 = 128)
	data = append(data, lxrPadUint256(amountIn)...)                     // amountIn
	data = append(data, lxrPadUint256(amountOutMin)...)                 // amountOutMinimum
	data = append(data, lxrPadUint256(sqrtPriceLimitX96)...)            // sqrtPriceLimitX96
	data = append(data, lxrPadUint256(big.NewInt(int64(len(packedPath))))...) // path length
	data = append(data, pathPadded...)                                  // path data

	return "0x" + hex.EncodeToString(data), nil
}

// BuildExactOutputSingleCalldata encodes an ExactOutputSingle call to LXRouter.
func BuildExactOutputSingleCalldata(tokenIn, tokenOut string, amountOut, amountInMax, sqrtPriceLimitX96 *big.Int, poolID string) (string, error) {
	inAddr, err := lxrDecodeAddress(tokenIn)
	if err != nil {
		return "", fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := lxrDecodeAddress(tokenOut)
	if err != nil {
		return "", fmt.Errorf("tokenOut: %w", err)
	}

	var poolBytes [32]byte
	if poolID != "" {
		pid, err := lxrDecodeBytes32(poolID)
		if err != nil {
			return "", fmt.Errorf("poolId: %w", err)
		}
		poolBytes = pid
	}

	data := make([]byte, 0, 196)
	data = append(data, SelectorExactOutputSingle...)
	data = append(data, lxrPadAddress(inAddr)...)
	data = append(data, lxrPadAddress(outAddr)...)
	data = append(data, lxrPadUint256(amountOut)...)
	data = append(data, lxrPadUint256(amountInMax)...)
	data = append(data, lxrPadUint256(sqrtPriceLimitX96)...)
	data = append(data, poolBytes[:]...)

	return "0x" + hex.EncodeToString(data), nil
}

// BuildExactOutputCalldata encodes a multi-hop ExactOutput call to LXRouter.
func BuildExactOutputCalldata(path []string, amountOut, amountInMax, sqrtPriceLimitX96 *big.Int) (string, error) {
	if len(path) < 2 {
		return "", fmt.Errorf("path must have at least 2 tokens, got %d", len(path))
	}

	packedPath := make([]byte, 0, len(path)*20)
	for i, addr := range path {
		a, err := lxrDecodeAddress(addr)
		if err != nil {
			return "", fmt.Errorf("path[%d]: %w", i, err)
		}
		packedPath = append(packedPath, a[:]...)
	}

	pathPadded := lxrPadBytes(packedPath)

	data := make([]byte, 0, 4+32*5+len(pathPadded))
	data = append(data, SelectorExactOutput...)
	data = append(data, lxrPadUint256(big.NewInt(128))...)
	data = append(data, lxrPadUint256(amountOut)...)
	data = append(data, lxrPadUint256(amountInMax)...)
	data = append(data, lxrPadUint256(sqrtPriceLimitX96)...)
	data = append(data, lxrPadUint256(big.NewInt(int64(len(packedPath))))...)
	data = append(data, pathPadded...)

	return "0x" + hex.EncodeToString(data), nil
}

// --- LXRouter ABI encoding primitives ---
// Prefixed with lxr to avoid collision with order_handlers.go helpers.

// lxrDecodeAddress parses a 0x-prefixed hex address into a 20-byte array.
func lxrDecodeAddress(s string) ([20]byte, error) {
	var addr [20]byte
	s = strings.TrimPrefix(s, "0x")
	if len(s) != 40 {
		return addr, fmt.Errorf("invalid address length: %d", len(s)+2)
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return addr, fmt.Errorf("invalid hex: %w", err)
	}
	copy(addr[:], b)
	return addr, nil
}

// lxrDecodeBytes32 parses a 0x-prefixed hex string into a 32-byte array.
func lxrDecodeBytes32(s string) ([32]byte, error) {
	var out [32]byte
	s = strings.TrimPrefix(s, "0x")
	if len(s) > 64 {
		return out, fmt.Errorf("bytes32 too long: %d hex chars", len(s))
	}
	// Left-pad with zeros if shorter than 64 hex chars.
	s = strings.Repeat("0", 64-len(s)) + s
	b, err := hex.DecodeString(s)
	if err != nil {
		return out, fmt.Errorf("invalid hex: %w", err)
	}
	copy(out[:], b)
	return out, nil
}

// lxrPadAddress left-pads a 20-byte address to 32 bytes (EVM ABI word).
func lxrPadAddress(addr [20]byte) []byte {
	var word [32]byte
	copy(word[12:], addr[:])
	return word[:]
}

// lxrPadUint256 encodes a big.Int as a 32-byte big-endian word.
// Nil is treated as zero.
func lxrPadUint256(n *big.Int) []byte {
	var word [32]byte
	if n != nil && n.Sign() > 0 {
		b := n.Bytes()
		if len(b) > 32 {
			b = b[:32] // truncate (should not happen for valid uint256)
		}
		copy(word[32-len(b):], b)
	}
	return word[:]
}

// lxrPadBytes pads arbitrary bytes to the next 32-byte boundary.
func lxrPadBytes(b []byte) []byte {
	padded := make([]byte, ((len(b)+31)/32)*32)
	copy(padded, b)
	return padded
}

// lxrMustDecodeHex decodes a hex string, panicking on error (init-time only).
func lxrMustDecodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic("invalid hex literal: " + s)
	}
	return b
}

// DecodeCalldata parses 0x-prefixed calldata and returns the selector and raw parameter bytes.
// Useful for verifying calldata in tests.
func DecodeCalldata(calldata string) (selector []byte, params []byte, err error) {
	calldata = strings.TrimPrefix(calldata, "0x")
	raw, err := hex.DecodeString(calldata)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid hex: %w", err)
	}
	if len(raw) < 4 {
		return nil, nil, fmt.Errorf("calldata too short: %d bytes", len(raw))
	}
	return raw[:4], raw[4:], nil
}

// DecodeUint256 reads a 32-byte big-endian uint256 from ABI-encoded params at the given word offset.
func DecodeUint256(params []byte, wordOffset int) (*big.Int, error) {
	start := wordOffset * 32
	if start+32 > len(params) {
		return nil, fmt.Errorf("params too short for word at offset %d", wordOffset)
	}
	return new(big.Int).SetBytes(params[start : start+32]), nil
}

// DecodeAddress reads a left-padded 20-byte address from ABI-encoded params at the given word offset.
func DecodeAddress(params []byte, wordOffset int) (string, error) {
	start := wordOffset * 32
	if start+32 > len(params) {
		return "", fmt.Errorf("params too short for word at offset %d", wordOffset)
	}
	// Address is in the last 20 bytes of the 32-byte word.
	addr := params[start+12 : start+32]
	return "0x" + hex.EncodeToString(addr), nil
}

// init-time validation that selectors are exactly 4 bytes.
func init() {
	for name, sel := range map[string][]byte{
		"ExactInputSingle":  SelectorExactInputSingle,
		"ExactInput":        SelectorExactInput,
		"ExactOutputSingle": SelectorExactOutputSingle,
		"ExactOutput":       SelectorExactOutput,
	} {
		if len(sel) != 4 {
			panic(fmt.Sprintf("selector %s is %d bytes, expected 4", name, len(sel)))
		}
	}
}
