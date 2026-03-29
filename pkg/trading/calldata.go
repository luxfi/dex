package trading

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
)

// EVM ABI encoding for LXRouter (0x9012) precompile calls.

// Precomputed function selectors (keccak256 prefix).
var (
	SelectorExactInputSingle  = mustDecodeHex("04e45aaf")
	SelectorExactInput        = mustDecodeHex("b858183f")
	SelectorExactOutputSingle = mustDecodeHex("5023b4df")
	SelectorExactOutput       = mustDecodeHex("09b81346")
)

func BuildExactInputSingleCalldata(tokenIn, tokenOut string, amountIn, amountOutMin, sqrtPriceLimitX96 *big.Int, poolID string) (string, error) {
	inAddr, err := decodeAddress(tokenIn)
	if err != nil {
		return "", fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := decodeAddress(tokenOut)
	if err != nil {
		return "", fmt.Errorf("tokenOut: %w", err)
	}

	var poolBytes [32]byte
	if poolID != "" {
		pid, err := decodeBytes32(poolID)
		if err != nil {
			return "", fmt.Errorf("poolId: %w", err)
		}
		poolBytes = pid
	}

	data := make([]byte, 0, 196)
	data = append(data, SelectorExactInputSingle...)
	data = append(data, padAddress(inAddr)...)
	data = append(data, padAddress(outAddr)...)
	data = append(data, padUint256(amountIn)...)
	data = append(data, padUint256(amountOutMin)...)
	data = append(data, padUint256(sqrtPriceLimitX96)...)
	data = append(data, poolBytes[:]...)

	return "0x" + hex.EncodeToString(data), nil
}

func BuildExactInputCalldata(path []string, amountIn, amountOutMin, sqrtPriceLimitX96 *big.Int) (string, error) {
	if len(path) < 2 {
		return "", fmt.Errorf("path must have at least 2 tokens, got %d", len(path))
	}

	packedPath := make([]byte, 0, len(path)*20)
	for i, addr := range path {
		a, err := decodeAddress(addr)
		if err != nil {
			return "", fmt.Errorf("path[%d]: %w", i, err)
		}
		packedPath = append(packedPath, a[:]...)
	}

	pathPadded := padBytes(packedPath)

	data := make([]byte, 0, 4+32*5+len(pathPadded))
	data = append(data, SelectorExactInput...)
	data = append(data, padUint256(big.NewInt(128))...)
	data = append(data, padUint256(amountIn)...)
	data = append(data, padUint256(amountOutMin)...)
	data = append(data, padUint256(sqrtPriceLimitX96)...)
	data = append(data, padUint256(big.NewInt(int64(len(packedPath))))...)
	data = append(data, pathPadded...)

	return "0x" + hex.EncodeToString(data), nil
}

func BuildExactOutputSingleCalldata(tokenIn, tokenOut string, amountOut, amountInMax, sqrtPriceLimitX96 *big.Int, poolID string) (string, error) {
	inAddr, err := decodeAddress(tokenIn)
	if err != nil {
		return "", fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := decodeAddress(tokenOut)
	if err != nil {
		return "", fmt.Errorf("tokenOut: %w", err)
	}

	var poolBytes [32]byte
	if poolID != "" {
		pid, err := decodeBytes32(poolID)
		if err != nil {
			return "", fmt.Errorf("poolId: %w", err)
		}
		poolBytes = pid
	}

	data := make([]byte, 0, 196)
	data = append(data, SelectorExactOutputSingle...)
	data = append(data, padAddress(inAddr)...)
	data = append(data, padAddress(outAddr)...)
	data = append(data, padUint256(amountOut)...)
	data = append(data, padUint256(amountInMax)...)
	data = append(data, padUint256(sqrtPriceLimitX96)...)
	data = append(data, poolBytes[:]...)

	return "0x" + hex.EncodeToString(data), nil
}

func BuildExactOutputCalldata(path []string, amountOut, amountInMax, sqrtPriceLimitX96 *big.Int) (string, error) {
	if len(path) < 2 {
		return "", fmt.Errorf("path must have at least 2 tokens, got %d", len(path))
	}

	packedPath := make([]byte, 0, len(path)*20)
	for i, addr := range path {
		a, err := decodeAddress(addr)
		if err != nil {
			return "", fmt.Errorf("path[%d]: %w", i, err)
		}
		packedPath = append(packedPath, a[:]...)
	}

	pathPadded := padBytes(packedPath)

	data := make([]byte, 0, 4+32*5+len(pathPadded))
	data = append(data, SelectorExactOutput...)
	data = append(data, padUint256(big.NewInt(128))...)
	data = append(data, padUint256(amountOut)...)
	data = append(data, padUint256(amountInMax)...)
	data = append(data, padUint256(sqrtPriceLimitX96)...)
	data = append(data, padUint256(big.NewInt(int64(len(packedPath))))...)
	data = append(data, pathPadded...)

	return "0x" + hex.EncodeToString(data), nil
}

// --- ABI encoding primitives ---

func decodeAddress(s string) ([20]byte, error) {
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

func decodeBytes32(s string) ([32]byte, error) {
	var out [32]byte
	s = strings.TrimPrefix(s, "0x")
	if len(s) > 64 {
		return out, fmt.Errorf("bytes32 too long: %d hex chars", len(s))
	}
	s = strings.Repeat("0", 64-len(s)) + s
	b, err := hex.DecodeString(s)
	if err != nil {
		return out, fmt.Errorf("invalid hex: %w", err)
	}
	copy(out[:], b)
	return out, nil
}

func padAddress(addr [20]byte) []byte {
	var word [32]byte
	copy(word[12:], addr[:])
	return word[:]
}

func padUint256(n *big.Int) []byte {
	var word [32]byte
	if n != nil && n.Sign() > 0 {
		b := n.Bytes()
		if len(b) > 32 {
			b = b[:32]
		}
		copy(word[32-len(b):], b)
	}
	return word[:]
}

func padBytes(b []byte) []byte {
	padded := make([]byte, ((len(b)+31)/32)*32)
	copy(padded, b)
	return padded
}

func mustDecodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic("invalid hex literal: " + s)
	}
	return b
}

// DecodeCalldata parses 0x-prefixed calldata and returns selector + params.
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

func DecodeUint256(params []byte, wordOffset int) (*big.Int, error) {
	start := wordOffset * 32
	if start+32 > len(params) {
		return nil, fmt.Errorf("params too short for word at offset %d", wordOffset)
	}
	return new(big.Int).SetBytes(params[start : start+32]), nil
}

func DecodeAddress(params []byte, wordOffset int) (string, error) {
	start := wordOffset * 32
	if start+32 > len(params) {
		return "", fmt.Errorf("params too short for word at offset %d", wordOffset)
	}
	addr := params[start+12 : start+32]
	return "0x" + hex.EncodeToString(addr), nil
}

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
