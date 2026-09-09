package gateway

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
)

// The ABI encoding every venue here shares: pad an address, pad a uint256,
// read an address back. One encoder, so two callers cannot disagree about what
// a word looks like.
//
// What used to sit above these — four calldata builders for the V4 router
// precompile at 0x9012 — is gone. They were reachable only from their own
// tests, and they sent selector 04e45aaf under a signature they named
// ExactInputSingle(address,address,uint256,uint256,uint160,bytes32). That
// signature hashes to f48f227e. 04e45aaf belongs to SwapRouter02's
// exactInputSingle((address,address,uint24,address,uint256,uint256,uint160)),
// a seven-field tuple, which is what venue_swap.go now encodes against the
// router each chain actually has. The precompile they addressed answers 0x on
// 96369; it gets a builder when it answers something, written against an ABI
// that can be checked on a live chain.

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

// lxrPadAddress left-pads a 20-byte address to 32 bytes (EVM ABI word).
func lxrPadAddress(addr [20]byte) []byte {
	var word [32]byte
	copy(word[12:], addr[:])
	return word[:]
}

// lxrPadUint256 encodes a big.Int as a 32-byte big-endian word.
// Nil is treated as zero.
// lxrPadUint256 right-aligns a number in a 32-byte word.
//
// A value too large to fit is written as its LOW 32 bytes, which is what the
// EVM itself does to a uint256, rather than its high ones — the previous
// `b[:32]` kept the leading bytes, so a number one bit too wide came out as a
// completely unrelated quantity in a field that spends money. Nothing should
// reach here that wide: `wholeUnits` refuses it where an amount enters.
func lxrPadUint256(n *big.Int) []byte {
	var word [32]byte
	if n != nil && n.Sign() > 0 {
		b := n.Bytes()
		if len(b) > 32 {
			b = b[len(b)-32:]
		}
		copy(word[32-len(b):], b)
	}
	return word[:]
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
