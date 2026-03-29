package gateway

import (
	"bytes"
	"math/big"
	"strings"
	"testing"
)

func TestBuildExactInputSingleCalldata(t *testing.T) {
	unit := e18()
	amountIn := new(big.Int).Mul(big.NewInt(1000), unit)
	amountOutMin := new(big.Int).Mul(big.NewInt(500), unit)

	calldata, err := BuildExactInputSingleCalldata(
		testLUSD, testWBTC, amountIn, amountOutMin, big.NewInt(0),
		"0x0000000000000000000000000000000000000000000000000000000000000001",
	)
	if err != nil {
		t.Fatalf("build calldata: %v", err)
	}

	// 4 (selector) + 6*32 (params) = 196 bytes = 392 hex + "0x" = 394 chars.
	if len(calldata) != 394 {
		t.Errorf("calldata length = %d, want 394", len(calldata))
	}

	sel, params, err := DecodeCalldata(calldata)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(sel, SelectorExactInputSingle) {
		t.Errorf("selector mismatch")
	}

	decoded, _ := DecodeAddress(params, 0)
	if !strings.EqualFold(decoded, testLUSD) {
		t.Errorf("tokenIn = %s, want %s", decoded, testLUSD)
	}

	decoded, _ = DecodeAddress(params, 1)
	if !strings.EqualFold(decoded, testWBTC) {
		t.Errorf("tokenOut = %s, want %s", decoded, testWBTC)
	}

	decodedAmt, _ := DecodeUint256(params, 2)
	if decodedAmt.Cmp(amountIn) != 0 {
		t.Errorf("amountIn = %s, want %s", decodedAmt, amountIn)
	}

	decodedMin, _ := DecodeUint256(params, 3)
	if decodedMin.Cmp(amountOutMin) != 0 {
		t.Errorf("amountOutMin = %s, want %s", decodedMin, amountOutMin)
	}
}

func TestBuildExactInputCalldata(t *testing.T) {
	unit := e18()
	amountIn := new(big.Int).Mul(big.NewInt(1000), unit)
	amountOutMin := new(big.Int).Mul(big.NewInt(16), unit)

	path := []string{testLUSD, testWBTC, testWETH}
	calldata, err := BuildExactInputCalldata(path, amountIn, amountOutMin, big.NewInt(0))
	if err != nil {
		t.Fatalf("build calldata: %v", err)
	}

	sel, params, err := DecodeCalldata(calldata)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(sel, SelectorExactInput) {
		t.Errorf("selector mismatch")
	}

	// Offset should be 128 (4 * 32 bytes of fixed params)
	offset, _ := DecodeUint256(params, 0)
	if offset.Int64() != 128 {
		t.Errorf("path offset = %d, want 128", offset.Int64())
	}

	// amountIn at word 1
	decodedAmt, _ := DecodeUint256(params, 1)
	if decodedAmt.Cmp(amountIn) != 0 {
		t.Errorf("amountIn = %s, want %s", decodedAmt, amountIn)
	}

	// Path length = 3 addresses * 20 bytes = 60
	pathLen, _ := DecodeUint256(params, 4)
	if pathLen.Int64() != 60 {
		t.Errorf("path length = %d, want 60", pathLen.Int64())
	}
}

func TestBuildExactInputCalldataPathTooShort(t *testing.T) {
	_, err := BuildExactInputCalldata(
		[]string{testLUSD},
		big.NewInt(1000),
		big.NewInt(500),
		big.NewInt(0),
	)
	if err == nil {
		t.Error("expected error for single-element path")
	}
}

func TestBuildExactOutputSingleCalldata(t *testing.T) {
	unit := e18()
	amountOut := new(big.Int).Div(unit, big.NewInt(100))
	amountInMax := new(big.Int).Mul(big.NewInt(700), unit)

	calldata, err := BuildExactOutputSingleCalldata(
		testLUSD, testWBTC, amountOut, amountInMax, big.NewInt(0), "",
	)
	if err != nil {
		t.Fatalf("build calldata: %v", err)
	}

	sel, _, err := DecodeCalldata(calldata)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(sel, SelectorExactOutputSingle) {
		t.Errorf("selector = %x, want %x", sel, SelectorExactOutputSingle)
	}
}

func TestBuildExactOutputCalldata(t *testing.T) {
	unit := e18()
	amountOut := new(big.Int).Mul(big.NewInt(16), unit)
	amountInMax := new(big.Int).Mul(big.NewInt(1000), unit)

	path := []string{testLUSD, testWBTC, testWETH}
	calldata, err := BuildExactOutputCalldata(path, amountOut, amountInMax, big.NewInt(0))
	if err != nil {
		t.Fatalf("build calldata: %v", err)
	}

	sel, _, err := DecodeCalldata(calldata)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(sel, SelectorExactOutput) {
		t.Errorf("selector = %x, want %x", sel, SelectorExactOutput)
	}
}

func TestBuildExactOutputCalldataPathTooShort(t *testing.T) {
	_, err := BuildExactOutputCalldata(
		[]string{testLUSD},
		big.NewInt(1000),
		big.NewInt(500),
		big.NewInt(0),
	)
	if err == nil {
		t.Error("expected error for single-element path")
	}
}

func TestDecodeCalldataInvalid(t *testing.T) {
	// Too short
	_, _, err := DecodeCalldata("0x0102")
	if err == nil {
		t.Error("expected error for short calldata")
	}

	// Invalid hex
	_, _, err = DecodeCalldata("0xGGGG")
	if err == nil {
		t.Error("expected error for invalid hex")
	}
}

func TestDecodeUint256OutOfBounds(t *testing.T) {
	_, err := DecodeUint256(make([]byte, 31), 0)
	if err == nil {
		t.Error("expected error for short params")
	}
}

func TestDecodeAddressOutOfBounds(t *testing.T) {
	_, err := DecodeAddress(make([]byte, 31), 0)
	if err == nil {
		t.Error("expected error for short params")
	}
}

func TestLxrDecodeAddressInvalid(t *testing.T) {
	_, err := lxrDecodeAddress("0x123")
	if err == nil {
		t.Error("expected error for short address")
	}

	_, err = lxrDecodeAddress("0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
	if err == nil {
		t.Error("expected error for invalid hex")
	}
}

func TestLxrDecodeBytes32(t *testing.T) {
	// Valid short hex (gets left-padded)
	out, err := lxrDecodeBytes32("0x01")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out[31] != 1 {
		t.Errorf("expected last byte = 1, got %d", out[31])
	}

	// Too long
	_, err = lxrDecodeBytes32("0x" + strings.Repeat("ab", 33))
	if err == nil {
		t.Error("expected error for too-long bytes32")
	}
}
