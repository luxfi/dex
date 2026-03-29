// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/big"
	"testing"
)

func TestAddDelta(t *testing.T) {
	tests := []struct {
		name    string
		x, y    int64
		want    int64
		wantErr error
	}{
		{"add positive", 100, 50, 150, nil},
		{"remove partial", 100, -50, 50, nil},
		{"add zero", 100, 0, 100, nil},
		{"remove all", 100, -100, 0, nil},
		{"underflow", 50, -100, 0, ErrLiquidityUnderflow},
		{"zero minus one", 0, -1, 0, ErrLiquidityUnderflow},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := AddDelta(big.NewInt(tt.x), big.NewInt(tt.y))
			if err != tt.wantErr {
				t.Fatalf("AddDelta(%d, %d) error = %v, wantErr %v", tt.x, tt.y, err, tt.wantErr)
			}
			if err == nil && got.Int64() != tt.want {
				t.Errorf("AddDelta(%d, %d) = %d, want %d", tt.x, tt.y, got.Int64(), tt.want)
			}
		})
	}
}

func TestAddDeltaOverflow(t *testing.T) {
	// uint128Max + 1 should overflow
	x := new(big.Int).Set(uint128Max)
	y := big.NewInt(1)
	_, err := AddDelta(x, y)
	if err != ErrLiquidityOverflow {
		t.Fatalf("AddDelta(uint128Max, 1) error = %v, want ErrLiquidityOverflow", err)
	}

	// uint128Max + 0 should be fine
	result, err := AddDelta(x, big.NewInt(0))
	if err != nil {
		t.Fatalf("AddDelta(uint128Max, 0) error = %v", err)
	}
	if result.Cmp(uint128Max) != 0 {
		t.Error("AddDelta(uint128Max, 0) != uint128Max")
	}
}

func TestAddDeltaLargeValues(t *testing.T) {
	// Large positive: half of uint128Max + half
	half := new(big.Int).Rsh(uint128Max, 1)
	result, err := AddDelta(half, half)
	if err != nil {
		t.Fatalf("AddDelta(half, half) error = %v", err)
	}
	expected := new(big.Int).Add(half, half)
	if result.Cmp(expected) != 0 {
		t.Errorf("AddDelta(half, half) = %s, want %s", result, expected)
	}

	// Large removal
	result, err = AddDelta(uint128Max, new(big.Int).Neg(half))
	if err != nil {
		t.Fatalf("AddDelta(uint128Max, -half) error = %v", err)
	}
	expected = new(big.Int).Sub(uint128Max, half)
	if result.Cmp(expected) != 0 {
		t.Errorf("AddDelta(uint128Max, -half) = %s, want %s", result, expected)
	}
}
