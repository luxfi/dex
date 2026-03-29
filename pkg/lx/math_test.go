// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"math/big"
	"testing"
)

// q96test = 2^96
var q96test = new(big.Int).Lsh(big.NewInt(1), 96)

func TestFullMath_MulDiv(t *testing.T) {
	tests := []struct {
		name   string
		a, b   *big.Int
		denom  *big.Int
		expect *big.Int
	}{
		{
			name:   "500*2/3=333 floor",
			a:      big.NewInt(500),
			b:      big.NewInt(2),
			denom:  big.NewInt(3),
			expect: big.NewInt(333),
		},
		{
			name:   "1*1/1=1",
			a:      big.NewInt(1),
			b:      big.NewInt(1),
			denom:  big.NewInt(1),
			expect: big.NewInt(1),
		},
		{
			name:   "0*100/7=0",
			a:      big.NewInt(0),
			b:      big.NewInt(100),
			denom:  big.NewInt(7),
			expect: big.NewInt(0),
		},
		{
			name:   "Q96*Q96/1=Q96^2",
			a:      q96test,
			b:      q96test,
			denom:  big.NewInt(1),
			expect: new(big.Int).Mul(q96test, q96test),
		},
		{
			name:   "Q96*Q96/Q96=Q96",
			a:      q96test,
			b:      q96test,
			denom:  q96test,
			expect: q96test,
		},
		{
			name:   "exact division 100*10/5=200",
			a:      big.NewInt(100),
			b:      big.NewInt(10),
			denom:  big.NewInt(5),
			expect: big.NewInt(200),
		},
		{
			name:   "7*11/3=25 floor",
			a:      big.NewInt(7),
			b:      big.NewInt(11),
			denom:  big.NewInt(3),
			expect: big.NewInt(25),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MulDiv(tt.a, tt.b, tt.denom)
			if err != nil {
				t.Fatalf("MulDiv(%s, %s, %s) unexpected error: %v", tt.a, tt.b, tt.denom, err)
			}
			if got.Cmp(tt.expect) != 0 {
				t.Errorf("MulDiv(%s, %s, %s) = %s, want %s", tt.a, tt.b, tt.denom, got, tt.expect)
			}
		})
	}
}

func TestFullMath_MulDivRoundingUp(t *testing.T) {
	tests := []struct {
		name   string
		a, b   *big.Int
		denom  *big.Int
		expect *big.Int
	}{
		{
			name:   "500*2/3=334 ceil",
			a:      big.NewInt(500),
			b:      big.NewInt(2),
			denom:  big.NewInt(3),
			expect: big.NewInt(334),
		},
		{
			name:   "exact 100*10/5=200 no rounding",
			a:      big.NewInt(100),
			b:      big.NewInt(10),
			denom:  big.NewInt(5),
			expect: big.NewInt(200),
		},
		{
			name:   "7*11/3=26 ceil",
			a:      big.NewInt(7),
			b:      big.NewInt(11),
			denom:  big.NewInt(3),
			expect: big.NewInt(26),
		},
		{
			name:   "1*1/2=1 ceil",
			a:      big.NewInt(1),
			b:      big.NewInt(1),
			denom:  big.NewInt(2),
			expect: big.NewInt(1),
		},
		{
			name:   "0*100/7=0",
			a:      big.NewInt(0),
			b:      big.NewInt(100),
			denom:  big.NewInt(7),
			expect: big.NewInt(0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MulDivRoundingUp(tt.a, tt.b, tt.denom)
			if err != nil {
				t.Fatalf("MulDivRoundingUp(%s, %s, %s) unexpected error: %v", tt.a, tt.b, tt.denom, err)
			}
			if got.Cmp(tt.expect) != 0 {
				t.Errorf("MulDivRoundingUp(%s, %s, %s) = %s, want %s", tt.a, tt.b, tt.denom, got, tt.expect)
			}
		})
	}
}

func TestFullMath_DivRoundingUp(t *testing.T) {
	tests := []struct {
		name   string
		a, b   *big.Int
		expect *big.Int
	}{
		{"10/3=4", big.NewInt(10), big.NewInt(3), big.NewInt(4)},
		{"9/3=3", big.NewInt(9), big.NewInt(3), big.NewInt(3)},
		{"1/2=1", big.NewInt(1), big.NewInt(2), big.NewInt(1)},
		{"0/5=0", big.NewInt(0), big.NewInt(5), big.NewInt(0)},
		{"7/1=7", big.NewInt(7), big.NewInt(1), big.NewInt(7)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DivRoundingUp(tt.a, tt.b)
			if err != nil {
				t.Fatalf("DivRoundingUp(%s, %s) unexpected error: %v", tt.a, tt.b, err)
			}
			if got.Cmp(tt.expect) != 0 {
				t.Errorf("DivRoundingUp(%s, %s) = %s, want %s", tt.a, tt.b, got, tt.expect)
			}
		})
	}
}

func TestFullMath_SimpleMulDiv(t *testing.T) {
	// SimpleMulDiv should produce same results as MulDiv
	got, err := SimpleMulDiv(big.NewInt(500), big.NewInt(2), big.NewInt(3))
	if err != nil {
		t.Fatalf("SimpleMulDiv(500, 2, 3) unexpected error: %v", err)
	}
	if got.Cmp(big.NewInt(333)) != 0 {
		t.Errorf("SimpleMulDiv(500, 2, 3) = %s, want 333", got)
	}

	got, err = SimpleMulDiv(q96test, q96test, big.NewInt(1))
	if err != nil {
		t.Fatalf("SimpleMulDiv(Q96, Q96, 1) unexpected error: %v", err)
	}
	expect := new(big.Int).Mul(q96test, q96test)
	if got.Cmp(expect) != 0 {
		t.Errorf("SimpleMulDiv(Q96, Q96, 1) = %s, want %s", got, expect)
	}
}

func TestFullMath_DivByZeroErrors(t *testing.T) {
	t.Run("MulDiv/0", func(t *testing.T) {
		_, err := MulDiv(big.NewInt(1), big.NewInt(1), big.NewInt(0))
		if err != ErrDivisionByZero {
			t.Fatalf("expected ErrDivisionByZero, got %v", err)
		}
	})
	t.Run("MulDivRoundingUp/0", func(t *testing.T) {
		_, err := MulDivRoundingUp(big.NewInt(1), big.NewInt(1), big.NewInt(0))
		if err != ErrDivisionByZero {
			t.Fatalf("expected ErrDivisionByZero, got %v", err)
		}
	})
	t.Run("DivRoundingUp/0", func(t *testing.T) {
		_, err := DivRoundingUp(big.NewInt(1), big.NewInt(0))
		if err != ErrDivisionByZero {
			t.Fatalf("expected ErrDivisionByZero, got %v", err)
		}
	})
	t.Run("SimpleMulDiv/0", func(t *testing.T) {
		_, err := SimpleMulDiv(big.NewInt(1), big.NewInt(1), big.NewInt(0))
		if err != ErrDivisionByZero {
			t.Fatalf("expected ErrDivisionByZero, got %v", err)
		}
	})
}

func TestFullMath_Uint256Overflow(t *testing.T) {
	// Result exceeding 2^256-1 must return ErrUint256Overflow.
	// MaxUint256 * 2 / 1 = 2 * MaxUint256 which is > 2^256-1.
	_, err := MulDiv(MaxUint256, big.NewInt(2), big.NewInt(1))
	if err != ErrUint256Overflow {
		t.Fatalf("MulDiv(MaxUint256, 2, 1) expected ErrUint256Overflow, got %v", err)
	}

	_, err = MulDivRoundingUp(MaxUint256, big.NewInt(2), big.NewInt(1))
	if err != ErrUint256Overflow {
		t.Fatalf("MulDivRoundingUp(MaxUint256, 2, 1) expected ErrUint256Overflow, got %v", err)
	}

	_, err = SimpleMulDiv(MaxUint256, big.NewInt(2), big.NewInt(1))
	if err != ErrUint256Overflow {
		t.Fatalf("SimpleMulDiv(MaxUint256, 2, 1) expected ErrUint256Overflow, got %v", err)
	}

	// Rounding up past MaxUint256: (MaxUint256 * 1 / 1) + rounding should be fine (exact),
	// but (MaxUint256 * 3 / 2) ceil should overflow.
	_, err = MulDivRoundingUp(MaxUint256, big.NewInt(3), big.NewInt(2))
	if err != ErrUint256Overflow {
		t.Fatalf("MulDivRoundingUp(MaxUint256, 3, 2) expected ErrUint256Overflow, got %v", err)
	}
}
