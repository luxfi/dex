// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import "unsafe"

// Go mirrors of the public C ABI types in
// luxcpp/cuda/include/lux/cuda/dex_swap.h. Layouts MUST stay byte-equal
// with the C structs — the same memory is reinterpreted on the cgo
// boundary in orderbook_gpu_cuda.go.
//
// These types are deliberately a separate flat representation from
// orderbook.go::Order (which has heap pointers, linked-list nodes, and
// status fields the CUDA kernel doesn't model). Use this surface when
// you want to drive the CUDA matching kernel directly — for parity
// testing, batch backtests, or future architectures that snapshot the
// book into a flat array.

// DEXPrice is fixed-point Q64.64. Maps to DEXPrice in dex_swap.h.
type DEXPrice struct {
	Integer  uint64
	Fraction uint64
}

// DEXSide / DEXOrderType / DEXOrderStatus values mirror the kernel enums.
const (
	DEXSideBid uint8 = 0
	DEXSideAsk uint8 = 1

	DEXOrderLimit  uint8 = 0
	DEXOrderMarket uint8 = 1
	DEXOrderIOC    uint8 = 2
	DEXOrderFOK    uint8 = 3

	DEXStatusOpen      uint8 = 0
	DEXStatusPartial   uint8 = 1
	DEXStatusFilled    uint8 = 2
	DEXStatusCancelled uint8 = 3
)

// DEXOrder is the 64-byte cache-line-aligned order record. Layout is
// byte-equal with DEXOrder in dex_swap.h and Order in dex_swap.cu.
//
// _pad is a named padding field (not a blank `_`) so it is addressable
// and we can normalize it to zero after any data path that may leave
// the bytes uninitialised. Same for DEXTrade.
type DEXOrder struct {
	OrderID uint64 // 0..8
	// UserID is a compact matcher/STP handle only. It is intentionally not
	// collision-resistant and MUST NOT be used for balances, custody, settlement,
	// withdraws, imports, exports, or liability accounting. Settlement identity is
	// resolved exclusively through the orderuser index to the full AccountID/userKey.
	UserID    uint64   // 8..16
	Price     DEXPrice // 16..32
	Quantity  uint64   // 32..40
	Remaining uint64   // 40..48
	Timestamp uint64   // 48..56
	Side      uint8    // 56
	Type      uint8    // 57
	Status    uint8    // 58
	_pad      [5]uint8 // 59..64
}

// DEXTrade is the 80-byte trade record produced by the matching kernel.
// Layout is byte-equal with DEXTrade in dex_swap.h.
type DEXTrade struct {
	TradeID      uint64   // 0..8
	MakerOrderID uint64   // 8..16
	TakerOrderID uint64   // 16..24
	MakerUserID  uint64   // 24..32
	TakerUserID  uint64   // 32..40
	Price        DEXPrice // 40..56
	Quantity     uint64   // 56..64
	Timestamp    uint64   // 64..72
	TakerSide    uint8    // 72
	_pad         [7]uint8 // 73..80
}

// ClearPadding zeros the trailing padding bytes. Called by CPU oracle
// output paths and after cgo device→host copies so structural
// comparisons of DEXOrder / DEXTrade values are deterministic.
func (o *DEXOrder) ClearPadding() { o._pad = [5]uint8{} }
func (t *DEXTrade) ClearPadding() { t._pad = [7]uint8{} }

// EqualFields compares every named field of two DEXTrade values,
// excluding padding bytes. Use this in tests and structural assertions
// instead of `a == b` so a kernel that fails to zero its trailing pad
// bytes does not cause a false-negative.
func (t DEXTrade) EqualFields(o DEXTrade) bool {
	return t.TradeID == o.TradeID &&
		t.MakerOrderID == o.MakerOrderID &&
		t.TakerOrderID == o.TakerOrderID &&
		t.MakerUserID == o.MakerUserID &&
		t.TakerUserID == o.TakerUserID &&
		t.Price == o.Price &&
		t.Quantity == o.Quantity &&
		t.Timestamp == o.Timestamp &&
		t.TakerSide == o.TakerSide
}

// EqualFields compares every named field of two DEXOrder values,
// excluding padding bytes.
func (o DEXOrder) EqualFields(other DEXOrder) bool {
	return o.OrderID == other.OrderID &&
		o.UserID == other.UserID &&
		o.Price == other.Price &&
		o.Quantity == other.Quantity &&
		o.Remaining == other.Remaining &&
		o.Timestamp == other.Timestamp &&
		o.Side == other.Side &&
		o.Type == other.Type &&
		o.Status == other.Status
}

// init runs the Go-side ABI mirror of the C-side _Static_assert checks
// in dex_swap.h. If a refactor reorders a field or changes a type, the
// process panics at startup — the same compile-time guarantee the C
// macros give for the C/CUDA side.
func init() {
	assertDEXABI()
}

func assertDEXABI() {
	if got := unsafe.Sizeof(DEXPrice{}); got != 16 {
		panic("lx: DEXPrice size drifted: " + sizeStr(uintptr(got)) + " != 16")
	}
	if got := unsafe.Offsetof(DEXPrice{}.Integer); got != 0 {
		panic("lx: DEXPrice.Integer offset drift")
	}
	if got := unsafe.Offsetof(DEXPrice{}.Fraction); got != 8 {
		panic("lx: DEXPrice.Fraction offset drift")
	}

	if got := unsafe.Sizeof(DEXOrder{}); got != 64 {
		panic("lx: DEXOrder size drifted: " + sizeStr(uintptr(got)) + " != 64")
	}
	if unsafe.Offsetof(DEXOrder{}.OrderID) != 0 ||
		unsafe.Offsetof(DEXOrder{}.UserID) != 8 ||
		unsafe.Offsetof(DEXOrder{}.Price) != 16 ||
		unsafe.Offsetof(DEXOrder{}.Quantity) != 32 ||
		unsafe.Offsetof(DEXOrder{}.Remaining) != 40 ||
		unsafe.Offsetof(DEXOrder{}.Timestamp) != 48 ||
		unsafe.Offsetof(DEXOrder{}.Side) != 56 ||
		unsafe.Offsetof(DEXOrder{}.Type) != 57 ||
		unsafe.Offsetof(DEXOrder{}.Status) != 58 {
		panic("lx: DEXOrder field offset drift")
	}

	if got := unsafe.Sizeof(DEXTrade{}); got != 80 {
		panic("lx: DEXTrade size drifted: " + sizeStr(uintptr(got)) + " != 80")
	}
	if unsafe.Offsetof(DEXTrade{}.TradeID) != 0 ||
		unsafe.Offsetof(DEXTrade{}.MakerOrderID) != 8 ||
		unsafe.Offsetof(DEXTrade{}.TakerOrderID) != 16 ||
		unsafe.Offsetof(DEXTrade{}.MakerUserID) != 24 ||
		unsafe.Offsetof(DEXTrade{}.TakerUserID) != 32 ||
		unsafe.Offsetof(DEXTrade{}.Price) != 40 ||
		unsafe.Offsetof(DEXTrade{}.Quantity) != 56 ||
		unsafe.Offsetof(DEXTrade{}.Timestamp) != 64 ||
		unsafe.Offsetof(DEXTrade{}.TakerSide) != 72 {
		panic("lx: DEXTrade field offset drift")
	}

	if got := unsafe.Sizeof(ReservePair{}); got != 16 {
		panic("lx: ReservePair size drifted: " + sizeStr(uintptr(got)) + " != 16")
	}
	if unsafe.Offsetof(ReservePair{}.ReserveX) != 0 ||
		unsafe.Offsetof(ReservePair{}.ReserveY) != 8 {
		panic("lx: ReservePair field offset drift")
	}
}

// sizeStr formats a uintptr as a decimal string without pulling in fmt.
// Kept tiny so the init path stays allocation-free in the happy case.
func sizeStr(n uintptr) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
