// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package zapwire

import (
	"math"
	"testing"
)

// TestFrameSizesFrozen pins the wire frame sizes. These bytes are shared
// verbatim with dex/pkg/api (server), the chains/dexvm proxy, and the EVM
// precompile. A change here is a wire-breaking change and must fail loudly.
func TestFrameSizesFrozen(t *testing.T) {
	cases := []struct {
		name string
		got  int
		want int
	}{
		{"PoolIDSize", PoolIDSize, 32},
		{"UserSize", UserSize, 16},
		{"EnsureMarketReqSize", EnsureMarketReqSize, 32},
		{"PlaceReqSize", PlaceReqSize, 65},
		{"CancelReqSize", CancelReqSize, 40},
		{"SubmitReqSize", SubmitReqSize, 66},
		{"AckSize", AckSize, 24},
		{"FillWireSize", FillWireSize, 17},
		// Custody frames (additive, FROZEN). The chains/dexvm proxy re-defines
		// these as DepositReqSize/WithdrawReqSize/BalanceRespSize (32/32/9) and the
		// precompile mirrors them; a change here breaks that cross-module parity.
		{"AssetIDSize", AssetIDSize, 8},
		{"DepositReqSize", DepositReqSize, 32},
		{"WithdrawReqSize", WithdrawReqSize, 32},
		{"OpenMarketReqSize", OpenMarketReqSize, 48},
		{"BalanceRespSize", BalanceRespSize, 9},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d (FROZEN wire frame changed)", c.name, c.got, c.want)
		}
	}
}

func TestMethodNamesFrozen(t *testing.T) {
	if MethodEnsureMarket != "clob_ensure_market" ||
		MethodPlace != "clob_place" ||
		MethodCancel != "clob_cancel" ||
		MethodSubmit != "clob_submit" {
		t.Fatalf("CLOB method names changed: %q %q %q %q",
			MethodEnsureMarket, MethodPlace, MethodCancel, MethodSubmit)
	}
	// Custody method names (additive, FROZEN, mirrored by the proxy).
	if MethodDeposit != "clob_deposit" ||
		MethodWithdraw != "clob_withdraw" ||
		MethodOpenMarket != "clob_open_market" {
		t.Fatalf("custody method names changed: %q %q %q",
			MethodDeposit, MethodWithdraw, MethodOpenMarket)
	}
}

// TestCustodyFramesRoundTrip pins the custody frame codecs (deposit/withdraw/
// open-market/balance-resp) round-trip exactly — the bytes the proxy's atomic
// rail and the d-chain ledger exchange.
func TestCustodyFramesRoundTrip(t *testing.T) {
	// deposit / withdraw
	for _, tc := range []struct {
		user   string
		asset  uint64
		amount uint64
	}{
		{"maker-a", 0x4c5558_00000001, 100},
		{"taker-x", 0xdeadbeefcafef00d, 1<<63 + 7},
		{"", 0, 0},
	} {
		du, da, dm, err := DecodeDeposit(EncodeDeposit(tc.user, tc.asset, tc.amount))
		if err != nil || du != tc.user || da != tc.asset || dm != tc.amount {
			t.Fatalf("deposit round-trip (%q,%#x,%d) = (%q,%#x,%d) err=%v", tc.user, tc.asset, tc.amount, du, da, dm, err)
		}
		wu, wa, wm, werr := DecodeWithdraw(EncodeWithdraw(tc.user, tc.asset, tc.amount))
		if werr != nil || wu != tc.user || wa != tc.asset || wm != tc.amount {
			t.Fatalf("withdraw round-trip (%q,%#x,%d) = (%q,%#x,%d) err=%v", tc.user, tc.asset, tc.amount, wu, wa, wm, werr)
		}
	}
	// open-market
	pool := [32]byte{0xde, 0xad}
	op, ob, oq, oerr := DecodeOpenMarket(EncodeOpenMarket(pool, 11, 22))
	if oerr != nil || op != pool || ob != 11 || oq != 22 {
		t.Fatalf("open-market round-trip = (%x,%d,%d) err=%v", op[:2], ob, oq, oerr)
	}
	// balance-resp
	bs, ba, berr := DecodeBalanceResp(EncodeBalanceResp(StatusPlaced, 12345))
	if berr != nil || bs != StatusPlaced || ba != 12345 {
		t.Fatalf("balance-resp round-trip = (%d,%d) err=%v", bs, ba, berr)
	}
}

func TestFloat64RoundTrip(t *testing.T) {
	for _, f := range []float64{0, 1, 0.5, 1e18, 1.0 / 3.0, math.MaxFloat64, math.SmallestNonzeroFloat64} {
		b := make([]byte, 8)
		PutFloat64(b, f)
		if got := Float64(b); got != f {
			t.Errorf("Float64(PutFloat64(%v)) = %v", f, got)
		}
	}
}

func TestSubmitRoundTrip(t *testing.T) {
	var poolID [32]byte
	for i := range poolID {
		poolID[i] = byte(i + 1)
	}
	req := EncodeSubmit(poolID, SideBuy, false, 123.5, 10.25, "alice")
	if len(req) != SubmitReqSize {
		t.Fatalf("EncodeSubmit len = %d, want %d", len(req), SubmitReqSize)
	}
	gotPool, side, isMarket, limit, size, user, err := DecodeSubmit(req)
	if err != nil {
		t.Fatalf("DecodeSubmit: %v", err)
	}
	if gotPool != poolID || side != SideBuy || isMarket || limit != 123.5 || size != 10.25 || user != "alice" {
		t.Fatalf("DecodeSubmit mismatch: pool=%x side=%d market=%v limit=%v size=%v user=%q",
			gotPool, side, isMarket, limit, size, user)
	}
}

func TestFillsRoundTrip(t *testing.T) {
	in := []Fill{
		{Price: 100, Size: 2, TakerSide: SideBuy},
		{Price: 101.5, Size: 0.5, TakerSide: SideBuy},
	}
	resp := EncodeFills(in)
	if len(resp) != 4+len(in)*FillWireSize {
		t.Fatalf("EncodeFills len = %d, want %d", len(resp), 4+len(in)*FillWireSize)
	}
	out, err := DecodeFills(resp)
	if err != nil {
		t.Fatalf("DecodeFills: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("DecodeFills len = %d, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("fill %d = %+v, want %+v", i, out[i], in[i])
		}
	}
	// Truncated frame must error, not panic.
	if _, err := DecodeFills(resp[:len(resp)-1]); err == nil {
		t.Error("DecodeFills(truncated) = nil error, want error")
	}
}

func TestAckRoundTrip(t *testing.T) {
	resp := EncodeAck(42, StatusPlaced, 7)
	if len(resp) != AckSize {
		t.Fatalf("EncodeAck len = %d, want %d", len(resp), AckSize)
	}
	id, status, err := DecodeAck(resp)
	if err != nil {
		t.Fatalf("DecodeAck: %v", err)
	}
	if id != 42 || status != StatusPlaced {
		t.Fatalf("DecodeAck = (%d,%d), want (42,%d)", id, status, StatusPlaced)
	}
}
