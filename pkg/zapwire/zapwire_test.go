// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package zapwire

import (
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
		// these as DepositReqSize/WithdrawReqSize/BalanceRespSize (88/88/9) and the
		// precompile mirrors them; a change here breaks that cross-module parity.
		// Deposit/Withdraw carry the FULL 32-byte injective asset id and the 32-byte
		// idempotency ref (user[16]+asset[32]+amount[8]+ref[32]=88); the asset is the
		// full id (NOT a truncated handle) so distinct assets never collide on the
		// ledger key (the native-vault-drain fix); the ref unifies the EVM-txHash
		// idempotency identity with the d-chain content-addressed seen: key.
		{"AssetIDSize", AssetIDSize, 32},
		{"RefSize", RefSize, 32},
		{"DepositReqSize", DepositReqSize, 88},
		{"WithdrawReqSize", WithdrawReqSize, 88},
		{"OpenMarketReqSize", OpenMarketReqSize, 96},
		{"BalanceReqSize", BalanceReqSize, 48},
		{"BalanceRespSize", BalanceRespSize, 9},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d (FROZEN wire frame changed)", c.name, c.got, c.want)
		}
	}
}

func TestMethodNamesFrozen(t *testing.T) {
	// Public wire names are dex_* (the D-Chain DEX namespace), per the luxd
	// service-prefix convention. Frozen against accidental drift.
	if MethodEnsureMarket != "dex_ensure_market" ||
		MethodPlace != "dex_place" ||
		MethodCancel != "dex_cancel" ||
		MethodSubmit != "dex_submit" {
		t.Fatalf("DEX method names changed: %q %q %q %q",
			MethodEnsureMarket, MethodPlace, MethodCancel, MethodSubmit)
	}
	// Custody method names (additive, FROZEN, mirrored by the proxy).
	if MethodDeposit != "dex_deposit" ||
		MethodWithdraw != "dex_withdraw" ||
		MethodOpenMarket != "dex_open_market" {
		t.Fatalf("custody method names changed: %q %q %q",
			MethodDeposit, MethodWithdraw, MethodOpenMarket)
	}
}

// TestCustodyFramesRoundTrip pins the custody frame codecs (deposit/withdraw/
// open-market/balance-resp) round-trip exactly — the bytes the proxy's atomic
// rail and the d-chain ledger exchange. The 32-byte idempotency ref MUST survive
// the round-trip: it is the field the d-chain folds into its seen: dedup key, so
// a lost/garbled ref would reopen the vault-drain replay window.
func TestCustodyFramesRoundTrip(t *testing.T) {
	// deposit / withdraw — asset is the FULL 32-byte injective id.
	mkAsset := func(b ...byte) [32]byte { var a [32]byte; copy(a[:], b); return a }
	for _, tc := range []struct {
		user   string
		asset  [32]byte
		amount uint64
		ref    [32]byte
	}{
		{"maker-a", mkAsset(0x4c, 0x55, 0x58, 0x00, 0x00, 0x00, 0x00, 0x01), 100, [32]byte{0x01, 0x02, 0xff}},
		{"taker-x", mkAsset(0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xf0, 0x0d), 1<<63 + 7, [32]byte{0xaa, 0xbb, 0xcc, 0xdd, 0xee}},
		// Full-width id whose distinguishing bytes live in the LOW 24 bytes — a
		// truncated 8-byte handle would lose them; the full id must survive intact.
		{"deep-id", mkAsset(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xab, 0xcd, 0xef), 42, [32]byte{0x09}},
		{"", [32]byte{}, 0, [32]byte{}},
	} {
		du, da, dm, dref, err := DecodeDeposit(EncodeDeposit(tc.user, tc.asset, tc.amount, tc.ref))
		if err != nil || du != tc.user || da != tc.asset || dm != tc.amount || dref != tc.ref {
			t.Fatalf("deposit round-trip (%q,%x,%d,%x) = (%q,%x,%d,%x) err=%v", tc.user, tc.asset[:8], tc.amount, tc.ref[:4], du, da[:8], dm, dref[:4], err)
		}
		wu, wa, wm, wref, werr := DecodeWithdraw(EncodeWithdraw(tc.user, tc.asset, tc.amount, tc.ref))
		if werr != nil || wu != tc.user || wa != tc.asset || wm != tc.amount || wref != tc.ref {
			t.Fatalf("withdraw round-trip (%q,%x,%d,%x) = (%q,%x,%d,%x) err=%v", tc.user, tc.asset[:8], tc.amount, tc.ref[:4], wu, wa[:8], wm, wref[:4], werr)
		}
	}

	// COLLISION RESISTANCE (the asset-key fix): two assets that share their leading
	// 8 bytes but differ in the LOW 24 bytes MUST encode to distinct asset fields.
	// The old 8-byte handle truncated them to the SAME key (the collision bug); the
	// full 32-byte id keeps them distinct on the wire (and so on the ledger key).
	sharedHi := mkAsset(0, 0, 0, 0, 0, 0, 0, 0, 0x11) // low byte 0x11
	sharedHiAlt := mkAsset(0, 0, 0, 0, 0, 0, 0, 0, 0x22)
	_, a1, _, _, _ := DecodeDeposit(EncodeDeposit("u", sharedHi, 1, [32]byte{0x01}))
	_, a2, _, _, _ := DecodeDeposit(EncodeDeposit("u", sharedHiAlt, 1, [32]byte{0x01}))
	if a1 == a2 {
		t.Fatal("two assets sharing leading 8 bytes decoded to the SAME id — truncation collision (native-vault-drain class)")
	}

	// The ref MUST change the encoded bytes: two custody frames identical in
	// (user,asset,amount) but differing in ref are DIFFERENT bytes — that is the
	// whole point (distinct d-chain txID -> distinct seen: key -> no replay).
	a := EncodeWithdraw("alice", mkAsset(7), 1000, [32]byte{0x01})
	b := EncodeWithdraw("alice", mkAsset(7), 1000, [32]byte{0x02})
	if string(a) == string(b) {
		t.Fatal("withdraw frames with distinct refs encoded identically — ref does not enter the wire (vault-drain dedup would collide)")
	}
	// open-market
	pool := [32]byte{0xde, 0xad}
	baseA, quoteA := mkAsset(11), mkAsset(22)
	op, ob, oq, oerr := DecodeOpenMarket(EncodeOpenMarket(pool, baseA, quoteA))
	if oerr != nil || op != pool || ob != baseA || oq != quoteA {
		t.Fatalf("open-market round-trip = (%x,%x,%x) err=%v", op[:2], ob[:2], oq[:2], oerr)
	}
	// balance-resp
	bs, ba, berr := DecodeBalanceResp(EncodeBalanceResp(StatusPlaced, 12345))
	if berr != nil || bs != StatusPlaced || ba != 12345 {
		t.Fatalf("balance-resp round-trip = (%d,%d) err=%v", bs, ba, berr)
	}
}

// TestExactIntegerWire pins that the value-bearing wire fields carry EXACT
// unsigned integers with no IEEE-754 float64 anywhere — a full-range uint64
// (including >2^53 and the top bit set) round-trips byte-for-byte, which is the
// property that makes the state transition architecture-independent.
func TestExactIntegerWire(t *testing.T) {
	var poolID [32]byte
	for i := range poolID {
		poolID[i] = byte(i + 1)
	}
	// price = 123.5 in ×1e8 fixed-point; size = huge base-unit count > 2^53.
	const wantPrice = uint64(12_350_000_000)
	const wantSize = uint64(0xFFFFFFFFFFFFFFF0) // near uint64 max, unrepresentable as an exact float64
	req := EncodeSubmit(poolID, SideBuy, false, wantPrice, wantSize, "alice")
	if len(req) != SubmitReqSize {
		t.Fatalf("EncodeSubmit len = %d, want %d", len(req), SubmitReqSize)
	}
	gotPool, side, isMarket, limit, size, user, err := DecodeSubmit(req)
	if err != nil {
		t.Fatalf("DecodeSubmit: %v", err)
	}
	if gotPool != poolID || side != SideBuy || isMarket || limit != wantPrice || size != wantSize || user != "alice" {
		t.Fatalf("DecodeSubmit mismatch: pool=%x side=%d market=%v limit=%d size=%d user=%q",
			gotPool, side, isMarket, limit, size, user)
	}
	// Place round-trip with the same exact integers.
	pReq := EncodePlace(poolID, SideSell, wantPrice, wantSize, "bob")
	gp, ps, pp, psz, pu, perr := DecodePlace(pReq)
	if perr != nil || gp != poolID || ps != SideSell || pp != wantPrice || psz != wantSize || pu != "bob" {
		t.Fatalf("DecodePlace mismatch: %x %d %d %d %q err=%v", gp, ps, pp, psz, pu, perr)
	}
}

func TestFillsRoundTrip(t *testing.T) {
	in := []Fill{
		{Price: 100, Size: 2, TakerSide: SideBuy},
		{Price: 10_150_000_000, Size: 0xFFFFFFFFFFFFFFF0, TakerSide: SideBuy},
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
