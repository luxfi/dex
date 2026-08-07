// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"bytes"
	"errors"
	"math/big"
	"math/rand"
	"testing"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
)

var (
	testSwapper   = common.Address{0x11}
	testRecipient = common.Address{0x22}
	testReactor   = common.Address{0x99, 0x99}
	testMarket    = ids.ID{0xAA}
	testUSDC      = ids.ID{0xBB}
	testLUX       = ids.ID{0xCC}
)

func amount(v uint64) [32]byte {
	var b [32]byte
	new(big.Int).SetUint64(v).FillBytes(b[:])
	return b
}

func testOrder() Order {
	return Order{
		Swapper:   testSwapper,
		Nonce:     123,
		Market:    testMarket,
		Side:      SideBuy,
		Input:     AssetAmount{Asset: testUSDC, Amount: amount(100)},
		MinOutput: AssetAmount{Asset: testLUX, Amount: amount(497)},
		Deadline:  1000,
		Recipient: testRecipient,
	}
}

func testBinding() Binding {
	return Binding{NetworkID: 1, CChainID: ids.ID{0x0C}, Reactor: testReactor}
}

// recoverAs is a signature verifier that always recovers a fixed address, so tests
// can drive the swapper check without a curve implementation.
type recoverAs struct {
	addr common.Address
	err  error
}

func (r recoverAs) Recover(ids.ID, []byte) (common.Address, error) { return r.addr, r.err }

func orderWitness(o Order) []byte { return append(o.Encode(), 0xDE, 0xAD) }

func okContext() OrderContext {
	return OrderContext{
		Binding:   testBinding(),
		BlockTime: 500,
		Signer:    recoverAs{addr: testSwapper},
		Nonces:    NewNonces(),
	}
}

func TestOrderRoundTrip(t *testing.T) {
	o := testOrder()
	b := o.Encode()
	if len(b) != OrderEncodedLen {
		t.Fatalf("encoded %d bytes, layout says %d", len(b), OrderEncodedLen)
	}
	got, err := DecodeOrder(b)
	if err != nil {
		t.Fatal(err)
	}
	if got != o {
		t.Fatalf("round trip changed the order:\n got %+v\nwant %+v", got, o)
	}
}

// A longer buffer must be REFUSED, not truncated. Accepting a suffix would let two
// distinct wire messages decode to the same order while carrying different signed
// bytes.
func TestOrderDecodeRefusesWrongWidth(t *testing.T) {
	o := testOrder()
	for _, b := range [][]byte{
		o.Encode()[:OrderEncodedLen-1],
		append(o.Encode(), 0x00),
		nil,
	} {
		if _, err := DecodeOrder(b); !errors.Is(err, ErrOrderWidth) {
			t.Fatalf("width %d: got %v, want ErrOrderWidth", len(b), err)
		}
	}
}

// THE HEADLINE SECURITY PROPERTY. 0x9999 runs on every EVM sharing the one D venue.
// If the chain binding were beside the signature rather than inside the hash, one
// signed order would settle on all of them and the trader would pay N times for the
// one trade they authorized. Changing ANY component of the binding must change the
// commitment.
func TestOrderCommitmentBindsChainIdentity(t *testing.T) {
	o := testOrder()
	base := testBinding()
	want := o.Commitment(base)

	mutations := map[string]func(*Binding){
		"NetworkID": func(b *Binding) { b.NetworkID++ },
		"CChainID":  func(b *Binding) { b.CChainID[31] ^= 1 },
		"Reactor":   func(b *Binding) { b.Reactor[19] ^= 1 },
	}
	for name, mutate := range mutations {
		b := base
		mutate(&b)
		if o.Commitment(b) == want {
			t.Fatalf("%s: commitment unchanged — this order replays across chains", name)
		}
	}
}

// The encoding is injective, so mutating any field must move the commitment.
func TestOrderCommitmentInjective(t *testing.T) {
	b := testBinding()
	seen := map[ids.ID]string{}
	mutations := map[string]func(*Order){
		"Swapper":   func(o *Order) { o.Swapper[19] ^= 1 },
		"Nonce":     func(o *Order) { o.Nonce++ },
		"Market":    func(o *Order) { o.Market[31] ^= 1 },
		"Side":      func(o *Order) { o.Side = SideSell },
		"InAsset":   func(o *Order) { o.Input.Asset[31] ^= 1 },
		"InAmount":  func(o *Order) { o.Input.Amount[31] ^= 1 },
		"OutAsset":  func(o *Order) { o.MinOutput.Asset[31] ^= 1 },
		"OutAmount": func(o *Order) { o.MinOutput.Amount[31] ^= 1 },
		"Deadline":  func(o *Order) { o.Deadline++ },
		"Recipient": func(o *Order) { o.Recipient[19] ^= 1 },
	}
	base := testOrder()
	seen[base.Commitment(b)] = "base"
	for name, mutate := range mutations {
		o := base
		mutate(&o)
		c := o.Commitment(b)
		if prev, dup := seen[c]; dup {
			t.Fatalf("%s collides with %s", name, prev)
		}
		seen[c] = name
	}
}

// THE BUG THIS EXISTS TO MAKE IMPOSSIBLE. D's book matches by price-time priority
// across every trader, so one trader's orders settle out of signing order. A
// sequential nonce would reject the low nonce arriving second and then block that
// trader forever. Consuming 500 first and 3 second must both succeed.
func TestNoncesAreUnordered(t *testing.T) {
	n := NewNonces()
	for _, nonce := range []uint64{500, 3, 1 << 20, 0, 255, 256} {
		if err := n.Consume(testSwapper, nonce); err != nil {
			t.Fatalf("nonce %d refused out of order: %v", nonce, err)
		}
	}
	for _, nonce := range []uint64{500, 3, 1 << 20, 0, 255, 256} {
		if !n.Used(testSwapper, nonce) {
			t.Fatalf("nonce %d not marked used", nonce)
		}
	}
}

// Neighbouring positions must not alias — the classic off-by-one in a bitmap is a
// nonce that silently consumes its neighbour, which would let a replay through.
func TestNoncesDoNotAlias(t *testing.T) {
	n := NewNonces()
	if err := n.Consume(testSwapper, 64); err != nil {
		t.Fatal(err)
	}
	for _, other := range []uint64{63, 65, 0, 128, 192, 320} {
		if n.Used(testSwapper, other) {
			t.Fatalf("consuming 64 also consumed %d", other)
		}
	}
	// And nonces are per-swapper.
	if n.Used(common.Address{0x77}, 64) {
		t.Fatal("one swapper's nonce leaked into another's bitmap")
	}
}

func TestNonceReplayRefused(t *testing.T) {
	n := NewNonces()
	if err := n.Consume(testSwapper, 7); err != nil {
		t.Fatal(err)
	}
	if err := n.Consume(testSwapper, 7); !errors.Is(err, ErrNonceUsed) {
		t.Fatalf("replay: got %v, want ErrNonceUsed", err)
	}
}

// Random positions, to catch word/bit arithmetic that works on the cases a human
// picks and fails elsewhere.
func TestNonceBitmapExhaustive(t *testing.T) {
	n := NewNonces()
	rng := rand.New(rand.NewSource(7))
	used := map[uint64]bool{}
	for i := 0; i < 2000; i++ {
		nonce := rng.Uint64() % 5000
		err := n.Consume(testSwapper, nonce)
		if used[nonce] {
			if !errors.Is(err, ErrNonceUsed) {
				t.Fatalf("nonce %d consumed twice without error", nonce)
			}
			continue
		}
		if err != nil {
			t.Fatalf("nonce %d: %v", nonce, err)
		}
		used[nonce] = true
	}
	for nonce := uint64(0); nonce < 5000; nonce++ {
		if n.Used(testSwapper, nonce) != used[nonce] {
			t.Fatalf("nonce %d: bitmap says %v, want %v", nonce, n.Used(testSwapper, nonce), used[nonce])
		}
	}
}

func TestVerifyOrder(t *testing.T) {
	o := testOrder()
	v, err := VerifyOrder(orderWitness(o), okContext())
	if err != nil {
		t.Fatal(err)
	}
	if v.Order() != o {
		t.Fatal("verified order differs from the signed one")
	}
	if v.OrderID() != o.Commitment(testBinding()) {
		t.Fatal("OrderID is not the chain-bound commitment")
	}
}

// The deadline is checked against the BLOCK's time, never a clock. A wall-clock read
// would make the same order valid on one validator and expired on another.
func TestVerifyOrderDeadlineUsesBlockTime(t *testing.T) {
	o := testOrder() // Deadline 1000
	ctx := okContext()
	ctx.BlockTime = 1001
	if _, err := VerifyOrder(orderWitness(o), ctx); !errors.Is(err, ErrOrderExpired) {
		t.Fatalf("got %v, want ErrOrderExpired", err)
	}
	// Exactly at the deadline is still valid.
	ctx.BlockTime = 1000
	if _, err := VerifyOrder(orderWitness(o), ctx); err != nil {
		t.Fatalf("order at its deadline was refused: %v", err)
	}
}

func TestVerifyOrderRequiresSwapperSignature(t *testing.T) {
	o := testOrder()
	ctx := okContext()
	ctx.Signer = recoverAs{addr: common.Address{0xFF}}
	if _, err := VerifyOrder(orderWitness(o), ctx); !errors.Is(err, ErrBadSignature) {
		t.Fatalf("got %v, want ErrBadSignature", err)
	}
}

func TestVerifyOrderRefusesUnboundContext(t *testing.T) {
	o := testOrder()
	for name, b := range map[string]Binding{
		"no network": {CChainID: ids.ID{0x0C}, Reactor: testReactor},
		"no chain":   {NetworkID: 1, Reactor: testReactor},
		"no reactor": {NetworkID: 1, CChainID: ids.ID{0x0C}},
	} {
		ctx := okContext()
		ctx.Binding = b
		if _, err := VerifyOrder(orderWitness(o), ctx); !errors.Is(err, ErrOrderUnbound) {
			t.Fatalf("%s: got %v, want ErrOrderUnbound", name, err)
		}
	}
}

func TestVerifyOrderRefusesConsumedNonce(t *testing.T) {
	o := testOrder()
	ctx := okContext()
	if err := ctx.Nonces.Consume(o.Swapper, o.Nonce); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyOrder(orderWitness(o), ctx); !errors.Is(err, ErrNonceUsed) {
		t.Fatalf("got %v, want ErrNonceUsed", err)
	}
}

// Verification is a pure question. Consuming the nonce inside it would burn a
// trader's position on an order that was merely inspected.
func TestVerifyOrderDoesNotConsumeNonce(t *testing.T) {
	o := testOrder()
	ctx := okContext()
	if _, err := VerifyOrder(orderWitness(o), ctx); err != nil {
		t.Fatal(err)
	}
	if ctx.Nonces.Used(o.Swapper, o.Nonce) {
		t.Fatal("verification consumed the nonce; consumption belongs at settlement")
	}
}

func TestOrderValidateRefusesMissingFields(t *testing.T) {
	cases := map[string]func(*Order){
		"Swapper":     func(o *Order) { o.Swapper = common.Address{} },
		"Recipient":   func(o *Order) { o.Recipient = common.Address{} },
		"Market":      func(o *Order) { o.Market = ids.Empty },
		"Input.Asset": func(o *Order) { o.Input.Asset = ids.Empty },
		"Out.Asset":   func(o *Order) { o.MinOutput.Asset = ids.Empty },
		"Side":        func(o *Order) { o.Side = Side(9) },
		"Input zero":  func(o *Order) { o.Input.Amount = [32]byte{} },
	}
	for name, mutate := range cases {
		o := testOrder()
		mutate(&o)
		if err := o.Validate(); err == nil {
			t.Fatalf("%s: Validate accepted it", name)
		}
	}
}

// A zero MinOutput is a market order — an explicit choice to take whatever the book
// gives — and must be allowed.
func TestOrderAllowsZeroMinOutput(t *testing.T) {
	o := testOrder()
	o.MinOutput.Amount = [32]byte{}
	if err := o.Validate(); err != nil {
		t.Fatalf("market order refused: %v", err)
	}
}

// An amount that does not fit must be REFUSED, not wrapped. A silently truncated
// amount is an authorization the trader never gave.
func TestAmountRefusesOverflowAndNegative(t *testing.T) {
	var a AssetAmount
	tooBig := new(big.Int).Lsh(big.NewInt(1), 256)
	if err := a.SetBig(tooBig); !errors.Is(err, ErrAmountOverflow) {
		t.Fatalf("got %v, want ErrAmountOverflow", err)
	}
	if err := a.SetBig(big.NewInt(-1)); !errors.Is(err, ErrAmountNegative) {
		t.Fatalf("got %v, want ErrAmountNegative", err)
	}
	// The largest representable amount round-trips.
	max := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	if err := a.SetBig(max); err != nil {
		t.Fatal(err)
	}
	if a.Big().Cmp(max) != 0 {
		t.Fatal("max amount did not round trip")
	}
}

// AtLeast is the bounds check the trader's MinOutput rests on. Big-endian fixed
// width means a byte compare IS the numeric compare — verify that, including across
// a byte boundary where a naive comparison would go wrong.
func TestAtLeast(t *testing.T) {
	got := AssetAmount{Asset: testLUX, Amount: amount(256)}
	if !got.AtLeast(AssetAmount{Asset: testLUX, Amount: amount(255)}) {
		t.Fatal("256 should cover 255")
	}
	if got.AtLeast(AssetAmount{Asset: testLUX, Amount: amount(257)}) {
		t.Fatal("256 must not cover 257")
	}
	if !got.AtLeast(AssetAmount{Asset: testLUX, Amount: amount(256)}) {
		t.Fatal("256 should cover itself")
	}
	// A different asset never covers, however large.
	if got.AtLeast(AssetAmount{Asset: testUSDC, Amount: amount(1)}) {
		t.Fatal("an amount of one asset must never satisfy a bound on another")
	}
}

// The order encoding must be byte-identical across runs — it is what the trader
// signed, so any instability is a signature that stops verifying.
func TestOrderEncodingStable(t *testing.T) {
	o := testOrder()
	first := o.Encode()
	for i := 0; i < 50; i++ {
		if !bytes.Equal(o.Encode(), first) {
			t.Fatal("encoding is not stable")
		}
	}
}
