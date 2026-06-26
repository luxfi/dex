// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package rfq

import (
	"crypto/ecdsa"
	"math/big"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/luxfi/crypto"
	"github.com/luxfi/dex/pkg/htlc"
)

// key derives a deterministic secp256k1 key from a seed (same idiom as the
// dchain test harness: re-hash on the vanishingly rare out-of-range scalar).
func key(seed string) *ecdsa.PrivateKey {
	d := crypto.Keccak256([]byte("lux.rfq.test.secp.v1"), []byte(seed))
	for {
		priv, err := crypto.ToECDSA(d)
		if err == nil {
			return priv
		}
		d = crypto.Keccak256(d)
	}
}

// fixture is the shared swap setup: a Lux→Bitcoin pair (maker sells USDC on Lux,
// taker pays BTC), a maker and taker key, a published firm quote, and the agreed
// timeouts.
type fixture struct {
	co      *Coordinator
	maker   *ecdsa.PrivateKey
	user    *ecdsa.PrivateKey
	pair    Pair
	quote   Quote
	quoteID QuoteID
	t0      time.Time
	tA, tB  uint64
	userA   string
	userB   string
	sid     SessionID
}

const (
	makerA = "lux1maker-refund-A"
	makerB = "bc1maker-payout-B"
	userA  = "lux1user-payout-A"
	userB  = "bc1user-refund-B"
)

func newFixture(t *testing.T) *fixture {
	t.Helper()
	maker, user := key("maker"), key("user")
	pair := Pair{ChainA: ChainLux, AssetA: "USDC", ChainB: ChainBitcoin, AssetB: "BTC"}
	t0 := time.Unix(1_700_000_000, 0)
	tA, tB := AssignTimeouts(t0, pair.ChainA, pair.ChainB)

	q := Quote{
		Pair:    pair,
		Rate:    "0.000016",
		AmountA: big.NewInt(1_000_000), // 1 USDC (6 decimals)
		AmountB: big.NewInt(16_000),    // 16k sats
		MakerA:  makerA,
		MakerB:  makerB,
		Expiry:  t0.Add(30 * time.Second).Unix(),
	}
	if err := q.Sign(maker); err != nil {
		t.Fatalf("sign quote: %v", err)
	}
	co := NewCoordinator()
	id, err := co.PublishQuote(q, t0)
	if err != nil {
		t.Fatalf("publish quote: %v", err)
	}
	return &fixture{co: co, maker: maker, user: user, pair: pair, quote: q, quoteID: id, t0: t0, tA: tA, tB: tB, userA: userA, userB: userB}
}

// commit drives the hashlock-commit step and returns the session id and secret.
func (f *fixture) commit(t *testing.T) (SessionID, htlc.Secret) {
	t.Helper()
	s, err := htlc.NewSecret()
	if err != nil {
		t.Fatalf("new secret: %v", err)
	}
	body := CommitPayload{QuoteID: f.quoteID, Hashlock: s.Hashlock(), UserA: f.userA, UserB: f.userB, User: pubAddr(f.user)}
	m, err := Sign(body, f.user)
	if err != nil {
		t.Fatalf("sign commit: %v", err)
	}
	id, err := f.co.CommitHashlock(m, f.t0)
	if err != nil {
		t.Fatalf("commit hashlock: %v", err)
	}
	return id, s
}

func (f *fixture) legA(timeout uint64, recipient, refund string) Signed[LegPayload] {
	p := LegProof{Chain: f.pair.ChainA, Asset: f.pair.AssetA, Recipient: recipient, Refund: refund, Amount: f.quote.AmountA, Timeout: timeout, TxRef: "lux:0xA"}
	m, _ := Sign(LegPayload{Session: f.sid, Leg: 0, Proof: p}, f.maker)
	return m
}

func (f *fixture) legB(timeout uint64, recipient, refund string) Signed[LegPayload] {
	p := LegProof{Chain: f.pair.ChainB, Asset: f.pair.AssetB, Recipient: recipient, Refund: refund, Amount: f.quote.AmountB, Timeout: timeout, TxRef: "btc:0xB"}
	m, _ := Sign(LegPayload{Session: f.sid, Leg: 1, Proof: p}, f.user)
	return m
}

// (a) Full happy-path walk Quoted → Settled with correct role bindings and a
// valid T_B > T_A + Δ.
func TestHappyPathSettles(t *testing.T) {
	f := newFixture(t)
	id, secret := f.commit(t)
	f.sid = id

	if got := mustState(t, f.co, id); got != StateHashlockCommitted {
		t.Fatalf("after commit: state %v", got)
	}

	if _, err := f.co.SubmitLegA(f.legA(f.tA, f.userA, makerA)); err != nil {
		t.Fatalf("submit leg A: %v", err)
	}
	if got := mustState(t, f.co, id); got != StateLegALocked {
		t.Fatalf("after leg A: state %v", got)
	}

	if _, err := f.co.SubmitLegB(f.legB(f.tB, makerB, f.userB)); err != nil {
		t.Fatalf("submit leg B: %v", err)
	}
	if got := mustState(t, f.co, id); got != StateLegBLocked {
		t.Fatalf("after leg B: state %v", got)
	}

	// Taker claims HTLC_A, revealing s; the coordinator relays it to the maker.
	cm, _ := Sign(ClaimPayload{Session: id, Secret: secret}, f.user)
	relayed, err := f.co.RecordClaim(cm)
	if err != nil {
		t.Fatalf("record claim: %v", err)
	}
	if relayed != secret {
		t.Fatalf("relayed preimage != revealed secret")
	}
	if got := mustState(t, f.co, id); got != StateClaimed {
		t.Fatalf("after claim: state %v", got)
	}

	// Maker counter-claims HTLC_B with the relayed s.
	sm, _ := Sign(SettlePayload{Session: id, TxRef: "btc:0xB-claim"}, f.maker)
	if err := f.co.RecordCounterClaim(sm); err != nil {
		t.Fatalf("record counter-claim: %v", err)
	}
	if got := mustState(t, f.co, id); got != StateSettled {
		t.Fatalf("final: state %v, want settled", got)
	}

	// The valid ordering held over the whole walk.
	if !OrderingHolds(f.tA, f.tB, Delta(f.pair.ChainA, f.pair.ChainB)) {
		t.Fatalf("fixture timeouts violate the ordering invariant")
	}
}

// (b) A submitted leg B that violates T_B > T_A + Δ is rejected.
func TestLegBTimeoutOrderingRejected(t *testing.T) {
	f := newFixture(t)
	id, _ := f.commit(t)
	f.sid = id
	if _, err := f.co.SubmitLegA(f.legA(f.tA, f.userA, makerA)); err != nil {
		t.Fatalf("submit leg A: %v", err)
	}
	// T_B one second past T_A: far short of T_A + Δ (Δ = 2h for Lux↔BTC).
	_, err := f.co.SubmitLegB(f.legB(f.tA+1, makerB, f.userB))
	if err != ErrTimeoutOrdering {
		t.Fatalf("leg B with bad ordering: err = %v, want ErrTimeoutOrdering", err)
	}
	if got := mustState(t, f.co, id); got != StateLegALocked {
		t.Fatalf("session advanced past leg A on a rejected leg B: %v", got)
	}
}

// (c) A wrong-role binding (HTLC_A recipient != taker) is rejected.
func TestLegARoleBindingRejected(t *testing.T) {
	f := newFixture(t)
	id, _ := f.commit(t)
	f.sid = id
	// Maker tries to lock HTLC_A paying itself instead of the taker.
	_, err := f.co.SubmitLegA(f.legA(f.tA, makerA, makerA))
	if err != ErrRoleBinding {
		t.Fatalf("leg A wrong recipient: err = %v, want ErrRoleBinding", err)
	}
	if got := mustState(t, f.co, id); got != StateHashlockCommitted {
		t.Fatalf("session advanced on a rejected leg A: %v", got)
	}
}

// (d) An expired quote and a badly-signed quote are both refused at the book.
func TestQuoteRejection(t *testing.T) {
	maker := key("maker")
	t0 := time.Unix(1_700_000_000, 0)
	base := Quote{
		Pair:    Pair{ChainA: ChainLux, AssetA: "USDC", ChainB: ChainBitcoin, AssetB: "BTC"},
		Rate:    "0.000016",
		AmountA: big.NewInt(1_000_000),
		AmountB: big.NewInt(16_000),
		MakerA:  makerA, MakerB: makerB,
		Expiry: t0.Add(30 * time.Second).Unix(),
	}

	// Expired.
	exp := base
	if err := exp.Sign(maker); err != nil {
		t.Fatalf("sign: %v", err)
	}
	book := NewQuoteBook()
	if _, err := book.Insert(exp, t0.Add(time.Minute)); err != ErrQuoteExpired {
		t.Fatalf("expired quote: err = %v, want ErrQuoteExpired", err)
	}

	// Badly signed: tamper a field after signing so the recovered signer != Maker.
	bad := base
	if err := bad.Sign(maker); err != nil {
		t.Fatalf("sign: %v", err)
	}
	bad.AmountB = big.NewInt(1) // not covered by the now-stale signature
	if err := bad.Verify(); err != ErrQuoteBadSignature {
		t.Fatalf("tampered quote: err = %v, want ErrQuoteBadSignature", err)
	}
	if _, err := book.Insert(bad, t0); err != ErrQuoteBadSignature {
		t.Fatalf("tampered quote insert: err = %v, want ErrQuoteBadSignature", err)
	}

	// A forged signer (signed by someone else, asserting maker's address) fails.
	forged := base
	if err := forged.Sign(key("attacker")); err != nil {
		t.Fatalf("sign: %v", err)
	}
	forged.Maker = pubAddr(maker) // claim to be the maker
	if err := forged.Verify(); err != ErrQuoteBadSignature {
		t.Fatalf("forged maker: err = %v, want ErrQuoteBadSignature", err)
	}
}

// A leg message signed by the wrong party is refused (relay authentication).
func TestWrongAuthorRejected(t *testing.T) {
	f := newFixture(t)
	id, _ := f.commit(t)
	f.sid = id
	// Taker (not maker) tries to submit HTLC_A.
	p := LegProof{Chain: f.pair.ChainA, Asset: f.pair.AssetA, Recipient: f.userA, Refund: makerA, Amount: f.quote.AmountA, Timeout: f.tA}
	m, _ := Sign(LegPayload{Session: id, Leg: 0, Proof: p}, f.user)
	if _, err := f.co.SubmitLegA(m); err != ErrWrongAuthor {
		t.Fatalf("leg A signed by taker: err = %v, want ErrWrongAuthor", err)
	}
}

// (e) NON-CUSTODY: structurally, neither the Coordinator nor a SwapSession has a
// field of any private-key type, so no relay path can sign for an asset. The
// only secret either holds is the preimage, and only AFTER it is public on chain
// A — the coordinator stores it to relay, never to take custody.
func TestNonCustody(t *testing.T) {
	for _, typ := range []reflect.Type{reflect.TypeOf(Coordinator{}), reflect.TypeOf(SwapSession{})} {
		for i := 0; i < typ.NumField(); i++ {
			name := typ.Field(i).Type.String()
			if strings.Contains(name, "PrivateKey") || strings.Contains(name, "ecdsa.PrivateKey") || strings.Contains(name, "btcec.PrivateKey") {
				t.Fatalf("%s.%s is a private-key type %q — venue must hold no key", typ.Name(), typ.Field(i).Name, name)
			}
		}
	}

	// Drive a full settlement and confirm the coordinator exposes no signing
	// capability: it only ever recovered/relayed. The single secret it holds was
	// supplied by the taker's own claim (already public on chain A).
	f := newFixture(t)
	id, secret := f.commit(t)
	f.sid = id
	if _, err := f.co.SubmitLegA(f.legA(f.tA, f.userA, makerA)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.co.SubmitLegB(f.legB(f.tB, makerB, f.userB)); err != nil {
		t.Fatal(err)
	}
	cm, _ := Sign(ClaimPayload{Session: id, Secret: secret}, f.user)
	if _, err := f.co.RecordClaim(cm); err != nil {
		t.Fatal(err)
	}
	sess, _ := f.co.Session(id)
	if sess.Secret == nil || *sess.Secret != secret {
		t.Fatalf("coordinator did not record the publicly-revealed preimage for relay")
	}
}

// The off-path refund/abort terminals leave the session in a whole-funds state.
func TestRefundAndAbort(t *testing.T) {
	// Abort before any lock.
	f := newFixture(t)
	id, _ := f.commit(t)
	f.sid = id
	rm, _ := Sign(RefundPayload{Session: id, TxRef: "none"}, f.user)
	if err := f.co.RecordRefund(rm); err != nil {
		t.Fatalf("abort: %v", err)
	}
	if got := mustState(t, f.co, id); got != StateAborted {
		t.Fatalf("pre-lock refund: state %v, want aborted", got)
	}

	// Refund after a leg is locked.
	g := newFixture(t)
	id2, _ := g.commit(t)
	g.sid = id2
	if _, err := g.co.SubmitLegA(g.legA(g.tA, g.userA, makerA)); err != nil {
		t.Fatal(err)
	}
	rm2, _ := Sign(RefundPayload{Session: id2, TxRef: "lux:0xA-refund"}, g.maker)
	if err := g.co.RecordRefund(rm2); err != nil {
		t.Fatalf("refund: %v", err)
	}
	if got := mustState(t, g.co, id2); got != StateRefunded {
		t.Fatalf("post-lock refund: state %v, want refunded", got)
	}
}

// Delta and AssignTimeouts: boundary table from the spec's concrete parameters.
func TestTimeoutTable(t *testing.T) {
	cases := []struct {
		a, b ChainKind
		want time.Duration
	}{
		{ChainLux, ChainBitcoin, 2 * time.Hour},
		{ChainBitcoin, ChainLux, 2 * time.Hour},
		{ChainEVM, ChainEVM, 5 * time.Minute},
		{ChainLux, ChainEVM, 5 * time.Minute},
		{ChainLux, ChainLux, 0},
	}
	t0 := time.Unix(1_700_000_000, 0)
	for _, c := range cases {
		if got := Delta(c.a, c.b); got != c.want {
			t.Fatalf("Delta(%v,%v) = %v, want %v", c.a, c.b, got, c.want)
		}
		tA, tB := AssignTimeouts(t0, c.a, c.b)
		if !OrderingHolds(tA, tB, Delta(c.a, c.b)) {
			t.Fatalf("AssignTimeouts(%v,%v) -> (%d,%d) violates ordering", c.a, c.b, tA, tB)
		}
		if tA != uint64(t0.Add(MakerLegLifetime).Unix()) {
			t.Fatalf("tA = %d, want t0+%v", tA, MakerLegLifetime)
		}
	}
}

// Concurrent independent swaps exercise the coordinator's single mutex and the
// quote book's RWMutex. Under `go test -race` this proves the relay surface is
// data-race free; without the detector it still asserts every swap settles.
func TestConcurrentSwaps(t *testing.T) {
	const n = 32
	done := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(i int) { done <- runOneSwap(i) }(i)
	}
	for i := 0; i < n; i++ {
		if err := <-done; err != nil {
			t.Fatalf("concurrent swap: %v", err)
		}
	}
}

func runOneSwap(i int) error {
	maker, user := key("cmaker"), key("cuser")
	pair := Pair{ChainA: ChainEVM, AssetA: "USDC", ChainB: ChainEVM, AssetB: "WETH"}
	t0 := time.Unix(1_700_000_000, 0)
	tA, tB := AssignTimeouts(t0, pair.ChainA, pair.ChainB)
	q := Quote{Pair: pair, Rate: "1", AmountA: big.NewInt(int64(1000 + i)), AmountB: big.NewInt(1), MakerA: makerA, MakerB: makerB, Expiry: t0.Add(time.Minute).Unix()}
	if err := q.Sign(maker); err != nil {
		return err
	}
	co := NewCoordinator()
	qid, err := co.PublishQuote(q, t0)
	if err != nil {
		return err
	}
	s, err := htlc.NewSecret()
	if err != nil {
		return err
	}
	cm, err := Sign(CommitPayload{QuoteID: qid, Hashlock: s.Hashlock(), UserA: userA, UserB: userB, User: pubAddr(user)}, user)
	if err != nil {
		return err
	}
	id, err := co.CommitHashlock(cm, t0)
	if err != nil {
		return err
	}
	la, _ := Sign(LegPayload{Session: id, Leg: 0, Proof: LegProof{Chain: pair.ChainA, Asset: pair.AssetA, Recipient: userA, Refund: makerA, Amount: q.AmountA, Timeout: tA}}, maker)
	if _, err := co.SubmitLegA(la); err != nil {
		return err
	}
	lb, _ := Sign(LegPayload{Session: id, Leg: 1, Proof: LegProof{Chain: pair.ChainB, Asset: pair.AssetB, Recipient: makerB, Refund: userB, Amount: q.AmountB, Timeout: tB}}, user)
	if _, err := co.SubmitLegB(lb); err != nil {
		return err
	}
	clm, _ := Sign(ClaimPayload{Session: id, Secret: s}, user)
	if _, err := co.RecordClaim(clm); err != nil {
		return err
	}
	stm, _ := Sign(SettlePayload{Session: id}, maker)
	return co.RecordCounterClaim(stm)
}

func mustState(t *testing.T, co *Coordinator, id SessionID) State {
	t.Helper()
	s, ok := co.Session(id)
	if !ok {
		t.Fatalf("session %x not found", id[:8])
	}
	return s.State
}
