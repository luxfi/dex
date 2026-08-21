// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"encoding/binary"
	"math"
	"math/big"
	"testing"
	"time"

	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// audit_order_frontdoor_test.go audits the pkg/dex ORDER FRONT DOOR: what a
// signature over an order actually promises, and whether every value the order's
// fields can hold reaches a decided outcome.
//
// Two separate questions, because they fail separately:
//
//	AUTHORITY  — does the digest a signature covers name the market, the side,
//	             the price, the size, the nonce and an expiry, or does one
//	             signature authorize more than one order?
//	TOTALITY   — does every byte the Side field can hold reach one branch, or
//	             does an out-of-domain value fall between two readers?
//
// SCOPE. pkg/dex is the matcher library. The D-Chain (pkg/dchain) authorizes its
// own transactions through a different digest (txAuthDigest), which DOES bind the
// network, the chain, the tx type and a monotonic nonce — that surface is audited
// in pkg/dchain/auth_test.go and is not what these tests are about. What is
// audited here is the SignedOrder digest this package exports and documents as
// "the auth surface for the GPU-batched ingestion path", plus the matcher
// primitives the D-Chain calls into.

// ---------------------------------------------------------------------------
// Signing harness.
// ---------------------------------------------------------------------------

// newSigner derives a stable key from a seed so a test's digests are reproducible.
func newSigner(t *testing.T, seed string) (priv []byte, addr common.Address) {
	t.Helper()
	d := crypto.Keccak256([]byte("lx.dex.audit.signer.v1"), []byte(seed))
	for {
		k, err := crypto.ToECDSA(d)
		if err == nil {
			return d, common.BytesToAddress(crypto.PubkeyToAddress(k.PublicKey).Bytes())
		}
		d = crypto.Keccak256(d)
	}
}

// sign attaches a valid signature over o.SigningHash() under priv.
func sign(t *testing.T, o *SignedOrder, priv []byte) {
	t.Helper()
	k, err := crypto.ToECDSA(priv)
	if err != nil {
		t.Fatalf("ToECDSA: %v", err)
	}
	h, err := o.SigningHash()
	if err != nil {
		t.Fatalf("SigningHash: %v", err)
	}
	sig, err := crypto.Sign(h[:], k)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}
	copy(o.Sig[:], sig)
}

// verifies reports the single-order verdict of the shipped batch verifier — the
// same function every ingestion path would call.
func verifies(t *testing.T, o SignedOrder) bool {
	t.Helper()
	out, err := BatchVerifyOrders([]SignedOrder{o})
	if err != nil {
		t.Fatalf("BatchVerifyOrders: %v", err)
	}
	return out[0]
}

// baseOrder is a plain, well-formed limit order: the shape the digest was written
// for, so anything it fails to bind is a gap and not a mis-built fixture.
func baseOrder(sender common.Address) SignedOrder {
	return SignedOrder{
		Order: Order{
			ID:       11,
			Symbol:   "LUX-LUSD",
			Type:     Limit,
			Side:     Buy,
			Price:    5,
			Size:     10,
			ClientID: "client-1",
		},
		Sender: sender,
	}
}

// ---------------------------------------------------------------------------
// AUTHORITY: what one signature authorizes.
// ---------------------------------------------------------------------------

// TestSignedOrderBindsEveryFieldItActsOn enumerates the order fields a matcher
// ACTS ON and requires each one to change the digest. A field the digest does not
// cover can be rewritten by anyone between signing and matching, and the
// signature still verifies — which is exactly "a verifier accepts what it must
// refuse", under a substituted field rather than a substituted key.
//
// SigningHash covers ID, Symbol, Side, Type, Price, Size, ClientID and Sender.
// Everything else on Order is outside it. That includes fields the matcher reads
// on the very next line: PostOnly and Flags decide whether a resting order is
// refused for taking liquidity (addRestingLocked), StopPrice decides when a Stop
// order triggers (validateOrder requires it positive), TimeInForce and ReduceOnly
// decide the order's lifetime and its effect on a position, DisplaySize decides
// how much of an iceberg is visible.
func TestSignedOrderBindsEveryFieldItActsOn(t *testing.T) {
	priv, addr := newSigner(t, "alice")

	// Each case takes a signed order and rewrites ONE field an attacker would
	// benefit from rewriting, exactly as a relay between the client and the
	// matcher could.
	rewrites := []struct {
		field string
		why   string
		apply func(o *SignedOrder)
	}{
		{"PostOnly", "a maker-only order becomes a taker and crosses the book", func(o *SignedOrder) { o.PostOnly = !o.PostOnly }},
		{"Flags", "the post-only / reduce-only / STP bits are cleared", func(o *SignedOrder) { o.Flags |= OrderFlagPostOnly }},
		{"StopPrice", "a stop order triggers at a price its owner never authorized", func(o *SignedOrder) { o.StopPrice = 1 }},
		{"LimitPrice", "a stop-limit fills at a price its owner never authorized", func(o *SignedOrder) { o.LimitPrice = 1 }},
		{"TimeInForce", "an IOC becomes a resting GTC and stays exposed", func(o *SignedOrder) { o.TimeInForce = "GTC" }},
		{"ReduceOnly", "a position-reducing order becomes position-opening", func(o *SignedOrder) { o.ReduceOnly = !o.ReduceOnly }},
		{"DisplaySize", "an iceberg reveals its full size", func(o *SignedOrder) { o.DisplaySize = 1 }},
		{"Hidden", "a hidden order becomes visible, or a visible one hides", func(o *SignedOrder) { o.Hidden = !o.Hidden }},
		{"PegOffset", "a pegged order's offset is moved", func(o *SignedOrder) { o.PegOffset = 1 }},
		{"TakeProfit", "the bracket's take-profit leg is moved", func(o *SignedOrder) { o.TakeProfit = 1 }},
		{"StopLoss", "the bracket's stop-loss leg is moved", func(o *SignedOrder) { o.StopLoss = 1 }},
		{"UserID", "the order is attributed to a different account for self-trade prevention", func(o *SignedOrder) { o.UserID = "someone-else" }},
		{"User", "the order is attributed to a different account for self-trade prevention", func(o *SignedOrder) { o.User = "someone-else" }},
	}

	var unbound []string
	for _, rw := range rewrites {
		o := baseOrder(addr)
		sign(t, &o, priv)
		if !verifies(t, o) {
			t.Fatalf("precondition: the unmodified %s fixture does not verify", rw.field)
		}
		rw.apply(&o) // signature untouched
		if verifies(t, o) {
			unbound = append(unbound, rw.field+" ("+rw.why+")")
		}
	}

	// Control: the fields the digest DOES cover must all reject a rewrite, or the
	// test is measuring nothing.
	bound := []struct {
		field string
		apply func(o *SignedOrder)
	}{
		{"ID", func(o *SignedOrder) { o.ID++ }},
		{"Symbol", func(o *SignedOrder) { o.Symbol = "ZOO-LUSD" }},
		{"Side", func(o *SignedOrder) { o.Side = Sell }},
		{"Type", func(o *SignedOrder) { o.Type = Market }},
		{"Price", func(o *SignedOrder) { o.Price += 1 }},
		{"Size", func(o *SignedOrder) { o.Size += 1 }},
		{"ClientID", func(o *SignedOrder) { o.ClientID = "client-2" }},
		{"Sender", func(o *SignedOrder) { o.Sender[0] ^= 0xff }},
	}
	for _, b := range bound {
		o := baseOrder(addr)
		sign(t, &o, priv)
		b.apply(&o)
		if verifies(t, o) {
			t.Fatalf("control failed: rewriting %s did not invalidate the signature — the test cannot\n"+
				"  distinguish a bound field from an unbound one", b.field)
		}
	}

	if len(unbound) > 0 {
		t.Fatalf("DEFECT: a SignedOrder signature does not cover %d fields the matcher acts on.\n"+
			"  Each of these was rewritten AFTER signing and BatchVerifyOrders still returned valid:\n"+
			"    %v\n"+
			"  SigningHash (pkg/dex/signed_order.go) packs only ID, Symbol, Side, Type, Price, Size,\n"+
			"  ClientID and Sender. Anything else on Order is outside the digest, so anyone who relays\n"+
			"  a signed order can rewrite it and the signature still verifies. PostOnly and Flags are\n"+
			"  read by addRestingLocked to refuse an order that would take liquidity; StopPrice is read\n"+
			"  by validateOrder; TimeInForce, ReduceOnly and DisplaySize decide lifetime, position\n"+
			"  effect and visibility. A signature that authorizes a maker-only order must not authorize\n"+
			"  the same order as a taker.\n"+
			"  FIX: either pack every field the matcher reads into the digest, or refuse at ingestion\n"+
			"  any order whose unbound fields are non-zero, so the signed subset IS the whole order.",
			len(unbound), unbound)
	}
}

// TestSignedOrderIsBoundToOneChainAndOneEpoch asks the replay question directly.
//
// The D-Chain's transaction digest binds (network, chain, type, nonce) precisely
// so a signed frame spends on exactly one chain, once — its own doc comment says
// "BINDING THE SCOPE IS WHAT MAKES A SIGNATURE SPEND ONCE, ON ONE CHAIN." The
// order digest in this package binds none of that. Symbol names the market, so a
// cross-MARKET replay is closed; nothing names the chain, the network, or a
// deadline, and the only field that could serve as a nonce (ID) is chosen by the
// client, so two chains that both accept SignedOrder accept the same bytes.
func TestSignedOrderIsBoundToOneChainAndOneEpoch(t *testing.T) {
	priv, addr := newSigner(t, "bob")
	o := baseOrder(addr)
	sign(t, &o, priv)
	if !verifies(t, o) {
		t.Fatal("precondition: the fixture does not verify")
	}

	h1, err := o.SigningHash()
	if err != nil {
		t.Fatalf("SigningHash: %v", err)
	}

	// The market IS bound (Symbol is in the preimage) — the one replay axis that
	// is closed. Assert it so the finding below is precise about what is missing.
	crossMarket := o
	crossMarket.Symbol = "ZOO-LUSD"
	if verifies(t, crossMarket) {
		t.Fatal("DEFECT: a signed order replays into a different market — Symbol is not bound")
	}

	// The measurement. Rebuild the preimage independently from the eight order
	// fields alone — no domain tag, no network, no chain, no expiry — and hash it.
	// If that equals the production digest then the production preimage IS just
	// those eight fields, which is the finding. The moment a scope is added to
	// SigningHash the two stop matching and this test passes.
	unscoped := func(o SignedOrder) [32]byte {
		var u8 [8]byte
		var buf []byte
		binary.BigEndian.PutUint64(u8[:], o.ID)
		buf = append(buf, u8[:]...)
		binary.BigEndian.PutUint16(u8[:2], uint16(len(o.Symbol)))
		buf = append(buf, u8[:2]...)
		buf = append(buf, o.Symbol...)
		buf = append(buf, byte(o.Side), byte(o.Type))
		binary.BigEndian.PutUint64(u8[:], uint64(int64(o.Price*PriceMultiplier)))
		buf = append(buf, u8[:]...)
		binary.BigEndian.PutUint64(u8[:], uint64(math.Round(o.Size*PriceMultiplier)))
		buf = append(buf, u8[:]...)
		binary.BigEndian.PutUint16(u8[:2], uint16(len(o.ClientID)))
		buf = append(buf, u8[:2]...)
		buf = append(buf, o.ClientID...)
		buf = append(buf, o.Sender[:]...)
		var out [32]byte
		copy(out[:], crypto.Keccak256(buf))
		return out
	}
	if unscoped(o) != h1 {
		return // the preimage carries something beyond the eight order fields.
	}

	t.Fatalf("DEFECT: a SignedOrder signature names no chain, no network, no epoch and no expiry.\n"+
		"  SigningHash (pkg/dex/signed_order.go) packs ID, Symbol, Side, Type, Price, Size, ClientID\n"+
		"  and Sender — and nothing else. Consequences, in order of severity:\n"+
		"    - CROSS-CHAIN REPLAY: nothing in the preimage distinguishes one D-Chain from another, so a\n"+
		"      signed order captured on a devnet is a valid signed order for the same account on\n"+
		"      mainnet. pkg/dchain/auth.go closes exactly this hole for transactions (txAuthDomain is\n"+
		"      'lux.dchain.tx.auth.v2' and the preimage carries network[4] and chain[32]); the order\n"+
		"      digest never got the same treatment.\n"+
		"    - NO EXPIRY: the digest has no deadline, so a captured order is submittable forever. A\n"+
		"      resting bid signed at one price is a standing option against the signer at every later\n"+
		"      price.\n"+
		"    - NO DOMAIN TAG: the preimage is raw packed fields with no domain separator, so it is not\n"+
		"      structurally distinct from any other keccak256 preimage in the stack.\n"+
		"    - THE ONLY NONCE IS CLIENT-CHOSEN: ID is a uint64 the client picks and nothing here\n"+
		"      enforces single use, so replay protection would have to come entirely from a\n"+
		"      seen-set the caller maintains.\n"+
		"  The digest computed for this order is %x. The same 32 bytes are the digest for this order\n"+
		"  on every chain, in every epoch.\n"+
		"  FIX: give the order digest the same scope the tx digest already has — a versioned domain\n"+
		"  tag, network[4], chain[32], and an explicit expiry field packed into the preimage.", h1)
}

// TestSigningHashIsInjectiveInSize pins that two orders which differ in a
// value-bearing field produce different digests. Size is a float64 the digest
// reduces to uint64(math.Round(Size * 1e8)), so an interval of distinct sizes
// collapses to one tick count and therefore to one digest — one signature covers
// every size in that interval.
func TestSigningHashIsInjectiveInSize(t *testing.T) {
	_, addr := newSigner(t, "carol")

	a := baseOrder(addr)
	b := baseOrder(addr)
	b.Size = a.Size + 3e-9 // distinct float64s, same tick count after rounding

	if a.Size == b.Size {
		t.Fatal("fixture: the two sizes are not distinct")
	}
	ha, err := a.SigningHash()
	if err != nil {
		t.Fatalf("SigningHash of an on-tick size must succeed: %v", err)
	}
	// The collision is closed by REFUSAL rather than by a wider digest: a size
	// that is not an exact number of ticks cannot be signed at all, so the two
	// orders below can never both exist to share a digest. Snapping it instead
	// would decide, silently, that the signer meant a quantity they did not
	// write — which is the one thing a signature is supposed to prevent.
	hb, err := b.SigningHash()
	if err != nil {
		return // off-tick size refused: the collision is unreachable
	}
	if ha == hb {
		t.Fatalf("DEFECT: two orders with DIFFERENT sizes share one signing digest.\n"+
			"  Size=%v and Size=%v both pack as uint64(math.Round(Size*1e8)) = %d, so one signature\n"+
			"  authorizes either. Whether that matters depends on which value the consumer reads:\n"+
			"  the digest commits to the ROUNDED tick count while Order.Size (the float the matcher's\n"+
			"  crossing loop and RemainingSize use) is unbound within the rounding interval. A\n"+
			"  signature is supposed to name ONE order.\n"+
			"  FIX: carry the size as the exact integer it settles in — the wire already does this\n"+
			"  (zapwire size is uint64 atomic base units, and pkg/dchain newWireOrder treats it as\n"+
			"  authoritative) — and pack that integer, so the digest and the settled quantity are the\n"+
			"  same value.", a.Size, b.Size, uint64(math.Round(a.Size*PriceMultiplier)))
	}
}

// ---------------------------------------------------------------------------
// DETERMINISM: a float that is not a number, converted to an integer.
// ---------------------------------------------------------------------------

// TestOrderGuardsRefuseNonFiniteFloats pins that no order field can carry a
// non-finite float into an integer conversion.
//
// Every bound check in this package is a strict inequality, and every strict
// inequality is FALSE for NaN. safePriceToInt asks `price < 0` and
// `price > MaxSafePrice`; SigningHash asks `Size < 0` and `Size > MaxSafePrice`;
// validateOrder asks `Price <= 0`, `Size <= 0` and `Price > MaxSafePrice`. NaN
// answers false to all six, so it passes every guard and reaches
// `PriceInt(price * PriceMultiplier)` and `uint64(math.Round(Size * 1e8))`.
//
// Go leaves a float-to-integer conversion of an out-of-range value
// implementation-defined, and the implementations differ: arm64 saturates (this
// host yields 0), amd64 wraps (it yields 0x8000000000000000). This repo's own
// wire design says so in as many words — pkg/zapwire's CONSENSUS-DETERMINISM
// note is the reason the D-Chain carries exact integers instead of floats:
// "DIFFERS BY CPU ARCHITECTURE (arm64 saturates, amd64 wraps). Two honest
// validators on different archs then derived different reserves / balances /
// fills from the SAME ordered block and forked."
//
// The D-Chain path is protected by construction — newWireOrder builds Price and
// Size from uint64 wire integers, so they are always finite. These guards are the
// last line for every OTHER caller of this exported package, and today they are
// not a line at all.
func TestOrderGuardsRefuseNonFiniteFloats(t *testing.T) {
	nan := math.NaN()

	t.Run("safePriceToInt", func(t *testing.T) {
		got, err := safePriceToInt(nan)
		if err == nil {
			t.Fatalf("DEFECT: safePriceToInt(NaN) returned %d with no error. Its two guards are\n"+
				"  `price < 0` and `price > MaxSafePrice`; both are false for NaN, so NaN reaches\n"+
				"  `PriceInt(price * PriceMultiplier)` — a float-to-int conversion of a non-finite\n"+
				"  value, which Go leaves implementation-defined and which differs by architecture\n"+
				"  (this host: %d; amd64: -9223372036854775808). Two honest validators computing this\n"+
				"  from the same order get different price grid keys.\n"+
				"  FIX: refuse !(price >= 0 && price <= MaxSafePrice), which is false for NaN.",
				got, got)
		}
	})

	t.Run("SigningHash", func(t *testing.T) {
		for _, c := range []struct {
			field string
			mut   func(o *SignedOrder)
		}{
			{"Size", func(o *SignedOrder) { o.Size = nan }},
			{"Price", func(o *SignedOrder) { o.Price = nan }},
		} {
			o := baseOrder(common.Address{})
			c.mut(&o)
			h, err := o.SigningHash()
			if err == nil {
				t.Fatalf("DEFECT: SigningHash accepted %s = NaN and produced digest %x.\n"+
					"  The guard is `o.Size < 0 || o.Size > MaxSafePrice` (and safePriceToInt's pair for\n"+
					"  Price); every one is false for NaN. NaN then reaches uint64(math.Round(...)) /\n"+
					"  PriceInt(...), whose result is architecture-dependent — so the SAME order bytes\n"+
					"  hash to DIFFERENT digests on arm64 and amd64, and a signature valid on one\n"+
					"  validator is invalid on the other.\n"+
					"  FIX: require the field to be finite before any conversion.", c.field, h)
			}
		}
	})

	t.Run("validateOrder", func(t *testing.T) {
		ob := NewOrderBook("LUX-LUSD")
		for _, c := range []struct {
			field string
			o     *Order
		}{
			{"Price", &Order{ID: 1, Symbol: "LUX-LUSD", Type: Limit, Side: Buy, Price: nan, Size: 1}},
			{"Size", &Order{ID: 1, Symbol: "LUX-LUSD", Type: Limit, Side: Buy, Price: 1, Size: nan}},
			{"StopPrice", &Order{ID: 1, Symbol: "LUX-LUSD", Type: Stop, Side: Buy, Price: 1, Size: 1, StopPrice: nan}},
		} {
			if err := ob.validateOrder(c.o); err == nil {
				t.Fatalf("DEFECT: validateOrder accepted %s = NaN. Its guards are `<= 0` and\n"+
					"  `> MaxSafePrice`, both false for NaN, so a NaN-priced order validates and rests.\n"+
					"  addRestingLocked then keys its price level with PriceInt(order.Price * 1e8) when\n"+
					"  PriceUnits is unset — an architecture-dependent level key, so two validators put\n"+
					"  the same order at different levels and derive different books.\n"+
					"  FIX: require the field to be finite before comparing it.", c.field)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// TOTALITY: the Side field, read by two rules that disagree.
// ---------------------------------------------------------------------------

// TestMarketableCrossAppliesItsLimitForEverySide is the matcher-level root of the
// D-Chain halt that pkg/dchain/audit_frontdoor_test.go reaches end to end.
//
// tryMatchImmediateLocked reads Side twice, with two different rules:
//
//	tree   : `if order.Side == Buy { asks } else { bids }`   — total, "not Buy" is Sell
//	limit  : `if order.Side == Buy … break; if order.Side == Sell … break` — partial
//
// The tree selection has an else, so a Side outside {Buy, Sell} sweeps the bid
// side, i.e. it is treated as a SELL. The limit-price guard has no else, so the
// SAME order matches neither arm and the price bound is never applied: an order
// declared Limit crosses as if it were Market, at any price the book offers.
//
// The test asks the property directly. A sell for 10 at a limit of 100 must not
// touch a bid at 5 — for EVERY value of Side that the matcher will sweep the bid
// side for.
func TestMarketableCrossAppliesItsLimitForEverySide(t *testing.T) {
	for side := 0; side < 256; side++ {
		s := Side(side)

		ob := NewOrderBook("LUX-LUSD")
		maker := &Order{
			ID: 1, Symbol: "LUX-LUSD", Type: Limit, Side: Buy,
			Price: 5, PriceUnits: PriceInt(5 * PriceMultiplier),
			Size: 10, SizeUnits: big.NewInt(10),
			User: "maker", UserID: "maker", Timestamp: time.Unix(0, 1),
		}
		if ob.ConsensusAddOrder(maker) == 0 {
			t.Fatalf("side=%d: the resting bid was rejected", side)
		}

		// A LIMIT order at 100. Against a bid of 5 it is not marketable on the sell
		// side and must produce no fills.
		taker := &Order{
			ID: 2, Symbol: "LUX-LUSD", Type: Limit, Side: s,
			Price: 100, PriceUnits: PriceInt(100 * PriceMultiplier),
			Size: 10, SizeUnits: big.NewInt(10),
			User: "taker", UserID: "taker", Timestamp: time.Unix(0, 2),
		}
		fills, err := ob.SubmitMarketable(taker)
		if err != nil {
			continue // refused outright: a decided outcome, nothing crossed.
		}
		if len(fills) == 0 {
			continue // the limit was applied.
		}
		if s == Buy {
			continue // a BUY at 100 legitimately does not touch the bid side; it swept asks.
		}
		t.Fatalf("DEFECT: a LIMIT order crossed past its own limit price because its Side is outside\n"+
			"  {Buy(%d), Sell(%d)}.\n"+
			"  Side=%d, limit 100, resting bid at 5 — %d fill(s) at price %v.\n"+
			"  tryMatchImmediateLocked (pkg/dex/orderbook.go) reads Side twice with different rules:\n"+
			"  the opposite-tree selection is `if Side == Buy { asks } else { bids }` (total — this\n"+
			"  order sweeps the BID side, i.e. it is treated as a SELL), while the price guard is\n"+
			"  `if Side == Buy && … break` followed by `if Side == Sell && … break` (partial — this\n"+
			"  order matches NEITHER arm, so no bound is ever applied). The order therefore crosses\n"+
			"  like a Market order while declaring itself Limit.\n"+
			"  This is reachable from the wire: zapwire carries side as a raw uint8 and\n"+
			"  pkg/dchain applySubmit casts it straight to dex.Side with no domain check.\n"+
			"  FIX: make the price guard total — bound the cross by the same predicate that chose the\n"+
			"  tree — and refuse a Side outside the domain at decode.",
			int(Buy), int(Sell), side, len(fills), fills[0].Price)
	}
}

// TestTradeAttributionIsTotalInTakerSide pins the second half of the same field.
// A fill names a buy leg and a sell leg, and which of the two is the incoming
// order is decided by `if order.Side == Buy { … } else { … }` — total. Settlement
// downstream then asks the mirror question from the taker side. The two must
// agree for every value the field can hold, because a disagreement makes a fill
// name the wrong account.
func TestTradeAttributionIsTotalInTakerSide(t *testing.T) {
	for side := 0; side < 256; side++ {
		s := Side(side)

		ob := NewOrderBook("LUX-LUSD")
		maker := &Order{
			ID: 1, Symbol: "LUX-LUSD", Type: Limit, Side: Buy,
			Price: 5, PriceUnits: PriceInt(5 * PriceMultiplier),
			Size: 10, SizeUnits: big.NewInt(10),
			User: "maker", UserID: "maker", Timestamp: time.Unix(0, 1),
		}
		if ob.ConsensusAddOrder(maker) == 0 {
			t.Fatalf("side=%d: the resting bid was rejected", side)
		}
		taker := &Order{
			ID: 2, Symbol: "LUX-LUSD", Type: Market, Side: s,
			Size: 10, SizeUnits: big.NewInt(10),
			User: "taker", UserID: "taker", Timestamp: time.Unix(0, 2),
		}
		fills, err := ob.SubmitMarketable(taker)
		if err != nil || len(fills) == 0 {
			continue
		}
		f := fills[0]
		// The maker rested on the BUY side, so it must be the buy leg of the fill,
		// and it must never be named as the taker.
		if f.BuyOrder != maker.ID || f.SellOrder != taker.ID {
			t.Fatalf("DEFECT: at Side=%d a fill against a resting BID attributed BuyOrder=%d SellOrder=%d;\n"+
				"  the resting maker is %d and the incoming taker is %d. Trade attribution reads\n"+
				"  `order.Side == Buy` while the settlement layer asks the mirror question from the\n"+
				"  taker side (pkg/dchain settle.go makerSettleKey uses `takerSide == sideSell`), so an\n"+
				"  out-of-domain byte makes the two name different legs and a fill credits the wrong\n"+
				"  account or no account at all.",
				side, f.BuyOrder, f.SellOrder, maker.ID, taker.ID)
		}
		if f.TakerSide != s {
			t.Fatalf("at Side=%d the fill records TakerSide=%d", side, int(f.TakerSide))
		}
	}
}

// ---------------------------------------------------------------------------
// The verifier's own contract.
// ---------------------------------------------------------------------------

// TestBatchVerifyRejectsMalformedSignaturesWithoutRecovering pins that the batch
// verifier fails CLOSED on every malformed signature shape, and that a
// zero-address Sender cannot be matched by a recovery that failed. A verifier
// that returns "valid" for an unrecoverable signature against a zero Sender would
// accept every order from the zero account.
func TestBatchVerifyRejectsMalformedSignaturesWithoutRecovering(t *testing.T) {
	priv, addr := newSigner(t, "dave")

	cases := []struct {
		name string
		mut  func(o *SignedOrder)
	}{
		{"zero signature", func(o *SignedOrder) { o.Sig = [65]byte{} }},
		{"recovery id 2", func(o *SignedOrder) { o.Sig[64] = 2 }},
		{"recovery id 27 (legacy v)", func(o *SignedOrder) { o.Sig[64] = 27 }},
		{"recovery id 255", func(o *SignedOrder) { o.Sig[64] = 255 }},
		{"r = 0", func(o *SignedOrder) {
			for i := 0; i < 32; i++ {
				o.Sig[i] = 0
			}
		}},
		{"s = 0", func(o *SignedOrder) {
			for i := 32; i < 64; i++ {
				o.Sig[i] = 0
			}
		}},
		{"r and s = curve order", func(o *SignedOrder) {
			for i := 0; i < 64; i++ {
				o.Sig[i] = 0xff
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			o := baseOrder(addr)
			sign(t, &o, priv)
			c.mut(&o)
			if verifies(t, o) {
				t.Fatalf("DEFECT: BatchVerifyOrders accepted a %s", c.name)
			}
			// And with a zero Sender, so a failed recovery cannot coincidentally match.
			zero := baseOrder(common.Address{})
			sign(t, &zero, priv) // signature is alice's; Sender is the zero address
			c.mut(&zero)
			if verifies(t, zero) {
				t.Fatalf("DEFECT: BatchVerifyOrders accepted a %s against the ZERO Sender address —\n"+
					"  a failed recovery that yields the zero address would authorize every order\n"+
					"  claiming to come from it", c.name)
			}
		})
	}

	// Positive control: the untouched fixture must verify, or the whole table is vacuous.
	ok := baseOrder(addr)
	sign(t, &ok, priv)
	if !verifies(t, ok) {
		t.Fatal("control failed: a correctly signed order does not verify")
	}
}

// TestVerifyOrderPQRefusesEveryWrongSignatureLength pins the strict-PQ verifier's
// stated contract: "wrong-length inputs are refused before any verification work
// runs". The pubkey length is checked against the FIPS 204 constant; the
// SIGNATURE length is only checked for emptiness, so a wrong-length signature
// reaches the ML-DSA verify. That is safe today because the verify refuses it,
// but the refusal is the library's, not this function's, and the doc claims
// otherwise.
func TestVerifyOrderPQRefusesEveryWrongSignatureLength(t *testing.T) {
	o := &SignedOrderPQ{
		Order:  Order{ID: 1, Symbol: "LUX-LUSD", Type: Limit, Side: Buy, Price: 5, Size: 10},
		PubKey: make([]byte, 1952),
		Sig:    make([]byte, 1), // one byte: non-empty, hopelessly wrong length
	}
	o.Sender = AddressFromMLDSAPubKey(o.PubKey)
	if err := VerifyOrderPQ(o); err == nil {
		t.Fatal("DEFECT: VerifyOrderPQ accepted a 1-byte ML-DSA-65 signature")
	}
	// Every length from 0 to the correct size must be refused, and so must longer.
	for _, n := range []int{0, 1, 64, 3292, 3294, 8192} {
		o.Sig = make([]byte, n)
		if err := VerifyOrderPQ(o); err == nil {
			t.Fatalf("DEFECT: VerifyOrderPQ accepted a %d-byte signature (want %d)", n, 3293)
		}
	}
}

// TestARestingOrderKeepsItsSideAcrossPersistence is the fifth and last reader of
// the side field, and the one that turns an untotalled byte into a chain halt
// somebody else's transaction triggers.
//
// A resting order lives in two places: the in-RAM tree the matcher crosses, and
// the order: row the next block rebuilds the book from. Which tree it goes into
// and which side byte gets written are decided by two predicates with OPPOSITE
// defaults:
//
//	addRestingLocked (orderbook.go) : `if Side == Buy { bids } else { asks }`
//	sideToDEX        (persist.go)   : `if s == Sell { Ask } else { Bid }`
//
// For Buy and Sell they agree. For anything else the order rests as an ASK and is
// written as a BID, so the very next block reads it back as a BUY — the same
// order, one block apart, on the opposite side of the book, still holding the
// reserve its original side required.
//
// The property has nothing to do with attacks: an order must come back from disk
// on the side it rested on. This asserts exactly that, for every value the field
// can hold.
func TestARestingOrderKeepsItsSideAcrossPersistence(t *testing.T) {
	// restsInBids is the book's own predicate — the one addRestingLocked and
	// tryMatchImmediateLocked both use to choose a tree.
	restsInBids := func(s Side) bool { return s == Buy }

	for side := 0; side < 256; side++ {
		s := Side(side)
		o := &Order{
			ID: 1, Symbol: "LUX-LUSD", Type: Limit, Side: s,
			PriceUnits: PriceInt(5 * PriceMultiplier), Price: 5,
			SizeUnits: big.NewInt(10), Size: 10, RemainingSize: 10,
			User: "u", UserID: "u", Timestamp: time.Unix(0, 1), Status: Open,
		}
		back := RowToOrder(OrderToRow(o))
		if restsInBids(back.Side) != restsInBids(o.Side) {
			t.Fatalf("DEFECT: a resting order CHANGES SIDE when it goes through disk.\n"+
				"  Side=%d rests in the %s tree in RAM (addRestingLocked: `Side == Buy ? bids : asks`)\n"+
				"  but comes back from its order: row as Side=%d, which rests in the %s tree\n"+
				"  (persist.sideToDEX: `s == Sell ? Ask : Bid`, then dexToSide: `s == Ask ? Sell : Buy`).\n"+
				"  The two predicates have OPPOSITE defaults, so they agree on Buy(%d) and Sell(%d) and on\n"+
				"  nothing else.\n"+
				"  On the D-Chain this is a halt: the order's custody reserve was taken for the side it\n"+
				"  rested on, the next block rebuilds it on the other side, and the first counterparty to\n"+
				"  cross it makes settlement spend a reserve that was never locked — ErrInsufficientLocked,\n"+
				"  which aborts the block rather than rejecting the tx. The transaction that halts the\n"+
				"  chain belongs to the counterparty, not to whoever planted the order.\n"+
				"  FIX: one predicate for 'which side of the book is this', used by the tree choice and by\n"+
				"  the row encoding alike — and a side outside the domain refused at decode.",
				side, treeName(restsInBids(o.Side)), int(back.Side), treeName(restsInBids(back.Side)),
				int(Buy), int(Sell))
		}
	}
}

// treeName renders which half of the book a predicate chose.
func treeName(bids bool) string {
	if bids {
		return "bid"
	}
	return "ask"
}
