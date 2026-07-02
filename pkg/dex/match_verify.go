// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"bytes"
	"sort"
	"strconv"
)

// match_verify.go is the GPU shadow gate for the consensus match path. It is the
// ExecuteVerified pattern applied to order matching: the pure-Go matcher
// (SubmitMarketable -> tryMatchImmediateLocked) is the AUTHORITY whose result is
// always committed; when a node opts into EngineGPUVerified, the GPU kernel is
// ALSO dispatched on the exact pre-cross inputs the authority matched, and its
// output is byte-compared against the authority. The committed bytes are never
// the GPU's — they are always the CPU authority's — so:
//
//   1. A GPU result that differs from the CPU by a single byte is recorded as a
//      divergence and discarded. It is structurally impossible to commit.
//   2. A validator running EngineGPUVerified and a validator running EngineCPU
//      commit byte-identical state for the same ordered block, so a mixed
//      GPU/CPU validator set cannot fork.
//
// The GPU here is a SPEED-AND-ASSURANCE overlay that is continuously proven
// equal to the oracle at Verify, never a trusted replacement for it.

// dexGPUMatch is the GPU matcher the shadow gate dispatches. It is a package
// var ONLY so the fail-closed negative test can substitute a deliberately
// divergent matcher and assert that a bad GPU result can never reach committed
// state. In production it is always MatchOrderGPU (the real per-platform GPU
// dispatch, which itself falls back to the CPU oracle on hosts without a GPU).
var dexGPUMatch = MatchOrderGPU

// dexShadowInput is the flat snapshot of a single marketable cross: the taker
// projected to the GPU ABI, the opposite side of the book in the EXACT price-
// time priority order the authority consumes it, and the deterministic trade-id
// base / timestamp the authority stamped. It is captured under ob.mu, before the
// cross mutates any maker, so the GPU runs on byte-identical inputs.
type dexShadowInput struct {
	incoming  DEXOrder
	book      []DEXOrder
	indices   []uint32
	tradeBase uint64
	timestamp uint64
}

// SubmitMarketableVerified is the consensus entrypoint for a marketable cross.
//
// With EngineCPU it is byte-for-byte SubmitMarketable (zero GPU overhead, the
// default). With EngineGPUVerified it captures the pre-cross inputs atomically
// with the match, runs the authority, then dispatches the GPU kernel on the
// captured inputs and byte-compares — recording the outcome and ALWAYS returning
// the authority's fills. The GPU cannot change the returned (committed) value.
func (ob *OrderBook) SubmitMarketableVerified(order *Order, engine VerifyEngine) ([]Trade, error) {
	if engine != EngineGPUVerified {
		return ob.SubmitMarketable(order)
	}
	fills, shadow, err := ob.submitMarketable(order, true)
	if err != nil || shadow == nil {
		return fills, err
	}
	// The shadow runs on captured copies; it never touches ob and never mutates
	// `fills`. A panic in experimental GPU glue must not take down consensus, so
	// it is contained — a failed shadow is, at worst, a missed verification, and
	// the authority result still stands.
	func() {
		defer func() {
			if r := recover(); r != nil {
				emitDivergence(MatchDivergence{Kind: "dex", Reason: "gpu_error",
					Detail: "panic in shadow: " + toStr(r)})
			}
		}()
		runDEXShadow(shadow, fills, order.Side)
	}()
	return fills, err
}

// runDEXShadow is the gate. It runs the CPU oracle and the GPU kernel on the
// captured pre-cross inputs and:
//
//   - asserts the GPU flat result == the CPU flat result, BYTE-IDENTICAL (fills,
//     remaining, and resulting book). This is the total, unconditional
//     consensus-safety gate: it holds for every input, with or without
//     self-trade prevention, market or limit. A failure here is the thing the
//     gate exists to catch — it is recorded and the GPU output discarded.
//   - opportunistically asserts the CPU flat result reproduces the rich
//     authority's committed fills (a fidelity diagnostic; the flat primitive
//     does not model self-trade-prevention, so this can legitimately differ —
//     recorded separately, never a consensus signal).
//
// It commits NOTHING. The caller has already captured the authority's fills.
func runDEXShadow(in *dexShadowInput, committedRich []Trade, takerSide Side) {
	// Two independent deep copies: MatchOrderCPU and MatchOrderGPU both mutate
	// the book slice in place, so each path gets its own.
	bookCPU := append([]DEXOrder(nil), in.book...)
	bookGPU := append([]DEXOrder(nil), in.book...)
	idx := in.indices

	incCPU := in.incoming
	cpuTrades, cpuRem := MatchOrderCPU(&incCPU, bookCPU, idx, in.tradeBase, in.timestamp)

	incGPU := in.incoming
	gpuTrades, gpuRem, gerr := dexGPUMatch(&incGPU, bookGPU, idx, in.tradeBase, in.timestamp)
	if gerr != nil {
		emitDivergence(MatchDivergence{Kind: "dex", Reason: "gpu_error", Detail: gerr.Error()})
		return // fail-closed: the authority result is what was committed
	}

	// --- consensus-safety gate: GPU flat == CPU flat, byte for byte ---
	if reason, ok := flatResultsEqual(cpuTrades, cpuRem, bookCPU, gpuTrades, gpuRem, bookGPU); !ok {
		emitDivergence(MatchDivergence{Kind: "dex", Reason: "gpu_ne_cpu", Detail: reason})
		return // fail-closed
	}
	statVerified.Add(1)

	// --- fidelity diagnostic: does the flat oracle reproduce the rich authority? ---
	if reason, ok := flatReproducesRich(cpuTrades, cpuRem, committedRich, takerSide); !ok {
		emitDivergence(MatchDivergence{Kind: "dex", Reason: "model", Detail: reason})
	}
}

// flatResultsEqual reports whether two flat match results are byte-identical:
// same remaining, same number of trades with byte-identical rows, and same
// resulting book rows. Returns a human-readable reason for the first difference.
func flatResultsEqual(
	aTrades []DEXTrade, aRem uint64, aBook []DEXOrder,
	bTrades []DEXTrade, bRem uint64, bBook []DEXOrder,
) (string, bool) {
	if aRem != bRem {
		return "remaining cpu=" + strconv.FormatUint(aRem, 10) + " gpu=" + strconv.FormatUint(bRem, 10), false
	}
	if len(aTrades) != len(bTrades) {
		return "trade count cpu=" + strconv.Itoa(len(aTrades)) + " gpu=" + strconv.Itoa(len(bTrades)), false
	}
	for i := range aTrades {
		if !bytes.Equal(EncodeTrade(aTrades[i]), EncodeTrade(bTrades[i])) {
			return "trade[" + strconv.Itoa(i) + "] bytes differ", false
		}
	}
	if len(aBook) != len(bBook) {
		return "book rows cpu=" + strconv.Itoa(len(aBook)) + " gpu=" + strconv.Itoa(len(bBook)), false
	}
	for i := range aBook {
		if !bytes.Equal(EncodeRow(aBook[i]), EncodeRow(bBook[i])) {
			return "book[" + strconv.Itoa(i) + "] bytes differ", false
		}
	}
	return "", true
}

// flatReproducesRich reports whether the flat CPU oracle's fills match the rich
// authority's committed fills, byte for byte after canonical projection. The
// flat primitive does not model self-trade prevention (the rich matcher skips-
// and-cancels a self-maker leg), so a self-cross legitimately differs; that is
// returned as a non-fatal reason and recorded as a fidelity diagnostic, never a
// consensus divergence.
func flatReproducesRich(flat []DEXTrade, flatRem uint64, rich []Trade, takerSide Side) (string, bool) {
	if len(flat) != len(rich) {
		return "fill count flat=" + strconv.Itoa(len(flat)) + " rich=" + strconv.Itoa(len(rich)), false
	}
	for i := range rich {
		want := TradeToRow(rich[i], takerSide)
		if !bytes.Equal(EncodeTrade(want), EncodeTrade(flat[i])) {
			return "fill[" + strconv.Itoa(i) + "] flat!=rich", false
		}
	}
	return "", true
}

// captureDEXShadowLocked snapshots the opposite side of the book for the GPU
// shadow, in the EXACT order tryMatchImmediateLocked consumes it: best price
// first, FIFO within a price level. In the consensus path a resting order's id
// is blockDeterministicID(height,txIndex), monotonic with arrival, so FIFO ==
// ascending order-id — which is the canonical (price, then id) book ordering.
//
// The caller holds ob.mu (write). This is a pure read of ob.Orders; it never
// mutates the book and never takes ob.mu (it is already held).
func (ob *OrderBook) captureDEXShadowLocked(order *Order) *dexShadowInput {
	wantSide := Sell
	if order.Side == Sell {
		wantSide = Buy
	}
	rows := make([]DEXOrder, 0, len(ob.Orders))
	for _, o := range ob.Orders {
		if o == nil || o.Side != wantSide {
			continue
		}
		// Only orders that can rest-and-match: an order already filled or
		// cancelled is not a maker the cross would touch.
		if o.Status != Open && o.Status != PartiallyFilled && o.Status != "" {
			continue
		}
		rows = append(rows, OrderToRow(o))
	}
	sortShadowRows(rows, order.Side)
	idx := make([]uint32, len(rows))
	for i := range idx {
		idx[i] = uint32(i)
	}
	return &dexShadowInput{
		incoming:  orderToFlatTaker(order),
		book:      rows,
		indices:   idx,
		tradeBase: ob.LastTradeID + 1,
		timestamp: uint64(order.Timestamp.UnixNano()),
	}
}

// sortShadowRows imposes price-time priority from the TAKER's perspective: a buy
// taker consumes asks ascending in price; a sell taker consumes bids descending
// in price; ties broken by ascending order-id (FIFO in the consensus path).
func sortShadowRows(rows []DEXOrder, takerSide Side) {
	sort.SliceStable(rows, func(i, j int) bool {
		pi, pj := rows[i].Price, rows[j].Price
		if pi.Integer != pj.Integer {
			if takerSide == Buy {
				return pi.Integer < pj.Integer // best ask = lowest price first
			}
			return pi.Integer > pj.Integer // best bid = highest price first
		}
		if pi.Fraction != pj.Fraction {
			if takerSide == Buy {
				return pi.Fraction < pj.Fraction
			}
			return pi.Fraction > pj.Fraction
		}
		return rows[i].OrderID < rows[j].OrderID // FIFO within a level
	})
}

// orderToFlatTaker projects the incoming marketable order to the flat taker ABI.
// A market order carries no price and sweeps the book unconditionally; the flat
// matcher gates every fill on pricesCross, so a market taker is given the maximal
// price so it crosses every resting level — mirroring the rich market sweep.
func orderToFlatTaker(order *Order) DEXOrder {
	t := DEXOrder{
		OrderID:   order.ID,
		UserID:    userToUint64(takerUserStr(order)),
		Quantity:  floatToFixedQty(order.Size),
		Timestamp: uint64(order.Timestamp.UnixNano()),
		Side:      sideToDEX(order.Side),
		Type:      typeToDEX(order.Type),
		Status:    DEXStatusOpen,
	}
	t.Remaining = t.Quantity
	if order.Type == Market {
		t.Price = DEXPrice{Integer: ^uint64(0), Fraction: ^uint64(0)}
	} else {
		t.Price = floatToFixedPrice(order.Price)
	}
	t.ClearPadding()
	return t
}

func takerUserStr(order *Order) string {
	if order.User != "" {
		return order.User
	}
	return order.UserID
}

// toStr renders a recovered panic value without pulling fmt into the hot path's
// happy case (it is only reached on the contained-panic branch).
func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return "non-string panic"
}
