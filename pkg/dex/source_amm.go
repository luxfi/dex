// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"github.com/luxfi/dex/pkg/lx"
)

// source_amm.go is the V2/V3 AMM liquidity source — the router's FALLTHROUGH when
// the OrderBook has insufficient depth. The AMM pools are EVM contracts in the SAME
// cEVM, so their state is read in-process and the swap executes in-process
// (deterministic constant-product math; every validator replays identically).
//
// dex is store-agnostic and knows nothing of EVM contract layout, so the
// concrete reserve read+write is injected via AMMPool: the precompile implements it
// over the real V2/V3 pool contract storage; a test implements it over an in-Store
// reserve row. The constant-product MATH (lx.ConstantProductOut, 128-bit exact) and
// the routing/settlement live HERE, in one place, so both homes price and execute
// an AMM leg identically. This is the seam an RFQ/external-CEX source would later
// slot into as well — a different reserve/quote backend behind the same Source
// contract.

// AMMPool is the precompile-injected view+executor of one V2/V3 pool, keyed by the
// market poolID. dex reads reserves to price/route and writes them back after a
// fill; the implementer maps (base,quote) to the pool's token0/token1 reserves in
// cEVM contract storage. All methods are pure replayable functions of prior state —
// NO live external call.
type AMMPool interface {
	// Reserves returns the pool's (baseReserve, quoteReserve) for the market, and
	// ok=false when no pool exists for this poolID (the router then skips the AMM).
	Reserves(db Store, poolID [32]byte) (baseReserve, quoteReserve uint64, ok bool, err error)

	// SetReserves writes the pool's post-fill reserves back. dex calls this once
	// after an AMM fill with the new (baseReserve, quoteReserve).
	SetReserves(db Store, poolID [32]byte, baseReserve, quoteReserve uint64) error

	// FeeBps is the pool's swap fee in basis points (e.g. 30 = 0.30%). Applied to
	// the input before the constant-product step, exactly as a V2 pool does.
	FeeBps() uint32
}

// AMMSource routes against a single AMM pool view. It holds the injected pool but
// no mutable state; reserves live in the Store (via the pool view).
type AMMSource struct {
	pool AMMPool
}

// NewAMMSource builds the AMM source over a concrete pool view.
func NewAMMSource(pool AMMPool) *AMMSource { return &AMMSource{pool: pool} }

// Kind identifies this as the AMM source.
func (*AMMSource) Kind() SourceKind { return SourceAMM }

// QuoteSwap prices up to remainingIn against the constant-product curve WITHOUT
// mutating reserves. Ok=false when no pool exists or the curve yields no output.
func (s *AMMSource) QuoteSwap(db Store, req SwapRequest, remainingIn uint64) (Quote, error) {
	baseR, quoteR, ok, err := s.pool.Reserves(db, req.PoolID)
	if err != nil {
		return Quote{}, err
	}
	if !ok || baseR == 0 || quoteR == 0 {
		return Quote{}, nil
	}
	inUsed, outGot := s.curveOut(req.Side, baseR, quoteR, remainingIn)
	if inUsed == 0 || outGot == 0 {
		return Quote{}, nil
	}
	return Quote{AmountInUsed: inUsed, AmountOut: outGot, Ok: true}, nil
}

// ExecuteSwap fills up to remainingIn against the curve, writes the new reserves
// back, and records the taker's proceeds as a C-surface credit in the journal. The
// taker's input was locked once by the top-level ExecuteSwap; the AMM leg's input
// is credited to the POOL's reserve (the pool gains the input, the taker gains the
// output). Conservation: input added to the pool's reserve == input removed from the
// taker's lock; output removed from the pool's reserve == output credited to the
// taker. Both legs balance to the unit.
func (s *AMMSource) ExecuteSwap(db Store, req SwapRequest, remainingIn uint64, j *Journal) (Fill, uint64, uint64, error) {
	baseR, quoteR, ok, err := s.pool.Reserves(db, req.PoolID)
	if err != nil {
		return Fill{}, 0, 0, err
	}
	if !ok || baseR == 0 || quoteR == 0 {
		return Fill{}, 0, 0, nil
	}
	inUsed, outGot := s.curveOut(req.Side, baseR, quoteR, remainingIn)
	if inUsed == 0 || outGot == 0 {
		return Fill{}, 0, 0, nil
	}

	// Apply the fill to the reserves. A BUY spends quote (in) and removes base (out);
	// a SELL spends base (in) and removes quote (out).
	var newBase, newQuote uint64
	var baseDelta, quoteDelta uint64
	if req.Side == lx.Buy {
		newBase = baseR - outGot   // base leaves the pool to the taker
		newQuote = quoteR + inUsed // quote enters the pool from the taker
		baseDelta, quoteDelta = outGot, inUsed
	} else {
		newBase = baseR + inUsed   // base enters the pool from the taker
		newQuote = quoteR - outGot // quote leaves the pool to the taker
		baseDelta, quoteDelta = inUsed, outGot
	}
	if err := s.pool.SetReserves(db, req.PoolID, newBase, newQuote); err != nil {
		return Fill{}, 0, 0, err
	}

	// The taker's input lock must be spent for the AMM leg (it went to the pool). The
	// top-level lock holds the taker's full input in the 0x9999 vault; the AMM leg's
	// input is consumed from that vault-held amount and NOT refunded. We spend the
	// taker's LOCKED input ledger units so the conservation invariant (lock == fills
	// spent + refund) holds across BOTH OrderBook and AMM legs uniformly.
	inAsset := req.Quote
	outAsset := req.Base
	if req.Side == lx.Sell {
		inAsset = req.Base
		outAsset = req.Quote
	}
	if serr := SpendLocked(db, req.TakerUser, inAsset, inUsed); serr != nil {
		return Fill{}, 0, 0, serr
	}
	// Credit the taker's dex AVAILABLE with the proceeds — SYMMETRIC with the OrderBook
	// source (whose SettleFills credits the taker's available). This makes the D-surface
	// settlement consistent across BOTH sources: after any swap the taker's available
	// holds exactly the proceeds the journal records, so a caller that drains the
	// transient by the journal amount drains correctly regardless of which source(s)
	// filled. Conservation holds across (ledger + pool reserves): the input left the
	// taker's locked into the pool's base reserve; the output left the pool's quote
	// reserve into the taker's available — both legs balance to the unit.
	if cerr := CreditAvailable(db, req.TakerUser, outAsset, outGot); cerr != nil {
		return Fill{}, 0, 0, cerr
	}
	// Mirror the proceeds to the C surface: the 0x9999 vault releases outGot of the
	// output asset to the taker's EVM balance.
	j.CreditTakerOutput(outAsset, outGot)

	return Fill{AMMBaseDelta: baseDelta, AMMQuoteDelta: quoteDelta, Source: SourceAMM}, inUsed, outGot, nil
}

// curveOut computes (inUsed, outGot) for filling up to remainingIn against the
// pool's (baseReserve, quoteReserve), with the fee applied to the input. For a SELL
// the input is base, the output is quote; for a BUY the input is quote, the output
// is base. The whole remainingIn is consumed (an AMM has no "depth limit" — it just
// prices worse as the input grows), so inUsed == remainingIn whenever outGot > 0.
func (s *AMMSource) curveOut(side lx.Side, baseR, quoteR, remainingIn uint64) (inUsed, outGot uint64) {
	if remainingIn == 0 {
		return 0, 0
	}
	inAfterFee := applyFee(remainingIn, s.pool.FeeBps())
	if inAfterFee == 0 {
		return 0, 0
	}
	if side == lx.Sell {
		// SELL base for quote: rx=baseReserve, ry=quoteReserve, amount=base in.
		outGot = lx.ConstantProductOut(baseR, quoteR, inAfterFee)
	} else {
		// BUY base with quote: rx=quoteReserve, ry=baseReserve, amount=quote in.
		outGot = lx.ConstantProductOut(quoteR, baseR, inAfterFee)
	}
	if outGot == 0 {
		return 0, 0
	}
	return remainingIn, outGot
}

// applyFee reduces amount by feeBps basis points (floor), exactly as a V2 pool
// withholds its fee from the input before the constant-product step. 0 bps = no fee.
func applyFee(amount uint64, feeBps uint32) uint64 {
	if feeBps == 0 {
		return amount
	}
	// amount * (10000 - feeBps) / 10000, in 128-bit so the product can't overflow.
	const denom = 10000
	keep := uint64(denom - feeBps)
	return mulDivU64(amount, keep, denom)
}

// mulDivU64 = (a*b)/d computed in 128-bit intermediates so a*b can exceed 2^64
// without truncating (the same shift-subtract long division lx's curve math uses,
// so the fee math shares the curve math's exact-integer overflow discipline). The
// result is the caller's bug if the quotient itself doesn't fit in 64 bits.
func mulDivU64(a, b, d uint64) uint64 {
	if d == 0 {
		return 0
	}
	aLo, aHi := a&0xFFFFFFFF, a>>32
	bLo, bHi := b&0xFFFFFFFF, b>>32
	ll := aLo * bLo
	lh := aLo * bHi
	hl := aHi * bLo
	hh := aHi * bHi
	mid := (ll >> 32) + (lh & 0xFFFFFFFF) + (hl & 0xFFFFFFFF)
	lo := (ll & 0xFFFFFFFF) | (mid << 32)
	hi := hh + (lh >> 32) + (hl >> 32) + (mid >> 32)
	if hi == 0 {
		return lo / d
	}
	var quotient, remainder uint64
	for i := 127; i >= 0; i-- {
		remainder <<= 1
		var bit uint64
		if i >= 64 {
			bit = (hi >> uint(i-64)) & 1
		} else {
			bit = (lo >> uint(i)) & 1
		}
		remainder |= bit
		if remainder >= d {
			remainder -= d
			if i < 64 {
				quotient |= uint64(1) << uint(i)
			}
		}
	}
	return quotient
}
