// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package rfq

import (
	"errors"
	"math/big"
	"time"

	"github.com/luxfi/dex/pkg/htlc"
	"github.com/luxfi/geth/common"
)

// session.go is the swap STATE MACHINE: the WHAT of each transition (role
// bindings, the cross-leg hashlock, the timeout-ordering invariant), kept
// orthogonal to the WHO (signature authentication lives in coordinator.go). A
// SwapSession holds only public values — the accepted quote, the committed
// hashlock, the two on-chain lock proofs, and (only AFTER it is revealed on chain
// A) the preimage. It holds NO private key of any asset and has no method that
// moves funds; it only records and validates relayed facts.

// State is the swap lifecycle. The happy path is Quoted → … → Settled; Refunded
// and Aborted are the off-path terminals.
type State uint8

const (
	StateQuoted            State = iota // a firm quote is accepted; no funds at risk
	StateHashlockCommitted              // taker committed h = SHA256(s)
	StateLegALocked                     // maker locked HTLC_A (recipient = taker)
	StateLegBLocked                     // taker locked HTLC_B (recipient = maker), T_B > T_A + Δ
	StateClaimed                        // taker claimed HTLC_A, revealing s
	StateSettled                        // maker counter-claimed HTLC_B with s
	StateRefunded                       // a locked leg refunded to its owner (off path)
	StateAborted                        // abandoned before any lock (off path)
)

func (s State) String() string {
	switch s {
	case StateQuoted:
		return "quoted"
	case StateHashlockCommitted:
		return "hashlock-committed"
	case StateLegALocked:
		return "leg-a-locked"
	case StateLegBLocked:
		return "leg-b-locked"
	case StateClaimed:
		return "claimed"
	case StateSettled:
		return "settled"
	case StateRefunded:
		return "refunded"
	case StateAborted:
		return "aborted"
	default:
		return "unknown"
	}
}

// Session-transition errors. Every one is a deterministic reject that leaves the
// session unchanged; none can move funds, because the session has none to move.
var (
	ErrWrongState       = errors.New("rfq: transition not allowed from current state")
	ErrLegMismatch      = errors.New("rfq: leg chain/asset/amount does not match the accepted quote")
	ErrRoleBinding      = errors.New("rfq: leg recipient/refund does not bind the agreed parties")
	ErrTimeoutOrdering  = errors.New("rfq: submitted leg violates T_B > T_A + Δ")
	ErrPreimageMismatch = errors.New("rfq: revealed preimage does not match the committed hashlock")
)

// LegProof is the relayed attestation of one on-chain HTLC lock. The coordinator
// does NOT verify it against a chain (it runs no light client); it records the
// declared terms and checks they bind the agreed parties and amounts. Each party
// independently verifies the real on-chain lock before acting (spec step 4).
type LegProof struct {
	Chain     ChainKind
	Asset     string
	Recipient string // who may claim with the preimage
	Refund    string // who is refunded after Timeout
	Amount    *big.Int
	Timeout   uint64 // absolute unix seconds (HTLC locktime)
	TxRef     string // chain tx id / proof reference; relayed, not interpreted
}

// SwapSession is one swap's recorded facts and current state. No field is a
// private key of any asset; Secret is populated only once it is already public on
// chain A (the claim that revealed it), so storing it relays — never custodies.
type SwapSession struct {
	ID       SessionID
	Quote    Quote
	Maker    common.Address // maker's coordination identity (= Quote.Maker)
	User     common.Address // taker's coordination identity (recovered at commit)
	Hashlock [32]byte       // h = SHA256(s), committed by the taker
	UserA    string         // taker payout on ChainA  (HTLC_A recipient)
	UserB    string         // taker refund on ChainB  (HTLC_B refund)
	State    State
	LegA     *LegProof
	LegB     *LegProof
	Secret   *htlc.Secret  // revealed preimage; nil until Claimed
	delta    time.Duration // required Δ floor for this pair
}

// newSession builds a session in StateQuoted from an accepted quote and the
// taker's recovered coordination identity.
func newSession(id SessionID, q Quote, user common.Address) *SwapSession {
	return &SwapSession{
		ID:    id,
		Quote: q,
		Maker: q.Maker,
		User:  user,
		State: StateQuoted,
		delta: Delta(q.Pair.ChainA, q.Pair.ChainB),
	}
}

// commit records the taker's hashlock and chain endpoints: Quoted →
// HashlockCommitted.
func (s *SwapSession) commit(c CommitPayload) error {
	if s.State != StateQuoted {
		return ErrWrongState
	}
	s.Hashlock = c.Hashlock
	s.UserA = c.UserA
	s.UserB = c.UserB
	s.State = StateHashlockCommitted
	return nil
}

// lockA records the maker's HTLC_A: HashlockCommitted → LegALocked. The maker
// locks first; recipient MUST be the taker (UserA) and refund the maker (MakerA),
// for the quoted asset and amount on chain A.
func (s *SwapSession) lockA(p LegProof) error {
	if s.State != StateHashlockCommitted {
		return ErrWrongState
	}
	if p.Chain != s.Quote.Pair.ChainA || p.Asset != s.Quote.Pair.AssetA || !amountEq(p.Amount, s.Quote.AmountA) {
		return ErrLegMismatch
	}
	if p.Recipient != s.UserA || p.Refund != s.Quote.MakerA {
		return ErrRoleBinding
	}
	leg := p
	s.LegA = &leg
	s.State = StateLegALocked
	return nil
}

// lockB records the taker's HTLC_B: LegALocked → LegBLocked. The taker locks
// second; recipient MUST be the maker (MakerB) and refund the taker (UserB), and
// the timeout MUST clear the ordering invariant T_B > T_A + Δ.
func (s *SwapSession) lockB(p LegProof) error {
	if s.State != StateLegALocked {
		return ErrWrongState
	}
	if p.Chain != s.Quote.Pair.ChainB || p.Asset != s.Quote.Pair.AssetB || !amountEq(p.Amount, s.Quote.AmountB) {
		return ErrLegMismatch
	}
	if p.Recipient != s.Quote.MakerB || p.Refund != s.UserB {
		return ErrRoleBinding
	}
	if !OrderingHolds(s.LegA.Timeout, p.Timeout, s.delta) {
		return ErrTimeoutOrdering
	}
	leg := p
	s.LegB = &leg
	s.State = StateLegBLocked
	return nil
}

// claim records the taker's claim of HTLC_A, which revealed s on chain A:
// LegBLocked → Claimed. The preimage MUST hash to the committed hashlock (the
// single cross-leg binding check, via pkg/htlc).
func (s *SwapSession) claim(secret htlc.Secret) error {
	if s.State != StateLegBLocked {
		return ErrWrongState
	}
	if !htlc.VerifyPreimage(secret, s.Hashlock) {
		return ErrPreimageMismatch
	}
	sc := secret
	s.Secret = &sc
	s.State = StateClaimed
	return nil
}

// settle records the maker's counter-claim of HTLC_B using the revealed s:
// Claimed → Settled.
func (s *SwapSession) settle() error {
	if s.State != StateClaimed {
		return ErrWrongState
	}
	s.State = StateSettled
	return nil
}

// refund records an off-path unwind. A leg that was locked refunds to its owner
// (Refunded); a swap abandoned before any lock is Aborted. Both terminals leave
// every party whole — funds return to whoever locked them.
func (s *SwapSession) refund() error {
	switch s.State {
	case StateLegALocked, StateLegBLocked, StateClaimed:
		s.State = StateRefunded
	case StateQuoted, StateHashlockCommitted:
		s.State = StateAborted
	default:
		return ErrWrongState
	}
	return nil
}

// amountEq reports big.Int value equality, treating nil as zero.
func amountEq(a, b *big.Int) bool {
	if a == nil {
		a = big.NewInt(0)
	}
	if b == nil {
		b = big.NewInt(0)
	}
	return a.Cmp(b) == 0
}
