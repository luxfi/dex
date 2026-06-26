// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package rfq is the NON-CUSTODIAL cross-chain swap coordinator — the venue V of
// the Lux HTLC atomic-swap design. Its ONE responsibility is to relay signed
// messages and sequence a swap session between a taker U (the secret-holder) and
// a CEX-backed maker M: it serves a firm-quote book, validates each typed, signed
// state transition (role bindings, the cross-leg hashlock, and the timeout
// ordering T_B > T_A + Δ), and relays the on-chain lock proofs and the revealed
// preimage.
//
// SAFETY PROPERTY (structural, not policy): the coordinator holds NO private key
// of any asset, is the recipient/refund of NO HTLC, and has NO code path that
// transfers funds. It builds on pkg/htlc for the shared SHA-256 hashlock/preimage
// and authenticates every message through the SINGLE signing primitive
// crypto.Ecrecover (see recoverAddr in quote.go) — the same secp256k1
// ecrecover-and-bind the order path and the D-Chain tx gate use. After the
// hashlock-commit step V is
// irrelevant to safety: its crashing or lying can only deny service or relay a
// stale quote, never move funds.
package rfq

import (
	"crypto/ecdsa"
	"errors"
	"math/big"
	"sync"
	"time"

	"github.com/luxfi/crypto"
	"github.com/luxfi/dex/pkg/htlc"
	"github.com/luxfi/geth/common"
)

// Coordination errors at the relay boundary.
var (
	ErrBadSignature   = errors.New("rfq: message signature invalid or signer is not the expected party")
	ErrUnknownSession = errors.New("rfq: no such session")
	ErrWrongAuthor    = errors.New("rfq: message author is not the party this transition expects")
)

// SessionID is the content address of a swap session: the digest of the taker's
// hashlock commit. Both parties derive the same id from the same commit.
type SessionID [32]byte

// payload is a value that contributes a domain-separated signing digest. Every
// message the coordinator authenticates is one of these, signed by its author.
type payload interface{ digest() [32]byte }

// Signed wraps a payload with its author's secp256k1 signature over digest().
// One envelope for every message type — the maker and the taker sign with the
// same scheme the quote uses.
type Signed[T payload] struct {
	Body T
	Sig  [65]byte
}

// Sign attaches priv's signature over body.digest(). The maker/taker client half
// of the one signing scheme; the coordinator only recovers.
func Sign[T payload](body T, priv *ecdsa.PrivateKey) (Signed[T], error) {
	d := body.digest()
	sig, err := crypto.Sign(d[:], priv)
	if err != nil {
		return Signed[T]{}, err
	}
	var m Signed[T]
	m.Body = body
	copy(m.Sig[:], sig)
	return m, nil
}

// signer recovers the secp256k1 address that signed m, through the single
// verification primitive shared with the order and D-Chain paths.
func signer[T payload](m Signed[T]) (common.Address, error) {
	d := m.Body.digest()
	addr, err := recoverAddr(d, m.Sig[:])
	if err != nil {
		return common.Address{}, errors.Join(ErrBadSignature, err)
	}
	return addr, nil
}

// CommitPayload is the taker's hashlock commit (spec step 2): it names the
// accepted quote, the hashlock h = SHA256(s), the taker's chain endpoints, and
// the taker's asserted coordination identity. Its digest IS the session id.
type CommitPayload struct {
	QuoteID  QuoteID
	Hashlock [32]byte
	UserA    string         // taker payout on ChainA (HTLC_A recipient)
	UserB    string         // taker refund on ChainB (HTLC_B refund)
	User     common.Address // asserted taker identity; the signature must recover to it
}

func (c CommitPayload) digest() [32]byte {
	return newEnc().
		bytes32(c.QuoteID).bytes32(c.Hashlock).
		str(c.UserA).str(c.UserB).addr(c.User).
		hash(domainCommit)
}

// SessionID is the session this commit opens (its own content address).
func (c CommitPayload) SessionID() SessionID { return SessionID(c.digest()) }

// LegPayload carries one lock proof for a named session. Leg distinguishes the
// maker's HTLC_A (0) from the taker's HTLC_B (1) so a proof for one leg can never
// be relayed as the other.
type LegPayload struct {
	Session SessionID
	Leg     uint8
	Proof   LegProof
}

func (p LegPayload) digest() [32]byte {
	return newEnc().
		bytes32(p.Session).u8(p.Leg).
		u8(uint8(p.Proof.Chain)).str(p.Proof.Asset).
		str(p.Proof.Recipient).str(p.Proof.Refund).
		big(p.Proof.Amount).u64(p.Proof.Timeout).str(p.Proof.TxRef).
		hash(domainLeg)
}

// ClaimPayload is the taker's report of its HTLC_A claim, carrying the now-public
// preimage for the coordinator to relay to the maker.
type ClaimPayload struct {
	Session SessionID
	Secret  htlc.Secret
}

func (p ClaimPayload) digest() [32]byte {
	return newEnc().bytes32(p.Session).bytes32(p.Secret).hash(domainClaim)
}

// SettlePayload is the maker's report of its HTLC_B counter-claim.
type SettlePayload struct {
	Session SessionID
	TxRef   string
}

func (p SettlePayload) digest() [32]byte {
	return newEnc().bytes32(p.Session).str(p.TxRef).hash(domainSettle)
}

// RefundPayload is either party's report of an off-path unwind.
type RefundPayload struct {
	Session SessionID
	TxRef   string
}

func (p RefundPayload) digest() [32]byte {
	return newEnc().bytes32(p.Session).str(p.TxRef).hash(domainRefund)
}

// Coordinator ties the quote book to live swap sessions. It has NO key field of
// any kind and no fund-moving method — see TestNonCustody. Every method
// authenticates the signed message, advances the session state machine, and
// returns only what the venue relays onward.
type Coordinator struct {
	book *QuoteBook

	mu       sync.Mutex
	sessions map[SessionID]*SwapSession
}

// NewCoordinator returns a coordinator with an empty quote book.
func NewCoordinator() *Coordinator {
	return &Coordinator{book: NewQuoteBook(), sessions: map[SessionID]*SwapSession{}}
}

// Book exposes the quote book so a maker client can publish (verified) quotes.
func (c *Coordinator) Book() *QuoteBook { return c.book }

// PublishQuote inserts a maker-signed quote into the book (verifying the maker
// signature). This is the venue side of spec step 1.
func (c *Coordinator) PublishQuote(q Quote, now time.Time) (QuoteID, error) {
	return c.book.Insert(q, now)
}

// RequestQuote serves the live quotes for a pair that can cover size (spec step
// 1, taker side). It moves nothing and signs nothing.
func (c *Coordinator) RequestQuote(pair Pair, size *big.Int, now time.Time) []Quote {
	return c.book.Match(pair, size, now)
}

// CommitHashlock opens a session from a taker-signed commit (spec step 2): it
// verifies the taker signature binds the asserted identity, looks up the accepted
// quote and requires it still live, and records the hashlock. Returns the session
// id the venue relays (with h and the quote) to the maker.
func (c *Coordinator) CommitHashlock(m Signed[CommitPayload], now time.Time) (SessionID, error) {
	addr, err := signer(m)
	if err != nil {
		return SessionID{}, err
	}
	if addr != m.Body.User {
		return SessionID{}, ErrBadSignature
	}
	q, ok := c.book.Get(m.Body.QuoteID)
	if !ok {
		return SessionID{}, ErrUnknownQuote
	}
	if !q.live(now) {
		return SessionID{}, ErrQuoteExpired
	}
	id := m.Body.SessionID()
	sess := newSession(id, q, addr)
	if err := sess.commit(m.Body); err != nil {
		return SessionID{}, err
	}
	c.mu.Lock()
	c.sessions[id] = sess
	c.mu.Unlock()
	return id, nil
}

// SubmitLegA records the maker's HTLC_A (spec step 3). The message MUST be signed
// by the quote's maker. Returns the proof the venue relays to the taker.
func (c *Coordinator) SubmitLegA(m Signed[LegPayload]) (LegProof, error) {
	return c.applyLeg(m, 0, func(s *SwapSession) common.Address { return s.Maker }, (*SwapSession).lockA)
}

// SubmitLegB records the taker's HTLC_B (spec step 4). The message MUST be signed
// by the session's taker, and the proof MUST satisfy T_B > T_A + Δ. Returns the
// proof the venue relays to the maker.
func (c *Coordinator) SubmitLegB(m Signed[LegPayload]) (LegProof, error) {
	return c.applyLeg(m, 1, func(s *SwapSession) common.Address { return s.User }, (*SwapSession).lockB)
}

// applyLeg is the shared leg path: look up the session, require the message
// author equals the expected party, require the declared leg index, advance the
// state machine, and return the recorded proof to relay.
func (c *Coordinator) applyLeg(m Signed[LegPayload], leg uint8, author func(*SwapSession) common.Address, lock func(*SwapSession, LegProof) error) (LegProof, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	sess, ok := c.sessions[m.Body.Session]
	if !ok {
		return LegProof{}, ErrUnknownSession
	}
	addr, err := signer(m)
	if err != nil {
		return LegProof{}, err
	}
	if addr != author(sess) {
		return LegProof{}, ErrWrongAuthor
	}
	if m.Body.Leg != leg {
		return LegProof{}, ErrLegMismatch
	}
	if err := lock(sess, m.Body.Proof); err != nil {
		return LegProof{}, err
	}
	return m.Body.Proof, nil
}

// RecordClaim records the taker's claim of HTLC_A (spec step 5) and returns the
// now-public preimage for the venue to relay to the maker, who counter-claims
// HTLC_B with it. The message MUST be signed by the taker.
func (c *Coordinator) RecordClaim(m Signed[ClaimPayload]) (htlc.Secret, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	sess, ok := c.sessions[m.Body.Session]
	if !ok {
		return htlc.Secret{}, ErrUnknownSession
	}
	addr, err := signer(m)
	if err != nil {
		return htlc.Secret{}, err
	}
	if addr != sess.User {
		return htlc.Secret{}, ErrWrongAuthor
	}
	if err := sess.claim(m.Body.Secret); err != nil {
		return htlc.Secret{}, err
	}
	return m.Body.Secret, nil
}

// RecordCounterClaim records the maker's counter-claim of HTLC_B (spec step 6):
// Claimed → Settled. The message MUST be signed by the maker.
func (c *Coordinator) RecordCounterClaim(m Signed[SettlePayload]) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	sess, ok := c.sessions[m.Body.Session]
	if !ok {
		return ErrUnknownSession
	}
	addr, err := signer(m)
	if err != nil {
		return err
	}
	if addr != sess.Maker {
		return ErrWrongAuthor
	}
	return sess.settle()
}

// RecordRefund records an off-path unwind (spec step 7). Either party may report
// it (the refunding party signs); the session moves to Refunded or Aborted.
func (c *Coordinator) RecordRefund(m Signed[RefundPayload]) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	sess, ok := c.sessions[m.Body.Session]
	if !ok {
		return ErrUnknownSession
	}
	addr, err := signer(m)
	if err != nil {
		return err
	}
	if addr != sess.Maker && addr != sess.User {
		return ErrWrongAuthor
	}
	return sess.refund()
}

// Session returns the live session for id (test/observability accessor).
func (c *Coordinator) Session(id SessionID) (*SwapSession, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	s, ok := c.sessions[id]
	return s, ok
}
