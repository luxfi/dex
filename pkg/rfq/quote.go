// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package rfq

import (
	"crypto/ecdsa"
	"errors"
	"math/big"
	"sync"
	"time"

	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// Quote / QuoteBook errors. All are deterministic rejects: a bad quote is refused
// at the book boundary and never served.
var (
	ErrQuoteUnsigned     = errors.New("rfq: quote has no maker address")
	ErrQuoteBadSignature = errors.New("rfq: quote maker signature invalid or signer is not the asserted maker")
	ErrQuoteExpired      = errors.New("rfq: quote firm-window has expired")
	ErrUnknownQuote      = errors.New("rfq: no such quote in the book")
)

// QuoteID is the content address of a quote: keccak256 over its signed fields. A
// taker references the exact firm quote it accepted by this id.
type QuoteID [32]byte

// Pair names a directed cross-chain asset pair: the maker sells AssetA on ChainA
// (the taker receives it) and is paid AssetB on ChainB. Chain identities and asset
// identifiers are values, not connections — the coordinator never dials a chain.
type Pair struct {
	ChainA ChainKind
	AssetA string
	ChainB ChainKind
	AssetB string
}

// Quote is a maker's FIRM offer, valid until Expiry and SIGNED by the maker. It is
// a pure value: the signature covers every economically-binding field plus the
// maker's own chain endpoints (MakerA = the chain-A refund address it will lock
// from; MakerB = the chain-B address it must be paid to), so a relayed quote can
// be neither re-priced nor re-attributed to a different maker.
//
// Maker is the asserted secp256k1 signer; Verify recovers the signature and
// REQUIRES the recovered address equals Maker — the same ecrecover-and-bind idiom
// the order path (pkg/dex/signed_order.go) and the D-Chain tx gate (pkg/dchain/
// auth.go) use, over the shared crypto.Ecrecover primitive (see recoverAddr).
type Quote struct {
	Maker   common.Address // asserted maker identity; Verify binds the signature to it
	Pair    Pair           // directed cross-chain asset pair
	Rate    string         // firm rate (human decimal); bound in the signature
	AmountA *big.Int       // amount of AssetA the maker locks on ChainA (recipient = taker)
	AmountB *big.Int       // amount of AssetB the taker locks on ChainB (recipient = maker)
	MakerA  string         // maker's refund address on ChainA (HTLC_A refund)
	MakerB  string         // maker's payout address on ChainB (HTLC_B recipient)
	Expiry  int64          // unix seconds; the quote is firm strictly before this instant
	Sig     [65]byte       // secp256k1 r‖s‖v over digest()
}

// digest is the 32-byte commitment the maker signature covers.
func (q *Quote) digest() [32]byte {
	return newEnc().
		addr(q.Maker).
		u8(uint8(q.Pair.ChainA)).str(q.Pair.AssetA).
		u8(uint8(q.Pair.ChainB)).str(q.Pair.AssetB).
		str(q.Rate).
		big(q.AmountA).big(q.AmountB).
		str(q.MakerA).str(q.MakerB).
		u64(uint64(q.Expiry)).
		hash(domainQuote)
}

// ID is the quote's content address, used by the taker to reference the accepted
// quote and by the book as its key.
func (q *Quote) ID() QuoteID { return QuoteID(q.digest()) }

// Sign sets Maker to priv's address and attaches the maker signature. This is the
// maker-side half of the one signing scheme; the coordinator only ever Verify-s.
func (q *Quote) Sign(priv *ecdsa.PrivateKey) error {
	q.Maker = pubAddr(priv)
	d := q.digest()
	sig, err := crypto.Sign(d[:], priv)
	if err != nil {
		return err
	}
	copy(q.Sig[:], sig)
	return nil
}

// Verify recovers the maker signature over digest() and requires it equals the
// asserted Maker. Fail-closed: any recovery failure or mismatch is a reject.
func (q *Quote) Verify() error {
	if q.Maker == (common.Address{}) {
		return ErrQuoteUnsigned
	}
	d := q.digest()
	addr, err := recoverAddr(d, q.Sig[:])
	if err != nil {
		return errors.Join(ErrQuoteBadSignature, err)
	}
	if addr != q.Maker {
		return ErrQuoteBadSignature
	}
	return nil
}

// live reports whether the quote is still firm at now.
func (q *Quote) live(now time.Time) bool { return now.Unix() < q.Expiry }

// QuoteBook stores and serves live, maker-signed quotes. It is the ONLY stateful
// piece of the venue, and it holds only public values: signed quotes. Insert
// verifies the maker signature, so the book never serves a forged or re-priced
// quote; Match serves the live quotes for a pair.
type QuoteBook struct {
	mu     sync.RWMutex
	quotes map[QuoteID]Quote
}

// NewQuoteBook returns an empty book.
func NewQuoteBook() *QuoteBook { return &QuoteBook{quotes: map[QuoteID]Quote{}} }

// Insert verifies the maker signature, rejects an already-expired quote, and
// stores the quote keyed by its content address. Re-inserting the same quote is
// idempotent (same id).
func (b *QuoteBook) Insert(q Quote, now time.Time) (QuoteID, error) {
	if err := q.Verify(); err != nil {
		return QuoteID{}, err
	}
	if !q.live(now) {
		return QuoteID{}, ErrQuoteExpired
	}
	id := q.ID()
	b.mu.Lock()
	b.quotes[id] = q
	b.mu.Unlock()
	return id, nil
}

// Get returns the stored quote for id.
func (b *QuoteBook) Get(id QuoteID) (Quote, bool) {
	b.mu.RLock()
	q, ok := b.quotes[id]
	b.mu.RUnlock()
	return q, ok
}

// Match returns the live quotes for pair whose AmountA can cover size, dropping
// any that have expired by now. The result order is unspecified (a map walk); a
// caller ranks by rate.
func (b *QuoteBook) Match(pair Pair, size *big.Int, now time.Time) []Quote {
	b.mu.RLock()
	defer b.mu.RUnlock()
	var out []Quote
	for _, q := range b.quotes {
		if q.Pair != pair || !q.live(now) {
			continue
		}
		if size != nil && q.AmountA != nil && q.AmountA.Cmp(size) < 0 {
			continue
		}
		out = append(out, q)
	}
	return out
}

// Domain-separation tags. Distinct constants keep a signature over one message
// type from ever verifying as another (a leg proof can never be replayed as a
// quote, a claim, or a refund), in the same spirit as pkg/dchain's txAuthDomain.
const (
	domainQuote  = "lux.rfq.quote.v1"
	domainCommit = "lux.rfq.commit.v1"
	domainLeg    = "lux.rfq.leg.v1"
	domainClaim  = "lux.rfq.claim.v1"
	domainSettle = "lux.rfq.settle.v1"
	domainRefund = "lux.rfq.refund.v1"
)

// enc is the package's single deterministic field encoder for signing digests:
// fixed-width scalars, 20-byte addresses, 32-byte words, and length-prefixed
// variable bytes, so two distinct field sequences can never produce the same
// pre-image. Every signed value in the package (quote, commit, leg, claim,
// settle, refund) hashes through this one builder — one signing scheme, one
// encoding.
type enc struct{ b []byte }

func newEnc() *enc { return &enc{} }

func (e *enc) u8(v uint8) *enc { e.b = append(e.b, v); return e }

func (e *enc) u64(v uint64) *enc {
	e.b = append(e.b, byte(v>>56), byte(v>>48), byte(v>>40), byte(v>>32),
		byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
	return e
}

func (e *enc) addr(a common.Address) *enc { e.b = append(e.b, a[:]...); return e }

func (e *enc) word(b []byte) *enc {
	var w [32]byte
	copy(w[32-len(b):], b)
	e.b = append(e.b, w[:]...)
	return e
}

func (e *enc) big(v *big.Int) *enc {
	if v == nil {
		return e.word(nil)
	}
	return e.word(v.Bytes())
}

func (e *enc) bytes32(h [32]byte) *enc { e.b = append(e.b, h[:]...); return e }

func (e *enc) str(s string) *enc {
	e.b = append(e.b, byte(len(s)>>8), byte(len(s)))
	e.b = append(e.b, s...)
	return e
}

func (e *enc) hash(domain string) [32]byte {
	var out [32]byte
	copy(out[:], crypto.Keccak256([]byte(domain), e.b))
	return out
}

// pubAddr is the geth/common address of a secp256k1 key. crypto.PubkeyToAddress
// returns the crypto-common address type; we bridge by bytes to the geth/common
// address recoverAddr returns, exactly as the dchain test harness does.
func pubAddr(priv *ecdsa.PrivateKey) common.Address {
	return common.BytesToAddress(crypto.PubkeyToAddress(priv.PublicKey).Bytes())
}

// errRecover marks a malformed or unrecoverable secp256k1 signature.
var errRecover = errors.New("rfq: secp256k1 recovery failed")

// recoverAddr recovers the secp256k1 signer of digest from a 65-byte r‖s‖v
// signature, returning its address — the low 20 bytes of keccak256(X‖Y). It is
// the verify half of the package's one signing scheme (Sign uses crypto.Sign),
// over the SAME crypto.Ecrecover primitive the order path (pkg/dex/signed_order.go)
// and the D-Chain tx gate (pkg/dchain/auth.go) use. A venue message is its own
// domain (the domain* tags), so this recover is a leaf helper rather than a borrow
// of pkg/dex's tx-auth wrapper — which would couple this pure relay to the GPU DEX
// matcher. secp256k1 is the only scheme venue messages use.
func recoverAddr(digest [32]byte, sig []byte) (common.Address, error) {
	if len(sig) != 65 || sig[64] > 1 { // ecrecover accepts only v ∈ {0,1}
		return common.Address{}, errRecover
	}
	pub, err := crypto.Ecrecover(digest[:], sig)
	if err != nil || len(pub) != 65 {
		return common.Address{}, errRecover
	}
	return common.BytesToAddress(crypto.Keccak256(pub[1:])[12:]), nil
}
