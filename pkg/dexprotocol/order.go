// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"

	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
)

// order.go is the trader's half of the seam.
//
// THE SHAPE. The trader signs an Order and nothing moves. D's book matches it and
// certifies an Execution. An operator carries Order + Execution + proof to the C
// reactor, which verifies the signature, the nonce, the deadline, the bounds and
// the D certificate, and settles. Only when that C block is accepted does the
// Execution become a Trade.
//
// WHY THE ORDER CARRIES NO C PARENT. An Order is a long-lived authorization; an
// Execution is a short-lived reservation against one accepted C parent. Binding the
// Order to a C head would make the trader responsible for guessing which block
// their order would be matched under — and would kill the property that makes the
// book work: an unmatched Order stays live, and an Order whose Execution was
// released can be matched AGAIN. That property is why CParent lives on Execution
// and only on Execution.
//
// AUTHORIZATION IS THE SIGNATURE, NOT A PRIOR TRANSFER. The trader does not first
// send funds anywhere. The signed Order says "up to this input, for at least this
// output, until this deadline" and the reactor pulls the input at settlement, the
// same way a witness transfer does. An order that never executes costs the trader
// one signature and nothing else.

// AssetAmount is an amount of one asset in that asset's OWN base units — wei for an
// 18-decimal ERC-20, not lots.
//
// THIS IS DELIBERATELY NOT Execution.Quantity's TYPE. Execution.Quantity is in the
// market's LOTS, the unit the book matches in; Input and MinOutput are in TOKEN base
// units, the unit the escrow actually moves. They are different quantities that
// happen to describe the same trade, and the conversion between them is the market's
// lot size. Giving them one type would invite adding a lot count to a wei balance,
// which is the kind of mistake that type-checks and then loses money.
//
// Amount is 32 bytes big-endian because it faces ERC-20 balances. A uint64 would cap
// an 18-decimal token at about 18.4 whole units.
type AssetAmount struct {
	Asset  ids.ID
	Amount [32]byte
}

// AssetAmountEncodedLen is fixed, so concatenation stays injective.
const AssetAmountEncodedLen = 32 + 32

// Big returns the amount as a big.Int. Convenience for arithmetic; the wire form is
// always the fixed-width bytes.
func (a AssetAmount) Big() *big.Int { return new(big.Int).SetBytes(a.Amount[:]) }

// SetBig writes v into the fixed-width amount, refusing anything that does not fit
// or is negative. Refusing rather than truncating: a silently wrapped amount is an
// authorization the trader never gave.
func (a *AssetAmount) SetBig(v *big.Int) error {
	if v.Sign() < 0 {
		return ErrAmountNegative
	}
	if v.BitLen() > 256 {
		return ErrAmountOverflow
	}
	var b [32]byte
	v.FillBytes(b[:])
	a.Amount = b
	return nil
}

// IsZero reports whether the amount is zero.
func (a AssetAmount) IsZero() bool { return a.Amount == [32]byte{} }

// AtLeast reports whether a covers want. Both are big-endian fixed width, so a byte
// compare IS the numeric compare — no parsing, no allocation.
func (a AssetAmount) AtLeast(want AssetAmount) bool {
	return a.Asset == want.Asset && bytes.Compare(a.Amount[:], want.Amount[:]) >= 0
}

// Order is what the trader signed. Every field is something the TRADER chose;
// nothing here is computed by D or by C.
type Order struct {
	Swapper   common.Address
	Nonce     uint64
	Market    ids.ID
	Side      Side
	Input     AssetAmount
	MinOutput AssetAmount
	Deadline  uint64
	Recipient common.Address
}

// OrderEncodedLen is the canonical encoding's exact width — every field fixed width,
// so the concatenation is injective and the commitment can rest on SHA-256 alone
// rather than on the hash plus an argument about parsing.
const OrderEncodedLen = 20 + // Swapper
	8 + // Nonce
	32 + // Market
	1 + // Side
	AssetAmountEncodedLen + // Input
	AssetAmountEncodedLen + // MinOutput
	8 + // Deadline
	20 // Recipient

const orderDomain = "lux.dex.order.v1"

var orderDomainTag = sha256.Sum256([]byte(orderDomain))

var (
	ErrOrderWidth     = errors.New("dexprotocol: encoded Order has the wrong width")
	ErrOrderScope     = errors.New("dexprotocol: Order is missing a required field")
	ErrOrderInput     = errors.New("dexprotocol: Order Input must be non-zero")
	ErrOrderExpired   = errors.New("dexprotocol: Order deadline has passed")
	ErrOrderSide      = errors.New("dexprotocol: Order Side is not a valid side")
	ErrAmountNegative = errors.New("dexprotocol: amount is negative")
	ErrAmountOverflow = errors.New("dexprotocol: amount does not fit in 256 bits")
	ErrNoNonces       = errors.New("dexprotocol: order verification requires the consumed-nonce set")
	ErrNonceUsed      = errors.New("dexprotocol: this nonce has already been consumed")
	ErrNoSignature    = errors.New("dexprotocol: order verification requires a signature verifier")
	ErrBadSignature   = errors.New("dexprotocol: order signature did not recover the swapper")
	ErrOrderUnbound   = errors.New("dexprotocol: order verification requires the chain binding")
)

// Encode returns the canonical fixed-width big-endian encoding. Total, so a
// malformed order still has a well-defined commitment: a verifier must be able to
// say "this signature is over order X and X is invalid" rather than being unable to
// name what it rejected.
func (o *Order) Encode() []byte {
	b := make([]byte, OrderEncodedLen)
	n := 0
	n += copy(b[n:], o.Swapper[:])
	binary.BigEndian.PutUint64(b[n:], o.Nonce)
	n += 8
	n += copy(b[n:], o.Market[:])
	b[n] = byte(o.Side)
	n++
	n += copy(b[n:], o.Input.Asset[:])
	n += copy(b[n:], o.Input.Amount[:])
	n += copy(b[n:], o.MinOutput.Asset[:])
	n += copy(b[n:], o.MinOutput.Amount[:])
	binary.BigEndian.PutUint64(b[n:], o.Deadline)
	n += 8
	n += copy(b[n:], o.Recipient[:])
	if n != OrderEncodedLen {
		panic(fmt.Sprintf("dexprotocol: order encoder wrote %d bytes, layout says %d", n, OrderEncodedLen))
	}
	return b
}

// DecodeOrder parses the canonical encoding, requiring the EXACT width. A longer
// buffer is refused rather than truncated: accepting a suffix would let two distinct
// wire messages decode to the same order while carrying different signed bytes.
func DecodeOrder(b []byte) (Order, error) {
	if len(b) != OrderEncodedLen {
		return Order{}, fmt.Errorf("%w: got %d, want %d", ErrOrderWidth, len(b), OrderEncodedLen)
	}
	var o Order
	n := 0
	n += copy(o.Swapper[:], b[n:n+20])
	o.Nonce = binary.BigEndian.Uint64(b[n:])
	n += 8
	n += copy(o.Market[:], b[n:n+32])
	o.Side = Side(b[n])
	n++
	n += copy(o.Input.Asset[:], b[n:n+32])
	n += copy(o.Input.Amount[:], b[n:n+32])
	n += copy(o.MinOutput.Asset[:], b[n:n+32])
	n += copy(o.MinOutput.Amount[:], b[n:n+32])
	o.Deadline = binary.BigEndian.Uint64(b[n:])
	n += 8
	copy(o.Recipient[:], b[n:n+20])
	return o, nil
}

// Binding is the chain identity an order's signature is scoped to.
//
// WITHOUT THIS AN ORDER IS SETTLEABLE EVERYWHERE. 0x9999 runs on every EVM that
// shares the one D venue. A portable signed order with no chain binding is a
// signature an operator can replay against each of them in turn, and the trader pays
// on all of them for the one trade they authorized. The binding is exactly the
// domain separator's job: which network, which C chain, which reactor.
//
// It is a VALUE supplied by the verifier and never read from the witness. A witness
// that carried its own binding would be asserting the one thing the whole
// cross-chain-replay defence rests on.
type Binding struct {
	NetworkID uint32
	CChainID  ids.ID
	Reactor   common.Address
}

// BindingEncodedLen is fixed width, like everything else in the preimage.
const BindingEncodedLen = 4 + 32 + 20

func (b Binding) encode() []byte {
	out := make([]byte, BindingEncodedLen)
	binary.BigEndian.PutUint32(out, b.NetworkID)
	n := 4
	n += copy(out[n:], b.CChainID[:])
	copy(out[n:], b.Reactor[:])
	return out
}

func (b Binding) zero() bool {
	return b.NetworkID == 0 || b.CChainID == ids.Empty || b.Reactor == (common.Address{})
}

// Commitment is what the trader signs:
//
//	commitment = SHA-256( SHA-256(domain) || binding || canonical encoding )
//
// Three fixed-width parts, so distinct (binding, order) pairs give distinct
// preimages. The binding sits INSIDE the hash rather than beside it, so a signature
// produced for one chain does not verify on another — the replay is not detected,
// it is unrepresentable.
func (o *Order) Commitment(b Binding) ids.ID {
	h := sha256.New()
	h.Write(orderDomainTag[:])
	h.Write(b.encode())
	h.Write(o.Encode())
	var id ids.ID
	copy(id[:], h.Sum(nil))
	return id
}

// OrderID is the order's identity, used as the key for remaining-quantity
// accounting and as the argument to cancel. It is the commitment, so an id can never
// be asserted independently of the order it names.
func (o *Order) OrderID(b Binding) ids.ID { return o.Commitment(b) }

// Validate refuses an order missing a field the security argument depends on.
func (o *Order) Validate() error {
	if o.Swapper == (common.Address{}) {
		return fmt.Errorf("%w: Swapper", ErrOrderScope)
	}
	if o.Recipient == (common.Address{}) {
		return fmt.Errorf("%w: Recipient", ErrOrderScope)
	}
	if o.Market == ids.Empty {
		return fmt.Errorf("%w: Market", ErrOrderScope)
	}
	if o.Input.Asset == ids.Empty {
		return fmt.Errorf("%w: Input.Asset", ErrOrderScope)
	}
	if o.MinOutput.Asset == ids.Empty {
		return fmt.Errorf("%w: MinOutput.Asset", ErrOrderScope)
	}
	if o.Side != SideBuy && o.Side != SideSell {
		return fmt.Errorf("%w: %d", ErrOrderSide, uint8(o.Side))
	}
	if o.Input.IsZero() {
		return ErrOrderInput
	}
	// MinOutput MAY be zero — that is a market order, an explicit choice to accept
	// whatever the book gives. Input may not be, because a zero-input order
	// authorizes a transfer of nothing and can only be noise on the book.
	return nil
}

// --- Unordered nonces ---------------------------------------------------------
//
// THE NONCE MUST NOT BE SEQUENTIAL. D's book matches by price-time priority across
// every trader, so reordering is not an edge case, it is the book's whole function:
// one trader's two resting orders WILL settle out of the order they were signed in.
// A sequential check (`nonce == expected++`) rejects the second one and then blocks
// that trader forever, because the expected value can never advance past an order
// that may never match.
//
// So a nonce is a POSITION IN A BITMAP, consumed independently of every other. Word
// is nonce>>8, bit is nonce&0xff. The trader picks any unused position; the reactor
// consumes exactly that bit.

// NonceWord is the bitmap word a nonce lives in.
func NonceWord(nonce uint64) uint64 { return nonce >> 8 }

// NonceBit is the position within that word.
func NonceBit(nonce uint64) uint8 { return uint8(nonce & 0xff) }

// Nonces is the reference model of the reactor's consumed-nonce bitmap. A persistent
// implementation must keep the same semantics: consumption is per-position and
// irreversible, never a counter.
type Nonces struct {
	words map[common.Address]map[uint64][4]uint64
}

func NewNonces() *Nonces {
	return &Nonces{words: make(map[common.Address]map[uint64][4]uint64)}
}

func slot(bit uint8) (idx int, mask uint64) { return int(bit >> 6), uint64(1) << (bit & 63) }

// Used reports whether this swapper has already consumed this nonce.
func (n *Nonces) Used(swapper common.Address, nonce uint64) bool {
	w, ok := n.words[swapper]
	if !ok {
		return false
	}
	idx, mask := slot(NonceBit(nonce))
	return w[NonceWord(nonce)][idx]&mask != 0
}

// Consume marks the nonce used, refusing a replay. Returns ErrNonceUsed rather than
// silently succeeding, because a repeated consume is exactly a settlement replay.
func (n *Nonces) Consume(swapper common.Address, nonce uint64) error {
	if n.Used(swapper, nonce) {
		return fmt.Errorf("%w: %s nonce %d", ErrNonceUsed, swapper, nonce)
	}
	w, ok := n.words[swapper]
	if !ok {
		w = make(map[uint64][4]uint64)
		n.words[swapper] = w
	}
	word := NonceWord(nonce)
	idx, mask := slot(NonceBit(nonce))
	cur := w[word]
	cur[idx] |= mask
	w[word] = cur
	return nil
}

// --- Authenticated orders -----------------------------------------------------
//
// Same discipline as VerifiedExecution, for the same reason: a function taking a
// plain Order is a standing invitation to settle something nobody authenticated, and
// review does not reliably catch the one call site that skipped the check. So the
// settlement path does not accept an Order at all.
//
// The unexported METHOD is the enforcement, not the unexported fields. A struct with
// unexported fields is still zero-constructible from outside — Order{} compiles
// anywhere — so the sentinel has to be something outside packages cannot implement.

// VerifiedOrder is an order whose signature, deadline and bounds have been checked
// against a specific chain binding. It cannot be constructed outside this package.
type VerifiedOrder interface {
	Order() Order
	OrderID() ids.ID
	Binding() Binding
	// verifiedOrder is unexported and therefore unimplementable outside this
	// package. It is the whole enforcement mechanism; do not export it.
	verifiedOrder()
}

type verifiedOrder struct {
	order   Order
	id      ids.ID
	binding Binding
}

func (v verifiedOrder) Order() Order     { return v.order }
func (v verifiedOrder) OrderID() ids.ID  { return v.id }
func (v verifiedOrder) Binding() Binding { return v.binding }
func (verifiedOrder) verifiedOrder()     {}

// SignatureVerifier recovers the signer of an order commitment. It is an interface
// so this package depends on the ABSTRACTION and never on a curve implementation, a
// keystore or a wallet — the same structural import ban CertificateVerifier gives us
// for D certificates.
type SignatureVerifier interface {
	Recover(commitment ids.ID, sig []byte) (common.Address, error)
}

// OrderContext is everything order verification is allowed to consult. It is a
// VALUE: no clients, no connections, no callbacks into a live process.
//
// BlockTime is the timestamp of the block being built, supplied by the verifier. It
// is deliberately not a clock read: a wall-clock deadline check would make the same
// order valid on one validator and expired on another, which is precisely the
// nondeterminism this whole seam exists to keep out.
type OrderContext struct {
	Binding   Binding
	BlockTime uint64
	Signer    SignatureVerifier
	Nonces    *Nonces
}

// VerifyOrder is the ONLY producer of a VerifiedOrder. It checks, in order: the
// order is well formed; it has not expired as of the block being built; its nonce is
// unconsumed; and the signature over the CHAIN-BOUND commitment recovers the
// swapper.
//
// It does NOT consume the nonce. Verification is a pure question; consumption is a
// state change that belongs at settlement, when C has decided. Folding them together
// would burn a trader's nonce on an order that was merely inspected.
func VerifyOrder(witness []byte, ctx OrderContext) (VerifiedOrder, error) {
	if ctx.Signer == nil {
		return nil, ErrNoSignature
	}
	if ctx.Binding.zero() {
		return nil, ErrOrderUnbound
	}
	o, err := DecodeOrder(orderBody(witness))
	if err != nil {
		return nil, err
	}
	if err := o.Validate(); err != nil {
		return nil, err
	}
	if o.Deadline < ctx.BlockTime {
		return nil, fmt.Errorf("%w: deadline %d, building at %d", ErrOrderExpired, o.Deadline, ctx.BlockTime)
	}
	// FAIL CLOSED, like ctx.Signer two checks above. A nil Nonces used to mean "skip
	// replay protection", so a caller that forgot to wire it got no enforcement and
	// no error — the quietest way to lose a safety property is to make its absence
	// look like a configuration choice.
	if ctx.Nonces == nil {
		return nil, ErrNoNonces
	}
	if ctx.Nonces.Used(o.Swapper, o.Nonce) {
		return nil, fmt.Errorf("%w: %s nonce %d", ErrNonceUsed, o.Swapper, o.Nonce)
	}
	commitment := o.Commitment(ctx.Binding)
	signer, err := ctx.Signer.Recover(commitment, orderSignature(witness))
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrBadSignature, err)
	}
	if signer != o.Swapper {
		return nil, fmt.Errorf("%w: recovered %s, order says %s", ErrBadSignature, signer, o.Swapper)
	}
	return verifiedOrder{order: o, id: commitment, binding: ctx.Binding}, nil
}

// orderBody / orderSignature split a witness into the canonical order encoding and
// the signature that follows it. The order is fixed width, so the split is
// unambiguous.
func orderBody(witness []byte) []byte {
	if len(witness) < OrderEncodedLen {
		return witness
	}
	return witness[:OrderEncodedLen]
}

func orderSignature(witness []byte) []byte {
	if len(witness) <= OrderEncodedLen {
		return nil
	}
	return witness[OrderEncodedLen:]
}
