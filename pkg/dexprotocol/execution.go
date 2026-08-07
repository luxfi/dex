// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package dexprotocol is the consensus-side vocabulary of the C<->D exchange seam.
//
// THE PROTOCOL, IN ONE PARAGRAPH:
//
//	The C transaction contains an Order. D matches it and certifies an Execution —
//	price and quantity guaranteed and reserved. The C block carries that Execution.
//	EVM execution verifies it and returns the resulting BalanceDelta. When the child
//	of the Execution's C parent is accepted, D settles it into a Trade if that child
//	used it, or Releases the reservation if it did not. Batch certification is an
//	optional optimization.
//
// THE INVARIANT IT EXISTS TO HOLD: C never queries D consensus state during EVM
// execution. D converts live exchange state into certified, single-consumption
// statements. C deterministically consumes those attestations with full synchronous
// V4 semantics. D finalizes or releases the reservation solely from proof of the
// accepted C successor. On replay D can be dead, the GPU gone, D a billion blocks
// ahead — C has parent state, transactions, the D statement, and the certificate.
// That is the entire input set.
//
// THE NORMATIVE RULE:
//
//	Lux DEX follows FIX Latest terminology and order-state semantics. ExecType
//	denotes an event; OrdStatus denotes the resulting order state. A D-side
//	Execution does not become a FIX Trade until the corresponding C-chain
//	execution is accepted.
//
// That last sentence is the whole cross-chain contract in one line.
//
// TWO VOCABULARIES, DELIBERATELY SPLIT.
//
// INTERNAL (cross-chain machinery):  Order -> Execution -> Trade | Release,
// with Match as the matching-engine verb. This is small and Lux-specific because
// the provisional reservation is internal consensus machinery.
//
// EXTERNAL (client-facing): FIX Latest, emitted ONLY once C acceptance makes an
// execution final — 35=8 ExecutionReport, 150=F ExecType=Trade, 39=1|2
// OrdStatus=PartiallyFilled|Filled, 17/31/32/14/151. Clients see a normal order
// lifecycle and never learn about C-parent reservations or cross-chain consensus.
// There is deliberately NO FIX state for the provisional reservation: mapping it
// onto one would make every FIX client responsible for understanding Lux's
// cross-chain machinery.
//
// Every earlier name for the provisional object — PreparedExecution,
// ExecutionFrame, Fill — asserted that execution had happened while C could still
// reject it and D could still restore the liquidity. That is economically false,
// which is why the internal name is plain Execution and the word Trade is reserved
// for the post-acceptance fact.
//
// Match stays an internal matching-engine action. It is not a cross-chain protocol
// object and must not appear on the wire.
//
// MARKET DATA: publish a Trade only after C acceptance. A reserved Execution must
// never appear in the trades feed, volume, candles, or history.
//
// WHAT THE CERTIFICATE PROVES, AND WHAT IT DOES NOT. It proves "this is the
// execution D reported, and D has reserved the price and quantity behind it for
// exactly this C opportunity." It does NOT say C must accept the transaction. C
// still applies minAmountOut, sqrtPriceLimit, hook rules, deadline and EVM
// authorization locally, and reverts when the trader's own bounds are unmet. D owns
// execution facts; C owns whether they satisfy the call.
//
// THE HONEST COST is not determinism — that is bought outright by moving the
// statement into the block. The cost is that D must hold the reservation until C
// consensus decides whether it was used. That is the fundamental cross-chain price.
package dexprotocol

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/luxfi/ids"
)

// Side is the order's direction on the book.
type Side uint8

const (
	SideBuy Side = iota
	SideSell
)

func (s Side) String() string {
	switch s {
	case SideBuy:
		return "buy"
	case SideSell:
		return "sell"
	default:
		return fmt.Sprintf("invalid(%d)", uint8(s))
	}
}

// ExecType is FIX ExecType(150) — the EVENT being reported. ExecType and OrdStatus
// are kept as separate fields, never collapsed into one, because that split is
// normative in FIX. Both exist ON THE WIRE only; internally, state is the object's
// TYPE and its storage domain (see state.go). There is no entry for the provisional
// reservation: it is never reported to a FIX client.
type ExecType uint8

const (
	ExecNew ExecType = iota
	ExecTrade
	ExecCanceled
	ExecRejected
)

func (e ExecType) String() string {
	switch e {
	case ExecNew:
		return "New"
	case ExecTrade:
		return "Trade"
	case ExecCanceled:
		return "Canceled"
	case ExecRejected:
		return "Rejected"
	default:
		return fmt.Sprintf("invalid(%d)", uint8(e))
	}
}

// OrdStatus is FIX OrdStatus(39) — the resulting ORDER STATE. Wire-only, same
// reasoning as ExecType.
type OrdStatus uint8

const (
	OrdNew OrdStatus = iota
	OrdPartiallyFilled
	OrdFilled
	OrdCanceled
	OrdRejected
)

func (o OrdStatus) String() string {
	switch o {
	case OrdNew:
		return "New"
	case OrdPartiallyFilled:
		return "PartiallyFilled"
	case OrdFilled:
		return "Filled"
	case OrdCanceled:
		return "Canceled"
	case OrdRejected:
		return "Rejected"
	default:
		return fmt.Sprintf("invalid(%d)", uint8(o))
	}
}

// Order lives in order.go. It is the trader's signed authorization and carries NO
// C parent: an order is portable across its validity window, and the same unmatched
// order may be matched again after an execution against it is released. CParent
// belongs to Execution, below, because the reservation — not the authorization — is
// what is scoped to one C opportunity.

// Execution is the object D certifies — a FIX ExecutionReport reduced to the
// fields that cross the seam.
//
// ExecID is D-assigned and the commitment COVERS it, so a certificate binds that id
// to exactly this content: an id can never be asserted independently of what it
// names, and two executions cannot share an id without sharing every byte.
//
// There is no reservation field: the reservation is keyed by
// (CChainID, CParent, ExecID) — see ReservationKey — so there is no separate value
// that could disagree with the key it is supposed to name.
// D RESOLVES, C VERIFIES. The execution carries the EXACT token amounts that will
// move, converted once by D using the market version named here. C checks those
// amounts against the order's bounds and moves them; it never recomputes lots from
// whatever the market metadata happens to say now. Recomputing would reintroduce the
// replay problem at a smaller scale: an execution certified under one lot size would
// settle differently after a market update, so the same block would replay to a
// different result.
//
// MarketVersion is therefore load-bearing, not documentation. It pins which metadata
// produced Quantity -> Input/Output, and the commitment covers it.
//
// There is no Price field. Price is Output/Input and would be a second statement of
// the same fact inside a signed struct — two things that can disagree. FIX LastPx(31)
// is derived at the edge where the ExecutionReport is built.
type Execution struct {
	ExecID        ids.ID
	OrderID       ids.ID
	MarketID      ids.ID
	MarketVersion uint64
	Quantity      uint64      // lots — the book's unit; FIX LastQty(32)
	Input         AssetAmount // exact token base units leaving the swapper
	Output        AssetAmount // exact token base units reaching the recipient
	Fee           AssetAmount
	CParent       ids.ID // the C opportunity this execution may be traded under
	DBlock        ids.ID // the D block that produced the match
}

// ExecutionEncodedLen is the canonical encoding's exact width. Every field is fixed
// width, so concatenation is INJECTIVE by construction — no length prefixes, no
// separators. That is what lets the commitment rest on SHA-256 alone rather than on
// the hash plus an argument about parsing.
const ExecutionEncodedLen = 32 + 32 + 32 + // ExecID, OrderID, MarketID
	8 + 8 + // MarketVersion, Quantity
	AssetAmountEncodedLen*3 + // Input, Output, Fee
	32 + 32 // CParent, DBlock

// execDomain domain-separates the execution commitment. It is hashed once into a
// fixed 32-byte tag so the preimage is tag(32) || encoding(ExecutionEncodedLen) —
// two fixed-width parts, hence unambiguous. Prefixing a variable-length ASCII
// string directly would reintroduce the parsing ambiguity the fixed-width encoding
// removes.
const execDomain = "lux.dex.execution.v1"

var execDomainTag = sha256.Sum256([]byte(execDomain))

var (
	ErrExecWidth = errors.New("dexprotocol: encoded Execution has the wrong width")
	ErrExecScope = errors.New("dexprotocol: Execution is missing a required scope field")
	ErrExecQty   = errors.New("dexprotocol: Execution Quantity must be non-zero")
)

// Encode returns the canonical fixed-width big-endian encoding. Total, so a
// malformed execution still has a well-defined commitment — a verifier must be able
// to say "this attestation is over execution X and X is invalid" rather than being
// unable to name what it rejected.
func (e *Execution) Encode() []byte {
	b := make([]byte, ExecutionEncodedLen)
	o := 0
	o += copy(b[o:], e.ExecID[:])
	o += copy(b[o:], e.OrderID[:])
	o += copy(b[o:], e.MarketID[:])
	binary.BigEndian.PutUint64(b[o:], e.MarketVersion)
	o += 8
	binary.BigEndian.PutUint64(b[o:], e.Quantity)
	o += 8
	for _, a := range [...]AssetAmount{e.Input, e.Output, e.Fee} {
		o += copy(b[o:], a.Asset[:])
		o += copy(b[o:], a.Amount[:])
	}
	o += copy(b[o:], e.CParent[:])
	o += copy(b[o:], e.DBlock[:])
	if o != ExecutionEncodedLen {
		panic(fmt.Sprintf("dexprotocol: encoder wrote %d bytes, layout says %d", o, ExecutionEncodedLen))
	}
	return b
}

// DecodeExecution parses the canonical encoding, requiring the EXACT width. A
// longer buffer is refused rather than truncated: accepting a suffix would let two
// distinct wire messages decode to the same execution while carrying different
// attested bytes.
func DecodeExecution(b []byte) (Execution, error) {
	if len(b) != ExecutionEncodedLen {
		return Execution{}, fmt.Errorf("%w: got %d, want %d", ErrExecWidth, len(b), ExecutionEncodedLen)
	}
	var e Execution
	o := 0
	o += copy(e.ExecID[:], b[o:o+32])
	o += copy(e.OrderID[:], b[o:o+32])
	o += copy(e.MarketID[:], b[o:o+32])
	e.MarketVersion = binary.BigEndian.Uint64(b[o:])
	o += 8
	e.Quantity = binary.BigEndian.Uint64(b[o:])
	o += 8
	for _, a := range [...]*AssetAmount{&e.Input, &e.Output, &e.Fee} {
		o += copy(a.Asset[:], b[o:o+32])
		o += copy(a.Amount[:], b[o:o+32])
	}
	o += copy(e.CParent[:], b[o:o+32])
	copy(e.DBlock[:], b[o:o+32])
	return e, nil
}

// Commitment is the execution's identity and THE THING D VALIDATORS SIGN:
//
//	commitment = SHA-256( SHA-256(domain) || canonical encoding )
//
// Both halves are fixed width, so distinct executions give distinct preimages and a
// collision requires breaking SHA-256 rather than confusing a parser.
func (e *Execution) Commitment() ids.ID {
	h := sha256.New()
	h.Write(execDomainTag[:])
	h.Write(e.Encode())
	var id ids.ID
	copy(id[:], h.Sum(nil))
	return id
}

// Validate refuses an execution missing a field the security argument depends on.
// A zero CParent would unbind it from a C opportunity — precisely the property that
// makes the reserved quantity settleable at most once.
func (e *Execution) Validate() error {
	if e.CParent == ids.Empty {
		return fmt.Errorf("%w: CParent", ErrExecScope)
	}
	if e.ExecID == ids.Empty {
		return fmt.Errorf("%w: ExecID", ErrExecScope)
	}
	if e.OrderID == ids.Empty {
		return fmt.Errorf("%w: OrderID", ErrExecScope)
	}
	if e.MarketID == ids.Empty {
		return fmt.Errorf("%w: MarketID", ErrExecScope)
	}
	// MarketVersion 0 would leave the lot conversion unattributed, which is the one
	// thing that makes the amounts reproducible.
	if e.MarketVersion == 0 {
		return fmt.Errorf("%w: MarketVersion", ErrExecScope)
	}
	if e.Quantity == 0 {
		return ErrExecQty
	}
	// An execution moving nothing in or out is not a trade, whatever the lots say.
	if e.Input.IsZero() || e.Output.IsZero() {
		return ErrExecAmount
	}
	if e.Input.Asset == ids.Empty || e.Output.Asset == ids.Empty || e.Fee.Asset == ids.Empty {
		return fmt.Errorf("%w: settlement asset", ErrExecScope)
	}
	return nil
}

// ScopedTo reports whether this execution may be settled by a block whose parent is
// cParent. Named rather than inlined because it is the single check the
// settle-at-most-once property rests on, and it should be greppable.
func (e *Execution) ScopedTo(cParent ids.ID) bool { return e.CParent == cParent }

// --- Authenticated values: an unauthenticated execution must not compile -------
//
// A function like Apply(Execution) is a standing invitation to apply something
// nobody verified, and review does not reliably catch the one call site that
// skipped it. So the lifecycle does not accept an Execution at all. It accepts a
// VerifiedExecution: an INTERFACE with an unexported method, so nothing outside
// this package can implement it, and the concrete type is unexported, so nothing
// outside can construct it. The only way to obtain one is VerifyExecution.

// VerifiedExecution is an execution whose certificate has been checked. It cannot
// be constructed outside this package.
type VerifiedExecution interface {
	Execution() Execution
	Commitment() ids.ID
	// verified is unexported and therefore unimplementable outside this package.
	// It is the whole enforcement mechanism; do not export it.
	verified()
}

type verifiedExecution struct {
	exec       Execution
	commitment ids.ID
}

func (v verifiedExecution) Execution() Execution { return v.exec }
func (v verifiedExecution) Commitment() ids.ID   { return v.commitment }
func (verifiedExecution) verified()              {}

// AcceptedBlock is a C block that C consensus has ACCEPTED, together with the
// ExecRoot of the executions that block consumed.
//
// CPARENT MEANS ACCEPTED C BLOCK. PERIOD. A reservation may only ever be taken
// against an accepted parent, which is what removes the abandoned-parent case
// entirely: acceptance is the finality boundary, so an accepted P is guaranteed
// exactly one accepted child, and that child permanently decides every execution
// reserved against P. Allowing a merely proposed parent would force two questions —
// "did P itself win?" and "which child of P won?" — plus conflict, height and
// ancestry proofs. One question is better than two.
//
// There is NO public constructor. The fields are unexported and the only producer
// is VerifyAcceptedBlock, so an unverified C parent cannot reach Reserve, Settle or
// Release. The `verified` sentinel closes the remaining hole that unexported fields
// alone leave open: a zero AcceptedBlock{} is constructible outside this package,
// so every consumer requires the sentinel rather than merely non-zero ids.
type AcceptedBlock struct {
	verified bool
	id       ids.ID
	parentID ids.ID
	height   uint64
	execRoot [32]byte
}

func (a AcceptedBlock) ID() ids.ID         { return a.id }
func (a AcceptedBlock) ParentID() ids.ID   { return a.parentID }
func (a AcceptedBlock) Height() uint64     { return a.height }
func (a AcceptedBlock) ExecRoot() [32]byte { return a.execRoot }

// AcceptedBlockEncodedLen is the canonical width of an acceptance statement:
// id(32) | parent(32) | height(8) | execRoot(32). Fixed width, so injective.
const AcceptedBlockEncodedLen = 32 + 32 + 8 + 32

const acceptedDomain = "lux.dex.accepted-block.v1"

var acceptedDomainTag = sha256.Sum256([]byte(acceptedDomain))

var (
	ErrAcceptedWidth      = errors.New("dexprotocol: encoded acceptance has the wrong width")
	ErrAcceptedScope      = errors.New("dexprotocol: acceptance is missing the block or parent id")
	ErrAcceptedUnverified = errors.New("dexprotocol: this C parent was never verified; only VerifyAcceptedBlock may produce one")
)

// EncodeAcceptedBlock is the canonical encoding D validators attest over. Exported
// so the C side can produce the statement it signs.
func EncodeAcceptedBlock(id, parentID ids.ID, height uint64, execRoot [32]byte) []byte {
	b := make([]byte, AcceptedBlockEncodedLen)
	o := 0
	o += copy(b[o:], id[:])
	o += copy(b[o:], parentID[:])
	binary.BigEndian.PutUint64(b[o:], height)
	o += 8
	copy(b[o:], execRoot[:])
	return b
}

// AcceptanceCommitment is what a certificate over an acceptance signs.
func AcceptanceCommitment(msg []byte) ids.ID {
	h := sha256.New()
	h.Write(acceptedDomainTag[:])
	h.Write(msg)
	var id ids.ID
	copy(id[:], h.Sum(nil))
	return id
}

// VerifyAcceptedBlock is the ONLY producer of an AcceptedBlock. It authenticates a
// C acceptance statement against a certificate.
//
// NEVER USE LOCAL C STATE AS THE PROOF. Even with C co-located, a check like
// `localC.LastAccepted() != execution.CParent` is wrong inside D consensus:
// different D validators observe C advancement at different instants, so the answer
// is not a function of the block being executed. The local C VM is DELIVERY; the
// authenticated statement is the consensus input. This is the same rule already
// proven in the other direction for C reading D.
func VerifyAcceptedBlock(msg []byte, cert []byte, v CertificateVerifier) (AcceptedBlock, error) {
	if v == nil {
		return AcceptedBlock{}, ErrNoVerifier
	}
	if len(msg) != AcceptedBlockEncodedLen {
		return AcceptedBlock{}, fmt.Errorf("%w: got %d, want %d", ErrAcceptedWidth, len(msg), AcceptedBlockEncodedLen)
	}
	var a AcceptedBlock
	o := 0
	o += copy(a.id[:], msg[o:o+32])
	o += copy(a.parentID[:], msg[o:o+32])
	a.height = binary.BigEndian.Uint64(msg[o:])
	o += 8
	copy(a.execRoot[:], msg[o:o+32])
	if a.id == ids.Empty || a.parentID == ids.Empty {
		return AcceptedBlock{}, ErrAcceptedScope
	}
	if err := v.VerifyCertificate(AcceptanceCommitment(msg), cert); err != nil {
		return AcceptedBlock{}, fmt.Errorf("%w: %s", ErrBadCertificate, err)
	}
	a.verified = true
	return a, nil
}

// CertificateVerifier checks a D-validator certificate over an execution
// commitment. It is an interface so the consensus package depends on the
// ABSTRACTION and never on a client, a socket, or a GPU — the import ban is
// structural, not advisory.
type CertificateVerifier interface {
	VerifyCertificate(commitment ids.ID, cert []byte) error
}

// VerifyContext is everything verification is allowed to consult. It is a VALUE:
// no clients, no connections, no callbacks into a live process. If a check cannot
// be made from this struct plus the witness bytes, it does not belong in the
// consensus path.
type VerifyContext struct {
	NetworkID uint32
	CChainID  ids.ID
	DChainID  ids.ID
	// CParent is the ACTUAL parent of the block being verified, supplied by the
	// verifier — never taken from the witness, or the binding would be self-asserted.
	CParent  ids.ID
	Verifier CertificateVerifier
}

var (
	ErrNoVerifier     = errors.New("dexprotocol: verification requires a certificate verifier")
	ErrWrongCParent   = errors.New("dexprotocol: execution is scoped to a different C opportunity")
	ErrBadCertificate = errors.New("dexprotocol: execution certificate did not verify")
)

// VerifyExecution is the ONLY producer of a VerifiedExecution. It checks, in order:
// the execution is well formed; it is scoped to the C opportunity actually being
// built on; and its commitment carries a valid D-validator certificate.
//
// The CParent check uses the VERIFIER's view of the parent, never the witness's
// claim about it. A witness that asserted its own scope would be asserting the one
// thing the at-most-once property depends on.
func VerifyExecution(witness []byte, ctx VerifyContext) (VerifiedExecution, error) {
	if ctx.Verifier == nil {
		return nil, ErrNoVerifier
	}
	e, err := DecodeExecution(witnessBody(witness))
	if err != nil {
		return nil, err
	}
	if err := e.Validate(); err != nil {
		return nil, err
	}
	if !e.ScopedTo(ctx.CParent) {
		return nil, fmt.Errorf("%w: scoped to %s, building on %s", ErrWrongCParent, e.CParent, ctx.CParent)
	}
	commitment := e.Commitment()
	if err := ctx.Verifier.VerifyCertificate(commitment, witnessCertificate(witness)); err != nil {
		return nil, fmt.Errorf("%w: %s", ErrBadCertificate, err)
	}
	return verifiedExecution{exec: e, commitment: commitment}, nil
}

// witnessBody / witnessCertificate split a witness into the canonical execution
// encoding and the certificate that follows it. The execution is fixed width, so
// the split is unambiguous.
func witnessBody(witness []byte) []byte {
	if len(witness) < ExecutionEncodedLen {
		return witness
	}
	return witness[:ExecutionEncodedLen]
}

func witnessCertificate(witness []byte) []byte {
	if len(witness) <= ExecutionEncodedLen {
		return nil
	}
	return witness[ExecutionEncodedLen:]
}
