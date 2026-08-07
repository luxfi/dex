// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"math/rand"
	"testing"

	"github.com/luxfi/ids"
)

var (
	testCChain = ids.ID{0xCC}
	testParent = ids.ID{0x0C}
	testCBlock = ids.ID{0xCB}
)

// okVerifier accepts any certificate. Verification of the certificate itself is the
// D-attestation layer's job; these tests pin the surrounding protocol.
type okVerifier struct{}

func (okVerifier) VerifyCertificate(ids.ID, []byte) error { return nil }

type rejectVerifier struct{}

func (rejectVerifier) VerifyCertificate(ids.ID, []byte) error { return errors.New("nope") }

func sampleExecution() Execution {
	return Execution{
		ExecID:   ids.ID{0xE1},
		OrderID:  ids.ID{0x01},
		Price:    10_000,
		Quantity: 250,
		Fee:      7,
		CParent:  testParent,
		DBlock:   ids.ID{0xDB},
	}
}

func witnessFor(e Execution) []byte {
	return append(e.Encode(), []byte("certificate-bytes")...)
}

// acceptedConsuming builds a verified AcceptedBlock whose ExecRoot contains exactly
// the given executions, plus a proof for the FIRST one (inclusion if it was listed,
// non-inclusion otherwise). This is how a relayer would present the two rules.
func acceptedConsuming(t *testing.T, id, parent ids.ID, consumed ...ids.ID) (AcceptedBlock, ExecProof) {
	t.Helper()
	set := NewExecSet()
	for _, c := range consumed {
		set.Add(c)
	}
	root := set.Root()
	msg := EncodeAcceptedBlock(id, parent, 1, root)
	ab, err := VerifyAcceptedBlock(msg, []byte("cert"), okVerifier{})
	if err != nil {
		t.Fatalf("VerifyAcceptedBlock: %v", err)
	}
	target := sampleExecution().ExecID
	if len(consumed) > 0 {
		target = consumed[0]
	}
	return ab, set.Prove(target)
}

// acceptedParent returns a verified AcceptedBlock for P — the parent every sample
// execution is scoped to. Rule 1 requires one to reserve at all.
func acceptedParent(t *testing.T) AcceptedBlock {
	t.Helper()
	msg := EncodeAcceptedBlock(testParent, ids.ID{0x0B}, 1, NewExecSet().Root())
	ab, err := VerifyAcceptedBlock(msg, []byte("cert"), okVerifier{})
	if err != nil {
		t.Fatalf("VerifyAcceptedBlock(parent): %v", err)
	}
	return ab
}

func verified(t *testing.T, e Execution) VerifiedExecution {
	t.Helper()
	v, err := VerifyExecution(witnessFor(e), VerifyContext{
		NetworkID: 3, CChainID: testCChain, DChainID: ids.ID{0xDD},
		CParent: e.CParent, Verifier: okVerifier{},
	})
	if err != nil {
		t.Fatalf("VerifyExecution: %v", err)
	}
	return v
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	e := sampleExecution()
	b := e.Encode()
	if len(b) != ExecutionEncodedLen {
		t.Fatalf("encoded width %d, want %d", len(b), ExecutionEncodedLen)
	}
	got, err := DecodeExecution(b)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got != e {
		t.Fatalf("round trip changed the execution:\n got %+v\nwant %+v", got, e)
	}
}

// TestDecodeRejectsWrongWidth: a longer buffer is REFUSED, not truncated. Accepting
// a suffix would let two distinct wire messages decode to the same execution while
// carrying different attested bytes.
func TestDecodeRejectsWrongWidth(t *testing.T) {
	se := sampleExecution()
	b := se.Encode()
	for name, in := range map[string][]byte{
		"short": b[:len(b)-1],
		"long":  append(append([]byte(nil), b...), 0x00),
		"empty": nil,
	} {
		if _, err := DecodeExecution(in); !errors.Is(err, ErrExecWidth) {
			t.Fatalf("%s: want ErrExecWidth, got %v", name, err)
		}
	}
}

// TestCommitmentIsInjectiveOverEveryField is the property the attestation rests on:
// changing ANY field must change the commitment. If one field did not reach the
// preimage, a validator's signature over execution A would also be valid for a
// different execution B — for example the same price and quantity bound to a
// different CParent, which is a double-settle of the reservation.
func TestCommitmentIsInjectiveOverEveryField(t *testing.T) {
	base := sampleExecution()
	baseC := base.Commitment()
	for name, mutate := range map[string]func(*Execution){
		"ExecID":   func(e *Execution) { e.ExecID[31] ^= 1 },
		"OrderID":  func(e *Execution) { e.OrderID[31] ^= 1 },
		"Price":    func(e *Execution) { e.Price ^= 1 },
		"Quantity": func(e *Execution) { e.Quantity ^= 1 },
		"Fee":      func(e *Execution) { e.Fee ^= 1 },
		"CParent":  func(e *Execution) { e.CParent[31] ^= 1 },
		"DBlock":   func(e *Execution) { e.DBlock[31] ^= 1 },
	} {
		e := base
		mutate(&e)
		if e.Commitment() == baseC {
			t.Fatalf("mutating %s did not change the commitment — that field is NOT covered "+
				"by the attestation and can be substituted under a valid signature", name)
		}
	}
}

// TestEncodingIsInjective: distinct executions must give distinct BYTES, with no
// parsing argument required. That is what lets the commitment rest on SHA-256 alone.
func TestEncodingIsInjective(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	seen := make(map[string]Execution, 4096)
	for i := 0; i < 4096; i++ {
		e := Execution{Price: rng.Uint64(), Quantity: rng.Uint64(), Fee: rng.Uint64()}
		rng.Read(e.ExecID[:])
		rng.Read(e.OrderID[:])
		rng.Read(e.CParent[:])
		rng.Read(e.DBlock[:])
		key := string(e.Encode())
		if prev, ok := seen[key]; ok && prev != e {
			t.Fatalf("two distinct executions encoded identically:\n%+v\n%+v", prev, e)
		}
		seen[key] = e
	}
}

// TestDomainSeparation: the commitment must not be a bare hash of the encoding, or
// any other protocol hashing a same-width structure could produce a value a D
// validator has already signed.
func TestDomainSeparation(t *testing.T) {
	e := sampleExecution()
	bare := sha256.Sum256(e.Encode())
	c := e.Commitment()
	if bytes.Equal(c[:], bare[:]) {
		t.Fatal("commitment equals the bare hash of the encoding — the domain tag is not applied")
	}
	if execDomainTag != sha256.Sum256([]byte("lux.dex.execution.v1")) {
		t.Fatal("execDomainTag is not sha256 of the documented domain string")
	}
}

// TestVerifyUsesTheVerifiersParentNotTheWitnesss is the at-most-once binding. A
// witness that could assert its own scope would be asserting the one thing the
// property depends on.
func TestVerifyUsesTheVerifiersParentNotTheWitnesss(t *testing.T) {
	e := sampleExecution()
	other := testParent
	other[0] ^= 1
	_, err := VerifyExecution(witnessFor(e), VerifyContext{
		CParent: other, Verifier: okVerifier{},
	})
	if !errors.Is(err, ErrWrongCParent) {
		t.Fatalf("an execution scoped to a different C opportunity must be refused, got %v", err)
	}
}

func TestVerifyRequiresACertificateAndAVerifier(t *testing.T) {
	e := sampleExecution()
	if _, err := VerifyExecution(witnessFor(e), VerifyContext{CParent: e.CParent}); !errors.Is(err, ErrNoVerifier) {
		t.Fatalf("verification without a verifier must be refused, got %v", err)
	}
	_, err := VerifyExecution(witnessFor(e), VerifyContext{CParent: e.CParent, Verifier: rejectVerifier{}})
	if !errors.Is(err, ErrBadCertificate) {
		t.Fatalf("a bad certificate must be refused, got %v", err)
	}
}

// TestSettleAndReleaseAreMutuallyExclusiveAndTerminal is the property that makes
// the shipping double-spend unwritable. Note what this test CANNOT express: after
// Settle there is no ReservedExecution left, so "settle then release" is not a
// runtime rejection — the ledger has nothing to hand back.
func TestSettleAndReleaseAreMutuallyExclusiveAndTerminal(t *testing.T) {
	e0 := sampleExecution()
	accepted, incl := acceptedConsuming(t, testCBlock, testParent, e0.ExecID)
	qEmpty, nonIncl := acceptedConsuming(t, testCBlock, testParent)

	t.Run("settled is terminal", func(t *testing.T) {
		l := NewLedger()
		e := sampleExecution()
		if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
			t.Fatalf("reserve: %v", err)
		}
		if _, err := l.Settle(e.ExecID, accepted, incl); err != nil {
			t.Fatalf("settle: %v", err)
		}
		if err := l.Exclusive(); err != nil {
			t.Fatalf("exclusivity broken after settle: %v", err)
		}
		if _, err := l.Release(e.ExecID, qEmpty, nonIncl); !errors.Is(err, ErrNotReserved) {
			t.Fatalf("releasing a SETTLED execution must find nothing reserved, got %v", err)
		}
		if _, err := l.Settle(e.ExecID, accepted, incl); !errors.Is(err, ErrNotReserved) {
			t.Fatalf("settling twice must find nothing reserved, got %v", err)
		}
		if r, tr, rel := l.Counts(); r != 0 || tr != 1 || rel != 0 {
			t.Fatalf("counts reserved=%d traded=%d released=%d, want 0/1/0", r, tr, rel)
		}
	})

	t.Run("released is terminal", func(t *testing.T) {
		l := NewLedger()
		e := sampleExecution()
		if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
			t.Fatalf("reserve: %v", err)
		}
		if _, err := l.Release(e.ExecID, qEmpty, nonIncl); err != nil {
			t.Fatalf("release: %v", err)
		}
		if _, err := l.Settle(e.ExecID, accepted, incl); !errors.Is(err, ErrNotReserved) {
			t.Fatalf("Reserved -> Released -> Traded MUST be impossible, got %v", err)
		}
		if err := l.Exclusive(); err != nil {
			t.Fatalf("exclusivity broken after release: %v", err)
		}
	})
}

// TestReplayedCertificateCannotReReserve: once terminal, an ExecID is spent
// forever. Without this, re-presenting the same certificate would take the
// reservation a second time.
func TestReplayedCertificateCannotReReserve(t *testing.T) {
	e := sampleExecution()
	accepted, incl := acceptedConsuming(t, testCBlock, testParent, e.ExecID)
	l := NewLedger()
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := l.Settle(e.ExecID, accepted, incl); err != nil {
		t.Fatalf("settle: %v", err)
	}
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); !errors.Is(err, ErrTerminalExists) {
		t.Fatalf("re-reserving a terminal ExecID must be refused, got %v", err)
	}
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); !errors.Is(err, ErrTerminalExists) {
		t.Fatalf("still refused on a second attempt, got %v", err)
	}
}

func TestDoubleReserveRefused(t *testing.T) {
	l := NewLedger()
	e := sampleExecution()
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); !errors.Is(err, ErrAlreadyOpen) {
		t.Fatalf("double reserve must be refused, got %v", err)
	}
}

// TestSettleChecksTheAcceptedBlocksParent: even a caller holding a genuine
// AcceptedBlock cannot spend a reservation against a C opportunity it was not
// scoped to.
func TestSettleChecksTheAcceptedBlocksParent(t *testing.T) {
	wrongParent := testParent
	wrongParent[0] ^= 1
	e := sampleExecution()
	accepted, incl := acceptedConsuming(t, testCBlock, wrongParent, e.ExecID)

	l := NewLedger()
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := l.Settle(e.ExecID, accepted, incl); !errors.Is(err, ErrWrongSuccessor) {
		t.Fatalf("settling against the wrong C parent must be refused, got %v", err)
	}
	if r, _, _ := l.Counts(); r != 1 {
		t.Fatal("a refused settle must leave the execution reserved")
	}
}

// TestStorageDomainsAreDistinct pins that the three states are physically separate
// key spaces, not one key with a status field.
func TestStorageDomainsAreDistinct(t *testing.T) {
	e := sampleExecution()
	accepted, incl := acceptedConsuming(t, testCBlock, testParent, e.ExecID)
	qEmpty, nonIncl := acceptedConsuming(t, testCBlock, testParent)

	l := NewLedger()
	r, _ := l.Reserve(acceptedParent(t), verified(t, e))
	tr, err := l.Settle(e.ExecID, accepted, incl)
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	l2 := NewLedger()
	r2, _ := l2.Reserve(acceptedParent(t), verified(t, e))
	rel, err := l2.Release(e.ExecID, qEmpty, nonIncl)
	if err != nil {
		t.Fatalf("release: %v", err)
	}
	keys := map[string]string{"reserved": r.Key(), "traded": tr.Key(), "released": rel.Key()}
	seen := map[string]string{}
	for name, k := range keys {
		if prev, ok := seen[k]; ok {
			t.Fatalf("%s and %s share the storage key %q — the domains are not distinct", prev, name, k)
		}
		seen[k] = name
	}
	if r2.Key() != r.Key() {
		t.Fatal("the reserved key must be a pure function of the ExecID")
	}
}

// TestReservationKeyIsScopedToTheParentOpportunity: the key must never involve a
// candidate block, or competing candidates would each take their own reservation.
func TestReservationKeyIsScopedToTheParentOpportunity(t *testing.T) {
	e := sampleExecution()
	a := ReservationKey(testCChain, e.CParent, e.ExecID)
	b := ReservationKey(testCChain, e.CParent, e.ExecID)
	if a != b {
		t.Fatal("the reservation key must be deterministic")
	}
	other := e.CParent
	other[0] ^= 1
	if ReservationKey(testCChain, other, e.ExecID) == a {
		t.Fatal("a different C parent must give a different reservation")
	}
}

func TestValidateRejectsUnscopedExecutions(t *testing.T) {
	for name, mutate := range map[string]func(*Execution){
		"zero CParent": func(e *Execution) { e.CParent = ids.Empty },
		"zero ExecID":  func(e *Execution) { e.ExecID = ids.Empty },
		"zero OrderID": func(e *Execution) { e.OrderID = ids.Empty },
	} {
		e := sampleExecution()
		mutate(&e)
		if err := e.Validate(); !errors.Is(err, ErrExecScope) {
			t.Fatalf("%s: want ErrExecScope, got %v", name, err)
		}
	}
	e := sampleExecution()
	e.Quantity = 0
	if err := e.Validate(); !errors.Is(err, ErrExecQty) {
		t.Fatalf("zero quantity must be refused, got %v", err)
	}
	ok := sampleExecution()
	if err := ok.Validate(); err != nil {
		t.Fatalf("a well-formed execution must validate, got %v", err)
	}
}

// TestNoStoppedStateOnTheWire: the provisional reservation is internal machinery
// and must never be reported to a FIX client. If a Stopped constant ever reappears
// in ExecType or OrdStatus, this fails.
func TestNoStoppedStateOnTheWire(t *testing.T) {
	for i := 0; i < 256; i++ {
		if ExecType(i).String() == "Stopped" {
			t.Fatalf("ExecType %d reports Stopped — the reservation must not surface to FIX clients", i)
		}
		if OrdStatus(i).String() == "Stopped" {
			t.Fatalf("OrdStatus %d reports Stopped — the reservation must not surface to FIX clients", i)
		}
	}
}
