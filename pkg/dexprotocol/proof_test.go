// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/luxfi/ids"
)

// proof_test.go pins the three-rule proof system and, above all, that
// NON-INCLUSION IS FIRST-CLASS: release carries a real proof against the same
// committed root, verified by the same code path as settle.

func randID(rng *rand.Rand) ids.ID {
	var id ids.ID
	rng.Read(id[:])
	return id
}

// TestNonInclusionIsFirstClass is the headline property. Absence must be provable
// against the committed root with the same force as presence — otherwise release
// would have to be justified by a clock or a local view, both forbidden.
func TestNonInclusionIsFirstClass(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	set := NewExecSet()
	present := make([]ids.ID, 0, 16)
	for i := 0; i < 16; i++ {
		id := randID(rng)
		set.Add(id)
		present = append(present, id)
	}
	root := set.Root()

	for _, id := range present {
		p := set.Prove(id)
		if !p.Included {
			t.Fatalf("a present key must prove INCLUDED")
		}
		if err := VerifyExecProof(root, id, p); err != nil {
			t.Fatalf("inclusion proof failed to verify: %v", err)
		}
	}
	for i := 0; i < 64; i++ {
		absent := randID(rng)
		p := set.Prove(absent)
		if p.Included {
			t.Fatalf("an absent key must prove NOT included")
		}
		if err := VerifyExecProof(root, absent, p); err != nil {
			t.Fatalf("NON-INCLUSION proof failed to verify: %v — release would be unprovable", err)
		}
	}
}

// TestEmptyExecRootProvesEverythingAbsent: a block that consumed nothing still
// commits to that, and every execution reserved against its parent is releasable.
func TestEmptyExecRootProvesEverythingAbsent(t *testing.T) {
	set := NewExecSet()
	root := set.Root()
	rng := rand.New(rand.NewSource(11))
	for i := 0; i < 32; i++ {
		id := randID(rng)
		p := set.Prove(id)
		if p.Included {
			t.Fatal("the empty set must not claim inclusion")
		}
		if err := VerifyExecProof(root, id, p); err != nil {
			t.Fatalf("non-inclusion against an empty root must verify: %v", err)
		}
	}
}

// TestForgedProofsAreRefused: neither answer may be asserted without the path.
func TestForgedProofsAreRefused(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	set := NewExecSet()
	real := randID(rng)
	set.Add(real)
	for i := 0; i < 8; i++ {
		set.Add(randID(rng))
	}
	root := set.Root()
	absent := randID(rng)

	// Claiming inclusion for an absent key using its own (non-inclusion) path.
	p := set.Prove(absent)
	p.Included = true
	if err := VerifyExecProof(root, absent, p); !errors.Is(err, ErrProofMismatch) {
		t.Fatalf("flipping Included must break the reconstruction, got %v", err)
	}

	// Claiming absence for a present key.
	q := set.Prove(real)
	q.Included = false
	if err := VerifyExecProof(root, real, q); !errors.Is(err, ErrProofMismatch) {
		t.Fatalf("claiming a present key is absent must be refused, got %v", err)
	}

	// A tampered sibling.
	r := set.Prove(real)
	if len(r.Siblings) > 0 {
		r.Siblings[0][0] ^= 1
		if err := VerifyExecProof(root, real, r); !errors.Is(err, ErrProofMismatch) {
			t.Fatalf("a tampered sibling must be refused, got %v", err)
		}
	}

	// A bitmap that disagrees with the sibling count.
	s := set.Prove(real)
	s.Siblings = s.Siblings[:len(s.Siblings)-1]
	if err := VerifyExecProof(root, real, s); !errors.Is(err, ErrProofMalformed) {
		t.Fatalf("a bitmap/sibling mismatch must be refused, got %v", err)
	}
}

// TestSettleRequiresInclusionReleaseRequiresNonInclusion pins that the two
// transitions cannot borrow each other's proof.
func TestSettleRequiresInclusionReleaseRequiresNonInclusion(t *testing.T) {
	e := sampleExecution()
	consumed, incl := acceptedConsuming(t, testCBlock, testParent, e.ExecID)
	_, nonIncl := acceptedConsuming(t, testCBlock, testParent)

	l := NewLedger()
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := l.Settle(e.ExecID, consumed, nonIncl); !errors.Is(err, ErrWrongProof) {
		t.Fatalf("settling with a NON-inclusion proof must be refused, got %v", err)
	}
	if _, err := l.Release(e.ExecID, consumed, incl); !errors.Is(err, ErrWrongProof) {
		t.Fatalf("releasing with an INCLUSION proof must be refused, got %v", err)
	}
	if r, _, _ := l.Counts(); r != 1 {
		t.Fatal("refused transitions must leave the execution reserved")
	}
}

// TestUnverifiedParentCannotSettleOrRelease closes the zero-value hole. A struct
// with unexported fields is still zero-constructible outside the package, so the
// verified sentinel is what actually enforces "unverified C parents must not
// compile" at the point where it matters.
func TestUnverifiedParentCannotSettleOrRelease(t *testing.T) {
	e := sampleExecution()
	_, incl := acceptedConsuming(t, testCBlock, testParent, e.ExecID)

	l := NewLedger()
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	var forged AcceptedBlock // the zero value an outside package can build
	if _, err := l.Settle(e.ExecID, forged, incl); !errors.Is(err, ErrAcceptedUnverified) {
		t.Fatalf("an unverified AcceptedBlock must never settle, got %v", err)
	}
	if _, err := l.Release(e.ExecID, forged, incl); !errors.Is(err, ErrAcceptedUnverified) {
		t.Fatalf("an unverified AcceptedBlock must never release, got %v", err)
	}
}

// TestVerifyAcceptedBlockIsTheOnlyProducer: the acceptance statement is
// authenticated, fixed width, and scoped.
func TestVerifyAcceptedBlockIsTheOnlyProducer(t *testing.T) {
	set := NewExecSet()
	msg := EncodeAcceptedBlock(testCBlock, testParent, 9, set.Root())

	if _, err := VerifyAcceptedBlock(msg, []byte("c"), nil); !errors.Is(err, ErrNoVerifier) {
		t.Fatalf("no verifier must be refused, got %v", err)
	}
	if _, err := VerifyAcceptedBlock(msg, []byte("c"), rejectVerifier{}); !errors.Is(err, ErrBadCertificate) {
		t.Fatalf("a bad certificate must be refused, got %v", err)
	}
	if _, err := VerifyAcceptedBlock(msg[:len(msg)-1], []byte("c"), okVerifier{}); !errors.Is(err, ErrAcceptedWidth) {
		t.Fatalf("a short statement must be refused, got %v", err)
	}
	bad := EncodeAcceptedBlock(ids.Empty, testParent, 9, set.Root())
	if _, err := VerifyAcceptedBlock(bad, []byte("c"), okVerifier{}); !errors.Is(err, ErrAcceptedScope) {
		t.Fatalf("an empty block id must be refused, got %v", err)
	}

	ab, err := VerifyAcceptedBlock(msg, []byte("c"), okVerifier{})
	if err != nil {
		t.Fatalf("a well-formed acceptance must verify: %v", err)
	}
	if ab.ID() != testCBlock || ab.ParentID() != testParent || ab.Height() != 9 {
		t.Fatalf("decoded acceptance is wrong: %+v", ab)
	}
}

// TestUselessButResolvableReservation is the edge case the ruling calls out. If D
// reserves against P only after C has already accepted Q, the reservation can never
// settle — but it MUST still be releasable, or liquidity strands. Correctness does
// not depend on D winning that race.
func TestUselessButResolvableReservation(t *testing.T) {
	e := sampleExecution() // scoped to testParent (== P)
	// Q was accepted before the reservation was taken, and did not consume E.
	q, nonIncl := acceptedConsuming(t, testCBlock, testParent)

	l := NewLedger()
	if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := l.Release(e.ExecID, q, nonIncl); err != nil {
		t.Fatalf("a useless reservation MUST still be resolvable by anyone: %v", err)
	}
	if _, _, rel := l.Counts(); rel != 1 {
		t.Fatal("the reservation should have been released")
	}
}

// TestExecRootIsOrderIndependent: the commitment is a function of the SET, so two
// validators that consumed the same executions in different orders commit to the
// same root.
func TestExecRootIsOrderIndependent(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	ids8 := make([]ids.ID, 8)
	for i := range ids8 {
		ids8[i] = randID(rng)
	}
	a := NewExecSet()
	for _, id := range ids8 {
		a.Add(id)
	}
	b := NewExecSet()
	for i := len(ids8) - 1; i >= 0; i-- {
		b.Add(ids8[i])
	}
	if a.Root() != b.Root() {
		t.Fatal("ExecRoot must be a function of the set, not of insertion order")
	}
}

// TestLeafAndNodeDomainsAreSeparated guards the classic second-preimage attack: a
// 64-byte value must not be reinterpretable as either a leaf or an internal node.
func TestLeafAndNodeDomainsAreSeparated(t *testing.T) {
	if leafDomain == nodeDomain {
		t.Fatal("leaf and node domains must differ")
	}
	var z ids.ID
	if hashLeaf(z) == hashNode(emptyHash[0], emptyHash[0]) {
		t.Fatal("a leaf hash collided with an internal node hash")
	}
	if emptyHash[0] == hashLeaf(z) {
		t.Fatal("the empty leaf must be distinct from any populated leaf")
	}
}

// TestRule1_ReserveRequiresTheAcceptedParent pins rule 1 of the proof system:
//
//	Accepted(P)  permits Reserve(E) with E.CParent == P
//
// Both halves are enforced. Without the first, a reservation could be taken
// against a merely proposed block, which is the abandoned-parent case the
// accepted-only rule exists to make nonexistent. Without the second, a reservation
// could be taken against the wrong opportunity entirely.
func TestRule1_ReserveRequiresTheAcceptedParent(t *testing.T) {
	e := sampleExecution() // scoped to testParent

	t.Run("unverified parent is refused", func(t *testing.T) {
		l := NewLedger()
		var proposed AcceptedBlock // the zero value an outside package can build
		if _, err := l.Reserve(proposed, verified(t, e)); !errors.Is(err, ErrAcceptedUnverified) {
			t.Fatalf("reserving against an unverified parent must be refused, got %v", err)
		}
		if r, _, _ := l.Counts(); r != 0 {
			t.Fatal("a refused reserve must not enter the ledger")
		}
	})

	t.Run("wrong accepted parent is refused", func(t *testing.T) {
		other := testParent
		other[0] ^= 1
		msg := EncodeAcceptedBlock(other, ids.ID{0x0B}, 1, NewExecSet().Root())
		wrong, err := VerifyAcceptedBlock(msg, []byte("cert"), okVerifier{})
		if err != nil {
			t.Fatalf("VerifyAcceptedBlock: %v", err)
		}
		l := NewLedger()
		if _, err := l.Reserve(wrong, verified(t, e)); !errors.Is(err, ErrWrongParent) {
			t.Fatalf("reserving against the wrong accepted parent must be refused, got %v", err)
		}
	})

	t.Run("the accepted parent it is scoped to succeeds", func(t *testing.T) {
		l := NewLedger()
		if _, err := l.Reserve(acceptedParent(t), verified(t, e)); err != nil {
			t.Fatalf("reserving against the correct accepted parent must succeed: %v", err)
		}
	})
}
