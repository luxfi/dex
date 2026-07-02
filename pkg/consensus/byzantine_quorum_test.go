// Copyright (C) 2020-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// byzantine_quorum_test.go proves the f < n/3 finality boundary for DEX
// SETTLEMENT under byzantine validators. Settlement (chains/dexvm: DexConsensusMode
// QUORUM_FINALITY) finalizes an attested fill-set only on a VERIFIED alpha-of-K
// quorum: at least `q` DISTINCT validators must sign the SAME settlement
// commitment (the fill-set / execution root), with each honest validator signing
// AT MOST ONE root per height (non-equivocation). This file models that rule with
// REAL Ed25519 validator signatures and drives the three byzantine faults the
// threat model names — EQUIVOCATION, WITHHOLDING, and FORGED votes — to establish
// the exact boundary:
//
//	q(n) = ⌊2n/3⌋ + 1                 (canonical BFT quorum: strictly > 2/3)
//	f*(n) = ⌊(n-1)/3⌋                 (max tolerated byzantine faults)
//
//	SAFETY   holds iff any two quorums intersect in an honest validator:
//	         2q - n > f. A non-equivocating honest validator never signs two
//	         conflicting roots, so two conflicting quorums cannot both form.
//	LIVENESS holds iff the honest set ALONE reaches quorum despite f withholders:
//	         n - f >= q.
//	BFT      is achievable iff BOTH are simultaneously satisfiable, iff f <= f*(n).
//
// The tests prove the task's exact cases numerically: n=4 tolerates 1 but FAILS at
// 2; n=7 tolerates 2 but FAILS at 3; plus n=10 (tolerates 3, fails at 4). Every
// signature is a genuine Ed25519 verify (a forged/outsider vote never counts), and
// keys are generated from a fixed seed so the run is fully deterministic (no
// wall-clock, no unseeded randomness).

package consensus

import (
	"crypto/sha256"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/luxfi/crypto/ed25519"
)

// quorumThreshold is the canonical BFT quorum for n validators: strictly more than
// two-thirds must sign. ⌊2n/3⌋+1 == ⌈(2n+1)/3⌉ (n=4→3, n=7→5, n=10→7).
func quorumThreshold(n int) int { return 2*n/3 + 1 }

// maxByzantine is the classical BFT fault ceiling f* = ⌊(n-1)/3⌋ (n=4→1, n=7→2,
// n=10→3).
func maxByzantine(n int) int { return (n - 1) / 3 }

// bftValidator is a settlement validator with a real Ed25519 keypair. The id is
// its index in the registered set; a vote only counts if it is signed by a
// REGISTERED validator's key (a forged/outsider signature never contributes to a
// quorum — the threat-model's forged-vote fault).
type bftValidator struct {
	id   int
	pub  ed25519.PublicKey
	priv ed25519.PrivateKey
}

// bftVote is one validator's signature over a settlement commitment (root) at a
// height. Byzantine validators may emit two votes over conflicting roots
// (equivocation) or none (withholding).
type bftVote struct {
	signer int
	root   [32]byte
	sig    []byte
}

// settlementMsg is the canonical bytes a validator signs to attest a settlement
// commitment: height binds the vote to one round so a signature cannot be
// replayed onto another height.
func settlementMsg(height uint64, root [32]byte) []byte {
	var m [40]byte
	binary.BigEndian.PutUint64(m[0:8], height)
	copy(m[8:40], root[:])
	sum := sha256.Sum256(m[:])
	return sum[:]
}

// newValidatorSet builds n validators with deterministic (seeded) Ed25519 keys, so
// the whole test is reproducible. Key VALUES do not matter to the assertions (they
// are threshold/verify statements); seeding only removes unseeded randomness.
func newValidatorSet(t *testing.T, n int, seed int64) []bftValidator {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	vals := make([]bftValidator, n)
	for i := 0; i < n; i++ {
		pub, priv, err := ed25519.GenerateKey(rng)
		if err != nil {
			t.Fatalf("validator %d keygen: %v", i, err)
		}
		vals[i] = bftValidator{id: i, pub: pub, priv: priv}
	}
	return vals
}

// sign produces validator i's vote over (height, root).
func (v bftValidator) sign(height uint64, root [32]byte) bftVote {
	return bftVote{signer: v.id, root: root, sig: ed25519.Sign(v.priv, settlementMsg(height, root))}
}

// certifyRoot counts the DISTINCT registered validators with a VALID Ed25519
// signature over (height, root) and reports whether that reaches quorum q. This is
// the settlement finality check: a fill-set commitment is final iff >= q distinct
// validators verifiably signed it. Forged (unregistered / bad) signatures and
// duplicate votes from one signer never inflate the count.
func certifyRoot(vals []bftValidator, votes []bftVote, height uint64, root [32]byte, q int) (int, bool) {
	byID := make(map[int]ed25519.PublicKey, len(vals))
	for _, v := range vals {
		byID[v.id] = v.pub
	}
	msg := settlementMsg(height, root)
	counted := make(map[int]bool)
	for _, vote := range votes {
		if vote.root != root {
			continue // a vote for a different root does not certify THIS root
		}
		pub, registered := byID[vote.signer]
		if !registered {
			continue // forged/outsider vote — not a member of the validator set
		}
		if counted[vote.signer] {
			continue // one signer contributes at most once (no self-inflation)
		}
		if !ed25519.Verify(pub, msg, vote.sig) {
			continue // signature does not verify — never counts
		}
		counted[vote.signer] = true
	}
	return len(counted), len(counted) >= q
}

// bftFeasible reports whether SOME quorum threshold q in [1,n] gives both SAFETY
// (2q-n > f) and LIVENESS (n-f >= q) against f byzantine faults. This is the
// executable statement of the BFT bound; the test asserts it equals f <= f*(n).
func bftFeasible(n, f int) bool {
	for q := 1; q <= n; q++ {
		safety := 2*q-n > f
		liveness := n-f >= q
		if safety && liveness {
			return true
		}
	}
	return false
}

// TestByzantineQuorum_FinalityBoundary is the executable BFT bound: for a table of
// (n, f) it proves a safe+live quorum threshold EXISTS iff f <= ⌊(n-1)/3⌋, and
// pins the task's exact cases (n=4: tolerate 1, fail 2; n=7: tolerate 2, fail 3).
func TestByzantineQuorum_FinalityBoundary(t *testing.T) {
	cases := []struct {
		n, f      int
		tolerated bool
	}{
		{4, 0, true}, {4, 1, true}, {4, 2, false}, {4, 3, false},
		{7, 1, true}, {7, 2, true}, {7, 3, false},
		{10, 2, true}, {10, 3, true}, {10, 4, false},
	}
	for _, c := range cases {
		feasible := bftFeasible(c.n, c.f)
		boundOK := c.f <= maxByzantine(c.n)
		if feasible != boundOK {
			t.Fatalf("n=%d f=%d: bftFeasible=%v but f<=f*(=%d) is %v (BFT bound and search disagree)",
				c.n, c.f, feasible, maxByzantine(c.n), boundOK)
		}
		if feasible != c.tolerated {
			t.Fatalf("n=%d f=%d: tolerated=%v, want %v (q(n)=%d, f*(n)=%d)",
				c.n, c.f, feasible, c.tolerated, quorumThreshold(c.n), maxByzantine(c.n))
		}
		t.Logf("n=%2d f=%d  q(n)=%d f*(n)=%d  tolerated=%v", c.n, c.f, quorumThreshold(c.n), maxByzantine(c.n), feasible)
	}
}

// TestByzantineQuorum_EquivocationNoDoubleFinalize proves SAFETY at and beyond the
// boundary with REAL signatures. f byzantine validators EQUIVOCATE — each signs
// BOTH of two conflicting settlement roots R_A and R_B — while the honest set is
// adversarially SPLIT across A and B (the worst case the network partition can
// produce). At the canonical quorum q(n):
//
//   - f <= f*(n): NO honest split lets both R_A and R_B certify — the two quorums
//     would have to intersect in an honest validator, who signed only one root. So
//     at most one root is ever final: no double-spend of the settlement.
//   - f  = f*(n)+1: to keep the chain LIVE the quorum must drop to n-f, and then a
//     concrete split double-finalizes — safety is unsalvageable. Demonstrated.
func TestByzantineQuorum_EquivocationNoDoubleFinalize(t *testing.T) {
	const height = 7
	rootA := sha256.Sum256([]byte("fill-set-A"))
	rootB := sha256.Sum256([]byte("fill-set-B"))

	// worstEquivocation returns whether SOME honest split lets both conflicting
	// roots reach quorum q, given f equivocating byzantine validators.
	worstEquivocation := func(t *testing.T, n, f, q int) bool {
		t.Helper()
		vals := newValidatorSet(t, n, int64(1000+n*10+f))
		byz := vals[:f]     // byzantine equivocators
		honest := vals[f:]  // honest, each signs exactly ONE root
		nHonest := len(honest)
		for splitA := 0; splitA <= nHonest; splitA++ {
			votes := make([]bftVote, 0, 2*f+nHonest)
			// Byzantine equivocate: sign BOTH roots.
			for _, b := range byz {
				votes = append(votes, b.sign(height, rootA), b.sign(height, rootB))
			}
			// Honest split: splitA sign A, the rest sign B. Each honest signs ONE.
			for i, h := range honest {
				if i < splitA {
					votes = append(votes, h.sign(height, rootA))
				} else {
					votes = append(votes, h.sign(height, rootB))
				}
			}
			_, aFinal := certifyRoot(vals, votes, height, rootA, q)
			_, bFinal := certifyRoot(vals, votes, height, rootB, q)
			if aFinal && bFinal {
				return true // conflicting double-finalize at this split
			}
		}
		return false
	}

	type tc struct{ n, f int }
	safe := []tc{{4, 1}, {7, 2}, {10, 3}} // f == f*(n): must be SAFE at canonical q
	for _, c := range safe {
		q := quorumThreshold(c.n)
		if worstEquivocation(t, c.n, c.f, q) {
			t.Fatalf("SAFETY VIOLATED: n=%d f=%d q=%d — an equivocation split double-finalized two conflicting roots",
				c.n, c.f, q)
		}
		t.Logf("SAFE  n=%2d f=%d q=%d: no equivocation split double-finalizes (honest quorum intersection)", c.n, c.f, q)
	}

	// Beyond the boundary (f = f*+1): the canonical quorum can no longer be reached
	// by the honest set alone (liveness needs q <= n-f), and at that liveness-forced
	// quorum a concrete equivocation split DOES double-finalize — proving the
	// boundary is tight (n=4 fails at 2).
	broken := []tc{{4, 2}, {7, 3}, {10, 4}}
	for _, c := range broken {
		qLive := c.n - c.f // the largest quorum that still lets honest-alone finalize
		if qLive < 1 {
			qLive = 1
		}
		if !worstEquivocation(t, c.n, c.f, qLive) {
			t.Fatalf("expected a safety break at n=%d f=%d when q is forced to the liveness bound %d, found none",
				c.n, c.f, qLive)
		}
		t.Logf("BROKEN n=%2d f=%d: f>f*(=%d); at liveness-forced q=%d two conflicting roots BOTH finalize (no safe+live q)",
			c.n, c.f, maxByzantine(c.n), qLive)
	}
}

// TestByzantineQuorum_WithholdingLiveness proves the LIVENESS half with real
// signatures. f byzantine validators WITHHOLD (emit no vote); the honest set signs
// the one correct settlement root. At the canonical quorum:
//
//   - f <= f*(n): the honest set alone reaches quorum — settlement finalizes.
//   - f  = f*(n)+1: honest-alone falls below quorum — settlement STALLS (the chain
//     cannot be forced to finalize an unsafe root, so it halts instead).
func TestByzantineQuorum_WithholdingLiveness(t *testing.T) {
	const height = 11
	root := sha256.Sum256([]byte("the-one-true-fill-set"))

	certifyWithHonestOnly := func(t *testing.T, n, f int) (int, bool) {
		t.Helper()
		vals := newValidatorSet(t, n, int64(2000+n*10+f))
		votes := make([]bftVote, 0, n-f)
		for _, h := range vals[f:] { // only honest validators sign
			votes = append(votes, h.sign(height, root))
		}
		return certifyRoot(vals, votes, height, root, quorumThreshold(n))
	}

	live := []struct{ n, f int }{{4, 1}, {7, 2}, {10, 3}}
	for _, c := range live {
		got, ok := certifyWithHonestOnly(t, c.n, c.f)
		if !ok {
			t.Fatalf("LIVENESS FAILED: n=%d f=%d withholders — honest-alone signers=%d < q=%d",
				c.n, c.f, got, quorumThreshold(c.n))
		}
		t.Logf("LIVE  n=%2d f=%d withhold: honest signers=%d >= q=%d — finalizes", c.n, c.f, got, quorumThreshold(c.n))
	}

	stall := []struct{ n, f int }{{4, 2}, {7, 3}, {10, 4}}
	for _, c := range stall {
		got, ok := certifyWithHonestOnly(t, c.n, c.f)
		if ok {
			t.Fatalf("n=%d f=%d: honest-alone certified (%d>=q) — a >f* withholding set should stall, not finalize",
				c.n, c.f, got)
		}
		t.Logf("STALL n=%2d f=%d withhold: honest signers=%d < q=%d — halts (safety over liveness)", c.n, c.f, got, quorumThreshold(c.n))
	}
}

// TestByzantineQuorum_ForgedVotesDoNotCount proves a byzantine actor cannot
// manufacture a quorum out of FORGED votes (threat #2 at the quorum layer). With
// only q-1 honest signatures (one short of finality), any number of (a) outsider
// signatures from unregistered keys, (b) tampered-root votes, or (c) bad-signature
// votes from registered validators fail to lift the tally to quorum — only genuine
// verified signatures from distinct registered validators count.
func TestByzantineQuorum_ForgedVotesDoNotCount(t *testing.T) {
	const (
		n      = 7
		height = 3
	)
	q := quorumThreshold(n) // 5
	root := sha256.Sum256([]byte("real-fill-set"))
	vals := newValidatorSet(t, n, 4242)

	// q-1 = 4 genuine honest votes: one short of quorum.
	votes := make([]bftVote, 0)
	for i := 0; i < q-1; i++ {
		votes = append(votes, vals[i].sign(height, root))
	}
	if _, ok := certifyRoot(vals, votes, height, root, q); ok {
		t.Fatalf("precondition wrong: %d genuine votes already certify at q=%d", q-1, q)
	}

	// (a) Outsider forgeries: unregistered keys signing the true root.
	outsiders := newValidatorSet(t, 5, 999)
	for i := range outsiders {
		outsiders[i].id = 100 + i // ids NOT in the registered set
		votes = append(votes, outsiders[i].sign(height, root))
	}
	// (b) A registered validator's vote with a TAMPERED signature (bit-flipped).
	bad := vals[q-1].sign(height, root)
	bad.sig[0] ^= 0xff
	votes = append(votes, bad)
	// (c) A registered validator signing a DIFFERENT root (won't match this root).
	other := sha256.Sum256([]byte("other-fill-set"))
	votes = append(votes, vals[q-1].sign(height, other))

	got, ok := certifyRoot(vals, votes, height, root, q)
	if ok {
		t.Fatalf("FORGED QUORUM: certified with only %d genuine votes + forgeries (count=%d, q=%d) — "+
			"forged votes must never contribute to settlement finality", q-1, got, q)
	}
	if got != q-1 {
		t.Fatalf("forged/duplicate/tampered votes leaked into the tally: counted=%d, want exactly %d genuine", got, q-1)
	}

	// The one honest vote that was actually missing lifts it to quorum — proving the
	// gate is not merely rejecting everything.
	votes = append(votes, vals[q-1].sign(height, root))
	if got, ok := certifyRoot(vals, votes, height, root, q); !ok {
		t.Fatalf("adding the q-th GENUINE vote must certify: counted=%d q=%d", got, q)
	}
}
