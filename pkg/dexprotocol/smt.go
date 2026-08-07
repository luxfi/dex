// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexprotocol

import (
	"crypto/sha256"
	"errors"
	"fmt"

	"github.com/luxfi/ids"
)

// smt.go is the ExecRoot commitment: which Executions a C block consumed.
//
// NON-INCLUSION IS FIRST-CLASS HERE, NOT BOLTED ON. The protocol needs both
// answers with equal standing:
//
//	Accepted(Q), Q.Parent==P, E ∈ Q.ExecRoot   permits Settle(E) → Trade
//	Accepted(Q), Q.Parent==P, E ∉ Q.ExecRoot   permits Release(E)
//
// Release is not an exception path or a fallback — it is half the protocol, and it
// is the half that returns a trader's reserved liquidity. A commitment that can
// only prove membership would force release to be justified some other way, and
// every "other way" available is a clock or a local view, both of which are
// forbidden. So the structure is a sparse Merkle tree over the full 256-bit ExecID
// keyspace, where absence has exactly the same proof shape as presence: the path to
// the key, terminating in the empty leaf instead of a populated one. One verifier,
// one code path, one bit of difference.
//
// Because the tree is sparse, the overwhelming majority of siblings on any path are
// the precomputed empty-subtree hashes. The proof carries a bitmap marking which
// siblings are non-empty and omits the rest, so a proof is proportional to the
// number of executions actually in the block, not to 256.

const smtDepth = 256

// emptyHash[d] is the root of a completely empty subtree of height d.
// emptyHash[0] is the empty LEAF; emptyHash[smtDepth] is the empty ROOT.
var emptyHash = func() [smtDepth + 1][32]byte {
	var h [smtDepth + 1][32]byte
	// The empty leaf is a domain-separated constant, distinct from any populated
	// leaf, so "absent" can never be forged by presenting a leaf that happens to
	// hash to it.
	h[0] = sha256.Sum256([]byte("lux.dex.execroot.empty.v1"))
	for d := 1; d <= smtDepth; d++ {
		h[d] = hashNode(h[d-1], h[d-1])
	}
	return h
}()

// leafDomain and nodeDomain keep a leaf hash from ever being reinterpretable as an
// internal node. Without the split, a 64-byte preimage could be presented as
// either, which is the classic second-preimage attack on a Merkle tree.
var (
	leafDomain = sha256.Sum256([]byte("lux.dex.execroot.leaf.v1"))
	nodeDomain = sha256.Sum256([]byte("lux.dex.execroot.node.v1"))
)

func hashLeaf(key ids.ID) [32]byte {
	h := sha256.New()
	h.Write(leafDomain[:])
	h.Write(key[:])
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func hashNode(l, r [32]byte) [32]byte {
	h := sha256.New()
	h.Write(nodeDomain[:])
	h.Write(l[:])
	h.Write(r[:])
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

// bit reports the d-th bit of key, MSB first. Depth 0 is the most significant bit,
// so the traversal order is stable and canonical.
func bit(key ids.ID, d int) bool {
	return key[d/8]&(1<<(7-uint(d%8))) != 0
}

// ExecSet is the set of ExecIDs a C block consumed. It commits to membership only —
// the Execution's content is already bound by its own commitment, and re-committing
// it here would be a second place for the same fact to live.
type ExecSet struct {
	keys map[ids.ID]struct{}
}

func NewExecSet() *ExecSet { return &ExecSet{keys: make(map[ids.ID]struct{})} }

// Add records that the block consumed this execution.
func (s *ExecSet) Add(id ids.ID) { s.keys[id] = struct{}{} }

// Contains is the local view; a remote party must use a proof instead.
func (s *ExecSet) Contains(id ids.ID) bool {
	_, ok := s.keys[id]
	return ok
}

func (s *ExecSet) Len() int { return len(s.keys) }

// Root is the ExecRoot committed in the C block. An empty set has a well-defined
// root (the empty-subtree root), so a block that consumed nothing still commits to
// that fact and every execution can be proven ABSENT from it.
func (s *ExecSet) Root() [32]byte {
	keys := make([]ids.ID, 0, len(s.keys))
	for k := range s.keys {
		keys = append(keys, k)
	}
	return s.subtree(keys, 0)
}

// subtree computes the root of the subtree at depth d containing exactly keys.
// Recursion terminates early on an empty key set (the precomputed empty hash),
// which is what makes a 256-deep tree cheap.
func (s *ExecSet) subtree(keys []ids.ID, d int) [32]byte {
	if len(keys) == 0 {
		return emptyHash[smtDepth-d]
	}
	if d == smtDepth {
		// Exactly one key can reach full depth: 256 bits uniquely identify it.
		return hashLeaf(keys[0])
	}
	var left, right []ids.ID
	for _, k := range keys {
		if bit(k, d) {
			right = append(right, k)
		} else {
			left = append(left, k)
		}
	}
	return hashNode(s.subtree(left, d+1), s.subtree(right, d+1))
}

// ExecProof proves membership OR non-membership of one ExecID in an ExecRoot. The
// two cases share the structure entirely; Included is the only difference, and the
// verifier recomputes the same path either way.
type ExecProof struct {
	// Included is what the proof asserts. It is checked, never trusted: verification
	// recomputes the root from the corresponding leaf and compares.
	Included bool
	// Bitmap marks, MSB-first by depth, which siblings are non-empty and therefore
	// present in Siblings. Empty siblings are the precomputed constants.
	Bitmap []byte
	// Siblings holds only the non-empty siblings, deepest-first.
	Siblings [][32]byte
}

const bitmapLen = smtDepth / 8

// Prove produces a proof of membership or non-membership for id.
func (s *ExecSet) Prove(id ids.ID) ExecProof {
	keys := make([]ids.ID, 0, len(s.keys))
	for k := range s.keys {
		keys = append(keys, k)
	}
	p := ExecProof{Included: s.Contains(id), Bitmap: make([]byte, bitmapLen)}
	s.walk(keys, id, 0, &p)
	return p
}

// walk descends toward id, recording the sibling subtree root at each depth.
// Siblings come out shallowest-first here and are reversed by the caller-free
// convention below: we append shallowest-first and the verifier consumes in
// reverse, so no explicit reversal is needed on either side.
func (s *ExecSet) walk(keys []ids.ID, id ids.ID, d int, p *ExecProof) {
	if d == smtDepth {
		return
	}
	var left, right []ids.ID
	for _, k := range keys {
		if bit(k, d) {
			right = append(right, k)
		} else {
			left = append(left, k)
		}
	}
	var sibling []ids.ID
	if bit(id, d) {
		sibling = left
	} else {
		sibling = right
	}
	sh := s.subtree(sibling, d+1)
	if sh != emptyHash[smtDepth-d-1] {
		p.Bitmap[d/8] |= 1 << (7 - uint(d%8))
		p.Siblings = append(p.Siblings, sh)
	}
	if bit(id, d) {
		s.walk(right, id, d+1, p)
	} else {
		s.walk(left, id, d+1, p)
	}
}

var (
	ErrProofMalformed = errors.New("dexprotocol: ExecRoot proof is malformed")
	ErrProofMismatch  = errors.New("dexprotocol: ExecRoot proof does not reconstruct the committed root")
)

// VerifyExecProof recomputes the root from the proof and compares it to the
// committed one. It is the SINGLE verifier for both answers: an inclusion proof
// starts from the populated leaf, a non-inclusion proof from the empty leaf, and
// everything after that is identical. A proof that reconstructs the root with the
// empty leaf is a proof of absence, with exactly the same force as a proof of
// presence.
func VerifyExecProof(root [32]byte, id ids.ID, p ExecProof) error {
	if len(p.Bitmap) != bitmapLen {
		return fmt.Errorf("%w: bitmap is %d bytes, want %d", ErrProofMalformed, len(p.Bitmap), bitmapLen)
	}
	want := 0
	for d := 0; d < smtDepth; d++ {
		if p.Bitmap[d/8]&(1<<(7-uint(d%8))) != 0 {
			want++
		}
	}
	if want != len(p.Siblings) {
		return fmt.Errorf("%w: bitmap marks %d siblings, proof carries %d",
			ErrProofMalformed, want, len(p.Siblings))
	}

	cur := emptyHash[0]
	if p.Included {
		cur = hashLeaf(id)
	}
	// Ascend from the leaf: depth smtDepth-1 up to 0. Siblings were appended
	// shallowest-first, so consume them from the end.
	next := len(p.Siblings)
	for d := smtDepth - 1; d >= 0; d-- {
		var sib [32]byte
		if p.Bitmap[d/8]&(1<<(7-uint(d%8))) != 0 {
			next--
			if next < 0 {
				return fmt.Errorf("%w: ran out of siblings at depth %d", ErrProofMalformed, d)
			}
			sib = p.Siblings[next]
		} else {
			sib = emptyHash[smtDepth-d-1]
		}
		if bit(id, d) {
			cur = hashNode(sib, cur)
		} else {
			cur = hashNode(cur, sib)
		}
	}
	if cur != root {
		return ErrProofMismatch
	}
	return nil
}
