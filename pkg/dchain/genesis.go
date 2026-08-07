// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"fmt"
	"time"

	"github.com/luxfi/crypto/hash"
)

// genesis.go is the single home of the D-Chain's height-0 block.
//
// A chain's genesis is creation data. It is NOT a property of whichever binary
// happens to be running, and the moment it becomes one the fleet can split without
// a single error: a node whose chainData is wiped re-derives genesis from its own
// code, agrees with nobody, and looks perfectly healthy in isolation. Wiping
// chainData is the ordinary repair for a stuck chain, so the ordinary repair is
// what strands the validator. That is what happened here — dex v1.13 and v1.14
// derive different height-0 blocks, because the custody ledger entered the root
// composition (ComposeRoot, state.go) and the block image changed with it.
//
// Three views of the genesis exist, and Initialize refuses to start unless they
// agree:
//
//	expected — from the chain-creation document, block.Init.Genesis. For the
//	           D-Chain that is the dchain.json recorded in the P-Chain
//	           CreateChainTx, delivered verbatim across the plugin boundary as
//	           InitializeRequest.GenesisBytes. Immutable: a wiped node reads the
//	           same document its chain was born from and rebuilds the same block.
//	derived  — what this binary computes with no document at all. The old and only
//	           view. TestGenesisGolden pins it byte for byte, so a change to the
//	           block image or the root composition fails in CI rather than on a
//	           fleet.
//	stored   — meta:genesis, written once in the batch that commits the height-0
//	           head and never rewritten. This node's record of which chain it is on.
//
// A node that cannot agree with its own chain about height 0 has nothing useful to
// do; starting anyway is exactly what makes the split silent.

// genesisTag domain-separates the digest of a chain-creation document from every
// other keccak preimage in the VM.
const genesisTag = "dex/genesis/v1"

// GenesisDigest is a chain's identity: the digest of the document it was created
// from.
//
//	genesis_digest = keccak256( "dex/genesis/v1" ‖ init.Genesis )
//
// The document is opaque — hashed exactly as delivered, never parsed, normalized or
// re-encoded. That keeps its schema out of consensus entirely (the D-Chain's fee
// table, market list and chain id bind to the chain without the VM knowing what any
// of them mean) and it is also the only safe treatment of the real documents: the
// live records differ in whether they end with a newline, and carry the em dash as
// the six ASCII characters — rather than as UTF-8. A JSON round trip changes
// the bytes, and with them the chain.
//
// Defined for every input including none, so a running chain always has a digest.
// "This chain has no creation document" is a policy question, answered once in
// genesis(); it is not a hole in the identity function.
func GenesisDigest(document []byte) [Size]byte {
	return hash.ComputeKeccak256Array([]byte(genesisTag), document)
}

// canonicalGenesis builds the height-0 block for a chain created from document: no
// parent, height 0, timestamp 0, no transactions, and the genesis root over the
// document's digest and the empty height-0 state. Pure — every validator and every
// restart computes the same bytes from the same document, and nothing else it
// reads, was compiled with, or has on disk can move the answer.
func (vm *VM) canonicalGenesis(document []byte) *Block {
	var emptyLedger [Size]byte // genesis has no custody ledger yet
	root := GenesisRoot(GenesisDigest(document), bookRoot(nil), tradeRoot(nil), txRoot(nil), emptyLedger)
	return newBlock(vm, genesisParent, 0, time.Unix(0, 0).UTC(), root, nil)
}

// recoverGenesis restores the genesis record of a chain born before the record
// existed. It is possible only at height 0, where the persisted head block IS the
// genesis: the bytes come from the chain's own data and the running binary has no
// say in them. Past height 0 the genesis is gone and the node cannot prove which
// chain it is on, which is a refusal — reconstructing it from the binary instead is
// the defect this file exists to remove. Must be called under vm.mu.
func (vm *VM) recoverGenesis() ([]byte, error) {
	raw, err := readHeadBlock(vm.db)
	if err != nil {
		return nil, fmt.Errorf("dchain: refusing to start: this database holds a chain with neither a "+
			"genesis record nor a head block, so its genesis is unknowable: %w", err)
	}
	head, err := parseBlock(vm, raw)
	if err != nil {
		return nil, fmt.Errorf("dchain: head block is unreadable: %w", err)
	}
	if head.height != 0 {
		return nil, fmt.Errorf(
			"dchain: refusing to start: this chain is at height %d and has no genesis record, so the node "+
				"cannot prove which chain it is on (it was born under a build that kept none). Re-sync from a "+
				"peer that has one. Do NOT wipe chainData: a wipe makes this node adopt its own binary's genesis "+
				"%s and partition permanently",
			head.height, describeGenesis(vm.canonicalGenesis(nil)))
	}
	// The head block only speaks for the chain if the chain points at it. Blessing
	// bytes the head pointer disowns would record a genesis out of thin air, which
	// is the thing being fixed.
	id, err := readLastAccepted(vm.db)
	if err != nil {
		return nil, fmt.Errorf("dchain: read head pointer: %w", err)
	}
	if head.id != id {
		return nil, fmt.Errorf(
			"dchain: refusing to start: the head pointer names %s but the stored head block is %s, "+
				"so no block on disk can be trusted as this chain's genesis", id, head.id)
	}
	if err := writeGenesis(vm.db, raw); err != nil {
		return nil, fmt.Errorf("dchain: record recovered genesis: %w", err)
	}
	return raw, nil
}

// genesis resolves the height-0 block this node must build on, and refuses to
// return one its own chain contradicts.
//
// document is block.Init.Genesis (the chain-creation record). stored is
// meta:genesis, empty only when this database holds no chain at all.
func (vm *VM) genesis(document, stored []byte) (*Block, error) {
	expected := vm.canonicalGenesis(document)
	if len(stored) == 0 {
		return expected, nil // chain is born here; Initialize records it
	}

	have, err := parseBlock(vm, stored)
	if err != nil {
		return nil, fmt.Errorf("dchain: stored genesis record is unreadable: %w (image %x)", err, stored)
	}
	if have.id != expected.id {
		return nil, genesisDisagreement(vm, document, expected, have)
	}
	return have, nil
}

// genesisDisagreement names all three views so an operator can see which one is the
// odd one out — the creation document, the binary, or the data on disk — without
// having to reproduce the derivation by hand.
func genesisDisagreement(vm *VM, document []byte, expected, stored *Block) error {
	origin := "no chain-creation document was supplied, so this is the binary's default"
	if len(document) > 0 {
		origin = fmt.Sprintf("%d-byte chain-creation document, digest %x", len(document), GenesisDigest(document))
	}
	return fmt.Errorf(
		"dchain: refusing to start: this node disagrees with its own chain about height 0.\n"+
			"  expected (chain-creation record): %s\n"+
			"           source: %s\n"+
			"  derived  (this VM binary, no document): %s\n"+
			"  stored   (this node's database): %s\n"+
			"This node would build on a genesis its own chain does not have, which no peer would\n"+
			"accept and no error would report. Run the dex build whose genesis matches the stored\n"+
			"one, or re-sync from a peer. Do NOT wipe chainData: a wipe removes the stored genesis\n"+
			"and lets this binary's genesis win silently",
		describeGenesis(expected), origin,
		describeGenesis(vm.canonicalGenesis(nil)),
		describeGenesis(stored))
}

// describeGenesis renders a height-0 block as its id, execution root and full
// canonical image. The image is 84 bytes, so printing it costs nothing and lets an
// operator diff two nodes directly.
func describeGenesis(b *Block) string {
	return fmt.Sprintf("%s (execRoot %x, image %x)", b.id, b.execRoot, b.bytes)
}
