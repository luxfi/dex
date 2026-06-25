// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"os"
	"strings"
	"testing"
)

// TestDChainVM_IsPureBlockChainVM is the STRUCTURAL regression lock for the
// D-Chain (dexvm) consensus fix: the VM must STAY a pure block.ChainVM forever.
//
// Why this matters: the write-path wedge that motivated the consensus fix was a
// SHARED-engine bug (a vote for a not-yet-tracked block was dropped, so a
// follower could never reach quorum). The dexvm itself has — and must keep —
// ZERO networking/gossip primitives: it parses, verifies, accepts, and matches
// blocks under consensus, but it NEVER propagates them. Propagation, gossip,
// vote distribution, and ancestor fetch are the consensus engine's job
// (luxfi/consensus), not the VM's. If someone reintroduces a networking symbol
// here, they are re-braiding two concerns that the architecture keeps separate
// (and re-opening the door to VM-side gossip bugs). This test fails the moment
// any package source file references such a symbol.
//
// It scans the package's OWN non-test .go files (string scan; _test.go skipped —
// tests may legitimately name these tokens, as this very file does).
func TestDChainVM_IsPureBlockChainVM(t *testing.T) {
	// Networking / gossip / vote-propagation / catch-up symbols. None of these
	// may appear in the VM: it is a block.ChainVM, not a network handler.
	forbidden := []string{
		"AppGossip",
		"SendAppGossip",
		"PushQuery",
		"PullQuery",
		"SendChits",
		"common.Sender",
		"RequestAncestors",
	}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}

	scanned := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		scanned++
		src := string(data)
		for _, tok := range forbidden {
			if idx := strings.Index(src, tok); idx >= 0 {
				line := 1 + strings.Count(src[:idx], "\n")
				t.Errorf("PURITY VIOLATION: %s:%d references networking symbol %q — "+
					"the dexvm must stay a pure block.ChainVM; propagation/gossip/catch-up "+
					"belongs in luxfi/consensus, not the VM. Remove it.", name, line, tok)
			}
		}
	}

	if scanned == 0 {
		t.Fatal("scanned 0 package source files — the purity walk is broken (no .go files found)")
	}
	t.Logf("dexvm purity: scanned %d non-test source files, no networking symbols — pure block.ChainVM", scanned)
}
