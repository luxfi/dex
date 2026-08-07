// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/hex"
	"strings"
	"testing"
	"time"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
)

// genesis_test.go is the golden vector for the D-Chain's height-0 block.
//
// These constants are not derived at test time. They are pinned images, and any
// change to the block encoding, the execution-root composition, or the way the
// chain-creation document binds into genesis breaks them here — in CI, on a laptop,
// before a release exists. That is the whole point: without a pinned vector the
// genesis is whatever the binary computes today, and the first time it changes the
// only symptom is a validator that wipes its chainData for an unrelated reason and
// never rejoins.
//
// THIS ALREADY HAPPENED. dex v1.4.2 through v1.13.0 and dex v1.14.0 onward derive
// different genesis blocks, because commit 2a164da put the custody ledger into
// ComposeRoot (state.go) and the 136-byte keccak preimage became 168 bytes. Both
// ids are pinned below so the split is a fact in the test suite rather than a story.

const (
	// genesisImageHex is the canonical 84-byte height-0 block for a chain created
	// with no creation document: 32-byte empty parent, height 0, timestamp 0,
	// 32-byte execution root, zero transactions.
	genesisImageHex = "0000000000000000000000000000000000000000000000000000000000000000" + // parent
		"0000000000000000" + // height
		"0000000000000000" + // timestamp
		"92ae9d5c5dca023b00d94750403b9cd9cd450c84eb46a9908c80de7a1c3cf698" + // execRoot
		"00000000" // tx count

	// genesisID is ids.Checksum256(genesisImage) — the block id every validator on
	// a document-less D-Chain must report for height 0.
	genesisID = "mpxP6xF3ahpjdwP6YgjtagSWQCZSR6pziEqDoDv573k2V3WJA"

	// genesisIDPreLedger is the height-0 id every dex build from v1.4.2 through
	// v1.13.0 produced, before the custody ledger entered the root composition. A
	// node running one of those builds derives THIS and can never agree with a
	// fleet on the current one. Pinned so the divergence is regression-tested, not
	// rediscovered.
	genesisIDPreLedger = "MuetnVSbs1UnPUDTH5q4kgbKcfRx97DjmzZXVxcCdQ52H9sES"

	// preLedgerImageHex is that block, byte for byte — the image lux-mainnet and
	// lux-testnet are actually running, read from luxd on 2026-08-07 via
	//   /v1/bc/D/dex/clob_get_markets -> {"height":0,"lastAccepted":"MuetnVSb…",
	//                                     "root":"f8e013a4…"}
	// on all 15 pods. It differs from genesisImageHex in the execution root alone.
	preLedgerImageHex = "0000000000000000000000000000000000000000000000000000000000000000" +
		"0000000000000000" +
		"0000000000000000" +
		"f8e013a48fbc8e0ae0523a4174e333f1687da718a0f3e3bec31b6e8f7cff6815" +
		"00000000"
)

// The three chain-creation documents below are the bytes actually recorded in each
// fleet's P-Chain CreateChainTx — recovered from the tx and confirmed byte for byte
// against each pod's own /data/genesis.json dChainGenesis field, two independent
// sources. They are NOT the dchain.json in the luxfi/genesis module: that file
// appends a "chainId" out of sort order and drops the trailing newline, and it
// created none of these chains. The module file is the binary's idea of the
// configuration; these are the chain's.
//
// Two traps live in these bytes, and both are why the VM treats the document as an
// opaque byte string end to end — never parsed, never normalized, never re-encoded:
//
//   - The trailing newline is load-bearing and NOT uniform: mainnet and testnet
//     have one, devnet does not. Trimming or appending one breaks exactly two of
//     the three.
//   - The em dash is the literal six-character sequence \u2014, not UTF-8. A JSON
//     decode/re-encode round trip changes both the length and the hash.
//
// Length is not an identity. devnet's on-chain document is 375 bytes and so is the
// module file, and they are different byte strings — a size check passes and means
// nothing. These tests compare digests.

// mainnetDocument is the D-Chain's chain-creation record on lux-mainnet, byte for
// byte: 356 bytes, keccak256 77c9bcee54ebf7e7848ca15c79abf9166557301b6847cbea17cf24da49559ce4,
// with a trailing newline.
const mainnetDocument = `{
  "description": "Decentralized Exchange \u2014 native CLOB + AMM + perpetuals",
  "feeConfig": {
    "makerFeeBps": 2,
    "takerFeeBps": 5
  },
  "liquidityPools": [],
  "message": "Lux D-Chain Genesis",
  "name": "D-Chain",
  "networkId": 1,
  "perpetualMarkets": [],
  "timestamp": 1730446786,
  "tradingPairs": [],
  "version": 1,
  "vm": "DexVM"
}
`

// testnetDocument is the D-Chain's chain-creation record on lux-testnet, byte for
// byte: 356 bytes, keccak256 f79028985fa19654ee5c7f3dad9320321ce16702ba998bdb37081129dd11d292,
// with a trailing newline.
const testnetDocument = `{
  "description": "Decentralized Exchange \u2014 native CLOB + AMM + perpetuals",
  "feeConfig": {
    "makerFeeBps": 2,
    "takerFeeBps": 5
  },
  "liquidityPools": [],
  "message": "Lux D-Chain Genesis",
  "name": "D-Chain",
  "networkId": 2,
  "perpetualMarkets": [],
  "timestamp": 1730531602,
  "tradingPairs": [],
  "version": 1,
  "vm": "DexVM"
}
`

// devnetDocument is the D-Chain's chain-creation record on lux-devnet, byte for
// byte: 375 bytes, keccak256 e24a3c8d4a06446a416698c48fc6129f9ed4a02146fffd78294c08421f8dc2e2,
// with NO trailing newline.
const devnetDocument = `{
  "chainId": 96470,
  "description": "Decentralized Exchange \u2014 native CLOB + AMM + perpetuals",
  "feeConfig": {
    "makerFeeBps": 2,
    "takerFeeBps": 5
  },
  "liquidityPools": [],
  "message": "Lux D-Chain Genesis",
  "name": "D-Chain",
  "networkId": 3,
  "perpetualMarkets": [],
  "timestamp": 1730531602,
  "tradingPairs": [],
  "version": 1,
  "vm": "DexVM"
}`

// TestGenesisGolden pins the document-less genesis image byte for byte. It fails on
// any change to the block encoding or the execution-root composition.
func TestGenesisGolden(t *testing.T) {
	want, err := hex.DecodeString(genesisImageHex)
	if err != nil {
		t.Fatal(err)
	}
	if len(want) != blockHeaderSize+4 {
		t.Fatalf("golden image is %d bytes, want %d", len(want), blockHeaderSize+4)
	}

	gen := (&VM{}).canonicalGenesis(nil)
	if hex.EncodeToString(gen.bytes) != genesisImageHex {
		t.Fatalf("genesis image changed.\n got %x\nwant %s\n"+
			"A D-Chain built by this binary cannot join a fleet built by any other.",
			gen.bytes, genesisImageHex)
	}
	if gen.id.String() != genesisID {
		t.Fatalf("genesis id = %s, want %s", gen.id, genesisID)
	}
	if gen.id.String() == genesisIDPreLedger {
		t.Fatalf("genesis reverted to the pre-ledger composition (%s)", genesisIDPreLedger)
	}

	// Parsing the image back must reproduce the same block: the id is a hash of
	// these bytes, so a decoder that disagrees with the encoder is the same split
	// by another route.
	back, err := parseBlock(&VM{}, want)
	if err != nil {
		t.Fatalf("parse golden image: %v", err)
	}
	if back.id != gen.id {
		t.Fatalf("round-trip id = %s, want %s", back.id, gen.id)
	}
}

// fleet is a live D-Chain: the bytes its CreateChainTx recorded, their digest, and
// the height-0 block those bytes produce under this binary.
type fleet struct {
	name    string
	doc     string
	digest  string
	genesis string
}

// fleets pins every live D-Chain creation document and the genesis each one yields.
// A change to any document, or to how a document reaches height 0, moves a real
// chain — so it must be seen here first.
var fleets = []fleet{
	{"lux-mainnet", mainnetDocument, "77c9bcee54ebf7e7848ca15c79abf9166557301b6847cbea17cf24da49559ce4", "w2iF9fBMeo67qux1ZKwGSK7bYLApPBcLTi9TqaVDyJFkj1Uzo"},
	{"lux-testnet", testnetDocument, "f79028985fa19654ee5c7f3dad9320321ce16702ba998bdb37081129dd11d292", "9LFXAU4RPXz3TWdoUergnwfrH5QbqhS9necFDzGvHKx7fmohZ"},
	{"lux-devnet", devnetDocument, "e24a3c8d4a06446a416698c48fc6129f9ed4a02146fffd78294c08421f8dc2e2", "2vEdPNLxNWn85q2RX4EhZFAEJuLcAAJE9yAJR9K7jic1sVNTVw"},
}

// TestGenesisBindsCreationDocument pins each fleet's genesis and proves the creation
// document actually reaches it. A document-bound genesis equal to the document-less
// one would mean init.Genesis is being ignored again.
func TestGenesisBindsCreationDocument(t *testing.T) {
	vm := &VM{}
	unbound := vm.canonicalGenesis(nil)
	seen := map[ids.ID]string{}

	for _, f := range fleets {
		doc := []byte(f.doc)

		// The digest is pinned separately from the genesis id so a bad transcription
		// of the document is distinguishable from a change in the derivation. It is
		// also the only honest identity check: devnet's document and the luxfi/genesis
		// module's are both 375 bytes and are not the same bytes.
		if got := genesisOrigin(doc); hex.EncodeToString(got[:]) != f.digest {
			t.Errorf("%s: creation document digest = %x, want %s", f.name, got, f.digest)
			continue
		}

		g := vm.canonicalGenesis(doc)
		if g.id.String() != f.genesis {
			t.Errorf("%s: genesis = %s, want %s\n"+
				"Either the creation document or the derivation moved; both split the chain.",
				f.name, g.id, f.genesis)
		}
		if g.id == unbound.id {
			t.Errorf("%s: genesis equals the document-less one; the creation document is not reaching height 0", f.name)
		}
		if other, dup := seen[g.id]; dup {
			t.Errorf("%s and %s share a genesis; a block from one is replayable on the other", f.name, other)
		}
		seen[g.id] = f.name
	}

	// A one-byte edit must move the genesis. Nothing in the document may be cosmetic
	// as far as chain identity is concerned — including the trailing newline, which
	// mainnet has and devnet does not.
	if vm.canonicalGenesis([]byte(mainnetDocument)).id ==
		vm.canonicalGenesis([]byte(strings.TrimRight(mainnetDocument, "\n"))).id {
		t.Fatal("trimming the trailing newline left the genesis unchanged")
	}
}

// TestGenesisHonoursSuppliedDocument drives the real Initialize path: a chain born
// from a creation document must adopt the document-bound genesis, not the binary's
// default, and must record it.
func TestGenesisHonoursSuppliedDocument(t *testing.T) {
	db := memdb.New()
	doc := []byte(mainnetDocument)

	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Genesis:  doc,
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	if vm.lastAcceptedID.String() != fleets[0].genesis {
		t.Fatalf("height-0 head = %s, want the document-bound genesis %s",
			vm.lastAcceptedID, fleets[0].genesis)
	}
	stored, err := readGenesis(db)
	if err != nil {
		t.Fatalf("the chain was born without recording its genesis: %v", err)
	}
	if hex.EncodeToString(stored) != hex.EncodeToString(vm.canonicalGenesis(doc).bytes) {
		t.Fatalf("recorded genesis %x is not the document-bound one", stored)
	}
}

// TestGenesisSurvivesWipe is the reported failure, run forwards. Wiping chainData
// is the ordinary repair for a stuck chain; a wiped validator must come back on the
// SAME chain it left. It does, because it rebuilds genesis from the immutable
// creation record rather than from whatever its binary believes.
func TestGenesisSurvivesWipe(t *testing.T) {
	doc := []byte(mainnetDocument)
	init := func(db *memdb.Database) *VM {
		t.Helper()
		vm := &VM{}
		if err := vm.Initialize(context.Background(), block.Init{
			DB:       db,
			Log:      log.NewNoOpLogger(),
			ToEngine: make(chan block.Message, 16),
			Genesis:  doc,
			Config:   authConfig(t),
		}); err != nil {
			t.Fatalf("Initialize: %v", err)
		}
		return vm
	}

	before := init(memdb.New()).lastAcceptedID
	after := init(memdb.New()).lastAcceptedID // the wipe: a brand-new, empty chainData
	if before != after {
		t.Fatalf("a wiped node adopted a different genesis: %s -> %s", before, after)
	}
}

// TestGenesisRefusesDisagreement covers the mechanism that used to be silent: a
// node whose chain says one thing about height 0 and whose inputs say another must
// refuse to start, and must name all three views so an operator can tell which one
// moved.
func TestGenesisRefusesDisagreement(t *testing.T) {
	db := memdb.New()
	born := []byte(mainnetDocument)

	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Genesis:  born,
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	// Restart the same chainData against a different creation document.
	other := append([]byte{}, born...)
	other[len(other)-2] = '8'
	err := (&VM{}).Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Genesis:  other,
		Config:   authConfig(t),
	})
	if err == nil {
		t.Fatal("the VM started on a chain whose genesis it disagrees with")
	}

	msg := err.Error()
	for _, want := range []string{
		"expected", "derived", "stored", // all three views named
		fleets[0].genesis, // the stored one
		(&VM{}).canonicalGenesis(other).id.String(), // the one it would have used
		(&VM{}).canonicalGenesis(nil).id.String(),   // the binary's default
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal does not name %q:\n%s", want, msg)
		}
	}
}

// TestGenesisLiveFleets pins what is actually running, not what ought to be.
//
// Read from luxd on 2026-08-07, all 15 pods agreeing, every chain at height 0 with
// zero markets — the genesis block is the only block that has ever existed:
//
//	lux-mainnet  D=29Rnd1kbr9ZRyPF1AobiZvvF8LWMP9bmJhRC6979VBJkiXbRoD  MuetnVSb…
//	lux-testnet  D=2wWhbR6rBZmfNHcxrBX1Rm1y7uH4rWnK5xzed6GqstcdyPyPyM  MuetnVSb…
//	lux-devnet   D=23wLjqidbQ6wHA2hfGvkM8WMgbwpxgY8ryyZJsjoWZr9uG2pp1  mpxP6xF3…
//
// The 84-byte images are reconstructed from each chain's reported execution root
// and the fixed block layout, then checked to hash to the id luxd reports. They are
// derivations, not database reads: the D-Chain exposes no block-read RPC and the
// indexer is off.
//
// Two facts are asserted here so nobody has to rediscover them. First, the fleets
// are on two different genesis blocks — mainnet and testnet on a root composition
// this code no longer contains. Second, and this is the defect itself: not one of
// the three is the genesis its own creation document implies. Every live D-Chain
// was born from the binary, not from its chain-creation record.
func TestGenesisLiveFleets(t *testing.T) {
	live := map[string]string{
		"lux-mainnet": preLedgerImageHex,
		"lux-testnet": preLedgerImageHex,
		"lux-devnet":  genesisImageHex,
	}
	byName := map[string]fleet{}
	for _, f := range fleets {
		byName[f.name] = f
	}

	for name, imageHex := range live {
		raw, err := hex.DecodeString(imageHex)
		if err != nil {
			t.Fatal(err)
		}
		blk, err := parseBlock(&VM{}, raw)
		if err != nil {
			t.Fatalf("%s: parse the live genesis image: %v", name, err)
		}

		want := genesisIDPreLedger
		if name == "lux-devnet" {
			want = genesisID
		}
		if blk.id.String() != want {
			t.Fatalf("%s: live image hashes to %s, not the id luxd reports (%s)", name, blk.id, want)
		}

		// The whole defect in one assertion: what is running is not what the chain
		// was created as.
		bound := (&VM{}).canonicalGenesis([]byte(byName[name].doc))
		if blk.id == bound.id {
			t.Fatalf("%s: the live genesis already matches its creation document (%s) — "+
				"this fleet no longer needs re-founding and the report is stale", name, bound.id)
		}
	}

	// Mainnet and testnet cannot be reached from this code at all: their genesis was
	// built by a root composition it no longer contains, so a dexvm plugin built
	// here must refuse to serve them rather than fork at block 1.
	preLedger, err := hex.DecodeString(preLedgerImageHex)
	if err != nil {
		t.Fatal(err)
	}
	stranded, err := parseBlock(&VM{}, preLedger)
	if err != nil {
		t.Fatal(err)
	}
	for _, doc := range [][]byte{
		nil,
		[]byte(mainnetDocument),
		[]byte(testnetDocument),
		[]byte(devnetDocument),
		[]byte("anything at all"),
	} {
		if (&VM{}).canonicalGenesis(doc).id == stranded.id {
			t.Fatalf("this binary reproduced the pre-ledger genesis from a %d-byte document; the pin is wrong", len(doc))
		}
	}
}

// TestGenesisRecoversHeight0ChainThenRefuses is the live mainnet upgrade, run in a
// test. A D-Chain born under a pre-v1.14 build carries no genesis record; at height
// 0 the record is recoverable from its own head block, and the new binary then
// discovers it cannot serve that chain and says so with all three values. The
// alternative — starting anyway — forks the fleet at block 1 in silence.
func TestGenesisRecoversHeight0ChainThenRefuses(t *testing.T) {
	db := memdb.New()
	live, err := hex.DecodeString(preLedgerImageHex)
	if err != nil {
		t.Fatal(err)
	}
	head, err := parseBlock(&VM{}, live)
	if err != nil {
		t.Fatal(err)
	}

	// Exactly what a v1.13-born D-Chain at height 0 has on disk: head meta and head
	// block, no meta:genesis.
	batch := db.NewBatch()
	for _, w := range []func() error{
		func() error { return writeLastAccepted(batch, head.id) },
		func() error { return writeHeight(batch, 0) },
		func() error { return writeRoot(batch, head.execRoot) },
		func() error { return writeHeadBlock(batch, live) },
	} {
		if err := w(); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Write(); err != nil {
		t.Fatal(err)
	}

	err = (&VM{}).Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Genesis:  []byte(mainnetDocument),
		Config:   authConfig(t),
	})
	if err == nil {
		t.Fatal("a current build started on a pre-ledger chain; it would fork at block 1")
	}
	if !strings.Contains(err.Error(), genesisIDPreLedger) {
		t.Fatalf("refusal does not name the chain's own genesis:\n%v", err)
	}

	// The record was recovered from the chain's own data before the refusal, so the
	// next start reports the same three values instead of a different complaint.
	recovered, rerr := readGenesis(db)
	if rerr != nil {
		t.Fatalf("genesis was not recovered from the height-0 head: %v", rerr)
	}
	if hex.EncodeToString(recovered) != preLedgerImageHex {
		t.Fatalf("recovered genesis %x is not the chain's own head block", recovered)
	}
}

// TestGenesisRecoversAndContinues is the benign half of the same recovery: a
// document-less chain born under an older build carries no genesis record, its
// height-0 head IS its genesis, and the new binary agrees with it — so it records
// what the chain already says and keeps running. This is the standalone venue and
// lux-devnet's D-Chain.
func TestGenesisRecoversAndContinues(t *testing.T) {
	db := memdb.New()
	gen := (&VM{}).canonicalGenesis(nil)

	batch := db.NewBatch()
	for _, w := range []func() error{
		func() error { return writeLastAccepted(batch, gen.id) },
		func() error { return writeHeight(batch, 0) },
		func() error { return writeRoot(batch, gen.execRoot) },
		func() error { return writeHeadBlock(batch, gen.bytes) },
	} {
		if err := w(); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Write(); err != nil {
		t.Fatal(err)
	}

	vm := &VM{}
	if err := vm.Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("a chain whose genesis this binary agrees with was refused: %v", err)
	}
	if vm.lastAcceptedID != gen.id {
		t.Fatalf("head = %s, want %s", vm.lastAcceptedID, gen.id)
	}
	recovered, err := readGenesis(db)
	if err != nil {
		t.Fatalf("the genesis record was not written on recovery: %v", err)
	}
	if hex.EncodeToString(recovered) != genesisImageHex {
		t.Fatalf("recorded genesis %x is not the chain's height-0 block", recovered)
	}
}

// TestGenesisRefusesRecordlessChainAboveHeight0 covers the case with no honest
// recovery: past height 0 the head block is not the genesis, and there is nothing
// on disk that says which chain this is.
func TestGenesisRefusesRecordlessChainAboveHeight0(t *testing.T) {
	db := memdb.New()
	vm := &VM{}
	head := newBlock(vm, vm.canonicalGenesis(nil).id, 1, time.Unix(1, 0).UTC(), [Size]byte{}, nil)

	batch := db.NewBatch()
	for _, w := range []func() error{
		func() error { return writeLastAccepted(batch, head.id) },
		func() error { return writeHeight(batch, 1) },
		func() error { return writeRoot(batch, head.execRoot) },
		func() error { return writeHeadBlock(batch, head.bytes) },
	} {
		if err := w(); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Write(); err != nil {
		t.Fatal(err)
	}

	err := (&VM{}).Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	})
	if err == nil || !strings.Contains(err.Error(), "height 1") {
		t.Fatalf("got %v, want a refusal naming the height it cannot recover from", err)
	}
}

// TestGenesisRefusesDisownedHead covers recovery's own failure mode: a head block
// the head pointer does not name proves nothing about the chain, and blessing it as
// the genesis would invent the answer rather than read it.
func TestGenesisRefusesDisownedHead(t *testing.T) {
	db := memdb.New()
	gen := (&VM{}).canonicalGenesis(nil)
	other := (&VM{}).canonicalGenesis([]byte("a different chain"))

	batch := db.NewBatch()
	for _, w := range []func() error{
		func() error { return writeLastAccepted(batch, other.id) }, // pointer says one thing
		func() error { return writeHeight(batch, 0) },
		func() error { return writeRoot(batch, gen.execRoot) },
		func() error { return writeHeadBlock(batch, gen.bytes) }, // the block says another
	} {
		if err := w(); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Write(); err != nil {
		t.Fatal(err)
	}

	err := (&VM{}).Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	})
	if err == nil || !strings.Contains(err.Error(), "head pointer") {
		t.Fatalf("got %v, want a refusal naming the disowned head", err)
	}
	if _, rerr := readGenesis(db); rerr == nil {
		t.Fatal("a genesis was recorded from a block the chain does not point at")
	}
}

// TestGenesisRefusesUnreadableRecord covers a corrupt genesis record. Anything the
// VM cannot read as a height-0 block leaves it unable to say which chain it is on,
// which is a refusal, not a warning.
func TestGenesisRefusesUnreadableRecord(t *testing.T) {
	db := memdb.New()
	if err := writeGenesis(db, []byte("not a block")); err != nil {
		t.Fatal(err)
	}
	err := (&VM{}).Initialize(context.Background(), block.Init{
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	})
	if err == nil || !strings.Contains(err.Error(), "unreadable") {
		t.Fatalf("corrupt genesis record: got %v, want a refusal naming it unreadable", err)
	}
}
