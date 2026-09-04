// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/sha3"

	"github.com/luxfi/consensus/engine/chain/block"
	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/dex"
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
		"c89155991ee17db4900c4dc4e5d1619322a118f2da26093f0abdf70d9c41ce32" + // execRoot
		"00000000" // tx count

	// genesisID is ids.Checksum256(genesisImage) — the block id every validator on
	// a document-less D-Chain must report for height 0.
	genesisID = "MnugNZNLVFx5R4ANwjwRBRZAjS5AsDcmVP4o6Kjozq4yud9mk"

	// The empty height-0 sub-roots. bookRoot, tradeRoot and txRoot over nothing are
	// all the RFC-6962 empty-tree digest; the custody ledger at height 0 is the zero
	// value, not a Merkle root, because no ledger has been committed yet. Pinned
	// because GenesisRoot hashes them: a change to any empty-root convention moves
	// every chain's genesis and must be seen here.
	emptyLeafRootHex   = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
	emptyLedgerRootHex = "0000000000000000000000000000000000000000000000000000000000000000"

	// genesisIDPreLedger is the height-0 id every dex build from v1.4.2 through
	// v1.13.0 produced, before the custody ledger entered the root composition. A
	// node running one of those builds derives THIS and can never agree with a
	// fleet on the current one. Pinned so the divergence is regression-tested, not
	// rediscovered.
	genesisIDPreLedger = "MuetnVSbs1UnPUDTH5q4kgbKcfRx97DjmzZXVxcCdQ52H9sES"

	// preLedgerImageHex is that block, byte for byte — the image lux-mainnet and
	// lux-testnet are actually running, read from luxd on 2026-08-07 via
	//   /v1/chain/d/dex/clob_get_markets -> {"height":0,"lastAccepted":"MuetnVSb…",
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

// fleet is a live D-Chain: the bytes its CreateChainTx recorded, a sha256 that
// pins those bytes to the record they were read from, and everything this binary
// derives from them.
type fleet struct {
	name    string
	doc     string
	sha256  string // of doc — pins the transcription above against the on-chain record
	digest  string // GenesisDigest(doc)
	root    string // GenesisRoot over that digest and the empty height-0 state
	genesis string // the height-0 block id
}

// fleets pins every live D-Chain creation document and everything it yields. A
// change to any document, or to how a document reaches height 0, moves a real chain
// — so it must be seen here first.
//
// sha256 is the second, independent pin. The documents above are Go string literals
// and a transcription error in one is invisible by inspection: the em dash is six
// ASCII characters, the trailing newline is present on two of the three fleets and
// absent on the third, and devnet's on-chain document is the same LENGTH as the
// luxfi/genesis module file it is not. These sha256 values were taken from the
// bytes recovered from each fleet's P-Chain CreateChainTx, so they hold the
// literals to the record rather than to each other.
var fleets = []fleet{
	{
		"lux-mainnet", mainnetDocument,
		"1125e26dce313be61133850378bd7126f3c294ff1931cd566191b296fa6d0db7",
		"d72bf3252e0dce1ecdba57526166080fbd01117620c61f29a0742cbf26cd5d2c",
		"b775b7f55a2f47e01bf19bc703b83f75d701d5c5b62c6d717a78c8cbdd3dfd4d",
		"2VmbYVHZVcPEomXcGsCgW85GAxU5rumpJxKVqspc1V5DV9M377",
	},
	{
		"lux-testnet", testnetDocument,
		"fb132e9312340ccc23530bf72ba3c9c65bf07c72c3a77d08780f55d86e1ed29e",
		"56a4c54daed5e2b7d40bcdc50d4c0f9a1cf1e0cc636ae9fef42ac55db2bc6275",
		"dcd3b6aa31049db62806232012fc32e9702110773ef5126089afcea27f6c6108",
		"25MZFfejPREcBvFtxboW5zeHdE1PNHRXQhG7wY1CC9UpkNLqPC",
	},
	{
		"lux-devnet", devnetDocument,
		"8915307876b00a0476ded52dfd527bc3710bb114a9675a565eaeafa3ad492360",
		"c6c7d2b55fa4f02784d991419cb4c19e1fd6c00219f749dae8697bc34b903964",
		"183be5f4e9833f4717e288d959d3f335aa456933cd09ba5b2bad8aec6d5b3877",
		"2jLRn4gcjemSZ4Tc6asg3dEi97cLdhn99rxWSXfLJw1wuDWnR7",
	},
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
		if got := GenesisDigest(doc); hex.EncodeToString(got[:]) != f.digest {
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

// specGenesisRoot recomputes a height-0 execution root straight from the written
// construction:
//
//	GenesisDigest = keccak256( "dex/genesis/v1"      ‖ document )
//	GenesisRoot   = keccak256( "dex/genesis-root/v1" ‖ digest ‖ book ‖ trade ‖ tx ‖ ledger )
//
// It shares no code with the VM — a different keccak implementation, the empty
// sub-roots read from pinned hex rather than computed, and the concatenation
// spelled out. So it compares the VM's derivation against the SPECIFICATION rather
// than against yesterday's output of the same functions. A hash-equals-hash vector
// moves the moment someone updates it to match; this one does not, because the
// spec side has to be edited to say something different before it will agree.
func specGenesisRoot(t *testing.T, document []byte) [Size]byte {
	t.Helper()
	leaf, err := hex.DecodeString(emptyLeafRootHex)
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := hex.DecodeString(emptyLedgerRootHex)
	if err != nil {
		t.Fatal(err)
	}

	k := sha3.NewLegacyKeccak256()
	k.Write([]byte("dex/genesis/v1"))
	k.Write(document)
	digest := k.Sum(nil)

	k = sha3.NewLegacyKeccak256()
	k.Write([]byte("dex/genesis-root/v1"))
	k.Write(digest)
	k.Write(leaf)   // book
	k.Write(leaf)   // trade
	k.Write(leaf)   // tx
	k.Write(ledger) // custody
	var root [Size]byte
	copy(root[:], k.Sum(nil))
	return root
}

// TestGenesisMatchesSpecification is the vector that would have caught v1.14.0.
//
// The defect then was that ComposeRoot gained a ledger operand and every height-0
// block silently moved with it, because genesis was built by calling the block
// composition with the creation document jammed into the parent slot. Genesis no
// longer touches that composition: it has its own tag, its own operands and no
// height term, so nothing done to the block root can reach it. This test holds that
// separation by recomputing the genesis root from the construction itself.
func TestGenesisMatchesSpecification(t *testing.T) {
	vm := &VM{}
	for _, f := range append([]fleet{{name: "document-less"}}, fleets...) {
		doc := []byte(f.doc)
		if f.doc == "" {
			doc = nil
		}
		want := specGenesisRoot(t, doc)
		if got := vm.canonicalGenesis(doc).execRoot; got != want {
			t.Errorf("%s: genesis root = %x, specification says %x", f.name, got, want)
		}
	}

	// The two constructions must not collide on identical operands, or the tags are
	// decorative and a block root could be presented as a genesis root.
	var d, b, tr, x, l [Size]byte
	d[0] = 1
	if GenesisRoot(d, b, tr, x, l) == ComposeRoot(d, b, tr, x, l, 0) {
		t.Fatal("genesis and block roots collide on the same operands; the domain tags do nothing")
	}
}

// TestGenesisDependsOnNothingButTheDocument is the other half of the gate: the same
// creation bytes must derive the same genesis regardless of what the binary was
// compiled with or what the process is doing.
//
// Every input other than the document is varied here — the VM value, its config,
// its database, its loaded books, its clock — and the derived bytes must not move.
// The pinned images then mean something: they are what ANY build deriving genesis
// this way produces, not what this one happens to produce today.
func TestGenesisDependsOnNothingButTheDocument(t *testing.T) {
	for _, f := range fleets {
		doc := []byte(f.doc)

		// A bare VM, and one carrying as much unrelated runtime state as can be
		// attached without executing anything.
		loaded := &VM{
			db:                 memdb.New(),
			books:              map[[32]byte]*dex.OrderBook{{1}: dex.NewOrderBook("ZOO-USD")},
			depositAuthority:   userKey{9, 9, 9},
			lastAcceptedHeight: 4321,
			lastRoot:           [Size]byte{7},
		}
		bare := (&VM{}).canonicalGenesis(doc)
		rich := loaded.canonicalGenesis(doc)

		if hex.EncodeToString(bare.bytes) != hex.EncodeToString(rich.bytes) {
			t.Fatalf("%s: genesis moved with runtime state\n bare %x\n rich %x", f.name, bare.bytes, rich.bytes)
		}
		if bare.id.String() != f.genesis {
			t.Errorf("%s: genesis = %s, want %s", f.name, bare.id, f.genesis)
		}
		if hex.EncodeToString(bare.execRoot[:]) != f.root {
			t.Errorf("%s: genesis root = %x, want %s", f.name, bare.execRoot, f.root)
		}
	}
}

// TestFleetDocumentsAreTheOnChainBytes holds the Go literals above to the records
// they were transcribed from. Without it a corrected-looking edit — normalising the
// em dash, adding the missing newline to devnet — passes every other test in this
// file, because they all derive from the literal.
func TestFleetDocumentsAreTheOnChainBytes(t *testing.T) {
	for _, f := range fleets {
		sum := sha256.Sum256([]byte(f.doc))
		if hex.EncodeToString(sum[:]) != f.sha256 {
			t.Errorf("%s: document is %d bytes sha256 %x, want %s — this literal is no longer the chain's record",
				f.name, len(f.doc), sum, f.sha256)
		}
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
		Runtime:  testRuntime(),
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
			Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
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
		Runtime:  testRuntime(),
		DB:       db,
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	})
	if err == nil || !strings.Contains(err.Error(), "unreadable") {
		t.Fatalf("corrupt genesis record: got %v, want a refusal naming it unreadable", err)
	}
}

// TestGenesisRefusesToFoundWithoutDocument closes the last way the binary could
// still decide a chain's identity.
//
// An empty database and an empty document is the one combination the three-way
// check cannot judge: nothing was delivered, nothing is stored, so there is no
// disagreement to detect and the node founds a chain from its own code — exactly
// the defect this file exists to remove, arrived at by the absence of its inputs
// rather than by their conflict. A D-Chain's document is recorded in its P-Chain
// CreateChainTx, so an empty one is a delivery failure across the plugin
// boundary, not a chain that has none.
func TestGenesisRefusesToFoundWithoutDocument(t *testing.T) {
	err := (&VM{}).Initialize(context.Background(), block.Init{
		Runtime:  testRuntime(),
		DB:       memdb.New(),
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Config:   authConfig(t),
	})
	if err == nil {
		t.Fatal("a chain was founded from no creation document; its genesis came from the binary")
	}
	// The refusal has to name the genesis it declined to use, or an operator
	// cannot tell this apart from the chain simply failing to start.
	for _, want := range []string{"refusing to found", genesisID} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("refusal does not mention %q:\n%s", want, err)
		}
	}

	// The same empty database WITH a document founds normally: the refusal is
	// about the missing document, not about founding.
	if err := (&VM{}).Initialize(context.Background(), block.Init{
		Runtime:  testRuntime(),
		DB:       memdb.New(),
		Log:      log.NewNoOpLogger(),
		ToEngine: make(chan block.Message, 16),
		Genesis:  []byte(mainnetDocument),
		Config:   authConfig(t),
	}); err != nil {
		t.Fatalf("a chain with a creation document was refused: %v", err)
	}
}
