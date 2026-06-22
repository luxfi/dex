package main

import (
	"bytes"
	"encoding/binary"
	"math/big"
	"testing"

	"github.com/luxfi/crypto"
	"github.com/luxfi/genesis/pkg/genesis"
	"github.com/luxfi/geth/common"
	ethtypes "github.com/luxfi/geth/core/types"
	"github.com/luxfi/ids"
	"github.com/luxfi/zap/dexsession"
)

// ---- maker-key derivation ----

// TestDeriveMakerKeyMatchesGenesisAuthority pins the keeper's BIP44 walk against the
// canonical genesis derivation (genesis.LoadBIP44WalletKeysFromMnemonic). A one-sided
// drift from m/44'/9000'/0'/0/i — different coin type, hardening, or hashing — changes
// the derived address and trips this test BEFORE the keeper ever signs with a wrong key.
func TestDeriveMakerKeyMatchesGenesisAuthority(t *testing.T) {
	const scan = 8
	authority, err := genesis.LoadBIP44WalletKeysFromMnemonic(genesis.LightMnemonic, scan)
	if err != nil {
		t.Fatalf("genesis.LoadBIP44WalletKeysFromMnemonic: %v", err)
	}
	for i, want := range authority {
		wantAddr := common.BytesToAddress(want.EVMAddr[:])
		got, err := deriveMakerKey(genesis.LightMnemonic, wantAddr, scan)
		if err != nil {
			t.Fatalf("deriveMakerKey index %d (%s): %v", i, wantAddr, err)
		}
		if got.Address != wantAddr {
			t.Fatalf("index %d: derived %s, genesis authority %s", i, got.Address, wantAddr)
		}
		if got.Index != i {
			t.Fatalf("index %d: keeper found it at index %d", i, got.Index)
		}
		// The ecdsa signing key must re-derive the SAME standard address (the C-Chain
		// caller / escrow-owner format keccak256(X||Y)[12:]).
		if a := evmAddress(got.ecdsa); a != wantAddr {
			t.Fatalf("index %d: ecdsa key address %s != genesis authority %s", i, a, wantAddr)
		}
	}
}

// TestDeriveMakerKeyKnownVector pins index 0 under the public LightMnemonic against a
// hardcoded address, so the derivation is anchored to a fixed value (not just to the
// authority, which could co-drift). This is the canonical BIP44 m/44'/9000'/0'/0/0 key.
func TestDeriveMakerKeyKnownVector(t *testing.T) {
	const wantHex = "0x5369615110ca435bdf798f31c20ba6163d7b0a54"
	want := common.HexToAddress(wantHex)
	got, err := deriveMakerKey(genesis.LightMnemonic, want, 4)
	if err != nil {
		t.Fatalf("deriveMakerKey: %v", err)
	}
	if got.Address != want {
		t.Fatalf("index 0 derived %s, want %s", got.Address, want)
	}
	if got.Index != 0 {
		t.Fatalf("known vector is index 0, keeper found %d", got.Index)
	}
}

// TestDeriveMakerKeyNotFound asserts a clean error when the wanted address is not in the
// scanned range (rather than a silent wrong key).
func TestDeriveMakerKeyNotFound(t *testing.T) {
	absent := common.HexToAddress("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
	if _, err := deriveMakerKey(genesis.LightMnemonic, absent, 4); err == nil {
		t.Fatal("expected error for absent address, got nil")
	}
	if _, err := deriveMakerKey("not a valid mnemonic", absent, 4); err == nil {
		t.Fatal("expected error for invalid mnemonic, got nil")
	}
}

// ---- Phase-A / Phase-B calldata ----

// TestPhaseAIntentCalldata asserts the keeper's Phase-A builder is byte-identical to a
// direct dexsession.EncodeSwapCalldata with a DI01 intent hookData, and that the calldata
// has the expected structural shape: 0xF3CD914C selector, 9 head words, then a DI01-tagged
// hookData carrying the deadline + nonce.
func TestPhaseAIntentCalldata(t *testing.T) {
	pk := dexsession.PoolKeyArgs{Hooks: dexsession.Account(poolManagerAddr9999), Fee: 3000, TickSpacing: 60}
	const (
		amountIn = uint64(5)
		deadline = uint64(1_700_000_000)
		nonce    = uint64(42)
	)
	got := phaseAIntentCalldata(pk, false, amountIn, deadline, nonce)
	want := dexsession.EncodeSwapCalldata(pk, false, amountIn, dexsession.EncodeIntentHookData(deadline, nonce))
	if !bytes.Equal(got, want) {
		t.Fatalf("phaseAIntentCalldata != dexsession.EncodeSwapCalldata\n got=%x\nwant=%x", got, want)
	}
	// Selector 0xF3CD914C (dexsession.SelectorSwap).
	if sel := binary.BigEndian.Uint32(got[0:4]); sel != dexsession.SelectorSwap {
		t.Fatalf("selector %#x, want %#x", sel, dexsession.SelectorSwap)
	}
	// Head: selector(4) + 9 words. The dynamic hookData tail begins after the head and
	// carries the DI01 tag.
	const headLen = 4 + 9*32
	if len(got) <= headLen {
		t.Fatalf("calldata len %d, want > head %d (a hookData tail)", len(got), headLen)
	}
	// The hookData length word is the last head word's pointee: the tail's first 32 bytes
	// are the length, then the DI01-tagged bytes.
	tail := got[headLen:]
	hookLen := binary.BigEndian.Uint64(tail[24:32])
	hookData := tail[32 : 32+hookLen]
	if string(hookData[:4]) != "DI01" {
		t.Fatalf("hookData tag %q, want DI01", hookData[:4])
	}
	// DI01 + deadline[32] + nonce[32] for a nonzero deadline+nonce.
	if hookLen != 4+64 {
		t.Fatalf("hookData len %d, want %d (tag + deadline + nonce)", hookLen, 4+64)
	}
	if d := binary.BigEndian.Uint64(hookData[4+24 : 4+32]); d != deadline {
		t.Fatalf("hookData deadline %d, want %d", d, deadline)
	}
	if n := binary.BigEndian.Uint64(hookData[4+32+24 : 4+32+32]); n != nonce {
		t.Fatalf("hookData nonce %d, want %d", n, nonce)
	}
}

// TestPhaseBSettleCalldata asserts the Phase-B builder carries a DS01 settlement hookData
// with the outputID|amount|intentID body (the intentID word is what binds the credit to
// the taker's intent record; a missing one is the wrong width and the precompile reverts).
func TestPhaseBSettleCalldata(t *testing.T) {
	pk := dexsession.PoolKeyArgs{Hooks: dexsession.Account(poolManagerAddr9999)}
	outputID := ids.ID{0x01, 0x02, 0x03}
	intentID := ids.ID{0xaa, 0xbb, 0xcc}
	const (
		amountIn  = uint64(7)
		amountOut = uint64(123)
	)
	got := phaseBSettleCalldata(pk, true, amountIn, outputID, intentID, amountOut)
	body := dexsession.EncodeSettlementHookData(dexsession.ID(outputID), amountOut, dexsession.ID(intentID))
	want := dexsession.EncodeSwapCalldata(pk, true, amountIn, body)
	if !bytes.Equal(got, want) {
		t.Fatalf("phaseBSettleCalldata != expected\n got=%x\nwant=%x", got, want)
	}
	const headLen = 4 + 9*32
	tail := got[headLen:]
	hookLen := binary.BigEndian.Uint64(tail[24:32])
	hookData := tail[32 : 32+hookLen]
	if string(hookData[:4]) != "DS01" {
		t.Fatalf("hookData tag %q, want DS01", hookData[:4])
	}
	// DS01 + outputID(32) + amount(32) + intentID(32) = 4 + 96.
	if hookLen != 4+96 {
		t.Fatalf("DS01 body len %d, want %d (tag + outputID + amount + intentID)", hookLen, 4+96)
	}
	if !bytes.Equal(hookData[4:36], outputID[:]) {
		t.Fatalf("DS01 outputID %x, want %x", hookData[4:36], outputID[:])
	}
	if a := binary.BigEndian.Uint64(hookData[36+24 : 36+32]); a != amountOut {
		t.Fatalf("DS01 amount %d, want %d", a, amountOut)
	}
	if !bytes.Equal(hookData[68:100], intentID[:]) {
		t.Fatalf("DS01 intentID %x, want %x", hookData[68:100], intentID[:])
	}
}

// ---- intent-id derivation ----

// TestDeriveIntentIDDeterministicAndInjective asserts the id is a pure function of its
// inputs (same inputs -> same id) and injective over the disambiguating axes (a changed
// nonce / account / amount changes the id). This mirrors precompile/dex DeriveIntentID;
// the keeper derives the SAME id the chain does from the same calldata (watch correlation).
func TestDeriveIntentIDDeterministicAndInjective(t *testing.T) {
	net := uint32(96369)
	cID := ids.ID{0xc0}
	dID := ids.ID{0xd0}
	acct := common.HexToAddress("0x9011E888251AB053B7bD1cdB598Db4f9DEd94714")
	asset := [32]byte{0x4c, 0x55, 0x58}
	market := [32]byte{0x4d}
	const amount, nonce = uint64(10), uint64(1)

	base := deriveIntentID(net, cID, dID, acct, asset, amount, market, nonce)
	if base != deriveIntentID(net, cID, dID, acct, asset, amount, market, nonce) {
		t.Fatal("deriveIntentID not deterministic")
	}
	if base == (ids.ID{}) {
		t.Fatal("deriveIntentID returned empty id")
	}
	// Each axis must change the id.
	cases := []struct {
		name string
		id   ids.ID
	}{
		{"nonce", deriveIntentID(net, cID, dID, acct, asset, amount, market, nonce+1)},
		{"amount", deriveIntentID(net, cID, dID, acct, asset, amount+1, market, nonce)},
		{"account", deriveIntentID(net, cID, dID, common.Address{0x01}, asset, amount, market, nonce)},
		{"network", deriveIntentID(net+1, cID, dID, acct, asset, amount, market, nonce)},
		{"market", deriveIntentID(net, cID, dID, acct, asset, amount, [32]byte{0x4e}, nonce)},
		{"dChain", deriveIntentID(net, cID, ids.ID{0xd1}, acct, asset, amount, market, nonce)},
	}
	for _, c := range cases {
		if c.id == base {
			t.Fatalf("changing %s did not change the intent id", c.name)
		}
	}
}

// ---- IntentSubmitted event decode ----

// makeIntentLog builds a synthetic IntentSubmitted log with the exact wire the precompile
// emits (3 topics, 10 ABI data words) so the decoder can be exercised without a chain.
func makeIntentLog(e intentEvent) ethtypes.Log {
	data := make([]byte, 0, 10*32)
	word := func(b []byte) { var w [32]byte; copy(w[32-len(b):], b); data = append(data, w[:]...) }
	addrWord := func(a common.Address) { word(a.Bytes()) }
	u64Word := func(v uint64) { var b [8]byte; binary.BigEndian.PutUint64(b[:], v); word(b[:]) }

	addrWord(e.Account)                   // 0 account
	data = append(data, e.AssetIn[:]...)  // 1 assetIn (full 32)
	u64Word(e.AmountIn)                   // 2 locked
	data = append(data, e.MarketID[:]...) // 3 marketID
	min := e.MinAmountOut
	if min == nil {
		min = big.NewInt(0)
	}
	word(min.Bytes())       // 4 minAmountOut
	addrWord(e.Recipient)   // 5 recipient
	u64Word(e.Deadline)     // 6 deadline
	u64Word(uint64(e.Kind)) // 7 kind
	u64Word(e.PriceLimit)   // 8 priceLimit
	var limitFlag uint64
	if e.LimitIsUpper {
		limitFlag = 1
	}
	u64Word(limitFlag) // 9 limitIsUpper

	return ethtypes.Log{
		Address: poolManagerAddr9999,
		Topics: []common.Hash{
			intentSubmittedSig,
			common.BytesToHash(e.IntentID[:]),
			common.BytesToHash(e.DChainID[:]),
		},
		Data: data,
	}
}

func TestDecodeIntentEventRoundTrip(t *testing.T) {
	want := intentEvent{
		IntentID:     ids.ID{0x11, 0x22, 0x33},
		DChainID:     ids.ID{0xd0, 0xd1},
		Account:      common.HexToAddress("0x9011E888251AB053B7bD1cdB598Db4f9DEd94714"),
		AssetIn:      [32]byte{0x4c, 0x55, 0x53, 0x44},
		AmountIn:     250,
		MarketID:     [32]byte{0x4d, 0x4b, 0x54},
		MinAmountOut: big.NewInt(99),
		Recipient:    common.HexToAddress("0x5369615110ca435bdf798f31c20ba6163d7b0a54"),
		Deadline:     1_700_000_000,
		Kind:         intentKindSwap,
		PriceLimit:   0x4034000000000000, // float64 bits of 20.0
		LimitIsUpper: true,
	}
	got, err := decodeIntentEvent(makeIntentLog(want))
	if err != nil {
		t.Fatalf("decodeIntentEvent: %v", err)
	}
	if got.IntentID != want.IntentID || got.DChainID != want.DChainID {
		t.Fatalf("ids: got (%s,%s) want (%s,%s)", got.IntentID, got.DChainID, want.IntentID, want.DChainID)
	}
	if got.Account != want.Account || got.Recipient != want.Recipient {
		t.Fatalf("addrs: got (%s,%s) want (%s,%s)", got.Account, got.Recipient, want.Account, want.Recipient)
	}
	if got.AssetIn != want.AssetIn || got.MarketID != want.MarketID {
		t.Fatalf("asset/market mismatch")
	}
	if got.AmountIn != want.AmountIn || got.Deadline != want.Deadline || got.Kind != want.Kind {
		t.Fatalf("amount/deadline/kind: got (%d,%d,%d) want (%d,%d,%d)", got.AmountIn, got.Deadline, got.Kind, want.AmountIn, want.Deadline, want.Kind)
	}
	if got.PriceLimit != want.PriceLimit || got.LimitIsUpper != want.LimitIsUpper {
		t.Fatalf("priceLimit/isUpper: got (%d,%v) want (%d,%v)", got.PriceLimit, got.LimitIsUpper, want.PriceLimit, want.LimitIsUpper)
	}
	if got.MinAmountOut.Cmp(want.MinAmountOut) != 0 {
		t.Fatalf("minAmountOut: got %s want %s", got.MinAmountOut, want.MinAmountOut)
	}
	// priceLimitFloat must decode the bits to the float the venue frame expects.
	if pf := priceLimitFloat(got.PriceLimit); pf != 20.0 {
		t.Fatalf("priceLimitFloat(%#x) = %v, want 20.0", got.PriceLimit, pf)
	}
}

// TestDecodeIntentEventRejectsMalformed asserts the boundary decoder fails closed on a
// foreign topic0, a wrong topic count, and a wrong data length — never silently acts on a
// log that is not an IntentSubmitted of the right shape.
func TestDecodeIntentEventRejectsMalformed(t *testing.T) {
	good := makeIntentLog(intentEvent{IntentID: ids.ID{0x01}, DChainID: ids.ID{0x02}, MinAmountOut: big.NewInt(0)})

	// foreign topic0
	bad := good
	bad.Topics = append([]common.Hash{}, good.Topics...)
	bad.Topics[0] = common.HexToHash("0xdeadbeef")
	if _, err := decodeIntentEvent(bad); err == nil {
		t.Fatal("expected error for foreign topic0")
	}
	// wrong topic count
	bad = good
	bad.Topics = good.Topics[:2]
	if _, err := decodeIntentEvent(bad); err == nil {
		t.Fatal("expected error for wrong topic count")
	}
	// wrong data length
	bad = good
	bad.Data = good.Data[:len(good.Data)-1]
	if _, err := decodeIntentEvent(bad); err == nil {
		t.Fatal("expected error for short data")
	}
}

// TestDecodeDEXFillEvent asserts the terminal DEXFill decode round-trips topics+data.
func TestDecodeDEXFillEvent(t *testing.T) {
	poolID := [32]byte{0xD0, 0x01}
	taker := common.HexToAddress("0x9011E888251AB053B7bD1cdB598Db4f9DEd94714")
	amountOut := big.NewInt(4321)
	blockNumber := uint64(123456)

	data := make([]byte, 0, 64)
	data = append(data, common.BigToHash(amountOut).Bytes()...)
	data = append(data, common.BigToHash(new(big.Int).SetUint64(blockNumber)).Bytes()...)
	log := ethtypes.Log{
		Address: poolManagerAddr9999,
		Topics: []common.Hash{
			dexFillSig,
			common.BytesToHash(poolID[:]),
			common.BytesToHash(taker.Bytes()),
		},
		Data: data,
	}
	got, err := decodeDEXFillEvent(log)
	if err != nil {
		t.Fatalf("decodeDEXFillEvent: %v", err)
	}
	if got.PoolID != poolID {
		t.Fatalf("poolID %x, want %x", got.PoolID, poolID)
	}
	if got.Taker != taker {
		t.Fatalf("taker %s, want %s", got.Taker, taker)
	}
	if got.AmountOut.Cmp(amountOut) != 0 {
		t.Fatalf("amountOut %s, want %s", got.AmountOut, amountOut)
	}
	if got.BlockNumber != blockNumber {
		t.Fatalf("blockNumber %d, want %d", got.BlockNumber, blockNumber)
	}
}

// ---- event signature pinning ----

// TestEventSignatures pins the two event topic0 hashes against keccak256 of the canonical
// ABI signature strings, so a drift from the precompile's emitted signature is caught here.
func TestEventSignatures(t *testing.T) {
	wantIntent := common.BytesToHash(crypto.Keccak256([]byte(
		"IntentSubmitted(bytes32,bytes32,address,bytes32,uint256,bytes32,uint256,address,uint64,uint8,uint64,uint8)")))
	if intentSubmittedSig != wantIntent {
		t.Fatalf("intentSubmittedSig %s != %s", intentSubmittedSig, wantIntent)
	}
	wantFill := common.BytesToHash(crypto.Keccak256([]byte("DEXFill(bytes32,address,uint256,uint256)")))
	if dexFillSig != wantFill {
		t.Fatalf("dexFillSig %s != %s", dexFillSig, wantFill)
	}
}
