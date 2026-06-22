// calldata.go holds the PURE wire logic the keeper drives the C<->D seam with:
//
//   - Phase-A / Phase-B 0x9999 swap calldata, built via github.com/luxfi/zap/dexsession
//     (the canonical reproduction of the precompile's calldata format).
//   - The intent-id derivation, mirroring precompile/dex DeriveIntentID. The keeper
//     CHOOSES the nonce when it signs Phase-A, so it derives the id locally; it also
//     reads the id back from the IntentSubmitted topic to correlate (authoritative).
//   - Decoders for the two 0x9999 events the keeper observes: IntentSubmitted (the
//     routing event it acts on) and DEXFill (the terminal settled-fill signal).
//
// Everything here is a deterministic function of values — no I/O, no chain handle — so
// it is unit-testable in isolation (calldata_test.go) and the encode side is asserted
// byte-identical to the precompile/dexsession authorities.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/big"

	"github.com/luxfi/crypto"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
	ethtypes "github.com/luxfi/geth/core/types"
	"github.com/luxfi/ids"
	"github.com/luxfi/zap/dexsession"
)

// poolManagerAddr9999 is the canonical V4-shaped settlement precompile (the money path).
// Phase-A intents and Phase-B settlements are both swap calls to this address; the
// IntentSubmitted / DEXFill events are emitted from it.
var poolManagerAddr9999 = common.HexToAddress("0x0000000000000000000000000000000000009999")

// nativeIntentDomain scopes the intent-id derivation. IDENTICAL to precompile/dex
// nativeIntentDomain — a one-sided change here would make the keeper's locally-derived
// id disagree with the chain's (the watch-correlation contract); calldata_test pins it.
const nativeIntentDomain = "lux.dex.native.intent.v1"

// IntentSubmitted(bytes32,bytes32,address,bytes32,uint256,bytes32,uint256,address,uint64,uint8,uint64,uint8)
// — the keeper-routing event. topics[1]=intentID, topics[2]=dChainID. Data is 10 ABI
// words: account,assetIn,locked,marketID,minAmountOut,recipient,deadline,kind,priceLimit,
// limitIsUpper (see precompile/dex native_events.go emitNativeRoutingEvent).
var intentSubmittedSig = common.BytesToHash(crypto.Keccak256([]byte(
	"IntentSubmitted(bytes32,bytes32,address,bytes32,uint256,bytes32,uint256,address,uint64,uint8,uint64,uint8)")))

// DEXFill(bytes32,address,uint256,uint256) — the terminal settled-fill signal.
// topics[1]=poolId, topics[2]=taker. Data is 2 ABI words: amountOut, blockNumber.
var dexFillSig = common.BytesToHash(crypto.Keccak256([]byte("DEXFill(bytes32,address,uint256,uint256)")))

// deriveIntentID re-derives the C->D intent id over the FULL identity, mirroring
// precompile/dex DeriveIntentID:
//
//	SHA-256( domain | networkID(4 BE) | cChainID | dChainID |
//	         account | assetIn | amountIn(8 BE) | marketID | nonce(8 BE) )
//
// The keeper picks the nonce, so it can derive the id BEFORE the Phase-A tx lands. It is
// only used to sanity-check the id it reads from the event topic (authoritative).
func deriveIntentID(networkID uint32, cChainID, dChainID ids.ID, account common.Address, assetIn [32]byte, amountIn uint64, marketID [32]byte, nonce uint64) ids.ID {
	h := sha256.New()
	h.Write([]byte(nativeIntentDomain))
	var u4 [4]byte
	binary.BigEndian.PutUint32(u4[:], networkID)
	h.Write(u4[:])
	h.Write(cChainID[:])
	h.Write(dChainID[:])
	h.Write(account[:])
	h.Write(assetIn[:])
	var u8 [8]byte
	binary.BigEndian.PutUint64(u8[:], amountIn)
	h.Write(u8[:])
	h.Write(marketID[:])
	binary.BigEndian.PutUint64(u8[:], nonce)
	h.Write(u8[:])
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return ids.ID(out)
}

// phaseAIntentCalldata builds the full 0x9999 Phase-A (DI01) swap calldata: a V4 swap
// with an explicit intent hookData carrying (deadline, nonce). The maker signs and sends
// this to 0x9999; SubmitSwapIntent locks the input and stages the C->D object.
func phaseAIntentCalldata(pk dexsession.PoolKeyArgs, zeroForOne bool, amountIn, deadline, nonce uint64) []byte {
	return dexsession.EncodeSwapCalldata(pk, zeroForOne, amountIn, dexsession.EncodeIntentHookData(deadline, nonce))
}

// phaseBSettleCalldata builds the full 0x9999 Phase-B (DS01) swap calldata: a V4 swap
// with a settlement hookData carrying (outputID, amount, intentID). The maker signs and
// sends this; ImportSettlement consumes the D->C proceeds object and emits DEXFill. The
// amount/zeroForOne mirror Phase-A so the swap ABI is well-formed; only the hookData
// (the phase selector) differs.
func phaseBSettleCalldata(pk dexsession.PoolKeyArgs, zeroForOne bool, amountIn uint64, outputID, intentID ids.ID, amount uint64) []byte {
	body := dexsession.EncodeSettlementHookData(dexsession.ID(outputID), amount, dexsession.ID(intentID))
	return dexsession.EncodeSwapCalldata(pk, zeroForOne, amountIn, body)
}

// intentEvent is the decoded IntentSubmitted routing payload the keeper acts on.
type intentEvent struct {
	IntentID     ids.ID // topics[1]
	DChainID     ids.ID // topics[2]
	Account      common.Address
	AssetIn      [32]byte
	AmountIn     uint64 // the locked principal (observed delta), low 64 bits of the word
	MarketID     [32]byte
	MinAmountOut *big.Int
	Recipient    common.Address
	Deadline     uint64
	Kind         uint8 // 0 swap, 1 position
	PriceLimit   uint64
	LimitIsUpper bool
}

// decodeIntentEvent decodes an IntentSubmitted log. It validates the topic[0] signature
// and the 10-word (320-byte) data layout at the boundary (untrusted chain input), then
// extracts the routing fields. Returns an error on any shape mismatch so a malformed or
// foreign log is never acted on.
func decodeIntentEvent(log ethtypes.Log) (intentEvent, error) {
	var e intentEvent
	if len(log.Topics) != 3 {
		return e, fmt.Errorf("IntentSubmitted: want 3 topics, got %d", len(log.Topics))
	}
	if log.Topics[0] != intentSubmittedSig {
		return e, fmt.Errorf("IntentSubmitted: topic0 %s != sig %s", log.Topics[0], intentSubmittedSig)
	}
	const wantWords = 10
	if len(log.Data) != wantWords*32 {
		return e, fmt.Errorf("IntentSubmitted: want %d data bytes, got %d", wantWords*32, len(log.Data))
	}
	e.IntentID = ids.ID(log.Topics[1])
	e.DChainID = ids.ID(log.Topics[2])
	word := func(i int) []byte { return log.Data[i*32 : i*32+32] }
	e.Account = common.BytesToAddress(word(0))
	copy(e.AssetIn[:], word(1))
	e.AmountIn = beUint64Low(word(2))
	copy(e.MarketID[:], word(3))
	e.MinAmountOut = new(big.Int).SetBytes(word(4))
	e.Recipient = common.BytesToAddress(word(5))
	e.Deadline = beUint64Low(word(6))
	e.Kind = word(7)[31]
	e.PriceLimit = beUint64Low(word(8))
	e.LimitIsUpper = word(9)[31] != 0
	return e, nil
}

// dexFillEvent is the decoded DEXFill terminal signal.
type dexFillEvent struct {
	PoolID      [32]byte // topics[1]
	Taker       common.Address
	AmountOut   *big.Int
	BlockNumber uint64
}

// decodeDEXFillEvent decodes a DEXFill log (the proof the C<->D round-trip settled).
func decodeDEXFillEvent(log ethtypes.Log) (dexFillEvent, error) {
	var e dexFillEvent
	if len(log.Topics) != 3 {
		return e, fmt.Errorf("DEXFill: want 3 topics, got %d", len(log.Topics))
	}
	if log.Topics[0] != dexFillSig {
		return e, fmt.Errorf("DEXFill: topic0 %s != sig %s", log.Topics[0], dexFillSig)
	}
	if len(log.Data) != 2*32 {
		return e, fmt.Errorf("DEXFill: want 64 data bytes, got %d", len(log.Data))
	}
	copy(e.PoolID[:], log.Topics[1].Bytes())
	e.Taker = common.BytesToAddress(log.Topics[2].Bytes())
	e.AmountOut = new(big.Int).SetBytes(log.Data[0:32])
	e.BlockNumber = beUint64Low(log.Data[32:64])
	return e, nil
}

// priceLimitFloat converts the V4-style price-limit bits the precompile carried in the
// event (a CLOB quote-per-base limit as float64 bits) into the float64 limit price the
// settling clob_submit frame needs. The precompile already did the SqrtPriceLimitX96 ->
// CLOB conversion (priceLimitToCLOB on the C side) and emitted the result, so the keeper
// carries it verbatim. Zero bits == 0.0 == "no limit" (a marketable order takes any price).
func priceLimitFloat(bits uint64) float64 {
	return math.Float64frombits(bits)
}

// beUint64Low reads the low 8 bytes of a 32-byte big-endian ABI word as a uint64. The
// precompile right-aligns uint64/uint8 values in their words (PutUint64(w[24:32], v)), so
// the value lives in the last 8 bytes.
func beUint64Low(word []byte) uint64 {
	return binary.BigEndian.Uint64(word[24:32])
}

// submitFrameFor builds the settling taker clob_submit frame the RelayOrderTx carries:
// a marketable order on poolID for `size` at the taker's worst-acceptable `limitPrice`,
// keyed to the maker user handle. side/limitPrice come from the intent direction +
// carried price limit. Byte-identical to zapwire.EncodeSubmit (the venue wire).
func submitFrameFor(poolID [32]byte, side uint8, isMarket bool, limitPrice, size float64, user string) []byte {
	return zapwire.EncodeSubmit(poolID, side, isMarket, limitPrice, size, user)
}
