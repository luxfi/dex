// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dexcore

import (
	"encoding/binary"

	"github.com/luxfi/crypto/hash"
	"github.com/luxfi/crypto/merkle"
	"github.com/luxfi/dex/pkg/lx"
)

// root.go is the DEX-state execution root — the same RFC-6962 tagged keccak256
// binary-Merkle construction the standalone D-Chain uses (vms/xvm execution_root
// lineage), now derived over the in-process state a swap leaves behind. In the
// cEVM home this is subsumed by the EVM block state root (every D write is a 0x9999
// storage write, so it is already committed by the EVM trie). dexcore still exposes
// a DEX-scoped root so a test (and a cross-validator diff) can compare the exact
// resulting DEX state independent of the surrounding EVM trie, and so the dchain
// home keeps an explicit DEX execution root.
//
// The root binds: the canonical resting rows of the touched market (the book), and
// the swap's realized fills (the trades). A validator that fabricates a fill the
// matcher would not produce credits a different amount / touches a different maker,
// so its rows or fills differ and the root differs — the basis for rejecting a
// Byzantine proposer (the EVM-state-root analog of the dchain claimed-vs-derived
// root check).

// Size is the byte length of a root (a keccak256 digest).
const Size = hash.KeccakSize

// SwapExecutionRoot returns the DEX execution root over the state a swap left in
// db for market poolID, plus the swap's realized fills. It folds:
//
//   - the canonical resting rows of poolID (rebuilt from order:* rows), and
//   - the swap's CLOB fills (the trade rows),
//
// into two RFC-6962 tagged Merkle sub-roots and composes them with a fixed-shape
// keccak256. Deterministic over deterministic state, so two validators replaying
// the same swap derive the identical root.
func SwapExecutionRoot(db Store, poolID [32]byte, res *SwapResult) [Size]byte {
	book := bookRootForMarket(db, poolID)
	trade := tradeRootForResult(res)
	return composeRoot(book, trade, res)
}

// bookRootForMarket reads the current canonical resting rows of a market and folds
// them into the RFC-6962 book root. An empty / unreadable book yields the empty
// root (a market with no resting liquidity is a valid state).
func bookRootForMarket(db Store, poolID [32]byte) [Size]byte {
	symbol, exists, err := ReadMarketSymbol(db, poolID)
	if err != nil || !exists {
		return merkle.EmptyRoot()
	}
	book, err := RebuildBook(db, poolID, symbol)
	if err != nil {
		return merkle.EmptyRoot()
	}
	rows := lx.BookToRows(book) // canonical (sorted) order from the matcher
	leaves := make([][Size]byte, len(rows))
	for i := range rows {
		leaves[i] = orderLeafDigest(rows[i], uint32(i))
	}
	return merkle.Root(leaves)
}

// tradeRootForResult folds a swap's realized CLOB fills into the RFC-6962 trade
// root. Trades are folded in execution order (the router's source/fill order), each
// bound to its index. AMM fills carry no maker rows; their (base,quote) deltas are
// committed via the reserve write + the journal and are included here as a synthetic
// trade leaf so a fabricated AMM amount also perturbs the root.
func tradeRootForResult(res *SwapResult) [Size]byte {
	var leaves [][Size]byte
	idx := uint32(0)
	for _, f := range res.Fills {
		for _, tr := range f.Trades {
			leaves = append(leaves, tradeLeafDigest(lx.TradeToRow(tr, tr.TakerSide), idx))
			idx++
		}
		if f.Source == SourceAMM && (f.AMMBaseDelta != 0 || f.AMMQuoteDelta != 0) {
			leaves = append(leaves, ammLeafDigest(f.AMMBaseDelta, f.AMMQuoteDelta, idx))
			idx++
		}
	}
	return merkle.Root(leaves)
}

// orderLeafDigest = keccak256(canonical row image ‖ index_le).
func orderLeafDigest(row lx.DEXOrder, i uint32) [Size]byte {
	img := lx.EncodeRow(row)
	var idx [4]byte
	binary.LittleEndian.PutUint32(idx[:], i)
	return hash.ComputeKeccak256Array(img, idx[:])
}

// tradeLeafDigest = keccak256(canonical trade image ‖ index_le).
func tradeLeafDigest(t lx.DEXTrade, i uint32) [Size]byte {
	img := lx.EncodeTrade(t)
	var idx [4]byte
	binary.LittleEndian.PutUint32(idx[:], i)
	return hash.ComputeKeccak256Array(img, idx[:])
}

// ammLeafDigest = keccak256("amm" ‖ base_le ‖ quote_le ‖ index_le) — binds an AMM
// leg's (base,quote) move into the trade root so a tampered AMM amount changes the
// root just as a tampered CLOB fill does.
func ammLeafDigest(base, quote uint64, i uint32) [Size]byte {
	var b [8]byte
	var q [8]byte
	var idx [4]byte
	binary.LittleEndian.PutUint64(b[:], base)
	binary.LittleEndian.PutUint64(q[:], quote)
	binary.LittleEndian.PutUint32(idx[:], i)
	return hash.ComputeKeccak256Array([]byte("amm"), b[:], q[:], idx[:])
}

// composeRoot binds the book + trade sub-roots and the swap's net amounts into the
// final root via a fixed-shape un-tagged keccak256 (the xvmroot.Compose lineage).
// Binding the realized AmountIn/AmountOut makes the root commit to the swap's value
// outcome directly, so any divergence in the credited amount perturbs the root.
func composeRoot(book, trade [Size]byte, res *SwapResult) [Size]byte {
	var in [8]byte
	var out [8]byte
	binary.LittleEndian.PutUint64(in[:], res.AmountIn)
	binary.LittleEndian.PutUint64(out[:], res.AmountOut)
	return hash.ComputeKeccak256Array(book[:], trade[:], in[:], out[:])
}
