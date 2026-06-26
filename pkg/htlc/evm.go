// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package htlc

import (
	"encoding/binary"
	"math/big"

	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// EVM-leg ABI signatures. These canonical strings are the SINGLE source of truth
// for the selectors below; the documented hex (asserted in the tests) is derived
// from them, never hand-entered, so it cannot drift from swap.go / LXSwapHTLC.sol.
const (
	lockSig   = "lock(bytes32,address,address,address,uint256,uint64)"
	claimSig  = "claim(bytes32,bytes32)"
	refundSig = "refund(bytes32)"
)

// 4-byte selectors: keccak256(signature)[:4]. lock=0x4da2c728, claim=0x84cc9dfb,
// refund=0x7249fbb6.
var (
	selLock   = selector(lockSig)
	selClaim  = selector(claimSig)
	selRefund = selector(refundSig)
)

func selector(sig string) [4]byte {
	var s [4]byte
	copy(s[:], crypto.Keccak256([]byte(sig)))
	return s
}

// LockCalldata encodes lock(hashlock, recipient, refund, asset, amount, timeout)
// for the swap precompile / LXSwapHTLC.sol. hashlock is the SAME h = SHA256(s)
// the Bitcoin leg locks under, so one preimage settles both.
func LockCalldata(hashlock [32]byte, recipient, refund, asset common.Address, amount *big.Int, timeout uint64) []byte {
	out := make([]byte, 0, 4+6*32)
	out = append(out, selLock[:]...)
	out = append(out, hashlock[:]...)
	out = append(out, word(recipient.Bytes())...)
	out = append(out, word(refund.Bytes())...)
	out = append(out, word(asset.Bytes())...)
	out = append(out, word(amount.Bytes())...)
	out = append(out, word(u64(timeout))...)
	return out
}

// ClaimCalldata encodes claim(swapId, preimage). preimage is the 32-byte secret s.
func ClaimCalldata(swapID [32]byte, preimage Secret) []byte {
	out := make([]byte, 0, 4+2*32)
	out = append(out, selClaim[:]...)
	out = append(out, swapID[:]...)
	out = append(out, preimage[:]...)
	return out
}

// RefundCalldata encodes refund(swapId).
func RefundCalldata(swapID [32]byte) []byte {
	out := make([]byte, 0, 4+32)
	out = append(out, selRefund[:]...)
	out = append(out, swapID[:]...)
	return out
}

// word left-pads a big-endian byte string to a 32-byte ABI slot.
func word(b []byte) []byte {
	w := make([]byte, 32)
	copy(w[32-len(b):], b)
	return w
}

func u64(v uint64) []byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	return b[:]
}
