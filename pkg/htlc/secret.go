// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package htlc constructs the three legs of a Lux cross-chain atomic swap from a
// single shared primitive: a 32-byte Secret and its SHA-256 hashlock. The legs
// are bound ONLY by that hashlock h = SHA256(s); one preimage s unlocks all of
// them.
//
//   - secret.go  the chain-agnostic Secret + hashlock (THIS file): the one source
//     of the hashlock every leg shares.
//   - bitcoin.go the BTC P2WSH OP_SHA256/OP_CLTV HTLC script, address, and the
//     real BIP143 segwit-v0 claim/refund witnesses.
//   - evm.go     lock/claim/refund calldata for the on-Lux swap precompile
//     (LP-90A0) and the LXSwapHTLC.sol counterparty contract.
//
// The hashlock is SHA-256 — matching Bitcoin's OP_SHA256 and the EVM 0x02
// precompile, NEVER keccak — and the preimage is fixed at exactly 32 bytes so the
// EVM bytes32 argument and the Bitcoin 32-byte witness push are the same value.
package htlc

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"io"
)

// Secret is the 32-byte preimage s. Its SHA-256 hash is the hashlock h that every
// leg of the swap is locked under; revealing s on any leg reveals it for all.
type Secret [32]byte

// NewSecret draws a uniform 32-byte secret from the OS CSPRNG.
func NewSecret() (Secret, error) {
	var s Secret
	if _, err := io.ReadFull(rand.Reader, s[:]); err != nil {
		return Secret{}, err
	}
	return s, nil
}

// Hashlock is h = SHA256(s): the value the BTC script's OP_SHA256...OP_EQUALVERIFY
// and the EVM legs' SHA-256 check all compare against.
func (s Secret) Hashlock() [32]byte {
	return sha256.Sum256(s[:])
}

// VerifyPreimage reports whether SHA256(secret) == hashlock, in constant time.
// This is the single check each leg performs on a revealed preimage.
func VerifyPreimage(secret Secret, hashlock [32]byte) bool {
	got := sha256.Sum256(secret[:])
	return subtle.ConstantTimeCompare(got[:], hashlock[:]) == 1
}
