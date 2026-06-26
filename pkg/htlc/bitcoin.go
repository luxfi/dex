// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package htlc

import (
	"crypto/sha256"
	"errors"

	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/btcec/v2/ecdsa"
	"github.com/btcsuite/btcd/btcutil"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/txscript"
	"github.com/btcsuite/btcd/wire"
)

// claimBranch is the witness selector that drives OP_IF down the claim path; any
// non-empty value is true. An empty witness item selects the OP_ELSE refund path.
var claimBranch = []byte{0x01}

var (
	ErrNilPubKey          = errors.New("htlc: recipient/refund public key must be non-nil")
	ErrShortWitnessScript = errors.New("htlc: witness script unexpectedly empty")
)

// BuildHTLCScript emits the fixed Bitcoin P2WSH HTLC witness script:
//
//	OP_IF
//	    OP_SHA256 <hashlock> OP_EQUALVERIFY
//	    <recipientPub> OP_CHECKSIG          ; claim:  witness <sig> <s> 1
//	OP_ELSE
//	    <locktime> OP_CHECKLOCKTIMEVERIFY OP_DROP
//	    <refundPub> OP_CHECKSIG             ; refund: witness <sig> 0
//	OP_ENDIF
//
// hashlock is h = SHA256(s); locktime is the absolute CLTV refund timeout (BIP65,
// unix seconds). Public keys are serialized compressed.
func BuildHTLCScript(hashlock [32]byte, recipientPub, refundPub *btcec.PublicKey, locktime uint32) ([]byte, error) {
	if recipientPub == nil || refundPub == nil {
		return nil, ErrNilPubKey
	}
	return txscript.NewScriptBuilder().
		AddOp(txscript.OP_IF).
		AddOp(txscript.OP_SHA256).AddData(hashlock[:]).AddOp(txscript.OP_EQUALVERIFY).
		AddData(recipientPub.SerializeCompressed()).AddOp(txscript.OP_CHECKSIG).
		AddOp(txscript.OP_ELSE).
		AddInt64(int64(locktime)).AddOp(txscript.OP_CHECKLOCKTIMEVERIFY).AddOp(txscript.OP_DROP).
		AddData(refundPub.SerializeCompressed()).AddOp(txscript.OP_CHECKSIG).
		AddOp(txscript.OP_ENDIF).
		Script()
}

// P2WSHAddress is the bech32 pay-to-witness-script-hash address committing to
// SHA256(script) — the address an HTLC output is funded to.
func P2WSHAddress(script []byte, net *chaincfg.Params) (btcutil.Address, error) {
	h := sha256.Sum256(script)
	return btcutil.NewAddressWitnessScriptHash(h[:], net)
}

// ClaimWitness builds the claim-path witness for input idx of tx, signing the real
// BIP143 segwit-v0 sighash over the HTLC script (the P2WSH scriptCode) with the
// recipient key. amount is the value of the HTLC output being spent. The returned
// stack is <sig> <preimage> 0x01 <script>; assign it to tx.TxIn[idx].Witness.
func ClaimWitness(script []byte, tx *wire.MsgTx, idx int, amount int64, preimage Secret, recipientKey *btcec.PrivateKey) (wire.TxWitness, error) {
	sig, err := sign(script, tx, idx, amount, recipientKey)
	if err != nil {
		return nil, err
	}
	return wire.TxWitness{sig, preimage[:], claimBranch, script}, nil
}

// RefundWitness builds the refund-path witness for input idx of tx with the refund
// key. The CALLER must give the tx an nLockTime >= the script locktime and the
// spending input an nSequence < 0xffffffff, or OP_CHECKLOCKTIMEVERIFY rejects the
// spend. The returned stack is <sig> <empty> <script>.
func RefundWitness(script []byte, tx *wire.MsgTx, idx int, amount int64, refundKey *btcec.PrivateKey) (wire.TxWitness, error) {
	sig, err := sign(script, tx, idx, amount, refundKey)
	if err != nil {
		return nil, err
	}
	return wire.TxWitness{sig, nil, script}, nil
}

// sign produces a low-S RFC6979 ECDSA signature over the BIP143 segwit-v0 sighash
// (SIGHASH_ALL) of input idx, with the trailing sighash-type byte appended.
func sign(script []byte, tx *wire.MsgTx, idx int, amount int64, key *btcec.PrivateKey) ([]byte, error) {
	if len(script) == 0 {
		return nil, ErrShortWitnessScript
	}
	fetcher := txscript.NewCannedPrevOutputFetcher(script, amount)
	sigHashes := txscript.NewTxSigHashes(tx, fetcher)
	digest, err := txscript.CalcWitnessSigHash(script, sigHashes, txscript.SigHashAll, tx, idx, amount)
	if err != nil {
		return nil, err
	}
	return append(ecdsa.Sign(key, digest).Serialize(), byte(txscript.SigHashAll)), nil
}
