// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package htlc

import (
	"encoding/hex"
	"math/big"
	"testing"

	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/chaincfg/chainhash"
	"github.com/btcsuite/btcd/txscript"
	"github.com/btcsuite/btcd/wire"
	"github.com/luxfi/geth/common"
	"github.com/stretchr/testify/require"
)

// --- shared primitive --------------------------------------------------------

func TestSecretHashlockRoundTrip(t *testing.T) {
	s, err := NewSecret()
	require.NoError(t, err)
	h := s.Hashlock()

	require.True(t, VerifyPreimage(s, h), "correct secret must verify against its hashlock")

	var wrong Secret
	copy(wrong[:], s[:])
	wrong[0] ^= 0xff
	require.False(t, VerifyPreimage(wrong, h), "mutated secret must not verify")
}

func TestNewSecretIsRandom(t *testing.T) {
	a, err := NewSecret()
	require.NoError(t, err)
	b, err := NewSecret()
	require.NoError(t, err)
	require.NotEqual(t, a, b, "two secrets must differ")
}

// --- EVM leg: selectors + calldata layout ------------------------------------

func TestSelectorsMatchSwapPrecompile(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  [4]byte
		want string
	}{
		{"lock", selLock, "4da2c728"},
		{"claim", selClaim, "84cc9dfb"},
		{"refund", selRefund, "7249fbb6"},
	} {
		require.Equal(t, tc.want, hex.EncodeToString(tc.got[:]), tc.name)
	}
}

func TestEVMCalldataLayout(t *testing.T) {
	s, err := NewSecret()
	require.NoError(t, err)
	h := s.Hashlock()
	recipient := common.HexToAddress("0x1111111111111111111111111111111111111111")
	refund := common.HexToAddress("0x2222222222222222222222222222222222222222")
	asset := common.HexToAddress("0x3333333333333333333333333333333333333333")
	amount := big.NewInt(1_000_000)
	timeout := uint64(1_700_000_600)
	var swapID [32]byte
	swapID[31] = 0x07

	lock := LockCalldata(h, recipient, refund, asset, amount, timeout)
	require.Len(t, lock, 4+6*32)
	require.Equal(t, selLock[:], lock[:4])
	require.Equal(t, h[:], lock[4:36])
	require.Equal(t, recipient.Bytes(), lock[4+32+12:4+2*32]) // address right-aligned in word
	require.Equal(t, amount.Bytes(), trimLeft(lock[4+4*32:4+5*32]))
	require.Equal(t, timeout, beUint64(lock[4+6*32-8:4+6*32]))

	claim := ClaimCalldata(swapID, s)
	require.Len(t, claim, 4+2*32)
	require.Equal(t, selClaim[:], claim[:4])
	require.Equal(t, swapID[:], claim[4:36])
	require.Equal(t, s[:], claim[36:68])

	ref := RefundCalldata(swapID)
	require.Len(t, ref, 4+32)
	require.Equal(t, selRefund[:], ref[:4])
	require.Equal(t, swapID[:], ref[4:36])
}

// --- Bitcoin leg: real script execution proof --------------------------------

const (
	htlcAmount = int64(100_000) // sats funded into the P2WSH output
	htlcFee    = int64(1_000)
	cltvTime   = uint32(1_700_000_000)
)

// htlcFixture holds a funded HTLC ready to be spent down either path.
type htlcFixture struct {
	script    []byte
	pkScript  []byte
	fundHash  chainhash.Hash
	preimage  Secret
	recipient *btcec.PrivateKey
	refundKey *btcec.PrivateKey
}

func newFixture(t *testing.T) htlcFixture {
	t.Helper()
	recipient, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	refundKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	s, err := NewSecret()
	require.NoError(t, err)
	h := s.Hashlock()

	script, err := BuildHTLCScript(h, recipient.PubKey(), refundKey.PubKey(), cltvTime)
	require.NoError(t, err)

	addr, err := P2WSHAddress(script, &chaincfg.MainNetParams)
	require.NoError(t, err)
	pkScript, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	fund := wire.NewMsgTx(2)
	fund.AddTxIn(&wire.TxIn{PreviousOutPoint: wire.OutPoint{Index: 0xffffffff}})
	fund.AddTxOut(wire.NewTxOut(htlcAmount, pkScript))

	return htlcFixture{
		script:    script,
		pkScript:  pkScript,
		fundHash:  fund.TxHash(),
		preimage:  s,
		recipient: recipient,
		refundKey: refundKey,
	}
}

// spendTx builds a 1-in/1-out tx spending the HTLC output with the given input
// sequence and nLockTime.
func (f htlcFixture) spendTx(sequence, lockTime uint32) *wire.MsgTx {
	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{Hash: f.fundHash, Index: 0},
		Sequence:         sequence,
	})
	tx.AddTxOut(wire.NewTxOut(htlcAmount-htlcFee, f.pkScript))
	tx.LockTime = lockTime
	return tx
}

// runEngine executes the full segwit-v0 script verification for input 0 and
// returns whether the spend is VALID under Bitcoin consensus (standard flags,
// including CLTV).
func (f htlcFixture) runEngine(t *testing.T, tx *wire.MsgTx) error {
	t.Helper()
	fetcher := txscript.NewCannedPrevOutputFetcher(f.pkScript, htlcAmount)
	sigHashes := txscript.NewTxSigHashes(tx, fetcher)
	vm, err := txscript.NewEngine(
		f.pkScript, tx, 0, txscript.StandardVerifyFlags, nil, sigHashes, htlcAmount, fetcher,
	)
	require.NoError(t, err)
	return vm.Execute()
}

func TestClaimPathSpends(t *testing.T) {
	f := newFixture(t)

	// Valid claim: correct preimage spends with no locktime constraint.
	tx := f.spendTx(wire.MaxTxInSequenceNum, 0)
	w, err := ClaimWitness(f.script, tx, 0, htlcAmount, f.preimage, f.recipient)
	require.NoError(t, err)
	tx.TxIn[0].Witness = w
	require.NoError(t, f.runEngine(t, tx), "claim with correct preimage must validate")

	// Wrong preimage: OP_EQUALVERIFY fails, spend is rejected.
	bad := f.spendTx(wire.MaxTxInSequenceNum, 0)
	var wrong Secret
	wrong[0] = 0xab
	wbad, err := ClaimWitness(f.script, bad, 0, htlcAmount, wrong, f.recipient)
	require.NoError(t, err)
	bad.TxIn[0].Witness = wbad
	require.Error(t, f.runEngine(t, bad), "claim with wrong preimage must fail")
}

func TestRefundPathSpends(t *testing.T) {
	f := newFixture(t)
	const nonFinal = wire.MaxTxInSequenceNum - 1

	// Valid refund: nLockTime >= CLTV and a non-final sequence enable OP_CLTV.
	tx := f.spendTx(nonFinal, cltvTime)
	w, err := RefundWitness(f.script, tx, 0, htlcAmount, f.refundKey)
	require.NoError(t, err)
	tx.TxIn[0].Witness = w
	require.NoError(t, f.runEngine(t, tx), "refund at/after locktime must validate")

	// Before locktime: nLockTime < CLTV, OP_CHECKLOCKTIMEVERIFY rejects.
	early := f.spendTx(nonFinal, cltvTime-1)
	we, err := RefundWitness(f.script, early, 0, htlcAmount, f.refundKey)
	require.NoError(t, err)
	early.TxIn[0].Witness = we
	require.Error(t, f.runEngine(t, early), "refund before locktime must fail")

	// Final sequence disables CLTV evaluation entirely -> rejected.
	finalSeq := f.spendTx(wire.MaxTxInSequenceNum, cltvTime)
	wf, err := RefundWitness(f.script, finalSeq, 0, htlcAmount, f.refundKey)
	require.NoError(t, err)
	finalSeq.TxIn[0].Witness = wf
	require.Error(t, f.runEngine(t, finalSeq), "refund with final sequence must fail CLTV")
}

func TestRefundKeyCannotTakeClaimPath(t *testing.T) {
	// Cross-leg binding: the refund key signing the claim branch must not spend,
	// even with the correct preimage (CHECKSIG is to the recipient pubkey).
	f := newFixture(t)
	tx := f.spendTx(wire.MaxTxInSequenceNum, 0)
	w, err := ClaimWitness(f.script, tx, 0, htlcAmount, f.preimage, f.refundKey)
	require.NoError(t, err)
	tx.TxIn[0].Witness = w
	require.Error(t, f.runEngine(t, tx), "wrong key on claim path must fail CHECKSIG")
}

// --- small test helpers ------------------------------------------------------

func trimLeft(b []byte) []byte {
	i := 0
	for i < len(b) && b[i] == 0 {
		i++
	}
	return b[i:]
}

func beUint64(b []byte) uint64 {
	var v uint64
	for _, x := range b {
		v = v<<8 | uint64(x)
	}
	return v
}
