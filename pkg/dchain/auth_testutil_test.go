// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	stdcrypto "crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"sync"
	"testing"

	"github.com/luxfi/crypto"
	"github.com/luxfi/crypto/mldsa"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/geth/common"
)

// auth_testutil_test.go is the shared test signing harness. The production gate
// (auth.go / block.execute) refuses any money-moving tx that is not signed by the
// asserted account, so every consensus-path test builds SIGNED txs through one
// identity model here: a deterministic per-(test,name) keypair, the 16-byte
// account it derives (Account16 — the SAME fold production uses), and a monotonic
// per-account nonce. Tests keep passing human names ("maker-a"); the helpers fold
// the name to its cryptographic account on the wire and sign.

// testAcct is a deterministic test identity. secp256k1 by default; ML-DSA-65 when
// built via pqAcctFor (the post-quantum lane). user is string(account) — exactly
// the zapwire user[16] field the wire carries and the ledger keys balances by.
type testAcct struct {
	scheme  lx.AuthScheme
	priv    *ecdsa.PrivateKey  // secp256k1 (nil for PQ)
	mlpriv  *mldsa.PrivateKey  // ML-DSA-65 (nil for classical)
	pubKey  []byte             // ML-DSA pubkey bytes (nil for secp)
	account userKey            // 16-byte settlement account
	user    string             // string(account[:]) — the wire user
	nonce   uint64             // next nonce to use
}

// authFor produces an authorization envelope binding (type, nonce, body) under
// the account's key. It does NOT advance the nonce (callers choose the nonce, so
// the security tests can forge replays/gaps explicitly).
func (a *testAcct) authFor(typ TxType, nonce uint64, body []byte) *TxAuth {
	digest := txAuthDigest(typ, a.scheme, nonce, body)
	switch a.scheme {
	case lx.AuthMLDSA65:
		sig, err := a.mlpriv.Sign(rand.Reader, digest[:], stdcrypto.Hash(0))
		if err != nil {
			panic("dchain test: ML-DSA sign: " + err.Error())
		}
		return &TxAuth{Scheme: a.scheme, Nonce: nonce, Sig: sig, PubKey: a.pubKey}
	default:
		sig, err := crypto.Sign(digest[:], a.priv)
		if err != nil {
			panic("dchain test: secp256k1 sign: " + err.Error())
		}
		return &TxAuth{Scheme: a.scheme, Nonce: nonce, Sig: sig}
	}
}

// signed builds a signed Tx at the account's CURRENT nonce and advances it. This
// is the normal path every migrated builder uses.
func (a *testAcct) signed(t *testing.T, typ TxType, body []byte) *Tx {
	t.Helper()
	auth := a.authFor(typ, a.nonce, body)
	a.nonce++
	tx, err := NewSignedTx(typ, body, auth)
	if err != nil {
		t.Fatalf("NewSignedTx(%s): %v", typ, err)
	}
	return tx
}

// identityRegistry maps (test, name) -> a stable testAcct so the same human name
// yields the same key/account/nonce within a test, independent across tests.
type identityRegistry struct {
	mu    sync.Mutex
	accts map[string]*testAcct
}

var idents = &identityRegistry{accts: map[string]*testAcct{}}

func regKey(t *testing.T, name string) string { return t.Name() + "\x00" + name }

// secpScalar derives a valid secp256k1 private scalar deterministically from a
// seed string (re-hash on the vanishingly rare out-of-range value).
func secpScalar(seed string) *ecdsa.PrivateKey {
	d := crypto.Keccak256([]byte("lux.dchain.test.secp.v1"), []byte(seed))
	for {
		priv, err := crypto.ToECDSA(d)
		if err == nil {
			return priv
		}
		d = crypto.Keccak256(d)
	}
}

// acctFor returns the deterministic secp256k1 identity for (test, name).
func acctFor(t *testing.T, name string) *testAcct {
	t.Helper()
	idents.mu.Lock()
	defer idents.mu.Unlock()
	k := regKey(t, name)
	if a, ok := idents.accts[k]; ok {
		return a
	}
	priv := secpScalar(name)
	// Derive the geth-common Address the gate recovers (crypto.PubkeyToAddress
	// returns crypto/common.Address; the gate keys on geth/common.Address — the
	// underlying 20 keccak bytes are identical, so bridge by bytes).
	addr := common.BytesToAddress(crypto.PubkeyToAddress(priv.PublicKey).Bytes())
	acc := Account16(lx.AuthSecp256k1, addr)
	a := &testAcct{scheme: lx.AuthSecp256k1, priv: priv, account: acc, user: string(acc[:])}
	idents.accts[k] = a
	return a
}

// pqAcctFor returns the ML-DSA-65 identity for (test, name) — the post-quantum
// lane used by the dual-scheme security tests. Keyed separately ("pq:") so a
// classical and a PQ identity for the same name never alias.
func pqAcctFor(t *testing.T, name string) *testAcct {
	t.Helper()
	idents.mu.Lock()
	defer idents.mu.Unlock()
	k := regKey(t, "pq:"+name)
	if a, ok := idents.accts[k]; ok {
		return a
	}
	priv, err := mldsa.GenerateKey(rand.Reader, mldsa.MLDSA65)
	if err != nil {
		t.Fatalf("mldsa.GenerateKey: %v", err)
	}
	pub := priv.PublicKey.Bytes()
	addr := lx.AddressFromMLDSAPubKey(pub)
	acc := Account16(lx.AuthMLDSA65, addr)
	a := &testAcct{scheme: lx.AuthMLDSA65, mlpriv: priv, pubKey: pub, account: acc, user: string(acc[:])}
	idents.accts[k] = a
	return a
}

// wireUser resolves a human name to the 16-byte account string the wire/ledger
// uses, so balance assertions key the SAME account the signed txs credited.
func wireUser(t *testing.T, name string) string {
	t.Helper()
	return acctFor(t, name).user
}

// idOf resolves a human name to the 16-byte settlement account (userKey) the
// ledger and the orderuser: index key by — the value the gate binds a signature
// to. Identity assertions compare against this, NOT the raw name bytes.
func idOf(t *testing.T, name string) userKey {
	t.Helper()
	return acctFor(t, name).account
}

// signedPayload returns the RPC payload a client sends to a money-moving handler:
// the frozen zapwire body followed by the auth envelope, signed by `name` at its
// next nonce. `body` MUST already carry name's account (wireUser) as its user[16].
func signedPayload(t *testing.T, name string, typ TxType, body []byte) []byte {
	t.Helper()
	a := acctFor(t, name)
	auth := a.authFor(typ, a.nonce, body)
	a.nonce++
	out := make([]byte, 0, len(body)+authHeaderSize+len(auth.Sig)+len(auth.PubKey))
	out = append(out, body...)
	return append(out, auth.encode()...)
}

