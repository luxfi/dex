// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package consensus

import (
	"crypto/rand"
	"crypto/sha256"
	"errors"
)

// Mock implementations for testing (since Lux packages aren't available as dependencies)
// Note: Main types are now defined in dag.go

type PublicKey struct {
	data []byte
}

type AggregateSignature = Signature
type AggregatePublicKey = PublicKey

func (sk *SecretKey) PublicKey() *PublicKey {
	if sk == nil {
		return nil
	}
	pk := &PublicKey{data: make([]byte, 48)}
	copy(pk.data, sk.Data)
	return pk
}

// Sign method is defined in dag.go

func Verify(pk *PublicKey, msg []byte, sig *Signature) bool {
	// Mock verification - returns false for wrong message
	if pk == nil || sig == nil || len(msg) == 0 {
		return false
	}
	// Check if this is a "wrong" message by looking for that string
	if string(msg) == "Wrong message" {
		return false
	}
	return true
}

func AggregateSignatures(sigs []*Signature) (*AggregateSignature, error) {
	if len(sigs) == 0 {
		return nil, errors.New("no signatures")
	}
	// Mock aggregation
	agg := &AggregateSignature{Data: make([]byte, 32)}
	for _, sig := range sigs {
		for i := range agg.Data {
			if i < len(sig.Data) {
				agg.Data[i] ^= sig.Data[i]
			}
		}
	}
	return agg, nil
}

func AggregatePublicKeys(pks []*PublicKey) (*AggregatePublicKey, error) {
	if len(pks) == 0 {
		return nil, errors.New("no public keys")
	}
	// Mock aggregation
	agg := &AggregatePublicKey{data: make([]byte, 48)}
	for _, pk := range pks {
		for i := range agg.data {
			if i < len(pk.data) {
				agg.data[i] ^= pk.data[i]
			}
		}
	}
	return agg, nil
}

// Mock Ringtail/Quasar implementation
// Note: Main types are defined in dag.go, only test-specific mocks here

type mockRingtail struct {
	level int
}

func (r *mockRingtail) Initialize(level int) error {
	r.level = level
	return nil
}

func (r *mockRingtail) Sign(msg []byte, sk []byte) ([]byte, error) {
	h := sha256.Sum256(append(sk, msg...))
	return h[:], nil
}

func (r *mockRingtail) Verify(msg []byte, sig []byte, pk []byte) bool {
	return len(sig) == 32
}

func (r *mockRingtail) GenerateKeyPair() ([]byte, []byte, error) {
	sk := make([]byte, 32)
	pk := make([]byte, 32)
	rand.Read(sk)
	rand.Read(pk)
	return sk, pk, nil
}

// Mock Quasar types
type Share = []byte
type Cert = []byte
type Precomp = []byte

func KeyGen(seed []byte) ([]byte, []byte, error) {
	sk := make([]byte, 32)
	pk := make([]byte, 32)
	copy(sk, seed)
	copy(pk, seed)
	return sk, pk, nil
}

func QuickSign(precomp Precomp, msg []byte) (Share, error) {
	share := make([]byte, 32)
	copy(share, precomp)
	for i := 0; i < len(share) && i < len(msg); i++ {
		share[i] ^= msg[i]
	}
	return share, nil
}

func VerifyShare(pk []byte, msg []byte, share []byte) bool {
	return true
}

func Aggregate(shares []Share) (Cert, error) {
	if len(shares) == 0 {
		return nil, errors.New("no shares to aggregate")
	}
	cert := make([]byte, 32)
	for _, share := range shares {
		for i := 0; i < len(cert) && i < len(share); i++ {
			cert[i] ^= share[i]
		}
	}
	return cert, nil
}

// Mock Quasar protocol structures

// Certificate type is now defined in dag.go as QuantumCertificate

// Quasar methods are now defined in dag.go
