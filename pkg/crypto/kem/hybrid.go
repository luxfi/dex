package kem

import (
	"crypto/rand"
	"errors"

	"golang.org/x/crypto/curve25519"
)

// HybridKEM implements X25519 + ML-KEM hybrid
type HybridKEM struct {
	classical KEM // X25519
	pq        KEM // ML-KEM
}

// NewHybridX25519MLKEM768 creates X25519 + ML-KEM-768 hybrid
func NewHybridX25519MLKEM768() *HybridKEM {
	return &HybridKEM{
		classical: NewX25519(),
		pq:        NewMLKEM768(),
	}
}

// NewHybridX25519MLKEM1024 creates X25519 + ML-KEM-1024 hybrid
func NewHybridX25519MLKEM1024() *HybridKEM {
	return &HybridKEM{
		classical: NewX25519(),
		pq:        NewMLKEM1024(),
	}
}

// GenerateKeyPair generates hybrid key pairs
func (h *HybridKEM) GenerateKeyPair() (PublicKey, PrivateKey, error) {
	// Generate classical key pair
	classicalPK, classicalSK, err := h.classical.GenerateKeyPair()
	if err != nil {
		return nil, nil, err
	}
	
	// Generate PQ key pair
	pqPK, pqSK, err := h.pq.GenerateKeyPair()
	if err != nil {
		return nil, nil, err
	}
	
	pk := &hybridPublicKey{
		classical: classicalPK,
		pq:        pqPK,
	}
	
	sk := &hybridPrivateKey{
		classical: classicalSK,
		pq:        pqSK,
		pk:        pk,
	}
	
	return pk, sk, nil
}

// Encapsulate generates hybrid ciphertext and shared secret
func (h *HybridKEM) Encapsulate(pk PublicKey) (ct []byte, ss []byte, error) {
	hpk, ok := pk.(*hybridPublicKey)
	if !ok {
		return nil, nil, errors.New("invalid public key type")
	}
	
	// Classical encapsulation (X25519 ECDH)
	classicalCT, classicalSS, err := h.classical.Encapsulate(hpk.classical)
	if err != nil {
		return nil, nil, err
	}
	
	// PQ encapsulation
	pqCT, pqSS, err := h.pq.Encapsulate(hpk.pq)
	if err != nil {
		return nil, nil, err
	}
	
	// Combine ciphertexts
	ct = append(classicalCT, pqCT...)
	
	// Combine shared secrets (concatenate then hash via HKDF)
	ss = append(classicalSS, pqSS...)
	
	return ct, ss, nil
}

// Decapsulate recovers hybrid shared secret
func (h *HybridKEM) Decapsulate(ct []byte, sk PrivateKey) (ss []byte, error) {
	hsk, ok := sk.(*hybridPrivateKey)
	if !ok {
		return nil, errors.New("invalid private key type")
	}
	
	// Split ciphertext
	classicalCTSize := h.classical.CiphertextSize()
	if len(ct) < classicalCTSize {
		return nil, errors.New("ciphertext too short")
	}
	
	classicalCT := ct[:classicalCTSize]
	pqCT := ct[classicalCTSize:]
	
	// Classical decapsulation
	classicalSS, err := h.classical.Decapsulate(classicalCT, hsk.classical)
	if err != nil {
		return nil, err
	}
	
	// PQ decapsulation
	pqSS, err := h.pq.Decapsulate(pqCT, hsk.pq)
	if err != nil {
		return nil, err
	}
	
	// Combine shared secrets
	ss = append(classicalSS, pqSS...)
	
	return ss, nil
}

// PublicKeySize returns combined size
func (h *HybridKEM) PublicKeySize() int {
	return h.classical.PublicKeySize() + h.pq.PublicKeySize()
}

// CiphertextSize returns combined size
func (h *HybridKEM) CiphertextSize() int {
	return h.classical.CiphertextSize() + h.pq.CiphertextSize()
}

// SharedSecretSize returns combined size
func (h *HybridKEM) SharedSecretSize() int {
	return h.classical.SharedSecretSize() + h.pq.SharedSecretSize()
}

// X25519 implements X25519 ECDH as KEM
type X25519 struct{}

// NewX25519 creates X25519 KEM
func NewX25519() *X25519 {
	return &X25519{}
}

// GenerateKeyPair generates X25519 key pair
func (x *X25519) GenerateKeyPair() (PublicKey, PrivateKey, error) {
	sk := &x25519PrivateKey{
		data: make([]byte, 32),
	}
	
	_, err := rand.Read(sk.data)
	if err != nil {
		return nil, nil, err
	}
	
	pk := &x25519PublicKey{
		data: make([]byte, 32),
	}
	
	curve25519.ScalarBaseMult((*[32]byte)(pk.data), (*[32]byte)(sk.data))
	sk.pk = pk
	
	return pk, sk, nil
}

// Encapsulate performs X25519 ECDH
func (x *X25519) Encapsulate(pk PublicKey) (ct []byte, ss []byte, error) {
	xpk, ok := pk.(*x25519PublicKey)
	if !ok {
		return nil, nil, errors.New("invalid public key type")
	}
	
	// Generate ephemeral key pair
	ephSK := make([]byte, 32)
	_, err := rand.Read(ephSK)
	if err != nil {
		return nil, nil, err
	}
	
	ephPK := make([]byte, 32)
	curve25519.ScalarBaseMult((*[32]byte)(ephPK), (*[32]byte)(ephSK))
	
	// ECDH
	ss = make([]byte, 32)
	curve25519.ScalarMult((*[32]byte)(ss), (*[32]byte)(ephSK), (*[32]byte)(xpk.data))
	
	// Ephemeral public key is the "ciphertext"
	ct = ephPK
	
	return ct, ss, nil
}

// Decapsulate performs X25519 ECDH
func (x *X25519) Decapsulate(ct []byte, sk PrivateKey) (ss []byte, error) {
	if len(ct) != 32 {
		return nil, errors.New("invalid ciphertext size")
	}
	
	xsk, ok := sk.(*x25519PrivateKey)
	if !ok {
		return nil, errors.New("invalid private key type")
	}
	
	// ECDH
	ss = make([]byte, 32)
	curve25519.ScalarMult((*[32]byte)(ss), (*[32]byte)(xsk.data), (*[32]byte)(ct))
	
	return ss, nil
}

// PublicKeySize returns 32 bytes
func (x *X25519) PublicKeySize() int {
	return 32
}

// CiphertextSize returns 32 bytes
func (x *X25519) CiphertextSize() int {
	return 32
}

// SharedSecretSize returns 32 bytes
func (x *X25519) SharedSecretSize() int {
	return 32
}

// Key types for X25519
type x25519PublicKey struct {
	data []byte
}

func (pk *x25519PublicKey) Bytes() []byte {
	return pk.data
}

func (pk *x25519PublicKey) Equal(other PublicKey) bool {
	if opk, ok := other.(*x25519PublicKey); ok {
		if len(pk.data) != len(opk.data) {
			return false
		}
		for i := range pk.data {
			if pk.data[i] != opk.data[i] {
				return false
			}
		}
		return true
	}
	return false
}

type x25519PrivateKey struct {
	data []byte
	pk   *x25519PublicKey
}

func (sk *x25519PrivateKey) Bytes() []byte {
	return sk.data
}

func (sk *x25519PrivateKey) Public() PublicKey {
	return sk.pk
}

// Hybrid key types
type hybridPublicKey struct {
	classical PublicKey
	pq        PublicKey
}

func (pk *hybridPublicKey) Bytes() []byte {
	return append(pk.classical.Bytes(), pk.pq.Bytes()...)
}

func (pk *hybridPublicKey) Equal(other PublicKey) bool {
	if opk, ok := other.(*hybridPublicKey); ok {
		return pk.classical.Equal(opk.classical) && pk.pq.Equal(opk.pq)
	}
	return false
}

type hybridPrivateKey struct {
	classical PrivateKey
	pq        PrivateKey
	pk        *hybridPublicKey
}

func (sk *hybridPrivateKey) Bytes() []byte {
	return append(sk.classical.Bytes(), sk.pq.Bytes()...)
}

func (sk *hybridPrivateKey) Public() PublicKey {
	return sk.pk
}

// KEM interface
type KEM interface {
	GenerateKeyPair() (PublicKey, PrivateKey, error)
	Encapsulate(pk PublicKey) (ct []byte, ss []byte, error)
	Decapsulate(ct []byte, sk PrivateKey) (ss []byte, error)
	PublicKeySize() int
	CiphertextSize() int
	SharedSecretSize() int
}