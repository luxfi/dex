package kdf

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/sha512"
	"errors"
	"hash"
	"io"

	"golang.org/x/crypto/hkdf"
)

// HKDF implements HKDF key derivation
type HKDF struct {
	hash func() hash.Hash
}

// NewHKDF creates a new HKDF instance
func NewHKDF(hashFunc string) (*HKDF, error) {
	h := &HKDF{}
	
	switch hashFunc {
	case "sha256":
		h.hash = sha256.New
	case "sha384":
		h.hash = sha512.New384
	case "sha512":
		h.hash = sha512.New
	default:
		return nil, errors.New("unsupported hash function")
	}
	
	return h, nil
}

// Extract performs HKDF-Extract
func (h *HKDF) Extract(salt, ikm []byte) []byte {
	if salt == nil {
		salt = make([]byte, h.hash().Size())
	}
	mac := hmac.New(h.hash, salt)
	mac.Write(ikm)
	return mac.Sum(nil)
}

// Expand performs HKDF-Expand
func (h *HKDF) Expand(prk, info []byte, length int) ([]byte, error) {
	r := hkdf.Expand(h.hash, prk, info)
	out := make([]byte, length)
	if _, err := io.ReadFull(r, out); err != nil {
		return nil, err
	}
	return out, nil
}

// DeriveKey derives a key using HKDF
func (h *HKDF) DeriveKey(secret, salt, info []byte, length int) ([]byte, error) {
	prk := h.Extract(salt, secret)
	return h.Expand(prk, info, length)
}

// DeriveHandshakeKeys derives all handshake keys from shared secrets
func DeriveHandshakeKeys(kemSecret, ecdheSecret []byte, hashFunc string) (*HandshakeKeys, error) {
	h, err := NewHKDF(hashFunc)
	if err != nil {
		return nil, err
	}
	
	// Combine secrets
	var combinedSecret []byte
	if kemSecret != nil && ecdheSecret != nil {
		// Hybrid mode: concatenate secrets
		combinedSecret = append(kemSecret, ecdheSecret...)
	} else if kemSecret != nil {
		combinedSecret = kemSecret
	} else if ecdheSecret != nil {
		combinedSecret = ecdheSecret
	} else {
		return nil, errors.New("no secrets provided")
	}
	
	// Extract handshake secret
	hsSecret := h.Extract(nil, combinedSecret)
	
	// Derive keys
	keys := &HandshakeKeys{}
	
	// Client traffic secret
	clientSecret, err := h.Expand(hsSecret, []byte("client traffic"), 32)
	if err != nil {
		return nil, err
	}
	keys.ClientKey, err = h.Expand(clientSecret, []byte("key"), 32)
	if err != nil {
		return nil, err
	}
	keys.ClientIV, err = h.Expand(clientSecret, []byte("iv"), 12)
	if err != nil {
		return nil, err
	}
	
	// Server traffic secret
	serverSecret, err := h.Expand(hsSecret, []byte("server traffic"), 32)
	if err != nil {
		return nil, err
	}
	keys.ServerKey, err = h.Expand(serverSecret, []byte("key"), 32)
	if err != nil {
		return nil, err
	}
	keys.ServerIV, err = h.Expand(serverSecret, []byte("iv"), 12)
	if err != nil {
		return nil, err
	}
	
	// Exporter secret
	keys.ExporterSecret, err = h.Expand(hsSecret, []byte("exporter"), 32)
	if err != nil {
		return nil, err
	}
	
	return keys, nil
}

// KeyUpdate performs key ratcheting
func KeyUpdate(prevKey []byte, hashFunc string) ([]byte, error) {
	h, err := NewHKDF(hashFunc)
	if err != nil {
		return nil, err
	}
	
	// Ratchet forward
	return h.Expand(prevKey, []byte("key update"), 32)
}

// HandshakeKeys contains derived keys
type HandshakeKeys struct {
	ClientKey      []byte
	ServerKey      []byte
	ClientIV       []byte
	ServerIV       []byte
	ExporterSecret []byte
	KeyID          uint32
}