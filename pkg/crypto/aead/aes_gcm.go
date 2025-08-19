package aead

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"sync/atomic"
)

// AES256GCM implements AES-256-GCM AEAD
type AES256GCM struct {
	aead     cipher.AEAD
	streamID uint32
	seqNo    uint64 // Atomic counter
}

// NewAES256GCM creates a new AES-256-GCM AEAD
func NewAES256GCM(key []byte, streamID uint32) (*AES256GCM, error) {
	if len(key) != 32 {
		return nil, errors.New("AES-256-GCM requires 32-byte key")
	}
	
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	
	return &AES256GCM{
		aead:     aead,
		streamID: streamID,
		seqNo:    0,
	}, nil
}

// constructNonce builds deterministic nonce: stream_id || seq_no
func (a *AES256GCM) constructNonce() []byte {
	nonce := make([]byte, 12) // 96-bit nonce
	binary.BigEndian.PutUint32(nonce[0:4], a.streamID)
	binary.BigEndian.PutUint64(nonce[4:12], atomic.LoadUint64(&a.seqNo))
	return nonce
}

// Seal encrypts and authenticates plaintext
func (a *AES256GCM) Seal(dst, nonce, plaintext, aad []byte) []byte {
	if nonce == nil {
		// Use deterministic nonce
		nonce = a.constructNonce()
		atomic.AddUint64(&a.seqNo, 1)
	}
	
	if len(nonce) != a.NonceSize() {
		panic("invalid nonce size")
	}
	
	return a.aead.Seal(dst, nonce, plaintext, aad)
}

// Open decrypts and verifies ciphertext
func (a *AES256GCM) Open(dst, nonce, ciphertext, aad []byte) ([]byte, error) {
	if len(nonce) != a.NonceSize() {
		return nil, errors.New("invalid nonce size")
	}
	
	return a.aead.Open(dst, nonce, ciphertext, aad)
}

// NonceSize returns the nonce size (96 bits)
func (a *AES256GCM) NonceSize() int {
	return 12
}

// Overhead returns the tag size (128 bits)
func (a *AES256GCM) Overhead() int {
	return 16
}

// NextSeqNo returns the next sequence number
func (a *AES256GCM) NextSeqNo() uint64 {
	return atomic.LoadUint64(&a.seqNo)
}

// SetSeqNo sets the sequence number (for resumption)
func (a *AES256GCM) SetSeqNo(seq uint64) {
	atomic.StoreUint64(&a.seqNo, seq)
}

// ChaCha20Poly1305 implements ChaCha20-Poly1305 AEAD
type ChaCha20Poly1305 struct {
	aead     cipher.AEAD
	streamID uint32
	seqNo    uint64
}

// NewChaCha20Poly1305 creates a new ChaCha20-Poly1305 AEAD
func NewChaCha20Poly1305(key []byte, streamID uint32) (*ChaCha20Poly1305, error) {
	if len(key) != 32 {
		return nil, errors.New("ChaCha20-Poly1305 requires 32-byte key")
	}
	
	aead, err := newChaCha20Poly1305(key)
	if err != nil {
		return nil, err
	}
	
	return &ChaCha20Poly1305{
		aead:     aead,
		streamID: streamID,
		seqNo:    0,
	}, nil
}

// Import ChaCha20-Poly1305 from golang.org/x/crypto
func newChaCha20Poly1305(key []byte) (cipher.AEAD, error) {
	// This would import from golang.org/x/crypto/chacha20poly1305
	// For now, return a placeholder
	return nil, errors.New("ChaCha20-Poly1305 not yet implemented")
}

// Seal encrypts and authenticates plaintext
func (c *ChaCha20Poly1305) Seal(dst, nonce, plaintext, aad []byte) []byte {
	if nonce == nil {
		nonce = c.constructNonce()
		atomic.AddUint64(&c.seqNo, 1)
	}
	return c.aead.Seal(dst, nonce, plaintext, aad)
}

// Open decrypts and verifies ciphertext
func (c *ChaCha20Poly1305) Open(dst, nonce, ciphertext, aad []byte) ([]byte, error) {
	return c.aead.Open(dst, nonce, ciphertext, aad)
}

// constructNonce builds deterministic nonce
func (c *ChaCha20Poly1305) constructNonce() []byte {
	nonce := make([]byte, 12)
	binary.BigEndian.PutUint32(nonce[0:4], c.streamID)
	binary.BigEndian.PutUint64(nonce[4:12], atomic.LoadUint64(&c.seqNo))
	return nonce
}

// NonceSize returns 96 bits
func (c *ChaCha20Poly1305) NonceSize() int {
	return 12
}

// Overhead returns 128 bits
func (c *ChaCha20Poly1305) Overhead() int {
	return 16
}

// GenerateKey generates a random 256-bit key
func GenerateKey() ([]byte, error) {
	key := make([]byte, 32)
	_, err := rand.Read(key)
	return key, err
}