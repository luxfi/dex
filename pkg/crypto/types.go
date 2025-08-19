// Package crypto provides post-quantum cryptographic primitives for QZMQ
package crypto

import (
	"crypto/rand"
	"errors"
	"time"
)

// Algorithm identifiers
type (
	KemID  string
	SigID  string
	AeadID string
	HashID string
)

// Supported algorithms
const (
	// KEMs
	KemMLKEM768       KemID = "mlkem768"
	KemMLKEM1024      KemID = "mlkem1024"
	KemX25519         KemID = "x25519"
	KemHybridX25519ML768  KemID = "x25519+mlkem768"
	KemHybridX25519ML1024 KemID = "x25519+mlkem1024"

	// Signatures
	SigMLDSA2  SigID = "mldsa2"
	SigMLDSA3  SigID = "mldsa3"
	SigSLHDSA  SigID = "slhdsa"

	// AEADs
	AeadAES256GCM        AeadID = "aes256gcm"
	AeadChaCha20Poly1305 AeadID = "chacha20poly1305"
	AeadAESGCMSIV        AeadID = "aesgcmsiv"

	// Hashes
	HashSHA256 HashID = "sha256"
	HashSHA384 HashID = "sha384"
	HashSHA512 HashID = "sha512"
)

// Suite defines a complete cryptographic suite
type Suite struct {
	Kem  KemID
	Sig  SigID
	Aead AeadID
	Hash HashID
}

// DefaultSuite is the recommended suite for most use cases
var DefaultSuite = Suite{
	Kem:  KemHybridX25519ML768,
	Sig:  SigMLDSA2,
	Aead: AeadAES256GCM,
	Hash: HashSHA256,
}

// PQOnlySuite uses only post-quantum algorithms
var PQOnlySuite = Suite{
	Kem:  KemMLKEM768,
	Sig:  SigMLDSA2,
	Aead: AeadAES256GCM,
	Hash: HashSHA256,
}

// HandshakeKeys contains derived keys from handshake
type HandshakeKeys struct {
	ClientKey      []byte
	ServerKey      []byte
	ClientIV       []byte
	ServerIV       []byte
	ExporterSecret []byte
	KeyID          uint32
}

// KeyUpdateConfig defines key rotation policies
type KeyUpdateConfig struct {
	MaxMessages int64         // Max messages before rekey
	MaxBytes    int64         // Max bytes before rekey
	MaxAge      time.Duration // Max time before rekey
}

// DefaultKeyUpdateConfig provides safe defaults
var DefaultKeyUpdateConfig = KeyUpdateConfig{
	MaxMessages: 1 << 32,       // 2^32 messages
	MaxBytes:    1 << 50,       // 2^50 bytes (~1 PiB)
	MaxAge:      10 * time.Minute,
}

// AEAD provides authenticated encryption
type AEAD interface {
	// Seal encrypts and authenticates plaintext
	Seal(dst, nonce, plaintext, aad []byte) []byte
	
	// Open decrypts and verifies ciphertext
	Open(dst, nonce, ciphertext, aad []byte) ([]byte, error)
	
	// NonceSize returns the size of the nonce
	NonceSize() int
	
	// Overhead returns the tag size
	Overhead() int
}

// KEM provides key encapsulation
type KEM interface {
	// GenerateKeyPair generates a new key pair
	GenerateKeyPair() (PublicKey, PrivateKey, error)
	
	// Encapsulate generates shared secret and ciphertext
	Encapsulate(pk PublicKey) (ct []byte, ss []byte, error)
	
	// Decapsulate recovers shared secret from ciphertext
	Decapsulate(ct []byte, sk PrivateKey) (ss []byte, error)
	
	// PublicKeySize returns the public key size
	PublicKeySize() int
	
	// CiphertextSize returns the ciphertext size
	CiphertextSize() int
	
	// SharedSecretSize returns the shared secret size
	SharedSecretSize() int
}

// Signature provides digital signatures
type Signature interface {
	// GenerateKeyPair generates a new key pair
	GenerateKeyPair() (PublicKey, PrivateKey, error)
	
	// Sign creates a signature
	Sign(msg []byte, sk PrivateKey) (sig []byte, error)
	
	// Verify checks a signature
	Verify(msg, sig []byte, pk PublicKey) error
	
	// PublicKeySize returns the public key size
	PublicKeySize() int
	
	// SignatureSize returns the signature size
	SignatureSize() int
}

// PublicKey represents a public key
type PublicKey interface {
	Bytes() []byte
	Equal(PublicKey) bool
}

// PrivateKey represents a private key
type PrivateKey interface {
	Bytes() []byte
	Public() PublicKey
}

// Errors
var (
	ErrInvalidKeySize       = errors.New("invalid key size")
	ErrInvalidNonceSize     = errors.New("invalid nonce size")
	ErrAuthenticationFailed = errors.New("authentication failed")
	ErrInvalidSignature     = errors.New("invalid signature")
	ErrInvalidCiphertext    = errors.New("invalid ciphertext")
	ErrUnsupportedAlgorithm = errors.New("unsupported algorithm")
)

// RandomBytes generates cryptographically secure random bytes
func RandomBytes(n int) ([]byte, error) {
	b := make([]byte, n)
	_, err := rand.Read(b)
	return b, err
}