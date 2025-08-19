// Package qzmq implements QuantumZMQ - post-quantum secure ZeroMQ transport
package qzmq

import (
	"encoding/binary"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/dex/pkg/crypto"
	"github.com/luxfi/dex/pkg/crypto/aead"
	"github.com/luxfi/dex/pkg/crypto/kem"
	"github.com/luxfi/dex/pkg/crypto/kdf"
)

// Constants
const (
	MechanismName    = "QZMQ1"
	ProtocolVersion  = 1
	MaxMessageSize   = 1 << 20 // 1MB
	NonceSize        = 12       // 96 bits
	TagSize          = 16       // 128 bits
	DefaultStreamID  = 1
)

// Message types
const (
	MsgClientHello byte = 0x01
	MsgServerHello byte = 0x02
	MsgClientKey   byte = 0x03
	MsgFinished    byte = 0x04
	MsgData        byte = 0x10
	MsgKeyUpdate   byte = 0x20
	MsgAlert       byte = 0x30
)

// Options for QZMQ
type Options struct {
	Suite             crypto.Suite
	PQOnly            bool          // Require PQ-only algorithms
	AllowPSK          bool          // Enable resumption
	Allow0RTT         bool          // Enable 0-RTT
	KeyUpdateMessages int64         // Messages before rekey
	KeyUpdateBytes    int64         // Bytes before rekey
	KeyUpdateTime     time.Duration // Time before rekey
	AntiDoSCookie     bool          // Enable stateless cookies
}

// DefaultOptions provides secure defaults
var DefaultOptions = Options{
	Suite:             crypto.DefaultSuite,
	PQOnly:            false,
	AllowPSK:          true,
	Allow0RTT:         false,
	KeyUpdateMessages: 1 << 32,
	KeyUpdateBytes:    1 << 50,
	KeyUpdateTime:     10 * time.Minute,
	AntiDoSCookie:     true,
}

// PQOnlyOptions enforces post-quantum only
var PQOnlyOptions = Options{
	Suite:             crypto.PQOnlySuite,
	PQOnly:            true,
	AllowPSK:          false,
	Allow0RTT:         false,
	KeyUpdateMessages: 1 << 30,
	KeyUpdateBytes:    1 << 48,
	KeyUpdateTime:     5 * time.Minute,
	AntiDoSCookie:     true,
}

// Connection represents a QZMQ connection
type Connection struct {
	// Configuration
	opts     Options
	isServer bool
	
	// Handshake state
	state           HandshakeState
	clientRandom    []byte
	serverRandom    []byte
	transcript      []byte
	sessionID       []byte
	
	// Crypto state
	kem             kem.KEM
	localKEMSK      kem.PrivateKey
	localKEMPK      kem.PublicKey
	remoteKEMPK     kem.PublicKey
	sharedSecret    []byte
	
	// Traffic keys
	keys            *kdf.HandshakeKeys
	clientAEAD      *aead.AES256GCM
	serverAEAD      *aead.AES256GCM
	
	// Stream management
	streamID        uint32
	inSeqNo         uint64
	outSeqNo        uint64
	lastInSeqNo     uint64 // For replay detection
	
	// Key rotation
	messagesSent    int64
	bytesSent       int64
	lastKeyUpdate   time.Time
	
	// Synchronization
	mu              sync.RWMutex
}

// HandshakeState tracks handshake progress
type HandshakeState int

const (
	StateInit HandshakeState = iota
	StateClientHello
	StateServerHello
	StateClientKey
	StateFinished
	StateEstablished
	StateError
)

// QZRecord represents an encrypted frame
type QZRecord struct {
	StreamID uint32
	SeqNo    uint64
	AADLen   uint8
	AAD      []byte
	CT       []byte
	Tag      [16]byte
}

// Marshal serializes QZRecord
func (r *QZRecord) Marshal() []byte {
	size := 4 + 8 + 1 + int(r.AADLen) + len(r.CT) + 16
	buf := make([]byte, size)
	
	offset := 0
	binary.BigEndian.PutUint32(buf[offset:], r.StreamID)
	offset += 4
	binary.BigEndian.PutUint64(buf[offset:], r.SeqNo)
	offset += 8
	buf[offset] = r.AADLen
	offset++
	if r.AADLen > 0 {
		copy(buf[offset:], r.AAD)
		offset += int(r.AADLen)
	}
	copy(buf[offset:], r.CT)
	offset += len(r.CT)
	copy(buf[offset:], r.Tag[:])
	
	return buf
}

// Unmarshal deserializes QZRecord
func (r *QZRecord) Unmarshal(data []byte) error {
	if len(data) < 13 {
		return errors.New("record too short")
	}
	
	offset := 0
	r.StreamID = binary.BigEndian.Uint32(data[offset:])
	offset += 4
	r.SeqNo = binary.BigEndian.Uint64(data[offset:])
	offset += 8
	r.AADLen = data[offset]
	offset++
	
	if len(data) < offset+int(r.AADLen)+16 {
		return errors.New("record truncated")
	}
	
	if r.AADLen > 0 {
		r.AAD = make([]byte, r.AADLen)
		copy(r.AAD, data[offset:])
		offset += int(r.AADLen)
	}
	
	ctLen := len(data) - offset - 16
	if ctLen < 0 {
		return errors.New("invalid ciphertext length")
	}
	
	r.CT = make([]byte, ctLen)
	copy(r.CT, data[offset:])
	offset += ctLen
	
	copy(r.Tag[:], data[offset:])
	
	return nil
}

// SessionCache stores resumption tickets
type SessionCache struct {
	mu      sync.RWMutex
	tickets map[string]*SessionTicket
}

// SessionTicket for resumption
type SessionTicket struct {
	ID           []byte
	MasterSecret []byte
	Suite        crypto.Suite
	CreatedAt    time.Time
	ExpiresAt    time.Time
}

// Metrics tracks connection statistics
type Metrics struct {
	HandshakeTime     time.Duration
	MessagesSent      uint64
	MessagesReceived  uint64
	BytesSent         uint64
	BytesReceived     uint64
	KeyUpdates        uint32
	AEADFailures      uint32
	ReplayAttempts    uint32
}

// GetMetrics returns connection metrics
func (c *Connection) GetMetrics() Metrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return Metrics{
		MessagesSent:     uint64(atomic.LoadInt64(&c.messagesSent)),
		BytesSent:        uint64(atomic.LoadInt64(&c.bytesSent)),
		KeyUpdates:       0, // TODO: track
		AEADFailures:     0, // TODO: track
		ReplayAttempts:   0, // TODO: track
	}
}

// NeedsKeyUpdate checks if key rotation is needed
func (c *Connection) NeedsKeyUpdate() bool {
	if atomic.LoadInt64(&c.messagesSent) >= c.opts.KeyUpdateMessages {
		return true
	}
	if atomic.LoadInt64(&c.bytesSent) >= c.opts.KeyUpdateBytes {
		return true
	}
	if time.Since(c.lastKeyUpdate) >= c.opts.KeyUpdateTime {
		return true
	}
	return false
}

// Errors
var (
	ErrInvalidState        = errors.New("invalid handshake state")
	ErrInvalidMessage      = errors.New("invalid message")
	ErrAuthenticationFailed = errors.New("authentication failed")
	ErrReplayDetected      = errors.New("replay detected")
	ErrKeyUpdateRequired   = errors.New("key update required")
	ErrUnsupportedSuite    = errors.New("unsupported cipher suite")
	ErrPQOnlyViolation     = errors.New("PQ-only policy violated")
)