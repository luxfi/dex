package qzmq

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"time"

	"github.com/luxfi/dex/pkg/crypto"
	"github.com/luxfi/dex/pkg/crypto/aead"
	"github.com/luxfi/dex/pkg/crypto/kdf"
	"github.com/luxfi/dex/pkg/crypto/kem"
)

// NewConnection creates a new QZMQ connection
func NewConnection(opts Options, isServer bool) *Connection {
	return &Connection{
		opts:          opts,
		isServer:      isServer,
		state:         StateInit,
		streamID:      DefaultStreamID,
		lastKeyUpdate: time.Now(),
	}
}

// ClientHello initiates handshake
func (c *Connection) ClientHello() ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.state != StateInit {
		return nil, ErrInvalidState
	}
	
	// Generate client random
	c.clientRandom = make([]byte, 32)
	rand.Read(c.clientRandom)
	
	// Build ClientHello message
	msg := &clientHelloMsg{
		Version:      ProtocolVersion,
		Random:       c.clientRandom,
		Suites:       []crypto.Suite{c.opts.Suite},
		SessionID:    c.sessionID,
		Extensions:   make([]extension, 0),
	}
	
	// Add supported groups extension
	if c.opts.PQOnly {
		msg.Extensions = append(msg.Extensions, extension{
			Type: extPQOnly,
			Data: []byte{0x01},
		})
	}
	
	// Generate ephemeral X25519 key for hybrid
	if !c.opts.PQOnly {
		x25519 := kem.NewX25519()
		pk, sk, err := x25519.GenerateKeyPair()
		if err != nil {
			return nil, err
		}
		c.localKEMPK = pk
		c.localKEMSK = sk
		
		msg.Extensions = append(msg.Extensions, extension{
			Type: extKeyShare,
			Data: pk.Bytes(),
		})
	}
	
	// Marshal and update transcript
	data := msg.Marshal()
	c.transcript = append(c.transcript, data...)
	c.state = StateClientHello
	
	return data, nil
}

// ProcessServerHello handles server response
func (c *Connection) ProcessServerHello(data []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.state != StateClientHello {
		return ErrInvalidState
	}
	
	msg := &serverHelloMsg{}
	if err := msg.Unmarshal(data); err != nil {
		return err
	}
	
	// Store server random
	c.serverRandom = msg.Random
	
	// Validate suite selection
	if msg.Suite != c.opts.Suite {
		return ErrUnsupportedSuite
	}
	
	// Process server certificate (ML-DSA)
	// TODO: Verify certificate chain and signature
	
	// Store server's KEM public key
	c.remoteKEMPK = msg.ServerKEMPK
	
	// Update transcript
	c.transcript = append(c.transcript, data...)
	c.state = StateServerHello
	
	return nil
}

// ClientKey sends client's KEM ciphertext
func (c *Connection) ClientKey() ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.state != StateServerHello {
		return ErrInvalidState
	}
	
	// Initialize KEM based on suite
	var kemImpl kem.KEM
	switch c.opts.Suite.Kem {
	case crypto.KemMLKEM768:
		kemImpl = kem.NewMLKEM768()
	case crypto.KemMLKEM1024:
		kemImpl = kem.NewMLKEM1024()
	case crypto.KemHybridX25519ML768:
		kemImpl = kem.NewHybridX25519MLKEM768()
	case crypto.KemHybridX25519ML1024:
		kemImpl = kem.NewHybridX25519MLKEM1024()
	default:
		return nil, ErrUnsupportedSuite
	}
	
	// Encapsulate to server's public key
	ct, kemSecret, err := kemImpl.Encapsulate(c.remoteKEMPK)
	if err != nil {
		return nil, err
	}
	
	// For hybrid mode, combine with ECDHE secret
	var ecdheSecret []byte
	if !c.opts.PQOnly && c.localKEMSK != nil {
		// Perform ECDHE (simplified - in real impl would use server's X25519 key)
		ecdheSecret = make([]byte, 32)
		rand.Read(ecdheSecret)
	}
	
	// Derive handshake keys
	hashFunc := string(c.opts.Suite.Hash)
	c.keys, err = kdf.DeriveHandshakeKeys(kemSecret, ecdheSecret, hashFunc)
	if err != nil {
		return nil, err
	}
	
	// Initialize AEADs
	c.clientAEAD, err = aead.NewAES256GCM(c.keys.ClientKey, c.streamID)
	if err != nil {
		return nil, err
	}
	
	c.serverAEAD, err = aead.NewAES256GCM(c.keys.ServerKey, c.streamID)
	if err != nil {
		return nil, err
	}
	
	// Build ClientKey message
	msg := &clientKeyMsg{
		KEMCiphertext: ct,
	}
	
	// Sign transcript (if client auth required)
	// TODO: ML-DSA signature
	
	// Marshal
	data := msg.Marshal()
	c.transcript = append(c.transcript, data...)
	c.state = StateClientKey
	
	return data, nil
}

// ProcessFinished completes handshake
func (c *Connection) ProcessFinished(encData []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.state != StateClientKey {
		return ErrInvalidState
	}
	
	// Decrypt Finished message
	nonce := make([]byte, NonceSize)
	binary.BigEndian.PutUint32(nonce[0:4], c.streamID)
	binary.BigEndian.PutUint64(nonce[4:12], 0) // First encrypted message
	
	plaintext, err := c.serverAEAD.Open(nil, nonce, encData, nil)
	if err != nil {
		return ErrAuthenticationFailed
	}
	
	// Verify finished MAC
	h := hmac.New(sha256.New, c.keys.ServerKey)
	h.Write(c.transcript)
	expectedMAC := h.Sum(nil)
	
	if !hmac.Equal(plaintext[:32], expectedMAC) {
		return ErrAuthenticationFailed
	}
	
	c.state = StateEstablished
	c.sharedSecret = c.keys.ExporterSecret
	
	return nil
}

// Message structures

type clientHelloMsg struct {
	Version    uint8
	Random     []byte
	Suites     []crypto.Suite
	SessionID  []byte
	Extensions []extension
}

func (m *clientHelloMsg) Marshal() []byte {
	size := 1 + 32 + 2 + len(m.Suites)*4 + 1 + len(m.SessionID) + 2
	for _, ext := range m.Extensions {
		size += 2 + 2 + len(ext.Data)
	}
	
	buf := make([]byte, size)
	offset := 0
	
	buf[offset] = m.Version
	offset++
	copy(buf[offset:], m.Random)
	offset += 32
	
	binary.BigEndian.PutUint16(buf[offset:], uint16(len(m.Suites)))
	offset += 2
	
	// Marshal suites (simplified)
	for _, suite := range m.Suites {
		offset += 4 // Skip suite encoding for now
	}
	
	buf[offset] = byte(len(m.SessionID))
	offset++
	if len(m.SessionID) > 0 {
		copy(buf[offset:], m.SessionID)
		offset += len(m.SessionID)
	}
	
	// Extensions
	binary.BigEndian.PutUint16(buf[offset:], uint16(len(m.Extensions)))
	offset += 2
	
	for _, ext := range m.Extensions {
		binary.BigEndian.PutUint16(buf[offset:], ext.Type)
		offset += 2
		binary.BigEndian.PutUint16(buf[offset:], uint16(len(ext.Data)))
		offset += 2
		copy(buf[offset:], ext.Data)
		offset += len(ext.Data)
	}
	
	return buf[:offset]
}

type serverHelloMsg struct {
	Version       uint8
	Random        []byte
	Suite         crypto.Suite
	SessionID     []byte
	ServerKEMPK   kem.PublicKey
	ServerCert    []byte
	ServerSig     []byte
	Cookie        []byte
	Extensions    []extension
}

func (m *serverHelloMsg) Unmarshal(data []byte) error {
	if len(data) < 34 {
		return errors.New("ServerHello too short")
	}
	
	offset := 0
	m.Version = data[offset]
	offset++
	
	m.Random = make([]byte, 32)
	copy(m.Random, data[offset:])
	offset += 32
	
	// Parse suite (simplified)
	offset += 4
	
	// Parse session ID
	sessionIDLen := int(data[offset])
	offset++
	if sessionIDLen > 0 {
		m.SessionID = make([]byte, sessionIDLen)
		copy(m.SessionID, data[offset:])
		offset += sessionIDLen
	}
	
	// TODO: Parse remaining fields
	
	return nil
}

type clientKeyMsg struct {
	KEMCiphertext []byte
	PSKBinder     []byte
	ClientSig     []byte
}

func (m *clientKeyMsg) Marshal() []byte {
	size := 2 + len(m.KEMCiphertext)
	if len(m.PSKBinder) > 0 {
		size += 2 + len(m.PSKBinder)
	}
	if len(m.ClientSig) > 0 {
		size += 2 + len(m.ClientSig)
	}
	
	buf := make([]byte, size)
	offset := 0
	
	binary.BigEndian.PutUint16(buf[offset:], uint16(len(m.KEMCiphertext)))
	offset += 2
	copy(buf[offset:], m.KEMCiphertext)
	offset += len(m.KEMCiphertext)
	
	// TODO: Marshal remaining fields
	
	return buf[:offset]
}

type extension struct {
	Type uint16
	Data []byte
}

// Extension types
const (
	extKeyShare    uint16 = 0x0033
	extPQOnly      uint16 = 0xFF01
	extPSK         uint16 = 0x0029
	extEarlyData   uint16 = 0x002A
)