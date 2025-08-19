package qzmq

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	zmq "github.com/pebbe/zmq4"
)

// ZMQSocket wraps a ZeroMQ socket with QZMQ security
type ZMQSocket struct {
	socket     *zmq.Socket
	connection *Connection
	opts       Options
	mu         sync.RWMutex
	
	// Message tracking
	sendSeq    uint64
	recvSeq    uint64
	
	// Performance metrics
	latencies  []time.Duration
	throughput uint64
}

// NewZMQSocket creates a QZMQ-secured ZeroMQ socket
func NewZMQSocket(socketType zmq.Type, opts Options) (*ZMQSocket, error) {
	socket, err := zmq.NewSocket(socketType)
	if err != nil {
		return nil, err
	}
	
	return &ZMQSocket{
		socket:    socket,
		opts:      opts,
		latencies: make([]time.Duration, 0, 1000),
	}, nil
}

// ConfigureQZMQ sets up QZMQ mechanism on the socket
func (s *ZMQSocket) ConfigureQZMQ(isServer bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	// Create QZMQ connection
	s.connection = NewConnection(s.opts, isServer)
	
	// Set socket options for QZMQ
	if err := s.socket.SetLinger(0); err != nil {
		return err
	}
	
	// Set high water marks
	if err := s.socket.SetSndhwm(10000); err != nil {
		return err
	}
	if err := s.socket.SetRcvhwm(10000); err != nil {
		return err
	}
	
	// Enable TCP keepalive for reliability
	if err := s.socket.SetTcpKeepalive(1); err != nil {
		return err
	}
	if err := s.socket.SetTcpKeepaliveIdle(120); err != nil {
		return err
	}
	
	// Set socket identity for routing
	identity := fmt.Sprintf("qzmq-%d-%s", time.Now().Unix(), s.opts.Suite.String())
	if err := s.socket.SetIdentity(identity); err != nil {
		return err
	}
	
	return nil
}

// Connect performs QZMQ handshake and connects to endpoint
func (s *ZMQSocket) Connect(endpoint string) error {
	// First connect the underlying socket
	if err := s.socket.Connect(endpoint); err != nil {
		return err
	}
	
	// Perform QZMQ handshake
	if err := s.performClientHandshake(); err != nil {
		s.socket.Disconnect(endpoint)
		return fmt.Errorf("QZMQ handshake failed: %w", err)
	}
	
	return nil
}

// Bind binds the socket and prepares for QZMQ handshakes
func (s *ZMQSocket) Bind(endpoint string) error {
	if err := s.socket.Bind(endpoint); err != nil {
		return err
	}
	
	// Server waits for client handshakes
	go s.acceptHandshakes()
	
	return nil
}

// SendSecure sends encrypted message via QZMQ
func (s *ZMQSocket) SendSecure(data []byte, flags zmq.Flag) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.connection == nil || s.connection.state != StateEstablished {
		return ErrNotEstablished
	}
	
	// Check if key update needed
	if s.connection.NeedsKeyUpdate() {
		if err := s.performKeyUpdate(); err != nil {
			return err
		}
	}
	
	// Track send time for latency measurement
	sendTime := time.Now()
	
	// Encrypt message
	s.sendSeq++
	nonce := make([]byte, NonceSize)
	binary.BigEndian.PutUint32(nonce[0:4], s.connection.streamID)
	binary.BigEndian.PutUint64(nonce[4:12], s.sendSeq)
	
	// Add authenticated data (message type, timestamp)
	aad := make([]byte, 16)
	binary.BigEndian.PutUint64(aad[0:8], uint64(sendTime.Unix()))
	binary.BigEndian.PutUint64(aad[8:16], s.sendSeq)
	
	// Encrypt with AEAD
	ciphertext := s.connection.clientAEAD.Seal(nil, nonce, data, aad)
	
	// Create QZMQ message frame
	frame := &MessageFrame{
		Version:    ProtocolVersion,
		StreamID:   s.connection.streamID,
		SequenceNo: s.sendSeq,
		Flags:      0,
		Payload:    ciphertext,
		AAD:        aad,
	}
	
	// Send encrypted frame
	frameData := frame.Marshal()
	if err := s.socket.SendBytes(frameData, flags); err != nil {
		return err
	}
	
	// Update metrics
	s.connection.metrics.MessagesSent++
	s.connection.metrics.BytesSent += uint64(len(frameData))
	s.throughput++
	
	return nil
}

// RecvSecure receives and decrypts message via QZMQ
func (s *ZMQSocket) RecvSecure(flags zmq.Flag) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.connection == nil || s.connection.state != StateEstablished {
		return nil, ErrNotEstablished
	}
	
	// Receive encrypted frame
	frameData, err := s.socket.RecvBytes(flags)
	if err != nil {
		return nil, err
	}
	
	recvTime := time.Now()
	
	// Parse QZMQ frame
	frame := &MessageFrame{}
	if err := frame.Unmarshal(frameData); err != nil {
		return nil, err
	}
	
	// Verify sequence number (detect replay)
	if frame.SequenceNo <= s.recvSeq {
		return nil, ErrReplayDetected
	}
	s.recvSeq = frame.SequenceNo
	
	// Construct nonce
	nonce := make([]byte, NonceSize)
	binary.BigEndian.PutUint32(nonce[0:4], frame.StreamID)
	binary.BigEndian.PutUint64(nonce[4:12], frame.SequenceNo)
	
	// Decrypt with AEAD
	plaintext, err := s.connection.serverAEAD.Open(nil, nonce, frame.Payload, frame.AAD)
	if err != nil {
		s.connection.metrics.Errors++
		return nil, ErrAuthenticationFailed
	}
	
	// Update metrics
	s.connection.metrics.MessagesReceived++
	s.connection.metrics.BytesReceived += uint64(len(frameData))
	
	// Calculate latency if timestamp in AAD
	if len(frame.AAD) >= 8 {
		sendTimestamp := binary.BigEndian.Uint64(frame.AAD[0:8])
		sendTime := time.Unix(int64(sendTimestamp), 0)
		latency := recvTime.Sub(sendTime)
		s.latencies = append(s.latencies, latency)
	}
	
	return plaintext, nil
}

// Send0RTT sends early data if 0-RTT is enabled
func (s *ZMQSocket) Send0RTT(data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if !s.opts.ZeroRTT || s.connection.psk == nil {
		return ErrNo0RTT
	}
	
	if len(data) > int(s.opts.MaxEarlyData) {
		return ErrTooMuchEarlyData
	}
	
	// Encrypt with PSK-derived key
	earlyKey := s.connection.deriveEarlyDataKey()
	
	// Create early data AEAD
	earlyAEAD, err := s.connection.createAEAD(earlyKey)
	if err != nil {
		return err
	}
	
	// Encrypt early data
	nonce := make([]byte, NonceSize)
	binary.BigEndian.PutUint32(nonce[0:4], EarlyDataStreamID)
	
	ciphertext := earlyAEAD.Seal(nil, nonce, data, nil)
	
	// Send as 0-RTT frame
	frame := &EarlyDataFrame{
		Version:   ProtocolVersion,
		SessionID: s.connection.sessionID,
		Payload:   ciphertext,
	}
	
	return s.socket.SendBytes(frame.Marshal(), 0)
}

// performClientHandshake executes client-side QZMQ handshake
func (s *ZMQSocket) performClientHandshake() error {
	// Step 1: Send ClientHello
	clientHello, err := s.connection.ClientHello()
	if err != nil {
		return err
	}
	
	if err := s.socket.SendBytes(clientHello, 0); err != nil {
		return err
	}
	
	// Step 2: Receive ServerHello
	serverHello, err := s.socket.RecvBytes(0)
	if err != nil {
		return err
	}
	
	if err := s.connection.ProcessServerHello(serverHello); err != nil {
		return err
	}
	
	// Step 3: Send ClientKey
	clientKey, err := s.connection.ClientKey()
	if err != nil {
		return err
	}
	
	if err := s.socket.SendBytes(clientKey, 0); err != nil {
		return err
	}
	
	// Step 4: Receive Finished
	finished, err := s.socket.RecvBytes(0)
	if err != nil {
		return err
	}
	
	if err := s.connection.ProcessFinished(finished); err != nil {
		return err
	}
	
	return nil
}

// acceptHandshakes handles incoming QZMQ handshakes (server)
func (s *ZMQSocket) acceptHandshakes() {
	// This would be implemented for server-side handshake handling
	// For now, placeholder for server acceptance logic
}

// performKeyUpdate executes key update protocol
func (s *ZMQSocket) performKeyUpdate() error {
	// Generate new key material
	newKeys, err := s.connection.UpdateKeys()
	if err != nil {
		return err
	}
	
	// Create key update message
	updateMsg := &KeyUpdateMessage{
		Timestamp: time.Now().Unix(),
		NewKeyID:  s.connection.metrics.KeyUpdates + 1,
	}
	
	// Send encrypted with current key
	if err := s.SendSecure(updateMsg.Marshal(), 0); err != nil {
		return err
	}
	
	// Switch to new keys
	s.connection.keys = newKeys
	s.connection.metrics.KeyUpdates++
	s.connection.lastKeyUpdate = time.Now()
	
	// Reset message counters
	s.connection.metrics.MessagesSent = 0
	s.connection.metrics.BytesSent = 0
	
	return nil
}

// GetMetrics returns performance metrics
func (s *ZMQSocket) GetMetrics() SocketMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	var avgLatency time.Duration
	if len(s.latencies) > 0 {
		var total time.Duration
		for _, lat := range s.latencies {
			total += lat
		}
		avgLatency = total / time.Duration(len(s.latencies))
	}
	
	return SocketMetrics{
		ConnectionMetrics: s.connection.GetMetrics(),
		AverageLatency:    avgLatency,
		Throughput:        s.throughput,
		LatencySamples:    len(s.latencies),
	}
}

// Close closes the QZMQ socket
func (s *ZMQSocket) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.connection != nil && s.connection.state == StateEstablished {
		// Send close notify
		closeMsg := &CloseMessage{
			Reason: "normal_closure",
		}
		_ = s.SendSecure(closeMsg.Marshal(), 0)
		
		s.connection.state = StateClosed
	}
	
	return s.socket.Close()
}

// Message frame structures

type MessageFrame struct {
	Version    uint8
	StreamID   uint32
	SequenceNo uint64
	Flags      uint8
	Payload    []byte
	AAD        []byte
}

func (f *MessageFrame) Marshal() []byte {
	size := 1 + 4 + 8 + 1 + 4 + len(f.Payload) + 2 + len(f.AAD)
	buf := make([]byte, size)
	offset := 0
	
	buf[offset] = f.Version
	offset++
	
	binary.BigEndian.PutUint32(buf[offset:], f.StreamID)
	offset += 4
	
	binary.BigEndian.PutUint64(buf[offset:], f.SequenceNo)
	offset += 8
	
	buf[offset] = f.Flags
	offset++
	
	binary.BigEndian.PutUint32(buf[offset:], uint32(len(f.Payload)))
	offset += 4
	copy(buf[offset:], f.Payload)
	offset += len(f.Payload)
	
	binary.BigEndian.PutUint16(buf[offset:], uint16(len(f.AAD)))
	offset += 2
	copy(buf[offset:], f.AAD)
	
	return buf
}

func (f *MessageFrame) Unmarshal(data []byte) error {
	if len(data) < 18 {
		return ErrInvalidMessage
	}
	
	offset := 0
	f.Version = data[offset]
	offset++
	
	f.StreamID = binary.BigEndian.Uint32(data[offset:])
	offset += 4
	
	f.SequenceNo = binary.BigEndian.Uint64(data[offset:])
	offset += 8
	
	f.Flags = data[offset]
	offset++
	
	payloadLen := binary.BigEndian.Uint32(data[offset:])
	offset += 4
	
	if len(data) < offset+int(payloadLen)+2 {
		return ErrInvalidMessage
	}
	
	f.Payload = make([]byte, payloadLen)
	copy(f.Payload, data[offset:])
	offset += int(payloadLen)
	
	aadLen := binary.BigEndian.Uint16(data[offset:])
	offset += 2
	
	if len(data) < offset+int(aadLen) {
		return ErrInvalidMessage
	}
	
	f.AAD = make([]byte, aadLen)
	copy(f.AAD, data[offset:])
	
	return nil
}

type EarlyDataFrame struct {
	Version   uint8
	SessionID []byte
	Payload   []byte
}

func (f *EarlyDataFrame) Marshal() []byte {
	size := 1 + 1 + len(f.SessionID) + 4 + len(f.Payload)
	buf := make([]byte, size)
	offset := 0
	
	buf[offset] = f.Version
	offset++
	
	buf[offset] = byte(len(f.SessionID))
	offset++
	copy(buf[offset:], f.SessionID)
	offset += len(f.SessionID)
	
	binary.BigEndian.PutUint32(buf[offset:], uint32(len(f.Payload)))
	offset += 4
	copy(buf[offset:], f.Payload)
	
	return buf
}

type KeyUpdateMessage struct {
	Timestamp int64
	NewKeyID  uint64
}

func (m *KeyUpdateMessage) Marshal() []byte {
	buf := make([]byte, 16)
	binary.BigEndian.PutUint64(buf[0:8], uint64(m.Timestamp))
	binary.BigEndian.PutUint64(buf[8:16], m.NewKeyID)
	return buf
}

type CloseMessage struct {
	Reason string
}

func (m *CloseMessage) Marshal() []byte {
	return []byte(m.Reason)
}

// SocketMetrics contains ZMQ socket performance metrics
type SocketMetrics struct {
	ConnectionMetrics Metrics
	AverageLatency    time.Duration
	Throughput        uint64
	LatencySamples    int
}

// Constants for special stream IDs
const (
	EarlyDataStreamID uint32 = 0xFFFFFFFF
)