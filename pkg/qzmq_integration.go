// Package dex integrates QZMQ for quantum-secure DEX operations
package dex

import (
	"fmt"
	"log"
	"time"

	"github.com/luxfi/qzmq"
)

// QZMQDEXServer represents a quantum-secure DEX server
type QZMQDEXServer struct {
	transport *qzmq.XChainDEXTransport
	config    *QZMQConfig
}

// QZMQConfig holds QZMQ configuration for the DEX
type QZMQConfig struct {
	NodeAPI     string
	ChainID     string
	ValidatorID string
	
	// Ports
	ConsensusPort  int
	OrderBookPort  int
	MatchingPort   int
	SettlementPort int
	MarketDataPort int
	
	// Security
	EnableQuantum bool
	KeyRotation   time.Duration
}

// NewQZMQDEXServer creates a new quantum-secure DEX server
func NewQZMQDEXServer(config *QZMQConfig) (*QZMQDEXServer, error) {
	// Load validator keys
	validatorKeys := &qzmq.ValidatorKeys{
		NodeID:  loadValidatorKey("node"),
		BLSKey:  loadValidatorKey("bls"),
		PQKey:   loadValidatorKey("pq"),
		DSAKey:  loadValidatorKey("dsa"),
	}
	
	// Create X-Chain DEX transport
	transport, err := qzmq.NewXChainDEXTransport(
		config.NodeAPI,
		config.ChainID,
		config.ValidatorID,
		validatorKeys,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create QZMQ transport: %w", err)
	}
	
	return &QZMQDEXServer{
		transport: transport,
		config:    config,
	}, nil
}

// Start starts the quantum-secure DEX server
func (s *QZMQDEXServer) Start() error {
	// Start consensus layer
	if err := s.transport.Listen(
		s.config.ConsensusPort,
		s.config.ConsensusPort+1,
		s.config.ConsensusPort+2,
	); err != nil {
		return fmt.Errorf("failed to start consensus: %w", err)
	}
	
	// Start DEX services
	if err := s.transport.StartDEX(
		s.config.OrderBookPort,
		s.config.MatchingPort,
		s.config.SettlementPort,
		s.config.MarketDataPort,
	); err != nil {
		return fmt.Errorf("failed to start DEX: %w", err)
	}
	
	log.Printf("QZMQ DEX Server started with quantum security")
	log.Printf("Consensus on ports %d-%d", s.config.ConsensusPort, s.config.ConsensusPort+2)
	log.Printf("DEX services on ports %d-%d", s.config.OrderBookPort, s.config.MarketDataPort)
	
	return nil
}

// SubmitOrder submits a quantum-signed order
func (s *QZMQDEXServer) SubmitOrder(order *qzmq.Order) error {
	return s.transport.SubmitOrder(order)
}

// ConnectToPeers connects to other validators
func (s *QZMQDEXServer) ConnectToPeers(peers []string) error {
	return s.transport.ConnectToPeers(peers)
}

// GetMetrics returns server metrics
func (s *QZMQDEXServer) GetMetrics() *qzmq.ConsensusMetrics {
	return s.transport.GetMetrics()
}

// Close shuts down the server
func (s *QZMQDEXServer) Close() error {
	return s.transport.Close()
}

// Helper function to load keys (integrates with Lux key management)
func loadValidatorKey(keyType string) []byte {
	// In production, load from Lux key store
	// For now, return dummy keys
	switch keyType {
	case "node":
		return make([]byte, 32) // Ed25519
	case "bls":
		return make([]byte, 32) // BLS secret
	case "pq":
		return make([]byte, 2400) // ML-KEM-768
	case "dsa":
		return make([]byte, 4032) // ML-DSA-87
	default:
		return nil
	}
}

// DefaultQZMQConfig returns default QZMQ configuration
func DefaultQZMQConfig() *QZMQConfig {
	return &QZMQConfig{
		NodeAPI:        "http://localhost:9650",
		ChainID:        "X",
		ValidatorID:    "NodeID-7Xhw2mDxuDS44j42TCB6U5579esbSt3Lg",
		ConsensusPort:  5000,
		OrderBookPort:  6000,
		MatchingPort:   6001,
		SettlementPort: 6002,
		MarketDataPort: 6003,
		EnableQuantum:  true,
		KeyRotation:    5 * time.Minute,
	}
}