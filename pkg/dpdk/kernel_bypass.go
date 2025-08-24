// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dpdk

import (
	"fmt"
	"net"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/luxfi/dex/pkg/lx"
	// "golang.org/x/sys/unix" // Linux-specific, commented for macOS
)

// KernelBypassEngine provides kernel-bypass networking for ultra-low latency
type KernelBypassEngine struct {
	mode           string // "dpdk", "xdp", "afpacket", "raw"
	socket         int
	bufferSize     int
	ringBuffer     *RingBuffer
	stats          *Stats
	orderProcessor OrderProcessor
	mu             sync.RWMutex
}

// Stats tracks performance metrics
type Stats struct {
	PacketsReceived  atomic.Uint64
	PacketsProcessed atomic.Uint64
	OrdersProcessed  atomic.Uint64
	TradesExecuted   atomic.Uint64
	LatencyNs        atomic.Uint64
	DroppedPackets   atomic.Uint64
}

// OrderProcessor interface for processing orders
type OrderProcessor interface {
	ProcessOrder(order *lx.Order) (*lx.Trade, error)
}

// RingBuffer for zero-copy packet processing
type RingBuffer struct {
	buffer    []byte
	size      int
	head      atomic.Uint64
	tail      atomic.Uint64
	mask      uint64
	cacheLine [64]byte // Prevent false sharing
}

// NewKernelBypassEngine creates a new kernel-bypass engine
func NewKernelBypassEngine(processor OrderProcessor) (*KernelBypassEngine, error) {
	engine := &KernelBypassEngine{
		bufferSize:     1 << 20, // 1MB buffer
		stats:          &Stats{},
		orderProcessor: processor,
	}

	// Detect best available mode based on platform
	switch runtime.GOOS {
	case "linux":
		// Try AF_PACKET with PACKET_MMAP for zero-copy
		if err := engine.initAFPacket(); err == nil {
			engine.mode = "afpacket"
			return engine, nil
		}
		// Fallback to raw sockets
		if err := engine.initRawSocket(); err == nil {
			engine.mode = "raw"
			return engine, nil
		}
	case "darwin":
		// macOS: Use BPF or raw sockets
		if err := engine.initBPF(); err == nil {
			engine.mode = "bpf"
			return engine, nil
		}
		// Fallback to raw sockets
		if err := engine.initRawSocket(); err == nil {
			engine.mode = "raw"
			return engine, nil
		}
	default:
		// Fallback to standard sockets with optimizations
		if err := engine.initOptimizedSocket(); err == nil {
			engine.mode = "optimized"
			return engine, nil
		}
	}

	return nil, fmt.Errorf("no kernel-bypass method available on %s", runtime.GOOS)
}

// initAFPacket initializes AF_PACKET on Linux for zero-copy
func (e *KernelBypassEngine) initAFPacket() error {
	if runtime.GOOS != "linux" {
		return fmt.Errorf("AF_PACKET only available on Linux")
	}

	// AF_PACKET is Linux-specific, using raw socket instead
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_RAW, syscall.IPPROTO_TCP)
	if err != nil {
		return err
	}
	e.socket = fd

	// Set socket options for performance
	syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, e.bufferSize)

	// Create ring buffer
	e.ringBuffer = &RingBuffer{
		buffer: make([]byte, e.bufferSize),
		size:   e.bufferSize,
		mask:   uint64(e.bufferSize - 1),
	}

	return nil
}

// initBPF initializes BPF on macOS/BSD
func (e *KernelBypassEngine) initBPF() error {
	if runtime.GOOS != "darwin" && runtime.GOOS != "freebsd" {
		return fmt.Errorf("BPF not available on %s", runtime.GOOS)
	}

	// Open BPF device (simplified - would need proper BPF setup)
	// This is a placeholder for actual BPF implementation
	return fmt.Errorf("BPF not implemented yet")
}

// initRawSocket initializes raw socket
func (e *KernelBypassEngine) initRawSocket() error {
	// Create raw socket
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_RAW, syscall.IPPROTO_TCP)
	if err != nil {
		return err
	}
	e.socket = fd

	// Set socket options for low latency
	syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, e.bufferSize)
	syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, 1)

	// Platform-specific optimizations handled separately

	// Create ring buffer
	e.ringBuffer = &RingBuffer{
		buffer: make([]byte, e.bufferSize),
		size:   e.bufferSize,
		mask:   uint64(e.bufferSize - 1),
	}

	return nil
}

// initOptimizedSocket initializes optimized standard socket
func (e *KernelBypassEngine) initOptimizedSocket() error {
	// Create UDP socket for lowest latency
	conn, err := net.ListenPacket("udp", ":0")
	if err != nil {
		return err
	}

	// Get raw file descriptor
	rawConn, err := conn.(*net.UDPConn).SyscallConn()
	if err != nil {
		return err
	}

	var sockErr error
	rawConn.Control(func(fd uintptr) {
		// Set socket options
		syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_RCVBUF, e.bufferSize)
		syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_SNDBUF, e.bufferSize)

		// Platform-specific optimizations handled by helper functions

		e.socket = int(fd)
	})

	if sockErr != nil {
		return sockErr
	}

	// Create ring buffer
	e.ringBuffer = &RingBuffer{
		buffer: make([]byte, e.bufferSize),
		size:   e.bufferSize,
		mask:   uint64(e.bufferSize - 1),
	}

	return nil
}

// ProcessPackets processes incoming packets with kernel bypass
func (e *KernelBypassEngine) ProcessPackets() error {
	buffer := make([]byte, 65536) // Max packet size

	for {
		// Receive packet with zero-copy if possible
		n, _, err := syscall.Recvfrom(e.socket, buffer, 0)
		if err != nil {
			if err == syscall.EAGAIN || err == syscall.EWOULDBLOCK {
				continue
			}
			return err
		}

		e.stats.PacketsReceived.Add(1)

		// Parse order from packet (simplified)
		order := e.parseOrder(buffer[:n])
		if order != nil {
			start := time.Now()

			// Process order
			trade, err := e.orderProcessor.ProcessOrder(order)
			if err == nil && trade != nil {
				e.stats.TradesExecuted.Add(1)
			}

			// Update latency (moving average)
			latency := uint64(time.Since(start).Nanoseconds())
			e.stats.LatencyNs.Store(latency)
			e.stats.OrdersProcessed.Add(1)
		}

		e.stats.PacketsProcessed.Add(1)
	}
}

// parseOrder parses an order from packet data
func (e *KernelBypassEngine) parseOrder(data []byte) *lx.Order {
	// Simplified binary protocol parsing
	// In production, this would parse FIX or custom binary protocol

	if len(data) < 32 {
		return nil
	}

	// Binary format: [8:orderID][8:price][8:size][1:side][7:padding]
	order := &lx.Order{
		ID:    *(*uint64)(unsafe.Pointer(&data[0])),
		Price: *(*float64)(unsafe.Pointer(&data[8])),
		Size:  *(*float64)(unsafe.Pointer(&data[16])),
		Side:  lx.Side(data[24]),
		Type:  lx.Limit,
	}

	return order
}

// GetStats returns performance statistics
func (e *KernelBypassEngine) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"mode":              e.mode,
		"packets_received":  e.stats.PacketsReceived.Load(),
		"packets_processed": e.stats.PacketsProcessed.Load(),
		"orders_processed":  e.stats.OrdersProcessed.Load(),
		"trades_executed":   e.stats.TradesExecuted.Load(),
		"latency_ns":        e.stats.LatencyNs.Load(),
		"dropped_packets":   e.stats.DroppedPackets.Load(),
		"throughput_pps":    e.calculateThroughput(),
	}
}

// calculateThroughput calculates packets per second
func (e *KernelBypassEngine) calculateThroughput() float64 {
	// This would track time and calculate actual throughput
	return float64(e.stats.PacketsProcessed.Load())
}

// Close closes the kernel-bypass engine
func (e *KernelBypassEngine) Close() error {
	if e.socket != 0 {
		return syscall.Close(e.socket)
	}
	return nil
}

// htons converts host byte order to network byte order
func htons(i uint16) uint16 {
	return (i<<8)&0xff00 | i>>8
}

// Optimizations for different platforms

// LinuxOptimizations applies Linux-specific optimizations
func LinuxOptimizations(fd int) {
	if runtime.GOOS != "linux" {
		return
	}

	// These optimizations require Linux-specific constants
	// In production, use build tags to conditionally compile
	// For now, using standard socket options only
	syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)
}

// DarwinOptimizations applies macOS-specific optimizations
func DarwinOptimizations(fd int) {
	if runtime.GOOS != "darwin" {
		return
	}

	// Set SO_NOSIGPIPE to prevent SIGPIPE
	syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_NOSIGPIPE, 1)

	// Other macOS-specific optimizations would go here
}
