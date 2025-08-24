//go:build fpga
// +build fpga

package fpga

import (
	"errors"
)

// NewIntelStratixAccelerator creates a new Intel Stratix accelerator (stub)
func NewIntelStratixAccelerator() FPGAAccelerator {
	return &stubAccelerator{name: "Intel Stratix"}
}

// NewXilinxAlveoAccelerator creates a new Xilinx Alveo accelerator (stub)
func NewXilinxAlveoAccelerator() FPGAAccelerator {
	return &stubAccelerator{name: "Xilinx Alveo"}
}

// stubAccelerator is a stub implementation for unsupported FPGAs
type stubAccelerator struct {
	name string
}

func (s *stubAccelerator) Initialize(config *FPGAConfig) error {
	return errors.New(s.name + " accelerator not implemented")
}

func (s *stubAccelerator) Shutdown() error {
	return nil
}

func (s *stubAccelerator) Reset() error {
	return nil
}

func (s *stubAccelerator) IsHealthy() bool {
	return false
}

func (s *stubAccelerator) GetStats() *FPGAStats {
	return &FPGAStats{}
}

func (s *stubAccelerator) GetTemperature() float64 {
	return 0
}

func (s *stubAccelerator) GetPowerUsage() float64 {
	return 0
}

func (s *stubAccelerator) ProcessOrder(order *FPGAOrder) (*FPGAResult, error) {
	return nil, errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) BatchProcessOrders(orders []*FPGAOrder) ([]*FPGAResult, error) {
	return nil, errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) CancelOrder(orderID uint64) error {
	return errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) UpdateOrderBook(symbol string, bids, asks []PriceLevel) error {
	return errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) GetOrderBook(symbol string) (*OrderBookSnapshot, error) {
	return nil, errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) LoadBitstream(path string) error {
	return errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) ReconfigurePartial(region int, bitstream []byte) error {
	return errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) SetClockFrequency(mhz int) error {
	return errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) AllocateDMABuffer(size int) (uintptr, error) {
	return 0, errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) FreeDMABuffer(addr uintptr) error {
	return errors.New(s.name + " not implemented")
}

func (s *stubAccelerator) DMATransfer(src, dst uintptr, size int) error {
	return errors.New(s.name + " not implemented")
}
