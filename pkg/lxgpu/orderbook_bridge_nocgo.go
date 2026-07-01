// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

// dexvm_gpu_nocgo.go — nocgo stub for the GPU plugin bridge.
//
// When CGO_ENABLED=0 the dlopen path in dexvm_gpu.go is unreachable,
// so every public method returns ErrGPUNotAvailable and AutoBackend()
// reports GPUBackendNone. The package surface is identical to the
// cgo build — consumers compile unconditionally and branch on
// AutoBackend() at runtime.

package lxgpu

// AutoBackend reports GPUBackendNone under !cgo — dlopen is unreachable.
func AutoBackend() GPUBackend { return GPUBackendNone }

// GPUPluginPath returns "" under !cgo — no plugin was loaded.
func GPUPluginPath() string { return "" }

// OrderBookArena is a zero-sized opaque type so the package surface
// compiles uniformly under both build tags. The nocgo build never
// returns a non-nil arena from ArenaCreate.
type OrderBookArena struct{}

// ArenaCreate returns ErrGPUNotAvailable under !cgo.
func ArenaCreate() (*OrderBookArena, error) {
	return nil, ErrGPUNotAvailable
}

// ArenaDestroy returns ErrGPUNotAvailable under !cgo. A nil arena is
// still a no-op for symmetry with the cgo build.
func ArenaDestroy(a *OrderBookArena) error {
	if a == nil {
		return nil
	}
	return ErrGPUNotAvailable
}

// OrderBookMatch returns ErrGPUNotAvailable under !cgo.
func OrderBookMatch(a *OrderBookArena, calldata []byte) (out [OrderBookOutLen]byte, numFills uint32, err error) {
	_ = a
	_ = calldata
	return out, 0, ErrGPUNotAvailable
}
