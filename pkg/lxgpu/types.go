// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package lxgpu is the dlopen bridge to the gpu-kernels DEX plugin and the
// byte-parity proof harness for the native OrderBook matcher.
//
// It is the SINGLE home for the GPU dispatch + its KAT (known-answer test)
// against the deterministic CPU oracle. The CPU oracle is the byte-parity
// source of truth (a Go port of the canonical C++ oracle
// gpu-kernels/tools/kat/dex_cpu_oracle.hpp); every GPU backend
// (cuda/hip/metal/vulkan/webgpu) must reproduce it byte-for-byte. The matcher
// proven here is the d-chain's matcher — it lives only in luxfi/dex, never in
// the chains/dexvm proxy.
//
// The plugin (luxcpp + lux-private/gpu-kernels) is NEVER a Go build dependency:
// it is a runtime .dylib/.so dlopen'd at process start. A host without it
// reports GPUBackendNone and every method returns ErrGPUNotAvailable — callers
// fall back to the CPU path.
package lxgpu

import "errors"

// GPUBackend names the GPU backend the plugin loader bound. The numeric values
// match the dlopen probe order used by init() — callers can rely on
// (GPUBackendCUDA < GPUBackendHIP < GPUBackendMetal < GPUBackendVulkan <
// GPUBackendWebGPU) as a stable ordering of preference.
type GPUBackend uint8

const (
	// GPUBackendNone means no plugin was loaded (or build was !cgo).
	GPUBackendNone GPUBackend = iota
	// GPUBackendCUDA is the NVIDIA CUDA plugin (libluxgpu_backend_cuda.so).
	GPUBackendCUDA
	// GPUBackendHIP is the AMD ROCm/HIP plugin (libluxgpu_backend_hip.so).
	GPUBackendHIP
	// GPUBackendMetal is the Apple Metal plugin (libluxgpu_backend_metal.dylib).
	GPUBackendMetal
	// GPUBackendVulkan is the Vulkan plugin (libluxgpu_backend_vulkan.{so,dylib,dll}).
	GPUBackendVulkan
	// GPUBackendWebGPU is the wgpu/Dawn plugin (libluxgpu_backend_webgpu.{so,dylib,dll}).
	GPUBackendWebGPU
)

// String returns the human-readable lowercase backend name used in the host
// launcher symbol prefix (e.g. "metal" -> "lux_metal_dex_orderbook_match").
func (b GPUBackend) String() string {
	switch b {
	case GPUBackendCUDA:
		return "cuda"
	case GPUBackendHIP:
		return "hip"
	case GPUBackendMetal:
		return "metal"
	case GPUBackendVulkan:
		return "vulkan"
	case GPUBackendWebGPU:
		return "webgpu"
	default:
		return "none"
	}
}

// ErrGPUNotAvailable is returned by every GPU method on the nocgo build, and by
// the cgo build when init() failed to bind any plugin. Consumers treat this as
// "fall back to the CPU path" rather than as a hard failure.
var ErrGPUNotAvailable = errors.New("lxgpu: GPU plugin not available")

// OrderBook ABI byte sizes from include/lux/gpu/dex.h.
const (
	// OrderBookCalldataLen is the on-wire calldata for EVM precompile 0x100:
	// side(1) | price(32 BE) | qty(32 BE) | user(20) | book_id(32) = 117.
	OrderBookCalldataLen = 117
	// OrderBookOutLen is the output: filled(32 BE) | avg_price(32 BE) | num_fills(4 BE) = 68.
	OrderBookOutLen = 68
	// OrderBookMaxLevels caps the per-side resting-level depth (bid_*/ask_* arrays).
	OrderBookMaxLevels = 1024
	// OrderBookBookIDOffset is where book_id starts inside calldata.
	OrderBookBookIDOffset = 85
	// OrderBookBookIDLen is the book_id width.
	OrderBookBookIDLen = 32
)
