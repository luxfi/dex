// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && darwin

package lx

// GPU-accelerated batch xy=k AMM evaluation via Metal.
//
// The Metal kernel and Objective-C++ driver live in luxcpp/dex/gpu/metal:
//   kernel: amm_xyk.metal           -> built by cmake into amm_xyk.metallib
//   driver: amm_xyk_driver.{h,mm}   -> built into libamm_xyk_metal.a
//
// One-time build:
//   cmake -S ~/work/luxcpp/dex -B ~/work/luxcpp/dex/build && \
//     cmake --build ~/work/luxcpp/dex/build --target amm_xyk_metal
//
// The metallib's runtime path is read from $LUX_DEX_AMM_METALLIB; if unset
// we fall back to the canonical build-tree location.

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../luxcpp/dex/gpu/metal

#cgo LDFLAGS: -L${SRCDIR}/../../../../luxcpp/dex/build
#cgo LDFLAGS: -lamm_xyk_metal -lstdc++
#cgo LDFLAGS: -framework Metal -framework Foundation

#include <stdint.h>
#include <stdlib.h>
#include "amm_xyk_driver.h"
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

// resolveMetallibPath returns the absolute path to the precompiled
// amm_xyk.metallib. We honour LUX_DEX_AMM_METALLIB so packagers can
// relocate the file at install time, but otherwise we fall back to the
// in-tree cmake build path so dev builds Just Work.
var (
	metallibOnce sync.Once
	metallibPath string
)

func resolveMetallibPath() string {
	metallibOnce.Do(func() {
		if p := os.Getenv("LUX_DEX_AMM_METALLIB"); p != "" {
			metallibPath = p
			return
		}
		// Default: ~/work/luxcpp/dex/build/amm_xyk.metallib. We can't
		// rely on ${SRCDIR} at runtime (cgo only expands it at compile
		// time), so we hardcode the canonical sibling layout that the
		// repo enforces: lx and luxcpp are sibling directories under
		// $LUX_WORKSPACE_ROOT (default ~/work).
		root := os.Getenv("LUX_WORKSPACE_ROOT")
		if root == "" {
			home, err := os.UserHomeDir()
			if err == nil {
				root = filepath.Join(home, "work")
			}
		}
		metallibPath = filepath.Join(root, "luxcpp", "dex", "build", "amm_xyk.metallib")
	})
	return metallibPath
}

// BatchEvalConstantProduct dispatches an N-tuple xy=k batch onto the Metal
// GPU and returns the per-tuple `out_amount`. Output is byte-identical to
// the CPU oracle BatchEvalConstantProductCPU — that contract is enforced by
// amm_gpu_test.go on every run.
//
// On any non-Apple build the cgo file is excluded by the `darwin` build tag
// and the nocgo fallback below answers via the CPU oracle.
func BatchEvalConstantProduct(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	if len(reserves) != len(amounts) {
		return nil, fmt.Errorf("lx: reserves/amounts length mismatch: %d vs %d",
			len(reserves), len(amounts))
	}
	n := len(reserves)
	if n == 0 {
		return nil, nil
	}

	libPath := resolveMetallibPath()
	if _, err := os.Stat(libPath); err != nil {
		return nil, fmt.Errorf("lx: amm metallib not found at %q: %w "+
			"(set LUX_DEX_AMM_METALLIB or run cmake --build "+
			"~/work/luxcpp/dex/build --target amm_xyk_metal)", libPath, err)
	}

	out := make([]uint64, n)

	// Pin the three host buffers and the metallib path C-string for the
	// duration of the cgo dispatch. The driver does its own internal
	// memcpy off these buffers into MTLBuffers, so the pin is only needed
	// for the in-flight call.
	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&reserves[0])
	pinner.Pin(&amounts[0])
	pinner.Pin(&out[0])

	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))

	rc := C.amm_xyk_batch_metal(
		(*C.LuxAmmReservePair)(unsafe.Pointer(&reserves[0])),
		(*C.uint64_t)(unsafe.Pointer(&amounts[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(n),
		cPath,
	)
	runtime.KeepAlive(reserves)
	runtime.KeepAlive(amounts)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("lx: amm_xyk_batch_metal returned %d", int(rc))
	}
	return out, nil
}
