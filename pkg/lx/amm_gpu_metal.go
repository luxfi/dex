// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && darwin

package lx

// Metal backend for batch xy=k AMM evaluation. Linked via
//   lux-dex-amm-metal pkg-config bundle
//     -> libamm_xyk_metal.a (luxcpp/dex/gpu/metal/amm_xyk_driver.mm)
//     -> amm_xyk.metallib (path resolved at runtime via LUX_DEX_AMM_METALLIB)

/*
#cgo pkg-config: lux-dex-amm-metal

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

var (
	metallibOnce sync.Once
	metallibPath string
)

// resolveMetallibPath returns the absolute path to the precompiled
// amm_xyk.metallib. LUX_DEX_AMM_METALLIB overrides; otherwise we fall
// back to the canonical sibling layout under $LUX_WORKSPACE_ROOT/luxcpp.
func resolveMetallibPath() string {
	metallibOnce.Do(func() {
		if p := os.Getenv("LUX_DEX_AMM_METALLIB"); p != "" {
			metallibPath = p
			return
		}
		root := os.Getenv("LUX_WORKSPACE_ROOT")
		if root == "" {
			if home, err := os.UserHomeDir(); err == nil {
				root = filepath.Join(home, "work")
			}
		}
		metallibPath = filepath.Join(root, "luxcpp", "dex", "build", "amm_xyk.metallib")
	})
	return metallibPath
}

// gpuBatchEval is the Metal implementation called by amm_gpu.go::BatchEvalConstantProduct.
func gpuBatchEval(reserves []ReservePair, amounts []uint64) ([]uint64, error) {
	libPath := resolveMetallibPath()
	if _, err := os.Stat(libPath); err != nil {
		return nil, fmt.Errorf("lx: amm metallib not found at %q: %w (set LUX_DEX_AMM_METALLIB)", libPath, err)
	}

	n := len(reserves)
	out := make([]uint64, n)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&reserves[0])
	pinner.Pin(&amounts[0])
	pinner.Pin(&out[0])

	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))

	rc := C.amm_xyk_batch_metal(
		(*C.AMMReservePair)(unsafe.Pointer(&reserves[0])),
		(*C.uint64_t)(unsafe.Pointer(&amounts[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(n),
		cPath,
	)
	runtime.KeepAlive(reserves)
	runtime.KeepAlive(amounts)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("lx: amm_xyk_batch_metal rc=%d", int(rc))
	}
	return out, nil
}
