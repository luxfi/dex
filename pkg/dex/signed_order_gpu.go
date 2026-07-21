// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && dexgpu

package dex

// GPU/batched signed-order verification.
//
// The C primitive lives in luxcpp/crypto:
//   header: $LUXCPP_PREFIX/include/lux/crypto/secp256k1.h
//   body:   libsecp256k1_cpu.a in $LUXCPP_PREFIX/lib (CPU oracle)
//   metal:  libsecp256k1_metal.a in luxcpp/crypto/build/secp256k1
//           (still consumed from the build tree — Metal-only, no .pc yet)
//
// The CPU/oracle entry point is `secp256k1_ecrecover_address_batch`: it
// takes (hash, r||s||v) tuples and writes 20-byte Ethereum addresses out.
// We then compare each recovered address with order.Sender and emit the
// per-order bool.
//
// CPU linkage goes through the lux-crypto-secp256k1 pkg-config bundle
// (libsecp256k1_cpu + libkeccak_cpu). Build + install once with:
//
//   cmake -S $HOME/work/luxcpp/crypto -B $HOME/work/luxcpp/crypto/build \
//         -DCMAKE_INSTALL_PREFIX=$HOME/work/luxcpp/install
//   cmake --build $HOME/work/luxcpp/crypto/build \
//         --target secp256k1_cpu keccak_cpu
//   cmake --install $HOME/work/luxcpp/crypto/build

/*
#cgo pkg-config: lux-crypto-secp256k1
//
// Metal driver: dispatch in cpp/ecrecover.cpp resolves the GPU symbol via
// dlsym(RTLD_DEFAULT, ...). The anchor file signed_order_metal_anchor_darwin.c
// takes its address so the linker pulls libsecp256k1_metal.a into the final
// binary. Set LUX_SECP256K1_METALLIB to enable GPU dispatch; unset or
// LUX_SECP256K1_BACKEND=cpu uses the CPU pipeline.
//
// libsecp256k1_metal is NOT in the lux-crypto-secp256k1 .pc bundle (Apple
// only, Metal/Foundation frameworks required). Keep the build-tree path
// until a separate lux-crypto-secp256k1-metal .pc lands.
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/secp256k1
#cgo darwin LDFLAGS: -lsecp256k1_metal -framework Metal -framework Foundation

#include <stdint.h>
#include <stddef.h>
#include "lux/crypto/secp256k1.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// BatchVerifyOrders verifies a batch of SignedOrders in a single cgo dispatch
// into the luxcpp/crypto secp256k1 batch ecrecover pipeline. Returns a slice
// of bools the same length as orders; out[i]=true means
// ecrecover(SigningHash(orders[i]), orders[i].Sig) == orders[i].Sender.
//
// A returned error means the batch primitive itself failed (bad arguments,
// linker mismatch); per-order rejection is conveyed via out[i]=false rather
// than as an error, so a single bad sig in a 10k batch does not abort the
// other 9,999 verifications.
func BatchVerifyOrders(orders []SignedOrder) ([]bool, error) {
	n := len(orders)
	if n == 0 {
		return nil, nil
	}

	// Stage 1: Go-side prep — pack hashes (n*32) and sigs (n*65) for the C
	// kernel. SigningHash failure flips that order's slot to "skip" so we
	// don't feed nonsense into the recovery and falsely match someone.
	hashes := make([]byte, n*32)
	sigs := make([]byte, n*65)
	skip := make([]bool, n)
	for i := range orders {
		h, err := orders[i].SigningHash()
		if err != nil {
			skip[i] = true
			continue
		}
		copy(hashes[i*32:(i+1)*32], h[:])
		copy(sigs[i*65:(i+1)*65], orders[i].Sig[:])
		// Reject obviously malformed v up front (the C kernel only
		// accepts {0,1}); marking it skipped here avoids a per-tuple
		// error code we'd then have to translate back.
		if v := orders[i].Sig[64]; v > 1 {
			skip[i] = true
		}
	}

	// Stage 2: single cgo dispatch. We pin the four base pointers for the
	// duration of the call — same pattern as cevm.BatchRecoverSenders.
	addrs := make([]byte, n*20)
	st := make([]byte, n)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&hashes[0])
	pinner.Pin(&sigs[0])
	pinner.Pin(&addrs[0])
	pinner.Pin(&st[0])

	rc := C.secp256k1_ecrecover_address_batch(
		C.size_t(n),
		(*C.uint8_t)(unsafe.Pointer(&hashes[0])),
		(*C.uint8_t)(unsafe.Pointer(&sigs[0])),
		(*C.uint8_t)(unsafe.Pointer(&addrs[0])),
		(*C.uint8_t)(unsafe.Pointer(&st[0])),
	)
	runtime.KeepAlive(hashes)
	runtime.KeepAlive(sigs)
	runtime.KeepAlive(addrs)
	runtime.KeepAlive(st)

	if rc != C.SECP256K1_OK {
		return nil, fmt.Errorf("lx: secp256k1_ecrecover_address_batch returned %d", int(rc))
	}

	// Stage 3: walk per-tuple status + compare recovered address with the
	// caller-claimed Sender. status != 0 means the kernel rejected this
	// signature (malformed r/s/v, point at infinity, etc.) — that's an
	// invalid order, not a batch error.
	out := make([]bool, n)
	for i := 0; i < n; i++ {
		if skip[i] || st[i] != 0 {
			out[i] = false
			continue
		}
		ok := true
		for j := 0; j < 20; j++ {
			if addrs[i*20+j] != orders[i].Sender[j] {
				ok = false
				break
			}
		}
		out[i] = ok
	}
	return out, nil
}
