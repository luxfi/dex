// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && darwin && dexgpu

package dex

/*
#include <stddef.h>

// Defined in signed_order_metal_anchor_darwin.c.
extern void* const lux_secp256k1_metal_anchor;

// Returning the anchor's value forces the linker to keep both the anchor
// global and (transitively) the secp256k1_ecrecover_address_batch_metal
// symbol it points at — which is what the dispatch in cpp/ecrecover.cpp
// dlsym's at runtime.
static inline void* lux_secp256k1_metal_anchor_keep(void) {
    return lux_secp256k1_metal_anchor;
}
*/
import "C"

// metalAnchorAddr is referenced by package init below so the entire chain
// (Go reference -> C inline -> anchor global -> Metal driver symbol) survives
// linker DCE.
var metalAnchorAddr = C.lux_secp256k1_metal_anchor_keep()

func init() {
	// Touch the address so the optimizer/linker cannot DCE it.
	if metalAnchorAddr == nil {
		// unreachable on darwin builds where the metal archive is linked.
		_ = metalAnchorAddr
	}
}
