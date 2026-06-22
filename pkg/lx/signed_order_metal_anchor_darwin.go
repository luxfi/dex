// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && darwin && lux_secp256k1_metal

// Force-link anchor for the OPTIONAL secp256k1 Metal accelerator archive
// (libsecp256k1_metal.a), Apple Silicon only. Compiled only under the
// `lux_secp256k1_metal` build tag, so a default CGO build never requires the
// archive to exist and runs the secp256k1 CPU pipeline.
//
// To enable GPU dispatch:
//
//  1. Build the Metal archive in luxcpp/crypto:
//       cmake --build $HOME/work/luxcpp/crypto/build --target secp256k1_metal
//     (emits libsecp256k1_metal.a under the build/secp256k1 tree)
//
//  2. Build the DEX with the tag, pointing the linker at the archive dir:
//       CGO_ENABLED=1 \
//       CGO_LDFLAGS="-L$HOME/work/luxcpp/crypto/build/secp256k1" \
//       go build -tags lux_secp256k1_metal ./...
//
//  3. At runtime export LUX_SECP256K1_METALLIB=<path/to/secp256k1.metallib>
//     (and leave LUX_SECP256K1_BACKEND unset / != "cpu"). The dispatch in
//     cpp/ecrecover.cpp dlsym's the GPU symbol this anchor keeps alive.
//
// Backend selection stays a RUNTIME decision: even with the archive linked,
// an unset LUX_SECP256K1_METALLIB or LUX_SECP256K1_BACKEND=cpu uses the CPU
// pipeline.

package lx

/*
// Apple-only accelerator; pulls in the Metal/Foundation frameworks. The
// archive directory is supplied by the operator via CGO_LDFLAGS (see
// header) so this file carries no build-tree path of its own.
#cgo darwin LDFLAGS: -lsecp256k1_metal -framework Metal -framework Foundation

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
