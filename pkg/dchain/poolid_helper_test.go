// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// No build tag: this helper is shared by both the !cgo venue e2e
// (localnet_e2e_test.go, fourpath_e2e_test.go) AND the always-built read RPC
// tests (read_test.go). It is pure-Go (a byte copy), so it belongs in the
// package's default test binary — keeping it in a !cgo-only file made the
// untagged read_test.go fail to compile under CGO_ENABLED=1.
package dchain

// poolIDForSymbol derives a deterministic 32-byte poolId from a market symbol.
// The real V4 poolId is keccak256(PoolKey); for tests any stable, collision-free
// 32-byte identity is sufficient (the d-chain keys markets by the raw 32 bytes).
// We embed the ASCII symbol so logs are human-readable and two distinct symbols
// map to two distinct books; the high byte is tagged so an all-symbol-prefix pool
// can't collide with a zero id.
func poolIDForSymbol(sym string) [32]byte {
	var p [32]byte
	copy(p[:], sym)
	p[31] = 0xD0
	return p
}
