// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dpdk

import "syscall"

// DarwinOptimizations applies macOS-specific socket options.
//
// This lives behind a build tag rather than a `runtime.GOOS != "darwin"` guard,
// because syscall.SO_NOSIGPIPE is a COMPILE-time constant that exists only on
// Darwin. A runtime check cannot save a reference the linux build has to
// resolve, so `go test ./...` failed on the runner with:
//
//	kernel_bypass.go:312:56: undefined: syscall.SO_NOSIGPIPE
func DarwinOptimizations(fd int) {
	// Prevent SIGPIPE on a closed peer.
	_ = syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_NOSIGPIPE, 1)
}
