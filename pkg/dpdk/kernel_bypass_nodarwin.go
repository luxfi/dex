// Copyright (C) 2020-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !darwin

package dpdk

// DarwinOptimizations is a no-op off Darwin. See kernel_bypass_darwin.go.
func DarwinOptimizations(fd int) {}
