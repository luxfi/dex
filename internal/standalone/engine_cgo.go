// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo
// +build cgo

package standalone

// engineName labels which matcher pkg/lx links into this build. Under cgo,
// pkg/lx selects the GPU matcher (amm_gpu_cuda / amm_gpu_metal / orderbook_cuda).
// This is a display string only; the matcher selection lives in pkg/lx.
const engineName = "gpu"
