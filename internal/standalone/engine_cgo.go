// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo
// +build cgo

package standalone

// engineName labels which matcher pkg/dex links into this build. Under cgo,
// pkg/dex selects the GPU matcher (amm_gpu_cuda / amm_gpu_metal / orderbook_gpu_cuda).
// This is a display string only; the matcher selection lives in pkg/dex.
const engineName = "gpu"
