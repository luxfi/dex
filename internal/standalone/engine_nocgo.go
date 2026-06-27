// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo
// +build !cgo

package standalone

// engineName labels which matcher pkg/lx links into this build. Without cgo,
// pkg/lx selects the pure-Go CPU matcher (amm_nogpu / orderbook_nogpu).
// This is a display string only; the matcher selection lives in pkg/lx.
const engineName = "cpu"
