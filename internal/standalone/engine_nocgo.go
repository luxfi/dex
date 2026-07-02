// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo
// +build !cgo

package standalone

// engineName labels which matcher pkg/dex links into this build. Without cgo,
// pkg/dex selects the pure-Go CPU matcher (amm_nogpu / orderbook_nogpu).
// This is a display string only; the matcher selection lives in pkg/dex.
const engineName = "cpu"
