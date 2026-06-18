// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build dchain && cgo
// +build dchain,cgo

// Package runner boots the standalone D-Chain DEX VM as a ZAP plugin process,
// mirroring github.com/luxfi/evm/plugin/runner. It is the //go:build dchain &&
// cgo VENUE entrypoint: the binary links the cgo/GPU matching engine (the venue
// accelerator) alongside the pure-Go consensus VM, so it is built and shipped by
// CI for each org+arch, never linked in the pure-Go test path.
package runner

import (
	"context"
	"fmt"
	"os"

	"github.com/luxfi/dex/pkg/dchain"
	log "github.com/luxfi/log"
	"github.com/luxfi/vm/rpc"
)

// Run boots the d-chain VM over the ZAP plugin transport. rpc.Serve connects to
// the node runtime named in the runtime.EngineAddressKey env var, opens this
// chain's own badgerdb/zapdb at ChainDataDir/db, hands it to VM.Initialize as
// init.DB, and serves block requests — the exact harness lux/evm boots through.
// It blocks until the runtime shuts the VM down. versionStr, when non-empty and
// the process was invoked with the version flag, is printed and the process
// exits 0 (the node probes plugin versions this way).
func Run(versionStr string) {
	if printVersion() && versionStr != "" {
		fmt.Println(versionStr)
		os.Exit(0)
	}
	if err := rpc.Serve(context.Background(), log.Root(), &dchain.VM{}); err != nil {
		fmt.Printf("dchain rpc.Serve error: %s\n", err)
		os.Exit(1)
	}
}

// printVersion reports whether the process was asked to just print its version
// and exit. The node passes a single "version" arg when probing a plugin.
func printVersion() bool {
	for _, a := range os.Args[1:] {
		if a == "version" || a == "--version" || a == "-version" {
			return true
		}
	}
	return false
}
