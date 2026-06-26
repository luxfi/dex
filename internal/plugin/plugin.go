// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package plugin boots the native D-Chain DEX VM as a luxd rpcchainvm plugin,
// mirroring github.com/luxfi/evm/plugin/runner. rpc.Serve takes the engine
// address from the runtime.EngineAddressKey env var, opens this chain's own
// badgerdb/zapdb, hands it to VM.Initialize, and serves block requests so
// BuildBlock/Verify/Accept run inside luxd's multi-validator consensus — the
// exact harness lux/evm boots through. The VM (pkg/dchain) is pure-Go (CGO=0);
// this entrypoint links no native code.
package plugin

import (
	"context"
	"fmt"
	"os"

	"github.com/luxfi/dex/pkg/dchain"
	log "github.com/luxfi/log"
	"github.com/luxfi/version"
	"github.com/luxfi/vm/rpc"
)

// Run boots the d-chain VM over the rpcchainvm plugin transport. It blocks
// until the node runtime shuts the VM down. When luxd probes the plugin version
// (a single "version" arg), it prints the version line and exits 0.
func Run() {
	versionStr := fmt.Sprintf("Lux-DChain/%s [node=%s, rpcchainvm=%d]",
		dchain.Version, version.Current, version.RPCChainVMProtocol)
	if printVersion() {
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
