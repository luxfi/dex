// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build dchain && cgo
// +build dchain,cgo

// Command dchain is the standalone D-Chain DEX VM plugin binary. It is the
// venue entrypoint: a ZAP plugin process the Lux node launches (like the EVM
// plugin), running the lx.OrderBook matcher inside consensus and persisting the
// authoritative order book to its own zapdb. Built under //go:build dchain &&
// cgo so it links the cgo/GPU matching accelerator; CI builds it per org+arch.
//
// The testable VM logic (pkg/dchain) is pure-Go and proven without this binary;
// this file only wires the proven VM into the rpc.Serve plugin harness.
package main

import (
	"fmt"

	"github.com/luxfi/dex/cmd/dchain/runner"
	"github.com/luxfi/dex/pkg/dchain"
	"github.com/luxfi/version"
)

func main() {
	versionString := fmt.Sprintf("Lux-DChain/%s [node=%s, rpcchainvm=%d]",
		dchain.Version, version.Current, version.RPCChainVMProtocol)
	runner.Run(versionString)
}
