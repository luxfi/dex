// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dchain is the D-Chain DEX VM plugin binary — the native consensus
// matcher VM the Lux node ("D" chain) loads from its PluginDir, exactly like
// the EVM plugin. It runs the lx.OrderBook matcher INSIDE luxd consensus
// (Block.Verify) and persists the authoritative order book to its own zapdb;
// the trade is the D-Chain state transition, sequenced by luxd's multi-validator
// engine — there is no standalone venue and no ZAP relay in the trading path.
//
// The VM (pkg/dchain) is pure-Go (CGO=0): the matcher, settlement, and block
// lifecycle carry no native dependency, so this plugin builds and ships CGO=0
// for every org+arch — the same harness as chains/*/cmd/plugin. The optional
// GPU AMM accelerator in pkg/lx is a SEPARATE concern gated by its own
// cuda/metal tags and is NOT linked by this consensus plugin.
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
