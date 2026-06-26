// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dexd is the ONE D-Chain DEX binary. It runs the DEX two ways:
//
//	dexd            run as a luxd VM plugin (rpcchainvm). luxd execs the plugin
//	dexd plugin     from its PluginDir with NO args and drives
//	                BuildBlock/Verify/Accept inside multi-validator consensus.
//	                NO-args == plugin is REQUIRED, not a convenience: it is
//	                exactly how luxd launches a plugin (the engine address
//	                arrives via the runtime.EngineAddressKey env var, never argv),
//	                and `dexd version` is the node's version probe.
//
//	dexd run        run the standalone D-Chain venue: open the on-disk zapdb,
//	dexd standalone serve the frozen dex_* surface over ZAP, and drive the
//	                consensus sealer loop itself (WaitForEvent -> BuildBlock ->
//	                Verify -> Accept).
//
// The matcher, VM, and engine all live in pkg/dchain + pkg/lx; this binary only
// selects a run mode and wires it — it duplicates no engine code. CGO_ENABLED
// picks the matcher in pkg/lx (GPU under cgo, pure-Go CPU without); the default
// build is CPU (CGO=0) and links no native code.
package main

import (
	"os"

	"github.com/luxfi/dex/internal/plugin"
	"github.com/luxfi/dex/internal/standalone"
)

func main() {
	// The first arg selects the run mode. Anything that is NOT an explicit
	// standalone/plugin subcommand falls through to the plugin entrypoint —
	// this preserves the luxd plugin contract byte-for-byte: luxd execs the
	// binary with NO args (serve the handshake) or with "version" (probe the
	// version), and both must reach plugin.Run.
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "run", "standalone":
			standalone.Run(os.Args[2:])
			return
		case "plugin":
			plugin.Run()
			return
		}
	}
	plugin.Run()
}
