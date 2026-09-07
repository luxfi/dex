// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/vm/manager"
)

// VMID is the D-Chain's identifier, the one the node resolves a CreateChainTx
// to. It is spelled here, beside the VM, so the chain and the id it answers to
// cannot drift apart.
var VMID = ids.ID{'d', 'e', 'x', 'v', 'm'}

// Factory builds D-Chain VMs. It is the ONE construction path: the node calls
// it to register the chain in-process, and the plugin entrypoint calls it
// before handing the VM to rpc.Serve. Two constructors would eventually build
// two different VMs and the difference would show up as a chain that behaves
// one way inside luxd and another way behind the plugin harness.
//
// A VM built here holds nothing. Initialize opens the database, rebuilds the
// book cache from the order rows, and does the rest.
type Factory struct{}

var _ manager.Factory = (*Factory)(nil)

func (*Factory) New(log.Logger) (interface{}, error) { return &VM{}, nil }
