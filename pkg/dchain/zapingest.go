// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/luxfi/rpc"
)

// zapingest.go is the D-Chain DEX ZAP ingestion seam for the IN-LUXD native VM —
// the CANONICAL native order path. It is the second transport over the one
// authoritative handler core (dexMethods, handler.go), symmetric with the HTTP
// compat transport (ingest.go / CreateHandlers):
//
//	ZAP socket  (this file) : RegisterDEX on rpc.Listen — the HFT native path,
//	                          binary wire, co-located on the pod. CANONICAL.
//	HTTP router (ingest.go) : the same dexMethods over the node's /v1/chain route.
//	                          COMPAT (web / exchange-api / debug).
//
// WHY BOTH, AND WHY THIS IS NOT DUPLICATION: transport is orthogonal to consensus.
// Either transport turns a FROZEN zapwire frame into the IDENTICAL Tx, Adds it to
// the SAME mempool, and the order matches at Block.Verify and settles at
// Block.Accept — validators replay from the accepted block BYTES, never from the
// wire stream or a wall clock. So the choice of socket vs HTTP is purely a
// submission perf/UX axis with no safety implication; the ONE matcher core
// (dexMethods -> submitTx -> mempool) is shared, never re-implemented per
// transport. The ZAP socket is the canonical native path because the DEX's whole
// point is the low-latency streaming HFT seam (co-locate the matcher with the ZAP
// relay; see dex perf notes); HTTP stays as the broad-compatibility surface.
//
// CONFIG, NOT CODE PATH: the socket is enabled by the chain's init Config bytes
// (zapIngestAddr), so whether the in-luxd plugin serves ZAP is a per-environment
// DATA decision the operator/genesis sets — not a build tag and not a forked entry
// point. When the address is empty (in-process tests; the standalone cmd/dvenue,
// which runs its OWN socket) the VM serves HTTP only and this seam is dormant. The
// VM owns the socket lifecycle: Initialize starts it after the durable state is
// loaded, Shutdown closes it.

// vmConfig is the VM's init Config (block.Init.Config) JSON shape. It is forwards-
// only: unknown fields are ignored, an empty/absent Config is the zero value (no
// ZAP socket). Only the fields the VM actually consumes are declared.
type vmConfig struct {
	// ZAPIngestAddr, when non-empty, is the TCP address the co-located ZAP DEX
	// socket listens on inside the plugin process (e.g. "0.0.0.0:9101"). Empty
	// means the canonical socket is not served (HTTP compat only). It is a
	// pod-local listen address the node/operator assigns; orders arriving here
	// flow through the identical submitTx -> mempool -> consensus path as HTTP.
	ZAPIngestAddr string `json:"zapIngestAddr"`

	// DepositAuthority is the hex-encoded 16-byte settlement account (Account16) of
	// the trusted bridge/proxy permitted to authorize TxDeposit — the entity that
	// custodies the backing C-side value. Empty (the default) means NO deposit
	// authority: every TxDeposit is rejected fail-closed, and value enters the ledger
	// ONLY via the backed atomic import. It is a CONSENSUS parameter — every
	// validator MUST configure the identical value, exactly like any chain param.
	DepositAuthority string `json:"depositAuthority"`
}

// depositAuthorityKey decodes DepositAuthority into a 16-byte account. Empty => the
// zero account (no authority; all deposits fail-closed). A non-empty value must be
// exactly 16 bytes of hex, else a hard error (a misconfigured backing authority
// must fail Initialize rather than silently admit or reject the wrong signer).
func (c vmConfig) depositAuthorityKey() (userKey, error) {
	var k userKey
	if c.DepositAuthority == "" {
		return k, nil
	}
	b, err := hex.DecodeString(c.DepositAuthority)
	if err != nil {
		return k, fmt.Errorf("depositAuthority not hex: %w", err)
	}
	if len(b) != len(k) {
		return k, fmt.Errorf("depositAuthority must be %d bytes, got %d", len(k), len(b))
	}
	copy(k[:], b)
	return k, nil
}

// parseConfig decodes the VM's init Config bytes. An empty config is valid and
// yields the zero vmConfig (no ZAP socket). A non-empty but malformed config is an
// error — a misconfigured chain must fail fast at Initialize, not boot with a
// silently-dropped transport setting.
func parseConfig(b []byte) (vmConfig, error) {
	var c vmConfig
	if len(b) == 0 {
		return c, nil
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("dchain: parse init config: %w", err)
	}
	return c, nil
}

// startZAPIngest opens the canonical co-located ZAP DEX socket and serves the SAME
// dexMethods the HTTP transport folds over. It is a no-op when addr is empty (the
// VM serves HTTP only). It is ADDITIVE: it reuses RegisterDEX (the existing socket
// seam cmd/dvenue uses) and adds no new handler logic. Must be called under vm.mu,
// after the durable state is loaded (so the socket never accepts an order before
// the VM can sequence it). The server runs in its own goroutine whose lifetime is
// the VM's — it is cancelled by Shutdown via zapIngestCancel, NOT by the (short-lived)
// Initialize context, so no Initialize ctx is taken.
func (vm *VM) startZAPIngest(addr string) error {
	if addr == "" {
		return nil // HTTP-compat-only: canonical socket not served in this environment
	}
	server, err := rpc.Listen(addr)
	if err != nil {
		return fmt.Errorf("dchain: ZAP ingest listen %q: %w", addr, err)
	}
	if err := vm.RegisterDEX(server); err != nil {
		_ = server.Close()
		return fmt.Errorf("dchain: register DEX on ZAP ingest socket: %w", err)
	}

	serveCtx, cancel := context.WithCancel(context.Background())
	vm.zapIngest = server
	vm.zapIngestCancel = cancel
	go func() {
		if serr := server.Serve(serveCtx); serr != nil && serveCtx.Err() == nil {
			vm.log.Error("dchain ZAP ingest serve", "addr", server.Addr(), "err", serr)
		}
	}()
	vm.log.Info("dchain ZAP ingest serving (canonical native DEX path)", "addr", server.Addr())
	return nil
}

// stopZAPIngest cancels and closes the ZAP ingest socket if one is running. Must be
// called under vm.mu. Idempotent.
func (vm *VM) stopZAPIngest() {
	if vm.zapIngestCancel != nil {
		vm.zapIngestCancel()
		vm.zapIngestCancel = nil
	}
	if vm.zapIngest != nil {
		_ = vm.zapIngest.Close()
		vm.zapIngest = nil
	}
}
