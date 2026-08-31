// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dchain

import (
	"context"
	"io"
	"net/http"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// ingest.go is the D-Chain DEX ingestion seam for the IN-LUXD native VM. When
// the VM runs as a luxd plugin (cmd/dchain via github.com/luxfi/vm/rpc.Serve),
// luxd asks the VM for HTTP handlers (CreateHandlers); the plugin transport stands
// up a local http.Server per handler inside the plugin process and the node
// reverse-proxies to it. luxd mounts each handler at /v1/chain/<chainID-or-alias>
// + <endpoint-key> as an EXACT gorilla/mux route (node/server/http/router.go:
// `r.router.Handle(base+endpoint, handler)` — exact path, no subtree). So we
// return ONE handler per method, each keyed by its full sub-path:
//
//	/v1/dex/dex/dex_place    -> handlePlace
//	/v1/dex/dex/dex_submit   -> handleSubmit
//	/v1/dex/dex/dex_cancel   -> handleCancel
//	...
//
// The request body is the FROZEN zapwire payload verbatim; the response body is
// the handler's raw bytes (zapwire ack / fills / balance frame).
//
// THE INVARIANT THAT MATTERS: this is pure transport. The dex_* handler each
// route dispatches to is the EXACT func value RegisterDEX binds on the ZAP socket
// (vm.dexMethods) — so an order POSTed here flows submitTx -> mempool.Add ->
// consensus PendingTxs -> BuildBlock -> Block.Verify (the matcher) -> Block.Accept,
// and the bytes returned are the CONSENSUS-COMPUTED outcome (every validator
// re-derived the same fills at Verify). It NEVER relays to an external matcher and
// NEVER mutates the book synchronously: the chain is the matcher.

// ingestNamespace is the path segment grouping the DEX methods under the chain
// route. The full external path of a method m is /v1/chain/<chainID>/dex/<m>.
const ingestNamespace = "dex"

// maxIngestBody bounds a request body. The largest frozen frame is the 96-byte
// OpenMarket; 4 KiB is comfortably above every frame and rejects a runaway body
// without parsing it (defence at the boundary — the handlers also length-check).
const maxIngestBody = 4 << 10

// httpHandlers returns the VM's DEX methods as luxd HTTP handlers, keyed by their
// full endpoint sub-path ("/dex/<method>"). It folds the SAME vm.dexMethods table
// the ZAP socket transport uses — one method->handler table, two transports — so a
// method added to dexMethods is reachable over HTTP automatically. luxd registers
// each key as an exact route under /v1/chain/<chainID> (see file doc).
//
// It ALSO folds the read surface (vm.readMethods, read.go): the dex_get_* query
// endpoints that return committed state as JSON. Reads are mounted on the same
// route group but never touch the mempool/consensus — orthogonal to the writes.
func (vm *VM) httpHandlers() map[string]http.Handler {
	out := make(map[string]http.Handler, len(vm.dexMethods())+len(vm.readMethods()))
	for _, m := range vm.dexMethods() {
		out["/"+ingestNamespace+"/"+m.method] = rawHandlerHTTP(m.handler)
	}
	for _, m := range vm.readMethods() {
		out["/"+ingestNamespace+"/"+m.method] = m.handler
	}
	return out
}

// rawHandlerHTTP adapts a github.com/luxfi/rpc.RawHandler (the ZAP handler
// signature) to an http.Handler. It enforces POST, reads the raw frozen-frame body
// as the payload, invokes the handler with the REQUEST context (so a client
// disconnect cancels the consensus wait and submitTx withdraws the tx — RED #4),
// and writes the handler's raw response bytes back. A handler error becomes 502
// with the error text (the order could not be turned into a consensus outcome); a
// success is application/octet-stream — the frozen wire frame, never JSON.
func rawHandlerHTTP(h rpc.RawHandler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		payload, err := io.ReadAll(io.LimitReader(r.Body, maxIngestBody))
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		resp, err := h(r.Context(), payload)
		if err != nil {
			// The order/op could not be sequenced into a consensus outcome (e.g.
			// the client disconnected and the wait was cancelled, or the frame was
			// undecodable). 502: the request was well-formed but the VM could not
			// produce an upstream (consensus) result.
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(resp)
	})
}

// compile-time guards.
var (
	_ rpc.RawHandler = func(context.Context, []byte) ([]byte, error) { return nil, nil }
	// the frozen method names are what we route on.
	_ = zapwire.MethodSubmit
)
