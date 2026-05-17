//go:build !grpc

// Default-build stubs for the gRPC transport. The CLI is fully
// functional via WebSocket without gRPC compiled in; the gRPC paths
// (the `ping` / `info` commands, `-protocol=grpc`, and the `switch
// grpc` interactive command) report that gRPC is not compiled in.
package main

import "errors"

// errGRPCNotBuilt is returned by ConnectGrpc when the binary was built
// without the `grpc` build tag. Rebuild with `go build -tags=grpc` to
// enable gRPC support.
var errGRPCNotBuilt = errors.New("gRPC support requires the `grpc` build tag (rebuild with -tags=grpc)")

// ConnectGrpc reports that gRPC is not compiled in.
func (m *ClientManager) ConnectGrpc(address string, verbose bool) error {
	return errGRPCNotBuilt
}
