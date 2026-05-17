// Package client — gRPC backend interface.
//
// Always compiled. The default build provides a no-op implementation in
// grpc_backend_nogrpc.go; building with `-tags=grpc` substitutes the
// real implementation in grpc_backend_grpc.go.
package client

import (
	"context"
	"errors"
)

// ErrGRPCNotBuilt is returned when a gRPC-only method is invoked in
// the default build. Rebuild with `go build -tags=grpc` to enable
// gRPC support.
var ErrGRPCNotBuilt = errors.New("client: gRPC support requires the `grpc` build tag (rebuild with -tags=grpc)")

// grpcBackend abstracts the gRPC transport. The ok return is true when
// the backend handled the call; the JSON-RPC fallback in the public
// method is used when ok is false.
type grpcBackend interface {
	connect(ctx context.Context, addr string) error
	close() error

	placeOrder(ctx context.Context, order *Order) (resp *OrderResponse, ok bool, err error)
	cancelOrder(ctx context.Context, orderID uint64) (ok bool, err error)
	getOrderBook(ctx context.Context, symbol string, depth int32) (ob *OrderBook, ok bool, err error)
	getTrades(ctx context.Context, symbol string, limit int32) (trades []*Trade, ok bool, err error)
	streamOrderBook(ctx context.Context, symbol string) (<-chan *OrderBook, error)
}
