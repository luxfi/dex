//go:build !grpc

package client

import "context"

// newGRPCBackend returns the no-op backend used in the default build.
func newGRPCBackend() grpcBackend { return noopGRPC{} }

// noopGRPC is the default-build backend. Every method reports
// "not handled" so the JSON-RPC fallback runs. ConnectGRPC and
// StreamOrderBook return ErrGRPCNotBuilt to make the missing
// capability explicit.
type noopGRPC struct{}

func (noopGRPC) connect(ctx context.Context, addr string) error { return ErrGRPCNotBuilt }
func (noopGRPC) close() error                                   { return nil }

func (noopGRPC) placeOrder(ctx context.Context, order *Order) (*OrderResponse, bool, error) {
	return nil, false, nil
}
func (noopGRPC) cancelOrder(ctx context.Context, orderID uint64) (bool, error) {
	return false, nil
}
func (noopGRPC) getOrderBook(ctx context.Context, symbol string, depth int32) (*OrderBook, bool, error) {
	return nil, false, nil
}
func (noopGRPC) getTrades(ctx context.Context, symbol string, limit int32) ([]*Trade, bool, error) {
	return nil, false, nil
}
func (noopGRPC) streamOrderBook(ctx context.Context, symbol string) (<-chan *OrderBook, error) {
	return nil, ErrGRPCNotBuilt
}
