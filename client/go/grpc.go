//go:build grpc

// gRPC transport for the LX CLI. Compiled only when the `grpc` build
// tag is set; the default build uses nogrpc.go which wires nil.
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	pb "github.com/luxfi/dex/pkg/grpc/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// GrpcClient implements Client over gRPC.
type GrpcClient struct {
	conn    *grpc.ClientConn
	client  pb.LXDEXServiceClient
	address string
	userID  string
	verbose bool
}

// NewGrpcClient creates a gRPC client.
func NewGrpcClient(address string, verbose bool) (*GrpcClient, error) {
	conn, err := grpc.NewClient(address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(100*1024*1024),
			grpc.MaxCallSendMsgSize(100*1024*1024),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("grpc dial failed: %w", err)
	}

	return &GrpcClient{
		conn:    conn,
		client:  pb.NewLXDEXServiceClient(conn),
		address: address,
		verbose: verbose,
	}, nil
}

// Protocol returns the protocol type.
func (c *GrpcClient) Protocol() Protocol {
	return ProtocolGRPC
}

// SetUserID sets the userID used in requests.
func (c *GrpcClient) SetUserID(userID string) {
	c.userID = userID
}

// PlaceOrder places an order via gRPC.
func (c *GrpcClient) PlaceOrder(ctx context.Context, order *Order) (*OrderResponse, error) {
	req := &pb.PlaceOrderRequest{
		Symbol:   order.Symbol,
		Side:     parseOrderSide(order.Side),
		Type:     parseOrderType(order.Type),
		Price:    order.Price,
		Size:     order.Size,
		UserId:   c.userID,
		ClientId: fmt.Sprintf("cli-%d", time.Now().UnixNano()),
	}

	if c.verbose {
		fmt.Printf(">> gRPC PlaceOrder: %+v\n", req)
	}

	resp, err := c.client.PlaceOrder(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc PlaceOrder failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %+v\n", resp)
	}

	return &OrderResponse{
		OrderID: resp.OrderId,
		Status:  resp.Status.String(),
		Message: resp.Message,
	}, nil
}

// CancelOrder cancels an order via gRPC.
func (c *GrpcClient) CancelOrder(ctx context.Context, orderID uint64) error {
	req := &pb.CancelOrderRequest{
		OrderId: orderID,
		UserId:  c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC CancelOrder: %+v\n", req)
	}

	resp, err := c.client.CancelOrder(ctx, req)
	if err != nil {
		return fmt.Errorf("grpc CancelOrder failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %+v\n", resp)
	}
	return nil
}

// GetOrders retrieves open orders via gRPC.
func (c *GrpcClient) GetOrders(ctx context.Context) ([]Order, error) {
	req := &pb.GetOrdersRequest{
		UserId: c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC GetOrders: %+v\n", req)
	}

	resp, err := c.client.GetOrders(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc GetOrders failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %d orders\n", len(resp.Orders))
	}

	orders := make([]Order, 0, len(resp.Orders))
	for _, o := range resp.Orders {
		orders = append(orders, Order{
			Symbol: o.Symbol,
			Side:   o.Side.String(),
			Type:   o.Type.String(),
			Price:  o.Price,
			Size:   o.Size,
		})
	}
	return orders, nil
}

// GetPositions retrieves positions via gRPC.
func (c *GrpcClient) GetPositions(ctx context.Context) ([]Position, error) {
	req := &pb.GetPositionsRequest{
		UserId: c.userID,
	}

	if c.verbose {
		fmt.Printf(">> gRPC GetPositions: %+v\n", req)
	}

	resp, err := c.client.GetPositions(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc GetPositions failed: %w", err)
	}

	if c.verbose {
		fmt.Printf("<< gRPC Response: %d positions\n", len(resp.Positions))
	}

	positions := make([]Position, 0, len(resp.Positions))
	for _, p := range resp.Positions {
		positions = append(positions, Position{
			Symbol:     p.Symbol,
			Size:       p.Size,
			EntryPrice: p.EntryPrice,
			MarkPrice:  p.MarkPrice,
			PnL:        p.Pnl,
		})
	}
	return positions, nil
}

// Subscribe subscribes to market data via gRPC streaming.
func (c *GrpcClient) Subscribe(ctx context.Context, symbol string) error {
	req := &pb.StreamOrderBookRequest{
		Symbol: symbol,
		Depth:  10,
	}

	if c.verbose {
		fmt.Printf(">> gRPC StreamOrderBook: %+v\n", req)
	}

	stream, err := c.client.StreamOrderBook(ctx, req)
	if err != nil {
		return fmt.Errorf("grpc StreamOrderBook failed: %w", err)
	}

	go func() {
		for {
			update, err := stream.Recv()
			if err != nil {
				if c.verbose {
					fmt.Printf("Stream ended: %v\n", err)
				}
				return
			}
			fmt.Printf("OrderBook %s: %d bids, %d asks\n", update.Symbol, len(update.BidUpdates), len(update.AskUpdates))
		}
	}()
	return nil
}

// Ping tests connectivity via gRPC.
func (c *GrpcClient) Ping(ctx context.Context) (time.Duration, error) {
	start := time.Now()
	req := &pb.PingRequest{
		Timestamp: start.UnixNano(),
	}

	resp, err := c.client.Ping(ctx, req)
	if err != nil {
		return 0, fmt.Errorf("grpc Ping failed: %w", err)
	}

	latency := time.Since(start)
	if c.verbose {
		fmt.Printf("Ping response: %s (latency: %v)\n", resp.Message, latency)
	}
	return latency, nil
}

// GetNodeInfo retrieves node information via gRPC. The returned value
// is the generated *pb.NodeInfo; the interface{} return matches the
// build-agnostic grpcSubClient signature in main.go.
func (c *GrpcClient) GetNodeInfo(ctx context.Context) (interface{}, error) {
	resp, err := c.client.GetNodeInfo(ctx, &pb.GetNodeInfoRequest{})
	if err != nil {
		return nil, fmt.Errorf("grpc GetNodeInfo failed: %w", err)
	}
	return resp, nil
}

// Close closes the gRPC connection.
func (c *GrpcClient) Close() error {
	return c.conn.Close()
}

// ConnectGrpc connects via gRPC.
func (m *ClientManager) ConnectGrpc(address string, verbose bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	client, err := NewGrpcClient(address, verbose)
	if err != nil {
		return err
	}

	m.grpcClient = client
	if m.active == nil {
		m.active = client
	}
	return nil
}

// parseOrderType maps the CLI's string type to a proto OrderType.
func parseOrderType(t string) pb.OrderType {
	switch strings.ToLower(t) {
	case "market":
		return pb.OrderType_MARKET
	case "stop":
		return pb.OrderType_STOP
	case "stop_limit":
		return pb.OrderType_STOP_LIMIT
	default:
		return pb.OrderType_LIMIT
	}
}

// parseOrderSide maps the CLI's string side to a proto OrderSide.
func parseOrderSide(s string) pb.OrderSide {
	switch strings.ToLower(s) {
	case "sell":
		return pb.OrderSide_SELL
	default:
		return pb.OrderSide_BUY
	}
}
