//go:build grpc

package client

import (
	"context"
	"fmt"

	pb "github.com/luxfi/dex/pkg/grpc/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// newGRPCBackend returns the real gRPC-backed implementation.
func newGRPCBackend() grpcBackend { return &grpcImpl{} }

// grpcImpl wraps a generated LX DEX gRPC client.
type grpcImpl struct {
	conn   *grpc.ClientConn
	client pb.LXDEXServiceClient
}

func (g *grpcImpl) connect(ctx context.Context, addr string) error {
	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to gRPC: %w", err)
	}
	g.conn = conn
	g.client = pb.NewLXDEXServiceClient(conn)
	return nil
}

func (g *grpcImpl) close() error {
	if g.conn != nil {
		return g.conn.Close()
	}
	return nil
}

func (g *grpcImpl) placeOrder(ctx context.Context, order *Order) (*OrderResponse, bool, error) {
	if g.client == nil {
		return nil, false, nil
	}
	req := &pb.PlaceOrderRequest{
		Symbol:      order.Symbol,
		Type:        pb.OrderType(order.Type),
		Side:        pb.OrderSide(order.Side),
		Price:       order.Price,
		Size:        order.Size,
		UserId:      order.UserID,
		ClientId:    order.ClientID,
		TimeInForce: timeInForceToProto(order.TimeInForce),
		PostOnly:    order.PostOnly,
		ReduceOnly:  order.ReduceOnly,
	}
	resp, err := g.client.PlaceOrder(ctx, req)
	if err != nil {
		return nil, true, err
	}
	return &OrderResponse{
		OrderID: resp.OrderId,
		Status:  orderStatusFromProto(resp.Status),
		Message: resp.Message,
	}, true, nil
}

func (g *grpcImpl) cancelOrder(ctx context.Context, orderID uint64) (bool, error) {
	if g.client == nil {
		return false, nil
	}
	_, err := g.client.CancelOrder(ctx, &pb.CancelOrderRequest{OrderId: orderID})
	return true, err
}

func (g *grpcImpl) getOrderBook(ctx context.Context, symbol string, depth int32) (*OrderBook, bool, error) {
	if g.client == nil {
		return nil, false, nil
	}
	resp, err := g.client.GetOrderBook(ctx, &pb.GetOrderBookRequest{
		Symbol: symbol,
		Depth:  depth,
	})
	if err != nil {
		return nil, true, err
	}
	ob := &OrderBook{
		Symbol:    resp.Symbol,
		Timestamp: resp.Timestamp,
		Bids:      make([]PriceLevel, len(resp.Bids)),
		Asks:      make([]PriceLevel, len(resp.Asks)),
	}
	for i, bid := range resp.Bids {
		ob.Bids[i] = PriceLevel{Price: bid.Price, Size: bid.Size}
	}
	for i, ask := range resp.Asks {
		ob.Asks[i] = PriceLevel{Price: ask.Price, Size: ask.Size}
	}
	return ob, true, nil
}

func (g *grpcImpl) getTrades(ctx context.Context, symbol string, limit int32) ([]*Trade, bool, error) {
	if g.client == nil {
		return nil, false, nil
	}
	resp, err := g.client.GetTrades(ctx, &pb.GetTradesRequest{
		Symbol: symbol,
		Limit:  limit,
	})
	if err != nil {
		return nil, true, err
	}
	trades := make([]*Trade, len(resp.Trades))
	for i, t := range resp.Trades {
		trades[i] = &Trade{
			TradeID:     t.TradeId,
			Symbol:      t.Symbol,
			Price:       t.Price,
			Size:        t.Size,
			Side:        OrderSide(t.Side),
			BuyOrderID:  t.BuyOrderId,
			SellOrderID: t.SellOrderId,
			BuyerID:     t.BuyerId,
			SellerID:    t.SellerId,
			Timestamp:   t.Timestamp,
		}
	}
	return trades, true, nil
}

func (g *grpcImpl) streamOrderBook(ctx context.Context, symbol string) (<-chan *OrderBook, error) {
	if g.client == nil {
		return nil, fmt.Errorf("gRPC not connected")
	}
	stream, err := g.client.StreamOrderBook(ctx, &pb.StreamOrderBookRequest{Symbol: symbol})
	if err != nil {
		return nil, err
	}
	ch := make(chan *OrderBook, 100)
	go func() {
		defer close(ch)
		for {
			update, err := stream.Recv()
			if err != nil {
				return
			}
			ob := &OrderBook{
				Symbol:    update.Symbol,
				Timestamp: update.Timestamp,
				Bids:      make([]PriceLevel, len(update.GetBidUpdates())),
				Asks:      make([]PriceLevel, len(update.GetAskUpdates())),
			}
			for i, bid := range update.GetBidUpdates() {
				ob.Bids[i] = PriceLevel{Price: bid.Price, Size: bid.Size}
			}
			for i, ask := range update.GetAskUpdates() {
				ob.Asks[i] = PriceLevel{Price: ask.Price, Size: ask.Size}
			}
			select {
			case ch <- ob:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch, nil
}

// timeInForceToProto converts SDK TimeInForce to proto TimeInForce.
func timeInForceToProto(tif TimeInForce) pb.TimeInForce {
	switch tif {
	case TimeInForceGTC:
		return pb.TimeInForce_GTC
	case TimeInForceIOC:
		return pb.TimeInForce_IOC
	case TimeInForceFOK:
		return pb.TimeInForce_FOK
	case TimeInForceDAY:
		return pb.TimeInForce_DAY
	default:
		return pb.TimeInForce_GTC
	}
}

// orderStatusFromProto converts proto OrderStatus to SDK string status.
func orderStatusFromProto(status pb.OrderStatus) string {
	switch status {
	case pb.OrderStatus_OPEN:
		return string(OrderStatusOpen)
	case pb.OrderStatus_PARTIAL:
		return string(OrderStatusPartial)
	case pb.OrderStatus_FILLED:
		return string(OrderStatusFilled)
	case pb.OrderStatus_CANCELLED:
		return string(OrderStatusCancelled)
	case pb.OrderStatus_REJECTED:
		return string(OrderStatusRejected)
	default:
		return string(OrderStatusOpen)
	}
}
