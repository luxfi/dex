package grpc

import (
	"context"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/luxfi/dex/pkg/grpc/pb"
	"github.com/luxfi/dex/pkg/lx"
	"github.com/luxfi/log"
)

// Server implements the gRPC LXDEXService
type Server struct {
	pb.UnimplementedLXDEXServiceServer
	orderBook *lx.OrderBook
	logger    log.Logger
	nodeID    string
	version   string
	network   string
}

// NewServer creates a new gRPC server
func NewServer(orderBook *lx.OrderBook, logger log.Logger, nodeID, version, network string) *Server {
	return &Server{
		orderBook: orderBook,
		logger:    logger,
		nodeID:    nodeID,
		version:   version,
		network:   network,
	}
}

// PlaceOrder places a new order
func (s *Server) PlaceOrder(ctx context.Context, req *pb.PlaceOrderRequest) (*pb.PlaceOrderResponse, error) {
	order := &lx.Order{
		Symbol: req.Symbol,
		Type:   lx.OrderType(req.Type),
		Side:   lx.Side(req.Side),
		Price:  req.Price,
		Size:   req.Size,
		UserID: req.UserId,
		ClientID: req.ClientId,
		PostOnly: req.PostOnly,
		ReduceOnly: req.ReduceOnly,
		Timestamp: time.Now(),
	}

	// Map TimeInForce
	switch req.TimeInForce {
	case pb.TimeInForce_IOC:
		order.TimeInForce = lx.ImmediateOrCancel
	case pb.TimeInForce_FOK:
		order.TimeInForce = lx.FillOrKill
	case pb.TimeInForce_DAY:
		order.TimeInForce = lx.Day
	default:
		order.TimeInForce = lx.GoodTillCancelled
	}

	orderID := s.orderBook.AddOrder(order)
	if orderID == 0 {
		return &pb.PlaceOrderResponse{
			OrderId: 0,
			Status:  pb.OrderStatus_REJECTED,
			Message: "Order rejected",
		}, nil
	}

	return &pb.PlaceOrderResponse{
		OrderId: orderID,
		Status:  pb.OrderStatus_OPEN,
		Message: "Order accepted",
	}, nil
}

// CancelOrder cancels an existing order
func (s *Server) CancelOrder(ctx context.Context, req *pb.CancelOrderRequest) (*pb.CancelOrderResponse, error) {
	err := s.orderBook.CancelOrder(req.OrderId)
	if err != nil {
		return &pb.CancelOrderResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}

	return &pb.CancelOrderResponse{
		Success: true,
		Message: "Order cancelled",
	}, nil
}

// GetOrder retrieves an order by ID
func (s *Server) GetOrder(ctx context.Context, req *pb.GetOrderRequest) (*pb.Order, error) {
	order := s.orderBook.GetOrder(req.OrderId)
	if order == nil {
		return nil, status.Errorf(codes.NotFound, "order not found")
	}

	return s.convertOrderToProto(order), nil
}

// GetOrders retrieves multiple orders
func (s *Server) GetOrders(ctx context.Context, req *pb.GetOrdersRequest) (*pb.GetOrdersResponse, error) {
	// TODO: Implement order filtering
	orders := make([]*pb.Order, 0)
	
	// For now, return empty list
	return &pb.GetOrdersResponse{
		Orders: orders,
	}, nil
}

// GetOrderBook returns the current order book
func (s *Server) GetOrderBook(ctx context.Context, req *pb.GetOrderBookRequest) (*pb.OrderBook, error) {
	snapshot := s.orderBook.GetSnapshot()
	
	depth := int(req.Depth)
	if depth <= 0 {
		depth = 10
	}

	bids := make([]*pb.PriceLevel, 0, len(snapshot.Bids))
	for i, bid := range snapshot.Bids {
		if i >= depth {
			break
		}
		bids = append(bids, &pb.PriceLevel{
			Price: bid.Price,
			Size:  bid.Size,
			Count: int32(1), // TODO: Add order count
		})
	}

	asks := make([]*pb.PriceLevel, 0, len(snapshot.Asks))
	for i, ask := range snapshot.Asks {
		if i >= depth {
			break
		}
		asks = append(asks, &pb.PriceLevel{
			Price: ask.Price,
			Size:  ask.Size,
			Count: int32(1),
		})
	}

	return &pb.OrderBook{
		Symbol:    req.Symbol,
		Bids:      bids,
		Asks:      asks,
		Timestamp: time.Now().Unix(),
	}, nil
}

// StreamOrderBook streams order book updates
func (s *Server) StreamOrderBook(req *pb.StreamOrderBookRequest, stream pb.LXDEXService_StreamOrderBookServer) error {
	// TODO: Implement real-time streaming
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-stream.Context().Done():
			return nil
		case <-ticker.C:
			// Send snapshot as update for now
			snapshot := s.orderBook.GetSnapshot()
			
			update := &pb.OrderBookUpdate{
				Symbol:    req.Symbol,
				Timestamp: time.Now().Unix(),
			}

			// Send top bid/ask as updates
			if len(snapshot.Bids) > 0 {
				update.BidUpdates = []*pb.PriceLevel{{
					Price: snapshot.Bids[0].Price,
					Size:  snapshot.Bids[0].Size,
				}}
			}
			if len(snapshot.Asks) > 0 {
				update.AskUpdates = []*pb.PriceLevel{{
					Price: snapshot.Asks[0].Price,
					Size:  snapshot.Asks[0].Size,
				}}
			}

			if err := stream.Send(update); err != nil {
				return err
			}
		}
	}
}

// GetTrades returns recent trades
func (s *Server) GetTrades(ctx context.Context, req *pb.GetTradesRequest) (*pb.GetTradesResponse, error) {
	trades := s.orderBook.GetTrades()
	
	limit := int(req.Limit)
	if limit <= 0 || limit > len(trades) {
		limit = len(trades)
	}

	pbTrades := make([]*pb.Trade, 0, limit)
	start := len(trades) - limit
	if start < 0 {
		start = 0
	}

	for i := start; i < len(trades); i++ {
		pbTrades = append(pbTrades, s.convertTradeToProto(&trades[i]))
	}

	return &pb.GetTradesResponse{
		Trades: pbTrades,
	}, nil
}

// StreamTrades streams real-time trades
func (s *Server) StreamTrades(req *pb.StreamTradesRequest, stream pb.LXDEXService_StreamTradesServer) error {
	// TODO: Implement real-time trade streaming
	return status.Errorf(codes.Unimplemented, "trade streaming not yet implemented")
}

// GetBalance returns user balance
func (s *Server) GetBalance(ctx context.Context, req *pb.GetBalanceRequest) (*pb.Balance, error) {
	// TODO: Implement balance tracking
	return &pb.Balance{
		Asset:     req.Asset,
		Available: 10000.0,
		Locked:    0.0,
		Total:     10000.0,
	}, nil
}

// GetPositions returns user positions
func (s *Server) GetPositions(ctx context.Context, req *pb.GetPositionsRequest) (*pb.GetPositionsResponse, error) {
	// TODO: Implement position tracking
	return &pb.GetPositionsResponse{
		Positions: []*pb.Position{},
	}, nil
}

// GetNodeInfo returns node information
func (s *Server) GetNodeInfo(ctx context.Context, req *pb.GetNodeInfoRequest) (*pb.NodeInfo, error) {
	return &pb.NodeInfo{
		NodeId:      s.nodeID,
		Version:     s.version,
		Network:     s.network,
		BlockHeight: 0, // TODO: Get from consensus
		OrderCount:  uint64(len(s.orderBook.Orders)),
		TradeCount:  uint64(len(s.orderBook.Trades)),
		Uptime:      time.Now().Unix(), // TODO: Track actual uptime
		Syncing:     false,
		SupportedMarkets: []string{s.orderBook.Symbol},
	}, nil
}

// GetPeers returns connected peers
func (s *Server) GetPeers(ctx context.Context, req *pb.GetPeersRequest) (*pb.GetPeersResponse, error) {
	// TODO: Implement peer tracking
	return &pb.GetPeersResponse{
		Peers: []*pb.Peer{},
	}, nil
}

// Ping responds to ping requests
func (s *Server) Ping(ctx context.Context, req *pb.PingRequest) (*pb.PingResponse, error) {
	return &pb.PingResponse{
		Timestamp: time.Now().Unix(),
		Message:   "pong",
	}, nil
}

// Helper functions

func (s *Server) convertOrderToProto(order *lx.Order) *pb.Order {
	return &pb.Order{
		OrderId:    order.ID,
		Symbol:     order.Symbol,
		Type:       pb.OrderType(order.Type),
		Side:       pb.OrderSide(order.Side),
		Price:      order.Price,
		Size:       order.Size,
		Filled:     order.Size - order.RemainingSize,
		Remaining:  order.RemainingSize,
		Status:     s.mapOrderStatus(order.Status),
		UserId:     order.UserID,
		ClientId:   order.ClientID,
		Timestamp:  order.Timestamp.Unix(),
		PostOnly:   order.PostOnly,
		ReduceOnly: order.ReduceOnly,
	}
}

func (s *Server) convertTradeToProto(trade *lx.Trade) *pb.Trade {
	return &pb.Trade{
		TradeId:     trade.ID,
		Symbol:      s.orderBook.Symbol,
		Price:       trade.Price,
		Size:        trade.Size,
		BuyOrderId:  trade.BuyOrder,
		SellOrderId: trade.SellOrder,
		BuyerId:     trade.BuyUserID,
		SellerId:    trade.SellUserID,
		Timestamp:   trade.Timestamp.Unix(),
	}
}

func (s *Server) mapOrderStatus(status lx.OrderStatus) pb.OrderStatus {
	switch status {
	case lx.Open:
		return pb.OrderStatus_OPEN
	case lx.Filled:
		return pb.OrderStatus_FILLED
	case lx.Cancelled:
		return pb.OrderStatus_CANCELLED
	case lx.Rejected:
		return pb.OrderStatus_REJECTED
	default:
		return pb.OrderStatus_OPEN
	}
}

// StartGRPCServer starts the gRPC server
func StartGRPCServer(ctx context.Context, port int, orderBook *lx.OrderBook, logger log.Logger) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	server := NewServer(orderBook, logger, "node1", "1.0.0", "lux-mainnet")
	pb.RegisterLXDEXServiceServer(grpcServer, server)

	go func() {
		<-ctx.Done()
		grpcServer.GracefulStop()
	}()

	logger.Info("gRPC server started", "port", port)
	return grpcServer.Serve(lis)
}