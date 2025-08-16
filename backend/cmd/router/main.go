package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	pb "github.com/luxexchange/engine/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

var (
	port          = flag.Int("port", 50050, "Router port")
	goEngine      = flag.String("go-engine", "localhost:50051", "Go engine endpoint")
	hybridEngine  = flag.String("hybrid-engine", "localhost:50052", "Hybrid engine endpoint")
	cppEngine     = flag.String("cpp-engine", "localhost:50053", "C++ engine endpoint")
	defaultEngine = flag.String("default", "hybrid", "Default engine: go, hybrid, or cpp")
)

type Router struct {
	pb.UnimplementedEngineServiceServer
	clients map[string]pb.EngineServiceClient
	conns   map[string]*grpc.ClientConn
	mu      sync.RWMutex
	defaultEngine string
}

func NewRouter() (*Router, error) {
	r := &Router{
		clients: make(map[string]pb.EngineServiceClient),
		conns:   make(map[string]*grpc.ClientConn),
		defaultEngine: *defaultEngine,
	}

	// Connect to engines
	engines := map[string]string{
		"go":     *goEngine,
		"hybrid": *hybridEngine,
		"cpp":    *cppEngine,
	}

	for name, endpoint := range engines {
		if endpoint == "" {
			continue
		}
		
		conn, err := grpc.Dial(endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Printf("Warning: Failed to connect to %s engine at %s: %v", name, endpoint, err)
			continue
		}
		
		r.conns[name] = conn
		r.clients[name] = pb.NewEngineServiceClient(conn)
		log.Printf("Connected to %s engine at %s", name, endpoint)
	}

	if len(r.clients) == 0 {
		return nil, fmt.Errorf("no engines available")
	}

	return r, nil
}

func (r *Router) getClient(engine string) (pb.EngineServiceClient, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if engine == "" {
		engine = r.defaultEngine
	}

	client, ok := r.clients[engine]
	if !ok {
		// Fallback to any available engine
		for _, c := range r.clients {
			return c, nil
		}
		return nil, status.Errorf(codes.Unavailable, "no engines available")
	}

	return client, nil
}

func (r *Router) SubmitOrder(ctx context.Context, req *pb.SubmitOrderRequest) (*pb.SubmitOrderResponse, error) {
	// Route based on metadata or default
	engine := getEngineFromContext(ctx)
	client, err := r.getClient(engine)
	if err != nil {
		return nil, err
	}
	
	return client.SubmitOrder(ctx, req)
}

func (r *Router) CancelOrder(ctx context.Context, req *pb.CancelOrderRequest) (*pb.CancelOrderResponse, error) {
	engine := getEngineFromContext(ctx)
	client, err := r.getClient(engine)
	if err != nil {
		return nil, err
	}
	
	return client.CancelOrder(ctx, req)
}

func (r *Router) GetOrderBook(ctx context.Context, req *pb.GetOrderBookRequest) (*pb.GetOrderBookResponse, error) {
	engine := getEngineFromContext(ctx)
	client, err := r.getClient(engine)
	if err != nil {
		return nil, err
	}
	
	return client.GetOrderBook(ctx, req)
}

func (r *Router) StreamOrderBook(req *pb.StreamOrderBookRequest, stream pb.EngineService_StreamOrderBookServer) error {
	engine := getEngineFromContext(stream.Context())
	client, err := r.getClient(engine)
	if err != nil {
		return err
	}
	
	// Create client stream
	clientStream, err := client.StreamOrderBook(stream.Context(), req)
	if err != nil {
		return err
	}
	
	// Proxy messages
	for {
		msg, err := clientStream.Recv()
		if err != nil {
			return err
		}
		if err := stream.Send(msg); err != nil {
			return err
		}
	}
}

func (r *Router) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	for name, conn := range r.conns {
		conn.Close()
		log.Printf("Closed connection to %s engine", name)
	}
}

func getEngineFromContext(ctx context.Context) string {
	// Could extract from metadata
	return ""
}

func main() {
	flag.Parse()

	router, err := NewRouter()
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Close()

	// Create gRPC server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterEngineServiceServer(grpcServer, router)
	
	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Handle shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		log.Println("Shutting down router...")
		grpcServer.GracefulStop()
	}()

	log.Printf("LX Router listening on %v", lis.Addr())
	log.Printf("Default engine: %s", router.defaultEngine)
	log.Printf("Available engines: %s", strings.Join(getKeys(router.clients), ", "))
	
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

func getKeys(m map[string]pb.EngineServiceClient) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}