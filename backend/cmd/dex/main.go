package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/luxfi/dex/backend/pkg/engine"
	"github.com/luxfi/dex/backend/pkg/orderbook"
	pb "github.com/luxfi/dex/backend/pkg/proto/engine"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

var (
	port       = flag.Int("port", 50051, "The server port")
	mode       = flag.String("mode", "hybrid", "Engine mode: go, hybrid, or cpp")
	cgoEnabled = os.Getenv("CGO_ENABLED") == "1"
)

func main() {
	flag.Parse()

	log.Printf("Starting LX DEX Engine on port %d (mode: %s, CGO: %v)", *port, *mode, cgoEnabled)

	// Create engine configuration
	config := engine.Config{
		Mode: *mode,
		OrderBook: orderbook.Config{
			MaxOrdersPerLevel: 10000,
			PricePrecision:    7,
		},
	}

	// Create engine instance
	eng, err := engine.New(config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}

	// Create gRPC server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()

	// Register engine service
	pb.RegisterEngineServiceServer(grpcServer, eng)

	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Handle shutdown gracefully
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down engine...")
		grpcServer.GracefulStop()
	}()

	log.Printf("LX DEX Engine listening on %v", lis.Addr())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
