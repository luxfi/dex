package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	// pb "github.com/luxexchange/engine/backend/pkg/proto/engine"
	// "google.golang.org/grpc"
	// "google.golang.org/grpc/credentials/insecure"
)

var (
	grpcEndpoint = flag.String("grpc-endpoint", "localhost:50051", "gRPC server endpoint")
	httpPort     = flag.Int("http-port", 8080, "HTTP gateway port")
)

func main() {
	flag.Parse()

	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create gRPC-Gateway mux
	mux := runtime.NewServeMux()
	
	// Register gRPC service handler
	// TODO: Add HTTP annotations to proto file for gRPC-Gateway
	_ = mux
	log.Printf("Gateway not yet implemented - needs HTTP annotations in proto file")

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", *httpPort),
		Handler:      allowCORS(mux),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	// Handle shutdown gracefully
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down gateway...")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		server.Shutdown(shutdownCtx)
		cancel()
	}()

	log.Printf("LX Gateway listening on http://localhost:%d", *httpPort)
	log.Printf("Proxying to gRPC server at %s", *grpcEndpoint)
	
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Failed to serve: %v", err)
	}
}

func allowCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		h.ServeHTTP(w, r)
	})
}