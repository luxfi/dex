#!/bin/bash

# LX - Run Script
# This script builds and runs various components of the LX

set -e

# Ensure CGO is disabled for compatibility
export CGO_ENABLED=0

case "$1" in
  "demo")
    echo "🚀 Running LX Demo..."
    go run ./cmd/demo
    ;;
    
  "server")
    echo "🌐 Starting DEX Server..."
    echo "Usage: $0 server [port]"
    PORT=${2:-8080}
    go run ./cmd/dex-server -port $PORT
    ;;
    
  "memory")
    echo "📊 Running Memory Analysis..."
    go run ./cmd/memory-analysis
    ;;
    
  "test")
    echo "🧪 Running Tests..."
    go test ./pkg/... -short -timeout 30s
    ;;
    
  "build")
    echo "🔨 Building all packages..."
    go build ./pkg/...
    echo "✅ Build successful!"
    ;;
    
  "benchmark")
    echo "⚡ Running Benchmarks..."
    go test ./pkg/lx -bench=. -run=^$ -benchtime=10s
    ;;
    
  *)
    echo "LX - Ultra High-Performance Order Book"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  demo       - Run interactive order book demo"
    echo "  server     - Start DEX server (default port: 8080)"
    echo "  memory     - Analyze memory usage for 1M markets"
    echo "  test       - Run all tests"
    echo "  build      - Build all packages"
    echo "  benchmark  - Run performance benchmarks"
    echo ""
    echo "Examples:"
    echo "  ./run.sh demo"
    echo "  ./run.sh server 9090"
    echo "  ./run.sh test"
    ;;
esac