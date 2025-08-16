.PHONY: all build server trader bench test clean help

# Simple DEX Makefile - ONE way to do everything
GO := go
CGO_ENABLED ?= 0

all: build

# Build all binaries
build:
	@echo "🔨 Building DEX binaries..."
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/server ./cmd/dex-server
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/trader ./cmd/dex-trader
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/bench ./cmd/bench
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/benchmark ./cmd/benchmark 2>/dev/null || true
	@echo "✅ Build complete (CGO_ENABLED=$(CGO_ENABLED))"

# Run server
server:
	@echo "🏦 Starting DEX Server..."
	@cd backend && $(GO) run ./cmd/dex-server

# Run trader (normal mode)
trader:
	@echo "💹 Starting DEX Trader..."
	@cd backend && $(GO) run ./cmd/dex-trader

# Run trader (auto-scale mode to find max throughput)
trader-auto:
	@echo "🚀 Starting Auto-Scaling Trader..."
	@cd backend && $(GO) run ./cmd/dex-trader -auto

# Run benchmark
bench:
	@echo "🏁 Running Benchmark (Pure Go)..."
	@cd backend && CGO_ENABLED=0 $(GO) run ./cmd/bench -iter 50000

# Run benchmark comparing Go vs C++
bench-compare:
	@echo "🏁 Comparing Pure Go vs C++ Performance..."
	@cd backend && CGO_ENABLED=1 $(GO) run ./cmd/bench -iter 50000 -impl auto

# Build C++ library
cpp:
	@echo "🔧 Building C++ library..."
	@cd backend/cpp && g++ -O3 -std=c++17 -fPIC -shared -o libfixengine.so fix_engine.cpp
	@echo "✅ C++ library built"

# Run tests
test:
	@echo "🧪 Running tests..."
	@cd backend && $(GO) test -v ./pkg/...

# Run tests with coverage
test-coverage:
	@echo "📊 Running tests with coverage..."
	@cd backend && $(GO) test -v -cover -coverprofile=coverage.out ./pkg/...
	@cd backend && $(GO) tool cover -html=coverage.out -o coverage.html
	@echo "✅ Coverage report: backend/coverage.html"

# Generate OpenAPI clients
generate:
	@echo "🔄 Generating OpenAPI clients..."
	@cd backend && $(GO) generate ./...
	@echo "✅ Client generation complete"

# Clean build artifacts
clean:
	@rm -rf bin/
	@rm -f backend/coverage.out backend/coverage.html
	@rm -f backend/cpp/*.so backend/cpp/*.dylib
	@echo "✅ Cleaned"

# Run CI tests locally
ci:
	@echo "🎯 Running CI tests locally..."
	@make clean
	@make build
	@make test
	@make bench
	@echo "✅ CI tests passed"

help:
	@echo "DEX Commands:"
	@echo "  make build        - Build all binaries"
	@echo "  make server       - Run DEX server"
	@echo "  make trader       - Run trader client"
	@echo "  make trader-auto  - Run auto-scaling trader"
	@echo "  make bench        - Run benchmark (Go only)"
	@echo "  make bench-compare - Compare Go vs C++ performance"
	@echo "  make cpp          - Build C++ library"
	@echo "  make test         - Run tests"
	@echo "  make test-coverage - Run tests with coverage"
	@echo "  make generate     - Generate OpenAPI clients"
	@echo "  make ci           - Run CI tests locally"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Environment:"
	@echo "  CGO_ENABLED=$(CGO_ENABLED) - Enable/disable CGO (0 or 1)"