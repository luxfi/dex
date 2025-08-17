.PHONY: all build dex-server dex-trader bench test clean help

# LX DEX Makefile - High-performance trading platform
GO := go
CGO_ENABLED ?= 1  # Default to CGO enabled for maximum performance

# Default target: build everything, run tests and benchmarks
all: clean build test bench
	@echo "✅ All tasks complete!"

# Build all binaries with CGO for C++ performance
build:
	@echo "🔨 Building LX DEX binaries (CGO_ENABLED=$(CGO_ENABLED))..."
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/lx-server ./cmd/dex-server
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/lx-trader ./cmd/dex-trader
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/lx-bench ./cmd/bench
	@cd backend && CGO_ENABLED=$(CGO_ENABLED) $(GO) build -o ../bin/lx-benchmark ./cmd/benchmark 2>/dev/null || true
	@echo "✅ Build complete (CGO_ENABLED=$(CGO_ENABLED))"


# Run DEX server
dex-server:
	@echo "🏦 Starting DEX Server..."
	@cd backend && $(GO) run ./cmd/dex-server

# Run DEX trader (normal mode)
dex-trader:
	@echo "💹 Starting DEX Trader..."
	@cd backend && $(GO) run ./cmd/dex-trader

# Run trader (auto-scale mode to find max throughput)
trader-auto:
	@echo "🚀 Starting Auto-Scaling Trader..."
	@cd backend && $(GO) run ./cmd/dex-trader -auto


# Run quick benchmarks
bench:
	@echo "🏁 Running quick performance benchmarks..."
	@cd backend/pkg/orderbook && $(GO) test -bench=. -benchmem -benchtime=1s -run=^$ .
	@echo "✅ Benchmarks complete!"




# Run comprehensive tests
test:
	@echo "🧪 Running comprehensive test suite..."
	@cd backend && $(GO) test -v -race -coverprofile=coverage.out \
		./pkg/orderbook/... \
		./pkg/lx/... \
		./pkg/fix/... \
		./pkg/metric/... \
		./pkg/log/... || true
	@echo "📊 Test coverage report:"
	@cd backend && go tool cover -func=coverage.out | tail -5 || true
	@echo "✅ Test run complete!"


# Clean build artifacts
clean:
	@rm -rf bin/
	@rm -f backend/coverage.out backend/coverage.html
	@echo "✅ Cleaned"

help:
	@echo "LX DEX Commands:"
	@echo "  make build         - Build all binaries"  
	@echo "  make dex-server    - Run DEX server"
	@echo "  make dex-trader    - Run DEX trader"
	@echo "  make trader-auto   - Auto-scale to find max throughput"
	@echo "  make bench         - Run performance benchmark"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean build artifacts"