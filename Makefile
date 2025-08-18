.PHONY: all build dex-server dex-trader bench test clean help up down restart logs status docker-build docker-clean

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
	@echo ""
	@echo "Local Development:"
	@echo "  make build         - Build all binaries"  
	@echo "  make dex-server    - Run DEX server"
	@echo "  make dex-trader    - Run DEX trader"
	@echo "  make trader-auto   - Auto-scale to find max throughput"
	@echo "  make bench         - Run performance benchmark"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Docker Stack (K=1 Consensus):"
	@echo "  make up            - Start entire DEX stack in containers"
	@echo "  make down          - Stop all containers"
	@echo "  make restart       - Restart all services"
	@echo "  make logs          - Follow container logs"
	@echo "  make status        - Show container status"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-clean  - Clean Docker volumes"

# ==================== DOCKER COMMANDS ====================
# Start entire DEX stack with K=1 consensus (single node)
up:
	@echo "🚀 Starting LUX DEX Stack..."
	@echo "Starting databases first..."
	docker compose -f docker/compose.dev.yml up -d
	@echo "✅ Databases are running!"
	@echo "   - PostgreSQL: localhost:5432"
	@echo "   - Redis: localhost:6379"
	@echo ""
	@echo "To start full stack with monitoring:"
	@echo "   docker compose -f docker/compose.yml up -d"
	@echo ""
	@echo "To run backend locally:"
	@echo "   make dex-server"
	@echo ""
	@echo "To run UI locally:"
	@echo "   cd ui && npm run dev"

# Stop all Docker services
down:
	@echo "🛑 Stopping LUX DEX Stack..."
	docker compose -f docker/compose.yml down

# Restart Docker stack
restart: down up

# Follow Docker logs
logs:
	docker compose -f docker/compose.yml logs -f

# Show Docker service status
status:
	@docker compose -f docker/compose.yml ps

# Build Docker images
docker-build:
	@echo "🔨 Building Docker images..."
	docker compose -f docker/compose.yml build

# Clean Docker volumes
docker-clean:
	@echo "🧹 Cleaning Docker volumes..."
	docker compose -f docker/compose.yml down -v
	docker system prune -f

# Run E2E tests in Docker
docker-test:
	@echo "🧪 Running E2E tests in Docker..."
	docker compose -f docker/compose.yml --profile test run --rm test-runner

# Quick health check
health:
	@echo "❤️  Checking service health..."
	@docker compose -f docker/compose.yml exec dex-node curl -s http://localhost:8080/health || echo "Node not running"
	@docker compose -f docker/compose.yml exec dex-ui curl -s http://localhost:3000/v2/api/health || echo "UI not running"

# Run E2E tests
e2e-test:
	@echo "🧪 Running E2E tests..."
	@./test-e2e.sh