# LX DEX Makefile
# Complete build, test, and deployment automation

SHELL := /bin/bash
.PHONY: all build test clean help

# Version and build info
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Go parameters
GO := go
GOBUILD := $(GO) build
GOCLEAN := $(GO) clean
GOTEST := $(GO) test
GOGET := $(GO) get
GOMOD := $(GO) mod
GOVET := $(GO) vet
GOFMT := gofmt

# Build parameters
CGO_ENABLED ?= 0
LDFLAGS := -ldflags "-X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT)"

# Binary output
BINARY_NAME := luxd
BINARY_DIR := bin

# Test parameters
TEST_TIMEOUT := 30s
BENCH_TIME := 10s

# Default target
all: clean fmt vet test build
	@echo "✅ Build complete!"

help:
	@echo "LX DEX Makefile Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make build         - Build all binaries"
	@echo "  make test          - Run all tests"
	@echo "  make bench         - Run benchmarks"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Running:"
	@echo "  make run           - Run single node"
	@echo "  make run-cluster   - Run 3-node cluster"
	@echo ""
	@echo "Quality:"
	@echo "  make fmt           - Format code"
	@echo "  make vet           - Run go vet"
	@echo "  make lint          - Run linters"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make up            - Start with docker-compose"
	@echo "  make down          - Stop docker-compose"

# Build targets
build: build-go build-tools

build-go:
	@echo "🔨 Building luxd..."
	@mkdir -p $(BINARY_DIR)
	@CGO_ENABLED=$(CGO_ENABLED) $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME) ./cmd/luxd
	@echo "✅ luxd built successfully!"

build-cpp:
	@echo "🔨 Building with C++ optimizations..."
	@mkdir -p $(BINARY_DIR)
	@CGO_ENABLED=1 $(GOBUILD) $(LDFLAGS) -tags cpp -o $(BINARY_DIR)/$(BINARY_NAME)-cpp ./cmd/luxd

build-gpu:
	@echo "🔨 Building with GPU support..."
	@mkdir -p $(BINARY_DIR)
	@CGO_ENABLED=1 $(GOBUILD) $(LDFLAGS) -tags "cpp gpu mlx" -o $(BINARY_DIR)/$(BINARY_NAME)-gpu ./cmd/luxd

build-tools:
	@echo "🔨 Building tools..."
	@$(GOBUILD) -o $(BINARY_DIR)/lx-trader ./cmd/trader 2>/dev/null || true
	@$(GOBUILD) -o $(BINARY_DIR)/lx-metrics ./cmd/lx-metrics 2>/dev/null || true

# Test targets
test:
	@echo "🧪 Running tests..."
	@$(GOTEST) -v -timeout $(TEST_TIMEOUT) ./pkg/lx/...
	@$(GOTEST) -v -timeout $(TEST_TIMEOUT) ./pkg/api/...
	@echo "✅ Tests passed!"

test-all:
	@./scripts/test-all.sh

test-race:
	@$(GOTEST) -race -timeout 2m ./pkg/...

# Benchmark targets
bench:
	@echo "🏁 Running benchmarks..."
	@$(GOTEST) -bench=. -benchmem -benchtime=$(BENCH_TIME) -run=^$$ ./pkg/lx/...

bench-all:
	@./scripts/run-comprehensive-benchmark.sh

# Run targets
run:
	@$(BINARY_DIR)/$(BINARY_NAME) --data-dir ~/.lxd --http-port 8080

run-cluster:
	@./scripts/run-lx-cluster.sh

run-dev:
	@air -c .air.toml 2>/dev/null || $(BINARY_DIR)/$(BINARY_NAME)

# Code quality
fmt:
	@$(GOFMT) -s -w .

vet:
	@$(GOVET) ./...

lint:
	@golangci-lint run --timeout 5m 2>/dev/null || echo "Install golangci-lint for linting"

# Docker targets
docker-build:
	@docker build -f docker/backend/Dockerfile -t lxdex:$(VERSION) .

up:
	@docker-compose -f docker/compose.yml up -d

down:
	@docker-compose -f docker/compose.yml down

logs:
	@docker-compose -f docker/compose.yml logs -f

# Utility targets
deps:
	@$(GOMOD) download
	@$(GOMOD) tidy

clean:
	@$(GOCLEAN)
	@rm -rf $(BINARY_DIR)
	@rm -f coverage.out coverage.html

version:
	@echo "Version: $(VERSION)"
	@echo "Commit:  $(GIT_COMMIT)"
	@echo "Built:   $(BUILD_TIME)"

# CI target
ci: clean fmt vet test build
	@echo "✅ CI pipeline complete!"

# 3-node benchmark (legacy compatibility)
3node-bench:
	@echo "🌐 Starting 3-node FPC network benchmark..."
	@./scripts/run-3node-bench.sh
	@echo "✅ 3-node benchmark complete!"

# Run demo
demo:
	@./scripts/demo.sh

# Test MLX acceleration
test-mlx:
	@./scripts/test-mlx.sh

# Build with MLX support
build-mlx: build-gpu

# Test CUDA acceleration
test-cuda:
	@./scripts/test-cuda.sh

# Docker CUDA build
docker-cuda:
	@docker build -f docker/backend/Dockerfile.cuda -t lxdex-cuda:$(VERSION) .

# Ensure 100% test passing
test-100:
	@./scripts/ensure-100-pass.sh

# Run performance tuning
perf-tune:
	@sudo ./scripts/perf-tune.sh

# Development setup
setup:
	@./scripts/setup.sh

# Install development tools
tools:
	@go install github.com/cosmtrek/air@latest
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install github.com/securego/gosec/v2/cmd/gosec@latest

# Generate protobuf code
proto:
	@protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/*.proto

# SDK targets
sdk-typescript:
	@cd sdk/typescript && npm install && npm run build

sdk-python:
	@cd sdk/python && pip install -e .

sdk-go:
	@cd sdk/go && go mod tidy && go build ./...

sdk-all: sdk-typescript sdk-python sdk-go

# Deploy targets
deploy-staging:
	@./scripts/deploy.sh staging deploy

deploy-production:
	@./scripts/deploy.sh production deploy

rollback:
	@./scripts/deploy.sh production rollback

# Monitoring
metrics:
	@open http://localhost:9090

grafana:
	@open http://localhost:3001

# Performance benchmarks
bench-10gbps:
	@./scripts/benchmark-10gbps.sh

bench-zmq:
	@./scripts/run-zmq-benchmark.sh

bench-comprehensive:
	@./scripts/run-comprehensive-benchmark.sh

# Network operations
run-testnet:
	@./scripts/run-lux-testnet.sh

run-local:
	@./scripts/run-local-network.sh

run-fpc:
	@./scripts/run-fpc-network.sh

run-qzmq:
	@./scripts/run-qzmq-network.sh

# Database operations
db-migrate:
	@migrate -path migrations -database "postgres://lxdex:lxdex123@localhost:5432/lxdex?sslmode=disable" up

db-rollback:
	@migrate -path migrations -database "postgres://lxdex:lxdex123@localhost:5432/lxdex?sslmode=disable" down 1

db-reset:
	@migrate -path migrations -database "postgres://lxdex:lxdex123@localhost:5432/lxdex?sslmode=disable" drop -f
	@make db-migrate

# Complete verification
verify-all: deps fmt vet lint test-100 bench
	@echo "✅ Complete verification passed!"

# Install everything for development
install: deps tools
	@echo "✅ Development environment ready!"

# Security scan
security:
	@gosec -fmt json -out security-report.json ./...

# Coverage report
coverage:
	@go test -v -race -coverprofile=coverage.out -covermode=atomic ./pkg/...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Production build
prod: clean verify-all build docker-build
	@echo "✅ Production build complete!"

# Quick development cycle
dev: fmt test-fast build-go run-dev

# Fast tests only
test-fast:
	@go test -short -timeout 30s ./pkg/...

# Integration tests
test-integration:
	@go test -tags integration -timeout 5m ./test/integration/...

# E2E tests
test-e2e:
	@./scripts/test-all.sh e2e

# Load testing
load-test:
	@k6 run test/load/scenario.js

# Initialize project
init:
	@chmod +x scripts/*.sh
	@mkdir -p bin
	@echo "✅ Project initialized!"

.PHONY: all build test bench clean help run docker deploy sdk db verify install init demo test-mlx test-cuda docker-cuda