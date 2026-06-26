# dexd Makefile — the ONE D-Chain DEX binary.
#
# dexd runs the DEX two ways (one binary, two modes):
#   dexd            run as a luxd rpcchainvm plugin (luxd execs it, no args)
#   dexd run        run the standalone D-Chain venue (ZAP + consensus sealer)
#
# Default build is CPU (CGO_ENABLED=0) — no GPU/MLX required. The GPU matcher
# (CGO_ENABLED=1, pkg/lx cuda kernels) is built only by Dockerfile.dvenue.

SHELL := /bin/bash
.PHONY: all build run test test-race bench vet fmt lint deps clean coverage \
        docker proto sdk-typescript sdk-python sdk-go sdk-all ci help

# Version / build info
VERSION    := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Go
GO          := go
CGO_ENABLED ?= 0
BINARY      := dexd
BINARY_DIR  := bin
IMAGE       := ghcr.io/luxfi/dex
TEST_TIMEOUT := 60s

export CGO_ENABLED

all: fmt vet test build

help:
	@echo "dexd Makefile:"
	@echo "  make build       - Build dexd (CPU, CGO_ENABLED=0) -> $(BINARY_DIR)/$(BINARY)"
	@echo "  make run         - Build and run the standalone venue (dexd run)"
	@echo "  make test        - Run package tests"
	@echo "  make vet         - go vet ./..."
	@echo "  make fmt         - gofmt -s -w ."
	@echo "  make lint        - golangci-lint run"
	@echo "  make docker      - Build the canonical image ($(IMAGE)) from ./Dockerfile"
	@echo "  make clean       - Remove build artifacts"

# Build the one binary (CPU).
build:
	@echo "Building $(BINARY) (CGO_ENABLED=$(CGO_ENABLED))..."
	@mkdir -p $(BINARY_DIR)
	@$(GO) build -trimpath -ldflags "-s -w -X main.Version=$(VERSION) -X main.GitCommit=$(GIT_COMMIT)" \
		-o $(BINARY_DIR)/$(BINARY) ./cmd/dexd
	@echo "Built $(BINARY_DIR)/$(BINARY)"

# Run the standalone venue.
run: build
	@$(BINARY_DIR)/$(BINARY) run

# Tests
test:
	@$(GO) test -timeout $(TEST_TIMEOUT) ./pkg/...

test-race:
	@$(GO) test -race -timeout 2m ./pkg/...

bench:
	@$(GO) test -bench=. -benchmem -benchtime=10s -run=^$$ ./pkg/lx/...

# Quality
vet:
	@$(GO) vet ./...

fmt:
	@gofmt -s -w .

lint:
	@golangci-lint run --timeout 5m 2>/dev/null || echo "Install golangci-lint for linting"

coverage:
	@$(GO) test -coverprofile=coverage.out -covermode=atomic ./pkg/...
	@$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Modules
deps:
	@$(GO) mod download
	@$(GO) mod tidy

clean:
	@$(GO) clean
	@rm -rf $(BINARY_DIR) coverage.out coverage.html

# Canonical image (CPU dexd).
docker:
	@docker build -f Dockerfile -t $(IMAGE):$(VERSION) .

# Protobuf
proto:
	@protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/*.proto

# SDKs
sdk-typescript:
	@cd sdk/typescript && npm install && npm run build

sdk-python:
	@cd sdk/python && pip install -e .

sdk-go:
	@cd sdk/go && go mod tidy && go build ./...

sdk-all: sdk-typescript sdk-python sdk-go

# CI pipeline
ci: fmt vet test build
	@echo "CI complete"
