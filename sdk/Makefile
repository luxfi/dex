# LX Trading SDK - Unified Build & Test Makefile
# Supports: Go, TypeScript, Python, Rust, C++

.PHONY: all test build clean help
.PHONY: test-go test-ts test-py test-rust test-cpp
.PHONY: build-go build-ts build-py build-rust build-cpp
.PHONY: lint fmt install

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	NPROC := $(shell sysctl -n hw.ncpu)
else
	NPROC := $(shell nproc 2>/dev/null || echo 4)
endif

# Colors
GREEN  := \033[0;32m
YELLOW := \033[1;33m
BLUE   := \033[0;34m
RED    := \033[0;31m
NC     := \033[0m

#------------------------------------------------------------------------------
# Main targets
#------------------------------------------------------------------------------

all: build test ## Build and test all SDKs

help: ## Show this help
	@echo "LX Trading SDK - Build & Test System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'

#------------------------------------------------------------------------------
# Build targets
#------------------------------------------------------------------------------

build: build-go build-ts build-rust build-cpp ## Build all SDKs
	@echo "$(GREEN)All SDKs built successfully$(NC)"

build-go: ## Build Go SDK
	@echo "$(BLUE)Building Go SDK...$(NC)"
	cd lx-trading-go && go build ./...

build-ts: ## Build TypeScript SDK
	@echo "$(BLUE)Building TypeScript SDK...$(NC)"
	cd lx-trading-ts && npm run build

build-py: ## Build Python SDK
	@echo "$(BLUE)Building Python SDK...$(NC)"
	cd lx-trading-py && pip install -e . 2>/dev/null || true

build-rust: ## Build Rust SDK
	@echo "$(BLUE)Building Rust SDK...$(NC)"
	cd lx-trading-core && cargo build --release

build-cpp: ## Build C++ SDK
	@echo "$(BLUE)Building C++ SDK...$(NC)"
	cd lx-trading-cpp && mkdir -p build && cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		make -j$(NPROC)

#------------------------------------------------------------------------------
# Test targets
#------------------------------------------------------------------------------

test: test-go test-ts test-py test-rust test-cpp ## Run all SDK tests
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  All SDK tests completed$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"

test-go: ## Run Go SDK tests
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Running Go SDK tests$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	cd lx-trading-go && go test -v ./...

test-ts: ## Run TypeScript SDK tests
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Running TypeScript SDK tests$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	cd lx-trading-ts && npm test

test-py: ## Run Python SDK tests
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Running Python SDK tests$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	cd lx-trading-py && PYTHONPATH=./python python -m pytest tests/ -v

test-rust: ## Run Rust SDK tests
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Running Rust SDK tests$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	cd lx-trading-core && cargo test --all-features

test-cpp: ## Run C++ SDK tests
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Running C++ SDK tests$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	cd lx-trading-cpp/build && ./lx_trading_tests 2>/dev/null || \
		(cd ../.. && $(MAKE) build-cpp && cd lx-trading-cpp/build && ./lx_trading_tests)

#------------------------------------------------------------------------------
# Lint & Format
#------------------------------------------------------------------------------

lint: lint-go lint-ts lint-rust ## Lint all SDKs
	@echo "$(GREEN)All linting passed$(NC)"

lint-go: ## Lint Go SDK
	cd lx-trading-go && go vet ./...

lint-ts: ## Lint TypeScript SDK
	cd lx-trading-ts && npx tsc --noEmit

lint-rust: ## Lint Rust SDK
	cd lx-trading-core && cargo clippy -- -D warnings

fmt: fmt-go fmt-rust ## Format all code
	@echo "$(GREEN)All code formatted$(NC)"

fmt-go: ## Format Go code
	cd lx-trading-go && go fmt ./...

fmt-rust: ## Format Rust code
	cd lx-trading-core && cargo fmt

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------

install: install-deps ## Install all dependencies

install-deps: ## Install dependencies for all SDKs
	@echo "$(BLUE)Installing Go dependencies...$(NC)"
	cd lx-trading-go && go mod download
	@echo "$(BLUE)Installing TypeScript dependencies...$(NC)"
	cd lx-trading-ts && npm ci
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	cd lx-trading-py && pip install pytest pytest-asyncio aiohttp websockets ccxt pydantic
	@echo "$(GREEN)All dependencies installed$(NC)"

#------------------------------------------------------------------------------
# Clean
#------------------------------------------------------------------------------

clean: ## Clean all build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf lx-trading-go/dist
	rm -rf lx-trading-ts/dist lx-trading-ts/node_modules
	rm -rf lx-trading-py/dist lx-trading-py/build lx-trading-py/*.egg-info
	rm -rf lx-trading-core/target
	rm -rf lx-trading-cpp/build
	@echo "$(GREEN)Clean complete$(NC)"

#------------------------------------------------------------------------------
# Coverage
#------------------------------------------------------------------------------

coverage: coverage-go coverage-ts coverage-py coverage-rust ## Generate coverage for all SDKs

coverage-go: ## Go coverage report
	cd lx-trading-go && go test -coverprofile=coverage.out ./... && \
		go tool cover -html=coverage.out -o coverage.html

coverage-ts: ## TypeScript coverage report
	cd lx-trading-ts && npm test -- --experimental-test-coverage

coverage-py: ## Python coverage report
	cd lx-trading-py && PYTHONPATH=./python pytest tests/ --cov=lx_trading --cov-report=html

coverage-rust: ## Rust coverage report
	cd lx-trading-core && cargo tarpaulin --out Html 2>/dev/null || cargo test

#------------------------------------------------------------------------------
# Release helpers
#------------------------------------------------------------------------------

version: ## Show version info
	@echo "Go SDK:"
	@grep -E "^module" lx-trading-go/go.mod
	@echo ""
	@echo "TypeScript SDK:"
	@grep '"version"' lx-trading-ts/package.json | head -1
	@echo ""
	@echo "Python SDK:"
	@grep 'version' lx-trading-py/pyproject.toml | head -1
	@echo ""
	@echo "Rust SDK:"
	@grep '^version' lx-trading-core/Cargo.toml | head -1
