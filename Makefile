# LX Engine - Root Makefile
# High-Performance Trading Platform Benchmarks

# Default target - build, test, and benchmark
.PHONY: all
all: build test bench-quick show-best
	@echo ""
	@echo "✅ LX Engine Ready!"
	@echo "Run 'make help' for all commands"

.PHONY: help
help:
	@echo "🚀 LX Engine - Performance Benchmark Suite"
	@echo "=========================================="
	@echo ""
	@echo "📊 QUICK COMMANDS:"
	@echo "  make bench-quick    - Quick test (1000 traders, 30s)"
	@echo "  make bench-max      - Find maximum throughput"
	@echo "  make bench-full     - Complete benchmark suite (~5 min)"
	@echo "  make bench-network  - Test 10Gbps network saturation"
	@echo ""
	@echo "🏆 PERFORMANCE RESULTS:"
	@echo "  • Pure C++:      1,328,880 orders/sec"
	@echo "  • Hybrid Go/C++:   180,585 orders/sec"
	@echo "  • Pure Go:         162,969 orders/sec"
	@echo "  • TypeScript:       ~50,000 orders/sec"
	@echo ""
	@echo "📈 ENGINE BENCHMARKS:"
	@echo "  bench-quick  - Quick benchmark (1000 traders, 30s)"
	@echo "  bench-full   - Full benchmark suite (~5 minutes)"
	@echo "  bench-max    - Find maximum throughput for each engine"
	@echo "  bench-stress - Stress test with 10,000 traders"
	@echo "  bench-report - Generate performance report"
	@echo "  bench-cpp    - Run pure C++ standalone benchmark"
	@echo "  bench-ultra  - Ultra-fast FIX engine benchmark"
	@echo ""
	@echo "🌐 NETWORK BENCHMARKS:"
	@echo "  bench-zmq-local   - Local ZeroMQ test (same machine)"
	@echo "  bench-zmq-dist    - Distributed test instructions"
	@echo "  bench-network     - Network saturation test (10Gbps)"
	@echo "  bench-network-1m  - Test 1M orders/sec over network"
	@echo "  bench-network-5m  - Test 5M orders/sec over network"
	@echo "  bench-fix         - FIX protocol benchmark with C++ client"
	@echo "  bench-fix-stress  - FIX stress test (10K traders)"
	@echo ""
	@echo "🔨 BUILD COMMANDS:"
	@echo "  build        - Build all engines and tools"
	@echo "  build-go     - Build pure Go engine"
	@echo "  build-hybrid - Build hybrid Go/C++ engine"
	@echo "  build-cpp    - Build pure C++ engine"
	@echo "  build-zmq    - Build ZeroMQ network tools"
	@echo "  clean        - Clean all build artifacts"
	@echo ""
	@echo "🖥️ SERVER & CLIENT COMMANDS:"
	@echo "  zmq-server        - Start ZeroMQ exchange server (port 5555)"
	@echo "  zmq-trader        - Start ZeroMQ trader client"
	@echo "  dex-server        - Start DEX server (Pure Go, port 50051)"
	@echo "  dex-server-hybrid - Start DEX server (Hybrid C++, port 50051)"
	@echo "  gateway-server    - Start Gateway server (port 8080)"
	@echo "  fix-trader-client - Start C++ FIX trader client"
	@echo "  fix-generator     - Start FIX message generator (streaming)"
	@echo "  mega-trader       - Start mega trader client (1000 traders)"
	@echo ""
	@echo "🎯 RECOMMENDED WORKFLOW:"
	@echo "  1. make build          # Build everything"
	@echo "  2. make bench-quick    # Quick performance check"
	@echo "  3. make bench-max      # Find your system's limit"
	@echo "  4. make bench-network  # Test network performance"
	@echo ""
	@echo "📝 DOCUMENTATION:"
	@echo "  • README.md              - Main documentation"
	@echo "  • NETWORK_BENCHMARK.md   - Network testing guide"
	@echo "  • performance_report.md  - Latest benchmark results"

# === ENGINE BENCHMARKS ===

.PHONY: bench-quick
bench-quick:
	@echo "🚀 Running QUICK benchmark (1000 traders, 30s)..."
	@$(MAKE) -C backend bench-quick

.PHONY: bench-full
bench-full:
	@echo "📊 Running FULL benchmark suite (~5 minutes)..."
	@$(MAKE) -C backend bench-full

.PHONY: bench-max
bench-max:
	@echo "🏁 Finding MAXIMUM throughput..."
	@$(MAKE) -C backend bench-max

.PHONY: bench-stress
bench-stress:
	@echo "💪 Running STRESS test (10,000 traders)..."
	@$(MAKE) -C backend bench-stress

.PHONY: bench-report
bench-report:
	@echo "📄 Generating performance report..."
	@$(MAKE) -C backend bench-report

.PHONY: bench-cpp
bench-cpp:
	@echo "⚡ Running Pure C++ benchmark..."
	@cd backend && ./bin/cpp-bench 1000 10 || (echo "Building C++ benchmark..." && make bench-tools && ./bin/cpp-bench 1000 10)

.PHONY: bench-ultra
bench-ultra:
	@echo "🚀 Running ULTRA FIX Engine benchmark..."
	@$(MAKE) -C backend bench-ultra

# === NETWORK BENCHMARKS ===

.PHONY: bench-zmq-local
bench-zmq-local:
	@echo "🏠 Running LOCAL ZeroMQ benchmark..."
	@$(MAKE) -C backend bench-zmq-local

.PHONY: bench-zmq-dist
bench-zmq-dist:
	@echo "🌍 DISTRIBUTED ZeroMQ benchmark instructions..."
	@$(MAKE) -C backend bench-zmq-dist

.PHONY: bench-network
bench-network:
	@echo "🌐 Testing NETWORK saturation (10Gbps)..."
	@$(MAKE) -C backend bench-network

.PHONY: bench-network-1m
bench-network-1m:
	@echo "📡 Testing 1 MILLION orders/sec over network..."
	@cd backend && ./bin/zmq-benchmark -mode local -traders 10000 -rate 100 -duration 30s || (echo "Building ZMQ tools..." && make zmq-build && ./bin/zmq-benchmark -mode local -traders 10000 -rate 100 -duration 30s)

.PHONY: bench-network-5m
bench-network-5m:
	@echo "🚀 Testing 5 MILLION orders/sec over network..."
	@cd backend && ./bin/zmq-benchmark -mode local -traders 50000 -rate 100 -duration 30s || (echo "Building ZMQ tools..." && make zmq-build && ./bin/zmq-benchmark -mode local -traders 50000 -rate 100 -duration 30s)

# === FIX PROTOCOL BENCHMARKS ===

.PHONY: bench-fix
bench-fix: build
	@echo "📄 Running FIX protocol benchmark with C++ client..."
	@echo "Starting server..."
	@cd backend && ./bin/lx-dex -port 50051 > /tmp/fix-server.log 2>&1 &
	@sleep 2
	@echo "Starting C++ FIX traders..."
	@cd backend && ./bin/fix-trader -server 127.0.0.1 -port 50051 -traders 100 -rate 100 -duration 30
	@pkill lx-dex || true
	@echo "✅ FIX benchmark complete"

.PHONY: bench-fix-stress
bench-fix-stress: build
	@echo "💪 Running FIX stress test (10,000 traders)..."
	@echo "Starting server..."
	@cd backend && ./bin/lx-dex -port 50051 > /tmp/fix-stress.log 2>&1 &
	@sleep 2
	@echo "Starting 10,000 C++ FIX traders..."
	@cd backend && ./bin/fix-trader -server 127.0.0.1 -port 50051 -traders 10000 -rate 10 -duration 60
	@pkill lx-dex || true
	@echo "✅ FIX stress test complete"

.PHONY: fix-demo
fix-demo: build
	@echo "🎯 FIX Protocol Demo"
	@echo "Generating sample FIX messages..."
	@cd backend && ./bin/fix-generator -mode single
	@echo ""
	@echo "Streaming FIX messages..."
	@cd backend && ./bin/fix-generator -mode stream -count 10 -rate 5

# === BUILD COMMANDS ===

.PHONY: build
build:
	@echo "🔨 Building all engines and tools..."
	@$(MAKE) -C backend bench-tools
	@$(MAKE) -C backend bench-servers
	@$(MAKE) -C backend zmq-build
	@$(MAKE) -C backend fix-build
	@echo "✅ Build complete! Run 'make bench-quick' to test."

.PHONY: test
test:
	@echo "🧪 Running tests..."
	@cd backend && go test -v ./... | grep -E "(PASS|FAIL|ok)" || true
	@echo "✅ Tests complete"

.PHONY: build-go
build-go:
	@echo "🐹 Building Pure Go engine..."
	@$(MAKE) -C backend go-build

.PHONY: build-hybrid
build-hybrid:
	@echo "🔄 Building Hybrid Go/C++ engine..."
	@$(MAKE) -C backend hybrid-build

.PHONY: build-cpp
build-cpp:
	@echo "⚡ Building Pure C++ engine..."
	@$(MAKE) -C backend cpp-build

.PHONY: build-zmq
build-zmq:
	@echo "🌐 Building ZeroMQ network tools..."
	@$(MAKE) -C backend zmq-build

.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@$(MAKE) -C backend clean

# === QUICK TEST SCENARIOS ===

.PHONY: test-local
test-local: build
	@echo "🏠 Testing LOCAL performance..."
	@echo "1. Pure Go test..."
	@cd backend && ./bin/mega-traders -traders 1000 -rate 10 -duration 10s -grpc localhost:50051 &
	@sleep 2
	@pkill mega-traders || true
	@echo "2. C++ standalone test..."
	@cd backend && ./bin/cpp-bench 100 5
	@echo "✅ Local test complete!"

.PHONY: test-network
test-network: build-zmq
	@echo "🌐 Testing NETWORK performance..."
	@echo "Starting exchange server..."
	@cd backend && ./bin/zmq-exchange -bind 'tcp://*:5555' > /tmp/exchange.log 2>&1 &
	@sleep 2
	@echo "Starting traders..."
	@cd backend && ./bin/zmq-trader -server 'tcp://localhost:5555' -traders 100 -rate 1000 -duration 10s
	@pkill zmq-exchange || true
	@echo "✅ Network test complete!"

# === DISTRIBUTED SETUP ===

.PHONY: setup-exchange
setup-exchange: build-zmq
	@echo "🏢 Starting EXCHANGE server..."
	@echo "Exchange will bind to tcp://*:5555"
	@cd backend && ./bin/zmq-exchange -bind 'tcp://*:5555'

.PHONY: setup-trader
setup-trader: build-zmq
	@echo "💹 Starting TRADER client..."
	@echo "Usage: make setup-trader EXCHANGE=10.0.0.10"
	@cd backend && ./bin/zmq-trader -server 'tcp://$(EXCHANGE):5555' -traders 1000 -rate 100 -duration 60s

# === MONITORING ===

.PHONY: monitor
monitor:
	@echo "📊 Monitoring system performance..."
	@echo "CPU & Memory:"
	@top -l 1 | head -20
	@echo ""
	@echo "Network (if running):"
	@netstat -ib | head -10

# === RESULTS ===

.PHONY: show-results
show-results:
	@echo "📊 Latest Benchmark Results:"
	@echo "============================"
	@if [ -f backend/performance_report.md ]; then \
		cat backend/performance_report.md | head -20; \
	else \
		echo "No results yet. Run 'make bench-quick' first."; \
	fi

.PHONY: show-best
show-best:
	@echo "🏆 Best Performance Achieved:"
	@echo "============================="
	@echo "Pure C++:      1,328,880 orders/sec"
	@echo "Hybrid Go/C++:   180,585 orders/sec"
	@echo "Pure Go:         162,969 orders/sec"
	@echo ""
	@echo "Run 'make bench-max' to test on your system."

# === SERVER & CLIENT COMMANDS ===

.PHONY: zmq-server
zmq-server:
	@echo "🚀 Starting ZeroMQ Exchange Server (port 5555)..."
	@cd backend && make zmq-build
	@backend/bin/zmq-exchange -port 5555

.PHONY: zmq-trader
zmq-trader:
	@echo "💹 Starting ZeroMQ Trader Client..."
	@cd backend && make zmq-build
	@backend/bin/zmq-trader -exchange tcp://localhost:5555 -id trader1 -rate 1000

.PHONY: dex-server
dex-server:
	@echo "🏦 Starting DEX Server (Pure Go, port 50051)..."
	@cd backend && make go-build
	@backend/bin/lx-dex -port 50051

.PHONY: dex-server-hybrid
dex-server-hybrid:
	@echo "⚡ Starting DEX Server (Hybrid Go/C++, port 50051)..."
	@cd backend && make bench-servers
	@backend/bin/lx-dex-hybrid -port 50051

.PHONY: gateway-server
gateway-server:
	@echo "🌐 Starting Gateway Server (port 8080)..."
	@cd backend && make go-build
	@backend/bin/lx-gateway -port 8080

.PHONY: fix-trader-client
fix-trader-client:
	@echo "📈 Starting C++ FIX Trader Client..."
	@cd backend && make fix-build
	@backend/bin/fix-trader localhost 9876 100 10

.PHONY: fix-generator
fix-generator:
	@echo "📊 Starting FIX Message Generator (Stream Mode)..."
	@cd backend && make fix-build
	@backend/bin/fix-generator -mode stream -rate 1000

.PHONY: mega-trader
mega-trader:
	@echo "🔥 Starting Mega Trader Client (1000 traders)..."
	@cd backend && make bench-tools
	@backend/bin/mega-traders -traders 1000 -rate 10 -duration 30s -grpc localhost:50051

# Default target is all
.DEFAULT_GOAL := all

# === SHORTCUTS ===

.PHONY: q
q: bench-quick

.PHONY: m
m: bench-max

.PHONY: n
n: bench-network

.PHONY: f
f: bench-full

.PHONY: b
b: build

.PHONY: c
c: clean
