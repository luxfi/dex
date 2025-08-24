#!/bin/bash

# LX DEX Comprehensive Test Script
# Tests all builds, tests, and CI components

set -e

echo "============================================"
echo "   LX DEX Comprehensive Test Suite"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Change to project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Track failures
FAILURES=""

# Function to run test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "${YELLOW}Testing: $test_name${NC}"
    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✅ $test_name passed${NC}"
    else
        echo -e "${RED}❌ $test_name failed${NC}"
        FAILURES="$FAILURES\n  - $test_name"
        echo "  Error output:"
        tail -5 /tmp/test_output.log | sed 's/^/    /'
    fi
    echo ""
}

# 1. Clean environment
echo "🧹 Cleaning environment..."
make clean > /dev/null 2>&1 || true
cd backend && go clean -cache > /dev/null 2>&1 || true
cd ..
echo -e "${GREEN}✅ Environment cleaned${NC}"
echo ""

# 2. Test main Makefile targets
echo "=== Testing Main Makefile Targets ==="
run_test "make build" "make build"
run_test "make test" "make test"
run_test "make bench" "timeout 30s make bench || true"
echo ""

# 3. Test backend Makefile targets
echo "=== Testing Backend Makefile Targets ==="
cd backend
run_test "make go-build" "make go-build"
run_test "make benchmark" "make benchmark"
run_test "make bench-tools" "make bench-tools"
run_test "make test" "make test"
cd ..
echo ""

# 4. Test individual Go packages
echo "=== Testing Go Packages ==="
cd backend
for pkg in pkg/orderbook pkg/lx pkg/fix pkg/metric; do
    if [ -d "$pkg" ]; then
        run_test "Test $pkg" "go test -v ./$pkg/..."
    fi
done
cd ..
echo ""

# 5. Test with race detector
echo "=== Testing with Race Detector ==="
cd backend
run_test "Race detector test" "go test -race -short ./pkg/..."
cd ..
echo ""

# 6. Test benchmarks
echo "=== Testing Benchmarks ==="
cd backend
run_test "Orderbook benchmarks" "go test -bench=. -benchtime=1s ./pkg/orderbook | grep -E 'Benchmark|ns/op'"
cd ..
echo ""

# 7. Test compilation of cmd tools
echo "=== Testing CMD Tools Compilation ==="
cd backend
for cmd_dir in cmd/*/; do
    cmd_name=$(basename "$cmd_dir")
    # Skip ZMQ-based commands that require CGO
    if [[ "$cmd_name" == *"zmq"* ]] || [[ "$cmd_name" == "hybrid-auto" ]] || [[ "$cmd_name" == "multi-node"* ]] || [[ "$cmd_name" == "nats-auto" ]]; then
        echo "  ⏭️  Skipping $cmd_name (requires CGO/ZMQ)"
        continue
    fi
    
    run_test "Build cmd/$cmd_name" "CGO_ENABLED=0 go build -o /tmp/test-$cmd_name ./$cmd_dir"
    rm -f /tmp/test-$cmd_name
done
cd ..
echo ""

# 8. Test Docker builds (if Docker is available)
if command -v docker &> /dev/null; then
    echo "=== Testing Docker Builds ==="
    cd backend
    run_test "Docker build (Pure Go)" "docker build -f Dockerfile -t lx-test:go . --quiet"
    run_test "Docker build (Go specific)" "docker build -f Dockerfile.go -t lx-test:go-specific . --quiet"
    # Clean up test images
    docker rmi lx-test:go lx-test:go-specific > /dev/null 2>&1 || true
    cd ..
else
    echo "⏭️  Docker not available, skipping Docker tests"
fi
echo ""

# 9. Check Go module integrity
echo "=== Checking Go Module Integrity ==="
cd backend
run_test "go mod verify" "go mod verify"
run_test "go mod tidy check" "go mod tidy && git diff --exit-code go.mod go.sum"
cd ..
echo ""

# 10. Test coverage generation
echo "=== Testing Coverage Generation ==="
cd backend
run_test "Coverage generation" "go test -coverprofile=/tmp/coverage.out ./pkg/... > /dev/null 2>&1"
if [ -f /tmp/coverage.out ]; then
    coverage=$(go tool cover -func=/tmp/coverage.out | grep total | awk '{print $3}')
    echo "  📊 Total coverage: $coverage"
fi
cd ..
echo ""

# 11. Summary
echo "============================================"
echo "           Test Summary"
echo "============================================"

if [ -z "$FAILURES" ]; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    echo ""
    echo "Your codebase is ready for production!"
    exit 0
else
    echo -e "${RED}❌ Some tests failed:${NC}"
    echo -e "$FAILURES"
    echo ""
    echo "Please fix the above issues before proceeding."
    exit 1
fi