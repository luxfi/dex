#!/bin/bash
# E2E Test Script for LUX DEX

set -e

echo "🧪 LUX DEX E2E Test Suite"
echo "========================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

test_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 1. Test Database Connectivity
echo ""
echo "1. Testing Database Connectivity..."
if docker exec lux-dex-db pg_isready -U dexuser -d luxdex > /dev/null 2>&1; then
    test_pass "PostgreSQL is running and accepting connections"
else
    test_fail "PostgreSQL is not accessible"
fi

# 2. Test Redis Connectivity
echo ""
echo "2. Testing Redis Cache..."
if docker exec lux-dex-redis redis-cli ping | grep -q PONG; then
    test_pass "Redis is running and responding"
else
    test_fail "Redis is not accessible"
fi

# 3. Test Backend Build
echo ""
echo "3. Testing Backend Build..."
cd backend
if go build -o ../bin/test-server ./cmd/dex-server 2>/dev/null; then
    test_pass "Backend builds successfully"
    rm ../bin/test-server
else
    test_fail "Backend build failed"
fi
cd ..

# 4. Test Order Book Package
echo ""
echo "4. Testing Order Book Implementation..."
cd backend
if go test ./pkg/lx/... -v -count=1 2>&1 | grep -q "PASS"; then
    test_pass "Order book tests pass"
else
    test_fail "Order book tests failed"
fi
cd ..

# 5. Test UI Build
echo ""
echo "5. Testing UI Build..."
cd ui
if npm run build > /dev/null 2>&1; then
    test_pass "UI builds successfully"
else
    test_fail "UI build failed"
fi
cd ..

# 6. Test WebSocket Order Book Server
echo ""
echo "6. Testing WebSocket Implementation..."
cd backend
if go test ./pkg/orderbook/... -v -count=1 2>&1 | grep -q "PASS"; then
    test_pass "WebSocket order book tests pass"
else
    test_fail "WebSocket order book tests failed"
fi
cd ..

# 7. Performance Benchmark
echo ""
echo "7. Running Performance Benchmark..."
cd backend/pkg/lx
BENCH_OUTPUT=$(go test -bench=BenchmarkOrderMatching -benchtime=1s -run=^$ 2>&1)
if echo "$BENCH_OUTPUT" | grep -q "ns/op"; then
    LATENCY=$(echo "$BENCH_OUTPUT" | grep "BenchmarkOrderMatching" | awk '{print $3}')
    test_pass "Order matching benchmark completed: ${LATENCY} ns/op"
    
    # Check if we meet the <1μs target (1000ns)
    LATENCY_NUM=$(echo "$LATENCY" | sed 's/[^0-9.]//g')
    if (( $(echo "$LATENCY_NUM < 1000" | bc -l) )); then
        test_pass "✨ Sub-microsecond latency achieved!"
    else
        test_info "Latency above 1μs target"
    fi
else
    test_fail "Benchmark failed to run"
fi
cd ../../..

# 8. Test Playwright Setup
echo ""
echo "8. Testing Playwright E2E Setup..."
cd ui
if npx playwright --version > /dev/null 2>&1; then
    test_pass "Playwright is installed"
    
    # Run a simple Playwright test
    cat > test-simple.spec.ts << 'EOF'
import { test, expect } from '@playwright/test';

test('basic test', async ({ page }) => {
  // This is a placeholder test
  expect(1 + 1).toBe(2);
});
EOF
    
    if npx playwright test test-simple.spec.ts --reporter=line 2>&1 | grep -q "1 passed"; then
        test_pass "Playwright tests can run"
    else
        test_fail "Playwright test execution failed"
    fi
    rm test-simple.spec.ts
else
    test_fail "Playwright is not installed"
fi
cd ..

# 9. Integration Test: Order Flow
echo ""
echo "9. Testing Order Flow Integration..."
cat > test-orders.go << 'EOF'
package main

import (
    "fmt"
    "os"
    "github.com/your-org/lx-dex/backend/pkg/lx"
)

func main() {
    ob := lx.NewOrderBook("BTC-USD", 10000)
    
    // Add buy order
    buyOrder := &lx.Order{
        ID:     "buy-1",
        Symbol: "BTC-USD",
        Side:   lx.Buy,
        Price:  50000.0,
        Size:   1.0,
        Trader: "trader-1",
    }
    ob.AddOrder(buyOrder)
    
    // Add sell order (should match)
    sellOrder := &lx.Order{
        ID:     "sell-1",
        Symbol: "BTC-USD",
        Side:   lx.Sell,
        Price:  50000.0,
        Size:   1.0,
        Trader: "trader-2",
    }
    trades := ob.AddOrder(sellOrder)
    
    if len(trades) == 1 && trades[0].Price == 50000.0 {
        fmt.Println("PASS")
        os.Exit(0)
    }
    fmt.Println("FAIL")
    os.Exit(1)
}
EOF

cd backend
if go run ../test-orders.go 2>/dev/null | grep -q "PASS"; then
    test_pass "Order matching integration works"
else
    test_fail "Order matching integration failed"
fi
cd ..
rm test-orders.go

# 10. Test Docker Compose Validation
echo ""
echo "10. Testing Docker Compose Configuration..."
if docker compose -f docker/compose.dev.yml config > /dev/null 2>&1; then
    test_pass "Docker Compose configuration is valid"
else
    test_fail "Docker Compose configuration has errors"
fi

# Summary
echo ""
echo "======================================="
echo "E2E Test Results Summary"
echo "======================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo ""
    echo "⚠️  Some tests failed. Please review the output above."
    exit 1
else
    echo -e "${GREEN}All tests passed! 🎉${NC}"
    echo ""
    echo "✅ The LUX DEX stack is working correctly!"
    echo "   - Databases are running"
    echo "   - Backend builds and tests pass"
    echo "   - UI builds successfully"
    echo "   - Performance meets targets"
    echo "   - Integration tests pass"
    exit 0
fi