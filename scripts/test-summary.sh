#!/bin/bash

echo "================================"
echo "LX - Test Suite Summary"
echo "================================"
echo ""

# Test core packages
echo "Testing Core Packages..."
echo "------------------------"

# OrderBook tests
echo -n "📦 pkg/lx (OrderBook): "
if go test ./pkg/lx -short 2>&1 | grep -q "^ok"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# MLX tests
echo -n "📦 pkg/mlx (GPU Engine): "
if go test ./pkg/mlx -short 2>&1 | grep -q "^ok"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# E2E tests
echo -n "📦 test/e2e (End-to-End): "
if go test ./test/e2e -short 2>&1 | grep -q "^ok"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

echo ""
echo "Running Full Test Suite..."
echo "------------------------"

# Count test results
PASSED=$(go test ./pkg/lx/... ./pkg/mlx/... ./test/e2e/... -v 2>&1 | grep -c "^--- PASS:")
FAILED=$(go test ./pkg/lx/... ./pkg/mlx/... ./test/e2e/... -v 2>&1 | grep -c "^--- FAIL:")
TOTAL=$((PASSED + FAILED))

echo ""
echo "================================"
echo "Test Results Summary"
echo "================================"
echo "Total Tests: $TOTAL"
echo "Passed: $PASSED ✅"
echo "Failed: $FAILED ❌"

if [ $FAILED -eq 0 ] && [ $TOTAL -gt 0 ]; then
    echo ""
    echo "🎉 100% PASS RATE - All tests passing!"
else
    if [ $TOTAL -gt 0 ]; then
        PASS_RATE=$((PASSED * 100 / TOTAL))
        echo ""
        echo "Pass Rate: $PASS_RATE%"
    fi
fi
