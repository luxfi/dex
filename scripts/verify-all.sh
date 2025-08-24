#!/bin/bash

# LX DEX - Final Verification Script
# Ensures everything builds, tests pass, and benchmarks work

set -e

echo "========================================="
echo "   LX DEX Complete Verification"  
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

cd "$(dirname "$0")/.."

# 1. Clean
echo "🧹 Cleaning..."
make clean > /dev/null 2>&1
echo -e "${GREEN}✅ Clean complete${NC}"

# 2. Build
echo "🔨 Building..."
if make build > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# 3. Test core packages
echo "🧪 Testing..."
cd backend
if go test ./pkg/orderbook/... ./pkg/lx/... > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Tests passed${NC}"
else
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
fi
cd ..

# 4. Benchmark
echo "⚡ Running benchmark..."
if timeout 30s make bench > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Benchmarks complete${NC}"
else
    echo -e "${RED}❌ Benchmark failed or timed out${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo -e "${GREEN}   ✅ ALL SYSTEMS OPERATIONAL${NC}"
echo "========================================="
echo ""
echo "Summary:"
echo "  • Build: ✅ Working"
echo "  • Tests: ✅ Passing" 
echo "  • Bench: ✅ Running"
echo ""
echo "The codebase is fully functional!"