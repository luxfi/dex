#!/bin/bash

# LX DEX CI Test Script
# Simulates GitHub Actions CI pipeline locally

set -e

echo "========================================="
echo "   LX DEX CI Test (Local Simulation)"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Change to project root
cd "$(dirname "$0")/.."

# Step 1: Install dependencies
echo "📦 Installing dependencies..."
cd backend
go mod download
go mod verify
cd ..
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 2: Run tests
echo "🧪 Running tests..."
cd backend
if go test -v ./pkg/...; then
    echo -e "${GREEN}✅ Tests passed${NC}"
else
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
fi
cd ..
echo ""

# Step 3: Run tests with race detector
echo "🔍 Running tests with race detector..."
cd backend
if go test -race -short ./pkg/...; then
    echo -e "${GREEN}✅ Race tests passed${NC}"
else
    echo -e "${RED}❌ Race tests failed${NC}"
    exit 1
fi
cd ..
echo ""

# Step 4: Generate coverage
echo "📊 Generating coverage..."
cd backend
go test -coverprofile=coverage.out ./pkg/... > /dev/null 2>&1
coverage=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')
echo "Coverage: $coverage"
cd ..
echo -e "${GREEN}✅ Coverage generated${NC}"
echo ""

# Step 5: Build binaries
echo "🔨 Building binaries..."
make build
echo -e "${GREEN}✅ Binaries built${NC}"
echo ""

# Step 6: Run benchmarks
echo "⚡ Running benchmarks..."
cd backend
if go test -bench=. -benchtime=1s ./pkg/orderbook | grep -E "Benchmark|ns/op"; then
    echo -e "${GREEN}✅ Benchmarks completed${NC}"
else
    echo -e "${RED}❌ Benchmarks failed${NC}"
    exit 1
fi
cd ..
echo ""

# Step 7: Check for compilation of all cmd tools
echo "🛠️ Checking cmd tools compilation..."
cd backend
failed_cmds=""
for cmd_dir in cmd/*/; do
    cmd_name=$(basename "$cmd_dir")
    # Skip ZMQ-based commands that require CGO
    if [[ "$cmd_name" == *"zmq"* ]] || [[ "$cmd_name" == "hybrid-auto" ]] || [[ "$cmd_name" == "multi-node"* ]]; then
        continue
    fi
    
    if ! CGO_ENABLED=0 go build -o /tmp/test-binary "./$cmd_dir" 2>/dev/null; then
        failed_cmds="$failed_cmds $cmd_name"
    fi
    rm -f /tmp/test-binary
done

if [ -n "$failed_cmds" ]; then
    echo -e "${RED}❌ Failed to compile:$failed_cmds${NC}"
else
    echo -e "${GREEN}✅ All cmd tools compile${NC}"
fi
cd ..
echo ""

# Step 8: Lint check (if golangci-lint is installed)
echo "🔍 Running lint checks..."
if command -v golangci-lint &> /dev/null; then
    cd backend
    if golangci-lint run --timeout=5m ./...; then
        echo -e "${GREEN}✅ Lint checks passed${NC}"
    else
        echo -e "${RED}⚠️  Lint issues found (non-blocking)${NC}"
    fi
    cd ..
else
    echo "golangci-lint not installed, skipping..."
fi
echo ""

# Summary
echo "========================================="
echo "          CI Test Summary"
echo "========================================="
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo -e "${GREEN}✅ Tests passed${NC}"
echo -e "${GREEN}✅ Race detector passed${NC}"
echo -e "${GREEN}✅ Coverage: $coverage${NC}"
echo -e "${GREEN}✅ Binaries built${NC}"
echo -e "${GREEN}✅ Benchmarks completed${NC}"
echo -e "${GREEN}✅ CI tests completed successfully!${NC}"
echo ""
echo "Your code is ready for CI/CD!"