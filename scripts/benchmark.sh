#!/bin/bash

# LX Engine Benchmark Script
# Tests Pure Go, Pure C++, and TypeScript engines with FIX data

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "======================================="
echo "LX Engine Benchmark Suite"
echo "======================================="

# Check if we should download FIX data
FIX_DATA_FILE="$PROJECT_ROOT/testdata/fix_sample.csv"
if [ ! -f "$FIX_DATA_FILE" ]; then
    echo "Creating sample FIX data..."
    mkdir -p "$PROJECT_ROOT/testdata"
    
    # Create sample CSV FIX data
    cat > "$FIX_DATA_FILE" << 'EOF'
MsgType,Symbol,OrderID,Side,OrderType,Price,Quantity,Timestamp
D,BTC-USD,1001,1,2,50100.50,0.5,2024-01-15T10:00:00Z
D,BTC-USD,1002,2,2,50105.00,0.3,2024-01-15T10:00:01Z
D,BTC-USD,1003,1,2,50102.00,0.7,2024-01-15T10:00:02Z
D,ETH-USD,1004,2,2,2510.25,2.0,2024-01-15T10:00:03Z
D,BTC-USD,1005,1,2,50099.00,1.0,2024-01-15T10:00:04Z
F,BTC-USD,1002,2,2,0,0,2024-01-15T10:00:05Z
D,BTC-USD,1006,2,2,50103.50,0.4,2024-01-15T10:00:06Z
G,BTC-USD,1003,1,2,50101.50,0.8,2024-01-15T10:00:07Z
D,ETH-USD,1007,1,2,2508.00,1.5,2024-01-15T10:00:08Z
D,BTC-USD,1008,2,2,50098.00,0.6,2024-01-15T10:00:09Z
EOF
    echo "Sample FIX data created at $FIX_DATA_FILE"
fi

# Build all engines
echo ""
echo "Building engines..."
echo "-------------------"

# Build Pure Go engine
echo "Building Pure Go engine..."
cd "$PROJECT_ROOT"
CGO_ENABLED=0 go build -o bin/benchmark-go ./cmd/benchmark

# Build Hybrid Go/C++ engine (if available)
if command -v g++ &> /dev/null; then
    echo "Building Hybrid Go/C++ engine..."
    CGO_ENABLED=1 go build -o bin/benchmark-hybrid ./cmd/benchmark
else
    echo "C++ compiler not found, skipping hybrid build"
fi

# Build TypeScript engine
if command -v npm &> /dev/null; then
    echo "Building TypeScript engine..."
    if [ ! -d "$PROJECT_ROOT/ts-engine/node_modules" ]; then
        cd "$PROJECT_ROOT/ts-engine"
        npm install
    fi
    cd "$PROJECT_ROOT/ts-engine"
    npm run build 2>/dev/null || echo "TypeScript build skipped (not configured)"
else
    echo "npm not found, skipping TypeScript build"
fi

# Run benchmarks
echo ""
echo "Running benchmarks..."
echo "====================="

# Test with synthetic data first (warmup)
echo ""
echo "Warmup with synthetic data (10k messages)..."
"$PROJECT_ROOT/bin/benchmark-go" -iter 1 -warmup 1000

# Benchmark Pure Go
echo ""
echo "1. Pure Go Engine"
echo "-----------------"
time "$PROJECT_ROOT/bin/benchmark-go" \
    -data "$FIX_DATA_FILE" \
    -engine go \
    -iter 3 \
    -warmup 2 \
    -v

# Benchmark Hybrid Go/C++
if [ -f "$PROJECT_ROOT/bin/benchmark-hybrid" ]; then
    echo ""
    echo "2. Hybrid Go/C++ Engine"
    echo "-----------------------"
    time "$PROJECT_ROOT/bin/benchmark-hybrid" \
        -data "$FIX_DATA_FILE" \
        -engine cpp \
        -iter 3 \
        -warmup 2 \
        -v
fi

# Benchmark TypeScript (if built)
if [ -f "$PROJECT_ROOT/ts-engine/dist/benchmark.js" ]; then
    echo ""
    echo "3. TypeScript Engine"
    echo "--------------------"
    cd "$PROJECT_ROOT/ts-engine"
    time node dist/benchmark.js \
        --data "$FIX_DATA_FILE" \
        --iter 3 \
        --warmup 2
fi

# Large scale test
echo ""
echo "======================================="
echo "Large Scale Test (100k synthetic messages)"
echo "======================================="

# Generate and test with large dataset
"$PROJECT_ROOT/bin/benchmark-go" -engine all -iter 1 -warmup 1000

echo ""
echo "======================================="
echo "Benchmark Complete!"
echo "======================================="

# Optional: Download real FIX data from public source
echo ""
echo "To test with real FIX data, you can download from:"
echo "  - https://www.fixprotocol.org/samples"
echo "  - https://github.com/quickfix/quickfix/tree/master/spec/fix"
echo ""
echo "Run with: ./benchmark.sh --data path/to/fix_data.csv"