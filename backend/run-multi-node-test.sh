#!/bin/bash

# Simple script to run multi-node test with ZMQ
echo "🚀 Starting LX DEX Multi-Node Test with ZeroMQ"
echo "=============================================="

# Check if ZMQ is installed
if ! command -v pkg-config &> /dev/null || ! pkg-config --exists libzmq; then
    echo "⚠️  ZeroMQ not found. Please install it first:"
    echo "  macOS:  brew install zeromq"
    echo "  Linux:  sudo apt-get install libzmq3-dev"
    exit 1
fi

# Build binaries
echo "📦 Building binaries..."
go build -o bin/multi-node ./cmd/multi-node/main.go
go build -o bin/test-client ./cmd/test-client/main.go

# Start nodes in background
echo "🔗 Starting 3 nodes..."

# Node 0 (Leader)
./bin/multi-node \
    --node node0 \
    --port 5000 \
    --peers "tcp://localhost:5100,tcp://localhost:5200" \
    --leader > logs/node0.log 2>&1 &
NODE0_PID=$!
echo "  ✓ Node 0 (Leader) started with PID $NODE0_PID"

sleep 1

# Node 1
./bin/multi-node \
    --node node1 \
    --port 5100 \
    --peers "tcp://localhost:5000,tcp://localhost:5200" > logs/node1.log 2>&1 &
NODE1_PID=$!
echo "  ✓ Node 1 started with PID $NODE1_PID"

sleep 1

# Node 2
./bin/multi-node \
    --node node2 \
    --port 5200 \
    --peers "tcp://localhost:5000,tcp://localhost:5100" > logs/node2.log 2>&1 &
NODE2_PID=$!
echo "  ✓ Node 2 started with PID $NODE2_PID"

sleep 2

# Run test client
echo ""
echo "📊 Running load test (1000 orders/sec for 10s)..."
./bin/test-client \
    --nodes "tcp://localhost:5002,tcp://localhost:5102,tcp://localhost:5202" \
    --duration 10s \
    --rate 1000

# Show results
echo ""
echo "📈 Test Results:"
echo "==============="
for i in 0 1 2; do
    echo "Node $i metrics:"
    grep "metrics" logs/node$i.log | tail -1 || echo "  No metrics yet"
done

# Cleanup
echo ""
echo "🧹 Cleaning up..."
kill $NODE0_PID $NODE1_PID $NODE2_PID 2>/dev/null
echo "✅ Test complete!"