#!/bin/bash

echo "🚀 LX Engine - DEX Server & Trader Demo"
echo "========================================"
echo ""

# Clean up any existing processes
pkill simple-dex 2>/dev/null
pkill dex-trader 2>/dev/null
sleep 1

# Build
echo "🔨 Building DEX components..."
(cd backend && go build -o bin/simple-dex ./cmd/simple-dex)
(cd backend && go build -o bin/dex-trader ./cmd/dex-trader)

# Start server
echo ""
echo "📡 Starting DEX Server on port 8080..."
backend/bin/simple-dex -port 8080 &
SERVER_PID=$!
sleep 2

# Check server is up
echo "🔍 Checking server health..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Server is healthy!"
else
    echo "❌ Server failed to start"
    exit 1
fi

# Start trader
echo ""
echo "💹 Starting DEX Trader (10 traders, 100 orders/sec each)..."
echo "-----------------------------------------------------------"
backend/bin/dex-trader \
    -server http://localhost:8080 \
    -traders 10 \
    -rate 100 \
    -duration 15s

# Get final stats
echo ""
echo "📊 Final Server Statistics:"
curl -s http://localhost:8080/stats | python3 -m json.tool

# Cleanup
echo ""
echo "🧹 Cleaning up..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""
echo "✅ Demo complete!"
echo ""
echo "To run server and trader separately:"
echo "  Terminal 1: make dex-server"
echo "  Terminal 2: make dex-trader"