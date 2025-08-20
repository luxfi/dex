#!/bin/bash

echo "==================================="
echo "LXD - Lux DEX Node Demo"
echo "==================================="
echo ""

# Build everything
echo "🔨 Building luxd and clients..."
make clean build > /dev/null 2>&1
go build -o bin/lux-client ./cmd/lux-client > /dev/null 2>&1
go build -o bin/ws-client ./cmd/ws-client > /dev/null 2>&1
echo "✅ Build complete"
echo ""

# Start luxd in background
echo "🚀 Starting luxd node..."
./bin/luxd > luxd.log 2>&1 &
LUXD_PID=$!
echo "✅ luxd started with PID $LUXD_PID"
echo ""

# Wait for startup
sleep 3

# Show node status
echo "📊 Node Status:"
tail -n 5 luxd.log | grep "LXD Node Status" | head -1
echo ""

# Test HTTP client
echo "🔍 Testing HTTP client..."
./bin/lux-client -action test 2>&1 | grep -E "(Client|order placed|Note:)"
echo ""

# Test WebSocket client
echo "🔍 Testing WebSocket client..."
timeout 2 ./bin/ws-client 2>&1 | grep -E "(Connecting|Note:|expected)"
echo ""

# Check database
echo "💾 Database Status:"
if [ -d "$HOME/.lxd/badgerdb" ]; then
    echo "✅ BadgerDB initialized at ~/.lxd/badgerdb"
    ls -la ~/.lxd/badgerdb/*.sst 2>/dev/null | head -3
else
    echo "⚠️  Using in-memory database"
fi
echo ""

# Show performance
echo "⚡ Performance Metrics:"
tail -n 20 luxd.log | grep "Performance metrics" | tail -1
tail -n 20 luxd.log | grep "Block time achievement" | tail -1
echo ""

# Clean up
echo "🧹 Cleaning up..."
kill $LUXD_PID 2>/dev/null
wait $LUXD_PID 2>/dev/null
echo "✅ Demo complete"
echo ""
echo "To run luxd manually: ./bin/luxd"
echo "To test client: ./bin/lux-client -action test"