#!/bin/bash

# LX DEX E2E Test Runner
# This script runs the full DEX server and performs end-to-end tests

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PORT=8080
SERVER_PID=""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "🚀 LX DEX End-to-End Test Suite"
echo "================================"
echo ""

# Step 1: Build the server
echo -e "${BLUE}Step 1: Building DEX API Server...${NC}"
go build -o ./bin/dex-api-server ./cmd/dex-api-server/main.go
echo -e "${GREEN}✓ Server built successfully${NC}"
echo ""

# Step 2: Start the server
echo -e "${BLUE}Step 2: Starting DEX API Server on port $PORT...${NC}"
./bin/dex-api-server -port=$PORT > server.log 2>&1 &
SERVER_PID=$!
sleep 2

# Check if server started
if ! ps -p $SERVER_PID > /dev/null; then
    echo -e "${RED}✗ Server failed to start${NC}"
    cat server.log
    exit 1
fi

echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
echo ""

# Step 3: Wait for server to be ready
echo -e "${BLUE}Step 3: Waiting for server to be ready...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:$PORT/api/stats > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}✗ Server failed to become ready${NC}"
        exit 1
    fi
    sleep 1
done
echo ""

# Step 4: Run E2E Tests
echo -e "${BLUE}Step 4: Running E2E Tests...${NC}"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
if curl -s http://localhost:$PORT/api/stats | grep -q "operational"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi

# Test 2: Order Book Should Have Initial Orders
echo "Test 2: Initial Order Book"
ORDERBOOK=$(curl -s http://localhost:$PORT/api/orderbook)
if echo "$ORDERBOOK" | grep -q "bids" && echo "$ORDERBOOK" | grep -q "asks"; then
    echo -e "${GREEN}✓ Order book initialized${NC}"
else
    echo -e "${RED}✗ Order book initialization failed${NC}"
    exit 1
fi

# Test 3: Place Orders
echo "Test 3: Place Orders"
# Place buy order
BUY_RESPONSE=$(curl -s -X POST http://localhost:$PORT/api/order \
  -H "Content-Type: application/json" \
  -d '{"type":"limit","side":"buy","price":49000,"size":1.0}')

if echo "$BUY_RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✓ Buy order placed${NC}"
else
    echo -e "${RED}✗ Failed to place buy order${NC}"
    echo "$BUY_RESPONSE"
    exit 1
fi

# Place sell order
SELL_RESPONSE=$(curl -s -X POST http://localhost:$PORT/api/order \
  -H "Content-Type: application/json" \
  -d '{"type":"limit","side":"sell","price":51000,"size":1.0}')

if echo "$SELL_RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✓ Sell order placed${NC}"
else
    echo -e "${RED}✗ Failed to place sell order${NC}"
    echo "$SELL_RESPONSE"
    exit 1
fi

# Test 4: Market Order Execution
echo "Test 4: Market Order Execution"
MARKET_RESPONSE=$(curl -s -X POST http://localhost:$PORT/api/order \
  -H "Content-Type: application/json" \
  -d '{"type":"market","side":"buy","price":0,"size":0.1}')

if echo "$MARKET_RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✓ Market order executed${NC}"
else
    echo -e "${RED}✗ Market order failed${NC}"
    echo "$MARKET_RESPONSE"
    exit 1
fi

# Test 5: Check Trades
echo "Test 5: Trade History"
TRADES=$(curl -s http://localhost:$PORT/api/trades)
if echo "$TRADES" | grep -q "data"; then
    echo -e "${GREEN}✓ Trades retrieved${NC}"
else
    echo -e "${RED}✗ Failed to retrieve trades${NC}"
    exit 1
fi

# Test 6: Stress Test - Place Many Orders
echo "Test 6: Stress Test (100 orders)"
SUCCESS_COUNT=0
for i in {1..100}; do
    PRICE=$((49000 + RANDOM % 2000))
    SIZE=$(echo "scale=2; $RANDOM/32768" | bc)
    SIDE=$((RANDOM % 2))
    
    if [ $SIDE -eq 0 ]; then
        SIDE_STR="buy"
    else
        SIDE_STR="sell"
    fi
    
    RESPONSE=$(curl -s -X POST http://localhost:$PORT/api/order \
      -H "Content-Type: application/json" \
      -d "{\"type\":\"limit\",\"side\":\"$SIDE_STR\",\"price\":$PRICE,\"size\":0.1}" 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q "success.*true" 2>/dev/null; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

if [ $SUCCESS_COUNT -gt 90 ]; then
    echo -e "${GREEN}✓ Stress test passed ($SUCCESS_COUNT/100 orders successful)${NC}"
else
    echo -e "${RED}✗ Stress test failed (only $SUCCESS_COUNT/100 orders successful)${NC}"
    exit 1
fi

# Test 7: Final Stats Check
echo "Test 7: Final Statistics"
FINAL_STATS=$(curl -s http://localhost:$PORT/api/stats)
if echo "$FINAL_STATS" | grep -q "total_orders" && echo "$FINAL_STATS" | grep -q "13M"; then
    echo -e "${GREEN}✓ Statistics endpoint working${NC}"
    echo "$FINAL_STATS" | python3 -m json.tool 2>/dev/null | head -20
else
    echo -e "${RED}✗ Statistics endpoint failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ ALL E2E TESTS PASSED SUCCESSFULLY!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Server logs saved to: server.log"
echo "Server PID: $SERVER_PID"
echo ""
echo "To test manually:"
echo "  curl http://localhost:$PORT/api/orderbook"
echo "  curl http://localhost:$PORT/api/stats"
echo ""
echo "To stop the server:"
echo "  kill $SERVER_PID"