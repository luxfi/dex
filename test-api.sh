#!/bin/bash

# LX DEX API Test Script
# This script tests the DEX API with curl commands

API_URL="http://localhost:8080"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🚀 LX DEX API Test Suite"
echo "========================"
echo ""

# Function to pretty print JSON
pretty_json() {
    echo "$1" | python3 -m json.tool 2>/dev/null || echo "$1"
}

# Test 1: Get Order Book
echo -e "${BLUE}Test 1: Get Order Book${NC}"
echo "curl $API_URL/api/orderbook"
response=$(curl -s $API_URL/api/orderbook)
pretty_json "$response"
echo ""

# Test 2: Place Buy Order
echo -e "${BLUE}Test 2: Place Buy Limit Order${NC}"
echo 'curl -X POST $API_URL/api/order -H "Content-Type: application/json" -d {"type":"limit","side":"buy","price":49950,"size":0.5}'
response=$(curl -s -X POST $API_URL/api/order \
  -H "Content-Type: application/json" \
  -d '{"type":"limit","side":"buy","price":49950,"size":0.5}')
pretty_json "$response"
echo ""

# Test 3: Place Sell Order
echo -e "${BLUE}Test 3: Place Sell Limit Order${NC}"
echo 'curl -X POST $API_URL/api/order -H "Content-Type: application/json" -d {"type":"limit","side":"sell","price":50050,"size":0.3}'
response=$(curl -s -X POST $API_URL/api/order \
  -H "Content-Type: application/json" \
  -d '{"type":"limit","side":"sell","price":50050,"size":0.3}')
pretty_json "$response"
echo ""

# Test 4: Place Market Order
echo -e "${BLUE}Test 4: Place Market Buy Order${NC}"
echo 'curl -X POST $API_URL/api/order -H "Content-Type: application/json" -d {"type":"market","side":"buy","price":0,"size":0.1}'
response=$(curl -s -X POST $API_URL/api/order \
  -H "Content-Type: application/json" \
  -d '{"type":"market","side":"buy","price":0,"size":0.1}')
pretty_json "$response"
echo ""

# Test 5: Get Recent Trades
echo -e "${BLUE}Test 5: Get Recent Trades${NC}"
echo "curl $API_URL/api/trades"
response=$(curl -s $API_URL/api/trades)
pretty_json "$response"
echo ""

# Test 6: Get Market Stats
echo -e "${BLUE}Test 6: Get Market Statistics${NC}"
echo "curl $API_URL/api/stats"
response=$(curl -s $API_URL/api/stats)
pretty_json "$response"
echo ""

# Test 7: Get Updated Order Book
echo -e "${BLUE}Test 7: Get Updated Order Book${NC}"
echo "curl $API_URL/api/orderbook"
response=$(curl -s $API_URL/api/orderbook)
pretty_json "$response"
echo ""

echo -e "${GREEN}✅ API Tests Complete!${NC}"
echo ""
echo "To test WebSocket connection, use:"
echo "  wscat -c ws://localhost:8080/ws"
echo "Or:"
echo "  curl --include --no-buffer --header \"Connection: Upgrade\" --header \"Upgrade: websocket\" --header \"Sec-WebSocket-Version: 13\" --header \"Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==\" http://localhost:8080/ws"