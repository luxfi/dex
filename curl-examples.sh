#!/bin/bash

echo "======================================"
echo "LX DEX JSON-RPC API - Curl Examples"
echo "======================================"
echo ""

# Base URL
URL="http://localhost:8080"

echo "1. Health Check"
echo "---------------"
curl -s $URL/health | jq .
echo ""

echo "2. Ping Test (JSON-RPC)"
echo "------------------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_ping",
    "params": {},
    "id": 1
  }' | jq .
echo ""

echo "3. Get Node Info"
echo "----------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_getInfo",
    "params": {},
    "id": 2
  }' | jq .
echo ""

echo "4. Get Order Book"
echo "-----------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_getOrderBook",
    "params": {"depth": 5},
    "id": 3
  }' | jq .
echo ""

echo "5. Place Buy Order"
echo "------------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_placeOrder",
    "params": {
      "symbol": "BTC-USD",
      "type": 0,
      "side": 0,
      "price": 49000,
      "size": 1.0,
      "userID": "user1"
    },
    "id": 4
  }' | jq .
echo ""

echo "6. Place Sell Order"
echo "-------------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_placeOrder",
    "params": {
      "symbol": "BTC-USD",
      "type": 0,
      "side": 1,
      "price": 51000,
      "size": 1.0,
      "userID": "user2"
    },
    "id": 5
  }' | jq .
echo ""

echo "7. Get Best Bid"
echo "---------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_getBestBid",
    "params": {},
    "id": 6
  }' | jq .
echo ""

echo "8. Get Best Ask"
echo "---------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_getBestAsk",
    "params": {},
    "id": 7
  }' | jq .
echo ""

echo "9. Get Recent Trades"
echo "--------------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_getTrades",
    "params": {"limit": 10},
    "id": 8
  }' | jq .
echo ""

echo "10. Cancel Order"
echo "----------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_cancelOrder",
    "params": {"orderId": 1},
    "id": 9
  }' | jq .
echo ""

echo "11. Get Order by ID"
echo "-------------------"
curl -s -X POST $URL/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "lx_getOrder",
    "params": {"orderId": 2},
    "id": 10
  }' | jq .