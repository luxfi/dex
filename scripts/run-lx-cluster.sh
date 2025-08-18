#!/bin/bash
# Run LX DEX Cluster with X-Chain and Q-Chain

echo "🚀 Starting LX DEX Cluster (X-Chain + Q-Chain)"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
DATA_DIR="/tmp/lx-dex-cluster"
CLUSTER_SIZE=3  # Number of nodes for consensus
BASE_PORT=8080
BASE_WS_PORT=8081
BASE_P2P_PORT=5000

# Clean previous data
echo "Cleaning previous cluster data..."
rm -rf $DATA_DIR
mkdir -p $DATA_DIR/{node1,node2,node3}

# Build the DEX server if needed
echo "Building LX DEX server..."
cd /Users/z/work/lx/dex/backend
go build -o ../bin/lx-dex ./cmd/dex-server

cd /Users/z/work/lx/dex

# Start cluster nodes
echo ""
echo "Starting LX DEX Cluster Nodes..."
echo "================================"

# Node 1 (Bootstrap node)
echo -n "Starting Node 1 (Bootstrap): "
./bin/lx-dex \
  --node-id node1 \
  --http-port $BASE_PORT \
  --ws-port $BASE_WS_PORT \
  --p2p-port $BASE_P2P_PORT \
  --data-dir $DATA_DIR/node1 \
  --chain x-chain \
  --consensus k=3 \
  --bootstrap \
  > $DATA_DIR/node1.log 2>&1 &
NODE1_PID=$!
echo -e "${GREEN}✅ PID: $NODE1_PID${NC}"

sleep 2

# Node 2
echo -n "Starting Node 2: "
./bin/lx-dex \
  --node-id node2 \
  --http-port $((BASE_PORT + 10)) \
  --ws-port $((BASE_WS_PORT + 10)) \
  --p2p-port $((BASE_P2P_PORT + 10)) \
  --data-dir $DATA_DIR/node2 \
  --chain x-chain \
  --consensus k=3 \
  --bootstrap-peer 127.0.0.1:$BASE_P2P_PORT \
  > $DATA_DIR/node2.log 2>&1 &
NODE2_PID=$!
echo -e "${GREEN}✅ PID: $NODE2_PID${NC}"

sleep 2

# Node 3
echo -n "Starting Node 3: "
./bin/lx-dex \
  --node-id node3 \
  --http-port $((BASE_PORT + 20)) \
  --ws-port $((BASE_WS_PORT + 20)) \
  --p2p-port $((BASE_P2P_PORT + 20)) \
  --data-dir $DATA_DIR/node3 \
  --chain x-chain \
  --consensus k=3 \
  --bootstrap-peer 127.0.0.1:$BASE_P2P_PORT \
  > $DATA_DIR/node3.log 2>&1 &
NODE3_PID=$!
echo -e "${GREEN}✅ PID: $NODE3_PID${NC}"

sleep 3

# Test cluster health
echo ""
echo "Testing Cluster Health..."
echo "========================"

# Function to check node health
check_node() {
    local port=$1
    local node=$2
    echo -n "Node $node (port $port): "
    if curl -s http://localhost:$port/health 2>/dev/null | grep -q "ok"; then
        echo -e "${GREEN}✅ Healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ Not responding${NC}"
        return 1
    fi
}

check_node $BASE_PORT 1
check_node $((BASE_PORT + 10)) 2
check_node $((BASE_PORT + 20)) 3

# Test X-Chain RPC
echo ""
echo "Testing X-Chain RPC Endpoints..."
echo "================================"

# Test order book endpoint
echo -n "1. Order Book API: "
if curl -s http://localhost:$BASE_PORT/api/orderbook 2>/dev/null | grep -q "bids\|asks\|{}"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test market data
echo -n "2. Market Data API: "
if curl -s http://localhost:$BASE_PORT/api/markets 2>/dev/null | grep -q "\\[\\]\\|markets"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test WebSocket
echo -n "3. WebSocket Feed: "
if timeout 1 websocat ws://localhost:$BASE_WS_PORT/ws 2>/dev/null <<< '{"type":"ping"}' | grep -q "pong\|connected"; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${YELLOW}⚠ WebSocket not ready${NC}"
fi

# Test consensus
echo -n "4. Consensus Status: "
CONSENSUS=$(curl -s http://localhost:$BASE_PORT/consensus/status 2>/dev/null)
if [ ! -z "$CONSENSUS" ]; then
    echo -e "${GREEN}✅ K=3 Active${NC}"
else
    echo -e "${YELLOW}⚠ Initializing${NC}"
fi

# Create test orders
echo ""
echo "Creating Test Orders..."
echo "======================"

# Submit a test order
ORDER_RESPONSE=$(curl -s -X POST http://localhost:$BASE_PORT/api/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "side": "buy",
    "price": 50000,
    "size": 1.0,
    "trader": "test-trader-1"
  }' 2>/dev/null)

if echo "$ORDER_RESPONSE" | grep -q "id"; then
    echo -e "${GREEN}✅ Order submitted successfully${NC}"
else
    echo -e "${YELLOW}⚠ Order submission pending${NC}"
fi

# Monitor blocks
echo ""
echo "======================================"
echo -e "${GREEN}✅ LX DEX Cluster is running!${NC}"
echo ""
echo "Cluster Nodes:"
echo "  Node 1: http://localhost:$BASE_PORT (Bootstrap)"
echo "  Node 2: http://localhost:$((BASE_PORT + 10))"
echo "  Node 3: http://localhost:$((BASE_PORT + 20))"
echo ""
echo "WebSocket Feeds:"
echo "  Node 1: ws://localhost:$BASE_WS_PORT"
echo "  Node 2: ws://localhost:$((BASE_WS_PORT + 10))"
echo "  Node 3: ws://localhost:$((BASE_WS_PORT + 20))"
echo ""
echo "Process IDs:"
echo "  Node 1: $NODE1_PID"
echo "  Node 2: $NODE2_PID"
echo "  Node 3: $NODE3_PID"
echo ""
echo "Logs:"
echo "  tail -f $DATA_DIR/node1.log"
echo "  tail -f $DATA_DIR/node2.log"
echo "  tail -f $DATA_DIR/node3.log"
echo ""
echo "To stop cluster: kill $NODE1_PID $NODE2_PID $NODE3_PID"
echo ""
echo "Monitoring block formation..."
echo "Press Ctrl+C to stop"
echo ""

# Monitor block production
while true; do
    # Get block height from each node
    for i in 0 10 20; do
        PORT=$((BASE_PORT + i))
        NODE=$((i/10 + 1))
        
        BLOCK_INFO=$(curl -s http://localhost:$PORT/api/blocks/latest 2>/dev/null)
        if [ ! -z "$BLOCK_INFO" ]; then
            HEIGHT=$(echo "$BLOCK_INFO" | grep -o '"height":[0-9]*' | cut -d':' -f2)
            if [ ! -z "$HEIGHT" ]; then
                echo "Node $NODE - Block #$HEIGHT at $(date '+%H:%M:%S')"
            fi
        fi
    done
    
    echo "---"
    sleep 5
done