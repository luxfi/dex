#!/bin/bash

# Multi-node ZMQ test runner for LX DEX
# This script launches multiple nodes and runs tests against them

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NODES=${NODES:-3}
BASE_PORT=${BASE_PORT:-5000}
TEST_DURATION=${TEST_DURATION:-30s}
ORDERS_PER_SEC=${ORDERS_PER_SEC:-1000}

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Kill all node processes
    for pid in ${NODE_PIDS[@]}; do
        if kill -0 $pid 2>/dev/null; then
            echo "Killing node process $pid"
            kill $pid
        fi
    done
    
    # Kill client if running
    if [[ ! -z "$CLIENT_PID" ]] && kill -0 $CLIENT_PID 2>/dev/null; then
        echo "Killing client process $CLIENT_PID"
        kill $CLIENT_PID
    fi
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Build the multi-node binary
echo -e "${GREEN}Building multi-node binary...${NC}"
cd backend
go build -o bin/multi-node ./cmd/multi-node/main.go

# Check if ZMQ is installed
if ! pkg-config --exists libzmq; then
    echo -e "${RED}ZeroMQ not found. Installing...${NC}"
    
    # Detect OS and install ZMQ
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install zeromq
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y libzmq3-dev
    else
        echo -e "${RED}Unsupported OS. Please install ZeroMQ manually.${NC}"
        exit 1
    fi
fi

# Install Go ZMQ bindings if needed
echo -e "${GREEN}Checking Go ZMQ bindings...${NC}"
go get -u github.com/pebbe/zmq4

# Array to store node PIDs
NODE_PIDS=()

# Start nodes
echo -e "${GREEN}Starting $NODES nodes...${NC}"

for i in $(seq 0 $((NODES-1))); do
    NODE_ID="node$i"
    NODE_PORT=$((BASE_PORT + i*100))
    
    # Build peer list (all nodes except current)
    PEERS=""
    for j in $(seq 0 $((NODES-1))); do
        if [ $j -ne $i ]; then
            PEER_PORT=$((BASE_PORT + j*100))
            if [ -z "$PEERS" ]; then
                PEERS="tcp://localhost:$PEER_PORT"
            else
                PEERS="$PEERS,tcp://localhost:$PEER_PORT"
            fi
        fi
    done
    
    # First node is the leader
    LEADER_FLAG=""
    if [ $i -eq 0 ]; then
        LEADER_FLAG="--leader"
    fi
    
    # Start node in background
    echo -e "${YELLOW}Starting $NODE_ID on port $NODE_PORT (peers: $PEERS)${NC}"
    ./bin/multi-node \
        --node "$NODE_ID" \
        --port $NODE_PORT \
        --peers "$PEERS" \
        $LEADER_FLAG \
        > logs/${NODE_ID}.log 2>&1 &
    
    NODE_PIDS+=($!)
    echo -e "${GREEN}Started $NODE_ID with PID ${NODE_PIDS[-1]}${NC}"
    
    # Give node time to start
    sleep 1
done

echo -e "${GREEN}All nodes started${NC}"

# Wait for nodes to establish connections
echo -e "${YELLOW}Waiting for nodes to establish connections...${NC}"
sleep 3

# Show node status
echo -e "${GREEN}Node Status:${NC}"
for pid in ${NODE_PIDS[@]}; do
    if kill -0 $pid 2>/dev/null; then
        echo "  Node PID $pid: Running ✓"
    else
        echo "  Node PID $pid: Failed ✗"
    fi
done

# Run test client
echo -e "${GREEN}Starting test client...${NC}"

# Build node addresses for client
CLIENT_NODES=""
for i in $(seq 0 $((NODES-1))); do
    NODE_PORT=$((BASE_PORT + i*100 + 2)) # Router port
    if [ -z "$CLIENT_NODES" ]; then
        CLIENT_NODES="tcp://localhost:$NODE_PORT"
    else
        CLIENT_NODES="$CLIENT_NODES,tcp://localhost:$NODE_PORT"
    fi
done

# Build and run test client
go build -o bin/test-client ./cmd/multi-node/client.go

echo -e "${YELLOW}Running load test: $ORDERS_PER_SEC orders/sec for $TEST_DURATION${NC}"
./bin/test-client \
    --nodes "$CLIENT_NODES" \
    --duration $TEST_DURATION \
    --rate $ORDERS_PER_SEC &

CLIENT_PID=$!

# Monitor test progress
echo -e "${GREEN}Test running... Monitoring logs:${NC}"

# Function to show metrics from logs
show_metrics() {
    echo -e "\n${YELLOW}=== Node Metrics ===${NC}"
    for i in $(seq 0 $((NODES-1))); do
        NODE_ID="node$i"
        if [ -f "logs/${NODE_ID}.log" ]; then
            echo -e "${GREEN}$NODE_ID:${NC}"
            tail -n 5 "logs/${NODE_ID}.log" | grep -E "metrics|Orders:|Trade" || echo "  No metrics yet"
        fi
    done
}

# Monitor for duration of test
START_TIME=$(date +%s)
DURATION_SEC=$(echo $TEST_DURATION | sed 's/s$//')

while [ $(($(date +%s) - START_TIME)) -lt $DURATION_SEC ]; do
    sleep 5
    show_metrics
done

# Wait for client to finish
wait $CLIENT_PID

echo -e "${GREEN}Test complete!${NC}"

# Show final metrics
show_metrics

# Aggregate results
echo -e "\n${YELLOW}=== Aggregate Results ===${NC}"

TOTAL_ORDERS=0
TOTAL_TRADES=0

for i in $(seq 0 $((NODES-1))); do
    NODE_ID="node$i"
    if [ -f "logs/${NODE_ID}.log" ]; then
        ORDERS=$(grep "Orders:" "logs/${NODE_ID}.log" | tail -1 | grep -oE "Orders: [0-9]+" | grep -oE "[0-9]+" || echo 0)
        TRADES=$(grep "Trades:" "logs/${NODE_ID}.log" | tail -1 | grep -oE "Trades: [0-9]+" | grep -oE "[0-9]+" || echo 0)
        TOTAL_ORDERS=$((TOTAL_ORDERS + ORDERS))
        TOTAL_TRADES=$((TOTAL_TRADES + TRADES))
    fi
done

echo -e "${GREEN}Total Orders Processed: $TOTAL_ORDERS${NC}"
echo -e "${GREEN}Total Trades Executed: $TOTAL_TRADES${NC}"
echo -e "${GREEN}Throughput: $((TOTAL_ORDERS / DURATION_SEC)) orders/sec${NC}"

# Optional: Keep nodes running for manual testing
if [ "$KEEP_ALIVE" = "true" ]; then
    echo -e "${YELLOW}Nodes still running. Press Ctrl+C to stop.${NC}"
    wait
fi