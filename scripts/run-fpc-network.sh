#!/bin/bash

# Run 3-node FPC DAG network with quantum finality
# Uses Fast Probabilistic Consensus with hybrid Ringtail+BLS signatures

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    LX DEX - FPC DAG Network with Quantum Finality${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Consensus:${NC} Fast Probabilistic Consensus (FPC)"
echo -e "${GREEN}Security:${NC}  Hybrid Ringtail+BLS (Quantum-resistant)"
echo -e "${GREEN}Protocol:${NC}  Nebula secured by Quasar dual-certificates"
echo -e "${GREEN}Network:${NC}   3 nodes with ZeroMQ messaging"
echo ""

# Build the DAG network binary
echo -e "${YELLOW}Building DAG network binary...${NC}"
cd ../cmd/dag-network
go build -o dag-network

# Clean up any existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f dag-network 2>/dev/null || true
sleep 1

# Start Node 0 (Leader)
echo -e "${GREEN}Starting Node 0 (Leader) with FPC consensus...${NC}"
./dag-network \
    -node node0 \
    -http 8080 \
    -pub 5000 \
    -rep 5002 \
    -peers "tcp://localhost:5010,tcp://localhost:5020" \
    -leader \
    > /tmp/node0.log 2>&1 &
NODE0_PID=$!
echo "Node 0 PID: $NODE0_PID"

sleep 1

# Start Node 1
echo -e "${GREEN}Starting Node 1 with quantum finality...${NC}"
./dag-network \
    -node node1 \
    -http 8081 \
    -pub 5010 \
    -rep 5012 \
    -peers "tcp://localhost:5000,tcp://localhost:5020" \
    > /tmp/node1.log 2>&1 &
NODE1_PID=$!
echo "Node 1 PID: $NODE1_PID"

sleep 1

# Start Node 2
echo -e "${GREEN}Starting Node 2 with Quasar security...${NC}"
./dag-network \
    -node node2 \
    -http 8082 \
    -pub 5020 \
    -rep 5022 \
    -peers "tcp://localhost:5000,tcp://localhost:5010" \
    > /tmp/node2.log 2>&1 &
NODE2_PID=$!
echo "Node 2 PID: $NODE2_PID"

sleep 2

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ FPC DAG Network Running with Quantum Finality${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Node Status:${NC}"
echo "  Node 0 (Leader): http://localhost:8080/stats"
echo "  Node 1:          http://localhost:8081/stats"
echo "  Node 2:          http://localhost:8082/stats"
echo ""
echo -e "${YELLOW}Submit Order:${NC}"
echo '  curl -X POST http://localhost:8080/order -d "{"symbol":"BTC-USD","side":"buy","price":50000,"size":1}"'
echo ""
echo -e "${YELLOW}View Logs:${NC}"
echo "  tail -f /tmp/node0.log  # Leader node with order generation"
echo "  tail -f /tmp/node1.log  # Node 1"
echo "  tail -f /tmp/node2.log  # Node 2"
echo ""
echo -e "${YELLOW}FPC Features:${NC}"
echo "  • 50ms consensus rounds for ultra-fast finality"
echo "  • Adaptive vote threshold (55%-65%)"
echo "  • Quantum-resistant signatures (Ringtail+BLS)"
echo "  • Quasar dual-certificate overlay"
echo "  • 256 votes per block limit"
echo "  • Execute-owned optimization"
echo ""

# Function to check node health
check_health() {
    for port in 8080 8081 8082; do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Node on port $port is healthy"
        else
            echo -e "  ${RED}✗${NC} Node on port $port is not responding"
        fi
    done
}

# Wait for nodes to initialize
echo -e "${YELLOW}Waiting for nodes to initialize...${NC}"
sleep 3

echo -e "${YELLOW}Health Check:${NC}"
check_health

# Monitor function
monitor_stats() {
    while true; do
        clear
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}    FPC DAG Network Statistics (Quantum Finality)${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        for port in 8080 8081 8082; do
            NODE_NUM=$((port - 8080))
            echo -e "${GREEN}Node $NODE_NUM Statistics:${NC}"
            curl -s http://localhost:$port/stats 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Node not responding"
            echo ""
        done
        
        echo -e "${YELLOW}Press Ctrl+C to stop monitoring (network continues running)${NC}"
        sleep 5
    done
}

# Trap to clean up on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down FPC DAG network...${NC}"
    kill $NODE0_PID $NODE1_PID $NODE2_PID 2>/dev/null || true
    echo -e "${GREEN}Network stopped${NC}"
    exit 0
}

trap cleanup INT TERM

# Ask if user wants to monitor
echo ""
echo -e "${YELLOW}Press 'm' to monitor network statistics, or any other key to exit:${NC}"
read -n 1 -r REPLY
echo ""

if [[ $REPLY =~ ^[Mm]$ ]]; then
    monitor_stats
else
    echo -e "${GREEN}Network is running in background. PIDs: $NODE0_PID $NODE1_PID $NODE2_PID${NC}"
    echo -e "${YELLOW}To stop the network, run: kill $NODE0_PID $NODE1_PID $NODE2_PID${NC}"
fi