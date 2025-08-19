#!/bin/bash
# Run a 3-node QZMQ-secured DEX network with post-quantum cryptography

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/qzmq-dex"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BASE_PORT=5000
NUM_NODES=3
LOG_DIR="$PROJECT_DIR/logs/qzmq"

# Create log directory
mkdir -p "$LOG_DIR"

# Clean up function
cleanup() {
    echo -e "${YELLOW}Shutting down QZMQ network...${NC}"
    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                echo -e "${RED}Stopped node with PID $PID${NC}"
            fi
        fi
    done
    rm -f "$LOG_DIR"/*.pid
    echo -e "${GREEN}QZMQ network shutdown complete${NC}"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Build the QZMQ DEX binary
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}    QZMQ Post-Quantum DEX Network      ${NC}"
echo -e "${CYAN}========================================${NC}"
echo

echo -e "${YELLOW}Building QZMQ DEX binary...${NC}"
cd "$PROJECT_DIR"
if CGO_ENABLED=1 go build -o "$BINARY" ./cmd/qzmq-dex/; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Generate ML-DSA certificates if needed
if [ ! -d "$PROJECT_DIR/certs" ]; then
    echo -e "${YELLOW}Generating ML-DSA certificates...${NC}"
    "$SCRIPT_DIR/generate-mldsa-certs.sh"
fi

# Function to start a node
start_node() {
    local NODE_ID=$1
    local IS_LEADER=$2
    local SUITE=$3
    local PQ_ONLY=$4
    
    local PORT=$((BASE_PORT + NODE_ID))
    local LOG_FILE="$LOG_DIR/node-${NODE_ID}.log"
    local PID_FILE="$LOG_DIR/node-${NODE_ID}.pid"
    
    echo -e "${BLUE}Starting Node ${NODE_ID}...${NC}"
    echo -e "  Port: ${PORT}"
    echo -e "  Suite: ${SUITE}"
    echo -e "  PQ-Only: ${PQ_ONLY}"
    echo -e "  Leader: ${IS_LEADER}"
    
    # Start the node
    $BINARY \
        -id "$NODE_ID" \
        -port "$PORT" \
        -leader="$IS_LEADER" \
        -pq-only="$PQ_ONLY" \
        -suite="$SUITE" \
        > "$LOG_FILE" 2>&1 &
    
    local PID=$!
    echo $PID > "$PID_FILE"
    
    echo -e "${GREEN}✓ Node ${NODE_ID} started (PID: $PID)${NC}"
    echo
}

# Display configuration
echo -e "${MAGENTA}Network Configuration:${NC}"
echo -e "  Nodes: ${NUM_NODES}"
echo -e "  Base Port: ${BASE_PORT}"
echo -e "  Cryptography: Post-Quantum (ML-KEM + ML-DSA)"
echo -e "  Consensus: K=3 with quantum finality"
echo

# Start nodes with different configurations
echo -e "${CYAN}Starting QZMQ nodes...${NC}"
echo

# Node 0: Leader with hybrid suite
start_node 0 true "hybrid768" false

# Node 1: Follower with PQ-only ML-KEM-768
start_node 1 false "mlkem768" true

# Node 2: Follower with hybrid ML-KEM-1024
start_node 2 false "hybrid1024" false

# Wait for nodes to initialize
echo -e "${YELLOW}Waiting for nodes to establish QZMQ connections...${NC}"
sleep 3

# Check node status
echo -e "${CYAN}Checking node status...${NC}"
for i in $(seq 0 $((NUM_NODES - 1))); do
    PID_FILE="$LOG_DIR/node-${i}.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${GREEN}✓ Node ${i} is running (PID: $PID)${NC}"
        else
            echo -e "${RED}✗ Node ${i} has stopped${NC}"
        fi
    else
        echo -e "${RED}✗ Node ${i} PID file not found${NC}"
    fi
done

echo
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}    QZMQ Network Running                ${NC}"
echo -e "${CYAN}========================================${NC}"
echo

# Display monitoring commands
echo -e "${MAGENTA}Monitor Commands:${NC}"
echo -e "  View logs:        tail -f $LOG_DIR/node-*.log"
echo -e "  Node 0 log:       tail -f $LOG_DIR/node-0.log"
echo -e "  Node 1 log:       tail -f $LOG_DIR/node-1.log"
echo -e "  Node 2 log:       tail -f $LOG_DIR/node-2.log"
echo

echo -e "${MAGENTA}Test Commands:${NC}"
echo -e "  Submit order:     curl -X POST http://localhost:8080/order -d '{...}'"
echo -e "  Check stats:      curl http://localhost:8080/stats"
echo -e "  Get orderbook:    curl http://localhost:8080/orderbook"
echo

echo -e "${MAGENTA}Security Features Active:${NC}"
echo -e "  ✓ ML-KEM-768/1024 key encapsulation"
echo -e "  ✓ ML-DSA-2/3 digital signatures"
echo -e "  ✓ AES-256-GCM authenticated encryption"
echo -e "  ✓ HKDF key derivation"
echo -e "  ✓ Anti-DoS cookies"
echo -e "  ✓ Key rotation every 10 minutes"
echo -e "  ✓ Perfect forward secrecy"
echo

# Function to display real-time stats
show_stats() {
    while true; do
        clear
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}    QZMQ Network Statistics             ${NC}"
        echo -e "${CYAN}========================================${NC}"
        echo
        
        for i in $(seq 0 $((NUM_NODES - 1))); do
            LOG_FILE="$LOG_DIR/node-${i}.log"
            if [ -f "$LOG_FILE" ]; then
                echo -e "${BLUE}Node ${i}:${NC}"
                tail -n 5 "$LOG_FILE" | grep -E "(Messages:|OrderBook|KeyUpdates)" || echo "  No recent activity"
                echo
            fi
        done
        
        echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
        sleep 5
    done
}

# Ask if user wants to monitor
echo -e "${YELLOW}Start real-time monitoring? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    show_stats
else
    echo -e "${GREEN}QZMQ network is running in the background.${NC}"
    echo -e "${YELLOW}Press Ctrl+C to shutdown the network.${NC}"
    
    # Keep script running
    while true; do
        sleep 1
    done
fi