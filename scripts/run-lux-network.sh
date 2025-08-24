#!/bin/bash
# Run LX DEX on Lux Network using netrunner
# This script starts a local Lux network with X-Chain and deploys the DEX

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NETRUNNER_PATH="${NETRUNNER_PATH:-$HOME/work/lux/netrunner}"
NODE_PATH="${NODE_PATH:-$HOME/work/lux/node/build/node}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
NUM_NODES="${NUM_NODES:-3}"

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check if netrunner is available
    if [ ! -f "$NETRUNNER_PATH/bin/netrunner" ]; then
        echo -e "${RED}Error: netrunner not found at $NETRUNNER_PATH${NC}"
        echo "Please build netrunner first:"
        echo "  cd $NETRUNNER_PATH && ./scripts/build.sh"
        exit 1
    fi
    
    # Check if Lux node is available
    if [ ! -f "$NODE_PATH" ]; then
        echo -e "${RED}Error: Lux node not found at $NODE_PATH${NC}"
        echo "Please build the Lux node first:"
        echo "  cd ~/work/lux/node && ./scripts/build.sh"
        exit 1
    fi
    
    # Check if DEX binary is built
    if [ ! -f "$PROJECT_ROOT/bin/xchain-dex" ]; then
        echo -e "${RED}Error: xchain-dex not found${NC}"
        echo "Building DEX..."
        cd "$PROJECT_ROOT"
        make build
    fi
    
    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Start netrunner server
start_netrunner_server() {
    echo -e "${BLUE}Starting netrunner server...${NC}"
    
    # Kill any existing netrunner server
    pkill -f "netrunner server" || true
    sleep 2
    
    # Start netrunner server in background
    mkdir -p "$LOG_DIR"
    "$NETRUNNER_PATH/bin/netrunner" server \
        --log-level debug \
        --port=":8080" \
        --grpc-gateway-port=":8081" \
        > "$LOG_DIR/netrunner-server.log" 2>&1 &
    
    NETRUNNER_PID=$!
    echo "Netrunner server started with PID $NETRUNNER_PID"
    
    # Wait for server to be ready
    echo "Waiting for netrunner server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8081/v1/ping > /dev/null 2>&1; then
            echo -e "${GREEN}Netrunner server is ready${NC}"
            break
        fi
        sleep 1
    done
}

# Start Lux network
start_lux_network() {
    echo -e "${BLUE}Starting Lux network with $NUM_NODES nodes...${NC}"
    
    # Create network configuration
    cat > "$LOG_DIR/network-config.json" <<EOF
{
    "execPath": "$NODE_PATH",
    "numNodes": $NUM_NODES,
    "logLevel": "INFO",
    "trackSubnets": "X",
    "enableWalletService": true,
    "globalNodeConfig": {
        "network-peer-list-gossip-frequency": "250ms",
        "network-max-reconnect-delay": "1s",
        "public-ip": "127.0.0.1",
        "health-check-frequency": "2s",
        "api-admin-enabled": true,
        "api-ipcs-enabled": true,
        "api-keystore-enabled": true,
        "api-metrics-enabled": true,
        "chain-config-dir": "$PROJECT_ROOT/configs/chains",
        "subnet-config-dir": "$PROJECT_ROOT/configs/subnets",
        "db-dir": "$LOG_DIR/db"
    }
}
EOF
    
    # Start the network using netrunner control
    "$NETRUNNER_PATH/bin/netrunner" control start \
        --endpoint="0.0.0.0:8080" \
        --node-path="$NODE_PATH" \
        --number-of-nodes=$NUM_NODES \
        --blockchain-specs="[]" \
        --global-node-config='{"api-admin-enabled":true,"api-ipcs-enabled":true,"api-keystore-enabled":true,"api-metrics-enabled":true}' \
        --log-level=debug
    
    echo -e "${GREEN}Lux network started with $NUM_NODES nodes${NC}"
    
    # Wait for network to be healthy
    echo "Waiting for network to be healthy..."
    sleep 10
}

# Deploy X-Chain DEX
deploy_xchain_dex() {
    echo -e "${BLUE}Deploying X-Chain DEX...${NC}"
    
    # Get node endpoints from netrunner
    NODE1_HTTP="http://127.0.0.1:9650"
    NODE2_HTTP="http://127.0.0.1:9652"
    NODE3_HTTP="http://127.0.0.1:9654"
    
    # Start DEX on each node
    echo "Starting DEX on Node 1..."
    "$PROJECT_ROOT/bin/xchain-dex" \
        --node-id node1 \
        --lux-endpoint "$NODE1_HTTP" \
        --http-port 8090 \
        --ws-port 8091 \
        --log-level debug \
        > "$LOG_DIR/dex-node1.log" 2>&1 &
    DEX1_PID=$!
    echo "DEX Node 1 started with PID $DEX1_PID"
    
    echo "Starting DEX on Node 2..."
    "$PROJECT_ROOT/bin/xchain-dex" \
        --node-id node2 \
        --lux-endpoint "$NODE2_HTTP" \
        --http-port 8092 \
        --ws-port 8093 \
        --log-level debug \
        > "$LOG_DIR/dex-node2.log" 2>&1 &
    DEX2_PID=$!
    echo "DEX Node 2 started with PID $DEX2_PID"
    
    echo "Starting DEX on Node 3..."
    "$PROJECT_ROOT/bin/xchain-dex" \
        --node-id node3 \
        --lux-endpoint "$NODE3_HTTP" \
        --http-port 8094 \
        --ws-port 8095 \
        --log-level debug \
        > "$LOG_DIR/dex-node3.log" 2>&1 &
    DEX3_PID=$!
    echo "DEX Node 3 started with PID $DEX3_PID"
    
    echo -e "${GREEN}X-Chain DEX deployed on all nodes${NC}"
}

# Test DEX functionality
test_dex() {
    echo -e "${BLUE}Testing DEX functionality...${NC}"
    
    # Wait for DEX to be ready
    sleep 5
    
    # Test health endpoint
    echo "Testing DEX health..."
    curl -s http://localhost:8090/health | jq .
    
    # Submit test order
    echo "Submitting test order..."
    curl -X POST http://localhost:8090/order \
        -H "Content-Type: application/json" \
        -d '{
            "symbol": "BTC-USD",
            "side": "buy",
            "type": "limit",
            "price": 50000,
            "size": 1.0,
            "user": "test-user"
        }' | jq .
    
    # Get order book
    echo "Getting order book..."
    curl -s http://localhost:8090/book/BTC-USD | jq .
    
    echo -e "${GREEN}DEX tests completed${NC}"
}

# Show status
show_status() {
    echo -e "${BLUE}=== Network Status ===${NC}"
    echo "Netrunner Server: http://localhost:8081"
    echo "Node 1 RPC: http://localhost:9650"
    echo "Node 2 RPC: http://localhost:9652"
    echo "Node 3 RPC: http://localhost:9654"
    echo ""
    echo "DEX Node 1: http://localhost:8090"
    echo "DEX Node 2: http://localhost:8092"
    echo "DEX Node 3: http://localhost:8094"
    echo ""
    echo "Logs: $LOG_DIR"
    echo ""
    echo -e "${GREEN}Network is running!${NC}"
    echo "Press Ctrl+C to stop"
}

# Cleanup function
cleanup() {
    echo -e "${RED}Stopping network...${NC}"
    
    # Stop DEX processes
    [ ! -z "$DEX1_PID" ] && kill $DEX1_PID 2>/dev/null || true
    [ ! -z "$DEX2_PID" ] && kill $DEX2_PID 2>/dev/null || true
    [ ! -z "$DEX3_PID" ] && kill $DEX3_PID 2>/dev/null || true
    
    # Stop network via netrunner
    "$NETRUNNER_PATH/bin/netrunner" control stop --endpoint="0.0.0.0:8080" 2>/dev/null || true
    
    # Stop netrunner server
    [ ! -z "$NETRUNNER_PID" ] && kill $NETRUNNER_PID 2>/dev/null || true
    
    echo -e "${GREEN}Network stopped${NC}"
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

# Main execution
main() {
    echo -e "${BLUE}=== LX DEX on Lux Network ===${NC}"
    
    check_prerequisites
    start_netrunner_server
    start_lux_network
    deploy_xchain_dex
    test_dex
    show_status
    
    # Keep running until interrupted
    while true; do
        sleep 60
        # Could add periodic health checks here
    done
}

# Run main function
main "$@"