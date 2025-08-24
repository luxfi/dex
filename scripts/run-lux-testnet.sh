#!/bin/bash
# Run Local Lux X-Chain Network for DEX

set -e

echo "🚀 Starting Local Lux X-Chain Network"
echo "======================================"

# Configuration
LUXD_PATH="${HOME}/work/lux/node/luxd"
DATA_DIR="/tmp/lux-dex-testnet"
NETWORK_ID=12345  # Local test network ID

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Clean previous data
echo "Cleaning previous test network data..."
rm -rf $DATA_DIR
mkdir -p $DATA_DIR/staking

# Generate staking keys
echo "Generating staking keys..."
cd $DATA_DIR/staking
openssl genrsa -out staker.key 4096 2>/dev/null
openssl req -new -x509 -key staker.key -out staker.crt -days 365 \
    -subj '/C=US/ST=NY/O=LuxDEX/CN=lux' 2>/dev/null
cd - > /dev/null

echo -e "${GREEN}✅ Staking keys generated${NC}"

# Create chain configs
mkdir -p $DATA_DIR/configs/chains/C
cat > $DATA_DIR/configs/chains/C/config.json << 'EOF'
{
  "snowman-api-enabled": false,
  "coreth-admin-api-enabled": false,
  "eth-apis": ["public-eth","public-eth-filter","net","web3","internal-public-eth","internal-public-blockchain","internal-public-transaction-pool"],
  "rpc-gas-cap": 50000000,
  "rpc-tx-fee-cap": 100,
  "pruning-enabled": true,
  "local-txs-enabled": true,
  "api-max-duration": 30000000000,
  "ws-cpu-refill-rate": 0,
  "ws-cpu-max-stored": 0,
  "api-max-blocks-per-request": 30,
  "allow-unfinalized-queries": false,
  "allow-unprotected-txs": false,
  "remote-tx-gossip-only-enabled": false,
  "log-level": "info"
}
EOF

# Start the node with minimal configuration for fast local testing
echo ""
echo -e "${YELLOW}Starting Lux Node...${NC}"
echo "Network ID: $NETWORK_ID"
echo "Data Directory: $DATA_DIR"
echo ""

# Kill any existing luxd processes
pkill -f luxd 2>/dev/null || true
sleep 2

# Run luxd with local testnet configuration
$LUXD_PATH \
  --network-id=local \
  --data-dir=$DATA_DIR \
  --api-admin-enabled=true \
  --api-ipcs-enabled=true \
  --api-keystore-enabled=true \
  --api-metrics-enabled=true \
  --http-host=0.0.0.0 \
  --http-port=9650 \
  --staking-port=9651 \
  --log-level=info \
  --staking-enabled=false \
  --public-ip=127.0.0.1 \
  --health-check-frequency=2s \
  --snow-sample-size=1 \
  --snow-quorum-size=1 \
  --index-enabled=true \
  --db-dir=$DATA_DIR/db \
  --chain-config-dir=$DATA_DIR/configs/chains \
  --staking-tls-cert-file=$DATA_DIR/staking/staker.crt \
  --staking-tls-key-file=$DATA_DIR/staking/staker.key \
  > $DATA_DIR/node.log 2>&1 &

NODE_PID=$!
echo "Node PID: $NODE_PID"
echo "Log file: $DATA_DIR/node.log"

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local data=$3
    
    echo -n "$name: "
    if curl -s -X POST --data "$data" -H 'content-type:application/json;' "$url" 2>/dev/null | grep -q "result\|error"; then
        echo -e "${GREEN}✅ Working${NC}"
        return 0
    else
        echo -e "${RED}❌ Not ready${NC}"
        return 1
    fi
}

# Wait for node to start
echo ""
echo "Waiting for node to initialize..."
for i in {1..60}; do
    if test_endpoint "Health Check" "http://127.0.0.1:9650/ext/health" '{}' 2>/dev/null; then
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo ""
echo "Testing Lux RPC Endpoints..."
echo "============================="

# Test info endpoint
test_endpoint "1. Info API" \
    "http://127.0.0.1:9650/ext/info" \
    '{"jsonrpc":"2.0","id":1,"method":"info.getNodeID"}'

# Test keystore
test_endpoint "2. Keystore API" \
    "http://127.0.0.1:9650/ext/keystore" \
    '{"jsonrpc":"2.0","id":1,"method":"keystore.listUsers"}'

# Test X-Chain (AVM - Asset Virtual Machine)
test_endpoint "3. X-Chain API" \
    "http://127.0.0.1:9650/ext/bc/X" \
    '{"jsonrpc":"2.0","id":1,"method":"avm.getAssetDescription","params":{"assetID":"LUX"}}'

# Test P-Chain (Platform Chain)
test_endpoint "4. P-Chain API" \
    "http://127.0.0.1:9650/ext/bc/P" \
    '{"jsonrpc":"2.0","id":1,"method":"platform.getHeight"}'

# Test C-Chain (Contract Chain - EVM)
test_endpoint "5. C-Chain API" \
    "http://127.0.0.1:9650/ext/bc/C/rpc" \
    '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

# Get blockchain info
echo ""
echo "Blockchain Status:"
echo "=================="

# Get node info
NODE_INFO=$(curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"info.getNodeVersion"
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/info 2>/dev/null)

if [ ! -z "$NODE_INFO" ]; then
    echo "Node Version: $(echo $NODE_INFO | grep -o '"version":"[^"]*"' | cut -d'"' -f4)"
fi

# Get X-Chain status
echo ""
echo "X-Chain (DEX Chain) Status:"
curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"avm.getHeight"
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/X 2>/dev/null | grep -o '"height":[0-9]*' || echo "Height: 0"

# Create test wallet
echo ""
echo "Creating Test Wallet..."
echo "======================="

# Create user
curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"keystore.createUser",
    "params":{
        "username":"dexuser",
        "password":"dexpass123"
    }
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/keystore > /dev/null 2>&1

# Create X-Chain address
X_ADDR=$(curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"avm.createAddress",
    "params":{
        "username":"dexuser",
        "password":"dexpass123"
    }
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/X 2>/dev/null | grep -o '"address":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$X_ADDR" ]; then
    echo -e "${GREEN}✅ X-Chain Address: $X_ADDR${NC}"
fi

echo ""
echo "======================================"
echo -e "${GREEN}✅ Local Lux Network is running!${NC}"
echo ""
echo "RPC Endpoints:"
echo "  Info API: http://localhost:9650/ext/info"
echo "  Health: http://localhost:9650/ext/health"
echo "  X-Chain: http://localhost:9650/ext/bc/X"
echo "  P-Chain: http://localhost:9650/ext/bc/P"
echo "  C-Chain: http://localhost:9650/ext/bc/C/rpc"
echo ""
echo "Node PID: $NODE_PID"
echo "Logs: tail -f $DATA_DIR/node.log"
echo "To stop: kill $NODE_PID"
echo ""

# Monitor X-Chain
echo "Monitoring X-Chain blocks..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    HEIGHT=$(curl -s -X POST --data '{
        "jsonrpc":"2.0",
        "id":1,
        "method":"avm.getHeight"
    }' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/X 2>/dev/null | grep -o '"height":[0-9]*' | cut -d':' -f2)
    
    if [ ! -z "$HEIGHT" ]; then
        echo "X-Chain Height: $HEIGHT at $(date '+%H:%M:%S')"
    fi
    
    # Also check P-Chain
    P_HEIGHT=$(curl -s -X POST --data '{
        "jsonrpc":"2.0",
        "id":1,
        "method":"platform.getHeight"
    }' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/P 2>/dev/null | grep -o '"height":[0-9]*' | cut -d':' -f2)
    
    if [ ! -z "$P_HEIGHT" ]; then
        echo "P-Chain Height: $P_HEIGHT"
    fi
    
    sleep 5
done