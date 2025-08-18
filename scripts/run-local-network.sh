#!/bin/bash
# Run Local Lux Network for DEX Testing

set -e

echo "🚀 Starting Local Lux Network..."
echo "================================"

# Configuration
LUXD_PATH="${HOME}/work/lux/node/luxd"
DATA_DIR="/tmp/lux-local"
NETWORK_ID=96369  # Lux testnet ID
CHAIN_ID=96369

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if luxd exists
if [ ! -f "$LUXD_PATH" ]; then
    echo -e "${RED}❌ luxd not found at $LUXD_PATH${NC}"
    echo "Building luxd..."
    cd ~/work/lux/node
    ./scripts/build.sh
    cd -
fi

# Clean previous data
echo "Cleaning previous data..."
rm -rf $DATA_DIR
mkdir -p $DATA_DIR/staking

# Generate staking keys
echo "Generating staking keys..."
cd $DATA_DIR/staking
openssl genrsa -out staker.key 4096 2>/dev/null
openssl req -new -x509 -key staker.key -out staker.crt -days 365 \
    -subj '/C=US/ST=NY/O=LuxDEX/CN=lux' 2>/dev/null
cd -

echo -e "${GREEN}✅ Staking keys generated${NC}"

# Create genesis file for local network
cat > $DATA_DIR/genesis.json << 'EOF'
{
  "networkID": 96369,
  "allocations": [
    {
      "ethAddr": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb7",
      "avaxAddr": "X-lux18jma8ppw3nhx5r4ap8clazz0dps7rv5u6wdp",
      "initialAmount": 100000000000000000000000000,
      "unlockSchedule": []
    }
  ],
  "startTime": 1630000000,
  "initialStakeDuration": 31536000,
  "initialStakeDurationOffset": 5400,
  "initialStakedFunds": [],
  "initialStakers": [],
  "cChainGenesis": "{\"config\":{\"chainId\":96369,\"homesteadBlock\":0,\"eip150Block\":0,\"eip150Hash\":\"0x2086799aeebeae135c246c65021c82b4e15a2c451340993aacfd2751886514f0\",\"eip155Block\":0,\"eip158Block\":0,\"byzantiumBlock\":0,\"constantinopleBlock\":0,\"petersburgBlock\":0,\"istanbulBlock\":0,\"muirGlacierBlock\":0,\"subnetEVMTimestamp\":0},\"nonce\":\"0x0\",\"timestamp\":\"0x0\",\"extraData\":\"0x00\",\"gasLimit\":\"0x5f5e100\",\"difficulty\":\"0x0\",\"mixHash\":\"0x0000000000000000000000000000000000000000000000000000000000000000\",\"coinbase\":\"0x0000000000000000000000000000000000000000\",\"alloc\":{\"742d35Cc6634C0532925a3b844Bc9e7595f0bEb7\":{\"balance\":\"0x52b7d2dcc80cd2e4000000\"}},\"number\":\"0x0\",\"gasUsed\":\"0x0\",\"parentHash\":\"0x0000000000000000000000000000000000000000000000000000000000000000\"}",
  "message": "Local DEX Test Network"
}
EOF

echo -e "${GREEN}✅ Genesis file created${NC}"

# Start the node
echo ""
echo -e "${YELLOW}Starting Lux Node...${NC}"
echo "Network ID: $NETWORK_ID"
echo "Data Directory: $DATA_DIR"
echo ""

# Run luxd with local network configuration
$LUXD_PATH \
  --network-id=$NETWORK_ID \
  --data-dir=$DATA_DIR \
  --api-admin-enabled=true \
  --api-ipcs-enabled=true \
  --api-keystore-enabled=true \
  --api-metrics-enabled=true \
  --http-host=0.0.0.0 \
  --http-port=9650 \
  --staking-port=9651 \
  --log-level=info \
  --bootstrap-ips="" \
  --bootstrap-ids="" \
  --staking-enabled=false \
  --network-id-override=$NETWORK_ID \
  --public-ip=127.0.0.1 \
  --health-check-frequency=2s \
  --network-initial-timeout=10s \
  --network-minimum-timeout=5s \
  --network-maximum-timeout=15s \
  --snow-sample-size=1 \
  --snow-quorum-size=1 \
  --snow-concurrent-repolls=1 \
  --index-enabled=true \
  --chain-config-dir=$DATA_DIR/chains \
  &

NODE_PID=$!
echo "Node PID: $NODE_PID"

# Wait for node to start
echo "Waiting for node to start..."
sleep 5

# Check if node is running
for i in {1..30}; do
    if curl -s -X POST --data '{
        "jsonrpc":"2.0",
        "id":1,
        "method":"info.isBootstrapped",
        "params": {"chain":"X"}
    }' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/info 2>/dev/null | grep -q "true"; then
        echo -e "${GREEN}✅ Node is bootstrapped and ready!${NC}"
        break
    fi
    echo "Waiting for bootstrap... ($i/30)"
    sleep 2
done

# Test RPC endpoints
echo ""
echo "Testing RPC Endpoints..."
echo "========================"

# Test info endpoint
echo -n "1. Info endpoint: "
if curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"info.getNetworkID"
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/info | grep -q "$NETWORK_ID"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test health endpoint
echo -n "2. Health endpoint: "
if curl -s http://127.0.0.1:9650/ext/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Not healthy${NC}"
fi

# Test X-Chain RPC
echo -n "3. X-Chain RPC: "
if curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"avm.getBalance",
    "params":{
        "address":"X-lux18jma8ppw3nhx5r4ap8clazz0dps7rv5u6wdp",
        "assetID":"LUX"
    }
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/X 2>/dev/null | grep -q "balance"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test C-Chain RPC (EVM)
echo -n "4. C-Chain RPC: "
if curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "method":"eth_chainId",
    "params":[],
    "id":1
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/C/rpc 2>/dev/null | grep -q "result"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test block production
echo -n "5. Block production: "
BLOCK_HEIGHT=$(curl -s -X POST --data '{
    "jsonrpc":"2.0",
    "method":"eth_blockNumber",
    "params":[],
    "id":1
}' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/C/rpc 2>/dev/null | grep -o '"result":"[^"]*"' | sed 's/"result":"//;s/"//')

if [ ! -z "$BLOCK_HEIGHT" ]; then
    echo -e "${GREEN}✅ Current block: $BLOCK_HEIGHT${NC}"
else
    echo -e "${RED}❌ No blocks${NC}"
fi

echo ""
echo "================================"
echo -e "${GREEN}✅ Local Lux Network is running!${NC}"
echo ""
echo "RPC Endpoints:"
echo "  Info API: http://localhost:9650/ext/info"
echo "  Health: http://localhost:9650/ext/health"
echo "  X-Chain: http://localhost:9650/ext/bc/X"
echo "  C-Chain: http://localhost:9650/ext/bc/C/rpc"
echo "  P-Chain: http://localhost:9650/ext/bc/P"
echo ""
echo "WebSocket:"
echo "  C-Chain WS: ws://localhost:9650/ext/bc/C/ws"
echo ""
echo "Node PID: $NODE_PID"
echo "To stop: kill $NODE_PID"
echo ""
echo "Monitoring blocks..."
echo "Press Ctrl+C to stop"
echo ""

# Monitor block production
while true; do
    BLOCK=$(curl -s -X POST --data '{
        "jsonrpc":"2.0",
        "method":"eth_blockNumber",
        "params":[],
        "id":1
    }' -H 'content-type:application/json;' http://127.0.0.1:9650/ext/bc/C/rpc 2>/dev/null | grep -o '"result":"[^"]*"' | sed 's/"result":"//;s/"//')
    
    if [ ! -z "$BLOCK" ]; then
        BLOCK_DEC=$((16#${BLOCK#0x}))
        echo -e "Block #$BLOCK_DEC produced at $(date '+%H:%M:%S')"
    fi
    sleep 5
done