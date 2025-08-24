#!/bin/bash
# Simple Test Chain for DEX Development

echo "🚀 Starting Test Blockchain for DEX"
echo "===================================="

# Use Hardhat for a simple test chain (EVM-compatible)
cd /Users/z/work/lx/dex

# Check if hardhat is installed
if ! command -v npx &> /dev/null; then
    echo "Installing Hardhat..."
    npm install --save-dev hardhat
fi

# Create a simple hardhat config if it doesn't exist
if [ ! -f "hardhat.config.js" ]; then
    cat > hardhat.config.js << 'EOF'
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.19",
  networks: {
    hardhat: {
      chainId: 96369,
      mining: {
        auto: true,
        interval: 2000  // Mine a block every 2 seconds
      },
      accounts: {
        mnemonic: "test test test test test test test test test test test junk",
        count: 10,
        accountsBalance: "10000000000000000000000"  // 10,000 ETH each
      }
    }
  },
  paths: {
    artifacts: './artifacts',
    cache: './cache',
    sources: './contracts',
    tests: './test',
  }
};
EOF
fi

# Install dependencies if needed
if [ ! -d "node_modules/@nomicfoundation/hardhat-toolbox" ]; then
    echo "Installing dependencies..."
    npm install --save-dev @nomicfoundation/hardhat-toolbox
fi

echo ""
echo "Starting Hardhat Node..."
echo "========================"

# Start Hardhat node
npx hardhat node --port 8545 &
NODE_PID=$!

echo "Node PID: $NODE_PID"
sleep 5

# Test RPC
echo ""
echo "Testing RPC Endpoints..."
echo "========================"

# Test connection
echo -n "1. RPC Connection: "
if curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  http://localhost:8545 | grep -q "0x17931"; then
    echo "✅ Connected (Chain ID: 96369)"
else
    echo "❌ Failed"
fi

# Get block number
echo -n "2. Block Production: "
BLOCK=$(curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545 | grep -o '"result":"[^"]*"' | sed 's/"result":"//;s/"//')

if [ ! -z "$BLOCK" ]; then
    BLOCK_DEC=$((16#${BLOCK#0x}))
    echo "✅ Current block: $BLOCK_DEC"
else
    echo "❌ No blocks"
fi

# Get accounts
echo -n "3. Test Accounts: "
ACCOUNTS=$(curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
  http://localhost:8545 | grep -o '"result":\[[^]]*\]' | grep -o '0x[^"]*' | wc -l)

if [ "$ACCOUNTS" -gt 0 ]; then
    echo "✅ $ACCOUNTS accounts available"
else
    echo "❌ No accounts"
fi

echo ""
echo "===================================="
echo "✅ Test Chain Running!"
echo ""
echo "RPC URL: http://localhost:8545"
echo "Chain ID: 96369 (0x17931)"
echo "WebSocket: ws://localhost:8545"
echo ""
echo "Test Accounts (10,000 ETH each):"
curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
  http://localhost:8545 | grep -o '0x[^"]*' | head -5

echo ""
echo "To stop: kill $NODE_PID"
echo ""

# Monitor blocks
echo "Monitoring block production..."
while true; do
    BLOCK=$(curl -s -X POST -H "Content-Type: application/json" \
      --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
      http://localhost:8545 2>/dev/null | grep -o '"result":"[^"]*"' | sed 's/"result":"//;s/"//')
    
    if [ ! -z "$BLOCK" ]; then
        BLOCK_DEC=$((16#${BLOCK#0x}))
        echo "Block #$BLOCK_DEC at $(date '+%H:%M:%S')"
    fi
    sleep 5
done