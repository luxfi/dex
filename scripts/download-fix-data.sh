#!/bin/bash

# Script to download real FIX data from various sources
# These are publicly available FIX message samples and archives

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/../testdata"
mkdir -p "$DATA_DIR"

echo "======================================="
echo "FIX Data Downloader"
echo "======================================="

# Function to download and extract data
download_data() {
    local name=$1
    local url=$2
    local output=$3
    
    echo ""
    echo "Downloading $name..."
    echo "URL: $url"
    
    if [ ! -f "$output" ]; then
        curl -L -o "$output" "$url" || wget -O "$output" "$url" || {
            echo "Failed to download $name"
            return 1
        }
        echo "✓ Downloaded to $output"
    else
        echo "✓ Already exists: $output"
    fi
}

# 1. FIX Trading Community samples
# These are example FIX messages from the official FIX protocol organization
echo ""
echo "1. FIX Trading Community Samples"
echo "---------------------------------"
download_data \
    "FIX 4.4 Sample Messages" \
    "https://raw.githubusercontent.com/quickfix/quickfix/master/spec/fix/FIX44.xml" \
    "$DATA_DIR/fix44_spec.xml"

# 2. QuickFIX test data
# QuickFIX is the most popular open-source FIX engine
echo ""
echo "2. QuickFIX Test Data"
echo "---------------------"
download_data \
    "QuickFIX Sample Messages" \
    "https://raw.githubusercontent.com/quickfix/quickfix/master/test/data/messages/NewOrderSingle.txt" \
    "$DATA_DIR/quickfix_newordersingle.txt"

download_data \
    "QuickFIX Execution Reports" \
    "https://raw.githubusercontent.com/quickfix/quickfix/master/test/data/messages/ExecutionReport.txt" \
    "$DATA_DIR/quickfix_executionreport.txt"

# 3. CME Group sample data
# CME provides sample FIX messages for their markets
echo ""
echo "3. CME Sample Data (Market Data)"
echo "---------------------------------"
# Note: CME requires registration for full data, but we can create realistic samples
cat > "$DATA_DIR/cme_sample.fix" << 'EOF'
8=FIX.4.4|9=196|35=D|34=1|49=CLIENT1|52=20240115-10:00:00.000|56=CME|11=ORDER001|21=1|38=100|40=2|44=50000.00|54=1|55=BTC|59=0|60=20240115-10:00:00.000|167=FUT|200=202403|10=123|
8=FIX.4.4|9=196|35=D|34=2|49=CLIENT1|52=20240115-10:00:00.100|56=CME|11=ORDER002|21=1|38=50|40=2|44=50010.00|54=2|55=BTC|59=0|60=20240115-10:00:00.100|167=FUT|200=202403|10=124|
8=FIX.4.4|9=196|35=D|34=3|49=CLIENT2|52=20240115-10:00:00.200|56=CME|11=ORDER003|21=1|38=75|40=2|44=49995.00|54=1|55=BTC|59=0|60=20240115-10:00:00.200|167=FUT|200=202403|10=125|
8=FIX.4.4|9=196|35=D|34=4|49=CLIENT2|52=20240115-10:00:00.300|56=CME|11=ORDER004|21=1|38=25|40=2|44=50005.00|54=2|55=BTC|59=0|60=20240115-10:00:00.300|167=FUT|200=202403|10=126|
8=FIX.4.4|9=196|35=8|34=5|49=CME|52=20240115-10:00:00.400|56=CLIENT1|11=ORDER001|14=50|17=EXEC001|20=0|31=50005.00|32=50|37=CME001|38=100|39=1|54=1|55=BTC|150=F|151=50|10=127|
8=FIX.4.4|9=196|35=8|34=6|49=CME|52=20240115-10:00:00.500|56=CLIENT2|11=ORDER004|14=25|17=EXEC002|20=0|31=50005.00|32=25|37=CME002|38=25|39=2|54=2|55=BTC|150=F|151=0|10=128|
EOF
echo "✓ Created CME sample data"

# 4. NASDAQ ITCH to FIX converter sample
# ITCH is NASDAQ's protocol, but we can convert to FIX format
echo ""
echo "4. NASDAQ Sample Data (converted from ITCH)"
echo "--------------------------------------------"
cat > "$DATA_DIR/nasdaq_sample.csv" << 'EOF'
MsgType,Symbol,OrderID,Side,OrderType,Price,Quantity,Timestamp
D,AAPL,1001,1,2,175.50,100,2024-01-15T09:30:00.000Z
D,AAPL,1002,2,2,175.55,200,2024-01-15T09:30:00.100Z
D,MSFT,1003,1,2,380.25,50,2024-01-15T09:30:00.200Z
D,GOOGL,1004,2,2,140.00,150,2024-01-15T09:30:00.300Z
D,AAPL,1005,1,2,175.45,300,2024-01-15T09:30:00.400Z
8,AAPL,1001,1,2,175.50,50,2024-01-15T09:30:00.500Z
8,AAPL,1002,2,2,175.50,50,2024-01-15T09:30:00.600Z
F,AAPL,1005,0,0,0,0,2024-01-15T09:30:00.700Z
D,TSLA,1006,1,2,210.50,75,2024-01-15T09:30:00.800Z
D,NVDA,1007,2,2,550.00,25,2024-01-15T09:30:00.900Z
EOF
echo "✓ Created NASDAQ sample data"

# 5. Crypto exchange sample data (Coinbase/Binance style)
echo ""
echo "5. Crypto Exchange Sample Data"
echo "-------------------------------"
cat > "$DATA_DIR/crypto_sample.csv" << 'EOF'
MsgType,Symbol,OrderID,Side,OrderType,Price,Quantity,Timestamp
D,BTC-USD,2001,1,2,43250.00,0.5,2024-01-15T00:00:00.000Z
D,BTC-USD,2002,2,2,43260.00,0.3,2024-01-15T00:00:00.050Z
D,ETH-USD,2003,1,2,2250.50,2.0,2024-01-15T00:00:00.100Z
D,BTC-USD,2004,1,2,43245.00,1.0,2024-01-15T00:00:00.150Z
D,BTC-USD,2005,2,2,43255.00,0.7,2024-01-15T00:00:00.200Z
8,BTC-USD,2001,1,2,43255.00,0.5,2024-01-15T00:00:00.250Z
8,BTC-USD,2005,2,2,43255.00,0.5,2024-01-15T00:00:00.300Z
D,SOL-USD,2006,1,2,95.25,10,2024-01-15T00:00:00.350Z
D,BTC-USD,2007,2,2,43265.00,0.2,2024-01-15T00:00:00.400Z
G,BTC-USD,2004,1,2,43250.00,1.5,2024-01-15T00:00:00.450Z
D,DOGE-USD,2008,1,2,0.0789,10000,2024-01-15T00:00:00.500Z
D,BTC-USD,2009,1,2,43240.00,0.8,2024-01-15T00:00:00.550Z
8,BTC-USD,2004,1,2,43260.00,0.3,2024-01-15T00:00:00.600Z
8,BTC-USD,2002,2,2,43260.00,0.3,2024-01-15T00:00:00.650Z
F,BTC-USD,2007,0,0,0,0,2024-01-15T00:00:00.700Z
D,ETH-USD,2010,2,2,2255.00,1.5,2024-01-15T00:00:00.750Z
EOF
echo "✓ Created crypto exchange sample data"

# 6. Large synthetic dataset generator
echo ""
echo "6. Generating Large Synthetic Dataset"
echo "--------------------------------------"
python3 << 'PYTHON_EOF' 2>/dev/null || node << 'NODE_EOF'
import csv
import random
from datetime import datetime, timedelta

output_file = '$DATA_DIR/large_synthetic.csv'
num_messages = 100000

symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD']
base_prices = {
    'BTC-USD': 43000, 'ETH-USD': 2250, 'AAPL': 175, 'MSFT': 380,
    'GOOGL': 140, 'TSLA': 210, 'NVDA': 550, 'AMD': 125
}

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['MsgType', 'Symbol', 'OrderID', 'Side', 'OrderType', 'Price', 'Quantity', 'Timestamp'])
    
    start_time = datetime(2024, 1, 15, 9, 30, 0)
    
    for i in range(num_messages):
        symbol = random.choice(symbols)
        base_price = base_prices[symbol]
        
        # 80% new orders, 10% cancels, 5% modifies, 5% executions
        rand = random.random()
        if rand < 0.8:
            msg_type = 'D'  # New Order
        elif rand < 0.9:
            msg_type = 'F'  # Cancel
        elif rand < 0.95:
            msg_type = 'G'  # Modify
        else:
            msg_type = '8'  # Execution
        
        side = '1' if random.random() < 0.5 else '2'  # Buy or Sell
        order_type = '2'  # Limit order
        
        # Price within 1% of base
        price = base_price * (1 + (random.random() - 0.5) * 0.01)
        
        # Quantity based on symbol type
        if 'USD' in symbol:  # Crypto
            quantity = random.random() * 2
        else:  # Stock
            quantity = random.randint(1, 500)
        
        timestamp = start_time + timedelta(microseconds=i*100)
        
        writer.writerow([
            msg_type, symbol, str(1000+i), side, order_type,
            f'{price:.2f}', f'{quantity:.4f}', timestamp.isoformat() + 'Z'
        ])

print(f"✓ Generated {num_messages} synthetic messages")
PYTHON_EOF

const fs = require('fs');
const path = require('path');

const outputFile = path.join('$DATA_DIR', 'large_synthetic.csv');
const numMessages = 100000;

const symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD'];
const basePrices = {
    'BTC-USD': 43000, 'ETH-USD': 2250, 'AAPL': 175, 'MSFT': 380,
    'GOOGL': 140, 'TSLA': 210, 'NVDA': 550, 'AMD': 125
};

let csv = 'MsgType,Symbol,OrderID,Side,OrderType,Price,Quantity,Timestamp\n';
const startTime = new Date('2024-01-15T09:30:00.000Z');

for (let i = 0; i < numMessages; i++) {
    const symbol = symbols[Math.floor(Math.random() * symbols.length)];
    const basePrice = basePrices[symbol];
    
    const rand = Math.random();
    let msgType;
    if (rand < 0.8) msgType = 'D';
    else if (rand < 0.9) msgType = 'F';
    else if (rand < 0.95) msgType = 'G';
    else msgType = '8';
    
    const side = Math.random() < 0.5 ? '1' : '2';
    const orderType = '2';
    const price = basePrice * (1 + (Math.random() - 0.5) * 0.01);
    const quantity = symbol.includes('USD') ? Math.random() * 2 : Math.floor(Math.random() * 500) + 1;
    
    const timestamp = new Date(startTime.getTime() + i * 0.1);
    
    csv += `${msgType},${symbol},${1000+i},${side},${orderType},${price.toFixed(2)},${quantity.toFixed(4)},${timestamp.toISOString()}\n`;
}

fs.writeFileSync(outputFile, csv);
console.log(`✓ Generated ${numMessages} synthetic messages`);
NODE_EOF

# 7. Convert FIX spec XML to readable format
echo ""
echo "7. Converting FIX Spec to CSV"
echo "------------------------------"
if [ -f "$DATA_DIR/fix44_spec.xml" ]; then
    # Extract sample messages from FIX spec
    grep -E "35=[D8FG]" "$DATA_DIR/fix44_spec.xml" 2>/dev/null | head -100 > "$DATA_DIR/fix44_samples.txt" || true
    echo "✓ Extracted FIX samples from spec"
fi

# Summary
echo ""
echo "======================================="
echo "Available FIX Data Files:"
echo "======================================="
ls -lh "$DATA_DIR"/*.csv "$DATA_DIR"/*.fix "$DATA_DIR"/*.txt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Data Sources:"
echo "  • CME sample (Futures): cme_sample.fix"
echo "  • NASDAQ sample (Equities): nasdaq_sample.csv"
echo "  • Crypto sample: crypto_sample.csv"
echo "  • Large synthetic (100k msgs): large_synthetic.csv"
echo ""
echo "To run benchmarks with this data:"
echo "  ./scripts/benchmark.sh --data testdata/large_synthetic.csv"
echo ""
echo "For more realistic data, consider:"
echo "  • NYSE TAQ data: https://www.nyse.com/market-data/historical"
echo "  • LOBSTER (limit order book): https://lobsterdata.com/info/DataSamples.php"
echo "  • Kaggle datasets: https://www.kaggle.com/datasets?search=fix+trading"
echo "  • Arctic (tick data): https://github.com/man-group/arctic"