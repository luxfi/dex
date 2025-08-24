#!/bin/bash

# Start DEX backend in test mode
cd "$(dirname "$0")/.."

echo "🚀 Starting LUX DEX Backend in Test Mode"
echo "========================================"

# Build if necessary
if [ ! -f "bin/lx-dex" ]; then
    echo "Building backend..."
    make build
fi

# Start with test flag
./bin/lx-dex -test -http 8080 -ws 8081 -engine hybrid